"""FLAME-38M-architecture-matched MoE language model with causal chunk-pool routing.

One code path serves all conditions:
  * routing mode "token": original FLAME routing (softmax over all E experts, top-k by probability,
    gate = full-softmax probability, no renormalization, dropless, k routed + shared experts).
  * routing mode "pool": for a chunk whose first `warmup` tokens are fully observed, the per-layer
    candidate pool P_c = TopM(W_r mean(h_prefix)); every later token of the chunk masks logits outside
    P_c, applies the original softmax normalization inside the pool, then the original top-k.
    Prefix tokens (chunk_pos < warmup) and chunks of length <= warmup use token routing.

Batch tensors (all (B, T) int64):
  input_ids, doc_ids (index of the document inside the packed row, -1 = pad), positions (position
  inside the document, used for RoPE), chunk_ids (chunk index inside the row, -1 = pad), chunk_pos
  (position inside the chunk). Attention is causal and block-diagonal over doc_ids.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 50304
    n_layer: int = 9
    d_model: int = 256
    n_head: int = 16
    dense_ffn: int = 1368
    moe_layers: tuple = (1, 2, 3, 4, 5, 6, 7, 8)
    n_experts: int = 64
    top_k: int = 6
    expert_ffn: int = 176
    shared_ffn: int = 352
    rope_base: float = 10000.0
    norm_eps: float = 1e-6
    max_seq_len: int = 1024
    init_std: float = 0.02
    # Tie the LM-head weight to the input embedding (one shared Parameter). Default False: configurations that do
    # not set it (all pre-csr-representative-v1 configs) are unchanged, and to_dict() omits the field for them.
    tie_embeddings: bool = False

    def to_dict(self):
        d = asdict(self)
        d["moe_layers"] = list(self.moe_layers)
        if not self.tie_embeddings:
            d.pop("tie_embeddings")
        return d

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        d["moe_layers"] = tuple(d.get("moe_layers", cls.moe_layers))
        return cls(**d)


@dataclass
class RoutingConfig:
    mode: str = "token"  # "token" | "pool"
    pool_size: int = 12  # m = min(E, 2k)
    warmup: int = 16  # w
    # Gate values for pool-constrained tokens. "global" (default): gate = full-softmax probability of the
    # selected expert, exactly as FLAME (only the eligible set changes). "pool": softmax renormalized inside the
    # pool (EMO-style); changes gate scale relative to token routing, so it is NOT the default.
    gate_norm: str = "global"

    def validate(self, mcfg: ModelConfig):
        assert self.mode in ("token", "pool"), self.mode
        assert self.gate_norm in ("global", "pool"), self.gate_norm
        assert mcfg.top_k <= self.pool_size <= mcfg.n_experts, (self.pool_size, mcfg.top_k)
        assert self.warmup >= 1


class RMSNorm(nn.Module):
    def __init__(self, d, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        xf = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * self.weight.to(y.dtype)).to(x.dtype)


def rope_cos_sin(positions, head_dim, base, dtype):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=positions.device, dtype=torch.float64) / head_dim))
    ang = positions.to(torch.float64).unsqueeze(-1) * inv  # (..., T, hd/2)
    ang = torch.cat([ang, ang], dim=-1)
    return ang.cos().to(dtype), ang.sin().to(dtype)


def apply_rope(x, cos, sin):  # x (B, H, T, hd); cos/sin (B, 1, T, hd)
    h = x.shape[-1] // 2
    rot = torch.cat([-x[..., h:], x[..., :h]], dim=-1)
    return x * cos + rot * sin


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.hd = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, attn_mask, kv_cache=None):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_head, self.hd).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if kv_cache is not None:  # incremental: append and attend to all cached (single document)
            if kv_cache.get("k") is not None:
                k = torch.cat([kv_cache["k"], k], dim=2)
                v = torch.cat([kv_cache["v"], v], dim=2)
            kv_cache["k"], kv_cache["v"] = k, v
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False)
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.proj(y.transpose(1, 2).reshape(B, T, D))


class SwiGLU(nn.Module):
    def __init__(self, d, f):
        super().__init__()
        self.w1 = nn.Linear(d, f, bias=False)
        self.w3 = nn.Linear(d, f, bias=False)
        self.w2 = nn.Linear(f, d, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def _router_dtype(x):
    return torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype


class MoE(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        E, d, f = cfg.n_experts, cfg.d_model, cfg.expert_ffn
        self.E, self.k = E, cfg.top_k
        self.router = nn.Linear(d, E, bias=False)
        self.w1 = nn.Parameter(torch.empty(E, d, f))
        self.w3 = nn.Parameter(torch.empty(E, d, f))
        self.w2 = nn.Parameter(torch.empty(E, f, d))
        self.shared = SwiGLU(d, cfg.shared_ffn) if cfg.shared_ffn > 0 else None

    def router_logits(self, x):
        rd = _router_dtype(x)
        with torch.autocast(device_type=x.device.type, enabled=False):  # FLAME: fp32 router
            return F.linear(x.to(rd), self.router.weight.to(rd))

    def select(self, logits, cand_mask, gate_norm="global"):
        """logits (N, E); cand_mask (N, E) bool or None. Returns topi, gates (N, k), probs_full (N, E).
        Unconstrained rows (cand all True or None) reduce exactly to FLAME routing under both gate_norm values."""
        probs_full = torch.softmax(logits, dim=-1)
        if cand_mask is None:
            gates, topi = probs_full.topk(self.k, dim=-1)
        elif gate_norm == "global":  # same top-k order as FLAME, restricted to the pool; gates = global probs
            _, topi = probs_full.masked_fill(~cand_mask, -1.0).topk(self.k, dim=-1)
            gates = probs_full.gather(1, topi)
        else:  # "pool": softmax renormalized inside the pool, then top-k
            gates, topi = torch.softmax(logits.masked_fill(~cand_mask, float("-inf")), dim=-1).topk(self.k, dim=-1)
        return topi, gates, probs_full

    def experts(self, x, topi, gates):
        """Dropless dispatch: padded per-expert batches + bmm. x (N, d)."""
        N, d = x.shape
        k = topi.shape[1]
        flat_e = topi.reshape(-1)
        order = torch.argsort(flat_e, stable=True)
        counts = torch.bincount(flat_e, minlength=self.E)
        maxc = int(counts.max().item()) if N > 0 else 0
        if maxc == 0:
            return x.new_zeros(N, d)
        starts = torch.cumsum(counts, 0) - counts
        sorted_e = flat_e[order]
        slot = torch.arange(N * k, device=x.device) - starts[sorted_e]
        dest_sorted = sorted_e * maxc + slot
        dest = torch.empty_like(dest_sorted)
        dest[order] = dest_sorted
        tok = torch.arange(N, device=x.device).repeat_interleave(k)
        buf = x.new_zeros(self.E * maxc, d).index_copy(0, dest, x[tok]).view(self.E, maxc, d)
        h = F.silu(torch.bmm(buf, self.w1.to(x.dtype))) * torch.bmm(buf, self.w3.to(x.dtype))
        out = torch.bmm(h, self.w2.to(x.dtype)).view(self.E * maxc, d)
        y = out[dest].view(N, k, d)
        return (y * gates.to(y.dtype).unsqueeze(-1)).sum(1)

    def forward(self, x, cand_mask=None, gate_norm="global"):
        """x (N, d). Returns y (N, d), dict(topi, gates, probs_full, logits)."""
        logits = self.router_logits(x)
        topi, gates, probs_full = self.select(logits, cand_mask, gate_norm)
        y = self.experts(x, topi, gates)
        if self.shared is not None:
            y = y + self.shared(x)
        return y, {"topi": topi, "gates": gates, "probs_full": probs_full, "logits": logits}


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, is_moe: bool):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.is_moe = is_moe
        self.ffn = MoE(cfg) if is_moe else SwiGLU(cfg.d_model, cfg.dense_ffn)


@dataclass
class ChunkLayout:
    """Chunks with a complete warmup prefix. Row r = (b, chunk start t0); prefix_idx (R, w) positions."""
    rows_b: torch.Tensor
    rows_t0: torch.Tensor
    rows_chunk: torch.Tensor
    prefix_idx: torch.Tensor
    token_row: torch.Tensor  # (B, T) row index for post-warmup tokens, -1 elsewhere
    eligible: torch.Tensor  # (B, T) bool: token is pool-eligible (chunk_pos >= w)


def build_chunk_layout(chunk_ids, chunk_pos, warmup):
    B, T = chunk_ids.shape
    dev = chunk_ids.device
    is_start = (chunk_pos == 0) & (chunk_ids >= 0)
    sb, st = is_start.nonzero(as_tuple=True)
    end = st + warmup - 1
    ok = end < T
    sb, st, end = sb[ok], st[ok], end[ok]
    same = (chunk_ids[sb, end] == chunk_ids[sb, st]) & (chunk_pos[sb, end] == warmup - 1)
    sb, st = sb[same], st[same]
    cid = chunk_ids[sb, st]
    prefix_idx = st.unsqueeze(1) + torch.arange(warmup, device=dev).unsqueeze(0)
    C = int(chunk_ids.max().item()) + 1 if chunk_ids.numel() and chunk_ids.max() >= 0 else 1
    table = torch.full((B, C), -1, dtype=torch.long, device=dev)
    table[sb, cid] = torch.arange(sb.numel(), device=dev)
    eligible = (chunk_pos >= warmup) & (chunk_ids >= 0)
    token_row = torch.where(eligible, table.gather(1, chunk_ids.clamp(min=0)), torch.full_like(chunk_ids, -1))
    eligible = eligible & (token_row >= 0)
    return ChunkLayout(sb, st, cid, prefix_idx, token_row, eligible)


@dataclass
class ForwardOut:
    logits: torch.Tensor
    aux_loss: torch.Tensor  # sum over MoE layers of the Switch load-balance loss (uncoefficiented)
    z_loss: torch.Tensor  # sum over MoE layers of mean logsumexp^2 (uncoefficiented)
    routing: list = field(default_factory=list)  # per MoE layer dict
    layout: Optional[ChunkLayout] = None


class CSRModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg, i in cfg.moe_layers) for i in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight  # same Parameter object: shared storage, one optimizer entry
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        std = self.cfg.init_std
        out_std = std / math.sqrt(2.0 * self.cfg.n_layer)
        for name, p in self.named_parameters():
            if p.dim() == 1:
                p.fill_(1.0)  # RMSNorm weights
            elif name.endswith(("attn.proj.weight", "ffn.w2.weight", "shared.w2.weight")) or name.endswith("ffn.w2"):
                p.normal_(0.0, out_std)
            else:
                p.normal_(0.0, std)

    def moe_blocks(self):
        return [b for b in self.blocks if b.is_moe]

    @staticmethod
    def attention_mask(doc_ids, positions):
        T = doc_ids.shape[1]
        same = doc_ids.unsqueeze(2) == doc_ids.unsqueeze(1)
        causal = torch.ones(T, T, dtype=torch.bool, device=doc_ids.device).tril()
        m = same & causal & (doc_ids >= 0).unsqueeze(2)
        pad_self = (doc_ids < 0).unsqueeze(2) & torch.eye(T, dtype=torch.bool, device=doc_ids.device)
        return (m | pad_self).unsqueeze(1)

    def forward(self, batch, routing: RoutingConfig, pool_override=None, constrain_rows=None,
                need_chunk_logits=False):
        """pool_override: optional list (one per MoE layer) of (R, E) bool pool masks replacing the
        model's own causal pools. constrain_rows: optional (R,) bool, which chunk rows are constrained
        (default: all rows in pool mode, none in token mode)."""
        ids, doc_ids, positions = batch["input_ids"], batch["doc_ids"], batch["positions"]
        B, T = ids.shape
        x = self.tok_emb(ids)
        cos, sin = rope_cos_sin(positions, self.cfg.d_model // self.cfg.n_head, self.cfg.rope_base, x.dtype)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        mask = self.attention_mask(doc_ids, positions)
        nonpad = doc_ids >= 0
        layout = None
        use_layout = routing.mode == "pool" or pool_override is not None or need_chunk_logits
        if use_layout:
            layout = build_chunk_layout(batch["chunk_ids"], batch["chunk_pos"], routing.warmup)
            if constrain_rows is None:
                constrain_rows = torch.full((layout.rows_b.numel(),), routing.mode == "pool",
                                            dtype=torch.bool, device=ids.device)
        acc = torch.float64 if x.dtype == torch.float64 else torch.float32
        aux_total = x.new_zeros((), dtype=acc)
        z_total = x.new_zeros((), dtype=acc)
        records = []
        moe_i = 0
        for blk in self.blocks:
            x = x + blk.attn(blk.norm1(x), cos, sin, mask)
            h = blk.norm2(x)
            if not blk.is_moe:
                x = x + blk.ffn(h)
                continue
            moe = blk.ffn
            cand = None
            constrained = torch.zeros(B, T, dtype=torch.bool, device=ids.device)
            chunk_logits = None
            if layout is not None:
                prefix_h = h[layout.rows_b.unsqueeze(1), layout.prefix_idx]  # (R, w, d)
                hbar = prefix_h.mean(dim=1)
                chunk_logits = moe.router_logits(hbar)  # (R, E), full and unmasked
                if pool_override is not None:
                    pool_mask = pool_override[moe_i]
                else:
                    pidx = chunk_logits.detach().topk(routing.pool_size, dim=-1).indices
                    pool_mask = torch.zeros_like(chunk_logits, dtype=torch.bool).scatter_(1, pidx, True)
                row_ok = constrain_rows[layout.token_row.clamp(min=0)] & layout.eligible
                constrained = row_ok & nonpad
                cand = torch.ones(B, T, moe.E, dtype=torch.bool, device=ids.device)
                cand[constrained] = pool_mask[layout.token_row[constrained]]
                cand = cand.view(B * T, moe.E)
            y, info = moe(h.reshape(B * T, -1), cand, routing.gate_norm)
            x = x + y.view(B, T, -1)
            # Load-balance and z-loss over non-pad tokens; P from full unmasked probabilities,
            # f from the experts actually selected (Megatron switch_load_balancing_loss form).
            npm = nonpad.reshape(-1)
            n = npm.sum().clamp(min=1)
            probs = info["probs_full"][npm]
            counts = torch.bincount(info["topi"][npm].reshape(-1), minlength=moe.E).to(probs.dtype)
            P = probs.sum(0) / n
            f = counts / (n * moe.k)
            aux_total = aux_total + (moe.E * (f * P).sum()).to(acc)
            z_total = z_total + (torch.logsumexp(info["logits"][npm], dim=-1).pow(2).sum() / n).to(acc)
            records.append({
                "topi": info["topi"].view(B, T, -1),
                "gates": info["gates"].view(B, T, -1),
                "constrained": constrained,
                "chunk_logits": chunk_logits,
                "probs_full": info["probs_full"].view(B, T, -1),
            })
            moe_i += 1
        logits = self.lm_head(self.norm_f(x))
        return ForwardOut(logits, aux_total, z_total, records, layout)

    # ---------------- incremental (single document, batch 1) ----------------
    def new_state(self):
        return {"kv": [dict() for _ in self.blocks], "moe": [dict() for _ in self.moe_blocks()]}

    @torch.no_grad()
    def step(self, token_id: int, position: int, chunk_pos: int, routing: RoutingConfig, state):
        """Process one token causally. Pools are cached only once the warmup prefix is observed.
        Returns logits (V,), list of per-layer (topi, constrained)."""
        dev = self.tok_emb.weight.device
        x = self.tok_emb(torch.tensor([[token_id]], device=dev))
        cos, sin = rope_cos_sin(torch.tensor([[position]], device=dev), self.cfg.d_model // self.cfg.n_head,
                                self.cfg.rope_base, x.dtype)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        out_routing = []
        moe_i = 0
        for bi, blk in enumerate(self.blocks):
            x = x + blk.attn(blk.norm1(x), cos, sin, None, kv_cache=state["kv"][bi])
            h = blk.norm2(x)
            if not blk.is_moe:
                x = x + blk.ffn(h)
                continue
            moe, st = blk.ffn, state["moe"][moe_i]
            hv = h.view(1, -1)
            if chunk_pos == 0:
                st.clear()
                st["prefix"] = []
            cand, constrained = None, False
            if chunk_pos < routing.warmup:
                st.setdefault("prefix", []).append(hv[0])
            elif routing.mode == "pool":
                if "pool" not in st:
                    assert len(st.get("prefix", [])) == routing.warmup, "pool requested before prefix observed"
                    hbar = torch.stack(st["prefix"], 0).unsqueeze(0).mean(dim=1)
                    pidx = moe.router_logits(hbar).topk(routing.pool_size, dim=-1).indices
                    st["pool"] = torch.zeros(1, moe.E, dtype=torch.bool, device=dev).scatter_(1, pidx, True)
                cand, constrained = st["pool"], True
            y, info = moe(hv, cand, routing.gate_norm)
            x = x + y.view(1, 1, -1)
            out_routing.append((info["topi"][0], constrained))
            moe_i += 1
        return self.lm_head(self.norm_f(x))[0, 0], out_routing


def trainable_parameter_names(model: nn.Module):
    return sorted(n for n, p in model.named_parameters() if p.requires_grad)


def count_parameters(model: CSRModel):
    """Unique parameters (a tied embedding / LM-head matrix is counted once)."""
    cfg = model.cfg
    total = sum(p.numel() for p in model.parameters())
    per_expert = 3 * cfg.d_model * cfg.expert_ffn
    n_moe = len(cfg.moe_layers)
    inactive = n_moe * (cfg.n_experts - cfg.top_k) * per_expert
    return {"total": total, "active_per_token": total - inactive}
