"""csr-value-routed-compute-v1: TOKEN-4 NanoMoE with block-aligned local/global attention and per-block routed-expert
count k in {1, 2, 4} via fixed four-slot dispatch (docs/VALUE_ROUTED_COMPUTE_V1_PROTOCOL.md).

New module; CSRModel / cla_attn are imported, never modified. VRCModel subclasses CSRModel, so construction and
initialisation are exactly TOKEN-4's for the same seed (no extra parameters).

Policy (per forward):
  ref=True               : FULL-TOKEN4 - full document-causal attention in every layer, k = 4 (training path).
  ref=False, gsel, kblk  : local attention (window W=512) in every layer; layers in GLOBAL_LAYERS (3, 7) use full
                           document-causal attention for query blocks with gsel=True. kblk (B, T/128) in {1,2,4}.
Fixed four-slot dispatch: top-4 over all 32 experts is always computed (every expert stays eligible); slots >= k are
NULL: gate 0, mapped to a trash row, never part of an expert GEMM. Gates are unrenormalised global-softmax probs.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .cla_attn import BLOCK, AttnContext, _block_mask, block_lists, dense_mask, flex_attend
from .model import CSRModel, ForwardOut, ModelConfig, apply_rope, rope_cos_sin

EXPERIMENT = "csr-value-routed-compute-v1"
WINDOW = 512
GLOBAL_LAYERS = (3, 7)
KS = (1, 2, 4)


def experts_null(moe, x, topi, gates, valid):
    """Dropless padded-bmm dispatch of only the valid (non-NULL) slots. x (N, d); topi, gates, valid (N, 4)."""
    N, d = x.shape
    k = topi.shape[1]
    E = moe.E
    flat_e = torch.where(valid, topi, torch.full_like(topi, E)).reshape(-1)
    order = torch.argsort(flat_e, stable=True)
    counts_all = torch.bincount(flat_e, minlength=E + 1)
    counts = counts_all[:E]
    maxc = int(counts.max().item()) if N > 0 else 0
    if maxc == 0:
        return x.new_zeros(N, d), 0
    starts = torch.cumsum(counts_all, 0) - counts_all
    sorted_e = flat_e[order]
    slot = torch.arange(N * k, device=x.device) - starts[sorted_e]
    dest_sorted = torch.where(sorted_e < E, sorted_e * maxc + slot, torch.full_like(slot, E * maxc))
    dest = torch.empty_like(dest_sorted)
    dest[order] = dest_sorted
    tok = torch.arange(N, device=x.device).repeat_interleave(k)
    buf = x.new_zeros(E * maxc + 1, d).index_copy(0, dest, x[tok])
    b = buf[:E * maxc].view(E, maxc, d)
    h = F.silu(torch.bmm(b, moe.w1.to(x.dtype))) * torch.bmm(b, moe.w3.to(x.dtype))
    out = torch.cat([torch.bmm(h, moe.w2.to(x.dtype)).view(E * maxc, d), x.new_zeros(1, d)], 0)
    y = out[dest].view(N, k, d)
    g = (gates * valid.to(gates.dtype)).to(y.dtype)
    return (y * g.unsqueeze(-1)).sum(1), int(counts.sum().item())


def mixed_block_mask(lay, window, gsel_qb):
    """Per query block: full document-causal key blocks if gsel_qb else the local (window) key blocks.
    gsel_qb (N/128,) bool. Exact: the mask_mod applies the same per-block rule."""
    gdoc = lay.gdoc
    W = int(window)
    gq = gsel_qb

    def mm(b, h, q, k):
        return (gdoc[q] == gdoc[k]) & (k <= q) & ((q - k < W) | gq[q // BLOCK])
    qpos = torch.arange(lay.N, device=gdoc.device)
    pl, fl = block_lists(qpos, lay, W, lay.N)
    pf, ff = block_lists(qpos, lay, None, lay.N)
    sel = gq[:, None]
    return _block_mask(torch.where(sel, pf, pl), torch.where(sel, ff, fl), mm)


class VRCContext(AttnContext):
    def mixed(self, q, k, v, window, gsel):
        """gsel (B, T/128) bool -> (B, T, H*hd)."""
        B, H, T, hd = q.shape
        if self.backend == "dense":
            loc = dense_mask(self.doc_ids, window)
            full = dense_mask(self.doc_ids, None)
            g = gsel.repeat_interleave(BLOCK, dim=1)[:, None, :, None]
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=torch.where(g, full, loc))
            return y.transpose(1, 2).reshape(B, T, H * hd)
        key = ("mixed", window)
        if key not in self._masks:
            self._masks[key] = mixed_block_mask(self.lay, window, gsel.reshape(-1))
        y = flex_attend(self._flat(q), self._flat(k), self._flat(v), self._masks[key], hd)
        return y.view(H, B, T, hd).permute(1, 2, 0, 3).reshape(B, T, H * hd)


class VRCModel(CSRModel):
    def __init__(self, cfg: ModelConfig, backend="dense"):
        super().__init__(cfg)
        self.backend = backend
        self.hd = cfg.d_model // cfg.n_head

    def forward(self, batch, ref=True, gsel=None, kblk=None, collect=False, return_kv=False):
        ids, doc_ids, positions = batch["input_ids"], batch["doc_ids"], batch["positions"]
        B, T = ids.shape
        assert T % BLOCK == 0
        x = self.tok_emb(ids)
        cos, sin = rope_cos_sin(positions, self.hd, self.cfg.rope_base, x.dtype)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        ctx = VRCContext(doc_ids, self.backend)
        nonpad = doc_ids >= 0
        acc = torch.float64 if x.dtype == torch.float64 else torch.float32
        aux_total = x.new_zeros((), dtype=acc)
        z_total = x.new_zeros((), dtype=acc)
        records, kvs = [], []
        n_eval = 0
        if not ref:
            assert gsel is not None and kblk is not None and gsel.shape == (B, T // BLOCK) == kblk.shape
            kt = kblk.repeat_interleave(BLOCK, dim=1).reshape(-1)
            valid4 = torch.arange(4, device=ids.device)[None, :] < kt[:, None]
        for li, blk in enumerate(self.blocks):
            h1 = blk.norm1(x)
            q, k, v = blk.attn.qkv(h1).view(B, T, 3, self.cfg.n_head, self.hd).permute(2, 0, 3, 1, 4)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
            if return_kv:
                kvs.append((k, v))
            if ref:
                y = ctx.attend(q, k, v, None)
            elif li in GLOBAL_LAYERS:
                y = ctx.mixed(q, k, v, WINDOW, gsel)
            else:
                y = ctx.attend(q, k, v, WINDOW)
            x = x + blk.attn.proj(y)
            h = blk.norm2(x)
            if not blk.is_moe:
                x = x + blk.ffn(h)
                continue
            moe = blk.ffn
            hf = h.reshape(B * T, -1)
            logits = moe.router_logits(hf)
            topi, gates, probs_full = moe.select(logits, None)
            if ref:
                y2 = moe.experts(hf, topi, gates)
                n_eval += int(nonpad.sum()) * moe.k
            else:
                y2, ne = experts_null(moe, hf, topi, gates, valid4)
                n_eval += int((valid4 & nonpad.reshape(-1, 1)).sum())
            y2 = y2 + moe.shared(hf)
            x = x + y2.view(B, T, -1)
            npm = nonpad.reshape(-1)
            n = npm.sum().clamp(min=1)
            pr = probs_full[npm]
            counts = torch.bincount(topi[npm].reshape(-1), minlength=moe.E).to(pr.dtype)
            aux_total = aux_total + (moe.E * ((counts / (n * moe.k)) * (pr.sum(0) / n)).sum()).to(acc)
            z_total = z_total + (torch.logsumexp(logits[npm], dim=-1).pow(2).sum() / n).to(acc)
            records.append({"topi": topi.view(B, T, -1), "gates": gates.view(B, T, -1),
                            "probs_full": probs_full.view(B, T, -1) if collect else None,
                            "h_mean": None})
        out = ForwardOut(self.lm_head(self.norm_f(x)), aux_total, z_total, records, None)
        out.n_expert_evals = n_eval
        out.kv = kvs if return_kv else None
        return out


def build_vrc_model(cfg, device, backend=None, dtype=torch.float32):
    torch.manual_seed(cfg["init"]["seed"])
    be = backend or ("flex" if torch.device(device).type == "cuda" else "dense")
    return VRCModel(ModelConfig.from_dict(cfg["model"]), be).to(device=device, dtype=dtype)
