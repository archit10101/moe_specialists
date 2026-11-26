"""csr-domain-forecast-curriculum-v1 models (docs/DOMAIN_FORECAST_CURRICULUM_V1_PREREG.md sections 3-5).

New module; CSRModel is imported, never modified.

DraftModel  : dense causal CSRModel (d128, 4 layers) + one residual bottleneck adapter per domain in every layer
              (x + U_d silu(D_d RMSNorm_d(x)), U init 0). Each token uses its own document's adapter.
draft_forecasts : greedy 4-token forecasts from EVERY prefix in parallel (tree attention; tokens after t invisible).
DFCModel    : the representative NanoMoE (TOKEN-4 backbone, identical init to CSRModel for the same seed) plus the
              forecast encoder / informative gates / domain-group prior / bounded scales, instantiated in every
              condition. router_logits = token_logits + strength * g * (s_f center(W_f Fn) + s_p center(P[dom])).
              With strength == 0 for every row the extra path is skipped: logits == token_logits exactly.
"""
from __future__ import annotations

import hashlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import CSRModel, ForwardOut, ModelConfig, RMSNorm, apply_rope, rope_cos_sin

N_DOMAINS = 5
GAMMA = 4
PROV_PER_DOMAIN = 4
PRIOR_INIT = 2.0
SCALE_MAX = 2.0
RHO_INIT = math.log(1.0 / 3.0)  # SCALE_MAX * sigmoid(RHO_INIT) = 0.5
ADAPTER_DIM = 32
DERANGE = {d: (d + 1) % N_DOMAINS for d in range(N_DOMAINS)}

CONDITIONS = {
    # name: (curriculum, use_forecast, use_prior, forecast_source)
    "TOKEN-MAIN": (False, False, False, None),
    "LABEL": (True, False, True, None),
    "FORECAST": (True, True, False, "correct"),
    "SOFT": (True, True, True, "correct"),
    "WRONG": (True, True, True, "deranged"),
    "TOKEN-ALL-DATA": (False, False, False, None),
    "TOKEN-COMPUTE": (False, False, False, None),
}


def provisional_group(d):
    return list(range(PROV_PER_DOMAIN * d, PROV_PER_DOMAIN * (d + 1)))


def added_seed(init_seed: int) -> int:
    return int.from_bytes(hashlib.sha256(f"dfcv1-added|{init_seed}".encode()).digest()[:4], "little")


# ------------------------------------------------------------------ draft
class Adapter(nn.Module):
    def __init__(self, d, r, eps):
        super().__init__()
        self.norm = RMSNorm(d, eps)
        self.down = nn.Linear(d, r, bias=False)
        self.up = nn.Linear(r, d, bias=False)

    def forward(self, x):
        return self.up(F.silu(self.down(self.norm(x))))


class DraftModel(CSRModel):
    """Dense CSRModel (moe_layers=()) + per-domain adapters after every block."""

    def __init__(self, cfg: ModelConfig, n_domains=N_DOMAINS, adapter_dim=ADAPTER_DIM):
        assert len(cfg.moe_layers) == 0
        super().__init__(cfg)
        self.n_domains = n_domains
        self.adapters = nn.ModuleList([nn.ModuleList([Adapter(cfg.d_model, adapter_dim, cfg.norm_eps)
                                                      for _ in range(n_domains)]) for _ in range(cfg.n_layer)])
        with torch.no_grad():
            for la in self.adapters:
                for a in la:
                    a.norm.weight.fill_(1.0)
                    a.down.weight.normal_(0.0, cfg.init_std)
                    a.up.weight.zero_()

    def apply_adapters(self, li, x, dom):
        """x (B, T, d); dom (B, T) long in [0, n_domains) (pads: any valid id). Per-token adapter selection."""
        # every adapter runs on every token (fixed shapes -> per-token results independent of batch composition);
        # each token keeps only its own domain's output
        out = torch.zeros_like(x)
        for d in range(self.n_domains):
            out = torch.where((dom == d).unsqueeze(-1), self.adapters[li][d](x).to(x.dtype), out)
        return x + out

    def forward(self, batch, adapter_domain=None):
        ids, doc_ids, positions = batch["input_ids"], batch["doc_ids"], batch["positions"]
        dom = (batch["domain"] if adapter_domain is None else adapter_domain).clamp(min=0)
        x = self.tok_emb(ids)
        cos, sin = rope_cos_sin(positions, self.cfg.d_model // self.cfg.n_head, self.cfg.rope_base, x.dtype)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        mask = self.attention_mask(doc_ids, positions)
        for li, blk in enumerate(self.blocks):
            x = x + blk.attn(blk.norm1(x), cos, sin, mask)
            x = x + blk.ffn(blk.norm2(x))
            x = self.apply_adapters(li, x, dom)
        z = x.new_zeros(())
        return ForwardOut(self.lm_head(self.norm_f(x)), z, z, [], None)


def _qkv(blk, x, cos, sin):
    a = blk.attn
    B, T, _ = x.shape
    q, k, v = a.qkv(blk.norm1(x)).view(B, T, 3, a.n_head, a.hd).permute(2, 0, 3, 1, 4)
    return apply_rope(q, cos, sin), apply_rope(k, cos, sin), v


@torch.no_grad()
def draft_forecasts(draft: DraftModel, input_ids, doc_ids, positions, adapter_domain, gamma=GAMMA):
    """out[b, t, j-1] = j-th greedy draft token generated from the prefix x_0..x_t of the document containing t,
    using adapter adapter_domain[b, t]. Real-prefix keys are masked document-causally (<= t); proposal j attends to
    the real prefix <= t and its own proposals 1..j-1 only. Tokens after t never enter the computation for t."""
    B, T = input_ids.shape
    cfg = draft.cfg
    hd = cfg.d_model // cfg.n_head
    dom = adapter_domain.clamp(min=0)
    x = draft.tok_emb(input_ids)
    cos, sin = rope_cos_sin(positions, hd, cfg.rope_base, x.dtype)
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    mask = CSRModel.attention_mask(doc_ids, positions)
    Kr, Vr = [], []
    for li, blk in enumerate(draft.blocks):
        q, k, v = _qkv(blk, x, cos, sin)
        Kr.append(k)
        Vr.append(v)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + blk.attn.proj(y.transpose(1, 2).reshape(B, T, -1))
        x = x + blk.ffn(blk.norm2(x))
        x = draft.apply_adapters(li, x, dom)
    cur = draft.lm_head(draft.norm_f(x)).argmax(-1)
    props = [cur]
    Kp = [[] for _ in draft.blocks]
    Vp = [[] for _ in draft.blocks]
    neg = torch.finfo(torch.float32).min
    scale = 1.0 / math.sqrt(hd)
    for j in range(2, gamma + 1):
        x = draft.tok_emb(cur)
        c2, s2 = rope_cos_sin(positions + (j - 1), hd, cfg.rope_base, x.dtype)
        c2, s2 = c2.unsqueeze(1), s2.unsqueeze(1)
        for li, blk in enumerate(draft.blocks):
            q, k, v = _qkv(blk, x, c2, s2)
            Kp[li].append(k)
            Vp[li].append(v)
            sr = torch.matmul(q, Kr[li].transpose(-1, -2)).float() * scale
            sr = sr.masked_fill(~mask, neg)
            kp = torch.stack(Kp[li], -2)
            vp = torch.stack(Vp[li], -2)
            sp = (q.unsqueeze(-2) * kp).sum(-1).float() * scale
            w = torch.softmax(torch.cat([sr, sp], -1), -1)
            wr, wp = w[..., :T], w[..., T:]
            y = torch.matmul(wr.to(Vr[li].dtype), Vr[li]) + (wp.unsqueeze(-1).to(vp.dtype) * vp).sum(-2)
            x = x + blk.attn.proj(y.transpose(1, 2).reshape(B, T, -1))
            x = x + blk.ffn(blk.norm2(x))
            x = draft.apply_adapters(li, x, dom)
        cur = draft.lm_head(draft.norm_f(x)).argmax(-1)
        props.append(cur)
    return torch.stack(props, -1)


def draft_config_dict():
    return {"vocab_size": 50304, "n_layer": 4, "d_model": 128, "n_head": 4, "dense_ffn": 512, "moe_layers": [],
            "n_experts": 1, "top_k": 1, "expert_ffn": 1, "shared_ffn": 0, "rope_base": 10000.0, "norm_eps": 1e-6,
            "max_seq_len": 1024, "init_std": 0.02, "tie_embeddings": True}


def build_draft(seed, device, dtype=torch.float32):
    torch.manual_seed(seed)
    return DraftModel(ModelConfig.from_dict(draft_config_dict())).to(device=device, dtype=dtype)


def load_frozen_draft(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = DraftModel(ModelConfig.from_dict(ck["config"]["model"]))
    m.load_state_dict(ck["model"])
    m = m.to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


# ------------------------------------------------------------------ main model
ADDED_PREFIXES = ("fc_", "rt_")


class DFCModel(CSRModel):
    def __init__(self, cfg: ModelConfig, condition: str, init_seed: int, n_domains=N_DOMAINS):
        super().__init__(cfg)  # backbone first (global RNG): identical to TOKEN-4 for the same seed
        assert condition in CONDITIONS, condition
        self.condition = condition
        self.curriculum, self.use_forecast, self.use_prior, self.forecast_source = CONDITIONS[condition]
        d, E = cfg.d_model, cfg.n_experts
        self.moe_layer_ids = [i for i, b in enumerate(self.blocks) if b.is_moe]
        L = len(self.moe_layer_ids)
        self.n_domains = n_domains
        # forecast encoder (shared)
        self.fc_pos = nn.Parameter(torch.empty(GAMMA, d))
        self.fc_gru = nn.GRU(d, d, num_layers=1, batch_first=True)
        self.fc_proj = nn.Linear(d, d, bias=False)
        self.fc_norm = RMSNorm(d, cfg.norm_eps)
        # per MoE layer routing residuals
        self.rt_wf = nn.Parameter(torch.empty(L, E, d))
        self.rt_gate_w = nn.Parameter(torch.empty(L, 2 * d))
        self.rt_gate_b = nn.Parameter(torch.zeros(L))
        prior = torch.zeros(L, n_domains, E)
        for dd in range(n_domains):
            prior[:, dd, provisional_group(dd)] = PRIOR_INIT
        self.rt_prior = nn.Parameter(prior)
        self.rt_rho_f = nn.Parameter(torch.full((L,), RHO_INIT))
        self.rt_rho_p = nn.Parameter(torch.full((L,), RHO_INIT))
        g = torch.Generator().manual_seed(added_seed(init_seed))
        with torch.no_grad():
            std = cfg.init_std
            self.fc_pos.copy_(torch.randn(self.fc_pos.shape, generator=g) * std)
            k = 1.0 / math.sqrt(d)  # PyTorch default GRU init range, from the dedicated generator
            for _, p in self.fc_gru.named_parameters():
                p.copy_((torch.rand(p.shape, generator=g) * 2 - 1) * k)
            self.fc_proj.weight.copy_(torch.randn(self.fc_proj.weight.shape, generator=g) * std)
            self.fc_norm.weight.fill_(1.0)
            self.rt_wf.copy_(torch.randn(self.rt_wf.shape, generator=g) * std)
            self.rt_gate_w.copy_(torch.randn(self.rt_gate_w.shape, generator=g) * std)

    # -------------------------------------------------- bookkeeping
    def added_parameter_count(self):
        return sum(p.numel() for n, p in self.named_parameters() if n.startswith(ADDED_PREFIXES))

    def backbone_state(self):
        return {k: v for k, v in self.named_parameters() if not k.startswith(ADDED_PREFIXES)}

    def scales(self):
        return (SCALE_MAX * torch.sigmoid(self.rt_rho_f), SCALE_MAX * torch.sigmoid(self.rt_rho_p))

    def forecast_repr(self, fc_ids):
        """fc_ids (B, T, 4) long -> Fn (B, T, d): stop-grad tied embeddings + learned relative positions -> GRU ->
        projection -> RMSNorm."""
        B, T, G = fc_ids.shape
        e = self.tok_emb(fc_ids).detach() + self.fc_pos.to(self.tok_emb.weight.dtype)
        e = e.reshape(B * T, G, -1)
        with torch.autocast(device_type=e.device.type, enabled=False):
            _, hN = self.fc_gru(e.to(self.fc_gru.weight_ih_l0.dtype))
        return self.fc_norm(self.fc_proj(hN[-1])).reshape(B, T, -1)

    # -------------------------------------------------- forward
    def forward(self, batch, strength=None, collect=False, disable=None):
        """batch: input_ids, doc_ids, positions (B, T); optional domain (B, T) (-1 pads), fc (B, T, 4) forecast ids.
        strength: None -> 0 everywhere; float; or (B,) per-row tensor. Only rows with strength > 0 can differ from
        TOKEN-4. disable: optional list of (layer_id, expert_set(list), (B, T) bool token mask): zero those experts'
        gates on the masked tokens (functional ablation; experts stay eligible)."""
        ids, doc_ids, positions = batch["input_ids"], batch["doc_ids"], batch["positions"]
        B, T = ids.shape
        dev = ids.device
        if strength is None:
            srow = None
        else:
            srow = torch.as_tensor(strength, dtype=torch.float32, device=dev)
            if srow.dim() == 0:
                srow = srow.expand(B)
            if not bool((srow > 0).any()) or not self.curriculum:
                srow = None
        active = srow is not None
        x = self.tok_emb(ids)
        hd = self.cfg.d_model // self.cfg.n_head
        cos, sin = rope_cos_sin(positions, hd, self.cfg.rope_base, x.dtype)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        mask = self.attention_mask(doc_ids, positions)
        nonpad = doc_ids >= 0
        Fn, dom = None, None
        if active:
            if self.use_forecast:
                Fn = self.forecast_repr(batch["fc"]).reshape(B * T, -1)
            if self.use_prior:
                dom = batch["domain"].clamp(min=0).reshape(-1)
                dom_1h = F.one_hot(dom, self.n_domains).float()
            s_tok = srow.reshape(B, 1).expand(B, T).reshape(-1, 1)
            s_f, s_p = self.scales()
        acc = torch.float64 if x.dtype == torch.float64 else torch.float32
        aux_total = x.new_zeros((), dtype=acc)
        z_total = x.new_zeros((), dtype=acc)
        records = []
        mi = 0
        for li, blk in enumerate(self.blocks):
            x = x + blk.attn(blk.norm1(x), cos, sin, mask)
            h = blk.norm2(x)
            if not blk.is_moe:
                x = x + blk.ffn(h)
                continue
            moe = blk.ffn
            hf = h.reshape(B * T, -1)
            tok_logits = moe.router_logits(hf)
            logits = tok_logits
            gate_val = None
            if active:
                with torch.autocast(device_type=dev.type, enabled=False):
                    hf32 = hf.float()
                    res = torch.zeros_like(tok_logits)
                    if Fn is not None:
                        Fn32 = Fn.float()
                        fr = F.linear(Fn32, self.rt_wf[mi].float())
                        res = res + s_f[mi] * (fr - fr.mean(-1, keepdim=True))
                        gin = torch.cat([hf32, Fn32], -1)
                    else:
                        gin = torch.cat([hf32, torch.zeros_like(hf32)], -1)
                    if dom is not None:
                        pr = dom_1h @ self.rt_prior[mi].float()  # one-hot matmul: deterministic backward
                        res = res + s_p[mi] * (pr - pr.mean(-1, keepdim=True))
                    gate_val = torch.sigmoid(gin @ self.rt_gate_w[mi].float() + self.rt_gate_b[mi].float())
                    logits = tok_logits + s_tok * gate_val.unsqueeze(-1) * res
            topi, gates, probs_full = moe.select(logits, None)
            if disable is not None:
                for (dl, eset, m_dis) in disable:
                    if dl == li:
                        kill = torch.zeros_like(topi, dtype=torch.bool)
                        for e in eset:
                            kill |= topi == e
                        kill &= m_dis.reshape(-1, 1)
                        gates = gates.masked_fill(kill, 0.0)
            y = moe.experts(hf, topi, gates) + moe.shared(hf)
            x = x + y.view(B, T, -1)
            npm = nonpad.reshape(-1)
            n = npm.sum().clamp(min=1)
            pr_ = probs_full[npm]
            counts = torch.bincount(topi[npm].reshape(-1), minlength=moe.E).to(pr_.dtype)
            aux_total = aux_total + (moe.E * ((counts / (n * moe.k)) * (pr_.sum(0) / n)).sum()).to(acc)
            z_total = z_total + (torch.logsumexp(logits[npm], dim=-1).pow(2).sum() / n).to(acc)
            rec = {"topi": topi.view(B, T, -1), "gates": gates.view(B, T, -1)}
            if collect:
                rec["probs_full"] = probs_full.view(B, T, -1)
                if gate_val is not None:
                    rec["gate"] = gate_val.view(B, T)
                    rec["topi_token"] = tok_logits.topk(moe.k, dim=-1).indices.view(B, T, -1)
            records.append(rec)
            mi += 1
        return ForwardOut(self.lm_head(self.norm_f(x)), aux_total, z_total, records, None)


def build_dfc_model(model_cfg: dict, condition: str, seed: int, device, dtype=torch.float32):
    torch.manual_seed(seed)
    m = DFCModel(ModelConfig.from_dict(model_cfg), condition, seed)
    return m.to(device=device, dtype=dtype)


def to_plain_csr(m: DFCModel):
    """Standalone TOKEN-4 export: the backbone state loaded into a plain CSRModel (no forecast/domain modules)."""
    plain = CSRModel(m.cfg)
    sd = {k: v for k, v in m.state_dict().items() if not k.startswith(ADDED_PREFIXES)}
    plain.load_state_dict(sd, strict=True)
    return plain.to(next(m.parameters()).device)


MAIN_MODEL_CFG = {"vocab_size": 50304, "n_layer": 8, "d_model": 192, "n_head": 8, "dense_ffn": 768,
                  "moe_layers": [1, 2, 3, 4, 5, 6, 7], "n_experts": 32, "top_k": 4, "expert_ffn": 128,
                  "shared_ffn": 256, "rope_base": 10000.0, "norm_eps": 1e-6, "max_seq_len": 1024, "init_std": 0.02,
                  "tie_embeddings": True}
