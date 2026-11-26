"""csr-latent-forecast-router-distill-v1 (lfr) models. New module; nothing existing is modified.

Backbone (every condition, identical init within a seed): the fspv2 inference backbone without its topic module:
12 layers, d=384, 6 heads (RoPE), layer 0 dense GELU 1792, layers 1..11 MoE with 32 routed GELU experts (FFN 256) +
one always-active shared GELU expert (768), tied embeddings V=50304. Routing interface everywhere: unrestricted top-4
over all 32 experts, p = softmax(executed logits), gates = p of the selected experts (FLAME).

Routers
  linear  : l = W h                                                   (TOKEN-4, EDL, OUTPUT-DISTILL, TOKEN-COMPUTE)
  student : l_S = W h + W_m RMSNorm(m_t), m_t = causal within-document EMA of P h_s (s <= t)   (inference router)
Teacher (training only, prefix "tc_"; deleted at the start of P3 and for inference):
  z_t = Agg(cached causal draft-forecast features); per MoE layer (u, s) = Head_l(GELU(Trunk[h, z + e_l, ent, agree,
  disagree])); l_T = (u - kappa exp(s/2)) / tau_T; executed logits = lam l_T + (1 - lam) l_S.
Auxiliary routing losses: Switch balance + z-loss (+ EDL) on the executed distribution with the standard gradient
flow; teacher->student KL and the latent-consistency loss use stop-gradient hidden states (router weights only).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import RMSNorm, rope_cos_sin
from .fsp_model import Attention, GELUMLP, edl_loss

TEACHER_PREFIX = "tc_"
N_DRAFT = 6
N_FEAT_DRAFT = 11
N_FEAT_AGG = 8


@dataclass
class LFRConfig:
    vocab_size: int = 50304
    n_layer: int = 12
    d_model: int = 384
    n_head: int = 6
    dense_ffn: int = 1792
    n_experts: int = 32
    top_k: int = 4
    expert_ffn: int = 256
    shared_ffn: int = 768
    rope_base: float = 10000.0
    norm_eps: float = 1e-6
    init_std: float = 0.02
    router: str = "linear"  # linear | student
    mem: str = "ema"  # ema | ema3
    mem_halflife: float = 128.0
    mem_dim: int = 64
    teacher: bool = False
    agg: str = "deepsets"  # deepsets | pool | xattn
    fc_dim: int = 64
    horizons: list = field(default_factory=lambda: [1, 4, 8])
    tc_hidden: int = 128
    inject: str = "all"  # all | late (MoE index >= 4)
    tau_t: float = 1.0
    kappa: float = 0.0

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


# ------------------------------------------------------------------ causal within-document EMA (chunked scan)
def causal_ema(v, doc_ids, positions, halflife, C=64):
    """m_t = sum_{s<=t, same doc} r^(t-s) v_s / sum_{k=0}^{pos_t} r^k, r = 0.5^(1/halflife). v (B,T,D) float.
    Exact (= dense causal EMA), O(T*C*D). Pads (doc_id < 0) give 0. positions = position within the document."""
    B, T, D = v.shape
    if T % C:
        C = T
    n = T // C
    r = 0.5 ** (1.0 / halflife)
    dev = v.device
    vc = v.reshape(B, n, C, D)
    dc = doc_ids.reshape(B, n, C)
    i = torch.arange(C, device=dev)
    lag = (i.view(C, 1) - i.view(1, C)).to(v.dtype)
    base = torch.where(lag >= 0, torch.pow(torch.tensor(r, device=dev, dtype=v.dtype), lag.clamp(min=0)),
                       torch.zeros((), device=dev, dtype=v.dtype))
    same = (dc.unsqueeze(-1) == dc.unsqueeze(-2)) & (dc >= 0).unsqueeze(-1)
    S_in = torch.matmul(base * same.to(v.dtype), vc)  # (B,n,C,D)
    dec = torch.pow(torch.tensor(r, device=dev, dtype=v.dtype), (i + 1).to(v.dtype))  # (C,)
    carry = torch.zeros(B, D, device=dev, dtype=v.dtype)
    cdoc = torch.full((B,), -2, device=dev, dtype=doc_ids.dtype)
    outs = []
    for c in range(n):
        cont = ((dc[:, c] == cdoc.view(B, 1)) & (dc[:, c] >= 0)).to(v.dtype)  # (B,C)
        S = S_in[:, c] + cont.unsqueeze(-1) * dec.view(1, C, 1) * carry.unsqueeze(1)
        outs.append(S)
        carry, cdoc = S[:, -1], dc[:, c, -1]
    S = torch.stack(outs, 1).reshape(B, T, D)
    Z = (1.0 - torch.pow(torch.tensor(r, device=dev, dtype=v.dtype), positions.to(v.dtype) + 1.0)) / (1.0 - r)
    return S / Z.unsqueeze(-1) * (doc_ids >= 0).unsqueeze(-1).to(v.dtype)


# ------------------------------------------------------------------ experts
class Experts(nn.Module):
    def __init__(self, E, d, f):
        super().__init__()
        self.E = E
        self.w1 = nn.Parameter(torch.empty(E, d, f))
        self.w2 = nn.Parameter(torch.empty(E, f, d))

    @torch.compiler.disable
    def forward(self, x, topi, gates):
        """Dropless sorted dispatch, one matmul pair per non-empty expert. x (N, d); topi/gates (N, k)."""
        N, d = x.shape
        k = topi.shape[1]
        flat = topi.reshape(-1)
        order = torch.argsort(flat, stable=True)
        counts = torch.bincount(flat, minlength=self.E).tolist()
        tok = order // k
        xs = x.index_select(0, tok)
        w1s, w2s = self.w1.to(x.dtype).unbind(0), self.w2.to(x.dtype).unbind(0)
        parts = torch.split(xs, counts)
        outs = [F.gelu(parts[e] @ w1s[e], approximate="tanh") @ w2s[e] for e, c in enumerate(counts) if c]
        ys = torch.cat(outs, 0) * gates.reshape(-1)[order].to(x.dtype).unsqueeze(-1)
        out = torch.empty_like(ys)
        out[order] = ys
        return out.view(N, k, d).sum(1)

    @torch.no_grad()
    def all_dot(self, x, g):
        """<E_e(x_n), g_n> for every expert e and row n (fp32). x, g (n, d) -> (n, E)."""
        dt = torch.float64 if x.dtype == torch.float64 else torch.float32
        a = F.gelu(torch.einsum("nd,edf->enf", x.to(dt), self.w1.to(dt)), approximate="tanh")
        b = torch.einsum("efd,nd->enf", self.w2.to(dt), g.to(dt))
        return (a * b).sum(-1).t()

    @torch.no_grad()
    def one(self, x, e):
        dt = torch.float64 if x.dtype == torch.float64 else torch.float32
        return F.gelu(x.to(dt) @ self.w1[e].to(dt), approximate="tanh") @ self.w2[e].to(dt)


class MoELayer(nn.Module):
    def __init__(self, cfg: LFRConfig):
        super().__init__()
        d, E = cfg.d_model, cfg.n_experts
        self.router = nn.Linear(d, E, bias=False)
        self.experts = Experts(E, d, cfg.expert_ffn)
        self.shared = GELUMLP(d, cfg.shared_ffn)


class Block(nn.Module):
    def __init__(self, cfg, is_moe):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.is_moe = is_moe
        self.ffn = MoELayer(cfg) if is_moe else GELUMLP(cfg.d_model, cfg.dense_ffn)


def mem_halflives(cfg):
    return [cfg.mem_halflife] if cfg.mem == "ema" else [16.0, 128.0, 1024.0]


def mem_width(cfg):
    return cfg.mem_dim if cfg.mem == "ema" else 3 * (cfg.mem_dim // 3)


class StudentMemory(nn.Module):
    """Per-MoE-layer student memory (inference parameters): P (d -> M), RMSNorm(M), W_m (M -> E, zero init)."""

    def __init__(self, cfg):
        super().__init__()
        M = mem_width(cfg)
        self.proj = nn.Linear(cfg.d_model, M, bias=False)
        self.norm = RMSNorm(M, cfg.norm_eps)
        self.out = nn.Linear(M, cfg.n_experts, bias=False)

    def memory(self, h32, doc_ids, positions, cfg):
        B, T = doc_ids.shape
        v = F.linear(h32, self.proj.weight.to(h32.dtype)).view(B, T, -1)
        hls = mem_halflives(cfg)
        if len(hls) == 1:
            return causal_ema(v, doc_ids, positions, hls[0]).reshape(B * T, -1)
        w = v.shape[-1] // len(hls)
        return torch.cat([causal_ema(v[..., i * w:(i + 1) * w], doc_ids, positions, hl)
                          for i, hl in enumerate(hls)], -1).reshape(B * T, -1)

    def logits(self, m):
        return F.linear(self.norm(m), self.out.weight.to(m.dtype))


class Aggregator(nn.Module):
    """z_t = Agg(per-draft features (6 x 11), aggregate features (8), forecast-token embeddings). Training only."""

    def __init__(self, cfg):
        super().__init__()
        fc, d = cfg.fc_dim, cfg.d_model
        self.kind = cfg.agg
        self.H = list(cfg.horizons)
        self.slot = nn.Parameter(torch.zeros(N_DRAFT, 16))
        self.phi1 = nn.Linear(N_FEAT_DRAFT + 16, 64)
        self.phi2 = nn.Linear(64, 64)
        self.eproj = nn.Linear(d * len(self.H), fc)
        self.rho1 = nn.Linear(64 + N_FEAT_AGG + fc, 2 * fc)
        self.rho2 = nn.Linear(2 * fc, fc)
        self.norm = RMSNorm(fc, cfg.norm_eps)
        if self.kind == "pool":
            self.pool_w = nn.Parameter(torch.zeros(N_DRAFT + 2))
        if self.kind == "xattn":
            self.q = nn.Parameter(torch.zeros(64))
            self.wk = nn.Linear(64, 64, bias=False)
            self.wv = nn.Linear(64, 64, bias=False)

    def forward(self, fc, emb_w):
        dt = self.phi1.weight.dtype
        df, af = fc["dfeat"].to(dt), fc["afeat"].to(dt)  # (N,6,11), (N,8)
        N = df.shape[0]
        ph = F.gelu(self.phi1(torch.cat([df, self.slot.unsqueeze(0).expand(N, -1, -1)], -1)), approximate="tanh")
        ph = self.phi2(ph)  # (N,6,64)
        if self.kind == "deepsets":
            s = ph.sum(1)
        elif self.kind == "pool":
            a, b = self.pool_w[0], self.pool_w[1]
            logit = a * df[..., 9] + b * df[..., 1] + self.pool_w[2:].view(1, -1)
            s = (torch.softmax(logit, -1).unsqueeze(-1) * ph).sum(1)
        else:  # xattn: learned query (independent of h) over the 6 draft tokens, 2 heads of 32
            k = self.wk(ph).view(N, N_DRAFT, 2, 32)
            v = self.wv(ph).view(N, N_DRAFT, 2, 32)
            q = self.q.view(1, 1, 2, 32)
            w = torch.softmax((k * q).sum(-1) / math.sqrt(32.0), 1)  # (N,6,2)
            s = (w.unsqueeze(-1) * v).sum(1).reshape(N, 64)
        W = emb_w.detach().to(dt)
        es = []
        for h in self.H:
            if h == 1:
                es.append((fc["ens_p"].to(dt).unsqueeze(-1) * W[fc["ens_ids"]]).sum(-2))
            else:
                es.append((fc[f"vote{h}"].to(dt).unsqueeze(-1) * W[fc[f"cand{h}"]]).sum(-2))
        e = self.eproj(torch.cat(es, -1))
        z = self.rho2(F.gelu(self.rho1(torch.cat([s, af, e], -1)), approximate="tanh"))
        return self.norm(z)


@dataclass
class CFRecord:
    pass


class _Tap(torch.autograd.Function):
    """Identity whose backward stores dL/dy at the CF-sampled rows (fp32) into a python dict."""

    @staticmethod
    def forward(ctx, y, idx, store, key):
        ctx.idx, ctx.store, ctx.key = idx, store, key
        return y.view_as(y)

    @staticmethod
    def backward(ctx, g):
        ctx.store[ctx.key] = g.index_select(0, ctx.idx).detach().float()
        return g, None, None, None


@torch.compiler.disable
def _tap(y, idx, store, key):
    return _Tap.apply(y, idx, store, key)


@torch.compiler.disable
def _record(store, key, val):
    store[key] = val


class LFRModel(nn.Module):
    def __init__(self, cfg: LFRConfig, seed: int):
        super().__init__()
        self.cfg = cfg
        self.n_moe = cfg.n_layer - 1
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg, i > 0) for i in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.d_model, cfg.norm_eps)
        torch.manual_seed(seed)  # backbone draw: identical across conditions within a seed
        self.reset_backbone()
        k = 1
        if cfg.router == "student":
            self.student = nn.ModuleList([StudentMemory(cfg) for _ in range(self.n_moe)])
            g = torch.Generator().manual_seed(seed + 1_000_003 * k)
            with torch.no_grad():
                for sm in self.student:
                    sm.proj.weight.copy_(torch.randn(sm.proj.weight.shape, generator=g) * cfg.init_std)
                    sm.norm.weight.fill_(1.0)
                    sm.out.weight.zero_()
        k += 1
        if cfg.teacher:
            g = torch.Generator().manual_seed(seed + 1_000_003 * k)
            fc, d, hid = cfg.fc_dim, cfg.d_model, cfg.tc_hidden
            self.tc_agg = Aggregator(cfg)
            self.tc_trunk = nn.Linear(d + fc + 3, hid)
            self.tc_layer = nn.Parameter(torch.zeros(self.n_moe, fc))
            self.tc_head_w = nn.Parameter(torch.zeros(self.n_moe, hid, 2 * cfg.n_experts))
            self.tc_head_b = nn.Parameter(torch.zeros(self.n_moe, 2 * cfg.n_experts))
            with torch.no_grad():
                for n, p in list(self.tc_agg.named_parameters()) + list(self.tc_trunk.named_parameters()) + \
                        [("tc_layer", self.tc_layer), ("tc_head_w", self.tc_head_w)]:
                    if n.endswith("bias") or (p.dim() == 1 and "norm" in n):
                        p.fill_(1.0 if "norm" in n else 0.0)
                    else:
                        p.copy_(torch.randn(p.shape, generator=g) * cfg.init_std)
                if cfg.agg == "pool":
                    self.tc_agg.pool_w.zero_()
                self.tc_head_b.zero_()

    @torch.no_grad()
    def reset_backbone(self):
        std = self.cfg.init_std
        out_std = std / math.sqrt(2.0 * self.cfg.n_layer)
        for name, p in self.named_parameters():
            if name.startswith(("student", TEACHER_PREFIX)):
                continue
            if p.dim() == 1:
                p.fill_(1.0)
            elif name.endswith(("attn.proj.weight", "w2.weight", "experts.w2")):
                p.normal_(0.0, out_std)
            else:
                p.normal_(0.0, std)

    # ------------------------------------------------ bookkeeping
    def backbone_names(self):
        return [n for n, _ in self.named_parameters() if not n.startswith(("student", TEACHER_PREFIX))]

    def inference_params(self):
        return {n: p for n, p in self.named_parameters() if not n.startswith(TEACHER_PREFIX)}

    def param_counts(self):
        cfg = self.cfg
        inf = sum(p.numel() for p in self.inference_params().values())
        teacher = sum(p.numel() for n, p in self.named_parameters() if n.startswith(TEACHER_PREFIX))
        per_expert = 2 * cfg.d_model * cfg.expert_ffn
        inactive = self.n_moe * (cfg.n_experts - cfg.top_k) * per_expert
        student = sum(p.numel() for n, p in self.named_parameters() if n.startswith("student"))
        emb = self.tok_emb.weight.numel()
        return {"total_inference": inf, "active_per_token": inf - inactive, "teacher_training_only": teacher,
                "student_memory": student, "embedding_tied": emb, "active_non_embedding": inf - inactive - emb}

    def strip_teacher(self):
        """Delete every training-only teacher parameter (P3 and inference export)."""
        for n in ("tc_agg", "tc_trunk", "tc_layer", "tc_head_w", "tc_head_b"):
            if hasattr(self, n):
                delattr(self, n)
        self.cfg = LFRConfig.from_dict({**self.cfg.to_dict(), "teacher": False})
        return self

    @staticmethod
    def attention_mask(doc_ids):
        T = doc_ids.shape[1]
        same = doc_ids.unsqueeze(2) == doc_ids.unsqueeze(1)
        causal = torch.ones(T, T, dtype=torch.bool, device=doc_ids.device).tril()
        m = same & causal & (doc_ids >= 0).unsqueeze(2)
        pad_self = (doc_ids < 0).unsqueeze(2) & torch.eye(T, dtype=torch.bool, device=doc_ids.device)
        return (m | pad_self).unsqueeze(1)

    def uses_teacher(self, mi):
        return self.cfg.teacher and (self.cfg.inject == "all" or mi >= 4)

    @torch.compiler.disable
    def aggregate(self, fc):
        """Forecast representation z (N, fc_dim) and the 3 explicit teacher scalars (N, 3): forecast entropy,
        agreement, disagreement. fp32, eager (compiler-disabled)."""
        flat = {k: v.reshape(-1, *v.shape[2:]) for k, v in fc.items()}
        with torch.autocast(device_type=self.tok_emb.weight.device.type, enabled=False):
            z = self.tc_agg(flat, self.tok_emb.weight)
            af = flat["afeat"].to(z.dtype)
            sc = torch.stack([af[:, 1], af[:, 2], af[:, 5]], -1)
        return z, sc

    @torch.compiler.disable
    def teacher_out(self, mi, h32, z, sc):
        """(u, s): predicted standardized utility and log-variance (N, 32 each), fp32 eager."""
        with torch.autocast(device_type=h32.device.type, enabled=False):
            a = torch.cat([h32.to(z.dtype), z + self.tc_layer[mi].view(1, -1), sc.to(z.dtype)], -1)
            t = F.gelu(F.linear(a, self.tc_trunk.weight, self.tc_trunk.bias), approximate="tanh")
            o = t @ self.tc_head_w[mi] + self.tc_head_b[mi]
        E = self.cfg.n_experts
        return o[:, :E], o[:, E:].clamp(-8.0, 8.0)

    def teacher_logits(self, u, s):
        return (u - self.cfg.kappa * torch.exp(0.5 * s)) / self.cfg.tau_t

    # ------------------------------------------------ forward
    def forward(self, batch, ctl=None):
        """batch (B,T): input_ids, doc_ids, positions; optional 'fc' (feature dict, teacher), 'lam' (0-d tensor),
        'domain' + 'seq_id' (EDL loss only), 'cf_idx' (n,) flat CF positions, 'exclude_mask' (B,T) bool.
        ctl: teach, edl, lc, collect, exclude {mi: [experts]}, perturb (mi, idx, delta), cf_store (dict).
        Returns dict(hidden, aux, z, kl, edl, lc, records)."""
        ctl = ctl or {}
        cfg = self.cfg
        ids, doc_ids, positions = batch["input_ids"], batch["doc_ids"], batch["positions"]
        B, T = ids.shape
        dev = ids.device
        N = B * T
        x = self.tok_emb(ids)
        cos, sin = rope_cos_sin(positions, cfg.d_model // cfg.n_head, cfg.rope_base, x.dtype)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        mask = self.attention_mask(doc_ids)
        nonpad = (doc_ids >= 0).reshape(-1)
        rd = torch.float64 if self.tok_emb.weight.dtype == torch.float64 else torch.float32
        npf = nonpad.to(rd)
        n_valid = npf.sum().clamp(min=1)
        teach = bool(ctl.get("teach")) and cfg.teacher
        student = cfg.router == "student"
        store = ctl.get("cf_store")
        cf = batch.get("cf_idx") if store is not None else None
        z = sc = None
        if teach:
            z, sc = self.aggregate(batch["fc"])
            lam = batch["lam"].to(rd)
        excl = ctl.get("exclude") or {}
        pert = ctl.get("perturb")
        z0 = torch.zeros((), device=dev, dtype=rd)
        aux, zl, kl, edl, lc = z0, z0, z0, z0, z0
        records = []
        mi = 0
        for blk in self.blocks:
            x = x + blk.attn(blk.norm1(x), cos, sin, mask)
            h = blk.norm2(x)
            if not blk.is_moe:
                x = x + blk.ffn(h)
                continue
            moe = blk.ffn
            hf = h.reshape(N, -1)
            use_t = teach and self.uses_teacher(mi)
            with torch.autocast(device_type=dev.type, enabled=False):
                h32 = hf.to(rd)
                lS = F.linear(h32, moe.router.weight.to(rd))
                m = None
                if student:
                    sm = self.student[mi]
                    m = sm.memory(h32, doc_ids, positions, cfg)
                    lS = lS + sm.logits(m)
                lroute = lS
                if use_t:
                    u, s = self.teacher_out(mi, h32, z, sc)
                    lT = self.teacher_logits(u, s)
                    lroute = lam * lT + (1.0 - lam) * lS
                    # teacher -> student KL on stop-gradient hidden states (student router weights only)
                    hd = h32.detach()
                    lSd = F.linear(hd, moe.router.weight.to(rd))
                    md = None
                    if student:
                        md = sm.memory(hd, doc_ids, positions, cfg)
                        lSd = lSd + sm.logits(md)
                    lpt = torch.log_softmax(lT.detach(), -1)
                    lps = torch.log_softmax(lSd, -1)
                    kl = kl + ((lpt.exp() * (lpt - lps)).sum(-1) * npf).sum() / n_valid
                if mi in excl:
                    em = torch.zeros(cfg.n_experts, dtype=torch.bool, device=dev)
                    em[list(excl[mi])] = True
                    kill = (batch["exclude_mask"].reshape(-1, 1) if "exclude_mask" in batch else
                            torch.ones(N, 1, dtype=torch.bool, device=dev)) & em.view(1, -1)
                    lroute = lroute.masked_fill(kill, float("-inf"))
                p = torch.softmax(lroute, -1)
                gates, topi = p.topk(cfg.top_k, -1)
                cnt = F.one_hot(topi, cfg.n_experts).to(rd).sum(1)
                f = (cnt * npf.view(-1, 1)).sum(0) / (n_valid * cfg.top_k)
                P = (p * npf.view(-1, 1)).sum(0) / n_valid
                aux = aux + cfg.n_experts * (f * P).sum()
                if not excl:
                    zl = zl + (torch.logsumexp(lroute, -1).pow(2) * npf).sum() / n_valid
                if ctl.get("edl") and "domain" in batch:
                    edl = edl + edl_loss(p, batch["domain"].reshape(-1), batch["seq_id"].reshape(-1), nonpad)
                if cf is not None:
                    rec = {"x": h32.index_select(0, cf).detach(), "r4": topi.index_select(0, cf)[:, -1],
                           "g4": gates.index_select(0, cf)[:, -1].detach(), "topi": topi.index_select(0, cf)}
                    if student:
                        rec["m"] = m.index_select(0, cf).detach()
                    if use_t:
                        rec["u"] = u.index_select(0, cf).detach()
                    _record(store, f"rec{mi}", rec)
                if ctl.get("lc") and use_t and cf is not None and cf.numel() > 1:
                    # soft latent consistency on stop-gradient states at CF positions (router weights only)
                    xd = h32.detach().index_select(0, cf)
                    ld = F.linear(xd, moe.router.weight.to(rd))
                    if student:
                        ld = ld + sm.logits(md.index_select(0, cf))
                    ut, st_ = self.teacher_out(mi, xd, z.index_select(0, cf), sc.index_select(0, cf))
                    pl = torch.softmax(lam * self.teacher_logits(ut, st_) + (1 - lam) * ld, -1)
                    lc = lc + latent_consistency(pl, z.index_select(0, cf).detach(), u.index_select(0, cf).detach())
            y = moe.experts(hf, topi, gates) + moe.shared(hf)
            if pert is not None and pert[0] == mi:
                y = y.index_add(0, pert[1], pert[2].to(y.dtype))
            if cf is not None and ctl.get("tap"):
                y = _tap(y, cf, store, f"grad{mi}")
            x = x + y.view(B, T, -1)
            if ctl.get("collect"):
                records.append({"topi": topi.view(B, T, -1), "p": p.view(B, T, -1)})
            mi += 1
        L = float(self.n_moe)
        nt = float(sum(self.uses_teacher(i) for i in range(self.n_moe))) or 1.0
        return {"hidden": self.norm_f(x), "aux": aux, "z": zl, "kl": kl / nt, "edl": edl / L, "lc": lc / nt,
                "records": records}

    def lm_logits(self, hidden):
        return F.linear(hidden, self.tok_emb.weight)


@torch.compiler.disable
def latent_consistency(p, z, u, max_pairs=4096, gen_seed=0):
    """Soft latent-consistency (protocol 6.6). p (n,E) routing distributions (grad), z (n,fc) forecast codes and
    u (n,E) predicted standardized utilities (both detached). Pull pairs with similar forecasts AND agreeing utility
    rankings together (JS), push pairs with dissimilar forecasts AND disagreeing rankings apart (bounded)."""
    n = p.shape[0]
    g = torch.Generator(device="cpu").manual_seed(gen_seed + n)
    i = torch.randint(n, (max_pairs,), generator=g).to(p.device)
    j = torch.randint(n, (max_pairs,), generator=g).to(p.device)
    keep = i != j
    i, j = i[keep], j[keep]
    sz = F.cosine_similarity(z[i], z[j], -1)
    uc = u - u.mean(-1, keepdim=True)
    su = F.cosine_similarity(uc[i], uc[j], -1)
    pull = (sz > 0.5) & (su > 0.3)
    push = (sz < 0) & (su < 0)
    w = torch.where(pull, sz * su, torch.zeros_like(sz))
    w2 = torch.where(push, (sz * su).abs(), torch.zeros_like(sz))
    pi, pj = p[i].clamp(min=1e-9), p[j].clamp(min=1e-9)
    mm = 0.5 * (pi + pj)
    js = 0.5 * (pi * (pi.log() - mm.log())).sum(-1) + 0.5 * (pj * (pj.log() - mm.log())).sum(-1)
    return (w * js).mean() - (w2 * js.clamp(max=0.3)).mean()


def standardize_utility(delta):
    """Per (t,l) standardization across experts: (Delta - mean) / (std + 1e-6)."""
    return (delta - delta.mean(-1, keepdim=True)) / (delta.std(-1, keepdim=True) + 1e-6)


def utility_losses(model, rec, feats_cf, util_target, mode, mi):
    """Teacher utility loss (rank or reg; s always trained by NLL at 0.1 for rank) and student utility loss on stop-
    gradient inputs. rec: stored CF record of layer mi; feats_cf: feature dict at the CF rows (flat (n,...));
    util_target: standardized utilities (n, E). Returns (teacher_loss, student_loss) tensors (means)."""
    tgt = util_target
    out_t = torch.zeros((), device=tgt.device)
    if model.cfg.teacher and model.uses_teacher(mi):
        z, sc = model.aggregate({k: v.unsqueeze(1) for k, v in feats_cf.items()})
        u, s = model.teacher_out(mi, rec["x"], z, sc)
        nll = 0.5 * ((tgt - u) ** 2 * torch.exp(-s) + s).mean()
        if mode == "rank":
            q = torch.softmax(tgt / 0.5, -1)
            out_t = (q * (torch.log(q.clamp(min=1e-12)) - torch.log_softmax(u, -1))).sum(-1).mean() + 0.1 * nll
        else:
            out_t = nll
    q = torch.softmax(tgt / 0.5, -1)
    blk = model.blocks[mi + 1].ffn
    lS = F.linear(rec["x"], blk.router.weight.to(rec["x"].dtype))
    if model.cfg.router == "student":
        lS = lS + model.student[mi].logits(rec["m"])
    out_s = (q * (torch.log(q.clamp(min=1e-12)) - torch.log_softmax(lS, -1))).sum(-1).mean()
    return out_t, out_s


MAIN_CFG = LFRConfig()
