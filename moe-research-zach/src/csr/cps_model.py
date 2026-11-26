"""csr-context-prior-specialization-v2 (cps) model. New module; V1 (lfr_*) modules are imported read-only.

Backbone: exactly the V1 / TOKEN-4 backbone (csr.lfr_model.LFRModel with the linear router), built and initialised by
LFRModel.__init__ before any V2 module exists -> parameter-for-parameter identical to TOKEN-4 within a seed.

Routing at every injected MoE layer l (unrestricted top-4 over all 32 experts, gates = softmax probabilities):
    logits(t, .) = W_l h_t  +  alpha_l(t) * prior_l(c_t)            (variant "prior"; "memory" = same, no drafts)
    logits(t, .) = 0.1 W_l h_t + prior_l(c_t)                          (variant "noresid": M6 diagnostic only)
  prior_l(c)  = U_l c (32 logits; optionally centred and rescaled to the RMS of the centred token logits, detached)
  alpha_l(t)  = alpha_max * sigmoid(b_l + w_l . c_t + v_l . RMSNorm(h_t)),  b_l = alpha_init (near 0), w_l = v_l = 0
  c_t         = context state of t's chunk, computed causally at the last position before the chunk (source index
                b(t) - 1); the first chunk of every document has the null context c = 0 (prior 0).
Context encoders (one context per forward, from the residual stream x entering the first injected block):
  s_t  = [causal within-document EMA_hl(P RMSNorm(x_t)) for hl in mem_hls]
  c^S  = RMSNorm(W2 GELU(W1 s))                                             (student: inference, prefix "cp_")
  c^T  = RMSNorm(W2' GELU(W1' [s, z_t, EMA_16(a_t), EMA_128(a_t)]))         (teacher: training only, prefix "tc_")
         z_t = V1 deepsets aggregator of the cached causal draft-forecast features (frozen drafts, prefix <= t),
         a_t = (forecast entropy, draft agreement, draft disagreement) raw cached features
  executed context during teacher phases: lam c^T + (1 - lam) c^S; student-only phase / inference: c^S.
Chunk boundaries: fixed (positions 0, L, 2L, ... of each document) or causal adaptive: rising edges of
  d_t = 1 - cos(EMA_4(R x_t), EMA_64(R x_t)) above a frozen threshold theta (R: frozen random token embedding,
  never trained), plus forced boundaries at multiples of 4L. Real future tokens are never used.
No expert is ever masked or made mandatory by the prior; the shared expert path is unchanged.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import RMSNorm, rope_cos_sin
from .lfr_model import LFRConfig, LFRModel, Aggregator, causal_ema, _tap, _record

CTX_PREFIX = "cp_"
TEACHER_PREFIX = "tc_"
DET_DIM = 32
DET_SEED = 20261112


@dataclass
class CPSConfig:
    variant: str = "prior"  # prior | memory | noresid | token (TOKEN-4 math: context never computed)
    teacher: bool = True  # draft-derived teacher context in phases 1-2
    chunk: str = "fixed"  # fixed | adaptive
    chunk_len: int = 64
    det_theta: float = 0.0  # frozen adaptive-boundary threshold (by chunk_len), set from the prereg calibration
    ctx_dim: int = 64
    mem_hls: list = field(default_factory=lambda: [16.0, 128.0, 1024.0])
    mem_w: int = 32  # EMA width per half-life
    enc_hidden: int = 128
    inject: str = "all"  # all | middle | late
    alpha_init: float = -4.6  # sigmoid(-4.6) ~ 0.01
    alpha_max: float = 1.0
    prior_norm: str = "match"  # match | none
    fc_dim: int = 64
    horizons: list = field(default_factory=lambda: [1])
    shuffle_ctx: bool = False  # M5: contexts rolled across rows (training and evaluation)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


def injected_layers(inject, n_moe=11):
    if inject == "all":
        return list(range(n_moe))
    if inject == "middle":
        return list(range(3, 8))
    if inject == "late":
        return list(range(6, n_moe))
    raise ValueError(inject)


def chunk_source(doc_ids, positions, boundary):
    """boundary (B,T) bool (True at chunk starts; position 0 of every document always True). Returns src (B,T) long:
    flat-within-row index of the last position before t's chunk, -1 for the first chunk of a document / pads."""
    B, T = doc_ids.shape
    ar = torch.arange(T, device=doc_ids.device).view(1, T).expand(B, T)
    bnd = boundary | (positions == 0)
    start = torch.where(bnd, ar, torch.zeros_like(ar))
    start = torch.cummax(start, 1).values
    src = start - 1
    first = positions[torch.arange(B, device=doc_ids.device).view(B, 1), start] == 0
    src = torch.where(first | (doc_ids < 0), torch.full_like(src, -1), src)
    return src


def fixed_boundaries(positions, L):
    return (positions % L) == 0


def detector_stat(ids, doc_ids, positions, R):
    """d_t = 1 - cos(EMA_4(R x), EMA_64(R x)) within the document (causal; R frozen)."""
    e = R[ids.clamp(min=0)].float()
    f = causal_ema(e, doc_ids, positions, 4.0)
    g = causal_ema(e, doc_ids, positions, 64.0)
    return 1.0 - F.cosine_similarity(f, g, dim=-1, eps=1e-6)


def adaptive_boundaries(ids, doc_ids, positions, R, theta, L):
    d = detector_stat(ids, doc_ids, positions, R)
    above = d > theta
    prev = torch.zeros_like(above)
    prev[:, 1:] = above[:, :-1]
    prev = prev & (positions > 0)
    return (above & ~prev) | ((positions % (4 * L)) == 0)


class CtxEncoder(nn.Module):
    def __init__(self, d_in, hid, d_out, eps):
        super().__init__()
        self.w1 = nn.Linear(d_in, hid)
        self.w2 = nn.Linear(hid, d_out)
        self.norm = RMSNorm(d_out, eps)

    def forward(self, x):
        dt = self.w1.weight.dtype
        return self.norm(self.w2(F.gelu(self.w1(x.to(dt)), approximate="tanh")))


class CPSModel(LFRModel):
    def __init__(self, ccfg: CPSConfig, seed: int, backbone: dict | None = None):
        bb = LFRConfig.from_dict({**(backbone or {}), "router": "linear", "teacher": False})
        super().__init__(bb, seed)  # identical TOKEN-4 backbone draw (V2 modules are created afterwards)
        self.ccfg = ccfg
        cc = ccfg
        self.inj = injected_layers(cc.inject, self.n_moe)
        d, E = bb.d_model, bb.n_experts
        S = len(cc.mem_hls) * cc.mem_w
        g = torch.Generator().manual_seed(seed + 7_000_001)
        self.cp_in = nn.Linear(d, S, bias=False)
        self.cp_in_norm = RMSNorm(d, bb.norm_eps)
        self.cp_enc = CtxEncoder(S, cc.enc_hidden, cc.ctx_dim, bb.norm_eps)
        nl = len(self.inj)
        self.cp_U = nn.Parameter(torch.empty(nl, cc.ctx_dim, E))
        self.cp_ab = nn.Parameter(torch.full((nl,), float(cc.alpha_init)))
        self.cp_aw = nn.Parameter(torch.zeros(nl, cc.ctx_dim))
        self.cp_av = nn.Parameter(torch.zeros(nl, d))
        self.register_buffer("cp_R", torch.randn(bb.vocab_size, DET_DIM, generator=torch.Generator().manual_seed(
            DET_SEED)), persistent=False)
        with torch.no_grad():  # fan-in scale: O(1) pre-norm activations (std 0.02 made the encoder RMSNorm amplify
            # gradients ~100x; found in the CPU dry run before any V2 result)
            for n, p, fan in [("in", self.cp_in.weight, d), ("w1", self.cp_enc.w1.weight, S),
                              ("w2", self.cp_enc.w2.weight, cc.enc_hidden), ("U", self.cp_U, cc.ctx_dim)]:
                p.copy_(torch.randn(p.shape, generator=g) / math.sqrt(fan))
            self.cp_enc.w1.bias.zero_()
            self.cp_enc.w2.bias.zero_()
            self.cp_enc.norm.weight.fill_(1.0)
            self.cp_in_norm.weight.fill_(1.0)
        if cc.teacher:
            gt = torch.Generator().manual_seed(seed + 7_000_003)
            acfg = LFRConfig.from_dict({**bb.to_dict(), "agg": "deepsets", "fc_dim": cc.fc_dim,
                                        "horizons": list(cc.horizons)})
            self.tc_agg = Aggregator(acfg)
            self.tc_enc = CtxEncoder(S + cc.fc_dim + 6, cc.enc_hidden, cc.ctx_dim, bb.norm_eps)
            with torch.no_grad():
                for n, p in list(self.tc_agg.named_parameters()) + list(self.tc_enc.named_parameters()):
                    if n.endswith("bias"):
                        p.zero_()
                    elif p.dim() == 1 and "norm" in n:
                        p.fill_(1.0)
                    elif n.startswith(("w1", "w2")):  # tc_enc linear layers: fan-in scale
                        p.copy_(torch.randn(p.shape, generator=gt) / math.sqrt(p.shape[1]))
                    else:  # V1 aggregator initialisation
                        p.copy_(torch.randn(p.shape, generator=gt) * bb.init_std)

    # ------------------------------------------------ bookkeeping
    def backbone_names(self):
        return [n for n, _ in self.named_parameters() if not n.startswith((CTX_PREFIX, TEACHER_PREFIX))]

    def inference_params(self):
        return {n: p for n, p in self.named_parameters() if not n.startswith(TEACHER_PREFIX)}

    def param_counts(self):
        cfg = self.cfg
        inf = sum(p.numel() for p in self.inference_params().values())
        teacher = sum(p.numel() for n, p in self.named_parameters() if n.startswith(TEACHER_PREFIX))
        ctx = sum(p.numel() for n, p in self.named_parameters() if n.startswith(CTX_PREFIX))
        inactive = self.n_moe * (cfg.n_experts - cfg.top_k) * 2 * cfg.d_model * cfg.expert_ffn
        emb = self.tok_emb.weight.numel()
        return {"total_inference": inf, "active_per_token": inf - inactive, "teacher_training_only": teacher,
                "context_prior": ctx, "embedding_tied": emb, "active_non_embedding": inf - inactive - emb}

    def has_teacher(self):
        return hasattr(self, "tc_enc")

    def strip_teacher(self):
        for n in ("tc_agg", "tc_enc"):
            if hasattr(self, n):
                delattr(self, n)
        self.ccfg = CPSConfig.from_dict({**self.ccfg.to_dict(), "teacher": False})
        return self

    # ------------------------------------------------ context
    def boundaries(self, ids, doc_ids, positions):
        cc = self.ccfg
        if cc.chunk == "fixed":
            return fixed_boundaries(positions, cc.chunk_len)
        with torch.autocast(device_type=ids.device.type, enabled=False):  # fp32 (bf16 would round EMA decays)
            return adaptive_boundaries(ids, doc_ids, positions, self.cp_R.float(), cc.det_theta, cc.chunk_len)

    def mem_state(self, x32, doc_ids, positions):
        """fp32 always: under bf16 autocast the EMA decay r = 0.5^(1/1024) rounds to 1 -> 0 normaliser (non-finite)."""
        with torch.autocast(device_type=x32.device.type, enabled=False):
            x32 = x32.float()
            v = F.linear(self.cp_in_norm(x32), self.cp_in.weight.float())
            w = self.ccfg.mem_w
            return torch.cat([causal_ema(v[..., i * w:(i + 1) * w], doc_ids, positions, hl)
                              for i, hl in enumerate(self.ccfg.mem_hls)], -1)

    @torch.compiler.disable
    def teacher_inputs(self, fc, doc_ids, positions):
        """(z (B,T,fc), a_ema (B,T,6)) from the cached draft-forecast features (fp32)."""
        B, T = doc_ids.shape
        flat = {k: v.reshape(-1, *v.shape[2:]) for k, v in fc.items()}
        with torch.autocast(device_type=doc_ids.device.type, enabled=False):
            z = self.tc_agg(flat, self.tok_emb.weight).view(B, T, -1)
            af = flat["afeat"].float()
            a = torch.stack([af[:, 1], af[:, 2], af[:, 5]], -1).view(B, T, 3)
            ae = torch.cat([causal_ema(a, doc_ids, positions, 16.0), causal_ema(a, doc_ids, positions, 128.0)], -1)
        return z, ae

    @torch.compiler.disable
    def enc_teacher(self, s, z, ae):
        with torch.autocast(device_type=s.device.type, enabled=False):
            return self.tc_enc(torch.cat([s.float(), z.float(), ae.float()], -1))

    @torch.compiler.disable
    def enc_student(self, s):
        with torch.autocast(device_type=s.device.type, enabled=False):
            return self.cp_enc(s.float())

    def prior_logits(self, j, c, lS=None):
        """Context prior of injected layer j (index into self.inj) for contexts c (n, ctx_dim)."""
        pr = c @ self.cp_U[j].to(c.dtype)
        if self.ccfg.prior_norm == "match" and lS is not None:
            pc = pr - pr.mean(-1, keepdim=True)
            sc = (lS - lS.mean(-1, keepdim=True)).detach().pow(2).mean(-1, keepdim=True).sqrt()
            pr = pc * torch.rsqrt(pc.pow(2).mean(-1, keepdim=True) + 1e-8) * sc  # finite gradient at c = 0
        return pr

    def alpha(self, j, c, h32):
        cc = self.ccfg
        hn = h32 * torch.rsqrt(h32.pow(2).mean(-1, keepdim=True) + 1e-6)
        a = self.cp_ab[j] + c @ self.cp_aw[j].to(c.dtype) + hn @ self.cp_av[j].to(hn.dtype)
        return cc.alpha_max * torch.sigmoid(a)

    @staticmethod
    def gather_src(v, src):
        """v (B,T,D), src (B,T) -> (B,T,D) with zeros where src < 0."""
        idx = src.clamp(min=0).unsqueeze(-1).expand(-1, -1, v.shape[-1])
        out = torch.gather(v, 1, idx)
        return out * (src >= 0).unsqueeze(-1).to(v.dtype)

    # ------------------------------------------------ forward
    def forward(self, batch, ctl=None):
        """batch (B,T): input_ids, doc_ids, positions; optional 'fc' + 'lam' (teacher), 'cf_idx', 'exclude_mask'.
        ctl: teach, collect, exclude {mi: [experts]}, perturb (mi, idx, delta), cf_store (dict), tap, stats,
        ctx_off (evaluation ablation: prior disabled). Returns dict(hidden, aux, z, kl, edl, lc, records, cstats)."""
        ctl = ctl or {}
        cfg, cc = self.cfg, self.ccfg
        ids, doc_ids, positions = batch["input_ids"], batch["doc_ids"], batch["positions"]
        B, T = ids.shape
        dev = ids.device
        N = B * T
        x = self.tok_emb(ids)
        cos, sin = rope_cos_sin(positions, cfg.d_model // cfg.n_head, cfg.rope_base, x.dtype)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        mask = self.attention_mask(doc_ids)
        nonpad = (doc_ids >= 0).reshape(-1)
        rd = torch.float32
        npf = nonpad.to(rd)
        n_valid = npf.sum().clamp(min=1)
        teach = bool(ctl.get("teach")) and self.has_teacher()
        ctx_off = bool(ctl.get("ctx_off")) or bool(getattr(self, "force_ctx_off", False)) or cc.variant == "token"
        store = ctl.get("cf_store")
        cf = batch.get("cf_idx") if store is not None else None
        excl = ctl.get("exclude") or {}
        pert = ctl.get("perturb")
        z0 = torch.zeros((), device=dev, dtype=rd)
        aux, zl, kl, areg = z0, z0, z0, z0
        records = []
        cst = {"alpha": [], "prior_top4_overlap": [], "prior_rel_scale": [], "prior_entropy": []} \
            if ctl.get("stats") else None
        cuse = None
        mi = 0
        first_inj = self.inj[0]
        for blk in self.blocks:
            if blk.is_moe and mi == first_inj and cuse is None and not ctx_off:
                # ---- context state (once), from the residual stream entering the first injected block
                x32 = x.float()
                s = self.mem_state(x32, doc_ids, positions)  # (B,T,S)
                src = chunk_source(doc_ids, positions, self.boundaries(ids, doc_ids, positions))
                cS = self.enc_student(s)
                if teach:
                    zt, ae = self.teacher_inputs(batch["fc"], doc_ids, positions)
                    cT = self.enc_teacher(s, zt, ae)
                    lam = batch["lam"].to(rd)
                    cexe = lam * cT + (1.0 - lam) * cS
                    # teacher -> student context distillation on stop-gradient states (encoder weights only)
                    cSd = self.enc_student(s.detach())
                    kl = (((cSd - cT.detach()) ** 2).mean(-1).reshape(-1) * npf).sum() / n_valid
                else:
                    cexe = cS
                cuse = self.gather_src(cexe, src)  # (B,T,dc)
                if cc.shuffle_ctx:  # M5: contexts of another row (same column), own padding mask
                    cuse = torch.roll(cuse, 1, 0) * (src >= 0).unsqueeze(-1).to(cuse.dtype)
                cflat = cuse.reshape(N, -1)
                if cf is not None:
                    own = src.reshape(-1).index_select(0, cf)
                    rowb = torch.div(cf, T, rounding_mode="floor")
                    if cc.shuffle_ctx:  # donor row b-1, donor's own chunk source at the same column
                        srcf = torch.roll(src, 1, 0).reshape(-1).index_select(0, cf)
                        rowb = (rowb - 1) % B
                    else:
                        srcf = own
                    valid = (own >= 0) & (srcf >= 0)
                    srcf = torch.where(valid, srcf, torch.full_like(srcf, -1))
                    gidx = (rowb * T + srcf.clamp(min=0))
                    crec = {"s": s.reshape(N, -1).index_select(0, gidx).detach(), "valid": valid,
                            "gidx": gidx, "doc": doc_ids.reshape(-1).index_select(0, cf),
                            "chunk": gidx * (srcf >= 0).long() + (-1 - cf) * (srcf < 0).long()}
                    if teach:
                        crec["z_in"] = {k: v.reshape(-1, *v.shape[2:]).index_select(0, gidx).detach()
                                        for k, v in batch["fc"].items()}
                        crec["ae"] = ae.reshape(N, -1).index_select(0, gidx).detach()
                    _record(store, "ctx", crec)
            x = x + blk.attn(blk.norm1(x), cos, sin, mask)
            h = blk.norm2(x)
            if not blk.is_moe:
                x = x + blk.ffn(h)
                continue
            moe = blk.ffn
            hf = h.reshape(N, -1)
            with torch.autocast(device_type=dev.type, enabled=False):
                h32 = hf.to(rd)
                lS = F.linear(h32, moe.router.weight.to(rd))
                lroute = lS
                if mi in self.inj and cuse is not None:
                    j = self.inj.index(mi)
                    pr = self.prior_logits(j, cflat, lS)
                    if cc.variant == "noresid":
                        lroute = 0.1 * lS + pr
                        al = torch.ones(N, 1, device=dev, dtype=rd)
                    else:
                        al = self.alpha(j, cflat, h32).unsqueeze(-1)
                        lroute = lS + al * pr
                        areg = areg + (al.squeeze(-1) * npf).sum() / n_valid / len(self.inj)
                    if cst is not None:
                        v_ = nonpad
                        cst["alpha"].append(float(al.squeeze(-1)[v_].mean()))
                        pt4 = pr.topk(cfg.top_k, -1).indices
                        et4 = lroute.topk(cfg.top_k, -1).indices
                        ov = (pt4.unsqueeze(-1) == et4.unsqueeze(-2)).any(-1).float().sum(-1)
                        cst["prior_top4_overlap"].append(float(ov[v_].mean()))
                        rs = (al * pr).std(-1) / lS.std(-1).clamp(min=1e-6)
                        cst["prior_rel_scale"].append(float(rs[v_].mean()))
                        pp = torch.softmax(pr, -1)
                        cst["prior_entropy"].append(float((-(pp * pp.clamp(min=1e-12).log()).sum(-1) /
                                                           math.log(cfg.n_experts))[v_].mean()))
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
                if cf is not None:
                    rec = {"x": h32.index_select(0, cf).detach(), "r4": topi.index_select(0, cf)[:, -1],
                           "g4": gates.index_select(0, cf)[:, -1].detach(), "topi": topi.index_select(0, cf),
                           "lS": lS.index_select(0, cf).detach()}
                    _record(store, f"rec{mi}", rec)
            y = moe.experts(hf, topi, gates) + moe.shared(hf)
            if pert is not None and pert[0] == mi:
                y = y.index_add(0, pert[1], pert[2].to(y.dtype))
            if cf is not None and ctl.get("tap"):
                y = _tap(y, cf, store, f"grad{mi}")
            x = x + y.view(B, T, -1)
            if ctl.get("collect"):
                records.append({"topi": topi.view(B, T, -1), "p": p.view(B, T, -1)})
            mi += 1
        return {"hidden": self.norm_f(x), "aux": aux, "z": zl, "kl": kl, "edl": z0, "lc": z0, "areg": areg,
                "records": records, "cstats": cst}

    # ------------------------------------------------ auxiliary losses (stop-gradient inputs)
    def cf_contexts(self, crec, lam, teach):
        """Recompute the executed / student / teacher contexts at CF rows from stored detached inputs (gradients
        reach only the context encoders, never the backbone)."""
        cS = self.enc_student(crec["s"])
        cT = None
        if teach and self.has_teacher() and "z_in" in crec:
            with torch.autocast(device_type=cS.device.type, enabled=False):
                z = self.tc_agg(crec["z_in"], self.tok_emb.weight)
            cT = self.enc_teacher(crec["s"], z, crec["ae"])
        v = crec["valid"].unsqueeze(-1).to(cS.dtype)
        cS = cS * v
        if cT is not None:
            cT = cT * v
            cE = lam * cT + (1.0 - lam) * cS
        else:
            cE = cS
        return cE, cS, cT


def standardize(delta):
    return (delta - delta.mean(-1, keepdim=True)) / (delta.std(-1, keepdim=True) + 1e-6)


def utility_loss(pred, tgt_std, mode):
    """pred (n,E) prior logits, tgt_std (n,E) standardized first-order utilities."""
    if mode == "corr":
        ps = standardize(pred)
        return (1.0 - (ps * tgt_std).mean(-1)).mean()
    if mode == "rank":
        q = torch.softmax(tgt_std / 0.5, -1)
        return (q * (torch.log(q.clamp(min=1e-12)) - torch.log_softmax(pred, -1))).sum(-1).mean()
    if mode == "mse":
        return ((standardize(pred) - tgt_std) ** 2).mean()
    raise ValueError(mode)


SPEC = {"c_hi": 0.5, "u_hi": 0.2, "c_lo": 0.2, "u_lo": 0.0, "temp": 0.1, "push_cap": 0.3, "max_pairs": 4096}


def spec_consistency(prior, c, u_std, u_raw, chunk, gen_seed=0):
    """Utility-conditioned latent consistency on CF rows of DIFFERENT chunks.
    prior (n,E) prior logits (with gradient), c (n,dc) contexts (detached), u_std / u_raw (n,E) standardized / raw
    first-order utilities (targets), chunk (n,) chunk keys. Pull together pairs with similar contexts AND agreeing
    utility rankings, push apart (bounded) pairs whose contexts and rankings both differ; a pair counts only when the
    top-4 experts preferred by each prior have positive estimated counterfactual utility at that row."""
    n = prior.shape[0]
    if n < 2:
        return prior.sum() * 0.0, {}
    g = torch.Generator(device="cpu").manual_seed(int(gen_seed))
    m = min(SPEC["max_pairs"], n * (n - 1) // 2)
    i = torch.randint(0, n, (m,), generator=g).to(prior.device)
    j = torch.randint(0, n, (m,), generator=g).to(prior.device)
    ok = chunk.index_select(0, i) != chunk.index_select(0, j)
    p = torch.softmax(prior.float(), -1)
    top = prior.detach().topk(4, -1).indices
    pos = u_raw.gather(1, top).sum(-1) > 0
    gate = (pos.index_select(0, i) & pos.index_select(0, j) & ok).float()
    cn = F.normalize(c.detach().float(), dim=-1)
    sc = (cn.index_select(0, i) * cn.index_select(0, j)).sum(-1)
    su = (u_std.index_select(0, i) * u_std.index_select(0, j)).mean(-1)
    t = SPEC["temp"]
    wp = torch.sigmoid((sc - SPEC["c_hi"]) / t) * torch.sigmoid((su - SPEC["u_hi"]) / t) * gate
    wn = torch.sigmoid((SPEC["c_lo"] - sc) / t) * torch.sigmoid((SPEC["u_lo"] - su) / t) * gate
    pi, pj = p.index_select(0, i), p.index_select(0, j)
    mm = 0.5 * (pi + pj)
    js = 0.5 * (pi * (pi.clamp(min=1e-12).log() - mm.clamp(min=1e-12).log())).sum(-1) + \
        0.5 * (pj * (pj.clamp(min=1e-12).log() - mm.clamp(min=1e-12).log())).sum(-1)
    loss = (wp * js).mean() - (wn * js.clamp(max=SPEC["push_cap"])).mean()
    return loss, {"pull_w": float(wp.mean()), "push_w": float(wn.mean()), "gate": float(gate.mean()),
                  "js": float(js.mean())}
