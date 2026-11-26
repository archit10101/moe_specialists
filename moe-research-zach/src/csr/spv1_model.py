"""csr-speculative-pool-routing-v1: chunk-level candidate pools + token-level top-k routing inside the pool
(docs/SPECULATIVE_POOL_ROUTING_V1_PREREG.md).

New module; CSRModel and every earlier routing mode are unchanged. PoolRouteModel subclasses CSRModel, so the
backbone (embedding, attention, dense layer 0, experts, shared experts, router matrices W_l) is constructed and
initialized exactly as TOKEN-4 for the same seed. Extra pool-selector / detector parameters are zero-initialized
or drawn from a dedicated generator afterwards, so they never change the backbone initialization.

Families (per MoE layer l; every token always activates exactly k=4 routed experts + the unchanged shared expert)
  token    : TOKEN-4. p_t = softmax(W_l h_t), top-4 by p, gate = p (global, no renormalisation).
  fixed    : blocks start at in-document positions 0, L, 2L, ... (L in {4, 8}).
  adaptive : a block starts at t iff t is a document start, or t - s >= H (s = current block start, H in {4, 8}),
             or score_l(t | pool of s) > tau_l, where score = (1 - sum_{e in pool} p_{t,e}) + w_l . sg(h_t) + b_l
             (router mass outside the cached pool + a learned linear term). The new pool applies to token t.
  spec     : a frozen draft proposes D[s, 1..B-1] greedily from the prefix x_0..x_s. A block starting at s ends
             before the first t = s + j (1 <= j < B) whose real input token x_t differs from D[s, j], or at s + B.
             (This is exactly the block structure of greedy speculative decoding: the next block starts at the
             target's correction / bonus token.)
Pool of a block starting at s (computed from causal state at s only):
  learned  : q_s = sg(W_l) h_s + U_l h_s [+ beta_l sg(W_l) g_s + V_l g_s for spec], with
             g_s = sum_j a_{l,j} rms(sg(Emb[D[s, j]])), a_l = softmax(alpha_l);  pool = TopM(q_s), M in {8, 16}.
  random   : pool = a fixed, content-independent, load-balanced table indexed by (layer, in-document position of s).
Token t in the block of s: logits z_t = W_l h_t (fp32), p_t = softmax(z_t / T_r); topi = top-4 of p_t restricted
to the pool; gates = p_t[topi] (global probabilities, as TOKEN-4). Gradient paths (forward values unchanged):
  selector : gates *= exp(log m_s[e] - sg(log m_s[e])), m_s = softmax(q_s / T_p)       (straight-through)
  detector : gates *= exp(log rho - sg(log rho)) at non-forced refreshes, exp(log(1-rho) - sg(.)) at reuses,
             rho = sigmoid((score - tau) / kappa)                                       (straight-through)
All block structures are computed with vectorized, GPU-resident pointer doubling (no per-token Python loop, no
host synchronisation, no CPU pass).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import CSRModel, ForwardOut, ModelConfig, RoutingConfig, rope_cos_sin

EXPERIMENT = "csr-speculative-pool-routing-v1"
FAMILIES = ("token", "fixed", "adaptive", "spec")
EXTRA_SEED_OFFSET = 2_000_003
RAND_TABLE_LEN = 1024
REFRESH_TARGETS = (0.5, 0.25, 0.15)


@dataclass
class PoolRouting:
    family: str = "token"  # token | fixed | adaptive | spec
    pool_assign: str = "learned"  # learned | random
    pool_size: int = 16  # M
    seg_len: int = 4  # fixed L / adaptive max hold H / speculative block B
    refresh_target: float = 0.5  # adaptive
    sel_temperature: float = 1.0
    sel_lr_mult: float = 1.0
    router_temperature: float = 1.0
    router_lr_mult: float = 1.0
    pool_dropout: float = 0.0
    pool_lb_coeff: float = 0.01
    sel_entropy_coeff: float = 0.0
    pool_warmup_frac: float = 0.0
    det_kappa: float = 0.05
    tau_lr: float = 0.5
    allow_test_sizes: bool = False  # unit tests only (configs never set it)

    def validate(self):
        assert self.family in FAMILIES, self.family
        if self.family == "token":
            return self
        assert self.pool_assign in ("learned", "random")
        if not self.allow_test_sizes:
            assert self.pool_size in (8, 16), self.pool_size
            assert self.seg_len in (4, 8), self.seg_len
            if self.family == "adaptive":
                assert self.refresh_target in REFRESH_TARGETS, self.refresh_target
                assert self.refresh_target > 1.0 / self.seg_len, "target refresh rate must exceed the 1/H floor"
        assert self.seg_len >= 2 and self.pool_size >= 1
        assert self.sel_temperature > 0 and self.router_temperature > 0 and self.det_kappa > 0
        assert self.sel_lr_mult > 0 and self.router_lr_mult > 0
        assert 0.0 <= self.pool_dropout < 0.5 and 0.0 <= self.pool_warmup_frac < 1.0
        return self

    @property
    def gamma(self):
        """Draft tokens per speculative block (B - 1)."""
        return self.seg_len - 1

    @classmethod
    def from_cfg(cls, r: dict):
        keys = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in r.items() if k in keys}).validate()

    def to_dict(self):
        return asdict(self)


# ------------------------------------------------------------------ vectorized block structure
def orbit_starts(nxt):
    """nxt: (B, S) long with nxt[:, i] in (i, S]; S is a sentinel. Returns (B, S) bool marking the orbit
    {0, nxt(0), nxt(nxt(0)), ...} computed by pointer doubling (log2(S) gathers, GPU-resident, no host sync)."""
    B, S = nxt.shape
    dev = nxt.device
    J = torch.cat([nxt, torch.full((B, 1), S, dtype=nxt.dtype, device=dev)], 1)
    K = max(1, int(S).bit_length())  # 2^K > S >= number of steps needed
    tables = [J]
    for _ in range(K - 1):
        tables.append(tables[-1].gather(1, tables[-1]))
    m = torch.arange(S + 1, device=dev)
    v = torch.zeros(B, S + 1, dtype=nxt.dtype, device=dev)
    for k in range(K):
        bit = ((m >> k) & 1).bool().unsqueeze(0)
        v = torch.where(bit, tables[k].gather(1, v), v)
    out = torch.zeros(B, S + 1, dtype=torch.bool, device=dev).scatter_(1, v, True)
    return out[:, :S]


def first_true_next(R):
    """R: (B, S, K) bool, R[:, s, k-1] = 'a new block starts at s + k if the current block started at s'; the
    last column must be all True. Returns nxt (B, S) = s + first such k, clamped to S."""
    B, S, K = R.shape
    k = R.int().argmax(-1) + 1  # first True (argmax returns the first maximal index)
    s = torch.arange(S, device=R.device).unsqueeze(0)
    return (s + k).clamp(max=S)


def block_start_index(starts):
    """starts (B, T) bool with starts[:, 0] True -> (B, T) index of the active block start of every token."""
    T = starts.shape[1]
    idx = torch.where(starts, torch.arange(T, device=starts.device).unsqueeze(0), torch.full_like(starts, -1,
                                                                                                    dtype=torch.long))
    return torch.cummax(idx, dim=1).values


def shift_left(x, k, fill):
    """x (B, T, ...) -> y with y[:, s] = x[:, s + k] (fill beyond the end)."""
    if k == 0:
        return x
    pad = torch.full_like(x[:, :k], fill)
    return torch.cat([x[:, k:], pad], 1)


def fixed_starts(positions, L):
    return (positions % L) == 0


def spec_starts(input_ids, positions, proposals, Bblk, forced=None):
    """Block starts of the speculative-pool rule. proposals (B, T, >= Bblk-1): D[s, j-1] = draft token proposed for
    position s + j from the prefix ending at s. forced: optional (B, T) bool of positions that must start a block."""
    B, T = input_ids.shape
    force = positions == 0
    if forced is not None:
        force = force | forced
    cols = []
    for j in range(1, Bblk):
        nxt_tok = shift_left(input_ids, j, -1)
        f = shift_left(force, j, True)
        cols.append(f | (nxt_tok != proposals[:, :, j - 1]))
    cols.append(torch.ones(B, T, dtype=torch.bool, device=input_ids.device))
    return orbit_starts(first_true_next(torch.stack(cols, -1)))


# ------------------------------------------------------------------ draft model (dense CSRModel) helpers
def draft_model_config(d=128, n_layer=4, n_head=4, ffn=512, vocab=50304, seq=1024):
    return ModelConfig(vocab_size=vocab, n_layer=n_layer, d_model=d, n_head=n_head, dense_ffn=ffn, moe_layers=(),
                       n_experts=1, top_k=1, expert_ffn=1, shared_ffn=0, max_seq_len=seq, tie_embeddings=True)


def draft_logits(draft: CSRModel, batch):
    return draft(batch, RoutingConfig()).logits


def _attn_qkv(blk, x, cos, sin):
    from .model import apply_rope
    a = blk.attn
    B, T, D = x.shape
    q, k, v = a.qkv(blk.norm1(x)).view(B, T, 3, a.n_head, a.hd).permute(2, 0, 3, 1, 4)
    return apply_rope(q, cos, sin), apply_rope(k, cos, sin), v


@torch.no_grad()
def draft_tree_proposals(draft: CSRModel, input_ids, doc_ids, positions, gamma: int):
    """Greedy draft proposals from EVERY prefix in parallel: out[b, s, j-1] = j-th greedy draft token generated from
    the prefix x_0..x_s of the document containing s (j = 1..gamma). Exactly equal (up to floating-point
    reduction order) to running the draft autoregressively from each prefix. Pad positions give arbitrary values.
    Cost ~ gamma draft forwards; no host synchronisation."""
    B, T = input_ids.shape
    cfg = draft.cfg
    hd = cfg.d_model // cfg.n_head
    x = draft.tok_emb(input_ids)
    cos, sin = rope_cos_sin(positions, hd, cfg.rope_base, x.dtype)
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    mask = CSRModel.attention_mask(doc_ids, positions)  # (B, 1, T, T)
    Kr, Vr = [], []
    for blk in draft.blocks:
        q, k, v = _attn_qkv(blk, x, cos, sin)
        Kr.append(k)
        Vr.append(v)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + blk.attn.proj(y.transpose(1, 2).reshape(B, T, -1))
        x = x + blk.ffn(blk.norm2(x))
    cur = draft.lm_head(draft.norm_f(x)).argmax(-1)
    props = [cur]
    Kp = [[] for _ in draft.blocks]
    Vp = [[] for _ in draft.blocks]
    neg = torch.finfo(torch.float32).min
    for j in range(2, gamma + 1):
        x = draft.tok_emb(cur)
        c2, s2 = rope_cos_sin(positions + (j - 1), hd, cfg.rope_base, x.dtype)
        c2, s2 = c2.unsqueeze(1), s2.unsqueeze(1)
        for li, blk in enumerate(draft.blocks):
            q, k, v = _attn_qkv(blk, x, c2, s2)  # (B, H, T, hd): one private query per prefix end s
            Kp[li].append(k)
            Vp[li].append(v)
            scale = 1.0 / math.sqrt(hd)
            sr = torch.matmul(q, Kr[li].transpose(-1, -2)).float() * scale  # (B, H, T, T) vs real prefix keys
            sr = sr.masked_fill(~mask, neg)
            kp = torch.stack(Kp[li], -2)  # (B, H, T, j-1, hd)
            vp = torch.stack(Vp[li], -2)
            sp = (q.unsqueeze(-2) * kp).sum(-1).float() * scale  # (B, H, T, j-1) vs own private keys
            w = torch.softmax(torch.cat([sr, sp], -1), -1)
            wr, wp = w[..., :T], w[..., T:]
            y = torch.matmul(wr.to(Vr[li].dtype), Vr[li]) + (wp.unsqueeze(-1).to(vp.dtype) * vp).sum(-2)
            x = x + blk.attn.proj(y.transpose(1, 2).reshape(B, T, -1))
            x = x + blk.ffn(blk.norm2(x))
        cur = draft.lm_head(draft.norm_f(x)).argmax(-1)
        props.append(cur)
    return torch.stack(props, -1)


def build_draft(dcfg: dict, device, dtype=torch.float32):
    torch.manual_seed(dcfg["init"]["seed"])
    m = CSRModel(ModelConfig.from_dict(dcfg["model"]))
    return m.to(device=device, dtype=dtype)


def load_frozen_draft(path, device, dtype=torch.float32):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = CSRModel(ModelConfig.from_dict(ck["config"]["model"]))
    m.load_state_dict(ck["model"])
    m = m.to(device=device, dtype=dtype).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


# ------------------------------------------------------------------ target model
class PoolRouteModel(CSRModel):
    def __init__(self, cfg: ModelConfig, pr: PoolRouting, extra_seed: int = 0):
        super().__init__(cfg)  # backbone: identical construction + init to TOKEN-4 under the same global seed
        self.pr = pr.validate()
        n_moe, E, d = len(cfg.moe_layers), cfg.n_experts, cfg.d_model
        self.n_moe = n_moe
        fam = pr.family
        if fam != "token":
            assert pr.pool_size >= cfg.top_k
            if pr.pool_assign == "learned":
                self.sel_U = nn.Parameter(torch.zeros(n_moe, E, d))
                if fam == "spec":
                    self.sel_V = nn.Parameter(torch.zeros(n_moe, E, d))
                    self.spec_beta = nn.Parameter(torch.ones(n_moe))
                    self.spec_alpha = nn.Parameter(torch.zeros(n_moe, pr.gamma))
            else:
                gen = torch.Generator().manual_seed(EXTRA_SEED_OFFSET + int(extra_seed))
                self.register_buffer("rand_pool", self._random_table(n_moe, E, pr.pool_size, gen))
            if fam == "adaptive":
                self.det_w = nn.Parameter(torch.zeros(n_moe, d))
                self.det_b = nn.Parameter(torch.zeros(n_moe))
                self.register_buffer("tau", torch.full((n_moe,), float("nan"), dtype=torch.float64))
        self.pool_on = True  # False during the optional restriction warm-up (training only)
        self.drop_seed = None  # int during training if pool dropout is used (deterministic per micro-batch)
        self._calibrating = False
        self._rate_stats = None
        self._draft = None  # frozen draft (not a submodule, never in the state dict)
        self.draft_autocast = True  # on-the-fly proposals in bf16 autocast on CUDA (evaluation); False in decoding

    # -------------------------------------------------------------- bookkeeping
    @staticmethod
    def _random_table(n_moe, E, M, gen):
        assert E % M == 0
        per = E // M
        n_perm = (RAND_TABLE_LEN + per - 1) // per
        tab = torch.zeros(n_moe, RAND_TABLE_LEN, E, dtype=torch.bool)
        for li in range(n_moe):
            seq = torch.cat([torch.randperm(E, generator=gen) for _ in range(n_perm)])
            for p in range(RAND_TABLE_LEN):
                tab[li, p, seq[p * M:(p + 1) * M]] = True
        return tab

    def attach_draft(self, draft):
        object.__setattr__(self, "_draft", draft)

    EXTRA_NAMES = ("sel_U", "sel_V", "spec_beta", "spec_alpha", "det_w", "det_b")

    def extra_parameter_names(self):
        return sorted(n for n, _ in self.named_parameters() if n in self.EXTRA_NAMES)

    def extra_parameter_count(self):
        names = set(self.extra_parameter_names())
        return sum(p.numel() for n, p in self.named_parameters() if n in names)

    def backbone_state(self):
        names = set(self.extra_parameter_names()) | {"tau", "rand_pool"}
        return {k: v for k, v in self.state_dict().items() if k not in names}

    def param_groups(self):
        """Names of parameters with family-specific learning-rate multipliers."""
        sel = [n for n in self.extra_parameter_names()]
        router = [n for n, _ in self.named_parameters() if n.endswith("ffn.router.weight")]
        return {"selector": sel, "router": router}

    # -------------------------------------------------------------- adaptive threshold controller
    def rate_stats_reset(self):
        self._rate_stats = [[0, 0] for _ in range(self.n_moe)] if self.pr.family == "adaptive" else None

    @torch.no_grad()
    def tau_update(self):
        """Between optimizer steps: tau_l += tau_lr * (observed refresh rate - target), step clipped to +-0.05."""
        if self.pr.family != "adaptive" or self._rate_stats is None:
            return None
        out = []
        for li, (nr, nt) in enumerate(self._rate_stats):
            nr, nt = float(nr), float(nt)
            if nt > 0:
                r = nr / nt
                step = max(-0.05, min(0.05, self.pr.tau_lr * (r - self.pr.refresh_target)))
                self.tau[li] = float(self.tau[li]) + step
                out.append(r)
            else:
                out.append(None)
        self.rate_stats_reset()
        return out

    @torch.no_grad()
    def tau_calibrate(self, batch, proposals=None):
        """No-grad pre-pass on the first training micro-batch: layer by layer, bisection sets each NaN tau so the
        refresh rate on that micro-batch matches the target."""
        self._calibrating = True
        was = self.training
        self.eval()
        try:
            self.forward(batch, proposals=proposals)
        finally:
            self._calibrating = False
            self.train(was)
        return self.tau.tolist()

    # -------------------------------------------------------------- pools
    def _selector_logits(self, li, moe, hs, ghat_l):
        """hs (..., d) fp32 detached block-start states; ghat_l (..., d) or None. Returns (..., E) fp32."""
        W = moe.router.weight.detach().to(hs.dtype)
        q = hs @ (W + self.sel_U[li].to(hs.dtype)).t()
        if self.pr.family == "spec":
            ghat_l = ghat_l.to(hs.dtype)
            q = q + self.spec_beta[li].to(hs.dtype) * (ghat_l @ W.t()) + ghat_l @ self.sel_V[li].to(hs.dtype).t()
        return q

    def _ghat(self, li, ehat):
        """ehat (..., gamma, d) rms-normalised detached draft-token embeddings -> (..., d)."""
        a = torch.softmax(self.spec_alpha[li], 0)
        return (ehat * a.view(*([1] * (ehat.dim() - 2)), -1, 1)).sum(-2)

    def _draft_features(self, proposals):
        w = self.tok_emb.weight.detach()
        e = w.to(torch.float64 if w.dtype == torch.float64 else torch.float32)[proposals[..., :self.pr.gamma]]
        return e * torch.rsqrt(e.pow(2).mean(-1, keepdim=True) + 1e-6)

    def _pools(self, li, moe, hs, positions, ghat_l):
        """Candidate pool if a block started at each position: (..., E) bool, plus selector logits (or None)."""
        E, M = moe.E, self.pr.pool_size
        if not self.pool_on:
            return torch.ones(*hs.shape[:-1], E, dtype=torch.bool, device=hs.device), (
                self._selector_logits(li, moe, hs, ghat_l) if self.pr.pool_assign == "learned" else None)
        if self.pr.pool_assign == "random":
            return self.rand_pool[li][positions % RAND_TABLE_LEN], None
        q = self._selector_logits(li, moe, hs, ghat_l)
        qs = q.detach()
        if self.training and self.pr.pool_dropout > 0 and self.drop_seed is not None:
            g = torch.Generator(device=q.device).manual_seed(int(self.drop_seed) * 131 + li)
            drop = torch.rand(qs.shape, generator=g, device=q.device) < self.pr.pool_dropout
            ok = (~drop).sum(-1, keepdim=True) >= M
            qs = torch.where(drop & ok, torch.full_like(qs, float("-inf")), qs)
        idx = qs.topk(M, dim=-1).indices
        return torch.zeros_like(qs, dtype=torch.bool).scatter_(-1, idx, True), q

    # -------------------------------------------------------------- adaptive structure
    def _adaptive_structure(self, li, probs_d, P_all, learned, positions, forced):
        """Returns starts (B, T). probs_d (B,T,E) detached; P_all (B,T,E); learned (B,T) detached score part."""
        H = self.pr.seg_len
        tau = self.tau[li]
        force = (positions == 0) if forced is None else ((positions == 0) | forced)
        Pf = P_all.to(probs_d.dtype)
        if self._calibrating and torch.isnan(tau):
            tau = self._bisect_tau(probs_d, Pf, learned, force, H)
            self.tau[li] = tau
        if torch.isnan(self.tau[li]):
            raise RuntimeError("tau not initialized (run tau_calibrate on a training micro-batch first)")
        return self._adaptive_starts_given_tau(probs_d, Pf, learned, force, H, self.tau[li].to(probs_d.dtype))

    @staticmethod
    def _adaptive_starts_given_tau(probs_d, Pf, learned, force, H, tau):
        B, T, E = probs_d.shape
        cols = []
        for k in range(1, H):
            pk = shift_left(probs_d, k, 0.0)
            miss = 1.0 - (pk * Pf).sum(-1)
            sc = miss + shift_left(learned, k, 0.0)
            cols.append(shift_left(force, k, True) | (sc > tau))
        cols.append(torch.ones(B, T, dtype=torch.bool, device=probs_d.device))
        return orbit_starts(first_true_next(torch.stack(cols, -1)))

    def _bisect_tau(self, probs_d, Pf, learned, force, H):
        lo, hi = -2.0, 3.0
        tgt = self.pr.refresh_target
        n = float(probs_d.shape[0] * probs_d.shape[1])
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            st = self._adaptive_starts_given_tau(probs_d, Pf, learned, force, H,
                                                 torch.tensor(mid, dtype=probs_d.dtype, device=probs_d.device))
            rate = float(st.sum()) / n
            if rate > tgt:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    # -------------------------------------------------------------- one MoE layer
    def _pool_moe(self, li, moe, h, ctx, rec):
        pr = self.pr
        B, T, d = h.shape
        N = B * T
        hf = h.reshape(N, d)
        positions, nonpad = ctx["positions"], ctx["nonpad"]
        E, k = moe.E, moe.k
        z = moe.router_logits(hf)  # (N, E) fp32
        probs = torch.softmax(z / pr.router_temperature, -1)
        with torch.autocast(device_type=h.device.type, enabled=False):
            hs = h.detach().float() if h.dtype != torch.float64 else h.detach()
            ghat_l = self._ghat(li, ctx["ehat"]) if (pr.family == "spec" and pr.pool_assign == "learned") else None
            P_all, q = self._pools(li, moe, hs, positions, ghat_l)  # (B, T, E)
            learned = None
            if pr.family == "fixed":
                starts = fixed_starts(positions, pr.seg_len)
            elif pr.family == "spec":
                starts = ctx["spec_starts"]
            else:
                learned_g = hs @ self.det_w[li].to(hs.dtype) + self.det_b[li].to(hs.dtype)  # (B, T) with grad
                learned = learned_g
                starts = self._adaptive_structure(li, probs.detach().view(B, T, E), P_all, learned_g.detach(),
                                                  positions, ctx.get("forced"))
            bs = block_start_index(starts)
            cand = P_all.gather(1, bs.unsqueeze(-1).expand(B, T, E))
            pv = probs.view(B, T, E)
            _, topi = pv.masked_fill(~cand, -1.0).topk(k, dim=-1)
            gates = pv.gather(-1, topi)
            pool_extra = {}
            if q is not None:
                logm = torch.log_softmax(q / pr.sel_temperature, -1)  # (B, T, E) for every potential start
                logm_tok = logm.gather(1, bs.unsqueeze(-1).expand(B, T, E))
                sel = logm_tok.gather(-1, topi)
                gates = gates * torch.exp(sel - sel.detach())
                pool_extra["logm_tok"] = logm_tok
            if learned is not None:
                prev_bs = torch.cat([torch.zeros_like(bs[:, :1]), bs[:, :-1]], 1)
                ref = torch.where(starts, prev_bs, bs)
                cand_ref = P_all.gather(1, ref.unsqueeze(-1).expand(B, T, E))
                miss = 1.0 - (pv.detach() * cand_ref.to(pv.dtype)).sum(-1)
                score = miss + learned
                xx = (score - self.tau[li].to(score.dtype)) / pr.det_kappa
                force = (positions == 0) | ((torch.arange(T, device=h.device).unsqueeze(0) - prev_bs) >= pr.seg_len)
                if ctx.get("forced") is not None:
                    force = force | ctx["forced"]
                lr_ = F.logsigmoid(xx)
                lnr = F.logsigmoid(-xx)
                fac_log = torch.where(starts & ~force, lr_ - lr_.detach(),
                                      torch.where(~starts, lnr - lnr.detach(), torch.zeros_like(lr_)))
                gates = gates * torch.exp(fac_log).unsqueeze(-1)
                if self.training and self._rate_stats is not None and not self._calibrating:
                    self._rate_stats[li][0] = self._rate_stats[li][0] + (starts & nonpad).sum()  # stays on device
                    self._rate_stats[li][1] = self._rate_stats[li][1] + nonpad.sum()
        topi_f = topi.reshape(N, k)
        y = moe.experts(hf, topi_f, gates.reshape(N, k))
        if moe.shared is not None:
            y = y + moe.shared(hf)
        # auxiliary losses over non-pad tokens
        npm = nonpad.reshape(-1)
        n = npm.sum().clamp(min=1)
        pr_ = probs[npm]
        counts = torch.bincount(topi_f[npm].reshape(-1), minlength=E).to(pr_.dtype)
        aux = E * ((counts / (n * k)) * (pr_.sum(0) / n)).sum()
        zl = torch.logsumexp(z[npm], dim=-1).pow(2).sum() / n
        pool_lb = z.new_zeros(())
        sel_ent = z.new_zeros(())
        if "logm_tok" in pool_extra:
            lmt = pool_extra["logm_tok"].reshape(N, E)[npm]
            mt = lmt.exp()
            cm = cand.reshape(N, E)[npm].to(mt.dtype)
            f = cm.sum(0) / (n * cm.sum(-1).mean().clamp(min=1))
            pool_lb = E * (f * (mt.sum(0) / n)).sum()
            sel_ent = (-(mt * lmt).sum(-1)).sum() / n
        if rec is not None:
            rec.update({"topi": topi, "gates": gates.detach(), "probs": pv.detach(), "starts": starts, "bs": bs,
                        "cand": cand, "z": z.view(B, T, E).detach()})
            if "logm_tok" in pool_extra:
                rec["logm_tok"] = pool_extra["logm_tok"].detach()
        return y.view(B, T, d), aux, zl, pool_lb, sel_ent

    def _token_moe(self, moe, h, nonpad, rec):
        """TOKEN-4 exactly as csr-representative-v1 / CSRModel token mode."""
        B, T, d = h.shape
        hf = h.reshape(B * T, d)
        z = moe.router_logits(hf)
        probs = torch.softmax(z, dim=-1)
        gates, topi = probs.topk(moe.k, dim=-1)
        y = moe.experts(hf, topi, gates)
        if moe.shared is not None:
            y = y + moe.shared(hf)
        npm = nonpad.reshape(-1)
        n = npm.sum().clamp(min=1)
        pr = probs[npm]
        counts = torch.bincount(topi[npm].reshape(-1), minlength=moe.E).to(pr.dtype)
        aux = moe.E * ((counts / (n * moe.k)) * (pr.sum(0) / n)).sum()
        zl = torch.logsumexp(z[npm], dim=-1).pow(2).sum() / n
        if rec is not None:
            rec.update({"topi": topi.view(B, T, -1), "gates": gates.view(B, T, -1),
                        "probs": probs.view(B, T, -1), "z": z.view(B, T, -1)})
        return y.view(B, T, d), aux, zl

    # -------------------------------------------------------------- forward (teacher forcing)
    def prepare_proposals(self, batch, proposals=None):
        if self.pr.family != "spec":
            return None
        if proposals is not None:
            return proposals
        if "proposals" in batch:
            return batch["proposals"]
        if self._draft is None:
            raise RuntimeError("spec family needs draft proposals (cached in the batch or an attached draft)")
        with torch.autocast(device_type=batch["input_ids"].device.type, dtype=torch.bfloat16,
                            enabled=batch["input_ids"].device.type == "cuda" and self.draft_autocast):
            return draft_tree_proposals(self._draft, batch["input_ids"], batch["doc_ids"], batch["positions"],
                                        self.pr.gamma)

    def forward(self, batch, proposals=None, forced=None, collect=False, return_kv=False):
        pr = self.pr
        ids, doc_ids, positions = batch["input_ids"], batch["doc_ids"], batch["positions"]
        B, T = ids.shape
        x = self.tok_emb(ids)
        cos, sin = rope_cos_sin(positions, self.cfg.d_model // self.cfg.n_head, self.cfg.rope_base, x.dtype)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        mask = self.attention_mask(doc_ids, positions)
        nonpad = doc_ids >= 0
        ctx = {"positions": positions, "nonpad": nonpad, "forced": forced}
        if pr.family == "spec":
            props = self.prepare_proposals(batch, proposals)
            ctx["proposals"] = props
            ctx["spec_starts"] = spec_starts(ids, positions, props, pr.seg_len, forced)
            if pr.pool_assign == "learned":
                ctx["ehat"] = self._draft_features(props)
        acc = torch.float64 if x.dtype == torch.float64 else torch.float32
        aux_total = x.new_zeros((), dtype=acc)
        z_total = x.new_zeros((), dtype=acc)
        plb_total = x.new_zeros((), dtype=acc)
        ent_total = x.new_zeros((), dtype=acc)
        records, kv = [], []
        moe_i = 0
        for blk in self.blocks:
            if return_kv:
                q_, k_, v_ = _attn_qkv(blk, x, cos, sin)
                kv.append((k_, v_))
            x = x + blk.attn(blk.norm1(x), cos, sin, mask)
            h = blk.norm2(x)
            if not blk.is_moe:
                x = x + blk.ffn(h)
                continue
            moe = blk.ffn
            rec = {} if collect else None
            if pr.family == "token":
                y, aux, zl = self._token_moe(moe, h, nonpad, rec)
            else:
                y, aux, zl, plb, ent = self._pool_moe(moe_i, moe, h, ctx, rec)
                plb_total = plb_total + plb.to(acc)
                ent_total = ent_total + ent.to(acc)
            x = x + y
            aux_total = aux_total + aux.to(acc)
            z_total = z_total + zl.to(acc)
            if collect:
                records.append(rec)
            moe_i += 1
        logits = self.lm_head(self.norm_f(x))
        out = ForwardOut(logits, aux_total, z_total, records, None)
        out.pool_lb = plb_total
        out.sel_entropy = ent_total
        out.spec_starts = ctx.get("spec_starts")
        out.proposals = ctx.get("proposals")
        out.kv = kv
        return out

    def objective_extras(self, out):
        """Family-specific auxiliary objective (added with the token-weighted micro-batch weight)."""
        pr = self.pr
        if pr.family == "token" or pr.pool_assign == "random":
            return out.aux_loss.new_zeros(())
        return pr.pool_lb_coeff * out.pool_lb - pr.sel_entropy_coeff * out.sel_entropy


def build_pool_model(cfg: dict, device, dtype=torch.float32):
    """Global seed -> backbone init (identical to TOKEN-4 for the same seed); extras zero / dedicated generator."""
    torch.manual_seed(cfg["init"]["seed"])
    pr = PoolRouting.from_cfg(cfg["routing"])
    m = PoolRouteModel(ModelConfig.from_dict(cfg["model"]), pr, extra_seed=cfg["init"]["seed"])
    return m.to(device=device, dtype=dtype)
