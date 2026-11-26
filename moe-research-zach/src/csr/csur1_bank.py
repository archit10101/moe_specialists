"""csr-crossfit-semantic-utility-residual-v1: cross-fitted semantic-utility residual-specialist banks.

All banks act on the DETACHED post-norm base hidden h and the detached tied head W (the base never receives a gradient
from anything here):  z = sg(z_base) + sg(W) a,  a = Expert_e(rms0(h)),  exactly one specialist per token.

UtilBank (conditions SU semantic-utility, SH shuffled-prior utility, SO semantic-only, UO utility-only):
  blocks  = 128-token views of a document (positions [128 j, 128 j + 128)); region = blocks (2i, 2i+1) = views A, B.
  utility u[v, k] = mean over the utility tokens of view v of CE(base) - CE(base + EMA-specialist k)  (exact, all k;
                    training: scored tokens with (pos + step) % USTRIDE == 0, a rotating 1-in-4 subset; eval: all)
  partner(v)      = the other view of v's region if it has >= MIN_PARTNER scored tokens, else the previous view of the
                    same document if it has; else none (u term 0).  Utilities of v NEVER weight v's own training.
  responsibility  r[v] = softmax( uc[partner(v)] / tau_t + eta log(prior[v] + eps) - lam log(K m) )
                  uc = u - ubar (RELATIVE utility: each specialist's utility centred by its token-weighted mean over the
                  step's views; absolute utility rewards the generically best specialist and collapses, GPU check)
                  tau_t = c * s_t, s_t = the step's mean within-view across-specialist SD of uc (lag-free, scale-free c);
                  eval uses EMA (rho 0.9) of training ubar and s_t
                  prior = p_sem of v's region (SU, SO), the persistently relabelled p_SH (SH), none (UO);
                  SO drops the u term, UO drops the prior term; m = EMA corpus-level token mass of r (mild corpus prior,
                  no per-minibatch balance).
  training        k_v ~ r[v] (one sample per view; stop-gradient through r), live specialist k_v trained on view v.
  student router  q[v] = softmax(head(GRU state after the completed views 0..v-1 of the same document)); the state
                  input is Linear(mean rms0(h) over the view); view 0 uses the zero state. Loss KL(sg(r[v]) || q[v]).
  inference       student: e(v) = argmax q[v] (deployable, causal); teacher: e(v) = argmax r[v] (training-only oracle:
                  uses the partner view's utility, possibly in the future, and the region's semantic prior).
TokenBank G = jsr_bank.Bank with a per-token linear top-1 router and Switch balance (orv1 C / jsr B)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from csr.csur1_paths import (BLOCK, D_MODEL, EXPERT_FFN, GRU_DIN, GRU_H, INIT_SEED_OFFSET, INIT_STD, K, MIN_PARTNER,
                             PSEM_EPS, SCALE_RHO, USE_RHO, USTRIDE)
from csr.jsr_bank import head_ce_grad, rms0  # noqa: F401  (re-exported)
from csr.lfr_model import Experts

UCHUNK = 512


def _ac(dev):
    return torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")


# --------------------------------------------------------------------------------------------- block layout
def layout(row, doc_local, pos, scored):
    """Token-level inputs (all doc tokens of the step, flattened; pads removed):
    row, doc_local, pos (N,) long; scored (N,) bool (has a target).
    Returns dict: blk (N,) block index per token; nb; blk_doc (nb,) doc index; blk_no (nb,); n_doc; n_scored (nb,);
    partner (nb,) (-1 = none); region_no (nb,)."""
    dev = row.device
    dkey = row * 4096 + doc_local
    key = dkey * 64 + torch.div(pos, BLOCK, rounding_mode="floor")
    ukey, blk = torch.unique(key, sorted=True, return_inverse=True)
    nb = ukey.numel()
    bdkey = torch.div(ukey, 64, rounding_mode="floor")
    blk_no = ukey - bdkey * 64
    udoc, blk_doc = torch.unique(bdkey, sorted=True, return_inverse=True)
    n_scored = torch.zeros(nb, device=dev, dtype=torch.long).index_add_(0, blk, scored.long())
    idx = torch.arange(nb, device=dev)
    same_next = torch.zeros(nb, dtype=torch.bool, device=dev)
    same_next[:-1] = blk_doc[1:] == blk_doc[:-1]
    same_prev = torch.zeros(nb, dtype=torch.bool, device=dev)
    same_prev[1:] = blk_doc[1:] == blk_doc[:-1]
    ok = n_scored >= MIN_PARTNER
    even = (blk_no % 2) == 0
    nxt = torch.where(same_next, idx + 1, idx)
    prv = torch.where(same_prev, idx - 1, idx)
    cand1 = torch.where(even, nxt, prv)          # the other view of the region
    good1 = torch.where(even, same_next, same_prev) & ok[cand1]
    good_prev = same_prev & ok[prv]               # fallback: previous view of the document
    partner = torch.where(good1, cand1, torch.where(good_prev, prv, torch.full_like(idx, -1)))
    return {"blk": blk, "nb": nb, "blk_doc": blk_doc, "blk_no": blk_no, "n_doc": udoc.numel(), "n_scored": n_scored,
            "partner": partner, "region_no": torch.div(blk_no, 2, rounding_mode="floor")}


def block_mean(x, blk, nb):
    s = torch.zeros(nb, x.shape[1], device=x.device, dtype=torch.float32).index_add_(0, blk, x.float())
    c = torch.zeros(nb, device=x.device, dtype=torch.float32).index_add_(0, blk, torch.ones_like(blk, dtype=torch.float32))
    return s / c.clamp(min=1).unsqueeze(1)


def block_feats(x, blk, nb):
    """[mean, last token] of rms0(h) per block, (nb, 2d). Columns [0, d) are exactly block_mean (the gen-0 router input,
    sliced by student_logits); columns [d, 2d) the last token of the block (engineering generation 1, prereg section 12
    trigger 5)."""
    m = block_mean(x, blk, nb)
    idx = torch.arange(blk.shape[0], device=blk.device)
    last = torch.full((nb,), -1, device=blk.device, dtype=idx.dtype).scatter_reduce(0, blk, idx, "amax", include_self=True)
    return torch.cat([m, x[last.clamp(min=0)].float()], 1)


@torch.no_grad()
def all_deltas(w1, w2, h, xn, W, y):
    """Exact equal-compute comparison for every specialist k of an Experts bank (w1 (K,d,f), w2 (K,f,d)):
    d[t, k] = CE(W h_t) - CE(W (h_t + a_k(xn_t))) (N, K) fp32, and CE_base (N,). Base logits are computed once (bf16
    matmul, fp32 softmax) and shared, the correction logits W a_k separately (small magnitude), so the base-logit
    rounding cancels to first order in d."""
    N = h.shape[0]
    d = torch.empty(N, K, device=h.device, dtype=torch.float32)
    ceb = torch.empty(N, device=h.device, dtype=torch.float32)
    for s in range(0, N, UCHUNK):
        hh, xx, yy = h[s:s + UCHUNK], xn[s:s + UCHUNK], y[s:s + UCHUNK]
        with _ac(h.device):
            zb = F.linear(hh, W).float()
            hid = F.gelu(torch.einsum("nd,kdf->knf", xx, w1), approximate="tanh")
            a = torch.einsum("knf,kfd->knd", hid, w2)
            dl = torch.einsum("knd,vd->knv", a, W)
        lse_b = torch.logsumexp(zb, -1)
        lse_k = torch.logsumexp(zb.unsqueeze(0) + dl.float(), -1)              # (K, n)
        dy = dl.float().gather(2, yy.view(1, -1, 1).expand(K, -1, 1)).squeeze(2)  # (K, n)
        ceb[s:s + UCHUNK] = lse_b - zb.gather(1, yy.view(-1, 1)).squeeze(1)
        d[s:s + UCHUNK] = (lse_b.unsqueeze(0) - lse_k + dy).t()               # CE_base - CE_k
        del zb, hid, a, dl
    return d, ceb


@torch.no_grad()
def ce_pair(h, a, W, y):
    """(CE_base, CE(base + a)) per token on the SAME numerical path as all_deltas (shared base logits, correction logits
    W a separately, fp32 log-sum-exp), so routed and forced-specialist CE are directly comparable. a may be None."""
    N = h.shape[0]
    ceb = torch.empty(N, device=h.device, dtype=torch.float32)
    cex = torch.empty(N, device=h.device, dtype=torch.float32)
    for s in range(0, N, 4 * UCHUNK):
        hh, yy = h[s:s + 4 * UCHUNK], y[s:s + 4 * UCHUNK]
        with _ac(h.device):
            zb = F.linear(hh, W).float()
            dl = F.linear(a[s:s + 4 * UCHUNK], W).float() if a is not None else None
        lse_b = torch.logsumexp(zb, -1)
        zy = zb.gather(1, yy.view(-1, 1)).squeeze(1)
        ceb[s:s + 4 * UCHUNK] = lse_b - zy
        if dl is None:
            cex[s:s + 4 * UCHUNK] = lse_b - zy
        else:
            cex[s:s + 4 * UCHUNK] = torch.logsumexp(zb + dl, -1) - zy - dl.gather(1, yy.view(-1, 1)).squeeze(1)
        del zb, dl
    return ceb, cex


# --------------------------------------------------------------------------------------------- banks
class UtilBank(nn.Module):
    """spec: name, prior ('sem'|'shuf'|None), util (bool), c, eta, ema, lam."""

    def __init__(self, spec, seed):
        super().__init__()
        self.spec = dict(spec)
        self.name = spec["name"]
        self.prior = spec.get("prior")
        self.util = bool(spec.get("util", True))
        self.c = float(spec.get("c", 1.0))
        self.eta = float(spec.get("eta", 0.0)) if self.prior else 0.0
        self.ema_d = float(spec.get("ema", 0.99))
        self.lam = float(spec.get("lam", 0.0))
        self.experts = Experts(K, D_MODEL, EXPERT_FFN)
        g = torch.Generator().manual_seed(seed + INIT_SEED_OFFSET)
        with torch.no_grad():
            self.experts.w1.copy_(torch.randn(self.experts.w1.shape, generator=g) * INIT_STD)
            self.experts.w2.zero_()
        gr = torch.Generator().manual_seed(seed + INIT_SEED_OFFSET + 1)
        # router 'mean' (gen 0): Linear(384->32) -> GRUCell(32,32) -> Linear(32->16), 19,152 params.
        # router 'meanlast' (gen 1, prereg section 12 trigger 5): [mean, last] Linear(768->20) -> GRUCell(20,20) -> Linear(20->16), 18,216.
        self.router_kind = spec.get("router", "mean")
        din, dh, dx = (GRU_DIN, GRU_H, D_MODEL) if self.router_kind == "mean" else (20, 20, 2 * D_MODEL)
        self.inp = nn.Linear(dx, din, bias=False)
        self.gru = nn.GRUCell(din, dh)
        self.head = nn.Linear(dh, K)
        with torch.no_grad():
            for p in (self.inp.weight, self.gru.weight_ih, self.gru.weight_hh, self.head.weight):
                p.copy_(torch.randn(p.shape, generator=gr) * (1.0 / math.sqrt(p.shape[1])))
            self.gru.bias_ih.zero_()
            self.gru.bias_hh.zero_()
            self.head.bias.zero_()
        self.register_buffer("ema_w1", self.experts.w1.detach().clone())
        self.register_buffer("ema_w2", self.experts.w2.detach().clone())
        self.register_buffer("m", torch.full((K,), 1.0 / K))
        self.register_buffer("scale", torch.zeros(()))
        self.register_buffer("ubar", torch.zeros(K))
        self.register_buffer("n_upd", torch.zeros((), dtype=torch.long))

    def n_params(self):
        r = sum(p.numel() for m in (self.inp, self.gru, self.head) for p in m.parameters())
        return {"experts": self.experts.w1.numel() + self.experts.w2.numel(), "router": r}

    def router_params(self):
        return [p for m in (self.inp, self.gru, self.head) for p in m.parameters()]

    def expert_params(self):
        return [self.experts.w1, self.experts.w2]

    # ---- utilities (EMA specialists, exact equal-compute CE comparison, all K)
    @torch.no_grad()
    def token_deltas(self, h, xn, W, y, use_ema=True):
        """d[t, k] = CE_base(t) - CE(base + specialist k)(t) for every token (N, K) fp32; also CE_base (N,)."""
        return all_deltas(self.ema_w1 if use_ema else self.experts.w1, self.ema_w2 if use_ema else self.experts.w2,
                          h, xn, W, y)

    def block_utility(self, d, blk_s, nb):
        """u (nb, K): mean over the scored tokens of each block; blocks without scored tokens get 0."""
        s = torch.zeros(nb, K, device=d.device, dtype=torch.float32).index_add_(0, blk_s, d)
        c = torch.zeros(nb, device=d.device, dtype=torch.float32).index_add_(0, blk_s, torch.ones_like(blk_s, dtype=torch.float32))
        return s / c.clamp(min=1).unsqueeze(1)

    def responsibilities(self, u, partner, logprior, update=False, weights=None, u_w=None):
        """r (nb, K) and diagnostics. u (nb, K) block utilities (None for SO), u_w (nb,) utility-token counts,
        partner (nb,), logprior (nb, K) or None. update=True (training): centring and tau from this step (and the EMAs
        are updated); update=False (eval): the stored training EMAs are used."""
        nb = partner.shape[0]
        dev = partner.device
        logit = torch.zeros(nb, K, device=dev)
        has = partner >= 0
        stats = {}
        ut = None
        if self.util and u is not None:
            if update:
                w = (u_w if u_w is not None else torch.ones(nb, device=dev)).float()
                ubar = (u * w.unsqueeze(1)).sum(0) / w.sum().clamp(min=1)
                uc = u - ubar
                okv = w > 0
                sd = uc[okv].std(1).mean() if bool(okv.any()) else torch.zeros((), device=dev)
                first = int(self.n_upd) == 0
                self.ubar.copy_(ubar if first else SCALE_RHO * self.ubar + (1 - SCALE_RHO) * ubar)
                self.scale.copy_(sd if first else SCALE_RHO * self.scale + (1 - SCALE_RHO) * sd)
            else:
                uc = u - self.ubar
                sd = self.scale
            tau = self.c * sd.clamp(min=1e-6)
            ut = torch.where(has.unsqueeze(1), uc[partner.clamp(min=0)] / tau, torch.zeros_like(uc))
            logit = logit + ut
            stats.update({"u_sd": float(sd), "tau": float(tau), "ubar_spread": float(self.ubar.max() - self.ubar.min())})
        if self.prior is not None and logprior is not None:
            logit = logit + self.eta * logprior
        if self.lam > 0:
            logit = logit - self.lam * torch.log(self.m * K).unsqueeze(0)
        r = torch.softmax(logit, -1)
        if update:
            w = weights.float() if weights is not None else torch.ones(nb, device=dev)
            mass = (r * w.unsqueeze(1)).sum(0) / w.sum().clamp(min=1)
            self.m.copy_(USE_RHO * self.m + (1 - USE_RHO) * mass)
            self.n_upd += 1
        am = r.argmax(1)
        stats.update({"r_ent": float(-(r * torch.log(r.clamp(min=1e-12))).sum(1).mean()), "r_max": float(r.max(1).values.mean())})
        if ut is not None:
            stats["agree_util"] = float((am[has] == ut[has].argmax(1)).float().mean()) if bool(has.any()) else float("nan")
        if self.prior is not None and logprior is not None:
            stats["agree_prior"] = float((am == logprior.argmax(1)).float().mean())
        return r, stats

    # ---- student router
    def student_logits(self, s_blk, lay):
        """s_blk (nb, d) mean rms0(h) per block (detached). Returns logits (nb, K) from the causal GRU state."""
        nb, n_doc = lay["nb"], lay["n_doc"]
        bd, bn = lay["blk_doc"], lay["blk_no"]
        maxb = int(bn.max()) + 1
        x = self.inp(s_blk[:, :self.inp.in_features].float())
        X = torch.zeros(n_doc, maxb, self.gru.input_size, device=x.device, dtype=x.dtype)
        X[bd, bn] = x
        st = torch.zeros(n_doc, self.gru.hidden_size, device=x.device, dtype=x.dtype)
        states = [st]
        for j in range(maxb - 1):
            st = self.gru(X[:, j], st)
            states.append(st)
        S = torch.stack(states, 1)              # S[:, j] = state after views 0..j-1
        return self.head(S[bd, bn])

    def correction(self, xn, e):
        with _ac(xn.device):
            a = self.experts(xn, e.unsqueeze(1), torch.ones(e.shape[0], 1, device=xn.device))
        return a.float()

    @torch.no_grad()
    def forced(self, xn, k):
        return self.correction(xn, torch.full((xn.shape[0],), k, dtype=torch.long, device=xn.device))

    @torch.no_grad()
    def ema_update(self):
        self.ema_w1.mul_(self.ema_d).add_(self.experts.w1.detach(), alpha=1 - self.ema_d)
        self.ema_w2.mul_(self.ema_d).add_(self.experts.w2.detach(), alpha=1 - self.ema_d)


def util_step(bank, step_gen, h_s, xn_s, W, y, lay, blk_s, s_blk, logprior, umask=None):
    """One residual-phase update of a UtilBank's gradients (no optimizer step).
    h_s, xn_s, y: scored tokens (N_s, d); blk_s (N_s,) block of each scored token; s_blk (nb, d) block summaries over
    all doc tokens; logprior (nb, K) log(prior + eps) or None. Returns diagnostics."""
    nb = lay["nb"]
    rec = {}
    u = None
    u_w = None
    if bank.util:
        um = torch.ones_like(y, dtype=torch.bool) if umask is None else umask
        d, _ = bank.token_deltas(h_s[um], xn_s[um], W, y[um], use_ema=True)
        u = bank.block_utility(d, blk_s[um], nb)
        u_w = torch.zeros(nb, device=y.device).index_add_(0, blk_s[um], torch.ones_like(y[um], dtype=torch.float32))
        rec.update(reliability(u, lay, u_w))
        rec["u_mean"] = float(u[u_w > 0].mean())
        del d
    r, st = bank.responsibilities(u, lay["partner"], logprior, update=True, weights=lay["n_scored"], u_w=u_w)
    rec.update(st)
    k_blk = torch.multinomial(r, 1, generator=step_gen).squeeze(1)
    e = k_blk[blk_s]
    a = bank.correction(xn_s, e)
    ce, ga = head_ce_grad(h_s, a.detach(), W, y)
    N = h_s.shape[0]
    a.backward(ga / N)
    lg = bank.student_logits(s_blk, lay)
    logq = torch.log_softmax(lg.float(), -1)
    wb = (lay["n_scored"] > 0).float()
    kl = ((r * (torch.log(r.clamp(min=1e-12)) - logq)).sum(1) * wb).sum() / wb.sum().clamp(min=1)
    kl.backward()
    f = torch.bincount(e, minlength=K).float() / N
    q = logq.exp()
    eq = q.argmax(1)[blk_s]
    fq = torch.bincount(eq, minlength=K).float() / N
    rec.update({"res": float(ce.mean()), "kl": float(kl.detach()), "acc_q_r": float((q.argmax(1) == r.argmax(1))[wb > 0].float().mean()),
                "q_ent": float(-(q * logq).sum(1).mean()), "load_max": float(f.max()), "load_eff": _eff(f),
                "n_dead": int((f < 1.0 / (10 * K)).sum()), "sload_max": float(fq.max()), "sload_eff": _eff(fq),
                "s_dead": int((fq < 1.0 / (10 * K)).sum()), "m_max": float(bank.m.max()), "m_min": float(bank.m.min()),
                "anorm": float(a.detach().norm(dim=1).mean())})
    return rec


def _eff(f):
    f = f[f > 0]
    return float(torch.exp(-(f * torch.log(f)).sum()))


@torch.no_grad()
def reliability(u, lay, u_w=None):
    """Split-half reliability of block utility on regions with both views scored (>= MIN_PARTNER scored tokens and >= 1
    utility token each): Pearson r of the double-centred (block and specialist means removed) utilities of view A vs
    view B, plus argmax agreement of the specialist-centred utilities."""
    bn, bd, ok = lay["blk_no"], lay["blk_doc"], lay["n_scored"] >= MIN_PARTNER
    if u_w is not None:
        ok = ok & (u_w > 0)
    nb = lay["nb"]
    idx = torch.arange(nb - 1, device=u.device)
    pair = ((bn[:-1] % 2) == 0) & (bd[1:] == bd[:-1]) & (bn[1:] == bn[:-1] + 1) & ok[:-1] & ok[1:]
    ia = idx[pair]
    ib = ia + 1
    if ia.numel() < 8:
        return {"rel": float("nan"), "rel_argmax": float("nan"), "n_pairs": int(ia.numel())}
    A, B = u[ia], u[ib]
    colm = 0.5 * (A.mean(0) + B.mean(0))
    A = A - A.mean(1, keepdim=True) - colm + colm.mean()
    B = B - B.mean(1, keepdim=True) - colm + colm.mean()
    a, b = A.flatten(), B.flatten()
    rr = float(((a - a.mean()) * (b - b.mean())).sum() / ((a - a.mean()).norm() * (b - b.mean()).norm()).clamp(min=1e-12))
    ca, cb = u[ia] - colm, u[ib] - colm
    return {"rel": rr, "rel_argmax": float((ca.argmax(1) == cb.argmax(1)).float().mean()), "n_pairs": int(ia.numel())}


def util_bank_optimizer(bank, t, device):
    return torch.optim.AdamW([p for p in bank.parameters()], lr=t["lr"], betas=tuple(t["betas"]), eps=1e-8,
                             weight_decay=t["weight_decay"], fused=device.type == "cuda")


def finite(rec):
    return math.isfinite(rec["res"]) and not any(isinstance(v, float) and math.isinf(v) for v in rec.values())
