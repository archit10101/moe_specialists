"""joint-semantic-residual-pretrain-v1: residual-specialist banks trained on the DETACHED base prediction.

For post-norm base hidden h (N, d) and tied LM head W (both detached):
    q = softmax(R rms0(h) / tau); e = argmax q; gate = q_e / sg(q_e)   (routed banks B, C, D; value 1, grad = dlog q_e)
    e = paper semantic group, gate = 1                                    (oracle bank E; unlabeled tokens get a = 0)
    a = Expert_e(rms0(h)) * gate        Experts(16, 384, 256) = csr.lfr_model.Experts (the repo's routed-expert module)
    z = W (h + a) = sg(z_base) + sg(W) a
L_res = mean CE(z, y); L_bal = K sum_e f_e mean(q_e) (router only); L_sd = mean_{a<b} -log(JSD(R_a, R_b) + eps) over
groups with >= SD_MIN_TOKENS tokens in the step (router only). Router gradient = g_res + alpha g_bal + beta_t g_sd with
beta_t = r * EMA||g_res|| / EMA||g_sd||. The base never receives a gradient from anything here (h and W are detached
and the CE gradient w.r.t. (h + a) is formed explicitly, chunk by chunk, without a graph through W)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from csr.jsr_paths import (D_MODEL, EMA_RHO, EXPERT_FFN, INIT_SEED_OFFSET, INIT_STD, K, SD_EPS, SD_MIN_TOKENS)
from csr.lfr_model import Experts

EPS = 1e-6
CHUNK = 8192


def rms0(x):
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)


def _ac(dev):
    return torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")


def head_ce_grad(h, a, W, y, want_grad=True, chunk=CHUNK):
    """Per-token CE of logits W (h + a) (bf16 matmul under CUDA autocast, fp32 softmax) and, if want_grad, the
    gradient of SUM CE w.r.t. a (== w.r.t. h + a), formed explicitly as (softmax - onehot) W. No autograd through W."""
    ce = torch.empty(h.shape[0], device=h.device, dtype=torch.float32)
    g = torch.empty_like(a, dtype=torch.float32) if want_grad else None
    for s in range(0, h.shape[0], chunk):
        x = h[s:s + chunk].float() + a[s:s + chunk].float()
        with _ac(h.device):
            lg = F.linear(x, W)
        lg = lg.float()
        lse = torch.logsumexp(lg, -1)
        yy = y[s:s + chunk]
        ce[s:s + chunk] = lse - lg.gather(1, yy.view(-1, 1)).squeeze(1)
        if want_grad:
            p = torch.exp(lg - lse.unsqueeze(1))
            p.scatter_add_(1, yy.view(-1, 1), -torch.ones_like(yy, dtype=p.dtype).view(-1, 1))
            with _ac(h.device):
                gg = F.linear(p, W.t())
            g[s:s + chunk] = gg.float()
        del lg
    return ce, g


def jsd_pairs(R):
    """R (G, K) distributions -> (G*(G-1)/2,) pairwise Jensen-Shannon divergences (nats)."""
    G = R.shape[0]
    ia, ib = torch.triu_indices(G, G, 1, device=R.device)
    P, Q = R[ia], R[ib]
    M = 0.5 * (P + Q)
    lp, lq, lm = (torch.log(x.clamp(min=1e-12)) for x in (P, Q, M))
    return 0.5 * (P * (lp - lm)).sum(-1) + 0.5 * (Q * (lq - lm)).sum(-1)


def sd_loss(q, grp):
    """Label-free semantic-divergence loss on router distributions q (N, K) with group ids grp (N,) (-1 = none).
    Returns (loss (0-d, differentiable in q) or None, mean JSD (float), n groups present)."""
    ok = grp >= 0
    if int(ok.sum()) == 0:
        return None, float("nan"), 0
    g = grp[ok]
    cnt = torch.bincount(g, minlength=K)
    present = torch.nonzero(cnt >= SD_MIN_TOKENS).squeeze(1)
    if present.numel() < 2:
        return None, float("nan"), int(present.numel())
    S = torch.zeros(K, q.shape[1], device=q.device, dtype=torch.float64).index_add(0, g, q[ok].double())
    R = S[present] / cnt[present].unsqueeze(1).double()
    j = jsd_pairs(R)  # float64: JSD of near-uniform group means is ~1e-4 early; fp32 would cost ~1e-4 relative
    return (-torch.log(j + SD_EPS)).mean().float(), float(j.mean()), int(present.numel())


class Bank(nn.Module):
    """One residual-specialist bank. spec: name, kind ('routed'|'oracle'), tau, r, groups ('sem'|'shuf'|None)."""

    def __init__(self, spec, seed):
        super().__init__()
        self.spec = dict(spec)
        self.name = spec["name"]
        self.routed = spec["kind"] == "routed"
        self.tau = float(spec.get("tau", 1.0))
        self.r = float(spec.get("r", 0.0))
        self.alpha = float(spec.get("alpha", 0.0))
        self.groups = spec.get("groups")
        self.experts = Experts(K, D_MODEL, EXPERT_FFN)
        g = torch.Generator().manual_seed(seed + INIT_SEED_OFFSET)
        with torch.no_grad():
            self.experts.w1.copy_(torch.randn(self.experts.w1.shape, generator=g) * INIT_STD)
            self.experts.w2.zero_()
        self.router = None
        if self.routed:
            self.router = nn.Linear(D_MODEL, K, bias=False)
            gr = torch.Generator().manual_seed(seed + INIT_SEED_OFFSET + 1)
            with torch.no_grad():
                self.router.weight.copy_(torch.randn(self.router.weight.shape, generator=gr) * INIT_STD)
        self.ctrl = {"ema_res": None, "ema_sd": None, "beta": 0.0}

    def n_params(self):
        return {"experts": self.experts.w1.numel() + self.experts.w2.numel(),
                "router": 0 if self.router is None else self.router.weight.numel()}

    def probs(self, xn):
        with torch.autocast(device_type=xn.device.type, enabled=False):
            return torch.softmax(F.linear(xn.float(), self.router.weight.float()) / self.tau, -1)

    def route(self, xn, oracle_lab=None):
        """-> e (N,), gate (N,) (differentiable for routed banks), q (N, K) or None, active mask (N,) bool."""
        if self.routed:
            q = self.probs(xn)
            e = q.argmax(-1)
            qe = q.gather(1, e.unsqueeze(1)).squeeze(1)
            return e, qe / qe.detach(), q, torch.ones_like(e, dtype=torch.bool)
        act = oracle_lab >= 0
        e = oracle_lab.clamp(min=0)
        return e, act.float(), None, act

    def correction(self, xn, e, gate):
        with _ac(xn.device):
            a = self.experts(xn, e.unsqueeze(1), gate.unsqueeze(1))
        return a.float()

    @torch.no_grad()
    def forced(self, xn, k):
        """Specialist k's correction for every token (N, d) fp32 (same dispatch kernel)."""
        e = torch.full((xn.shape[0],), k, dtype=torch.long, device=xn.device)
        return self.correction(xn, e, torch.ones(xn.shape[0], device=xn.device))

    def ema(self, key, v):
        o = self.ctrl[key]
        self.ctrl[key] = v if o is None else EMA_RHO * o + (1 - EMA_RHO) * v


def residual_step(bank, h, W, y, sem, shuf, xn=None):
    """One residual-phase update of a bank's gradients (no optimizer step) on the detached step tokens.
    h (N, d) detached post-norm hidden of valid tokens; W detached tied head; y (N,) targets; sem/shuf (N,) group ids.
    Leaves .grad on bank parameters; returns a diagnostics dict."""
    if xn is None:
        xn = rms0(h)
    N = h.shape[0]
    e, gate, q, act = bank.route(xn, sem if not bank.routed else None)
    a = bank.correction(xn, e, gate)
    ce, ga = head_ce_grad(h, a.detach(), W, y)
    a.backward(ga / N, retain_graph=bank.routed)
    rec = {"res": float(ce.mean())}
    if bank.routed:
        Wr = bank.router.weight
        g_res = Wr.grad.detach().clone() if Wr.grad is not None else torch.zeros_like(Wr)
        f = torch.bincount(e, minlength=K).float() / N
        P = q.mean(0)
        l_bal = K * (f * P).sum()
        grp = {"sem": sem, "shuf": shuf}.get(bank.groups)
        lsd, jm, npres = (sd_loss(q, grp) if (grp is not None and bank.r > 0) else (None, float("nan"), 0))
        outs = [l_bal] + ([lsd] if lsd is not None else [])
        gr = torch.autograd.grad(outs, [Wr] * len(outs), allow_unused=True)
        g_bal = gr[0]
        n_res = float(g_res.norm())
        bank.ema("ema_res", n_res)
        beta, n_sd = 0.0, 0.0
        g_tot = g_res + bank.alpha * g_bal
        if lsd is not None:
            g_sd = gr[1]
            n_sd = float(g_sd.norm())
            bank.ema("ema_sd", n_sd)
            if bank.ctrl["ema_sd"] and bank.ctrl["ema_sd"] > 0:
                beta = bank.r * bank.ctrl["ema_res"] / bank.ctrl["ema_sd"]
            g_tot = g_tot + beta * g_sd
            rec.update({"sd": float(lsd.detach()), "jsd": jm, "n_groups": npres})
        bank.ctrl["beta"] = beta
        Wr.grad = g_tot
        load = f
        rec.update({"bal": float(l_bal.detach()), "beta": beta, "gn_res_router": n_res, "gn_sd_router": n_sd,
                    "ratio": (beta * n_sd / n_res) if n_res > 0 else float("nan"),
                    "load_max": float(load.max()), "load_min": float(load.min()),
                    "load_eff": float(torch.exp(-(load[load > 0] * torch.log(load[load > 0])).sum())),
                    "n_dead": int((load < 1.0 / (10 * K)).sum()), "q_ent": float(-(q * torch.log(q.clamp(min=1e-12))).sum(1).mean()),
                    "anorm": float(a.detach().norm(dim=1).mean())})
    else:
        rec.update({"active": float(act.float().mean()), "anorm": float(a.detach().norm(dim=1).mean())})
    return rec


def bank_optimizer(bank, t, device):
    return torch.optim.AdamW([p for p in bank.parameters()], lr=t["lr"], betas=tuple(t["betas"]), eps=1e-8,
                             weight_decay=t["weight_decay"], fused=device.type == "cuda")


def finite(rec):
    """res must be finite; other floats may be nan (no groups present) but never +-inf."""
    return math.isfinite(rec["res"]) and not any(isinstance(v, float) and math.isinf(v) for v in rec.values())
