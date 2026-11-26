"""asrs2 K-generic residual-specialist machinery (K = 5; user amendment). New code; imports the repo's Experts module,
head_ce_grad and rms0 unchanged.

Bank5      K specialists 384 -> 256 -> 384 (lfr_model.Experts), h' = h + a_e(rms0(h)); optional token router (generic
           banks G / ALT-G): Linear(384, K), q = softmax, e = argmax q, gate = q_e / sg(q_e) (value 1).
spec_grad  target-free assigned specialist update: CE(sg(W)(sg(h) + a_e(sg(xn)))) with e GIVEN (never chosen from y).
gen_grad   token-routed generic update (jsr residual_step without the semantic-divergence term): experts + router
           (LM-through-gate + alpha * Switch balance); callers drop the router or expert grads for ALT-G windows.
deltas     exact equal-compute CE_base - CE_k for all K specialists (copy of csur1 all_deltas, K from the weights).
r1_loss    router distillation KL(sg(pi) || q) + lam * balance, pi = the frozen region prior p5 (R1, target-free)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from csr.jsr_bank import head_ce_grad, rms0  # noqa: F401
from csr.lfr_model import Experts

INIT_STD = 0.02
INIT_SEED_OFFSET = 7_000_003
UCHUNK = 1024


def _ac(dev):
    return torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")


class Bank5(nn.Module):
    def __init__(self, K, d, f, seed, routed=False, alpha=0.01):
        super().__init__()
        self.K, self.routed, self.alpha = K, routed, alpha
        self.experts = Experts(K, d, f)
        g = torch.Generator().manual_seed(int(seed) + INIT_SEED_OFFSET)
        with torch.no_grad():
            self.experts.w1.copy_(torch.randn(self.experts.w1.shape, generator=g) * INIT_STD)
            self.experts.w2.zero_()
        self.router = None
        if routed:
            self.router = nn.Linear(d, K, bias=False)
            gr = torch.Generator().manual_seed(int(seed) + INIT_SEED_OFFSET + 1)
            with torch.no_grad():
                self.router.weight.copy_(torch.randn(self.router.weight.shape, generator=gr) * INIT_STD)

    def n_params(self):
        return {"experts": self.experts.w1.numel() + self.experts.w2.numel(),
                "router": 0 if self.router is None else self.router.weight.numel()}

    def expert_params(self):
        return [self.experts.w1, self.experts.w2]

    def probs(self, xn):
        with torch.autocast(device_type=xn.device.type, enabled=False):
            return torch.softmax(F.linear(xn.float(), self.router.weight.float()), -1)

    def correction(self, xn, e, gate=None):
        if gate is None:
            gate = torch.ones(xn.shape[0], device=xn.device)
        with _ac(xn.device):
            a = self.experts(xn, e.unsqueeze(1), gate.unsqueeze(1))
        return a.float()


def spec_grad(bank, h, xn, W, y, e):
    """Leaves expert grads for the target-free assignment e (N,); returns per-token CE (detached, before update)."""
    N = h.shape[0]
    a = bank.correction(xn, e)
    ce, ga = head_ce_grad(h, a.detach(), W, y)
    a.backward(ga / N)
    return ce


def gen_grad(bank, h, xn, W, y):
    """Token-routed generic bank: experts + router gradients (router: LM-through-gate + alpha * Switch balance)."""
    N = h.shape[0]
    q = bank.probs(xn)
    e = q.argmax(-1)
    qe = q.gather(1, e.unsqueeze(1)).squeeze(1)
    a = bank.correction(xn, e, qe / qe.detach())
    ce, ga = head_ce_grad(h, a.detach(), W, y)
    a.backward(ga / N, retain_graph=True)
    Wr = bank.router.weight
    g_res = Wr.grad.detach().clone() if Wr.grad is not None else torch.zeros_like(Wr)
    f = torch.bincount(e, minlength=bank.K).float() / N
    l_bal = bank.K * (f * q.mean(0)).sum()
    g_bal, = torch.autograd.grad(l_bal, [Wr])
    Wr.grad = g_res + bank.alpha * g_bal
    return ce, e, {"bal": float(l_bal.detach()), "load_max": float(f.max()), "n_dead": int((f < 1.0 / (10 * bank.K)).sum())}


@torch.no_grad()
def deltas(w1, w2, h, xn, W, y):
    """d[t, k] = CE_base(t) - CE(base + a_k)(t) (N, K) fp32 and CE_base (N,)."""
    K = w1.shape[0]
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
        lse_k = torch.logsumexp(zb.unsqueeze(0) + dl.float(), -1)
        dy = dl.float().gather(2, yy.view(1, -1, 1).expand(K, -1, 1)).squeeze(2)
        ceb[s:s + UCHUNK] = lse_b - zb.gather(1, yy.view(-1, 1)).squeeze(1)
        d[s:s + UCHUNK] = (lse_b.unsqueeze(0) - lse_k + dy).t()
        del zb, hid, a, dl
    return d, ceb


def r1_target(p, eps):
    p = p.float() + eps
    return p / p.sum(1, keepdim=True)


def r1_loss(logits, pi, lam):
    K = logits.shape[1]
    logq = torch.log_softmax(logits.float(), 1)
    kl = (pi * (torch.log(pi.clamp(min=1e-12)) - logq)).sum(1).mean()
    qb = logq.exp().mean(0)
    qb = qb / qb.sum()
    bal = (qb * torch.log((K * qb).clamp(min=1e-12))).sum()
    return kl + lam * bal, {"kl": float(kl.detach()), "bal": float(bal.detach())}


def assign(p, mode, gen=None):
    """Target-free specialist assignment from the region prior p (N, K): 'hard' = argmax; 'soft' = one sample from p
    with a dedicated generator (fixed per window/step by the caller)."""
    if mode == "hard":
        return p.argmax(1)
    q = p.float().clamp(min=0)
    q = q / q.sum(1, keepdim=True)
    u = torch.rand(q.shape[0], generator=gen, device="cpu").to(q.device)
    c = torch.cumsum(q, 1)
    return torch.searchsorted(c, u.unsqueeze(1).clamp(max=1 - 1e-7)).squeeze(1).clamp(max=q.shape[1] - 1)


def pathology(e, K):
    f = torch.bincount(e, minlength=K).float() / max(1, e.numel())
    return {"load_max": float(f.max()), "load_eff": float(1.0 / (f ** 2).sum().clamp(min=1e-12)),
            "n_dead": int((f < 1.0 / (10 * K)).sum())}


def finite(*vals):
    return all(math.isfinite(float(v)) for v in vals)
