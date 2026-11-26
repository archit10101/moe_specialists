"""scur1 subject-conditioned utility router (K = 5).

SCURouter: causal backbone (ccsr1 features: [rms0(h_t), causal mean of the preceding 16 states] (+ optional longer
causal context) -> Linear -> GRU(H)) with two heads:
  subject head    p_subject(s | x_<=t) = softmax(W_s g_t)
  correction head delta_t = tanh(W_c g_t)  (bounded in [-1, 1])
  p_prior(e)      = sum_s p_subject(s) A[s, e]                    A = softmax_e(U[s, e] / tau), frozen, detached
  logits_e        = log(p_prior(e) + eps) / prior_temp + rho * delta_e   (prior_temp = 1 unless a generation sets it);   p_route = softmax(logits);  deployed e = argmax
Modes: 'scu' (as above), 'subject' (A = I, rho = 0: subject-prior-only control, no U), 'utility' (no subject posterior:
logits = W_c g_t unbounded, trained by the routed LM loss only). U is computed from TRAIN tokens only, cross-fitted
over two paper folds for training targets (a token's own fold never contributes to the U that shapes its targets);
the deployed A uses all TRAIN tokens. No target, region label or oracle identity is a router INPUT."""
import math

import torch
import torch.nn as nn

from csr.ccsr1_router import FEATS, build_feats

D_MODEL = 384


class SCURouter(nn.Module):
    def __init__(self, H, seed, K, feats="base", mode="scu", rho=1.0, eps=1e-4, prior_temp=1.0):
        super().__init__()
        self.H, self.K, self.feats, self.mode, self.rho, self.eps = H, K, feats, mode, float(rho), float(eps)
        self.prior_temp = float(prior_temp)
        self.inp = nn.Linear(FEATS[feats] * D_MODEL, H, bias=False)
        self.gru = nn.GRU(H, H, batch_first=True)
        self.subj = nn.Linear(H, K)
        self.corr = nn.Linear(H, K)
        g = torch.Generator().manual_seed(int(seed) * 7919 + 17)
        with torch.no_grad():
            for p in self.parameters():
                if p.dim() >= 2:
                    p.copy_(torch.randn(p.shape, generator=g) / math.sqrt(p.shape[1]))
                else:
                    p.zero_()
            self.corr.weight.mul_(0.1)
        self.register_buffer("A", torch.eye(K))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def set_A(self, A):
        with torch.no_grad():
            self.A.copy_(A)

    def backbone(self, x, lengths):
        s, _ = self.gru(self.inp(build_feats(x, lengths, self.feats)))
        return s

    def heads(self, g, A=None):
        """g: (..., H) backbone states. A: (K, K) or per-token (..., K, K). Returns subject logits, route logits."""
        sl = self.subj(g).float()
        cr = self.corr(g).float()
        if self.mode == "utility":
            return sl, cr
        ps = torch.softmax(sl, -1)
        if self.mode == "subject":
            return sl, torch.log(ps + self.eps)
        A = self.A if A is None else A
        pp = torch.einsum("...s,...se->...e", ps, A) if A.dim() > 2 else ps @ A
        return sl, torch.log(pp + self.eps) / self.prior_temp + self.rho * torch.tanh(cr)

    def forward(self, x, lengths):
        return self.heads(self.backbone(x, lengths))


def utility_matrix(q, gain, w=None):
    """U[s, e] = sum_t q[t, s] gain[t, e] / sum_t q[t, s]; gain = CE_ref - CE_e (positive = expert helps)."""
    q = q.double()
    if w is not None:
        q = q * w.double().unsqueeze(1)
    num = q.t() @ gain.double()
    den = q.sum(0).clamp(min=1e-12).unsqueeze(1)
    return (num / den).float()


def mapping(U, tau_factor, kind="soft"):
    """A[s, e] from U: 'soft' softmax_e(U / (tau_factor * sd(U))); 'hard' one-hot argmax; 'sinkhorn' balanced soft."""
    sd = float(U.std()) if U.numel() > 1 else 1.0
    if kind == "hard":
        return torch.nn.functional.one_hot(U.argmax(1), U.shape[1]).float()
    A = torch.softmax(U / max(1e-8, tau_factor * sd), 1)
    if kind == "sinkhorn":
        for _ in range(50):
            A = A / A.sum(0, keepdim=True) * (A.shape[0] / A.shape[1])
            A = A / A.sum(1, keepdim=True)
    return A


def losses(sl, rl, q_sem, A_tok, U_tok, d_tok, w):
    """Per-batch losses. sl/rl: subject / route logits (N, K); q_sem (N, K) frozen soft semantic assignment (target);
    A_tok (N, K, K) cross-fitted mapping; U_tok (N, K, K) cross-fitted utility; d_tok (N, K) = CE_e - CE_ref (router LM
    term; experts frozen); w: dict of weights. Returns total, parts."""
    K = rl.shape[1]
    logq = torch.log_softmax(rl, 1)
    p = logq.exp()
    parts = {}
    tot = torch.zeros((), device=rl.device)
    if w.get("subject", 0) > 0:
        lps = torch.log_softmax(sl, 1)
        parts["subject"] = (q_sem * (torch.log(q_sem.clamp(min=1e-12)) - lps)).sum(1).mean()
        tot = tot + w["subject"] * parts["subject"]
    if w.get("utility", 0) > 0:
        qu = torch.einsum("ns,nse->ne", q_sem, A_tok)
        parts["utility"] = (qu * (torch.log(qu.clamp(min=1e-12)) - logq)).sum(1).mean()
        tot = tot + w["utility"] * parts["utility"]
    if w.get("regret", 0) > 0:
        us = torch.einsum("ns,nse->ne", q_sem, U_tok)
        reg = us.max(1, keepdim=True).values - us
        sdu = float(U_tok[0].std()) + 1e-8
        parts["regret"] = (p * reg).sum(1).mean() / sdu
        tot = tot + w["regret"] * parts["regret"]
    if w.get("lm", 0) > 0:
        parts["lm"] = (p * d_tok).sum(1).mean()
        tot = tot + w["lm"] * parts["lm"]
    if w.get("balance", 0) > 0:
        qb = p.mean(0)
        qb = qb / qb.sum()
        parts["balance"] = (qb * torch.log((K * qb).clamp(min=1e-12))).sum()
        tot = tot + w["balance"] * parts["balance"]
    return tot, {k: float(v.detach()) for k, v in parts.items()}
