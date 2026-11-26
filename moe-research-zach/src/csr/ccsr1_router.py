"""csr-current-context-specialist-router-v1 (ccsr1): strictly causal per-token residual-specialist router.

At token t of a document the router sees only x_<=t through the frozen base: its input is
    feat_t = [rms0(h_t), causal mean of rms0(h_s) over s in (t - WIN, t] within the document]   (feats='base')
(engineering variants: 'ctx' adds the causal mean over the last 256 tokens; 'chg' adds mean_16 - mean_(t-64, t-16])
-> Linear(n_feat * 384 -> H, no bias) -> GRU(H, H) (previous router state) -> Linear(H -> n_out).
q_t = softmax(logits_t) over the 16 frozen specialists (n_out = 17 adds a zero-residual/base-only action, engineering
only). The deployed choice is argmax q_t and the selected specialist's correction is applied to h_t to predict x_{t+1}.
The target token, utilities and region priors are used only to build training targets.

Targets (training only): u[t,k] = CE_base(t) - CE(base + residual_k)(t) = -d[t,k];
  R1 pi = p[k | current 256-token region]                     (semantic teacher; token weight 1)
  R2 pi = softmax(u / tau)                                     (exact utility teacher)
  R3 pi = softmax(u / tau + eta * log(p + eps))                (semantic-utility posterior; primary)
  tau = c * s_u, s_u = median over TRAIN tokens of the per-token SD of u over the 16 specialists (bank-specific).
  R2/R3 token weight w_t = clip(m_t / median(m), 0.25, 4), m_t = max_k u[t,k] - mean_k u[t,k] (utility margin).
Loss = sum_t w_t KL(pi_t || q_t) / sum_t w_t + lambda * sum_k qbar_k log(K qbar_k)  (qbar: mean q over the >= 64-doc,
~63k-token batch spanning many papers = corpus-level balance; zero action excluded from qbar)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from csr.ccsr1_paths import D_MODEL, K, PSEM_EPS, WIN

FEATS = {"base": 2, "ctx": 3, "chg": 3}


def causal_mean(x, lengths, win):
    """x (B, L, d) padded per document (positions 0..len-1); returns mean over the last `win` positions (inclusive)."""
    c = torch.cumsum(x.float(), 1)
    c = F.pad(c, (0, 0, 1, 0))                     # c[:, t+1] = sum_{s<=t}
    L = x.shape[1]
    t = torch.arange(L, device=x.device)
    lo = (t + 1 - win).clamp(min=0)
    s = c[:, t + 1] - c[:, lo]
    n = (t + 1 - lo).float().view(1, -1, 1)
    return s / n


def build_feats(x, lengths, kind):
    """x: (B, L, 384) rms0(h) padded. Causal features (B, L, n*384)."""
    m16 = causal_mean(x, lengths, WIN)
    if kind == "base":
        return torch.cat([x.float(), m16], -1)
    if kind == "ctx":
        return torch.cat([x.float(), m16, causal_mean(x, lengths, 256)], -1)
    if kind == "chg":
        m64 = causal_mean(x, lengths, 64)
        # mean over (t-64, t-16]: (64 * m64 - 16 * m16) / 48 where both windows are full; near the start use what exists
        L = x.shape[1]
        t = torch.arange(L, device=x.device).view(1, -1, 1).float()
        n64, n16 = torch.clamp(t + 1, max=64), torch.clamp(t + 1, max=16)
        prev = (n64 * m64 - n16 * m16) / torch.clamp(n64 - n16, min=1)
        prev = torch.where(n64 > n16, prev, m16)
        return torch.cat([x.float(), m16, m16 - prev], -1)
    raise ValueError(kind)


class CausalRouter(nn.Module):
    def __init__(self, H, seed, feats="base", n_out=K):
        super().__init__()
        self.H, self.feats, self.n_out = H, feats, n_out
        self.inp = nn.Linear(FEATS[feats] * D_MODEL, H, bias=False)
        self.gru = nn.GRU(H, H, batch_first=True)
        self.head = nn.Linear(H, n_out)
        g = torch.Generator().manual_seed(int(seed) * 7919 + 17)
        with torch.no_grad():
            for p in self.parameters():
                if p.dim() >= 2:
                    p.copy_(torch.randn(p.shape, generator=g) / math.sqrt(p.shape[1]))
                else:
                    p.zero_()

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, x, lengths):
        """x (B, L, 384) padded rms0(h); returns logits (B, L, n_out). Padding is only at the end (causal GRU), so
        padded positions never influence valid ones."""
        f = build_feats(x, lengths, self.feats)
        z = self.inp(f)
        s, _ = self.gru(z)
        return self.head(s)

    @torch.no_grad()
    def step(self, x_t, cache):
        """One-token routing for ONE document. x_t (384,). cache: dict(hist list of past x (up to 256), state (1,1,H)).
        Returns logits (n_out,) and the updated cache. Equivalent to forward() on the prefix."""
        hist = cache.get("hist", []) + [x_t.float()]
        hist = hist[-256:]
        X = torch.stack(hist)                                 # (n, 384)
        xm = lambda w: X[-w:].mean(0)
        if self.feats == "base":
            f = torch.cat([x_t.float(), xm(WIN)])
        elif self.feats == "ctx":
            f = torch.cat([x_t.float(), xm(WIN), xm(256)])
        else:
            n = X.shape[0]
            n64, n16 = min(n, 64), min(n, 16)
            prev = X[-n64:-n16].mean(0) if n64 > n16 else xm(16)
            f = torch.cat([x_t.float(), xm(16), xm(16) - prev])
        z = self.inp(f).view(1, 1, -1)
        s, st = self.gru(z, cache.get("state"))
        return self.head(s).view(-1), {"hist": hist, "state": st}


def teacher(kind, u, logp, c, eta, s_u, n_out=K):
    """Training-only target pi (N, n_out). u (N,16) utilities; logp (N,16) log(p + eps) of the bank's own prior."""
    if n_out == K + 1:
        u = torch.cat([u, torch.zeros_like(u[:, :1])], 1)
        logp = torch.cat([logp, torch.full_like(logp[:, :1], math.log(1.0 / K))], 1)
    if kind == "R1":
        p = logp.exp()
        if n_out == K + 1:
            p[:, K] = 0.0
        return p / p.sum(1, keepdim=True)
    tau = c * s_u
    z = u / tau
    if kind == "R3":
        z = z + eta * logp
    return torch.softmax(z, 1)


def token_weight(kind, u, med_margin):
    if kind == "R1":
        return torch.ones(u.shape[0], device=u.device)
    from csr.ccsr1_paths import W_CLIP
    m = u.max(1).values - u.mean(1)
    return (m / med_margin).clamp(W_CLIP[0], W_CLIP[1])


def router_loss(logits, pi, w, lam, u=None, beta_eu=0.0, s_u=1.0):
    logq = torch.log_softmax(logits.float(), 1)
    kl = (pi * (torch.log(pi.clamp(min=1e-12)) - logq)).sum(1)
    loss = (w * kl).sum() / w.sum()
    q = logq.exp()
    qb = q[:, :K].mean(0)
    qb = qb / qb.sum()
    bal = (qb * torch.log((K * qb).clamp(min=1e-12))).sum()
    tot = loss + lam * bal
    eu = torch.zeros((), device=logits.device)
    if beta_eu > 0 and u is not None:
        uu = u if q.shape[1] == K else torch.cat([u, torch.zeros_like(u[:, :1])], 1)
        eu = -(q * uu).sum(1).mean() / s_u
        tot = tot + beta_eu * eu
    return tot, {"kl": float(loss.detach()), "bal": float(bal.detach()), "eu": float(eu.detach())}


def log_prior(P, rid):
    return torch.log(P[rid] + PSEM_EPS)
