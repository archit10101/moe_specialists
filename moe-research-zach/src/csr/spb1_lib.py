"""spb1 core: residual-specialist banks with K*M specialists on the frozen TOKEN-4 base (orv1 insertion point), the
training-time routers, and the deployed unrestricted causal router.

Insertion (identical to orv1): h'_t = x_t + a_{e(t)}(rms0(x_t)), logits_t = W_emb norm_f(h'_t); x_t = residual stream
after the final MoE block; a_e = W2_e GELU(W1_e .) (384 -> 256 -> 384, W2 zero-initialised, W1 from a seed-specific
generator IDENTICAL across conditions). Exactly one specialist per token.

Training-time routing (one shared linear router 384 -> E = K*M logits on rms0(x_t), straight-through gate p/sg(p)):
  'pair' (SEMANTIC-M / SHUFFLED-M): the token's paper cluster c (semantic or shuffled) restricts the choice to the M
          specialists {c*M, ..., c*M + M - 1}; the router chooses freely among them; Switch load balance only within
          each cluster's M-tuple: aux = sum_c (n_c / n) * M * sum_{e in c} f_e P_e  (f, P measured over cluster-c tokens)
  'free'  (GENERIC-E): unrestricted over all E; aux = E * sum_e f_e P_e
All logits are computed for every condition (identical router parameters and FLOPs); 'pair' masks them.

Deployed router (fresh, after the bank is frozen; training-time routers discarded): ccsr1 current-context input
[rms0(x_t), causal mean of rms0(x_s) over the last 16 positions of the document] (+ optional causal semantic posterior)
-> Linear -> GRU(H) -> Linear(H -> E) logits / temp; NO mask, NO semantic bias, NO prescribed mapping: every token can
select any of the E specialists. Trained by the routed LM loss (expected CE under softmax over the frozen bank's exact
per-token CE table) + a standard corpus-level balance term."""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from csr.lfr_model import Experts
from csr.spb1_paths import D_MODEL, EXPERT_FFN, INIT_SEED_OFFSET, INIT_STD, WIN

EPS = 1e-6


def rms0(x):
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)


class PairBank(nn.Module):
    def __init__(self, name, seed, K, M, mode):
        super().__init__()
        assert mode in ("pair", "free")
        self.name, self.K, self.M, self.E, self.mode = name, K, M, K * M, mode
        self.bank = Experts(self.E, D_MODEL, EXPERT_FFN)
        self.router = nn.Linear(D_MODEL, self.E, bias=False)
        g = torch.Generator().manual_seed(seed + INIT_SEED_OFFSET)
        with torch.no_grad():
            self.bank.w1.copy_(torch.randn(self.bank.w1.shape, generator=g) * INIT_STD)
            self.bank.w2.zero_()
        gr = torch.Generator().manual_seed(seed + INIT_SEED_OFFSET + 1)
        with torch.no_grad():
            self.router.weight.copy_(torch.randn(self.router.weight.shape, generator=gr) * INIT_STD)

    def n_params(self):
        return {"bank": self.bank.w1.numel() + self.bank.w2.numel(), "router": self.router.weight.numel()}

    def logits(self, xn):
        with torch.autocast(device_type=xn.device.type, enabled=False):
            return F.linear(xn.float(), self.router.weight.float())

    def allowed(self, group):
        """(N, E) bool: specialists selectable for each token in this mode."""
        N = group.shape[0]
        if self.mode == "free":
            return torch.ones(N, self.E, dtype=torch.bool, device=group.device)
        e_grp = torch.arange(self.E, device=group.device) // self.M
        return e_grp.view(1, -1) == group.view(-1, 1)

    def route(self, xn, group, valid):
        lg = self.logits(xn)
        lg = lg.masked_fill(~self.allowed(group), float("-inf"))
        p = torch.softmax(lg, -1)
        idx = p.argmax(-1)
        pe = p.gather(1, idx.unsqueeze(1)).squeeze(1)
        gate = pe / pe.detach()
        vf = valid.float()
        oh = F.one_hot(idx, self.E).float() * vf.unsqueeze(1)
        pv = p * vf.unsqueeze(1)
        if self.mode == "free":
            nv = vf.sum().clamp(min=1)
            aux = self.E * ((oh.sum(0) / nv) * (pv.sum(0) / nv)).sum()
        else:
            e_grp = torch.arange(self.E, device=xn.device) // self.M
            g1 = F.one_hot(group.clamp(min=0), self.K).float() * vf.unsqueeze(1)   # (N, K)
            n_c = g1.sum(0)                                                        # tokens per cluster
            n_e = n_c[e_grp].clamp(min=1)
            f = oh.sum(0) / n_e
            P = pv.sum(0) / n_e
            w_e = (n_c[e_grp] / vf.sum().clamp(min=1))
            aux = self.M * (w_e * f * P).sum()
        return idx, gate, aux, p

    def forward(self, xn, group, valid):
        idx, gate, aux, p = self.route(xn, group, valid)
        with torch.autocast(device_type=xn.device.type, dtype=torch.bfloat16, enabled=xn.device.type == "cuda"):
            delta = self.bank(xn, idx.unsqueeze(1), gate.unsqueeze(1))
        return delta.float(), idx, aux

    @torch.no_grad()
    def one(self, xn, e):
        """Specialist e applied to every token (same dispatch kernel as training/eval)."""
        idx = torch.full((xn.shape[0], 1), e, dtype=torch.long, device=xn.device)
        with torch.autocast(device_type=xn.device.type, dtype=torch.bfloat16, enabled=xn.device.type == "cuda"):
            return self.bank(xn, idx, torch.ones(xn.shape[0], 1, device=xn.device)).float()


def make_banks(seed, K, M, device):
    s, h, g = (f"SEM{M}", f"SHUF{M}", f"GEN{K * M}")
    return {s: PairBank(s, seed, K, M, "pair").to(device), h: PairBank(h, seed, K, M, "pair").to(device),
            g: PairBank(g, seed, K, M, "free").to(device)}


def token_groups(bt, lab_sem, lab_shuf):
    gd = bt["gdoc"].reshape(-1)
    ok = gd >= 0
    g = gd.clamp(min=0)
    return torch.where(ok, lab_sem[g], torch.zeros_like(g)), torch.where(ok, lab_shuf[g], torch.zeros_like(g))


# ------------------------------------------------------------------------------------------- deployed router
def causal_mean(x, win):
    """x (B, L, d) padded per document (positions 0..len-1); mean over the last `win` positions (inclusive)."""
    c = F.pad(torch.cumsum(x.float(), 1), (0, 0, 1, 0))
    L = x.shape[1]
    t = torch.arange(L, device=x.device)
    lo = (t + 1 - win).clamp(min=0)
    return (c[:, t + 1] - c[:, lo]) / (t + 1 - lo).float().view(1, -1, 1)


class FreeRouter(nn.Module):
    """Unrestricted causal router over E specialists. extra: optional per-token causal features (e.g. a frozen causal
    semantic posterior), concatenated to the input; the router may ignore them (no mask, no bias, no mapping)."""

    def __init__(self, H, seed, E, n_extra=0, temp=1.0):
        super().__init__()
        self.H, self.E, self.n_extra, self.temp = H, E, n_extra, float(temp)
        self.inp = nn.Linear(2 * D_MODEL + n_extra, H, bias=False)
        self.gru = nn.GRU(H, H, batch_first=True)
        self.out = nn.Linear(H, E)
        g = torch.Generator().manual_seed(int(seed) * 7919 + 23)
        with torch.no_grad():
            for p in self.parameters():
                if p.dim() >= 2:
                    p.copy_(torch.randn(p.shape, generator=g) / math.sqrt(p.shape[1]))
                else:
                    p.zero_()
            self.out.weight.mul_(0.1)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, x, extra=None):
        """x: (B, L, 384) rms0 states padded per document. Returns logits (B, L, E). Causal by construction."""
        f = [x.float(), causal_mean(x, WIN)]
        if self.n_extra:
            f.append(extra.float())
        s, _ = self.gru(self.inp(torch.cat(f, -1)))
        return self.out(s).float() / self.temp


class SubjectNet(nn.Module):
    """Causal semantic-posterior model (secondary ablation only): same backbone family, predicts the token's paper
    semantic cluster from x_<=t. Its softmax is a frozen input feature of the ablation router."""

    def __init__(self, H, seed, K):
        super().__init__()
        self.inp = nn.Linear(2 * D_MODEL, H, bias=False)
        self.gru = nn.GRU(H, H, batch_first=True)
        self.out = nn.Linear(H, K)
        g = torch.Generator().manual_seed(int(seed) * 7919 + 29)
        with torch.no_grad():
            for p in self.parameters():
                if p.dim() >= 2:
                    p.copy_(torch.randn(p.shape, generator=g) / math.sqrt(p.shape[1]))
                else:
                    p.zero_()

    def forward(self, x):
        s, _ = self.gru(self.inp(torch.cat([x.float(), causal_mean(x, WIN)], -1)))
        return self.out(s).float()


def doc_index(doc, pos):
    """Documents as contiguous runs of the table with positions 0..n-1. Returns starts, lengths."""
    doc = np.asarray(doc)
    pos = np.asarray(pos).astype(np.int64)
    brk = np.nonzero(np.diff(doc) != 0)[0] + 1
    starts = np.concatenate([[0], brk]).astype(np.int64)
    lens = np.diff(np.concatenate([starts, [len(doc)]])).astype(np.int64)
    if len(np.unique(doc[starts])) != len(starts):
        raise RuntimeError("document tokens are not contiguous in the table")
    if not np.array_equal(np.arange(len(doc)) - np.repeat(starts, lens), pos):
        raise RuntimeError("table positions are not 0..n-1 within each document")
    return starts, lens


def batch_index(starts, lens, docs, dev):
    st = torch.as_tensor(starts[docs], device=dev)
    ln = torch.as_tensor(lens[docs], device=dev)
    L = int(ln.max())
    t = torch.arange(L, device=dev).view(1, -1)
    mask = t < ln.view(-1, 1)
    idx = (st.view(-1, 1) + t).clamp(max=int(starts[-1] + lens[-1] - 1))
    return idx, mask, ln


def router_loss(logits, d, bal):
    """logits (N, E); d (N, E) = CE_e - CE_base (exact, frozen bank). Expected routed CE (relative to base) + bal *
    KL(qbar || uniform)."""
    p = torch.softmax(logits, -1)
    lm = (p * d).sum(1).mean()
    qb = p.mean(0)
    b = (qb * torch.log((qb * logits.shape[1]).clamp(min=1e-12))).sum()
    return lm + bal * b, float(lm.detach()), float(b.detach())
