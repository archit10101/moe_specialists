"""Residual specialist bank (doc §23) and its routers (§23 token router, §27 current-context GRU router).

The bank sits after the final MoE block of a frozen base:
        h' = h + a_e(RMSNorm(h)),      a_e: d -> hidden -> d (GELU, no bias),   exactly one e per token.
With d=384, hidden=256 and 16 specialists the bank has 3,145,728 parameters (doc: 3,145,728).
The output matrices start at zero, so every bank starts exactly at the frozen base.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .moe import RMSNorm


class Bank(nn.Module):
    def __init__(self, d: int, n: int, hidden: int, init_std: float = 0.02):
        super().__init__()
        self.n, self.d, self.hidden = n, d, hidden
        self.norm = RMSNorm(d, learnable=False)
        self.w1 = nn.Parameter(torch.randn(n, d, hidden) * init_std)   # original: N(0, 0.02^2)
        self.w2 = nn.Parameter(torch.zeros(n, hidden, d))

    def delta_one(self, x: torch.Tensor, e: int) -> torch.Tensor:
        """Residual of specialist e for x [M, d] (already normalised)."""
        return F.gelu(x @ self.w1[e]) @ self.w2[e]

    def forward(self, h: torch.Tensor, assign: torch.Tensor, scale: torch.Tensor | None = None):
        """h [B, T, d]; assign [B, T] specialist ids; scale [B, T] optional multiplier (router grads)."""
        B, T, d = h.shape
        x = self.norm(h.float()).reshape(-1, d)
        a = assign.reshape(-1)
        out = torch.zeros_like(x)
        order = torch.argsort(a, stable=True)
        counts = torch.bincount(a, minlength=self.n).tolist()
        start = 0
        for e, c in enumerate(counts):
            if c:
                rows = order[start:start + c]
                out[rows] = self.delta_one(x[rows], e)
                start += c
        if scale is not None:
            out = out * scale.reshape(-1, 1)
        return h + out.view(B, T, d).to(h.dtype)

    def forward_fixed(self, h: torch.Tensor, e: int) -> torch.Tensor:
        """Every token through specialist e (forced-equal-exposure evaluation)."""
        B, T, d = h.shape
        x = self.norm(h.float()).reshape(-1, d)
        return h + self.delta_one(x, e).view(B, T, d).to(h.dtype)


class TokenRouter(nn.Module):
    """§23 condition C: causal linear top-1 router on the current hidden state (+ Switch balance)."""

    def __init__(self, d: int, n: int):
        super().__init__()
        self.norm = RMSNorm(d, learnable=False)
        self.proj = nn.Linear(d, n, bias=False)
        nn.init.normal_(self.proj.weight, 0.0, 0.02)
        self.n = n

    def forward(self, h):
        """Returns (assign [B,T], scale [B,T] = p/p.detach(), balance loss)."""
        p = F.softmax(self.proj(self.norm(h.float())), dim=-1)
        top = p.argmax(-1)
        pt = p.gather(-1, top.unsqueeze(-1)).squeeze(-1)
        f = torch.bincount(top.flatten(), minlength=self.n).float() / top.numel()
        bal = self.n * torch.sum(f * p.reshape(-1, self.n).mean(0))
        return top, pt / pt.detach(), bal


def causal_window_mean(h: torch.Tensor, n: int) -> torch.Tensor:
    """Mean of h over positions max(0, t-n+1)..t (from moe-window-routing/model.py)."""
    B, T, C = h.shape
    S = torch.cumsum(h.float(), dim=1)
    S_prev = torch.cat([S.new_zeros(B, n, C), S[:, :T - n]], dim=1) if n < T else torch.zeros_like(S)
    count = torch.arange(1, T + 1, device=h.device, dtype=torch.float32).clamp(max=n)
    return (S - S_prev) / count[None, :, None]


class CtxRouter(nn.Module):
    """§27 router:  [RMSNorm(h_t), mean(RMSNorm(h_{t-w+1..t}))] -> Linear(no bias) -> GRU -> n logits.
    d=384, w=16, hidden 64, n=16  ->  768*64 + GRU(64,64) 24,960 + 64*16+16  =  75,152 parameters.
    Strictly causal: the GRU runs left to right and the window mean only looks back."""

    def __init__(self, d: int, n: int, hidden: int = 64, inputs: str = "both", ctx_window: int = 16,
                 arch: str = "gru"):
        super().__init__()
        assert inputs in ("both", "token", "context") and arch in ("gru", "mlp")
        self.inputs, self.w, self.arch = inputs, ctx_window, arch
        self.norm = RMSNorm(d, learnable=False)
        self.proj = nn.Linear(d * (2 if inputs == "both" else 1), hidden, bias=False)
        if arch == "gru":
            self.core = nn.GRU(hidden, hidden, batch_first=True)
        else:
            self.core = nn.Sequential(nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.out = nn.Linear(hidden, n)

    def features(self, h):
        parts = []
        if self.inputs in ("both", "token"):
            parts.append(self.norm(h.float()))
        if self.inputs in ("both", "context"):
            parts.append(causal_window_mean(self.norm(h.float()), self.w))   # mean of normalised tokens (original)
        return torch.cat(parts, -1)

    def forward(self, h):
        z = self.proj(self.features(h))
        z = self.core(z)[0] if self.arch == "gru" else self.core(z)
        return self.out(z)          # [B, T, n] logits
