"""TOKEN-4 MoE language model (doc §2.2, "later LFR / TOKEN-4 backbone").

Pre-norm GPT; every block's FFN is an MoE layer = one shared SwiGLU expert (every token) + ``n_exp``
routed SwiGLU experts with unrestricted top-k routing and renormalised gates. No noise, no capacity
limit, no token dropping. Ported from Class-Project/moe-window-routing/model.py (MoELayer) with:
  * sort-based dispatch (one argsort per layer instead of a where() per expert),
  * ``ban``: experts removed from a layer's candidate set (the next-ranked expert is admitted, k stay
    active) - used by every ablation metric,
  * ``keep_probs``: the full router softmax is kept *with* gradients for the EDL loss,
  * ``trunk()`` / ``head()`` split so a residual bank can be inserted after the final block.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelCfg


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6, learnable=True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d)) if learnable else None

    def forward(self, x):
        y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return y * self.weight if self.weight is not None else y


class LayerNorm(nn.Module):
    """Bias-free LayerNorm (nanoGPT)."""

    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, None, 1e-5)


class Attention(nn.Module):
    def __init__(self, c: ModelCfg):
        super().__init__()
        self.qkv = nn.Linear(c.d, 3 * c.d, bias=False)
        self.proj = nn.Linear(c.d, c.d, bias=False)
        self.h = c.n_head

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class SwiGLU(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.w_gate = nn.Linear(d, hidden, bias=False)
        self.w_up = nn.Linear(d, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class MoELayer(nn.Module):
    def __init__(self, c: ModelCfg):
        super().__init__()
        self.n_exp, self.top_k = c.n_exp, c.top_k
        self.router_norm = RMSNorm(c.d)
        self.router = nn.Linear(c.d, c.n_exp, bias=False)
        self.shared = SwiGLU(c.d, c.shared_hidden)
        self.experts = nn.ModuleList([SwiGLU(c.d, c.expert_hidden) for _ in range(c.n_exp)])
        # controls
        self.ban: torch.Tensor | None = None     # bool [n_exp]: experts removed from this layer
        self.keep_probs = False
        # filled by forward
        self.aux = self.z = None                  # differentiable scalars
        self.probs = None                         # [N, E] router softmax (with grad iff keep_probs)
        self.top_i = self.top_p = None            # [N, k] selected experts and their renormalised gates
        self.top_raw = None                       # [N, k] their raw router probabilities
        self.next_p = None                        # [N] probability of the best non-selected expert

    def forward(self, h):
        B, T, C = h.shape
        N, k = B * T, self.top_k
        with torch.autocast(device_type=h.device.type, enabled=False):
            logits = self.router(self.router_norm(h.float())).reshape(N, self.n_exp)
            if self.ban is not None:
                logits = logits.masked_fill(self.ban.to(logits.device), float("-inf"))
            probs = F.softmax(logits, dim=-1)
            top2 = probs.topk(min(k + 1, self.n_exp), dim=-1)
            top_p, top_i = top2.values[:, :k], top2.indices[:, :k]
            gates = top_p / top_p.sum(-1, keepdim=True)

        x = h.reshape(N, C)
        flat_e = top_i.reshape(-1)
        order = torch.argsort(flat_e, stable=True)
        counts = torch.bincount(flat_e, minlength=self.n_exp)
        tok = order // k
        g = gates.reshape(-1)[order].unsqueeze(-1)
        out = torch.zeros_like(x)
        start = 0
        for e, n_e in enumerate(counts.tolist()):
            if n_e:
                sl = slice(start, start + n_e)
                y = self.experts[e](x[tok[sl]]) * g[sl].to(x.dtype)
                out.index_add_(0, tok[sl], y.to(out.dtype))
                start += n_e
        out = out + self.shared(x)

        f = counts.float() / (N * k)
        P = probs.mean(0)
        self.aux = self.n_exp * torch.sum(f.detach() * P)
        finite = torch.where(torch.isfinite(logits), logits, torch.zeros_like(logits))
        self.z = torch.logsumexp(finite, dim=-1).pow(2).mean()
        self.probs = probs if self.keep_probs else probs.detach()
        self.top_i, self.top_p, self.top_raw = top_i.detach(), gates.detach(), top_p.detach()
        self.next_p = top2.values[:, k].detach() if top2.values.shape[1] > k else torch.zeros(N, device=h.device)
        return out.view(B, T, C)


class Block(nn.Module):
    def __init__(self, c: ModelCfg):
        super().__init__()
        self.ln_1, self.attn = LayerNorm(c.d), Attention(c)
        self.ln_2, self.moe = LayerNorm(c.d), MoELayer(c)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.moe(self.ln_2(x))


class MoEGPT(nn.Module):
    def __init__(self, c: ModelCfg):
        super().__init__()
        self.cfg = c
        self.wte = nn.Embedding(c.vocab_size, c.d)
        self.wpe = nn.Embedding(c.ctx, c.d)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.n_layer)])
        self.ln_f = LayerNorm(c.d)
        self.lm_head = nn.Linear(c.d, c.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight        # tied
        self.apply(self._init)
        std = 0.02 / math.sqrt(2 * c.n_layer)
        for n, p in self.named_parameters():
            if n.endswith("attn.proj.weight") or n.endswith("w_down.weight"):
                nn.init.normal_(p, 0.0, std)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)

    # -- forward pieces
    def trunk(self, idx):
        """Hidden state after the final block, before ln_f: [B, T, d]."""
        x = self.wte(idx) + self.wpe(torch.arange(idx.shape[1], device=idx.device))
        for b in self.blocks:
            x = b(x)
        return x

    def head(self, h):
        return self.lm_head(self.ln_f(h))

    def forward(self, idx):
        return self.head(self.trunk(idx))

    # -- MoE helpers
    def moe_layers(self) -> list[MoELayer]:
        return [b.moe for b in self.blocks]

    def set_ban(self, bans: dict | None) -> None:
        """bans: {layer: iterable of expert ids}; None clears every ban."""
        for l, m in enumerate(self.moe_layers()):
            ids = None if bans is None else bans.get(l)
            if ids is None or len(ids) == 0:
                m.ban = None
            else:
                mask = torch.zeros(m.n_exp, dtype=torch.bool)
                mask[list(ids)] = True
                m.ban = mask

    def set_keep_probs(self, on: bool) -> None:
        for m in self.moe_layers():
            m.keep_probs = on

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())   # tied weights are counted once


def token_ce(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-token next-token cross-entropy in nats, shape of y."""
    return F.cross_entropy(logits.float().flatten(0, -2), y.flatten(), reduction="none").view_as(y)
