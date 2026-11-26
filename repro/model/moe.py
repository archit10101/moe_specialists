"""TOKEN-4 MoE language model (doc §2.2, "later LFR / TOKEN-4 backbone").

Pre-norm GPT following the original code (lfr_model.py): RMSNorm, RoPE, block 0 a dense GELU FFN, blocks
1..11 MoE = one shared GELU expert (every token) + ``n_exp`` routed GELU experts with unrestricted top-k
routing. Gates are the raw router softmax of the selected experts (no renormalisation, "FLAME"). No noise,
no capacity limit, no token dropping. 103,609,728 parameters with the default ModelCfg. Additions:
  * sort-based dispatch (one argsort per layer),
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


def rope_cos_sin(T: int, hd: int, base: float, device):
    inv = 1.0 / (base ** (torch.arange(0, hd, 2, device=device, dtype=torch.float32) / hd))
    f = torch.outer(torch.arange(T, device=device, dtype=torch.float32), inv)
    return f.cos(), f.sin()                                       # [T, hd/2]


def apply_rope(x, cos, sin):
    """x [B, H, T, hd]; rotate-half convention."""
    x1, x2 = x.float().chunk(2, dim=-1)
    c, s_ = cos[None, None], sin[None, None]
    return torch.cat([x1 * c - x2 * s_, x2 * c + x1 * s_], -1).to(x.dtype)


class Attention(nn.Module):
    def __init__(self, c: ModelCfg):
        super().__init__()
        self.qkv = nn.Linear(c.d, 3 * c.d, bias=False)
        self.proj = nn.Linear(c.d, c.d, bias=False)
        self.h = c.n_head
        self.rope = c.pos == "rope"

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        if self.rope:
            cos, sin = rope_cos_sin(T, C // self.h, 10000.0, x.device)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class GELUMLP(nn.Module):
    """2-matrix tanh-GELU MLP, no bias (the original's experts, shared expert and dense FFN)."""

    def __init__(self, d, hidden):
        super().__init__()
        self.w_up = nn.Linear(d, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.w_down(F.gelu(self.w_up(x), approximate="tanh"))


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
        self.n_exp, self.top_k, self.renorm = c.n_exp, c.top_k, c.renorm_gates
        self.router_norm = RMSNorm(c.d) if c.router_norm else None
        self.router = nn.Linear(c.d, c.n_exp, bias=False)
        mlp = GELUMLP if c.expert_act == "gelu" else SwiGLU
        self.shared = mlp(c.d, c.shared_hidden)
        self.experts = nn.ModuleList([mlp(c.d, c.expert_hidden) for _ in range(c.n_exp)])
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
            hr = h.float() if self.router_norm is None else self.router_norm(h.float())
            logits = F.linear(hr, self.router.weight.float()).reshape(N, self.n_exp)
            if self.ban is not None:
                logits = logits.masked_fill(self.ban.to(logits.device), float("-inf"))
            probs = F.softmax(logits, dim=-1)
            top2 = probs.topk(min(k + 1, self.n_exp), dim=-1)
            top_p, top_i = top2.values[:, :k], top2.indices[:, :k]
            gates = top_p / top_p.sum(-1, keepdim=True) if self.renorm else top_p

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


def make_norm(c: ModelCfg):
    return RMSNorm(c.d) if c.norm == "rms" else LayerNorm(c.d)


class Block(nn.Module):
    def __init__(self, c: ModelCfg, is_moe: bool = True):
        super().__init__()
        self.ln_1, self.attn = make_norm(c), Attention(c)
        self.ln_2 = make_norm(c)
        self.is_moe = is_moe
        if is_moe:
            self.moe = MoELayer(c)
        else:
            self.ffn = GELUMLP(c.d, c.dense_hidden) if c.expert_act == "gelu" else SwiGLU(c.d, c.dense_hidden)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + (self.moe(self.ln_2(x)) if self.is_moe else self.ffn(self.ln_2(x)))


class MoEGPT(nn.Module):
    def __init__(self, c: ModelCfg):
        super().__init__()
        self.cfg = c
        self.wte = nn.Embedding(c.vocab_size, c.d)
        self.wpe = nn.Embedding(c.ctx, c.d) if c.pos == "learned" else None
        self.blocks = nn.ModuleList([Block(c, is_moe=i >= c.dense_layers) for i in range(c.n_layer)])
        self.ln_f = make_norm(c)
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
        x = self.wte(idx)
        if self.wpe is not None:
            x = x + self.wpe(torch.arange(idx.shape[1], device=idx.device))
        for b in self.blocks:
            x = b(x)
        return x

    def head(self, h):
        return self.lm_head(self.ln_f(h))

    def forward(self, idx):
        return self.head(self.trunk(idx))

    # -- MoE helpers
    def moe_layers(self) -> list[MoELayer]:
        return [b.moe for b in self.blocks if b.is_moe]

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
