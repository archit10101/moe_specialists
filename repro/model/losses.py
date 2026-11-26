"""Auxiliary losses: Switch load balance, router z-loss, and Expert Divergence Learning (doc §3).

EDL (ported from Project-davin/subj/subj_moe.py ``edl_loss``, extended to soft group weights):
for each group g present in the micro-batch,
    R_g = sum_t w_tg p(e | h_t) / sum_t w_tg          (mean router distribution of the group)
    L_EDL = mean over pairs (g_i, g_j) of  -log( JSD(R_gi, R_gj) + eps )
Hard labels use one-hot w; Semantic C6 uses (sharpened) soft per-paper cluster posteriors.
Routing itself stays unrestricted top-k per token: only group-average distributions are pushed apart.
"""
from __future__ import annotations

import torch


def switch_loss(model) -> torch.Tensor:
    return torch.stack([m.aux for m in model.moe_layers()]).mean()


def z_loss(model) -> torch.Tensor:
    return torch.stack([m.z for m in model.moe_layers()]).mean()


def edl_layer_loss(probs: torch.Tensor, w: torch.Tensor, min_mass: float, eps: float = 1e-6):
    """probs [N, E] (fp32, with grad), w [N, G] group weights. Returns None if < 2 groups present."""
    mass = w.sum(0)                                   # [G]
    keep = mass >= min_mass
    if int(keep.sum()) < 2:
        return None
    R = (w[:, keep].T @ probs) / mass[keep, None]     # [G', E]
    R = R.clamp_min(1e-9)
    M = 0.5 * (R[:, None] + R[None])                  # [G', G', E]
    kl = lambda a, m: (a * (a.log() - m.log())).sum(-1)
    jsd = 0.5 * (kl(R[:, None], M) + kl(R[None], M))
    i, j = torch.triu_indices(len(R), len(R), 1, device=R.device)
    # float error can make JSD of near-identical rows slightly negative -> clamp before the log
    return -(jsd[i, j].clamp_min(0) + eps).log().mean()


def edl_loss(model, w: torch.Tensor, layers: str, min_mass: float):
    """Mean EDL loss over the selected MoE layers. w: [B, T, G] weights for this micro-batch."""
    moes = model.moe_layers()
    idx = range(len(moes)) if layers == "all" else range(len(moes) // 2, len(moes))
    wf = w.reshape(-1, w.shape[-1]).float()
    terms = [edl_layer_loss(moes[l].probs, wf, min_mass) for l in idx]
    terms = [t for t in terms if t is not None]
    if not terms:
        return None
    return torch.stack(terms).mean()


def edl_layers(model, layers: str) -> list[int]:
    n = len(model.moe_layers())
    return list(range(n)) if layers == "all" else list(range(n // 2, n))
