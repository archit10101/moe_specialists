"""Relational semantic-routing objective.

  S_ij = u_i . u_j                     (frozen, detached, L2-normalized prefix embeddings)
  Q_i  = softmax_{j != i}(S_ij / tau_s)
  p_i^l = softmax(W_r^l hbar_i^l)      (full, unmasked chunk-router probabilities)
  R_i^l = softmax_{j != i}((p_i^l . p_j^l) / tau_r)
  L_sem = mean_{l, i} KL(Q_i || R_i^l)

All affinity math is done in float32 (or float64 if inputs are float64). Self pairs are excluded with a
large finite negative logit so that exp underflows to exactly 0 and no NaN can arise.
"""
from __future__ import annotations

import torch

NEG = -1e9


def select_valid_chunks(u, prefix_hash, n_max, valid=None):
    """Deterministic comparison set: finite, nonzero-norm, de-duplicated (first occurrence of an identical
    prefix hash wins), then at most n_max rows taken at evenly spaced positions."""
    ok = torch.isfinite(u).all(dim=1) & (u.norm(dim=1) > 1e-6)
    if valid is not None:
        ok &= valid
    seen, keep = set(), []
    for i in torch.nonzero(ok).flatten().tolist():
        h = int(prefix_hash[i])
        if h in seen:
            continue
        seen.add(h)
        keep.append(i)
    idx = torch.tensor(keep, dtype=torch.long, device=u.device)
    if idx.numel() > n_max:
        pos = torch.linspace(0, idx.numel() - 1, n_max, device=u.device).round().long()
        idx = idx[pos]
    return idx


def _offdiag_log_softmax(scores):
    n = scores.shape[-1]
    eye = torch.eye(n, dtype=torch.bool, device=scores.device)
    return torch.log_softmax(scores.masked_fill(eye, NEG), dim=-1), eye


def relational_kl(u, chunk_logits, tau_s, tau_r, min_chunks=8, doc_of=None):
    """u (N, D) frozen targets; chunk_logits (L, N, E). Returns (loss, stats). Loss is 0 (with a valid graph)
    when N < min_chunks."""
    N = u.shape[0]
    L = chunk_logits.shape[0]
    stats = {"n_chunks": int(N), "skipped": N < min_chunks}
    if N < min_chunks:
        return chunk_logits.sum() * 0.0, stats
    work = torch.float64 if u.dtype == torch.float64 else torch.float32
    u = torch.nn.functional.normalize(u.detach().to(work), dim=-1)
    S = u @ u.T
    logQ, eye = _offdiag_log_softmax(S / tau_s)
    Q = logQ.exp()
    p = torch.softmax(chunk_logits.to(work), dim=-1)  # (L, N, E)
    G = p @ p.transpose(1, 2)  # (L, N, N)
    logR, _ = _offdiag_log_softmax(G / tau_r)
    terms = (Q.unsqueeze(0) * (logQ.unsqueeze(0) - logR)).masked_fill(eye, 0.0)
    kl = terms.sum(-1)  # (L, N)
    loss = kl.mean()
    with torch.no_grad():
        off = ~eye
        stats.update({
            "kl_per_layer": kl.mean(1).tolist(),
            "q_entropy_eff": torch.exp(-(Q * logQ).masked_fill(eye, 0).sum(-1)).mean().item(),
            "r_entropy_eff": torch.exp(-(logR.exp() * logR).masked_fill(eye.unsqueeze(0), 0).sum(-1)).mean().item(),
            "chunk_router_entropy": (-(p * p.clamp_min(1e-30).log()).sum(-1)).mean().item(),
            "chunk_router_maxprob": p.max(-1).values.mean().item(),
            "chunk_argmax_mode_frac": max(
                (torch.bincount(p[l].argmax(-1), minlength=p.shape[-1]).max().item() / N) for l in range(L)),
            "S_offdiag_mean": S[off].mean().item(),
            "G_offdiag_mean": G[:, off].mean().item(),
        })
        if doc_of is not None:
            same = (doc_of.unsqueeze(0) == doc_of.unsqueeze(1)) & off
            cross = (~same) & off
            for name, m in (("same_doc", same), ("cross_doc", cross)):
                stats[f"{name}_pairs"] = int(m.sum().item())
                if m.any():
                    stats[f"S_{name}"] = S[m].mean().item()
                    stats[f"G_{name}"] = G[:, m].mean().item()
                    stats[f"Q_mass_{name}"] = (Q * m).sum(-1).mean().item()
                    stats[f"R_mass_{name}"] = (logR.exp() * m).sum(-1).mean().item()
    return loss, stats
