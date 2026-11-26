"""Pure-numpy metric helpers shared by every evaluation."""
from __future__ import annotations

import math

import numpy as np


def nmi(a: np.ndarray, b: np.ndarray, na: int, nb: int) -> float:
    """Normalised mutual information I(a;b)/sqrt(H(a)H(b)) of two label arrays."""
    j = np.zeros((na, nb))
    np.add.at(j, (np.asarray(a, np.int64), np.asarray(b, np.int64)), 1)
    return nmi_from_counts(j)


def nmi_from_counts(j: np.ndarray) -> float:
    j = j / max(j.sum(), 1e-12)
    pa, pb = j.sum(1), j.sum(0)
    nz = j > 0
    mi = (j[nz] * np.log(j[nz] / np.outer(pa, pb)[nz])).sum()
    ha = -(pa[pa > 0] * np.log(pa[pa > 0])).sum()
    hb = -(pb[pb > 0] * np.log(pb[pb > 0])).sum()
    return float(mi / max(1e-12, math.sqrt(ha * hb)))


def quantile_bins(v: np.ndarray, k: int) -> np.ndarray:
    """Bin index 0..k-1 by quantiles of v."""
    if len(v) == 0:
        return np.zeros(0, np.int64)
    edges = np.quantile(v, np.linspace(0, 1, k + 1)[1:-1])
    return np.searchsorted(edges, v, side="right")


def matched_difference(delta: np.ndarray, own: np.ndarray, bins: np.ndarray, min_per_bin: int = 3):
    """Exposure-matched contrast: inside each covariate bin, mean(delta | own) - mean(delta | other),
    averaged over bins with weights = number of own tokens. Returns (value, coverage) or (nan, 0)."""
    diffs, w = [], []
    for b in np.unique(bins[own]):
        o, c = own & (bins == b), ~own & (bins == b)
        if o.sum() >= min_per_bin and c.sum() >= min_per_bin:
            diffs.append(delta[o].mean() - delta[c].mean())
            w.append(o.sum())
    if not w:
        return float("nan"), 0.0
    return float(np.average(diffs, weights=w)), float(sum(w) / max(1, own.sum()))


def summarize(values) -> dict:
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
    if len(v) == 0:
        return dict(mean=float("nan"), se=float("nan"), n=0, pos_frac=float("nan"))
    se = float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else float("nan")
    return dict(mean=float(v.mean()), se=se, n=int(len(v)), pos_frac=float((v > 0).mean()))


def double_centred_diag(M: np.ndarray) -> float:
    """Specificity of a swap matrix M[group, specialist] of CE: mean diagonal of the double-centred
    utility (-M). Removes 'some groups are harder' and 'some specialists are better overall', so a
    bank with no group-specific function scores ~0."""
    U = -M
    Uc = U - U.mean(1, keepdims=True) - U.mean(0, keepdims=True) + U.mean()
    return float(np.mean(np.diag(Uc)))


def swap_specificity(ce: np.ndarray, group: np.ndarray, paper: np.ndarray, k: int,
                     n_boot: int = 200, seed: int = 0) -> dict:
    """ce [N, k] CE of every token under every specialist; group [N] in 0..k-1; paper [N] ids.
    Returns swap matrix, specificity, own-best fraction and a hierarchical (paper-level) bootstrap CI."""
    papers, pinv = np.unique(paper, return_inverse=True)
    P = len(papers)
    S = np.zeros((P, k))
    np.add.at(S, pinv, ce)
    cnt = np.bincount(pinv, minlength=P).astype(np.float64)
    pg = np.zeros(P, np.int64)
    pg[pinv] = group

    def matrix(idx):
        M = np.zeros((k, k))
        for g in range(k):
            sel = idx[pg[idx] == g]
            if len(sel):
                M[g] = S[sel].sum(0) / cnt[sel].sum()
            else:
                M[g] = np.nan
        return M

    full = matrix(np.arange(P))
    ok = ~np.isnan(full).any(1)
    Mf = full[ok][:, ok] if ok.sum() >= 2 else full
    spec = double_centred_diag(Mf)
    own_best = float(np.mean([np.argmin(full[g]) == g for g in range(k) if ok[g]]))
    rng = np.random.default_rng(seed)
    boots = []
    by_group = [np.flatnonzero(pg == g) for g in range(k)]
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(ix, len(ix)) for ix in by_group if len(ix)])
        Mb = matrix(idx)
        okb = ~np.isnan(Mb).any(1)
        if okb.sum() >= 2:
            boots.append(double_centred_diag(Mb[okb][:, okb]))
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (float("nan"), float("nan")))
    return dict(specificity=spec, ci=[float(lo), float(hi)], own_best_frac=own_best,
                swap_matrix=full.tolist())


def paired(a: dict, b: dict) -> dict:
    """Paired per-seed difference a - b over the seeds both dicts have."""
    seeds = sorted(set(a) & set(b))
    d = [a[s] - b[s] for s in seeds]
    out = summarize(d)
    out["per_seed"] = {str(s): float(x) for s, x in zip(seeds, d)}
    out["neg_frac"] = float(np.mean([x < 0 for x in d])) if d else float("nan")
    if len(d) > 1:
        from scipy import stats
        t = stats.t.ppf(0.975, len(d) - 1)
        out["ci95"] = [out["mean"] - t * out["se"], out["mean"] + t * out["se"]]
    return out
