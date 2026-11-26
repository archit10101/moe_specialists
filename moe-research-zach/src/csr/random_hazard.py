"""Fit the condition-F random closure hazard on TRAINING documents only, by matching the REALIZED (document-end-
censored) chunk-length process of semantic condition D, and compare realized distributions.

Inputs used for fitting: training documents' lengths, sentence-end positions, and D's cached chunk starts.
Inputs used when APPLYING the policy: length, sentence-end positions, document key, seed (see hazard_starts).
No embeddings, similarities or text are used anywhere in this module.
"""
from __future__ import annotations

import numpy as np

from .segment import HazardPolicy, chunk_lengths, hazard_starts


def scripted_replay(L, ends, starts, policy):
    """Replay D's boundaries through the shared eligibility state machine. Returns (replayed_starts, opportunities)
    where opportunities = list of (stratum, cur_len, closed)."""
    sset = set(int(s) for s in starts[1:])
    opp = []

    def decide(s, cur_len, t_end):
        c = (t_end + 1) in sset
        opp.append((s, cur_len, c))
        return c
    return hazard_starts(L, ends, policy, "replay", 0, decide=decide), opp


def chunk_records(L, starts, events_closed=None, max_len=256):
    """(length, is_event) per chunk. Event = closed by the threshold/random rule (non-final, length < max_len).
    Censored = document end, forced/ambiguous at max_len. Identical rule for D and F."""
    ls = chunk_lengths(L, starts)
    rec = []
    for j, ln in enumerate(ls):
        final = j == len(ls) - 1
        rec.append((int(ln), (not final) and ln < max_len))
    return rec


def bin_hazards(records_by_stratum, policy):
    """Per stratum and length bin: events / chunks at risk at the bin start (life-table hazard)."""
    nb = len(policy.hazard[0])
    lam = np.full((len(policy.hazard), nb), np.nan)
    for s, recs in records_by_stratum.items():
        if not recs:
            continue
        ln = np.array([r[0] for r in recs])
        ev = np.array([r[1] for r in recs])
        for b in range(nb):
            lo = policy.min_len + b * policy.len_bin
            hi = lo + policy.len_bin
            risk = (ln >= lo).sum()
            e = (ev & (ln >= lo) & (ln < hi)).sum()
            if risk > 0:
                lam[s, b] = e / risk
    return lam


def records(docs, starts_list, policy):
    out = {s: [] for s in range(len(policy.hazard))}
    for (L, _, _), st in zip(docs, starts_list):
        out[policy.stratum(L)].extend(chunk_records(L, st, max_len=policy.max_len))
    return out


def simulate(docs, policy, seed):
    return [hazard_starts(L, ends, policy, key, seed) for (L, ends, key) in docs]


def fit_hazard(docs, d_starts, strata_edges, len_bin=8, min_len=64, max_len=256, iters=20,
               fit_seeds=(1001, 1002, 1003, 1004), damping=0.5, select_seeds=(2001, 2002), log=None):
    """docs: list of (L, sentence_ends, doc_key) for TRAINING documents; d_starts: D's cached starts."""
    nb = (max_len - min_len) // len_bin
    pol = HazardPolicy(list(strata_edges), len_bin, min_len, max_len, [[0.0] * nb for _ in range(len(strata_edges) - 1)])
    # initial hazard: per-opportunity closure frequency of D, from the exact replay
    cnt = np.zeros((len(pol.hazard), nb))
    clo = np.zeros_like(cnt)
    n_mismatch = 0
    for (L, ends, _), st in zip(docs, d_starts):
        rep, opp = scripted_replay(L, ends, st, pol)
        n_mismatch += rep != list(st)
        for s, cl, c in opp:
            if cl < max_len:
                b = (cl - min_len) // len_bin
                cnt[s, b] += 1
                clo[s, b] += c
    h = np.where(cnt > 0, clo / np.maximum(cnt, 1), 0.0)
    pol.hazard = h.tolist()
    lam_d = bin_hazards(records(docs, d_starts, pol), pol)
    history = [{"iter": 0, "replay_mismatch_docs": int(n_mismatch)}]
    Ls = [d[0] for d in docs]
    D = describe(Ls, d_starts)
    best, best_score = None, float("inf")
    for it in range(1, iters + 1):
        lam_f = np.nanmean([bin_hazards(records(docs, simulate(docs, pol, sd), pol), pol) for sd in fit_seeds], axis=0)
        ratio = np.where((lam_f > 0) & np.isfinite(lam_d), lam_d / np.where(lam_f > 0, lam_f, 1), 1.0)
        step = 1.0 if it <= 3 else damping  # damped multiplicative update after the first iterations
        h = np.clip(np.asarray(pol.hazard) * ratio ** step, 0.0, 1.0)
        pol.hazard = h.tolist()
        # selection score on SEPARATE seeds (never the frozen application seed): worst normalized criterion,
        # excluding final_mean (structurally position-dependent; reported, not optimized)
        cm = [compare(D, describe(Ls, simulate(docs, pol, sd))) for sd in select_seeds]
        crit = {k: float(np.mean([c[k] for c in cm])) for k in ACCEPT}
        score = max(crit[k] / ACCEPT[k] for k in ACCEPT if k != "final_mean_abs")
        history.append({"iter": it, "max_abs_binhazard_gap_before_update": float(np.nanmax(np.abs(lam_f - lam_d))),
                        "select_score": score, **{f"sel_{k}": v for k, v in crit.items()}})
        if score < best_score:
            best, best_score = [list(r) for r in pol.hazard], score
        if log:
            log({k: (round(v, 4) if isinstance(v, float) else v) for k, v in history[-1].items()})
    pol.hazard = best
    history.append({"selected_score": best_score})
    return pol, history


# ----------------------------- realized-distribution comparison -----------------------------
def _ks(a, b):
    grid = np.arange(1, max(a.max(), b.max()) + 1)
    ca = np.searchsorted(np.sort(a), grid, side="right") / len(a)
    cb = np.searchsorted(np.sort(b), grid, side="right") / len(b)
    return float(np.max(np.abs(ca - cb)))


def describe(Ls, starts_list, w=16):
    allc, nonf, fin = [], [], []
    for L, st in zip(Ls, starts_list):
        ls = chunk_lengths(int(L), st)
        allc.extend(ls.tolist())
        nonf.extend(ls[:-1].tolist())
        fin.append(int(ls[-1]))
    allc, nonf, fin = map(np.array, (allc, nonf, fin))
    tot = float(np.sum(Ls))
    return {"n_chunks": int(len(allc)), "all_mean": float(allc.mean()), "all_std": float(allc.std()),
            "nonfinal_mean": float(nonf.mean()), "final_mean": float(fin.mean()),
            "all_p10_p25_p50_p75_p90": np.percentile(allc, [10, 25, 50, 75, 90]).tolist(),
            "nonfinal_p50_p75_p90": np.percentile(nonf, [50, 75, 90]).tolist(),
            "min_max": [int(allc.min()), int(allc.max())], "pct_exactly_128": 100 * float((allc == 128).mean()),
            "mass_241_256_pct": 100 * float(((allc >= 241) & (allc <= 256)).mean()),
            "pool_eligible_fraction": float(np.maximum(allc - w, 0).sum() / tot),
            "_all": allc, "_nonfinal": nonf}


ACCEPT = {"nonfinal_mean_abs": 1.0, "all_mean_abs": 1.0, "chunk_count_rel": 0.01, "eligible_frac_abs_pp": 0.5,
          "ks_all": 0.02, "ks_nonfinal": 0.02, "quantile_abs_tokens": 5.0, "final_mean_abs": 3.0,
          "mass_241_256_abs_pp": 1.0}


def compare(d, f):
    q_gap = max(abs(x - y) for x, y in zip(d["all_p10_p25_p50_p75_p90"][2:], f["all_p10_p25_p50_p75_p90"][2:]))
    m = {"nonfinal_mean_abs": abs(d["nonfinal_mean"] - f["nonfinal_mean"]),
         "all_mean_abs": abs(d["all_mean"] - f["all_mean"]),
         "chunk_count_rel": abs(d["n_chunks"] - f["n_chunks"]) / d["n_chunks"],
         "eligible_frac_abs_pp": 100 * abs(d["pool_eligible_fraction"] - f["pool_eligible_fraction"]),
         "ks_all": _ks(d["_all"], f["_all"]), "ks_nonfinal": _ks(d["_nonfinal"], f["_nonfinal"]),
         "quantile_abs_tokens": float(q_gap), "final_mean_abs": abs(d["final_mean"] - f["final_mean"]),
         "mass_241_256_abs_pp": abs(d["mass_241_256_pct"] - f["mass_241_256_pct"])}
    m["pass"] = {k: bool(m[k] < ACCEPT[k]) for k in ACCEPT}
    m["all_pass"] = all(m["pass"].values())
    return m
