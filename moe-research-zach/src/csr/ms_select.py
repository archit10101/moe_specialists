"""csr-multiscale-routing-v1 decision rules (pure functions; docs/MULTISCALE_ROUTING_V1_PREREG.md §7-§11)."""
from __future__ import annotations

import math

import numpy as np

GRID = (0.0, 0.25, 0.5, 1.0)
REJECT_WORSE = 0.10
MIN_GAIN = 0.002
CATASTROPHIC = 0.05
TIE = 1e-4
COLLAPSE_SHARE = 0.5
BUDGETS = (100_000_000, 50_000_000, 25_000_000)
GPU_CAP_S = 20 * 3600
MARGIN = 1.2
ANALYSIS_ALLOWANCE_S = 3.0 * 3600
VRAM_LIMIT_GIB = 56.0
DISK_MIN_BYTES = 80 * 1024**3
PILOT_TOKENS = 10_000_000
FUTILITY = 0.05
SUPPORT = -0.01


def select_coefficient(results, grid=GRID):
    """results: list of dicts {coef, seed, placement, ce, finite(bool), status, max_share[list], unused[list]}.
    Returns (chosen coefficient, per-candidate table)."""
    zero = {(r["seed"], r["placement"]): r for r in results if r["coef"] == 0.0}
    cells = sorted(zero)
    table = {}
    for c in grid:
        runs = {(r["seed"], r["placement"]): r for r in results if r["coef"] == c}
        t = {"coef": c, "n_runs": len(runs), "reasons": []}
        if sorted(runs) != cells or not cells:
            t["reasons"].append("incomplete design")
            t.update(rejected=True, qualifies=False)
            table[c] = t
            continue
        for key, r in sorted(runs.items()):
            z = zero[key]
            tag = f"seed{key[0]}/{key[1]}"
            if r["status"] != "COMPLETE" or not r["finite"] or not math.isfinite(r["ce"]):
                t["reasons"].append(f"{tag}: non-finite or not COMPLETE")
                continue
            if max(r["max_share"]) > COLLAPSE_SHARE:
                t["reasons"].append(f"{tag}: collapse (max share {max(r['max_share']):.3f})")
            for li, (u, u0) in enumerate(zip(r["unused"], z["unused"])):
                if u > 0 and u0 == 0:
                    t["reasons"].append(f"{tag}: {u} unused expert(s) in layer {li + 1} (zero run: none)")
            if r["ce"] - z["ce"] > REJECT_WORSE:
                t["reasons"].append(f"{tag}: {r['ce'] - z['ce']:+.4f} worse than zero (> {REJECT_WORSE})")
        t["rejected"] = bool(t["reasons"])
        ces = [runs[k]["ce"] for k in cells]
        t["aggregate_ce"] = float(np.mean(ces))
        t["diff_vs_zero"] = float(np.mean([runs[k]["ce"] - zero[k]["ce"] for k in cells]))
        seeds = sorted({k[0] for k in cells})
        places = sorted({k[1] for k in cells})
        t["seed_diff"] = {str(s): float(np.mean([runs[k]["ce"] - zero[k]["ce"] for k in cells if k[0] == s])) for s in seeds}
        t["placement_diff"] = {p: float(np.mean([runs[k]["ce"] - zero[k]["ce"] for k in cells if k[1] == p]))
                               for p in places}
        t["per_run"] = {f"seed{k[0]}/{k[1]}": {"ce": runs[k]["ce"], "diff_vs_zero": runs[k]["ce"] - zero[k]["ce"]}
                        for k in cells}
        if c == 0.0:
            t["qualifies"] = False
        else:
            t["gain_ok"] = t["diff_vs_zero"] <= -MIN_GAIN
            t["both_seeds_beneficial"] = all(v < 0 for v in t["seed_diff"].values())
            t["no_catastrophic_placement"] = all(v <= CATASTROPHIC for v in t["placement_diff"].values())
            t["qualifies"] = (not t["rejected"]) and t["gain_ok"] and t["both_seeds_beneficial"] and \
                t["no_catastrophic_placement"]
        table[c] = t
    q = [c for c in grid if c != 0.0 and table[c]["qualifies"]]
    if not q:
        return 0.0, table
    best = min(table[c]["aggregate_ce"] for c in q)
    chosen = min(c for c in q if table[c]["aggregate_ce"] - best < TIE)
    return chosen, table


def select_budget(consumed_s, tps, overhead_s, val_eval_s, peak_reserved_gib, disk_free_bytes, disk_per_run_bytes,
                  n_seeds=3, analysis_s=ANALYSIS_ALLOWANCE_S, budgets=BUDGETS, pilot_tokens=PILOT_TOKENS):
    """tps: {condition: valid tokens/s}. Returns (chosen budget or None, table)."""
    conds = sorted(tps)
    pilot = sum(pilot_tokens / tps[c] + overhead_s for c in conds)
    table = []
    for B in budgets:
        main = n_seeds * sum(B / tps[c] + overhead_s + val_eval_s for c in conds)
        total = MARGIN * (consumed_s + pilot + main + analysis_s)
        disk_after = disk_free_bytes - (len(conds) * (n_seeds + 1)) * disk_per_run_bytes
        ok = total <= GPU_CAP_S and peak_reserved_gib < VRAM_LIMIT_GIB and disk_after >= DISK_MIN_BYTES
        table.append({"budget": B, "consumed_s": consumed_s, "pilot_s": pilot, "main_s": main,
                      "analysis_s": analysis_s, "projected_total_s_with_margin": total,
                      "projected_total_gpu_h_with_margin": total / 3600, "disk_free_after_bytes": disk_after,
                      "peak_reserved_gib": peak_reserved_gib, "fits": ok})
    chosen = next((r["budget"] for r in table if r["fits"]), None)
    return chosen, table


def ols_slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2:
        return float("nan")
    xm = x - x.mean()
    return float((xm * (y - y.mean())).sum() / (xm * xm).sum())


def final20_slope(records, n_steps):
    r = [q for q in records if q["step"] >= 0.8 * n_steps]
    return ols_slope([q["valid_tokens_seen"] for q in r], [q["lm_ce"] for q in r])


def pilot_futility(ce, slopes, contextual, correctness_issue=False):
    """ce, slopes: {condition: value} including TOKEN. Stop iff every contextual condition is > FUTILITY worse than
    TOKEN, none has a more negative final-20% slope than TOKEN, and no correctness issue explains it."""
    ctx = [c for c in ce if c in contextual]
    all_worse = bool(ctx) and all(ce[c] - ce["TOKEN"] > FUTILITY for c in ctx)
    none_better_slope = all(not (slopes[c] < slopes["TOKEN"]) for c in ctx)
    stop = all_worse and none_better_slope and not correctness_issue
    return stop, {"contextual": ctx, "all_contextual_worse_than_0.05": all_worse,
                  "no_better_final20_slope": none_better_slope, "correctness_issue": correctness_issue,
                  "diff_vs_token": {c: ce[c] - ce["TOKEN"] for c in ce},
                  "slope_minus_token": {c: slopes[c] - slopes["TOKEN"] for c in slopes}}


def paired_bootstrap_multi(doc_x, doc_y, doc_n, n_boot=2000, seed=0):
    """doc_x, doc_y: {seed: np.array per-document NLL sums over a common document order}; doc_n: per-document valid
    counts (identical for every run, asserted by the caller). One resampled document index vector per bootstrap draw
    is applied to every seed. Returns per-seed and three-seed-mean point estimates and percentile intervals."""
    seeds = sorted(doc_x)
    N = np.asarray(doc_n, float)
    D = {s: np.asarray(doc_x[s], float) - np.asarray(doc_y[s], float) for s in seeds}
    rng = np.random.default_rng(seed)
    nd = len(N)
    boots = {s: np.empty(n_boot) for s in seeds}
    for b in range(n_boot):
        i = rng.integers(0, nd, nd)
        tot = N[i].sum()
        for s in seeds:
            boots[s][b] = D[s][i].sum() / tot
    point = {s: float(D[s].sum() / N.sum()) for s in seeds}
    mean_boot = np.mean([boots[s] for s in seeds], axis=0)
    out = {"per_seed": {str(s): {"diff": point[s], "ci95": np.percentile(boots[s], [2.5, 97.5]).tolist()} for s in seeds},
           "mean_diff": float(np.mean(list(point.values()))), "mean_ci95": np.percentile(mean_boot, [2.5, 97.5]).tolist(),
           "seed_sd": float(np.std(list(point.values()), ddof=1)) if len(seeds) > 1 else 0.0,
           "seed_range": [float(min(point.values())), float(max(point.values()))], "n_docs": nd, "n_boot": n_boot}
    out["support"] = bool(out["mean_diff"] <= SUPPORT and all(v < 0 for v in point.values()) and out["mean_ci95"][1] < 0)
    out["non_inferior"] = bool(out["mean_diff"] <= 0.005 and out["mean_ci95"][1] < 0.01)
    return out
