"""csr-speculative-pool-routing-v1: frozen candidate definitions and deterministic selection rules (pure functions).

Every rule is preregistered (docs/SPECULATIVE_POOL_ROUTING_V1_PREREG.md §8-§12). Inputs are CALIB-SEARCH results only,
except blind_selection() (CALIB-SELECTION) and the final analysis (validation).
"""
from __future__ import annotations

import json

import numpy as np

from .spv1_model import PoolRouting

MAX_UNIQUE = 24
GEN1_MAX, GEN2_MAX, GEN3_MAX = 12, 6, 3
GPU_CAP_S = 20 * 3600
USABLE_FRACTION = 0.80  # 20% safety margin: projected total must stay <= 16 GPU-hours
BUDGETS = (100_000_000, 50_000_000, 25_000_000)
SELECTION_BUDGET = 50_000_000  # Generation-3 budget (random controls fall back to it)
TIE_NATS = 0.002
GEN2_REJECT_NATS = 0.05
MAX_EXTRA_PARAM_FRACTION = 0.005
SUBSTANTIAL = 0.02
NONINF = 0.01
SPEED_MIN_GAIN = 0.10
N_BOOT = 2000
BOOT_SEED = 20261107
CAUSAL = ("fixed", "adaptive")

TOKEN = PoolRouting().to_dict()


def _r(**kw):
    return PoolRouting(**kw).validate().to_dict()


def _c(cid, fam, desc, **kw):
    return {"id": cid, "family": fam, "routing": _r(family=fam, **kw), "desc": desc}


# Generation 1 (preregistered: 4 fixed, 4 adaptive, 4 speculative; seed 98, 5M tokens)
GEN1 = [
    _c("G1-F4M8", "fixed", "fixed blocks of 4, pool 8", seg_len=4, pool_size=8),
    _c("G1-F4M16", "fixed", "fixed blocks of 4, pool 16", seg_len=4, pool_size=16),
    _c("G1-F8M8", "fixed", "fixed blocks of 8, pool 8", seg_len=8, pool_size=8),
    _c("G1-F8M16", "fixed", "fixed blocks of 8, pool 16", seg_len=8, pool_size=16),
    _c("G1-A4M16R50", "adaptive", "adaptive, max hold 4, pool 16, refresh ~50%", seg_len=4, pool_size=16,
       refresh_target=0.5),
    _c("G1-A8M16R25", "adaptive", "adaptive, max hold 8, pool 16, refresh ~25%", seg_len=8, pool_size=16,
       refresh_target=0.25),
    _c("G1-A8M8R25", "adaptive", "adaptive, max hold 8, pool 8, refresh ~25%", seg_len=8, pool_size=8,
       refresh_target=0.25),
    _c("G1-A8M16R15", "adaptive", "adaptive, max hold 8, pool 16, refresh ~12.5% region (target 0.15)", seg_len=8,
       pool_size=16, refresh_target=0.15),
    _c("G1-S4M8", "spec", "speculative block 4 (3 draft tokens), pool 8", seg_len=4, pool_size=8),
    _c("G1-S4M16", "spec", "speculative block 4 (3 draft tokens), pool 16", seg_len=4, pool_size=16),
    _c("G1-S8M8", "spec", "speculative block 8 (7 draft tokens), pool 8", seg_len=8, pool_size=8),
    _c("G1-S8M16", "spec", "speculative block 8 (7 draft tokens), pool 16", seg_len=8, pool_size=16),
]
GEN2_VARIANT = {"sel_lr_mult": 3.0, "pool_warmup_frac": 0.1}
FAMILY_TAG = {"fixed": "F", "adaptive": "A", "spec": "S"}


def routing_key(r: dict) -> str:
    return json.dumps(PoolRouting.from_cfg(r).to_dict(), sort_keys=True)


def is_valid(res: dict) -> bool:
    """Complete in every seed, finite, no collapse, extra parameters <= 0.5%."""
    return bool(res.get("complete") and res.get("ce") is not None and np.isfinite(res["ce"])
                and not res.get("collapsed", True)
                and res.get("extra_param_fraction", 0.0) <= MAX_EXTRA_PARAM_FRACTION)


def rank_key(res):
    """CE, then (near ties only, by construction of best()) lower refresh rate, fewer distinct experts per 8-token
    window, higher training throughput."""
    return (res.get("refresh_rate") if res.get("refresh_rate") is not None else 1.0,
            res.get("distinct8") if res.get("distinct8") is not None else 99.0,
            -(res.get("train_tok_s") or 0.0))


def best(cands, results):
    """Lowest mean CE; among candidates within TIE_NATS of it: lower refresh rate, fewer distinct experts per
    8-token window, higher training throughput, then id."""
    v = [c for c in cands if is_valid(results.get(c["id"], {}))]
    if not v:
        return None
    lo = min(results[c["id"]]["ce"] for c in v)
    near = [c for c in v if results[c["id"]]["ce"] <= lo + TIE_NATS]
    return sorted(near, key=lambda c: (rank_key(results[c["id"]]), results[c["id"]]["ce"], c["id"]))[0]


def gen2_candidates(gen1_results: dict):
    """Promote the strongest valid Generation-1 candidate of each family unchanged, plus one preregistered variant
    of each (selector learning-rate multiplier 3 and pool-restriction warm-up over the first 10% of steps)."""
    keep, var = [], []
    for fam in ("fixed", "adaptive", "spec"):
        b = best([c for c in GEN1 if c["family"] == fam], gen1_results)
        if b is None:
            continue
        t = FAMILY_TAG[fam]
        keep.append({"id": f"G2-{t}1", "family": fam, "routing": b["routing"], "parent": b["id"],
                     "same_as": b["id"], "change": "promoted unchanged"})
        r = dict(b["routing"])
        r.update(GEN2_VARIANT)
        var.append({"id": f"G2-{t}2", "family": fam, "routing": _r(**r), "parent": b["id"], "same_as": None,
                    "change": "variant: sel_lr_mult 3.0 + pool warm-up 10%"})
    out = keep + var
    assert len(out) <= GEN2_MAX
    return out


def gen2_reject(res_by_seed: dict, token_by_seed: dict):
    """Rejected if invalid, or worse than TOKEN-4 by more than 0.05 nats/token in BOTH seeds."""
    if not all(r["complete"] for r in res_by_seed.values()):
        return True, "incomplete / non-finite"
    if any(r["collapsed"] for r in res_by_seed.values()):
        return True, "collapse / unused expert"
    d = [res_by_seed[s]["ce"] - token_by_seed[s]["ce"] for s in res_by_seed]
    if all(x > GEN2_REJECT_NATS for x in d):
        return True, f"worse than TOKEN-4 by > {GEN2_REJECT_NATS} in both seeds ({d})"
    return False, ""


def gen3_finalists(gen2_results: dict, gen2_specs, rejected: dict):
    """Best causal (fixed/adaptive) and best speculative Generation-2 candidate, plus the best remaining one."""
    ok = [s for s in gen2_specs if not rejected.get(s["id"], (True,))[0]]
    picks = []
    for fams in (CAUSAL, ("spec",)):
        b = best([s for s in ok if s["family"] in fams], gen2_results)
        if b is not None:
            picks.append(b)
    rest = [s for s in ok if s not in picks]
    b = best(rest, gen2_results)
    if b is not None:
        picks.append(b)
    out = []
    for s in picks[:GEN3_MAX]:
        out.append({**s, "parent": s["id"], "id": s["id"].replace("G2-", "G3-"), "same_as": s["id"]})
    return out


def freeze_finalists(gen3_results: dict, gen3_specs):
    return {"causal": best([s for s in gen3_specs if s["family"] in CAUSAL], gen3_results),
            "spec": best([s for s in gen3_specs if s["family"] == "spec"], gen3_results)}


def random_control(spec: dict, role: str):
    r = dict(spec["routing"])
    r["pool_assign"] = "random"
    return {"id": f"RC-{role}", "family": spec["family"], "routing": _r(**r), "parent": spec["id"],
            "desc": f"random-pool control of {spec['id']} (same structure, content-independent load-balanced pools)"}


def blind_selection(sel: dict):
    """sel: {'TOKEN-4': agg, 'causal': agg or None, 'spec': agg or None} on CALIB-SELECTION (3-seed means). Ranking
    by CE with the near-tie rule; a finalist proceeds to final training unless invalid (failed / collapse)."""
    tok = sel["TOKEN-4"]["ce"]
    fams = {}
    for k in ("causal", "spec"):
        r = sel.get(k)
        if r is None:
            fams[k] = {"proceeds": False, "reason": "no valid frozen candidate"}
        elif not is_valid(r):
            fams[k] = {"proceeds": False, "reason": "invalid (failed / collapse / parameters)"}
        else:
            fams[k] = {"proceeds": True, "delta_vs_token": r["ce"] - tok}
    pool = {k: v for k, v in sel.items() if v is not None and is_valid(v)}
    lo = min(v["ce"] for v in pool.values())
    ranking = sorted(pool, key=lambda k: (0, rank_key(pool[k]), pool[k]["ce"]) if pool[k]["ce"] <= lo + TIE_NATS
                     else (1, (), pool[k]["ce"]))
    return {"finalists": fams, "ranking": ranking}


def choose_budget(used_gpu_s, sec_per_token: dict, principal: list, controls: list, overhead_s=0.0,
                  cap_s=GPU_CAP_S, usable=USABLE_FRACTION):
    """Largest equal final budget B (100M > 50M > 25M) with used + 3 seeds x sum_c B x s_c + overhead <= 80% of the
    cap. Options in preference order: all conditions at B; principal at B + random controls at the selection budget
    (secondary). Returns (budget, controls_budget, rows)."""
    rows = []
    opts = []
    for B in BUDGETS:
        opts.append((B, B))
        if controls and min(B, SELECTION_BUDGET) != B:
            opts.append((B, SELECTION_BUDGET))
    for B, Bc in opts:
        proj = 3 * (sum(B * sec_per_token[c] for c in principal) + sum(Bc * sec_per_token[c] for c in controls)) \
            + overhead_s
        total = used_gpu_s + proj
        rows.append({"budget": B, "controls_budget": Bc, "projected_s": proj, "used_s": used_gpu_s,
                     "total_s": total, "limit_s": usable * cap_s, "fits": total <= usable * cap_s})
    ch = next((r for r in rows if r["fits"]), None)
    return (ch["budget"], ch["controls_budget"], rows) if ch else (None, None, rows)


# ------------------------------------------------------------------ final analysis
def paired_bootstrap_multi(tok_runs, cand_runs, n_boot=N_BOOT, seed=BOOT_SEED):
    """tok_runs/cand_runs: lists (one per seed, same order) of (doc_nll dict, doc_n dict). Per-document difference of
    seed-averaged NLL sums; documents resampled (the interval reflects document sampling, not seed variation)."""
    docs = sorted(set.intersection(*[set(map(str, r[0])) for r in tok_runs + cand_runs]))
    n = np.array([tok_runs[0][1][d] for d in docs], float)
    for r in tok_runs + cand_runs:
        assert all(r[1][d] == tok_runs[0][1][d] for d in docs), "valid-token counts differ"
    T = np.mean([[r[0][d] for d in docs] for r in tok_runs], 0)
    S = np.mean([[r[0][d] for d in docs] for r in cand_runs], 0)
    diff = S - T
    rng = np.random.default_rng(seed)
    bs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(docs), len(docs))
        bs[i] = diff[idx].sum() / n[idx].sum()
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"diff": float(diff.sum() / n.sum()), "ci95": [float(lo), float(hi)], "n_docs": len(docs),
            "n_boot": n_boot, "seed": seed, "frac_docs_improved": float((diff < 0).mean()),
            "frac_docs_worse": float((diff > 0).mean())}


def criteria(per_seed_diffs, boot, collapsed, matched_tokens=True, same_active_experts=True):
    d = np.asarray(per_seed_diffs, float)
    mean = float(d.mean())
    substantial = {
        "mean_improvement_ge_0.02": mean <= -SUBSTANTIAL,
        "all_seeds_beneficial": bool((d < 0).all()),
        "bootstrap_ci_below_zero": boot["ci95"][1] < 0,
        "no_collapse_or_unused_expert": not collapsed,
        "identical_training_tokens": matched_tokens,
        "identical_active_experts": same_active_experts,
    }
    comparable = {
        "mean_degradation_le_0.01": mean <= NONINF,
        "no_seed_worse_than_0.01": bool((d <= NONINF).all()),
        "bootstrap_upper_below_0.01": boot["ci95"][1] < NONINF,
    }
    return {"mean_diff": mean, "sd_diff": float(d.std(ddof=1)) if d.size > 1 else None, "per_seed": d.tolist(),
            "substantial": substantial, "substantially_better": all(substantial.values()),
            "comparable": comparable, "accuracy_comparable": all(comparable.values())}


def speed_claim(cand_tok_s, ctrl_tok_s, accuracy_comparable):
    gain = cand_tok_s / ctrl_tok_s - 1.0
    return {"gain": gain, "ge_10pct": gain >= SPEED_MIN_GAIN, "accuracy_comparable": accuracy_comparable,
            "speed_claim": bool(gain >= SPEED_MIN_GAIN and accuracy_comparable)}
