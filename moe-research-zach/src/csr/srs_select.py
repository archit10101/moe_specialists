"""csr-sequence-routing-search-v1: frozen candidate definitions and deterministic selection rules (pure functions).

Every rule here is preregistered (docs/SEQUENCE_ROUTING_SEARCH_V1_PREREG.md §6-§10). Inputs are CALIB-SEARCH results
only, except promote() (CALIB-SELECTION) and final_analysis() (validation).
"""
from __future__ import annotations

import copy
import hashlib
import json

import numpy as np

MAX_UNIQUE = 24
GEN1_MAX, GEN2_MAX = 12, 8
GPU_CAP_S = 20 * 3600
SAFETY = 1.2
BUDGETS = (100_000_000, 50_000_000, 25_000_000)
TIE_NATS = 0.002
CATASTROPHIC_NATS = 0.10
MAX_EXTRA_PARAM_FRACTION = 0.005
MAX_EXTRA_COMPUTE = 0.10
SUBSTANTIAL = 0.02
NONINF = 0.01
N_BOOT = 2000
BOOT_SEED = 20261106

TOKEN = {"mode": "token"}


def _seq(**kw):
    r = {"mode": "sequence", "seg": "fixed", "seg_len": 64, "opening": 0, "target_len": 64, "boundary_source": "layer",
         "summary": "mean", "state": "none", "temperature": 1.0, "lb_weighting": "token"}
    r.update(kw)
    return r


# Generation 1 (preregistered, 12 candidates: 7 fixed-family, 5 dynamic-family)
GEN1 = [
    {"id": "G1-F16", "family": "fixed", "routing": _seq(seg="fixed", seg_len=16), "desc": "fixed 16, mean"},
    {"id": "G1-F32", "family": "fixed", "routing": _seq(seg="fixed", seg_len=32), "desc": "fixed 32, mean"},
    {"id": "G1-F64", "family": "fixed", "routing": _seq(seg="fixed", seg_len=64), "desc": "fixed 64, mean"},
    {"id": "G1-F128", "family": "fixed", "routing": _seq(seg="fixed", seg_len=128), "desc": "fixed 128, mean"},
    {"id": "G1-F256", "family": "fixed", "routing": _seq(seg="fixed", seg_len=256), "desc": "fixed 256, mean"},
    {"id": "G1-F32-GRU", "family": "fixed", "routing": _seq(seg="fixed", seg_len=32, state="gru"),
     "desc": "fixed 32, mean + GRU over [W pool, previous route]"},
    {"id": "G1-F32-ATTN", "family": "fixed", "routing": _seq(seg="fixed", seg_len=32, summary="attn"),
     "desc": "fixed 32, learned-query attention pooling"},
    {"id": "G1-CP32", "family": "dynamic", "routing": _seq(seg="cp", target_len=32),
     "desc": "per-layer change point (h_l), target 32"},
    {"id": "G1-CP64", "family": "dynamic", "routing": _seq(seg="cp", target_len=64),
     "desc": "per-layer change point (h_l), target 64"},
    {"id": "G1-CP128", "family": "dynamic", "routing": _seq(seg="cp", target_len=128),
     "desc": "per-layer change point (h_l), target 128"},
    {"id": "G1-CP64-L0", "family": "dynamic", "routing": _seq(seg="cp", target_len=64, boundary_source="layer0"),
     "desc": "shared change point on the dense-layer-0 output, target 64"},
    {"id": "G1-LBD64", "family": "dynamic", "routing": _seq(seg="lbd", target_len=64),
     "desc": "learned per-layer boundary detector, target 64"},
]
F32_SUMMARY_PROBES = {"G1-F32": ("mean", "none"), "G1-F32-GRU": ("mean", "gru"), "G1-F32-ATTN": ("attn", "none")}


def routing_key(r: dict) -> str:
    return hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest()[:16]


# ------------------------------------------------------------------ validity
def is_valid(res: dict) -> bool:
    """res: {'complete', 'ce', 'collapsed', 'extra_param_fraction'} for one run (or aggregated over seeds)."""
    return (bool(res.get("complete")) and res.get("ce") is not None and np.isfinite(res["ce"])
            and not res.get("collapsed", True) and res.get("extra_param_fraction", 1.0) <= MAX_EXTRA_PARAM_FRACTION)


def rank_family(specs, results, family):
    """Valid candidates of a family sorted by CALIB-SEARCH CE (ties: candidate id)."""
    c = [s for s in specs if s["family"] == family and is_valid(results.get(s["id"], {}))]
    return sorted(c, key=lambda s: (results[s["id"]]["ce"], s["id"]))


def _opening_applicable(r):
    if r.get("opening", 0):
        return False
    return r["seg"] in ("cp", "lbd") or r.get("seg_len", 0) > 16


def gen2_candidates(gen1_results: dict, gen1_specs=GEN1):
    """Deterministic Generation-2 rule (prereg §7). Returns <= 8 specs (4 fixed + 4 dynamic directions)."""
    known = {routing_key(s["routing"]): s["id"] for s in gen1_specs}
    probes = [(gen1_results[k]["ce"], v) for k, v in F32_SUMMARY_PROBES.items() if is_valid(gen1_results.get(k, {}))]
    summary_pref = [v for _, v in sorted(probes)]
    out = {}
    for fam in ("fixed", "dynamic"):
        ranked = rank_family(gen1_specs, gen1_results, fam)
        dirs = []

        def add(spec_routing, parent, change, fam=fam):
            key = routing_key(spec_routing)
            if any(routing_key(d["routing"]) == key for d in dirs):
                return False
            dirs.append({"family": fam, "routing": spec_routing, "parent": parent, "change": change,
                         "same_as": known.get(key)})
            return True

        if ranked:
            b1 = ranked[0]
            add(copy.deepcopy(b1["routing"]), b1["id"], "carry (best)")
            if len(ranked) > 1:
                add(copy.deepcopy(ranked[1]["routing"]), ranked[1]["id"], "carry (second best)")
            r = copy.deepcopy(b1["routing"])
            if _opening_applicable(r):
                r["opening"] = 16
                add(r, b1["id"], "best + 16-token opening segment")
            else:
                r["state"] = "ema" if r["state"] != "ema" else "none"
                add(r, b1["id"], f"best + state={r['state']} (opening not applicable)")
            own = (b1["routing"]["summary"], b1["routing"]["state"])
            for sc in summary_pref:
                if sc == own:
                    continue
                r = copy.deepcopy(b1["routing"])
                r["summary"], r["state"] = sc
                if add(r, b1["id"], f"best + summary={sc[0]} state={sc[1]} (best F32 summary probe)"):
                    break
            extras = []
            for b in ranked[1:4]:
                rr = copy.deepcopy(b["routing"])
                if _opening_applicable(rr):
                    rr["opening"] = 16
                    extras.append((rr, b["id"], "+ 16-token opening segment"))
            for b in ranked[2:6]:
                extras.append((copy.deepcopy(b["routing"]), b["id"], "carry"))
            out[fam] = (dirs, extras)
        else:
            out[fam] = ([], [])
    # balance: 4 per family; a family with fewer valid directions is filled from the other family's extras
    final = []
    for fam, other in (("fixed", "dynamic"), ("dynamic", "fixed")):
        final += out[fam][0][:4]
    for fam, other in (("fixed", "dynamic"), ("dynamic", "fixed")):
        short = 4 - min(4, len(out[fam][0]))
        for rr, pid, ch in out[other][1]:
            if short <= 0 or len(final) >= GEN2_MAX:
                break
            if any(routing_key(d["routing"]) == routing_key(rr) for d in final):
                continue
            final.append({"family": other, "routing": rr, "parent": pid, "change": ch + " (family balance fill)",
                          "same_as": known.get(routing_key(rr))})
            short -= 1
    final = final[:GEN2_MAX]
    n = {"fixed": 0, "dynamic": 0}
    for d in final:
        n[d["family"]] += 1
        d["id"] = f"G2-{'F' if d['family'] == 'fixed' else 'D'}{n[d['family']]}"
    return final


def gen3_finalists(gen2_results: dict, gen2_specs):
    """Top two valid candidates per family by mean CALIB-SEARCH CE over seeds 98 and 99 (prereg §8)."""
    out = []
    for fam in ("fixed", "dynamic"):
        out += rank_family(gen2_specs, gen2_results, fam)[:2]
    return out


def best_with_tiebreak(cands, results):
    """Lowest CE; among candidates within TIE_NATS of the lowest CE, the lowest GPU seconds."""
    v = [c for c in cands if is_valid(results.get(c["id"], {}))]
    if not v:
        return None
    lo = min(results[c["id"]]["ce"] for c in v)
    near = [c for c in v if results[c["id"]]["ce"] <= lo + TIE_NATS]
    return sorted(near, key=lambda c: (results[c["id"]].get("gpu_s", 0.0), results[c["id"]]["ce"], c["id"]))[0]


def freeze_best(gen3_results: dict, gen3_specs):
    return {fam: best_with_tiebreak([s for s in gen3_specs if s["family"] == fam], gen3_results)
            for fam in ("fixed", "dynamic")}


def promote(sel: dict):
    """sel: {'TOKEN': {...}, 'fixed': {...} or None, 'dynamic': {...} or None}; each {'ce','collapsed','complete',
    'gpu_s', 'extra_param_fraction'} on CALIB-SELECTION (3-seed mean). Promotion is never replaced by another
    candidate (prereg §9)."""
    tok = sel["TOKEN"]["ce"]
    res = {}
    for fam in ("fixed", "dynamic"):
        r = sel.get(fam)
        if r is None:
            res[fam] = {"promoted": False, "reason": "no valid frozen candidate"}
        elif not is_valid(r):
            res[fam] = {"promoted": False, "reason": "invalid (failed / collapse / parameters)"}
        elif r["ce"] - tok > CATASTROPHIC_NATS:
            res[fam] = {"promoted": False, "reason": f"catastrophic: {r['ce'] - tok:+.4f} > +{CATASTROPHIC_NATS}"}
        else:
            res[fam] = {"promoted": True, "delta_vs_token": r["ce"] - tok}
    ranked = [f for f in ("fixed", "dynamic") if sel.get(f) is not None and is_valid(sel[f])]
    lo = min((sel[f]["ce"] for f in ranked), default=None)
    ranked = sorted(ranked, key=lambda f: (0 if sel[f]["ce"] <= lo + TIE_NATS else 1,
                                           sel[f]["gpu_s"] if sel[f]["ce"] <= lo + TIE_NATS else sel[f]["ce"]))
    return {"families": res, "ranking": ranked}


def choose_budget(used_gpu_s, sec_per_token: dict, n_seeds=3, overhead_s=0.0, cap_s=GPU_CAP_S, safety=SAFETY):
    """Largest budget B with used + safety * (n_seeds * sum_c B * sec_per_token[c] + overhead) <= cap."""
    rows = []
    for B in BUDGETS:
        proj = n_seeds * sum(B * s for s in sec_per_token.values()) + overhead_s
        total = used_gpu_s + safety * proj
        rows.append({"budget": B, "projected_s": proj, "used_s": used_gpu_s, "total_with_margin_s": total,
                     "fits": total <= cap_s})
    chosen = next((r["budget"] for r in rows if r["fits"]), None)
    return chosen, rows


# ------------------------------------------------------------------ final analysis
def paired_bootstrap_multi(tok_runs, seq_runs, n_boot=N_BOOT, seed=BOOT_SEED):
    """tok_runs/seq_runs: lists (one per seed, same order) of (doc_nll dict, doc_n dict). Per-document difference of
    the seed-averaged NLL sums; documents resampled (seed variation is NOT part of this interval)."""
    docs = sorted(set.intersection(*[set(map(str, r[0])) for r in tok_runs + seq_runs]))
    n = np.array([tok_runs[0][1][d] for d in docs], float)
    for r in tok_runs + seq_runs:
        assert all(r[1][d] == tok_runs[0][1][d] for d in docs), "valid-token counts differ"
    T = np.mean([[r[0][d] for d in docs] for r in tok_runs], 0)
    S = np.mean([[r[0][d] for d in docs] for r in seq_runs], 0)
    diff = S - T
    rng = np.random.default_rng(seed)
    bs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(docs), len(docs))
        bs[i] = diff[idx].sum() / n[idx].sum()
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"diff": float(diff.sum() / n.sum()), "ci95": [float(lo), float(hi)], "n_docs": len(docs),
            "n_boot": n_boot, "seed": seed}


def criteria(per_seed_diffs, boot, collapsed, extra_compute_frac, same_algorithm=True, matched_tokens=True,
             same_active_experts=True):
    d = np.asarray(per_seed_diffs, float)
    mean = float(d.mean())
    substantial = {
        "mean_improvement_ge_0.02": mean <= -SUBSTANTIAL,
        "all_seeds_beneficial": bool((d < 0).all()),
        "bootstrap_ci_below_zero": boot["ci95"][1] < 0,
        "no_collapse": not collapsed,
        "no_more_active_experts": same_active_experts,
        "matched_training_tokens": matched_tokens,
        "extra_compute_le_10pct": extra_compute_frac <= MAX_EXTRA_COMPUTE,
        "same_algorithm_train_inference": same_algorithm,
    }
    comparable = {
        "mean_degradation_le_0.01": mean <= NONINF,
        "every_seed_within_0.01": bool((d <= NONINF).all()),
        "bootstrap_upper_below_0.01": boot["ci95"][1] < NONINF,
    }
    return {"mean_diff": mean, "sd_diff": float(d.std(ddof=1)) if d.size > 1 else None, "per_seed": d.tolist(),
            "substantial": substantial, "substantially_better": all(substantial.values()),
            "comparable": comparable, "accuracy_comparable": all(comparable.values())}
