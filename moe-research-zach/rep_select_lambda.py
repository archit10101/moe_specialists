#!/usr/bin/env python
"""csr-representative-v1 lambda selection from the five disposable E tuning runs (docs/REP_V1_PREREG.md §6).
Training/calibration data only. CPU, read-only on the tuning runs; writes a new selection JSON.

Smallest lambda satisfying all of:
 1. median weighted-semantic / LM router-gradient ratio over the final 20% of tuning steps in [0.05, 0.20];
 2. mean logged semantic loss over the final 20% of steps < mean over the first 20%;
 3. status COMPLETE (no non-finite value);
 4. no MoE layer with > 50% of calib routing assignments on one expert;
 5. mean training LM CE over the final 20% <= that of the next smaller candidate + 0.02 (vacuous for the smallest).
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

GRID = [0.0001, 0.0003, 0.001, 0.003, 0.01]
RATIO_LO, RATIO_HI, LM_TOL, COLLAPSE = 0.05, 0.20, 0.02, 0.5


def summarize_run(status, n_steps, step_records, calib_eval, n_experts):
    """step_records: dicts with step, lm_ce, sem, router_gradnorm_ratio (may be None)."""
    import numpy as np
    early = [r for r in step_records if r["step"] < 0.2 * n_steps]
    final = [r for r in step_records if r["step"] >= 0.8 * n_steps]
    ratios = [r["router_gradnorm_ratio"] for r in final if r.get("router_gradnorm_ratio") is not None]
    vals = [r[k] for r in step_records for k in ("lm_ce", "sem") if r.get(k) is not None]
    finite = all(math.isfinite(v) for v in vals) and all(math.isfinite(x) for x in ratios)
    shares = [m / n_experts for m in calib_eval["expert_load_max_over_mean"]] if calib_eval else []
    return {"status": status, "n_steps": n_steps, "n_final_ratio_samples": len(ratios),
            "median_ratio_final20": float(np.median(ratios)) if ratios else None,
            "sem_early20_mean": float(np.mean([r["sem"] for r in early])) if early else None,
            "sem_final20_mean": float(np.mean([r["sem"] for r in final])) if final else None,
            "lm_final20_mean": float(np.mean([r["lm_ce"] for r in final])) if final else None,
            "calib_max_expert_share": shares, "finite": finite,
            "collapse": any(x > COLLAPSE for x in shares) if shares else None}


def select_lambda(summ):
    """summ: {lambda: summary}. Returns (chosen lambda or None, per-candidate criteria)."""
    crit = {}
    for i, lam in enumerate(GRID):
        s = summ.get(lam)
        if s is None:
            crit[lam] = {"missing": True, "all_pass": False}
            continue
        c = {"ratio_in_band": s["median_ratio_final20"] is not None and RATIO_LO <= s["median_ratio_final20"] <= RATIO_HI,
             "sem_decreases": s["sem_final20_mean"] is not None and s["sem_early20_mean"] is not None
             and s["sem_final20_mean"] < s["sem_early20_mean"],
             "complete_finite": s["status"] == "COMPLETE" and s["finite"],
             "no_collapse": s["collapse"] is False}
        if i == 0:
            c["lm_vs_next_smaller"] = True
            c["lm_vs_next_smaller_note"] = "vacuous (smallest candidate)"
        else:
            prev = summ.get(GRID[i - 1])
            c["lm_vs_next_smaller"] = bool(prev and prev["lm_final20_mean"] is not None and s["lm_final20_mean"] is not None
                                           and s["lm_final20_mean"] - prev["lm_final20_mean"] <= LM_TOL)
            if prev and prev["lm_final20_mean"] is not None and s["lm_final20_mean"] is not None:
                c["lm_delta_vs_next_smaller"] = s["lm_final20_mean"] - prev["lm_final20_mean"]
        c["all_pass"] = all(v for k, v in c.items() if isinstance(v, bool))
        crit[lam] = c
    chosen = next((lam for lam in GRID if crit[lam]["all_pass"]), None)
    return chosen, crit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="TSV: lambda<TAB>run_dir<TAB>calib_eval_json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from csr import guard
    guard.check_root()
    summ, paths = {}, {}
    for line in Path(a.runs).read_text().splitlines():
        lam_s, rd, ev = line.split("\t")
        lam, rd = float(lam_s), Path(rd)
        st = json.loads((rd / "summary.json").read_text())
        cfg = json.loads((rd / "config.json").read_text())
        assert cfg["semantic"]["lambda"] == lam and cfg["condition"] == "E" and cfg["init"]["seed"] == 99
        recs = [json.loads(x) for x in (rd / "metrics.jsonl").read_text().splitlines()]
        steps = [r for r in recs if "step" in r]
        ce = json.loads(Path(ev).read_text()) if ev and Path(ev).exists() else None
        if ce is not None:
            assert ce["split"] == "calib" and ce["status"] == "COMPLETE"
        summ[lam] = summarize_run(st["status"], st.get("steps", 0), steps, ce, cfg["model"]["n_experts"])
        summ[lam]["trajectory"] = [{k: r.get(k) for k in ("step", "lm_ce", "sem", "router_gradnorm_ratio")} for r in steps]
        paths[lam] = {"run": str(rd), "calib_eval": ev}
    chosen, crit = select_lambda(summ)
    out = {"chosen_lambda": chosen, "criteria": {str(k): v for k, v in crit.items()},
           "summaries": {str(k): v for k, v in summ.items()}, "paths": {str(k): v for k, v in paths.items()},
           "rule": __doc__, "validation_accessed": False}
    guard.atomic_write_json(guard.inside_root(a.out), out)
    guard.ledger("REP_V1_LAMBDA", f"chosen lambda={chosen} -> {a.out}")
    print(json.dumps({"chosen_lambda": chosen, "criteria": out["criteria"]}, indent=1, default=str))
    sys.exit(0 if chosen is not None else 1)


if __name__ == "__main__":
    main()
