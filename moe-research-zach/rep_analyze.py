#!/usr/bin/env python
"""csr-representative-v1 analysis (docs/REP_V1_PREREG.md §8). CPU, read-only on runs; writes one new JSON.

Per seed and contrast: row-minus-column token-weighted validation CE, paired document bootstrap (2,000, seed 0).
Prospective criterion: same sign in all 3 seeds, mean |delta| >= 0.01, CI excludes 0 in every seed.
H1: B-A layer-mean adjacent-token Jaccard > 0 in every seed and B-A CE <= +0.02 in every seed.
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402
from csr.config import check_matched  # noqa: E402
from csr.train_lib import paired_bootstrap  # noqa: E402

CONTRASTS = [("D", "F", "H2"), ("E", "D", "H3"), ("B", "A", "H1 support"), ("C", "B", "semantic loss, fixed chunks"),
             ("E", "C", "semantic vs fixed, with loss"), ("D", "B", "secondary semantic vs fixed")]
DESCRIPTIVE = [("E", "A", "descriptive only")]


def tokens_to_target(evals, target):
    """First crossing of the monitoring-subset CE below target, linear interpolation in tokens."""
    prev = None
    for e in evals:
        x, y = e["valid_tokens_seen"], e["ce_nats_per_token"]
        if y <= target:
            if prev is None:
                return float(x)
            x0, y0 = prev
            return float(x0 + (x - x0) * (y0 - target) / (y0 - y)) if y0 != y else float(x)
        prev = (x, y)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=".tmp/rep_v1_main/runs.tsv")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    guard.check_root()
    import numpy as np
    rows = [l.split("\t") for l in Path(a.runs).read_text().splitlines() if l.strip()]
    runs = {(int(s), c): (Path(r), Path(e.strip())) for s, c, r, e in rows}
    seeds = sorted({s for s, _ in runs})
    gpu = {}
    for l in (guard.ROOT / "ledger" / "gpu_time_ledger.tsv").read_text().splitlines()[1:]:
        f = l.split("\t")
        if len(f) >= 6:
            gpu.setdefault(f[3], 0.0)
            gpu[f[3]] += float(f[2])
    table, per_seed, traj = {}, {}, {}
    for s in seeds:
        ev, cf, sm, rf, mets = {}, {}, {}, {}, {}
        for c in "ABCDEF":
            if (s, c) not in runs:
                continue
            r, e = runs[(s, c)]
            ev[c] = json.loads(e.read_text())
            cf[c] = json.loads((r / "config.json").read_text())
            sm[c] = json.loads((r / "summary.json").read_text())
            rf[c] = json.loads((r / "run_facts.json").read_text())
            mets[c] = [json.loads(x) for x in (r / "metrics.jsonl").read_text().splitlines()]
        complete = sorted(ev) == list("ABCDEF")
        if complete:
            ref = ev["A"]["doc_n"]
            assert all(ev[c]["doc_n"] == ref for c in ev), f"seed {s}: per-document valid counts differ"
            facts = {c: {"trainable_sha256": rf[c]["trainable_sha256"], "valid_tokens": sm[c]["valid_train_tokens"],
                         "eval_tokens": ev[c]["n_valid_tokens"]} for c in cf}
            check_matched(cf, facts)
        evals = {c: [m["eval"] for m in mets[c] if "eval" in m] for c in mets}
        target = evals["A"][-1]["ce_nats_per_token"] if "A" in evals and evals["A"] else None
        for c in ev:
            e, m, smc = ev[c], ev[c], sm[c]
            E = cf[c]["model"]["n_experts"]
            share = [x / E for x in e["expert_load_max_over_mean"]]
            r = runs[(s, c)][0]
            table[f"s{s}_{c}"] = {
                "seed": s, "condition": c, "run": str(r), "eval_json": str(runs[(s, c)][1]),
                "val_ce": e["ce_nats_per_token"], "ppl": e["ppl"], "n_valid_tokens": e["n_valid_tokens"],
                "n_docs": len(e["doc_n"]), "steps": smc["steps"], "train_tokens": smc["valid_train_tokens"],
                "loop_seconds": smc["loop_seconds"], "tok_per_s": smc["train_valid_tokens_per_s_after_warmup"],
                "peak_alloc_gib": smc["peak_alloc_gib"], "peak_reserved_gib": smc["peak_reserved_gib"],
                "gpu_seconds_train_ledger": gpu.get(r.name, None), "coverage": e["pool_constrained_coverage"],
                "jaccard": e["adjacent_token_expert_jaccard"], "jaccard_mean": float(np.mean(e["adjacent_token_expert_jaccard"])),
                "load_max_over_mean": e["expert_load_max_over_mean"], "load_cv": e["expert_load_cv"],
                "unused": e["experts_unused"], "token_router_entropy": e["token_router_entropy"],
                "pool_usage_entropy": e["pool_usage_entropy"], "pool_distinct_fraction": e["pool_distinct_fraction"],
                "max_expert_share": share, "collapse_layers": [i + 1 for i, x in enumerate(share) if x > 0.5],
                "tokens_to_A_final_monitor_ce": tokens_to_target(evals[c], target) if target is not None else None,
                "final_monitor_ce": evals[c][-1]["ce_nats_per_token"] if evals[c] else None}
            steps = [x for x in mets[c] if "step" in x]
            traj[f"s{s}_{c}"] = [{k: x.get(k) for k in ("step", "valid_tokens_seen", "lm_ce", "sem", "router_gradnorm_ratio",
                                                       "router_gradnorm_lm", "router_gradnorm_sem_weighted", "coverage")}
                                 for x in steps]
        con = {}
        if complete:
            dn = {c: {d: float(v) for d, v in ev[c]["doc_nll"].items()} for c in ev}
            nn = {c: {d: int(v) for d, v in ev[c]["doc_n"].items()} for c in ev}
            for row, col, role in CONTRASTS + DESCRIPTIVE:
                b = paired_bootstrap(dn[col], dn[row], nn[col], nn[row], n_boot=2000, seed=0)  # row - col
                con[f"{row}-{col}"] = {"role": role, **b, "excludes_zero": b["ci95"][0] > 0 or b["ci95"][1] < 0,
                                       "jaccard_diff": table[f"s{s}_{row}"]["jaccard_mean"] - table[f"s{s}_{col}"]["jaccard_mean"]}
        per_seed[s] = {"complete": complete, "contrasts": con, "A_final_monitor_ce_target": target}
    agg = {}
    full = [s for s in seeds if per_seed[s]["complete"]]
    for row, col, role in CONTRASTS + DESCRIPTIVE:
        k = f"{row}-{col}"
        ds = [per_seed[s]["contrasts"][k]["diff"] for s in full]
        if not ds:
            continue
        same_sign = len(full) == 3 and (all(x < 0 for x in ds) or all(x > 0 for x in ds))
        excl = len(full) == 3 and all(per_seed[s]["contrasts"][k]["excludes_zero"] for s in full)
        mabs = float(np.mean(np.abs(ds)))
        agg[k] = {"role": role, "per_seed_diff": ds, "mean_diff": float(np.mean(ds)), "mean_abs_diff": mabs,
                  "per_seed_ci95": [per_seed[s]["contrasts"][k]["ci95"] for s in full], "same_sign_all_3": same_sign,
                  "ci_excludes_zero_all_3": excl, "mean_abs_ge_0.01": mabs >= 0.01,
                  "passes_prospective_criterion": bool(same_sign and excl and mabs >= 0.01) if role != "descriptive only" else None}
    h1 = None
    if len(full) == 3:
        jd = [per_seed[s]["contrasts"]["B-A"]["jaccard_diff"] for s in full]
        ce = [per_seed[s]["contrasts"]["B-A"]["diff"] for s in full]
        h1 = {"jaccard_diff_B_minus_A": jd, "ce_B_minus_A": ce, "jaccard_up_all_seeds": all(x > 0 for x in jd),
              "ce_noninferior_all_seeds": all(x <= 0.02 for x in ce)}
        h1["supported"] = h1["jaccard_up_all_seeds"] and h1["ce_noninferior_all_seeds"]
    hyp = {"H1": h1,
           "H2": {"D-F": agg.get("D-F"), "supported": bool(agg.get("D-F", {}).get("passes_prospective_criterion")
                                                           and agg["D-F"]["mean_diff"] < 0)} if "D-F" in agg else None,
           "H3": {"E-D": agg.get("E-D"), "supported": bool(agg.get("E-D", {}).get("passes_prospective_criterion")
                                                           and agg["E-D"]["mean_diff"] < 0)} if "E-D" in agg else None}
    cond_mean = {}
    for c in "ABCDEF":
        vals = [table[f"s{s}_{c}"] for s in seeds if f"s{s}_{c}" in table]
        if vals:
            cond_mean[c] = {k: float(np.mean([v[k] for v in vals])) for k in
                            ("val_ce", "ppl", "tok_per_s", "peak_reserved_gib", "coverage", "jaccard_mean", "loop_seconds")}
            cond_mean[c]["val_ce_sd_across_seeds"] = float(np.std([v["val_ce"] for v in vals], ddof=1)) if len(vals) > 1 else None
    tot_gpu = sum(v for k, v in gpu.items() if "rep_v1" in k or "rep_" in k)
    out = {"experiment": "csr-representative-v1", "seeds": seeds, "complete_seeds": full, "table": table,
           "per_seed": {str(s): v for s, v in per_seed.items()}, "aggregate_contrasts": agg, "hypotheses": hyp,
           "condition_means": cond_mean, "trajectories": traj, "gpu_seconds_rep_ledger_total": tot_gpu,
           "created": guard.now()}
    guard.atomic_write_json(guard.inside_root(a.out), out)
    print(json.dumps({"aggregate_contrasts": agg, "hypotheses": {k: (v if k == "H1" else v and v["supported"]) for k, v in hyp.items()},
                      "condition_means": cond_mean}, indent=1, default=str))


if __name__ == "__main__":
    main()
