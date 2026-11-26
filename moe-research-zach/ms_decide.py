#!/usr/bin/env python
"""csr-multiscale-routing-v1 CPU decision steps (no GPU, no model). Each output is written once, never overwritten.

  ms_decide.py calibration --runs <tsv>            -> artifacts/multiscale-routing-v1/calibration.json
  ms_decide.py feasibility --runs <bench tsv>      -> artifacts/multiscale-routing-v1/feasibility.json
  ms_decide.py pilot --runs <pilot tsv>            -> artifacts/multiscale-routing-v1/pilot.json
TSV rows: key<TAB>run_dir (calibration key: seed98/beta_0p25__BROAD-EARLY; bench/pilot key: condition).
"""
import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402
from csr import ms_select  # noqa: E402
from csr.multiscale import CONTEXTUAL, branch_of  # noqa: E402

ART = guard.ROOT / "artifacts" / "multiscale-routing-v1"
VAL_TOKENS, CALIB_TOKENS = 10_732_175, 5_270_143


def rows(tsv):
    return [ln.split("\t") for ln in Path(tsv).read_text().splitlines() if ln.strip()]


def ledger_seconds(run_id):
    s = 0
    for ln in (guard.ROOT / "ledger" / "gpu_time_ledger.tsv").read_text().splitlines()[1:]:
        p = ln.split("\t")
        if len(p) >= 4 and p[3] == run_id:
            s += int(p[2])
    return s


def msv1_gpu_seconds():
    s = 0
    for ln in (guard.ROOT / "ledger" / "gpu_time_ledger.tsv").read_text().splitlines()[1:]:
        p = ln.split("\t")
        if len(p) >= 5 and (p[4].startswith("msv1") or "msv1" in p[3]):
            s += int(p[2])
    return s


def run_record(rd):
    rd = guard.ROOT / rd
    st = json.loads((rd / "summary.json").read_text())
    cfg = json.loads((rd / "config.json").read_text())
    met = [json.loads(x) for x in (rd / "metrics.jsonl").read_text().splitlines()]
    steps = [m for m in met if "step" in m and "lm_ce" in m]
    finite = all(math.isfinite(m[k]) for m in steps for k in ("lm_ce", "aux", "z", "grad_norm"))
    ce = json.loads((rd / "calib_eval.json").read_text()) if (rd / "calib_eval.json").exists() else None
    return st, cfg, steps, finite, ce


def calibration(tsv):
    out_p = ART / "calibration.json"
    if out_p.exists():
        raise guard.GuardError(f"{out_p} exists")
    res = {"alpha": [], "beta": []}
    runs = {}
    for key, rd in rows(tsv):
        st, cfg, steps, finite, ce = run_record(rd)
        seed = int(key.split("/")[0].replace("seed", ""))
        kind, rest = key.split("/")[1].split("_", 1)
        coef_s, placement = rest.split("__")
        coef = float(coef_s.replace("p", "."))
        assert cfg["routing"][kind] == coef and cfg["condition"] == placement and cfg["init"]["seed"] == seed
        assert ce is None or (ce["split"] == "calib" and ce["n_valid_tokens"] == CALIB_TOKENS)
        rec = {"coef": coef, "seed": seed, "placement": placement, "status": st["status"], "finite": finite,
               "ce": ce["ce_nats_per_token"] if ce else float("nan"),
               "max_share": ce["expert_max_share"] if ce else [1.0] * 7,
               "unused": ce["experts_unused"] if ce else [32] * 7}
        res[kind].append(rec)
        runs[key] = {"run": rd, **{k: v for k, v in rec.items()}, "train_tokens": st.get("valid_train_tokens"),
                     "gpu_ledger_seconds": ledger_seconds(Path(rd).name), "loop_seconds": st.get("loop_seconds"),
                     "calib_eval_seconds": st.get("calib_eval_seconds")}
    assert len(res["alpha"]) == 24 and len(res["beta"]) == 16, (len(res["alpha"]), len(res["beta"]))
    alpha, ta = ms_select.select_coefficient(res["alpha"])
    beta, tb = ms_select.select_coefficient(res["beta"])
    br = branch_of(alpha, beta)
    out = {"alpha": alpha, "beta": beta, "branch": br, "alpha_table": {str(k): v for k, v in ta.items()},
           "beta_table": {str(k): v for k, v in tb.items()}, "runs": runs, "validation_accessed": False,
           "rule": "docs/MULTISCALE_ROUTING_V1_PREREG.md section 7", "created": guard.now()}
    guard.atomic_write_json(out_p, out, mode=0o400)
    guard.ledger("MS_V1_CALIBRATION", f"alpha={alpha} beta={beta} branch={br} sha256={guard.sha256_file(out_p)}")
    print(json.dumps({"alpha": alpha, "beta": beta, "branch": br,
                      "alpha_agg": {k: (v.get("aggregate_ce"), v.get("diff_vs_zero"), v["qualifies"]) for k, v in ta.items()},
                      "beta_agg": {k: (v.get("aggregate_ce"), v.get("diff_vs_zero"), v["qualifies"]) for k, v in tb.items()}},
                     indent=1))


def feasibility(tsv):
    out_p = ART / "feasibility.json"
    if out_p.exists():
        raise guard.GuardError(f"{out_p} exists")
    cal = json.loads((ART / "calibration.json").read_text())
    tps, peak, bench = {}, 0.0, {}
    for cond, rd in rows(tsv):
        st = json.loads((guard.ROOT / rd / "summary.json").read_text())
        assert st["status"] == "COMPLETE"
        tps[cond] = st["train_valid_tokens_per_s_after_warmup"]
        peak = max(peak, st["peak_reserved_gib"])
        size = sum(f.stat().st_size for f in (guard.ROOT / rd).iterdir())
        bench[cond] = {"run": rd, "tokens_per_s": tps[cond], "peak_alloc_gib": st["peak_alloc_gib"],
                       "peak_reserved_gib": st["peak_reserved_gib"], "run_dir_bytes": size,
                       "gpu_ledger_seconds": ledger_seconds(Path(rd).name)}
    over = [r["gpu_ledger_seconds"] - r["loop_seconds"] - (r["calib_eval_seconds"] or 0) for r in cal["runs"].values()]
    overhead = float(statistics.median(over))
    calib_eval = float(statistics.median(r["calib_eval_seconds"] for r in cal["runs"].values()))
    val_eval = calib_eval * VAL_TOKENS / CALIB_TOKENS + 30.0
    disk_per_run = 115_530_248 + 22_000_000 + 30_000_000 + max(b["run_dir_bytes"] for b in bench.values())
    consumed = msv1_gpu_seconds()
    free = guard.disk_free_bytes()
    budget, table = ms_select.select_budget(consumed, tps, overhead, val_eval, peak, free, disk_per_run)
    out = {"budget": budget, "table": table, "benchmark": bench, "overhead_s": overhead, "calib_eval_s": calib_eval,
           "val_eval_s_projected": val_eval, "disk_per_run_bytes_projected": disk_per_run,
           "consumed_gpu_s": consumed, "disk_free_bytes": free, "peak_reserved_gib": peak,
           "alpha": cal["alpha"], "beta": cal["beta"], "branch": cal["branch"],
           "interpretation": {100_000_000: "intended main study", 50_000_000: "resource-limited replicated study",
                              25_000_000: "extended replicated pilot", None: "calibration and pilot only"}[budget],
           "rule": "docs/MULTISCALE_ROUTING_V1_PREREG.md section 9", "created": guard.now()}
    guard.atomic_write_json(out_p, out, mode=0o400)
    guard.ledger("MS_V1_BUDGET", f"budget={budget} branch={cal['branch']} projected="
                                 f"{[round(r['projected_total_gpu_h_with_margin'], 2) for r in table]} GPU-h "
                                 f"sha256={guard.sha256_file(out_p)}")
    print(json.dumps({k: v for k, v in out.items() if k != "benchmark"}, indent=1, default=str))


def pilot(tsv):
    out_p = ART / "pilot.json"
    if out_p.exists():
        raise guard.GuardError(f"{out_p} exists")
    ce, slopes, recs = {}, {}, {}
    for cond, rd in rows(tsv):
        st, cfg, steps, finite, cev = run_record(rd)
        assert st["status"] == "COMPLETE" and finite and cev["split"] == "calib"
        ce[cond] = cev["ce_nats_per_token"]
        slopes[cond] = ms_select.final20_slope(steps, st["steps"])
        recs[cond] = {"run": rd, "calib_ce": ce[cond], "final20_slope_per_token": slopes[cond],
                      "train_tokens": st["valid_train_tokens"], "expert_max_share": cev["expert_max_share"],
                      "experts_unused": cev["experts_unused"], "loop_seconds": st["loop_seconds"],
                      "tokens_per_s": st["train_valid_tokens_per_s_after_warmup"]}
    stop, detail = ms_select.pilot_futility(ce, slopes, CONTEXTUAL)
    out = {"futility_stop": stop, "detail": detail, "runs": recs, "rule": "docs/MULTISCALE_ROUTING_V1_PREREG.md §10",
           "note": "no condition is selected from pilot rank differences", "created": guard.now()}
    guard.atomic_write_json(out_p, out, mode=0o400)
    guard.ledger("MS_V1_PILOT", f"futility_stop={stop} sha256={guard.sha256_file(out_p)}")
    print(json.dumps(out, indent=1, default=str)[:3000])
    sys.exit(3 if stop else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["calibration", "feasibility", "pilot"])
    ap.add_argument("--runs", required=True)
    a = ap.parse_args()
    guard.check_root()
    {"calibration": calibration, "feasibility": feasibility, "pilot": pilot}[a.cmd](a.runs)


if __name__ == "__main__":
    main()
