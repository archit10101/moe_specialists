#!/usr/bin/env python
"""csr-forecast-latent-edl-v1 GPU checks (prereg section 10) through the driver:
  smoke  : 20 compiled steps of FL-EDL, SHUF, FD (gamma 1) and EDL (finite; TOKEN-4 covered by the T-EQ run)
  resume : FL-EDL (C01 knobs, log every step) 60 steps uninterrupted vs 30 + resume 30; pass if the mean |dlm| over
           steps 30..59 <= max(0.003, 3 x the A-vs-C uninterrupted run-to-run noise) and all values are finite.
Writes artifacts/forecast-latent-edl-v1/gpu_checks.json."""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402
from csr.fle_paths import ART, UPPER  # noqa: E402


def metrics(rn):
    from fle_search import run_dir_of
    rd = guard.ROOT / run_dir_of(rn)
    return {r["step"]: r for r in map(json.loads, (rd / "metrics.jsonl").read_text().splitlines()) if "step" in r}


def main():
    guard.check_root()
    from fle_configs import latent_cond, make
    from fle_driver import run_job
    from fle_search import remember_run_dir, run_dir_of
    cid = "A1_K16_L256"
    res = {"time": guard.now()}
    lc = latent_cond(cid, 1e-4, UPPER)
    conds = {"smoke_FL": lc, "smoke_SHUF": latent_cond(cid, 1e-4, UPPER, shuffle=True),
             "smoke_FD": {"fd": {**{k: lc["latent"][k] for k in ("cfg_id", "table", "table_sha256")}, "gamma": 1.0}},
             "smoke_EDL": {"edl_beta": 5e-3}}
    for rn, cond in conds.items():
        p, _ = make(rn, 100, cond, "search", diag_every=10)
        run_job(f"train_{rn}", ["fle_train.py", "--config", str(p), "--max-steps", "20"], gpu=True, expect=(75,))
        remember_run_dir(rn)
        m = metrics(rn)
        st = json.loads((guard.ROOT / run_dir_of(rn) / "status.json").read_text())
        res[rn] = {"finite": all(math.isfinite(v) for r in m.values() for v in r.values() if isinstance(v, float)),
                   "lm_step0": m[0]["lm"], "tokens_per_s": st["tokens_per_s"], "peak_alloc_gib": st["peak_alloc_gib"]}
    pA, _ = make("rc_A", 100, lc, "search", train_override={"log_every": 1})
    pB, _ = make("rc_B", 100, lc, "search", train_override={"log_every": 1})
    run_job("train_rc_A", ["fle_train.py", "--config", str(pA), "--max-steps", "60"], gpu=True, expect=(75,))
    remember_run_dir("rc_A")
    run_job("train_rc_B1", ["fle_train.py", "--config", str(pB), "--max-steps", "30"], gpu=True, expect=(75,))
    remember_run_dir("rc_B")
    run_job("train_rc_B2", ["fle_train.py", "--config", str(pB), "--max-steps", "60", "--run-dir", run_dir_of("rc_B")],
            gpu=True, expect=(75,))
    pC, _ = make("rc_C", 100, lc, "search", train_override={"log_every": 1})
    run_job("train_rc_C", ["fle_train.py", "--config", str(pC), "--max-steps", "60"], gpu=True, expect=(75,))
    remember_run_dir("rc_C")
    a, b, c = metrics("rc_A"), metrics("rc_B"), metrics("rc_C")
    noise = sum(abs(a[s]["lm"] - c[s]["lm"]) for s in range(30, 60)) / 30
    d_pre = [abs(a[s]["lm"] - b[s]["lm"]) for s in range(30)]
    d_post = [abs(a[s]["lm"] - b[s]["lm"]) for s in range(30, 60)]
    res["resume"] = {"mean_abs_dlm_steps_0_29": sum(d_pre) / 30, "mean_abs_dlm_steps_30_59": sum(d_post) / 30,
                     "max_abs_dlm_steps_30_59": max(d_post), "fledl_step59": [a[59]["fledl"], b[59]["fledl"]],
                     "noise_uninterrupted_A_vs_C_steps_30_59": noise,
                     "pass": sum(d_post) / 30 <= max(0.003, 3 * noise)}
    res["all_ok"] = res["resume"]["pass"] and all(res[k]["finite"] for k in conds)
    guard.atomic_write_json(ART / "gpu_checks.json", res, overwrite=True)
    print(json.dumps(res, indent=1))
    sys.exit(0 if res["all_ok"] else 1)


if __name__ == "__main__":
    main()
