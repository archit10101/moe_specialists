#!/usr/bin/env python
"""csr-semantic-edl-long-adaptive-k-v1 GPU checks (prereg section 9); adapted from sem_gpu_checks.py. Each training
check runs as a child process (one GPU process at a time); run dirs go to .tmp/semantic-edl-long-k/gpu_check_runs.

  (1) SEMANTIC-C6: finite losses; stop/resume via an injected SIGINT at step 20 (EXTERNALLY_STOPPED, rc 77) then resume
      to step 40, vs an uninterrupted 40-step run and a second uninterrupted run (noise)
  (2) T-EQ: slk_train TOKEN-4 seed 0 vs the reused fle final_T4_s0 metrics (step 0 bit identity; steps 50/100 reported)
  (3) beta = 0 paper_latent (C6 table) vs TOKEN-4 seed 0: step-0 bit identity, step-50 difference
  (4) SHUFFLED-C6 20 steps: finite; parameter counts / active parameters per token equal across T4, C6, SHUF, beta0
Writes artifacts/semantic-edl-long-k/gpu_checks.json. Run via: slk_driver.py job gpu_checks --gpu -- slk_gpu_checks.py"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402

PY = str(guard.ROOT / ".venv/bin/python")


def run_train(cfg_p, extra, env):
    return subprocess.run([PY, "slk_train.py", "--config", str(cfg_p), "--device", "cuda", "--execute"] + extra,
                          env=env, capture_output=True, text=True, cwd=guard.ROOT)


def metrics(run_dir):
    out, diag = {}, {}
    for x in (Path(run_dir) / "metrics.jsonl").read_text().splitlines():
        r = json.loads(x)
        if "diag" in r:
            diag[r["diag"]["step"]] = r["diag"]
        elif "step" in r:
            out[r["step"]] = r
    return out, diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    from slk_configs import make, paper_cond, phase1_conds
    from csr.slk_paths import ART, C6_TABLE, C6_TABLE_SHA, REUSED, TMP, check_prereg
    check_prereg()
    RD = TMP / "gpu_check_runs"
    RD.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "MOE_RUNS_DIR": str(RD)}
    res = {"time": guard.now()}
    t0 = time.time()
    conds = phase1_conds()
    dirs = {}

    def train_new(tag, cfg_p, extra):
        before = set(RD.iterdir())
        rr = run_train(cfg_p, extra, env)
        new = sorted(set(RD.iterdir()) - before)
        dirs[tag] = new[0]
        res[f"rc_{tag}"] = rr.returncode
        if rr.returncode not in (0, 75, 77):
            res[f"err_{tag}"] = (rr.stdout + rr.stderr)[-3000:]
        return new[0]

    def params(d):
        legs = sorted(Path(d).glob("leg_*.json"))
        return json.loads(legs[-1].read_text())["params"]

    # (1) C6 stop/resume
    cfg_c6, _ = make("chk_C6_s0", 0, conds["C6"], "calibration", diag_every=10)
    for tag, extra in (("straight1", ["--max-steps", "40"]), ("straight2", ["--max-steps", "40"]),
                       ("stopped", ["--max-steps", "40", "--test-signal-at-step", "20"])):
        d = train_new(tag, cfg_c6, extra)
        if tag == "stopped":
            rr2 = run_train(cfg_c6, ["--max-steps", "40", "--run-dir", str(d)], env)
            res["rc_resumed"] = rr2.returncode
            res["stopped_status"] = json.loads((d / "leg_01.json").read_text())["status"]
    lm = {t: metrics(dirs[t]) for t in ("straight1", "straight2", "stopped")}
    noise = max(abs(lm["straight1"][1][s]["val_lm"] - lm["straight2"][1][s]["val_lm"]) for s in (20, 30))
    dres = max(abs(lm["straight1"][1][s]["val_lm"] - lm["stopped"][1][s]["val_lm"]) for s in (20, 30))
    finite = all(all(v == v and abs(v) < 1e9 for k, v in x.items() if isinstance(v, float))
                 for t in lm.values() for part in t for x in part.values())
    tol = max(0.003, 3 * noise)
    res["c6_resume"] = {"max_abs_dlm_resumed_vs_straight": dres, "run_to_run_noise": noise, "tolerance": tol,
                        "pass": bool(dres <= tol and res["rc_stopped"] == 77 and res["rc_resumed"] == 75),
                        "finite": finite, "gcedl_step0": lm["straight1"][0][0].get("gcedl")}
    # (2) T-EQ against the reused fle final_T4_s0
    cfg_t4, _ = make("chk_T4eq_s0", 0, {}, "calibration", diag_every=100)
    d_t4 = train_new("teq", cfg_t4, ["--max-steps", "101"])
    mine, _ = metrics(d_t4)
    ref, _ = metrics(guard.ROOT / REUSED["T4_s0"])
    res["teq"] = {"step0_mine": mine[0]["lm"], "step0_ref": ref[0]["lm"], "step0_bit_identical": mine[0]["lm"] ==
                  ref[0]["lm"], "d_step50": mine[50]["lm"] - ref[50]["lm"], "d_step100": mine[100]["lm"] - ref[100]["lm"]}
    # (3) beta = 0 paper_latent vs TOKEN-4
    cfg_b0, _ = make("chk_beta0_s0", 0, paper_cond(C6_TABLE, C6_TABLE_SHA, 0.0), "calibration", diag_every=100)
    d_b0 = train_new("beta0", cfg_b0, ["--max-steps", "51"])
    b0, _ = metrics(d_b0)
    res["beta0"] = {"step0_lm": b0[0]["lm"], "step0_bit_identical_to_T4": b0[0]["lm"] == mine[0]["lm"],
                    "d_step50_vs_T4": b0[50]["lm"] - mine[50]["lm"]}
    # (4) SHUF short run + parameter equality
    cfg_sh, _ = make("chk_SHUF_s0", 0, conds["SHUF"], "calibration", diag_every=100)
    d_sh = train_new("shuf", cfg_sh, ["--max-steps", "20"])
    shm, _ = metrics(d_sh)
    P = {k: params(d) for k, d in (("T4", d_t4), ("C6", dirs["straight1"]), ("SHUF", d_sh), ("beta0", d_b0))}
    res["params"] = P
    res["params_equal"] = all(p == P["T4"] for p in P.values())
    res["shuf_finite"] = all(v == v for x in shm.values() for v in x.values() if isinstance(v, float))
    res["all_pass"] = bool(res["c6_resume"]["pass"] and finite and res["teq"]["step0_bit_identical"]
                           and res["beta0"]["step0_bit_identical_to_T4"] and res["params_equal"] and res["shuf_finite"])
    res["seconds"] = time.time() - t0
    res["run_dirs"] = {k: str(v.relative_to(guard.ROOT)) for k, v in dirs.items()}
    guard.atomic_write_json(ART / "gpu_checks.json", res, overwrite=True)
    with open(ART / "deleted_artifacts.tsv", "a") as f:
        for d in dirs.values():
            sl = TMP / "slots" / d.name
            if sl.exists():
                shutil.rmtree(sl)
                f.write(f"{guard.now()}\t{sl.relative_to(guard.ROOT)}\tgpu-check resumable slot (disposable)\n")
    print(json.dumps(res, indent=1, default=str))
    sys.exit(0 if res["all_pass"] else 1)


if __name__ == "__main__":
    main()
