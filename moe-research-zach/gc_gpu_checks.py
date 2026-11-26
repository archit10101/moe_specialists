#!/usr/bin/env python
"""csr-gradient-consistent-latent-edl-v1 GPU checks (prereg section 7). The parent never initialises CUDA; each check
runs as a child process (so the occupancy monitor sees only one GPU process at a time).

  (1) fingerprint GPU (fp32) vs CPU (float64) agreement on 64 real views, bitwise GPU replay
  (2) GC run: finite losses; stop/resume via an injected SIGINT at step 20 (EXTERNALLY_STOPPED path, rc 77) then
      resume to step 40, vs an uninterrupted 40-step run and a second uninterrupted run (noise)
  (3) T-EQ: gc_train TOKEN-4 seed 100 vs the reused fle s_T4_s100 metrics (step 0 bit identity; 50/100 reported)
Writes artifacts/gc-edl-v1/gpu_checks.json.  Run via: gc_driver.py job gpu_checks --gpu -- gc_gpu_checks.py"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402

PY = str(guard.ROOT / ".venv/bin/python")


def child_fp():
    import numpy as np
    import torch
    from gc_fingerprint import load_scout
    from csr.gc_lib import Sketch, scout_fingerprints
    from csr.gc_paths import DISC
    z = np.load(DISC / "inputs_calib_search.npz")
    ii = np.nonzero(z["nv"] >= 1)[0][:64]
    ids = torch.as_tensor(z["views"][ii, 0].astype(np.int64))
    sk = Sketch(768, 256, 128)
    torch.backends.cuda.matmul.allow_tf32 = False
    gpu, _ = load_scout(torch.device("cuda"))
    g1 = scout_fingerprints(gpu, ids.cuda(), sk)
    g2 = scout_fingerprints(gpu, ids.cuda(), sk)
    replay = all(torch.equal(a, b) for a, b in zip(g1, g2))
    cpu, _ = load_scout(torch.device("cpu"))
    gc = scout_fingerprints(cpu, ids, sk)
    a, b = g1[0].cpu().numpy(), gc[0].numpy()
    cos = (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
    rel = float(np.abs(a - b).max() / np.abs(b).max())
    print(json.dumps({"fp_gpu_cpu_min_cos": float(cos.min()), "fp_gpu_cpu_max_rel_err": rel, "fp_gpu_replay": replay,
                      "norm_rel_err": float((g1[1].cpu() - gc[1]).abs().max() / gc[1].abs().max()),
                      "ce_abs_err": float((g1[2].cpu() - gc[2]).abs().max())}))


def run_train(cfg_p, extra, env):
    r = subprocess.run([PY, "gc_train.py", "--config", str(cfg_p), "--device", "cuda", "--execute"] + extra,
                       env=env, capture_output=True, text=True, cwd=guard.ROOT)
    return r


def metrics(run_dir):
    out = {}
    for x in (Path(run_dir) / "metrics.jsonl").read_text().splitlines():
        r = json.loads(x)
        if "step" in r:
            out[r["step"]] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", choices=["fp"])
    ap.add_argument("--table", default="artifacts/gc-edl-v1/latent/table_COMBINED.npz")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    if a.child == "fp":
        child_fp()
        return
    from gc_configs import gc_cond, make
    from csr.gc_paths import ART, T4_S100_RUN, TMP, check_prereg
    check_prereg()
    env = {**os.environ, "MOE_RUNS_DIR": str(TMP / "gpu_check_runs")}
    (TMP / "gpu_check_runs").mkdir(parents=True, exist_ok=True)
    res = {"time": guard.now()}
    t0 = time.time()
    r = subprocess.run([PY, "gc_gpu_checks.py", "--child", "fp"], capture_output=True, text=True, cwd=guard.ROOT,
                       env=env)
    res["fingerprint"] = json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else {"error": r.stderr[-2000:]}
    # (2) GC stop/resume
    cond = gc_cond(a.table, 5e-3)
    cfg_p, cfg = make("chk_GC_s100", 100, cond, "pilot", diag_every=10)
    lm = {}
    dirs = {}
    for tag, extra in (("straight1", ["--max-steps", "40"]), ("straight2", ["--max-steps", "40"]),
                       ("stopped", ["--max-steps", "40", "--test-signal-at-step", "20"])):
        before = set((TMP / "gpu_check_runs").iterdir())
        rr = run_train(cfg_p, extra, env)
        new = sorted(set((TMP / "gpu_check_runs").iterdir()) - before)
        dirs[tag] = new[0]
        res[f"rc_{tag}"] = rr.returncode
        if tag == "stopped":
            rr2 = run_train(cfg_p, ["--max-steps", "40", "--run-dir", str(new[0])], env)
            res["rc_resumed"] = rr2.returncode
            res["stopped_status"] = json.loads((new[0] / "leg_01.json").read_text())["status"]
        lm[tag] = metrics(new[0])
    noise = max(abs(lm["straight1"][s]["lm"] - lm["straight2"][s]["lm"]) for s in range(20, 40))
    dres = max(abs(lm["straight1"][s]["lm"] - lm["stopped"][s]["lm"]) for s in range(20, 40))
    finite = all(all(v == v and abs(v) < 1e9 for v in (x.get("lm", 0), x.get("gcedl", 0))) for t in lm.values()
                 for x in t.values())
    tol = max(0.003, 3 * noise)
    res["gc_resume"] = {"max_abs_dlm_resumed_vs_straight_steps20_39": dres, "run_to_run_noise": noise,
                        "tolerance": tol, "pass": bool(dres <= tol and res["rc_stopped"] == 77
                                                       and res["rc_resumed"] == 75),
                        "finite": finite, "gcedl_step0": lm["straight1"][0].get("gcedl"),
                        "jsd_step0": lm["straight1"][0].get("jsd")}
    # (3) T-EQ against the reused fle seed-100 TOKEN-4
    cfg_p2, _ = make("chk_T4eq_s100", 100, {}, "pilot", diag_every=25)
    before = set((TMP / "gpu_check_runs").iterdir())
    rr = run_train(cfg_p2, ["--max-steps", "101"], env)
    teq_dir = sorted(set((TMP / "gpu_check_runs").iterdir()) - before)[0]
    mine = metrics(teq_dir)
    ref = metrics(guard.ROOT / T4_S100_RUN)
    res["teq"] = {"rc": rr.returncode, "step0_mine": mine[0]["lm"], "step0_ref": ref[0]["lm"],
                  "step0_bit_identical": mine[0]["lm"] == ref[0]["lm"],
                  "d_step50": mine[50]["lm"] - ref[50]["lm"], "d_step100": mine[100]["lm"] - ref[100]["lm"]}
    res["all_pass"] = bool(res["fingerprint"].get("fp_gpu_cpu_min_cos", 0) > 0.9999 and res["fingerprint"].get(
        "fp_gpu_replay") and res["gc_resume"]["pass"] and finite and res["teq"]["step0_bit_identical"])
    res["seconds"] = time.time() - t0
    res["run_dirs"] = {k: str(v.relative_to(guard.ROOT)) for k, v in {**dirs, "teq": teq_dir}.items()}
    guard.atomic_write_json(ART / "gpu_checks.json", res, overwrite=True)
    # disposable check runs: delete resumable slots (ledgered); keep metrics/logs
    import shutil
    with open(ART / "deleted_artifacts.tsv", "a") as f:
        for d in list(dirs.values()) + [teq_dir]:
            sl = TMP / "slots" / d.name
            if sl.exists():
                shutil.rmtree(sl)
                f.write(f"{guard.now()}\t{sl.relative_to(guard.ROOT)}\tgpu-check resumable slot (disposable)\n")
    print(json.dumps(res, indent=1, default=str))
    sys.exit(0 if res["all_pass"] else 1)


if __name__ == "__main__":
    main()
