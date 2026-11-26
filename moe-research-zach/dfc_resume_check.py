#!/usr/bin/env python
"""csr-domain-forecast-curriculum-v1: GPU interrupted/resumed vs uninterrupted check with the real training script
(SOFT_s0 config, curriculum active). A: 6 steps uninterrupted (stops resumable at step 6). B: 3 steps, then resumed in
place to 6. Separate slot dirs and runs dir under .tmp (never runs/). Compares the step-6 slots: data position, the
per-step loss trace, and parameters (GPU kernels with scatter-add backward are not bitwise deterministic, so the
parameter comparison reports max |diff|; the CPU suite checks bitwise equality). Writes resume_check_gpu_<stamp>.json.
Launched by the driver as a CPU process; each training leg goes through scripts/gpu_guarded.sh."""
import json, os, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402
from csr.dfc_lib import ART, CFG_DIR, ST  # noqa: E402
def leg(base, slot, extra):
    env = dict(os.environ, MOE_GPU_LAUNCH="1", MOE_RESEARCH_APPROVE="gpu")
    while True:
        p = subprocess.run(["scripts/gpu_guarded.sh", "dfc resume check", "dfc_train.py", "--config",
                            str(CFG_DIR / "main/SOFT_s0.json"), "--device", "cuda", "--execute", "--runs-dir", str(base),
                            "--slot-dir", str(base / slot), "--no-ledger"] + extra, cwd=guard.ROOT, env=env,
                           capture_output=True, text=True)
        if p.returncode == 3 and "BLOCKED" in (p.stdout + p.stderr):
            time.sleep(120); continue
        rd = [ln.split(" ", 1)[1] for ln in p.stdout.splitlines() if ln.startswith("RUN_DIR ")]
        return p.returncode, (rd[-1] if rd else None), p.stdout[-2000:] + p.stderr[-2000:]
def main():
    guard.check_root()
    import torch
    t_iso = guard.now()
    base = ST / "gpu_resume_check" / guard.utc_stamp()
    base.mkdir(parents=True)
    ra = leg(base, "A", ["--max-steps", "6"])
    rb = leg(base, "B", ["--max-steps", "3"])
    rb2 = leg(base, "B", ["--max-steps", "6", "--run-dir", rb[1]])
    sa = torch.load(base / "A/resume_slot.pt", map_location="cpu", weights_only=False)
    sb = torch.load(base / "B/resume_slot.pt", map_location="cpu", weights_only=False)
    diffs = {k: float((sa["model"][k].float() - sb["model"][k].float()).abs().max()) for k in sa["model"]}
    def trace(rd):
        out = {}
        for ln in (guard.ROOT / rd / "metrics.jsonl").read_text().splitlines():
            r = json.loads(ln)
            if "lm_ce" in r:
                out[r["step"]] = r["lm_ce"]
        return out
    ta, tb = trace(ra[1]), trace(rb[1])
    doc = {"time": guard.now(), "rcs": [ra[0], rb[0], rb2[0]], "run_dirs": [ra[1], rb[1]],
           "next_step": [sa["st"]["next_step"], sb["st"]["next_step"]], "seen": [sa["st"]["seen"], sb["st"]["seen"]],
           "loss_trace_A": ta, "loss_trace_B": tb, "param_max_abs_diff": max(diffs.values()),
           "n_params_bitwise_equal": sum(v == 0.0 for v in diffs.values()), "n_tensors": len(diffs),
           "same_init": sa["init_param_sha256"] == sb["init_param_sha256"]}
    ok = (doc["rcs"] == [75, 75, 75] and doc["next_step"] == [6, 6] and doc["seen"][0] == doc["seen"][1]
          and doc["same_init"] and doc["param_max_abs_diff"] < 1e-3
          and all(abs(ta[k] - tb[k]) < 5e-3 for k in ta if k in tb))
    doc["status"] = "PASS" if ok else "FAIL"
    out = ART / f"resume_check_gpu_{guard.utc_stamp()}.json"
    guard.atomic_write_json(out, doc)
    guard.gpu_ledger(t_iso, guard.now(), out.stem, "dfc resume check (3 guarded legs)", doc["status"])
    print(json.dumps(doc, indent=1)[:3000])
    sys.exit(0 if ok else 1)
if __name__ == "__main__":
    main()
