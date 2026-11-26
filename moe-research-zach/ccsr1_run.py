#!/usr/bin/env python
"""csr-current-context-specialist-router-v1 orchestrator. Every GPU job goes through ccsr1_driver.py
(scripts/gpu_guarded.sh).

  ccsr1_run.py extract_train   TRAIN tables (router-training supervision only; no selection data)
  ccsr1_run.py pilot           pre-freeze GPU checks on TRAIN only: extraction consistency, router speed,
                               determinism, convergence at TRAIN_STEPS vs 2x (in-sample, TRAIN)
  ccsr1_run.py stage1          SEL tables -> routers (3 seeds x configs x {SEM, SHUF}) -> Stage-1 gates + selection
  ccsr1_run.py stage2          freeze -> CALIB-SELECTION labels/priors/tables -> evaluate frozen routers -> gates
  ccsr1_run.py all             stage1 -> stage2 as far as the frozen rules allow
  ccsr1_run.py stage3a|stage3b|efficiency   Stage-3 joint continuation / section-11 study (Stage-3 amendment)

External interruption (driver exit 77): stop, write STOPPED_BY_OPERATOR + resume command, never restart."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402
from csr import ccsr1_log as LG  # noqa: E402
from csr.ccsr1_paths import ART, CFG_DIR, GPU_CAP_H, ROUTER_SEEDS, ROUTERS, TABLES, TMP  # noqa: E402

PY = str(guard.ROOT / ".venv/bin/python")
CMD_NOW = ""
GEN = int(os.environ.get("CCSR1_GEN", "0"))
SFX = f"_g{GEN}" if GEN else ""


def stop_external(job_id):
    LG.set_state("STOPPED_BY_OPERATOR", job=job_id)
    LG.write_resume("STOPPED_BY_OPERATOR", f"{PY} ccsr1_driver.py authorize-resume && nohup {PY} ccsr1_run.py {CMD_NOW} &",
                    notes=f"Interrupted job: `{job_id}`. Treated as intentional GPU reclamation; nothing restarts "
                          f"automatically. Router jobs skip routers already written; rerunning the phase resumes.")
    print(f"EXTERNAL STOP at {job_id}; see RESUME.md", flush=True)
    sys.exit(77)


def job(job_id, stage, argv, gpu=True, expect="0", disk_gb=3.0, seed="", env=None):
    tot, _ = LG.gpu_hours()
    if gpu and tot >= GPU_CAP_H["total"]:
        LG.set_state("STOPPED_GPU_CAP", gpu_h=tot)
        LG.write_resume("STOPPED_GPU_CAP", f"{PY} ccsr1_run.py {CMD_NOW}", notes=f"GPU cap {GPU_CAP_H['total']} h reached")
        raise SystemExit(f"GPU cap reached ({tot:.2f} h)")
    free = guard.disk_free_bytes() / 2**30
    if free - disk_gb < 100.0:
        LG.set_state("STOPPED_DISK", free_gib=free, need_gib=disk_gb)
        LG.write_resume("STOPPED_DISK", f"{PY} ccsr1_run.py {CMD_NOW}",
                        notes=f"Disk shortfall: free {free:.2f} GiB, need 100 + {disk_gb:.2f} GiB.")
        raise SystemExit(f"disk shortfall {100 + disk_gb - free:.2f} GiB")
    LG.disk(job_id, disk_gb)
    st = json.loads((ART / "driver_state.json").read_text()) if (ART / "driver_state.json").exists() else {"jobs": {}}
    if st["jobs"].get(job_id, {}).get("status") == "COMPLETE":
        return 0
    t0, t_iso = time.time(), guard.now()
    cmd = [PY, "ccsr1_driver.py", "job", job_id, "--gpu" if gpu else "--cpu", "--expect-rc", expect, "--disk-gb",
           str(disk_gb), "--"] + argv
    rc = subprocess.run(cmd, cwd=guard.ROOT, env={**os.environ, **(env or {})}).returncode
    dt = time.time() - t0
    st = json.loads((ART / "driver_state.json").read_text())
    j = st["jobs"].get(job_id, {})
    if gpu:
        LG.compute(job_id, stage, t_iso, guard.now(), dt, j.get("status", "?"))
    LG.manifest({"time": guard.now(), "job": job_id, "stage": stage, "seed": seed, "status": j.get("status"),
                 "rc": j.get("rc"), "seconds": round(dt, 1), "gpu_h": round(dt / 3600, 4) if gpu else 0,
                 "cmd": " ".join(argv)})
    LG.run_log(job=job_id, stage=stage, seed=seed, cmd=argv, gpu=gpu, start=t_iso, end=guard.now(), rc=rc,
               status=j.get("status"))
    if rc == 77:
        stop_external(job_id)
    if rc != 0:
        LG.set_state("FAILED", job=job_id, rc=rc)
        LG.write_resume("FAILED", f"{PY} ccsr1_run.py {CMD_NOW}", notes=f"Job {job_id} failed (rc {rc}); see job_logs.")
        raise SystemExit(f"job {job_id} failed rc={rc}")
    return j.get("rc")


def rel(p):
    return str(Path(p).relative_to(guard.ROOT))


def extract(part, stage):
    out = TABLES / f"{part}.npz"
    if not out.exists():
        job(f"extract_{part}", stage, ["ccsr1_extract.py", "--part", part, "--out", rel(out)], disk_gb=3.0)
    return out


def write_cfg(name, cfgs):
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    p = CFG_DIR / f"{name}.json"
    if p.exists():
        if json.loads(p.read_text()) != cfgs:
            raise guard.GuardError(f"config {p} exists with different content")
        return p
    guard.atomic_write_json(p, cfgs, mode=0o400)
    return p


# ------------------------------------------------------------------------------------------------ pilot (pre-freeze)
def pilot():
    tr = extract("TRAIN", "pilot")
    cfgs = [{"tag": "P_R3", "target": "R3", "c": 1.0, "eta": 1.0, "H": 64, "lam": 0.01},
            {"tag": "P_R2", "target": "R2", "c": 1.0, "H": 64, "lam": 0.01},
            {"tag": "P_R1", "target": "R1", "H": 64, "lam": 0.01}]
    cp = write_cfg("pilot", cfgs)
    root = TMP / "pilot"
    for steps in (600, 1200, 2400, 4800):
        job(f"pilot_ho_{steps}", "pilot", ["ccsr1_router_job.py", "--stage", f"PHO{steps}", "--configs", rel(cp), "--seed",
                                            "0", "--train", rel(tr), "--eval-part", "TRAIN", "--eval", rel(tr), "--banks",
                                            "SEM", "--steps", str(steps), "--train-papers-frac", "0.8", "--out-root",
                                            rel(root)], disk_gb=1.0)
    for steps in (600, 1200, 2400, 4800):
        for rep in ("a", "b") if steps == 600 else ("a",):
            job(f"pilot_{steps}_{rep}", "pilot", ["ccsr1_router_job.py", "--stage", f"P{steps}{rep}", "--configs", rel(cp),
                                                   "--seed", "0", "--train", rel(tr), "--eval-part", "TRAIN", "--eval",
                                                   rel(tr), "--banks", "SEM", "--steps", str(steps), "--out-root", rel(root)],
                disk_gb=1.0)


# ------------------------------------------------------------------------------------------------ stages
def stage1_configs():
    from ccsr1_configs import stage1
    return stage1(GEN)


def stage1():
    if (ART / f"stage1{SFX}.json").exists():
        return json.loads((ART / f"stage1{SFX}.json").read_text())
    from csr.ccsr1_paths import check_prereg
    check_prereg()
    LG.set_state(f"STAGE1{SFX}_RUNNING")
    tr = extract("TRAIN", "stage1")
    sel = extract("SEL", "stage1")
    cp = write_cfg(f"S1{SFX}", stage1_configs())
    for s in ROUTER_SEEDS:
        for leg in range(1, 6):
            job(f"routers_S1{SFX}_s{s}_leg{leg}", "stage1",
                ["ccsr1_router_job.py", "--stage", f"S1{SFX}", "--configs", rel(cp), "--seed", str(s), "--train", rel(tr),
                 "--eval-part", "SEL", "--eval", rel(sel)], disk_gb=2.0, seed=s)
            if all((ROUTERS / f"S1{SFX}" / c["tag"] / f"{b}_s{s}_SEL.npz").exists()
                   for c in json.loads(cp.read_text()) for b in ("SEM", "SHUF")):
                break
    job(f"analyze_stage1{SFX}", "stage1", ["ccsr1_analyze.py", "stage1", "--gen", str(GEN)], gpu=False, disk_gb=1.0)
    s1 = json.loads((ART / f"stage1{SFX}.json").read_text())
    return s1


def main():
    global CMD_NOW
    what = sys.argv[1]
    CMD_NOW = " ".join(sys.argv[1:])
    guard.check_root()
    TMP.mkdir(parents=True, exist_ok=True)
    if what == "extract_train":
        extract("TRAIN", "pilot")
    elif what == "pilot":
        pilot()
    elif what in ("stage1", "all"):
        s1 = stage1()
        if not s1["stage1_pass"]:
            LG.set_state(f"STAGE1{SFX}_FAIL")
            LG.write_resume(f"STAGE1{SFX}_FAIL", "true  # engineering generation tied to the measured failure (prereg)")
            return
        if what == "all":
            import ccsr1_stage2
            ccsr1_stage2.run(job, GEN, SFX)
    elif what == "stage2":
        import ccsr1_stage2
        ccsr1_stage2.run(job, GEN, SFX)
    elif what in ("stage3a", "stage3b", "efficiency"):
        import ccsr1_stage3
        getattr(ccsr1_stage3, what)(job)


if __name__ == "__main__":
    main()
