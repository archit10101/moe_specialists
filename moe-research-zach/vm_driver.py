#!/usr/bin/env python
"""Single resumable driver for csr-value-routed-compute-v1 (A) and csr-m2d2-domain-prior-v1 (B).
Absolute wall clock (America/New_York, 2026-11-09): no GPU launch after 08:35; every launched job gets --stop-at 08:45
(save + INTERRUPTED); reports by 08:55. Tasks are idempotent: a task whose artifact exists is skipped, so the driver can
be re-run after any interruption. One GPU job at a time; GPU occupancy checked before every launch.
  .venv/bin/python vm_driver.py   (CPU process; launches GPU jobs only through scripts/gpu_guarded.sh)
"""
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402
from csr.vm_run import EXP, LAUNCH_DEADLINE, STOP_DEADLINE  # noqa: E402

ROOT = guard.ROOT
ST = EXP["m2d2"]["st"] / "driver_state.json"
LOG = EXP["m2d2"]["st"] / "driver.log"
M2 = EXP["m2d2"]["cfg"]
SEARCH = [M2 / "search" / f"s100_{c}.json" for c in ("B0", "B1", "B2", "B3")]
FINAL_CORE = [M2 / "final" / f"s{s}_{c}.json" for s in (0, 1) for c in ("B0", "B1", "B2")]
FINAL_B3 = [M2 / "final" / f"s{s}_B3.json" for s in (0, 1)]


def log(msg):
    line = f"{guard.now()} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def state():
    return json.loads(ST.read_text()) if ST.exists() else {"tasks": {}, "durations": {}}


def save(s):
    guard.atomic_write_json(ST, s, overwrite=True)


def gpu_free():
    out = subprocess.run(["nvidia-smi", "-i", "0", "-q", "-d", "PIDS"], capture_output=True, text=True).stdout
    return out.count("Process ID") == 0


def disk_ok():
    free = guard.disk_free_bytes()
    if free < 40 * 1024**3:
        log(f"STOP: disk free {free / 1024**3:.1f} GiB < 40 GiB")
        return False
    return True


def complete_run(cfg_path):
    sha = guard.sha256_file(cfg_path)
    for r in sorted(glob.glob(str(ROOT / "runs" / "*_m2d2_train_*")) + sorted(glob.glob(str(ROOT / "runs" / "*_vrc_train_*")))):
        r = Path(r)
        try:
            if guard.sha256_file(r / "config.json") == sha and \
                    json.loads((r / "status.json").read_text())["status"] == "COMPLETE" and (r / "model.pt").exists():
                return r
        except FileNotFoundError:
            continue
    return None


def interrupted_run(cfg_path):
    sha = guard.sha256_file(cfg_path)
    best = None
    for r in sorted(glob.glob(str(ROOT / "runs" / "*_m2d2_train_*"))):
        r = Path(r)
        try:
            if guard.sha256_file(r / "config.json") == sha and (r / "train_state.pt").exists() and \
                    json.loads((r / "status.json").read_text())["status"] == "INTERRUPTED":
                best = r
        except FileNotFoundError:
            continue
    return best


def launch(purpose, args):
    """One guarded GPU job; returns (exit code, seconds). Refuses after the launch deadline or if the GPU is busy."""
    if time.time() > LAUNCH_DEADLINE:
        log(f"SKIP launch '{purpose}': past 08:35 launch deadline")
        return None, 0.0
    if not gpu_free():
        log(f"STOP: foreign/other process on GPU0 before '{purpose}'")
        guard.ledger("VM_DRIVER_STOP", f"GPU occupied before {purpose}")
        sys.exit(3)
    if not disk_ok():
        sys.exit(4)
    env = dict(os.environ, MOE_GPU_LAUNCH="1", MOE_RESEARCH_APPROVE="gpu")
    t0 = time.time()
    log(f"LAUNCH {purpose}: {' '.join(args)}")
    with open(EXP["m2d2"]["st"] / "logs" / f"{guard.utc_stamp()}_{purpose.replace(' ', '_')}.log", "w") as f:
        p = subprocess.run(["scripts/gpu_guarded.sh", purpose] + args, cwd=ROOT, env=env, stdout=f,
                           stderr=subprocess.STDOUT)
    dt = time.time() - t0
    log(f"DONE {purpose}: exit {p.returncode} in {dt:.0f}s")
    return p.returncode, dt


def train(cfg_path, s, est_s=None):
    name = cfg_path.stem + "_" + cfg_path.parent.name
    r = complete_run(cfg_path)
    if r is not None:
        s["tasks"][f"train:{name}"] = {"status": "COMPLETE", "run": str(r.relative_to(ROOT))}
        return r
    if est_s is not None and time.time() + est_s > LAUNCH_DEADLINE:
        log(f"SKIP train {name}: projected end {time.ctime(time.time() + est_s)} after 08:35")
        s["tasks"][f"train:{name}"] = {"status": "NOT_STARTED_DEADLINE"}
        save(s)
        return None
    args = ["vm_train.py", "--config", str(cfg_path.relative_to(ROOT)), "--device", "cuda",
            "--stop-at", str(STOP_DEADLINE), "--execute"]
    prev = interrupted_run(cfg_path)
    if prev is not None:
        args += ["--resume", str(prev.relative_to(ROOT))]
    rc, dt = launch(f"m2d2 train {name}", args)
    if rc is None:
        s["tasks"][f"train:{name}"] = {"status": "NOT_STARTED_DEADLINE"}
        save(s)
        return None
    r = complete_run(cfg_path)
    if r is None and rc != 0:
        # one safe retry for non-finite values only
        last = sorted(glob.glob(str(ROOT / "runs" / f"*_m2d2_train_{json.loads(cfg_path.read_text())['run_name']}*")))[-1]
        reason = json.loads((Path(last) / "summary.json").read_text()).get("reason", "")
        if ("non-finite" in reason or "FloatingPoint" in reason) and not s["tasks"].get(f"retry:{name}"):
            s["tasks"][f"retry:{name}"] = True
            save(s)
            log(f"RETRY {name} once after non-finite: {reason}")
            rc, dt = launch(f"m2d2 train {name} retry", args)
            r = complete_run(cfg_path)
            if r is None:
                log(f"STOP: non-finite values persist after one retry for {name}")
                sys.exit(5)
    s["durations"][cfg_path.parent.name] = max(s["durations"].get(cfg_path.parent.name, 0), dt)
    s["tasks"][f"train:{name}"] = {"status": "COMPLETE" if r else "INCOMPLETE",
                                   "run": str(r.relative_to(ROOT)) if r else last_run(cfg_path)}
    save(s)
    return r


def last_run(cfg_path):
    rn = json.loads(cfg_path.read_text())["run_name"]
    xs = sorted(glob.glob(str(ROOT / "runs" / f"*_m2d2_train_{rn}*")))
    return str(Path(xs[-1]).relative_to(ROOT)) if xs else None


def evaluate(run, s, ablation):
    out = EXP["m2d2"]["art"] / "evals" / f"val__{run.name}"
    if (out / "summary.json").exists() and json.loads((out / "summary.json").read_text())["status"] == "COMPLETE":
        s["tasks"][f"eval:{run.name}"] = "COMPLETE"
        return True
    if out.exists():
        log(f"eval dir exists but incomplete for {run.name}; not overwriting (left INCOMPLETE)")
        return False
    args = ["m2d2_evaluate.py", "--run", str(run.relative_to(ROOT)), "--evalset", "val", "--execute"]
    if ablation:
        args.insert(-1, "--ablation")
    rc, _ = launch(f"m2d2 eval {run.name}", args)
    s["tasks"][f"eval:{run.name}"] = "COMPLETE" if rc == 0 else f"FAILED rc={rc}"
    save(s)
    return rc == 0


def freeze():
    p = EXP["m2d2"]["art"] / "frozen.json"
    if p.exists():
        return
    cfgs = SEARCH + FINAL_CORE + FINAL_B3
    guard.atomic_write_json(p, {
        "time": guard.now(), "config_sha256s": [guard.sha256_file(c) for c in cfgs],
        "configs": [str(c.relative_to(ROOT)) for c in cfgs],
        "statement": "All settings were fixed by the protocol before training; the 10M search results select nothing. "
                     "Validation is opened only now, once per checkpoint.",
        "protocol_sha256": guard.sha256_file(EXP["m2d2"]["protocol"])}, mode=0o400)
    guard.ledger("M2D2_FROZEN", str(p))
    log("FROZEN m2d2 settings (frozen.json written)")


def main():
    guard.check_root()
    s = state()
    log(f"driver start; launch deadline {time.ctime(LAUNCH_DEADLINE)}, stop {time.ctime(STOP_DEADLINE)}")
    # ---------------------------------------------------------------- A: diagnostic gate already decided?
    diags = sorted(glob.glob(str(EXP["vrc"]["art"] / "diagnostic_*.json")))
    if diags:
        dg = json.loads(Path(diags[-1]).read_text())
        s["tasks"]["vrc:diagnostic"] = {"status": dg["status"], "gate_pass": dg.get("gate_pass"), "file": diags[-1]}
        if dg.get("status") == "COMPLETE" and not dg.get("gate_pass"):
            s["tasks"]["vrc:pilot"] = "NOT_RUN (diagnostic gate failed; Experiment A stopped per protocol A.5)"
    save(s)
    # ---------------------------------------------------------------- B
    for c in SEARCH:
        train(c, s)
    freeze()
    for c in FINAL_CORE:
        est = s["durations"].get("final")
        r = train(c, s, est_s=est)
        if r is not None:
            evaluate(r, s, ablation=True)
    # search-run validation evaluations (nothing is selected from them; screening evidence)
    for c in SEARCH:
        r = complete_run(c)
        if r is not None:
            evaluate(r, s, ablation=False)
    # B3 at the final budget only if it can finish before 08:35 (train + eval)
    for c in FINAL_B3:
        est = s["durations"].get("final", 600) + 120
        r = train(c, s, est_s=est)
        if r is not None:
            evaluate(r, s, ablation=True)
    # A post-hoc timing breakdown (diagnostic explanation only; never changes the gate)
    pt = sorted(glob.glob(str(EXP["vrc"]["art"] / "posthoc_timing_*.json")))
    if not pt and (ROOT / "vrc_posthoc_timing.py").exists() and diags:
        run = json.loads(Path(diags[-1]).read_text())["run"]
        if time.time() + 300 < LAUNCH_DEADLINE:
            launch("vrc posthoc timing", ["vrc_posthoc_timing.py", "--run", run, "--execute"])
    s["tasks"]["driver"] = {"status": "DONE", "time": guard.now()}
    save(s)
    log("driver done; GPU work finished")


if __name__ == "__main__":
    main()
