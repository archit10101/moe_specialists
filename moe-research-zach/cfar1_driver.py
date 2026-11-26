#!/usr/bin/env python
# csr-centered-fedavg-residual-v1 (cfar1): path-only copy of csur1_driver.py (csur1: copy of jsr_driver.py) (jsr: copy of orv1_driver.py; orv1: copy of scrgp_driver.py).
"""csr-semantic-lift-expert-learning-v1 driver (copy of scu_driver.py / slk_driver.py with sll paths). Idempotent job runner with an atomic driver_state.json,
append-only driver_log.txt / run_ledger.tsv, guarded GPU launches (scripts/gpu_guarded.sh), disk-reserve checks and the
external-interruption policy: a job that ends with rc 77/78 (signal, STOP marker, foreign GPU process) or a blocked
launcher (rc 3) marks the job INCOMPLETE, records the resume command, sets state["external_stop"] and the driver exits;
it never relaunches automatically (restart requires --authorized-resume).

  cfar1_driver.py job  <id> --gpu|--cpu [--expect-rc 0,75] [--disk-gb N] -- <script> [args...]
  cfar1_driver.py status
  cfar1_driver.py authorize-resume   (clears external_stop; only on explicit user instruction)"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402
from csr.cfar1_paths import ART, STOP_MARKER  # noqa: E402

STATE = ART / "driver_state.json"
LOG = ART / "driver_log.txt"
LEDGER = ART / "run_ledger.tsv"


import contextlib
import fcntl


@contextlib.contextmanager
def locked():
    with open(ART / ".driver.lock", "a") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def load_state():
    return json.loads(STATE.read_text()) if STATE.exists() else {"jobs": {}, "created": guard.now()}


def save_state(st):
    guard.atomic_write_json(STATE, st, overwrite=True)


def update_state(fn):
    with locked():
        st = load_state()
        fn(st)
        save_state(st)
        return st


def log(msg):
    line = f"{guard.now()} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def ledger(row):
    new = not LEDGER.exists()
    with open(LEDGER, "a") as f:
        if new:
            f.write("time\tjob\tkind\tstatus\trc\tseconds\tcmd\n")
        f.write("\t".join(str(x) for x in row) + "\n")


def run_job(job_id, argv, gpu=True, expect=(0,), disk_gb=2.0, force=False):
    st = load_state()
    if st.get("external_stop") and not force:
        log(f"REFUSE {job_id}: external stop recorded ({st['external_stop']}); needs --authorized-resume")
        sys.exit(77)
    if STOP_MARKER.exists():
        log(f"REFUSE {job_id}: STOP marker present")
        sys.exit(77)
    j = st["jobs"].get(job_id)
    if j and j.get("status") == "COMPLETE" and not force:
        log(f"skip {job_id}: COMPLETE")
        return 0
    guard.require_disk(int(disk_gb * 1024**3), f"cfar1 job {job_id}")
    py = str(guard.ROOT / ".venv/bin/python")
    if gpu:
        cmd = ["scripts/gpu_guarded.sh", f"cfar1 {job_id}"] + argv + ["--device", "cuda", "--execute"]
        env = {**os.environ, "MOE_GPU_LAUNCH": "1", "MOE_RESEARCH_APPROVE": "gpu"}
    else:
        cmd = [py] + argv + ["--execute"]
        env = {**os.environ, "MOE_RESEARCH_APPROVE": "cpu-model"}
    update_state(lambda s_: (s_["jobs"].__setitem__(job_id, {"status": "RUNNING", "start": guard.now(), "cmd": cmd,
                                                             "gpu": gpu}), s_.__setitem__("active_" + ("gpu" if gpu else "cpu"), job_id)))
    log(f"START {job_id}: {' '.join(cmd)}")
    t0 = time.time()
    outp = ART / "job_logs" / f"{job_id}.log"
    outp.parent.mkdir(exist_ok=True)
    with open(outp, "a") as fo:
        rc = subprocess.run(cmd, env=env, stdout=fo, stderr=subprocess.STDOUT, cwd=guard.ROOT).returncode
    dt = time.time() - t0
    if rc in expect:
        status = "COMPLETE"
    elif rc in (77, 78, 3, 124, 130, -2, -15):
        status = "INCOMPLETE"
    else:
        status = "FAILED"

    def fin(s_):
        if status == "INCOMPLETE":
            s_["state"] = "stopped_by_operator"
            s_["external_stop"] = {"job": job_id, "rc": rc, "time": guard.now(), "cmd": cmd,
                                   "resume": f"cfar1_driver.py authorize-resume, then rerun the same phase command"}
        s_["jobs"][job_id].update({"status": status, "rc": rc, "end": guard.now(), "seconds": round(dt, 1)})
        s_["active_" + ("gpu" if gpu else "cpu")] = None
    update_state(fin)
    ledger([guard.now(), job_id, "gpu" if gpu else "cpu", status, rc, round(dt, 1), " ".join(argv)])
    log(f"END {job_id}: {status} rc={rc} {dt:.0f}s (log {outp.relative_to(guard.ROOT)})")
    if status == "INCOMPLETE":
        log("EXTERNAL INTERRUPTION or GPU contention: driver stops; no automatic relaunch")
        sys.exit(77)
    if status == "FAILED":
        sys.exit(1)
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["job", "status", "authorize-resume"])
    ap.add_argument("job_id", nargs="?")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--expect-rc", default="0")
    ap.add_argument("--disk-gb", type=float, default=2.0)
    ap.add_argument("--force", action="store_true")
    argv = sys.argv[1:]
    rest = []
    if "--" in argv:
        i = argv.index("--")
        argv, rest = argv[:i], argv[i + 1:]
    a = ap.parse_args(argv)
    guard.check_root()
    if a.cmd == "status":
        print(json.dumps(load_state(), indent=1)[:20000])
        return
    if a.cmd == "authorize-resume":
        update_state(lambda st: (st.setdefault("authorized_resumes", []).append(
            {"time": guard.now(), "cleared": st.pop("external_stop", None)}), st.__setitem__("state", "running")))
        log("authorized resume recorded (explicit user instruction)")
        return
    run_job(a.job_id, rest, gpu=a.gpu, expect=tuple(int(x) for x in a.expect_rc.split(",")), disk_gb=a.disk_gb,
            force=a.force)


if __name__ == "__main__":
    main()
