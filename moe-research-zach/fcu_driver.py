#!/usr/bin/env python
"""csr-forecast-counterfactual-utility-v1 job driver (prereg section 4).

  fcu_driver.py job <id> --gpu|--cpu [--disk-gb X] -- <script> [args...]
      Runs one idempotent job (COMPLETE jobs are skipped). GPU jobs go only through scripts/gpu_guarded.sh with
      --device cuda --execute appended. Exit code 77/78 (EXTERNALLY_STOPPED) or a blocked launch (rc 3) latches
      driver_state["external_stop"]; afterwards every job is refused until `authorize-resume` (explicit human OK).
  fcu_driver.py status | authorize-resume
"""
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402
from csr.fcu_lib import ART  # noqa: E402

STATE = ART / "driver_state.json"
LOG = ART / "driver_log.txt"
LOCK = ART / ".driver.lock"


def load():
    return json.loads(STATE.read_text()) if STATE.exists() else {"jobs": {}, "external_stop": None}


def save(st):
    guard.atomic_write_json(STATE, st, overwrite=True)


def log(msg):
    with open(LOG, "a") as f:
        f.write(f"{guard.now()} {msg}\n")


def main():
    guard.check_root()
    a = sys.argv[1:]
    if not a:
        print(__doc__)
        return
    with open(LOCK, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        st = load()
        if a[0] == "status":
            print(json.dumps(st, indent=1))
            return
        if a[0] == "authorize-resume":
            log(f"authorize-resume (was external_stop={st.get('external_stop')})")
            st["external_stop"] = None
            save(st)
            return
        assert a[0] == "job"
        jid = a[1]
        sep = a.index("--")
        opts, cmd = a[2:sep], a[sep + 1:]
        gpu = "--gpu" in opts
        disk_gb = float(opts[opts.index("--disk-gb") + 1]) if "--disk-gb" in opts else 1.0
        if st.get("external_stop"):
            print(f"REFUSED: external stop latched ({st['external_stop']})")
            sys.exit(4)
        if st["jobs"].get(jid, {}).get("status") == "COMPLETE":
            print(f"{jid}: already COMPLETE")
            return
        guard.require_disk(int(disk_gb * 2 ** 30), f"fcu job {jid}")
        if (ART / "STOP").exists():
            print("REFUSED: STOP marker present")
            sys.exit(4)
        if gpu:
            full = ["scripts/gpu_guarded.sh", f"fcu {jid}"] + cmd + ["--device", "cuda", "--execute"]
            env = {**os.environ, "MOE_GPU_LAUNCH": "1", "MOE_RESEARCH_APPROVE": "gpu"}
        else:
            full = [".venv/bin/python"] + cmd + ["--execute"]
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "MOE_RESEARCH_APPROVE": "cpu-model"}
        st["jobs"][jid] = {"status": "RUNNING", "cmd": full, "start": guard.now()}
        save(st)
    log(f"START {jid}: {' '.join(full)}")
    t0 = time.time()
    lp = ART / "job_logs" / f"{jid}.log"
    with open(lp, "a") as f:
        rc = subprocess.call(full, stdout=f, stderr=subprocess.STDOUT, env=env)
    dt = time.time() - t0
    with open(LOCK, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        st = load()
        if rc == 0:
            status = "COMPLETE"
        elif rc in (77, 78):
            status = "EXTERNALLY_STOPPED"
            st["external_stop"] = f"{jid} rc={rc} at {guard.now()}"
        elif rc == 3:
            status = "BLOCKED"
            st["external_stop"] = f"{jid} launch blocked (GPU occupied) at {guard.now()}"
        else:
            status = "FAILED"
        st["jobs"][jid].update({"status": status, "rc": rc, "end": guard.now(), "seconds": round(dt, 1)})
        save(st)
    log(f"END {jid}: {status} rc={rc} {dt:.0f}s (log {lp.relative_to(guard.ROOT)})")
    sys.exit(0 if status == "COMPLETE" else 1)


if __name__ == "__main__":
    main()
