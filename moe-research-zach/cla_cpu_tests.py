#!/usr/bin/env python
"""csr-conditional-local-attention-v1: run the COMPLETE CPU test suite with CUDA hidden and record the result
(artifacts/conditional-local-attention-v1/cpu_tests_<stamp>.json, with code hashes so the driver can detect changes)."""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402
from csr.cla_run import ART, code_hashes, prereg_sha  # noqa: E402


def main():
    guard.check_root()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4")
    t0 = time.time()
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"], cwd=guard.ROOT, env=env, capture_output=True, text=True)
    tail = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
    counts = {k: int(v) for v, k in re.findall(r"(\d+) (passed|failed|skipped|error|errors)", tail)}
    status = "PASS" if p.returncode == 0 and counts.get("failed", 0) == 0 and not counts.get("error") \
        and not counts.get("errors") else "FAIL"
    doc = {"status": status, "returncode": p.returncode, "summary": tail, "counts": counts,
           "seconds": time.time() - t0, "time": guard.now(), "code_sha256": code_hashes(),
           "prereg_sha256": prereg_sha(), "cuda_visible_devices": "", "stdout_tail": p.stdout[-6000:],
           "stderr_tail": p.stderr[-3000:]}
    out = ART / f"cpu_tests_{guard.utc_stamp()}.json"
    guard.atomic_write_json(out, doc)
    guard.ledger("CLA_V1_CPU_TESTS", f"{out.relative_to(guard.ROOT)} {status} {tail}")
    print(status, tail, out)
    sys.exit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()
