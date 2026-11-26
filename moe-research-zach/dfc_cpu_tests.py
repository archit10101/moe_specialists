#!/usr/bin/env python
"""csr-domain-forecast-curriculum-v1: run the CPU test suite with CUDA hidden and record the result
(artifacts/domain-forecast-curriculum-v1/cpu_tests_<stage>_<stamp>.json with code hashes).
  dfc_cpu_tests.py pre   -> tests/test_dfc_v1.py (resume test skipped until caches/configs exist)
  dfc_cpu_tests.py full  -> tests/test_dfc_v1.py incl. the real-script CPU resume-equality test"""
import json, os, re, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402
from csr.dfc_lib import ART, code_hashes, prereg_sha  # noqa: E402
def main():
    guard.check_root()
    stage = sys.argv[1]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4")
    t0 = time.time()
    targets = ["tests/test_dfc_v1.py"]
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rs"] + targets,
                       cwd=guard.ROOT, env=env, capture_output=True, text=True)
    lines = [ln for ln in p.stdout.strip().splitlines() if re.search(r"\d+ (passed|failed)", ln)]
    tail = lines[-1] if lines else ""
    counts = {k: int(v) for v, k in re.findall(r"(\d+) (passed|failed|skipped|error|errors)", tail)}
    status = "PASS" if p.returncode == 0 and not counts.get("failed") and not counts.get("error") \
        and not counts.get("errors") else "FAIL"
    doc = {"stage": stage, "status": status, "returncode": p.returncode, "summary": tail, "counts": counts,
           "seconds": time.time() - t0, "time": guard.now(), "code_sha256": code_hashes(),
           "prereg_sha256": prereg_sha(), "cuda_visible_devices": "", "stdout_tail": p.stdout[-8000:],
           "stderr_tail": p.stderr[-3000:]}
    out = ART / f"cpu_tests_{stage}_{guard.utc_stamp()}.json"
    guard.atomic_write_json(out, doc)
    guard.ledger("DFC_CPU_TESTS", f"{out.relative_to(guard.ROOT)} {status} {tail}")
    print(status, tail, out)
    sys.exit(0 if status == "PASS" else 1)
if __name__ == "__main__":
    main()
