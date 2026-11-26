#!/usr/bin/env python
"""Delete this experiment's own superseded scratch directories (under .tmp/csr-centered-fedavg-residual-v1
or its gpu_tests runs only), logging sha256 + bytes of every file to ART/deletions.tsv.
  csur1_cleanup.py DIR [DIR ...] --reason TEXT"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402
from csr import cfar1_log as L  # noqa: E402
from csr.cfar1_paths import TMP  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("dirs", nargs="+")
ap.add_argument("--reason", required=True)
a = ap.parse_args()
tot = 0
for d in a.dirs:
    p = Path(d).resolve()
    if not str(p).startswith(str(TMP.resolve())):
        raise SystemExit(f"refusing to delete outside {TMP}: {p}")
    if not p.exists():
        continue
    for f in sorted(x for x in p.rglob("*") if x.is_file()):
        nb = f.stat().st_size
        tot += nb
        sha = guard.sha256_file(f) if nb > 0 else "empty"
        L.deletion(str(f.relative_to(guard.ROOT)), sha, nb, a.reason)
    shutil.rmtree(p)
print(f"deleted {tot / 2**30:.2f} GiB")
