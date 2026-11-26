#!/usr/bin/env python
"""csr-sequence-routing-search-v1: protected-artifact inventory (read-only) and post-run verification.

Extends ms_inventory.collect() (rep-v1 data/caches/configs/checkpoints, audits, docs, all pre-existing sources) with
every file that existed under runs/, artifacts/, configs/ and the reports/manifests before this experiment. Files of
the locked test split (artifacts/data/**/test_*) are excluded BY NAME and never opened.

  srs_inventory.py --out artifacts/sequence-routing-search-v1/protected_inventory.json
  srs_inventory.py --verify artifacts/sequence-routing-search-v1/protected_inventory.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path("/home/zapan4/moe-research")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
import ms_inventory as msi  # noqa: E402  (read-only reuse; ms_inventory.py itself is unchanged)

EXTRA_DIRS = ["runs", "artifacts", "configs", "docs"]
EXTRA_FILES = ["EXPERIMENT_MANIFEST.json", "EXPERIMENT_MANIFEST.v1.superseded.json", "EXPERIMENT_REPORT.md",
               "MULTISCALE_ROUTING_V1_REPORT.md", "SPECIALIZATION_AUDIT_REPORT.md", "MODEL_DECISION.md",
               "PRIOR_ART.md", "README.md", "SAFETY.md", "env.sh", "requirements-cpu.lock.txt",
               "requirements-cuda.lock.txt"]
OWN_PREFIXES = ("artifacts/sequence-routing-search-v1", "configs/sequence-routing-search-v1")


def own(rel):
    return rel.startswith(OWN_PREFIXES) or "_srsv1_" in rel or rel.startswith("docs/SEQUENCE_ROUTING_SEARCH_V1")


def collect():
    files = msi.collect()
    for d in EXTRA_DIRS:
        files += [f for f in msi.walk(d) if not own(f)]
    files += [f for f in EXTRA_FILES if (ROOT / f).exists()]
    seen, out = set(), []
    for f in files:
        if f not in seen and not msi.is_test_split_name(f) and not own(f):
            seen.add(f)
            out.append(f)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--verify")
    a = ap.parse_args()
    from csr import guard
    guard.check_root()
    if a.out:
        out = guard.inside_root(a.out)
        files = collect()
        rec = {f: {"sha256": msi.sha256(ROOT / f), "bytes": (ROOT / f).stat().st_size} for f in files}
        doc = {"experiment": "csr-sequence-routing-search-v1", "created": guard.now(), "n_files": len(rec),
               "excluded": "locked test-split files under artifacts/data/ (excluded by name; never opened)",
               "files": rec, "environment": msi.env_state()}
        guard.atomic_write_json(out, doc, mode=0o400)
        guard.ledger("SRS_V1_INVENTORY", f"{a.out} n_files={len(rec)} sha256={guard.sha256_file(out)}")
        print(json.dumps({"n_files": len(rec), "bytes": sum(r["bytes"] for r in rec.values()),
                          "environment": doc["environment"]}, indent=1))
    elif a.verify:
        src = guard.inside_root(a.verify)
        inv = json.loads(src.read_text())["files"]
        bad = []
        for f, r in inv.items():
            p = ROOT / f
            if not p.exists():
                bad.append((f, "MISSING"))
            elif msi.sha256(p) != r["sha256"]:
                bad.append((f, "MISMATCH"))
        out = src.with_name(src.stem + f".verify_{guard.utc_stamp()}.json")
        doc = {"verified": guard.now(), "n_files": len(inv), "n_bad": len(bad), "bad": bad, "all_match": not bad,
               "environment": msi.env_state()}
        guard.atomic_write_json(out, doc, mode=0o400)
        guard.ledger("SRS_V1_VERIFY", f"{out.relative_to(ROOT)} n={len(inv)} bad={len(bad)}")
        print(json.dumps({k: v for k, v in doc.items() if k != "environment"})[:2000])
        print(f"VERIFY_JSON {out.relative_to(ROOT)}")
        sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
