#!/usr/bin/env python
"""csr-semantic-edl-v1 (sem): protected-artifact inventory (read-only) and verification.
Protects every pre-existing project file (top-level files, src/, docs/, configs/, scripts/, runs/, artifacts/ except
this experiment's, and the resumable .tmp state of earlier experiments), including all fle, fcu and gc outputs. Excludes
locked test-split files (never opened), shared ledgers, caches, __pycache__ and this experiment's own files.
  sem_inventory.py --out <json> | --verify <json>"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path("/home/zapan4/moe-research")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
import ms_inventory as msi  # noqa: E402

OWN = ("artifacts/semantic-edl-v1", "configs/semantic-edl-v1", ".tmp/semantic-edl-v1", "sem_", "src/csr/sem_",
       "docs/SEMANTIC_EDL_V1", "SEMANTIC_EDL_V1")
DIRS = ["src", "docs", "configs", "scripts", "runs", "artifacts", ".tmp/fspv2", ".tmp/latent-forecast-router-distill-v1",
        ".tmp/context-prior-specialization-v2", ".tmp/context-prior-fast-engineering", ".tmp/forecast-latent-edl-v1", ".tmp/gc-edl-v1"]


def collect():
    files = set()
    for d in DIRS:
        if (ROOT / d).exists():
            files.update(msi.walk(d))
    files.update(p.name for p in ROOT.iterdir() if p.is_file())
    return sorted(f for f in files if not msi.is_test_split_name(f) and not f.startswith(OWN)
                  and "semantic_edl_v1" not in f and "__pycache__" not in f
                  and "/locked/" not in f and not f.startswith("ledger/") and not f.endswith(".lock"))


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
        guard.atomic_write_json(out, {"created": guard.now(), "n_files": len(rec), "files": rec}, mode=0o400)
        guard.ledger("SEM_INVENTORY", f"{a.out} n_files={len(rec)} sha256={guard.sha256_file(out)}")
        print(json.dumps({"n_files": len(rec), "bytes": sum(r["bytes"] for r in rec.values())}))
    else:
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
        doc = {"verified": guard.now(), "n_files": len(inv), "n_bad": len(bad), "bad": bad, "all_match": not bad}
        guard.atomic_write_json(out, doc, mode=0o400)
        guard.ledger("SEM_VERIFY", f"{out.relative_to(ROOT)} n={len(inv)} bad={len(bad)}")
        print(json.dumps(doc)[:3000])
        sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
