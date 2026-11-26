#!/usr/bin/env python
"""csr-domain-forecast-curriculum-v1: protected-artifact inventory (read-only) and verification.
Everything vm_inventory.collect() protects plus every vrc/m2d2 file (previous experiment), every other pre-existing
root file; excludes locked test-split files by name (never opened) and this experiment's own new files (dfc_ prefixes,
domain-forecast-curriculum-v1 dirs, DOMAIN_FORECAST_CURRICULUM_V1 docs, runs/*_dfc_*).
  dfc_inventory.py --out <json> | --verify <json>"""
import argparse, json, sys
from pathlib import Path
ROOT = Path("/home/zapan4/moe-research")
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
import ms_inventory as msi  # noqa: E402
import vm_inventory as vmi  # noqa: E402
OWN = ("artifacts/domain-forecast-curriculum-v1", "configs/domain-forecast-curriculum-v1",
       ".tmp/domain-forecast-curriculum-v1", "dfc_", "src/csr/dfc_", "tests/test_dfc_", "docs/DOMAIN_FORECAST_CURRICULUM_V1",
       "DOMAIN_FORECAST_CURRICULUM_V1")
EXTRA_DIRS = ["artifacts/value-routed-compute-v1", "artifacts/m2d2-domain-prior-v1", "configs/value-routed-compute-v1",
              "configs/m2d2-domain-prior-v1", ".tmp/m2d2-domain-prior-v1", ".tmp/value-routed-compute-v1", "runs", "src",
              "tests", "docs", "scripts", "configs", "ledger"]
def collect():
    files = set(vmi.collect())
    for d in EXTRA_DIRS:
        if (ROOT / d).exists():
            files.update(msi.walk(d))
    files.update(p.name for p in ROOT.iterdir() if p.is_file())
    return sorted(f for f in files if not msi.is_test_split_name(f) and not f.startswith(OWN) and "_dfc_" not in f
                  and "__pycache__" not in f and not f.endswith(vmi.cli.VOLATILE_SUFFIX) and not f.startswith("ledger/"))
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out"); ap.add_argument("--verify"); a = ap.parse_args()
    from csr import guard
    guard.check_root()
    if a.out:
        out = guard.inside_root(a.out); files = collect()
        rec = {f: {"sha256": msi.sha256(ROOT / f), "bytes": (ROOT / f).stat().st_size} for f in files}
        guard.atomic_write_json(out, {"created": guard.now(), "n_files": len(rec), "files": rec}, mode=0o400)
        guard.ledger("DFC_INVENTORY", f"{a.out} n_files={len(rec)} sha256={guard.sha256_file(out)}")
        print(json.dumps({"n_files": len(rec), "bytes": sum(r["bytes"] for r in rec.values())}))
    else:
        src = guard.inside_root(a.verify); inv = json.loads(src.read_text())["files"]; bad = []
        for f, r in inv.items():
            p = ROOT / f
            if not p.exists(): bad.append((f, "MISSING"))
            elif msi.sha256(p) != r["sha256"]: bad.append((f, "MISMATCH"))
        out = src.with_name(src.stem + f".verify_{guard.utc_stamp()}.json")
        doc = {"verified": guard.now(), "n_files": len(inv), "n_bad": len(bad), "bad": bad, "all_match": not bad}
        guard.atomic_write_json(out, doc, mode=0o400)
        guard.ledger("DFC_VERIFY", f"{out.relative_to(ROOT)} n={len(inv)} bad={len(bad)}")
        print(json.dumps(doc)[:3000]); sys.exit(0 if not bad else 1)
if __name__ == "__main__":
    main()
