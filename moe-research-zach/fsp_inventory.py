#!/usr/bin/env python
"""csr-forecast-specialist-pretraining-v2: protected-artifact inventory (read-only) and verification.
Protects everything dfc_inventory.collect() protects PLUS every csr-domain-forecast-curriculum-v1 (dfc) file, every
other pre-existing root file and all pre-existing runs. Excludes locked test-split files by name (never opened) and this
experiment's own files (fsp_ prefixes, forecast-specialist-pretraining-v2 dirs, FORECAST_SPECIALIST_PRETRAINING_V2
docs, runs/*_fsp_*).  fsp_inventory.py --out <json> | --verify <json>"""
import argparse, json, sys
from pathlib import Path
ROOT = Path("/home/zapan4/moe-research")
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
import ms_inventory as msi  # noqa: E402
import dfc_inventory as dfi  # noqa: E402
OWN = ("artifacts/forecast-specialist-pretraining-v2", "configs/forecast-specialist-pretraining-v2",
       ".tmp/fspv2", "fsp_", "src/csr/fsp_", "tests/test_fsp_", "docs/FORECAST_SPECIALIST_PRETRAINING_V2",
       "FORECAST_SPECIALIST_PRETRAINING_V2")
DFC_DIRS = ["artifacts/domain-forecast-curriculum-v1", "configs/domain-forecast-curriculum-v1"]
def collect():
    files = set(dfi.collect())
    for d in DFC_DIRS + ["runs", "src", "docs", "configs", "scripts"]:
        if (ROOT / d).exists():
            files.update(msi.walk(d))
    files.update(p.name for p in ROOT.iterdir() if p.is_file())
    return sorted(f for f in files if not msi.is_test_split_name(f) and not f.startswith(OWN) and "_fsp_" not in f
                  and "__pycache__" not in f and not f.startswith("ledger/") and not f.startswith(".tmp/")
                  and not f.endswith((".tmp", ".lock")))
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out"); ap.add_argument("--verify"); a = ap.parse_args()
    from csr import guard
    guard.check_root()
    if a.out:
        out = guard.inside_root(a.out); files = collect()
        rec = {f: {"sha256": msi.sha256(ROOT / f), "bytes": (ROOT / f).stat().st_size} for f in files}
        guard.atomic_write_json(out, {"created": guard.now(), "n_files": len(rec), "files": rec}, mode=0o400)
        guard.ledger("FSP_INVENTORY", f"{a.out} n_files={len(rec)} sha256={guard.sha256_file(out)}")
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
        guard.ledger("FSP_VERIFY", f"{out.relative_to(ROOT)} n={len(inv)} bad={len(bad)}")
        print(json.dumps(doc)[:3000]); sys.exit(0 if not bad else 1)
if __name__ == "__main__":
    main()
