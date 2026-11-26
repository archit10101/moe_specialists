#!/usr/bin/env python
"""csr-value-routed-compute-v1 + csr-m2d2-domain-prior-v1: protected-artifact inventory (read-only) and verification.
Everything cla_inventory.collect() protects, plus every clav1 file; excludes locked test-split files by name (never
opened) and this session's new files (vrc_/m2d2_/vm_ prefixes and their artifact/config/.tmp dirs).
  vm_inventory.py --out <json> | --verify <json>"""
import argparse, json, sys
from pathlib import Path
ROOT = Path("/home/zapan4/moe-research")
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
import cla_inventory as cli  # noqa: E402
import ms_inventory as msi  # noqa: E402
OWN = ("artifacts/value-routed-compute-v1", "artifacts/m2d2-domain-prior-v1", "configs/value-routed-compute-v1",
       "configs/m2d2-domain-prior-v1", ".tmp/value-routed-compute-v1", ".tmp/m2d2-domain-prior-v1", "vrc_", "m2d2_",
       "vm_", "src/csr/vrc_", "src/csr/m2d2_", "src/csr/vm_", "tests/test_vrc_", "tests/test_m2d2_",
       "docs/VALUE_ROUTED_COMPUTE_V1", "docs/M2D2_DOMAIN_PRIOR_V1", "VALUE_ROUTED_COMPUTE_V1", "M2D2_DOMAIN_PRIOR_V1")
def collect():
    files = set(cli.collect())
    for d in cli.DIRS + [".tmp/conditional_local_attention_v1"]:
        if (ROOT / d).exists():
            files.update(msi.walk(d))
    return sorted(f for f in files if not msi.is_test_split_name(f) and not f.startswith(OWN) and "_vrc_" not in f
                  and "_m2d2_" not in f and "__pycache__" not in f and not f.endswith(cli.VOLATILE_SUFFIX))
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out"); ap.add_argument("--verify"); a = ap.parse_args()
    from csr import guard
    guard.check_root()
    if a.out:
        out = guard.inside_root(a.out); files = collect()
        rec = {f: {"sha256": msi.sha256(ROOT / f), "bytes": (ROOT / f).stat().st_size} for f in files}
        guard.atomic_write_json(out, {"created": guard.now(), "n_files": len(rec), "files": rec}, mode=0o400)
        guard.ledger("VM_INVENTORY", f"{a.out} n_files={len(rec)} sha256={guard.sha256_file(out)}")
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
        guard.ledger("VM_VERIFY", f"{out.relative_to(ROOT)} n={len(inv)} bad={len(bad)}")
        print(json.dumps(doc)[:3000]); sys.exit(0 if not bad else 1)
if __name__ == "__main__":
    main()
