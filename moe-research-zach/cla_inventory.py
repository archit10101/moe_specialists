#!/usr/bin/env python
"""csr-conditional-local-attention-v1: protected-artifact inventory (read-only) and post-run verification.

Reuses spv1_inventory.collect() / ms_inventory (read-only imports; those files are unchanged) and adds every file of
the previous experiment (csr-speculative-pool-routing-v1: artifacts, configs, runs, sources, report, state dir) that
existed before this experiment. Files of the locked test split (artifacts/data/**/test_*) are excluded BY NAME and
never opened. This experiment's own new files (cla_*, conditional-local-attention-v1) are excluded.

  cla_inventory.py --out artifacts/conditional-local-attention-v1/protected_inventory.json
  cla_inventory.py --verify artifacts/conditional-local-attention-v1/protected_inventory.json
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path("/home/zapan4/moe-research")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
import ms_inventory as msi  # noqa: E402

DIRS = ["runs", "artifacts", "configs", "docs", "src", "tests", "scripts", ".tmp/sequence_routing_v1",
        ".tmp/rep_v1_main", ".tmp/spec_audit_v1", ".tmp/speculative_pool_v1"]
OWN_PREFIXES = ("artifacts/conditional-local-attention-v1", "configs/conditional-local-attention-v1", "cla_",
                "src/csr/cla_", "tests/test_conditional_local_attention_v1", "docs/CONDITIONAL_LOCAL_ATTENTION_V1",
                "CONDITIONAL_LOCAL_ATTENTION_V1", ".tmp/conditional_local_attention_v1")
VOLATILE_SUFFIX = ("heartbeat.json", ".lock")


def own(rel):
    return rel.startswith(OWN_PREFIXES) or "_clav1_" in rel or "__pycache__" in rel


def collect():
    files = msi.collect()
    for d in DIRS:
        if (ROOT / d).exists():
            files += msi.walk(d)
    files += [p.name for p in ROOT.iterdir() if p.is_file() and p.suffix in (".py", ".md", ".json", ".txt", ".sh")]
    seen, out = set(), []
    for f in files:
        if f in seen or msi.is_test_split_name(f) or own(f) or f.endswith(VOLATILE_SUFFIX):
            continue
        seen.add(f)
        out.append(f)
    return sorted(out)


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
        doc = {"experiment": "csr-conditional-local-attention-v1", "created": guard.now(), "n_files": len(rec),
               "excluded": "locked test-split files under artifacts/data/ (excluded by name; never opened); this "
                           "experiment's own new files; __pycache__; heartbeat/lock files",
               "files": rec, "environment": msi.env_state()}
        guard.atomic_write_json(out, doc, mode=0o400)
        guard.ledger("CLA_V1_INVENTORY", f"{a.out} n_files={len(rec)} sha256={guard.sha256_file(out)}")
        print(json.dumps({"n_files": len(rec), "bytes": sum(r["bytes"] for r in rec.values())}, indent=1))
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
        doc = {"verified": guard.now(), "n_files": len(inv), "n_bad": len(bad), "bad": bad, "all_match": not bad}
        guard.atomic_write_json(out, doc, mode=0o400)
        guard.ledger("CLA_V1_VERIFY", f"{out.relative_to(ROOT)} n={len(inv)} bad={len(bad)}")
        print(json.dumps(doc)[:3000])
        sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
