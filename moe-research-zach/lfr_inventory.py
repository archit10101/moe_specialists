#!/usr/bin/env python
"""csr-latent-forecast-router-distill-v1 (lfr): protected-artifact inventory (read-only) and verification.
Protects everything fsp_inventory.collect() protects PLUS every csr-forecast-specialist-pretraining-v2 (fspv2) file:
its artifacts, configs, code, docs, runs and its .tmp/fspv2 resumable checkpoints (the stopped experiment must stay
resumable). Excludes locked test-split files by name (never opened), shared ledgers, and this experiment's own files.
  lfr_inventory.py --out <json> | --verify <json>"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path("/home/zapan4/moe-research")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
import ms_inventory as msi  # noqa: E402
import fsp_inventory as fsi  # noqa: E402

OWN = ("artifacts/latent-forecast-router-distill-v1", "configs/latent-forecast-router-distill-v1",
       ".tmp/latent-forecast-router-distill-v1", "lfr_", "src/csr/lfr_", "docs/LATENT_FORECAST_ROUTER_DISTILL_V1",
       "LATENT_FORECAST_ROUTER_DISTILL_V1")
FSP_DIRS = ["artifacts/forecast-specialist-pretraining-v2", "configs/forecast-specialist-pretraining-v2",
            ".tmp/fspv2"]


def collect():
    files = set(fsi.collect())
    for d in FSP_DIRS + ["runs", "src", "docs", "configs", "scripts", "artifacts/encoder"]:
        if (ROOT / d).exists():
            files.update(msi.walk(d))
    files.update(p.name for p in ROOT.iterdir() if p.is_file())
    return sorted(f for f in files if not msi.is_test_split_name(f) and not f.startswith(OWN)
                  and "latent_forecast_router_distill_v1" not in f and "__pycache__" not in f
                  and "/locked/" not in f and not f.rsplit("/", 1)[-1].startswith("test_")
                  and not f.startswith("ledger/") and not f.endswith((".lock",)))


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
        guard.ledger("LFR_INVENTORY", f"{a.out} n_files={len(rec)} sha256={guard.sha256_file(out)}")
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
        guard.ledger("LFR_VERIFY", f"{out.relative_to(ROOT)} n={len(inv)} bad={len(bad)}")
        print(json.dumps(doc)[:3000])
        sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
