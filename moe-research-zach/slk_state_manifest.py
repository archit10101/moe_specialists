#!/usr/bin/env python
"""csr-semantic-edl-long-adaptive-k-v1: manifest of the preserved resumable state (CPU, read-only on the state).
  slk_state_manifest.py --out <json>      hash every Phase-1 checkpoint (state.pt, meta.json), run status/config,
                                           evaluation output, config, driver state, ledger and log of this experiment
  slk_state_manifest.py --verify <json>   re-hash and report MISSING / MISMATCH (append-only files may only grow:
                                           checked as prefix-hash of their recorded length)"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402
from csr.slk_paths import ART  # noqa: E402

ROOT = guard.ROOT
APPEND_ONLY = {"RUN_LOG.jsonl", "DECISION_LOG.md", "DEVIATIONS.jsonl", "RESULTS_LIVE.md", "compute_ledger.tsv",
               "checkpoint_ledger.tsv", "run_ledger.tsv", "driver_log.txt", "deleted_artifacts.tsv"}


def prefix_sha(p, n):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        left = n
        while left > 0:
            b = f.read(min(1 << 22, left))
            if not b:
                break
            h.update(b)
            left -= len(b)
    return h.hexdigest()


def collect():
    import slk_run
    files = []
    for rn, rd in sorted(slk_run.run_dirs().items()):
        if not rn.startswith("P1_"):
            continue
        d = ROOT / rd
        for p in sorted(d.glob("ckpt_*/*")) + [d / "status.json", d / "config.json", d / "summary.json"]:
            if p.is_file():
                files.append(p)
    files += sorted(p for p in (ART / "evals").rglob("*.json"))
    files += sorted((ROOT / "configs/semantic-edl-long-k").glob("*.json"))
    files += sorted(p for p in ART.glob("*") if p.is_file() and not p.name.startswith("state_manifest_")
                    and p.name != "RESUME.md")  # RESUME.md records this manifest's hash
    files += sorted(p for p in (ART / "latent").glob("*") if p.is_file())
    files += sorted(p for p in (ART / "phase2").glob("*") if p.is_file()) if (ART / "phase2").exists() else []
    files += [ROOT / "docs/SEMANTIC_EDL_LONG_ADAPTIVE_K_PREREG.md"] + sorted(ROOT.glob("slk_*.py")) + \
        sorted((ROOT / "src/csr").glob("slk_*.py"))
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--verify")
    a = ap.parse_args()
    guard.check_root()
    if a.out:
        out = guard.inside_root(a.out)
        rec = {}
        for p in collect():
            rel = str(p.relative_to(ROOT))
            rec[rel] = {"sha256": guard.sha256_file(p), "bytes": p.stat().st_size, "append_only": p.name in APPEND_ONLY}
        ck = {}
        led = ART / "checkpoint_ledger.tsv"
        for line in led.read_text().splitlines()[1:]:
            f = line.split("\t")
            if len(f) > 7:
                ck[f[4]] = f[7]
        cross = {"n_ckpt_state_files": 0, "n_in_ledger": 0, "n_ledger_mismatch": 0, "mismatch": []}
        for rel, r in rec.items():
            if rel.endswith("/state.pt") and "/ckpt_" in rel:
                cross["n_ckpt_state_files"] += 1
                key = rel.rsplit("/", 1)[0]
                if key in ck:
                    cross["n_in_ledger"] += 1
                    if ck[key] != r["sha256"]:
                        cross["n_ledger_mismatch"] += 1
                        cross["mismatch"].append(rel)
        doc = {"created": guard.now(), "n_files": len(rec), "bytes": sum(r["bytes"] for r in rec.values()),
               "checkpoint_ledger_crosscheck": cross, "files": rec}
        guard.atomic_write_json(out, doc, mode=0o400)
        print(json.dumps({k: v for k, v in doc.items() if k != "files"}))
    else:
        src = guard.inside_root(a.verify)
        inv = json.loads(src.read_text())["files"]
        bad = []
        for rel, r in inv.items():
            p = ROOT / rel
            if not p.exists():
                bad.append((rel, "MISSING"))
            elif r["append_only"]:
                if p.stat().st_size < r["bytes"] or prefix_sha(p, r["bytes"]) != r["sha256"]:
                    bad.append((rel, "PREFIX_MISMATCH"))
            elif guard.sha256_file(p) != r["sha256"]:
                bad.append((rel, "MISMATCH"))
        print(json.dumps({"verified": guard.now(), "n_files": len(inv), "n_bad": len(bad), "bad": bad[:100]}))
        sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
