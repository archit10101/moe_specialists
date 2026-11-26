#!/usr/bin/env python
"""csr-multiscale-routing-v1: protected-file inventory (read-only) and environment snapshot.

Hashes every protected pre-existing file needed by or produced before this experiment. Files whose name contains
"test_" under artifacts/data/ are NEVER opened, hashed or stat'ed for content: the locked test split is excluded by
name before any filesystem access to the file itself (directory listings are filtered on names only).

Usage:  ms_inventory.py --out <new json path>     (initial inventory)
        ms_inventory.py --verify <inventory json> (post-run verification; writes <inventory>.verify_<stamp>.json)
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/zapan4/moe-research")
sys.path.insert(0, str(ROOT / "src"))

PROTECTED_DIRS = [
    "artifacts/data/fwe-rep-v1", "artifacts/segments/8e7edc7e14b3c880", "artifacts/segments_random/0c57f77e51ea3734",
    "configs/rep-v1", "artifacts/rep-v1", "artifacts/specialization-audit-v1", "artifacts/encoder",
    "artifacts/tokenizer", "docs",
]
PROTECTED_FILES = [
    "EXPERIMENT_MANIFEST.json", "EXPERIMENT_REPORT.md", "SPECIALIZATION_AUDIT_REPORT.md", "SAFETY.md", "env.sh",
    ".tmp/rep_v1_main/runs.tsv", ".tmp/spec_audit_v1/runs.tsv",
]
# All pre-existing source files; this experiment adds new files only and must leave these byte-identical.
SOURCE_GLOBS = ["*.py", "src/csr/*.py", "scripts/*.sh", "tests/*.py"]


def is_test_split_name(rel: str) -> bool:
    name = rel.rsplit("/", 1)[-1]
    return rel.startswith("artifacts/data/") and (name.startswith("test_") or "_test_" in name or name == "test")


def sha256(p: Path):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def walk(rel_dir):
    out = []
    base = ROOT / rel_dir
    for dp, dns, fns in os.walk(base):
        dns.sort()
        for fn in sorted(fns):
            rel = str((Path(dp) / fn).relative_to(ROOT))
            if is_test_split_name(rel):
                continue  # never touched
            out.append(rel)
    return out


def collect(existing_only_before=None):
    files = []
    for d in PROTECTED_DIRS:
        files += walk(d)
    files += [f for f in PROTECTED_FILES if (ROOT / f).exists()]
    inv = json.loads((ROOT / "artifacts/specialization-audit-v1/inventory.json").read_text())["checkpoints"]
    for k in sorted(inv):
        files.append(inv[k]["checkpoint"])
        for extra in ("config.json", "summary.json", "status.json"):
            p = Path(inv[k]["run"]) / extra
            if (ROOT / p).exists():
                files.append(str(p))
    for line in (ROOT / ".tmp/rep_v1_main/runs.tsv").read_text().splitlines():
        parts = line.split("\t")
        if len(parts) >= 4:
            files.append(parts[3])  # eval json
    for g in SOURCE_GLOBS:
        files += sorted(str(p.relative_to(ROOT)) for p in ROOT.glob(g))
    seen, out = set(), []
    for f in files:
        if f not in seen and not is_test_split_name(f):
            seen.add(f)
            out.append(f)
    return out


def env_state():
    import importlib
    vers = {}
    for m in ("torch", "numpy", "transformers", "tokenizers", "pyarrow", "huggingface_hub"):
        try:
            vers[m] = importlib.import_module(m).__version__
        except Exception as e:  # noqa: BLE001
            vers[m] = f"unavailable: {type(e).__name__}"
    df = os.statvfs("/")
    gpu = subprocess.run(["nvidia-smi", "-i", "0", "--query-gpu=name,memory.used,memory.total,driver_version",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    nproc = subprocess.run(["nvidia-smi", "-i", "0", "-q", "-d", "PIDS"], capture_output=True, text=True).stdout
    return {"python": sys.version, "platform": platform.platform(), "packages": vers,
            "disk_free_bytes_root": df.f_bavail * df.f_frsize, "gpu0": gpu,
            "gpu0_process_count": nproc.count("Process ID"), "nproc": os.cpu_count(),
            "cuda_visible_devices_env": os.environ.get("CUDA_VISIBLE_DEVICES")}


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
        rec = {f: {"sha256": sha256(ROOT / f), "bytes": (ROOT / f).stat().st_size} for f in files}
        doc = {"experiment": "csr-multiscale-routing-v1", "created": guard.now(), "n_files": len(rec),
               "excluded": "locked test-split files under artifacts/data/ (excluded by name; never opened)",
               "files": rec, "environment": env_state()}
        guard.atomic_write_json(out, doc, mode=0o400)
        guard.ledger("MS_V1_INVENTORY", f"{a.out} n_files={len(rec)} sha256={guard.sha256_file(out)}")
        print(json.dumps({"n_files": len(rec), "environment": doc["environment"]}, indent=1))
    elif a.verify:
        src = guard.inside_root(a.verify)
        inv = json.loads(src.read_text())["files"]
        bad = []
        for f, r in inv.items():
            p = ROOT / f
            if not p.exists():
                bad.append((f, "MISSING"))
            elif sha256(p) != r["sha256"]:
                bad.append((f, "MISMATCH"))
        out = src.with_name(src.stem + f".verify_{guard.utc_stamp()}.json")
        doc = {"verified": guard.now(), "n_files": len(inv), "n_bad": len(bad), "bad": bad, "all_match": not bad,
               "environment": env_state()}
        guard.atomic_write_json(out, doc, mode=0o400)
        guard.ledger("MS_V1_VERIFY", f"{out.relative_to(ROOT)} n={len(inv)} bad={len(bad)}")
        print(json.dumps({k: v for k, v in doc.items() if k != "environment"}))
        sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
