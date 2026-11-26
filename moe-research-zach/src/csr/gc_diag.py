"""GC-EDL-v1 diagnostic ledger (diagnostic addendum of 2026-11-15; logging and post-hoc analysis only).

artifacts/gc-edl-v1/diagnostic_ledger.jsonl is append-only: one JSON object per line, written with a single
O_APPEND write + fsync under an exclusive lock, so an interruption never leaves a partial line.
Fields: time, stage, candidate, seed, step, tokens, config_sha256, data_manifest_sha256, name, value, units,
n, split, kind (preregistered | descriptive | post-hoc), artifact."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os

from . import guard
from .gc_paths import ART

LEDGER = ART / "diagnostic_ledger.jsonl"
KINDS = ("preregistered", "descriptive", "post-hoc")
_DM = None


def data_manifest_sha():
    """sha256 over the fspv2 split manifest hash and the GC discovery export hashes (stable identifier)."""
    global _DM
    if _DM is None:
        man = json.loads((guard.ROOT / "artifacts/forecast-specialist-pretraining-v2/data/manifest.json").read_text())
        parts = [man.get("splits_sha256", "")]
        for s in ("main", "calib_search", "calib_sel"):
            p = ART / "discovery" / f"inputs_{s}.npz"
            parts.append(guard.sha256_file(p) if p.exists() else "")
        _DM = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return _DM


def cfg_sha(obj):
    if obj is None:
        return None
    if isinstance(obj, str):
        return obj
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def _num(v):
    try:
        import numpy as np
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, np.ndarray):
            return v.tolist()
    except ImportError:
        pass
    return v


def log(stage, name, value, units, n, split, kind, artifact, candidate=None, seed=None, step=None, tokens=None,
        config=None):
    assert kind in KINDS, kind
    rec = {"time": guard.now(), "stage": stage, "candidate": candidate, "seed": seed, "step": step,
           "tokens": tokens, "config_sha256": cfg_sha(config), "data_manifest_sha256": data_manifest_sha(),
           "name": name, "value": _num(value), "units": units, "n": _num(n), "split": split, "kind": kind,
           "artifact": str(artifact) if artifact is not None else None}
    line = (json.dumps(rec, default=str) + "\n").encode()
    fd = os.open(LEDGER, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        os.write(fd, line)
        os.fsync(fd)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    return rec


def log_many(rows, **common):
    for r in rows:
        log(**{**common, **r})
