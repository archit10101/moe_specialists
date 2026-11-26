#!/usr/bin/env python
"""csr-gradient-consistent-latent-edl-v1 run-config builder. MODEL/TRAIN are imported unchanged from fle_configs (the
matched recipe). Configs are written once (0400) to configs/gc-edl-v1/<run_name>.json; an existing config must be
identical."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402

SEARCH_MS = {"s5": 80, "s20": 318, "s50": 793}  # identical to the fle seed-100 TOKEN-4 reference


def make(run_name, seed, cond, kind, diag_every=0):
    """kind: search (milestones s5/s20/s50) | pilot (none) | final (p040/p060/p080/p100)."""
    from fle_configs import MODEL, TRAIN
    from csr.gc_paths import CFG_DIR, FINAL_BUDGET, prereg_sha
    n = 3880
    if kind == "search":
        ms = dict(SEARCH_MS)
    elif kind == "pilot":
        ms = {}
    else:
        ms = {f"p{int(f * 100):03d}": max(1, int(round(f * n))) for f in (0.4, 0.6, 0.8, 1.0)}
    cfg = {"experiment": "csr-gradient-consistent-latent-edl-v1", "run_name": run_name, "seed": seed,
           "budget_tokens": FINAL_BUDGET, "prereg_sha256": prereg_sha(), "model": MODEL, "train": dict(TRAIN),
           "cond": cond, "kind": kind, "milestones": ms, "diag_every": diag_every,
           "category": {"search": "search", "pilot": "calibration", "final": "final"}[kind]}
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    p = CFG_DIR / f"{run_name}.json"
    if p.exists():
        if json.loads(p.read_text()) != cfg:
            raise guard.GuardError(f"config {p} exists and differs")
    else:
        guard.atomic_write_json(p, cfg, mode=0o400)
    return p, cfg


def gc_cond(table_rel, beta, warmup_frac=None):
    c = {"table": table_rel, "table_sha256": guard.sha256_file(guard.ROOT / table_rel), "beta": beta}
    if warmup_frac:
        c["warmup_frac"] = warmup_frac
    return {"paper_latent": c}


def matched_fields(a, b):
    """Scientifically relevant config fields that must match for reuse."""
    keys = ("seed", "budget_tokens", "model", "train", "milestones", "diag_every")
    return {k: a.get(k) == b.get(k) for k in keys}
