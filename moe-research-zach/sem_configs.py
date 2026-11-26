#!/usr/bin/env python
"""csr-semantic-edl-v1 run-config builder. MODEL/TRAIN imported unchanged from fle_configs (matched recipe). Configs are
written once (0400) to configs/semantic-edl-v1/<run_name>.json; an existing config must be identical."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402

SEARCH_MS = {"s5": 80, "s20": 318, "s50": 793}  # identical to the fle seed-100 TOKEN-4 reference


def make(run_name, seed, cond, kind, diag_every=0):
    """kind: search (s5/s20/s50) | pilot (none) | final (p040/p060/p080/p100)."""
    from fle_configs import MODEL, TRAIN
    from csr.sem_paths import CFG_DIR, FINAL_BUDGET, prereg_sha
    n = 3880
    ms = {"search": dict(SEARCH_MS), "pilot": {}}.get(kind)
    if ms is None:
        ms = {f"p{int(f * 100):03d}": max(1, int(round(f * n))) for f in (0.4, 0.6, 0.8, 1.0)}
    cfg = {"experiment": "csr-semantic-edl-v1", "run_name": run_name, "seed": seed, "budget_tokens": FINAL_BUDGET,
           "prereg_sha256": prereg_sha(), "model": MODEL, "train": dict(TRAIN), "cond": cond, "kind": kind,
           "milestones": ms, "diag_every": diag_every,
           "category": {"search": "search", "pilot": "calibration", "final": "final"}[kind]}
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    p = CFG_DIR / f"{run_name}.json"
    if p.exists():
        if json.loads(p.read_text()) != cfg:
            raise guard.GuardError(f"config {p} exists and differs")
    else:
        guard.atomic_write_json(p, cfg, mode=0o400)
    return p, cfg


def sem_cond(table_rel, beta, layers=None, warmup_frac=None):
    c = {"table": table_rel, "table_sha256": guard.sha256_file(guard.ROOT / table_rel), "beta": beta}
    if layers is not None:
        c["layers"] = list(layers)
    if warmup_frac:
        c["warmup_frac"] = warmup_frac
    return {"paper_latent": c}
