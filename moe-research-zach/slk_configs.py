#!/usr/bin/env python
"""csr-semantic-edl-long-adaptive-k-v1 run-config builder. MODEL/TRAIN imported unchanged from fle_configs (the exact
recipe of the reused fle TOKEN-4 / explicit-EDL finals). Every run uses the FINAL plan (3,880 steps, 245,032,457
tokens, WSD over 3,880 steps); screening runs stop at a milestone with --max-steps (same schedule, same data prefix).
Configs are written once (0400) to configs/semantic-edl-long-k/<run_name>.json; an existing config must be identical."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402


def make(run_name, seed, cond, category, diag_every=100):
    from fle_configs import MODEL, TRAIN
    from csr.slk_paths import CFG_DIR, EXP, FINAL_BUDGET, MILESTONES, prereg_sha
    cfg = {"experiment": EXP, "run_name": run_name, "seed": seed, "budget_tokens": FINAL_BUDGET,
           "prereg_sha256": prereg_sha(), "model": MODEL, "train": dict(TRAIN), "cond": cond, "kind": "final",
           "milestones": dict(MILESTONES), "diag_every": diag_every, "category": category}
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    p = CFG_DIR / f"{run_name}.json"
    if p.exists():
        if json.loads(p.read_text()) != cfg:
            raise guard.GuardError(f"config {p} exists and differs")
    else:
        guard.atomic_write_json(p, cfg, mode=0o400)
    return p, cfg


def paper_cond(table_rel, table_sha, beta, layers=None):
    if guard.sha256_file(guard.ROOT / table_rel) != table_sha:
        raise guard.GuardError(f"table hash mismatch {table_rel}")
    c = {"table": table_rel, "table_sha256": table_sha, "beta": beta}
    if layers is not None:
        c["layers"] = list(layers)
    return {"paper_latent": c}


def phase1_conds():
    from csr.slk_paths import C6_BETA, C6_SHUF_TABLE, C6_SHUF_TABLE_SHA, C6_TABLE, C6_TABLE_SHA, EDL_BETA
    return {"T4": {}, "EDL": {"edl_beta": EDL_BETA},
            "C6": paper_cond(C6_TABLE, C6_TABLE_SHA, C6_BETA),
            "SHUF": paper_cond(C6_SHUF_TABLE, C6_SHUF_TABLE_SHA, C6_BETA)}
