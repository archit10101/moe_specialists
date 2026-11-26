#!/usr/bin/env python
"""csr-semantic-utility-edl-v1 run-config builder. MODEL/TRAIN imported unchanged from fle_configs (the recipe of every
reused run). Every run uses the FINAL plan (3,880 steps, 245,032,457 tokens, WSD over 3,880 steps); search runs stop at a
milestone with --max-steps. Configs are written once (0400) to configs/semantic-utility-edl-v1/<run_name>.json."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402

FIXED = {"probe_rows": 16, "probe_passes": 4, "refresh_every": 10, "m_min": 2.0, "n_min": 8, "clip": 3.0,
         "min_cov": 0.5, "min_len": 256, "ramp_steps": 50}


def make(run_name, seed, cond, category, budget=None, milestones=None, diag_every=100):
    from fle_configs import MODEL, TRAIN
    from csr.sue_paths import CFG_DIR, EXP, FINAL_BUDGET, MILESTONES, prereg_sha
    cfg = {"experiment": EXP, "run_name": run_name, "seed": seed, "budget_tokens": budget or FINAL_BUDGET,
           "prereg_sha256": prereg_sha(), "model": MODEL, "train": dict(TRAIN), "cond": cond, "kind": "final",
           "milestones": dict(milestones or MILESTONES), "diag_every": diag_every, "category": category}
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    p = CFG_DIR / f"{run_name}.json"
    if p.exists():
        if json.loads(p.read_text()) != cfg:
            raise guard.GuardError(f"config {p} exists and differs")
    else:
        guard.atomic_write_json(p, cfg, mode=0o400)
    return p, cfg


def util_cond(table, table_sha, beta, gamma, tau, rho, onset, layers=None, **over):
    if guard.sha256_file(guard.ROOT / table) != table_sha:
        raise guard.GuardError(f"table hash mismatch {table}")
    u = {**FIXED, "gamma": gamma, "tau": tau, "rho": rho, "onset_step": onset}
    if layers is not None:
        u["layers"] = list(layers)
    u.update(over)
    return {"paper_latent": {"table": table, "table_sha256": table_sha, "beta": beta}, "utility": u}
