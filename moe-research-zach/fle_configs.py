#!/usr/bin/env python
"""csr-forecast-latent-edl-v1: run-config builder (prereg sections 2, 7, 8). Configs are written once (0400) to
configs/forecast-latent-edl-v1/<run_name>.json; an existing config must be identical."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402

MODEL = {"d_model": 384, "dense_ffn": 1792, "expert_ffn": 256, "init_std": 0.02, "n_experts": 32, "n_head": 6,
         "n_layer": 12, "norm_eps": 1e-06, "rope_base": 10000.0, "router": "linear", "shared_ffn": 768,
         "teacher": False, "top_k": 4, "vocab_size": 50304}
TRAIN = {"aux_coeff": 0.01, "betas": [0.9, 0.95], "clip": 1.0, "decay_frac": 0.2, "log_every": 50, "lr": 0.0006,
         "micro_batch_rows": 32, "min_lr": 6e-05, "rows_per_step": 64, "warmup_frac": 0.01, "weight_decay": 0.1,
         "z_coeff": 0.001}


def stage_steps():
    from csr.fle_paths import FINAL_BUDGET
    from csr.lfr_lib import main_stream
    sys.path.insert(0, str(guard.ROOT))
    from fle_cache import stage_steps as ss
    sp, steps, _ = main_stream(FINAL_BUDGET)
    return ss(sp, steps), len(steps)


def make(run_name, seed, cond, kind, diag_every=0, train_override=None):
    """kind: search (milestones s5/s20/s50) | final (40/60/80/100 %)."""
    from csr.fle_paths import CFG_DIR, FINAL_BUDGET, prereg_sha
    st, n = stage_steps()
    if kind == "search":
        ms = dict(st)
    else:
        ms = {f"p{int(f * 100):03d}": max(1, int(round(f * n))) for f in (0.4, 0.6, 0.8, 1.0)}
    cfg = {"experiment": "csr-forecast-latent-edl-v1", "run_name": run_name, "seed": seed,
           "budget_tokens": FINAL_BUDGET, "prereg_sha256": prereg_sha(), "model": MODEL,
           "train": {**TRAIN, **(train_override or {})},
           "cond": cond, "kind": kind, "milestones": ms, "diag_every": diag_every,
           "category": "search" if kind == "search" else "final"}
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    p = CFG_DIR / f"{run_name}.json"
    if p.exists():
        if json.loads(p.read_text()) != cfg:
            raise guard.GuardError(f"config {p} exists and differs")
    else:
        guard.atomic_write_json(p, cfg, mode=0o400)
    return p, cfg


def latent_cond(cfg_id, beta, layers, shuffle=False, cache="search_main"):
    from csr.fle_paths import LATENT
    tab = LATENT / cfg_id / f"q_{cache}.npz"
    return {"latent": {"cfg_id": cfg_id, "table": str(tab.relative_to(guard.ROOT)), "table_sha256":
                       guard.sha256_file(tab), "beta": beta, "layers": layers, "shuffle": shuffle}}


if __name__ == "__main__":
    guard.check_root()
    print(stage_steps())
