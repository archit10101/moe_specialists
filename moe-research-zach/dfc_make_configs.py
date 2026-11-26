#!/usr/bin/env python
"""csr-domain-forecast-curriculum-v1: write the resolved configs (prereg sections 3, 5, 6). Never overwrites.
  dfc_make_configs.py base            -> draft.json + main/<COND>_s<seed>.json (conditions 1-6)
  dfc_make_configs.py compute <N>     -> main/TOKEN-COMPUTE_s<seed>.json (N from compute_match.json only)"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402
from csr.dfc_lib import CFG_DIR, EXP_ID, MAIN_TOKENS, manifest, prereg_sha  # noqa: E402
from csr.dfc_model import MAIN_MODEL_CFG, draft_config_dict  # noqa: E402

SEEDS = (0, 1, 2)
MAIN_CONDS = ("TOKEN-MAIN", "LABEL", "FORECAST", "SOFT", "WRONG")
LONG = {"TOKEN-MAIN": "TOKEN-MAIN", "LABEL": "LABEL->TOKEN-S40", "FORECAST": "FORECAST->TOKEN-S40",
        "SOFT": "SOFT-SPECIALIST->TOKEN-S40", "WRONG": "WRONG-FORECAST->TOKEN-S40", "TOKEN-ALL-DATA": "TOKEN-ALL-DATA",
        "TOKEN-COMPUTE": "TOKEN-COMPUTE"}


def train_main(budget):
    return {"micro_batch_rows": 16, "accum": 2, "rows_per_step": 32, "lr": 6e-4, "min_lr": 6e-5, "warmup_steps": 60,
            "decay_frac": 0.2, "betas": [0.9, 0.95], "weight_decay": 0.1, "clip": 1.0, "aux_coeff": 0.01,
            "z_coeff": 0.001, "precision": "bf16-autocast", "router_precision": "fp32", "log_every": 25,
            "state_every_seconds": 600, "throughput_warmup_steps": 10, "token_budget": budget}


def base_fields():
    man = manifest()
    return {"experiment": EXP_ID, "prereg_sha256": prereg_sha(),
            "data": {"dir": "artifacts/domain-forecast-curriculum-v1/data", "splits_sha256": man["splits_sha256"],
                     "domains": man["domains"], "seq_len": 1024}}


def write(path, cfg):
    path.parent.mkdir(parents=True, exist_ok=True)
    guard.atomic_write_json(path, cfg, mode=0o400)
    print("wrote", path.relative_to(guard.ROOT))


def main_cfg(cond, seed, budget, ckpts):
    return {**base_fields(), "kind": "main", "condition": cond, "condition_long": LONG[cond], "seed": seed,
            "run_name": f"{cond}_s{seed}", "model": MAIN_MODEL_CFG, "checkpoints": ckpts,
            "init": {"backbone": "global RNG seed", "added": "dedicated generator sha256('dfcv1-added|seed')"},
            "order_seed": 1000 + seed, "train": train_main(budget)}


def main():
    guard.check_root()
    cmd = sys.argv[1]
    if cmd == "base":
        d = {**base_fields(), "kind": "draft", "condition": "DRAFT", "seed": 0, "run_name": "draft_d128_l4_adapt",
             "model": draft_config_dict(), "adapters": {"per_domain_per_layer": True, "bottleneck": 32},
             "checkpoints": "final",
             "train": {**train_main(30_000_000), "lr": 1e-3, "min_lr": 1e-4, "order_seed": 7}}
        write(CFG_DIR / "draft.json", d)
        for seed in SEEDS:
            for c in MAIN_CONDS:
                write(CFG_DIR / "main" / f"{c}_s{seed}.json", main_cfg(c, seed, MAIN_TOKENS, "fractions"))
            write(CFG_DIR / "main" / f"TOKEN-ALL-DATA_s{seed}.json",
                  main_cfg("TOKEN-ALL-DATA", seed, MAIN_TOKENS + 30_000_000, "final"))
    elif cmd == "compute":
        n = int(sys.argv[2])
        cm = json.loads((guard.ROOT / "artifacts/domain-forecast-curriculum-v1/compute_match.json").read_text())
        if cm["n_tokens"] != n:
            raise guard.GuardError("N must equal compute_match.json n_tokens")
        for seed in SEEDS:
            c = main_cfg("TOKEN-COMPUTE", seed, n, "final")
            c["compute_match_sha256"] = guard.sha256_file(
                guard.ROOT / "artifacts/domain-forecast-curriculum-v1/compute_match.json")
            write(CFG_DIR / "main" / f"TOKEN-COMPUTE_s{seed}.json", c)
    else:
        raise SystemExit(cmd)


if __name__ == "__main__":
    main()
