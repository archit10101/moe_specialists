#!/usr/bin/env python
"""Write configs for csr-value-routed-compute-v1 (diagnostic baseline) and csr-m2d2-domain-prior-v1 (B0-B3).
  vm_make_configs.py vrc --ctx 8192
  vm_make_configs.py m2d2
Refuses to overwrite existing configs."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402
from csr.vm_run import EXP  # noqa: E402

MODEL = {"d_model": 192, "dense_ffn": 768, "expert_ffn": 128, "init_std": 0.02, "max_seq_len": 1024,
         "moe_layers": [1, 2, 3, 4, 5, 6, 7], "n_experts": 32, "n_head": 8, "n_layer": 8, "norm_eps": 1e-06,
         "rope_base": 10000.0, "shared_ffn": 256, "tie_embeddings": True, "top_k": 4, "vocab_size": 50304}
TRAIN = {"accum": 2, "aux_coeff": 0.01, "betas": [0.9, 0.95], "clip": 1.0, "decay_frac": 0.2,
         "eval_every_tokens": 5_000_000, "log_every": 25, "lr": 0.0006, "max_seconds": 5400, "micro_batch_rows": 16,
         "min_lr": 6e-05, "precision": "bf16-autocast", "save_checkpoint": True, "state_every_seconds": 300,
         "throughput_warmup_steps": 10, "warmup_steps": 60, "weight_decay": 0.1, "z_coeff": 0.001}
M2D2_KINDS = {"B0": ("token", False), "B1": ("domain_emb", False), "B2": ("prior", False), "B3": ("prior", True)}
FINAL_BUDGET = 43_000_000


def write(path, cfg):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) == cfg:
            return path
        raise guard.GuardError(f"{path} exists with different content")
    guard.atomic_write_json(path, cfg, mode=0o400)
    return path


def main():
    which = sys.argv[1]
    if which == "vrc":
        C = int(sys.argv[sys.argv.index("--ctx") + 1])
        d = f"artifacts/value-routed-compute-v1/data/fwe-rep-v1-L{C}"
        man = json.loads((guard.ROOT / d / "manifest.json").read_text())
        mb = {8192: 2, 4096: 4, 2048: 8}[C]
        cfg = {"experiment": EXP["vrc"]["id"], "protocol_sha256": guard.sha256_file(EXP["vrc"]["protocol"]),
               "run_name": f"diag_FULL-TOKEN4_L{C}_s0", "stage": "diagnostic", "condition": "FULL-TOKEN4",
               "model": {**MODEL, "max_seq_len": C},
               "data": {"dir": d, "seq_len": C, "splits_sha256": man["splits_sha256"],
                        "packing": "greedy sequential in stored train order, block-diagonal document-causal"},
               "init": {"seed": 0, "type": "random"},
               "train": {**TRAIN, "micro_batch_rows": mb, "token_budget": 5_000_000, "seed": 0},
               "eval": {"final_calib_search": False, "max_rows": 16, "rows_per_batch": mb}}
        print(write(EXP["vrc"]["cfg"] / "diag" / f"FULL-TOKEN4_L{C}_s0.json", cfg))
        return
    d = "artifacts/m2d2-domain-prior-v1/data"
    man = json.loads((guard.ROOT / d / "manifest.json").read_text())
    out = []
    for stage, seeds, budget in (("search", [100], 10_000_000), ("final", [0, 1], FINAL_BUDGET)):
        for s in seeds:
            for c, (kind, shuf) in M2D2_KINDS.items():
                cfg = {"experiment": EXP["m2d2"]["id"],
                       "protocol_sha256": guard.sha256_file(EXP["m2d2"]["protocol"]),
                       "run_name": f"{stage}_s{s}_{c}", "stage": stage, "condition": c, "model": MODEL,
                       "domain": {"kind": kind, "shuffle_train_labels": shuf, "shuffle_seed": 4242,
                                  "prior_layers": [4, 5, 6, 7], "alpha_init": 0.5, "bias_init": "zeros",
                                  "dom_emb_init": "N(0,0.02) dedicated generator"},
                       "data": {"dir": d, "seq_len": 1024, "splits_sha256": man["splits_sha256"],
                                "train_order_seed": 17,
                                "packing": "single-domain rows; per-domain seeded shuffle; seeded row interleave"},
                       "init": {"seed": s, "type": "random"},
                       "train": {**TRAIN, "token_budget": budget, "seed": s},
                       "eval": {"final_calib_search": True, "max_rows": 64, "rows_per_batch": 16}}
                out.append(str(write(EXP["m2d2"]["cfg"] / stage / f"s{s}_{c}.json", cfg)))
    print("\n".join(out))


if __name__ == "__main__":
    main()
