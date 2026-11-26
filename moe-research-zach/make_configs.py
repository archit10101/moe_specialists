#!/usr/bin/env python
"""Generate configs/base.json, configs/conditions/{A..F}.json and EXPERIMENT_MANIFEST.json from the hashed data
and segmentation manifests. Refuses to overwrite existing files (bump --tag instead). CPU only, no model."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402
from csr.config import CONDITIONS, CONTRASTS, build_condition_config, manifest  # noqa: E402
from csr.model import ModelConfig  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="artifacts/data/fwe-v1")
    ap.add_argument("--segments", required=True)
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--random-cache", required=True, help="accepted F-v2 random cache directory")
    a = ap.parse_args()
    guard.check_root()
    dd, sd = guard.inside_root(a.data), guard.inside_root(a.segments)
    dman = json.loads((dd / "manifest.json").read_text())
    sman = json.loads((sd / "manifest.json").read_text())
    rdir = guard.inside_root(a.random_cache)
    rman = json.loads((rdir / "manifest.json").read_text())
    assert rman["accepted"] and rman["spec"]["parent_semantic_cache"] == sman["cache_id"]
    base = {
        "experiment": f"csr-{a.tag}",
        "model": ModelConfig().to_dict(),
        "routing": {"mode": "token", "pool_size": min(ModelConfig().n_experts, 2 * ModelConfig().top_k), "warmup": sman["spec"]["warmup"], "gate_norm": "global"},
        "segmentation": {"strategy": "none", "cache": str(sd.relative_to(guard.ROOT)), "cache_id": sman["cache_id"],
                         "fixed_size": sman["spec"]["fixed_size"], "semantic_tau": sman["tau"],
                         "min_len": sman["spec"]["min_len"], "max_len": sman["spec"]["max_len"],
                         "random_cache": str(rdir.relative_to(guard.ROOT)), "random_cache_id": rman["random_cache_id"],
                         "random_generator": rman["spec"]["version"], "random_seed": rman["spec"]["seed"],
                         "random_acceptance": rman["acceptance_mode"]},
        "semantic": {"enabled": False, "lambda": None, "tau_s": None, "tau_r": None, "n_max": 256, "min_chunks": 16,
                     "comparison_set": "all valid de-duplicated chunks of the micro-batch (complete w-prefix), "
                                       "evenly-spaced subsample to n_max; multiple documents per micro-batch"},
        "init": {"type": "random", "seed": 1234},
        "data": {"dir": str(dd.relative_to(guard.ROOT)), "splits_sha256": guard.sha256_json(dman["splits"]),
                 "seq_len": 1024, "train_order_seed": 17, "packing": "greedy sequential, block-diagonal doc masks"},
        "train": {"micro_batch_rows": 16, "accum": 1, "token_budget": 3_000_000, "lr": 3e-4, "min_lr": 3e-5,
                  "warmup_steps": 10, "decay_frac": 0.1, "weight_decay": 0.1, "betas": [0.9, 0.95], "clip": 1.0,
                  "seed": 0, "precision": "bf16-autocast", "router_precision": "fp32", "compile": False,
                  "activation_checkpointing": False, "dropout": 0.0, "aux_coeff": 0.01, "z_coeff": 0.001,
                  "aux_scope": "per micro-batch, per MoE layer, summed over layers, non-pad tokens",
                  "log_every": 10, "grad_probe_every": 25, "eval_every_tokens": 1_000_000,
                  "throughput_warmup_steps": 10, "save_checkpoint": True, "max_seconds": 5400},
        "eval": {"split": "val", "max_rows": 64, "rows_per_batch": 16},
        "trainable": "all LM parameters (see MODEL_DECISION.md)",
    }
    cdir = guard.ROOT / "configs" / a.tag
    cdir.mkdir(parents=True, exist_ok=False)
    guard.atomic_write_json(cdir / "base.json", base)
    cfgs = {c: build_condition_config(base, c) for c in CONDITIONS}
    for c, cfg in cfgs.items():
        guard.atomic_write_json(cdir / f"condition_{c}.json", cfg)
    man = manifest(cfgs, {"path": str(dd.relative_to(guard.ROOT)), "splits": dman["splits"], "dedup": dman["dedup"],
                          "dataset": dman["dataset"], "tokenizer": dman["tokenizer"]},
                   {"cache_id": sman["cache_id"], "tau": sman["tau"], "spec": sman["spec"],
                    "fixed_and_semantic_chunk_summary": {k: v for k, v in sman["chunk_summary"].items()
                                                         if not k.endswith("/random")},
                    "timings_s": sman["timings_s"],
                    "random_F": {"cache": str(rdir.relative_to(guard.ROOT)), "id": rman["random_cache_id"],
                                 "generator": rman["spec"]["version"], "seed": rman["spec"]["seed"],
                                 "acceptance_mode": rman["acceptance_mode"], "user_exemptions": rman["user_exemptions"],
                                 "train_comparison_vs_D": rman["report"]["train"]["compare"],
                                 "superseded": "F-v1 tables in the semantic cache (double document-end censoring)"}},
                   notes={"semantic_hparams": "lambda/tau_s/tau_r are null until calibrate.py runs on the pilot "
                                              "initialization checkpoint; then frozen for ALL conditions",
                          "pre_registration": "see EXPERIMENT_REPORT.md §Pre-registration",
                          "primary_endpoint": "token-weighted held-out next-token CE, nats per valid token (val; test locked)",
                          "contrasts": CONTRASTS})
    guard.atomic_write_json(guard.ROOT / f"EXPERIMENT_MANIFEST.json", man)
    print("wrote", cdir, "and EXPERIMENT_MANIFEST.json")


if __name__ == "__main__":
    main()
