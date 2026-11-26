#!/usr/bin/env python
"""csr-representative-v1 configurations (docs/REP_V1_PREREG.md). Never overwrites.

  precal : configs/rep-v1/precal/base.json      semantic values null (tau calibration, feasibility benchmarks)
  tuning : configs/rep-v1/tuning/base.json      seed 99, 5M tokens, probe/log every 5, tau_s/tau_r frozen
  final  : configs/rep-v1/seed{0,1,2}/base.json + condition_{A..F}.json + configs/rep-v1/MANIFEST.json
Each stage verifies the parameter-count requirements and the matched-field check across A-F.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402
from csr.config import CONDITIONS, build_condition_config, check_matched, manifest  # noqa: E402

OUT = guard.ROOT / "configs" / "rep-v1"
LABEL = "csr-representative-v1: three-seed, 100M-token, ~29M-parameter prospective follow-up (not a confirmation)"
TOTAL, ACTIVE, EMB = 28_873_920, 14_423_232, 50304 * 192


def model_block():
    return {"d_model": 192, "dense_ffn": 768, "expert_ffn": 128, "init_std": 0.02, "max_seq_len": 1024,
            "moe_layers": [1, 2, 3, 4, 5, 6, 7], "n_experts": 32, "n_head": 8, "n_layer": 8, "norm_eps": 1e-6,
            "rope_base": 10000.0, "shared_ffn": 256, "top_k": 4, "vocab_size": 50304, "tie_embeddings": True}


def base_config(seg_dir, rnd_dir):
    dd = guard.ROOT / "artifacts" / "data" / "fwe-rep-v1"
    dman = json.loads((dd / "manifest.json").read_text())
    sman = json.loads((guard.inside_root(seg_dir) / "manifest.json").read_text())
    rman = json.loads((guard.inside_root(rnd_dir) / "manifest.json").read_text())
    assert rman["accepted"] and rman["spec"]["parent_semantic_cache"] == sman["cache_id"]
    prereg = (guard.ROOT / "docs" / "REP_V1_PREREG.sha256").read_text().split()[0]
    assert guard.sha256_file(guard.ROOT / "docs" / "REP_V1_PREREG.md") == prereg, "preregistration changed"
    return {
        "experiment": "csr-representative-v1", "prereg_sha256": prereg, "result_label": LABEL,
        "data": {"dir": "artifacts/data/fwe-rep-v1", "packing": "greedy sequential, block-diagonal doc masks",
                 "seq_len": 1024, "splits_sha256": guard.sha256_json(dman["splits"]), "train_order_seed": 17},
        "eval": {"max_rows": 128, "rows_per_batch": 16, "split": "val",
                 "monitoring_subset": "first 128 packed validation rows, natural order (monitoring only)"},
        "init": {"seed": 0, "type": "random"},
        "model": model_block(),
        "routing": {"gate_norm": "global", "mode": "token", "pool_size": 8, "warmup": 16},
        "segmentation": {"cache": str(Path(seg_dir)), "cache_id": sman["cache_id"], "fixed_size": 128, "max_len": 256,
                         "min_len": 64, "semantic_tau": sman["tau"], "strategy": "none",
                         "random_cache": str(Path(rnd_dir)), "random_cache_id": rman["random_cache_id"],
                         "random_generator": rman["spec"]["version"], "random_seed": rman["spec"]["seed"],
                         "random_acceptance": "all prereg criteria on train and calib (first passing seed)"},
        "semantic": {"comparison_set": "all valid de-duplicated chunks of the micro-batch (complete w-prefix), "
                                       "evenly-spaced subsample to n_max; multiple documents per micro-batch",
                     "enabled": False, "lambda": None, "min_chunks": 16, "n_max": 256, "tau_r": None, "tau_s": None},
        "train": {"accum": 2, "activation_checkpointing": False, "aux_coeff": 0.01,
                  "aux_scope": "per micro-batch, per MoE layer, summed over layers, non-pad tokens",
                  "betas": [0.9, 0.95], "clip": 1.0, "compile": False, "decay_frac": 0.2, "dropout": 0.0,
                  "eval_every_tokens": 5_000_000, "grad_probe_every": 25, "log_every": 25, "lr": 6e-4,
                  "max_seconds": 5400, "micro_batch_rows": 16, "min_lr": 6e-5, "precision": "bf16-autocast",
                  "router_precision": "fp32", "save_checkpoint": True, "seed": 0, "throughput_warmup_steps": 10,
                  "token_budget": 100_000_000, "warmup_steps": 60, "weight_decay": 0.1, "z_coeff": 0.001},
        "trainable": "all LM parameters (embedding tied to LM head)",
    }


def verify(base):
    import torch
    from csr.model import count_parameters
    from csr.train_lib import build_model
    cfgs = {c: build_condition_config(base, c, run_name=f"x_{c}") for c in CONDITIONS}
    check_matched({c: {**v, "run_name": None} for c, v in cfgs.items()})
    sds, counts = {}, None
    for c, cfg in cfgs.items():
        m = build_model(cfg, torch.device("cpu"))
        assert m.lm_head.weight is m.tok_emb.weight
        cp = count_parameters(m)
        assert (cp["total"], cp["active_per_token"]) == (TOTAL, ACTIVE), cp
        sds[c] = m.state_dict()
    assert 20e6 <= TOTAL <= 30e6 and (TOTAL - EMB) / TOTAL >= 0.5
    for c in "BCDEF":
        assert list(sds[c]) == list(sds["A"]) and all(torch.equal(sds[c][k], sds["A"][k]) for k in sds["A"])
    return {"total": TOTAL, "active": ACTIVE, "nonemb_frac": (TOTAL - EMB) / TOTAL, "conditions_identical_init": True}


def build_seed_configs(base, tau_s, tau_r, lam):
    """One globally frozen (tau_s, tau_r, lambda) written identically into every A-F config of every seed
    (amendment 1 §5), including conditions with the semantic loss disabled. Returns {seed: (base, {cond: cfg})}."""
    out = {}
    for s in (0, 1, 2):
        b = copy.deepcopy(base)
        b["semantic"].update({"tau_s": tau_s, "tau_r": tau_r, "lambda": lam})
        b["init"]["seed"] = b["train"]["seed"] = s
        b["data"]["train_order_seed"] = 17 + s
        out[s] = (b, {c: build_condition_config(b, c, run_name=f"rep_v1_s{s}_{c}") for c in CONDITIONS})
    return out


def write(path, obj):
    guard.atomic_write_json(path, obj, mode=0o400)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["precal", "tuning", "final"])
    ap.add_argument("--segments", required=True)
    ap.add_argument("--random", required=True)
    ap.add_argument("--tau-s", type=float)
    ap.add_argument("--tau-r", type=float)
    ap.add_argument("--lam", type=float)
    a = ap.parse_args()
    guard.check_root()
    base = base_config(a.segments, a.random)
    if a.stage == "precal":
        d = OUT / "precal"
        d.mkdir(parents=True, exist_ok=False)
        info = verify(base)
        write(d / "base.json", base)
        print(json.dumps(info))
        return
    assert a.tau_s is not None and a.tau_r is not None
    base["semantic"].update({"tau_s": a.tau_s, "tau_r": a.tau_r})
    if a.stage == "tuning":
        d = OUT / "tuning"
        d.mkdir(parents=True, exist_ok=False)
        t = copy.deepcopy(base)
        t["init"]["seed"] = t["train"]["seed"] = 99
        t["data"]["train_order_seed"] = 17 + 99
        t["train"].update({"token_budget": 5_000_000, "grad_probe_every": 5, "log_every": 5})
        t["result_label"] = LABEL + " [disposable lambda-tuning run, training data only]"
        verify(t)
        write(d / "base.json", t)
        print("tuning config written")
        return
    assert a.lam is not None
    base["semantic"]["lambda"] = a.lam
    mans = {}
    for s, (b, cfgs) in build_seed_configs(base, a.tau_s, a.tau_r, a.lam).items():
        d = OUT / f"seed{s}"
        d.mkdir(parents=True, exist_ok=False)
        info = verify(b)
        write(d / "base.json", b)
        for c in CONDITIONS:
            write(d / f"condition_{c}.json", cfgs[c])
        m = manifest({c: {**v, "run_name": None} for c, v in cfgs.items()}, None, None,
                     notes={"seed": s, "params": info})
        mans[f"seed{s}"] = m
    write(OUT / "MANIFEST.json", {"experiment": "csr-representative-v1", "prereg_sha256": base["prereg_sha256"],
                                  "frozen_semantic": base["semantic"], "per_seed": mans, "created": guard.now()})
    guard.ledger("REP_V1_CONFIGS", f"configs/rep-v1 final frozen: tau_s={a.tau_s} tau_r={a.tau_r} lambda={a.lam}")
    print("final configs written")


if __name__ == "__main__":
    main()
