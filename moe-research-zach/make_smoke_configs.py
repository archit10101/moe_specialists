#!/usr/bin/env python
"""Generate the separate, versioned smoke-small-v1 configuration (ENGINEERING SMOKE TESTS ONLY).

Derived from the frozen configs/v2/base.json: identical data, segmentation caches (semantic, fixed, accepted F-v2
random), warmup w, gate normalization, optimizer and schedule; ONLY the model shape (and m = min(E, 2k)) and the
default smoke token budget differ. Does not modify configs/v2 or EXPERIMENT_MANIFEST.json; writes its own manifest
inside configs/smoke-small-v1/. Refuses to overwrite. No model, no CUDA.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402
from csr.config import CONDITIONS, build_condition_config, check_matched, flatten, manifest, set_path  # noqa: E402

TAG = "smoke-small-v1"
LABEL = "ENGINEERING SMOKE TEST ONLY - small model, not experimental evidence"
MODEL = {"d_model": 128, "n_layer": 6, "moe_layers": [1, 2, 3, 4, 5], "n_head": 8, "n_experts": 16, "top_k": 2,
         "expert_ffn": 64, "shared_ffn": 128, "dense_ffn": 512, "max_seq_len": 1024}


def main():
    guard.check_root()
    src = guard.ROOT / "configs" / "v2" / "base.json"
    v2 = json.loads(src.read_text())
    base = json.loads(src.read_text())
    base["experiment"] = f"csr-{TAG}"
    base["result_label"] = LABEL
    base["derived_from"] = {"path": "configs/v2/base.json", "sha256": guard.sha256_file(src)}
    for k, v in MODEL.items():
        set_path(base, f"model.{k}", v)
    base["routing"]["pool_size"] = min(MODEL["n_experts"], 2 * MODEL["top_k"])  # m = min(E, 2k) = 4
    base["train"]["token_budget"] = 300_000  # smoke range 100k-500k; benchmark commands override
    assert base["data"]["seq_len"] == MODEL["max_seq_len"] == 1024
    # everything except the declared model/routing/budget/label fields must equal v2
    allowed = {f"model.{k}" for k in MODEL} | {"routing.pool_size", "train.token_budget", "experiment",
                                               "result_label", "derived_from.path", "derived_from.sha256"}
    fv, fb = flatten(v2), flatten(base)
    diff = sorted(k for k in set(fv) | set(fb) if fv.get(k, "<M>") != fb.get(k, "<M>") and k not in allowed)
    assert not diff, diff
    out = guard.ROOT / "configs" / TAG
    out.mkdir(parents=True, exist_ok=False)
    guard.atomic_write_json(out / "base.json", base)
    cfgs = {c: build_condition_config(base, c) for c in CONDITIONS}
    check_matched(cfgs)
    for c, cfg in cfgs.items():
        guard.atomic_write_json(out / f"condition_{c}.json", cfg)
    man = manifest(cfgs, None, None, notes={"label": LABEL, "derived_from": base["derived_from"],
                                            "differs_from_v2_only_in": sorted(allowed)})
    guard.atomic_write_json(out / "MANIFEST.json", man)
    (out / "SMOKE_ONLY.txt").write_text(LABEL + "\nFull-size experiment design: configs/v2 (frozen).\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
