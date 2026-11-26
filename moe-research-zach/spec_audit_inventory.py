#!/usr/bin/env python
"""specialization-audit-v1 Stage 0: read-only inventory of the 18 csr-representative-v1 checkpoints.

Verifies, without modifying anything: run status, checkpoint sha256 against summary.json, resolved config equal to
the frozen configs/rep-v1 condition file, condition mapping (routing mode / segmentation / semantic loss), strict
state-dict load, identical architecture and tokenizer, parameter counts, original semantic-teacher identity.
Writes artifacts/specialization-audit-v1/inventory.json (never overwritten). CPU only.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402

OUT = guard.ROOT / "artifacts" / "specialization-audit-v1"
EXPECTED = {  # condition -> (routing.mode, segmentation.strategy, semantic.enabled)
    "A": ("token", "none", False), "B": ("pool", "fixed", False), "C": ("pool", "fixed", True),
    "D": ("pool", "semantic", False), "E": ("pool", "semantic", True), "F": ("pool", "random", False)}


def verify_mapping(cfg, cond):
    got = (cfg["routing"]["mode"], cfg["segmentation"]["strategy"], cfg["semantic"]["enabled"])
    if got != EXPECTED[cond]:
        raise guard.GuardError(f"condition {cond} mapping mismatch: {got} != {EXPECTED[cond]}")
    return got


def load_runs(tsv):
    rows = [l.split("\t") for l in Path(tsv).read_text().splitlines() if l.strip()]
    runs = {}
    for s, c, r, _e in rows:
        key = (int(s), c)
        if key in runs:
            raise guard.GuardError(f"ambiguous mapping: {key} listed twice")
        runs[key] = guard.inside_root(r)
    if sorted(runs) != [(s, c) for s in (0, 1, 2) for c in "ABCDEF"]:
        raise guard.GuardError(f"expected 18 (seed, condition) runs, got {sorted(runs)}")
    return runs


def main():
    guard.check_root()
    import torch
    from csr.artifacts import ENCODER_REPO, ENCODER_REV
    from csr.model import CSRModel, ModelConfig, count_parameters
    runs = load_runs(guard.ROOT / ".tmp" / "rep_v1_main" / "runs.tsv")
    inv, archs, toks = {}, set(), set()
    for (s, c), r in sorted(runs.items()):
        st = json.loads((r / "status.json").read_text())["status"]
        sm = json.loads((r / "summary.json").read_text())
        cfg = json.loads((r / "config.json").read_text())
        frozen = json.loads((guard.ROOT / "configs" / "rep-v1" / f"seed{s}" / f"condition_{c}.json").read_text())
        if st != "COMPLETE":
            raise guard.GuardError(f"{r.name} status {st}")
        if cfg != frozen:
            raise guard.GuardError(f"{r.name} config differs from frozen condition file")
        if cfg["condition"] != c or cfg["init"]["seed"] != s or cfg["run_name"] != f"rep_v1_s{s}_{c}":
            raise guard.GuardError(f"{r.name} seed/condition labels inconsistent")
        mapping = verify_mapping(cfg, c)
        ck = r / "model.pt"
        h = guard.sha256_file(ck)
        if h != sm["checkpoint_sha256"]:
            raise guard.GuardError(f"{r.name} checkpoint sha256 mismatch (corrupt?)")
        sd = torch.load(ck, map_location="cpu", weights_only=True)
        if sd["config"] != cfg:
            raise guard.GuardError(f"{r.name} checkpoint-embedded config differs from config.json")
        m = CSRModel(ModelConfig.from_dict(cfg["model"]))
        m.load_state_dict(sd["model"], strict=True)
        assert m.lm_head.weight is m.tok_emb.weight
        cp = count_parameters(m)
        finite = all(torch.isfinite(v).all().item() for v in sd["model"].values() if v.is_floating_point())
        if not finite:
            raise guard.GuardError(f"{r.name} non-finite weights")
        archs.add(json.dumps(cfg["model"], sort_keys=True))
        dman = json.loads((guard.inside_root(cfg["data"]["dir"]) / "manifest.json").read_text())
        toks.add(json.dumps(dman["tokenizer"], sort_keys=True))
        inv[f"s{s}_{c}"] = {"seed": s, "condition": c, "run": str(r.relative_to(guard.ROOT)),
                            "checkpoint": str(ck.relative_to(guard.ROOT)), "checkpoint_sha256": h,
                            "checkpoint_bytes": ck.stat().st_size, "routing_mode": mapping[0],
                            "segmentation": mapping[1], "semantic_loss": mapping[2],
                            "gate_norm": cfg["routing"]["gate_norm"], "pool_size": cfg["routing"]["pool_size"],
                            "warmup": cfg["routing"]["warmup"], "params": cp, "steps": sm["steps"],
                            "train_tokens": sm["valid_train_tokens"], "segmentation_cache": cfg["segmentation"]["cache"],
                            "random_cache": cfg["segmentation"]["random_cache"],
                            "random_seed": cfg["segmentation"]["random_seed"]}
        print(f"s{s}_{c} OK {h[:12]} {mapping}", flush=True)
    if len(archs) != 1 or len(toks) != 1:
        raise guard.GuardError(f"architecture/tokenizer not identical across checkpoints: {len(archs)} {len(toks)}")
    model = json.loads(next(iter(archs)))
    seg_man = json.loads((guard.ROOT / "artifacts/segments/8e7edc7e14b3c880/manifest.json").read_text())
    out = {"audit": "specialization-audit-v1", "stage": "0 inventory (read-only)", "created": guard.now(),
           "n_checkpoints": len(inv), "checkpoints": inv, "architecture": model,
           "moe_block_indices": model["moe_layers"], "n_moe_layers": len(model["moe_layers"]),
           "primary_intervention_blocks": [model["moe_layers"][0], model["moe_layers"][len(model["moe_layers"]) // 2],
                                           model["moe_layers"][-1]],
           "tokenizer": json.loads(next(iter(toks))),
           "semantic_teacher": {"repo": ENCODER_REPO, "revision": ENCODER_REV,
                                "segmentation_spec_encoder": seg_man["spec"]["encoder"],
                                "role": "native D/E segmentation + training-time semantic targets (never audit labeler)"},
           "segmentation_tau": seg_man["tau"], "checkpoints_modified": False,
           "note": "checkpoints opened read-only (sha256 + torch.load weights_only); no file under runs/ written"}
    OUT.mkdir(parents=True, exist_ok=True)
    guard.atomic_write_json(OUT / "inventory.json", out, mode=0o400)
    guard.ledger("SPEC_AUDIT_INVENTORY", f"18/18 checkpoints verified (sha256, strict load, mapping, identical arch); "
                                         f"artifacts/specialization-audit-v1/inventory.json")
    print("primary blocks", out["primary_intervention_blocks"])


if __name__ == "__main__":
    main()
