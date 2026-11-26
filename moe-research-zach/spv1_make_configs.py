#!/usr/bin/env python
"""csr-speculative-pool-routing-v1 config generation (never overwrites).

  spv1_make_configs.py draft                          -> configs/speculative-pool-routing-v1/draft/draft_d128_l4.json
  spv1_make_configs.py gen1                           -> gen1/seed98/              (5M)
  spv1_make_configs.py gen --gen 2 --specs F.json     -> gen2/seed{98,99}/         (20M)
  spv1_make_configs.py gen --gen 3 --specs F.json     -> gen3/seed{97,98,99}/      (50M, checkpoints)
  spv1_make_configs.py final --specs F.json --budget N [--controls-budget M]
                                                      -> final_B<N>M/seed{0,1,2}/  (checkpoints, resumable state)
Base: configs/rep-v1/seed0/condition_A.json (exact csr-representative-v1 model, data, optimizer, schedule,
precision, auxiliary losses). Within a directory only the fields in APPROVED differ (and, for random controls run
at a secondary budget, train.token_budget). Every target directory contains TOKEN-4.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402
from csr.spv1_model import EXPERIMENT, PoolRouting, draft_model_config  # noqa: E402
from csr.spv1_run import CFG_ROOT, prereg_sha  # noqa: E402
from csr.spv1_select import GEN1, TOKEN  # noqa: E402

APPROVED = {"candidate", "run_name", "family", "description", "role"} | {f"routing.{k}" for k in
                                                                         PoolRouting.__dataclass_fields__}
STAGES = {1: ("gen1", (98,), 5_000_000, False), 2: ("gen2", (98, 99), 20_000_000, False),
          3: ("gen3", (97, 98, 99), 50_000_000, True)}
MAIN_SEEDS = (0, 1, 2)


def base_cfg(seed, budget, stage, save_ckpt):
    b = copy.deepcopy(json.loads((guard.ROOT / "configs/rep-v1/seed0/condition_A.json").read_text()))
    for k in ("semantic", "condition", "segmentation"):
        b.pop(k, None)
    b["experiment"] = EXPERIMENT
    b["kind"] = "target"
    b["prereg_sha256"] = prereg_sha()
    b["stage"] = stage
    b["result_label"] = f"{EXPERIMENT}: {stage}"
    b["init"] = {"type": "random", "seed": seed}
    b["data"]["train_order_seed"] = 17 + seed
    t = b["train"]
    long_run = stage.startswith("final") or stage == "gen3"
    t.update({"seed": seed, "token_budget": int(budget), "grad_probe_every": 0, "log_every": 25,
              "save_checkpoint": bool(save_ckpt), "max_seconds": 5400, "eval_every_tokens": 5_000_000,
              "state_every_seconds": 600 if long_run else 0})
    t.pop("aux_scope", None)
    b["eval"] = {"rows_per_batch": 16, "max_rows": 128, "final_calib_search": True,
                 "monitoring_subset": "first 128 packed rows of CALIB-SEARCH (monitoring only)"}
    return b


def make(seed, budget, stage, save_ckpt, spec, role="candidate"):
    c = base_cfg(seed, budget, stage, save_ckpt)
    c["candidate"] = spec["id"]
    c["family"] = spec["family"]
    c["role"] = role
    c["description"] = spec.get("desc", spec.get("change", ""))
    c["routing"] = PoolRouting.from_cfg(spec["routing"]).to_dict()
    c["run_name"] = f"{stage}_s{seed}_{spec['id']}"
    return c


def check_matched(cfgs, extra_ok=()):
    from csr.config import flatten
    flats = {k: flatten(c) for k, c in cfgs.items()}
    keys = set().union(*[f.keys() for f in flats.values()])
    bad = []
    for k in sorted(keys):
        if k in APPROVED or k in extra_ok:
            continue
        vals = {c: json.dumps(f.get(k, "<MISSING>"), sort_keys=True) for c, f in flats.items()}
        if len(set(vals.values())) > 1:
            bad.append((k, vals))
    if bad:
        raise ValueError("unmatched config fields: " + "; ".join(f"{k}: {v}" for k, v in bad))


def write_dir(d: Path, cfgs: dict, extra_ok=()):
    if d.exists():
        raise guard.GuardError(f"{d} exists; never overwritten")
    check_matched(cfgs, extra_ok)
    d.mkdir(parents=True)
    idx = {}
    for name, c in sorted(cfgs.items()):
        p = d / f"{name}.json"
        guard.atomic_write_json(p, c, mode=0o400)
        idx[name] = {"path": str(p.relative_to(guard.ROOT)), "sha256": guard.sha256_file(p), "family": c["family"],
                     "role": c.get("role"), "token_budget": c["train"]["token_budget"]}
    guard.atomic_write_json(d / "INDEX.json", {"created": guard.now(), "matched_check": "passed",
                                               "approved_differing_fields": sorted(APPROVED) + list(extra_ok),
                                               "configs": idx}, mode=0o400)
    return idx


TOKEN_SPEC = {"id": "TOKEN-4", "family": "token", "routing": TOKEN, "desc": "standard token router, global top-4"}


def gen_dirs(gen, specs):
    stage, seeds, budget, ck = STAGES[gen]
    out = {}
    for s in seeds:
        cfgs = {sp["id"]: make(s, budget, stage, ck, sp, "baseline" if sp["id"] == "TOKEN-4" else "candidate")
                for sp in [TOKEN_SPEC] + list(specs)}
        out[s] = write_dir(CFG_ROOT / stage / f"seed{s}", cfgs)
    guard.ledger("SP_V1_CONFIGS", f"{stage}: {sum(len(v) for v in out.values())} configs")
    return out


def final_dirs(specs, budget, controls_budget):
    out = {}
    stage = f"final_B{budget // 1_000_000}M"
    for s in MAIN_SEEDS:
        cfgs = {}
        for sp in [TOKEN_SPEC] + list(specs):
            role = "baseline" if sp["id"] == "TOKEN-4" else sp.get("role", "finalist")
            B = controls_budget if role == "random_control" else budget
            cfgs[sp["id"]] = make(s, B, stage, True, sp, role)
        extra = ("train.token_budget",) if controls_budget != budget else ()
        out[s] = write_dir(CFG_ROOT / stage / f"seed{s}", cfgs, extra)
    guard.ledger("SP_V1_CONFIGS", f"final B={budget} controls={controls_budget}: "
                                  f"{sum(len(v) for v in out.values())} configs")
    return out


def draft_cfg():
    b = base_cfg(0, 100_000_000, "draft", True)
    b["kind"] = "draft"
    b.pop("routing", None)
    b["candidate"] = "DRAFT-d128-L4"
    b["family"] = "draft"
    b["role"] = "draft"
    b["description"] = "frozen dense causal draft: d128, 4 layers, 4 heads, SwiGLU 512, tied embeddings"
    b["model"] = draft_model_config().to_dict()
    b["train"]["lr"] = 0.001
    b["train"]["min_lr"] = 0.0001
    b["train"]["state_every_seconds"] = 600
    b["run_name"] = "draft_d128_l4"
    b["trainable"] = "all draft parameters (embedding tied to LM head)"
    return b


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("draft")
    sub.add_parser("gen1")
    g = sub.add_parser("gen")
    g.add_argument("--gen", type=int, choices=[2, 3], required=True)
    g.add_argument("--specs", required=True)
    f = sub.add_parser("final")
    f.add_argument("--specs", required=True)
    f.add_argument("--budget", type=int, required=True)
    f.add_argument("--controls-budget", type=int)
    a = ap.parse_args()
    guard.check_root()
    if a.cmd == "draft":
        d = CFG_ROOT / "draft"
        if d.exists():
            raise guard.GuardError(f"{d} exists")
        d.mkdir(parents=True)
        p = d / "draft_d128_l4.json"
        guard.atomic_write_json(p, draft_cfg(), mode=0o400)
        guard.ledger("SP_V1_CONFIGS", f"draft config {p.relative_to(guard.ROOT)} sha256={guard.sha256_file(p)}")
        print(p)
        return
    if a.cmd == "gen1":
        r = gen_dirs(1, GEN1)
    elif a.cmd == "gen":
        r = gen_dirs(a.gen, json.loads(guard.inside_root(a.specs).read_text())["specs"])
    else:
        r = final_dirs(json.loads(guard.inside_root(a.specs).read_text())["specs"], a.budget,
                       a.controls_budget or a.budget)
    print(json.dumps({str(k): sorted(v) for k, v in r.items()}))


if __name__ == "__main__":
    main()
