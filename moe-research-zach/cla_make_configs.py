#!/usr/bin/env python
"""csr-conditional-local-attention-v1 config generation (never overwrites).

  cla_make_configs.py gen1                       -> gen1/seed98/        (5M tokens; baseline + all 10 candidates)
  cla_make_configs.py gen2 --ids A,B,...         -> gen2/seed{98,99}/   (20M tokens; checkpoints)
  cla_make_configs.py final --winner ID          -> final/seed{0,1,2}/  (100M tokens; checkpoints; resumable state)
Base: configs/rep-v1/seed0/condition_A.json (exact csr-representative-v1 TOKEN-4 model, data, optimizer, schedule,
precision, auxiliary losses). init seed = seed, train_order_seed = 17 + seed (the convention of all earlier final
runs). Within a directory only the approved fields differ (candidate id, condition, attention plan, run name, role).
"""
import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402
from csr.cla_model import EXPERIMENT, AttnPlan  # noqa: E402
from csr.cla_run import CFG_ROOT, prereg_sha  # noqa: E402
from csr.cla_select import BASELINE, CANDIDATES, CONDITION, PLANS  # noqa: E402

APPROVED = {"candidate", "condition", "run_name", "role"} | {f"attention.{k}" for k in AttnPlan.__dataclass_fields__}
STAGES = {"gen1": ((98,), 5_000_000, False), "gen2": ((98, 99), 20_000_000, True),
          "final": ((0, 1, 2), 100_000_000, True)}


def base_cfg(seed, budget, stage, save_ckpt):
    b = copy.deepcopy(json.loads((guard.ROOT / "configs/rep-v1/seed0/condition_A.json").read_text()))
    for k in ("semantic", "condition", "segmentation", "routing"):
        b.pop(k, None)
    b["experiment"] = EXPERIMENT
    b["prereg_sha256"] = prereg_sha()
    b["stage"] = stage
    b["result_label"] = f"{EXPERIMENT}: {stage}"
    b["init"] = {"type": "random", "seed": seed}
    b["data"]["train_order_seed"] = 17 + seed
    t = b["train"]
    t.update({"seed": seed, "token_budget": int(budget), "grad_probe_every": 0, "log_every": 25,
              "save_checkpoint": bool(save_ckpt), "max_seconds": 5400, "eval_every_tokens": 5_000_000,
              "state_every_seconds": 600 if stage == "final" else 0, "throughput_warmup_steps": 10})
    t.pop("aux_scope", None)
    b["eval"] = {"rows_per_batch": 16, "max_rows": 128, "final_calib_search": True,
                 "monitoring_subset": "first 128 packed rows of CALIB-SEARCH (monitoring only)"}
    b["routing_note"] = "unrestricted TOKEN-4: global softmax top-4 of 32, fp32 router, no renormalisation, dropless"
    return b


def make(seed, budget, stage, save_ckpt, cid):
    c = base_cfg(seed, budget, stage, save_ckpt)
    c["candidate"] = cid
    c["condition"] = CONDITION[cid]
    c["role"] = "baseline" if cid == BASELINE else "candidate"
    c["attention"] = AttnPlan.from_cfg(PLANS[cid]).to_dict()
    c["run_name"] = f"{stage}_s{seed}_{cid}"
    return c


def check_matched(cfgs):
    from csr.config import flatten
    flats = {k: flatten(c) for k, c in cfgs.items()}
    keys = set().union(*[f.keys() for f in flats.values()])
    bad = []
    for k in sorted(keys):
        if k in APPROVED:
            continue
        vals = {c: json.dumps(f.get(k, "<MISSING>"), sort_keys=True) for c, f in flats.items()}
        if len(set(vals.values())) > 1:
            bad.append((k, vals))
    if bad:
        raise ValueError("unmatched config fields: " + "; ".join(f"{k}: {v}" for k, v in bad))


def write_dir(d: Path, cfgs: dict):
    if d.exists():
        raise guard.GuardError(f"{d} exists; never overwritten")
    check_matched(cfgs)
    d.mkdir(parents=True)
    idx = {}
    for name, c in sorted(cfgs.items()):
        p = d / f"{name}.json"
        guard.atomic_write_json(p, c, mode=0o400)
        idx[name] = {"path": str(p.relative_to(guard.ROOT)), "sha256": guard.sha256_file(p),
                     "token_budget": c["train"]["token_budget"]}
    guard.atomic_write_json(d / "INDEX.json", {"created": guard.now(), "matched_check": "passed",
                                               "approved_differing_fields": sorted(APPROVED), "configs": idx},
                            mode=0o400)
    return idx


def stage_dirs(stage, ids):
    seeds, budget, ck = STAGES[stage]
    out = {}
    for s in seeds:
        out[s] = write_dir(CFG_ROOT / stage / f"seed{s}", {i: make(s, budget, stage, ck, i) for i in ids})
    guard.ledger("CLA_V1_CONFIGS", f"{stage}: {sum(len(v) for v in out.values())} configs ids={ids}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["gen1", "gen2", "final"])
    ap.add_argument("--ids", default="")
    ap.add_argument("--winner", default="")
    a = ap.parse_args()
    guard.check_root()
    valid_ids = {c[0] for c in CANDIDATES}
    if a.stage == "gen1":
        ids = [BASELINE] + [c[0] for c in CANDIDATES]
    elif a.stage == "gen2":
        ids = [x for x in a.ids.split(",") if x]
        assert ids and len(ids) <= 4 and set(ids) <= valid_ids, ids
        ids = [BASELINE] + ids
    else:
        assert a.winner in valid_ids, a.winner
        ids = [BASELINE, a.winner]
    print(json.dumps(stage_dirs(a.stage, ids), indent=1))


if __name__ == "__main__":
    main()
