#!/usr/bin/env python
"""csr-sequence-routing-search-v1 config generation (never overwrites).

  srs_make_configs.py gen1                          -> configs/sequence-routing-search-v1/gen1/seed98/
  srs_make_configs.py gen  --gen 2 --specs F.json   -> gen2/seed{98,99}/   (10M)
  srs_make_configs.py gen  --gen 3 --specs F.json   -> gen3/seed{97,98,99}/ (30M, checkpoints)
  srs_make_configs.py final --specs F.json --budget N -> final_B<N>M/seed{0,1,2}/ (checkpoints, resumable state)
Base: configs/rep-v1/seed0/condition_A.json (exact csr-representative-v1 model, data, optimizer, schedule, precision,
auxiliary losses). Only the fields in APPROVED differ within a directory; every directory contains TOKEN-4.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402
from csr.seqroute import EXPERIMENT, SRRouting  # noqa: E402
from csr.srs_run import CFG_ROOT, prereg_sha  # noqa: E402
from csr.srs_select import GEN1, TOKEN  # noqa: E402

APPROVED = {"candidate", "run_name", "family", "description"} | {f"routing.{k}" for k in
                                                                 SRRouting.__dataclass_fields__}
STAGES = {1: ("gen1", (98,), 5_000_000, False), 2: ("gen2", (98, 99), 10_000_000, False),
          3: ("gen3", (97, 98, 99), 30_000_000, True)}
MAIN_SEEDS = (0, 1, 2)


def base_cfg(seed, budget, stage, save_ckpt):
    b = copy.deepcopy(json.loads((guard.ROOT / "configs/rep-v1/seed0/condition_A.json").read_text()))
    b.pop("semantic", None)
    b.pop("condition", None)
    b["experiment"] = EXPERIMENT
    b["prereg_sha256"] = prereg_sha()
    b["stage"] = stage
    b["result_label"] = f"{EXPERIMENT}: {stage}"
    b["init"] = {"type": "random", "seed": seed}
    b["data"]["train_order_seed"] = 17 + seed
    b["segmentation"] = {"note": "sequence segmentation is computed inside the model (seqroute.py); no cache used"}
    t = b["train"]
    t.update({"seed": seed, "token_budget": int(budget), "grad_probe_every": 0, "log_every": 25,
              "save_checkpoint": bool(save_ckpt), "max_seconds": 5400, "eval_every_tokens": 5_000_000,
              "state_every_seconds": 600 if stage.startswith("final") else 0})
    t.pop("aux_scope", None)
    b["eval"] = {"rows_per_batch": 16, "max_rows": 128, "final_calib_search": True,
                 "monitoring_subset": "first 128 packed rows of CALIB-SEARCH (monitoring only)"}
    return b


def make(seed, budget, stage, save_ckpt, spec):
    c = base_cfg(seed, budget, stage, save_ckpt)
    c["candidate"] = spec["id"]
    c["family"] = spec["family"]
    c["description"] = spec.get("desc", spec.get("change", ""))
    r = SRRouting.from_cfg(spec["routing"]).to_dict()
    c["routing"] = r
    c["run_name"] = f"{stage}_s{seed}_{spec['id']}"
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
        idx[name] = {"path": str(p.relative_to(guard.ROOT)), "sha256": guard.sha256_file(p), "family": c["family"]}
    guard.atomic_write_json(d / "INDEX.json", {"created": guard.now(), "matched_check": "passed",
                                               "approved_differing_fields": sorted(APPROVED), "configs": idx},
                            mode=0o400)
    return idx


TOKEN_SPEC = {"id": "TOKEN-4", "family": "token", "routing": TOKEN, "desc": "standard token router, global top-4"}


def gen_dirs(gen, specs):
    stage, seeds, budget, ck = STAGES[gen]
    out = {}
    for s in seeds:
        cfgs = {sp["id"]: make(s, budget, stage, ck, sp) for sp in [TOKEN_SPEC] + list(specs)}
        out[s] = write_dir(CFG_ROOT / stage / f"seed{s}", cfgs)
    guard.ledger("SRS_V1_CONFIGS", f"{stage}: {sum(len(v) for v in out.values())} configs")
    return out


def final_dirs(specs, budget):
    out = {}
    for s in MAIN_SEEDS:
        cfgs = {sp["id"]: make(s, budget, f"final_B{budget // 1_000_000}M", True, sp)
                for sp in [TOKEN_SPEC] + list(specs)}
        out[s] = write_dir(CFG_ROOT / f"final_B{budget // 1_000_000}M" / f"seed{s}", cfgs)
    guard.ledger("SRS_V1_CONFIGS", f"final B={budget}: {sum(len(v) for v in out.values())} configs")
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("gen1")
    g = sub.add_parser("gen")
    g.add_argument("--gen", type=int, choices=[2, 3], required=True)
    g.add_argument("--specs", required=True)
    f = sub.add_parser("final")
    f.add_argument("--specs", required=True)
    f.add_argument("--budget", type=int, required=True)
    a = ap.parse_args()
    guard.check_root()
    if a.cmd == "gen1":
        r = gen_dirs(1, GEN1)
    elif a.cmd == "gen":
        r = gen_dirs(a.gen, json.loads(guard.inside_root(a.specs).read_text())["specs"])
    else:
        r = final_dirs(json.loads(guard.inside_root(a.specs).read_text())["specs"], a.budget)
    print(json.dumps({str(k): sorted(v) for k, v in r.items()}))


if __name__ == "__main__":
    main()
