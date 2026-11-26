#!/usr/bin/env python
"""csr-multiscale-routing-v1 config generation (never overwrites).

  ms_make_configs.py calibration                        -> configs/multiscale-routing-v1/calibration/...
  ms_make_configs.py branch --alpha A --beta B           -> benchmark/, pilot/ for the selected branch
  ms_make_configs.py main --alpha A --beta B --budget N  -> main_B<N>M/seed{0,1,2}/
Base: configs/rep-v1/seed0/condition_A.json (exact rep-v1 model/optimizer/data); only the fields listed in the
preregistration differ. Every directory gets an INDEX.json with per-file sha256 and the matched-field check result.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402
from csr.multiscale import (APPROVED_DIFF_CALIB, APPROVED_DIFF_MAIN, BRANCH_CONDITIONS, EXPERIMENT, branch_of,  # noqa: E402
                            check_matched, config_for)
from csr.ms_run import prereg_sha  # noqa: E402
from csr.ms_select import GRID, PILOT_TOKENS  # noqa: E402

OUT = guard.ROOT / "configs" / "multiscale-routing-v1"
CAL_SEEDS = (98, 99)
MAIN_SEEDS = (0, 1, 2)
BETA_PLACEMENTS = ("BROAD-EARLY", "BROAD-LATE")
ALPHA_PLACEMENTS = ("LOCAL-FIXED", "LOCAL-SEMANTIC", "LOCAL-RANDOM")


def base_cfg(seed, budget, stage):
    b = json.loads((guard.ROOT / "configs/rep-v1/seed0/condition_A.json").read_text())
    b = copy.deepcopy(b)
    b.pop("semantic", None)
    b.pop("prereg_sha256", None)
    b["experiment"] = EXPERIMENT
    b["prereg_sha256"] = prereg_sha()
    b["stage"] = stage
    b["result_label"] = f"{EXPERIMENT}: {stage} (exploratory; causal multiscale context-conditioned token routing)"
    b["init"] = {"type": "random", "seed": seed}
    b["data"]["train_order_seed"] = 17 + seed
    t = b["train"]
    t.update({"seed": seed, "token_budget": int(budget), "grad_probe_every": 25, "log_every": 25,
              "save_checkpoint": stage == "main", "max_seconds": 5400, "eval_every_tokens": 5_000_000})
    t.pop("aux_scope", None)
    b["eval"] = {"rows_per_batch": 16, "max_rows": 128,
                 "monitor_splits": ["calib", "val"] if stage == "main" else ["calib"],
                 "final_full_calib": stage in ("calibration", "pilot"),
                 "monitoring_subset": "first 128 packed rows of each monitor split (monitoring only)"}
    return b


def tag(x):
    return str(float(x)).replace(".", "p")


def write_dir(d: Path, cfgs: dict, approved):
    if d.exists():
        raise guard.GuardError(f"{d} exists; never overwritten")
    check_matched(cfgs, approved)
    d.mkdir(parents=True)
    idx = {}
    for name, c in sorted(cfgs.items()):
        p = d / f"{name}.json"
        guard.atomic_write_json(p, c, mode=0o400)
        idx[name] = {"path": str(p.relative_to(guard.ROOT)), "sha256": guard.sha256_file(p)}
    guard.atomic_write_json(d / "INDEX.json", {"created": guard.now(), "matched_check": "passed",
                                               "approved_differing_fields": sorted(approved), "configs": idx}, mode=0o400)
    return idx


def calibration():
    out = {}
    for seed in CAL_SEEDS:
        cfgs = {}
        for c in GRID:
            for pl in BETA_PLACEMENTS:
                cfgs[f"beta_{tag(c)}__{pl}"] = config_for(base_cfg(seed, 5_000_000, "calibration"), pl, 0.0, c,
                                                          f"cal_s{seed}_beta{tag(c)}_{pl}")
            for pl in ALPHA_PLACEMENTS:
                cfgs[f"alpha_{tag(c)}__{pl}"] = config_for(base_cfg(seed, 5_000_000, "calibration"), pl, c, 0.0,
                                                           f"cal_s{seed}_alpha{tag(c)}_{pl}")
        out[seed] = write_dir(OUT / "calibration" / f"seed{seed}", cfgs, APPROVED_DIFF_CALIB)
    guard.ledger("MS_V1_CONFIGS", f"calibration configs: {sum(len(v) for v in out.values())} files")
    return out


def branch_configs(alpha, beta):
    br = branch_of(alpha, beta)
    conds = BRANCH_CONDITIONS[br]
    if not conds:
        raise guard.GuardError("branch A: no further configs")
    out = {}
    bench = {c: config_for(base_cfg(0, 100_000_000, "benchmark"), c, alpha, beta, f"bench_s0_{c}") for c in conds}
    out["benchmark"] = write_dir(OUT / "benchmark", bench, APPROVED_DIFF_MAIN)
    pilot = {c: config_for(base_cfg(0, PILOT_TOKENS, "pilot"), c, alpha, beta, f"pilot_s0_{c}") for c in conds}
    out["pilot"] = write_dir(OUT / "pilot", pilot, APPROVED_DIFF_MAIN)
    guard.ledger("MS_V1_CONFIGS", f"benchmark+pilot configs branch={br} alpha={alpha} beta={beta}")
    return out


def main_configs(alpha, beta, budget):
    conds = BRANCH_CONDITIONS[branch_of(alpha, beta)]
    out = {}
    for s in MAIN_SEEDS:
        main_c = {c: config_for(base_cfg(s, budget, "main"), c, alpha, beta, f"main_s{s}_{c}") for c in conds}
        out[f"main_seed{s}"] = write_dir(OUT / f"main_B{budget // 1_000_000}M" / f"seed{s}", main_c, APPROVED_DIFF_MAIN)
    guard.ledger("MS_V1_CONFIGS", f"main configs alpha={alpha} beta={beta} budget={budget}")
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("calibration")
    s = sub.add_parser("branch")
    s.add_argument("--alpha", type=float, required=True)
    s.add_argument("--beta", type=float, required=True)
    m = sub.add_parser("main")
    m.add_argument("--alpha", type=float, required=True)
    m.add_argument("--beta", type=float, required=True)
    m.add_argument("--budget", type=int, required=True)
    a = ap.parse_args()
    guard.check_root()
    if a.cmd == "calibration":
        print(json.dumps({k: len(v) for k, v in calibration().items()}))
    elif a.cmd == "branch":
        print(json.dumps({k: len(v) for k, v in branch_configs(a.alpha, a.beta).items()}))
    else:
        print(json.dumps({k: len(v) for k, v in main_configs(a.alpha, a.beta, a.budget).items()}))


if __name__ == "__main__":
    main()
