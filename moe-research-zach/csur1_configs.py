#!/usr/bin/env python
"""csr-crossfit-semantic-utility-residual-v1 configs. TOKEN-4 model/train/budget/diag copied from the reused T4 s0 config.

  csur1_configs.py gputest                      test configs (no prereg hash; tests only)
  csur1_configs.py stage1 [--gen G] [--grid JSON]   Stage-1 search config (seed 100; G + 7 SU/SH candidate pairs)
  csur1_configs.py final --seed S --c C --eta E --ema D --lam L [--gen G]   Stage-2/final config (G, SU, SH, SO, UO)"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402

BASE = {"c": 1.0, "eta": 0.5, "ema": 0.99, "lam": 0.3}
GRID = [("bp", {}), ("c0.5", {"c": 0.5}), ("c2", {"c": 2.0}), ("eta0.25", {"eta": 0.25}), ("eta1", {"eta": 1.0}),
        ("ema0.9", {"ema": 0.9}), ("lam1", {"lam": 1.0})]


def t4_base():
    from csr.csur1_paths import T4_CFG_RUN
    c = json.loads((guard.ROOT / T4_CFG_RUN / "config.json").read_text())
    return {k: c[k] for k in ("budget_tokens", "diag_every", "model", "train")}


GEN_ROUTER = {}  # filled from --gen: generation >= 1 uses the trigger-5 router (prereg section 12)


def util(name, prior, util_=True, **hp):
    return {"name": name, "kind": "util", "prior": prior, "util": util_, **{**BASE, **hp}, **GEN_ROUTER}


def final_banks(hp):
    return [{"name": "G", "kind": "token"}, util("SU", "sem", **hp), util("SH", "shuf", **hp),
            util("SO", "sem", util_=False, **hp), util("UO", None, **hp)]


def stage1_banks(grid=None):
    out = [{"name": "G", "kind": "token"}]
    for tag, hp in (grid or GRID):
        out += [util(f"SU_{tag}", "sem", **hp), util(f"SH_{tag}", "shuf", **hp)]
    return out


def write(name, cfg):
    from csr.csur1_paths import CFG_DIR
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    p = CFG_DIR / f"{name}.json"
    if p.exists():
        if json.loads(p.read_text()) != cfg:
            raise guard.GuardError(f"config {p} exists with different content")
        return p
    guard.atomic_write_json(p, cfg, mode=0o400)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["gputest", "stage1", "final"])
    ap.add_argument("--seed", type=int)
    for k in ("c", "eta", "ema", "lam"):
        ap.add_argument(f"--{k}", type=float)
    ap.add_argument("--gen", type=int, default=0)
    ap.add_argument("--grid", help="JSON list of [tag, overrides] (engineering generations only)")
    a = ap.parse_args()
    guard.check_root()
    from csr.csur1_paths import N_STEPS, SEARCH_SEED, prereg_sha
    common = {"experiment": "csr-crossfit-semantic-utility-residual-v1", **t4_base()}
    if a.what == "gputest":
        for nm, banks in (("gt_banks", final_banks({})), ("gt_nobank", [])):
            cfg = {**common, "run_name": nm, "seed": 0, "test": True, "cond": {"banks": banks},
                   "milestones": {"t60": 60} if banks else {}, "bank_milestones": {}}
            print(write(nm, cfg))
        cfg = {**common, "run_name": "gt_bench15", "seed": SEARCH_SEED, "test": True,
               "cond": {"banks": stage1_banks()}, "milestones": {}, "bank_milestones": {}}
        print(write("gt_bench15", cfg))
        return
    sha = prereg_sha()
    tb = json.loads((guard.ROOT / "artifacts/joint-semantic-residual-pretrain-v1/tables/tables.json").read_text())
    bs = tb["budget_steps"]
    g = f"_g{a.gen}" if a.gen else ""
    if a.gen >= 1:
        GEN_ROUTER["router"] = "meanlast"
    if a.what == "stage1":
        grid = [tuple(x) for x in json.loads(a.grid)] if a.grid else None
        nm = f"S1{g}_s{SEARCH_SEED}"
        cfg = {**common, "run_name": nm, "seed": SEARCH_SEED, "prereg_sha256": sha, "generation": a.gen,
               "cond": {"banks": stage1_banks(grid)}, "milestones": {"m20": bs["S1"]}, "bank_milestones": {}}
        print(write(nm, cfg))
    else:
        hp = {k: getattr(a, k) for k in ("c", "eta", "ema", "lam")}
        nm = f"F{g}_c{a.c}_e{a.eta}_d{a.ema}_l{a.lam}_s{a.seed}"
        cfg = {**common, "run_name": nm, "seed": a.seed, "prereg_sha256": sha, "generation": a.gen,
               "cond": {"banks": final_banks(hp)}, "milestones": {"m50": bs["S2"], "p100": N_STEPS},
               "bank_milestones": {"m100": bs["M100"], "m150": bs["M150"], "m200": bs["M200"]}}
        print(write(nm, cfg))


if __name__ == "__main__":
    main()
