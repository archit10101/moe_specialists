#!/usr/bin/env python
"""joint-semantic-residual-pretrain-v1 configs. TOKEN-4 model/train/budget/diag copied from the reused T4 s0 config.

  jsr_configs.py gputest                      test configs (no protocol hash; GPU checks only)
  jsr_configs.py stage1 [--alpha A] [--gen G] Stage-1 search config (seed 100; 8 banks)
  jsr_configs.py final --seed S --r R --tau T --alpha A [--gen G]   Stage-2/3 config (banks B, C, D, E)"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402


def t4_base():
    from csr.jsr_paths import T4_CFG_RUN
    c = json.loads((guard.ROOT / T4_CFG_RUN / "config.json").read_text())
    return {k: c[k] for k in ("budget_tokens", "diag_every", "model", "train")}


def bank(name, kind="routed", tau=1.0, r=0.0, alpha=0.01, groups=None):
    return {"name": name, "kind": kind, "tau": tau, "r": r, "alpha": alpha, "groups": groups} if kind == "routed" else \
        {"name": name, "kind": "oracle"}


def final_banks(r, tau, alpha):
    return [bank("B", tau=tau, r=0.0, alpha=alpha), bank("C", tau=tau, r=r, alpha=alpha, groups="shuf"),
            bank("D", tau=tau, r=r, alpha=alpha, groups="sem"), bank("E", kind="oracle")]


def stage1_banks(alpha, taus=None, rs=None):
    from csr.jsr_paths import R_GRID, TAU_GRID
    out = []
    for tau in (taus or TAU_GRID):
        out.append(bank(f"B_t{tau}", tau=tau, alpha=alpha))
        for r in (rs or R_GRID):
            out.append(bank(f"D_r{r}_t{tau}", tau=tau, r=r, alpha=alpha, groups="sem"))
    return out


def write(name, cfg):
    from csr.jsr_paths import CFG_DIR
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
    ap.add_argument("--r", type=float)
    ap.add_argument("--tau", type=float)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--gen", type=int, default=0)
    ap.add_argument("--taus")
    ap.add_argument("--rs")
    a = ap.parse_args()
    guard.check_root()
    from csr.jsr_paths import ART, SEARCH_SEED, prereg_sha
    tb = json.loads((ART / "tables/tables.json").read_text())
    bs = tb["budget_steps"]
    common = {"experiment": "joint-semantic-residual-pretrain-v1", **t4_base()}
    if a.what == "gputest":
        for nm, banks in (("gt_banks", final_banks(0.02, 1.0, 0.01)), ("gt_nobank", [])):
            cfg = {**common, "run_name": nm, "seed": 0, "test": True, "cond": {"banks": banks, "lam_r": 1.0},
                   "milestones": {}, "bank_milestones": {}}
            print(write(nm, cfg))
        cfg = {**common, "run_name": "gt_bench8", "seed": SEARCH_SEED, "test": True,
               "cond": {"banks": stage1_banks(0.01), "lam_r": 1.0}, "milestones": {}, "bank_milestones": {}}
        print(write("gt_bench8", cfg))
        return
    sha = prereg_sha()
    g = f"_g{a.gen}" if a.gen else ""
    if a.what == "stage1":
        taus = tuple(float(x) for x in a.taus.split(",")) if a.taus else None
        rs = tuple(float(x) for x in a.rs.split(",")) if a.rs else None
        nm = f"S1{g}_a{a.alpha}" + (f"_t{a.taus}_r{a.rs}" if (a.taus or a.rs) else "") + f"_s{SEARCH_SEED}"
        cfg = {**common, "run_name": nm, "seed": SEARCH_SEED, "prereg_sha256": sha, "generation": a.gen,
               "cond": {"banks": stage1_banks(a.alpha, taus, rs), "lam_r": 1.0},
               "milestones": {"m20": bs["S1"]}, "bank_milestones": {}}
        print(write(nm, cfg))
    else:
        nm = f"F{g}_r{a.r}_t{a.tau}_a{a.alpha}_s{a.seed}"
        cfg = {**common, "run_name": nm, "seed": a.seed, "prereg_sha256": sha, "generation": a.gen,
               "cond": {"banks": final_banks(a.r, a.tau, a.alpha), "lam_r": 1.0},
               "milestones": {"m50": bs["S2"], "p100": 3880},
               "bank_milestones": {"m100": bs["M100"], "m150": bs["M150"], "m200": bs["M200"]}}
        print(write(nm, cfg))


if __name__ == "__main__":
    main()
