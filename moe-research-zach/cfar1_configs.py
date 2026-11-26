#!/usr/bin/env python
"""csr-centered-fedavg-residual-v1 configs (TOKEN-4 model/train/budget from the reused T4 s0 config).

  cfar1_configs.py gputest                         test configs
  cfar1_configs.py stage1 [--gen G]                Stage-1 search (seed 100): G, L, F1 x T, F2 sem/shuf x T x w_p
  cfar1_configs.py final --seed S --T T --wp W     Stage-2/final: G, F2s, F2h, F1s (same T), L"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402


def t4_base():
    from csr.cfar1_paths import T4_CFG_RUN
    c = json.loads((guard.ROOT / T4_CFG_RUN / "config.json").read_text())
    return {k: c[k] for k in ("budget_tokens", "diag_every", "model", "train")}


def fed(name, prior, mode, T=0, w_p=256):
    # student router: csur1 generation-1 'meanlast' ([mean, last token] -> Linear(768->20) -> GRUCell(20) -> 16; 18,216 params)
    return {"name": name, "kind": "fed", "prior": prior, "mode": mode, "T": T, "w_p": w_p, "router": "meanlast"}


def final_banks(T, wp):
    return [{"name": "G", "kind": "token"}, fed("F2s", "sem", "centered", T, wp), fed("F2h", "shuf", "centered", T, wp),
            fed("F1s", "sem", "fedavg", T), fed("L", "sem", "local")]


def stage1_banks():
    from csr.cfar1_paths import T_GRID, WP_GRID
    out = [{"name": "G", "kind": "token"}, fed("L", "sem", "local")]
    for T in T_GRID:
        out.append(fed(f"F1s_T{T}", "sem", "fedavg", T))
        for wp in WP_GRID:
            out += [fed(f"F2s_T{T}_w{wp}", "sem", "centered", T, wp), fed(f"F2h_T{T}_w{wp}", "shuf", "centered", T, wp)]
    return out


def write(name, cfg):
    from csr.cfar1_paths import CFG_DIR
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
    ap.add_argument("--T", type=int)
    ap.add_argument("--wp", type=int)
    ap.add_argument("--gen", type=int, default=0)
    a = ap.parse_args()
    guard.check_root()
    from csr.cfar1_paths import N_STEPS, SEARCH_SEED, prereg_sha
    common = {"experiment": "csr-centered-fedavg-residual-v1", **t4_base()}
    if a.what == "gputest":
        cfg = {**common, "run_name": "gt_fed", "seed": 0, "test": True,
               "cond": {"banks": [{"name": "G", "kind": "token"}, fed("F2s", "sem", "centered", 20, 192),
                                  fed("F2h", "shuf", "centered", 20, 192), fed("F1s", "sem", "fedavg", 20),
                                  fed("L", "sem", "local")]},
               "milestones": {"t60": 60}, "bank_milestones": {}}
        print(write("gt_fed", cfg))
        cfg = {**common, "run_name": "gt_fed_nobank", "seed": 0, "test": True, "cond": {"banks": []},
               "milestones": {}, "bank_milestones": {}}
        print(write("gt_fed_nobank", cfg))
        cfg = {**common, "run_name": "gt_fed_bench", "seed": SEARCH_SEED, "test": True, "cond": {"banks": stage1_banks()},
               "milestones": {"t40": 40}, "bank_milestones": {}}
        print(write("gt_fed_bench", cfg))
        return
    sha = prereg_sha()
    tb = json.loads((guard.ROOT / "artifacts/joint-semantic-residual-pretrain-v1/tables/tables.json").read_text())
    bs = tb["budget_steps"]
    g = f"_g{a.gen}" if a.gen else ""
    if a.what == "stage1":
        nm = f"S1{g}_s{SEARCH_SEED}"
        cfg = {**common, "run_name": nm, "seed": SEARCH_SEED, "prereg_sha256": sha, "generation": a.gen,
               "cond": {"banks": stage1_banks()}, "milestones": {"m20": bs["S1"]}, "bank_milestones": {}}
    else:
        nm = f"F{g}_T{a.T}_w{a.wp}_s{a.seed}"
        cfg = {**common, "run_name": nm, "seed": a.seed, "prereg_sha256": sha, "generation": a.gen,
               "cond": {"banks": final_banks(a.T, a.wp)}, "milestones": {"m50": bs["S2"], "p100": N_STEPS},
               "bank_milestones": {"m100": bs["M100"], "m150": bs["M150"], "m200": bs["M200"]}}
    print(write(nm, cfg))


if __name__ == "__main__":
    main()
