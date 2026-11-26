#!/usr/bin/env python
"""csr-forecast-latent-edl-v1 final stage (prereg sections 7, 9, 11). Requires frozen.json.

  fle_finals.py prep              : PAST latent fit (same K/L/tau rule), FD gamma calibration (5M, seed 100, training
                                    data), final-region cache (frozen L), assignment tables for the full FINAL plan
  fle_finals.py baselines         : T4 s1, EDL s0/s1 (independent of the search; may run early)
  fle_finals.py train --seeds 0 1 : FL, PAST, SHUF, FD (and T4/EDL where missing) at FINAL scale
  fle_finals.py eval              : VAL once per final run (audits at milestones; per-expert at p100 for T4/EDL/PAST/FL)
  fle_finals.py seed3             : apply the third-seed rule (T4, FL, PAST seed 2) and evaluate them"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402
from csr.fle_paths import ART, LATENT, T4_S0_RUN  # noqa: E402

FDIR = ART / "finals"
EDL_BETA = 5e-3


def frozen():
    return json.loads((ART / "frozen.json").read_text())


def tables(fz):
    from fle_driver import run_job
    for cid in (fz["latent_cfg"], fz["past_cfg"]):
        run_job(f"latent_fit_{cid}", ["fle_latent.py", "fit", "--cfg", cid], gpu=False)
        run_job(f"latent_apply_{cid}_final_main", ["fle_latent.py", "apply", "--cfg", cid, "--cache", "final_main"],
                gpu=False)
        run_job(f"latent_merge_{cid}", ["fle_latent.py", "merge", "--cfg", cid], gpu=False)


def cond_of(name, fz):
    from fle_configs import latent_cond
    L = int(fz["latent_cfg"].split("_")[2][1:])
    if name == "T4":
        return {}
    if name == "EDL":
        return {"edl_beta": EDL_BETA}
    if name in ("FL", "SHUF"):
        return latent_cond(fz["latent_cfg"], fz["beta"], fz["layers"], shuffle=(name == "SHUF"), cache="full")
    if name == "PAST":
        return latent_cond(fz["past_cfg"], fz["beta"], fz["layers"], cache="full")
    if name == "FD":
        g = json.loads((FDIR / "fd_gamma.json").read_text())["gamma"]
        c = latent_cond(fz["latent_cfg"], 0.0, [], cache="full")["latent"]
        return {"fd": {"cfg_id": c["cfg_id"], "table": c["table"], "table_sha256": c["table_sha256"], "gamma": g}}
    raise ValueError(name)


def eq_pass():
    p = ART / "equivalence_check.json"
    return p.exists() and json.loads(p.read_text())["pass"]


def run_dir(rn):
    from fle_search import run_dir_of
    if rn == "final_T4_s0" and eq_pass():
        return T4_S0_RUN
    return run_dir_of(rn)


def train_final(name, seed, fz):
    from fle_configs import make
    from fle_driver import run_job
    from fle_search import remember_run_dir
    rn = f"final_{name}_s{seed}"
    if rn == "final_T4_s0" and eq_pass():
        return
    p, _ = make(rn, seed, cond_of(name, fz), "final", diag_every=100)
    first = run_dir(rn) is None
    args = ["fle_train.py", "--config", str(p)] + ([] if first else ["--run-dir", run_dir(rn)])
    run_job(f"train_{rn}", args, gpu=True, disk_gb=4.0)
    remember_run_dir(rn)


def eval_final(name, seed, fz):
    from fle_driver import run_job
    rn = f"final_{name}_s{seed}"
    tags = "p040,p060,p080,p100" if name in ("T4", "EDL", "PAST", "FL") else "p040,p100"
    args = ["fle_evaluate.py", "final", "--run-dir", run_dir(rn), "--run-name", rn, "--tags", tags,
            "--latent", fz["latent_cfg"]]
    if name in ("T4", "EDL", "PAST", "FL"):
        args.append("--per-expert")
    run_job(f"eval_final_{rn}", args, gpu=True, disk_gb=0.5)


def fd_gamma(fz):
    """gamma = median ||grad beta*L_FL-EDL|| of the frozen winner's stage-1 diagnostics / median ||grad L_FD|| of a
    gamma=1 calibration run (seed 100, 5M tokens, training data)."""
    import numpy as np
    from fle_configs import latent_cond, make
    from fle_driver import run_job
    from fle_search import remember_run_dir, run_dir_of
    p = FDIR / "fd_gamma.json"
    if p.exists():
        return
    c = latent_cond(fz["latent_cfg"], 0.0, [])["latent"]
    cfg_p, _ = make("cal_FD_s100", 100, {"fd": {"cfg_id": c["cfg_id"], "table": c["table"],
                                                "table_sha256": c["table_sha256"], "gamma": 1.0}}, "search",
                    diag_every=10)
    run_job("train_cal_FD_s100", ["fle_train.py", "--config", str(cfg_p), "--max-steps", "80"], gpu=True,
            expect=(75, 0), disk_gb=3.0)
    remember_run_dir("cal_FD_s100")
    m = [json.loads(x) for x in (guard.ROOT / run_dir_of("cal_FD_s100") / "metrics.jsonl").read_text().splitlines()]
    g_fd = float(np.median([x["diag"]["gn_fd_unit"] for x in m if "diag" in x]))
    w = [json.loads(x) for x in (guard.ROOT / run_dir_of(f"s_{fz['winner']}v2_s100") / "metrics.jsonl").read_text()
         .splitlines()]
    g_fl = float(np.median([x["diag"]["gn_aux"] for x in w if "diag" in x and x["diag"]["step"] < 80]))
    gamma = float(f"{g_fl / g_fd:.2g}")
    guard.atomic_write_json(p, {"gamma": gamma, "median_gn_fledl_winner_stage1": g_fl, "median_gn_fd_unit": g_fd,
                                "time": guard.now()}, mode=0o400)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "baselines", "train", "eval", "seed3"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1])
    ap.add_argument("--conds", nargs="*", default=["FL", "PAST", "T4", "EDL", "SHUF", "FD"])
    a = ap.parse_args()
    guard.check_root()
    FDIR.mkdir(parents=True, exist_ok=True)
    from fle_driver import run_job
    if a.cmd == "baselines":
        fz = {"latent_cfg": "A1_K16_L256"}
        for name, seed in (("T4", 0), ("T4", 1), ("EDL", 0), ("EDL", 1)):
            train_final(name, seed, fz)
        return
    fz = frozen()
    if a.cmd == "prep":
        L = int(fz["latent_cfg"].split("_")[2][1:])
        run_job("cache_final_main", ["fle_cache.py", "--name", "final_main", "--L", str(L), "--batch-docs", "512"],
                gpu=True, disk_gb=12.0)
        tables(fz)
        run_job("cache_val_audit", ["fle_cache.py", "--name", "val_audit", "--L", str(L)], gpu=True, disk_gb=1.0)
        run_job(f"latent_apply_{fz['latent_cfg']}_val_audit", ["fle_latent.py", "apply", "--cfg", fz["latent_cfg"],
                                                                "--cache", "val_audit"], gpu=False)
        fd_gamma(fz)
    elif a.cmd == "train":
        for seed in a.seeds:
            for name in a.conds:
                train_final(name, seed, fz)
    elif a.cmd == "eval":
        for seed in a.seeds:
            for name in a.conds:
                eval_final(name, seed, fz)
    elif a.cmd == "seed3":
        fa = json.loads((ART / "final_analysis.json").read_text())
        trig = fa["criteria"].get("third_seed_triggered")
        rec = {"time": guard.now(), "triggered": trig, "criteria": fa["criteria"], "conds": ["T4", "FL", "PAST"]}
        p = FDIR / "third_seed_decision.json"
        if not p.exists():
            guard.atomic_write_json(p, rec, mode=0o400)
        if json.loads(p.read_text())["triggered"]:
            for name in ("T4", "FL", "PAST"):
                train_final(name, 2, fz)
            for name in ("T4", "FL", "PAST"):
                eval_final(name, 2, fz)


if __name__ == "__main__":
    main()
