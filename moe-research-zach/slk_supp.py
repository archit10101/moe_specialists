#!/usr/bin/env python
"""Supplementary check (DECISION_LOG 2026-11-15): does the collect / beta = 0 paper_latent code path change TOKEN-4
training beyond run-to-run GPU noise? beta0 (C6 table, beta 0) vs two TOKEN-4 replicates, seed 0, 318 steps (20M
tokens), identical FINAL plan. Child training runs in .tmp/semantic-edl-long-k/supp_runs. Output supp_beta0.json.
Run via: slk_driver.py job supp_beta0 --gpu -- slk_supp.py"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402

PY = str(guard.ROOT / ".venv/bin/python")


def main():
    guard.check_root()
    if "--execute" not in sys.argv:
        return
    from slk_configs import make, paper_cond
    from slk_gpu_checks import metrics
    from csr.slk_paths import ART, C6_TABLE, C6_TABLE_SHA, TMP, check_prereg
    check_prereg()
    RD = TMP / "supp_runs"
    RD.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "MOE_RUNS_DIR": str(RD)}
    t0 = time.time()
    out, dirs = {"time": guard.now()}, {}
    for tag, cond in (("T4a", {}), ("T4b", {}), ("beta0", paper_cond(C6_TABLE, C6_TABLE_SHA, 0.0))):
        p, _ = make(f"supp_{tag}_s0", 0, cond, "calibration", diag_every=100)
        before = set(RD.iterdir())
        r = subprocess.run([PY, "slk_train.py", "--config", str(p), "--device", "cuda", "--execute", "--max-steps",
                            "318"], env=env, capture_output=True, text=True, cwd=guard.ROOT)
        d = sorted(set(RD.iterdir()) - before)[0]
        dirs[tag] = d
        out[f"rc_{tag}"] = r.returncode
        if r.returncode == 77:
            out["status"] = "EXTERNALLY_STOPPED"
            break
    if out.get("status") != "EXTERNALLY_STOPPED":
        m = {t: metrics(d) for t, d in dirs.items()}
        steps = sorted(set(m["T4a"][0]) & set(m["T4b"][0]) & set(m["beta0"][0]))
        dsteps = sorted(set(m["T4a"][1]) & set(m["T4b"][1]) & set(m["beta0"][1]))
        out["lm"] = {s: {t: m[t][0][s]["lm"] for t in m} for s in steps}
        out["diag_val_lm"] = {s: {t: m[t][1][s]["val_lm"] for t in m} for s in dsteps}
        late = [s for s in steps if s >= 150]
        out["replicate_spread_late"] = max(abs(out["lm"][s]["T4a"] - out["lm"][s]["T4b"]) for s in late)
        out["beta0_vs_T4_late"] = max(max(abs(out["lm"][s]["beta0"] - out["lm"][s]["T4a"]),
                                          abs(out["lm"][s]["beta0"] - out["lm"][s]["T4b"])) for s in late)
        out["mean_late"] = {t: sum(out["lm"][s][t] for s in late) / len(late) for t in m}
        out["status"] = "COMPLETE"
    out["seconds"] = time.time() - t0
    out["run_dirs"] = {k: str(v.relative_to(guard.ROOT)) for k, v in dirs.items()}
    guard.atomic_write_json(ART / "supp_beta0.json", out, overwrite=True)
    for d in dirs.values():
        sl = TMP / "slots" / d.name
        if sl.exists():
            shutil.rmtree(sl)
            with open(ART / "deleted_artifacts.tsv", "a") as f:
                f.write(f"{guard.now()}\t{sl.relative_to(guard.ROOT)}\tsupplementary check slot (disposable)\n")
    print(json.dumps(out, indent=1, default=str)[:3000])
    sys.exit(0 if out["status"] == "COMPLETE" else 77)


if __name__ == "__main__":
    main()
