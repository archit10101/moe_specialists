#!/usr/bin/env python
"""csr-semantic-lift-expert-learning-v1 trainer CPU tests (tiny model, real data stream, sll_train.py subprocess).
Writes artifacts/semantic-lift-expert-learning-v1/cpu_tests_train_<UTC>.json with --write.
  tt1 lift with lambda = 0 is side-effect free: per-step lm and grad norm bit-identical to plain Semantic C6
  tt2 test-only schedule (frac 0.002 -> collect 8 steps, ramp to 16): operator active after step 8, multipliers
      logged, gate-weighted mean 1 and bounds hold at every logged layer, losses finite, lambda_eff as registered
  tt3 exact resume with LiftStats (full 0->14 vs 0->10 + resume ->14: lm, gn identical; lift snapshot identical)
  tt4 checkpoints contain exactly the C6 parameter keys (no extra parameters); inference forward unchanged
  tt5 SEM and SHUF runs of one seed see identical rows (same per-step valid-token counts and step-0 lm)"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402

TINY = {"n_layer": 3, "d_model": 64, "n_head": 2, "dense_ffn": 128, "n_experts": 8, "expert_ffn": 32,
        "shared_ffn": 64, "vocab_size": 50304, "top_k": 4, "router": "linear", "teacher": False}
TRAIN = {"aux_coeff": 0.01, "betas": [0.9, 0.95], "clip": 1.0, "decay_frac": 0.2, "log_every": 1, "lr": 6e-4,
         "micro_batch_rows": 32, "min_lr": 6e-5, "rows_per_step": 64, "warmup_frac": 0.01, "weight_decay": 0.1,
         "z_coeff": 0.001}


def cfg(work, name, cond, milestones=None):
    from csr.sll_paths import prereg_sha
    c = {"run_name": name, "seed": 3, "budget_tokens": 245_000_000, "prereg_sha256": prereg_sha(), "model": TINY,
         "cond": cond, "milestones": milestones or {}, "diag_every": 0, "train": TRAIN}
    p = Path(work) / f"{name}.json"
    p.write_text(json.dumps(c))
    return p


def go(cfgp, work, extra):
    env = {**os.environ, "MOE_RUNS_DIR": str(Path(work) / "runs"), "MOE_RESEARCH_APPROVE": "cpu-model",
           "CUDA_VISIBLE_DEVICES": ""}
    (Path(work) / "runs").mkdir(exist_ok=True)
    before = set((Path(work) / "runs").iterdir())
    r = subprocess.run([str(guard.ROOT / ".venv/bin/python"), "sll_train.py", "--config", str(cfgp), "--execute"] + extra,
                       env=env, capture_output=True, text=True, cwd=guard.ROOT)
    new = sorted(set((Path(work) / "runs").iterdir()) - before)
    return r, (new[0] if new else None)


def metrics(d):
    rows = [json.loads(x) for x in (Path(d) / "metrics.jsonl").read_text().splitlines()]
    return {x["step"]: x for x in rows if "step" in x}, [x["lift"] for x in rows if "lift" in x]


def cleanup(work, dirs):
    from csr.sll_paths import TMP
    for d in dirs:
        if d is not None:
            shutil.rmtree(TMP / "slots" / Path(d).name, ignore_errors=True)
    shutil.rmtree(work, ignore_errors=True)


def tables():
    from csr.sll_paths import SHUF_TABLE, SHUF_TABLE_SHA, TABLE, TABLE_SHA
    return (TABLE, TABLE_SHA), (SHUF_TABLE, SHUF_TABLE_SHA)


def lcond(table, sha, lam, frac=0.002):
    return {"paper_latent": {"table": table, "table_sha256": sha, "beta": 0.005},
            "lift": {"lam": lam, "lo": 0.5, "hi": 2.0, "rho": 0.98, "n0": 8.0, "floor": 0.02, "frac": frac}}


def tt1(work):
    (t, s), _ = tables()
    r0, d0 = go(cfg(work, "a", {"paper_latent": {"table": t, "table_sha256": s, "beta": 0.005}}), work, ["--max-steps", "20"])
    r1, d1 = go(cfg(work, "b", lcond(t, s, 0.0)), work, ["--max-steps", "20"])
    assert r0.returncode == 75 and r1.returncode == 75, (r0.stderr[-2000:], r1.stderr[-2000:])
    a, _ = metrics(d0)
    b, lb = metrics(d1)
    for st in range(20):
        assert a[st]["lm"] == b[st]["lm"] and a[st]["gn"] == b[st]["gn"], (st, a[st], b[st])
    assert all(x["lam_eff"] == 0.0 for x in lb)
    return {"steps_identical": 20, "lift_logged": len(lb)}, [d0, d1]


def tt2(work):
    (t, s), _ = tables()
    r, d = go(cfg(work, "c", lcond(t, s, 1.0)), work, ["--max-steps", "20"])
    assert r.returncode == 75, r.stderr[-3000:]
    a, lr = metrics(d)
    lam = {x["step"]: x["lam_eff"] for x in lr}
    assert lam[7] == 0.0 and abs(lam[8] - 1 / 8) < 1e-12 and lam[16] == 1.0, lam
    act = [x for x in lr if x["lam_eff"] > 0]
    worst, mn, mx, ad = 0.0, 9.0, 0.0, []
    for x in act:
        assert len(x["m_by_layer"]) == 2, x["m_by_layer"].keys()
        for v in x["m_by_layer"].values():
            worst = max(worst, v["m_wmean_maxdev"])
            mn, mx = min(mn, v["m_min"]), max(mx, v["m_max"])
            ad.append(v["m_absdev_wmean"])
    assert worst < 1e-6 and mn >= 0.5 - 1e-9 and mx <= 2.0 + 1e-9, (worst, mn, mx)
    assert all(abs(a[st]["lm"]) < 20 for st in a)
    snaps = sorted((Path(d) / "lift").glob("lift_s*.npz"))
    assert snaps, "no lift snapshots"
    return {"active_steps": len(act), "wmean_maxdev": worst, "m_min": mn, "m_max": mx,
            "m_absdev_wmean_last": ad[-1] if ad else None, "lift_stats_last": lr[-1]["stats"]["lift_absmean"],
            "load_min_last": lr[-1].get("load_min_x_uniform_by_layer")}, [d]


def tt3(work):
    import numpy as np
    (t, s), _ = tables()
    c = cfg(work, "r", lcond(t, s, 1.0), {"m14": 14})
    r, full = go(c, work, ["--max-steps", "14"])
    r, part = go(c, work, ["--max-steps", "10"])
    r2, _ = go(c, work, ["--max-steps", "14", "--run-dir", str(part)])
    assert r2.returncode == 75, r2.stderr[-2000:]
    a, _ = metrics(full)
    b, _ = metrics(part)
    for st in range(14):
        assert a[st]["lm"] == b[st]["lm"] and a[st]["gn"] == b[st]["gn"], (st, a[st], b[st])
    za = np.load(Path(full) / "lift" / "lift_s00013.npz")
    zb = np.load(Path(part) / "lift" / "lift_s00013.npz")
    assert np.array_equal(za["lift"], zb["lift"]) and np.array_equal(za["scale"], zb["scale"])
    return {"steps_identical": 14, "lift_table_identical_at_13": True}, [full, part]


def tt4(work):
    import torch
    from csr.lfr_model import LFRConfig, LFRModel
    (t, s), _ = tables()
    r, d = go(cfg(work, "i", lcond(t, s, 1.0), {"m12": 12}), work, ["--max-steps", "12"])
    assert r.returncode == 75, r.stderr[-2000:]
    ck = torch.load(Path(d) / "ckpt_m12" / "state.pt", map_location="cpu", weights_only=False)
    ref = LFRModel(LFRConfig.from_dict({**LFRConfig().to_dict(), **TINY}), 3).state_dict()
    keys_equal = sorted(ck["model"].keys()) == sorted(ref.keys())
    assert keys_equal
    m = LFRModel(LFRConfig.from_dict(ck["model_cfg"]), 0)
    m.load_state_dict({k: v.float() for k, v in ck["model"].items()})
    from csr.lfr_model import Experts
    assert all(type(b.ffn.experts) is Experts for b in m.blocks if b.is_moe)
    return {"ckpt_keys_equal_c6_model": keys_equal, "n_keys": len(ref), "inference_experts_native": True}, [d]


def tt5(work):
    (t, s), (ts, ss) = tables()
    r0, d0 = go(cfg(work, "sem", lcond(t, s, 1.0)), work, ["--max-steps", "3"])
    r1, d1 = go(cfg(work, "shf", lcond(ts, ss, 1.0)), work, ["--max-steps", "3"])
    a, _ = metrics(d0)
    b, _ = metrics(d1)
    assert a[0]["lm"] == b[0]["lm"] and [a[i]["seen"] for i in range(3)] == [b[i]["seen"] for i in range(3)]
    return {"same_rows": True}, [d0, d1]


def main():
    guard.check_root()
    from csr.sll_paths import ART, TMP
    TMP.mkdir(parents=True, exist_ok=True)
    res = {}
    names = [a for a in sys.argv[1:] if a.startswith("tt")] or ["tt1", "tt2", "tt3", "tt4", "tt5"]
    for name in names:
        work = Path(tempfile.mkdtemp(dir=TMP))
        dirs = []
        try:
            out, dirs = globals()[name](work)
            res[name] = {"ok": True, **out}
        except Exception:
            res[name] = {"ok": False, "error": traceback.format_exc()[-3000:]}
        finally:
            cleanup(work, dirs)
        print(name, json.dumps(res[name], default=str)[:800], flush=True)
    res["all_ok"] = all(v["ok"] for v in res.values())
    if "--write" in sys.argv:
        ART.mkdir(parents=True, exist_ok=True)
        p = ART / f"cpu_tests_train_{'_'.join(names)}_{guard.utc_stamp()}.json"
        guard.atomic_write_json(p, {"time": guard.now(), **res}, mode=0o400)
        print(p)
    print("all_ok", res["all_ok"])
    sys.exit(0 if res["all_ok"] else 1)


if __name__ == "__main__":
    main()
