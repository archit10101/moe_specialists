#!/usr/bin/env python
"""csr-semantic-utility-edl-v1 trainer CPU tests (tiny model, real data stream, sue_train.py subprocess). Writes
artifacts/semantic-utility-edl-v1/cpu_tests_train_<UTC>.json.
  tt1 utility machinery with gamma = 0 is side-effect free: lm and grad-norm bit-identical to the Semantic-EDL run
  tt2 gamma > 0: L_use active after onset (targets lagged: first refresh uses only earlier steps), finite losses
  tt3 exact resume with the target bank (lm, use identical, full vs 2 steps + resume)
  tt4 no extra parameters in checkpoints; inference forward never opens a table / bank
  tt5 SEM and SHUF runs of one seed make identical probe observation counts (same docs/layers/passes)"""
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
    from csr.sue_paths import prereg_sha
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
    r = subprocess.run([str(guard.ROOT / ".venv/bin/python"), "sue_train.py", "--config", str(cfgp), "--execute"] + extra,
                       env=env, capture_output=True, text=True, cwd=guard.ROOT)
    new = sorted(set((Path(work) / "runs").iterdir()) - before)
    return r, (new[0] if new else None)


def metrics(d):
    rows = [json.loads(x) for x in (Path(d) / "metrics.jsonl").read_text().splitlines()]
    return {x["step"]: x for x in rows if "step" in x}, [x["util_diag"] for x in rows if "util_diag" in x]


def cleanup(work, dirs):
    from csr.sue_paths import TMP
    for d in dirs:
        if d is not None:
            shutil.rmtree(TMP / "slots" / Path(d).name, ignore_errors=True)
    shutil.rmtree(work, ignore_errors=True)


def ucond(table, sha, gamma, **over):
    u = {"probe_rows": 2, "probe_passes": 2, "refresh_every": 1, "m_min": 0.01, "n_min": 1, "clip": 3.0,
         "min_cov": 0.0, "min_len": 256, "ramp_steps": 0, "gamma": gamma, "tau": 0.2, "rho": 0.995, "onset_step": 1}
    u.update(over)
    return {"paper_latent": {"table": table, "table_sha256": sha, "beta": 0.005}, "utility": u}


def tables():
    from csr.sue_paths import C6_SHUF_TABLE, C6_SHUF_TABLE_SHA, C6_TABLE, C6_TABLE_SHA
    return (C6_TABLE, C6_TABLE_SHA), (C6_SHUF_TABLE, C6_SHUF_TABLE_SHA)


def tt1(work):
    (t, s), _ = tables()
    r0, d0 = go(cfg(work, "a", {"paper_latent": {"table": t, "table_sha256": s, "beta": 0.005}}), work, ["--max-steps", "4"])
    r1, d1 = go(cfg(work, "b", ucond(t, s, 0.0)), work, ["--max-steps", "4"])
    assert r0.returncode == 75 and r1.returncode == 75, (r0.stderr[-1500:], r1.stderr[-1500:])
    m0, _ = metrics(d0)
    m1, dg = metrics(d1)
    for st in range(4):
        assert m0[st]["lm"] == m1[st]["lm"] and m0[st]["gn"] == m1[st]["gn"], (st, m0[st], m1[st])
    assert "cf_s" in m1[0] and "cf_u_mean" in m1[0] and len(dg) >= 2
    return {"steps_identical": 4, "cf_s_step0": m1[0]["cf_s"], "n_refresh": len(dg)}, [d0, d1]


def tt2(work):
    (t, s), _ = tables()
    r, d = go(cfg(work, "c", ucond(t, s, 1.0)), work, ["--max-steps", "5"])
    assert r.returncode == 75, r.stderr[-2000:]
    m, dg = metrics(d)
    assert "use" not in m[0]  # onset 1
    act = [m[st].get("use_active", 0) for st in range(1, 5)]
    assert any(a > 0 for a in act), act
    assert dg[0]["step"] == 1 and dg[0]["n_obs"] == 2 * 2  # first refresh: only step-0 observations (lagged)
    return {"use_active": act, "use": [m[st].get("use") for st in range(1, 5)], "first_refresh": dg[0]["n_obs"]}, [d]


def tt3(work):
    (t, s), _ = tables()
    c = cfg(work, "r", ucond(t, s, 1.0), {"m4": 4})
    r, full = go(c, work, ["--max-steps", "5"])
    r, part = go(c, work, ["--max-steps", "2"])
    r2, _ = go(c, work, ["--max-steps", "5", "--run-dir", str(part)])
    assert r2.returncode == 75, r2.stderr[-2000:]
    a, _ = metrics(full)
    b, _ = metrics(part)
    for st in range(5):
        assert a[st]["lm"] == b[st]["lm"] and a[st].get("use") == b[st].get("use"), (st, a[st], b[st])
    return {"steps_identical": 5}, [full, part]


def tt4(work):
    import torch
    from csr.lfr_model import LFRConfig, LFRModel
    (t, s), _ = tables()
    r, d = go(cfg(work, "i", ucond(t, s, 1.0), {"m3": 3}), work, ["--max-steps", "4"])
    ck = torch.load(Path(d) / "ckpt_m3" / "state.pt", map_location="cpu", weights_only=False)
    same = set(ck["model"]) == set(LFRModel(LFRConfig.from_dict(TINY), 0).state_dict())
    code = ("import sys,torch; sys.path.insert(0,'src')\n"
            "def hook(e,a):\n"
            "    if e=='open' and a and any(s in str(a[0]) for s in ('gc-edl-v1','semantic-edl-v1/tables','slots')):\n"
            "        raise PermissionError(str(a[0]))\n"
            "from csr.lfr_model import LFRConfig, LFRModel\n"
            f"ck=torch.load('{d}/ckpt_m3/state.pt',map_location='cpu',weights_only=False)\n"
            "sys.addaudithook(hook)\n"
            "m=LFRModel(LFRConfig.from_dict({**ck['model_cfg'],'teacher':False}),0); m.load_state_dict(ck['model'])\n"
            "ids=torch.randint(0,50304,(1,32)); o=m({'input_ids':ids,'doc_ids':torch.zeros_like(ids),"
            "'positions':torch.arange(32)[None]}); print('OK')\n")
    rr = subprocess.run([str(guard.ROOT / ".venv/bin/python"), "-c", code], capture_output=True, text=True,
                        cwd=guard.ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    assert same and "OK" in rr.stdout, (same, rr.stderr[-1500:])
    return {"no_extra_params": same, "inference_without_tables": True}, [d]


def tt5(work):
    (t, s), (ts, ss) = tables()
    r0, d0 = go(cfg(work, "sem", ucond(t, s, 1.0)), work, ["--max-steps", "3"])
    r1, d1 = go(cfg(work, "shf", ucond(ts, ss, 1.0)), work, ["--max-steps", "3"])
    _, g0 = metrics(d0)
    _, g1 = metrics(d1)
    n0, n1 = [x["n_obs"] for x in g0], [x["n_obs"] for x in g1]
    assert n0 == n1, (n0, n1)
    return {"n_obs_sem": n0, "n_obs_shuf": n1}, [d0, d1]


def main():
    guard.check_root()
    from csr.sue_paths import ART, TMP
    TMP.mkdir(parents=True, exist_ok=True)
    res = {}
    for name in ("tt1", "tt2", "tt3", "tt4", "tt5"):
        work = Path(tempfile.mkdtemp(dir=TMP))
        dirs = []
        try:
            out, dirs = globals()[name](work)
            res[name] = {"ok": True, **out}
        except Exception:
            res[name] = {"ok": False, "error": traceback.format_exc()[-3000:]}
        finally:
            cleanup(work, dirs)
        print(name, json.dumps(res[name], default=str)[:600], flush=True)
    res["all_ok"] = all(v["ok"] for v in res.values())
    if "--write" in sys.argv:
        ART.mkdir(parents=True, exist_ok=True)
        p = ART / f"cpu_tests_train_{guard.utc_stamp()}.json"
        guard.atomic_write_json(p, {"time": guard.now(), **res}, mode=0o400)
        print(p)
    print("all_ok", res["all_ok"])
    sys.exit(0 if res["all_ok"] else 1)


if __name__ == "__main__":
    main()
