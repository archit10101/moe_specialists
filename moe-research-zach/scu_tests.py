#!/usr/bin/env python
"""csr-soft-cf-usefulness-v2 CPU tests. Unit tests (in process) + trainer tests (tiny model, real data stream,
scu_train.py subprocess). Test run directories are KEPT under .tmp/soft-cf-usefulness-v2/tests/<stamp>/ (operator
rule: no deletion). With --write: artifacts/soft-cf-usefulness-v2/cpu_tests_<UTC>.json (0400).
  u1 confidence scores: |DeltaU| within z*SE -> 0; unsupported -> 0; sign kept; mean estimator = DeltaU / scale
  u2 soft target: tilt with s = 0 -> q = prior; uniform form with s = 0 -> uniform; rows sum to 1; evidence mask
  u3 cf_loss: KL >= 0, zero when P = q; one gradient step on router logits moves group means toward q
  tt1 soft_cf with gamma = 0 and beta = 0 is side-effect free: lm and grad-norm bit-identical to TOKEN-4
  tt2 gamma > 0: L_CF active after onset; first refresh lagged; mechanism snapshot written at diag steps
  tt3 exact resume with bank + prior + accumulators (lm, L_CF identical; mechanism snapshot identical)
  tt4 no extra parameters; inference forward never opens a table / bank / slot
  tt5 SEM and SHUF runs of one seed make identical probe observation counts
  tt6 uniform-prior form and mean estimator code paths run (finite, active)"""
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402

TINY = {"n_layer": 3, "d_model": 64, "n_head": 2, "dense_ffn": 128, "n_experts": 8, "expert_ffn": 32,
        "shared_ffn": 64, "vocab_size": 50304, "top_k": 4, "router": "linear", "teacher": False}
TRAIN = {"aux_coeff": 0.01, "betas": [0.9, 0.95], "clip": 1.0, "decay_frac": 0.2, "log_every": 1, "lr": 6e-4,
         "micro_batch_rows": 32, "min_lr": 6e-5, "rows_per_step": 64, "warmup_frac": 0.01, "weight_decay": 0.1,
         "z_coeff": 0.001}


def u1():
    from csr.scu_cf import confidence_scores
    K, L, E = 3, 2, 8
    rng = np.random.default_rng(0)
    d = rng.normal(0, 1, (K, L, E))
    se = np.full((K, L, E), 0.5)
    v = np.ones((K, L, E), bool)
    v[0, 0, 0] = False
    s, sc = confidence_scores(d, se, v, 1.0, "lcb")
    dz, sz = d / sc[None, :, None], se / sc[None, :, None]
    inside = (np.abs(dz) <= sz) & v
    assert np.all(s[inside] == 0) and s[0, 0, 0] == 0
    out = (~inside) & v
    assert np.all(np.sign(s[out]) == np.sign(d[out]))
    assert np.all(np.abs(s[out]) < np.abs(np.clip(dz[out], -3, 3)) + 1e-12)
    m, _ = confidence_scores(d, se, v, 1.0, "mean")
    assert np.allclose(m[v], np.clip(dz[v], -3, 3)) and m[0, 0, 0] == 0
    return {"zeroed_inside_ci": int(inside.sum()), "n_cells": int(v.sum())}


def u2():
    from csr.scu_cf import soft_target
    K, L, E = 3, 2, 8
    rng = np.random.default_rng(1)
    R = rng.dirichlet(np.ones(E), (K, L))
    s = np.zeros((K, L, E))
    s[1, 0, 3] = 2.0
    q, ev = soft_target(s, R, 1.0, "tilt")
    assert np.allclose(q[0], R[0]) and np.allclose(q.sum(-1), 1) and ev[1, 0] and not ev[0].any()
    assert q[1, 0, 3] > R[1, 0, 3]
    qu, evu = soft_target(s, R, 1.0, "uniform")
    assert np.allclose(qu[0], 1.0 / E) and (evu == ev).all()
    return {"tilt_gain": float(q[1, 0, 3] / R[1, 0, 3])}


def u3():
    import torch
    from csr.scu_cf import cf_loss
    torch.manual_seed(0)
    N, E, S, K = 64, 8, 4, 3
    logits = torch.randn(N, E, requires_grad=True)
    seq_id = torch.arange(N) % S
    qseq = torch.tensor([[1., 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])
    nonpad = torch.ones(N, dtype=torch.bool)
    Q = torch.softmax(torch.randn(K, 1, E) * 2, -1)
    EV = torch.ones(K, 1, dtype=torch.bool)
    p = torch.softmax(logits, -1)
    l0, n, kl, _ = cf_loss([p], [0], qseq, seq_id, nonpad, Q, EV)
    assert float(l0) >= 0 and n == K
    l0.backward()
    with torch.no_grad():
        lg2 = logits - 5.0 * logits.grad
    l1, _, _, _ = cf_loss([torch.softmax(lg2, -1)], [0], qseq, seq_id, nonpad, Q, EV)
    assert float(l1) < float(l0), (float(l0), float(l1))
    return {"kl_before": float(l0), "kl_after_step": float(l1)}


def cfg(work, name, cond, milestones=None, diag_every=0):
    from csr.scu_paths import prereg_sha
    c = {"run_name": name, "seed": 3, "budget_tokens": 245_000_000, "prereg_sha256": prereg_sha(), "model": TINY,
         "cond": cond, "milestones": milestones or {}, "diag_every": diag_every, "train": TRAIN}
    p = Path(work) / f"{name}.json"
    p.write_text(json.dumps(c))
    return p


def go(cfgp, work, extra):
    env = {**os.environ, "MOE_RUNS_DIR": str(Path(work) / "runs"), "MOE_RESEARCH_APPROVE": "cpu-model",
           "CUDA_VISIBLE_DEVICES": ""}
    (Path(work) / "runs").mkdir(exist_ok=True)
    before = set((Path(work) / "runs").iterdir())
    r = subprocess.run([str(guard.ROOT / ".venv/bin/python"), "scu_train.py", "--config", str(cfgp), "--execute"] + extra,
                       env=env, capture_output=True, text=True, cwd=guard.ROOT)
    new = sorted(set((Path(work) / "runs").iterdir()) - before)
    return r, (new[0] if new else None)


def metrics(d):
    rows = [json.loads(x) for x in (Path(d) / "metrics.jsonl").read_text().splitlines()]
    return {x["step"]: x for x in rows if "step" in x}, [x["util_diag"] for x in rows if "util_diag" in x]


def scond(table, sha, gamma, **over):
    u = {"probe_rows": 2, "probe_passes": 2, "refresh_every": 1, "m_min": 0.01, "n_min": 1, "clip": 3.0,
         "min_len": 256, "ramp_steps": 0, "gamma": gamma, "tau": 1.0, "rho": 0.998, "onset_step": 1,
         "form": "tilt", "estimator": "lcb", "z_lcb": 1.0, "prior_rho": 0.98}
    u.update(over)
    return {"paper_latent": {"table": table, "table_sha256": sha, "beta": 0.0}, "soft_cf": u}


def tables():
    from csr.scu_paths import SHUF_TABLE, SHUF_TABLE_SHA, TABLE, TABLE_SHA
    return (TABLE, TABLE_SHA), (SHUF_TABLE, SHUF_TABLE_SHA)


def tt1(work):
    (t, s), _ = tables()
    r0, d0 = go(cfg(work, "a", {}), work, ["--max-steps", "4"])
    r1, d1 = go(cfg(work, "b", scond(t, s, 0.0)), work, ["--max-steps", "4"])
    assert r0.returncode == 75 and r1.returncode == 75, (r0.stderr[-1500:], r1.stderr[-1500:])
    m0, _ = metrics(d0)
    m1, dg = metrics(d1)
    for st in range(4):
        assert m0[st]["lm"] == m1[st]["lm"] and m0[st]["gn"] == m1[st]["gn"], (st, m0[st], m1[st])
    assert "cf_s" in m1[0] and len(dg) >= 2
    return {"steps_identical_to_token4": 4, "n_refresh": len(dg)}


def tt2(work):
    (t, s), _ = tables()
    r, d = go(cfg(work, "c", scond(t, s, 1.0), diag_every=2), work, ["--max-steps", "5"])
    assert r.returncode == 75, r.stderr[-2000:]
    m, dg = metrics(d)
    assert "use" not in m[0]
    act = [m[st].get("use_active", 0) for st in range(1, 5)]
    assert any(a > 0 for a in act), (act, dg[-1].get("target"))
    assert dg[0]["step"] == 1 and dg[0]["n_obs"] == 2 * 2
    mf = sorted((Path(d) / "mech").glob("mech_s*.npz"))
    z = np.load(mf[-1])
    assert len(mf) >= 2 and np.allclose(z["R"].sum(-1), 1) and np.isfinite(z["G"]).all() and z["G_mass"].sum() > 0
    return {"use_active": act, "use": [m[st].get("use") for st in range(1, 5)], "first_refresh": dg[0]["n_obs"],
            "mech_files": [x.name for x in mf], "target": dg[-1].get("target")}


def tt3(work):
    (t, s), _ = tables()
    c = cfg(work, "r", scond(t, s, 1.0), {"m4": 4}, diag_every=2)
    r, full = go(c, work, ["--max-steps", "5"])
    r, part = go(c, work, ["--max-steps", "3"])
    r2, _ = go(c, work, ["--max-steps", "5", "--run-dir", str(part)])
    assert r2.returncode == 75, r2.stderr[-2000:]
    a, _ = metrics(full)
    b, _ = metrics(part)
    for st in range(5):
        assert a[st]["lm"] == b[st]["lm"] and a[st].get("use") == b[st].get("use"), (st, a[st], b[st])
    za, zb = np.load(Path(full) / "mech" / "mech_s00004.npz"), np.load(Path(part) / "mech" / "mech_s00004.npz")
    for k in ("R", "X", "G", "d", "Q"):
        assert np.array_equal(za[k], zb[k]), k
    return {"steps_identical": 5, "mech_identical_after_resume": True}


def tt4(work):
    import torch
    from csr.lfr_model import LFRConfig, LFRModel
    (t, s), _ = tables()
    r, d = go(cfg(work, "i", scond(t, s, 1.0), {"m3": 3}), work, ["--max-steps", "4"])
    ck = torch.load(Path(d) / "ckpt_m3" / "state.pt", map_location="cpu", weights_only=False)
    same = set(ck["model"]) == set(LFRModel(LFRConfig.from_dict(TINY), 0).state_dict())
    code = ("import sys,torch; sys.path.insert(0,'src')\n"
            "def hook(e,a):\n"
            "    if e=='open' and a and any(s in str(a[0]) for s in ('gc-edl-v1','semantic-edl-v1/tables','slots','mech')):\n"
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
    return {"no_extra_params": same, "inference_without_tables": True}


def tt5(work):
    (t, s), (ts, ss) = tables()
    r0, d0 = go(cfg(work, "sem", scond(t, s, 1.0)), work, ["--max-steps", "3"])
    r1, d1 = go(cfg(work, "shf", scond(ts, ss, 1.0)), work, ["--max-steps", "3"])
    _, g0 = metrics(d0)
    _, g1 = metrics(d1)
    n0, n1 = [x["n_obs"] for x in g0], [x["n_obs"] for x in g1]
    assert n0 == n1, (n0, n1)
    return {"n_obs_sem": n0, "n_obs_shuf": n1}


def tt6(work):
    (t, s), _ = tables()
    out = {}
    for nm, over in (("uni", {"form": "uniform"}), ("mean", {"estimator": "mean"})):
        r, d = go(cfg(work, nm, scond(t, s, 1.0, **over)), work, ["--max-steps", "6"])
        assert r.returncode == 75, r.stderr[-2000:]
        m, _ = metrics(d)
        act = [m[st].get("use_active", 0) for st in range(1, 6)]
        assert any(a > 0 for a in act) and all(np.isfinite(m[st]["lm"]) for st in range(6)), act
        out[nm] = act
    return out


def main():
    guard.check_root()
    from csr.scu_paths import ART, TMP
    root = TMP / "tests" / guard.utc_stamp()
    root.mkdir(parents=True, exist_ok=True)
    res = {}
    only = [x for x in sys.argv[1:] if not x.startswith("--")]
    for name in only or ("u1", "u2", "u3", "tt1", "tt2", "tt3", "tt4", "tt5", "tt6"):
        work = root / name
        work.mkdir()
        try:
            out = globals()[name](work) if name.startswith("tt") else globals()[name]()
            res[name] = {"ok": True, **out}
        except Exception:
            res[name] = {"ok": False, "error": traceback.format_exc()[-3000:]}
        print(name, json.dumps(res[name], default=str)[:700], flush=True)
    res["all_ok"] = all(v["ok"] for v in res.values())
    res["kept_test_dirs"] = str(root.relative_to(guard.ROOT))
    if "--write" in sys.argv:
        ART.mkdir(parents=True, exist_ok=True)
        p = ART / f"cpu_tests_{guard.utc_stamp()}.json"
        guard.atomic_write_json(p, {"time": guard.now(), **res}, mode=0o400)
        print(p)
    print("all_ok", res["all_ok"])
    sys.exit(0 if res["all_ok"] else 1)


if __name__ == "__main__":
    main()
