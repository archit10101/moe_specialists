#!/usr/bin/env python
"""csr-current-context-specialist-router-v1 CPU correctness tests (prereg section 7). Real frozen base + banks.
  T1  causality end-to-end: changing tokens after position t (incl. later tokens of the same 256-token region and of
      later regions) leaves router logits at positions <= t unchanged (tokens -> frozen base -> router)
  T2  cached/sequential == batched: router.step token by token reproduces router.forward (all feature variants,
      16 and 17 actions): identical argmax, max |logit diff| < 1e-4
  T3  one-token decoding == batched: base re-run on each prefix x_<=t + router.step gives the same specialist as the
      batched pipeline at every position
  T4  teacher information absent at inference: deployed choices are unchanged when the utilities, region priors and
      targets are replaced by random values (only the CE lookup changes)
  T5  legality of the current hidden state: table target at t equals the document token at t+1; h_t is computed from
      x_<=t only (T1) and predicts x_{t+1}
  T6  expert selection: CE by table lookup == CE with the chosen specialist's correction actually applied (random choices)
  T7  equal compute / parameters: router params <= cap, identical for R1/R2/R3 at a width; one specialist per token
      (same active expert FLOPs as G); FLOP accounting
  T8  resume / determinism: router training is bitwise reproducible; a job interrupted after one router and rerun yields
      outputs identical to an uninterrupted job
  T9  paper-disjoint partitions (TRAIN vs SEL; CALIB-SELECTION vs CALIB-SEARCH)
  T10 teacher targets: R1 == region prior; R3(eta=0) == R2; argmax R2 == utility argmax; zero action handled
Writes ART/cpu_tests_<ts>.json (0400). Exit 0 iff all pass."""
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402

PY = str(guard.ROOT / ".venv/bin/python")
TMP = guard.ROOT / ".tmp/csr-current-context-specialist-router-v1/tests"
_M = {}


def model():
    if "m" not in _M:
        import torch
        from csr.ccsr1_paths import CKPT
        from fle_evaluate import load_milestone
        m, ck = load_milestone(CKPT, torch.device("cpu"))
        for p in m.parameters():
            p.requires_grad_(False)
        _M["m"], _M["ck"] = m, ck
    return _M["m"], _M["ck"]


def a_doc(min_len=600):
    import numpy as np
    from ccsr1_extract import partition
    sp, rows, *_ = partition("TRAIN")
    for r in rows:
        for d in r:
            if sp.lengths[d] >= min_len:
                return sp, d
    raise RuntimeError("no long doc")


def hidden_of(ids):
    """rms0(h) (L, 384) for ONE document given its token ids (single row, doc-masked causal base, fp32 CPU)."""
    import numpy as np
    import torch
    from csr.jsr_bank import rms0
    m, _ = model()
    L = len(ids)
    from csr.orv1_lib import T_SEQ, PAD_ID
    a = {"input_ids": np.full((1, T_SEQ), PAD_ID), "doc_ids": np.full((1, T_SEQ), -1), "positions": np.zeros((1, T_SEQ), np.int64)}
    a["input_ids"][0, :L] = ids
    a["doc_ids"][0, :L] = 0
    a["positions"][0, :L] = np.arange(L)
    b = {k: torch.as_tensor(v) for k, v in a.items()}
    with torch.no_grad():
        o = m(b)
    return rms0(o["hidden"][0, :L])


def t1_causality():
    import numpy as np
    import torch
    from csr.ccsr1_router import CausalRouter
    sp, d = a_doc(600)
    ids = np.asarray(sp.doc(d))[:600].astype(np.int64)
    r = CausalRouter(64, 0, "base")
    ok = True
    rng = np.random.default_rng(0)
    x0 = hidden_of(ids)
    with torch.no_grad():
        q0 = r(x0.unsqueeze(0), None)[0]
    checks = []
    for t in (0, 5, 130, 300, 383, 510):           # 130: mid view; 300/383: mid region; 510: region 1 end
        ids2 = ids.copy()
        ids2[t + 1:] = rng.integers(0, 50000, len(ids2) - t - 1)
        x1 = hidden_of(ids2)
        with torch.no_grad():
            q1 = r(x1.unsqueeze(0), None)[0]
        same = float((q1[:t + 1] - q0[:t + 1]).abs().max())
        changed_after = float((q1[t + 1:] - q0[t + 1:]).abs().max())
        checks.append({"t": t, "max_diff_le_t": same, "max_diff_after": changed_after})
        ok &= same < 1e-5 and changed_after > 1e-4
    return {"T1": bool(ok), "checks": checks}


def t2_cached_vs_batched():
    import torch
    from csr.ccsr1_router import CausalRouter
    torch.manual_seed(0)
    x = torch.randn(300, 384)
    res, ok = [], True
    for feats in ("base", "ctx", "chg"):
        for n_out in (16, 17):
            r = CausalRouter(32, 1, feats, n_out)
            with torch.no_grad():
                full = r(x.unsqueeze(0), None)[0]
            cache, seq = {}, []
            for t in range(x.shape[0]):
                lg, cache = r.step(x[t], cache)
                seq.append(lg)
            seq = torch.stack(seq)
            md = float((seq - full).abs().max())
            am = bool((seq.argmax(1) == full.argmax(1)).all())
            res.append({"feats": feats, "n_out": n_out, "max_abs": md, "argmax_equal": am})
            ok &= md < 1e-4 and am
    return {"T2": bool(ok), "detail": res}


def t3_one_token_decoding():
    import numpy as np
    import torch
    from csr.ccsr1_router import CausalRouter
    sp, d = a_doc(80)
    ids = np.asarray(sp.doc(d))[:48].astype(np.int64)
    r = CausalRouter(64, 2, "base")
    with torch.no_grad():
        full = r(hidden_of(ids).unsqueeze(0), None)[0].argmax(1)
    cache, ch = {}, []
    for t in range(len(ids)):
        xt = hidden_of(ids[:t + 1])[-1]          # base re-run on the prefix only
        lg, cache = r.step(xt, cache)
        ch.append(int(lg.argmax()))
    eq = bool((torch.as_tensor(ch) == full).all())
    return {"T3": eq, "n_positions": len(ids)}


def small_tables():
    """2-row TRAIN table (built once by the CPU extraction)."""
    p = TMP / "x_train2.npz"
    if not p.exists():
        env = {**os.environ, "MOE_RESEARCH_APPROVE": "cpu-model"}
        r = subprocess.run([PY, "ccsr1_extract.py", "--part", "TRAIN", "--rows-limit", "2", "--micro-rows", "2", "--device",
                            "cpu", "--out", str(p.relative_to(guard.ROOT)), "--execute"], env=env, capture_output=True, text=True,
                           cwd=guard.ROOT)
        if r.returncode != 0:
            raise RuntimeError(r.stdout[-2000:] + r.stderr[-2000:])
    return p


def t4_teacher_absent():
    import numpy as np
    import torch
    import ccsr1_router_job as J
    from csr.ccsr1_router import CausalRouter
    p = small_tables()
    m, _ = model()
    T = J.load_part("TRAIN", str(p), m, torch.device("cpu"), rows_limit=2)
    r = CausalRouter(32, 0, "base")
    s1, a1 = J.evaluate_router(r, "SEM", T, torch.device("cpu"), "TRAIN")
    T2 = dict(T)
    g = torch.Generator().manual_seed(5)
    T2["d_SEM"] = torch.randn(T["d_SEM"].shape, generator=g).half()
    T2["P"] = {k: torch.rand(v.shape, generator=g) for k, v in T["P"].items()}
    s2, a2 = J.evaluate_router(r, "SEM", T2, torch.device("cpu"), "TRAIN")
    same = bool(np.array_equal(a1["e"], a2["e"]))
    return {"T4": same and s1["ce"] != s2["ce"], "choices_identical": same}


def t5_target_alignment():
    import numpy as np
    from ccsr1_extract import partition
    z = np.load(small_tables())
    sp, rows, *_ = partition("TRAIN")
    ok = True
    for i in range(0, len(z["tgt"]), 97):
        ids = sp.doc(int(z["doc"][i]))
        ok &= int(ids[int(z["pos"][i]) + 1]) == int(z["tgt"][i])
    return {"T5": bool(ok)}


def t6_expert_selection():
    import numpy as np
    import torch
    from ccsr1_extract import load_banks, partition
    from csr.csur1_bank import ce_pair
    from csr.ccsr1_lib import hidden_table
    from csr.jsr_bank import rms0
    from csr.orv1_lib import make_batch, to_dev
    m, ck = model()
    banks = load_banks(ck, torch.device("cpu"))
    z = np.load(small_tables())
    sp, rows, *_ = partition("TRAIN")
    bt = to_dev(make_batch(rows[:2], sp), torch.device("cpu"))
    with torch.no_grad():
        o = m({"input_ids": bt["input_ids"], "doc_ids": bt["doc_ids"], "positions": bt["positions"]})
    dm = bt["doc_ids"].reshape(-1) >= 0
    H = o["hidden"].reshape(-1, 384)[dm]
    Y = bt["targets"].reshape(-1)[dm]
    sc = Y >= 0
    hs, ys = H[sc], Y[sc]
    W = m.tok_emb.weight
    g = torch.Generator().manual_seed(3)
    e = torch.randint(0, 16, (hs.shape[0],), generator=g)
    res = {}
    for X in ("SEM", "SHUF"):
        b = banks[X]
        with torch.no_grad():
            cb, cx = ce_pair(hs, b.correction(rms0(hs), e), W, ys)
        look = cb.numpy().astype(np.float64) + z[f"d_{X}"].astype(np.float64)[np.arange(len(e)), e.numpy()]
        res[X] = float(np.abs(look - cx.numpy()).max())
    return {"T6": all(v < 2e-3 for v in res.values()), "max_abs": res, "note": "table d stored in fp16"}


def t7_equal_compute():
    from csr.ccsr1_paths import ROUTER_PARAM_CAP
    from csr.ccsr1_router import CausalRouter
    _, ck = model()
    n = {H: CausalRouter(H, 0, "base").n_params() for H in (32, 64)}
    n_eng = {f: CausalRouter(64, 0, f, 17).n_params() for f in ("base", "ctx", "chg")}
    spec_flops = 2 * 384 * 256 * 2              # one 384->256->384 specialist per token (MACs x2)
    router_flops = {H: 2 * (768 * H + 6 * H * H) + 2 * H * 16 for H in (32, 64)}
    g_router_flops = 2 * 384 * 16
    ok = all(v <= ROUTER_PARAM_CAP for v in list(n.values()) + list(n_eng.values()))
    return {"T7": bool(ok), "router_params": n, "router_params_engineering": n_eng, "cap": ROUTER_PARAM_CAP,
            "flops_per_token": {"specialist": spec_flops, "router": router_flops, "G_router": g_router_flops,
                                "base_active_params": 43054464},
            "note": "candidate and G both run exactly one 384->256->384 specialist per token"}


def run_job(cfg, out_root, extra=()):
    env = {**os.environ, "MOE_RESEARCH_APPROVE": "cpu-model"}
    p = str(small_tables().relative_to(guard.ROOT))
    cmd = [PY, "ccsr1_router_job.py", "--stage", "T", "--configs", str(cfg), "--seed", "0", "--train", p, "--eval-part",
           "TRAIN", "--eval", p, "--rows-limit", "2", "--steps", "6", "--out-root", str(out_root), "--device", "cpu",
           "--execute", *extra]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=guard.ROOT)
    if r.returncode != 0:
        raise RuntimeError(r.stdout[-2000:] + r.stderr[-2000:])


def t8_resume():
    import shutil
    import numpy as np
    import torch
    cfgs = [{"tag": "a", "target": "R3", "c": 1.0, "eta": 1.0, "H": 32, "lam": 0.01},
            {"tag": "b", "target": "R2", "c": 1.0, "H": 32, "lam": 0.01}]
    cp = TMP / "cfg_resume.json"
    cp.write_text(json.dumps(cfgs))
    A, B = TMP / "rA", TMP / "rB"
    for d in (A, B):
        if d.exists():
            shutil.rmtree(d)
    run_job(cp, A)
    run_job(cp, B, ("--max-routers", "1"))
    run_job(cp, B)
    ok = True
    for t in ("a", "b"):
        for bk in ("SEM", "SHUF"):
            ea = np.load(A / "T" / t / f"{bk}_s0_TRAIN.npz")["e"]
            eb = np.load(B / "T" / t / f"{bk}_s0_TRAIN.npz")["e"]
            sa = torch.load(A / "T" / t / f"{bk}_s0.pt", weights_only=False)["state"]
            sb = torch.load(B / "T" / t / f"{bk}_s0.pt", weights_only=False)["state"]
            ok &= bool(np.array_equal(ea, eb)) and all(torch.equal(sa[k], sb[k]) for k in sa)
    return {"T8": bool(ok)}


def t9_disjoint():
    import numpy as np
    from csr.ccsr1_paths import ORV1
    from csr.fsp_lib import Split
    sp = Split("calib_search")
    pz = np.load(ORV1 / "tables/parts.npz")
    a = {sp.papers[d] for d in pz["SEL_docs"]}
    b = {sp.papers[d] for d in pz["EVAL_docs"]}
    c = set(Split("calib_sel").papers)
    return {"T9": len(a & b) == 0 and len(c & set(sp.papers)) == 0, "n_train_papers": len(b), "n_sel_papers": len(a),
            "n_cs2_papers": len(c)}


def t10_targets():
    import torch
    from csr.ccsr1_router import teacher
    g = torch.Generator().manual_seed(1)
    u = torch.randn(500, 16, generator=g) * 0.05
    p = torch.softmax(torch.randn(500, 16, generator=g), 1)
    lp = torch.log(p + 1e-4)
    r1 = teacher("R1", u, lp, 1.0, 1.0, 0.05)
    r2 = teacher("R2", u, lp, 1.0, 0.0, 0.05)
    r30 = teacher("R3", u, lp, 1.0, 0.0, 0.05)
    r3 = teacher("R3", u, lp, 1.0, 1.0, 0.05)
    z3 = teacher("R3", u, lp, 1.0, 1.0, 0.05, n_out=17)
    z1 = teacher("R1", u, lp, 1.0, 1.0, 0.05, n_out=17)
    ok = (float((r1 - (p + 1e-4) / (p + 1e-4).sum(1, keepdim=True)).abs().max()) < 1e-5 and torch.allclose(r2, r30)
          and bool((r2.argmax(1) == u.argmax(1)).all()) and torch.allclose(r3.sum(1), torch.ones(500))
          and z3.shape[1] == 17 and float(z1[:, 16].abs().max()) == 0.0)
    return {"T10": bool(ok)}


def main():
    guard.check_root()
    if os.environ.get("MOE_RESEARCH_APPROVE") != "cpu-model":
        print("set MOE_RESEARCH_APPROVE=cpu-model")
        sys.exit(2)
    import torch
    torch.set_num_threads(16)
    TMP.mkdir(parents=True, exist_ok=True)
    from csr.ccsr1_paths import ART
    ART.mkdir(parents=True, exist_ok=True)
    res = {}
    only = os.environ.get("CCSR1_TESTS")
    fns = (t1_causality, t2_cached_vs_batched, t3_one_token_decoding, t4_teacher_absent, t5_target_alignment,
           t6_expert_selection, t7_equal_compute, t8_resume, t9_disjoint, t10_targets)
    for fn in fns:
        if only and fn.__name__ not in only.split(","):
            continue
        try:
            r = fn()
        except Exception:  # noqa: BLE001
            import traceback
            r = {fn.__name__: False, "error": traceback.format_exc()[-3000:]}
        res[fn.__name__] = r
        print(fn.__name__, json.dumps(r, default=str)[:600], flush=True)
    keys = [f"T{i}" for i in range(1, 11)]
    flat = {k: v for r in res.values() for k, v in r.items()}
    passed = {k: bool(flat.get(k, False)) for k in keys}
    out = {"time": guard.now(), "passed": passed, "all_pass": all(passed.values()), "detail": res}
    if not only:
        p = ART / f"cpu_tests_{guard.utc_stamp()}.json"
        guard.atomic_write_json(p, out, mode=0o400)
        print(p)
    print(json.dumps(passed), "ALL PASS" if out["all_pass"] else "FAIL")
    sys.exit(0 if out["all_pass"] else 1)


if __name__ == "__main__":
    main()
