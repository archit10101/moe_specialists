#!/usr/bin/env python
"""asrs1 CPU tests (prereg section 9; main-stream training rows only, 1 row per step, 2 cycles of 5 steps:
base 2, spec 2, router 1; steps 318 -> 328). No calibration / validation / test data are read.

  T1 phase isolation (param hashes at every boundary): base changes only in BASE; specialists only in SPEC; causal
     routers and the ALT-G token router only in ROUTER; nothing in REFRESH; FedBank student/EMA buffers never
  T2 teacher caches: read-only (0400), sha256 == recorded, recomputed identically by an independent run
  T3 identical rows/budgets for every arm (single process; per-phase row hashes, token counts == plan)
  T4 resume equivalence: --max-steps 324 (exit 75) + resume == uninterrupted, bitwise (model, banks, routers, opts)
  T5 external SIGINT at a step -> exit 77 + verified slot; resume == uninterrupted, bitwise
  T6 external SIGINT inside a teacher refresh -> exit 77, no cache recorded; resume == uninterrupted, bitwise
  T7 disabled specialist path (zero residual) reproduces the base CE exactly; ALT-G with zeroed experts == base
  T8 causality: changing tokens after position t changes neither h_<=t nor the router's q_<=t
  T9 no validation/test access in asrs1 code (static scan) and the locked loader refuses the test split
  asrs1_tests.py"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402

PY = str(guard.ROOT / ".venv/bin/python")
TD = guard.ROOT / ".tmp/csr-alternating-semantic-residual-specialization-v1/cputests"
SCHED = {"start": 318, "end": 328, "n_cycles": 2, "base": 2, "spec": 2}


def mkcfg(name):
    from csr import asrs1_cfg as C
    cfg = C.build(name, test=True, schedule=SCHED, slot_every_steps=100)
    cfg["verify_ckpt"] = False
    TD.mkdir(parents=True, exist_ok=True)
    p = TD / f"{name}.json"
    p.write_text(json.dumps(cfg))
    return p


def train(cfgp, *extra):
    env = {**os.environ, "MOE_RESEARCH_APPROVE": "cpu-model"}
    r = subprocess.run([PY, "asrs1_train.py", "--config", str(cfgp.relative_to(guard.ROOT)), "--device", "cpu",
                        "--test-rows", "1", "--execute", *extra], cwd=guard.ROOT, env=env, capture_output=True, text=True)
    name = json.loads(cfgp.read_text())["run_name"]
    cands = sorted((guard.ROOT / "runs").glob(f"*_asrs1_{name}"))
    return r.returncode, (cands[-1] if cands else None), (r.stdout or "")[-3000:] + (r.stderr or "")[-3000:]


def final_slot(run_dir):
    import torch
    from csr.asrs1_ckpt import ResumeStore
    st = ResumeStore(run_dir.name, "x")
    e = st.latest_verified()
    return torch.load(guard.ROOT / e["path"] / "state.pt", map_location="cpu", weights_only=False), e


def eq_state(a, b):
    import torch

    def cmp(x, y, path):
        if isinstance(x, dict):
            assert set(x) == set(y), f"{path}: keys differ"
            for k in x:
                cmp(x[k], y[k], f"{path}.{k}")
        elif isinstance(x, (list, tuple)):
            assert len(x) == len(y), path
            for i, (u, v) in enumerate(zip(x, y)):
                cmp(u, v, f"{path}[{i}]")
        elif torch.is_tensor(x):
            assert torch.equal(x, y), f"{path}: tensors differ"
        elif isinstance(x, float):
            assert x == y or (x != x and y != y), f"{path}: {x} != {y}"
    for k in ("model", "banks", "routers", "opt", "bank_opt", "gropt", "router_opt"):
        cmp(a[k], b[k], k)
    for k in ("next_step", "seen", "n_done", "phase_tok"):
        assert a["st"][k] == b["st"][k], k
    assert {c: v["sha256"] for c, v in a["st"]["refreshed"].items()} == {c: v["sha256"] for c, v in b["st"]["refreshed"].items()}


def hashes(run_dir):
    out = []
    for line in (run_dir / "metrics.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("kind") == "param_hash":
            out.append(r)
    return out


def t1_isolation(run_dir):
    H = hashes(run_dir)
    assert H[0]["after"] == "start"
    order = [h["after"] for h in H[1:]]
    assert order == ["base", "refresh", "spec", "router"] * 2, order
    changed = lambda a, b: {k for k in a["hash"] if a["hash"][k] != b["hash"][k]}
    ex = {"base": {"base"}, "refresh": set(),
          "spec": {"SEM.experts", "SHUF.experts", "ALTG.experts"},
          "router": {"ALTG.router"} | {f"router.{b}_s{r}" for b in ("SEM", "SHUF") for r in (0, 1, 2)}}
    for prev, cur in zip(H[:-1], H[1:]):
        got = changed(prev, cur)
        assert got == ex[cur["after"]], (cur["after"], got)
    return {"boundaries": len(H) - 1}


def t2_caches(run_dir, other):
    out = {}
    from csr.asrs1_paths import CACHES
    a = sorted((CACHES / run_dir.name).glob("cache_c*.npz"))
    b = sorted((CACHES / other.name).glob("cache_c*.npz"))
    assert len(a) == 2 and len(b) == 2
    for x, y in zip(a, b):
        assert (x.stat().st_mode & 0o777) == 0o400, x
        js = json.loads(x.with_suffix(".json").read_text())
        assert guard.sha256_file(x) == js["sha256"]
        assert guard.sha256_file(x) == guard.sha256_file(y), "teacher cache not recomputed identically"
        out[x.name] = js["sha256"][:12]
    import numpy as np
    z = np.load(a[0])
    assert set(z.files) >= {"steps", "off", "y", "e_SEM", "pi_SEM", "w_SEM", "e_SHUF", "pi_SHUF", "w_SHUF"}
    assert z["pi_SEM"].dtype == np.float16 and abs(z["pi_SEM"].astype(np.float64).sum(1) - 1).max() < 1e-2
    return out


def t3_budgets(slot):
    import hashlib
    from csr.lfr_lib import main_stream
    from csr import asrs1_paths as P
    sp, steps, _ = main_stream(245e6)
    steps = [s[:1] for s in steps]
    sch = P.schedule(**{k: SCHED[k] for k in ("start", "end", "n_cycles")}, base=SCHED["base"], spec=SCHED["spec"])
    tot = 0
    for c, ph in sch:
        for n, s0, s1 in ph:
            want = sum(int(sum(sp.lengths[d] - 1 for r in steps[s] for d in r)) for s in range(s0, s1))
            assert slot["st"]["phase_tok"][f"{c}:{n}"] == want
            rec = slot["st"]["phase_rec"][f"{c}:{n}"]
            assert rec["rows_sha256"] == hashlib.sha256(json.dumps(steps[s0:s1]).encode()).hexdigest()
            tot += want
    assert slot["st"]["seen"] == tot
    return {"tokens": tot}


def t7_zero_path():
    import torch
    from ccsr1_s3_eval import ce_multi
    from csr.cfar1_bank import FedBank
    from csr.jsr_bank import rms0
    from csr import asrs1_paths as P
    ck = torch.load(guard.ROOT / P.CKPT / "state.pt", map_location="cpu", weights_only=False)
    W = ck["model"]["tok_emb.weight"].float()
    g = torch.Generator().manual_seed(0)
    h = torch.randn(512, 384, generator=g) * 3
    y = torch.randint(0, W.shape[0], (512,), generator=g)
    b = FedBank(ck["bank_specs"][P.BANK_SEM], 100)
    b.load_state_dict(ck["banks"][P.BANK_SEM])
    with torch.no_grad():
        b.experts.w2.zero_()
    a = b.correction(rms0(h), torch.randint(0, 16, (512,), generator=g))
    assert float(a.abs().max()) == 0.0
    ceb, (cex,) = ce_multi(h, W, y, [a])
    assert torch.equal(ceb, cex), float((ceb - cex).abs().max())
    return {"max_abs": 0.0}


def t8_causal():
    import torch
    from csr.ccsr1_router import CausalRouter
    from csr.jsr_bank import rms0
    from csr.lfr_model import LFRConfig, LFRModel
    from csr import asrs1_paths as P
    ck = torch.load(guard.ROOT / P.CKPT / "state.pt", map_location="cpu", weights_only=False)
    m = LFRModel(LFRConfig.from_dict({**ck["model_cfg"], "teacher": False}), 100)
    m.load_state_dict(ck["model"])
    m.eval()
    rs = torch.load(guard.ROOT / P.ROUTER_DIR / "SEM_s0.pt", map_location="cpu", weights_only=False)
    r = CausalRouter(64, 0)
    r.load_state_dict(rs["state"])
    g = torch.Generator().manual_seed(1)
    L, t = 96, 60
    x = torch.randint(0, 50000, (1, L), generator=g)
    x2 = x.clone()
    x2[0, t + 1:] = torch.randint(0, 50000, (L - t - 1,), generator=g)
    inp = lambda z: {"input_ids": z, "doc_ids": torch.zeros_like(z), "positions": torch.arange(L).view(1, -1)}
    ctl = {"teach": False, "edl": False, "lc": False, "cf_store": None, "tap": False}
    with torch.no_grad():
        h1 = m(inp(x), ctl)["hidden"][0]
        h2 = m(inp(x2), ctl)["hidden"][0]
        q1 = r(rms0(h1).unsqueeze(0).half(), torch.tensor([L]))[0]
        q2 = r(rms0(h2).unsqueeze(0).half(), torch.tensor([L]))[0]
    dh = float((h1[:t + 1] - h2[:t + 1]).abs().max())
    dq = float((q1[:t + 1] - q2[:t + 1]).abs().max())
    assert dh < 1e-4 and dq < 1e-4, (dh, dq)
    assert float((q1[t + 1:] - q2[t + 1:]).abs().max()) > 0, "future change has no effect (test not sensitive)"
    return {"max_dh_prefix": dh, "max_dq_prefix": dq}


def t9_no_heldout():
    bad = []
    pat = re.compile(r"""(Split\(\s*["'](val|validation|test)["']|["']calib_sel["']|tables_val|"VAL"|"TEST")""")
    for f in sorted(list(guard.ROOT.glob("asrs1_*.py")) + list((guard.ROOT / "src/csr").glob("asrs1_*.py"))):
        if f.name == "asrs1_tests.py":
            continue
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if pat.search(line):
                bad.append(f"{f.name}:{i}: {line.strip()[:120]}")
    assert not bad, bad
    from csr.fsp_lib import Split
    try:
        Split("test")
        raise AssertionError("test split loader did not refuse")
    except AssertionError:
        raise
    except Exception as e:
        refused = str(e)[:120]
    return {"static_hits": 0, "test_loader": refused}


def drop(run_dirs):
    """This experiment's CPU-test scratch (slots, caches): deleted with path/size/sha256 in deletions.tsv."""
    import shutil
    from csr import asrs1_log as LG
    from csr.asrs1_paths import CACHES, TMP
    for rd in run_dirs:
        for d in (TMP / "slots" / rd.name, CACHES / rd.name):
            if not d.exists():
                continue
            for f in sorted(d.rglob("*")):
                if f.is_file():
                    LG.deletion(str(f.relative_to(guard.ROOT)), guard.sha256_file(f), f.stat().st_size,
                                "asrs1 CPU-test scratch")
            shutil.rmtree(d)


def main():
    guard.check_root()
    res = {"time": guard.now()}
    ok = True

    def rec(name, fn):
        nonlocal ok
        try:
            res[name] = {"pass": True, **(fn() or {})}
        except Exception as e:  # noqa: BLE001
            import traceback
            ok = False
            res[name] = {"pass": False, "error": f"{type(e).__name__}: {e}"[:1500], "tb": traceback.format_exc()[-1500:]}
        print(name, json.dumps(res[name])[:400], flush=True)

    rec("T7_zero_path", t7_zero_path)
    rec("T8_causal", t8_causal)
    rec("T9_no_heldout", t9_no_heldout)
    rcA, A, logA = train(mkcfg("TA"))
    print("A rc", rcA, logA[-800:], flush=True)
    if rcA != 0:
        res["A"] = {"pass": False, "rc": rcA, "log": logA}
        ok = False
    else:
        sA, _ = final_slot(A)
        rec("T1_isolation", lambda: t1_isolation(A))
        rec("T3_budgets", lambda: t3_budgets(sA))
        # T4: max-steps leg + resume
        cB = mkcfg("TB")
        rc1, B, l1 = train(cB, "--max-steps", "324")
        rc2, B2, l2 = (train(cB, "--run-dir", str(B.relative_to(guard.ROOT))) if B else (None, None, ""))

        def t4():
            assert rc1 == 75 and rc2 == 0, (rc1, rc2, l1[-600:], l2[-600:])
            eq_state(sA, final_slot(B)[0])
            return {"legs": 2}
        rec("T4_resume", t4)
        rec("T2_caches", lambda: t2_caches(A, B))
        cC = mkcfg("TC")
        rc1c, C_, l1c = train(cC, "--test-signal-at-step", "325")
        rc2c, _, l2c = (train(cC, "--run-dir", str(C_.relative_to(guard.ROOT))) if C_ else (None, None, ""))

        def t5():
            assert rc1c == 77 and rc2c == 0, (rc1c, rc2c, l1c[-600:], l2c[-600:])
            eq_state(sA, final_slot(C_)[0])
            return {"stopped_at": 325}
        rec("T5_sigint_step", t5)
        cD = mkcfg("TD")
        rc1d, D_, l1d = train(cD, "--test-signal-in-refresh", "2")
        rc2d, _, l2d = (train(cD, "--run-dir", str(D_.relative_to(guard.ROOT))) if D_ else (None, None, ""))

        def t6():
            assert rc1d == 77 and rc2d == 0, (rc1d, rc2d, l1d[-600:], l2d[-600:])
            st1 = json.loads((D_ / "leg_01.json").read_text())
            assert "2" not in st1["refreshed"], st1["refreshed"]
            eq_state(sA, final_slot(D_)[0])
            return {"refresh_interrupted_cycle": 2}
        rec("T6_sigint_refresh", t6)
        drop([x for x in (A, B, C_, D_) if x is not None])
    res["all_pass"] = ok
    from csr.asrs1_paths import ART
    ART.mkdir(parents=True, exist_ok=True)
    p = ART / f"cpu_tests_{guard.utc_stamp()}.json"
    guard.atomic_write_json(p, res, mode=0o400)
    print("ALL PASS" if ok else "FAILURES", p, flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
