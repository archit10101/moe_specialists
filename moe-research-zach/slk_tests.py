#!/usr/bin/env python
"""csr-semantic-edl-long-adaptive-k-v1 CPU tests (prereg section 9), CUDA hidden.
Writes artifacts/semantic-edl-long-k/cpu_tests_<UTC>.json. Usage: CUDA_VISIBLE_DEVICES= .venv/bin/python slk_tests.py"""
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402

RES, TESTS = {}, []


def test(fn):
    TESTS.append(fn)
    return fn


def tab(rel):
    import numpy as np
    z = np.load(guard.ROOT / rel)
    return z["opaque"], z["q"].astype(np.float64)


@test
def t01_paper_constant_assignment():
    import numpy as np
    from csr.gc_lib import PaperQ
    from csr.lfr_lib import main_stream
    from csr.slk_paths import C6_TABLE, C6_TABLE_SHA
    sp, _, _ = main_stream(245_000_000)
    pq = PaperQ(guard.ROOT / C6_TABLE, sp, C6_TABLE_SHA)
    first, n_multi = {}, 0
    for di, p in enumerate(sp.papers):
        if p in first:
            assert np.array_equal(pq.q_of_doc[di], pq.q_of_doc[first[p]]), p
            n_multi += 1
        else:
            first[p] = di
    return {"coverage": pq.coverage, "segments_checked": n_multi}


@test
def t02_shuffle_matching_exact():
    import numpy as np
    from csr.slk_lib import shuffle_table
    from csr.slk_paths import C6_SHUF_TABLE, C6_SHUF_TABLE_SHA, C6_TABLE, C6_TABLE_SHA
    assert guard.sha256_file(guard.ROOT / C6_TABLE) == C6_TABLE_SHA
    assert guard.sha256_file(guard.ROOT / C6_SHUF_TABLE) == C6_SHUF_TABLE_SHA
    o1, q = tab(C6_TABLE)
    o2, qs = tab(C6_SHUF_TABLE)
    assert (o1 == o2).all()
    has = q.sum(1) > 0
    assert np.array_equal(has, qs.sum(1) > 0)
    a = q[has][np.lexsort(q[has].T[::-1])]
    b = qs[has][np.lexsort(qs[has].T[::-1])]
    assert np.array_equal(a, b), "row multiset differs"
    assert np.allclose(q.sum(0), qs.sum(0), rtol=0, atol=1e-6 * len(q)), "cluster masses differ"
    assert np.array_equal(np.sort(q[has].max(1)), np.sort(qs[has].max(1))), "confidence distribution differs"
    rep = shuffle_table(o1, q, 20261117)
    assert np.array_equal(rep.astype(np.float32), qs.astype(np.float32)), "shuffle not reproducible"
    hard_same = float((q[has].argmax(1) == qs[has].argmax(1)).mean())
    return {"n_papers": int(has.sum()), "hard_agreement": hard_same,
            "mass_max_abs_diff": float(np.abs(q.sum(0) - qs.sum(0)).max())}


@test
def t03_sharpening_reproduces_c6():
    import numpy as np
    from csr.slk_lib import sharpen
    from csr.slk_paths import C6_TABLE, SEM_TABLE
    _, q = tab(SEM_TABLE.relative_to(guard.ROOT))
    _, q6 = tab(C6_TABLE)
    assert np.allclose(sharpen(q, 0.5).astype(np.float32), q6, atol=1e-6)
    return {"max_abs": float(np.abs(sharpen(q, 0.5) - q6).max())}


@test
def t04_hierarchy_nesting_and_mass():
    import numpy as np
    from csr.slk_lib import aggregate, build_hierarchy
    rng = np.random.default_rng(0)
    q = rng.dirichlet(np.ones(32) * 0.3, size=600)
    X = rng.normal(size=(600, 16))
    maps = build_hierarchy(q, X, (16, 8, 4))
    assert sorted(maps) == [4, 8, 16]
    for fine, coarse in ((16, 8), (8, 4)):
        for i in range(32):
            for j in range(32):
                if maps[fine][i] == maps[fine][j]:
                    assert maps[coarse][i] == maps[coarse][j], "not nested"
    for K, m in maps.items():
        assert len(set(m.tolist())) == K
        a = aggregate(q, m, K)
        assert np.allclose(a.sum(1), 1.0) and np.isclose(a.sum(), q.sum())
    return {"sizes": {K: np.bincount(m).tolist() for K, m in maps.items()}}


@test
def t05_loss_scaling_onehot_equals_edl():
    import torch
    from csr.fsp_model import edl_loss
    from csr.gc_lib import gc_edl_loss
    g = torch.Generator().manual_seed(0)
    N, E, S = 512, 32, 8
    p = torch.softmax(torch.randn(N, E, generator=g), -1)
    sid = torch.arange(N) // (N // S)
    lab_seq = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2])
    lab = lab_seq[sid]
    nonpad = torch.ones(N, dtype=torch.bool)
    q = torch.nn.functional.one_hot(lab_seq, 5).float()
    a, b = float(edl_loss(p, lab, sid, nonpad)), float(gc_edl_loss(p, q, sid, nonpad))
    assert abs(a - b) < 1e-5, (a, b)
    return {"edl": a, "gc": b}


@test
def t06_bootstrap_stats_equal_direct():
    import numpy as np
    from csr.slk_lib import N_DEC, interaction_from_stats, strata
    rng = np.random.default_rng(1)
    N, D = 5000, 40
    doc = rng.integers(0, D, N)
    pos = rng.integers(0, 1024, N)
    bce = rng.gamma(2.0, 1.5, N)
    d = rng.normal(0, 0.01, N) + 0.002 * (doc % 3 == 0)
    own = (doc % 3 == 0)
    S, dec = strata(bce, pos)
    nS = int(S.max()) + 1
    own_s = np.bincount(doc[own], weights=d[own], minlength=D)
    own_c = np.bincount(doc[own] * nS + S[own], minlength=D * nS).reshape(D, nS).astype(float)
    oth = ~own
    oth_s = np.bincount(doc[oth] * nS + S[oth], weights=d[oth], minlength=D * nS).reshape(D, nS)
    oth_c = np.bincount(doc[oth] * nS + S[oth], minlength=D * nS).reshape(D, nS).astype(float)
    got = float(interaction_from_stats(own_s, own_c, oth_s, oth_c)[0])
    ok = [j for j in range(nS) if (own & (S == j)).any() and (oth & (S == j)).any()]
    pj = np.asarray([(own & (S == j)).sum() for j in ok], float)
    pj /= pj.sum()
    xm = np.asarray([d[oth & (S == j)].mean() for j in ok])
    want = float(d[own].mean() - (pj * xm).sum())
    assert abs(got - want) < 1e-12, (got, want)
    # decile-only strata reproduce the fle per-expert formula
    own_c2 = np.bincount(doc[own] * N_DEC + dec[own], minlength=D * N_DEC).reshape(D, N_DEC).astype(float)
    oth_s2 = np.bincount(doc[oth] * N_DEC + dec[oth], weights=d[oth], minlength=D * N_DEC).reshape(D, N_DEC)
    oth_c2 = np.bincount(doc[oth] * N_DEC + dec[oth], minlength=D * N_DEC).reshape(D, N_DEC).astype(float)
    got2 = float(interaction_from_stats(own_s, own_c2, oth_s2, oth_c2)[0])
    wk = np.bincount(dec[own], minlength=N_DEC) / own.sum()
    xm2 = np.asarray([d[oth & (dec == q)].mean() for q in range(N_DEC)])
    want2 = float(d[own].mean() - (wk * xm2).sum())
    assert abs(got2 - want2) < 1e-12
    w = rng.multinomial(D, np.ones(D) / D, size=200).astype(float)
    bs = interaction_from_stats(own_s, own_c, oth_s, oth_c, w)
    return {"point": got, "boot_sd": float(bs.std()), "decile_only": got2}


@test
def t07_routing_metrics_and_probe_sanity():
    import numpy as np
    from csr.slk_lib import probe_logreg, routing_subject
    rng = np.random.default_rng(2)
    N, L, E, K = 20000, 2, 32, 5
    lab = rng.integers(0, K, N)
    gdoc = np.arange(N) // 50
    lab = lab[gdoc * 50 % N]  # document-constant labels
    topi_s = np.stack([np.stack([(lab * 6 + j) % E for j in range(4)], 1)] * L, 1).astype(np.uint8)
    topi_r = rng.integers(0, E, (N, L, 4)).astype(np.uint8)
    a = routing_subject(topi_s, lab, gdoc, K, E)
    b = routing_subject(topi_r, lab, gdoc, K, E)
    assert a["nmi_mean"] > 0.5 and b["nmi_mean"] < 0.01 and a["nmi_null_route_mean"] < 0.01
    X = rng.normal(size=(2000, 20))
    y = (X[:, 0] > 0).astype(int) + 2 * (X[:, 1] > 0)
    good = probe_logreg(X[:1000], y[:1000], X[1000:], y[1000:], 4)
    rnd = probe_logreg(X[:1000], rng.integers(0, 4, 1000), X[1000:], y[1000:], 4)
    assert good["acc"] > 0.85 and rnd["acc"] < 0.4
    return {"nmi_struct": a["nmi_mean"], "nmi_random": b["nmi_mean"], "probe": good["acc"], "probe_rnd": rnd["acc"]}


@test
def t08_conditions_differ_only_in_table():
    from slk_configs import phase1_conds
    c = phase1_conds()
    a, b = dict(c["C6"]["paper_latent"]), dict(c["SHUF"]["paper_latent"])
    diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert diff == {"table", "table_sha256"}, diff
    assert "layers" not in a and a["beta"] == 0.005
    return {"differs": sorted(diff)}


@test
def t09_access_guards():
    from csr.fsp_lib import Split
    try:
        Split("test")
        raise AssertionError("test split loadable")
    except guard.GuardError:
        pass
    from csr.slk_paths import ART, REUSED
    assert not (ART / "PHASE1_VAL_OPEN.json").exists()
    env = {**os.environ, "MOE_RESEARCH_APPROVE": "cpu-model", "CUDA_VISIBLE_DEVICES": ""}
    r = subprocess.run([sys.executable, "slk_evaluate.py", "val", "--run-dir", REUSED["T4_s0"], "--run-name",
                        "guardtest", "--tag", "p100", "--execute"], capture_output=True, text=True, env=env,
                       cwd=guard.ROOT, timeout=600)
    assert r.returncode != 0 and "VAL is opened only" in (r.stdout + r.stderr), r.stdout[-500:] + r.stderr[-1500:]
    r = subprocess.run([sys.executable, "slk_evaluate.py", "calibsel", "--run-dir", REUSED["T4_s0"], "--run-name",
                        "guardtest", "--tag", "p100", "--execute"], capture_output=True, text=True, env=env,
                       cwd=guard.ROOT, timeout=600)
    assert r.returncode != 0 and "CALIB-SELECTION is opened only" in (r.stdout + r.stderr)
    import shutil
    shutil.rmtree(ART / "evals" / "guardtest", ignore_errors=True)
    return {"test_locked": True, "val_guard": True, "calibsel_guard": True}


@test
def t10_reused_baselines_intact():
    from csr.slk_ckpt import verify_dir
    from csr.slk_paths import MILESTONES, REUSED
    out = {}
    for k, rd in REUSED.items():
        st = json.loads((guard.ROOT / rd / "status.json").read_text())
        cfg = json.loads((guard.ROOT / rd / "config.json").read_text())
        assert st["status"] == "COMPLETE" and st["seen"] == 245032457
        assert st["plan_sha256"].startswith("f4bc3c93")
        for tag in ("p040", "p060", "p080", "p100"):
            c = st["ckpts"][tag]
            assert c["step"] == MILESTONES[tag]
            assert verify_dir(guard.ROOT / c["path"]), c["path"]
        out[k] = {"cond": cfg["cond"], "seed": cfg["seed"]}
    return out


def main():
    guard.check_root()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from csr.slk_paths import ART, check_prereg
    check_prereg()
    for fn in TESTS:
        t0 = time.time()
        try:
            RES[fn.__name__] = {"ok": True, "detail": fn(), "s": round(time.time() - t0, 1)}
        except Exception:
            RES[fn.__name__] = {"ok": False, "error": traceback.format_exc()[-3000:], "s": round(time.time() - t0, 1)}
        print(fn.__name__, "OK" if RES[fn.__name__]["ok"] else "FAIL", flush=True)
    out = ART / f"cpu_tests_{guard.utc_stamp()}.json"
    guard.atomic_write_json(out, {"time": guard.now(), "all_ok": all(r["ok"] for r in RES.values()), "results": RES})
    print(out, "ALL_OK" if all(r["ok"] for r in RES.values()) else "FAILURES")
    sys.exit(0 if all(r["ok"] for r in RES.values()) else 1)


if __name__ == "__main__":
    main()
