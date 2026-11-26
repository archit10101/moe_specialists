#!/usr/bin/env python
"""csr-forecast-latent-edl-v1 CPU tests (prereg section 10), CUDA hidden. Writes artifacts/.../cpu_tests_<UTC>.json."""
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402

RES = {}


def test(fn):
    t0 = time.time()
    try:
        fn()
        RES[fn.__name__] = {"ok": True, "s": round(time.time() - t0, 2)}
    except Exception:
        RES[fn.__name__] = {"ok": False, "err": traceback.format_exc()[-2500:]}
    print(fn.__name__, RES[fn.__name__]["ok"], flush=True)
    return fn


def _draft():
    import torch
    from csr.fle_lib import DRAFT_G0, DRAFT_G0_SHA, load_draft
    return load_draft(DRAFT_G0, torch.device("cpu"), DRAFT_G0_SHA)


@test
def t01_causal_forecast():
    import torch
    from csr.fle_lib import draft_forecast, region_points
    from csr.fsp_lib import draft_hidden
    from csr.fle_lib import uniform_from
    dm = _draft()
    torch.manual_seed(1)
    T = 400
    ids = torch.randint(2, 50000, (2, T))
    points = torch.tensor([[128, 256], [128, -1]])
    keys = torch.tensor([[5, 6], [7, 0]])
    tok, p, e, t1 = draft_forecast(dm, ids, None, points, keys, H=5)
    ids2 = ids.clone()
    ids2[0, 256:] = torch.randint(2, 50000, (T - 256,))
    ids2[1, 128:] = torch.randint(2, 50000, (T - 128,))
    tok2, p2, _, _ = draft_forecast(dm, ids2, None, points, keys, H=5)
    assert torch.equal(tok[:, :, :], tok2), "future tokens changed a forecast"
    assert torch.allclose(p, p2)
    # equals naive incremental generation from the prefix only
    seq = ids[0, :256].clone()
    for j in range(1, 6):
        x = seq.view(1, -1)
        h = draft_hidden(dm, x, torch.zeros_like(x), torch.arange(x.shape[1]).view(1, -1))
        lg = (h[0, -1] @ dm.tok_emb.weight.t()).float()
        tv, ti = (lg / 0.8).topk(50)
        kk = torch.tensor([6 * 131 + j * 1_000_033 + r for r in range(50)], dtype=torch.int64)
        g = -torch.log(-torch.log(uniform_from(kk))).float()
        tk = ti[(tv + g).argmax()]
        assert int(tk) == int(tok[0, 1, j - 1]), (j, int(tk), int(tok[0, 1, j - 1]))
        seq = torch.cat([seq, tk.view(1)])
    assert region_points(1024, 128) == [128, 256, 384, 512, 640, 768, 896]
    assert region_points(1000, 256) == [256, 512, 768]
    assert region_points(160, 128) == [128]
    assert region_points(150, 128) == []


@test
def t02_permutation_invariance():
    import numpy as np
    import torch
    from csr.fle_lib import DeepSetsCV, a1_meanstat, set_stats
    g = np.random.default_rng(0)
    emb = g.normal(size=(10, 4, 384)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=-1, keepdims=True)
    ds = g.normal(size=(10, 4, 3)).astype(np.float32)
    perm = [2, 0, 3, 1]
    ss = set_stats(emb, ds, np.zeros((10, 4)))
    ssp = set_stats(emb[:, perm], ds[:, perm], np.zeros((10, 4)))
    assert np.allclose(ss, ssp, atol=1e-5)
    mu, sd = np.zeros(5), np.ones(5)
    assert np.allclose(a1_meanstat(emb, ss, mu, sd), a1_meanstat(emb[:, perm], ssp, mu, sd), atol=1e-5)
    net = DeepSetsCV().eval()
    a = net(torch.as_tensor(emb), torch.as_tensor(ds), torch.as_tensor(ss, dtype=torch.float32))
    b = net(torch.as_tensor(emb[:, perm]), torch.as_tensor(ds[:, perm]), torch.as_tensor(ssp, dtype=torch.float32))
    assert torch.allclose(a, b, atol=1e-5)


@test
def t03_balanced_sinkhorn():
    import numpy as np
    import torch
    from csr.fle_lib import assign, fit_balanced, sinkhorn_log
    g = np.random.default_rng(0)
    X = np.concatenate([g.normal(loc=3, size=(900, 16)), g.normal(loc=-3, size=(100, 16))]).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Q, b = sinkhorn_log(torch.as_tensor(X @ X[:8].T), 0.05, iters=50)
    col = Q.sum(0).numpy()
    assert np.allclose(Q.sum(1).numpy(), 1, atol=1e-5)
    assert np.abs(col / col.mean() - 1).max() < 0.01, col
    Y = np.concatenate([g.normal(loc=0.4, size=(900, 16)), g.normal(loc=-0.4, size=(100, 16))]).astype(np.float32)
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)  # overlapping (realistic) 9:1 imbalance
    C, b = fit_balanced(Y, 4, 0.05)
    q = assign(Y, C, b, 0.05)
    assert np.allclose(q.sum(1), 1, atol=1e-5)
    mass = q.sum(0)
    assert mass.min() / mass.max() > 0.5, mass  # approximately balanced despite a 9:1 imbalance


@test
def t04_frozen_detached_assignments():
    import torch
    from csr.fle_lib import fledl_loss
    torch.manual_seed(0)
    lg = torch.randn(60, 8, requires_grad=True)
    p = torch.softmax(lg, -1)
    reg = torch.arange(60) // 10
    q = torch.softmax(torch.randn(6, 3), -1).requires_grad_(True)
    loss, _ = fledl_loss([p], reg, q)
    loss.backward()
    assert q.grad is None, "gradient reached the assignments"
    assert lg.grad is not None and lg.grad.abs().sum() > 0


@test
def t05_fledl_equals_edl_onehot():
    import torch
    from csr.fle_lib import fledl_loss
    from csr.fsp_model import edl_loss
    torch.manual_seed(0)
    N, E = 120, 8
    p = torch.softmax(torch.randn(N, E), -1)
    seq = torch.arange(N) // 20  # 6 segments
    dom = torch.tensor([0, 1, 2, 0, 1, 2])[seq]
    a = edl_loss(p, dom, seq, torch.ones(N, dtype=torch.bool))
    q = torch.nn.functional.one_hot(torch.tensor([0, 1, 2, 0, 1, 2]), 3).float()
    b, _ = fledl_loss([p], seq, q)
    assert torch.allclose(a, b, atol=1e-5), (float(a), float(b))


@test
def t13_fledl_finite_identical_clusters():
    """Regression (deviation 6): identical cluster routing means (JSD rounding below 0) must stay finite."""
    import torch
    from csr.fle_lib import fledl_loss
    torch.manual_seed(0)
    p = torch.softmax(torch.randn(1, 32).repeat(400, 1) + 1e-7 * torch.randn(400, 32), -1).requires_grad_(True)
    reg = torch.arange(400) // 10
    q = torch.softmax(torch.randn(40, 32) * 0.01, -1)
    loss, _ = fledl_loss([p], reg, q)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(p.grad).all()


@test
def t06_qtable_and_deterministic_cache_reuse():
    import numpy as np
    from csr.fle_lib import QTable
    d = Path(tempfile.mkdtemp(dir=guard.ROOT / ".tmp/forecast-latent-edl-v1"))
    q = np.random.default_rng(0).dirichlet(np.ones(4), size=6).astype(np.float32)
    np.savez(d / "t.npz", gdoc=np.array([10, 10, 10, 11, 11, 12], np.int32), j=np.array([1, 2, 3, 1, 2, 1], np.int16),
             q=q, K=np.int64(4), L=np.int64(256))
    t1, t2 = QTable(d / "t.npz"), QTable(d / "t.npz")
    assert t1.sha == t2.sha
    B, T = 2, 1024
    gdoc = np.full((B, T), -1)
    pos = np.zeros((B, T), np.int64)
    did = np.full((B, T), -1)
    gdoc[0, :900], pos[0, :900], did[0, :900] = 10, np.arange(900), 0
    gdoc[0, 900:1000], pos[0, 900:1000], did[0, 900:1000] = 12, np.arange(100), 1
    gdoc[1, :600], pos[1, :600], did[1, :600] = 11, np.arange(600), 0
    reg, qq = t1.batch(gdoc, pos, did)
    reg2, qq2 = t2.batch(gdoc, pos, did)
    assert np.array_equal(reg, reg2) and np.array_equal(qq, qq2)
    assert (reg[0, :256] == -1).all() and (reg[0, 900:] == -1).all()  # region 0 / doc 12 region 0: none
    assert len(qq) == 5 and np.allclose(qq[0], q[0])
    assert (reg[1, 256:512] >= 0).all() and (reg[1, :256] == -1).all()
    # SHUF: permutation preserves the multiset of q (T10)
    _, qs = t1.batch(gdoc, pos, did, perm_seed=3)
    assert np.allclose(np.sort(qs, 0), np.sort(qq, 0))
    import shutil
    shutil.rmtree(d)


@test
def t08_no_validation_access_during_search():
    from csr.fsp_lib import Split
    for bad in ("test", "locked"):
        try:
            Split(bad)
            raise AssertionError(f"{bad} loadable")
        except guard.GuardError:
            pass
    root = guard.ROOT
    for f in ("fle_train.py", "fle_latent.py", "fle_draft_train.py"):
        s = (root / f).read_text()
        assert 'Split("val")' not in s and 'Split("test")' not in s and 'Split("calib_sel")' not in s, f
    s = (root / "fle_cache.py").read_text()
    assert 'STAGE3_OPEN.json' in s and 'frozen.json' in s
    s = (root / "fle_evaluate.py").read_text() if (root / "fle_evaluate.py").exists() else ""
    if s:
        assert "STAGE3_OPEN.json" in s and "frozen.json" in s


@test
def t09_inference_without_draft_artifacts():
    import inspect
    import torch
    from csr.lfr_model import LFRConfig, LFRModel
    cfg = LFRConfig(n_layer=3, d_model=64, n_head=2, dense_ffn=128, n_experts=8, expert_ffn=32, shared_ffn=64,
                    vocab_size=512)
    m = LFRModel(cfg, 0).eval()
    sig = inspect.signature(LFRModel.forward)
    assert list(sig.parameters) == ["self", "batch", "ctl"]
    ids = torch.randint(0, 512, (2, 64))
    out = m({"input_ids": ids, "doc_ids": torch.zeros_like(ids), "positions": torch.arange(64).expand(2, 64)})
    assert torch.isfinite(out["hidden"]).all()
    assert not any(n.startswith(("fd_", "tc_")) for n in m.state_dict())
    src = (guard.ROOT / "fle_evaluate.py").read_text() if (guard.ROOT / "fle_evaluate.py").exists() else ""
    assert "draft_forecast" not in src and "RegionCache(" not in src.split("def latent_labels")[0]


@test
def t11_matched_init_and_order():
    import torch
    from csr.lfr_model import LFRConfig, LFRModel
    cfg = LFRConfig(n_layer=3, d_model=64, n_head=2, dense_ffn=128, n_experts=8, expert_ffn=32, shared_ffn=64,
                    vocab_size=512)
    a = LFRModel(cfg, 5).state_dict()
    _ = torch.nn.Linear(64, 16)  # an FD head drawn after the backbone never changes it
    b = LFRModel(cfg, 5).state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)


@test
def t12_unrestricted_routing():
    import torch
    from csr.lfr_model import LFRConfig, LFRModel
    cfg = LFRConfig(n_layer=3, d_model=64, n_head=2, dense_ffn=128, n_experts=8, expert_ffn=32, shared_ffn=64,
                    vocab_size=512)
    m = LFRModel(cfg, 0).eval()
    ids = torch.randint(0, 512, (2, 64))
    out = m({"input_ids": ids, "doc_ids": torch.zeros_like(ids), "positions": torch.arange(64).expand(2, 64)},
            {"collect": True})
    for r in out["records"]:
        ti = r["topi"].reshape(-1, 4)
        assert (ti.sort(-1).values.diff(dim=-1) > 0).all(), "duplicate experts"
        assert (r["p"] > 0).all()  # every expert reachable
    for blk in m.blocks[1:]:
        assert blk.ffn.shared is not None


@test
def t07_exact_resume_cpu_tiny():
    """Tiny-model CPU run of fle_train (FL-EDL with a synthetic table): 4 steps uninterrupted vs 2 + resume 2."""
    import numpy as np
    import torch
    from csr.fle_paths import prereg_sha
    from csr.lfr_lib import main_stream
    work = Path(tempfile.mkdtemp(dir=guard.ROOT / ".tmp/forecast-latent-edl-v1"))
    sp, steps, _ = main_stream(245_000_000)
    docs = sorted({d for st in steps[:4] for r in st for d in r})
    rng = np.random.default_rng(0)
    g, j = [], []
    for d in docs:
        for jj in range(1, int(sp.lengths[d]) // 256 + 1):
            if sp.lengths[d] - jj * 256 >= 32:
                g.append(d)
                j.append(jj)
    np.savez(work / "q.npz", gdoc=np.asarray(g, np.int32), j=np.asarray(j, np.int16),
             q=rng.dirichlet(np.ones(4), size=len(g)).astype(np.float32), K=np.int64(4), L=np.int64(256))
    from csr.fle_lib import QTable
    sha = QTable(work / "q.npz").sha
    cfg = {"run_name": "t07", "seed": 3, "budget_tokens": 245_000_000, "prereg_sha256": prereg_sha(),
           "model": {"n_layer": 3, "d_model": 64, "n_head": 2, "dense_ffn": 128, "n_experts": 8, "expert_ffn": 32,
                     "shared_ffn": 64, "vocab_size": 50304, "top_k": 4, "router": "linear", "teacher": False},
           "cond": {"latent": {"table": str((work / "q.npz").relative_to(guard.ROOT)), "table_sha256": sha,
                               "beta": 1e-2, "layers": [0, 1], "shuffle": False}},
           "train": {"aux_coeff": 0.01, "betas": [0.9, 0.95], "clip": 1.0, "decay_frac": 0.2, "log_every": 1,
                     "lr": 6e-4, "micro_batch_rows": 32, "min_lr": 6e-5, "rows_per_step": 64, "warmup_frac": 0.01,
                     "weight_decay": 0.1, "z_coeff": 0.001}}
    (work / "cfg.json").write_text(json.dumps(cfg))
    env = {**os.environ, "MOE_RUNS_DIR": str(work / "runs"), "MOE_RESEARCH_APPROVE": "cpu-model"}
    (work / "runs").mkdir()
    py = str(guard.ROOT / ".venv/bin/python")

    def go(extra):
        r = subprocess.run([py, "fle_train.py", "--config", str(work / "cfg.json"), "--execute"] + extra, env=env,
                           capture_output=True, text=True, cwd=guard.ROOT)
        return r
    r = go(["--max-steps", "4"])
    assert r.returncode == 75, r.stdout[-2000:] + r.stderr[-2000:]
    runs = sorted((work / "runs").iterdir())
    full = runs[0]
    r = go(["--max-steps", "2"])
    assert r.returncode == 75, r.stderr[-2000:]
    part = sorted((work / "runs").iterdir())[1]
    r = go(["--max-steps", "4", "--run-dir", str(part)])
    assert r.returncode == 75, r.stderr[-2000:]
    la = [json.loads(x) for x in (full / "metrics.jsonl").read_text().splitlines()]
    lb = [json.loads(x) for x in (part / "metrics.jsonl").read_text().splitlines()]
    la = {x["step"]: x for x in la if "step" in x}
    lb = {x["step"]: x for x in lb if "step" in x}
    for s in range(4):
        assert la[s]["lm"] == lb[s]["lm"] and la[s]["fledl"] == lb[s]["fledl"], (s, la[s], lb[s])
    import shutil
    from csr.fle_paths import TMP
    for rr in (full, part):
        shutil.rmtree(TMP / "slots" / rr.name, ignore_errors=True)
    shutil.rmtree(work)


def main():
    guard.check_root()
    assert os.environ.get("CUDA_VISIBLE_DEVICES", "") == "", "CUDA must be hidden"
    from csr.fle_paths import ART
    ok = all(v["ok"] for v in RES.values())
    out = ART / f"cpu_tests_{guard.utc_stamp()}.json"
    guard.atomic_write_json(out, {"time": guard.now(), "all_ok": ok, "results": RES})
    print(json.dumps({"all_ok": ok, "out": str(out)}))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
