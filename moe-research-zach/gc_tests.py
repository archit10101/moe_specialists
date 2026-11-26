#!/usr/bin/env python
"""csr-gradient-consistent-latent-edl-v1 CPU tests (prereg section 7), CUDA hidden.
Writes artifacts/gc-edl-v1/cpu_tests_<UTC>.json.  Usage: CUDA_VISIBLE_DEVICES= .venv/bin/python gc_tests.py [names]"""
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
TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


def run_test(fn):
    t0 = time.time()
    try:
        fn()
        RES[fn.__name__] = {"ok": True, "s": round(time.time() - t0, 2)}
    except Exception:
        RES[fn.__name__] = {"ok": False, "err": traceback.format_exc()[-2500:]}
    print(fn.__name__, RES[fn.__name__]["ok"], flush=True)


def _tiny_scout():
    import torch
    from csr.fsp_lib import DRAFT_CFG
    from csr.model import CSRModel, ModelConfig
    torch.manual_seed(0)
    cfg = ModelConfig.from_dict({**DRAFT_CFG, "vocab_size": 64, "n_layer": 2, "d_model": 16, "n_head": 2,
                                 "dense_ffn": 24, "max_seq_len": 64})
    m = CSRModel(cfg).double().eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def _explicit_grads(m, ids):
    import torch
    from csr.fsp_lib import draft_hidden
    w = m.blocks[-1].ffn.w2.weight
    out = []
    for b in range(ids.shape[0]):
        w.requires_grad_(True)
        x = ids[b:b + 1]
        pos = torch.arange(x.shape[1]).expand(1, -1)
        h = draft_hidden(m, x, torch.zeros_like(pos), pos)
        lg = h[:, :-1] @ m.tok_emb.weight.t()
        ce = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), x[:, 1:].reshape(-1))
        (g,) = torch.autograd.grad(ce, w)
        out.append(g.detach().clone())
        w.requires_grad_(False)
    return out


@test
def g1_sketch_equals_explicit_projection():
    import numpy as np
    import torch
    from csr.gc_lib import Sketch, scout_fingerprints
    m = _tiny_scout()
    ids = torch.randint(0, 64, (5, 20), generator=torch.Generator().manual_seed(1))
    sk = Sketch(d_in=24, d_out=16, m=8, seed=3)
    g, nrm, ce = scout_fingerprints(m, ids, sk)
    P = sk.explicit()
    G = _explicit_grads(m, ids)
    for b in range(5):
        ref = P @ G[b].numpy().reshape(-1)
        err = np.abs(g[b].numpy() - ref).max() / max(1e-30, np.abs(ref).max())
        assert err < 1e-10, (b, err)
    # the explicit projection has exactly one +-1 per column (CountSketch)
    assert (np.abs(P).sum(0) == 1).all()


@test
def g2_gram_norm_equals_frobenius():
    import numpy as np
    import torch
    from csr.gc_lib import Sketch, scout_fingerprints
    m = _tiny_scout()
    ids = torch.randint(0, 64, (4, 20), generator=torch.Generator().manual_seed(2))
    _, nrm, _ = scout_fingerprints(m, ids, Sketch(24, 16, 8))
    G = _explicit_grads(m, ids)
    for b in range(4):
        assert abs(float(nrm[b]) - float(G[b].norm())) / float(G[b].norm()) < 1e-10


@test
def g3_deterministic_and_batch_tolerant():
    import torch
    from csr.gc_lib import Sketch, scout_fingerprints
    m = _tiny_scout()
    sk = Sketch(24, 16, 8)
    ids = torch.randint(0, 64, (3, 20), generator=torch.Generator().manual_seed(4))
    a = scout_fingerprints(m, ids, sk)
    b = scout_fingerprints(m, ids, sk)
    assert all(torch.equal(x, y) for x, y in zip(a, b)), "replay not bitwise identical"
    c = scout_fingerprints(m, ids[1:2], sk)
    assert (a[0][1] - c[0][0]).abs().max() / a[0][1].abs().max() < 1e-10


@test
def g4_tokens_outside_view_do_not_matter():
    import numpy as np
    import torch
    from csr.gc_lib import Sketch, scout_fingerprints, view_starts
    m = _tiny_scout()
    sk = Sketch(24, 16, 8)
    rng = np.random.default_rng(5)
    V = 16
    N = 120
    paper = rng.integers(0, 64, N)
    st = view_starts(N, V)
    assert len(st) == 4 and all(st[i + 1] - st[i] >= 2 * V for i in range(3)), st
    inside = np.zeros(N, bool)
    for s in st:
        inside[s:s + V] = True
    p2 = paper.copy()
    p2[~inside] = rng.integers(0, 64, (~inside).sum())
    f1 = scout_fingerprints(m, torch.as_tensor(np.stack([paper[s:s + V] for s in st])), sk)[0]
    f2 = scout_fingerprints(m, torch.as_tensor(np.stack([p2[s:s + V] for s in st])), sk)[0]
    assert torch.equal(f1, f2)
    p3 = paper.copy()
    p3[st[1] + 3] = (p3[st[1] + 3] + 1) % 64
    f3 = scout_fingerprints(m, torch.as_tensor(np.stack([p3[s:s + V] for s in st])), sk)[0]
    assert torch.equal(f1[0], f3[0]) and not torch.equal(f1[1], f3[1])
    # view rule: separation >= V everywhere
    for n in range(256, 5000, 37):
        s = view_starts(n, 256)
        assert len(s) <= 4 and all(s[i + 1] - s[i] >= 512 for i in range(len(s) - 1)) and s[-1] + 256 <= n


@test
def l1_label_firewall():
    py = str(guard.ROOT / ".venv/bin/python")
    code = ("import sys; sys.path.insert(0,'src'); from csr.gc_lib import install_label_firewall as f; f()\n"
            "import numpy as np\n"
            "np.load('artifacts/gc-edl-v1/discovery/inputs_calib_search.npz')['nv']\n"
            "bad=['artifacts/forecast-specialist-pretraining-v2/data/main_docs.json',"
            "'artifacts/latent-forecast-router-distill-v1/labels/windows_calib.npz',"
            "'artifacts/gc-edl-v1/sealed/opaque_map_main.json']\n"
            "n=0\n"
            "for b in bad:\n"
            "    try:\n        open(b).read(1)\n    except PermissionError:\n        n+=1\n"
            "print('BLOCKED', n)\n")
    r = subprocess.run([py, "-c", code], capture_output=True, text=True, cwd=guard.ROOT,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    assert "BLOCKED 3" in r.stdout, r.stdout + r.stderr
    for f in ("gc_fingerprint.py", "gc_discover.py"):
        src = (guard.ROOT / f).read_text()
        for bad in ("Split(", ".domains", "_docs.json", "true_domain", "lfr_audit", "windows_", "opaque_map",
                    "fle_labels", "\"domain\"", "'domain'"):
            assert bad not in src, (f, bad)
        assert "install_label_firewall()" in src, f


@test
def l2_export_contains_no_labels():
    import json as _j
    import numpy as np
    from csr.fsp_lib import Split
    for s in ("main", "calib_search", "calib_sel"):
        z = np.load(guard.ROOT / f"artifacts/gc-edl-v1/discovery/inputs_{s}.npz")
        assert set(z.files) == {"opaque", "nv", "views", "starts", "paper_len", "n_seg"}, z.files
        op = [str(o) for o in z["opaque"]]
        assert all(len(o) == 20 for o in op) and op == sorted(op)
        papers = set(Split(s).papers)
        assert not (set(op) & papers)
        assert z["views"].dtype == np.uint16 and z["views"].shape[1:] == (4, 256)
        omap = _j.loads((guard.ROOT / f"artifacts/gc-edl-v1/sealed/opaque_map_{s}.json").read_text())
        assert set(omap) == set(op) and set(omap.values()) == papers



@test
def l3_split_isolation():
    from csr.fsp_lib import Split
    from csr.gc_lib import disc_part, opaque_id
    ps = {s: set(Split(s).papers) for s in ("main", "calib_search", "calib_sel", "val", "draft_shared", "draft_cont")}
    names = list(ps)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert not (ps[names[i]] & ps[names[j]]), (names[i], names[j])
    parts = [disc_part(opaque_id(p)) for p in sorted(ps["main"])]
    f = sum(1 for x in parts if x == 0) / len(parts)
    assert 0.78 < f < 0.82, f
    assert parts == [disc_part(opaque_id(p)) for p in sorted(ps["main"])]


def _synthetic_table(work, K=4, seed=0):
    import numpy as np
    z = np.load(guard.ROOT / "artifacts/gc-edl-v1/discovery/inputs_main.npz")
    rng = np.random.default_rng(seed)
    q = rng.dirichlet(np.ones(K), size=len(z["opaque"])).astype(np.float32)
    q[z["nv"] == 0] = 0
    p = Path(work) / "q.npz"
    np.savez(p, opaque=z["opaque"], q=q)
    return p


@test
def q1_q_constant_within_paper_and_batch():
    import numpy as np
    from csr.fsp_lib import Split, make_batch
    from csr.gc_lib import PaperQ
    from csr.lfr_lib import main_stream
    work = Path(tempfile.mkdtemp(dir=guard.ROOT / ".tmp/gc-edl-v1"))
    sp, steps, _ = main_stream(245_000_000)
    pq = PaperQ(_synthetic_table(work), sp)
    by = {}
    for di, p in enumerate(sp.papers[:50000]):
        by.setdefault(p, []).append(di)
    for p, ds in by.items():
        assert all(np.array_equal(pq.q_of_doc[ds[0]], pq.q_of_doc[d]) for d in ds)
    npb = make_batch(steps[0][:32], sp)
    qs = pq.qseq(npb)
    sid, gd = npb["seq_id"].reshape(-1), npb["gdoc"].reshape(-1)
    for s in range(qs.shape[0]):
        g = np.unique(gd[sid == s])
        assert len(g) == 1 and np.array_equal(qs[s], pq.q_of_doc[g[0]])
    import shutil
    shutil.rmtree(work)
    assert pq.coverage > 0.95


@test
def q2_gc_loss_independent_of_domain_labels():
    import inspect

    import numpy as np
    from csr.fsp_lib import Split
    from csr.gc_lib import PaperQ, gc_edl_loss
    work = Path(tempfile.mkdtemp(dir=guard.ROOT / ".tmp/gc-edl-v1"))
    sp = Split("main")
    t = _synthetic_table(work)
    a = PaperQ(t, sp).q_of_doc
    sp.domains = np.random.default_rng(0).permutation(sp.domains)
    sp.shuffled = np.zeros_like(sp.shuffled)
    b = PaperQ(t, sp).q_of_doc
    assert np.array_equal(a, b)
    assert list(inspect.signature(gc_edl_loss).parameters) == ["p", "qseq", "seq_id", "nonpad", "eps", "min_mass"]
    src = (guard.ROOT / "gc_train.py").read_text()
    gc_part = src[src.index("def gc_aux"):src.index("def gc_beta")]
    assert "domain" not in gc_part
    import shutil
    shutil.rmtree(work)


@test
def e1_soft_loss_equals_edl_onehot():
    import torch
    from csr.fsp_model import edl_loss
    from csr.gc_lib import gc_edl_loss
    g = torch.Generator().manual_seed(0)
    for trial in range(5):
        S, E = 12, 8
        lens = torch.randint(5, 40, (S,), generator=g)
        sid = torch.repeat_interleave(torch.arange(S), lens)
        N = len(sid)
        p = torch.softmax(torch.randn(N, E, generator=g, dtype=torch.float64), -1)
        lab_seq = torch.randint(0, 5, (S,), generator=g)
        lab_seq[0] = 4
        lab_seq[lab_seq == 2] = 3  # domain 2 absent
        nonpad = torch.rand(N, generator=g) > 0.05
        a = edl_loss(p, lab_seq[sid], sid, nonpad)
        q = torch.nn.functional.one_hot(lab_seq, 5).double()
        b = gc_edl_loss(p, q, sid, nonpad)
        assert abs(float(a) - float(b)) < 1e-9, (trial, float(a), float(b))


@test
def e2_gradients_reach_router_probs_only():
    import torch
    from csr.gc_lib import gc_edl_loss
    g = torch.Generator().manual_seed(1)
    sid = torch.repeat_interleave(torch.arange(6), 10)
    logits = torch.randn(60, 8, generator=g, requires_grad=True)
    q = torch.softmax(torch.randn(6, 4, generator=g), -1)
    loss = gc_edl_loss(torch.softmax(logits, -1), q, sid, torch.ones(60, dtype=torch.bool))
    loss.backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0 and not q.requires_grad and q.grad is None
    assert torch.isfinite(loss)
    # identical q for every sequence -> all groups equal -> JSD 0 -> finite clamp
    l2 = gc_edl_loss(torch.softmax(logits, -1), torch.ones(6, 4) / 4, sid, torch.ones(60, dtype=torch.bool))
    assert torch.isfinite(l2)


@test
def p1_prototypes_balanced_and_no_collapse():
    import numpy as np
    import torch
    from csr.gc_lib import fit_prototypes, paper_q, sinkhorn, view_logits
    s = torch.rand(256, 4, generator=torch.Generator().manual_seed(0)) * 2 - 1  # cosine-range scores
    t = sinkhorn(s)
    assert torch.allclose(t.sum(1), torch.ones(256), atol=1e-5)
    assert (t.sum(0) / 64 - 1).abs().max() < 0.2, t.sum(0)  # 3 SwAV iterations: approximately balanced
    rng = np.random.default_rng(0)
    P, K = 3000, 4
    cen = rng.normal(size=(K, 16)) * 2
    lab = rng.integers(0, K, P)
    X = (cen[lab][:, None, :] + rng.normal(size=(P, 4, 16))).astype(np.float32)
    Fh = rng.normal(size=(P, 4, 8))
    Fh /= np.linalg.norm(Fh, axis=-1, keepdims=True)
    nv = np.full(P, 4)
    net, hist = fit_prototypes(X, nv, Fh.astype(np.float32), K, 2.0, 0, epochs=15, batch=512)
    q = paper_q(view_logits(net, X), nv)
    hard = q.argmax(1)
    frac = np.bincount(hard, minlength=K) / P
    assert frac.min() > 0.1, frac
    from csr.gc_lib import matched_kappa
    kap, _ = matched_kappa(lab, hard, K)
    assert kap > 0.8, kap


@test
def r1_exact_resume_cpu_tiny_and_inference_without_gc_artifacts():
    import numpy as np
    import torch
    from csr.gc_lib import PaperQ
    from csr.gc_paths import TMP, prereg_sha
    from csr.lfr_lib import main_stream
    work = Path(tempfile.mkdtemp(dir=TMP))
    sp, steps, _ = main_stream(245_000_000)
    t = _synthetic_table(work)
    sha = guard.sha256_file(t)
    cfg = {"run_name": "r1", "seed": 3, "budget_tokens": 245_000_000, "prereg_sha256": prereg_sha(),
           "model": {"n_layer": 3, "d_model": 64, "n_head": 2, "dense_ffn": 128, "n_experts": 8, "expert_ffn": 32,
                     "shared_ffn": 64, "vocab_size": 50304, "top_k": 4, "router": "linear", "teacher": False},
           "cond": {"paper_latent": {"table": str(t.relative_to(guard.ROOT)), "table_sha256": sha, "beta": 1e-2}},
           "milestones": {"m4": 4}, "diag_every": 2,
           "train": {"aux_coeff": 0.01, "betas": [0.9, 0.95], "clip": 1.0, "decay_frac": 0.2, "log_every": 1,
                     "lr": 6e-4, "micro_batch_rows": 32, "min_lr": 6e-5, "rows_per_step": 64, "warmup_frac": 0.01,
                     "weight_decay": 0.1, "z_coeff": 0.001}}
    (work / "cfg.json").write_text(json.dumps(cfg))
    env = {**os.environ, "MOE_RUNS_DIR": str(work / "runs"), "MOE_RESEARCH_APPROVE": "cpu-model",
           "CUDA_VISIBLE_DEVICES": ""}
    (work / "runs").mkdir()
    py = str(guard.ROOT / ".venv/bin/python")

    def go(extra):
        return subprocess.run([py, "gc_train.py", "--config", str(work / "cfg.json"), "--execute"] + extra, env=env,
                              capture_output=True, text=True, cwd=guard.ROOT)
    r = go(["--max-steps", "5"])
    assert r.returncode == 75, r.stdout[-3000:] + r.stderr[-3000:]
    full = sorted((work / "runs").iterdir())[0]
    r = go(["--max-steps", "2"])
    assert r.returncode == 75, r.stderr[-2000:]
    part = sorted((work / "runs").iterdir())[1]
    r = go(["--max-steps", "5", "--run-dir", str(part)])
    assert r.returncode == 75, r.stderr[-2000:]
    la = {x["step"]: x for x in map(json.loads, (full / "metrics.jsonl").read_text().splitlines()) if "step" in x}
    lb = {x["step"]: x for x in map(json.loads, (part / "metrics.jsonl").read_text().splitlines()) if "step" in x}
    for s in range(5):
        assert la[s]["lm"] == lb[s]["lm"] and la[s]["gcedl"] == lb[s]["gcedl"], (s, la[s], lb[s])
    diag = [json.loads(x)["diag"] for x in (full / "metrics.jsonl").read_text().splitlines() if "diag" in x]
    assert diag and all(d["gn_aux"] > 0 for d in diag)
    # inference: milestone holds only model weights; it loads and runs with every GC artifact access blocked
    from csr.lfr_model import LFRConfig, LFRModel
    ck = torch.load(full / "ckpt_m4" / "state.pt", map_location="cpu", weights_only=False)
    ref = LFRModel(LFRConfig.from_dict(cfg["model"]), 0).state_dict()
    assert set(ck["model"]) == set(ref)
    code = ("import sys,torch; sys.path.insert(0,'src')\n"
            "def hook(e,a):\n"
            "    if e=='open' and a and any(s in str(a[0]) for s in ('gc-edl-v1','bge-small','draft_G_s11','q.npz')):\n"
            "        raise PermissionError(str(a[0]))\n"
            "from csr.lfr_model import LFRConfig, LFRModel\n"
            f"ck=torch.load('{full}/ckpt_m4/state.pt',map_location='cpu',weights_only=False)\n"
            "sys.addaudithook(hook)\n"
            "m=LFRModel(LFRConfig.from_dict({**ck['model_cfg'],'teacher':False}),0); m.load_state_dict(ck['model'])\n"
            "ids=torch.randint(0,50304,(1,32)); pos=torch.arange(32)[None]\n"
            "o=m({'input_ids':ids,'doc_ids':torch.zeros_like(ids),'positions':pos}); print('OK', o['hidden'].shape)\n")
    r = subprocess.run([py, "-c", code], capture_output=True, text=True, cwd=guard.ROOT, env=env)
    assert "OK" in r.stdout, r.stdout + r.stderr
    import shutil
    for rr in (full, part):
        shutil.rmtree(TMP / "slots" / rr.name, ignore_errors=True)
    shutil.rmtree(work)


def main():
    guard.check_root()
    assert os.environ.get("CUDA_VISIBLE_DEVICES", "") == "", "CUDA must be hidden"
    sel = set(sys.argv[1:])
    for fn in TESTS:
        if not sel or fn.__name__ in sel or any(fn.__name__.startswith(s) for s in sel):
            run_test(fn)
    from csr.gc_paths import ART
    ok = all(v["ok"] for v in RES.values())
    out = ART / f"cpu_tests_{guard.utc_stamp()}.json"
    guard.atomic_write_json(out, {"time": guard.now(), "all_ok": ok, "selected": sorted(sel), "results": RES})
    print(json.dumps({"all_ok": ok, "out": str(out)}))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
