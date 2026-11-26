#!/usr/bin/env python
"""csr-latent-forecast-router-distill-v1 CPU correctness suite (protocol 14), run with CUDA hidden.
Writes artifacts/latent-forecast-router-distill-v1/cpu_tests_<stamp>.json; exit 0 iff every test passes."""
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from csr import guard  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

TINY = dict(vocab_size=512, n_layer=3, d_model=64, n_head=2, dense_ffn=128, n_experts=8, top_k=4, expert_ffn=32,
            shared_ffn=64, mem_dim=12, fc_dim=16, tc_hidden=16)
RESULTS = {}


def test(fn):
    name = fn.__name__
    try:
        out = fn()
        RESULTS[name] = {"pass": True, **(out or {})}
    except Exception:  # noqa: BLE001
        RESULTS[name] = {"pass": False, "error": traceback.format_exc()[-2500:]}
    print(("PASS " if RESULTS[name]["pass"] else "FAIL ") + name, flush=True)
    return fn


def tiny(router="student", teacher=True, **kw):
    from csr.lfr_model import LFRConfig, LFRModel
    cfg = LFRConfig(**{**TINY, "router": router, "teacher": teacher, **kw})
    torch.manual_seed(0)
    return LFRModel(cfg, seed=3).double()


def fake_batch(B=2, T=64, docs=((40, 24), (64,)), V=512, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(2, V, (B, T), generator=g)
    doc_ids = torch.full((B, T), -1)
    pos = torch.zeros(B, T, dtype=torch.long)
    for b, lens in enumerate(docs[:B]):
        t = 0
        for j, L in enumerate(lens):
            doc_ids[b, t:t + L] = j
            pos[b, t:t + L] = torch.arange(L)
            t += L
    return ids, doc_ids, pos


def fake_raw(ids, doc_ids, seed=1, V=512):
    g = torch.Generator().manual_seed(seed)
    B, T = ids.shape
    r = {"ens_ids": torch.randint(0, V, (B, T, 4), generator=g), "ens_p": torch.randint(0, 255, (B, T, 4),
                                                                                      generator=g).to(torch.uint8),
         "cand4": torch.randint(0, V, (B, T, 2), generator=g), "cand8": torch.randint(0, V, (B, T, 2), generator=g),
         "code": torch.randint(0, 127, (B, T, 6), generator=g).to(torch.uint8),
         "ent1": torch.randint(0, 255, (B, T, 6), generator=g).to(torch.uint8),
         "p1": torch.randint(0, 255, (B, T, 6), generator=g).to(torch.uint8),
         "score": torch.randint(0, 255, (B, T, 6), generator=g).to(torch.uint8), "src_tok": ids.clone()}
    return r


def feats_of(raw, doc_ids, pos):
    from csr.lfr_lib import build_features
    f = build_features(raw, doc_ids, pos)
    return {k: (v.double() if v.is_floating_point() else v) for k, v in f.items()}


def fwd(m, ids, doc_ids, pos, feats=None, lam=0.9, ctl=None, **extra):
    b = {"input_ids": ids, "doc_ids": doc_ids, "positions": pos, **extra}
    if feats is not None:
        b["fc"] = feats
        b["lam"] = torch.tensor(lam, dtype=torch.float64)
    c = {"teach": feats is not None, "collect": True}
    c.update(ctl or {})
    return m(b, c)


# ------------------------------------------------------------------ T1 causal forecast generation
@test
def T01_causal_forecasts():
    from csr.model import CSRModel, ModelConfig
    from csr.lfr_lib import draft_outputs
    cfg = {"vocab_size": 512, "n_layer": 2, "d_model": 32, "n_head": 2, "dense_ffn": 64, "moe_layers": [],
           "n_experts": 1, "top_k": 1, "expert_ffn": 1, "shared_ffn": 0, "rope_base": 10000.0, "norm_eps": 1e-6,
           "max_seq_len": 64, "init_std": 0.02, "tie_embeddings": True}
    torch.manual_seed(0)
    dr = CSRModel(ModelConfig.from_dict(cfg)).float().eval()
    ids = torch.randint(2, 512, (1, 48))
    L = torch.tensor([48])
    o1 = draft_outputs(dr, ids, L, H=8, topk1=4)
    t = 20
    ids2 = ids.clone()
    ids2[0, t + 1:] = torch.randint(2, 512, (48 - t - 1,))
    o2 = draft_outputs(dr, ids2, L, H=8, topk1=4)
    for k in ("top_ids", "roll"):
        assert torch.equal(o1[k][0, :t + 1], o2[k][0, :t + 1]), k
    for k in ("top_p", "ent1", "score"):
        assert torch.allclose(o1[k][0, :t + 1], o2[k][0, :t + 1], atol=1e-5), k
    # rollout step 1 = argmax of the h1 distribution; greedy rollout k uses only prefix <= t + own proposals
    assert torch.equal(o1["roll"][..., 0], o1["top_ids"][..., 0])
    # feature builder causality: perturb raw fields after t
    ids_, dids, pos = fake_batch()
    raw = fake_raw(ids_, dids)
    f1 = feats_of(raw, dids, pos)
    raw2 = {k: v.clone() for k, v in raw.items()}
    for k in raw2:
        raw2[k][:, t + 1:] = raw2[k][:, t + 1:].flip(1)
    f2 = feats_of(raw2, dids, pos)
    for k in f1:
        assert torch.allclose(f1[k][:, :t + 1].double(), f2[k][:, :t + 1].double()), k
    return {"checked_t": t}


@test
def T02_route_causality():
    m = tiny()
    ids, dids, pos = fake_batch()
    raw = fake_raw(ids, dids)
    t = 30
    o1 = fwd(m, ids, dids, pos, feats_of(raw, dids, pos))
    ids2 = ids.clone()
    ids2[:, t + 1:] = torch.randint(2, 512, ids2[:, t + 1:].shape)
    raw2 = {k: v.clone() for k, v in raw.items()}
    raw2["src_tok"] = ids2.clone()
    for k in raw2:
        if k != "src_tok":
            raw2[k][:, t + 1:] = raw2[k][:, t + 1:].flip(1)
    o2 = fwd(m, ids2, dids, pos, feats_of(raw2, dids, pos))
    for r1, r2 in zip(o1["records"], o2["records"]):
        assert torch.equal(r1["topi"][:, :t + 1], r2["topi"][:, :t + 1])
        assert torch.allclose(r1["p"][:, :t + 1], r2["p"][:, :t + 1], atol=1e-10)
    l1, l2 = m.lm_logits(o1["hidden"]), m.lm_logits(o2["hidden"])
    assert torch.allclose(l1[:, :t + 1], l2[:, :t + 1], atol=1e-9)
    # student-only path too
    s1 = fwd(m, ids, dids, pos)
    s2 = fwd(m, ids2, dids, pos)
    assert torch.allclose(m.lm_logits(s1["hidden"])[:, :t + 1], m.lm_logits(s2["hidden"])[:, :t + 1], atol=1e-9)
    return {}


@test
def T03_split_disjointness():
    from csr.fsp_lib import Split, OPEN_SPLITS
    papers = {}
    for s in OPEN_SPLITS:
        papers[s] = set(Split(s).papers)
    names = list(papers)
    bad = [(a, b, len(papers[a] & papers[b])) for i, a in enumerate(names) for b in names[i + 1:]
           if papers[a] & papers[b]]
    assert not bad, bad
    man = json.loads((ROOT / "artifacts/forecast-specialist-pretraining-v2/data/manifest.json").read_text())
    oa = man.get("overlap_audit") or man.get("audit") or {}
    return {"n_papers": {k: len(v) for k, v in papers.items()}, "manifest_overlap_audit": str(oa)[:300]}


@test
def T04_no_val_test_access():
    from csr.fsp_lib import Split
    try:
        Split("test")
        raise AssertionError("test split loadable")
    except guard.GuardError:
        pass
    import re
    bad = []
    for f in ["lfr_train.py", "src/csr/lfr_model.py", "src/csr/lfr_lib.py", "src/csr/lfr_diag.py", "lfr_cache.py"]:
        src = (ROOT / f).read_text()
        for pat in (r'Split\(\s*"val"', r'Split\(\s*"calib_sel"', r'Split\(\s*"test"', r'"val_tokens', r"locked/"):
            if re.search(pat, src):
                bad.append((f, pat))
    assert not bad, bad
    import lfr_train
    assert set(lfr_train.ALLOWED_SPLITS) == {"main", "calib_search"}
    return {}


@test
def T05_unrestricted_availability():
    m = tiny()
    ids, dids, pos = fake_batch(B=2, T=64)
    raw = fake_raw(ids, dids)
    seen = set()
    for lam in (0.9, 0.45, 0.0):
        o = fwd(m, ids, dids, pos, feats_of(raw, dids, pos), lam=lam)
        for r in o["records"]:
            v = r["p"][dids >= 0]
            assert (v > 0).all() and torch.isfinite(v).all()
            seen |= set(r["topi"][dids >= 0].reshape(-1).tolist())
    # forcing each expert to the top through a router bias reaches every expert
    with torch.no_grad():
        for e in range(8):
            for blk in m.blocks[1:]:
                blk.ffn.router.weight.zero_()
                blk.ffn.router.weight[e] = 50.0 * m.blocks[1].norm2.weight.sign()[:64]
            o = fwd(m, ids, dids, pos)
            seen |= set(o["records"][0]["topi"].reshape(-1).tolist())
    assert seen == set(range(8)), seen
    return {}


@test
def T06_top4_and_shared():
    m = tiny(router="linear", teacher=False)
    ids, dids, pos = fake_batch()
    o = fwd(m, ids, dids, pos)
    for r in o["records"]:
        ti = r["topi"].reshape(-1, 4)
        assert ti.shape[1] == 4 and all(len(set(x)) == 4 for x in ti.tolist())
    blk = m.blocks[1].ffn
    x = torch.randn(10, 64, dtype=torch.float64)
    topi = torch.tensor([[0, 1, 2, 3]] * 10)
    gates = torch.full((10, 4), 0.25, dtype=torch.float64)
    with torch.no_grad():
        w1 = blk.experts.w1.clone()
        blk.experts.w1.zero_()
        y = blk.experts(x, topi, gates) + blk.shared(x)
        assert torch.allclose(y, blk.shared(x)) and blk.shared(x).abs().sum() > 0
        blk.experts.w1.copy_(w1)
    return {}


def _sub_identity(m, li, t):
    """y with r4 replaced by e (same gate) minus reference y equals g4 (E_e(x) - E_r4(x))."""
    ids, dids, pos = fake_batch()
    store = {}
    b = {"input_ids": ids, "doc_ids": dids, "positions": pos, "cf_idx": torch.tensor([t])}
    m(b, {"cf_store": store})
    rec = store[f"rec{li}"]
    ex = m.blocks[li + 1].ffn.experts
    x = rec["x"]
    topi = rec["topi"].clone()
    gates = torch.ones_like(topi, dtype=torch.float64) * rec["g4"].view(-1, 1)
    y_ref = ex(x.double(), topi, gates)
    out = []
    for e in range(8):
        if e in topi[0, :3].tolist():
            continue
        t2 = topi.clone()
        t2[0, 3] = e
        y_sub = ex(x.double(), t2, gates)
        delta = rec["g4"].double() * (ex.one(x, e).double() - ex.one(x, int(rec["r4"][0])).double())
        out.append(float((y_sub - y_ref - delta).abs().max()))
    return max(out)


@test
def T07_exact_substitution():
    from csr.lfr_eval import exact_delta
    m = tiny(router="linear", teacher=False).float()
    err = _sub_identity(m.double(), 1, 10)
    assert err < 1e-9, err
    m = m.float()
    ids, dids, pos = fake_batch(B=1, T=64, docs=((64,),))
    tg = torch.cat([ids[:, 1:], torch.full((1, 1), -100)], 1)
    store = {}
    m({"input_ids": ids, "doc_ids": dids, "positions": pos, "cf_idx": torch.tensor([20])}, {"cf_store": store})
    rec = store["rec1"]
    row = {"input_ids": ids, "doc_ids": dids, "positions": pos, "_targets": tg}
    d = exact_delta(m, row, 1, 20, rec["x"][0], int(rec["r4"][0]), float(rec["g4"][0]), torch.device("cpu"))
    assert abs(float(d[int(rec["r4"][0])])) < 1e-9, d
    return {"max_identity_err": err}


@test
def T08_first_order_vs_exact():
    from csr.lfr_eval import exact_delta, first_order_delta, spearman
    import torch.nn.functional as F
    m = tiny(router="linear", teacher=False).float()
    ids, dids, pos = fake_batch(B=1, T=64, docs=((64,),))
    tg = torch.cat([ids[:, 1:], torch.full((1, 1), -100)], 1)
    t, li = 20, 1
    store = {}
    b = {"input_ids": ids, "doc_ids": dids, "positions": pos, "cf_idx": torch.tensor([t])}
    o = m(b, {"cf_store": store, "tap": True})
    loss = F.cross_entropy(m.lm_logits(o["hidden"]).reshape(-1, 512), tg.reshape(-1), ignore_index=-100,
                           reduction="sum")
    loss.backward()
    rec, gr = store[f"rec{li}"], store[f"grad{li}"]
    eps = 1e-3
    fo = first_order_delta(m.blocks[li + 1].ffn.experts, rec["x"], gr, rec["r4"], rec["g4"] * eps, 1.0)[0]
    row = {"input_ids": ids, "doc_ids": dids, "positions": pos, "_targets": tg}
    ex = exact_delta(m.double(), {k: v for k, v in row.items()}, li, t, rec["x"][0].double(), int(rec["r4"][0]),
                     float(rec["g4"][0]) * eps, torch.device("cpu"))
    rho = spearman(fo.numpy(), ex.numpy())
    rel = float((fo.double() - ex).abs().max() / max(1e-12, float(ex.abs().max())))
    assert rho > 0.95 and rel < 0.05, (rho, rel)
    return {"rho": rho, "rel_err": rel}


@test
def T09_no_same_position_target_leakage():
    import inspect
    from csr.lfr_model import LFRModel
    src = inspect.getsource(LFRModel.forward)
    assert '"targets"' not in src and "_targets" not in src
    m = tiny()
    ids, dids, pos = fake_batch()
    raw = fake_raw(ids, dids)
    f = feats_of(raw, dids, pos)
    o1 = fwd(m, ids, dids, pos, f, targets=torch.zeros_like(ids))
    o2 = fwd(m, ids, dids, pos, f, targets=torch.ones_like(ids))
    for r1, r2 in zip(o1["records"], o2["records"]):
        assert torch.equal(r1["topi"], r2["topi"])
    return {}


@test
def T10_stopgrad_utility_losses():
    from csr.lfr_model import utility_losses, standardize_utility
    m = tiny()
    ids, dids, pos = fake_batch()
    raw = fake_raw(ids, dids)
    f = feats_of(raw, dids, pos)
    store = {}
    cf = torch.tensor([3, 10, 70])
    b = {"input_ids": ids, "doc_ids": dids, "positions": pos, "fc": f, "lam": torch.tensor(0.9, dtype=torch.float64),
         "cf_idx": cf}
    m(b, {"teach": True, "cf_store": store})
    m.zero_grad()
    fl = {k: v.reshape(-1, *v.shape[2:]).index_select(0, cf) for k, v in f.items()}
    tgt = standardize_utility(torch.randn(3, 8, dtype=torch.float64))
    lt, ls = utility_losses(m, store["rec0"], fl, tgt, "rank", 0)
    (lt + ls).backward()
    got = {n for n, p in m.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
    allowed = lambda n: n.startswith(("tc_", "student")) or n.endswith("router.weight")
    assert got and all(allowed(n) for n in got), sorted(got)
    return {"grad_params": sorted(got)[:8]}


@test
def T11_p3_student_without_draft_code():
    code = ("import sys; sys.path.insert(0,'src'); import csr.lfr_model as M; "
            "bad=[k for k in sys.modules if 'lfr_lib' in k or 'lfr_cache' in k or 'fsp_lib' in k or 'draft' in k]; "
            "print(bad)")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    assert r.returncode == 0 and r.stdout.strip() == "[]", (r.stdout, r.stderr[-500:])
    m = tiny().strip_teacher()
    assert not any(n.startswith("tc_") for n, _ in m.named_parameters())
    ids, dids, pos = fake_batch()
    o = fwd(m, ids, dids, pos)
    assert torch.isfinite(o["hidden"]).all()
    return {}


@test
def T12_inference_export():
    m = tiny()
    m.strip_teacher()
    ids, dids, pos = fake_batch()
    b = {"input_ids": ids, "doc_ids": dids, "positions": pos}
    o = m(b, {"teach": True})  # teach requested but no teacher exists: student path, no fc needed
    assert torch.isfinite(o["hidden"]).all() and not m.cfg.teacher
    return {}


@test
def T13_no_domain_label_in_router():
    m = tiny()
    ids, dids, pos = fake_batch()
    raw = fake_raw(ids, dids)
    f = feats_of(raw, dids, pos)
    o1 = fwd(m, ids, dids, pos, f, domain=torch.zeros_like(ids), seq_id=torch.zeros_like(ids))
    o2 = fwd(m, ids, dids, pos, f, domain=torch.full_like(ids, 3), seq_id=torch.zeros_like(ids))
    assert torch.allclose(o1["hidden"], o2["hidden"])
    import inspect
    from csr.lfr_model import LFRModel
    src = inspect.getsource(LFRModel.forward)
    assert src.count('batch["domain"]') == 1 and 'ctl.get("edl")' in src
    return {}


@test
def T14_checkpoint_store_cpu():
    import shutil
    from csr import lfr_ckpt
    root = ROOT / ".tmp/latent-forecast-router-distill-v1/test_slots"
    if root.exists():
        shutil.rmtree(root)
    st = lfr_ckpt.ResumeStore("T14run", "T14", root=root)
    old_ledger = lfr_ckpt.LEDGER
    lfr_ckpt.LEDGER = root / "ledger.tsv"
    try:
        for s in (2, 4, 6):
            pl = {"run_id": "T14run", "model": {"a": torch.ones(3) * s}, "st": {"next_step": s, "seen": 10 * s}}
            st.save(pl, s, 10 * s)
        idx = st.index()
        assert [e["step"] for e in idx["entries"]] == [4, 6]
        assert not (root / "T14run" / "resume_s0000002").exists()
        p6 = root / "T14run" / "resume_s0000006" / "state.pt"
        with open(p6, "r+b") as f:
            f.seek(20)
            b = f.read(1)
            f.seek(20)
            f.write(bytes([b[0] ^ 0xFF]))
        e = st.latest_verified(log=lambda m_: None)
        assert e["step"] == 4
        st.prune_after_complete()
        assert not list((root / "T14run").glob("resume_s*"))
    finally:
        lfr_ckpt.LEDGER = old_ledger
        shutil.rmtree(root)
    return {"note": "GPU resume equivalence: lfr_resume_check.py"}


@test
def T15_matched_init_and_order():
    from csr.lfr_model import LFRConfig, LFRModel
    a = LFRModel(LFRConfig(router="linear", teacher=False), seed=5)
    b = LFRModel(LFRConfig(router="student", teacher=True, agg="xattn"), seed=5)
    sa, sb = a.state_dict(), b.state_dict()
    names = a.backbone_names()
    assert names == b.backbone_names()
    assert all(torch.equal(sa[n], sb[n]) for n in names)
    c = LFRModel(LFRConfig(router="linear", teacher=False), seed=6)
    assert not torch.equal(sa["tok_emb.weight"], c.state_dict()["tok_emb.weight"])
    pc = a.param_counts()
    from csr.lfr_lib import main_stream, BUDGET
    sp, s1, t1 = main_stream(BUDGET["S1"])
    _, s2, t2 = main_stream(BUDGET["S2"])
    _, sf, tf = main_stream(BUDGET["FINAL"])
    assert s2[:len(s1)] == s1 and sf[:len(s2)] == s2
    dom = np.zeros(5)
    for r in s1:
        for row in r:
            for d in row:
                dom[sp.domains[d]] += sp.lengths[d] - 1
    assert dom.max() - dom.min() <= 1100 * 2, dom
    return {"token4_params": pc, "S1_steps": len(s1), "S2_steps": len(s2), "FINAL_steps": len(sf),
            "S1_domain_tokens": dom.tolist(), "student_params": LFRModel(LFRConfig(router="student"), 0).param_counts()}


@test
def T16_matched_tokens_steps_experts():
    import lfr_driver as D
    cfgs = D.all_condition_configs_for_test()
    keys = {json.dumps({k: c[k] for k in ("budget_tokens", "train", "seed")}, sort_keys=True) for c in cfgs}
    assert len(keys) == 1, keys
    bb = {json.dumps({k: v for k, v in c["model"].items() if k in ("n_layer", "d_model", "n_head", "dense_ffn",
                                                                   "n_experts", "top_k", "expert_ffn", "shared_ffn",
                                                                   "vocab_size")}, sort_keys=True) for c in cfgs}
    assert len(bb) == 1, bb
    return {"n_conditions": len(cfgs)}


@test
def T17_edl():
    from csr.fsp_model import edl_loss
    p = torch.tensor([[0.7, 0.3], [0.5, 0.5], [0.2, 0.8], [0.4, 0.6]], dtype=torch.float64)
    lab = torch.tensor([0, 0, 1, 1])
    sid = torch.tensor([0, 1, 2, 3])
    nonpad = torch.ones(4, dtype=torch.bool)
    P0 = (p[0] + p[1]) / 2
    P1 = (p[2] + p[3]) / 2
    M_ = (P0 + P1) / 2
    kl = lambda a, b: (a * (a / b).log()).sum()
    js = 0.5 * kl(P0, M_) + 0.5 * kl(P1, M_)
    v = edl_loss(p, lab, sid, nonpad)
    assert abs(float(v) - float(-torch.log(js + 1e-8))) < 1e-9
    assert float(edl_loss(p, torch.zeros(4, dtype=torch.long), sid, nonpad)) == 0.0
    return {}


@test
def T18_deterministic_selection():
    from csr import lfr_search as S
    t4 = {"ce": 4.0, "audit": {"fsi_domain": 0.01, "mi_mean": 0.05}, "d": {"p1end_40": {"ce_student": 4.5},
                                                                            "end": {"ce_student": 4.0}}, "tok_s": 1e5}
    edl = {"ce": 3.99}
    cands = []
    for i in range(5):
        cid, kn, why = S.next_candidate(cands, t4)
        assert cid == f"P0{i + 1}"
        cands.append({"id": cid, "knobs": kn, "ce": 4.0 - 0.001 * i, "flags": {}, "audit": {"fsi_domain": 0.02}})
    cands[-1]["flags"] = {"D": {}}
    r1 = S.next_candidate(cands, t4)
    r2 = S.next_candidate(json.loads(json.dumps(cands)), t4)
    assert r1 == r2 and r1[1]["tau_t"] == 0.5, r1
    adv = S.advance(cands, t4)
    assert [a for a, _ in adv] == ["P05", "P04", "P03", "P02"]
    sel = S.select({"P01": {"ce_sel": [3.9, 3.95], "fsi": [0.02, 0.03], "H": [False, False]},
                    "P02": {"ce_sel": [3.8, 3.85], "fsi": [0.0, 0.0], "H": [False, False]}}, 0.01)
    assert sel == ("P01", True), sel
    c = dict(cands[0])
    c["d"] = {"p1end": {"ce_teacher": 4.6, "ce_teacher_shuffled": 4.6005, "probe_ratio": 1.0}}
    f, _ = S.diagnose(c, t4, edl)
    assert "B" in f, f
    return {}


@test
def T19_audit_correctness():
    from csr.lfr_audit import fsi_from_deltas, routing_metrics, mutual_info
    rng = np.random.default_rng(0)
    N, K, E, L = 20000, 4, 8, 2
    lab = rng.integers(0, K, N)
    base = rng.gamma(2.0, 1.5, N)
    groups = {k: {0: [2 * k, 2 * k + 1]} for k in range(K)}
    planted = {k: 0.01 + 0.2 * (lab == k) + rng.normal(0, 0.01, N) for k in range(K)}
    rand = {k: 0.01 + rng.normal(0, 0.01, N) for k in range(K)}
    f1 = fsi_from_deltas(base, planted, lab, K, groups, L)
    f0 = fsi_from_deltas(base, rand, lab, K, groups, L)
    fs = fsi_from_deltas(base, planted, rng.permutation(lab), K, groups, L)
    assert f1["fsi"] > 0.1 and abs(f0["fsi"]) < 0.01 and abs(fs["fsi"]) < 0.02, (f1["fsi"], f0["fsi"], fs["fsi"])
    topi = np.stack([np.stack([2 * lab, 2 * lab + 1, (2 * lab + 2) % E, (2 * lab + 3) % E], 1)] * L, 1)
    rm = routing_metrics(topi, lab, K, E)
    c = np.zeros((K, E))
    for i in range(N):
        for e in topi[i, 0]:
            c[lab[i], e] += 1
    assert abs(rm["mi"][0] - mutual_info(c)) < 1e-9 and rm["mi"][0] > rm["mi_null"][0] + 0.1
    return {"fsi_planted": f1["fsi"], "fsi_random": f0["fsi"], "fsi_shuffled_labels": fs["fsi"]}


@test
def T20_load_preserving_null():
    from csr.lfr_audit import routing_metrics
    rng = np.random.default_rng(1)
    topi = rng.integers(0, 8, (5000, 2, 4))
    lab = rng.integers(0, 3, 5000)
    rm = routing_metrics(topi, lab, 3, 8)
    shuf = topi[np.random.default_rng(0).permutation(5000)]
    for li in range(2):
        assert np.array_equal(np.bincount(topi[:, li].reshape(-1), minlength=8),
                              np.bincount(shuf[:, li].reshape(-1), minlength=8))
    return {"mi": rm["mi_mean"], "mi_null": rm["mi_null_mean"]}


@test
def T21_shuffled_forecast_mapping():
    from csr.lfr_lib import shuffle_donors
    lengths = np.asarray([1024, 1024, 500, 1024, 300, 1024])
    rows = [[0], [1], [2, 4], [3], [5]]
    mp = shuffle_donors(rows, lengths)
    assert set(mp) == set(range(6)) and all(mp[k] != k for k in mp) and sorted(mp.values()) == list(range(6))
    assert shuffle_donors([[0]], lengths) == {}
    return {"map": {int(k): int(v) for k, v in mp.items()}}


@test
def T22_chunked_ema():
    from csr.lfr_model import causal_ema
    from csr.fsp_model import ema_matrix
    ids, dids, pos = fake_batch(B=2, T=256, docs=((40, 100, 116), (256,)))
    v = torch.randn(2, 256, 5, dtype=torch.float64)
    for hl in (16.0, 128.0, 1024.0):
        a = causal_ema(v, dids, pos, hl, C=64)
        W = ema_matrix(dids, pos, hl, torch.float64)
        b = torch.bmm(W, v) * (dids >= 0).unsqueeze(-1)
        assert torch.allclose(a, b, atol=1e-6), (hl, float((a - b).abs().max()))  # reference uses fp32 pow
    return {}


@test
def T23_cf_sample_and_phases():
    from csr.lfr_lib import cf_uniform, phase_at
    g = np.arange(2000)
    p = np.arange(2000) % 1024
    u1, u2 = cf_uniform(g, p), cf_uniform(g, p)
    assert np.array_equal(u1, u2) and 0.0 <= u1.min() and u1.max() < 1.0
    big = cf_uniform(np.repeat(np.arange(200), 1000), np.tile(np.arange(1000), 200))
    assert abs((big < 0.075).mean() - 0.075) < 0.005
    assert phase_at(0.1)[1] == 1 and phase_at(0.5)[1] == 2 and abs(phase_at(0.5)[0] - 0.45) < 1e-9
    assert phase_at(0.6)[1] == 3 and phase_at(0.6)[0] == 0.0
    assert phase_at(0.45, "30/20/50")[1] == 2 and phase_at(0.5, "30/20/50")[1] == 3
    return {}


@test
def T24_cache_compression():
    from csr.lfr_lib import compress_ensemble
    g = torch.Generator().manual_seed(0)
    N, D = 50, 6
    ids = torch.stack([torch.randperm(100, generator=g)[:16] for _ in range(N * D)]).view(N, D, 16)
    ids[:, :, 0] = 7  # all drafts agree on token 7 as top-1
    p = torch.softmax(torch.randn(N, D, 16, generator=g), -1).sort(-1, descending=True).values
    r4 = torch.randint(0, 3, (N, D), generator=g)
    r8 = torch.full((N, D), 5)
    out, st = compress_ensemble(ids, p, torch.rand(N, D), -torch.rand(N, D) * 5, r4, r8)
    assert (out["ens_ids"][:, 0] == 7).all() and st["h1_argmax_in_ens_top4"] == 1.0 and st["h8_in_cands"] == 1.0
    code = out["code"].astype(np.int64)
    assert ((code & 7) == 0).all() and (((code >> 5) & 3) == 0).all()
    # h4 candidate pair covers the two most frequent ids; codes consistent
    c4 = out["cand4"].astype(np.int64)
    for i in range(N):
        vals, cnt = np.unique(r4[i].numpy(), return_counts=True)
        assert cnt.max() == (r4[i].numpy() == c4[i, 0]).sum()
    return {"stats": st}


def main():
    guard.check_root()
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
        print("CUDA must be hidden for the CPU suite")
        sys.exit(2)
    from csr.lfr_lib import ART
    ok = all(r["pass"] for r in RESULTS.values())
    out = ART / f"cpu_tests_{guard.utc_stamp()}.json"
    guard.atomic_write_json(out, {"time": guard.now(), "all_pass": ok, "results": RESULTS})
    print(json.dumps({"all_pass": ok, "n": len(RESULTS), "failed": [k for k, v in RESULTS.items() if not v["pass"]],
                      "out": str(out.relative_to(ROOT))}))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
