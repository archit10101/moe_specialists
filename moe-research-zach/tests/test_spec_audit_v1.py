"""specialization-audit-v1 tests (docs/SPECIALIZATION_AUDIT_V1_PREREG.md). CPU only, CUDA hidden.

Mapping, common masks, label isolation, deterministic clustering/permutations/bootstrap, joint distributions,
counterfactual replacement (exactly top-k, pool membership, shared expert, identity when the expert is not selected,
equality with a brute-force full forward), no parameter mutation, document-level split isolation, and the in-project
numerical routines (Hungarian, logistic regression, TF-IDF/SVD, weighted cosine, interaction estimator)."""
import inspect
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from csr import specaudit as sa  # noqa: E402
from csr import audit  # noqa: E402
from csr.model import CSRModel, ModelConfig, RoutingConfig  # noqa: E402

AUD = ROOT / "artifacts" / "specialization-audit-v1"


# ------------------------------------------------------------------ tiny model fixture
def tiny(pool=True, seed=0):
    torch.manual_seed(seed)
    cfg = ModelConfig(vocab_size=97, n_layer=4, d_model=32, n_head=4, dense_ffn=64, moe_layers=(1, 2, 3), n_experts=8,
                      top_k=2, expert_ffn=16, shared_ffn=16, max_seq_len=64, tie_embeddings=True)
    m = CSRModel(cfg).double().eval()
    B, T = 2, 40
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(1, 97, (B, T), generator=g)
    doc = torch.zeros(B, T, dtype=torch.long)
    doc[0, 25:] = 1
    doc[1, 35:] = -1
    pos = torch.zeros_like(doc)
    for b in range(B):
        for d in doc[b].unique():
            if d < 0:
                continue
            idx = (doc[b] == d).nonzero().flatten()
            pos[b, idx] = torch.arange(len(idx))
    cid = torch.where(doc >= 0, doc * 3 + pos // 10, -1)
    cpos = torch.where(doc >= 0, pos % 10, 0)
    batch = dict(input_ids=ids, doc_ids=doc, positions=pos, chunk_ids=cid, chunk_pos=cpos)
    r = RoutingConfig(mode="pool" if pool else "token", pool_size=4, warmup=3)
    return m, batch, r


QPOS = torch.tensor([[3, 10, 20, 24], [5, 12, 30, 34]])
QV = torch.ones_like(QPOS, dtype=torch.bool)
AR = torch.arange(2)[:, None]


# ------------------------------------------------------------------ mapping
def test_condition_checkpoint_mapping():
    from spec_audit_inventory import EXPECTED, load_runs, verify_mapping
    assert EXPECTED == {"A": ("token", "none", False), "B": ("pool", "fixed", False), "C": ("pool", "fixed", True),
                        "D": ("pool", "semantic", False), "E": ("pool", "semantic", True),
                        "F": ("pool", "random", False)}
    cfg = {"routing": {"mode": "pool"}, "segmentation": {"strategy": "semantic"}, "semantic": {"enabled": True}}
    assert verify_mapping(cfg, "E")
    from csr.guard import GuardError
    with pytest.raises(GuardError):
        verify_mapping(cfg, "D")
    inv_p = AUD / "inventory.json"
    if inv_p.exists():
        inv = json.loads(inv_p.read_text())["checkpoints"]
        assert len(inv) == 18
        for key, v in inv.items():
            assert key == f"s{v['seed']}_{v['condition']}"
            assert (v["routing_mode"], v["segmentation"], v["semantic_loss"]) == EXPECTED[v["condition"]]
            assert v["run"].endswith(f"rep_v1_s{v['seed']}_{v['condition']}")
        assert len({v["checkpoint_sha256"] for v in inv.values()}) == 18
    tsv = ROOT / ".tmp" / "rep_v1_main" / "runs.tsv"
    if tsv.exists():
        runs = load_runs(tsv)
        assert len(runs) == 18


def test_ambiguous_mapping_rejected(tmp_path, monkeypatch):
    from spec_audit_inventory import load_runs
    from csr.guard import GuardError
    p = ROOT / ".tmp" / f"_test_runs_{id(tmp_path)}.tsv"
    lines = [f"{s}\t{c}\truns/x_{s}{c}\te" for s in (0, 1, 2) for c in "ABCDEF"] + ["0\tA\truns/dup\te"]
    p.write_text("\n".join(lines))
    try:
        with pytest.raises(GuardError, match="ambiguous"):
            load_runs(p)
    finally:
        p.unlink()


# ------------------------------------------------------------------ data units
def test_windows_rules():
    assert sa.select_windows(65) == [(0, 0, 64)]
    assert sa.select_windows(64) == []
    assert sa.candidate_windows(1 + 128 + 63) == [(0, 128)]
    assert sa.candidate_windows(1 + 128 + 64) == [(0, 128), (128, 192)]
    w = sa.select_windows(1024)  # 1023 positions -> 7 full + partial 127 -> 8 candidates
    assert len(w) == 4 and w[0][0] == 0 and w[-1][0] == 7
    for L in range(65, 1025, 7):
        ws = sa.select_windows(L)
        assert 1 <= len(ws) <= 4
        assert all(e - s >= 64 and e - s <= 128 and e <= L - 1 for _, s, e in ws)
        assert [x[0] for x in ws] == sorted(set(x[0] for x in ws))


def test_common_mask_identical_across_conditions():
    """The mask is a function of document lengths only; the extraction index has no condition input."""
    from spec_audit_extract import window_index
    assert "key" not in inspect.signature(window_index).parameters
    if not (AUD / "data" / "windows.npz").exists():
        pytest.skip("audit data absent")
    W = np.load(AUD / "data" / "windows.npz")
    off = np.load(AUD / "data" / "audit_offsets.npy")
    L = np.diff(off)
    rec = [(d, wi, s, e) for d in range(len(L)) for wi, s, e in sa.select_windows(int(L[d]))]
    got = list(zip(W["doc"].tolist(), W["win_index"].tolist(), W["start"].tolist(), W["end"].tolist()))
    assert got == rec
    wl, woff = window_index(W, off)
    assert woff[-1] == wl.sum()


def test_document_level_split_isolation():
    ids = [f"<urn:uuid:{i:08d}>" for i in range(2000)]
    h = [sa.half_of(i) for i in ids]
    assert h == [sa.half_of(i) for i in ids]
    assert 0.45 < np.mean(h) < 0.55
    if (AUD / "data" / "windows.npz").exists():
        W = np.load(AUD / "data" / "windows.npz")
        docs = json.loads((AUD / "data" / "audit_docs.json").read_text())["ids"]
        for d in np.unique(W["doc"]):
            hs = set(W["half"][W["doc"] == d].tolist())
            assert hs == {sa.half_of(docs[d])}
        disc = set(W["doc"][W["half"] == 0].tolist())
        eff = set(W["doc"][W["half"] == 1].tolist())
        assert not disc & eff


def test_no_overlap_and_no_test_access_in_audit_data():
    man_p = AUD / "data" / "manifest.json"
    if not man_p.exists():
        pytest.skip("audit data absent")
    man = json.loads(man_p.read_text())
    assert man["n_docs"] == 8192 and man["dedup"]["id_overlap_with_references"] == 0
    used = set(json.loads((ROOT / "artifacts/download_manifest_rep_v1_20261101T033146Z.json").read_text())["row_groups"])
    used |= set(json.loads((ROOT / "artifacts/download_manifest_20261031T212112Z.json").read_text())["row_groups"])
    assert not set(man["source_row_groups"]) & used
    for f in ("spec_audit_data.py", "spec_audit_labels.py", "spec_audit_extract.py", "spec_audit_segments.py",
              "spec_audit_analyze.py"):
        p = ROOT / f
        if p.exists():
            src = p.read_text()
            assert "test_tokens" not in src and "test_docs" not in src and "test_offsets" not in src
            assert '"test")' not in src and "'test')" not in src


# ------------------------------------------------------------------ label isolation
def test_topic_labels_never_enter_model_forward():
    for mod in (audit, ):
        src = inspect.getsource(mod)
        assert "labels" not in src.replace("label-free", "") and "kmeans" not in src and "specaudit" not in src
    ex = (ROOT / "spec_audit_extract.py").read_text()
    assert "labels" not in ex and "encoder_embeddings" not in ex and "kmeans" not in ex
    for fn in (audit.original_pass, audit.counterfactual):
        params = set(inspect.signature(fn).parameters)
        assert not params & {"labels", "topics", "label", "topic"}


# ------------------------------------------------------------------ determinism
def test_deterministic_kmeans_and_permutations():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 8)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    a = sa.kmeans(X, 5, n_init=4)
    b = sa.kmeans(X, 5, n_init=4)
    assert np.array_equal(a[1], b[1]) and np.array_equal(a[0], b[0]) and a[2] == b[2]
    # well-separated blobs are recovered exactly
    C = np.eye(8)[:4] * 5
    Y = np.concatenate([C[i] + 0.01 * rng.normal(size=(50, 8)) for i in range(4)]).astype(np.float32)
    lab = sa.kmeans(Y, 4, n_init=3)[1]
    assert all(len(set(lab[i * 50:(i + 1) * 50])) == 1 for i in range(4)) and len(set(lab)) == 4
    s = np.array([0] * 10 + [1] * 7 + [17] * 5)
    P1 = sa.stratified_permutations(s, n_perm=20)
    P2 = sa.stratified_permutations(s, n_perm=20)
    assert np.array_equal(P1, P2)
    for p in P1:
        assert sorted(p) == list(range(len(s)))
        assert np.array_equal(s[p], s)  # within strata
    assert not all(np.array_equal(P1[0], P1[i]) for i in range(1, 20))


def test_bootstrap_reproducibility():
    M1, M2 = sa.bootstrap_counts(50, n_boot=30), sa.bootstrap_counts(50, n_boot=30)
    assert np.array_equal(M1, M2) and (M1.sum(1) == 50).all()
    from csr.train_lib import paired_bootstrap  # same index stream as the main experiment's bootstrap
    rng = np.random.default_rng(sa.SEED)
    assert np.array_equal(M1[0], np.bincount(rng.integers(0, 50, 50), minlength=50))


# ------------------------------------------------------------------ joint distributions and metrics
def test_normalized_routing_joint():
    topi = np.array([[0, 2], [1, 2], [3, 0]])
    gates = np.array([[0.3, 0.1], [0.2, 0.2], [0.5, 0.5]])
    M = sa.normalized_gate_mass(topi, gates, 4)
    assert np.allclose(M.sum(1), 1) and np.isclose(M[0, 0], 0.75) and np.isclose(M[0, 2], 0.25)
    wm = sa.window_mass(topi, gates, [0, 0, 1], 2, 4)
    J = sa.joint_from_window_mass(wm, [1, 0], 2)
    assert np.isclose(J.sum(), 1) and np.allclose(J[1], (M[0] + M[1]) / 3) and np.allclose(J[0], M[2] / 3)
    assert np.isclose(sa.mutual_information(np.outer([0.5, 0.5], [0.25] * 4)), 0, atol=1e-12)
    assert np.isclose(sa.mutual_information(np.eye(4) / 4), np.log(4))
    assert np.isclose(sa.nmi(np.eye(4) / 4), 1.0)


def test_weighted_pairwise_cosine_exact():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(12, 5))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    m = rng.random(12)
    num = sum(m[i] * m[j] * X[i] @ X[j] for i in range(12) for j in range(12) if i != j)
    den = sum(m[i] * m[j] for i in range(12) for j in range(12) if i != j)
    assert np.isclose(sa.weighted_pairwise_cosine(X, m), num / den)


def test_hungarian_matches_bruteforce():
    rng = np.random.default_rng(2)
    for n in (1, 2, 4, 6):
        C = rng.random((n, n))
        col = sa.hungarian(C)
        best = min(itertools.permutations(range(n)), key=lambda p: sum(C[i, p[i]] for i in range(n)))
        assert np.isclose(C[np.arange(n), col].sum(), sum(C[i, best[i]] for i in range(n)))
        assert sorted(col) == list(range(n))


def test_logreg_optimum_and_metrics():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(200, 5))
    y = (X[:, 0] > 0).astype(int) + (X[:, 1] > 1).astype(int)
    W, b, info = sa.logreg_fit(X, y, 3)
    assert info["max_abs_grad"] < 1e-4
    W2, b2, _ = sa.logreg_fit(X, y, 3)
    assert np.allclose(W, W2) and np.allclose(b, b2)
    acc = np.mean(sa.logreg_predict(X, W, b) == y)
    assert acc > 0.85
    assert sa.macro_f1([0, 1, 1], [0, 1, 1]) == 1.0 and sa.balanced_accuracy([0, 0, 1], [0, 1, 1]) == 0.75


def test_tfidf_svd_deterministic():
    texts = ["the cat sat on the mat", "dogs and cats are pets", "stock markets fell sharply", "the market rallied",
             "cats chase mice", "bond yields rose"] * 3
    vocab, idf = sa.tfidf_fit(texts)
    assert "cat sat" in vocab and "the" in vocab
    A = sa.tfidf_transform(texts, vocab, idf)
    V1 = sa.randomized_svd_fit(A, len(vocab), k=4)
    V2 = sa.randomized_svd_fit(A, len(vocab), k=4)
    assert np.allclose(V1, V2)
    Z = sa.svd_project(A, len(vocab), V1)
    assert np.allclose(np.linalg.norm(Z, axis=1), 1, atol=1e-5)


def test_interaction_estimator():
    rng = np.random.default_rng(4)
    n = 4000
    cells = rng.integers(0, 5, n)
    is_top = (rng.random(n) < 0.3).astype(int)
    windows = rng.integers(0, 400, n)
    delta = 0.1 * cells + 0.05 * is_top + 0.001 * rng.normal(size=n)  # confounded by cell, true effect 0.05
    r = sa.interaction(delta, is_top, cells, windows)
    assert abs(r["estimate"] - 0.05) < 2e-3
    r2 = sa.interaction(delta, is_top, cells, windows % 10)  # < 32 windows -> insufficient support
    assert r2["estimate"] is None


# ------------------------------------------------------------------ counterfactual replacement
@pytest.mark.parametrize("pool", [False, True])
def test_original_pass_matches_model_forward(pool):
    m, batch, r = tiny(pool)
    ref = m(batch, r)
    o = audit.original_pass(m, batch, r)
    assert torch.allclose(ref.logits, o["logits"], atol=1e-12)
    assert all(torch.equal(a["topi"], b["topi"]) for a, b in zip(ref.routing, o["records"]))


@pytest.mark.parametrize("pool", [False, True])
def test_replacement_exactly_topk_and_inside_pool(pool):
    m, batch, r = tiny(pool)
    o = audit.original_pass(m, batch, r)
    for L in range(3):
        top = o["records"][L]["topi"][AR, QPOS]
        for j in range(2):
            cf = audit.counterfactual(m, o, r, L, QPOS, QV, top[..., j])
            nt = cf["new_topi"]
            assert cf["affected"].all()
            for b in range(2):
                for n in range(QPOS.shape[1]):
                    new, old = set(nt[b, n].tolist()), set(top[b, n].tolist())
                    assert len(new) == 2 and top[b, n, j].item() not in new
                    assert len(new & old) == 1 and len(new - old) == 1
                    t = QPOS[b, n].item()
                    if pool and o["records"][L]["constrained"][b, t]:
                        pool_set = set(torch.nonzero(o["records"][L]["pool_tok"][b, t]).flatten().tolist())
                        assert new <= pool_set
                    if not (pool and o["records"][L]["constrained"][b, t]):
                        probs = o["records"][L]["probs_full"][b, t].clone()
                        probs[top[b, n, j]] = -1
                        assert new == set(probs.topk(2).indices.tolist())


@pytest.mark.parametrize("pool", [False, True])
def test_counterfactual_equals_bruteforce_single_token(pool):
    m, batch, r = tiny(pool)
    o = audit.original_pass(m, batch, r)
    L = 1
    rm = o["records"][L]["topi"][AR, QPOS][..., 0]
    cf = audit.counterfactual(m, o, r, L, QPOS, QV, rm)
    blk = [x for x in m.blocks if x.is_moe][L].ffn
    orig_sel = blk.select
    try:
        for b in range(2):
            for n in range(QPOS.shape[1]):
                t, e = QPOS[b, n].item(), rm[b, n].item()

                def sel(logits, cand, gate_norm="global", b=b, t=t, e=e):
                    cand = torch.ones_like(logits, dtype=torch.bool) if cand is None else cand.clone()
                    cand = cand.view(2, 40, -1)
                    cand[b, t, e] = False
                    return orig_sel(logits, cand.view(80, -1), gate_norm)
                blk.select = sel
                out = m(batch, r).logits[b, t]
                blk.select = orig_sel
                assert torch.allclose(out, cf["logits"][b, n], atol=1e-10)
    finally:
        blk.select = orig_sel


@pytest.mark.parametrize("pool", [False, True])
def test_identity_when_expert_not_selected(pool):
    m, batch, r = tiny(pool)
    o = audit.original_pass(m, batch, r)
    top = o["records"][2]["topi"][AR, QPOS]
    allE = torch.arange(8)
    notsel = torch.stack([torch.stack([allE[~torch.isin(allE, top[b, n])][0] for n in range(4)]) for b in range(2)])
    cf = audit.counterfactual(m, o, r, 2, QPOS, QV, notsel)
    assert not cf["affected"].any()
    assert torch.equal(cf["logits"], o["logits"][AR, QPOS])
    assert torch.equal(cf["new_topi"], top)


def test_shared_expert_unchanged():
    m, batch, r = tiny(True)
    o = audit.original_pass(m, batch, r)
    fp_shared = [audit.param_fingerprint(b.ffn.shared) for b in m.blocks if b.is_moe]
    # with every routed expert's output zeroed, any replacement yields exactly the original stream: the shared expert
    # path is identical in the counterfactual
    zeroed = [b.ffn for b in m.blocks if b.is_moe]
    saved = [z.experts for z in zeroed]
    for z in zeroed:
        z.experts = lambda x, topi, gates: x.new_zeros(x.shape)
    try:
        o0 = audit.original_pass(m, batch, r)
        top = o0["records"][0]["topi"][AR, QPOS][..., 0]
        cf = audit.counterfactual(m, o0, r, 0, QPOS, QV, top)
        assert torch.allclose(cf["logits"], o0["logits"][AR, QPOS], atol=1e-12)
    finally:
        for z, f in zip(zeroed, saved):
            z.experts = f
    assert fp_shared == [audit.param_fingerprint(b.ffn.shared) for b in m.blocks if b.is_moe]


def test_no_parameter_or_checkpoint_mutation(tmp_path):
    m, batch, r = tiny(True)
    p = ROOT / ".tmp" / f"_test_ck_{id(tmp_path)}.pt"
    torch.save({"model": m.state_dict()}, p)
    try:
        from csr.guard import sha256_file
        h0, fp0 = sha256_file(p), audit.param_fingerprint(m)
        o = audit.original_pass(m, batch, r)
        for L in range(3):
            audit.counterfactual(m, o, r, L, QPOS, QV, o["records"][L]["topi"][AR, QPOS][..., 1])
        assert audit.param_fingerprint(m) == fp0 and sha256_file(p) == h0
    finally:
        p.unlink()


# ------------------------------------------------------------------ analysis kernels
def test_vectorized_functional_kernel_matches_reference_and_bootstrap_identity():
    import spec_audit_analyze as an
    rng = np.random.default_rng(5)
    n_win, n_docs, n_ent = 600, 150, 30000
    an.G.clear()
    an.G["wl"] = np.full(n_win, 128)
    an.G["eff_docs"] = np.arange(n_docs)
    an.G["M"] = np.ones((3, n_docs), np.float32)  # unit weights -> every replicate equals the point estimate
    doc_of_w = rng.integers(0, n_docs, n_win)
    e = rng.integers(0, 32, n_ent)
    win = rng.integers(0, n_win, n_ent)
    cellkey = rng.integers(0, 6, n_ent)
    lab_w = rng.integers(0, 5, n_win)
    top = rng.integers(0, 5, 32)
    top[3] = -1  # zero-load expert
    d = 0.01 * cellkey + 0.02 * (lab_w[win] == top[e]) + 0.001 * rng.normal(size=n_ent)
    cell = e * 16000 + cellkey
    ucell, cid = np.unique(cell, return_inverse=True)
    pair = e * n_win + win
    upair, pid = np.unique(pair, return_inverse=True)
    S = {"e": e, "d": d, "win": win, "cid": cid, "e_of_c": ucell // 16000, "n_c": np.bincount(cid).astype(float),
         "S_c": np.bincount(cid, weights=d), "pid": pid, "e_of_p": upair // n_win, "doc": doc_of_w[win],
         "n_e": np.bincount(e, minlength=32)}
    res = an.functional_estimate(S, lab_w, top, want_boot=True)
    for ex in range(32):
        s = e == ex
        if top[ex] < 0:
            assert not res["elig"][ex]
            continue
        ref = sa.interaction(d[s], (lab_w[win[s]] == top[ex]).astype(int), cellkey[s], win[s])
        if ref["estimate"] is None:
            assert not res["elig"][ex]
        else:
            assert res["elig"][ex] and np.isclose(res["I"][ex], ref["estimate"])
    bs = an.boot_summary([res["boot_parts"]], [res["elig"]])
    assert np.allclose(bs[:, 0], np.nanmean(res["I"][res["elig"]]))
    assert abs(np.nanmean(res["I"][res["elig"]]) - 0.02) < 2e-3
