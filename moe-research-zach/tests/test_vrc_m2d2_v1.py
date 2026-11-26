"""CPU tests (CUDA hidden) for csr-value-routed-compute-v1 and csr-m2d2-domain-prior-v1 mechanisms."""
import json

import numpy as np
import pytest
import torch

from csr.model import CSRModel, ModelConfig, RoutingConfig

SMALL = dict(vocab_size=512, n_layer=8, d_model=48, n_head=2, dense_ffn=64, moe_layers=(1, 2, 3, 4, 5, 6, 7),
             n_experts=32, top_k=4, expert_ffn=16, shared_ffn=32, max_seq_len=1024, tie_embeddings=True)


def mk_batch(B, T, docs_per_row=2, seed=0, vocab=512):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(2, vocab, (B, T), generator=g)
    doc_ids = torch.zeros(B, T, dtype=torch.long)
    positions = torch.zeros(B, T, dtype=torch.long)
    for b in range(B):
        cuts = sorted(torch.randint(1, T - 1, (docs_per_row - 1,), generator=g).tolist())
        bounds = [0] + cuts + [T]
        for j in range(len(bounds) - 1):
            doc_ids[b, bounds[j]:bounds[j + 1]] = j
            positions[b, bounds[j]:bounds[j + 1]] = torch.arange(bounds[j + 1] - bounds[j])
    doc_ids[:, -5:] = -1  # padding
    positions[:, -5:] = 0
    ids[:, -5:] = 1
    return {"input_ids": ids, "doc_ids": doc_ids, "positions": positions,
            "gdoc": doc_ids.clone(), "domain": torch.where(doc_ids >= 0, (doc_ids + b) % 6, -1)}


def csr_ref(seed=0, cfg=None):
    torch.manual_seed(seed)
    return CSRModel(ModelConfig(**(cfg or SMALL))).double()


# ============================================================ VRC
def vrc(seed=0):
    from csr.vrc_model import VRCModel
    torch.manual_seed(seed)
    return VRCModel(ModelConfig(**SMALL), "dense").double()


def test_vrc_ref_equals_token4():
    m, r = vrc(), csr_ref()
    tb = mk_batch(2, 256)
    a = m(tb).logits
    b = r(tb, RoutingConfig(mode="token", pool_size=8, warmup=4)).logits
    assert torch.allclose(a, b, atol=1e-10)
    assert m.state_dict().keys() == r.state_dict().keys()


def test_vrc_fallback_equals_ref_when_short():
    m = vrc()
    tb = mk_batch(2, 256)  # T <= W: local == full exactly
    nb = 256 // 128
    for g in (True, False):
        out = m(tb, ref=False, gsel=torch.full((2, nb), g), kblk=torch.full((2, nb), 4))
        assert torch.allclose(out.logits, m(tb).logits, atol=1e-10)


def test_experts_null_k4_and_k1():
    from csr.vrc_model import experts_null
    m = vrc()
    moe = m.blocks[1].ffn
    x = torch.randn(40, 48, dtype=torch.float64)
    logits = moe.router_logits(x)
    topi, gates, _ = moe.select(logits, None)
    full = moe.experts(x, topi, gates)
    y4, n4 = experts_null(moe, x, topi, gates, torch.ones_like(topi, dtype=torch.bool))
    assert torch.allclose(full, y4, atol=1e-12) and n4 == 160
    v1 = torch.zeros_like(topi, dtype=torch.bool)
    v1[:, 0] = True
    y1, n1 = experts_null(moe, x, topi, gates, v1)
    assert n1 == 40
    man = torch.zeros_like(x)
    for i in range(40):
        e = int(topi[i, 0])
        xi = x[i:i + 1]
        h = torch.nn.functional.silu(xi @ moe.w1[e]) * (xi @ moe.w3[e])
        man[i] = (h @ moe.w2[e])[0] * gates[i, 0]
    assert torch.allclose(y1, man, atol=1e-12)


def test_vrc_mixed_k_is_per_block():
    m = vrc()
    tb = mk_batch(1, 256, docs_per_row=1)
    o4 = m(tb, ref=False, gsel=torch.ones(1, 2, dtype=torch.bool), kblk=torch.tensor([[4, 4]]))
    o1 = m(tb, ref=False, gsel=torch.ones(1, 2, dtype=torch.bool), kblk=torch.tensor([[4, 1]]))
    assert torch.allclose(o4.logits[:, :128], o1.logits[:, :128], atol=1e-12)  # causal: block 0 unaffected
    assert not torch.allclose(o4.logits[:, 128:], o1.logits[:, 128:])
    assert o1.n_expert_evals < o4.n_expert_evals


def test_vrc_causality_and_doc_isolation_mixed_policy():
    m = vrc()
    T = 1024
    tb = mk_batch(1, T, docs_per_row=2, seed=3)
    g = torch.Generator().manual_seed(1)
    gsel = torch.rand(1, T // 128, generator=g) > 0.5
    kblk = torch.tensor([[1, 2, 4, 1, 2, 4, 2, 1]])
    base = m(tb, ref=False, gsel=gsel, kblk=kblk).logits
    p = 700
    tb2 = {k: v.clone() for k, v in tb.items()}
    tb2["input_ids"][0, p] = (tb2["input_ids"][0, p] + 7) % 500 + 2
    pert = m(tb2, ref=False, gsel=gsel, kblk=kblk).logits
    assert torch.allclose(base[0, :p], pert[0, :p], atol=1e-12)
    d = tb["doc_ids"][0]
    first_doc_end = int((d == 0).sum())
    if first_doc_end < p:  # perturbation in doc 1 must not affect doc 0
        assert torch.allclose(base[0, :first_doc_end], pert[0, :first_doc_end], atol=1e-12)


def test_vrc_global_only_in_layers_3_7_matters_long():
    m = vrc()
    T = 1024
    tb = mk_batch(1, T, docs_per_row=1, seed=5)
    kb = torch.full((1, 8), 4)
    a = m(tb, ref=False, gsel=torch.zeros(1, 8, dtype=torch.bool), kblk=kb).logits
    b = m(tb, ref=False, gsel=torch.ones(1, 8, dtype=torch.bool), kblk=kb).logits
    assert torch.allclose(a[0, :512], b[0, :512], atol=1e-12)  # positions < W: local == global
    assert not torch.allclose(a[0, 600:900], b[0, 600:900])


def test_vrc_mixed_block_lists_exact_superset():
    from csr.cla_attn import make_layout, BLOCK
    from csr.vrc_model import mixed_block_mask
    T, B = 1024, 1
    tb = mk_batch(B, T, docs_per_row=3, seed=9)
    lay = make_layout(tb["doc_ids"])
    gq = torch.tensor([True, False, True, False, False, True, False, True])
    bm = mixed_block_mask(lay, 128, gq)
    gd = lay.gdoc
    q = torch.arange(T)[:, None]
    k = torch.arange(T)[None, :]
    allowed = (gd[q] == gd[k]) & (k <= q) & ((q - k < 128) | gq[q // BLOCK])
    nqb = T // BLOCK
    part = torch.zeros(nqb, nqb, dtype=torch.bool)
    full = torch.zeros(nqb, nqb, dtype=torch.bool)
    for i in range(nqb):
        for j in bm.kv_indices[0, 0, i, :int(bm.kv_num_blocks[0, 0, i])]:
            part[i, j] = True
        for j in bm.full_kv_indices[0, 0, i, :int(bm.full_kv_num_blocks[0, 0, i])]:
            full[i, j] = True
    blk = allowed.view(nqb, BLOCK, nqb, BLOCK)
    anyb = blk.any(3).any(1)
    allb = blk.all(3).all(1)
    assert bool(((part | full) >= anyb).all())  # superset of allowed pairs
    assert bool((~full | allb).all())  # full blocks contain only allowed pairs
    # local-only query blocks never list far key blocks (no hidden dense attention)
    for i in range(nqb):
        if not gq[i]:
            js = torch.nonzero(part[i] | full[i]).flatten()
            assert all(i - int(j) <= 1 for j in js)


# ============================================================ M2D2
def dom(kind, seed=0):
    from csr.m2d2_model import DomainModel
    torch.manual_seed(seed)
    return DomainModel(ModelConfig(**SMALL), kind, seed).double()


def test_m2d2_token_kind_equals_token4():
    m, r = dom("token"), csr_ref()
    tb = mk_batch(2, 128)
    assert torch.allclose(m(tb).logits, r(tb, RoutingConfig(mode="token", pool_size=8, warmup=4)).logits, atol=1e-10)


def test_m2d2_backbones_identical_and_b2_equals_b1_at_init():
    from csr.vm_run import param_sha
    ms = {k: dom(k) for k in ("token", "domain_emb", "prior")}
    shas = {k: param_sha(m.backbone_state().items()) for k, m in ms.items()}
    assert len(set(shas.values())) == 1
    assert torch.equal(ms["domain_emb"].dom_emb, ms["prior"].dom_emb)
    tb = mk_batch(2, 128)
    assert torch.allclose(ms["domain_emb"](tb).logits, ms["prior"](tb).logits, atol=1e-12)
    assert ms["prior"].added_parameter_count() == 6 * 48 + 4 * 6 * 32 + 4


def test_m2d2_prior_only_layers_4_7_and_unrestricted_topk():
    m = dom("prior")
    with torch.no_grad():
        m.dom_bias.normal_(0, 3.0)
    tb = mk_batch(2, 128)
    out = m(tb, collect=True)
    off = m(tb, collect=True, prior_off=True)
    for li, (a, b) in enumerate(zip(out.routing, off.routing)):
        layer = li + 1
        if layer < 4:
            assert torch.equal(a["topi"], b["topi"])
        assert (a["topi"].sort(-1).values.diff(dim=-1) > 0).all()  # 4 distinct experts per token
    # selection equals plain top-4 of the biased logits over all 32 experts (no mask / pool)
    changed = any(not torch.equal(r["topi"], r["topi_noprior"]) for r in out.routing[3:])
    assert changed


def test_m2d2_domain_embedding_used():
    m = dom("domain_emb")
    tb = mk_batch(2, 128)
    tb2 = dict(tb)
    tb2["domain"] = torch.where(tb["domain"] >= 0, (tb["domain"] + 1) % 6, -1)
    assert not torch.allclose(m(tb).logits, m(tb2).logits)
    z = dom("token")
    assert torch.allclose(z(tb).logits, z(tb2).logits)


def test_m2d2_shuffled_labels_preserve_frequencies():
    from csr.vm_run import shuffled_labels
    d = np.repeat(np.arange(6), [5, 9, 3, 7, 2, 4])
    s = shuffled_labels(d, 4242)
    assert np.array_equal(np.bincount(s), np.bincount(d)) and not np.array_equal(s, d)
    assert np.array_equal(s, shuffled_labels(d, 4242))


def test_m2d2_real_data_rows_single_domain_and_splits_disjoint():
    from pathlib import Path
    from csr.vm_run import load_split, m2d2_train_rows, m2d2_eval_rows, SplitLocked
    d = Path("/home/zapan4/moe-research/artifacts/m2d2-domain-prior-v1/data")
    if not d.exists():
        pytest.skip("data not built")
    sp, doms = load_split("m2d2", d, "train")
    rows = m2d2_train_rows(sp, doms, 1024, 17)
    assert all(len(set(doms[r].tolist())) == 1 for r in rows)
    assert all(sp.lengths()[r].sum() <= 1024 for r in rows[:2000])
    cs, cdoms = load_split("m2d2", d, "calib_search")
    assert all(len(set(cdoms[r].tolist())) == 1 for r in m2d2_eval_rows(cs, cdoms, 1024))
    ids_tr = set(json.loads((d / "train_docs.json").read_text())["ids"])
    ids_cs = set(json.loads((d / "calib_search_docs.json").read_text())["ids"])
    ids_va = set(json.loads((d / "val_docs.json").read_text())["ids"])
    assert not (ids_tr & ids_cs) and not (ids_tr & ids_va) and not (ids_cs & ids_va)
    assert np.array_equal(np.bincount(doms), np.bincount(doms)[[0] * 6]) or len(set(np.bincount(doms))) > 0
    with pytest.raises(SplitLocked):
        load_split("m2d2", d, "test")
    man = json.loads((d / "manifest.json").read_text())
    assert all(v["decoded"] is False for v in man["locked_test"].values())
    assert not (d / "test_tokens.npy").exists()


def test_split_locks():
    from csr.vm_run import load_split, SplitLocked, EXP
    with pytest.raises(SplitLocked):
        load_split("vrc", "/nonexistent", "test")
    with pytest.raises(SplitLocked):
        load_split("vrc", "/nonexistent", "val")
    if not (EXP["m2d2"]["art"] / "frozen.json").exists():
        with pytest.raises(SplitLocked):
            load_split("m2d2", "/nonexistent", "val")


def test_m2d2_mutual_information_helper():
    import sys
    sys.path.insert(0, "/home/zapan4/moe-research")
    from m2d2_evaluate import mi_from_counts
    assert abs(mi_from_counts(np.ones((6, 32)))) < 1e-12
    eye = np.zeros((6, 32))
    for d in range(6):
        eye[d, d] = 10
    assert abs(mi_from_counts(eye) - np.log(6)) < 1e-12


def test_m2d2_bootstrap_paired_sign():
    import sys
    sys.path.insert(0, "/home/zapan4/moe-research")
    from m2d2_analyze import boot
    rng = np.random.default_rng(0)
    n = rng.integers(50, 500, 300).astype(float)
    a = n * 3.0 + rng.normal(0, 1, 300)
    b = a - 0.05 * n
    r = boot(a, b, n, n_boot=200)
    assert abs(r["diff"] + 0.05) < 1e-9 and r["ci95"][1] < 0
