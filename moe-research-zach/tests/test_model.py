"""Gates 1-7, 9, 10, 12, 13 for the model and routing."""
import copy

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from conftest import single_doc_batch, tiny_cfg
from csr.model import CSRModel, RoutingConfig, build_chunk_layout, rope_cos_sin, apply_rope
from csr.segment import fixed_starts, starts_to_arrays

TOK = RoutingConfig("token", 4, 4)
POOL = RoutingConfig("pool", 4, 4)
TOL = dict(rtol=0, atol=1e-10)


def rand_ids(n, seed=0, V=97):
    return np.random.default_rng(seed).integers(2, V, size=n)


# ---------------------------------------------------------------- independent naive reference (gate 3)
def reference_forward(model, ids):
    """Per-token, per-expert loop implementation of FLAME routing (softmax over all experts, top-k by prob,
    unnormalized gates, dropless, shared expert), written independently of the vectorized dispatch."""
    cfg = model.cfg
    x = model.tok_emb.weight[torch.as_tensor(ids)]  # (T, d)
    T = len(ids)
    pos = torch.arange(T)
    hd = cfg.d_model // cfg.n_head
    aux, z = 0.0, 0.0
    topis = []
    for blk in model.blocks:
        h = blk.norm1(x)
        qkv = h @ blk.attn.qkv.weight.T
        q, k, v = qkv.view(T, 3, cfg.n_head, hd).unbind(1)
        cos, sin = rope_cos_sin(pos, hd, cfg.rope_base, x.dtype)
        q = apply_rope(q.transpose(0, 1), cos, sin)
        k = apply_rope(k.transpose(0, 1), cos, sin)
        att = (q @ k.transpose(-1, -2)) / hd ** 0.5
        att = att.masked_fill(~torch.ones(T, T, dtype=torch.bool).tril(), float("-inf")).softmax(-1)
        y = (att @ v.transpose(0, 1)).transpose(0, 1).reshape(T, -1)
        x = x + y @ blk.attn.proj.weight.T
        h = blk.norm2(x)
        if not blk.is_moe:
            f = blk.ffn
            x = x + (F.silu(h @ f.w1.weight.T) * (h @ f.w3.weight.T)) @ f.w2.weight.T
            continue
        m = blk.ffn
        logits = h @ m.router.weight.T
        probs = logits.softmax(-1)
        out = torch.zeros_like(h)
        tl = []
        counts = torch.zeros(m.E, dtype=h.dtype)
        for t in range(T):
            order = sorted(range(m.E), key=lambda e: -float(probs[t, e]))[:m.k]
            tl.append(sorted(order))
            for e in order:
                counts[e] += 1
                he = F.silu(h[t] @ m.w1[e]) * (h[t] @ m.w3[e])
                out[t] += probs[t, e] * (he @ m.w2[e])
            s = m.shared
            out[t] += (F.silu(h[t] @ s.w1.weight.T) * (h[t] @ s.w3.weight.T)) @ s.w2.weight.T
        x = x + out
        aux += float(m.E * ((counts / (T * m.k)) * probs.mean(0)).sum())
        z += float((torch.logsumexp(logits, -1) ** 2).mean())
        topis.append(tl)
    return model.lm_head(model.norm_f(x)), aux, z, topis


def test_baseline_matches_independent_reference(tiny_model):
    ids = rand_ids(40)
    b = single_doc_batch(ids, [0])
    with torch.no_grad():
        out = tiny_model(b, TOK)
        ref_logits, aux, z, topis = reference_forward(tiny_model, ids)
    assert torch.allclose(out.logits[0], ref_logits, **TOL)
    assert abs(float(out.aux_loss) - aux) < 1e-10 and abs(float(out.z_loss) - z) < 1e-10
    for r, tl in zip(out.routing, topis):
        assert [sorted(x) for x in r["topi"][0].tolist()] == tl


def test_pool_with_m_equal_E_reduces_to_token_routing(tiny_model):
    ids = rand_ids(60, 1)
    b = single_doc_batch(ids, fixed_starts(60, 16))
    with torch.no_grad():
        a = tiny_model(b, TOK)
        p = tiny_model(b, RoutingConfig("pool", tiny_model.cfg.n_experts, 4))
    assert torch.allclose(a.logits, p.logits, **TOL)
    assert float(a.aux_loss) == pytest.approx(float(p.aux_loss), abs=1e-12)


def test_constrained_experts_inside_pool_and_k_active(tiny_model):
    ids = rand_ids(80, 2)
    starts = [0, 3, 20, 25, 50]  # includes chunks shorter than and longer than warmup
    b = single_doc_batch(ids, starts)
    with torch.no_grad():
        out = tiny_model(b, POOL)
    lay = out.layout
    cidx, cpos = starts_to_arrays(80, starts)
    exp_eligible = cpos >= POOL.warmup
    assert np.array_equal(lay.eligible[0].numpy(), exp_eligible)
    for r in out.routing:
        topi = r["topi"][0]
        assert topi.shape[-1] == tiny_model.cfg.top_k
        assert all(len(set(row)) == tiny_model.cfg.top_k for row in topi.tolist())
        pools = r["chunk_logits"].topk(POOL.pool_size, -1).indices
        con = r["constrained"][0]
        assert torch.equal(con, lay.eligible[0])
        for t in torch.nonzero(con).flatten().tolist():
            row = int(lay.token_row[0, t])
            assert set(topi[t].tolist()) <= set(pools[row].tolist())
        # prefix tokens and short chunks are routed exactly as token routing
    with torch.no_grad():
        tok = tiny_model(b, TOK)
    first_eligible = int(np.nonzero(exp_eligible)[0][0])
    assert torch.allclose(out.logits[0, :first_eligible], tok.logits[0, :first_eligible], **TOL)


@pytest.mark.parametrize("routing", [TOK, POOL])
def test_future_tokens_do_not_change_past(tiny_model, routing):
    ids = rand_ids(90, 3)
    starts = fixed_starts(90, 20)
    cut = 47
    ids2 = ids.copy()
    ids2[cut + 1:] = rand_ids(90 - cut - 1, 99)
    with torch.no_grad():
        o1 = tiny_model(single_doc_batch(ids, starts), routing)
        o2 = tiny_model(single_doc_batch(ids2, starts), routing)
    assert torch.allclose(o1.logits[0, :cut + 1], o2.logits[0, :cut + 1], **TOL)
    assert not torch.allclose(o1.logits[0, cut + 1:], o2.logits[0, cut + 1:])
    for r1, r2 in zip(o1.routing, o2.routing):
        assert torch.equal(r1["topi"][0, :cut + 1], r2["topi"][0, :cut + 1])
        assert torch.equal(r1["constrained"][0, :cut + 1], r2["constrained"][0, :cut + 1])
        if r1["chunk_logits"] is not None:
            lay = o1.layout
            done = (lay.rows_t0 + routing.warmup - 1) <= cut  # prefix fully observed by `cut`
            assert torch.allclose(r1["chunk_logits"][done], r2["chunk_logits"][done], **TOL)


@pytest.mark.parametrize("routing", [TOK, POOL])
def test_incremental_matches_parallel(tiny_model, routing):
    ids = rand_ids(70, 4)
    starts = [0, 9, 30, 33, 52]
    _, cpos = starts_to_arrays(70, starts)
    with torch.no_grad():
        par = tiny_model(single_doc_batch(ids, starts), routing)
        st = tiny_model.new_state()
        for t in range(70):
            lg, rt = tiny_model.step(int(ids[t]), t, int(cpos[t]), routing, st)
            assert torch.allclose(lg, par.logits[0, t], rtol=0, atol=1e-9), t
            for li, (topi, con) in enumerate(rt):
                assert torch.equal(topi, par.routing[li]["topi"][0, t])
                assert con == bool(par.routing[li]["constrained"][0, t])


def test_document_isolation_in_packed_rows(tiny_model):
    d1, d2 = rand_ids(30, 5), rand_ids(25, 6)
    T = 64
    ids = np.full(T, 1)
    ids[:30], ids[30:55] = d1, d2
    doc = np.full(T, -1)
    doc[:30], doc[30:55] = 0, 1
    pos = np.zeros(T, dtype=np.int64)
    pos[:30], pos[30:55] = np.arange(30), np.arange(25)
    cid = np.full(T, -1)
    cpos = np.zeros(T, dtype=np.int64)
    c1, p1 = starts_to_arrays(30, [0, 10])
    c2, p2 = starts_to_arrays(25, [0, 12])
    cid[:30], cid[30:55] = c1, c2 + 2
    cpos[:30], cpos[30:55] = p1, p2
    t = lambda a: torch.as_tensor(a).unsqueeze(0)
    b = {"input_ids": t(ids), "doc_ids": t(doc), "positions": t(pos), "chunk_ids": t(cid), "chunk_pos": t(cpos)}
    for routing in (TOK, POOL):
        with torch.no_grad():
            packed = tiny_model(b, routing).logits[0]
            alone = tiny_model(single_doc_batch(d2, [0, 12]), routing).logits[0]
        assert torch.allclose(packed[30:55], alone, rtol=0, atol=1e-9)


def test_make_batch_targets_never_cross_documents():
    from csr.data import TokenSplit, make_batch
    docs = [np.array([5, 6, 7, 0]), np.array([8, 9, 0]), np.array([11, 12, 13, 14, 0])]
    off = np.concatenate([[0], np.cumsum([len(d) for d in docs])])
    sp = TokenSplit(np.concatenate(docs).astype(np.uint16), off, ["a", "b", "c"])
    b = make_batch([[0, 1], [2]], sp, 8)
    assert b["targets"][0].tolist() == [6, 7, 0, -100, 9, 0, -100, -100]
    assert b["targets"][1].tolist() == [12, 13, 14, 0, -100, -100, -100, -100]
    assert b["doc_ids"][0].tolist() == [0, 0, 0, 0, 1, 1, 1, -1]
    assert b["positions"][0].tolist()[:7] == [0, 1, 2, 3, 0, 1, 2]


def test_gradients_finite_and_frozen_unchanged():
    torch.manual_seed(0)
    m = CSRModel(tiny_cfg()).double()
    m.tok_emb.weight.requires_grad_(False)
    frozen = m.tok_emb.weight.detach().clone()
    ids = rand_ids(64, 7)
    b = single_doc_batch(ids, fixed_starts(64, 16))
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-2)
    out = m(b, POOL)
    loss = F.cross_entropy(out.logits[0, :-1], torch.as_tensor(ids[1:])) + 0.01 * out.aux_loss + 1e-3 * out.z_loss
    loss.backward()
    for blk in m.moe_blocks():
        g = blk.ffn.router.weight.grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
        used = torch.unique(out.routing[0]["topi"])
        for e in used.tolist():
            assert blk.ffn.w1.grad[e].abs().sum() > 0
        assert torch.isfinite(blk.ffn.w1.grad).all()
    opt.step()
    assert torch.equal(m.tok_emb.weight, frozen)


def test_seeded_init_and_forward_deterministic():
    outs = []
    for _ in range(2):
        torch.manual_seed(123)
        m = CSRModel(tiny_cfg()).double().eval()
        with torch.no_grad():
            outs.append(m(single_doc_batch(rand_ids(50, 8), fixed_starts(50, 12)), POOL).logits)
    assert torch.equal(outs[0], outs[1])


def test_layout_handles_padding_and_short_chunks():
    cid = torch.tensor([[0, 0, 0, 0, 0, 0, 1, 1, -1, -1]])
    cpos = torch.tensor([[0, 1, 2, 3, 4, 5, 0, 1, 0, 0]])
    lay = build_chunk_layout(cid, cpos, 4)
    assert lay.rows_t0.tolist() == [0]
    assert lay.eligible[0].tolist() == [False] * 4 + [True, True] + [False] * 4
    assert lay.token_row[0].tolist()[:6] == [-1, -1, -1, -1, 0, 0]


@pytest.mark.parametrize("gate_norm", ["global", "pool"])
def test_gate_norm_semantics(tiny_model, gate_norm):
    ids = rand_ids(60, 11)
    b = single_doc_batch(ids, [0, 30])
    r = RoutingConfig("pool", 4, 4, gate_norm)
    with torch.no_grad():
        out = tiny_model(b, r)
    rec = out.routing[0]
    con, probs, topi = rec["constrained"][0], rec["probs_full"][0], rec["topi"][0]
    pools = rec["chunk_logits"].topk(4, -1).indices
    assert con.any()
    for t in torch.nonzero(con).flatten().tolist():
        pool = pools[int(out.layout.token_row[0, t])]
        best = pool[probs[t, pool].argsort(descending=True)[:2]]
        # both gate norms select the top-k pool experts by global probability (softmax is monotone)
        assert set(best.tolist()) == set(topi[t].tolist())
    gates = rec["gates"][0]
    expect = probs.gather(1, topi)
    if gate_norm == "global":  # gate scale identical to token routing everywhere
        assert torch.allclose(gates, expect, rtol=0, atol=1e-12)
    else:  # renormalized inside the pool for constrained tokens only
        assert (gates[con].sum(-1) > expect[con].sum(-1)).all()
        assert torch.allclose(gates[~con], expect[~con], rtol=0, atol=1e-12)
