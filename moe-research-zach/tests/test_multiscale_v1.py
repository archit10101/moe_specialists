"""csr-multiscale-routing-v1 regression tests (CPU, CUDA hidden). docs/MULTISCALE_ROUTING_V1_PREREG.md §6."""
import copy
import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from conftest import fake_detector, fake_embed, single_doc_batch, tiny_cfg, topic_doc
from csr.model import CSRModel, RoutingConfig
from csr.multiscale import (APPROVED_DIFF_CALIB, BRANCH_CONDITIONS, CONDITIONS, CONTEXTUAL, SCHEDULES, MSRouting,
                            MultiscaleModel, branch_of, center, check_matched, config_for,
                            context_means, load_split, load_starts, rms)
from csr import ms_select
from csr.multiscale import TestSplitLocked as SplitLocked
from csr.segment import OnlineSemanticChunker, SemanticParams, fixed_starts, semantic_starts, starts_to_arrays

TOL = dict(rtol=0, atol=1e-10)


def ms_cfg(**kw):
    base = dict(n_layer=8, moe_layers=(1, 2, 3, 4, 5, 6, 7), n_experts=8, top_k=4, d_model=32, n_head=4,
                dense_ffn=48, expert_ffn=16, shared_ffn=24, max_seq_len=256, vocab_size=97)
    base.update(kw)
    return tiny_cfg(**base)


@pytest.fixture(scope="module")
def msm():
    torch.manual_seed(0)
    return MultiscaleModel(ms_cfg()).double().eval()


def rids(n, seed=0, V=97):
    return np.random.default_rng(seed).integers(2, V, size=n).astype(np.int64)


def routing_for(cond, alpha=1.0, beta=1.0, **kw):
    sched, _, scale, _ = CONDITIONS[cond]
    return MSRouting(alpha=alpha, beta=beta, schedule=SCHEDULES[sched], scale_control=scale, **kw).validate()


ALL = sorted(CONDITIONS)


def doc_batch(ids, starts):
    return single_doc_batch(ids, starts)


# ---------------------------------------------------------------- exact reductions
@pytest.mark.parametrize("cond", ALL)
def test_zero_coefficients_reproduce_token_exactly(msm, cond):
    ids = rids(80, 1)
    b = doc_batch(ids, fixed_starts(80, 20))
    with torch.no_grad():
        tok = msm(b, routing_for("TOKEN", 0, 0))
        out = msm(b, routing_for(cond, 0.0, 0.0))
    assert torch.equal(out.logits, tok.logits)
    for r1, r2 in zip(out.routing, tok.routing):
        assert torch.equal(r1["topi"], r2["topi"]) and torch.equal(r1["gates"], r2["gates"])
    assert torch.equal(out.aux_loss, tok.aux_loss) and torch.equal(out.z_loss, tok.z_loss)


def test_token_condition_equals_existing_token_mode(msm):
    ids = rids(70, 2)
    b = doc_batch(ids, [0])
    base = CSRModel(msm.cfg).double().eval()
    base.load_state_dict(msm.state_dict())
    with torch.no_grad():
        a = msm(b, routing_for("TOKEN", 0, 0))
        r = base(b, RoutingConfig("token", 8, 16))
    assert torch.equal(a.logits, r.logits)
    for x, y in zip(a.routing, r.routing):
        assert torch.equal(x["topi"], y["topi"])


def test_no_new_parameters_and_router_has_no_bias(msm):
    base = CSRModel(msm.cfg)
    assert sorted(n for n, _ in base.named_parameters()) == sorted(n for n, _ in msm.named_parameters())
    for blk in msm.moe_blocks():
        assert blk.ffn.router.bias is None


def test_context_projection_is_bias_free_W_times_mean(msm):
    ids = rids(60, 3)
    b = doc_batch(ids, [0, 25])
    with torch.no_grad():
        out = msm(b, routing_for("HIER-FIXED"), collect=True)
    for li, r in enumerate(out.routing):
        if r["kind"] != "hier":
            continue
        W = msm.moe_blocks()[li].ffn.router.weight
        m_l = r["m_local"].view(1, 60, -1)
        m_b = r["m_broad"].view(1, 60, -1)
        n_b = torch.arange(60, dtype=torch.float64).clamp(max=1024)
        ramp = (n_b / 512).clamp(max=1).view(1, 60, 1)
        expect = 1.0 * center(m_l @ W.T) + 1.0 * center(m_b @ W.T) * ramp
        assert torch.allclose(r["ctx"], expect, **TOL)


def test_linear_equivalence_W_mean_h_equals_mean_W_h():
    torch.manual_seed(1)
    for dt, tol in ((torch.float64, 1e-12), (torch.float32, 2e-5)):
        W = torch.randn(32, 192, dtype=dt)
        h = torch.randn(1, 300, 192, dtype=dt)
        pos = torch.arange(300).unsqueeze(0)
        cpos = torch.as_tensor(starts_to_arrays(300, [0, 40, 130])[1]).unsqueeze(0)
        nonpad = torch.ones(1, 300, dtype=torch.bool)
        m_l, n_l, m_b, n_b = context_means(h, pos, cpos, nonpad, True, True)
        z = h @ W.T
        zl, _, zb, _ = context_means(z, pos, cpos, nonpad, True, True)
        assert (m_l @ W.T - zl).abs().max() < tol
        assert (m_b @ W.T - zb).abs().max() < tol


# ---------------------------------------------------------------- context definitions
def brute_context(h, doc_ids, positions, chunk_pos, local_max=16, window=1024):
    B, T, d = h.shape
    ml, mb = torch.zeros_like(h), torch.zeros_like(h)
    nl, nb = torch.zeros(B, T, dtype=torch.long), torch.zeros(B, T, dtype=torch.long)
    for b in range(B):
        for t in range(T):
            if doc_ids[b, t] < 0:
                continue
            p, cp = int(positions[b, t]), int(chunk_pos[b, t])
            t0 = t - cp
            n = min(cp, local_max)
            idx = list(range(t0, t0 + n))
            assert all(doc_ids[b, i] == doc_ids[b, t] and i < t for i in idx)
            if n:
                ml[b, t] = h[b, idx].mean(0)
            nl[b, t] = n
            n2 = min(p, window)
            idx = list(range(t - n2, t))
            assert all(doc_ids[b, i] == doc_ids[b, t] for i in idx)
            if n2:
                mb[b, t] = h[b, idx].mean(0)
            nb[b, t] = n2
    return ml, nl, mb, nb


def packed_two_docs(L1=50, L2=40, T=100, s1=(0, 20), s2=(0, 7, 30)):
    doc = np.full(T, -1)
    pos = np.zeros(T, np.int64)
    cid = np.full(T, -1)
    cpos = np.zeros(T, np.int64)
    doc[:L1], doc[L1:L1 + L2] = 0, 1
    pos[:L1], pos[L1:L1 + L2] = np.arange(L1), np.arange(L2)
    c1, p1 = starts_to_arrays(L1, list(s1))
    c2, p2 = starts_to_arrays(L2, list(s2))
    cid[:L1], cid[L1:L1 + L2] = c1, c2 + c1.max() + 1
    cpos[:L1], cpos[L1:L1 + L2] = p1, p2
    t = lambda a: torch.as_tensor(a).unsqueeze(0)
    return t(doc), t(pos), t(cid), t(cpos)


@pytest.mark.parametrize("local_max,window", [(16, 1024), (4, 10)])
def test_context_means_match_bruteforce_caps_causality_and_documents(local_max, window):
    doc, pos, cid, cpos = packed_two_docs()
    torch.manual_seed(2)
    h = torch.randn(1, 100, 6, dtype=torch.float64)
    ml, nl, mb, nb = context_means(h, pos, cpos, doc >= 0, True, True, local_max, window)
    rl, rnl, rb, rnb = brute_context(h, doc, pos, cpos, local_max, window)
    assert torch.allclose(ml, rl, **TOL) and torch.allclose(mb, rb, **TOL)
    assert torch.equal(nl, rnl) and torch.equal(nb, rnb)
    assert int(nl.max()) <= local_max and int(nb.max()) <= window
    # first token of each chunk / document has zero context
    assert (nl[cpos == 0] == 0).all() and (nb[pos == 0] == 0).all()


def test_context_excludes_current_and_future_tokens():
    doc, pos, cid, cpos = packed_two_docs()
    torch.manual_seed(3)
    h = torch.randn(1, 100, 6, dtype=torch.float64)
    base = context_means(h, pos, cpos, doc >= 0, True, True)
    for t in (5, 30, 49, 60, 89):
        h2 = h.clone()
        h2[0, t:] = torch.randn_like(h2[0, t:])  # current and all later tokens change
        new = context_means(h2, pos, cpos, doc >= 0, True, True)
        assert torch.allclose(base[0][0, :t + 1], new[0][0, :t + 1], **TOL)
        assert torch.allclose(base[2][0, :t + 1], new[2][0, :t + 1], **TOL)


@pytest.mark.parametrize("cond", ALL)
def test_future_tokens_do_not_change_past_outputs_or_routing(msm, cond):
    ids = rids(120, 4)
    starts = [0, 20, 51, 90]
    cut = 63
    ids2 = ids.copy()
    ids2[cut + 1:] = rids(120 - cut - 1, 77)
    r = routing_for(cond)
    with torch.no_grad():
        o1 = msm(doc_batch(ids, starts), r, collect=True)
        o2 = msm(doc_batch(ids2, starts), r, collect=True)
    assert torch.allclose(o1.logits[0, :cut + 1], o2.logits[0, :cut + 1], **TOL)
    for a, b in zip(o1.routing, o2.routing):
        assert torch.equal(a["topi"][0, :cut + 1], b["topi"][0, :cut + 1])
        assert torch.allclose(a["z_route"][0, :cut + 1], b["z_route"][0, :cut + 1], **TOL)


@pytest.mark.parametrize("cond", ALL)
def test_future_chunk_structure_cannot_change_past(msm, cond):
    ids = rids(120, 5)
    cut = 70
    s1 = [0, 30, 64, 80, 100]
    s2 = [0, 30, 64, 95]  # identical up to the cut; later endpoints/lengths differ
    r = routing_for(cond)
    with torch.no_grad():
        o1 = msm(doc_batch(ids, s1), r)
        o2 = msm(doc_batch(ids, s2), r)
    assert torch.allclose(o1.logits[0, :cut + 1], o2.logits[0, :cut + 1], **TOL)


@pytest.mark.parametrize("cond", ALL)
def test_document_isolation_in_packed_rows(msm, cond):
    doc, pos, cid, cpos = packed_two_docs(30, 25, 64, (0, 10), (0, 12))
    d1, d2 = rids(30, 6), rids(25, 7)
    ids = np.full(64, 1)
    ids[:30], ids[30:55] = d1, d2
    b = {"input_ids": torch.as_tensor(ids).unsqueeze(0), "doc_ids": doc, "positions": pos, "chunk_ids": cid,
         "chunk_pos": cpos}
    r = routing_for(cond)
    with torch.no_grad():
        packed = msm(b, r).logits[0]
        alone = msm(doc_batch(d2, [0, 12]), r).logits[0]
    assert torch.allclose(packed[30:55], alone, rtol=0, atol=1e-9)


@pytest.mark.parametrize("cond", ALL)
def test_incremental_matches_full_sequence(msm, cond):
    ids = rids(90, 8)
    starts = [0, 9, 30, 47, 70]
    _, cpos = starts_to_arrays(90, starts)
    r = routing_for(cond, 0.5, 1.0)
    with torch.no_grad():
        par = msm(doc_batch(ids, starts), r, collect=True)
        st = msm.new_ms_state()
        for t in range(90):
            lg, rt = msm.ms_step(int(ids[t]), t, int(cpos[t]), r, st)
            assert torch.allclose(lg, par.logits[0, t], rtol=0, atol=1e-9), (cond, t)
            for li, (topi, zr) in enumerate(rt):
                assert torch.equal(topi, par.routing[li]["topi"][0, t])
                assert torch.allclose(zr, par.routing[li]["z_route"][0, t], rtol=0, atol=1e-9)


@pytest.mark.parametrize("cond", ALL)
def test_exactly_four_distinct_global_experts_no_pool(msm, cond):
    ids = rids(100, 9)
    with torch.no_grad():
        out = msm(doc_batch(ids, [0, 33, 70]), routing_for(cond), collect=True)
    for r in out.routing:
        topi = r["topi"][0]
        assert topi.shape[-1] == 4
        assert all(len(set(row.tolist())) == 4 for row in topi)
        # selection is the unrestricted global top-4 of softmax(z_route) over all experts
        expect = r["probs_full"][0].topk(4, dim=-1).indices
        assert torch.equal(topi, expect)
        assert (r["probs_full"][0] > 0).all()
        assert torch.allclose(r["gates"][0], r["probs_full"][0].gather(1, topi), **TOL)


# ---------------------------------------------------------------- schedules and scale controls
def test_schedules():
    n = lambda k: sum(x != "token" for x in SCHEDULES[k])
    assert n("BROAD-EARLY") == 3 and n("BROAD-LATE") == 3
    assert SCHEDULES["BROAD-EARLY"][:3] == ("broad",) * 3 and SCHEDULES["BROAD-LATE"][4:] == ("broad",) * 3
    assert SCHEDULES["LOCAL"] == ("token",) * 2 + ("local",) * 5
    assert SCHEDULES["HIER"] == ("token",) * 2 + ("local",) * 2 + ("hier",) * 3
    assert SCHEDULES["TOKEN"] == ("token",) * 7
    for c, (sched, seg, scale, ref) in CONDITIONS.items():
        if scale:
            rs, rseg, rscale, _ = CONDITIONS[ref]
            assert sched == rs and seg == rseg and not rscale
    assert CONTEXTUAL == set(CONDITIONS) - {"TOKEN", "SCALE-ONLY-LOCAL", "SCALE-ONLY-BROAD-LATE", "SCALE-ONLY-HIER"}


@pytest.mark.parametrize("ctrl", ["SCALE-ONLY-LOCAL", "SCALE-ONLY-BROAD-LATE", "SCALE-ONLY-HIER"])
def test_scale_only_preserves_token_ranking_and_matches_reference_rms(msm, ctrl):
    ref = CONDITIONS[ctrl][3]
    ids = rids(150, 10)
    b = doc_batch(ids, [0, 40, 100])
    with torch.no_grad():
        oc = msm(b, routing_for(ctrl, 0.5, 1.0), collect=True)
    for li, r in enumerate(oc.routing):
        zt, zr = r["z_token"][0], r["z_route"][0]
        assert torch.equal(zt.argsort(dim=-1, descending=True, stable=True), zr.argsort(dim=-1, descending=True, stable=True))
        assert torch.equal(r["topi"][0], zt.topk(4, -1).indices)
        if r["kind"] == "token":
            assert torch.equal(zr, zt)
            continue
        # hypothetical reference z_raw at the same hidden states: z_token + context contribution
        z_raw = zt + r["ctx"][0]
        assert torch.allclose(rms(center(zr)), rms(center(z_raw)), rtol=1e-12, atol=1e-12)
        assert torch.allclose(zr.mean(-1), zt.mean(-1), rtol=0, atol=1e-12)


def test_scale_only_reference_context_equals_reference_condition(msm):
    # at identical hidden states (layer 3, first contextual layer: earlier layers are TOKEN in both) the control's
    # hypothetical z_raw equals the reference condition's actual z_route
    ids = rids(120, 11)
    b = doc_batch(ids, [0, 35, 80])
    for ctrl in ("SCALE-ONLY-LOCAL", "SCALE-ONLY-HIER", "SCALE-ONLY-BROAD-LATE"):
        ref = CONDITIONS[ctrl][3]
        with torch.no_grad():
            oc = msm(b, routing_for(ctrl, 0.5, 1.0), collect=True)
            orf = msm(b, routing_for(ref, 0.5, 1.0), collect=True)
        li = next(i for i, r in enumerate(oc.routing) if r["kind"] != "token")
        assert torch.allclose(oc.routing[li]["z_token"] + oc.routing[li]["ctx"], orf.routing[li]["z_route"], **TOL)


def test_scale_control_scalar_is_detached():
    torch.manual_seed(4)
    m = MultiscaleModel(ms_cfg()).double()
    moe = m.moe_blocks()[4].ffn
    h = torch.randn(10, 32, dtype=torch.float64, requires_grad=True)
    ml = torch.randn(10, 32, dtype=torch.float64, requires_grad=True)
    mb = torch.randn(10, 32, dtype=torch.float64, requires_grad=True)
    ctx = {"m_local": ml, "m_broad": mb, "ramp": torch.ones(10, dtype=torch.float64)}
    r = routing_for("SCALE-ONLY-HIER", 0.5, 1.0)
    z = m.route_logits(moe, h, "hier", r, ctx)
    (z * torch.randn_like(z)).sum().backward()
    assert ml.grad is None and mb.grad is None  # no gradient path through the context or the scale
    assert h.grad is not None and h.grad.abs().sum() > 0
    rr = routing_for("HIER-SEMANTIC", 0.5, 1.0)
    ml2, mb2 = ml.detach().clone().requires_grad_(), mb.detach().clone().requires_grad_()
    z2 = m.route_logits(moe, h.detach(), "hier", rr, {"m_local": ml2, "m_broad": mb2, "ramp": ctx["ramp"]})
    (z2 * torch.randn_like(z2)).sum().backward()
    assert ml2.grad.abs().sum() > 0 and mb2.grad.abs().sum() > 0


@pytest.mark.parametrize("cond", ["BROAD-LATE", "LOCAL-SEMANTIC", "HIER-RANDOM", "BROAD-EARLY"])
def test_lm_gradients_reach_context_path_and_are_finite(cond):
    torch.manual_seed(5)
    m = MultiscaleModel(ms_cfg()).double()
    ids = rids(120, 12)
    b = doc_batch(ids, [0, 40, 90])
    r = routing_for(cond)
    out = m(b, r, collect=True)
    # loss only on the final token: earlier hidden states at a contextual layer must get gradient via the context
    li = next(i for i, q in enumerate(out.routing) if q["kind"] != "token")
    h = out.routing[li]["h"]
    h.retain_grad()
    for q in out.routing:
        q["h"].retain_grad()
    loss = F.cross_entropy(out.logits[0, -1:], torch.as_tensor(ids[-1:]))
    loss = loss + 0.01 * out.aux_loss + 0.001 * out.z_loss
    assert torch.isfinite(loss)
    loss.backward()
    for p in m.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()
    assert torch.isfinite(out.logits).all()
    # gradient into the context means (non-detached path) -- compare with a detached-context counterfactual
    moe = m.moe_blocks()[li].ffn
    kind = out.routing[li]["kind"]
    hh = h.detach().clone().requires_grad_()
    pos = b["positions"]
    m_l, n_l, m_b, n_b = context_means(hh, pos, b["chunk_pos"], b["doc_ids"] >= 0, r.uses_local(kind), r.uses_broad(kind))
    ctx = {}
    if m_l is not None:
        ctx["m_local"] = m_l.reshape(120, -1)
    if m_b is not None:
        ctx["m_broad"] = m_b.reshape(120, -1)
        ctx["ramp"] = (n_b.double() / 512).clamp(max=1).reshape(-1)
    z = m.route_logits(moe, hh.reshape(120, -1), kind, r, ctx)
    z[-1].pow(2).sum().backward()
    g = hh.grad[0, :-1].abs().sum(-1)
    assert g.sum() > 0  # earlier tokens receive gradient from the last token's routing logits


# ---------------------------------------------------------------- semantic boundaries (prefix reproducibility)
def test_online_semantic_boundaries_are_prefix_reproducible_and_future_invariant():
    from csr.segment import sentence_spans
    det = fake_detector()
    p = SemanticParams(0.5, min_len=20, max_len=60)
    rng = np.random.default_rng(0)
    for trial in range(5):
        ids = np.asarray(topic_doc(rng), dtype=np.int64)
        L = len(ids)
        ends = det.sentence_ends(ids)
        embs = fake_embed([ids[s:e + 1] for s, e in sentence_spans(ends)])
        batch = semantic_starts(L, ends, embs, p)
        ch = OnlineSemanticChunker(det, fake_embed, p, L=L)
        announced = {}
        for t, tok in enumerate(ids.tolist()):
            if ch.push(tok):
                announced[t + 1] = t
        assert ch.starts == batch
        for b in batch[1:]:
            assert announced[b] == b - 1  # available only after consuming token b-1
        # future perturbation after a sentence end keeps every boundary <= cut+1
        cut = int(ends[len(ends) // 2])
        ids2 = ids.copy()
        ids2[cut + 1:] = np.asarray(topic_doc(np.random.default_rng(100 + trial)), dtype=np.int64)[:L - cut - 1] \
            if len(topic_doc(np.random.default_rng(100 + trial))) >= L - cut - 1 else ids2[cut + 1:][::-1]
        ch2 = OnlineSemanticChunker(det, fake_embed, p, L=L)
        for tok in ids2.tolist():
            ch2.push(tok)
        assert [s for s in ch2.starts if s <= cut + 1] == [s for s in batch if s <= cut + 1]


# ---------------------------------------------------------------- configs, data guards, selection logic
def _base():
    return json.loads(open("/home/zapan4/moe-research/configs/rep-v1/seed0/condition_A.json").read())


def test_configs_differ_only_in_preregistered_fields():
    base = _base()
    cfgs = {c: config_for(base, c, 0.5, 0.25, f"r_{c}") for c in CONDITIONS}
    assert check_matched(cfgs)
    bad = copy.deepcopy(cfgs)
    bad["TOKEN"]["train"]["lr"] = 1e-3
    with pytest.raises(ValueError):
        check_matched(bad)
    cal = {f"{c}{a}": config_for(base, c, a, 0.0, f"x{c}{a}") for c in ("LOCAL-FIXED", "LOCAL-RANDOM") for a in (0, 0.5)}
    with pytest.raises(ValueError):
        check_matched(cal)  # alpha differs: not approved for main
    assert check_matched(cal, APPROVED_DIFF_CALIB)


def test_locked_test_split_cannot_be_loaded(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(np, "load", lambda *a, **k: calls.append(a) or (_ for _ in ()).throw(AssertionError("np.load")))
    for split in ("test", "TEST", "../test", None):
        with pytest.raises(SplitLocked):
            load_split(tmp_path, split)
        with pytest.raises(SplitLocked):
            load_starts({"segmentation": {"strategy": "fixed"}}, split, tmp_path)
    assert calls == []


def _res(coef, ces, unused=None, share=0.2):
    out = []
    for (seed, pl), ce in ces.items():
        out.append({"coef": coef, "seed": seed, "placement": pl, "ce": ce, "finite": True, "status": "COMPLETE",
                    "max_share": [share] * 7, "unused": (unused or {}).get((seed, pl), [0] * 7)})
    return out


def _grid(zero, deltas):
    res = []
    for c, d in deltas.items():
        res += _res(c, {k: v + (d[k] if isinstance(d, dict) else d) for k, v in zero.items()})
    return res


ZERO = {(98, "BROAD-EARLY"): 5.0, (98, "BROAD-LATE"): 5.0, (99, "BROAD-EARLY"): 5.1, (99, "BROAD-LATE"): 5.1}


def test_calibration_aggregates_over_all_required_runs_and_selects_deterministically():
    res = _grid(ZERO, {0.0: 0.0, 0.25: -0.003, 0.5: -0.010, 1.0: -0.0095})
    c, t = ms_select.select_coefficient(res)
    assert c == 0.5 and t[0.5]["n_runs"] == 4 and abs(t[0.5]["diff_vs_zero"] + 0.010) < 1e-12
    # tie within 1e-4 -> smaller coefficient
    c, _ = ms_select.select_coefficient(_grid(ZERO, {0.0: 0, 0.25: -0.01, 0.5: -0.01005, 1.0: 0.0}))
    assert c == 0.25
    # no gain >= 0.002 -> zero wins
    c, _ = ms_select.select_coefficient(_grid(ZERO, {0.0: 0, 0.25: -0.001, 0.5: 0.01, 1.0: 0.02}))
    assert c == 0.0
    # beneficial in only one seed -> not qualifying
    d = {k: (-0.02 if k[0] == 98 else 0.001) for k in ZERO}
    c, t = ms_select.select_coefficient(_grid(ZERO, {0.0: 0, 0.25: d, 0.5: 0.0, 1.0: 0.0}))
    assert c == 0.0 and not t[0.25]["both_seeds_beneficial"]
    # one constituent run > 0.10 worse -> rejected even if aggregate improves
    d = {k: -0.2 for k in ZERO}
    d[(99, "BROAD-LATE")] = 0.11
    c, t = ms_select.select_coefficient(_grid(ZERO, {0.0: 0, 0.25: d, 0.5: 0, 1.0: 0}))
    assert c == 0.0 and t[0.25]["rejected"]
    # catastrophic placement (two-seed mean > +0.05) -> not qualifying
    d = {k: (-0.2 if k[1] == "BROAD-EARLY" else 0.06) for k in ZERO}
    c, t = ms_select.select_coefficient(_grid(ZERO, {0.0: 0, 0.25: d, 0.5: 0, 1.0: 0}))
    assert c == 0.0 and not t[0.25]["no_catastrophic_placement"]
    # method-attributable unused expert / collapse -> rejected
    res = _grid(ZERO, {0.0: 0, 0.25: 0, 0.5: 0, 1.0: 0}) + _res(0.25, {k: v - 0.01 for k, v in ZERO.items()},
                                                                  unused={(98, "BROAD-LATE"): [0, 0, 0, 0, 1, 0, 0]})
    res = [r for r in res if not (r["coef"] == 0.25 and r["ce"] == ZERO[(r["seed"], r["placement"])])]
    c, t = ms_select.select_coefficient(res)
    assert c == 0.0 and t[0.25]["rejected"]
    res = _grid(ZERO, {0.0: 0, 0.5: 0, 1.0: 0}) + _res(0.25, {k: v - 0.01 for k, v in ZERO.items()}, share=0.6)
    c, t = ms_select.select_coefficient(res)
    assert c == 0.0 and t[0.25]["rejected"]
    # incomplete design -> rejected
    res = [r for r in _grid(ZERO, {0.0: 0, 0.25: -0.05, 0.5: 0, 1.0: 0}) if not (r["coef"] == 0.25 and r["seed"] == 99)]
    c, t = ms_select.select_coefficient(res)
    assert c == 0.0
    # repeat call is identical (determinism)
    a = ms_select.select_coefficient(_grid(ZERO, {0.0: 0, 0.25: -0.003, 0.5: -0.01, 1.0: -0.0095}))
    b = ms_select.select_coefficient(_grid(ZERO, {0.0: 0, 0.25: -0.003, 0.5: -0.01, 1.0: -0.0095}))
    assert a[0] == b[0] and json.dumps(a[1], sort_keys=True, default=str) == json.dumps(b[1], sort_keys=True, default=str)


def test_branch_selection():
    assert branch_of(0, 0) == "A" and branch_of(0.5, 0) == "B" and branch_of(0, 0.25) == "C" and branch_of(1, 1) == "D"
    assert BRANCH_CONDITIONS["A"] == []
    assert BRANCH_CONDITIONS["C"] == ["TOKEN", "SCALE-ONLY-BROAD-LATE", "BROAD-EARLY", "BROAD-LATE"]
    assert len(BRANCH_CONDITIONS["D"]) == 10 and BRANCH_CONDITIONS["D"][:2] == ["TOKEN", "SCALE-ONLY-HIER"]


def test_budget_selection_and_pilot_futility():
    tps = {c: 100000.0 for c in BRANCH_CONDITIONS["D"]}
    b, t = ms_select.select_budget(0, tps, 60, 90, 22.0, 120 * 1024**3, 200_000_000)
    assert b == 100_000_000
    slow = {c: 30000.0 for c in tps}
    b, t = ms_select.select_budget(3600, slow, 60, 90, 22.0, 120 * 1024**3, 200_000_000)
    assert b in (50_000_000, 25_000_000, None) and b != 100_000_000
    b, _ = ms_select.select_budget(0, tps, 60, 90, 57.0, 120 * 1024**3, 1)
    assert b is None
    ce = {"TOKEN": 4.0, "BROAD-LATE": 4.06, "BROAD-EARLY": 4.07, "SCALE-ONLY-BROAD-LATE": 4.0}
    sl = {"TOKEN": -1e-8, "BROAD-LATE": -0.5e-8, "BROAD-EARLY": -0.9e-8, "SCALE-ONLY-BROAD-LATE": -5e-8}
    stop, _ = ms_select.pilot_futility(ce, sl, CONTEXTUAL)
    assert stop  # scale control is not contextual, so its better slope does not prevent the stop
    sl["BROAD-LATE"] = -2e-8
    assert not ms_select.pilot_futility(ce, sl, CONTEXTUAL)[0]
    ce["BROAD-LATE"] = 4.04
    sl["BROAD-LATE"] = 0
    assert not ms_select.pilot_futility(ce, sl, CONTEXTUAL)[0]


def test_paired_bootstrap_multi_support_rule():
    rng = np.random.default_rng(1)
    n = np.full(400, 100.0)
    y = {s: rng.normal(400, 20, 400) for s in range(3)}
    x = {s: y[s] - 2.0 for s in range(3)}  # -0.02 nats/token in every seed
    out = ms_select.paired_bootstrap_multi(x, y, n)
    assert out["support"] and abs(out["mean_diff"] + 0.02) < 1e-12 and out["mean_ci95"][1] < 0
    x[2] = y[2] + 0.5
    assert not ms_select.paired_bootstrap_multi(x, y, n)["support"]


def test_restart_from_scratch_is_deterministic():
    from csr.train_lib import lr_at

    def run():
        torch.manual_seed(7)
        m = MultiscaleModel(ms_cfg())
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        r = routing_for("HIER-SEMANTIC", 0.5, 0.5)
        for step in range(3):
            ids = rids(64, 20 + step)
            out = m(doc_batch(ids, [0, 20, 45]), r)
            loss = F.cross_entropy(out.logits[0, :-1], torch.as_tensor(ids[1:])) + 0.01 * out.aux_loss + 1e-3 * out.z_loss
            loss.backward()
            opt.step()
            opt.zero_grad()
        return [p.detach().clone() for p in m.parameters()]
    a, b = run(), run()
    assert all(torch.equal(x, y) for x, y in zip(a, b))


# ---------------------------------------------------------------- counterfactual audit pass (secondary analysis)
@pytest.mark.parametrize("cond", ["TOKEN", "LOCAL-SEMANTIC", "HIER-RANDOM", "SCALE-ONLY-HIER", "BROAD-EARLY"])
def test_ms_counterfactual_matches_bruteforce(msm, cond):
    from csr.ms_audit import ms_counterfactual, ms_original_pass
    ids = rids(90, 31)
    starts = [0, 25, 60]
    b = doc_batch(ids, starts)
    r = routing_for(cond, 0.5, 1.0)
    o = ms_original_pass(msm, b, r)
    with torch.no_grad():
        ref = msm(b, r)
    assert torch.allclose(o["logits"], ref.logits, rtol=0, atol=1e-9)
    for L in (0, 3, 6):
        qpos = torch.tensor([[5, 30, 61, 88]])
        rm = o["records"][L]["topi"][0, qpos[0], 1].unsqueeze(0)  # remove the rank-1 expert of each query
        cf = ms_counterfactual(msm, o, r, L, qpos, torch.ones_like(qpos, dtype=torch.bool), rm)
        assert cf["affected"].all()
        for j, t in enumerate(qpos[0].tolist()):
            # brute force: prefix up to t (t is last, so only its own stream changes), expert removed for t at L
            class Brute(MultiscaleModel):
                def __init__(s):
                    nn_init = False
                    torch.nn.Module.__init__(s)
                    s.__dict__.update({k: v for k, v in msm.__dict__.items() if k != "_modules"})
                    s._modules = msm._modules
                    s.layer_seen = 0
            brute = Brute()
            calls = {"i": 0}
            orig_sel = MultiscaleModel.select_apply

            def sel(moe, h_flat, z_route, _L=L, _e=int(rm[0, j]), calls=calls):
                li = calls["i"]
                calls["i"] += 1
                if li != _L:
                    return orig_sel(moe, h_flat, z_route)
                probs = torch.softmax(z_route, dim=-1)
                p2 = probs.clone()
                p2[-1, _e] = -1.0
                gates, topi = p2.topk(moe.k, dim=-1)
                gates = probs.gather(1, topi)
                y = moe.experts(h_flat, topi, gates)
                if moe.shared is not None:
                    y = y + moe.shared(h_flat)
                return y, topi, gates, probs
            brute.select_apply = sel
            bp = doc_batch(ids[:t + 1], [s for s in starts if s <= t])
            with torch.no_grad():
                out = brute(bp, r)
            assert torch.allclose(cf["logits"][0, j], out.logits[0, t], rtol=0, atol=1e-9), (cond, L, t)
