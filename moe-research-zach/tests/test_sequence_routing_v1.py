"""csr-sequence-routing-search-v1 correctness tests (CPU, CUDA hidden). docs/SEQUENCE_ROUTING_SEARCH_V1_PREREG.md §5."""
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import tiny_cfg
from csr.model import CSRModel, RoutingConfig
from csr.seqroute import (SRRouting, SeqRouteModel, build_sr_model, fixed_starts_mask, scan_dynamic_starts,
                          seg_index, window_change_score)
from csr import srs_select as sel
from csr import srs_run

ROOT = Path(__file__).resolve().parents[1]
K = 4


def cfg_(**kw):
    base = dict(n_layer=4, moe_layers=(1, 2, 3), n_experts=8, top_k=K, d_model=32, n_head=4, dense_ffn=48,
                expert_ffn=16, shared_ffn=24, max_seq_len=1024, vocab_size=97)
    base.update(kw)
    return tiny_cfg(**base)


VARIANTS = {
    "F16": dict(seg="fixed", seg_len=16), "F32": dict(seg="fixed", seg_len=32), "F64": dict(seg="fixed", seg_len=64),
    "F128": dict(seg="fixed", seg_len=128), "F256": dict(seg="fixed", seg_len=256),
    "F32-OPEN16": dict(seg="fixed", seg_len=32, opening=16),
    "F32-GRU": dict(seg="fixed", seg_len=32, state="gru"), "F32-ATTN": dict(seg="fixed", seg_len=32, summary="attn"),
    "F16-EMA": dict(seg="fixed", seg_len=16, state="ema"), "F32-LAST": dict(seg="fixed", seg_len=32, summary="last"),
    "CP32": dict(seg="cp", target_len=32), "CP64-L0": dict(seg="cp", target_len=64, boundary_source="layer0"),
    "LBD64": dict(seg="lbd", target_len=64), "CP32-OPEN16-GRU-ATTN": dict(seg="cp", target_len=32, opening=16,
                                                                          state="gru", summary="attn"),
    "CP32-T05-SEGLB": dict(seg="cp", target_len=32, temperature=0.5, lb_weighting="segment"),
}
SEQ = sorted(VARIANTS)


def rids(n, seed=0, V=97):
    return np.random.default_rng(seed).integers(2, V, size=n).astype(np.int64)


def batch(docs_rows, T):
    B = len(docs_rows)
    a = {k: np.full((B, T), f, np.int64) for k, f in (("input_ids", 1), ("doc_ids", -1), ("positions", 0),
                                                      ("targets", -100), ("gdoc", -1))}
    g = 0
    for b, row in enumerate(docs_rows):
        t = 0
        for k, ids in enumerate(row):
            L = len(ids)
            a["input_ids"][b, t:t + L] = ids
            a["doc_ids"][b, t:t + L] = k
            a["positions"][b, t:t + L] = np.arange(L)
            a["targets"][b, t:t + L - 1] = ids[1:]
            a["gdoc"][b, t:t + L] = g
            g += 1
            t += L
    return {k: torch.as_tensor(v) for k, v in a.items()}


def model_for(name, seed=0, calib_batch=None, **cfgkw):
    torch.manual_seed(seed)
    m = SeqRouteModel(cfg_(**cfgkw), SRRouting(**VARIANTS[name]) if name != "TOKEN" else SRRouting(mode="token"),
                      extra_seed=seed).double()
    if m.sr.dynamic:
        b = calib_batch if calib_batch is not None else batch([[rids(400, 99), rids(300, 98)], [rids(700, 97)]], 720)
        m.train()
        m.tau_calibrate(b)
    m.eval()
    return m


@pytest.fixture(scope="module")
def models():
    return {n: model_for(n) for n in SEQ + ["TOKEN"]}


@pytest.fixture(scope="module")
def two_doc_batch():
    return batch([[rids(300, 1), rids(130, 2)], [rids(500, 3)]], 512)


def fwd(m, b):
    with torch.no_grad():
        return m(b, collect=True)


# ---------------------------------------------------------------- 1, 2: identical ids and weights within a segment
@pytest.mark.parametrize("name", SEQ)
def test_segment_tokens_share_expert_ids_and_weights(models, two_doc_batch, name):
    m, b = models[name], two_doc_batch
    o = fwd(m, b)
    npm = b["doc_ids"] >= 0
    for r in o.routing:
        gid = r["seg_gid"]
        for g in torch.unique(gid[npm]).tolist():
            sel_ = gid == g
            ids = r["topi"][sel_]
            gates = r["gates"][sel_]
            assert (ids == ids[0]).all(), "expert identities differ inside a segment"
            assert (gates == gates[0]).all(), "routing weights differ inside a segment"
            assert torch.equal(ids[0], r["seg_topi"][g]) and torch.equal(gates[0], r["seg_gates"][g])


# ---------------------------------------------------------------- 3: route fixed before the segment begins
@pytest.mark.parametrize("name", SEQ)
def test_changing_token_inside_segment_keeps_its_route(models, name):
    m = models[name]
    ids = rids(400, 5)
    o = fwd(m, batch([[ids]], 400))
    r0 = o.routing[1]
    # pick a segment j >= 1 and change a token in its middle
    j1 = (r0["seg_j"] >= 1).nonzero().squeeze(1)[0].item()
    first = int(r0["seg_first"][j1])
    cnt = int(r0["seg_count"][j1])
    pos = first + cnt // 2
    ids2 = ids.copy()
    ids2[pos] = (ids2[pos] + 7) % 95 + 2
    o2 = fwd(m, batch([[ids2]], 400))
    for x, y in zip(o.routing, o2.routing):
        # routes of every token up to and including the changed one are unchanged
        assert torch.equal(x["topi"][0, :pos + 1], y["topi"][0, :pos + 1])
        assert torch.equal(x["gates"][0, :pos + 1], y["gates"][0, :pos + 1])
        # the containing segment keeps its route (its extent may end after the changed token: boundaries are
        # decided after observing each token)
        gx, gy = int(x["seg_gid"][0, pos]), int(y["seg_gid"][0, pos])
        assert torch.equal(x["seg_topi"][gx], y["seg_topi"][gy]) and torch.equal(x["seg_gates"][gx], y["seg_gates"][gy])
        if m.sr.seg == "fixed":
            a, n = int(x["seg_first"][gx]), int(x["seg_count"][gx])
            assert torch.equal(x["topi"][0, a:a + n], y["topi"][0, a:a + n])


# ---------------------------------------------------------------- 4, 14: future invariance
@pytest.mark.parametrize("name", SEQ + ["TOKEN"])
def test_future_text_never_changes_earlier_boundaries_routes_or_outputs(models, name):
    m = models[name]
    ids = rids(600, 11)
    for cut in (37, 150, 333):
        ids2 = ids.copy()
        ids2[cut + 1:] = rids(600, 12)[cut + 1:]
        o1, o2 = fwd(m, batch([[ids]], 600)), fwd(m, batch([[ids2]], 600))
        assert torch.allclose(o1.logits[0, :cut + 1], o2.logits[0, :cut + 1], rtol=0, atol=1e-10)
        for x, y in zip(o1.routing, o2.routing):
            e = cut + 1 if name == "TOKEN" else cut + 2  # sequence: the route of token cut+1 was decided at cut
            assert torch.equal(x["topi"][0, :e], y["topi"][0, :e])
            if name != "TOKEN":
                assert torch.equal(x["starts"][0, :cut + 2], y["starts"][0, :cut + 2])
        # shorter document (future length) does not change anything earlier either
        o3 = fwd(m, batch([[ids[:cut + 1]]], 600))
        assert torch.allclose(o1.logits[0, :cut + 1], o3.logits[0, :cut + 1], rtol=0, atol=1e-10)


# ---------------------------------------------------------------- 5: dependence on the completed previous segment
@pytest.mark.parametrize("name", ["F16", "F32-GRU", "F32-ATTN", "CP32", "F16-EMA"])
def test_next_route_depends_on_completed_previous_segment(models, name):
    m = models[name]
    ids = rids(300, 21)
    o = fwd(m, batch([[ids]], 300))
    changed = 0
    for trial in range(12):
        ids2 = ids.copy()
        r = o.routing[0]
        g = 1  # second segment
        a, n = int(r["seg_first"][g]), int(r["seg_count"][g])
        ids2[a:a + n] = rids(n, 100 + trial)
        o2 = fwd(m, batch([[ids2]], 300))
        g3 = 2
        a3 = int(o.routing[0]["seg_first"][g3])
        if name.startswith("CP"):
            changed += sum(int(not torch.equal(x["topi"][0, a + n:], y["topi"][0, a + n:]))
                           for x, y in zip(o.routing, o2.routing))
        else:
            changed += sum(int(not torch.equal(x["z_seg"][g3], y["z_seg"][g3]))
                           for x, y in zip(o.routing, o2.routing))
            assert torch.equal(o.routing[0]["topi"][0, :a3], o2.routing[0]["topi"][0, :a3])
    assert changed > 0, "later routes never respond to the content of the completed previous segment"


# ---------------------------------------------------------------- 6, 7: document isolation and content-free BOS
@pytest.mark.parametrize("name", SEQ)
def test_routes_never_cross_documents_and_first_segment_is_content_free(models, name):
    m = models[name]
    d1, d2 = rids(300, 31), rids(220, 32)
    packed = fwd(m, batch([[d1, d2]], 600))
    alone = fwd(m, batch([[d2]], 600))
    for li, (x, y) in enumerate(zip(packed.routing, alone.routing)):
        # segments never span the document boundary
        assert x["starts"][0, 300]
        assert torch.equal(x["topi"][0, 300:520], y["topi"][0, :220])
        assert torch.equal(x["starts"][0, 300:520], y["starts"][0, :220])
        # first-segment route = BOS route of the layer for every document (no document content)
        bos = torch.softmax(m.bos[li] / m.sr.temperature, -1).topk(K).indices
        for xx in (x, y):
            j0 = xx["seg_j"] == 0
            assert (xx["seg_topi"][j0] == bos).all()
    assert torch.allclose(packed.logits[0, 300:520], alone.logits[0, :220], atol=1e-10, rtol=0)


@pytest.mark.parametrize("name", ["F64", "CP32", "LBD64"])
def test_first_segment_route_identical_across_documents(models, name):
    m = models[name]
    o = fwd(m, batch([[rids(100, 41)], [rids(100, 42)], [rids(100, 43)]], 100))
    for r in o.routing:
        assert (r["topi"][:, 0] == r["topi"][0, 0]).all()
        assert (r["gates"][:, 0] == r["gates"][0, 0]).all()


# ---------------------------------------------------------------- 8, 9: incremental == full; train == inference
@pytest.mark.parametrize("name", SEQ + ["TOKEN"])
def test_incremental_decoding_reproduces_full_sequence(models, name):
    m = models[name]
    docs = [rids(333, 51), rids(333, 52), rids(333, 53)]
    full = [fwd(m, batch([[d]], 333)) for d in docs]
    st = m.new_sr_state(3)
    for p in range(333):
        lg, rts = m.sr_step(torch.tensor([int(d[p]) for d in docs]), p, st)
        for bi in range(3):
            assert torch.allclose(lg[bi], full[bi].logits[0, p], atol=1e-9, rtol=0)
            for li, rr in enumerate(rts):
                assert torch.equal(rr[bi], full[bi].routing[li]["topi"][0, p])
    if name != "TOKEN":
        n_segments = sum(int(r["S"]) for r in full[0].routing) - len(full[0].routing)
        assert st["router_calls"] >= 1  # one router call per completed segment (batched step: any completion)


@pytest.mark.parametrize("name", SEQ)
def test_training_and_inference_use_same_routing(name):
    m = model_for(name)
    b = batch([[rids(400, 61), rids(200, 62)]], 620)
    m.eval()
    with torch.no_grad():
        oe = m(b, collect=True)
    m.train()
    ot = m(b, collect=True)
    for x, y in zip(oe.routing, ot.routing):
        assert torch.equal(x["topi"], y["topi"])
        assert torch.equal(x["starts"], y["starts"])
        assert torch.allclose(x["gates"], y["gates"].detach(), atol=0, rtol=0)
    assert torch.allclose(oe.logits, ot.logits.detach(), atol=1e-12, rtol=0)


# ---------------------------------------------------------------- 10, 11: top-4 dispatch; no token-dependent routing
@pytest.mark.parametrize("name", SEQ + ["TOKEN"])
def test_exactly_top4_routed_experts_per_token(models, two_doc_batch, name):
    m = models[name]
    seen = []
    orig = type(m.moe_blocks()[0].ffn).experts

    def spy(self, x, topi, gates):
        seen.append((x.shape[0], topi.clone()))
        return orig(self, x, topi, gates)

    type(m.moe_blocks()[0].ffn).experts = spy
    try:
        fwd(m, two_doc_batch)
    finally:
        type(m.moe_blocks()[0].ffn).experts = orig
    assert len(seen) == 3
    for n, topi in seen:
        assert topi.shape == (n, K)
        assert (torch.sort(topi, -1).values.diff(dim=-1) > 0).all(), "duplicate expert ids"


@pytest.mark.parametrize("name", SEQ)
def test_no_token_dependent_routing_path(models, two_doc_batch, name):
    m = models[name]
    rows = []
    for blk in m.moe_blocks():
        blk.ffn._orig_rl = blk.ffn.router_logits
        blk.ffn.router_logits = (lambda f: (lambda x: (rows.append(x.shape[0]), f(x))[1]))(blk.ffn._orig_rl)
    try:
        o = fwd(m, two_doc_batch)
    finally:
        for blk in m.moe_blocks():
            blk.ffn.router_logits = blk.ffn._orig_rl
            del blk.ffn._orig_rl
    n_tok = int((two_doc_batch["doc_ids"] >= 0).sum())
    S = [int(r["S"]) for r in o.routing]
    assert all(r < n_tok for r in rows), "router applied to a token-level tensor"
    if m.sr.state == "none":
        assert rows == S  # exactly one router row per segment summary
    # token-level perturbation that preserves each segment's pooled summary leaves every route unchanged:
    # routes of segment j only use summaries of segments < j, so randomizing the LAST segment of each document
    # cannot change any route.
    b = {k: v.clone() for k, v in two_doc_batch.items()}
    r0 = o.routing[0]
    last = [g for g in range(r0["S"]) if g == r0["S"] - 1 or int(r0["seg_j"][g + 1]) == 0]
    flat = b["input_ids"].view(-1)
    for g in last:
        a, n = int(r0["seg_first"][g]), int(r0["seg_count"][g])
        flat[a:a + n] = torch.as_tensor(rids(n, 7 + g))
    o2 = fwd(m, b)
    if m.sr.seg == "fixed":
        for x, y in zip(o.routing, o2.routing):
            assert torch.equal(x["topi"], y["topi"])


def test_token_mode_equals_original_token_router():
    torch.manual_seed(3)
    m = SeqRouteModel(cfg_(), SRRouting(mode="token")).double().eval()
    torch.manual_seed(3)
    ref = CSRModel(cfg_()).double().eval()
    assert all(torch.equal(a, b) for a, b in zip(m.state_dict().values(), ref.state_dict().values()))
    b = batch([[rids(200, 71), rids(100, 72)]], 320)
    b["chunk_ids"] = torch.zeros_like(b["input_ids"])
    b["chunk_pos"] = b["positions"].clone()
    with torch.no_grad():
        o1 = m(b, collect=True)
        o2 = ref(b, RoutingConfig(mode="token", pool_size=8, warmup=16))
    assert torch.equal(o1.logits, o2.logits)
    assert torch.allclose(o1.aux_loss, o2.aux_loss) and torch.allclose(o1.z_loss, o2.z_loss)
    for x, y in zip(o1.routing, o2.routing):
        assert torch.equal(x["topi"], y["topi"])


# ---------------------------------------------------------------- 13: initialization matched within a seed
@pytest.mark.parametrize("seed", [97, 0])
def test_candidate_and_baseline_backbone_initialization_match(seed):
    base = json.loads((ROOT / "configs/rep-v1/seed0/condition_A.json").read_text())
    base["init"] = {"type": "random", "seed": seed}
    base["model"].update(d_model=32, n_head=4, dense_ffn=48, expert_ffn=16, shared_ffn=24, vocab_size=97)
    ref = None
    for spec in [{"routing": sel.TOKEN}] + sel.GEN1:
        c = copy.deepcopy(base)
        c["routing"] = spec["routing"]
        m = build_sr_model(c, torch.device("cpu"))
        st = srs_run.param_sha(m.backbone_state())
        ref = ref or st
        assert st == ref, spec.get("id")
        assert m.extra_parameter_count() <= 0.005 * sum(p.numel() for p in m.parameters()) or m.cfg.d_model == 32


def test_full_size_extra_parameters_below_half_percent():
    base = json.loads((ROOT / "configs/rep-v1/seed0/condition_A.json").read_text())
    for spec in sel.GEN1 + [{"id": "x", "routing": dict(sel.GEN1[0]["routing"], summary="attn", state="gru", seg="lbd",
                                                          seg_len=64)}]:
        c = copy.deepcopy(base)
        c["routing"] = spec["routing"]
        from csr.model import ModelConfig
        torch.manual_seed(0)
        m = SeqRouteModel(ModelConfig.from_dict(c["model"]), SRRouting.from_cfg(c["routing"]))
        total = sum(p.numel() for p in m.parameters())
        assert m.extra_parameter_count() / total < 0.005, (spec["id"], m.extra_parameter_count(), total)


# ---------------------------------------------------------------- 15: finite losses and gradients
@pytest.mark.parametrize("name", SEQ + ["TOKEN"])
def test_gradients_and_losses_finite_and_reach_router(name):
    m = model_for(name)
    m.train()
    b = batch([[rids(400, 81), rids(200, 82)], [rids(600, 83)]], 620)
    o = m(b)
    lm = torch.nn.functional.cross_entropy(o.logits.view(-1, 97), b["targets"].view(-1), ignore_index=-100)
    loss = lm + 0.01 * o.aux_loss + 0.001 * o.z_loss + m.sr.bd_coeff * o.bd_loss
    loss.backward()
    assert torch.isfinite(loss)
    for n, p in m.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), n
    assert all(blk.ffn.router.weight.grad is not None and blk.ffn.router.weight.grad.abs().sum() > 0
               for blk in m.moe_blocks())
    if name != "TOKEN":
        assert m.bos.grad.abs().sum() > 0
    if m.sr.seg == "lbd":
        assert m.bd_w.grad.abs().sum() > 0 and float(o.bd_loss) > 0


# ---------------------------------------------------------------- segmentation rules
def brute_force_starts(cand_row, L, min_len, max_len, opening):
    starts, s, j = [0], 0, 0
    for t in range(L):
        sl = t - s + 1
        b = (sl == opening) if (opening and j == 0) else (sl >= max_len or (sl >= min_len and cand_row[t]))
        if b and t + 1 < L:
            starts.append(t + 1)
            s, j = t + 1, j + 1
    return starts


@pytest.mark.parametrize("opening", [0, 16])
def test_dynamic_scan_equals_token_by_token_rule(opening):
    rng = np.random.default_rng(0)
    for trial in range(30):
        Ls = [int(x) for x in rng.integers(1, 700, size=3)]
        T = sum(Ls) + 5
        cand = rng.random((1, T)) < rng.choice([0.01, 0.05, 0.3])
        pos = np.zeros((1, T), np.int64)
        npd = np.zeros((1, T), bool)
        t = 0
        for L in Ls:
            pos[0, t:t + L] = np.arange(L)
            npd[0, t:t + L] = True
            t += L
        st = scan_dynamic_starts(cand, pos, npd, 16, 256, opening)
        t = 0
        for L in Ls:
            ref = brute_force_starts(cand[0, t:t + L], L, 16, 256, opening)
            got = (np.flatnonzero(st[0, t:t + L])).tolist()
            assert got == ref
            seglens = np.diff(ref + [L])
            assert (seglens[:-1] >= 16).all() and (seglens <= 256).all()
            t += L


def test_fixed_segments_respect_documents_and_lengths():
    b = batch([[rids(300, 1), rids(70, 2)], [rids(1000, 3)]], 1024)
    for L in (16, 32, 64, 128, 256):
        st = fixed_starts_mask(b["positions"], b["doc_ids"] >= 0, L)
        si = seg_index(st, b["positions"], b["doc_ids"] >= 0)
        assert int(si.count.max()) <= L
        assert (si.count[si.closed] == L).all()
        assert bool(st[0, 300])  # document start always starts a segment


def test_dynamic_segment_lengths_within_limits(models):
    for name in ("CP32", "CP64-L0", "LBD64", "CP32-OPEN16-GRU-ATTN"):
        o = fwd(models[name], batch([[rids(1000, 4)], [rids(900, 5)]], 1000))
        for r in o.routing:
            c = r["seg_count"][r["seg_closed"]]
            assert (c >= 16).all() and (c <= 256).all()
            if "OPEN16" in name:
                assert (r["seg_count"][(r["seg_j"] == 0) & r["seg_closed"]] == 16).all()


def test_tau_controller_moves_toward_target():
    m = model_for("CP32")
    m.train()
    b = batch([[rids(1000, 6)], [rids(1000, 7)]], 1000)
    for direction in ("too_short", "too_long"):
        m.tau_stats_reset()
        tau0 = m.tau.clone()
        m._tau_stats = [[10, 10 * (8 if direction == "too_short" else 200)] for _ in m._tau_stats]
        m.tau_update()
        if direction == "too_short":
            assert (m.tau > tau0).all()
        else:
            assert (m.tau < tau0).all()
    with torch.no_grad():
        m.eval()
        m(b)  # eval forward never updates tau
    t1 = m.tau.clone()
    m(b)
    assert torch.equal(t1, m.tau)


def test_window_score_is_causal_and_document_clipped():
    torch.manual_seed(0)
    h = torch.randn(1, 200, 8, dtype=torch.float64)
    pos = torch.cat([torch.arange(120), torch.arange(80)]).unsqueeze(0)
    npd = torch.ones(1, 200, dtype=torch.bool)
    s1, v1 = window_change_score(h, pos, npd, 8, 32, 8)
    h2 = h.clone()
    h2[0, 150:] = torch.randn(50, 8, dtype=torch.float64)
    s2, _ = window_change_score(h2, pos, npd, 8, 32, 8)
    assert torch.equal(s1[0, :150], s2[0, :150])
    h3 = h.clone()
    h3[0, :120] = torch.randn(120, 8, dtype=torch.float64)  # previous document
    s3, _ = window_change_score(h3, pos, npd, 8, 32, 8)
    assert torch.allclose(s1[0, 120:], s3[0, 120:], atol=1e-10, rtol=0)
    assert not v1[0, 120:120 + 15].any() and v1[0, 120 + 15]


# ---------------------------------------------------------------- 12: evaluation isolation
def test_partition_deterministic_and_disjoint():
    ids = [f"doc{i}" for i in range(5000)]
    a1, b1 = srs_run.calib_partition(ids)
    a2, b2 = srs_run.calib_partition(list(ids))
    assert a1 == a2 and b1 == b2 and not set(a1) & set(b1) and len(a1) + len(b1) == 5000
    assert 0.67 < len(a1) / 5000 < 0.73


def test_split_locks(tmp_path, monkeypatch):
    monkeypatch.setattr(srs_run, "SEARCH_FROZEN", tmp_path / "search_frozen.json")
    monkeypatch.setattr(srs_run, "FINAL_FROZEN", tmp_path / "final_frozen.json")
    monkeypatch.setattr(srs_run, "EVAL_REGISTRY", tmp_path / "reg")
    assert srs_run.check_eval_access("calib_search")
    with pytest.raises(srs_run.SplitLocked):
        srs_run.check_eval_access("calib_selection", "abc")
    with pytest.raises(srs_run.SplitLocked):
        srs_run.check_eval_access("val", "abc")
    with pytest.raises(srs_run.SplitLocked):
        srs_run.check_eval_access("test", "abc")
    with pytest.raises(srs_run.SplitLocked):
        srs_run.load_raw_split(ROOT / "artifacts/data/fwe-rep-v1", "test")
    (tmp_path / "search_frozen.json").write_text(json.dumps({"config_sha256s": ["abc"]}))
    assert srs_run.check_eval_access("calib_selection", "abc")
    with pytest.raises(srs_run.SplitLocked):
        srs_run.check_eval_access("calib_selection", "zzz")
    with pytest.raises(srs_run.SplitLocked):
        srs_run.check_eval_access("val", "abc")
    (tmp_path / "final_frozen.json").write_text(json.dumps({"config_sha256s": ["f1"]}))
    assert srs_run.check_eval_access("val", "f1")
    p = srs_run.claim_eval("val", "ck1")
    p.write_text(json.dumps({"status": "COMPLETE", "result": "x"}))
    with pytest.raises(srs_run.SplitLocked):
        srs_run.claim_eval("val", "ck1")


def test_training_and_selection_code_never_reference_val_or_selection():
    src = (ROOT / "srs_train.py").read_text()
    assert '"val"' not in src and "calib_selection" not in src and '"test"' not in src
    sel_src = (ROOT / "src/csr/srs_select.py").read_text()
    assert "load_raw_split" not in sel_src and "eval_source" not in sel_src
    drv = (ROOT / "srs_driver.py").read_text()
    # the driver opens CALIB-SELECTION only after writing search_frozen.json, and val only after final_frozen.json
    assert drv.index("write_once(SEARCH_FROZEN") < drv.index('eval_item("calib_selection"')
    assert drv.index("write_once(FINAL_FROZEN") < drv.index('eval_item("val"')
    assert drv.index('gen3_finalists') < drv.index('eval_item("calib_selection"')


# ---------------------------------------------------------------- selection rules
def fake_results(ces, collapsed=()):
    return {k: {"complete": True, "ce": v, "collapsed": k in collapsed, "extra_param_fraction": 0.001, "gpu_s": 70}
            for k, v in ces.items()}


def test_gen2_rule_deterministic_balanced_and_bounded():
    ces = {s["id"]: 6.5 + 0.01 * i for i, s in enumerate(sel.GEN1)}
    ces["G1-CP64"] = 6.40
    ces["G1-F32-ATTN"] = 6.45
    r = fake_results(ces)
    g2 = sel.gen2_candidates(r)
    assert g2 == sel.gen2_candidates(copy.deepcopy(r))
    assert len(g2) == 8 and sum(d["family"] == "fixed" for d in g2) == 4
    keys = [sel.routing_key(d["routing"]) for d in g2]
    assert len(set(keys)) == 8
    fixed = [d for d in g2 if d["family"] == "fixed"]
    assert fixed[0]["parent"] == "G1-F32-ATTN" and fixed[0]["same_as"] == "G1-F32-ATTN"
    dyn = [d for d in g2 if d["family"] == "dynamic"]
    assert dyn[0]["parent"] == "G1-CP64" and dyn[2]["routing"]["opening"] == 16
    assert dyn[3]["routing"]["summary"] == "attn"  # best F32 summary probe
    uniq = {sel.routing_key(s["routing"]) for s in sel.GEN1 + g2}
    assert len(uniq) <= sel.MAX_UNIQUE


def test_gen2_rule_family_balance_and_invalid():
    ces = {s["id"]: 6.5 + 0.01 * i for i, s in enumerate(sel.GEN1)}
    dyn_ids = [s["id"] for s in sel.GEN1 if s["family"] == "dynamic"]
    r = fake_results(ces, collapsed=set(dyn_ids[1:]))
    g2 = sel.gen2_candidates(r)
    assert len(g2) <= 8 and all(sel.is_valid(r[d["parent"]]) for d in g2)
    assert sum(d["family"] == "fixed" for d in g2) >= 4


def test_gen3_promote_budget_and_criteria():
    specs = [{"id": f"G2-F{i}", "family": "fixed", "routing": {}} for i in range(1, 5)] + \
            [{"id": f"G2-D{i}", "family": "dynamic", "routing": {}} for i in range(1, 5)]
    r = fake_results({s["id"]: 6.0 + 0.01 * i for i, s in enumerate(specs)}, collapsed={"G2-F1"})
    fin = sel.gen3_finalists(r, specs)
    assert [s["id"] for s in fin] == ["G2-F2", "G2-F3", "G2-D1", "G2-D2"]
    tok = {"complete": True, "ce": 4.0, "collapsed": False, "extra_param_fraction": 0, "gpu_s": 1}
    pr = sel.promote({"TOKEN": tok, "fixed": dict(tok, ce=4.05), "dynamic": dict(tok, ce=4.2)})
    assert pr["families"]["fixed"]["promoted"] and not pr["families"]["dynamic"]["promoted"]
    b, rows = sel.choose_budget(3600, {"a": 1e-5, "b": 1e-5, "c": 1e-5}, overhead_s=1000)
    assert b == 100_000_000
    b, _ = sel.choose_budget(65000, {"a": 1e-5, "b": 1e-5, "c": 1e-5})
    assert b == 50_000_000
    boot = {"ci95": [-0.05, -0.03]}
    c = sel.criteria([-0.03, -0.025, -0.04], boot, False, 0.01)
    assert c["substantially_better"] and c["accuracy_comparable"]
    c = sel.criteria([-0.03, 0.001, -0.04], boot, False, 0.01)
    assert not c["substantially_better"]
    c = sel.criteria([0.005, 0.012, 0.0], {"ci95": [0.0, 0.009]}, False, 0.0)
    assert not c["accuracy_comparable"]


def test_bootstrap_deterministic_and_seed_averaged():
    rng = np.random.default_rng(0)
    n = {str(i): int(rng.integers(10, 100)) for i in range(300)}
    mk = lambda off: ({k: v * (4.0 + off + rng.normal(0, 0.1)) for k, v in n.items()}, n)  # noqa: E731
    tok = [mk(0) for _ in range(3)]
    seq = [mk(0.05) for _ in range(3)]
    a = sel.paired_bootstrap_multi(tok, seq)
    b = sel.paired_bootstrap_multi(tok, seq)
    assert a == b and a["n_boot"] == 2000 and a["ci95"][0] > 0


# ---------------------------------------------------------------- 16: resume reproducibility (real data, tiny model)
def _tiny_train_cfg(tmpdir):
    from csr.srs_run import prereg_sha
    c = json.loads((ROOT / "configs/rep-v1/seed0/condition_A.json").read_text())
    c.pop("semantic", None)
    c.update(experiment="csr-sequence-routing-search-v1", prereg_sha256=prereg_sha(), candidate="TEST-CP32",
             family="dynamic", run_name="cputest_resume", stage="test", description="test")
    c["init"] = {"type": "random", "seed": 5}
    c["model"].update(d_model=32, n_head=4, dense_ffn=48, expert_ffn=16, shared_ffn=24, n_experts=8)
    c["routing"] = SRRouting(seg="cp", target_len=32).to_dict()
    c["train"].update(token_budget=30_000, micro_batch_rows=2, accum=2, log_every=1, eval_every_tokens=10**9,
                      save_checkpoint=True, state_every_seconds=1e-9, warmup_steps=2, seed=5)
    c["eval"] = {"rows_per_batch": 4, "max_rows": 4, "final_calib_search": False}
    p = tmpdir / "cfg.json"
    p.write_text(json.dumps(c))
    return p


@pytest.mark.skipif(not (ROOT / "docs/SEQUENCE_ROUTING_SEARCH_V1_PREREG.md").exists(), reason="prereg not written")
def test_interrupted_run_resumes_reproducibly():
    tmp = ROOT / ".tmp" / "srsv1_pytest" / f"resume_{os.getpid()}"
    tmp.mkdir(parents=True)
    cfgp = _tiny_train_cfg(tmp)
    env = dict(os.environ, MOE_RESEARCH_APPROVE="cpu-model", MOE_RUNS_DIR=str(tmp / "runs"), CUDA_VISIBLE_DEVICES="")
    (tmp / "runs").mkdir()

    def run(*extra):
        out = subprocess.run([sys.executable, "srs_train.py", "--config", str(cfgp), "--execute", *extra], cwd=ROOT,
                             env=env, capture_output=True, text=True, timeout=900)
        rd = [ln.split(" ", 1)[1] for ln in out.stdout.splitlines() if ln.startswith("RUN_DIR ")]
        assert rd, out.stdout[-3000:] + out.stderr[-3000:]
        return ROOT / rd[-1], out

    full, o1 = run()
    assert json.loads((full / "status.json").read_text())["status"] == "COMPLETE", o1.stdout[-2000:]
    part, _ = run("--max-steps", "2")
    assert json.loads((part / "status.json").read_text())["status"] == "INTERRUPTED"
    res, o3 = run("--resume", str(part))
    assert json.loads((res / "status.json").read_text())["status"] == "COMPLETE", o3.stdout[-2000:]
    a = torch.load(full / "model.pt", weights_only=False)["model"]
    b = torch.load(res / "model.pt", weights_only=False)["model"]
    assert a.keys() == b.keys()
    for k in a:
        assert torch.equal(a[k], b[k]), k
    sa = json.loads((full / "summary.json").read_text())
    sb = json.loads((res / "summary.json").read_text())
    assert sa["valid_train_tokens"] == sb["valid_train_tokens"]
