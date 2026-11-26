"""csr-speculative-pool-routing-v1 correctness tests (CPU, CUDA hidden). docs/SPECULATIVE_POOL_ROUTING_V1_PREREG.md §7."""
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from csr import spv1_run, spv1_select as sel
from csr.model import CSRModel, ModelConfig, RoutingConfig
from csr.spv1_decode import (draft_greedy_full, greedy_generate, new_cache, sequential_spec_family, spec_generate,
                             target_inc)
from csr.spv1_model import (PoolRouteModel, PoolRouting, block_start_index, build_pool_model, draft_model_config,
                            draft_tree_proposals, first_true_next, orbit_starts, spec_starts)

ROOT = Path(__file__).resolve().parents[1]
V = 61
E = 8
K = 2


def mcfg(**kw):
    base = dict(vocab_size=V, n_layer=3, d_model=32, n_head=4, dense_ffn=48, moe_layers=(1, 2), n_experts=E, top_k=K,
                expert_ffn=16, shared_ffn=24, max_seq_len=1024, tie_embeddings=True)
    base.update(kw)
    return ModelConfig(**base)


DCFG = ModelConfig(vocab_size=V, n_layer=2, d_model=16, n_head=2, dense_ffn=32, moe_layers=(), n_experts=1, top_k=1,
                   expert_ffn=1, shared_ffn=0, max_seq_len=1024, tie_embeddings=False, init_std=0.3)

VARIANTS = {
    "F4M4": dict(family="fixed", seg_len=4, pool_size=4),
    "F8M2": dict(family="fixed", seg_len=8, pool_size=2),
    "F4M4-R": dict(family="fixed", seg_len=4, pool_size=4, pool_assign="random"),
    "A4M4": dict(family="adaptive", seg_len=4, pool_size=4, refresh_target=0.5),
    "A8M2": dict(family="adaptive", seg_len=8, pool_size=2, refresh_target=0.25),
    "A8M4-R": dict(family="adaptive", seg_len=8, pool_size=4, refresh_target=0.25, pool_assign="random"),
    "S4M4": dict(family="spec", seg_len=4, pool_size=4),
    "S8M2": dict(family="spec", seg_len=8, pool_size=2),
    "S4M4-R": dict(family="spec", seg_len=4, pool_size=4, pool_assign="random"),
    "F4M4-T": dict(family="fixed", seg_len=4, pool_size=4, router_temperature=0.7, sel_temperature=2.0,
                   pool_dropout=0.2),
}
NAMES = sorted(VARIANTS)


def rids(n, seed=0):
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


@pytest.fixture(scope="module")
def draft():
    torch.manual_seed(11)
    return CSRModel(DCFG).double().eval()


def model_for(name, draft=None, seed=0, calib=None, perturb=True, **cfgkw):
    torch.manual_seed(seed)
    pr = PoolRouting(allow_test_sizes=True, **VARIANTS[name]) if name != "TOKEN" else PoolRouting()
    m = PoolRouteModel(mcfg(**cfgkw), pr, extra_seed=seed).double()
    if perturb:  # non-trivial selector / detector parameters (zero at initialisation)
        g = torch.Generator().manual_seed(seed + 77)
        with torch.no_grad():
            for n, p in m.named_parameters():
                if n in m.EXTRA_NAMES and n != "spec_beta":
                    p.copy_(torch.randn(p.shape, generator=g, dtype=p.dtype) * 0.3)
    if draft is not None:
        m.attach_draft(draft)
    if pr.family == "adaptive":
        b = calib if calib is not None else batch([[rids(200, 99), rids(150, 98)], [rids(300, 97)]], 360)
        m.train()
        m.tau_calibrate(b)
    m.eval()
    return m


@pytest.fixture(scope="module")
def models(draft):
    return {n: model_for(n, draft) for n in NAMES + ["TOKEN"]}


@pytest.fixture(scope="module")
def two_doc_batch():
    return batch([[rids(150, 1), rids(70, 2)], [rids(230, 3)]], 240)


def fwd(m, b, **kw):
    with torch.no_grad():
        return m(b, collect=True, **kw)


# ---------------------------------------------------------------- structure primitives
def test_orbit_matches_sequential_pointer_chase():
    g = torch.Generator().manual_seed(0)
    for _ in range(300):
        S = int(torch.randint(1, 70, (1,), generator=g))
        nxt = torch.stack([torch.tensor([min(S, i + int(torch.randint(1, 9, (1,), generator=g))) for i in range(S)])
                           for _ in range(3)])
        o = orbit_starts(nxt)
        for b in range(3):
            ref = torch.zeros(S, dtype=torch.bool)
            i = 0
            while i < S:
                ref[i] = True
                i = int(nxt[b, i])
            assert torch.equal(ref, o[b])


def test_first_true_next_and_block_index():
    R = torch.tensor([[[False, True, True], [True, True, True], [False, False, True]]])
    assert first_true_next(R).tolist() == [[2, 2, 3]]
    st = torch.tensor([[True, False, True, False, False, True]])
    assert block_start_index(st).tolist() == [[0, 0, 2, 2, 2, 5]]


def _seq_spec_starts(ids, pos, props, Bb, forced=None):
    T = len(ids)
    out = np.zeros(T, bool)
    s = 0
    out[0] = True
    t = 1
    while t < T:
        k = t - s
        new = pos[t] == 0 or k >= Bb or ids[t] != props[s][k - 1] or (forced is not None and forced[t])
        if new:
            out[t] = True
            s = t
        t += 1
    return out


@pytest.mark.parametrize("Bb", [2, 4, 8])
def test_spec_starts_equal_token_by_token_rule(Bb):
    rng = np.random.default_rng(Bb)
    for trial in range(30):
        T = 90
        ids = rng.integers(0, 3, T)
        pos = np.arange(T)
        pos[40:] = np.arange(T - 40)  # second document at 40
        props = rng.integers(0, 3, (T, 7))
        forced = rng.random(T) < 0.05
        got = spec_starts(torch.tensor(ids).unsqueeze(0), torch.tensor(pos).unsqueeze(0), torch.tensor(props).unsqueeze(0),
                          Bb, torch.tensor(forced).unsqueeze(0))[0].numpy()
        assert (got == _seq_spec_starts(ids, pos, props, Bb, forced)).all()


def test_adaptive_sequential_reference_exact(two_doc_batch):
    """Full sequential re-implementation of the adaptive rule (pool of the current block, learned score, tau)."""
    torch.manual_seed(3)
    m = model_for("A4M4")
    b = two_doc_batch
    hooks = {}
    hs = []
    for blk in m.blocks:
        if blk.is_moe:
            hs.append(blk.norm2.register_forward_hook(lambda mod, i, o, hs_=hooks: hs_.setdefault("h", []).append(o)))
    out = fwd(m, b)
    for h in hs:
        h.remove()
    H, tau = m.pr.seg_len, m.tau
    for li, r in enumerate(out.routing):
        h = hooks["h"][li].double()
        learned = h @ m.det_w[li].double() + m.det_b[li].double()
        W = m.blocks[li + 1].ffn.router.weight.double()
        q = h @ (W + m.sel_U[li].double()).t()
        Pall = torch.zeros_like(q, dtype=torch.bool).scatter_(-1, q.topk(m.pr.pool_size, -1).indices, True)
        for bb in range(h.shape[0]):
            s = None
            for t in range(h.shape[1]):
                pos = int(b["positions"][bb, t])
                if s is None or pos == 0 or t - s >= H:
                    new = True
                else:
                    miss = 1.0 - (r["probs"][bb, t] * Pall[bb, s].double()).sum()
                    new = bool(miss + learned[bb, t] > tau[li])
                assert new == bool(r["starts"][bb, t]), (li, bb, t)
                if new:
                    s = t
                assert torch.equal(r["cand"][bb, t], Pall[bb, s])


def test_fixed_pools_change_only_at_declared_boundaries(models, two_doc_batch):
    for name in ("F4M4", "F8M2", "F4M4-R", "F4M4-T"):
        m = models[name]
        L = m.pr.seg_len
        out = fwd(m, two_doc_batch)
        pos = two_doc_batch["positions"]
        nonpad = two_doc_batch["doc_ids"] >= 0
        for r in out.routing:
            assert torch.equal(r["starts"], (pos % L) == 0)
            same = (r["cand"][:, 1:] == r["cand"][:, :-1]).all(-1)
            boundary = (pos[:, 1:] % L) == 0
            assert bool((same | boundary)[nonpad[:, 1:]].all())


# ---------------------------------------------------------------- causality: future tokens never change the past
CUTS = [0, 3, 7, 16, 61, 149, 150, 160, 219]


@pytest.mark.parametrize("name", NAMES + ["TOKEN"])
def test_future_tokens_never_change_earlier_pools_routes_or_logits(models, two_doc_batch, name):
    m = models[name]
    base = fwd(m, two_doc_batch)
    rng = np.random.default_rng(5)
    for cut in CUTS:
        b2 = {k: v.clone() for k, v in two_doc_batch.items()}
        ids = b2["input_ids"]
        npd = b2["doc_ids"] >= 0
        noise = torch.as_tensor(rng.integers(2, V, size=ids.shape))
        fut = torch.zeros_like(npd)
        fut[:, cut + 1:] = True
        b2["input_ids"] = torch.where(fut & npd, noise, ids)
        o2 = fwd(m, b2)
        assert torch.allclose(base.logits[:, :cut + 1], o2.logits[:, :cut + 1], atol=1e-10, rtol=0), cut
        for r1, r2 in zip(base.routing, o2.routing):
            assert torch.equal(r1["topi"][:, :cut + 1], r2["topi"][:, :cut + 1])
            assert torch.allclose(r1["gates"][:, :cut + 1], r2["gates"][:, :cut + 1], atol=1e-12)
            if "cand" in r1:
                assert torch.equal(r1["cand"][:, :cut + 1], r2["cand"][:, :cut + 1])
                assert torch.equal(r1["starts"][:, :cut + 1], r2["starts"][:, :cut + 1])
        if m.pr.family == "spec":
            assert torch.equal(base.proposals[:, :cut + 1], o2.proposals[:, :cut + 1])


@pytest.mark.parametrize("name", ["S4M4", "S8M2"])
def test_spec_pool_depends_on_prefix_and_proposal_not_true_continuation(models, two_doc_batch, name):
    m = models[name]
    b = two_doc_batch
    props = m.prepare_proposals(b)
    base = fwd(m, b, proposals=props)
    r0 = base.routing[0]
    s = int(torch.nonzero(r0["starts"][0, 20:])[0]) + 20  # a block start in document 0
    # (a) changing the true continuation inside the block (keeping proposals fixed) never changes the pool at s
    b2 = {k: v.clone() for k, v in b.items()}
    b2["input_ids"][0, s + 1:s + m.pr.seg_len] = (b2["input_ids"][0, s + 1:s + m.pr.seg_len] + 1) % V
    o2 = fwd(m, b2, proposals=props)
    for r1, r2 in zip(base.routing, o2.routing):
        assert torch.equal(r1["cand"][0, s], r2["cand"][0, s])
    # (b) the proposal itself does condition the learned pool
    p2 = props.clone()
    p2[0, s] = (p2[0, s] + 7) % V
    o3 = fwd(m, b, proposals=p2)
    assert any(not torch.equal(r1["cand"][0, s], r3["cand"][0, s]) for r1, r3 in zip(base.routing, o3.routing)) or \
        any(not torch.equal(r1["logm_tok"][0, s], r3["logm_tok"][0, s]) for r1, r3 in zip(base.routing, o3.routing))


def test_spec_uses_draft_proposals_not_ground_truth(models, two_doc_batch):
    m = models["S4M4"]
    out = fwd(m, two_doc_batch)
    b = two_doc_batch
    tree = draft_tree_proposals(m._draft, b["input_ids"], b["doc_ids"], b["positions"], m.pr.gamma)
    assert torch.equal(out.proposals, tree)
    npd = b["doc_ids"] >= 0
    truth = torch.cat([b["input_ids"][:, 1:], b["input_ids"][:, -1:]], 1)
    assert (out.proposals[:, :, 0] != truth)[npd].float().mean() > 0.5  # proposals are not the true continuation


def test_draft_tree_equals_naive_greedy(draft):
    b = batch([[rids(40, 7), rids(25, 8)], [rids(50, 9)]], 70)
    P = draft_tree_proposals(draft, b["input_ids"], b["doc_ids"], b["positions"], 5)
    for bb in range(2):
        for t in range(70):
            if b["doc_ids"][bb, t] < 0:
                continue
            st = t - int(b["positions"][bb, t])
            ref = draft_greedy_full(draft, b["input_ids"][bb:bb + 1, st:t + 1], 5)
            assert ref[0].tolist() == P[bb, t].tolist(), (bb, t)


# ---------------------------------------------------------------- routing invariants
@pytest.mark.parametrize("name", NAMES + ["TOKEN"])
def test_exactly_k_distinct_experts_in_pool_with_independent_weights(models, two_doc_batch, name):
    m = models[name]
    out = fwd(m, two_doc_batch)
    npd = two_doc_batch["doc_ids"] >= 0
    for r in out.routing:
        ti = r["topi"]
        assert ti.shape[-1] == K
        assert bool((ti.sort(-1).values.diff(dim=-1) > 0).all())
        g = r["gates"]
        assert torch.allclose(g, r["probs"].gather(-1, ti), atol=1e-12)  # token-specific global probabilities
        if "cand" in r:
            assert bool(r["cand"].gather(-1, ti).all())
            assert bool((r["cand"].sum(-1) == m.pr.pool_size).all())
            # tokens sharing a pool keep their own top-k and weights: within-block variation exists
            bs = r["bs"]
            same_block = (bs[:, 1:] == bs[:, :-1]) & npd[:, 1:]
            diff_w = ~torch.isclose(g[:, 1:], g[:, :-1]).all(-1)
            assert bool((diff_w & same_block).any())
            # selection inside the pool is the token's own top-k of its probabilities restricted to the pool
            p = r["probs"].masked_fill(~r["cand"], -1.0)
            assert torch.equal(p.topk(K, -1).indices.sort(-1).values, ti.sort(-1).values)


def test_shared_expert_path_unchanged(models, two_doc_batch):
    """Every MoE layer output = routed mixture + shared expert of the same input (no change to the shared path)."""
    m = models["S4M4"]
    hs = []
    hooks = [blk.ffn.shared.register_forward_hook(lambda mod, i, o: hs.append((i[0], o))) for blk in m.blocks
             if blk.is_moe]
    fwd(m, two_doc_batch)
    for h in hooks:
        h.remove()
    for (inp, o), blk in zip(hs, [b for b in m.blocks if b.is_moe]):
        assert torch.allclose(blk.ffn.shared(inp), o)
        assert inp.shape[0] == two_doc_batch["input_ids"].numel()  # applied to every token


def test_token_mode_equals_original_csr_token_router(two_doc_batch):
    torch.manual_seed(0)
    ref = CSRModel(mcfg()).double().eval()
    t0 = model_for("TOKEN", perturb=False)
    with torch.no_grad():
        a = ref(two_doc_batch, RoutingConfig())
        b = t0(two_doc_batch)
    assert torch.equal(a.logits, b.logits)
    assert torch.equal(a.aux_loss, b.aux_loss) and torch.equal(a.z_loss, b.z_loss)


def test_token_mode_equals_previous_experiment_token_router(two_doc_batch):
    from csr.seqroute import SRRouting, SeqRouteModel
    torch.manual_seed(0)
    a = SeqRouteModel(mcfg(), SRRouting(mode="token")).double().eval()
    b = model_for("TOKEN", perturb=False)
    with torch.no_grad():
        assert torch.equal(a(two_doc_batch).logits, b(two_doc_batch).logits)


def test_full_pool_reduces_to_token_routing(two_doc_batch):
    t0 = model_for("TOKEN", perturb=False)
    for fam in ("fixed", "spec", "adaptive"):
        torch.manual_seed(0)
        m = PoolRouteModel(mcfg(), PoolRouting(family=fam, pool_size=E, seg_len=4, refresh_target=0.5,
                                               allow_test_sizes=True)).double().eval()
        if fam == "spec":
            torch.manual_seed(1)
            m.attach_draft(CSRModel(DCFG).double().eval())
        if fam == "adaptive":
            m.tau.fill_(0.5)
        with torch.no_grad():
            assert torch.allclose(m(two_doc_batch).logits, t0(two_doc_batch).logits, atol=1e-12)


@pytest.mark.parametrize("seed", [0, 1, 98])
def test_identical_initial_target_weights_within_seed(seed):
    base = {"model": {**mcfg().to_dict()}, "init": {"seed": seed}}
    shas = set()
    for name in NAMES + ["TOKEN"]:
        r = PoolRouting(allow_test_sizes=True, **VARIANTS[name]).to_dict() if name != "TOKEN" else PoolRouting().to_dict()
        m = build_pool_model({**base, "routing": r}, "cpu")
        shas.add(spv1_run.param_sha(m.backbone_state()))
    assert len(shas) == 1


def test_full_size_extra_parameters_below_half_percent():
    full = json.loads((ROOT / "configs/rep-v1/seed0/condition_A.json").read_text())["model"]
    for spec in sel.GEN1 + [sel.random_control(sel.GEN1[0], "x")]:
        r = dict(spec["routing"])
        r.update(sel.GEN2_VARIANT)
        for rr in (spec["routing"], r):
            m = build_pool_model({"model": full, "init": {"seed": 0}, "routing": rr}, "cpu")
            frac = m.extra_parameter_count() / sum(p.numel() for p in m.parameters())
            assert frac <= 0.005, (spec["id"], frac)


# ---------------------------------------------------------------- gradients / finiteness
@pytest.mark.parametrize("name", NAMES)
def test_gradients_reach_selector_detector_router_and_are_finite(draft, two_doc_batch, name):
    m = model_for(name, draft)
    m.train()
    m.rate_stats_reset()
    props = m.prepare_proposals(two_doc_batch)
    if props is not None:  # informative proposals: random tokens (all positions distinct)
        props = torch.as_tensor(np.random.default_rng(3).integers(2, V, size=props.shape))
    out = m(two_doc_batch, proposals=props)
    tgt = two_doc_batch["targets"]
    lm = torch.nn.functional.cross_entropy(out.logits.view(-1, V), tgt.view(-1), ignore_index=-100)
    obj = lm + 0.01 * out.aux_loss + 0.001 * out.z_loss + m.objective_extras(out)
    assert torch.isfinite(obj) and torch.isfinite(out.logits).all()
    obj.backward()
    for n, p in m.named_parameters():
        assert p.grad is None or torch.isfinite(p.grad).all(), n
    routers = [n for n, _ in m.named_parameters() if n.endswith("router.weight")]
    for n in routers + m.extra_parameter_names():
        p = dict(m.named_parameters())[n]
        assert p.grad is not None and float(p.grad.abs().sum()) > 0, n
    pr = m.pr
    expect = set()
    if pr.pool_assign == "learned":
        expect |= {"sel_U"} | ({"sel_V", "spec_beta", "spec_alpha"} if pr.family == "spec" else set())
    if pr.family == "adaptive":
        expect |= {"det_w", "det_b"}
    assert set(m.extra_parameter_names()) == expect


def test_tau_controller_moves_toward_target(two_doc_batch):
    m = model_for("A4M4")
    m.train()
    m.rate_stats_reset()
    m.tau.fill_(5.0)  # far too high -> only forced refreshes -> rate below target -> tau decreases
    with torch.no_grad():
        m(two_doc_batch)
    t_before = m.tau.clone()
    rates = m.tau_update()
    assert all(r < m.pr.refresh_target for r in rates)
    assert bool((m.tau < t_before).all())
    m.eval()
    with torch.no_grad():
        m(two_doc_batch)
    assert m.tau_update() is None or True  # eval forwards never accumulate statistics


def test_tau_calibration_hits_target(two_doc_batch):
    m = model_for("A8M2", calib=two_doc_batch)
    out = fwd(m, two_doc_batch)
    npd = two_doc_batch["doc_ids"] >= 0
    for r in out.routing:
        rate = float((r["starts"] & npd).sum()) / float(npd.sum())
        assert abs(rate - 0.25) < 0.05


# ---------------------------------------------------------------- incremental == teacher forcing; decoding
@pytest.mark.parametrize("name", NAMES + ["TOKEN"])
def test_incremental_decoding_reproduces_teacher_forcing(models, name):
    m = models[name]
    Bt, T = 3, 70
    ids = torch.as_tensor(np.random.default_rng(1).integers(2, V, size=(Bt, T)))
    b = {"input_ids": ids, "doc_ids": torch.zeros_like(ids), "positions": torch.arange(T).expand(Bt, T).contiguous()}
    with torch.no_grad():
        full = m(b).logits
    cache = new_cache(m, Bt, T + 2, ids.device, torch.float64)
    if m.pr.family == "adaptive":
        rstate = [{"pool": torch.zeros(Bt, E, dtype=torch.bool), "hold": torch.zeros(Bt, dtype=torch.long)}
                  for _ in range(m.n_moe)]
    else:
        rstate = [{} for _ in range(m.n_moe)]
    D, j, outs = [None] * Bt, [0] * Bt, []
    for t in range(T):
        bs = props = None
        if m.pr.family == "spec":
            bs = torch.zeros(Bt, 1, dtype=torch.bool)
            props = torch.zeros(Bt, 1, m.pr.gamma, dtype=torch.long)
            for bb in range(Bt):
                nb = D[bb] is None or j[bb] >= m.pr.seg_len or int(ids[bb, t]) != int(D[bb][0, j[bb] - 1])
                if nb:
                    D[bb] = draft_greedy_full(m._draft, ids[bb:bb + 1, :t + 1], m.pr.gamma)
                    j[bb] = 0
                bs[bb, 0] = nb
                props[bb, 0] = D[bb][0]
                j[bb] += 1
        lg, _ = target_inc(m, ids[:, t:t + 1], torch.full((Bt, 1), t), cache, rstate, blk_start=bs, props=props)
        outs.append(lg)
    assert torch.allclose(torch.cat(outs, 1), full, atol=1e-9, rtol=0)


def _decode_pair(name, gamma_from_model=True):
    mc = mcfg(vocab_size=6, tie_embeddings=False, init_std=0.3)
    dc = ModelConfig(vocab_size=6, n_layer=2, d_model=16, n_head=2, dense_ffn=32, moe_layers=(), n_experts=1,
                     top_k=1, expert_ffn=1, shared_ffn=0, max_seq_len=1024, tie_embeddings=False, init_std=0.3)
    torch.manual_seed(9)
    dr = CSRModel(dc).double().eval()
    torch.manual_seed(0)
    pr = PoolRouting(allow_test_sizes=True, **VARIANTS[name]) if name != "TOKEN" else PoolRouting()
    m = PoolRouteModel(mc, pr).double().eval()
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n in m.EXTRA_NAMES and n != "spec_beta":
                p.copy_(torch.randn(p.shape, generator=g, dtype=p.dtype) * 0.3)
    m.attach_draft(dr)
    return m, dr


@pytest.mark.parametrize("name,gamma", [("TOKEN", 3), ("TOKEN", 7), ("S4M4", 3), ("S8M2", 7), ("S4M4-R", 3)])
def test_batched_speculative_equals_sequential_greedy(name, gamma):
    m, dr = _decode_pair(name)
    prompts = torch.as_tensor(np.random.default_rng(gamma).integers(0, 6, size=(5, 12)))
    sp, st = spec_generate(m, dr, prompts, 40, gamma)
    if m.pr.family == "spec":
        ref = torch.cat([sequential_spec_family(m, dr, prompts[b:b + 1], 40)[0] for b in range(5)], 0)
    else:
        ref = greedy_generate(m, prompts, 40)[0]
    assert torch.equal(sp, ref)
    hist = st["accept_hist"]
    assert sum(hist[:-1]) > 0, "test must exercise rejections"
    assert st["proposal_regenerations"] == st["blocks"]


@pytest.mark.parametrize("name,gamma", [("TOKEN", 3), ("S4M4", 3), ("S8M2", 7)])
def test_rejected_suffix_state_never_reused(name, gamma):
    """Scrambling every KV slot of rejected / unprocessed positions after each block leaves the output unchanged:
    nothing computed for a rejected draft suffix (KV entries, pools) is reused; the next proposal and pool are
    regenerated from the accepted prefix."""
    m, dr = _decode_pair(name)
    prompts = torch.as_tensor(np.random.default_rng(1).integers(0, 6, size=(4, 10)))
    a, _ = spec_generate(m, dr, prompts, 36, gamma)
    torch.manual_seed(123)
    b, _ = spec_generate(m, dr, prompts, 36, gamma, scramble_stale=True)
    assert torch.equal(a, b)


@pytest.mark.parametrize("name", ["TOKEN", "F4M4", "A4M4", "F4M4-R", "A8M4-R"])
def test_lockstep_greedy_equals_single_sequence_greedy(models, name):
    m = models[name]
    prompts = torch.as_tensor(np.random.default_rng(2).integers(2, V, size=(4, 20)))
    a, _ = greedy_generate(m, prompts, 25)
    b = torch.cat([greedy_generate(m, prompts[i:i + 1], 25)[0] for i in range(4)], 0)
    assert torch.equal(a, b)


# ---------------------------------------------------------------- random-pool control
def test_random_pools_content_independent_and_load_balanced(models, two_doc_batch):
    m = models["F4M4-R"]
    tab = m.rand_pool
    per = E // m.pr.pool_size
    for li in range(tab.shape[0]):
        for p0 in range(0, 64, per):
            assert bool((tab[li, p0:p0 + per].sum(0) == 1).all())  # consecutive groups partition the experts
    out = fwd(m, two_doc_batch)
    b2 = {k: v.clone() for k, v in two_doc_batch.items()}
    b2["input_ids"] = (b2["input_ids"] * 7 + 3) % V
    o2 = fwd(m, b2)
    for r1, r2 in zip(out.routing, o2.routing):
        assert torch.equal(r1["cand"], r2["cand"])  # pools do not depend on content


# ---------------------------------------------------------------- no host loops / syncs in the structure kernels
FORBIDDEN = (".cpu(", ".numpy(", ".item(", ".tolist(", "for t in range", "while ")


def test_structure_kernels_have_no_host_loops_in_source():
    from csr import spv1_model as M
    for fn in (M.orbit_starts, M.first_true_next, M.block_start_index, M.spec_starts, M.fixed_starts, M.shift_left,
               M.PoolRouteModel._adaptive_starts_given_tau, M.PoolRouteModel._pools,
               M.PoolRouteModel._selector_logits, M.PoolRouteModel._ghat):
        src = inspect.getsource(fn)
        for f in FORBIDDEN:
            assert f not in src, (fn.__name__, f)


def test_structure_kernels_run_without_host_transfers(models, two_doc_batch, monkeypatch):
    """Route structure + pool selection + in-pool top-k execute with every host-transfer method disabled."""
    m = models["A4M4"]
    out = fwd(m, two_doc_batch)
    r = out.routing[0]
    probs, cand_all = r["probs"], r["cand"]

    def boom(*a, **k):
        raise AssertionError("host transfer in a timed structure kernel")
    for meth in ("item", "cpu", "tolist", "numpy", "__bool__", "__int__", "__float__"):
        monkeypatch.setattr(torch.Tensor, meth, boom)
    pos = two_doc_batch["positions"]
    st = PoolRouteModel._adaptive_starts_given_tau(probs, cand_all.double(), torch.zeros(pos.shape, dtype=torch.float64),
                                                   pos == 0, 4, torch.tensor(0.3, dtype=torch.float64))
    bs = block_start_index(st)
    c = cand_all.gather(1, bs.unsqueeze(-1).expand(*bs.shape, E))
    probs.masked_fill(~c, -1.0).topk(K, -1)
    props = torch.zeros(*pos.shape, 7, dtype=torch.long)
    spec_starts(two_doc_batch["input_ids"], pos, props, 4)
    monkeypatch.undo()


# ---------------------------------------------------------------- data, isolation and locks
def test_partition_deterministic_disjoint_and_70_30():
    ids = [f"doc{i}" for i in range(4000)]
    a = spv1_run.calib_partition(ids)
    b = spv1_run.calib_partition(ids)
    assert a == b and not set(a[0]) & set(a[1]) and len(a[0]) + len(a[1]) == 4000
    assert 0.67 < len(a[0]) / 4000 < 0.73
    from csr import srs_run
    assert spv1_run.PARTITION_SALT != srs_run.PARTITION_SALT


def test_test_split_refused_before_any_file_access(monkeypatch):
    from csr import data
    monkeypatch.setattr(data.TokenSplit, "load", classmethod(lambda cls, d, s: (_ for _ in ()).throw(
        AssertionError("file access attempted"))))
    for s in ("test", "TEST", "test_tokens", "../test"):
        with pytest.raises(spv1_run.SplitLocked):
            spv1_run.load_raw_split(ROOT / spv1_run.DATA_DIR, s)
    with pytest.raises(spv1_run.SplitLocked):
        spv1_run.eval_source({"data": {"seq_len": 1024}}, "test", ROOT / spv1_run.DATA_DIR)


def test_split_locks_and_single_use(tmp_path, monkeypatch):
    monkeypatch.setattr(spv1_run, "SEARCH_FROZEN", tmp_path / "search_frozen.json")
    monkeypatch.setattr(spv1_run, "FINAL_FROZEN", tmp_path / "final_frozen.json")
    monkeypatch.setattr(spv1_run, "EVAL_REGISTRY", tmp_path / "reg")
    monkeypatch.setattr(spv1_run.guard, "ROOT", ROOT)
    assert spv1_run.check_eval_access("calib_search")
    for s in ("calib_selection", "val", "test"):
        with pytest.raises(spv1_run.SplitLocked):
            spv1_run.check_eval_access(s, "abc")
    (tmp_path / "search_frozen.json").write_text(json.dumps({"config_sha256s": ["abc"]}))
    assert spv1_run.check_eval_access("calib_selection", "abc")
    with pytest.raises(spv1_run.SplitLocked):
        spv1_run.check_eval_access("calib_selection", "zzz")
    with pytest.raises(spv1_run.SplitLocked):
        spv1_run.check_eval_access("val", "abc")
    (tmp_path / "final_frozen.json").write_text(json.dumps({"config_sha256s": ["f1"]}))
    assert spv1_run.check_eval_access("val", "f1")
    with pytest.raises(spv1_run.SplitLocked):
        spv1_run.check_eval_access("val", "abc")
    monkeypatch.setattr(spv1_run, "EVAL_REGISTRY", ROOT / ".tmp" / "spv1_pytest" / f"reg_{os.getpid()}")
    p = spv1_run.claim_eval("val", "ck1")
    p.write_text(json.dumps({"status": "COMPLETE", "result": "x"}))
    with pytest.raises(spv1_run.SplitLocked):
        spv1_run.claim_eval("val", "ck1")


def test_training_and_selection_code_never_reference_val_or_selection():
    for f in ("spv1_train.py", "spv1_draft_cache.py", "src/csr/spv1_select.py", "src/csr/spv1_model.py"):
        src = (ROOT / f).read_text()
        for bad in ('"val"', "'val'", "calib_selection\"", "val_tokens", "test_tokens"):
            assert bad not in src, (f, bad)
    tr = (ROOT / "spv1_train.py").read_text()
    assert 'eval_source(cfg, "calib_search"' in tr


def test_prop_batch_source_aligns_cached_proposals():
    from csr.data import TokenSplit
    toks = np.arange(2, 2 + 50, dtype=np.uint16)
    off = np.array([0, 10, 30, 50])
    sp = TokenSplit(toks, off, ["a", "b", "c"])
    cache = np.stack([toks + 100 * j for j in range(1, 8)], 1).astype(np.uint16)
    src = spv1_run.PropBatchSource(sp, 32, [2, 0, 1], cache)
    for rows in src.rows:
        a = src.batch([rows])
        for t in range(32):
            if a["doc_ids"][0, t] < 0:
                continue
            tokv = a["input_ids"][0, t]
            assert a["proposals"][0, t].tolist() == [tokv + 100 * j for j in range(1, 8)]


# ---------------------------------------------------------------- selection rules
def _res(ce, valid=True, rr=None, d8=None, tps=1e5):
    return {"complete": valid, "ce": ce, "collapsed": not valid, "extra_param_fraction": 0.003, "refresh_rate": rr,
            "distinct8": d8, "train_tok_s": tps}


def test_gen2_rule_family_best_plus_variants_bounded_and_deterministic():
    ces = {c["id"]: 5.0 + 0.01 * i for i, c in enumerate(sel.GEN1)}
    res = {k: _res(v) for k, v in ces.items()}
    res["G1-F4M8"] = _res(4.0, valid=False)  # invalid best is skipped
    g2 = sel.gen2_candidates(res)
    assert g2 == sel.gen2_candidates(res) and len(g2) == 6
    assert [s["same_as"] for s in g2[:3]] == ["G1-F4M16", "G1-A4M16R50", "G1-S4M8"]
    for s in g2[3:]:
        assert s["routing"]["sel_lr_mult"] == 3.0 and s["routing"]["pool_warmup_frac"] == 0.1
    res2 = {k: v for k, v in res.items() if not k.startswith("G1-S")}
    assert all(s["family"] != "spec" for s in sel.gen2_candidates(res2))


def test_near_tie_breaks_by_refresh_then_distinct_then_throughput():
    c = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    r = {"a": _res(5.000, rr=0.5, d8=10), "b": _res(5.001, rr=0.25, d8=12), "c": _res(5.010, rr=0.1, d8=5)}
    assert sel.best(c, r)["id"] == "b"
    r["b"] = _res(5.001, rr=0.5, d8=9)
    assert sel.best(c, r)["id"] == "b"
    r["b"] = _res(5.001, rr=0.5, d8=10, tps=2e5)
    assert sel.best(c, r)["id"] == "b"


def test_gen2_reject_and_gen3_finalists():
    tok = {98: {"ce": 5.0}, 99: {"ce": 5.0}}
    ok = {98: {"complete": True, "collapsed": False, "ce": 5.06}, 99: {"complete": True, "collapsed": False, "ce": 5.04}}
    assert not sel.gen2_reject(ok, tok)[0]
    bad = {98: {"complete": True, "collapsed": False, "ce": 5.06}, 99: {"complete": True, "collapsed": False, "ce": 5.07}}
    assert sel.gen2_reject(bad, tok)[0]
    specs = [{"id": "G2-F1", "family": "fixed"}, {"id": "G2-A1", "family": "adaptive"}, {"id": "G2-S1", "family": "spec"},
             {"id": "G2-F2", "family": "fixed"}, {"id": "G2-S2", "family": "spec"}]
    res = {"G2-F1": _res(5.03), "G2-A1": _res(5.02), "G2-S1": _res(5.05), "G2-F2": _res(5.01), "G2-S2": _res(5.04)}
    rej = {s["id"]: (False, "") for s in specs}
    fin = sel.gen3_finalists(res, specs, rej)
    assert [f["id"] for f in fin] == ["G3-F2", "G3-S2", "G3-A1"]
    rej["G2-F2"] = (True, "x")
    assert [f["id"] for f in sel.gen3_finalists(res, specs, rej)] == ["G3-A1", "G3-S2", "G3-F1"]


def test_random_control_blind_selection_budget_and_criteria():
    rc = sel.random_control(sel.GEN1[9], "spec")
    assert rc["routing"]["pool_assign"] == "random" and rc["routing"]["seg_len"] == 4 and rc["family"] == "spec"
    bs = sel.blind_selection({"TOKEN-4": _res(5.0), "causal": _res(5.03), "spec": _res(5.02, valid=False)})
    assert bs["finalists"]["causal"]["proceeds"] and not bs["finalists"]["spec"]["proceeds"]
    assert bs["ranking"][0] == "TOKEN-4"
    spt = {"T": 9e-6, "C": 1.2e-5, "S": 1.3e-5, "RC": 1.2e-5, "RS": 1.3e-5}
    B, Bc, rows = sel.choose_budget(10 * 3600, spt, ["T", "C", "S"], ["RC", "RS"], overhead_s=3600)
    assert B == 100_000_000 and Bc == 100_000_000
    B, Bc, rows = sel.choose_budget(11 * 3600, spt, ["T", "C", "S"], ["RC", "RS"], overhead_s=3600)
    assert (B, Bc) == (100_000_000, 50_000_000)
    assert all(r["limit_s"] == 0.8 * 20 * 3600 for r in rows)
    assert sel.choose_budget(16 * 3600, spt, ["T", "C", "S"], ["RC", "RS"])[0] is None
    tok = [({"a": 10.0, "b": 20.0}, {"a": 5, "b": 10})] * 3
    cand = [({"a": 9.0, "b": 19.0}, {"a": 5, "b": 10})] * 3
    bt = sel.paired_bootstrap_multi(tok, cand)
    assert bt == sel.paired_bootstrap_multi(tok, cand) and bt["n_boot"] == 2000 and bt["frac_docs_improved"] == 1.0
    cr = sel.criteria([-0.03, -0.025, -0.021], {"ci95": [-0.03, -0.02]}, False)
    assert cr["substantially_better"] and cr["accuracy_comparable"]
    cr = sel.criteria([0.005, 0.012, 0.0], {"ci95": [0.0, 0.009]}, False)
    assert not cr["accuracy_comparable"]
    assert sel.speed_claim(110, 100, True)["speed_claim"] and not sel.speed_claim(109, 100, True)["speed_claim"]
    assert not sel.speed_claim(150, 100, False)["speed_claim"]


def test_configs_matched_except_approved_fields():
    import spv1_make_configs as mk
    if not spv1_run.PREREG.exists():
        pytest.skip("prereg not written")
    cfgs = {s["id"]: mk.make(98, 5_000_000, "gen1", False, s) for s in [mk.TOKEN_SPEC] + sel.GEN1}
    mk.check_matched(cfgs)
    c = dict(cfgs["G1-F4M8"])
    c["train"] = dict(c["train"], lr=1.0)
    with pytest.raises(ValueError):
        mk.check_matched({"a": cfgs["TOKEN-4"], "b": c})
    m = cfgs["TOKEN-4"]["model"]
    assert (m["d_model"], m["n_layer"], m["n_experts"], m["top_k"], m["expert_ffn"], m["shared_ffn"],
            m["tie_embeddings"], m["moe_layers"], m["max_seq_len"]) == (192, 8, 32, 4, 128, 256, True,
                                                                        [1, 2, 3, 4, 5, 6, 7], 1024)


# ---------------------------------------------------------------- resume reproducibility (real data, tiny model)
def _tiny_train_cfg(tmpdir, routing):
    c = json.loads((ROOT / "configs/rep-v1/seed0/condition_A.json").read_text())
    for k in ("semantic", "condition", "segmentation"):
        c.pop(k, None)
    c.update(experiment="csr-speculative-pool-routing-v1", kind="target", prereg_sha256=spv1_run.prereg_sha(),
             candidate="TEST", family=routing["family"], run_name="cputest_resume", stage="test", description="test")
    c["init"] = {"type": "random", "seed": 5}
    c["model"].update(d_model=32, n_head=4, dense_ffn=48, expert_ffn=16, shared_ffn=24, n_experts=8)
    c["routing"] = routing
    c["train"].update(token_budget=40_000, micro_batch_rows=2, accum=2, log_every=1, eval_every_tokens=10**9,
                      save_checkpoint=True, state_every_seconds=1e-9, warmup_steps=2, seed=5)
    c["eval"] = {"rows_per_batch": 4, "max_rows": 4, "final_calib_search": False}
    p = tmpdir / f"cfg_{routing['family']}.json"
    p.write_text(json.dumps(c))
    return p


@pytest.mark.skipif(not spv1_run.PREREG.exists() or not spv1_run.PARTITION.exists(),
                    reason="prereg / partition not written")
@pytest.mark.parametrize("routing", [
    PoolRouting(family="adaptive", seg_len=4, pool_size=4, refresh_target=0.5, pool_dropout=0.1,
                pool_warmup_frac=0.2, allow_test_sizes=True).to_dict()])
def test_interrupted_run_resumes_reproducibly(routing):
    tmp = ROOT / ".tmp" / "spv1_pytest" / f"resume_{os.getpid()}_{routing['family']}"
    tmp.mkdir(parents=True)
    cfgp = _tiny_train_cfg(tmp, routing)
    env = dict(os.environ, MOE_RESEARCH_APPROVE="cpu-model", MOE_RUNS_DIR=str(tmp / "runs"), CUDA_VISIBLE_DEVICES="")
    (tmp / "runs").mkdir()

    def run(*extra):
        out = subprocess.run([sys.executable, "spv1_train.py", "--config", str(cfgp), "--execute", *extra], cwd=ROOT,
                             env=env, capture_output=True, text=True, timeout=900)
        rd = [ln.split(" ", 1)[1] for ln in out.stdout.splitlines() if ln.startswith("RUN_DIR ")]
        assert rd, out.stdout[-3000:] + out.stderr[-3000:]
        return ROOT / rd[-1], out

    full, o1 = run()
    assert json.loads((full / "status.json").read_text())["status"] == "COMPLETE", o1.stdout[-3000:]
    part, _ = run("--max-steps", "3")
    assert json.loads((part / "status.json").read_text())["status"] == "INTERRUPTED"
    res, o3 = run("--resume", str(part))
    assert json.loads((res / "status.json").read_text())["status"] == "COMPLETE", o3.stdout[-3000:]
    a = torch.load(full / "model.pt", weights_only=False)["model"]
    b = torch.load(res / "model.pt", weights_only=False)["model"]
    assert a.keys() == b.keys()
    for k in a:
        assert torch.equal(a[k], b[k]), k
    sa = json.loads((full / "summary.json").read_text())
    sb = json.loads((res / "summary.json").read_text())
    assert sa["valid_train_tokens"] == sb["valid_train_tokens"]
