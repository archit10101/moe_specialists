"""csr-conditional-local-attention-v1 correctness tests (CUDA hidden). Numbered comments map to the protocol list:
 1 full baseline == existing implementation        8 zero gradients from future tokens
 2 local attention == independent dense reference  9 full-sequence == incremental decoding
 3 no future-token influence                       10 checkpoint save / resume equivalence
 4 padding and document boundaries                 11 identical MoE expert choices old vs new baseline
 5 compact conditional attention == dense ref      12 identical initial weights across paired conditions
 6 hard gate forward behaviour                     13 training-token / optimizer-step accounting
 7 gate gradients on counterfactual steps          14 validation / test never read during search
"""
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from conftest import tiny_cfg

from csr import cla_run
from csr.cla_attn import (BLOCK, AttnContext, block_lists, compact_positions, dense_mask, make_layout)
from csr.cla_decode import greedy
from csr.cla_model import AttnPlan, CLAModel, build_cla_model, cf_steps, count_cla_parameters, quantile_threshold
from csr.cla_select import (BASELINE, CANDIDATES, PLANS, accuracy_verdict, blind_select, final_verdict,
                            gen1_advance, gen2_advance, speed_qualifies)
from csr.model import CSRModel, ModelConfig, RoutingConfig

ROOT = Path(__file__).resolve().parents[1]
W = 24  # tiny-model window
TINY_PLANS = {
    "full": AttnPlan("full"),
    "local": AttnPlan("local", window=W),
    "hybrid": AttnPlan("hybrid", window=W, n_full_late=2),
    "cond": AttnPlan("conditional", window=W, target_rate=0.3),
    "cond_g": AttnPlan("conditional", window=W, target_rate=0.3, force="global"),
    "cond_l": AttnPlan("conditional", window=W, target_rate=0.3, force="local"),
}


def cfg4():
    return tiny_cfg(n_layer=4, moe_layers=(1, 2, 3), max_seq_len=256)


def make_batch(B=2, T=128, seed=0, pad_tail=20, vocab=97):
    g = np.random.default_rng(seed)
    ids = g.integers(2, vocab, (B, T))
    doc = np.full((B, T), -1)
    pos = np.zeros((B, T), int)
    gd = np.full((B, T), -1)
    k = 0
    for b in range(B):
        t, d = 0, 0
        end = T - pad_tail if b == 0 else T
        while t < end:
            L = min(int(g.integers(8, 90)), end - t)
            doc[b, t:t + L] = d
            pos[b, t:t + L] = np.arange(L)
            gd[b, t:t + L] = k
            t, d, k = t + L, d + 1, k + 1
    ids[doc < 0] = 1
    tgt = np.full((B, T), -100)
    for b in range(B):
        for t in range(T - 1):
            if doc[b, t] >= 0 and doc[b, t + 1] == doc[b, t]:
                tgt[b, t] = ids[b, t + 1]
    tt = lambda a: torch.as_tensor(a, dtype=torch.long)  # noqa: E731
    return {"input_ids": tt(ids), "doc_ids": tt(doc), "positions": tt(pos), "targets": tt(tgt), "gdoc": tt(gd)}


def build(plan, seed=3, backend="dense", dtype=torch.float64, calibrate_on=None):
    torch.manual_seed(seed)
    m = CLAModel(cfg4(), plan, seed, backend).to(dtype)
    if plan.kind == "conditional":
        tb = calibrate_on if calibrate_on is not None else make_batch(seed=99)
        with torch.no_grad():
            m.train()
            m(tb, calibrate=True)
    m.eval()
    return m


def share_weights(dst, src):
    dst.load_state_dict(src.state_dict())
    return dst


# ------------------------------------------------------------------ 1, 11
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_1_full_baseline_bit_identical_to_original_csrmodel(dtype):
    tb = make_batch()
    torch.manual_seed(3)
    ref = CSRModel(cfg4()).to(dtype).eval()
    m = build(AttnPlan("full"), dtype=dtype)
    with torch.no_grad():
        a, b = ref(tb, RoutingConfig()), m(tb)
    assert torch.equal(a.logits, b.logits)
    assert torch.equal(a.aux_loss, b.aux_loss) and torch.equal(a.z_loss, b.z_loss)
    for x, y in zip(a.routing, b.routing):
        assert torch.equal(x["topi"], y["topi"]) and torch.equal(x["gates"], y["gates"])


def test_1_flex_backend_matches_dense_backend_cpu_eager():
    tb = make_batch(T=128)
    for name, plan in TINY_PLANS.items():
        d = build(plan)
        f = share_weights(CLAModel(cfg4(), plan, 3, "flex").double().eval(), d)
        with torch.no_grad():
            a, b = d(tb), f(tb)
        assert (a.logits - b.logits).abs().max() < 1e-10, name
        for x, y in zip(a.routing, b.routing):
            assert torch.equal(x["topi"], y["topi"]), name
        for ga, gb in zip(a.gate, b.gate):
            assert torch.equal(ga["sel"], gb["sel"]), name


def test_11_full_size_expert_choices_identical_old_vs_new_baseline():
    cfg = json.loads((ROOT / "configs/rep-v1/seed0/condition_A.json").read_text())
    mc = ModelConfig.from_dict(cfg["model"])
    torch.manual_seed(0)
    ref = CSRModel(mc).eval()
    torch.manual_seed(0)
    new = CLAModel(mc, AttnPlan("full"), 0, "dense").eval()
    tb = make_batch(B=1, T=384, seed=5, pad_tail=0, vocab=mc.vocab_size)
    with torch.no_grad():
        a, b = ref(tb, RoutingConfig()), new(tb)
    assert torch.equal(a.logits, b.logits)
    for x, y in zip(a.routing, b.routing):
        assert torch.equal(x["topi"], y["topi"])
    f = share_weights(CLAModel(mc, AttnPlan("full"), 0, "flex").eval(), new)
    with torch.no_grad():
        c = f(tb)
    assert (c.logits - a.logits).abs().max() < 2e-4
    agree = np.mean([float((x["topi"].sort(-1).values == y["topi"].sort(-1).values).all(-1).float().mean())
                     for x, y in zip(a.routing, c.routing)])
    assert agree > 0.999


# ------------------------------------------------------------------ 2
def loop_reference(q, k, v, doc, win):
    """Independent per-query softmax over explicitly enumerated allowed keys. q,k,v (B,H,T,hd)."""
    B, H, T, hd = q.shape
    out = torch.zeros_like(q)
    for b in range(B):
        for t in range(T):
            if doc[b, t] < 0:
                keys = [t]
            else:
                keys = [s for s in range(t + 1) if doc[b, s] == doc[b, t] and (win is None or t - s < win)]
            ks = torch.tensor(keys)
            sc = (q[b, :, t:t + 1] @ k[b, :, ks].transpose(-1, -2)) / math.sqrt(hd)
            out[b, :, t] = (torch.softmax(sc, -1) @ v[b, :, ks])[:, 0]
    return out


@pytest.mark.parametrize("win", [None, 1, 7, 24, 64])
@pytest.mark.parametrize("backend", ["dense", "flex"])
def test_2_local_attention_matches_independent_loop_reference(win, backend):
    tb = make_batch(B=2, T=128, seed=1)
    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(2, 2, 128, 8, generator=g, dtype=torch.float64) for _ in range(3))
    ctx = AttnContext(tb["doc_ids"], backend)
    y = ctx.attend(q, k, v, win).view(2, 128, 2, 8).transpose(1, 2)
    ref = loop_reference(q, k, v, tb["doc_ids"], win)
    assert (y - ref).abs().max() < 1e-12


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("win", [None, 64, 128, 256, 512, 100])
def test_2_block_lists_cover_every_allowed_pair_and_full_blocks_are_full(seed, win):
    g = np.random.default_rng(seed)
    B, T = 3, 1024
    doc = np.full((B, T), -1)
    for b in range(B):
        t, d = 0, 0
        end = T - int(g.integers(0, 300))
        while t < end:
            L = min(int(g.integers(1, 1100)), end - t)
            doc[b, t:t + L] = d
            t, d = t + L, d + 1
    doc = torch.as_tensor(doc)
    lay = make_layout(doc)
    M = torch.block_diag(*[dense_mask(doc[b:b + 1], win)[0, 0] for b in range(B)])
    nb = lay.N // BLOCK
    for qpos in (torch.arange(lay.N), compact_positions(torch.as_tensor(g.random(lay.N) < g.random()))):
        p, f = block_lists(qpos, lay, win, lay.N)
        Mq = torch.zeros(lay.N, lay.N, dtype=torch.bool)
        v = qpos >= 0
        Mq[v] = M[qpos[v]]
        Mb = Mq.view(nb, BLOCK, nb, BLOCK).permute(0, 2, 1, 3)
        need, allb = Mb.any(-1).any(-1), Mb.all(-1).all(-1)
        assert ((p | f).bool() | ~need).all(), "a needed block is missing"
        assert (~f.bool() | allb).all(), "a 'full' block contains a masked pair"
        assert not (p.bool() & f.bool()).any()
        rows_q = qpos.view(nb, BLOCK)
        assert ((p | f).sum(1)[(rows_q < 0).all(1)] == 0).all(), "empty query blocks must have no key blocks"


# ------------------------------------------------------------------ 3, 4
@pytest.mark.parametrize("name", list(TINY_PLANS))
@pytest.mark.parametrize("backend", ["dense", "flex"])
def test_3_future_tokens_never_change_earlier_outputs(name, backend):
    tb = make_batch(B=1, T=128, seed=2, pad_tail=0)
    d = build(TINY_PLANS[name])
    m = share_weights(CLAModel(cfg4(), TINY_PLANS[name], 3, backend).double().eval(), d)
    t = 70
    tb2 = {k: v.clone() for k, v in tb.items()}
    g = torch.Generator().manual_seed(1)
    tb2["input_ids"][0, t + 1:] = torch.randint(2, 97, (128 - t - 1,), generator=g)
    with torch.no_grad():
        a, b = m(tb), m(tb2)
    # continuous values: <= 1e-12 in fp64 (the unchanged MoE dispatch batches tokens per expert, so a different
    # future routing can change bmm blocking at the 1e-17 level, exactly as in the original CSRModel);
    # discrete decisions (experts, gate selections) must be identical
    assert (a.logits[:, :t + 1] - b.logits[:, :t + 1]).abs().max() < 1e-12
    for x, y in zip(a.routing, b.routing):
        assert torch.equal(x["topi"][:, :t + 1], y["topi"][:, :t + 1])
    for x, y in zip(a.gate, b.gate):
        assert torch.equal(x["sel"][:, :t + 1], y["sel"][:, :t + 1])
        assert (x["z"][:, :t + 1] - y["z"][:, :t + 1]).abs().max() < 1e-12


@pytest.mark.parametrize("name", list(TINY_PLANS))
def test_4_documents_and_padding_isolated(name):
    tb = make_batch(B=2, T=128, seed=4, pad_tail=20)
    m = build(TINY_PLANS[name])
    doc = tb["doc_ids"]
    # change every token of document 0 of row 0 -> all other documents unchanged
    tb2 = {k: v.clone() for k, v in tb.items()}
    tb2["input_ids"][0][doc[0] == 0] = 5
    tb3 = {k: v.clone() for k, v in tb.items()}
    tb3["input_ids"][doc < 0] = 7  # change pad token ids
    with torch.no_grad():
        a, b, c = m(tb), m(tb2), m(tb3)
    other = doc != 0
    other[1] = True
    other &= doc >= 0
    assert (a.logits[other] - b.logits[other]).abs().max() < 1e-12
    assert (a.logits[doc >= 0] - c.logits[doc >= 0]).abs().max() < 1e-12
    for x, y in zip(a.routing, b.routing):
        assert torch.equal(x["topi"][other], y["topi"][other])
    for x, y in zip(a.gate, b.gate):
        assert torch.equal(x["sel"][other], y["sel"][other])
    for ga in a.gate:
        assert not (ga["elig"] & ((doc < 0) | (tb["positions"] < W))).any()


def test_4_local_window_restarts_at_document_start():
    """Tokens at in-document position < W see their whole prefix: local == full for them; documents shorter than W
    give identical logits under local and full attention with the same weights."""
    tb = make_batch(B=2, T=128, seed=6)
    full = build(AttnPlan("full"))
    loc = share_weights(CLAModel(cfg4(), AttnPlan("local", window=W), 3).double().eval(), full)
    with torch.no_grad():
        a, b = full(tb), loc(tb)
    doc, pos = tb["doc_ids"], tb["positions"]
    short = torch.zeros_like(doc, dtype=torch.bool)
    for r in range(2):
        for dd in doc[r].unique():
            if dd >= 0 and (doc[r] == dd).sum() <= W:
                short[r] |= doc[r] == dd
    assert short.any()
    assert (a.logits[short] - b.logits[short]).abs().max() < 1e-12
    early = (pos < W) & (doc >= 0)
    ctx = AttnContext(doc, "dense")
    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(2, 2, 128, 8, generator=g, dtype=torch.float64) for _ in range(3))
    assert torch.equal(ctx.attend(q, k, v, None)[early], ctx.attend(q, k, v, W)[early])


# ------------------------------------------------------------------ 5, 6
@pytest.mark.parametrize("backend", ["dense", "flex"])
def test_5_compact_query_attention_matches_dense_reference(backend):
    tb = make_batch(B=2, T=256, seed=7)
    g = torch.Generator().manual_seed(1)
    q, k, v = (torch.randn(2, 2, 256, 8, generator=g, dtype=torch.float64) for _ in range(3))
    ctx = AttnContext(tb["doc_ids"], backend)
    full = ctx.attend(q, k, v, None).reshape(512, -1)
    for frac in (0.0, 0.05, 0.3, 1.0):
        sel = (torch.rand(2, 256, generator=g) < frac) & (tb["doc_ids"] >= 0)
        qpos, order, y = ctx.attend_compact(q, k, v, sel)
        ok = qpos >= 0
        assert torch.equal(order[ok], qpos[ok]) and order.unique().numel() == order.numel()
        assert int(ok.sum()) == int(sel.sum())
        assert torch.equal(qpos[ok].sort().values, torch.nonzero(sel.reshape(-1)).squeeze(1))
        assert (y[ok] - full[qpos[ok]]).abs().max() < 1e-12 if ok.any() else True
        if backend == "flex":
            assert (y[~ok] == 0).all()


def test_5_6_model_level_compact_equals_counterfactual_and_forced_controls():
    tb = make_batch(B=2, T=128, seed=8)
    for backend in ("dense", "flex"):
        c = build(TINY_PLANS["cond"])
        c = share_weights(CLAModel(cfg4(), TINY_PLANS["cond"], 3, backend).double().eval(), c)
        full = share_weights(CLAModel(cfg4(), AttnPlan("full"), 3, backend).double().eval(), c)
        loc = share_weights(CLAModel(cfg4(), AttnPlan("local", window=W), 3, backend).double().eval(), c)
        cg = share_weights(CLAModel(cfg4(), TINY_PLANS["cond_g"], 3, backend).double().eval(), c)
        cl = share_weights(CLAModel(cfg4(), TINY_PLANS["cond_l"], 3, backend).double().eval(), c)
        with torch.no_grad():
            o1, o2 = c(tb, cf=False), c(tb, cf=True)
            assert (o1.logits - o2.logits).abs().max() < 1e-10
            assert all(torch.equal(x["sel"], y["sel"]) for x, y in zip(o1.gate, o2.gate))
            assert (cg(tb).logits - full(tb).logits).abs().max() < 1e-10
            assert (cl(tb).logits - loc(tb).logits).abs().max() < 1e-10
            rates = [float(gi["sel"].sum()) for gi in o1.gate]
            assert 0 < min(rates) and max(rates) < float((tb["doc_ids"] >= 0).sum())


def test_6_hard_gate_layer_output_is_exactly_local_or_full():
    tb = make_batch(B=2, T=128, seed=9)
    m = build(TINY_PLANS["cond"])
    captured = {}

    def hook(mod, inp):
        captured.setdefault("y", []).append(inp[0].detach().clone())
    hs = [blk.attn.proj.register_forward_pre_hook(hook) for blk in m.blocks]
    orig_attend = AttnContext.attend
    seen = {"full": [], "loc": []}

    def spy(self, q, k, v, window):
        y = orig_attend(self, q, k, v, window)
        if window is not None:
            seen["loc"].append(y.detach().clone())
            seen["full"].append(orig_attend(self, q, k, v, None).detach().clone())
        return y
    AttnContext.attend = spy
    try:
        with torch.no_grad():
            out = m(tb)
    finally:
        AttnContext.attend = orig_attend
        for h in hs:
            h.remove()
    for li, gi in enumerate(out.gate):
        y, yl, yf = captured["y"][li], seen["loc"][li], seen["full"][li]
        s = gi["sel"]
        assert torch.equal(y[~s], yl[~s])
        assert (y[s] - yf[s]).abs().max() < 1e-12
        assert torch.equal(s, gi["elig"] & (gi["z"] > m.thr[li]))
    # moving the threshold changes decisions monotonically
    with torch.no_grad():
        m.thr += 1e6
        assert all(not gi["sel"].any() for gi in m(tb).gate)
        m.thr -= 2e6
        assert all(torch.equal(gi["sel"], gi["elig"]) for gi in m(tb).gate)


def test_6_quantile_threshold_and_controller():
    z = torch.tensor([0.5, -1.0, 2.0, 0.1, 0.3])
    for k in range(0, 6):
        assert int((z > quantile_threshold(z, k)).sum()) == k
    tb = make_batch(B=2, T=256, seed=10)
    m = build(AttnPlan("conditional", window=W, target_rate=0.25), calibrate_on=tb)
    nonpad = int((tb["doc_ids"] >= 0).sum())
    with torch.no_grad():
        out = m(tb)
    for gi in out.gate:
        assert abs(float(gi["sel"].sum()) / nonpad - 0.25) < 0.01
    m.train()
    thr0 = m.thr.clone()
    z_lists = [[gi["z"][gi["elig"]] + 5.0] for gi in out.gate]  # gate drifts up -> threshold follows (EMA 0.1)
    m.thr_update(z_lists, nonpad)
    assert torch.allclose(m.thr, thr0 + 0.1 * 5.0, atol=1e-6)


# ------------------------------------------------------------------ 7
@pytest.mark.parametrize("cf", [True, False])
@pytest.mark.parametrize("backend", ["dense", "flex"])
def test_7_gate_gradient_is_the_documented_straight_through_estimator(cf, backend):
    tb = make_batch(B=2, T=128, seed=11)
    m0 = build(TINY_PLANS["cond"])
    m = share_weights(CLAModel(cfg4(), TINY_PLANS["cond"], 3, backend).double(), m0)
    m.train()
    zs, ys = [], []
    orig = CLAModel.gate_logits

    def gl(self, li, y_loc):
        z = orig(self, li, y_loc)
        z.retain_grad()
        zs.append((z, y_loc.detach().clone()))
        return z
    hs = [blk.attn.proj.register_forward_pre_hook(lambda mod, inp: (inp[0].retain_grad(), ys.append(inp[0]))[1])
          for blk in m.blocks]
    CLAModel.gate_logits = gl
    try:
        out = m(tb, cf=cf)
        loss = F.cross_entropy(out.logits.view(-1, 97), tb["targets"].view(-1), ignore_index=-100)
        loss.backward()
    finally:
        CLAModel.gate_logits = orig
        for h in hs:
            h.remove()
    for li, ((z, yl), y, gi) in enumerate(zip(zs, ys, out.gate)):
        avail = gi["elig"] if cf else gi["sel"]
        sig = torch.sigmoid(z - m.thr[li])
        dsig = (sig * (1 - sig)).detach()
        g = z.grad
        assert torch.isfinite(g).all()
        assert (g[~avail] == 0).all(), "gate gradient leaked to tokens without a computed full-attention path"
        if not cf:
            expect = (y.grad * (y.detach() - yl)).sum(-1) * dsig
            assert torch.allclose(g[avail], expect[avail], atol=1e-14, rtol=1e-9)
    gg = [p.grad for n, p in m.named_parameters() if n.startswith("gates.")]
    assert all(x is not None and torch.isfinite(x).all() for x in gg)
    assert sum(float(x.abs().sum()) for x in gg) > 0


def test_7_counterfactual_step_gives_gradient_to_unselected_tokens_exactly():
    """With every token unselected, an ordinary step gives no gate gradient; a counterfactual step gives exactly
    dL/dz_t = <dL/dout_t, full_t - local_t> * sigmoid'(z_t - lambda) for every eligible token."""
    tb = make_batch(B=2, T=128, seed=12)
    m = build(TINY_PLANS["cond"])
    m.train()
    m.thr.fill_(10.0)  # nothing selected; sigmoid'(z - lambda) is small but nonzero in fp64
    out = m(tb, cf=False)
    F.cross_entropy(out.logits.view(-1, 97), tb["targets"].view(-1), ignore_index=-100).backward()
    assert all(p.grad is None or float(p.grad.abs().sum()) == 0 for n, p in m.named_parameters()
               if n.startswith("gates."))
    m.zero_grad(set_to_none=True)
    zs, ys, fl = [], [], []
    orig = CLAModel.gate_logits
    orig_attend = AttnContext.attend

    def gl(self, li, y_loc):
        z = orig(self, li, y_loc)
        z.retain_grad()
        zs.append((z, y_loc.detach().clone()))
        return z

    def spy(self, q, k, v, window):
        y = orig_attend(self, q, k, v, window)
        if window is None:
            fl.append(y.detach().clone())
        return y
    hs = [blk.attn.proj.register_forward_pre_hook(lambda mod, inp: (inp[0].retain_grad(), ys.append(inp[0]))[1])
          for blk in m.blocks]
    CLAModel.gate_logits, AttnContext.attend = gl, spy
    try:
        out = m(tb, cf=True)
        F.cross_entropy(out.logits.view(-1, 97), tb["targets"].view(-1), ignore_index=-100).backward()
    finally:
        CLAModel.gate_logits, AttnContext.attend = orig, orig_attend
        for h in hs:
            h.remove()
    for li, ((z, yl), y, yf, gi) in enumerate(zip(zs, ys, fl, out.gate)):
        assert not gi["sel"].any()
        assert torch.equal(y.detach(), yl), "forward must keep the hard (local) decision on a counterfactual step"
        sig = torch.sigmoid(z - m.thr[li]).detach()
        expect = (y.grad * (yf - yl)).sum(-1) * sig * (1 - sig)
        e = gi["elig"]
        assert torch.allclose(z.grad[e], expect[e], atol=1e-16, rtol=1e-8)
        assert (z.grad[~e] == 0).all()
        assert float(z.grad[e].abs().sum()) > 0
    assert sum(float(p.grad.abs().sum()) for n, p in m.named_parameters() if n.startswith("gates.")) > 0


def test_7_cf_schedule_exactly_ten_percent_deterministic():
    for n in (153, 611, 3052, 7):
        a = cf_steps(98, n, 0.10)
        assert a == cf_steps(98, n, 0.10) and len(a) == round(0.1 * n) and all(0 <= s < n for s in a)
    assert cf_steps(0, 3052, 0.1) != cf_steps(1, 3052, 0.1)


# ------------------------------------------------------------------ 8
@pytest.mark.parametrize("name", ["full", "local", "hybrid", "cond"])
@pytest.mark.parametrize("cf", [False, True])
def test_8_zero_gradients_from_future_tokens(name, cf):
    tb = make_batch(B=1, T=128, seed=13, pad_tail=0)
    m = build(TINY_PLANS[name])
    m.train()
    emb = {}

    def hook(mod, inp, out):
        out.retain_grad()
        emb["x"] = out
    h = m.tok_emb.register_forward_hook(hook)
    out = m(tb, cf=cf)
    h.remove()
    t = 60
    out.logits[0, t].sum().backward()
    g = emb["x"].grad[0]
    assert (g[t + 1:] == 0).all()
    assert float(g[:t + 1].abs().sum()) > 0


# ------------------------------------------------------------------ 9
@pytest.mark.parametrize("name", list(TINY_PLANS))
def test_9_incremental_decoding_equals_full_sequence(name):
    m = build(TINY_PLANS[name])
    g = torch.Generator().manual_seed(4)
    prompt = torch.randint(2, 97, (3, 20), generator=g)
    n_new = 40  # crosses the window (W = 24) so local / conditional paths are exercised
    gen, logits, sels = greedy(m, prompt, n_new, torch.float64)
    seq = torch.cat([prompt, gen[:, :-1]], 1)
    L = seq.shape[1]
    tb = {"input_ids": seq, "doc_ids": torch.zeros_like(seq),
          "positions": torch.arange(L).unsqueeze(0).expand(3, L).contiguous()}
    with torch.no_grad():
        out = m(tb)
    full = out.logits[:, 19:]
    inc = torch.stack(logits, 1)
    assert (full - inc).abs().max() < 1e-10
    if name.startswith("cond"):
        for i, step_sel in enumerate(sels):
            t = 20 + i
            for ci, s in enumerate(step_sel):
                assert torch.equal(s, out.gate[ci]["sel"][:, t])
        assert any(bool(s.any()) for step_sel in sels for s in step_sel) or name == "cond_l"


# ------------------------------------------------------------------ 12, parameters
def _real_cfg(seed, plan):
    c = json.loads((ROOT / "configs/rep-v1/seed0/condition_A.json").read_text())
    return {"model": c["model"], "init": {"type": "random", "seed": seed}, "attention": plan}


@pytest.mark.parametrize("seed", [0, 98])
def test_12_identical_initial_weights_across_all_conditions_within_seed(seed):
    from csr.train_lib import build_model
    sds = {cid: build_cla_model(_real_cfg(seed, p), "cpu").state_dict() for cid, p in PLANS.items()}
    ref = sds[BASELINE]
    for cid, sd in sds.items():
        assert sd.keys() == ref.keys()
        assert all(torch.equal(sd[k], ref[k]) for k in ref), cid
    c = json.loads((ROOT / "configs/rep-v1/seed0/condition_A.json").read_text())
    c["init"] = {"type": "random", "seed": seed}
    tok = build_model(c, "cpu").state_dict()
    for k, v in tok.items():
        assert torch.equal(ref[k], v), k  # backbone == the inherited TOKEN-4 initialisation
    other = build_cla_model(_real_cfg(seed + 1, PLANS[BASELINE]), "cpu").state_dict()
    assert not torch.equal(other["gates.0.fc1.weight"], ref["gates.0.fc1.weight"])


def test_gate_parameters_below_point_three_percent_and_inventory():
    m = build_cla_model(_real_cfg(0, PLANS["G256-25"]), "cpu")
    c = count_cla_parameters(m)
    assert c["backbone_total"] == 28_873_920
    assert c["gate"] == 8 * (192 * 32 + 32 + 32 + 1) and c["gate_fraction"] < 0.003
    for cid, p in PLANS.items():
        mm = build_cla_model(_real_cfg(0, p), "cpu")
        assert count_cla_parameters(mm)["total"] == c["total"]


# ------------------------------------------------------------------ 13
def test_13_token_and_step_accounting_identical_across_conditions():
    sys.path.insert(0, str(ROOT))
    import cla_make_configs as mk
    from csr.data import TokenSplit, train_order
    from csr.train_lib import BatchSource, plan_steps, valid_tokens
    tr = TokenSplit.load(ROOT / cla_run.DATA_DIR, "train")
    for seed, budget in ((98, 5_000_000), (0, 2_000_000)):
        plans = {}
        for cid in PLANS:
            c = mk.make(seed, budget, "gen1", False, cid)
            assert c["train"]["micro_batch_rows"] * c["train"]["accum"] == 32
            src = BatchSource(tr, c["data"]["seq_len"], None, train_order(tr.n_docs, c["data"]["train_order_seed"]))
            plans[cid] = plan_steps(src, 32, budget)
        steps0, tot0 = plans[BASELINE]
        for cid, (steps, tot) in plans.items():
            assert steps == steps0 and tot == tot0, cid
        assert tot0 >= budget and tot0 - valid_tokens(steps0[-1], tr) < budget
        assert sum(valid_tokens(r, tr) for r in steps0) == tot0


# ------------------------------------------------------------------ 14
def test_14_test_split_refused_before_any_file_access(monkeypatch):
    from csr import data
    monkeypatch.setattr(data.TokenSplit, "load", classmethod(lambda cls, d, s: (_ for _ in ()).throw(
        AssertionError("file access attempted"))))
    for s in ("test", "TEST", "test_tokens", "../test"):
        with pytest.raises(cla_run.SplitLocked):
            cla_run.load_raw_split(ROOT / cla_run.DATA_DIR, s)
    with pytest.raises(cla_run.SplitLocked):
        cla_run.eval_source({"data": {"seq_len": 1024}}, "test", ROOT / cla_run.DATA_DIR)


def test_14_split_locks_and_single_use(tmp_path, monkeypatch):
    monkeypatch.setattr(cla_run, "SEARCH_FROZEN", tmp_path / "search_frozen.json")
    monkeypatch.setattr(cla_run, "FINAL_FROZEN", tmp_path / "final_frozen.json")
    monkeypatch.setattr(cla_run, "FINAL_TRAINED", tmp_path / "final_trained.json")
    assert cla_run.check_eval_access("calib_search")
    for s in ("calib_selection", "val", "test"):
        with pytest.raises(cla_run.SplitLocked):
            cla_run.check_eval_access(s, "abc")
    (tmp_path / "search_frozen.json").write_text(json.dumps({"config_sha256s": ["abc"]}))
    assert cla_run.check_eval_access("calib_selection", "abc")
    with pytest.raises(cla_run.SplitLocked):
        cla_run.check_eval_access("calib_selection", "zzz")
    (tmp_path / "final_frozen.json").write_text(json.dumps({"config_sha256s": ["f1", "f2"]}))
    with pytest.raises(cla_run.SplitLocked):  # final runs not all complete yet
        cla_run.check_eval_access("val", "f1")
    (tmp_path / "final_trained.json").write_text(json.dumps({"checkpoints": {"a": 1}}))
    with pytest.raises(cla_run.SplitLocked):
        cla_run.check_eval_access("val", "f1")
    (tmp_path / "final_trained.json").write_text(json.dumps({"checkpoints": {"a": 1, "b": 2}}))
    assert cla_run.check_eval_access("val", "f1")
    with pytest.raises(cla_run.SplitLocked):
        cla_run.check_eval_access("val", "abc")
    monkeypatch.setattr(cla_run, "EVAL_REGISTRY", ROOT / ".tmp" / "clav1_pytest" / f"reg_{os.getpid()}")
    p = cla_run.claim_eval("val", "ck1")
    p.write_text(json.dumps({"status": "COMPLETE", "result": "x"}))
    with pytest.raises(cla_run.SplitLocked):
        cla_run.claim_eval("val", "ck1")


def test_14_search_code_never_references_val_selection_or_test():
    for f in ("cla_train.py", "src/csr/cla_select.py", "src/csr/cla_model.py", "src/csr/cla_attn.py",
              "cla_make_configs.py"):
        src = (ROOT / f).read_text()
        for bad in ('"val"', "'val'", 'calib_selection"', "val_tokens", "test_tokens", "selection_doc"):
            assert bad not in src, (f, bad)
    assert 'eval_source(cfg, "calib_search"' in (ROOT / "cla_train.py").read_text()


# ------------------------------------------------------------------ selection rules (frozen)
def _r(ce, tps=1e5, gpu=100.0, ok=True):
    return {"complete": ok, "finite": ok, "collapsed": not ok, "ce": ce, "train_tok_s": tps, "gpu_s": gpu}


def test_gen1_elimination_ranking_and_limit():
    res = {BASELINE: _r(6.00)}
    ces = [6.06, 6.01, 6.003, 6.002, 6.004, 5.99, 6.00, 6.001, 5.995, 6.02]
    for (cid, _, _), ce in zip(CANDIDATES, ces):
        res[cid] = _r(ce, gpu=100 + len(cid))
    res["G256-25"] = _r(6.00, tps=0.85e5)  # slow, no compensating benefit -> eliminated
    res["G256-50"] = _r(5.985, tps=0.85e5)  # slow but 0.015 better -> kept
    res["G512-25"] = _r(6.00, ok=False)
    adv, order, elim = gen1_advance(res)
    assert elim["LOCAL-W128"][0] and elim["G256-25"][0] and elim["G512-25"][0]
    assert not elim["G256-50"][0]
    assert len(adv) == 4 and adv[0] == "G256-50"
    # near tie: G128-25 (5.99) vs G256-10 (6.00)? 0.01 apart -> no tie; LOCAL-W512/HYBRID tie within 0.002
    assert adv[1] == "G128-25"


def test_gen2_and_blind_selection_and_verdicts():
    ids = ["A1", "A2", "A3"]
    per = {BASELINE: {98: _r(5.0), 99: _r(5.1)}, "A1": {98: _r(5.010, tps=1.0e5), 99: _r(5.110, tps=1.0e5)},
           "A2": {98: _r(5.012, tps=1.2e5), 99: _r(5.111, tps=1.2e5)}, "A3": {98: _r(4.9), 99: _r(5.0, ok=False)}}
    adv, order, md, tput, inval = gen2_advance(per, ids)
    assert inval == ["A3"] and adv[0] == "A2" and len(adv) == 2
    import csr.cla_select as cs
    old = dict(cs.PLANS)
    cs.PLANS.update({"A1": {"kind": "conditional"}, "A2": {"kind": "local"}})
    try:
        sel = {BASELINE: per[BASELINE], "A1": per["A1"], "A2": per["A2"]}
        w, info = blind_select(sel, ["A1", "A2"], {"A1": 1e5, "A2": 1.004e5}, {"A1": 0.2, "A2": 0.0})
        assert w == "A2"  # tied on CE and speed (<1%), lower global-query fraction
    finally:
        cs.PLANS.clear()
        cs.PLANS.update(old)
    v = accuracy_verdict({0: 0.009, 1: 0.012, 2: 0.008})
    assert v["comparable"] and not v["better"]  # mean 0.0097 <= 0.010 and every seed <= 0.020
    assert not accuracy_verdict({0: 0.011, 1: 0.012, 2: 0.010})["comparable"]  # mean 0.011 > 0.010
    v = accuracy_verdict({0: 0.009, 1: 0.019, 2: 0.0})
    assert v["comparable"] and not v["better"]
    v = accuracy_verdict({0: -0.009, 1: 0.021, 2: -0.03})
    assert not v["comparable"]
    assert accuracy_verdict({0: -0.02, 1: -0.01, 2: -0.005})["better"]
    assert speed_qualifies(0.89) and not speed_qualifies(0.91)
    assert final_verdict(True, True, True) == "SUCCESS" and final_verdict(True, True, False) == "ACCURACY-ONLY"
    assert final_verdict(True, False, True) == "SPEED-ONLY" and final_verdict(True, False, False) == "NEGATIVE"
    assert final_verdict(False, True, True) == "INVALID"


# ------------------------------------------------------------------ 10 resume, 14 dynamic isolation (real data, tiny)
def _tiny_train_cfg(tmpdir, plan):
    c = json.loads((ROOT / "configs/rep-v1/seed0/condition_A.json").read_text())
    for k in ("semantic", "condition", "segmentation", "routing"):
        c.pop(k, None)
    c.update(experiment="csr-conditional-local-attention-v1", prereg_sha256=cla_run.prereg_sha(), candidate="TEST",
             condition="G", role="candidate", run_name="cputest_resume", stage="test")
    c["init"] = {"type": "random", "seed": 5}
    c["model"].update(d_model=32, n_head=4, dense_ffn=48, expert_ffn=16, shared_ffn=24, n_experts=8)
    c["attention"] = plan
    c["train"].update(token_budget=40_000, micro_batch_rows=2, accum=2, log_every=1, eval_every_tokens=10**9,
                      save_checkpoint=True, state_every_seconds=1e-9, warmup_steps=2, seed=5)
    c["eval"] = {"rows_per_batch": 4, "max_rows": 4, "final_calib_search": False}
    p = tmpdir / f"cfg_{plan['kind']}.json"
    p.write_text(json.dumps(c))
    return p


AUDIT = ("import sys, runpy, json\nlog=open(sys.argv[1],'w')\n"
         "def h(ev,args):\n    if ev=='open' and isinstance(args[0],str): log.write(args[0]+'\\n'); log.flush()\n"
         "sys.addaudithook(h)\nsys.argv=sys.argv[2:]\nrunpy.run_path(sys.argv[0], run_name='__main__')\n")


@pytest.mark.skipif(not cla_run.PREREG.exists() or not cla_run.PARTITION.exists(),
                    reason="prereg / partition not written")
@pytest.mark.parametrize("plan", [AttnPlan("conditional", window=64, target_rate=0.3, cf_frac=0.3).to_dict(),
                                  AttnPlan("hybrid", window=64, n_full_late=1).to_dict()])
def test_10_14_interrupted_run_resumes_bit_exactly_and_never_opens_val_or_test(plan):
    tmp = ROOT / ".tmp" / "clav1_pytest" / f"resume_{os.getpid()}_{plan['kind']}"
    tmp.mkdir(parents=True)
    cfgp = _tiny_train_cfg(tmp, plan)
    env = dict(os.environ, MOE_RESEARCH_APPROVE="cpu-model", MOE_RUNS_DIR=str(tmp / "runs"), CUDA_VISIBLE_DEVICES="")
    (tmp / "runs").mkdir()
    (tmp / "audit.py").write_text(AUDIT)

    def run(tag, *extra):
        out = subprocess.run([sys.executable, str(tmp / "audit.py"), str(tmp / f"opened_{tag}.txt"), "cla_train.py",
                              "--config", str(cfgp), "--execute", *extra], cwd=ROOT, env=env, capture_output=True,
                             text=True, timeout=900)
        rd = [ln.split(" ", 1)[1] for ln in out.stdout.splitlines() if ln.startswith("RUN_DIR ")]
        assert rd, out.stdout[-3000:] + out.stderr[-3000:]
        return ROOT / rd[-1], out

    full, o1 = run("full")
    assert json.loads((full / "status.json").read_text())["status"] == "COMPLETE", o1.stdout[-3000:]
    part, _ = run("part", "--max-steps", "3")
    assert json.loads((part / "status.json").read_text())["status"] == "INTERRUPTED"
    res, o3 = run("res", "--resume", str(part))
    assert json.loads((res / "status.json").read_text())["status"] == "COMPLETE", o3.stdout[-3000:]
    a = torch.load(full / "model.pt", weights_only=False)["model"]
    b = torch.load(res / "model.pt", weights_only=False)["model"]
    assert a.keys() == b.keys()
    for k in a:
        assert torch.equal(a[k], b[k]), k
    sa = json.loads((full / "summary.json").read_text())
    sb = json.loads((res / "summary.json").read_text())
    assert sa["valid_train_tokens"] == sb["valid_train_tokens"] and sa["optimizer_steps"] == sb["optimizer_steps"]
    fa = json.loads((full / "run_facts.json").read_text())
    if plan["kind"] == "conditional":
        assert sa["n_cf_steps"] == fa["n_cf_steps"] == round(0.3 * fa["n_steps"])
    for tag in ("full", "part", "res"):
        opened = (tmp / f"opened_{tag}.txt").read_text().splitlines()
        data_files = [o for o in opened if "artifacts/data" in o]
        assert data_files, "audit hook saw no data access"
        for o in data_files:
            name = o.rsplit("/", 1)[-1]
            assert not name.startswith(("val_", "test_")), o
            assert name.startswith(("train_", "calib_", "manifest")), o


@pytest.mark.parametrize("seed", range(4))
def test_2_direct_block_mask_identical_to_torch_construction(seed):
    from torch.nn.attention.flex_attention import _create_sparse_block_from_block_mask
    from csr.cla_attn import _block_mask
    g = torch.Generator().manual_seed(seed)
    p = (torch.rand(24, 24, generator=g) < 0.3).to(torch.int8)
    f = ((torch.rand(24, 24, generator=g) < 0.2) & ~p.bool()).to(torch.int8)
    a = _block_mask(p, f, None)
    b = _create_sparse_block_from_block_mask((p[None, None], f[None, None]), None)
    for name in ("kv_num_blocks", "kv_indices", "full_kv_num_blocks", "full_kv_indices", "q_num_blocks", "q_indices",
                 "full_q_num_blocks", "full_q_indices"):
        x, y = getattr(a, name), getattr(b, name)
        n = getattr(a, name.replace("indices", "num_blocks")) if "indices" in name else None
        if n is None:
            assert torch.equal(x, y), name
        else:  # only the first n entries of each row are meaningful
            for r in range(x.shape[-2]):
                k = int(n[0, 0, r])
                assert torch.equal(x[0, 0, r, :k], y[0, 0, r, :k]), name
