#!/usr/bin/env python
# semantic-lift-245m-confirm-v1 (slc): path-only copy of sll_tests.py (output -> slc artifacts); the tested module
# src/csr/sll_lift.py is the reused, unchanged implementation.
"""csr-semantic-lift-expert-learning-v1 CPU correctness tests of the lift-weighted expert-gradient operator (prereg
section 6, tests 1-9, 11 at unit level; trainer-level tests 1, 10, 12 run on GPU in sll_gpu_checks.py).
Writes artifacts/semantic-lift-expert-learning-v1/cpu_tests_<stamp>.json. Tiny LFR model (same classes as the main
model), float64 (exactness) and CPU bf16 autocast (the training precision path)."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402

RES = []


def rec(name, ok, **info):
    RES.append({"test": name, "ok": bool(ok), **info})
    print(("PASS " if ok else "FAIL ") + name + " " + json.dumps(info, default=str)[:400], flush=True)


def tiny(seed=0, dtype=None):
    import torch
    from csr.lfr_model import LFRConfig, LFRModel
    cfg = LFRConfig(vocab_size=97, n_layer=4, d_model=32, n_head=2, dense_ffn=48, n_experts=8, top_k=4, expert_ffn=16,
                    shared_ffn=24)
    m = LFRModel(cfg, seed)
    if dtype is not None:
        m = m.to(dtype)
    return m


def batch(seed=0, B=3, T=24, K=5):
    import numpy as np
    import torch
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 97, (B, T), generator=g)
    doc = torch.zeros(B, T, dtype=torch.long)
    pos = torch.zeros(B, T, dtype=torch.long)
    sid = torch.zeros(B, T, dtype=torch.long)
    s = 0
    for b in range(B):
        cut = int(torch.randint(5, T - 5, (1,), generator=g))
        doc[b, :cut], doc[b, cut:] = 2 * b, 2 * b + 1
        pos[b, :cut], pos[b, cut:] = torch.arange(cut), torch.arange(T - cut)
        sid[b, :cut], sid[b, cut:] = s, s + 1
        s += 2
    doc[-1, -3:] = -1  # pads
    sid[-1, -3:] = -1
    pos[-1, -3:] = 0
    q = torch.softmax(torch.randn(s, K, generator=g) * 2, -1)
    q[1] = 0.0  # a segment without assignment
    return {"input_ids": ids, "doc_ids": doc, "positions": pos}, sid, q


def install_ctx(m, K, lam, seed=1, dtype=None):
    import torch
    from csr.sll_lift import LiftContext, install
    ctx = LiftContext(m.n_moe, K, m.cfg.n_experts)
    install(m, ctx)
    g = torch.Generator().manual_seed(seed)
    ctx.table = (torch.randn(m.n_moe, K, m.cfg.n_experts, generator=g) * 0.7).to(torch.float32)
    ctx.scale = [0.4] * m.n_moe
    ctx.lam = lam
    return ctx


def set_mb(ctx, sid, q, bt):
    from csr.sll_lift import q_tokens
    ctx.q_tok = q_tokens(q, sid)
    ctx.nonpad = (bt["doc_ids"] >= 0).reshape(-1)


def loss_of(out, R):
    return (out["hidden"] * R).sum() + 0.01 * out["aux"] + 0.001 * out["z"]


def grads(m, bt, R, ctl=None):
    import torch
    m.zero_grad(set_to_none=True)
    out = m(bt, ctl or {"collect": True})
    loss_of(out, R).backward()
    return out, {n: (p.grad.clone() if p.grad is not None else torch.zeros_like(p)) for n, p in m.named_parameters()}


def t_forward_and_grads():
    """Tests 2-7 in float64: forward identical; router / shared / attention / non-expert / expert-input gradients
    identical; expert parameter gradients equal the explicit per-example reference sum_i m_i J_i^T g_i."""
    import torch
    torch.manual_seed(0)
    K = 5
    bt, sid, q = batch()
    nat = tiny(dtype=torch.float64)
    lif = tiny(dtype=torch.float64)
    ctx = install_ctx(lif, K, lam=1.0, dtype=torch.float64)
    set_mb(ctx, sid, q, bt)
    ctx.record = True
    R = torch.randn(*bt["input_ids"].shape, 32, dtype=torch.float64, generator=torch.Generator().manual_seed(5))
    # capture expert inputs / output gradients of the native model (for the per-example reference)
    caps = {}

    def fh(mod, inp, out, li):
        caps[li] = {"x": inp[0].detach(), "topi": inp[1].detach(), "gates": inp[2].detach()}
        out.register_hook(lambda g: caps[li].__setitem__("g", g.detach()))
    hs = []
    li = 0
    for blk in nat.blocks:
        if blk.is_moe:
            hs.append(blk.ffn.experts.register_forward_hook(lambda mod, i, o, li=li: fh(mod, i, o, li)))
            li += 1
    o_n, g_n = grads(nat, bt, R)
    for h in hs:
        h.remove()
    # multipliers the lift model applies (recomputed with the same function, per layer)
    from csr.sll_lift import multipliers
    ms = {}
    for l_ in range(nat.n_moe):
        qL = ctx.q_tok.float() @ ctx.table[l_]
        s_sel = qL.gather(1, caps[l_]["topi"])
        mm, _ = multipliers(s_sel, caps[l_]["gates"], 1.0, ctx.scale[l_])
        ms[l_] = torch.where(ctx.nonpad.view(-1, 1), mm, torch.ones_like(mm))
    o_l, g_l = grads(lif, bt, R)
    fwd_same = torch.equal(o_n["hidden"], o_l["hidden"]) and all(
        torch.equal(a["topi"], b["topi"]) and torch.equal(a["p"], b["p"]) for a, b in zip(o_n["records"], o_l["records"]))
    rec("T2 forward / logits / top-4 / gates identical with the operator (float64, bitwise)", fwd_same)
    cats = {"router": [], "shared": [], "attention": [], "other_nonexpert": [], "expert": []}
    for n in g_n:
        if ".ffn.router." in n:
            cats["router"].append(n)
        elif ".ffn.shared." in n:
            cats["shared"].append(n)
        elif ".attn." in n:
            cats["attention"].append(n)
        elif ".ffn.experts." in n:
            cats["expert"].append(n)
        else:
            cats["other_nonexpert"].append(n)
    for c, tn in (("router", "T3"), ("shared", "T4"), ("attention", "T5a"), ("other_nonexpert", "T5b")):
        md = max(float((g_n[n] - g_l[n]).abs().max()) for n in cats[c])
        rec(f"{tn} {c} gradients unchanged (float64; max abs diff)", md == 0.0, max_abs_diff=md, n_params=len(cats[c]))
    # T6: gradient into the expert input: compare dL/dx at each MoE expert input via hooks on both models
    xin = {"nat": {}, "lif": {}}

    def hook_in(model, tag):
        hl = []
        li_ = 0
        for blk in model.blocks:
            if blk.is_moe:
                def pre(mod, inp, li=li_):
                    x = inp[0]
                    if x.requires_grad:
                        x.register_hook(lambda g, li=li: xin[tag].__setitem__(li, g.detach().clone()))
                hl.append(blk.ffn.experts.register_forward_pre_hook(pre))
                li_ += 1
        return hl
    h1, h2 = hook_in(nat, "nat"), hook_in(lif, "lif")
    grads(nat, bt, R)
    grads(lif, bt, R)
    for h in h1 + h2:
        h.remove()
    md = max(float((xin["nat"][k] - xin["lif"][k]).abs().max()) for k in xin["nat"])
    rec("T6 gradient passed to the expert input unchanged (float64)", md == 0.0 and len(xin["nat"]) == nat.n_moe,
        max_abs_diff=md)
    # T7: explicit per-example reference for the expert parameter gradients
    import torch.nn.functional as F
    worst = 0.0
    li_ = 0
    for bi, blk in enumerate(nat.blocks):
        if not blk.is_moe:
            continue
        c = caps[li_]
        x, topi, gates, G = c["x"], c["topi"], c["gates"], c["g"]
        w1, w2 = blk.ffn.experts.w1.detach(), blk.ffn.experts.w2.detach()
        r1, r2 = torch.zeros_like(w1), torch.zeros_like(w2)
        for n in range(x.shape[0]):
            for j in range(topi.shape[1]):
                e = int(topi[n, j])
                W1 = w1[e].clone().requires_grad_(True)
                W2 = w2[e].clone().requires_grad_(True)
                y = F.gelu(x[n:n + 1] @ W1, approximate="tanh") @ W2  # detached input: parameter path only
                contrib = (G[n:n + 1] * gates[n, j] * y).sum()
                a1, a2 = torch.autograd.grad(contrib, (W1, W2))
                r1[e] += ms[li_][n, j] * a1
                r2[e] += ms[li_][n, j] * a2
        n1, n2 = f"blocks.{bi}.ffn.experts.w1", f"blocks.{bi}.ffn.experts.w2"
        rel = max(float((g_l[n1] - r1).abs().max() / r1.abs().max()), float((g_l[n2] - r2).abs().max() / r2.abs().max()))
        worst = max(worst, rel)
        # sanity: the native gradient equals the m = 1 reference and differs from the weighted one
        li_ += 1
    rec("T7 expert parameter gradients == explicit per-example reference sum_i m_i J_i^T g_i (float64)", worst < 1e-10,
        max_rel_diff=worst)
    md_e = max(float((g_n[n] - g_l[n]).abs().max()) for n in cats["expert"])
    rec("T7b expert gradients do differ from native when lambda > 0 (operator active)", md_e > 1e-8, max_abs_diff=md_e)
    d = ctx.diag
    ok8 = all(v["m_wmean_maxdev"] < 1e-9 for v in d.values()) and len(d) == nat.n_moe
    rec("T8 gate-weighted mean multiplier == 1 for every token/layer (training path)", ok8,
        max_dev=max(v["m_wmean_maxdev"] for v in d.values()))
    ok9 = all(v["m_min"] >= 0.5 - 1e-12 and v["m_max"] <= 2.0 + 1e-12 for v in d.values())
    rec("T9 bounds 0.5 <= m <= 2 respected (training path)", ok9, m_min=min(v["m_min"] for v in d.values()),
        m_max=max(v["m_max"] for v in d.values()))


def t_lambda0_and_m1():
    """Test 1 (unit level): lambda = 0 -> native code path, bitwise identical forward and all gradients; m = 1 forced
    through the custom operator -> identical (float64) and identical/close under CPU bf16 autocast."""
    import torch
    K = 5
    bt, sid, q = batch(seed=3)
    R = torch.randn(*bt["input_ids"].shape, 32, generator=torch.Generator().manual_seed(9))
    for dt, tag in ((torch.float64, "float64"), (torch.float32, "float32")):
        nat, lif = tiny(dtype=dt), tiny(dtype=dt)
        ctx = install_ctx(lif, K, lam=0.0)
        set_mb(ctx, sid, q, bt)
        o_n, g_n = grads(nat, bt, R.to(dt))
        o_l, g_l = grads(lif, bt, R.to(dt))
        same = torch.equal(o_n["hidden"], o_l["hidden"]) and all(torch.equal(g_n[n], g_l[n]) for n in g_n)
        rec(f"T1 lambda = 0 bitwise identical to native Semantic C6 path ({tag}, forward + all grads)", same)
        ctx.force_m = {l_: torch.ones(bt["input_ids"].numel(), 4, dtype=torch.float64) for l_ in range(lif.n_moe)}
        o_l, g_l = grads(lif, bt, R.to(dt))
        md = max(float((g_n[n] - g_l[n]).abs().max()) for n in g_n)
        rec(f"T1b custom operator with m == 1 reproduces native forward + grads ({tag})",
            torch.equal(o_n["hidden"], o_l["hidden"]) and md == 0.0, max_abs_diff=md)
    # CPU bf16 autocast (training precision path)
    nat, lif = tiny(), tiny()
    ctx = install_ctx(lif, K, lam=0.0)
    set_mb(ctx, sid, q, bt)
    ctx.force_m = {l_: torch.ones(bt["input_ids"].numel(), 4, dtype=torch.float64) for l_ in range(lif.n_moe)}

    def g_ac(m):
        m.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            out = m(bt, {"collect": True})
            loss = loss_of(out, R)
        loss.backward()
        return out, {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    o_n, g_n = g_ac(nat)
    o_l, g_l = g_ac(lif)
    md = max(float((g_n[n] - g_l[n]).abs().max() / (g_n[n].abs().max() + 1e-30)) for n in g_n)
    rec("T1c m == 1 under CPU bf16 autocast: forward bitwise identical, grads identical (max rel diff)",
        torch.equal(o_n["hidden"], o_l["hidden"]) and md == 0.0, max_rel_diff=md)


def t_projection():
    """Tests 8-9 on adversarial inputs: exact projection vs 200-step bisection; weighted mean 1; bounds."""
    import torch
    from csr.sll_lift import multipliers, project_bisect, project_weighted_mean_one
    g = torch.Generator().manual_seed(0)
    N = 20000
    r = torch.exp(torch.randn(N, 4, generator=g) * 2.5)
    w = torch.softmax(torch.randn(N, 4, generator=g) * 3, -1)
    w[:10] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    r[10:20] = torch.tensor([1e-6, 1e6, 1.0, 1.0])
    r[20:30] = 1.0
    m = project_weighted_mean_one(r, w)
    mb = project_bisect(r, w)
    dev = float(((w.double() * m).sum(1) - 1).abs().max())
    rec("T8b projection: sum_j w_j m_j == 1 (adversarial r, w; float64)", dev < 1e-12, max_dev=dev)
    rec("T9b projection: bounds [0.5, 2]", bool((m >= 0.5).all() and (m <= 2.0).all()))
    rec("T8c exact projection == bisection reference", float((m - mb).abs().max()) < 1e-9,
        max_abs_diff=float((m - mb).abs().max()))
    w64 = w.double() / w.double().sum(1, keepdim=True)
    m_c = project_weighted_mean_one(r[20:30], w64[20:30])
    rec("T8d r == const -> m == 1 (|m - 1| < 1e-12; float64-normalised w)", float((m_c - 1).abs().max()) < 1e-12,
        max_dev=float((m_c - 1).abs().max()))
    s = torch.randn(N, 4, generator=g)
    gates = torch.rand(N, 4, generator=g) * 0.3
    m0, _ = multipliers(s, gates, 0.0, 0.5)
    rec("T8e lambda = 0 -> m == 1 (|m - 1| < 1e-12; the trainer uses the native path at lambda_eff = 0)",
        float((m0 - 1).abs().max()) < 1e-12, max_dev=float((m0 - 1).abs().max()))
    for lam in (0.25, 0.5, 1.0):
        mm, ww = multipliers(s, gates, lam, 0.5)
        rec(f"T9c lambda {lam}: unclipped range within [e^-2lam, e^2lam]-consistent bounds",
            bool((mm >= 0.5).all() and (mm <= 2).all()) and float(((ww * mm).sum(1) - 1).abs().max()) < 1e-12,
            clip_frac=float(((mm <= 0.5 + 1e-12) | (mm >= 2 - 1e-12)).double().mean()), m_std=float(mm.std()))


def t_stats():
    """LiftStats against a direct numpy computation: EMA, shrinkage, pi-weighted marginal, lift, Kish ESS, resume
    round-trip; no future information (accumulate without commit leaves the table unchanged)."""
    import numpy as np
    import torch
    from csr.sll_lift import EPS, LiftStats, lam_at
    rng = np.random.default_rng(0)
    K, L, E, rho, n0 = 4, 3, 6, 0.9, 2.0
    st = LiftStats(K, L, E, rho, n0, 0.05, torch.device("cpu"))
    N_ = np.zeros((K, L, E))
    D_ = np.zeros(K)
    D2 = np.zeros(K)
    for step in range(7):
        S, T = 5, 40
        sid = torch.as_tensor(rng.integers(-1, S, size=T))
        nonpad = sid >= 0
        q = torch.as_tensor(rng.dirichlet(np.ones(K), size=S))
        ps = [torch.softmax(torch.as_tensor(rng.normal(size=(T, E))), -1) for _ in range(L)]
        before = st.lift().clone()
        st.accumulate(ps, q, sid, nonpad)
        rec_ok = torch.equal(before, st.lift()) if step == 3 else None
        if rec_ok is not None:
            rec("T11a no future information: accumulating the current step leaves the lift table unchanged until commit",
                rec_ok)
        aN = np.zeros((K, L, E))
        aD, aD2 = np.zeros(K), np.zeros(K)
        for s in range(S):
            msk = (sid.numpy() == s)
            if not msk.any():
                continue
            sm = np.stack([p.numpy()[msk].mean(0) for p in ps])  # (L, E)
            aN += q.numpy()[s][:, None, None] * sm[None]
            aD += q.numpy()[s]
            aD2 += q.numpy()[s] ** 2
        N_, D_, D2 = rho * N_ + aN, rho * D_ + aD, rho * rho * D2 + aD2
        st.commit(step)
    pbar = N_.sum(0) / D_.sum()
    Pk = (N_ + n0 * pbar[None]) / (D_ + n0)[:, None, None]
    Pm = ((D_ / D_.sum())[:, None, None] * Pk).sum(0)
    Lref = (np.log(Pk + EPS) - np.log(Pm + EPS)[None]).transpose(1, 0, 2)
    md = float(np.abs(st.lift().numpy() - Lref).max())
    rec("T11b lift table == direct computation (EMA, shrinkage, pi-weighted marginal)", md < 1e-12, max_abs_diff=md)
    ess_ref = D_ ** 2 / D2
    rec("T11c Kish ESS == direct computation", float(np.abs(st.ess() - ess_ref).max()) < 1e-9)
    st2 = LiftStats(K, L, E, rho, n0, 0.05, torch.device("cpu"))
    st2.load(st.state())
    rec("T10a LiftStats state/load round-trip exact", torch.equal(st2.lift(), st.lift()) and
        torch.equal(st2.scale(), st.scale()))
    sch = [lam_at(s, 3880, 1.0) for s in (0, 193, 194, 195, 290, 387, 388, 3879)]
    rec("T11d schedule: 0 on [0,194), linear ramp on [194,388), full from 388 (3,880-step plan)",
        sch[0] == 0 and sch[1] == 0 and abs(sch[2] - 1 / 194) < 1e-12 and abs(sch[5] - 1.0) < 1e-12 and sch[6] == 1.0
        and sch[7] == 1.0, values=sch)
    src = (Path(__file__).resolve().parent / "src/csr/sll_lift.py").read_text()
    rec("T11e lift module reads no domain label / split / evaluation data (static check)",
        all(w not in src for w in ("domain", "calib_", "Split(", "windows_", "edl_beta", "test_")))


def main():
    guard.check_root()
    import torch
    torch.set_num_threads(8)
    t0 = time.time()
    for fn in (t_projection, t_stats, t_lambda0_and_m1, t_forward_and_grads):
        try:
            fn()
        except Exception as e:  # a crashing test is a failing test
            import traceback
            rec(fn.__name__ + " crashed", False, error=traceback.format_exc()[-1500:])
    from csr.slc_paths import ART
    ART.mkdir(parents=True, exist_ok=True)
    out = ART / f"cpu_tests_{guard.utc_stamp()}.json"
    doc = {"time": guard.now(), "seconds": time.time() - t0, "torch": torch.__version__, "n": len(RES),
           "n_fail": sum(not r["ok"] for r in RES), "results": RES}
    guard.atomic_write_json(out, doc, mode=0o400)
    print(json.dumps({"n": doc["n"], "n_fail": doc["n_fail"], "out": str(out)}))
    sys.exit(0 if doc["n_fail"] == 0 else 1)


if __name__ == "__main__":
    main()
