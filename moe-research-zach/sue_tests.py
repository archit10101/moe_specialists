#!/usr/bin/env python
"""csr-semantic-utility-edl-v1 CPU test suite (fp64 tiny models; no data, no GPU). Writes
artifacts/semantic-utility-edl-v1/cpu_tests_<UTC>.json.
  t1 lean base forward == LFRModel.forward (hidden, top-k)
  t2 restart from the cached post-attention residual without removal == base
  t3 exact counterfactual: batched per-row removals (different expert per row, one pass) == brute-force full
     LFRModel.forward with exclude={l:[e]} and an explicit per-token exclude_mask (tokens of that row selecting e)
  t4 removal semantics: exactly k active experts, removed expert absent on affected tokens, unaffected tokens identical
  t5 TargetBank: recovers planted group-specific usefulness; difficulty matching removes a pure difficulty confound;
     unseen cells are masked (not zero); split-half reliability high with signal, ~0 without
  t6 use_loss: gradient reaches router probabilities only; targets carry no gradient; descent moves group k toward
     experts with high Z_k relative to other groups
  t7 ProbePlan: deterministic, table-independent docs/layers; support-balanced expert choice
  (trainer tests: sue_tests_train.py)"""
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402


def tiny(seed=0):
    import torch
    from csr.lfr_model import LFRConfig, LFRModel
    cfg = LFRConfig(vocab_size=64, n_layer=4, d_model=32, n_head=2, dense_ffn=48, n_experts=8, top_k=4, expert_ffn=16,
                    shared_ffn=24)
    m = LFRModel(cfg, seed).double().eval()
    with torch.no_grad():  # larger router weights so top-k sets are diverse
        for blk in m.blocks[1:]:
            blk.ffn.router.weight.mul_(20.0)
    return m


def batch(seed=0, B=4, T=20):
    import torch
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(2, 64, (B, T), generator=g)
    doc = torch.zeros(B, T, dtype=torch.long)
    pos = torch.arange(T).repeat(B, 1)
    doc[B - 1, T // 2:] = 1  # last row: two documents
    pos[B - 1, T // 2:] = torch.arange(T - T // 2)
    doc[0, T - 3:] = -1  # padding at the end of row 0
    ids[0, T - 3:] = 1
    tgt = torch.roll(ids, -1, 1)
    tgt[:, -1] = -100
    tgt[0, T - 4:] = -100
    tgt[B - 1, T // 2 - 1] = -100
    return {"input_ids": ids, "doc_ids": doc, "positions": pos, "targets": tgt}


def full_ce(model, b, ctl=None, mask=None):
    import torch
    import torch.nn.functional as F
    mi = {k: b[k] for k in ("input_ids", "doc_ids", "positions")}
    if mask is not None:
        mi["exclude_mask"] = mask
    out = model(mi, ctl or {"collect": True})
    h = out["hidden"]
    ce = F.cross_entropy(model.lm_logits(h).reshape(-1, h.shape[-1] * 0 + model.cfg.vocab_size),
                         b["targets"].reshape(-1), ignore_index=-100, reduction="none").view(b["targets"].shape)
    v = (b["targets"] >= 0).double()
    return (ce * v).sum(1) / v.sum(1), out


def t1():
    import torch
    from csr.sue_cf import trunk
    m, b = tiny(), batch()
    _, out = full_ce(m, b)
    r = trunk(m, b["input_ids"], b["doc_ids"], b["positions"], cache=True)
    dh = float((r["hidden"] - out["hidden"]).abs().max())
    same_topi = all(bool(torch.equal(r["topi"][i].view_as(out["records"][i]["topi"]), out["records"][i]["topi"]))
                    for i in range(m.n_moe))
    return {"ok": dh == 0.0 and same_topi, "max_abs_hidden": dh, "topi_equal": same_topi}


def t2():
    from csr.sue_cf import trunk
    m, b = tiny(), batch()
    r = trunk(m, b["input_ids"], b["doc_ids"], b["positions"], cache=True)
    worst = 0.0
    for l_ in range(m.n_moe):
        s = trunk(m, b["input_ids"], b["doc_ids"], b["positions"], start=(l_, r["mids"][l_]))
        worst = max(worst, float((s["hidden"] - r["hidden"]).abs().max()))
    return {"ok": worst == 0.0, "max_abs_hidden": worst}


def t3():
    import torch
    from csr.sue_cf import row_ce, trunk
    m, b = tiny(), batch()
    B, T = b["input_ids"].shape
    r = trunk(m, b["input_ids"], b["doc_ids"], b["positions"], cache=True)
    base_rows, _ = row_ce(m, r["hidden"], b["targets"])
    worst, n_cmp = 0.0, 0
    for l_ in range(m.n_moe):
        topi = r["topi"][l_].view(B, T, -1)
        for rep in range(3):  # several random multi-row assignments per layer
            g = torch.Generator().manual_seed(100 * l_ + rep)
            ke = torch.full((B,), -1, dtype=torch.long)
            for bb in range(B):
                valid = b["doc_ids"][bb] >= 0
                sel = torch.unique(topi[bb][valid])
                ke[bb] = sel[torch.randint(len(sel), (1,), generator=g)]
            cf = trunk(m, b["input_ids"], b["doc_ids"], b["positions"], start=(l_, r["mids"][l_]), kill=(l_, ke))
            cf_rows, _ = row_ce(m, cf["hidden"], b["targets"])
            full_kill = trunk(m, b["input_ids"], b["doc_ids"], b["positions"], kill=(l_, ke))
            fk_rows, _ = row_ce(m, full_kill["hidden"], b["targets"])
            for bb in range(B):
                e = int(ke[bb])
                mask = torch.zeros(B, T, dtype=torch.bool)
                mask[bb] = (topi[bb] == e).any(-1) & (b["doc_ids"][bb] >= 0)
                ref, _ = full_ce(m, b, {"collect": True, "exclude": {l_: [e]}}, mask)
                worst = max(worst, abs(float(cf_rows[bb] - ref[bb])), abs(float(fk_rows[bb] - ref[bb])))
                n_cmp += 1
                # rows without a removal are unchanged
            d_other = float((cf_rows - base_rows)[ke < 0].abs().max()) if bool((ke < 0).any()) else 0.0
            worst = max(worst, d_other)
    return {"ok": worst < 1e-10, "max_abs_ce_diff": worst, "n_comparisons": n_cmp}


def t4():
    import torch
    from csr.sue_cf import _route
    m = tiny()
    moe = m.blocks[1].ffn
    g = torch.Generator().manual_seed(3)
    hf = torch.randn(50, 32, generator=g, dtype=torch.float64)
    g0, t0 = _route(moe, hf, 4, torch.float64)
    ke = torch.full((50,), -1, dtype=torch.long)
    ke[:25] = t0[:25, 0]  # remove each token's top-1 expert for the first 25 tokens
    ke[25:40] = 7 - t0[25:40, 0] % 1 * 0  # possibly unselected expert 7
    g1, t1_ = _route(moe, hf, 4, torch.float64, ke)
    k_ok = t1_.shape[1] == 4 and all(len(set(r.tolist())) == 4 for r in t1_)
    removed = all(int(ke[i]) not in t1_[i].tolist() for i in range(25))
    sel = torch.tensor([(int(ke[i]) >= 0 and int(ke[i]) in t0[i].tolist()) for i in range(50)])
    untouched = bool(torch.equal(t1_[~sel], t0[~sel]) and torch.equal(g1[~sel], g0[~sel]))
    # next-ranked admission: new set = old set minus e plus the 5th-ranked expert
    p = torch.softmax(torch.nn.functional.linear(hf, moe.router.weight), -1)
    t5 = p.topk(5, -1).indices
    nxt = all(set(t1_[i].tolist()) == (set(t5[i].tolist()) - {int(ke[i])}) for i in range(25))
    return {"ok": bool(k_ok and removed and untouched and nxt), "k_ok": k_ok, "removed": removed,
            "untouched": untouched, "next_ranked_admitted": nxt}


def t5():
    import numpy as np
    from csr.sue_cf import TargetBank
    rng = np.random.default_rng(0)
    K, L, E = 4, 2, 6
    a = rng.normal(0, 0.05, (K, L, E))
    gen = rng.normal(0.1, 0.05, (L, E))

    def sim(n, signal, conf):
        tb = TargetBank(K, L, E, rho=1.0, m_min=1.0, n_min=2)
        for _ in range(n):
            k = rng.integers(K)
            q = np.zeros(K)
            q[k] = 1.0
            l_, e = rng.integers(L), rng.integers(E)
            b = (3 if k == 0 else rng.integers(4)) if conf else rng.integers(4)
            u = gen[l_, e] + (0.2 * b if conf else 0.0) + (a[k, l_, e] if signal else 0.0) + rng.normal(0, 0.05)
            tb.add(q[None], [l_], [e], np.array([u]), [b], [rng.integers(2)])
        return tb
    tb = sim(40000, True, False)
    d, se, v = tb.delta()
    ac = a - a.mean(0, keepdims=True)  # delta targets own minus others' mean
    r = float(np.corrcoef(d[v], (a - (a.sum(0, keepdims=True) - a) / (K - 1))[v])[0, 1])
    _, _, diag = tb.targets()
    tbn = sim(40000, False, False)
    _, _, dn = tbn.targets()
    tbc = sim(40000, False, True)  # group 0 always hardest bin; usefulness rises with difficulty; no group effect
    dc, _, vc = tbc.delta()
    conf_removed = float(np.abs(dc[0][vc[0]]).mean())
    empty = TargetBank(K, L, E, rho=1.0)
    _, ve, _ = empty.targets()
    del ac
    ok = (r > 0.95 and diag["split_half_r"] > 0.8 and abs(dn["split_half_r"]) < 0.2 and conf_removed < 0.02
          and not ve.any())
    return {"ok": bool(ok), "recovery_r": r, "split_half_signal": diag["split_half_r"],
            "split_half_null": dn["split_half_r"], "confound_abs_delta_group0": conf_removed,
            "unseen_cells_masked": bool(not ve.any())}


def t6():
    import torch
    from csr.sue_cf import use_loss
    torch.manual_seed(0)
    K, L, E, S, Tn = 3, 2, 6, 6, 5
    logits = torch.randn(S * Tn, E, dtype=torch.float64, requires_grad=True)
    p = torch.softmax(logits, -1)
    sid = torch.arange(S).repeat_interleave(Tn)
    nonpad = torch.ones(S * Tn, dtype=torch.bool)
    q = torch.zeros(S, K, dtype=torch.float64)
    q[torch.arange(S), torch.arange(S) % K] = 1.0
    Z = torch.zeros(K, L, E, dtype=torch.float64)
    for k in range(K):
        Z[k, 1, 2 * k] = 2.0  # expert 2k is disproportionately useful for group k at layer 1
    V = torch.ones(K, L, E, dtype=torch.bool)
    loss, n, marg = use_loss([p], [1], q, sid, nonpad, Z, V, tau=0.2)
    loss.backward()
    g = logits.grad.view(S, Tn, E)
    own_ok = all(float(g[s, :, 2 * (s % K)].mean()) < 0 for s in range(S))  # descent raises own expert logit
    return {"ok": bool(n == K and own_ok and not Z.requires_grad and torch.isfinite(loss)), "loss": float(loss),
            "n_terms": n, "own_direction": own_ok, "margin": marg}


def t7():
    import numpy as np
    from csr.sue_cf import ProbePlan
    pool = np.arange(1000)
    a, b = ProbePlan(pool, 100, 16, 4, range(11)), ProbePlan(pool, 100, 16, 4, range(11))
    same = all(np.array_equal(a.docs(t), b.docs(t)) for t in range(50))
    lay = [a.layer(t, m) for t in range(3) for m in range(4)]
    topi = np.zeros((2, 5, 4), np.int64)
    topi[0] = [0, 1, 2, 3]
    topi[1] = [4, 5, 6, 7]
    sup = np.array([5, 0, 5, 5, 9, 9, 1, 9], float)
    ex = a.experts(3, 1, topi, np.ones((2, 5), bool), sup)
    return {"ok": bool(same and lay[:12] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0] and list(ex) == [1, 6]),
            "deterministic": same, "layers": lay, "balanced_choice": ex.tolist()}


def main():
    guard.check_root()
    res = {}
    for name in ("t1", "t2", "t3", "t4", "t5", "t6", "t7"):
        try:
            res[name] = globals()[name]()
        except Exception:
            res[name] = {"ok": False, "error": traceback.format_exc()[-2000:]}
        print(name, json.dumps(res[name], default=str)[:400], flush=True)
    res["all_ok"] = all(v["ok"] for v in res.values() if isinstance(v, dict))
    if "--write" in sys.argv:
        from csr.sue_paths import ART
        ART.mkdir(parents=True, exist_ok=True)
        p = ART / f"cpu_tests_{guard.utc_stamp()}.json"
        guard.atomic_write_json(p, {"time": guard.now(), **res}, mode=0o400)
        print(p)
    print("all_ok", res["all_ok"])
    sys.exit(0 if res["all_ok"] else 1)


if __name__ == "__main__":
    main()
