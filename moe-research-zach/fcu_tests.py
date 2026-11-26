#!/usr/bin/env python
"""csr-forecast-counterfactual-utility-v1: validation tests of the utility engine (prereg 2.2 V1-V5).

  .venv/bin/python fcu_tests.py            CPU (CUDA hidden): V1 hand case vs the model's own exclude path (float64),
                                           V2 identical-compute replacement, V5 null intervention, probe helpers
  scripts/gpu_guarded.sh "fcu tests gpu" fcu_tests.py --gpu --execute
                                           V3 deterministic replay and V4 engine == full forward on the real
                                           frozen TOKEN-4 s0 checkpoint (fp32, TF32 off)
Writes artifacts/forecast-counterfactual-utility-v1/tests_{cpu,gpu}_<stamp>.json."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402


def tiny_model(seed=0):
    import torch
    from csr.lfr_model import LFRConfig, LFRModel
    cfg = LFRConfig(vocab_size=64, n_layer=4, d_model=32, n_head=2, dense_ffn=64, n_experts=8, top_k=4,
                    expert_ffn=16, shared_ffn=32)
    m = LFRModel(cfg, seed).double().eval()
    with torch.no_grad():  # larger router weights so routing is decisive and experts differ
        for blk in m.blocks[1:]:
            blk.ffn.router.weight.mul_(20.0)
            blk.ffn.experts.w1.mul_(5.0)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def ref_utility(m, ids, ends, s, mi, e):
    """Reference: the model's own forward with ctl exclude restricted (exclude_mask) to region tokens where e is
    in the normal top-4 at layer mi; CE over region targets."""
    import torch
    B, T = ids.shape
    pos = torch.arange(T).expand(B, T)
    doc = torch.where(pos < ends.view(-1, 1), torch.zeros_like(pos), torch.full_like(pos, -1))
    batch = {"input_ids": ids, "doc_ids": doc, "positions": pos}
    o = m(batch, {"collect": True})
    topi = o["records"][mi]["topi"]
    region = (pos >= s) & (pos < ends.view(-1, 1))
    em = (topi == e).any(-1) & region
    o2 = m({**batch, "exclude_mask": em}, {"exclude": {mi: [e]}})

    def ce(hid):
        lg = m.lm_logits(hid)[:, s:T - 1]
        t = ids[:, s + 1:T]
        c = torch.logsumexp(lg, -1) - lg.gather(-1, t.unsqueeze(-1)).squeeze(-1)
        v = (s + 1 + torch.arange(T - 1 - s).view(1, -1)) < ends.view(-1, 1)
        return (c * v).sum(1) / v.sum(1), c
    a, ca = ce(o["hidden"])
    b, cb = ce(o2["hidden"])
    return (b - a).numpy(), em, a.numpy()


def cpu_tests():
    import numpy as np
    import torch
    from csr.fcu_lib import UtilityEngine, auroc, moe_ffn, ridge_cv, ridge_fit, ridge_pred
    res = {}
    torch.manual_seed(0)
    m = tiny_model(1)
    s, R, B = 16, 16, 5
    ids = torch.randint(2, 64, (B, s + R))
    ends = torch.tensor([s + R, s + R, s + 12, s + R, s + 9])
    eng = UtilityEngine(m, layers=[1, 2])
    st = eng.normal(ids, s, ends)
    out = eng.utilities(st, max_pairs=7)
    # V1: every cell vs the reference implementation
    worst = 0.0
    nact = 0
    for li, mi in enumerate([1, 2]):
        for e in range(8):
            du, em, base = ref_utility(m, ids, ends, s, mi, e)
            for r in range(B):
                if out["nsel"][r, li, e] > 0:
                    nact += 1
                    worst = max(worst, abs(float(out["u"][r, li, e]) - float(du[r])))
                else:
                    worst = max(worst, abs(float(du[r])))  # inactive: reference must also be 0
                assert int(em[r].sum()) == int(out["nsel"][r, li, e])
    res["V1_max_abs_diff"] = worst
    res["V1_active_cells"] = nact
    assert nact > 20 and worst <= 1e-5, worst
    # engine normal CE == model forward CE
    _, _, base = ref_utility(m, ids, ends, s, 1, 0)
    res["V4_cpu_normal_ce_diff"] = float(np.abs(out["ce_full"] - base).max())
    assert res["V4_cpu_normal_ce_diff"] <= 1e-9
    # V2: identical compute and replacement rule, one token
    blk = m.blocks[2]
    h = blk.norm2(st["xmid"][1][0:1, 3])  # (1,d) region token 3 of row 0 at MoE layer 1
    _, p, topi = moe_ffn(blk.ffn, h, 4)
    e = int(topi[0, 1])
    kill = torch.zeros(1, 8, dtype=torch.bool)
    kill[0, e] = True
    _, p2, topi2 = moe_ffn(blk.ffn, h, 4, kill)
    rank5 = int(torch.argsort(p[0], descending=True)[4])
    old, new = set(topi[0].tolist()), set(topi2[0].tolist())
    assert len(new) == 4 and new == (old - {e}) | {rank5}
    g_expect = p[0] / (1 - p[0, e])
    for j in new:
        assert abs(float(p2[0, j] - g_expect[j])) < 1e-12
    res["V2"] = {"replaced": e, "by": rank5, "k_active": len(new), "gate_rule": "p/(1-p_e) exact"}
    # V5: null intervention (an expert never selected in a row's region) -> identical CE
    ends_n = ends.clone()
    ends_n[0] = s + 1  # one region token -> 4 of 8 experts inactive in row 0
    st = eng.normal(ids, s, ends_n)
    tp = st["topi"][1]
    found = None
    for r in range(B):
        for e in range(8):
            if not bool((tp[r][st["rvalid"][r]] == e).any()):
                found = (r, e)
                break
        if found:
            break
    if found is None:
        res["V5"] = "no inactive cell in tiny case (checked on GPU)"
    else:
        base_ce, _ = eng.tail(st, 1, [found[0]], [-1])
        cf_ce, sel = eng.tail(st, 1, [found[0]], [found[1]])
        assert int(sel.sum()) == 0
        res["V5_null_max_abs"] = float((cf_ce - base_ce).abs().max())
        assert res["V5_null_max_abs"] == 0.0
    # probe helpers
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 10))
    Y = X @ rng.normal(size=(10, 3)) + 0.1 * rng.normal(size=(400, 3))
    a, _ = ridge_cv(X, Y, np.arange(400) // 4)
    W = ridge_fit(X, Y, a)
    r2 = 1 - ((ridge_pred(W, X) - Y) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()
    assert r2 > 0.99
    sc = np.array([0.1, 0.4, 0.35, 0.8])
    assert abs(auroc(sc, np.array([0, 0, 1, 1])) - 0.75) < 1e-12
    res["probe_helpers"] = "ok"
    return res


def gpu_tests():
    import numpy as np
    import torch
    from csr.fcu_lib import UtilityEngine, build_rows, load_model, regions
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = guard.setup_cuda_device()
    m = load_model("T4_s0", dev)
    reg, sp, _ = regions()
    res = {}
    sel = np.nonzero((reg["s"] == 512) & (reg["half"] == 0))[0][:6]
    s = 512
    ids = torch.as_tensor(build_rows(sp, reg["gdoc"][sel], s), device=dev)
    ends = torch.as_tensor(reg["end"][sel], device=dev)
    eng = UtilityEngine(m, layers=[7])
    st1 = eng.normal(ids, s, ends)
    o1 = eng.utilities(st1, max_pairs=96)
    st2 = eng.normal(ids, s, ends)
    o2 = eng.utilities(st2, max_pairs=96)
    res["V3_bitwise_identical"] = bool(np.array_equal(o1["u"], o2["u"]) and np.array_equal(o1["tokd"], o2["tokd"]))
    res["V3_cells"] = int((o1["nsel"] > 0).sum())
    assert res["V3_bitwise_identical"] and res["V3_cells"] >= 128
    # V4: engine normal CE vs the model's own forward (fp32) and cached-prefix CF vs full-sequence CF
    B, T = ids.shape
    pos = torch.arange(T, device=dev).expand(B, T)
    doc = torch.where(pos < ends.view(-1, 1), torch.zeros_like(pos), torch.full_like(pos, -1))
    batch = {"input_ids": ids, "doc_ids": doc, "positions": pos}
    with torch.no_grad():
        o = m(batch, {"collect": True})
        lg = m.lm_logits(o["hidden"])[:, s:T - 1].float()
        t = ids[:, s + 1:T]
        c = torch.logsumexp(lg, -1) - lg.gather(-1, t.unsqueeze(-1)).squeeze(-1)
        v = (s + 1 + torch.arange(T - 1 - s, device=dev).view(1, -1)) < ends.view(-1, 1)
        base = ((c * v).sum(1) / v.sum(1)).cpu().numpy()
        res["V4_normal_ce_max_abs"] = float(np.abs(base - o1["ce_full"]).max())
        res["V4_null_tail_vs_full_max_abs"] = float(np.abs(base - o1["ce_null"]).max())
        topi = o["records"][7]["topi"]
        region = (pos >= s) & (pos < ends.view(-1, 1))
        diffs = []
        rng = np.random.default_rng(0)
        for _ in range(8):
            r = int(rng.integers(0, B))
            act = np.nonzero(o1["nsel"][r, 0] > 0)[0]
            e = int(rng.choice(act))
            em = (topi == e).any(-1) & region
            o2_ = m({**batch, "exclude_mask": em}, {"exclude": {7: [e]}})
            lg2 = m.lm_logits(o2_["hidden"])[:, s:T - 1].float()
            c2 = torch.logsumexp(lg2, -1) - lg2.gather(-1, t.unsqueeze(-1)).squeeze(-1)
            full_u = float((((c2 - c) * v).sum(1) / v.sum(1))[r])
            diffs.append(abs(full_u - float(o1["u"][r, 0, e])))
        res["V4_cf_engine_vs_full_max_abs"] = float(max(diffs))
    assert res["V4_normal_ce_max_abs"] <= 1e-4 and res["V4_null_tail_vs_full_max_abs"] <= 1e-4
    assert res["V4_cf_engine_vs_full_max_abs"] <= 1e-4
    # V5 on the real model: an inactive expert gives zero change
    r = 0
    inact = np.nonzero(o1["nsel"][r, 0] == 0)[0]
    if len(inact):
        b0, _ = eng.tail(st1, 7, [r], [-1])
        b1, sl = eng.tail(st1, 7, [r], [int(inact[0])])
        res["V5_gpu_null_max_abs"] = float((b1 - b0).abs().max())
        assert int(sl.sum()) == 0 and res["V5_gpu_null_max_abs"] == 0.0
    else:
        res["V5_gpu_null"] = "every expert active in row 0 (null checked on CPU)"
    res["u_summary_layer7"] = {"mean": float(o1["u"][o1["nsel"] > 0].mean()),
                               "abs_mean": float(np.abs(o1["u"][o1["nsel"] > 0]).mean()),
                               "frac_pos": float((o1["u"][o1["nsel"] > 0] > 0).mean())}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    from csr.fcu_lib import ART
    t0 = time.time()
    if a.gpu:
        if not guard.require_execution(a.execute, "gpu"):
            return
        guard.occupancy_gate("fcu tests gpu")
        guard.register_self("fcu tests gpu")
        res = gpu_tests()
        kind = "gpu"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        res = cpu_tests()
        kind = "cpu"
    res["seconds"] = time.time() - t0
    res["time"] = guard.now()
    res["status"] = "PASS"
    p = ART / f"tests_{kind}_{guard.utc_stamp()}.json"
    guard.atomic_write_json(p, res)
    if kind == "gpu":
        with open(ART / "compute_ledger.tsv", "a") as f:
            f.write(f"{guard.now()}\tfcu_tests_gpu\tchecks\t{res['seconds']:.1f}\tCOMPLETE\t\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
