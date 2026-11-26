#!/usr/bin/env python
"""scrgp-v1 CPU correctness tests (prereg section 4). Tiny random LFR models in float64 unless stated; real CALIB-SEARCH
half-A rows only where data are needed (B is never touched). Writes artifacts/scrgp-v1/cpu_tests_<UTC>.json (0400).
  T-NAT    probe natural forward == LFRModel.forward (CE, float64)
  T-INS    forced 4th-slot insertion == explicit Experts.forward with the edited top-4 (top-3 kept, slot-4 gate = p_(4),
           shared / attention / later layers natural); rank-4 tokens unchanged; e-in-top-3 tokens unchanged
  T-OFF    probes disabled: forced(theta) with e = 4th expert everywhere-eligible reproduces natural; natural_updated
           with theta' = theta is bit-identical to natural
  T-GRAD   expert_grads == central finite differences of the forced weighted loss (float64), for 3 weight vectors
  T-ISO    gradient isolation: no model parameter requires / receives grad; only theta_e leaves are differentiated;
           changing another expert's weights in layer l does not change the returned gradient
  T-BATCH  J weight vectors in one call == J separate calls; linearity (grad(aW1 + bW2) = a g1 + b g2)
  T-PROJ   projection orthogonality (|cos(u, g)| < 1e-12), u = d when d is orthogonal to g, homogeneity
  T-SOFT   soft cluster weights: one-hot q reproduces hard groups; soft weights proportional to q_t(k*)
  T-GATE   matching gives identical weighted stratum (incl. gate-tercile) distributions for positive and control
  T-ZERO   identical positive / control token distributions -> d = 0 exactly -> u = 0
  T-NORM   no per-signal norm amplification: u scales with the weights; the virtual update is eta * u with one global
           eta (relative update sizes of two pairs preserved)
  T-STRAT  strata ranges; ineligibility of top-3 tokens; conditional strata only on exposed tokens
  T-RESUME scrgp_stage1.py est (test pairs, 2 micro-batches, real C6_s0 weights, CPU float32): an interrupted run
           (leg exit after the first micro-batch of every pass) resumed from its slot == an uninterrupted run (exact)
  T-EVAL   scrgp_stage1.py eval end-to-end on CPU (test pairs, tiny rows): completes, model hash restored
           (values not inspected)"""
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402

RES = []


def rec(name, ok, **kw):
    RES.append({"test": name, "ok": bool(ok), **kw})
    print(name, ok, json.dumps(kw, default=str)[:600], flush=True)


def tiny(seed=0):
    import torch
    from csr.lfr_model import LFRConfig, LFRModel
    torch.manual_seed(seed)
    cfg = {"n_layer": 3, "d_model": 32, "n_head": 2, "dense_ffn": 64, "n_experts": 8, "expert_ffn": 16,
           "shared_ffn": 32, "vocab_size": 50304, "top_k": 4, "router": "linear", "teacher": False}
    m = LFRModel(LFRConfig.from_dict({**LFRConfig().to_dict(), **cfg}), seed).double()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "router" in n:
                p.normal_(0, 0.5)  # spread router probabilities
    return m.eval()


def data(n_rows=3):
    from csr.fsp_lib import Split, pack_rows
    from csr.lfr_eval import paper_half
    from csr.scrgp_lib import make_batch, to_dev, u_hash
    import torch
    cs = Split("calib_search")
    docs = sorted(paper_half(cs, 0), key=lambda i: u_hash("scrgp-test", f"{cs.papers[i]}|{i}"))
    rows = pack_rows(cs.lengths, docs)[:n_rows]
    bt = to_dev(make_batch(rows, cs), torch.device("cpu"))
    bt["input_ids"] = bt["input_ids"] % 50304
    return bt


def t_nat_ins_off():
    import torch
    from csr.scrgp_lib import Probe
    m = tiny()
    bt = data(2)
    pr = Probe(m, torch.device("cpu"))
    ce, probs, cache, ctx = pr.natural(bt, keep_cache=True)
    out = m({"input_ids": bt["input_ids"], "doc_ids": bt["doc_ids"], "positions": bt["positions"]})
    ref = pr.ce_of(out["hidden"], bt["targets"])
    v = bt["targets"].reshape(-1) >= 0
    nat_err = float((ce - ref)[v].abs().max())
    rec("T-NAT probe natural == model.forward", nat_err < 1e-10, max_abs=nat_err, bitwise=bool(torch.equal(ce[v], ref[v])))
    # forced insertion vs explicit Experts.forward with edited top-4
    shape = tuple(bt["input_ids"].shape) + (m.cfg.d_model,)
    B, T, D = shape
    errs, unchanged_top3, unchanged_r4 = [], [], []
    for li, e in ((0, 3), (1, 6)):
        c = cache[li]
        w1, w2 = m.blocks[li + 1].ffn.experts.w1[e], m.blocks[li + 1].ffn.experts.w2[e]
        with torch.no_grad():
            hf = pr.forced_hidden(li, e, c, ctx, shape, w1, w2)
            topi = c["topi"].clone()
            gates = c["gates"].clone()
            top3 = (topi[:, :3] == e).any(1)
            topi[~top3, 3] = e
            moe = m.blocks[li + 1].ffn
            y = moe.experts(c["hf"], topi, gates) + moe.shared(c["hf"])
            x = c["x_mid"] + y.view(B, T, -1)
            href = pr._run(x, ctx, li + 2)
            errs.append(float((hf - href).abs().max()))
            # rank-4 tokens / e-in-top-3 tokens: layer-li output identical to natural
            sl = c["slots"]
            ynat = sl.sum(1) + c["sh"]
            s4 = torch.where(top3.unsqueeze(1), sl[:, 3],
                             (pr._expert_out(c["hf"], w1, w2) * c["gates"][:, 3:4].double()).to(sl.dtype))
            yf = torch.cat([sl[:, :3], s4.unsqueeze(1)], 1).sum(1) + c["sh"]
            r4 = c["topi"][:, 3] == e
            unchanged_top3.append(float((yf - ynat)[top3].abs().max()) if top3.any() else 0.0)
            unchanged_r4.append(float((yf - ynat)[r4].abs().max()) if r4.any() else 0.0)
    rec("T-INS forced 4th-slot insertion", max(errs) < 1e-10 and max(unchanged_top3) == 0 and max(unchanged_r4) < 1e-12,
        max_abs=max(errs), top3_unchanged=max(unchanged_top3), rank4_unchanged=max(unchanged_r4))
    # probes disabled
    li, e = 1, 2
    c = cache[li]
    w1, w2 = m.blocks[li + 1].ffn.experts.w1[e], m.blocks[li + 1].ffn.experts.w2[e]
    with torch.no_grad():
        h_nat = pr._run(c["x_mid"] + (c["slots"].sum(1) + c["sh"]).view(B, T, -1), ctx, li + 2)
        h_upd = pr.natural_updated_hidden(li, e, c, ctx, shape, w1, w2, w1.clone(), w2.clone())
        f0 = pr.forced_hidden(li, e, c, ctx, shape, w1, w2)
        f1 = pr.forced_hidden(li, e, c, ctx, shape, w1.clone(), w2.clone())
        ce_nat2 = pr.ce_of(h_nat, bt["targets"])
    rec("T-OFF probes disabled == natural (bitwise)", bool(torch.equal(h_nat, h_upd)) and bool(torch.equal(f0, f1))
        and bool(torch.equal(ce_nat2, ce)))


def t_grad_iso_batch():
    import torch
    from csr.scrgp_lib import Probe
    m = tiny(1)
    bt = data(2)
    pr = Probe(m, torch.device("cpu"))
    ce, probs, cache, ctx = pr.natural(bt, keep_cache=True)
    shape = tuple(bt["input_ids"].shape) + (m.cfg.d_model,)
    N = ce.numel()
    g = torch.Generator().manual_seed(0)
    W = torch.rand(3, N, generator=g, dtype=torch.float64) * (bt["targets"].reshape(-1) >= 0)
    W[1] -= W[1].mean()
    rel_all = {}
    for li, e, eps in ((1, 5, 1e-5), (0, 3, 1e-7)):  # li = 1: last MoE layer (no downstream discrete routing)
        G = pr.expert_grads(li, e, cache[li], ctx, shape, bt["targets"], W).double()
        ex = m.blocks[li + 1].ffn.experts
        w1, w2 = ex.w1[e].detach().clone(), ex.w2[e].detach().clone()
        d, f = w1.shape
        rel = []
        for j in range(3):
            v = torch.randn(d * f + f * d, generator=g, dtype=torch.float64)
            v1, v2 = v[:d * f].view(d, f), v[d * f:].view(f, d)
            with torch.no_grad():
                lp = float((pr.ce_of(pr.forced_hidden(li, e, cache[li], ctx, shape, w1 + eps * v1, w2 + eps * v2),
                                     bt["targets"]) * W[j]).sum())
                lm = float((pr.ce_of(pr.forced_hidden(li, e, cache[li], ctx, shape, w1 - eps * v1, w2 - eps * v2),
                                     bt["targets"]) * W[j]).sum())
            fd = (lp - lm) / (2 * eps)
            rel.append(abs(fd - float((G[j] * v).sum())) / max(1e-12, abs(fd)))
        rel_all[f"l{li}"] = rel
    rec("T-GRAD exact gradient vs finite differences (fp32 output)", max(rel_all["l1"]) < 1e-5 and
        max(rel_all["l0"]) < 1e-3, rel_err=rel_all)
    li, e = 0, 5
    G = pr.expert_grads(li, e, cache[li], ctx, shape, bt["targets"], W).double()
    ex = m.blocks[li + 1].ffn.experts
    any_grad = any(p.requires_grad or p.grad is not None for p in m.parameters())
    with torch.no_grad():
        ex.w1[(e + 1) % 8].add_(0.3)  # change ANOTHER expert of the same layer: cached top-3/slot-4 outputs unchanged
    G2 = pr.expert_grads(li, e, cache[li], ctx, shape, bt["targets"], W).double()
    rec("T-ISO gradient isolation", not any_grad and torch.equal(G, G2) and
        not any(p.grad is not None for p in m.parameters()), any_param_grad=any_grad)
    Gs = torch.stack([pr.expert_grads(li, e, cache[li], ctx, shape, bt["targets"], W[j:j + 1])[0].double()
                      for j in range(3)])
    Gl = pr.expert_grads(li, e, cache[li], ctx, shape, bt["targets"], (2.0 * W[0] - 0.5 * W[2]).unsqueeze(0))[0].double()
    lin = float((Gl - (2.0 * G[0] - 0.5 * G[2])).abs().max() / G.abs().max())
    rec("T-BATCH multi-vector == separate; linearity", float((Gs - G).abs().max()) <= 1e-12 * float(G.abs().max())
        and lin < 1e-6, sep_err=float((Gs - G).abs().max()), lin_rel=lin)


def t_proj_soft_gate_zero_norm():
    import numpy as np
    import torch
    from csr.scrgp_lib import N_STRATA, cos, pair_weights, project, token_q
    g = torch.Generator().manual_seed(3)
    d, gg = torch.randn(50, 1000, generator=g, dtype=torch.float64), torch.randn(50, 1000, generator=g, dtype=torch.float64)
    u = project(d, gg)
    orth = float(cos(u, gg).abs().max())
    dp = project(d, gg)  # d orthogonal to g -> u = d
    u2 = project(dp, gg)
    hom = float((project(3.7 * d, gg) - 3.7 * u).abs().max())
    rec("T-PROJ projection", orth < 1e-12 and float((u2 - dp).abs().max()) < 1e-12 and hom < 1e-12, max_cos=orth,
        homog=hom)
    # soft weights
    rng = np.random.default_rng(0)
    N, K = 5000, 4
    stratum = torch.as_tensor(rng.integers(0, 30, N))
    stratum[:100] = -1
    hard = rng.integers(0, K, N)
    qh = torch.as_tensor(np.eye(K)[hard])
    P = torch.zeros(N_STRATA, dtype=torch.float64).index_add_(0, stratum[stratum >= 0], qh[stratum >= 0, 1])
    C = torch.zeros(N_STRATA, dtype=torch.float64).index_add_(0, stratum[stratum >= 0], 1 - qh[stratum >= 0, 1])
    ratio = torch.where((P > 0) & (C > 0), P / C.clamp(min=1e-300), torch.zeros_like(P))
    ptot = float((P * (ratio > 0)).sum())
    mk = torch.zeros(K, dtype=torch.float64)
    mk += (qh[stratum >= 0]).sum(0)
    wp, wm, wg = pair_weights(qh, stratum, 1, ratio, ptot, mk)
    ok_hard = bool(((wp > 0) == (torch.as_tensor(hard == 1) & (stratum >= 0) & (ratio[stratum.clamp(min=0)] > 0))).all())
    ok_sum = abs(float(wp.sum()) - 1) < 1e-12 and abs(float(wm.sum()) - 1) < 1e-12 and abs(float(wg.sum()) - 1) < 1e-12
    hp = torch.zeros(N_STRATA, dtype=torch.float64).index_add_(0, stratum.clamp(min=0), wp)
    hm = torch.zeros(N_STRATA, dtype=torch.float64).index_add_(0, stratum.clamp(min=0), wm)
    gate_eq = float((hp - hm).abs().max())
    qs = torch.as_tensor(rng.dirichlet(np.ones(K) * 0.5, N))
    P2 = torch.zeros(N_STRATA, dtype=torch.float64).index_add_(0, stratum.clamp(min=0), qs[:, 2] * (stratum >= 0))
    C2 = torch.zeros(N_STRATA, dtype=torch.float64).index_add_(0, stratum.clamp(min=0), (1 - qs[:, 2]) * (stratum >= 0))
    r2 = torch.where((P2 > 0) & (C2 > 0), P2 / C2.clamp(min=1e-300), torch.zeros_like(P2))
    wp2, wm2, _ = pair_weights(qs, stratum, 2, r2, float((P2 * (r2 > 0)).sum()), qs.sum(0))
    okm = (stratum >= 0)
    prop = float((wp2[okm] / qs[okm, 2] - wp2[okm][0] / qs[okm, 2][0]).abs().max())
    rec("T-SOFT soft cluster weighting", ok_hard and ok_sum and prop < 1e-12, hard_groups=ok_hard, sums=ok_sum,
        soft_prop=prop)
    rec("T-GATE matched stratum (gate-tercile) distributions identical", gate_eq < 1e-12, max_abs=gate_eq)
    # zero update for identical distributions: q = 0.5 everywhere -> ratio 1 -> w+ == w-
    qz = torch.full((N, K), 0.25, dtype=torch.float64)
    qz[:, 1] = 0.5
    qz[:, 0] = 0.0
    Pz = torch.zeros(N_STRATA, dtype=torch.float64).index_add_(0, stratum.clamp(min=0), 0.5 * (stratum >= 0).double())
    rz = torch.where(Pz > 0, torch.ones_like(Pz), torch.zeros_like(Pz))
    wpz, wmz, _ = pair_weights(qz, stratum, 1, rz, float(Pz.sum()), qz.sum(0))
    dz = (wpz - wmz)
    uz = project(dz.unsqueeze(0) @ torch.randn(N, 64, generator=g, dtype=torch.float64), gg[:1, :64])
    rec("T-ZERO identical pos/control -> d = u = 0", float(dz.abs().max()) == 0.0 and float(uz.abs().max()) == 0.0)
    # no per-signal amplification: two pairs, global eta; the update keeps the norm ratio of u
    ua, ub = torch.randn(1000, generator=g, dtype=torch.float64), 0.01 * torch.randn(1000, generator=g, dtype=torch.float64)
    eta = 0.002 / float(torch.stack([ua.pow(2).sum(), ub.pow(2).sum()]).median())
    ra = float((eta * ub).norm() / (eta * ua).norm())
    rec("T-NORM no per-signal norm amplification", abs(ra - float(ub.norm() / ua.norm())) < 1e-12, update_ratio=ra)


def t_strata():
    import numpy as np
    import torch
    from csr.scrgp_lib import Cuts, N_COND, N_STRATA, cond_strata_all, ranks_of, strata_all
    g = torch.Generator().manual_seed(5)
    N, E, L = 4000, 32, 2
    p = torch.softmax(torch.randn(N, E, generator=g) * 2, -1)
    ce = torch.rand(N, generator=g) * 6
    pos = torch.randint(0, 1024, (N,), generator=g)
    tgt = torch.randint(-1, 100, (N,), generator=g)
    cuts = Cuts(np.array([1.0, 2, 3, 4]), np.array([[0.02, 0.05]] * L), np.array([[0.5, 1.5]] * L),
                np.random.default_rng(0).integers(0, 3, 50304))
    s = strata_all(p, ce, pos, tgt, 0, cuts, torch.device("cpu"))
    r = ranks_of(p)
    ok1 = bool(((s == -1) == ((r < 3) | (tgt < 0).unsqueeze(1))).all()) and int(s.max()) < N_STRATA
    cd = {"ce": np.linspace(0.5, 5.5, 9), "gate": np.tile(np.linspace(0.02, 0.3, 9), (L, 1)),
          "marg": np.tile(np.linspace(0.0, 0.2, 9), (L, 1))}
    sc = cond_strata_all(p, ce, pos, tgt, 1, cd, torch.device("cpu"))
    ok2 = bool(((sc == -1) == ((r > 3) | (tgt < 0).unsqueeze(1))).all()) and int(sc.max()) < N_COND
    rec("T-STRAT strata", ok1 and ok2, forced=ok1, cond=ok2)


def run_stage1(args, out, max_seconds=None):
    env = {**os.environ, "MOE_RESEARCH_APPROVE": "cpu-model", "CUDA_VISIBLE_DEVICES": ""}
    cmd = [str(guard.ROOT / ".venv/bin/python"), "scrgp_stage1.py", *args, "--device", "cpu", "--test-pairs",
           "--rows-limit", "2", "--mb-rows", "1", "--out-dir", str(out), "--execute"]
    if max_seconds is not None:
        cmd += ["--max-seconds", str(max_seconds)]
    return subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=guard.ROOT)


def t_resume_eval():
    import torch
    from csr.scrgp_paths import TMP
    base = TMP / "test_resume"
    shutil.rmtree(base, ignore_errors=True)
    a, b = base / "full", base / "legs"
    t0 = time.time()
    r = run_stage1(["--phase", "est", "--seed", "0"], a)
    legs, rcs = 0, []
    while True:
        rr = run_stage1(["--phase", "est", "--seed", "0"], b, max_seconds=0)
        rcs.append(rr.returncode)
        legs += 1
        if rr.returncode != 75 or legs > 40:
            break
    ok = r.returncode == 0 and rcs[-1] == 0 and legs > 3
    if ok:
        ua = torch.load(a / "est_s0.pt", weights_only=False)["u"]
        ub = torch.load(b / "est_s0.pt", weights_only=False)["u"]
        same = all(torch.equal(ua[k], ub[k]) for k in ua)
        nz = float(sum(ua[k].abs().sum() for k in ua))
    else:
        same, nz = False, 0.0
    rec("T-RESUME interrupted + resumed == uninterrupted (exact)", ok and same and nz > 0, legs=legs, rc_full=r.returncode,
        rcs=rcs[-3:], nonzero=nz > 0, seconds=time.time() - t0, err=(r.stderr[-1500:] if r.returncode else ""))
    re = run_stage1(["--phase", "eval", "--seed", "0"], a)
    j = (a / "eval_s0.json")
    okj = re.returncode == 0 and j.exists() and json.loads(j.read_text()).get("hash_restored") is True
    rec("T-EVAL eval end-to-end, model hash restored (values not inspected)", okj, rc=re.returncode,
        err=re.stderr[-1500:] if re.returncode else "")
    for p in base.rglob("*"):
        if p.is_file():
            os.chmod(p, 0o600)
    shutil.rmtree(base, ignore_errors=True)


def main():
    guard.check_root()
    import torch
    torch.set_num_threads(16)
    from csr.scrgp_paths import ART
    names = [x for x in sys.argv[1:] if x.startswith("t_")] or \
        ["t_nat_ins_off", "t_grad_iso_batch", "t_proj_soft_gate_zero_norm", "t_strata", "t_resume_eval"]
    t0 = time.time()
    for n in names:
        try:
            globals()[n]()
        except Exception:
            rec(n + " crashed", False, error=traceback.format_exc()[-2500:])
    doc = {"time": guard.now(), "seconds": time.time() - t0, "n": len(RES), "n_fail": sum(not r["ok"] for r in RES),
           "results": RES}
    out = ART / f"cpu_tests_{'_'.join(x[2:] for x in names)}_{guard.utc_stamp()}.json"
    guard.atomic_write_json(out, doc, mode=0o400)
    print(json.dumps({"n": doc["n"], "n_fail": doc["n_fail"], "out": str(out)}))
    sys.exit(0 if doc["n_fail"] == 0 else 1)


if __name__ == "__main__":
    main()
