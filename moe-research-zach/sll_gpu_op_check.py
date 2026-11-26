#!/usr/bin/env python
"""csr-semantic-lift-expert-learning-v1 GPU check G2 (prereg section 5): on the reused REF_C6_s100 s20 milestone and
one training micro-batch of the registered plan (step 318, 32 rows), CUDA bf16 autocast exactly as in training:
  (a) native vs LiftExperts with forced m == 1: forward (hidden, top-4, router softmax) bitwise identical;
      every gradient identical (max relative difference reported);
  (b) native vs LiftExperts with real multipliers (lambda 1.0, a lift table from LiftStats accumulated on 20 plan steps):
      forward identical; router / shared / attention / other non-expert gradients and the gradient into each MoE
      expert input identical or within GPU reduction-order tolerance; expert gradients differ; weighted mean 1 and bounds.
  (c) timing of the expert path (native vs operator) on the micro-batch.
Writes artifacts/semantic-lift-expert-learning-v1/gpu_op_check.json (0400)."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402

REF = "runs/20261117T073831Z_semantic_utility_edl_v1_REF_C6_s100"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    if not guard.require_execution(a.execute, "gpu" if a.device == "cuda" else "cpu-model"):
        return
    import torch
    from fle_evaluate import load_milestone
    from csr.fsp_lib import make_batch, to_torch
    from csr.gc_lib import PaperQ
    from csr.lfr_lib import main_stream
    from csr.sll_lift import LiftContext, LiftStats, install, q_tokens
    from csr.sll_paths import ART, TABLE, TABLE_SHA, check_prereg
    check_prereg()
    mon = None
    if a.device == "cuda":
        guard.occupancy_gate("sll gpu op check")
        device = guard.setup_cuda_device()
        mon = guard.OccupancyMonitor(interval=15.0)
        mon.start()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        device = torch.device("cpu")
    guard.register_self("sll gpu op check")
    t_iso, t0 = guard.now(), time.time()
    split, steps, _ = main_stream(245_000_000)
    pq = PaperQ(guard.ROOT / TABLE, split, TABLE_SHA)
    ck = json.loads((guard.ROOT / REF / "status.json").read_text())["ckpts"]["s20"]
    res = {"time": guard.now(), "ref": REF, "ckpt": ck}
    status = "COMPLETE"
    try:
        def fresh():
            m, _ = load_milestone(ck["path"], device)
            m.train()
            return m

        npb = make_batch(steps[318][:32], split)
        bt = to_torch(npb, device)
        minp = {"input_ids": bt["input_ids"], "doc_ids": bt["doc_ids"], "positions": bt["positions"]}
        nonpad = (bt["doc_ids"] >= 0).reshape(-1)
        sid = bt["seq_id"].reshape(-1)
        qs = torch.as_tensor(pq.qseq(npb), device=device)
        g = torch.Generator(device=device).manual_seed(7)
        R = torch.randn(*bt["input_ids"].shape, 384, device=device, generator=g)

        def run(m, ctx=None):
            m.zero_grad(set_to_none=True)
            xin = {}
            hs = []
            li = 0
            for blk in m.blocks:
                if blk.is_moe:
                    def pre(mod, inp, li=li):
                        if inp[0].requires_grad:
                            inp[0].register_hook(lambda gg, li=li: xin.__setitem__(li, gg.detach().clone()))
                    hs.append(blk.ffn.experts.register_forward_pre_hook(pre))
                    li += 1
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = m(minp, {"collect": True})
                loss = (out["hidden"].float() * R).sum() / R.numel() + 0.01 * out["aux"] + 0.001 * out["z"]
            loss.backward()
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - t1
            for h in hs:
                h.remove()
            gr = {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None}
            return out, gr, xin, dt

        def cat(n):
            return ("router" if ".ffn.router." in n else "shared" if ".ffn.shared." in n else "attention" if ".attn." in n
                    else "expert" if ".ffn.experts." in n else "other_nonexpert")

        def cmp(gn, gl):
            d = {}
            for n in gn:
                c = cat(n)
                rel = float((gn[n].float() - gl[n].float()).abs().max() / (gn[n].float().abs().max() + 1e-30))
                eq = bool(torch.equal(gn[n], gl[n]))
                cur = d.setdefault(c, {"max_rel": 0.0, "all_bitwise": True, "n": 0})
                cur["max_rel"] = max(cur["max_rel"], rel)
                cur["all_bitwise"] = cur["all_bitwise"] and eq
                cur["n"] += 1
            return d

        m0 = fresh()
        o_n, g_n, x_n, dt_n = run(m0)
        o_n2, g_n2, _, _ = run(m0)  # native run-to-run reproducibility (GPU reduction order)
        res["native_repeat"] = cmp(g_n, g_n2)
        # (a) forced m == 1
        m1 = fresh()
        ctx = LiftContext(m1.n_moe, pq.K, m1.cfg.n_experts)
        install(m1, ctx)
        ctx.table = torch.zeros(m1.n_moe, pq.K, m1.cfg.n_experts, device=device)
        ctx.scale = [0.02] * m1.n_moe
        ctx.q_tok, ctx.nonpad = q_tokens(qs, sid), nonpad
        ctx.force_m = {l_: torch.ones(nonpad.numel(), 4, device=device, dtype=torch.float64) for l_ in range(m1.n_moe)}
        o_a, g_a, x_a, dt_a = run(m1)
        fa = bool(torch.equal(o_n["hidden"], o_a["hidden"]) and all(
            torch.equal(p["topi"], q["topi"]) and torch.equal(p["p"], q["p"]) for p, q in zip(o_n["records"], o_a["records"])))
        res["a_forced_m1"] = {"forward_bitwise": fa, "grads": cmp(g_n, g_a),
                              "expert_input_grad_max_rel": max(float((x_n[k] - x_a[k]).abs().max() / (x_n[k].abs().max() + 1e-30)) for k in x_n)}
        # (b) real multipliers: lift table from 20 plan steps of statistics (no grad)
        st = LiftStats(pq.K, m1.n_moe, m1.cfg.n_experts, 0.98, 8.0, 0.02, device)
        with torch.no_grad():
            for j in range(20):
                nb = make_batch(steps[300 + j][:32], split)
                b2 = to_torch(nb, device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    o2 = m0({"input_ids": b2["input_ids"], "doc_ids": b2["doc_ids"], "positions": b2["positions"]},
                            {"collect": True})
                st.accumulate([r_["p"].reshape(-1, 32).float() for r_ in o2["records"]],
                              torch.as_tensor(pq.qseq(nb), device=device), b2["seq_id"].reshape(-1),
                              (b2["doc_ids"] >= 0).reshape(-1))
                st.commit()
        m2 = fresh()
        ctx2 = LiftContext(m2.n_moe, pq.K, m2.cfg.n_experts)
        install(m2, ctx2)
        ctx2.table = st.lift().float()
        ctx2.scale = [0.22] * m2.n_moe  # calibration scale of the SEM lift
        ctx2.lam = 1.0
        ctx2.record = True
        ctx2.q_tok, ctx2.nonpad = q_tokens(qs, sid), nonpad
        o_b, g_b, x_b, dt_b = run(m2)
        fb = bool(torch.equal(o_n["hidden"], o_b["hidden"]) and all(
            torch.equal(p["topi"], q["topi"]) for p, q in zip(o_n["records"], o_b["records"])))
        d = ctx2.diag
        res["b_real_m"] = {"forward_bitwise": fb, "grads": cmp(g_n, g_b),
                           "expert_input_grad_max_rel": max(float((x_n[k] - x_b[k]).abs().max() / (x_n[k].abs().max() + 1e-30)) for k in x_n),
                           "wmean_maxdev": max(v["m_wmean_maxdev"] for v in d.values()),
                           "m_min": min(v["m_min"] for v in d.values()), "m_max": max(v["m_max"] for v in d.values()),
                           "m_absdev_wmean_by_layer": [round(d[k]["m_absdev_wmean"], 4) for k in sorted(d)],
                           "clip_frac_by_layer": [round(d[k]["clip_lo"] + d[k]["clip_hi"], 4) for k in sorted(d)]}
        # (c) timing (3 repeats each)
        tn = [run(m0)[3] for _ in range(3)]
        tb = [run(m2)[3] for _ in range(3)]
        res["timing_s_fwd_bwd_32rows"] = {"native": tn, "operator": tb, "ratio_median": sorted(tb)[1] / sorted(tn)[1]}
        tol = max(1e-6, 10 * max(v["max_rel"] for v in res["native_repeat"].values()))
        res["tolerance_rel"] = tol
        ok_a = fa and all(v["max_rel"] <= tol for v in res["a_forced_m1"]["grads"].values()) and \
            res["a_forced_m1"]["expert_input_grad_max_rel"] <= tol
        gb = res["b_real_m"]["grads"]
        ok_b = fb and all(gb[c]["max_rel"] <= tol for c in gb if c != "expert") and gb["expert"]["max_rel"] > 1e-3 and \
            res["b_real_m"]["expert_input_grad_max_rel"] <= tol and res["b_real_m"]["wmean_maxdev"] <= 1e-6 and \
            res["b_real_m"]["m_min"] >= 0.5 - 1e-9 and res["b_real_m"]["m_max"] <= 2.0 + 1e-9
        res["pass_a"], res["pass_b"] = bool(ok_a), bool(ok_b)
        res["pass"] = bool(ok_a and ok_b)
        print(json.dumps(res, default=str)[:4000], flush=True)
    except KeyboardInterrupt:
        status = "EXTERNALLY_STOPPED"
    finally:
        if mon is not None:
            mon.stop()
            if mon.contaminated and status == "COMPLETE":
                status = "EXTERNALLY_STOPPED"
    res["seconds"] = time.time() - t0
    if status == "COMPLETE":
        guard.atomic_write_json(ART / "gpu_op_check.json", res, mode=0o400)
    if a.device == "cuda":
        guard.gpu_ledger(t_iso, guard.now(), "sll_gpu_op_check", "sll gpu op check", status)
        with open(ART / "compute_ledger.tsv", "a") as f:
            f.write(f"{t_iso}\t{guard.now()}\t{time.time() - t0:.1f}\tgpu_op_check\tgpu_op_check\tchecks\t{status}\n")
    sys.exit(0 if status == "COMPLETE" else 77)


if __name__ == "__main__":
    main()
