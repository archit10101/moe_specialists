#!/usr/bin/env python
"""csr-multiscale-routing-v1 focused GPU correctness checks (guarded launcher; no training).

1. Tiny model in float64 on GPU: zero coefficients == TOKEN (exact), future invariance, incremental == full,
   scale-control ranking/RMS properties, for every condition.
2. Full-size model (rep-v1 architecture, random init seed 0) on 4 real calib rows per condition (alpha = beta = 0.5):
   fp32 forward + backward (finite loss and gradients, exactly 4 distinct experts), bf16-autocast forward (finite),
   linear-equivalence error |W mean(h) - mean(W h)| at contextual layers.
Writes artifacts/multiscale-routing-v1/gpu_checks.json (never overwritten). Exit 0 iff all checks pass.
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE / "tests"))
from csr import guard  # noqa: E402

OUT = guard.ROOT / "artifacts" / "multiscale-routing-v1" / f"gpu_checks_{guard.utc_stamp()}.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    if OUT.exists():
        raise guard.GuardError(f"{OUT} exists")
    print("msv1 GPU correctness checks")
    if not guard.require_execution(a.execute, "gpu"):
        return
    guard.occupancy_gate("msv1 gpu checks")
    dev = guard.setup_cuda_device()
    mon = guard.OccupancyMonitor(10.0)
    mon.start()
    guard.register_self("msv1 gpu checks")
    t0iso, t0 = guard.now(), time.time()
    lim = guard.RunLimits(30 * 60, monitor=mon)
    status, reason, res = "INCOMPLETE", "", {}
    try:
        res = run(dev, lim)
        status = "COMPLETE"
    except guard.GuardError as e:
        status, reason = ("CONTAMINATED" if "CONTAMINATED" in str(e) else "ABORTED"), str(e)
    except Exception as e:  # noqa: BLE001
        status, reason = "FAILED", f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        lim.close()
        mon.stop()
        if mon.contaminated and status == "COMPLETE":
            status, reason = "CONTAMINATED", mon.detail
        res.update(status=status, reason=reason, seconds=time.time() - t0, peak=lim.peak_gib(), created=guard.now())
        guard.atomic_write_json(OUT, res, mode=0o400)
        guard.gpu_ledger(t0iso, guard.now(), "msv1_gpu_checks", "msv1 gpu checks", status)
        guard.ledger("MS_V1_GPU_CHECKS", f"status={status} pass={res.get('all_pass')} {reason}")
        print(json.dumps(res, indent=1, default=str)[:6000])
    sys.exit(0 if status == "COMPLETE" and res.get("all_pass") else 1)


def run(dev, lim):
    import numpy as np
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from conftest import single_doc_batch
    from test_multiscale_v1 import ms_cfg, rids, routing_for
    from csr.data import to_torch
    from csr.model import ModelConfig
    from csr.multiscale import CONDITIONS, MultiscaleModel, center, config_for, context_means, rms
    from csr.ms_run import make_sources
    res = {"tiny": {}, "full": {}}
    ok = True
    torch.manual_seed(0)
    tm = MultiscaleModel(ms_cfg()).double().to(dev).eval()
    mv = lambda b: {k: v.to(dev) for k, v in b.items()}
    ids = rids(120, 4)
    starts = [0, 20, 51, 90]
    cut = 63
    ids2 = ids.copy()
    ids2[cut + 1:] = rids(120 - cut - 1, 77)
    from csr.segment import starts_to_arrays
    _, cpos = starts_to_arrays(120, starts)
    with torch.no_grad():
        tok = tm(mv(single_doc_batch(ids, starts)), routing_for("TOKEN", 0, 0))
        for cond in sorted(CONDITIONS):
            r = routing_for(cond, 0.5, 1.0)
            z0 = tm(mv(single_doc_batch(ids, starts)), routing_for(cond, 0, 0))
            o1 = tm(mv(single_doc_batch(ids, starts)), r, collect=True)
            o2 = tm(mv(single_doc_batch(ids2, starts)), r)
            st = tm.new_ms_state()
            inc = max(float((tm.ms_step(int(ids[t]), t, int(cpos[t]), r, st)[0] - o1.logits[0, t]).abs().max())
                      for t in range(120))
            rec = {"zero_equals_token": bool(torch.equal(z0.logits, tok.logits)),
                   "future_invariance_max_diff": float((o1.logits[0, :cut + 1] - o2.logits[0, :cut + 1]).abs().max()),
                   "incremental_max_diff": inc}
            if CONDITIONS[cond][2]:
                rk, rmsd = True, 0.0
                for q in o1.routing:
                    zt, zr = q["z_token"][0], q["z_route"][0]
                    rk &= bool(torch.equal(q["topi"][0], zt.topk(4, -1).indices))
                    if q["ctx"] is not None:
                        rmsd = max(rmsd, float((rms(center(zr)) - rms(center(zt + q["ctx"][0]))).abs().max()))
                rec.update(scale_top4_equals_token=rk, scale_rms_max_diff=rmsd)
            rec["pass"] = rec["zero_equals_token"] and rec["future_invariance_max_diff"] <= 1e-10 and \
                rec["incremental_max_diff"] <= 1e-9 and rec.get("scale_top4_equals_token", True) and \
                rec.get("scale_rms_max_diff", 0.0) <= 1e-10
            ok &= rec["pass"]
            res["tiny"][cond] = rec
    lim.check()
    base = json.loads((guard.ROOT / "configs/multiscale-routing-v1/calibration/seed98/alpha_0p0__LOCAL-FIXED.json").read_text())
    d = guard.ROOT / base["data"]["dir"]
    for cond in sorted(CONDITIONS):
        cfg = config_for(base, cond, 0.5, 0.5, f"gpucheck_{cond}")
        _, src = make_sources(cfg, d, "calib")
        tb = to_torch(src.batch(src.rows[:4]), dev)
        torch.manual_seed(0)
        m = MultiscaleModel(ModelConfig.from_dict(cfg["model"])).to(dev).float()
        r = routing_for(cond, 0.5, 0.5)
        out = m(tb, r, collect=True)
        tgt = tb["targets"]
        loss = torch.nn.functional.cross_entropy(out.logits.view(-1, out.logits.shape[-1]), tgt.view(-1),
                                                 ignore_index=-100) + 0.01 * out.aux_loss + 0.001 * out.z_loss
        loss.backward()
        grads_finite = all(torch.isfinite(p.grad).all().item() for p in m.parameters() if p.grad is not None)
        nonpad = tb["doc_ids"] >= 0
        distinct4 = all(bool((q["topi"][nonpad].sort(-1).values.diff(dim=-1) > 0).all()) and q["topi"].shape[-1] == 4
                        for q in out.routing)
        lin = 0.0
        for li, q in enumerate(out.routing):
            if q["ctx"] is None:
                continue
            W = m.moe_blocks()[li].ffn.router.weight.detach()
            zt = q["z_token"].detach()
            ml, _, mb, _ = context_means(q["h"].detach(), tb["positions"], tb["chunk_pos"], nonpad,
                                         r.uses_local(q["kind"]), r.uses_broad(q["kind"]))
            zl, _, zb, _ = context_means(zt, tb["positions"], tb["chunk_pos"], nonpad,
                                         r.uses_local(q["kind"]), r.uses_broad(q["kind"]))
            if ml is not None:
                lin = max(lin, float((ml @ W.T - zl).abs().max()))
            if mb is not None:
                lin = max(lin, float((mb @ W.T - zb).abs().max()))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            ob = m(tb, r)
        rec = {"loss_fp32": float(loss), "finite_loss": bool(torch.isfinite(loss)), "grads_finite": grads_finite,
               "exactly_4_distinct": distinct4, "bf16_finite": bool(torch.isfinite(ob.logits).all()),
               "linear_equivalence_max_abs_err_fp32": lin}
        rec["pass"] = rec["finite_loss"] and grads_finite and distinct4 and rec["bf16_finite"] and lin < 1e-3
        ok &= rec["pass"]
        res["full"][cond] = rec
        del m, out, ob
        torch.cuda.empty_cache()
        lim.check()
    res["all_pass"] = bool(ok)
    return res


if __name__ == "__main__":
    main()
