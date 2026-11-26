#!/usr/bin/env python
"""Training-data-only calibration of tau_s, tau_r and lambda for the semantic objective (task §9).

Declared grids and rules (fixed before any held-out result is seen):
  tau_s in {0.02, 0.05, 0.1, 0.2}: median effective neighbour count exp(H(Q_i)) closest to 8.
  tau_r in {1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2}: at the initialization checkpoint, median exp(H(R_i)) closest
        to that of Q at the chosen tau_s.
  lambda: ratio target 0.10 (inside the 5-20% band) of the weighted semantic router-gradient norm to the LM
        router-gradient norm, averaged over K training micro-batches, rounded to one significant figure.
Uses condition E's (semantic) segmentation and the training split only. Dry-run by default.
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402
from train import load_seg, plan, resolve_config  # noqa: E402

TAU_S_GRID = [0.02, 0.05, 0.1, 0.2]
TAU_R_GRID = [1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2]
TARGET_NEIGHBOURS = 8.0
TARGET_RATIO = 0.10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="configs/v2/base.json")
    ap.add_argument("--set", action="append")
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    cfg = resolve_config(argparse.Namespace(base=a.base, condition="E", set=a.set, run_name="calibrate"))
    print(f"calibrate on train, init={cfg['init']}, grids tau_s={TAU_S_GRID} tau_r={TAU_R_GRID}")
    if not guard.require_execution(a.execute, "gpu" if a.device == "cuda" else "cpu-model"):
        return
    mon = None
    if a.device == "cuda":
        guard.occupancy_gate("calibrate")
        dev = guard.setup_cuda_device()
        mon = guard.OccupancyMonitor(10.0)
        mon.start()
    import numpy as np
    import torch
    from csr.data import to_torch, train_order
    from csr.semantic_loss import relational_kl, select_valid_chunks
    from csr.train_lib import BatchSource, build_model, forward_losses, router_grad_norms, routing_from_cfg
    if a.device == "cpu":
        dev = torch.device("cpu")
    guard.register_self("calibrate")
    rd = guard.RunDir("calibrate", cfg)
    t0iso = guard.now()
    lim = guard.RunLimits(30 * 60, monitor=mon)
    status, reason, res = "INCOMPLETE", "", {}
    try:
        d, tr, steps, _ = plan(cfg, require_semantic_calibration=False)  # produces them
        seg = load_seg(cfg, "train")
        src = BatchSource(tr, cfg["data"]["seq_len"], seg, train_order(tr.n_docs, cfg["data"]["train_order_seed"]))
        model = build_model(cfg, dev)
        mb = cfg["train"]["micro_batch_rows"]
        batches = [src.rows[i * mb:(i + 1) * mb] for i in range(a.batches)]
        # collect u and chunk logits at init (no grad) for temperature selection
        qeff, reff = {t: [] for t in TAU_S_GRID}, {t: [] for t in TAU_R_GRID}
        for rows in batches:
            tb = to_torch(src.batch(rows), dev)
            with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                out = model(tb, routing_from_cfg(cfg), need_chunk_logits=True)
            lay = out.layout
            keys = tb["chunk_key"][lay.rows_b, lay.rows_t0].cpu().numpy()
            u = torch.as_tensor(seg.u[keys].astype(np.float32), device=dev)
            idx = select_valid_chunks(u, seg.phash[keys], cfg["semantic"]["n_max"])
            cl = torch.stack([r["chunk_logits"] for r in out.routing])[:, idx]
            for ts in TAU_S_GRID:
                _, st = relational_kl(u[idx], cl, ts, 1e-2, cfg["semantic"]["min_chunks"])
                qeff[ts].append(st["q_entropy_eff"])
            for tr_ in TAU_R_GRID:
                _, st = relational_kl(u[idx], cl, 0.1, tr_, cfg["semantic"]["min_chunks"])
                reff[tr_].append(st["r_entropy_eff"])
        q_med = {t: float(np.median(v)) for t, v in qeff.items()}
        r_med = {t: float(np.median(v)) for t, v in reff.items()}
        tau_s = min(TAU_S_GRID, key=lambda t: abs(q_med[t] - TARGET_NEIGHBOURS))
        tau_r = min(TAU_R_GRID, key=lambda t: abs(r_med[t] - q_med[tau_s]))
        cfg["semantic"].update({"tau_s": tau_s, "tau_r": tau_r, "lambda": 1.0})
        ratios = []
        for rows in batches:
            tb = to_torch(src.batch(rows), dev)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                parts, _ = forward_losses(model, tb, cfg, seg)
            g_lm, g_sem = router_grad_norms(model, parts, cfg)
            ratios.append((g_lm, g_sem))
            model.zero_grad(set_to_none=True)
            lim.check()
        g_lm = float(np.mean([r[0] for r in ratios]))
        g_sem = float(np.mean([r[1] for r in ratios]))
        lam_raw = TARGET_RATIO * g_lm / max(g_sem, 1e-12)
        lam = float(f"{lam_raw:.0e}")
        res = {"tau_s": tau_s, "tau_r": tau_r, "lambda": lam, "lambda_raw": lam_raw,
               "weighted_ratio_at_lambda": lam * g_sem / g_lm, "router_gradnorm_lm": g_lm,
               "router_gradnorm_sem_unweighted": g_sem, "q_eff_median": q_med, "r_eff_median": r_med,
               "init": cfg["init"], "batches": a.batches}
        status = "COMPLETE"
    except guard.GuardError as e:
        status, reason = ("CONTAMINATED" if "CONTAMINATED" in str(e) else "ABORTED"), str(e)
    except Exception as e:  # noqa: BLE001
        import traceback
        status, reason = "FAILED", f"{type(e).__name__}: {e}"
        rd.log(traceback.format_exc())
    finally:
        lim.close()
        if mon is not None:
            mon.stop()
        guard.atomic_write_json(rd.path / "summary.json", {"status": status, "reason": reason,
                                                           "result_label": cfg.get("result_label", "experiment"), **res})
        rd.set_status(status, reason=reason)
        if a.device == "cuda":
            guard.gpu_ledger(t0iso, guard.now(), rd.run_id, "calibrate", status)
        print(json.dumps(res, indent=1), status, reason)


if __name__ == "__main__":
    main()
