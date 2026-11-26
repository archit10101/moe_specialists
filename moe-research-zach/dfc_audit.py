#!/usr/bin/env python
"""csr-domain-forecast-curriculum-v1: specialisation audits of saved checkpoints on the CALIBRATION split only
(prereg section 7). For each requested (tag, mode):
  full calibration pass: CE overall/per domain, per-(layer, domain, expert) top-4 counts, per-document counts ->
  MI (+ shuffled-domain null), H(d|e), p(e|d), loads/collapse, provisional-group mass, gate/reroute stats (active);
  CALIB-ABL functional ablations: per (layer, domain) the domain's 4 most-enriched experts; per domain all layers
  (enriched); per domain all layers (provisional group 4d..4d+3). Experts' gates are zeroed (they stay eligible).
mode 'normal' = the run's own curriculum strength at that checkpoint (1 at p20/p40 for curriculum conditions) with
its own forecast source; 'neutral' = strength 0 (no retraining). Writes <run>/audit_<tag>_<mode>.json (never
overwritten). Diagnostic only: nothing is selected from these numbers. Dry-run by default.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--jobs", required=True, help="comma list of tag:mode, e.g. p20:normal,p20:neutral,p50:neutral")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--max-rows", type=int, help="(tests only) truncate evaluation rows")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    if not guard.require_execution(a.execute, "gpu" if a.device == "cuda" else "cpu-model"):
        return
    import numpy as np
    import torch
    from csr.dfc_eval import enriched_experts, eval_pass, routing_metrics
    from csr.dfc_lib import ABL_ROWS_PER_DOM, N_DOM, code_hashes, eval_rows, load_cache, load_split
    from csr.dfc_model import CONDITIONS, build_dfc_model, provisional_group
    torch.set_num_threads(4)
    monitor = None
    if a.device == "cuda":
        guard.occupancy_gate("dfc audit")
        device = guard.setup_cuda_device()
        monitor = guard.OccupancyMonitor(interval=15.0)
        monitor.start()
    else:
        device = torch.device("cpu")
    guard.register_self("dfc audit")
    t_iso = guard.now()
    rd = guard.inside_root(a.run_dir)
    cfg = json.loads((rd / "config.json").read_text())
    cond = cfg["condition"]
    curriculum, use_fc, _, src = CONDITIONS[cond]
    status, reason = "FAILED", ""
    limits = guard.RunLimits(max_seconds=5100, monitor=monitor)
    done = []
    try:
        csp, cdoms, _ = load_split("calib")
        rows_all, _ = eval_rows(csp, cdoms)
        rows_abl, _ = eval_rows(csp, cdoms, ABL_ROWS_PER_DOM)
        if a.max_rows:
            rows_all, rows_abl = rows_all[:a.max_rows], rows_abl[:a.max_rows]
        cache = load_cache("calib", src) if (curriculum and use_fc) else None
        doc_dom = {i: int(d) for i, d in enumerate(cdoms)}
        for job in a.jobs.split(","):
            tag, mode = job.split(":")
            outp = rd / f"audit_{tag}_{mode}.json"
            if outp.exists():
                print("exists", outp)
                continue
            t0 = time.time()
            ck = torch.load(rd / f"ckpt_{tag}.pt", map_location="cpu", weights_only=False)
            model = build_dfc_model(cfg["model"], cond, cfg["seed"], device)
            model.load_state_dict(ck["model"])
            model.eval()
            strength = 1.0 if (mode == "normal" and curriculum and tag in ("p20", "p40")) else 0.0
            if mode == "normal" and curriculum and tag not in ("p20", "p40"):
                raise ValueError(f"normal mode at {tag} has strength 0 by construction; use neutral")
            full = eval_pass(model, rows_all, csp, cdoms, device, cache=cache, strength=strength, counts=True,
                             per_doc=True, gate_stats=strength > 0)
            limits.check()
            C = full["counts"]
            met = routing_metrics(C, full["doc_counts"], doc_dom)
            base = eval_pass(model, rows_abl, csp, cdoms, device, cache=cache, strength=strength)
            moe_ids = model.moe_layer_ids
            abl = {"base_ce_domain": base["ce_domain"], "per_layer_domain": [], "all_layers_enriched": [],
                   "all_layers_provisional": []}

            def run_dis(spec):
                def fn(tb):
                    m = tb["doc_ids"] >= 0
                    return [(lid, es, m) for lid, es in spec]
                r = eval_pass(model, rows_abl, csp, cdoms, device, cache=cache, strength=strength, disable_fn=fn)
                dce = [r["ce_domain"][d] - base["ce_domain"][d] for d in range(N_DOM)]
                return dce
            for mi, lid in enumerate(moe_ids):
                row = []
                for d in range(N_DOM):
                    es = enriched_experts(C[mi], d)
                    dce = run_dis([(lid, es)])
                    spec_ = dce[d] - float(np.mean([dce[o] for o in range(N_DOM) if o != d]))
                    row.append({"domain": d, "experts": es, "dce": dce, "specificity": spec_,
                                "overlap_provisional": len(set(es) & set(provisional_group(d)))})
                abl["per_layer_domain"].append(row)
                limits.check()
            for kind in ("enriched", "provisional"):
                for d in range(N_DOM):
                    spec = [(lid, enriched_experts(C[mi], d) if kind == "enriched" else provisional_group(d))
                            for mi, lid in enumerate(moe_ids)]
                    dce = run_dis(spec)
                    abl[f"all_layers_{kind}"].append(
                        {"domain": d, "dce": dce,
                         "specificity": dce[d] - float(np.mean([dce[o] for o in range(N_DOM) if o != d]))})
            sp_cells = [c["specificity"] for row in abl["per_layer_domain"] for c in row]
            abl["mean_specificity_layer_domain"] = float(np.mean(sp_cells))
            abl["mean_specificity_all_layers_enriched"] = float(np.mean(
                [c["specificity"] for c in abl["all_layers_enriched"]]))
            abl["mean_specificity_all_layers_provisional"] = float(np.mean(
                [c["specificity"] for c in abl["all_layers_provisional"]]))
            sf, sp_ = model.scales()
            doc = {"run_dir": str(rd.relative_to(guard.ROOT)), "condition": cond, "seed": cfg["seed"], "tag": tag,
                   "mode": mode, "strength": strength, "ckpt_seen": ck["seen"], "ckpt_step": ck["step"],
                   "ckpt_sha256": guard.sha256_file(rd / f"ckpt_{tag}.pt"),
                   "calib_ce": full["ce"], "calib_ce_domain": full["ce_domain"], "calib_tokens": full["n_tokens"],
                   "counts": C.tolist(), "metrics": met, "ablation": abl,
                   "gate_mean": full.get("gate_mean"), "rerouted_frac": full.get("rerouted_frac"),
                   "scale_f": sf.detach().float().tolist(), "scale_p": sp_.detach().float().tolist(),
                   "gate_bias": model.rt_gate_b.detach().float().tolist(),
                   "seconds": time.time() - t0, "code_sha256": code_hashes(), "created": guard.now()}
            guard.atomic_write_json(outp, doc, mode=0o400)
            done.append(job)
            print(f"audit {cond} s{cfg['seed']} {tag}:{mode} ce {full['ce']:.4f} mi {np.round(met['mi'], 3).tolist()}"
                  f" spec {abl['mean_specificity_layer_domain']:.4f} {time.time() - t0:.0f}s", flush=True)
            del model
        status = "COMPLETE"
    except guard.GuardError as e:
        status, reason = ("CONTAMINATED" if "CONTAMINATED" in str(e) else "INTERRUPTED"), str(e)
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        status, reason = "FAILED", f"{type(e).__name__}: {e}"
    finally:
        limits.close()
        if monitor is not None:
            monitor.stop()
            if monitor.contaminated and status == "COMPLETE":
                status = "CONTAMINATED"
        if a.device == "cuda":
            guard.gpu_ledger(t_iso, guard.now(), f"dfc_audit_{rd.name}_{guard.utc_stamp()}", "dfc audit", status)
        print("STATUS", status, reason, done)
    sys.exit(0 if status == "COMPLETE" else (75 if status == "INTERRUPTED" else 1))


if __name__ == "__main__":
    main()
