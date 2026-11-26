#!/usr/bin/env python
"""csr-multiscale-routing-v1: train one config (calibration / benchmark / pilot / main). One run per process.

Dry-run by default. GPU execution only via scripts/gpu_guarded.sh with --device cuda --execute and
MOE_RESEARCH_APPROVE=gpu. Monitoring evaluations use the splits in cfg["eval"]["monitor_splits"] (first
cfg["eval"]["max_rows"] packed rows); cfg["eval"]["final_full_calib"] adds a full calib-split evaluation at the end
(calibration and pilot runs). The val split is read only when listed in monitor_splits (main runs).
"""
import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--mode", choices=["train", "benchmark"], default="train")
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    cfg_path = guard.inside_root(a.config)
    cfg = json.loads(cfg_path.read_text())
    from csr.ms_run import check_cfg
    d = check_cfg(cfg)
    print(json.dumps({"config": str(cfg_path), "condition": cfg["condition"], "run_name": cfg["run_name"],
                      "mode": a.mode, "device": a.device, "token_budget": cfg["train"]["token_budget"]}))
    if not guard.require_execution(a.execute, "gpu" if a.device == "cuda" else "cpu-model"):
        return
    sys.exit(0 if run(cfg, cfg_path, d, a) == "COMPLETE" else 1)


def run(cfg, cfg_path, d, a):
    import numpy as np
    import torch
    from csr.data import to_torch, train_order
    from csr.model import count_parameters, trainable_parameter_names
    from csr.multiscale import MSRouting
    from csr.ms_run import (build_ms_model, make_sources, ms_evaluate, require_disk_ms, router_lm_gradnorm,
                            train_plan)
    from csr.train_lib import BatchSource, lr_at, make_optimizer, valid_tokens
    torch.set_num_threads(4)
    monitor = None
    if a.device == "cuda":
        guard.occupancy_gate(f"msv1 {a.mode}:{cfg['run_name']}")
        device = guard.setup_cuda_device()
        monitor = guard.OccupancyMonitor(interval=15.0)
        monitor.start()
    else:
        device = torch.device("cpu")
    guard.register_self(f"msv1 {a.mode} {cfg['run_name']} on {a.device}")
    require_disk_ms(400_000_000, f"msv1 {a.mode} {cfg['run_name']}")
    rd = guard.RunDir(f"msv1_{a.mode}_{cfg['run_name']}", cfg)
    t_start_iso, t_proc0 = guard.now(), time.time()
    limits = guard.RunLimits(max_seconds=cfg["train"].get("max_seconds", guard.MAX_ATOMIC_SECONDS), monitor=monitor)
    status, reason, summary = "INCOMPLETE", "", {}
    try:
        routing = MSRouting.from_cfg(cfg["routing"])
        tr, steps, total = train_plan(cfg, d)
        if a.max_steps is not None:
            steps = steps[:a.max_steps]
            total = sum(valid_tokens(s, tr) for s in steps)
        torch.manual_seed(cfg["train"]["seed"])
        from csr.multiscale import load_starts
        seg_tr = load_starts(cfg, "train", guard.ROOT)
        src = BatchSource(tr, cfg["data"]["seq_len"], seg_tr, train_order(tr.n_docs, cfg["data"]["train_order_seed"]))
        model = build_ms_model(cfg, device)
        opt = make_optimizer(model, cfg, device)
        mon_src = {}
        if a.mode == "train":
            for sp in cfg["eval"]["monitor_splits"]:
                mon_src[sp] = make_sources(cfg, d, sp)[1]
        trainable = trainable_parameter_names(model)
        facts = {"trainable_sha256": guard.sha256_json(trainable), "params": count_parameters(model),
                 "planned_valid_tokens": total, "n_steps": len(steps), "config_sha256": guard.sha256_file(cfg_path),
                 "config_path": str(cfg_path.relative_to(guard.ROOT)), "routing": routing.to_dict(),
                 "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
                 "torch": torch.__version__, "init_param_sha256": _param_sha(model)}
        guard.atomic_write_json(rd.path / "run_facts.json", facts)
        rd.log(json.dumps(facts))
        use_amp = device.type == "cuda" and cfg["train"]["precision"] == "bf16-autocast"
        t = cfg["train"]
        seen, t_loop = 0, time.time()
        next_eval = t["eval_every_tokens"]
        warm_t, warm_tok = None, 0
        model.train()
        for si, rows in enumerate(steps):
            lr = lr_at(si, len(steps), t)
            for g in opt.param_groups:
                g["lr"] = lr
            n_step = valid_tokens(rows, tr)
            mbs = [rows[i:i + t["micro_batch_rows"]] for i in range(0, len(rows), t["micro_batch_rows"])]
            agg = {"lm": 0.0, "aux": 0.0, "z": 0.0, "n": 0}
            gn = None
            for mi, mb in enumerate(mbs):
                tb = to_torch(src.batch(mb), device)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                    out = model(tb, routing)
                logits = out.logits.float()
                tgt = tb["targets"]
                lm_sum = torch.nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), tgt.view(-1),
                                                           ignore_index=-100, reduction="sum")
                n_valid = int((tgt != -100).sum())
                w = n_valid / n_step
                obj = lm_sum / n_step + w * (t["aux_coeff"] * out.aux_loss + t["z_coeff"] * out.z_loss)
                if mi == 0 and t["grad_probe_every"] and si % t["grad_probe_every"] == 0:
                    gn = router_lm_gradnorm(model, lm_sum / max(n_valid, 1))
                obj.backward()
                agg["lm"] += float(lm_sum)
                agg["aux"] += float(out.aux_loss) * n_valid
                agg["z"] += float(out.z_loss) * n_valid
                agg["n"] += n_valid
            if agg["n"] != n_step:
                raise guard.GuardError(f"accounting mismatch {agg['n']} != {n_step}")
            if not math.isfinite(agg["lm"]) or not math.isfinite(agg["aux"]) or not math.isfinite(agg["z"]):
                raise FloatingPointError("non-finite loss")
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), t["clip"])
            if not torch.isfinite(gnorm):
                raise FloatingPointError("non-finite grad norm")
            opt.step()
            opt.zero_grad(set_to_none=True)
            seen += n_step
            if si == t.get("throughput_warmup_steps", 10) - 1:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                warm_t, warm_tok = time.time(), seen
            limits.check()
            if si % t["log_every"] == 0 or si == len(steps) - 1:
                pa, pr = limits.peak_gib()
                rec = {"step": si, "valid_tokens_seen": seen, "lr": lr, "lm_ce": agg["lm"] / agg["n"],
                       "aux": agg["aux"] / agg["n"], "z": agg["z"] / agg["n"], "grad_norm": float(gnorm),
                       "elapsed_s": time.time() - t_loop, "peak_alloc_gib": pa, "peak_reserved_gib": pr}
                if gn is not None:
                    rec["router_lm_gradnorm"] = gn
                rd.metric(rec)
                rd.log(f"step {si} tok {seen} ce {rec['lm_ce']:.4f} gn {float(gnorm):.2f} peak {pr:.2f}GiB")
            if a.mode == "train" and mon_src and (seen >= next_eval or si == len(steps) - 1):
                ev = {}
                for sp, s_ in mon_src.items():
                    r = ms_evaluate(model, s_, routing, device, cfg["eval"]["rows_per_batch"], cfg["eval"]["max_rows"],
                                    autocast=use_amp)
                    ev[sp] = {k: v for k, v in r.items() if k not in ("doc_nll", "doc_n")}
                rd.metric({"monitor_eval": ev, "valid_tokens_seen": seen, "step": si})
                rd.log("monitor " + " ".join(f"{sp} {v['ce_nats_per_token']:.4f}" for sp, v in ev.items()))
                next_eval += t["eval_every_tokens"]
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_end = time.time()
        pa, pr = limits.peak_gib()
        tput = (seen - warm_tok) / (t_end - warm_t) if warm_t and seen > warm_tok else None
        summary = {"valid_train_tokens": seen, "steps": len(steps), "loop_seconds": t_end - t_loop,
                   "train_valid_tokens_per_s_after_warmup": tput, "peak_alloc_gib": pa, "peak_reserved_gib": pr}
        if a.mode == "train" and cfg["eval"].get("final_full_calib"):
            t0 = time.time()
            _, csrc = make_sources(cfg, d, "calib")
            ev = ms_evaluate(model, csrc, routing, device, cfg["eval"]["rows_per_batch"], None, autocast=use_amp)
            ev["eval_seconds"] = time.time() - t0
            ev["split"] = "calib"
            guard.atomic_write_json(rd.path / "calib_eval.json", ev)
            summary["calib_ce"] = ev["ce_nats_per_token"]
            summary["calib_eval_seconds"] = ev["eval_seconds"]
            limits.check()
        if a.mode == "train" and t.get("save_checkpoint"):
            require_disk_ms(2 * 4 * sum(p.numel() for p in model.parameters()), "msv1 checkpoint")
            tmp = rd.path / ".model.pt.tmp"
            torch.save({"model": model.state_dict(), "config": cfg}, tmp)
            tmp.rename(rd.path / "model.pt")
            summary["checkpoint_sha256"] = guard.sha256_file(rd.path / "model.pt")
        status = "COMPLETE"
    except guard.GuardError as e:
        status = "CONTAMINATED" if "CONTAMINATED" in str(e) else "ABORTED"
        reason = str(e)
    except Exception as e:  # noqa: BLE001
        status, reason = "FAILED", f"{type(e).__name__}: {e}"
        rd.log(traceback.format_exc())
    finally:
        limits.close()
        if monitor is not None:
            monitor.stop()
            if monitor.contaminated and status == "COMPLETE":
                status, reason = "CONTAMINATED", monitor.detail
        summary["process_seconds"] = time.time() - t_proc0
        guard.atomic_write_json(rd.path / "summary.json", {"status": status, "reason": reason, **summary})
        rd.set_status(status, reason=reason)
        if a.device == "cuda":
            guard.gpu_ledger(t_start_iso, guard.now(), rd.run_id, f"msv1 {a.mode}", status)
        rd.log(f"STATUS {status} {reason}")
        print(f"RUN_DIR {rd.path.relative_to(guard.ROOT)}")
    return status


def _param_sha(model):
    import hashlib
    h = hashlib.sha256()
    for n, p in sorted(model.state_dict().items()):
        h.update(n.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


if __name__ == "__main__":
    main()
