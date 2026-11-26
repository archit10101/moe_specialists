#!/usr/bin/env python
"""csr-sequence-routing-search-v1: train one config (one run per process). Dry-run by default.

GPU execution only via scripts/gpu_guarded.sh with --device cuda --execute and MOE_RESEARCH_APPROVE=gpu.
Monitoring evaluations read only the first cfg.eval.max_rows packed rows of CALIB-SEARCH; at the end the full
CALIB-SEARCH set is evaluated (cfg.eval.final_calib_search). The val split and CALIB-SELECTION are never read here.
Resume: --resume <run dir> continues an interrupted run from its last train_state.pt into a NEW run directory.
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
    ap.add_argument("--max-steps", type=int, help="stop after this many steps (testing / interruption simulation)")
    ap.add_argument("--resume", help="run directory of an interrupted run with train_state.pt")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    cfg_path = guard.inside_root(a.config)
    cfg = json.loads(cfg_path.read_text())
    from csr.srs_run import check_cfg
    d = check_cfg(cfg)
    print(json.dumps({"config": str(cfg_path), "candidate": cfg["candidate"], "run_name": cfg["run_name"],
                      "device": a.device, "token_budget": cfg["train"]["token_budget"], "resume": a.resume}))
    if not guard.require_execution(a.execute, "gpu" if a.device == "cuda" else "cpu-model"):
        return
    sys.exit(0 if run(cfg, cfg_path, d, a) == "COMPLETE" else 1)


def run(cfg, cfg_path, d, a):
    import numpy as np
    import torch
    from csr.data import to_torch
    from csr.model import count_parameters, trainable_parameter_names
    from csr.seqroute import build_sr_model
    from csr.srs_run import (collapse_check, eval_source, flops_per_token, param_sha, require_disk_srs,
                             sr_evaluate, train_source)
    from csr.train_lib import lr_at, make_optimizer, valid_tokens
    torch.set_num_threads(4)
    monitor = None
    if a.device == "cuda":
        guard.occupancy_gate(f"srsv1 train:{cfg['run_name']}")
        device = guard.setup_cuda_device()
        monitor = guard.OccupancyMonitor(interval=15.0)
        monitor.start()
    else:
        device = torch.device("cpu")
    guard.register_self(f"srsv1 train {cfg['run_name']} on {a.device}")
    require_disk_srs(600_000_000, f"srsv1 train {cfg['run_name']}")
    resume_state, resumed_from = None, None
    if a.resume:
        prev = guard.inside_root(a.resume)
        if json.loads((prev / "config.json").read_text()) != cfg:
            raise guard.GuardError("resume: config differs from the interrupted run")
        resume_state = torch.load(prev / "train_state.pt", map_location="cpu", weights_only=False)
        resumed_from = str(prev.relative_to(guard.ROOT))
    rd = guard.RunDir(f"srsv1_train_{cfg['run_name']}" + ("_resume" if a.resume else ""), cfg)
    t_start_iso, t_proc0 = guard.now(), time.time()
    limits = guard.RunLimits(max_seconds=cfg["train"].get("max_seconds", guard.MAX_ATOMIC_SECONDS), monitor=monitor)
    status, reason, summary = "INCOMPLETE", "", {}
    try:
        tr, src, steps, total = train_source(cfg, d)
        if a.max_steps is not None:
            steps_run = min(a.max_steps, len(steps))
        else:
            steps_run = len(steps)
        model = build_sr_model(cfg, device)
        init_sha = param_sha(model.state_dict())
        init_backbone_sha = param_sha(model.backbone_state())
        opt = make_optimizer(model, cfg, device)
        t = cfg["train"]
        mon = eval_source(cfg, "calib_search", d)[1]
        start_step, seen = 0, 0
        if resume_state is not None:
            model.load_state_dict(resume_state["model"])
            opt.load_state_dict(resume_state["opt"])
            start_step, seen = resume_state["next_step"], resume_state["seen"]
            if resume_state["init_param_sha256"] != init_sha:
                raise guard.GuardError("resume: initialization hash differs")
            rd.log(f"resumed from {resumed_from} at step {start_step} seen {seen}")
        mcfg = model.cfg
        n_params = count_parameters(model)
        facts = {"trainable_sha256": guard.sha256_json(trainable_parameter_names(model)), "params": n_params,
                 "extra_router_params": model.extra_parameter_count(),
                 "extra_router_param_names": model.extra_parameter_names(),
                 "extra_router_param_fraction": model.extra_parameter_count() / n_params["total"],
                 "planned_valid_tokens": total, "n_steps": len(steps), "config_sha256": guard.sha256_file(cfg_path),
                 "config_path": str(cfg_path.relative_to(guard.ROOT)), "routing": model.sr.to_dict(),
                 "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
                 "torch": torch.__version__, "init_param_sha256": init_sha,
                 "init_backbone_param_sha256": init_backbone_sha, "resumed_from": resumed_from,
                 "train_rows_sha256": guard.sha256_json(steps[:3]) + ":" + guard.sha256_json(len(steps))}
        guard.atomic_write_json(rd.path / "run_facts.json", facts)
        rd.log(json.dumps({k: v for k, v in facts.items() if k != "extra_router_param_names"}))
        use_amp = device.type == "cuda" and t["precision"] == "bf16-autocast"
        model.train()
        model.tau_stats_reset()
        if model.sr.dynamic and start_step == 0:
            tb0 = to_torch(src.batch(steps[0][:t["micro_batch_rows"]]), device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                tau0 = model.tau_calibrate(tb0)
            rd.log(f"tau initialized from a no-grad pre-pass on the first training micro-batch: {tau0}")
            rd.metric({"tau_init": tau0})
        t_loop = time.time()
        next_eval = (seen // t["eval_every_tokens"] + 1) * t["eval_every_tokens"]
        warm_t, warm_tok = None, 0
        last_state_save = time.time()
        for si in range(start_step, steps_run):
            rows = steps[si]
            lr = lr_at(si, len(steps), t)
            for g in opt.param_groups:
                g["lr"] = lr
            n_step = valid_tokens(rows, tr)
            mbs = [rows[i:i + t["micro_batch_rows"]] for i in range(0, len(rows), t["micro_batch_rows"])]
            agg = {"lm": 0.0, "aux": 0.0, "z": 0.0, "bd": 0.0, "n": 0}
            for mb in mbs:
                tb = to_torch(src.batch(mb), device)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                    out = model(tb)
                logits = out.logits.float()
                tgt = tb["targets"]
                lm_sum = torch.nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), tgt.view(-1),
                                                           ignore_index=-100, reduction="sum")
                n_valid = int((tgt != -100).sum())
                w = n_valid / n_step
                obj = lm_sum / n_step + w * (t["aux_coeff"] * out.aux_loss + t["z_coeff"] * out.z_loss
                                             + model.sr.bd_coeff * out.bd_loss)
                obj.backward()
                agg["lm"] += float(lm_sum)
                agg["aux"] += float(out.aux_loss) * n_valid
                agg["z"] += float(out.z_loss) * n_valid
                agg["bd"] += float(out.bd_loss) * n_valid
                agg["n"] += n_valid
            if agg["n"] != n_step:
                raise guard.GuardError(f"accounting mismatch {agg['n']} != {n_step}")
            if not all(math.isfinite(agg[k]) for k in ("lm", "aux", "z", "bd")):
                raise FloatingPointError("non-finite loss")
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), t["clip"])
            if not torch.isfinite(gnorm):
                raise FloatingPointError("non-finite grad norm")
            opt.step()
            opt.zero_grad(set_to_none=True)
            seg_means = model.tau_update()  # threshold adaptation from this (completed) step only
            seen += n_step
            if si == start_step + t.get("throughput_warmup_steps", 10) - 1:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                warm_t, warm_tok = time.time(), seen
            limits.check()
            if si % t["log_every"] == 0 or si == len(steps) - 1:
                pa, pr = limits.peak_gib()
                rec = {"step": si, "valid_tokens_seen": seen, "lr": lr, "lm_ce": agg["lm"] / agg["n"],
                       "aux": agg["aux"] / agg["n"], "z": agg["z"] / agg["n"], "bd": agg["bd"] / agg["n"],
                       "grad_norm": float(gnorm), "elapsed_s": time.time() - t_loop, "peak_alloc_gib": pa,
                       "peak_reserved_gib": pr}
                if model.sr.dynamic:
                    rec["tau"] = model.tau.tolist()
                    rec["closed_seg_mean_len"] = seg_means
                rd.metric(rec)
                rd.log(f"step {si} tok {seen} ce {rec['lm_ce']:.4f} gn {float(gnorm):.2f} peak {pr:.2f}GiB")
            if seen >= next_eval or si == len(steps) - 1:
                r = sr_evaluate(model, mon, device, cfg["eval"]["rows_per_batch"], cfg["eval"]["max_rows"],
                                autocast=use_amp, diagnostics=False)
                rd.metric({"monitor_eval": {"calib_search_head": {k: v for k, v in r.items()
                                                                  if k not in ("doc_nll", "doc_n")}},
                           "valid_tokens_seen": seen, "step": si})
                rd.log(f"monitor calib_search_head {r['ce_nats_per_token']:.4f}")
                next_eval += t["eval_every_tokens"]
            if t.get("state_every_seconds") and (time.time() - last_state_save > t["state_every_seconds"]
                                                 or si == steps_run - 1) and si < len(steps) - 1:
                tmp = rd.path / ".train_state.pt.tmp"
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "next_step": si + 1, "seen": seen,
                            "init_param_sha256": init_sha}, tmp)
                tmp.replace(rd.path / "train_state.pt")
                last_state_save = time.time()
        if steps_run < len(steps):
            raise guard.GuardError(f"INTERRUPTED (max-steps) after step {steps_run - 1}; resumable")
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_end = time.time()
        pa, pr = limits.peak_gib()
        tput = (seen - warm_tok) / (t_end - warm_t) if warm_t and seen > warm_tok else None
        summary = {"valid_train_tokens": seen, "steps": len(steps), "loop_seconds": t_end - t_loop,
                   "train_valid_tokens_per_s_after_warmup": tput, "peak_alloc_gib": pa, "peak_reserved_gib": pr,
                   "resumed_from": resumed_from}
        if cfg["eval"].get("final_calib_search"):
            ev = sr_evaluate(model, mon, device, cfg["eval"]["rows_per_batch"], None, autocast=use_amp)
            ev["evalset"] = "calib_search"
            ev["collapse"] = collapse_check(ev)
            mean_len = float(np.mean(ev["mean_segment_length"])) if "mean_segment_length" in ev else None
            ev["flops_per_token"] = flops_per_token(mcfg, model.sr, cfg["data"]["seq_len"], mean_len)
            guard.atomic_write_json(rd.path / "calib_search_eval.json", ev)
            summary["calib_search_ce"] = ev["ce_nats_per_token"]
            summary["calib_search_valid_tokens"] = ev["n_valid_tokens"]
            summary["calib_search_eval_seconds"] = ev["eval_seconds"]
            summary["collapse"] = ev["collapse"]
            summary["train_flops"] = ev["flops_per_token"]["train"] * seen
            limits.check()
        if t.get("save_checkpoint"):
            require_disk_srs(2 * 4 * sum(p.numel() for p in model.parameters()), "srsv1 checkpoint")
            tmp = rd.path / ".model.pt.tmp"
            torch.save({"model": model.state_dict(), "config": cfg}, tmp)
            tmp.rename(rd.path / "model.pt")
            summary["checkpoint_sha256"] = guard.sha256_file(rd.path / "model.pt")
        status = "COMPLETE"
    except guard.GuardError as e:
        status = "CONTAMINATED" if "CONTAMINATED" in str(e) else ("INTERRUPTED" if "INTERRUPTED" in str(e)
                                                                  else "ABORTED")
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
            guard.gpu_ledger(t_start_iso, guard.now(), rd.run_id, "srsv1 train", status)
        rd.log(f"STATUS {status} {reason}")
        print(f"RUN_DIR {rd.path.relative_to(guard.ROOT)}")
    return status


if __name__ == "__main__":
    main()
