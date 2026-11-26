#!/usr/bin/env python
"""Train one config of csr-value-routed-compute-v1 (FULL-TOKEN4 diagnostic baseline) or csr-m2d2-domain-prior-v1
(B0-B3). Recipe = the project / clav1 training loop (AdamW, WSD, token-weighted CE + aux + z, bf16 autocast, fp32
router), unchanged. Dry-run by default; GPU only via scripts/gpu_guarded.sh with --device cuda --execute.
Monitoring evaluations read the first eval.max_rows rows of CALIB-SEARCH only. --stop-at <epoch>: save resumable state
and stop (INTERRUPTED) when the absolute wall-clock time is reached. --resume <run dir> continues into a NEW run dir.
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
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--resume")
    ap.add_argument("--stop-at", type=float)
    ap.add_argument("--runs-dir")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    cfg_path = guard.inside_root(a.config)
    cfg = json.loads(cfg_path.read_text())
    from csr.vm_run import check_cfg
    key, d = check_cfg(cfg)
    print(json.dumps({"config": str(cfg_path), "run_name": cfg["run_name"], "device": a.device,
                      "token_budget": cfg["train"]["token_budget"], "resume": a.resume}))
    if not guard.require_execution(a.execute, "gpu" if a.device == "cuda" else "cpu-model"):
        return
    sys.exit(0 if run(cfg, cfg_path, key, d, a) == "COMPLETE" else 1)


def make_optimizer(model, cfg, device):
    import torch
    t = cfg["train"]
    decay = [p for _, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for _, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    groups = [{"params": decay, "weight_decay": t["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=t["lr"], betas=tuple(t["betas"]), eps=1e-8, fused=(device.type == "cuda"))


def run(cfg, cfg_path, key, d, a):
    import numpy as np
    import torch
    from csr.data import to_torch
    from csr.model import count_parameters, trainable_parameter_names
    from csr.train_lib import lr_at, valid_tokens
    from csr.vm_run import (EXP, STOP_DEADLINE, basic_eval, build_model, code_hashes, collapse_check, eval_source,
                            param_sha, require_disk, train_source)
    torch.set_num_threads(4)
    monitor = None
    if a.device == "cuda":
        guard.occupancy_gate(f"{key} train:{cfg['run_name']}")
        device = guard.setup_cuda_device()
        monitor = guard.OccupancyMonitor(interval=15.0)
        monitor.start()
    else:
        device = torch.device("cpu")
    guard.register_self(f"{key} train {cfg['run_name']} on {a.device}")
    require_disk(600_000_000, f"{key} train {cfg['run_name']}")
    stop_at = min(a.stop_at or STOP_DEADLINE, STOP_DEADLINE)
    resume_state, resumed_from = None, None
    if a.resume:
        prev = guard.inside_root(a.resume)
        if json.loads((prev / "config.json").read_text()) != cfg:
            raise guard.GuardError("resume: config differs from the interrupted run")
        resume_state = torch.load(prev / "train_state.pt", map_location="cpu", weights_only=False)
        resumed_from = str(prev.relative_to(guard.ROOT))
    rd = guard.RunDir(f"{key}_train_{cfg['run_name']}" + ("_resume" if a.resume else ""), cfg,
                      base=guard.inside_root(a.runs_dir) if a.runs_dir else None)
    t_start_iso, t_proc0 = guard.now(), time.time()
    max_s = min(cfg["train"].get("max_seconds", guard.MAX_ATOMIC_SECONDS), max(60.0, stop_at - time.time() + 90))
    limits = guard.RunLimits(max_seconds=max_s, monitor=monitor)
    status, reason, summary = "INCOMPLETE", "", {}

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    def save_state(si, seen, acc_t, init_sha):
        tmp = rd.path / ".train_state.pt.tmp"
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "next_step": si + 1, "seen": seen,
                    "init_param_sha256": init_sha, "acc_t": acc_t}, tmp)
        tmp.replace(rd.path / "train_state.pt")
    try:
        tr, src, steps, total = train_source(cfg, key, d)
        n_steps = len(steps)
        steps_run = n_steps if a.max_steps is None else min(a.max_steps, n_steps)
        model = build_model(cfg, device)
        init_sha = param_sha(model.named_parameters())
        bb = model.backbone_state() if hasattr(model, "backbone_state") else dict(model.named_parameters())
        init_backbone_sha = param_sha(bb.items())
        opt = make_optimizer(model, cfg, device)
        t = cfg["train"]
        mon = eval_source(cfg, key, d, "calib_search")
        start_step, seen = 0, 0
        acc_t = {"step_s": 0.0, "step_s_after_warm": 0.0, "tok_after_warm": 0, "eval_s": 0.0}
        if resume_state is not None:
            model.load_state_dict(resume_state["model"])
            opt.load_state_dict(resume_state["opt"])
            start_step, seen = resume_state["next_step"], resume_state["seen"]
            acc_t.update(resume_state.get("acc_t", {}))
            if resume_state["init_param_sha256"] != init_sha:
                raise guard.GuardError("resume: initialization hash differs")
            rd.log(f"resumed from {resumed_from} at step {start_step} seen {seen}")
        n_params = count_parameters(model)
        if hasattr(model, "added_parameter_count"):
            n_params["added"] = model.added_parameter_count()
            n_params["added_fraction"] = n_params["added"] / n_params["total"]
        facts = {"trainable_sha256": guard.sha256_json(trainable_parameter_names(model)), "params": n_params,
                 "planned_valid_tokens": total, "n_steps": n_steps, "config_sha256": guard.sha256_file(cfg_path),
                 "config_path": str(cfg_path.relative_to(guard.ROOT)),
                 "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
                 "torch": torch.__version__, "init_param_sha256": init_sha,
                 "init_backbone_param_sha256": init_backbone_sha, "resumed_from": resumed_from,
                 "train_rows_sha256": guard.sha256_json(steps), "code_sha256": code_hashes(),
                 "micro_batch_rows": t["micro_batch_rows"], "accum": t["accum"], "stop_at": stop_at}
        guard.atomic_write_json(rd.path / "run_facts.json", facts)
        rd.log(json.dumps({k: v for k, v in facts.items() if k != "code_sha256"}))
        use_amp = device.type == "cuda" and t["precision"] == "bf16-autocast"
        model.train()
        next_eval = (seen // t["eval_every_tokens"] + 1) * t["eval_every_tokens"]
        warm = t.get("throughput_warmup_steps", 10)
        last_state_save = time.time()
        sync()
        for si in range(start_step, steps_run):
            if time.time() > stop_at:
                save_state(si - 1, seen, acc_t, init_sha)
                raise guard.GuardError(f"INTERRUPTED (deadline) before step {si}; resumable")
            ts0 = time.time()
            rows = steps[si]
            lr = lr_at(si, n_steps, t)
            for g in opt.param_groups:
                g["lr"] = lr
            n_step = valid_tokens(rows, tr)
            mbs = [rows[i:i + t["micro_batch_rows"]] for i in range(0, len(rows), t["micro_batch_rows"])]
            agg = {"lm": 0.0, "aux": 0.0, "z": 0.0, "n": 0}
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
                obj = lm_sum / n_step + w * (t["aux_coeff"] * out.aux_loss + t["z_coeff"] * out.z_loss)
                obj.backward()
                agg["lm"] += float(lm_sum)
                agg["aux"] += float(out.aux_loss) * n_valid
                agg["z"] += float(out.z_loss) * n_valid
                agg["n"] += n_valid
            if agg["n"] != n_step:
                raise guard.GuardError(f"accounting mismatch {agg['n']} != {n_step}")
            if not all(math.isfinite(agg[k]) for k in ("lm", "aux", "z")):
                raise FloatingPointError("non-finite loss")
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), t["clip"])
            if not torch.isfinite(gnorm):
                raise FloatingPointError("non-finite grad norm")
            opt.step()
            opt.zero_grad(set_to_none=True)
            seen += n_step
            sync()
            dt = time.time() - ts0
            acc_t["step_s"] += dt
            if si >= warm:
                acc_t["step_s_after_warm"] += dt
                acc_t["tok_after_warm"] += n_step
            limits.check()
            if si % t["log_every"] == 0 or si == n_steps - 1:
                pa, pr_ = limits.peak_gib()
                rec = {"step": si, "valid_tokens_seen": seen, "lr": lr, "lm_ce": agg["lm"] / agg["n"],
                       "aux": agg["aux"] / agg["n"], "z": agg["z"] / agg["n"], "grad_norm": float(gnorm),
                       "step_s": dt, "peak_alloc_gib": pa, "peak_reserved_gib": pr_}
                if hasattr(model, "alpha"):
                    rec["alpha"] = model.alpha.detach().float().tolist()
                rd.metric(rec)
                rd.log(f"step {si} tok {seen} ce {rec['lm_ce']:.4f} gn {float(gnorm):.2f} {dt * 1e3:.0f}ms")
            if seen >= next_eval or si == n_steps - 1:
                te = time.time()
                r = basic_eval(model, mon, device, cfg["eval"]["rows_per_batch"], cfg["eval"]["max_rows"],
                               autocast=use_amp)
                sync()
                acc_t["eval_s"] += time.time() - te
                rd.metric({"monitor_eval": {"calib_search_head_ce": r["ce_nats_per_token"]},
                           "valid_tokens_seen": seen, "step": si})
                rd.log(f"monitor calib_search_head {r['ce_nats_per_token']:.4f}")
                next_eval += t["eval_every_tokens"]
            if t.get("state_every_seconds") and (time.time() - last_state_save > t["state_every_seconds"]
                                                 or si == steps_run - 1) and si < n_steps - 1:
                save_state(si, seen, acc_t, init_sha)
                last_state_save = time.time()
        if steps_run < n_steps:
            raise guard.GuardError(f"INTERRUPTED (max-steps) after step {steps_run - 1}; resumable")
        sync()
        pa, pr_ = limits.peak_gib()
        tput = acc_t["tok_after_warm"] / acc_t["step_s_after_warm"] if acc_t["step_s_after_warm"] > 0 else None
        summary = {"valid_train_tokens": seen, "steps": n_steps, "train_step_seconds": acc_t["step_s"],
                   "monitor_eval_seconds": acc_t["eval_s"], "train_valid_tokens_per_s_after_warmup": tput,
                   "peak_alloc_gib": pa, "peak_reserved_gib": pr_, "resumed_from": resumed_from}
        if hasattr(model, "alpha"):
            summary["alpha"] = model.alpha.detach().float().tolist()
        if cfg["eval"].get("final_calib_search"):
            ev = basic_eval(model, mon, device, cfg["eval"]["rows_per_batch"], None, autocast=use_amp)
            ev["collapse"] = collapse_check(ev)
            ev["evalset"] = "calib_search"
            guard.atomic_write_json(rd.path / "calib_search_eval.json", ev)
            summary.update({"collapse": ev["collapse"], "calib_search_ce": ev["ce_nats_per_token"],
                            "calib_search_valid_tokens": ev["n_valid_tokens"]})
            limits.check()
        if t.get("save_checkpoint"):
            require_disk(2 * 4 * sum(p.numel() for p in model.parameters()), "checkpoint")
            tmp = rd.path / ".model.pt.tmp"
            torch.save({"model": model.state_dict(), "config": cfg}, tmp)
            tmp.rename(rd.path / "model.pt")
            summary["checkpoint_sha256"] = guard.sha256_file(rd.path / "model.pt")
        summary["final_param_sha256"] = param_sha(model.named_parameters())
        status = "COMPLETE"
    except guard.GuardError as e:
        status = "CONTAMINATED" if "CONTAMINATED" in str(e) else ("INTERRUPTED" if "INTERRUPTED" in str(e)
                                                                  else "ABORTED")
        reason = str(e)
        if status == "INTERRUPTED":
            try:
                summary["final_param_sha256"] = param_sha(model.named_parameters())
            except Exception:  # noqa: BLE001
                pass
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
            guard.gpu_ledger(t_start_iso, guard.now(), rd.run_id, f"{key} train", status)
        rd.log(f"STATUS {status} {reason}")
        print(f"RUN_DIR {rd.path.relative_to(guard.ROOT)}")
    return status


if __name__ == "__main__":
    main()
