#!/usr/bin/env python
"""csr-conditional-local-attention-v1: train one config (FULL-TOKEN-4 baseline or a local / hybrid / conditional
candidate).

Dry-run by default; GPU only via scripts/gpu_guarded.sh with --device cuda --execute and MOE_RESEARCH_APPROVE=gpu.
Monitoring evaluations read only the first cfg.eval.max_rows packed rows of CALIB-SEARCH; at the end the full
CALIB-SEARCH set is evaluated. CALIB-SELECTION and the validation split are never read here; the test split is refused.
Resume: --resume <run dir> continues from its train_state.pt into a NEW run directory (same step sequence).
Throughput: optimizer-step wall time is accumulated separately (CUDA-synchronised around every monitoring
evaluation and checkpoint write), so training tokens/s exclude evaluation time.
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
    from csr.cla_run import check_cfg
    d = check_cfg(cfg)
    print(json.dumps({"config": str(cfg_path), "candidate": cfg["candidate"], "run_name": cfg["run_name"],
                      "device": a.device, "token_budget": cfg["train"]["token_budget"], "resume": a.resume}))
    if not guard.require_execution(a.execute, "gpu" if a.device == "cuda" else "cpu-model"):
        return
    sys.exit(0 if run(cfg, cfg_path, d, a) == "COMPLETE" else 1)


def make_optimizer(model, cfg, device):
    """AdamW exactly as the project recipe: decay for dim >= 2, none otherwise (gate MLP included, same lr)."""
    import torch
    t = cfg["train"]
    decay = [p for _, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for _, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    groups = [{"params": decay, "weight_decay": t["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=t["lr"], betas=tuple(t["betas"]), eps=1e-8, fused=(device.type == "cuda"))


def run(cfg, cfg_path, d, a):
    import numpy as np
    import torch
    from csr.cla_model import build_cla_model, cf_steps, count_cla_parameters
    from csr.cla_run import (collapse_check, cla_evaluate, code_hashes, eval_source, model_param_sha, param_sha,
                             require_disk, train_source)
    from csr.data import to_torch
    from csr.model import trainable_parameter_names
    from csr.train_lib import lr_at, valid_tokens
    torch.set_num_threads(4)
    monitor = None
    if a.device == "cuda":
        guard.occupancy_gate(f"clav1 train:{cfg['run_name']}")
        device = guard.setup_cuda_device()
        monitor = guard.OccupancyMonitor(interval=15.0)
        monitor.start()
    else:
        device = torch.device("cpu")
    guard.register_self(f"clav1 train {cfg['run_name']} on {a.device}")
    require_disk(600_000_000, f"clav1 train {cfg['run_name']}")
    resume_state, resumed_from = None, None
    if a.resume:
        prev = guard.inside_root(a.resume)
        if json.loads((prev / "config.json").read_text()) != cfg:
            raise guard.GuardError("resume: config differs from the interrupted run")
        resume_state = torch.load(prev / "train_state.pt", map_location="cpu", weights_only=False)
        resumed_from = str(prev.relative_to(guard.ROOT))
    rd = guard.RunDir(f"clav1_train_{cfg['run_name']}" + ("_resume" if a.resume else ""), cfg)
    t_start_iso, t_proc0 = guard.now(), time.time()
    limits = guard.RunLimits(max_seconds=cfg["train"].get("max_seconds", guard.MAX_ATOMIC_SECONDS), monitor=monitor)
    status, reason, summary = "INCOMPLETE", "", {}

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()
    try:
        tr, src, steps, total = train_source(cfg, d)
        n_steps = len(steps)
        steps_run = n_steps if a.max_steps is None else min(a.max_steps, n_steps)
        model = build_cla_model(cfg, device)
        plan = model.plan
        cond = plan.kind == "conditional"
        cfset = cf_steps(cfg["init"]["seed"], n_steps, plan.cf_frac) if cond else frozenset()
        init_sha = model_param_sha(model)
        init_backbone_sha = param_sha({k: v for k, v in model.named_parameters() if not k.startswith("gates.")})
        opt = make_optimizer(model, cfg, device)
        t = cfg["train"]
        mon = eval_source(cfg, "calib_search", d)[1]
        start_step, seen = 0, 0
        acc_t = {"step_s": 0.0, "step_s_after_warm": 0.0, "tok_after_warm": 0, "cf_step_s": 0.0, "n_cf_done": 0,
                 "eval_s": 0.0}
        if resume_state is not None:
            model.load_state_dict(resume_state["model"])
            opt.load_state_dict(resume_state["opt"])
            start_step, seen = resume_state["next_step"], resume_state["seen"]
            acc_t.update(resume_state.get("acc_t", {}))
            if resume_state["init_param_sha256"] != init_sha:
                raise guard.GuardError("resume: initialization hash differs")
            rd.log(f"resumed from {resumed_from} at step {start_step} seen {seen}")
        mcfg = model.cfg
        n_params = count_cla_parameters(model)
        facts = {"trainable_sha256": guard.sha256_json(trainable_parameter_names(model)), "params": n_params,
                 "planned_valid_tokens": total, "n_steps": n_steps, "config_sha256": guard.sha256_file(cfg_path),
                 "config_path": str(cfg_path.relative_to(guard.ROOT)), "attention": plan.to_dict(),
                 "layer_modes": [list(m) for m in model.modes], "backend": model.backend,
                 "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
                 "torch": torch.__version__, "init_param_sha256": init_sha,
                 "init_backbone_param_sha256": init_backbone_sha, "resumed_from": resumed_from,
                 "n_cf_steps": len(cfset), "cf_steps_sha256": guard.sha256_json(sorted(cfset)),
                 "train_rows_sha256": guard.sha256_json(steps), "code_sha256": code_hashes(),
                 "micro_batch_rows": t["micro_batch_rows"], "accum": t["accum"]}
        guard.atomic_write_json(rd.path / "run_facts.json", facts)
        rd.log(json.dumps({k: v for k, v in facts.items() if k != "code_sha256"}))
        use_amp = device.type == "cuda" and t["precision"] == "bf16-autocast"
        model.train()
        if cond and start_step == 0:
            tb0 = to_torch(src.batch(steps[0][:t["micro_batch_rows"]]), device)
            with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                model(tb0, calibrate=True)
            model.thr_ready.fill_(True)
            rd.log(f"thresholds initialised by a no-grad pre-pass on the first training micro-batch: "
                   f"{model.thr.tolist()}")
            rd.metric({"thr_init": model.thr.tolist()})
        next_eval = (seen // t["eval_every_tokens"] + 1) * t["eval_every_tokens"]
        warm = t.get("throughput_warmup_steps", 10)
        last_state_save = time.time()
        sync()
        for si in range(start_step, steps_run):
            ts0 = time.time()
            rows = steps[si]
            lr = lr_at(si, n_steps, t)
            for g in opt.param_groups:
                g["lr"] = lr
            is_cf = si in cfset
            n_step = valid_tokens(rows, tr)
            mbs = [rows[i:i + t["micro_batch_rows"]] for i in range(0, len(rows), t["micro_batch_rows"])]
            agg = {"lm": 0.0, "aux": 0.0, "z": 0.0, "n": 0}
            z_lists = [[] for _ in range(mcfg.n_layer)] if cond else None
            n_nonpad = 0
            sel_cnt = np.zeros(mcfg.n_layer)
            for mb in mbs:
                tb = to_torch(src.batch(mb), device)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                    out = model(tb, cf=is_cf)
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
                if cond:
                    n_nonpad += int((tb["doc_ids"] >= 0).sum())
                    for li, gd in enumerate(out.gate):
                        z_lists[li].append(gd["z"][gd["elig"]])
                        sel_cnt[li] += float(gd["sel"].sum())
            if agg["n"] != n_step:
                raise guard.GuardError(f"accounting mismatch {agg['n']} != {n_step}")
            if not all(math.isfinite(agg[k]) for k in ("lm", "aux", "z")):
                raise FloatingPointError("non-finite loss")
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), t["clip"])
            if not torch.isfinite(gnorm):
                raise FloatingPointError("non-finite grad norm")
            gate_gn = None
            if cond:
                gg = [p.grad for n, p in model.named_parameters() if n.startswith("gates.") and p.grad is not None]
                gate_gn = float(torch.sqrt(sum((x.float() ** 2).sum() for x in gg))) if gg else 0.0
            opt.step()
            opt.zero_grad(set_to_none=True)
            rates = model.thr_update(z_lists, n_nonpad) if cond else None
            seen += n_step
            sync()
            dt = time.time() - ts0
            acc_t["step_s"] += dt
            if is_cf:
                acc_t["cf_step_s"] += dt
                acc_t["n_cf_done"] += 1
            if si >= warm:
                acc_t["step_s_after_warm"] += dt
                acc_t["tok_after_warm"] += n_step
            limits.check()
            if si % t["log_every"] == 0 or si == n_steps - 1 or is_cf:
                pa, pr_ = limits.peak_gib()
                rec = {"step": si, "valid_tokens_seen": seen, "lr": lr, "lm_ce": agg["lm"] / agg["n"],
                       "aux": agg["aux"] / agg["n"], "z": agg["z"] / agg["n"], "grad_norm": float(gnorm),
                       "step_s": dt, "cf": is_cf, "peak_alloc_gib": pa, "peak_reserved_gib": pr_}
                if cond:
                    rec.update({"thr": model.thr.tolist(), "rate_after_update": rates,
                                "rate_step": (sel_cnt / max(n_nonpad, 1)).tolist(), "gate_grad_norm": gate_gn})
                rd.metric(rec)
                rd.log(f"step {si} tok {seen} ce {rec['lm_ce']:.4f} gn {float(gnorm):.2f} {dt * 1e3:.0f}ms"
                       + (f" cf={is_cf} rate={np.mean(rec['rate_step']):.3f}" if cond else ""))
            if seen >= next_eval or si == n_steps - 1:
                te = time.time()
                r = cla_evaluate(model, mon, device, cfg["eval"]["rows_per_batch"], cfg["eval"]["max_rows"],
                                 autocast=use_amp, diagnostics=False)
                sync()
                acc_t["eval_s"] += time.time() - te
                rd.metric({"monitor_eval": {"calib_search_head": {k: v for k, v in r.items()
                                                                  if k not in ("doc_nll", "doc_n")}},
                           "valid_tokens_seen": seen, "step": si})
                rd.log(f"monitor calib_search_head {r['ce_nats_per_token']:.4f}")
                next_eval += t["eval_every_tokens"]
            if t.get("state_every_seconds") and (time.time() - last_state_save > t["state_every_seconds"]
                                                 or si == steps_run - 1) and si < n_steps - 1:
                tmp = rd.path / ".train_state.pt.tmp"
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "next_step": si + 1, "seen": seen,
                            "init_param_sha256": init_sha, "acc_t": acc_t}, tmp)
                tmp.replace(rd.path / "train_state.pt")
                last_state_save = time.time()
        if steps_run < n_steps:
            raise guard.GuardError(f"INTERRUPTED (max-steps) after step {steps_run - 1}; resumable")
        sync()
        pa, pr_ = limits.peak_gib()
        tput = acc_t["tok_after_warm"] / acc_t["step_s_after_warm"] if acc_t["step_s_after_warm"] > 0 else None
        summary = {"valid_train_tokens": seen, "steps": n_steps, "optimizer_steps": n_steps,
                   "n_cf_steps": acc_t["n_cf_done"], "train_step_seconds": acc_t["step_s"],
                   "cf_step_seconds": acc_t["cf_step_s"], "monitor_eval_seconds": acc_t["eval_s"],
                   "train_valid_tokens_per_s_after_warmup": tput,
                   "train_steps_per_s_after_warmup": (n_steps - warm) / acc_t["step_s_after_warm"]
                   if acc_t["step_s_after_warm"] > 0 else None,
                   "peak_alloc_gib": pa, "peak_reserved_gib": pr_, "resumed_from": resumed_from,
                   "final_thresholds": model.thr.tolist()}
        if cfg["eval"].get("final_calib_search"):
            ev = cla_evaluate(model, mon, device, cfg["eval"]["rows_per_batch"], None, autocast=use_amp)
            ev["collapse"] = collapse_check(ev)
            ev["evalset"] = "calib_search"
            guard.atomic_write_json(rd.path / "calib_search_eval.json", ev)
            summary["collapse"] = ev["collapse"]
            summary["calib_search_ce"] = ev["ce_nats_per_token"]
            summary["calib_search_valid_tokens"] = ev["n_valid_tokens"]
            summary["calib_search_eval_seconds"] = ev["eval_seconds"]
            summary["global_query_rate"] = ev.get("global_query_rate_mean")
            limits.check()
        if t.get("save_checkpoint"):
            require_disk(2 * 4 * sum(p.numel() for p in model.parameters()), "clav1 checkpoint")
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
            guard.gpu_ledger(t_start_iso, guard.now(), rd.run_id, "clav1 train", status)
        rd.log(f"STATUS {status} {reason}")
        print(f"RUN_DIR {rd.path.relative_to(guard.ROOT)}")
    return status


if __name__ == "__main__":
    main()
