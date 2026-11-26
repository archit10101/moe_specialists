#!/usr/bin/env python
"""Train one condition (A-F) for a fixed valid-token budget. One condition per process.

Dry-run by default. CPU execution: --execute + MOE_RESEARCH_APPROVE=cpu-model.
GPU execution: launch through scripts/gpu_guarded.sh (occupancy gate, CUDA_VISIBLE_DEVICES=0) with
--device cuda --execute + MOE_RESEARCH_APPROVE=gpu.
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
from csr.config import build_condition_config, load_json  # noqa: E402


def parse_sets(sets):
    out = {}
    for s in sets or []:
        k, v = s.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def resolve_config(a):
    base = load_json(a.base)
    return build_condition_config(base, a.condition, parse_sets(a.set), run_name=a.run_name)


def load_seg(cfg, split):
    from csr.segcache import SegTable
    strat = cfg["segmentation"]["strategy"]
    if strat == "none":
        return None
    if strat == "random":  # F-v1 tables inside the semantic cache are SUPERSEDED (double document-end censoring)
        rc = cfg["segmentation"].get("random_cache")
        if not rc:
            raise guard.GuardError("F-v1 random cache superseded; an accepted F-v2 random_cache is required")
        rd_ = guard.inside_root(rc)
        rman = json.loads((rd_ / "manifest.json").read_text())
        if not rman.get("accepted") or rman["random_cache_id"] != cfg["segmentation"].get("random_cache_id") \
                or rman["spec"]["parent_semantic_cache"] != cfg["segmentation"]["cache_id"]:
            raise guard.GuardError("random cache not accepted or mismatched")
        return SegTable.load(rd_ / f"{split}_random.npz", strat)
    d = guard.inside_root(cfg["segmentation"]["cache"])
    man = json.loads((d / "manifest.json").read_text())
    if man["cache_id"] != cfg["segmentation"]["cache_id"]:
        raise guard.GuardError("stale/mismatched segmentation cache")
    if man["spec"]["warmup"] != cfg["routing"]["warmup"]:
        raise guard.GuardError("segmentation cache warmup differs from routing warmup")
    return SegTable.load(d / f"{split}_{strat}.npz", strat)


def check_data(cfg):
    d = guard.inside_root(cfg["data"]["dir"])
    man = json.loads((d / "manifest.json").read_text())
    if guard.sha256_json(man["splits"]) != cfg["data"]["splits_sha256"]:
        raise guard.GuardError("data manifest hash mismatch")
    return d


def plan(cfg, *, require_semantic_calibration=True):
    """Data/step plan for a run. require_semantic_calibration=False is used ONLY by calibrate.py, the operation
    that produces lambda/tau_s/tau_r; train and benchmark runs (run()) always use the default and fail closed."""
    import numpy as np
    from csr.data import TokenSplit, train_order
    from csr.train_lib import BatchSource, plan_steps
    d = check_data(cfg)
    if require_semantic_calibration and cfg["semantic"]["enabled"] and \
            any(cfg["semantic"][k] is None for k in ("lambda", "tau_s", "tau_r")):
        raise guard.GuardError("semantic loss enabled but lambda/tau_s/tau_r not calibrated")
    tr = TokenSplit.load(d, "train")
    src = BatchSource(tr, cfg["data"]["seq_len"], None, train_order(tr.n_docs, cfg["data"]["train_order_seed"]))
    t = cfg["train"]
    steps, total = plan_steps(src, t["micro_batch_rows"] * t["accum"], t["token_budget"])
    return d, tr, steps, total


def run(cfg, device_kind, execute, mode="train", max_steps=None):
    """mode: train | benchmark. Returns the run directory path (or None for dry-run)."""
    d, tr, steps, total = plan(cfg)
    if max_steps is not None:
        steps = steps[:max_steps]
        total = sum(_vt(s, tr) for s in steps)
    print(json.dumps({"condition": cfg["condition"], "mode": mode, "device": device_kind, "steps": len(steps),
                      "valid_train_tokens": total, "rows_per_step": len(steps[0]) if steps else 0}))
    action = "gpu" if device_kind == "cuda" else "cpu-model"
    if not guard.require_execution(execute, action):
        return None
    import numpy as np
    import torch
    from csr.data import TokenSplit, to_torch, train_order
    from csr.model import count_parameters, trainable_parameter_names
    from csr.train_lib import (BatchSource, build_model, evaluate, forward_losses, lr_at, make_optimizer,
                               router_grad_norms, step_objective)
    torch.set_num_threads(4)
    monitor = None
    if device_kind == "cuda":
        guard.occupancy_gate(f"{mode}:{cfg['run_name']}")
        device = guard.setup_cuda_device()
        monitor = guard.OccupancyMonitor(interval=15.0)
        monitor.start()
    else:
        device = torch.device("cpu")
    guard.register_self(f"{mode} {cfg['run_name']} on {device_kind}")
    rd = guard.RunDir(f"{mode}_{cfg['run_name']}", cfg)
    t_start_iso = guard.now()
    limits = guard.RunLimits(max_seconds=cfg["train"].get("max_seconds", guard.MAX_ATOMIC_SECONDS), monitor=monitor)
    status, reason = "INCOMPLETE", ""
    summary = {}
    try:
        torch.manual_seed(cfg["train"]["seed"])
        seg_tr = load_seg(cfg, "train")
        src = BatchSource(tr, cfg["data"]["seq_len"], seg_tr, train_order(tr.n_docs, cfg["data"]["train_order_seed"]))
        model = build_model(cfg, device)
        trainable = trainable_parameter_names(model)
        opt = make_optimizer(model, cfg, device)
        va = TokenSplit.load(d, cfg["eval"]["split"]) if cfg["eval"]["split"] != "test" else None
        assert va is not None, "test split is locked"
        seg_va = load_seg(cfg, cfg["eval"]["split"])
        eval_src = BatchSource(va, cfg["data"]["seq_len"], seg_va)
        facts = {"result_label": cfg.get("result_label", "experiment"),
                 "trainable_sha256": guard.sha256_json(trainable), "n_trainable": len(trainable),
                 "params": count_parameters(model), "planned_valid_tokens": total, "n_steps": len(steps),
                 "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
                 "torch": torch.__version__}
        guard.atomic_write_json(rd.path / "run_facts.json", facts)
        rd.log(json.dumps(facts))
        use_amp = device.type == "cuda" and cfg["train"]["precision"] == "bf16-autocast"
        t = cfg["train"]
        seen, t_loop = 0, time.time()
        evals = []
        next_eval = t["eval_every_tokens"]
        model.train()
        warm_t, warm_tok = None, 0
        for si, rows in enumerate(steps):
            lr = lr_at(si, len(steps), t)
            for g in opt.param_groups:
                g["lr"] = lr
            n_step = _vt(rows, tr)
            mbs = [rows[i:i + t["micro_batch_rows"]] for i in range(0, len(rows), t["micro_batch_rows"])]
            agg = {"lm": 0.0, "aux": 0.0, "z": 0.0, "sem": 0.0, "cov": 0, "n": 0}
            sem_stats = {}
            gn = None
            for mi, mb in enumerate(mbs):
                tb = to_torch(src.batch(mb), device)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                    parts, _ = forward_losses(model, tb, cfg, seg_tr)
                obj = step_objective(parts, n_step, cfg)
                if mi == 0 and t["grad_probe_every"] and si % t["grad_probe_every"] == 0:
                    gn = router_grad_norms(model, parts, cfg)
                obj.backward()
                agg["lm"] += float(parts.lm_sum)
                agg["aux"] += float(parts.aux) * parts.n_valid
                agg["z"] += float(parts.z) * parts.n_valid
                agg["sem"] += float(parts.sem) * parts.n_valid
                agg["cov"] += parts.coverage_constrained
                agg["n"] += parts.n_valid
                sem_stats = parts.sem_stats or sem_stats
            if agg["n"] != n_step:
                raise guard.GuardError(f"accounting mismatch {agg['n']} != {n_step}")
            if not math.isfinite(agg["lm"]) or not math.isfinite(agg["sem"]):
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
                       "aux": agg["aux"] / agg["n"], "z": agg["z"] / agg["n"], "sem": agg["sem"] / agg["n"],
                       "coverage": agg["cov"] / agg["n"], "grad_norm": float(gnorm), "elapsed_s": time.time() - t_loop,
                       "peak_alloc_gib": pa, "peak_reserved_gib": pr}
                if gn is not None:
                    lam = cfg["semantic"]["lambda"] if cfg["semantic"]["enabled"] else 0.0
                    rec.update({"router_gradnorm_lm": gn[0], "router_gradnorm_sem_weighted": lam * gn[1],
                                "router_gradnorm_ratio": (lam * gn[1] / gn[0]) if gn[0] > 0 else None})
                if sem_stats:
                    rec["sem_stats"] = sem_stats
                rd.metric(rec)
                rd.log(f"step {si} tok {seen} ce {rec['lm_ce']:.4f} cov {rec['coverage']:.3f} "
                       f"gn {float(gnorm):.2f} peak {pr:.2f}GiB")
            if mode == "train" and (seen >= next_eval or si == len(steps) - 1):
                ev = evaluate(model, eval_src, cfg, device, cfg["eval"]["rows_per_batch"], cfg["eval"]["max_rows"],
                              autocast=use_amp)
                ev_small = {k: v for k, v in ev.items() if k not in ("doc_nll", "doc_n")}
                evals.append({"valid_tokens_seen": seen, "step": si, "gpu_elapsed_s": time.time() - t_loop, **ev_small})
                rd.metric({"eval": evals[-1]})
                rd.log(f"eval tok {seen} val_ce {ev['ce_nats_per_token']:.4f} cov {ev['pool_constrained_coverage']:.3f}")
                next_eval += t["eval_every_tokens"]
                if si == len(steps) - 1:
                    guard.atomic_write_json(rd.path / "final_eval_docs.json", {"doc_nll": ev["doc_nll"], "doc_n": ev["doc_n"],
                                                                                 "result_label": cfg.get("result_label", "experiment")})
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_end = time.time()
        pa, pr = limits.peak_gib()
        tput = (seen - warm_tok) / (t_end - warm_t) if warm_t and seen > warm_tok else None
        summary = {"valid_train_tokens": seen, "steps": len(steps), "loop_seconds": t_end - t_loop,
                   "train_valid_tokens_per_s_after_warmup": tput, "peak_alloc_gib": pa, "peak_reserved_gib": pr,
                   "final_eval": evals[-1] if evals else None}
        if t.get("save_checkpoint"):
            guard.require_disk(2 * 4 * sum(p.numel() for p in model.parameters()), "checkpoint")
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
        guard.atomic_write_json(rd.path / "summary.json", {"status": status, "reason": reason,
                                                           "result_label": cfg.get("result_label", "experiment"), **summary})
        rd.set_status(status, reason=reason)
        if device_kind == "cuda":
            guard.gpu_ledger(t_start_iso, guard.now(), rd.run_id, mode, status)
        rd.log(f"STATUS {status} {reason}")
    return rd.path


def _vt(rows, split):
    from csr.train_lib import valid_tokens
    return valid_tokens(rows, split)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="configs/v2/base.json")
    ap.add_argument("--condition", required=True, choices=list("ABCDEF"))
    ap.add_argument("--set", action="append", help="dotted.key=json_value override")
    ap.add_argument("--run-name")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    cfg = resolve_config(a)
    run(cfg, a.device, a.execute, "train", a.max_steps)


if __name__ == "__main__":
    main()
