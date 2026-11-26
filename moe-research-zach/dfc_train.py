#!/usr/bin/env python
"""csr-domain-forecast-curriculum-v1: train the shared draft (kind=draft) or one main-model run (kind=main).

Recipe = the project training loop (AdamW, WSD, token-weighted CE + aux + z, bf16 autocast, fp32 router).
Main runs: per-row curriculum strength from the plan (P1 = 1, P2 linear 1 -> 0, P3 = 0); checkpoints at 20/40/50/75/100%
of 112.5M tokens (curriculum + TOKEN-MAIN) or final only (ALL-DATA, COMPUTE); CALIB-MON learning curves every 5.625M
tokens (active + neutral). Resumable in place: one rotating resume slot (.tmp/domain-forecast-curriculum-v1/
resume_slot.pt) saved every 10 min, at every checkpoint and on interruption. Exit codes: 0 COMPLETE, 75 INTERRUPTED
(resumable), 76 OOM (driver halves micro-batch, doubles accumulation, resumes), 1 other failure.
Dry-run by default; GPU only via scripts/gpu_guarded.sh with --device cuda --execute.
"""
import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402

SLOT_NAME = "resume_slot.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--run-dir", help="existing run dir to resume in place")
    ap.add_argument("--micro-batch-rows", type=int)
    ap.add_argument("--max-steps", type=int, help="stop (INTERRUPTED, resumable) after this many total steps")
    ap.add_argument("--max-seconds", type=float, default=5100)
    ap.add_argument("--slot-dir", default=".tmp/domain-forecast-curriculum-v1")
    ap.add_argument("--runs-dir")
    ap.add_argument("--no-ledger", action="store_true")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    cfg_path = guard.inside_root(a.config)
    cfg = json.loads(cfg_path.read_text())
    from csr.dfc_lib import prereg_sha
    if cfg.get("prereg_sha256") != prereg_sha():
        raise guard.GuardError("config prereg hash does not match the frozen prereg")
    print(json.dumps({"config": str(cfg_path), "run_name": cfg["run_name"], "device": a.device,
                      "resume": a.run_dir}))
    if not guard.require_execution(a.execute, "gpu" if a.device == "cuda" else "cpu-model"):
        return
    rc = run(cfg, cfg_path, a)
    sys.exit(rc)


def make_optimizer(model, t, device):
    import torch
    decay = [p for _, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for _, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    groups = [{"params": decay, "weight_decay": t["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=t["lr"], betas=tuple(t["betas"]), eps=1e-8, fused=(device.type == "cuda"))


class InPlaceRun:
    def __init__(self, path):
        self.path = path
        self.run_id = path.name

    def set_status(self, status, **extra):
        guard.atomic_write_json(self.path / "status.json", {"status": status, "time": guard.now(), **extra},
                                overwrite=True)

    def log(self, msg):
        line = f"{guard.now()} {msg}"
        print(line, flush=True)
        with open(self.path / "log.txt", "a") as f:
            f.write(line + "\n")

    def metric(self, rec):
        with open(self.path / "metrics.jsonl", "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")


def param_sha(named):
    import hashlib
    h = hashlib.sha256()
    for n, p in sorted(named, key=lambda x: x[0]):
        h.update(n.encode())
        h.update(p.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def run(cfg, cfg_path, a):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from csr.dfc_lib import (CKPT_FRACS, MAIN_TOKENS, MON_EVERY, MON_ROWS_PER_DOM, code_hashes, eval_rows,
                             load_cache, load_split, make_batch, plan_for, to_torch)
    from csr.dfc_model import CONDITIONS, build_dfc_model, build_draft
    from csr.model import count_parameters
    from csr.train_lib import lr_at
    torch.set_num_threads(4)
    monitor = None
    if a.device == "cuda":
        guard.occupancy_gate(f"dfc train:{cfg['run_name']}")
        device = guard.setup_cuda_device()
        monitor = guard.OccupancyMonitor(interval=15.0)
        monitor.start()
    else:
        device = torch.device("cpu")
    guard.register_self(f"dfc train {cfg['run_name']} on {a.device}")
    slot_dir = guard.inside_root(a.slot_dir)
    slot_dir.mkdir(parents=True, exist_ok=True)
    slot = slot_dir / SLOT_NAME
    t = cfg["train"]
    kind = cfg["kind"]
    if a.run_dir:
        rd = InPlaceRun(guard.inside_root(a.run_dir))
        if json.loads((rd.path / "config.json").read_text()) != cfg:
            raise guard.GuardError("resume: config differs")
    else:
        r0 = guard.RunDir(f"dfc_{cfg['run_name']}", cfg, base=guard.inside_root(a.runs_dir) if a.runs_dir else None)
        rd = InPlaceRun(r0.path)
    rd.set_status("RUNNING")
    t_start_iso, t_proc0 = guard.now(), time.time()
    limits = guard.RunLimits(max_seconds=a.max_seconds, monitor=monitor)
    status, reason, summary, rc = "INCOMPLETE", "", {}, 1
    mbr = a.micro_batch_rows or t["micro_batch_rows"]
    use_amp = device.type == "cuda" and t["precision"] == "bf16-autocast"

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    model = opt = None
    clean = [True]  # False only between opt.step() and the state update (never save a slot there)
    st = {"next_step": 0, "seen": 0, "acc_t": {"step_s": 0.0, "step_s_after_warm": 0.0, "tok_after_warm": 0,
                                               "mon_s": 0.0, "ckpt_s": 0.0},
          "ckpts": {}, "next_mon": MON_EVERY, "attempt": 0}

    def save_slot():
        tmp = slot_dir / f".{SLOT_NAME}.tmp"
        torch.save({"run_dir": str(rd.path), "config": cfg, "model": model.state_dict(), "opt": opt.state_dict(),
                    "st": st, "init_param_sha256": init_sha, "time": guard.now()}, tmp)
        os.replace(tmp, slot)
        rd.log(f"resume slot saved at next_step {st['next_step']} seen {st['seen']}")
    try:
        if kind == "draft":
            from csr.dfc_lib import Plan, interleave, T_SEQ, N_DOM
            from csr.data import pack_rows
            sp, doms, _ = load_split("draft")
            L = sp.lengths()
            per = []
            for d in range(N_DOM):
                idx = np.nonzero(doms == d)[0]
                idx = idx[np.random.default_rng([t["order_seed"], d, 0]).permutation(len(idx))]
                per.append(pack_rows(L, idx.tolist(), T_SEQ))
            rows, _ = interleave(per)
            plan = Plan(sp, doms, rows, [0.0] * len(rows), [0] * len(rows), None, "draft")
            torch.manual_seed(cfg["seed"])
            model = build_draft(cfg["seed"], device)
            curriculum = False
            mon_cache = None
        else:
            plan = plan_for(cfg)
            model = build_dfc_model(cfg["model"], cfg["condition"], cfg["seed"], device)
            curriculum = CONDITIONS[cfg["condition"]][0]
            src = CONDITIONS[cfg["condition"]][3]
            mon_cache = load_cache("calib", src) if src else None
        if plan.total != t["token_budget"]:
            raise guard.GuardError(f"plan tokens {plan.total} != budget {t['token_budget']}")
        n_steps = len(plan.steps)
        init_sha = param_sha(model.named_parameters())
        opt = make_optimizer(model, t, device)
        csp, cdoms, _ = load_split("calib")
        mon_rows, _ = eval_rows(csp, cdoms, MON_ROWS_PER_DOM)
        if a.run_dir:
            ck = torch.load(slot, map_location="cpu", weights_only=False)
            if ck["run_dir"] != str(rd.path) or ck["config"] != cfg:
                raise guard.GuardError("resume slot belongs to another run")
            if ck["init_param_sha256"] != init_sha:
                raise guard.GuardError("resume: initialization hash differs")
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            st = ck["st"]
            rd.log(f"RESUMED in place at step {st['next_step']} seen {st['seen']} (slot {ck['time']})")
        st["attempt"] += 1
        thresholds = {}
        if kind == "main":
            if cfg["checkpoints"] == "fractions":
                thresholds = {k: round(f * MAIN_TOKENS) for k, f in CKPT_FRACS.items()}
            else:
                thresholds = {"final": plan.total}
        else:
            thresholds = {"final": plan.total}
        n_params = count_parameters(model)
        if hasattr(model, "added_parameter_count"):
            n_params["added"] = model.added_parameter_count()
        facts = {"attempt": st["attempt"], "params": n_params, "planned_valid_tokens": plan.total, "n_steps": n_steps,
                 "n_rows": len(plan.rows), "plan_fingerprint": plan.fingerprint(),
                 "per_domain_phase_tokens": plan.per_domain_phase_tokens() if kind == "main" else None,
                 "config_sha256": guard.sha256_file(cfg_path), "config_path": str(cfg_path.relative_to(guard.ROOT)),
                 "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
                 "torch": torch.__version__, "init_param_sha256": init_sha,
                 "init_backbone_param_sha256": param_sha(model.backbone_state().items())
                 if hasattr(model, "backbone_state") else init_sha,
                 "code_sha256": code_hashes(), "micro_batch_rows": mbr, "rows_per_step": len(plan.steps[0]),
                 "accum": math.ceil(len(plan.steps[0]) / mbr), "thresholds": thresholds,
                 "compute_cycles": getattr(plan, "cycles", None)}
        guard.atomic_write_json(rd.path / f"run_facts_attempt{st['attempt']}.json", facts)
        rd.log(json.dumps({k: v for k, v in facts.items() if k not in ("code_sha256", "per_domain_phase_tokens")}))

        def monitor_eval(s_now):
            te = time.time()
            model.eval()
            res = {}
            modes = [("neutral", 0.0)] + ([("active", s_now)] if curriculum and s_now > 0 else [])
            with torch.no_grad():
                for name, sv in modes:
                    tot, n = 0.0, 0
                    for i in range(0, len(mon_rows), 16):
                        rr = mon_rows[i:i + 16]
                        tb = to_torch(make_batch(rr, csp, cdoms, mon_cache if sv > 0 else None), device)
                        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                            out = model(tb, strength=sv) if kind == "main" else model(tb)
                        tgt = tb["targets"]
                        nll = F.cross_entropy(out.logits.float().view(-1, out.logits.shape[-1]), tgt.view(-1),
                                              ignore_index=-100, reduction="sum")
                        tot += float(nll)
                        n += int((tgt != -100).sum())
                    res[name] = tot / n
            model.train()
            sync()
            st["acc_t"]["mon_s"] += time.time() - te
            return res

        def save_ckpt(tag):
            p = rd.path / f"ckpt_{tag}.pt"
            if p.exists():
                rd.log(f"checkpoint {tag} already exists (saved before an interruption); kept unchanged")
                return
            tc = time.time()
            guard.require_disk(int(1.2 * 4 * sum(x.numel() for x in model.parameters())), f"dfc ckpt {tag}")
            tmp = rd.path / f".ckpt_{tag}.pt.tmp"
            torch.save({"model": model.state_dict(), "config": cfg, "seen": st["seen"], "step": st["next_step"] - 1,
                        "tag": tag}, tmp)
            os.replace(tmp, p)
            os.chmod(p, 0o400)
            st["ckpts"][tag] = {"path": str(p.relative_to(guard.ROOT)), "sha256": guard.sha256_file(p),
                                "seen": st["seen"], "step": st["next_step"] - 1}
            st["acc_t"]["ckpt_s"] += time.time() - tc
            rd.log(f"checkpoint {tag} at seen {st['seen']}")

        model.train()
        warm = t.get("throughput_warmup_steps", 10)
        if not a.run_dir:
            save_slot()  # step-0 slot: an OOM / interruption before the first periodic save is resumable
        last_slot = time.time()
        sync()
        si = st["next_step"]
        while si < n_steps:
            if a.max_steps is not None and si >= a.max_steps:
                save_slot()
                raise guard.GuardError(f"INTERRUPTED (max-steps) before step {si}; resumable")
            ts0 = time.time()
            ridx = plan.steps[si]
            lr = lr_at(si, n_steps, t)
            for g in opt.param_groups:
                g["lr"] = lr
            n_step = plan.step_tokens(si)
            agg = {"lm": 0.0, "aux": 0.0, "z": 0.0, "n": 0}
            smean = float(plan.row_strength[ridx].mean())
            for i in range(0, len(ridx), mbr):
                mb = ridx[i:i + mbr]
                rows = [plan.rows[r] for r in mb]
                srow = plan.row_strength[mb]
                need_fc = curriculum and kind == "main" and bool((srow > 0).any()) and plan.cache is not None
                tb = to_torch(make_batch(rows, plan.split, plan.doms, plan.cache if need_fc else None), device)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                    if kind == "main":
                        out = model(tb, strength=torch.as_tensor(srow, dtype=torch.float32, device=device))
                    else:
                        out = model(tb)
                logits = out.logits.float()
                tgt = tb["targets"]
                lm_sum = F.cross_entropy(logits.view(-1, logits.shape[-1]), tgt.view(-1), ignore_index=-100,
                                         reduction="sum")
                n_valid = int((tgt != -100).sum())
                w = n_valid / n_step
                obj = lm_sum / n_step
                if kind == "main":
                    obj = obj + w * (t["aux_coeff"] * out.aux_loss + t["z_coeff"] * out.z_loss)
                obj.backward()
                agg["lm"] += float(lm_sum)
                agg["aux"] += float(out.aux_loss) * n_valid
                agg["z"] += float(out.z_loss) * n_valid
                agg["n"] += n_valid
                del out, logits, lm_sum, obj
            if agg["n"] != n_step:
                raise guard.GuardError(f"accounting mismatch {agg['n']} != {n_step}")
            if not all(math.isfinite(agg[k]) for k in ("lm", "aux", "z")):
                raise FloatingPointError(f"non-finite loss at step {si}")
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), t["clip"])
            if not torch.isfinite(gnorm):
                raise FloatingPointError(f"non-finite grad norm at step {si}")
            clean[0] = False
            opt.step()
            opt.zero_grad(set_to_none=True)
            st["seen"] += n_step
            st["next_step"] = si + 1
            clean[0] = True
            sync()
            dt = time.time() - ts0
            st["acc_t"]["step_s"] += dt
            if si >= warm:
                st["acc_t"]["step_s_after_warm"] += dt
                st["acc_t"]["tok_after_warm"] += n_step
            if si % t["log_every"] == 0 or si == n_steps - 1:
                pa, pr_ = limits.peak_gib()
                rec = {"step": si, "valid_tokens_seen": st["seen"], "lr": lr, "strength_mean": smean,
                       "lm_ce": agg["lm"] / agg["n"], "aux": agg["aux"] / agg["n"], "z": agg["z"] / agg["n"],
                       "grad_norm": float(gnorm), "step_s": dt, "peak_alloc_gib": pa, "attempt": st["attempt"]}
                if hasattr(model, "scales"):
                    sf, sp_ = model.scales()
                    rec["scale_f"] = sf.detach().float().tolist()
                    rec["scale_p"] = sp_.detach().float().tolist()
                rd.metric(rec)
                rd.log(f"step {si}/{n_steps} tok {st['seen']} s {smean:.3f} ce {rec['lm_ce']:.4f} "
                       f"gn {float(gnorm):.2f} {dt * 1e3:.0f}ms")
            saved = False
            for tag, thr in thresholds.items():
                if tag not in st["ckpts"] and st["seen"] >= thr:
                    save_ckpt(tag)
                    saved = True
            if st["seen"] >= st["next_mon"] or si == n_steps - 1:
                s_now = float(plan.row_strength[ridx[-1]])
                r = monitor_eval(s_now)
                rd.metric({"monitor": r, "valid_tokens_seen": st["seen"], "step": si, "strength_now": s_now,
                           "attempt": st["attempt"]})
                rd.log(f"monitor {json.dumps(r)}")
                while st["next_mon"] <= st["seen"]:
                    st["next_mon"] += MON_EVERY
            limits.check()
            if si < n_steps - 1 and (saved or time.time() - last_slot > t["state_every_seconds"]):
                save_slot()
                last_slot = time.time()
            si += 1
        sync()
        acc = st["acc_t"]
        tput = acc["tok_after_warm"] / acc["step_s_after_warm"] if acc["step_s_after_warm"] > 0 else None
        final_tag = "p100" if "p100" in thresholds else "final"
        summary = {"valid_train_tokens": st["seen"], "steps": n_steps, "train_step_seconds": acc["step_s"],
                   "monitor_eval_seconds": acc["mon_s"], "ckpt_seconds": acc["ckpt_s"],
                   "train_valid_tokens_per_s_after_warmup": tput, "checkpoints": st["ckpts"],
                   "final_checkpoint": st["ckpts"][final_tag]["path"], "attempts": st["attempt"],
                   "final_param_sha256": param_sha(model.named_parameters()), "init_param_sha256": init_sha,
                   "peak_alloc_gib": limits.peak_gib()[0], "micro_batch_rows_last": mbr}
        if hasattr(model, "scales"):
            sf, sp_ = model.scales()
            summary["scale_f"] = sf.detach().float().tolist()
            summary["scale_p"] = sp_.detach().float().tolist()
        status, rc = "COMPLETE", 0
    except guard.GuardError as e:
        msg = str(e)
        if "CONTAMINATED" in msg:
            status, rc = "CONTAMINATED", 75  # foreign GPU process: resumable from the last clean slot
        elif "VRAM near ceiling" in msg:
            status, rc = "OOM", 76
        elif any(k in msg for k in ("wall-clock", "INTERRUPTED", "disk reserve")):
            status, rc = "INTERRUPTED", 75
        else:
            status, rc = "FAILED", 1
        reason = msg
        if status == "INTERRUPTED" and model is not None and opt is not None and "max-steps" not in str(e):
            try:
                save_slot()
            except Exception as e2:  # noqa: BLE001
                rd.log(f"slot save failed: {e2}")
    except KeyboardInterrupt:
        status, reason, rc = "INTERRUPTED", "SIGINT", 75
        if model is not None and opt is not None and clean[0]:
            try:
                save_slot()
            except Exception as e2:  # noqa: BLE001
                rd.log(f"slot save failed: {e2}")
    except torch.cuda.OutOfMemoryError as e:
        status, reason, rc = "OOM", f"OutOfMemoryError: {str(e)[:300]}", 76
    except Exception as e:  # noqa: BLE001
        status, reason, rc = "FAILED", f"{type(e).__name__}: {e}", 1
        rd.log(traceback.format_exc())
    finally:
        limits.close()
        if monitor is not None:
            monitor.stop()
            if monitor.contaminated:  # numerics unaffected; only timing (compute matching) is flagged
                summary["foreign_gpu_process_seen"] = monitor.detail
        summary["process_seconds"] = time.time() - t_proc0
        guard.atomic_write_json(rd.path / f"summary_attempt{st['attempt']}.json",
                                {"status": status, "reason": reason, **summary}, overwrite=True)
        if status == "COMPLETE":
            guard.atomic_write_json(rd.path / "summary.json", {"status": status, "reason": reason, **summary})
        rd.set_status(status, reason=reason, attempt=st["attempt"])
        if a.device == "cuda" and not a.no_ledger:
            guard.gpu_ledger(t_start_iso, guard.now(), rd.run_id, f"dfc train {kind}", status)
        rd.log(f"STATUS {status} {reason}")
        print(f"RUN_DIR {rd.path.relative_to(guard.ROOT)}")
    return rc


if __name__ == "__main__":
    main()
