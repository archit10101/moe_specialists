#!/usr/bin/env python
"""csr-forecast-specialist-pretraining-v2: train a draft (kind draft_shared | draft_cont) or one main run (kind main).

Resumable in place (Operational Amendment 1): verified, atomically written resumable checkpoints
(.tmp/fspv2/slots/<run_id>/resume_s<step>.pt, latest two kept, ledger artifacts/.../checkpoint_ledger.tsv) every
max(1, floor(10% of steps)) steps or 15 min, at milestones and at every stop. The data stream, curriculum, teacher-forcing
draws and LR are pure functions of the step index, so a resumed run replays exactly. External cancellation (SIGINT /
SIGTERM / SIGHUP, the STOP marker, a foreign GPU process) stops at the next optimizer-step boundary with a checkpoint.
Exit codes: 0 COMPLETE, 75 INTERRUPTED (internal leg limit; resumable), 76 OOM, 77 EXTERNALLY_STOPPED (resumable, never
auto-restarted), 78 BLOCKED (foreign GPU process before start), 1 failure.
Dry-run by default; GPU only via scripts/gpu_guarded.sh with --device cuda --execute.
"""
import argparse
import json
import math
import os
import signal
import sys
import threading
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402

SLOTS = ".tmp/fspv2/slots"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--run-dir")
    ap.add_argument("--micro-batch-rows", type=int)
    ap.add_argument("--max-steps", type=int, help="stop INTERRUPTED after this many total steps (tests)")
    ap.add_argument("--max-seconds", type=float, default=4700)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--test-signal-at-step", type=int, help="test only: send SIGINT to self before this step")
    ap.add_argument("--test-kill-at-step", type=int, help="test only: SIGKILL self before this step (no checkpoint)")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    cfg_path = guard.inside_root(a.config)
    cfg = json.loads(cfg_path.read_text())
    from csr.fsp_lib import protocol_sha
    if cfg.get("protocol_sha256") != protocol_sha():
        raise guard.GuardError("config protocol hash does not match the frozen protocol")
    print(json.dumps({"config": str(cfg_path), "run_name": cfg["run_name"], "device": a.device, "resume": a.run_dir}))
    if not guard.require_execution(a.execute, "gpu" if a.device == "cuda" else "cpu-model"):
        return
    try:
        rc = run(cfg, a)
    except guard.GuardError as e:
        if str(e).startswith("BLOCKED"):  # foreign GPU process present before start: yield (amendment B6)
            print(f"EXTERNAL BLOCKED {e}", flush=True)
            sys.exit(78)
        raise
    sys.exit(rc)


class StopFlag:
    """Cooperative external-stop flag (amendment B). Signals only set the flag during training (acted on at the next
    optimizer-step boundary); during the end-of-run evaluation (no training state is mutated) they abort it."""

    def __init__(self, monitor):
        self.reason, self.in_eval, self.monitor, self._pending = None, False, monitor, None
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGUSR1):
            signal.signal(sig, self._handler)
        self._stop = threading.Event()
        threading.Thread(target=self._watch, daemon=True).start()

    def _handler(self, signum, frame):
        if self.reason is None:
            self.reason = "signal " + signal.Signals(signum).name if signum != signal.SIGUSR1 else self._pending
        if self.in_eval:
            raise KeyboardInterrupt(self.reason)

    def _watch(self):
        from csr.fsp_ckpt import STOP_MARKER
        while not self._stop.wait(5.0):
            why = None
            if STOP_MARKER.exists():
                why = f"stop marker {STOP_MARKER.name}"
            elif self.monitor is not None and self.monitor.contaminated:
                why = f"foreign GPU process: {self.monitor.detail}"
            if why and self.reason is None:
                self._pending = why
                if self.in_eval:
                    os.kill(os.getpid(), signal.SIGUSR1)
                else:
                    self.reason = why

    def close(self):
        self._stop.set()


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


def build_stream(cfg):
    """Returns (split_like, steps(list of row lists), planned valid tokens)."""
    import numpy as np
    from csr.fsp_lib import Split, main_order, pack_rows, plan_steps
    kind = cfg["kind"]
    rps = cfg["train"]["rows_per_step"]
    if kind == "draft_shared":
        sp = Split("draft_shared")
        rows = pack_rows(sp.lengths, np.random.default_rng(7).permutation(sp.n_docs).tolist())
        steps = [rows[s:s + rps] for s in range(0, len(rows) - rps + 1, rps)]
        return sp, steps, int(sum(sp.lengths[d] - 1 for r in rows[:len(steps) * rps] for d in r))
    if kind == "draft_cont":
        sp = Split("draft_cont")
        idx = [i for i in np.random.default_rng(8).permutation(sp.n_docs).tolist() if sp.domains[i] == cfg["domain"]]
        rows = pack_rows(sp.lengths, idx)
        steps = [rows[s:s + rps] for s in range(0, len(rows) - rps + 1, rps)]
        return sp, steps, int(sum(sp.lengths[d] - 1 for r in rows[:len(steps) * rps] for d in r))
    if cfg.get("stream", "main") == "main":
        sp = Split("main")
        rows = pack_rows(sp.lengths, main_order(sp))
        steps, total = plan_steps(rows, sp.lengths, rps, cfg["budget_tokens"])
        return sp, steps, total
    # TOKEN-COMPUTE: main stream, then the balanced draft corpora, then a second pass over main
    from csr.fsp_lib import ConcatSplit
    cs = ConcatSplit(["main", "draft_shared", "draft_cont"])
    m = cs.parts[0]
    rows = pack_rows(cs.lengths, main_order(m))
    extra = [cs.base[1] + i for i in range(cs.parts[1].n_docs)] + [cs.base[2] + i for i in range(cs.parts[2].n_docs)]
    extra = [extra[i] for i in np.random.default_rng(2027).permutation(len(extra))]
    rows = rows + pack_rows(cs.lengths, extra) + pack_rows(cs.lengths, main_order(m))
    steps, total = plan_steps(rows, cs.lengths, rps, cfg["budget_tokens"])
    return cs, steps, total


def run(cfg, a):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from csr.fsp_lib import (ForecastCache, build_draft, causal_scalars, curriculum, draft_hidden, lr_at, make_batch,
                             to_torch)
    from csr.fsp_model import FSPConfig, FSPModel, TEACHER_PREFIX
    torch.set_num_threads(4)
    monitor = None
    if a.device == "cuda":
        guard.occupancy_gate(f"fsp train:{cfg['run_name']}")
        device = guard.setup_cuda_device()
        monitor = guard.OccupancyMonitor(interval=15.0)
        monitor.start()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        device = torch.device("cpu")
    guard.register_self(f"fsp train {cfg['run_name']} on {a.device}")
    from csr import fsp_ckpt
    stopf = StopFlag(monitor)
    slot_dir = guard.inside_root(SLOTS)
    slot_dir.mkdir(parents=True, exist_ok=True)
    slot = slot_dir / f"{cfg['run_name']}.pt"  # legacy single slot (read-only fallback)
    t = cfg["train"]
    kind = cfg["kind"]
    if a.run_dir:
        rd = InPlaceRun(guard.inside_root(a.run_dir))
        if json.loads((rd.path / "config.json").read_text()) != cfg:
            raise guard.GuardError("resume: config differs")
    else:
        r0 = guard.RunDir(f"fsp_{cfg['run_name']}", cfg)
        rd = InPlaceRun(r0.path)
    rd.set_status("RUNNING")
    store = fsp_ckpt.ResumeStore(slot_dir, rd.run_id, cfg["run_name"])
    cfg_sha = __import__("hashlib").sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    t_start_iso, t_proc0 = guard.now(), time.time()
    limits = guard.RunLimits(max_seconds=a.max_seconds + 600, monitor=monitor)
    status, reason, rc = "INCOMPLETE", "", 1
    mbr = a.micro_batch_rows or t["micro_batch_rows"]
    rps = t["rows_per_step"]
    accum = rps // mbr
    assert accum * mbr == rps
    use_amp = device.type == "cuda"
    split, steps, planned_tokens = build_stream(cfg)
    n_steps = len(steps)
    rd.log(f"stream: {n_steps} steps, {planned_tokens} valid tokens, micro {mbr} x accum {accum}")
    cond = cfg.get("cond", {})
    is_main = kind == "main"
    # ---------------- model
    if is_main:
        mc = FSPConfig.from_dict(cfg["model"])
        model = FSPModel(mc, cfg["seed"]).to(device)
        cache = ForecastCache(guard.ROOT / cfg["cache_dir"]) if cond.get("forecast_source") else None
    else:
        model = build_draft(cfg["seed"])
        if cfg.get("init_from"):
            sd = torch.load(guard.ROOT / cfg["init_from"], map_location="cpu", weights_only=True)
            model.load_state_dict(sd["model"])
        model = model.to(device)
        cache = None
    mult_names = ("router.weight", "topic_proj", "topic_norm", "dom_topic", "dom_tok")
    groups = {"decay": [], "nodecay": [], "decay_m": [], "nodecay_m": []}
    for n, p in model.named_parameters():
        m = is_main and cond.get("router_lr_mult", 1.0) != 1.0 and any(k in n for k in mult_names)
        groups[("decay" if p.dim() >= 2 else "nodecay") + ("_m" if m else "")].append(p)
    pg = []
    for k, ps in groups.items():
        if ps:
            pg.append({"params": ps, "weight_decay": t["weight_decay"] if k.startswith("decay") else 0.0,
                       "lr_mult": cond.get("router_lr_mult", 1.0) if k.endswith("_m") else 1.0})
    opt = torch.optim.AdamW(pg, lr=t["lr"], betas=tuple(t["betas"]), eps=1e-8, fused=device.type == "cuda")
    st = {"next_step": 0, "seen": 0, "acc": {"step_s": 0.0, "tok_after_warm": 0, "step_s_after_warm": 0.0},
          "ckpts": {}, "legs": 0}
    if a.run_dir:
        ent = store.latest_verified(log=rd.log)
        if ent is not None:
            ck = torch.load(guard.ROOT / ent["path"], map_location="cpu", weights_only=False)
            if ck["run_id"] != rd.run_id or ck.get("config_sha256") != cfg_sha:
                raise guard.GuardError("resume checkpoint belongs to another run/config")
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            st = ck["st"]
            fsp_ckpt.set_rng_state(ck["rng"])
            rd.log(f"resumed at step {st['next_step']} from {ent['path']} (sha256 {ent['sha256'][:12]} verified)")
            del ck
        elif slot.exists():
            ck = torch.load(slot, map_location="cpu", weights_only=False)
            if ck["run_id"] != rd.run_id:
                raise guard.GuardError("slot belongs to another run")
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            st = ck["st"]
            torch.set_rng_state(ck["rng_cpu"])
            rd.log(f"resumed at step {st['next_step']} from legacy slot")
        else:
            rd.log("resume requested but no verified checkpoint exists: starting at step 0")
    st["legs"] += 1
    fwd = model
    ce_fn = lambda h, w, y: F.cross_entropy(F.linear(h, w).float(), y, ignore_index=-100, reduction="sum")
    if device.type == "cuda" and not a.no_compile:
        fwd = torch.compile(model) if is_main else model
        ce_fn = torch.compile(ce_fn)
    emb_w = (lambda: model.tok_emb.weight)

    def save_slot(note=""):
        """Fully resumable checkpoint at an optimizer-step boundary (amendment A3/A4)."""
        if device.type == "cuda":
            torch.cuda.synchronize()
        step_ = st["next_step"]
        payload = {"run_id": rd.run_id, "run_name": cfg["run_name"], "config_sha256": cfg_sha,
                   "model": model.state_dict(), "opt": opt.state_dict(), "st": st, "rng": fsp_ckpt.rng_state(),
                   "cursor": {"next_step": step_, "n_steps": n_steps, "rows_per_step": rps,
                              "stream": cfg.get("stream", kind), "kind": kind},
                   "scheduler": {"type": "stateless lr_at(step) WSD", "next_lr": lr_at(min(step_, n_steps - 1),
                                                                                       n_steps, t)},
                   "scaler": None, "amp": "bf16 autocast (no GradScaler)", "driver": fsp_ckpt.driver_snapshot(),
                   "time": guard.now()}
        t0_ = time.time()
        ent = store.save(payload, step_, st["seen"], note=note)
        st["acc"]["ckpt_s"] = st["acc"].get("ckpt_s", 0.0) + (time.time() - t0_)
        st["acc"]["n_resume_ckpts"] = st["acc"].get("n_resume_ckpts", 0) + 1
        return ent

    def save_ckpt(tag, half):
        sd = {k: (v.to(torch.bfloat16) if half and v.is_floating_point() else v) for k, v in model.state_dict().items()}
        p = rd.path / f"ckpt_{tag}.pt"
        step_, seen_ = st["next_step"], st["seen"]
        if p.exists():  # written in an earlier leg that stopped before its resumable checkpoint: verify and record
            ld = torch.load(p, map_location="cpu", weights_only=True, mmap=True)
            if ld["step"] != step_ or ld["seen"] != seen_:
                raise guard.GuardError(f"existing milestone {p} inconsistent with replay")
            st["ckpts"][tag] = {"path": str(p.relative_to(guard.ROOT)), "sha256": fsp_ckpt.sha256_file(p),
                                "step": step_, "seen": seen_}
            return
        sha, nb, sec = fsp_ckpt.atomic_torch_save(
            {"model": sd, "config": cfg, "step": step_, "seen": seen_, "dtype": "bf16" if half else "fp32"}, p,
            lambda ld: ld["step"] == step_ and ld["seen"] == seen_ and len(ld["model"]) == len(sd))
        st["ckpts"][tag] = {"path": str(p.relative_to(guard.ROOT)), "sha256": sha, "step": step_, "seen": seen_}
        fsp_ckpt.ledger({"time": guard.now(), "run_id": rd.run_id, "run_name": cfg["run_name"], "kind": "milestone",
                         "path": st["ckpts"][tag]["path"], "step": step_, "seen": seen_, "sha256": sha, "bytes": nb,
                         "save_seconds": round(sec, 2), "note": tag})

    ck_fracs = cfg.get("checkpoint_fracs", [1.0])
    ck_steps = {max(1, int(round(f * n_steps))): f for f in ck_fracs}
    teacher_cond = is_main and mc.teacher != "none"
    spec = is_main and mc.routing == "spec"
    last_slot = time.time()
    last_slot_step = st["next_step"]
    every_steps = max(1, int(0.10 * n_steps))  # amendment A1: 10% of planned steps or 15 min, whichever first
    every_s = 900.0
    rng_tf = torch.Generator()

    def step_controls(step):
        frac = step / n_steps
        if not is_main:
            return {}
        sig, tf, ph = curriculum(frac, cond.get("anneal_shape", "linear")) if cond.get("curriculum") else (0., 0., 0)
        dc = cond.get("dom12", 0.0) if frac < 0.5 else cond.get("dom3", 0.0)
        return {"sigma": sig if teacher_cond else 0.0, "tf": tf, "phase": ph, "dom_coef": dc}

    try:
        rd.log(f"params: {model.param_counts() if is_main else sum(p.numel() for p in model.parameters())}")
        model.train()
        while st["next_step"] < n_steps:
            step = st["next_step"]
            if a.test_kill_at_step is not None and step == a.test_kill_at_step and st["legs"] == 1:
                os.kill(os.getpid(), signal.SIGKILL)
            if a.test_signal_at_step is not None and step == a.test_signal_at_step and st["legs"] == 1:
                os.kill(os.getpid(), signal.SIGINT)
            if stopf.reason is None and monitor is not None and monitor.contaminated:
                stopf.reason = f"foreign GPU process: {monitor.detail}"
            if stopf.reason is None and fsp_ckpt.STOP_MARKER.exists():
                stopf.reason = "stop marker STOP"
            if stopf.reason is not None:  # external cancellation: stop at this optimizer-step boundary
                status, reason, rc = "EXTERNALLY_STOPPED", stopf.reason, 77
                break
            if a.max_steps is not None and step >= a.max_steps:
                status, reason, rc = "INTERRUPTED", "max-steps", 75
                break
            if time.time() - t_proc0 > a.max_seconds:
                status, reason, rc = "INTERRUPTED", "leg time limit", 75
                break
            limits.check()
            lr = lr_at(step, n_steps, t)
            for g in opt.param_groups:
                g["lr"] = lr * g.get("lr_mult", 1.0)
            sc = step_controls(step)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            rows = steps[step]
            n_valid = int(sum(split.lengths[d] - 1 for r in rows for d in r))
            parts = {}
            for mi_ in range(accum):
                rr = rows[mi_ * mbr:(mi_ + 1) * mbr]
                npb = make_batch(rr, split, label_mode=cond.get("label_mode", "true"))
                bt = to_torch(npb, device)
                if not is_main:
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                        h = draft_hidden(model, bt["input_ids"], bt["doc_ids"], bt["positions"])
                        hflat, y = h.reshape(-1, h.shape[-1]), bt["targets"].reshape(-1)
                        lm = sum(ce_fn(hflat[i:i + 8192], emb_w(), y[i:i + 8192]) for i in range(0, y.numel(), 8192))
                        loss = lm / n_valid
                    loss.backward()
                    parts["lm"] = parts.get("lm", 0.0) + float(lm.detach())
                    continue
                mi = {"input_ids": bt["input_ids"], "doc_ids": bt["doc_ids"], "positions": bt["positions"]}
                need_lab = cond.get("use_dom") or spec or cond.get("edl_beta", 0) > 0
                if need_lab:
                    mi["domain"] = bt["domain"]
                    mi["seq_id"] = bt["seq_id"]
                if spec:
                    if cond.get("oracle"):
                        forced = bt["domain"] >= 0
                    elif sc["tf"] >= 1.0:
                        forced = bt["domain"] >= 0
                    elif sc["tf"] > 0:
                        rng_tf.manual_seed(cfg["seed"] * 1_000_003 + step * 31 + mi_)
                        u = torch.rand(bt["domain"].shape, generator=rng_tf).to(device)
                        forced = (u < sc["tf"]) & (bt["domain"] >= 0)
                    else:
                        forced = torch.zeros_like(bt["domain"], dtype=torch.bool)
                    mi["forced"] = forced
                teach = teacher_cond and sc["sigma"] > 0
                if teach:
                    g = cache.gather(rr, split, cond["forecast_source"])
                    gt = {k: torch.as_tensor(v, device=device) for k, v in g.items()}
                    scal = causal_scalars(gt, bt["input_ids"], bt["doc_ids"], bt["positions"], bt["seg_index"])
                    mi["fc"] = {"roll": gt["roll"], "scal": scal}
                    mi["sigma"] = torch.tensor(sc["sigma"], device=device)
                ctl = {"teach": teach, "use_dom": bool(cond.get("use_dom") or spec),
                       "edl": cond.get("edl_beta", 0) > 0}
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                    out = fwd(mi, ctl)
                    h = out["hidden"]
                    hflat, y = h.reshape(-1, h.shape[-1]), bt["targets"].reshape(-1)
                    lm = sum(ce_fn(hflat[i:i + 8192], emb_w(), y[i:i + 8192]) for i in range(0, y.numel(), 8192))
                    loss = lm / n_valid + (t["aux_coeff"] * out["aux"] + t["z_coeff"] * out["z"]) / accum
                    dc = sc["dom_coef"]
                    if dc > 0:
                        dl = out["dom_ce"]
                        if teach:
                            dl = dl + out["dom_ce_teacher"] + out["dom_kl"]
                        loss = loss + dc * dl / accum
                    if teach:
                        loss = loss + cond.get("distill_coef", 0.0) * out["distill"] / accum
                    if cond.get("edl_beta", 0) > 0:
                        loss = loss + cond["edl_beta"] * out["edl"] / accum
                loss.backward()
                for k in ("aux", "z", "dom_ce", "dom_ce_teacher", "dom_kl", "distill", "edl"):
                    parts[k] = parts.get(k, 0.0) + float(out[k].detach()) / accum
                parts["lm"] = parts.get("lm", 0.0) + float(lm.detach())
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), t["clip"])
            opt.step()
            opt.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - t0
            lm_mean = parts["lm"] / n_valid
            if not math.isfinite(lm_mean):
                raise FloatingPointError(f"non-finite loss at step {step}")
            st["next_step"] = step + 1
            st["seen"] += n_valid
            st["acc"]["step_s"] += dt
            if step >= 10 + (st["legs"] - 1) * 0:
                st["acc"]["tok_after_warm"] += n_valid
                st["acc"]["step_s_after_warm"] += dt
            if step % t.get("log_every", 50) == 0 or step == n_steps - 1:
                rec = {"step": step, "lr": lr, "seen": st["seen"], "lm": lm_mean, "gn": float(gn), "dt": dt,
                       **{k: v for k, v in parts.items() if k != "lm"}, **sc}
                rd.metric(rec)
                if step % (t.get("log_every", 50) * 10) == 0:
                    rd.log(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}))
            if st["next_step"] in ck_steps:
                f = ck_steps[st["next_step"]]
                save_ckpt(f"p{int(round(f * 100)):03d}", half=(f < 1.0))
                save_slot(note=f"milestone p{int(round(f * 100)):03d}")
                last_slot, last_slot_step = time.time(), st["next_step"]
            elif st["next_step"] - last_slot_step >= every_steps or time.time() - last_slot >= every_s:
                save_slot(note="cadence")
                last_slot, last_slot_step = time.time(), st["next_step"]
        else:
            status, reason, rc = "COMPLETE", "", 0
            if last_slot_step != st["next_step"]:  # final-step resumable checkpoint (eval can rerun after a stop)
                save_slot(note="final step")
        if status in ("INTERRUPTED", "EXTERNALLY_STOPPED"):
            save_slot(note=f"{status}: {reason}"[:120])
    except KeyboardInterrupt:  # should not occur during training (signals only set the flag); keep files, no save
        status, reason, rc = "EXTERNALLY_STOPPED", f"KeyboardInterrupt mid-step: {stopf.reason}", 77
    except torch.cuda.OutOfMemoryError:
        status, reason, rc = "OOM", traceback.format_exc()[-2000:], 76
    except guard.GuardError as e:  # raised by limits.check() at the top of a step (a step boundary)
        if "CONTAMINATED" in str(e):  # foreign GPU process: external stop (amendment B)
            status, reason, rc = "EXTERNALLY_STOPPED", f"foreign GPU process: {e}", 77
        else:
            status, reason, rc = "ABORTED", str(e), 75 if "wall-clock" in str(e) else 1
        if rc in (75, 77):
            save_slot(note=f"{status}: {reason}"[:120])
    except Exception:
        status, reason, rc = "FAILED", traceback.format_exc()[-4000:], 1
    finally:
        limits.close()
    peak = limits.peak_gib()
    t_end = guard.now()
    summary = {"status": status, "reason": reason, "steps_done": st["next_step"], "n_steps": n_steps,
               "planned_tokens": planned_tokens, "seen": st["seen"], "leg_seconds": time.time() - t_proc0,
               "train_step_seconds": st["acc"]["step_s"],
               "tokens_per_s": st["acc"]["tok_after_warm"] / max(1e-9, st["acc"]["step_s_after_warm"]),
               "peak_alloc_gib": peak[0], "peak_reserved_gib": peak[1], "ckpts": st["ckpts"], "legs": st["legs"],
               "micro_batch_rows": mbr, "ckpt_seconds": st["acc"].get("ckpt_s", 0.0),
               "n_resume_ckpts": st["acc"].get("n_resume_ckpts", 0)}
    if status == "COMPLETE" and is_main and cfg.get("eval_splits"):
        try:
            if stopf.reason is not None:
                raise KeyboardInterrupt(stopf.reason)
            stopf.in_eval = True
            summary["eval"] = end_eval(model, cfg, device)
            stopf.in_eval = False
        except KeyboardInterrupt:
            stopf.in_eval = False
            # training finished (final-step resumable checkpoint exists); the evaluation reruns on resume
            status, reason, rc = "EXTERNALLY_STOPPED", f"during end-of-run eval: {stopf.reason}", 77
            summary.update(status=status, reason=reason, training_complete=True)
        except Exception:
            stopf.in_eval = False
            summary["eval_error"] = traceback.format_exc()[-3000:]
            status, rc = "EVAL_FAILED", 1
    stopf.close()
    if monitor is not None:
        monitor.stop()
    if status == "COMPLETE" and stopf.reason is not None:
        summary["external_stop_requested_after_training"] = stopf.reason
        rc = 77  # run is genuinely complete; the driver must still yield
    if status == "COMPLETE" and not is_main:
        sd = model.state_dict()
        p = rd.path / "ckpt_final.pt"
        torch.save({"model": sd, "config": cfg}, p)
        summary["ckpts"]["final"] = {"path": str(p.relative_to(guard.ROOT)), "sha256": guard.sha256_file(p)}
    if a.device == "cuda":
        guard.gpu_ledger(t_start_iso, t_end, rd.run_id, f"fsp train {cfg['run_name']}", status)
        with open(guard.ROOT / "artifacts/forecast-specialist-pretraining-v2/compute_ledger.tsv", "a") as f:
            f.write(f"{t_start_iso}\t{t_end}\t{time.time() - t_proc0:.1f}\t{rd.run_id}\t{cfg['run_name']}\t"
                    f"{cfg.get('category', 'main_training')}\t{status}\n")
    with open(rd.path / f"leg_{st['legs']:02d}.json", "w") as f:
        json.dump(summary, f, indent=1, default=str)
    rd.set_status(status, **{k: v for k, v in summary.items() if k not in ("eval", "status")})
    if status == "COMPLETE":
        guard.atomic_write_json(rd.path / "summary.json", summary, overwrite=True)
        store.prune_after_complete()
    rd.log(f"{status} {reason[:500]}")
    return rc


def end_eval(model, cfg, device):
    from csr.fsp_eval import eval_rows, evaluate
    from csr.fsp_lib import Split
    res = {}
    oracle = bool(cfg.get("cond", {}).get("oracle"))
    model._eval_use_dom = bool(cfg.get("cond", {}).get("use_dom"))
    for sp_name in cfg["eval_splits"]:
        sp = Split(sp_name)
        r = evaluate(model, sp, eval_rows(sp), device, oracle=oracle, routing_stats=True)
        r.pop("doc_ce", None)
        res[sp_name] = r
    return res


if __name__ == "__main__":
    main()
