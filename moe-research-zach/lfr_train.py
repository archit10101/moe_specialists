#!/usr/bin/env python
"""csr-latent-forecast-router-distill-v1: train ONE main run (any condition) on the balanced main stream.

Resumable in place: verified, atomically written directory checkpoints (protocol 15.1) every max(1, floor(10% of steps))
steps or 15 min, at every phase boundary, at milestones and at every stop. The stream, phases, CF sample and LR are pure
functions of the step index, so a resumed run replays exactly. External cancellation (SIGINT/SIGTERM/SIGHUP, the STOP
marker, a foreign GPU process) stops at the next optimizer-step boundary with a checkpoint.
Exit codes: 0 COMPLETE, 75 INTERRUPTED (internal leg limit), 76 OOM, 77 EXTERNALLY_STOPPED, 78 BLOCKED, 1 failure.
Training code opens only the `main` split and (for in-process development diagnostics) `calib_search`.
Dry-run by default; GPU only via scripts/gpu_guarded.sh with --device cuda --execute.
"""
import argparse
import hashlib
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

ALLOWED_SPLITS = ("main", "calib_search")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--run-dir")
    ap.add_argument("--micro-batch-rows", type=int)
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--max-seconds", type=float, default=4700)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--test-signal-at-step", type=int)
    ap.add_argument("--test-kill-at-step", type=int)
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    cfg = json.loads(guard.inside_root(a.config).read_text())
    from csr.lfr_lib import protocol_sha
    if cfg.get("protocol_sha256") != protocol_sha():
        raise guard.GuardError("config protocol hash does not match the frozen protocol")
    print(json.dumps({"config": a.config, "run_name": cfg["run_name"], "device": a.device, "resume": a.run_dir}))
    if not guard.require_execution(a.execute, "gpu" if a.device == "cuda" else "cpu-model"):
        return
    try:
        rc = run(cfg, a)
    except guard.GuardError as e:
        if str(e).startswith("BLOCKED"):
            print(f"EXTERNAL BLOCKED {e}", flush=True)
            sys.exit(78)
        raise
    sys.exit(rc)


class StopFlag:
    """Cooperative external-stop flag: signals set the flag during training (acted on at the next optimizer-step
    boundary); during evaluation (no training state mutated) they abort it."""

    def __init__(self, monitor):
        from csr.lfr_lib import STOP_MARKER
        self.marker = STOP_MARKER
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
        while not self._stop.wait(5.0):
            why = None
            if self.marker.exists():
                why = "stop marker STOP"
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
        self.path = Path(path)
        self.run_id = self.path.name

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


def build_optimizer(model, t, cond, device):
    import torch
    mult = cond.get("router_lr_mult", 1.0)
    groups = {}
    for n, p in model.named_parameters():
        m = mult != 1.0 and (n.startswith(("student", "tc_")) or n.endswith("router.weight"))
        k = ("decay" if p.dim() >= 2 else "nodecay") + ("_m" if m else "")
        groups.setdefault(k, []).append(p)
    pg = []
    for k in ("decay", "nodecay", "decay_m", "nodecay_m"):
        if groups.get(k):
            pg.append({"params": groups[k], "weight_decay": t["weight_decay"] if k.startswith("decay") else 0.0,
                       "lr_mult": mult if k.endswith("_m") else 1.0})
    return torch.optim.AdamW(pg, lr=t["lr"], betas=tuple(t["betas"]), eps=1e-8, fused=device.type == "cuda")


def run(cfg, a):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from csr import lfr_ckpt
    from csr.fsp_lib import make_batch, to_torch
    from csr.lfr_lib import (CACHE_DIR, ForecastCache, build_features, cf_uniform, gather_raw, lr_at, main_stream,
                             phase_at, shuffle_donors)
    from csr.lfr_model import LFRConfig, LFRModel, standardize_utility, utility_losses
    from csr.lfr_eval import first_order_delta, exact_delta, spearman, topk_overlap
    torch.set_num_threads(4)
    monitor = None
    if a.device == "cuda":
        guard.occupancy_gate(f"lfr train:{cfg['run_name']}")
        device = guard.setup_cuda_device()
        monitor = guard.OccupancyMonitor(interval=15.0)
        monitor.start()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        device = torch.device("cpu")
    guard.register_self(f"lfr train {cfg['run_name']} on {a.device}")
    stopf = StopFlag(monitor)
    t = cfg["train"]
    cond = cfg["cond"]
    if a.run_dir:
        rd = InPlaceRun(guard.inside_root(a.run_dir))
        if json.loads((rd.path / "config.json").read_text()) != cfg:
            raise guard.GuardError("resume: config differs")
    else:
        from csr.lfr_lib import RUN_PREFIX
        rd = InPlaceRun(guard.RunDir(RUN_PREFIX + cfg["run_name"], cfg).path)
    rd.set_status("RUNNING")
    store = lfr_ckpt.ResumeStore(rd.run_id, cfg["run_name"])
    cfg_sha = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    t_start_iso, t_proc0 = guard.now(), time.time()
    limits = guard.RunLimits(max_seconds=a.max_seconds + 600, monitor=monitor)
    status, reason, rc = "INCOMPLETE", "", 1
    mbr = a.micro_batch_rows or t["micro_batch_rows"]
    rps = t["rows_per_step"]
    accum = rps // mbr
    assert accum * mbr == rps
    use_amp = device.type == "cuda"
    split, steps, planned_tokens = main_stream(cfg["budget_tokens"])
    n_steps = len(steps)
    plan_sha = hashlib.sha256(json.dumps(steps).encode()).hexdigest()
    rd.log(f"stream: {n_steps} steps, {planned_tokens} valid tokens, micro {mbr} x accum {accum}, plan {plan_sha[:12]}")
    mc = LFRConfig.from_dict(cfg["model"])
    model = LFRModel(mc, cfg["seed"]).to(device)
    teacher_cond = mc.teacher
    uses_cache = teacher_cond or cond.get("od_alpha", 0) > 0
    cache = ForecastCache(CACHE_DIR, "main") if uses_cache else None
    cf_on = cond.get("cf", False)
    schedule = cond.get("schedule", "40/20/40")
    sched_cond = uses_cache  # conditions with teacher / distillation phases
    st = {"next_step": 0, "seen": 0, "acc": {"step_s": 0.0, "tok_after_warm": 0, "step_s_after_warm": 0.0},
          "ckpts": {}, "legs": 0, "stripped": False, "diag": {}, "cfaudit": []}
    ck = None
    if a.run_dir:
        ent = store.latest_verified(log=rd.log)
        if ent is not None:
            ck = torch.load(guard.ROOT / ent["path"] / "state.pt", map_location="cpu", weights_only=False)
            if ck["run_id"] != rd.run_id or ck.get("config_sha256") != cfg_sha:
                raise guard.GuardError("resume checkpoint belongs to another run/config")
            if ck["st"].get("stripped"):
                model.strip_teacher()
            model.load_state_dict(ck["model"])
            st = ck["st"]
            rd.log(f"resumed at step {st['next_step']} from {ent['path']} (sha256 {ent['sha256'][:12]} verified)")
        else:
            rd.log("resume requested but no verified checkpoint exists: starting at step 0")
    opt = build_optimizer(model, t, cond, device)
    if ck is not None:
        opt.load_state_dict(ck["opt"])
        lfr_ckpt.set_rng_state(ck["rng"])
        del ck
    st["legs"] += 1
    compile_on = device.type == "cuda" and not a.no_compile
    fwd = torch.compile(model) if compile_on else model

    def lm_kd(h, w, y, ids=None, q=None):
        lg = F.linear(h, w).float()
        lse = torch.logsumexp(lg, -1)
        valid = y >= 0
        ce = ((lse - lg.gather(1, y.clamp(min=0).view(-1, 1)).squeeze(1)) * valid).sum()
        if ids is None:
            return ce, torch.zeros((), device=h.device)
        kd = -(q * (lg.gather(1, ids) - lse.unsqueeze(1))).sum(-1)
        return ce, (kd * valid).sum()
    lm_fn = torch.compile(lm_kd) if compile_on else lm_kd

    def payload():
        step_ = st["next_step"]
        lam_, ph_, _ = phase_at(min(step_, n_steps - 1) / n_steps, schedule) if sched_cond else (0.0, 3, 0.0)
        return {"run_id": rd.run_id, "run_name": cfg["run_name"], "config_sha256": cfg_sha,
                "model": model.state_dict(), "opt": opt.state_dict(), "st": st, "rng": lfr_ckpt.rng_state(),
                "cursor": {"next_step": step_, "n_steps": n_steps, "rows_per_step": rps, "plan_sha256": plan_sha,
                           "stream": "balanced main (lfr_lib.balanced_order)"},
                "scheduler": {"type": "stateless WSD lr_at(step)", "next_lr": lr_at(min(step_, n_steps - 1), n_steps, t)},
                "scaler": None, "amp": "bf16 autocast (no GradScaler)", "phase": {"phase": ph_, "lam": lam_},
                "candidate": cfg.get("candidate"), "driver": lfr_ckpt.driver_snapshot(),
                "cache_manifest_sha256": cfg.get("cache_manifest_sha256"), "time": guard.now()}

    def save_slot(note=""):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0_ = time.time()
        ent_ = store.save(payload(), st["next_step"], st["seen"], note=note)
        st["acc"]["ckpt_s"] = st["acc"].get("ckpt_s", 0.0) + (time.time() - t0_)
        st["acc"]["n_resume_ckpts"] = st["acc"].get("n_resume_ckpts", 0) + 1
        return ent_

    def save_milestone(tag):
        keep32 = lambda k, v: v.dim() <= 1 or "router" in k or k.startswith("student") or "norm" in k
        sd = {k: (v if keep32(k, v) or not v.is_floating_point() else v.to(torch.bfloat16))
              for k, v in model.state_dict().items() if not k.startswith("tc_")}
        p = rd.path / f"ckpt_{tag}"
        step_, seen_ = st["next_step"], st["seen"]
        if p.exists():
            if not lfr_ckpt.verify_dir(p):
                raise guard.GuardError(f"existing milestone {p} fails verification")
            meta = json.loads((p / "meta.json").read_text())
            if meta.get("step") != step_:
                raise guard.GuardError(f"existing milestone {p} inconsistent with replay")
            st["ckpts"][tag] = {"path": str(p.relative_to(guard.ROOT)), "sha256": meta["sha256"], "step": step_,
                                "seen": seen_}
            return
        obj = {"model": sd, "config": cfg, "step": step_, "seen": seen_, "dtype": "bf16 matmul weights, fp32 rest",
               "model_cfg": model.cfg.to_dict()}
        sha, nb, sec = lfr_ckpt.atomic_ckpt_dir(obj, p, lambda ld: ld["step"] == step_ and len(ld["model"]) == len(sd),
                                                {"step": step_, "seen": seen_, "tag": tag})
        st["ckpts"][tag] = {"path": str(p.relative_to(guard.ROOT)), "sha256": sha, "step": step_, "seen": seen_}
        lfr_ckpt.ledger({"time": guard.now(), "run_id": rd.run_id, "run_name": cfg["run_name"], "kind": "milestone",
                         "path": st["ckpts"][tag]["path"], "step": step_, "seen": seen_, "sha256": sha, "bytes": nb,
                         "save_seconds": round(sec, 2), "note": tag})

    ck_fracs = cfg.get("checkpoint_fracs", [1.0])
    ck_steps = {max(1, int(round(f * n_steps))): f for f in ck_fracs}
    bounds = set()
    if sched_cond:
        a_, b_, _ = [int(v) for v in schedule.split("/")]
        bounds = {int(round(a_ / 100 * n_steps)), int(round((a_ + b_) / 100 * n_steps))}
        # first steps whose phase changes (phase_at uses step/n_steps): recompute exactly
        bounds = {s for s in range(1, n_steps) if phase_at(s / n_steps, schedule)[1] != phase_at((s - 1) / n_steps,
                                                                                               schedule)[1]}
    diag_steps = {}
    if cfg.get("diag"):
        for nm, frac in cfg["diag"].get("at", {}).items():
            diag_steps[max(1, int(round(frac * n_steps)))] = nm
    last_slot = time.time()
    last_slot_step = st["next_step"]
    every_steps = max(1, int(0.10 * n_steps))
    every_s = 900.0
    audit_every = cfg.get("cf_audit_every", 100)

    def strip_now():
        nonlocal opt, fwd
        old = opt
        model.strip_teacher()
        opt = build_optimizer(model, t, cond, device)
        for g_new in opt.param_groups:
            for p_ in g_new["params"]:
                if p_ in old.state:
                    opt.state[p_] = old.state[p_]
        del old
        st["stripped"] = True
        fwd = torch.compile(model) if compile_on else model
        rd.log(f"P3: teacher stripped; params now {model.param_counts()}")

    def run_diag(name):
        """In-process development diagnostics on the CALIB-SEARCH diagnostic set (protocol 10)."""
        from csr.lfr_diag import inprocess_diag
        stopf.in_eval = True
        try:
            res = inprocess_diag(model, cfg, device, name, schedule)
        finally:
            stopf.in_eval = False
        st["diag"][name] = res
        rd.log(f"diag {name}: " + json.dumps({k: v for k, v in res.items() if not isinstance(v, (list, dict))}))
        model.train()

    try:
        if st.get("stripped") is False and sched_cond and teacher_cond and \
                phase_at(st["next_step"] / n_steps, schedule)[1] == 3 and st["next_step"] > 0:
            strip_now()
        rd.log(f"params: {model.param_counts()}")
        model.train()
        while st["next_step"] < n_steps:
            step = st["next_step"]
            if a.test_kill_at_step is not None and step == a.test_kill_at_step and st["legs"] == 1:
                os.kill(os.getpid(), signal.SIGKILL)
            if a.test_signal_at_step is not None and step == a.test_signal_at_step and st["legs"] == 1:
                os.kill(os.getpid(), signal.SIGINT)
            if stopf.reason is None and monitor is not None and monitor.contaminated:
                stopf.reason = f"foreign GPU process: {monitor.detail}"
            if stopf.reason is None and stopf.marker.exists():
                stopf.reason = "stop marker STOP"
            if stopf.reason is not None:
                status, reason, rc = "EXTERNALLY_STOPPED", stopf.reason, 77
                break
            if a.max_steps is not None and step >= a.max_steps:
                status, reason, rc = "INTERRUPTED", "max-steps", 75
                break
            if time.time() - t_proc0 > a.max_seconds:
                status, reason, rc = "INTERRUPTED", "leg time limit", 75
                break
            limits.check()
            if step in diag_steps and diag_steps[step] not in st["diag"]:
                run_diag(diag_steps[step])
            lam, phase, tscale = phase_at(step / n_steps, schedule) if sched_cond else (0.0, 3, 0.0)
            if phase == 3 and teacher_cond and not st["stripped"]:
                strip_now()
            lr = lr_at(step, n_steps, t)
            for g in opt.param_groups:
                g["lr"] = lr * g.get("lr_mult", 1.0)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            rows = steps[step]
            n_valid = int(sum(split.lengths[d] - 1 for r in rows for d in r))
            donors = shuffle_donors(rows, split.lengths) if (cond.get("shuffle") and phase < 3) else None
            parts = {}
            teach = teacher_cond and phase < 3
            od = cond.get("od_alpha", 0) > 0 and phase < 3
            cfl = cf_on and teach
            for mi_ in range(accum):
                rr = rows[mi_ * mbr:(mi_ + 1) * mbr]
                npb = make_batch(rr, split)
                bt = to_torch(npb, device)
                minp = {"input_ids": bt["input_ids"], "doc_ids": bt["doc_ids"], "positions": bt["positions"]}
                if cond.get("edl_beta", 0) > 0:
                    minp["domain"], minp["seq_id"] = bt["domain"], bt["seq_id"]
                feats = None
                if teach or od:
                    raw = gather_raw(rr, split, cache, donors)
                    rt = {k: torch.as_tensor(v.astype(np.int64) if v.dtype == np.uint16 else v, device=device)
                          for k, v in raw.items()}
                    feats = build_features(rt, bt["doc_ids"], bt["positions"])
                if teach:
                    minp["fc"] = feats
                    minp["lam"] = torch.tensor(lam, device=device)
                cstore = None
                if cfl:
                    u = cf_uniform(npb["gdoc"], npb["positions"])
                    sel = (u < cond["r_cf"]) & (npb["doc_ids"] >= 0)
                    minp["cf_idx"] = torch.as_tensor(np.flatnonzero(sel.reshape(-1)), device=device)
                    cstore = {}
                ctl = {"teach": teach, "edl": cond.get("edl_beta", 0) > 0, "lc": cfl and cond.get("c_lc", 0) > 0,
                       "cf_store": cstore, "tap": cfl}
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                    out = fwd(minp, ctl)
                    h = out["hidden"]
                    hflat, y = h.reshape(-1, h.shape[-1]), bt["targets"].reshape(-1)
                    lm = torch.zeros((), device=device)
                    kd = torch.zeros((), device=device)
                    for i in range(0, y.numel(), 8192):
                        if od:
                            ids = feats["ens_ids"].reshape(-1, 4)[i:i + 8192]
                            q = feats["ens_p"].reshape(-1, 4)[i:i + 8192]
                            c_, k_ = lm_fn(hflat[i:i + 8192], model.tok_emb.weight, y[i:i + 8192], ids, q)
                            kd = kd + k_
                        else:
                            c_, _ = lm_fn(hflat[i:i + 8192], model.tok_emb.weight, y[i:i + 8192])
                        lm = lm + c_
                    loss = lm / n_valid + (t["aux_coeff"] * out["aux"] + t["z_coeff"] * out["z"]) / accum
                    if teach:
                        loss = loss + cond.get("c_kl", 0.0) * tscale * out["kl"] / accum
                        if cond.get("c_lc", 0) > 0:
                            loss = loss + cond["c_lc"] * tscale * out["lc"] / accum
                    if cond.get("edl_beta", 0) > 0:
                        loss = loss + cond["edl_beta"] * out["edl"] / accum
                    if od:
                        loss = loss + cond["od_alpha"] * tscale * kd / n_valid
                loss.backward()
                for k in ("aux", "z", "kl", "edl", "lc"):
                    parts[k] = parts.get(k, 0.0) + float(out[k].detach()) / accum
                parts["lm"] = parts.get("lm", 0.0) + float(lm.detach())
                if od:
                    parts["kd"] = parts.get("kd", 0.0) + float(kd.detach()) / n_valid
                if cfl:  # counterfactual utility targets from the captured LM gradient; stop-gradient utility losses
                    cf_idx = minp["cf_idx"]
                    fl = {k: v.reshape(-1, *v.shape[2:]).index_select(0, cf_idx) for k, v in feats.items()}
                    ut_l, us_l = [], []
                    lu = torch.zeros((), device=device)
                    nl = 0
                    deltas = {}
                    for li in range(model.n_moe):
                        rec, gr = cstore.get(f"rec{li}"), cstore.get(f"grad{li}")
                        if rec is None or gr is None or rec["x"].shape[0] == 0:
                            continue
                        ex = model.blocks[li + 1].ffn.experts
                        dlt = first_order_delta(ex, rec["x"], gr, rec["r4"], rec["g4"], float(n_valid))
                        deltas[li] = dlt
                        tgt = standardize_utility(dlt)
                        lt_, ls_ = utility_losses(model, rec, fl, tgt, cond.get("util_mode", "rank"), li)
                        lu = lu + cond.get("c_util", 0.0) * lt_ + cond.get("c_sutil", 0.0) * ls_
                        ut_l.append(float(lt_.detach()))
                        us_l.append(float(ls_.detach()))
                        nl += 1
                    if nl:
                        (tscale * lu / (nl * accum)).backward()
                        parts["util_t"] = parts.get("util_t", 0.0) + sum(ut_l) / nl / accum
                        parts["util_s"] = parts.get("util_s", 0.0) + sum(us_l) / nl / accum
                    if mi_ == 0 and step % audit_every == 0 and deltas:
                        aud = cf_audit(model, minp, bt, cstore, deltas, step, device, exact_delta, spearman,
                                       topk_overlap, cfg["seed"])
                        st["cfaudit"].extend(aud)
                        rd.metric({"step": step, "cfaudit": aud})
                        model.train()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), t["clip"])
            if teach and hasattr(model, "tc_trunk"):
                tg = [p.grad.norm() for n, p in model.named_parameters() if n.startswith("tc_") and p.grad is not None]
                parts["tc_gn"] = float(torch.stack(tg).norm()) if tg else 0.0
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
            if step >= 10:
                st["acc"]["tok_after_warm"] += n_valid
                st["acc"]["step_s_after_warm"] += dt
            if phase < 3:
                st["acc"]["teacher_phase_s"] = st["acc"].get("teacher_phase_s", 0.0) + dt
            if step % t.get("log_every", 50) == 0 or step == n_steps - 1:
                rec = {"step": step, "lr": lr, "seen": st["seen"], "lm": lm_mean, "gn": float(gn), "dt": dt,
                       "lam": lam, "phase": phase, **{k: v for k, v in parts.items() if k != "lm"}}
                rd.metric(rec)
                if step % (t.get("log_every", 50) * 10) == 0:
                    rd.log(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}))
            ns = st["next_step"]
            if ns in ck_steps:
                f = ck_steps[ns]
                save_milestone(f"p{int(round(f * 100)):03d}")
                save_slot(note=f"milestone p{int(round(f * 100)):03d}")
                last_slot, last_slot_step = time.time(), ns
            elif ns in bounds:
                save_slot(note=f"phase boundary -> phase {phase_at(ns / n_steps, schedule)[1]}")
                last_slot, last_slot_step = time.time(), ns
            elif ns - last_slot_step >= every_steps or time.time() - last_slot >= every_s:
                save_slot(note="cadence")
                last_slot, last_slot_step = time.time(), ns
        else:
            status, reason, rc = "COMPLETE", "", 0
            if last_slot_step != st["next_step"]:
                save_slot(note="final step (before end-of-run evaluation)")
            if "end" in (cfg.get("diag") or {}).get("at_end", []) and "end" not in st["diag"]:
                run_diag("end")
        if status in ("INTERRUPTED", "EXTERNALLY_STOPPED"):
            save_slot(note=f"{status}: {reason}"[:120])
    except KeyboardInterrupt:
        status, reason, rc = "EXTERNALLY_STOPPED", f"KeyboardInterrupt (during diagnostics): {stopf.reason}", 77
        try:
            save_slot(note="stop during in-process diagnostics")
        except Exception:
            pass
    except torch.cuda.OutOfMemoryError:
        status, reason, rc = "OOM", traceback.format_exc()[-2000:], 76
    except guard.GuardError as e:
        if "CONTAMINATED" in str(e):
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
               "teacher_phase_seconds": st["acc"].get("teacher_phase_s", 0.0),
               "peak_alloc_gib": peak[0], "peak_reserved_gib": peak[1], "ckpts": st["ckpts"], "legs": st["legs"],
               "micro_batch_rows": mbr, "ckpt_seconds": st["acc"].get("ckpt_s", 0.0),
               "n_resume_ckpts": st["acc"].get("n_resume_ckpts", 0), "params": model.param_counts(),
               "plan_sha256": plan_sha, "diag": st["diag"], "cfaudit": st["cfaudit"]}
    if status == "COMPLETE" and cfg.get("end_eval"):
        try:
            if stopf.reason is not None:
                raise KeyboardInterrupt(stopf.reason)
            stopf.in_eval = True
            summary["eval"] = end_eval(model, cfg, device)
            stopf.in_eval = False
        except KeyboardInterrupt:
            stopf.in_eval = False
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
        rc = 77
    if a.device == "cuda":
        guard.gpu_ledger(t_start_iso, t_end, rd.run_id, f"lfr train {cfg['run_name']}", status)
        from csr.lfr_lib import ART
        with open(ART / "compute_ledger.tsv", "a") as f:
            f.write(f"{t_start_iso}\t{t_end}\t{time.time() - t_proc0:.1f}\t{rd.run_id}\t{cfg['run_name']}\t"
                    f"{cfg.get('category', 'main_training')}\t{status}\n")
    with open(rd.path / f"leg_{st['legs']:02d}.json", "w") as f:
        json.dump(summary, f, indent=1, default=str)
    rd.set_status(status, **{k: v for k, v in summary.items() if k not in ("eval", "status", "diag", "cfaudit")})
    if status == "COMPLETE":
        guard.atomic_write_json(rd.path / "summary.json", summary, overwrite=True)
        store.prune_after_complete()
    rd.log(f"{status} {reason[:500]}")
    return rc


def cf_audit(model, minp, bt, cstore, deltas, step, device, exact_delta, spearman, topk_overlap, seed):
    """Exact substitution audit of 2 CF-sampled (position, layer) pairs (protocol 6.4)."""
    import numpy as np
    import torch
    rng = np.random.default_rng(seed * 7919 + step)
    cf_idx = minp["cf_idx"].cpu().numpy()
    T = bt["input_ids"].shape[1]
    out = []
    layers = sorted(deltas)
    for _ in range(2):
        li = int(rng.choice(layers))
        j = int(rng.integers(len(cf_idx)))
        flat = int(cf_idx[j])
        b, tpos = flat // T, flat % T
        rec = cstore[f"rec{li}"]
        row = {k: v[b:b + 1] for k, v in minp.items() if k in ("input_ids", "doc_ids", "positions")}
        if "fc" in minp:
            row["fc"] = {k: v[b:b + 1] for k, v in minp["fc"].items()}
            row["lam"] = minp["lam"]
        row["_targets"] = bt["targets"][b:b + 1]
        ex = exact_delta(model, row, li, tpos, rec["x"][j], int(rec["r4"][j]), float(rec["g4"][j]), device,
                         expect_topi=rec["topi"][j].tolist())
        route_ok = bool(getattr(exact_delta, "route_ok", True))
        ex = ex.cpu().numpy()
        fo = deltas[li][j].detach().cpu().numpy()
        r = {"step": step, "layer": li, "flat": flat, "rho_fo_exact": spearman(fo, ex),
             "top4_fo_exact": topk_overlap(fo, ex), "exact_std": float(ex.std()), "fo_std": float(fo.std()),
             "route_ok": route_ok}
        if "u" in rec:
            r["rho_teacher_exact"] = spearman(rec["u"][j].cpu().numpy(), ex)
        out.append(r)
    return out


def end_eval(model, cfg, device):
    """Development end-of-run evaluation (student mode) on CALIB-SEARCH only."""
    from csr.fsp_lib import Split
    from csr.lfr_eval import eval_rows, evaluate
    res = {}
    for sp_name in cfg["end_eval"]:
        if sp_name not in ALLOWED_SPLITS:
            raise guard.GuardError(f"training code may not evaluate {sp_name}")
        sp = Split(sp_name)
        r = evaluate(model, sp, eval_rows(sp), device, routing=True)
        r.pop("doc_ce", None)
        res[sp_name] = r
    return res


if __name__ == "__main__":
    main()
