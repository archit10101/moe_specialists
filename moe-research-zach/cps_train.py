#!/usr/bin/env python
"""csr-context-prior-specialization-v2: train ONE V2 run (context-prior conditions M1-M6 and candidates C01-C12).

Identical to the V1 trainer (lfr_train.py) in everything that defines the comparison with the reused TOKEN-4 runs:
balanced main stream (lfr_lib.main_stream prefix plans), 64 rows/step, stateless WSD lr_at(step), AdamW(0.9, 0.95,
wd 0.1 on >=2-d params), clip 1.0, bf16 autocast, Switch aux 0.01 + z 0.001, backbone initialisation (CPSModel builds
the TOKEN-4 backbone first). Resumable in place with V1's verified atomic directory checkpoints (cps_ckpt): every
max(1, 10% of steps) steps or 15 min, at every phase boundary, at milestones and at every stop. External cancellation
(SIGINT/SIGTERM/SIGHUP, the V2 STOP marker, a foreign GPU process) stops at the next optimizer-step boundary with a
checkpoint. Exit codes: 0 COMPLETE, 75 INTERRUPTED (leg limit), 76 OOM, 77 EXTERNALLY_STOPPED, 78 BLOCKED, 1 failure.
Training code opens only `main` (+ `calib_search` for the end-of-run development CE). Dry-run by default.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
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
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    cfg = json.loads(guard.inside_root(a.config).read_text())
    from csr.cps_lib import protocol_sha
    if cfg.get("protocol_sha256") != protocol_sha():
        raise guard.GuardError("config protocol hash does not match the frozen V2 protocol")
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
    """Cooperative external-stop flag (V2 STOP marker)."""

    def __init__(self, monitor):
        from csr.cps_lib import STOP_MARKER
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
    """V1 grouping: decay for >=2-d parameters; optional LR multiplier for the context-prior / teacher parameters."""
    import torch
    mult = cond.get("ctx_lr_mult", 1.0)
    groups = {}
    for n, p in model.named_parameters():
        m = mult != 1.0 and n.startswith(("cp_", "tc_"))
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
    from csr import cps_ckpt
    from csr.fsp_lib import make_batch, to_torch
    from csr.cps_lib import (CACHE_DIR, ForecastCache, build_features, cf_uniform, gather_raw, lr_at, main_stream,
                             phase_at, teacher_end_ok)
    from csr.cps_model import CPSConfig, CPSModel, standardize, utility_loss, spec_consistency
    from csr.lfr_eval import first_order_delta, exact_delta, spearman, topk_overlap
    torch.set_num_threads(4)
    monitor = None
    if a.device == "cuda":
        guard.occupancy_gate(f"cps train:{cfg['run_name']}")
        device = guard.setup_cuda_device()
        monitor = guard.OccupancyMonitor(interval=15.0)
        monitor.start()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        device = torch.device("cpu")
    guard.register_self(f"cps train {cfg['run_name']} on {a.device}")
    stopf = StopFlag(monitor)
    t = cfg["train"]
    cond = cfg["cond"]
    if a.run_dir:
        rd = InPlaceRun(guard.inside_root(a.run_dir))
        if json.loads((rd.path / "config.json").read_text()) != cfg:
            raise guard.GuardError("resume: config differs")
    else:
        from csr.cps_lib import RUN_PREFIX
        rd = InPlaceRun(guard.RunDir(RUN_PREFIX + cfg["run_name"], cfg).path)
    rd.set_status("RUNNING")
    store = cps_ckpt.ResumeStore(rd.run_id, cfg["run_name"])
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
    cc = CPSConfig.from_dict(cfg["ccfg"])
    model = CPSModel(cc, cfg["seed"], cfg["backbone"]).to(device)
    teacher_cond = cc.teacher
    schedule = cond.get("schedule", "40/20/40")
    if teacher_cond and not teacher_end_ok(schedule):
        raise guard.GuardError(f"schedule {schedule} uses drafts beyond the reused cache")
    cache = ForecastCache(CACHE_DIR, "main") if teacher_cond else None
    cf_on = cond.get("c_util", 0.0) > 0 or cond.get("c_spec", 0.0) > 0
    cf_phases = cond.get("cf_phases", "all")
    lam_max = cond.get("lam_max", 0.9)
    st = {"next_step": 0, "seen": 0, "acc": {"step_s": 0.0, "tok_after_warm": 0, "step_s_after_warm": 0.0},
          "ckpts": {}, "legs": 0, "stripped": False, "cfaudit": []}
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
        cps_ckpt.set_rng_state(ck["rng"])
        del ck
    st["legs"] += 1
    compile_on = device.type == "cuda" and not a.no_compile
    fwd = torch.compile(model) if compile_on else model

    def lm_ce(h, w, y):
        lg = F.linear(h, w).float()
        lse = torch.logsumexp(lg, -1)
        valid = y >= 0
        return ((lse - lg.gather(1, y.clamp(min=0).view(-1, 1)).squeeze(1)) * valid).sum()
    lm_fn = torch.compile(lm_ce) if compile_on else lm_ce

    def phase(step_):
        if not teacher_cond:
            return 0.0, 3, 0.0
        return phase_at(step_ / n_steps, schedule, lam_max)

    def payload():
        step_ = st["next_step"]
        lam_, ph_, _ = phase(min(step_, n_steps - 1))
        return {"run_id": rd.run_id, "run_name": cfg["run_name"], "config_sha256": cfg_sha,
                "model": model.state_dict(), "opt": opt.state_dict(), "st": st, "rng": cps_ckpt.rng_state(),
                "cursor": {"next_step": step_, "n_steps": n_steps, "rows_per_step": rps, "plan_sha256": plan_sha,
                           "stream": "balanced main (lfr_lib.balanced_order)"},
                "scheduler": {"type": "stateless WSD lr_at(step)", "next_lr": lr_at(min(step_, n_steps - 1), n_steps, t)},
                "scaler": None, "amp": "bf16 autocast (no GradScaler)", "phase": {"phase": ph_, "lam": lam_},
                "candidate": cfg.get("candidate"), "driver": cps_ckpt.driver_snapshot(), "time": guard.now()}

    def save_slot(note=""):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0_ = time.time()
        ent_ = store.save(payload(), st["next_step"], st["seen"], note=note)
        st["acc"]["ckpt_s"] = st["acc"].get("ckpt_s", 0.0) + (time.time() - t0_)
        st["acc"]["n_resume_ckpts"] = st["acc"].get("n_resume_ckpts", 0) + 1
        return ent_

    def save_milestone(tag):
        keep32 = lambda k, v: v.dim() <= 1 or "router" in k or k.startswith("cp_") or "norm" in k
        sd = {k: (v if keep32(k, v) or not v.is_floating_point() else v.to(torch.bfloat16))
              for k, v in model.state_dict().items() if not k.startswith("tc_")}
        p = rd.path / f"ckpt_{tag}"
        step_, seen_ = st["next_step"], st["seen"]
        if p.exists():
            if not cps_ckpt.verify_dir(p):
                raise guard.GuardError(f"existing milestone {p} fails verification")
            meta = json.loads((p / "meta.json").read_text())
            if meta.get("step") != step_:
                raise guard.GuardError(f"existing milestone {p} inconsistent with replay")
            st["ckpts"][tag] = {"path": str(p.relative_to(guard.ROOT)), "sha256": meta["sha256"], "step": step_,
                                "seen": seen_}
            return
        obj = {"model": sd, "config": cfg, "step": step_, "seen": seen_, "dtype": "bf16 matmul weights, fp32 rest",
               "ccfg": {**model.ccfg.to_dict(), "teacher": False}, "backbone": cfg["backbone"]}
        sha, nb, sec = cps_ckpt.atomic_ckpt_dir(obj, p, lambda ld: ld["step"] == step_ and len(ld["model"]) == len(sd),
                                                {"step": step_, "seen": seen_, "tag": tag})
        st["ckpts"][tag] = {"path": str(p.relative_to(guard.ROOT)), "sha256": sha, "step": step_, "seen": seen_}
        cps_ckpt.ledger({"time": guard.now(), "run_id": rd.run_id, "run_name": cfg["run_name"], "kind": "milestone",
                         "path": st["ckpts"][tag]["path"], "step": step_, "seen": seen_, "sha256": sha, "bytes": nb,
                         "save_seconds": round(sec, 2), "note": tag})

    ck_fracs = cfg.get("checkpoint_fracs", [1.0])
    ck_steps = {max(1, int(round(f * n_steps))): f for f in ck_fracs}
    bounds = set()
    if teacher_cond:
        bounds = {s for s in range(1, n_steps) if phase(s)[1] != phase(s - 1)[1]}
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
        rd.log(f"P3: teacher (drafts) stripped; params now {model.param_counts()}")

    try:
        if st.get("stripped") is False and teacher_cond and phase(st["next_step"])[1] == 3 and st["next_step"] > 0:
            strip_now()
        rd.log(f"params: {model.param_counts()}")
        model.train()
        while st["next_step"] < n_steps:
            step = st["next_step"]
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
            lam, ph, tscale = phase(step)
            if ph == 3 and teacher_cond and not st["stripped"]:
                strip_now()
            lr = lr_at(step, n_steps, t)
            for g in opt.param_groups:
                g["lr"] = lr * g.get("lr_mult", 1.0)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            rows = steps[step]
            n_valid = int(sum(split.lengths[d] - 1 for r in rows for d in r))
            parts = {}
            teach = teacher_cond and ph < 3
            cfl = cf_on and (teach or cf_phases == "all")
            log_now = step % t.get("log_every", 50) == 0 or step == n_steps - 1
            for mi_ in range(accum):
                rr = rows[mi_ * mbr:(mi_ + 1) * mbr]
                npb = make_batch(rr, split)
                bt = to_torch(npb, device)
                minp = {"input_ids": bt["input_ids"], "doc_ids": bt["doc_ids"], "positions": bt["positions"]}
                if teach:
                    raw = gather_raw(rr, split, cache, None)
                    rt = {k: torch.as_tensor(v.astype(np.int64) if v.dtype == np.uint16 else v, device=device)
                          for k, v in raw.items()}
                    minp["fc"] = build_features(rt, bt["doc_ids"], bt["positions"])
                    minp["lam"] = torch.tensor(lam, device=device)
                cstore = None
                if cfl:
                    u = cf_uniform(npb["gdoc"], npb["positions"])
                    sel = (u < cond["r_cf"]) & (npb["doc_ids"] >= 0)
                    minp["cf_idx"] = torch.as_tensor(np.flatnonzero(sel.reshape(-1)), device=device)
                    cstore = {}
                ctl = {"teach": teach, "cf_store": cstore, "tap": cfl}
                if log_now and mi_ == 0 and cc.variant != "token":  # routing statistics: separate no-grad eager pass
                    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                                         enabled=use_amp):
                        so = model({k: v for k, v in minp.items() if k != "cf_idx"}, {"teach": teach, "stats": True})
                    for k, v in (so.get("cstats") or {}).items():
                        parts[f"cs_{k}"] = float(np.mean(v)) if v else None
                    del so
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                    out = fwd(minp, ctl)
                    h = out["hidden"]
                    hflat, y = h.reshape(-1, h.shape[-1]), bt["targets"].reshape(-1)
                    lm = torch.zeros((), device=device)
                    for i in range(0, y.numel(), 8192):
                        lm = lm + lm_fn(hflat[i:i + 8192], model.tok_emb.weight, y[i:i + 8192])
                    loss = lm / n_valid + (t["aux_coeff"] * out["aux"] + t["z_coeff"] * out["z"]) / accum
                    if teach:
                        loss = loss + cond.get("c_kl", 0.0) * tscale * out["kl"] / accum
                    if cond.get("alpha_l1", 0.0) > 0:
                        loss = loss + cond["alpha_l1"] * out["areg"] / accum
                loss.backward()
                for k in ("aux", "z", "kl", "areg"):
                    parts[k] = parts.get(k, 0.0) + float(out[k].detach()) / accum
                parts["lm"] = parts.get("lm", 0.0) + float(lm.detach())
                if cfl and "ctx" in cstore:
                    crec = cstore["ctx"]
                    cE, cS, cT = model.cf_contexts(crec, torch.tensor(lam, device=device), teach)
                    valid = crec["valid"]
                    lu = torch.zeros((), device=device)
                    ut_l, sp_l, deltas = [], [], {}
                    for j, li in enumerate(model.inj):
                        rec, gr = cstore.get(f"rec{li}"), cstore.get(f"grad{li}")
                        if rec is None or gr is None or rec["x"].shape[0] == 0 or not bool(valid.any()):
                            continue
                        ex = model.blocks[li + 1].ffn.experts
                        dlt = first_order_delta(ex, rec["x"], gr, rec["r4"], rec["g4"], float(n_valid))
                        deltas[li] = dlt
                        tstd = standardize(dlt)
                        vs = valid.nonzero().squeeze(-1)
                        lSv = rec["lS"].index_select(0, vs)
                        if cond.get("c_util", 0.0) > 0:
                            l_u = utility_loss(model.prior_logits(j, cE.index_select(0, vs), lSv),
                                               tstd.index_select(0, vs), cond["util_mode"])
                            if teach and cT is not None:
                                l_u = 0.5 * (l_u + utility_loss(model.prior_logits(j, cS.index_select(0, vs), lSv),
                                                                tstd.index_select(0, vs), cond["util_mode"]))
                            lu = lu + cond["c_util"] * l_u
                            ut_l.append(float(l_u.detach()))
                        if cond.get("c_spec", 0.0) > 0:
                            l_s, sinfo = spec_consistency(model.prior_logits(j, cE.index_select(0, vs), lSv),
                                                          cE.index_select(0, vs), tstd.index_select(0, vs),
                                                          dlt.index_select(0, vs), crec["chunk"].index_select(0, vs),
                                                          gen_seed=cfg["seed"] * 100003 + step * 31 + mi_ * 7 + li)
                            lu = lu + cond["c_spec"] * l_s
                            sp_l.append(float(l_s.detach()))
                            for k_, v_ in sinfo.items():
                                parts[f"sp_{k_}"] = parts.get(f"sp_{k_}", 0.0) + v_ / (accum * len(model.inj))
                    nl = max(1, len(deltas))
                    if deltas and lu.requires_grad:
                        (lu / (nl * accum)).backward()
                    if ut_l:
                        parts["util"] = parts.get("util", 0.0) + sum(ut_l) / len(ut_l) / accum
                    if sp_l:
                        parts["spec"] = parts.get("spec", 0.0) + sum(sp_l) / len(sp_l) / accum
                    if mi_ == 0 and step % audit_every == 0 and deltas:
                        from lfr_train import cf_audit
                        aud = cf_audit(model, minp, bt, cstore, deltas, step, device, exact_delta, spearman,
                                       topk_overlap, cfg["seed"])
                        # prior-vs-exact agreement for the audited rows
                        for r_ in aud:
                            r_.pop("rho_teacher_exact", None)
                        st["cfaudit"].extend(aud)
                        rd.metric({"step": step, "cfaudit": aud})
                        model.train()
            # backbone: exactly the TOKEN-4 rule (clip its own gradient norm); V2-only modules clipped separately so
            # auxiliary-loss gradients can never rescale backbone updates
            bb_p = [p_ for n_, p_ in model.named_parameters() if not n_.startswith(("cp_", "tc_"))]
            v2_p = [p_ for n_, p_ in model.named_parameters() if n_.startswith(("cp_", "tc_"))]
            gn = torch.nn.utils.clip_grad_norm_(bb_p, t["clip"])
            if v2_p:
                parts["gn_v2"] = float(torch.nn.utils.clip_grad_norm_(v2_p, t["clip"]))
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
            if ph < 3:
                st["acc"]["teacher_phase_s"] = st["acc"].get("teacher_phase_s", 0.0) + dt
            if log_now:
                rec = {"step": step, "lr": lr, "seen": st["seen"], "lm": lm_mean, "gn": float(gn), "dt": dt,
                       "lam": lam, "phase": ph, **{k: v for k, v in parts.items() if k != "lm"}}
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
                save_slot(note=f"phase boundary -> phase {phase(ns)[1]}")
                last_slot, last_slot_step = time.time(), ns
            elif ns - last_slot_step >= every_steps or time.time() - last_slot >= every_s:
                save_slot(note="cadence")
                last_slot, last_slot_step = time.time(), ns
        else:
            status, reason, rc = "COMPLETE", "", 0
            if last_slot_step != st["next_step"]:
                save_slot(note="final step (before end-of-run evaluation)")
        if status in ("INTERRUPTED", "EXTERNALLY_STOPPED"):
            save_slot(note=f"{status}: {reason}"[:120])
    except KeyboardInterrupt:
        status, reason, rc = "EXTERNALLY_STOPPED", f"KeyboardInterrupt: {stopf.reason}", 77
        try:
            save_slot(note="stop (KeyboardInterrupt)")
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
               "plan_sha256": plan_sha, "cfaudit": st["cfaudit"]}
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
        guard.gpu_ledger(t_start_iso, t_end, rd.run_id, f"cps train {cfg['run_name']}", status)
        from csr.cps_lib import ART
        with open(ART / "compute_ledger.tsv", "a") as f:
            f.write(f"{t_start_iso}\t{t_end}\t{time.time() - t_proc0:.1f}\t{rd.run_id}\t{cfg['run_name']}\t"
                    f"{cfg.get('category', 'main_training')}\t{status}\n")
    with open(rd.path / f"leg_{st['legs']:02d}.json", "w") as f:
        json.dump(summary, f, indent=1, default=str)
    rd.set_status(status, **{k: v for k, v in summary.items() if k not in ("eval", "status", "cfaudit")})
    if status == "COMPLETE":
        guard.atomic_write_json(rd.path / "summary.json", summary, overwrite=True)
        store.prune_after_complete()
    rd.log(f"{status} {reason[:500]}")
    return rc


def end_eval(model, cfg, device):
    """Development end-of-run evaluation on CALIB-SEARCH only: inference mode (no drafts) with the prior, and the same
    model with the context prior disabled (Q1: does the prior carry useful preferences?)."""
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
        model.force_ctx_off = True
        try:
            r0 = evaluate(model, sp, eval_rows(sp), device, routing=True)
        finally:
            model.force_ctx_off = False
        res[sp_name + "_prior_off"] = {"ce": r0["ce"], "ce_dom": r0["ce_dom"], "ent": r0["ent"]}
    return res


if __name__ == "__main__":
    main()
