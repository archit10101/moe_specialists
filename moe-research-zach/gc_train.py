#!/usr/bin/env python
"""csr-gradient-consistent-latent-edl-v1: train ONE main run (GC prereg sections 1, 6, 10). Copy of fle_train.py;
the TOKEN-4 and EDL code paths are textually identical. Added condition:
  GC latent EDL (SEM/GRAD/COMBINED/SHUF)  {"paper_latent": {"table", "table_sha256", "beta", "warmup_frac"?}}
    = fsp_model.edl_loss with the one-hot domain replaced by the frozen per-paper soft assignment q (gc_lib),
      all 11 MoE layers; q is a constant of the training-only loss and never enters a forward pass.
Original fle docstring follows.
csr-forecast-latent-edl-v1: train ONE main run on the lfr balanced main stream (prereg sections 2, 6, 7).

Conditions (cfg["cond"]):
  TOKEN-4                          {}                                   (identical code path to lfr_train TOKEN-4)
  explicit-label EDL               {"edl_beta": 5e-3}                   (identical code path to lfr_train EDL)
  latent EDL (FL / PAST / SHUF)    {"latent": {"table", "beta", "layers", "shuffle"}}
  forecast distillation (FD)       {"fd": {"table", "gamma"}}
The backbone, init, stream, optimizer, schedule, Switch/z losses and routing are identical in every condition; q never
enters a forward pass (it is a constant in the training-only loss). Milestones: cfg["milestones"] maps tag -> step.
Resumable in place (verified slot, KEEP=1); --max-steps stops a search stage with a slot (status INTERRUPTED).
External stop (signal, STOP marker, foreign GPU process): checkpoint at the next step boundary, exit 77.
Exit codes: 0 COMPLETE, 75 INTERRUPTED (max-steps / leg limit), 76 OOM, 77 EXTERNALLY_STOPPED, 1 failure."""
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
    from csr.gc_paths import prereg_sha
    if cfg.get("prereg_sha256") != prereg_sha():
        raise guard.GuardError("config prereg hash does not match the frozen preregistration")
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
    def __init__(self, monitor):
        from csr.gc_paths import STOP_MARKER
        self.marker = STOP_MARKER
        self.reason, self.monitor = None, monitor
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, self._handler)
        self._stop = threading.Event()
        threading.Thread(target=self._watch, daemon=True).start()

    def _handler(self, signum, frame):
        if self.reason is None:
            self.reason = "signal " + signal.Signals(signum).name

    def _watch(self):
        while not self._stop.wait(5.0):
            if self.reason is None and self.marker.exists():
                self.reason = "stop marker STOP"
            elif self.reason is None and self.monitor is not None and self.monitor.contaminated:
                self.reason = f"foreign GPU process: {self.monitor.detail}"

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


def build_optimizer(model, t, device):
    """Identical to lfr_train.build_optimizer with router_lr_mult = 1."""
    import torch
    groups = {}
    for n, p in model.named_parameters():
        k = "decay" if p.dim() >= 2 else "nodecay"
        groups.setdefault(k, []).append(p)
    pg = []
    for k in ("decay", "nodecay"):
        if groups.get(k):
            pg.append({"params": groups[k], "weight_decay": t["weight_decay"] if k == "decay" else 0.0,
                       "lr_mult": 1.0})
    return torch.optim.AdamW(pg, lr=t["lr"], betas=tuple(t["betas"]), eps=1e-8, fused=device.type == "cuda")


def run(cfg, a):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from csr import gc_ckpt as fle_ckpt
    from csr.fle_lib import QTable, fd_loss, fledl_loss
    from csr.gc_paths import ART, RUN_PREFIX
    from csr.gc_lib import PaperQ, gc_edl_loss
    from csr.fsp_lib import make_batch, to_torch
    from csr.lfr_lib import lr_at, main_stream
    from csr.lfr_model import LFRConfig, LFRModel
    torch.set_num_threads(4)
    monitor = None
    if a.device == "cuda":
        guard.occupancy_gate(f"gc train:{cfg['run_name']}")
        device = guard.setup_cuda_device()
        monitor = guard.OccupancyMonitor(interval=15.0)
        monitor.start()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        device = torch.device("cpu")
    guard.register_self(f"gc train {cfg['run_name']} on {a.device}")
    stopf = StopFlag(monitor)
    t = cfg["train"]
    cond = cfg["cond"]
    if a.run_dir:
        rd = InPlaceRun(guard.inside_root(a.run_dir))
        if json.loads((rd.path / "config.json").read_text()) != cfg:
            raise guard.GuardError("resume: config differs")
    else:
        rd = InPlaceRun(guard.RunDir(RUN_PREFIX + cfg["run_name"], cfg).path)
    rd.set_status("RUNNING")
    store = fle_ckpt.ResumeStore(rd.run_id, cfg["run_name"])
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
    edl_on = cond.get("edl_beta", 0) > 0
    lat = cond.get("latent")
    fdc = cond.get("fd")
    plc = cond.get("paper_latent")
    pq = None
    if plc:
        pq = PaperQ(guard.ROOT / plc["table"], split, plc["table_sha256"])
        rd.log(f"paper_latent: K={pq.K} coverage={pq.coverage:.4f} beta={plc['beta']} table {pq.sha[:12]}")
    qt = None
    if lat or fdc:
        tab = (lat or fdc)["table"]
        qt = QTable(guard.ROOT / tab)
        if qt.sha != (lat or fdc)["table_sha256"]:
            raise guard.GuardError("assignment table hash mismatch")
    fd_head = fd_opt = None
    if fdc:
        g_ = torch.Generator().manual_seed(cfg["seed"] + 7_000_003)
        fd_head = torch.nn.Linear(mc.d_model, qt.K).to(device)
        with torch.no_grad():
            fd_head.weight.copy_(torch.randn(fd_head.weight.shape, generator=g_).to(device) * 0.02)
            fd_head.bias.zero_()
        fd_opt = torch.optim.AdamW(fd_head.parameters(), lr=t["lr"], betas=tuple(t["betas"]), eps=1e-8,
                                   weight_decay=0.0)
    st = {"next_step": 0, "seen": 0, "acc": {"step_s": 0.0, "tok_after_warm": 0, "step_s_after_warm": 0.0},
          "ckpts": {}, "legs": 0, "diag": []}
    ck = None
    if a.run_dir:
        ent = store.latest_verified(log=rd.log)
        if ent is not None:
            ck = torch.load(guard.ROOT / ent["path"] / "state.pt", map_location="cpu", weights_only=False)
            if ck["run_id"] != rd.run_id or ck.get("config_sha256") != cfg_sha:
                raise guard.GuardError("resume checkpoint belongs to another run/config")
            model.load_state_dict(ck["model"])
            st = ck["st"]
            rd.log(f"resumed at step {st['next_step']} from {ent['path']} (sha256 {ent['sha256'][:12]} verified)")
        else:
            rd.log("resume requested but no verified checkpoint exists: starting at step 0")
    opt = build_optimizer(model, t, device)
    if ck is not None:
        opt.load_state_dict(ck["opt"])
        if fd_head is not None:
            fd_head.load_state_dict(ck["fd_head"])
            fd_opt.load_state_dict(ck["fd_opt"])
        fle_ckpt.set_rng_state(ck["rng"])
        del ck
    st["legs"] += 1
    compile_on = device.type == "cuda" and not a.no_compile
    fwd = torch.compile(model) if compile_on else model

    def lm_kd(h, w, y):
        lg = F.linear(h, w).float()
        lse = torch.logsumexp(lg, -1)
        valid = y >= 0
        ce = ((lse - lg.gather(1, y.clamp(min=0).view(-1, 1)).squeeze(1)) * valid).sum()
        return ce, torch.zeros((), device=h.device)
    lm_fn = torch.compile(lm_kd) if compile_on else lm_kd

    def payload():
        step_ = st["next_step"]
        p = {"run_id": rd.run_id, "run_name": cfg["run_name"], "config_sha256": cfg_sha,
             "model": model.state_dict(), "opt": opt.state_dict(), "st": st, "rng": fle_ckpt.rng_state(),
             "cursor": {"next_step": step_, "n_steps": n_steps, "rows_per_step": rps, "plan_sha256": plan_sha},
             "scheduler": {"type": "stateless WSD lr_at(step)", "next_lr": lr_at(min(step_, n_steps - 1), n_steps, t)},
             "scaler": None, "amp": "bf16 autocast (no GradScaler)", "driver": fle_ckpt.driver_snapshot(),
             "table_sha256": qt.sha if qt else (pq.sha if pq else None), "time": guard.now()}
        if fd_head is not None:
            p["fd_head"], p["fd_opt"] = fd_head.state_dict(), fd_opt.state_dict()
        return p

    def save_slot(note=""):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0_ = time.time()
        ent_ = store.save(payload(), st["next_step"], st["seen"], note=note)
        st["acc"]["ckpt_s"] = st["acc"].get("ckpt_s", 0.0) + (time.time() - t0_)
        return ent_

    def save_milestone(tag):
        keep32 = lambda k, v: v.dim() <= 1 or "router" in k or "norm" in k
        sd = {k: (v if keep32(k, v) or not v.is_floating_point() else v.to(torch.bfloat16))
              for k, v in model.state_dict().items()}
        p = rd.path / f"ckpt_{tag}"
        step_, seen_ = st["next_step"], st["seen"]
        if p.exists():
            if not fle_ckpt.verify_dir(p):
                raise guard.GuardError(f"existing milestone {p} fails verification")
            meta = json.loads((p / "meta.json").read_text())
            if meta.get("step") != step_:
                raise guard.GuardError(f"existing milestone {p} inconsistent with replay")
            st["ckpts"][tag] = {"path": str(p.relative_to(guard.ROOT)), "sha256": meta["sha256"], "step": step_,
                                "seen": seen_}
            return
        obj = {"model": sd, "config": cfg, "step": step_, "seen": seen_, "dtype": "bf16 matmul weights, fp32 rest",
               "model_cfg": model.cfg.to_dict()}
        sha, nb, sec = fle_ckpt.atomic_ckpt_dir(obj, p, lambda ld: ld["step"] == step_ and len(ld["model"]) == len(sd),
                                                {"step": step_, "seen": seen_, "tag": tag})
        st["ckpts"][tag] = {"path": str(p.relative_to(guard.ROOT)), "sha256": sha, "step": step_, "seen": seen_}
        fle_ckpt.ledger({"time": guard.now(), "run_id": rd.run_id, "run_name": cfg["run_name"], "kind": "milestone",
                         "path": st["ckpts"][tag]["path"], "step": step_, "seen": seen_, "sha256": sha, "bytes": nb,
                         "save_seconds": round(sec, 2), "note": tag})

    ms_steps = {int(v): k for k, v in cfg.get("milestones", {}).items()}
    last_slot = time.time()
    last_slot_step = st["next_step"]
    every_steps = max(1, int(0.10 * n_steps))
    every_s = 600.0
    diag_every = cfg.get("diag_every", 0)
    layers = (lat or {}).get("layers", [])

    def latent_batch(npb, step, mi_):
        perm_seed = None
        if lat and lat.get("shuffle"):
            perm_seed = (cfg["seed"] * 1_000_003 + step) * 7 + mi_
        reg, q = qt.batch(npb["gdoc"], npb["positions"], npb["doc_ids"], perm_seed)
        return torch.as_tensor(reg.reshape(-1), device=device), torch.as_tensor(q, device=device)

    def gc_aux(out, npb, bt):
        """GC-EDL over all 11 MoE layers (fp32, autocast off): mean over layers of gc_edl_loss."""
        E = mc.n_experts
        qs = torch.as_tensor(pq.qseq(npb), device=device)
        sid = bt["seq_id"].reshape(-1)
        nonpad = (bt["doc_ids"] >= 0).reshape(-1)
        with torch.autocast(device_type=device.type, enabled=False):
            ls = [gc_edl_loss(out["records"][li]["p"].reshape(-1, E).float(), qs, sid, nonpad)
                  for li in range(model.n_moe)]
            lf = torch.stack(ls).mean()
        return lf, float(torch.exp(-torch.stack(ls).detach()).mean())

    def gc_beta(step):
        wf = plc.get("warmup_frac") or 0.0
        if wf <= 0:
            return plc["beta"]
        return plc["beta"] * min(1.0, step / max(1.0, wf * n_steps))

    def grad_diag(step):
        """Per-component gradient norms on backbone parameters (eager, separate pass on 8 rows of the step;
        gradients discarded; the training update is unaffected)."""
        params = [p for p in model.parameters() if p.requires_grad]
        rr = steps[step][:8]
        npb = make_batch(rr, split)
        bt = to_torch(npb, device)
        minp = {"input_ids": bt["input_ids"], "doc_ids": bt["doc_ids"], "positions": bt["positions"]}
        if edl_on:
            minp["domain"], minp["seq_id"] = bt["domain"], bt["seq_id"]
        ctl = {"teach": False, "edl": edl_on, "lc": False, "cf_store": None, "tap": False,
               "collect": bool(lat) or bool(plc)}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            out = model(minp, ctl)
            h = out["hidden"]
            hf, y = h.reshape(-1, h.shape[-1]), bt["targets"].reshape(-1)
            nv = int((y >= 0).sum())
            lm = sum(lm_kd(hf[i:i + 8192], model.tok_emb.weight, y[i:i + 8192])[0]
                     for i in range(0, y.numel(), 8192)) / nv
            lb = t["aux_coeff"] * out["aux"] + t["z_coeff"] * out["z"]
            comps = {"lm": lm, "lb": lb}
            rec = {"step": step}
            if edl_on:
                comps["aux"] = cond["edl_beta"] * out["edl"]
            if lat:
                reg, q = latent_batch(npb, step, 99)
                E = mc.n_experts
                pl = [out["records"][li]["p"].reshape(-1, E) for li in layers]
                lf, jm = fledl_loss(pl, reg, q)
                comps["aux"] = lat["beta"] * lf
                rec["fledl"], rec["jsd"] = float(lf), float(jm)
                # routing-separation at all layers (diagnostic)
                allj = [float(fledl_loss([out["records"][li]["p"].reshape(-1, E)], reg, q)[1])
                        for li in range(model.n_moe)]
                rec["jsd_by_layer"] = allj
            if plc:
                lf, jm = gc_aux(out, npb, bt)
                comps["aux"] = gc_beta(step) * lf
                comps["gc_unit"] = lf
                rec["gcedl"], rec["jsd"] = float(lf), jm
            if fdc:
                reg, q = latent_batch(npb, step, 99)
                comps["aux"] = fdc["gamma"] * fd_loss(h, reg, q, fd_head)
                comps["fd_unit"] = fd_loss(h, reg, q, fd_head)
        for k, v in comps.items():
            if not v.requires_grad:
                rec[f"gn_{k}"] = 0.0
                continue
            gs = torch.autograd.grad(v, params, retain_graph=True, allow_unused=True)
            rec[f"gn_{k}"] = float(torch.sqrt(sum((g_.float() ** 2).sum() for g_ in gs if g_ is not None)))
            rec[f"val_{k}"] = float(v.detach())
        model.zero_grad(set_to_none=True)
        return rec

    try:
        rd.log(f"params: {model.param_counts()}" + (f" + fd_head training-only {sum(p.numel() for p in fd_head.parameters())}"
                                                     if fd_head is not None else ""))
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
            if diag_every and step % diag_every == 0:
                drec = grad_diag(step)
                st["diag"].append(drec)
                rd.metric({"diag": drec})
                model.train()
            lr = lr_at(step, n_steps, t)
            for g in opt.param_groups:
                g["lr"] = lr * g.get("lr_mult", 1.0)
            if fd_opt is not None:
                for g in fd_opt.param_groups:
                    g["lr"] = lr
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            rows = steps[step]
            n_valid = int(sum(split.lengths[d] - 1 for r in rows for d in r))
            parts = {}
            for mi_ in range(accum):
                rr = rows[mi_ * mbr:(mi_ + 1) * mbr]
                npb = make_batch(rr, split)
                bt = to_torch(npb, device)
                minp = {"input_ids": bt["input_ids"], "doc_ids": bt["doc_ids"], "positions": bt["positions"]}
                if edl_on:
                    minp["domain"], minp["seq_id"] = bt["domain"], bt["seq_id"]
                ctl = {"teach": False, "edl": edl_on, "lc": False, "cf_store": None, "tap": False}
                if plc:
                    ctl["collect"] = True
                if lat:
                    ctl["collect"] = True
                    reg, q = latent_batch(npb, step, mi_)
                elif fdc:
                    reg, q = latent_batch(npb, step, mi_)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                    out = fwd(minp, ctl)
                    h = out["hidden"]
                    hflat, y = h.reshape(-1, h.shape[-1]), bt["targets"].reshape(-1)
                    lm = torch.zeros((), device=device)
                    for i in range(0, y.numel(), 8192):
                        c_, _ = lm_fn(hflat[i:i + 8192], model.tok_emb.weight, y[i:i + 8192])
                        lm = lm + c_
                    loss = lm / n_valid + (t["aux_coeff"] * out["aux"] + t["z_coeff"] * out["z"]) / accum
                    if edl_on:
                        loss = loss + cond["edl_beta"] * out["edl"] / accum
                    if lat:
                        E = mc.n_experts
                        pl = [out["records"][li]["p"].reshape(-1, E) for li in layers]
                        lf, jm = fledl_loss(pl, reg, q)
                        loss = loss + lat["beta"] * lf / accum
                        parts["fledl"] = parts.get("fledl", 0.0) + float(lf.detach()) / accum
                        parts["jsd"] = parts.get("jsd", 0.0) + float(jm) / accum
                        parts["n_reg"] = parts.get("n_reg", 0) + int(q.shape[0])
                    if plc:
                        lf, jm = gc_aux(out, npb, bt)
                        loss = loss + gc_beta(step) * lf / accum
                        parts["gcedl"] = parts.get("gcedl", 0.0) + float(lf.detach()) / accum
                        parts["jsd"] = parts.get("jsd", 0.0) + jm / accum
                    if fdc:
                        lfd = fd_loss(h, reg, q, fd_head)
                        loss = loss + fdc["gamma"] * lfd / accum
                        parts["fd"] = parts.get("fd", 0.0) + float(lfd.detach()) / accum
                loss.backward()
                for k in ("aux", "z", "edl"):
                    parts[k] = parts.get(k, 0.0) + float(out[k].detach()) / accum
                parts["lm"] = parts.get("lm", 0.0) + float(lm.detach())
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), t["clip"])
            opt.step()
            opt.zero_grad(set_to_none=True)
            if fd_opt is not None:
                torch.nn.utils.clip_grad_norm_(fd_head.parameters(), t["clip"])
                fd_opt.step()
                fd_opt.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - t0
            lm_mean = parts["lm"] / n_valid
            if not math.isfinite(lm_mean) or not all(math.isfinite(v) for v in parts.values()):
                raise FloatingPointError(f"non-finite loss at step {step}")
            st["next_step"] = step + 1
            st["seen"] += n_valid
            st["acc"]["step_s"] += dt
            if step >= 10:
                st["acc"]["tok_after_warm"] += n_valid
                st["acc"]["step_s_after_warm"] += dt
            if step % t.get("log_every", 50) == 0 or step == n_steps - 1:
                rec = {"step": step, "lr": lr, "seen": st["seen"], "lm": lm_mean, "gn": float(gn), "dt": dt,
                       **{k: v for k, v in parts.items() if k != "lm"}}
                rd.metric(rec)
                if step % (t.get("log_every", 50) * 10) == 0:
                    rd.log(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}))
            ns = st["next_step"]
            if ns in ms_steps:
                save_milestone(ms_steps[ns])
                save_slot(note=f"milestone {ms_steps[ns]}")
                last_slot, last_slot_step = time.time(), ns
            elif ns - last_slot_step >= every_steps or time.time() - last_slot >= every_s:
                save_slot(note="cadence")
                last_slot, last_slot_step = time.time(), ns
        else:
            status, reason, rc = "COMPLETE", "", 0
            if last_slot_step != st["next_step"]:
                save_slot(note="final step")
        if status in ("INTERRUPTED", "EXTERNALLY_STOPPED") and last_slot_step != st["next_step"]:
            save_slot(note=f"{status}: {reason}"[:120])
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
               "peak_alloc_gib": peak[0], "peak_reserved_gib": peak[1], "ckpts": st["ckpts"], "legs": st["legs"],
               "micro_batch_rows": mbr, "ckpt_seconds": st["acc"].get("ckpt_s", 0.0), "params": model.param_counts(),
               "plan_sha256": plan_sha, "table_sha256": qt.sha if qt else (pq.sha if pq else None)}
    stopf.close()
    if monitor is not None:
        monitor.stop()
    if a.device == "cuda":
        guard.gpu_ledger(t_start_iso, t_end, rd.run_id, f"gc train {cfg['run_name']}", status)
        with open(ART / "compute_ledger.tsv", "a") as f:
            f.write(f"{t_start_iso}\t{t_end}\t{time.time() - t_proc0:.1f}\t{rd.run_id}\t{cfg['run_name']}\t"
                    f"{cfg.get('category', 'main_training')}\t{status}\n")
    with open(rd.path / f"leg_{st['legs']:02d}.json", "w") as f:
        json.dump(summary, f, indent=1, default=str)
    rd.set_status(status, **{k: v for k, v in summary.items() if k != "status"})
    if status == "COMPLETE":
        guard.atomic_write_json(rd.path / "summary.json", summary, overwrite=True)
        store.prune_after_complete()
    rd.log(f"{status} {reason[:500]}")
    return rc


if __name__ == "__main__":
    main()
