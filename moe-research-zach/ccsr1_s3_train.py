#!/usr/bin/env python
# csr-current-context-specialist-router-v1 (ccsr1) Stage 3 trainer (docs/CSR_CURRENT_CONTEXT_SPECIALIST_ROUTER_V1_STAGE3_AMENDMENT.md).
"""Matched joint continuation: the cfar1 TOKEN-4 base continues its 245M plan from milestone m20 (step 318) exactly as
cfar1_train.py (same stream, WSD lr_at schedule, micro-batching, bf16 autocast, torch.compile, Switch + z losses), with
fresh AdamW moments and a 20-step re-warmup, plus ten residual banks trained on the DETACHED step hidden states:

  G         jsr token-routed generic bank (residual_step, as cfar1)
  SEMf_r    copy of L, deployed by the FROZEN Stage-1 causal router R3_c2e0.5/SEM_s{r}
  SEMj_r    copy of L, router initialised from SEM_s{r} and updated by continued R3 distillation
  SHUFf_r   copy of F2h_T500_w240, deployed by the frozen Stage-1 router R3_c2e0.5/SHUF_s{r}

Per step: base forward/backward -> residual phase: router choices e = argmax q (all arms, before any update) ->
joint routers: R3 KL on the rotating 1-in-4 scored-token subset (exact utilities of the arm's live specialists) ->
chosen specialists: LM CE on sg(h), sg(W) -> base step, bank steps, joint-router steps. No bank ever produces a gradient
on a base parameter; no federation sync; no auxiliary specialisation loss.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--run-dir")
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--max-seconds", type=float, default=4700)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--test-signal-at-step", type=int)
    ap.add_argument("--test-rows", type=int, default=0, help="CPU tests: use only the first N rows of each step")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    cfg = json.loads(guard.inside_root(a.config).read_text())
    if not cfg.get("test"):
        from csr.ccsr1_paths import check_prereg
        from csr.ccsr1_s3_paths import check_amendment
        if cfg.get("prereg_sha256") != check_prereg() or cfg.get("amend_sha256") != check_amendment():
            raise guard.GuardError("config protocol hashes do not match the frozen prereg / Stage-3 amendment")
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
        from csr.ccsr1_paths import STOP_MARKER
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


def doc_batch(dkey, pos):
    """Scored tokens of a step, flattened in row order: documents are contiguous runs of dkey with positions 0..n-1.
    Returns padded index (B, L) into the flat order, validity mask, lengths (all on device)."""
    import torch
    dev = dkey.device
    N = dkey.numel()
    brk = torch.nonzero(dkey[1:] != dkey[:-1]).squeeze(1) + 1
    starts = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), brk])
    lens = torch.diff(torch.cat([starts, torch.tensor([N], device=dev)]))
    if torch.unique(dkey[starts]).numel() != starts.numel():
        raise guard.GuardError("document tokens are not contiguous in the step")
    exp = torch.arange(N, device=dev) - torch.repeat_interleave(starts, lens)
    if not bool((exp == pos).all()):
        raise guard.GuardError("scored positions are not 0..n-1 within each document")
    L = int(lens.max())
    t = torch.arange(L, device=dev).view(1, -1)
    mask = t < lens.view(-1, 1)
    idx = (starts.view(-1, 1) + t).clamp(max=N - 1)
    return idx, mask, lens


def route(r, X, idx, mask, lens, grad=False):
    """Router logits for every scored token (flat order) and the deployed argmax choice."""
    import torch
    with torch.set_grad_enabled(grad):
        lg = r(X[idx], lens)[mask]
    ti = idx[mask]
    out = torch.empty_like(lg)
    out = out.index_copy(0, ti, lg) if grad else out.index_copy_(0, ti, lg)
    return out, out.detach().argmax(1)


def run(cfg, a):
    import numpy as np
    import torch
    from csr import ccsr1_ckpt as CK
    from csr.cfar1_bank import FedBank
    from csr.cfar1_paths import ALPHA_G
    from csr.ccsr1_router import CausalRouter, router_loss, teacher, token_weight
    from csr.csur1_bank import all_deltas, finite
    from csr.jsr_bank import Bank, bank_optimizer, head_ce_grad, residual_step, rms0
    from csr.fsp_lib import make_batch, to_torch
    from csr.lfr_lib import lr_at, main_stream
    from csr.lfr_model import LFRConfig, LFRModel
    from cfar1_train import build_optimizer, load_psem
    torch.set_num_threads(4)
    monitor = None
    if a.device == "cuda":
        guard.occupancy_gate(f"ccsr1 s3 train:{cfg['run_name']}")
        device = guard.setup_cuda_device()
        monitor = guard.OccupancyMonitor(interval=15.0)
        monitor.start()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        device = torch.device("cpu")
    guard.register_self(f"ccsr1 s3 train {cfg['run_name']} on {a.device}")
    stopf = StopFlag(monitor)
    base_cfg = cfg["base_config"]
    t = base_cfg["train"]
    J = cfg["joint"]
    if a.run_dir:
        rd = InPlaceRun(guard.inside_root(a.run_dir))
        if json.loads((rd.path / "config.json").read_text()) != cfg:
            raise guard.GuardError("resume: config differs")
    else:
        rd = InPlaceRun(guard.RunDir("ccsr1_" + cfg["run_name"], cfg).path)
    rd.set_status("RUNNING")
    store = CK.ResumeStore(rd.run_id, cfg["run_name"])
    cfg_sha = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    t_start_iso, t_proc0 = guard.now(), time.time()
    limits = guard.RunLimits(max_seconds=a.max_seconds + 600, monitor=monitor)
    status, reason, rc = "INCOMPLETE", "", 1
    mbr = t["micro_batch_rows"]
    rps = t["rows_per_step"]
    accum = rps // mbr
    use_amp = device.type == "cuda"
    split, steps, planned_tokens = main_stream(base_cfg["budget_tokens"])
    if a.test_rows:
        steps = [s[:a.test_rows] for s in steps]
        mbr = max(1, a.test_rows // accum)
    n_steps = len(steps)
    start, end = int(cfg["start_step"]), int(cfg["end_step"])
    assert 0 < start < end <= n_steps
    plan_sha = hashlib.sha256(json.dumps(steps).encode()).hexdigest()
    rd.log(f"stream: {n_steps} steps (continuing {start} -> {end}), micro {mbr} x accum {accum}, plan {plan_sha[:12]}")
    # ---- base and banks from the reused cfar1 milestone (hash-verified)
    ckp = guard.ROOT / cfg["base_ckpt"] / "state.pt"
    if not cfg.get("test") or cfg.get("verify_ckpt", True):
        if guard.sha256_file(ckp) != cfg["base_ckpt_sha256"]:
            raise guard.GuardError("reused checkpoint hash mismatch")
    ck0 = torch.load(ckp, map_location="cpu", weights_only=False)
    if int(ck0["step"]) != start:
        raise guard.GuardError(f"milestone step {ck0['step']} != start {start}")
    mc = LFRConfig.from_dict({**ck0["model_cfg"], "teacher": False})
    model = LFRModel(mc, base_cfg["seed"])
    model.load_state_dict(ck0["model"], strict=True)
    model = model.to(device)
    arms = cfg["arms"]
    banks, routers, ropt, rmeta = {}, {}, {}, {}
    for arm in arms:
        src = arm["init"]
        if arm["router"] == "token":
            b = Bank({"name": src, "kind": "routed", "tau": 1.0, "r": 0.0, "alpha": ALPHA_G, "groups": None},
                     base_cfg["seed"])
        else:
            b = FedBank(ck0["bank_specs"][src], base_cfg["seed"])
        b.load_state_dict(ck0["banks"][src])
        b = b.to(device)
        if arm["router"] != "token":
            for p in b.router_params():
                p.requires_grad_(False)
            rs = torch.load(guard.ROOT / arm["router_ckpt"], map_location="cpu", weights_only=False)
            if guard.sha256_file(guard.ROOT / arm["router_ckpt"]) != arm["router_sha256"]:
                raise guard.GuardError(f"router checkpoint hash mismatch {arm['router_ckpt']}")
            rc_ = rs["cfg"]
            r = CausalRouter(int(rc_["H"]), int(rs["seed"]), rc_.get("feats", "base"), int(rc_.get("n_out", 16)))
            r.load_state_dict(rs["state"])
            r = r.to(device)
            if arm["router"] == "frozen":
                r.eval()
                for p in r.parameters():
                    p.requires_grad_(False)
            else:
                r.train()
                ropt[arm["name"]] = torch.optim.AdamW(r.parameters(), lr=J["lr"], weight_decay=J["wd"],
                                                      betas=tuple(J["betas"]))
            routers[arm["name"]] = r
            rmeta[arm["name"]] = {"cfg": rc_, "stats": rs["stats"], "prior": "sem" if rs["bank"] == "SEM" else "shuf"}
        banks[arm["name"]] = b
    del ck0
    psem_off, psem_P, psem_Psh, psem_sha = load_psem(split, device)
    rd.log(f"arms: {[(x['name'], x['init'], x['router']) for x in arms]} psem {psem_sha[:12]}")
    st = {"next_step": start, "seen": 0, "acc": {"step_s": 0.0, "tok_after_warm": 0, "step_s_after_warm": 0.0},
          "ckpts": {}, "legs": 0}

    def bank_opt(name, b):
        if isinstance(b, FedBank):
            return torch.optim.AdamW(b.expert_params(), lr=t["lr"], betas=tuple(t["betas"]), eps=1e-8,
                                     weight_decay=t["weight_decay"], fused=device.type == "cuda")
        return bank_optimizer(b, t, device)
    opt = build_optimizer(model, t, device)
    bopt = {n: bank_opt(n, b) for n, b in banks.items()}
    ck = None
    if a.run_dir:
        ent = store.latest_verified(log=rd.log)
        if ent is not None:
            ck = torch.load(guard.ROOT / ent["path"] / "state.pt", map_location="cpu", weights_only=False)
            if ck["run_id"] != rd.run_id or ck.get("config_sha256") != cfg_sha:
                raise guard.GuardError("resume checkpoint belongs to another run/config")
            model.load_state_dict(ck["model"])
            st = ck["st"]
            for n, b in banks.items():
                b.load_state_dict(ck["banks"][n])
                if n in ck["bank_ctrl"]:
                    b.ctrl = dict(ck["bank_ctrl"][n])
            for n, r in routers.items():
                r.load_state_dict(ck["routers"][n])
            opt.load_state_dict(ck["opt"])
            for n in banks:
                bopt[n].load_state_dict(ck["bank_opt"][n])
            for n in ropt:
                ropt[n].load_state_dict(ck["router_opt"][n])
            CK.set_rng_state(ck["rng"])
            rd.log(f"resumed at step {st['next_step']} from {ent['path']} (sha256 {ent['sha256'][:12]} verified)")
            del ck
        else:
            rd.log("resume requested but no verified checkpoint exists: starting at the milestone")
    st["legs"] += 1
    compile_on = device.type == "cuda" and not a.no_compile
    fwd = torch.compile(model) if compile_on else model

    def lm_sum(h, w, y):
        lg = torch.nn.functional.linear(h, w).float()
        lse = torch.logsumexp(lg, -1)
        valid = y >= 0
        return ((lse - lg.gather(1, y.clamp(min=0).view(-1, 1)).squeeze(1)) * valid).sum()
    lm_fn = torch.compile(lm_sum) if compile_on else lm_sum

    def payload():
        return {"run_id": rd.run_id, "run_name": cfg["run_name"], "config_sha256": cfg_sha, "model": model.state_dict(),
                "opt": opt.state_dict(), "st": st, "rng": CK.rng_state(),
                "cursor": {"next_step": st["next_step"], "n_steps": n_steps, "rows_per_step": rps, "plan_sha256": plan_sha},
                "banks": {n: b.state_dict() for n, b in banks.items()},
                "bank_opt": {n: o.state_dict() for n, o in bopt.items()},
                "bank_ctrl": {n: dict(b.ctrl) for n, b in banks.items() if hasattr(b, "ctrl")},
                "routers": {n: r.state_dict() for n, r in routers.items()},
                "router_opt": {n: o.state_dict() for n, o in ropt.items()}, "time": guard.now()}

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
            if not CK.verify_dir(p):
                raise guard.GuardError(f"existing milestone {p} fails verification")
            meta = json.loads((p / "meta.json").read_text())
            if meta.get("step") != step_:
                raise guard.GuardError(f"existing milestone {p} inconsistent with replay")
            st["ckpts"][tag] = {"path": str(p.relative_to(guard.ROOT)), "sha256": meta["sha256"], "step": step_}
            return
        obj = {"model": sd, "banks": {n: {k: v.detach().cpu() for k, v in b.state_dict().items()} for n, b in banks.items()},
               "bank_specs": {n: b.spec for n, b in banks.items()}, "arms": arms,
               "routers": {n: {k: v.detach().cpu() for k, v in r.state_dict().items()} for n, r in routers.items()},
               "router_meta": rmeta, "config": cfg, "step": step_, "seen": seen_, "model_cfg": model.cfg.to_dict(),
               "base_seed": base_cfg["seed"], "dtype": "base: bf16 matmul weights, fp32 rest; banks and routers fp32"}
        sha, nb, sec = CK.atomic_ckpt_dir(obj, p, lambda ld: ld["step"] == step_ and len(ld["model"]) == len(sd),
                                          {"step": step_, "seen": seen_, "tag": tag})
        st["ckpts"][tag] = {"path": str(p.relative_to(guard.ROOT)), "sha256": sha, "step": step_, "seen": seen_}
        CK.ledger({"time": guard.now(), "run_id": rd.run_id, "run_name": cfg["run_name"], "kind": "milestone",
                   "path": st["ckpts"][tag]["path"], "step": step_, "seen": seen_, "sha256": sha, "bytes": nb,
                   "save_seconds": round(sec, 2), "note": tag})

    ms_steps = {int(v): k for k, v in cfg.get("milestones", {}).items()}
    last_slot, last_slot_step = time.time(), st["next_step"]
    every_steps, every_s = int(cfg.get("slot_every_steps", 100)), float(cfg.get("slot_every_s", 600))
    rewarm = int(cfg.get("rewarm", 20))
    try:
        rd.log(f"params: {model.param_counts()} banks { {n: b.n_params() for n, b in banks.items()} } "
               f"routers { {n: r.n_params() for n, r in routers.items()} }")
        model.train()
        while st["next_step"] < end:
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
            lr = lr_at(step, n_steps, t) * min(1.0, (step - start + 1) / rewarm)
            for g in opt.param_groups:
                g["lr"] = lr * g.get("lr_mult", 1.0)
            for o in bopt.values():
                for g in o.param_groups:
                    g["lr"] = lr
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            rows = steps[step]
            n_valid = int(sum(split.lengths[d] - 1 for r in rows for d in r))
            parts = {}
            keep = {"h": [], "y": [], "gd": [], "row": [], "dl": [], "pos": []}
            for mi_ in range(accum):
                rr = rows[mi_ * mbr:(mi_ + 1) * mbr]
                bt = to_torch(make_batch(rr, split), device)
                minp = {"input_ids": bt["input_ids"], "doc_ids": bt["doc_ids"], "positions": bt["positions"]}
                ctl = {"teach": False, "edl": False, "lc": False, "cf_store": None, "tap": False}
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                    out = fwd(minp, ctl)
                    h = out["hidden"]
                    hflat, y = h.reshape(-1, h.shape[-1]), bt["targets"].reshape(-1)
                    lm = torch.zeros((), device=device)
                    for i in range(0, y.numel(), 8192):
                        lm = lm + lm_fn(hflat[i:i + 8192], model.tok_emb.weight, y[i:i + 8192])
                    loss = lm / n_valid + (t["aux_coeff"] * out["aux"] + t["z_coeff"] * out["z"]) / accum
                loss.backward()
                v_ = bt["doc_ids"].reshape(-1) >= 0
                keep["h"].append(hflat.detach()[v_])
                keep["y"].append(y[v_])
                keep["gd"].append(bt["gdoc"].reshape(-1)[v_])
                rowt = (torch.arange(bt["doc_ids"].shape[0], device=device) + mi_ * mbr).view(-1, 1).expand_as(bt["doc_ids"])
                keep["row"].append(rowt.reshape(-1)[v_])
                keep["dl"].append(bt["doc_ids"].reshape(-1)[v_])
                keep["pos"].append(bt["positions"].reshape(-1)[v_])
                for k in ("aux", "z"):
                    parts[k] = parts.get(k, 0.0) + float(out[k].detach()) / accum
                parts["lm"] = parts.get("lm", 0.0) + float(lm.detach())
            # ---------------- residual phase (detached hidden states, pre-step tied head)
            Hall, Yall = torch.cat(keep["h"]), torch.cat(keep["y"])
            sc = Yall >= 0
            Hs, Ys = Hall[sc], Yall[sc]
            GDs = torch.cat(keep["gd"])[sc]
            pos_s = torch.cat(keep["pos"])[sc].long()
            dkey = torch.cat(keep["row"])[sc].long() * 4096 + torch.cat(keep["dl"])[sc].long()
            del keep, Hall, Yall
            Wd = model.tok_emb.weight.detach()
            xn_s = rms0(Hs)
            N = Hs.shape[0]
            X = xn_s.to(torch.float16)
            idx, mask, lens = doc_batch(dkey, pos_s)
            o_ = psem_off[GDs]
            if bool((o_ < 0).any()):
                raise guard.GuardError("doc without p_sem regions in the step")
            rid = o_ + torch.div(pos_s, 256, rounding_mode="floor")
            um = (pos_s + step) % int(J["ustride"]) == 0
            brec = {}
            choice = {}
            for n, r in routers.items():
                joint = n in ropt
                lg, e = route(r, X, idx, mask, lens, grad=joint)
                choice[n] = e
                rec_r = {}
                if joint:
                    m_ = rmeta[n]
                    tok = torch.nonzero(um).squeeze(1)
                    with torch.no_grad():
                        u, _ = all_deltas(banks[n].experts.w1, banks[n].experts.w2, Hs[tok], xn_s[tok], Wd, Ys[tok])
                        P_ = psem_P if m_["prior"] == "sem" else psem_Psh
                        lp = torch.log(P_[rid[tok]] + 1e-4)
                        pi = teacher("R3", u, lp, float(J["c"]), float(J["eta"]), m_["stats"]["s_u"])
                        w = token_weight("R3", u, m_["stats"]["med_margin"])
                    rl, rr_ = router_loss(lg[tok], pi, w, float(J["lam"]))
                    ropt[n].zero_grad(set_to_none=True)
                    rl.backward()
                    rec_r = {"r_loss": float(rl.detach()), **rr_,
                             "r_gn": float(torch.nn.utils.clip_grad_norm_(r.parameters(), float(J["clip"]))),
                             "acc_pi": float((e[tok] == pi.argmax(1)).float().mean()), "n_u": int(tok.numel())}
                    del lg, u, pi, w, lp
                f = torch.bincount(e, minlength=16).float() / N
                same = dkey[1:] == dkey[:-1]
                rec_r.update({"load_max": float(f.max()), "n_dead": int((f < 1 / 160).sum()),
                              "switch": float((e[1:] != e[:-1])[same].float().mean())})
                brec[n] = rec_r
            for n, b in banks.items():
                if isinstance(b, FedBank):
                    e = choice[n]
                    a_ = b.correction(xn_s, e)
                    ce, ga = head_ce_grad(Hs, a_.detach(), Wd, Ys)
                    a_.backward(ga / N)
                    brec[n]["res"] = float(ce.mean())
                    brec[n]["gn"] = float(torch.nn.utils.clip_grad_norm_(b.expert_params(), t["clip"]))
                    del a_, ga
                else:
                    dummy = torch.full_like(Ys, -1)
                    rec_b = residual_step(b, Hs, Wd, Ys, dummy, dummy, xn=xn_s)
                    rec_b["gn"] = float(torch.nn.utils.clip_grad_norm_(b.parameters(), t["clip"]))
                    brec[n] = rec_b
                if not finite(brec[n]):
                    raise FloatingPointError(f"non-finite bank {n} at step {step}: {brec[n]}")
            for nj in ropt:
                nf = nj.replace("SEMj", "SEMf")
                if nf in choice:
                    brec[nj]["agree_frozen"] = float((choice[nj] == choice[nf]).float().mean())
            del Hs, Ys, xn_s, X, idx, mask, lens, choice
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), t["clip"])
            opt.step()
            opt.zero_grad(set_to_none=True)
            for n in banks:
                bopt[n].step()
                bopt[n].zero_grad(set_to_none=True)
            for n in ropt:
                ropt[n].step()
                ropt[n].zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - t0
            lm_mean = parts["lm"] / n_valid
            if not math.isfinite(lm_mean) or not all(math.isfinite(v) for v in parts.values()):
                raise FloatingPointError(f"non-finite loss at step {step}")
            st["next_step"] = step + 1
            st["seen"] += n_valid
            st["acc"]["step_s"] += dt
            if step - start >= 5:
                st["acc"]["tok_after_warm"] += n_valid
                st["acc"]["step_s_after_warm"] += dt
            rec = {"step": step, "lr": lr, "seen": st["seen"], "lm": lm_mean, "gn": float(gn), "dt": dt,
                   **{k: v for k, v in parts.items() if k != "lm"}, "banks": brec}
            rd.metric(rec)
            if step % 50 == 0:
                rd.log(json.dumps({"step": step, "lm": round(lm_mean, 4), "dt": round(dt, 2), "lr": lr,
                                   "res": {n: round(v.get("res", float("nan")), 4) for n, v in brec.items()}}))
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
    summary = {"status": status, "reason": reason, "steps_done": st["next_step"], "start": start, "end": end,
               "seen": st["seen"], "leg_seconds": time.time() - t_proc0, "train_step_seconds": st["acc"]["step_s"],
               "tokens_per_s": st["acc"]["tok_after_warm"] / max(1e-9, st["acc"]["step_s_after_warm"]),
               "peak_alloc_gib": peak[0], "peak_reserved_gib": peak[1], "ckpts": st["ckpts"], "legs": st["legs"],
               "ckpt_seconds": st["acc"].get("ckpt_s", 0.0), "plan_sha256": plan_sha}
    stopf.close()
    if monitor is not None:
        monitor.stop()
    if a.device == "cuda":
        guard.gpu_ledger(t_start_iso, guard.now(), rd.run_id, f"ccsr1 s3 train {cfg['run_name']}", status)
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
