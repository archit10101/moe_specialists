#!/usr/bin/env python
# csr-continuous-base-target-free-asrs-v2 (asrs2) trainer (docs/CSR_CONTINUOUS_BASE_TARGET_FREE_ASRS_V2_PREREG.md).
# Base path copied from ccsr1_s3_train.py (imported helpers, never modified).
"""Two modes, one process each, one base, all arms on identical rows:

mode 'frozen' (Stage 0F): base FROZEN at the cfar1 step-318 milestone (forward only). Steps 0..317 of main_stream(245e6)
  (the rows the base saw). Per specialist seed s: G5_s (token-routed generic, experts + router every step), SEM5_s and
  SHUF5_s (hard target-free R1 assignment e = argmax p5 / p5_sh of the token's current 256-token region, every step),
  causal routers RSEM_s / RSHUF_s distilled from p5 / p5_sh (KL + 0.01 balance) every step. G5/SEM5/SHUF5 of a seed share
  the expert initialisation. Output: milestone ckpt_F (banks + routers; the base is the reused milestone).

mode 'cont' (continuation): steps 318..1110 (50,086,846 tokens). The base receives its ORDINARY LM update (LM + Switch aux
  + z, exactly ccsr1 S3) on EVERY step; residual arms see only the detached hidden states and the detached pre-step head.
  Auxiliary windows of 48 steps: offsets 0..31 SPEC, 32..47 ROUTER. Per seed s (initialised from ckpt_F):
    G_s       ordinary generic: experts + router every step
    ALTG_s    experts in SPEC windows only, router in ROUTER windows only (schedule-matched generic)
    SEMh_s / SHUFh_s   SPEC: assigned specialist e = argmax p (target-free); ROUTER: router R1 distillation
    SEMs_s / SHUFs_s   SPEC: e ~ p (one sample per token from a generator seeded by (seed, step, bank)); ROUTER: R1
    FROZ_s    the frozen Stage-0F SEM5 bank (never updated) + its router refreshed in ROUTER windows (R1)
  No token's target ever enters a specialist choice. Held-out diagnostics on a 1-in-4 token subset of every step,
  computed before any update of that step.
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
    ap.add_argument("--max-seconds", type=float, default=4800)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--test-signal-at-step", type=int)
    ap.add_argument("--test-rows", type=int, default=0, help="CPU tests: use only the first N rows of each step")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    cfg = json.loads(guard.inside_root(a.config).read_text())
    if not cfg.get("test"):
        from csr.asrs2_paths import check_prereg
        if cfg.get("prereg_sha256") != check_prereg():
            raise guard.GuardError("config prereg hash does not match the frozen preregistration")
    print(json.dumps({"config": a.config, "run_name": cfg["run_name"], "mode": cfg["mode"], "device": a.device,
                      "resume": a.run_dir}))
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
        from csr.asrs2_paths import STOP_MARKER
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
            self.poll()

    def poll(self):
        if self.reason is None and self.marker.exists():
            self.reason = "stop marker STOP"
        elif self.reason is None and self.monitor is not None and self.monitor.contaminated:
            self.reason = f"foreign GPU process: {self.monitor.detail}"
        return self.reason

    def close(self):
        self._stop.set()


def arm_list(cfg):
    """Expand cfg['arms'] (kinds) x cfg['seeds'] -> [(name, kind, seed)]."""
    return [(f"{k}_{s}", k, s) for s in cfg["seeds"] for k in cfg["arms"]]


def run(cfg, a):
    import numpy as np
    import torch
    from csr import asrs2_ckpt as CK
    from csr import asrs2_paths as P
    from csr.asrs2_lib import Bank5, assign, deltas, gen_grad, r1_loss, r1_target, rms0, spec_grad
    from csr.ccsr1_router import CausalRouter
    from csr.fsp_lib import make_batch, to_torch
    from csr.lfr_lib import lr_at, main_stream
    from csr.lfr_model import LFRConfig, LFRModel
    from cfar1_train import build_optimizer
    from ccsr1_s3_train import InPlaceRun, doc_batch, route
    torch.set_num_threads(4)
    monitor = None
    if a.device == "cuda":
        guard.occupancy_gate(f"asrs2 train:{cfg['run_name']}")
        device = guard.setup_cuda_device()
        monitor = guard.OccupancyMonitor(interval=15.0)
        monitor.start()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        device = torch.device("cpu")
    guard.register_self(f"asrs2 train {cfg['run_name']} on {a.device}")
    stopf = StopFlag(monitor)
    mode = cfg["mode"]
    frozen_mode = mode == "frozen"
    base_cfg = cfg["base_config"]
    t = base_cfg["train"]
    RO = cfg["router_opt"]
    K = int(cfg["K"])
    if a.run_dir:
        rd = InPlaceRun(guard.inside_root(a.run_dir))
        if json.loads((rd.path / "config.json").read_text()) != cfg:
            raise guard.GuardError("resume: config differs")
    else:
        rd = InPlaceRun(guard.RunDir(P.RUN_PREFIX + cfg["run_name"], cfg).path)
    rd.set_status("RUNNING")
    store = CK.ResumeStore(rd.run_id, cfg["run_name"])
    cfg_sha = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    t_start_iso, t_proc0 = guard.now(), time.time()
    limits = guard.RunLimits(max_seconds=a.max_seconds + 900, monitor=monitor)
    status, reason, rc = "INCOMPLETE", "", 1
    mbr = t["micro_batch_rows"]
    rps = t["rows_per_step"]
    accum = rps // mbr
    use_amp = device.type == "cuda"
    split, steps, _ = main_stream(base_cfg["budget_tokens"])
    plan_sha_full = hashlib.sha256(json.dumps(steps).encode()).hexdigest()
    if not cfg.get("test") and plan_sha_full != P.PLAN_SHA:
        raise guard.GuardError("main-stream plan hash mismatch")
    if a.test_rows:
        steps = [s[:a.test_rows] for s in steps]
        mbr = max(1, a.test_rows // accum)
    n_steps = len(steps)
    start, end = int(cfg["start_step"]), int(cfg["end_step"])
    sw, rw = int(cfg.get("spec_w", P.SPEC_W)), int(cfg.get("router_w", P.ROUTER_W))
    rd.log(f"mode {mode}: stream {n_steps} steps, {start} -> {end}, windows spec {sw} / router {rw}, plan {plan_sha_full[:12]}")
    # ---------------- base (reused milestone, hash-verified)
    ckp = guard.ROOT / cfg["base_ckpt"] / "state.pt"
    if not cfg.get("test") or cfg.get("verify_ckpt", True):
        if guard.sha256_file(ckp) != cfg["base_ckpt_sha256"]:
            raise guard.GuardError("reused checkpoint hash mismatch")
    ck0 = torch.load(ckp, map_location="cpu", weights_only=False)
    if int(ck0["step"]) != P.START:
        raise guard.GuardError(f"milestone step {ck0['step']} != {P.START}")
    model = LFRModel(LFRConfig.from_dict({**ck0["model_cfg"], "teacher": False}), base_cfg["seed"])
    model.load_state_dict(ck0["model"], strict=True)
    model = model.to(device)
    model_cfg_dict = ck0["model_cfg"]
    del ck0
    if frozen_mode:
        model.eval()
        for p_ in model.parameters():
            p_.requires_grad_(False)
    # ---------------- K=5 region priors (main stream)
    tab, tab_sha = P.tables("main")
    off = np.full(split.n_docs, -1, np.int64)
    off[tab["docs"]] = tab["ptr"][:-1]
    off_t = torch.as_tensor(off, device=device)
    PRI = {"sem": torch.as_tensor(tab["p"].astype(np.float32), device=device),
           "shuf": torch.as_tensor(tab["p_sh"].astype(np.float32), device=device)}
    del tab
    # ---------------- arms
    arms = arm_list(cfg)
    banks, routers, rmeta = {}, {}, {}
    prior_of = {"SEM5": "sem", "SHUF5": "shuf", "SEMh": "sem", "SHUFh": "shuf", "SEMs": "sem", "SHUFs": "shuf",
                "FROZ": "sem"}
    init = None
    if not frozen_mode and cfg.get("init_ckpt"):
        ip = guard.ROOT / cfg["init_ckpt"]
        if not cfg.get("test") and not CK.verify_dir(ip):
            raise guard.GuardError(f"init milestone {ip} fails verification")
        if not cfg.get("test") and json.loads((ip / "meta.json").read_text())["sha256"] != cfg["init_ckpt_sha256"]:
            raise guard.GuardError("init milestone sha256 mismatch")
        init = torch.load(ip / "state.pt", map_location="cpu", weights_only=False)
    src_of = {"G": "G5", "ALTG": "G5", "SEMh": "SEM5", "SEMs": "SEM5", "FROZ": "SEM5", "SHUFh": "SHUF5", "SHUFs": "SHUF5"}
    for name, kind, s in arms:
        routed = kind in ("G5", "G", "ALTG")
        b = Bank5(K, P.D_MODEL, P.FFN, seed=int(s), routed=routed, alpha=P.ALPHA_G)
        if init is not None:
            b.load_state_dict(init["banks"][f"{src_of[kind]}_{s}"])
        banks[name] = b.to(device)
        if kind == "FROZ":
            for p_ in b.parameters():
                p_.requires_grad_(False)
        if kind in ("SEM5", "SHUF5", "SEMh", "SHUFh", "SEMs", "SHUFs", "FROZ"):
            r = CausalRouter(P.ROUTER_H, int(s), "base", K)
            src_r = ("SEM5" if prior_of[kind] == "sem" else "SHUF5") + f"_{s}"
            if init is not None:
                r.load_state_dict(init["routers"][src_r])
            routers[name] = r.to(device)
            rmeta[name] = {"bank": name, "prior": prior_of[kind], "seed": int(s), "init": src_r}
    rd.log(f"arms {[x[0] for x in arms]} routers {list(routers)} p5 {tab_sha[:12]}")
    st = {"next_step": start, "seen": 0, "n_upd": {}, "acc": {"step_s": 0.0}, "ckpts": {}, "legs": 0, "win": {}}
    opt = None if frozen_mode else build_optimizer(model, t, device)
    mk = lambda ps: torch.optim.AdamW(ps, lr=t["lr"], betas=tuple(t["betas"]), eps=1e-8, weight_decay=t["weight_decay"],
                                      fused=device.type == "cuda")
    eopt, gropt, ropt = {}, {}, {}
    for name, kind, s in arms:
        if kind == "FROZ":
            continue
        eopt[name] = mk(banks[name].expert_params())
        if banks[name].routed:
            gropt[name] = mk([banks[name].router.weight])
    for n, r in routers.items():
        ropt[n] = torch.optim.AdamW(r.parameters(), lr=RO["lr"], weight_decay=RO["wd"], betas=tuple(RO["betas"]))
    if a.run_dir:
        ent = store.latest_verified(log=rd.log)
        if ent is not None:
            ck = torch.load(guard.ROOT / ent["path"] / "state.pt", map_location="cpu", weights_only=False)
            if ck["run_id"] != rd.run_id or ck.get("config_sha256") != cfg_sha:
                raise guard.GuardError("resume checkpoint belongs to another run/config")
            if not frozen_mode:
                model.load_state_dict(ck["model"])
                opt.load_state_dict(ck["opt"])
            st = ck["st"]
            for n, b in banks.items():
                b.load_state_dict(ck["banks"][n])
            for n, r in routers.items():
                r.load_state_dict(ck["routers"][n])
            for d_, src in ((eopt, "eopt"), (gropt, "gropt"), (ropt, "ropt")):
                for n in d_:
                    d_[n].load_state_dict(ck[src][n])
            CK.set_rng_state(ck["rng"])
            rd.log(f"resumed at step {st['next_step']} from {ent['path']} (sha256 {ent['sha256'][:12]} verified)")
            del ck
        else:
            rd.log("resume requested but no verified checkpoint exists: starting at the beginning")
    st["legs"] += 1
    compile_on = device.type == "cuda" and not a.no_compile
    fwd = torch.compile(model) if compile_on else model

    def lm_sum(h, w, y):
        lg = torch.nn.functional.linear(h, w).float()
        lse = torch.logsumexp(lg, -1)
        valid = y >= 0
        return ((lse - lg.gather(1, y.clamp(min=0).view(-1, 1)).squeeze(1)) * valid).sum()
    lm_fn = torch.compile(lm_sum) if compile_on else lm_sum

    def set_lr(o, lr):
        for g in o.param_groups:
            g["lr"] = lr * g.get("lr_mult", 1.0)

    def bump(key):
        st["n_upd"][key] = st["n_upd"].get(key, 0) + 1

    def warm(key, n=None):
        n = st["n_upd"].get(key, 0) if n is None else n
        return min(1.0, (n + 1) / P.REWARM)

    def payload():
        return {"run_id": rd.run_id, "run_name": cfg["run_name"], "config_sha256": cfg_sha,
                "model": {} if frozen_mode else model.state_dict(), "opt": None if frozen_mode else opt.state_dict(),
                "st": st, "rng": CK.rng_state(),
                "cursor": {"next_step": st["next_step"], "n_steps": n_steps, "rows_per_step": rps, "plan_sha256": plan_sha_full},
                "banks": {n: b.state_dict() for n, b in banks.items()}, "routers": {n: r.state_dict() for n, r in routers.items()},
                "eopt": {n: o.state_dict() for n, o in eopt.items()}, "gropt": {n: o.state_dict() for n, o in gropt.items()},
                "ropt": {n: o.state_dict() for n, o in ropt.items()}, "time": guard.now()}

    def save_slot(note=""):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0_ = time.time()
        pl = payload()
        if frozen_mode:
            pl["model"] = {"_frozen": torch.zeros(1)}
        ent_ = store.save(pl, st["next_step"], st["seen"], note=note)
        st["acc"]["ckpt_s"] = st["acc"].get("ckpt_s", 0.0) + (time.time() - t0_)
        return ent_

    def save_milestone(tag):
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
        if frozen_mode:
            sd = {}
        else:
            keep32 = lambda k, v: v.dim() <= 1 or "router" in k or "norm" in k
            sd = {k: (v if keep32(k, v) or not v.is_floating_point() else v.to(torch.bfloat16))
                  for k, v in model.state_dict().items()}
        obj = {"model": sd, "base_from": None if not frozen_mode else {"path": cfg["base_ckpt"], "sha256": cfg["base_ckpt_sha256"]},
               "banks": {n: {k: v.detach().cpu() for k, v in b.state_dict().items()} for n, b in banks.items()},
               "routers": {n: {k: v.detach().cpu() for k, v in r.state_dict().items()} for n, r in routers.items()},
               "router_meta": rmeta, "arms": arms, "config": cfg, "step": step_, "seen": seen_, "n_upd": dict(st["n_upd"]),
               "model_cfg": model_cfg_dict, "base_seed": base_cfg["seed"], "K": K, "mode": mode,
               "dtype": "base: bf16 matmul weights, fp32 rest (cont); banks and routers fp32"}
        chk = (lambda ld: ld["step"] == step_ and len(ld["banks"]) == len(banks))
        sha, nb, sec = CK.atomic_ckpt_dir(obj, p, chk, {"step": step_, "seen": seen_, "tag": tag})
        st["ckpts"][tag] = {"path": str(p.relative_to(guard.ROOT)), "sha256": sha, "step": step_, "seen": seen_}
        CK.ledger({"time": guard.now(), "run_id": rd.run_id, "run_name": cfg["run_name"], "kind": "milestone",
                   "path": st["ckpts"][tag]["path"], "step": step_, "seen": seen_, "sha256": sha, "bytes": nb,
                   "save_seconds": round(sec, 2), "note": tag})

    def param_hash(tensors):
        h = hashlib.sha256()
        for v in tensors:
            h.update(v.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        return h.hexdigest()[:16]

    def base_and_feats(step):
        """Base forward (+ ordinary LM backward in cont mode, no optimizer step yet). Returns base parts and the detached
        scored-token features in training flattening order."""
        rows = steps[step]
        n_valid = int(sum(split.lengths[d] - 1 for r in rows for d in r))
        parts = {}
        keep = {"h": [], "y": [], "gd": [], "row": [], "dl": [], "pos": []}
        for mi_ in range(accum):
            rr = rows[mi_ * mbr:(mi_ + 1) * mbr]
            if not rr:
                continue
            bt = to_torch(make_batch(rr, split), device)
            minp = {"input_ids": bt["input_ids"], "doc_ids": bt["doc_ids"], "positions": bt["positions"]}
            ctl = {"teach": False, "edl": False, "lc": False, "cf_store": None, "tap": False}
            if frozen_mode:
                with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                    out = fwd(minp, ctl)
                h = out["hidden"]
                hflat, y = h.reshape(-1, h.shape[-1]), bt["targets"].reshape(-1)
            else:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                    out = fwd(minp, ctl)
                    h = out["hidden"]
                    hflat, y = h.reshape(-1, h.shape[-1]), bt["targets"].reshape(-1)
                    lm = torch.zeros((), device=device)
                    for i in range(0, y.numel(), 8192):
                        lm = lm + lm_fn(hflat[i:i + 8192], model.tok_emb.weight, y[i:i + 8192])
                    loss = lm / n_valid + (t["aux_coeff"] * out["aux"] + t["z_coeff"] * out["z"]) / accum
                loss.backward()
                for k in ("aux", "z"):
                    parts[k] = parts.get(k, 0.0) + float(out[k].detach()) / accum
                parts["lm"] = parts.get("lm", 0.0) + float(lm.detach())
            v_ = bt["doc_ids"].reshape(-1) >= 0
            keep["h"].append(hflat.detach()[v_])
            keep["y"].append(y[v_])
            keep["gd"].append(bt["gdoc"].reshape(-1)[v_])
            rowt = (torch.arange(bt["doc_ids"].shape[0], device=device) + mi_ * mbr).view(-1, 1).expand_as(bt["doc_ids"])
            keep["row"].append(rowt.reshape(-1)[v_])
            keep["dl"].append(bt["doc_ids"].reshape(-1)[v_])
            keep["pos"].append(bt["positions"].reshape(-1)[v_])
        Y = torch.cat(keep["y"])
        sc = Y >= 0
        Hs = torch.cat(keep["h"])[sc]
        pos = torch.cat(keep["pos"])[sc].long()
        dkey = torch.cat(keep["row"])[sc].long() * 4096 + torch.cat(keep["dl"])[sc].long()
        GDs = torch.cat(keep["gd"])[sc]
        o_ = off_t[GDs]
        if bool((o_ < 0).any()):
            raise guard.GuardError("doc without p5 regions in the step")
        rid = o_ + torch.div(pos, 256, rounding_mode="floor")
        return parts, n_valid, {"h": Hs, "y": Y[sc], "xn": rms0(Hs), "pos": pos, "dkey": dkey, "rid": rid}

    def soft_gen(seed, step, name):
        g = torch.Generator()
        g.manual_seed(int.from_bytes(hashlib.sha256(f"asrs2-soft|{seed}|{step}|{name}".encode()).digest()[:8], "little"))
        return g

    def aux_step(step, f, phase):
        """All residual-arm gradient computations of a step (no optimizer steps). phase: 'all' (frozen mode),
        'spec' or 'router'. Diagnostics computed before any update."""
        W = model.tok_emb.weight.detach()
        N = f["h"].shape[0]
        rec = {}
        lr_t = lr_at(step, n_steps, t)
        sub = ((f["pos"] + step) % 4 == 0)
        tok = torch.nonzero(sub).squeeze(1)
        X = f["xn"].to(torch.float16)
        idx, mask, lens = doc_batch(f["dkey"], f["pos"])
        same = f["dkey"][1:] == f["dkey"][:-1]
        e_prior = {k: v[f["rid"]].argmax(1) for k, v in PRI.items()}
        for name, kind, s in arms:
            b = banks[name]
            r_ = {}
            if kind in ("G5", "G", "ALTG"):
                train_e = kind != "ALTG" or phase in ("spec", "all")
                train_r = kind != "ALTG" or phase in ("router", "all")
                ce, e, info = gen_grad(b, f["h"], f["xn"], W, f["y"])
                if not train_e:
                    for p_ in b.expert_params():
                        p_.grad = None
                if not train_r:
                    b.router.weight.grad = None
                r_ = {"res": float(ce.mean()), **info, "train_e": train_e, "train_r": train_r}
                if train_e:
                    key = f"{name}.e"
                    mult = float(cfg.get("spec_lr_mult", 1.0)) if kind == "ALTG" else 1.0
                    set_lr(eopt[name], lr_t * (warm(key) if not frozen_mode else 1.0) * mult)
                    r_["gn_e"] = float(torch.nn.utils.clip_grad_norm_(b.expert_params(), t["clip"]))
                    r_["do_e"] = True
                if train_r:
                    key = f"{name}.r"
                    set_lr(gropt[name], lr_t * (warm(key) if not frozen_mode else 1.0))
                    r_["gn_r"] = float(torch.nn.utils.clip_grad_norm_([b.router.weight], t["clip"]))
                    r_["do_r"] = True
            else:
                pri = rmeta[name]["prior"]
                r = routers[name]
                # held-out diagnostics (subset, before updates)
                with torch.no_grad():
                    lg_all, e_rt = route(r, X, idx, mask, lens, grad=False)
                    if tok.numel() > 0:
                        u, _ = deltas(b.experts.w1, b.experts.w2, f["h"][tok], f["xn"][tok], W, f["y"][tok])
                        ep = e_prior[pri][tok]
                        ar = torch.arange(tok.numel(), device=device)
                        r_.update({"u_prior": float(u[ar, ep].mean()), "u_router": float(u[ar, e_rt[tok]].mean()),
                                   "u_mean": float(u.mean()), "u_oracle": float(u.max(1).values.mean())})
                    r_.update({"agree_prior": float((e_rt == e_prior[pri]).float().mean()),
                               "switch": float((e_rt[1:] != e_rt[:-1])[same].float().mean())})
                    ld = torch.bincount(e_rt, minlength=K).float() / N
                    r_.update({"load_max": float(ld.max()), "n_dead": int((ld < 1.0 / (10 * K)).sum())})
                do_spec = kind != "FROZ" and phase in ("spec", "all")
                do_rout = phase in ("router", "all")
                if do_spec:
                    if kind in ("SEMs", "SHUFs"):
                        e = assign(PRI[pri][f["rid"]], "soft", soft_gen(s, step, name))
                    else:
                        e = e_prior[pri]
                    ce = spec_grad(b, f["h"], f["xn"], W, f["y"], e)
                    key = f"{name}.e"
                    mult = 1.0 if frozen_mode else float(cfg.get("spec_lr_mult", 1.0))
                    set_lr(eopt[name], lr_t * (warm(key) if not frozen_mode else 1.0) * mult)
                    r_["gn_e"] = float(torch.nn.utils.clip_grad_norm_(b.expert_params(), t["clip"]))
                    r_.update({"res": float(ce.mean()), "do_e": True,
                               "assign_ent": float(-(lambda q: (q * torch.log(q.clamp(min=1e-12))).sum())(
                                   torch.bincount(e, minlength=K).float() / N)),
                               "assign_load_max": float((torch.bincount(e, minlength=K).float() / N).max())})
                if do_rout:
                    pi = r1_target(PRI[pri][f["rid"]], P.PSEM_EPS)
                    r.train()
                    lg, _ = route(r, X, idx, mask, lens, grad=True)
                    loss, rr_ = r1_loss(lg, pi, float(RO["lam"]))
                    ropt[name].zero_grad(set_to_none=True)
                    loss.backward()
                    lr_r = float(RO["lr"]) * (warm(f"{name}.rr") if frozen_mode else 1.0)
                    set_lr(ropt[name], lr_r)
                    r_.update({"r_loss": float(loss.detach()), **rr_,
                               "r_gn": float(torch.nn.utils.clip_grad_norm_(r.parameters(), float(RO["clip"]))),
                               "do_rr": True})
                    del lg, loss
            rec[name] = r_
        return rec

    def apply_updates(rec):
        for name, kind, s in arms:
            r_ = rec[name]
            if r_.get("do_e"):
                eopt[name].step()
                bump(f"{name}.e")
            if r_.get("do_r"):
                gropt[name].step()
                bump(f"{name}.r")
            if r_.get("do_rr"):
                ropt[name].step()
                bump(f"{name}.rr")
        for d_ in (eopt, gropt, ropt):
            for o in d_.values():
                o.zero_grad(set_to_none=True)
        for b in banks.values():
            for p_ in b.parameters():
                p_.grad = None

    ms_steps = {int(v): k for k, v in cfg.get("milestones", {}).items()}
    last_slot, last_slot_step = time.time(), st["next_step"]
    every_steps, every_s = int(cfg.get("slot_every_steps", P.SLOT_EVERY_STEPS)), float(cfg.get("slot_every_s", P.SLOT_EVERY_S))
    try:
        rd.log(f"params: base {model.param_counts()} banks { {n: b.n_params() for n, b in banks.items()} } "
               f"routers { {n: r.n_params() for n, r in routers.items()} }")
        if not frozen_mode:
            model.train()
        while st["next_step"] < end:
            step = st["next_step"]
            if a.test_signal_at_step is not None and step == a.test_signal_at_step and st["legs"] == 1:
                os.kill(os.getpid(), signal.SIGINT)
            if stopf.poll() is not None:
                status, reason, rc = "EXTERNALLY_STOPPED", stopf.reason, 77
                break
            if a.max_steps is not None and step >= a.max_steps:
                status, reason, rc = "INTERRUPTED", "max-steps", 75
                break
            if time.time() - t_proc0 > a.max_seconds:
                status, reason, rc = "INTERRUPTED", "leg time limit", 75
                break
            limits.check()
            if frozen_mode:
                phase, win, offw = "all", 0, step
            else:
                phase, win, offw = P.window_phase(step, start, sw, rw)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            parts, n_valid, f = base_and_feats(step)
            rec = aux_step(step, f, phase)
            gn = None
            if not frozen_mode:
                lr_b = lr_at(step, n_steps, t) * min(1.0, (step - start + 1) / P.REWARM)
                set_lr(opt, lr_b)
                gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), t["clip"]))
                opt.step()
                opt.zero_grad(set_to_none=True)
                bump("base")
            apply_updates(rec)
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - t0
            for name, r_ in rec.items():
                for k_, v_ in r_.items():
                    if isinstance(v_, float) and not math.isfinite(v_) and k_ in ("res", "r_loss", "gn_e", "gn_r", "r_gn"):
                        raise FloatingPointError(f"non-finite {k_} for {name} at step {step}")
            lm_mean = parts["lm"] / n_valid if "lm" in parts else None
            if lm_mean is not None and not math.isfinite(lm_mean):
                raise FloatingPointError(f"non-finite base loss at step {step}")
            st["next_step"] = step + 1
            st["seen"] += n_valid
            st["acc"]["step_s"] += dt
            rows_sha = hashlib.sha256(json.dumps(steps[step]).encode()).hexdigest()[:16]
            m = {"step": step, "phase": phase, "window": win, "seen": st["seen"], "dt": dt, "rows_sha": rows_sha,
                 "lm": lm_mean, "gn": gn, "aux": parts.get("aux"), "z": parts.get("z"), "arms": rec}
            rd.metric(m)
            ns = st["next_step"]
            wkey = str(win)
            W_ = st["win"].setdefault(wkey, {"n": 0, "lm": 0.0})
            W_["n"] += 1
            W_["lm"] += lm_mean or 0.0
            if step % 50 == 0 or ns == end:
                rd.log(json.dumps({"step": step, "phase": phase, "lm": None if lm_mean is None else round(lm_mean, 4),
                                   "dt": round(dt, 2), "res": {n: round(v.get("res", float("nan")), 4) for n, v in rec.items()}}))
            boundary = (not frozen_mode) and (P.window_phase(ns, start, sw, rw)[0] != phase or ns == end)
            if boundary and not frozen_mode:
                hs = {"base": param_hash(model.parameters())}
                for n_, b_ in banks.items():
                    hs[n_ + ".experts"] = param_hash(b_.expert_params())
                    if b_.routed:
                        hs[n_ + ".router"] = param_hash([b_.router.weight])
                for n_, r_m in routers.items():
                    hs["router." + n_] = param_hash(r_m.parameters())
                rd.metric({"kind": "param_hash", "after": phase, "window": win, "step": ns, "hash": hs})
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
    summary = {"status": status, "reason": reason, "mode": mode, "steps_done": st["next_step"], "start": start, "end": end,
               "seen": st["seen"], "n_upd": st["n_upd"], "leg_seconds": time.time() - t_proc0,
               "train_step_seconds": st["acc"]["step_s"], "peak_alloc_gib": peak[0], "peak_reserved_gib": peak[1],
               "ckpts": st["ckpts"], "legs": st["legs"], "ckpt_seconds": st["acc"].get("ckpt_s", 0.0),
               "plan_sha256": plan_sha_full, "p5_sha256": tab_sha}
    stopf.close()
    if monitor is not None:
        monitor.stop()
    if a.device == "cuda":
        guard.gpu_ledger(t_start_iso, guard.now(), rd.run_id, f"asrs2 train {cfg['run_name']}", status)
    with open(rd.path / f"leg_{st['legs']:02d}.json", "w") as f_:
        json.dump(summary, f_, indent=1, default=str)
    rd.set_status(status, **{k: v for k, v in summary.items() if k != "status"})
    if status == "COMPLETE":
        guard.atomic_write_json(rd.path / "summary.json", summary, overwrite=True)
        store.prune_after_complete()
    rd.log(f"{status} {reason[:500]}")
    return rc


if __name__ == "__main__":
    main()
