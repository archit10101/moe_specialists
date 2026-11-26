#!/usr/bin/env python
# csr-alternating-semantic-residual-specialization-v1 (asrs1) alternating trainer
# (docs/CSR_ALTERNATING_SEMANTIC_RESIDUAL_SPECIALIZATION_V1_PREREG.md, section 4). Infrastructure adapted from
# ccsr1_s3_train.py (base step, stop handling, checkpoints) - that file is imported, never modified.
"""One process, one base, three residual banks, alternating complete cycles over steps 318 .. 1110:

  BASE      base trains (LM + Switch aux + z); banks and routers not run
  REFRESH   everything frozen; exact 16-specialist utilities of SEM and SHUF on the cycle's specialist+router rows ->
            R3 teacher pi = softmax(u/(c s_u) + eta log(p + 1e-4)), e* = argmax pi, w = clip(margin/med, .25, 4);
            cache written atomically, read-only, sha256 before use
  SPEC      base + routers frozen; each token trains only its cached teacher specialist (LM CE on sg(h), sg(W));
            ALT-G trains its experts on its own (frozen) token-router choices
  ROUTER    base + specialists frozen; causal routers: w-KL(pi || q) + 0.01 balance; ALT-G: router only
            (residual_step LM-through-gate + Switch balance)
Arms: ALTG (cfar1 G), SEM (cfar1 L) + routers SEM_s{0,1,2}, SHUF (cfar1 F2h_T500_w240) + routers SHUF_s{0,1,2}.
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

TB = ("SEM", "SHUF")          # teacher-assigned banks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--run-dir")
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--max-seconds", type=float, default=4800)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--test-signal-at-step", type=int)
    ap.add_argument("--test-signal-in-refresh", type=int, help="tests: SIGINT during the refresh of this cycle")
    ap.add_argument("--test-rows", type=int, default=0, help="CPU tests: use only the first N rows of each step")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    cfg = json.loads(guard.inside_root(a.config).read_text())
    if not cfg.get("test"):
        from csr.asrs1_paths import check_prereg
        if cfg.get("prereg_sha256") != check_prereg():
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
        from csr.asrs1_paths import STOP_MARKER
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


class Stopped(Exception):
    pass


def run(cfg, a):
    import numpy as np
    import torch
    from csr import asrs1_ckpt as CK
    from csr import asrs1_paths as P
    from csr.cfar1_bank import FedBank
    from csr.cfar1_paths import ALPHA_G
    from csr.ccsr1_router import CausalRouter, router_loss, teacher
    from csr.csur1_bank import all_deltas, finite
    from csr.jsr_bank import Bank, head_ce_grad, residual_step, rms0
    from csr.fsp_lib import make_batch, to_torch
    from csr.lfr_lib import lr_at, main_stream
    from csr.lfr_model import LFRConfig, LFRModel
    from cfar1_train import build_optimizer, load_psem
    from ccsr1_s3_train import InPlaceRun, doc_batch, route
    torch.set_num_threads(4)
    monitor = None
    if a.device == "cuda":
        guard.occupancy_gate(f"asrs1 train:{cfg['run_name']}")
        device = guard.setup_cuda_device()
        monitor = guard.OccupancyMonitor(interval=15.0)
        monitor.start()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        device = torch.device("cpu")
    guard.register_self(f"asrs1 train {cfg['run_name']} on {a.device}")
    stopf = StopFlag(monitor)
    base_cfg = cfg["base_config"]
    t = base_cfg["train"]
    TC, RO = cfg["teacher"], cfg["router_opt"]
    if a.run_dir:
        rd = InPlaceRun(guard.inside_root(a.run_dir))
        if json.loads((rd.path / "config.json").read_text()) != cfg:
            raise guard.GuardError("resume: config differs")
    else:
        rd = InPlaceRun(guard.RunDir(P.RUN_PREFIX + cfg["run_name"], cfg).path)
    rd.set_status("RUNNING")
    store = CK.ResumeStore(rd.run_id, cfg["run_name"])
    cache_dir = P.CACHES / rd.run_id
    cfg_sha = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    t_start_iso, t_proc0 = guard.now(), time.time()
    limits = guard.RunLimits(max_seconds=a.max_seconds + 900, monitor=monitor)
    status, reason, rc = "INCOMPLETE", "", 1
    mbr = t["micro_batch_rows"]
    rps = t["rows_per_step"]
    accum = rps // mbr
    use_amp = device.type == "cuda"
    split, steps, planned_tokens = main_stream(base_cfg["budget_tokens"])
    plan_sha_full = hashlib.sha256(json.dumps(steps).encode()).hexdigest()
    if not cfg.get("test") and plan_sha_full != P.PLAN_SHA:
        raise guard.GuardError("main-stream plan hash mismatch")
    if a.test_rows:
        steps = [s[:a.test_rows] for s in steps]
        mbr = max(1, a.test_rows // accum)
    n_steps = len(steps)
    S = cfg["schedule"]
    sched = P.schedule(S["start"], S["end"], S["n_cycles"], S["base"], S["spec"])
    start, end = int(S["start"]), int(S["end"])
    phase_of = lambda s: P.phase_of(s, sched)
    rd.log(f"stream: {n_steps} steps, alternating {start} -> {end} in {len(sched)} cycles "
           f"{[(c, [(n, s1 - s0) for n, s0, s1 in ph]) for c, ph in sched]}, plan {plan_sha_full[:12]}")
    # ---------------- reused start checkpoint
    ckp = guard.ROOT / cfg["base_ckpt"] / "state.pt"
    if not cfg.get("test") or cfg.get("verify_ckpt", True):
        if guard.sha256_file(ckp) != cfg["base_ckpt_sha256"]:
            raise guard.GuardError("reused checkpoint hash mismatch")
    ck0 = torch.load(ckp, map_location="cpu", weights_only=False)
    if int(ck0["step"]) != start:
        raise guard.GuardError(f"milestone step {ck0['step']} != start {start}")
    model = LFRModel(LFRConfig.from_dict({**ck0["model_cfg"], "teacher": False}), base_cfg["seed"])
    model.load_state_dict(ck0["model"], strict=True)
    model = model.to(device)
    banks, routers, rmeta = {}, {}, {}
    for arm in cfg["arms"]:
        n = arm["name"]
        if arm["kind"] == "token":
            b = Bank({"name": arm["init"], "kind": "routed", "tau": 1.0, "r": 0.0, "alpha": ALPHA_G, "groups": None},
                     base_cfg["seed"])
        else:
            b = FedBank(ck0["bank_specs"][arm["init"]], base_cfg["seed"])
            for p in b.router_params():
                p.requires_grad_(False)
        b.load_state_dict(ck0["banks"][arm["init"]])
        banks[n] = b.to(device)
        for rr in arm.get("routers", []):
            p_ = guard.ROOT / rr["ckpt"]
            if (not cfg.get("test")) and guard.sha256_file(p_) != rr["sha256"]:
                raise guard.GuardError(f"router checkpoint hash mismatch {rr['ckpt']}")
            rs = torch.load(p_, map_location="cpu", weights_only=False)
            rc_ = rs["cfg"]
            r = CausalRouter(int(rc_["H"]), int(rs["seed"]), rc_.get("feats", "base"), int(rc_.get("n_out", 16)))
            r.load_state_dict(rs["state"])
            routers[rr["name"]] = r.to(device)
            rmeta[rr["name"]] = {"bank": n, "cfg": rc_, "seed": int(rs["seed"]), "init_ckpt": rr["ckpt"],
                                 "stage1_stats": rs["stats"]}
    del ck0
    if not cfg.get("test"):
        P.verify_reused(rd.log)
    psem_off, psem_P, psem_Psh, psem_sha = load_psem(split, device)
    PRI = {"SEM": psem_P, "SHUF": psem_Psh}
    rd.log(f"arms {[(x['name'], x['init'], x['kind']) for x in cfg['arms']]} routers {list(routers)} psem {psem_sha[:12]}")
    G = banks["ALTG"]
    st = {"next_step": start, "seen": 0, "n_done": {"base": 0, "spec": 0, "router": 0}, "refreshed": {},
          "phase_tok": {}, "acc": {"step_s": 0.0, "refresh_s": 0.0}, "ckpts": {}, "legs": 0, "phase_rec": {}}
    opt = build_optimizer(model, t, device)
    mk = lambda ps: torch.optim.AdamW(ps, lr=t["lr"], betas=tuple(t["betas"]), eps=1e-8,
                                      weight_decay=t["weight_decay"], fused=device.type == "cuda")
    bopt = {n: mk(list(b.expert_params()) if isinstance(b, FedBank) else [b.experts.w1, b.experts.w2])
            for n, b in banks.items()}
    gropt = mk([G.router.weight])
    ropt = {n: torch.optim.AdamW(r.parameters(), lr=RO["lr"], weight_decay=RO["wd"], betas=tuple(RO["betas"]))
            for n, r in routers.items()}
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
            for n, r in routers.items():
                r.load_state_dict(ck["routers"][n])
            opt.load_state_dict(ck["opt"])
            for n in banks:
                bopt[n].load_state_dict(ck["bank_opt"][n])
            gropt.load_state_dict(ck["gropt"])
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

    def set_lr(o, lr):
        for g in o.param_groups:
            g["lr"] = lr * g.get("lr_mult", 1.0)

    def n_valid_of(step):
        return int(sum(split.lengths[d] - 1 for r in steps[step] for d in r))

    def frozen_scored(step):
        """No-grad base forward of a step: scored tokens in the training flattening order (row, doc, position)."""
        keep = {"h": [], "y": [], "gd": [], "row": [], "dl": [], "pos": []}
        rows = steps[step]
        with torch.no_grad():
            for mi_ in range(accum):
                rr = rows[mi_ * mbr:(mi_ + 1) * mbr]
                if not rr:
                    continue
                bt = to_torch(make_batch(rr, split), device)
                minp = {"input_ids": bt["input_ids"], "doc_ids": bt["doc_ids"], "positions": bt["positions"]}
                ctl = {"teach": False, "edl": False, "lc": False, "cf_store": None, "tap": False}
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                    out = fwd(minp, ctl)
                h = out["hidden"]
                v_ = bt["doc_ids"].reshape(-1) >= 0
                keep["h"].append(h.reshape(-1, h.shape[-1])[v_])
                keep["y"].append(bt["targets"].reshape(-1)[v_])
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
        o_ = psem_off[GDs]
        if bool((o_ < 0).any()):
            raise guard.GuardError("doc without p_sem regions in the step")
        rid = o_ + torch.div(pos, 256, rounding_mode="floor")
        return {"h": Hs, "y": Y[sc], "xn": rms0(Hs), "pos": pos, "dkey": dkey, "rid": rid}

    def param_hashes():
        """sha256 (first 16 hex) of every parameter group: phase-isolation evidence (logged at every boundary)."""
        def hh(tensors):
            h = hashlib.sha256()
            for v in tensors:
                h.update(v.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
            return h.hexdigest()[:16]
        out = {"base": hh(model.state_dict().values())}
        for n, b in banks.items():
            out[f"{n}.experts"] = hh([b.experts.w1, b.experts.w2])
            if n == "ALTG":
                out["ALTG.router"] = hh([b.router.weight])
            else:
                out[f"{n}.frozen_buffers_student"] = hh([*b.buffers(), *b.router_params()])
        for n, r in routers.items():
            out[f"router.{n}"] = hh(r.parameters())
        return out

    # ---------------- checkpoints
    def payload():
        return {"run_id": rd.run_id, "run_name": cfg["run_name"], "config_sha256": cfg_sha, "model": model.state_dict(),
                "opt": opt.state_dict(), "st": st, "rng": CK.rng_state(),
                "cursor": {"next_step": st["next_step"], "n_steps": n_steps, "rows_per_step": rps, "plan_sha256": plan_sha_full},
                "banks": {n: b.state_dict() for n, b in banks.items()},
                "bank_opt": {n: o.state_dict() for n, o in bopt.items()}, "gropt": gropt.state_dict(),
                "routers": {n: r.state_dict() for n, r in routers.items()},
                "router_opt": {n: o.state_dict() for n, o in ropt.items()}, "time": guard.now()}

    def save_slot(note="", tag=""):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0_ = time.time()
        ent_ = store.save(payload(), st["next_step"], st["seen"], note=note, tag=tag)
        st["acc"]["ckpt_s"] = st["acc"].get("ckpt_s", 0.0) + (time.time() - t0_)
        return ent_

    def save_milestone(tag, cyc):
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
        deploy = [{"name": "ALTG", "bank": "ALTG", "router": "token"}]
        for rn, m_ in rmeta.items():
            deploy.append({"name": rn, "bank": m_["bank"], "router": rn, "prior": m_["bank"].lower(), "seed": m_["seed"]})
        obj = {"model": sd, "banks": {n: {k: v.detach().cpu() for k, v in b.state_dict().items()} for n, b in banks.items()},
               "bank_specs": {n: b.spec for n, b in banks.items()}, "arms": cfg["arms"], "deploy": deploy,
               "routers": {n: {k: v.detach().cpu() for k, v in r.state_dict().items()} for n, r in routers.items()},
               "router_meta": rmeta, "teacher_stats": {c_: v["stats"] for c_, v in st["refreshed"].items()},
               "n_done": dict(st["n_done"]), "cycle": cyc, "config": cfg, "step": step_, "seen": seen_,
               "model_cfg": model.cfg.to_dict(), "base_seed": base_cfg["seed"],
               "dtype": "base: bf16 matmul weights, fp32 rest; banks and routers fp32"}
        sha, nb, sec = CK.atomic_ckpt_dir(obj, p, lambda ld: ld["step"] == step_ and len(ld["model"]) == len(sd),
                                          {"step": step_, "seen": seen_, "tag": tag})
        st["ckpts"][tag] = {"path": str(p.relative_to(guard.ROOT)), "sha256": sha, "step": step_, "seen": seen_}
        CK.ledger({"time": guard.now(), "run_id": rd.run_id, "run_name": cfg["run_name"], "kind": "milestone",
                   "path": st["ckpts"][tag]["path"], "step": step_, "seen": seen_, "sha256": sha, "bytes": nb,
                   "save_seconds": round(sec, 2), "note": tag})

    # ---------------- teacher cache
    def refresh(cyc):
        ph = {n: (s0, s1) for n, s0, s1 in sched[cyc - 1][1]}
        sp_steps = list(range(*ph["spec"]))
        ro_steps = list(range(*ph["router"]))
        allst = sp_steps + ro_steps
        t0_ = time.time()
        U = {b: [] for b in TB}
        RID, Y, off = [], [], [0]
        Wd = model.tok_emb.weight.detach()
        for j, step in enumerate(allst):
            if a.test_signal_in_refresh is not None and cyc == a.test_signal_in_refresh and j == 1 and st["legs"] == 1:
                os.kill(os.getpid(), signal.SIGINT)
            if stopf.poll() is not None:
                raise Stopped(stopf.reason)
            limits.check()
            f = frozen_scored(step)
            with torch.no_grad():
                for b in TB:
                    u, _ = all_deltas(banks[b].experts.w1, banks[b].experts.w2, f["h"], f["xn"], Wd, f["y"])
                    U[b].append(u)
            RID.append(f["rid"])
            Y.append(f["y"])
            off.append(off[-1] + int(f["y"].numel()))
        rid, y = torch.cat(RID), torch.cat(Y)
        n_sp = off[len(sp_steps)]
        arrs = {"steps": np.asarray(allst, np.int64), "off": np.asarray(off, np.int64), "n_spec_tokens": np.int64(n_sp),
                "y": y.cpu().numpy().astype(np.int32)}
        stats = {"cycle": cyc, "spec_steps": [ph["spec"][0], ph["spec"][1]], "router_steps": [ph["router"][0], ph["router"][1]],
                 "n_tokens": int(y.numel()), "n_spec_tokens": int(n_sp)}
        for b in TB:
            u = torch.cat(U[b])
            lp = torch.log(PRI[b][rid] + float(TC["psem_eps"]))
            s_u = float(u.std(1).median())
            m = u.max(1).values - u.mean(1)
            med = float(m.median())
            pi = teacher("R3", u, lp, float(TC["c"]), float(TC["eta"]), s_u)
            e = pi.argmax(1)
            w = (m / med).clamp(float(TC["w_clip"][0]), float(TC["w_clip"][1]))
            pa = lp.argmax(1)
            es = e[:n_sp]
            load = torch.bincount(es, minlength=16).double() / max(1, n_sp)
            pr = load / load.sum()
            ar = torch.arange(u.shape[0], device=u.device)
            stats[b] = {"s_u": s_u, "med_margin": med, "override_frac_spec": float((es != pa[:n_sp]).float().mean()),
                        "override_frac_all": float((e != pa).float().mean()),
                        "teacher_load_max": float(load.max()), "teacher_load_eff": float(1.0 / (pr ** 2).sum()),
                        "teacher_n_dead": int((load < 1 / 160).sum()),
                        "u_teacher": float(u[ar, e].mean()), "u_prior_argmax": float(u[ar, pa].mean()),
                        "u_oracle": float(u.max(1).values.mean()), "u_mean_specialist": float(u.mean()),
                        "pi_top": float(pi.max(1).values.mean())}
            arrs[f"e_{b}"] = e.cpu().numpy().astype(np.int8)
            arrs[f"pi_{b}"] = pi[n_sp:].cpu().numpy().astype(np.float16)
            arrs[f"w_{b}"] = w[n_sp:].cpu().numpy().astype(np.float16)
            del u, lp, pi
        del U
        cache_dir.mkdir(parents=True, exist_ok=True)
        pth = cache_dir / f"cache_c{cyc:02d}.npz"
        if pth.exists():   # orphan of an interrupted refresh (written, not yet in a slot): kept aside, never reused
            aside = cache_dir / f"cache_c{cyc:02d}.orphan-{guard.utc_stamp()}.npz"
            os.replace(pth, aside)
            rd.log(f"orphan teacher cache moved aside: {aside.name}")
        tmp = cache_dir / f".cache_c{cyc:02d}.tmp.npz"
        if tmp.exists():
            tmp.unlink()
        np.savez(tmp, **arrs)
        with open(tmp, "rb") as fh:
            os.fsync(fh.fileno())
        os.replace(tmp, pth)
        pth.chmod(0o400)
        sha = guard.sha256_file(pth)
        stats["seconds"] = time.time() - t0_
        st["acc"]["refresh_s"] += stats["seconds"]
        st["refreshed"][str(cyc)] = {"path": str(pth.relative_to(guard.ROOT)), "sha256": sha, "stats": stats}
        guard.atomic_write_json(cache_dir / f"cache_c{cyc:02d}.json", st["refreshed"][str(cyc)], mode=0o400)
        rd.log(f"refresh cycle {cyc}: cache {pth.name} sha {sha[:12]} " + json.dumps(
            {b: {k: round(v, 5) if isinstance(v, float) else v for k, v in stats[b].items()} for b in TB}))
        rd.metric({"kind": "refresh", "cycle": cyc, "stats": stats, "sha256": sha})

    loaded = {"cyc": None}

    def cache_for(cyc):
        if loaded["cyc"] == cyc:
            return loaded
        rec = st["refreshed"][str(cyc)]
        pth = guard.ROOT / rec["path"]
        if guard.sha256_file(pth) != rec["sha256"]:
            raise guard.GuardError(f"teacher cache {pth} hash mismatch")
        if (pth.stat().st_mode & 0o222) != 0:
            raise guard.GuardError(f"teacher cache {pth} is writable")
        z = np.load(pth)
        loaded.clear()
        loaded.update({"cyc": cyc, "steps": z["steps"].tolist(), "off": z["off"], "n_sp": int(z["n_spec_tokens"]),
                       "y": torch.as_tensor(z["y"].astype(np.int64), device=device)})
        for b in TB:
            loaded[f"e_{b}"] = torch.as_tensor(z[f"e_{b}"].astype(np.int64), device=device)
            loaded[f"pi_{b}"] = torch.as_tensor(z[f"pi_{b}"].astype(np.float32), device=device)
            loaded[f"w_{b}"] = torch.as_tensor(z[f"w_{b}"].astype(np.float32), device=device)
        return loaded

    def cache_slice(cyc, step, y):
        c_ = cache_for(cyc)
        j = c_["steps"].index(step)
        lo, hi = int(c_["off"][j]), int(c_["off"][j + 1])
        if hi - lo != y.numel() or not bool((c_["y"][lo:hi] == y).all()):
            raise guard.GuardError(f"teacher cache misaligned at step {step}")
        return c_, lo, hi

    # ---------------- phase steps
    def base_step(step, lr):
        set_lr(opt, lr)
        rows = steps[step]
        n_valid = n_valid_of(step)
        parts = {}
        for mi_ in range(accum):
            rr = rows[mi_ * mbr:(mi_ + 1) * mbr]
            if not rr:
                continue
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
            for k in ("aux", "z"):
                parts[k] = parts.get(k, 0.0) + float(out[k].detach()) / accum
            parts["lm"] = parts.get("lm", 0.0) + float(lm.detach())
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), t["clip"])
        opt.step()
        opt.zero_grad(set_to_none=True)
        lmm = parts["lm"] / n_valid
        if not math.isfinite(lmm):
            raise FloatingPointError(f"non-finite base loss at step {step}")
        return {"lm": lmm, "gn": float(gn), "aux": parts["aux"], "z": parts["z"]}

    def spec_step(step, cyc, lr):
        f = frozen_scored(step)
        c_, lo, hi = cache_slice(cyc, step, f["y"])
        Wd = model.tok_emb.weight.detach()
        N = f["h"].shape[0]
        rec = {}
        for b in TB:
            set_lr(bopt[b], lr * float(cfg.get("spec_lr_mult", 1.0)))
            e = c_[f"e_{b}"][lo:hi]
            a_ = banks[b].correction(f["xn"], e)
            ce, ga = head_ce_grad(f["h"], a_.detach(), Wd, f["y"])
            a_.backward(ga / N)
            gnb = float(torch.nn.utils.clip_grad_norm_(banks[b].expert_params(), t["clip"]))
            bopt[b].step()
            bopt[b].zero_grad(set_to_none=True)
            ld = torch.bincount(e, minlength=16).float() / N
            rec[b] = {"res": float(ce.mean()), "gn": gnb, "load_max": float(ld.max())}
            del a_, ga
        set_lr(bopt["ALTG"], lr)
        dummy = torch.full_like(f["y"], -1)
        rg = residual_step(G, f["h"], Wd, f["y"], dummy, dummy, xn=f["xn"])
        G.router.weight.grad = None                       # router frozen in the specialist phase
        rg["gn"] = float(torch.nn.utils.clip_grad_norm_([G.experts.w1, G.experts.w2], t["clip"]))
        bopt["ALTG"].step()
        bopt["ALTG"].zero_grad(set_to_none=True)
        rec["ALTG"] = rg
        for b, v in rec.items():
            if not finite(v):
                raise FloatingPointError(f"non-finite bank {b} at step {step}: {v}")
        return rec

    def router_step(step, cyc, lr_g):
        f = frozen_scored(step)
        c_, lo, hi = cache_slice(cyc, step, f["y"])
        n_sp = c_["n_sp"]
        X = f["xn"].to(torch.float16)
        idx, mask, lens = doc_batch(f["dkey"], f["pos"])
        same = f["dkey"][1:] == f["dkey"][:-1]
        rec = {}
        for rn, r in routers.items():
            b = rmeta[rn]["bank"]
            pi = c_[f"pi_{b}"][lo - n_sp:hi - n_sp]
            w = c_[f"w_{b}"][lo - n_sp:hi - n_sp]
            r.train()
            lg, e = route(r, X, idx, mask, lens, grad=True)
            loss, rr_ = router_loss(lg, pi, w, float(RO["lam"]))
            ropt[rn].zero_grad(set_to_none=True)
            loss.backward()
            gnr = float(torch.nn.utils.clip_grad_norm_(r.parameters(), float(RO["clip"])))
            ropt[rn].step()
            ropt[rn].zero_grad(set_to_none=True)
            ld = torch.bincount(e, minlength=16).float() / e.numel()
            rec[rn] = {"loss": float(loss.detach()), **rr_, "gn": gnr,
                       "acc_pi": float((e == pi.argmax(1)).float().mean()),
                       "agree_teacher_e": float((e == c_[f"e_{b}"][lo:hi]).float().mean()),
                       "switch": float((e[1:] != e[:-1])[same].float().mean()), "load_max": float(ld.max()),
                       "n_dead": int((ld < 1 / 160).sum())}
            del lg, loss
        set_lr(gropt, lr_g)
        Wd = model.tok_emb.weight.detach()
        dummy = torch.full_like(f["y"], -1)
        rg = residual_step(G, f["h"], Wd, f["y"], dummy, dummy, xn=f["xn"])
        G.experts.w1.grad = None                           # experts frozen in the router phase
        G.experts.w2.grad = None
        rg["gn"] = float(torch.nn.utils.clip_grad_norm_([G.router.weight], t["clip"]))
        gropt.step()
        gropt.zero_grad(set_to_none=True)
        rec["ALTG"] = rg
        return rec

    def phase_summary(cyc, name, recs):
        """Concise per-phase evidence for OBSERVATIONS (means over the phase; router first/last step)."""
        if not recs:
            return {}
        out = {"cycle": cyc, "phase": name, "n_steps": len(recs)}
        if name == "base":
            out["lm"] = float(np.mean([r["lm"] for r in recs]))
        elif name == "spec":
            for b in list(TB) + ["ALTG"]:
                out[b] = {"res": float(np.mean([r[b]["res"] for r in recs])),
                          "load_max": float(np.mean([r[b]["load_max"] for r in recs]))}
        else:
            for rn in list(routers) + ["ALTG"]:
                v = [r[rn] for r in recs]
                if rn == "ALTG":
                    out[rn] = {"res": float(np.mean([x["res"] for x in v])), "load_max": float(np.mean([x["load_max"] for x in v]))}
                else:
                    out[rn] = {"kl_first": v[0]["kl"], "kl_last": v[-1]["kl"], "acc_first": v[0]["acc_pi"],
                               "acc_last": v[-1]["acc_pi"], "switch": float(np.mean([x["switch"] for x in v])),
                               "load_max": float(np.mean([x["load_max"] for x in v])),
                               "n_dead_max": int(max(x["n_dead"] for x in v))}
        return out

    last_slot, last_slot_step = time.time(), st["next_step"]
    every_steps, every_s = int(cfg.get("slot_every_steps", 100)), float(cfg.get("slot_every_s", 600))
    rewarm = int(cfg.get("rewarm", 20))
    phase_recs = st.setdefault("open_phase", {"key": None, "recs": []})
    try:
        rd.log(f"params: {model.param_counts()} banks { {n: b.n_params() for n, b in banks.items()} } "
               f"routers { {n: r.n_params() for n, r in routers.items()} }")
        model.train()
        if st["next_step"] == start and st["legs"] == 1:
            rd.metric({"kind": "param_hash", "cycle": 0, "after": "start", "step": start, "hash": param_hashes()})
        while st["next_step"] < end:
            step = st["next_step"]
            cyc, ph, s0, s1 = phase_of(step)
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
            if ph in ("spec", "router") and str(cyc) not in st["refreshed"]:
                if ph != "spec" or step != s0:
                    raise guard.GuardError(f"cycle {cyc} reached {ph} step {step} without a teacher refresh")
                try:
                    refresh(cyc)
                except Stopped as e_:
                    status, reason, rc = "EXTERNALLY_STOPPED", str(e_), 77
                    break
                rd.metric({"kind": "param_hash", "cycle": cyc, "after": "refresh", "step": st["next_step"], "hash": param_hashes()})
                save_slot(note=f"teacher refresh cycle {cyc}", tag="refresh")
                last_slot, last_slot_step = time.time(), st["next_step"]
                continue
            n_ph = st["n_done"][ph]
            warm = min(1.0, (n_ph + 1) / rewarm)
            lr_t = lr_at(step, n_steps, t) * warm
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            if ph == "base":
                rec = base_step(step, lr_t)
            elif ph == "spec":
                rec = spec_step(step, cyc, lr_t)
            else:
                rec = router_step(step, cyc, lr_t)
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - t0
            n_valid = n_valid_of(step)
            st["next_step"] = step + 1
            st["seen"] += n_valid
            st["n_done"][ph] += 1
            key = f"{cyc}:{ph}"
            st["phase_tok"][key] = st["phase_tok"].get(key, 0) + n_valid
            st["acc"]["step_s"] += dt
            if phase_recs["key"] != key:
                phase_recs.update({"key": key, "recs": []})
            phase_recs["recs"].append(rec)
            rd.metric({"step": step, "cycle": cyc, "phase": ph, "lr": lr_t, "seen": st["seen"], "dt": dt, "rec": rec})
            ns = st["next_step"]
            if ns == s1:
                summ = phase_summary(cyc, ph, phase_recs["recs"])
                summ["tokens"] = st["phase_tok"][key]
                summ["rows_sha256"] = hashlib.sha256(json.dumps(steps[s0:s1]).encode()).hexdigest()
                st["phase_rec"][key] = summ
                rd.log("phase end " + json.dumps(summ, default=lambda x: round(x, 5) if isinstance(x, float) else str(x)))
                rd.metric({"kind": "phase_end", **summ})
                phase_recs.update({"key": None, "recs": []})
                rd.metric({"kind": "param_hash", "cycle": cyc, "after": ph, "step": ns, "hash": param_hashes()})
                if ph == "router":
                    save_milestone(f"c{cyc:02d}", cyc)
                save_slot(note=f"end of cycle {cyc} {ph} phase", tag=ph)
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
               "seen": st["seen"], "n_done": st["n_done"], "leg_seconds": time.time() - t_proc0,
               "train_step_seconds": st["acc"]["step_s"], "refresh_seconds": st["acc"]["refresh_s"],
               "peak_alloc_gib": peak[0], "peak_reserved_gib": peak[1], "ckpts": st["ckpts"], "legs": st["legs"],
               "ckpt_seconds": st["acc"].get("ckpt_s", 0.0), "plan_sha256": plan_sha_full,
               "refreshed": {k: {"path": v["path"], "sha256": v["sha256"]} for k, v in st["refreshed"].items()}}
    stopf.close()
    if monitor is not None:
        monitor.stop()
    if a.device == "cuda":
        guard.gpu_ledger(t_start_iso, guard.now(), rd.run_id, f"asrs1 train {cfg['run_name']}", status)
    with open(rd.path / f"leg_{st['legs']:02d}.json", "w") as f:
        json.dump(summary, f, indent=1, default=str)
    rd.set_status(status, **{k: v for k, v in summary.items() if k != "status"})
    if status == "COMPLETE":
        guard.atomic_write_json(rd.path / "summary.json", summary, overwrite=True)
        guard.atomic_write_json(rd.path / "phase_records.json", st["phase_rec"], overwrite=True)
        store.prune_after_complete()
    rd.log(f"{status} {reason[:500]}")
    return rc


if __name__ == "__main__":
    main()
