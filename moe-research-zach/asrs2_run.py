#!/usr/bin/env python
"""csr-continuous-base-target-free-asrs-v2 (asrs2) orchestrator. Every job goes through asrs2_driver.py (GPU jobs
through scripts/gpu_guarded.sh).

  asrs2_run.py tests       CPU tests (asrs2_tests.py)
  asrs2_run.py gpucheck    pre-freeze powered base-matching + SIGINT/resume + speed check (main-stream rows only)
  asrs2_run.py freeze      write the prereg sha256 sidecar (0400)
  asrs2_run.py stage0f     Stage 0F: frozen-stage K=5 banks + routers (4 specialist seeds), seed-0 SEL evaluation, gate
  asrs2_run.py stage1      Stage 1A: seed-0 continuation (all arms), milestone SEL evaluations/analyses, criteria
                           (env ASRS2_GEN / ASRS2_CHANGE for a logged engineering generation)
  asrs2_run.py final       final_analysis.json + protected inventory verification
External interruption (driver exit 77): INCOMPLETE_STOPPED_BY_EXTERNAL_USER + exact resume command; never restarts."""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402
from csr import asrs2_log as LG  # noqa: E402
from csr import asrs2_paths as P  # noqa: E402

PY = str(guard.ROOT / ".venv/bin/python")
CMD_NOW = ""
CAP = {"gpucheck": "stage0", "stage0f": "stage0", "stage1": "stage1", "stage2": "stage2", "stage3": "stage3"}


def rel(p):
    return str(Path(p).relative_to(guard.ROOT))


def stop_external(job_id):
    LG.set_state("INCOMPLETE_STOPPED_BY_EXTERNAL_USER", job=job_id)
    LG.write_resume("INCOMPLETE_STOPPED_BY_EXTERNAL_USER",
                    f"{PY} asrs2_driver.py authorize-resume && nohup {PY} asrs2_run.py {CMD_NOW} > "
                    f"{rel(P.ART)}/run_{'_'.join(CMD_NOW.split())}.log 2>&1 &",
                    notes=f"Interrupted job: `{job_id}`. Treated as another user needing the GPU; nothing restarts "
                          f"automatically; all asrs2 launchers/watchers stopped. Wait for explicit user permission. "
                          f"Rerunning the same phase resumes from the latest verified slot (training) or redoes the "
                          f"interrupted evaluation.")
    print(f"EXTERNAL STOP at {job_id}; see RESUME.md", flush=True)
    sys.exit(77)


def job(job_id, stage, argv, gpu=True, expect="0", disk_gb=1.0, seed="", env_extra=None):
    tot, by = LG.gpu_hours()
    grp = CAP.get(stage.split("_")[0], "stage1")
    used = sum(v for k, v in by.items() if CAP.get(k.split("_")[0], "stage1") == grp)
    if gpu and (used >= P.GPU_CAP_H[grp] or tot >= P.GPU_CAP_H["total"]):
        LG.set_state("STOPPED_GPU_CAP", gpu_h=used, group=grp)
        LG.write_resume("STOPPED_GPU_CAP", f"{PY} asrs2_run.py {CMD_NOW}", notes=f"{grp} GPU cap reached ({used:.2f} h)")
        raise SystemExit(f"GPU cap reached ({used:.2f} h)")
    free = guard.disk_free_bytes() / 2**30
    if free - disk_gb < P.DISK_RESERVE_GIB:
        LG.set_state("STOPPED_DISK", free_gib=free, need_gib=disk_gb)
        LG.write_resume("STOPPED_DISK", f"{PY} asrs2_run.py {CMD_NOW}",
                        notes=f"Disk shortfall: free {free:.2f} GiB, need 100 + {disk_gb:.2f} GiB.")
        raise SystemExit(f"disk shortfall {100 + disk_gb - free:.2f} GiB")
    LG.disk(job_id, disk_gb)
    st = json.loads((P.ART / "driver_state.json").read_text()) if (P.ART / "driver_state.json").exists() else {"jobs": {}}
    if st["jobs"].get(job_id, {}).get("status") == "COMPLETE":
        return 0
    LG.write_resume(f"RUNNING {job_id}", f"{PY} asrs2_run.py {CMD_NOW}",
                    notes="Running normally. If this process is gone without a final state, rerun the command above "
                          "(completed jobs are skipped; training resumes from the latest verified slot).")
    t0, t_iso = time.time(), guard.now()
    cmd = [PY, "asrs2_driver.py", "job", job_id, "--gpu" if gpu else "--cpu", "--expect-rc", expect, "--disk-gb",
           str(disk_gb), "--"] + argv
    env = os.environ.copy()
    env.update(env_extra or {})
    rc = subprocess.run(cmd, cwd=guard.ROOT, env=env).returncode
    dt = time.time() - t0
    st = json.loads((P.ART / "driver_state.json").read_text())
    j = st["jobs"].get(job_id, {})
    if gpu:
        LG.compute(job_id, stage, t_iso, guard.now(), dt, j.get("status", "?"))
    LG.manifest({"time": guard.now(), "job": job_id, "stage": stage, "seed": seed, "status": j.get("status"),
                 "rc": j.get("rc"), "seconds": round(dt, 1), "gpu_h": round(dt / 3600, 4) if gpu else 0,
                 "cmd": " ".join(argv)})
    LG.run_log(job=job_id, stage=stage, seed=seed, cmd=argv, gpu=gpu, start=t_iso, end=guard.now(), rc=rc,
               status=j.get("status"))
    if rc == 77:
        stop_external(job_id)
    if rc != 0:
        LG.set_state("FAILED", job=job_id, rc=rc)
        LG.write_resume("FAILED", f"{PY} asrs2_run.py {CMD_NOW}", notes=f"Job {job_id} failed (rc {rc}); see job_logs.")
        raise SystemExit(f"job {job_id} failed rc={rc}")
    return j.get("rc")


def run_dir(name, base=None):
    c = sorted(Path(base or guard.ROOT / "runs").glob(f"*_asrs2_{name}"))
    return c[-1] if c else None


def train(name, cfgp, stage, max_legs=6, disk_gb=4.0, env_extra=None, base=None):
    for leg in range(1, max_legs + 1):
        rd = run_dir(name, base)
        if rd is not None and json.loads((rd / "status.json").read_text())["status"] == "COMPLETE":
            return rd
        argv = ["asrs2_train.py", "--config", rel(cfgp)]
        if rd is not None:
            argv += ["--run-dir", rel(rd)]
        st = json.loads((P.ART / "driver_state.json").read_text()) if (P.ART / "driver_state.json").exists() else {"jobs": {}}
        k = leg
        while st["jobs"].get(f"train_{name}_leg{k}", {}).get("status") == "COMPLETE":
            k += 1
        job(f"train_{name}_leg{k}", stage, argv, expect="0,75", disk_gb=disk_gb, env_extra=env_extra)
    rd = run_dir(name, base)
    if json.loads((rd / "status.json").read_text())["status"] != "COMPLETE":
        raise guard.GuardError("too many training legs")
    return rd


def final_slot(rd):
    import torch
    idx = json.loads((P.TMP / "slots" / rd.name / "resume_index.json").read_text())
    e = idx["entries"][-1]
    return torch.load(guard.ROOT / e["path"] / "state.pt", map_location="cpu", weights_only=False), e


def drop_slots(rds, why):
    for rd in rds:
        d = P.TMP / "slots" / rd.name
        if d.exists():
            for f in d.rglob("state.pt"):
                LG.deletion(rel(f), guard.sha256_file(f), f.stat().st_size, why)
            shutil.rmtree(d)


# ------------------------------------------------------------------------------------------------ pre-freeze
def gpucheck():
    """Powered base-matching check (main-stream rows only, before the freeze):
      GF  frozen-mode scratch run, steps 0..20 (seed 0) -> init milestone for the continuation checks
      GA  continuation with ALL seed-0 arms, steps 318..381 (64 steps: SPEC 32, ROUTER 16, SPEC 16)
      GB  base-only continuation, same steps;  GC  base-only repeat (GPU nondeterminism floor)
      GD  = GA config with an external SIGINT at step 350 -> exit 77 -> resume
    Pass (fixed before running): per-step base loss mean|GA-GB| <= 1.5 mean|GB-GC| + 2e-5 and |mean(GA-GB)| <=
    3 sd(GB-GC)/sqrt(64) + 2e-5; final fp32 base relative L2 distance GA-GB <= 1.5 x GB-GC + 1e-7; 64 base updates in
    every run; identical row hashes; GD legs EXTERNALLY_STOPPED then COMPLETE with base distance to GA within 1.5 x
    the floor + 1e-7."""
    import numpy as np
    import torch
    from csr import asrs2_cfg as C
    base = P.TMP / "gpucheck_runs"
    base.mkdir(parents=True, exist_ok=True)
    env = {"MOE_RUNS_DIR": str(base)}
    out = {"time": guard.now()}
    cf = C.build("GF", "frozen", seeds=[0], test=True, start=0, end=20, milestones={"F": 20})
    cf["verify_ckpt"] = True
    rdF = train("GF", C.write(cf, "GPUCHK_GF"), "gpucheck", env_extra=env, base=base, disk_gb=2.0)
    initF = rel(rdF / "ckpt_F")
    kw = dict(test=True, start=318, end=382, init_ckpt=initF, init_sha=None, milestones={})
    runs = {}
    for nm, seeds, arms, extra in (("GA", [0], None, []), ("GB", [], [], []), ("GC", [], [], []),
                                   ("GD", [0], None, ["--test-signal-at-step", "350"])):
        cfg = C.build(nm, "cont", seeds=seeds, arms=arms, **kw)
        cfg["verify_ckpt"] = True
        cp = C.write(cfg, f"GPUCHK_{nm}")
        rd = run_dir(nm, base)
        if rd is None or json.loads((rd / "status.json").read_text())["status"] != "COMPLETE":
            argv = ["asrs2_train.py", "--config", rel(cp)] + extra
            if rd is None:
                job(f"gpucheck_{nm}_1", "gpucheck", argv, expect="0,77" if extra else "0", disk_gb=4.0, env_extra=env)
            rd = run_dir(nm, base)
            if json.loads((rd / "status.json").read_text())["status"] != "COMPLETE":
                job(f"gpucheck_{nm}_2", "gpucheck", ["asrs2_train.py", "--config", rel(cp), "--run-dir", rel(rd)],
                    disk_gb=4.0, env_extra=env)
        runs[nm] = run_dir(nm, base)
    M = {nm: [json.loads(x) for x in open(rd / "metrics.jsonl")] for nm, rd in runs.items()}
    lm = {nm: np.asarray([m["lm"] for m in ms if "lm" in m and m.get("lm") is not None]) for nm, ms in M.items()}
    rows = {nm: [m["rows_sha"] for m in ms if "rows_sha" in m] for nm, ms in M.items()}
    S = {nm: final_slot(rd)[0] for nm, rd in runs.items()}
    flat = lambda s: torch.cat([v.float().reshape(-1) for k, v in sorted(s["model"].items())])
    th = {nm: flat(s) for nm, s in S.items()}
    rl2 = lambda a, b: float((th[a] - th[b]).norm() / th[b].norm())
    dab, dbc = lm["GA"] - lm["GB"], lm["GB"] - lm["GC"]
    n = len(dab)
    floor_p = rl2("GB", "GC")
    out["per_step"] = {"n_steps": n, "mean_abs_GA_GB": float(np.abs(dab).mean()), "mean_abs_GB_GC": float(np.abs(dbc).mean()),
                       "mean_GA_GB": float(dab.mean()), "sd_GB_GC": float(dbc.std()), "max_abs_GA_GB": float(np.abs(dab).max()),
                       "max_abs_GB_GC": float(np.abs(dbc).max()), "mean_abs_GD_GA": float(np.abs(lm["GD"] - lm["GA"]).mean())}
    s3 = {r["step"]: r["lm"] for r in (json.loads(x) for x in open(guard.ROOT / P.CCSR1_S3_RUN / "metrics.jsonl")) if "lm" in r}
    out["per_step"]["mean_abs_GB_vs_ccsr1_S3_recorded"] = float(np.mean([abs(l_ - s3[318 + i]) for i, l_ in enumerate(lm["GB"])]))
    out["params"] = {"rel_l2_GA_GB": rl2("GA", "GB"), "rel_l2_GB_GC": floor_p, "rel_l2_GD_GA": rl2("GD", "GA")}
    out["counts"] = {nm: s["st"]["n_upd"].get("base") for nm, s in S.items()}
    legsD = [json.loads(x.read_text()) for x in sorted(runs["GD"].glob("leg_*.json"))]
    ps = out["per_step"]
    out["checks"] = {
        "lm_mean_abs": ps["mean_abs_GA_GB"] <= 1.5 * ps["mean_abs_GB_GC"] + 2e-5,
        "lm_no_bias": abs(ps["mean_GA_GB"]) <= 3 * ps["sd_GB_GC"] / np.sqrt(n) + 2e-5,
        "params": out["params"]["rel_l2_GA_GB"] <= 1.5 * floor_p + 1e-7,
        "base_updates_64": all(v == 64 for v in out["counts"].values()),
        "rows_identical": rows["GA"] == rows["GB"] == rows["GC"] == rows["GD"] and len(rows["GA"]) == 64,
        "sigint_resume": len(legsD) == 2 and legsD[0]["status"] == "EXTERNALLY_STOPPED" and legsD[1]["status"] == "COMPLETE",
        "resume_params": out["params"]["rel_l2_GD_GA"] <= 1.5 * floor_p + 1e-7}
    out["pass"] = bool(all(out["checks"].values()))
    la = json.loads(sorted(runs["GA"].glob("leg_*.json"))[0].read_text())
    lb = json.loads(sorted(runs["GB"].glob("leg_*.json"))[0].read_text())
    lf = json.loads(sorted(rdF.glob("leg_*.json"))[0].read_text())
    out["timing"] = {"GA_step_s": la["train_step_seconds"] / 64, "GB_step_s": lb["train_step_seconds"] / 64,
                     "aux_overhead_frac": la["train_step_seconds"] / lb["train_step_seconds"] - 1,
                     "GF_step_s_seed0": lf["train_step_seconds"] / 20, "peak_alloc_GA_gib": la["peak_alloc_gib"],
                     "peak_alloc_GF_gib": lf["peak_alloc_gib"]}
    out["projection"] = {"stage0f_h_4seeds": 318 * out["timing"]["GF_step_s_seed0"] * 2.5 / 3600,
                         "stage1_train_h": 793 * out["timing"]["GA_step_s"] / 3600}
    p = P.ART / f"gpu_check_{guard.utc_stamp()}.json"
    guard.atomic_write_json(p, out, mode=0o400)
    drop_slots(list(runs.values()) + [rdF], "asrs2 GPU-check scratch resume slot")
    for rd in list(runs.values()) + [rdF]:
        for f in rd.rglob("state.pt"):
            LG.deletion(rel(f), guard.sha256_file(f), f.stat().st_size, "asrs2 GPU-check scratch milestone")
            f.unlink()
    print(json.dumps({k: out[k] for k in ("per_step", "params", "counts", "checks", "pass", "timing", "projection")}, default=str))
    return out


def freeze():
    if P.PREREG_SHA_FILE.exists():
        return P.check_prereg()
    h = guard.sha256_file(P.PREREG)
    P.PREREG.chmod(0o400)
    guard.atomic_write_bytes(P.PREREG_SHA_FILE, f"{h}  {P.PREREG.name}\n".encode())
    P.PREREG_SHA_FILE.chmod(0o400)
    return h


# ------------------------------------------------------------------------------------------------ stages
def eval_ckpt(tag, ckpt, out, stage, part="SEL", seeds="0"):
    if not (out / f"summary_{part}.json").exists():
        job(f"eval_{tag}", stage, ["asrs2_eval.py", "--ckpt", ckpt, "--part", part, "--out-dir", rel(out), "--seeds", seeds],
            disk_gb=1.5)
    job(f"analyze_{tag}", stage, ["asrs2_analyze.py", "ckpt", "--tag", tag, "--eval-dir", rel(out), "--part", part,
                                  "--seeds", seeds], gpu=False, disk_gb=0.1)
    return json.loads((P.ANALYSIS / f"{tag}.json").read_text())


def stage0f():
    from csr import asrs2_cfg as C
    P.check_prereg()
    if (P.ART / "stage0f.json").exists():
        return json.loads((P.ART / "stage0f.json").read_text())
    LG.set_state("STAGE0F_RUNNING")
    cfg = C.build("S0F", "frozen", seeds=list(P.SPEC_SEEDS))
    rd = train("S0F", C.write(cfg, "S0F"), "stage0f", disk_gb=2.0)
    st = json.loads((rd / "status.json").read_text())
    ck = st["ckpts"]["F"]
    LG.observe(f"Stage 0F training COMPLETE ({rd.name}): {st['seen']} tokens (steps 0..317, base frozen), seeds "
               f"{list(P.SPEC_SEEDS)}, step s {st['train_step_seconds']:.0f}, peak {st['peak_alloc_gib']:.1f} GiB; milestone "
               f"{ck['path']} sha256 {ck['sha256'][:12]}")
    A = eval_ckpt("F_s0", ck["path"], P.EVALS / "F_s0", "stage0f")
    import asrs2_analyze as AN
    g = AN.stage0f("F_s0")
    S = A["seeds"]["0"]
    F_ = S["functional"]["SEM"]
    out = {"time": guard.now(), "run": rd.name, "ckpt": ck, **g, "ce_mean": S["ce_mean"], "contrasts": S["contrasts"],
           "functional": {k: {kk: v[kk] for kk in ("I", "I_SHUF", "dI", "dI_ci", "per_specialist", "load", "switch",
                                                    "agree_region_prior", "I_ordinary", "I_ordinary_SHUF")}
                          for k, v in S["functional"].items()},
           "predictability": S["predictability"], "r2_reliability": S.get("r2_reliability")}
    guard.atomic_write_json(P.ART / "stage0f.json", out, mode=0o400)
    c = S["contrasts"]
    LG.observe(f"## Stage 0F gate (SEL, seed 0, frozen base step 318) — {'PASS' if g['pass'] else 'FAIL'}\n\n"
               f"**Measurements:** base {S['ce_mean']['base']:.5f}, G5 {S['ce_mean']['G']:.5f}; SEM5 (causal R1 router) "
               f"- G5 {c['SEM-G']['v']:+.5f} {c['SEM-G']['ci']}, - SHUF5 {c['SEM-SEM_SHUF']['v']:+.5f} {c['SEM-SEM_SHUF']['ci']}; "
               f"region teacher SEM - G5 {c['T_SEM-G']['v']:+.5f}, SHUF teacher - G5 {c['T_SEM_SHUF-G']['v']:+.5f}; "
               f"exposure-matched I_SEM {F_['I']:.5f}, I_SHUF {F_['I_SHUF']:.5f}, dI {F_['dI']:+.5f} {F_['dI_ci']}; "
               f"ordinary (unmatched) I {F_['I_ordinary']:.5f} vs SHUF {F_['I_ordinary_SHUF']:.5f}; positive specialists "
               f"{F_['per_specialist']['n_pos']}/5 (significant {F_['per_specialist']['n_sig']}); load {[round(x, 3) for x in F_['load']]}, "
               f"switch {F_['switch']:.3f}, region-prior agreement {F_['agree_region_prior']:.3f}; oracle predictability "
               f"{json.dumps(S['predictability']['SEM']['oracle'])}; R2 reliability {json.dumps({k: v for k, v in (S.get('r2_reliability') or {}).items() if k != 'U_A'})}.\n"
               f"**Governing rule:** prereg Stage-0F gate (F0-1 dI > 0 with CI lower > 0; F0-2 >= 4/5 positive).")
    LG.decision(f"stage0f.json: dI {F_['dI']:+.5f} {F_['dI_ci']}, n_pos {F_['per_specialist']['n_pos']}/5",
                "semantic function vs persistent grouping (SHUF) vs generic (G5)",
                "proceed to Stage 1A" if g["pass"] else "TERMINAL_STAGE0F_FAIL (no functional K=5 frozen bank to preserve)",
                "prereg Stage-0F gate", "Stage-1 criteria" if g["pass"] else "n/a", stage="stage0f")
    return out


def stage1(gen=0, change=None):
    from csr import asrs2_cfg as C
    P.check_prereg()
    sfx = f"_g{gen}" if gen else ""
    name = f"S1{sfx}"
    if (P.ART / f"stage1_gen{gen}.json").exists():
        return json.loads((P.ART / f"stage1_gen{gen}.json").read_text())
    s0 = json.loads((P.ART / "stage0f.json").read_text())
    if not s0["pass"]:
        raise guard.GuardError("Stage 0F failed: Stage 1 not permitted")
    LG.set_state(f"STAGE1{sfx}_RUNNING")
    kw = {}
    if change == "router_lr_x3":
        kw["router_opt"] = {**P.ROUTER_OPT, "lr": P.ROUTER_OPT["lr"] * 3}
    elif change == "router_lr_x1/3":
        kw["router_opt"] = {**P.ROUTER_OPT, "lr": P.ROUTER_OPT["lr"] / 3}
    elif change == "spec_lr_x0.5":
        kw["spec_lr_mult"] = 0.5
    elif change == "window_24_24":
        kw.update(spec_w=24, router_w=24)
    elif change not in (None, ""):
        raise guard.GuardError(f"change {change} needs a logged amendment before it can run")
    cfg = C.build(name, "cont", seeds=[0], init_ckpt=s0["ckpt"]["path"], init_sha=s0["ckpt"]["sha256"], gen=gen,
                  change=change, **kw)
    rd = train(name, C.write(cfg, name), f"stage1{sfx}", disk_gb=4.0)
    st = json.loads((rd / "status.json").read_text())
    LG.observe(f"Stage-1{sfx} training COMPLETE ({rd.name}): {st['seen']} tokens, base updates {st['n_upd'].get('base')}, "
               f"step s {st['train_step_seconds']:.0f}, peak {st['peak_alloc_gib']:.1f} GiB, legs {st['legs']}")
    window_summary(rd)
    tags = []
    for ms in sorted(st["ckpts"], key=lambda k: st["ckpts"][k]["step"]):
        tag = f"{rd.name}_{ms}"
        res = eval_ckpt(tag, st["ckpts"][ms]["path"], P.EVALS / rd.name / ms, f"stage1{sfx}")
        tags.append(tag)
        window_observation(rd.name, ms, res)
    bc = sorted(P.ART.glob("base_match_*.json"))
    job(f"basematch_{rd.name}", f"stage1{sfx}", ["asrs2_basematch.py", "--run", rd.name], gpu=False, disk_gb=0.05)
    bc = sorted(P.ART.glob(f"base_match_{rd.name}.json"))
    job(f"analyze_stage1{sfx}", f"stage1{sfx}", ["asrs2_analyze.py", "stage1", "--frozen", "F_s0", "--final", tags[-1],
                                                 "--traj", ",".join(["F_s0"] + tags), "--base-check", rel(bc[-1]).split(rel(P.ART) + "/")[1],
                                                 "--gen", str(gen)], gpu=False, disk_gb=0.1)
    return json.loads((P.ART / f"stage1_gen{gen}.json").read_text())


def window_summary(rd):
    """Per auxiliary window (48 steps) training-time record from metrics.jsonl: base loss (and vs the recorded ccsr1 S3
    ordinary continuation), per-arm held-out utilities (assigned = prior choice, deployed router choice, average,
    oracle; computed before each step's update on a 1-in-4 token subset), residual CE, router KL, agreement, switch,
    load. Written to ART/windows_<run>.json and summarised in OBSERVATIONS."""
    import numpy as np
    M = [json.loads(x) for x in open(rd / "metrics.jsonl")]
    M = [m for m in M if "rows_sha" in m]
    s3 = {r["step"]: r["lm"] for r in (json.loads(x) for x in open(guard.ROOT / P.CCSR1_S3_RUN / "metrics.jsonl")) if "lm" in r}
    W = {}
    for m in M:
        w = W.setdefault(m["window"], {"steps": [], "lm": [], "dlm_s3": [], "arms": {}})
        w["steps"].append(m["step"])
        w["lm"].append(m["lm"])
        w["dlm_s3"].append(m["lm"] - s3[m["step"]])
        for a_, r_ in m["arms"].items():
            A = w["arms"].setdefault(a_, {})
            for k, v in r_.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    A.setdefault(f"{m['phase']}:{k}", []).append(v)
    out = {}
    for w, v in sorted(W.items()):
        out[str(w)] = {"steps": [min(v["steps"]), max(v["steps"])], "lm": float(np.mean(v["lm"])),
                       "dlm_vs_ccsr1_S3": float(np.mean(v["dlm_s3"])),
                       "arms": {a_: {k: float(np.mean(x)) for k, x in A.items()} for a_, A in v["arms"].items()}}
    guard.atomic_write_json(P.ART / f"windows_{rd.name}.json", out, overwrite=True)
    lines = ["| window | steps | base lm | vs S3 | SEMh u_prior / u_router / u_mean / u_oracle | SHUFh u_prior / u_mean | "
             "SEMs u_prior | FROZ u_prior | SEMh agree / switch | G res - ALTG res |", "|" + "---|" * 10]
    g = lambda d_, k: d_.get("spec:" + k, d_.get("router:" + k, float("nan")))
    for w, v in out.items():
        A = v["arms"]
        se, sh, ss, fr = (A.get(f"{k}_0", {}) for k in ("SEMh", "SHUFh", "SEMs", "FROZ"))
        gr = g(A.get("G_0", {}), "res") - g(A.get("ALTG_0", {}), "res")
        lines.append(f"| {w} | {v['steps'][0]}-{v['steps'][1]} | {v['lm']:.4f} | {v['dlm_vs_ccsr1_S3']:+.4f} | "
                     f"{g(se, 'u_prior'):+.4f} / {g(se, 'u_router'):+.4f} / {g(se, 'u_mean'):+.4f} / {g(se, 'u_oracle'):+.4f} | "
                     f"{g(sh, 'u_prior'):+.4f} / {g(sh, 'u_mean'):+.4f} | {g(ss, 'u_prior'):+.4f} | {g(fr, 'u_prior'):+.4f} | "
                     f"{g(se, 'agree_prior'):.3f} / {g(se, 'switch'):.3f} | {gr:+.4f} |")
    LG.observe(f"## Per-window training record {rd.name} (held-out utilities = CE_base - CE_specialist on each step's "
               f"fresh tokens before its update; positive = helpful)\n\n" + "\n".join(lines) + "\n")
    return out


def window_observation(run, ms, res):
    S = res["seeds"]["0"]
    c = S["contrasts"]
    f = lambda x: f"{x:+.5f}"
    parts = []
    for tagp in ("SEM", "SOFT0", "FROZ"):
        if tagp not in S["functional"]:
            continue
        F_ = S["functional"][tagp]
        parts.append(f"{tagp}: - G {f(c[f'{tagp}-G']['v'])}, - ALT-G {f(c[f'{tagp}-ALTG']['v'])}, - its SHUF "
                     f"{f(c[f'{tagp}-{tagp}_SHUF']['v'])}; region teacher - ALT-G {f(c[f'T_{tagp}-ALTG']['v'])}; dI {f(F_['dI'])} "
                     f"[{F_['dI_ci'][0]:+.4f}, {F_['dI_ci'][1]:+.4f}]; pos specialists {F_['per_specialist']['n_pos']}/5; "
                     f"mean specialist delta {f(F_['mean_specialist_delta'])} (SHUF {f(F_['mean_specialist_delta_SHUF'])}); "
                     f"switch {F_['switch']:.3f}, prior agreement {F_['agree_region_prior']:.3f}, load_max {F_['load_max']:.3f}, "
                     f"dead {F_['n_dead']}; post-switch vs ALT-G {f(F_['post_switch']['vs_ref'])}, stable {f(F_['stable']['vs_ref'])}")
    pr = S["predictability"]["SEM"]
    LG.observe(f"## asrs2 {run} milestone {ms} (step {res['step']}; SEL; automatic record)\n\n"
               f"**Observed:** base {S['ce_mean']['base']:.5f}, G {S['ce_mean']['G']:.5f}, ALT-G {S['ce_mean']['ALTG']:.5f} "
               f"(G - ALT-G {f(c['G-ALTG']['v'])}); " + "; ".join(parts) +
               f". Oracle-specialist predictability (SEM): {json.dumps({k: round(v, 3) for k, v in pr['oracle'].items()})}; "
               f"assigned: {json.dumps({k: round(v, 3) for k, v in pr['assigned'].items()})}.\n")


def c1diag():
    """Two base-only continuations 318 -> 1111 in separate processes + SEL base CE (C1 localisation; decision logged)."""
    import numpy as np
    from csr import asrs2_cfg as C
    P.check_prereg()
    out = {"time": guard.now(), "runs": {}}
    s3 = {r["step"]: r["lm"] for r in (json.loads(x) for x in open(guard.ROOT / P.CCSR1_S3_RUN / "metrics.jsonl")) if "lm" in r}
    lm = {"S3": np.asarray([s3[s] for s in range(P.START, P.END)])}
    s1 = run_dir("S1")
    lm["S1"] = np.asarray([m["lm"] for m in (json.loads(x) for x in open(s1 / "metrics.jsonl")) if "rows_sha" in m])
    sel = {"S3": 3.6227712345149667,
           "S1": json.loads((P.ANALYSIS / f"{s1.name}_m1111.json").read_text())["ce_base"]}
    for nm in ("C1B1", "C1B2"):
        cfg = C.build(nm, "cont", seeds=[], arms=[], milestones={"m1111": 1111})
        rd = train(nm, C.write(cfg, nm), "stage1_c1diag", disk_gb=3.0)
        st = json.loads((rd / "status.json").read_text())
        ev = P.EVALS / rd.name / "m1111"
        if not (ev / "summary_SEL.json").exists():
            job(f"eval_{rd.name}_m1111", "stage1_c1diag", ["asrs2_eval.py", "--ckpt", st["ckpts"]["m1111"]["path"], "--part",
                                                            "SEL", "--out-dir", rel(ev)], disk_gb=0.5)
        sel[nm] = json.loads((ev / "summary_SEL.json").read_text())["ce_base"]
        lm[nm] = np.asarray([m["lm"] for m in (json.loads(x) for x in open(rd / "metrics.jsonl")) if "rows_sha" in m])
        out["runs"][nm] = {"run": rd.name, "base_updates": st["n_upd"].get("base"), "step_s": st["train_step_seconds"] / 793}
    names = list(lm)
    out["sel_ce_base"] = sel
    out["pairs"] = {}
    for i, a_ in enumerate(names):
        for b_ in names[i + 1:]:
            d = lm[a_] - lm[b_]
            out["pairs"][f"{a_}-{b_}"] = {"mean_dlm": float(d.mean()), "mean_abs_dlm": float(np.abs(d).mean()),
                                          "sel_diff": float(sel[a_] - sel[b_])}
    spread = abs(out["pairs"]["C1B1-C1B2"]["sel_diff"])
    s1_dev = max(abs(sel["S1"] - sel["C1B1"]), abs(sel["S1"] - sel["C1B2"]))
    out["bug_indicated"] = bool(s1_dev > 1.5 * max(spread, abs(sel["C1B1"] - sel["S3"]), abs(sel["C1B2"] - sel["S3"])) + 1e-4)
    guard.atomic_write_json(P.ART / "c1_diagnostic.json", out, mode=0o400)
    print(json.dumps(out, indent=1))
    return out


def final():
    tot, by = LG.gpu_hours()
    job("inventory_verify", "final", ["asrs2_inventory.py", "verify"], gpu=False, disk_gb=0.05)
    inv = sorted(P.ART.glob("protected_inventory_verify_*.json"))
    out = {"time": guard.now(), "experiment": P.EXP, "prereg_sha256": P.check_prereg(), "gpu_hours": tot, "gpu_by_stage": by,
           "stage0f": json.loads((P.ART / "stage0f.json").read_text()) if (P.ART / "stage0f.json").exists() else None,
           "stage1": {p.name: json.loads(p.read_text()) for p in sorted(P.ART.glob("stage1_gen*.json"))},
           "protected_inventory": json.loads(inv[-1].read_text()) if inv else None, "test_split": "never loaded"}
    guard.atomic_write_json(P.ART / "final_analysis.json", out, overwrite=True)
    return out


def main():
    global CMD_NOW
    what = sys.argv[1]
    CMD_NOW = " ".join(sys.argv[1:])
    guard.check_root()
    P.TMP.mkdir(parents=True, exist_ok=True)
    P.ART.mkdir(parents=True, exist_ok=True)
    if what == "tests":
        job("cpu_tests", "setup", ["asrs2_tests.py"], gpu=False, disk_gb=3.0)
    elif what == "gpucheck":
        gpucheck()
    elif what == "freeze":
        print(freeze())
    elif what == "stage0f":
        s0 = stage0f()
        LG.set_state("STAGE0F_PASS" if s0["pass"] else "TERMINAL_STAGE0F_FAIL")
        LG.write_resume(json.loads((P.ART / "STATE.json").read_text())["state"],
                        f"{PY} asrs2_run.py stage1" if s0["pass"] else "true  # terminal; see report")
    elif what == "stage1":
        gen = int(os.environ.get("ASRS2_GEN", "0"))
        change = os.environ.get("ASRS2_CHANGE") or None
        s1 = stage1(gen, change)
        LG.set_state(f"STAGE1_GEN{gen}_{'PASS' if s1['pass'] else 'FAIL'}_AWAITING_DECISION")
        LG.write_resume(json.loads((P.ART / "STATE.json").read_text())["state"], "true  # see DECISION_LOG")
    elif what == "c1diag":
        c1diag()
    elif what == "final":
        final()


if __name__ == "__main__":
    main()
