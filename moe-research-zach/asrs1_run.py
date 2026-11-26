#!/usr/bin/env python
"""csr-alternating-semantic-residual-specialization-v1 (asrs1) orchestrator. Every job goes through asrs1_driver.py
(GPU jobs through scripts/gpu_guarded.sh).

  asrs1_run.py inventory   protected inventory (before) + baseline-reuse manifest
  asrs1_run.py tests       CPU tests (asrs1_tests.py)
  asrs1_run.py gpucheck    pre-freeze GPU correctness/speed check on main-stream training rows only (1 short cycle,
                           uninterrupted + SIGINT/resume); test scratch deleted with sha256 afterwards
  asrs1_run.py freeze      write the prereg sha256 sidecar (0400)
  asrs1_run.py stage1      c00 + ordinary-continuation reference evaluations on SEL, the alternating run, per-cycle SEL
                           evaluations and analyses, Stage-1 criteria, section-8 generation rule (<= 2 generations)
  asrs1_run.py final       final_analysis.json + protected inventory verification
External interruption (driver exit 77): INCOMPLETE_STOPPED_BY_EXTERNAL_USER + exact resume command; never restarts."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402
from csr import asrs1_log as LG  # noqa: E402
from csr import asrs1_paths as P  # noqa: E402

PY = str(guard.ROOT / ".venv/bin/python")
CMD_NOW = ""


def rel(p):
    return str(Path(p).relative_to(guard.ROOT))


def stop_external(job_id):
    LG.set_state("INCOMPLETE_STOPPED_BY_EXTERNAL_USER", job=job_id)
    LG.write_resume("INCOMPLETE_STOPPED_BY_EXTERNAL_USER",
                    f"{PY} asrs1_driver.py authorize-resume && nohup {PY} asrs1_run.py {CMD_NOW} > "
                    f"{rel(P.ART)}/run_{'_'.join(CMD_NOW.split())}.log 2>&1 &",
                    notes=f"Interrupted job: `{job_id}`. Treated as another user needing the GPU; nothing restarts "
                          f"automatically. Wait for explicit user permission. Rerunning the same phase resumes from the "
                          f"latest verified slot (training) or redoes the interrupted evaluation.")
    print(f"EXTERNAL STOP at {job_id}; see RESUME.md", flush=True)
    sys.exit(77)


def gpu_used(groups):
    tot, by = LG.gpu_hours()
    return sum(v for k, v in by.items() if any(k.startswith(g) for g in groups)), tot


def job(job_id, stage, argv, gpu=True, expect="0", disk_gb=1.0, seed="", cap_groups=("gpucheck", "stage1")):
    used, tot = gpu_used(cap_groups)
    if gpu and (used >= P.GPU_CAP_H["stage1"] or tot >= P.GPU_CAP_H["total"]):
        LG.set_state("STOPPED_GPU_CAP", gpu_h=used)
        LG.write_resume("STOPPED_GPU_CAP", f"{PY} asrs1_run.py {CMD_NOW}", notes=f"Stage-1 GPU cap reached ({used:.2f} h)")
        raise SystemExit(f"GPU cap reached ({used:.2f} h)")
    free = guard.disk_free_bytes() / 2**30
    if free - disk_gb < P.DISK_RESERVE_GIB:
        LG.set_state("STOPPED_DISK", free_gib=free, need_gib=disk_gb)
        LG.write_resume("STOPPED_DISK", f"{PY} asrs1_run.py {CMD_NOW}",
                        notes=f"Disk shortfall: free {free:.2f} GiB, need 100 + {disk_gb:.2f} GiB.")
        raise SystemExit(f"disk shortfall {100 + disk_gb - free:.2f} GiB")
    LG.disk(job_id, disk_gb)
    st = json.loads((P.ART / "driver_state.json").read_text()) if (P.ART / "driver_state.json").exists() else {"jobs": {}}
    if st["jobs"].get(job_id, {}).get("status") == "COMPLETE":
        return 0
    LG.write_resume(f"RUNNING {job_id}", f"{PY} asrs1_run.py {CMD_NOW}",
                    notes="Running normally. If this process is gone without a final state, rerun the command above "
                          "(completed jobs are skipped; training resumes from the latest verified slot).")
    t0, t_iso = time.time(), guard.now()
    cmd = [PY, "asrs1_driver.py", "job", job_id, "--gpu" if gpu else "--cpu", "--expect-rc", expect, "--disk-gb",
           str(disk_gb), "--"] + argv
    rc = subprocess.run(cmd, cwd=guard.ROOT, env=os.environ.copy()).returncode
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
        LG.write_resume("FAILED", f"{PY} asrs1_run.py {CMD_NOW}", notes=f"Job {job_id} failed (rc {rc}); see job_logs.")
        raise SystemExit(f"job {job_id} failed rc={rc}")
    return j.get("rc")


def write_cfg(name, cfg):
    P.CFG_DIR.mkdir(parents=True, exist_ok=True)
    p = P.CFG_DIR / f"{name}.json"
    if p.exists():
        if json.loads(p.read_text()) != cfg:
            raise guard.GuardError(f"config {p} exists with different content")
        return p
    guard.atomic_write_json(p, cfg, mode=0o400)
    return p


def run_dir(name):
    c = sorted((guard.ROOT / "runs").glob(f"*_asrs1_{name}"))
    return c[-1] if c else None


def train(name, cfgp, stage, max_legs=6, cap_groups=("gpucheck", "stage1")):
    for leg in range(1, max_legs + 1):
        rd = run_dir(name)
        if rd is not None and json.loads((rd / "status.json").read_text())["status"] == "COMPLETE":
            return rd
        argv = ["asrs1_train.py", "--config", rel(cfgp)]
        if rd is not None:
            argv += ["--run-dir", rel(rd)]
        st = json.loads((P.ART / "driver_state.json").read_text()) if (P.ART / "driver_state.json").exists() else {"jobs": {}}
        k = leg
        while st["jobs"].get(f"train_{name}_leg{k}", {}).get("status") == "COMPLETE":
            k += 1
        job(f"train_{name}_leg{k}", stage, argv, expect="0,75", disk_gb=6.0, cap_groups=cap_groups)
    rd = run_dir(name)
    if json.loads((rd / "status.json").read_text())["status"] != "COMPLETE":
        raise guard.GuardError("too many training legs")
    return rd


# ------------------------------------------------------------------------------------------------ pre-freeze
def gpucheck():
    """1 cycle (base 3, spec 3, router 2; steps 318..326) on full main-stream rows: uninterrupted run A and run B with an
    external SIGINT inside the specialist phase + resume. Checks exit codes, phase isolation, speed, memory."""
    from csr import asrs1_cfg as C
    import asrs1_tests as T
    sched = {"start": 318, "end": 326, "n_cycles": 1, "base": 3, "spec": 3}
    out = {"time": guard.now()}
    for nm, extra in (("GPUCHK_A", []), ("GPUCHK_B", ["--test-signal-at-step", "323"])):
        cfg = C.build(nm, test=True, schedule=sched)
        cfg["verify_ckpt"] = True
        cp = write_cfg(nm, cfg)
        if run_dir(nm) is None or json.loads((run_dir(nm) / "status.json").read_text())["status"] != "COMPLETE":
            argv = ["asrs1_train.py", "--config", rel(cp)] + extra
            try:
                job(f"gpucheck_{nm}_1", "gpucheck", argv, expect="0,77" if extra else "0", disk_gb=6.0)
            except SystemExit:
                raise
            if extra:
                job(f"gpucheck_{nm}_2", "gpucheck", ["asrs1_train.py", "--config", rel(cp), "--run-dir", rel(run_dir(nm))],
                    disk_gb=6.0)
        rd = run_dir(nm)
        legs = sorted(rd.glob("leg_*.json"))
        out[nm] = {"run": rd.name, "legs": [json.loads(x.read_text()) for x in legs]}
    A, B = run_dir("GPUCHK_A"), run_dir("GPUCHK_B")
    H = T.hashes(A)
    order = [h["after"] for h in H]
    changed = lambda a, b: sorted(k for k in a["hash"] if a["hash"][k] != b["hash"][k])
    out["isolation"] = [{"after": c["after"], "changed": changed(p, c)} for p, c in zip(H[:-1], H[1:])]
    ex = {"base": ["base"], "refresh": [], "spec": ["ALTG.experts", "SEM.experts", "SHUF.experts"],
          "router": sorted(["ALTG.router"] + [f"router.{b}_s{r}" for b in ("SEM", "SHUF") for r in (0, 1, 2)])}
    out["isolation_ok"] = order == ["start", "base", "refresh", "spec", "router"] and all(
        x["changed"] == sorted(ex[x["after"]]) for x in out["isolation"])
    legsB = out["GPUCHK_B"]["legs"]
    out["sigint_ok"] = len(legsB) == 2 and legsB[0]["status"] == "EXTERNALLY_STOPPED" and legsB[1]["status"] == "COMPLETE"
    import torch
    sA, _ = T.final_slot(A)
    sB, _ = T.final_slot(B)
    diffs = {}
    for k in ("model",):
        d = max(float((sA[k][n].float() - sB[k][n].float()).abs().max()) for n in sA[k])
        diffs[k] = d
    for n in sA["banks"]:
        diffs[f"bank_{n}"] = max(float((sA["banks"][n][x].float() - sB["banks"][n][x].float()).abs().max()) for x in sA["banks"][n])
    out["A_vs_B_max_abs_param_diff"] = diffs
    out["teacher_cache_sha"] = {"A": sA["st"]["refreshed"]["1"]["sha256"], "B": sB["st"]["refreshed"]["1"]["sha256"]}
    out["refresh_stats_A"] = sA["st"]["refreshed"]["1"]["stats"]
    la = out["GPUCHK_A"]["legs"][0]
    out["timing"] = {"train_step_seconds": la["train_step_seconds"], "refresh_seconds": la["refresh_seconds"],
                     "ckpt_seconds": la["ckpt_seconds"], "steps": 8, "peak_alloc_gib": la["peak_alloc_gib"]}
    tps = la["train_step_seconds"] / 8
    out["projection_stage1_train_hours"] = (793 * tps + 10 * la["refresh_seconds"] * 48 / 5 + 31 * la["ckpt_seconds"] / 3) / 3600
    out["all_ok"] = bool(out["isolation_ok"] and out["sigint_ok"])
    p = P.ART / f"gpu_check_{guard.utc_stamp()}.json"
    guard.atomic_write_json(p, out, mode=0o400)
    T.drop([A, B])
    for rd in (A, B):
        for ms in rd.glob("ckpt_c*"):
            for f in ms.glob("*"):
                LG.deletion(rel(f), guard.sha256_file(f), f.stat().st_size, "asrs1 GPU-check scratch milestone")
            import shutil
            shutil.rmtree(ms)
    print(json.dumps({k: out[k] for k in ("isolation_ok", "sigint_ok", "A_vs_B_max_abs_param_diff", "timing",
                                          "projection_stage1_train_hours", "all_ok")}, default=str))
    return out


def freeze():
    if P.PREREG_SHA_FILE.exists():
        return P.check_prereg()
    h = guard.sha256_file(P.PREREG)
    P.PREREG.chmod(0o400)
    guard.atomic_write_bytes(P.PREREG_SHA_FILE, f"{h}  {P.PREREG.name}\n".encode())
    P.PREREG_SHA_FILE.chmod(0o400)
    return h


# ------------------------------------------------------------------------------------------------ Stage 1
def eval_ckpt(tag, fmt, ckpt, out, stage):
    if not (out / "summary_SEL.json").exists():
        job(f"eval_{tag}", stage, ["asrs1_eval.py", "--format", fmt, "--ckpt", ckpt, "--out-dir", rel(out)], disk_gb=1.0)
    job(f"analyze_{tag}", stage, ["asrs1_analyze.py", "ckpt", "--tag", tag, "--eval-dir", rel(out)], gpu=False, disk_gb=0.1)
    return json.loads((P.ART / "analysis" / f"{tag}.json").read_text())


def cycle_observation(run, c, res, ref, c00):
    import numpy as np
    rs = sorted([x for x in res["records"] if x["pair"] == "SEM" and not x["teacher_mode"]], key=lambda x: x["seed"])
    tm = [x for x in res["records"] if x["pair"] == "SEM" and x["teacher_mode"]]
    ordG = ref["ce_G"]
    r0 = [x for x in c00["records"] if x["pair"] == "SEM" and not x["teacher_mode"]]
    dIf = float(np.mean([x["I_SEM_minus_SHUF"] for x in r0]))
    ts = (res.get("teacher_stats") or {}).get(str(c), {})
    f = lambda v: f"{v:+.5f}"
    txt = (f"## asrs1 {run} cycle {c} (step {res['step']}; SEL; automatic per-cycle record)\n\n"
           f"**What changed:** one complete alternating cycle (base 32 steps -> teacher refresh -> specialist 32 -> router "
           f"{'16' if c in (4, 7, 10) else '15'}), prereg section 4; allowed by the frozen schedule (no tuning).\n"
           f"**Observed:** ALT-base {res['ce_base']:.5f}, ALT-G {res['ce_G']:.5f} (ordinary G at step 1111: {ordG:.5f}); "
           f"ASRS-SEM s0/s1/s2 vs ALT-G {', '.join(f(x['stu_minus_G']) for x in rs)}; vs ordinary G "
           f"{', '.join(f(x['ce_stu_SEM'] - ordG) for x in rs)}; exposure-matched dI {', '.join(f(x['I_SEM_minus_SHUF']) for x in rs)} "
           f"(frozen-stage mean {dIf:.5f}; retention {', '.join(f'{x['I_SEM_minus_SHUF'] / dIf:.2f}' for x in rs)}); "
           f"region-teacher-deployed dI {tm[0]['I_SEM_minus_SHUF']:+.5f}; positive specialists "
           f"{'/'.join(str(x['n_pos_specialists_SEM']) for x in rs)}; switch {', '.join(f'{x['switch_rate_SEM']:.3f}' for x in rs)}; "
           f"region-prior agreement {', '.join(f'{x['agree_region_prior_SEM']:.2f}' for x in rs)}; load_max "
           f"{max(x['path_SEM']['load_max'] for x in rs):.3f}, eff {min(x['path_SEM']['load_eff'] for x in rs):.1f}, dead "
           f"{max(x['path_SEM']['n_dead'] for x in rs)}; teacher CE vs ALT-G (region teacher) {rs[0]['ce_T_SEM'] - res['ce_G']:+.5f}; "
           f"post-switch vs ALT-G {', '.join(f(x['post_switch']['stu_minus_G']) for x in rs)}; stable vs ALT-G "
           f"{', '.join(f(x['stable']['stu_minus_G']) for x in rs)}.\n"
           f"**Teacher refresh (training rows):** SEM override share {ts.get('SEM', {}).get('override_frac_spec', float('nan')):.3f}, "
           f"s_u {ts.get('SEM', {}).get('s_u', float('nan')):.4f}, teacher load_max {ts.get('SEM', {}).get('teacher_load_max', float('nan')):.3f}; "
           f"headroom ALT-G - ALT-base {res['ce_G'] - res['ce_base']:+.5f}.\n")
    LG.observe(txt)


def stage1(gen=0, change=None):
    from csr import asrs1_cfg as C
    P.check_prereg()
    sfx = f"_g{gen}" if gen else ""
    name = f"S1{sfx}"
    if (P.ART / f"stage1{sfx}.json").exists():
        return json.loads((P.ART / f"stage1{sfx}.json").read_text())
    LG.set_state(f"STAGE1{sfx}_RUNNING")
    c00 = eval_ckpt("c00", "cfar1", P.CKPT, P.EVALS / "c00", "stage1")
    ref = eval_ckpt("ref_ccsr1_m1111", "ccsr1", P.CCSR1_S3, P.EVALS / "ref_ccsr1_m1111", "stage1")
    kw = {}
    if change == "phase_ratio_0.8":
        kw["schedule"] = {"start": P.START, "end": P.END, "n_cycles": P.N_CYCLES, "base": 63, "spec": 11}
    elif change == "router_lr_x1/3":
        kw["router_opt"] = {**P.ROUTER_OPT, "lr": P.ROUTER_OPT["lr"] / 3}
    elif change == "c=4":
        kw["teacher"] = {**P.TEACHER, "c": 4.0}
    cfg = C.build(name, gen=gen, change=change, **kw)
    if change == "spec_lr_x0.5":
        cfg["spec_lr_mult"] = 0.5
    if change == "refresh_before_router":
        raise guard.GuardError("refresh_before_router needs a logged trainer change before it can run")
    cp = write_cfg(name, cfg)
    rd = train(name, cp, f"stage1{sfx}")
    run = rd.name
    st = json.loads((rd / "status.json").read_text())
    LG.observe(f"asrs1 {name} training COMPLETE: {st['seen']} tokens, n_done {st['n_done']}, train step s "
               f"{st['train_step_seconds']:.0f}, refresh s {st['refresh_seconds']:.0f}, peak {st['peak_alloc_gib']:.1f} GiB; "
               f"teacher caches {list(st['refreshed'])}")
    for c in range(1, P.N_CYCLES + 1):
        ck = st["ckpts"][f"c{c:02d}"]["path"]
        res = eval_ckpt(f"{run}_c{c:02d}", "asrs1", ck, P.EVALS / run / f"c{c:02d}", f"stage1{sfx}")
        cycle_observation(run, c, res, ref, c00)
    job(f"analyze_stage1{sfx}", f"stage1{sfx}", ["asrs1_analyze.py", "stage1", "--run", run, "--gen", str(gen)],
        gpu=False, disk_gb=0.1)
    return json.loads((P.ART / f"stage1{sfx}.json").read_text())


def final():
    tot, by = LG.gpu_hours()
    s1 = {p.name: json.loads(p.read_text()) for p in sorted(P.ART.glob("stage1*.json"))}
    job("inventory_verify", "final", ["asrs1_inventory.py", "verify"], gpu=False, disk_gb=0.05)
    inv = sorted(P.ART.glob("protected_inventory_verify_*.json"))
    out = {"time": guard.now(), "experiment": P.EXP, "prereg_sha256": P.check_prereg(), "gpu_hours": tot, "gpu_by_stage": by,
           "stage1": {k: {kk: v[kk] for kk in ("stage1_pass", "criteria", "generation_rule", "identity_check", "dI_frozen",
                                                "references", "trajectory")} for k, v in s1.items()},
           "protected_inventory": json.loads(inv[-1].read_text()) if inv else None,
           "test_split": "never loaded", "validation": "never opened", "calib_selection": "never opened (Stage 2 not reached)"}
    guard.atomic_write_json(P.ART / "final_analysis.json", out, overwrite=True)
    return out


def main():
    global CMD_NOW
    what = sys.argv[1]
    CMD_NOW = " ".join(sys.argv[1:])
    guard.check_root()
    P.TMP.mkdir(parents=True, exist_ok=True)
    P.ART.mkdir(parents=True, exist_ok=True)
    if what == "inventory":
        job("inventory_before", "setup", ["asrs1_inventory.py", "before"], gpu=False, disk_gb=0.05)
    elif what == "tests":
        job("cpu_tests", "setup", ["asrs1_tests.py"], gpu=False, disk_gb=3.0)
    elif what == "gpucheck":
        gpucheck()
    elif what == "freeze":
        print(freeze())
    elif what == "stage1":
        gen = int(os.environ.get("ASRS1_GEN", "0"))
        change = os.environ.get("ASRS1_CHANGE") or None
        s1 = stage1(gen, change)
        nxt = s1["generation_rule"].get("generation")
        if s1["stage1_pass"]:
            LG.set_state("STAGE1_PASS_AWAITING_STAGE2_AMENDMENT")
        elif nxt and gen < 2:
            LG.set_state(f"STAGE1{'_g' + str(gen) if gen else ''}_FAIL_GENERATION_PENDING", next=nxt)
        else:
            LG.set_state("TERMINAL_STAGE1_FAIL", gen=gen)
        LG.write_resume(json.loads((P.ART / "STATE.json").read_text())["state"], "true  # see DECISION_LOG / report")
    elif what == "final":
        final()


if __name__ == "__main__":
    main()
