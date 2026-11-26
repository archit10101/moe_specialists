#!/usr/bin/env python
"""subject-conditioned-utility-router-v1 (scur1) orchestrator. Every job goes through scur1_driver.py (GPU jobs through
scripts/gpu_guarded.sh).

  scur1_run.py inventory        protected inventory (before) + reuse manifest
  scur1_run.py tests            CPU tests (scur1_tests.py)
  scur1_run.py freeze           write the prereg sha256 sidecar (0400)
  scur1_run.py stageA           Stage A generation (env SCUR1_GEN = 0..4; SCUR1_CHANGE = JSON overrides of the candidate,
                                logged in DECISION_LOG before the run): TRAIN extraction (once), routers (candidate +
                                controls, router seed 0) on frozen m1111 banks, SEL analysis, criteria
  scur1_run.py final            final_analysis.json + protected inventory verification
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
from csr import scur1_log as LG  # noqa: E402
from csr import scur1_paths as P  # noqa: E402

PY = str(guard.ROOT / ".venv/bin/python")
CMD_NOW = ""
SEL_DIR = {m: f"{P.A2}/evals/20261122T053544Z_asrs2_S1/{m}" for m in P.MILESTONES}

BASE_W = {"subject": 1.0, "utility": 1.0, "regret": 1.0, "lm": 1.0, "balance": 0.01}
CAND0 = {"tag": "SCU", "mode": "scu", "bank": "SEM", "w": BASE_W, "tau": 1.0, "rho": 1.0, "feats": "base", "A": "soft"}


def controls():
    return [{"tag": "SUBJ", "mode": "subject", "bank": "SEM", "w": {"subject": 1.0}, "feats": "base"},
            {"tag": "SUBJ_SHUF", "mode": "subject", "bank": "SHUF", "w": {"subject": 1.0}, "feats": "base", "twin_of": "SUBJ"},
            {"tag": "UTIL", "mode": "utility", "bank": "SEM", "w": {"lm": 1.0, "balance": 0.01}, "feats": "base"},
            {"tag": "UTIL_SHUF", "mode": "utility", "bank": "SHUF", "w": {"lm": 1.0, "balance": 0.01}, "feats": "base",
             "twin_of": "UTIL"}]


def candidate(gen, change):
    c = json.loads(json.dumps(CAND0))
    for g in range(1, gen + 1):
        ch = change if g == gen else json.loads((P.ART / f"gen{g}_change.json").read_text())["change"]
        for k, v in ch.items():
            if k == "w":
                c["w"] = {**c["w"], **v}
            else:
                c[k] = v
    c["tag"] = f"SCU_g{gen}"
    tw = json.loads(json.dumps(c))
    tw.update({"tag": f"SCU_g{gen}_SHUF", "bank": "SHUF", "twin_of": c["tag"]})
    return c, tw


def rel(p):
    return str(Path(p).relative_to(guard.ROOT))


def stop_external(job_id):
    LG.set_state("INCOMPLETE_STOPPED_BY_EXTERNAL_USER", job=job_id)
    LG.write_resume("INCOMPLETE_STOPPED_BY_EXTERNAL_USER",
                    f"{PY} scur1_driver.py authorize-resume && nohup {PY} scur1_run.py {CMD_NOW} > "
                    f"{rel(P.ART)}/run_{'_'.join(CMD_NOW.split())}.log 2>&1 &",
                    notes=f"Interrupted job: `{job_id}`. Treated as another user needing the GPU; nothing restarts "
                          f"automatically; all scur1 launchers/watchers stopped. Wait for explicit user permission. "
                          f"Rerunning the phase skips completed jobs and routers.")
    print(f"EXTERNAL STOP at {job_id}; see RESUME.md", flush=True)
    sys.exit(77)


def job(job_id, stage, argv, gpu=True, expect="0", disk_gb=1.0, seed=""):
    tot, by = LG.gpu_hours()
    grp = "stageB" if stage.startswith("stageB") else "stageA"
    used = sum(v for k, v in by.items() if k.startswith(grp))
    if gpu and (used >= P.GPU_CAP_H[grp] or tot >= P.GPU_CAP_H["total"]):
        LG.set_state("STOPPED_GPU_CAP", gpu_h=used, group=grp)
        LG.write_resume("STOPPED_GPU_CAP", f"{PY} scur1_run.py {CMD_NOW}", notes=f"{grp} GPU cap reached ({used:.2f} h)")
        raise SystemExit(f"GPU cap reached ({used:.2f} h)")
    free = guard.disk_free_bytes() / 2**30
    if free - disk_gb < P.DISK_RESERVE_GIB:
        LG.set_state("STOPPED_DISK", free_gib=free, need_gib=disk_gb)
        LG.write_resume("STOPPED_DISK", f"{PY} scur1_run.py {CMD_NOW}",
                        notes=f"Disk shortfall: free {free:.2f} GiB, need 100 + {disk_gb:.2f} GiB.")
        raise SystemExit(f"disk shortfall {100 + disk_gb - free:.2f} GiB")
    LG.disk(job_id, disk_gb)
    st = json.loads((P.ART / "driver_state.json").read_text()) if (P.ART / "driver_state.json").exists() else {"jobs": {}}
    if st["jobs"].get(job_id, {}).get("status") == "COMPLETE":
        return 0
    LG.write_resume(f"RUNNING {job_id}", f"{PY} scur1_run.py {CMD_NOW}",
                    notes="Running normally. If this process is gone without a final state, rerun the command above "
                          "(completed jobs and routers are skipped).")
    t0, t_iso = time.time(), guard.now()
    cmd = [PY, "scur1_driver.py", "job", job_id, "--gpu" if gpu else "--cpu", "--expect-rc", expect, "--disk-gb",
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
        LG.write_resume("FAILED", f"{PY} scur1_run.py {CMD_NOW}", notes=f"Job {job_id} failed (rc {rc}); see job_logs.")
        raise SystemExit(f"job {job_id} failed rc={rc}")
    return j.get("rc")


def write_cfg(name, obj):
    P.CFG_DIR.mkdir(parents=True, exist_ok=True)
    p = P.CFG_DIR / f"{name}.json"
    if p.exists():
        if json.loads(p.read_text()) != obj:
            raise guard.GuardError(f"config {p} exists with different content")
        return p
    guard.atomic_write_json(p, obj, mode=0o400)
    return p


def freeze():
    if P.PREREG_SHA_FILE.exists():
        return P.check_prereg()
    h = guard.sha256_file(P.PREREG)
    P.PREREG.chmod(0o400)
    guard.atomic_write_bytes(P.PREREG_SHA_FILE, f"{h}  {P.PREREG.name}\n".encode())
    P.PREREG_SHA_FILE.chmod(0o400)
    return h


def extract_train(m):
    out = P.TABLES / f"TRAIN_{m}"
    if not (out / "summary_TRAIN.json").exists():
        path, _ = P.MILESTONES[m]
        job(f"extract_TRAIN_{m}", "stageA", ["scur1_extract.py", "--ckpt", path, "--part", "TRAIN", "--out-dir", rel(out),
                                              "--banks", f"{P.SEM_BANK},{P.SHUF_BANK},{P.G_BANK},{P.ALTG_BANK}"], disk_gb=1.0)
    return out


def stageA(gen, change):
    import scur1_analyze as SA
    P.check_prereg()
    outp = P.ART / f"stageA_gen{gen}.json"
    if outp.exists():
        return json.loads(outp.read_text())
    if gen > 0:
        if gen > P.MAX_GENERATIONS:
            raise guard.GuardError("generation limit reached")
        if not (P.ART / f"gen{gen}_change.json").exists():
            guard.atomic_write_json(P.ART / f"gen{gen}_change.json", {"time": guard.now(), "change": change}, mode=0o400)
        change = json.loads((P.ART / f"gen{gen}_change.json").read_text())["change"]
    LG.set_state(f"STAGEA_GEN{gen}_RUNNING")
    m = P.PRIMARY
    tr = extract_train(m)
    cand, twin = candidate(gen, change)
    cfgs = [cand, twin] + controls()
    cp = write_cfg(f"stageA_gen{gen}", cfgs)
    stg = f"A_gen{gen}"
    job(f"routers_{stg}_s0", "stageA", ["scur1_router_job.py", "--milestone", m, "--train", rel(tr), "--eval", SEL_DIR[m],
                                        "--eval-part", "SEL", "--configs", rel(cp), "--seed", "0", "--stage", stg],
        disk_gb=1.5)
    job(f"analyze_{stg}", "stageA", ["scur1_analyze.py", "stage", "--stage", stg, "--eval-dir", SEL_DIR[m], "--part", "SEL"],
        gpu=False, disk_gb=0.05)
    A = json.loads((P.ANALYSIS / f"{stg}.json").read_text())
    crit = SA.criteria(A, cand["tag"], 0)
    res = {"time": guard.now(), "gen": gen, "change": change, "candidate": cand, "criteria": crit, "pass": crit["pass"]}
    guard.atomic_write_json(outp, res, mode=0o400)
    observe_stage(A, cand["tag"], gen, crit)
    return res


def observe_stage(A, ctag, gen, crit):
    f = lambda x: "n/a" if x is None else f"{x:+.5f}"
    lines = []
    for n, R in A["routers"].items():
        C = R["contrasts"]
        lines.append(f"| {n} | {R['ce']:.5f} | {f(C['ALTG']['v'])} [{C['ALTG']['ci'][0]:+.5f}, {C['ALTG']['ci'][1]:+.5f}] | "
                     f"{f(C['G']['v'])} | {f(C['R1_existing']['v'])} | "
                     f"{'n/a' if R['closure_teacher_vs_ALTG'] is None else round(R['closure_teacher_vs_ALTG'], 3)} | "
                     f"{R['closure_oracle_vs_ALTG']:.4f} | {R['routing']['agree_teacher']:.3f} | {R['routing']['switch']:.4f} | "
                     f"{R.get('subject', {}).get('acc', float('nan')):.3f} / {R.get('subject', {}).get('acc_first32', float('nan')):.3f} / "
                     f"{R.get('subject', {}).get('acc_post_switch', float('nan')):.3f} | {f(R['by_mask_vs_ALTG']['first32'])} / "
                     f"{f(R['by_mask_vs_ALTG']['post_switch'])} / {f(R['by_mask_vs_ALTG']['stable'])} |")
    fun = "; ".join(f"{k}: dI {v['dI']:+.5f} [{v['dI_ci'][0]:+.5f}, {v['dI_ci'][1]:+.5f}], pos {v['per_specialist']['n_pos']}/5, "
                    f"SEM-SHUF CE {v['sem_minus_shuf_ce']['v']:+.5f}" for k, v in A["functional"].items())
    gd = A["routers"].get(f"{ctag}_s0", {}).get("gap_decomposition")
    LG.observe(f"## Stage A generation {gen} (SEL, m1111 frozen banks, router seed 0) — candidate {ctag}: "
               f"{'PASS' if crit['pass'] else 'FAIL ' + str(crit['failed'])}\n\n"
               f"Reference CE: base {A['ce_mean']['base']:.5f}, G {A['ce_mean']['G']:.5f}, ALT-G {A['ce_mean']['ALTG']:.5f}, "
               f"existing R1 router {A['ce_mean']['R1_existing']:.5f}, region teacher {A['ce_mean']['T_SEM']:.5f} "
               f"(T - ALT-G {A['teacher_gain']['T_SEM-ALTG']:+.5f} {A['teacher_gain_ci_vs_ALTG']}), subject-mapping oracle "
               f"{A['ce_mean']['MAPORACLE_SEM']:.5f} (map {A['mapping_oracle_sem']}), per-token oracle {A['ce_mean']['O_SEM']:.5f}.\n\n"
               f"| router | CE | - ALT-G [CI] | - G | - R1 existing | closure teacher (ALT-G) | closure oracle | teacher agree | switch | "
               f"subject acc all / first32 / post-switch | vs ALT-G first32 / post / stable |\n|" + "---|" * 11 + "\n"
               + "\n".join(lines) + f"\n\nFunctional: {fun}.\nGap decomposition ({ctag}): {json.dumps({k: round(v, 5) for k, v in (gd or {}).items()})}.\n"
               f"U_all ({ctag}): {json.dumps([[round(x, 4) for x in r] for r in A['routers'].get(f'{ctag}_s0', {}).get('U_all', [])])}\n")


def final():
    tot, by = LG.gpu_hours()
    job("inventory_verify", "final", ["scur1_inventory.py", "verify"], gpu=False, disk_gb=0.05)
    inv = sorted(P.ART.glob("protected_inventory_verify_*.json"))
    out = {"time": guard.now(), "experiment": P.EXP, "prereg_sha256": P.check_prereg(), "gpu_hours": tot, "gpu_by_stage": by,
           "stageA": {p.name: json.loads(p.read_text()) for p in sorted(P.ART.glob("stageA_gen*.json"))},
           "protected_inventory": json.loads(inv[-1].read_text()) if inv else None,
           "calib_selection": "opened" if (P.ART / "FROZEN_A.json").exists() else "never opened",
           "validation": "never opened", "test_split": "never loaded"}
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
        job("inventory_before", "setup", ["scur1_inventory.py", "before"], gpu=False, disk_gb=0.05)
    elif what == "tests":
        job("cpu_tests", "setup", ["scur1_tests.py"], gpu=False, disk_gb=1.0)
    elif what == "freeze":
        print(freeze())
    elif what == "stageA":
        gen = int(os.environ.get("SCUR1_GEN", "0"))
        change = json.loads(os.environ.get("SCUR1_CHANGE", "{}"))
        r = stageA(gen, change)
        LG.set_state(f"STAGEA_GEN{gen}_{'PASS' if r['pass'] else 'FAIL'}_AWAITING_DECISION")
        LG.write_resume(json.loads((P.ART / "STATE.json").read_text())["state"], "true  # see DECISION_LOG")
    elif what == "final":
        final()


if __name__ == "__main__":
    main()
