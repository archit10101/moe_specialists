#!/usr/bin/env python
"""joint-semantic-residual-pretrain-v1 orchestrator. Every GPU job goes through jsr_driver.py (scripts/gpu_guarded.sh).

  jsr_run.py stage1            Stage-1 search run (seed 100, 20M) + SEL eval + selection (+ Stage-1b alpha if required)
  jsr_run.py stage2            Stage-2 runs (seeds 0, 1 to 50M) + EVAL evals + promotion
  jsr_run.py stage3            Stage-3 (seeds 0, 1 continue; 2, 3 new; 245M) + VAL evals + reference evals + analysis
  jsr_run.py all               stage1 -> stage2 -> stage3 as far as the rules allow

External interruption (driver exit 77): the orchestrator stops, writes STOPPED_BY_OPERATOR + resume command to RESUME.md
and never restarts by itself."""
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402
from csr import jsr_log as LG  # noqa: E402
from csr.jsr_paths import ART, CFG_DIR, EVALS, GPU_CAP_H, RUN_PREFIX, STAGE2_SEEDS, STAGE3_SEEDS  # noqa: E402

PY = str(guard.ROOT / ".venv/bin/python")
CMD_NOW = ""


def stop_external(job):
    LG.set_state("STOPPED_BY_OPERATOR", job=job)
    LG.write_resume("STOPPED_BY_OPERATOR", f"{PY} jsr_driver.py authorize-resume && nohup {PY} jsr_run.py {CMD_NOW} &",
                    notes=f"Interrupted job: `{job}`. Treated as intentional GPU reclamation; nothing restarts "
                          f"automatically. The trainer saved a verified slot at the last step boundary; rerunning the "
                          f"same phase resumes from it.")
    print(f"EXTERNAL STOP at {job}; see RESUME.md", flush=True)
    sys.exit(77)


def job(job_id, stage, argv, gpu=True, expect="0", disk_gb=3.0, seed=""):
    tot, _ = LG.gpu_hours()
    if gpu and tot >= GPU_CAP_H["total"]:
        LG.set_state("STOPPED_GPU_CAP", gpu_h=tot)
        raise SystemExit(f"GPU cap reached ({tot:.2f} h)")
    free = guard.disk_free_bytes() / 2**30
    if free - disk_gb < 100.0:
        LG.set_state("STOPPED_DISK", free_gib=free, need_gib=disk_gb)
        LG.write_resume("STOPPED_DISK", f"{PY} jsr_run.py {CMD_NOW}",
                        notes=f"Disk shortfall: free {free:.2f} GiB, need 100 + {disk_gb:.2f} GiB "
                              f"(short by {100 + disk_gb - free:.2f} GiB).")
        raise SystemExit(f"disk shortfall {100 + disk_gb - free:.2f} GiB")
    LG.disk(job_id, disk_gb)
    st = json.loads((ART / "driver_state.json").read_text()) if (ART / "driver_state.json").exists() else {"jobs": {}}
    if st["jobs"].get(job_id, {}).get("status") == "COMPLETE":
        return 0
    t0, t_iso = time.time(), guard.now()
    cmd = [PY, "jsr_driver.py", "job", job_id, "--gpu" if gpu else "--cpu", "--expect-rc", expect, "--disk-gb",
           str(disk_gb), "--"] + argv
    rc = subprocess.run(cmd, cwd=guard.ROOT).returncode
    dt = time.time() - t0
    st = json.loads((ART / "driver_state.json").read_text())
    j = st["jobs"].get(job_id, {})
    if gpu:
        LG.compute(job_id, stage, t_iso, guard.now(), dt, j.get("status", "?"))
    LG.manifest({"time": guard.now(), "job": job_id, "stage": stage, "seed": seed, "status": j.get("status"),
                 "rc": j.get("rc"), "seconds": round(dt, 1), "gpu_h": round(dt / 3600, 4) if gpu else 0,
                 "cmd": " ".join(argv)})
    if rc == 77:
        stop_external(job_id)
    if rc != 0:
        LG.set_state("FAILED", job=job_id, rc=rc)
        LG.write_resume("FAILED", f"{PY} jsr_run.py {CMD_NOW}", notes=f"Job {job_id} failed (rc {rc}); see job_logs.")
        raise SystemExit(f"job {job_id} failed rc={rc}")
    return j.get("rc")


def run_dir_of(name):
    c = sorted((guard.ROOT / "runs").glob(f"*_{RUN_PREFIX}{name}"))
    return c[-1] if c else None


def train(cfg_name, stage, seed, max_steps=None):
    """Train (in legs) until COMPLETE or max-steps reached. Returns the run dir."""
    leg = 1
    while True:
        rd = run_dir_of(cfg_name)
        if rd is not None:
            stj = json.loads((rd / "status.json").read_text())
            if stj["status"] == "COMPLETE":
                return rd
            if max_steps is not None and stj.get("steps_done", 0) >= max_steps and stj["status"] == "INTERRUPTED":
                return rd
        argv = ["jsr_train.py", "--config", str((CFG_DIR / f"{cfg_name}.json").relative_to(guard.ROOT))]
        if rd is not None:
            argv += ["--run-dir", str(rd.relative_to(guard.ROOT))]
        if max_steps is not None:
            argv += ["--max-steps", str(max_steps)]
        tag = f"_to{max_steps}" if max_steps else "_full"
        while True:
            jid = f"train_{cfg_name}{tag}_leg{leg}"
            st = json.loads((ART / "driver_state.json").read_text()) if (ART / "driver_state.json").exists() else {"jobs": {}}
            if st["jobs"].get(jid, {}).get("status") == "COMPLETE":
                leg += 1
                continue
            break
        job(jid, stage, argv, expect="0,75", disk_gb=4.0, seed=seed)
        leg += 1


def evaluate(ckpt_dir, part, out_name, stage, seed, disk_gb):
    out = EVALS / f"{out_name}.npz"
    if out.exists():
        return out
    job(f"eval_{out_name}", stage, ["jsr_eval.py", "--ckpt", str(Path(ckpt_dir).relative_to(guard.ROOT)), "--part", part,
                                    "--out", str(out.relative_to(guard.ROOT))], disk_gb=disk_gb, seed=seed)
    return out


def cpu(job_id, stage, argv):
    job(job_id, stage, argv, gpu=False, disk_gb=1.0)


def read(name):
    return json.loads((ART / f"{name}.json").read_text())


def configs(*args):
    out = subprocess.run([PY, "jsr_configs.py", *args], cwd=guard.ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(out.stderr)
    return Path(out.stdout.strip().splitlines()[-1]).stem


def stage1():
    from csr.jsr_paths import ALPHA, ALPHA_FALLBACK
    tb = json.loads((ART / "tables/tables.json").read_text())
    n1 = tb["budget_steps"]["S1"]
    if (ART / "stage1_decision.json").exists():
        return read("stage1_decision")
    LG.set_state("STAGE1_RUNNING")
    name = configs("stage1", "--alpha", str(ALPHA))
    rd = train(name, "stage1", 100, max_steps=n1)
    ev = evaluate(rd / "ckpt_m20", "SEL", f"S1_a{ALPHA}_SEL", "stage1", 100, disk_gb=5.0)
    cpu("analyze_stage1", "stage1", ["jsr_analyze.py", "stage1", "--eval", str(ev.relative_to(guard.ROOT)), "--run",
                                     str(rd.relative_to(guard.ROOT)), "--out", "stage1"])
    s1 = read("stage1")
    dec = {"stage1": "stage1.json", "selected": s1["selected"], "cfg": s1["selected_cfg"], "alpha": ALPHA,
           "stage1_pass": s1["stage1_pass"]}
    if s1["stage1_pass"] and s1.get("needs_stage1b_alpha"):
        c = s1["selected_cfg"]
        name_b = configs("stage1", "--alpha", str(ALPHA_FALLBACK), "--taus", str(c["tau"]), "--rs", str(c["r"]))
        rdb = train(name_b, "stage1", 100, max_steps=n1)
        evb = evaluate(rdb / "ckpt_m20", "SEL", f"S1b_a{ALPHA_FALLBACK}_SEL", "stage1", 100, disk_gb=2.0)
        cpu("analyze_stage1b", "stage1", ["jsr_analyze.py", "stage1", "--eval", str(evb.relative_to(guard.ROOT)), "--run",
                                          str(rdb.relative_to(guard.ROOT)), "--out", "stage1b"])
        s1b = read("stage1b")
        sel_b = f"D_r{c['r']}_t{c['tau']}"
        if not s1b["candidates"][sel_b]["R2_collapse"]:
            dec["alpha"] = ALPHA_FALLBACK
        dec["stage1b"] = {"file": "stage1b.json", "R2_pass_at_0.03": not s1b["candidates"][sel_b]["R2_collapse"]}
    dec["time"] = guard.now()
    guard.atomic_write_json(ART / "stage1_decision.json", dec, mode=0o400)
    LG.decision(evidence=json.dumps({k: {kk: s1["candidates"][k][kk] for kk in ("rejected", "R1_ce", "R2_collapse",
                                                                                 "R3_domination", "I_D_minus_I_B", "ce_minus_B")}
                                     for k in s1["candidates"]}),
                competing="selection noise on one seed at 20M; selection is confirmed on fresh seeds (0, 1) and a different partition (EVAL) in Stage 2",
                decision_=f"Stage 1 {'PASS' if dec['stage1_pass'] else 'FAIL'}; selected {dec['selected']} alpha {dec['alpha']}",
                rationale="preregistered rejection rules R1-R3, then max I_D - I_B(tau), ties to smaller r and tau 1.0",
                falsifier="Stage 2 P2/P3 failing on fresh seeds would show the selection did not generalise", stage="stage1")
    return dec


def stage2(dec):
    tb = json.loads((ART / "tables/tables.json").read_text())
    n2 = tb["budget_steps"]["S2"]
    if (ART / "stage2.json").exists():
        return read("stage2")
    LG.set_state("STAGE2_RUNNING")
    c = dec["cfg"]
    evs = []
    for s in STAGE2_SEEDS:
        name = configs("final", "--seed", str(s), "--r", str(c["r"]), "--tau", str(c["tau"]), "--alpha", str(dec["alpha"]))
        rd = train(name, "stage2", s, max_steps=n2)
        evs.append(evaluate(rd / "ckpt_m50", "EVAL", f"S2_s{s}_EVAL", "stage2", s, disk_gb=3.0))
    cpu("analyze_stage2", "stage2", ["jsr_analyze.py", "stage2", "--evals",
                                     ",".join(str(e.relative_to(guard.ROOT)) for e in evs), "--out", "stage2"])
    s2 = read("stage2")
    LG.decision(evidence=json.dumps(s2["criteria"]), competing="seed noise with 2 seeds; EVAL partition only 498 papers",
                decision_=f"Stage 2 promote={s2['promote']}", rationale="preregistered P1-P4, all required",
                falsifier="Stage 3 VAL criteria", stage="stage2")
    return s2


def stage3(dec):
    if (ART / "stage3.json").exists():
        return read("stage3")
    LG.set_state("STAGE3_RUNNING")
    c = dec["cfg"]
    evs = []
    from csr.jsr_paths import REUSED
    for s in STAGE3_SEEDS:
        name = configs("final", "--seed", str(s), "--r", str(c["r"]), "--tau", str(c["tau"]), "--alpha", str(dec["alpha"]))
        rd = train(name, "stage3", s)
        evs.append(evaluate(rd / "ckpt_p100", "VAL", f"S3_s{s}_VAL", "stage3", s, disk_gb=6.0))
    refs = {}
    for kind in ("T4", "EDL", "C6"):
        refs[kind] = []
        for s in STAGE3_SEEDS:
            out = EVALS / f"REF_{kind}_s{s}_VAL.npz"
            if not out.exists():
                job(f"eval_REF_{kind}_s{s}_VAL", "evals", ["jsr_eval.py", "--ref", REUSED[kind][s], "--part", "VAL",
                                                          "--out", str(out.relative_to(guard.ROOT))], disk_gb=1.0, seed=s)
            refs[kind].append(str(out))
    guard.atomic_write_json(ART / "refs_VAL.json", refs, overwrite=True)
    cpu("analyze_stage3", "stage3", ["jsr_analyze.py", "stage3", "--evals",
                                     ",".join(str(e.relative_to(guard.ROOT)) for e in evs), "--refs",
                                     str((ART / "refs_VAL.json").relative_to(guard.ROOT)), "--out", "stage3"])
    s3 = read("stage3")
    LG.decision(evidence=json.dumps({"criteria": s3["criteria"], "outcomes": s3["outcomes"]}),
                competing="see report", decision_=f"terminal: {s3['outcomes']['primary']}",
                rationale="preregistered S1-S4 and interpretation table, evaluated once on VAL",
                falsifier="n/a (terminal)", stage="stage3")
    return s3


def main():
    global CMD_NOW
    guard.check_root()
    from csr.jsr_paths import check_prereg
    check_prereg()
    what = sys.argv[1]
    CMD_NOW = what
    if what in ("stage1", "all"):
        dec = stage1()
        if not dec["stage1_pass"]:
            LG.set_state("STAGE1_FAIL")
            LG.write_resume("STAGE1_FAIL", "true  # engineering generation required (protocol 12)")
            return
        if what == "stage1":
            return
    dec = read("stage1_decision")
    if what in ("stage2", "all"):
        s2 = stage2(dec)
        if not s2["promote"]:
            LG.set_state("STAGE2_NOT_PROMOTED")
            LG.write_resume("STAGE2_NOT_PROMOTED", "true  # engineering generation or terminal (protocol 12)")
            return
        if what == "stage2":
            return
    if what in ("stage3", "all"):
        s3 = stage3(dec)
        LG.set_state(f"TERMINAL_{s3['outcomes']['primary']}")
        LG.write_resume(f"TERMINAL_{s3['outcomes']['primary']}", "true  # nothing to resume")


if __name__ == "__main__":
    main()
