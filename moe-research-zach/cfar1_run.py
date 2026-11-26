#!/usr/bin/env python
"""csr-centered-fedavg-residual-v1 (cfar1) orchestrator (copy of csur1_run.py). Every GPU job goes through cfar1_driver.py
(scripts/gpu_guarded.sh).

  cfar1_run.py gputest      GPU checks (G-EQ, G-ISO, G-RESUME, G-BENCH, G-EVAL) on test configs (no prereg needed)
  cfar1_run.py stage1       Stage-1 search (seed 100, 20M) + SEL eval + selection
  cfar1_run.py stage2       Stage-2 (seeds 100, 101 to 50M) + EVAL evals + promotion
  cfar1_run.py final        freeze -> VAL p_sem -> 245M x 4 seeds -> VAL evals -> reference evals -> final analysis
  cfar1_run.py all          stage1 -> stage2 -> final as far as the frozen rules allow

External interruption (driver exit 77): the orchestrator stops, writes STOPPED_BY_OPERATOR + the resume command to
RESUME.md and never restarts by itself."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402
from csr import cfar1_log as LG  # noqa: E402
from csr.cfar1_paths import (ART, CFG_DIR, EVALS, FINAL_SEEDS, GPU_CAP_H, RUN_PREFIX, STAGE2_SEEDS, TMP)  # noqa: E402

PY = str(guard.ROOT / ".venv/bin/python")
CMD_NOW = ""
GEN = int(os.environ.get("CFAR1_GEN", "0"))
SFX = f"_g{GEN}" if GEN else ""


def stop_external(job_id):
    LG.set_state("STOPPED_BY_OPERATOR", job=job_id)
    LG.write_resume("STOPPED_BY_OPERATOR", f"{PY} cfar1_driver.py authorize-resume && nohup {PY} cfar1_run.py {CMD_NOW} &",
                    notes=f"Interrupted job: `{job_id}`. Treated as intentional GPU reclamation; nothing restarts "
                          f"automatically. Trainers save a verified slot at the last step boundary; rerunning the same "
                          f"phase resumes from it.")
    print(f"EXTERNAL STOP at {job_id}; see RESUME.md", flush=True)
    sys.exit(77)


def job(job_id, stage, argv, gpu=True, expect="0", disk_gb=3.0, seed="", env=None):
    tot, _ = LG.gpu_hours()
    if gpu and tot >= GPU_CAP_H["total"]:
        LG.set_state("STOPPED_GPU_CAP", gpu_h=tot)
        LG.write_resume("STOPPED_GPU_CAP", f"{PY} cfar1_run.py {CMD_NOW}", notes=f"GPU cap {GPU_CAP_H['total']} h reached")
        raise SystemExit(f"GPU cap reached ({tot:.2f} h)")
    free = guard.disk_free_bytes() / 2**30
    if free - disk_gb < 100.0:
        LG.set_state("STOPPED_DISK", free_gib=free, need_gib=disk_gb)
        LG.write_resume("STOPPED_DISK", f"{PY} cfar1_run.py {CMD_NOW}",
                        notes=f"Disk shortfall: free {free:.2f} GiB, need 100 + {disk_gb:.2f} GiB "
                              f"(short by {100 + disk_gb - free:.2f} GiB).")
        raise SystemExit(f"disk shortfall {100 + disk_gb - free:.2f} GiB")
    LG.disk(job_id, disk_gb)
    st = json.loads((ART / "driver_state.json").read_text()) if (ART / "driver_state.json").exists() else {"jobs": {}}
    if st["jobs"].get(job_id, {}).get("status") == "COMPLETE":
        return 0
    t0, t_iso = time.time(), guard.now()
    cmd = [PY, "cfar1_driver.py", "job", job_id, "--gpu" if gpu else "--cpu", "--expect-rc", expect, "--disk-gb",
           str(disk_gb), "--"] + argv
    rc = subprocess.run(cmd, cwd=guard.ROOT, env={**os.environ, **(env or {})}).returncode
    dt = time.time() - t0
    st = json.loads((ART / "driver_state.json").read_text())
    j = st["jobs"].get(job_id, {})
    if gpu:
        LG.compute(job_id, stage, t_iso, guard.now(), dt, j.get("status", "?"))
    LG.manifest({"time": guard.now(), "job": job_id, "stage": stage, "seed": seed, "status": j.get("status"),
                 "rc": j.get("rc"), "seconds": round(dt, 1), "gpu_h": round(dt / 3600, 4) if gpu else 0,
                 "cmd": " ".join(argv)})
    LG.run_log(job=job_id, stage=stage, seed=seed, cmd=argv, gpu=gpu, start=t_iso, end=guard.now(), rc=rc,
               status=j.get("status"), env={k: v for k, v in (env or {}).items()})
    if rc == 77:
        stop_external(job_id)
    if rc != 0:
        LG.set_state("FAILED", job=job_id, rc=rc)
        LG.write_resume("FAILED", f"{PY} cfar1_run.py {CMD_NOW}", notes=f"Job {job_id} failed (rc {rc}); see job_logs.")
        raise SystemExit(f"job {job_id} failed rc={rc}")
    return j.get("rc")


def run_dir_of(name, root=None):
    c = sorted((root or guard.ROOT / "runs").glob(f"*_{RUN_PREFIX}{name}"))
    return c[-1] if c else None


def train(cfg_name, stage, seed, max_steps=None, disk_gb=5.0):
    leg = 1
    while True:
        rd = run_dir_of(cfg_name)
        if rd is not None:
            stj = json.loads((rd / "status.json").read_text())
            if stj["status"] == "COMPLETE":
                return rd
            if max_steps is not None and stj.get("steps_done", 0) >= max_steps and stj["status"] == "INTERRUPTED":
                return rd
        argv = ["cfar1_train.py", "--config", str((CFG_DIR / f"{cfg_name}.json").relative_to(guard.ROOT)),
                "--max-seconds", "7000"]
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
        job(jid, stage, argv, expect="0,75", disk_gb=disk_gb, seed=seed)
        leg += 1


def evaluate(ckpt_dir, part, out_name, stage, seed, disk_gb):
    out = EVALS / f"{out_name}.npz"
    if out.exists():
        return out
    job(f"eval_{out_name}", stage, ["cfar1_eval.py", "--ckpt", str(Path(ckpt_dir).relative_to(guard.ROOT)), "--part",
                                    part, "--out", str(out.relative_to(guard.ROOT))], disk_gb=disk_gb, seed=seed)
    return out


def cpu(job_id, stage, argv):
    job(job_id, stage, argv, gpu=False, disk_gb=1.0)


def read(name):
    return json.loads((ART / f"{name}.json").read_text())


def configs(*args):
    out = subprocess.run([PY, "cfar1_configs.py", *args], cwd=guard.ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(out.stderr)
    return Path(out.stdout.strip().splitlines()[-1]).stem


def free_superseded(paths, reason):
    """Delete this experiment's superseded raw eval arrays (analysis JSON retained), hashing and logging each."""
    for p in paths:
        p = Path(p)
        if p.exists() and str(p).startswith(str(EVALS)):
            sha = guard.sha256_file(p)
            nb = p.stat().st_size
            p.chmod(0o600)
            p.unlink()
            LG.deletion(str(p.relative_to(guard.ROOT)), sha, nb, reason)


# ----------------------------------------------------------------------------------------------- gputest
def gputest():
    env = {"MOE_RUNS_DIR": str(TMP / "gpu_tests/runs")}
    root = TMP / "gpu_tests/runs"
    root.mkdir(parents=True, exist_ok=True)
    out = subprocess.run([PY, "cfar1_configs.py", "gputest"], cwd=guard.ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(out.stderr)
    cfg = lambda n: str((CFG_DIR / f"{n}.json").relative_to(guard.ROOT))
    job("gt_banks_60", "tests", ["cfar1_train.py", "--config", cfg("gt_fed"), "--max-steps", "60"], expect="75", env=env)
    job("gt_resume_1", "tests", ["cfar1_train.py", "--config", cfg("gt_fed"), "--max-steps", "30"], expect="75", env=env)
    rd = sorted(root.glob(f"*_{RUN_PREFIX}gt_fed"))[-1]
    job("gt_resume_2", "tests", ["cfar1_train.py", "--config", cfg("gt_fed"), "--run-dir", str(rd.relative_to(guard.ROOT)),
                                 "--max-steps", "60"], expect="75", env=env)
    job("gt_banks_60b", "tests", ["cfar1_train.py", "--config", cfg("gt_fed"), "--max-steps", "60"], expect="75", env=env)
    job("gt_banks_60c", "tests", ["cfar1_train.py", "--config", cfg("gt_fed"), "--max-steps", "60"], expect="75", env=env)
    job("gt_nobank_a", "tests", ["cfar1_train.py", "--config", cfg("gt_fed_nobank"), "--max-steps", "60"], expect="75", env=env)
    job("gt_nobank_b", "tests", ["cfar1_train.py", "--config", cfg("gt_fed_nobank"), "--max-steps", "60"], expect="75", env=env)
    job("gt_bench", "tests", ["cfar1_train.py", "--config", cfg("gt_fed_bench"), "--max-steps", "40"], expect="75", env=env)
    rd0 = sorted(root.glob(f"*_{RUN_PREFIX}gt_fed"))[0]
    job("gt_eval_smoke2", "tests", ["cfar1_eval.py", "--ckpt", str((rd0 / "ckpt_t60").relative_to(guard.ROOT)), "--part",
                                    "EVAL", "--rows-limit", "96", "--out",
                                    str((TMP / "gpu_tests/eval_smoke2.npz").relative_to(guard.ROOT))], env=env)
    rdb = sorted(root.glob(f"*_{RUN_PREFIX}gt_fed_bench"))[-1]
    job("gt_eval_bench", "tests", ["cfar1_eval.py", "--ckpt", str((rdb / "ckpt_t40").relative_to(guard.ROOT)), "--part",
                                   "SEL", "--rows-limit", "128", "--out",
                                   str((TMP / "gpu_tests/eval_bench.npz").relative_to(guard.ROOT))], env=env)
    cpu("gpu_check", "tests", ["cfar1_gpu_check.py"])


# ----------------------------------------------------------------------------------------------- stages
def stage1():
    tb = json.loads((guard.ROOT / "artifacts/joint-semantic-residual-pretrain-v1/tables/tables.json").read_text())
    n1 = tb["budget_steps"]["S1"]
    if (ART / f"stage1{SFX}.json").exists():
        return read(f"stage1{SFX}")
    LG.set_state(f"STAGE1{SFX}_RUNNING")
    extra = ["--gen", str(GEN)]
    name = configs("stage1", *extra)
    rd = train(name, "stage1", 100, max_steps=n1, disk_gb=5.0)
    ev = evaluate(rd / "ckpt_m20", "SEL", f"S1{SFX}_SEL", "stage1", 100, disk_gb=5.0)
    cpu(f"analyze_stage1{SFX}", "stage1", ["cfar1_analyze.py", "stage1", "--eval", str(ev.relative_to(guard.ROOT)),
                                           "--out", f"stage1{SFX}"])
    s1 = read(f"stage1{SFX}")
    LG.decision(evidence=json.dumps({c: {k: r[k] for k in ("ce_F2s_minus_G", "ce_F2s_minus_L", "I_F2s_minus_I_F2h",
                                                           "I_diff_ci", "gated")} for c, r in s1["candidates"].items()}),
                competing="one seed at 20M on SEL: the selected candidate may win on noise (winner's curse); Stage 2 on "
                          "EVAL (disjoint papers) with seeds 100/101 under frozen hp is the test",
                decision_=f"Stage 1 {'PASS' if s1['stage1_pass'] else 'FAIL'}; selected {s1.get('selected')}",
                rationale="preregistered gates (student CE <= G + 0.002; no pathology) then max I_F2s - I_F2h",
                falsifier="Stage-2 Q1 (I_F2s > I_F2h in both seeds) failing", stage=f"stage1{SFX}")
    return s1


def stage2(s1):
    tb = json.loads((guard.ROOT / "artifacts/joint-semantic-residual-pretrain-v1/tables/tables.json").read_text())
    n2 = tb["budget_steps"]["S2"]
    if (ART / f"stage2{SFX}.json").exists():
        return read(f"stage2{SFX}")
    LG.set_state(f"STAGE2{SFX}_RUNNING")
    hp = s1["selected_hp"]
    evs = []
    for s in STAGE2_SEEDS:
        name = configs("final", "--seed", str(s), "--gen", str(GEN), "--T", str(hp["T"]), "--wp", str(hp["w_p"]))
        rd = train(name, "stage2", s, max_steps=n2, disk_gb=5.0)
        evs.append(evaluate(rd / "ckpt_m50", "EVAL", f"S2{SFX}_s{s}_EVAL", "stage2", s, disk_gb=4.0))
    cpu(f"analyze_stage2{SFX}", "stage2", ["cfar1_analyze.py", "stage2", "--evals",
                                           ",".join(str(e.relative_to(guard.ROOT)) for e in evs), "--out", f"stage2{SFX}"])
    s2 = read(f"stage2{SFX}")
    LG.decision(evidence=json.dumps(s2["criteria"], default=str),
                competing="two seeds; EVAL has ~500 papers; seed 100 also served Stage-1 selection (different papers)",
                decision_=f"Stage 2 promote={s2['promote']}", rationale="preregistered Q1-Q4, all required",
                falsifier="final VAL criteria BF1-BF4", stage=f"stage2{SFX}")
    return s2


def final(s1):
    from csr.cfar1_paths import REUSED
    if (ART / "final_analysis.json").exists():
        return read("final_analysis")
    hp = s1["selected_hp"]
    names = {s: configs("final", "--seed", str(s), "--gen", str(GEN), "--T", str(hp["T"]), "--wp", str(hp["w_p"]))
             for s in FINAL_SEEDS}
    if not (ART / "FROZEN_FINAL.json").exists():
        fz = {"time": guard.now(), "generation": GEN, "hp": hp, "configs": {s: n for s, n in names.items()},
              "config_sha256": {s: guard.sha256_file(CFG_DIR / f"{n}.json") for s, n in names.items()},
              "stage2": f"stage2{SFX}.json", "note": "all choices frozen before VAL is opened"}
        guard.atomic_write_json(ART / "FROZEN_FINAL.json", fz, mode=0o400)
        LG.decision(evidence=json.dumps(fz, default=str), competing="none (freeze)", decision_="FREEZE final configuration",
                    rationale="Stage-2 promotion passed; VAL opened only after this record", falsifier="n/a", stage="freeze")
    from csr.cfar1_paths import VAL_TABLES
    if not (VAL_TABLES / "psem_val.npz").exists():
        job("psem_val", "final", ["csur1_psem.py", "--split", "val", "--tables-dir", str(VAL_TABLES.relative_to(guard.ROOT)),
                                  "--frozen", str((ART / "FROZEN_FINAL.json").relative_to(guard.ROOT))], disk_gb=1.0)
    if not (VAL_TABLES / "psem_val_sh.npz").exists():
        job("psem_sh_val", "final", ["csur1_psem_shuffle.py", "--split", "val", "--tables-dir",
                                     str(VAL_TABLES.relative_to(guard.ROOT)), "--frozen",
                                     str((ART / "FROZEN_FINAL.json").relative_to(guard.ROOT))], gpu=False, disk_gb=1.0)
    LG.set_state("FINAL_RUNNING")
    evs = []
    for s in FINAL_SEEDS:
        rd = train(names[s], "final", s, disk_gb=5.0)
        evs.append(evaluate(rd / "ckpt_p100", "VAL", f"F_s{s}_VAL", "final", s, disk_gb=4.0))
    refs = {}
    for kind in ("T4", "EDL"):
        refs[kind] = []
        for i, s in enumerate((0, 1, 2, 3)):
            out = EVALS / f"REF_{kind}_s{s}_VAL.npz"
            if not out.exists():
                job(f"eval_REF_{kind}_s{s}_VAL", "evals", ["cfar1_eval.py", "--ref", REUSED[kind][s], "--part", "VAL",
                                                          "--out", str(out.relative_to(guard.ROOT))], disk_gb=0.5, seed=s)
            refs[kind].append(str(out.relative_to(guard.ROOT)))
    guard.atomic_write_json(ART / "refs_VAL.json", refs, overwrite=True)
    cpu("analyze_final", "final", ["cfar1_analyze.py", "final", "--evals", ",".join(str(e.relative_to(guard.ROOT)) for e in evs),
                                   "--refs", str((ART / "refs_VAL.json").relative_to(guard.ROOT)), "--out", "final_analysis"])
    fa = read("final_analysis")
    LG.decision(evidence=json.dumps(fa["criteria"]), competing="see report", decision_=f"terminal: {fa['outcome']}",
                rationale="preregistered BF1-BF4, evaluated once on VAL", falsifier="n/a (terminal)", stage="final")
    return fa


def main():
    global CMD_NOW
    guard.check_root()
    what = sys.argv[1]
    CMD_NOW = what
    if what == "gputest":
        gputest()
        return
    from csr.cfar1_paths import check_prereg
    check_prereg()
    if what in ("stage1", "all"):
        s1 = stage1()
        if not s1["stage1_pass"]:
            LG.set_state(f"STAGE1{SFX}_FAIL")
            LG.write_resume(f"STAGE1{SFX}_FAIL", "true  # engineering generation tied to the measured failure (prereg 12)")
            return
        if what == "stage1":
            return
    s1 = read(f"stage1{SFX}")
    if what in ("stage2", "all"):
        s2 = stage2(s1)
        free_superseded([EVALS / f"S1{SFX}_SEL.npz"], "Stage-1 raw eval arrays superseded by stage1 analysis JSON (disk)")
        if not s2["promote"]:
            LG.set_state(f"STAGE2{SFX}_NOT_PROMOTED")
            LG.write_resume(f"STAGE2{SFX}_NOT_PROMOTED", "true  # engineering generation or Experiment B (prereg 12/13)")
            return
        if what == "stage2":
            return
    if what in ("final", "all"):
        fa = final(s1)
        LG.set_state(f"TERMINAL_{fa['outcome']}")
        LG.write_resume(f"TERMINAL_{fa['outcome']}", "true  # nothing to resume")


if __name__ == "__main__":
    main()
