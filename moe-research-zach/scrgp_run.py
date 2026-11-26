#!/usr/bin/env python
"""scrgp-v1 orchestrator (prereg section 9). Idempotent; skips completed work; never relaunches after an external stop.
  scrgp_run.py gputests   GPU tests (G-GRAD, G-OFF) + stage-1 test-mode GPU jobs (G-RESUME, G-EVAL)
  scrgp_run.py stage1     per seed: est legs then eval legs (seeds 0..3)
  scrgp_run.py analyze    scrgp_analyze.py (+ inventory verify, then re-analyze with integrity)
  scrgp_run.py all        gputests -> stage1 -> analyze"""
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402
from csr import scrgp_log as LG  # noqa: E402
from csr.scrgp_paths import ART, GPU_CAP_H, SEEDS, STOP_MARKER, TMP, check_prereg  # noqa: E402

PY = str(guard.ROOT / ".venv/bin/python")
S1 = ART / "stage1"


def driver(job_id, argv, gpu=True, expect="0", disk_gb=2.0):
    cmd = [PY, "scrgp_driver.py", "job", job_id, "--gpu" if gpu else "--cpu", "--expect-rc", expect,
           "--disk-gb", str(disk_gb), "--"] + argv
    LG.disk(job_id, disk_gb)
    t_iso, t0 = guard.now(), time.time()
    rc = subprocess.run(cmd, cwd=guard.ROOT).returncode
    dt = time.time() - t0
    st = json.loads((ART / "driver_state.json").read_text())
    j = st["jobs"].get(job_id, {})
    status = j.get("status", "UNKNOWN")
    if gpu and status != "UNKNOWN" and j.get("seconds") is not None and "end" in j and j.get("start", "") >= t_iso[:19]:
        LG.compute(job_id, stage_of(job_id), j["start"], j["end"], float(j["seconds"]), status)
    LG.manifest({"time": guard.now(), "job": job_id, "stage": stage_of(job_id), "seed": seed_of(job_id),
                 "status": status, "rc": j.get("rc", rc), "seconds": round(dt, 1),
                 "gpu_h": round(float(j.get("seconds", 0)) / 3600, 4) if gpu else 0, "cmd": " ".join(argv)})
    if rc == 77 or st.get("external_stop"):
        external_stop(job_id)
    if rc != 0:
        LG.set_state("FAILED", job=job_id, rc=rc)
        LG.write_resume("FAILED", f"{PY} scrgp_run.py all", f"job {job_id} failed rc={rc}; see job_logs/{job_id}.log")
        sys.exit(rc)
    return j.get("rc", 0)


def stage_of(job_id):
    return "stage0" if job_id.startswith(("stage0", "gputest")) else "stage1"


def seed_of(job_id):
    for p in job_id.split("_"):
        if p.startswith("s") and p[1:].isdigit():
            return int(p[1:])
    return ""


def external_stop(job_id):
    state = "INCOMPLETE_STOPPED_BY_EXTERNAL_OPERATOR"
    LG.set_state(state, job=job_id)
    LG.decision(f"job {job_id} stopped externally (signal / foreign GPU process / STOP marker)",
                "intentional GPU reclamation by someone outside this session (policy)", "stop all launchers; no relaunch",
                decision_="wait for explicit user permission")
    LG.write_resume(state, f"{PY} scrgp_driver.py authorize-resume && {PY} scrgp_run.py all",
                    f"Last complete slot is kept under .tmp/scrgp-v1/slots; the pipeline skips completed work.")
    print(state, flush=True)
    sys.exit(77)


def gpu_cap(stage, need_h):
    tot, by = LG.gpu_hours()
    if tot + need_h > GPU_CAP_H["total"] or by.get(stage, 0.0) + need_h > GPU_CAP_H.get(stage, 1e9):
        LG.set_state("STOPPED_GPU_CAP", stage=stage, used=by)
        LG.write_resume("STOPPED_GPU_CAP", f"{PY} scrgp_run.py all", f"GPU cap reached: {by}")
        sys.exit(3)


def legs(prefix, argv, disk_gb, need_h, stage="stage1"):
    n = 1
    st = json.loads((ART / "driver_state.json").read_text()) if (ART / "driver_state.json").exists() else {"jobs": {}}
    while f"{prefix}_leg{n}" in st["jobs"]:
        if st["jobs"][f"{prefix}_leg{n}"].get("rc") == 0 and st["jobs"][f"{prefix}_leg{n}"]["status"] == "COMPLETE":
            return
        n += 1
    while True:
        if STOP_MARKER.exists():
            external_stop(prefix)
        gpu_cap(stage, need_h)
        rc = driver(f"{prefix}_leg{n}", argv, expect="0,75", disk_gb=disk_gb)
        if rc == 0:
            return
        n += 1


def gputests():
    tests = sorted(ART.glob("gpu_tests_2*.json"))
    if not (tests and json.loads(tests[-1].read_text())["n_fail"] == 0):
        driver("gputest_unit", ["scrgp_gpu_tests.py"], disk_gb=1)
        tests = sorted(ART.glob("gpu_tests_2*.json"))
        if json.loads(tests[-1].read_text())["n_fail"]:
            LG.set_state("FAILED_GPU_TESTS")
            sys.exit(1)
    out = TMP / "gpu_test_stage1"
    done = ART / "gpu_tests_stage1.json"
    if done.exists() and json.loads(done.read_text())["ok"]:
        return
    common = ["--seed", "0", "--test-pairs", "--rows-limit", "4", "--mb-rows", "2"]
    legs("gputest_est_full", ["scrgp_stage1.py", "--phase", "est", *common, "--out-dir", str(out / "full")], 4, 0.05,
         "stage0")
    legs("gputest_est_legs", ["scrgp_stage1.py", "--phase", "est", *common, "--out-dir", str(out / "legs"),
                              "--max-seconds", "0"], 4, 0.05, "stage0")
    legs("gputest_eval", ["scrgp_stage1.py", "--phase", "eval", *common, "--out-dir", str(out / "full")], 2, 0.05,
         "stage0")
    import torch
    ua = torch.load(out / "full" / "est_s0.pt", weights_only=False)["u"]
    ub = torch.load(out / "legs" / "est_s0.pt", weights_only=False)["u"]
    cs = {}
    for k in ua:
        a, b = ua[k].double().reshape(-1), ub[k].double().reshape(-1)
        cs[k] = float(a @ b / (a.norm() * b.norm()))
    ev = json.loads((out / "full" / "eval_s0.json").read_text())
    ok = min(cs.values()) >= 0.999 and ev.get("hash_restored") is True
    guard.atomic_write_json(done, {"time": guard.now(), "G-RESUME_cos_u": cs, "G-EVAL_hash_restored":
                                   ev.get("hash_restored"), "ok": ok}, mode=0o400)
    LG.observe("GPU stage-1 test jobs", resume_cos=cs, eval_hash_restored=ev.get("hash_restored"), ok=ok)
    import shutil
    for p in out.rglob("*"):
        if p.is_file():
            p.chmod(0o600)
    shutil.rmtree(out, ignore_errors=True)
    if not ok:
        LG.set_state("FAILED_GPU_TESTS")
        sys.exit(1)


def stage1():
    S1.mkdir(parents=True, exist_ok=True)
    LG.set_state("RUNNING_STAGE1")
    LG.write_resume("RUNNING_STAGE1", f"{PY} scrgp_run.py all", "Completed seeds/phases are skipped automatically.")
    for s in SEEDS:
        if not (S1 / f"est_s{s}.json").exists():
            legs(f"stage1_est_s{s}", ["scrgp_stage1.py", "--phase", "est", "--seed", str(s)], 4.0, 1.2)
            LG.observe(f"stage1 est s{s} complete", **json.loads((S1 / f"est_s{s}.json").read_text()))
        if not (S1 / f"eval_s{s}.json").exists():
            legs(f"stage1_eval_s{s}", ["scrgp_stage1.py", "--phase", "eval", "--seed", str(s)], 1.5, 1.0)
            LG.observe(f"stage1 eval s{s} complete", **json.loads((S1 / f"eval_s{s}.json").read_text()))


def analyze():
    r = subprocess.run([PY, "scrgp_analyze.py"], cwd=guard.ROOT)
    if r.returncode:
        LG.set_state("FAILED_ANALYSIS")
        sys.exit(r.returncode)
    if not sorted(ART.glob("protected_inventory_before.verify_*.json")):
        subprocess.run([PY, "scrgp_inventory.py", "--verify", str(ART / "protected_inventory_before.json")],
                       cwd=guard.ROOT)
        subprocess.run([PY, "scrgp_analyze.py"], cwd=guard.ROOT, check=True)
    fa = json.loads((ART / "final_analysis.json").read_text())
    LG.decision("Stage-1 analysis complete", "prereg 5.4 criteria evaluated once", f"verdict {fa['verdict']}",
                result=json.dumps(fa["criteria"]), decision_=("Stage 2 per prereg 6" if fa["stage1_pass"] else
                                                              "STOP: no Stage 2; negative report"))
    LG.set_state("STAGE1_ANALYZED", verdict=fa["verdict"])
    LG.write_resume(f"STAGE1_ANALYZED ({fa['verdict']})", "true  # nothing to resume",
                    "Stage-1 verdict in final_analysis.json.")


def main():
    guard.check_root()
    check_prereg()
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("gputests", "all"):
        gputests()
    if what in ("stage1", "all"):
        stage1()
    if what in ("analyze", "all"):
        analyze()


if __name__ == "__main__":
    main()
