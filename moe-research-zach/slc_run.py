#!/usr/bin/env python
"""semantic-lift-245m-confirm-v1 orchestrator (docs/SEMANTIC_LIFT_245M_CONFIRM_V1_PREREG.md). Every GPU job goes
through slc_driver.run_job (idempotent; completed jobs are skipped). An external interruption (signal, foreign GPU
process, STOP marker, launcher block) marks the experiment PAUSED_BY_OPERATOR, writes RESUME.md and exits; nothing
relaunches automatically.

  slc_run.py prep     configs for E/F seeds 0-3 + config equivalence vs the reused C6/SHUF configs
  slc_run.py checks   GPU check (CHK_E_s0, CHK_E_s0b, CHK_F_s0 to step 51 vs P1_C6_s0 / P1_SHUF_s0)
  slc_run.py train    the eight 245M runs in the registered counterbalanced order (no early performance stop)
  slc_run.py open     verify all eight runs and their milestones -> FINAL_OPEN.json
  slc_run.py evals    40 frozen evaluations (p100 x 24, s50 x 16)
  slc_run.py all      prep -> checks -> train -> open -> evals -> slc_analyze.py"""
import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402
from csr.slc_paths import (ART, CFG_DIR, EXP, GPU_CAP_H, LIFT, MILESTONES, N_STEPS, ORDER, PLAN_SHA,  # noqa: E402
                           REUSED, RUN_PREFIX, SEEDS, SHUF_TABLE, SHUF_TABLE_SHA, TABLE, TABLE_SHA, TMP, check_prereg,
                           prereg_sha)

RUNS = ART / "run_dirs.json"
RESUME_CMD = ".venv/bin/python slc_run.py all"
TRAIN_DISK_GIB = 3.0
EVAL_DISK_GIB = 0.2


# ------------------------------------------------------------------------------------------------ logging
def code_hash():
    h = hashlib.sha256()
    files = sorted([str(p.relative_to(guard.ROOT)) for p in guard.ROOT.glob("slc_*.py")] +
                   [str(p.relative_to(guard.ROOT)) for p in (guard.ROOT / "src/csr").glob("slc_*.py")] +
                   ["src/csr/sll_lift.py", "fle_evaluate.py", "fle_configs.py", "src/csr/lfr_model.py",
                    "src/csr/lfr_eval.py", "src/csr/lfr_lib.py", "src/csr/slk_lib.py", "src/csr/gc_lib.py",
                    "src/csr/guard.py"])
    for f in files:
        p = guard.ROOT / f
        if p.exists():
            h.update(f.encode() + b"\0" + guard.sha256_file(p).encode() + b"\n")
    return h.hexdigest()


def append(name, text):
    ART.mkdir(parents=True, exist_ok=True)
    with open(ART / name, "a") as f:
        f.write(text)


def run_log(rec, md=None):
    append("RUN_LOG.jsonl", json.dumps({"time": guard.now(), "git_commit": None,
                                        "git_note": "no git repository; code_tree_sha256 = hash of slc code + imports",
                                        "code_tree_sha256": code_hash(), **rec}, default=str) + "\n")
    if md:
        if not (ART / "RUN_LOG.md").exists():
            append("RUN_LOG.md", f"# RUN_LOG — {EXP} (append-only)\n")
        append("RUN_LOG.md", f"\n### {guard.now()} — {md[0]}\n{md[1]}\n")


def decision(observation, evidence, hypothesis, action, rejected=""):
    if not (ART / "DECISION_LOG.md").exists():
        append("DECISION_LOG.md", f"# DECISION_LOG — {EXP} (append-only)\n")
    append("DECISION_LOG.md", f"\n### {guard.now()}\n- Observation: {observation}\n- Evidence: {evidence}\n"
                              f"- Hypothesis: {hypothesis}\n- Action: {action}\n"
                              + (f"- Rejected alternatives: {rejected}\n" if rejected else ""))


def observe(text):
    if not (ART / "OBSERVATIONS.md").exists():
        append("OBSERVATIONS.md", f"# OBSERVATIONS — {EXP} (append-only)\n\n")
    append("OBSERVATIONS.md", f"- {guard.now()} {text}\n")


def deviation(text, effect, can_affect):
    p = ART / "deviations.jsonl"
    n = sum(1 for _ in open(p)) if p.exists() else 0
    append("deviations.jsonl", json.dumps({"n": n, "time": guard.now(), "text": text, "effect": effect,
                                           "can_affect_conclusion": can_affect}) + "\n")


def results(line):
    if not (ART / "RESULTS_LIVE.md").exists():
        append("RESULTS_LIVE.md", "# RESULTS_LIVE (append-only; every completed result)\n\n| time | item | values |\n"
                                  "|---|---|---|\n")
    append("RESULTS_LIVE.md", f"| {guard.now()} | {line} |\n")


def disk_log(stage, est, note=""):
    if not (ART / "disk_ledger.tsv").exists():
        append("disk_ledger.tsv", "time\tstage\tfree_gib\testimated_peak_gib\tnote\n")
    append("disk_ledger.tsv", f"{guard.now()}\t{stage}\t{guard.disk_free_bytes() / 2**30:.1f}\t{est}\t{note}\n")


def compute_log(t0, t1, secs, job_id, kind, phase, status):
    if not (ART / "compute_ledger.tsv").exists():
        append("compute_ledger.tsv", "start\tend\tgpu_seconds\tjob\tkind\tphase\tstatus\n")
    append("compute_ledger.tsv", f"{t0}\t{t1}\t{secs}\t{job_id}\t{kind}\t{phase}\t{status}\n")


def gpu_hours(phases=None):
    tot = 0.0
    p = ART / "RUN_LOG.jsonl"
    if not p.exists():
        return 0.0
    for line in p.read_text().splitlines():
        r = json.loads(line)
        if phases is None or r.get("phase") in phases:
            tot += float(r.get("gpu_seconds") or 0)
    return tot / 3600


def set_state(state, **extra):
    guard.atomic_write_json(ART / "STATE.json", {"state": state, "time": guard.now(), **extra}, overwrite=True)


def write_resume(state, cmd, notes=""):
    txt = (f"# RESUME — {EXP}\n\nUpdated {guard.now()}.\n\n**State:** {state}\n\n"
           f"GPU used: {gpu_hours():.2f} GPU-h (cap {GPU_CAP_H['total']}). Disk free: "
           f"{guard.disk_free_bytes() / 2**30:.1f} GiB (reserve 100 GiB).\n\n**Resume command"
           f"{' (only after explicit operator permission)' if 'PAUSED' in state or 'FAILED' in state else ''}:**\n\n"
           f"```\ncd {guard.ROOT} && source env.sh && {cmd}\n```\n\n{notes}\n")
    guard.atomic_write_bytes(ART / "RESUME.md", txt.encode(), overwrite=True)


def jl(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


# ------------------------------------------------------------------------------------------------ runs / jobs
def run_dirs():
    return json.loads(RUNS.read_text()) if RUNS.exists() else {}


def remember(rn):
    d = run_dirs()
    if rn not in d:
        c = sorted((guard.ROOT / "runs").glob(f"*_{RUN_PREFIX}{rn}"))
        if c:
            d[rn] = str(c[-1].relative_to(guard.ROOT))
            guard.atomic_write_json(RUNS, d, overwrite=True)
    return d.get(rn)


def status_of(rn):
    rd = run_dirs().get(rn)
    if not rd or not (guard.ROOT / rd / "status.json").exists():
        return None
    return json.loads((guard.ROOT / rd / "status.json").read_text())


def cap_check(phase):
    used, tot = gpu_hours([phase]), gpu_hours()
    if used >= GPU_CAP_H[phase] or tot >= GPU_CAP_H["total"]:
        decision(f"GPU cap reached before a {phase} job", f"phase {used:.2f}/{GPU_CAP_H[phase]} h, total {tot:.2f}/"
                 f"{GPU_CAP_H['total']} h", "-", "stop cleanly (prereg 10)")
        set_state("STOPPED_GPU_CAP", phase=phase)
        write_resume(f"STOPPED_GPU_CAP — phase {phase}", RESUME_CMD, "A cap increase needs explicit operator approval.")
        raise SystemExit(f"GPU cap reached ({phase})")


def disk_check(stage, est):
    free = guard.disk_free_bytes() / 2**30
    disk_log(stage, est, "pre-job estimate")
    if free - est < 100:
        need = 100 + est - free
        decision(f"Disk estimate blocks {stage}", f"free {free:.2f} GiB, estimated peak {est} GiB, reserve 100", "-",
                 f"stop cleanly; {need:.1f} GiB more free space needed")
        set_state("PAUSED_DISK_RESERVE", stage=stage, need_gib=need)
        write_resume(f"PAUSED_DISK_RESERVE — {stage} needs ~{est} GiB; free {free:.1f} GiB", RESUME_CMD,
                     "Free disk, then resume (completed jobs skipped).")
        raise SystemExit(f"disk: {stage} needs {need:.1f} GiB more")


def job(job_id, argv, phase, kind, run_name=None, seed=None, cfg_p=None, expect=(0,), disk_gb=2.0, pre=None):
    from slc_driver import load_state, run_job
    if (load_state()["jobs"].get(job_id) or {}).get("status") == "COMPLETE":
        return 0
    cap_check(phase)
    disk_check(job_id, disk_gb)
    if pre:
        run_log({"phase": phase, "kind": "pre_run", "job": job_id, "gpu_seconds": 0, **pre},
                (f"PRE {job_id}", "\n".join(f"- {k}: {v}" for k, v in pre.items())))
    rec = {"phase": phase, "kind": kind, "job": job_id, "argv": argv, "run_name": run_name, "seed": seed,
           "config": str(cfg_p) if cfg_p else None, "config_sha256": guard.sha256_file(cfg_p) if cfg_p else None}
    t0 = guard.now()
    try:
        rc = run_job(job_id, argv, gpu=True, expect=expect, disk_gb=disk_gb)
        ok = True
    except SystemExit as e:
        rc, ok = e.code, False
    js = load_state()["jobs"].get(job_id, {})
    compute_log(t0, guard.now(), js.get("seconds"), job_id, kind, phase, js.get("status"))
    if run_name and kind == "train":
        remember(run_name)
    post = status_of(run_name) if run_name and kind == "train" else None
    rd = run_dirs().get(run_name) if run_name else None
    rec.update({"exit_status": js.get("status"), "rc": rc, "gpu_seconds": js.get("seconds"), "run_dir": rd,
                "end_tokens": (post or {}).get("seen"), "run_status": (post or {}).get("status"),
                "steps_done": (post or {}).get("steps_done"), "plan_sha256": (post or {}).get("plan_sha256"),
                "tokens_per_s": (post or {}).get("tokens_per_s"),
                "checkpoints": {k: {"path": v["path"], "sha256": v["sha256"]} for k, v in
                                ((post or {}).get("ckpts") or {}).items()} if post else None,
                "disk_free_gib": guard.disk_free_bytes() / 2**30,
                "validity": "valid" if ok else ("INCOMPLETE (external interruption; not evidence)"
                                                if js.get("status") == "INCOMPLETE" else "FAILED (not evidence)")})
    run_log(rec, (f"END {job_id} ({js.get('status')}, rc {rc}, {float(js.get('seconds') or 0):.0f} s)",
                  f"- run dir {rd}; steps {rec['steps_done']}; tokens {rec['end_tokens']}; tok/s "
                  f"{rec['tokens_per_s']}; free disk {rec['disk_free_gib']:.1f} GiB" if run_name else
                  f"- free disk {rec['disk_free_gib']:.1f} GiB"))
    if post:
        append("run_manifest.jsonl", json.dumps({"time": guard.now(), "run_name": run_name, "run_dir": rd, "seed": seed,
                                                 "config": str(cfg_p), "config_sha256": rec["config_sha256"],
                                                 "status": post.get("status"), "steps_done": post.get("steps_done"),
                                                 "seen": post.get("seen"), "plan_sha256": post.get("plan_sha256"),
                                                 "ckpts": post.get("ckpts"), "tokens_per_s": post.get("tokens_per_s"),
                                                 "params": post.get("params"), "peak_alloc_gib": post.get("peak_alloc_gib"),
                                                 "legs": post.get("legs")}, default=str) + "\n")
    if ok:
        write_resume(f"RUNNING — last completed job {job_id} ({guard.now()}); orchestrator active", RESUME_CMD,
                     "If no slc process is alive the orchestrator stopped: read STATE.json, driver_state.json and the "
                     "last RUN_LOG record before resuming. Completed jobs are skipped on resume.")
        return rc
    if js.get("status") in ("INCOMPLETE", None):
        set_state("PAUSED_BY_OPERATOR", job=job_id, run_dir=rd)
        decision(f"External interruption during {job_id}", f"driver status {js.get('status')}, rc {rc}",
                 "operator wants the GPU (prereg 10)", "PAUSED_BY_OPERATOR: trainer checkpointed at the next step "
                 "boundary (if training), driver stopped, GPU released, no restart")
        write_resume("PAUSED_BY_OPERATOR — external interruption (signal / foreign GPU process / STOP marker / launcher "
                     f"block) during job {job_id}; no automatic relaunch",
                     ".venv/bin/python slc_driver.py authorize-resume && " + RESUME_CMD,
                     f"Last job `{job_id}`; run dir {rd}. Trainers keep model/optimizer/RNG/data cursor/LiftStats in their "
                     "verified resumable slot and resume in place (--run-dir). Completed jobs are skipped.")
    else:
        set_state("FAILED_JOB", job=job_id, rc=rc)
        write_resume(f"FAILED — job {job_id} (rc {rc}); diagnose before resuming", RESUME_CMD,
                     f"Job log: artifacts/semantic-lift-245m-confirm-v1/job_logs/{job_id}.log")
    raise SystemExit(rc if isinstance(rc, int) else 1)


def drop_slot(rn, why):
    """Delete this experiment's own superseded resumable slot after its milestones verify (ledgered)."""
    from csr.slc_ckpt import verify_dir
    rd, st = run_dirs().get(rn), status_of(rn)
    if not rd or not st:
        return
    for c in (st.get("ckpts") or {}).values():
        if not verify_dir(guard.ROOT / c["path"]):
            raise guard.GuardError(f"milestone {c['path']} fails verification; slot kept")
    sl = TMP / "slots" / Path(rd).name
    if sl.exists() and str(sl).startswith(str(TMP)):
        nb = sum(p.stat().st_size for p in sl.rglob("*") if p.is_file())
        shutil.rmtree(sl)
        append("deleted_artifacts.tsv", f"{guard.now()}\t{sl.relative_to(guard.ROOT)}\t{nb}\t{why}\n")
        disk_log("slot_delete", 0, f"{sl.name} {nb / 2**30:.2f} GiB")


def drop_run_dir(rn, why):
    """Check runs only (scratch created by this experiment): delete the run directory after its result is hashed."""
    rd = run_dirs().get(rn)
    if not rd or not rn.startswith(("CHK_", "DIAG")):
        return
    p = guard.ROOT / rd
    if p.exists() and p.name.endswith(RUN_PREFIX + rn):
        nb = sum(q.stat().st_size for q in p.rglob("*") if q.is_file())
        shutil.rmtree(p)
        append("deleted_artifacts.tsv", f"{guard.now()}\t{rd}\t{nb}\t{why}\n")


# ------------------------------------------------------------------------------------------------ configs
def make_cfg(run_name, seed, kind, category, milestones=None):
    from fle_configs import MODEL, TRAIN
    t, s = (TABLE, TABLE_SHA) if kind == "E" else (SHUF_TABLE, SHUF_TABLE_SHA)
    if guard.sha256_file(guard.ROOT / t) != s:
        raise guard.GuardError(f"table hash mismatch {t}")
    cfg = {"experiment": EXP, "run_name": run_name, "seed": seed, "budget_tokens": 245 * 1_000_000,
           "prereg_sha256": prereg_sha(), "model": MODEL, "train": dict(TRAIN),
           "cond": {"paper_latent": {"table": t, "table_sha256": s, "beta": 0.005}, "lift": dict(LIFT)},
           "kind": "final", "milestones": dict(MILESTONES if milestones is None else milestones), "diag_every": 100,
           "category": category}
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    p = CFG_DIR / f"{run_name}.json"
    if p.exists():
        if json.loads(p.read_text()) != cfg:
            raise guard.GuardError(f"config {p} exists and differs")
    else:
        guard.atomic_write_json(p, cfg, mode=0o400)
    return p, cfg


def train(rn, cfg_p, seed, phase, to_step=N_STEPS, pre=None, disk_gb=TRAIN_DISK_GIB):
    """Run (or continue) training of run rn up to to_step (in legs); idempotent."""
    for leg in range(1, 9):
        st = status_of(rn)
        if st and (st.get("status") == "COMPLETE" or (st.get("steps_done") or 0) >= to_step):
            return st
        rd = run_dirs().get(rn)
        argv = ["slc_train.py", "--config", str(cfg_p.relative_to(guard.ROOT))]
        if rd:
            argv += ["--run-dir", rd]
        if to_step < N_STEPS:
            argv += ["--max-steps", str(to_step)]
        jid = f"train_{rn}_to{to_step}_leg{leg}" + (f"_from{st.get('steps_done')}" if st else "")
        p = dict(pre or {})
        p.update({"leg": leg, "starting_state": f"resume from verified slot at step {st.get('steps_done')}" if st else
                  "fresh init LFRModel(cfg, seed)", "estimated_peak_storage_gib": disk_gb})
        job(jid, argv, phase, "train", rn, seed, cfg_p, expect=(0, 75), disk_gb=disk_gb, pre=p)
    raise guard.GuardError(f"{rn}: not at step {to_step} after 8 legs")


def lift_logs(rn):
    rd = run_dirs()[rn]
    rows = [json.loads(x) for x in (guard.ROOT / rd / "metrics.jsonl").read_text().splitlines()]
    return [x["lift"] for x in rows if "lift" in x], [x for x in rows if "step" in x]


def activity(rn):
    """Multiplier statistics / invariants / stability / training load from the training logs."""
    lr, steps = lift_logs(rn)
    act = [x for x in lr if x["lam_eff"] > 0 and x["m_by_layer"]]
    full = [x for x in act if x["lam_eff"] >= LIFT["lam"] - 1e-12]
    last = act[-1] if act else None

    def lmean(x, k):
        return sum(v[k] for v in x["m_by_layer"].values()) / len(x["m_by_layer"])
    worst = max((v["m_wmean_maxdev"] for x in act for v in x["m_by_layer"].values()), default=0.0)
    mn = min((v["m_min"] for x in act for v in x["m_by_layer"].values()), default=1.0)
    mx = max((v["m_max"] for x in act for v in x["m_by_layer"].values()), default=1.0)
    finite = all(math.isfinite(x["lm"]) for x in steps)
    lmin = min((min(x.get("load_min_x_uniform_by_layer") or [1.0]) for x in lr if x["step"] >= 388), default=None)
    lmax = max((max(x.get("load_max_x_uniform_by_layer") or [1.0]) for x in lr if x["step"] >= 388), default=None)
    return {"last_step": last["step"] if last else None, "lam_eff_last": last["lam_eff"] if last else 0.0,
            "m_absdev_wmean_last": lmean(last, "m_absdev_wmean") if last else 0.0,
            "m_absdev_wmean_full_mean": (sum(lmean(x, "m_absdev_wmean") for x in full) / len(full)) if full else 0.0,
            "clip_frac_last": (lmean(last, "clip_lo") + lmean(last, "clip_hi")) if last else 0.0,
            "wmean_maxdev": worst, "m_min": mn, "m_max": mx,
            "invariants_ok": bool(worst <= 1e-6 and mn >= 0.5 - 1e-9 and mx <= 2.0 + 1e-9),
            "stability_r_last": lr[-1]["stats"].get("stability_r") if lr else None,
            "ess_min_last": lr[-1]["stats"].get("ess_min") if lr else None,
            "lift_absmean_last": lr[-1]["stats"].get("lift_absmean") if lr else None,
            "finite": finite, "train_load_min_x_uniform_after_388": lmin, "train_load_max_x_uniform_after_388": lmax,
            "lm_last": steps[-1]["lm"] if steps else None, "n_logged": len(lr)}


# ------------------------------------------------------------------------------------------------ stages
def prep():
    check_prereg()
    rm = jl(ART / "reuse_manifest.json")
    if not rm or not rm["all_ok"]:
        raise guard.GuardError("reuse manifest missing or not all_ok")
    out = ART / "config_equivalence.json"
    if out.exists():
        return jl(out)
    allowed = {"experiment", "run_name", "prereg_sha256", "category", "milestones", "cond.lift"}
    rows, ok = {}, True

    def flat(d, p=""):
        o = {}
        for k, v in d.items():
            if isinstance(v, dict) and k != "lift":
                o.update(flat(v, p + k + "."))
            else:
                o[p + k] = v
        return o
    for kind, s in ORDER:
        rn = f"{kind}_s{s}"
        p, cfg = make_cfg(rn, s, kind, "final")
        base = json.loads((guard.ROOT / REUSED[f"{'C6' if kind == 'E' else 'SHUF'}_s{s}"] / "config.json").read_text())
        fa, fb = flat(cfg), flat(base)
        diff = sorted(k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k))
        bad = [k for k in diff if not any(k == a_ or k.startswith(a_ + ".") for a_ in allowed)]
        rows[rn] = {"config": str(p.relative_to(guard.ROOT)), "config_sha256": guard.sha256_file(p),
                    "baseline": REUSED[f"{'C6' if kind == 'E' else 'SHUF'}_s{s}"], "differing_fields": diff,
                    "disallowed": bad}
        ok = ok and not bad and cfg["cond"]["lift"] == LIFT
    doc = {"time": guard.now(), "allowed_differences": sorted(allowed), "runs": rows, "all_ok": ok}
    guard.atomic_write_json(out, doc, mode=0o400)
    decision("configs written and compared with the reused C6/SHUF configs",
             json.dumps({k: v["differing_fields"] for k, v in rows.items()}), "-",
             "proceed" if ok else "STOP: config differs beyond the allowed fields")
    if not ok:
        raise SystemExit("config equivalence failed")
    return doc


def checks():
    out = ART / "gpu_checks.json"
    if out.exists():
        r = jl(out)
        if not r["all_pass"]:
            v2 = jl(ART / "gpu_checks_v2.json")  # deviation 2: diagnosis-based correction of the tolerance spec
            if v2 and v2["all_pass"]:
                return v2
            raise SystemExit("GPU checks failed; see gpu_checks.json")
        return r
    for rn, kind in (("CHK_E_s0", "E"), ("CHK_E_s0b", "E"), ("CHK_F_s0", "F")):
        p, _ = make_cfg(rn, 0, kind, "check")
        train(rn, p, 0, "checks", to_step=51, pre={"purpose": "GPU check (prereg 4.3)"}, disk_gb=3.0)
    lm = lambda rn: {x["step"]: x["lm"] for x in lift_logs(rn)[1]}  # noqa: E731
    ref = lambda key: {x["step"]: x["lm"] for x in (json.loads(y) for y in (guard.ROOT / REUSED[key] / "metrics.jsonl")  # noqa: E731
                                                       .read_text().splitlines()) if "step" in x}
    A, B, Fm = lm("CHK_E_s0"), lm("CHK_E_s0b"), lm("CHK_F_s0")
    C6, SH = ref("C6_s0"), ref("SHUF_s0")
    noise = abs(A[50] - B[50])
    tol = max(0.003, 3 * noise)
    g = {"E_step0_abs_vs_C6": abs(A[0] - C6[0]), "E_repeat_step0_abs_vs_C6": abs(B[0] - C6[0]),
         "F_step0_abs_vs_SHUF": abs(Fm[0] - SH[0]), "E_step50_abs_vs_C6": abs(A[50] - C6[50]),
         "E_repeat_step50_abs_vs_C6": abs(B[50] - C6[50]), "F_step50_abs_vs_SHUF": abs(Fm[50] - SH[50]),
         "noise_step50_E_vs_Erepeat": noise, "tol": tol, "values": {"E": [A[0], A[50]], "E_repeat": [B[0], B[50]],
                                                                     "F": [Fm[0], Fm[50]], "C6": [C6[0], C6[50]],
                                                                     "SHUF": [SH[0], SH[50]]}}
    g["all_pass"] = bool(g["E_step0_abs_vs_C6"] < 1e-6 and g["E_repeat_step0_abs_vs_C6"] < 1e-6 and
                         g["F_step0_abs_vs_SHUF"] < 1e-6 and g["E_step50_abs_vs_C6"] <= tol)
    g["descriptive_F_step50_within_tol"] = bool(g["F_step50_abs_vs_SHUF"] <= tol)
    g["time"] = guard.now()
    g["run_dirs"] = {rn: run_dirs().get(rn) for rn in ("CHK_E_s0", "CHK_E_s0b", "CHK_F_s0")}
    guard.atomic_write_json(out, g, mode=0o400)
    results(f"GPU checks | step0 E-C6 {g['E_step0_abs_vs_C6']:.2e}, F-SHUF {g['F_step0_abs_vs_SHUF']:.2e}; step50 E-C6 "
            f"{g['E_step50_abs_vs_C6']:.4f} (noise {noise:.4f}, tol {tol:.4f}); pass {g['all_pass']}")
    decision("GPU equivalence checks vs the reused 245M baselines", json.dumps({k: v for k, v in g.items()
                                                                               if k not in ("values", "run_dirs")}),
             "-", "proceed to the eight 245M runs" if g["all_pass"] else "STOP: implementation / baseline mismatch")
    for rn in ("CHK_E_s0", "CHK_E_s0b", "CHK_F_s0"):
        drop_slot(rn, "GPU check run; slot superseded (test only)")
        drop_run_dir(rn, "GPU check run directory; result hashed in gpu_checks.json (test only)")
    if not g["all_pass"]:
        raise SystemExit("GPU checks failed")
    return g


def train_all():
    for i, (kind, s) in enumerate(ORDER):
        rn = f"{kind}_s{s}"
        p, _ = make_cfg(rn, s, kind, "final")
        st = status_of(rn)
        if st and st.get("status") == "COMPLETE" and (ART / "train_summaries" / f"{rn}.json").exists():
            continue
        pre = {"run": rn, "seed": s, "condition": kind, "order_index": i, "config_sha256": guard.sha256_file(p),
               "prereg_sha256": prereg_sha(), "free_disk_gib": round(guard.disk_free_bytes() / 2**30, 1)}
        st = train(rn, p, s, "train", pre=pre)
        if st.get("status") != "COMPLETE" or st.get("seen") != 245_032_457 or st.get("plan_sha256") != PLAN_SHA:
            raise guard.GuardError(f"{rn} finished without the full registered plan: {st.get('status')}")
        act = activity(rn)
        (ART / "train_summaries").mkdir(exist_ok=True)
        guard.atomic_write_json(ART / "train_summaries" / f"{rn}.json",
                                {"time": guard.now(), "run": rn, "status": st, "activity": act}, mode=0o400)
        results(f"train {rn} | COMPLETE {st['seen']} tok, {st['tokens_per_s']:.0f} tok/s, lm_last {act['lm_last']:.4f}, "
                f"|m-1| full-strength mean {act['m_absdev_wmean_full_mean']:.3f}, clip {act['clip_frac_last']:.3f}, "
                f"stability {act['stability_r_last']}, invariants {act['invariants_ok']}, finite {act['finite']}, train "
                f"load after ramp [{act['train_load_min_x_uniform_after_388']}, {act['train_load_max_x_uniform_after_388']}]")
        observe(f"{rn} complete (training-log statistics only; no evaluation): {json.dumps(act)}")
        if not act["invariants_ok"]:
            decision(f"multiplier invariants violated in {rn}", json.dumps(act), "integrity failure",
                     "STOP (prereg 5.5): integrity")
            set_state("STOPPED_INTEGRITY", run=rn)
            raise SystemExit("invariants violated")
        drop_slot(rn, "run COMPLETE; milestones s50/p100 verified; rolling slot superseded")


def open_final():
    out = ART / "FINAL_OPEN.json"
    if out.exists():
        return jl(out)
    from csr.slc_ckpt import verify_dir
    rows = {}
    for kind, s in ORDER:
        rn = f"{kind}_s{s}"
        st = status_of(rn)
        ok = (st and st["status"] == "COMPLETE" and st["seen"] == 245_032_457 and st["plan_sha256"] == PLAN_SHA and
              st["steps_done"] == N_STEPS and set(st["ckpts"]) == set(MILESTONES) and
              all(verify_dir(guard.ROOT / c["path"]) for c in st["ckpts"].values()))
        rows[rn] = {"run_dir": run_dirs()[rn], "ok": bool(ok),
                    "ckpts": {k: {"path": v["path"], "sha256": v["sha256"]} for k, v in st["ckpts"].items()} if st else None}
        if not ok:
            raise guard.GuardError(f"{rn} not complete / milestones do not verify")
    doc = {"time": guard.now(), "runs": rows, "note": "all eight E/F runs complete; configurations and checkpoints "
                                                      "frozen; evaluation may start (prereg 5)"}
    guard.atomic_write_json(out, doc, mode=0o400)
    decision("all eight 245M runs complete and verified", json.dumps({k: v["ok"] for k, v in rows.items()}), "-",
             "FINAL_OPEN written; run the 40 frozen evaluations")
    return doc


def eval_keys():
    keys = []
    for s in SEEDS:
        for c in ("E", "F", "C6", "SHUF", "EDL", "T4"):
            keys.append((f"{c}_s{s}", "p100"))
    for s in SEEDS:
        for c in ("E", "F", "C6", "SHUF"):
            keys.append((f"{c}_s{s}", "s50"))
    return keys


def evals():
    open_final()
    for key, tag in eval_keys():
        out = ART / "evals" / f"{key}_{tag}.json"
        if out.exists():
            continue
        job(f"eval_{key}_{tag}", ["slc_eval.py", "--key", key, "--tag", tag], "eval", "eval", disk_gb=EVAL_DISK_GIB,
            pre={"checkpoint": f"{key} {tag}", "purpose": "frozen evaluation (prereg 5)"})
        d = jl(out)
        f = d["function"]
        results(f"eval {key} {tag} | VAL CE {d['val']['ce']:.5f}; domain NMI {d['routing_domain']['nmi_mean']:.4f}; "
                f"I_ord {f['mean_interaction']:.3e} (n {f['n_defined']}); I_cond {f['cond_mean_interaction']} "
                f"(adequate {f['cond_n_adequate']}); spec {f['n_specialized']}; pathology {d['pathology']}; "
                f"{sum(d['seconds'].values()):.0f} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prep", "checks", "train", "open", "evals", "all"])
    a = ap.parse_args()
    guard.check_root()
    check_prereg()
    ART.mkdir(parents=True, exist_ok=True)
    if a.stage in ("prep", "all"):
        prep()
    if a.stage in ("checks", "all"):
        set_state("RUNNING_CHECKS")
        checks()
    if a.stage in ("train", "all"):
        set_state("RUNNING_TRAINING")
        train_all()
    if a.stage in ("open", "all"):
        open_final()
    if a.stage in ("evals", "all"):
        set_state("RUNNING_EVALUATION")
        evals()
        set_state("EVALUATION_COMPLETE")
    if a.stage == "all":
        r = subprocess.run([sys.executable, "slc_analyze.py"], cwd=guard.ROOT)
        if r.returncode != 0:
            raise SystemExit("analysis failed")
        set_state("ANALYSIS_COMPLETE")
        write_resume("ANALYSIS_COMPLETE — results.json written; write the report", "true  # nothing to resume")


if __name__ == "__main__":
    main()
