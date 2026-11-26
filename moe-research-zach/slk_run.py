#!/usr/bin/env python
"""csr-semantic-edl-long-adaptive-k-v1 orchestrator. Every GPU job goes through slk_driver.run_job (idempotent;
an external interruption latches, records INCOMPLETE and stops; nothing relaunches automatically).

  slk_run.py checks        CPU suites (new + ported) and GPU checks
  slk_run.py p1train       Phase-1 training (priority order; seed-3 group subject to the budget rule)
  slk_run.py p1eval        opens VAL once (PHASE1_VAL_OPEN.json) after all Phase-1 training, then VAL evaluations
  slk_run.py phase1        checks + p1train + p1eval
Logs: RUN_LOG.jsonl, DECISION_LOG.md, RESULTS_LIVE.md, DEVIATIONS.jsonl, RESUME.md, SPECIALIZATION_TRAJECTORY.json."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402
from csr.slk_paths import ART, GPU_CAP_H, REUSED, RUN_PREFIX, TMP  # noqa: E402

RUNS = ART / "run_dirs.json"
P1_ORDER = [("C6", 0), ("SHUF", 0), ("C6", 1), ("SHUF", 1), ("C6", 2), ("SHUF", 2), ("EDL", 2),
            ("C6", 3), ("SHUF", 3), ("T4", 3), ("EDL", 3)]
EST_TRAIN_H = 0.65
# Stage-limited resume (user instructions 2026-11-16): finishes Phase 1 + Phase 2A, never dispatches Phase 2B.
RESUME_CMD = ".venv/bin/python slk_run.py resume-through-2a"
CODE_FILES = ["slk_train.py", "slk_driver.py", "slk_configs.py", "slk_evaluate.py", "slk_run.py", "slk_tests.py",
              "slk_gpu_checks.py", "fle_evaluate.py", "fle_configs.py", "src/csr/slk_paths.py", "src/csr/slk_lib.py",
              "src/csr/slk_ckpt.py", "src/csr/gc_lib.py", "src/csr/sem_lib.py", "src/csr/lfr_eval.py",
              "src/csr/lfr_audit.py", "src/csr/lfr_model.py", "src/csr/lfr_lib.py", "src/csr/fsp_lib.py",
              "src/csr/fsp_model.py", "src/csr/fle_lib.py", "src/csr/guard.py"]


# ------------------------------------------------------------------------------------------------ logging
def code_hash():
    h = hashlib.sha256()
    for f in sorted(CODE_FILES + [str(p.relative_to(guard.ROOT)) for p in guard.ROOT.glob("slk_*.py")]):
        p = guard.ROOT / f
        if p.exists():
            h.update(f.encode() + b"\0" + guard.sha256_file(p).encode() + b"\n")
    return h.hexdigest()


def append(name, text):
    with open(ART / name, "a") as f:
        f.write(text)


def run_log(rec):
    append("RUN_LOG.jsonl", json.dumps({"time": guard.now(), "git_commit": None,
                                        "git_note": "no git repository; code_tree_sha256 = hash of all slk code and "
                                                    "imported modules", "code_tree_sha256": code_hash(), **rec},
                                       default=str) + "\n")


def decision(observation, evidence, hypothesis, decision_, rejected="", prediction="", supported=""):
    append("DECISION_LOG.md", f"\n### {guard.now()}\n- Observation: {observation}\n- Evidence: {evidence}\n"
                              f"- Hypothesis: {hypothesis}\n- Decision: {decision_}\n"
                              + (f"- Rejected alternatives: {rejected}\n" if rejected else "")
                              + (f"- Prediction before next run: {prediction}\n" if prediction else "")
                              + (f"- Prediction supported?: {supported}\n" if supported else ""))


def deviation(text, effect, can_affect_conclusion):
    n = sum(1 for _ in open(ART / "DEVIATIONS.jsonl")) + 1 if (ART / "DEVIATIONS.jsonl").exists() else 1
    append("DEVIATIONS.jsonl", json.dumps({"n": n, "time": guard.now(), "text": text, "effect": effect,
                                           "can_affect_conclusion": can_affect_conclusion}) + "\n")


def results(line):
    if not (ART / "RESULTS_LIVE.md").exists():
        append("RESULTS_LIVE.md", "# RESULTS_LIVE (append-only; every completed result, favourable or not)\n\n"
                                  "| time | run | what | values |\n|---|---|---|---|\n")
    append("RESULTS_LIVE.md", f"| {guard.now()} | {line}\n")


def gpu_hours(category=None):
    p = ART / "compute_ledger.tsv"
    if not p.exists():
        return 0.0
    tot = 0.0
    for line in p.read_text().splitlines():
        f = line.split("\t")
        try:
            if category is None or (len(f) > 5 and f[5] in category):
                tot += float(f[2])
        except (IndexError, ValueError):
            pass
    return tot / 3600


def write_resume(state, cmd, notes=""):
    txt = (f"# RESUME — csr-semantic-edl-long-adaptive-k-v1\n\nUpdated {guard.now()}.\n\n**State:** {state}\n\n"
           f"GPU used: {gpu_hours():.2f} GPU-h (cap {GPU_CAP_H['total']}). Disk free: "
           f"{guard.disk_free_bytes() / 2**30:.1f} GiB (reserve 100 GiB).\n\n"
           f"**Resume command (only after explicit permission if the state says INCOMPLETE/external stop):**\n\n"
           f"```\ncd {guard.ROOT} && source env.sh && {cmd}\n```\n\n{notes}\n")
    guard.atomic_write_bytes(ART / "RESUME.md", txt.encode(), overwrite=True)


# ------------------------------------------------------------------------------------------------ runs
def run_dirs():
    d = json.loads(RUNS.read_text()) if RUNS.exists() else {}
    for k, v in REUSED.items():
        d.setdefault(f"P1_{k}", v)
    return d


def remember(rn):
    d = json.loads(RUNS.read_text()) if RUNS.exists() else {}
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


def job(job_id, argv, phase, kind, run_name=None, seed=None, cfg_p=None, expect=(0,), disk_gb=2.0):
    from slk_driver import load_state, run_job
    if (load_state()["jobs"].get(job_id) or {}).get("status") == "COMPLETE":
        return 0
    pre = status_of(run_name) if run_name else None
    rec = {"phase": phase, "kind": kind, "job": job_id, "argv": argv, "run_name": run_name, "seed": seed,
           "config": str(cfg_p) if cfg_p else None,
           "config_sha256": guard.sha256_file(cfg_p) if cfg_p else None,
           "start_tokens": (pre or {}).get("seen", 0) if kind == "train" else None}
    try:
        rc = run_job(job_id, argv, gpu=True, expect=expect, disk_gb=disk_gb)
        ok = True
    except SystemExit as e:
        rc, ok = e.code, False
    js = load_state()["jobs"].get(job_id, {})
    if run_name and kind == "train":
        remember(run_name)
    post = status_of(run_name) if run_name else None
    rd = run_dirs().get(run_name) if run_name else None
    rec.update({"exit_status": js.get("status"), "rc": rc, "gpu_seconds": js.get("seconds"),
                "end_tokens": (post or {}).get("seen") if kind == "train" else None,
                "run_dir": rd, "run_status": (post or {}).get("status") if kind == "train" else None,
                "plan_sha256": (post or {}).get("plan_sha256") if kind == "train" else None,
                "checkpoints": {k: v["path"] for k, v in ((post or {}).get("ckpts") or {}).items()} if kind == "train"
                else None,
                "peak_vram_gib": (post or {}).get("peak_alloc_gib") if kind == "train" else None,
                "disk_free_gib": guard.disk_free_bytes() / 2**30,
                "run_dir_bytes": (int(subprocess.run(["du", "-sb", str(guard.ROOT / rd)], capture_output=True,
                                                     text=True).stdout.split()[0]) if rd and kind == "train" else None),
                "validity": "valid" if ok else ("INCOMPLETE (external interruption; not evidence)"
                                                if js.get("status") == "INCOMPLETE" else "FAILED (not evidence)")})
    run_log(rec)
    if not ok:
        write_resume(f"INCOMPLETE — job {job_id} ended {js.get('status')} (rc {rc}). No automatic relaunch.",
                     RESUME_CMD,
                     f"Last job: `{job_id}`; run dir: {rd}; partial runs are INCOMPLETE and are not evidence. "
                     "The trainer saved model/optimizer/RNG/data-cursor/driver state in its verified resumable slot; "
                     "`slk_run.py` resumes it in place (--run-dir).")
        raise SystemExit(rc if isinstance(rc, int) else 1)
    return rc


def drop_slot(rn, why):
    from csr.slk_ckpt import verify_dir
    rd = run_dirs().get(rn)
    st = status_of(rn)
    if not rd or not st:
        return
    for c in (st.get("ckpts") or {}).values():
        if not verify_dir(guard.ROOT / c["path"]):
            raise guard.GuardError(f"milestone {c['path']} fails verification; slot kept")
    sl = TMP / "slots" / Path(rd).name
    if sl.exists():
        nb = sum(p.stat().st_size for p in sl.rglob("*") if p.is_file())
        shutil.rmtree(sl)
        append("deleted_artifacts.tsv", f"{guard.now()}\t{sl.relative_to(guard.ROOT)}\t{nb}\t{why}\n")


# ------------------------------------------------------------------------------------------------ checks
def checks():
    from slk_driver import run_job
    import glob
    if not glob.glob(str(ART / "cpu_tests_2*.json")) or not glob.glob(str(ART / "cpu_tests_ported_*.json")):
        raise SystemExit("run the CPU suites first (slk_tests.py, slk_tests_ported.py)")
    for pat in ("cpu_tests_2*.json", "cpu_tests_ported_*.json"):
        last = sorted(glob.glob(str(ART / pat)))[-1]
        if not json.loads(Path(last).read_text())["all_ok"]:
            raise SystemExit(f"CPU suite failed: {last}")
    job("gpu_checks", ["slk_gpu_checks.py"], "checks", "check", disk_gb=6.0)
    g = json.loads((ART / "gpu_checks.json").read_text())
    if not g["all_pass"]:
        raise SystemExit("GPU checks failed; see gpu_checks.json")
    results(f"gpu_checks | equivalence/resume | T-EQ step0 bit-identical {g['teq']['step0_bit_identical']} "
            f"(d50 {g['teq']['d_step50']:+.4f}, d100 {g['teq']['d_step100']:+.4f}); beta0 step0 identical "
            f"{g['beta0']['step0_bit_identical_to_T4']}; resume diff {g['c6_resume']['max_abs_dlm_resumed_vs_straight']:.2e}"
            f" (noise {g['c6_resume']['run_to_run_noise']:.2e}); params equal {g['params_equal']} |")


# ------------------------------------------------------------------------------------------------ phase 1
def p1_eval_estimate(n_seeds):
    # hours: p100 full+pe100k ~0.5, p040 pe50k ~0.3, s50 pe50k ~0.3 (C6/SHUF), plain audit ~0.07
    n_runs = 4 * n_seeds
    plain = 2 * n_seeds + 2 * n_runs + (2 if n_seeds > 3 else 1) + 1
    return n_runs * 0.5 + n_runs * 0.3 + 2 * n_seeds * 0.3 + plain * 0.07


def p1train():
    from slk_configs import make, phase1_conds
    conds = phase1_conds()
    seed3 = True
    for cond, seed in P1_ORDER:
        rn = f"P1_{cond}_s{seed}"
        if seed == 3 and seed3 and cond == "C6":
            st = status_of(rn)
            if not (st and st.get("status") == "COMPLETE"):
                used = gpu_hours()
                proj = used + 4 * EST_TRAIN_H + p1_eval_estimate(4)
                if proj > GPU_CAP_H["phase1"]:
                    seed3 = False
                    decision("Budget check before the seed-3 group", f"used {used:.2f} h; projected Phase-1 total "
                             f"{proj:.2f} h > cap {GPU_CAP_H['phase1']}", "four seeds do not fit",
                             "drop seed 3 (prereg section 10 drop order); final comparison uses 3 seed pairs")
                else:
                    decision("Budget check before the seed-3 group", f"used {used:.2f} h; projected Phase-1 total "
                             f"{proj:.2f} h <= cap {GPU_CAP_H['phase1']}", "four seed pairs fit the cap",
                             "train seed 3 (C6, SHUF, T4, EDL)")
        if seed == 3 and not seed3:
            continue
        st = status_of(rn)
        if st and st.get("status") == "COMPLETE":
            continue
        p, cfg = make(rn, seed, conds[cond], "final", diag_every=100)
        for leg in range(4):
            prev = run_dirs().get(rn) if rn in (json.loads(RUNS.read_text()) if RUNS.exists() else {}) else None
            argv = ["slk_train.py", "--config", str(p)] + (["--run-dir", prev] if prev else [])
            job(f"train_{rn}_leg{leg}", argv, "phase1", "train", rn, seed, p, expect=(0, 75), disk_gb=4.0)
            st = status_of(rn)
            if st and st.get("status") == "COMPLETE":
                break
        st = status_of(rn)
        if not st or st.get("status") != "COMPLETE":
            raise SystemExit(f"{rn} not COMPLETE after 4 legs")
        results(f"{rn} | train 245M | status {st['status']}; tokens {st['seen']}; steps {st['steps_done']}; "
                f"tok/s {st['tokens_per_s']:.0f}; peak {st['peak_alloc_gib']:.1f} GiB |")
        write_resume(f"Phase 1 training in progress; last completed {rn}.", RESUME_CMD)


def p1_runs():
    out = {}
    for seed in (0, 1, 2, 3):
        for cond in ("C6", "SHUF", "T4", "EDL"):
            rn = f"P1_{cond}_s{seed}"
            st = status_of(rn)
            if st and st.get("status") == "COMPLETE":
                out[rn] = run_dirs()[rn]
    return out


def p1eval():
    runs = p1_runs()
    seeds = sorted({int(r.split("_s")[-1]) for r in runs if r.startswith("P1_C6")})
    complete = all(f"P1_{c}_s{s}" in runs for s in seeds for c in ("C6", "SHUF", "T4", "EDL"))
    if not complete or len(seeds) < 3:
        raise SystemExit(f"Phase-1 training incomplete: {sorted(runs)}")
    vo = ART / "PHASE1_VAL_OPEN.json"
    if not vo.exists():
        guard.atomic_write_json(vo, {"time": guard.now(), "seeds": seeds, "runs": runs,
                                     "note": "VAL opened once for the Phase-1 evaluation; no Phase-1 configuration is "
                                             "selected from data"}, mode=0o400)
        decision("All Phase-1 training complete", json.dumps(runs), "-", "open VAL (PHASE1_VAL_OPEN.json) and run the "
                 "preregistered evaluations in priority order", prediction="see DECISION_LOG Phase-1 predictions")
    order = []
    for s in seeds:
        for c in ("C6", "SHUF", "T4", "EDL"):
            order.append((f"P1_{c}_s{s}", "p100", ["--full", "--pe", "100000"], 0.5))
    for s in seeds:
        for c in ("C6", "SHUF", "T4", "EDL"):
            order.append((f"P1_{c}_s{s}", "p040", ["--pe", "50000"], 0.3))
    for s in seeds:
        for c in ("C6", "SHUF"):
            order.append((f"P1_{c}_s{s}", "s50", ["--pe", "50000"], 0.3))
    for s in seeds:
        for c in ("C6", "SHUF", "T4", "EDL"):
            rn = f"P1_{c}_s{s}"
            ck = json.loads((guard.ROOT / runs[rn] / "status.json").read_text())["ckpts"]
            for tag in ("s20", "s50", "p060", "p080"):
                if tag in ck and not (tag == "s50" and c in ("C6", "SHUF")):
                    order.append((rn, tag, [], 0.07))
    for rn, tag, extra, est in order:
        suffix = ("_full" if "--full" in extra else "") + (f"_pe{int(extra[extra.index('--pe') + 1]) // 1000}k"
                                                          if "--pe" in extra else "")
        outp = ART / "evals" / rn / f"val_{tag}{suffix}.json"
        if outp.exists():
            continue  # completed earlier (its RESULTS_LIVE row already exists); never re-run or re-logged
        if gpu_hours() + est > GPU_CAP_H["phase1"] + 0.0 and (tag in ("s50", "p060", "p080") or "--pe" not in extra):
            decision("Phase-1 budget cap reached", f"used {gpu_hours():.2f} h", "-", f"skip {rn} {tag} (drop order)")
            continue
        if True:
            job(f"val_{rn}_{tag}{suffix}", ["slk_evaluate.py", "val", "--run-dir", runs[rn], "--run-name", rn, "--tag",
                                           tag] + extra, "phase1", "eval", rn, int(rn.split("_s")[-1]), disk_gb=0.5)
        e = json.loads(outp.read_text())
        r = e["routing"]["domain"]
        line = (f"{rn} | VAL {tag}{suffix} | CE {e['val']['ce']:.4f}; domain NMI {r['nmi_mean']:.4f} (route-null "
                f"{r['nmi_null_route_mean']:.4f}); probe {r['probe_logreg']['acc']:.3f}; domain FSI {r['fsi']:.5f}; "
                f"coherence {e['coherence_e5']:.4f}")
        if "per_expert" in e:
            pe = e["per_expert"]["domain"]
            line += (f"; per-expert domain interaction {pe['mean_interaction']:.6f} [{pe['mean_interaction_ci95'][0]:.6f},"
                     f" {pe['mean_interaction_ci95'][1]:.6f}]; specialised experts {pe['n_subject_specialized']}")
        results(line + " |")
        write_resume(f"Phase-1 evaluation in progress; last {rn} {tag}.", RESUME_CMD)


def resume_through_2a():
    """Stage-limited resume (explicit user instruction required to run it). Archives the user STOP marker (renamed,
    never deleted), clears a recorded external stop, then finishes Phase 1 (completed jobs/outputs are skipped) and
    Phase 2A; slk_phase2.run_all(stage_limit=True) never imports or dispatches Phase 2B."""
    from slk_driver import update_state
    from csr.slk_paths import STOP_MARKER
    t = guard.utc_stamp()
    cleared = {}
    if STOP_MARKER.exists():
        dst = STOP_MARKER.with_name(f"STOP.cleared_{t}.json")
        STOP_MARKER.rename(dst)
        cleared["stop_marker"] = str(dst.relative_to(guard.ROOT))
    st = update_state(lambda s_: (s_.setdefault("authorized_resumes", []).append(
        {"time": guard.now(), "cleared": s_.pop("external_stop", None), "user_stop": s_.pop("user_stop", None),
         "via": "slk_run.py resume-through-2a"}), s_.__setitem__("state", "running")))
    cleared["driver"] = st["authorized_resumes"][-1]
    run_log({"phase": "control", "kind": "resume", "job": "resume-through-2a", "cleared": cleared,
             "gpu_seconds": 0, "validity": "operational"})
    decision("Stage-limited resume (explicit user instruction)", json.dumps(cleared, default=str), "-",
             "finish remaining Phase-1 evaluations and Phase 2A; stop before Phase 2B")
    from slk_driver import load_state
    if (load_state()["jobs"].get("gpu_checks") or {}).get("status") != "COMPLETE":
        checks()
    p1train()
    p1eval()
    import slk_phase2
    slk_phase2.run_all(stage_limit=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["checks", "p1train", "p1eval", "phase1", "all", "resume-through-2a"])
    a = ap.parse_args()
    guard.check_root()
    from csr.slk_paths import check_prereg
    check_prereg()
    if a.cmd == "resume-through-2a":
        resume_through_2a()
        return
    if a.cmd in ("checks", "phase1", "all"):
        checks()
    if a.cmd in ("p1train", "phase1", "all"):
        p1train()
    if a.cmd in ("p1eval", "phase1", "all"):
        p1eval()
    if a.cmd == "all":
        import slk_phase2
        slk_phase2.run_all()


if __name__ == "__main__":
    main()
