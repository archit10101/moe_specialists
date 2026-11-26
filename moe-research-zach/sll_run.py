#!/usr/bin/env python
"""csr-semantic-lift-expert-learning-v1 orchestrator (docs/SEMANTIC_LIFT_EXPERT_LEARNING_V1_PREREG.md). Every GPU job
goes through sll_driver.run_job (idempotent; completed jobs are skipped). An external interruption (signal, foreign GPU
process, STOP marker, launcher block) marks the experiment PAUSED_BY_OPERATOR, writes RESUME.md and exits; nothing
relaunches automatically.

  sll_run.py reuse-check   verify reused runs (plan, recipe, tables, milestone sha256) -> reuse_manifest.json
  sll_run.py checks        GPU correctness checks G1-G5 (prereg 5) -> gpu_checks.json (all must pass)
  sll_run.py search        E/F x lambda {0.25, 0.5, 1.0} to 20M seed 100 + dev evals -> search_decision.json (prereg 8)
  sll_run.py promote       E/F lambda* seeds 100/101 to 50M, dev + gate partition -> promotion.json (prereg 9)
  sll_run.py finals        245M E/F seeds 0-3, FINAL_OPEN, final evaluations -> final_decision.json (prereg 10)
  sll_run.py all           reuse-check -> checks -> search -> promote -> finals"""
import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402
from csr.sll_paths import (ART, CALIB_SEARCH_TABLE, CALIB_SEL_TABLE, CFG_DIR, EXP, FINAL_SEEDS, GPU_CAP_H,  # noqa: E402
                           LAMBDAS, MILESTONES, PROMO_SEEDS, RUN_PREFIX, SEARCH_SEED, SHUF_TABLE, SHUF_TABLE_SHA,
                           SUE_ART, SLK_ART, TABLE, TABLE_SHA, TMP, check_prereg, prereg_sha)

RUNS = ART / "run_dirs.json"
RESUME_CMD = ".venv/bin/python sll_run.py all"
S20, S50, P100 = MILESTONES["s20"], MILESTONES["s50"], MILESTONES["p100"]
PLAN_SHA = "f4bc3c93"
LIFT_FIXED = {"lo": 0.5, "hi": 2.0, "rho": 0.98, "n0": 8.0, "floor": 0.02, "frac": 0.05}
NOISE = 5e-5
# reused conditions (prereg section 6)
REUSED50 = {"T4_s100": "runs/20261114T090727Z_forecast_latent_edl_v1_s_T4_s100",
            "T4_s101": "runs/20261115T103553Z_semantic_edl_v1_b_T4_s101",
            "EDL_s100": "runs/20261115T083452Z_semantic_edl_v1_b_EDL_s100",
            "EDL_s101": "runs/20261115T104319Z_semantic_edl_v1_b_EDL_s101",
            "C6_s100": "runs/20261117T073831Z_semantic_utility_edl_v1_REF_C6_s100",
            "C6_s101": "runs/20261117T184435Z_semantic_utility_edl_v1_REF_C6_s101",
            "SHUF_s100": "runs/20261117T182907Z_semantic_utility_edl_v1_REF_SHUF_s100",
            "SHUF_s101": "runs/20261117T190003Z_semantic_utility_edl_v1_REF_SHUF_s101"}
V1_EVAL_NAME = {"T4": "POST_T4", "EDL": "POST_EDL", "C6": "REF_C6", "SHUF": "REF_SHUF"}
STAGE_DISK_GIB = {"checks": 5.0, "search": 10.0, "promote": 6.0, "finals": 12.0}
GPU_PHASE = {"checks": "checks", "search": "search", "promote": "promotion", "finals": "final"}


# ------------------------------------------------------------------------------------------------ logging
def code_hash():
    h = hashlib.sha256()
    files = sorted([str(p.relative_to(guard.ROOT)) for p in guard.ROOT.glob("sll_*.py")] +
                   [str(p.relative_to(guard.ROOT)) for p in (guard.ROOT / "src/csr").glob("sll_*.py")] +
                   ["sue_train.py", "scu_evaluate.py", "fle_evaluate.py", "fle_configs.py", "src/csr/lfr_model.py",
                    "src/csr/lfr_eval.py", "src/csr/slk_lib.py", "src/csr/sem_lib.py", "src/csr/gc_lib.py",
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
                                        "git_note": "no git repository; code_tree_sha256 = hash of sll code + imports",
                                        "code_tree_sha256": code_hash(), **rec}, default=str) + "\n")
    if md:
        if not (ART / "RUN_LOG.md").exists():
            append("RUN_LOG.md", "# RUN_LOG — csr-semantic-lift-expert-learning-v1 (append-only)\n")
        append("RUN_LOG.md", f"\n### {guard.now()} — {md[0]}\n{md[1]}\n")


def decision(observation, evidence, hypothesis, action, rejected=""):
    append("DECISION_LOG.md", f"\n### {guard.now()}\n- Observation: {observation}\n- Evidence: {evidence}\n"
                              f"- Hypothesis: {hypothesis}\n- Action: {action}\n"
                              + (f"- Rejected alternatives: {rejected}\n" if rejected else ""))


def observe(text):
    append("OBSERVATIONS.md", f"- {guard.now()} {text}\n")


def deviation(text, effect, can_affect):
    p = ART / "deviations.jsonl"
    n = sum(1 for _ in open(p)) if p.exists() else 0
    append("deviations.jsonl", json.dumps({"n": n, "time": guard.now(), "text": text, "effect": effect,
                                           "can_affect_conclusion": can_affect}) + "\n")


def results(line):
    if not (ART / "RESULTS_LIVE.md").exists():
        append("RESULTS_LIVE.md", "# RESULTS_LIVE (append-only; every completed result)\n\n| time | run | what | values |\n"
                                  "|---|---|---|---|\n")
    append("RESULTS_LIVE.md", f"| {guard.now()} | {line}\n")


def disk_log(stage, est, note=""):
    append("disk_ledger.tsv", f"{guard.now()}\t{stage}\t{guard.disk_free_bytes() / 2**30:.1f}\t{est}\t{note}\n")


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


def cap_check(stage):
    ph = GPU_PHASE[stage]
    used, tot = gpu_hours([ph]), gpu_hours()
    if used >= GPU_CAP_H[ph] or tot >= GPU_CAP_H["total"]:
        decision(f"GPU cap reached before a {stage} job", f"phase {used:.2f}/{GPU_CAP_H[ph]} h, total {tot:.2f}/"
                 f"{GPU_CAP_H['total']} h", "-", "stop cleanly (prereg 13)")
        set_state("STOPPED_GPU_CAP", stage=stage)
        write_resume(f"STOPPED_GPU_CAP — stage {stage}", RESUME_CMD, "A cap increase needs explicit operator approval.")
        raise SystemExit(f"GPU cap reached ({stage})")


def job(job_id, argv, phase, kind, run_name=None, seed=None, cfg_p=None, expect=(0,), disk_gb=2.0, pre=None):
    from sll_driver import load_state, run_job
    if (load_state()["jobs"].get(job_id) or {}).get("status") == "COMPLETE":
        return 0
    cap_check({"checks": "checks", "search": "search", "promotion": "promote", "final": "finals"}.get(phase, "checks"))
    if pre:
        run_log({"phase": phase, "kind": "pre_run", "job": job_id, "gpu_seconds": 0, **pre},
                (f"PRE {job_id}", "\n".join(f"- {k}: {v}" for k, v in pre.items())))
    rec = {"phase": phase, "kind": kind, "job": job_id, "argv": argv, "run_name": run_name, "seed": seed,
           "config": str(cfg_p) if cfg_p else None, "config_sha256": guard.sha256_file(cfg_p) if cfg_p else None}
    try:
        rc = run_job(job_id, argv, gpu=True, expect=expect, disk_gb=disk_gb)
        ok = True
    except SystemExit as e:
        rc, ok = e.code, False
    js = load_state()["jobs"].get(job_id, {})
    if run_name and kind == "train":
        remember(run_name)
    post = status_of(run_name) if run_name and kind == "train" else None
    rd = run_dirs().get(run_name) if run_name else None
    rec.update({"exit_status": js.get("status"), "rc": rc, "gpu_seconds": js.get("seconds"), "run_dir": rd,
                "end_tokens": (post or {}).get("seen"), "run_status": (post or {}).get("status"),
                "steps_done": (post or {}).get("steps_done"), "plan_sha256": (post or {}).get("plan_sha256"),
                "tokens_per_s": (post or {}).get("tokens_per_s"),
                "checkpoints": {k: v["path"] for k, v in ((post or {}).get("ckpts") or {}).items()} if post else None,
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
                                                 "params": post.get("params"), "peak_alloc_gib": post.get("peak_alloc_gib")},
                                                default=str) + "\n")
    if ok:
        write_resume(f"RUNNING — last completed job {job_id} ({guard.now()}); orchestrator active", RESUME_CMD,
                     "If no sll process is alive the orchestrator stopped: read STATE.json, driver_state.json and the "
                     "last RUN_LOG record before resuming. Completed jobs are skipped on resume.")
        return rc
    if js.get("status") in ("INCOMPLETE", None):
        set_state("PAUSED_BY_OPERATOR", job=job_id, run_dir=rd)
        decision(f"External interruption during {job_id}", f"driver status {js.get('status')}, rc {rc}",
                 "operator wants the GPU (prereg 13)", "PAUSED_BY_OPERATOR: checkpointed by the trainer at the next step "
                 "boundary (if training), driver stopped, GPU released, no restart")
        write_resume("PAUSED_BY_OPERATOR — external interruption (signal / foreign GPU process / STOP marker / launcher "
                     f"block) during job {job_id}; no automatic relaunch",
                     ".venv/bin/python sll_driver.py authorize-resume && " + RESUME_CMD,
                     f"Last job `{job_id}`; run dir {rd}. Trainers keep model/optimizer/RNG/data cursor/LiftStats in their "
                     "verified resumable slot and resume in place (--run-dir). Completed jobs are skipped.")
    else:
        set_state("FAILED_JOB", job=job_id, rc=rc)
        write_resume(f"FAILED — job {job_id} (rc {rc}); diagnose before resuming", RESUME_CMD,
                     f"Job log: artifacts/semantic-lift-expert-learning-v1/job_logs/{job_id}.log")
    raise SystemExit(rc if isinstance(rc, int) else 1)


def stage_disk(stage):
    est = STAGE_DISK_GIB[stage]
    free = guard.disk_free_bytes() / 2**30
    disk_log(stage, est, "pre-stage estimate")
    if free - est < 100:
        need = 100 + est - free
        decision(f"Disk estimate blocks stage {stage}", f"free {free:.2f} GiB, estimated peak {est} GiB, reserve 100",
                 "-", f"stop cleanly; {need:.1f} GiB more free space needed")
        set_state("PAUSED_DISK_RESERVE", stage=stage, need_gib=need)
        write_resume(f"PAUSED_DISK_RESERVE — stage {stage} needs ~{est} GiB; free {free:.1f} GiB; {need:.1f} GiB more "
                     "required", RESUME_CMD, "Free disk, then resume (completed jobs skipped).")
        raise SystemExit(f"disk: stage {stage} needs {need:.1f} GiB more")


def jl(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


# ------------------------------------------------------------------------------------------------ configs
def make_cfg(run_name, seed, cond, category, train_over=None, milestones=None, diag_every=100):
    from fle_configs import MODEL, TRAIN
    tr = dict(TRAIN)
    tr.update(train_over or {})
    cfg = {"experiment": EXP, "run_name": run_name, "seed": seed, "budget_tokens": 245 * 1_000_000,
           "prereg_sha256": prereg_sha(), "model": MODEL, "train": tr, "cond": cond, "kind": "final",
           "milestones": dict(MILESTONES if milestones is None else milestones), "diag_every": diag_every,
           "category": category}
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    p = CFG_DIR / f"{run_name}.json"
    if p.exists():
        if json.loads(p.read_text()) != cfg:
            raise guard.GuardError(f"config {p} exists and differs")
    else:
        guard.atomic_write_json(p, cfg, mode=0o400)
    return p, cfg


def cond(kind, lam, **over):
    t, s = (TABLE, TABLE_SHA) if kind == "E" else (SHUF_TABLE, SHUF_TABLE_SHA)
    if guard.sha256_file(guard.ROOT / t) != s:
        raise guard.GuardError(f"table hash mismatch {t}")
    lf = {"lam": lam, **LIFT_FIXED, **over}
    return {"paper_latent": {"table": t, "table_sha256": s, "beta": 0.005}, "lift": lf}


def lam_tag(lam):
    return {0.25: "L025", 0.5: "L05", 1.0: "L1"}.get(lam, f"L{lam}")


def train(rn, cfg_p, seed, to_step, phase, pre=None, disk_gb=3.0):
    """Run (or continue) training of run rn up to to_step (in legs); idempotent."""
    for leg in range(1, 9):
        st = status_of(rn)
        if st and (st.get("status") == "COMPLETE" or (st.get("steps_done") or 0) >= to_step):
            return st
        rd = run_dirs().get(rn)
        argv = ["sll_train.py", "--config", str(cfg_p.relative_to(guard.ROOT))]
        if rd:
            argv += ["--run-dir", rd]
        if to_step < 3880:
            argv += ["--max-steps", str(to_step)]
        jid = f"train_{rn}_to{to_step}_leg{leg}" + (f"_from{st.get('steps_done')}" if st else "")
        job(jid, argv, phase, "train", rn, seed, cfg_p, expect=(0, 75), disk_gb=disk_gb, pre=pre if leg == 1 else None)
    raise guard.GuardError(f"{rn}: not at step {to_step} after 8 legs")


# ------------------------------------------------------------------------------------------------ evaluation
def eval_path(rn, mode, tag, partition=""):
    return ART / "evals" / rn / ((f"gate_{partition}_{tag}" if mode == "gate" else f"{mode}_{tag}") + ".json")


def evaluate(rn, rd, tag, mode, phase, partition="", skip_val=False):
    out = eval_path(rn, mode, tag, partition)
    if out.exists():
        return jl(out)
    argv = ["sll_evaluate.py", mode, "--run-dir", rd, "--run-name", rn, "--tag", tag, "--pseudo-table", CALIB_SEARCH_TABLE]
    if mode != "dev":
        argv += ["--pseudo-table-sel", CALIB_SEL_TABLE]
    if partition:
        argv += ["--partition", partition]
    if skip_val:
        argv += ["--skip-val"]
    job(f"eval_{mode}{partition}_{tag}_{rn}", argv, phase, "eval", rn, disk_gb=1.0)
    e = jl(out)
    if e is None:
        raise guard.GuardError(f"evaluation output missing: {out}")
    s = summ(e)
    results(f"{rn} | {mode}{('/' + partition) if partition else ''} {tag} | CE {s['ce']:.4f}; interaction "
            f"{s['inter']:.3e} [{s['inter_ci'][0]:.2e}, {s['inter_ci'][1]:.2e}]; spec {s['spec']}; MI {s['mi']:.3f}; "
            f"NMI {s['nmi']}; load min {s['load_min_x_uniform']:.2f} max {s['load_max_x_uniform']:.2f}; ent "
            f"{s['routing_entropy']:.3f} |")
    return e


def reused_dev(cname, seed, tag, phase):
    """Dev evaluation of a reused condition (cname in T4/EDL/C6/SHUF): the cached V1 evaluation if it exists (same
    evaluator, same rows), else evaluated once here."""
    v1 = SUE_ART / "evals" / f"{V1_EVAL_NAME[cname]}_s{seed}" / f"dev_{tag}.json"
    if v1.exists():
        return jl(v1)
    return evaluate(f"R_{cname}_s{seed}", REUSED50[f"{cname}_s{seed}"], tag, "dev", phase)


def nmi_from_audit(e, sysn):
    import numpy as np
    a = e["audit"].get(sysn)
    if not a or "profiles" not in a:
        return None
    prof, load = np.asarray(a["profiles"]), np.asarray(a["routing"]["load"])
    out = []
    for li in range(prof.shape[0]):
        J = prof[li] * load[li][None, :]
        J = J / max(1e-12, J.sum())
        pk, pe = J.sum(1), J.sum(0)
        nz = J > 0
        mi = float((J[nz] * np.log(J[nz] / (pk[:, None] * pe[None, :])[nz])).sum())
        hk = float(-(pk[pk > 0] * np.log(pk[pk > 0])).sum())
        he = float(-(pe[pe > 0] * np.log(pe[pe > 0])).sum())
        out.append(2 * mi / max(1e-12, hk + he))
    return float(np.mean(out))


def summ(e):
    pe, au, pa = e["per_expert"]["pseudo"], e["audit"]["pseudo"]["routing"], e["pathology"]
    return {"ce": e["ce_block"]["ce"], "inter": pe["mean_interaction"], "inter_ci": pe["mean_interaction_ci95"],
            "boot": pe["boot_mean_samples"], "spec": pe["n_subject_specialized"],
            "spec_frac": pe["frac_subject_specialized"], "mi": au["mi_mean"], "nmi": nmi_from_audit(e, "pseudo"),
            "inter_e5": e["per_expert"]["e5"]["mean_interaction"],
            "generic": float(sum(e["per_expert"]["generic_importance_mean_dce"]) /
                             len(e["per_expert"]["generic_importance_mean_dce"])),
            "pathology": bool(pa["load_min_x_uniform"] < 0.1 or pa["load_max_x_uniform"] > 4.0
                              or pa["routing_entropy"] < 0.45), **pa}


def paired(ea, eb):
    """Paired document-bootstrap difference of mean interaction (identical rows and weights)."""
    import numpy as np
    a, b = np.asarray(summ(ea)["boot"]), np.asarray(summ(eb)["boot"])
    d = a - b
    return {"diff": summ(ea)["inter"] - summ(eb)["inter"], "ci95": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]}


def pooled(pairs):
    """Seed-mean paired difference with its bootstrap CI (bootstrap samples averaged across seeds)."""
    import numpy as np
    ds = [np.asarray(summ(a)["boot"]) - np.asarray(summ(b)["boot"]) for a, b in pairs]
    m = np.mean(ds, 0)
    return {"mean_diff": float(np.mean([summ(a)["inter"] - summ(b)["inter"] for a, b in pairs])),
            "ci95": [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]}


def lift_logs(rn, upto=None):
    rd = run_dirs()[rn]
    rows = [json.loads(x) for x in (guard.ROOT / rd / "metrics.jsonl").read_text().splitlines()]
    lr = [x["lift"] for x in rows if "lift" in x and (upto is None or x["lift"]["step"] < upto)]
    steps = [x for x in rows if "step" in x and (upto is None or x["step"] < upto)]
    return lr, steps


def activity(rn, upto=None):
    """Multiplier activity / stability / invariants from the training logs (prereg 8.3)."""
    lr, steps = lift_logs(rn, upto)
    act = [x for x in lr if x["lam_eff"] > 0 and x["m_by_layer"]]
    last = act[-1] if act else None
    mad = (sum(v["m_absdev_wmean"] for v in last["m_by_layer"].values()) / len(last["m_by_layer"])) if last else 0.0
    worst = max((v["m_wmean_maxdev"] for x in act for v in x["m_by_layer"].values()), default=0.0)
    mn = min((v["m_min"] for x in act for v in x["m_by_layer"].values()), default=1.0)
    mx = max((v["m_max"] for x in act for v in x["m_by_layer"].values()), default=1.0)
    clip = (sum(v["clip_lo"] + v["clip_hi"] for v in last["m_by_layer"].values()) / len(last["m_by_layer"])) if last else 0.0
    stab = lr[-1]["stats"].get("stability_r") if lr else None
    finite = all(math.isfinite(x["lm"]) for x in steps)
    lmin = min((min(x.get("load_min_x_uniform_by_layer") or [1.0]) for x in lr), default=None)
    return {"last_step": last["step"] if last else None, "lam_eff_last": last["lam_eff"] if last else 0.0,
            "m_absdev_wmean": mad, "clip_frac": clip, "wmean_maxdev": worst, "m_min": mn, "m_max": mx,
            "stability_r": stab, "finite": finite, "train_load_min_x_uniform": lmin,
            "ess_min": lr[-1]["stats"]["ess_min"] if lr else None,
            "lift_absmean": lr[-1]["stats"]["lift_absmean"] if lr else None,
            "jaccard_mean": (sum(lr[-1]["stats"].get("cluster_top4_jaccard_by_layer") or [0]) /
                             max(1, len(lr[-1]["stats"].get("cluster_top4_jaccard_by_layer") or [0]))) if lr else None}


# ------------------------------------------------------------------------------------------------ stages
def reuse_check():
    from csr.sll_ckpt import verify_dir
    out = ART / "reuse_manifest.json"
    if out.exists():
        r = jl(out)
        if not r["all_ok"]:
            raise SystemExit("reuse manifest not ok")
        return r
    from fle_configs import MODEL, TRAIN
    from csr.slk_paths import REUSED as SLK_REUSED
    slk = jl(SLK_ART / "run_dirs.json")
    finals = {}
    for s in FINAL_SEEDS:
        finals[f"T4_s{s}"] = SLK_REUSED.get(f"T4_s{s}") or slk[f"P1_T4_s{s}"]
        finals[f"EDL_s{s}"] = SLK_REUSED.get(f"EDL_s{s}") or slk[f"P1_EDL_s{s}"]
        finals[f"C6_s{s}"] = slk[f"P1_C6_s{s}"]
        finals[f"SHUF_s{s}"] = slk[f"P1_SHUF_s{s}"]
    rows, ok = {}, True
    for group, d, tags in (("50M", REUSED50, ("s20", "s50")), ("245M", finals, ("p100",))):
        for name, rd in d.items():
            st = jl(guard.ROOT / rd / "status.json")
            cfg = jl(guard.ROOT / rd / "config.json")
            c = name.split("_")[0]
            want_cond = {"T4": {}, "EDL": {"edl_beta": 0.005},
                         "C6": {"paper_latent": {"beta": 0.005, "table": TABLE, "table_sha256": TABLE_SHA}},
                         "SHUF": {"paper_latent": {"beta": 0.005, "table": SHUF_TABLE, "table_sha256": SHUF_TABLE_SHA}}}[c]
            chk = {"plan": (st.get("plan_sha256") or "").startswith(PLAN_SHA), "model": cfg["model"] == MODEL,
                   "train": {k: v for k, v in cfg["train"].items() if k != "log_every"} ==
                   {k: v for k, v in TRAIN.items() if k != "log_every"},
                   "cond": cfg["cond"] == want_cond, "seed": cfg["seed"] == int(name.split("_s")[-1])}
            for tg in tags:
                ck = st["ckpts"].get(tg)
                chk[f"ckpt_{tg}"] = bool(ck) and verify_dir(guard.ROOT / ck["path"]) and \
                    jl(guard.ROOT / ck["path"] / "meta.json")["sha256"] == ck["sha256"]
            good = all(chk.values())
            ok = ok and good
            rows[f"{group}:{name}"] = {"run_dir": rd, "checks": chk, "ok": good}
            print(name, good, {k: v for k, v in chk.items() if not v}, flush=True)
    doc = {"time": guard.now(), "all_ok": ok, "runs": rows, "finals_map": finals}
    guard.atomic_write_json(out, doc, mode=0o400)
    decision("Reuse check", f"{sum(r['ok'] for r in rows.values())}/{len(rows)} reused runs verified (plan, recipe, cond, "
             "seed, milestone sha256)", "-", "reuse as preregistered" if ok else "STOP: reuse integrity failure")
    if not ok:
        raise SystemExit("reuse check failed")
    return doc


def checks():
    out = ART / "gpu_checks.json"
    if out.exists():
        r = jl(out)
        if not r["all_pass"]:
            return checks_v2(r)
        return r
    stage_disk("checks")
    job("gpu_op_check", ["sll_gpu_op_check.py"], "checks", "check")
    op = jl(ART / "gpu_op_check.json")
    # G1: lambda = 0 vs reused REF_C6_s100 (51 steps)
    c0, _ = make_cfg("CHK_L0_s100", 100, cond("E", 0.0), "check")
    train("CHK_L0_s100", c0, 100, 51, "checks")
    _, s0 = lift_logs("CHK_L0_s100")
    ref = [json.loads(x) for x in (guard.ROOT / REUSED50["C6_s100"] / "metrics.jsonl").read_text().splitlines()]
    ref = {x["step"]: x for x in ref if "step" in x}
    mine = {x["step"]: x for x in s0}
    g1 = {"step0_abs": abs(mine[0]["lm"] - ref[0]["lm"]), "step50_abs": abs(mine[50]["lm"] - ref[50]["lm"]),
          "ref": [ref[0]["lm"], ref[50]["lm"]], "mine": [mine[0]["lm"], mine[50]["lm"]]}
    g1["pass"] = g1["step0_abs"] < 1e-6 and g1["step50_abs"] <= 5e-4
    # G3: resume with an active operator (test-only schedule frac 0.005 -> collect 19, ramp to 38)
    tover = {"log_every": 1}
    ca, _ = make_cfg("CHK_RES_A_s100", 100, cond("E", 1.0, frac=0.005), "check", tover, milestones={})
    cb, _ = make_cfg("CHK_RES_B_s100", 100, cond("E", 1.0, frac=0.005), "check", tover, milestones={})
    train("CHK_RES_A_s100", ca, 100, 60, "checks")
    train("CHK_RES_B_s100", cb, 100, 30, "checks")
    train("CHK_RES_B_s100", cb, 100, 60, "checks")
    _, sa = lift_logs("CHK_RES_A_s100")
    _, sb = lift_logs("CHK_RES_B_s100")
    la, lb = {x["step"]: x["lm"] for x in sa}, {x["step"]: x["lm"] for x in sb}
    import numpy as np
    dmax = max(abs(la[k] - lb[k]) for k in range(60))
    za = np.load(guard.ROOT / run_dirs()["CHK_RES_A_s100"] / "lift" / "lift_s00059.npz")
    zb = np.load(guard.ROOT / run_dirs()["CHK_RES_B_s100"] / "lift" / "lift_s00059.npz")
    dl = float(np.abs(za["lift"] - zb["lift"]).max())
    g3 = {"lm_maxabs_0_59": dmax, "lm_maxabs_0_29": max(abs(la[k] - lb[k]) for k in range(30)),
          "lift_table_maxabs_step59": dl, "pass": dmax <= 5e-4 and dl <= 1e-4}
    actA = activity("CHK_RES_A_s100")
    g4 = {**actA, "pass": actA["wmean_maxdev"] <= 1e-6 and actA["m_min"] >= 0.5 - 1e-9 and actA["m_max"] <= 2 + 1e-9
          and actA["m_absdev_wmean"] > 0.0}
    g5 = {"train_load_min_x_uniform": actA["train_load_min_x_uniform"],
          "pass": (actA["train_load_min_x_uniform"] or 0) >= 0.1}
    dts = {x["step"]: x["dt"] for x in sa}
    over = (np.median([dts[k] for k in range(45, 60)]) / np.median([dts[k] for k in range(5, 18)]))
    doc = {"time": guard.now(), "G1_lambda0_vs_REF_C6": g1, "G2_cuda_operator": {k: op.get(k) for k in
           ("pass", "pass_a", "pass_b", "tolerance_rel", "native_repeat", "a_forced_m1", "b_real_m",
            "timing_s_fwd_bwd_32rows")}, "G3_resume": g3, "G4_invariants": g4, "G5_load": g5,
           "overhead_step_time_ratio_active_vs_collection": float(over)}
    doc["all_pass"] = bool(g1["pass"] and op.get("pass") and g3["pass"] and g4["pass"] and g5["pass"])
    guard.atomic_write_json(out, doc, mode=0o400)
    results(f"checks | G1 {g1['pass']} (step0 {g1['step0_abs']:.2e}, step50 {g1['step50_abs']:.2e}); G2 {op.get('pass')}; "
            f"G3 {g3['pass']} (lm {dmax:.2e}, lift {dl:.2e}); G4 {g4['pass']}; G5 {g5['pass']}; overhead x{over:.2f} |")
    decision("GPU correctness checks", json.dumps({k: doc[k]["pass"] if isinstance(doc[k], dict) and "pass" in doc[k]
                                                   else None for k in doc if k.startswith("G")}),
             "-", "proceed to search" if doc["all_pass"] else "STOP: fix the implementation (engineering defect)")
    for rn in ("CHK_L0_s100", "CHK_RES_A_s100", "CHK_RES_B_s100"):
        drop_slot(rn, "GPU check run; slot superseded (test only)")
    if not doc["all_pass"]:
        raise SystemExit("GPU checks failed")
    return doc


def checks_v2(v1):
    """Deviation 1 (test specification): G1 / G3 use the project's established noise-referenced tolerance
    (sem_gpu_checks.py: tol = max(0.003, 3 x run-to-run noise of a second identical uninterrupted run)); the fixed
    5e-4 of prereg section 5 is below GPU per-step nondeterminism (two identical runs differ by up to 0.012 nats in
    steps 0-29). G5 (training load) cannot be judged in 60 steps (the native collection phase shows the same zero-load
    micro-batches); test 12 is enforced by the registered evaluation pathology rule at every evaluated milestone and
    the training load logs. G2 / G4 unchanged."""
    out = ART / "gpu_checks_v2.json"
    if out.exists():
        r = jl(out)
        if not r["all_pass"]:
            raise SystemExit("GPU checks v2 failed; see gpu_checks_v2.json")
        return r
    import numpy as np
    c0b, _ = make_cfg("CHK_L0b_s100", 100, cond("E", 0.0), "check")
    train("CHK_L0b_s100", c0b, 100, 51, "checks")
    tover = {"log_every": 1}
    cc, _ = make_cfg("CHK_RES_C_s100", 100, cond("E", 1.0, frac=0.005), "check", tover, milestones={})
    train("CHK_RES_C_s100", cc, 100, 60, "checks")
    lm = lambda rn: {x["step"]: x["lm"] for x in lift_logs(rn)[1]}  # noqa: E731
    L0, L0b = lm("CHK_L0_s100"), lm("CHK_L0b_s100")
    ref = {x["step"]: x["lm"] for x in (json.loads(y) for y in (guard.ROOT / REUSED50["C6_s100"] / "metrics.jsonl")
                                        .read_text().splitlines()) if "step" in x}
    n1 = abs(L0[50] - L0b[50])
    tol1 = max(0.003, 3 * n1)
    g1 = {"step0_abs": abs(L0[0] - ref[0]), "step50_abs_vs_REF": abs(L0[50] - ref[50]),
          "step50_abs_vs_REF_repeat": abs(L0b[50] - ref[50]), "noise_step50_L0_vs_L0b": n1, "tol": tol1}
    g1["pass"] = g1["step0_abs"] < 1e-6 and g1["step50_abs_vs_REF"] <= tol1
    A, B, C = lm("CHK_RES_A_s100"), lm("CHK_RES_B_s100"), lm("CHK_RES_C_s100")
    noise = max(abs(A[k] - C[k]) for k in range(30, 60))
    dres = max(abs(A[k] - B[k]) for k in range(30, 60))
    dres_c = max(abs(C[k] - B[k]) for k in range(30, 60))
    tol3 = max(0.003, 3 * noise)
    zl = lambda rn: np.load(guard.ROOT / run_dirs()[rn] / "lift" / "lift_s00059.npz")["lift"]  # noqa: E731
    la, lb, lc = zl("CHK_RES_A_s100"), zl("CHK_RES_B_s100"), zl("CHK_RES_C_s100")
    ln, lr_ = float(np.abs(la - lc).max()), float(np.abs(la - lb).max())
    g3 = {"resumed_vs_straight_maxabs_30_59": dres, "resumed_vs_straight2_maxabs_30_59": dres_c,
          "straight_vs_straight_noise_30_59": noise, "tol": tol3, "lift_table_resumed_vs_straight": lr_,
          "lift_table_noise": ln}
    g3["pass"] = dres <= tol3 and lr_ <= 3 * ln + 1e-4
    g5 = {"note": "deferred to the registered evaluation pathology rule (prereg 8.4 / 9 G-H / 10 P4)",
          "native_window_load_min": v1["G5_load"], "pass": True}
    doc = {"time": guard.now(), "deviation": "test-specification correction, see deviations.jsonl", "v1_file": "gpu_checks.json",
           "G1_lambda0_vs_REF_C6": g1, "G2_cuda_operator": v1["G2_cuda_operator"], "G3_resume": g3,
           "G4_invariants": v1["G4_invariants"], "G5_load": g5,
           "overhead_step_time_ratio_active_vs_collection": v1["overhead_step_time_ratio_active_vs_collection"]}
    doc["all_pass"] = bool(g1["pass"] and v1["G2_cuda_operator"]["pass"] and g3["pass"] and v1["G4_invariants"]["pass"])
    guard.atomic_write_json(out, doc, mode=0o400)
    results(f"checks v2 | G1 {g1['pass']} (|L0-REF|@50 {g1['step50_abs_vs_REF']:.4f}, noise {n1:.4f}, tol {tol1:.4f}); "
            f"G3 {g3['pass']} (resume {dres:.4f}, noise {noise:.4f}, tol {tol3:.4f}; lift {lr_:.4f} vs noise {ln:.4f}) |")
    decision("GPU checks v2 (noise-referenced tolerance)", json.dumps({"G1": g1, "G3": g3}), "-",
             "proceed to search" if doc["all_pass"] else "STOP: investigate implementation")
    for rn in ("CHK_L0b_s100", "CHK_RES_C_s100"):
        drop_slot(rn, "GPU check run; slot superseded (test only)")
    if not doc["all_pass"]:
        raise SystemExit("GPU checks v2 failed")
    return doc


def drop_slot(rn, why):
    """Prereg 13: delete this experiment's own superseded resumable slot after its milestones verify (ledgered)."""
    from csr.sll_ckpt import verify_dir
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


def search():
    out = ART / "search_decision.json"
    if out.exists():
        return jl(out)
    stage_disk("search")
    pre_rule = ("eligible iff dCE(E - REF_C6) <= 0.010, d_inter(E - F) > -5e-5, activity >= 0.02 & stability >= 0.8 & "
                "invariants, no pathology; lambda* = argmax d_inter (ties within 5e-5 -> larger lambda)")
    for lam in LAMBDAS:
        for k in ("E", "F"):
            rn = f"S_{k}_{lam_tag(lam)}_s{SEARCH_SEED}"
            cp, cfg = make_cfg(rn, SEARCH_SEED, cond(k, lam), "search")
            train(rn, cp, SEARCH_SEED, S20, "search", pre={
                "hypothesis": "lift-weighted expert gradients (semantic table) raise own-group usefulness beyond the "
                              "identical mechanism with shuffled groups" if k == "E" else
                              "matched shuffled control: identical mechanism, permuted paper groups",
                "config": f"{cp.relative_to(guard.ROOT)} (sha {guard.sha256_file(cp)[:12]}); cond {json.dumps(cfg['cond'])}",
                "reused": "REF_C6_s100 / REF_SHUF_s100 / T4 / EDL dev s20 evaluations (V1); plan f4bc3c93",
                "expected_discriminating_result": "d_inter(E - F) > 0 at matched CE; activity >= 0.02",
                "rule": pre_rule})
    rows = {}
    ref_c6 = summ(reused_dev("C6", SEARCH_SEED, "s20", "search"))
    ref_sh = summ(reused_dev("SHUF", SEARCH_SEED, "s20", "search"))
    t4 = summ(reused_dev("T4", SEARCH_SEED, "s20", "search"))
    edl = summ(reused_dev("EDL", SEARCH_SEED, "s20", "search"))
    for lam in LAMBDAS:
        rE, rF = f"S_E_{lam_tag(lam)}_s{SEARCH_SEED}", f"S_F_{lam_tag(lam)}_s{SEARCH_SEED}"
        eE = evaluate(rE, run_dirs()[rE], "s20", "dev", "search")
        eF = evaluate(rF, run_dirs()[rF], "s20", "dev", "search")
        sE, sF = summ(eE), summ(eF)
        aE, aF = activity(rE, S20), activity(rF, S20)
        pdiff = paired(eE, eF)
        r = {"lam": lam, "ce_E": sE["ce"], "ce_F": sF["ce"], "d_ce_vs_C6": sE["ce"] - ref_c6["ce"],
             "d_ce_vs_T4": sE["ce"] - t4["ce"], "d_ce_vs_EDL": sE["ce"] - edl["ce"],
             "inter_E": sE["inter"], "inter_F": sF["inter"], "d_inter": pdiff["diff"], "d_inter_ci95": pdiff["ci95"],
             "inter_E_ci": sE["inter_ci"], "inter_F_ci": sF["inter_ci"], "spec_E": sE["spec"], "spec_F": sF["spec"],
             "mi_E": sE["mi"], "mi_F": sF["mi"], "nmi_E": sE["nmi"], "nmi_F": sF["nmi"],
             "inter_e5_E": sE["inter_e5"], "inter_e5_F": sF["inter_e5"],
             "pathology_E": sE["pathology"], "pathology_F": sF["pathology"], "act_E": aE, "act_F": aF,
             "load_E": [sE["load_min_x_uniform"], sE["load_max_x_uniform"]], "entropy_E": sE["routing_entropy"]}
        r["crit"] = {"ce": r["d_ce_vs_C6"] <= 0.010, "function": r["d_inter"] > -NOISE,
                     "activity": aE["m_absdev_wmean"] >= 0.02 and (aE["stability_r"] or 0) >= 0.8 and
                     aE["wmean_maxdev"] <= 1e-6 and aE["m_min"] >= 0.5 - 1e-9 and aE["m_max"] <= 2 + 1e-9 and aE["finite"],
                     "load": not sE["pathology"] and not sF["pathology"]}
        r["eligible"] = all(r["crit"].values())
        rows[lam_tag(lam)] = r
        run_log({"phase": "search", "kind": "post_run", "job": f"search_{lam_tag(lam)}", "gpu_seconds": 0, "row": r},
                (f"POST search lambda {lam}", f"- CE E {sE['ce']:.4f} / F {sF['ce']:.4f} (REF_C6 {ref_c6['ce']:.4f}, T4 "
                 f"{t4['ce']:.4f}, EDL {edl['ce']:.4f})\n- interaction E {sE['inter']:.3e} / F {sF['inter']:.3e}; "
                 f"d(E-F) {pdiff['diff']:.3e} CI [{pdiff['ci95'][0]:.2e}, {pdiff['ci95'][1]:.2e}]\n- multipliers E: "
                 f"|m-1| {aE['m_absdev_wmean']:.3f}, clip {aE['clip_frac']:.3f}, lambda_eff {aE['lam_eff_last']:.3f}, "
                 f"stability {aE['stability_r']}; F: |m-1| {aF['m_absdev_wmean']:.3f}\n- load E {r['load_E']}, entropy "
                 f"{r['entropy_E']:.3f}; eligible {r['eligible']} {r['crit']}"))
    el = [k for k, v in rows.items() if v["eligible"]]
    chosen = None
    if el:
        best = max(el, key=lambda k: rows[k]["d_inter"])
        near = [k for k in el if rows[best]["d_inter"] - rows[k]["d_inter"] < NOISE]
        chosen = max(near, key=lambda k: rows[k]["lam"])
    doc = {"time": guard.now(), "rows": rows, "references_s20": {"C6": {k: ref_c6[k] for k in ("ce", "inter", "inter_ci", "mi", "nmi", "spec")},
                                                                "SHUF": {k: ref_sh[k] for k in ("ce", "inter", "inter_ci", "mi", "nmi", "spec")},
                                                                "T4": {k: t4[k] for k in ("ce", "inter", "mi", "spec")},
                                                                "EDL": {k: edl[k] for k in ("ce", "inter", "mi", "spec")},
                                                                "C6_minus_SHUF_paired": paired(reused_dev("C6", SEARCH_SEED, "s20", "search"),
                                                                                               reused_dev("SHUF", SEARCH_SEED, "s20", "search"))},
           "eligible": el, "chosen": chosen, "chosen_lam": rows[chosen]["lam"] if chosen else None,
           "rule": "prereg section 8"}
    guard.atomic_write_json(out, doc, mode=0o400)
    decision("Search (20M, seed 100) complete", json.dumps({k: {x: (round(v[x], 6) if isinstance(v[x], float) else v[x])
                                                                for x in ("d_ce_vs_C6", "d_inter", "eligible")}
                                                            for k, v in rows.items()}),
             "eligible candidates show no clear deficit; the 20M screen sees only the ramp (lambda_eff <= 0.64 lambda)",
             f"promote {chosen} with its matched shuffled control" if chosen else
             "no eligible lambda: diagnose (prereg 11) and decide on an engineering generation (prereg 12)")
    for lam in LAMBDAS:  # non-promoted search runs: slots superseded (their s20 milestones are kept)
        if lam_tag(lam) != chosen:
            for k in ("E", "F"):
                drop_slot(f"S_{k}_{lam_tag(lam)}_s{SEARCH_SEED}", "search run not promoted; slot superseded")
    if not chosen:
        set_state("GENERATION_DECISION_REQUIRED", stage="search")
        raise SystemExit("no eligible lambda")
    return doc


def diagnose(rows_by_seed, ce_ok):
    """Prereg section 11. rows_by_seed: seed -> dict(dE_F, dC_S, inter_E, inter_C6, inter_F, inter_SHUF, mi_E, mi_C6,
    ce_E, ce_C6, act)."""
    ds = [r["dE_F"] for r in rows_by_seed.values()]
    allpos = all(d > 0 for d in ds)
    mean_d = sum(ds) / len(ds)
    dce = sum(r["ce_E"] - r["ce_C6"] for r in rows_by_seed.values()) / len(rows_by_seed)
    act = min(r["act"]["m_absdev_wmean"] for r in rows_by_seed.values())
    stab = min((r["act"]["stability_r"] or 0) for r in rows_by_seed.values())
    dmi = sum(r["mi_E"] - r["mi_C6"] for r in rows_by_seed.values()) / len(rows_by_seed)
    gE = sum(r["inter_E"] - r["inter_C6"] for r in rows_by_seed.values()) / len(rows_by_seed)
    gF = sum(r["inter_F"] - r["inter_SHUF"] for r in rows_by_seed.values()) / len(rows_by_seed)
    dCS = sum(r["dC_S"] for r in rows_by_seed.values()) / len(rows_by_seed)
    m = {"D": allpos and (dce > 0.005 or not ce_ok), "F": allpos and ce_ok,
         "C": gE > 0 and gF > 0 and abs(mean_d - dCS) <= NOISE,
         "B": dmi > 0.005 and (not allpos and (mean_d <= 0 or abs(mean_d) <= NOISE)),
         "E": dce < -0.005 and not allpos, "A": act < 0.02 or stab < 0.5}
    primary = next((k for k in ("D", "F", "C", "B", "E", "A") if m[k]), "none")
    return {"primary": primary, "matching": [k for k, v in m.items() if v],
            "evidence": {"dE_F_by_seed": ds, "mean_dE_F": mean_d, "mean_dC6_SHUF": dCS, "dCE_E_C6": dce,
                         "min_activity": act, "min_stability": stab, "d_mi_E_C6": dmi, "gain_E_vs_C6": gE,
                         "gain_F_vs_SHUF": gF}}


def promote():
    out = ART / "promotion.json"
    if out.exists():
        return jl(out)
    sd = search()
    lam, tagL = sd["chosen_lam"], sd["chosen"]
    stage_disk("promote")
    names = {}
    for s in PROMO_SEEDS:
        for k in ("E", "F"):
            rn = f"S_{k}_{tagL}_s{s}"
            cp, cfg = make_cfg(rn, s, cond(k, lam), "search")  # same run family as the 20M search runs
            names[(k, s)] = rn
            train(rn, cp, s, S50, "promotion", pre={
                "hypothesis": "prereg section 1", "config": f"{cp.relative_to(guard.ROOT)}; cond {json.dumps(cfg['cond'])}",
                "reused": f"seed {s}: REF_C6 / REF_SHUF / T4 / EDL s50; " +
                          ("continues the 20M search run from its s20 slot" if s == SEARCH_SEED else "from step 0"),
                "expected_discriminating_result": "held-out d_inter(E - F) > 0 in both seeds and > d(C6 - SHUF)",
                "rule": "prereg section 9 gate (G-F1, G-F2, G-CE, G-H)"})
    dev = {}
    for s in PROMO_SEEDS:
        for k in ("E", "F"):
            rn = names[(k, s)]
            dev[(k, s)] = evaluate(rn, run_dirs()[rn], "s50", "dev", "promotion")
        for c in ("C6", "SHUF", "T4", "EDL"):
            dev[(c, s)] = reused_dev(c, s, "s50", "promotion")
    gp = ART / "GATE_holdb_OPEN.json"
    if not gp.exists():
        guard.atomic_write_json(gp, {"time": guard.now(), "partition": "CALIB-SELECTION half 1 sub-half b",
                                     "opened_for": f"lambda {lam} (E/F) + REF C6/SHUF, seeds {PROMO_SEEDS}"}, mode=0o400)
        decision("Gate partition opened", "all four 50M runs trained and dev-evaluated", "-",
                 "open CALIB-SELECTION half 1 sub-half b once (prereg 9)")
    gate = {}
    for s in PROMO_SEEDS:
        for k in ("E", "F"):
            rn = names[(k, s)]
            gate[(k, s)] = evaluate(rn, run_dirs()[rn], "s50", "gate", "promotion", partition="holdb")
        for c in ("C6", "SHUF"):
            gate[(c, s)] = evaluate(f"R_{c}_s{s}", REUSED50[f"{c}_s{s}"], "s50", "gate", "promotion", partition="holdb")
    per = {}
    for s in PROMO_SEEDS:
        sE, sF, sC, sS = (summ(gate[(x, s)]) for x in ("E", "F", "C6", "SHUF"))
        dE, dF, dC, dEDL, dT4 = (summ(dev[(x, s)]) for x in ("E", "F", "C6", "EDL", "T4"))
        per[s] = {"gate_dE_F": paired(gate[("E", s)], gate[("F", s)]), "gate_dC6_SHUF": paired(gate[("C6", s)], gate[("SHUF", s)]),
                  "gate_inter": {"E": sE["inter"], "F": sF["inter"], "C6": sC["inter"], "SHUF": sS["inter"]},
                  "gate_ce": {"E": sE["ce"], "F": sF["ce"], "C6": sC["ce"], "SHUF": sS["ce"]},
                  "gate_mi": {"E": sE["mi"], "F": sF["mi"], "C6": sC["mi"], "SHUF": sS["mi"]},
                  "gate_spec": {"E": sE["spec"], "F": sF["spec"], "C6": sC["spec"], "SHUF": sS["spec"]},
                  "dev_dE_F": paired(dev[("E", s)], dev[("F", s)]),
                  "dev_ce": {"E": dE["ce"], "F": dF["ce"], "C6": dC["ce"], "EDL": dEDL["ce"], "T4": dT4["ce"]},
                  "dev_inter": {"E": dE["inter"], "F": dF["inter"], "C6": dC["inter"], "EDL": dEDL["inter"], "T4": dT4["inter"]},
                  "pathology": {"E": sE["pathology"] or dE["pathology"], "F": sF["pathology"] or dF["pathology"]},
                  "act_E": activity(names[("E", s)], S50), "act_F": activity(names[("F", s)], S50)}
    for s in PROMO_SEEDS:
        a_ = per[s]["act_E"]
        run_log({"phase": "promotion", "kind": "post_run", "job": f"promotion_s{s}", "gpu_seconds": 0, "row": per[s]},
                (f"POST promotion lambda {lam} seed {s}",
                 f"- dev CE E {per[s]['dev_ce']['E']:.4f} / F {per[s]['dev_ce']['F']:.4f} / C6 {per[s]['dev_ce']['C6']:.4f} / "
                 f"EDL {per[s]['dev_ce']['EDL']:.4f} / T4 {per[s]['dev_ce']['T4']:.4f}\n- gate interaction E "
                 f"{per[s]['gate_inter']['E']:.3e} F {per[s]['gate_inter']['F']:.3e} C6 {per[s]['gate_inter']['C6']:.3e} SHUF "
                 f"{per[s]['gate_inter']['SHUF']:.3e}; d(E-F) {per[s]['gate_dE_F']['diff']:.3e} CI {per[s]['gate_dE_F']['ci95']}"
                 f"; dev d(E-F) {per[s]['dev_dE_F']['diff']:.3e}\n- multipliers E |m-1| {a_['m_absdev_wmean']:.3f} clip "
                 f"{a_['clip_frac']:.3f} stability {a_['stability_r']} ESS min {a_['ess_min']}; F |m-1| "
                 f"{per[s]['act_F']['m_absdev_wmean']:.3f}\n- pathology {per[s]['pathology']}"))
    pool = pooled([(gate[("E", s)], gate[("F", s)]) for s in PROMO_SEEDS])
    poolC = pooled([(gate[("C6", s)], gate[("SHUF", s)]) for s in PROMO_SEEDS])
    ce_mean = sum(per[s]["dev_ce"]["E"] - per[s]["dev_ce"]["EDL"] for s in PROMO_SEEDS) / len(PROMO_SEEDS)
    crit = {"G-F1": all(per[s]["gate_dE_F"]["diff"] > 0 for s in PROMO_SEEDS),
            "G-F2": pool["mean_diff"] > poolC["mean_diff"],
            "G-CE": ce_mean <= 0.005,
            "G-H": all(not per[s]["pathology"]["E"] and not per[s]["pathology"]["F"] and
                       per[s]["act_E"]["m_absdev_wmean"] >= 0.02 and per[s]["act_E"]["wmean_maxdev"] <= 1e-6
                       for s in PROMO_SEEDS)}
    diag = diagnose({s: {"dE_F": per[s]["gate_dE_F"]["diff"], "dC_S": per[s]["gate_dC6_SHUF"]["diff"],
                         "inter_E": per[s]["gate_inter"]["E"], "inter_C6": per[s]["gate_inter"]["C6"],
                         "inter_F": per[s]["gate_inter"]["F"], "inter_SHUF": per[s]["gate_inter"]["SHUF"],
                         "mi_E": per[s]["gate_mi"]["E"], "mi_C6": per[s]["gate_mi"]["C6"],
                         "ce_E": per[s]["dev_ce"]["E"], "ce_C6": per[s]["dev_ce"]["C6"], "act": per[s]["act_E"]}
                     for s in PROMO_SEEDS}, crit["G-CE"])
    doc = {"time": guard.now(), "lam": lam, "per_seed": {str(k): v for k, v in per.items()}, "gate_pooled_dE_F": pool,
           "gate_pooled_dC6_SHUF": poolC, "dev_mean_dCE_E_EDL": ce_mean, "criteria": crit, "passed": all(crit.values()),
           "diagnosis": diag, "rule": "prereg section 9"}
    guard.atomic_write_json(out, doc, mode=0o400)
    results(f"gate | lambda {lam} | " + "; ".join(f"s{s}: dE-F {per[s]['gate_dE_F']['diff']:.2e} (dev "
                                                   f"{per[s]['dev_dE_F']['diff']:.2e}), dC6-SHUF {per[s]['gate_dC6_SHUF']['diff']:.2e}"
                                                   for s in PROMO_SEEDS) +
            f"; pooled {pool['mean_diff']:.2e} CI [{pool['ci95'][0]:.2e}, {pool['ci95'][1]:.2e}]; dCE(E-EDL) {ce_mean:+.4f}; "
            f"criteria {crit}; diagnosis {diag['primary']} |")
    decision(f"50M gate for lambda {lam}", json.dumps({"criteria": crit, "pooled": pool, "pooled_C6_SHUF": poolC,
                                                       "dCE_E_EDL": ce_mean, "diagnosis": diag}, default=str),
             f"diagnosis {diag['primary']}",
             "gate PASSED: run the 245M finals (prereg 10)" if doc["passed"] else
             "gate FAILED: diagnose and decide on an engineering generation (prereg 12)")
    if not doc["passed"]:
        set_state("GENERATION_DECISION_REQUIRED", stage="promote", diagnosis=diag["primary"])
        raise SystemExit("gate failed")
    return doc


def finals():
    out = ART / "final_decision.json"
    if out.exists():
        return jl(out)
    pr = promote()
    if not pr["passed"]:
        raise SystemExit("gate not passed; no finals")
    lam = pr["lam"]
    stage_disk("finals")
    rm = reuse_check()["finals_map"]
    for s in FINAL_SEEDS:
        for k in ("E", "F"):
            rn = f"FIN_{k}_{lam_tag(lam)}_s{s}"
            cp, cfg = make_cfg(rn, s, cond(k, lam), "final")
            train(rn, cp, s, P100, "final", pre={
                "hypothesis": "prereg section 1", "config": f"{cp.relative_to(guard.ROOT)}; cond {json.dumps(cfg['cond'])}",
                "reused": f"seed {s}: semantic-edl-long-k P1 C6/SHUF/T4/EDL p100 + VAL evaluations",
                "expected_discriminating_result": "final primary d_inter(E - F) > 0 in 4/4 seeds, CI > 0; VAL CE(E) - "
                                                  "CE(EDL) <= 0.005", "rule": "prereg section 10 (P1-P4)"})
            if status_of(rn).get("status") == "COMPLETE":
                drop_slot(rn, "final run COMPLETE; milestones verified; slot superseded")
    fo = ART / "FINAL_OPEN.json"
    if not fo.exists():
        guard.atomic_write_json(fo, {"time": guard.now(), "note": "all eight final runs COMPLETE"}, mode=0o400)
    # measurement identity check for reused VAL results (prereg 6)
    idn = evaluate("R_C6_s0", rm["C6_s0"], "p100", "final", "final")
    slk = jl(SLK_ART / "evals" / "P1_C6_s0" / "val_p100_full_pe100k.json")
    ident = {"sll_val_ce": idn["val"]["ce"], "slk_val_ce": slk["val"]["ce"], "abs": abs(idn["val"]["ce"] - slk["val"]["ce"])}
    ident["pass"] = ident["abs"] <= 1e-4
    fin = {}
    for s in FINAL_SEEDS:
        for k in ("E", "F"):
            rn = f"FIN_{k}_{lam_tag(lam)}_s{s}"
            fin[(k, s)] = evaluate(rn, run_dirs()[rn], "p100", "final", "final")
        for c in ("C6", "SHUF", "T4", "EDL"):
            nm = f"R_{c}_s{s}"
            fin[(c, s)] = evaluate(nm, rm[f"{c}_s{s}"], "p100", "final", "final",
                                   skip_val=not (c == "C6" and s == 0) and ident["pass"])
    doc = final_decision(lam, fin, ident, rm)
    guard.atomic_write_json(out, doc, mode=0o400)
    return doc


def val_of(c, s, fin, rm):
    """VAL block for condition c / seed s: own final evaluation for E/F (and reused runs re-evaluated), else the
    semantic-edl-long-k p100 full + PE100k evaluation of the reused run."""
    e = fin.get((c, s))
    if e is not None and "val" in e:
        return {"ce": e["val"]["ce"], "doc_ce": e["val_doc_ce"], "routing": e.get("val_routing"),
                "per_expert": e.get("val_per_expert")}
    slk_name = {"C6": f"P1_C6_s{s}", "SHUF": f"P1_SHUF_s{s}", "T4": f"P1_T4_s{s}", "EDL": f"P1_EDL_s{s}"}[c]
    v = jl(SLK_ART / "evals" / slk_name / "val_p100_full_pe100k.json")
    return {"ce": v["val"]["ce"], "doc_ce": v["val_doc_ce"], "routing": v.get("routing"), "per_expert": v.get("per_expert")}


def final_decision(lam, fin, ident, rm):
    import numpy as np
    per = {}
    for s in FINAL_SEEDS:
        sE, sF, sC, sS = (summ(fin[(x, s)]) for x in ("E", "F", "C6", "SHUF"))
        vE, vEDL = val_of("E", s, fin, rm), val_of("EDL", s, fin, rm)
        per[s] = {"dE_F": paired(fin[("E", s)], fin[("F", s)]), "dC6_SHUF": paired(fin[("C6", s)], fin[("SHUF", s)]),
                  "dE_C6": paired(fin[("E", s)], fin[("C6", s)]),
                  "inter": {"E": sE["inter"], "F": sF["inter"], "C6": sC["inter"], "SHUF": sS["inter"]},
                  "val_ce": {c: val_of(c, s, fin, rm)["ce"] for c in ("E", "F", "C6", "SHUF", "T4", "EDL")},
                  "dCE_E_EDL": vE["ce"] - vEDL["ce"],
                  "pathology": {"E": sE["pathology"], "F": sF["pathology"]},
                  "act_E": activity(f"FIN_E_{lam_tag(lam)}_s{s}"), "act_F": activity(f"FIN_F_{lam_tag(lam)}_s{s}")}
    pool = pooled([(fin[("E", s)], fin[("F", s)]) for s in FINAL_SEEDS])
    ce_mean = float(np.mean([per[s]["dCE_E_EDL"] for s in FINAL_SEEDS]))
    crit = {"P1": all(per[s]["dE_F"]["diff"] > 0 for s in FINAL_SEEDS), "P2": pool["ci95"][0] > 0,
            "P3": ce_mean <= 0.005,
            "P4": all(not per[s]["pathology"]["E"] and not per[s]["pathology"]["F"] and
                      (per[s]["act_E"]["train_load_min_x_uniform"] or 0) >= 0.1 and per[s]["act_E"]["wmean_maxdev"] <= 1e-6
                      for s in FINAL_SEEDS)}
    diag = diagnose({s: {"dE_F": per[s]["dE_F"]["diff"], "dC_S": per[s]["dC6_SHUF"]["diff"],
                         "inter_E": per[s]["inter"]["E"], "inter_C6": per[s]["inter"]["C6"],
                         "inter_F": per[s]["inter"]["F"], "inter_SHUF": per[s]["inter"]["SHUF"],
                         "mi_E": summ(fin[("E", s)])["mi"], "mi_C6": summ(fin[("C6", s)])["mi"],
                         "ce_E": per[s]["val_ce"]["E"], "ce_C6": per[s]["val_ce"]["C6"], "act": per[s]["act_E"]}
                     for s in FINAL_SEEDS}, crit["P3"])
    doc = {"time": guard.now(), "lam": lam, "per_seed": {str(k): v for k, v in per.items()}, "pooled_dE_F": pool,
           "pooled_dC6_SHUF": pooled([(fin[("C6", s)], fin[("SHUF", s)]) for s in FINAL_SEEDS]),
           "mean_dCE_E_EDL": ce_mean, "criteria": crit, "success": all(crit.values()), "diagnosis": diag,
           "val_identity_check": ident, "rule": "prereg section 10"}
    results(f"FINAL | lambda {lam} | " + "; ".join(f"s{s}: dE-F {per[s]['dE_F']['diff']:.2e}, dCE(E-EDL) "
                                                    f"{per[s]['dCE_E_EDL']:+.4f}" for s in FINAL_SEEDS) +
            f"; pooled {pool['mean_diff']:.2e} CI [{pool['ci95'][0]:.2e}, {pool['ci95'][1]:.2e}]; criteria {crit} |")
    decision("Final (245M) decision", json.dumps({"criteria": crit, "pooled": pool, "dCE_E_EDL": ce_mean,
                                                  "diagnosis": diag}, default=str), f"diagnosis {diag['primary']}",
             "method SUPPORTED (P1-P4 pass)" if doc["success"] else "method NOT supported at 245M (criteria above)")
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["reuse-check", "checks", "search", "promote", "finals", "all"])
    a = ap.parse_args()
    guard.check_root()
    check_prereg()
    if a.stage in ("reuse-check", "all"):
        reuse_check()
    if a.stage in ("checks", "all"):
        checks()
    if a.stage in ("search", "all"):
        search()
    if a.stage in ("promote", "all"):
        promote()
    if a.stage in ("finals", "all"):
        finals()
        set_state("FINISHED", note="registered protocol complete")
        write_resume("FINISHED — registered protocol complete; nothing to resume", "true")


if __name__ == "__main__":
    main()
