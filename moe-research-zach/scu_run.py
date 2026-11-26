#!/usr/bin/env python
"""csr-soft-cf-usefulness-v2 orchestrator (docs/SOFT_CF_USEFULNESS_V2_PREREG.md). Every GPU job goes through
scu_driver.run_job (idempotent; an external interruption latches, records INCOMPLETE, this orchestrator writes RESUME.md
with state INCOMPLETE_STOPPED_BY_OPERATOR and stops; nothing relaunches automatically). Infrastructure copied from
sue_run.py (V1) with scu paths; final resumable slots are retained (no deletion); RESUME.md after every job.

  scu_run.py freeze    hash the preregistration (once; requires V1 TERMINAL)
  scu_run.py anchor    gamma anchor (prereg 6) -> anchor.json, gamma.json
  scu_run.py search    C1-C6 at 20M seed 100 (prereg 7) -> search_decision.json
  scu_run.py promote   50M SEM/SHUF seeds 100/101, dev + hold50, criteria 1-6, diagnosis (prereg 8, 10) -> promotion.json
  scu_run.py finals    245M finals + compute-matched TOKEN-4 + final evaluation (prereg 9) if promoted
  scu_run.py all       anchor -> search -> promote -> finals"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402
from csr.scu_paths import (ART, CALIB_SEARCH_TABLE, CALIB_SEL_TABLE, FINAL_SEEDS, GPU_CAP_H, MILESTONES,  # noqa: E402
                           PREREG, PREREG_SHA_FILE, RUN_PREFIX, SEARCH_SEEDS, SHUF_TABLE, SHUF_TABLE_SHA, SLK_ART,
                           SUE_ART, TABLE, TABLE_SHA, TMP)

RUNS = ART / "run_dirs.json"
RESUME_CMD = ".venv/bin/python scu_run.py all"
S20, S50 = MILESTONES["s20"], MILESTONES["s50"]
FIXED = {"probe_rows": 16, "probe_passes": 8, "refresh_every": 10, "m_min": 2.0, "n_min": 8, "clip": 3.0,
         "min_len": 256, "ramp_steps": 50, "rho": 0.998, "onset_step": 100, "z_lcb": 1.0, "prior_rho": 0.98}
REUSED50 = {"T4_s100": "runs/20261114T090727Z_forecast_latent_edl_v1_s_T4_s100",
            "T4_s101": "runs/20261115T103553Z_semantic_edl_v1_b_T4_s101",
            "EDL_s100": "runs/20261115T083452Z_semantic_edl_v1_b_EDL_s100",
            "EDL_s101": "runs/20261115T104319Z_semantic_edl_v1_b_EDL_s101"}
STAGE_DISK_GIB = {"search": 9.0, "promote": 6.0, "finals": 24.0}


# ------------------------------------------------------------------------------------------------ logging
def code_hash():
    h = hashlib.sha256()
    files = sorted([str(p.relative_to(guard.ROOT)) for p in guard.ROOT.glob("scu_*.py")] +
                   [str(p.relative_to(guard.ROOT)) for p in (guard.ROOT / "src/csr").glob("scu_*.py")] +
                   ["sue_evaluate.py", "sue_diag.py", "src/csr/sue_cf.py", "fle_evaluate.py", "fle_configs.py",
                    "src/csr/lfr_model.py", "src/csr/lfr_eval.py", "src/csr/slk_lib.py", "src/csr/sem_lib.py",
                    "src/csr/gc_lib.py", "src/csr/guard.py"])
    for f in files:
        p = guard.ROOT / f
        if p.exists():
            h.update(f.encode() + b"\0" + guard.sha256_file(p).encode() + b"\n")
    return h.hexdigest()


def append(name, text):
    ART.mkdir(parents=True, exist_ok=True)
    with open(ART / name, "a") as f:
        f.write(text)


def run_log(rec):
    append("RUN_LOG.jsonl", json.dumps({"time": guard.now(), "git_commit": None,
                                        "git_note": "no git repository; code_tree_sha256 = hash of scu code + imports",
                                        "code_tree_sha256": code_hash(), **rec}, default=str) + "\n")


def decision(observation, evidence, hypothesis, action, rejected=""):
    append("DECISION_LOG.md", f"\n### {guard.now()}\n- Observation: {observation}\n- Evidence: {evidence}\n"
                              f"- Hypothesis: {hypothesis}\n- Action: {action}\n"
                              + (f"- Rejected alternatives: {rejected}\n" if rejected else ""))


def deviation(text, effect, can_affect):
    n = sum(1 for _ in open(ART / "DEVIATIONS.jsonl")) + 1 if (ART / "DEVIATIONS.jsonl").exists() else 1
    append("DEVIATIONS.jsonl", json.dumps({"n": n, "time": guard.now(), "text": text, "effect": effect,
                                           "can_affect_conclusion": can_affect}) + "\n")


def results(line):
    if not (ART / "RESULTS_LIVE.md").exists():
        append("RESULTS_LIVE.md", "# RESULTS_LIVE (append-only; every completed result)\n\n| time | run | what | values |\n"
                                  "|---|---|---|---|\n")
    append("RESULTS_LIVE.md", f"| {guard.now()} | {line}\n")


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


def write_resume(state, cmd, notes=""):
    txt = (f"# RESUME — csr-soft-cf-usefulness-v2\n\nUpdated {guard.now()}.\n\n**State:** {state}\n\n"
           f"GPU used: {gpu_hours():.2f} GPU-h (cap {GPU_CAP_H['total']}). Disk free: "
           f"{guard.disk_free_bytes() / 2**30:.1f} GiB (reserve 100 GiB).\n\n**Resume command"
           f"{' (only after explicit permission)' if 'INCOMPLETE' in state or 'STOPPED' in state else ''}:**\n\n"
           f"```\ncd {guard.ROOT} && source env.sh && {cmd}\n```\n\n{notes}\n")
    guard.atomic_write_bytes(ART / "RESUME.md", txt.encode(), overwrite=True)


def stage_disk(stage):
    est = STAGE_DISK_GIB[stage]
    free = guard.disk_free_bytes() / 2**30
    run_log({"phase": stage, "kind": "disk_estimate", "job": None, "gpu_seconds": 0, "free_gib": free,
             "estimated_peak_gib": est, "reserve_gib": 100})
    if free - est < 100:
        decision(f"Disk estimate blocks stage {stage}", f"free {free:.2f} GiB, estimated peak need {est} GiB, reserve 100",
                 "-", f"stop cleanly; {100 + est - free:.1f} GiB more free space needed")
        write_resume(f"INCOMPLETE_STOPPED_DISK_RESERVE — stage {stage} needs ~{est} GiB; free {free:.1f} GiB; "
                     f"{100 + est - free:.1f} GiB more required", RESUME_CMD, "Free disk, then resume (completed jobs skipped).")
        raise SystemExit(f"disk: stage {stage} needs {100 + est - free:.1f} GiB more")


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


def job(job_id, argv, phase, kind, run_name=None, seed=None, cfg_p=None, expect=(0,), disk_gb=2.0):
    from scu_driver import load_state, run_job
    if (load_state()["jobs"].get(job_id) or {}).get("status") == "COMPLETE":
        return 0
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
                "plan_sha256": (post or {}).get("plan_sha256"), "peak_alloc_gib": (post or {}).get("peak_alloc_gib"),
                "checkpoints": {k: v["path"] for k, v in ((post or {}).get("ckpts") or {}).items()} if post else None,
                "disk_free_gib": guard.disk_free_bytes() / 2**30,
                "validity": "valid" if ok else ("INCOMPLETE (external interruption; not evidence)"
                                                if js.get("status") == "INCOMPLETE" else "FAILED (not evidence)")})
    run_log(rec)
    if ok:
        write_resume(f"RUNNING — last completed job {job_id} ({guard.now()}); orchestrator active", RESUME_CMD,
                     "If no scu process is alive, the run was stopped: check driver_state.json and the last RUN_LOG "
                     "record before resuming. Completed jobs are skipped on resume.")
    if not ok:
        state = ("INCOMPLETE_STOPPED_BY_OPERATOR — external interruption (foreign GPU process / signal / STOP marker / "
                 f"launcher block) during job {job_id}; no automatic relaunch") if js.get("status") == "INCOMPLETE" \
            or js.get("status") is None else f"FAILED — job {job_id} (rc {rc}); investigate before resuming"
        write_resume(state, ".venv/bin/python scu_driver.py authorize-resume && " + RESUME_CMD,
                     f"Last job `{job_id}`; run dir {rd}. Trainers save model/optimizer/RNG/data cursor/bank/prior in "
                     "their verified resumable slot and resume in place (--run-dir). Completed jobs are skipped.")
        raise SystemExit(rc if isinstance(rc, int) else 1)
    return rc


def jl(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def cap_ok(phase, est):
    return gpu_hours((phase,)) + est <= GPU_CAP_H.get(phase, GPU_CAP_H["total"]) and gpu_hours() + est <= GPU_CAP_H["total"]


def make_cfg(run_name, seed, cond, category, budget=None, milestones=None, diag_every=100):
    from fle_configs import MODEL, TRAIN
    from csr.scu_paths import CFG_DIR, EXP, FINAL_BUDGET, prereg_sha
    cfg = {"experiment": EXP, "run_name": run_name, "seed": seed, "budget_tokens": budget or FINAL_BUDGET,
           "prereg_sha256": prereg_sha(), "model": MODEL, "train": dict(TRAIN), "cond": cond, "kind": "final",
           "milestones": dict(milestones or MILESTONES), "diag_every": diag_every, "category": category}
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    p = CFG_DIR / f"{run_name}.json"
    if p.exists():
        if json.loads(p.read_text()) != cfg:
            raise guard.GuardError(f"config {p} exists and differs")
    else:
        guard.atomic_write_json(p, cfg, mode=0o400)
    return p, cfg


def train_to(rn, seed, cond, max_steps, phase, category, budget=None, milestones=None, est=0.2):
    st = status_of(rn)
    if st and (st.get("status") == "COMPLETE" or (max_steps and st.get("steps_done", 0) >= max_steps)):
        return rn
    if not cap_ok(phase, est):
        decision(f"{phase} GPU cap reached", f"{phase} {gpu_hours((phase,)):.2f} h; total {gpu_hours():.2f} h", "-",
                 f"do not train {rn}")
        raise SystemExit(f"{phase} cap reached before {rn}")
    p, cfg = make_cfg(rn, seed, cond, category, budget=budget, milestones=milestones)
    for leg in range(8):
        prev = run_dirs().get(rn)
        argv = ["scu_train.py", "--config", str(p)] + (["--max-steps", str(max_steps)] if max_steps else []) + \
            (["--run-dir", prev] if prev else [])
        job(f"train_{rn}_to{max_steps or 'end'}_leg{leg}", argv, phase, "train", rn, seed, p, expect=(0, 75), disk_gb=4.0)
        st = status_of(rn)
        if st and (st.get("status") == "COMPLETE" or (max_steps and st.get("steps_done", 0) >= max_steps)):
            break
    st = status_of(rn)
    results(f"{rn} | train to {st['steps_done']} steps | tokens {st['seen']}; status {st['status']}; tok/s "
            f"{st.get('tokens_per_s', 0):.0f}; cf_s {st.get('cf_seconds', 0):.0f}; train_step_s "
            f"{st.get('train_step_seconds', 0):.0f}; peak {st.get('peak_alloc_gib', 0):.1f} GiB |")
    return rn


def evaluate(rn, mode, tag, phase, key=None, extra=()):
    key = key or rn
    outp = ART / "evals" / key / f"{mode}_{tag}.json"
    fresh = not outp.exists()
    if fresh:
        argv = ["scu_evaluate.py", mode, "--run-dir", run_dirs().get(rn, rn), "--run-name", key, "--tag", tag,
                "--pseudo-table", CALIB_SEARCH_TABLE] + (["--pseudo-table-sel", CALIB_SEL_TABLE] if mode != "dev" else []) \
            + list(extra)
        job(f"{mode}_{key}_{tag}", argv, phase, "eval", key, None, disk_gb=0.5)
    e = jl(outp)
    if fresh:
        pe, au = e["per_expert"]["pseudo"], e["audit"]["pseudo"]["routing"]
        results(f"{key} | {mode} {tag} ({e['block']}) | CE {e['ce_block']['ce']:.4f}; pseudo interaction "
                f"{pe['mean_interaction']:.6f} [{pe['mean_interaction_ci95'][0]:.6f}, {pe['mean_interaction_ci95'][1]:.6f}]; "
                f"spec {pe['n_subject_specialized']}; pseudo MI {au['mi_mean']:.4f}; e5 interaction "
                f"{e['per_expert']['e5']['mean_interaction']:.6f}; load max {e['pathology']['load_max_x_uniform']:.2f} min "
                f"{e['pathology']['load_min_x_uniform']:.2f} ent {e['pathology']['routing_entropy']:.3f} |")
    return e


def summary(e):
    pe, au, pa = e["per_expert"]["pseudo"], e["audit"]["pseudo"]["routing"], e["pathology"]
    return {"ce": e["ce_block"]["ce"], "inter": pe["mean_interaction"], "inter_ci": pe["mean_interaction_ci95"],
            "boot": pe["boot_mean_samples"], "spec": pe["n_subject_specialized"],
            "frac_spec": pe.get("frac_subject_specialized"), "mi": au["mi_mean"], "nmi": au.get("nmi_mean"),
            "topic_entropy": au.get("topic_entropy_mean"), "topic_pred": e["audit"]["pseudo"].get("topic_pred_acc"),
            "inter_e5": e["per_expert"]["e5"]["mean_interaction"],
            "generic": float(sum(e["per_expert"]["generic_importance_mean_dce"]) /
                             len(e["per_expert"]["generic_importance_mean_dce"])),
            "pathology": bool(pa["load_min_x_uniform"] < 0.1 or pa["load_max_x_uniform"] > 4.0
                              or pa["routing_entropy"] < 0.45), **pa}


def nob(s):
    return {k: v for k, v in s.items() if k != "boot"}


def sue_eval(key, tag):
    """Reused V1 dev evaluations (identical evaluator and rows), read-only."""
    e = jl(SUE_ART / "evals" / key / f"dev_{tag}.json")
    if e is None:
        raise guard.GuardError(f"missing reused V1 evaluation {key} dev_{tag}")
    return summary(e)


def run_metrics(rn):
    rd = run_dirs()[rn]
    rows = [json.loads(x) for x in (guard.ROOT / rd / "metrics.jsonl").read_text().splitlines()]
    return [x for x in rows if "step" in x], [x["util_diag"] for x in rows if "util_diag" in x]


def active_frac(rn, onset):
    rows, _ = run_metrics(rn)
    post = [r for r in rows if r["step"] >= onset]
    return float(sum(1 for r in post if r.get("use_active", 0) > 0) / max(1, len(post)))


def first_active(rn):
    rows, _ = run_metrics(rn)
    a = [r for r in rows if r.get("use_active", 0) > 0]
    return {"step": a[0]["step"], "tokens": a[0]["seen"]} if a else None


def reliability(rn, upto):
    _, dg = run_metrics(rn)
    sb = [x.get("spearman_brown") for x in dg if x["step"] <= upto]
    sb = [x for x in sb if x is not None and math.isfinite(x)][-5:]
    return float(sum(sb) / len(sb)) if sb else float("nan")


def paired_ci(diffs_boot):
    import numpy as np
    m = np.mean(np.asarray(diffs_boot), 0)
    return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


# ------------------------------------------------------------------------------------------------ mechanism
def mech_chain(rn, onset):
    """Prereg section 10 chain from run_dir/mech snapshots (first post-onset snapshot with >= 10 evidence cells vs last)."""
    import numpy as np

    def spearmanr(x, y):
        rx, ry = np.argsort(np.argsort(x)).astype(float), np.argsort(np.argsort(y)).astype(float)
        return (float(np.corrcoef(rx, ry)[0, 1]),)
    rd = guard.ROOT / run_dirs()[rn] / "mech"
    snaps = sorted(rd.glob("mech_s*.npz"))
    zs = [np.load(p) for p in snaps]
    post = [z for z in zs if int(z["step"]) >= onset and z["EV"].sum() >= 10]
    if len(post) < 2:
        return {"n_snapshots": len(zs), "insufficient": True}
    a, b = post[0], post[-1]
    ev = a["EV"]
    cells = np.broadcast_to(ev[..., None], a["scores"].shape) & (a["scores"] != 0)
    dlogR = np.log(np.maximum(b["R"], 1e-12)) - np.log(np.maximum(a["R"], 1e-12))
    dX = b["X"] - a["X"]
    both = a["valid"] & b["valid"]
    ddU = b["d"] - a["d"]
    Gx = b["G"] / np.maximum(b["X"], 1e-9)

    def sp(x, y, m):
        m = m & np.isfinite(x) & np.isfinite(y)
        return float(spearmanr(x[m], y[m])[0]) if m.sum() > 10 else float("nan")
    ts = []
    for z in zs:
        R, Q, EV = z["R"], z["Q"], z["EV"]
        kl = float(np.mean((Q[EV] * (np.log(np.maximum(Q[EV], 1e-12)) - np.log(np.maximum(R[EV], 1e-12)))).sum(-1))) \
            if EV.any() else float("nan")
        ts.append({"step": int(z["step"]), "tokens": int(z["seen"]), "n_evidence": int(EV.sum()), "kl_q_R": kl,
                   "valid_frac": float(z["valid"].mean())})
    pos = a["scores"] > 0
    fav = pos.sum(0)  # (L, E): number of groups with a positive score for this expert
    gen_mass = float((a["scores"] * pos * (fav[None] >= 8)).sum() / max(1e-12, (a["scores"] * pos).sum()))
    return {"first": int(a["step"]), "last": int(b["step"]), "n_evidence_cells": int(cells.sum()),
            "corr_score_dlogR": sp(a["scores"], dlogR, cells),
            "corr_dlogR_dX": sp(dlogR, dX, cells),
            "corr_X_G_per_exposure": sp(b["X"], Gx, both),
            "corr_G_ddU": sp(b["G"], ddU, both),
            "corr_dX_ddU": sp(dX, ddU, both),
            "corr_score_ddU": sp(a["scores"], ddU, cells & both),
            "generic_positive_mass": gen_mass, "n_le_favoured_by_ge8_groups": int((fav >= 8).sum()),
            "timeseries": ts}


def diagnose(sem_runs, pairs, refs, onset):
    """Prereg section 10: first matching link. pairs: {seed: {SEM, SHUF summaries}}; refs: REF_C6 summaries by seed."""
    info = {}
    for s, rn in sem_runs.items():
        _, dg = run_metrics(rn)
        last = dg[-1] if dg else {}
        info[s] = {"reliability_sb": reliability(rn, 10 ** 9), "target": last.get("target"),
                   "chain": mech_chain(rn, onset)}
    if any(not (v["reliability_sb"] >= 0.3) for v in info.values()):
        return "A_measurement", info
    flat = []
    for v in info.values():
        t = v["target"] or {}
        ch = v["chain"]
        flat.append((t.get("frac_evidence") or 0) < 0.10 or not ((t.get("kl_q_prior") or 0) >= 0.005)
                    or (ch.get("generic_positive_mass") or 0) > 0.5)
    if any(flat):
        return "B_target", info
    if any(not (v["chain"].get("corr_score_dlogR", float("nan")) > 0.05) for v in info.values()):
        return "C_router", info
    if any(not (v["chain"].get("corr_dX_ddU", float("nan")) > 0.05) for v in info.values()) or \
            any(pairs[s]["SEM"]["inter"] <= refs[s]["inter"] for s in pairs if s in refs):
        return "D_expert_learning", info
    return "beyond_D", info


# ------------------------------------------------------------------------------------------------ stages
def freeze():
    if PREREG_SHA_FILE.exists():
        raise guard.GuardError("prereg already frozen")
    v1 = (SUE_ART / "RESUME.md").read_text()
    if "TERMINAL" not in v1 or not (SUE_ART / "confsel.json").exists():
        raise guard.GuardError("V1 must be terminal before V2 is frozen")
    PREREG.chmod(0o400)
    PREREG_SHA_FILE.write_text(f"{guard.sha256_file(PREREG)}  {PREREG.name}\n")
    PREREG_SHA_FILE.chmod(0o400)
    print(PREREG_SHA_FILE.read_text())


def cond_of(table, sha, c):
    u = {**FIXED, "gamma": float(f"{c['gamma']:.6g}"), "tau": c["tau"], "form": c["form"], "estimator": c["estimator"]}
    return {"paper_latent": {"table": table, "table_sha256": sha, "beta": 0.0}, "soft_cf": u}


def anchor():
    p = ART / "gamma.json"
    if p.exists():
        return jl(p)
    if not (ART / "anchor.json").exists():
        job("anchor", ["scu_diag.py"], "diagnostic", "diagnostic", disk_gb=1.0)
    a = jl(ART / "anchor.json")
    g = a["grad_anchor"]
    if g["gn_cf_unit"] > 0:
        raw = 0.003 * g["gn_use_unit"] / g["gn_cf_unit"]
        e = math.floor(math.log10(raw))
        gm = round(raw / 10 ** e) * 10 ** e
    else:
        raw, gm = None, 0.01
    out = {"time": guard.now(), "gamma_mid": gm, "raw": raw, "grad_anchor": g,
           "rule": "0.003 * ||grad L_use|| / ||grad L_CF|| (anchor REF_C6_s100 s20, 400-step bank), 1 significant digit"}
    guard.atomic_write_json(p, out, mode=0o400)
    s4 = a["by_size"]
    results(f"anchor | 400 probe steps on REF_C6_s100 s20 | SEM SB {s4['SEM'][str(a['steps'])]['spearman_brown']:.3f}, "
            f"SHUF SB {s4['SHUF'][str(a['steps'])]['spearman_brown']:.3f}; SEM target {json.dumps(g['target'])}; SHUF target "
            f"{json.dumps(a['shuf_target'])}; gamma_mid {gm} (raw {raw}) |")
    decision("gamma anchor", json.dumps({"grad": g, "shuf_target": a["shuf_target"]}), "-", f"gamma_mid = {gm} (prereg rule)")
    return out


def candidates():
    gm = anchor()["gamma_mid"]
    base = {"gamma": gm, "tau": 1.0, "form": "tilt", "estimator": "lcb"}
    var = {"C1": {}, "C2": {"gamma": gm / 3}, "C3": {"gamma": gm * 3}, "C4": {"tau": 0.5}, "C5": {"estimator": "mean"},
           "C6": {"form": "uniform"}}
    return {k: {**base, **v} for k, v in var.items()}


def search():
    p = ART / "search_decision.json"
    if p.exists():
        return jl(p)
    cands = candidates()
    ref = sue_eval("POST_T4_s100", "s20")
    rows = {}
    for name, c in cands.items():
        rn = f"S_{name}_s100"
        train_to(rn, 100, cond_of(TABLE, TABLE_SHA, c), S20, "search", "search", est=0.15)
        s = summary(evaluate(rn, "dev", "s20", "search"))
        s["active_frac"] = active_frac(rn, FIXED["onset_step"])
        s["first_active"] = first_active(rn)
        s["reliability_sb"] = reliability(rn, S20)
        s["d_inter_vs_t4"], s["d_ce_vs_t4"] = s["inter"] - ref["inter"], s["ce"] - ref["ce"]
        s["eligible"] = bool(s["d_inter_vs_t4"] > 0 and s["d_ce_vs_t4"] <= 0.010 and not s["pathology"]
                             and s["active_frac"] >= 0.5)
        rows[name] = nob(s)
    el = sorted([k for k, v in rows.items() if v["eligible"]], key=lambda k: (-rows[k]["inter"], rows[k]["ce"]))
    if len(el) > 1:
        top = rows[el[0]]["inter"]
        tied = [k for k in el if top - rows[k]["inter"] <= 2e-5]
        el = sorted(tied, key=lambda k: rows[k]["ce"]) + [k for k in el if k not in tied]
    out = {"time": guard.now(), "reference_T4_s100_s20": nob(ref), "candidates": rows, "specs": cands,
           "promoted": el[:1]}
    guard.atomic_write_json(p, out, mode=0o400)
    decision("V2 search (20M, seed 100) complete",
             json.dumps({k: {x: rows[k][x] for x in ("inter", "ce", "d_inter_vs_t4", "d_ce_vs_t4", "eligible",
                                                     "active_frac", "reliability_sb")} for k in rows}), "-",
             f"promote {el[:1]}" if el else "no eligible candidate: no promotion; diagnose on search runs")
    return out


def promote():
    p = ART / "promotion.json"
    if p.exists():
        return jl(p)
    sd = search()
    cands = candidates()
    if not sd["promoted"]:
        best = max(sd["candidates"], key=lambda k: sd["candidates"][k]["inter"])
        cause, info = diagnose({100: f"S_{best}_s100"}, {}, {}, FIXED["onset_step"])
        out = {"time": guard.now(), "promoted": None, "diagnosis_basis": best, "diagnosis": cause, "info": info,
               "pass": False}
        guard.atomic_write_json(p, out, mode=0o400)
        decision("No V2 candidate eligible at 20M", json.dumps({"basis": best, "cause": cause}), cause,
                 "stop V2 (prereg section 7: no generations)")
        return out
    k = sd["promoted"][0]
    c = cands[k]
    pairs = {}
    for seed in SEARCH_SEEDS:
        rs, rh = (f"S_{k}_s{seed}", f"H_{k}_s{seed}")
        train_to(rs, seed, cond_of(TABLE, TABLE_SHA, c), S50, "promotion", "search", est=0.3)
        train_to(rh, seed, cond_of(SHUF_TABLE, SHUF_TABLE_SHA, c), S50, "promotion", "search", est=0.3)
        pairs[seed] = {"SEM": summary(evaluate(rs, "dev", "s50", "promotion")),
                       "SHUF": summary(evaluate(rh, "dev", "s50", "promotion")), "sem_run": rs, "shuf_run": rh}
    if not (ART / "HOLD50_OPEN.json").exists():
        guard.atomic_write_json(ART / "HOLD50_OPEN.json", {"time": guard.now(), "candidate": k}, mode=0o400)
    hold, boots = {}, []
    import numpy as np
    for seed, v in pairs.items():
        hs, hh = summary(evaluate(v["sem_run"], "hold50", "s50", "promotion")), \
            summary(evaluate(v["shuf_run"], "hold50", "s50", "promotion"))
        boots.append(np.asarray(hs["boot"]) - np.asarray(hh["boot"]))
        hold[seed] = {"SEM": nob(hs), "SHUF": nob(hh), "d_inter": hs["inter"] - hh["inter"], "d_ce": hs["ce"] - hh["ce"]}
    ci = paired_ci(boots)
    t4 = {s: sue_eval(f"POST_T4_s{s}", "s50") for s in SEARCH_SEEDS}
    edl = {s: sue_eval(f"POST_EDL_s{s}", "s50") for s in SEARCH_SEEDS}
    refc6 = {s: sue_eval(f"REF_C6_s{s}", "s50") for s in SEARCH_SEEDS}
    refsh = {s: sue_eval(f"REF_SHUF_s{s}", "s50") for s in SEARCH_SEEDS}
    d_dev = {s: pairs[s]["SEM"]["inter"] - pairs[s]["SHUF"]["inter"] for s in SEARCH_SEEDS}
    d_ce_t4 = {s: pairs[s]["SEM"]["ce"] - t4[s]["ce"] for s in SEARCH_SEEDS}
    rel = {s: reliability(pairs[s]["sem_run"], S50) for s in SEARCH_SEEDS}
    path = any(pairs[s][x]["pathology"] or hold[s][x]["pathology"] for s in SEARCH_SEEDS for x in ("SEM", "SHUF"))
    crit = {"1_ce_vs_t4": float(np.mean(list(d_ce_t4.values()))) <= 0.010,
            "2_sem_gt_shuf_mean": float(np.mean(list(d_dev.values()))) > 0,
            "3_same_sign_both_seeds": all(x > 0 for x in d_dev.values()),
            "4_hold50_ci_lower_gt_0": ci[0] > 0, "5_no_pathology": not path,
            "6_reliability_sb_ge_0.3": all(x >= 0.3 for x in rel.values())}
    passed = bool(all(crit.values()))
    cause, info = diagnose({s: pairs[s]["sem_run"] for s in SEARCH_SEEDS}, pairs, refc6, FIXED["onset_step"])
    info_sh = {s: mech_chain(pairs[s]["shuf_run"], FIXED["onset_step"]) for s in SEARCH_SEEDS}
    if cause == "beyond_D":
        if all(x > 0 for x in d_dev.values()) and not crit["4_hold50_ci_lower_gt_0"]:
            cause = "E_transfer"
        elif crit["2_sem_gt_shuf_mean"] and crit["3_same_sign_both_seeds"] and crit["4_hold50_ci_lower_gt_0"] \
                and not crit["1_ce_vs_t4"]:
            cause = "F_accuracy"
        elif not passed:
            cause = "SEM_equals_SHUF"
        else:
            cause = "none"
    out = {"time": guard.now(), "promoted": k, "spec": c, "pairs": {str(s): {"SEM": nob(v["SEM"]), "SHUF": nob(v["SHUF"]),
                                                                              "sem_run": v["sem_run"], "shuf_run": v["shuf_run"]}
                                                                     for s, v in pairs.items()},
           "hold50": {str(s): v for s, v in hold.items()}, "hold50_pooled_ci95": ci, "d_dev": d_dev,
           "d_ce_vs_t4": d_ce_t4, "reliability_sb": rel,
           "references": {"T4": {s: nob(v) for s, v in t4.items()}, "EDL": {s: nob(v) for s, v in edl.items()},
                          "REF_C6": {s: nob(v) for s, v in refc6.items()}, "REF_SHUF": {s: nob(v) for s, v in refsh.items()}},
           "criteria": crit, "pass": passed, "diagnosis": cause, "mechanism_sem": info, "mechanism_shuf": info_sh,
           "first_active": {s: first_active(pairs[s]["sem_run"]) for s in SEARCH_SEEDS}}
    guard.atomic_write_json(p, out, mode=0o400)
    results(f"promotion {k} | 50M SEM vs SHUF dev d {d_dev}; hold50 {[(h['d_inter'], h['d_ce']) for h in hold.values()]}; "
            f"CI {ci}; criteria {crit}; diagnosis {cause} |")
    decision("V2 promotion decision", json.dumps({"criteria": crit, "d_dev": d_dev, "hold50_ci": ci, "rel": rel}), cause,
             "train finals (245M, seeds 0-2)" if passed else "not promoted: no 245M runs")
    return out


def finals():
    pr = promote()
    if not pr["pass"]:
        return {"finals": False}
    c = pr["spec"]
    for seed in FINAL_SEEDS:
        train_to(f"F_SEM_s{seed}", seed, cond_of(TABLE, TABLE_SHA, c), None, "final", "final", est=1.4)
        train_to(f"F_SHUF_s{seed}", seed, cond_of(SHUF_TABLE, SHUF_TABLE_SHA, c), None, "final", "final", est=1.4)
    sts = [status_of(f"F_SEM_s{s}") for s in FINAL_SEEDS]
    ovh = sum(s["cf_seconds"] for s in sts) / max(1e-9, sum(s["train_step_seconds"] - s["cf_seconds"] for s in sts))
    cm = None
    if ovh > 0.05:
        from csr.lfr_lib import main_stream
        from csr.scu_paths import FINAL_BUDGET
        budget = int(FINAL_BUDGET * (1 + ovh))
        cm = {"overhead": ovh, "budget": budget}
        for seed in FINAL_SEEDS:
            if not cap_ok("final", 1.5):
                decision("Compute-matched TOKEN-4: budget", f"final {gpu_hours(('final',)):.2f} h", "-",
                         f"stop compute-matched runs before seed {seed}")
                break
            _, stp, _ = main_stream(budget)
            train_to(f"CM_T4_s{seed}", seed, {}, None, "final", "final", budget=budget, milestones={"p100": len(stp)}, est=1.5)
    if not (ART / "FINAL_OPEN.json").exists():
        guard.atomic_write_json(ART / "FINAL_OPEN.json", {"time": guard.now(), "overhead": ovh, "compute_matched": cm},
                                mode=0o400)
    from csr.slk_paths import REUSED as SLK_REUSED
    slk_dirs = {**{f"P1_{k}": v for k, v in SLK_REUSED.items()}, **(jl(SLK_ART / "run_dirs.json") or {})}
    reused = {f"{c_}_s{s}": slk_dirs.get(f"P1_{c_}_s{s}", "") for s in FINAL_SEEDS for c_ in ("T4", "EDL", "C6", "SHUF")}
    ev = {}
    for seed in FINAL_SEEDS:
        for rn in (f"F_SEM_s{seed}", f"F_SHUF_s{seed}") + ((f"CM_T4_s{seed}",) if status_of(f"CM_T4_s{seed}") else ()):
            ev[rn] = evaluate(rn, "final", "p100", "final")
    for key, rd in reused.items():
        if rd:
            ev[key] = evaluate(rd, "final", "p100", "final", key=f"R_{key}", extra=("--skip-val",))
    return final_decision(ev)


def final_decision(ev):
    import numpy as np
    p = ART / "final_decision.json"
    if p.exists():
        return jl(p)
    S = list(FINAL_SEEDS)
    sm = {k: summary(v) for k, v in ev.items()}
    d = [sm[f"F_SEM_s{s}"]["inter"] - sm[f"F_SHUF_s{s}"]["inter"] for s in S]
    ci = paired_ci([np.asarray(sm[f"F_SEM_s{s}"]["boot"]) - np.asarray(sm[f"F_SHUF_s{s}"]["boot"]) for s in S])
    sd_shuf = float(np.std([sm[f"F_SHUF_s{s}"]["inter"] for s in S], ddof=1))
    val = lambda rn: ev[rn]["val"]["ce"]  # noqa: E731
    slk_val = lambda key: jl(SLK_ART / "evals" / f"P1_{key}" / "val_p100_full_pe100k.json")["val"]["ce"]  # noqa: E731
    d_t4 = [val(f"F_SEM_s{s}") - slk_val(f"T4_s{s}") for s in S]
    d_edl = [val(f"F_SEM_s{s}") - slk_val(f"EDL_s{s}") for s in S]
    path = [bool(ev[f"F_SEM_s{s}"]["val"]["ent"] < 0.45 or np.asarray(ev[f"F_SEM_s{s}"]["val"]["load"]).max() * 32 > 4
                 or np.asarray(ev[f"F_SEM_s{s}"]["val"]["load"]).min() * 32 < 0.1) for s in S]
    mass = []
    for s in S:
        a, b = ev[f"F_SEM_s{s}"]["per_expert"], ev[f"F_SHUF_s{s}"]["per_expert"]
        la, lb = np.asarray(a["load"]).reshape(-1), np.asarray(b["load"]).reshape(-1)
        ia, ib = np.asarray(a["pseudo"]["interaction"], float), np.asarray(b["pseudo"]["interaction"], float)
        edges = np.quantile(np.concatenate([la, lb]), [0.2, 0.4, 0.6, 0.8])
        diffs = []
        for qi in range(5):
            ma, mb = (np.digitize(la, edges) == qi) & np.isfinite(ia), (np.digitize(lb, edges) == qi) & np.isfinite(ib)
            if ma.any() and mb.any():
                diffs.append(float(ia[ma].mean() - ib[mb].mean()))
        mass.append(float(np.mean(diffs)))
    crit = {"F1_every_seed": all(x > 0 for x in d), "F2_ci_excludes_0": ci[0] > 0,
            "F2_exceeds_seed_noise": float(np.mean(d)) > sd_shuf, "F3_no_pathology": not any(path),
            "F4_ce_vs_t4": float(np.mean(d_t4)) <= 0.010, "F4_ce_vs_edl": float(np.mean(d_edl)) <= 0.010,
            "F5_mass_matched": float(np.mean(mass)) > 0}
    out = {"time": guard.now(), "d_inter": d, "paired_ci95": ci, "sd_shuf_seed": sd_shuf, "d_ce_vs_t4": d_t4,
           "d_ce_vs_edl": d_edl, "pathology": path, "mass_matched": mass, "criteria": crit,
           "success": bool(all(crit.values())), "summaries": {k: nob(v) for k, v in sm.items()}}
    guard.atomic_write_json(p, out, mode=0o400)
    decision("V2 final preregistered decision", json.dumps(crit), "-",
             f"SEM-V2 {'SUCCEEDS' if out['success'] else 'does NOT succeed'}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["freeze", "anchor", "search", "promote", "finals", "all"])
    a = ap.parse_args()
    guard.check_root()
    if a.cmd == "freeze":
        freeze()
        return
    from csr.scu_paths import check_prereg
    check_prereg()
    if a.cmd in ("anchor", "all"):
        anchor()
    if a.cmd in ("search", "all"):
        if not (ART / "search_decision.json").exists():
            stage_disk("search")
        search()
    if a.cmd in ("promote", "all"):
        if not (ART / "promotion.json").exists():
            stage_disk("promote")
        promote()
    if a.cmd in ("finals", "all"):
        pr = promote()
        if pr["pass"] and not (ART / "final_decision.json").exists():
            stage_disk("finals")
        r = finals()
        write_resume("TERMINAL — all preregistered V2 stages complete" + ("" if r.get("finals", True) else
                                                                         " (finals not reached by rule)"),
                     ".venv/bin/python scu_analyze.py", "Nothing to resume; analysis/report only.")


if __name__ == "__main__":
    main()
