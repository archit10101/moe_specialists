#!/usr/bin/env python
"""csr-semantic-utility-edl-v1 orchestrator (docs/SEMANTIC_UTILITY_EDL_V1_PREREG.md). Every GPU job goes through
sue_driver.run_job (idempotent; an external interruption latches, records INCOMPLETE, the orchestrator writes
RESUME.md with state INCOMPLETE_STOPPED_BY_OPERATOR and stops; nothing relaunches automatically).

  sue_run.py resolve     print the section-3 grouping resolution from Task-1 artifacts (read-only)
  sue_run.py freeze      append the resolution as prereg section 13, write the prereg hash (once)
  sue_run.py diag        reliability diagnostic + gamma anchor (prereg 6)
  sue_run.py search      U1-U8 at 20M seed 100 + REF-C6 s100 reference (prereg 7) -> search_decision.json
  sue_run.py promote     50M SEM/SHUF seeds 100/101 + references, freeze or generations (prereg 8-9)
  sue_run.py confsel     CALIB-SELECTION half 0, once (prereg 10)
  sue_run.py finals      245M finals + compute-matched TOKEN-4 + final evaluation (prereg 11) if confirmed
  sue_run.py all         diag -> search -> promote -> confsel -> finals"""
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
from csr.sue_paths import (ART, C6_BETA, C6_SHUF_TABLE, C6_SHUF_TABLE_SHA, C6_TABLE, C6_TABLE_SHA, FINAL_SEEDS,  # noqa: E402
                           GPU_CAP_H, MILESTONES, PREREG, PREREG_SHA_FILE, RUN_PREFIX, SEARCH_SEEDS, SLK_ART, TMP)

RUNS = ART / "run_dirs.json"
RESUME_CMD = ".venv/bin/python sue_run.py all"
S20, S50 = MILESTONES["s20"], MILESTONES["s50"]
ANCHOR = "runs/20261115T100411Z_semantic_edl_v1_s_C6_s100/ckpt_s20"
P1 = {"C6": "runs/20261115T205836Z_semantic_edl_long_k_P1_C6_s0", "SHUF": "runs/20261115T213547Z_semantic_edl_long_k_P1_SHUF_s0"}


# ------------------------------------------------------------------------------------------------ logging
def code_hash():
    h = hashlib.sha256()
    files = sorted([str(p.relative_to(guard.ROOT)) for p in guard.ROOT.glob("sue_*.py")] +
                   [str(p.relative_to(guard.ROOT)) for p in (guard.ROOT / "src/csr").glob("sue_*.py")] +
                   ["fle_evaluate.py", "fle_configs.py", "src/csr/lfr_model.py", "src/csr/lfr_eval.py",
                    "src/csr/slk_lib.py", "src/csr/sem_lib.py", "src/csr/gc_lib.py", "src/csr/guard.py"])
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
                                        "git_note": "no git repository; code_tree_sha256 = hash of sue code + imports",
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
    txt = (f"# RESUME — csr-semantic-utility-edl-v1\n\nUpdated {guard.now()}.\n\n**State:** {state}\n\n"
           f"GPU used: {gpu_hours():.2f} GPU-h (cap {GPU_CAP_H['total']}). Disk free: "
           f"{guard.disk_free_bytes() / 2**30:.1f} GiB (reserve 100 GiB).\n\n**Resume command"
           f"{' (only after explicit permission)' if 'INCOMPLETE' in state or 'STOPPED' in state else ''}:**\n\n"
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


def job(job_id, argv, phase, kind, run_name=None, seed=None, cfg_p=None, expect=(0,), disk_gb=2.0):
    from sue_driver import load_state, run_job
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
                "plan_sha256": (post or {}).get("plan_sha256"),
                "checkpoints": {k: v["path"] for k, v in ((post or {}).get("ckpts") or {}).items()} if post else None,
                "disk_free_gib": guard.disk_free_bytes() / 2**30,
                "validity": "valid" if ok else ("INCOMPLETE (external interruption; not evidence)"
                                                if js.get("status") == "INCOMPLETE" else "FAILED (not evidence)")})
    run_log(rec)
    if ok:
        write_resume(f"RUNNING — last completed job {job_id} ({guard.now()}); orchestrator active", RESUME_CMD,
                     "If no sue process is alive, the run was stopped: check driver_state.json and the last RUN_LOG "
                     "record before resuming. Completed jobs are skipped on resume.")
    if not ok:
        state = ("INCOMPLETE_STOPPED_BY_OPERATOR — external interruption (foreign GPU process / signal / STOP marker / "
                 f"launcher block) during job {job_id}; no automatic relaunch") if js.get("status") == "INCOMPLETE" \
            or js.get("status") is None else f"FAILED — job {job_id} (rc {rc}); investigate before resuming"
        write_resume(state, ".venv/bin/python sue_driver.py authorize-resume && " + RESUME_CMD,
                     f"Last job `{job_id}`; run dir {rd}. Trainers save model/optimizer/RNG/data cursor/target bank in "
                     "their verified resumable slot and resume in place (--run-dir). Completed jobs are skipped. "
                     "Phase 2B of semantic-edl-long-k is unrelated and stays blocked.")
        raise SystemExit(rc if isinstance(rc, int) else 1)
    return rc


def drop_slot(rn, why):
    from csr.sue_ckpt import verify_dir
    rd, st = run_dirs().get(rn), status_of(rn)
    if not rd or not st:
        return
    for c in (st.get("ckpts") or {}).values():
        if not verify_dir(guard.ROOT / c["path"]):
            raise guard.GuardError(f"milestone {c['path']} fails verification; slot kept")
    sl = TMP / "slots" / Path(rd).name
    if sl.exists():  # operator rule 2026-11-17: no deletion without permission -> the final slot is retained
        nb = sum(p.stat().st_size for p in sl.rglob("*") if p.is_file())
        append("retained_slots.tsv", f"{guard.now()}\t{sl.relative_to(guard.ROOT)}\t{nb}\t{why}; retained (not deleted)\n")


STAGE_DISK_GIB = {"search": 8.0, "promote": 34.0, "confsel": 1.0, "finals": 21.0}


def stage_disk(stage):
    """Operator rule: estimate peak storage before each stage; stop cleanly if free - estimate < 100 GiB reserve."""
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


def jl(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


# ------------------------------------------------------------------------------------------------ grouping
def resolve():
    """Prereg section 3 rule, from Task-1 artifacts only (read-only)."""
    cal = jl(SLK_ART / "phase2/calibsel_2a.json")
    if cal and cal.get("confirmed"):
        dec = jl(SLK_ART / "phase2/decision_2a.json")
        w = cal["winner"]
        s = dec["screen"][w]
        K = s["K"]
        tb = jl(SLK_ART / "latent/tables.json")[f"K{K}"]
        cand = s.get("cand") or {}
        tt = s.get("tables") or {}
        g = {"source": f"semantic-edl-long-k Phase-2A confirmed winner {w}", "K": K,
             "table": tt.get("sem", tb["sem"]), "table_sha256": tt.get("sem_sha256", tb["sem_sha256"]),
             "shuf_table": tt.get("shuf", tb["shuf"]), "shuf_table_sha256": tt.get("shuf_sha256", tb["shuf_sha256"]),
             "beta": cand.get("beta", C6_BETA), "T": cand.get("T", 0.5)}
    else:
        tb = jl(SLK_ART / "latent/tables.json")["K16"]
        g = {"source": "fallback: no CALIB-SELECTION-confirmed Phase-2A winner"
                       + (f" (calibsel_2a: winner {cal['winner']}, confirmed {cal['confirmed']})" if cal else
                          " (no K qualified for CALIB-SELECTION)"),
             "K": 16, "table": C6_TABLE, "table_sha256": C6_TABLE_SHA, "shuf_table": C6_SHUF_TABLE,
             "shuf_table_sha256": C6_SHUF_TABLE_SHA, "beta": C6_BETA, "T": 0.5}
        tb_ok = tb["sem_sha256"] == C6_TABLE_SHA and tb["shuf_sha256"] == C6_SHUF_TABLE_SHA
        if not tb_ok:
            raise guard.GuardError("K16 tables.json does not reproduce the C6 tables")
    g.update({"calib_search": tb["calib_search"], "calib_search_sha256": tb["calib_search_sha256"],
              "calib_sel": tb["calib_sel"], "calib_sel_sha256": tb["calib_sel_sha256"], "layers": "all 11 MoE layers"})
    for k in ("table", "shuf_table", "calib_search", "calib_sel"):
        if guard.sha256_file(guard.ROOT / g[k]) != g[f"{k}_sha256"]:
            raise guard.GuardError(f"hash mismatch {g[k]}")
    g["slk_state"] = (jl(SLK_ART / "STATE.json") or {}).get("state")
    g["slk_gate_2b"] = jl(SLK_ART / "phase2/gate_2b.json")
    return g


def grouping():
    g = jl(ART / "grouping.json")
    if g is None:
        raise guard.GuardError("grouping not frozen (sue_run.py freeze)")
    return g


def freeze_prereg():
    if PREREG_SHA_FILE.exists():
        raise guard.GuardError("prereg already frozen")
    g = resolve()
    st = (jl(SLK_ART / "STATE.json") or {}).get("state")
    if st != "STOPPED_BEFORE_PHASE2B_BY_USER":
        raise guard.GuardError(f"Task 1 is not at its Phase-2A terminal state ({st})")
    txt = (f"\nResolved {guard.now()} from read-only Task-1 artifacts (semantic-edl-long-k state {st}), before any "
           f"Task-2 candidate result:\n\n```\n{json.dumps({k: v for k, v in g.items() if k != 'slk_gate_2b'}, indent=1)}\n```\n")
    with open(PREREG, "a") as f:
        f.write(txt)
    PREREG.chmod(0o400)
    PREREG_SHA_FILE.write_text(f"{guard.sha256_file(PREREG)}  {PREREG.name}\n")
    ART.mkdir(parents=True, exist_ok=True)
    guard.atomic_write_json(ART / "grouping.json", {"time": guard.now(), **g}, mode=0o400)
    print(json.dumps(g, indent=1))


# ------------------------------------------------------------------------------------------------ training / eval
def cap_ok(phase, est):
    from csr.sue_paths import GPU_CAP_H as C
    ph = {"search": ("search",), "final": ("final",), "diagnostic": ("diagnostic",)}[phase]
    return gpu_hours(ph) + est <= C[phase] and gpu_hours() + est <= C["total"]


def train_to(rn, seed, cond, max_steps, phase, category, budget=None, milestones=None, est=0.2):
    from sue_configs import make
    st = status_of(rn)
    if st and (st.get("status") == "COMPLETE" or (max_steps and st.get("steps_done", 0) >= max_steps)):
        return rn
    if not cap_ok(phase, est):
        decision(f"{phase} GPU cap reached", f"{phase} {gpu_hours((phase,)):.2f} h; total {gpu_hours():.2f} h", "-",
                 f"do not train {rn}")
        raise SystemExit(f"{phase} cap reached before {rn}")
    p, cfg = make(rn, seed, cond, category, budget=budget, milestones=milestones)
    for leg in range(6):
        prev = run_dirs().get(rn)
        argv = ["sue_train.py", "--config", str(p)] + (["--max-steps", str(max_steps)] if max_steps else []) + \
            (["--run-dir", prev] if prev else [])
        job(f"train_{rn}_to{max_steps or 'end'}_leg{leg}", argv, phase, "train", rn, seed, p, expect=(0, 75), disk_gb=4.0)
        st = status_of(rn)
        if st and (st.get("status") == "COMPLETE" or (max_steps and st.get("steps_done", 0) >= max_steps)):
            break
    st = status_of(rn)
    results(f"{rn} | train to {st['steps_done']} steps | tokens {st['seen']}; status {st['status']}; tok/s "
            f"{st.get('tokens_per_s', 0):.0f}; cf_s {st.get('cf_seconds', 0):.0f}; util r "
            f"{(st.get('util_last_diag') or {}).get('split_half_r')} |")
    return rn


def evaluate(rn, mode, tag, phase, key=None, extra=()):
    g = grouping()
    key = key or rn
    outp = ART / "evals" / key / f"{mode}_{tag}.json"
    fresh = not outp.exists()
    if fresh:
        argv = ["sue_evaluate.py", mode, "--run-dir", run_dirs().get(rn, rn), "--run-name", key, "--tag", tag,
                "--pseudo-table", g["calib_search"]] + (["--pseudo-table-sel", g["calib_sel"]] if mode != "dev" else []) \
            + list(extra)
        job(f"{mode}_{key}_{tag}", argv, phase, "eval", key, None, disk_gb=0.5)
    e = jl(outp)
    pe = e["per_expert"]["pseudo"]
    au = e["audit"]["pseudo"]["routing"]
    if fresh:  # cached evaluations are not re-appended on relaunch
        results(f"{key} | {mode} {tag} ({e['block']}) | CE {e['ce_block']['ce']:.4f}; pseudo interaction "
            f"{pe['mean_interaction']:.6f} [{pe['mean_interaction_ci95'][0]:.6f}, {pe['mean_interaction_ci95'][1]:.6f}]; "
            f"spec {pe['n_subject_specialized']}; pseudo MI {au['mi_mean']:.4f} NMI {au.get('nmi_mean', 'n/a')}; e5 interaction "
            f"{e['per_expert']['e5']['mean_interaction']:.6f}; load max {e['pathology']['load_max_x_uniform']:.2f} min "
            f"{e['pathology']['load_min_x_uniform']:.2f} ent {e['pathology']['routing_entropy']:.3f} |")
    return e


def summary(e):
    pe, au, pa = e["per_expert"]["pseudo"], e["audit"]["pseudo"]["routing"], e["pathology"]
    return {"ce": e["ce_block"]["ce"], "inter": pe["mean_interaction"], "inter_ci": pe["mean_interaction_ci95"],
            "boot": pe["boot_mean_samples"], "spec": pe["n_subject_specialized"], "mi": au["mi_mean"],
            "nmi": au.get("nmi_mean"), "inter_e5": e["per_expert"]["e5"]["mean_interaction"],
            "generic": float(sum(e["per_expert"]["generic_importance_mean_dce"]) /
                             len(e["per_expert"]["generic_importance_mean_dce"])),
            "pathology": bool(pa["load_min_x_uniform"] < 0.1 or pa["load_max_x_uniform"] > 4.0
                              or pa["routing_entropy"] < 0.45), **pa}


def run_metrics(rn):
    rd = run_dirs()[rn]
    rows = [json.loads(x) for x in (guard.ROOT / rd / "metrics.jsonl").read_text().splitlines()]
    return [x for x in rows if "step" in x], [x["util_diag"] for x in rows if "util_diag" in x]


# ------------------------------------------------------------------------------------------------ stages
def diag():
    if (ART / "gamma.json").exists() and (ART / "smoke.json").exists():
        return jl(ART / "gamma.json")
    g = grouping()
    if not (ART / "diagnostic.json").exists():
        ck = f"anchor={ANCHOR},c6_50m={P1['C6']}/ckpt_s50,shuf_50m={P1['SHUF']}/ckpt_s50"
        job("diagnostic", ["sue_diag.py", "--table", g["table"], "--table-sha", g["table_sha256"], "--shuf-table",
                           g["shuf_table"], "--shuf-table-sha", g["shuf_table_sha256"], "--beta", str(g["beta"]),
                           "--ckpts", ck], "diagnostic", "diagnostic", disk_gb=1.0)
    d = jl(ART / "diagnostic.json")
    ga = d["models"]["anchor"]["grad_anchor"]
    if ga["gn_use_unit"] > 0:
        raw = 0.5 * ga["gn_sem_edl_beta"] / ga["gn_use_unit"]
        e = math.floor(math.log10(raw))
        gb = round(raw / 10 ** e) * 10 ** e
    else:
        raw, gb = None, 0.01
    out = {"time": guard.now(), "gamma_base": gb, "raw": raw, "grad_anchor": ga,
           "rule": "0.5 * ||grad beta L_SemEDL|| / ||grad L_use|| (tau 0.2, anchor bank), 1 significant digit; 0.01 "
                   "if ||grad L_use|| = 0"}
    if not (ART / "gamma.json").exists():
        guard.atomic_write_json(ART / "gamma.json", out, mode=0o400)
    out = jl(ART / "gamma.json")
    gb = out["gamma_base"]
    # GPU smoke (not a candidate): 30 steps, onset 10, refresh 10 -> exercises compiled training + eager probes in bf16
    from sue_configs import util_cond
    train_to("SMOKE_s100", 100, util_cond(g["table"], g["table_sha256"], g["beta"], gb, 0.2, 0.995, 10, m_min=0.5,
                                          n_min=2), 30, "diagnostic", "calibration", est=0.1)
    rows, dg = run_metrics("SMOKE_s100")
    sm = {"cf_s_mean": sum(r.get("cf_s", 0) for r in rows) / len(rows), "dt_mean": sum(r["dt"] for r in rows[5:]) / max(1, len(rows[5:])),
          "use_active_steps": sum(1 for r in rows if r.get("use_active", 0) > 0), "n_refresh": len(dg),
          "last_util_diag": {k: dg[-1].get(k) for k in ("n_obs", "coverage_valid", "split_half_r")} if dg else None,
          "lm_last": rows[-1]["lm"]}
    sm["overhead_frac"] = sm["cf_s_mean"] / max(1e-9, sm["dt_mean"] - sm["cf_s_mean"])
    guard.atomic_write_json(ART / "smoke.json", sm, mode=0o400)
    drop_slot("SMOKE_s100", "GPU smoke run (not a candidate)")
    results(f"SMOKE_s100 | 30-step GPU smoke | {json.dumps(sm)} |")
    for name, m in d["models"].items():
        last = m["by_size"]["SEM"]["1000"]
        lastS = m["by_size"]["SHUF"]["1000"]
        results(f"diag {name} | 1000 probe steps | SEM split-half r {last['split_half_r']:.3f} (SB "
                f"{last['spearman_brown']:.3f}), coverage {last['coverage_valid']:.3f}, own margin "
                f"{last['own_margin_mean']:.6f} {last['own_margin_halves']}; SHUF r {lastS['split_half_r']:.3f}, own "
                f"margin {lastS['own_margin_mean']:.6f}; noise floor mean {m['noise_floor']['abs_ce_change_unmodified_rows']['mean']:.2e} |")
    decision("Diagnostic and gamma anchor", json.dumps({n: {s: {k: m['by_size'][s]['1000'][k] for k in
                                                                ('split_half_r', 'own_margin_mean', 'coverage_valid')}
                                                            for s in ('SEM', 'SHUF')} for n, m in d['models'].items()})
             + f"; grad anchor {json.dumps(ga)}", "-", f"gamma_base = {gb} (prereg rule); proceed to the search "
             "(the diagnostic is recorded, never a stopping rule)")
    return out


def candidates():
    g, gb = grouping(), diag()["gamma_base"]
    base = {"gamma": gb, "tau": 0.2, "rho": 0.995, "onset": 100, "layers": None}
    var = {"U1": {}, "U2": {"gamma": gb * 0.3}, "U3": {"gamma": gb * 3}, "U4": {"tau": 0.05}, "U5": {"tau": 1.0},
           "U6": {"rho": 0.98}, "U7": {"onset": 30}, "U8": {"layers": [5, 6, 7, 8, 9, 10]}}
    return {k: {**base, **v} for k, v in var.items()}, g


def cond_of(c, table, sha, beta, **over):
    from sue_configs import util_cond
    c = {**c, **over}
    return util_cond(table, sha, beta, float(f"{c['gamma']:.6g}"), c["tau"], c["rho"], c["onset"], c["layers"],
                     **{k: v for k, v in c.items() if k in ("probe_passes",)})


def ref_cond(g, shuf=False):
    from sue_configs import FIXED  # noqa: F401
    t, s = (g["shuf_table"], g["shuf_table_sha256"]) if shuf else (g["table"], g["table_sha256"])
    return {"paper_latent": {"table": t, "table_sha256": s, "beta": g["beta"]}}


def use_active_frac(rn, onset):
    rows, _ = run_metrics(rn)
    post = [r for r in rows if r["step"] >= onset]
    return float(sum(1 for r in post if r.get("use_active", 0) > 0) / max(1, len(post)))


def search():
    p = ART / "search_decision.json"
    if p.exists():
        return jl(p)
    cands, g = candidates()
    train_to("REF_C6_s100", 100, ref_cond(g), S20, "search", "search")
    ref = summary(evaluate("REF_C6_s100", "dev", "s20", "search"))
    rows = {}
    for name, c in cands.items():
        rn = f"S_{name}_s100"
        train_to(rn, 100, cond_of(c, g["table"], g["table_sha256"], g["beta"]), S20, "search", "search")
        s = summary(evaluate(rn, "dev", "s20", "search"))
        s["use_active_frac"] = use_active_frac(rn, c["onset"])
        _, dg = run_metrics(rn)
        s["util_last"] = {k: dg[-1].get(k) for k in ("split_half_r", "coverage_valid", "n_obs")} if dg else None
        s["d_inter"], s["d_ce"] = s["inter"] - ref["inter"], s["ce"] - ref["ce"]
        s["eligible"] = bool(s["d_inter"] > 0 and s["d_ce"] <= 0.010 and not s["pathology"]
                             and s["use_active_frac"] >= 0.5)
        rows[name] = {k: v for k, v in s.items() if k != "boot"}
    el = sorted([k for k, v in rows.items() if v["eligible"]], key=lambda k: (-rows[k]["d_inter"], rows[k]["ce"]))
    # ties within 2e-5 -> lower CE
    if len(el) > 1:
        top = rows[el[0]]["d_inter"]
        tied = [k for k in el if top - rows[k]["d_inter"] <= 2e-5]
        el = sorted(tied, key=lambda k: rows[k]["ce"]) + [k for k in el if k not in tied]
    out = {"time": guard.now(), "reference": {k: v for k, v in ref.items() if k != "boot"}, "candidates": rows,
           "candidate_specs": cands, "promoted": el[:2]}
    guard.atomic_write_json(p, out, mode=0o400)
    decision("Search (20M, seed 100) complete", json.dumps({k: {x: rows[k][x] for x in ("d_inter", "d_ce", "eligible",
                                                                                      "use_active_frac")}
                                                           for k in rows}), "-",
             f"promote {el[:2]}" if el else "no eligible candidate: diagnose per prereg section 9")
    return out


def run_pair(tag, c, g, seeds=SEARCH_SEEDS, sem_name=None):
    """SEM/SHUF utility runs to 50M + dev s50; returns per-seed summaries."""
    out = {}
    for seed in seeds:
        rs = sem_name(seed) if sem_name else f"S_{tag}_s{seed}"
        rh = f"H_{tag}_s{seed}"
        train_to(rs, seed, cond_of(c, g["table"], g["table_sha256"], g["beta"]), S50, "search", "search")
        train_to(rh, seed, cond_of(c, g["shuf_table"], g["shuf_table_sha256"], g["beta"]), S50, "search", "search")
        es, eh = summary(evaluate(rs, "dev", "s50", "search")), summary(evaluate(rh, "dev", "s50", "search"))
        for rn in (rs, rh):
            drop_slot(rn, "search run complete at its 50M milestone")
        out[seed] = {"SEM": es, "SHUF": eh, "sem_run": rs, "shuf_run": rh}
    return out


def judge(pairs):
    d = {s: {"d_inter": v["SEM"]["inter"] - v["SHUF"]["inter"], "d_ce": v["SEM"]["ce"] - v["SHUF"]["ce"],
             "path": v["SEM"]["pathology"] or v["SHUF"]["pathology"]} for s, v in pairs.items()}
    mean = {k: sum(x[k] for x in d.values()) / len(d) for k in ("d_inter", "d_ce")}
    ok = bool(all(x["d_inter"] > 0 for x in d.values()) and mean["d_ce"] <= 0.005 and not any(x["path"] for x in d.values()))
    return {"per_seed": d, "mean": mean, "pass": ok}


def references(g):
    out = {}
    for seed in SEARCH_SEEDS:
        for name, shuf in (("REF_C6", False), ("REF_SHUF", True)):
            rn = f"{name}_s{seed}"
            train_to(rn, seed, ref_cond(g, shuf), S50, "search", "search")
            out[rn] = summary(evaluate(rn, "dev", "s50", "search"))
            drop_slot(rn, "reference run complete at its 50M milestone")
    return out


def diagnose(tag, pairs, refs, c):
    """Prereg section 9, first matching failure."""
    j = judge(pairs)
    info = {}
    for seed, v in pairs.items():
        rows, dg = run_metrics(v["sem_run"])
        post = [r for r in rows if r["step"] >= c["onset"] and "use_margin" in r]
        m0 = post[0]["use_margin"] if post else float("nan")
        m1 = sum(r["use_margin"] for r in post[-5:]) / max(1, len(post[-5:])) if post else float("nan")
        r50 = [x for x in dg if x["step"] <= S50][-1]["split_half_r"] if dg else float("nan")
        info[seed] = {"split_half_r": r50, "margin_onset": m0, "margin_end": m1,
                      "mi_sem": v["SEM"]["mi"], "mi_ref_c6": refs[f"REF_C6_s{seed}"]["mi"]}
    if any(x["path"] for x in j["per_seed"].values()):
        return "pathology", info
    if any(not (x["split_half_r"] >= 0.2) for x in info.values()):
        return "noisy", info
    no_resp = any(not (x["margin_end"] > x["margin_onset"]) or x["mi_sem"] <= x["mi_ref_c6"] for x in info.values())
    if no_resp:
        return "no_routing_response", info
    if all(x["d_inter"] > 0 for x in j["per_seed"].values()) and j["mean"]["d_ce"] > 0.005:
        return "ce_harm", info
    if any(x["d_inter"] <= 0 for x in j["per_seed"].values()):
        return "routing_without_function", info
    return "sem_equals_shuf", info


GEN_RULES = {"noisy": ("support/EMA", [{"probe_passes": 8, "rho": 0.998}]),
             "no_routing_response": ("strength/temperature", [{"gamma_x": 3, "tau_x": 0.5}, {"gamma_x": 10, "tau_x": 1.0}]),
             "ce_harm": ("onset/strength", [{"onset_x": 2, "gamma_x": 0.5}])}


def promote():
    p = ART / "frozen.json"
    if p.exists():
        return jl(p)
    sd = search()
    cands, g = candidates()
    refs = references(g)
    tried = {}
    queue = [(k, cands[k]) for k in sd["promoted"]]
    if not queue:  # diagnose using the search failure: run U1 pair as the measured basis
        decision("No search candidate eligible", json.dumps({k: (v['d_inter'], v['d_ce'], v['use_active_frac'])
                                                             for k, v in sd['candidates'].items()}),
                 "diagnosis needs the matched 50M measurement", "run the base candidate U1 at 50M with SHUF controls "
                 "as the diagnostic basis (it can also pass the section-8 rule)")
        queue = [("U1", cands["U1"])]
    for k, c in queue:
        pr = run_pair(k, c, g, sem_name=(lambda s, k=k: f"S_{k}_s{s}"))
        tried[k] = {"spec": c, "pairs": strip(pr), "judge": judge(pr), "gen": 0}
        results(f"promotion {k} | 50M SEM vs SHUF | {json.dumps(tried[k]['judge'])} |")
    passers = [k for k, v in tried.items() if v["judge"]["pass"]]
    n_gen = 0
    while not passers and n_gen < 2:
        rnd = [k for k, v in tried.items() if v["gen"] == n_gen]
        last = max(rnd, key=lambda k: tried[k]["judge"]["mean"]["d_inter"])  # best of the most recent round
        cause, info = diagnose(last, rehydrate(tried[last]), refs, tried[last]["spec"])
        tried[last]["diagnosis"] = {"cause": cause, "info": info}
        if cause not in GEN_RULES:
            decision(f"Diagnosis after {last}: {cause}", json.dumps(info), cause,
                     "stop the search (prereg section 9: reject / conclude; no generation)")
            break
        n_gen += 1
        fam, mods = GEN_RULES[cause]
        base = tried[last]["spec"]
        gc = []
        for mdx in mods:
            c = dict(base)
            if "gamma_x" in mdx:
                c["gamma"] = base["gamma"] * mdx["gamma_x"]
            if "tau_x" in mdx:
                c["tau"] = base["tau"] * mdx["tau_x"]
            if "onset_x" in mdx:
                c["onset"] = int(base["onset"] * mdx["onset_x"])
            for kk in ("probe_passes", "rho"):
                if kk in mdx:
                    c[kk] = mdx[kk]
            gc.append(c)
        doc = guard.ROOT / "docs" / f"SUE_GEN{n_gen}.md"
        if not doc.exists():
            doc.write_text(f"# SUE engineering generation {n_gen} — {fam}\n\nWritten {guard.now()} before any run of this "
                           f"generation.\n\nMeasured failure ({last}): {cause}\n\nEvidence: {json.dumps(info)}\n\n"
                           f"Change family: {fam} (prereg section 9). Candidates (50M, seeds 100/101, SHUF controls):\n\n"
                           + "".join(f"- {json.dumps(c)}\n" for c in gc) + "\nAcceptance: prereg section-8 rule.\n")
            doc.chmod(0o400)
            (doc.parent / f"SUE_GEN{n_gen}.md.sha256").write_text(f"{guard.sha256_file(doc)}  {doc.name}\n")
        decision(f"Diagnosis after {last}: {cause}", json.dumps(info), cause,
                 f"generation {n_gen} ({fam}): {gc} (docs/SUE_GEN{n_gen}.md sha {guard.sha256_file(doc)[:12]})")
        for i, c in enumerate(gc):
            k = f"G{n_gen}{chr(97 + i)}"
            pr = run_pair(k, c, g)
            tried[k] = {"spec": c, "pairs": strip(pr), "judge": judge(pr), "gen": n_gen}
            results(f"generation {n_gen} {k} | 50M SEM vs SHUF | {json.dumps(tried[k]['judge'])} |")
        passers = [k for k, v in tried.items() if v["judge"]["pass"]]
    if not passers and n_gen:
        rnd = [k for k, v in tried.items() if v["gen"] == n_gen]
        last = max(rnd, key=lambda k: tried[k]["judge"]["mean"]["d_inter"])
        cause, info = diagnose(last, rehydrate(tried[last]), refs, tried[last]["spec"])
        tried[last]["diagnosis"] = {"cause": cause, "info": info}
        decision(f"Final diagnosis after generation {n_gen} ({last}): {cause}", json.dumps(info), cause,
                 "stop the search (generation limit reached)")
    frozen = max(passers, key=lambda k: tried[k]["judge"]["mean"]["d_inter"]) if passers else None
    out = {"time": guard.now(), "tried": tried, "references_50m": {k: {x: y for x, y in v.items() if x != "boot"}
                                                                   for k, v in refs.items()},
           "n_generations": n_gen, "frozen": frozen, "frozen_spec": tried[frozen]["spec"] if frozen else None}
    guard.atomic_write_json(p, out, mode=0o400)
    decision("Freeze", json.dumps({k: v["judge"] for k, v in tried.items()}), "-",
             f"frozen candidate {frozen}" if frozen else "no candidate passes: no CALIB-SELECTION, no finals")
    return out


def strip(pairs):
    return {str(s): {"SEM": {k: v for k, v in x["SEM"].items() if k != "boot"},
                     "SHUF": {k: v for k, v in x["SHUF"].items() if k != "boot"},
                     "sem_run": x["sem_run"], "shuf_run": x["shuf_run"]} for s, x in pairs.items()}


def rehydrate(t):
    return {int(s): v for s, v in t["pairs"].items()}


def paired_ci(diffs_boot):
    import numpy as np
    m = np.mean(np.asarray(diffs_boot), 0)
    return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def confsel():
    p = ART / "confsel.json"
    if p.exists():
        return jl(p)
    fz = promote()
    if not fz["frozen"]:
        return {"confirmed": False, "reason": "no frozen candidate"}
    if not (ART / "CONFSEL_OPEN.json").exists():
        guard.atomic_write_json(ART / "CONFSEL_OPEN.json", {"time": guard.now(), "frozen": fz["frozen"]}, mode=0o400)
    t = fz["tried"][fz["frozen"]]
    rows, boots = {}, []
    for s, v in t["pairs"].items():
        es = summary(evaluate(v["sem_run"], "confsel", "s50", "search"))
        eh = summary(evaluate(v["shuf_run"], "confsel", "s50", "search"))
        import numpy as np
        boots.append(np.asarray(es["boot"]) - np.asarray(eh["boot"]))
        rows[s] = {"SEM": {k: x for k, x in es.items() if k != "boot"}, "SHUF": {k: x for k, x in eh.items() if k != "boot"},
                   "d_inter": es["inter"] - eh["inter"], "d_ce": es["ce"] - eh["ce"]}
    ci = paired_ci(boots)
    conf = bool(all(r["d_inter"] > 0 for r in rows.values()) and sum(r["d_ce"] for r in rows.values()) / len(rows)
                <= 0.005 and ci[0] > 0)
    out = {"time": guard.now(), "frozen": fz["frozen"], "rows": rows, "pooled_paired_ci95": ci, "confirmed": conf}
    guard.atomic_write_json(p, out, mode=0o400)
    results(f"CALIB-SELECTION half 0 (once) | {fz['frozen']} | d_inter {[r['d_inter'] for r in rows.values()]}; "
            f"d_ce {[r['d_ce'] for r in rows.values()]}; pooled CI {ci}; confirmed {conf} |")
    decision("CALIB-SELECTION confirmation", json.dumps({s: (r["d_inter"], r["d_ce"]) for s, r in rows.items()}) +
             f"; CI {ci}", "-", "train finals" if conf else "not confirmed: no finals")
    return out


def finals():
    cs = confsel()
    if not cs.get("confirmed"):
        return {"finals": False}
    fz = jl(ART / "frozen.json")
    g, c = grouping(), fz["frozen_spec"]
    for seed in FINAL_SEEDS:
        train_to(f"F_SEM_s{seed}", seed, cond_of(c, g["table"], g["table_sha256"], g["beta"]), None, "final", "final", est=0.9)
        train_to(f"F_SHUF_s{seed}", seed, cond_of(c, g["shuf_table"], g["shuf_table_sha256"], g["beta"]), None, "final",
                 "final", est=0.9)
    sts = [status_of(f"F_SEM_s{s}") for s in FINAL_SEEDS]
    ovh = sum(s["cf_seconds"] for s in sts) / max(1e-9, sum(s["train_step_seconds"] - s["cf_seconds"] for s in sts))
    cm = None
    if ovh > 0.05:
        from csr.sue_paths import FINAL_BUDGET
        budget = int(FINAL_BUDGET * (1 + ovh))
        cm = {"overhead": ovh, "budget": budget}
        for seed in FINAL_SEEDS:
            if not cap_ok("final", 0.9):
                decision("Compute-matched TOKEN-4: budget", f"final {gpu_hours(('final',)):.2f} h", "-",
                         f"stop compute-matched runs before seed {seed}")
                break
            from csr.lfr_lib import main_stream
            _, stp, _ = main_stream(budget)
            train_to(f"CM_T4_s{seed}", seed, {}, None, "final", "final", budget=budget,
                     milestones={"p100": len(stp)}, est=1.0)
    if not (ART / "FINAL_OPEN.json").exists():
        guard.atomic_write_json(ART / "FINAL_OPEN.json", {"time": guard.now(), "overhead": ovh, "compute_matched": cm},
                                mode=0o400)
    from csr.slk_paths import REUSED as SLK_REUSED  # read-only constants of Task 1
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
    crit = {"S1_every_seed": all(x > 0 for x in d), "S2_ci_excludes_0": ci[0] > 0,
            "S2_exceeds_seed_noise": float(np.mean(d)) > sd_shuf, "S3_no_pathology": not any(path),
            "S4_ce_vs_t4": float(np.mean(d_t4)) <= 0.010, "S4_ce_vs_edl": float(np.mean(d_edl)) <= 0.010,
            "S5_mass_matched": float(np.mean(mass)) > 0}
    out = {"time": guard.now(), "d_inter": d, "paired_ci95": ci, "sd_shuf_seed": sd_shuf, "d_ce_vs_t4": d_t4,
           "d_ce_vs_edl": d_edl, "pathology": path, "mass_matched": mass, "criteria": crit,
           "success": bool(all(crit.values())), "summaries": {k: {x: y for x, y in v.items() if x != "boot"}
                                                              for k, v in sm.items()}}
    guard.atomic_write_json(p, out, mode=0o400)
    decision("Final preregistered decision", json.dumps(crit), "-", f"SEM-UTILITY {'SUCCEEDS' if out['success'] else 'does NOT succeed'}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["resolve", "freeze", "diag", "search", "promote", "confsel", "finals", "all"])
    a = ap.parse_args()
    guard.check_root()
    if a.cmd == "resolve":
        print(json.dumps(resolve(), indent=1))
        return
    if a.cmd == "freeze":
        freeze_prereg()
        return
    from csr.sue_paths import check_prereg
    check_prereg()
    try:
        if a.cmd in ("diag", "all"):
            diag()
        if a.cmd in ("search", "all"):
            if not ({"search": ART / "search_decision.json", "promote": ART / "frozen.json", "confsel": ART / "confsel.json",
                     "finals": ART / "final_decision.json"}["search"]).exists():
                stage_disk("search")
            search()
        if a.cmd in ("promote", "all"):
            if not ({"search": ART / "search_decision.json", "promote": ART / "frozen.json", "confsel": ART / "confsel.json",
                     "finals": ART / "final_decision.json"}["promote"]).exists():
                stage_disk("promote")
            promote()
        if a.cmd in ("confsel", "all"):
            if not ({"search": ART / "search_decision.json", "promote": ART / "frozen.json", "confsel": ART / "confsel.json",
                     "finals": ART / "final_decision.json"}["confsel"]).exists():
                stage_disk("confsel")
            confsel()
        if a.cmd in ("finals", "all"):
            if not ({"search": ART / "search_decision.json", "promote": ART / "frozen.json", "confsel": ART / "confsel.json",
                     "finals": ART / "final_decision.json"}["finals"]).exists():
                stage_disk("finals")
            r = finals()
            write_resume("TERMINAL — all preregistered stages complete" + ("" if r.get("finals", True) else
                                                                          " (finals not reached by rule)"),
                         ".venv/bin/python sue_analyze.py", "Nothing to resume; analysis/report only.")
    except SystemExit:
        raise


if __name__ == "__main__":
    main()
