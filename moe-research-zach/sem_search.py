#!/usr/bin/env python
"""csr-semantic-edl-v1 orchestrator (prereg sections 4-10). Every GPU job goes through sem_driver.run_job (idempotent;
external interruption latches and stops). Phases:

  sem_search.py stage0        GPU checks, explicit-EDL seed-100 baseline (50M), baseline dev evals, calibration pilots
  sem_search.py stage1        C1-C8 at 20M (seed 100) + dev evals; decision (search/stage1_decision.json)
  sem_search.py stage2        best two: seed 100 -> 50M (continued), seed 101 50M, SHUF controls, T4/EDL s101; decision
  sem_search.py gen --n N     engineering generation N (requires hashed docs/SEMANTIC_EDL_V1_GEN<N>.md + search/genN.json)
  sem_search.py freeze        frozen.json, opens CALIB-SELECTION once, confirmation decision
  sem_search.py finals        (only if confirmed) final training, VAL table, VAL evaluations
"""
import argparse
import json
import math
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402
from csr.sem_paths import (ALL, ART, BETA0, EDL_BETA, FINAL_RUNS, GPU_BUDGET_H, S20, S50, SEM_TABLE,  # noqa: E402
                           T4_S100_RUN, TMP, UPPER)

SD = ART / "search"
RUNS = ART / "run_dirs.json"
TABLES = {"soft": (str(SEM_TABLE.relative_to(guard.ROOT)), "artifacts/semantic-edl-v1/tables/table_SHUF_SEM.npz"),
          "sharp": ("artifacts/semantic-edl-v1/tables/table_SEM_T05.npz",
                    "artifacts/semantic-edl-v1/tables/table_SHUF_SEM_T05.npz")}
PLACE = {"all": None, "upper": UPPER, "lower": list(range(0, 5))}
STAGE1 = [{"id": "C1", "placement": "all", "assign": "soft", "target": 0.02, "ramp": 0.0},
          {"id": "C2", "placement": "all", "assign": "soft", "target": 0.01, "ramp": 0.0},
          {"id": "C3", "placement": "all", "assign": "soft", "target": 0.005, "ramp": 0.0},
          {"id": "C4", "placement": "upper", "assign": "soft", "target": 0.02, "ramp": 0.0},
          {"id": "C5", "placement": "upper", "assign": "soft", "target": 0.01, "ramp": 0.0},
          {"id": "C6", "placement": "all", "assign": "sharp", "target": 0.02, "ramp": 0.0},
          {"id": "C7", "placement": "all", "assign": "soft", "target": 0.02, "ramp": 0.02},
          {"id": "C8", "placement": "all", "assign": "sharp", "target": 0.01, "ramp": 0.0}]


# ------------------------------------------------------------------------------------------------ infrastructure
def gpu_hours():
    p = ART / "compute_ledger.tsv"
    if not p.exists():
        return 0.0
    tot = 0.0
    for line in p.read_text().splitlines():
        f = line.split("\t")
        try:
            tot += float(f[2])
        except (IndexError, ValueError):
            pass
    return tot / 3600


def budget_ok(need_h, what):
    used = gpu_hours()
    if used + need_h > GPU_BUDGET_H:
        log_event(f"BUDGET: {what} needs {need_h:.2f} h, used {used:.2f} h of {GPU_BUDGET_H}")
        return False
    return True


def log_event(msg):
    with open(ART / "driver_log.txt", "a") as f:
        f.write(f"{guard.now()} {msg}\n")
    print(msg, flush=True)


def run_dirs():
    return json.loads(RUNS.read_text()) if RUNS.exists() else {}


def remember(rn):
    from csr.sem_paths import RUN_PREFIX
    d = run_dirs()
    if rn not in d:
        c = sorted((guard.ROOT / "runs").glob(f"*_{RUN_PREFIX}{rn}"))
        if c:
            d[rn] = str(c[-1].relative_to(guard.ROOT))
            guard.atomic_write_json(RUNS, d, overwrite=True)
    return d.get(rn)


def train(rn, cond, kind, diag, max_steps, seed):
    from sem_configs import make
    from sem_driver import run_job
    p, _ = make(rn, seed, cond, kind, diag_every=diag)
    prev = run_dirs().get(rn)
    args = ["sem_train.py", "--config", str(p), "--max-steps", str(max_steps)] + (["--run-dir", prev] if prev else [])
    run_job(f"train_{rn}_to{max_steps}", args, gpu=True, expect=(75, 0), disk_gb=4.0)
    return remember(rn)


def drop_slot(rn, why):
    from csr.sem_ckpt import verify_dir
    rd = run_dirs().get(rn)
    if not rd:
        return
    st = json.loads((guard.ROOT / rd / "status.json").read_text())
    for c in st.get("ckpts", {}).values():
        if not verify_dir(guard.ROOT / c["path"]):
            raise guard.GuardError(f"milestone {c['path']} fails verification; slot kept")
    sl = TMP / "slots" / Path(rd).name
    if sl.exists():
        nb = sum(p.stat().st_size for p in sl.rglob("*") if p.is_file())
        shutil.rmtree(sl)
        with open(ART / "deleted_artifacts.tsv", "a") as f:
            f.write(f"{guard.now()}\t{sl.relative_to(guard.ROOT)}\t{nb}\t{why}\n")


def dev(rn, tag, run_dir=None):
    from sem_driver import run_job
    rd = run_dir or run_dirs()[rn]
    run_job(f"dev_{rn}_{tag}", ["sem_evaluate.py", "dev", "--run-dir", rd, "--run-name", rn, "--tag", tag], gpu=True,
            disk_gb=0.5)
    return json.loads((ART / "evals" / rn / f"dev_{tag}.json").read_text())


def diag_of(rn_or_dir):
    d = run_dirs().get(rn_or_dir, rn_or_dir)
    return [json.loads(x)["diag"] for x in (guard.ROOT / d / "metrics.jsonl").read_text().splitlines()
            if "diag" in x]


def ratio_median(diags, lo, hi):
    import numpy as np
    r = [x["gn_aux"] / x["gn_lm"] for x in diags if lo <= x["step"] <= hi and x.get("gn_lm")]
    return float(np.median(r)) if r else None


def ledger(rec):
    with open(SD / "ledger.jsonl", "a") as f:
        f.write(json.dumps({"time": guard.now(), **rec}, default=str) + "\n")


# ------------------------------------------------------------------------------------------------ stage 0
def stage0():
    from sem_driver import run_job
    run_job("gpu_checks", ["sem_gpu_checks.py"], gpu=True, disk_gb=6.0)
    gc_ = json.loads((ART / "gpu_checks.json").read_text())
    if not gc_.get("all_pass"):
        raise SystemExit("GPU checks failed: stop (integrity)")
    train("b_EDL_s100", {"edl_beta": EDL_BETA}, "search", 10, S50, 100)
    drop_slot("b_EDL_s100", "baseline 50M milestone verified; search baseline never resumes")
    for tag in ("s5", "s20", "s50"):
        dev("b_T4_s100", tag, T4_S100_RUN)
        dev("b_EDL_s100", tag)
    calib()


def calib():
    import numpy as np
    from sem_configs import sem_cond
    out = SD / "calibration.json"
    if out.exists():
        return json.loads(out.read_text())
    ed = diag_of("b_EDL_s100")
    rho_edl = ratio_median(ed, 100, 160)
    fle = {}
    for k in ("EDL_s0", "EDL_s1"):
        dd = diag_of(FINAL_RUNS[k])
        fle[k] = {s: next((x["gn_aux"] / x["gn_lm"] for x in dd if x["step"] == s), None) for s in (100, 200, 300)}
    mine = {s: next((x["gn_aux"] / x["gn_lm"] for x in ed if x["step"] == s), None) for s in (100, 200, 300)}
    repro = {s: [mine[s] / fle[k][s] for k in fle] for s in (100, 200, 300)}
    rec = {"rho_edl_100_160": rho_edl, "edl_ratio_s100_steps": mine, "fle_edl_ratio_steps": fle,
           "reproduction_factor": repro,
           "edl_reproduced_within_1.5x": bool(all(1 / 1.5 <= v <= 1.5 for vs in repro.values() for v in vs)),
           "variants": {}}
    for pl, asg in sorted({(c["placement"], c["assign"]) for c in STAGE1}):
        rn = f"p_{pl}_{asg}_s100"
        train(rn, sem_cond(TABLES[asg][0], BETA0, PLACE[pl]), "pilot", 10, 160, 100)
        drop_slot(rn, "pilot slot (metrics kept)")
        rho = ratio_median(diag_of(rn), 100, 160)
        rec["variants"][f"{pl}_{asg}"] = {"rho_beta0": rho, "run": run_dirs()[rn],
                                          "ratios": [(x["step"], x["gn_aux"] / x["gn_lm"]) for x in diag_of(rn)]}
    rec["betas"] = {}
    for c in STAGE1:
        rho = rec["variants"][f"{c['placement']}_{c['assign']}"]["rho_beta0"]
        rec["betas"][c["id"]] = float(f"{BETA0 * (c['target'] / 0.02) * rho_edl / rho:.2g}")
    rec["time"] = guard.now()
    guard.atomic_write_json(out, rec, mode=0o400)
    log_event(f"calibration: {json.dumps(rec['betas'])} rho_edl={rho_edl:.4f}")
    return rec


def cand_cond(c, beta, shuf=False):
    from sem_configs import sem_cond
    return sem_cond(TABLES[c["assign"]][1 if shuf else 0], beta, PLACE[c["placement"]], c.get("ramp") or None)


# ------------------------------------------------------------------------------------------------ summaries
def summ(rn, tag, ref_t4, ref_edl):
    ev = json.loads((ART / "evals" / rn / f"dev_{tag}.json").read_text())
    a = ev["audit"]
    ce = ev["calib_search_B"]["ce"]
    return {"ce": ce, "dce_t4": ce - ref_t4["ce"] if ref_t4 else None, "dce_edl": ce - ref_edl["ce"] if ref_edl else None,
            "fsi_pseudo": a["pseudo"]["fsi"], "fsi_e5": a["e5"]["fsi"], "mi_pseudo": a["pseudo"]["routing"]["mi_mean"],
            "mi_pseudo_null": a["pseudo"]["routing"]["mi_null_mean"], "mi_e5": a["e5"]["routing"]["mi_mean"],
            "topic_entropy_pseudo": a["pseudo"]["routing"]["topic_entropy_mean"],
            "topic_pred_pseudo": a["pseudo"]["topic_pred_acc"], "routing_entropy": a["routing_entropy"],
            "load_min_x": a["load_min_x_uniform"], "load_max_x": a["load_max_x_uniform"],
            "n_starved": a["n_below_0.1_uniform"]}


def base_summ(rn, tag, run_dir=None):
    ev = json.loads((ART / "evals" / rn / f"dev_{tag}.json").read_text())
    return summ(rn, tag, None, None) | {"ce": ev["calib_search_B"]["ce"]}


def grad_summary(rn, lo, hi):
    import numpy as np
    d = [x for x in diag_of(rn) if lo <= x["step"] <= hi]
    if not d:
        return {}
    med = lambda k: float(np.median([x.get(k, 0.0) for x in d]))  # noqa: E731
    return {"gn_lm": med("gn_lm"), "gn_lb_incl_z": med("gn_lb"), "gn_aux": med("gn_aux"),
            "ratio_aux_lm": float(np.median([x["gn_aux"] / x["gn_lm"] for x in d])),
            "ratio_lb_lm": float(np.median([x["gn_lb"] / x["gn_lm"] for x in d])),
            "train_jsd_geo": med("jsd") if "jsd" in d[0] else None, "n": len(d)}


def pathology(s):
    r = []
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (s["ce"], s["fsi_pseudo"], s["fsi_e5"])):
        r.append("non-finite")
    if s["n_starved"] > 0:
        r.append("starvation")
    if s["routing_entropy"] < 0.45 or s["load_max_x"] > 4.0:
        r.append("collapse")
    return r


# ------------------------------------------------------------------------------------------------ stage 1
def stage1():
    cal = calib()
    for c in STAGE1:
        rn = f"s_{c['id']}_s100"
        if not budget_ok(0.15, rn):
            raise SystemExit("budget")
        train(rn, cand_cond(c, cal["betas"][c["id"]]), "search", 25, S20, 100)
        for tag in ("s5", "s20"):
            dev(rn, tag)
    decide1()


def decide1():
    out = SD / "stage1_decision.json"
    if out.exists():
        return json.loads(out.read_text())
    cal = calib()
    t4 = {t: base_summ("b_T4_s100", t) for t in ("s5", "s20")}
    ed = {t: base_summ("b_EDL_s100", t) for t in ("s5", "s20")}
    edl_win = ratio_median(diag_of("b_EDL_s100"), 100, 318)
    rows = {}
    for c in STAGE1:
        rn = f"s_{c['id']}_s100"
        s5, s20 = summ(rn, "s5", t4["s5"], ed["s5"]), summ(rn, "s20", t4["s20"], ed["s20"])
        g = grad_summary(rn, 100, 318)
        rel = g["ratio_aux_lm"] / edl_win
        intended = c["target"] / 0.02
        rej = pathology(s20)
        if s20["dce_t4"] > 0.020:
            rej.append("ce>T4+0.020")
        if not (0.4 * intended <= rel <= 2.5 * intended):
            rej.append(f"ratio outside range (EDL-relative {rel:.3f} vs intended {intended:.3f})")
        if s20["mi_pseudo"] > t4["s20"]["mi_pseudo"] and s20["fsi_pseudo"] <= t4["s20"]["fsi_pseudo"]:
            rej.append("routing without function")
        if s5["fsi_pseudo"] - t4["s5"]["fsi_pseudo"] > 0 and s20["fsi_pseudo"] - t4["s20"]["fsi_pseudo"] <= 0:
            rej.append("instability (s5 advantage lost at s20)")
        rows[c["id"]] = {"cand": c, "beta": cal["betas"][c["id"]], "s5": s5, "s20": s20, "grad": g,
                         "ratio_edl_relative": rel, "intended_edl_relative": intended,
                         "fsi_pseudo_adv_s20": s20["fsi_pseudo"] - t4["s20"]["fsi_pseudo"], "reject": rej}
    surv = [k for k in rows if not rows[k]["reject"]]
    comp = [k for k in surv if rows[k]["s20"]["dce_edl"] <= 0.010]
    ranked = sorted(comp, key=lambda k: (-rows[k]["fsi_pseudo_adv_s20"], rows[k]["s20"]["ce"]))
    rest = sorted([k for k in surv if k not in comp], key=lambda k: rows[k]["s20"]["ce"])
    adv = (ranked + rest)[:2]
    rec = {"time": guard.now(), "t4": t4, "edl": ed, "edl_ratio_100_318": edl_win, "rows": rows, "survivors": surv,
           "ce_comparable_to_edl": comp, "advance": adv,
           "edl_grad": grad_summary("b_EDL_s100", 100, 318)}
    guard.atomic_write_json(out, rec, mode=0o400)
    for k, r in rows.items():
        ledger({"generation": "stage1", "candidate": k, "hypothesis": f"{r['cand']}", "config": r["cand"],
                "beta": r["beta"], "tokens": "20M", "seeds": [100], "ce_s5_s20": [r["s5"]["ce"], r["s20"]["ce"]],
                "dce_t4_s20": r["s20"]["dce_t4"], "dce_edl_s20": r["s20"]["dce_edl"],
                "routing_entropy": r["s20"]["routing_entropy"], "load_min_max_x": [r["s20"]["load_min_x"],
                                                                                 r["s20"]["load_max_x"]],
                "grad_norms": r["grad"], "mi_pseudo": r["s20"]["mi_pseudo"], "fsi_pseudo": r["s20"]["fsi_pseudo"],
                "fsi_e5": r["s20"]["fsi_e5"], "decision": "advance" if k in adv else ("reject: " + "; ".join(
                    r["reject"]) if r["reject"] else "survived, not top-2")})
    for c in STAGE1:
        if c["id"] not in adv:
            drop_slot(f"s_{c['id']}_s100", "stage-1 candidate not advancing (s5/s20 milestones kept)")
    log_event(f"stage1 decision: advance {adv}; survivors {surv}")
    return rec


# ------------------------------------------------------------------------------------------------ stage 2 / gens
def baselines101():
    train("b_T4_s101", {}, "search", 25, S50, 101)
    drop_slot("b_T4_s101", "baseline milestone verified")
    train("b_EDL_s101", {"edl_beta": EDL_BETA}, "search", 25, S50, 101)
    drop_slot("b_EDL_s101", "baseline milestone verified")
    for rn in ("b_T4_s101", "b_EDL_s101"):
        for tag in ("s20", "s50"):
            dev(rn, tag)


def confirm_cand(cid, c, beta, label):
    """50M seeds 100/101 for candidate + its SHUF control; returns decision record."""
    rn0, rn1 = f"s_{cid}_s100", f"s_{cid}_s101"
    train(rn0, cand_cond(c, beta), "search", 25, S50, 100)
    drop_slot(rn0, "50M milestone verified")
    train(rn1, cand_cond(c, beta), "search", 25, S50, 101)
    drop_slot(rn1, "50M milestone verified")
    for rn, tags in ((rn0, ("s20", "s50")), (rn1, ("s20", "s50"))):
        for t in tags:
            dev(rn, t)
    t4 = {(s, t): base_summ(f"b_T4_s{s}", t) for s in (100, 101) for t in ("s20", "s50")}
    ed = {(s, t): base_summ(f"b_EDL_s{s}", t) for s in (100, 101) for t in ("s20", "s50")}
    S = {(s, t): summ(f"s_{cid}_s{s}", t, t4[(s, t)], ed[(s, t)]) for s in (100, 101) for t in ("s20", "s50")}
    mean_dce_edl = (S[(100, "s50")]["dce_edl"] + S[(101, "s50")]["dce_edl"]) / 2
    i_ok = mean_dce_edl <= 0.005
    sh = {}
    if i_ok:
        for s in (100, 101):
            rn = f"shuf_{cid}_s{s}"
            train(rn, cand_cond(c, beta, shuf=True), "search", 25, S50, s)
            drop_slot(rn, "50M milestone verified")
            for t in ("s20", "s50"):
                dev(rn, t)
            sh[s] = {t: summ(rn, t, t4[(s, t)], ed[(s, t)]) for t in ("s20", "s50")}
    ii = bool(sh) and all(S[(s, "s50")]["fsi_pseudo"] > t4[(s, "s50")]["fsi_pseudo"] and
                          S[(s, "s50")]["fsi_pseudo"] > sh[s]["s50"]["fsi_pseudo"] for s in (100, 101))
    iii = all(S[(s, "s50")]["fsi_e5"] >= t4[(s, "s50")]["fsi_e5"] for s in (100, 101))
    iv = all(S[(s, t)]["fsi_pseudo"] > t4[(s, t)]["fsi_pseudo"] for s in (100, 101) for t in ("s20", "s50"))
    v = all(not pathology(S[(s, "s50")]) for s in (100, 101))
    rec = {"candidate": cid, "cand": c, "beta": beta, "label": label, "mean_dce_edl_s50": mean_dce_edl,
           "criteria": {"i_ce_comparable_edl": i_ok, "ii_fsi_pseudo_gt_t4_and_shuf": ii, "iii_fsi_e5_ge_t4": iii,
                        "iv_persistence_s20_s50": iv, "v_no_pathology": v},
           "advance": bool(i_ok and ii and iii and iv and v),
           "summ": {f"{s}_{t}": S[(s, t)] for (s, t) in S}, "shuf": {str(s): sh[s] for s in sh},
           "grad": {s: grad_summary(f"s_{cid}_s{s}", 100, 793) for s in (100, 101)},
           "t4": {f"{s}_{t}": t4[(s, t)] for (s, t) in t4}, "edl": {f"{s}_{t}": ed[(s, t)] for (s, t) in ed}}
    ledger({"generation": label, "candidate": cid, "config": c, "beta": beta, "tokens": "50M", "seeds": [100, 101],
            "ce_s50": [S[(s, "s50")]["ce"] for s in (100, 101)], "mean_dce_edl_s50": mean_dce_edl,
            "fsi_pseudo_s50": [S[(s, "s50")]["fsi_pseudo"] for s in (100, 101)],
            "fsi_pseudo_t4": [t4[(s, "s50")]["fsi_pseudo"] for s in (100, 101)],
            "fsi_pseudo_shuf": [sh[s]["s50"]["fsi_pseudo"] for s in sh], "mi_pseudo_s50":
            [S[(s, "s50")]["mi_pseudo"] for s in (100, 101)], "grad_norms": rec["grad"],
            "routing_entropy": [S[(s, "s50")]["routing_entropy"] for s in (100, 101)],
            "criteria": rec["criteria"], "decision": "advance" if rec["advance"] else "reject"})
    return rec


def stage2():
    d1 = decide1()
    out = SD / "stage2_decision.json"
    if out.exists():
        return json.loads(out.read_text())
    baselines101()
    recs = {}
    for cid in d1["advance"]:
        c = d1["rows"][cid]["cand"]
        recs[cid] = confirm_cand(cid, c, d1["rows"][cid]["beta"], "stage2")
    rec = {"time": guard.now(), "candidates": recs, "advanced": [k for k, v in recs.items() if v["advance"]]}
    guard.atomic_write_json(out, rec, mode=0o400)
    log_event(f"stage2 decision: advanced {rec['advanced']}")
    return rec


def gen(n):
    from csr.sem_paths import prereg_sha  # noqa: F401
    doc = guard.ROOT / f"docs/SEMANTIC_EDL_V1_GEN{n}.md"
    spec_p = SD / f"gen{n}.json"
    if not (doc.exists() and (doc.parent / (doc.name + ".sha256")).exists() and spec_p.exists()):
        raise SystemExit("generation must be preregistered (doc + sha256 + spec) before running")
    spec = json.loads(spec_p.read_text())
    if spec["doc_sha256"] != guard.sha256_file(doc):
        raise guard.GuardError("generation doc hash mismatch")
    out = SD / f"gen{n}_decision.json"
    if out.exists():
        return json.loads(out.read_text())
    from sem_configs import sem_cond
    cal = calib()
    recs = {}
    for c in spec["candidates"]:
        key = f"{c['placement']}_{c['assign']}"
        rho = cal["variants"].get(key, {}).get("rho_beta0")
        if rho is None:
            vp = SD / f"pilot_{key}.json"
            if not vp.exists():
                rn = f"p_{key}_s100"
                train(rn, sem_cond(TABLES[c["assign"]][0], BETA0, PLACE[c["placement"]]), "pilot", 10, 160, 100)
                drop_slot(rn, "pilot slot")
                guard.atomic_write_json(vp, {"rho_beta0": ratio_median(diag_of(rn), 100, 160), "run": run_dirs()[rn]},
                                        mode=0o400)
            rho = json.loads(vp.read_text())["rho_beta0"]
        beta = float(f"{BETA0 * (c['target'] / 0.02) * cal['rho_edl_100_160'] / rho:.2g}")
        if not budget_ok(0.6, c["id"]):
            break
        recs[c["id"]] = confirm_cand(c["id"], c, beta, f"gen{n}")
    rec = {"time": guard.now(), "candidates": recs, "advanced": [k for k, v in recs.items() if v["advance"]]}
    guard.atomic_write_json(out, rec, mode=0o400)
    log_event(f"gen{n} decision: advanced {rec['advanced']}")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stage0", "calib", "stage1", "decide1", "stage2", "gen"])
    ap.add_argument("--n", type=int)
    a = ap.parse_args()
    guard.check_root()
    from csr.sem_paths import check_prereg
    check_prereg()
    SD.mkdir(parents=True, exist_ok=True)
    if a.cmd == "stage0":
        stage0()
    elif a.cmd == "calib":
        print(json.dumps(calib(), indent=1))
    elif a.cmd == "stage1":
        stage1()
    elif a.cmd == "decide1":
        print(json.dumps(decide1()["advance"]))
    elif a.cmd == "stage2":
        print(json.dumps(stage2()["advanced"]))
    elif a.cmd == "gen":
        print(json.dumps(gen(a.n)["advanced"]))


if __name__ == "__main__":
    main()
