#!/usr/bin/env python
"""csr-latent-forecast-router-distill-v1 final analysis (CPU, read-only on run artifacts). Applies the frozen
criteria of protocol 13 and writes final_analysis.json, specialization_trajectory.json and
counterfactual_utility_analysis.json under artifacts/latent-forecast-router-distill-v1/."""
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from csr import guard  # noqa: E402
from csr.lfr_lib import ART, RUN_PREFIX  # noqa: E402

CONDS = ["TOKEN-4", "EDL", "OUTPUT-DISTILL", "CF-FORECAST-STUDENT", "EDL+CF", "FORECAST-SIMILARITY",
         "SHUFFLED-FORECAST", "MEMORY-ONLY", "TOKEN-COMPUTE"]
SHORT = {"TOKEN-4": "A", "EDL": "B", "FORECAST-SIMILARITY": "C", "OUTPUT-DISTILL": "D", "CF-FORECAST-STUDENT": "E",
         "SHUFFLED-FORECAST": "F", "EDL+CF": "G", "MEMORY-ONLY": "M", "TOKEN-COMPUTE": "TC"}


def rd_of(name):
    c = sorted(glob.glob(str(ROOT / f"runs/*_{RUN_PREFIX}{name}")))
    return Path(c[-1]) if c else None


def load_final():
    out = {}
    for c in CONDS:
        for s in (0, 1, 2):
            rd = rd_of(f"final_{c}_s{s}")
            if rd is None or not (rd / "final_eval.json").exists():
                continue
            ev = json.loads((rd / "final_eval.json").read_text())
            summ = json.loads((rd / "summary.json").read_text())
            out[(c, s)] = {"rd": rd, "ev": ev, "summ": summ}
    return out


def paired_boot(F, a, b, seeds, n_boot=2000, seed=0):
    """Token-weighted paired per-document difference (a - b), averaged over seeds; document bootstrap."""
    docs = None
    for s in seeds:
        ka, kb = set(F[(a, s)]["ev"]["val_doc_ce"]), set(F[(b, s)]["ev"]["val_doc_ce"])
        docs = (ka & kb) if docs is None else docs & ka & kb
    docs = sorted(docs)
    diff = np.zeros(len(docs))
    n = np.zeros(len(docs))
    for s in seeds:
        da, db = F[(a, s)]["ev"]["val_doc_ce"], F[(b, s)]["ev"]["val_doc_ce"]
        diff += np.asarray([da[d][0] - db[d][0] for d in docs]) / len(seeds)
        n += np.asarray([da[d][1] for d in docs]) / len(seeds)
    rng = np.random.default_rng(seed)
    est = diff.sum() / n.sum()
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(docs), len(docs))
        bs.append(diff[i].sum() / n[i].sum())
    return {"mean": float(est), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "n_docs": len(docs)}


def fsi(F, c, s, tag="p100", system="domain"):
    try:
        a = F[(c, s)]["ev"]["audit"][tag][system]
        return a["fsi"], a["fsi_shuf"]
    except KeyError:
        return None, None


def main():
    guard.check_root()
    F = load_final()
    st = json.loads((ART / "driver_state.json").read_text())
    fz = json.loads((ART / "frozen.json").read_text()) if (ART / "frozen.json").exists() else None
    seeds_e = sorted(s for (c, s) in F if c == "CF-FORECAST-STUDENT")
    prim = [s for s in seeds_e if ("EDL", s) in F]
    res = {"time": guard.now(), "frozen": fz, "primary_seeds": prim, "val": {}, "comparisons": {}}
    for (c, s), r in sorted(F.items()):
        res["val"].setdefault(c, {})[str(s)] = {"ce": r["ev"]["val"]["ce"], "ce_dom": r["ev"]["val"]["ce_dom"],
                                                 "tokens_per_s_train": r["summ"]["tokens_per_s"],
                                                 "leg_seconds": r["summ"]["leg_seconds"], "seen": r["summ"]["seen"]}
    ce = lambda c, s: F[(c, s)]["ev"]["val"]["ce"]
    for a, b in [("CF-FORECAST-STUDENT", "EDL"), ("CF-FORECAST-STUDENT", "TOKEN-4"),
                 ("CF-FORECAST-STUDENT", "OUTPUT-DISTILL"), ("EDL+CF", "EDL"), ("EDL", "TOKEN-4"),
                 ("FORECAST-SIMILARITY", "CF-FORECAST-STUDENT"), ("SHUFFLED-FORECAST", "CF-FORECAST-STUDENT"),
                 ("MEMORY-ONLY", "CF-FORECAST-STUDENT"), ("OUTPUT-DISTILL", "CF-FORECAST-STUDENT"),
                 ("TOKEN-COMPUTE", "CF-FORECAST-STUDENT"), ("OUTPUT-DISTILL", "TOKEN-4"), ("EDL+CF", "TOKEN-4")]:
        ss = [s for s in (0, 1, 2) if (a, s) in F and (b, s) in F]
        if not ss:
            continue
        d = [ce(a, s) - ce(b, s) for s in ss]
        res["comparisons"][f"{SHORT[a]}-{SHORT[b]}"] = {"seeds": ss, "per_seed": d, "mean": float(np.mean(d)),
                                                         "all_negative": all(x < 0 for x in d),
                                                         "sign_agree": len({x < 0 for x in d}) == 1,
                                                         "bootstrap": paired_boot(F, a, b, ss)}
    cmp_ = res["comparisons"]
    # ---------------- specialisation trajectory
    traj = {}
    for (c, s), r in sorted(F.items()):
        for tag, au in r["ev"]["audit"].items():
            for system in ("domain", "cluster"):
                if system not in au:
                    continue
                x = au[system]
                traj.setdefault(c, {}).setdefault(str(s), {}).setdefault(tag, {})[system] = {
                    "fsi": x["fsi"], "fsi_raw": x["functional"]["fsi_raw"], "fsi_shuf": x["fsi_shuf"],
                    "mi_mean": x["routing"]["mi_mean"], "mi_null_mean": x["routing"]["mi_null_mean"],
                    "topic_entropy_mean": x["routing"]["topic_entropy_mean"], "topic_pred_acc": x["topic_pred_acc"],
                    "topic_pred_chance": x["topic_pred_chance"],
                    "within_expert_similarity": x.get("within_expert_similarity"),
                    "load_min": x["routing"]["load_min"], "load_max": x["routing"]["load_max"],
                    "n_below_0.1_uniform": x["routing"]["n_below_0.1_uniform"], "group_sizes":
                        x["functional"]["group_sizes"]}
            traj[c][str(s)][tag]["routing_entropy"] = au["routing_entropy"]
    # cross-seed stability (seed 0 vs 1, 100% checkpoint)
    from csr.lfr_audit import cross_seed_stability
    stab = {}
    for c in CONDS:
        if (c, 0) in F and (c, 1) in F:
            for system in ("domain", "cluster"):
                try:
                    pa = F[(c, 0)]["ev"]["audit"]["p100"][system]["routing"]["profiles"]
                    pb = F[(c, 1)]["ev"]["audit"]["p100"][system]["routing"]["profiles"]
                    stab.setdefault(c, {})[system] = cross_seed_stability(pa, pb)
                except KeyError:
                    pass
    guard.atomic_write_json(ART / "specialization_trajectory.json", {"trajectory": traj, "cross_seed": stab},
                            overwrite=True)
    # ---------------- criteria
    E, A, B, D, G = "CF-FORECAST-STUDENT", "TOKEN-4", "EDL", "OUTPUT-DISTILL", "EDL+CF"
    crit = {}
    eb = cmp_.get("E-B")
    crit["1_E_lt_B_every_seed"] = bool(eb and eb["all_negative"] and len(eb["seeds"]) >= 2)
    crit["2_mean_E_minus_B_le_-0.005"] = bool(eb and eb["mean"] <= -0.005)
    crit["3_paired_ci_excludes_0"] = bool(eb and eb["bootstrap"]["ci95"][1] < 0)
    ea, ed = cmp_.get("E-A"), cmp_.get("E-D")
    crit["4_E_beats_A"] = bool(ea and ea["mean"] < 0 and ea["all_negative"])
    crit["5_E_beats_D"] = bool(ed and ed["mean"] < 0 and ed["all_negative"])
    c6, c7, c8 = True, True, True
    for system in ("domain", "cluster"):
        fe = [fsi(F, E, s, "p100", system)[0] for s in prim]
        fa = [fsi(F, A, s, "p100", system)[0] for s in prim]
        fb = [fsi(F, B, s, "p100", system)[0] for s in prim]
        if None in fe or None in fa:
            c6 = False
        else:
            c6 &= all(x > y for x, y in zip(fe, fa))
        if None in fe or None in fb:
            c7 = False
        else:
            c7 &= float(np.mean(fe)) >= float(np.mean(fb))
    for s in prim:
        f100, sh = fsi(F, E, s, "p100", "domain")
        f40, _ = fsi(F, E, s, "p040", "domain")
        fa, _ = fsi(F, A, s, "p100", "domain")
        if None in (f100, f40, fa, sh):
            c8 = False
        else:
            c8 &= f100 > fa and f100 >= 0.5 * f40 and f100 - sh > 0
    crit["6_FSI_E_gt_A_every_seed_both_systems"] = bool(c6 and prim)
    crit["7_meanFSI_E_ge_B_both_systems"] = bool(c7 and prim)
    crit["8_specialisation_survives"] = bool(c8 and prim)
    tests = sorted(glob.glob(str(ART / "cpu_tests_*.json")))
    t_ok = False
    if tests:
        tr = json.loads(Path(tests[-1]).read_text())["results"]
        t_ok = all(tr[k]["pass"] for k in tr if k.startswith(("T11", "T12", "T13")))
    crit["9_no_draft_forecast_label_at_inference"] = t_ok
    bench = json.loads((ART / "bench.json").read_text()) if (ART / "bench.json").exists() else {}
    be = bench.get(E, {})
    crit["10_speed_and_active_params_within_3pct"] = bool(be and be.get("speed_ratio_vs_token4", 0) >= 0.97 and
                                                          be.get("active_ratio_vs_token4", 9) <= 1.03)
    c11 = True
    for s in prim:
        try:
            ld = np.asarray(F[(E, s)]["ev"]["audit"]["p100"]["load"])
            c11 &= bool(ld.min() >= 0.1 / 32 and ld.max() <= 4 / 32)
        except KeyError:
            c11 = False
    crit["11_no_collapse"] = bool(c11 and prim)
    res["criteria"] = crit
    # ---------------- terminal status (protocol 13)
    gb = cmp_.get("G-B")
    hyb = bool(gb and gb["all_negative"] and len(gb["seeds"]) >= 2 and gb["mean"] <= -0.005 and
               gb["bootstrap"]["ci95"][1] < 0)
    spec_ok = crit["6_FSI_E_gt_A_every_seed_both_systems"] and crit["7_meanFSI_E_ge_B_both_systems"] and \
        crit["8_specialisation_survives"]
    acc_ok = all(crit[k] for k in ("1_E_lt_B_every_seed", "2_mean_E_minus_B_le_-0.005", "3_paired_ci_excludes_0",
                                    "4_E_beats_A", "5_E_beats_D"))
    if eb is None or len(eb["seeds"]) < 2:
        status = "FAILED DUE TO IMPLEMENTATION OR RESOURCE ERROR"
    elif all(crit.values()):
        status = "CONFIRMED STANDALONE IMPROVEMENT OVER EDL"
    elif not eb["sign_agree"]:
        status = "UNRESOLVED"
    elif not (crit["1_E_lt_B_every_seed"] and crit["2_mean_E_minus_B_le_-0.005"] and crit["3_paired_ci_excludes_0"]) \
            and hyb:
        status = "HYBRID-ONLY IMPROVEMENT"
    elif spec_ok and not (crit["1_E_lt_B_every_seed"] and crit["2_mean_E_minus_B_le_-0.005"]):
        status = "USEFUL SPECIALIZATION WITHOUT ACCURACY IMPROVEMENT"
    elif acc_ok and not spec_ok:
        status = "ACCURACY IMPROVEMENT WITHOUT SPECIALIZATION"
    else:
        status = "NEGATIVE"
    res["terminal_status"] = status
    res["hybrid_satisfies_1_to_3"] = hyb
    read = {}
    for k, nm in (("D-E", "ordinary knowledge distillation"), ("F-E", "forecast semantics not responsible"),
                  ("C-E", "utility supervision adds nothing beyond forecast similarity"),
                  ("M-E", "router architecture rather than teacher")):
        if k in cmp_:
            read[k] = {"mean": cmp_[k]["mean"], "matches_E_within_0.003": abs(cmp_[k]["mean"]) < 0.003, "reading": nm}
    res["mechanism_readings"] = read
    res["seed2_trigger"] = st.get("seed2_trigger")
    res["tc_budget"] = st.get("tc_budget")
    res["bench"] = bench
    # ---------------- compute views
    led = []
    p = ART / "compute_ledger.tsv"
    for line in p.read_text().splitlines():
        f = line.split("\t")
        if len(f) >= 7:
            try:
                led.append({"seconds": float(f[2]), "run": f[3], "name": f[4], "category": f[5], "status": f[6]})
            except ValueError:
                pass
    cat = {}
    for r in led:
        cat[r["category"]] = cat.get(r["category"], 0.0) + r["seconds"]
    res["compute"] = {"gpu_seconds_by_category": cat, "total_gpu_hours": sum(cat.values()) / 3600,
                      "draft_training_gpu_s_fspv2": 877.1}
    per = {}
    for (c, s), r in F.items():
        per.setdefault(c, []).append(sum(json.loads(q.read_text())["leg_seconds"] for q in
                                         sorted(r["rd"].glob("leg_*.json"))))
    res["compute"]["final_run_gpu_seconds_mean"] = {c: float(np.mean(v)) for c, v in per.items()}
    if E in per and A in per:
        e_s, a_s = np.mean(per[E]), np.mean(per[A])
        oh = 877.1 + cat.get("forecast_cache", 0.0)
        res["compute"]["E_over_A_training_time"] = float(e_s / a_s)
        res["compute"]["E_total_over_A_unamortized"] = float((e_s + oh) / a_s)
        res["compute"]["E_total_over_A_amortized_over_final_E_seeds"] = float((e_s + oh / len(per[E])) / a_s)
        res["compute"]["E_total_over_A_amortized_over_10_trainings"] = float((e_s + oh / 10) / a_s)
    guard.atomic_write_json(ART / "final_analysis.json", res, overwrite=True)
    # ---------------- counterfactual utility analysis
    cu = {"validation": None, "in_run_audits": {}, "diagnostics": {}}
    if (ART / "cf_validation.json").exists():
        v = json.loads((ART / "cf_validation.json").read_text())
        cu["validation"] = {k: v[k] for k in v if k != "pairs"}
    for rd in sorted(glob.glob(str(ROOT / f"runs/*_{RUN_PREFIX}*"))):
        sp = Path(rd) / "summary.json"
        if not sp.exists():
            continue
        s = json.loads(sp.read_text())
        au = s.get("cfaudit") or []
        if au:
            ok = [x for x in au if x.get("route_ok", True)]
            cu["in_run_audits"][Path(rd).name] = {
                "n": len(au), "n_route_ok": len(ok),
                "rho_fo_exact_mean": float(np.mean([x["rho_fo_exact"] for x in au])),
                "rho_teacher_exact_mean": float(np.mean([x["rho_teacher_exact"] for x in au if "rho_teacher_exact" in x]))
                if any("rho_teacher_exact" in x for x in au) else None,
                "top4_fo_exact_mean": float(np.mean([x["top4_fo_exact"] for x in au]))}
        d = s.get("diag") or {}
        if "p1end" in d:
            cu["diagnostics"][Path(rd).name] = {k: d["p1end"].get(k) for k in (
                "ce_teacher", "ce_teacher_shuffled", "ce_student", "probe_z", "probe_raw", "probe_ratio",
                "rho_teacher_exact_mean", "rho_fo_exact_mean", "top4_overlap_mean")}
            if "p2end" in d:
                cu["diagnostics"][Path(rd).name]["p2end_top4_overlap"] = d["p2end"].get("top4_overlap_mean")
                cu["diagnostics"][Path(rd).name]["p2end_ce_teacher"] = d["p2end"].get("ce_teacher")
                cu["diagnostics"][Path(rd).name]["p2end_ce_student"] = d["p2end"].get("ce_student")
            if "end" in d:
                cu["diagnostics"][Path(rd).name]["end_ce_student"] = d["end"].get("ce_student")
    guard.atomic_write_json(ART / "counterfactual_utility_analysis.json", cu, overwrite=True)
    print(json.dumps({"terminal_status": status, "criteria": crit,
                      "comparisons": {k: (v["per_seed"], v["mean"], v["bootstrap"]["ci95"]) for k, v in cmp_.items()}},
                     indent=1))


if __name__ == "__main__":
    main()
