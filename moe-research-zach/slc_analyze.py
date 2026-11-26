#!/usr/bin/env python
"""semantic-lift-245m-confirm-v1 frozen analysis (CPU; prereg sections 6-8). Reads the 40 evaluations, training
summaries and logs; computes per-seed differences, hierarchical (seed x document) bootstrap CIs, the primary claim
S1-S5, the CE contrasts, routing summaries, cross-seed profile consistency, trajectory 50M -> 245M and the outcome
classification A-G. Writes artifacts/semantic-lift-245m-confirm-v1/results.json and tables.md (derived; regenerated)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402

CONDS = ("E", "F", "C6", "SHUF", "EDL", "T4")
PAIRS = (("E", "F"), ("E", "C6"), ("E", "EDL"), ("E", "T4"), ("C6", "SHUF"), ("F", "SHUF"), ("F", "C6"),
         ("EDL", "T4"), ("C6", "T4"))
PAIRS50 = (("E", "F"), ("E", "C6"), ("C6", "SHUF"), ("F", "SHUF"))
N_REP = 10000
CE_B = 2000


def main():
    import numpy as np
    from csr.slc_lib import hier_boot
    from csr.slc_paths import ART, LIFT, REUSED, SEEDS, check_prereg
    from csr.slk_lib import match_profiles
    check_prereg()
    ev, npz = {}, {}
    for p in sorted((ART / "evals").glob("*.json")):
        d = json.loads(p.read_text())
        if d.get("test_mode"):
            continue
        k = (d["key"].rsplit("_s", 1)[0], int(d["key"].rsplit("_s", 1)[1]), d["tag"])
        ev[k] = d
        npz[k] = dict(np.load(p.with_suffix(".npz")))
    have = lambda c, tag: all((c, s, tag) in ev for s in SEEDS)  # noqa: E731
    res = {"time": guard.now(), "n_evals": len(ev), "seeds": list(SEEDS)}

    def per_seed_contrast(a, b, tag, kind):
        """kind 'ord' | 'cond'. Returns per-seed points, doc CIs, hierarchical summary, replicate array."""
        km, kb = ("mean_interaction", "boot_mean") if kind == "ord" else ("cond_mean_interaction", "cboot_mean")
        pts, B, dci = [], [], []
        for s in SEEDS:
            fa, fb = ev[(a, s, tag)]["function"], ev[(b, s, tag)]["function"]
            pa, pb = fa[km], fb[km]
            ba, bb = npz[(a, s, tag)]["f_" + kb], npz[(b, s, tag)]["f_" + kb]
            if pa is None or pb is None:
                return None
            pts.append(pa - pb)
            B.append(ba - bb)
            dci.append([float(np.nanquantile(ba - bb, 0.025)), float(np.nanquantile(ba - bb, 0.975))])
        h, reps = hier_boot(np.asarray(B), N_REP, point=pts)
        return {"per_seed": pts, "n_pos": int(sum(x > 0 for x in pts)), "mean": float(np.mean(pts)),
                "sd": float(np.std(pts, ddof=1)), "doc_ci95_per_seed": dci, "hier": h}, np.asarray(B)

    func, boots = {}, {}
    for tag, pairs in (("p100", PAIRS), ("s50", PAIRS50)):
        for a, b in pairs:
            if not (have(a, tag) and have(b, tag)):
                continue
            for kind in ("ord", "cond"):
                r = per_seed_contrast(a, b, tag, kind)
                if r is None:
                    continue
                func[f"{a}-{b}|{tag}|{kind}"], boots[(a, b, tag, kind)] = r
    # difference-in-differences and trajectory
    for tag in ("p100", "s50"):
        for kind in ("ord", "cond"):
            if (("E", "F", tag, kind) in boots) and (("C6", "SHUF", tag, kind) in boots):
                X = boots[("E", "F", tag, kind)] - boots[("C6", "SHUF", tag, kind)]
                pts = [x - y for x, y in zip(func[f"E-F|{tag}|{kind}"]["per_seed"], func[f"C6-SHUF|{tag}|{kind}"]["per_seed"])]
                h, _ = hier_boot(X, N_REP, point=pts)
                func[f"DiD(E-F)-(C6-SHUF)|{tag}|{kind}"] = {"per_seed": pts, "n_pos": int(sum(x > 0 for x in pts)),
                                                           "mean": float(np.mean(pts)), "hier": h}
    for kind in ("ord", "cond"):
        if ("E", "F", "p100", kind) in boots and ("E", "F", "s50", kind) in boots:
            X = boots[("E", "F", "p100", kind)] - boots[("E", "F", "s50", kind)]
            pts = [x - y for x, y in zip(func[f"E-F|p100|{kind}"]["per_seed"], func[f"E-F|s50|{kind}"]["per_seed"])]
            h, _ = hier_boot(X, N_REP, point=pts)
            func[f"traj D245-D50 E-F|{kind}"] = {"per_seed": pts, "mean": float(np.mean(pts)), "hier": h}
    res["functional_contrasts"] = func
    # per-condition functional summaries
    cond_tab = {}
    for (c, s, tag), d in sorted(ev.items()):
        f = d["function"]
        cond_tab.setdefault(f"{c}|{tag}", []).append(
            {"seed": s, "I_ord": f["mean_interaction"], "I_ord_doc_ci": f["mean_interaction_ci95_doc"],
             "I_cond": f["cond_mean_interaction"], "I_cond_doc_ci": f["cond_mean_interaction_ci95_doc"],
             "cond_n_adequate": f["cond_n_adequate"], "n_specialized": f["n_specialized"],
             "cond_n_specialized": f["cond_n_specialized"], "frac_positive": f["frac_positive"],
             "by_layer": f["by_layer"], "cond_by_layer": f["cond_by_layer"],
             "cond_support_median": f["cond_support_median"], "mean_generic_dce": f["mean_generic_dce"],
             "mean_own_dce": f["mean_own_dce"], "n_defined": f["n_defined"]})
    res["functional_by_condition"] = cond_tab
    # ---------------------------------------------------------------- CE (hierarchical seeds x documents)
    ce = {}
    gd0 = None
    for k in ev:
        g = npz[k]["doc_gdoc"]
        if gd0 is None:
            gd0 = g
        if not np.array_equal(g, gd0):
            raise SystemExit("VAL document sets differ between evaluations")
    rng = np.random.default_rng(20261119)
    Wce = rng.multinomial(len(gd0), np.ones(len(gd0)) / len(gd0), size=CE_B).astype(np.float64)

    def ce_boot(k):
        return (Wce @ npz[k]["doc_sum"]) / (Wce @ npz[k]["doc_n"])
    ce_boot_cache = {}
    for tag in ("p100", "s50"):
        for c in CONDS:
            if not have(c, tag):
                continue
            v = [ev[(c, s, tag)]["val"]["ce"] for s in SEEDS]
            dom = np.asarray([ev[(c, s, tag)]["val"]["ce_dom"] for s in SEEDS])
            ce[f"{c}|{tag}"] = {"per_seed": v, "mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)),
                                "by_domain_mean": dom.mean(0).tolist(), "by_domain_sd": dom.std(0, ddof=1).tolist()}
            for s in SEEDS:
                ce_boot_cache[(c, s, tag)] = ce_boot((c, s, tag))
        for a, b in (PAIRS if tag == "p100" else PAIRS50):
            if not (have(a, tag) and have(b, tag)):
                continue
            pts = [ev[(a, s, tag)]["val"]["ce"] - ev[(b, s, tag)]["val"]["ce"] for s in SEEDS]
            B = np.asarray([ce_boot_cache[(a, s, tag)] - ce_boot_cache[(b, s, tag)] for s in SEEDS])
            h, _ = hier_boot(B, N_REP, point=pts)
            dd = [[float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))] for x in B]
            dom = [(np.asarray(ev[(a, s, tag)]["val"]["ce_dom"]) - np.asarray(ev[(b, s, tag)]["val"]["ce_dom"])).tolist()
                   for s in SEEDS]
            ce[f"{a}-{b}|{tag}"] = {"per_seed": pts, "n_neg": int(sum(x < 0 for x in pts)), "mean": float(np.mean(pts)),
                                    "sd": float(np.std(pts, ddof=1)), "doc_ci95_per_seed": dd, "hier": h,
                                    "by_domain_mean": np.mean(dom, 0).tolist()}
    res["ce"] = ce
    # ---------------------------------------------------------------- routing
    rout = {}
    for tag in ("p100", "s50"):
        for c in CONDS:
            if not have(c, tag):
                continue
            g = lambda f: [f(ev[(c, s, tag)]) for s in SEEDS]  # noqa: E731
            row = {"domain_nmi": g(lambda d: d["routing_domain"]["nmi_mean"]),
                   "domain_nmi_null_doclabel": g(lambda d: d["routing_domain"]["nmi_null_doclabel_mean"]),
                   "domain_mi": g(lambda d: d["routing_domain"]["mi_mean"]),
                   "e5_nmi": g(lambda d: d["routing_e5"]["nmi_mean"]),
                   "topk_overlap": g(lambda d: d["domain_topk_overlap"]["mean_pairwise_overlap"]),
                   "overlap_by_layer_mean": np.mean(g(lambda d: d["domain_topk_overlap"]["by_layer"]), 0).tolist(),
                   "routing_specialized_cells": g(lambda d: d["routing_domain"]["n_routing_specialized_cells"]),
                   "domain_mean_max_lift": g(lambda d: d["routing_domain"]["mean_max_lift"]),
                   "e5_mean_max_lift": g(lambda d: d["routing_e5"]["mean_max_lift"]),
                   "load_min_x": g(lambda d: d["load"]["full_val_min_x_uniform"]),
                   "load_max_x": g(lambda d: d["load"]["full_val_max_x_uniform"]),
                   "n_underused": g(lambda d: d["load"]["n_underused_full_val"]),
                   "n_unused": g(lambda d: d["load"]["n_unused_full_val"]),
                   "entropy": g(lambda d: d["load"]["routing_entropy_full_val"]),
                   "pathology": g(lambda d: d["pathology"])}
            # cross-seed consistency of expert subject profiles (Hungarian per layer)
            pr = [npz[(c, s, tag)]["domain_lift_profiles"] for s in SEEDS]
            cs_ = []
            for i in range(len(SEEDS)):
                for j in range(i + 1, len(SEEDS)):
                    cs_.append(float(np.mean([match_profiles(pr[i][li], pr[j][li])[0] for li in range(pr[i].shape[0])])))
            row["cross_seed_profile_consistency"] = {"pairs": cs_, "mean": float(np.mean(cs_))}
            rout[f"{c}|{tag}"] = row
        for a, b in (PAIRS if tag == "p100" else PAIRS50):
            if f"{a}|{tag}" in rout and f"{b}|{tag}" in rout:
                dn = [x - y for x, y in zip(rout[f"{a}|{tag}"]["domain_nmi"], rout[f"{b}|{tag}"]["domain_nmi"])]
                do = [x - y for x, y in zip(rout[f"{a}|{tag}"]["topk_overlap"], rout[f"{b}|{tag}"]["topk_overlap"])]
                rout[f"{a}-{b}|{tag}"] = {"domain_nmi_diff": dn, "nmi_n_pos": int(sum(x > 0 for x in dn)),
                                          "nmi_mean": float(np.mean(dn)), "overlap_diff": do,
                                          "overlap_mean": float(np.mean(do))}
    res["routing"] = rout
    # ---------------------------------------------------------------- training: multipliers, throughput, compute
    tr = {}
    for p in sorted((ART / "train_summaries").glob("*.json")):
        d = json.loads(p.read_text())
        tr[d["run"]] = {"tokens_per_s": d["status"]["tokens_per_s"], "legs": d["status"].get("legs"),
                        "train_step_seconds": d["status"].get("train_step_seconds"), **d["activity"]}
    rm = json.loads((ART / "reuse_manifest.json").read_text())
    res["training"] = {"new_runs": tr, "baseline_tokens_per_s": {k: v["tokens_per_s"] for k, v in rm["runs"].items()}}
    by = {}
    for line in (ART / "RUN_LOG.jsonl").read_text().splitlines():
        r = json.loads(line)
        by[r.get("phase", "?")] = by.get(r.get("phase", "?"), 0.0) + float(r.get("gpu_seconds") or 0) / 3600
    res["compute_gpu_h_by_phase"] = by
    res["compute_gpu_h_total"] = sum(by.values())
    # ---------------------------------------------------------------- criteria and outcomes
    ef = func.get("E-F|p100|ord")
    efc = func.get("E-F|p100|cond")
    S1 = bool(ef and ef["mean"] > 0)
    S2 = bool(ef and ef["n_pos"] >= 3)
    S3 = bool(ef and ef["hier"]["ci95"][0] > 0)
    S4 = bool(efc and efc["mean"] > 0 and efc["n_pos"] >= 3)
    path_any = any(any(rout[f"{c}|p100"]["pathology"]) for c in ("E", "F"))
    inv_ok = all(tr.get(f"{c}_s{s}", {}).get("invariants_ok") and tr.get(f"{c}_s{s}", {}).get("finite")
                 for c in ("E", "F") for s in SEEDS)
    S5 = bool(not path_any and inv_ok)
    ce_e_c6 = ce["E-C6|p100"]["mean"]
    ce_e_edl = ce["E-EDL|p100"]["mean"]
    ce_worse = bool(ce_e_c6 > 0.005 or ce_e_edl > 0.005)
    ORD, COND, PATH = S1 and S2 and S3, S4, S5
    ec6, fsh = func.get("E-C6|p100|ord"), func.get("F-SHUF|p100|ord")
    outcomes = []
    if ORD and COND and PATH and not ce_worse:
        outcomes.append("E")
    if ORD and COND and PATH and ce_worse:
        outcomes.append("D")
    if ORD and not COND:
        outcomes.append("C")
    if (not ORD and ec6 and fsh and ec6["mean"] > 0 and ec6["n_pos"] >= 3 and fsh["mean"] > 0 and fsh["n_pos"] >= 3):
        outcomes.append("B")
    nm = rout.get("E-F|p100")
    if not ORD and nm and nm["nmi_mean"] > 0 and nm["nmi_n_pos"] >= 3:
        outcomes.append("A")
    traj = []
    ef50 = func.get("E-F|s50|ord")
    if ef50:
        g50 = ef50["mean"] > 0 and ef50["n_pos"] >= 3
        g245 = bool(ef and ef["mean"] > 0 and ef["n_pos"] >= 3)
        if g50 and not g245:
            traj.append("F")
        if ORD and ef["mean"] > ef50["mean"]:
            traj.append("G")
    res["criteria"] = {"S1_Dfunc_gt0": S1, "S2_ge3of4_pos": S2, "S3_hier_ci_excludes0": S3, "S4_conditional_same_dir": S4,
                       "S5_no_pathology_invariants": S5, "pathology_any_E_F_p100": path_any, "invariants_and_finite": inv_ok,
                       "D_func": ef["mean"] if ef else None, "D_func_hier_ci95": ef["hier"]["ci95"] if ef else None,
                       "D_func_per_seed": ef["per_seed"] if ef else None,
                       "D_func_cond": efc["mean"] if efc else None, "D_func_cond_hier_ci95": efc["hier"]["ci95"] if efc else None,
                       "D_func_cond_per_seed": efc["per_seed"] if efc else None,
                       "CE_mean_E_minus_C6": ce_e_c6, "CE_mean_E_minus_EDL": ce_e_edl, "CE_worsened": ce_worse,
                       "supports_semantic_functional_specialization": bool(ORD and COND and PATH)}
    res["outcomes"] = {"matching": outcomes, "primary": outcomes[0] if outcomes else "none",
                       "trajectory": traj or ["no trajectory claim (stable or absent)"],
                       "method_successful": bool(outcomes and outcomes[0] == "E")}
    res["lift_config"] = LIFT
    res["reused"] = REUSED
    # measurement identity (descriptive)
    slk = json.loads((guard.ROOT / "artifacts/semantic-edl-long-k/phase2a_analysis.json").read_text())
    stored = slk["phase1"]["metrics"]["C6"]["0"]["ce"]
    mine = ev[("C6", 0, "p100")]["val"]["ce"] if ("C6", 0, "p100") in ev else None
    res["measurement_identity_C6_s0"] = {"stored_slk": stored, "slc": mine,
                                         "abs_diff": abs(stored - mine) if mine is not None else None,
                                         "within_1e-4": bool(mine is not None and abs(stored - mine) <= 1e-4)}
    inv = sorted(ART.glob("protected_inventory_before.verify_*.json"))
    res["integrity"] = json.loads(inv[-1].read_text()) if inv else None
    res["deviations"] = [json.loads(x) for x in (ART / "deviations.jsonl").read_text().splitlines()] \
        if (ART / "deviations.jsonl").exists() else []
    res["disk_free_gib"] = guard.disk_free_bytes() / 2**30
    guard.atomic_write_json(ART / "results.json", res, overwrite=True)
    # ---------------------------------------------------------------- tables
    L = ["# semantic-lift-245m-confirm-v1 — tables (derived from results.json)\n"]
    L.append("## Primary criteria\n")
    L.append("```\n" + json.dumps(res["criteria"], indent=1) + "\n```\n")
    L.append(f"Outcomes: {res['outcomes']}\n")
    L.append("## Functional contrasts (per seed s0..s3; mean; hierarchical 95% CI)\n")
    L.append("| contrast | per seed | n>0 | mean | hier CI |\n|---|---|---|---|---|")
    for k, v in func.items():
        L.append(f"| {k} | {', '.join(f'{x:+.2e}' for x in v['per_seed'])} | {v.get('n_pos', '')} | {v['mean']:+.2e} | "
                 f"[{v['hier']['ci95'][0]:+.2e}, {v['hier']['ci95'][1]:+.2e}] |")
    L.append("\n## Functional by condition (mean over seeds)\n")
    L.append("| cond | tag | I_ord per seed | I_cond per seed | adequate cells | specialised cells |\n|---|---|---|---|---|---|")
    for k, v in cond_tab.items():
        c, tag = k.split("|")
        L.append(f"| {c} | {tag} | {', '.join(f'{x['I_ord']:.2e}' for x in v)} | "
                 f"{', '.join(f'{x['I_cond']:.2e}' if x['I_cond'] is not None else 'NA' for x in v)} | "
                 f"{', '.join(str(x['cond_n_adequate']) for x in v)} | {', '.join(str(x['n_specialized']) for x in v)} |")
    L.append("\n## CE (VAL)\n")
    L.append("| item | per seed | mean | sd | hier CI |\n|---|---|---|---|---|")
    for k, v in ce.items():
        hc = f"[{v['hier']['ci95'][0]:+.4f}, {v['hier']['ci95'][1]:+.4f}]" if "hier" in v else ""
        L.append(f"| {k} | {', '.join(f'{x:.4f}' for x in v['per_seed'])} | {v['mean']:.4f} | {v['sd']:.4f} | {hc} |")
    L.append("\n## Routing (p100 and s50)\n")
    L.append("| cond | domain NMI | top-4 overlap | routing-spec cells | e5 NMI | profile consistency | load min/max x | entropy |\n|---|---|---|---|---|---|---|---|")
    for k, v in rout.items():
        if "domain_nmi" not in v:
            continue
        L.append(f"| {k} | {np.mean(v['domain_nmi']):.4f} | {np.mean(v['topk_overlap']):.4f} | "
                 f"{np.mean(v['routing_specialized_cells']):.1f} | {np.mean(v['e5_nmi']):.4f} | "
                 f"{v['cross_seed_profile_consistency']['mean']:.3f} | {min(v['load_min_x']):.2f}/{max(v['load_max_x']):.2f} | "
                 f"{np.mean(v['entropy']):.3f} |")
    L.append("\n## Training (new runs)\n")
    L.append("| run | tok/s | lm last | abs(m-1) full-strength mean | clip last | stability | invariants | train load after ramp |\n|---|---|---|---|---|---|---|---|")
    for k, v in tr.items():
        L.append(f"| {k} | {v['tokens_per_s']:.0f} | {v['lm_last']:.4f} | {v['m_absdev_wmean_full_mean']:.3f} | "
                 f"{v['clip_frac_last']:.3f} | {v['stability_r_last']} | {v['invariants_ok']} | "
                 f"{v['train_load_min_x_uniform_after_388']}/{v['train_load_max_x_uniform_after_388']} |")
    guard.atomic_write_bytes(ART / "tables.md", "\n".join(L).encode(), overwrite=True)
    print(json.dumps({"criteria": res["criteria"], "outcomes": res["outcomes"]}, indent=1, default=str))


if __name__ == "__main__":
    main()
