#!/usr/bin/env python
"""csr-context-prior-specialization-v2 analysis (CPU, read-only on results): applies the preregistered rules of protocol
section 10 and writes final_analysis.json, specialization_trajectory.json, counterfactual_utility_analysis.json in
artifacts/context-prior-specialization-v2/. Development-NEGATIVE path (no freeze) is handled. No GPU, no data access."""
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from csr import guard  # noqa: E402

FINAL_SEEDS = (0, 1)


def main():
    guard.check_root()
    from csr.cps_lib import ART, RUN_PREFIX, V1_RUN_PREFIX
    from csr.cps_audit import paired_bootstrap
    from csr.lfr_audit import cross_seed_stability
    st = json.loads((ART / "driver_state.json").read_text())

    def rdir(name, v1=False):
        c = sorted(glob.glob(str(ROOT / f"runs/*_{V1_RUN_PREFIX if v1 else RUN_PREFIX}{name}")))
        return Path(c[-1]) if c else None

    def feval(rd):
        from csr import cps_jsonz as jz
        p = ART / "evals" / rd.name / "final_eval.json" if rd else None
        return jz.load_json(p) if p and jz.exists(p) else None
    dev = {"screen": [{k: c.get(k) for k in ("id", "knobs", "reason", "ce", "ce_prior_off", "fsi_domain",
                                             "fsi_cluster", "mi_domain", "score", "h_failure", "q", "stats", "run_name")}
                      for c in st.get("cands", [])],
           "controls_s1": {k: {kk: v.get(kk) for kk in ("ce", "ce_prior_off", "fsi_domain", "fsi_cluster", "mi_domain",
                                                         "stats", "knobs")} for k, v in st.get("dev", {}).items()},
           "advanced": st.get("advanced"), "s2": st.get("s2"), "s2_eligible": st.get("s2_eligible"),
           "refs": {k: v for k, v in st.get("refs", {}).items() if k != "S1" and k != "S2"},
           "t4_s1": {k: v for k, v in (st.get("refs", {}).get("S1", {}).get("100") or {}).items() if k != "lm"},
           "t4_s2": {s: {k: v for k, v in r.items() if k != "lm"} for s, r in st.get("refs", {}).get("S2", {}).items()}}
    out = {"time": guard.now(), "experiment": "csr-context-prior-specialization-v2", "development": dev}
    traj, cfu = {}, {"runs": {}}
    if (ART / "NEGATIVE_DEV.json").exists():
        out["frozen"] = None
        out["negative_dev"] = json.loads((ART / "NEGATIVE_DEV.json").read_text())
        out["terminal_status"] = "NEGATIVE"
        out["status_reason"] = "no candidate met the S2 development criteria; VAL never opened"
    else:
        fz = json.loads((ART / "frozen.json").read_text())
        out["frozen"] = fz
        runs = {"CAND": {s: rdir(f"final_CAND_s{s}") for s in FINAL_SEEDS},
                "TOKEN-4": {0: rdir("final_TOKEN-4_s0", v1=True), 1: rdir("final_TOKEN_s1")},
                "MEMORY": {s: rdir(f"final_MEMORY_s{s}") for s in FINAL_SEEDS},
                "TOKEN-COMPUTE": {s: rdir(f"final_TOKEN-COMPUTE_s{s}") for s in FINAL_SEEDS}}
        ev = {c: {s: feval(rd) for s, rd in d.items()} for c, d in runs.items()}
        t4v2_0 = feval(rdir("final_TOKEN_s0"))  # deviation 4: V2-code TOKEN-4 s0 sensitivity comparator
        per = {}
        for s in FINAL_SEEDS:
            c, t = ev["CAND"][s], ev["TOKEN-4"][s]
            if not c or not t:
                continue
            ca, ta = c["audit"]["p100"], t["audit"]["p100"]
            pb_d = paired_bootstrap(ca["domain"]["functional"]["docagg"], ta["domain"]["functional"]["docagg"])
            pb_c = paired_bootstrap(ca["cluster"]["functional"]["docagg"], ta["cluster"]["functional"]["docagg"])
            sens = None
            if s == 0 and t4v2_0:
                tb = t4v2_0["audit"]["p100"]
                sens = {"ce_t4_v2code": t4v2_0["val"]["ce"], "ce_diff_vs_v2code": c["val"]["ce"] - t4v2_0["val"]["ce"],
                        "ce_t4_v1_minus_v2code": t["val"]["ce"] - t4v2_0["val"]["ce"],
                        "fsi_domain_bootstrap_vs_v2code": paired_bootstrap(ca["domain"]["functional"]["docagg"],
                                                                           tb["domain"]["functional"]["docagg"]),
                        "fsi_cluster_bootstrap_vs_v2code": paired_bootstrap(ca["cluster"]["functional"]["docagg"],
                                                                            tb["cluster"]["functional"]["docagg"])}
            per[s] = {"ce_cand": c["val"]["ce"], "ce_t4": t["val"]["ce"], "ce_diff": c["val"]["ce"] - t["val"]["ce"],
                      "ce_cand_prior_off": (c.get("val_prior_off") or {}).get("ce"),
                      "ce_dom_cand": c["val"]["ce_dom"], "ce_dom_t4": t["val"]["ce_dom"],
                      "fsi_domain_bootstrap": pb_d, "fsi_cluster_bootstrap": pb_c,
                      "specialists_domain_cand": ca["domain"]["specialists"],
                      "specialists_domain_t4": ta["domain"]["specialists"],
                      "specialists_cluster_cand": ca["cluster"]["specialists"],
                      "specialists_cluster_t4": ta["cluster"]["specialists"],
                      "mi_domain": [ca["domain"]["routing"]["mi_mean"], ta["domain"]["routing"]["mi_mean"]],
                      "mi_domain_null": [ca["domain"]["routing"]["mi_null_mean"],
                                         ta["domain"]["routing"]["mi_null_mean"]],
                      "topic_entropy_domain": [ca["domain"]["routing"]["topic_entropy_mean"],
                                               ta["domain"]["routing"]["topic_entropy_mean"]],
                      "fsi_shuf_domain": [ca["domain"]["fsi_shuf"], ta["domain"]["fsi_shuf"]],
                      "sensitivity_v2code_t4": sens}
            for cn in ("MEMORY", "TOKEN-COMPUTE"):
                e = ev[cn][s]
                if e:
                    per[s][f"ce_{cn}"] = e["val"]["ce"]
                    per[s][f"fsi_domain_{cn}"] = e["audit"]["p100"]["domain"]["fsi"]
                    per[s][f"ce_cand_minus_{cn}"] = c["val"]["ce"] - e["val"]["ce"]
        out["per_seed"] = {str(k): v for k, v in per.items()}
        complete = len(per) == len(FINAL_SEEDS)

        def ce_gaps(s):  # deviation 4: seed 0 must hold against BOTH TOKEN-4 versions (conservative)
            g = [per[s]["ce_diff"]]
            if per[s].get("sensitivity_v2code_t4"):
                g.append(per[s]["sensitivity_v2code_t4"]["ce_diff_vs_v2code"])
            return g

        def spec_ok(s):
            boots = [(per[s]["fsi_domain_bootstrap"], per[s]["fsi_cluster_bootstrap"])]
            sv = per[s].get("sensitivity_v2code_t4")
            if sv:
                boots.append((sv["fsi_domain_bootstrap_vs_v2code"], sv["fsi_cluster_bootstrap_vs_v2code"]))
            return all(d["diff"] > 0 and d["ci95"][0] > 0 and c_["diff"] > 0 for d, c_ in boots)
        A = complete and all(max(ce_gaps(s)) <= 0.01 for s in FINAL_SEEDS)
        Ap = complete and all(max(ce_gaps(s)) < 0 for s in FINAL_SEEDS)
        B = complete and all(spec_ok(s) for s in FINAL_SEEDS)
        bench = json.loads((ART / "bench.json").read_text()) if (ART / "bench.json").exists() else {}
        E = bool(bench.get("criterion_E", {}).get("pass"))
        out["criteria"] = {"A_ce_within_0.01_every_seed": A, "A+_ce_lower_every_seed": Ap,
                           "B_specialisation_significant_every_seed": B,
                           "C_survives_draft_removal": "implied by B (100% student-only, no drafts)" if B else False,
                           "D_no_drafts_unrestricted_top4": True, "E_inference_cost": E,
                           "bench": {k: v for k, v in bench.items() if isinstance(v, dict) and "speed_ratio_vs_token4" in v}}
        if not complete:
            status = "FAILED DUE TO IMPLEMENTATION OR RESOURCE ERROR"
        elif Ap and B and E:
            status = "FULL SUCCESS"
        elif A and B and E:
            status = "ACCURACY-COMPARABLE SPECIALIZATION SUCCESS"
        elif Ap and not B:
            status = "ACCURACY IMPROVEMENT, SPECIALIZATION HYPOTHESIS UNSUPPORTED"
        elif B and not A:
            status = "SPECIALIZATION WORKS BUT COSTS TOO MUCH ACCURACY"
        elif B and A and not E:
            status = "SPECIALIZATION WORKS BUT COSTS TOO MUCH (INFERENCE COST)"
        else:
            status = "NEGATIVE"
        out["terminal_status"] = status
        # trajectory + cross-seed stability
        for cn, d in runs.items():
            for s, rd in d.items():
                e = ev[cn][s]
                if not e:
                    continue
                traj.setdefault(cn, {})[str(s)] = {
                    tag: {sysn: {"fsi": a[sysn]["fsi"], "fsi_shuf": a[sysn]["fsi_shuf"],
                                 "mi": a[sysn]["routing"]["mi_mean"], "mi_null": a[sysn]["routing"]["mi_null_mean"],
                                 "topic_entropy": a[sysn]["routing"]["topic_entropy_mean"],
                                 "topic_pred_acc": a[sysn]["topic_pred_acc"],
                                 "topic_pred_chance": a[sysn]["topic_pred_chance"],
                                 "n_specialists": a[sysn]["specialists"]["n_significant"]}
                          for sysn in ("domain", "cluster") if sysn in a} | {"ce_meas": a["ce_meas"],
                                                                            "routing_entropy": a["routing_entropy"]}
                    for tag, a in e["audit"].items()}
        stab = {}
        for cn in ("CAND", "TOKEN-4"):
            e0, e1 = ev[cn][0], ev[cn][1]
            if e0 and e1:
                for tag in sorted(set(e0["audit"]) & set(e1["audit"])):
                    for sysn in ("domain", "cluster"):
                        stab.setdefault(cn, {}).setdefault(tag, {})[sysn] = cross_seed_stability(
                            e0["audit"][tag][sysn]["routing"]["profiles"], e1["audit"][tag][sysn]["routing"]["profiles"])
        traj["cross_seed_stability"] = stab
        out["equal_compute"] = {"tc_budget": st.get("tc_budget"),
                                "ce_cand_minus_tc": {str(s): per[s].get("ce_cand_minus_TOKEN-COMPUTE") for s in per}}
        out["forecast_attribution"] = {"ce_cand_minus_memory": {str(s): per[s].get("ce_cand_minus_MEMORY") for s in per},
                                       "fsi_domain_memory": {str(s): per[s].get("fsi_domain_MEMORY") for s in per}}
    # counterfactual-utility analysis (all V2 runs with in-run audits)
    for p in sorted(glob.glob(str(ROOT / f"runs/*_{RUN_PREFIX}*/summary.json"))):
        s = json.loads(Path(p).read_text())
        au = s.get("cfaudit") or []
        rd = Path(p).parent
        util = [json.loads(l_).get("util") for l_ in (rd / "metrics.jsonl").read_text().splitlines()
                if '"util"' in l_] if (rd / "metrics.jsonl").exists() else []
        if au or util:
            cfu["runs"][rd.name] = {"n_audits": len(au),
                                    "rho_fo_exact_mean": (sum(a["rho_fo_exact"] for a in au) / len(au)) if au else None,
                                    "top4_fo_exact_mean": (sum(a["top4_fo_exact"] for a in au) / len(au)) if au else None,
                                    "route_ok_frac": (sum(bool(a["route_ok"]) for a in au) / len(au)) if au else None,
                                    "util_loss_first": util[0] if util else None,
                                    "util_loss_last": util[-1] if util else None}
    cfu["v1_estimator_validation"] = json.loads((ROOT / "artifacts/latent-forecast-router-distill-v1/cf_validation.json"
                                                 ).read_text())["mean_spearman"]
    guard.atomic_write_json(ART / "final_analysis.json", out, overwrite=True)
    guard.atomic_write_json(ART / "specialization_trajectory.json", traj, overwrite=True)
    guard.atomic_write_json(ART / "counterfactual_utility_analysis.json", cfu, overwrite=True)
    print(json.dumps({"terminal_status": out["terminal_status"]}))


if __name__ == "__main__":
    main()
