#!/usr/bin/env python
"""csr-semantic-edl-long-adaptive-k-v1 analysis (CPU; reads evaluation JSONs only).

  slk_analyze.py phase1     preregistered Phase-1 decision (section 5) -> phase1_decision.json (written once)
  slk_analyze.py final      final_analysis.json, SPECIALIZATION_TRAJECTORY.json, run_manifest.tsv (derived; rewritten)"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402
from csr.slk_paths import ART  # noqa: E402

CONDS = ("C6", "SHUF", "T4", "EDL")
P100 = "val_p100_full_pe100k.json"


def jl(p):
    return json.loads(Path(p).read_text())


def seeds_done():
    out = []
    for s in (0, 1, 2, 3):
        if all((ART / "evals" / f"P1_{c}_s{s}" / P100).exists() for c in CONDS):
            out.append(s)
    return out


def p100(c, s):
    return jl(ART / "evals" / f"P1_{c}_s{s}" / P100)


def metrics_of(e):
    r = e["routing"]["domain"]
    pe = e["per_expert"]["domain"]
    return {"ce": e["val"]["ce"], "ce_dom": e["val"]["ce_dom"], "nmi": r["nmi_mean"], "mi": r["mi_mean"],
            "nmi_null_route": r["nmi_null_route_mean"], "nmi_null_doclabel": r["nmi_null_doclabel_mean"],
            "probe": r["probe_logreg"]["acc"], "probe_bal": r["probe_logreg"]["balanced_acc"],
            "probe_centroid": r["probe_centroid_acc"], "fsi_domain": r["fsi"], "fsi_shuf_labels": r.get("fsi_shuffled_labels"),
            "subject_entropy": r["subject_entropy_load_adj_mean"], "max_lift": r["max_lift_mean"],
            "interaction": pe["mean_interaction"], "interaction_ci95": pe["mean_interaction_ci95"],
            "interaction_decile_only": pe["mean_interaction_decile_only"], "n_specialized": pe["n_subject_specialized"],
            "frac_positive": pe["frac_positive"], "corr_interaction_load": pe["corr_interaction_load"],
            "interaction_e5": e["per_expert"]["e5"]["mean_interaction"],
            "nmi_e5": e["routing"]["cluster"]["nmi_mean"], "fsi_e5": e["routing"]["cluster"].get("fsi"),
            "coherence_e5": e["coherence_e5"], "load_min_x": r["load_min_x_uniform"], "load_max_x": r["load_max_x_uniform"],
            "n_below_0.1": r["n_below_0.1_uniform"], "routing_entropy": e["meas"]["routing_entropy"],
            "generic_importance": sum(e["per_expert"]["generic_importance_mean_dce"]) /
            len(e["per_expert"]["generic_importance_mean_dce"])}


def sign_rule(diffs):
    n = len(diffs)
    need = n - 1 if n >= 4 else n
    pos = sum(1 for d in diffs if d > 0)
    return {"diffs": diffs, "n_pos": pos, "need": need, "mean": sum(diffs) / n, "pass": bool(pos >= need and
                                                                                        sum(diffs) / n > 0)}


def phase1_decision():
    p = ART / "phase1_decision.json"
    if p.exists():
        return jl(p)
    S = seeds_done()
    if len(S) < 3:
        raise SystemExit(f"fewer than 3 complete seed quadruples: {S}")
    M = {c: {s: metrics_of(p100(c, s)) for s in S} for c in CONDS}
    R_nmi = sign_rule([M["C6"][s]["nmi"] - M["SHUF"][s]["nmi"] for s in S])
    R_probe = sign_rule([M["C6"][s]["probe"] - M["SHUF"][s]["probe"] for s in S])
    F = sign_rule([M["C6"][s]["interaction"] - M["SHUF"][s]["interaction"] for s in S])
    dce_t4 = [M["C6"][s]["ce"] - M["T4"][s]["ce"] for s in S]
    ce_ok = sum(dce_t4) / len(S) <= 0.010
    path = {s: bool(M["C6"][s]["n_below_0.1"] == 0 and M["C6"][s]["load_max_x"] <= 4.0
                    and M["C6"][s]["routing_entropy"] >= 0.45) for s in S}
    gc = jl(ART / "gpu_checks.json")
    P = bool(all(path.values()) and gc["params_equal"])
    R_pass = bool(R_nmi["pass"] and R_probe["pass"])
    success = bool(R_pass and F["pass"] and ce_ok and P)
    # explicit-EDL reference (semantic-edl-v1 definitions)
    edl_s = [s for s in S]
    dce_edl = [M["C6"][s]["ce"] - M["EDL"][s]["ce"] for s in edl_s]
    inter_edl_mean = sum(M["EDL"][s]["interaction"] for s in edl_s) / len(edl_s)
    comparable = bool(sum(dce_edl) / len(edl_s) <= 0.005 and all(d < 0 for d in dce_t4)
                      and all(M["C6"][s]["interaction"] >= 0.8 * inter_edl_mean for s in edl_s)
                      and all(M["C6"][s]["interaction"] > max(M["T4"][s]["interaction"], M["SHUF"][s]["interaction"])
                              for s in edl_s)
                      and all(M["C6"][s]["fsi_domain"] > max(M["T4"][s]["fsi_domain"], M["SHUF"][s]["fsi_domain"])
                              for s in edl_s))
    beats = bool((all(d < 0 for d in dce_edl) and all(M["C6"][s]["interaction"] >= 0.8 * inter_edl_mean
                                                      for s in edl_s))
                 or (all(M["C6"][s]["interaction"] > M["EDL"][s]["interaction"] for s in edl_s)
                     and sum(dce_edl) / len(edl_s) <= 0.005))
    out = {"time": guard.now(), "seeds": S, "metrics": M,
           "R_domain_nmi": R_nmi, "R_probe": R_probe, "R_pass": R_pass, "F_interaction": F,
           "ce_c6_minus_t4": {"per_seed": dce_t4, "mean": sum(dce_t4) / len(S), "margin": 0.010, "pass": ce_ok},
           "pathology_ok_by_seed": path, "params_equal": gc["params_equal"], "P_pass": P,
           "semantic_beyond_grouping": bool(R_pass and F["pass"]), "success": success,
           "edl_reference": {"ce_c6_minus_edl": dce_edl, "mean": sum(dce_edl) / len(edl_s), "comparable": comparable,
                             "beats": beats, "edl_interaction_mean": inter_edl_mean}}
    guard.atomic_write_json(p, out, mode=0o400)
    import slk_run
    slk_run.decision("Phase-1 preregistered decision", f"phase1_decision.json: R_nmi {R_nmi}, R_probe {R_probe}, "
                     f"F {F}, CE {out['ce_c6_minus_t4']}, P {P}", "-",
                     f"SEMANTIC-C6 {'SUCCEEDS' if success else 'does NOT succeed'}; semantic beyond grouping: "
                     f"{out['semantic_beyond_grouping']}")
    return out


# ------------------------------------------------------------------------------------------------ descriptive
def paired_boot(da, db, B=2000, seed=0):
    import numpy as np
    keys = sorted(set(da) & set(db))
    a = np.array([da[k][0] for k in keys]), np.array([da[k][1] for k in keys])
    b = np.array([db[k][0] for k in keys]), np.array([db[k][1] for k in keys])
    pt = a[0].sum() / a[1].sum() - b[0].sum() / b[1].sum()
    rng = np.random.default_rng(seed)
    W = rng.multinomial(len(keys), np.ones(len(keys)) / len(keys), size=B).astype(float)
    bs = (W @ a[0]) / (W @ a[1]) - (W @ b[0]) / (W @ b[1])
    return {"diff": float(pt), "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
            "n_docs": len(keys)}


def cross_seed(S, cond, tag_file=P100):
    import numpy as np
    from csr.slk_lib import match_profiles
    prof = {s: np.asarray(jl(ART / "evals" / f"P1_{cond}_s{s}" / tag_file)["routing"]["domain"]["profiles_lift"])
            for s in S}
    vals = []
    for i, a in enumerate(S):
        for b in S[i + 1:]:
            vals.append(np.mean([match_profiles(prof[a][li], prof[b][li])[0] for li in range(prof[a].shape[0])]))
    # null: match against a profile set with experts' subject columns permuted
    rng = np.random.default_rng(0)
    nulls = []
    for i, a in enumerate(S):
        for b in S[i + 1:]:
            pb = prof[b][:, :, rng.permutation(prof[b].shape[2])]
            nulls.append(np.mean([match_profiles(prof[a][li], pb[li])[0] for li in range(prof[a].shape[0])]))
    return {"mean_matched_corr": float(np.mean(vals)), "pairs": [float(v) for v in vals],
            "null_subject_permuted": float(np.mean(nulls))}


def mass_control(S):
    import numpy as np
    out = {}
    for s in S:
        rec = {}
        for c in ("C6", "SHUF"):
            pe = p100(c, s)["per_expert"]
            rec[c] = (np.asarray(pe["load"]).reshape(-1), np.asarray(pe["domain"]["interaction"], float))
        allload = np.concatenate([rec["C6"][0], rec["SHUF"][0]])
        edges = np.quantile(allload, [0.2, 0.4, 0.6, 0.8])
        diffs = []
        for qi in range(5):
            m = {c: (np.digitize(rec[c][0], edges) == qi) & np.isfinite(rec[c][1]) for c in rec}
            if m["C6"].any() and m["SHUF"].any():
                diffs.append(float(rec["C6"][1][m["C6"]].mean() - rec["SHUF"][1][m["SHUF"]].mean()))
        out[s] = {"load_quintile_diffs": diffs, "mean_load_matched_diff": float(np.mean(diffs)),
                  "corr_load_interaction": {c: float(np.corrcoef(rec[c][0][np.isfinite(rec[c][1])],
                                                                 rec[c][1][np.isfinite(rec[c][1])])[0, 1]) for c in rec}}
    return out


def trajectory():
    traj = {}
    for d in sorted((ART / "evals").glob("*")):
        for f in sorted(d.glob("val_*.json")):
            e = jl(f)
            r = e["routing"]["domain"]
            t = traj.setdefault(d.name, {}).setdefault(e["tag"], {})
            t.update({"step": e["ckpt"]["step"], "tokens": e["ckpt"]["seen"], "val_ce": e["val"]["ce"],
                      "val_ce_dom": e["val"]["ce_dom"], "domain_nmi": r["nmi_mean"], "domain_mi": r["mi_mean"],
                      "domain_nmi_null_route": r["nmi_null_route_mean"], "probe_logreg": r["probe_logreg"]["acc"],
                      "fsi_domain": r["fsi"], "subject_entropy_load_adj": r["subject_entropy_load_adj_mean"],
                      "load": r["load"], "load_max_x": r["load_max_x_uniform"], "load_min_x": r["load_min_x_uniform"],
                      "e5_nmi": e["routing"]["cluster"]["nmi_mean"], "coherence_e5": e["coherence_e5"],
                      "effective_K": None, "hierarchy_level_weights": None})
            if "per_expert" in e:
                pe = e["per_expert"]["domain"]
                t.update({"per_expert_domain_interaction": pe["mean_interaction"],
                          "per_expert_domain_interaction_ci95": pe["mean_interaction_ci95"],
                          "n_subject_specialized": pe["n_subject_specialized"],
                          "per_expert_by_layer": pe["by_layer"], "pe_tokens_per_domain": e["pe_tokens_per_domain"]})
        for f in sorted(d.glob("dev_*.json")):
            e = jl(f)
            a = e["audit"]
            t = traj.setdefault(d.name, {}).setdefault("dev_" + e["tag"], {})
            t.update({"step": e["ckpt"]["step"], "calib_search_ce": e["calib_search_B"]["ce"],
                      "fsi_e5": a["e5"]["fsi"], "fsi_pseudo": a["pseudo"]["fsi"], "mi_pseudo":
                      a["pseudo"]["routing"]["mi_mean"], "effective_K": None})
    return traj


def final():
    import numpy as np
    S = seeds_done()
    fa = {"time": guard.now(), "seeds": S}
    if (ART / "phase1_decision.json").exists():
        fa["phase1_decision"] = jl(ART / "phase1_decision.json")
    if S:
        M = {c: {s: metrics_of(p100(c, s)) for s in S} for c in CONDS}
        fa["phase1_metrics"] = M
        pairs = (("C6", "SHUF"), ("C6", "T4"), ("C6", "EDL"), ("SHUF", "T4"), ("EDL", "T4"))
        fa["paired_doc_bootstrap_ce"] = {f"{a}-{b}": {s: paired_boot(p100(a, s)["val_doc_ce"], p100(b, s)["val_doc_ce"])
                                                      for s in S} for a, b in pairs}
        fa["per_seed_differences"] = {f"{a}-{b}": {m: [M[a][s][m] - M[b][s][m] for s in S]
                                                   for m in ("ce", "nmi", "probe", "fsi_domain", "interaction",
                                                             "n_specialized", "nmi_e5", "coherence_e5")}
                                      for a, b in pairs}
        fa["summary"] = {c: {m: {"mean": float(np.mean([M[c][s][m] for s in S])),
                                 "sd": float(np.std([M[c][s][m] for s in S], ddof=1)) if len(S) > 1 else None}
                             for m in ("ce", "nmi", "probe", "fsi_domain", "interaction", "n_specialized", "nmi_e5",
                                       "coherence_e5", "subject_entropy")} for c in CONDS}
        fa["per_domain_ce"] = {c: {s: M[c][s]["ce_dom"] for s in S} for c in CONDS}
        fa["cross_seed_profile_consistency"] = {c: cross_seed(S, c) for c in CONDS}
        fa["mass_control"] = mass_control(S)
    for name in ("phase2/screen.json", "phase2/decision_2a.json", "phase2/calibsel_2a.json", "phase2/gate_2b.json",
                 "latent/k_metrics.json", "latent/tables.json", "phase2b/decision_2b.json"):
        if (ART / name).exists():
            fa[name.replace("/", "__").replace(".json", "")] = jl(ART / name)
    import slk_run
    fa["compute"] = {"gpu_h_total": slk_run.gpu_hours(), "final_training_h": slk_run.gpu_hours({"final"}),
                     "search_training_h": slk_run.gpu_hours({"search"}), "evaluation_h": slk_run.gpu_hours({"evaluation"}),
                     "calibration_checks_h": slk_run.gpu_hours({"calibration", "check"})}
    guard.atomic_write_json(ART / "final_analysis.json", fa, overwrite=True)
    guard.atomic_write_json(ART / "SPECIALIZATION_TRAJECTORY.json", trajectory(), overwrite=True)
    rows = ["run_name\trun_dir\tstatus\tseed\tcond\tsteps\ttokens\tconfig_sha256\tplan_sha256\treused"]
    for rn, rd in sorted(slk_run.run_dirs().items()):
        st = slk_run.status_of(rn) or {}
        cfgp = guard.ROOT / rd / "config.json"
        cfg = jl(cfgp) if cfgp.exists() else {}
        rows.append("\t".join(str(x) for x in (rn, rd, st.get("status"), cfg.get("seed"), json.dumps(cfg.get("cond")),
                                                st.get("steps_done"), st.get("seen"),
                                                guard.sha256_file(cfgp) if cfgp.exists() else None,
                                                st.get("plan_sha256"), "forecast_latent_edl_v1" in rd)))
    guard.atomic_write_bytes(ART / "run_manifest.tsv", ("\n".join(rows) + "\n").encode(), overwrite=True)
    print(json.dumps({"seeds": S, "gpu_h": fa["compute"]["gpu_h_total"]}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["phase1", "final"])
    a = ap.parse_args()
    guard.check_root()
    {"phase1": phase1_decision, "final": final}[a.cmd]()


if __name__ == "__main__":
    main()
