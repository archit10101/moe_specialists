#!/usr/bin/env python
"""csr-forecast-latent-edl-v1 final analysis (prereg sections 9, 11). CPU, read-only over evals/ledgers.
Writes artifacts/forecast-latent-edl-v1/final_analysis.json, specialization_trajectory.json, final_table.md."""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402
from csr.fle_paths import ART  # noqa: E402

CONDS = ["T4", "EDL", "PAST", "FL", "SHUF", "FD"]
NB = 2000


def ev(rn):
    p = ART / "evals" / rn / "final.json"
    return json.loads(p.read_text()) if p.exists() else None


def seeds_of(c):
    return [s for s in (0, 1, 2) if ev(f"final_{c}_s{s}")]


def paired_ci(a_list, b_list, rng):
    """Per-document CE difference (a - b) averaged over the seeds present for both; document bootstrap (token-weighted
    per document, documents resampled)."""
    import numpy as np
    docs = None
    for a in a_list + b_list:
        k = set(a["val_doc_ce"])
        docs = k if docs is None else docs & k
    docs = sorted(docs)
    da = np.mean([[a["val_doc_ce"][d][0] / a["val_doc_ce"][d][1] for d in docs] for a in a_list], 0)
    db = np.mean([[b["val_doc_ce"][d][0] / b["val_doc_ce"][d][1] for d in docs] for b in b_list], 0)
    n = np.asarray([a_list[0]["val_doc_ce"][d][1] for d in docs], float)
    diff = da - db
    point = float((diff * n).sum() / n.sum())
    bs = []
    for _ in range(NB):
        i = rng.integers(0, len(docs), len(docs))
        bs.append(float((diff[i] * n[i]).sum() / n[i].sum()))
    return {"mean": point, "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "n_docs": len(docs)}


def main():
    import numpy as np
    from csr.lfr_audit import cross_seed_stability
    guard.check_root()
    rng = np.random.default_rng(2026)
    E = {c: {s: ev(f"final_{c}_s{s}") for s in seeds_of(c)} for c in CONDS}
    out = {"time": guard.now(), "frozen": json.loads((ART / "frozen.json").read_text())}
    # ---------------- CE
    ce = {c: {s: E[c][s]["val"]["ce"] for s in E[c]} for c in CONDS}
    ce_dom = {c: {s: E[c][s]["val"]["ce_dom"] for s in E[c]} for c in CONDS}
    out["val_ce"] = ce
    out["val_ce_dom"] = ce_dom
    diffs = {}
    for c in CONDS:
        if c == "T4":
            continue
        common = sorted(set(E[c]) & set(E["T4"]))
        if not common:
            continue
        d = {s: ce[c][s] - ce["T4"][s] for s in common}
        diffs[f"{c}-T4"] = {"per_seed": d, "mean": float(np.mean(list(d.values()))),
                            "spread": float(np.ptp(list(d.values()))) if len(d) > 1 else 0.0,
                            "paired_doc_ci": paired_ci([E[c][s] for s in common], [E["T4"][s] for s in common], rng),
                            "per_domain_mean": np.mean([np.asarray(ce_dom[c][s]) - np.asarray(ce_dom["T4"][s])
                                                        for s in common], 0).tolist()}
    for a_, b_ in (("FL", "PAST"), ("FL", "EDL"), ("FL", "SHUF"), ("FL", "FD"), ("PAST", "SHUF")):
        common = sorted(set(E[a_]) & set(E[b_]))
        if common:
            d = {s: ce[a_][s] - ce[b_][s] for s in common}
            diffs[f"{a_}-{b_}"] = {"per_seed": d, "mean": float(np.mean(list(d.values()))),
                                   "paired_doc_ci": paired_ci([E[a_][s] for s in common], [E[b_][s] for s in common],
                                                              rng)}
    out["ce_differences"] = diffs
    out["seed_spread_T4"] = float(np.ptp(list(ce["T4"].values()))) if len(ce["T4"]) > 1 else None
    # ---------------- specialisation trajectories
    traj = {}
    for c in CONDS:
        for s in E[c]:
            for tag, a in E[c][s]["audit"].items():
                r = {"fsi_domain": a["domain"]["fsi"], "fsi_cluster": a["cluster"]["fsi"],
                     "fsi_raw_domain": a["domain"]["fsi_raw"], "fsi_raw_cluster": a["cluster"]["fsi_raw"],
                     "mi_domain": a["domain"]["routing"]["mi_mean"], "mi_domain_null": a["domain"]["routing"]["mi_null_mean"],
                     "mi_cluster": a["cluster"]["routing"]["mi_mean"],
                     "mi_cluster_null": a["cluster"]["routing"]["mi_null_mean"],
                     "mi_domain_by_layer": a["domain"]["routing"]["mi"],
                     "topic_entropy_domain": a["domain"]["routing"]["topic_entropy_mean"],
                     "topic_pred_domain": a["domain"]["topic_pred_acc"], "topic_pred_cluster": a["cluster"]["topic_pred_acc"],
                     "within_expert_similarity": a["cluster"].get("within_expert_similarity"),
                     "load_min": a["domain"]["routing"]["load_min"], "load_max": a["domain"]["routing"]["load_max"],
                     "routing_entropy": a["routing_entropy"], "ce_meas": a["ce_meas"]}
                if "fsi_shuf" in a["domain"]:
                    r["fsi_shuf_domain"], r["fsi_shuf_cluster"] = a["domain"]["fsi_shuf"], a["cluster"]["fsi_shuf"]
                if "latent" in a:
                    r["mi_latent"], r["mi_latent_null"] = a["latent"]["mi_mean"], a["latent"]["mi_null_mean"]
                if "per_expert" in a:
                    pe = a["per_expert"]
                    r["per_expert_mean_interaction"] = pe["mean_interaction"]
                    r["per_expert_frac_positive"] = pe["frac_positive"]
                    inter = np.asarray(pe["interaction"]).reshape(11, 32)
                    r["per_expert_mean_interaction_by_layer"] = inter.mean(1).tolist()
                traj.setdefault(c, {}).setdefault(str(s), {})[tag] = r
    out["trajectory"] = traj
    # cross-seed stability at p100 (domain profiles)
    stab = {}
    for c in CONDS:
        ss = sorted(E[c])
        if len(ss) >= 2:
            pa = E[c][ss[0]]["audit"]["p100"]["domain"]["routing"]["profiles"]
            pb = E[c][ss[1]]["audit"]["p100"]["domain"]["routing"]["profiles"]
            stab[c] = cross_seed_stability(pa, pb)
            stab[c].pop("per_layer", None)
    out["cross_seed_stability_domain"] = stab
    # ---------------- frozen criteria
    fl, t4 = traj.get("FL", {}), traj.get("T4", {})
    common = sorted(set(fl) & set(t4))
    crit = {}
    if common:
        g = lambda tr, s, k: tr[s]["p100"][k]
        crit["i_fsi_gt_t4_every_seed"] = all(g(fl, s, "fsi_domain") > g(t4, s, "fsi_domain") and
                                             g(fl, s, "fsi_cluster") > g(t4, s, "fsi_cluster") for s in common)
        rd = np.mean([g(fl, s, "fsi_domain") / g(t4, s, "fsi_domain") for s in common])
        rc = np.mean([g(fl, s, "fsi_cluster") / g(t4, s, "fsi_cluster") for s in common])
        crit["ii_ratio_domain"], crit["ii_ratio_cluster"] = float(rd), float(rc)
        crit["ii_ratio_ge_1.15_both"] = bool(rd >= 1.15 and rc >= 1.15)
        crit["iii_fsi_minus_shuf_pos"] = all(g(fl, s, "fsi_domain") - g(fl, s, "fsi_shuf_domain") > 0 for s in common)
        crit["iv_per_expert_gt_t4"] = all(g(fl, s, "per_expert_mean_interaction") >
                                          g(t4, s, "per_expert_mean_interaction") for s in common)
        crit["meaningful_specialization"] = bool(crit["i_fsi_gt_t4_every_seed"] and crit["ii_ratio_ge_1.15_both"] and
                                                 crit["iii_fsi_minus_shuf_pos"] and crit["iv_per_expert_gt_t4"])
        per_seed_spec = {s: bool(g(fl, s, "fsi_domain") > g(t4, s, "fsi_domain") and
                                 g(fl, s, "fsi_cluster") > g(t4, s, "fsi_cluster") and
                                 g(fl, s, "per_expert_mean_interaction") > g(t4, s, "per_expert_mean_interaction"))
                         for s in common}
        crit["per_seed_specialization_direction"] = per_seed_spec
        d = diffs["FL-T4"]
        crit["better"] = bool(d["mean"] < 0 and all(v < 0 for v in d["per_seed"].values()))
        crit["comparable"] = bool(d["mean"] <= 0.01)
        if crit["meaningful_specialization"] and crit["better"]:
            v = 1
        elif crit["meaningful_specialization"] and crit["comparable"]:
            v = 2
        elif crit["meaningful_specialization"]:
            v = 3
        elif crit["comparable"]:
            v = 4
        else:
            v = 5
        crit["verdict"] = v
        s01 = [s for s in (0, 1) if s in d["per_seed"]]
        signs = {math.copysign(1, d["per_seed"][s]) for s in s01}
        crit["third_seed_triggered"] = bool(len(signs) > 1 or abs(np.mean([d["per_seed"][s] for s in s01])) <= 0.01 or
                                            len({per_seed_spec[s] for s in s01 if s in per_seed_spec}) > 1)
    out["criteria"] = crit
    # ---------------- costs
    rows = [l.split("\t") for l in (ART / "compute_ledger.tsv").read_text().splitlines() if l.strip()]
    cost = {}
    for r in rows:
        cat = r[5]
        if r[4].startswith(("eq_", "smoke_", "rc_")):
            cat = "checks"
        cost[cat] = cost.get(cat, 0.0) + float(r[2])
    out["gpu_seconds_by_category"] = cost
    out["gpu_hours_total"] = sum(cost.values()) / 3600
    guard.atomic_write_json(ART / "final_analysis.json", out, overwrite=True)
    guard.atomic_write_json(ART / "specialization_trajectory.json", traj, overwrite=True)
    print(json.dumps({"criteria": crit, "ce_diffs": {k: (v["mean"], v.get("paired_doc_ci", {}).get("ci95"))
                                                     for k, v in diffs.items()}}, indent=1, default=str))


if __name__ == "__main__":
    main()
