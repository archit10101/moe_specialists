"""specialization-audit-v1: report tables (CPU, read-only over analysis outputs).

Derives report-ready CSV/JSON summaries from artifacts/specialization-audit-v1/analysis/ and the existing,
unchanged csr-representative-v1 analysis (read-only, for CE context). Writes a NEW directory
artifacts/specialization-audit-v1/report_tables/ and refuses to run if it exists. No statistic used by the frozen
decision rule is recomputed or altered here; the "descriptive_*" outputs are labelled descriptive sensitivity
checks (prereg §10 category 4: "measurement sensitivity is investigated descriptively without changing this
analysis").
"""
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AUD = ROOT / "artifacts" / "specialization-audit-v1"
AN = AUD / "analysis"
OUT = AUD / "report_tables"
PRIMARY_BLOCKS = (1, 4, 7)


def wcsv(name, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    p = OUT / name
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    os.chmod(p, 0o400)


def rcsv(name):
    with open(AN / name, newline="") as f:
        return list(csv.DictReader(f))


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    if "--execute" not in sys.argv:
        print("dry run; pass --execute")
        return
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; never overwritten")
    os.umask(0o077)
    OUT.mkdir()
    a = json.loads((AN / "analysis.json").read_text())
    rep = json.loads((ROOT / "artifacts" / "rep-v1" / "analysis.json").read_text())
    comp = {r["key"]: r for r in csv.DictReader(open(AUD / "compute_runs.csv"))}
    per = a["per_checkpoint"]
    fsum = rcsv("functional_summary.csv")
    fexp = rcsv("functional_experts.csv")
    probe = {(r["key"], r["view"]): r for r in rcsv("probe.csv")}

    # 1. per-checkpoint primary/secondary metrics with CE context
    rows = []
    for k in sorted(per):
        p = per[k]
        r32, t32 = p["routing"]["enc_K32"], p["routing"]["tfidf_K32"]
        fu = p["functional"]
        rows.append(dict(
            key=k, condition=k[-1], seed=int(k[1]),
            val_ce_rep_v1=rep["table"][k]["val_ce"], audit_ce_fp32_mask=float(comp[k]["mean_ce_fp32_mask"]),
            R_enc32=mean(r32["mi_adj"]), R_tfidf32=mean(t32["mi_adj"]), R_enc16=mean(p["routing"]["enc_K16"]["mi_adj"]),
            R_enc64=mean(p["routing"]["enc_K64"]["mi_adj"]),
            MI_enc32=mean(r32["mi"]), NMI_enc32=mean(r32["nmi"]), null_MI_mean_enc32=mean(r32["null_mi_mean"]),
            null_MI_p975_enc32=mean(r32["null_mi_p975"]), perm_p_layer_mean=r32["perm_p_layer_mean"],
            pool_MI_adj_enc32=(mean(r32["pool_mi_adj"]) if r32.get("pool_mi_adj") and None not in r32["pool_mi_adj"] else ""),
            H_T_given_E=mean(r32["H_T_given_E"]), purity_mw=mean(r32["purity_massweighted"]),
            jsd_diversity=mean(r32["between_expert_jsd_diversity"]),
            within_expert_cos_excess=mean(r32["within_expert_cosine_excess_massweighted"]),
            zero_load_total=sum(r32["zero_load_experts"]), low_support_total=sum(r32["low_support_experts"]),
            P_enc32=float(probe[(k, "enc_K32")]["macro_f1"]), P_tfidf32=float(probe[(k, "tfidf_K32")]["macro_f1"]),
            P_shuffled_enc32=float(probe[(k, "enc_K32")]["shuffled_macro_f1"]),
            P_majority_enc32=float(probe[(k, "enc_K32")]["majority_macro_f1"]),
            F_enc32=fu["enc_K32"]["F_primary"], F_tfidf32=fu["tfidf_K32"]["F_primary"],
            F_enc16=fu["enc_K16"]["F_primary"], F_enc64=fu["enc_K64"]["F_primary"],
            F_all7_secondary_enc32=fu["enc_K32"]["F_all7_secondary"],
            null_F_mean_enc32=fu["enc_K32"]["null_F_primary_mean"], null_F_p975_enc32=fu["enc_K32"]["null_F_primary_p975"]))
    wcsv("per_checkpoint.csv", rows)
    byk = {r["key"]: r for r in rows}

    # 2. per-condition means and seed ranges
    crow = []
    metrics = [m for m in rows[0] if m not in ("key", "condition", "seed")]
    for c in "ABCDEF":
        ks = [f"s{s}_{c}" for s in range(3)]
        d = {"condition": c}
        for m in metrics:
            v = [byk[k][m] for k in ks]
            if all(isinstance(x, (int, float)) for x in v):
                d[m + "_mean"] = mean(v)
                d[m + "_min"] = min(v)
                d[m + "_max"] = max(v)
        crow.append(d)
    wcsv("per_condition.csv", crow)

    # 3. primary-layer functional interactions (blocks 1, 4, 7), each checkpoint and view, plus descriptive scaling
    prim = []
    for r in fsum:
        if int(r["layer_block"]) in PRIMARY_BLOCKS:
            g = float(r["global_mean_dce_mean_over_experts"])
            ew = float(r["equal_weighted"]) if r["equal_weighted"] not in ("", "None") else None
            prim.append(dict(key=r["key"], condition=r["key"][-1], seed=int(r["key"][1]), view=r["view"],
                             layer_block=int(r["layer_block"]), equal_weighted=ew,
                             utilization_weighted=r["utilization_weighted"], n_eligible=int(r["n_eligible"]),
                             n_zero_load=int(r["n_zero_load"]), n_insufficient_support=int(r["n_insufficient_support"]),
                             global_mean_dce=g,
                             descriptive_ratio_interaction_over_global_dce=(ew / g if (ew is not None and g) else "")))
    wcsv("primary_layer_functional.csv", prim)

    # 4. descriptive scale check per checkpoint (enc_K32): mean over primary blocks of I/global ΔCE
    scale = []
    for k in sorted(per):
        pr = [p for p in prim if p["key"] == k and p["view"] == "enc_K32"]
        scale.append(dict(key=k, condition=k[-1], seed=int(k[1]), F_enc32=byk[k]["F_enc32"],
                          global_mean_dce_primary=mean(p["global_mean_dce"] for p in pr),
                          descriptive_scaled_F=mean(p["descriptive_ratio_interaction_over_global_dce"] for p in pr)))
    wcsv("descriptive_scaled_F.csv", scale)
    sk = {r["key"]: r for r in scale}

    # 5. coverage of excluded experts (primary layers): share of selected (token, expert) events in eligible experts
    cov = defaultdict(lambda: dict(experts=0, eligible=0, tokens=0, tokens_eligible=0))
    for r in fexp:
        if r["primary_layer"] != "True":
            continue
        key = (r["key"][-1], r["view"], int(r["layer_block"]))
        c = cov[key]
        c["experts"] += 1
        n = int(r["n_tokens"])
        c["tokens"] += n
        if r["eligible"] == "True":
            c["eligible"] += 1
            c["tokens_eligible"] += n
    crows = []
    for (cnd, view, blk), c in sorted(cov.items()):
        crows.append(dict(condition=cnd, view=view, layer_block=blk, experts_total_3seeds=c["experts"],
                          experts_eligible=c["eligible"], experts_excluded=c["experts"] - c["eligible"],
                          selected_token_events=c["tokens"],
                          share_selected_events_in_eligible=(c["tokens_eligible"] / c["tokens"] if c["tokens"] else "")))
    wcsv("coverage_primary.csv", crows)

    # 6. contrasts with CE context, compact
    con = a["contrasts"]
    cc = []
    for name in ("E-D", "C-B", "D-F", "B-A"):
        X, Y = name.split("-")
        for s in "012":
            d = con[name]["seeds"][s]
            r, f, t = d["routing_enc_K32"], d["functional_enc_K32"], d["functional_tfidf_K32"]
            cc.append(dict(
                contrast=name, seed=int(s), dCE_val_rep_v1=rep["per_seed"][s]["contrasts"][name]["diff"],
                dR_enc32=r["dR"], dR_ci_lo=r["dR_ci95"][0], dR_ci_hi=r["dR_ci95"][1],
                null_dR_p025=r["null_dMI_p025"], null_dR_p975=r["null_dMI_p975"],
                dR_tfidf32=d["routing_tfidf_K32"]["dR"], dP_enc32=d["probe_enc_K32"]["dP"],
                dP_ci_lo=d["probe_enc_K32"]["dP_ci95"][0], dP_ci_hi=d["probe_enc_K32"]["dP_ci95"][1],
                dF_enc32=f["dF"], dF_ci_lo=f["dF_ci95"][0], dF_ci_hi=f["dF_ci95"][1],
                null_dF_p025=f["null_dF_p025"], null_dF_p975=f["null_dF_p975"],
                dF_tfidf32=t["dF"], dF_tfidf_ci_lo=t["dF_ci95"][0], dF_tfidf_ci_hi=t["dF_ci95"][1],
                dF_enc16=d["functional_enc_K16"]["dF"], dF_enc64=d["functional_enc_K64"]["dF"],
                dF_all7_secondary=f["dF_all7_secondary"],
                descriptive_d_scaledF=sk[f"s{s}_{X}"]["descriptive_scaled_F"] - sk[f"s{s}_{Y}"]["descriptive_scaled_F"],
                descriptive_d_global_dce=sk[f"s{s}_{X}"]["global_mean_dce_primary"] - sk[f"s{s}_{Y}"]["global_mean_dce_primary"]))
    wcsv("contrasts_compact.csv", cc)

    summ = {"decision": a["decision"], "routing_criteria": {n: con[n]["routing_increase"] for n in con},
            "functional_criteria": {n: con[n]["functional_increase"] for n in con},
            "routing_decrease_descriptive": {n: con[n].get("routing_decrease_descriptive") for n in con},
            "functional_decrease_descriptive": {n: con[n].get("functional_decrease_descriptive") for n in con},
            "cross_seed": a["cross_seed"], "note": "descriptive_* fields are post-hoc descriptive sensitivity checks, "
            "not part of the frozen decision rule"}
    p = OUT / "summary.json"
    p.write_text(json.dumps(summ, indent=1, default=str))
    os.chmod(p, 0o400)
    print("wrote", sorted(x.name for x in OUT.iterdir()))


if __name__ == "__main__":
    main()
