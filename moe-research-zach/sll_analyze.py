#!/usr/bin/env python
"""csr-semantic-lift-expert-learning-v1 analysis (CPU; reads frozen decisions, evaluations, training logs and the
post-terminal descriptive VAL pass). Writes (derived, regenerated): results.json and tables.md. Separates
preregistered results (search_decision.json, promotion.json), post-hoc diagnostics and descriptive VAL analyses."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402


def main():
    import numpy as np
    import sll_run as R
    from csr.sll_paths import ART, GPU_CAP_H
    jl = R.jl
    res = {"time": guard.now(), "state": jl(ART / "STATE.json"), "calibration_note": "calibration.json",
           "checks": {"cpu_unit": "cpu_tests_20261118T064656Z.json", "cpu_trainer": sorted(p.name for p in ART.glob("cpu_tests_train_tt*.json")),
                      "gpu_v1": jl(ART / "gpu_checks.json")["all_pass"], "gpu_v2": jl(ART / "gpu_checks_v2.json")["all_pass"]},
           "search": jl(ART / "search_decision.json"), "promotion": jl(ART / "promotion.json")}
    # trajectories of the multiplier diagnostics (E / F, lambda 0.5)
    traj = {}
    for rn in ("S_E_L05_s100", "S_F_L05_s100", "S_E_L05_s101", "S_F_L05_s101"):
        lr, _ = R.lift_logs(rn)
        traj[rn] = [{"step": x["step"], "lam_eff": x["lam_eff"],
                     "m_absdev_wmean": float(np.mean([v["m_absdev_wmean"] for v in x["m_by_layer"].values()])) if x["m_by_layer"] else 0.0,
                     "clip": float(np.mean([v["clip_lo"] + v["clip_hi"] for v in x["m_by_layer"].values()])) if x["m_by_layer"] else 0.0,
                     "m_q": (list(x["m_by_layer"].values())[5]["q"] if x["m_by_layer"] else None),
                     "lift_absmean": x["stats"]["lift_absmean"], "stability": x["stats"]["stability_r"],
                     "ess_min": x["stats"]["ess_min"], "jaccard": float(np.mean(x["stats"].get("cluster_top4_jaccard_by_layer") or [0])),
                     "gn_expert": x.get("gn_expert_params"), "gn_nonexpert": x.get("gn_nonexpert_params"),
                     "entropy_min": min(x.get("routing_entropy_by_layer") or [0]),
                     "train_load_min": min(x.get("load_min_x_uniform_by_layer") or [0]),
                     "train_load_max": max(x.get("load_max_x_uniform_by_layer") or [0])} for x in lr]
    res["multiplier_trajectories"] = traj
    res["posthoc_chain"] = jl(guard.ROOT / ".tmp/semantic-lift-expert-learning-v1/posthoc_chain.json")
    pv = {}
    for f in sorted((ART / "post_val").glob("*.json")):
        d = jl(f)
        pv[f.stem] = {"val_ce": d["val"]["ce"], "val_ce_dom": d["val"]["ce_dom"],
                      "domain_nmi": d["routing_domain"]["nmi_mean"], "domain_nmi_null_doclabel": d["routing_domain"]["nmi_null_doclabel_mean"],
                      "e5_nmi": d["routing_e5"]["nmi_mean"], "domain_topk_overlap": d["domain_topk_overlap"]["mean_pairwise_overlap"],
                      "domain_inter": d["per_expert"]["domain"]["mean_interaction"], "domain_inter_ci": d["per_expert"]["domain"]["mean_interaction_ci95"],
                      "domain_spec": d["per_expert"]["domain"]["n_subject_specialized"],
                      "domain_inter_boot": d["per_expert"]["domain"]["boot_mean_samples"],
                      "e5_inter": d["per_expert"]["e5"]["mean_interaction"], "routing_entropy": d["routing_entropy"],
                      "load_min_x_uniform": float(np.asarray(d["load"]).min() * 32), "load_max_x_uniform": float(np.asarray(d["load"]).max() * 32)}
    paired = {}
    for s in (100, 101):
        for a, b in (("E", "F"), ("E", "C6"), ("E", "EDL"), ("C6", "SHUF"), ("F", "SHUF"), ("EDL", "T4")):
            ka, kb = f"{a}_s{s}", f"{b}_s{s}"
            if ka in pv and kb in pv:
                d = np.asarray(pv[ka]["domain_inter_boot"]) - np.asarray(pv[kb]["domain_inter_boot"])
                paired[f"{a}-{b}_s{s}"] = {"domain_inter_diff": pv[ka]["domain_inter"] - pv[kb]["domain_inter"],
                                           "ci95": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))],
                                           "val_ce_diff": pv[ka]["val_ce"] - pv[kb]["val_ce"],
                                           "domain_nmi_diff": pv[ka]["domain_nmi"] - pv[kb]["domain_nmi"]}
    for v in pv.values():
        v.pop("domain_inter_boot")
    res["post_val_descriptive"] = pv
    res["post_val_paired"] = paired
    by = {}
    for line in (ART / "RUN_LOG.jsonl").read_text().splitlines():
        r = json.loads(line)
        by[r.get("phase", "?")] = by.get(r.get("phase", "?"), 0.0) + float(r.get("gpu_seconds") or 0) / 3600
    led = {}
    for line in (ART / "compute_ledger.tsv").read_text().splitlines():
        p = line.split("\t")
        if len(p) >= 6:
            led[p[5]] = led.get(p[5], 0.0) + float(p[2]) / 3600
    res["compute"] = {"run_log_gpu_h_by_phase": by, "compute_ledger_gpu_h_by_category": led,
                      "total_gpu_h_ledger": sum(led.values()), "caps": GPU_CAP_H,
                      "tokens_per_s": {rn: R.status_of(rn)["tokens_per_s"] for rn in R.run_dirs() if rn.startswith("S_")}}
    res["deviations"] = [json.loads(x) for x in (ART / "deviations.jsonl").read_text().splitlines()]
    inv = sorted(ART.glob("protected_inventory_before.verify_*.json"))
    res["integrity"] = jl(inv[-1]) if inv else None
    res["disk_free_gib"] = guard.disk_free_bytes() / 2**30
    guard.atomic_write_json(ART / "results.json", res, overwrite=True)
    print(json.dumps({"post_val": {k: {x: (round(y, 5) if isinstance(y, float) else y) for x, y in v.items()
                                       if x in ("val_ce", "domain_nmi", "domain_inter", "domain_spec", "domain_topk_overlap", "e5_inter")}
                                   for k, v in pv.items()}, "paired": paired, "compute": res["compute"]["compute_ledger_gpu_h_by_category"]},
                     default=str, indent=0)[:6000])


if __name__ == "__main__":
    main()
