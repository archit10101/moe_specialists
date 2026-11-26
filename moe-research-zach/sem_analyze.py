#!/usr/bin/env python
"""csr-semantic-edl-v1 final analysis (CPU only; reads existing JSON outputs, never a model, VAL or test).

Writes (artifacts/semantic-edl-v1/, overwritten on each call; derived files only):
  final_analysis.json, run_manifest.json, candidate_manifest.json, specialization_trajectories.json,
  compute_summary.json"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402
from csr.sem_paths import ART, T4_S100_RUN  # noqa: E402

FLE = guard.ROOT / "artifacts" / "forecast-latent-edl-v1" / "evals"
SEEDS = (100, 101)


def jl(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run_dirs():
    d = jl(ART / "run_dirs.json")
    d["b_T4_s100"] = T4_S100_RUN
    return d


def dev_row(e):
    a = e["audit"]
    blk = e.get("calib_search_B") or e.get("calib_sel")
    return {"ce": blk["ce"], "fsi_pseudo": a["pseudo"]["fsi"], "fsi_e5": a["e5"]["fsi"],
            "mi_pseudo": a["pseudo"]["routing"]["mi_mean"], "mi_e5": a["e5"]["routing"]["mi_mean"],
            "routing_entropy": a["routing_entropy"], "load_max_x": a["load_max_x_uniform"],
            "load_min_x": a["load_min_x_uniform"]}


def trajectories():
    out = {}
    for d in sorted((ART / "evals").iterdir()):
        r = {}
        for tag in ("s5", "s20", "s50"):
            p = d / f"dev_{tag}.json"
            if p.exists():
                r[tag] = dev_row(jl(p))
        p = d / "calibsel_s50.json"
        if p.exists():
            r["calibsel_s50"] = dev_row(jl(p))
        for tag in ("s20", "s50"):
            for suf in ("_pe", ""):
                p = ART / "post" / f"audit_{d.name}_{tag}{suf}.json"
                if p.exists():
                    a = jl(p)["audit"]
                    r.setdefault("domain_post", {})[tag] = {
                        "fsi_domain": a["domain"]["fsi"], "fsi_domain_shuf_labels": a["domain"].get("fsi_shuf"),
                        "mi_domain": a["domain"]["routing"]["mi_mean"],
                        "mi_domain_null": a["domain"]["routing"]["mi_null_mean"],
                        "topic_pred_domain": a["domain"]["topic_pred_acc"], "fsi_e5_audit_one": a["cluster"]["fsi"],
                        "ce_meas": a["ce_meas"]}
                    break
        out[d.name] = r
    return out


def per_expert():
    out = {}
    for p in sorted((ART / "post").glob("audit_*_s50_pe.json")):
        e = jl(p)
        pe = e["per_expert"]
        rec = {}
        for sysn in ("domain", "pseudo", "e5"):
            s = pe[sysn]
            import numpy as np
            inter = np.asarray(s["interaction"], float)
            ok = np.isfinite(inter)  # NaN = the expert's strongest discovery group has no tokens in the PE rows
            L = inter.reshape(11, 32)
            rec[sysn] = {"mean_interaction": float(inter[ok].mean()), "frac_positive": float((inter[ok] > 0).mean()),
                         "n_experts_defined": int(ok.sum()), "mean_interaction_raw_field": s["mean_interaction"],
                         "by_layer": [float(np.nanmean(r)) for r in L], "n_tokens": s["n_tokens"]}
        g = pe["generic_importance"]
        rec["generic_mean_dce"] = sum(g["mean_dce"]) / len(g["mean_dce"])
        rec["generic_mean_abs_dce"] = sum(g["mean_abs_dce"]) / len(g["mean_abs_dce"])
        out[e["run_name"]] = rec
    return out


def compute():
    rows = []
    for line in (ART / "compute_ledger.tsv").read_text().splitlines():
        p = line.split("\t")
        if len(p) >= 7:
            try:
                rows.append({"start": p[0], "end": p[1], "seconds": float(p[2]), "run": p[3], "job": p[4],
                             "category": p[5], "status": p[6]})
            except ValueError:
                pass
    by = {}
    for r in rows:
        k = f"{r['category']}/{r['status']}"
        by[k] = by.get(k, 0.0) + r["seconds"] / 3600
    du = {}
    rd = run_dirs()
    tot = 0
    for name, d in rd.items():
        if name == "b_T4_s100":
            continue
        b = int(subprocess.run(["du", "-sb", str(guard.ROOT / d)], capture_output=True, text=True).stdout.split()[0])
        du[name] = b
        tot += b
    art_b = int(subprocess.run(["du", "-sb", str(ART)], capture_output=True, text=True).stdout.split()[0])
    free = int(subprocess.run(["df", "-B1", "--output=avail", str(guard.ROOT)], capture_output=True,
                              text=True).stdout.split()[-1])
    return {"gpu_hours_total": sum(r["seconds"] for r in rows) / 3600, "gpu_hours_by_category_status": by,
            "n_jobs": len(rows), "budget_gpu_hours": 24.0, "disk_runs_bytes": du, "disk_runs_total_gib": tot / 2**30,
            "disk_artifacts_gib": art_b / 2**30, "disk_free_gib_now": free / 2**30}


def main():
    s1 = jl(ART / "search" / "stage1_decision.json")
    s2 = jl(ART / "search" / "stage2_decision.json")
    cs = jl(ART / "calibsel_decision.json")
    fz = jl(ART / "frozen.json")
    pur = jl(ART / "post" / "cluster_domain_purity.json")
    traj = trajectories()
    pe = per_expert()
    comp = compute()
    win = fz["frozen"]

    def g(run, *ks):
        x = traj.get(run, {})
        for k in ks:
            if not isinstance(x, dict) or k not in x:
                return None
            x = x[k]
        return x

    comps = {}
    for s in SEEDS:
        c, sh, e, t = f"s_{win}_s{s}", f"shuf_{win}_s{s}", f"b_EDL_s{s}", f"b_T4_s{s}"
        row = {}
        for where, ks in (("calib_search_s50", ("s50",)), ("calib_sel_s50", ("calibsel_s50",))):
            for m in ("ce", "fsi_pseudo", "fsi_e5", "mi_pseudo"):
                vals = {n: g(r, *ks, m) for n, r in (("sem", c), ("shuf", sh), ("edl", e), ("t4", t))}
                row[f"{where}/{m}"] = vals
        for m in ("fsi_domain", "mi_domain", "topic_pred_domain"):
            row[f"domain_post_s50/{m}"] = {n: g(r, "domain_post", "s50", m)
                                          for n, r in (("sem", c), ("shuf", sh), ("edl", e), ("t4", t))}
        for sysn in ("domain", "pseudo", "e5"):
            row[f"per_expert/{sysn}_interaction"] = {n: (pe.get(r, {}).get(sysn) or {}).get("mean_interaction")
                                                     for n, r in (("sem", c), ("shuf", sh), ("edl", e), ("t4", t))}
        comps[s] = row

    fle_val = {}
    for k in ("final_T4_s0", "final_T4_s1", "final_T4_s2", "final_EDL_s0", "final_EDL_s1"):
        p = FLE / k / "final.json"
        if p.exists():
            d = jl(p)
            a = d["audit"]["p100"]
            fle_val[k] = {"val_ce": d["val"]["ce"], "fsi_domain": a["domain"]["fsi"],
                          "per_expert_domain_interaction": (a.get("per_expert") or {}).get("mean_interaction")}

    verdict = {"status": "FAILED — NOT CONFIRMED AT CALIB-SELECTION (no finals; VAL never opened)",
               "vs_explicit_edl": "not established (no VAL); 50M dev evidence: CE comparable/slightly lower, "
                                  "pseudo-domain specialisation not distinguishable from the shuffled control",
               "calibsel_criteria": cs["criteria"]}
    fa = {"time": guard.now(), "prereg_sha256": sha(guard.ROOT / "docs" / "SEMANTIC_EDL_V1_PREREG.md"),
          "verdict": verdict, "frozen": fz, "stage1": {"advance": s1["advance"], "survivors": s1["survivors"],
                                                     "edl_ratio_100_318": s1["edl_ratio_100_318"],
                                                     "rows": {k: {"cand": r["cand"], "beta": r["beta"],
                                                                  "ratio_edl_relative": r["ratio_edl_relative"],
                                                                  "fsi_pseudo_adv_s20": r["fsi_pseudo_adv_s20"],
                                                                  "s5": r["s5"], "s20": r["s20"], "reject": r["reject"]}
                                                              for k, r in s1["rows"].items()}},
          "stage2": {k: {"advance": r["advance"], "criteria": r["criteria"], "mean_dce_edl_s50": r["mean_dce_edl_s50"],
                         "grad": r.get("grad")} for k, r in s2["candidates"].items()},
          "calibsel": cs, "generation_decision": jl(ART / "search" / "generation_decision.json"),
          "comparisons_by_seed": comps, "per_expert": pe, "cluster_domain_purity": pur,
          "fle_val_reference": fle_val, "compute": comp}
    guard.atomic_write_json(ART / "final_analysis.json", fa, overwrite=True)
    guard.atomic_write_json(ART / "specialization_trajectories.json", traj, overwrite=True)
    guard.atomic_write_json(ART / "compute_summary.json", comp, overwrite=True)

    rm = {}
    for name, d in run_dirs().items():
        st = jl(guard.ROOT / d / "status.json")
        cfg = guard.ROOT / d / "config.json"
        rm[name] = {"dir": d, "status": st.get("status"), "reason": st.get("reason"), "steps_done": st.get("steps_done"),
                    "seen_tokens": st.get("seen"), "ckpts": sorted(st.get("ckpts", {})),
                    "config_sha256": sha(cfg) if cfg.exists() else None, "table_sha256": st.get("table_sha256"),
                    "reused": name == "b_T4_s100",
                    "note": ("stopped at its last scheduled search milestone by design (max-steps)"
                             if st.get("reason") == "max-steps" else "")}
    guard.atomic_write_json(ART / "run_manifest.json", rm, overwrite=True)
    cm = {}
    for k, r in s1["rows"].items():
        st2 = s2["candidates"].get(k)
        cm[k] = {"cand": r["cand"], "beta": r["beta"], "stage1": "advanced" if k in s1["advance"] else
                 ("survived, not ranked top-2" if not r["reject"] else f"rejected {r['reject']}"),
                 "stage2": (None if st2 is None else ("advanced" if st2["advance"] else
                                                      f"failed {[c for c, v in st2['criteria'].items() if not v]}")),
                 "frozen": k == win, "calibsel": cs["criteria"] if k == win else None}
    guard.atomic_write_json(ART / "candidate_manifest.json", cm, overwrite=True)
    print(json.dumps({"verdict": verdict["status"], "gpu_h": comp["gpu_hours_total"], "per_expert_runs": list(pe)}))


if __name__ == "__main__":
    main()
