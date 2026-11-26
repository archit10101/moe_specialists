#!/usr/bin/env python
"""csr-domain-forecast-curriculum-v1: final analysis (prereg sections 8-9). CPU only; reads frozen.json, the driver
state, run summaries/metrics, calibration audits, validation evals, caches, bench and ledgers. Writes (never
overwrites; a re-run writes a new stamped copy and refreshes nothing else):
  artifacts/domain-forecast-curriculum-v1/final_analysis.json, run_manifest.tsv, compute_ledger.tsv,
  data_manifest.json, specialization_trajectory.json, report_tables.md
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402
from csr.dfc_lib import ART, CACHE, DATA, FROZEN, ST, bootstrap_diff  # noqa: E402

SEEDS = (0, 1, 2)
CONDS = ("TOKEN-MAIN", "LABEL", "FORECAST", "SOFT", "WRONG", "TOKEN-ALL-DATA", "TOKEN-COMPUTE")
CURR = ("LABEL", "FORECAST", "SOFT", "WRONG")
TAGS = ("p20", "p40", "p50", "p75", "p100")
ROOT = guard.ROOT


def jl(p):
    return json.loads(Path(p).read_text())


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx, ry = x.argsort().argsort().astype(float), y.argsort().argsort().astype(float)
    return float(np.corrcoef(rx, ry)[0, 1]) if len(x) > 2 else None


def boot_mean_ci(vals, B=2000, seed=0):
    v = np.asarray(vals, float)
    rng = np.random.default_rng(seed)
    bs = v[rng.integers(0, len(v), (B, len(v)))].mean(1)
    return {"mean": float(v.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "n": len(v)}


def profile_dev(C_layer):
    """C (D, E) -> per-expert deviation of p(d|e) from p(d): (E, D)."""
    C = np.asarray(C_layer, float)
    pd = C.sum(1) / C.sum()
    pde = C / np.maximum(C.sum(0, keepdims=True), 1)
    return (pde - pd[:, None]).T


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def stability(counts_by_seed):
    """counts_by_seed: list over seeds of (L, D, E). Index-aligned and best-match cosine of deviation profiles."""
    al, bm = [], []
    for i in range(len(counts_by_seed)):
        for j in range(i + 1, len(counts_by_seed)):
            for li in range(len(counts_by_seed[i])):
                A, B = profile_dev(counts_by_seed[i][li]), profile_dev(counts_by_seed[j][li])
                al.append(np.mean([cos(A[e], B[e]) for e in range(len(A))]))
                bm.append(np.mean([max(cos(A[e], B[f]) for f in range(len(B))) for e in range(len(A))]))
    return {"index_aligned_cos": float(np.mean(al)), "best_match_cos": float(np.mean(bm))}


def main():
    guard.check_root()
    st = jl(ST / "driver_state.json")
    frozen = jl(FROZEN)
    stamp = guard.utc_stamp()
    out = {"created": guard.now(), "prereg_sha256": frozen["prereg_sha256"], "frozen_sha256": guard.sha256_file(FROZEN)}
    man = jl(DATA / "manifest.json")
    inv = jl(ART / "inventory.json")
    # ------------------------------------------------ validation
    val = {}
    for c in CONDS:
        for s in SEEDS:
            val[(c, s)] = jl(ART / "evals" / f"val_{c}_s{s}.json")
    ref_idx = val[("TOKEN-MAIN", 0)]["doc_index"]
    for v in val.values():
        assert v["doc_index"] == ref_idx
    # bootstrap / paired units are PAPERS (all segments of a paper aggregated)
    papers = val[("TOKEN-MAIN", 0)]["doc_paper"]
    uniq = sorted(set(papers))
    pidx = np.asarray([uniq.index(p) for p in papers]) if len(uniq) < 5000 else \
        np.asarray([{u: i for i, u in enumerate(uniq)}[p] for p in papers])

    def by_paper(x):
        return np.bincount(pidx, weights=np.asarray(x, float), minlength=len(uniq))
    for v in val.values():
        v["doc_nll"] = by_paper(v["doc_nll"]).tolist()
    seg_dom = np.asarray(val[("TOKEN-MAIN", 0)]["doc_domain"])
    doc_n = by_paper(val[("TOKEN-MAIN", 0)]["doc_n"])
    doc_dom = np.zeros(len(uniq), dtype=int)
    doc_dom[pidx] = seg_dom
    out["n_val_papers"] = len(uniq)
    out["n_val_segments"] = len(papers)
    table = {}
    for c in CONDS:
        ces = [val[(c, s)]["val_ce"] for s in SEEDS]
        dom = np.asarray([val[(c, s)]["val_ce_domain"] for s in SEEDS])
        table[c] = {"ce_per_seed": ces, "mean": float(np.mean(ces)), "sd": float(np.std(ces, ddof=1)),
                    "ce_domain_mean": dom.mean(0).tolist(), "ce_domain_per_seed": dom.tolist(),
                    "worst_domain_ce_per_seed": dom.max(1).tolist(), "worst_domain_ce_mean": float(dom.max(1).mean()),
                    "macro_ce_mean": float(dom.mean())}
    out["validation"] = table
    out["domains"] = man["domains"]
    comps = {}
    for c in CONDS:
        if c == "SOFT":
            continue
        per_seed = []
        diffs = []
        for s in SEEDS:
            a = np.asarray(val[("SOFT", s)]["doc_nll"])
            b = np.asarray(val[(c, s)]["doc_nll"])
            per_seed.append(bootstrap_diff(a, doc_n, b, doc_n, seed=s))
            diffs.append(a - b)
        dm = np.mean(diffs, axis=0)
        pooled = bootstrap_diff(dm, doc_n, np.zeros_like(dm), doc_n, seed=99)
        per_dom = {}
        for d in range(5):
            m = doc_dom == d
            per_dom[man["domains"][d]] = float(dm[m].sum() / doc_n[m].sum())
        better = pooled["ci95"][1] < 0 and sum(p["mean"] < 0 for p in per_seed) >= 2
        worse = pooled["ci95"][0] > 0 and sum(p["mean"] > 0 for p in per_seed) >= 2
        comps[f"SOFT_vs_{c}"] = {"per_seed": per_seed, "pooled": pooled, "per_domain_pooled": per_dom,
                                 "verdict": "SOFT better" if better else ("SOFT worse" if worse else "not resolved"),
                                 "seed_wins_soft": sum(p["mean"] < 0 for p in per_seed)}
    for c in CONDS:  # every condition vs TOKEN-MAIN for context
        if c == "TOKEN-MAIN":
            continue
        diffs = [np.asarray(val[(c, s)]["doc_nll"]) - np.asarray(val[("TOKEN-MAIN", s)]["doc_nll"]) for s in SEEDS]
        dm = np.mean(diffs, axis=0)
        comps[f"{c}_minus_TOKEN-MAIN"] = {"pooled": bootstrap_diff(dm, doc_n, np.zeros_like(dm), doc_n, seed=7),
                                       "per_seed": [float(d.sum() / doc_n.sum()) for d in diffs]}
    out["comparisons"] = comps
    claims = {"curriculum_improves_final_accuracy": comps["SOFT_vs_TOKEN-MAIN"]["verdict"],
              "forecasts_help_beyond_labels": comps["SOFT_vs_LABEL"]["verdict"],
              "specialization_helps_beyond_forecast_features": comps["SOFT_vs_FORECAST"]["verdict"],
              "semantic_forecasting_matters": comps["SOFT_vs_WRONG"]["verdict"],
              "data_efficiency_vs_all_data": comps["SOFT_vs_TOKEN-ALL-DATA"]["verdict"],
              "compute_efficiency_vs_equal_compute": comps["SOFT_vs_TOKEN-COMPUTE"]["verdict"]}
    out["claims"] = claims
    # ------------------------------------------------ runs, summaries, learning curves
    runs = {}
    for k, rdir in st["runs"].items():
        rd = ROOT / rdir
        if not (rd / "summary.json").exists():
            continue
        runs[k] = {"run_dir": rdir, "summary": jl(rd / "summary.json"), "config": jl(rd / "config.json")}
    curves = {}
    for c in CONDS:
        for s in SEEDS:
            k = f"{c}_s{s}"
            pts = {}
            for ln in (ROOT / runs[k]["run_dir"] / "metrics.jsonl").read_text().splitlines():
                r = json.loads(ln)
                if "monitor" in r:
                    pts[r["valid_tokens_seen"]] = r["monitor"]  # later attempts overwrite repeated points
            curves[k] = sorted([[t, v.get("neutral"), v.get("active")] for t, v in pts.items()])
    out["learning_curves_calib_mon"] = curves
    mt = {}
    for c in CONDS:
        grid = {}
        for s in SEEDS:
            for t, neu, _ in curves[f"{c}_s{s}"]:
                g = int(round(t / 5_625_000))
                grid.setdefault(g, []).append(neu)
        mt[c] = {str(g * 5_625_000): float(np.mean(v)) for g, v in sorted(grid.items()) if len(v) == 3}
    out["matched_token_curves_mean_neutral"] = mt
    # ------------------------------------------------ specialisation trajectory (calibration audits)
    traj = {}
    for c in CONDS:
        for s in SEEDS:
            k = f"{c}_s{s}"
            rd = ROOT / runs[k]["run_dir"]
            for p in sorted(rd.glob("audit_*.json")):
                a = jl(p)
                met = a["metrics"]
                ab = a["ablation"]
                traj[f"{k}|{a['tag']}|{a['mode']}"] = {
                    "condition": c, "seed": s, "tag": a["tag"], "mode": a["mode"], "strength": a["strength"],
                    "seen": a["ckpt_seen"], "calib_ce": a["calib_ce"], "calib_ce_domain": a["calib_ce_domain"],
                    "mi": met["mi"], "mi_mean": float(np.mean(met["mi"])), "mi_norm_mean": float(np.mean(met["mi_norm"])),
                    "mi_null_mean": met.get("mi_null_mean"), "mi_null_p95": met.get("mi_null_p95"),
                    "h_dom_given_expert_mean": float(np.mean(met["h_dom_given_expert"])),
                    "h_dom_given_expert_min": float(np.min(met["h_dom_given_expert"])),
                    "prov_mass_mean": float(np.mean(met["prov_mass"])), "prov_mass": met["prov_mass"],
                    "load": met["load"], "max_unused": max(x["unused"] for x in met["load"]),
                    "max_share": max(x["max_share"] for x in met["load"]),
                    "collapsed": any(x["unused"] > 0 or x["max_share"] > 0.5 for x in met["load"]),
                    "p_e_given_d": met["p_e_given_d"], "counts": a["counts"],
                    "specificity_cells": [cell["specificity"] for row in ab["per_layer_domain"] for cell in row],
                    "specificity_layer_domain_mean": ab["mean_specificity_layer_domain"],
                    "specificity_all_layers_enriched": ab["mean_specificity_all_layers_enriched"],
                    "specificity_all_layers_provisional": ab["mean_specificity_all_layers_provisional"],
                    "all_layers_enriched": ab["all_layers_enriched"], "all_layers_provisional": ab["all_layers_provisional"],
                    "gate_mean": a.get("gate_mean"), "rerouted_frac": a.get("rerouted_frac"),
                    "scale_f": a["scale_f"], "scale_p": a["scale_p"]}
    summ_traj = {}
    for c in CONDS:
        for tag in TAGS + ("final",):
            for mode in ("normal", "neutral"):
                cells = [traj.get(f"{c}_s{s}|{tag}|{mode}") for s in SEEDS]
                if any(x is None for x in cells):
                    continue
                spec = [v for x in cells for v in x["specificity_cells"]]
                summ_traj[f"{c}|{tag}|{mode}"] = {
                    "mi_mean": float(np.mean([x["mi_mean"] for x in cells])),
                    "mi_mean_per_seed": [x["mi_mean"] for x in cells],
                    "mi_null_mean": float(np.mean([np.mean(x["mi_null_mean"]) for x in cells])),
                    "mi_norm_mean": float(np.mean([x["mi_norm_mean"] for x in cells])),
                    "h_dom_given_expert_mean": float(np.mean([x["h_dom_given_expert_mean"] for x in cells])),
                    "prov_mass_mean": float(np.mean([x["prov_mass_mean"] for x in cells])),
                    "calib_ce_mean": float(np.mean([x["calib_ce"] for x in cells])),
                    "specificity_layer_domain": boot_mean_ci(spec),
                    "functional_specialization": boot_mean_ci(spec)["ci95"][0] > 0,
                    "specificity_all_layers_enriched_mean": float(np.mean(
                        [x["specificity_all_layers_enriched"] for x in cells])),
                    "specificity_all_layers_provisional_mean": float(np.mean(
                        [x["specificity_all_layers_provisional"] for x in cells])),
                    "any_collapse": any(x["collapsed"] for x in cells), "max_unused": max(x["max_unused"] for x in cells),
                    "max_share": max(x["max_share"] for x in cells),
                    "stability": stability([x["counts"] for x in cells]),
                    "gate_mean": float(np.mean([np.mean(x["gate_mean"]) for x in cells])) if cells[0]["gate_mean"] else None,
                    "rerouted_frac": float(np.mean([np.mean(x["rerouted_frac"]) for x in cells]))
                    if cells[0]["rerouted_frac"] else None}
    out["specialization_summary"] = summ_traj
    fs = {}
    for c in ("TOKEN-MAIN",) + CURR:
        fs[c] = {tag: summ_traj.get(f"{c}|{tag}|neutral", {}).get("functional_specialization") for tag in TAGS}
        if c in CURR:
            fs[c]["p40_normal"] = summ_traj.get(f"{c}|p40|normal", {}).get("functional_specialization")
    out["functional_specialization"] = fs
    # correlation: specialisation vs final CE across the 15 main-plan runs
    xs40, xs100, sp40, sp100, ys = [], [], [], [], []
    for c in ("TOKEN-MAIN",) + CURR:
        for s in SEEDS:
            k = f"{c}_s{s}"
            xs40.append(traj[f"{k}|p40|neutral"]["mi_mean"])
            xs100.append(traj[f"{k}|p100|neutral"]["mi_mean"])
            sp40.append(traj[f"{k}|p40|neutral"]["specificity_layer_domain_mean"])
            sp100.append(traj[f"{k}|p100|neutral"]["specificity_layer_domain_mean"])
            ys.append(val[(c, s)]["val_ce"])
    out["spec_vs_ce_spearman"] = {"mi_p40_neutral": spearman(xs40, ys), "mi_p100": spearman(xs100, ys),
                                  "specificity_p40_neutral": spearman(sp40, ys),
                                  "specificity_p100": spearman(sp100, ys), "n_runs": len(ys),
                                  "note": "negative rho = more specialisation goes with lower (better) CE"}
    # ------------------------------------------------ compute, throughput, overhead
    gl = []
    for ln in (ROOT / "ledger/gpu_time_ledger.tsv").read_text().splitlines()[1:]:
        f = ln.split("\t")
        if len(f) >= 6 and f[4].startswith("dfc"):
            gl.append({"start": f[0], "end": f[1], "seconds": float(f[2]), "run_id": f[3], "purpose": f[4],
                       "status": f[5]})
    cat = {}
    for r in gl:
        p = r["purpose"]
        k = ("draft_training" if p == "dfc train draft" else "main_training" if p == "dfc train main" else
             "forecast_cache" if "cache" in p else "audits_calibration" if "audit" in p else
             "final_validation_eval" if "val eval" in p else "benchmarks" if "bench" in p else "checks_and_tests")
        cat[k] = cat.get(k, 0.0) + r["seconds"]
    cache_jobs = {}
    for f in sorted(CACHE.glob("*_*.json")):
        if f.name.startswith("manifest"):
            continue
        cache_jobs[f.stem] = jl(f)
    cman = sorted(CACHE.glob("manifest_*.json"))
    draft_s = runs["draft"]["summary"]
    overhead = {"draft_train_step_seconds": draft_s["train_step_seconds"],
                "draft_train_step_seconds_amortized_3_seeds": draft_s["train_step_seconds"] / 3,
                "cache_main_correct_gpu_seconds": cache_jobs["main_correct"]["gpu_seconds"],
                "cache_main_correct_amortized_3_seeds": cache_jobs["main_correct"]["gpu_seconds"] / 3,
                "cache_all_jobs_gpu_seconds": {k: v["gpu_seconds"] for k, v in cache_jobs.items()},
                "cache_manifest": str(cman[-1].relative_to(ROOT)) if cman else None,
                "cache_invariance": jl(cman[-1]).get("invariance_check") if cman else None,
                "calib_forecast_accuracy": jl(cman[-1]).get("calib_forecast_accuracy") if cman else None,
                "data_build_cpu_wall_seconds": man.get("build_seconds")}
    out["overhead"] = overhead
    out["compute_match"] = jl(ART / "compute_match.json")
    tp = {}
    for c in CONDS:
        v = [runs[f"{c}_s{s}"]["summary"] for s in SEEDS]
        tp[c] = {"train_tokens_per_s_mean": float(np.mean([x["train_valid_tokens_per_s_after_warmup"] for x in v])),
                 "train_step_seconds_mean": float(np.mean([x["train_step_seconds"] for x in v])),
                 "valid_train_tokens": v[0]["valid_train_tokens"], "attempts": [x["attempts"] for x in v],
                 "micro_batch_rows_last": [x["micro_batch_rows_last"] for x in v]}
    out["training_throughput"] = tp
    out["gpu_seconds_by_category"] = cat
    out["gpu_seconds_total"] = float(sum(cat.values()))
    bench = sorted(ART.glob("bench_*.json"))
    out["bench"] = jl(bench[-1]) if bench else None
    out["params"] = out["bench"]["runs"]["SOFT_s0"]["params_standalone"] if bench else None
    out["val_loads"] = {f"{c}_s{s}": {"max_unused": max(x["unused"] for x in val[(c, s)]["load_overall"]),
                                      "max_share": max(x["max_share"] for x in val[(c, s)]["load_overall"]),
                                      "mi_mean": float(np.mean(val[(c, s)]["mi"]))} for c in CONDS for s in SEEDS}
    out["data"] = {"selected": man["domains"], "substitutions": inv["substitutions"],
                   "splits": {k: v["valid_tokens"] for k, v in man["splits"].items()},
                   "splits_sha256": man["splits_sha256"]}
    out["deviations"] = [json.loads(x) for x in (ART / "deviations.jsonl").read_text().splitlines()] \
        if (ART / "deviations.jsonl").exists() else []
    # ------------------------------------------------ write files
    def wr(name, obj):
        p = ART / name
        if p.exists():
            p = ART / f"{Path(name).stem}_{stamp}{Path(name).suffix}"
        if isinstance(obj, str):
            guard.atomic_write_bytes(p, obj.encode())
        else:
            guard.atomic_write_json(p, obj)
        return p
    wr("final_analysis.json", out)
    wr("specialization_trajectory.json", {"per_run": traj, "summary": summ_traj, "functional_specialization": fs})
    dm = {"manifest": man, "manifest_sha256": guard.sha256_file(DATA / "manifest.json"), "inventory": inv,
          "inventory_sha256": guard.sha256_file(ART / "inventory.json"),
          "files": {p.name: guard.sha256_file(p) for p in sorted(DATA.iterdir())},
          "caches": {p.name: cache_jobs.get(p.stem, {}).get("sha256") for p in sorted(CACHE.glob("*.npy"))}}
    wr("data_manifest.json", dm)
    lines = ["key\tcondition\tseed\trun_dir\tconfig_sha256\tfinal_checkpoint\tfinal_checkpoint_sha256\tvalid_train_tokens"
             "\tsteps\ttrain_step_seconds\ttokens_per_s\tattempts\tmicro_batch_rows_last\tval_ce\tsuperseded_dirs"]
    for k in ["draft"] + [f"{c}_s{s}" for s in SEEDS for c in CONDS]:
        r = runs[k]
        sm = r["summary"]
        vce = val[(r["config"]["condition"], r["config"]["seed"])]["val_ce"] if k != "draft" else ""
        lines.append("\t".join(str(x) for x in [
            k, r["config"]["condition"], r["config"]["seed"], r["run_dir"], guard.sha256_file(ROOT / r["run_dir"] /
                                                                                             "config.json"),
            sm["final_checkpoint"], guard.sha256_file(ROOT / sm["final_checkpoint"]), sm["valid_train_tokens"],
            sm["steps"], round(sm["train_step_seconds"], 1), round(sm["train_valid_tokens_per_s_after_warmup"] or 0),
            sm["attempts"], sm["micro_batch_rows_last"], vce, ",".join(st.get("superseded", {}).get(k, []))]))
    wr("run_manifest.tsv", "\n".join(lines) + "\n")
    cl = ["start\tend\tgpu_seconds\trun_id\tpurpose\tstatus\tcategory"]
    for r in gl:
        p = r["purpose"]
        k = ("draft_training" if p == "dfc train draft" else "main_training" if p == "dfc train main" else
             "forecast_cache" if "cache" in p else "audits_calibration" if "audit" in p else
             "final_validation_eval" if "val eval" in p else "benchmarks" if "bench" in p else "checks_and_tests")
        cl.append("\t".join(str(r[x]) for x in ("start", "end", "seconds", "run_id", "purpose", "status")) + f"\t{k}")
    cl.append(f"#\t\t\t\tdataset_build_cpu_wall_seconds\t\t{man.get('build_seconds')}")
    for k, v in sorted(cat.items()):
        cl.append(f"#total\t\t{v:.0f}\t\t{k}\t\t{k}")
    wr("compute_ledger.tsv", "\n".join(cl) + "\n")
    print(json.dumps({"validation": {c: (round(v["mean"], 4), round(v["sd"], 4)) for c, v in table.items()},
                      "claims": claims, "gpu_hours": out["gpu_seconds_total"] / 3600}, indent=1))


if __name__ == "__main__":
    main()
