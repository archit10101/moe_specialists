#!/usr/bin/env python
"""csr-semantic-utility-edl-v1: post-terminal DESCRIPTIVE result tables for the report (reads frozen decisions, eval
JSONs, run metrics and status files; never used for any decision). Writes artifacts/semantic-utility-edl-v1/
report_tables.json and report_tables.md (derived, regenerated)."""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402
from csr.sue_paths import ART  # noqa: E402


def jl(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def ev(key, mode, tag):
    import sue_run as R
    e = jl(ART / "evals" / key / f"{mode}_{tag}.json")
    if e is None:
        return None
    return {**R.summary(e), "topic_pred": e["audit"]["pseudo"].get("topic_pred_acc"),
            "topic_entropy": e["audit"]["pseudo"]["routing"].get("topic_entropy_mean")}


def runinfo(rn, upto=None):
    import sue_run as R
    rd = R.run_dirs().get(rn)
    if not rd:
        return None
    st = R.status_of(rn) or {}
    rows = [json.loads(x) for x in (guard.ROOT / rd / "metrics.jsonl").read_text().splitlines()]
    steps = [x for x in rows if "step" in x and (upto is None or x["step"] < upto)]
    dg = [x["util_diag"] for x in rows if "util_diag" in x and (upto is None or x["util_diag"]["step"] < upto)]
    seen_end = (steps[-1]["seen"] if steps else 0) if upto is not None else st.get("seen", 0)
    act = [x for x in steps if x.get("use_active", 0) > 0]
    cfg = jl(guard.ROOT / rd / "config.json") or {}
    u = (cfg.get("cond") or {}).get("utility")
    last = steps[-1] if steps else {}
    sb = [x.get("spearman_brown") for x in dg if x.get("spearman_brown") is not None and math.isfinite(x["spearman_brown"])]
    rr = [x.get("split_half_r") for x in dg if x.get("split_half_r") is not None and math.isfinite(x["split_half_r"])]
    return {"run_dir": rd, "steps": st.get("steps_done"), "tokens": st.get("seen"),
            "first_active_step": act[0]["step"] if act else None, "first_active_tokens": act[0]["seen"] if act else None,
            "tokens_influenced": (seen_end - act[0]["seen"]) if act else 0,
            "coverage_last": dg[-1].get("coverage_valid") if dg else None,
            "split_half_r_last": dg[-1].get("split_half_r") if dg else None,
            "sb_last5_mean": (sum(sb[-5:]) / len(sb[-5:])) if sb else None,
            "split_half_r_sign_pos_frac": (sum(1 for x in rr if x > 0) / len(rr)) if rr else None,
            "n_refresh": len(dg), "tokens_per_s": st.get("tokens_per_s"), "cf_seconds": st.get("cf_seconds"),
            "train_step_seconds": st.get("train_step_seconds"),
            "overhead_frac": (st["cf_seconds"] / max(1e-9, st["train_step_seconds"] - st["cf_seconds"]))
            if st.get("cf_seconds") else 0.0, "peak_alloc_gib": st.get("peak_alloc_gib"),
            "use_margin_last": last.get("use_margin"), "use_active_last": last.get("use_active"),
            "gamma": u.get("gamma") if u else None, "probe_passes": u.get("probe_passes") if u else None}


def main():
    guard.check_root()
    import sue_run as R
    sd, fz, cs = jl(ART / "search_decision.json"), jl(ART / "frozen.json"), jl(ART / "confsel.json")
    t4_20 = ev("POST_T4_s100", "dev", "s20")
    out = {"time": guard.now(), "search": {}, "promotion": {}, "confsel": cs, "refs50": {}, "runs": {}}
    ref20 = ev("REF_C6_s100", "dev", "s20")
    for k in sd["candidates"]:
        rn = f"S_{k}_s100"
        s = ev(rn, "dev", "s20")
        out["search"][k] = {**{x: y for x, y in s.items() if x != "boot"},
                            "d_ce_vs_T4": s["ce"] - t4_20["ce"] if t4_20 else None, "d_ce_vs_C6": s["ce"] - ref20["ce"],
                            "d_inter_vs_C6": s["inter"] - ref20["inter"],
                            "d_inter_vs_T4": s["inter"] - t4_20["inter"] if t4_20 else None, **{"run": runinfo(rn, upto=318)}}
    out["search_refs"] = {"REF_C6_s100": {x: y for x, y in ref20.items() if x != "boot"},
                          "T4_s100": {x: y for x, y in t4_20.items() if x != "boot"} if t4_20 else None}
    for seed in (100, 101):
        refs = {n: ev(f"{n}_s{seed}", "dev", "s50") for n in ("REF_C6", "REF_SHUF", "POST_T4", "POST_EDL")}
        out["refs50"][seed] = {n: ({x: y for x, y in v.items() if x != "boot"} if v else None) for n, v in refs.items()}
        for cand in fz["tried"]:
            for side, pre in (("SEM", "S"), ("SHUF", "H")):
                rn = f"{pre}_{cand}_s{seed}"
                s = ev(rn, "dev", "s50")
                if s is None:
                    continue
                row = {x: y for x, y in s.items() if x != "boot"}
                for n, v in refs.items():
                    if v:
                        row[f"d_ce_vs_{n}"] = s["ce"] - v["ce"]
                        row[f"d_inter_vs_{n}"] = s["inter"] - v["inter"]
                row["run"] = runinfo(rn)
                cse = ev(rn, "confsel", "s50")
                if cse:
                    row["confsel"] = {x: y for x, y in cse.items() if x != "boot"}
                out["promotion"].setdefault(cand, {}).setdefault(str(seed), {})[side] = row
    out["compute"] = {"by_phase_gpu_h": {}, "total_gpu_h": R.gpu_hours()}
    for line in (ART / "RUN_LOG.jsonl").read_text().splitlines():
        r = json.loads(line)
        ph = r.get("phase", "?")
        out["compute"]["by_phase_gpu_h"][ph] = out["compute"]["by_phase_gpu_h"].get(ph, 0) + float(r.get("gpu_seconds") or 0) / 3600
    guard.atomic_write_json(ART / "report_tables.json", out, overwrite=True)
    L = ["# V1 report tables (descriptive, regenerated)", "", "## Search (20M, seed 100; dev = CALIB-SEARCH half B)", "",
         "| cand | CE | ΔCE vs T4 | ΔCE vs C6 | interaction [95% CI] | Δ vs C6 | spec (frac) | MI | topic pred | load min/max | ent | first active (step, M tok) | M tok influenced | bank cov | SB last5 | overhead |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in out["search"].items():
        ri = v["run"]
        L.append(f"| {k} | {v['ce']:.4f} | {v['d_ce_vs_T4']:+.4f} | {v['d_ce_vs_C6']:+.4f} | {v['inter']:.6f} "
                 f"[{v['inter_ci'][0]:.6f}, {v['inter_ci'][1]:.6f}] | {v['d_inter_vs_C6']:+.6f} | {v['spec']} "
                 f"({v['spec'] / 352:.3f}) | {v['mi']:.4f} | {v.get('topic_pred') or float('nan'):.3f} | "
                 f"{v['load_min_x_uniform']:.2f}/{v['load_max_x_uniform']:.2f} | {v['routing_entropy']:.3f} | "
                 f"{ri['first_active_step']}, {(ri['first_active_tokens'] or 0) / 1e6:.1f} | {ri['tokens_influenced'] / 1e6:.1f} | "
                 f"{ri['coverage_last']:.2f} | {ri['sb_last5_mean'] if ri['sb_last5_mean'] is None else round(ri['sb_last5_mean'], 3)} | "
                 f"{ri['overhead_frac']:.2f} |")
    L += ["", "## 50M (seeds 100/101): candidates vs references (dev)", "",
          "| run | seed | CE | ΔCE vs T4 | ΔCE vs EDL | ΔCE vs C6 | ΔCE vs SHUF-C6 | interaction [95% CI] | spec | MI | ent | M tok influenced | SB last5 | overhead |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for cand, bys in out["promotion"].items():
        for seed, sides in bys.items():
            for side, v in sides.items():
                ri = v["run"]
                L.append(f"| {side}-{cand} | {seed} | {v['ce']:.4f} | {v.get('d_ce_vs_POST_T4', float('nan')):+.4f} | "
                         f"{v.get('d_ce_vs_POST_EDL', float('nan')):+.4f} | {v['d_ce_vs_REF_C6']:+.4f} | "
                         f"{v['d_ce_vs_REF_SHUF']:+.4f} | {v['inter']:.6f} [{v['inter_ci'][0]:.6f}, {v['inter_ci'][1]:.6f}] | "
                         f"{v['spec']} | {v['mi']:.4f} | {v['routing_entropy']:.3f} | {ri['tokens_influenced'] / 1e6:.1f} | "
                         f"{ri['sb_last5_mean'] if ri['sb_last5_mean'] is None else round(ri['sb_last5_mean'], 3)} | {ri['overhead_frac']:.2f} |")
    for seed, refs in out["refs50"].items():
        for n, v in refs.items():
            if v:
                L.append(f"| {n} | {seed} | {v['ce']:.4f} | | | | | {v['inter']:.6f} [{v['inter_ci'][0]:.6f}, "
                         f"{v['inter_ci'][1]:.6f}] | {v['spec']} | {v['mi']:.4f} | {v['routing_entropy']:.3f} | | | |")
    (ART / "report_tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
