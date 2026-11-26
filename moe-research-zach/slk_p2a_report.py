#!/usr/bin/env python
"""csr-semantic-edl-long-adaptive-k-v1 interim (Phase 1 + Phase 2A) analysis, CPU only, reads existing JSON outputs.
Writes artifacts/semantic-edl-long-k/phase2a_analysis.json and phase2a_tables.md (derived; rewritten on each call).
Run after `slk_analyze.py final` and `slk_integrity.py`."""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402
from csr.slk_paths import ART, GPU_CAP_H  # noqa: E402

ROOT = guard.ROOT
CONDS = ("C6", "SHUF", "T4", "EDL")
MET = ("ce", "nmi", "probe", "fsi_domain", "interaction", "n_specialized", "nmi_e5", "coherence_e5", "routing_entropy",
       "load_max_x", "n_below_0.1")


def jl(p):
    p = ART / p if not str(p).startswith("/") else Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def compute():
    by_phase, by_kind = {}, {}
    for line in (ART / "RUN_LOG.jsonl").read_text().splitlines():
        r = json.loads(line)
        h = float(r.get("gpu_seconds") or 0) / 3600
        ph = r.get("phase", "?")
        by_phase[ph] = by_phase.get(ph, 0.0) + h
        k = f"{ph}/{r.get('kind', '?')}"
        by_kind[k] = by_kind.get(k, 0.0) + h
    import slk_run
    return {"gpu_h_total_compute_ledger": slk_run.gpu_hours(), "by_phase_run_log": by_phase, "by_phase_kind": by_kind,
            "ledger_categories": {c: slk_run.gpu_hours({c}) for c in ("final", "search", "evaluation", "calibration")},
            "caps": GPU_CAP_H,
            "note": "compute_ledger.tsv counts every GPU process including interrupted attempts; RUN_LOG counts "
                    "driver jobs. Search = Phase-2A screening/generations; final = Phase-1 245M training."}


def disk():
    import slk_run
    tot = 0
    own = [d for k, d in slk_run.run_dirs().items() if "semantic_edl_long_k" in d]
    for d in own:
        r = subprocess.run(["du", "-sb", str(ROOT / d)], capture_output=True, text=True)
        tot += int(r.stdout.split()[0]) if r.stdout else 0
    sz = lambda p: int(subprocess.run(["du", "-sb", str(p)], capture_output=True, text=True).stdout.split()[0])  # noqa
    return {"own_run_dirs": len(own), "own_run_dirs_gib": tot / 2**30, "artifacts_gib": sz(ART) / 2**30,
            "tmp_gib": sz(ROOT / ".tmp/semantic-edl-long-k") / 2**30, "free_gib": guard.disk_free_bytes() / 2**30,
            "reserve_gib": 100}


def main():
    guard.check_root()
    p1 = jl("phase1_decision.json")
    fa = jl("final_analysis.json") or {}
    traj = jl("SPECIALIZATION_TRAJECTORY.json") or {}
    S = p1["seeds"]
    M = p1["metrics"]
    out = {"time": guard.now(), "state": (jl("STATE.json") or {}).get("state"),
           "scope": "Phase 1 + Phase 2A only (user scope reduction 2026-11-16, DEVIATIONS n=2); Phase 2B not run",
           "phase1": {"seeds": S, "metrics": {c: {str(s): {m: M[c][str(s)][m] for m in MET} for s in S} for c in CONDS},
                      "c6_minus_shuf": {m: {str(s): M["C6"][str(s)][m] - M["SHUF"][str(s)][m] for s in S} for m in MET},
                      "criteria": {"R_domain_nmi": p1["R_domain_nmi"], "R_probe": p1["R_probe"], "R_pass": p1["R_pass"],
                                   "F_interaction": p1["F_interaction"], "CE_c6_minus_t4": p1["ce_c6_minus_t4"],
                                   "P_pass": p1["P_pass"], "pathology_ok_by_seed": p1["pathology_ok_by_seed"],
                                   "params_equal": p1["params_equal"],
                                   "semantic_beyond_grouping": p1["semantic_beyond_grouping"], "success": p1["success"]},
                      "edl_reference": p1["edl_reference"],
                      "paired_doc_bootstrap_ce": fa.get("paired_doc_bootstrap_ce"),
                      "per_seed_differences": fa.get("per_seed_differences"), "summary": fa.get("summary"),
                      "cross_seed_profile_consistency": fa.get("cross_seed_profile_consistency"),
                      "mass_control": fa.get("mass_control"),
                      "trajectory_c6_shuf": {f"{c}_s{s}": traj.get(f"P1_{c}_s{s}") for c in ("C6", "SHUF", "T4", "EDL")
                                             for s in S}}}
    scr = jl("phase2/screen.json")
    dec = jl("phase2/decision_2a.json")
    cal = jl("phase2/calibsel_2a.json")
    gate = jl("phase2/gate_2b.json")
    km = jl("latent/k_metrics.json")
    out["phase2a"] = {"k_metrics": km["metrics"] if km else None, "screen": scr["results"] if scr else None,
                      "decision": dec, "calibsel": cal, "gate_2b": gate,
                      "budget_checks": {p.name: jl(p) for p in sorted((ART / "phase2").glob("budget_*.json"))},
                      "gen_docs": {p.name: guard.sha256_file(p) for p in sorted((ROOT / "docs").glob("SLK_GEN*.md"))}}
    out["compute"] = compute()
    out["disk"] = disk()
    out["deviations"] = [json.loads(x) for x in (ART / "DEVIATIONS.jsonl").read_text().splitlines()]
    integ = sorted(ART.glob("integrity_*.json"))
    out["integrity"] = {"file": str(integ[-1].relative_to(ROOT)), **{k: v for k, v in jl(integ[-1]).items()
                                                                     if k != "phase1_runs"},
                        "phase1_runs_summary": {k: v for k, v in jl(integ[-1])["phase1_runs"].items() if k != "runs"}} \
        if integ else None
    guard.atomic_write_json(ART / "phase2a_analysis.json", out, overwrite=True)

    # ---------------------------------------------------------------- tables
    L = []
    f = lambda x, n=4: "—" if x is None else (f"{x:+.{n}f}" if isinstance(x, float) and x < 0 else  # noqa
                                               (f"{x:.{n}f}" if isinstance(x, float) else str(x)))
    L.append("## Phase 1 — VAL at 245M tokens (p100), per seed\n")
    L.append("| cond | seed | CE | domain NMI | probe acc | domain FSI | interaction | spec. experts | e5 NMI |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for c in CONDS:
        for s in S:
            m = M[c][str(s)]
            L.append(f"| {c} | {s} | {m['ce']:.4f} | {m['nmi']:.4f} | {m['probe']:.3f} | {m['fsi_domain']:.5f} | "
                     f"{m['interaction']:.6f} | {m['n_specialized']} | {m['nmi_e5']:.4f} |")
    L.append("\n## Phase 1 — C6 − SHUF per seed\n")
    L.append("| metric | " + " | ".join(f"s{s}" for s in S) + " | mean | positive |")
    L.append("|---|" + "---|" * (len(S) + 2))
    for m in ("ce", "nmi", "probe", "fsi_domain", "interaction", "n_specialized", "nmi_e5", "coherence_e5"):
        d = [M["C6"][str(s)][m] - M["SHUF"][str(s)][m] for s in S]
        L.append(f"| {m} | " + " | ".join(f"{x:+.6f}" if abs(x) < 0.01 else f"{x:+.4f}" for x in d) +
                 f" | {sum(d) / len(d):+.6f} | {sum(x > 0 for x in d)}/{len(d)} |")
    if scr:
        L.append("\n## Phase 2A — CALIB-SEARCH screening at 50M (s50), semantic vs matched shuffled\n")
        L.append("| K | seed | CE sem | CE shuf | FSI_e5 sem | FSI_e5 shuf | Δe5 | FSI_ps sem | FSI_ps shuf | Δpseudo | "
                 "ΔCE | pathology |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        pool = dict(scr["results"])
        for k, v in pool.items():
            for s in ("0", "1"):
                a, b, d = v["rows"][f"SEM_s{s}_s50"], v["rows"][f"SHUF_s{s}_s50"], v["per_seed"][s]
                L.append(f"| {k} | {s} | {a['ce']:.4f} | {b['ce']:.4f} | {a['fsi_e5']:.5f} | {b['fsi_e5']:.5f} | "
                         f"{d['d_e5']:+.5f} | {a['fsi_pseudo']:.5f} | {b['fsi_pseudo']:.5f} | {d['d_ps']:+.5f} | "
                         f"{d['d_ce']:+.4f} | {d['pathology']} |")
        L.append("\n| K | mean Δe5 | mean Δpseudo | mean ΔCE | eligible | Δe5>0 both | repeatable |")
        L.append("|---|---|---|---|---|---|---|")
        allp = dec["screen"] if dec else {k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in pool.items()}
        for k, v in allp.items():
            L.append(f"| {k} | {v['mean']['d_e5']:+.5f} | {v['mean']['d_ps']:+.5f} | {v['mean']['d_ce']:+.4f} | "
                     f"{v['clustering_eligible']} | {v['e5_pos_both']} | {v['repeatable']} |")
        L.append("\n### s20 rows (descriptive)\n")
        L.append("| K | seed | Δe5 s20 | Δpseudo s20 | ΔCE s20 |")
        L.append("|---|---|---|---|---|")
        for k, v in pool.items():
            for s in ("0", "1"):
                a, b = v["rows"][f"SEM_s{s}_s20"], v["rows"][f"SHUF_s{s}_s20"]
                L.append(f"| {k} | {s} | {a['fsi_e5'] - b['fsi_e5']:+.5f} | {a['fsi_pseudo'] - b['fsi_pseudo']:+.5f} | "
                         f"{a['ce'] - b['ce']:+.4f} |")
    if dec:
        L.append(f"\nGenerations: {json.dumps(dec['generations'], default=str)[:3000]}\n")
        L.append(f"Repeatable: {dec['repeatable']}; winner: {dec['winner']}\n")
    if cal:
        L.append(f"CALIB-SELECTION: {json.dumps(cal['diff'])}; confirmed {cal['confirmed']}\n")
    if gate:
        L.append(f"Gate 2B: {json.dumps(gate)}\n")
    c = out["compute"]
    L.append(f"\nCompute: total {c['gpu_h_total_compute_ledger']:.2f} GPU-h; by phase {json.dumps({k: round(v, 3) for k, v in c['by_phase_run_log'].items()})}\n")
    L.append(f"Disk: {json.dumps({k: round(v, 2) if isinstance(v, float) else v for k, v in out['disk'].items()})}\n")
    guard.atomic_write_bytes(ART / "phase2a_tables.md", ("\n".join(L) + "\n").encode(), overwrite=True)
    print("\n".join(L))


if __name__ == "__main__":
    main()
