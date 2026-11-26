#!/usr/bin/env python
"""csr-soft-cf-usefulness-v2 analysis (CPU; reads frozen stage outputs, eval JSONs, run metrics and mechanism
snapshots). Writes (derived, regenerated): results.json, run_manifest.tsv, tables.md.
Mechanism time series (prereg section 10, link D): for the cells with a positive confidence score at the first
post-onset snapshot ("favoured") vs cells with score 0 ("neutral") and negative score ("disfavoured"): mean router
preference R, exposure X, gradient per unit exposure G/X and measured DeltaU at every snapshot."""
import json
import math
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402
from csr.scu_paths import ART, GPU_CAP_H  # noqa: E402


def jl(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def mech_series(rn, onset=100):
    import numpy as np
    import scu_run as R
    rd = R.run_dirs().get(rn)
    if not rd:
        return None
    snaps = sorted((guard.ROOT / rd / "mech").glob("mech_s*.npz"))
    zs = [np.load(p) for p in snaps]
    base = [z for z in zs if int(z["step"]) >= onset and z["EV"].sum() >= 10]
    if not base:
        return None
    s0 = base[0]["scores"]
    groups = {"favoured": s0 > 0, "neutral": s0 == 0, "disfavoured": s0 < 0}
    out = {"basis_step": int(base[0]["step"]), "n_cells": {k: int(v.sum()) for k, v in groups.items()}, "series": []}
    for z in zs:
        row = {"step": int(z["step"]), "tokens": int(z["seen"])}
        for g, m in groups.items():
            gx = np.where(z["X"] > 0, z["G"] / np.maximum(z["X"], 1e-12), np.nan)
            row[g] = {"R": float(z["R"][m].mean()), "X": float(z["X"][m].mean()),
                      "G_per_X": float(np.nanmean(gx[m])) if np.isfinite(gx[m]).any() else float("nan"),
                      "dU": float(np.where(z["valid"], z["d"], np.nan)[m][np.isfinite(np.where(z["valid"], z["d"], np.nan)[m])].mean())
                      if (z["valid"] & m).any() else float("nan")}
        out["series"].append(row)
    return out


def nmi_from_audit(e, sysn):
    """Exact post-hoc NMI = 2 MI / (H_topic + H_expert) per layer from the saved audit profiles P(topic|expert) and
    expert loads (the joint is profile x load), averaged over layers (semantic-edl-long-k definition)."""
    import numpy as np
    a = e["audit"].get(sysn)
    if not a or "profiles" not in a:
        return None
    prof, load = np.asarray(a["profiles"]), np.asarray(a["routing"]["load"])
    out = []
    for li in range(prof.shape[0]):
        J = prof[li] * load[li][None, :]
        J = J / max(1e-12, J.sum())
        pk, pe = J.sum(1), J.sum(0)
        nz = J > 0
        mi = float((J[nz] * np.log(J[nz] / (pk[:, None] * pe[None, :])[nz])).sum())
        hk = float(-(pk[pk > 0] * np.log(pk[pk > 0])).sum())
        he = float(-(pe[pe > 0] * np.log(pe[pe > 0])).sum())
        out.append(2 * mi / max(1e-12, hk + he))
    return float(np.mean(out))


def dev_nmi_table():
    """Semantic-cluster (pseudo) and e5-cluster routing NMI for every dev / hold50 evaluation of V2 and the reused V1
    evaluations (descriptive, post hoc)."""
    from csr.scu_paths import SUE_ART
    rows = {}
    for root, tag in ((ART / "evals", "V2"), (SUE_ART / "evals", "V1")):
        for f in sorted(root.glob("*/*.json")):
            e = jl(f)
            rows[f"{tag}:{f.parent.name}:{f.stem}"] = {"pseudo_nmi": nmi_from_audit(e, "pseudo"),
                                                       "e5_nmi": nmi_from_audit(e, "e5"),
                                                       "pseudo_mi": e["audit"]["pseudo"]["routing"]["mi_mean"]}
    return rows


def runrow(rn, upto=None):
    import scu_run as R
    rd = R.run_dirs().get(rn)
    if not rd:
        return None
    st = R.status_of(rn) or {}
    rows = [json.loads(x) for x in (guard.ROOT / rd / "metrics.jsonl").read_text().splitlines()]
    steps = [x for x in rows if "step" in x and (upto is None or x["step"] < upto)]
    dg = [x["util_diag"] for x in rows if "util_diag" in x and (upto is None or x["util_diag"]["step"] < upto)]
    act = [x for x in steps if x.get("use_active", 0) > 0]
    seen_end = 20_057_391 if upto == 318 else st.get("seen", 0)
    sb = [x.get("spearman_brown") for x in dg if x.get("spearman_brown") is not None and math.isfinite(x["spearman_brown"])]
    rr = [x.get("split_half_r") for x in dg if x.get("split_half_r") is not None and math.isfinite(x["split_half_r"])]
    return {"run_dir": rd, "first_active_step": act[0]["step"] if act else None,
            "first_active_tokens": act[0]["seen"] if act else None,
            "tokens_influenced": (seen_end - act[0]["seen"]) if act else 0,
            "coverage_last": dg[-1].get("coverage_valid") if dg else None,
            "sb_last5_mean": (sum(sb[-5:]) / len(sb[-5:])) if sb else None,
            "split_half_r_positive_frac": (sum(1 for x in rr if x > 0) / len(rr)) if rr else None,
            "target_last": dg[-1].get("target") if dg else None,
            "tokens_per_s": st.get("tokens_per_s"),
            "overhead_frac": (st["cf_seconds"] / max(1e-9, st["train_step_seconds"] - st["cf_seconds"]))
            if st.get("cf_seconds") else 0.0, "peak_alloc_gib": st.get("peak_alloc_gib")}


def compute():
    by = {}
    for line in (ART / "RUN_LOG.jsonl").read_text().splitlines():
        r = json.loads(line)
        by[r.get("phase", "?")] = by.get(r.get("phase", "?"), 0.0) + float(r.get("gpu_seconds") or 0) / 3600
    return {"by_phase_gpu_h": by, "total_gpu_h": sum(by.values()), "caps": GPU_CAP_H}


def main():
    guard.check_root()
    import scu_run as R
    sd, pr, fd = jl(ART / "search_decision.json"), jl(ART / "promotion.json"), jl(ART / "final_decision.json")
    res = {"time": guard.now(), "gamma": jl(ART / "gamma.json"), "anchor_summary": None, "search": sd,
           "promotion": pr, "final": fd, "compute": compute(), "runs": {}, "mechanism": {}}
    an = jl(ART / "anchor.json")
    if an:
        res["anchor_summary"] = {"by_size": an["by_size"], "shuf_target": an["shuf_target"],
                                 "sem_target": an["grad_anchor"]["target"]}
    for rn in R.run_dirs():
        res["runs"][rn] = runrow(rn, upto=318 if rn.startswith("S_") and not (pr and pr.get("promoted") and
                                                                              rn.startswith(f"S_{pr['promoted']}_")) else None)
        res["mechanism"][rn] = mech_series(rn)
    if sd:
        for k in sd["candidates"]:
            res["runs"][f"S_{k}_s100@20M"] = runrow(f"S_{k}_s100", upto=318)
    res["dev_nmi"] = dev_nmi_table()
    pv = {}
    for f in sorted((ART / "post_val").glob("*.json")):
        d = jl(f)
        pv[f.stem] = {"val_ce": d["ce"], "val_ce_dom": d["ce_dom"], "domain_nmi": d["routing_domain"]["nmi_mean"],
                      "domain_nmi_null_doclabel": d["routing_domain"]["nmi_null_doclabel_mean"],
                      "e5_nmi": d["routing_e5"]["nmi_mean"], "routing_entropy": d["routing_entropy"]}
    res["post_val_descriptive"] = pv
    res["deviations"] = [json.loads(x) for x in (ART / "DEVIATIONS.jsonl").read_text().splitlines()] \
        if (ART / "DEVIATIONS.jsonl").exists() else []
    inv = sorted(ART.glob("protected_inventory_before.verify_*.json"))
    res["integrity"] = jl(inv[-1]) if inv else None
    sz = lambda p: int(subprocess.run(["du", "-sb", str(p)], capture_output=True, text=True).stdout.split()[0])  # noqa
    own = list(R.run_dirs().values())
    slots = guard.ROOT / ".tmp/soft-cf-usefulness-v2/slots"
    res["disk"] = {"own_runs": len(own), "own_runs_gib": sum(sz(guard.ROOT / d) for d in own) / 2**30,
                   "slots_gib": (sz(slots) / 2**30) if slots.exists() else 0.0, "artifacts_gib": sz(ART) / 2**30,
                   "free_gib": guard.disk_free_bytes() / 2**30}
    guard.atomic_write_json(ART / "results.json", res, overwrite=True)
    rows = ["run_name\trun_dir\tstatus\tseed\tsteps\ttokens\tconfig_sha256\tplan_sha256\tcond"]
    for rn, rd in sorted(R.run_dirs().items()):
        st = R.status_of(rn) or {}
        cfgp = guard.ROOT / rd / "config.json"
        cfg = jl(cfgp) or {}
        rows.append("\t".join(str(x) for x in (rn, rd, st.get("status"), cfg.get("seed"), st.get("steps_done"),
                                                st.get("seen"), guard.sha256_file(cfgp) if cfgp.exists() else None,
                                                st.get("plan_sha256"), json.dumps(cfg.get("cond")))))
    guard.atomic_write_bytes(ART / "run_manifest.tsv", ("\n".join(rows) + "\n").encode(), overwrite=True)
    L = ["# V2 tables (derived, regenerated)", ""]
    for rn, m in res["mechanism"].items():
        if not m:
            continue
        L += [f"## Mechanism chain {rn} (cells by score sign at step {m['basis_step']}; n {m['n_cells']})", "",
              "| step | M tok | R fav / neu / dis | X fav / neu / dis | G/X fav / neu / dis | ΔU fav / neu / dis |",
              "|---|---|---|---|---|---|"]
        for r in m["series"]:
            f = lambda k: " / ".join(f"{r[g][k]:.4g}" for g in ("favoured", "neutral", "disfavoured"))  # noqa
            L.append(f"| {r['step']} | {r['tokens'] / 1e6:.1f} | {f('R')} | {f('X')} | {f('G_per_X')} | {f('dU')} |")
        L.append("")
    (ART / "tables.md").write_text("\n".join(L) + "\n")
    print(json.dumps({"compute": res["compute"], "disk": res["disk"]}))


if __name__ == "__main__":
    main()
