#!/usr/bin/env python
"""scrgp-v1 Stage-1 analysis (prereg sections 5.3-5.5): per-seed virtual-update metrics, paper bootstrap (1,000
replicates, seed 20261121, shared across seeds/pairs/tables), hierarchical seeds x papers bootstrap (10,000; slc_lib
.hier_boot), criteria V0-V7, failure-mode classification, descriptive stability / concentration tables.
Writes artifacts/scrgp-v1/final_analysis.json and stage1_tables.md (overwritten on rerun; inputs are read-only)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402

N_BOOT, BOOT_SEED, N_HIER = 1000, 20261121, 10000
NUIS = ("ce", "pos", "rank", "log_g4", "margin", "log_freq", "log_doclen")


def main():
    guard.check_root()
    import numpy as np
    import torch
    from csr.scrgp_lib import contrast_boot, paper_boot_weights
    from csr.scrgp_paths import ART, SEEDS
    from csr.slc_lib import hier_boot
    sd = ART / "stage1"
    seeds = [s for s in SEEDS if (sd / f"eval_s{s}.npz").exists() and (sd / f"est_s{s}.pt").exists()]
    if len(seeds) != len(SEEDS):
        raise guard.GuardError(f"missing seeds: have {seeds}")
    est = {s: torch.load(sd / f"est_s{s}.pt", map_location="cpu", weights_only=False)["stats"] for s in seeds}
    ev = {s: dict(np.load(sd / f"eval_s{s}.npz")) for s in seeds}
    evj = {s: json.loads((sd / f"eval_s{s}.json").read_text()) for s in seeds}
    papers = ev[seeds[0]]["papers"]
    for s in seeds:
        if not np.array_equal(ev[s]["papers"], papers):
            raise guard.GuardError("B paper sets differ across seeds")
    NP = len(papers)
    W = paper_boot_weights(NP, N_BOOT, BOOT_SEED)
    L, E = ev[seeds[0]]["fps_SEM"].shape[:2]
    npair = L * E
    R = {"seeds": seeds, "n_pairs": npair, "n_papers_B": NP, "per_seed": {}, "eta": {s: evj[s] for s in seeds}}
    per = {}  # per[(metric, tau)] -> (S, npair) point, (S, B) seed-level boot
    for tau in ("SEM", "SHUF"):
        for s in seeds:
            e_ = ev[s]
            r2 = lambda k: e_[f"{k}_{tau}"].reshape(npair, NP)  # noqa: E731
            fpt, fbt = contrast_boot(r2("fps"), r2("fpw"), r2("fcs"), r2("fcw"), W)
            F_pt, F_bt = -fpt, -fbt
            npt, nbt = contrast_boot(r2("nos"), r2("now"), r2("nxs"), r2("nxw"), W)
            N_pt, N_bt = -npt, -nbt
            gs = e_[f"gs_{tau}"].reshape(npair, NP, -1)
            gw = e_[f"gw_{tau}"].reshape(npair, NP, -1)
            with np.errstate(invalid="ignore", divide="ignore"):
                gk = gs.sum(1) / gw.sum(1)  # (npair, K)
                G_pt = np.nanmean(gk, 1)
                gkb = np.einsum("bp,qpk->bqk", W, gs) / np.einsum("bp,qpk->bqk", W, gw)
                G_bt = np.nanmean(gkb, 2)
                sim = e_[f"sim_{tau}"] - e_["base_sum"]
                Sim_pt = float(sim.sum() / e_["base_n"].sum())
                Sim_bt = (W @ sim) / (W @ e_["base_n"])
                nall = e_[f"nall_{tau}"].reshape(npair, NP)
                Nall_pt = nall.sum(1) / e_["ntok"].sum()
            un = est[s]["pairs"][tau]["norm_u"].reshape(-1)
            pred = evj[s]["eta"] * un ** 2
            for name, pt, bt in (("F", F_pt, F_bt), ("N", N_pt, N_bt), ("G", G_pt, G_bt)):
                per.setdefault((name, tau), {"pt": [], "bt": [], "pair": []})
                per[(name, tau)]["pt"].append(float(np.nanmean(pt)))
                per[(name, tau)]["bt"].append(np.nanmean(bt, 1))
                per[(name, tau)]["pair"].append(pt)
            per.setdefault(("Sim", tau), {"pt": [], "bt": [], "pair": []})
            per[("Sim", tau)]["pt"].append(Sim_pt)
            per[("Sim", tau)]["bt"].append(Sim_bt)
            per.setdefault(("pred", tau), {"pt": [], "pair": []})
            per[("pred", tau)]["pt"].append(float(np.mean(pred)))
            per[("pred", tau)]["pair"].append(pred)
            R["per_seed"].setdefault(tau, {}).setdefault(s, {})
            R["per_seed"][tau][s] = {
                "F": float(np.nanmean(F_pt)), "N": float(np.nanmean(N_pt)), "G": float(np.nanmean(G_pt)), "Sim": Sim_pt,
                "Nall_mean": float(np.nanmean(Nall_pt)), "pred_mean": float(np.mean(pred)),
                "F_nan_pairs": int(np.isnan(F_pt).sum()), "N_nan_pairs": int(np.isnan(N_pt).sum()),
                "F_frac_pairs_pos": float(np.nanmean(F_pt > 0)), "N_frac_pairs_pos": float(np.nanmean(N_pt > 0)),
                "realised_over_pred": float(np.nanmean(F_pt) / np.mean(pred)),
                "F_doc_ci": [float(np.nanquantile(np.nanmean(F_bt, 1), q)) for q in (0.025, 0.975)],
                "N_doc_ci": [float(np.nanquantile(np.nanmean(N_bt, 1), q)) for q in (0.025, 0.975)]}
    hb = {}
    for (name, tau), d in per.items():
        if "bt" not in d:
            continue
        r, _ = hier_boot(np.stack(d["bt"]), N_HIER, point=d["pt"])
        hb[f"{name}_{tau}"] = {**r, "per_seed": d["pt"], "n_pos": int(sum(x > 0 for x in d["pt"]))}
    # semantic - shuffled
    dF_bt = np.stack(per[("F", "SEM")]["bt"]) - np.stack(per[("F", "SHUF")]["bt"])
    dF_pt = [a - b for a, b in zip(per[("F", "SEM")]["pt"], per[("F", "SHUF")]["pt"])]
    r, _ = hier_boot(dF_bt, N_HIER, point=dF_pt)
    hb["F_SEM_minus_SHUF"] = {**r, "per_seed": dF_pt, "n_pos": int(sum(x > 0 for x in dF_pt))}
    dN_bt = np.stack(per[("N", "SEM")]["bt"]) - np.stack(per[("N", "SHUF")]["bt"])
    dN_pt = [a - b for a, b in zip(per[("N", "SEM")]["pt"], per[("N", "SHUF")]["pt"])]
    r, _ = hier_boot(dN_bt, N_HIER, point=dN_pt)
    hb["N_SEM_minus_SHUF"] = {**r, "per_seed": dN_pt, "n_pos": int(sum(x > 0 for x in dN_pt))}
    R["hier"] = hb
    # stability / norms (descriptive) + V0 / V7b inputs
    stab = {}
    for tau in ("SEM", "SHUF"):
        st = {}
        for k in ("cos_d12", "cos_u12", "cos_g12", "ratio_d_g", "ratio_u_d", "norm_d", "norm_u", "norm_g", "cos_dg",
                  "orth_ug"):
            arr = np.stack([est[s]["pairs"][tau][k].reshape(L, E) for s in seeds])  # (S, L, E)
            st[k] = {"mean_per_seed": arr.reshape(len(seeds), -1).mean(1).tolist(),
                     "median_per_seed": np.median(arr.reshape(len(seeds), -1), 1).tolist(),
                     "by_layer_mean": arr.mean((0, 2)).tolist(), "mean": float(arr.mean()),
                     "q10_q50_q90": np.quantile(arr, [0.1, 0.5, 0.9]).tolist()}
        u2 = np.stack([est[s]["pairs"][tau]["norm_u"].reshape(L, E) ** 2 for s in seeds])
        st["u2_share_top_layer"] = float((u2.sum((0, 2)) / u2.sum()).max())
        flat = np.sort(u2.mean(0).reshape(-1))[::-1]
        st["u2_share_top5pct_pairs"] = float(flat[:int(np.ceil(0.05 * npair))].sum() / flat.sum())
        ks = np.stack([est[s]["kstar"][tau].reshape(L, E) for s in seeds])
        cl = np.zeros(16)
        np.add.at(cl, ks.reshape(-1), u2.reshape(-1))
        st["u2_share_top_cluster"] = float(cl.max() / cl.sum())
        lifts = np.stack([np.take_along_axis(est[s]["lift"][tau], est[s]["kstar"][tau][:, None, :], 1)[:, 0, :]
                          for s in seeds])
        st["kstar_lift_median"] = float(np.median(lifts))
        st["kstar_lift_by_layer"] = np.median(lifts, (0, 2)).tolist()
        st["kstar_cluster_counts"] = np.bincount(ks.reshape(-1), minlength=16).tolist()
        stab[tau] = st
    R["stability"] = stab
    # V0 matching
    v0 = {"per_seed": {}}
    ok0 = True
    for s in seeds:
        p = est[s]["pairs"]["SEM"]
        mx = {h: np.nanmax(np.abs(p[f"smd_{h}"].reshape(npair, len(NUIS))), 1) for h in ("A1", "A2")}
        mf = {h: float(np.median(p[f"matched_frac_{h}"])) for h in ("A1", "A2")}
        mfB = float(np.median(ev[s]["ffrac_SEM"]))
        smd_med = {h: float(np.median(mx[h])) for h in mx}
        per_var = {h: np.median(np.abs(p[f"smd_{h}"].reshape(npair, len(NUIS))), 0).tolist() for h in ("A1", "A2")}
        ok = all(v <= 0.10 for v in smd_med.values()) and all(v >= 0.80 for v in mf.values()) and mfB >= 0.80
        ok0 &= ok
        v0["per_seed"][s] = {"median_max_abs_smd": smd_med, "median_abs_smd_by_var": per_var, "vars": NUIS,
                             "matched_frac_median": {**mf, "B": mfB},
                             "cond_matched_frac_B_median": float(np.median(ev[s]["cfrac_SEM"])), "ok": ok}
    v0["pass"] = ok0
    # V6 concentration on seed-mean per-pair F_SEM
    Fp = np.stack(per[("F", "SEM")]["pair"])  # (S, npair)
    Fm = np.nanmean(Fp, 0)
    lolo = [float(np.nanmean(np.delete(Fm.reshape(L, E), li, 0))) for li in range(L)]
    ks = np.stack([est[s]["kstar"]["SEM"].reshape(-1) for s in seeds])  # (S, npair)
    loco = []
    for c in range(16):
        vals = [np.nanmean(Fp[i][ks[i] != c]) for i in range(len(seeds))]
        loco.append(float(np.mean(vals)))
    tot = np.nansum(Fm)
    srt = np.sort(np.nan_to_num(Fm))[::-1]
    top5 = float(srt[:int(np.ceil(0.05 * npair))].sum() / tot) if tot > 0 else float("nan")
    lay_share = (np.nansum(Fm.reshape(L, E), 1) / tot).tolist() if tot > 0 else [float("nan")] * L
    cl_sum = np.zeros(16)
    np.add.at(cl_sum, ks.reshape(-1), np.nan_to_num(Fp).reshape(-1))
    cl_share = (cl_sum / np.nansum(Fp)).tolist() if np.nansum(Fp) > 0 else [float("nan")] * 16
    v6 = {"leave_one_layer_out": lolo, "leave_one_cluster_out": loco, "top5pct_pairs_share": top5,
          "layer_share": lay_share, "cluster_share": cl_share,
          "pass": bool(min(lolo) > 0 and min(loco) > 0 and tot > 0 and top5 <= 0.5 and max(lay_share) <= 0.5
                       and max(cl_share) <= 0.5)}
    Fh, Nh, Gh, Sh, Dh = hb["F_SEM"], hb["N_SEM"], hb["G_SEM"], hb["Sim_SEM"], hb["F_SEM_minus_SHUF"]
    cosu = {tau: stab[tau]["cos_u12"]["mean_per_seed"] for tau in stab}
    crit = {
        "V0_matching": v0["pass"],
        "V1_direction": Fh["point"] > 0,
        "V2_interval": Fh["ci95"][0] > 0,
        "V3_seeds": Fh["n_pos"] >= 3,
        "V4_global_ce": bool(Gh["point"] <= 0.5 * Fh["point"] and Sh["point"] <= 0.001),
        "V5_exposure": bool(Nh["point"] > 0 and Nh["n_pos"] >= 3),
        "V6_concentration": v6["pass"],
        "V7a_sem_gt_shuf": bool(Dh["point"] > 0 and Dh["n_pos"] >= 3 and Dh["ci95"][0] > 0),
        "V7b_stability_sem_gt_shuf": int(sum(a > b for a, b in zip(cosu["SEM"], cosu["SHUF"]))) >= 3,
    }
    crit["V7_semantic_gt_shuffled"] = crit["V7a_sem_gt_shuf"] and crit["V7b_stability_sem_gt_shuf"]
    passed = all(crit[k] for k in ("V0_matching", "V1_direction", "V2_interval", "V3_seeds", "V4_global_ce",
                                   "V5_exposure", "V6_concentration", "V7_semantic_gt_shuffled"))
    pred_mean = float(np.mean(per[("pred", "SEM")]["pt"]))
    dnorm = (stab["SEM"]["norm_d"]["mean"], stab["SHUF"]["norm_d"]["mean"])
    fm = {"unstable_subject_differential_gradients": stab["SEM"]["cos_u12"]["mean"] < 0.10,
          "no_difference_from_shuffled": not crit["V7_semantic_gt_shuffled"],
          "global_ce_conflict": not crit["V4_global_ce"],
          "matching_failure": not crit["V0_matching"],
          "insufficient_semantic_information": bool(stab["SEM"]["kstar_lift_median"] < 1.5 or dnorm[0] <= 1.1 * dnorm[1]),
          "layer_expert_cluster_concentration": not crit["V6_concentration"],
          "first_order_prediction_fails_heldout": bool(crit["V0_matching"] and Fh["point"] < 0.25 * pred_mean)}
    R.update({"criteria": crit, "V0": v0, "V6": v6, "stage1_pass": passed,
              "failure_modes": {k: bool(v) for k, v in fm.items()} if not passed else {},
              "predicted_mean_SEM": pred_mean, "norm_d_mean": {"SEM": dnorm[0], "SHUF": dnorm[1]},
              "per_pair_F_SEM_seed_mean_by_layer": np.nanmean(Fm.reshape(L, E), 1).tolist(),
              "per_pair_N_SEM_seed_mean_by_layer": np.nanmean(np.nanmean(np.stack(per[("N", "SEM")]["pair"]), 0)
                                                              .reshape(L, E), 1).tolist(),
              "per_pair_F_SHUF_seed_mean_by_layer": np.nanmean(np.nanmean(np.stack(per[("F", "SHUF")]["pair"]), 0)
                                                               .reshape(L, E), 1).tolist(),
              "verdict": "STAGE1_PASS" if passed else "STAGE1_FAIL", "time": guard.now()})
    inv = sorted(ART.glob("protected_inventory_before.verify_*.json"))
    R["integrity"] = json.loads(inv[-1].read_text()) if inv else None
    guard.atomic_write_json(ART / "final_analysis.json", R, overwrite=True)
    # tables
    f = lambda x: f"{x:+.3e}"  # noqa: E731
    lines = ["# scrgp-v1 Stage 1 — tables (derived from final_analysis.json)\n", f"verdict: **{R['verdict']}**\n",
             "## Criteria\n", "```", json.dumps(crit, indent=1), "```\n",
             "## Held-out virtual-update metrics (mean over 352 pairs; per seed s0..s3; hierarchical 95% CI)\n",
             "| metric | per seed | n>0 | mean | hier CI |", "|---|---|---|---|---|"]
    for k in ("F_SEM", "F_SHUF", "F_SEM_minus_SHUF", "N_SEM", "N_SHUF", "N_SEM_minus_SHUF", "G_SEM", "G_SHUF",
              "Sim_SEM", "Sim_SHUF"):
        h = hb[k]
        lines.append(f"| {k} | {', '.join(f(x) for x in h['per_seed'])} | {h['n_pos']} | {f(h['point'])} | "
                     f"[{f(h['ci95'][0])}, {f(h['ci95'][1])}] |")
    lines += ["", f"predicted in-sample mean (SEM) eta|u|^2: {pred_mean:.3e}; eta per seed: "
              f"{[round(evj[s]['eta'], 6) for s in seeds]}; cap binding: {[evj[s].get('cap_binding') for s in seeds]}",
              "", "## Split-half stability and norms (A1 vs A2; mean over pairs)\n",
              "| quantity | SEM per seed | SHUF per seed |", "|---|---|---|"]
    for k in ("cos_d12", "cos_u12", "cos_g12", "ratio_d_g", "ratio_u_d", "cos_dg"):
        lines.append(f"| {k} | {', '.join(f'{x:.4f}' for x in stab['SEM'][k]['mean_per_seed'])} | "
                     f"{', '.join(f'{x:.4f}' for x in stab['SHUF'][k]['mean_per_seed'])} |")
    lines += ["", "cos_u12 by layer (SEM): " + ", ".join(f"{x:.3f}" for x in stab["SEM"]["cos_u12"]["by_layer_mean"]),
              "cos_u12 by layer (SHUF): " + ", ".join(f"{x:.3f}" for x in stab["SHUF"]["cos_u12"]["by_layer_mean"]),
              f"k* routing lift median: SEM {stab['SEM']['kstar_lift_median']:.3f}, SHUF {stab['SHUF']['kstar_lift_median']:.3f}",
              "", "## V0 matching\n", "```", json.dumps(v0, indent=1, default=str)[:4000], "```",
              "", "## V6 concentration\n", "```", json.dumps(v6, indent=1)[:3000], "```",
              "", "## Per-seed detail\n", "```", json.dumps(R["per_seed"], indent=1)[:6000], "```"]
    (ART / "stage1_tables.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"verdict": R["verdict"], "criteria": crit}, indent=1))


if __name__ == "__main__":
    main()
