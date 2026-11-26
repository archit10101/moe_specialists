#!/usr/bin/env python
"""csr-forecast-counterfactual-utility-v1: DESCRIPTIVE robustness of the Phase-1 decision (cannot change it).

Run after the frozen decision (probes/phase1_T4_s0.json, branch A). CALIB-SEARCH only.
  1. Split-half reliability of region-level utility (first vs second half of each region's targets, token-level
     deltas): how much of u is a stable region-level quantity at all (ceiling for any region-level predictor).
  2. Token-level structure: concentration of |delta| in the top 5 % of tokens; direct (selected-token) share.
  3. Ridge-penalty sensitivity: every preregistered set chose the grid boundary alpha = 1e5; refit with
     alpha in {1e5, 3e5, 1e6, 3e6, 1e7} (grouped CV) for H, H+O32, H+O128, H+Zcur_mean, H+Zcur_cat.
  4. Maximal oracles from the utility model's own processing of the actual future region (features/oracle_model.npz):
     H+O_model_hidden, H+O_model_route (region-mean router distribution: an extreme, routing-aware upper bound).
  fcu_driver.py job robust_phase1 --cpu -- fcu_robust.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--utility", default="T4_s0")
    ap.add_argument("--nboot", type=int, default=1000)
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    import numpy as np
    from fcu_probe import MIN_SEL, load_utility, zfeat
    from csr.fcu_lib import (ART, LAYERS, cluster_bootstrap, prereg_sha, regions, ridge_cv, ridge_fit, ridge_pred,
                             standardize)
    t0 = time.time()
    reg, sp, cache = regions()
    N = len(reg["s"])
    U, _ = load_utility(a.utility, reg)
    Lr, E = len(LAYERS), 32
    tr, te = reg["half"] == 0, reg["half"] == 1
    papers = np.asarray(sp.papers)[reg["gdoc"]]
    res = {"prereg_sha256": prereg_sha(), "note": "descriptive; the preregistered decision is unchanged"}
    # ---- 1/2: token-level deltas
    od = ART / "utility" / a.utility
    man = json.loads((od / "manifest.json").read_text())
    rA = np.zeros((N, Lr, E), np.float32)
    rB = np.zeros((N, Lr, E), np.float32)
    conc, dshare = [], []
    for c in man["chunks"]:
        z = np.load(od / c["file"])
        ri, tk, nv = z["ridx"], z["tokd"].astype(np.float32), z["nvalid"].astype(int)
        for j, r in enumerate(ri):
            n = nv[j]
            h = n // 2
            rA[r] = tk[j, :, :, :h].mean(-1)
            rB[r] = tk[j, :, :, h:n].mean(-1)
            act = z["nsel"][j] >= MIN_SEL
            ab = np.abs(tk[j, :, :, :n])[act]  # (cells, n)
            if len(ab):
                k = max(1, int(0.05 * n))
                srt = np.sort(ab, 1)[:, ::-1]
                conc.append(float((srt[:, :k].sum(1) / np.maximum(ab.sum(1), 1e-12)).mean()))
                ud, u_, ns = z["ud"][j][act], z["u"][j][act], z["nsel"][j][act].astype(float)
                ok = np.abs(u_) > 1e-5
                dshare.append(float(np.median((ud[ok] * ns[ok]) / (u_[ok] * n))))
    act = U["nsel"] >= MIN_SEL
    cm_A = np.nanmean(np.where(act[tr], rA[tr], np.nan), 0)
    cm_B = np.nanmean(np.where(act[tr], rB[tr], np.nan), 0)
    xa, xb = (rA - cm_A)[te][act[te]], (rB - cm_B)[te][act[te]]
    r_half = float(np.corrcoef(xa, xb)[0, 1])
    res["split_half"] = {"r_first_vs_second_half": r_half,
                         "spearman_brown_full_region_reliability": float(2 * r_half / (1 + r_half)),
                         "interpretation": "upper bound on R2 of ANY region-level predictor of u (beyond cell means)"
                                           " is approximately this reliability"}
    res["token_structure"] = {"top5pct_tokens_share_of_abs_delta_mean": float(np.mean(conc)),
                              "direct_selected_token_share_of_u_median": float(np.median(dshare))}
    # ---- 3/4: probes with extended alphas and maximal oracles
    base = np.load(ART / "features" / "base.npz")
    cidx = reg["cidx"]
    Hf = np.concatenate([cache["emb_past"][cidx].astype(np.float32), U["hlast"], U["hmean"]], 1)
    zm, zc, _ = zfeat(cache, cidx)
    sets = {"H": Hf, "H+O32": np.concatenate([Hf, base["o32"].astype(np.float32)], 1),
            "H+O128": np.concatenate([Hf, base["o128"].astype(np.float32)], 1),
            "H+Zcur_mean": np.concatenate([Hf, zm], 1), "H+Zcur_cat": np.concatenate([Hf, zc], 1)}
    om_p = ART / "features" / "oracle_model.npz"
    if om_p.exists():
        om = np.load(om_p)
        sets["H+Omodel_hidden"] = np.concatenate([Hf, om["o_model_hidden"]], 1)
        sets["H+Omodel_route"] = np.concatenate([Hf, om["o_model_route"]], 1)
        sets["Omodel_route_only"] = om["o_model_route"]
    u = U["u"].reshape(N, Lr * E).astype(np.float64)
    actf = act.reshape(N, Lr * E)
    mu, sd = u[tr].mean(0), u[tr].std(0) + 1e-9
    Y = (u - mu) / sd
    ute, ate = u[te], actf[te]
    sst_i = (((ute - mu) ** 2) * ate).sum(1)
    ur, ar = ute.reshape(-1, Lr, E), ate.reshape(-1, Lr, E)
    lay_ok = ar.sum(2) >= 2
    nl = lay_ok.sum(1)
    orc = (np.where(ar, ur, 1e9).min(2) * lay_ok).sum(1)
    per, out = {}, {}
    for name, X in sets.items():
        Xtr, Xte = standardize(X[tr], X[te])
        alpha, cvs = ridge_cv(Xtr, Y[tr], papers[tr], alphas=(1e4, 1e5, 3e5, 1e6, 3e6, 1e7), mask=actf[tr])
        W = ridge_fit(Xtr, Y[tr], alpha)
        uh = ridge_pred(W, Xte) * sd + mu
        sse = (((ute - uh) ** 2) * ate).sum(1)
        pick = np.where(ar, uh.reshape(-1, Lr, E), 1e9).argmin(2)
        real = (np.take_along_axis(ur, pick[..., None], 2)[..., 0] * lay_ok).sum(1)
        per[name] = (sse, real)
        out[name] = {"alpha": alpha, "R2": float(1 - sse.sum() / sst_i.sum()),
                     "M5_realized": float(real.sum() / nl.sum())}
    gte = papers[te]
    for name in sets:
        if name == "H" or name.endswith("_only"):
            continue
        sa, ra = per["H"]
        sb, rb = per[name]

        def f(ix):
            return np.array([(sa[ix].sum() - sb[ix].sum()) / sst_i[ix].sum(),
                             (ra[ix].sum() - rb[ix].sum()) / max(1e-12, ra[ix].sum() - orc[ix].sum())])
        pt = f(np.arange(len(sa)))
        bs = cluster_bootstrap(f, gte, a.nboot)
        out[name].update({"dR2_vs_H": float(pt[0]), "gap_closure_vs_H": float(pt[1]),
                          "ci95_dR2": [float(np.quantile(bs[:, 0], q)) for q in (0.025, 0.975)],
                          "ci95_gap": [float(np.quantile(bs[:, 1], q)) for q in (0.025, 0.975)]})
    res["probes_extended"] = out
    res["M5_oracle_min"] = float(orc.sum() / nl.sum())
    res["seconds"] = time.time() - t0
    res["time"] = guard.now()
    p = ART / "probes" / f"robust_{a.utility}.json"
    guard.atomic_write_json(p, res, overwrite=True)
    with open(ART / "artifact_hashes.tsv", "a") as fh:
        fh.write(f"{guard.now()}\t{p.relative_to(guard.ROOT)}\t{guard.sha256_file(p)}\t{p.stat().st_size}\n")
    with open(ART / "compute_ledger.tsv", "a") as fh:
        fh.write(f"{guard.now()}\trobust_{a.utility}\tcpu_probe\t{res['seconds']:.1f}\tCOMPLETE\tCPU\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
