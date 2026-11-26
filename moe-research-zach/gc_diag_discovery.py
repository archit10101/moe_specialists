#!/usr/bin/env python
"""GC-EDL-v1 diagnostic addendum, sections 1-2 (DESCRIPTIVE; label firewall on; never feeds back into selection).

  gc_diag_discovery.py --variant "" | r1
Fingerprint diagnostics (SEM / GRAD / COMBINED inputs, DISC-FIT and DISC-HOLD): same-paper vs matched random-paper
cosine (paper bootstrap CI), gradient-norm distribution, participation ratio and PCA variance fractions, 128-vs-256
projection neighbourhood overlap, prototype-seed neighbourhood overlap, train vs held-out coherence, per-nuisance
fingerprint variance explained. Cluster diagnostics for every (rep, K). Compact table for the selected clusterings.
Outputs: artifacts/gc-edl-v1/diagnostics/discovery_<variant>.json, selected_table_<rep><suffix>.npz, ledger rows."""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="", choices=["", "r1"])
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    import numpy as np
    import torch  # noqa: F401
    from csr import gc_diag
    from csr.gc_lib import NUIS_NAMES, install_label_firewall, matched_kappa, paper_q, ridge_fit, ridge_pred
    from csr.gc_paths import ART, KS, LAT, PROTO_SEEDS, REPS
    gc_diag.data_manifest_sha()  # resolve before the firewall
    install_label_firewall()
    V = a.variant
    lat = LAT / V if V else LAT
    tag = V or "base"
    od = ART / "diagnostics"
    od.mkdir(parents=True, exist_ok=True)
    out_p = od / f"discovery_{tag}.json"
    fz = np.load(lat / "features.npz")
    zm = np.load(ART / "discovery/inputs_main.npz")
    fpm = np.load(ART / "fingerprints" / ("fp_main.npz" if not V else "fp_main_m256.npz"))
    dec = json.loads((ART / ("discovery_decision.json" if not V else f"discovery_decision_{V}.json")).read_text())
    res_meta = json.loads((lat / "residualizer.json").read_text())
    nv, part = zm["nv"], fz["part_main"]
    rng = np.random.default_rng(11)
    stage = f"discovery{'_' + V if V else ''}"
    art = str(out_p.relative_to(guard.ROOT))
    L = []  # ledger rows

    def lg(name, value, units, n, split, kind="descriptive", candidate=None, seed=None):
        L.append(dict(stage=stage, name=name, value=value, units=units, n=n, split=split, kind=kind,
                      artifact=art, candidate=candidate, seed=seed))

    F = fz["F_main"].astype(np.float64)
    S = fz["S_main"].astype(np.float64)
    U = fz["U_main"].astype(np.float64)
    m = F.shape[-1]
    reps_in = {"SEM": S, "GRAD": F, "COMBINED": np.concatenate([S / math.sqrt(384), F / math.sqrt(m)], -1),
               "RAW_U": U}
    out = {"variant": tag, "projection_dim": m, "fingerprint": {}, "clusters": {}}
    for split_name, pm in (("DISC-FIT", 0), ("DISC-HOLD", 1)):
        ps = np.nonzero((part == pm) & (nv >= 2))[0]
        perm = rng.permutation(ps)
        same_shift = perm == ps
        perm[same_shift] = ps[(np.nonzero(same_shift)[0] + 1) % len(ps)]
        for rn, X in reps_in.items():
            Xn = X / np.maximum(1e-12, np.linalg.norm(X, axis=-1, keepdims=True))
            last = nv[ps] - 1
            same = np.einsum("pd,pd->p", Xn[ps, 0], Xn[ps, last])  # most separated pair
            rand = np.einsum("pd,pd->p", Xn[ps, 0], Xn[perm, nv[perm] - 1])
            gap = same - rand
            B = 300
            bs = np.asarray([gap[rng.integers(0, len(gap), len(gap))].mean() for _ in range(B)])
            rec = {"same_mean": float(same.mean()), "same_q": np.quantile(same, [0.05, 0.25, 0.5, 0.75, 0.95]).tolist(),
                   "random_mean": float(rand.mean()), "gap_mean": float(gap.mean()),
                   "gap_ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
                   "gap_in_random_sd_units": float(gap.mean() / max(1e-12, rand.std())), "n_papers": int(len(ps))}
            out["fingerprint"].setdefault(rn, {})[split_name] = rec
            lg(f"fp_same_minus_random_cos_{rn}", rec["gap_mean"], "cosine", len(ps), split_name)
            lg(f"fp_same_cos_{rn}", rec["same_mean"], "cosine", len(ps), split_name)
            lg(f"fp_random_cos_{rn}", rec["random_mean"], "cosine", len(ps), split_name)
    # gradient norm distribution, effective rank, PCA variance fractions (DISC-HOLD views)
    ii, jj = np.nonzero((np.arange(4)[None, :] < nv[:, None]) & (part[:, None] == 1))
    gn = fpm["gnorm"][ii, jj]
    out["grad_norm_quantiles"] = dict(zip(["q05", "q25", "q50", "q75", "q95"],
                                          np.quantile(gn, [0.05, 0.25, 0.5, 0.75, 0.95]).tolist()))
    lg("grad_norm_median", out["grad_norm_quantiles"]["q50"], "Frobenius norm", len(gn), "DISC-HOLD")
    for rn in ("RAW_U", "GRAD"):
        Z = reps_in[rn][ii, jj]
        Z = Z - Z.mean(0)
        ev = np.linalg.eigvalsh(Z.T @ Z / len(Z))[::-1]
        pr = float(ev.sum() ** 2 / (ev ** 2).sum())
        fr = {str(k): float(ev[:k].sum() / ev.sum()) for k in (1, 4, 8, 16, 32)}
        out["fingerprint"][rn]["participation_ratio"] = pr
        out["fingerprint"][rn]["pca_var_frac"] = fr
        lg(f"participation_ratio_{rn}", pr, "dimensions", len(Z), "DISC-HOLD")
        for k, v in fr.items():
            lg(f"pca_var_frac_top{k}_{rn}", v, "fraction", len(Z), "DISC-HOLD")
    # nuisance: before/after predictability (from residualisation) + single-feature variance explained of u
    out["nuisance_pred_r2_from_u_hold"] = res_meta["nuisance_pred_r2_from_u_hold"]
    out["nuisance_pred_r2_from_residual_hold"] = res_meta["nuisance_pred_r2_from_residual_hold"]
    for k, v in res_meta["nuisance_pred_r2_from_u_hold"].items():
        lg(f"nuisance_heldout_r2_from_raw_fingerprint_{k}", v, "R2", res_meta["n_views_hold"], "DISC-HOLD")
        lg(f"nuisance_heldout_r2_from_residual_fingerprint_{k}", res_meta["nuisance_pred_r2_from_residual_hold"][k],
           "R2", res_meta["n_views_hold"], "DISC-HOLD")
    lg("nuisance_to_fingerprint_r2_all31", res_meta["r2_nuis_to_u_hold"], "R2 (multi-output, held-out)",
       res_meta["n_views_hold"], "DISC-HOLD")
    nu = fz["nuis_main"].astype(np.float64)
    iif, jjf = np.nonzero((np.arange(4)[None, :] < nv[:, None]) & (part[:, None] == 0))
    single = {}
    Uf, Uh = U[iif, jjf], U[ii, jj]
    for c, name in enumerate(NUIS_NAMES):
        xf, xh = nu[iif, jjf, c:c + 1], nu[ii, jj, c:c + 1]
        mu_, sd_ = xf.mean(0), xf.std(0) + 1e-12
        Wb = ridge_fit((xf - mu_) / sd_, Uf, 1.0)
        pred = ridge_pred(Wb, (xh - mu_) / sd_)
        single[name] = float(1 - ((Uh - pred) ** 2).sum() / ((Uh - Uf.mean(0)) ** 2).sum())
    out["single_nuisance_r2_of_fingerprint_hold"] = single
    top = max(single, key=single.get)
    out["top_nuisance"] = {"name": top, "r2": single[top]}
    for k, v in single.items():
        lg(f"fingerprint_var_explained_by_{k}", v, "R2 (held-out, multi-output)", len(Uh), "DISC-HOLD")
    # projection-dimension neighbourhood overlap (128 vs 256), paper-mean functional fingerprints, DISC-HOLD subset
    if V == "r1":
        f128 = np.load(LAT / "features.npz")["F_main"].astype(np.float64)
        sub = rng.choice(np.nonzero((part == 1) & (nv >= 2))[0], 3000, replace=False)

        def pm(Fx):
            w = (np.arange(4)[None, :] < nv[sub][:, None])[..., None]
            x = (Fx[sub] * w).sum(1) / w.sum(1)
            return x / np.maximum(1e-12, np.linalg.norm(x, axis=1, keepdims=True))

        def knn(x, k=10):
            s = x @ x.T
            np.fill_diagonal(s, -np.inf)
            return np.argsort(-s, 1)[:, :k]
        A, Bk = knn(pm(f128)), knn(pm(F))
        ov = float(np.mean([len(set(A[i]) & set(Bk[i])) / 10 for i in range(len(sub))]))
        out["knn10_overlap_128_vs_256"] = ov
        out["knn10_overlap_chance"] = 10 / (len(sub) - 1)
        lg("knn10_overlap_projection_128_vs_256_GRAD", ov, "fraction of 10-NN shared", len(sub), "DISC-HOLD")
    # cluster diagnostics for every (rep, K)
    hold_all = np.nonzero((part == 1) & (nv >= 1))[0]
    semp = S.copy()
    w = (np.arange(4)[None, :] < nv[:, None])[..., None]
    semp = (semp * w).sum(1) / np.maximum(1, w.sum(1))
    semp /= np.maximum(1e-12, np.linalg.norm(semp, axis=1, keepdims=True))
    for key, mt in dec["metrics"].items():
        rep, K = mt["rep"], mt["K"]
        src = LAT if (V and rep == "SEM") else lat
        qs = [paper_q(np.load(src / f"{key}_s{s}" / "view_logits.npz")["main"].astype(np.float64), nv)
              for s in PROTO_SEEDS]
        q = qs[mt["medoid_seed"]][hold_all]
        hard = q.argmax(1)
        share = np.bincount(hard, minlength=K) / len(hard)
        ent = -(q * np.log(np.maximum(1e-12, q))).sum(1) / math.log(K)
        pbar = q.mean(0)
        g = semp[hold_all].mean(0)
        g /= np.linalg.norm(g)
        coh = []
        for k in range(K):
            mk = hard == k
            if mk.sum() < 2:
                continue
            c = semp[hold_all][mk].mean(0)
            c /= np.linalg.norm(c)
            coh.append(float((semp[hold_all][mk] @ c).mean() - (semp[hold_all][mk] @ g).mean()))
        net = __import__("torch").load(src / f"{key}_s{mt['medoid_seed']}" / "net.pt", weights_only=False)
        C = net["state"]["C"]
        Cn = (C / C.norm(dim=1, keepdim=True)).numpy()
        pc = Cn @ Cn.T
        np.fill_diagonal(pc, -1)
        # nearest-neighbour paper overlap across prototype seeds (soft-assignment space)
        sub = hold_all[rng.choice(len(hold_all), 2000, replace=False)]

        def knnq(qq, k=10):
            s = qq[sub] @ qq[sub].T
            np.fill_diagonal(s, -np.inf)
            return np.argsort(-s, 1)[:, :k]
        nn = [knnq(x) for x in qs]
        nno = float(np.mean([len(set(nn[0][i]) & set(nn[1][i])) / 10 for i in range(len(sub))]))
        rec = {"utilization_nonempty": int((share > 0).sum()), "share_min": float(share.min()),
               "share_max": float(share.max()), "mean_norm_entropy": float(ent.mean()),
               "eff_clusters": float(np.exp(-(pbar * np.log(np.maximum(1e-12, pbar))).sum())),
               "frac_confident_maxq_gt_0.5": float((q.max(1) > 0.5).mean()),
               "frac_diffuse_norm_entropy_gt_0.9": float((ent > 0.9).mean()),
               "kappa_view": mt["kappa_view"], "kappa_seed": mt["kappa_seed"], "C_heldout_coherence": mt["C"],
               "LD_heldout_nats_per_token": mt["LD"], "S": mt["S"], "S_se": mt["S_se"],
               "semantic_coherence": float(np.mean(coh)) if coh else None,
               "max_prototype_cosine": float(pc.max()), "near_duplicate_prototypes_cos_gt_0.95": int((pc > 0.95).sum() // 2),
               "knn10_overlap_across_proto_seeds_0v1": nno, "eligible": mt["eligible"]}
        out["clusters"][key] = rec
        for nme, val in rec.items():
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                lg(f"cluster_{nme}", val, "see name", len(hold_all), "DISC-HOLD", candidate=key)
    # compact table for the selected clusterings
    zz = np.load(ART / "discovery/inputs_main.npz")
    for rep, key in dec["chosen"].items():
        if key is None:
            continue
        mt = dec["metrics"][key]
        src = LAT if (V and rep == "SEM") else lat
        vl = np.load(src / f"{key}_s{mt['medoid_seed']}" / "view_logits.npz")["main"].astype(np.float64)
        q = paper_q(vl, nv)
        net = __import__("torch").load(src / f"{key}_s{mt['medoid_seed']}" / "net.pt", weights_only=False)
        tau = 0.1
        ml = (vl * (np.arange(4)[None, :] < nv[:, None])[..., None]).sum(1) / np.maximum(1, nv)[:, None]
        cosd = 1 - ml * tau  # mean cosine distance of the paper's views to each prototype
        ent = -(q * np.log(np.maximum(1e-12, q))).sum(1) / math.log(mt["K"])
        tp = od / f"selected_table_{rep}{'_' + V if V else ''}.npz"
        np.savez_compressed(tp, opaque=zz["opaque"], starts=zz["starts"], nv=nv, q=q.astype(np.float32),
                            entropy=ent.astype(np.float32), proto_cos_dist=cosd.astype(np.float32),
                            scout_ce=fpm["ce"].astype(np.float32), nuisance=fz["nuis_main"].astype(np.float16),
                            nuisance_names=np.asarray(NUIS_NAMES), disc_part=part)
        out.setdefault("selected_tables", {})[rep] = str(tp.relative_to(guard.ROOT))
        del net
    guard.atomic_write_json(out_p, out, overwrite=True)
    for r in L:
        gc_diag.log(**r)
    print(json.dumps({"fingerprint_gap_hold": {k: v["DISC-HOLD"]["gap_mean"] for k, v in out["fingerprint"].items()},
                      "top_nuisance": out["top_nuisance"], "n_ledger_rows": len(L),
                      "knn_128_256": out.get("knn10_overlap_128_vs_256")}, indent=1))


if __name__ == "__main__":
    main()
