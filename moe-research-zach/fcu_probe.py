#!/usr/bin/env python
"""csr-forecast-counterfactual-utility-v1: Phase-1 information tests and the mandatory decision tree (prereg 2.4-2.6).

CPU only. Probe-train = CALIB-SEARCH paper-half A, probe-test = paper-half B (paper-disjoint). Every feature set gets
the same probe family (closed-form ridge / multinomial logistic, paper-grouped 5-fold CV for the penalty).
Paired cluster bootstrap over half-B papers (2000 resamples) for every reported difference.

  fcu_driver.py job probe_phase1 --cpu -- fcu_probe.py --utility T4_s0 [--strong]
Writes artifacts/forecast-counterfactual-utility-v1/probes/phase1_<utility>[_strong].json (+ utility summary).
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402

MIN_SEL = 8
NBOOT = 2000


def fast_auroc(score, label):
    import numpy as np
    label = label.astype(bool)
    npos, nneg = int(label.sum()), int((~label).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    _, inv, cnt = np.unique(score, return_inverse=True, return_counts=True)
    cum = np.cumsum(cnt)
    ranks = (cum - (cnt - 1) / 2.0)[inv]
    return float((ranks[label].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def load_utility(name, reg):
    import numpy as np
    from csr.fcu_lib import ART
    od = ART / "utility" / name
    man = json.loads((od / "manifest.json").read_text())
    if not man.get("complete"):
        raise guard.GuardError(f"utility {name} incomplete")
    N = len(reg["s"])
    out = {}
    for c in man["chunks"]:
        p = od / c["file"]
        if guard.sha256_file(p) != c["sha256"]:
            raise guard.GuardError(f"hash mismatch {p}")
        z = np.load(p)
        ri = z["ridx"]
        for k in ("u", "ud", "nsel", "pe", "ce_null", "ce_full", "hlast", "hmean"):
            if k not in out:
                out[k] = np.zeros((N,) + z[k].shape[1:], z[k].dtype)
                out[k + "_have"] = np.zeros(N, bool)
            out[k][ri] = z[k]
            out[k + "_have"][ri] = True
    return out, man


def zfeat(cache, cidx, which="emb"):
    """Z-current features from the fle cache rows: mean (A1 inputs) and concatenated (no aggregation)."""
    import numpy as np
    from csr.fle_lib import LN_V
    e = cache[which][cidx].astype(np.float32)  # (N, D, 384)
    p = cache["ptok"][cidx].astype(np.float32) / 255.0
    en = cache["ent"][cidx].astype(np.float32) / 255.0 * LN_V
    t1 = cache["top1"][cidx].astype(np.float32) / 255.0
    Hh = 16 if which == "emb16" else 32
    p, en, t1 = p[..., :Hh], en[..., :Hh], t1[..., :Hh]
    en_ = e / (np.linalg.norm(e, axis=-1, keepdims=True) + 1e-8)
    D = e.shape[1]
    cos = np.einsum("nid,njd->nij", en_, en_)
    iu = np.triu_indices(D, 1)
    disagree = 1.0 - cos[:, iu[0], iu[1]].mean(1)
    ment = en.mean(-1)  # (N, D)
    stats5 = np.stack([disagree, ment.mean(1), np.log(p + 1e-6).mean((1, 2)), t1.mean((1, 2)), ment.std(1)], 1)
    mean = en_.mean(1)
    mean = mean / (np.linalg.norm(mean, axis=-1, keepdims=True) + 1e-8)
    per = np.stack([np.log(p + 1e-6).mean(-1), ment, t1.mean(-1)], -1).reshape(len(cidx), -1)
    return np.concatenate([mean, stats5], 1), np.concatenate([en_.reshape(len(cidx), -1), per], 1), \
        {"disagreement": disagree, "first_tok": cache["tok"][cidx][:, :, 0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--utility", default="T4_s0")
    ap.add_argument("--strong", action="store_true", help="include Z-strong feature sets (features/zstrong.npz)")
    ap.add_argument("--variants", default="", help="comma list of branch-B variants to add (B1,B2,B3,B4)")
    ap.add_argument("--nboot", type=int, default=NBOOT)
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    import numpy as np
    from csr.fcu_lib import (ART, LAYERS, cluster_bootstrap, logreg_cv, logreg_fit, logreg_logprobs, prereg_sha,
                             regions, ridge_cv, ridge_fit, ridge_pred, standardize)
    t0 = time.time()
    reg, sp, cache = regions()
    N = len(reg["s"])
    U, uman = load_utility(a.utility, reg)
    assert U["u_have"].all(), "utility missing for some regions"
    base = np.load(ART / "features" / "base.npz")
    assert (base["ridx"] == np.arange(N)).all()
    cidx = reg["cidx"]
    papers = np.asarray(sp.papers)[reg["gdoc"]]
    tr, te = reg["half"] == 0, reg["half"] == 1
    # ---------------- feature sets
    Hf = np.concatenate([cache["emb_past"][cidx].astype(np.float32), U["hlast"], U["hmean"]], 1)
    zm, zc, zinfo = zfeat(cache, cidx)
    feats = {"P2": base["p2"].astype(np.float32), "Zcur_mean": zm, "Zcur_cat": zc,
             "O32": base["o32"].astype(np.float32), "O128": base["o128"].astype(np.float32)}
    if a.strong:
        zs = np.load(ART / "features" / "zstrong.npz")
        assert (zs["ridx"] == np.arange(N)).all()
        sm, sc, _ = zfeat({k: zs[k] for k in ("emb", "ptok", "ent", "top1", "tok")}, np.arange(N))
        feats["Zstr_mean"], feats["Zstr_cat"] = sm, sc
    for v in [x for x in a.variants.split(",") if x]:
        if v == "B1":
            feats["B1_cur_mean16"] = zfeat(cache, cidx, "emb16")[0]
    rng = np.random.default_rng(2026)

    def shuf(X):
        Y = X.copy()
        for msk in (tr, te):
            ii = np.nonzero(msk)[0]
            Y[ii] = X[rng.permutation(ii)]
        return Y
    sets = {"H": Hf}
    for k, X in feats.items():
        sets[f"H+{k}"] = np.concatenate([Hf, X], 1)
        if k != "P2":
            sets[f"H+{k}-shuf"] = np.concatenate([Hf, shuf(X)], 1)
    # ---------------- utility targets
    Lr, E = len(LAYERS), 32
    u = U["u"].reshape(N, Lr * E).astype(np.float64)
    act = (U["nsel"].reshape(N, Lr * E) >= MIN_SEL)
    mu, sd = u[tr].mean(0), u[tr].std(0) + 1e-9
    Y = (u - mu) / sd
    gtr = papers[tr]
    res = {"prereg_sha256": prereg_sha(), "utility": a.utility, "utility_manifest_complete": uman["complete"],
           "n_train": int(tr.sum()), "n_test": int(te.sum()), "min_sel": MIN_SEL, "sets": {}}
    # utility summary (descriptive)
    ua = np.where(act, u, np.nan)
    res["utility_summary"] = {
        "active_frac": float(act.mean()), "mean_u_active": float(np.nanmean(ua)),
        "mean_abs_u_active": float(np.nanmean(np.abs(ua))), "frac_pos_active": float(np.nanmean(ua > 0)),
        "per_layer_mean_u": [float(np.nanmean(ua.reshape(N, Lr, E)[:, l])) for l in range(Lr)],
        "per_layer_frac_pos": [float(np.nanmean(ua.reshape(N, Lr, E)[:, l] > 0)) for l in range(Lr)],
        "within_region_sd_mean": float(np.nanmean(np.nanstd(ua.reshape(N, Lr, E), axis=2))),
        "between_region_sd_of_cellmean_mean": float(np.nanmean(np.nanstd(ua, axis=0))),
        "ce_full_mean": float(U["ce_full"].mean()), "ce_null_minus_full_max": float(np.abs(U["ce_null"] -
                                                                                       U["ce_full"]).max())}
    # ---------------- per-region decomposable metrics for every set
    ute, ate, mute = u[te], act[te], mu
    sst_i = (((ute - mute) ** 2) * ate).sum(1)
    ur = ute.reshape(-1, Lr, E)
    ar = ate.reshape(-1, Lr, E)
    big = 1e9
    or_min = np.where(ar, ur, big).min(2)  # (n, Lr)
    rnd = np.where(ar, ur, 0).sum(2) / np.maximum(ar.sum(2), 1)
    best_true = np.where(ar, ur, -big).argmax(2)
    lay_ok = ar.sum(2) >= 2
    per = {}
    preds = {}
    for name, X in sets.items():
        Xtr, Xte = standardize(X[tr], X[te])
        alpha, cvs = ridge_cv(Xtr, Y[tr], gtr, mask=act[tr])
        W = ridge_fit(Xtr, Y[tr], alpha)
        uh = ridge_pred(W, Xte) * sd + mu
        preds[name] = uh
        sse_i = (((ute - uh) ** 2) * ate).sum(1)
        uhr = uh.reshape(-1, Lr, E)
        pick_min = np.where(ar, uhr, big).argmin(2)
        realized = np.take_along_axis(ur, pick_min[..., None], 2)[..., 0]
        pick_max = np.where(ar, uhr, -big).argmax(2)
        best_ok = (pick_max == best_true) & lay_ok
        x_, y_ = (ute - mute)[ate], (uh - mute)[ate]
        per[name] = {"sse": sse_i, "realized": (realized * lay_ok).sum(1), "best": best_ok.sum(1)}
        res["sets"][name] = {
            "dim": int(X.shape[1]), "alpha": alpha,
            "M1_R2": float(1 - sse_i.sum() / sst_i.sum()),
            "M2_pearson": float(np.corrcoef(x_, y_)[0, 1]),
            "M3_auroc_pos": fast_auroc(uh[ate], ute[ate] > 0),
            "M4_best_acc": float(best_ok.sum() / lay_ok.sum()),
            "M5_realized_dce": float((realized * lay_ok).sum() / lay_ok.sum())}
    nl = lay_ok.sum(1)
    res["M5_bounds"] = {"oracle_min": float((or_min * lay_ok).sum() / nl.sum()),
                        "random_mean": float((rnd * lay_ok).sum() / nl.sum()),
                        "M4_chance": float((lay_ok / np.maximum(ar.sum(2), 1)).sum() / nl.sum())}
    orc_i = (or_min * lay_ok).sum(1)
    gte = papers[te]

    def boot_diff(A, B):
        """Paired cluster-bootstrap of B minus A improvements: dR2 = (SSE_A - SSE_B)/SST, dM5 = (R_A - R_B)/n,
        gap = (R_A - R_B)/(R_A - R_oracle)."""
        sa, sb = per[A]["sse"], per[B]["sse"]
        ra, rb = per[A]["realized"], per[B]["realized"]

        def f(ix):
            return np.array([(sa[ix].sum() - sb[ix].sum()) / sst_i[ix].sum(),
                             (ra[ix].sum() - rb[ix].sum()) / nl[ix].sum(),
                             (ra[ix].sum() - rb[ix].sum()) / max(1e-12, ra[ix].sum() - orc_i[ix].sum())])
        pt = f(np.arange(len(sa)))
        bs = cluster_bootstrap(f, gte, a.nboot)
        q = lambda c: [float(np.quantile(bs[:, j], c)) for j in range(3)]  # noqa: E731
        return {"dR2": float(pt[0]), "dM5": float(pt[1]), "gap_closure": float(pt[2]),
                "ci95_lo": q(0.025), "ci95_hi": q(0.975), "ci99_lo": q(0.005), "ci99_hi": q(0.995)}
    cmp = {}
    for name in sets:
        if name == "H":
            continue
        cmp[f"{name} vs H"] = boot_diff("H", name)
        if not name.endswith("-shuf") and f"{name}-shuf" in sets:
            cmp[f"{name} vs {name}-shuf"] = boot_diff(f"{name}-shuf", name)
        if name.startswith("H+Z") or name.startswith("H+B"):
            if not name.endswith("-shuf"):
                cmp[f"{name} vs H+P2"] = boot_diff("H+P2", name)
    res["comparisons"] = cmp
    # ---------------- label probes (domain 5-way, e5 cluster 16-way)
    w = np.load(guard.ROOT / "artifacts/latent-forecast-router-distill-v1/labels/windows_calib.npz")
    key = {(int(d), int(s0)): int(c) for d, s0, c in zip(w["doc"], w["start"], w["cluster"])}
    e5 = np.asarray([key.get((int(g), int(s0)), -1) for g, s0 in zip(reg["gdoc"], reg["s"])])
    res["e5_label_coverage"] = float((e5 >= 0).mean())
    lab = {}
    for lname, yv, C in (("domain", reg["domain"], 5), ("e5", e5, 16)):
        ok = yv >= 0
        lab[lname] = {}
        nll_i = {}
        for name, X in sets.items():
            if name.endswith("-shuf") and not (name.startswith("H+O") or name.startswith("H+Zcur")):
                continue
            Xtr, Xte = standardize(X[tr & ok], X[te & ok])
            wd, _ = logreg_cv(Xtr, yv[tr & ok], C, papers[tr & ok])
            lp = logreg_logprobs(logreg_fit(Xtr, yv[tr & ok], C, wd), Xte)
            yt = yv[te & ok]
            nll = -lp[np.arange(len(yt)), yt]
            nll_i[name] = nll
            lab[lname][name] = {"logloss": float(nll.mean()), "acc": float((lp.argmax(1) == yt).mean()), "wd": wd}
        g_ok = papers[te & ok]
        for name in nll_i:
            if name == "H":
                continue
            d = nll_i["H"] - nll_i[name]
            bs = cluster_bootstrap(lambda ix: d[ix].mean(), g_ok, a.nboot)
            lab[lname][name]["dlogloss_vs_H"] = float(d.mean())
            lab[lname][name]["ci95"] = [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]
    res["label_probes"] = lab
    # ---------------- forecast diagnostics (M7, descriptive)
    diag = {"disagreement_mean": float(zinfo["disagreement"].mean())}
    ft = zinfo["first_tok"]
    diag["first_token_agreement"] = float(np.mean([(ft[:, i] == ft[:, j]).mean() for i in range(4)
                                                   for j in range(i + 1, 4)]))
    e = cache["emb"][cidx].astype(np.float32)
    o32 = base["o32"].astype(np.float32)
    ep = cache["emb_past"][cidx].astype(np.float32)
    diag["cos_draft_to_actual32"] = [float((e[:, d] * o32).sum(1).mean()) for d in range(4)]
    diag["cos_draftmean_to_actual32"] = float(((e.mean(1) / np.linalg.norm(e.mean(1), axis=1, keepdims=True))
                                               * o32).sum(1).mean())
    diag["cos_past192_to_actual32"] = float((ep * o32).sum(1).mean())
    ec = e - e.mean(0, keepdims=True)
    diag["inter_draft_corr"] = float(np.mean([np.mean([np.corrcoef(ec[:, i, k], ec[:, j, k])[0, 1]
                                                       for k in range(0, 384, 16)])
                                              for i in range(4) for j in range(i + 1, 4)]))
    for ens in ("current", "strong"):
        p = ART / "features" / f"quality_{ens}.npz"
        if p.exists():
            q = np.load(p)
            ce = q["ce"].astype(np.float64)
            t1p, t1c = q["top1p"].astype(np.float64), q["top1c"]
            ece = []
            for d in range(ce.shape[1]):
                bins = np.minimum((t1p[:, d].ravel() * 10).astype(int), 9)
                c_ = t1c[:, d].ravel()
                pp = t1p[:, d].ravel()
                ece.append(float(sum(abs(c_[bins == b].mean() - pp[bins == b].mean()) * (bins == b).mean()
                                     for b in range(10) if (bins == b).any())))
            diag[f"quality_{ens}"] = {"members": [str(x) for x in q["members"]],
                                      "ce_next32_per_member": [float(x) for x in ce.mean((0, 2))],
                                      "ce_next32_mean": float(ce.mean()), "mixture_nll": float(-q["mix_ll"].mean()),
                                      "top1_ece": ece}
    if (ART / "features" / "quality_strong.npz").exists() and (ART / "features" / "quality_current.npz").exists():
        qc, qs = np.load(ART / "features" / "quality_current.npz"), np.load(ART / "features" / "quality_strong.npz")
        trn = reg["half"] == 0
        dce = qc["ce"].mean((1, 2)) - qs["ce"].mean((1, 2))
        dmx = (-qc["mix_ll"].mean(1)) - (-qs["mix_ll"].mean(1))
        out_s = {}
        for nm, d in (("ce_improvement", dce), ("mixture_nll_improvement", dmx)):
            bs = cluster_bootstrap(lambda ix: d[trn][ix].mean(), papers[trn], a.nboot)
            out_s[nm] = {"mean_halfA": float(d[trn].mean()), "ci95": [float(np.quantile(bs, 0.025)),
                                                                     float(np.quantile(bs, 0.975))]}
        out_s["demonstrably_stronger"] = bool(out_s["ce_improvement"]["ci95"][0] > 0 and
                                              out_s["mixture_nll_improvement"]["ci95"][0] > 0)
        diag["strong_vs_current"] = out_s
    res["forecast_diagnostics"] = diag
    # ---------------- decision tree (prereg 2.6)
    def material(o):
        c = cmp[f"H+{o} vs H"]
        cs = cmp[f"H+{o} vs H+{o}-shuf"]
        a_ok = c["dR2"] >= 0.02 and c["ci95_lo"][0] > 0 and cs["ci95_lo"][0] > 0
        b_ok = c["gap_closure"] >= 0.10 and c["ci95_lo"][1] > 0 and cs["ci95_lo"][1] > 0
        return {"a_R2": bool(a_ok), "b_routing": bool(b_ok), "material": bool(a_ok or b_ok)}
    dec = {"oracle": {o: material(o) for o in ("O32", "O128")}}
    dec["oracle_headroom"] = any(v["material"] for v in dec["oracle"].values())
    zs_ = [z for z in ("Zcur_mean", "Zcur_cat", "Zstr_mean", "Zstr_cat") if f"H+{z}" in sets]
    zs_ += [k for k in feats if k.startswith("B")]
    zres = {}
    for z in zs_:
        c, cs, cp = cmp[f"H+{z} vs H"], cmp[f"H+{z} vs H+{z}-shuf"], cmp[f"H+{z} vs H+P2"]
        m1 = c["ci99_lo"][0] > 0 and cs["ci99_lo"][0] > 0 and cp["ci99_lo"][0] > 0
        m5 = c["ci99_lo"][1] > 0 and cs["ci99_lo"][1] > 0 and cp["ci99_lo"][1] > 0
        zres[z] = {"M1_sig": bool(m1), "M5_sig": bool(m5), "significant": bool(m1 or m5)}
    dec["draft_information"] = zres
    dec["any_Z_significant"] = any(v["significant"] for v in zres.values())
    dec["direct_routing_value"] = any(v["M5_sig"] for v in zres.values())
    if not dec["oracle_headroom"]:
        dec["branch"] = "A"
    elif not dec["any_Z_significant"]:
        dec["branch"] = "B"
    else:
        dec["branch"] = "C" + ("+D-eligible" if dec["direct_routing_value"] else "")
    res["decision"] = dec
    res["seconds"] = time.time() - t0
    res["time"] = guard.now()
    tag = a.utility + ("_strong" if a.strong else "") + (("_" + a.variants.replace(",", "")) if a.variants else "")
    p = ART / "probes" / f"phase1_{tag}.json"
    guard.atomic_write_json(p, res, overwrite=True)
    with open(ART / "artifact_hashes.tsv", "a") as fh:
        fh.write(f"{guard.now()}\t{p.relative_to(guard.ROOT)}\t{guard.sha256_file(p)}\t{p.stat().st_size}\n")
    np.savez(ART / "probes" / f"utility_matrix_{a.utility}.npz",
             mean_u=np.nanmean(ua.reshape(N, Lr, E), 0), frac_pos=np.nanmean(ua.reshape(N, Lr, E) > 0, 0),
             active=act.reshape(N, Lr, E).mean(0))
    with open(ART / "compute_ledger.tsv", "a") as fh:
        fh.write(f"{guard.now()}\tprobe_{tag}\tcpu_probe\t{res['seconds']:.1f}\tCOMPLETE\tCPU\n")
    print(json.dumps({"decision": dec, "sets": res["sets"], "M5_bounds": res["M5_bounds"]}, indent=1))


if __name__ == "__main__":
    main()
