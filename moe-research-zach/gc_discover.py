#!/usr/bin/env python
"""csr-gradient-consistent-latent-edl-v1 Stages 1b-2 (prereg sections 3-5), CPU, label firewall on.

  gc_discover.py residualize            nuisance features, DISC-FIT ridge residualisation (cross-fitted), standardised
                                        functional fingerprints f, semantic features; before/after diagnostics
  gc_discover.py fit --rep R --k K --seed S
                                        balanced soft prototypes on DISC-FIT; view logits for main/calib papers
  gc_discover.py select                 label-free DISC-HOLD metrics, composite score, K per representation, validity,
                                        frozen assignment tables (medoid seed) incl. the shuffled-paper control
Inputs: artifacts/gc-edl-v1/discovery/inputs_*.npz and fingerprints/fp_*.npz only.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402

SPLITS = ("main", "calib_search", "calib_sel")
ALPHAS = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)


VARIANT = {"": {"suffix": "", "lat": ""}, "r1": {"suffix": "_m256", "lat": "r1"}}
VAR = ""


def latdir():
    from csr.gc_paths import LAT
    return LAT / VARIANT[VAR]["lat"] if VAR else LAT


def load_split(s):
    import numpy as np
    from csr.gc_paths import DISC, FP
    z = dict(np.load(DISC / f"inputs_{s}.npz"))
    f = dict(np.load(FP / f"fp_{s}{VARIANT[VAR]['suffix']}.npz"))
    assert (z["opaque"] == f["opaque"]).all()
    return z, f


def residualize():
    import numpy as np
    from csr.gc_lib import (NUIS_NAMES, disc_part, fold_of, grouped_cv_alpha, r2, ridge_fit, ridge_pred,
                            token_features)
    LAT = latdir()
    LAT.mkdir(parents=True, exist_ok=True)
    out_p = LAT / "features.npz"
    if out_p.exists():
        print("features exist")
        return
    D = {s: load_split(s) for s in SPLITS}
    zm = D["main"][0]
    part = np.asarray([disc_part(str(o)) for o in zm["opaque"]])
    fold = np.asarray([fold_of(str(o)) for o in zm["opaque"]])
    # DISC-FIT unigram (views only)
    V = 50304
    cnt = np.zeros(V, np.float64)
    for i in np.nonzero(part == 0)[0]:
        for j in range(int(zm["nv"][i])):
            cnt += np.bincount(zm["views"][i, j].astype(np.int64), minlength=V)
    logp = np.log((cnt + 1.0) / (cnt.sum() + V))
    top = np.zeros(V, bool)
    top[np.argsort(-cnt)[:100]] = True

    def nuis(z, f):
        P = len(z["nv"])
        out = np.zeros((P, 4, len(NUIS_NAMES)), np.float64)
        ii, jj = np.nonzero(np.arange(4)[None, :] < z["nv"][:, None])
        tf = token_features(z["views"][ii, jj], logp, top)
        rel = (z["starts"][ii, jj] + 128) / z["paper_len"][ii]
        cols = [f["ce"][ii, jj], np.log(np.maximum(1e-30, f["gnorm"][ii, jj])), np.log(z["paper_len"][ii]), rel,
                np.log(z["n_seg"][ii])]
        out[ii, jj] = np.concatenate([np.stack(cols, 1), tf, f["textf"][ii, jj]], 1)
        return out, ii, jj

    N, IJ = {}, {}
    for s in SPLITS:
        N[s], ii, jj = nuis(*D[s])
        IJ[s] = (ii, jj)
    ii, jj = IJ["main"]
    fitv = part[ii] == 0
    holdv = ~fitv
    Xn = N["main"][ii, jj]
    nm, nsd = Xn[fitv].mean(0), Xn[fitv].std(0) + 1e-9
    g = D["main"][1]["g"][ii, jj]
    U = g / np.linalg.norm(g, axis=1, keepdims=True)
    Xs_ = (Xn - nm) / nsd
    alpha, cvs = grouped_cv_alpha(Xs_[fitv], U[fitv], fold[ii][fitv], ALPHAS)
    full = ridge_fit(Xs_[fitv], U[fitv], alpha)
    R = U - ridge_pred(full, Xs_)
    fv_idx = np.nonzero(fitv)[0]
    for k in range(5):  # cross-fitted residuals on DISC-FIT
        trm = fold[ii][fv_idx] != k
        Wk = ridge_fit(Xs_[fv_idx[trm]], U[fv_idx[trm]], alpha)
        te = fv_idx[~trm]
        R[te] = U[te] - ridge_pred(Wk, Xs_[te])
    rm, rsd = R[fitv].mean(0), R[fitv].std(0) + 1e-12
    sem = D["main"][1]["sem"][ii, jj].astype(np.float64)
    sm, ssd = sem[fitv].mean(0), sem[fitv].std(0) + 1e-9
    rep = {"nuisance_names": list(NUIS_NAMES), "alpha": alpha, "cv_r2_by_alpha": {str(k): v for k, v in cvs.items()},
           "n_views_fit": int(fitv.sum()), "n_views_hold": int(holdv.sum())}
    rep["r2_nuis_to_u_hold"] = float(1 - ((U[holdv] - ridge_pred(full, Xs_[holdv])) ** 2).sum()
                                     / ((U[holdv] - U[fitv].mean(0)) ** 2).sum())

    # before/after: predict each nuisance variable from u vs from residual (fit on DISC-FIT, R2 on DISC-HOLD)
    def pred_r2(Z):
        Zm, Zs = Z[fitv].mean(0), Z[fitv].std(0) + 1e-12
        Zz = (Z - Zm) / Zs
        a_, _ = grouped_cv_alpha(Zz[fitv], Xs_[fitv], fold[ii][fitv], ALPHAS)
        W = ridge_fit(Zz[fitv], Xs_[fitv], a_)
        return r2(Xs_[holdv], ridge_pred(W, Zz[holdv]), Xs_[fitv].mean(0)).tolist()
    rep["nuisance_pred_r2_from_u_hold"] = dict(zip(NUIS_NAMES, pred_r2(U)))
    rep["nuisance_pred_r2_from_residual_hold"] = dict(zip(NUIS_NAMES, pred_r2(R)))

    feats = {}
    for s in SPLITS:
        z, f = D[s]
        ii_, jj_ = IJ[s]
        P = len(z["nv"])
        Fst = np.zeros((P, 4, R.shape[1]), np.float32)
        Sst = np.zeros((P, 4, 384), np.float32)
        gg = f["g"][ii_, jj_]
        Uu = gg / np.linalg.norm(gg, axis=1, keepdims=True)
        Rr = R if s == "main" else Uu - ridge_pred(full, (N[s][ii_, jj_] - nm) / nsd)
        Fst[ii_, jj_] = ((Rr - rm) / rsd).astype(np.float32)
        Sst[ii_, jj_] = ((f["sem"][ii_, jj_] - sm) / ssd).astype(np.float32)
        Uf = np.zeros((P, 4, R.shape[1]), np.float32)
        Uf[ii_, jj_] = Uu
        feats[s] = (Fst, Sst, Uf)
    # stability diagnostics on DISC-HOLD: same-paper vs different-paper cosine
    hold_p = np.nonzero((part == 1) & (zm["nv"] >= 2))[0]
    rng = np.random.default_rng(7)

    def cos_gap(A):
        An = A / np.maximum(1e-12, np.linalg.norm(A, axis=-1, keepdims=True))
        same = float(np.mean([An[p, 0] @ An[p, 1] for p in hold_p]))
        perm = rng.permutation(hold_p)
        diff = float(np.mean([An[p, 0] @ An[q, 1] for p, q in zip(hold_p, perm) if p != q]))
        return {"same_paper": same, "different_paper": diff, "gap": same - diff}
    Fm, Sm, Um = feats["main"]
    rep["cosine_raw_direction_u"] = cos_gap(Um)
    rep["cosine_functional_f"] = cos_gap(Fm)
    rep["cosine_semantic"] = cos_gap(Sm)
    np.savez(out_p, **{f"{k}_{s}": v for s in SPLITS for k, v in zip(("F", "S", "U"), feats[s])},
             nuis_main=N["main"].astype(np.float32), part_main=part, logp_uni=logp, top100=top)
    guard.atomic_write_json(LAT / "residualizer.json", {**rep, "nuis_mean": nm.tolist(), "nuis_sd": nsd.tolist(),
                                                        "res_mean": rm.tolist(), "res_sd": rsd.tolist(),
                                                        "sem_mean": sm.tolist(), "sem_sd": ssd.tolist(),
                                                        "W": full[0].tolist(), "b": full[1].tolist(),
                                                        "time": guard.now()}, mode=0o400)
    np.savez(LAT / "residualizer_params.npz", W=full[0], b=full[1], nm=nm, nsd=nsd, rm=rm, rsd=rsd, sm=sm, ssd=ssd,
             logp=logp, top=top)
    with open(guard.ROOT / "artifacts/gc-edl-v1/artifact_hashes.tsv", "a") as fh:
        for p in (out_p, LAT / "residualizer.json", LAT / "residualizer_params.npz"):
            fh.write(f"{guard.now()}\t{p.relative_to(guard.ROOT)}\t{guard.sha256_file(p)}\t{p.stat().st_size}\n")
    print(json.dumps({k: v for k, v in rep.items() if "pred_r2" not in k}, indent=1))


def rep_inputs(rep, F_, S_):
    import numpy as np
    if rep == "SEM":
        return S_
    if rep == "GRAD":
        return F_
    return np.concatenate([S_ / math.sqrt(S_.shape[-1]), F_ / math.sqrt(F_.shape[-1])], -1).astype(np.float32)


def fit(rep, K, seed):
    import numpy as np
    import torch
    from csr.gc_lib import fit_prototypes, view_logits
    LAT = latdir()
    od = LAT / f"{rep}_K{K}_s{seed}"
    if (od / "done.json").exists():
        print("exists")
        return
    torch.set_num_threads(4)
    fz = np.load(LAT / "features.npz")
    zm = np.load(guard.ROOT / "artifacts/gc-edl-v1/discovery/inputs_main.npz")
    nv = zm["nv"]
    part = fz["part_main"]
    X = rep_inputs(rep, fz["F_main"], fz["S_main"])
    Fh = fz["F_main"] / np.maximum(1e-12, np.linalg.norm(fz["F_main"], axis=-1, keepdims=True))
    tr = np.nonzero((part == 0) & (nv >= 2))[0]
    lam = 0.0 if rep == "SEM" else 2.0
    t0 = time.time()
    net, hist = fit_prototypes(X[tr], nv[tr], Fh[tr].astype(np.float32), K, lam, seed)
    od.mkdir(parents=True, exist_ok=True)
    out = {"main": view_logits(net, X)}
    for s in ("calib_search", "calib_sel"):
        out[s] = view_logits(net, rep_inputs(rep, fz[f"F_{s}"], fz[f"S_{s}"]))
    np.savez(od / "view_logits.npz", **{k: v.astype(np.float32) for k, v in out.items()})
    torch.save({"state": net.state_dict(), "mu": net.mu, "rep": rep, "K": K, "seed": seed, "lam_coh": lam},
               od / "net.pt")
    guard.atomic_write_json(od / "done.json", {"hist": hist, "seconds": time.time() - t0, "n_train_papers": len(tr),
                                               "lam_coh": lam, "time": guard.now(),
                                               "view_logits_sha256": guard.sha256_file(od / "view_logits.npz")})
    print(json.dumps({"rep": rep, "K": K, "seed": seed, "s": round(time.time() - t0, 1), "last": hist[-1]}))


# ----------------------------------------------------------------------------------------------------- selection


def softmax(x):
    import numpy as np
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def select(nboot=500):
    import numpy as np
    from csr.gc_lib import cohen_kappa, grouped_cv_alpha, matched_kappa, paper_q, ridge_fit, ridge_pred
    from csr.gc_paths import ART, KS, LAT as LAT0, PROTO_SEEDS, REPS
    LAT = latdir()
    dec_p = ART / ("discovery_decision.json" if not VAR else f"discovery_decision_{VAR}.json")
    if dec_p.exists():
        print("decision exists")
        return
    fz = np.load(LAT / "features.npz")
    zm = np.load(guard.ROOT / "artifacts/gc-edl-v1/discovery/inputs_main.npz")
    nv, part = zm["nv"], fz["part_main"]
    Fm = fz["F_main"].astype(np.float64)
    Fh = Fm / np.maximum(1e-12, np.linalg.norm(Fm, axis=-1, keepdims=True))
    nuis = fz["nuis_main"]
    logp_g = fz["logp_uni"]
    V = 50304
    hold = np.nonzero((part == 1) & (nv >= 2))[0]
    fitp = np.nonzero((part == 0) & (nv >= 1))[0]
    hold_all = np.nonzero((part == 1) & (nv >= 1))[0]
    H = len(hold)
    rng = np.random.default_rng(20261115)
    shuf_of = hold[rng.permutation(H)]
    same = shuf_of == hold
    shuf_of[same] = hold[(np.nonzero(same)[0] + 1) % H]
    W = rng.multinomial(H, np.ones(H) / H, size=nboot).astype(np.float64)  # bootstrap paper weights (B, H)
    # per-cluster unigram counts need DISC-FIT token counts per view
    views = zm["views"]

    def contributions(vl):
        """Per held-out paper contributions for one fitted assignment model. vl (P,4,K) view logits."""
        K = vl.shape[-1]
        p = softmax(vl)
        a = vl.argmax(-1)
        # kappa_view contingency per paper (all unordered view pairs)
        kv = np.zeros((H, K * K))
        sep = np.zeros((H, K * K))
        for h, d in enumerate(hold):
            n = int(nv[d])
            for i in range(n):
                for j in range(i + 1, n):
                    kv[h, a[d, i] * K + a[d, j]] += 0.5
                    kv[h, a[d, j] * K + a[d, i]] += 0.5
            sep[h, a[d, 0] * K + a[d, n - 1]] += 0.5
            sep[h, a[d, n - 1] * K + a[d, 0]] += 0.5
        # coherence: mu from DISC-FIT views
        ii, jj = np.nonzero(np.arange(4)[None, :] < nv[fitp][:, None])
        pf, ff = p[fitp][ii, jj], Fh[fitp][ii, jj]
        mu = pf.T @ ff
        mu /= np.maximum(1e-12, np.linalg.norm(mu, axis=1, keepdims=True))
        # LD: per-cluster unigram from DISC-FIT views
        cntk = np.zeros((K, V))
        for pi, (d, j) in enumerate(zip(fitp[ii], jj)):
            np.add.at(cntk, (slice(None), views[d, j].astype(np.int64)), pf[pi][:, None])
        Ug = np.exp(logp_g)
        Uk = 0.9 * cntk / np.maximum(1e-12, cntk.sum(1, keepdims=True)) + 0.1 * Ug[None, :]
        lUk = np.log(Uk)
        csum = np.zeros((H, 2))  # true, shuf sums of coherence
        lsum = np.zeros((H, 2))
        npairs = np.zeros(H)
        for h, d in enumerate(hold):
            n = int(nv[d])
            ds = shuf_of[h]
            Pv = p[d, :n]
            Ps = np.repeat(p[ds, :1], n, 0)
            G = Pv @ mu
            G /= np.maximum(1e-12, np.linalg.norm(G, axis=1, keepdims=True))
            Gs = Ps @ mu
            Gs /= np.maximum(1e-12, np.linalg.norm(Gs, axis=1, keepdims=True))
            Ft = Fh[d, :n]
            cm = G @ Ft.T
            cs = Gs @ Ft.T
            off = ~np.eye(n, dtype=bool)
            csum[h, 0] = cm[off].sum()
            csum[h, 1] = cs[off].sum()
            tok = views[d, :n].astype(np.int64)  # (n,256)
            M = np.exp(lUk[:, tok])  # (K, n, 256)
            base = logp_g[tok]  # (n,256)
            mix = np.einsum("ik,knt->int", Pv, M)  # (n_i, n_j, 256)
            mixs = np.einsum("ik,knt->int", Ps, M)
            gain = (np.log(mix) - base[None]).mean(-1)
            gains = (np.log(mixs) - base[None]).mean(-1)
            lsum[h, 0] = gain[off].sum()
            lsum[h, 1] = gains[off].sum()
            npairs[h] = off.sum()
        return {"kv": kv, "sep": sep, "csum": csum, "lsum": lsum, "npairs": npairs, "K": K}

    def kappa_from_counts(c, K):
        c = c.reshape(K, K)
        n = c.sum()
        po = np.trace(c) / n
        pe = (c.sum(0) / n) @ (c.sum(1) / n)
        return (po - pe) / max(1e-12, 1 - pe)

    res = {}
    t0 = time.time()
    for rep in REPS:
        for K in KS:
            key = f"{rep}_K{K}"
            per_seed, qs, hard_hold, elig_seed = [], [], [], []
            for s in PROTO_SEEDS:
                src = LAT0 if (VAR and rep == "SEM") else LAT  # revision 1: SEM unchanged, reused
                vl = np.load(src / f"{key}_s{s}" / "view_logits.npz")["main"].astype(np.float64)
                c = contributions(vl)
                q = paper_q(vl, nv)
                qs.append(q)
                hh = q[hold_all].argmax(1)
                hard_hold.append(hh)
                frac = np.bincount(hh, minlength=K) / len(hh)
                pbar = q[hold_all].mean(0)
                eff = float(np.exp(-(pbar * np.log(np.maximum(1e-12, pbar))).sum()))
                # nuisance predictability from view assignment probabilities (DISC-HOLD, 5-fold CV)
                ii, jj = np.nonzero(np.arange(4)[None, :] < nv[hold_all][:, None])
                pv = softmax(vl[hold_all][ii, jj])
                pz = (pv - pv.mean(0)) / (pv.std(0) + 1e-9)
                fo = (np.arange(len(hold_all))[ii] * 7919) % 5
                r2n = {}
                for name, col in (("scout_ce", 0), ("log_grad_norm", 1)):
                    y = nuis[hold_all][ii, jj, col][:, None]
                    sse = sst = 0.0
                    for f in range(5):
                        trm, tem = fo != f, fo == f
                        Wb = ridge_fit(pz[trm], y[trm], 1.0)
                        sse += float(((y[tem] - ridge_pred(Wb, pz[tem])) ** 2).sum())
                        sst += float(((y[tem] - y[trm].mean()) ** 2).sum())
                    r2n[name] = 1 - sse / sst
                el = {"frac_min": float(frac.min()), "frac_max": float(frac.max()), "eff_clusters": eff,
                      "nuis_r2": r2n, "mean_norm_entropy_q": float((-(q[hold_all] * np.log(
                          np.maximum(1e-12, q[hold_all]))).sum(1)).mean() / math.log(K))}
                el["eligible"] = bool(frac.min() >= 0.2 / K and frac.max() <= 3.0 / K and eff >= 0.75 * K
                                      and r2n["scout_ce"] <= 0.5 and r2n["log_grad_norm"] <= 0.5)
                elig_seed.append(el)
                per_seed.append(c)
            # seed stability (fixed Hungarian matching on the full sample), per-paper one-hot contingencies
            pair_counts, pair_k = [], []
            for x in range(3):
                for y in range(x + 1, 3):
                    kap, inv = matched_kappa(hard_hold[x], hard_hold[y], K)
                    pair_k.append(kap)
                    hx = hard_hold[x]
                    hy = inv[hard_hold[y]]
                    # restrict to the >= 2-view held-out papers for bootstrap alignment
                    pos = np.searchsorted(hold_all, hold)
                    cc = np.zeros((H, K * K))
                    cc[np.arange(H), hx[pos] * K + hy[pos]] = 1.0
                    pair_counts.append(cc)
            medoid = int(np.argmax([np.mean([pair_k[i] for i, (x, y) in enumerate([(0, 1), (0, 2), (1, 2)])
                                             if s in (x, y)]) for s in range(3)]))

            def stats(w):
                kv = np.mean([kappa_from_counts(w @ c["kv"], K) for c in per_seed])
                ks = np.mean([kappa_from_counts(w @ cc, K) for cc in pair_counts])
                C = np.mean([(w @ c["csum"][:, 0] - w @ c["csum"][:, 1]) / (w @ c["npairs"]) for c in per_seed])
                LD = np.mean([(w @ c["lsum"][:, 0] - w @ c["lsum"][:, 1]) / (w @ c["npairs"]) for c in per_seed])
                return np.array([kv, ks, C, LD])
            pt = stats(np.ones(H))
            bs = np.stack([stats(W[b]) for b in range(nboot)])
            sep = np.mean([kappa_from_counts(c["sep"].sum(0), K) for c in per_seed])
            res[key] = {"rep": rep, "K": K, "kappa_view": pt[0], "kappa_seed": pt[1], "C": pt[2], "LD": pt[3],
                        "kappa_seed_pairs": pair_k, "kappa_most_separated_pair": float(sep),
                        "ci95": {m: [float(np.quantile(bs[:, i], 0.025)), float(np.quantile(bs[:, i], 0.975))]
                                 for i, m in enumerate(("kappa_view", "kappa_seed", "C", "LD"))},
                        "eligibility_per_seed": elig_seed, "eligible": all(e["eligible"] for e in elig_seed),
                        "medoid_seed": medoid, "_bs": bs, "_q": qs[medoid]}
            print(key, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[key].items()
                                   if not k.startswith("_") and k not in ("eligibility_per_seed", "ci95")}),
                  f"{time.time() - t0:.0f}s", flush=True)
    elig = [k for k in res if res[k]["eligible"]]
    star = {}
    for i, m in enumerate(("kappa_view", "kappa_seed", "C", "LD")):
        star[m] = max([res[k][m] for k in elig], default=0.0)
    terms = [m for m in star if star[m] > 0]
    for k in res:
        pt = np.array([res[k][m] for m in ("kappa_view", "kappa_seed", "C", "LD")])
        sel = [i for i, m in enumerate(("kappa_view", "kappa_seed", "C", "LD")) if m in terms]
        den = np.array([star[m] for m in ("kappa_view", "kappa_seed", "C", "LD")])
        res[k]["S"] = float(np.mean(pt[sel] / den[sel]))
        sb = (res[k]["_bs"][:, sel] / den[sel]).mean(1)
        res[k]["S_se"] = float(sb.std(ddof=1))
    chosen = {}
    for rep in REPS:
        ks = [k for k in elig if res[k]["rep"] == rep]
        if not ks:
            chosen[rep] = None
            continue
        best = max(ks, key=lambda k: res[k]["S"])
        thr = res[best]["S"] - res[best]["S_se"]
        chosen[rep] = min([k for k in ks if res[k]["S"] >= thr], key=lambda k: res[k]["K"])
    # validity
    va = {}
    for rep in ("GRAD", "COMBINED"):
        k = chosen[rep]
        va[rep] = bool(k is not None and all(res[k]["ci95"][m][0] > 0 for m in ("kappa_view", "C", "LD")))
    vb = {}
    for rep in ("GRAD", "COMBINED"):
        k, ks = chosen[rep], chosen["SEM"]
        if k is None:
            vb[rep] = None
            continue
        if ks is None:
            vb[rep] = {"diff": None, "ci95": None, "pass": True, "note": "SEM has no eligible candidate"}
            continue
        d = res[k]["_bs"][:, 2] - res[ks]["_bs"][:, 2]
        vb[rep] = {"diff": res[k]["C"] - res[ks]["C"], "ci95": [float(np.quantile(d, 0.025)),
                                                                 float(np.quantile(d, 0.975))],
                   "pass": bool(np.quantile(d, 0.025) > 0)}
    valid_a = any(va.values())
    valid_b = any(v and v["pass"] for v in vb.values())
    gc_cands = [r for r in ("GRAD", "COMBINED") if chosen[r] is not None and va[r]]
    winner = max(gc_cands, key=lambda r: res[chosen[r]]["S"]) if gc_cands else None
    # frozen assignment tables (medoid seed)
    tabs = {}
    for rep, k in chosen.items():
        if k is None:
            continue
        r = res[k]
        q = r["_q"].astype(np.float32)
        p = LAT / f"table_{rep}.npz"
        np.savez(p, opaque=zm["opaque"], q=q, K=np.int64(r["K"]), seed=np.int64(r["medoid_seed"]))
        tabs[rep] = {"path": str(p.relative_to(guard.ROOT)), "sha256": guard.sha256_file(p), "config": k,
                     "medoid_seed": r["medoid_seed"]}
        for s in ("calib_search", "calib_sel"):
            zc = np.load(guard.ROOT / f"artifacts/gc-edl-v1/discovery/inputs_{s}.npz")
            src = LAT0 if (VAR and rep == "SEM") else LAT
            vlc = np.load(src / f"{k}_s{r['medoid_seed']}" / "view_logits.npz")[s].astype(np.float64)
            pc = LAT / f"table_{rep}_{s}.npz"
            np.savez(pc, opaque=zc["opaque"], q=paper_q(vlc, zc["nv"]).astype(np.float32), K=np.int64(r["K"]))
            tabs[rep][f"{s}_path"] = str(pc.relative_to(guard.ROOT))
            tabs[rep][f"{s}_sha256"] = guard.sha256_file(pc)
    if winner is not None:
        q = np.load(LAT / f"table_{winner}.npz")["q"]
        has = np.nonzero(nv > 0)[0]
        perm = np.random.default_rng(20261116).permutation(has)
        qs_ = q.copy()
        qs_[has] = q[perm]
        p = LAT / "table_SHUF.npz"
        np.savez(p, opaque=zm["opaque"], q=qs_, K=np.int64(q.shape[1]), source=np.asarray(winner))
        tabs["SHUF"] = {"path": str(p.relative_to(guard.ROOT)), "sha256": guard.sha256_file(p), "source": winner}
    clean = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in res.items()}
    dec = {"time": guard.now(), "metrics": clean, "star": star, "terms_in_S": terms, "chosen": chosen,
           "validity_a_beats_shuffled": va, "validity_b_grad_beyond_sem": vb, "valid_a": valid_a,
           "valid_b": valid_b, "discovery_valid": bool(valid_a and valid_b), "winner_gc": winner, "tables": tabs,
           "nboot": nboot, "n_hold_papers_2views": int(H), "seconds": time.time() - t0}
    guard.atomic_write_json(dec_p, dec, mode=0o400)
    with open(ART / "artifact_hashes.tsv", "a") as fh:
        fh.write(f"{guard.now()}\t{dec_p.relative_to(guard.ROOT)}\t{guard.sha256_file(dec_p)}\t\n")
        for t in tabs.values():
            fh.write(f"{guard.now()}\t{t['path']}\t{t['sha256']}\t\n")
    print(json.dumps({k: v for k, v in dec.items() if k != "metrics"}, indent=1, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["residualize", "fit", "select"])
    ap.add_argument("--rep")
    ap.add_argument("--k", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--variant", default="", choices=["", "r1"])
    a = ap.parse_args()
    guard.check_root()
    import numpy  # noqa: F401
    import torch  # noqa: F401
    from csr import gc_lib, lfr_eval  # noqa: F401
    from csr.gc_lib import install_label_firewall
    from csr.gc_paths import check_prereg
    check_prereg()
    install_label_firewall()
    global VAR
    VAR = a.variant
    if a.cmd == "residualize":
        residualize()
    elif a.cmd == "fit":
        fit(a.rep, a.k, a.seed)
    else:
        select()


if __name__ == "__main__":
    main()
