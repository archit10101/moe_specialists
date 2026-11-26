#!/usr/bin/env python
"""specialization-audit-v1 analysis (docs/SPECIALIZATION_AUDIT_V1_PREREG.md §7-§10). CPU only; read-only on runs.

Per checkpoint (worker processes): routing joint P(topic, expert) from normalized gate mass on effect windows, MI /
NMI / permutation-adjusted MI with the 1,000-permutation load-preserving null, document bootstrap (2,000), per-expert
entropy / purity / KL / within-expert cosine / diversity / support, pool-level MI, topic probe (C = 1) with controls,
functional interactions (stratified ATT on the discovery-defined top topic) with fixed-weight document bootstrap and
the shuffled-topic null (primary view, primary layers). Aggregation: cross-seed Hungarian alignment with null, paired
contrasts E-D, C-B, D-F, B-A per seed, prereg decision rules. Writes artifacts/specialization-audit-v1/analysis/.
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402
from csr import specaudit as sa  # noqa: E402

AUD = guard.ROOT / "artifacts" / "specialization-audit-v1"
COMPARISONS = [("E", "D", "semantic alignment loss | semantic chunks"), ("C", "B", "semantic alignment loss | fixed chunks"),
               ("D", "F", "semantic vs matched-random boundaries"), ("B", "A", "hard chunk pooling vs token routing")]
PRIMARY_VIEW, ROBUST_VIEW = "enc_K32", "tfidf_K32"
ROUTING_VIEWS = ["enc_K32", "tfidf_K32", "enc_K16", "enc_K64", "tfidf_K16", "tfidf_K64"]
FUNC_VIEWS = ["enc_K32", "tfidf_K32", "enc_K16", "enc_K64"]
PRIMARY_LAYERS = [0, 3, 6]  # MoE layer indices of blocks 1, 4, 7
G = {}  # shared read-only globals (fork)


def kof(view):
    return int(view.split("_K")[1])


def setup_globals():
    import numpy as np
    W = dict(np.load(AUD / "data" / "windows.npz"))
    man = json.loads((AUD / "data" / "manifest.json").read_text())
    lab = dict(np.load(AUD / "labels" / "labels.npz"))
    G["W"] = W
    G["wl"] = W["end"] - W["start"]
    G["woff"] = np.concatenate([[0], np.cumsum(G["wl"])]).astype(np.int64)
    G["wid_tok"] = np.repeat(np.arange(len(W["doc"])), G["wl"])
    G["labels"] = {v: lab[v].astype(np.int64) for v in ROUTING_VIEWS}
    G["qedges"] = man["doc_length_quartile_edges"]
    G["emb"] = np.load(AUD / "labels" / "encoder_embeddings.npy")
    eW = W["half"] == 1
    G["eW"], G["dW"] = np.nonzero(eW)[0], np.nonzero(~eW)[0]
    eff_docs = np.unique(W["doc"][eW])
    G["eff_docs"] = eff_docs
    dmap = -np.ones(int(W["doc"].max()) + 1, np.int64)
    dmap[eff_docs] = np.arange(len(eff_docs))
    G["doceff_of_w"] = dmap[W["doc"]]
    G["P"] = sa.stratified_permutations(W["half"] * 16 + W["stratum"], n_perm=sa.N_PERM)
    G["M"] = sa.bootstrap_counts(len(eff_docs), n_boot=sa.N_BOOT)


# ------------------------------------------------------------------ routing
def window_masses(R, E=32):
    import numpy as np
    W = len(G["wl"])
    n_moe = R["topi"].shape[0]
    WM = np.zeros((n_moe, W, E))
    PM = np.zeros((n_moe, W, E))
    wt = G["wid_tok"]
    for l in range(n_moe):
        g = R["gates"][l].astype(np.float64)
        g = g / g.sum(1, keepdims=True)
        WM[l] = np.bincount((wt[:, None] * E + R["topi"][l].astype(np.int64)).ravel(), weights=g.ravel(),
                            minlength=W * E).reshape(W, E)
        con = R["constrained"][l]
        if con.any():
            bits = (R["pool"][l][con][:, None] >> np.arange(E, dtype=np.uint32)) & 1
            for e in range(E):
                PM[l][:, e] = np.bincount(wt[con], weights=bits[:, e] / 8.0, minlength=W)
    return WM, PM


def joints_perm(lab_w, Xcat, K, perm_idx=None, chunk=50):
    """Joint mass (n, K, F) for label arrays lab_w[P[p]] (or observed if perm_idx None); Xcat (Wsub, F) rows aligned
    with the window subset; perm_idx (n, Wsub) indices into the full window array."""
    import numpy as np
    import torch
    X = torch.as_tensor(Xcat, dtype=torch.float32)
    if perm_idx is None:
        oh = torch.nn.functional.one_hot(torch.as_tensor(lab_w), K).float()
        return (oh.T @ X).double().numpy()[None]
    out = []
    labs = torch.as_tensor(G["labels_cur"])
    for s in range(0, len(perm_idx), chunk):
        li = labs[torch.as_tensor(perm_idx[s:s + chunk].astype(np.int64))]
        oh = torch.nn.functional.one_hot(li, K).float()  # (c, Wsub, K)
        out.append(torch.einsum("cwk,wf->ckf", oh, X).double().numpy())
    return np.concatenate(out)


def mi_layers(J, n_moe, E=32):
    import numpy as np
    n, K, _ = J.shape
    return sa.mutual_information(J.reshape(n, K, n_moe, E).transpose(0, 2, 1, 3))  # (n, n_moe)


def routing_view(view, WM, PM, pooled, do_null, do_boot):
    import numpy as np
    import torch
    K = kof(view)
    lab = G["labels"][view]
    G["labels_cur"] = lab
    n_moe, Wn, E = WM.shape
    eW, dW = G["eW"], G["dW"]
    WMcat = WM.transpose(1, 0, 2).reshape(Wn, n_moe * E)
    PMcat = PM.transpose(1, 0, 2).reshape(Wn, n_moe * E)
    J = joints_perm(lab[eW], WMcat[eW], K)[0]
    out = {"K": K}
    mi = mi_layers(J[None], n_moe)[0]
    out["mi"] = mi.tolist()
    Jl = J.reshape(K, n_moe, E).transpose(1, 0, 2)  # (layer, K, E)
    nm, hte, pur, kl, div, zero, low, cos_ex, cos_e_mean = [], [], [], [], [], [], [], [], []
    selwin = (WM[:, eW] > 0).sum(1)  # (layer, E) distinct effect windows selecting expert
    X = G["emb"][eW]
    for l in range(n_moe):
        P_ = Jl[l] / Jl[l].sum()
        nm.append(sa.nmi(P_))
        pe = P_.sum(0)
        pk = P_.sum(1)
        cond = np.where(pe > 0, P_ / np.where(pe > 0, pe, 1), 0)  # P(k|e) columns
        sup = selwin[l] >= sa.SUPPORT_WINDOWS
        h = sa.entropy(cond.T, axis=1)
        hte.append(float(np.sum(pe * h)))
        pur.append(float(np.sum(pe * cond.max(0))))
        with np.errstate(divide="ignore", invalid="ignore"):
            kle = np.where(cond > 0, cond * (np.log(cond) - np.log(pk[:, None])), 0).sum(0)
        kl.append(float(np.mean(kle[sup])) if sup.any() else None)
        prof = cond[:, sup].T
        if len(prof) > 1:
            jj = [sa.jsd(prof[a], prof[b]) for a in range(len(prof)) for b in range(a + 1, len(prof))]
            div.append(float(np.mean(jj)))
        else:
            div.append(None)
        zero.append(int((WM[l][eW].sum(0) == 0).sum()))
        low.append(int((selwin[l] < sa.SUPPORT_WINDOWS).sum()))
        if view == PRIMARY_VIEW:
            m_all = WM[l][eW].sum(1)
            cg = sa.weighted_pairwise_cosine(X, m_all)
            ce = np.array([sa.weighted_pairwise_cosine(X, WM[l][eW, e]) if sup[e] else np.nan for e in range(E)])
            w = WM[l][eW].sum(0)
            ok = np.isfinite(ce)
            cos_ex.append(float(np.sum(w[ok] * (ce[ok] - cg)) / w[ok].sum()) if ok.any() else None)
            cos_e_mean.append(float(np.nanmean(ce)) if ok.any() else None)
    out.update({"nmi": nm, "H_T_given_E": hte, "purity_massweighted": pur, "kl_expert_vs_global_mean": kl,
                "between_expert_jsd_diversity": div, "zero_load_experts": zero, "low_support_experts": low,
                "H_T": float(sa.entropy(Jl[0].sum(1) / Jl[0].sum())),
                "H_E": [float(sa.entropy(Jl[l].sum(0) / Jl[l].sum())) for l in range(n_moe)]})
    if view == PRIMARY_VIEW:
        out["within_expert_cosine_excess_massweighted"] = cos_ex
        out["within_expert_cosine_mean"] = cos_e_mean
    res = {"summary": out}
    if pooled:
        Jp = joints_perm(lab[eW], PMcat[eW], K)[0]
        res["summary"]["pool_mi"] = mi_layers(Jp[None], n_moe)[0].tolist()
    if do_null:
        Jn = joints_perm(None, WMcat[eW], K, perm_idx=G["P"][:, eW])
        res["null_mi"] = mi_layers(Jn, n_moe)  # (1000, n_moe)
        if pooled:
            res["null_pool_mi"] = mi_layers(joints_perm(None, PMcat[eW], K, perm_idx=G["P"][:, eW]), n_moe)
        if view == PRIMARY_VIEW:
            res["null_disc_joint"] = joints_perm(None, WMcat[dW], K, perm_idx=G["P"][:, dW]).astype(np.float32)
    res["disc_joint"] = joints_perm(lab[dW], WMcat[dW], K)[0]
    if do_boot:
        D = len(G["eff_docs"])
        idx = torch.as_tensor(G["doceff_of_w"][eW] * K + lab[eW])
        DJ = torch.zeros(D * K, n_moe * E, dtype=torch.float64).index_add_(0, idx, torch.as_tensor(WMcat[eW]))
        Jb = (torch.as_tensor(G["M"], dtype=torch.float64) @ DJ.view(D, K * n_moe * E)).view(-1, K, n_moe * E).numpy()
        res["boot_mi"] = mi_layers(Jb, n_moe)  # (2000, n_moe)
    return res


def probe_view(view, WM):
    import numpy as np
    import torch
    K = kof(view)
    lab = G["labels"][view]
    n_moe, Wn, E = WM.shape
    F_ = (WM / G["wl"][None, :, None]).transpose(1, 0, 2).reshape(Wn, n_moe * E)
    dW, eW = G["dW"], G["eW"]
    mu, sd = F_[dW].mean(0), F_[dW].std(0)
    Z = (F_ - mu) / np.where(sd > 1e-8, sd, 1.0)
    W_, b_, info = sa.logreg_fit(Z[dW], lab[dW], K)
    yhat = sa.logreg_predict(Z[eW], W_, b_)
    y = lab[eW]
    classes = np.unique(y)
    out = {"macro_f1": sa.macro_f1(y, yhat, classes), "balanced_accuracy": sa.balanced_accuracy(y, yhat),
           "accuracy": float(np.mean(y == yhat)), "fit": info}
    maj = np.bincount(lab[dW], minlength=K).argmax()
    out["majority_macro_f1"] = sa.macro_f1(y, np.full_like(y, maj), classes)
    out["majority_balanced_accuracy"] = sa.balanced_accuracy(y, np.full_like(y, maj))
    ysh = lab[G["P"][0][dW]]
    Ws, bs, _ = sa.logreg_fit(Z[dW], ysh, K)
    ysh_hat = sa.logreg_predict(Z[eW], Ws, bs)
    out["shuffled_macro_f1"] = sa.macro_f1(y, ysh_hat, classes)
    out["shuffled_balanced_accuracy"] = sa.balanced_accuracy(y, ysh_hat)
    # bootstrap macro-F1 (documents resampled; probe not refit)
    Mw = torch.as_tensor(G["M"][:, G["doceff_of_w"][eW]], dtype=torch.float64)
    oh = torch.nn.functional.one_hot(torch.as_tensor(y * K + yhat), K * K).double()
    conf = (Mw @ oh).view(-1, K, K).numpy()  # (B, true, pred)
    tp = np.einsum("bkk->bk", conf)
    fp = conf.sum(1) - tp
    fn = conf.sum(2) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
    out_boot = f1[:, classes].mean(1)
    return out, out_boot


# ------------------------------------------------------------------ functional
def functional_layer_setup(R, CF, l, li, strat):
    """Entries (effect token x rank) for MoE layer l; cells per expert. Returns dict of arrays."""
    import numpy as np
    eff = CF["eff_index"]
    geff = np.nonzero(eff >= 0)[0]
    assert np.array_equal(eff[geff], np.arange(len(geff)))
    k = R["topi"].shape[2]
    e = R["topi"][l, geff].astype(np.int64).reshape(-1)
    d = CF["dce"][li].T.reshape(-1).astype(np.float64)
    gate = R["gates"][l, geff].astype(np.float64).reshape(-1)
    tok = np.repeat(np.arange(len(geff)), k)
    gt = geff[tok]
    win = G["wid_tok"][gt]
    ce0 = R["ce0"][gt].astype(np.float64)
    pos = R["position"][gt].astype(np.int64)
    cbin = np.zeros(len(e), np.int64) if strat == "none" else sa.bin_of(R["chunk_len"][gt], sa.CHUNK_BINS[1:])
    dbin = sa.bin_of(G["W"]["doc_len"][win], G["qedges"])
    assert np.isfinite(d).all()
    cell = np.zeros(len(e), np.int64)
    for ex in range(32):
        s = np.nonzero(e == ex)[0]
        if len(s):
            cell[s] = ex * 16000 + sa.cell_keys(ce0[s], gate[s], pos[s], cbin[s], dbin[s])
    ucell, cid = np.unique(cell, return_inverse=True)
    e_of_c = (ucell // 16000).astype(np.int64)
    n_c = np.bincount(cid).astype(np.float64)
    S_c = np.bincount(cid, weights=d)
    pair = e * len(G["wl"]) + win
    upair, pid = np.unique(pair, return_inverse=True)
    e_of_p = (upair // len(G["wl"])).astype(np.int64)
    return {"e": e, "d": d, "win": win, "cid": cid, "e_of_c": e_of_c, "n_c": n_c, "S_c": S_c, "pid": pid,
            "e_of_p": e_of_p, "doc": G["doceff_of_w"][win], "n_e": np.bincount(e, minlength=32)}


def functional_estimate(S, lab_w, top, want_boot=False):
    """Vectorized stratified interaction for all 32 experts given window labels and per-expert top topic."""
    import numpy as np
    e, cid = S["e"], S["cid"]
    t_e = top[e]
    is_top = ((t_e >= 0) & (lab_w[S["win"]] == t_e)).astype(np.float64)
    C = len(S["n_c"])
    nT = np.bincount(cid, weights=is_top, minlength=C)
    sT = np.bincount(cid, weights=S["d"] * is_top, minlength=C)
    nO, sO = S["n_c"] - nT, S["S_c"] - sT
    ok = (nT > 0) & (nO > 0)
    NT = np.bincount(S["e_of_c"], weights=nT * ok, minlength=32)
    with np.errstate(divide="ignore", invalid="ignore"):
        pi = np.where(ok, nT / NT[S["e_of_c"]], 0.0)
        contrib = np.where(ok, pi * (sT / np.where(nT > 0, nT, 1) - sO / np.where(nO > 0, nO, 1)), 0.0)
    I = np.bincount(S["e_of_c"], weights=contrib, minlength=32)
    tok_ok = ok[cid]
    selT = tok_ok & (is_top == 1)
    selO = tok_ok & (is_top == 0)
    npair = len(S["e_of_p"])
    hasT = np.bincount(S["pid"][selT], minlength=npair) > 0
    hasO = np.bincount(S["pid"][selO], minlength=npair) > 0
    wT = np.bincount(S["e_of_p"], weights=hasT, minlength=32)
    wO = np.bincount(S["e_of_p"], weights=hasO, minlength=32)
    elig = (top >= 0) & (wT >= sa.SUPPORT_WINDOWS) & (wO >= sa.SUPPORT_WINDOWS)
    res = {"I": np.where(elig, I, np.nan), "elig": elig, "wT": wT, "wO": wO,
           "nT_matched": np.bincount(S["e_of_c"], weights=nT * ok, minlength=32),
           "nO_matched": np.bincount(S["e_of_c"], weights=nO * ok, minlength=32),
           "nT": np.bincount(e, weights=is_top, minlength=32)}
    if want_boot:
        D = len(G["eff_docs"])
        with np.errstate(divide="ignore", invalid="ignore"):
            omega = np.where(ok, pi / np.where(nO > 0, nO, 1), 0.0)[cid]
        key = S["doc"] * 32 + e
        AT = np.bincount(key, weights=S["d"] * selT, minlength=D * 32).reshape(D, 32)
        BT = np.bincount(key, weights=selT.astype(float), minlength=D * 32).reshape(D, 32)
        AO = np.bincount(key, weights=S["d"] * omega * selO, minlength=D * 32).reshape(D, 32)
        BO = np.bincount(key, weights=omega * selO, minlength=D * 32).reshape(D, 32)
        res["boot_parts"] = np.concatenate([AT, BT, AO, BO], 1)  # (D, 128)
    return res


def summarize(res, n_e):
    import numpy as np
    el = res["elig"]
    if not el.any():
        return None, None
    eq = float(np.mean(res["I"][el]))
    ut = float(np.sum(n_e[el] * res["I"][el]) / np.sum(n_e[el]))
    return eq, ut


def boot_summary(parts_by_layer, elig_by_layer):
    """parts (D, 128) per layer -> (B, n_layers) equal-expert-weighted interaction replicates (weights fixed)."""
    import numpy as np
    import torch
    M = torch.as_tensor(G["M"], dtype=torch.float64)
    out = []
    for parts, el in zip(parts_by_layer, elig_by_layer):
        Mb = (M @ torch.as_tensor(parts)).numpy()
        AT, BT, AO, BO = Mb[:, :32], Mb[:, 32:64], Mb[:, 64:96], Mb[:, 96:]
        with np.errstate(divide="ignore", invalid="ignore"):
            I = AT / BT - AO / BO
        out.append(np.nanmean(np.where(el[None], I, np.nan), 1) if el.any() else np.full(len(Mb), np.nan))
    return np.stack(out, 1)


# ------------------------------------------------------------------ per-checkpoint worker
def analyze_checkpoint(arg):
    key, run_dir, strat = arg
    import numpy as np
    import torch
    torch.set_num_threads(1)
    t0 = time.time()
    R = dict(np.load(Path(run_dir) / "routing.npz"))
    CF = dict(np.load(Path(run_dir) / "cf.npz"))
    assert R["seen"].all() and np.isfinite(R["ce0"]).all()
    pooled = strat != "none"
    WM, PM = window_masses(R)
    out = {"key": key, "routing": {}, "probe": {}, "functional": {}}
    raw = {}
    for v in ROUTING_VIEWS:
        r = routing_view(v, WM, PM, pooled, do_null=True, do_boot=v in (PRIMARY_VIEW, ROBUST_VIEW))
        out["routing"][v] = r["summary"]
        null = r["null_mi"]
        out["routing"][v]["null_mi_mean"] = null.mean(0).tolist()
        out["routing"][v]["null_mi_p975"] = np.percentile(null, 97.5, axis=0).tolist()
        out["routing"][v]["mi_adj"] = (np.array(r["summary"]["mi"]) - null.mean(0)).tolist()
        out["routing"][v]["perm_p_layer_mean"] = float((1 + np.sum(null.mean(1) >= np.mean(r["summary"]["mi"]))) / (1 + len(null)))
        if pooled:
            pn = r["null_pool_mi"]
            out["routing"][v]["pool_mi_adj"] = (np.array(r["summary"]["pool_mi"]) - pn.mean(0)).tolist()
            raw[f"{v}/null_pool_mi"] = pn.astype(np.float32)
        raw[f"{v}/null_mi"] = null.astype(np.float32)
        raw[f"{v}/disc_joint"] = r["disc_joint"].astype(np.float32)
        if "boot_mi" in r:
            raw[f"{v}/boot_mi"] = r["boot_mi"].astype(np.float32)
        if "null_disc_joint" in r:
            raw[f"{v}/null_disc_joint"] = r["null_disc_joint"]
    for v in (PRIMARY_VIEW, ROBUST_VIEW):
        o, b = probe_view(v, WM)
        out["probe"][v] = o
        raw[f"{v}/probe_boot_f1"] = b.astype(np.float32)
    n_moe = WM.shape[0]
    layers = [int(x) for x in CF["layers"]]
    assert layers == list(range(n_moe))
    setups = {l: functional_layer_setup(R, CF, l, li, strat) for li, l in enumerate(layers)}
    for v in FUNC_VIEWS:
        K = kof(v)
        lab = G["labels"][v]
        dj = raw[f"{v}/disc_joint"].reshape(K, n_moe, 32)
        fv = {"layers": {}}
        parts, elig = [], []
        for l in layers:
            mass = dj[:, l, :]
            top = np.where(mass.sum(0) > 0, mass.argmax(0), -1)
            S = setups[l]
            res = functional_estimate(S, lab, top, want_boot=v in (PRIMARY_VIEW, ROBUST_VIEW))
            eq, ut = summarize(res, S["n_e"])
            glob = np.bincount(S["e"], weights=S["d"], minlength=32) / np.maximum(S["n_e"], 1)
            fv["layers"][str(l)] = {"equal_weighted": eq, "utilization_weighted": ut, "n_eligible": int(res["elig"].sum()),
                                    "n_zero_load": int((top < 0).sum()),
                                    "n_insufficient_support": int(((top >= 0) & ~res["elig"]).sum()),
                                    "global_mean_dce_mean_over_experts": float(np.mean(glob[S["n_e"] > 0])),
                                    "experts": [{"expert": ex, "top_topic": int(top[ex]),
                                                 "estimate": None if not res["elig"][ex] else float(res["I"][ex]),
                                                 "eligible": bool(res["elig"][ex]), "n_tokens": int(S["n_e"][ex]),
                                                 "n_T": int(res["nT"][ex]), "n_T_matched": int(res["nT_matched"][ex]),
                                                 "n_O_matched": int(res["nO_matched"][ex]), "windows_T": int(res["wT"][ex]),
                                                 "windows_O": int(res["wO"][ex]), "global_mean_dce": float(glob[ex])}
                                                for ex in range(32)]}
            if "boot_parts" in res:
                parts.append(res["boot_parts"])
                elig.append(res["elig"])
        if parts:
            bs = boot_summary(parts, elig)
            raw[f"{v}/func_boot"] = bs.astype(np.float32)
        prim = [fv["layers"][str(l)]["equal_weighted"] for l in PRIMARY_LAYERS]
        fv["F_primary"] = float(np.mean(prim)) if all(p is not None for p in prim) else None
        allv = [fv["layers"][str(l)]["equal_weighted"] for l in layers]
        fv["F_all7_secondary"] = float(np.mean(allv)) if all(p is not None for p in allv) else None
        out["functional"][v] = fv
    # shuffled-topic null, primary view, primary layers
    v = PRIMARY_VIEW
    K = kof(v)
    lab = G["labels"][v]
    ndj = raw[f"{v}/null_disc_joint"].reshape(-1, K, n_moe, 32)
    fn = np.full((sa.N_PERM, len(PRIMARY_LAYERS)), np.nan)
    for pi_ in range(sa.N_PERM):
        lab_p = lab[G["P"][pi_]]
        for j, l in enumerate(PRIMARY_LAYERS):
            mass = ndj[pi_, :, l, :]
            top = np.where(mass.sum(0) > 0, mass.argmax(0), -1)
            res = functional_estimate(setups[l], lab_p, top)
            eq, _ = summarize(res, setups[l]["n_e"])
            fn[pi_, j] = np.nan if eq is None else eq
    raw[f"{v}/func_null"] = fn.astype(np.float32)
    out["functional"][v]["null_F_primary_mean"] = float(np.nanmean(fn.mean(1)))
    out["functional"][v]["null_F_primary_p975"] = float(np.nanpercentile(fn.mean(1), 97.5))
    out["seconds"] = time.time() - t0
    rawp = AUD / "analysis" / "raw" / f"{key}.npz"
    np.savez(rawp, **{k.replace("/", "__"): val for k, val in raw.items()})
    print(f"{key} done {out['seconds']:.0f}s", flush=True)
    return out


# ------------------------------------------------------------------ aggregation
def cross_seed(per, raws):
    import numpy as np
    K = 32
    out = {}
    pairs = [(0, 1), (0, 2), (1, 2)]

    def profiles(J):  # J (K, n_moe*32) -> (n_moe, 32, K) P(T|e), zero mass -> global
        Jl = J.reshape(K, 7, 32).transpose(1, 2, 0)
        tot = Jl.sum(2, keepdims=True)
        glob = Jl.sum(1, keepdims=True) / Jl.sum((1, 2), keepdims=True)
        return np.where(tot > 0, Jl / np.where(tot > 0, tot, 1), glob), Jl.sum(1, keepdims=True) / Jl.sum((1, 2), keepdims=True)

    def metrics(profs, globs):
        res = {"matched_jsd": [], "repro_mass": [], "repro_lift": []}
        for l in range(7):
            assign = {}
            mj = []
            for a, b in pairs:
                cost = sa.jsd(profs[a][l][:, None, :], profs[b][l][None, :, :])
                col = sa.hungarian_fast(cost)
                assign[(a, b)] = col
                mj.append(float(cost[np.arange(32), col].mean()))
            res["matched_jsd"].append(float(np.mean(mj)))
            am = [profs[s][l].argmax(1) for s in range(3)]
            al = [(profs[s][l] / np.maximum(globs[s][l], 1e-12)).argmax(1) for s in range(3)]
            p1, p2 = assign[(0, 1)], assign[(0, 2)]
            res["repro_mass"].append(float(np.mean((am[0] == am[1][p1]) & (am[0] == am[2][p2]))))
            res["repro_lift"].append(float(np.mean((al[0] == al[1][p1]) & (al[0] == al[2][p2]))))
        return {k: float(np.mean(v)) for k, v in res.items()} | {f"{k}_by_layer": v for k, v in res.items()}

    for c in "ABCDEF":
        obs_p = [profiles(raws[f"s{s}_{c}"][f"{PRIMARY_VIEW}__disc_joint"]) for s in range(3)]
        obs = metrics([p[0] for p in obs_p], [p[1][:, 0] for p in obs_p])
        nulls = {"matched_jsd": [], "repro_mass": [], "repro_lift": []}
        nd = [raws[f"s{s}_{c}"][f"{PRIMARY_VIEW}__null_disc_joint"] for s in range(3)]
        for pi_ in range(sa.N_PERM):
            pp = [profiles(nd[s][pi_]) for s in range(3)]
            m = metrics([p[0] for p in pp], [p[1][:, 0] for p in pp])
            for k_ in nulls:
                nulls[k_].append(m[k_])
        out[c] = {"observed": obs, **{f"null_{k_}_mean": float(np.mean(v)) for k_, v in nulls.items()},
                  **{f"null_{k_}_ci95": sa.ci95(v) for k_, v in nulls.items()},
                  "p_matched_jsd_lower": float((1 + np.sum(np.array(nulls["matched_jsd"]) <= obs["matched_jsd"])) / (1 + sa.N_PERM)),
                  "p_repro_mass_higher": float((1 + np.sum(np.array(nulls["repro_mass"]) >= obs["repro_mass"])) / (1 + sa.N_PERM)),
                  "p_repro_lift_higher": float((1 + np.sum(np.array(nulls["repro_lift"]) >= obs["repro_lift"])) / (1 + sa.N_PERM))}
        print("cross-seed", c, json.dumps({k: out[c][k] for k in ("observed",)})[:200], flush=True)
    return out


def contrasts(per, raws):
    import numpy as np
    res = {}
    for X, Y, role in COMPARISONS:
        cs = {"role": role, "seeds": {}}
        for s in range(3):
            kx, ky = f"s{s}_{X}", f"s{s}_{Y}"
            rx, ry = raws[kx], raws[ky]
            d = {}
            for v in (PRIMARY_VIEW, ROBUST_VIEW, "enc_K16", "enc_K64", "tfidf_K16", "tfidf_K64"):
                mx, my = np.mean(per[kx]["routing"][v]["mi"]), np.mean(per[ky]["routing"][v]["mi"])
                nx, ny = rx[f"{v}__null_mi"].mean(1), ry[f"{v}__null_mi"].mean(1)
                dR = (mx - nx.mean()) - (my - ny.mean())
                dnull = nx - ny
                e = {"dR": float(dR), "dMI_raw": float(mx - my), "null_dMI_p975": float(np.percentile(dnull, 97.5)),
                     "null_dMI_p025": float(np.percentile(dnull, 2.5)),
                     "exceeds_null": bool(mx - my > np.percentile(dnull, 97.5)),
                     "below_null": bool(mx - my < np.percentile(dnull, 2.5))}
                if f"{v}__boot_mi" in rx:
                    bd = rx[f"{v}__boot_mi"].mean(1) - ry[f"{v}__boot_mi"].mean(1) - (nx.mean() - ny.mean())
                    e["dR_ci95"] = sa.ci95(bd)
                    e["dR_ci_excludes_0"] = bool(e["dR_ci95"][0] > 0 or e["dR_ci95"][1] < 0)
                d[f"routing_{v}"] = e
            for v in (PRIMARY_VIEW, ROBUST_VIEW):
                px, py = per[kx]["probe"][v]["macro_f1"], per[ky]["probe"][v]["macro_f1"]
                bd = rx[f"{v}__probe_boot_f1"] - ry[f"{v}__probe_boot_f1"]
                d[f"probe_{v}"] = {"dP": float(px - py), "dP_ci95": sa.ci95(bd)}
            for v in FUNC_VIEWS:
                fx, fy = per[kx]["functional"][v]["F_primary"], per[ky]["functional"][v]["F_primary"]
                e = {"dF": None if fx is None or fy is None else float(fx - fy),
                     "dF_all7_secondary": None if per[kx]["functional"][v]["F_all7_secondary"] is None or per[ky]["functional"][v]["F_all7_secondary"] is None
                     else float(per[kx]["functional"][v]["F_all7_secondary"] - per[ky]["functional"][v]["F_all7_secondary"])}
                if f"{v}__func_boot" in rx:
                    bx = np.nanmean(rx[f"{v}__func_boot"][:, PRIMARY_LAYERS], 1)
                    by = np.nanmean(ry[f"{v}__func_boot"][:, PRIMARY_LAYERS], 1)
                    e["dF_ci95"] = sa.ci95((bx - by)[np.isfinite(bx - by)])
                    e["dF_ci_excludes_0"] = bool(e["dF_ci95"][0] > 0 or e["dF_ci95"][1] < 0)
                if f"{v}__func_null" in rx:
                    nd = np.nanmean(rx[f"{v}__func_null"], 1) - np.nanmean(ry[f"{v}__func_null"], 1)
                    e["null_dF_p975"] = float(np.nanpercentile(nd, 97.5))
                    e["null_dF_p025"] = float(np.nanpercentile(nd, 2.5))
                    e["exceeds_null"] = bool(e["dF"] is not None and e["dF"] > e["null_dF_p975"])
                    e["below_null"] = bool(e["dF"] is not None and e["dF"] < e["null_dF_p025"])
                d[f"functional_{v}"] = e
            cs["seeds"][str(s)] = d
        S = cs["seeds"]
        rp = [S[str(s)][f"routing_{PRIMARY_VIEW}"] for s in range(3)]
        cs["routing_increase"] = {
            "a_dR_pos_all": all(x["dR"] > 0 for x in rp), "b_ci_excl0_all": all(x["dR_ci_excludes_0"] for x in rp),
            "c_exceeds_null_all": all(x["exceeds_null"] for x in rp),
            "d_probe_pos_all": all(S[str(s)][f"probe_{PRIMARY_VIEW}"]["dP"] > 0 for s in range(3)),
            "e_tfidf_pos_all": all(S[str(s)][f"routing_{ROBUST_VIEW}"]["dR"] > 0 for s in range(3))}
        cs["routing_increase"]["met"] = all(cs["routing_increase"].values())
        cs["routing_decrease_descriptive"] = all(x["dR"] < 0 and x["dR_ci_excludes_0"] and x["below_null"] for x in rp)
        fp = [S[str(s)][f"functional_{PRIMARY_VIEW}"] for s in range(3)]
        cs["functional_increase"] = {
            "dF_pos_all": all(x["dF"] is not None and x["dF"] > 0 for x in fp),
            "ci_excl0_all": all(x.get("dF_ci_excludes_0", False) for x in fp),
            "exceeds_null_all": all(x.get("exceeds_null", False) for x in fp),
            "tfidf_pos_all": all((S[str(s)][f"functional_{ROBUST_VIEW}"]["dF"] or 0) > 0 for s in range(3))}
        cs["functional_increase"]["met"] = all(cs["functional_increase"].values())
        cs["functional_decrease_descriptive"] = all(x["dF"] is not None and x["dF"] < 0 and x.get("dF_ci_excludes_0")
                                                     and x.get("below_null") for x in fp)
        res[f"{X}-{Y}"] = cs
    return res


def categorize(con):
    ed, cb = con["E-D"], con["C-B"]
    r = [ed["routing_increase"]["met"], cb["routing_increase"]["met"]]
    f = [ed["functional_increase"]["met"], cb["functional_increase"]["met"]]
    both = [r[i] and f[i] for i in range(2)]
    if all(both):
        cat = 1
    elif sum(both) == 1:
        cat = 2
    elif any(r) and not any(f[i] for i in range(2) if r[i]):
        cat = 3
    elif any(f) and not any(r):
        cat = 4
    elif not any(r) and not any(f):
        cat = 5
    else:
        cat = 3 if any(r) else 4
    names = {1: "Strong evidence of semantic specialization", 2: "Context-specific specialization",
             3: "Routing segregation without functional specialization",
             4: "Functional specialization without clear routing segregation", 5: "No evidence of specialization"}
    return {"category": cat, "name": names[cat], "harmful_flag": cat in (1, 2),
            "routing_increase": {"E-D": r[0], "C-B": r[1]}, "functional_increase": {"E-D": f[0], "C-B": f[1]}}


def write_csvs(per, con, xs, outd):
    rows = []
    for key, p in sorted(per.items()):
        for v, r in p["routing"].items():
            for l in range(7):
                rows.append({"key": key, "view": v, "layer_block": l + 1, "mi": r["mi"][l], "mi_adj": r["mi_adj"][l],
                             "null_mi_mean": r["null_mi_mean"][l], "null_mi_p975": r["null_mi_p975"][l], "nmi": r["nmi"][l],
                             "H_T_given_E": r["H_T_given_E"][l], "purity_mw": r["purity_massweighted"][l],
                             "kl_mean": r["kl_expert_vs_global_mean"][l], "jsd_diversity": r["between_expert_jsd_diversity"][l],
                             "zero_load": r["zero_load_experts"][l], "low_support": r["low_support_experts"][l],
                             "pool_mi": (r.get("pool_mi") or [None] * 7)[l], "pool_mi_adj": (r.get("pool_mi_adj") or [None] * 7)[l],
                             "cos_excess": (r.get("within_expert_cosine_excess_massweighted") or [None] * 7)[l]})
    _csv(outd / "routing_metrics.csv", rows)
    rows = [{"key": k, "view": v, **{kk: vv for kk, vv in o.items() if kk != "fit"}} for k, p in sorted(per.items())
            for v, o in p["probe"].items()]
    _csv(outd / "probe.csv", rows)
    rows = []
    for k, p in sorted(per.items()):
        for v, fv in p["functional"].items():
            for l, lv in fv["layers"].items():
                for ex in lv["experts"]:
                    rows.append({"key": k, "view": v, "layer_block": int(l) + 1, "primary_layer": int(l) in PRIMARY_LAYERS, **ex})
    _csv(outd / "functional_experts.csv", rows)
    rows = []
    for k, p in sorted(per.items()):
        for v, fv in p["functional"].items():
            for l, lv in fv["layers"].items():
                rows.append({"key": k, "view": v, "layer_block": int(l) + 1, **{kk: vv for kk, vv in lv.items() if kk != "experts"}})
    _csv(outd / "functional_summary.csv", rows)
    rows = []
    for c, cv in con.items():
        for s, d in cv["seeds"].items():
            for m, e in d.items():
                rows.append({"contrast": c, "seed": s, "metric": m, **{kk: json.dumps(vv) if isinstance(vv, list) else vv
                                                                        for kk, vv in e.items()}})
    _csv(outd / "contrasts.csv", rows)
    _csv(outd / "cross_seed.csv", [{"condition": c, **{k: v for k, v in x["observed"].items() if not k.endswith("by_layer")},
                                    **{k: json.dumps(v) if isinstance(v, list) else v for k, v in x.items() if k != "observed"}}
                                   for c, x in xs.items()])


def _mean_or_none(xs):
    return None if any(x is None for x in xs) else float(sum(xs) / len(xs))


def _csv(path, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    path.chmod(0o400)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=".tmp/spec_audit_v1/runs.tsv")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    print("specialization-audit-v1 analysis (CPU)")
    if not guard.require_execution(a.execute, "cpu-model"):
        return
    outd = AUD / "analysis"
    if outd.exists():
        raise guard.GuardError(f"{outd} exists; never overwritten")
    guard.require_disk(3_000_000_000, "specaudit analysis")
    (outd / "raw").mkdir(parents=True)
    guard.register_self("spec_audit_analyze")
    import multiprocessing as mp
    import numpy as np
    t0 = time.time()
    setup_globals()
    inv = json.loads((AUD / "inventory.json").read_text())["checkpoints"]
    runs = dict(l.split("\t") for l in Path(a.runs).read_text().splitlines() if l.strip())
    assert sorted(runs) == sorted(inv), "need all 18 extraction runs"
    for k, rd in runs.items():
        st = json.loads((guard.ROOT / rd / "status.json").read_text())["status"]
        if st != "COMPLETE":
            raise guard.GuardError(f"{rd} {st}")
    jobs = [(k, str(guard.ROOT / runs[k]), inv[k]["segmentation"]) for k in sorted(runs)]
    with mp.get_context("fork").Pool(a.workers) as pool:
        outs = pool.map(analyze_checkpoint, jobs, chunksize=1)
    per = {o["key"]: o for o in outs}
    raws = {k: dict(np.load(outd / "raw" / f"{k}.npz")) for k in per}
    con = contrasts(per, raws)
    xs = cross_seed(per, raws)
    cat = categorize(con)
    for X, Y, _ in COMPARISONS[2:]:
        c = con[f"{X}-{Y}"]
        c["classification"] = {"routing_increase": c["routing_increase"]["met"],
                               "functional_increase": c["functional_increase"]["met"]}
    cond_means = {}
    for c in "ABCDEF":
        ks = [f"s{s}_{c}" for s in range(3)]
        cond_means[c] = {
            "R_primary": float(np.mean([np.mean(per[k]["routing"][PRIMARY_VIEW]["mi_adj"]) for k in ks])),
            "R_tfidf": float(np.mean([np.mean(per[k]["routing"][ROBUST_VIEW]["mi_adj"]) for k in ks])),
            "MI_primary": float(np.mean([np.mean(per[k]["routing"][PRIMARY_VIEW]["mi"]) for k in ks])),
            "NMI_primary": float(np.mean([np.mean(per[k]["routing"][PRIMARY_VIEW]["nmi"]) for k in ks])),
            "probe_macro_f1": float(np.mean([per[k]["probe"][PRIMARY_VIEW]["macro_f1"] for k in ks])),
            "F_primary": _mean_or_none([per[k]["functional"][PRIMARY_VIEW]["F_primary"] for k in ks]),
            "F_tfidf": _mean_or_none([per[k]["functional"][ROBUST_VIEW]["F_primary"] for k in ks])}
    labinfo = json.loads((AUD / "labels" / "labels.json").read_text())
    names = [" / ".join(t[:3]) for t in labinfo["top_terms_discovery_K32"]["enc_K32"]]
    out = {"audit": "specialization-audit-v1", "label": "prospectively frozen post-training audit (designed after the "
           "main CE results were known); not part of the csr-representative-v1 preregistration",
           "prereg_sha256": (guard.ROOT / "docs" / "SPECIALIZATION_AUDIT_V1_PREREG.sha256").read_text().split()[0],
           "per_checkpoint": per, "contrasts": con, "cross_seed": xs, "decision": cat, "condition_means": cond_means,
           "cluster_names_enc_K32_descriptive": names, "primary_view": PRIMARY_VIEW, "robust_view": ROBUST_VIEW,
           "primary_layers_blocks": [l + 1 for l in PRIMARY_LAYERS], "seconds": time.time() - t0, "created": guard.now(),
           "note": "document-bootstrap intervals measure audit-document uncertainty only, not training-seed uncertainty"}
    guard.atomic_write_json(outd / "analysis.json", out, mode=0o400)
    write_csvs(per, con, xs, outd)
    for f in (outd / "raw").iterdir():
        f.chmod(0o400)
    guard.ledger("SPEC_AUDIT_ANALYSIS", f"artifacts/specialization-audit-v1/analysis/analysis.json category={cat['category']} "
                                        f"({cat['name']}); sha256={guard.sha256_file(outd / 'analysis.json')}")
    print(json.dumps({"decision": cat, "condition_means": cond_means}, indent=1))


if __name__ == "__main__":
    main()
