"""lfr specialisation audit (protocol 12; frozen in spec_audit_config.json). Routing metrics, functional
specialisation index (FSI) with difficulty matching, shuffled-label control, load-preserving shuffled-routing null.
Labels (domain / e5 latent clusters) are evaluation-only and never enter a model."""
from __future__ import annotations

import math

import numpy as np

from .lfr_eval import evaluate, hungarian

N_DEC = 10
SHUF_SEED = 7


def token_labels(per_token, split, label_system, windows=None):
    """Per-token labels for the evaluated tokens. domain: document domain; cluster: e5 window cluster
    (windows npz: doc, start, cluster). Positions are input positions (labels of the token being read)."""
    if label_system == "domain":
        return per_token["dom"].astype(np.int64), 5
    doc, pos = per_token["gdoc"].astype(np.int64), per_token["pos"].astype(np.int64)
    wd, ws, wc = (np.asarray(windows["doc"], np.int64), np.asarray(windows["start"], np.int64),
                  np.asarray(windows["cluster"], np.int64))
    o = np.lexsort((ws, wd))
    wd, ws, wc = wd[o], ws[o], wc[o]
    wdocs, wfirst, wcnt = np.unique(wd, return_index=True, return_counts=True)
    wmap = {int(d): (int(f), int(c)) for d, f, c in zip(wdocs, wfirst, wcnt)}
    to = np.argsort(doc, kind="stable")
    tdocs, tfirst, tcnt = np.unique(doc[to], return_index=True, return_counts=True)
    lab = np.empty(len(doc), np.int64)
    for d, f, c in zip(tdocs, tfirst, tcnt):
        idx = to[f:f + c]
        wf, wn = wmap[int(d)]
        k = np.searchsorted(ws[wf:wf + wn], pos[idx], side="right") - 1
        lab[idx] = wc[wf + np.maximum(k, 0)]
    return lab, int(np.asarray(windows["cluster"]).max()) + 1


def shuffled_labels(per_token, lab, label_system, windows=None):
    """Label control: a fixed permutation of document domains (domain) or of window clusters (cluster)."""
    rng = np.random.default_rng(SHUF_SEED)
    if label_system == "domain":
        docs = np.unique(per_token["gdoc"])
        dom_of = {}
        for g, l_ in zip(per_token["gdoc"], lab):
            dom_of.setdefault(int(g), int(l_))
        vals = np.asarray([dom_of[int(d)] for d in docs])
        perm = dict(zip(docs.tolist(), vals[rng.permutation(len(vals))].tolist()))
        return np.asarray([perm[int(g)] for g in per_token["gdoc"]], np.int64)
    w = dict(windows)
    w["cluster"] = windows["cluster"][rng.permutation(len(windows["cluster"]))]
    return token_labels(per_token, None, "cluster", w)[0]


def joint_counts(topi, lab, K, E):
    """topi (N, L, 4), lab (N,) -> counts (L, K, E) of (topic, selected expert) pairs."""
    N, L, k = topi.shape
    out = np.zeros((L, K, E))
    for li in range(L):
        idx = lab.repeat(k) * E + topi[:, li, :].reshape(-1).astype(np.int64)
        out[li] = np.bincount(idx, minlength=K * E).reshape(K, E)
    return out


def mutual_info(c):
    p = c / max(1.0, c.sum())
    pk, pe = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    m = p > 0
    return float((p[m] * np.log(p[m] / (pk @ pe)[m])).sum())


def routing_metrics(topi, lab, K, E, null_seed=0):
    L = topi.shape[1]
    C = joint_counts(topi, lab, K, E)
    rng = np.random.default_rng(null_seed)
    shuf = topi[rng.permutation(topi.shape[0])]  # load-preserving null: expert sets permuted across tokens
    Cn = joint_counts(shuf, lab, K, E)
    mi = [mutual_info(C[li]) for li in range(L)]
    mi_null = [mutual_info(Cn[li]) for li in range(L)]
    prof = C / np.maximum(1, C.sum(1, keepdims=True))  # P(topic | expert) (L, K, E)
    ent = -(prof * np.log(np.clip(prof, 1e-12, 1))).sum(1) / math.log(K)  # (L, E)
    load = C.sum(1) / np.maximum(1, C.sum((1, 2)))[:, None]
    return {"mi": mi, "mi_mean": float(np.mean(mi)), "mi_null": mi_null, "mi_null_mean": float(np.mean(mi_null)),
            "topic_entropy_per_expert": ent.tolist(), "topic_entropy_mean": float(ent.mean()),
            "profiles": prof.tolist(), "load": load.tolist(), "load_min": float(load.min()),
            "load_max": float(load.max()), "n_below_0.1_uniform": int((load < 0.1 / E).sum())}


def window_hist(per_token, topi, E, lab, wins_of_token, K):
    """Per-window normalized (layer, expert) selection histograms and window labels (majority label)."""
    w = wins_of_token
    uw, inv = np.unique(w, return_inverse=True)
    N, L, k = topi.shape
    H = np.zeros((len(uw), L * E), np.float32)
    for li in range(L):
        idx = inv.repeat(k) * (L * E) + li * E + topi[:, li, :].reshape(-1).astype(np.int64)
        H += np.bincount(idx, minlength=len(uw) * L * E).reshape(len(uw), L * E)
    H /= np.maximum(1, H.sum(1, keepdims=True))
    wl = np.bincount(inv * K + lab, minlength=len(uw) * K).reshape(len(uw), K).argmax(1)
    return H, wl


def topic_prediction_acc(Hd, yd, Hm, ym, K):
    cen = np.stack([Hd[yd == k].mean(0) if (yd == k).any() else np.zeros(Hd.shape[1]) for k in range(K)])
    cen /= np.maximum(1e-9, np.linalg.norm(cen, axis=1, keepdims=True))
    Hn = Hm / np.maximum(1e-9, np.linalg.norm(Hm, axis=1, keepdims=True))
    return float(((Hn @ cen.T).argmax(1) == ym).mean())


def within_expert_similarity(Hm_counts_by_expert, emb):
    """Hm_counts_by_expert (W, L*E) raw selection counts per window; emb (W, D) normalized window embeddings.
    Returns mean over (layer, expert) of the selection-weighted mean cosine to the expert centroid, minus the same
    quantity for the global centroid."""
    Wt = Hm_counts_by_expert
    g = emb.mean(0)
    g /= max(1e-9, np.linalg.norm(g))
    base = float((Wt.sum(1) * (emb @ g)).sum() / max(1e-9, Wt.sum()))
    sims = []
    for j in range(Wt.shape[1]):
        w = Wt[:, j]
        if w.sum() <= 0:
            continue
        c = (w[:, None] * emb).sum(0)
        c /= max(1e-9, np.linalg.norm(c))
        sims.append(float((w * (emb @ c)).sum() / w.sum()))
    return float(np.mean(sims) - base) if sims else 0.0


def assign_topics(topi_disc, lab_disc, K, E):
    C = joint_counts(topi_disc, lab_disc, K, E)
    pk = C.sum((0, 2))
    pk = pk / max(1, pk.sum())
    groups = {k: {} for k in range(K)}
    for li in range(C.shape[0]):
        pe = C[li].sum(0)
        for e in range(E):
            if pe[e] <= 0:
                continue
            lift = (C[li, :, e] / pe[e]) / np.maximum(1e-12, pk)
            k = int(lift.argmax())
            if lift[k] > 1.0:
                groups[k].setdefault(li, []).append(e)
    return groups


def fsi_from_deltas(base_ce, deltas, lab, K, groups, n_moe):
    """deltas {k: per-token dCE when G_k disabled}. Difficulty matching by baseline-CE deciles."""
    dec = np.minimum((np.argsort(np.argsort(base_ce)) * N_DEC) // len(base_ce), N_DEC - 1)
    I, O, X, A = [], [], [], []
    for k in range(K):
        nk = sum(len(v) for v in groups[k].values())
        a_k = max(1, nk) / n_moe
        if nk == 0 or k not in deltas or not (lab == k).any():
            I.append(0.0)
            O.append(0.0)
            X.append(0.0)
            A.append(a_k)
            continue
        d = deltas[k]
        own = lab == k
        o_k = float(d[own].mean())
        wk = np.bincount(dec[own], minlength=N_DEC) / own.sum()
        oth = ~own
        xm = np.asarray([d[oth & (dec == j)].mean() if (oth & (dec == j)).any() else 0.0 for j in range(N_DEC)])
        x_k = float((wk * xm).sum())
        I.append(o_k - x_k)
        O.append(o_k)
        X.append(x_k)
        A.append(a_k)
    fsi = float(np.mean([i / a for i, a in zip(I, A)]))
    return {"fsi": fsi, "fsi_raw": float(np.mean(I)), "I": I, "O": O, "X": X,
            "group_sizes": [sum(len(v) for v in groups[k].values()) for k in range(K)]}


def functional(model, split, rows_meas, device, base, lab, K, groups, eval_kw=None):
    deltas = {}
    for k in range(K):
        if not groups[k]:
            continue
        r = evaluate(model, split, rows_meas, device, exclude=groups[k], per_token=True, routing=False,
                     **(eval_kw or {}))
        deltas[k] = r["per_token"]["ce"] - base["ce"]
    return fsi_from_deltas(base["ce"], deltas, lab, K, groups, model.n_moe)


def audit(model, split_disc, rows_disc, split_meas, rows_meas, device, windows_disc=None, windows_meas=None,
          systems=("domain", "cluster")):
    """Full audit of one checkpoint. Returns dict per label system."""
    E = model.cfg.n_experts
    disc = evaluate(model, split_disc, rows_disc, device, per_token=True)["per_token"]
    meas_full = evaluate(model, split_meas, rows_meas, device, per_token=True)
    meas = meas_full["per_token"]
    out = {"ce_meas": meas_full["ce"], "load": meas_full["load"], "routing_entropy": meas_full["ent"]}
    for sysn in systems:
        if sysn == "cluster" and (windows_disc is None or windows_meas is None):
            continue
        ld, K = token_labels(disc, split_disc, sysn, windows_disc)
        lm, _ = token_labels(meas, split_meas, sysn, windows_meas)
        rm = routing_metrics(meas["topi"], lm, K, E)
        groups = assign_topics(disc["topi"], ld, K, E)
        fs = functional(model, split_meas, rows_meas, device, meas, lm, K, groups)
        lds = shuffled_labels(disc, ld, sysn, windows_disc)
        lms = shuffled_labels(meas, lm, sysn, windows_meas)
        gs = assign_topics(disc["topi"], lds, K, E)
        fss = functional(model, split_meas, rows_meas, device, meas, lms, K, gs)
        res = {"routing": rm, "functional": fs, "fsi": fs["fsi"], "fsi_shuf": fss["fsi"],
               "functional_shuffled": fss, "groups": {str(k): {str(l_): v for l_, v in g.items()}
                                                      for k, g in groups.items()}}
        # topic prediction from selected experts (256-token windows: doc*10000 + pos//256)
        wd = disc["gdoc"].astype(np.int64) * 10000 + disc["pos"].astype(np.int64) // 256
        wm = meas["gdoc"].astype(np.int64) * 10000 + meas["pos"].astype(np.int64) // 256
        Hd, yd = window_hist(disc, disc["topi"], E, ld, wd, K)
        Hm, ym = window_hist(meas, meas["topi"], E, lm, wm, K)
        res["topic_pred_acc"] = topic_prediction_acc(Hd, yd, Hm, ym, K)
        res["topic_pred_chance"] = float(np.bincount(ym, minlength=K).max() / len(ym))
        if sysn == "cluster" and windows_meas is not None and "emb" in windows_meas:
            # within-expert semantic similarity on measurement windows (labeler windows)
            emb = windows_meas["emb"].astype(np.float32)
            key = windows_meas["doc"].astype(np.int64) * 10000 + windows_meas["start"].astype(np.int64) // 256
            pos_of = {int(k_): i for i, k_ in enumerate(key)}
            sel = np.asarray([pos_of.get(int(k_), -1) for k_ in wm])
            ok = sel >= 0
            L = meas["topi"].shape[1]
            cnt = np.zeros((len(emb), L * E), np.float32)
            for li in range(L):
                for kk in range(meas["topi"].shape[2]):
                    np.add.at(cnt, (sel[ok], li * E + meas["topi"][ok, li, kk].astype(np.int64)), 1.0)
            res["within_expert_similarity"] = within_expert_similarity(cnt, emb)
        out[sysn] = res
    return out


def cross_seed_stability(prof_a, prof_b, null_seed=0):
    """prof (L, K, E) topic profiles of two seeds. Per layer: Hungarian alignment of experts maximizing profile
    correlation; returns mean matched correlation and a null with seed-b experts' profiles from a random topic
    permutation (destroys topic identity, keeps profile shapes)."""
    A, B = np.asarray(prof_a), np.asarray(prof_b)
    L, K, E = A.shape

    def matched(A_, B_):
        vals = []
        for li in range(L):
            a = A_[li].T - A_[li].T.mean(1, keepdims=True)
            b = B_[li].T - B_[li].T.mean(1, keepdims=True)
            na = np.linalg.norm(a, axis=1, keepdims=True)
            nb = np.linalg.norm(b, axis=1, keepdims=True)
            corr = (a / np.maximum(1e-12, na)) @ (b / np.maximum(1e-12, nb)).T
            col = hungarian(-corr)
            vals.append(float(corr[np.arange(E), col].mean()))
        return float(np.mean(vals)), vals
    m, per = matched(A, B)
    rng = np.random.default_rng(null_seed)
    Bn = B[:, rng.permutation(K), :]
    mn, _ = matched(A, Bn)
    return {"matched_corr": m, "per_layer": per, "null_matched_corr": mn}
