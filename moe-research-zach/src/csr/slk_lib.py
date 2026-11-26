"""csr-semantic-edl-long-adaptive-k-v1 evaluation / analysis helpers (new module; prereg sections 6-7).

  routing_subject   per-layer MI / NMI between a label system and expert selection, load-preserving and
                    document-label-shuffle nulls, load-adjusted per-expert subject entropy, expert subject profiles
  probe_logreg      multinomial logistic-regression probe on complete routing features (fit on one split, evaluated
                    on another; torch LBFGS, L2; standardisation fitted on the training split only)
  per_expert_boot   equal-compute single-expert replacement interventions (fle protocol) with an expert's subject
                    determined on calibration discovery tokens only, own-subject vs difficulty- AND position-matched
                    other-subject dCE, per-document sufficient statistics and document-bootstrap CIs
  match_profiles    permutation-aware cross-seed expert matching (Hungarian on profile correlation)
  build_hierarchy   nested hierarchy from a fine assignment table (Ward merge of q-weighted centroids)
  shuffle_table     paper-permutation control (exact preservation of cluster masses and confidence distribution)"""
from __future__ import annotations

import math

import numpy as np

N_DEC = 10
POS_EDGES = (128, 512)  # position bins: [0,128), [128,512), [512,1024)


def _mi(c):
    p = c / max(1.0, c.sum())
    pk, pe = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    m = p > 0
    mi = float((p[m] * np.log(p[m] / (pk @ pe)[m])).sum())
    hk = float(-(pk[pk > 0] * np.log(pk[pk > 0])).sum())
    he = float(-(pe[pe > 0] * np.log(pe[pe > 0])).sum())
    return mi, hk, he


def joint(topi, lab, K, E):
    N, L, k = topi.shape
    out = np.zeros((L, K, E))
    for li in range(L):
        idx = lab.repeat(k) * E + topi[:, li, :].reshape(-1).astype(np.int64)
        out[li] = np.bincount(idx, minlength=K * E).reshape(K, E)
    return out


def routing_subject(topi, lab, gdoc, K, E, seed=0):
    """topi (N,L,4) uint8; lab (N,) int; gdoc (N,) document ids. Returns per-layer and mean metrics."""
    lab = np.asarray(lab, np.int64)
    C = joint(topi, lab, K, E)
    rng = np.random.default_rng(seed)
    Cn = joint(topi[rng.permutation(topi.shape[0])], lab, K, E)  # load-preserving routing shuffle
    docs, inv = np.unique(gdoc, return_inverse=True)
    lab_of_doc = np.zeros(len(docs), np.int64)
    lab_of_doc[inv] = lab
    labs_perm = lab_of_doc[rng.permutation(len(docs))][inv]  # document-label shuffle (keeps doc structure)
    Cd = joint(topi, labs_perm, K, E)
    L = C.shape[0]
    mi, nmi, mi_n, nmi_n, mi_d, nmi_d = [], [], [], [], [], []
    for li in range(L):
        a, hk, he = _mi(C[li])
        b, hkb, heb = _mi(Cn[li])
        d_, hkd, hed = _mi(Cd[li])
        mi.append(a), nmi.append(2 * a / max(1e-12, hk + he))
        mi_n.append(b), nmi_n.append(2 * b / max(1e-12, hkb + heb))
        mi_d.append(d_), nmi_d.append(2 * d_ / max(1e-12, hkd + hed))
    pk = C.sum((0, 2)) / max(1.0, C.sum())
    load = C.sum(1) / np.maximum(1, C.sum((1, 2)))[:, None]  # (L,E)
    pke = C / np.maximum(1, C.sum(1, keepdims=True))  # p(k|e) (L,K,E)
    lift = pke / np.maximum(1e-12, pk)[None, :, None]
    lift_n = lift / np.maximum(1e-12, lift.sum(1, keepdims=True))
    ent = -(lift_n * np.log(np.clip(lift_n, 1e-12, 1))).sum(1) / math.log(K)  # load-adjusted (prior-normalised)
    return {"mi": mi, "nmi": nmi, "mi_null_route": mi_n, "nmi_null_route": nmi_n, "mi_null_doclabel": mi_d,
            "nmi_null_doclabel": nmi_d, "mi_mean": float(np.mean(mi)), "nmi_mean": float(np.mean(nmi)),
            "nmi_null_route_mean": float(np.mean(nmi_n)), "nmi_null_doclabel_mean": float(np.mean(nmi_d)),
            "mi_null_route_mean": float(np.mean(mi_n)), "mi_null_doclabel_mean": float(np.mean(mi_d)),
            "subject_entropy_load_adj_mean": float(ent.mean()),
            "subject_entropy_load_adj_wmean": float((ent * load).sum() / load.sum()),
            "subject_entropy_by_layer": ent.mean(1).tolist(), "load": load.tolist(),
            "load_min_x_uniform": float(load.min() * E), "load_max_x_uniform": float(load.max() * E),
            "n_below_0.1_uniform": int((load < 0.1 / E).sum()), "p_subject": pk.tolist(),
            "profiles_lift": lift.transpose(0, 2, 1).tolist(),  # (L,E,K)
            "max_lift_mean": float(lift.max(1).mean())}


def probe_logreg(Xtr, ytr, Xte, yte, K, l2=1e-3, iters=200, seed=0):
    import torch
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    A = torch.as_tensor((Xtr - mu) / sd, dtype=torch.float64)
    B = torch.as_tensor((Xte - mu) / sd, dtype=torch.float64)
    y = torch.as_tensor(np.asarray(ytr, np.int64))
    W = torch.zeros(A.shape[1], K, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(K, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(A @ W + b, y) + l2 * (W ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        pred = (B @ W + b).argmax(1).numpy()
    yte = np.asarray(yte, np.int64)
    acc = float((pred == yte).mean())
    rec = [float((pred[yte == k] == k).mean()) for k in range(K) if (yte == k).any()]
    return {"acc": acc, "balanced_acc": float(np.mean(rec)), "chance_majority": float(np.bincount(yte).max() / len(yte)),
            "n_train": int(len(ytr)), "n_test": int(len(yte))}


def strata(base_ce, pos):
    dec = np.minimum((np.argsort(np.argsort(base_ce)) * N_DEC) // len(base_ce), N_DEC - 1)
    pb = np.digitize(np.asarray(pos, np.int64), POS_EDGES)
    return dec * (len(POS_EDGES) + 1) + pb, dec


def interaction_from_stats(own_s, own_c, oth_s, oth_c, w=None):
    """own_s (D,) own dCE sums per doc; own_c (D,S) own counts per doc/stratum; oth_s/oth_c (D,S). w (B,D) doc weights.
    interaction = mean own dCE - sum_j p_own(j) * mean other dCE in stratum j (strata empty on either side dropped,
    own weights renormalised)."""
    if w is None:
        w = np.ones((1, own_s.shape[0]))
    os_ = w @ own_s
    oc = w @ own_c  # (B,S)
    xs, xc = w @ oth_s, w @ oth_c
    ok = (oc > 0) & (xc > 0)
    pj = np.where(ok, oc, 0.0)
    pj = pj / np.maximum(1e-12, pj.sum(1, keepdims=True))
    xm = np.where(ok, xs / np.maximum(1e-12, xc), 0.0)
    own_mean = os_ / np.maximum(1e-12, oc.sum(1))
    return own_mean - (pj * xm).sum(1)


def per_expert_boot(model, sd, rows_d, sm, rows_pe, device, disc, systems, n_boot=1000, seed=20261118,
                    lift_min=1.5):
    """systems: {name: (lab_disc (Nd,), lab_of_meas_fn(per_token)->(N,), K)}. One replacement pass per (layer,
    expert) shared by all systems. Expert subject k*(l,e) = argmax lift on discovery (calibration) tokens only.
    Returns per system: per-expert interaction (primary: CE-decile x position-bin matched; also decile-only),
    own / other-matched dCE, per-expert bootstrap CI, condition-level mean with document-bootstrap CI, expert load on
    the measurement tokens, generic importance."""
    from .lfr_eval import evaluate
    E, Lm = model.cfg.n_experts, model.n_moe
    base = evaluate(model, sm, rows_pe, device, per_token=True, routing=True)
    bpt = base["per_token"]
    bce = bpt["ce"]
    S_all, dec = strata(bce, bpt["pos"])
    nS = N_DEC * (len(POS_EDGES) + 1)
    docs, dinv = np.unique(bpt["gdoc"], return_inverse=True)
    D = len(docs)
    rng = np.random.default_rng(seed)
    W = rng.multinomial(D, np.ones(D) / D, size=n_boot).astype(np.float64)
    Ld = np.asarray(base["load"])
    prep = {}
    for name, (ld, fn, K) in systems.items():
        C = joint(disc["topi"], np.asarray(ld, np.int64), K, E)
        pk = C.sum((0, 2)) / C.sum()
        prep[name] = (C, pk, np.asarray(fn(bpt), np.int64), K)
    rec = {n: {"inter": [], "inter_dec": [], "own": [], "oth": [], "kstar": [], "lift": [], "n_own": [], "ci": [],
               "boot": np.zeros((n_boot, Lm * E))} for n in systems}
    gen = []
    j = 0
    for li in range(Lm):
        for e in range(E):
            r = evaluate(model, sm, rows_pe, device, exclude={li: [e]}, per_token=True, routing=False)
            d = r["per_token"]["ce"] - bce
            gen.append(float(d.mean()))
            for name, (C, pk, lab, K) in prep.items():
                pe = C[li, :, e].sum()
                lift = (C[li, :, e] / max(1.0, pe)) / np.maximum(1e-12, pk)
                k = int(lift.argmax())
                own = lab == k
                R = rec[name]
                if not own.any():
                    R["inter"].append(float("nan")), R["inter_dec"].append(float("nan"))
                    R["own"].append(float("nan")), R["oth"].append(float("nan"))
                    R["kstar"].append(k), R["lift"].append(float(lift[k])), R["n_own"].append(0)
                    R["ci"].append([float("nan"), float("nan")])
                    R["boot"][:, j] = np.nan
                    continue
                oth = ~own
                own_s = np.bincount(dinv[own], weights=d[own], minlength=D)
                own_c = np.bincount(dinv[own] * nS + S_all[own], minlength=D * nS).reshape(D, nS).astype(float)
                oth_s = np.bincount(dinv[oth] * nS + S_all[oth], weights=d[oth], minlength=D * nS).reshape(D, nS)
                oth_c = np.bincount(dinv[oth] * nS + S_all[oth], minlength=D * nS).reshape(D, nS).astype(float)
                pt_ = float(interaction_from_stats(own_s, own_c, oth_s, oth_c)[0])
                bs = interaction_from_stats(own_s, own_c, oth_s, oth_c, W)
                # decile-only matching (fle definition) for comparability
                wk = np.bincount(dec[own], minlength=N_DEC) / own.sum()
                xm = np.asarray([d[oth & (dec == q)].mean() if (oth & (dec == q)).any() else 0.0
                                 for q in range(N_DEC)])
                o = float(d[own].mean())
                R["inter"].append(pt_), R["inter_dec"].append(o - float((wk * xm).sum()))
                R["own"].append(o), R["oth"].append(o - pt_)
                R["kstar"].append(k), R["lift"].append(float(lift[k])), R["n_own"].append(int(own.sum()))
                R["ci"].append([float(np.nanquantile(bs, 0.025)), float(np.nanquantile(bs, 0.975))])
                R["boot"][:, j] = bs
            j += 1
    out = {"n_tokens": int(len(bce)), "n_docs": int(D), "n_boot": n_boot, "load": Ld.tolist(),
           "generic_importance_mean_dce": gen, "matching": "baseline-CE decile x position bin " + str(POS_EDGES)}
    load_flat = Ld.reshape(-1)
    for name, R in rec.items():
        inter = np.asarray(R["inter"], float)
        ok = np.isfinite(inter)
        bmean = np.nanmean(R["boot"], 1)
        lifts = np.asarray(R["lift"])
        ci = np.asarray(R["ci"])
        spec = ok & (lifts >= lift_min) & (ci[:, 0] > 0)
        out[name] = {"interaction": R["inter"], "interaction_decile_only": R["inter_dec"], "own": R["own"],
                     "other_matched": R["oth"], "kstar": R["kstar"], "lift_kstar": R["lift"], "n_own": R["n_own"],
                     "ci95": R["ci"], "mean_interaction": float(inter[ok].mean()),
                     "mean_interaction_ci95": [float(np.quantile(bmean, 0.025)), float(np.quantile(bmean, 0.975))],
                     "mean_interaction_decile_only": float(np.nanmean(R["inter_dec"])),
                     "frac_positive": float((inter[ok] > 0).mean()), "n_defined": int(ok.sum()),
                     "n_subject_specialized": int(spec.sum()), "frac_subject_specialized": float(spec.mean()),
                     "by_layer": [float(np.nanmean(x)) for x in inter.reshape(Lm, E)],
                     "corr_interaction_load": float(np.corrcoef(inter[ok], load_flat[ok])[0, 1]),
                     "boot_mean_samples": bmean.tolist()}
    return out


def match_profiles(Pa, Pb):
    """Pa, Pb (E,K) expert subject profiles (lift). Hungarian match maximising correlation; returns mean matched
    correlation and the matching."""
    from .lfr_eval import hungarian
    A = Pa - Pa.mean(1, keepdims=True)
    B = Pb - Pb.mean(1, keepdims=True)
    A /= np.maximum(1e-12, np.linalg.norm(A, axis=1, keepdims=True))
    B /= np.maximum(1e-12, np.linalg.norm(B, axis=1, keepdims=True))
    R = A @ B.T
    col = hungarian(-R)
    return float(R[np.arange(len(col)), col].mean()), col.tolist()


def shuffle_table(opaque, q, seed):
    """Paper permutation over papers that have an assignment: one persistent assignment per paper; the multiset of
    q rows (hence cluster masses and confidence distribution) is preserved exactly."""
    has = np.nonzero(q.sum(1) > 0)[0]
    perm = np.random.default_rng(seed).permutation(has)
    qs = q.copy()
    qs[has] = q[perm]
    return qs


def sharpen(q, T):
    if T == 1.0:
        return q.copy()
    z = np.where(q > 0, q, 0.0) ** (1.0 / T)
    s = z.sum(1, keepdims=True)
    return np.where(s > 0, z / np.maximum(1e-30, s), 0.0)


def build_hierarchy(q_fine, X_paper, levels=(16, 8, 4)):
    """Nested hierarchy: fine clusters merged by Ward linkage on q-weighted centroids of paper embeddings X_paper
    (P,d). Returns {K: merge map fine->level index} with exact nesting (each level is a coarsening of the next)."""
    Kf = q_fine.shape[1]
    w = q_fine.sum(0)
    cen = (q_fine.T @ X_paper) / np.maximum(1e-12, w)[:, None]
    groups = [[i] for i in range(Kf)]
    gw = list(w)
    gc = [cen[i] for i in range(Kf)]
    maps = {}
    while len(groups) > min(levels):
        best = None
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                cost = gw[a] * gw[b] / (gw[a] + gw[b]) * float(((gc[a] - gc[b]) ** 2).sum())
                if best is None or cost < best[0]:
                    best = (cost, a, b)
        _, a, b = best
        nw = gw[a] + gw[b]
        gc[a] = (gw[a] * gc[a] + gw[b] * gc[b]) / nw
        gw[a] = nw
        groups[a] = groups[a] + groups[b]
        del groups[b], gw[b], gc[b]
        if len(groups) in levels:
            m = np.zeros(Kf, np.int64)
            for gi, g in enumerate(groups):
                m[g] = gi
            maps[len(groups)] = m
    return maps


def aggregate(q_fine, m, K):
    out = np.zeros((q_fine.shape[0], K))
    for i, g in enumerate(m):
        out[:, g] += q_fine[:, i]
    return out
