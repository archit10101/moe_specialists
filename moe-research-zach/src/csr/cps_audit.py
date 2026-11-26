"""csr-context-prior-specialization-v2 specialisation audit = the V1 audit (csr.lfr_audit, imported read-only: same
label systems, discovery / measurement rows, load-preserving routing null, topic entropy, topic prediction, within-expert
similarity, difficulty-matched FSI, shuffled-label control) + per-document aggregates of the functional ablation
deltas, which allow (i) a paired document bootstrap of FSI(candidate) - FSI(TOKEN-4) on the same measurement documents
and (ii) a per-topic specialist test (I_k > 0: disabling topic k's inferred specialists harms topic k more than
difficulty-matched other contexts). Document bootstrap is NOT seed replication; both are reported separately."""
from __future__ import annotations

import numpy as np

from .lfr_audit import (N_DEC, assign_topics, routing_metrics, shuffled_labels, token_labels, topic_prediction_acc,
                        window_hist, within_expert_similarity, fsi_from_deltas)
from .lfr_eval import evaluate


def docagg(base_ce, deltas, lab, K, groups, n_moe, gdoc):
    """Per-document sufficient statistics of fsi_from_deltas."""
    dec = np.minimum((np.argsort(np.argsort(base_ce)) * N_DEC) // len(base_ce), N_DEC - 1)
    docs, inv = np.unique(gdoc, return_inverse=True)
    nd = len(docs)
    out = {"docs": docs.tolist(), "K": K, "a": [], "own_sum": [], "own_n": [], "own_dec_n": [], "oth_sum": [],
           "oth_n": [], "has": []}
    for k in range(K):
        nk = sum(len(v) for v in groups[k].values())
        out["a"].append(max(1, nk) / n_moe)
        has = nk > 0 and k in deltas and bool((lab == k).any())
        out["has"].append(bool(has))
        if not has:
            for f in ("own_sum", "own_n"):
                out[f].append(np.zeros(nd).tolist())
            for f in ("own_dec_n", "oth_sum", "oth_n"):
                out[f].append(np.zeros((nd, N_DEC)).tolist())
            continue
        d = deltas[k]
        own = lab == k
        idx_own = inv[own]
        out["own_sum"].append(np.bincount(idx_own, weights=d[own], minlength=nd).tolist())
        out["own_n"].append(np.bincount(idx_own, minlength=nd).astype(float).tolist())
        out["own_dec_n"].append(np.bincount(idx_own * N_DEC + dec[own], minlength=nd * N_DEC)
                                .reshape(nd, N_DEC).astype(float).tolist())
        oth = ~own
        j = inv[oth] * N_DEC + dec[oth]
        out["oth_sum"].append(np.bincount(j, weights=d[oth], minlength=nd * N_DEC).reshape(nd, N_DEC).tolist())
        out["oth_n"].append(np.bincount(j, minlength=nd * N_DEC).reshape(nd, N_DEC).astype(float).tolist())
    return out


def fsi_docagg(ag, w=None):
    """FSI and I_k from per-document aggregates with document weights w (None = 1: point estimate)."""
    K = ag["K"]
    nd = len(ag["docs"])
    w = np.ones(nd) if w is None else np.asarray(w, float)
    I = []
    for k in range(K):
        if not ag["has"][k]:
            I.append(0.0)
            continue
        on = np.asarray(ag["own_n"][k]) @ w
        if on <= 0:
            I.append(0.0)
            continue
        o = (np.asarray(ag["own_sum"][k]) @ w) / on
        wk = (np.asarray(ag["own_dec_n"][k]).T @ w) / on
        osum, on2 = np.asarray(ag["oth_sum"][k]).T @ w, np.asarray(ag["oth_n"][k]).T @ w
        xm = np.where(on2 > 0, osum / np.maximum(on2, 1e-12), 0.0)
        I.append(float(o - (wk * xm).sum()))
    fsi = float(np.mean([i / a for i, a in zip(I, ag["a"])]))
    return fsi, I


def paired_bootstrap(ag_a, ag_b, n_boot=2000, seed=20261112):
    """FSI(a) - FSI(b) over the same measurement documents: point, 95% percentile CI, P(diff <= 0)."""
    assert ag_a["docs"] == ag_b["docs"], "paired bootstrap needs identical measurement documents"
    nd = len(ag_a["docs"])
    rng = np.random.default_rng(seed)
    fa, _ = fsi_docagg(ag_a)
    fb, _ = fsi_docagg(ag_b)
    diffs = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, nd, nd), minlength=nd)
        diffs.append(fsi_docagg(ag_a, w)[0] - fsi_docagg(ag_b, w)[0])
    diffs = np.asarray(diffs)
    return {"fsi_a": fa, "fsi_b": fb, "diff": fa - fb, "ci95": [float(np.quantile(diffs, 0.025)),
                                                              float(np.quantile(diffs, 0.975))],
            "p_le_0": float((diffs <= 0).mean()), "n_boot": n_boot, "n_docs": nd}


def specialist_test(ag, n_boot=2000, seed=20261113):
    """Per topic: I_k point and bootstrap 95% CI; a topic has a functional specialist group iff CI lower bound > 0."""
    nd = len(ag["docs"])
    rng = np.random.default_rng(seed)
    _, I = fsi_docagg(ag)
    B = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, nd, nd), minlength=nd)
        B.append(fsi_docagg(ag, w)[1])
    B = np.asarray(B)
    lo, hi = np.quantile(B, 0.025, axis=0), np.quantile(B, 0.975, axis=0)
    sig = [bool(ag["has"][k] and lo[k] > 0) for k in range(ag["K"])]
    return {"I": I, "ci_lo": lo.tolist(), "ci_hi": hi.tolist(), "significant": sig, "n_significant": int(sum(sig))}


def functional_v2(model, split, rows_meas, device, base, lab, K, groups):
    deltas = {}
    for k in range(K):
        if not groups[k]:
            continue
        r = evaluate(model, split, rows_meas, device, exclude=groups[k], per_token=True, routing=False)
        deltas[k] = r["per_token"]["ce"] - base["ce"]
    fs = fsi_from_deltas(base["ce"], deltas, lab, K, groups, model.n_moe)
    fs["docagg"] = docagg(base["ce"], deltas, lab, K, groups, model.n_moe, base["gdoc"])
    return fs


def audit_v2(model, split_disc, rows_disc, split_meas, rows_meas, device, windows_disc=None, windows_meas=None,
             systems=("domain", "cluster")):
    """V1 audit (identical computations) + document aggregates. Returns dict per label system."""
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
        fs = functional_v2(model, split_meas, rows_meas, device, meas, lm, K, groups)
        lds = shuffled_labels(disc, ld, sysn, windows_disc)
        lms = shuffled_labels(meas, lm, sysn, windows_meas)
        gs = assign_topics(disc["topi"], lds, K, E)
        fss = functional_v2(model, split_meas, rows_meas, device, meas, lms, K, gs)
        res = {"routing": rm, "functional": fs, "fsi": fs["fsi"], "fsi_shuf": fss["fsi"],
               "functional_shuffled": fss, "groups": {str(k): {str(l_): v for l_, v in g.items()}
                                                      for k, g in groups.items()},
               "specialists": specialist_test(fs["docagg"])}
        wd = disc["gdoc"].astype(np.int64) * 10000 + disc["pos"].astype(np.int64) // 256
        wm = meas["gdoc"].astype(np.int64) * 10000 + meas["pos"].astype(np.int64) // 256
        Hd, yd = window_hist(disc, disc["topi"], E, ld, wd, K)
        Hm, ym = window_hist(meas, meas["topi"], E, lm, wm, K)
        res["topic_pred_acc"] = topic_prediction_acc(Hd, yd, Hm, ym, K)
        res["topic_pred_chance"] = float(np.bincount(ym, minlength=K).max() / len(ym))
        if sysn == "cluster" and windows_meas is not None and "emb" in windows_meas:
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
