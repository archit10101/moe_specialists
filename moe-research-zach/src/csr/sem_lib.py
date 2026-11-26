"""csr-semantic-edl-v1 evaluation helpers (new module).

  pseudo_windows      : windows-format pseudo-domain labels (one window per document = argmax of the paper's q)
  label_free_audit    : routing metrics / topic prediction / FSI for windows-format label systems (never reads 'dom')
  per_expert_multi    : fle per-expert equal-compute replacement interventions, computing the own-group vs
                        difficulty-matched other-group interaction for several label systems from the SAME passes;
                        the 'domain' system reproduces fle_evaluate.per_expert exactly (test)."""
from __future__ import annotations

import numpy as np


def pseudo_windows(split, table_npz):
    from .gc_lib import opaque_id
    z = np.load(table_npz)
    idx = {str(o): i for i, o in enumerate(z["opaque"])}
    q = z["q"]
    K = q.shape[1]
    lab = np.full(split.n_docs, K, np.int64)  # K = unassigned (paper without views)
    cache = {}
    for d, p in enumerate(split.papers):
        o = cache.setdefault(p, opaque_id(p))
        i = idx.get(o)
        if i is not None and q[i].sum() > 0:
            lab[d] = int(q[i].argmax())
    return {"doc": np.arange(split.n_docs), "start": np.zeros(split.n_docs, np.int64), "cluster": lab}


def doc_labels(windows):
    return np.asarray(windows["cluster"], np.int64)


def label_free_audit(model, sd, rows_d, sm, rows_m, device, systems, fsi_systems, disc=None, meas_full=None):
    from .lfr_audit import (assign_topics, functional, routing_metrics, token_labels, topic_prediction_acc,
                            window_hist)
    from .lfr_eval import evaluate
    E = model.cfg.n_experts
    if disc is None:
        disc = evaluate(model, sd, rows_d, device, per_token=True)["per_token"]
    mf = meas_full if meas_full is not None else evaluate(model, sm, rows_m, device, per_token=True)
    meas = mf["per_token"]
    ld_ = np.asarray(mf["load"])
    out = {"ce_meas": mf["ce"], "load": mf["load"], "routing_entropy": mf["ent"],
           "load_min_x_uniform": float(ld_.min() * E), "load_max_x_uniform": float(ld_.max() * E),
           "n_below_0.1_uniform": int((ld_ < 0.1 / E).sum())}
    for name, (wd, wm) in systems.items():
        ld, K1 = token_labels(disc, sd, "cluster", wd)
        lm, K2 = token_labels(meas, sm, "cluster", wm)
        K = max(K1, K2)
        rm = routing_metrics(meas["topi"], lm, K, E)
        res = {"K": K, "routing": {k: v for k, v in rm.items() if k not in ("topic_entropy_per_expert", "profiles")},
               "profiles": rm["profiles"], "topic_entropy_per_expert": rm["topic_entropy_per_expert"]}
        wdk = disc["gdoc"].astype(np.int64) * 10000 + disc["pos"].astype(np.int64) // 256
        wmk = meas["gdoc"].astype(np.int64) * 10000 + meas["pos"].astype(np.int64) // 256
        Hd, yd = window_hist(disc, disc["topi"], E, ld, wdk, K)
        Hm, ym = window_hist(meas, meas["topi"], E, lm, wmk, K)
        res["topic_pred_acc"] = topic_prediction_acc(Hd, yd, Hm, ym, K)
        res["topic_pred_chance"] = float(np.bincount(ym, minlength=K).max() / len(ym))
        if name in fsi_systems:
            groups = assign_topics(disc["topi"], ld, K, E)
            fs = functional(model, sm, rows_m, device, meas, lm, K, groups)
            res.update({"fsi": fs["fsi"], "fsi_raw": fs["fsi_raw"], "functional": fs,
                        "groups": {str(k): {str(l_): v for l_, v in g.items()} for k, g in groups.items()}})
        out[name] = res
    return out


def per_expert_multi(model, sd, rows_d, sm, rows_pe, device, disc, systems):
    """systems: {name: (lab_disc (Nd,), lab_of_meas_fn(per_token)->(N,), K)}. One intervention pass per (layer,
    expert); returns per system the fle per-expert record (+ generic importance)."""
    from .lfr_audit import N_DEC, joint_counts
    from .lfr_eval import evaluate
    E, Lm = model.cfg.n_experts, model.n_moe
    base = evaluate(model, sm, rows_pe, device, per_token=True, routing=False)["per_token"]
    bce = base["ce"]
    dec = np.minimum((np.argsort(np.argsort(bce)) * N_DEC) // len(bce), N_DEC - 1)
    prep = {}
    for name, (ld, fn, K) in systems.items():
        C = joint_counts(disc["topi"], np.asarray(ld, np.int64), K, E)
        pk = C.sum((0, 2)) / C.sum()
        prep[name] = (C, pk, np.asarray(fn(base), np.int64), K)
    rec = {n: {"interaction": [], "own": [], "other_matched": [], "kstar": [], "lift_kstar": [], "n_own": [],
               "se_own": [], "mat": np.zeros((Lm, E, prep[n][3]))} for n in systems}
    gen, gen_abs = [], []
    for li in range(Lm):
        for e in range(E):
            r = evaluate(model, sm, rows_pe, device, exclude={li: [e]}, per_token=True, routing=False)
            d = r["per_token"]["ce"] - bce
            gen.append(float(d.mean()))
            gen_abs.append(float(np.abs(d).mean()))
            for name, (C, pk, lab, K) in prep.items():
                pe = C[li, :, e].sum()
                lift = (C[li, :, e] / max(1.0, pe)) / np.maximum(1e-12, pk)
                k = int(lift.argmax())
                for dd in range(K):
                    m_ = lab == dd
                    rec[name]["mat"][li, e, dd] = float(d[m_].mean()) if m_.any() else np.nan
                own = lab == k
                o = float(d[own].mean())
                wk = np.bincount(dec[own], minlength=N_DEC) / max(1, own.sum())
                oth = ~own
                xm = np.asarray([d[oth & (dec == j)].mean() if (oth & (dec == j)).any() else 0.0
                                 for j in range(N_DEC)])
                x = float((wk * xm).sum())
                R = rec[name]
                R["interaction"].append(o - x)
                R["own"].append(o)
                R["other_matched"].append(x)
                R["kstar"].append(k)
                R["lift_kstar"].append(float(lift[k]))
                R["n_own"].append(int(own.sum()))
                R["se_own"].append(float(d[own].std() / np.sqrt(max(1, own.sum()))))
    out = {}
    for name, R in rec.items():
        inter = np.asarray(R["interaction"])
        out[name] = {"interaction": R["interaction"], "mean_interaction": float(inter.mean()),
                     "frac_positive": float((inter > 0).mean()), "own": R["own"], "other_matched": R["other_matched"],
                     "kstar": R["kstar"], "lift_kstar": R["lift_kstar"], "n_own": R["n_own"], "se_own": R["se_own"],
                     "dce_by_group": R["mat"].tolist(), "n_tokens": int(len(bce)),
                     "mean_interaction_by_layer": inter.reshape(Lm, E).mean(1).tolist()}
    out["generic_importance"] = {"mean_dce": gen, "mean_abs_dce": gen_abs}
    return out
