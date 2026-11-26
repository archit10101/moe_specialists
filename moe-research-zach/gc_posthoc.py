#!/usr/bin/env python
"""GC-EDL-v1 POST-HOC analysis with unsealed domain labels (requires UNSEALED.json; never feeds back; descriptive).

Addendum sections 1 (post-hoc part), 2 (representatives) and 8 (grammar vs subject):
  * real-domain predictability (ridge one-vs-rest classifier, DISC-FIT -> DISC-HOLD) from SEM / GRAD / COMBINED /
    raw gradient direction, paper-mean and view-level; confusion matrices
  * NMI / AMI of every selected clustering (base + revision 1) with the domain; cluster x domain contingencies
  * what predicts GRAD cluster identity and the functional fingerprint itself: domain vs semantic embedding vs
    equation/code/citation density vs token-frequency statistics vs scout CE vs position vs formatting proxies
  * gradient vs semantic nearest neighbours: shared domain, CE difference, formatting distance, lexical overlap
  * representative opaque paper IDs / view offsets per cluster
Output: artifacts/gc-edl-v1/diagnostics/posthoc.json + ledger rows (kind post-hoc)."""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402


def nmi_ami(a, b):
    import numpy as np
    from math import lgamma
    a = np.asarray(a)
    b = np.asarray(b)
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    C = np.zeros((len(ua), len(ub)))
    np.add.at(C, (ia, ib), 1)
    n = C.sum()
    r, c = C.sum(1), C.sum(0)
    nz = C > 0
    mi = float((C[nz] / n * np.log(C[nz] * n / np.outer(r, c)[nz])).sum())
    ha = float(-(r / n * np.log(r / n)).sum())
    hb = float(-(c / n * np.log(c / n)).sum())
    nmi = mi / max(1e-12, math.sqrt(ha * hb))
    N = int(n)
    lg = [lgamma(i + 1) for i in range(N + 1)]
    emi = 0.0
    for ai in r.astype(int):
        for bj in c.astype(int):
            lo, hi = max(1, ai + bj - N), min(ai, bj)
            for k in range(lo, hi + 1):
                lp = (lg[ai] + lg[bj] + lg[N - ai] + lg[N - bj] - lg[N] - lg[k] - lg[ai - k] - lg[bj - k]
                      - lg[N - ai - bj + k])
                emi += k / N * math.log(N * k / (ai * bj)) * math.exp(lp)
    ami = (mi - emi) / max(1e-12, 0.5 * (ha + hb) - emi)
    return nmi, ami, C


def ridge_cls(Xtr, ytr, Xte, yte, K, alpha=1.0):
    import numpy as np
    from csr.gc_lib import ridge_fit, ridge_pred
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Y = np.eye(K)[ytr]
    Wb = ridge_fit((Xtr - mu) / sd, Y, alpha * len(Xtr) / 1000)
    pred = ridge_pred(Wb, (Xte - mu) / sd).argmax(1)
    conf = np.zeros((K, K), int)
    np.add.at(conf, (yte, pred), 1)
    return float((pred == yte).mean()), conf


def multi_r2(Xtr, Ytr, Xte, Yte, alpha=10.0):
    import numpy as np
    from csr.gc_lib import ridge_fit, ridge_pred
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Wb = ridge_fit((Xtr - mu) / sd, Ytr, alpha)
    P = ridge_pred(Wb, (Xte - mu) / sd)
    return float(1 - ((Yte - P) ** 2).sum() / ((Yte - Ytr.mean(0)) ** 2).sum())


def main():
    guard.check_root()
    import numpy as np
    from csr import gc_diag
    from csr.fle_lib import decode_batch
    from csr.fsp_lib import Split
    from csr.gc_lib import NUIS_NAMES, opaque_id, paper_q
    from csr.gc_paths import ART, LAT
    if not (ART / "UNSEALED.json").exists():
        raise guard.GuardError("labels are sealed")
    od = ART / "diagnostics"
    out_p = od / "posthoc.json"
    art = str(out_p.relative_to(guard.ROOT))
    L = []

    def lg(name, value, units, n, split, candidate=None):
        L.append(dict(stage="posthoc_unsealed", name=name, value=value, units=units, n=n, split=split,
                      kind="post-hoc", artifact=art, candidate=candidate))
    sp = Split("main")
    dom_of = {}
    for p, d in zip(sp.papers, sp.domains):
        dom_of[opaque_id(p)] = int(d)
    names = ["cs", "math", "physics", "q-bio", "stat"]
    zm = np.load(ART / "discovery/inputs_main.npz")
    nv = zm["nv"]
    y = np.asarray([dom_of[str(o)] for o in zm["opaque"]])
    out = {"domain_names": names}
    for var, latd, fpn in (("base", LAT, "fp_main.npz"), ("r1", LAT / "r1", "fp_main_m256.npz")):
        fz = np.load(latd / "features.npz")
        part = fz["part_main"]
        w = (np.arange(4)[None, :] < nv[:, None])[..., None].astype(np.float64)
        def pmean(X):
            return (X.astype(np.float64) * w).sum(1) / np.maximum(1, w.sum(1))
        F, S, U = pmean(fz["F_main"]), pmean(fz["S_main"]), pmean(fz["U_main"])
        reps = {"SEM": S, "GRAD": F, "COMBINED": np.concatenate([S / math.sqrt(S.shape[1]), F / math.sqrt(F.shape[1])], 1),
                "RAW_U": U}
        tr = np.nonzero((part == 0) & (nv >= 1))[0]
        te = np.nonzero((part == 1) & (nv >= 1))[0]
        chance = float(np.bincount(y[te], minlength=5).max() / len(te))
        rec = {"chance_majority": chance, "domain_pred_acc_paper_mean": {}, "confusion": {}}
        for rn, X in reps.items():
            acc, conf = ridge_cls(X[tr], y[tr], X[te], y[te], 5)
            rec["domain_pred_acc_paper_mean"][rn] = acc
            rec["confusion"][rn] = conf.tolist()
            lg(f"domain_pred_acc_{rn}_{var}", acc, "accuracy (5-way, chance %.3f)" % chance, len(te), "DISC-HOLD")
        # view-level (single view) domain predictability
        ii, jj = np.nonzero(w[..., 0] > 0)
        vt = np.isin(ii, tr)
        vh = np.isin(ii, te)
        rec["domain_pred_acc_single_view"] = {}
        for rn, X in {"SEM": fz["S_main"], "GRAD": fz["F_main"], "RAW_U": fz["U_main"]}.items():
            Xv = X[ii, jj].astype(np.float64)
            acc, _ = ridge_cls(Xv[vt], y[ii][vt], Xv[vh], y[ii][vh], 5)
            rec["domain_pred_acc_single_view"][rn] = acc
            lg(f"domain_pred_acc_single_view_{rn}_{var}", acc, "accuracy", int(vh.sum()), "DISC-HOLD views")
        # clusterings: NMI / AMI vs domain
        dec = json.loads((ART / ("discovery_decision.json" if var == "base" else "discovery_decision_r1.json"))
                         .read_text())
        rec["clusterings"] = {}
        for key, mt in dec["metrics"].items():
            src = LAT if (var == "r1" and mt["rep"] == "SEM") else latd
            vl = np.load(src / f"{key}_s{mt['medoid_seed']}" / "view_logits.npz")["main"].astype(np.float64)
            q = paper_q(vl, nv)
            hard = q[te].argmax(1)
            nmi, ami, C = nmi_ami(hard, y[te])
            rec["clusterings"][key] = {"nmi": nmi, "ami": ami, "contingency_cluster_x_domain": C.tolist(),
                                       "selected": dec["chosen"].get(mt["rep"]) == key}
            lg(f"cluster_domain_nmi_{key}_{var}", nmi, "NMI", len(te), "DISC-HOLD", key)
            lg(f"cluster_domain_ami_{key}_{var}", ami, "AMI", len(te), "DISC-HOLD", key)
        # grammar vs subject: what predicts the functional fingerprint and the GRAD cluster identity
        nu = (fz["nuis_main"].astype(np.float64) * w).sum(1) / np.maximum(1, w.sum(1))
        ix = {n: i for i, n in enumerate(NUIS_NAMES)}
        groups = {"real_domain": np.eye(5)[y], "semantic_embedding": S,
                  "equation_code_citation_density": nu[:, [ix["math_markup_rate"], ix["citation_per_kchar"],
                                                           ix["digit_rate"]]],
                  "token_frequency_stats": nu[:, [ix["mean_logp_unigram"], ix["frac_top100"], ix["type_token_ratio"]]
                                             + [ix[f"hash{i}"] for i in range(16)]],
                  "scout_ce": nu[:, [ix["scout_ce"]]], "grad_norm": nu[:, [ix["log_grad_norm"]]],
                  "document_position_length": nu[:, [ix["rel_pos"], ix["log_paper_len"], ix["log_nseg"]]],
                  "formatting_proxies": nu[:, [ix["punct_rate"], ix["upper_rate"], ix["whitespace_rate"],
                                               ix["mean_word_len"]]]}
        gk = dec["chosen"]["GRAD"]
        mt = dec["metrics"][gk]
        vl = np.load(latd / f"{gk}_s{mt['medoid_seed']}" / "view_logits.npz")["main"].astype(np.float64)
        ml = (vl * w).sum(1) / np.maximum(1, w.sum(1))
        hard = paper_q(vl, nv).argmax(1)
        h = np.arange(len(te)) % 2 == 0
        a_, b_ = te[h], te[~h]
        gv = {}
        for gname, G in groups.items():
            gv[gname] = {"r2_functional_fingerprint_paper_mean": multi_r2(G[tr], F[tr], G[te], F[te]),
                         "r2_raw_gradient_direction_paper_mean": multi_r2(G[tr], U[tr], G[te], U[te]),
                         "r2_grad_cluster_logits": multi_r2(G[a_], ml[a_], G[b_], ml[b_]),
                         "acc_grad_cluster_identity": ridge_cls(G[a_], hard[a_], G[b_], hard[b_], mt["K"])[0]}
            for k, v in gv[gname].items():
                lg(f"grammar_vs_subject_{k}_from_{gname}_{var}", v, "R2/accuracy (held-out)", len(te), "DISC-HOLD",
                   gk)
        rec["grammar_vs_subject"] = {"grad_clustering": gk, "chance_cluster_identity":
                                     float(np.bincount(hard[b_]).max() / len(b_)), "by_group": gv}
        # nearest neighbours: gradient vs semantic
        rng = np.random.default_rng(3)
        sub = te[rng.choice(len(te), 3000, replace=False)]
        toks = [set(zm["views"][d, :nv[d]].reshape(-1).tolist()) for d in sub]
        fmt = nu[sub][:, [ix["punct_rate"], ix["upper_rate"], ix["whitespace_rate"], ix["mean_word_len"],
                          ix["math_markup_rate"], ix["citation_per_kchar"], ix["digit_rate"]]]
        fmt = (fmt - fmt.mean(0)) / (fmt.std(0) + 1e-9)
        ce = nu[sub, ix["scout_ce"]]
        nnr = {}
        for rn in ("GRAD", "SEM", "RAW_U"):
            X = reps[rn][sub]
            X = X / np.maximum(1e-12, np.linalg.norm(X, axis=1, keepdims=True))
            s_ = X @ X.T
            np.fill_diagonal(s_, -np.inf)
            nn = s_.argmax(1)
            rnd = rng.permutation(len(sub))
            jac = lambda i, j: len(toks[i] & toks[j]) / max(1, len(toks[i] | toks[j]))  # noqa: E731
            nnr[rn] = {"same_domain_nn": float((y[sub] == y[sub][nn]).mean()),
                       "same_domain_random": float((y[sub] == y[sub][rnd]).mean()),
                       "abs_dce_nn": float(np.abs(ce - ce[nn]).mean()), "abs_dce_random": float(np.abs(ce - ce[rnd]).mean()),
                       "format_dist_nn": float(np.linalg.norm(fmt - fmt[nn], axis=1).mean()),
                       "format_dist_random": float(np.linalg.norm(fmt - fmt[rnd], axis=1).mean()),
                       "token_jaccard_nn": float(np.mean([jac(i, nn[i]) for i in range(len(sub))])),
                       "token_jaccard_random": float(np.mean([jac(i, rnd[i]) for i in range(len(sub))]))}
            for k, v in nnr[rn].items():
                lg(f"nearest_neighbour_{k}_{rn}_{var}", v, "fraction/abs diff/distance", len(sub), "DISC-HOLD")
        rec["nearest_neighbours"] = nnr
        # representatives per selected cluster (top-3 papers by q) with domain mix
        reps_out = {}
        for rep, key in dec["chosen"].items():
            mt = dec["metrics"][key]
            src = LAT if (var == "r1" and rep == "SEM") else latd
            q = paper_q(np.load(src / f"{key}_s{mt['medoid_seed']}" / "view_logits.npz")["main"].astype(np.float64), nv)
            hard = q.argmax(1)
            cl = {}
            for k in range(mt["K"]):
                mk = np.nonzero((hard == k) & (part == 1))[0]
                top = mk[np.argsort(-q[mk, k])[:3]]
                cl[str(k)] = {"n_hold": int(len(mk)), "domain_mix": np.bincount(y[mk], minlength=5).tolist(),
                              "representatives": [{"opaque": str(zm["opaque"][d]), "view_starts": zm["starts"][d][:nv[d]].tolist(),
                                                   "q": float(q[d, k]), "domain": names[y[d]],
                                                   "snippet": decode_batch([zm["views"][d, 0][:48]])[0][:160]}
                                                  for d in top]}
            reps_out[key] = cl
        rec["cluster_representatives"] = reps_out
        out[var] = rec
    guard.atomic_write_json(out_p, out, overwrite=True)
    for r in L:
        gc_diag.log(**r)
    s = {v: {"domain_acc": out[v]["domain_pred_acc_paper_mean"], "chance": out[v]["chance_majority"],
             "nmi_selected": {k: round(x["nmi"], 4) for k, x in out[v]["clusterings"].items() if x["selected"]},
             "nn": {k: (round(x["same_domain_nn"], 3), round(x["same_domain_random"], 3)) for k, x in
                    out[v]["nearest_neighbours"].items()}} for v in ("base", "r1")}
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
