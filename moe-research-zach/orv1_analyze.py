#!/usr/bin/env python
"""oracle-residual-v1 frozen analysis (CPU; prereg sections 7-9). Evaluated once per stage.

  orv1_analyze.py --stage S1|S2 [--seeds 0,1] [--test-dir DIR]

Reads evals/<stage>_s<seed>_EVAL.{npz,json} (full replacement matrices), the run checkpoints (W2 norms) and training
metrics; writes stage_<stage>.json (0400) and, for the last stage reached, final_analysis.json via orv1_run.py.
Statistics: paper bootstrap over EVAL papers (N_BOOT replicates, BOOT_SEED), shared across seeds and conditions;
pooled = mean over seeds of the per-seed statistic under the same replicate; 95 % percentile intervals."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402

CONDS = ("A", "B", "C")
S3_BUDGET = 50000000  # fixed by the prereg 8.3 rule (stage2_extra_rules.json)
N_HIER = 10000


def ci(b):
    import numpy as np
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def strata(z, feats):
    """Matching strata per token: target-frequency tercile x base-CE quintile x position bin x doc-length tercile x
    target class (3 x 5 x 3 x 3 x 5 = 675). Cuts are token-level quantiles on the evaluated partition (per seed)."""
    import numpy as np
    lf = np.log(feats["freq"][z["tgt"]].astype(np.float64) + 1)
    fb = np.searchsorted(np.quantile(lf, [1 / 3, 2 / 3]), lf, side="right")
    cb = np.searchsorted(np.quantile(z["ce_D"], [0.2, 0.4, 0.6, 0.8]), z["ce_D"], side="right")
    pb = np.searchsorted([128, 512], z["pos"], side="right")
    dl = z["doclen"].astype(np.float64)
    db = np.searchsorted(np.quantile(dl, [1 / 3, 2 / 3]), dl, side="right")
    tc = feats["cls"][z["tgt"]].astype(np.int64)
    return (((fb * 5 + cb) * 3 + pb) * 3 + db) * 5 + tc, 675


def functional(z, mat, lab, st, nst, P, K, W):
    """Replacement-based specificity of bank `mat` (N, K) = CE(specialist k forced) - CE_D, own labels `lab`.
    g_k(t) = mean_{j not in {k, lab(t)}} mat[t, j] - mat[t, k]   (k better than a random non-own peer on t)
    U_own(k)  = matched mean of g_k over own-cluster tokens; U_oth(k) = matched mean over other-cluster tokens
    reweighted to the own-cluster stratum distribution (common-support strata; weights fixed at full-sample values).
    Returns point values, bootstrap arrays (B, K) and per-cluster replacement summaries."""
    import numpy as np
    d = mat.astype(np.float32)
    rs = d.sum(1)
    own_d = d[np.arange(len(lab)), lab]
    out = {k: {} for k in ("U_own", "U_oth", "S")}
    nb = W.shape[0]
    bo = {k: np.zeros((nb, K)) for k in out}
    matched = np.zeros(K)
    M = np.zeros((K, K))
    for k in range(K):
        dk = d[:, k]
        is_own = lab == k
        base = np.where(is_own, (rs - dk) / (K - 1), (rs - dk - own_d) / (K - 2))
        g = base - dk
        n_own = np.bincount(st[is_own], minlength=nst).astype(np.float64)
        n_oth = np.bincount(st[~is_own], minlength=nst).astype(np.float64)
        com = (n_own > 0) & (n_oth > 0)
        matched[k] = n_own[com].sum() / max(1.0, n_own.sum())
        pi = np.where(com, n_own, 0) / max(1.0, n_own[com].sum())
        w_own = np.where(com, pi / np.maximum(n_own, 1), 0)[st]
        w_oth = np.where(com, pi / np.maximum(n_oth, 1), 0)[st]
        w = np.where(is_own, w_own, w_oth)
        res = {}
        for side, m in (("own", is_own), ("oth", ~is_own)):
            num = np.bincount(z["paper"][m], weights=(w * g)[m], minlength=P)
            den = np.bincount(z["paper"][m], weights=w[m], minlength=P)
            res[side] = (num.sum() / den.sum(), (W @ num) / (W @ den))
        out["U_own"][k], out["U_oth"][k] = float(res["own"][0]), float(res["oth"][0])
        out["S"][k] = out["U_own"][k] - out["U_oth"][k]
        bo["U_own"][:, k], bo["U_oth"][:, k] = res["own"][1], res["oth"][1]
        bo["S"][:, k] = res["own"][1] - res["oth"][1]
        M[k] = d[is_own].mean(0)  # mean CE change of every specialist on cluster k tokens
    best = np.array([np.min(np.delete(M[k], k)) - M[k, k] for k in range(K)])
    return {k: [out[k][i] for i in range(K)] for k in out}, bo, {"matched_frac": matched.tolist(),
                                                               "U_best_alt": best.tolist(), "M": M.tolist()}


def analyze_stage(stage, seeds, edir, rdir, test=False, part="EVAL"):
    import numpy as np
    import torch
    from csr.orv1_paths import BOOT_SEED, K, N_BOOT, NEAR_TIE, TABLES
    feats = dict(np.load(TABLES / "token_feats.npz"))
    zs = {s: dict(np.load(edir / f"{stage}_s{s}_{part}.npz")) for s in seeds}
    js = {s: json.loads((edir / f"{stage}_s{s}_{part}.json").read_text()) for s in seeds}
    papers = zs[seeds[0]]["papers"]
    for s in seeds:
        if not np.array_equal(zs[s]["papers"], papers):
            raise guard.GuardError(f"{part} paper sets differ across seeds")
    P = len(papers)
    rng = np.random.default_rng(BOOT_SEED)
    W = rng.multinomial(P, np.full(P, 1.0 / P), size=N_BOOT).astype(np.float64)
    R = {"stage": stage, "part": part, "seeds": list(seeds), "n_papers": P, "n_boot": N_BOOT, "per_seed": {}}
    pt, bt = {}, {}
    clus = {}
    func = {}
    for s in seeds:
        z = zs[s]
        n = np.bincount(z["paper"], minlength=P).astype(np.float64)
        sums = {c: np.bincount(z["paper"], weights=z[f"ce_{c}"].astype(np.float64), minlength=P) for c in "DABC"}
        ce = {c: sums[c].sum() / n.sum() for c in "DABC"}
        ceb = {c: (W @ sums[c]) / (W @ n) for c in "DABC"}
        stat = {"dCE_A": (ce["D"] - ce["A"], ceb["D"] - ceb["A"]), "dCE_B": (ce["D"] - ce["B"], ceb["D"] - ceb["B"]),
                "dCE_C": (ce["D"] - ce["C"], ceb["D"] - ceb["C"]), "AB": (ce["B"] - ce["A"], ceb["B"] - ceb["A"]),
                "AC": (ce["C"] - ce["A"], ceb["C"] - ceb["A"]), "CB": (ce["B"] - ce["C"], ceb["B"] - ceb["C"])}
        for k, (p_, b_) in stat.items():
            pt.setdefault(k, []).append(p_)
            bt.setdefault(k, []).append(b_)
        lab = z["lab_sem"].astype(np.int64)
        nk = np.bincount(lab, minlength=K).astype(np.float64)
        ck = {c: np.bincount(lab, weights=z[f"ce_{c}"].astype(np.float64), minlength=K) / np.maximum(nk, 1)
              for c in "DABC"}
        clus[s] = {"n": nk, **ck}
        st, nst = strata(z, feats)
        fa, fab, fx = functional(z, z["dA"], lab, st, nst, P, K, W)
        fb, fbb, fy = functional(z, z["dB"], z["lab_shuf"].astype(np.int64), st, nst, P, K, W)
        shufrep = [float((z["dB"][lab == k, k].astype(np.float32) - z["dA"][lab == k, k].astype(np.float32)).mean())
                   for k in range(K)]
        func[s] = {"A": fa, "B": fb, "Ab": fab, "Bb": fbb, "Ax": fx, "Bx": fy, "shufrep": shufrep}
        R["per_seed"][s] = {"ce": ce, "n_tokens": int(n.sum()), "loadC": js[s]["loadC"], "loadC_max": js[s]["loadC_max"],
                            "loadC_eff": js[s]["loadC_eff"], "base_unchanged": js[s]["base_unchanged"],
                            "matrix_consistency_max_abs": js[s]["matrix_consistency_max_abs"],
                            **{k: float(v[-1]) for k, v in pt.items()},
                            "S_A": float(np.mean(fa["S"])), "S_B": float(np.mean(fb["S"])),
                            "U_own_A_mean": float(np.mean(fa["U_own"])), "U_own_B_mean": float(np.mean(fb["U_own"]))}
    pooled = {k: {"mean": float(np.mean(pt[k])), "per_seed": [float(x) for x in pt[k]], "ci": ci(np.mean(bt[k], 0)),
                  "n_pos": int(sum(x > 0 for x in pt[k]))} for k in pt}
    R["pooled"] = pooled
    # per-cluster (semantic EVAL clusters), pooled seed-mean
    pc = {}
    for c in "ABC":
        per = np.array([clus[s]["D"] - clus[s][c] for s in seeds])
        pc[f"dCE_{c}"] = {"per_seed": per.tolist(), "mean": per.mean(0).tolist()}
    abk = np.array([clus[s]["B"] - clus[s]["A"] for s in seeds])
    nks = np.array([clus[s]["n"] for s in seeds])
    pc["AB"] = {"per_seed": abk.tolist(), "mean": abk.mean(0).tolist()}
    pc["n_tokens"] = nks[0].tolist()
    R["per_cluster"] = pc
    n_nonneg = int(sum(x >= 0 for x in pc["dCE_A"]["mean"]))
    contrib = (abk * nks).mean(0) / nks[0].sum()
    tot = float(contrib.sum())
    pos = float(np.clip(contrib, 0, None).sum())
    loo = [float((tot - contrib[k]) / (1 - nks[0][k] / nks[0].sum())) for k in range(K)]
    dom = {"contrib": contrib.tolist(), "total": tot, "max_share_of_positive": float(contrib.max() / pos) if pos > 0
           else None, "leave_one_cluster_out": loo,
           "pass": bool(tot > 0 and pos > 0 and contrib.max() <= 0.5 * pos and min(loo) > 0)}
    R["dominance"] = dom
    # functional, pooled
    SA = np.mean([np.mean(func[s]["Ab"]["S"], 1) for s in seeds], 0)
    SB = np.mean([np.mean(func[s]["Bb"]["S"], 1) for s in seeds], 0)
    uownA = np.mean([func[s]["A"]["U_own"] for s in seeds], 0)
    fpool = {"S_A": float(np.mean([np.mean(func[s]["A"]["S"]) for s in seeds])), "S_A_ci": ci(SA),
             "S_B": float(np.mean([np.mean(func[s]["B"]["S"]) for s in seeds])), "S_B_ci": ci(SB),
             "S_A_minus_S_B": float(np.mean([np.mean(func[s]["A"]["S"]) - np.mean(func[s]["B"]["S"]) for s in seeds])),
             "S_A_minus_S_B_ci": ci(SA - SB), "U_own_A_per_cluster": uownA.tolist(),
             "U_oth_A_per_cluster": np.mean([func[s]["A"]["U_oth"] for s in seeds], 0).tolist(),
             "S_A_per_cluster": np.mean([func[s]["A"]["S"] for s in seeds], 0).tolist(),
             "S_B_per_cluster": np.mean([func[s]["B"]["S"] for s in seeds], 0).tolist(),
             "U_own_B_per_cluster": np.mean([func[s]["B"]["U_own"] for s in seeds], 0).tolist(),
             "U_best_alt_A": np.mean([func[s]["Ax"]["U_best_alt"] for s in seeds], 0).tolist(),
             "U_best_alt_B": np.mean([func[s]["Bx"]["U_best_alt"] for s in seeds], 0).tolist(),
             "U_shuffled_replacement_A": np.mean([func[s]["shufrep"] for s in seeds], 0).tolist(),
             "matched_frac_A": np.mean([func[s]["Ax"]["matched_frac"] for s in seeds], 0).tolist(),
             "replacement_matrix_A_seedmean": np.mean([func[s]["Ax"]["M"] for s in seeds], 0).tolist(),
             "per_seed": {s: {"S_A": float(np.mean(func[s]["A"]["S"])), "S_B": float(np.mean(func[s]["B"]["S"]))}
                          for s in seeds}}
    fpool["F1"] = fpool["S_A"] > 0 and fpool["S_A_ci"][0] > 0
    fpool["F2"] = fpool["S_A_minus_S_B"] > 0 and fpool["S_A_minus_S_B_ci"][0] > 0
    fpool["F3"] = int(sum(u > 0 for u in uownA)) >= 12
    fpool["F3_n_clusters_U_own_pos"] = int(sum(u > 0 for u in uownA))
    fpool["F4"] = all(v["S_A"] > 0 and v["S_A"] > v["S_B"] for v in fpool["per_seed"].values())
    fpool["functional_pass"] = bool(fpool["F1"] and fpool["F2"] and fpool["F3"] and fpool["F4"])
    R["functional"] = fpool
    # pathology / collapse
    path = {}
    for s in seeds:
        info = {"loadC_max": js[s]["loadC_max"], "loadC_eff": js[s]["loadC_eff"],
                "finite": bool(all(np.isfinite(R["per_seed"][s]["ce"][c]) for c in "DABC"))}
        if not test:
            ck = sorted((rdir / f"s{s}").glob("ckpt_*.pt"))
            want = {"S1": "ckpt_20000000.pt", "S2": "ckpt_50000000.pt", "S3": f"ckpt_{S3_BUDGET}.pt"}[stage]
            S_ = torch.load(rdir / f"s{s}" / want, map_location="cpu", weights_only=False)
            info["W2_norm_min"] = {c: float(S_["conds"][c]["bank.w2"].flatten(1).norm(dim=1).min()) for c in CONDS}
            info["ckpts"] = [p.name for p in ck]
        info["router_ok"] = info["loadC_max"] <= 0.5 and info["loadC_eff"] >= 4.0
        info["banks_ok"] = all(v > 0 and np.isfinite(v) for v in info.get("W2_norm_min", {"x": 1.0}).values())
        path[s] = info
    R["pathology"] = path
    # stage promotion criteria (prereg 8.1)
    ps = pooled
    near = all(ps["AC"]["per_seed"][i] >= -NEAR_TIE for i in range(len(seeds)))
    crit = {"P1_A_improves_base_all_seeds": all(x > 0 for x in ps["dCE_A"]["per_seed"]),
            "P2_A_beats_B_all_seeds": all(x > 0 for x in ps["AB"]["per_seed"]),
            "P3_A_beats_C_all_seeds_or_near_tie_with_function": all(x > 0 for x in ps["AC"]["per_seed"]) or (
                near and fpool["F1"] and fpool["F2"]),
            "P4_pooled_AB_ci_excludes_0": ps["AB"]["ci"][0] > 0,
            "P5_ge12_clusters_nonneg": n_nonneg >= 12,
            "P6_not_one_cluster": dom["pass"],
            "P7_no_pathology": all(v["finite"] and v["router_ok"] and v["banks_ok"] and
                                   R["per_seed"][s]["base_unchanged"] for s, v in path.items())}
    crit["n_clusters_nonneg_dCE_A"] = n_nonneg
    crit["P3_detail"] = {"A_beats_C_all": all(x > 0 for x in ps["AC"]["per_seed"]), "near_tie_all": near}
    R["criteria"] = crit
    R["promote"] = bool(all(v for k, v in crit.items() if k.startswith("P")))
    if stage == "S3":
        # hierarchical seed x paper bootstrap (prereg 8.3): seeds resampled with replacement + one shared paper replicate
        hr = np.random.default_rng(BOOT_SEED + 1)
        Sn = len(seeds)
        sd = hr.integers(0, Sn, size=(N_HIER, Sn))
        rp = hr.integers(0, N_BOOT, size=N_HIER)
        hier = {}
        for k in ("dCE_A", "AB", "AC", "dCE_B", "dCE_C"):
            arr = np.stack(bt[k])  # (S, B)
            hier[k] = ci(arr[sd, rp[:, None]].mean(1))
        SAs = np.stack([np.mean(func[s]["Ab"]["S"], 1) for s in seeds])
        SBs = np.stack([np.mean(func[s]["Bb"]["S"], 1) for s in seeds])
        hier["S_A"] = ci(SAs[sd, rp[:, None]].mean(1))
        hier["S_A_minus_S_B"] = ci((SAs - SBs)[sd, rp[:, None]].mean(1))
        R["hier_ci"] = hier
        n_ab = sum(x > 0 for x in ps["AB"]["per_seed"])
        n_ac = sum(x > 0 for x in ps["AC"]["per_seed"])
        n_near = sum(x >= -NEAR_TIE for x in ps["AC"]["per_seed"])
        need = 3 if Sn == 4 else Sn
        fin = {"V1_val_dCE_A_pos_hier_ci": ps["dCE_A"]["mean"] > 0 and hier["dCE_A"][0] > 0,
               "V2_A_beats_B_ge3of4": n_ab >= need,
               "V3_A_beats_C_ge3of4_or_near_tie_with_function": n_ac >= need or (n_near >= need and fpool["F1"]
                                                                                   and fpool["F2"]),
               "V4_hier_ci_AB_excludes_0": hier["AB"][0] > 0,
               "V5_clusters_P5_P6": n_nonneg >= 12 and dom["pass"],
               "V6_base_hashes_unchanged": all(R["per_seed"][s]["base_unchanged"] for s in seeds),
               "V7_identical_budgets": True,
               "V8_no_pathology": crit["P7_no_pathology"]}
        fin["counts"] = {"A_beats_B": int(n_ab), "A_beats_C": int(n_ac), "near_tie_AC": int(n_near)}
        R["final_criteria"] = fin
        R["promote"] = bool(all(v for k, v in fin.items() if k.startswith("V")))
    R["outcomes"] = outcomes(ps, fpool, R["promote"])
    return R


def outcomes(ps, fpool, promote):
    """Prereg 9 interpretation rules (all that apply; primary by precedence A > C > E > B > D > MIXED)."""
    lo = lambda k: ps[k]["ci"][0]  # noqa: E731
    hi = lambda k: ps[k]["ci"][1]  # noqa: E731
    o = {"A_semantic_residual_functions": bool(promote and fpool["functional_pass"]),
         "C_token_level_better": bool(hi("AC") < 0 and lo("CB") > 0),
         "E_semantic_helps_clusters_hurts_global": bool(fpool["F1"] and hi("dCE_A") < 0),
         "B_partitioning_helps_semantics_do_not": bool(lo("dCE_A") > 0 and lo("dCE_B") > 0 and lo("AB") <= 0),
         "D_no_residual_benefit": bool(lo("dCE_A") <= 0 and lo("dCE_B") <= 0 and lo("dCE_C") <= 0)}
    order = ["A_semantic_residual_functions", "C_token_level_better", "E_semantic_helps_clusters_hurts_global",
             "B_partitioning_helps_semantics_do_not", "D_no_residual_benefit"]
    prim = next((k for k in order if o[k]), "MIXED_not_promoted")
    return {"applies": o, "primary": prim}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["S1", "S2", "S3"])
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--test-dir", default="")
    a = ap.parse_args()
    guard.check_root()
    from csr.orv1_paths import ART, EVALS, RUNS, check_prereg
    seeds = tuple(int(x) for x in a.seeds.split(","))
    if a.test_dir:
        R = analyze_stage(a.stage, seeds, Path(a.test_dir), Path(a.test_dir), test=True,
                          part="VAL" if a.stage == "S3" else "EVAL")
        print(json.dumps({k: R.get(k) for k in ("criteria", "final_criteria", "hier_ci", "promote", "outcomes")}, indent=1,
                         default=str))
        return
    check_prereg()
    out = ART / f"stage_{a.stage}.json"
    if out.exists():
        print("exists")
        return
    R = analyze_stage(a.stage, seeds, EVALS, RUNS, part="VAL" if a.stage == "S3" else "EVAL")
    R["time"] = guard.now()
    guard.atomic_write_json(out, R, mode=0o400)
    print(json.dumps({k: R.get(k) for k in ("pooled", "criteria", "final_criteria", "hier_ci", "promote", "outcomes")},
                     indent=1, default=str))


if __name__ == "__main__":
    main()
