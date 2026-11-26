#!/usr/bin/env python
"""csr-m2d2-domain-prior-v1: CPU analysis of completed validation evaluations -> final_analysis.json + tables.
Paired per-document bootstrap (2,000 resamples) for B2-B1, B2-B0, B3-B2 (and B1-B0) per seed and stage; per-domain CE;
macro CE; seed spread; specialisation summary; changed-vs-unchanged token CE (B2 vs B1 on the same tokens)."""
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402
from csr.vm_run import EXP  # noqa: E402

DOMS = ["Mathematics_and_logic", "Technology_and_applied_sciences", "Natural_and_physical_sciences",
        "Health_and_fitness", "History_and_events", "Culture_and_the_arts"]
PAIRS = [("B2", "B1"), ("B2", "B0"), ("B3", "B2"), ("B1", "B0"), ("B3", "B1")]


def boot(a_nll, b_nll, n, n_boot=2000, seed=0, strata=None):
    """(b - a) token-weighted CE difference with document resampling."""
    rng = np.random.default_rng(seed)
    pt = (b_nll.sum() - a_nll.sum()) / n.sum()
    diffs = np.empty(n_boot)
    m = len(n)
    for i in range(n_boot):
        ix = rng.integers(0, m, m)
        diffs[i] = (b_nll[ix].sum() - a_nll[ix].sum()) / n[ix].sum()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": float(pt), "ci95": [float(lo), float(hi)], "n_docs": int(m)}


def main():
    art = EXP["m2d2"]["art"]
    evs = {}
    for p in sorted(glob.glob(str(art / "evals" / "val__*" / "summary.json"))):
        s = json.loads(Path(p).read_text())
        if s.get("status") != "COMPLETE":
            continue
        s["_dir"] = str(Path(p).parent)
        evs[(s["stage"], s["seed"], s["condition"])] = s
    out = {"experiment": EXP["m2d2"]["id"], "created": guard.now(), "evaluations": {}, "comparisons": {},
           "seed_spread": {}, "specialisation": {}, "changed_tokens": {}}
    for (stg, sd, c), s in sorted(evs.items()):
        out["evaluations"][f"{stg}_s{sd}_{c}"] = {
            "run": s["run"], "ce": s["ce_nats_per_token"], "macro_domain_ce": s["macro_domain_ce"],
            "per_domain_ce": {DOMS[int(k)]: v for k, v in s["per_domain_ce"].items()}, "n_valid_tokens": s["n_valid_tokens"],
            "alpha": s.get("alpha"), "frac_tokens_any_change": s.get("frac_tokens_any_change"),
            "functional_ablation": s.get("functional_ablation")}
        sp = s["specialisation"]
        out["specialisation"][f"{stg}_s{sd}_{c}"] = [
            {k: r[k] for k in ("layer", "mi_domain_expert", "mi_null_mean", "mi_null_max", "mean_domain_entropy_per_expert",
                               "experts_unused", "expert_max_share", "adjacent_token_jaccard")} |
            ({"mi_without_prior_same_states": r["mi_without_prior_same_states"],
              "frac_tokens_top4_changed_by_prior": r["frac_tokens_top4_changed_by_prior"]}
             if "mi_without_prior_same_states" in r else {}) for r in sp]
    for (stg, sd, c), s in evs.items():
        for b, a in PAIRS:
            if c != b or (stg, sd, a) not in evs:
                continue
            A, B = evs[(stg, sd, a)], s
            na, nb = np.array(A["doc_n"]), np.array(B["doc_n"])
            assert np.array_equal(na, nb), "token counts differ"
            an, bn = np.array(A["doc_nll"]), np.array(B["doc_nll"])
            keep = na > 0
            dom = np.array(A["doc_domain"])
            res = {"overall": boot(an[keep], bn[keep], na[keep]),
                   "macro_domain_diff": float(np.mean([B["per_domain_ce"][str(k)] - A["per_domain_ce"][str(k)]
                                                       for k in range(6)])),
                   "per_domain": {}}
            for k in range(6):
                m = keep & (dom == k)
                res["per_domain"][DOMS[k]] = boot(an[m], bn[m], na[m], n_boot=1000, seed=k + 1)
            out["comparisons"][f"{stg}_s{sd}_{b}-{a}"] = res
            # descriptive (post-hoc): difference by in-document position bucket (tokens are stored per document in
            # order and documents are contiguous, so position = index - first index of the document)
            tdoc = np.load(Path(A["_dir"]) / "token_doc.npy")
            assert np.array_equal(tdoc, np.load(Path(B["_dir"]) / "token_doc.npy"))
            first = np.r_[0, np.nonzero(np.diff(tdoc))[0] + 1]
            starts = np.repeat(first, np.diff(np.r_[first, len(tdoc)]))
            posn = np.arange(len(tdoc)) - starts
            ta = np.load(Path(A["_dir"]) / "token_nll.npy").astype(np.float64)
            tb2 = np.load(Path(B["_dir"]) / "token_nll.npy").astype(np.float64)
            res["by_position_posthoc"] = {f"{lo}-{hi}": {"diff": float((tb2 - ta)[(posn >= lo) & (posn < hi)].mean()),
                                                         "n": int(((posn >= lo) & (posn < hi)).sum())}
                                          for lo, hi in ((0, 16), (16, 64), (64, 256), (256, 512), (512, 1024))}
            # changed vs unchanged tokens (prior models vs the other model on the same tokens)
            if b in ("B2", "B3"):
                chg = np.load(Path(B["_dir"]) / "changed.npy")
                tb_ = np.load(Path(B["_dir"]) / "token_nll.npy").astype(np.float64)
                ta_ = np.load(Path(A["_dir"]) / "token_nll.npy").astype(np.float64)
                out["changed_tokens"][f"{stg}_s{sd}_{b}-{a}"] = {
                    "frac_changed": float(chg.mean()),
                    "ce_b_changed": float(tb_[chg].mean()) if chg.any() else None,
                    "ce_a_changed": float(ta_[chg].mean()) if chg.any() else None,
                    "diff_changed": float((tb_ - ta_)[chg].mean()) if chg.any() else None,
                    "ce_b_unchanged": float(tb_[~chg].mean()), "ce_a_unchanged": float(ta_[~chg].mean()),
                    "diff_unchanged": float((tb_ - ta_)[~chg].mean())}
    for stg in ("final", "search"):
        for c in ("B0", "B1", "B2", "B3"):
            v = [evs[(stg, sd, c)]["ce_nats_per_token"] for sd in (0, 1, 100) if (stg, sd, c) in evs]
            if len(v) >= 2:
                out["seed_spread"][f"{stg}_{c}"] = {"ces": v, "range": float(max(v) - min(v)), "mean": float(np.mean(v))}
    # verdict helpers (fixed interpretation rules)
    ver = {}
    for sd in (0, 1):
        k21, k20, k32 = (f"final_s{sd}_B2-B1", f"final_s{sd}_B2-B0", f"final_s{sd}_B3-B2")
        if k21 in out["comparisons"]:
            c21 = out["comparisons"][k21]["overall"]
            ver[f"s{sd}_B2_beats_B1"] = c21["ci95"][1] < 0
        if k20 in out["comparisons"]:
            ver[f"s{sd}_B2_beats_B0"] = out["comparisons"][k20]["overall"]["ci95"][1] < 0
        if k32 in out["comparisons"]:
            c = out["comparisons"][k32]["overall"]
            ver[f"s{sd}_B3_worse_than_B2"] = c["ci95"][0] > 0
    out["verdict_flags"] = ver
    out["note"] = ("Screening pilot; one-epoch 43M-token final runs, two seeds; differences below ~0.003 nats are "
                   "within known run-to-run noise. Interpretation rules are fixed in the protocol.")
    p = art / "final_analysis.json"
    guard.atomic_write_json(p, out, overwrite=True)
    print(json.dumps({"evaluations": {k: (v["ce"], v["macro_domain_ce"]) for k, v in out["evaluations"].items()},
                      "comparisons": {k: v["overall"] for k, v in out["comparisons"].items()},
                      "verdict_flags": ver, "seed_spread": out["seed_spread"]}, indent=1))


if __name__ == "__main__":
    main()
