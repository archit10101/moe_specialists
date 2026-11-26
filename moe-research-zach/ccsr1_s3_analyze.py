#!/usr/bin/env python
"""ccsr1 Stage-3 analysis (CPU; amendment section C) of one continuation checkpoint on CALIB-SELECTION.

For every router seed r and SEM arm A in {SEMf, SEMj}: rebuild a Stage-2-format table (d_SEM from A_r, d_SHUF from
SHUFf_r; full 16-specialist rows on the swap subset, the deployed-choice entry elsewhere) and run the Stage-2 code
(ccsr1_analyze.evaluate_config) with the continued G. Then apply the Stage-3 criteria against stage2.json.
  ccsr1_s3_analyze.py --tag m1111 [--selftest]"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402


def full_d(n, e, dR, sub, d_sub):
    """(N, 16) fp32 with NaN except the deployed-choice entry (every token) and full rows on the swap subset."""
    import numpy as np
    d = np.full((n, 16), np.nan, np.float32)
    d[np.arange(n), e.astype(np.int64)] = dR.astype(np.float32)
    d[np.nonzero(sub)[0]] = d_sub.astype(np.float32)
    return d


def seed_table(common, arm_sem, arm_shuf, e_sem, e_shuf):
    import numpy as np
    z = dict(common)
    n = len(z["ce_base"])
    sub = z["sub"].astype(bool)
    for X, A, e in (("SEM", arm_sem, e_sem), ("SHUF", arm_shuf, e_shuf)):
        z[f"d_{X}"] = full_d(n, e, A["dR"], sub, A["d_sub"])
        for k in ("dPV", "ePV", "dT", "eT"):
            z[f"{k}_{X}"] = A[k]
    z["_oracle_sub_SEM"] = float((z["ce_base"][sub].astype(np.float64) + arm_sem["d_sub"].astype(np.float64).min(1)).mean())
    return z


def criteria(recs, s2recs):
    """Amendment section C for one SEM arm: recs / s2recs = per-seed records (Stage 3 / Stage 2), same seed order."""
    per = []
    for r, s in zip(recs, s2recs):
        dce, di = r["stu_minus_G"], r["I_SEM_minus_SHUF"]
        ci_lo = r["I_diff_ci"][0]
        viol = (dce > 0.002) or not (di > 0)
        ce_ok = dce <= 0
        sp_ok = (di >= 0.5 * s["I_SEM_minus_SHUF"]) and (ci_lo > 0)
        per.append({"seed": r["seed"], "dCE": dce, "dCE_stage2": s["stu_minus_G"], "ce_improved": dce < s["stu_minus_G"],
                    "dI": di, "dI_ci": r["I_diff_ci"], "dI_stage2": s["I_SEM_minus_SHUF"], "stop_violation": bool(viol),
                    "ce_preserved": bool(ce_ok), "spec_preserved": bool(sp_ok), "both": bool(ce_ok and sp_ok)})
    n_both = sum(p["both"] for p in per)
    return {"per_seed": per, "n_seeds_both": n_both, "any_stop_violation": any(p["stop_violation"] for p in per),
            "pass": bool(n_both >= 2 and not any(p["stop_violation"] for p in per))}


def analyze(tag):
    import numpy as np
    import ccsr1_analyze as AN
    from csr.ccsr1_paths import ART, ORV1, ROUTER_SEEDS
    from csr.ccsr1_s3_paths import S3, S3_ROUTERS, S3_TABLES
    tdir = S3_TABLES / tag
    summ = json.loads((tdir / "summary_CS2.json").read_text())
    for f, h in summ["sha256"].items():
        if guard.sha256_file(tdir / f) != h:
            raise guard.GuardError(f"{tdir / f} hash mismatch")
    common = dict(np.load(tdir / "common_CS2.npz"))
    feats = dict(np.load(ORV1 / "tables/token_feats.npz"))
    s2 = json.loads((ART / "stage2.json").read_text())
    sel = s2["frozen"]["selected"]
    s2recs = {r["seed"]: r for r in s2["configs"][sel]["per_seed"]}
    res = {"time": guard.now(), "tag": tag, "step": summ["step"], "summary": summ, "arms": {}}
    for kind in ("SEMf", "SEMj"):
        recs = []
        for r in ROUTER_SEEDS:
            A = dict(np.load(tdir / f"arm_{kind}_s{r}_CS2.npz"))
            B = dict(np.load(tdir / f"arm_SHUFf_s{r}_CS2.npz"))
            rd = S3_ROUTERS / tag / kind
            es = np.load(rd / f"SEM_s{r}_CS2.npz")["e"]
            eh = np.load(rd / f"SHUF_s{r}_CS2.npz")["e"]
            z = seed_table(common, A, B, es, eh)
            rec = AN.evaluate_config(z, feats, rd, "CS2", (r,))[0]
            rec["ce_oracle_SEM_subset"] = z["_oracle_sub_SEM"]
            rec["ce_base_subset"] = float(common["ce_base"][common["sub"].astype(bool)].astype(np.float64).mean())
            rec.pop("ce_oracle_SEM", None)
            rec.pop("regret_vs_oracle", None)
            recs.append(rec)
            print(kind, r, json.dumps({k: round(rec[k], 5) for k in ("stu_minus_G", "I_SEM_minus_SHUF", "retention",
                                                                      "stu_minus_PV")}), flush=True)
        crit = criteria(recs, [s2recs[r] for r in ROUTER_SEEDS])
        res["arms"][kind] = {"per_seed": recs, "criteria": crit,
                             "stage2_gates_on_continuation": AN.gates(recs),
                             "mean_ce_stu_SEM": float(np.mean([x["ce_stu_SEM"] for x in recs]))}
        print(kind, "criteria", json.dumps({k: crit[k] for k in ("n_seeds_both", "any_stop_violation", "pass")}))
    res["any_arm_pass"] = any(v["criteria"]["pass"] for v in res["arms"].values())
    S3.mkdir(parents=True, exist_ok=True)
    guard.atomic_write_json(S3 / f"stage3_{tag}.json", AN.strip(res), mode=0o400)
    print(json.dumps({"tag": tag, "any_arm_pass": res["any_arm_pass"],
                      **{k: v["criteria"]["pass"] for k, v in res["arms"].items()}}))


def selftest():
    import numpy as np
    s2 = [{"stu_minus_G": -0.004, "I_SEM_minus_SHUF": 0.009}] * 3
    good = [{"seed": i, "stu_minus_G": -0.001, "I_SEM_minus_SHUF": 0.006, "I_diff_ci": [0.002, 0.01]} for i in range(3)]
    assert criteria(good, s2)["pass"]
    bad = [dict(x) for x in good]
    bad[0]["stu_minus_G"] = 0.0021
    assert not criteria(bad, s2)["pass"] and criteria(bad, s2)["any_stop_violation"]
    bad = [dict(x) for x in good]
    bad[0]["I_SEM_minus_SHUF"], bad[1]["I_SEM_minus_SHUF"] = 0.004, 0.004
    assert not criteria(bad, s2)["pass"]
    one = [dict(x) for x in good]
    one[0]["stu_minus_G"] = 0.001
    assert criteria(one, s2)["pass"] and criteria(one, s2)["n_seeds_both"] == 2
    e = np.array([1, 2, 3])
    d = full_d(3, e, np.array([0.1, 0.2, 0.3]), np.array([False, True, False]), np.arange(16).reshape(1, 16))
    assert d[0, 1] == np.float32(0.1) and np.isnan(d[0, 0]) and d[1, 5] == 5 and d[2, 3] == np.float32(0.3)
    print("s3 selftest ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    if a.selftest:
        selftest()
    else:
        analyze(a.tag)


if __name__ == "__main__":
    main()
