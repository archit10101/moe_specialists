#!/usr/bin/env python
"""csr-centered-fedavg-residual-v1 analysis (CPU), reusing the csur1 estimators (csur1_analyze: exposure-matched
functional metric, paper/hierarchical bootstraps, routing, retention, position bins).

Bank aliases inside the shared estimator: F2s -> SU, F2h -> SH, L -> SO, F1s -> UO, G -> G; renamed back in outputs.

  cfar1_analyze.py stage1 --eval NPZ        gates + selection over (T, w_p)
  cfar1_analyze.py stage2 --evals A,B       Q1-Q4
  cfar1_analyze.py final --evals x4 --refs JSON   B-F1..B-F4"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402
import csur1_analyze as CA  # noqa: E402

ALIAS = {"F2s": "SU", "F2h": "SH", "L": "SO", "F1s": "UO", "G": "G"}
BACK = [(r"(?<![A-Za-z0-9])T?SU(?![A-Za-z0-9])", "F2s"), (r"(?<![A-Za-z0-9])T?SH(?![A-Za-z0-9])", "F2h"),
        (r"(?<![A-Za-z0-9])T?SO(?![A-Za-z0-9])", "L"), (r"(?<![A-Za-z0-9])T?UO(?![A-Za-z0-9])", "F1s")]


def rename_out(obj):
    s = json.dumps(obj, default=str)

    def sub(pat, new):
        nonlocal s
        s = re.sub(pat, lambda m: ("T" + new) if m.group(0).startswith("T") else new, s)
    for pat, new in BACK:
        sub(pat, new)
    return json.loads(s)


def alias_seed(S, mapping):
    z = S.z
    for new, old in mapping.items():
        for pre in ("dce_", "e_", "qtop_", "margin_", "d_", "q_", "dceT_", "eT_", "blkstats_", "ce_", "ce_T"):
            if pre + new in z:
                z[pre + old] = z.pop(pre + new)
    S.banks = [mapping.get(b, b) for b in S.banks]
    S.util = [mapping.get(b, b) for b in S.util]
    return S


def write(name, R):
    from csr.cfar1_paths import ART
    p = ART / f"{name}.json"
    guard.atomic_write_json(p, R, mode=0o400)
    return p


def load(paths, mapping):
    seeds = CA.load_seeds(paths)
    return [alias_seed(S, mapping) for S in seeds]


def stage1(a):
    import numpy as np
    from csr.cfar1_paths import T_GRID, WP_GRID
    S = CA.load_seeds([a.eval])[0]
    W = CA.boot_weights(S.P, 2000, 20261231)
    out = {"time": guard.now(), "eval": a.eval, "step": S.js["step"], "ce_base": S.ce("base"), "ce_G": S.ce("G"),
           "ce_L": S.ce("L"), "candidates": {}, "F1": {}}
    for T in T_GRID:
        out["F1"][T] = {"ce": S.ce(f"F1s_T{T}"), "I": CA.functional(S, f"F1s_T{T}")["I"]}
        for wp in WP_GRID:
            s_, h_ = f"F2s_T{T}_w{wp}", f"F2h_T{T}_w{wp}"
            fs, fh = CA.functional(S, s_), CA.functional(S, h_)
            v, r = CA.func_diff([fs], [fh], W)
            ps, pt = CA.pathology(S.load(s_)), CA.pathology(S.load(s_, True))
            rec = {"T": T, "w_p": wp, "ce_F2s": S.ce(s_), "ce_F2h": S.ce(h_), "ce_F2s_minus_G": S.ce(s_) - S.ce("G"),
                   "ce_F2s_minus_L": S.ce(s_) - S.ce("L"), "I_F2s": fs["I"], "I_F2h": fh["I"],
                   "I_F2s_minus_I_F2h": float(v[0]), "I_diff_ci": CA.ci(r[0]), "pathology_student": ps,
                   "pathology_teacher": pt, "nmi_sem_F2s": CA.nmi_val(S, s_, "lab_sem")}
            rec["gated"] = rec["ce_F2s_minus_G"] <= 0.002 and ps["ok"] and pt["ok"]
            out["candidates"][f"T{T}_w{wp}"] = rec
    ok = {c: r for c, r in out["candidates"].items() if r["gated"] and np.isfinite(r["I_F2s_minus_I_F2h"])}
    if ok:
        best = max(r["I_F2s_minus_I_F2h"] for r in ok.values())
        tied = [c for c in ok if best - ok[c]["I_F2s_minus_I_F2h"] < 1e-5]
        sel = "T250_w192" if "T250_w192" in tied else tied[0]
        out.update({"selected": sel, "selected_hp": {"T": ok[sel]["T"], "w_p": ok[sel]["w_p"]}, "stage1_pass": True,
                    "warning_best_score_nonpositive": bool(best <= 0)})
    else:
        out.update({"selected": None, "stage1_pass": False})
    p = write(a.out or "stage1", out)
    print(json.dumps({k: v for k, v in out.items() if k != "candidates"}, default=str))
    for c, r in out["candidates"].items():
        print(c, {k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items() if not isinstance(v, dict)})
    print(p)


def stage2(a):
    seeds = load(a.evals.split(","), ALIAS)
    R, W = CA.core(seeds, hierarchical=False)
    R = CA.strip(R)
    ps = R["per_seed"]
    Q1 = all(v > 0 for v in R["func"]["I_SU-I_SH"]["per_seed"])
    Q2 = all(x["ce_SU"] - x["ce_SO"] < 0 for x in ps)
    Q3 = all(x["ce_SU"] - x["ce_G"] <= 0.002 for x in ps)
    Q4 = all(p["ok"] for p in R["routing"]["SU"]["pathology"]) and all(p["ok"] for p in R["routing"]["SU"]["teacher_pathology"])
    R["criteria"] = {"Q1_func_F2s_gt_F2h_both": Q1, "Q1_values": R["func"]["I_SU-I_SH"]["per_seed"],
                     "Q2_F2s_beats_local_both": Q2, "Q2_values": [x["ce_SU"] - x["ce_SO"] for x in ps],
                     "Q3_F2s_le_G_plus_0.002_each": Q3, "Q3_values": [x["ce_SU"] - x["ce_G"] for x in ps],
                     "Q4_no_pathology": Q4}
    R["promote"] = bool(Q1 and Q2 and Q3 and Q4)
    R["time"] = guard.now()
    R["bank_alias"] = ALIAS
    R = rename_out(R)
    p = write(a.out or "stage2", R)
    print(json.dumps(R["criteria"], default=str), "PROMOTE" if R["promote"] else "NOT PROMOTED", p)


def final(a):
    seeds = load(a.evals.split(","), ALIAS)
    R, W = CA.core(seeds, hierarchical=True)
    R = CA.strip(R)
    fd = R["func"]["I_SU-I_SH"]
    F1 = fd["hier_ci"][0] > 0 and fd["n_pos"] >= 3
    F2 = R["ce"]["SU-SO"]["hier_ci"][1] < 0
    F3 = R["ce"]["SU-G"]["hier_ci"][1] < 0.002
    F4 = all(p["ok"] for p in R["routing"]["SU"]["pathology"])
    R["criteria"] = {"BF1_func_F2s_gt_F2h": F1, "BF2_F2s_beats_local": F2, "BF3_F2s_noninferior_G": F3, "BF4_integrity": F4}
    R["positive"] = bool(F1 and F2 and F3 and F4)
    R["outcome"] = "POSITIVE" if R["positive"] else ("ACCURACY_TRANSFER_WITHOUT_SEMANTIC_FUNCTION" if F2 and not F1 else
                                                     "SEMANTIC_FUNCTION_WITHOUT_TRANSFER_GAIN" if F1 and not F2 else "NEGATIVE")
    R["refs"] = json.loads(Path(a.refs).read_text()) if a.refs else {}
    R["time"] = guard.now()
    R["bank_alias"] = ALIAS
    R = rename_out(R)
    p = write(a.out or "final_analysis", R)
    print(json.dumps(R["criteria"]), R["outcome"], p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stage1", "stage2", "final"])
    ap.add_argument("--eval")
    ap.add_argument("--evals")
    ap.add_argument("--refs")
    ap.add_argument("--out")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    {"stage1": stage1, "stage2": stage2, "final": final}[a.cmd](a)


if __name__ == "__main__":
    main()
