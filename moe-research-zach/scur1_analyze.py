#!/usr/bin/env python
"""scur1 analysis (CPU). Exact table lookups on the per-token K = 5 specialist tables of an evaluation partition
(asrs2 SEL evaluation of the milestone, or a scur1 CS2 extraction) plus scur1 router deployments.

  scur1_analyze.py stage --stage NAME --eval-dir DIR --part SEL --seeds 0 [--r1-dep DIR]   -> ANALYSIS/<stage>.json
  scur1_analyze.py selftest

Per deployment: CE and paired paper-bootstrap contrasts (vs base = TOKEN-4 replicate, G, ALT-G, the existing asrs2 R1
router, the subject-prior-only control); teacher closure (reference ALT-G, primary; G reported) and per-token-oracle
closure; exposure-matched functional interaction (asrs2_analyze.functional; SEM candidate vs its SHUF twin, own groups =
paper-level K=5 groups) with per-specialist effects; routing (loads, entropy, switch, margin, teacher agreement);
subject inference (cross-entropy vs the soft region assignment, accuracy, ECE, NMI, by position, first 32, after
region changes, stable); U matrices; oracle-gap decomposition (correction head, subject recognition, mapping,
subject-granularity bank limit)."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402
import asrs2_analyze as AN  # noqa: E402

POS_BINS = AN.POS_BINS


class SEval(AN.EvalSet):
    """asrs2 EvalSet whose deployments may come from scur1 router outputs (name -> (npz path, bank))."""

    def __init__(self, d, part, extra):
        super().__init__(d, part)
        self.extra = extra       # name -> {"npz": path, "bank": bank name, "json": dict}
        self._ps = {}

    def d(self, arm):
        b = self.extra[arm]["bank"] if arm in self.extra else arm
        return super().d(b)

    def dep(self, arm):
        if arm in self.extra:
            if arm not in self._dep:
                z = np.load(self.extra[arm]["npz"])
                self._dep[arm] = (z["e"].astype(np.int64), z["q"].astype(np.float32))
                self._ps[arm] = z["ps"].astype(np.float32)
            return self._dep[arm]
        return super().dep(arm)

    def ps(self, arm):
        self.dep(arm)
        return self._ps[arm]


def nmi(a, b, K):
    J = np.bincount(a * K + b, minlength=K * K).reshape(K, K).astype(np.float64)
    J /= J.sum()
    pa, pb = J.sum(1), J.sum(0)
    nz = J > 0
    mi = float((J[nz] * np.log(J[nz] / np.outer(pa, pb)[nz])).sum())
    h = lambda p: float(-(p[p > 0] * np.log(p[p > 0])).sum())
    return 2 * mi / max(1e-12, h(pa) + h(pb))


def ece(conf, correct, nb=10):
    b = np.minimum((conf * nb).astype(np.int64), nb - 1)
    n = np.bincount(b, minlength=nb).astype(np.float64)
    c = np.bincount(b, weights=conf, minlength=nb)
    a = np.bincount(b, weights=correct.astype(np.float64), minlength=nb)
    m = n > 0
    return float((np.abs(c[m] - a[m])).sum() / n.sum())


def subject_metrics(E, arm, q_soft, lab, masks):
    ps = E.ps(arm)
    pred = ps.argmax(1)
    xent = -(q_soft * np.log(np.maximum(ps, 1e-9))).sum(1)
    out = {"xent": float(xent.mean()), "acc": float((pred == lab).mean()), "nmi": nmi(pred, lab, E.K),
           "ece": ece(ps.max(1), pred == lab)}
    for nm, m in masks.items():
        out[f"acc_{nm}"] = float((pred[m] == lab[m]).mean())
        out[f"xent_{nm}"] = float(xent[m].mean())
    return out


def routing(E, arm, lab):
    e, q = E.dep(arm)
    K = E.K
    ld = np.bincount(e, minlength=K) / E.N
    same = E.z["doc"][1:] == E.z["doc"][:-1]
    qs = np.sort(q, 1)
    return {"load": ld.tolist(), "load_max": float(ld.max()), "load_eff": float(1 / (ld ** 2).sum()),
            "n_dead": int((ld < 1 / (10 * K)).sum()), "entropy": float(-(q * np.log(np.maximum(q, 1e-9))).sum(1).mean()),
            "switch": float((e[1:] != e[:-1])[same].mean()), "margin": float((qs[:, -1] - qs[:, -2]).mean()),
            "confidence": float(qs[:, -1].mean()), "agree_teacher": float((e == lab).mean())}


def stage(a):
    from csr import scur1_paths as P
    from csr import asrs2_paths as P2
    ev = guard.inside_root(a.eval_dir)
    part = a.part
    seeds = [int(x) for x in a.seeds.split(",")]
    ddir = P.DEPS / a.stage
    extra = {}
    for js in sorted(ddir.glob(f"*_{part}.json")):
        j = json.loads(js.read_text())
        npz = js.with_suffix(".npz")
        if guard.sha256_file(npz) != j["npz_sha256"]:
            raise guard.GuardError(f"dep hash mismatch {npz}")
        if j["eval_dir"].rstrip("/") != str(Path(a.eval_dir)).rstrip("/") and j["eval_dir"] != a.eval_dir:
            continue
        extra[js.stem[:-(len(part) + 1)]] = {"npz": npz, "bank": j["bank"], "json": j}
    E = SEval(ev, part, extra)
    sb = int(E.summ["arms"][0][2])
    SEM, SHUF, G, ALTG = f"SEMh_{sb}", f"SHUFh_{sb}", f"G_{sb}", f"ALTG_{sb}"
    W = AN.boot_weights(E.P, P.N_BOOT, P.BOOT_SEED)
    tab = P2.tables("calib_search")[0] if part in ("SEL", "TRAIN") else dict(np.load(P.TABLES / "p5_calib_sel.npz"))
    rid = E.z["rid"].astype(np.int64)
    qsoft = {"sem": tab["p"].astype(np.float32)[rid], "shuf": tab["p_sh"].astype(np.float32)[rid]}
    for k in qsoft:
        qsoft[k] = (qsoft[k] + 1e-6) / (qsoft[k] + 1e-6).sum(1, keepdims=True)
    lab = {"sem": E.z["lab_rsem"].astype(np.int64), "shuf": E.z["lab_rshuf"].astype(np.int64)}
    pos = E.z["pos"].astype(np.int64)
    post, stable = AN.switch_masks(E.z)
    masks = {"first32": pos < 32, "32_128": (pos >= 32) & (pos < 128), "128_512": (pos >= 128) & (pos < 512),
             "ge512": pos >= 512, "post_switch": post, "stable": stable}
    ar = np.arange(E.N)
    ce = {"base": E.cb, "G": E.ce_choice(G, E.routed_e(G)), "ALTG": E.ce_choice(ALTG, E.routed_e(ALTG)),
          "R1_existing": E.ce_choice(SEM, E.dep(SEM)[0]), "R1_existing_SHUF": E.ce_choice(SHUF, E.dep(SHUF)[0]),
          "T_SEM": E.ce_choice(SEM, lab["sem"]), "T_SHUF": E.ce_choice(SHUF, lab["shuf"]),
          "O_SEM": E.cb + E.d(SEM).astype(np.float64).min(1), "O_SHUF": E.cb + E.d(SHUF).astype(np.float64).min(1)}
    # subject-level mapping oracle in hindsight (diagnostic; uses evaluation data)
    dS = E.d(SEM).astype(np.float64)
    best = np.stack([dS[lab["sem"] == s].mean(0) for s in range(E.K)]).argmin(1)
    ce["MAPORACLE_SEM"] = E.ce_choice(SEM, best[lab["sem"]])
    out = {"stage": a.stage, "eval_dir": str(ev.relative_to(guard.ROOT)), "part": part, "step": E.summ["step"],
           "n_tokens": E.N, "n_papers": E.P, "mapping_oracle_sem": best.tolist(), "masks_frac": {k: float(m.mean()) for k, m in masks.items()},
           "routers": {}, "time": guard.now()}
    for name, x in extra.items():
        j = x["json"]
        cfg = j["cfg"]
        pri = "sem" if cfg["bank"] == "SEM" else "shuf"
        e, q = E.dep(name)
        ce[name] = E.ce_choice(name, e)
        A = np.asarray(j["info"]["A_deploy"], np.float32)
        R = {"cfg": cfg, "seed": j["seed"], "bank": j["bank"], "U_all": j["info"]["U_all"], "U_fold0": j["info"]["U_fold0"],
             "U_fold1": j["info"]["U_fold1"], "U_split_half_r": j["info"]["U_split_half_r"], "A_deploy": j["info"]["A_deploy"],
             "train_curve_last": j["train_curve_last"], "n_params": j["n_params"]}
        R["routing"] = routing(E, name, lab[pri])
        if cfg["mode"] != "utility":
            R["subject"] = subject_metrics(E, name, qsoft[pri], lab[pri], masks)
        ps = E.ps(name)
        bank = x["bank"]
        if cfg["mode"] == "scu":
            ce[name + "__rho0"] = E.ce_choice(name, (ps @ A).argmax(1))
            ce[name + "__UT"] = E.ce_choice(name, (qsoft[pri] @ A).argmax(1))
        out["routers"][name] = R
    refs = ["ALTG", "G", "base", "R1_existing"]
    for name, R in out["routers"].items():
        cfg = R["cfg"]
        pri = "SEM" if cfg["bank"] == "SEM" else "SHUF"
        C = {}
        for rf in refs + ([f"SUBJ_s{R['seed']}"] if f"SUBJ_s{R['seed']}" in ce and pri == "SEM" else []):
            c = AN.contrast(E, ce[name], ce[rf], W)
            C[rf] = {"v": c["v"], "ci": c["ci"]}
        R["ce"] = float(ce[name].mean())
        R["contrasts"] = C
        T_, O_ = ce[f"T_{pri}"].mean(), ce[f"O_{pri}"].mean()
        for rf in ("ALTG", "G"):
            den = ce[rf].mean() - T_
            R[f"closure_teacher_vs_{rf}"] = float((ce[rf].mean() - ce[name].mean()) / den) if abs(den) > 1e-9 else None
            R[f"teacher_gain_vs_{rf}"] = float(den)
            R[f"closure_oracle_vs_{rf}"] = float((ce[rf].mean() - ce[name].mean()) / (ce[rf].mean() - O_))
        R["by_mask_vs_ALTG"] = {k: float((ce[name] - ce["ALTG"])[m].mean()) for k, m in masks.items()}
        R["by_mask_vs_teacher"] = {k: float((ce[name] - ce[f"T_{pri}"])[m].mean()) for k, m in masks.items()}
        labg = E.z["lab_sem" if pri == "SEM" else "lab_shuf"].astype(np.int64)
        ng = np.bincount(labg[labg >= 0], minlength=E.K).astype(np.float64)
        R["per_subject_vs_ALTG"] = (np.bincount(labg[labg >= 0], weights=(ce[name] - ce["ALTG"])[labg >= 0], minlength=E.K)
                                    / np.maximum(ng, 1)).tolist()
        if cfg["mode"] == "scu" and pri == "SEM":
            m = {k: float(ce[k].mean()) for k in (name, name + "__rho0", name + "__UT", "MAPORACLE_SEM", "O_SEM", "T_SEM")}
            R["gap_decomposition"] = {"correction_head": m[name] - m[name + "__rho0"],
                                      "subject_recognition": m[name + "__rho0"] - m[name + "__UT"],
                                      "mapping_U_vs_hindsight": m[name + "__UT"] - m["MAPORACLE_SEM"],
                                      "subject_granularity_bank_limit": m["MAPORACLE_SEM"] - m["O_SEM"],
                                      "UT_minus_region_teacher": m[name + "__UT"] - m["T_SEM"],
                                      "subject_recognition_first32": float((ce[name + "__rho0"] - ce[name + "__UT"])[masks["first32"]].mean()),
                                      "subject_recognition_post_switch": float((ce[name + "__rho0"] - ce[name + "__UT"])[masks["post_switch"]].mean()),
                                      "subject_recognition_stable": float((ce[name + "__rho0"] - ce[name + "__UT"])[masks["stable"]].mean())}
    # functional: SEM router vs its SHUF twin (same tag with _SHUF), and the existing R1 pair
    pairs = [("R1_existing", SEM, SHUF)]
    for name, R in out["routers"].items():
        if R["cfg"]["bank"] == "SEM":
            tw = name.replace("_s", "_SHUF_s", 1) if "_SHUF" not in name else None
            base_tag = R["cfg"]["tag"]
            for n2, R2 in out["routers"].items():
                if R2["cfg"]["bank"] == "SHUF" and R2["cfg"].get("twin_of") == base_tag and R2["seed"] == R["seed"]:
                    pairs.append((name, name, n2))
    out["functional"] = {}
    for tagp, a_, b_ in pairs:
        fa = AN.functional(E, a_, "lab_sem")
        fb = AN.functional(E, b_, "lab_shuf")
        dI = AN.func_diff(fa, fb, W)
        ps_ = AN.per_specialist(fa, W)
        c = AN.contrast(E, ce[tagp if tagp in ce else a_], ce[b_] if b_ in ce else E.ce_choice(b_, E.dep(b_)[0]), W)
        out["functional"][tagp] = {"I": fa["I"], "I_SHUF": fb["I"], "dI": dI["v"], "dI_ci": dI["ci"],
                                   "I_ordinary": fa["I_ordinary"], "I_ordinary_SHUF": fb["I_ordinary"],
                                   "per_specialist": ps_, "sigma": fa["sigma"],
                                   "sem_minus_shuf_ce": {"v": c["v"], "ci": c["ci"]}}
    out["ce_mean"] = {k: float(v.mean()) for k, v in ce.items()}
    out["teacher_gain"] = {"T_SEM-ALTG": float(ce["T_SEM"].mean() - ce["ALTG"].mean()),
                           "T_SEM-G": float(ce["T_SEM"].mean() - ce["G"].mean())}
    c = AN.contrast(E, ce["T_SEM"], ce["ALTG"], W)
    out["teacher_gain_ci_vs_ALTG"] = c["ci"]
    P.ANALYSIS.mkdir(parents=True, exist_ok=True)
    guard.atomic_write_json(P.ANALYSIS / f"{a.stage}.json", AN.strip(out), overwrite=True)
    brief = {n: {"ce-ALTG": round(R["contrasts"]["ALTG"]["v"], 5), "ce-G": round(R["contrasts"]["G"]["v"], 5),
                 "closure_T_ALTG": R["closure_teacher_vs_ALTG"], "agree": round(R["routing"]["agree_teacher"], 3),
                 "switch": round(R["routing"]["switch"], 4), "subj_acc": R.get("subject", {}).get("acc")}
             for n, R in out["routers"].items()}
    print(json.dumps({"teacher_gain": out["teacher_gain"], "routers": brief,
                      "functional": {k: (round(v["dI"], 5), [round(x, 5) for x in v["dI_ci"]], v["per_specialist"]["n_pos"])
                                     for k, v in out["functional"].items()}}, indent=1, default=str))
    return out


def criteria(A, cand, seed=0):
    """Stage-A promotion criteria P1-P7 for SEM candidate router `cand` (tag) of router seed `seed`."""
    n = f"{cand}_s{seed}"
    R = A["routers"][n]
    F = A["functional"].get(n)
    C = R["contrasts"]
    rt = R["routing"]
    c = {"P1 beats ALT-G (CI upper < 0)": C["ALTG"]["ci"][1] < 0,
         "P2 CE below G (point estimate)": C["G"]["v"] < 0,
         "P3 teacher closure vs ALT-G >= 0.75": (R["closure_teacher_vs_ALTG"] or -9) >= 0.75,
         "P4 dI(SEM - SHUF twin) > 0, CI lower > 0": bool(F) and F["dI"] > 0 and F["dI_ci"][0] > 0,
         "P5 >= 4/5 positive specialists": bool(F) and F["per_specialist"]["n_pos"] >= 4,
         "P6 no collapse (0 dead, eff >= 3, max load <= 0.5)": rt["n_dead"] == 0 and rt["load_eff"] >= 3 and rt["load_max"] <= 0.5,
         "P7 beats subject-prior-only control (CI upper < 0)": f"SUBJ_s{seed}" in C and C[f"SUBJ_s{seed}"]["ci"][1] < 0}
    return {"pass": all(c.values()), "criteria": c, "failed": [k for k, v in c.items() if not v]}


def selftest():
    a = np.array([0, 0, 1, 1, 2, 2])
    assert abs(nmi(a, a, 3) - 1) < 1e-9
    assert ece(np.array([0.9, 0.9]), np.array([True, True])) < 0.11
    print(json.dumps({"selftest": "ok"}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--stage")
    ap.add_argument("--eval-dir")
    ap.add_argument("--part", default="SEL")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    if a.cmd == "stage":
        stage(a)
    elif a.cmd == "selftest":
        selftest()


if __name__ == "__main__":
    main()
