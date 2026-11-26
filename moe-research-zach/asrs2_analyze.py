#!/usr/bin/env python
"""asrs2 analysis (CPU). Exact table lookups on the per-token K=5 specialist tables written by asrs2_eval.py.

  asrs2_analyze.py ckpt --eval-dir DIR --part SEL --tag TAG [--seeds 0]   per-milestone metrics -> ANALYSIS/TAG.json
  asrs2_analyze.py stage0f --tag TAG                                       Stage-0F gate (frozen stage)
  asrs2_analyze.py stage1 --frozen TAG_F --final TAG_1111 [--traj TAG,...] --gen N   Stage-1 criteria -> stage1_genN.json
  asrs2_analyze.py selftest

Functional metric (exposure-matched own-vs-swap usefulness; csur1/ccsr1 definition ported to K = 5 and to full
tables): g_e(t) = mean_{j != e} d[t, j] - d[t, e] (d = CE_j - CE_base). Own specialist sigma(k) = argmax routing lift
P(e | k) / P(e) of the bank's deployed (causal router) choice for paper-level semantic group k, fitted on one paper half
and measured on the other: own set = group-k tokens routed to sigma(k), other set = other-group tokens routed to
sigma(k); strata = target-frequency tercile x base-CE quintile x position bin x doc-length tercile x q_sigma(k)
quintile x router-margin tercile (2,025 strata), common support, own-stratum weights; I = mean over scored groups of
the two-direction average. Delta I = I_SEM - I_SHUF with per-paper influence-function paper bootstrap (2,000)."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402

MIN_SET = 2000
POS_BINS = [(0, 32), (32, 128), (128, 512), (512, 100000)]


def qcut(x, qs):
    return np.searchsorted(np.quantile(x, qs), x, side="right")


def ci(x):
    return [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))]


def boot_weights(P, n, seed):
    return np.random.default_rng(seed).multinomial(P, np.full(P, 1.0 / P), size=n).astype(np.float64)


class EvalSet:
    def __init__(self, d, part):
        from orv1_tables import u_hash
        self.dir = Path(d)
        self.part = part
        self.summ = json.loads((self.dir / f"summary_{part}.json").read_text())
        for fn, sha in self.summ["sha256"].items():
            if guard.sha256_file(self.dir / fn) != sha:
                raise guard.GuardError(f"eval file hash mismatch {self.dir / fn}")
        self.z = dict(np.load(self.dir / f"common_{part}.npz"))
        z = self.z
        self.cb = z["ce_base"].astype(np.float64)
        self.N = len(self.cb)
        self.papers = z["papers"]
        self.P = len(self.papers)
        self.paper = z["paper"].astype(np.int64)
        self.n = np.bincount(self.paper, minlength=self.P).astype(np.float64)
        self.half = np.asarray([u_hash("asrs2-half", str(p)) < 0.5 for p in self.papers])
        feats = dict(np.load(guard.ROOT / "artifacts/oracle-residual-v1/tables/token_feats.npz"))
        lf = np.log(feats["freq"][z["tgt"]].astype(np.float64) + 1)
        fb = qcut(lf, [1 / 3, 2 / 3])
        cbq = qcut(self.cb, [0.2, 0.4, 0.6, 0.8])
        pb = np.searchsorted([128, 512], z["pos"], side="right")
        db = qcut(z["doclen"].astype(np.float64), [1 / 3, 2 / 3])
        self.base_st = ((fb * 5 + cbq) * 3 + pb) * 3 + db
        self._d, self._dep = {}, {}
        self.K = int(self.summ["K"])

    def d(self, arm):
        if arm not in self._d:
            self._d[arm] = np.load(self.dir / f"bank_{arm}_{self.part}.npz")["d"].astype(np.float32)
        return self._d[arm]

    def routed_e(self, arm):
        return np.load(self.dir / f"bank_{arm}_{self.part}.npz")["e"].astype(np.int64)

    def dep(self, arm):
        if arm not in self._dep:
            z = np.load(self.dir / f"dep_{arm}_{self.part}.npz")
            self._dep[arm] = (z["e"].astype(np.int64), z["q"].astype(np.float32))
        return self._dep[arm]

    def ce_choice(self, arm, e):
        return self.cb + self.d(arm)[np.arange(self.N), e].astype(np.float64)

    def psums(self, v, mask=None):
        if mask is None:
            return np.bincount(self.paper, weights=v, minlength=self.P)
        return np.bincount(self.paper[mask], weights=v[mask], minlength=self.P)


def contrast(E, a, b, W, mask=None):
    """Mean CE difference a - b (token-weighted) with a paper-bootstrap CI."""
    sa, sb = E.psums(a, mask), E.psums(b, mask)
    n = E.n if mask is None else np.bincount(E.paper[mask], minlength=E.P).astype(np.float64)
    v = float((sa.sum() - sb.sum()) / n.sum())
    reps = (W @ sa - W @ sb) / np.maximum(W @ n, 1)
    return {"v": v, "ci": ci(reps), "reps": reps}


def own_map(lab, e, mask, K):
    ok = mask & (lab >= 0)
    C = np.bincount(lab[ok] * K + e[ok], minlength=K * K).reshape(K, K).astype(np.float64)
    pe = C.sum(0) / max(1.0, C.sum())
    lift = (C / np.maximum(C.sum(1, keepdims=True), 1)) / np.maximum(pe, 1e-12)
    return lift.argmax(1)


def functional(E, arm, lab_key):
    K = E.K
    d = E.d(arm).astype(np.float64)
    e, q = E.dep(arm)
    lab = E.z[lab_key].astype(np.int64)
    pap = E.paper
    tok_half = E.half[pap]
    rowsum = d.sum(1)
    ms = np.sort(q, 1)
    marg = (ms[:, -1] - ms[:, -2]).astype(np.float64)
    Ik = np.full((2, K), np.nan)
    IFk = np.zeros((2, K, E.P))
    Iord = np.full((2, K), np.nan)
    sig = []
    nst = 135 * 15
    for di, (fit_h, meas_h) in enumerate(((True, False), (False, True))):
        sigma = own_map(lab, e, tok_half == fit_h, K)
        sig.append(sigma.tolist())
        for k in range(K):
            ek = int(sigma[k])
            inv = (tok_half == meas_h) & (e == ek) & (lab >= 0)
            own, oth = inv & (lab == k), inv & (lab != k)
            if own.sum() < MIN_SET or oth.sum() < MIN_SET:
                continue
            g = (rowsum - d[:, ek]) / (K - 1) - d[:, ek]
            Iord[di, k] = float(g[own].mean() - g[oth].mean())
            qe = q[:, ek].astype(np.float64)
            qb = np.searchsorted(np.quantile(qe[inv], [0.2, 0.4, 0.6, 0.8]), qe, side="right")
            mb = np.searchsorted(np.quantile(marg[inv], [1 / 3, 2 / 3]), marg, side="right")
            st = (E.base_st * 5 + qb) * 3 + mb
            so, sn = st[own], st[oth]
            no = np.bincount(so, minlength=nst).astype(np.float64)
            nn = np.bincount(sn, minlength=nst).astype(np.float64)
            common = (no > 0) & (nn > 0)
            ko, kn = common[so], common[sn]
            if ko.sum() < MIN_SET or kn.sum() < MIN_SET:
                continue
            go, gn_ = g[own][ko], g[oth][kn]
            so, sn = so[ko], sn[kn]
            no = np.bincount(so, minlength=nst).astype(np.float64)
            nn = np.bincount(sn, minlength=nst).astype(np.float64)
            w = no / no.sum()
            mo = np.bincount(so, weights=go, minlength=nst) / np.maximum(no, 1)
            mn = np.bincount(sn, weights=gn_, minlength=nst) / np.maximum(nn, 1)
            Ik[di, k] = float((w * (mo - mn)).sum())
            ifo = w[so] * (go - mo[so]) / no[so]
            ifn = -w[sn] * (gn_ - mn[sn]) / nn[sn]
            IFk[di, k] = (np.bincount(pap[own][ko], weights=ifo, minlength=E.P)
                          + np.bincount(pap[oth][kn], weights=ifn, minlength=E.P))
    scored = ~np.isnan(Ik).any(0)
    if scored.sum() == 0:
        return {"I": float("nan"), "n_scored": 0, "IF": np.zeros(E.P), "IFk": np.zeros((K, E.P)), "I_k": [None] * K,
                "sigma": sig, "I_ordinary": float("nan")}
    Ig = 0.5 * (Ik[0] + Ik[1])
    IFg = 0.5 * (IFk[0] + IFk[1])
    return {"I": float(np.nanmean(Ig[scored])), "I_k": [None if not scored[k] else float(Ig[k]) for k in range(K)],
            "n_scored": int(scored.sum()), "IF": IFg[scored].mean(0), "IFk": IFg, "sigma": sig,
            "distinct_sigma": [len(set(s)) for s in sig], "I_ordinary": float(np.nanmean(0.5 * (Iord[0] + Iord[1])))}


def func_diff(fa, fb, W):
    v = fa["I"] - fb["I"]
    D = fa["IF"] - fb["IF"]
    reps = v + W @ D - D.sum()
    return {"v": float(v), "ci": ci(reps), "reps": reps}


def per_specialist(f, W):
    out = []
    for k, Ik in enumerate(f["I_k"]):
        if Ik is None:
            out.append({"k": k, "I": None})
            continue
        reps = Ik + W @ f["IFk"][k] - f["IFk"][k].sum()
        out.append({"k": k, "I": Ik, "ci": ci(reps)})
    sc = [o for o in out if o["I"] is not None]
    pos = np.asarray([max(0.0, o["I"]) for o in sc]) if sc else np.zeros(1)
    return {"groups": out, "n_scored": len(sc), "n_pos": int(sum(o["I"] > 0 for o in sc)),
            "n_sig": int(sum(o["ci"][0] > 0 for o in sc)),
            "max_pos_share": float(pos.max() / pos.sum()) if pos.sum() > 0 else float("nan")}


def switch_masks(z):
    rid = z["rid"].astype(np.int64)
    lab = z["lab_rsem"].astype(np.int64)
    pos = z["pos"].astype(np.int64)
    reg = pos // 256
    starts = pos % 256 == 0
    keys, vals = rid[starts], lab[starts]
    o = np.argsort(keys)
    keys, vals = keys[o], vals[o]
    idx = np.clip(np.searchsorted(keys, rid - 1), 0, len(keys) - 1)
    prev = np.where((keys[idx] == rid - 1) & (reg > 0), vals[idx], -1)
    return (prev >= 0) & (prev != lab) & (pos % 256 < 64), (prev >= 0) & (prev == lab) & (pos % 256 >= 64)


def majority_acc(feat, lab, fold, K, top=2000):
    feat, lab = np.asarray(feat, np.int64), np.asarray(lab, np.int64)
    accs, ns = [], []
    for fit in (fold, ~fold):
        ev = ~fit
        vals, cnt = np.unique(feat[fit], return_counts=True)
        sk = np.sort(vals[np.argsort(-cnt)[:top]])
        cf = np.where(np.isin(feat[fit], sk), np.searchsorted(sk, feat[fit]), len(sk))
        tab = np.bincount(cf * K + lab[fit], minlength=(len(sk) + 1) * K).reshape(len(sk) + 1, K)
        pred = tab.argmax(1)
        pred[tab.sum(1) == 0] = int(np.bincount(lab[fit], minlength=K).argmax())
        ce_ = np.where(np.isin(feat[ev], sk), np.searchsorted(sk, feat[ev]), len(sk))
        accs.append(float((pred[ce_] == lab[ev]).mean()))
        ns.append(int(ev.sum()))
    return float(np.average(accs, weights=ns))


def predictability(E, arm, router_arm, lab_key, rlab_key):
    """Cross-fitted (paper halves) majority-vote accuracy of the per-token ORACLE specialist and of the target-free
    ASSIGNED specialist (region-prior argmax) from target / input token / paper group / region prior / causal router."""
    z = E.z
    K = E.K
    rng = np.random.default_rng(0)
    take = np.sort(rng.choice(E.N, size=min(E.N, 2_000_000), replace=False))
    d = E.d(arm)[take]
    orc = d.argmin(1)
    fold = E.half[E.paper[take]]
    tgt = z["tgt"][take]
    prev = np.concatenate([[-1], z["tgt"][:-1]])
    same = np.concatenate([[False], z["doc"][1:] == z["doc"][:-1]])
    xin = np.where(same & (z["pos"] > 0), prev, -1)[take]
    grp, rl = z[lab_key][take], z[rlab_key][take]
    out = {}
    for nm, lab in (("oracle", orc), ("assigned", rl.astype(np.int64))):
        o = {"majority": float(np.bincount(lab, minlength=K).max() / len(lab)),
             "target": majority_acc(tgt, lab, fold, K), "input_token": majority_acc(xin, lab, fold, K),
             "paper_group": majority_acc(grp, lab, fold, K, top=K), "region_prior": majority_acc(rl, lab, fold, K, top=K)}
        if router_arm:
            o["causal_router"] = majority_acc(E.dep(router_arm)[0][take], lab, fold, K, top=K)
        out[nm] = o
    return out


def r2_reliability(E, arm):
    """Partner-view utility reliability (prereg): U[g, e] = mean CE reduction of specialist e on tokens of region group g,
    estimated on each paper half; split-half r of the K x K tables; on half B, region-teacher CE of
    argmax(log(p5 + eps) + U_A[g] / tau) vs argmax p5 (tau from {0.5, 1, 2} x sd(U_A) chosen on half A)."""
    from csr import asrs2_paths as P
    tab, _ = P.tables("calib_search" if E.part == "SEL" else "calib_sel")
    p = tab["p"].astype(np.float32)[E.z["rid"].astype(np.int64)]
    lp = np.log(p + P.PSEM_EPS)
    g = E.z["lab_rsem"].astype(np.int64)
    u = -E.d(arm).astype(np.float64)
    K = E.K
    hA = E.half[E.paper]
    U = []
    for m in (hA, ~hA):
        U.append(np.stack([u[m & (g == k)].mean(0) for k in range(K)]))
    r = float(np.corrcoef(U[0].ravel(), U[1].ravel())[0, 1])
    sd = float(U[0].std())
    ar = np.arange(E.N)
    best = None
    for f in (0.5, 1.0, 2.0):
        ch = np.argmax(lp + U[0][g] / (f * sd), 1)
        ceA = float((E.cb - u[ar, ch])[hA].mean())
        if best is None or ceA < best[1]:
            best = (f, ceA)
    ch2 = np.argmax(lp + U[0][g] / (best[0] * sd), 1)
    ch1 = np.argmax(lp, 1)
    ceB2 = float((E.cb - u[ar, ch2])[~hA].mean())
    ceB1 = float((E.cb - u[ar, ch1])[~hA].mean())
    return {"split_half_r": r, "tau_factor": best[0], "sd_U": sd, "ceB_R2_teacher": ceB2, "ceB_R1_teacher": ceB1,
            "R2_minus_R1": ceB2 - ceB1, "U_A": U[0].tolist(),
            "reliable": bool(r >= 0.8 and (ceB1 - ceB2) >= 0.001)}


def analyze_ckpt(eval_dir, part, seeds):
    from csr import asrs2_paths as P
    E = EvalSet(eval_dir, part)
    W = boot_weights(E.P, P.N_BOOT, P.BOOT_SEED)
    arms = {n: (k, s) for n, k, s in E.summ["arms"]}
    mode = E.summ["mode"]
    out = {"eval_dir": str(Path(eval_dir).relative_to(guard.ROOT)), "part": part, "step": E.summ["step"], "mode": mode,
           "n_tokens": E.N, "n_papers": E.P, "ce_base": float(E.cb.mean()), "seeds": {}}
    post, stable = switch_masks(E.z)
    pos = E.z["pos"].astype(np.int64)
    ar = np.arange(E.N)
    for s in seeds:
        R = {}
        if mode == "frozen":
            gen, sem, shuf, extra = f"G5_{s}", f"SEM5_{s}", f"SHUF5_{s}", []
            altg = None
        else:
            gen, altg, sem, shuf = f"G_{s}", f"ALTG_{s}", f"SEMh_{s}", f"SHUFh_{s}"
            extra = [(f"SEMs_{s}", f"SHUFs_{s}")]
        ce = {"base": E.cb, "G": E.ce_choice(gen, E.routed_e(gen))}
        if altg:
            ce["ALTG"] = E.ce_choice(altg, E.routed_e(altg))
        pairs = [("SEM", sem, shuf)] + [(f"SOFT{i}", a_, b_) for i, (a_, b_) in enumerate(extra)]
        if mode != "frozen":
            pairs.append(("FROZ", f"FROZ_{s}", shuf))
        for tagp, a_, b_ in pairs:
            for nm, arm in ((tagp, a_), (tagp + "_SHUF", b_)):
                if arm not in arms:
                    continue
                ce[nm] = E.ce_choice(arm, E.dep(arm)[0])
                rl = E.z["lab_rshuf" if "SHUF" in nm and tagp != "FROZ" else "lab_rsem"].astype(np.int64)
                if nm.endswith("_SHUF") or nm.startswith("SHUF"):
                    rl = E.z["lab_rshuf"].astype(np.int64)
                ce["T_" + nm] = E.ce_choice(arm, rl)
                ce["O_" + nm] = E.cb + E.d(arm).astype(np.float64).min(1)
        R["ce_mean"] = {k: float(v.mean()) for k, v in ce.items()}
        cmp = {}
        refs = ["G"] + (["ALTG"] if altg else []) + ["base"]
        for k in ce:
            if k in ("base", "G", "ALTG"):
                continue
            for rf in refs:
                c = contrast(E, ce[k], ce[rf], W)
                cmp[f"{k}-{rf}"] = {"v": c["v"], "ci": c["ci"]}
        for tagp, a_, b_ in pairs:
            if tagp in ce and tagp + "_SHUF" in ce:
                c = contrast(E, ce[tagp], ce[tagp + "_SHUF"], W)
                cmp[f"{tagp}-{tagp}_SHUF"] = {"v": c["v"], "ci": c["ci"]}
        if altg:
            c = contrast(E, ce["G"], ce["ALTG"], W)
            cmp["G-ALTG"] = {"v": c["v"], "ci": c["ci"]}
        cmp["G-base"] = {"v": float((ce["G"] - E.cb).mean())}
        R["contrasts"] = cmp
        # functional + per specialist + routing pathology per pair
        F = {}
        for tagp, a_, b_ in pairs:
            if a_ not in arms or b_ not in arms:
                continue
            fa = functional(E, a_, "lab_sem")
            fb = functional(E, b_, "lab_shuf")
            dI = func_diff(fa, fb, W)
            ps = per_specialist(fa, W)
            e_a = E.dep(a_)[0]
            ld = np.bincount(e_a, minlength=E.K) / E.N
            same = E.z["doc"][1:] == E.z["doc"][:-1]
            F[tagp] = {"I": fa["I"], "I_SHUF": fb["I"], "dI": dI["v"], "dI_ci": dI["ci"], "I_k": fa["I_k"],
                       "I_ordinary": fa["I_ordinary"], "I_ordinary_SHUF": fb["I_ordinary"], "sigma": fa["sigma"],
                       "distinct_sigma": fa.get("distinct_sigma"), "per_specialist": ps,
                       "load": ld.tolist(), "load_max": float(ld.max()), "load_eff": float(1 / (ld ** 2).sum()),
                       "n_dead": int((ld < 1.0 / (10 * E.K)).sum()),
                       "switch": float((e_a[1:] != e_a[:-1])[same].mean()),
                       "agree_region_prior": float((e_a == E.z["lab_rsem"]).mean()),
                       "mean_specialist_delta": float(E.d(a_).mean()),
                       "mean_specialist_delta_SHUF": float(E.d(b_).mean())}
            # position bins, transitions, groups vs the reference generic (ALTG in cont, G in frozen)
            ref = ce["ALTG"] if altg else ce["G"]
            pb = {}
            for lo, hi in POS_BINS:
                m = (pos >= lo) & (pos < hi)
                pb[f"{lo}_{hi}"] = {"frac": float(m.mean()), "vs_ref": float((ce[tagp] - ref)[m].mean()),
                                    "vs_G": float((ce[tagp] - ce["G"])[m].mean()),
                                    "T_vs_ref": float((ce["T_" + tagp] - ref)[m].mean())}
            F[tagp]["pos_bins"] = pb
            m128 = pos >= 128
            F[tagp]["ge128_vs_ref"] = float((ce[tagp] - ref)[m128].mean())
            F[tagp]["post_switch"] = {"frac": float(post.mean()), "vs_ref": float((ce[tagp] - ref)[post].mean()),
                                      "vs_T": float((ce[tagp] - ce["T_" + tagp])[post].mean())}
            F[tagp]["stable"] = {"frac": float(stable.mean()), "vs_ref": float((ce[tagp] - ref)[stable].mean()),
                                 "vs_T": float((ce[tagp] - ce["T_" + tagp])[stable].mean())}
            lab = E.z["lab_sem"].astype(np.int64)
            ng = np.bincount(lab[lab >= 0], minlength=E.K).astype(np.float64)
            dg = np.bincount(lab[lab >= 0], weights=(ce[tagp] - ref)[lab >= 0], minlength=E.K) / np.maximum(ng, 1)
            imp = np.maximum(-dg * ng, 0)
            F[tagp]["group_vs_ref"] = {"delta": dg.tolist(), "n_better": int((dg < 0).sum()),
                                       "max_share": float(imp.max() / imp.sum()) if imp.sum() > 0 else float("nan")}
        R["functional"] = F
        R["predictability"] = {"SEM": predictability(E, sem, sem, "lab_sem", "lab_rsem"),
                               "SHUF": predictability(E, shuf, shuf, "lab_shuf", "lab_rshuf")}
        if mode == "frozen":
            R["r2_reliability"] = r2_reliability(E, sem)
        out["seeds"][str(s)] = R
    return out


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if k not in ("reps", "IF", "IFk")}
    if isinstance(o, list):
        return [strip(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def cmd_ckpt(a):
    from csr import asrs2_paths as P
    P.ANALYSIS.mkdir(parents=True, exist_ok=True)
    outp = P.ANALYSIS / f"{a.tag}.json"
    if outp.exists():
        print("exists")
        return
    res = analyze_ckpt(guard.inside_root(a.eval_dir), a.part, [int(x) for x in a.seeds.split(",")])
    res["time"] = guard.now()
    guard.atomic_write_json(outp, strip(res), mode=0o400)
    print(json.dumps(strip(res), default=str)[:4000])


# ------------------------------------------------------------------------------------------------ gates / criteria
def stage0f(tagF, seed=0):
    from csr import asrs2_paths as P
    A = json.loads((P.ANALYSIS / f"{tagF}.json").read_text())["seeds"][str(seed)]
    F = A["functional"]["SEM"]
    c = {"F0-1 dI>0 and CI lower>0": F["dI"] > 0 and F["dI_ci"][0] > 0,
         "F0-2 >=4/5 positive specialists": F["per_specialist"]["n_pos"] >= 4}
    return {"pass": all(c.values()), "criteria": c, "dI": F["dI"], "dI_ci": F["dI_ci"],
            "n_pos": F["per_specialist"]["n_pos"], "SEM-G": A["contrasts"]["SEM-G"], "SEM-SEM_SHUF": A["contrasts"]["SEM-SEM_SHUF"],
            "T_SEM-G": A["contrasts"].get("T_SEM-G")}


def candidate_criteria(A, AF, tagp, base_ok):
    """Stage-1 criteria for one candidate (tagp: 'SEM' hard or 'SOFT0' soft) at the final milestone; AF = frozen stage."""
    F = A["functional"][tagp]
    C = A["contrasts"]
    dIF = AF["functional"]["SEM"]["dI"]
    pred = A["predictability"]
    tgt_sem, tgt_sh = pred["SEM"]["assigned"]["target"], pred["SHUF"]["assigned"]["target"]
    crit = {
        "C1 base matches ordinary base": base_ok,
        "C2 mean specialist delta <= +0.02 (no catastrophic divergence)": F["mean_specialist_delta"] <= 0.02,
        "C3 target predicts assigned specialist <= SHUF + 0.05": tgt_sem <= tgt_sh + 0.05,
        "C4/S1-8 SEM beats persistent SHUF (CI excludes 0)": C[f"{tagp}-{tagp}_SHUF"]["ci"][1] < 0,
        "C5/S1-6 no collapse (0 dead, eff >= 3, max load <= 0.5)": F["n_dead"] == 0 and F["load_eff"] >= 3 and F["load_max"] <= 0.5,
        "S1-1 CE <= ordinary G + 0.001": C[f"{tagp}-G"]["v"] <= 0.001,
        "S1-2 CE < ALT-G": C[f"{tagp}-ALTG"]["v"] < 0,
        "S1-3 dI > 0, CI lower > 0": F["dI"] > 0 and F["dI_ci"][0] > 0,
        "S1-4 >= 4/5 positive specialists": F["per_specialist"]["n_pos"] >= 4,
        "S1-5 retention dI >= 0.75 x frozen": dIF > 0 and F["dI"] >= 0.75 * dIF,
        "S1-7 not confined (pos>=128 < ALT-G; >=4/5 groups better; max share <= 0.5)":
            F["ge128_vs_ref"] < 0 and F["group_vs_ref"]["n_better"] >= 4 and F["group_vs_ref"]["max_share"] <= 0.5,
    }
    return {"pass": all(crit.values()), "criteria": crit, "failed": [k for k, v in crit.items() if not v]}


def stage1(a):
    from csr import asrs2_paths as P
    AF = json.loads((P.ANALYSIS / f"{a.frozen}.json").read_text())["seeds"]["0"]
    A = json.loads((P.ANALYSIS / f"{a.final}.json").read_text())["seeds"]["0"]
    bm = json.loads((P.ART / a.base_check).read_text()) if a.base_check else {"pass": False}
    base_ok = bool(bm.get("pass"))
    res = {"time": guard.now(), "gen": a.gen, "frozen": a.frozen, "final": a.final, "base_check": bm,
           "candidates": {"R1_hard": candidate_criteria(A, AF, "SEM", base_ok),
                          "R1_soft": candidate_criteria(A, AF, "SOFT0", base_ok)},
           "frozen_control": {"FROZ-G": A["contrasts"].get("FROZ-G"), "FROZ-ALTG": A["contrasts"].get("FROZ-ALTG"),
                              "dI": A["functional"].get("FROZ", {}).get("dI")}}
    passing = [k for k, v in res["candidates"].items() if v["pass"]]
    if passing:
        ces = {k: A["contrasts"][("SEM" if k == "R1_hard" else "SOFT0") + "-G"]["v"] for k in passing}
        res["selected"] = min(passing, key=lambda k: (ces[k], k != "R1_hard"))
    res["pass"] = bool(passing)
    res["traj"] = {}
    for tg in (a.traj.split(",") if a.traj else []):
        X = json.loads((P.ANALYSIS / f"{tg}.json").read_text())
        S = X["seeds"]["0"]
        res["traj"][tg] = {"step": X["step"], "ce": S["ce_mean"],
                           **{k: {"dI": v["dI"], "n_pos": v["per_specialist"]["n_pos"], "switch": v["switch"],
                                  "agree": v["agree_region_prior"], "msd": v["mean_specialist_delta"]}
                              for k, v in S["functional"].items()}}
    guard.atomic_write_json(P.ART / f"stage1_gen{a.gen}.json", res, mode=0o400)
    print(json.dumps({k: v for k, v in res.items() if k != "traj"}, default=str, indent=1)[:5000])


def selftest():
    """Synthetic check of the functional metric: own specialists better on own groups -> I > 0; shuffled -> ~0."""
    rng = np.random.default_rng(0)

    class Fake:
        pass
    E = Fake()
    K, N, Pp = 5, 400_000, 200
    E.K, E.N, E.P = K, N, Pp
    E.paper = rng.integers(0, Pp, N)
    E.half = rng.random(Pp) < 0.5
    lab = rng.integers(0, K, Pp)[E.paper]
    E.z = {"lab_sem": lab, "lab_shuf": rng.permutation(lab)}
    E.base_st = rng.integers(0, 135, N)
    d = rng.normal(0, 0.1, (N, K))
    d[np.arange(N), lab] -= 0.05
    e = np.where(rng.random(N) < 0.7, lab, rng.integers(0, K, N))
    q = np.full((N, K), 0.1)
    q[np.arange(N), e] = 0.6
    E.d = lambda arm: d
    E.dep = lambda arm: (e, q)
    f = functional(E, "x", "lab_sem")
    f2 = functional(E, "x", "lab_shuf")
    assert f["I"] > 0.03, f["I"]
    assert abs(f2["I"]) < 0.02, f2["I"]
    print(json.dumps({"selftest": "ok", "I_sem": f["I"], "I_shuf": f2["I"]}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--eval-dir")
    ap.add_argument("--part", default="SEL")
    ap.add_argument("--tag")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--frozen")
    ap.add_argument("--final")
    ap.add_argument("--traj", default="")
    ap.add_argument("--base-check", default="")
    ap.add_argument("--gen", type=int, default=0)
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    if a.cmd == "ckpt":
        cmd_ckpt(a)
    elif a.cmd == "stage0f":
        print(json.dumps(stage0f(a.tag), default=str, indent=1))
    elif a.cmd == "stage1":
        stage1(a)
    elif a.cmd == "selftest":
        selftest()


if __name__ == "__main__":
    main()
