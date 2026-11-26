#!/usr/bin/env python
"""csr-crossfit-semantic-utility-residual-v1 analysis (CPU; prereg sections 8-11).

  csur1_analyze.py stage1 --eval NPZ [--out NAME]           Stage-1 gates + selection (SEL, seed 100)
  csur1_analyze.py stage2 --evals NPZ_a,NPZ_b [--out NAME]   Stage-2 promotion P1-P4 (EVAL, seeds 100/101)
  csur1_analyze.py final --evals NPZ x4 --refs JSON [--out NAME]   final criteria F1-F4 + outcome (VAL)
  csur1_analyze.py selftest

Functional metric (exposure-matched own-specialist usefulness, jsr/orv1 lineage): g_e(t) = mean_{j != e} d[t, j] - d[t, e]
on swap-subset tokens (d = CE with LIVE specialist j forced - CE_base). Own specialist sigma(k) = argmax_e routing lift
P(e | k) / P(e) of the bank's DEPLOYED (student) assignment on one paper half; measured on the other half: own set =
group-k tokens routed to sigma(k), other set = other-group tokens routed to sigma(k); strata q_e quintile x margin
tercile x base-CE quintile x position bin x doc-length tercile x target-frequency tercile; common support;
I_k = sum_s w_s (mean_own_s g - mean_oth_s g) with own-stratum weights; both directions averaged. Groups = paper-level
C6 semantic clusters (primary), region-level p_sem argmax and real domains (secondary/post hoc). CIs: per-paper
influence contributions (paper bootstrap within seed; hierarchical seed x paper bootstrap across seeds)."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402
from jsr_analyze import boot_weights, ci, hier, nmi_counts, pathology, qcut, summarize  # noqa: E402

MIN_SET = 2000
UTIL = ("SU", "SH", "SO", "UO")


def halves(papers):
    import numpy as np
    from orv1_tables import u_hash
    return np.asarray([u_hash("csur1-half", str(p)) < 0.5 for p in papers])


class Seed:
    def __init__(self, path, feats):
        import numpy as np
        self.path = Path(path)
        self.z = dict(np.load(path))
        self.js = json.loads(self.path.with_suffix(".json").read_text())
        z = self.z
        self.banks = [str(b) for b in z["bank_names"]]
        self.util = [b for b in self.banks if f"eT_{b}" in z]
        cb = z["ce_base"].astype(np.float64)
        for b in self.banks:
            z[f"ce_{b}"] = cb + z[f"dce_{b}"].astype(np.float64)
            if b in self.util:
                z[f"ce_T{b}"] = cb + z[f"dceT_{b}"].astype(np.float64)
        self.papers = z["papers"]
        self.P = len(self.papers)
        self.n = np.bincount(z["paper"], minlength=self.P).astype(np.float64)
        self.half = halves(self.papers)
        self.sub_idx = np.nonzero(z["sub"].astype(bool))[0]
        lf = np.log(feats["freq"][z["tgt"]].astype(np.float64) + 1)
        self.fb = qcut(lf, [1 / 3, 2 / 3])
        self.cb = qcut(cb, [0.2, 0.4, 0.6, 0.8])
        self.pb = np.searchsorted([128, 512], z["pos"], side="right")
        self.db = qcut(z["doclen"].astype(np.float64), [1 / 3, 2 / 3])
        self.base_st = ((self.fb * 5 + self.cb) * 3 + self.pb) * 3 + self.db

    def ce_sums(self, name, mask=None):
        import numpy as np
        key = "ce_base" if name == "base" else f"ce_{name}"
        w = self.z[key].astype(np.float64)
        p = self.z["paper"]
        if mask is not None:
            return np.bincount(p[mask], weights=w[mask], minlength=self.P)
        return np.bincount(p, weights=w, minlength=self.P)

    def ce(self, name, mask=None):
        import numpy as np
        n = self.n if mask is None else np.bincount(self.z["paper"][mask], minlength=self.P)
        return float(self.ce_sums(name, mask).sum() / max(1, n.sum()))

    def load(self, b, teacher=False):
        import numpy as np
        e = self.z[f"eT_{b}" if teacher else f"e_{b}"].astype(np.int64)
        return np.bincount(e[e >= 0], minlength=16) / max(1, len(e))

    def counts(self, b, lab_key="lab_sem", nlab=16, teacher=False):
        import numpy as np
        z = self.z
        lab = z[lab_key].astype(np.int64)
        e = z[f"eT_{b}" if teacher else f"e_{b}"].astype(np.int64)
        ok = (lab >= 0) & (e >= 0)
        idx = (z["paper"][ok].astype(np.int64) * nlab + lab[ok]) * 16 + e[ok]
        return np.bincount(idx, minlength=self.P * nlab * 16).reshape(self.P, nlab, 16).astype(np.float64)


def load_seeds(paths):
    import numpy as np
    from csr.csur1_paths import ORV1
    feats = dict(np.load(ORV1 / "tables/token_feats.npz"))
    seeds = [Seed(p, feats) for p in paths]
    for S in seeds[1:]:
        if not np.array_equal(S.papers, seeds[0].papers) or len(S.z["ce_base"]) != len(seeds[0].z["ce_base"]):
            raise guard.GuardError("token sets differ across seeds")
    return seeds


# ----------------------------------------------------------------------------------------------- functional
def own_map(S, b, lab_key, mask_tok):
    import numpy as np
    z = S.z
    lab = z[lab_key].astype(np.int64)
    e = z[f"e_{b}"].astype(np.int64)
    ok = mask_tok & (lab >= 0) & (e >= 0)
    C = np.bincount(lab[ok] * 16 + e[ok], minlength=256).reshape(16, 16).astype(np.float64)
    pe = C.sum(0) / max(1.0, C.sum())
    lift = (C / np.maximum(C.sum(1, keepdims=True), 1)) / np.maximum(pe, 1e-12)
    return lift.argmax(1)


def functional(S, b, lab_key="lab_sem", nlab=16):
    """Cross-fitted exposure-matched own-specialist advantage of bank b's deployed routing. Returns I, I_k (per group),
    per-paper influence IF (P,) and per-group IFk (nlab, P), reliability, counts."""
    import numpy as np
    z = S.z
    si = S.sub_idx
    d = z[f"d_{b}"].astype(np.float64)
    lab = z[lab_key].astype(np.int64)[si]
    e = z[f"e_{b}"].astype(np.int64)[si]
    pap = z["paper"][si].astype(np.int64)
    tok_half = S.half[z["paper"].astype(np.int64)]
    sub_half = tok_half[si]
    rowsum = d.sum(1)
    q = z[f"q_{b}"].astype(np.float64)
    Ik = np.full((2, nlab), np.nan)
    IFk = np.zeros((2, nlab, S.P))
    sig = []
    for di, (fit_h, meas_h) in enumerate(((True, False), (False, True))):
        sigma = own_map(S, b, lab_key, tok_half == fit_h)
        sig.append(sigma.tolist())
        for k in range(nlab):
            ek = int(sigma[k])
            inv = (sub_half == meas_h) & (e == ek) & (lab >= 0)
            own, oth = inv & (lab == k), inv & (lab != k)
            if own.sum() < MIN_SET or oth.sum() < MIN_SET:
                continue
            g = (rowsum - d[:, ek]) / 15.0 - d[:, ek]
            qe = q[:, ek]
            m_ = np.sort(q, 1)
            marg = m_[:, -1] - m_[:, -2]
            qb = np.searchsorted(np.quantile(qe[inv], [0.2, 0.4, 0.6, 0.8]), qe, side="right")
            mb = np.searchsorted(np.quantile(marg[inv], [1 / 3, 2 / 3]), marg, side="right")
            st = (S.base_st[si] * 5 + qb) * 3 + mb
            nst = 135 * 15
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
            IFk[di, k] = (np.bincount(pap[own][ko], weights=ifo, minlength=S.P)
                          + np.bincount(pap[oth][kn], weights=ifn, minlength=S.P))
    scored = ~np.isnan(Ik).any(0)
    if scored.sum() == 0:
        return {"I": float("nan"), "n_scored": 0, "IF": np.zeros(S.P), "IFk": np.zeros((nlab, S.P)),
                "I_k": [None] * nlab, "sigma": sig}
    Ig = 0.5 * (Ik[0] + Ik[1])
    IFg = 0.5 * (IFk[0] + IFk[1])
    IF = IFg[scored].mean(0)
    rel = float(np.corrcoef(Ik[0][scored], Ik[1][scored])[0, 1]) if scored.sum() >= 3 else float("nan")
    return {"I": float(np.nanmean(Ig[scored])), "I_k": [None if not scored[k] else float(Ig[k]) for k in range(nlab)],
            "n_scored": int(scored.sum()), "n_pos": int((Ig[scored] > 0).sum()), "IF": IF, "IFk": IFg,
            "split_half_r": rel, "sigma": sig, "distinct_sigma": [len(set(s)) for s in sig]}


def func_vals(fs, W):
    import numpy as np
    vals = np.asarray([f["I"] for f in fs])
    reps = np.stack([v + W @ f["IF"] - f["IF"].sum() for v, f in zip(vals, fs)])
    return vals, reps


def func_diff(fa, fb, W):
    import numpy as np
    vals = np.asarray([a["I"] - b["I"] for a, b in zip(fa, fb)])
    reps = np.stack([v + W @ (a["IF"] - b["IF"]) - (a["IF"] - b["IF"]).sum() for v, a, b in zip(vals, fa, fb)])
    return vals, reps


def per_specialist(fs, W, hierarchical=False):
    """Per semantic group k (own specialist sigma(k)): pooled I_k over seeds with paper-bootstrap CI; counts of
    individually significant groups and the largest share of the positive contribution."""
    import numpy as np
    nk = len(fs[0]["I_k"])
    out = []
    for k in range(nk):
        vals = [f["I_k"][k] for f in fs]
        if any(v is None for v in vals):
            out.append({"k": k, "I": None})
            continue
        reps = np.stack([v + W @ f["IFk"][k] - f["IFk"][k].sum() for v, f in zip(vals, fs)])
        out.append({"k": k, "I": float(np.mean(vals)), "per_seed": [float(v) for v in vals],
                    "n_pos_seeds": int((np.asarray(vals) > 0).sum()), "ci": ci(reps.mean(0))})
    sc = [o for o in out if o["I"] is not None]
    pos = np.asarray([max(0.0, o["I"]) for o in sc])
    return {"groups": out, "n_scored": len(sc), "n_pos": int(sum(o["I"] > 0 for o in sc)),
            "n_sig": int(sum(o["ci"][0] > 0 for o in sc)),
            "max_pos_share": float(pos.max() / pos.sum()) if pos.sum() > 0 else float("nan")}


# ----------------------------------------------------------------------------------------------- CE + routing
def ce_contrast(seeds, a, b, W, mask_fn=None):
    import numpy as np
    vals, reps = [], []
    for S in seeds:
        m = mask_fn(S) if mask_fn else None
        sa, sb = S.ce_sums(a, m), S.ce_sums(b, m)
        n = S.n if m is None else np.bincount(S.z["paper"][m], minlength=S.P).astype(np.float64)
        vals.append(float((sa.sum() - sb.sum()) / n.sum()))
        reps.append((W @ sa - W @ sb) / np.maximum(W @ n, 1))
    return np.asarray(vals), np.stack(reps)


def nmi_val(S, b, lab_key, teacher=False, nlab=16):
    return float(nmi_counts(S.counts(b, lab_key, nlab, teacher).sum(0)))


def nmi_contrast(seeds, a, b, W, lab_key="lab_sem"):
    import numpy as np
    vals, reps = [], []
    for S in seeds:
        Ca, Cb = S.counts(a, lab_key), S.counts(b, lab_key)
        vals.append(float(nmi_counts(Ca.sum(0)) - nmi_counts(Cb.sum(0))))
        reps.append(nmi_counts((W @ Ca.reshape(S.P, -1)).reshape(-1, 16, 16))
                    - nmi_counts((W @ Cb.reshape(S.P, -1)).reshape(-1, 16, 16)))
    return np.asarray(vals), np.stack(reps)


def swap_summary(S, b, sigma, lab_key="lab_sem"):
    """Complete group x specialist swap matrix M[k, j] = mean d over group-k subset tokens (unmatched); diagonal
    advantage of the own specialist sigma(k): mean_{j != sigma(k)} M[k, j] - M[k, sigma(k)] (positive = own better)."""
    import numpy as np
    z = S.z
    si = S.sub_idx
    lab = z[lab_key].astype(np.int64)[si]
    d = z[f"d_{b}"].astype(np.float64)
    M = np.full((16, 16), np.nan)
    for k in range(16):
        m = lab == k
        if m.any():
            M[k] = d[m].mean(0)
    adv = []
    for k in range(16):
        j = int(sigma[k])
        adv.append(float(np.nanmean(np.delete(M[k], j)) - M[k, j]))
    return M, np.asarray(adv)


def position_masks():
    return {"0_127": lambda S: S.z["pos"] < 128, "128_511": lambda S: (S.z["pos"] >= 128) & (S.z["pos"] < 512),
            "512p": lambda S: S.z["pos"] >= 512}


def stability(seeds, b):
    """Assignment stability: across disjoint views (same region, view A vs B: P(e_A == e_B) of the deployed and teacher
    assignment) and across seeds (NMI between two seeds' deployed assignments on the same tokens)."""
    import numpy as np
    out = {"views": [], "views_teacher": []}
    for S in seeds:
        z = S.z
        key = z["doc"].astype(np.int64) * 8 + (z["blk"].astype(np.int64) // 2)
        isA = (z["blk"] % 2) == 0
        for nm, ek in (("views", f"e_{b}"), ("views_teacher", f"eT_{b}")):
            if ek not in z:
                continue
            e = z[ek].astype(np.int64)
            # one assignment per view (blocks have a single assignment for util banks; token-level for G -> mode)
            vk = key * 2 + (~isA)
            uk, inv = np.unique(vk, return_inverse=True)
            mode = np.zeros(len(uk), np.int64)
            cnt = np.zeros((len(uk), 16), np.int64)
            np.add.at(cnt, (inv, e), 1)
            mode = cnt.argmax(1)
            reg, isb = uk // 2, uk % 2
            ra = dict(zip(reg[isb == 0], mode[isb == 0]))
            pairs = [(ra[r], m) for r, m, b_ in zip(reg, mode, isb) if b_ == 1 and r in ra]
            agree = float(np.mean([x == y for x, y in pairs])) if pairs else float("nan")
            out[nm].append(agree)
    if len(seeds) >= 2:
        nm = []
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                ei = seeds[i].z[f"e_{b}"].astype(np.int64)
                ej = seeds[j].z[f"e_{b}"].astype(np.int64)
                C = np.bincount(ei * 16 + ej, minlength=256).reshape(16, 16)
                nm.append(float(nmi_counts(C)))
        out["cross_seed_nmi"] = nm
    return out


# ----------------------------------------------------------------------------------------------- core
def core(seeds, hierarchical):
    import numpy as np
    from csr.csur1_paths import BOOT_SEED, HIER_SEED, N_BOOT, N_HIER
    P = seeds[0].P
    W = boot_weights(P, N_BOOT, BOOT_SEED)
    names = seeds[0].banks
    util = seeds[0].util
    hrep = (lambda v, r, k: hier(v, r, N_HIER, HIER_SEED + k)) if hierarchical else (lambda v, r, k: None)
    R = {"n_seeds": len(seeds), "n_papers": P, "per_seed": []}
    for S in seeds:
        R["per_seed"].append({"eval": str(S.path), "step": S.js["step"], "seen": S.js["seen"], "ce_base": S.ce("base"),
                              **{f"ce_{b}": S.ce(b) for b in names}, **{f"ceT_{b}": S.ce(f"T{b}") for b in util}})
    pairs = [(x, y) for x, y in (("SU", "SH"), ("SU", "G"), ("SU", "SO"), ("SU", "UO"), ("SH", "G"), ("SO", "G"),
                                 ("UO", "G"), ("SU", "base"), ("G", "base"), ("SH", "base"), ("SO", "base"),
                                 ("UO", "base"), ("TSU", "TSH"), ("TSU", "G"), ("TSU", "SU"), ("TSH", "SH"),
                                 ("TSO", "SO"), ("TUO", "UO"), ("TSU", "base"))
             if x.lstrip("T") in names + ["base"] and y.lstrip("T") in names + ["base"] or x == "base"]
    R["ce"] = {}
    for j, (x, y) in enumerate(pairs):
        v, r = ce_contrast(seeds, x, y, W)
        R["ce"][f"{x}-{y}"] = summarize(v, r, hrep(v, r, j))
    R["ce_by_position"] = {}
    for pn, mf in position_masks().items():
        R["ce_by_position"][pn] = {}
        for j, (x, y) in enumerate((("SU", "SH"), ("SU", "G"), ("SU", "base"), ("G", "base"), ("TSU", "TSH"),
                                    ("TSU", "base"), ("SH", "base"), ("SO", "base"), ("UO", "base"))):
            if x.lstrip("T") in names and (y == "base" or y.lstrip("T") in names):
                v, r = ce_contrast(seeds, x, y, W, mf)
                R["ce_by_position"][pn][f"{x}-{y}"] = summarize(v, r)
    # retention of the teacher's semantic-over-shuffled CE advantage by the causal student
    if "SU" in util and "SH" in util:
        ts = np.asarray([S.ce("TSH") - S.ce("TSU") for S in seeds])
        ss = np.asarray([S.ce("SH") - S.ce("SU") for S in seeds])
        tg = np.asarray([S.ce("base") - S.ce("TSU") for S in seeds])
        sg = np.asarray([S.ce("base") - S.ce("SU") for S in seeds])
        R["retention"] = {"teacher_sem_minus_shuf": ts.tolist(), "student_sem_minus_shuf": ss.tolist(),
                          "retention_pooled": float(ss.sum() / ts.sum()) if ts.sum() > 0 else float("nan"),
                          "teacher_adv_pooled_positive": bool(ts.sum() > 0),
                          "gain_retention_vs_base": float(sg.sum() / tg.sum()) if tg.sum() > 0 else float("nan")}
        R["retention_by_position"] = {}
        for pn, mf in position_masks().items():
            t_ = sum(S.ce("TSH", mf(S)) - S.ce("TSU", mf(S)) for S in seeds)
            s_ = sum(S.ce("SH", mf(S)) - S.ce("SU", mf(S)) for S in seeds)
            R["retention_by_position"][pn] = {"teacher": t_ / len(seeds), "student": s_ / len(seeds),
                                              "retention": (s_ / t_) if t_ > 0 else float("nan")}
    # functional
    F = {b: [functional(S, b) for S in seeds] for b in names}
    R["func"] = {}
    for b in names:
        v, r = func_vals(F[b], W)
        R["func"][f"I_{b}"] = {**summarize(v, r), "n_pos": [f.get("n_pos") for f in F[b]],
                               "n_scored": [f["n_scored"] for f in F[b]],
                               "split_half_r": [f.get("split_half_r") for f in F[b]],
                               "distinct_sigma": [f.get("distinct_sigma") for f in F[b]],
                               "I_k": [f["I_k"] for f in F[b]]}
    for j, (x, y) in enumerate((("SU", "SH"), ("SU", "G"), ("SU", "SO"), ("SU", "UO"), ("SH", "G"), ("SO", "SH"),
                                ("UO", "SH"))):
        if x in names and y in names:
            v, r = func_diff(F[x], F[y], W)
            R["func"][f"I_{x}-I_{y}"] = summarize(v, r, hrep(v, r, 100 + j))
    R["per_specialist"] = {b: per_specialist(F[b], W) for b in names}
    # secondary functional groupings (region-level semantic argmax; real domains, post hoc)
    R["func_secondary"] = {}
    for lk, nl in (("lab_rsem", 16), ("domain", 5)):
        Fs = {b: [functional(S, b, lk, nl) for S in seeds] for b in names}
        R["func_secondary"][lk] = {f"I_{b}": [f["I"] for f in Fs[b]] for b in names}
        if "SU" in names and "SH" in names:
            v, r = func_diff(Fs["SU"], Fs["SH"], W)
            R["func_secondary"][lk]["I_SU-I_SH"] = summarize(v, r)
    # swap matrices + diagonal advantage
    R["swap"] = {}
    for b in names:
        Ms, advs = [], []
        for S, f in zip(seeds, F[b]):
            M, adv = swap_summary(S, b, f["sigma"][0])
            Ms.append(M)
            advs.append(adv)
        A = np.asarray(advs)
        R["swap"][b] = {"matrix_seed_mean": np.nanmean(Ms, 0).round(6).tolist(),
                        "diag_adv_mean": float(np.nanmean(A)), "diag_adv_per_seed": np.nanmean(A, 1).tolist(),
                        "n_groups_diag_adv_pos": [int((a > 0).sum()) for a in A]}
    # routing
    R["routing"] = {}
    for b in names:
        rr = {"nmi_sem": [nmi_val(S, b, "lab_sem") for S in seeds],
              "nmi_rsem": [nmi_val(S, b, "lab_rsem") for S in seeds],
              "nmi_domain_posthoc": [float(nmi_counts(S.counts(b, "domain", 5).sum(0))) for S in seeds],
              "nmi_shuf": [nmi_val(S, b, "lab_shuf") for S in seeds],
              "load": [S.load(b).round(5).tolist() for S in seeds],
              "pathology": [pathology(S.load(b)) for S in seeds],
              "util_entropy": [float(-(l_[l_ > 0] * np.log(l_[l_ > 0])).sum()) for l_ in (S.load(b) for S in seeds)],
              "stability": stability(seeds, b)}
        if b in util:
            rr["teacher_nmi_sem"] = [nmi_val(S, b, "lab_sem", True) for S in seeds]
            rr["teacher_nmi_rsem"] = [nmi_val(S, b, "lab_rsem", True) for S in seeds]
            rr["teacher_pathology"] = [pathology(S.load(b, True)) for S in seeds]
            bs = np.asarray([S.z[f"blkstats_{b}"] for S in seeds]).sum(0)
            rr["teacher_student_by_view"] = [{"view": i, "n": float(bs[i, 0]),
                                              "kl": float(bs[i, 1] / max(1, bs[i, 0])),
                                              "agree": float(bs[i, 2] / max(1, bs[i, 0])),
                                              "q_entropy": float(bs[i, 3] / max(1, bs[i, 0]))} for i in range(8)]
            agree_tok = [float((S.z[f"e_{b}"] == S.z[f"eT_{b}"]).mean()) for S in seeds]
            rr["teacher_student_token_agreement"] = agree_tok
        R["routing"][b] = rr
    for x, y in (("SU", "SH"), ("SU", "G")):
        if x in names and y in names:
            v, r = nmi_contrast(seeds, x, y, W)
            R["routing"][f"nmi_sem_{x}-{y}"] = summarize(v, r)
    # per semantic group CE (seed mean) and per real domain (post hoc)
    R["per_group_ce"] = {}
    for lk, nl in (("lab_sem", 16), ("domain", 5)):
        tab = {}
        for b in [x for x in names]:
            tab[b] = [float(np.mean([S.ce(b, S.z[lk] == k) - S.ce("base", S.z[lk] == k) for S in seeds])) for k in range(nl)]
        R["per_group_ce"][lk] = tab
    R["_F"] = F
    return R, W


def strip(R):
    R.pop("_F", None)
    return R


def write(name, R):
    from csr.csur1_paths import ART
    p = ART / f"{name}.json"
    guard.atomic_write_json(p, strip(R), mode=0o400)
    return p


# ----------------------------------------------------------------------------------------------- stages
def stage1(a):
    """Gates per candidate c (SU_c): student CE_SU_c - CE_G <= 0.002 and no pathology (student and teacher loads);
    selection = argmax (I_SU_c - I_SH_c) among gated candidates; ties (< 1e-5) -> base point, then grid order."""
    import numpy as np
    seeds = load_seeds([a.eval])
    S = seeds[0]
    W = boot_weights(S.P, 2000, 20261211)
    cands = sorted({b.split("_", 1)[1] for b in S.banks if b.startswith("SU_")},
                   key=lambda t: [x[0] for x in __import__("csur1_configs").GRID].index(t))
    out = {"time": guard.now(), "eval": a.eval, "step": S.js["step"], "seen": S.js["seen"], "ce_base": S.ce("base"),
           "ce_G": S.ce("G"), "G_pathology": pathology(S.load("G")), "candidates": {}}
    FG = functional(S, "G")
    out["I_G"] = FG["I"]
    for c in cands:
        su, sh = f"SU_{c}", f"SH_{c}"
        fsu, fsh = functional(S, su), functional(S, sh)
        v, r = func_diff([fsu], [fsh], W)
        ps, pt = pathology(S.load(su)), pathology(S.load(su, True))
        rec = {"ce_SU": S.ce(su), "ce_SH": S.ce(sh), "ceT_SU": S.ce(f"T{su}"), "ceT_SH": S.ce(f"T{sh}"),
               "ce_SU_minus_G": S.ce(su) - S.ce("G"), "I_SU": fsu["I"], "I_SH": fsh["I"],
               "I_SU_minus_I_SH": float(v[0]), "I_diff_ci": ci(r[0]), "n_pos_SU": fsu.get("n_pos"),
               "split_half_r_SU": fsu.get("split_half_r"), "pathology_SU_student": ps, "pathology_SU_teacher": pt,
               "nmi_sem_SU": nmi_val(S, su, "lab_sem"), "nmi_sem_SH": nmi_val(S, sh, "lab_sem"),
               "nmi_sem_SU_teacher": nmi_val(S, su, "lab_sem", True)}
        rec["gate_ce"] = rec["ce_SU_minus_G"] <= 0.002
        rec["gate_pathology"] = ps["ok"] and pt["ok"]
        rec["gated"] = rec["gate_ce"] and rec["gate_pathology"]
        out["candidates"][c] = rec
    ok = {c: r for c, r in out["candidates"].items() if r["gated"] and np.isfinite(r["I_SU_minus_I_SH"])}
    if ok:
        best = max(r["I_SU_minus_I_SH"] for r in ok.values())
        tied = [c for c in cands if c in ok and best - ok[c]["I_SU_minus_I_SH"] < 1e-5]
        sel = "bp" if "bp" in tied else tied[0]
        out["selected"] = sel
        out["selected_hp"] = {**__import__("csur1_configs").BASE, **dict(__import__("csur1_configs").GRID)[sel]}
        out["stage1_pass"] = True
        out["warning_best_score_nonpositive"] = bool(best <= 0)
    else:
        out["selected"] = None
        out["stage1_pass"] = False
    p = write(a.out or "stage1", out)
    print(json.dumps({k: v for k, v in out.items() if k != "candidates"}, default=str))
    for c, r in out["candidates"].items():
        print(c, {k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items() if not isinstance(v, dict)})
    print(p)


def stage2(a):
    import numpy as np
    seeds = load_seeds(a.evals.split(","))
    R, W = core(seeds, hierarchical=False)
    P1 = all(v > 0 for v in R["func"]["I_SU-I_SH"]["per_seed"])
    ret = R["retention"]
    P2 = bool(ret["teacher_adv_pooled_positive"] and ret["retention_pooled"] >= 0.4)
    d_g = [R["per_seed"][i]["ce_SU"] - R["per_seed"][i]["ce_G"] for i in range(len(seeds))]
    P3 = all(x <= 0.002 for x in d_g)
    P4 = all(p["ok"] for p in R["routing"]["SU"]["pathology"]) and all(p["ok"] for p in R["routing"]["SU"]["teacher_pathology"])
    R["criteria"] = {"P1_func_SU_gt_SH_both_seeds": P1, "P1_values": R["func"]["I_SU-I_SH"]["per_seed"],
                     "P2_retention_ge_0.4": P2, "P2_values": ret, "P3_student_CE_le_G_plus_0.002_each_seed": P3,
                     "P3_values": d_g, "P4_no_pathology_SU": P4}
    R["promote"] = bool(P1 and P2 and P3 and P4)
    R["time"] = guard.now()
    p = write(a.out or "stage2", R)
    print(json.dumps(R["criteria"], default=str), "PROMOTE" if R["promote"] else "NOT PROMOTED", p)


def final(a):
    import numpy as np
    seeds = load_seeds(a.evals.split(","))
    R, W = core(seeds, hierarchical=True)
    refs = json.loads(Path(a.refs).read_text()) if a.refs else {}
    R["refs"] = refs
    fd = R["func"]["I_SU-I_SH"]
    F1 = fd["hier_ci"][0] > 0 and fd["n_pos"] >= 3
    cg = R["ce"]["SU-G"]
    F2 = cg["hier_ci"][1] < 0.002
    ps = R["per_specialist"]["SU"]
    F3 = ps["n_sig"] >= 4 and ps["max_pos_share"] <= 0.35
    F4 = all(p["ok"] for p in R["routing"]["SU"]["pathology"])
    R["criteria"] = {"F1_func_SU_gt_SH": F1, "F2_student_noninferior_G": F2, "F3_multiple_specialists": F3,
                     "F4_integrity": F4}
    R["positive"] = bool(F1 and F2 and F3 and F4)
    if R["positive"]:
        R["outcome"] = "POSITIVE"
    elif F1 and F3 and not F2:
        R["outcome"] = "SPECIALISED_BUT_COSTLY"
    elif F2 and not F1:
        R["outcome"] = "ACCURACY_WITHOUT_SEMANTIC_FUNCTION"
    elif F1 and not F3:
        R["outcome"] = "FUNCTION_CONCENTRATED"
    else:
        R["outcome"] = "NEGATIVE"
    R["time"] = guard.now()
    p = write(a.out or "final_analysis", R)
    print(json.dumps(R["criteria"]), R["outcome"], p)


def selftest(a):
    """Planted functional effect is recovered and a null is ~0 (synthetic Seed objects)."""
    import numpy as np
    rng = np.random.default_rng(0)

    class Fake:
        pass
    out = {}
    for planted in (0.01, 0.0):
        S = Fake()
        N, Pp = 400_000, 400
        S.P = Pp
        paper = rng.integers(0, Pp, N)
        lab = (paper % 16).astype(np.int64)
        e = np.where(rng.random(N) < 0.7, (lab + 3) % 16, rng.integers(0, 16, N))
        d = rng.normal(0, 0.05, (N, 16))
        own = e == (lab + 3) % 16
        d[np.arange(N), e] -= planted * own
        S.z = {"d_X": d, "e_X": e, "q_X": rng.dirichlet(np.ones(16), N), "lab_sem": lab, "paper": paper}
        S.sub_idx = np.arange(N)
        S.half = rng.random(Pp) < 0.5
        S.base_st = rng.integers(0, 135, N)
        f = functional(S, "X")
        out[f"planted_{planted}"] = f["I"]
    ok = abs(out["planted_0.01"] - 0.01) < 0.002 and abs(out["planted_0.0"]) < 0.001
    print(out, "OK" if ok else "FAIL")
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stage1", "stage2", "final", "selftest"])
    ap.add_argument("--eval")
    ap.add_argument("--evals")
    ap.add_argument("--refs")
    ap.add_argument("--out")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    {"stage1": stage1, "stage2": stage2, "final": final, "selftest": selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
