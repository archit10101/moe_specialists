#!/usr/bin/env python
"""joint-semantic-residual-pretrain-v1 analysis (CPU; protocol sections 7, 9, 10, 11).

  jsr_analyze.py stage1 --eval NPZ --run RUNDIR [--out NAME]          Stage-1 rejection + selection (SEL)
  jsr_analyze.py stage2 --evals NPZ0,NPZ1 [--out NAME]                Stage-2 promotion (EVAL)
  jsr_analyze.py stage3 --evals NPZ0..3 --refs JSON [--out NAME]      Stage-3 final criteria + interpretation (VAL)
  jsr_analyze.py selftest                                             synthetic checks of the estimators

Functional metric (9.1): g_e(t) = mean_{j != e} d[t, j] - d[t, e]. Own specialist sigma(k) = argmax_e routing lift on
one paper half (all tokens), measured on the other half's swap-matrix tokens: own set O = group-k tokens with e_t =
sigma(k), other set N = other-group tokens with e_t = sigma(k); strata q_e quintile x margin tercile (routed banks only)
x base-CE quintile x position bin x doc-length tercile x target-frequency tercile; common support; weights = own
stratum distribution; I_k = sum_s w_s (mean_O_s g - mean_N_s g); cross-fitted (both directions averaged). CIs: linearised
per-paper influence contributions with fixed full-sample weights (paper bootstrap within seed; hierarchical seed x
paper bootstrap across seeds with a shared paper replicate)."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402

MIN_SET = 2000


# ----------------------------------------------------------------------------------------------- basic helpers
def qcut(x, qs):
    import numpy as np
    return np.searchsorted(np.quantile(x, qs), x, side="right")


def nmi_counts(C):
    """Arithmetic-mean NMI 2 I / (H_rows + H_cols) of a joint count matrix (..., A, B) (batched)."""
    import numpy as np
    C = np.asarray(C, dtype=np.float64)
    n = C.sum(axis=(-2, -1), keepdims=True)
    P = C / np.maximum(n, 1e-300)
    pa, pb = P.sum(-1, keepdims=True), P.sum(-2, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        mi = np.nansum(np.where(P > 0, P * np.log(P / (pa * pb)), 0.0), axis=(-2, -1))
        ha = -np.nansum(np.where(pa > 0, pa * np.log(pa), 0.0), axis=(-2, -1))
        hb = -np.nansum(np.where(pb > 0, pb * np.log(pb), 0.0), axis=(-2, -1))
    return 2 * mi / np.maximum(ha + hb, 1e-300)


def boot_weights(P, n, seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    return rng.multinomial(P, np.full(P, 1.0 / P), size=n).astype(np.float64)


def ci(x):
    import numpy as np
    return [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))]


def halves(papers):
    import numpy as np
    from orv1_tables import u_hash
    return np.asarray([u_hash("jsr-half", str(p)) < 0.5 for p in papers])


# ----------------------------------------------------------------------------------------------- per-seed object
class Seed:
    """One evaluation npz (one seed, one partition)."""

    def __init__(self, path, feats):
        import numpy as np
        self.path = Path(path)
        self.z = dict(np.load(path))
        self.js = json.loads(self.path.with_suffix(".json").read_text())
        z = self.z
        self.banks = [str(b) for b in z["bank_names"]]
        self.papers = z["papers"]
        self.P = len(self.papers)
        self.n = np.bincount(z["paper"], minlength=self.P).astype(np.float64)
        self.half = halves(self.papers)
        sub = z["sub"].astype(bool)
        self.sub_idx = np.nonzero(sub)[0]
        lf = np.log(feats["freq"][z["tgt"]].astype(np.float64) + 1)
        self.fb = qcut(lf, [1 / 3, 2 / 3])
        self.cb = qcut(z["ce_base"], [0.2, 0.4, 0.6, 0.8])
        self.pb = np.searchsorted([128, 512], z["pos"], side="right")
        self.db = qcut(z["doclen"].astype(np.float64), [1 / 3, 2 / 3])
        self.tcls = feats["cls"][z["tgt"]].astype(np.int64)
        self.base_st = ((self.fb * 5 + self.cb) * 3 + self.pb) * 3 + self.db  # 135 strata

    def ce_sums(self, name):
        import numpy as np
        key = "ce_base" if name == "base" else f"ce_{name}"
        return np.bincount(self.z["paper"], weights=self.z[key].astype(np.float64), minlength=self.P)

    def ce(self, name):
        return float(self.ce_sums(name).sum() / self.n.sum())

    def load(self, b):
        import numpy as np
        e = self.z[f"e_{b}"].astype(np.int64)
        e = e[e >= 0]
        return np.bincount(e, minlength=16) / max(1, len(self.z[f"e_{b}"]))

    def counts(self, b, lab_key="lab_sem", nlab=16):
        """Per-paper joint counts (P, nlab, 16) of (label, routed specialist) over tokens with label >= 0."""
        import numpy as np
        z = self.z
        lab = z[lab_key].astype(np.int64) if isinstance(lab_key, str) else lab_key
        e = z[f"e_{b}"].astype(np.int64)
        ok = (lab >= 0) & (e >= 0)
        idx = (z["paper"][ok].astype(np.int64) * nlab + lab[ok]) * 16 + e[ok]
        return np.bincount(idx, minlength=self.P * nlab * 16).reshape(self.P, nlab, 16).astype(np.float64)


def pathology(load):
    import numpy as np
    l_ = np.asarray(load)
    eff = float(np.exp(-(l_[l_ > 0] * np.log(l_[l_ > 0])).sum()))
    dead = int((l_ < 1.0 / 160).sum())
    return {"load_max": float(l_.max()), "load_eff": eff, "n_dead": dead,
            "ok": bool(l_.max() <= 0.25 and eff >= 8 and dead <= 1)}


# ----------------------------------------------------------------------------------------------- functional metric
def own_map(S, b, lab_key, half_mask_tok):
    """sigma(k) = argmax_e P(e | k) / P(e) on tokens in half_mask_tok (all tokens, not only the swap subset)."""
    import numpy as np
    z = S.z
    lab = z[lab_key].astype(np.int64)
    e = z[f"e_{b}"].astype(np.int64)
    ok = half_mask_tok & (lab >= 0) & (e >= 0)
    C = np.bincount(lab[ok] * 16 + e[ok], minlength=256).reshape(16, 16).astype(np.float64)
    pe = C.sum(0) / max(1.0, C.sum())
    pek = C / np.maximum(C.sum(1, keepdims=True), 1)
    lift = pek / np.maximum(pe, 1e-12)
    return lift.argmax(1), C.sum(1)


def functional(S, b, lab_key="lab_sem", oracle=False):
    """Cross-fitted exposure-matched own-specialist advantage. Returns dict with I, per-group I_k (both directions),
    per-paper influence IF (P,), reliability, counts."""
    import numpy as np
    z = S.z
    si = S.sub_idx
    d = z[f"d_{b}"].astype(np.float64)  # (Ns, 16)
    lab_all = z[lab_key].astype(np.int64)
    lab = lab_all[si]
    e_all = z[f"e_{b}"].astype(np.int64)
    e = e_all[si]
    pap = z["paper"][si].astype(np.int64)
    tok_half = S.half[z["paper"].astype(np.int64)]
    sub_half = tok_half[si]
    rowsum = d.sum(1)
    q = None if oracle else z[f"q_{b}"].astype(np.float64)
    res = {"dirs": {}}
    Ik = np.full((2, 16), np.nan)
    IFk = np.zeros((2, 16, S.P))
    for di, (fit_h, meas_h) in enumerate(((True, False), (False, True))):
        if oracle:
            sigma = np.arange(16)
        else:
            sigma, _ = own_map(S, b, lab_key, tok_half == fit_h)
        info = {}
        for k in range(16):
            ek = int(sigma[k])
            # oracle: e == label by construction, so the routed-population "other" set is empty; the oracle
            # specialist is measured on all swap-subset tokens of the half (forced d columns), own vs other groups
            # (fix 2026-11-20, see deviations.jsonl; diagnostic only, not a selection or promotion criterion)
            inv = (sub_half == meas_h) & (lab >= 0) & ((e == ek) | oracle)
            own = inv & (lab == k)
            oth = inv & (lab != k)
            if own.sum() < MIN_SET or oth.sum() < MIN_SET:
                info[k] = {"sigma": ek, "n_own": int(own.sum()), "n_oth": int(oth.sum()), "scored": False}
                continue
            g = (rowsum - d[:, ek]) / 15.0 - d[:, ek]
            st = S.base_st[si]
            nst = 135
            if not oracle:
                qe = q[:, ek]
                m_ = q.copy()
                m_.sort(1)
                marg = m_[:, -1] - m_[:, -2]
                qb = np.searchsorted(np.quantile(qe[inv], [0.2, 0.4, 0.6, 0.8]), qe, side="right")
                mb = np.searchsorted(np.quantile(marg[inv], [1 / 3, 2 / 3]), marg, side="right")
                st = (st * 5 + qb) * 3 + mb
                nst = 135 * 15
            so, sn = st[own], st[oth]
            no = np.bincount(so, minlength=nst).astype(np.float64)
            nn = np.bincount(sn, minlength=nst).astype(np.float64)
            common = (no > 0) & (nn > 0)
            ko, kn = common[so], common[sn]
            if ko.sum() < MIN_SET or kn.sum() < MIN_SET:
                info[k] = {"sigma": ek, "n_own": int(ko.sum()), "n_oth": int(kn.sum()), "scored": False}
                continue
            go, gn_ = g[own][ko], g[oth][kn]
            so, sn = so[ko], sn[kn]
            no = np.bincount(so, minlength=nst).astype(np.float64)
            nn = np.bincount(sn, minlength=nst).astype(np.float64)
            w = no / no.sum()
            mo = np.bincount(so, weights=go, minlength=nst) / np.maximum(no, 1)
            mn = np.bincount(sn, weights=gn_, minlength=nst) / np.maximum(nn, 1)
            I = float((w * (mo - mn)).sum())
            # influence: own token t: w_s (g_t - mo_s) / no_s; other token: -w_s (g_t - mn_s) / nn_s
            ifo = w[so] * (go - mo[so]) / no[so]
            ifn = -w[sn] * (gn_ - mn[sn]) / nn[sn]
            po, pn = pap[own][ko], pap[oth][kn]
            IFk[di, k] = np.bincount(po, weights=ifo, minlength=S.P) + np.bincount(pn, weights=ifn, minlength=S.P)
            Ik[di, k] = I
            info[k] = {"sigma": ek, "n_own": int(ko.sum()), "n_oth": int(kn.sum()), "scored": True, "I": I,
                       "frac_own_matched": float(ko.mean()) if len(ko) else 0.0}
        res["dirs"][di] = info
    scored = ~np.isnan(Ik).any(0)
    if scored.sum() == 0:
        return {"I": float("nan"), "n_scored": 0, "IF": np.zeros(S.P), "Ik": Ik.tolist(), "dirs": res["dirs"]}
    Ig = 0.5 * (Ik[0] + Ik[1])
    IF = 0.5 * (IFk[0][scored] + IFk[1][scored]).mean(0)
    rel = float(np.corrcoef(Ik[0][scored], Ik[1][scored])[0, 1]) if scored.sum() >= 3 else float("nan")
    return {"I": float(np.nanmean(Ig[scored])), "I_k": [None if not scored[k] else float(Ig[k]) for k in range(16)],
            "n_scored": int(scored.sum()), "n_pos": int((Ig[scored] > 0).sum()), "split_half_r": rel,
            "IF": IF, "dirs": res["dirs"]}


def swap_matrix(S, b, lab_key="lab_sem"):
    """M[k, j] = mean over group-k swap-subset tokens of d[t, j] (CE change vs base with specialist j forced)."""
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
    return M


def piu(S, b, ref=None):
    """Per-invocation usefulness g_{e_t}(t) on the swap subset; if ref is given, ref bank's PIU reweighted to b's
    exposure distribution (base-CE quintile x pos bin x doclen tercile x q decile x margin tercile)."""
    import numpy as np
    z = S.z
    si = S.sub_idx

    def parts(bb):
        d = z[f"d_{bb}"].astype(np.float64)
        e = z[f"e_{bb}"].astype(np.int64)[si]
        de = d[np.arange(len(e)), e]
        g = (d.sum(1) - de) / 15.0 - de
        q = z[f"q_{bb}"].astype(np.float64)
        qs = np.sort(q, 1)
        qt, mg = qs[:, -1], qs[:, -1] - qs[:, -2]
        return g, qt, mg
    g, qt, mg = parts(b)
    if ref is None:
        return float(g.mean())
    g2, qt2, mg2 = parts(ref)
    allq, allm = np.concatenate([qt, qt2]), np.concatenate([mg, mg2])
    qe, me = np.quantile(allq, np.linspace(0.1, 0.9, 9)), np.quantile(allm, [1 / 3, 2 / 3])
    base = ((S.cb[si] * 3 + S.pb[si]) * 3 + S.db[si])
    s1 = (base * 10 + np.searchsorted(qe, qt, side="right")) * 3 + np.searchsorted(me, mg, side="right")
    s2 = (base * 10 + np.searchsorted(qe, qt2, side="right")) * 3 + np.searchsorted(me, mg2, side="right")
    ns = 45 * 30
    n1, n2 = np.bincount(s1, minlength=ns).astype(float), np.bincount(s2, minlength=ns).astype(float)
    c = (n1 > 0) & (n2 > 0)
    w = np.where(c, n1, 0) / n1[c].sum()
    m1 = np.bincount(s1, weights=g, minlength=ns) / np.maximum(n1, 1)
    m2 = np.bincount(s2, weights=g2, minlength=ns) / np.maximum(n2, 1)
    return float((w * (m1 - m2)).sum()), float(c[s1].mean())


# ----------------------------------------------------------------------------------------------- contrasts
def ce_contrast(seeds, a, b, W):
    """CE_a - CE_b per seed and with paper-bootstrap replicates (rows of W; papers shared across seeds)."""
    import numpy as np
    vals, reps = [], []
    for S in seeds:
        sa, sb = S.ce_sums(a), S.ce_sums(b)
        vals.append(float((sa.sum() - sb.sum()) / S.n.sum()))
        reps.append((W @ sa - W @ sb) / (W @ S.n))
    return np.asarray(vals), np.stack(reps)  # (n_seeds,), (n_seeds, R)


def hier(vals, reps, n_rep, seed):
    """Hierarchical bootstrap: seeds resampled with replacement, shared paper replicate (reps column)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    ns = len(vals)
    R = reps.shape[1]
    si = rng.integers(0, ns, size=(n_rep, ns))
    pj = rng.integers(0, R, size=n_rep)
    return reps[si, pj[:, None]].mean(1)


def summarize(vals, reps, hrep=None):
    import numpy as np
    out = {"mean": float(np.mean(vals)), "per_seed": [float(v) for v in vals], "n_pos": int((np.asarray(vals) > 0).sum()),
           "seed_ci": [ci(r) for r in reps]}
    out["pooled_ci"] = ci(reps.mean(0))
    if hrep is not None:
        out["hier_ci"] = ci(hrep)
    return out


def func_contrast(fs_a, fs_b, W):
    """fs_*: list of functional() dicts per seed. Value and paper-bootstrap replicates of I_a - I_b per seed."""
    import numpy as np
    vals = np.asarray([fa["I"] - fb["I"] for fa, fb in zip(fs_a, fs_b)])
    reps = np.stack([v + W @ (fa["IF"] - fb["IF"]) - (fa["IF"] - fb["IF"]).sum() for v, fa, fb in zip(vals, fs_a, fs_b)])
    return vals, reps


def func_single(fs, W):
    import numpy as np
    vals = np.asarray([f["I"] for f in fs])
    reps = np.stack([v + W @ f["IF"] - f["IF"].sum() for v, f in zip(vals, fs)])
    return vals, reps


def nmi_contrast(seeds, a, b, W, lab_key="lab_sem"):
    """NMI(e_a; lab) - NMI(e_b; lab) per seed + paper-bootstrap replicates."""
    import numpy as np
    vals, reps = [], []
    for S in seeds:
        Ca, Cb = S.counts(a, lab_key), S.counts(b, lab_key)
        va, vb = nmi_counts(Ca.sum(0)), nmi_counts(Cb.sum(0))
        vals.append(float(va - vb))
        ra = nmi_counts((W @ Ca.reshape(S.P, -1)).reshape(-1, 16, 16))
        rb = nmi_counts((W @ Cb.reshape(S.P, -1)).reshape(-1, 16, 16))
        reps.append(ra - rb)
    return np.asarray(vals), np.stack(reps)


# ----------------------------------------------------------------------------------------------- secondary
def secondary(S, b, oracle=False):
    import numpy as np
    z = S.z
    e = z[f"e_{b}"].astype(np.int64)
    out = {"nmi_sem": float(nmi_counts(S.counts(b).sum(0))),
           "nmi_shuf": float(nmi_counts(S.counts(b, "lab_shuf").sum(0))),
           "nmi_domain": float(nmi_counts(S.counts(b, z["domain"].astype(np.int64), 5).sum(0))),
           "nmi_target_class": float(nmi_counts(S.counts(b, S.tcls, 5).sum(0))),
           "nmi_freq_tercile": float(nmi_counts(S.counts(b, S.fb, 3).sum(0))),
           "nmi_pos_bin": float(nmi_counts(S.counts(b, S.pb, 3).sum(0))),
           "anorm_mean": float(z[f"anorm_{b}"].astype(np.float64).mean()),
           "ce_first128": float(z[f"ce_{b}"][z["pos"] < 128].astype(np.float64).mean()),
           "ce_base_first128": float(z["ce_base"][z["pos"] < 128].astype(np.float64).mean())}
    l_ = S.load(b)
    out["util_entropy"] = float(-(l_[l_ > 0] * np.log(l_[l_ > 0])).sum())
    C = S.counts(b).sum(0)
    top3 = [set(np.argsort(-C[k])[:3].tolist()) for k in range(16)]
    ov = [len(top3[i] & top3[j]) / 3 for i in range(16) for j in range(i + 1, 16)]
    out["group_top3_overlap"] = float(np.mean(ov))
    sec = S.js.get("secondary", {}).get(b, {})
    out.update({k: v for k, v in sec.items()})
    return out


def per_group_ce(S, b):
    import numpy as np
    z = S.z
    lab = z["lab_sem"].astype(np.int64)
    out = []
    for k in range(16):
        m = lab == k
        out.append(float((z[f"ce_{b}"][m].astype(np.float64) - z["ce_base"][m]).mean()) if m.any() else None)
    return out


def train_ratio(run_dir, bank, first_step=None):
    """Median realised ||beta g_sd|| / ||g_res|| over the last half of logged steps of a bank (training metrics)."""
    import numpy as np
    m = [json.loads(x) for x in (guard.ROOT / run_dir / "metrics.jsonl").read_text().splitlines()]
    rs = [(r["step"], r["banks"][bank].get("ratio")) for r in m if "banks" in r and bank in r["banks"]]
    if not rs:
        return float("nan")
    smax = max(s for s, _ in rs)
    vals = [v for s, v in rs if s >= smax / 2 and v is not None and np.isfinite(v)]
    return float(np.median(vals)) if vals else float("nan")


# ----------------------------------------------------------------------------------------------- stages
def stage1(a):
    import numpy as np
    from csr.jsr_paths import ART, BOOT_SEED, N_BOOT, ORV1
    feats = dict(np.load(ORV1 / "tables/token_feats.npz"))
    S = Seed(a.eval, feats)
    W = boot_weights(S.P, N_BOOT, BOOT_SEED)
    out = {"time": guard.now(), "eval": a.eval, "run": a.run, "ce_base": S.ce("base"), "banks": {}}
    F = {}
    for b in S.banks:
        F[b] = functional(S, b)
        p = pathology(S.load(b))
        out["banks"][b] = {"ce": S.ce(b), "I_sem": F[b]["I"], "n_scored": F[b]["n_scored"], "n_pos": F[b].get("n_pos"),
                           "split_half_r": F[b].get("split_half_r"), "pathology": p,
                           "nmi_sem": float(nmi_counts(S.counts(b).sum(0))), "train_ratio_median": train_ratio(a.run, b)}
    cand = {}
    for b in S.banks:
        if not b.startswith("D_"):
            continue
        r = float(b.split("_r")[1].split("_t")[0])
        tau = float(b.split("_t")[1])
        bb = f"B_t{tau}"
        o, ob = out["banks"][b], out["banks"][bb]
        R1 = o["ce"] > ob["ce"] + 0.002 or o["ce"] > out["ce_base"] or not np.isfinite(o["ce"])
        R2 = not o["pathology"]["ok"]
        R3 = not (o["train_ratio_median"] <= 2 * r)
        vals, reps = func_contrast([F[b]], [F[bb]], W)
        cand[b] = {"r": r, "tau": tau, "R1_ce": bool(R1), "R2_collapse": bool(R2), "R3_domination": bool(R3),
                   "rejected": bool(R1 or R2 or R3), "ce_minus_B": o["ce"] - ob["ce"], "ce_minus_base": o["ce"] - out["ce_base"],
                   "I_D_minus_I_B": float(vals[0]), "I_D_minus_I_B_ci": ci(reps[0]),
                   "nmi_minus_B": o["nmi_sem"] - ob["nmi_sem"]}
    out["candidates"] = cand
    surv = {k: v for k, v in cand.items() if not v["rejected"]}
    sel = None
    if surv:
        best = max(v["I_D_minus_I_B"] for v in surv.values())
        tied = [k for k, v in surv.items() if v["I_D_minus_I_B"] >= best - 1e-4]
        tied.sort(key=lambda k: (surv[k]["r"], 0 if surv[k]["tau"] == 1.0 else 1))
        sel = tied[0]
    out["selected"] = sel
    out["selected_cfg"] = None if sel is None else {"r": cand[sel]["r"], "tau": cand[sel]["tau"]}
    if sel is not None:
        bb = f"B_t{cand[sel]['tau']}"
        out["needs_stage1b_alpha"] = bool(out["banks"][sel]["pathology"]["n_dead"] >= 2 or
                                          out["banks"][bb]["pathology"]["n_dead"] >= 2)
    out["stage1_pass"] = sel is not None
    p = ART / f"{a.out or 'stage1'}.json"
    guard.atomic_write_json(p, out, mode=0o400)
    print(json.dumps(out, indent=1, default=str)[:6000])


def load_seeds(paths):
    import numpy as np
    from csr.jsr_paths import ORV1
    feats = dict(np.load(ORV1 / "tables/token_feats.npz"))
    seeds = [Seed(p, feats) for p in paths]
    for S in seeds[1:]:
        if not np.array_equal(S.papers, seeds[0].papers):
            raise guard.GuardError("paper sets differ across seeds")
    return seeds


def core(seeds, hierarchical):
    """Everything shared by stages 2 and 3."""
    import numpy as np
    from csr.jsr_paths import BOOT_SEED, HIER_SEED, N_BOOT, N_HIER
    P = seeds[0].P
    W = boot_weights(P, N_BOOT, BOOT_SEED)
    names = seeds[0].banks
    R = {"n_seeds": len(seeds), "n_papers": P, "per_seed": [], "ce": {}, "func": {}, "nmi": {}, "pathology": {},
         "secondary": {}, "per_group_ce": {}, "swap_matrix": {}, "piu": {}}
    hrep = (lambda v, r, k: hier(v, r, N_HIER, HIER_SEED + k)) if hierarchical else (lambda v, r, k: None)
    for i, S in enumerate(seeds):
        R["per_seed"].append({"eval": str(S.path), "step": S.js["step"], "seen": S.js["seen"], "ce_base": S.ce("base"),
                              **{f"ce_{b}": S.ce(b) for b in names}})
    pairs = [("D", "B"), ("D", "C"), ("C", "B"), ("D", "base"), ("B", "base"), ("C", "base"), ("E", "base"),
             ("E", "B"), ("D", "E")]
    for j, (x, y) in enumerate(pairs):
        v, r = ce_contrast(seeds, x, y, W)
        R["ce"][f"{x}-{y}"] = summarize(v, r, hrep(v, r, j))
    F = {b: [functional(S, b, oracle=(b == "E")) for S in seeds] for b in names}
    Fsh = [functional(S, "C", lab_key="lab_shuf") for S in seeds]
    for b in names:
        v, r = func_single(F[b], W)
        R["func"][f"I_{b}_sem"] = {**summarize(v, r, hrep(v, r, 20 + names.index(b))),
                                   "n_scored": [f["n_scored"] for f in F[b]], "n_pos": [f.get("n_pos") for f in F[b]],
                                   "split_half_r": [f.get("split_half_r") for f in F[b]],
                                   "I_k": [f.get("I_k") for f in F[b]]}
    v, r = func_single(Fsh, W)
    R["func"]["I_C_shuf"] = {**summarize(v, r, hrep(v, r, 30)), "n_scored": [f["n_scored"] for f in Fsh]}
    for j, (x, y) in enumerate((("D", "C"), ("D", "B"), ("E", "B"), ("C", "B"), ("E", "D"))):
        v, r = func_contrast(F[x], F[y], W)
        R["func"][f"I_{x}-I_{y}"] = summarize(v, r, hrep(v, r, 40 + j))
    v, r = func_contrast(F["D"], Fsh, W)
    R["func"]["I_D_sem-I_C_shuf"] = summarize(v, r, hrep(v, r, 50))
    for j, (x, y) in enumerate((("D", "C"), ("D", "B"), ("C", "B"))):
        v, r = nmi_contrast(seeds, x, y, W)
        R["nmi"][f"{x}-{y}"] = summarize(v, r, hrep(v, r, 60 + j))
    for b in names:
        R["nmi"][b] = [float(nmi_counts(S.counts(b).sum(0))) for S in seeds]
        R["pathology"][b] = [pathology(S.load(b)) for S in seeds]
        R["secondary"][b] = [secondary(S, b, oracle=(b == "E")) for S in seeds]
        R["per_group_ce"][b] = np.mean([per_group_ce(S, b) for S in seeds], 0).tolist()
        R["swap_matrix"][b] = np.nanmean([swap_matrix(S, b) for S in seeds], 0).tolist()
    for x, y in (("D", "C"), ("D", "B")):
        R["piu"][f"{x}-{y}"] = [piu(S, x, y) for S in seeds]
    for b in ("B", "C", "D"):
        R["piu"][b] = [piu(S, b) for S in seeds]
    R["per_group_ce_counts"] = np.bincount(seeds[0].z["lab_sem"][seeds[0].z["lab_sem"] >= 0].astype(np.int64),
                                           minlength=16).tolist()
    return R


def stage2(a):
    from csr.jsr_paths import ART
    seeds = load_seeds(a.evals.split(","))
    R = core(seeds, hierarchical=False)
    ce = {b: sum(R["per_seed"][i][f"ce_{b}"] for i in range(len(seeds))) / len(seeds) for b in ("B", "C", "D")}
    P1 = ce["D"] - min(ce["B"], ce["C"]) <= 0.005
    P2 = all(v > 0 for v in R["nmi"]["D-C"]["per_seed"])
    P3 = all(v > 0 for v in R["func"]["I_D-I_C"]["per_seed"])
    P4 = all(p["ok"] for b in ("B", "C", "D") for p in R["pathology"][b])
    R["criteria"] = {"P1_ce_noninferior_0.005": bool(P1), "P1_value": ce["D"] - min(ce["B"], ce["C"]),
                     "P2_nmi_D_gt_C_both": bool(P2), "P3_I_D_gt_I_C_both": bool(P3), "P4_no_pathology": bool(P4)}
    R["promote"] = bool(P1 and P2 and P3 and P4)
    R["time"] = guard.now()
    guard.atomic_write_json(ART / f"{a.out or 'stage2'}.json", R, mode=0o400)
    print(json.dumps({k: R[k] for k in ("criteria", "promote")}, indent=1))


def refs_contrast(seeds, refs, W):
    """D - reference (TOKEN-4 reused / EDL / C6) per seed from per-paper sums (same VAL papers)."""
    import numpy as np
    from csr.jsr_paths import HIER_SEED, N_HIER
    out = {}
    for j, kind in enumerate(("T4", "EDL", "C6")):
        vals, reps, rce = [], [], []
        for i, S in enumerate(seeds):
            z = np.load(refs[kind][i])
            if not np.array_equal(z["papers"], S.papers):
                raise guard.GuardError("reference paper set differs")
            sd = S.ce_sums("D")
            vals.append(float(sd.sum() / S.n.sum() - z["sum"].sum() / z["n"].sum()))
            reps.append((W @ sd) / (W @ S.n) - (W @ z["sum"]) / (W @ z["n"]))
            rce.append(float(z["sum"].sum() / z["n"].sum()))
        v, r = np.asarray(vals), np.stack(reps)
        out[f"D-{kind}"] = summarize(v, r, hier(v, r, N_HIER, HIER_SEED + 70 + j))
        out[f"ce_{kind}"] = rce
    return out


def stage3(a):
    from csr.jsr_paths import ART, BOOT_SEED, N_BOOT
    seeds = load_seeds(a.evals.split(","))
    R = core(seeds, hierarchical=True)
    refs = json.loads(Path(a.refs).read_text())
    W = boot_weights(seeds[0].P, N_BOOT, BOOT_SEED)
    R["refs"] = refs_contrast(seeds, refs, W)
    ce, fn, nm = R["ce"], R["func"], R["nmi"]
    S1 = ce["D-B"]["hier_ci"][1] < 0.005 and ce["D-C"]["hier_ci"][1] < 0.005 and ce["D-base"]["hier_ci"][1] < 0
    S2 = nm["D-C"]["hier_ci"][0] > 0 and nm["D-C"]["n_pos"] >= 3
    S3 = (fn["I_D-I_C"]["hier_ci"][0] > 0 and fn["I_D-I_C"]["n_pos"] >= 3 and
          fn["I_D-I_B"]["hier_ci"][0] > 0 and fn["I_D-I_B"]["n_pos"] >= 3)
    S4 = all(p["ok"] for b in ("B", "C", "D") for p in R["pathology"][b])
    acc_impr = ce["D-B"]["hier_ci"][1] < 0 and ce["D-C"]["hier_ci"][1] < 0
    success = bool(S1 and S2 and S3 and S4)
    O = {"O1": success and acc_impr, "O1prime": success and not acc_impr,
         "O6": S3 and min(ce["D-B"]["hier_ci"][0], ce["D-C"]["hier_ci"][0]) > 0,
         "O2": S2 and not S3,
         "O4": (ce["E-B"]["hier_ci"][1] < 0 and fn["I_E-I_B"]["hier_ci"][0] > 0) and not success,
         "O3": (ce["D-B"]["hier_ci"][1] < 0 and ce["C-B"]["hier_ci"][1] < 0 and
                ce["D-C"]["hier_ci"][0] <= 0 <= ce["D-C"]["hier_ci"][1] and
                not (fn["I_D-I_C"]["hier_ci"][0] > 0 and fn["I_D-I_C"]["n_pos"] >= 3)),
         "O5": not (ce["D-B"]["hier_ci"][1] < 0) and not (ce["E-B"]["hier_ci"][1] < 0)}
    order = ["O1", "O1prime", "O6", "O2", "O4", "O3", "O5"]
    primary = next((k for k in order if O[k]), "MIXED")
    R["criteria"] = {"S1_accuracy": bool(S1), "S2_routing": bool(S2), "S3_function": bool(S3), "S4_integrity": bool(S4),
                     "success": success, "accuracy_improvement": bool(acc_impr)}
    R["outcomes"] = {"applies": {k: bool(v) for k, v in O.items()}, "primary": primary}
    R["time"] = guard.now()
    guard.atomic_write_json(ART / f"{a.out or 'stage3'}.json", R, mode=0o400)
    print(json.dumps({k: R[k] for k in ("criteria", "outcomes")}, indent=1))


def selftest():
    """Synthetic checks: NMI bounds; functional metric recovers a planted own-group advantage and is ~0 without."""
    import numpy as np
    rng = np.random.default_rng(0)
    C = np.eye(16) * 100 + 1
    assert 0.5 < nmi_counts(C) <= 1.0 and abs(nmi_counts(np.ones((16, 16)))) < 1e-12

    class Fake(Seed):
        def __init__(self, planted):
            N, P = 800_000, 400
            pap = rng.integers(0, P, N)
            lab_p = rng.integers(0, 16, P)
            lab = lab_p[pap]
            e = np.where(rng.random(N) < 0.5, lab, rng.integers(0, 16, N))
            d = rng.normal(0, 0.05, (N, 16))
            if planted:
                d[np.arange(N), lab] -= 0.01  # own-group specialist better by 0.01 on its group
            q = rng.dirichlet(np.ones(16), N)
            q[np.arange(N), e] += 1.0
            q /= q.sum(1, keepdims=True)
            self.z = {"papers": np.arange(P).astype(str), "paper": pap, "lab_sem": lab, "lab_shuf": lab, f"e_X": e,
                      "d_X": d, "q_X": q, "sub": np.ones(N, bool), "ce_base": rng.random(N), "pos": rng.integers(0, 1000, N),
                      "doclen": rng.integers(100, 1000, N), "tgt": rng.integers(0, 50000, N)}
            self.papers, self.P = self.z["papers"], P
            self.half = halves(self.papers)
            self.sub_idx = np.arange(N)
            self.base_st = rng.integers(0, 135, N)
            self.n = np.bincount(pap, minlength=P).astype(float)
    f1 = functional(Fake(True), "X")
    f0 = functional(Fake(False), "X")
    ok = abs(f1["I"] - 0.01 * 16 / 15) < 0.003 and abs(f0["I"]) < 0.003 and f1["n_scored"] >= 14
    print(json.dumps({"planted": f1["I"], "null": f0["I"], "n_scored": f1["n_scored"], "ok": bool(ok)}))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["stage1", "stage2", "stage3", "selftest"])
    ap.add_argument("--eval")
    ap.add_argument("--evals")
    ap.add_argument("--run")
    ap.add_argument("--refs")
    ap.add_argument("--out")
    ap.add_argument("--execute", action="store_true")  # accepted (driver CPU jobs); no effect
    a = ap.parse_args()
    guard.check_root()
    if a.what == "selftest":
        sys.exit(0 if selftest() else 1)
    {"stage1": stage1, "stage2": stage2, "stage3": stage3}[a.what](a)


if __name__ == "__main__":
    main()
