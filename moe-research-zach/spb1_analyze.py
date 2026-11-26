#!/usr/bin/env python
"""spb1 analysis (CPU; exact table lookups on the per-token full specialist CE tables of spb1_extract.py).

  spb1_analyze.py stage1 --gen G [--seeds 0,1,2]   bank diagnostics + Stage-1 gate  -> ANALYSIS/stage1_g<G>.json
  spb1_analyze.py stage2 --gen G [--seeds ...]     deployed unrestricted routers + Stage-2 (promotion) gate
                                                   -> ANALYSIS/stage2_g<G>.json
  (--root / --stage-name / --test for CPU tests)

Definitions (prereg section 6):
  d_b[t, e] = CE(base + specialist e of bank b)(t) - CE_base(t); routed CE = CE_base + d_b[t, e(t)].
  Relative usefulness g_e(t) = mean_{j != e} d[t, j] - d[t, e] (specialist e's advantage over the bank's other
  specialists); base-relative usefulness gb_e(t) = -d[t, e] (secondary).
  Exposure-matched own-cluster usefulness of specialist e with training cluster c(e) (pair banks: e // M):
    own set   = SEL tokens whose paper cluster (semantic for SEM, shuffled for SHUF) is c(e) and routed to e;
    other set = SEL tokens of other clusters routed to e by the SAME routing function;
    Stage 1 ("intrinsic"): routing = the training pair subrouter (own: e wins its pair under the token's own cluster;
      other: e beats its pair partner under the shared router logits, i.e. it would be chosen if the token belonged to
      c(e)); strata = target-frequency tercile x base-CE quintile x position bin (4) x doc-length tercile x pair-margin
      tercile (540);
    Stage 2 ("deployed"): routing = the deployed unrestricted router; strata = the 180 base strata x top-1 prob
      quintile x top-1 margin tercile (2,700).
    I_e = sum_s w_s (mean_own_s g - mean_other_s g), w = own-set stratum shares on common support (>= MIN_SET tokens in
    each set); I_bank = mean over scored specialists. Delta I = I_SEM - I_SHUF. CIs: per-paper influence-function
    bootstrap (2,000 multinomial paper resamples; SEL papers are shared by all seeds -> pooled paired bootstrap).
  CE differences: paired per-paper ratio bootstrap over the same SEL papers, pooled over seeds and router seeds."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402
from csr import spb1_paths as P  # noqa: E402

POS_BINS = {"pos_0_31": (0, 32), "pos_32_127": (32, 128), "pos_ge128": (128, 1 << 30)}


def ci(x):
    return [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))]


def boot_w(n_p, seed=P.BOOT_SEED, n=P.N_BOOT):
    return np.random.default_rng(seed).multinomial(n_p, np.full(n_p, 1.0 / n_p), size=n).astype(np.float64)


class Common:
    def __init__(self, cdir, part="SEL"):
        z = np.load(Path(cdir) / f"common_{part}.npz")
        self.z = {k: z[k] for k in z.files}
        self.N = len(self.z["tgt"])
        self.paper = self.z["paper"].astype(np.int64)
        self.P = len(self.z["papers"])
        self.ce_base = self.z["ce_base"].astype(np.float64)
        self.pos = self.z["pos"].astype(np.int64)
        tf = np.load(P.ORV1_TABLES / "token_feats.npz")
        fr = np.log1p(tf["freq"][self.z["tgt"].astype(np.int64)].astype(np.float64))
        qc = lambda x, qs: np.searchsorted(np.quantile(x, qs), x, side="right")
        f3 = qc(fr, [1 / 3, 2 / 3])
        c5 = qc(self.ce_base, [0.2, 0.4, 0.6, 0.8])
        pb = np.searchsorted([32, 128, 512], self.pos, side="right")
        l3 = qc(self.z["doclen"].astype(np.float64), [1 / 3, 2 / 3])
        self.base_st = ((f3 * 5 + c5) * 4 + pb) * 3 + l3            # 180 strata
        self.lab = {"sem": self.z["lab_sem"].astype(np.int64), "shuf": self.z["lab_shuf"].astype(np.int64)}
        self.n_p = np.bincount(self.paper, minlength=self.P).astype(np.float64)
        self.masks = {"all": np.ones(self.N, bool)}
        for k, (lo, hi) in POS_BINS.items():
            self.masks[k] = (self.pos >= lo) & (self.pos < hi)
        if "lab_rsem" in self.z and len(self.z["lab_rsem"]) == self.N:
            r, rp = self.z["lab_rsem"].astype(np.int64), self.z["lab_rsem_prev"].astype(np.int64)
            self.masks["transition_first32"] = (rp >= 0) & (r != rp) & (self.pos % 256 < 32)
            self.masks["stable_context"] = (rp >= 0) & (r == rp)
        self.same_doc = self.z["doc"][1:] == self.z["doc"][:-1]

    def psum(self, v, mask=None):
        if mask is None:
            return np.bincount(self.paper, weights=v, minlength=self.P), self.n_p
        return (np.bincount(self.paper[mask], weights=v[mask], minlength=self.P),
                np.bincount(self.paper[mask], minlength=self.P).astype(np.float64))


def matched(g, own, oth, st, nst, paper, P_):
    """Exposure/strata-matched own-minus-other mean of g. Returns (value, IF per paper, n_own, n_oth) or None."""
    if own.sum() < P.MIN_SET or oth.sum() < P.MIN_SET:
        return None
    so, sn = st[own], st[oth]
    no = np.bincount(so, minlength=nst).astype(np.float64)
    nn = np.bincount(sn, minlength=nst).astype(np.float64)
    common = (no > 0) & (nn > 0)
    ko, kn = common[so], common[sn]
    if ko.sum() < P.MIN_SET or kn.sum() < P.MIN_SET:
        return None
    go, gn = g[own][ko], g[oth][kn]
    so, sn = so[ko], sn[kn]
    po, pn = paper[own][ko], paper[oth][kn]
    no = np.bincount(so, minlength=nst).astype(np.float64)
    nn = np.bincount(sn, minlength=nst).astype(np.float64)
    w = no / no.sum()
    mo = np.bincount(so, weights=go, minlength=nst) / np.maximum(no, 1)
    mn = np.bincount(sn, weights=gn, minlength=nst) / np.maximum(nn, 1)
    val = float((w * (mo - mn)).sum())
    IF = (np.bincount(po, weights=w[so] * (go - mo[so]) / no[so], minlength=P_)
          - np.bincount(pn, weights=w[sn] * (gn - mn[sn]) / nn[sn], minlength=P_))
    return val, IF, int(ko.sum()), int(kn.sum())


def tercile(x, m):
    return np.searchsorted(np.quantile(x[m], [1 / 3, 2 / 3]), x, side="right")


def quintile(x, m):
    return np.searchsorted(np.quantile(x[m], [0.2, 0.4, 0.6, 0.8]), x, side="right")


def bank_functional(C, d, M, lab, route, kind):
    """kind 'intrinsic': route = dict(e_tr, pdiff); kind 'deployed': route = dict(e, p1, marg). Returns per-specialist
    results and bank mean (value + IF)."""
    E = d.shape[1]
    rowsum = d.sum(1, dtype=np.float64)
    res, IFs, vals, IFb, valsb = [], [], [], [], []
    for e in range(E):
        c = e // M
        de = d[:, e].astype(np.float64)
        g = (rowsum - de) / (E - 1) - de
        if kind == "intrinsic":
            own = (lab == c) & (route["e_tr"] == e)
            if M == 2:
                pdc = route["pdiff"][:, c].astype(np.float64)
                win = pdc > 0 if e % 2 == 0 else pdc < 0
                marg = np.abs(pdc)
            else:
                lg = route["lg"][:, c * M:(c + 1) * M].astype(np.float64)
                win = lg.argmax(1) == (e - c * M)
                srt = np.sort(lg, 1)
                marg = srt[:, -1] - srt[:, -2]
            oth = (lab != c) & (lab >= 0) & win
            if own.sum() < P.MIN_SET or oth.sum() < P.MIN_SET:
                res.append({"e": e, "I": None, "n_own": int(own.sum()), "n_oth": int(oth.sum())})
                continue
            m3 = tercile(marg, own | oth)
            st, nst = C.base_st * 3 + m3, 180 * 3
        else:
            ed = route["e"].astype(np.int64)
            own = (lab == c) & (ed == e)
            oth = (lab != c) & (lab >= 0) & (ed == e)
            if own.sum() < P.MIN_SET or oth.sum() < P.MIN_SET:
                res.append({"e": e, "I": None, "n_own": int(own.sum()), "n_oth": int(oth.sum())})
                continue
            q5 = quintile(route["p1"].astype(np.float64), own | oth)
            m3 = tercile(route["marg"].astype(np.float64), own | oth)
            st, nst = (C.base_st * 5 + q5) * 3 + m3, 180 * 15
        r = matched(g, own, oth, st, nst, C.paper, C.P)
        rb = matched(-de, own, oth, st, nst, C.paper, C.P)
        if r is None:
            res.append({"e": e, "I": None, "n_own": int(own.sum()), "n_oth": int(oth.sum())})
            continue
        res.append({"e": e, "I": r[0], "I_base_rel": rb[0] if rb else None, "n_own": r[2], "n_oth": r[3], "_IF": r[1]})
        vals.append(r[0])
        IFs.append(r[1])
        if rb:
            valsb.append(rb[0])
            IFb.append(rb[1])
    if not vals:
        return {"specialists": res, "I": None, "n_scored": 0}
    return {"specialists": res, "I": float(np.mean(vals)), "IF": np.mean(IFs, 0), "n_scored": len(vals),
            "I_base_rel": float(np.mean(valsb)) if valsb else None, "IF_base_rel": np.mean(IFb, 0) if IFb else None}


def summarize_spec(f, W):
    out, npos, nsig = [], 0, 0
    for s in f["specialists"]:
        if s.get("I") is None:
            out.append({k: v for k, v in s.items() if k != "_IF"})
            continue
        reps = s["I"] + W @ s["_IF"] - s["_IF"].sum()
        c = ci(reps)
        npos += s["I"] > 0
        nsig += c[0] > 0
        out.append({**{k: v for k, v in s.items() if k != "_IF"}, "ci": c})
    return out, int(npos), int(nsig)


def ce_diff_pooled(C, pairs, W, mask=None):
    """pairs: list of (ce_a, ce_b) per-token arrays (same tokens). Pooled paired per-paper ratio estimator."""
    sa = np.zeros(C.P)
    na = np.zeros(C.P)
    per = []
    for a, b in pairs:
        s, n = C.psum(a - b, mask)
        sa += s
        na += n
        per.append(float(s.sum() / max(1, n.sum())))
    v = float(sa.sum() / max(1, na.sum()))
    reps = (W @ sa) / np.maximum(W @ na, 1)
    return {"mean": v, "ci": ci(reps), "per_combo": per, "n_tokens": int(na.sum() / max(1, len(pairs)))}


def loads(e, E):
    ld = np.bincount(e.astype(np.int64), minlength=E) / len(e)
    nz = ld[ld > 0]
    return {"load_max": float(ld.max()), "eff_exp_entropy": float(np.exp(-(nz * np.log(nz)).sum())),
            "eff_inv_simpson": float(1 / (ld ** 2).sum()), "n_dead": int((ld < 1 / (10 * E)).sum()),
            "n_used": int((ld > 0).sum())}


def nmi(a, b):
    a, b = np.asarray(a, np.int64), np.asarray(b, np.int64)
    ok = (a >= 0) & (b >= 0)
    a, b = a[ok], b[ok]
    c = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(c, (a, b), 1)
    p = c / c.sum()
    pa, pb = p.sum(1), p.sum(0)
    nz = p > 0
    mi = (p[nz] * np.log(p[nz] / np.outer(pa, pb)[nz])).sum()
    ha, hb = -(pa[pa > 0] * np.log(pa[pa > 0])).sum(), -(pb[pb > 0] * np.log(pb[pb > 0])).sum()
    return float(mi / max(1e-12, np.sqrt(ha * hb)))


def dirs(gen, seed, root):
    G = P.gen_cfg(gen) if not root else json.loads((Path(root) / "gen_cfg.json").read_text())
    base = Path(root) if root else P.ART
    bg = P.bank_gen(gen) if not root else gen
    tdir = base / "extract" / f"g{bg}_s{seed}_S1" if not root else base / f"g{gen}_s{seed}"
    cdir = base / "extract" / f"common_s{seed}" if not root else base / f"common_s{seed}"
    return G, tdir, cdir


def load_bank(tdir, cdir, name, part="SEL"):
    if name == "COARSE16":
        z = np.load(cdir / f"coarse_{part}.npz")
    else:
        z = np.load(tdir / f"bank_{name}_{part}.npz")
    return {k: z[k] for k in z.files}


# ------------------------------------------------------------------------------------------------ stage 1
def stage1(a):
    seeds = [int(s) for s in a.seeds.split(",")]
    out = {"time": guard.now(), "gen": a.gen, "seeds": seeds, "per_seed": {}}
    W = None
    pooled_if = {}
    ce_gap = []
    for s in seeds:
        G, tdir, cdir = dirs(a.gen, s, a.root)
        M = int(G["M"])
        sem_n, shuf_n, gen_n = P.bank_names(M)
        C = Common(cdir)
        if W is None:
            W = boot_w(C.P)
        R = {"n_tokens": C.N, "ce_base": float(C.ce_base.mean()), "banks": {}}
        fun = {}
        for name in (sem_n, shuf_n, gen_n, "COARSE16"):
            B = load_bank(tdir, cdir, name)
            d = B["d"].astype(np.float32)
            E = d.shape[1]
            e_tr = B["e_tr"].astype(np.int64)
            dtr = d[np.arange(C.N), e_tr].astype(np.float64)
            r = {"E": E, "ce_train_path": float(C.ce_base.mean() + dtr.mean()),
                 "ce_token_oracle": float(C.ce_base.mean() + d.min(1).astype(np.float64).mean()),
                 "train_path_loads": loads(e_tr, E), "nmi_train_route_vs_sem": nmi(e_tr, C.lab["sem"])}
            for mk, m in C.masks.items():
                r.setdefault("ce_train_path_by_mask", {})[mk] = float(C.ce_base[m].mean() + dtr[m].mean())
            if name in (sem_n, shuf_n):
                lab = C.lab["sem" if name == sem_n else "shuf"]
                grp = d.reshape(C.N, E // M, M)[np.arange(C.N), lab]
                r["ce_own_cluster_pair_hindsight"] = float(C.ce_base.mean() + grp.min(1).astype(np.float64).mean())
                # replacement losses on training-path tokens
                part = e_tr ^ 1 if M == 2 else None
                rl = []
                rowsum = d.sum(1, dtype=np.float64)
                for e in range(E):
                    m = e_tr == e
                    if m.sum() == 0:
                        rl.append({"e": e, "n": 0})
                        continue
                    de = d[m, e].astype(np.float64)
                    x = {"e": e, "n": int(m.sum()), "remove_to_base": float(-de.mean()),
                         "replace_by_mean_others": float(((rowsum[m] - de) / (E - 1) - de).mean())}
                    if part is not None:
                        x["replace_by_partner"] = float((d[m, e ^ 1].astype(np.float64) - de).mean())
                    rl.append(x)
                r["replacement_loss"] = rl
                for key in ("remove_to_base", "replace_by_mean_others", "replace_by_partner"):
                    v = [x[key] for x in rl if key in x]
                    if v:
                        r[f"replacement_{key}_summary"] = {"mean": float(np.mean(v)), "min": float(np.min(v)),
                                                           "n_pos": int(sum(x > 0 for x in v)), "n": len(v)}
                # within-pair balance
                cnt = np.bincount(e_tr, minlength=E).reshape(E // M, M)
                r["within_cluster_max_member_share"] = (cnt.max(1) / np.maximum(cnt.sum(1), 1)).round(4).tolist()
                route = {"e_tr": e_tr, "pdiff": B.get("pdiff"), "lg": B.get("lg")}
                f = bank_functional(C, d, M, lab, route, "intrinsic")
                spec, npos, nsig = summarize_spec(f, W)
                r["functional_intrinsic"] = {"I": f["I"], "I_base_rel": f.get("I_base_rel"), "n_scored": f["n_scored"],
                                             "n_pos": npos, "n_sig": nsig, "specialists": spec}
                if f["I"] is not None:
                    reps = f["I"] + W @ f["IF"] - f["IF"].sum()
                    r["functional_intrinsic"]["ci"] = ci(reps)
                fun[name] = f
            if name == "COARSE16":
                # orv1-style own-cluster usefulness (no router: own = cluster tokens, other = others; base strata)
                lab = C.lab["sem"]
                rowsum = d.sum(1, dtype=np.float64)
                vals = []
                for e in range(E):
                    de = d[:, e].astype(np.float64)
                    rr = matched((rowsum - de) / (E - 1) - de, lab == e, (lab != e) & (lab >= 0), C.base_st, 180,
                                 C.paper, C.P)
                    if rr:
                        vals.append(rr[0])
                r["functional_own_cluster_base_strata"] = {"I": float(np.mean(vals)) if vals else None,
                                                           "n_scored": len(vals),
                                                           "n_pos": int(sum(v > 0 for v in vals))}
                r["note"] = ("COARSE-16 = orv1 condition A (oracle semantic label; 16 specialists; half the bank "
                             "parameters). Its functional value has no router exposure to match (descriptive only).")
            R["banks"][name] = r
            del d
        # Stage-1 functional contrast
        fs, fh = fun[sem_n], fun[shuf_n]
        if fs["I"] is not None and fh["I"] is not None:
            v = fs["I"] - fh["I"]
            D = fs["IF"] - fh["IF"]
            reps = v + W @ D - D.sum()
            R["dI_intrinsic"] = {"v": float(v), "ci": ci(reps), "pass": bool(ci(reps)[0] > 0)}
            pooled_if.setdefault("v", []).append(v)
            pooled_if.setdefault("D", []).append(D)
            if fs.get("I_base_rel") is not None and fh.get("I_base_rel") is not None:
                vb = fs["I_base_rel"] - fh["I_base_rel"]
                Db = fs["IF_base_rel"] - fh["IF_base_rel"]
                R["dI_intrinsic_base_rel"] = {"v": float(vb), "ci": ci(vb + W @ Db - Db.sum())}
        else:
            R["dI_intrinsic"] = {"v": None, "pass": False}
        gap = R["banks"][sem_n]["ce_train_path"] - R["banks"][gen_n]["ce_train_path"]
        R["sem_oracle_minus_gen"] = gap
        ce_gap.append(gap)
        out["per_seed"][str(s)] = R
    if pooled_if:
        v = float(np.mean(pooled_if["v"]))
        D = np.mean(pooled_if["D"], 0)
        out["dI_intrinsic_pooled"] = {"v": v, "ci": ci(v + W @ D - D.sum())}
    n_pass = sum(out["per_seed"][str(s)]["dI_intrinsic"]["pass"] for s in seeds)
    out["gate"] = {"G1a_functional_seeds_pass": int(n_pass), "G1a_pass": bool(n_pass >= 2),
                   "G1b_sem_oracle_minus_gen_mean": float(np.mean(ce_gap)),
                   "G1b_pass": bool(np.mean(ce_gap) <= P.GATE_ORACLE_MARGIN)}
    out["gate"]["advance"] = bool(out["gate"]["G1a_pass"] and out["gate"]["G1b_pass"])
    out["promoted_seeds"] = [s for s in seeds if out["per_seed"][str(s)]["dI_intrinsic"]["pass"]] \
        if out["gate"]["advance"] else []
    return out


# ------------------------------------------------------------------------------------------------ stage 2
def stage2(a):
    seeds = [int(s) for s in a.seeds.split(",")]
    stage = a.stage_name or f"S2_g{a.gen}"
    rdir = (Path(a.root) / "routers" if a.root else P.ROUTERS) / stage
    out = {"time": guard.now(), "gen": a.gen, "seeds": seeds, "stage": stage, "per_seed": {}}
    W = None
    comb = {}
    fun_pool = {}
    for s in seeds:
        G, tdir, cdir = dirs(a.gen, s, a.root)
        M = int(G["M"])
        sem_n, shuf_n, gen_n = P.bank_names(M)
        C = Common(cdir)
        if W is None:
            W = boot_w(C.P)
        R = {"routers": {}, "oracle": {}}
        deps = sorted(rdir.glob(f"*_s{s}_r*_SEL.json"))
        tabs = {}
        for jf in deps:
            js = json.loads(jf.read_text())
            tag, rs, bank = js["cfg"]["tag"], js["cfg"]["router_seed"], js["cfg"]["bank"]
            if bank not in tabs:
                tabs[bank] = load_bank(tdir, cdir, bank)["d"].astype(np.float32)
            dz = np.load(jf.with_suffix(".npz"))
            e = dz["e"].astype(np.int64)
            d = tabs[bank]
            E = d.shape[1]
            ce = C.ce_base + d[np.arange(C.N), e].astype(np.float64)
            r = {"bank": bank, "router_seed": rs, "post": bool(js["cfg"].get("post")), "ce": float(ce.mean()),
                 **loads(e, E), "switch_rate": float((e[1:] != e[:-1])[C.same_doc].mean()),
                 "nmi_vs_sem": nmi(e, C.lab["sem"]), "ce_minus_base_by_mask": {}}
            for mk, m in C.masks.items():
                r["ce_minus_base_by_mask"][mk] = float((ce - C.ce_base)[m].mean())
            key = f"{tag}_r{rs}"
            comb.setdefault(tag, {})[(s, rs)] = ce
            if bank in (sem_n, shuf_n):
                lab = C.lab["sem" if bank == sem_n else "shuf"]
                f = bank_functional(C, d, M, lab, {"e": e, "p1": dz["p1"], "marg": dz["marg"]}, "deployed")
                spec, npos, nsig = summarize_spec(f, W)
                r["functional_deployed"] = {"I": f["I"], "I_base_rel": f.get("I_base_rel"), "n_scored": f["n_scored"],
                                            "n_pos": npos, "n_sig": nsig, "specialists": spec}
                if f["I"] is not None:
                    fun_pool.setdefault((tag, s), []).append(f)
            R["routers"][key] = r
        for bank, d in tabs.items():
            R["oracle"][bank] = {"ce_token_oracle": float(C.ce_base.mean() + d.min(1).astype(np.float64).mean())}
        sm = json.loads((tdir / "summary_SEL.json").read_text())
        R["oracle"]["train_path"] = sm["ce_train_path"]
        R["ce_base"] = float(C.ce_base.mean())
        out["per_seed"][str(s)] = R
        del tabs
    # pooled comparisons (same SEL tokens and papers across seeds)
    Cref = C
    tagS, tagH, tagG = "R_SEM", "R_SHUF", "R_GEN"

    def pairs(ta, tb):
        ks = sorted(set(comb.get(ta, {})) & set(comb.get(tb, {})))
        return [(comb[ta][k], comb[tb][k]) for k in ks], ks

    out["comparisons"] = {}
    for name, (ta, tb) in {"SEM_minus_GEN": (tagS, tagG), "SEM_minus_SHUF": (tagS, tagH),
                           "SEM_minus_COARSE16": (tagS, "R_COARSE16"), "GEN_minus_COARSE16": (tagG, "R_COARSE16"),
                           "SEMpost_minus_SEM": ("R_SEM_POST", tagS), "GENpost_minus_GEN": ("R_GEN_POST", tagG),
                           "SHUFpost_minus_SHUF": ("R_SHUF_POST", tagH), "SEMpost_minus_GENpost": ("R_SEM_POST", "R_GEN_POST")}.items():
        pr, ks = pairs(ta, tb)
        if not pr:
            continue
        out["comparisons"][name] = {"combos": [list(k) for k in ks]}
        for mk, m in Cref.masks.items():
            out["comparisons"][name][mk] = ce_diff_pooled(Cref, pr, W, None if mk == "all" else m)
        per_seed = {}
        for s in seeds:
            pp = [(comb[ta][k], comb[tb][k]) for k in ks if k[0] == s]
            if pp:
                per_seed[str(s)] = ce_diff_pooled(Cref, pp, W)["mean"]
        out["comparisons"][name]["per_seed"] = per_seed
    # pooled deployed functional Delta I (average over router seeds within seed, then over seeds)
    dI = {}
    for s in seeds:
        fs, fh = fun_pool.get((tagS, s)), fun_pool.get((tagH, s))
        if fs and fh:
            v = np.mean([f["I"] for f in fs]) - np.mean([f["I"] for f in fh])
            D = np.mean([f["IF"] for f in fs], 0) - np.mean([f["IF"] for f in fh], 0)
            dI[s] = (v, D)
            out["per_seed"][str(s)]["dI_deployed"] = {"v": float(v), "ci": ci(v + W @ D - D.sum())}
    if dI:
        v = float(np.mean([x[0] for x in dI.values()]))
        D = np.mean([x[1] for x in dI.values()], 0)
        out["dI_deployed_pooled"] = {"v": v, "ci": ci(v + W @ D - D.sum()),
                                     "per_seed": {str(s): float(x[0]) for s, x in dI.items()}}
    # gates P1-P4
    g = {}
    c = out["comparisons"].get("SEM_minus_GEN", {}).get("all")
    if c:
        g["P1_mean"] = c["mean"]
        g["P1_ci"] = c["ci"]
        g["P1_pass"] = bool(c["mean"] <= P.GATE_CE and c["ci"][1] < 0)
        ps = out["comparisons"]["SEM_minus_GEN"]["per_seed"]
        loso = {s: float(np.mean([v for k, v in ps.items() if k != s])) for s in ps} if len(ps) > 1 else {}
        g["P3_per_seed"] = ps
        g["P3_loso"] = loso
        g["P3_pass"] = bool(all(v < 0 for v in ps.values()) and (all(v <= P.GATE_LOSO for v in loso.values())
                                                                 if loso else False))
    dp = out.get("dI_deployed_pooled")
    g["P2_pass"] = bool(dp is not None and dp["ci"][0] > 0 and all(v > 0 for v in dp["per_seed"].values())
                        and len(dp["per_seed"]) == len(seeds))
    semr = [r for s in seeds for k, r in out["per_seed"][str(s)]["routers"].items() if k.startswith(tagS + "_r")]
    g["P4_eff_min"] = float(min(r["eff_exp_entropy"] for r in semr)) if semr else None
    g["P4_load_max"] = float(max(r["load_max"] for r in semr)) if semr else None
    g["P4_pass"] = bool(semr and g["P4_eff_min"] >= P.GATE_EFF_MIN and g["P4_load_max"] <= P.GATE_LOAD_MAX)
    g["promote"] = bool(g.get("P1_pass") and g["P2_pass"] and g.get("P3_pass") and g["P4_pass"])
    out["gate"] = g
    return out


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return None
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stage1", "stage2"])
    ap.add_argument("--gen", type=int, default=0)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--root", default="")
    ap.add_argument("--stage-name", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    res = stage1(a) if a.cmd == "stage1" else stage2(a)
    res = jsonable(res)
    outp = Path(a.out) if a.out else P.ANALYSIS / f"{a.cmd}_g{a.gen}{('_' + a.stage_name) if a.stage_name else ''}.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    guard.atomic_write_json(outp, res, overwrite=True)
    print(json.dumps(res.get("gate"), indent=1))


if __name__ == "__main__":
    main()
