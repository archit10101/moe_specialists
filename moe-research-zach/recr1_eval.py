#!/usr/bin/env python
"""recr1 evaluation pipeline (CPU): candidate heads, kappa calibration, exact policy simulation of every condition on
one partition, metrics, diagnostics and preregistered gates.

  recr1_eval.py heads --gen G                 train missing heads (UTRAIN) for generation G's head kind
  recr1_eval.py eval --gen G --part CSEARCH   evaluate generation G on CSEARCH -> analysis/gen<G>_CSEARCH.json
  recr1_eval.py eval --gen G --part CSELECT --frozen   one-time confirmation (requires FROZEN.json)"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402
from csr import recr1_lib as L  # noqa: E402
from csr import recr1_log as LG  # noqa: E402
from csr import recr1_paths as P  # noqa: E402

GATE_GRID_C = (0.3, 0.5, 0.7, 0.9)
MASKS = ("all", "pos_0_7", "pos_0_31", "pos_32_127", "pos_ge128", "transition_first8", "transition_first32",
         "stable_context")


# ------------------------------------------------------------------------------------------------------ heads
def head_gen(gen):
    c = P.gen_cfg(gen)
    for g in range(gen + 1):
        if P.gen_cfg(g)["head"] == c["head"]:
            return g
    return gen


def head_path(gen, bank, s, r, fold=None):
    d = P.HEADS / f"h{head_gen(gen)}"
    d.mkdir(parents=True, exist_ok=True)
    return d / (f"{bank}_s{s}_r{r}" + (f"_f{fold}" if fold is not None else "") + ".pt")


def ensure_heads(gen):
    kind = P.gen_cfg(gen)["head"]
    for s in P.BANK_SEEDS:
        part = None
        for bank in ("SEM2", "SHUF2"):
            for r in P.ROUTER_SEEDS:
                paths = [head_path(gen, bank, s, r)] + [head_path(gen, bank, s, r, f) for f in (0, 1)]
                done = [p.with_suffix(".done.json").exists() for p in paths]
                if all(done):
                    continue
                part = part or L.Part(s, "UTRAIN")
                X, Y = L.features(part, bank, r), L.targets(part, bank, r)
                fold = part.paper % 2
                for p, f in zip(paths, (None, 0, 1)):
                    if p.with_suffix(".done.json").exists():
                        continue
                    t0 = time.time()
                    m = np.ones(part.N, bool) if f is None else fold != f
                    toks = P.HEAD_TOKENS if f is None else P.HEAD_TOKENS // 2
                    h = L.train_head(X[m], Y[m], kind, 1000 * s + 10 * r + (0 if f is None else 1 + f), toks, p,
                                     log=(lambda d: None) if P.SMOKE else
                                     (lambda d, p=p: LG.run_log(kind="head_ckpt", head=str(p.name), **d)),
                                     stop_check=lambda: P.STOP_MARKER.exists())
                    info = {"time": guard.now(), "bank": bank, "seed": s, "router_seed": r, "fold": f, "kind": kind,
                            "tokens": toks, "n_params": L.n_params(h), "macs": h.macs, "seconds": time.time() - t0,
                            "sha256": guard.sha256_file(p), "train_tokens_unique": int(m.sum())}
                    guard.atomic_write_json(p.with_suffix(".done.json"), info, overwrite=True)
                    if not P.SMOKE:
                        LG.run_log(event="head_done", **info)
                del X, Y


def predictions(gen, part, bank, s, r, X=None):
    X = L.features(part, bank, r) if X is None else X
    h = L.load_head(head_path(gen, bank, s, r), X.shape[1])
    return L.predict(h, X), h


# ------------------------------------------------------------------------------------------------ calibration
def utrain_stats(gen, cfg):
    """kappa (exact and strided) and per-option evidence sd, per (seed, router seed, bank), from UTRAIN only."""
    key = json.dumps({k: cfg[k] for k in ("gamma", "tau", "kappa_rate", "norm", "stride_cm")}, sort_keys=True)
    cp = (P.TMP / "smoke" if P.SMOKE else P.ART) / "kappa_cache.json"
    cp.parent.mkdir(parents=True, exist_ok=True)
    cache = json.loads(cp.read_text()) if cp.exists() else {}
    if key in cache:
        return cache[key]
    out = {}
    for s in P.BANK_SEEDS:
        part = L.Part(s, "UTRAIN")
        for r in P.ROUTER_SEEDS:
            for bank in ("SEM2", "SHUF2"):
                dS, eS, dG = part.path_d(bank, r)
                Dall = np.concatenate([part.tables[bank], dS[:, None], dG[:, None]], 1)
                sd = np.r_[Dall.std(0), 1.0]
                ks = {}
                for stride in (1, int(cfg["stride_cm"])):
                    ks[str(stride)] = L.calibrate_kappa(part, dS, cfg["gamma"], cfg["tau"], cfg["kappa_rate"], stride,
                                                        norm=cfg["norm"], norm_sd=sd[L.OPT_S])
                out[f"{s}_{r}_{bank}"] = {"kappa": ks, "sd": sd.tolist()}
    cache[key] = out
    guard.atomic_write_json(cp, cache, overwrite=True)
    return out


# --------------------------------------------------------------------------------------------------- helpers
class PartCommon:
    """Minimal spb1_analyze.Common equivalent restricted to a partition (strata for exposure matching)."""

    def __init__(self, part):
        tf = np.load(guard.ROOT / "artifacts/oracle-residual-v1/tables/token_feats.npz")
        fr = np.log1p(tf["freq"][part.tgt].astype(np.float64))
        qc = lambda x, qs: np.searchsorted(np.quantile(x, qs), x, side="right")
        f3 = qc(fr, [1 / 3, 2 / 3])
        c5 = qc(part.ce_base, [0.2, 0.4, 0.6, 0.8])
        pb = np.searchsorted([32, 128, 512], part.pos, side="right")
        l3 = qc(part.doclen.astype(np.float64), [1 / 3, 2 / 3])
        self.base_st = ((f3 * 5 + c5) * 4 + pb) * 3 + l3
        self.paper, self.P = part.paper, part.P


def bank_exec(sim_opt, eS):
    """Executed BANK specialist per token (-1 if the generic path or no residual executed)."""
    o = sim_opt.astype(np.int64)
    return np.where(o == L.OPT_S, eS, np.where((o >= 0) & (o < 32), o, -1))


def change_metrics(part, sim, dB):
    """Alarms vs descriptive semantic changes (region-prior argmax change at a 256-token region start) and vs
    hindsight functional changes (best bank specialist of the region differs from the previous region's)."""
    reg = part.doc * 4 + part.pos // 256
    nreg = int(reg.max()) + 1
    off = part.pos % 256
    first = np.full(nreg, 999)
    a = sim["alarm"]
    np.minimum.at(first, reg[a], off[a])
    rstart = off == 0
    rid = reg[rstart]
    sem_chg = np.zeros(nreg, bool)
    sem_chg[rid] = (part.rp[rstart] >= 0) & (part.r[rstart] != part.rp[rstart])
    has_prev = np.zeros(nreg, bool)
    has_prev[rid] = part.pos[rstart] >= 256
    cnt = np.bincount(reg, minlength=nreg)
    sums = np.zeros((nreg, 32))
    np.add.at(sums, reg, dB.astype(np.float64))
    best = (sums / np.maximum(cnt, 1)[:, None]).argmin(1)
    fun_chg = np.zeros(nreg, bool)
    fun_chg[1:] = has_prev[1:] & (best[1:] != best[:-1]) & (cnt[1:] >= 64)
    alarms_in_window = a & (off < 64) & sem_chg[reg]
    rate = a.sum() / part.N
    out = {"alarms_per_1k": float(1000 * rate),
           "chance_recall_32": float(1 - (1 - rate) ** 32), "chance_recall_64": float(1 - (1 - rate) ** 64),
           "precision_sem_64": float(alarms_in_window.sum() / max(1, a.sum())),
           "false_alarms_per_1k": float(1000 * (a & ~((off < 64) & sem_chg[reg])).sum() / part.N)}
    for name, ch in (("semantic", sem_chg), ("functional", fun_chg), ("semantic_only", sem_chg & ~fun_chg),
                     ("functional_only", fun_chg & ~sem_chg), ("both", sem_chg & fun_chg)):
        n = int(ch.sum())
        f = first[ch]
        det = f < 256
        out[name] = {"n_changes": n, "recall_32": float((f < 32).mean()) if n else None,
                     "recall_64": float((f < 64).mean()) if n else None,
                     "recall_256": float(det.mean()) if n else None,
                     "median_delay_detected": float(np.median(f[det])) if det.any() else None}
    # false-alarm baseline: alarm rate in regions with neither change
    q = has_prev & ~sem_chg & ~fun_chg
    out["no_change_region_alarm_rate_per_1k"] = float(1000 * (a & q[reg]).sum() / max(1, q[reg].sum()))
    return out, sem_chg[reg] & (off < 32), fun_chg[reg] & (off < 32), (sem_chg & ~fun_chg)[reg] & (off < 32), \
        (fun_chg & ~sem_chg)[reg] & (off < 32)


def commit_metrics(part, commits, dB, dS, dG, fb_d, hold=64):
    """Candidate recall and evidence reliability at commits (CSEARCH/CSELECT, evaluation only)."""
    if not commits:
        return {}
    DA = part.padded(np.concatenate([dB, dS[:, None]], 1).astype(np.float32), fill=np.nan)
    FB = part.padded(fb_d.astype(np.float32), fill=np.nan)
    LAB = part.padded(part.lab_sem, fill=-1)
    rec, own_rec, rel_x, rel_y, gain, n_eval = [], [], [], [], [], 0
    for (i, t, new, top, sc) in commits:
        w = DA[i, t + 1:t + 1 + hold]
        if w.shape[0] < 16 or np.isnan(w[:, 0]).all():
            continue
        n_eval += 1
        m = np.nanmean(w, 0)
        fbm = np.nanmean(FB[i, t + 1:t + 1 + hold])
        hb = int(np.nanargmin(m))
        rec.append(hb in top)
        lab = LAB[i, t]
        own_rec.append(lab >= 0 and (2 * lab in top or 2 * lab + 1 in top))
        if None not in sc and len(top) == 2:
            rel_x.append(sc[0] - sc[1])
            rel_y.append((fbm - m[top[0]]) - (fbm - m[top[1]]))
        gain.append(fbm - (m[new] if new < 33 else fbm))
    rx, ry = np.array(rel_x), np.array(rel_y)
    return {"n_commits": n_eval, "hindsight_best_in_top2": float(np.mean(rec)), "own_pair_in_top2": float(np.mean(own_rec)),
            "evidence_reliability_corr": float(np.corrcoef(rx, ry)[0, 1]) if len(rx) > 10 else None,
            "evidence_sign_agreement": float(np.mean(np.sign(rx) == np.sign(ry))) if len(rx) else None,
            "commit_hold_gain_vs_fallback": float(np.mean(gain)),
            "frac_commit_fallback": float(np.mean([c[2] >= 33 for c in commits]))}


def calib_metrics(U, dB, dS, dG):
    adv = U[:, :33] - U[:, 33:34]
    top = adv.argmax(1)
    pred = adv[np.arange(len(top)), top]
    Dall = np.concatenate([dB, dS[:, None]], 1)
    real = dG - Dall[np.arange(len(top)), top]
    q = np.quantile(pred, np.linspace(0, 1, 11))
    b = np.clip(np.searchsorted(q, pred, side="right") - 1, 0, 9)
    rel = [{"pred": float(pred[b == k].mean()), "real": float(real[b == k].mean()), "n": int((b == k).sum())}
           for k in range(10)]
    slope = float(np.cov(pred, real)[0, 1] / max(1e-18, pred.var()))
    return {"corr": float(np.corrcoef(pred, real)[0, 1]), "slope": slope, "mean_pred": float(pred.mean()),
            "mean_real": float(real.mean()), "frac_pred_positive": float((pred > 0).mean()),
            "frac_real_positive_given_pred_positive": float((real[pred > 0] > 0).mean()) if (pred > 0).any() else None,
            "reliability": rel}


# ---------------------------------------------------------------------------------------------------- evaluate
def evaluate(gen, part_name, frozen=None):
    cfg = P.gen_cfg(gen) if frozen is None else frozen["cfg"]
    stats = utrain_stats(gen, cfg)
    t_start = time.time()
    comb, extra, sims_meta = {}, {}, {}
    W, part_ref = None, None
    papers_ref = None
    theta_C = None if frozen is None else frozen.get("theta_C")
    fun = {}
    for s in P.BANK_SEEDS:
        part = L.Part(s, part_name)
        if papers_ref is None:
            papers_ref = part.paper_raw.copy()
            part_ref = part
            W = L.boot_w(part.P)
        assert np.array_equal(part.paper_raw, papers_ref)
        masks = part.masks()
        C = PartCommon(part)
        for r in P.ROUTER_SEEDS:
            key = (s, r)
            for bank in ("SEM2", "SHUF2"):
                st = stats[f"{s}_{r}_{bank}"]
                sd = np.array(st["sd"])
                k1, kc = st["kappa"]["1"][0], st["kappa"][str(int(cfg["stride_cm"]))][0]
                dB = part.tables[bank]
                dS, eS, dG = part.path_d(bank, r)
                X = L.features(part, bank, r)
                U, head = predictions(gen, part, bank, s, r, X)
                del X
                sfx = "" if bank == "SEM2" else "_SHUF"
                t0 = time.time()
                simE = L.simulate(part, dB, dS, dG, U, cfg, k1, mode="E", stride=1, norm_sd=sd, collect=True)
                simC = L.simulate(part, dB, dS, dG, U, cfg, kc, mode="E", stride=int(cfg["stride_cm"]), norm_sd=sd)
                simD = L.simulate(part, dB, dS, dG, U, cfg, k1, mode="D", stride=1, norm_sd=sd)
                sim_sec = time.time() - t0
                comb.setdefault("E" + sfx, {})[key] = simE["d"]
                comb.setdefault("Ecm" + sfx, {})[key] = simC["d"]
                comb.setdefault("D" + sfx, {})[key] = simD["d"]
                comb.setdefault("S" + sfx, {})[key] = dS
                rs = part.route(L.TWIN[bank], r)
                ex = {"E": bank_exec(simE["opt"], eS), "S": eS}
                for nm, e_ in ex.items():
                    f = __import__("spb1_analyze").bank_functional(
                        C, dB, 2, part.lab_sem if bank == "SEM2" else part.lab_shuf,
                        {"e": e_, "p1": rs["p1"], "marg": rs["marg"]}, "deployed")
                    fun.setdefault((nm, bank), {})[key] = f
                nr = 3
                sims_meta.setdefault("E" + sfx, {})[str(key)] = {**L.policy_macs(simE, head.macs, nr), "sim_seconds": sim_sec}
                sims_meta.setdefault("Ecm" + sfx, {})[str(key)] = L.policy_macs(simC, head.macs, nr)
                sims_meta.setdefault("D" + sfx, {})[str(key)] = L.policy_macs(simD, head.macs, nr)
                if bank == "SEM2":
                    comb.setdefault("G", {})[key] = dG
                    comb.setdefault("H", {})[key] = np.where(part.pos < 32, dG, dS)
                    comb.setdefault("F", {})[key] = dB[np.arange(part.N), part.e_tr]
                    adv = U[:, :33] - U[:, 33:34]
                    top = adv.argmax(1)
                    Dall = np.concatenate([dB, dS[:, None]], 1)
                    comb.setdefault("Ugreedy", {})[key] = np.where(adv[np.arange(part.N), top] > cfg["abstain"],
                                                                   Dall[np.arange(part.N), top], dG)
                    p1 = rs["p1"].astype(np.float64)
                    for th in GATE_GRID_C:
                        comb.setdefault(f"Cgate_{th}", {})[key] = np.where(p1 >= th, dS, dG)
                    ch, sem32, fun32, semonly32, funonly32 = change_metrics(part, simE, dB)
                    extra.setdefault("change", {})[str(key)] = ch
                    extra.setdefault("commit", {})[str(key)] = commit_metrics(part, simE["commits"], dB, dS, dG,
                                                                              dG if cfg["fallback"] == "generic"
                                                                              else np.zeros_like(dG))
                    extra.setdefault("calibration", {})[str(key)] = calib_metrics(U, dB, dS, dG)
                    extra.setdefault("modes", {})[str(key)] = {
                        "search_frac": float(simE["search"].mean()), "alarms_per_1k": float(1000 * simE["alarm"].mean()),
                        "exec_share": {"G_or_base": float((simE["opt"] < 0).mean() + (simE["opt"] == 33).mean()),
                                       "S": float((simE["opt"] == 32).mean()),
                                       "specialist": float(((simE["opt"] >= 0) & (simE["opt"] < 32)).mean())},
                        "kappa": k1, "kappa_cm": kc}
                    masks_x = {"sem_change_first32": sem32, "fun_change_first32": fun32,
                               "semantic_only_first32": semonly32, "functional_only_first32": funonly32}
                    extra.setdefault("_masks", {})[s] = masks_x
                    if part_name == "CSEARCH":           # non-deployable diagnostics (engineering only)
                        diag = {}
                        o1 = L.simulate(part, dB, dS, dG, U, cfg, k1, mode="E", norm_sd=sd, oracle={"commit_true": True})
                        diag["O1_evidence_oracle"] = o1["d"]
                        own = np.stack([2 * part.lab_sem, 2 * part.lab_sem + 1], 1)
                        o2 = L.simulate(part, dB, dS, dG, U, cfg, k1, mode="E", norm_sd=sd, oracle={"cand": own})
                        diag["O2_candidate_oracle"] = o2["d"]
                        cs = (part.pos % 256 == 0) & (part.rp >= 0) & (part.r != part.rp)
                        o3 = L.simulate(part, dB, dS, dG, U, cfg, k1, mode="E", norm_sd=sd, oracle={"alarm_at": cs})
                        diag["O3_detection_oracle"] = o3["d"]
                        o123 = L.simulate(part, dB, dS, dG, U, cfg, k1, mode="E", norm_sd=sd,
                                          oracle={"commit_true": True, "cand": own, "alarm_at": cs})
                        diag["O123_all"] = o123["d"]
                        # O4: hindsight-best fixed option (33 incl. S, plus G) over each region's first 32 tokens
                        reg = part.doc * 4 + part.pos // 256
                        wmask = (part.pos % 256) < 32
                        Dall2 = np.concatenate([dB, dS[:, None], dG[:, None]], 1).astype(np.float64)
                        nreg = int(reg.max()) + 1
                        sums = np.zeros((nreg, 34))
                        np.add.at(sums, reg[wmask], Dall2[wmask])
                        bst = sums.argmin(1)
                        o4 = np.where(wmask, Dall2[np.arange(part.N), bst[reg]], dS)
                        diag["O4_window_hindsight"] = o4
                        for k_, v_ in diag.items():
                            comb.setdefault(k_, {})[key] = v_
                del U
        del part
    # ---------------- pooled comparisons
    masks = part_ref.masks()
    for k_, v_ in extra.get("_masks", {}).get(P.BANK_SEEDS[0], {}).items():
        masks[k_] = v_
    if theta_C is None:                              # gate threshold chosen on CSEARCH (lowest overall CE)
        theta_C = min(GATE_GRID_C, key=lambda th: np.mean([comb[f"Cgate_{th}"][k].mean() for k in comb["G"]]))
    comb["C"] = comb[f"Cgate_{theta_C}"]

    def cmp(a, b, mk="all"):
        ks = sorted(set(comb[a]) & set(comb[b]))
        pr = [(comb[a][k], comb[b][k]) for k in ks]
        return L.pooled_diff(part_ref, pr, W, None if mk == "all" else masks[mk])

    res = {"time": guard.now(), "gen": gen, "part": part_name, "cfg": cfg, "theta_C": theta_C,
           "n_tokens": int(part_ref.N), "n_papers": int(part_ref.P), "n_docs": int(part_ref.n_docs)}
    conds = ["G", "S", "C", "H", "Ugreedy", "D", "E", "Ecm", "F", "E_SHUF", "S_SHUF", "D_SHUF", "Ecm_SHUF"] + \
            [k for k in comb if k.startswith("O")]
    res["ce_vs_base"] = {c: {mk: float(np.mean([comb[c][k][masks[mk]].mean() for k in comb[c]])) for mk in
                             list(MASKS) + ["sem_change_first32", "fun_change_first32", "semantic_only_first32",
                                            "functional_only_first32"]} for c in conds if c in comb}
    res["ce_vs_base_per_seed"] = {c: {str(s): float(np.mean([comb[c][(s, r)].mean() for r in P.ROUTER_SEEDS]))
                                      for s in P.BANK_SEEDS} for c in conds if c in comb}
    comps = {}
    for a, b in [("E", "G"), ("E", "S"), ("E", "C"), ("E", "H"), ("E", "D"), ("E", "Ugreedy"), ("Ecm", "G"),
                 ("Ecm", "S"), ("Ecm", "E"), ("D", "S"), ("H", "S"), ("C", "S"), ("S", "G"), ("Ugreedy", "S"),
                 ("E_SHUF", "S_SHUF"), ("E", "E_SHUF")] + [(k, "E") for k in comb if k.startswith("O")]:
        if a in comb and b in comb:
            comps[f"{a}_minus_{b}"] = {mk: cmp(a, b, mk) for mk in list(MASKS) + ["sem_change_first32",
                                                                                   "fun_change_first32",
                                                                                   "semantic_only_first32",
                                                                                   "functional_only_first32"]}
            comps[f"{a}_minus_{b}"]["per_seed"] = {
                str(s): L.pooled_diff(part_ref, [(comb[a][(s, r)], comb[b][(s, r)]) for r in P.ROUTER_SEEDS], W)["mean"]
                for s in P.BANK_SEEDS}
    res["comparisons"] = comps
    ks = sorted(comb["G"])
    for x in ("E", "Ecm", "S", "D", "H", "C"):
        res.setdefault("oracle_recovery", {})[x] = {
            mk: L.recovery(part_ref, [comb["G"][k] for k in ks], [comb[x][k] for k in ks], [comb["F"][k] for k in ks],
                           W, None if mk == "all" else masks[mk]) for mk in ("all", "pos_0_31", "transition_first32",
                                                                            "stable_context")}
    # functional: deployed exposure-matched dI = I_SEM - I_SHUF under the same policy (pooled over combos)
    import spb1_analyze as SA
    for nm in ("E", "S"):
        per_seed, IFs, vals = {}, [], []
        for k in ks:
            fs, fh = fun[(nm, "SEM2")][k], fun[(nm, "SHUF2")][k]
            if fs["I"] is None or fh["I"] is None:
                continue
            vals.append(fs["I"] - fh["I"])
            IFs.append(fs["IF"] - fh["IF"])
            per_seed.setdefault(str(k[0]), []).append(fs["I"] - fh["I"])
        v = float(np.mean(vals)) if vals else None
        reps = v + W @ np.mean(IFs, 0) - np.mean(IFs, 0).sum() if vals else None
        k0 = ks[0]
        spec, npos, nsig = SA.summarize_spec(fun[(nm, "SEM2")][k0], W) if fun[(nm, "SEM2")][k0]["I"] is not None else ([], 0, 0)
        res.setdefault("functional", {})[nm] = {
            "dI": v, "ci": L.ci(reps) if vals else None, "per_seed": {s_: float(np.mean(x)) for s_, x in per_seed.items()},
            "I_SEM_mean": float(np.mean([fun[(nm, "SEM2")][k]["I"] for k in ks if fun[(nm, "SEM2")][k]["I"] is not None])),
            "I_SHUF_mean": float(np.mean([fun[(nm, "SHUF2")][k]["I"] for k in ks if fun[(nm, "SHUF2")][k]["I"] is not None])),
            "s0r0_specialists_positive": npos, "s0r0_specialists_significant": nsig,
            "s0r0_n_scored": fun[(nm, "SEM2")][k0]["n_scored"]}
    res["compute"] = sims_meta
    res["change_detection"] = extra.get("change")
    res["commits"] = extra.get("commit")
    res["head_calibration"] = extra.get("calibration")
    res["modes"] = extra.get("modes")
    res["wall_seconds"] = time.time() - t_start
    res["gates"] = gates(res)
    return res


def gates(res):
    c = res["comparisons"]
    out = {}
    for cand in ("E", "Ecm"):
        g = {}
        g["P1a"] = c[f"{cand}_minus_G"]["all"]["mean"] < 0 and c[f"{cand}_minus_G"]["all"]["ci"][1] < 0
        g["P1b"] = c[f"{cand}_minus_S"]["all"]["mean"] < 0 and c[f"{cand}_minus_S"]["all"]["ci"][1] < 0
        g["P2_first32_vs_G"] = c[f"{cand}_minus_G"]["pos_0_31"]["mean"] <= P.GATE_FIRST32_TOL
        g["P2_trans8_vs_S"] = c[f"{cand}_minus_S"]["transition_first8"]["mean"] < 0
        t32 = c[f"{cand}_minus_S"]["transition_first32"]
        g["P2_trans32_vs_S"] = t32["mean"] < 0 and t32["ci"][1] < 0
        gs = -c["S_minus_G"]["stable_context"]["mean"]
        ge = -c[f"{cand}_minus_G"]["stable_context"]["mean"]
        ret = ge / gs if gs > 0 else None
        g["stable_retention"] = ret
        g["P3_stable"] = ret is not None and ret >= P.GATE_STABLE_RET
        g["P4_seeds"] = all(v < 0 for v in c[f"{cand}_minus_S"]["per_seed"].values())
        f = res["functional"]["E"]
        g["P5_functional"] = (f["ci"] is not None and f["ci"][0] > 0 and all(v > 0 for v in f["per_seed"].values()))
        g["PASS"] = all(g[k] for k in ("P1a", "P1b", "P2_first32_vs_G", "P2_trans8_vs_S", "P2_trans32_vs_S",
                                       "P3_stable", "P4_seeds", "P5_functional"))
        out[cand] = g
    bm = "pos_0_31"
    out["mechanism"] = {
        "E_vs_D_first32": c["E_minus_D"][bm], "E_vs_H_first32": c["E_minus_H"][bm],
        "E_vs_D_trans32": c["E_minus_D"]["transition_first32"], "E_vs_H_trans32": c["E_minus_H"]["transition_first32"]}
    out["mechanism"]["supported"] = bool(out["E"]["PASS"] and all(
        v["ci"][1] < 0 for k, v in out["mechanism"].items() if isinstance(v, dict)))
    ov = np.mean([v["overhead"] for v in res["compute"]["Ecm"].values()])
    out["Ecm"]["overhead"] = float(ov)
    out["Ecm"]["P7_overhead"] = bool(ov <= P.OVERHEAD_TARGET)
    out["compute_matched_PASS"] = bool(out["Ecm"]["PASS"] and out["Ecm"]["P7_overhead"])
    out["E_overhead"] = float(np.mean([v["overhead"] for v in res["compute"]["E"].values()]))
    out["R_E"] = res["oracle_recovery"]["E"]["all"]["R"]
    out["R_preferred_target_met"] = bool(out["R_E"] >= 0.5)
    return out


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--gen", type=int, default=0)
    ap.add_argument("--part", default="CSEARCH")
    ap.add_argument("--frozen", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    P.check_prereg()
    if a.part not in ("CSEARCH", "CSELECT"):
        raise guard.GuardError("only CSEARCH / CSELECT exist in this experiment")
    if a.cmd == "heads":
        ensure_heads(a.gen)
    elif a.cmd == "eval":
        frozen = None
        if a.part == "CSELECT":
            fz = P.ART / "FROZEN.json"
            if not (a.frozen and fz.exists()):
                raise guard.GuardError("CSELECT requires FROZEN.json and --frozen")
            frozen = json.loads(fz.read_text())
            out = P.ANALYSIS / "CSELECT.json"
            if out.exists():
                raise guard.GuardError("CSELECT already evaluated once")
        else:
            out = P.ANALYSIS / f"gen{a.gen}_CSEARCH.json"
        ensure_heads(a.gen if frozen is None else frozen["gen"])
        res = evaluate(a.gen if frozen is None else frozen["gen"], a.part, frozen)
        P.ANALYSIS.mkdir(parents=True, exist_ok=True)
        guard.atomic_write_json(out, jsonable(res), overwrite=a.part == "CSEARCH")
        print(json.dumps(jsonable(res["gates"]))[:2000])


if __name__ == "__main__":
    main()
