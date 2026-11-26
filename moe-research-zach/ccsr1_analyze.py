#!/usr/bin/env python
"""csr-current-context-specialist-router-v1 analysis (CPU; prereg sections 6-9).

  ccsr1_analyze.py stage1 [--gen G]          SEL: gates G1-G6 per configuration and router seed, selection, triggers
  ccsr1_analyze.py stage2 [--gen G]          CALIB-SELECTION: gates G1-G6 for the frozen configuration
  ccsr1_analyze.py selftest                  planted-effect check of the gate code on synthetic choices

All router CE values are exact table lookups: CE_t = ce_base_t + d_X[t, choice_t] (zero action: + 0).
Functional metric: csur1 exposure-matched own-specialist advantage (csur1_analyze.functional) of the DEPLOYED router
choices, groups = paper-level C6 semantic clusters; ordinary (unmatched) own-vs-other ablation reported alongside."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402

POS_BINS = [(0, 32), (32, 128), (128, 512), (512, 100000)]


class SeedLite:
    """Minimal csur1_analyze.Seed-compatible object for one (table, router choices) pair."""

    def __init__(self, z, feats, choices):
        import numpy as np
        import csur1_analyze as CA
        from jsr_analyze import qcut
        self.papers = z["papers"]
        self.P = len(self.papers)
        sub = z["sub"].astype(bool)
        self.sub_idx = np.nonzero(sub)[0]
        self.half = CA.halves(self.papers)
        cb = z["ce_base"].astype(np.float64)
        lf = np.log(feats["freq"][z["tgt"]].astype(np.float64) + 1)
        self.base_st = ((qcut(lf, [1 / 3, 2 / 3]) * 5 + qcut(cb, [0.2, 0.4, 0.6, 0.8])) * 3
                        + np.searchsorted([128, 512], z["pos"], side="right")) * 3 + qcut(z["doclen"].astype(np.float64), [1 / 3, 2 / 3])
        self.z = {"lab_sem": z["lab_sem"], "paper": z["paper"]}
        for b, (e, qtop, marg) in choices.items():
            d = z[f"d_{b}"][sub]
            self.z[f"d_{b}"] = d
            self.z[f"e_{b}"] = e.astype(np.int64)
            es = e[sub].astype(np.int64)
            q = np.zeros((len(es), 16), np.float32)
            ok = es < 16
            q[np.nonzero(ok)[0], es[ok]] = qtop[ok].astype(np.float32)
            q[np.nonzero(ok)[0], (es[ok] + 1) % 16] = (qtop[ok] - marg[ok]).astype(np.float32)
            self.z[f"q_{b}"] = q

    def counts(self, b, lab_key="lab_sem", nlab=16, teacher=False):
        import numpy as np
        lab = self.z[lab_key].astype(np.int64)
        e = self.z[f"e_{b}"].astype(np.int64)
        ok = (lab >= 0) & (e >= 0) & (e < 16)
        idx = (self.z["paper"][ok].astype(np.int64) * nlab + lab[ok]) * 16 + e[ok]
        return np.bincount(idx, minlength=self.P * nlab * 16).reshape(self.P, nlab, 16).astype(np.float64)


def ce_of(z, bank, e):
    import numpy as np
    d = z[f"d_{bank}"]
    e = e.astype(np.int64)
    v = np.zeros(len(e), np.float64)
    ok = e < 16
    v[ok] = d[np.nonzero(ok)[0], e[ok]].astype(np.float64)
    return z["ce_base"].astype(np.float64) + v


def pathology(e):
    import numpy as np
    f = np.bincount(e[e < 16].astype(np.int64), minlength=16) / max(1, len(e))
    p = f / max(1e-12, f.sum())
    r = {"load_max": float(f.max()), "load_eff": float(1.0 / max(1e-12, (p ** 2).sum())), "n_dead": int((f < 1 / 160).sum())}
    r["ok"] = r["load_max"] <= 0.25 and r["load_eff"] >= 8 and r["n_dead"] <= 1
    return r


def ordinary_own(S, b):
    """Unmatched own-vs-other usefulness of each group's own specialist (same own map as the functional metric)."""
    import numpy as np
    import csur1_analyze as CA
    z = S.z
    si = S.sub_idx
    d = z[f"d_{b}"].astype(np.float64)
    lab = z["lab_sem"].astype(np.int64)[si]
    e = z[f"e_{b}"].astype(np.int64)[si]
    g = (d.sum(1)[:, None] - d) / 15.0 - d      # usefulness of specialist j relative to the mean of the others
    tok_half = S.half[z["paper"].astype(np.int64)][si]
    vals = []
    for fit_h, meas_h in ((True, False), (False, True)):
        sigma = CA.own_map(S, b, "lab_sem", S.half[z["paper"].astype(np.int64)] == fit_h)
        for k in range(16):
            ek = int(sigma[k])
            inv = (tok_half == meas_h) & (e == ek) & (lab >= 0)
            own, oth = inv & (lab == k), inv & (lab != k)
            if own.sum() >= 2000 and oth.sum() >= 2000:
                vals.append(g[own, ek].mean() - g[oth, ek].mean())
    return float(np.mean(vals)) if vals else float("nan")


def per_group(z, ce_a, ce_b):
    import numpy as np
    lab = z["lab_sem"].astype(np.int64)
    ok = lab >= 0
    n = np.bincount(lab[ok], minlength=16).astype(np.float64)
    s = np.bincount(lab[ok], weights=(ce_a - ce_b)[ok], minlength=16)
    return s / np.maximum(n, 1), n


def switch_masks(z):
    """Tokens in the first 64 positions of a region whose argmax p_sem differs from the previous region of the same
    document (post-switch), and tokens >= 64 into a region with the same label as the previous region (stable)."""
    import numpy as np
    rid = z["rid"].astype(np.int64)
    lab = z["lab_rsem"].astype(np.int64)
    pos = z["pos"].astype(np.int64)
    reg = pos // 256
    # label of the previous region of the same document (region rows of a document are consecutive): labels at
    # region starts, looked up at rid - 1 when the token is not in the document's first region
    starts = pos % 256 == 0
    keys = rid[starts]
    vals = lab[starts]
    order = np.argsort(keys)
    keys, vals = keys[order], vals[order]
    idx = np.searchsorted(keys, rid - 1)
    idx = np.clip(idx, 0, len(keys) - 1)
    prev = np.where((keys[idx] == rid - 1) & (reg > 0), vals[idx], -1)
    post = (prev >= 0) & (prev != lab) & (pos % 256 < 64)
    stable = (prev >= 0) & (prev == lab) & (pos % 256 >= 64)
    return post, stable


def evaluate_config(z, feats, rdir, part, seeds, want_func=True):
    """Per-seed metrics for one configuration (SEM and SHUF routers)."""
    import numpy as np
    import csur1_analyze as CA
    cb = z["ce_base"].astype(np.float64)
    ceG = cb + z["dce_G"].astype(np.float64)
    pos = z["pos"].astype(np.int64)
    post, stable = switch_masks(z)
    out = []
    for s in seeds:
        rec = {"seed": s}
        ch, ces = {}, {}
        for b in ("SEM", "SHUF"):
            a = np.load(rdir / f"{b}_s{s}_{part}.npz")
            js = json.loads((rdir / f"{b}_s{s}_{part}.json").read_text())
            if guard.sha256_file(rdir / f"{b}_s{s}_{part}.npz") != js["npz_sha256"]:
                raise guard.GuardError(f"router output hash mismatch {rdir}/{b}_s{s}")
            ch[b] = (a["e"], a["qtop"], a["margin"])
            ces[b] = ce_of(z, b, a["e"])
            rec[f"agree_region_prior_{b}"] = js["agree_region_prior"]
            rec[f"switch_rate_{b}"] = js["switch_rate"]
            rec[f"zero_action_frac_{b}"] = js.get("zero_action_frac", 0.0)
        ceT = {b: cb + z[f"dT_{b}"].astype(np.float64) for b in ("SEM", "SHUF")}
        cePV = cb + z["dPV_SEM"].astype(np.float64)
        rec.update({"ce_base": cb.mean(), "ce_G": ceG.mean(), "ce_PV_SEM": cePV.mean(), "ce_PV_SHUF": (cb + z["dPV_SHUF"].astype(np.float64)).mean(),
                    "ce_T_SEM": ceT["SEM"].mean(), "ce_T_SHUF": ceT["SHUF"].mean(),
                    "ce_oracle_SEM": (cb + z["d_SEM"].astype(np.float64).min(1)).mean(),
                    "ce_stu_SEM": ces["SEM"].mean(), "ce_stu_SHUF": ces["SHUF"].mean()})
        rec["stu_minus_G"] = rec["ce_stu_SEM"] - rec["ce_G"]
        rec["stu_minus_PV"] = rec["ce_stu_SEM"] - rec["ce_PV_SEM"]
        rec["regret_vs_oracle"] = rec["ce_stu_SEM"] - rec["ce_oracle_SEM"]
        rec["regret_vs_teacher"] = rec["ce_stu_SEM"] - rec["ce_T_SEM"]
        tadv = rec["ce_T_SHUF"] - rec["ce_T_SEM"]
        sadv = rec["ce_stu_SHUF"] - rec["ce_stu_SEM"]
        rec.update({"teacher_sem_adv": tadv, "student_sem_adv": sadv, "retention": sadv / tadv if tadv > 0 else float("nan")})
        rec["teacher_adv_vs_G_retained"] = ((rec["ce_G"] - rec["ce_stu_SEM"]) / (rec["ce_G"] - rec["ce_T_SEM"])
                                            if rec["ce_G"] > rec["ce_T_SEM"] else float("nan"))
        for lo, hi in POS_BINS:
            m = (pos >= lo) & (pos < hi)
            rec[f"pos_{lo}_{hi}"] = {"frac": float(m.mean()), "stu_SEM": float(ces["SEM"][m].mean() - cb[m].mean()),
                                     "stu_SHUF": float(ces["SHUF"][m].mean() - cb[m].mean()), "G": float(ceG[m].mean() - cb[m].mean()),
                                     "PV": float(cePV[m].mean() - cb[m].mean()), "T_SEM": float(ceT["SEM"][m].mean() - cb[m].mean())}
        m128 = pos >= 128
        rec["ge128_stu_minus_G"] = float(ces["SEM"][m128].mean() - ceG[m128].mean())
        rec["ge128_stu_minus_PV"] = float(ces["SEM"][m128].mean() - cePV[m128].mean())
        rec["post_switch"] = {"frac": float(post.mean()), "stu_minus_T": float((ces["SEM"] - ceT["SEM"])[post].mean()),
                              "stu_minus_G": float((ces["SEM"] - ceG)[post].mean())}
        rec["stable"] = {"frac": float(stable.mean()), "stu_minus_T": float((ces["SEM"] - ceT["SEM"])[stable].mean()),
                         "stu_minus_G": float((ces["SEM"] - ceG)[stable].mean())}
        dg, ng = per_group(z, ces["SEM"], cePV)
        imp = np.maximum(-dg * ng, 0)
        rec["group_vs_PV"] = {"delta": dg.tolist(), "n_improved": int((dg < 0).sum()),
                              "max_share": float(imp.max() / max(1e-12, imp.sum()))}
        dgG, _ = per_group(z, ces["SEM"], ceG)
        rec["group_vs_G"] = {"delta": dgG.tolist(), "n_better": int((dgG < 0).sum())}
        rec["path_SEM"] = pathology(ch["SEM"][0])
        rec["path_SHUF"] = pathology(ch["SHUF"][0])
        e = ch["SEM"][0].astype(np.int64)
        low = (ch["SEM"][1] < 0.25)
        es = e[z["sub"].astype(bool)]
        dsub = z["d_SEM"][z["sub"].astype(bool)].astype(np.float64)
        dch = np.where(es < 16, dsub[np.arange(len(es)), np.clip(es, 0, 15)], 0.0)
        rec["low_conf"] = {"frac": float(low.mean()), "mean_d_chosen": float(dch[low].mean()) if low.any() else float("nan")}
        if want_func:
            S = SeedLite(z, feats, ch)
            fs = CA.functional(S, "SEM")
            fh = CA.functional(S, "SHUF")
            W = CA.boot_weights(S.P, 2000, 20261301 + s)
            v, r = CA.func_diff([fs], [fh], W)
            ps = CA.per_specialist([fs], W) if fs["n_scored"] else None
            rec.update({"I_SEM": fs["I"], "I_SHUF": fh["I"], "I_SEM_minus_SHUF": float(v[0]), "I_diff_ci": CA.ci(r[0]),
                        "n_pos_specialists_SEM": (ps or {}).get("n_pos"), "n_sig_specialists_SEM": (ps or {}).get("n_sig"), "per_specialist_SEM": ps,
                        "I_ordinary_SEM": ordinary_own(S, "SEM"), "I_ordinary_SHUF": ordinary_own(S, "SHUF"),
                        "nmi_sem_SEM": CA.nmi_val(S, "SEM", "lab_sem"), "nmi_sem_SHUF": CA.nmi_val(S, "SHUF", "lab_sem"),
                        "swap_SEM": CA.swap_summary(S, "SEM", np.asarray(fs["sigma"][0]))})
        out.append(rec)
    return out


def gates(recs):
    import numpy as np
    g = {}
    g["G1_ce_vs_G_each_seed"] = all(r["stu_minus_G"] <= 0.002 for r in recs)
    g["G2_mean_better_than_PV_by_0.003"] = float(np.mean([r["stu_minus_PV"] for r in recs])) <= -0.003
    g["G3_retention_ge_0.6_each_seed"] = all(np.isfinite(r["retention"]) and r["retention"] >= 0.6 for r in recs)
    g["G4_functional_sem_gt_shuf_each_seed"] = all(np.isfinite(r.get("I_SEM_minus_SHUF", np.nan)) and r["I_SEM_minus_SHUF"] > 0 for r in recs)
    g["G5_no_collapse_each_seed"] = all(r["path_SEM"]["ok"] for r in recs)
    g["G6_not_only_first128_or_one_subject"] = all(r["ge128_stu_minus_G"] <= 0.002 and r["ge128_stu_minus_PV"] < 0
                                                   and r["group_vs_PV"]["n_improved"] >= 12
                                                   and r["group_vs_PV"]["max_share"] <= 0.5 for r in recs)
    g["all"] = all(g.values())
    return g


def triggers(recs):
    """Prereg section 9 triggers on the reference configuration (lowest mean student CE), precedence E5 > E2 > E3 >
    E4 > E1; returns the first that applies (None if none)."""
    import numpy as np
    m = lambda f: float(np.mean([f(r) for r in recs]))
    g1_fail = not all(r["stu_minus_G"] <= 0.002 for r in recs)
    sem_acc = m(lambda r: r["agree_region_prior_SEM"])
    post_gap = m(lambda r: r["post_switch"]["stu_minus_T"])
    stable_gap = m(lambda r: r["stable"]["stu_minus_T"])
    low_frac = m(lambda r: r["low_conf"]["frac"])
    low_harm = m(lambda r: r["low_conf"]["mean_d_chosen"])
    diag = {"g1_fail": g1_fail, "sem_acc": sem_acc, "post_switch_gap": post_gap, "stable_gap": stable_gap,
            "low_conf_frac": low_frac, "low_conf_mean_d": low_harm,
            "collapse": not all(r["path_SEM"]["ok"] for r in recs)}
    t = None
    if diag["collapse"]:
        t = "E5"
    elif g1_fail and sem_acc < 0.35:
        t = "E2"
    elif g1_fail and post_gap > 2 * max(stable_gap, 1e-6):
        t = "E3"
    elif low_frac >= 0.10 and low_harm > 0:
        t = "E4"
    elif g1_fail and sem_acc >= 0.35:
        t = "E1"
    return t, diag


def load_table(part):
    import numpy as np
    from csr.ccsr1_paths import ORV1, TABLES
    p = TABLES / f"{part}.npz"
    meta = json.loads(p.with_suffix(".json").read_text())
    if guard.sha256_file(p) != meta["sha256"]:
        raise guard.GuardError(f"{p} hash mismatch")
    return dict(np.load(p)), dict(np.load(ORV1 / "tables/token_feats.npz")), meta


def strip(o):
    import numpy as np
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if not (isinstance(v, np.ndarray) and v.size > 64)}
    if isinstance(o, (list, tuple)):
        return [strip(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def stage1(a):
    import numpy as np
    from csr.ccsr1_paths import ART, ROUTER_SEEDS, ROUTERS
    import ccsr1_configs as CC
    sfx = f"_g{a.gen}" if a.gen else ""
    z, feats, meta = load_table("SEL")
    cfgs = CC.stage1(a.gen)
    res = {"time": guard.now(), "generation": a.gen, "table_sha256": meta["sha256"], "configs": {}}
    for c in cfgs:
        recs = evaluate_config(z, feats, ROUTERS / f"S1{sfx}" / c["tag"], "SEL", ROUTER_SEEDS)
        res["configs"][c["tag"]] = {"cfg": c, "per_seed": recs, "gates": gates(recs),
                                    "mean_ce_stu_SEM": float(np.mean([r["ce_stu_SEM"] for r in recs]))}
        print(c["tag"], json.dumps({k: v for k, v in res["configs"][c["tag"]]["gates"].items()}),
              round(res["configs"][c["tag"]]["mean_ce_stu_SEM"], 5), flush=True)
    r3 = {t: v for t, v in res["configs"].items() if t.startswith("R3_")}
    ok = {t: v for t, v in r3.items() if v["gates"]["all"]}
    if ok:
        best = min(v["mean_ce_stu_SEM"] for v in ok.values())
        tied = [t for t in ok if ok[t]["mean_ce_stu_SEM"] - best < 1e-5]
        sel = "R3_bp" if "R3_bp" in tied else tied[0]
        res.update({"stage1_pass": True, "selected": sel, "selected_cfg": ok[sel]["cfg"]})
        res["trigger"], res["trigger_diag"] = None, None
    else:
        ref = min(r3, key=lambda t: r3[t]["mean_ce_stu_SEM"])
        t, diag = triggers(r3[ref]["per_seed"])
        res.update({"stage1_pass": False, "selected": None, "reference_config": ref, "trigger": t, "trigger_diag": diag})
    res["references"] = {k: res["configs"]["R3_bp"]["per_seed"][0][k] for k in
                         ("ce_base", "ce_G", "ce_PV_SEM", "ce_PV_SHUF", "ce_T_SEM", "ce_T_SHUF", "ce_oracle_SEM")}
    guard.atomic_write_json(ART / f"stage1{sfx}.json", strip(res), mode=0o400)
    print(json.dumps({k: res[k] for k in ("stage1_pass", "selected", "trigger", "trigger_diag")}, default=str))


def stage2(a):
    import numpy as np
    from csr.ccsr1_paths import ART, ROUTER_SEEDS, ROUTERS
    sfx = f"_g{a.gen}" if a.gen else ""
    fz = json.loads((ART / "FROZEN_S2.json").read_text())
    z, feats, meta = load_table("CS2")
    res = {"time": guard.now(), "generation": a.gen, "table_sha256": meta["sha256"], "frozen": fz, "configs": {}}
    for tag in fz["evaluate"]:
        recs = evaluate_config(z, feats, ROUTERS / f"S2{sfx}" / tag, "CS2", ROUTER_SEEDS)
        res["configs"][tag] = {"per_seed": recs, "gates": gates(recs),
                               "mean_ce_stu_SEM": float(np.mean([r["ce_stu_SEM"] for r in recs]))}
    sel = fz["selected"]
    res["stage2_pass"] = bool(res["configs"][sel]["gates"]["all"])
    res["preferred_student_beats_G_each_seed"] = all(r["stu_minus_G"] < 0 for r in res["configs"][sel]["per_seed"])
    guard.atomic_write_json(ART / f"stage2{sfx}.json", strip(res), mode=0o400)
    print(json.dumps({"stage2_pass": res["stage2_pass"], "gates": res["configs"][sel]["gates"]}))


def selftest(a):
    """Gate logic on synthetic per-seed records."""
    base = {"stu_minus_G": -0.001, "stu_minus_PV": -0.01, "retention": 0.8, "I_SEM_minus_SHUF": 0.01,
            "path_SEM": {"ok": True}, "ge128_stu_minus_G": 0.0, "ge128_stu_minus_PV": -0.005,
            "group_vs_PV": {"n_improved": 15, "max_share": 0.2}}
    good = [dict(base) for _ in range(3)]
    assert gates(good)["all"]
    bad = [dict(base) for _ in range(3)]
    bad[1] = {**base, "stu_minus_G": 0.0021}
    assert not gates(bad)["G1_ce_vs_G_each_seed"]
    bad = [dict(base) for _ in range(3)]
    bad[2] = {**base, "retention": 0.59}
    assert not gates(bad)["G3_retention_ge_0.6_each_seed"]
    bad = [dict(base) for _ in range(3)]
    bad[0] = {**base, "group_vs_PV": {"n_improved": 15, "max_share": 0.6}}
    assert not gates(bad)["G6_not_only_first128_or_one_subject"]
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stage1", "stage2", "selftest"])
    ap.add_argument("--gen", type=int, default=0)
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    {"stage1": stage1, "stage2": stage2, "selftest": selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
