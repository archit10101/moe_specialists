#!/usr/bin/env python
"""asrs1 analysis (CPU; prereg sections 6-8).

  asrs1_analyze.py ckpt --tag TAG            metrics of one evaluated checkpoint (cached in ART/analysis/TAG.json)
  asrs1_analyze.py stage1 --run RUN [--gen G]  Stage-1 criteria S1-1..S1-7 at the final cycle, per-cycle trajectories,
                                              references (frozen stage c00, ordinary base / G, ccsr1 SEMf / SEMj),
                                              failure attribution and the section-8 generation rule
  asrs1_analyze.py selftest

Per (checkpoint, router seed r) the ccsr1 Stage-2 code (ccsr1_analyze.evaluate_config, unchanged) runs on a table whose
SEM side is the semantic bank deployed by its router r and whose SHUF side is the shuffled bank deployed by its router r;
G := the checkpoint's token-routed generic bank (ALT-G for alternating milestones). Comparisons with the ordinary G use
per-token paired CE on identical SEL tokens (same rows, order and flattening)."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402

PAIRS = {"asrs1": [("SEM_s{r}", "SEM", "SHUF_s{r}", "SHUF")], "cfar1": [("SEM_s{r}", "SEM", "SHUF_s{r}", "SHUF")],
         "ccsr1": [("SEMf_s{r}", "SEMf_s{r}", "SHUFf_s{r}", "SHUFf_s{r}"), ("SEMj_s{r}", "SEMj_s{r}", "SHUFf_s{r}", "SHUFf_s{r}")]}


def load_eval(d):
    import numpy as np
    summ = json.loads((d / "summary_SEL.json").read_text())
    for f, h in summ["sha256"].items():
        if guard.sha256_file(d / f) != h:
            raise guard.GuardError(f"{d / f} hash mismatch")
    common = dict(np.load(d / "common_SEL.npz"))
    banks = {n: dict(np.load(d / f"bank_{n}_SEL.npz")) for n in summ["banks"]}
    deps = {x["name"]: dict(np.load(d / f"dep_{x['name']}_SEL.npz")) for x in summ["deployments"]}
    return summ, common, banks, deps


def write_router_file(rdir, b, r, dep, common, lab_key):
    import numpy as np
    rdir.mkdir(parents=True, exist_ok=True)
    p = rdir / f"{b}_s{r}_SEL.npz"
    if not p.exists():
        np.savez(p, e=dep["e"], qtop=dep["qtop"], margin=dep["margin"])
    e = dep["e"].astype(np.int64)
    load = np.bincount(e, minlength=16)[:16] / len(e)
    same = common["doc"][1:] == common["doc"][:-1]
    js = {"bank": b, "seed": r, "load": load.round(5).tolist(), "load_max": float(load.max()),
          "agree_region_prior": float((e == common[lab_key].astype(np.int64)).mean()),
          "switch_rate": float((dep["e"][1:] != dep["e"][:-1])[same].mean()), "zero_action_frac": 0.0,
          "npz_sha256": guard.sha256_file(p)}
    guard.atomic_write_json(p.with_suffix(".json"), js, overwrite=True)


def r3_teacher_subset(common, bank, prior_P, s_u=None, c=2.0, eta=0.5):
    """Target-dependent R3 teacher (non-causal diagnostic) on the swap subset: argmax(u/(c s_u) + eta log(p + 1e-4))."""
    import numpy as np
    sub = common["sub"].astype(bool)
    d = bank["d_sub"].astype(np.float64)
    u = -d
    if s_u is None:
        s_u = float(np.median(u.std(1, ddof=1)))
    lp = np.log(prior_P[common["rid"][sub].astype(np.int64)] + 1e-4)
    e = (u / (c * s_u) + eta * lp).argmax(1)
    cb = common["ce_base"][sub].astype(np.float64)
    return {"ce_R3_teacher_subset": float((cb + d[np.arange(len(e)), e]).mean()), "s_u_subset": s_u,
            "ce_oracle_subset": float((cb + d.min(1)).mean()), "ce_base_subset": float(cb.mean()),
            "ce_prior_argmax_subset": float((cb + d[np.arange(len(e)), lp.argmax(1)]).mean())}


def teacher_deployed(common, bank_s, bank_h):
    """Pseudo router outputs = region-teacher choices (for the specialist-forgetting diagnostic)."""
    import numpy as np
    sub = common["sub"].astype(bool)
    out = {}
    for b, B in (("SEM", bank_s), ("SHUF", bank_h)):
        e = B["eT"]
        out[b] = {"e": e, "qtop": np.ones(int(sub.sum()), np.float16), "margin": np.ones(int(sub.sum()), np.float16),
                  "dR": B["dT"]}
    return out


def one_pair(args):
    """evaluate_config for one (tag, pair, seed); runs in a worker process."""
    import numpy as np
    import ccsr1_analyze as AN
    from ccsr1_s3_analyze import seed_table
    tag, pair_name, r, edir, adir, dS, bS, dH, bH, teacher_mode = args
    summ, common, banks, deps = load_eval(Path(edir))
    feats = dict(np.load(guard.ROOT / "artifacts/oracle-residual-v1/tables/token_feats.npz"))
    if teacher_mode:
        td = teacher_deployed(common, banks[bS], banks[bH])
        depS, depH = td["SEM"], td["SHUF"]
    else:
        depS, depH = deps[dS], deps[dH]
    arm = lambda dep, B: {"dR": dep["dR"], "d_sub": banks[B]["d_sub"], "dPV": banks[B]["dPV"], "ePV": banks[B]["ePV"],
                          "dT": banks[B]["dT"], "eT": banks[B]["eT"]}
    z = seed_table(common, arm(depS, bS), arm(depH, bH), depS["e"], depH["e"])
    rdir = Path(adir) / tag / (pair_name + ("_teacher" if teacher_mode else ""))
    write_router_file(rdir, "SEM", r, depS, common, "lab_rsem")
    write_router_file(rdir, "SHUF", r, depH, common, "lab_rshuf")
    rec = AN.evaluate_config(z, feats, rdir, "SEL", (r,))[0]
    rec.pop("ce_oracle_SEM", None)
    rec.pop("regret_vs_oracle", None)
    rec["pair"] = pair_name
    rec["teacher_mode"] = teacher_mode
    return tag, pair_name, r, teacher_mode, AN.strip(rec) if hasattr(AN, "strip") else rec


def analyze_ckpt(tag, edir, workers=6):
    """All per-seed records of one evaluated checkpoint (cached)."""
    import numpy as np
    from concurrent.futures import ProcessPoolExecutor
    from csr import asrs1_paths as P
    from ccsr1_extract import partition
    out_p = P.ART / "analysis" / f"{tag}.json"
    summ = json.loads((Path(edir) / "summary_SEL.json").read_text())
    if out_p.exists():
        res = json.loads(out_p.read_text())
        if res.get("eval_sha256") == summ["sha256"]:
            return res
        raise guard.GuardError(f"{out_p} exists for different evaluation outputs")
    fmt = summ["format"]
    jobs = []
    for r in P.ROUTER_SEEDS:
        for dS, bS, dH, bH in PAIRS[fmt]:
            dS_, bS_, dH_, bH_ = (x.format(r=r) for x in (dS, bS, dH, bH))
            pname = dS_.split("_s")[0]
            jobs.append((tag, pname, r, str(edir), str(P.ART / "analysis_rdirs"), dS_, bS_, dH_, bH_, False))
    # region-teacher deployment of the banks (router-independent; once per bank pair)
    for dS, bS, dH, bH in PAIRS[fmt]:
        dS_, bS_, dH_, bH_ = (x.format(r=0) for x in (dS, bS, dH, bH))
        jobs.append((tag, dS_.split("_s")[0], 0, str(edir), str(P.ART / "analysis_rdirs"), dS_, bS_, dH_, bH_, True))
    recs = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(one_pair, jobs):
            recs.append(res)
    _, common, banks, _ = load_eval(Path(edir))
    sp, rows, sem, shuf, (off, Pt, Psh) = partition("SEL")
    teach = {}
    for dS, bS, dH, bH in PAIRS[fmt]:
        bS_, bH_ = bS.format(r=0), bH.format(r=0)
        teach[dS.split("_s")[0]] = {"SEM": r3_teacher_subset(common, banks[bS_], Pt), "SHUF": r3_teacher_subset(common, banks[bH_], Psh)}
    res = {"tag": tag, "time": guard.now(), "eval_dir": str(Path(edir).relative_to(guard.ROOT)), "eval_sha256": summ["sha256"],
           "format": fmt, "step": summ["step"], "seen": summ["seen"], "cycle": summ.get("cycle"),
           "ce_base": summ["ce_base"], "ce_G": summ["ce_G"], "teacher_stats": summ.get("teacher_stats"),
           "records": [{"pair": p_, "seed": r, "teacher_mode": tm, **rec} for (_, p_, r, tm, rec) in recs],
           "R3_teacher_subset": teach}
    out_p.parent.mkdir(parents=True, exist_ok=True)
    guard.atomic_write_json(out_p, res, mode=0o400)
    return res


def paired_vs(common_a, ce_a, common_b, ce_b, n_boot=2000, seed=20261301):
    """mean(ce_a - ce_b) on identical tokens + paper-bootstrap 95% CI."""
    import numpy as np
    for k in ("tgt", "doc", "pos"):
        if not np.array_equal(common_a[k], common_b[k]):
            raise guard.GuardError(f"evaluations are not token-aligned ({k})")
    diff = ce_a - ce_b
    pap = common_a["paper"].astype(np.int64)
    P_ = int(pap.max()) + 1
    s = np.bincount(pap, weights=diff, minlength=P_)
    n = np.bincount(pap, minlength=P_).astype(np.float64)
    rng = np.random.default_rng(seed)
    W = rng.multinomial(P_, np.full(P_, 1.0 / P_), size=n_boot).astype(np.float64)
    reps = (W @ s) / (W @ n)
    return {"mean": float(diff.mean()), "ci": [float(np.quantile(reps, 0.025)), float(np.quantile(reps, 0.975))]}


def ce_tok(common, dep=None, dce=None):
    import numpy as np
    cb = common["ce_base"].astype(np.float64)
    if dep is not None:
        return cb + dep["dR"].astype(np.float64)
    if dce is not None:
        return cb + dce.astype(np.float64)
    return cb


def train_diag(run_dir):
    """Per-cycle training diagnostics from metrics.jsonl (refresh stats, phase summaries)."""
    out = {}
    for line in (run_dir / "metrics.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("kind") == "refresh":
            out.setdefault(str(r["cycle"]), {})["refresh"] = {b: r["stats"][b] for b in ("SEM", "SHUF")}
        elif r.get("kind") == "phase_end":
            out.setdefault(str(r["cycle"]), {})[r["phase"]] = {k: v for k, v in r.items() if k not in ("kind",)}
    return out


def base_deficit_prediction(share=0.8):
    """Section 8.1: ordinary ccsr1 S3 training-loss deficit at round(share * 793) base steps (window means of 20 steps)."""
    import numpy as np
    rows = [json.loads(x) for x in (guard.ROOT / "runs/20261121T194254Z_ccsr1_S3/metrics.jsonl").read_text().splitlines()]
    lm = {r["step"]: r["lm"] for r in rows if "lm" in r}
    nb = int(round(share * 793))
    at = 318 + nb
    w_at = np.mean([lm[s] for s in range(at - 10, at + 10) if s in lm])
    w_end = np.mean([lm[s] for s in range(1091, 1111) if s in lm])
    return {"share": share, "base_steps": nb, "lm_at": float(w_at), "lm_end": float(w_end), "pred_deficit": float(w_at - w_end)}


def stage1(run, gen=0):
    import numpy as np
    from csr import asrs1_paths as P
    from csr import asrs1_log as LG
    run_dir = guard.ROOT / "runs" / run
    E = P.EVALS
    tags = {"c00": E / "c00", "ref_ccsr1_m1111": E / "ref_ccsr1_m1111"}
    for c in range(1, P.N_CYCLES + 1):
        tags[f"{run}_c{c:02d}"] = E / run / f"c{c:02d}"
    A = {t: analyze_ckpt(t, d) for t, d in tags.items()}
    final_tag = f"{run}_c{P.N_CYCLES:02d}"
    sel = lambda res, pair, tm=False: sorted([x for x in res["records"] if x["pair"] == pair and x["teacher_mode"] == tm],
                                             key=lambda x: x["seed"])
    # ---- frozen stage (c00) and identity check vs ccsr1 stage1.json
    s1 = json.loads((P.CCSR1_ART / "stage1.json").read_text())["configs"]["R3_c2e0.5"]["per_seed"]
    c00 = sel(A["c00"], "SEM")
    ident = []
    for a_, b_ in zip(c00, s1):
        ident.append({"seed": a_["seed"], "d_ce_stu": a_["ce_stu_SEM"] - b_["ce_stu_SEM"], "d_ce_G": a_["ce_G"] - b_["ce_G"],
                      "d_ce_base": a_["ce_base"] - b_["ce_base"], "d_dI": a_["I_SEM_minus_SHUF"] - b_["I_SEM_minus_SHUF"]})
    ident_ok = all(abs(x["d_ce_stu"]) <= 3e-4 and abs(x["d_ce_G"]) <= 3e-4 and abs(x["d_ce_base"]) <= 3e-4
                   and abs(x["d_dI"]) <= 1e-3 for x in ident)
    dI_frozen = float(np.mean([x["I_SEM_minus_SHUF"] for x in c00]))
    # ---- token-level references
    _, cref, _, dref = load_eval(tags["ref_ccsr1_m1111"])
    ce_ordG = ce_tok(cref, dce=cref["dce_G"])
    ce_ordB = ce_tok(cref)
    _, cfin, _, dfin = load_eval(tags[final_tag])
    ce_altG = ce_tok(cfin, dce=cfin["dce_G"])
    fin = sel(A[final_tag], "SEM")
    crit = []
    for rec in fin:
        r = rec["seed"]
        ce_s = ce_tok(cfin, dep=dfin[f"SEM_s{r}"])
        vs_ord = paired_vs(cfin, ce_s, cref, ce_ordG)
        vs_alt = paired_vs(cfin, ce_s, cfin, ce_altG)
        gvG = np.asarray(rec["group_vs_G"]["delta"])
        lab = cfin["lab_sem"].astype(np.int64)
        n_g = np.bincount(lab[lab >= 0], minlength=16).astype(np.float64)
        imp = np.maximum(-gvG * n_g, 0)
        c_ = {"seed": r, "ce_ASRS_SEM": rec["ce_stu_SEM"], "ce_ALTG": rec["ce_G"], "ce_ordG": float(ce_ordG.mean()),
              "minus_ordG": vs_ord, "minus_ALTG": vs_alt, "dI": rec["I_SEM_minus_SHUF"], "dI_ci": rec["I_diff_ci"],
              "dI_frozen": dI_frozen, "retention_dI": rec["I_SEM_minus_SHUF"] / dI_frozen,
              "n_pos_specialists": rec["n_pos_specialists_SEM"], "path": rec["path_SEM"],
              "ge128_minus_ALTG": rec["ge128_stu_minus_G"], "groups_better_than_ALTG": int((gvG < 0).sum()),
              "max_group_share": float(imp.max() / max(1e-12, imp.sum())),
              "post_switch": rec["post_switch"], "stable": rec["stable"]}
        c_["S1-1"] = vs_ord["mean"] <= 0.001
        c_["S1-2"] = rec["ce_stu_SEM"] < rec["ce_G"]
        c_["S1-3"] = rec["I_SEM_minus_SHUF"] > 0 and rec["I_diff_ci"][0] > 0
        c_["S1-4"] = rec["I_SEM_minus_SHUF"] >= 0.75 * dI_frozen
        c_["S1-5"] = (rec["n_pos_specialists_SEM"] or 0) >= 12
        c_["S1-6"] = rec["path_SEM"]["n_dead"] == 0 and rec["path_SEM"]["load_eff"] >= 8 and rec["path_SEM"]["load_max"] <= 0.25
        c_["S1-7"] = (rec["ge128_stu_minus_G"] < 0 and c_["groups_better_than_ALTG"] >= 12 and c_["max_group_share"] <= 0.5)
        c_["all"] = all(c_[f"S1-{i}"] for i in range(1, 8))
        crit.append(c_)
    passed = all(c["all"] for c in crit)
    # ---- trajectories and references
    traj = []
    for t in ["c00"] + [f"{run}_c{c:02d}" for c in range(1, P.N_CYCLES + 1)]:
        res = A[t]
        rs = sel(res, "SEM")
        tm = sel(res, "SEM", True)
        traj.append({"tag": t, "cycle": 0 if t == "c00" else int(t[-2:]), "step": res["step"], "ce_base": res["ce_base"],
                     "ce_G": res["ce_G"], "ce_T_SEM": rs[0]["ce_T_SEM"], "ce_T_SHUF": rs[0]["ce_T_SHUF"],
                     "ce_stu_SEM": [x["ce_stu_SEM"] for x in rs], "ce_stu_SHUF": [x["ce_stu_SHUF"] for x in rs],
                     "stu_minus_G": [x["stu_minus_G"] for x in rs], "dI": [x["I_SEM_minus_SHUF"] for x in rs],
                     "dI_teacher_deployed": tm[0]["I_SEM_minus_SHUF"] if tm else None,
                     "I_SEM": [x["I_SEM"] for x in rs], "I_SHUF": [x["I_SHUF"] for x in rs],
                     "n_pos": [x["n_pos_specialists_SEM"] for x in rs], "switch": [x["switch_rate_SEM"] for x in rs],
                     "agree_prior": [x["agree_region_prior_SEM"] for x in rs], "load_max": [x["path_SEM"]["load_max"] for x in rs],
                     "load_eff": [x["path_SEM"]["load_eff"] for x in rs], "n_dead": [x["path_SEM"]["n_dead"] for x in rs],
                     "retention_teacher_adv": [x["retention"] for x in rs], "regret_vs_teacher": [x["regret_vs_teacher"] for x in rs],
                     "post_switch_minus_G": [x["post_switch"]["stu_minus_G"] for x in rs],
                     "stable_minus_G": [x["stable"]["stu_minus_G"] for x in rs],
                     "pos_0_32_stu": [x["pos_0_32"]["stu_SEM"] for x in rs], "pos_0_32_G": rs[0]["pos_0_32"]["G"],
                     "R3_teacher_subset": res["R3_teacher_subset"]["SEM"]})
    refs = {"ordinary_base_ce": float(ce_ordB.mean()), "ordinary_G_ce": float(ce_ordG.mean()),
            "ALT_base_ce": A[final_tag]["ce_base"], "ALTG_ce": A[final_tag]["ce_G"]}
    for pair in ("SEMf", "SEMj"):
        rs = sel(A["ref_ccsr1_m1111"], pair)
        refs[pair] = [{"seed": x["seed"], "ce": x["ce_stu_SEM"], "minus_ordG": x["stu_minus_G"], "dI": x["I_SEM_minus_SHUF"],
                       "dI_ci": x["I_diff_ci"], "n_pos": x["n_pos_specialists_SEM"], "switch": x["switch_rate_SEM"],
                       "retention_dI_vs_c00": x["I_SEM_minus_SHUF"] / dI_frozen} for x in rs]
    for c_ in crit:
        r = c_["seed"]
        ce_s = ce_tok(cfin, dep=dfin[f"SEM_s{r}"])
        for pair in ("SEMf", "SEMj"):
            c_[f"minus_ccsr1_{pair}"] = paired_vs(cfin, ce_s, cref, ce_tok(cref, dep=dref[f"{pair}_s{r}"]))
        c_["ALTbase_minus_ordbase"] = paired_vs(cfin, ce_tok(cfin), cref, ce_ordB)
    # ---- section 8 generation rule
    gen_rule = {"failed": sorted({k for c in crit for k in [f"S1-{i}" for i in range(1, 8)] if not c[k]})}
    excess = max(c["minus_ordG"]["mean"] - 0.001 for c in crit)
    gen_rule["S1_1_excess_max"] = excess
    pred = base_deficit_prediction(0.8)
    gen_rule["phase_ratio_prediction"] = pred
    diag = train_diag(run_dir)
    if passed:
        gen_rule["class"], gen_rule["generation"] = None, None
    elif "S1-1" in gen_rule["failed"] and excess > 0.004:
        gen_rule["class"] = "base_update_deficit"
        gen_rule["generation"] = "phase_ratio_0.8" if pred["pred_deficit"] <= 0.003 else None
    else:
        sw = float(np.mean([c["path"]["load_max"] for c in crit]))
        tr_fin = traj[-1]
        collapse = (min(tr_fin["switch"]) < 0.03) or ("S1-6" in gen_rule["failed"])
        acc = [(int(c), v.get("router", {})) for c, v in sorted(diag.items(), key=lambda kv: int(kv[0]))]
        drift_n = 0
        for (c0, r0), (c1, r1) in zip(acc[:-1], acc[1:]):
            keys = [k for k in r1 if k.startswith("SEM_s")]
            if keys and all(r1[k]["acc_first"] < 0.8 * r0[k]["acc_last"] for k in keys):
                drift_n += 1
        drift = drift_n >= 5 and min(tr_fin["switch"]) > 2 * max(traj[0]["switch"])
        ovr = float(np.mean([v["refresh"]["SEM"]["override_frac_spec"] for v in diag.values() if "refresh" in v]))
        noise = ovr > 0.5 and ({"S1-3", "S1-4"} & set(gen_rule["failed"]))
        forget = ("S1-4" in gen_rule["failed"] and tr_fin["dI_teacher_deployed"] is not None and traj[0]["dI_teacher_deployed"]
                  and tr_fin["dI_teacher_deployed"] < 0.75 * traj[0]["dI_teacher_deployed"])
        gen_rule.update({"collapse": bool(collapse), "drift_cycles": drift_n, "drift": bool(drift), "override_frac_spec": ovr,
                         "teacher_noise": bool(noise), "forgetting": bool(forget), "mean_load_max": sw})
        cls = ("router_collapse" if collapse else "router_drift" if drift else "teacher_noise" if noise
               else "specialist_forgetting" if forget else None)
        gen_rule["class"] = cls
        gen_rule["generation"] = {"router_collapse": "router_lr_x1/3", "router_drift": "refresh_before_router",
                                  "teacher_noise": "c=4", "specialist_forgetting": "spec_lr_x0.5"}.get(cls)
    if gen >= 2:
        gen_rule["generation"] = None
    out = {"time": guard.now(), "run": run, "generation": gen, "stage1_pass": passed, "criteria": crit,
           "identity_check": {"ok": ident_ok, "per_seed": ident}, "dI_frozen": dI_frozen, "trajectory": traj,
           "references": refs, "generation_rule": gen_rule, "train_diag": diag, "prereg_sha256": P.check_prereg()}
    p = P.ART / (f"stage1{'_g' + str(gen) if gen else ''}.json")
    guard.atomic_write_json(p, out, mode=0o400)
    print(json.dumps({"stage1_pass": passed, "identity_ok": ident_ok, "failed": gen_rule["failed"],
                      "class": gen_rule["class"], "generation": gen_rule["generation"],
                      "per_seed": [{k: (v if not isinstance(v, dict) else v.get("mean", v)) for k, v in c.items()
                                    if k in ("seed", "minus_ordG", "minus_ALTG", "dI", "retention_dI", "n_pos_specialists")} for c in crit]},
                     default=str), flush=True)
    LG.observe(f"stage1 analysis written ({p.name}): pass={passed}, failed={gen_rule['failed']}, class={gen_rule['class']}")
    return out


def selftest():
    import numpy as np
    rng = np.random.default_rng(0)
    n = 4000
    ca = {"tgt": np.arange(n), "doc": np.repeat(np.arange(40), 100), "pos": np.tile(np.arange(100), 40),
          "paper": np.repeat(np.arange(20), 200)}
    a = rng.normal(size=n)
    r = paired_vs(ca, a + 0.5, ca, a)
    assert abs(r["mean"] - 0.5) < 1e-12 and r["ci"][0] <= 0.5 <= r["ci"][1]
    cb = dict(ca)
    cb["tgt"] = cb["tgt"].copy()
    cb["tgt"][0] = -5
    try:
        paired_vs(ca, a, cb, a)
        raise AssertionError("misalignment not detected")
    except guard.GuardError:
        pass
    common = {"sub": np.array([1, 1, 0], bool), "rid": np.array([0, 1, 1]), "ce_base": np.array([1.0, 2.0, 3.0])}
    bank = {"d_sub": np.array([[-0.1, 0.0] + [0.0] * 14, [0.0, -0.2] + [0.0] * 14], np.float32)}
    Pr = np.full((2, 16), 1 / 16)
    t = r3_teacher_subset(common, bank, Pr, s_u=0.05)
    assert abs(t["ce_R3_teacher_subset"] - (0.9 + 1.8) / 2) < 1e-6 and abs(t["ce_oracle_subset"] - 1.35) < 1e-6
    print("asrs1_analyze selftest ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["ckpt", "stage1", "selftest"])
    ap.add_argument("--tag")
    ap.add_argument("--eval-dir")
    ap.add_argument("--run")
    ap.add_argument("--gen", type=int, default=0)
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    if a.cmd == "selftest":
        selftest()
    elif a.cmd == "ckpt":
        res = analyze_ckpt(a.tag, guard.inside_root(a.eval_dir))
        print(json.dumps({"tag": a.tag, "n_records": len(res["records"])}))
    else:
        stage1(a.run, a.gen)


if __name__ == "__main__":
    main()
