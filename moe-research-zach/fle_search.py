#!/usr/bin/env python
"""csr-forecast-latent-edl-v1 training-only search (prereg section 8). Seed 100, FINAL plan, successive halving at
5M (step 80) / 20M (318) / 50M (793). CALIB-SEARCH for stages 1-2; CALIB-SELECTION opened once at stage 3. VAL never.

  fle_search.py t4                  : TOKEN-4 search reference (to 50M) + dev evals at s5/s20/s50
  fle_search.py stage1 [--only C..] : registered stage-1 candidates (train to s5 + dev eval + diagnosis, in order)
  fle_search.py promote --stage 1|2 : frozen promotion rule -> search/promotion_s<k>.json
  fle_search.py stage2 | stage3     : continue promoted candidates; stage3 also opens CALIB-SELECTION and freezes
  fle_search.py table               : candidate table (markdown + tsv)"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402
from csr.fle_paths import ALL, ART, LATENT, UPPER  # noqa: E402

SEED = 100
V = "v2"  # candidate run-name code version (deviation 6: JSD clamp fix)
STEPS = {"s5": 80, "s20": 318, "s50": 793}
TOL = {"s5": 0.02, "s20": 0.015, "s50": 0.01}
SDIR = ART / "search"
BASE = {"cfg": "A1_K16_L256", "beta": 1e-4, "layers": "upper"}
REGISTERED = {
    "C01": {}, "C02": {"cfg": "A1_K8_L256"}, "C03": {"cfg": "A1_K32_L256"}, "C04": {"beta": 5e-5},
    "C05": {"beta": 5e-4}, "C06": {"cfg": "A1_K16_L128"}, "C07": {"cfg": "A2_K16_L256"}, "C08": {"layers": "all"},
    "C09": {"cfg": "A1_K16_L256_H16"}, "C10": {"cfg": "A1_K16_L256_D2"},
}


def cand_spec(cid):
    reg = json.loads((SDIR / "candidates.json").read_text()) if (SDIR / "candidates.json").exists() else {}
    knobs = {**BASE, **(REGISTERED[cid] if cid in REGISTERED else reg[cid]["knobs"])}
    return knobs


def run_dir_of(run_name):
    st = json.loads((ART / "driver_state.json").read_text()) if (ART / "driver_state.json").exists() else {}
    rd = st.get("run_dirs", {}).get(run_name)
    if rd:
        return rd
    cands = sorted((guard.ROOT / "runs").glob(f"*_forecast_latent_edl_v1_{run_name}"))
    return str(cands[-1].relative_to(guard.ROOT)) if cands else None


def remember_run_dir(run_name):
    from fle_driver import update_state
    rd = run_dir_of(run_name)
    update_state(lambda st: st.setdefault("run_dirs", {}).__setitem__(run_name, rd))
    return rd


def ensure_latent(cfg_id):
    from fle_driver import run_job
    run_job(f"latent_fit_{cfg_id}", ["fle_latent.py", "fit", "--cfg", cfg_id], gpu=False)
    run_job(f"latent_apply_{cfg_id}_calib_search", ["fle_latent.py", "apply", "--cfg", cfg_id, "--cache",
                                                       "calib_search"], gpu=False)


def train_to(run_name, cfg_path, tag, first):
    from fle_driver import run_job
    args = ["fle_train.py", "--config", str(cfg_path), "--max-steps", str(STEPS[tag])]
    if not first:
        args += ["--run-dir", run_dir_of(run_name)]
    run_job(f"train_{run_name}_{tag}", args, gpu=True, expect=(75, 0), disk_gb=3.0)
    remember_run_dir(run_name)


def dev_eval(run_name, tag, latent=None):
    from fle_driver import run_job
    args = ["fle_evaluate.py", "dev", "--run-dir", run_dir_of(run_name), "--run-name", run_name, "--tag", tag]
    if latent:
        args += ["--latent", latent]
    run_job(f"eval_dev_{run_name}_{tag}", args, gpu=True, disk_gb=0.5)


def do_t4():
    from fle_configs import make
    p, _ = make(f"s_T4_s{SEED}", SEED, {}, "search", diag_every=25)
    first = run_dir_of(f"s_T4_s{SEED}") is None
    train_to(f"s_T4_s{SEED}", p, "s50", first)
    for tag in ("s5", "s20", "s50"):
        dev_eval(f"s_T4_s{SEED}", tag, latent=BASE["cfg"])


def build_candidate(cid):
    from fle_configs import latent_cond, make
    k = cand_spec(cid)
    ensure_latent(k["cfg"])
    cond = latent_cond(k["cfg"], k["beta"], UPPER if k["layers"] == "upper" else ALL)
    if k.get("target"):
        pass
    return make(f"s_{cid}{V}_s{SEED}", SEED, cond, "search", diag_every=25)[0], k


def stage(cid, tag):
    p, k = build_candidate(cid)
    rn = f"s_{cid}{V}_s{SEED}"
    train_to(rn, p, tag, first=(tag == "s5" and run_dir_of(rn) is None))
    dev_eval(rn, tag, latent=k["cfg"])
    diagnose(cid, tag)


def load_eval(run_name, tag, mode="dev"):
    p = ART / "evals" / run_name / f"{mode}_{tag}.json"
    return json.loads(p.read_text()) if p.exists() else None


def diagnose(cid, tag):
    """Frozen diagnostics (prereg 8.2) of one candidate at one milestone vs the T4 reference at the same step."""
    import numpy as np
    rn = f"s_{cid}{V}_s{SEED}"
    ev, t4 = load_eval(rn, tag), load_eval(f"s_T4_s{SEED}", tag)
    k = cand_spec(cid)
    meta = json.loads((LATENT / k["cfg"] / "meta.json").read_text())
    rd = guard.ROOT / run_dir_of(rn)
    mets = [json.loads(x) for x in (rd / "metrics.jsonl").read_text().splitlines()]
    diag = [m["diag"] for m in mets if "diag" in m and m["diag"]["step"] < STEPS[tag]]
    tmets = [json.loads(x) for x in (guard.ROOT / run_dir_of(f"s_T4_s{SEED}") / "metrics.jsonl").read_text()
             .splitlines()]
    tl = {m["step"]: m["lm"] for m in tmets if "step" in m}
    gap = {m["step"]: m["lm"] - tl[m["step"]] for m in mets if "step" in m and m["step"] in tl and m["step"] < STEPS[tag]}
    a, b = ev["audit"], t4["audit"]
    load = np.asarray(ev["audit"]["load"])
    E = load.shape[1]
    rdc, rdt = np.asarray(a["route_dist_by_domain"]), np.asarray(b["route_dist_by_domain"])
    m_ = 0.5 * (rdc + rdt)
    kl = lambda x, y: (x * (np.log(np.clip(x, 1e-12, 1)) - np.log(np.clip(y, 1e-12, 1)))).sum(-1)
    js_t4 = (0.5 * kl(rdc, m_) + 0.5 * kl(rdt, m_)).mean(1)  # (L,)
    med = lambda key: float(np.median([d[key] for d in diag if key in d])) if diag else float("nan")
    out = {"cid": cid, "tag": tag, "knobs": k, "ce": ev["calib_search_B"]["ce"], "ce_t4": t4["calib_search_B"]["ce"],
           "dce": ev["calib_search_B"]["ce"] - t4["calib_search_B"]["ce"],
           "fsi_domain": a["domain"]["fsi"], "fsi_cluster": a["cluster"]["fsi"],
           "fsi_domain_t4": b["domain"]["fsi"], "fsi_cluster_t4": b["cluster"]["fsi"],
           "mi_domain": a["domain"]["routing"]["mi_mean"], "mi_cluster": a["cluster"]["routing"]["mi_mean"],
           "mi_domain_t4": b["domain"]["routing"]["mi_mean"], "mi_cluster_t4": b["cluster"]["routing"]["mi_mean"],
           "mi_latent": (a.get("latent") or {}).get("mi_mean"), "mi_latent_t4": (b.get("latent") or {}).get("mi_mean")
           if k["cfg"] == BASE["cfg"] else None,
           "topic_pred_domain": a["domain"]["topic_pred_acc"], "topic_pred_domain_t4": b["domain"]["topic_pred_acc"],
           "routing_entropy": a["routing_entropy"], "load_min": float(load.min()), "load_max": float(load.max()),
           "js_route_vs_t4_by_layer": js_t4.tolist(), "median_gn_lm": med("gn_lm"), "median_gn_lb": med("gn_lb"),
           "median_gn_aux": med("gn_aux"), "median_jsd": med("jsd"),
           "jsd_by_layer_last": diag[-1].get("jsd_by_layer") if diag else None,
           "train_gap_vs_t4": {str(s): g for s, g in sorted(gap.items())},
           "cluster": {kk: meta["diag"][kk] for kk in ("tau", "assign_entropy", "stability_hungarian_agreement",
                                                       "predictive_gain_nats_per_token", "max_prob_mean")},
           "time": guard.now()}
    out["aux_lm_grad_ratio"] = out["median_gn_aux"] / max(1e-12, out["median_gn_lm"])
    flags = []
    if not all(math.isfinite(v) for v in (out["ce"], out["fsi_domain"], out["fsi_cluster"])):
        flags.append("nonfinite")
    if out["load_min"] < 0.1 / E or out["load_max"] > 4 / E:
        flags.append("collapse_or_unused")
    ent_cap = max(0.98, b["routing_entropy"] + 0.01)  # deviation 5: relative to T4 at the same milestone
    out["routing_entropy_t4"], out["routing_entropy_cap"] = b["routing_entropy"], ent_cap
    if not (0.2 < out["routing_entropy"] < ent_cap):
        flags.append("routing_entropy")
    if out["aux_lm_grad_ratio"] < 1e-3:
        flags.append("aux_weak")
    if out["dce"] > 0.02 and out["aux_lm_grad_ratio"] > 0.1:
        flags.append("aux_interferes")
    if out["cluster"]["assign_entropy"] > 0.7:
        flags.append("diffuse_assignments")
    out["flags"] = flags
    out["valid"] = not any(f in flags for f in ("nonfinite", "collapse_or_unused", "routing_entropy"))
    out["eligible"] = out["valid"] and out["dce"] <= tol(tag)
    (ART / "diagnostics").mkdir(exist_ok=True)
    guard.atomic_write_json(ART / "diagnostics" / f"{cid}_{tag}.json", out, overwrite=True)
    print(json.dumps({k_: v for k_, v in out.items() if not isinstance(v, (dict, list))}))
    return out


def tol(tag):
    p = SDIR / "noise_calibration.json"
    if tag == "s5" and p.exists():
        return json.loads(p.read_text())["tol_s5_used"]  # deviation 7
    return TOL[tag]


def rank_candidates(rows, n_keep):
    """Frozen rule: eligible rows ranked by mean rank of FSI_domain and FSI_cluster (descending); ties -> lower dCE.
    If none is eligible: the single lowest-dCE valid row, flagged INELIGIBLE."""
    for r in rows:
        if "tag" in r:
            r["eligible"] = bool(r["valid"] and r["dce"] <= tol(r["tag"]))
    el = [r for r in rows if r["eligible"]]
    if not el:
        val = sorted([r for r in rows if r["valid"]], key=lambda r: r["dce"])
        return [dict(val[0], flag="INELIGIBLE")] if val else []

    def ranks(key):
        order = sorted(el, key=lambda r: -r[key])
        return {r["cid"]: i for i, r in enumerate(order)}
    rd, rc = ranks("fsi_domain"), ranks("fsi_cluster")
    for r in el:
        r["mean_rank"] = 0.5 * (rd[r["cid"]] + rc[r["cid"]])
    return sorted(el, key=lambda r: (r["mean_rank"], r["dce"]))[:n_keep]


def promote(stage_n):
    tag = {1: "s5", 2: "s20"}[stage_n]
    cands = json.loads((SDIR / f"stage{stage_n}_list.json").read_text())
    rows = [json.loads((ART / "diagnostics" / f"{c}_{tag}.json").read_text()) for c in cands]
    keep = rank_candidates(rows, 4 if stage_n == 1 else 2)
    out = {"stage": stage_n, "tag": tag, "rule": "eligible (valid, dCE<=tol) ranked by mean FSI rank (domain, cluster)",
           "tol": tol(tag), "rows": [{k: r.get(k) for k in ("cid", "dce", "fsi_domain", "fsi_cluster", "valid",
                                                          "eligible", "mean_rank", "flags")} for r in rows],
           "promoted": [r["cid"] for r in keep], "flags": {r["cid"]: r.get("flag") for r in keep}, "time": guard.now()}
    p = SDIR / f"promotion_s{stage_n}.json"
    if p.exists():
        raise guard.GuardError(f"{p} exists")
    guard.atomic_write_json(p, out, mode=0o400)
    print(json.dumps(out, indent=1))
    return out


def respond(max_new=2):
    """Prereg 8: up to 2 diagnosis-driven stage-1 candidates from the incumbent's first matching flag."""
    lst = json.loads((SDIR / "stage1_list.json").read_text())
    regp = SDIR / "candidates.json"
    reg = json.loads(regp.read_text()) if regp.exists() else {}
    added = [c for c in lst if c not in REGISTERED]
    log = []
    while len(added) < max_new:
        rows = [json.loads((ART / "diagnostics" / f"{c}_s5.json").read_text()) for c in lst]
        top = rank_candidates([dict(r) for r in rows], 1)
        if not top:
            break
        inc = [r for r in rows if r["cid"] == top[0]["cid"]][0]
        k = dict(inc["knobs"])
        new = None
        for fl in inc["flags"]:
            if fl == "aux_weak":
                b10 = k["beta"] * 10
                new = {**k, "beta": b10} if b10 <= 5e-4 + 1e-12 else {**k, "cfg": "A1_K8_L256" if k["cfg"].startswith("A1")
                                                                       else k["cfg"]}
            elif fl == "aux_interferes":
                new = {**k, "beta": k["beta"] / 2} if k["beta"] / 2 in (5e-5, 1e-4) else {**k, "layers": "upper"}
            elif fl == "diffuse_assignments":
                new = {**k, "cfg": k["cfg"] + "_T25"}
            if new is not None:
                break
        ran = [cand_spec(c) for c in lst]
        if new is None or new in ran:
            log.append({"incumbent": inc["cid"], "flags": inc["flags"], "action": "none (no matching flag or already run)"})
            break
        cid = f"C{11 + len(added)}"
        reg[cid] = {"knobs": {kk: new[kk] for kk in ("cfg", "beta", "layers")}, "reason": f"incumbent {inc['cid']} "
                    f"flags {inc['flags']}", "time": guard.now()}
        guard.atomic_write_json(regp, reg, overwrite=True)
        lst.append(cid)
        guard.atomic_write_json(SDIR / "stage1_list.json", lst, overwrite=True)
        log.append({"incumbent": inc["cid"], "flags": inc["flags"], "new": cid, "knobs": reg[cid]["knobs"]})
        stage(cid, "s5")
        added.append(cid)
    guard.atomic_write_json(SDIR / f"respond_{guard.utc_stamp()}.json", {"log": log}, mode=0o400)
    print(json.dumps(log))


def do_stage3():
    """Continue <= 2 stage-2 promoted candidates to 50M; open CALIB-SELECTION once; freeze exactly one winner."""
    from fle_driver import run_job
    pr = json.loads((SDIR / "promotion_s2.json").read_text())["promoted"]
    guard.atomic_write_json(SDIR / "stage3_list.json", pr, overwrite=True)
    for c in pr:
        stage(c, "s50")
    if not (ART / "STAGE3_OPEN.json").exists():
        guard.atomic_write_json(ART / "STAGE3_OPEN.json", {"time": guard.now(), "candidates": pr,
                                                           "reference": f"s_T4_s{SEED}"}, mode=0o400)
    run_job("labels_calib_sel", ["fle_labels.py"], gpu=True, disk_gb=0.5)
    for rn in [f"s_T4_s{SEED}"] + [f"s_{c}{V}_s{SEED}" for c in pr]:
        run_job(f"eval_calibsel_{rn}", ["fle_evaluate.py", "calibsel", "--run-dir", run_dir_of(rn), "--run-name", rn,
                                        "--tag", "s50"], gpu=True, disk_gb=0.5)
    t4 = load_eval(f"s_T4_s{SEED}", "s50", "calibsel")
    rows = []
    for c in pr:
        ev = load_eval(f"s_{c}{V}_s{SEED}", "s50", "calibsel")
        d50 = json.loads((ART / "diagnostics" / f"{c}_s50.json").read_text())
        dce = ev["calib_sel"]["ce"] - t4["calib_sel"]["ce"]
        rows.append({"cid": c, "dce": dce, "ce": ev["calib_sel"]["ce"], "ce_t4": t4["calib_sel"]["ce"],
                     "fsi_domain": ev["audit"]["domain"]["fsi"], "fsi_cluster": ev["audit"]["cluster"]["fsi"],
                     "fsi_domain_t4": t4["audit"]["domain"]["fsi"], "fsi_cluster_t4": t4["audit"]["cluster"]["fsi"],
                     "valid": d50["valid"], "eligible": d50["valid"] and dce <= TOL["s50"]})
    keep = rank_candidates([dict(r) for r in rows], 1)
    win = keep[0]["cid"]
    k = cand_spec(win)
    frozen = {"winner": win, "flag": keep[0].get("flag"), "knobs": k, "selection_rows": rows,
              "rule": "CALIB-SELECTION: dCE<=+0.01 then best mean FSI rank (domain, e5 cluster); else lowest dCE",
              "latent_cfg": k["cfg"], "beta": k["beta"], "layers": UPPER if k["layers"] == "upper" else ALL,
              "drafts": json.loads((ART / "drafts.json").read_text()),
              "latent_meta_sha256": guard.sha256_file(LATENT / k["cfg"] / "meta.json"),
              "past_cfg": "PAST_" + "_".join(k["cfg"].split("_")[1:3]), "time": guard.now()}
    guard.atomic_write_json(ART / "frozen.json", frozen, mode=0o400)
    (ART / "frozen.sha256").write_text(guard.sha256_file(ART / "frozen.json") + "  frozen.json\n")
    print(json.dumps(frozen, indent=1)[:3000])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["t4", "stage1", "respond", "promote", "stage2", "stage3", "diag"])
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--stage", type=int)
    ap.add_argument("--cid")
    ap.add_argument("--tag")
    a = ap.parse_args()
    guard.check_root()
    SDIR.mkdir(parents=True, exist_ok=True)
    if a.cmd == "t4":
        do_t4()
    elif a.cmd == "stage1":
        lst = a.only or list(REGISTERED)
        p = SDIR / "stage1_list.json"
        cur = json.loads(p.read_text()) if p.exists() else []
        for c in lst:
            if c not in cur:
                cur.append(c)
        guard.atomic_write_json(p, cur, overwrite=True)
        for c in lst:
            stage(c, "s5")
    elif a.cmd == "respond":
        respond()
    elif a.cmd == "promote":
        promote(a.stage)
    elif a.cmd == "stage2":
        pr = json.loads((SDIR / "promotion_s1.json").read_text())["promoted"]
        guard.atomic_write_json(SDIR / "stage2_list.json", pr, overwrite=True)
        for c in pr:
            stage(c, "s20")
    elif a.cmd == "stage3":
        do_stage3()
    elif a.cmd == "diag":
        diagnose(a.cid, a.tag)


if __name__ == "__main__":
    main()
