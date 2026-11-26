#!/usr/bin/env python
"""csr-multiscale-routing-v1 final analysis (CPU; docs/MULTISCALE_ROUTING_V1_PREREG.md §11-§12).

Primary: full-validation CE per run, per-condition seed tables, preregistered contrasts with the paired document
bootstrap (2,000 resamples, default_rng(0), one document draw shared by all seeds) and the practical-support rule.
Secondary (descriptive): learning curves, throughput / GPU time / VRAM / disk, validation CE by document position
(available broad history) and after teacher-semantic / fixed / random boundaries, router diagnostics (diag_*.json),
specialization (spec/spec_analysis.json), calibration / feasibility / pilot summaries.
Writes artifacts/multiscale-routing-v1/analysis.json and tables/*.csv (never overwritten).
"""
import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402

ART = guard.ROOT / "artifacts" / "multiscale-routing-v1"
ST = guard.ROOT / ".tmp" / "multiscale_v1"
POS_BINS = [(0, 1), (1, 16), (16, 64), (64, 256), (256, 512), (512, 1024)]


def tsv(p):
    return {ln.split("\t")[0]: ln.split("\t")[1:] for ln in Path(p).read_text().splitlines() if ln.strip()}


def gpu_ledger():
    rows = []
    for ln in (guard.ROOT / "ledger" / "gpu_time_ledger.tsv").read_text().splitlines()[1:]:
        p = ln.split("\t")
        if len(p) >= 6 and (p[4].startswith("msv1") or "msv1" in p[3]):
            rows.append({"start": p[0], "end": p[1], "seconds": int(p[2]), "run_id": p[3], "purpose": p[4], "status": p[5]})
    return rows


def wcsv(path, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    path.chmod(0o400)


def canonical_val_tokens(cfg):
    """(gdoc, position) of every valid val prediction token in the evaluator's canonical packed order."""
    import numpy as np
    from csr.multiscale import load_split
    from csr.train_lib import BatchSource
    sp = load_split(guard.ROOT / cfg["data"]["dir"], "val")
    src = BatchSource(sp, cfg["data"]["seq_len"], None)
    L = sp.lengths()
    g, pos = [], []
    for row in src.rows:
        for di in row:
            n = int(L[di]) - 1
            g.append(np.full(n, di, np.int32))
            pos.append(np.arange(n, dtype=np.int32))
    return sp, np.concatenate(g), np.concatenate(pos)


def boundary_windows(sp, gdoc, pos, cfg):
    """Masks of tokens 0-15 and 16-63 positions after non-initial boundaries for semantic (non-forced: preceding chunk
    length < 256), fixed and random boundary sets (val split, frozen caches)."""
    import numpy as np
    from csr.multiscale import load_starts
    out = {}
    L = sp.lengths()
    for strat in ("semantic", "fixed", "random"):
        c = dict(cfg)
        c["segmentation"] = dict(cfg["segmentation"], strategy=strat)
        tab = load_starts(c, "val", guard.ROOT)
        bounds = {}
        for di in range(sp.n_docs):
            st = tab.starts(di)
            q = []
            for j in range(1, len(st)):
                if strat == "semantic" and st[j] - st[j - 1] >= 256:
                    continue  # forced close, not a semantic transition
                q.append(st[j])
            bounds[di] = np.asarray(q, np.int64)
        gd = gdoc.astype(np.int64)
        res = np.full(len(gdoc), -1, np.int64)  # distance from the most recent qualifying boundary (same document)
        starts_idx = np.concatenate([[0], np.nonzero(np.diff(gd))[0] + 1, [len(gd)]])
        for a, b in zip(starts_idx[:-1], starts_idx[1:]):
            bd = bounds[int(gd[a])]
            if len(bd) == 0:
                continue
            p = pos[a:b].astype(np.int64)
            k = np.searchsorted(bd, p, side="right") - 1
            ok = k >= 0
            r = np.full(b - a, -1, np.int64)
            r[ok] = p[ok] - bd[k[ok]]
            res[a:b] = r
        out[strat] = {"after_0_15": (res >= 0) & (res < 16), "after_16_63": (res >= 16) & (res < 64),
                      "n_boundaries": int(sum(len(v) for v in bounds.values()))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    out_p = ART / "analysis.json"
    if out_p.exists():
        raise guard.GuardError("analysis.json exists")
    print("msv1 final analysis (CPU)")
    if not guard.require_execution(a.execute, "cpu-model"):
        return
    import numpy as np
    from csr import ms_select
    from csr.multiscale import BRANCH_CONDITIONS, CONTEXTUAL, CONTRASTS
    guard.register_self("msv1 analyze")
    cal = json.loads((ART / "calibration.json").read_text())
    fea = json.loads((ART / "feasibility.json").read_text())
    pil = json.loads((ART / "pilot.json").read_text())
    conds = BRANCH_CONDITIONS[cal["branch"]]
    runs, evals = tsv(ST / "main_runs.tsv"), tsv(ST / "main_evals.tsv")
    led = gpu_ledger()
    led_by_run = {}
    for r in led:
        led_by_run.setdefault(r["run_id"], 0)
        led_by_run[r["run_id"]] += r["seconds"]
    per_run, docsum, doc_n_ref, tok_nll = {}, {}, None, {}
    for s in (0, 1, 2):
        for c in conds:
            key = f"seed{s}/{c}"
            rd = guard.ROOT / runs[key][0]
            ev = json.loads((guard.ROOT / evals[key][0]).read_text())
            sm = json.loads((rd / "summary.json").read_text())
            cfg = json.loads((rd / "config.json").read_text())
            met = [json.loads(x) for x in (rd / "metrics.jsonl").read_text().splitlines()]
            steps = [q for q in met if "lm_ce" in q]
            mon = [q for q in met if "monitor_eval" in q]
            dn = {int(k): int(v) for k, v in ev["doc_n"].items()}
            if doc_n_ref is None:
                doc_n_ref = dn
            assert dn == doc_n_ref, f"valid-token counts differ for {key}"
            docsum[key] = {int(k): float(v) for k, v in ev["doc_nll"].items()}
            tok_nll[key] = np.load(Path(guard.ROOT / evals[key][0]).with_name(ev["per_token_file"]))["nll_f16"].astype(np.float32)
            eid = Path(evals[key][0]).parent.name
            gpu_train = led_by_run.get(rd.name, 0)
            per_run[key] = {
                "seed": s, "condition": c, "run": runs[key][0], "eval": evals[key][0], "val_ce": ev["ce_nats_per_token"],
                "val_ppl": ev["ppl"], "n_valid_tokens": ev["n_valid_tokens"], "train_tokens": sm["valid_train_tokens"],
                "steps": sm["steps"], "loop_seconds": sm["loop_seconds"],
                "tokens_per_s": sm["train_valid_tokens_per_s_after_warmup"], "peak_alloc_gib": sm["peak_alloc_gib"],
                "peak_reserved_gib": sm["peak_reserved_gib"], "gpu_seconds_train_and_eval_ledger": gpu_train,
                "checkpoint_bytes": (rd / "model.pt").stat().st_size, "checkpoint_sha256": sm["checkpoint_sha256"],
                "val_expert_max_share": ev["expert_max_share"], "val_experts_unused": ev["experts_unused"],
                "final20_train_slope": ms_select.final20_slope(steps, sm["steps"]),
                "final_train_ce_last5_logged": float(np.mean([q["lm_ce"] for q in steps[-5:]])),
                "router_lm_gradnorm_mean_final20": [float(np.mean([q["router_lm_gradnorm"][li] for q in steps
                                                                   if "router_lm_gradnorm" in q and q["step"] >= 0.8 * sm["steps"]]))
                                                    for li in range(7)],
                "curve_train": [(q["valid_tokens_seen"], q["lm_ce"]) for q in steps],
                "curve_monitor": [(q["valid_tokens_seen"], q["monitor_eval"]["calib"]["ce_nats_per_token"],
                                   q["monitor_eval"].get("val", {}).get("ce_nats_per_token")) for q in mon]}
            per_run[key]["loss_per_gpu_hour_note"] = "val CE at fixed token budget; GPU-hours in gpu_seconds field"
    docs = sorted(doc_n_ref)
    N = np.array([doc_n_ref[d] for d in docs], float)
    D = {k: np.array([docsum[k][d] for d in docs]) for k in docsum}
    # per condition
    per_cond = {}
    for c in conds:
        v = [per_run[f"seed{s}/{c}"]["val_ce"] for s in (0, 1, 2)]
        per_cond[c] = {"val_ce_by_seed": v, "mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)),
                       "range": [float(min(v)), float(max(v))], "ppl_mean": float(np.mean([math.exp(x) for x in v])),
                       "tokens_per_s_mean": float(np.mean([per_run[f"seed{s}/{c}"]["tokens_per_s"] for s in (0, 1, 2)])),
                       "peak_reserved_gib_max": float(max(per_run[f"seed{s}/{c}"]["peak_reserved_gib"] for s in (0, 1, 2)))}
    # contrasts
    con = {}
    for cid, row, col in CONTRASTS:
        if row in conds and col in conds:
            res = ms_select.paired_bootstrap_multi({s: D[f"seed{s}/{row}"] for s in (0, 1, 2)},
                                                   {s: D[f"seed{s}/{col}"] for s in (0, 1, 2)}, N, n_boot=2000, seed=0)
            res.update(id=cid, row=row, col=col)
            con[f"{cid}: {row} - {col}"] = res
            print(cid, row, col, round(res["mean_diff"], 5), [round(x, 5) for x in res["mean_ci95"]], res["support"], flush=True)
    # per-token secondary breakdowns
    any_cfg = json.loads((guard.ROOT / runs[f"seed0/{conds[0]}"][0] / "config.json").read_text())
    sp, gdoc, pos = canonical_val_tokens(any_cfg)
    assert all(len(v) == len(gdoc) for v in tok_nll.values()), "per-token order length mismatch"
    for k, v in tok_nll.items():  # canonical order check: per-document sums agree with float64 doc sums
        sums = np.bincount(gdoc, weights=v.astype(np.float64), minlength=sp.n_docs)
        assert abs(sums.sum() - sum(docsum[k].values())) / sum(docsum[k].values()) < 1e-3
    bw = boundary_windows(sp, gdoc, pos, any_cfg)
    pos_tab, trans_tab = {}, {}
    for c in conds:
        pos_tab[c], trans_tab[c] = {}, {}
        for lo, hi in POS_BINS:
            msk = (pos >= lo) & (pos < hi)
            pos_tab[c][f"{lo}-{hi - 1}"] = float(np.mean([tok_nll[f"seed{s}/{c}"][msk].mean() for s in (0, 1, 2)]))
        for strat, mm in bw.items():
            for w in ("after_0_15", "after_16_63"):
                trans_tab[c][f"{strat}/{w}"] = float(np.mean([tok_nll[f"seed{s}/{c}"][mm[w]].mean() for s in (0, 1, 2)]))
    rel = lambda tab: {c: {k: v - tab["TOKEN"][k] for k, v in tab[c].items()} for c in conds} if "TOKEN" in tab else {}
    pos_counts = {f"{lo}-{hi - 1}": int(((pos >= lo) & (pos < hi)).sum()) for lo, hi in POS_BINS}
    # secondary artifacts
    diag = sorted(ART.glob("diag_*.json"))
    diag = json.loads(diag[-1].read_text()) if diag else None
    specp = ART / "spec" / "spec_analysis.json"
    spec = json.loads(specp.read_text()) if specp.exists() else None
    spec_sum = None
    if spec:
        spec_sum = {}
        for c in conds:
            ks = [f"seed{s}/{c}" for s in (0, 1, 2) if f"seed{s}/{c}" in spec["per_checkpoint"]]
            if len(ks) < 3:
                continue
            g = lambda f: [f(spec["per_checkpoint"][k]) for k in ks]
            spec_sum[c] = {"R_enc32_by_seed": g(lambda p: float(np.mean(p["routing"]["enc_K32"]["mi_adj"]))),
                           "R_tfidf32_by_seed": g(lambda p: float(np.mean(p["routing"]["tfidf_K32"]["mi_adj"]))),
                           "probe_macro_f1_by_seed": g(lambda p: p["probe"]["enc_K32"]["macro_f1"]),
                           "F_enc32_by_seed": g(lambda p: p["functional"]["enc_K32"]["F_primary"]),
                           "F_tfidf32_by_seed": g(lambda p: p["functional"]["tfidf_K32"]["F_primary"]),
                           "null_F_p975_by_seed": g(lambda p: p["functional"]["enc_K32"]["null_F_primary_p975"]),
                           "cross_seed": spec["cross_seed"].get(c)}
    # compute
    wall = None
    if led:
        from datetime import datetime
        wall = (datetime.fromisoformat(led[-1]["end"]) - datetime.fromisoformat(led[0]["start"])).total_seconds()
    by_purpose = {}
    for r in led:
        k = r["purpose"].split(" ")[1] if " " in r["purpose"] else r["purpose"]
        by_purpose[k] = by_purpose.get(k, 0) + r["seconds"]
    out = {"experiment": "csr-multiscale-routing-v1", "created": guard.now(),
           "prereg_sha256": guard.sha256_file(guard.ROOT / "docs/MULTISCALE_ROUTING_V1_PREREG.md"),
           "alpha": cal["alpha"], "beta": cal["beta"], "branch": cal["branch"], "budget": fea["budget"],
           "conditions": conds, "per_run": per_run, "per_condition": per_cond, "contrasts": con,
           "contextual_conditions": sorted(set(conds) & CONTEXTUAL),
           "val_ce_by_position_bin_seedmean": pos_tab, "val_ce_by_position_minus_token": rel(pos_tab),
           "position_bin_token_counts": pos_counts,
           "val_ce_after_boundaries_seedmean": trans_tab, "val_ce_after_boundaries_minus_token": rel(trans_tab),
           "boundary_counts_val": {k: v["n_boundaries"] for k, v in bw.items()},
           "boundary_window_token_counts": {f"{k}/{w}": int(v[w].sum()) for k, v in bw.items() for w in ("after_0_15", "after_16_63")},
           "router_diagnostics": diag, "specialization_summary": spec_sum,
           "calibration": {k: cal[k] for k in ("alpha", "beta", "branch")}, "pilot_futility_stop": pil["futility_stop"],
           "gpu_seconds_total_msv1": sum(r["seconds"] for r in led), "gpu_seconds_by_purpose": by_purpose,
           "gpu_wall_span_seconds": wall, "disk_free_bytes_now": guard.disk_free_bytes(),
           "notes": ["document-bootstrap intervals are not seed uncertainty",
                     "boundary-window CE is descriptive; semantic boundaries are teacher-defined",
                     "position bins: position in document = available broad history length"]}
    guard.atomic_write_json(out_p, out, mode=0o400)
    tdir = ART / "tables"
    tdir.mkdir(exist_ok=True)
    wcsv(tdir / "main_runs.csv", [{k: v for k, v in r.items() if not k.startswith("curve")} | {"key": k_}
                                  for k_, r in per_run.items()])
    wcsv(tdir / "per_condition.csv", [{"condition": c, **{k: v for k, v in r.items()}} for c, r in per_cond.items()])
    wcsv(tdir / "contrasts.csv", [{"contrast": k, "mean_diff": v["mean_diff"], "mean_ci95": v["mean_ci95"],
                                   **{f"seed{s}_diff": v["per_seed"][str(s)]["diff"] for s in (0, 1, 2)},
                                   **{f"seed{s}_ci95": v["per_seed"][str(s)]["ci95"] for s in (0, 1, 2)},
                                   "seed_sd": v["seed_sd"], "support": v["support"], "non_inferior": v["non_inferior"]}
                                  for k, v in con.items()])
    guard.ledger("MS_V1_ANALYSIS", f"analysis.json sha256={guard.sha256_file(out_p)}")
    print(json.dumps({"per_condition": per_cond, "contrasts": {k: (v["mean_diff"], v["mean_ci95"], v["support"]) for k, v in con.items()}},
                     indent=1, default=str))


if __name__ == "__main__":
    main()
