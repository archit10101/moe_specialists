#!/usr/bin/env python
"""csr-sequence-routing-search-v1: final analysis (CPU). Reads final runs, their single validation evaluations,
CALIB results, latency and ledgers; writes artifacts/sequence-routing-search-v1/final_analysis.json (+ per-document
validation CE file). Dry-run by default (needs --execute and MOE_RESEARCH_APPROVE=cpu-model)."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402

ST = guard.ROOT / ".tmp" / "sequence_routing_v1"


def tsv(name):
    p = ST / name
    return {ln.split("\t")[0]: ln.split("\t")[1:] for ln in p.read_text().splitlines() if ln.strip()} if p.exists() else {}


def jread(p):
    return json.loads(Path(p).read_text())


def curves(rd):
    tr, mon = [], []
    for ln in (guard.ROOT / rd / "metrics.jsonl").read_text().splitlines():
        r = json.loads(ln)
        if "lm_ce" in r:
            tr.append([r["valid_tokens_seen"], r["lm_ce"]])
        if "monitor_eval" in r:
            mon.append([r["valid_tokens_seen"], r["monitor_eval"]["calib_search_head"]["ce_nats_per_token"]])
    return {"train_lm_ce": tr, "calib_search_head_ce": mon}


def gpu_ledger():
    rows = []
    for ln in (guard.ROOT / "ledger" / "gpu_time_ledger.tsv").read_text().splitlines()[1:]:
        p = ln.split("\t")
        if len(p) >= 6 and p[4].startswith("srsv1"):
            rows.append({"start": p[0], "end": p[1], "seconds": int(p[2]), "run": p[3], "purpose": p[4],
                         "status": p[5]})
    by = {}
    for r in rows:
        k = r["purpose"] + (" " + r["run"].split("_srsv1_train_")[-1].split("_s")[0] if "train" in r["purpose"] else "")
        by[k] = by.get(k, 0) + r["seconds"]
    return rows, by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    if not guard.require_execution(a.execute, "cpu-model"):
        return
    from csr import srs_select as sel
    from csr.srs_run import ART, FINAL_FROZEN
    ff = jread(FINAL_FROZEN)
    ids = ff["ids"]
    fam_of = {s["id"]: s["family"] for s in jread(ART / "final_specs.json")["specs"]}
    fam_of["TOKEN-4"] = "token"
    runs, evals = tsv("final_runs.tsv"), tsv("val_evals.tsv")
    per = {i: {} for i in ids}
    docs = {i: [] for i in ids}
    for s in (0, 1, 2):
        for i in ids:
            key = f"seed{s}/{i}"
            rd = runs[key][0]
            ev = jread(guard.ROOT / evals[key][0])
            assert ev["status"] == "COMPLETE" and ev["evalset"] == "val"
            summ = jread(guard.ROOT / rd / "summary.json")
            facts = jread(guard.ROOT / rd / "run_facts.json")
            per[i][s] = {"run": rd, "val_eval": evals[key][0], "val_ce": ev["ce_nats_per_token"], "val_ppl": ev["ppl"],
                         "n_valid_tokens": ev["n_valid_tokens"], "n_docs": len(ev["doc_n"]),
                         "collapse": ev["collapse"], "experts_unused": ev["experts_unused"],
                         "expert_max_share": ev["expert_max_share"], "expert_load_cv": ev["expert_load_cv"],
                         "usage_entropy": ev["usage_entropy"], "routing_entropy_mean": ev["routing_entropy_mean"],
                         "route_changes_per_1k_tokens": ev["route_changes_per_1k_tokens"],
                         "consecutive_route_jaccard": ev["consecutive_route_jaccard"],
                         "router_invocations_per_1k_tokens": ev["router_invocations_per_1k_tokens"],
                         "dispatch_max_over_mean": ev["dispatch_max_over_mean"],
                         "ce_by_doc_position": ev["ce_by_doc_position"],
                         "val_tokens_per_s": ev["eval_valid_tokens_per_s"],
                         "train_tokens_per_s": summ["train_valid_tokens_per_s_after_warmup"],
                         "train_process_seconds": summ["process_seconds"], "valid_train_tokens": summ["valid_train_tokens"],
                         "train_flops": summ["train_flops"], "peak_reserved_gib": summ["peak_reserved_gib"],
                         "extra_router_params": facts["extra_router_params"],
                         "extra_router_param_fraction": facts["extra_router_param_fraction"],
                         "params": facts["params"], "init_backbone_param_sha256": facts["init_backbone_param_sha256"],
                         "calib_search_ce_end_of_training": summ.get("calib_search_ce"), "curves": curves(rd)}
            for k in ("segments", "mean_segment_length", "mean_closed_segment_length", "segment_length_quantiles",
                      "boundaries_per_1k_tokens", "bos_route_token_fraction", "ce_by_segment_position_layer1"):
                if k in ev:
                    per[i][s][k] = ev[k]
            docs[i].append((ev["doc_nll"], ev["doc_n"]))
    # integrity
    for s in (0, 1, 2):
        toks = {per[i][s]["valid_train_tokens"] for i in ids}
        inits = {per[i][s]["init_backbone_param_sha256"] for i in ids}
        vt = {per[i][s]["n_valid_tokens"] for i in ids}
        assert len(toks) == 1 and len(inits) == 1 and len(vt) == 1, f"seed {s}: mismatch {toks} {inits} {vt}"
    tok_ce = [per["TOKEN-4"][s]["val_ce"] for s in (0, 1, 2)]
    contrasts = {}
    for i in ids[1:]:
        diffs = [per[i][s]["val_ce"] - per["TOKEN-4"][s]["val_ce"] for s in (0, 1, 2)]
        boot = sel.paired_bootstrap_multi(docs["TOKEN-4"], docs[i])
        per_seed_boot = [sel.paired_bootstrap_multi([docs["TOKEN-4"][k]], [docs[i][k]]) for k in range(3)]
        collapsed = any(per[i][s]["collapse"]["collapsed"] for s in (0, 1, 2))
        fl = np.mean([per[i][s]["train_flops"] for s in (0, 1, 2)]) / np.mean(
            [per["TOKEN-4"][s]["train_flops"] for s in (0, 1, 2)]) - 1
        gt = np.mean([per[i][s]["train_process_seconds"] for s in (0, 1, 2)]) / np.mean(
            [per["TOKEN-4"][s]["train_process_seconds"] for s in (0, 1, 2)]) - 1
        crit = sel.criteria(diffs, boot, collapsed, fl)
        # per-document distribution of the seed-averaged CE difference
        d0 = sorted(set(map(str, docs["TOKEN-4"][0][0])))
        dd = np.array([np.mean([docs[i][k][0][d] - docs["TOKEN-4"][k][0][d] for k in range(3)]) / docs[i][0][1][d]
                       for d in d0])
        contrasts[f"{i} - TOKEN-4"] = {
            "family": fam_of[i], "criteria": crit, "paired_doc_bootstrap_seed_averaged": boot,
            "paired_doc_bootstrap_per_seed": per_seed_boot, "extra_training_flops_fraction": float(fl),
            "extra_training_gpu_time_fraction": float(gt), "compute_matched": bool(fl <= sel.MAX_EXTRA_COMPUTE),
            "doc_level_ce_diff_quantiles": dict(zip(["p05", "p25", "p50", "p75", "p95"],
                                                    np.quantile(dd, [0.05, 0.25, 0.5, 0.75, 0.95]).tolist())),
            "fraction_docs_better": float((dd < 0).mean()),
            "ce_by_doc_position_diff": {b: float(np.mean([per[i][s]["ce_by_doc_position"][b]
                                                          - per["TOKEN-4"][s]["ce_by_doc_position"][b]
                                                          for s in (0, 1, 2)]))
                                        for b in per["TOKEN-4"][0]["ce_by_doc_position"]},
        }
    if len(ids) == 3:
        a_, b_ = ids[1], ids[2]
        diffs = [per[b_][s]["val_ce"] - per[a_][s]["val_ce"] for s in (0, 1, 2)]
        contrasts[f"{b_} - {a_} (dynamic - fixed)" if fam_of[b_] == "dynamic" else f"{b_} - {a_}"] = {
            "per_seed": diffs, "mean": float(np.mean(diffs)), "sd": float(np.std(diffs, ddof=1)),
            "paired_doc_bootstrap_seed_averaged": sel.paired_bootstrap_multi(docs[a_], docs[b_])}
    # per-document validation CE file (seed-averaged)
    d0 = sorted(set(map(str, docs["TOKEN-4"][0][0])))
    doc_file = {"docs": d0, "n_tokens": [docs["TOKEN-4"][0][1][d] for d in d0],
                "ce_seed_mean": {i: [float(np.mean([docs[i][k][0][d] for k in range(3)]) / docs[i][0][1][d])
                                     for d in d0] for i in ids}}
    dpath = ART / "val_doc_ce.json"
    if not dpath.exists():
        guard.atomic_write_json(dpath, doc_file, mode=0o400)
    lat = sorted([p for p in ART.glob("latency_*.json") if jread(p).get("status") == "COMPLETE"])
    rows, by = gpu_ledger()
    out = {
        "experiment": "csr-sequence-routing-search-v1", "created": guard.now(), "budget": ff["budget"],
        "final_ids": ids, "families": fam_of, "per_run": per, "token_val_ce": tok_ce, "contrasts": contrasts,
        "summary_val_ce": {i: {"mean": float(np.mean([per[i][s]["val_ce"] for s in (0, 1, 2)])),
                               "sd": float(np.std([per[i][s]["val_ce"] for s in (0, 1, 2)], ddof=1)),
                               "ppl_mean": float(np.mean([per[i][s]["val_ppl"] for s in (0, 1, 2)]))} for i in ids},
        "latency": jread(lat[-1]) if lat else None,
        "gpu_time": {"total_s": sum(r["seconds"] for r in rows), "by_purpose": by, "n_entries": len(rows)},
        "calib_selection": jread(ART / "calib_selection_results.json"),
        "search_frozen": jread(ART / "search_frozen.json"),
        "budget_gate": jread(ART / "budget.json"),
        "val_doc_ce_file": str(dpath.relative_to(guard.ROOT)),
        "disk_free_bytes": guard.disk_free_bytes(),
    }
    p = ART / "final_analysis.json"
    guard.atomic_write_json(p, out, mode=0o400)
    guard.ledger("SRS_V1_ANALYSIS", f"{p.relative_to(guard.ROOT)} sha256={guard.sha256_file(p)}")
    print(json.dumps({k: v["criteria"] if "criteria" in v else v for k, v in contrasts.items()}, indent=1, default=str))


if __name__ == "__main__":
    main()
