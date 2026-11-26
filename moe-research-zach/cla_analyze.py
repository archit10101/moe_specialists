#!/usr/bin/env python
"""csr-conditional-local-attention-v1 final analysis (CPU; reads recorded results only, never data splits).

Writes artifacts/conditional-local-attention-v1/final_analysis.json and .tmp/conditional_local_attention_v1/
report_tables.md. Accuracy: validation CE per seed, paired deltas, paired per-document bootstrap (2,000 resamples,
seed 20261108; per seed and seed-averaged), verdict per prereg section 9. Speed: the four preregistered primary
measures (matched-batch training throughput; end-to-end 100M-token training GPU time; prefill latency and cached
decoding throughput at batch 8 / prompt 1024), each against the FASTEST measured baseline variant; full grid secondary.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/zapan4/moe-research")
sys.path.insert(0, str(ROOT / "src"))
from csr import guard  # noqa: E402
from csr import cla_select as sel  # noqa: E402
from csr.cla_run import ART, FINAL_FROZEN, FINAL_TRAINED, PARTITION, PREREG, SEARCH_FROZEN, ST_DIR  # noqa: E402

BOOT, BOOT_SEED = 2000, 20261108


def jr(p):
    return json.loads(Path(p).read_text())


def tsv(name):
    p = ST_DIR / name
    return {ln.split("\t")[0]: ln.split("\t")[1:] for ln in p.read_text().splitlines() if ln.strip()} if p.exists() else {}


def bootstrap(doc_c: list, doc_b: list, n: list):
    """doc_c / doc_b: per-seed dicts doc -> NLL sum; n: per-seed dicts doc -> token count (identical). Returns the
    seed-averaged paired token-weighted delta and its document-bootstrap CI, plus per-seed CIs."""
    docs = sorted(set(doc_b[0]))
    for k in range(len(doc_b)):
        assert set(doc_c[k]) == set(doc_b[k]) == set(docs)
        assert all(n[k][d] == n[0][d] for d in docs)
    N = np.array([n[0][d] for d in docs], float)
    D = np.array([[doc_c[k][d] - doc_b[k][d] for d in docs] for k in range(len(doc_b))])  # seeds x docs
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(docs), (BOOT, len(docs)))
    out = {"n_docs": len(docs), "n_tokens": int(N.sum())}
    dm = D.mean(0)
    bs = np.array([dm[i].sum() / N[i].sum() for i in idx])
    out["seed_mean"] = {"delta": float(dm.sum() / N.sum()), "ci95": [float(x) for x in np.percentile(bs, [2.5, 97.5])],
                        "frac_docs_improved": float((dm < 0).mean())}
    out["per_seed"] = []
    for k in range(D.shape[0]):
        bk = np.array([D[k][i].sum() / N[i].sum() for i in idx])
        out["per_seed"].append({"delta": float(D[k].sum() / N.sum()),
                                "ci95": [float(x) for x in np.percentile(bk, [2.5, 97.5])],
                                "frac_docs_improved": float((D[k] < 0).mean())})
    return out


def flops_model(cfg_model, keys_per_layer, plan):
    """Analytic forward FLOPs per token (multiply-add = 2): attention (projections + QK + AV, measured keys) + gate
    (conditional) + dense FFN layer 0 + 4 active experts + shared expert + routers + tied LM head."""
    from csr.cla_model import AttnPlan
    d, f, fs, E = cfg_model["d_model"], cfg_model["expert_ffn"], cfg_model["shared_ffn"], cfg_model["n_experts"]
    n_moe = len(cfg_model["moe_layers"])
    p = AttnPlan.from_cfg(plan)
    attn = []
    for (mode, _), kp in zip(p.layer_modes(cfg_model["n_layer"]), keys_per_layer):
        attn.append(2 * 4 * d * d + 4 * d * kp + (2 * (d * p.gate_hidden + 2 * p.gate_hidden) if mode == "cond" else 0))
    other = (cfg_model["n_layer"] - n_moe) * 2 * 3 * d * cfg_model["dense_ffn"] \
        + n_moe * (cfg_model["top_k"] * 2 * 3 * d * f + 2 * 3 * d * fs + 2 * d * E) + 2 * d * cfg_model["vocab_size"]
    return {"attention_by_layer": attn, "attention_total": float(sum(attn)),
            "attention_score_part": float(sum(4 * d * kp for kp in keys_per_layer)),
            "non_attention": float(other), "total": float(sum(attn) + other)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    if not guard.require_execution(a.execute, "cpu-model"):
        return
    out = {"experiment": "csr-conditional-local-attention-v1", "created": guard.now(),
           "prereg_sha256": guard.sha256_file(PREREG), "partition_sha256": guard.sha256_file(PARTITION)}
    man = jr(ART / "manifest.json")
    out["prereg_unchanged"] = out["prereg_sha256"] == man["prereg_sha256"]
    g1 = jr(ART / "gen1_results.json")
    out["gen1"] = g1
    base1 = g1["results"][sel.BASELINE]
    out["gen1_table"] = {i: {"ce": r["ce"], "delta": (r["ce"] - base1["ce"]) if r["ce"] is not None else None,
                             "train_tok_s": r.get("train_tok_s"),
                             "throughput_ratio": (r["train_tok_s"] / base1["train_tok_s"]) if r.get("train_tok_s") else None,
                             "gpu_s": r.get("gpu_s"), "global_query_rate": r.get("global_query_rate"),
                             "attention_work_fraction_vs_full": r.get("attention_work_fraction_vs_full"),
                             "peak_reserved_gib": r.get("peak_reserved_gib"), "collapsed": r.get("collapsed"),
                             "eliminated": g1["elimination"].get(i, [False, ""]),
                             "advanced": i in g1["advance"]} for i, r in g1["results"].items()}
    if (ART / "gen2_results.json").exists():
        out["gen2"] = jr(ART / "gen2_results.json")
    if (ART / "calib_selection_results.json").exists():
        out["calib_selection"] = jr(ART / "calib_selection_results.json")
    out["search_frozen"] = jr(SEARCH_FROZEN) if SEARCH_FROZEN.exists() else None
    verdict_inputs = {"valid": True, "comparable": False, "speed": False}
    if FINAL_TRAINED.exists():
        ft = jr(FINAL_TRAINED)
        ff = jr(FINAL_FROZEN)
        w = ff["winner"]
        out["winner"] = w
        out["winner_plan"] = sel.PLANS[w]
        evals = tsv("evals.tsv")
        val = {}
        for i in (sel.BASELINE, w):
            for s in (0, 1, 2):
                e = jr(ROOT / evals[f"val/final/seed{s}/{i}"][0])
                assert e["status"] == "COMPLETE" and e["evalset"] == "val"
                val[(i, s)] = e
        ce = {i: {s: val[(i, s)]["ce_nats_per_token"] for s in (0, 1, 2)} for i in (sel.BASELINE, w)}
        deltas = {s: ce[w][s] - ce[sel.BASELINE][s] for s in (0, 1, 2)}
        acc = sel.accuracy_verdict(deltas)
        bs = bootstrap([val[(w, s)]["doc_nll"] for s in (0, 1, 2)], [val[(sel.BASELINE, s)]["doc_nll"] for s in (0, 1, 2)],
                       [val[(sel.BASELINE, s)]["doc_n"] for s in (0, 1, 2)])
        out["final_validation"] = {
            "ce": {i: {str(s): v for s, v in d.items()} for i, d in ce.items()},
            "ce_mean": {i: float(np.mean(list(d.values()))) for i, d in ce.items()},
            "delta_per_seed": {str(s): v for s, v in deltas.items()}, "accuracy_verdict": acc, "bootstrap": bs,
            "n_valid_tokens": {i: [val[(i, s)]["n_valid_tokens"] for s in (0, 1, 2)] for i in (sel.BASELINE, w)},
            "collapse": {i: [val[(i, s)]["collapse"] for s in (0, 1, 2)] for i in (sel.BASELINE, w)},
            "ce_by_doc_position": {i: {str(s): val[(i, s)]["ce_by_doc_position"] for s in (0, 1, 2)}
                                   for i in (sel.BASELINE, w)}}
        verdict_inputs["comparable"] = acc["comparable"]
        # gate / attention diagnostics on validation
        diag = {}
        for i in (sel.BASELINE, w):
            e0 = [val[(i, s)] for s in (0, 1, 2)]
            diag[i] = {"keys_attended_per_token_by_layer": np.mean([e["keys_attended_per_token_by_layer"] for e in e0],
                                                                   0).tolist(),
                       "attention_work_fraction_vs_full": float(np.mean([e["attention_work_fraction_vs_full"]
                                                                         for e in e0])),
                       "layer_modes": e0[0]["layer_modes"], "thresholds": [e["thresholds"] for e in e0]}
            if e0[0].get("global_query_rate_by_layer"):
                diag[i]["global_query_rate_by_layer"] = {str(s): val[(i, s)]["global_query_rate_by_layer"] for s in (0, 1, 2)}
                diag[i]["global_query_rate_mean"] = {str(s): val[(i, s)]["global_query_rate_mean"] for s in (0, 1, 2)}
                diag[i]["global_rate_among_eligible_by_layer"] = {str(s): val[(i, s)]["global_rate_among_eligible_by_layer"]
                                                                  for s in (0, 1, 2)}
                diag[i]["global_query_rate_by_layer_and_position"] = {
                    str(s): val[(i, s)]["global_query_rate_by_layer_and_position"] for s in (0, 1, 2)}
                diag[i]["eligible_fraction"] = val[(i, 0)]["eligible_fraction"]
                diag[i]["token_analysis"] = token_analysis([val[(i, s)] for s in (0, 1, 2)])
        out["validation_diagnostics"] = diag
        cfgm = jr(ROOT / ft["runs"][f"seed0/{sel.BASELINE}"] / "config.json")["model"]
        out["theoretical_flops_per_token"] = {
            i: flops_model(cfgm, diag[i]["keys_attended_per_token_by_layer"], sel.PLANS[i]) for i in (sel.BASELINE, w)}
        fb, fw = out["theoretical_flops_per_token"][sel.BASELINE], out["theoretical_flops_per_token"][w]
        out["theoretical_savings"] = {"attention_flops_ratio": fw["attention_total"] / fb["attention_total"],
                                      "total_forward_flops_ratio": fw["total"] / fb["total"],
                                      "attention_share_of_baseline_flops": fb["attention_total"] / fb["total"]}
        # fairness: matched init, data, tokens, steps; parameters
        fair = {}
        for s in (0, 1, 2):
            rb, rw = ft["runs"][f"seed{s}/{sel.BASELINE}"], ft["runs"][f"seed{s}/{w}"]
            fb_, fw_ = jr(ROOT / rb / "run_facts.json"), jr(ROOT / rw / "run_facts.json")
            sb, sw = jr(ROOT / rb / "summary.json"), jr(ROOT / rw / "summary.json")
            fair[str(s)] = {k: fb_[k] == fw_[k] for k in ("init_param_sha256", "train_rows_sha256", "n_steps",
                                                         "planned_valid_tokens", "micro_batch_rows", "accum")}
            fair[str(s)].update({"valid_train_tokens": [sb["valid_train_tokens"], sw["valid_train_tokens"]],
                                 "optimizer_steps": [sb["optimizer_steps"], sw["optimizer_steps"]],
                                 "train_step_seconds": [sb["train_step_seconds"], sw["train_step_seconds"]],
                                 "process_seconds": [sb["process_seconds"], sw["process_seconds"]],
                                 "n_cf_steps": sw.get("n_cf_steps"), "params": [fb_["params"], fw_["params"]],
                                 "peak_reserved_gib": [sb["peak_reserved_gib"], sw["peak_reserved_gib"]]})
        out["fairness"] = fair
        out["fairness_all_matched"] = all(all(v for k, v in d.items() if isinstance(v, bool)) for d in fair.values())
        verdict_inputs["valid"] = out["fairness_all_matched"] and out["prereg_unchanged"]
        e2e = {i: [jr(ROOT / ft["runs"][f"seed{s}/{i}"] / "summary.json")["train_step_seconds"] for s in (0, 1, 2)]
               for i in (sel.BASELINE, w)}
        e2e_p = {i: [jr(ROOT / ft["runs"][f"seed{s}/{i}"] / "summary.json")["process_seconds"] for s in (0, 1, 2)]
                 for i in (sel.BASELINE, w)}
        out["end_to_end_100M"] = {"train_step_seconds": e2e, "process_seconds": e2e_p}
        # speed
        benches = [p for p in sorted(ART.glob("bench_2*.json")) if jr(p).get("status") == "COMPLETE"]
        if benches:
            b = jr(benches[-1])
            out["bench_file"] = str(benches[-1].relative_to(ROOT))
            out["bench"] = {k: b[k] for k in ("train_matched", "train_matched_repeat", "train_max_batch", "prefill",
                                              "decode") if k in b}
            tm = b["train_matched"]
            base_variants = [k for k in tm if k.startswith(sel.BASELINE)]
            best_base_tps = max(tm[k]["tokens_per_s"] for k in base_variants)
            pre = b["prefill"]["B8_L1024"]
            best_base_pre = min(v["median_latency_ms"] for k, v in pre.items() if k.startswith(sel.BASELINE))
            dec = b["decode"]["B8_L1024"]
            best_base_dec = max(v["tokens_per_s"] for k, v in dec.items() if k.startswith(sel.BASELINE))
            primary = {
                "matched_batch_training": {"time_ratio": best_base_tps / tm[w]["tokens_per_s"],
                                           "baseline_best_tokens_per_s": best_base_tps,
                                           "winner_tokens_per_s": tm[w]["tokens_per_s"]},
                "end_to_end_100M_training_gpu_time": {
                    "time_ratio": float(np.mean(e2e_p[w]) / np.mean(e2e_p[sel.BASELINE])),
                    "baseline_mean_s": float(np.mean(e2e_p[sel.BASELINE])), "winner_mean_s": float(np.mean(e2e_p[w]))},
                "prefill_latency_B8_L1024": {"time_ratio": pre[w]["median_latency_ms"] / best_base_pre,
                                             "baseline_best_ms": best_base_pre, "winner_ms": pre[w]["median_latency_ms"]},
                "decode_throughput_B8_L1024": {"time_ratio": best_base_dec / dec[w]["tokens_per_s"],
                                               "baseline_best_tokens_per_s": best_base_dec,
                                               "winner_tokens_per_s": dec[w]["tokens_per_s"]}}
            for k, v in primary.items():
                v["qualifies_10pct"] = sel.speed_qualifies(v["time_ratio"])
            out["speed_primary"] = primary
            verdict_inputs["speed"] = any(v["qualifies_10pct"] for v in primary.values())
            grid = {}
            for key in b["prefill"]:
                p_ = b["prefill"][key]
                d_ = b["decode"][key]
                bp = min(v["median_latency_ms"] for k, v in p_.items() if k.startswith(sel.BASELINE))
                bd = max(v["tokens_per_s"] for k, v in d_.items() if k.startswith(sel.BASELINE))
                grid[key] = {"prefill_time_ratio": p_[w]["median_latency_ms"] / bp,
                             "decode_time_ratio": bd / d_[w]["tokens_per_s"]}
            out["speed_grid_secondary"] = grid
        else:
            out["speed_primary"] = None
            verdict_inputs["valid"] = verdict_inputs["valid"]  # speed missing -> no speed claim
    else:
        out["winner"] = None
    out["verdict_inputs"] = verdict_inputs
    out["verdict"] = sel.final_verdict(verdict_inputs["valid"], verdict_inputs["comparable"], verdict_inputs["speed"])
    devs = ART / "deviations.jsonl"
    out["deviations"] = [json.loads(x) for x in devs.read_text().splitlines()] if devs.exists() else []
    gl = [ln.split("\t") for ln in (ROOT / "ledger/gpu_time_ledger.tsv").read_text().splitlines()]
    out["gpu_seconds"] = sum(int(x[2]) for x in gl if len(x) >= 5 and x[4].startswith("clav1") and x[2].isdigit())
    out["gpu_ledger_rows"] = [x for x in gl if len(x) >= 5 and x[4].startswith("clav1")]
    out["test_split_access"] = "never: load_raw_split refuses 'test' before any file access; inventories exclude " \
                               "test files by name; audited training subprocesses opened no val_/test_ file"
    guard.atomic_write_json(ART / "final_analysis.json", out, overwrite=True)
    guard.ledger("CLA_V1_ANALYSIS", f"final_analysis.json verdict={out['verdict']}")
    print(json.dumps({"verdict": out["verdict"], "inputs": verdict_inputs, "winner": out.get("winner")}))


def token_analysis(evs, min_count=300, top=25):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(ROOT / "artifacts/tokenizer/pythia-12b/tokenizer.json"))
    res = {}
    for field in ("token_request_counts", "prev_token_request_counts"):
        agg = {}
        for e in evs:
            for t, (n, k) in e.get(field, {}).items():
                a = agg.setdefault(int(t), [0, 0])
                a[0] += n
                a[1] += k
        tot_n = sum(v[0] for v in agg.values())
        tot_k = sum(v[1] for v in agg.values())
        rows = [(t, n, k, k / n) for t, (n, k) in agg.items() if n >= min_count]
        rows.sort(key=lambda r: -r[3])
        fmt = lambda r: {"token_id": r[0], "text": tok.decode([r[0]]), "n_eligible_layer_queries": r[1],  # noqa
                         "request_rate": r[3]}
        res[field] = {"overall_rate_among_eligible": tot_k / max(tot_n, 1), "highest": [fmt(r) for r in rows[:top]],
                      "lowest": [fmt(r) for r in rows[-top:]], "n_tokens_with_min_count": len(rows),
                      "min_count": min_count}
    return res


if __name__ == "__main__":
    main()
