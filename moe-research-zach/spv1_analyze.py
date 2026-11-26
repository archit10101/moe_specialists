#!/usr/bin/env python
"""csr-speculative-pool-routing-v1: final analysis (prereg §12-§13). CPU only; reads the final runs, their single
validation evaluations, the benchmarks and the ledgers; writes artifacts/speculative-pool-routing-v1/final_analysis.json
once. Dry-run by default (requires --execute and MOE_RESEARCH_APPROVE=cpu-model)."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402

ST = guard.ROOT / ".tmp" / "speculative_pool_v1"


def jl(p):
    return json.loads(Path(p).read_text())


def tsv(name):
    p = ST / name
    if not p.exists():
        return {}
    return {ln.split("\t")[0]: ln.split("\t")[1:] for ln in p.read_text().splitlines() if ln.strip()}


def curves(rd):
    """Monitor CE (first 128 CALIB-SEARCH rows) and training CE at matched token counts."""
    mon, trn = {}, {}
    for ln in (guard.ROOT / rd / "metrics.jsonl").read_text().splitlines():
        r = json.loads(ln)
        if "monitor_eval" in r:
            mon[int(r["valid_tokens_seen"] // 1_000_000)] = r["monitor_eval"]["calib_search_head"]["ce_nats_per_token"]
        elif "lm_ce" in r:
            trn[int(r["valid_tokens_seen"] // 1_000_000)] = r["lm_ce"]
    return mon, trn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    if not guard.require_execution(a.execute, "cpu-model"):
        return
    import numpy as np
    from csr import spv1_select as sel
    from csr.spv1_run import ART, FINAL_FROZEN
    out_p = ART / "final_analysis.json"
    if out_p.exists():
        print("final_analysis.json exists; never overwritten")
        return
    fz = jl(FINAL_FROZEN)
    ids = fz["ids"]
    roles = fz["roles"]
    runs = tsv("final_runs.tsv")
    vals = tsv("val_evals.tsv")
    seeds = (0, 1, 2)
    per = {i: {} for i in ids}
    for i in ids:
        for s in seeds:
            k = f"seed{s}/{i}"
            rd = runs[k][0]
            ev = jl(guard.ROOT / vals[k][0])
            assert ev["status"] == "COMPLETE" and ev["evalset"] == "val"
            summ = jl(guard.ROOT / rd / "summary.json")
            facts = jl(guard.ROOT / rd / "run_facts.json")
            per[i][s] = {"run": rd, "eval": vals[k][0], "ce": ev["ce_nats_per_token"], "doc_nll": ev["doc_nll"],
                         "doc_n": ev["doc_n"], "n_valid_tokens": ev["n_valid_tokens"],
                         "collapse": ev["collapse"], "valid_train_tokens": summ["valid_train_tokens"],
                         "train_tok_s": summ["train_valid_tokens_per_s_after_warmup"],
                         "process_seconds": summ["process_seconds"], "peak_reserved_gib": summ["peak_reserved_gib"],
                         "extra_param_fraction": facts["extra_param_fraction"], "extra_params": facts["extra_params"],
                         "init_backbone_param_sha256": facts["init_backbone_param_sha256"],
                         "diag": {k_: ev.get(k_) for k_ in (
                             "refresh_rate", "mean_block_length", "block_length_hist", "adjacent_expert_jaccard",
                             "distinct_experts_per_window", "pool_recall_vs_unrestricted_top4",
                             "unrestricted_top4_contained_frac", "router_entropy", "router_max_prob",
                             "selected_gate_mass", "selector_pool_mass", "selector_entropy", "expert_max_share",
                             "experts_unused", "expert_load_cv", "teacher_forced_spec", "ce_by_doc_position",
                             "eval_valid_tokens_per_s")},
                         "curves": curves(rd)}
    tok = per["TOKEN-4"]
    for s in seeds:  # matched initialisation and evaluation tokens within a seed
        for i in ids:
            assert per[i][s]["init_backbone_param_sha256"] == tok[s]["init_backbone_param_sha256"], (i, s)
            assert per[i][s]["n_valid_tokens"] == tok[s]["n_valid_tokens"], (i, s)
    res = {"ids": ids, "roles": roles, "budget": fz["budget"], "controls_budget": fz.get("controls_budget"),
           "per_seed_ce": {i: {s: per[i][s]["ce"] for s in seeds} for i in ids},
           "mean_ce": {i: float(np.mean([per[i][s]["ce"] for s in seeds])) for i in ids},
           "sd_ce": {i: float(np.std([per[i][s]["ce"] for s in seeds], ddof=1)) for i in ids},
           "comparisons": {}, "diagnostics": {}, "curves": {}}

    def comp(a_id, b_id):
        """b - a (negative = b better)."""
        d = [per[b_id][s]["ce"] - per[a_id][s]["ce"] for s in seeds]
        boot = sel.paired_bootstrap_multi([(per[a_id][s]["doc_nll"], per[a_id][s]["doc_n"]) for s in seeds],
                                          [(per[b_id][s]["doc_nll"], per[b_id][s]["doc_n"]) for s in seeds])
        return d, boot

    for i in ids:
        if i == "TOKEN-4":
            continue
        d, boot = comp("TOKEN-4", i)
        coll = any(per[i][s]["collapse"]["collapsed"] for s in seeds)
        matched = per[i][0]["valid_train_tokens"] == tok[0]["valid_train_tokens"]
        res["comparisons"][f"{i} vs TOKEN-4"] = {"per_seed_diff": d, "bootstrap": boot,
                                                 "criteria": sel.criteria(d, boot, coll, matched_tokens=matched),
                                                 "matched_training_tokens": matched}
    pairs = []
    if "causal" in roles and "RC-causal" in roles:
        pairs.append((roles["RC-causal"], roles["causal"], "content dependence (causal finalist vs random control)"))
    if "spec" in roles and "RC-spec" in roles:
        pairs.append((roles["RC-spec"], roles["spec"], "content dependence (spec finalist vs random control)"))
    if "causal" in roles and "spec" in roles:
        pairs.append((roles["causal"], roles["spec"], "speculative vs causal pool selection"))
    for a_id, b_id, what in pairs:
        d, boot = comp(a_id, b_id)
        res["comparisons"][f"{b_id} vs {a_id}"] = {"what": what, "per_seed_diff": d, "bootstrap": boot,
                                                   "mean_diff": float(np.mean(d)),
                                                   "all_seeds_same_sign": bool(all(x < 0 for x in d) or
                                                                               all(x > 0 for x in d))}
    for i in ids:
        diag = {}
        for k_ in per[i][0]["diag"]:
            vals_ = [per[i][s]["diag"][k_] for s in seeds]
            diag[k_] = vals_[0] if not isinstance(vals_[0], list) else np.mean(np.array(vals_, dtype=float), 0).tolist()
        res["diagnostics"][i] = {"seed0": per[i][0]["diag"], "seed_mean": diag,
                                 "train_tok_s": [per[i][s]["train_tok_s"] for s in seeds],
                                 "process_seconds": [per[i][s]["process_seconds"] for s in seeds],
                                 "peak_reserved_gib": [per[i][s]["peak_reserved_gib"] for s in seeds],
                                 "extra_params": per[i][0]["extra_params"],
                                 "extra_param_fraction": per[i][0]["extra_param_fraction"],
                                 "collapse": [per[i][s]["collapse"] for s in seeds]}
        mt = {}
        for s in seeds:
            mon, trn = per[i][s]["curves"]
            mt0, tt0 = tok[s]["curves"]
            mt[s] = {"monitor_ce_minus_token": {m_: mon[m_] - mt0[m_] for m_ in sorted(mon) if m_ in mt0},
                     "train_ce_minus_token_at_Mtokens": {m_: trn[m_] - tt0[m_] for m_ in sorted(trn)
                                                          if m_ in tt0 and m_ % 10 == 0}}
        res["curves"][i] = mt
    # ---- benchmarks
    benches = sorted(ART.glob("bench_final_*.json"))
    good = [b for b in benches if jl(b).get("status") == "COMPLETE"]
    if good:
        bj = jl(good[-1])
        res["bench_file"] = str(good[-1].relative_to(guard.ROOT))
        res["bench"] = bj["results"]
        conds = bj["results"]["conditions"]
        sp = {}
        tok_c = conds.get("TOKEN-4", {})
        for lab, c in conds.items():
            if lab == "TOKEN-4":
                continue
            fam = c["family"]
            crit = res["comparisons"].get(f"{c['candidate']} vs TOKEN-4", {}).get("criteria", {})
            ac = bool(crit.get("accuracy_comparable"))
            for key, v in c["decode"].items():
                if key.startswith("greedy_"):
                    ctrl = tok_c.get("decode", {}).get(key)
                else:
                    ctrl = tok_c.get("decode", {}).get(key)  # TOKEN-4 + speculation, same gamma and batch
                if ctrl:
                    sp[f"{lab}:{key}"] = {"control": f"TOKEN-4:{key}", **sel.speed_claim(v["tokens_per_s"],
                                                                                        ctrl["tokens_per_s"], ac)}
            for key in ("train_step_tokens_per_s", "prefill_b16_tokens_per_s", "prefill_b1_tokens_per_s"):
                if key in c and key in tok_c:
                    sp[f"{lab}:{key}"] = {"control": f"TOKEN-4:{key}", **sel.speed_claim(c[key], tok_c[key], ac)}
            del fam
        res["speed"] = sp
    else:
        res["bench"] = None
        res["speed"] = None
    # ---- resources
    led = (guard.ROOT / "ledger" / "gpu_time_ledger.tsv").read_text().splitlines()[1:]
    gs = [ln.split("\t") for ln in led if len(ln.split("\t")) >= 6 and ln.split("\t")[4].startswith("spv1")]
    res["gpu_seconds_total"] = sum(int(p[2]) for p in gs)
    res["gpu_hours_total"] = res["gpu_seconds_total"] / 3600
    res["gpu_ledger_entries"] = len(gs)
    res["created"] = guard.now()
    guard.atomic_write_json(out_p, res, mode=0o400)
    guard.ledger("SP_V1_FINAL_ANALYSIS", f"{out_p.relative_to(guard.ROOT)} sha256={guard.sha256_file(out_p)}")
    print(json.dumps({k: res[k] for k in ("mean_ce", "gpu_hours_total")}, indent=1))
    for k, v in res["comparisons"].items():
        print(k, json.dumps({kk: v[kk] for kk in v if kk in ("per_seed_diff", "mean_diff")}),
              v["bootstrap"]["ci95"], v.get("criteria", {}).get("accuracy_comparable"),
              v.get("criteria", {}).get("substantially_better"))


if __name__ == "__main__":
    main()
