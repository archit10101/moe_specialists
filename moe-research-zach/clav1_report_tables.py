#!/usr/bin/env python
"""csr-conditional-local-attention-v1: render markdown tables for the report from recorded results only
(final_analysis.json, gen1/gen2/calib_selection results, bench JSON). Writes .tmp/conditional_local_attention_v1/
report_tables.md. Reporting only; no data access, no model."""
import json
from pathlib import Path

ROOT = Path("/home/zapan4/moe-research")
ART = ROOT / "artifacts" / "conditional-local-attention-v1"
B = "FULL-TOKEN-4"


def f(x, d=4, sign=False):
    if x is None:
        return "–"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def main():
    a = json.loads((ART / "final_analysis.json").read_text())
    L = []
    g1 = a["gen1"]
    L.append("### Generation 1 (seed 98, 5M tokens, CALIB-SEARCH)\n")
    L.append("| candidate | CE | ΔCE vs FULL | train throughput ×FULL | step GPU s | global-query rate (measured) | "
             "attention work ×FULL | peak reserved GiB | collapse | outcome |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    order = [B] + g1["ranking"] + [k for k in g1["results"] if k not in g1["ranking"] and k != B]
    for i in order:
        t = a["gen1_table"][i]
        el = t["eliminated"]
        out = "baseline" if i == B else ("advanced" if t["advanced"] else (f"eliminated: {el[1]}" if el and el[0]
                                                                            else "not advanced (rank > 4)"))
        L.append(f"| {i} | {f(t['ce'])} | {f(t['delta'], sign=True)} | {f(t['throughput_ratio'], 3)} | "
                 f"{f(t['gpu_s'], 1)} | {f(t['global_query_rate'], 3)} | {f(t['attention_work_fraction_vs_full'], 3)} | "
                 f"{f(t['peak_reserved_gib'], 1)} | {t['collapsed']} | {out} |")
    if a.get("gen2"):
        g2 = a["gen2"]
        L.append("\n### Generation 2 (seeds 98 / 99, 20M tokens, CALIB-SEARCH)\n")
        L.append("| candidate | CE s98 | CE s99 | Δ s98 | Δ s99 | mean paired Δ | throughput tok/s (mean) | ×FULL | "
                 "global rate | outcome |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        res = g2["results"]
        bt = (res[B]["98"]["train_tok_s"] + res[B]["99"]["train_tok_s"]) / 2
        for i in [B] + g2["ranking"] + g2.get("invalid", []):
            r = res[i]
            d = [r[s]["ce"] - res[B][s]["ce"] if r[s]["ce"] is not None else None for s in ("98", "99")]
            tp = (r["98"]["train_tok_s"] + r["99"]["train_tok_s"]) / 2 if r["98"].get("train_tok_s") else None
            gr = [r[s].get("global_query_rate") for s in ("98", "99")]
            out = "baseline" if i == B else ("advanced" if i in g2["advance"] else
                                            ("invalid" if i in g2.get("invalid", []) else "not advanced"))
            L.append(f"| {i} | {f(r['98']['ce'])} | {f(r['99']['ce'])} | {f(d[0], sign=True)} | {f(d[1], sign=True)} | "
                     f"{f(g2['mean_paired_delta'].get(i), sign=True) if i != B else '–'} | {f(tp, 0)} | "
                     f"{f(tp / bt, 3) if tp else '–'} | {f(gr[0], 3) if gr[0] is not None else '–'} | {out} |")
    if a.get("calib_selection"):
        cs = a["calib_selection"]
        L.append("\n### CALIB-SELECTION (once per frozen Gen-2 checkpoint)\n")
        L.append("| candidate | CE s98 | CE s99 | mean paired Δ vs FULL | speed (Gen-2 tok/s) | global-query fraction |")
        L.append("|---|---|---|---|---|---|")
        info = cs["selection_info"]
        for i, v in cs["per_seed"].items():
            L.append(f"| {i} | {f(v['98']['ce'])} | {f(v['99']['ce'])} | "
                     f"{f(info.get('mean_paired_delta', {}).get(i), sign=True) if i != B else '–'} | "
                     f"{f(info.get('speed_tok_s', {}).get(i), 0) if i != B else '–'} | "
                     f"{f(info.get('global_query_fraction', {}).get(i), 3) if i != B else '–'} |")
        L.append(f"\nSelected: **{cs['winner']}**")
    fv = a.get("final_validation")
    if fv:
        w = a["winner"]
        L.append("\n### Final validation (100M tokens, 3 seeds, 16,850 documents)\n")
        L.append("| | seed 0 | seed 1 | seed 2 | mean |")
        L.append("|---|---|---|---|---|")
        for i in (B, w):
            c = fv["ce"][i]
            L.append(f"| {i} CE | {f(c['0'])} | {f(c['1'])} | {f(c['2'])} | {f(fv['ce_mean'][i])} |")
        d = fv["delta_per_seed"]
        bs = fv["bootstrap"]
        L.append(f"| ΔCE ({w} − FULL) | {f(d['0'], sign=True)} | {f(d['1'], sign=True)} | {f(d['2'], sign=True)} | "
                 f"{f(fv['accuracy_verdict']['mean_delta'], sign=True)} |")
        L.append("| paired doc-bootstrap 95% CI | " + " | ".join(
            f"[{f(p['ci95'][0], sign=True)}, {f(p['ci95'][1], sign=True)}]" for p in bs["per_seed"])
                 + f" | [{f(bs['seed_mean']['ci95'][0], sign=True)}, {f(bs['seed_mean']['ci95'][1], sign=True)}] |")
        L.append("| docs improved | " + " | ".join(f"{p['frac_docs_improved']:.1%}" for p in bs["per_seed"])
                 + f" | {bs['seed_mean']['frac_docs_improved']:.1%} |")
        L.append("\nCE by in-document position (seed mean):\n")
        L.append("| position | " + " | ".join(fv["ce_by_doc_position"][B]["0"].keys()) + " |")
        L.append("|---|" + "---|" * len(fv["ce_by_doc_position"][B]["0"]))
        for i in (B, w):
            keys = fv["ce_by_doc_position"][i]["0"].keys()
            vals = [sum(fv["ce_by_doc_position"][i][s][k] for s in "012") / 3 for k in keys]
            L.append(f"| {i} | " + " | ".join(f(v) for v in vals) + " |")
    sp = a.get("speed_primary")
    if sp:
        L.append("\n### Primary speed measures (ratio = candidate time / fastest baseline time; ≤ 0.90 qualifies)\n")
        L.append("| measure | baseline | winner | ratio | qualifies |")
        L.append("|---|---|---|---|---|")
        for k, v in sp.items():
            vals = [x for kk, x in v.items() if kk not in ("time_ratio", "qualifies_10pct")]
            L.append(f"| {k} | {f(vals[0], 2)} | {f(vals[1], 2)} | {f(v['time_ratio'], 3)} | {v['qualifies_10pct']} |")
        bn = a["bench"]
        L.append("\n### Matched-batch training benchmark (100 warm-up + 500 timed steps, 32 × 1,024 tokens/step)\n")
        L.append("| condition | tokens/s | steps/s | median step ms | peak alloc GiB | peak reserved GiB | CF steps timed |")
        L.append("|---|---|---|---|---|---|---|")
        for k, v in bn["train_matched"].items():
            L.append(f"| {k} | {v['tokens_per_s']:.0f} | {v['steps_per_s']:.3f} | {v['median_step_ms']:.1f} | "
                     f"{v['peak_alloc_gib']:.1f} | {v['peak_reserved_gib']:.1f} | {v['cf_steps_in_timed']} |")
        for k, v in bn.get("train_matched_repeat", {}).items():
            L.append(f"| {k} (repeat) | {v['tokens_per_s']:.0f} | {v['steps_per_s']:.3f} | {v['median_step_ms']:.1f} | "
                     f"{v['peak_alloc_gib']:.1f} | {v['peak_reserved_gib']:.1f} | {v['cf_steps_in_timed']} |")
        L.append("\n### Maximum safe batch (separate; not the primary measure)\n")
        L.append("| condition | micro-batch rows | tokens/s | peak reserved GiB |")
        L.append("|---|---|---|---|")
        for k, v in bn["train_max_batch"].items():
            L.append(f"| {k} | {v['max_safe_micro_batch_rows']} | {v['tokens_per_s']:.0f} | {v['peak_reserved_gib']:.1f} |")
        L.append("\n### Prefill (median latency ms / tokens per s; 100 timed reps)\n")
        conds = list(next(iter(bn["prefill"].values())).keys())
        L.append("| batch, prompt | " + " | ".join(conds) + " |")
        L.append("|---|" + "---|" * len(conds))
        for key, v in bn["prefill"].items():
            L.append(f"| {key} | " + " | ".join(f"{v[c]['median_latency_ms']:.2f} / {v[c]['tokens_per_s']:.0f}"
                                               for c in conds) + " |")
        L.append("\n### Cached greedy decoding of 128 tokens (tokens/s / per-token ms; 100 timed generations)\n")
        dconds = list(next(iter(bn["decode"].values())).keys())
        L.append("| batch, prompt | " + " | ".join(dconds) + " | KV cache MiB | peak alloc GiB (FULL / winner) |")
        L.append("|---|" + "---|" * (len(dconds) + 2))
        for key, v in bn["decode"].items():
            L.append(f"| {key} | " + " | ".join(f"{v[c]['tokens_per_s']:.1f} / {v[c]['per_token_latency_ms']:.2f}"
                                               for c in dconds)
                     + f" | {v[B]['kv_cache_bytes'] / 2**20:.1f} | {v[B]['peak_alloc_gib']:.2f} / "
                       f"{v[a['winner']]['peak_alloc_gib']:.2f} |")
    (ROOT / ".tmp/conditional_local_attention_v1/report_tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
