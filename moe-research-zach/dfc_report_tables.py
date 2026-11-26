#!/usr/bin/env python
"""csr-domain-forecast-curriculum-v1: render markdown tables from final_analysis.json + specialization_trajectory.json
(stdout). CPU only, read-only."""
import json
import sys
from pathlib import Path

import numpy as np

ART = Path("/home/zapan4/moe-research/artifacts/domain-forecast-curriculum-v1")
CONDS = ("TOKEN-MAIN", "LABEL", "FORECAST", "SOFT", "WRONG", "TOKEN-ALL-DATA", "TOKEN-COMPUTE")
LONG = {"TOKEN-MAIN": "TOKEN-MAIN", "LABEL": "LABEL→TOKEN-S40", "FORECAST": "FORECAST→TOKEN-S40",
        "SOFT": "SOFT-SPECIALIST→TOKEN-S40", "WRONG": "WRONG-FORECAST→TOKEN-S40", "TOKEN-ALL-DATA": "TOKEN-ALL-DATA",
        "TOKEN-COMPUTE": "TOKEN-COMPUTE"}


def f(x, n=4):
    return "—" if x is None else f"{x:.{n}f}"


def main():
    fa = json.loads((ART / sys.argv[1] if len(sys.argv) > 1 else ART / "final_analysis.json").read_text())
    st = json.loads((ART / "specialization_trajectory.json").read_text())
    doms = fa["domains"]
    v = fa["validation"]
    print("### Validation CE (nats/token), final checkpoints\n")
    print("| condition | seed 0 | seed 1 | seed 2 | mean | sd | worst-domain (mean) | " + " | ".join(doms) + " |")
    print("|---|" + "---|" * (6 + len(doms)))
    for c in CONDS:
        r = v[c]
        print(f"| {LONG[c]} | " + " | ".join(f(x) for x in r["ce_per_seed"]) + f" | **{f(r['mean'])}** | {f(r['sd'])} | "
              f"{f(r['worst_domain_ce_mean'])} | " + " | ".join(f(x) for x in r["ce_domain_mean"]) + " |")
    print("\n### Paired comparisons (SOFT − X, per-paper token-weighted; negative favours SOFT)\n")
    print("| comparison | seed 0 [95% CI] | seed 1 [95% CI] | seed 2 [95% CI] | pooled [95% CI] | verdict |")
    print("|---|---|---|---|---|---|")
    for c in CONDS:
        if c == "SOFT":
            continue
        k = fa["comparisons"][f"SOFT_vs_{c}"]
        cells = [f"{p['mean']:+.4f} [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}]" for p in k["per_seed"]]
        po = k["pooled"]
        print(f"| SOFT − {LONG[c]} | " + " | ".join(cells) +
              f" | **{po['mean']:+.4f}** [{po['ci95'][0]:+.4f}, {po['ci95'][1]:+.4f}] | {k['verdict']} |")
    print("\n### Every condition vs TOKEN-MAIN (pooled over seeds)\n")
    print("| condition − TOKEN-MAIN | per seed | pooled [95% CI] |")
    print("|---|---|---|")
    for c in CONDS:
        if c == "TOKEN-MAIN":
            continue
        k = fa["comparisons"][f"{c}_minus_TOKEN-MAIN"]
        po = k["pooled"]
        print(f"| {LONG[c]} | " + ", ".join(f"{x:+.4f}" for x in k["per_seed"]) +
              f" | {po['mean']:+.4f} [{po['ci95'][0]:+.4f}, {po['ci95'][1]:+.4f}] |")
    print("\n### Specialisation trajectory (calibration; mean over 3 seeds)\n")
    print("| condition | ckpt | mode | calib CE | MI mean (nats) | MI null | H(d|e) mean | prov. mass | specificity "
          "[95% CI] | functional? | all-layer enriched spec. | all-layer provisional spec. | stability idx / best | "
          "collapse |")
    print("|---|" + "---|" * 13)
    for c in CONDS:
        for tag in ("p20", "p40", "p50", "p75", "p100", "final"):
            for mode in ("normal", "neutral"):
                k = st["summary"].get(f"{c}|{tag}|{mode}")
                if not k:
                    continue
                sp = k["specificity_layer_domain"]
                print(f"| {c} | {tag} | {mode} | {f(k['calib_ce_mean'])} | {f(k['mi_mean'])} | {f(k['mi_null_mean'])} | "
                      f"{f(k['h_dom_given_expert_mean'], 3)} | {f(k['prov_mass_mean'], 3)} | {sp['mean']:+.4f} "
                      f"[{sp['ci95'][0]:+.4f}, {sp['ci95'][1]:+.4f}] | {'yes' if k['functional_specialization'] else 'no'}"
                      f" | {k['specificity_all_layers_enriched_mean']:+.4f} | "
                      f"{k['specificity_all_layers_provisional_mean']:+.4f} | "
                      f"{k['stability']['index_aligned_cos']:.2f} / {k['stability']['best_match_cos']:.2f} | "
                      f"{'YES' if k['any_collapse'] else 'no'} (max share {k['max_share']:.3f}, unused {k['max_unused']}) |")
    print("\n### Matched-token CALIB-MON curves (neutral CE, mean over seeds)\n")
    mt = fa["matched_token_curves_mean_neutral"]
    ticks = [t for t in mt["TOKEN-MAIN"]]
    sel = sorted([t for t in ticks if int(t) in (5_625_000, 22_500_000, 45_000_000, 56_250_000, 84_375_000, 112_500_000)], key=int)
    print("| condition | " + " | ".join(f"{int(t) / 1e6:.2f}M" for t in sel) + " |")
    print("|---|" + "---|" * len(sel))
    for c in CONDS:
        print(f"| {c} | " + " | ".join(f(mt[c].get(t)) for t in sel) + " |")
    print("\n### Throughput, overhead and compute\n")
    tp = fa["training_throughput"]
    print("| condition | train tokens | train tok/s | train-step s (mean) |")
    print("|---|---|---|---|")
    for c in CONDS:
        print(f"| {c} | {tp[c]['valid_train_tokens']:,} | {tp[c]['train_tokens_per_s_mean']:,.0f} | "
              f"{tp[c]['train_step_seconds_mean']:.0f} |")
    o = fa["overhead"]
    print(f"\nDraft train-step s {o['draft_train_step_seconds']:.0f} (amortised /3: "
          f"{o['draft_train_step_seconds_amortized_3_seeds']:.0f}); MAIN-correct cache GPU s "
          f"{o['cache_main_correct_gpu_seconds']:.0f} (amortised /3: {o['cache_main_correct_amortized_3_seeds']:.0f}); "
          f"all cache jobs {json.dumps({k: round(x) for k, x in o['cache_all_jobs_gpu_seconds'].items()})}")
    print(f"\nGPU seconds by category: {json.dumps({k: round(x) for k, x in fa['gpu_seconds_by_category'].items()})}; "
          f"total {fa['gpu_seconds_total'] / 3600:.2f} GPU-h")
    cm = fa["compute_match"]
    print(f"\nCompute match: C = {cm['C_seconds']:.0f} s -> N = {cm['n_tokens']:,} tokens "
          f"({cm['n_tokens_over_main']:.3f} x 112.5M)")
    b = fa.get("bench")
    if b:
        print("\n| model (seed 0) | prefill tok/s | decode ms/token | curriculum-path forward overhead |")
        print("|---|---|---|---|")
        for k, r in b["runs"].items():
            print(f"| {k} | {r['prefill_tokens_per_s']:,.0f} | {r['decode_ms_per_token']:.2f} | "
                  f"{100 * r['curriculum_forward_overhead_frac']:+.1f}% |")
        r = b["runs"]["SOFT_s0"]
        print(f"\nParams: standalone total {r['params_standalone']['total']:,}, active/token "
              f"{r['params_standalone']['active_per_token']:,}; with curriculum modules {r['params_with_curriculum_modules']:,}"
              f" (added {r['added_params']:,})")
    print("\n### Specialisation vs final CE (Spearman, 15 main-plan runs)\n")
    print(json.dumps(fa["spec_vs_ce_spearman"]))
    print("\n### Forecast accuracy on calibration (j = 1..4)\n")
    for s, a in o["calib_forecast_accuracy"].items():
        print(f"- {s}: " + ", ".join(f"{x:.3f}" for x in a["overall_per_j"]))


if __name__ == "__main__":
    main()
