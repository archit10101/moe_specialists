#!/usr/bin/env python
"""Markdown tables for M2D2_DOMAIN_PRIOR_V1_REPORT.md from artifacts/m2d2-domain-prior-v1/final_analysis.json."""
import json, sys
from pathlib import Path
A = json.loads(Path("/home/zapan4/moe-research/artifacts/m2d2-domain-prior-v1/final_analysis.json").read_text())
D = ["Mathematics_and_logic", "Technology_and_applied_sciences", "Natural_and_physical_sciences", "Health_and_fitness",
     "History_and_events", "Culture_and_the_arts"]
ab = ["Math", "Tech", "NatSci", "Health", "History", "Culture"]
o = ["| run | overall CE | macro CE | " + " | ".join(ab) + " | alpha (l4..l7) |", "|---" * (4 + len(ab)) + "|"]
for k, v in A["evaluations"].items():
    al = "–" if not v["alpha"] else ", ".join(f"{x:.2f}" for x in v["alpha"])
    o.append(f"| {k} | {v['ce']:.4f} | {v['macro_domain_ce']:.4f} | " + " | ".join(f"{v['per_domain_ce'][d]:.4f}" for d in D) + f" | {al} |")
o += ["", "| comparison | Δ overall [95% CI] | Δ macro | " + " | ".join(ab) + " |", "|---" * (3 + len(ab)) + "|"]
for k, v in A["comparisons"].items():
    c = v["overall"]
    o.append(f"| {k} | {c['diff']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}] | {v['macro_domain_diff']:+.4f} | " +
             " | ".join(f"{v['per_domain'][d]['diff']:+.4f}" for d in D) + " |")
o += ["", "| comparison | pos 0-16 | 16-64 | 64-256 | 256-512 | 512-1024 |", "|---|---|---|---|---|---|"]
for k, v in A["comparisons"].items():
    b = v["by_position_posthoc"]
    o.append(f"| {k} | " + " | ".join(f"{b[x]['diff']:+.4f}" for x in ("0-16", "16-64", "64-256", "256-512", "512-1024")) + " |")
o += ["", "| run | MI(domain;expert) l1..l7 | mean MI l4-7 | null max | mean H(domain|expert) | min/max unused | max load share | adj. Jaccard l4-7 | top-4 changed by prior l4-7 | MI w/o prior l4-7 |",
      "|---|---|---|---|---|---|---|---|---|---|"]
for k, v in A["specialisation"].items():
    mi = [r["mi_domain_expert"] for r in v]
    late = v[3:]
    ch = ", ".join(f"{r['frac_tokens_top4_changed_by_prior']:.3f}" for r in late) if "frac_tokens_top4_changed_by_prior" in late[0] else "–"
    mn = ", ".join(f"{r['mi_without_prior_same_states']:.4f}" for r in late) if "mi_without_prior_same_states" in late[0] else "–"
    o.append(f"| {k} | " + ", ".join(f"{x:.4f}" for x in mi) + f" | {sum(mi[3:]) / 4:.4f} | {max(r['mi_null_max'] for r in v):.5f} | "
             f"{sum(r['mean_domain_entropy_per_expert'] for r in v) / len(v):.3f} | {max(r['experts_unused'] for r in v)} | "
             f"{max(r['expert_max_share'] for r in v):.3f} | {sum(r['adjacent_token_jaccard'] for r in late) / 4:.3f} | {ch} | {mn} |")
o += ["", "| comparison | frac changed | Δ on changed tokens | Δ on unchanged tokens |", "|---|---|---|---|"]
for k, v in A["changed_tokens"].items():
    o.append(f"| {k} | {v['frac_changed']:.3f} | {v['diff_changed']:+.4f} | {v['diff_unchanged']:+.4f} |")
o += ["", "| run | layer: expert (top domain, share) ΔCE top domain / other domains |", "|---|---|"]
for k, v in A["evaluations"].items():
    if v["functional_ablation"]:
        o.append(f"| {k} | " + "; ".join(f"l{f['layer']}: e{f['expert']} ({ab[f['top_domain']]}, {f['top_domain_share']:.2f}) "
                                          f"{f['dce_top_domain']:+.4f} / {f['dce_other_domains']:+.4f}" for f in v["functional_ablation"]) + " |")
print("\n".join(o))
