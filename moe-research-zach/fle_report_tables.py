#!/usr/bin/env python
"""csr-forecast-latent-edl-v1: markdown tables for the report from the frozen artifacts (read-only).
Writes artifacts/forecast-latent-edl-v1/report_tables.md and candidate_table.tsv."""
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402
from csr.fle_paths import ART  # noqa: E402


def f(x, n=4):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{n}f}"
    return str(x)


def main():
    guard.check_root()
    out = []
    # ---------------- search table
    rows = []
    for p in sorted(glob.glob(str(ART / "diagnostics" / "C*_s*.json"))):
        d = json.loads(Path(p).read_text())
        rows.append(d)
    rows.sort(key=lambda d: (d["cid"], {"s5": 0, "s20": 1, "s50": 2}[d["tag"]]))
    out.append("## Candidate search (CALIB-SEARCH, seed 100)\n")
    out.append("| cand | stage | latent cfg | beta | layers | dCE vs T4 | FSI dom (T4) | FSI e5 (T4) | MI latent (T4 base) | "
               "aux/LM grad | JSD | load min x unif | flags |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    tsv = ["cid\ttag\tcfg\tbeta\tlayers\tdce\tfsi_domain\tfsi_domain_t4\tfsi_cluster\tfsi_cluster_t4\tmi_latent\t"
           "aux_lm_grad_ratio\tmedian_jsd\tload_min\tvalid\tflags"]
    for d in rows:
        k = d["knobs"]
        out.append(f"| {d['cid']} | {d['tag']} | {k['cfg']} | {k['beta']:g} | {k['layers']} | {d['dce']:+.4f} | "
                   f"{f(d['fsi_domain'], 5)} ({f(d['fsi_domain_t4'], 5)}) | {f(d['fsi_cluster'], 5)} "
                   f"({f(d['fsi_cluster_t4'], 5)}) | {f(d['mi_latent'])} | {f(d['aux_lm_grad_ratio'])} | "
                   f"{f(d['median_jsd'], 5)} | {d['load_min'] * 32:.3f} | {','.join(d['flags']) or '—'} |")
        tsv.append("\t".join(str(x) for x in (d["cid"], d["tag"], k["cfg"], k["beta"], k["layers"], d["dce"],
                                              d["fsi_domain"], d["fsi_domain_t4"], d["fsi_cluster"],
                                              d["fsi_cluster_t4"], d["mi_latent"], d["aux_lm_grad_ratio"],
                                              d["median_jsd"], d["load_min"], d["valid"], ",".join(d["flags"]))))
    (ART / "candidate_table.tsv").write_text("\n".join(tsv) + "\n")
    for s in (1, 2):
        p = ART / "search" / f"promotion_s{s}.json"
        if p.exists():
            pr = json.loads(p.read_text())
            out.append(f"\nStage {s} promotion (tol {pr['tol']:.4f}): {pr['promoted']} {pr['flags']}")
    if (ART / "frozen.json").exists():
        fz = json.loads((ART / "frozen.json").read_text())
        out.append(f"\nFrozen winner: **{fz['winner']}** {fz['knobs']} flag={fz['flag']}")
        out.append("\n| cand | CALIB-SEL CE | T4 CE | dCE | FSI dom (T4) | FSI e5 (T4) | eligible |")
        out.append("|---|---|---|---|---|---|---|")
        for r in fz["selection_rows"]:
            out.append(f"| {r['cid']} | {r['ce']:.4f} | {r['ce_t4']:.4f} | {r['dce']:+.4f} | {r['fsi_domain']:.5f} "
                       f"({r['fsi_domain_t4']:.5f}) | {r['fsi_cluster']:.5f} ({r['fsi_cluster_t4']:.5f}) | {r['eligible']} |")
    # ---------------- final tables
    fa_p = ART / "final_analysis.json"
    if fa_p.exists():
        fa = json.loads(fa_p.read_text())
        out.append("\n## Final VAL CE (nats/token)\n")
        out.append("| condition | seed 0 | seed 1 | seed 2 | mean |")
        out.append("|---|---|---|---|---|")
        for c, v in fa["val_ce"].items():
            if not v:
                continue
            vals = [v.get(str(s), v.get(s)) for s in (0, 1, 2)]
            got = [x for x in vals if x is not None]
            out.append(f"| {c} | " + " | ".join(f(x) for x in vals) + f" | {sum(got) / len(got):.4f} |")
        out.append("\n| comparison | per seed | mean | paired doc 95% CI |")
        out.append("|---|---|---|---|")
        for k, v in fa["ce_differences"].items():
            ps = ", ".join(f"s{s}: {x:+.4f}" for s, x in v["per_seed"].items())
            ci = v.get("paired_doc_ci", {}).get("ci95")
            out.append(f"| {k} | {ps} | {v['mean']:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] |" if ci else
                       f"| {k} | {ps} | {v['mean']:+.4f} | — |")
        out.append("\n## Specialisation at 100% (VAL audit subset)\n")
        out.append("| condition | seed | FSI dom | FSI e5 | FSI_shuf dom | MI dom (null) | MI e5 | MI latent | topic pred dom |"
                   " within-expert sim | per-expert interaction (frac>0) |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for c, bys in fa["trajectory"].items():
            for s, bt in sorted(bys.items()):
                r = bt.get("p100")
                if not r:
                    continue
                out.append(f"| {c} | {s} | {r['fsi_domain']:.5f} | {r['fsi_cluster']:.5f} | {f(r.get('fsi_shuf_domain'), 5)} | "
                           f"{r['mi_domain']:.4f} ({r['mi_domain_null']:.5f}) | {r['mi_cluster']:.4f} | {f(r.get('mi_latent'))} | "
                           f"{r['topic_pred_domain']:.3f} | {f(r.get('within_expert_similarity'))} | "
                           f"{f(r.get('per_expert_mean_interaction'), 5)} ({f(r.get('per_expert_frac_positive'), 3)}) |")
        out.append("\n## Specialisation trajectory (FSI domain / e5)\n")
        out.append("| condition | seed | 40% | 60% | 80% | 100% |")
        out.append("|---|---|---|---|---|---|")
        for c, bys in fa["trajectory"].items():
            for s, bt in sorted(bys.items()):
                cells = [f"{bt[t]['fsi_domain']:.5f} / {bt[t]['fsi_cluster']:.5f}" if t in bt else "—"
                         for t in ("p040", "p060", "p080", "p100")]
                out.append(f"| {c} | {s} | " + " | ".join(cells) + " |")
        out.append("\n## Criteria\n")
        out.append("```\n" + json.dumps(fa["criteria"], indent=1) + "\n```")
        out.append("\n## GPU seconds by category\n")
        for k, v in sorted(fa["gpu_seconds_by_category"].items()):
            out.append(f"- {k}: {v / 3600:.2f} h")
        out.append(f"- total: {fa['gpu_hours_total']:.2f} h")
    (ART / "report_tables.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
