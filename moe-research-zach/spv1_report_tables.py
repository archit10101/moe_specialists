#!/usr/bin/env python
"""csr-speculative-pool-routing-v1: build markdown report tables from the experiment's artifacts (read-only except
.tmp/speculative_pool_v1/report_tables.md and, once, artifacts/speculative-pool-routing-v1/run_inventory.json)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402

ART = guard.ROOT / "artifacts" / "speculative-pool-routing-v1"
ST = guard.ROOT / ".tmp" / "speculative_pool_v1"


def j(p):
    return json.loads(Path(p).read_text())


def tsv(name):
    p = ST / name
    return [ln.split("\t") for ln in p.read_text().splitlines() if ln.strip()] if p.exists() else []


def f4(x, sign=False):
    return "–" if x is None else (f"{x:+.4f}" if sign else f"{x:.4f}")


def gen_table(res_file, specs, rej=None):
    r = j(ART / res_file)["results"]
    tok = r["TOKEN-4"]
    out = ["| candidate | family | description | CALIB-SEARCH CE | Δ vs TOKEN-4 | per-seed Δ | status | refresh | "
           "pool recall | top-4 contained | distinct/8 tok | adj. Jaccard | max unused / share | train k tok/s | GPU s |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    from csr import spv1_select as sel
    rows = [("TOKEN-4", "token", "token router, global top-4")] + [(s["id"], s["family"], s.get("desc") or
                                                                    s.get("change", "")) for s in specs]
    for cid, fam, desc in rows:
        a = r[cid]
        ps = a["per_seed"]
        comp = [k for k in ps if ps[k]["complete"]]
        d = [ps[k]["ce"] - tok["per_seed"][k]["ce"] for k in comp] if cid != "TOKEN-4" else []
        mu = max((ps[k]["collapse"]["max_unused"] for k in comp), default=None)
        ms = max((ps[k]["collapse"]["max_share"] for k in comp), default=None)
        tps = [ps[k]["train_tok_s"] / 1e3 for k in comp]
        gs = [ps[k]["gpu_s"] for k in comp]
        jac = [ps[k].get("adjacent_jaccard") for k in comp if ps[k].get("adjacent_jaccard") is not None]
        stt = "valid" if sel.is_valid(a) else "INVALID"
        if rej and rej.get(cid, [False])[0]:
            stt = "REJECTED"
        out.append(f"| {cid} | {fam} | {desc} | {f4(a['ce'])} | "
                   f"{f4(a['ce'] - tok['ce'], True) if cid != 'TOKEN-4' and a['ce'] is not None else '–'} | "
                   f"{', '.join(f'{x:+.4f}' for x in d) or '–'} | {stt} | {f4(a.get('refresh_rate'))} | "
                   f"{f4(a.get('pool_recall'))} | {f4(a.get('contained'))} | {f4(a.get('distinct8'))} | "
                   f"{f4(sum(jac) / len(jac)) if jac else '–'} | {mu} / {f4(ms)} | "
                   f"{sum(tps) / max(len(tps), 1):.1f} | {sum(gs) / max(len(gs), 1):.0f} |")
    return "\n".join(out)


def main():
    guard.check_root()
    from csr import spv1_select as sel
    md = ["## Generation 1 (seed 98, 5M tokens, CALIB-SEARCH)\n", gen_table("gen1_results.json", sel.GEN1)]
    if (ART / "gen2_results.json").exists():
        g2 = j(ART / "gen2_specs.json")["specs"]
        rej = j(ART / "gen2_results.json").get("rejections")
        md += ["\n## Generation 2 (seeds 98, 99; 20M tokens; CALIB-SEARCH)\n", gen_table("gen2_results.json", g2, rej),
               "\n| Gen-2 | parent (Gen-1) | change | rejection |\n|---|---|---|---|"]
        for s in g2:
            rr = rej.get(s["id"], [False, ""])
            md.append(f"| {s['id']} | {s['parent']} | {s['change']} | {rr[1] if rr[0] else '–'} |")
    if (ART / "gen3_results.json").exists():
        g3 = j(ART / "gen3_specs.json")["specs"]
        md += ["\n## Generation 3 (seeds 97, 98, 99; 50M tokens; CALIB-SEARCH)\n", gen_table("gen3_results.json", g3)]
    rows = []
    for ln in (guard.ROOT / "ledger" / "gpu_time_ledger.tsv").read_text().splitlines()[1:]:
        p = ln.split("\t")
        if len(p) >= 6 and p[4].startswith("spv1"):
            rows.append(p)
    by = {}
    for p in rows:
        if p[4] == "spv1 train":
            run = p[3].split("_spv1_train_")[-1]
            stage = run.split("_s")[0] if not run.startswith("draft") else "draft"
        else:
            stage = p[4]
        by.setdefault(stage, [0, 0])
        by[stage][0] += int(p[2])
        by[stage][1] += 1
    md.append("\n## GPU-time ledger (purpose `spv1 …`)\n\n| stage | entries | GPU seconds | GPU hours |\n|---|---|---|---|")
    for k, (s, n) in sorted(by.items()):
        md.append(f"| {k} | {n} | {s} | {s / 3600:.2f} |")
    tot = sum(int(p[2]) for p in rows)
    md.append(f"| **total** | {len(rows)} | {tot} | **{tot / 3600:.2f}** |")
    if rows:
        md.append(f"\nGPU wall span: {rows[0][0]} → {rows[-1][1]}; non-COMPLETE/PASS entries: "
                  f"{[(p[3], p[5]) for p in rows if p[5] not in ('COMPLETE', 'PASS')]}")
    fl = tsv("failures.tsv")
    md.append("\n## Failures registry\n")
    md.append("\n".join(f"- {' | '.join(x)}" for x in fl) or "- none")
    out = ST / "report_tables.md"
    out.write_text("\n".join(md) + "\n")
    inv_p = ART / "run_inventory.json"
    if (ART / "final_analysis.json").exists() and not inv_p.exists():
        inv = {"configs": {}, "runs": {}, "artifacts": {}}
        for c in sorted((guard.ROOT / "configs/speculative-pool-routing-v1").rglob("*.json")):
            inv["configs"][str(c.relative_to(guard.ROOT))] = guard.sha256_file(c)
        for reg in ("draft_runs.tsv", "gen1_runs.tsv", "gen2_runs.tsv", "gen3_runs.tsv", "final_runs.tsv"):
            for p in tsv(reg):
                rd = guard.ROOT / p[1]
                e = {"status": p[2] if len(p) > 2 else "?", "config_sha256": guard.sha256_file(rd / "config.json")}
                if (rd / "model.pt").exists():
                    e["checkpoint"] = str((rd / "model.pt").relative_to(guard.ROOT))
                    e["checkpoint_sha256"] = guard.sha256_file(rd / "model.pt")
                for ev in sorted(rd.glob("eval_*.json")):
                    e[ev.name] = guard.sha256_file(ev)
                if (rd / "calib_search_eval.json").exists():
                    e["calib_search_eval_sha256"] = guard.sha256_file(rd / "calib_search_eval.json")
                inv["runs"][f"{reg.split('_')[0]}/{p[0]}"] = {"run": p[1], **e}
        for a in sorted(ART.glob("*.json")) + sorted(ART.glob("*.tsv")) + sorted(ART.glob("*.jsonl")):
            inv["artifacts"][a.name] = guard.sha256_file(a)
        guard.atomic_write_json(inv_p, inv, mode=0o400)
        guard.ledger("SP_V1_RUN_INVENTORY", f"{inv_p.relative_to(guard.ROOT)} sha256={guard.sha256_file(inv_p)}")
    print(out)


if __name__ == "__main__":
    main()
