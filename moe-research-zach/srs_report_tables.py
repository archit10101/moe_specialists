#!/usr/bin/env python
"""csr-sequence-routing-search-v1: build report tables (markdown) and the checkpoint/config/hash inventory from the
experiment's artifacts. Read-only except for writing artifacts/sequence-routing-search-v1/run_inventory.json (once)
and .tmp/sequence_routing_v1/report_tables.md."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402

ART = guard.ROOT / "artifacts" / "sequence-routing-search-v1"
ST = guard.ROOT / ".tmp" / "sequence_routing_v1"


def j(p):
    return json.loads(Path(p).read_text())


def tsv(name):
    p = ST / name
    return [ln.split("\t") for ln in p.read_text().splitlines() if ln.strip()] if p.exists() else []


def f4(x, sign=False):
    return "–" if x is None else (f"{x:+.4f}" if sign else f"{x:.4f}")


def gen_table(gen, res_file, specs):
    r = j(ART / res_file)["results"]
    tok = r["TOKEN-4"]
    out = [f"| candidate | family | description | CALIB-SEARCH CE | Δ vs TOKEN-4 | per-seed Δ | valid | collapse "
           f"(max unused / max share) | train k tok/s | GPU s/run |", "|---|---|---|---|---|---|---|---|---|---|"]
    from csr import srs_select as sel
    rows = [("TOKEN-4", "token", "token router, global top-4")] + [(s["id"], s["family"], s.get("desc") or s.get("change", ""))
                                                                   for s in specs]
    for cid, fam, desc in rows:
        a = r[cid]
        ps = a["per_seed"]
        d = [ps[k]["ce"] - tok["per_seed"][k]["ce"] for k in ps if ps[k]["complete"]] if cid != "TOKEN-4" else []
        mu = max((ps[k]["collapse"]["max_unused"] for k in ps if ps[k]["complete"]), default=None)
        ms = max((ps[k]["collapse"]["max_share"] for k in ps if ps[k]["complete"]), default=None)
        tps = [ps[k]["train_tok_s"] / 1e3 for k in ps if ps[k]["complete"]]
        gs = [ps[k]["gpu_s"] for k in ps if ps[k]["complete"]]
        out.append(f"| {cid} | {fam} | {desc} | {f4(a['ce'])} | {f4(a['ce'] - tok['ce'], True) if cid != 'TOKEN-4' and a['ce'] is not None else '–'} "
                   f"| {', '.join(f'{x:+.4f}' for x in d) or '–'} | {'yes' if sel.is_valid(a) else 'NO'} | {mu} / {f4(ms)} "
                   f"| {sum(tps) / max(len(tps), 1):.1f} | {sum(gs) / max(len(gs), 1):.0f} |")
    return "\n".join(out)


def main():
    guard.check_root()
    from csr import srs_select as sel
    md = []
    md.append("## Generation 1 (seed 98, 5M tokens, CALIB-SEARCH)\n")
    md.append(gen_table(1, "gen1_results.json", sel.GEN1))
    if (ART / "gen2_specs.json").exists() and (ART / "gen2_results.json").exists():
        g2 = j(ART / "gen2_specs.json")["specs"]
        md.append("\n## Generation 2 (seeds 98, 99; 10M tokens; CALIB-SEARCH)\n")
        md.append(gen_table(2, "gen2_results.json", g2))
        md.append("\n### Genealogy\n\n| Gen-2 | parent (Gen-1) | change | identical config to |\n|---|---|---|---|")
        for s in g2:
            md.append(f"| {s['id']} | {s['parent']} | {s['change']} | {s.get('same_as') or 'new'} |")
    if (ART / "gen3_specs.json").exists() and (ART / "gen3_results.json").exists():
        g3 = j(ART / "gen3_specs.json")["specs"]
        md.append("\n## Generation 3 (seeds 97, 98, 99; 30M tokens; CALIB-SEARCH)\n")
        md.append(gen_table(3, "gen3_results.json", g3))
        md.append("\n| Gen-3 | = Gen-2 | Gen-1 ancestor |\n|---|---|---|")
        g2m = {s["id"]: s for s in j(ART / "gen2_specs.json")["specs"]}
        for s in g3:
            md.append(f"| {s['id']} | {s['parent']} | {g2m[s['parent']]['parent']} |")
    # ledger
    rows = []
    for ln in (guard.ROOT / "ledger" / "gpu_time_ledger.tsv").read_text().splitlines()[1:]:
        p = ln.split("\t")
        if len(p) >= 6 and p[4].startswith("srsv1"):
            rows.append(p)
    by = {}
    for p in rows:
        stage = p[3].split("_srsv1_train_")[-1].split("_s")[0] if "train" in p[4] else p[4]
        by.setdefault(stage, [0, 0])
        by[stage][0] += int(p[2])
        by[stage][1] += 1
    md.append("\n## GPU-time ledger (purpose `srsv1 …`)\n\n| stage | entries | GPU seconds | GPU hours |\n|---|---|---|---|")
    for k, (s, n) in sorted(by.items()):
        md.append(f"| {k} | {n} | {s} | {s / 3600:.2f} |")
    tot = sum(int(p[2]) for p in rows)
    md.append(f"| **total** | {len(rows)} | {tot} | **{tot / 3600:.2f}** |")
    if rows:
        md.append(f"\nGPU wall span: {rows[0][0]} → {rows[-1][1]}; non-COMPLETE entries: "
                  f"{[(p[3], p[5]) for p in rows if p[5] not in ('COMPLETE', 'PASS')]}")
    # failures
    fl = tsv("failures.tsv")
    md.append("\n## Failures registry\n")
    md.append("\n".join(f"- {' | '.join(x)}" for x in fl) or "- none")
    out = ST / "report_tables.md"
    out.write_text("\n".join(md) + "\n")
    # inventory of configs / runs / checkpoints
    inv_p = ART / "run_inventory.json"
    if (ART / "final_analysis.json").exists() and not inv_p.exists():
        inv = {"configs": {}, "runs": {}}
        for c in sorted((guard.ROOT / "configs/sequence-routing-search-v1").rglob("*.json")):
            inv["configs"][str(c.relative_to(guard.ROOT))] = guard.sha256_file(c)
        for reg in ("gen1_runs.tsv", "gen2_runs.tsv", "gen3_runs.tsv", "final_runs.tsv"):
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
            inv.setdefault("artifacts", {})[a.name] = guard.sha256_file(a)
        guard.atomic_write_json(inv_p, inv, mode=0o400)
        guard.ledger("SRS_V1_RUN_INVENTORY", f"{inv_p.relative_to(guard.ROOT)} sha256={guard.sha256_file(inv_p)}")
    print(out)


if __name__ == "__main__":
    main()
