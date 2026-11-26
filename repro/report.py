"""Stage `report`: collect every eval.json into results/REPORT.md and results/summary.json.

Each doc claim becomes one row: our paired-seed number, the doc's number, and PASS / FAIL / PENDING.
Absolute CE values will not match the doc (different data); signs, paired deltas and the
semantic-vs-shuffled contrasts are what a reproduction can confirm.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .config import EXPERIMENTS, Profile, get_experiment
from .eval.metrics import paired, summarize
from .utils import log, read_json, results_dir, run_dir, write_json


def collect(p: Profile, exp_name: str, key: str) -> dict:
    """{seed: value} of a (possibly nested 'a.b') key across the finished seeds of an experiment."""
    out = {}
    for s in get_experiment(exp_name).seeds:
        f = run_dir(p, exp_name, s) / "eval.json"
        if f.exists():
            v = read_json(f)
            for part in key.split("."):
                v = v.get(part) if isinstance(v, dict) else None
            if isinstance(v, (int, float)) and v is not None:
                out[s] = float(v)
    return out


def fmt(x, digits=5):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{x:+.{digits}f}" if abs(x) < 1 else f"{x:.4f}"


class Claims:
    def __init__(self):
        self.rows = []

    def add(self, section, claim, ours, doc, verdict):
        self.rows.append(dict(section=section, claim=claim, ours=ours, doc=doc, verdict=verdict))

    def md(self):
        lines = ["| § | claim | ours | doc | verdict |", "|---|---|---|---|---|"]
        for r in self.rows:
            lines.append(f"| {r['section']} | {r['claim']} | {r['ours']} | {r['doc']} | **{r['verdict']}** |")
        return "\n".join(lines)


def verdict(cond, have) -> str:
    if not have:
        return "PENDING"
    return "PASS" if cond else "FAIL"


def describe_paired(d: dict) -> str:
    if not d or d.get("n", 0) == 0:
        return "–"
    s = f"{d['mean']:+.5f}"
    if "ci95" in d:
        s += f" [{d['ci95'][0]:+.5f}, {d['ci95'][1]:+.5f}]"
    return s + f" (n={d['n']})"


def mean_table(p, names, keys) -> str:
    head = "| experiment | seeds | " + " | ".join(keys) + " |"
    lines = [head, "|" + "---|" * (len(keys) + 2)]
    for n in names:
        vals = []
        n_seeds = 0
        for k in keys:
            c = collect(p, n, k)
            n_seeds = max(n_seeds, len(c))
            s = summarize(c.values())
            vals.append("–" if s["n"] == 0 else f"{s['mean']:.5f}" + (f" ± {s['se']:.5f}" if s["n"] > 1 else ""))
        lines.append(f"| {n} | {n_seeds} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(p: Profile) -> None:
    out = Path(os.environ.get("REPRO_REPORT_DIR", results_dir(p)))   # override: keep result folders read-only
    out.mkdir(parents=True, exist_ok=True)
    C = Claims()
    summary = {}

    # ---------------------------------------------------------------- Stage 1: base
    ce = {n: collect(p, n, "val_ce") for n in ("base.token4", "base.edl", "base.sem_c6", "base.shuf_c6")}
    d_edl = paired(ce["base.edl"], ce["base.token4"])
    d_sem = paired(ce["base.sem_c6"], ce["base.token4"])
    d_sem_shuf = paired(ce["base.sem_c6"], ce["base.shuf_c6"])
    summary["edl_minus_token4_ce"] = d_edl
    C.add("§18", "explicit EDL lowers val CE vs TOKEN-4", describe_paired(d_edl), "−0.0098 (4/4)",
          verdict(d_edl.get("mean", 1) < 0 and d_edl.get("neg_frac", 0) >= 0.75, d_edl["n"] >= 2))
    C.add("§18", "Semantic C6 lowers val CE vs TOKEN-4", describe_paired(d_sem), "−0.0066",
          verdict(d_sem.get("mean", 1) < 0, d_sem["n"] >= 2))
    C.add("§18", "Semantic C6 vs shuffled C6 CE (doc: ~tie)", describe_paired(d_sem_shuf), "+0.0006",
          verdict(abs(d_sem_shuf.get("mean", 1)) < 0.003, d_sem_shuf["n"] >= 2))
    nmi = {n: collect(p, n, "nmi_domain") for n in ce}
    d_nmi = paired(nmi["base.sem_c6"], nmi["base.shuf_c6"])
    C.add("§18", "Semantic C6 routes more by domain than shuffled C6 (NMI, every seed)", describe_paired(d_nmi),
          "0.0547 vs 0.0452", verdict(d_nmi.get("pos_frac", 0) == 1.0, d_nmi["n"] >= 2))
    d_nmi_edl = paired(nmi["base.edl"], nmi["base.token4"])
    C.add("§18", "EDL raises domain NMI vs TOKEN-4", describe_paired(d_nmi_edl), "0.0574 vs 0.0126",
          verdict(d_nmi_edl.get("mean", -1) > 0, d_nmi_edl["n"] >= 2))
    inter = {n: collect(p, n, "interaction_domain.mean") for n in ce}
    d_int = paired(inter["base.sem_c6"], inter["base.shuf_c6"])
    C.add("§18", "Semantic − shuffled per-expert interaction ≈ 0 (no functional gain)", describe_paired(d_int),
          "+2.3e-6 (2/4 seeds)", verdict(abs(d_int.get("mean", 1)) < 1e-4, d_int["n"] >= 2))
    matched = {n: collect(p, n, "matched_domain.mean") for n in ce}
    d_m = paired(matched["base.edl"], matched["base.token4"])
    C.add("§21", "EDL exposure-matched interaction not above TOKEN-4", describe_paired(d_m), "lower than TOKEN-4",
          verdict(d_m.get("mean", 1) <= 0 or (d_m.get("ci95") or [0, 0])[0] < 0, d_m["n"] >= 2))

    # ---------------------------------------------------------------- Stage 2: banks
    imp = {n: collect(p, n, "improvement") for n in ("bank.sem", "bank.shuf", "bank.tok")}
    s_sem = summarize(imp["bank.sem"].values())
    C.add("§23", "semantic oracle bank improves on the frozen base", fmt(s_sem["mean"]), "+0.00337",
          verdict(s_sem["mean"] > 0 and s_sem["pos_frac"] == 1.0, s_sem["n"] >= 2))
    d_ab = paired(imp["bank.sem"], imp["bank.shuf"])
    d_ac = paired(imp["bank.sem"], imp["bank.tok"])
    C.add("§23", "semantic oracle beats shuffled oracle (every seed)", describe_paired(d_ab), "+0.00240 (4/4)",
          verdict(d_ab.get("pos_frac", 0) == 1.0, d_ab["n"] >= 2))
    C.add("§23", "semantic oracle beats learned token router (every seed)", describe_paired(d_ac), "+0.00127 (4/4)",
          verdict(d_ac.get("pos_frac", 0) == 1.0, d_ac["n"] >= 2))
    spec_sem = collect(p, "bank.sem", "swap_own.specificity")
    spec_shuf = collect(p, "bank.shuf", "swap_own.specificity")
    d_spec = paired(spec_sem, spec_shuf)
    lo = [read_json(run_dir(p, "bank.sem", s) / "eval.json").get("swap_own", {}).get("ci", [np.nan])[0]
          for s in spec_sem]
    C.add("§23", "semantic specialists are functionally specific (swap matrix, CI > 0)",
          f"{summarize(spec_sem.values())['mean']:.5f}" if spec_sem else "–", "0.00484",
          verdict(bool(lo) and all(x > 0 for x in lo), bool(spec_sem)))
    C.add("§23", "shuffled specialists are not (≈ 0); semantic − shuffled > 0", describe_paired(d_spec),
          "0.00002; diff 0.00482", verdict(d_spec.get("pos_frac", 0) == 1.0, d_spec["n"] >= 2))
    own_best = summarize(collect(p, "bank.sem", "swap_own.own_best_frac").values())
    C.add("§23", "own specialist is best for its cluster (share of clusters)", fmt(own_best["mean"], 3), "all",
          verdict(own_best["mean"] >= 0.9, own_best["n"] >= 1))

    # ---------------------------------------------------------------- Stage 3: router
    rg = collect(p, "router.sem", "router_minus_g")
    s_rg = summarize(rg.values())
    C.add("§27", "causal router over semantic bank beats generic bank G (every router seed)",
          fmt(s_rg["mean"]) + f" (n={s_rg['n']})", "−0.0038…−0.0041 (3/3)",
          verdict(s_rg["n"] > 0 and all(v < 0 for v in rg.values()), s_rg["n"] >= 1))
    r_sem, r_shuf = collect(p, "router.sem", "router_ce"), collect(p, "router.shuf", "router_ce")
    o_sem, o_shuf = collect(p, "router.sem", "oracle_ce"), collect(p, "router.shuf", "oracle_ce")
    ret = {s: (r_shuf[s] - r_sem[s]) / (o_shuf[s] - o_sem[s])
           for s in set(r_sem) & set(r_shuf) & set(o_sem) & set(o_shuf) if abs(o_shuf[s] - o_sem[s]) > 1e-9}
    s_ret = summarize(ret.values())
    summary["router_retention"] = ret
    C.add("§27", "router retains most of the oracle's semantic-over-shuffled advantage", fmt(s_ret["mean"], 3),
          "0.744–0.748", verdict(s_ret["mean"] > 0.5, s_ret["n"] >= 1))
    mi_sem = summarize(collect(p, "router.sem", "matched_interaction.mean").values())
    mi_shuf = summarize(collect(p, "router.shuf", "matched_interaction.mean").values())
    C.add("§27", "exposure-matched interaction: semantic ≫ shuffled",
          f"{fmt(mi_sem['mean'])} vs {fmt(mi_shuf['mean'])}", "0.0092–0.0094 vs 0.0001–0.0005",
          verdict(mi_sem["mean"] > mi_shuf["mean"] + 1e-3, mi_sem["n"] >= 1 and mi_shuf["n"] >= 1))
    n_sem = summarize(collect(p, "router.sem", "routing_nmi_sem").values())
    n_shuf = summarize(collect(p, "router.shuf", "routing_nmi_sem").values())
    C.add("§27", "router choices align with semantic clusters (NMI) far more than shuffled",
          f"{fmt(n_sem['mean'], 3)} vs {fmt(n_shuf['mean'], 3)}", "0.395 vs 0.048",
          verdict(n_sem["mean"] > 2 * n_shuf["mean"], n_sem["n"] >= 1 and n_shuf["n"] >= 1))

    # ---------------------------------------------------------------- tables
    md = [f"# Reproduction report ({p.name} profile)", "",
          "Generated by `python -m repro.run report`. Doc numbers are from docs/source_history.md. "
          "Ours are paired over seeds; brackets are 95% t-intervals over seeds.", "",
          "## Claims", "", C.md(), "",
          "## Stage 1 — base models", "",
          mean_table(p, [e.name for e in EXPERIMENTS if e.kind == "base"],
                     ["val_ce", "nmi_domain", "nmi_sem", "fsi_domain.mean", "fsi_sem.mean",
                      "interaction_domain.mean", "matched_domain.mean"]), "",
          "## Stage 2 — residual banks (improvement = base CE − bank CE, higher is better)", "",
          mean_table(p, [e.name for e in EXPERIMENTS if e.kind == "bank"],
                     ["improvement", "base_ce", "swap_own.specificity", "swap_sem.specificity",
                      "clusters_improved_frac"]), "",
          "## Stage 3 — routers (router − G < 0 means the router beats the generic bank)", "",
          mean_table(p, [e.name for e in EXPERIMENTS if e.kind == "router"],
                     ["router_minus_g", "refresh16_minus_g", "oracle_ce", "router_ce", "g_ce",
                      "matched_interaction.mean", "routing_nmi_sem",
                      "by_position_router_minus_g.first32", "by_position_router_minus_g.after_boundary",
                      "by_position_router_minus_g.stable", "prefill_time_ratio"]), "",
          "## B4 — does subject specialisation need a competent base?", "",
          b4_table(p), ""]
    (out / "REPORT.md").write_text("\n".join(md))
    summary["claims"] = C.rows
    write_json(out / "summary.json", summary)
    log(f"wrote {out / 'REPORT.md'}")
    print(C.md())


def b4_table(p: Profile) -> str:
    lines = ["| base checkpoint | base CE | sem | shuf | tok (G) | sem − tok | sem − shuf |",
             "|---|---|---|---|---|---|---|"]
    rows = [(f"{int(f * 100)}%", f"abl.bank_{{}}_at{int(f * 100)}") for f in (0.1, 0.25, 0.5)] + [("100%", "bank.{}")]
    for label, pat in rows:
        vals = {r: collect(p, pat.format(r), "improvement") for r in ("sem", "shuf", "tok")}
        base = summarize(collect(p, pat.format("sem"), "base_ce").values())["mean"]
        m = {r: summarize(v.values())["mean"] for r, v in vals.items()}
        lines.append(f"| {label} | {fmt(base)} | {fmt(m['sem'])} | {fmt(m['shuf'])} | {fmt(m['tok'])} | "
                     f"{describe_paired(paired(vals['sem'], vals['tok']))} | "
                     f"{describe_paired(paired(vals['sem'], vals['shuf']))} |")
    return "\n".join(lines)
