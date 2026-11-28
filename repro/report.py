"""Stage `report`: collect every eval.json into results/ (REPORT.md, PAPER.md, per_seed.csv, figures/).

REPORT.md   one row per doc claim: our paired-seed number, the doc's number, PASS / FAIL / PENDING, and
            "not run" for claims whose experiments are not part of the run (e.g. the §18 EDL runs in the
            paper tiers). Absolute CE values will not match the doc (different data and tokenizer); signs,
            paired deltas and the semantic-vs-shuffled contrasts are what a reproduction can confirm.
PAPER.md    the §23 / §27 paper analysis: every comparison as a paired-over-seeds mean, 95% t-interval, n
            and sign count; main table, B4 (gap vs base competence, with a slope test), B6, B9, and the
            paper_extra sections (B8, explicit-domain oracle, router inputs) when those runs exist; the swap
            matrix heatmap and router-by-position table.
per_seed.csv  every numeric scalar of every eval.json: experiment, seed, metric, value.
figures/    PNG + PDF.

Missing runs never crash the report: a section without data says so.
"""
from __future__ import annotations

import csv
import math
import os
import traceback
from pathlib import Path

import numpy as np

from .config import EXPERIMENTS, Profile
from .eval.metrics import paired, summarize
from .utils import log, read_json, results_dir, run_dir, write_json

FRACS = (("10%", "_at10"), ("25%", "_at25"), ("50%", "_at50"), ("100%", ""))


# ============================================================================== loading

def run_seeds(p: Profile, exp_name: str) -> list[int]:
    """Seeds of an experiment that have an eval.json, from the run directories (not from the config: a seed
    may come from any tier)."""
    d = run_dir(p, exp_name, 0).parent
    if not d.exists():
        return []
    out = []
    for c in d.iterdir():
        if c.name.startswith("s") and c.name[1:].isdigit() and (c / "eval.json").exists():
            out.append(int(c.name[1:]))
    return sorted(out)


def load(p: Profile, exp_name: str, seed: int) -> dict | None:
    f = run_dir(p, exp_name, seed) / "eval.json"
    try:
        return read_json(f) if f.exists() else None
    except Exception:                              # a half-written file must not kill the report
        return None


def get(d, key: str):
    for part in key.split("."):
        d = d.get(part) if isinstance(d, dict) else None
    return d


def collect(p: Profile, exp_name: str, key: str) -> dict:
    """{seed: value} of a (possibly nested 'a.b') numeric key across the finished seeds of an experiment."""
    out = {}
    for s in run_seeds(p, exp_name):
        v = get(load(p, exp_name, s), key)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
            out[s] = float(v)
    return out


def have(p: Profile, *names: str) -> bool:
    return any(run_seeds(p, n) for n in names)


# ============================================================================== statistics

def stat(d: dict) -> dict:
    """Mean, 95% t-interval, n and sign counts of {seed: value}."""
    v = np.asarray([d[k] for k in sorted(d)], dtype=np.float64)
    n = len(v)
    out = dict(n=n, mean=float("nan"), lo=float("nan"), hi=float("nan"), pos=int((v > 0).sum()),
               neg=int((v < 0).sum()), per_seed={str(k): float(d[k]) for k in sorted(d)})
    if n:
        out["mean"] = float(v.mean())
    if n > 1:
        from scipy import stats
        se = v.std(ddof=1) / math.sqrt(n)
        t = stats.t.ppf(0.975, n - 1)
        out["lo"], out["hi"] = float(v.mean() - t * se), float(v.mean() + t * se)
    return out


def pair(a: dict, b: dict) -> dict:
    """Paired-over-seeds difference a - b on the seeds both have."""
    return stat({s: a[s] - b[s] for s in sorted(set(a) & set(b))})


def num(x, digits=5) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "–"
    return f"{x:+.{digits}f}"


def show(st: dict, good: str = ">", digits: int = 5) -> str:
    """'+0.00123 [+0.00100, +0.00150], n=4, 4/4 > 0' ("good" = the sign the claim predicts)."""
    if not st or st["n"] == 0:
        return "not run"
    k = st["pos"] if good == ">" else st["neg"]
    ci = f" [{num(st['lo'], digits)}, {num(st['hi'], digits)}]" if st["n"] > 1 else ""
    return f"{num(st['mean'], digits)}{ci}, n={st['n']}, {k}/{st['n']} {good} 0"


def ci_excludes_zero(st: dict, good: str = ">") -> bool:
    if not st or st["n"] < 2 or not math.isfinite(st["lo"]):
        return False
    return st["lo"] > 0 if good == ">" else st["hi"] < 0


def md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or np.ptp(x) == 0:
        return float("nan")
    return float(np.polyfit(x, y, 1)[0])


# ============================================================================== claims (REPORT.md)

class Claims:
    def __init__(self):
        self.rows = []

    def add(self, section, claim, ours, doc, verdict):
        self.rows.append(dict(section=section, claim=claim, ours=ours, doc=doc, verdict=verdict))

    def md(self):
        return md_table(["§", "claim", "ours", "doc", "verdict"],
                        [[r["section"], r["claim"], r["ours"], r["doc"], f"**{r['verdict']}**"] for r in self.rows])


def verdict(cond, have_data, run=True) -> str:
    if not run:
        return "not run"
    if not have_data:
        return "PENDING"
    return "PASS" if cond else "FAIL"


def describe_paired(d: dict) -> str:
    if not d or d.get("n", 0) == 0:
        return "–"
    s = f"{d['mean']:+.5f}"
    if "ci95" in d:
        s += f" [{d['ci95'][0]:+.5f}, {d['ci95'][1]:+.5f}]"
    return s + f" (n={d['n']})"


def claims(p: Profile) -> Claims:
    C = Claims()
    # ---- §18: only if the EDL family is part of the run
    edl_run = have(p, "base.edl", "base.sem_c6", "base.shuf_c6")
    ce = {n: collect(p, n, "val_ce") for n in ("base.token4", "base.edl", "base.sem_c6", "base.shuf_c6")}
    d_edl = paired(ce["base.edl"], ce["base.token4"])
    d_sem = paired(ce["base.sem_c6"], ce["base.token4"])
    d_sem_shuf = paired(ce["base.sem_c6"], ce["base.shuf_c6"])
    C.add("§18", "explicit EDL lowers val CE vs TOKEN-4", describe_paired(d_edl), "−0.0098 (4/4)",
          verdict(d_edl.get("mean", 1) < 0 and d_edl.get("neg_frac", 0) >= 0.75, d_edl["n"] >= 2, edl_run))
    C.add("§18", "Semantic C6 lowers val CE vs TOKEN-4", describe_paired(d_sem), "−0.0066",
          verdict(d_sem.get("mean", 1) < 0, d_sem["n"] >= 2, edl_run))
    C.add("§18", "Semantic C6 vs shuffled C6 CE (doc: ~tie)", describe_paired(d_sem_shuf), "+0.0006",
          verdict(abs(d_sem_shuf.get("mean", 1)) < 0.003, d_sem_shuf["n"] >= 2, edl_run))
    nmi = {n: collect(p, n, "nmi_domain") for n in ce}
    d_nmi = paired(nmi["base.sem_c6"], nmi["base.shuf_c6"])
    C.add("§18", "Semantic C6 routes more by domain than shuffled C6 (NMI, every seed)", describe_paired(d_nmi),
          "0.0547 vs 0.0452", verdict(d_nmi.get("pos_frac", 0) == 1.0, d_nmi["n"] >= 2, edl_run))
    d_nmi_edl = paired(nmi["base.edl"], nmi["base.token4"])
    C.add("§18", "EDL raises domain NMI vs TOKEN-4", describe_paired(d_nmi_edl), "0.0574 vs 0.0126",
          verdict(d_nmi_edl.get("mean", -1) > 0, d_nmi_edl["n"] >= 2, edl_run))
    inter = {n: collect(p, n, "interaction_domain.mean") for n in ce}
    d_int = paired(inter["base.sem_c6"], inter["base.shuf_c6"])
    C.add("§18", "Semantic − shuffled per-expert interaction ≈ 0 (no functional gain)", describe_paired(d_int),
          "+2.3e-6 (2/4 seeds)", verdict(abs(d_int.get("mean", 1)) < 1e-4, d_int["n"] >= 2, edl_run))
    matched = {n: collect(p, n, "matched_domain.mean") for n in ce}
    d_m = paired(matched["base.edl"], matched["base.token4"])
    C.add("§21", "EDL exposure-matched interaction not above TOKEN-4", describe_paired(d_m), "lower than TOKEN-4",
          verdict(d_m.get("mean", 1) <= 0 or (d_m.get("ci95") or [0, 0])[0] < 0, d_m["n"] >= 2, edl_run))

    # ---- §23
    imp = {n: collect(p, n, "improvement") for n in ("bank.sem", "bank.shuf", "bank.tok")}
    s_sem = summarize(imp["bank.sem"].values())
    C.add("§23", "semantic oracle bank improves on the frozen base", num(s_sem["mean"]), "+0.00337",
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
    lo = [(get(load(p, "bank.sem", s), "swap_own.ci") or [float("nan")])[0] for s in spec_sem]
    C.add("§23", "semantic specialists are functionally specific (swap matrix, CI > 0)",
          f"{summarize(spec_sem.values())['mean']:.5f}" if spec_sem else "–", "0.00484",
          verdict(bool(lo) and all(x > 0 for x in lo), bool(spec_sem)))
    C.add("§23", "shuffled specialists are not (≈ 0); semantic − shuffled > 0", describe_paired(d_spec),
          "0.00002; diff 0.00482", verdict(d_spec.get("pos_frac", 0) == 1.0, d_spec["n"] >= 2))
    own_best = summarize(collect(p, "bank.sem", "swap_own.own_best_frac").values())
    C.add("§23", "own specialist is best for its cluster (share of clusters)", num(own_best["mean"], 3), "all",
          verdict(own_best["mean"] >= 0.9, own_best["n"] >= 1))

    # ---- §27
    rg = collect(p, "router.sem", "router_minus_g")
    s_rg = summarize(rg.values())
    C.add("§27", "causal router over semantic bank beats generic bank G (every router seed)",
          num(s_rg["mean"]) + f" (n={s_rg['n']})", "−0.0038…−0.0041 (3/3)",
          verdict(s_rg["n"] > 0 and all(v < 0 for v in rg.values()), s_rg["n"] >= 1))
    ret = retention(p)
    s_ret = summarize(ret.values())
    C.add("§27", "router retains most of the oracle's semantic-over-shuffled advantage", num(s_ret["mean"], 3),
          "0.744–0.748", verdict(s_ret["mean"] > 0.5, s_ret["n"] >= 1))
    mi_sem = summarize(collect(p, "router.sem", "matched_interaction.mean").values())
    mi_shuf = summarize(collect(p, "router.shuf", "matched_interaction.mean").values())
    C.add("§27", "exposure-matched interaction: semantic ≫ shuffled",
          f"{num(mi_sem['mean'])} vs {num(mi_shuf['mean'])}", "0.0092–0.0094 vs 0.0001–0.0005",
          verdict(mi_sem["mean"] > mi_shuf["mean"] + 1e-3, mi_sem["n"] >= 1 and mi_shuf["n"] >= 1))
    n_sem = summarize(collect(p, "router.sem", "routing_nmi_sem").values())
    n_shuf = summarize(collect(p, "router.shuf", "routing_nmi_sem").values())
    C.add("§27", "router choices align with semantic clusters (NMI) far more than shuffled",
          f"{num(n_sem['mean'], 3)} vs {num(n_shuf['mean'], 3)}", "0.395 vs 0.048",
          verdict(n_sem["mean"] > 2 * n_shuf["mean"], n_sem["n"] >= 1 and n_shuf["n"] >= 1))
    return C


def retention(p: Profile) -> dict:
    """Per router seed: (r_shuf − r_sem) / (o_shuf − o_sem), the share of the oracle's semantic-over-shuffled
    CE advantage the causal router keeps."""
    r_sem, r_shuf = collect(p, "router.sem", "router_ce"), collect(p, "router.shuf", "router_ce")
    o_sem, o_shuf = collect(p, "router.sem", "oracle_ce"), collect(p, "router.shuf", "oracle_ce")
    return {s: (r_shuf[s] - r_sem[s]) / (o_shuf[s] - o_sem[s])
            for s in sorted(set(r_sem) & set(r_shuf) & set(o_sem) & set(o_shuf)) if abs(o_shuf[s] - o_sem[s]) > 1e-9}


def mean_table(p, names, keys) -> str:
    rows = []
    for n in names:
        if not run_seeds(p, n):
            continue
        vals, n_seeds = [], 0
        for k in keys:
            c = collect(p, n, k)
            n_seeds = max(n_seeds, len(c))
            s = summarize(c.values())
            vals.append("–" if s["n"] == 0 else f"{s['mean']:.5f}" + (f" ± {s['se']:.5f}" if s["n"] > 1 else ""))
        rows.append([n, n_seeds] + vals)
    if not rows:
        return "_no finished runs_"
    return md_table(["experiment", "seeds"] + keys, rows)


# ============================================================================== figures

def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


def save_fig(fig, figdir: Path, name: str) -> list[str]:
    figdir.mkdir(parents=True, exist_ok=True)
    out = []
    for ext in ("png", "pdf"):
        f = figdir / f"{name}.{ext}"
        fig.savefig(f, dpi=200, bbox_inches="tight")
        out.append(f.name)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return out


# ============================================================================== paper sections

def section_main(p, out) -> str:
    imp = {n: collect(p, n, "improvement") for n in ("bank.sem", "bank.shuf", "bank.tok")}
    spec = {n: collect(p, n, "swap_own.specificity") for n in ("bank.sem", "bank.shuf")}
    rg = {n: collect(p, n, "router_minus_g") for n in ("router.sem", "router.shuf")}
    ret = retention(p)
    rows = [
        ["§23", "semantic bank improves on the frozen base (base CE − bank CE)", show(stat(imp["bank.sem"])), "+0.00337"],
        ["§23", "semantic − shuffled improvement", show(pair(imp["bank.sem"], imp["bank.shuf"])), "+0.00240 (4/4)"],
        ["§23", "semantic − token-routed (G) improvement", show(pair(imp["bank.sem"], imp["bank.tok"])), "+0.00127 (4/4)"],
        ["§23", "shuffled bank improvement (context)", show(stat(imp["bank.shuf"])), "+0.00097"],
        ["§23", "token-routed G improvement (context)", show(stat(imp["bank.tok"])), "+0.00210"],
        ["§23", "exposure-matched specificity, semantic", show(stat(spec["bank.sem"])), "0.00484"],
        ["§23", "specificity semantic − shuffled", show(pair(spec["bank.sem"], spec["bank.shuf"])), "0.00482"],
        ["§27", "router (semantic bank) − G, CE (< 0 = router better)", show(stat(rg["router.sem"]), "<"),
         "−0.0038…−0.0041 (3/3)"],
        ["§27", "router (shuffled bank) − G, CE", show(stat(rg["router.shuf"]), "<"), "–"],
        ["§27", "share of the oracle's sem-over-shuf advantage retained", show(stat(ret), ">", 3), "0.744–0.748"],
        ["§27", "exposure-matched interaction, semantic − shuffled router",
         show(pair(collect(p, "router.sem", "matched_interaction.mean"),
                   collect(p, "router.shuf", "matched_interaction.mean"))), "0.0092 vs 0.0001–0.0005"],
    ]
    out["main"] = dict(improvement={k: stat(v) for k, v in imp.items()},
                       sem_minus_shuf=pair(imp["bank.sem"], imp["bank.shuf"]),
                       sem_minus_tok=pair(imp["bank.sem"], imp["bank.tok"]),
                       specificity={k: stat(v) for k, v in spec.items()},
                       router_minus_g={k: stat(v) for k, v in rg.items()}, retention=stat(ret))
    return ("## Main results (§23 oracle specialists, §27 causal router)\n\n"
            "Paired over seeds: mean [95% t-interval], n, and how many seeds have the predicted sign. "
            "§23 seeds are bank seeds (all on TOKEN-4 seed 0); §27 seeds are router seeds over bank seed 0. "
            "Improvements are base CE − bank CE in nats/token on val (higher is better).\n\n"
            + md_table(["§", "comparison", "ours", "doc"], rows))


def section_b4(p, out, figdir) -> str:
    rows, per = [], {}
    for label, suf in FRACS:
        names = {r: f"bank.{r}" if not suf else f"abl.bank_{r}{suf}" for r in ("sem", "shuf", "tok")}
        imp = {r: collect(p, n, "improvement") for r, n in names.items()}
        base_ce = collect(p, names["sem"], "base_ce") or collect(p, names["tok"], "base_ce")
        st_tok, st_shuf = pair(imp["sem"], imp["tok"]), pair(imp["sem"], imp["shuf"])
        per[label] = dict(base_ce=base_ce, sem_tok={s: imp["sem"][s] - imp["tok"][s] for s in set(imp["sem"]) & set(imp["tok"])},
                          sem_shuf={s: imp["sem"][s] - imp["shuf"][s] for s in set(imp["sem"]) & set(imp["shuf"])},
                          st_tok=st_tok, st_shuf=st_shuf, imp={r: stat(v) for r, v in imp.items()})
        bce = np.mean(list(base_ce.values())) if base_ce else float("nan")
        rows.append([label, f"{bce:.4f}" if math.isfinite(bce) else "–",
                     num(per[label]["imp"]["sem"]["mean"]), num(per[label]["imp"]["shuf"]["mean"]),
                     num(per[label]["imp"]["tok"]["mean"]), show(st_tok), show(st_shuf)])
    if not any(per[l]["st_tok"]["n"] or per[l]["st_shuf"]["n"] for l in per):
        return "## B4 — does the semantic advantage need a competent base?\n\n_not run yet_"

    # slope of the gap against base CE, per seed over the checkpoints it has, then across seeds
    def slopes(key):
        sl = {}
        seeds = set.union(*[set(per[l][key]) for l in per])
        for s in seeds:
            pts = [(per[l]["base_ce"].get(s, np.mean(list(per[l]["base_ce"].values())) if per[l]["base_ce"] else np.nan),
                    per[l][key][s]) for l in per if s in per[l][key]]
            pts = [(x, y) for x, y in pts if math.isfinite(x)]
            if len(pts) >= 2:
                sl[s] = ols_slope([x for x, _ in pts], [y for _, y in pts])
        return {s: v for s, v in sl.items() if math.isfinite(v)}

    sl_tok, sl_shuf = stat(slopes("sem_tok")), stat(slopes("sem_shuf"))
    out["b4"] = dict(rows=rows, slope_sem_minus_tok_vs_base_ce=sl_tok, slope_sem_minus_shuf_vs_base_ce=sl_shuf)

    def verdict_text(st, what):
        if st["n"] == 0:
            return f"- {what}: not enough checkpoints yet."
        grows = st["mean"] < 0
        sure = ci_excludes_zero(st, "<") if grows else ci_excludes_zero(st, ">")
        direction = "grows as the base gets more competent (lower CE)" if grows else "shrinks as the base gets more competent"
        return (f"- {what}: slope vs base val CE = {show(st, '<')}. The gap {direction}; "
                + ("the 95% interval excludes 0." if sure else "the 95% interval includes 0, so the trend is not established."))

    text = ["## B4 — does the semantic advantage need a competent base?", "",
            "Banks trained on TOKEN-4 seed 0 at 10/25/50/100% of its training. *base val CE* is the frozen "
            "checkpoint's CE on val, recorded by each bank's evaluation. Gaps are paired over bank seeds.", "",
            md_table(["base checkpoint", "base val CE", "sem impr.", "shuf impr.", "tok impr.", "sem − tok", "sem − shuf"], rows),
            "", "**Does the gap grow with base competence?** Per bank seed: OLS slope of the gap against base val CE "
            "across checkpoints, then a t-interval over seeds. A negative slope means the gap is larger for a more "
            "competent (lower-CE) base.", "",
            verdict_text(sl_tok, "sem − tok"), verdict_text(sl_shuf, "sem − shuf")]

    plt = _plt()
    if plt is not None:
        xs = np.arange(len(FRACS))
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        for key, lab, mk in (("st_tok", "semantic − token-routed (G)", "o"), ("st_shuf", "semantic − shuffled", "s")):
            m = np.array([per[l][key]["mean"] for l, _ in FRACS])
            lo = np.array([per[l][key]["lo"] for l, _ in FRACS])
            hi = np.array([per[l][key]["hi"] for l, _ in FRACS])
            err = np.vstack([np.where(np.isfinite(lo), m - lo, 0), np.where(np.isfinite(hi), hi - m, 0)])
            ax.errorbar(xs, m, yerr=err, marker=mk, capsize=3, label=lab)
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_xticks(xs, [l for l, _ in FRACS])
        ax.set_xlabel("TOKEN-4 checkpoint (fraction of training)")
        ax.set_ylabel("improvement gap (nats/token)")
        top = ax.secondary_xaxis("top")
        top.set_xticks(xs, [f"{np.mean(list(per[l]['base_ce'].values())):.3f}" if per[l]["base_ce"] else "–" for l, _ in FRACS])
        top.set_xlabel("base val CE")
        ax.legend(fontsize=8)
        text += ["", "Figure: " + ", ".join(f"`figures/{f}`" for f in save_fig(fig, figdir, "b4_gap_vs_base"))]
    return "\n".join(text)


def section_b6(p, out) -> str:
    sem = collect(p, "bank.sem", "improvement")
    dc, dp = collect(p, "abl.bank_dense_compute", "improvement"), collect(p, "abl.bank_dense_params", "improvement")
    if not dc and not dp:
        return "## B6 — routing or capacity?\n\n_not run yet_"
    rows = [["semantic bank (16 × 256, one active)", show(stat(sem))],
            ["dense, compute-matched (one 256-unit MLP, always on)", show(stat(dc))],
            ["dense, parameter-matched (one 4096-unit MLP, always on)", show(stat(dp))],
            ["semantic − compute-matched", show(pair(sem, dc))],
            ["semantic − parameter-matched", show(pair(sem, dp))]]
    out["b6"] = dict(sem_minus_compute=pair(sem, dc), sem_minus_params=pair(sem, dp))
    return ("## B6 — routing or capacity?\n\nImprovement over the frozen base, paired by bank seed. If the semantic "
            "bank beats the parameter-matched dense residual, its gain is not just extra capacity.\n\n"
            + md_table(["condition", "improvement / difference"], rows))


def section_b9(p, out) -> str:
    rows, d = [], {"sem": {}, "shuf": {}}
    for r in ("sem", "shuf"):
        v0 = collect(p, f"router.{r}", "router_minus_g").get(0)
        if v0 is not None:
            d[r][0] = v0
        for b in (1, 2, 3):
            v = collect(p, f"abl.router_{r}_bank{b}", "router_minus_g").get(0)
            if v is not None:
                d[r][b] = v
    if not d["sem"] and not d["shuf"]:
        return "## B9 — does the router result replicate across specialist banks?\n\n_not run yet_"
    for b in range(4):
        rows.append([f"bank seed {b}", num(d["sem"].get(b)), num(d["shuf"].get(b))])
    rows.append(["**across bank seeds**", show(stat(d["sem"]), "<"), show(stat(d["shuf"]), "<")])
    out["b9"] = dict(router_minus_g_by_bank_seed=d, sem=stat(d["sem"]), shuf=stat(d["shuf"]))
    return ("## B9 — does the router result replicate across specialist banks?\n\nRouter − G CE (< 0 = the causal "
            "router beats the generic bank) for router seed 0 over each independently trained bank; G is the "
            "token-routed bank with the same bank seed.\n\n"
            + md_table(["bank", "semantic router − G", "shuffled router − G"], rows))


def section_extras(p, out) -> str:
    parts = []
    # B8: other base seeds
    if have(p, *[f"abl.bank_{r}_base{b}" for b in (1, 2, 3) for r in ("sem", "shuf", "tok")]):
        rows, g_tok, g_shuf = [], {}, {}
        for b in range(4):
            names = {r: (f"bank.{r}" if b == 0 else f"abl.bank_{r}_base{b}") for r in ("sem", "shuf", "tok")}
            v = {r: collect(p, n, "improvement").get(0) for r, n in names.items()}
            ce = collect(p, names["sem"], "base_ce").get(0)
            if v["sem"] is not None and v["tok"] is not None:
                g_tok[b] = v["sem"] - v["tok"]
            if v["sem"] is not None and v["shuf"] is not None:
                g_shuf[b] = v["sem"] - v["shuf"]
            rows.append([f"TOKEN-4 seed {b}", f"{ce:.4f}" if ce is not None else "–", num(v["sem"]),
                         num(g_tok.get(b)), num(g_shuf.get(b))])
        rows.append(["**across bases**", "", "", show(stat(g_tok)), show(stat(g_shuf))])
        out["b8"] = dict(sem_minus_tok=stat(g_tok), sem_minus_shuf=stat(g_shuf))
        parts.append("## B8 — does it depend on the particular base?\n\nBank seed 0 on each TOKEN-4 seed.\n\n"
                     + md_table(["base", "base val CE", "sem impr.", "sem − tok", "sem − shuf"], rows))
    if have(p, "abl.bank_domain"):
        sem, dom = collect(p, "bank.sem", "improvement"), collect(p, "abl.bank_domain", "improvement")
        out["bank_domain"] = dict(domain=stat(dom), sem_minus_domain=pair(sem, dom))
        parts.append("## Explicit-domain oracle (from B7)\n\nFive specialists, one per arXiv domain, same total width "
                     "as the 16 semantic specialists.\n\n"
                     + md_table(["condition", "improvement / difference"],
                                [["5-domain oracle bank", show(stat(dom))], ["semantic (16) − domain (5)", show(pair(sem, dom))]]))
    if have(p, "abl.router_token_only", "abl.router_ctx_only"):
        full = collect(p, "router.sem", "router_minus_g").get(0)
        tok = collect(p, "abl.router_token_only", "router_minus_g").get(0)
        ctx = collect(p, "abl.router_ctx_only", "router_minus_g").get(0)
        out["router_inputs"] = dict(full=full, token_only=tok, context_only=ctx)
        parts.append("## Router inputs (from B10)\n\nRouter seed 0 over bank seed 0; router − G CE (< 0 = better than G).\n\n"
                     + md_table(["router input", "router − G"],
                                [["current token + context mean (main)", num(full)], ["current token only", num(tok)],
                                 ["context mean only", num(ctx)]]))
    return "\n\n".join(parts)


def section_swap(p, out, figdir) -> str:
    mats = {}
    for n in ("bank.sem", "bank.shuf"):
        seeds = run_seeds(p, n)
        if seeds:
            m = get(load(p, n, seeds[0]), "swap_own.swap_matrix")
            if m:
                mats[n] = (seeds[0], np.asarray([[np.nan if x is None else x for x in row] for row in m], float))
    if not mats:
        return "## Swap matrix\n\n_not run yet_"
    text = ["## Swap (forced-exposure) matrix", "",
            "Every val probe token is pushed through every specialist. Cell (cluster g, specialist j) shows the "
            "double-centred utility −CE (row and column means removed), so the diagonal is the specificity: a "
            "positive diagonal means a cluster's own specialist is better for it than its overall quality predicts."]
    plt = _plt()
    if plt is not None:
        fig, axes = plt.subplots(1, len(mats), figsize=(5.4 * len(mats), 4.4), squeeze=False, constrained_layout=True)
        for ax, (n, (seed, M)) in zip(axes[0], mats.items()):
            ok = ~np.isnan(M).all(1)                  # clusters with no val tokens are left blank
            U = np.full_like(M, np.nan)
            Mo = -M[ok][:, ok]
            U[np.ix_(ok, ok)] = Mo - Mo.mean(1, keepdims=True) - Mo.mean(0, keepdims=True) + Mo.mean()
            v = np.nanmax(np.abs(U)) or 1.0
            im = ax.imshow(U, cmap="RdBu_r", vmin=-v, vmax=v)
            k = U.shape[0]
            step = 1 if k <= 16 else max(1, k // 8)
            ax.set_xticks(range(0, k, step))
            ax.set_yticks(range(0, k, step))
            ax.set_title(f"{n} (seed {seed})", fontsize=9)
            ax.set_xlabel("specialist")
            ax.set_ylabel("cluster (own label)")
            cb = fig.colorbar(im, ax=ax, shrink=0.85)
            cb.set_label("double-centred utility (nats)", fontsize=8)
            cb.formatter.set_powerlimits((-2, 2))
            cb.update_ticks()
        text += ["", "Figure: " + ", ".join(f"`figures/{f}`" for f in save_fig(fig, figdir, "swap_matrix"))]
    return "\n".join(text)


def section_positions(p, out) -> str:
    cats = (("first32", "document start (first 32 positions of a window)"),
            ("after_boundary", "transition (≤ 64 tokens after a paper boundary)"),
            ("stable", "steady state (everything else)"))
    rows, have_any = [], False
    for key, label in cats:
        row = [label]
        for n in ("router.sem", "router.shuf"):
            st = stat(collect(p, n, f"by_position_router_minus_g.{key}"))
            have_any |= st["n"] > 0
            row.append(show(st, "<"))
        share = collect(p, "router.sem", f"position_token_share.{key}")
        row.append(f"{np.mean(list(share.values())):.3f}" if share else "–")
        rows.append(row)
    if not have_any:
        return "## Router by position\n\n_not run yet_"
    return ("## Router by position\n\nRouter − G CE by token position, over router seeds. *Transition* here is a "
            "paper boundary inside the 1,024-token window; the original also had topic switches inside a paper "
            "(region-level priors, RECONSTRUCTION.md), which our paper-level setup does not define.\n\n"
            + md_table(["position", "semantic router − G", "shuffled router − G", "token share"], rows))


# ============================================================================== CSV

def flatten(d, prefix=""):
    if isinstance(d, dict):
        for k, v in d.items():
            yield from flatten(v, f"{prefix}{k}.")
    elif isinstance(d, list):
        if d and all(isinstance(x, (int, float, list, type(None))) and not isinstance(x, bool) for x in d):
            for i, v in enumerate(d):
                yield from flatten(v, f"{prefix}{i}.")
    elif isinstance(d, (int, float)) and not isinstance(d, bool):
        if math.isfinite(d):
            yield prefix[:-1], float(d)


def write_csv(p: Profile, path: Path) -> int:
    n = 0
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "seed", "metric", "value"])
        for e in EXPERIMENTS:
            for s in run_seeds(p, e.name):
                for k, v in flatten({k: v for k, v in (load(p, e.name, s) or {}).items()
                                     if k not in ("seed",) and not k.startswith("cells_")}):
                    w.writerow([e.name, s, k, repr(v)])
                    n += 1
    return n


# ============================================================================== entry point

def write_report(p: Profile) -> None:
    out_dir = Path(os.environ.get("REPRO_REPORT_DIR", results_dir(p)))   # override: keep result folders read-only
    out_dir.mkdir(parents=True, exist_ok=True)
    figdir = out_dir / "figures"
    summary = {}

    def safe(fn, *a):
        try:
            return fn(*a)
        except Exception:
            log(f"report section {fn.__name__} failed:\n{traceback.format_exc()}")
            return f"## {fn.__name__}\n\n_this section failed to render: see the job log_"

    C = safe(claims, p)
    claims_md = C.md() if isinstance(C, Claims) else C
    md = [f"# Reproduction report ({p.name} profile)", "",
          "Generated by `python -m repro.run report`. Doc numbers are from docs/source_history.md. Ours are paired "
          "over seeds; brackets are 95% t-intervals over seeds. Claims whose experiments are not part of this run "
          "say **not run**. The paper analysis (§23 / §27, ablations, figures) is in `PAPER.md`.", "",
          "## Claims", "", claims_md, "",
          "## Stage 1 — base models", "",
          safe(mean_table, p, [e.name for e in EXPERIMENTS if e.kind == "base"],
               ["val_ce", "nmi_domain", "nmi_sem", "fsi_domain.mean", "fsi_sem.mean",
                "interaction_domain.mean", "matched_domain.mean"]), "",
          "## Stage 2 — residual banks (improvement = base CE − bank CE, higher is better)", "",
          safe(mean_table, p, [e.name for e in EXPERIMENTS if e.kind == "bank"],
               ["improvement", "base_ce", "swap_own.specificity", "swap_sem.specificity", "clusters_improved_frac"]), "",
          "## Stage 3 — routers (router − G < 0 means the router beats the generic bank)", "",
          safe(mean_table, p, [e.name for e in EXPERIMENTS if e.kind == "router"],
               ["router_minus_g", "refresh16_minus_g", "oracle_ce", "router_ce", "g_ce",
                "matched_interaction.mean", "routing_nmi_sem", "prefill_time_ratio"]), ""]
    (out_dir / "REPORT.md").write_text("\n".join(md))

    paper = {}
    sections = [safe(section_main, p, paper), safe(section_b4, p, paper, figdir), safe(section_b6, p, paper),
                safe(section_b9, p, paper), safe(section_extras, p, paper), safe(section_swap, p, paper, figdir),
                safe(section_positions, p, paper)]
    n_rows = safe(write_csv, p, out_dir / "per_seed.csv")
    pmd = [f"# Post-hoc subject specialists and their causal router ({p.name} profile)",
           "All comparisons are paired over seeds: mean [95% t-interval], n, and the number of seeds with the "
           "predicted sign. Sections appear when their runs exist. Every per-seed number is in `per_seed.csv` "
           f"({n_rows} rows); figures are in `figures/` (PNG and PDF)."]
    pmd += [s for s in sections if s]
    (out_dir / "PAPER.md").write_text("\n\n".join(pmd) + "\n")

    summary["claims"] = C.rows if isinstance(C, Claims) else []
    summary["paper"] = paper
    write_json(out_dir / "summary.json", summary)
    log(f"wrote {out_dir / 'REPORT.md'}, {out_dir / 'PAPER.md'}, {out_dir / 'per_seed.csv'}")
    if isinstance(C, Claims):
        print(C.md())
