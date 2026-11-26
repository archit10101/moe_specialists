"""csr-conditional-local-attention-v1: preregistered candidates and frozen selection / verdict rules
(docs/CONDITIONAL_LOCAL_ATTENTION_V1_PREREG.md sections 6-9). Pure functions of recorded results; no data access."""
from __future__ import annotations

BASELINE = "FULL-TOKEN-4"
CANDIDATES = [  # (id, condition letter, attention plan); the only candidates of this experiment
    ("LOCAL-W128", "B", {"kind": "local", "window": 128}),
    ("LOCAL-W256", "C", {"kind": "local", "window": 256}),
    ("LOCAL-W512", "D", {"kind": "local", "window": 512}),
    ("HYBRID-W256-LATE2", "E", {"kind": "hybrid", "window": 256, "n_full_late": 2}),
    ("HYBRID-W256-LATE4", "F", {"kind": "hybrid", "window": 256, "n_full_late": 4}),
    ("G128-25", "G", {"kind": "conditional", "window": 128, "target_rate": 0.25}),
    ("G256-10", "G", {"kind": "conditional", "window": 256, "target_rate": 0.10}),
    ("G256-25", "G", {"kind": "conditional", "window": 256, "target_rate": 0.25}),
    ("G256-50", "G", {"kind": "conditional", "window": 256, "target_rate": 0.50}),
    ("G512-25", "G", {"kind": "conditional", "window": 512, "target_rate": 0.25}),
]
BASELINE_PLAN = {"kind": "full"}
PLANS = {BASELINE: BASELINE_PLAN, **{c[0]: c[2] for c in CANDIDATES}}
CONDITION = {BASELINE: "A", **{c[0]: c[1] for c in CANDIDATES}}
SIMPLICITY = {"local": 0, "hybrid": 1, "conditional": 2, "full": -1}

GEN1_MAX_DELTA = 0.05  # eliminate if CE worse than the same-seed baseline by more than this
GEN1_SLOW_RATIO = 0.90  # eliminate if throughput <= 0.90 x baseline ...
GEN1_COMPENSATING = -0.010  # ... unless CE is better than the baseline by at least 0.010
GEN1_TIE = 0.002
GEN1_ADVANCE = 4
GEN2_TIE = 0.003
GEN2_ADVANCE = 2
SEL_TIE = 0.003
SPEED_TIE_REL = 0.01
GQF_TIE = 0.01
COMPARABLE_MEAN = 0.010
COMPARABLE_SEED = 0.020
BETTER_MEAN = -0.010
SPEEDUP_TIME_RATIO = 0.90  # candidate time / baseline time must be <= 0.90 (throughput >= baseline / 0.90)


def global_query_fraction(cid, measured_rate=None, n_layer=8):
    p = PLANS[cid]
    if p["kind"] == "local":
        return 0.0
    if p["kind"] == "hybrid":
        return p["n_full_late"] / n_layer
    if p["kind"] == "full":
        return 1.0
    return measured_rate


def valid(r):
    return bool(r and r.get("complete") and r.get("finite", True) and not r.get("collapsed", True)
                and r.get("ce") is not None)


def gen1_filter(results: dict):
    """results[id] = {complete, finite, collapsed, ce, train_tok_s, gpu_s}. Returns {id: (eliminated, reason)}."""
    base = results[BASELINE]
    out = {}
    for cid, _, _ in CANDIDATES:
        r = results.get(cid)
        if not valid(r):
            out[cid] = (True, "invalid: incomplete / non-finite / correctness failure / expert collapse")
            continue
        d = r["ce"] - base["ce"]
        if d > GEN1_MAX_DELTA:
            out[cid] = (True, f"CE {d:+.4f} > +{GEN1_MAX_DELTA} vs same-seed baseline")
            continue
        if r["train_tok_s"] <= GEN1_SLOW_RATIO * base["train_tok_s"] and d > GEN1_COMPENSATING:
            out[cid] = (True, f"throughput {r['train_tok_s'] / base['train_tok_s']:.3f}x baseline (<= 0.90) with "
                              f"no compensating accuracy benefit (dCE {d:+.4f} > {GEN1_COMPENSATING})")
            continue
        out[cid] = (False, "")
    return out


def rank_with_ties(items, key_ce, tie, tiebreak_key):
    """Repeatedly take the remaining item with the lowest CE; among items within `tie` nats of it, take the one with
    the smallest tiebreak_key. Deterministic (ids break exact ties)."""
    rem = sorted(items)
    order = []
    while rem:
        best = min(key_ce(i) for i in rem)
        group = [i for i in rem if key_ce(i) <= best + tie]
        pick = min(group, key=lambda i: (tiebreak_key(i), key_ce(i), i))
        order.append(pick)
        rem.remove(pick)
    return order


def gen1_advance(results: dict):
    elim = gen1_filter(results)
    surv = [c for c, (e, _) in elim.items() if not e]
    order = rank_with_ties(surv, lambda i: results[i]["ce"], GEN1_TIE, lambda i: results[i]["gpu_s"])
    return order[:GEN1_ADVANCE], order, elim


def paired_delta(per_seed_c: dict, per_seed_b: dict):
    seeds = sorted(per_seed_b)
    ds = [per_seed_c[s]["ce"] - per_seed_b[s]["ce"] for s in seeds]
    return sum(ds) / len(ds), ds


def gen2_advance(per_seed: dict, ids):
    """per_seed[id][seed] = run result. Valid = every seed valid. Rank by mean paired dCE; within 0.003 prefer the
    higher mean measured training throughput."""
    base = per_seed[BASELINE]
    ok = [i for i in ids if all(valid(per_seed[i].get(s)) for s in base)]
    md = {i: paired_delta(per_seed[i], base)[0] for i in ok}
    tput = {i: sum(per_seed[i][s]["train_tok_s"] for s in base) / len(base) for i in ok}
    order = rank_with_ties(ok, lambda i: md[i], GEN2_TIE, lambda i: -tput[i])
    return order[:GEN2_ADVANCE], order, md, tput, [i for i in ids if i not in ok]


def blind_select(sel_per_seed: dict, ids, speed: dict, gqf: dict):
    """CALIB-SELECTION choice of exactly one final candidate.
    1 smallest mean paired dCE; 2 within 0.003: higher measured end-to-end speed (relative ties < 1%);
    3 lower global-query fraction (ties < 0.01); 4 simpler (local < hybrid < conditional); 5 id."""
    base = sel_per_seed[BASELINE]
    ok = [i for i in ids if all(valid(sel_per_seed[i].get(s)) for s in base)]
    if not ok:
        return None, {}
    md = {i: paired_delta(sel_per_seed[i], base)[0] for i in ok}
    best = min(md.values())
    pool = [i for i in ok if md[i] <= best + SEL_TIE]
    smax = max(speed[i] for i in pool)
    pool = [i for i in pool if speed[i] >= smax * (1 - SPEED_TIE_REL)]
    gmin = min(gqf[i] for i in pool)
    pool = [i for i in pool if gqf[i] <= gmin + GQF_TIE]
    pool.sort(key=lambda i: (SIMPLICITY[PLANS[i]["kind"]], md[i], i))
    return pool[0], {"mean_paired_delta": md, "speed_tok_s": speed, "global_query_fraction": gqf, "valid": ok}


def accuracy_verdict(deltas_by_seed: dict):
    ds = list(deltas_by_seed.values())
    mean = sum(ds) / len(ds)
    comparable = mean <= COMPARABLE_MEAN and max(ds) <= COMPARABLE_SEED
    better = mean <= BETTER_MEAN and all(d < 0 for d in ds)
    return {"mean_delta": mean, "per_seed": deltas_by_seed, "max_seed_delta": max(ds), "comparable": comparable,
            "better": better}


def speed_qualifies(time_ratio):
    """time_ratio = candidate time / baseline time (or baseline throughput / candidate throughput)."""
    return time_ratio is not None and time_ratio <= SPEEDUP_TIME_RATIO


def final_verdict(valid_experiment: bool, comparable: bool, speed: bool):
    if not valid_experiment:
        return "INVALID"
    if comparable and speed:
        return "SUCCESS"
    if comparable:
        return "ACCURACY-ONLY"
    if speed:
        return "SPEED-ONLY"
    return "NEGATIVE"
