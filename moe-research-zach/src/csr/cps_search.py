"""csr-context-prior-specialization-v2 bounded search (protocol section 8): knob space, the fixed first wave, the progress
score, the diagnosis of every candidate (questions Q1-Q8) and the preregistered failure-response decision tree.
Pure functions of recorded results -> deterministic (test T-search). At most 12 candidates C01..C12."""
from __future__ import annotations

import json

MAX_CANDIDATES = 12

BASE = {
    "chunk": "fixed", "chunk_len": 64, "ctx_dim": 64, "mem_hls": [16.0, 128.0, 1024.0], "inject": "all",
    "alpha_init": -4.6, "alpha_max": 1.0, "alpha_l1": 0.0, "prior_norm": "match", "util_mode": "corr",
    "c_util": 1.0, "c_spec": 0.1, "c_kl": 1.0, "lam_max": 0.9, "schedule": "40/20/40", "r_cf": 0.075,
    "cf_phases": "all", "ctx_lr_mult": 1.0,
}
# allowed values (protocol 8.1); every candidate's knobs must lie in this space
SPACE = {
    "chunk": ["fixed", "adaptive"], "chunk_len": [16, 32, 64, 128], "ctx_dim": [32, 64, 128],
    "mem_hls": [[16.0, 128.0, 1024.0], [8.0, 64.0, 512.0], [32.0, 256.0, 2048.0]],
    "inject": ["all", "middle", "late"], "alpha_init": [-4.6, -2.2], "alpha_max": [1.0, 0.5],
    "alpha_l1": [0.0, 1e-3], "prior_norm": ["match", "none"], "util_mode": ["corr", "rank"],
    "c_util": [0.0, 1.0, 3.0], "c_spec": [0.0, 0.1, 0.3], "c_kl": [1.0, 3.0], "lam_max": [0.9, 0.5, 0.0],
    "schedule": ["40/20/40", "30/30/40"], "r_cf": [0.075], "cf_phases": ["all"], "ctx_lr_mult": [1.0, 3.0],
}
# first wave (fixed before any V2 result): M2 / M3 / M4 forms of the base
WAVE_A = [("C01", "M2 context prior (no utility, no specialisation loss)", {"c_util": 0.0, "c_spec": 0.0}),
          ("C02", "M3 utility context prior", {"c_spec": 0.0}),
          ("C03", "M4 specialised context prior (base)", {})]

# decision tree: (rule id, question, responses in order); each response is ONE knob change applied to the incumbent
RULES = [
    ("R1", "Q3 accuracy degraded during teacher-controlled training",
     [{"lam_max": 0.5}, {"lam_max": 0.0}, {"alpha_l1": 1e-3}]),
    ("R2", "Q4 degradation during distillation / student transfer",
     [{"c_kl": 3.0}, {"schedule": "30/30/40"}, {"lam_max": 0.5}]),
    ("R3", "Q2 token routing lost flexibility / prior costs accuracy",
     [{"alpha_l1": 1e-3}, {"inject": "late"}, {"alpha_max": 0.5}]),
    ("R4", "Q1 context prior inert",
     [{"alpha_init": -2.2}, {"prior_norm": "none"}, {"ctx_lr_mult": 3.0}]),
    ("R6", "Q6 specialisation present but not functional",
     [{"util_mode": "rank"}, {"c_util": 3.0}]),
    ("R5", "Q5 specialisation not increased",
     [{"c_spec": 0.3}, {"chunk_len": 128}, {"chunk": "adaptive"}]),
    ("R7", "all S1 criteria met: robustness refinements",
     [{"chunk_len": 32}, {"mem_hls": [8.0, 64.0, 512.0]}, {"ctx_dim": 128}]),
]

# thresholds (protocol 8.3)
TH = {"ce_tol": 0.01, "teacher_gap": 0.02, "distill_jump": 0.02, "alpha_high": 0.3, "overlap_high": 3.0,
      "alpha_inert": 0.03, "prior_gain_min": 0.002, "adv_ce": 0.02}


def apply(base, change):
    k = json.loads(json.dumps(base))
    k.update(json.loads(json.dumps(change)))
    for n, v in k.items():
        if n not in SPACE or v not in SPACE[n]:
            raise ValueError(f"knob {n}={v} outside the preregistered space")
    return k


def key(knobs):
    return json.dumps(knobs, sort_keys=True)


def score(rec, ref):
    """Progress toward BOTH targets (0 = both met at this scale; lower is better)."""
    dce = rec["ce"] - ref["ce"]
    return max(0.0, dce - TH["ce_tol"]) / TH["ce_tol"] + max(0.0, (ref["fsi_domain"] - rec["fsi_domain"]) /
                                                           ref["fsi_domain"])


def h_failure(rec):
    """Class-H optimisation failure: non-finite, dead/overloaded experts, collapsed routing entropy."""
    a = rec.get("audit") or {}
    load = a.get("load") or []
    flat = [x for row in load for x in row]
    E = 32
    return (rec.get("ce") is None) or (flat and (min(flat) < 0.1 / E or max(flat) > 4.0 / E)) or \
        (a.get("routing_entropy") is not None and not 0.2 <= a["routing_entropy"] <= 0.98)


def diagnose(rec, ref, t4_lm, m1=None, m5=None):
    """Answers Q1-Q8 for one S1 run. rec: dict(ce, ce_prior_off, fsi_domain, fsi_cluster, mi_domain, lm {step: lm},
    stats {alpha, prior_top4_overlap}, schedule, n_steps, teacher). ref: TOKEN-4 S1 dev results; t4_lm {step: lm}."""
    n = rec["n_steps"]
    common = sorted(s for s in rec["lm"] if s in t4_lm)

    def gap_at(frac):
        tgt = frac * n
        win = [s for s in common if abs(s - tgt) <= 0.04 * n]
        return sum(rec["lm"][s] - t4_lm[s] for s in win) / len(win) if win else None
    a_, b_, _ = [int(v) for v in rec["schedule"].split("/")]
    g1 = gap_at(a_ / 100) if rec["teacher"] else None
    g2 = gap_at((a_ + b_) / 100) if rec["teacher"] else None
    gend = gap_at(0.97)
    st = rec.get("stats") or {}
    dce = rec["ce"] - ref["ce"]
    q = {
        "Q1_prior_useful": {"alpha_end": st.get("alpha"), "ce_gain_from_prior": rec["ce_prior_off"] - rec["ce"],
                            "inert": (st.get("alpha") is not None and st["alpha"] < TH["alpha_inert"]) or
                            (rec["ce_prior_off"] - rec["ce"] < TH["prior_gain_min"])},
        "Q2_token_flexibility": {"prior_top4_overlap": st.get("prior_top4_overlap"), "alpha_end": st.get("alpha"),
                                 "lost": dce > TH["ce_tol"] and ((st.get("alpha") or 0) > TH["alpha_high"] or
                                                                  (st.get("prior_top4_overlap") or 0) >
                                                                  TH["overlap_high"])},
        "Q3_teacher_phase": {"lm_gap_p1_end": g1, "degraded": g1 is not None and g1 > TH["teacher_gap"]},
        "Q4_distillation": {"lm_gap_p2_end": g2, "lm_gap_end": gend,
                            "degraded": g2 is not None and gend is not None and gend - g2 > TH["distill_jump"]},
        "Q5_specialisation": {"fsi_domain": rec["fsi_domain"], "t4": ref["fsi_domain"],
                              "fsi_cluster": rec.get("fsi_cluster"), "t4_cluster": ref.get("fsi_cluster"),
                              "increased": rec["fsi_domain"] > ref["fsi_domain"]},
        "Q6_functional": {"mi_domain": rec.get("mi_domain"), "t4_mi": ref.get("mi_domain"),
                          "not_functional": (rec.get("mi_domain") or 0) > (ref.get("mi_domain") or 0) and
                          rec["fsi_domain"] <= ref["fsi_domain"]},
        "Q7_forecast_vs_memory": {"memory_only": m1},
        "Q8_shuffled": {"shuffled": m5},
        "ce_minus_t4": dce,
    }
    return q


def next_change(inc_knobs, q, tried):
    """First matching rule (fixed order) with an untried response -> (rule, change) or (None, None)."""
    met = q["ce_minus_t4"] <= TH["ce_tol"] and q["Q5_specialisation"]["increased"]
    match = {"R1": q["Q3_teacher_phase"]["degraded"], "R2": q["Q4_distillation"]["degraded"],
             "R3": q["Q2_token_flexibility"]["lost"], "R4": q["Q1_prior_useful"]["inert"],
             "R6": q["Q6_functional"]["not_functional"], "R5": not q["Q5_specialisation"]["increased"],
             "R7": met}
    for rid, _, resp in RULES:
        if not match[rid]:
            continue
        for ch in resp:
            try:
                k = apply(inc_knobs, ch)
            except ValueError:
                continue
            if key(k) not in tried:
                return rid, ch
    for rid, _, resp in RULES:  # nothing matched with an untried response: robustness refinements
        if rid != "R7":
            continue
        for ch in resp:
            k = apply(inc_knobs, ch)
            if key(k) not in tried:
                return rid, ch
    return None, None


def incumbent(cands, ref):
    """Lowest score; ties -> higher FSI_domain, then lower CE, then earlier id. Only finished, non-H runs."""
    ok = [c for c in cands if c.get("ce") is not None and not c.get("h_failure")]
    if not ok:
        return None
    return sorted(ok, key=lambda c: (score(c, ref), -c["fsi_domain"], c["ce"], c["id"]))[0]


def advance(cands, ref, k=3):
    """S1 -> S2: candidates with CE - T4 <= +0.02 AND FSI_domain >= T4 (moving toward both), best score first, at most
    k. If none qualifies, the single best-score candidate advances marked INELIGIBLE (confirmatory only)."""
    ok = [c for c in cands if c.get("ce") is not None and not c.get("h_failure")]
    q = [c for c in ok if c["ce"] - ref["ce"] <= TH["adv_ce"] and c["fsi_domain"] >= ref["fsi_domain"]]
    q = sorted(q, key=lambda c: (score(c, ref), -c["fsi_domain"], c["ce"], c["id"]))[:k]
    if q:
        return [(c["id"], True) for c in q]
    inc = incumbent(cands, ref)
    return [(inc["id"], False)] if inc else []


def s2_eligible(c_by_seed, t4_by_seed):
    """S2 dev eligibility over both dev seeds: mean CE gap <= +0.01, every seed <= +0.015, FSI_domain above the
    same-seed TOKEN-4 in every seed, no class-H failure."""
    gaps = [c_by_seed[s]["ce"] - t4_by_seed[s]["ce"] for s in t4_by_seed]
    fs = [c_by_seed[s]["fsi_domain"] > t4_by_seed[s]["fsi_domain"] for s in t4_by_seed]
    hf = any(c_by_seed[s].get("h_failure") for s in t4_by_seed)
    return (sum(gaps) / len(gaps) <= TH["ce_tol"]) and max(gaps) <= 0.015 and all(fs) and not hf


def select(s2, t4_sel):
    """Stage 3 (CALIB-SELECTION opened once): among S2-eligible candidates, those whose CALIB-SELECTION CE <=
    TOKEN-4 + 0.01 in both seeds; choose the highest mean dev FSI_domain (tie -> lower mean selection CE). If none
    passes the selection CE rule, choose the S2-eligible candidate with the lowest mean selection-CE gap, flagged."""
    el = [c for c in s2 if c["s2_eligible"]]
    if not el:
        return None, "no S2-eligible candidate: development NEGATIVE; validation stays untouched"
    ok = [c for c in el if all(c["sel_ce"][s] - t4_sel[s] <= TH["ce_tol"] for s in t4_sel)]
    if ok:
        best = sorted(ok, key=lambda c: (-c["fsi_mean"], sum(c["sel_ce"].values())))[0]
        return best, "selection rule met"
    best = sorted(el, key=lambda c: sum(c["sel_ce"][s] - t4_sel[s] for s in t4_sel))[0]
    return best, "SELECTION-CE-INELIGIBLE (frozen as the closest S2-eligible candidate; flagged)"
