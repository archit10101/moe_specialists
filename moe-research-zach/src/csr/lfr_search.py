"""lfr bounded engineering search (protocol 8-10): search space, deterministic failure diagnosis, next-candidate
choice, stage-2 advancement and stage-3 selection rules. Frozen in search_space.json (hashed before any candidate)."""
from __future__ import annotations

import copy
import json

import numpy as np

BASE = {"horizons": [1, 4, 8], "agg": "deepsets", "fc_dim": 64, "mem": "ema", "mem_halflife": 128.0,
        "util_mode": "rank", "c_util": 1.0, "c_sutil": 0.3, "c_kl": 1.0, "tau_t": 1.0, "kappa": 0.0,
        "router_lr_mult": 1.0, "r_cf": 0.075, "schedule": "40/20/40", "inject": "all", "c_lc": 0.0}
INITIAL = [("P01", {}), ("P02", {"agg": "pool"}), ("P03", {"agg": "xattn"}),
           ("P04", {"util_mode": "reg", "kappa": 0.5}), ("P05", {"c_lc": 0.05})]
RESPONSES = {
    "I": [{"r_cf": 0.05}],
    "H": [{"router_lr_mult": 0.3}, {"tau_t": "TAU_BY_EVIDENCE"}],
    "B": [{"fc_dim": 128}, {"horizons": [1]}],
    "C": [{"r_cf": 0.10}, {"util_mode": "SWAP"}],
    "D": [{"tau_t": 0.5}, {"kappa": 0.5}, {"c_util": 3.0}],
    "E": [{"c_kl": 3.0}, {"mem": "ema3"}, {"schedule": "30/20/50"}],
    "F": [{"router_lr_mult": 3.0}, {"inject": "late"}],
    "G": [{"c_lc": 0.05}, {"inject": "late"}],
}
REFINEMENT = [{"schedule": "30/20/50"}, {"router_lr_mult": 3.0}, {"r_cf": 0.10}, {"mem": "ema3"},
              {"horizons": [1, 4]}]
PRIORITY = ["I", "H", "B", "C", "D", "E", "F", "G"]
MAX_CANDIDATES = 12
ALLOWED = {"horizons": [[1, 4, 8], [1, 4], [1], [4, 8], [1, 8], [4], [8]], "agg": ["pool", "deepsets", "xattn"],
           "fc_dim": [64, 128], "mem": ["ema", "ema3"], "mem_halflife": [32.0, 128.0, 512.0],
           "util_mode": ["rank", "reg"], "c_kl": [1.0, 3.0], "c_util": [1.0, 3.0], "c_sutil": [0.3],
           "tau_t": [0.5, 1.0, 2.0], "router_lr_mult": [0.3, 1.0, 3.0], "r_cf": [0.05, 0.075, 0.10],
           "schedule": ["40/20/40", "30/20/50"], "kappa": [0.0, 0.5], "inject": ["all", "late"],
           "c_lc": [0.0, 0.05]}
THRESH = {"B_shuffle_gap": 0.002, "B_probe_ratio": 0.8, "C_rho_teacher": 0.2, "C_rho_fo": 0.3, "D_margin": 0.002,
          "E_overlap": 2.0, "E_slack": 0.01, "F_stability": 0.5, "H_load_min": 0.1 / 32, "H_load_max": 4.0 / 32,
          "H_ent_hi": 0.98, "H_ent_lo": 0.2, "H_tc_gn": 1e-7, "I_step_ratio": 2.5, "success_margin": 0.005}


def search_space_doc():
    return {"base": BASE, "initial": INITIAL, "responses": RESPONSES, "refinement": REFINEMENT, "priority": PRIORITY,
            "max_candidates": MAX_CANDIDATES, "allowed": ALLOWED, "thresholds": THRESH,
            "incumbent": "eligible proposed candidate with lowest CALIB-SEARCH CE (else lowest CE)",
            "eligible": "no class-H flag, finite, dev FSI_domain > TOKEN-4 S1 reference",
            "advance": "top-4 eligible by CALIB-SEARCH CE, filled with lowest-CE ineligible (marked)",
            "select": "S2-eligible (no H in either seed, mean dev FSI_domain > mean S2 TOKEN-4) with lowest mean "
                      "CALIB-SELECTION CE over seeds 100/101; else lowest CE marked INELIGIBLE"}


def apply(knobs, upd, evidence=None):
    k = copy.deepcopy(knobs)
    for key, v in upd.items():
        if v == "SWAP":
            v = "reg" if k[key] == "rank" else "rank"
        elif v == "TAU_BY_EVIDENCE":
            ent = (evidence or {}).get("ent")
            v = 0.5 if (ent is not None and ent > THRESH["H_ent_hi"]) else 2.0
        k[key] = v
    for key, v in k.items():
        if key in ALLOWED and v not in ALLOWED[key]:
            raise ValueError(f"knob {key}={v} outside the preregistered space")
    return k


def key_of(knobs):
    return json.dumps(knobs, sort_keys=True)


def diagnose(c, t4, edl):
    """c, t4, edl: candidate / TOKEN-4 / EDL development records (same stage + seed) with fields:
    ce (CALIB-SEARCH), d (in-process diag dict by name), audit (dev audit: fsi_domain, mi_mean), tok_s,
    tc_gn_p1 (mean teacher grad norm over P1), cfaudit_rho (in-run first-order vs exact), profiles_p1/profiles_end,
    p1name (diag name at the end of P1), finite (bool). Returns (flags {class: evidence}, success bool)."""
    T = THRESH
    f = {}
    d = c.get("d", {})
    p1 = d.get(c.get("p1name", "p1end"), {})
    p2 = d.get("p2end", {})
    end = d.get("end", {})
    # I systems
    if c.get("tok_s") and t4.get("tok_s") and t4["tok_s"] / c["tok_s"] > T["I_step_ratio"]:
        f["I"] = {"step_time_ratio": t4["tok_s"] / c["tok_s"]}
    if c.get("read_error"):
        f["I"] = {"read_error": c["read_error"]}
    # H optimisation
    h = {}
    if not c.get("finite", True):
        h["nonfinite"] = True
    if end:
        if end.get("load_min", 1) < T["H_load_min"]:
            h["load_min"] = end["load_min"]
        if end.get("load_max", 0) > T["H_load_max"]:
            h["load_max"] = end["load_max"]
        if end.get("ent", 0.5) > T["H_ent_hi"] or end.get("ent", 0.5) < T["H_ent_lo"]:
            h["ent"] = end["ent"]
    if c.get("tc_gn_p1") is not None and c["tc_gn_p1"] < T["H_tc_gn"]:
        h["tc_gn_p1"] = c["tc_gn_p1"]
    if h:
        f["H"] = h
    # B aggregation
    if p1.get("ce_teacher") is not None:
        gap = p1["ce_teacher_shuffled"] - p1["ce_teacher"]
        if gap < T["B_shuffle_gap"] or p1.get("probe_ratio", 1) < T["B_probe_ratio"]:
            f["B"] = {"shuffle_gap": gap, "probe_ratio": p1.get("probe_ratio")}
    # C utility target
    c_ok = True
    if p1.get("rho_teacher_exact_mean") is not None:
        rho_fo = c.get("cfaudit_rho", p1.get("rho_fo_exact_mean"))
        if p1["rho_teacher_exact_mean"] < T["C_rho_teacher"] or (rho_fo is not None and rho_fo < T["C_rho_fo"]):
            f["C"] = {"rho_teacher_exact": p1["rho_teacher_exact_mean"], "rho_fo_exact": rho_fo}
            c_ok = False
    # D teacher
    t4p1 = t4.get("d", {}).get(c.get("p1name_ref", "p1end_40"), {}).get("ce_student")
    d_ok = True
    if c_ok and p1.get("ce_teacher") is not None and t4p1 is not None:
        if p1["ce_teacher"] >= t4p1 - T["D_margin"]:
            f["D"] = {"ce_teacher_p1": p1["ce_teacher"], "t4_ce_p1": t4p1}
            d_ok = False
    # E distillation
    if d_ok and p1.get("ce_teacher") is not None:
        t4_end = t4.get("d", {}).get("end", {}).get("ce_student")
        c_end = end.get("ce_student")
        ev = {}
        if c_end is not None and t4_end is not None and c_end >= t4_end:
            ev["student_not_better_than_t4"] = [c_end, t4_end]
        if p2.get("top4_overlap_mean") is not None and p2["top4_overlap_mean"] < T["E_overlap"]:
            ev["overlap_p2end"] = p2["top4_overlap_mean"]
        if c_end is not None and t4_end is not None and t4p1 is not None:
            if (c_end - p1["ce_teacher"]) > (t4_end - t4p1) + T["E_slack"]:
                ev["student_lost_teacher_gain"] = [c_end - p1["ce_teacher"], t4_end - t4p1]
        if ev:
            f["E"] = ev
    # F / G specialisation
    a_c, a_t = c.get("audit", {}), t4.get("audit", {})
    stab = None
    if c.get("profiles_p1") is not None and c.get("profiles_end") is not None:
        A, B = np.asarray(c["profiles_p1"], float), np.asarray(c["profiles_end"], float)
        cs = []
        for li in range(A.shape[0]):
            x, y = A[li].reshape(-1), B[li].reshape(-1)
            if x.std() > 0 and y.std() > 0:
                cs.append(float(np.corrcoef(x, y)[0, 1]))
        stab = float(np.mean(cs)) if cs else None
    f_flag = False
    if a_c.get("mi_mean") is not None and a_t.get("mi_mean") is not None and a_c["mi_mean"] <= a_t["mi_mean"]:
        f_flag = True
    if stab is not None and stab < T["F_stability"]:
        f_flag = True
    if f_flag:
        f["F"] = {"mi": a_c.get("mi_mean"), "mi_t4": a_t.get("mi_mean"), "stability_p1_end": stab}
    elif a_c.get("fsi_domain") is not None and a_t.get("fsi_domain") is not None and \
            a_c["fsi_domain"] <= a_t["fsi_domain"]:
        f["G"] = {"fsi": a_c["fsi_domain"], "fsi_t4": a_t["fsi_domain"]}
    success = (not f and c.get("ce") is not None and edl.get("ce") is not None and
               c["ce"] <= edl["ce"] - T["success_margin"] and a_c.get("fsi_domain", -1e9) > a_t.get("fsi_domain", 1e9))
    return f, bool(success)


def eligible(c, t4):
    return c.get("finite", True) and "H" not in c.get("flags", {}) and \
        c.get("audit", {}).get("fsi_domain") is not None and t4.get("audit", {}).get("fsi_domain") is not None and \
        c["audit"]["fsi_domain"] > t4["audit"]["fsi_domain"]


def incumbent(cands, t4):
    """cands: list of dev records (stage 1) with id, knobs, ce, flags, audit. Deterministic."""
    done = [c for c in cands if c.get("ce") is not None]
    if not done:
        return None
    el = [c for c in done if eligible(c, t4)]
    pool = el if el else done
    return sorted(pool, key=lambda c: (c["ce"], c["id"]))[0]


def next_candidate(cands, t4):
    """Returns (id, knobs, reason) of the next diagnosis-driven candidate, or None when the budget/lists are spent."""
    if len(cands) >= MAX_CANDIDATES:
        return None
    ids = {c["id"] for c in cands}
    for cid, upd in INITIAL:
        if cid not in ids:
            return cid, apply(BASE, upd), "initial wave"
    seen = {key_of(c["knobs"]) for c in cands}
    inc = incumbent(cands, t4)
    nid = f"P{len(cands) + 1:02d}"
    flags = inc.get("flags", {})
    for cls in [k for k in PRIORITY if k in flags]:
        for upd in RESPONSES[cls]:
            try:
                kn = apply(inc["knobs"], upd, flags.get(cls))
            except ValueError:
                continue
            if key_of(kn) not in seen:
                return nid, kn, f"incumbent {inc['id']} class {cls} -> {upd}"
    for upd in REFINEMENT:
        try:
            kn = apply(inc["knobs"], upd)
        except ValueError:
            continue
        if key_of(kn) not in seen:
            return nid, kn, f"incumbent {inc['id']} refinement -> {upd}"
    return None


def advance(cands, t4, k=4):
    done = sorted([c for c in cands if c.get("ce") is not None], key=lambda c: (c["ce"], c["id"]))
    el = [c for c in done if eligible(c, t4)]
    inel = [c for c in done if not eligible(c, t4)]
    out = [(c["id"], True) for c in el[:k]]
    for c in inel:
        if len(out) >= k:
            break
        out.append((c["id"], False))
    return out


def select(s2, t4_fsi_mean):
    """s2: {cid: {"ce_sel": [seed100, seed101], "fsi": [..], "H": [bool, bool]}}. Returns (cid, eligible)."""
    rows = []
    for cid, r in sorted(s2.items()):
        ok = (not any(r["H"])) and np.mean(r["fsi"]) > t4_fsi_mean
        rows.append((not ok, float(np.mean(r["ce_sel"])), cid, ok))
    rows.sort()
    best = rows[0]
    return best[2], bool(best[3])
