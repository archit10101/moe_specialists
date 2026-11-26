#!/usr/bin/env python
"""fspv2 final analysis (CPU): aggregates data/draft/search/final/eval/compute artifacts into
artifacts/forecast-specialist-pretraining-v2/final_analysis.json and prints compact tables for the report.
Applies the protocol decision rules (section 9) and specialisation criteria (section 10) mechanically."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from csr import guard  # noqa: E402

ART = ROOT / "artifacts/forecast-specialist-pretraining-v2"
DOMS = ["cs", "math", "physics", "q-bio", "stat"]
CONDS = ["TOKEN-4", "EDL", "DOMAIN-AUX", "FORECAST-SPECIALIST", "TOKEN-COMPUTE", "LABEL-ORACLE", "SHUFFLED",
         "WRONG-DRAFT", "FORECAST-NOPART"]


def jl(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else None


def ev(cond, seed, what="full"):
    return jl(ART / "evals" / f"final_{cond}_s{seed}_{what}.json")


def val_ce(cond, seed):
    e = ev(cond, seed) or ev(cond, seed, "ce")
    return None if e is None else e["p100"]["val"]


def bootstrap(a_cond, b_cond, seed, n=2000):
    """Paired document bootstrap of CE(b) - CE(a) on VAL (token-weighted). Not a substitute for seeds."""
    pa = ART / "evals" / f"final_{a_cond}_s{seed}_p100_val_doc_ce.npz"
    pb = ART / "evals" / f"final_{b_cond}_s{seed}_p100_val_doc_ce.npz"
    if not (pa.exists() and pb.exists()):
        return None
    A, B = np.load(pa), np.load(pb)
    assert (A["doc"] == B["doc"]).all()
    rng = np.random.default_rng(0)
    n_doc = len(A["doc"])
    ds = []
    for _ in range(n):
        i = rng.integers(0, n_doc, n_doc)
        ds.append(B["ce_sum"][i].sum() / B["n"][i].sum() - A["ce_sum"][i].sum() / A["n"][i].sum())
    return {"mean": float(B["ce_sum"].sum() / B["n"].sum() - A["ce_sum"].sum() / A["n"].sum()),
            "ci95": [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))]}


def spec_summary(e, tag):
    r = e.get(tag, {})
    sub = r.get("val_sub", {}).get("routing", {})
    ab = r.get("ablation")
    out = {"ce_val_sub": r.get("val_sub", {}).get("ce")}
    for k in ("spec_group_acc_mean", "min_spec_share", "spec_intended_mass", "mi_bits", "unused_experts_per_layer",
              "domain_pred_acc", "domain_pred_acc_by_pos", "domain_pred_acc_by_segidx"):
        if k in sub:
            out[k] = sub[k]
    if "mi_bits" in sub:
        out["mi_bits_mean"] = float(np.mean(sub["mi_bits"]))
    if ab:
        out["ablation_own"] = ab["own"]
        out["ablation_other_mean"] = ab["other_mean"]
        out["ablation_own_minus_other"] = ab["own_minus_other"]
        out["n_groups_own_gt_other"] = ab["n_groups_own_gt_other"]
    if "swap" in r:
        out["swap_harm"] = r["swap"]["swap_harm"]
    return out


def role_agreement(e0, e1):
    """Spec models: fraction of (layer, specialist) whose argmax domain (VAL-sub profile) agrees across seeds, and
    mean correlation of domain profiles."""
    try:
        p0 = np.asarray(e0["p100"]["val_sub"]["routing"]["expert_domain_profile"])  # (L, D, E)
        p1 = np.asarray(e1["p100"]["val_sub"]["routing"]["expert_domain_profile"])
    except (KeyError, TypeError):
        return None
    sp = slice(22, 32)
    a0, a1 = p0[:, :, sp].argmax(1), p1[:, :, sp].argmax(1)
    cor = [np.corrcoef(p0[l, :, e], p1[l, :, e])[0, 1] for l in range(p0.shape[0]) for e in range(22, 32)]
    intended = np.repeat(np.arange(5), 2)
    return {"argmax_domain_agreement": float((a0 == a1).mean()), "profile_corr_mean": float(np.nanmean(cor)),
            "frac_argmax_is_intended_s0": float((a0 == intended).mean()),
            "frac_argmax_is_intended_s1": float((a1 == intended).mean())}


def main():
    guard.check_root()
    fa = {"experiment": "csr-forecast-specialist-pretraining-v2", "created": guard.now(),
          "protocol_sha256": guard.sha256_file(ROOT / "docs/FORECAST_SPECIALIST_PRETRAINING_V2_PROTOCOL.md")}
    man = jl(ART / "data/manifest.json")
    fa["data"] = {k: man[k] for k in ("allocation_rule", "quotas_valid_tokens_per_domain", "A_min_available_valid_tokens",
                                      "scarcity_order", "splits_sha256", "near_dedup", "exact_dedup", "deviations")}
    fa["data"]["splits"] = {k: {kk: v[kk] for kk in ("n_docs", "valid_tokens", "n_papers", "frac_exclusive_segments",
                                                     "sha256_tokens")} for k, v in man["splits"].items()}
    fa["draft_diagnostics"] = jl(ART / "draft_diagnostics.json")
    fa["search_stage1"] = jl(ART / "search_stage1.json")
    fa["frozen"] = jl(ART / "frozen.json")
    st = jl(ART / "driver_state.json")
    seeds = [0, 1] + ([2] if st and st["done"].get("seed2_triggered") else [])
    # ---------------- CE tables
    ce = {}
    for c in CONDS:
        for s in seeds:
            v = val_ce(c, s)
            if v:
                ce.setdefault(c, {})[s] = {"ce": v["ce"], "ce_domain": v["ce_domain"]}
    fa["val_ce"] = ce
    deltas = {}
    for c in CONDS:
        if c == "TOKEN-4" or c not in ce:
            continue
        for ref in ("TOKEN-4", "EDL", "DOMAIN-AUX", "TOKEN-COMPUTE"):
            if ref == c or ref not in ce:
                continue
            ds = {s: ce[c][s]["ce"] - ce[ref][s]["ce"] for s in ce[c] if s in ce[ref]}
            if ds:
                deltas.setdefault(c, {})[ref] = {
                    "per_seed": ds, "mean": float(np.mean(list(ds.values()))),
                    "per_domain": {s: (np.asarray(ce[c][s]["ce_domain"]) - np.asarray(ce[ref][s]["ce_domain"])).tolist()
                                   for s in ds},
                    "bootstrap": {s: bootstrap(ref, c, s) for s in ds}}
    fa["deltas"] = deltas
    # ---------------- decision rules
    D = "FORECAST-SPECIALIST"
    dec = {}
    if D in deltas and "TOKEN-4" in deltas[D]:
        dA = deltas[D]["TOKEN-4"]["per_seed"]
        dC = deltas[D].get("EDL", {}).get("per_seed", {})
        dec["comparable_to_token4"] = all(v <= 0.005 for v in dA.values())
        dec["better_than_token4"] = (np.mean(list(dA.values())) <= -0.005 and all(v < 0 for v in dA.values()))
        dec["provisionally_better_than_token4_and_edl"] = all(dA[s] < 0 and dC.get(s, 1) < 0 for s in dA)
        signs = {np.sign(v) for v in dA.values()}
        dec["seed_signs_vs_token4"] = "agree" if len(signs) == 1 else "disagree (unresolved)"
        for ref in ("EDL", "DOMAIN-AUX", "TOKEN-COMPUTE"):
            if ref in deltas[D]:
                v = deltas[D][ref]["per_seed"]
                dec[f"beats_{ref}_all_seeds"] = all(x < 0 for x in v.values())
                dec[f"delta_vs_{ref}_mean"] = float(np.mean(list(v.values())))
    fa["decision"] = dec
    # ---------------- specialisation
    spec = {}
    for c in CONDS:
        for s in seeds:
            e = ev(c, s)
            if not e:
                continue
            for tag in ("p040", "p050", "p075", "p100"):
                if tag in e:
                    spec.setdefault(c, {}).setdefault(s, {})[tag] = spec_summary(e, tag)
    fa["specialization"] = spec
    roles = {}
    for c in ("FORECAST-SPECIALIST", "LABEL-ORACLE", "SHUFFLED", "WRONG-DRAFT"):
        e0, e1 = ev(c, 0), ev(c, 1)
        if e0 and e1:
            roles[c] = role_agreement(e0, e1)
    fa["role_stability"] = roles
    crit = {}
    for c in ("FORECAST-SPECIALIST", "SHUFFLED", "WRONG-DRAFT", "LABEL-ORACLE"):
        if c not in spec:
            continue
        per = {}
        for s, tags in spec[c].items():
            p = tags.get("p100", {})
            per[s] = {"acc_ge_0.5": (p.get("spec_group_acc_mean") or 0) >= 0.5,
                      "own_gt_other_ge4": (p.get("n_groups_own_gt_other") or 0) >= 4,
                      "min_load_ge_2pct": (p.get("min_spec_share") or 0) >= 0.02}
        ra = roles.get(c) or {}
        crit[c] = {"per_seed": per, "role_agreement_ge_0.8": (ra.get("argmax_domain_agreement") or 0) >= 0.8}
        crit[c]["persistent_and_functional"] = bool(per) and all(all(v.values()) for v in per.values()) and \
            crit[c]["role_agreement_ge_0.8"]
    fa["specialization_criteria"] = crit
    if D in crit and "comparable_to_token4" in dec:
        fa["success"] = bool(crit[D]["persistent_and_functional"] and
                             (dec["comparable_to_token4"] or dec["better_than_token4"]))
    # ---------------- dev-scale controls and search
    fa["dev_controls"] = {}
    sl = ART / "search_ledger.tsv"
    if sl.exists():
        rows = [l.split("\t") for l in sl.read_text().splitlines()]
        hdr, body = rows[0], rows[1:]
        fa["search_ledger"] = [dict(zip(hdr, r)) for r in body]
    # ---------------- compute
    comp = {}
    for line in (ART / "compute_ledger.tsv").read_text().splitlines():
        f = line.split("\t")
        if len(f) >= 7:
            comp.setdefault(f[5], 0.0)
            comp[f[5]] += float(f[2])
    comp["pre_protocol_probes(estimate)"] = 420.0
    fa["gpu_seconds_by_category"] = comp
    fa["gpu_hours_total"] = sum(comp.values()) / 3600
    cm = jl(ART / "compute_match.json")
    if cm:
        dc = cm["draft_training_gpu_s"] + cm["forecast_cache_gpu_s"]
        main_d = float(np.mean(cm["proposed_main_gpu_s"]))
        fa["draft_cost"] = {"draft_plus_cache_gpu_s": dc, "proposed_main_gpu_s_mean": main_d,
                            "unamortized_overhead_frac": dc / main_d,
                            "amortized_over_actual_seeds": dc / cm["n_final_proposed_seeds"] / main_d,
                            "projected_amortized_10_trainings": dc / 10 / main_d, "compute_match": cm}
    bl = sorted(ART.glob("bench_*.json"))
    fa["bench"] = jl(bl[-1]) if bl else None
    fa["deviations"] = [json.loads(l) for l in (ART / "deviations.jsonl").read_text().splitlines()] \
        if (ART / "deviations.jsonl").exists() else []
    guard.atomic_write_json(ART / "final_analysis.json", fa, overwrite=True)
    print(json.dumps({"val_ce": {c: {s: round(v["ce"], 4) for s, v in d.items()} for c, d in ce.items()},
                      "decision": dec, "success": fa.get("success"), "gpu_h": round(fa["gpu_hours_total"], 2)},
                     indent=1, default=str))


if __name__ == "__main__":
    main()
