#!/usr/bin/env python
"""csr-multiscale-routing-v1 secondary specialization analysis (prereg §12d), CPU only.

Reuses the specialization-audit-v1 estimators (spec_audit_analyze: window masses, permutation-adjusted MI with the
load-preserving null, topic probe, stratified functional ATT, document bootstrap, cross-seed Hungarian alignment) on
the multiscale extraction runs (ms_spec_extract.py). Differences from the audit wrapper: counterfactuals exist for the
three preregistered blocks 1, 4, 7 only (no all-7-layer secondary), conditions are the multiscale ones, and outputs go
to artifacts/multiscale-routing-v1/spec/ (the audit's directories are only read). Descriptive; never changes the
primary CE conclusions.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402

OUT = guard.ROOT / "artifacts" / "multiscale-routing-v1" / "spec"
m = None


def analyze_ms(arg):
    """Same computations as spec_audit_analyze.analyze_checkpoint, restricted to the counterfactual layers present."""
    key, run_dir, strat = arg
    import numpy as np
    import torch
    torch.set_num_threads(1)
    sa = m.sa
    t0 = time.time()
    R = dict(np.load(Path(run_dir) / "routing.npz"))
    CF = dict(np.load(Path(run_dir) / "cf.npz"))
    assert R["seen"].all() and np.isfinite(R["ce0"]).all()
    WM, PM = m.window_masses(R)
    out = {"key": key, "routing": {}, "probe": {}, "functional": {}}
    raw = {}
    for v in m.ROUTING_VIEWS:
        r = m.routing_view(v, WM, PM, False, do_null=True, do_boot=v in (m.PRIMARY_VIEW, m.ROBUST_VIEW))
        out["routing"][v] = r["summary"]
        null = r["null_mi"]
        out["routing"][v]["null_mi_mean"] = null.mean(0).tolist()
        out["routing"][v]["null_mi_p975"] = np.percentile(null, 97.5, axis=0).tolist()
        out["routing"][v]["mi_adj"] = (np.array(r["summary"]["mi"]) - null.mean(0)).tolist()
        out["routing"][v]["perm_p_layer_mean"] = float((1 + np.sum(null.mean(1) >= np.mean(r["summary"]["mi"]))) / (1 + len(null)))
        raw[f"{v}/null_mi"] = null.astype(np.float32)
        raw[f"{v}/disc_joint"] = r["disc_joint"].astype(np.float32)
        if "boot_mi" in r:
            raw[f"{v}/boot_mi"] = r["boot_mi"].astype(np.float32)
        if "null_disc_joint" in r:
            raw[f"{v}/null_disc_joint"] = r["null_disc_joint"]
    for v in (m.PRIMARY_VIEW, m.ROBUST_VIEW):
        o, b = m.probe_view(v, WM)
        out["probe"][v] = o
        raw[f"{v}/probe_boot_f1"] = b.astype(np.float32)
    n_moe = WM.shape[0]
    layers = [int(x) for x in CF["layers"]]
    assert layers == m.PRIMARY_LAYERS, layers
    setups = {l: m.functional_layer_setup(R, CF, l, li, strat) for li, l in enumerate(layers)}
    for v in m.FUNC_VIEWS:
        K = m.kof(v)
        lab = m.G["labels"][v]
        dj = raw[f"{v}/disc_joint"].reshape(K, n_moe, 32)
        fv = {"layers": {}}
        parts, elig = [], []
        for l in layers:
            mass = dj[:, l, :]
            top = np.where(mass.sum(0) > 0, mass.argmax(0), -1)
            S = setups[l]
            res = m.functional_estimate(S, lab, top, want_boot=v in (m.PRIMARY_VIEW, m.ROBUST_VIEW))
            eq, ut = m.summarize(res, S["n_e"])
            glob = np.bincount(S["e"], weights=S["d"], minlength=32) / np.maximum(S["n_e"], 1)
            fv["layers"][str(l)] = {"equal_weighted": eq, "utilization_weighted": ut, "n_eligible": int(res["elig"].sum()),
                                    "n_zero_load": int((top < 0).sum()),
                                    "n_insufficient_support": int(((top >= 0) & ~res["elig"]).sum()),
                                    "global_mean_dce_mean_over_experts": float(np.mean(glob[S["n_e"] > 0]))}
            if "boot_parts" in res:
                parts.append(res["boot_parts"])
                elig.append(res["elig"])
        if parts:
            raw[f"{v}/func_boot"] = m.boot_summary(parts, elig).astype(np.float32)
        prim = [fv["layers"][str(l)]["equal_weighted"] for l in m.PRIMARY_LAYERS]
        fv["F_primary"] = float(np.mean(prim)) if all(p is not None for p in prim) else None
        out["functional"][v] = fv
    v = m.PRIMARY_VIEW
    K = m.kof(v)
    lab = m.G["labels"][v]
    ndj = raw[f"{v}/null_disc_joint"].reshape(-1, K, n_moe, 32)
    fn = np.full((sa.N_PERM, len(m.PRIMARY_LAYERS)), np.nan)
    for pi_ in range(sa.N_PERM):
        lab_p = lab[m.G["P"][pi_]]
        for j, l in enumerate(m.PRIMARY_LAYERS):
            mass = ndj[pi_, :, l, :]
            top = np.where(mass.sum(0) > 0, mass.argmax(0), -1)
            res = m.functional_estimate(setups[l], lab_p, top)
            eq, _ = m.summarize(res, setups[l]["n_e"])
            fn[pi_, j] = np.nan if eq is None else eq
    raw[f"{v}/func_null"] = fn.astype(np.float32)
    out["functional"][v]["null_F_primary_mean"] = float(np.nanmean(fn.mean(1)))
    out["functional"][v]["null_F_primary_p975"] = float(np.nanpercentile(fn.mean(1), 97.5))
    out["seconds"] = time.time() - t0
    np.savez(OUT / "raw" / f"{key.replace('/', '__')}.npz", **{k.replace("/", "__"): val for k, val in raw.items()})
    print(f"{key} done {out['seconds']:.0f}s", flush=True)
    return out


def _cross_seed_one(raws_A):
    import numpy as np
    sa = m.sa
    K = 32
    pairs = [(0, 1), (0, 2), (1, 2)]

    def profiles(J):
        Jl = J.reshape(K, 7, 32).transpose(1, 2, 0)
        tot = Jl.sum(2, keepdims=True)
        glob = Jl.sum(1, keepdims=True) / Jl.sum((1, 2), keepdims=True)
        return np.where(tot > 0, Jl / np.where(tot > 0, tot, 1), glob), Jl.sum(1, keepdims=True) / Jl.sum((1, 2), keepdims=True)

    def metrics(profs, globs):
        res = {"matched_jsd": [], "repro_mass": [], "repro_lift": []}
        for l in range(7):
            assign, mj = {}, []
            for a, b in pairs:
                cost = sa.jsd(profs[a][l][:, None, :], profs[b][l][None, :, :])
                col = sa.hungarian_fast(cost)
                assign[(a, b)] = col
                mj.append(float(cost[np.arange(32), col].mean()))
            res["matched_jsd"].append(float(np.mean(mj)))
            am = [profs[s][l].argmax(1) for s in range(3)]
            al = [(profs[s][l] / np.maximum(globs[s][l], 1e-12)).argmax(1) for s in range(3)]
            p1, p2 = assign[(0, 1)], assign[(0, 2)]
            res["repro_mass"].append(float(np.mean((am[0] == am[1][p1]) & (am[0] == am[2][p2]))))
            res["repro_lift"].append(float(np.mean((al[0] == al[1][p1]) & (al[0] == al[2][p2]))))
        return {k: float(np.mean(v)) for k, v in res.items()} | {f"{k}_by_layer": v for k, v in res.items()}

    V = m.PRIMARY_VIEW
    obs_p = [profiles(raws_A[f"s{s}_A"][f"{V}__disc_joint"]) for s in range(3)]
    obs = metrics([p[0] for p in obs_p], [p[1][:, 0] for p in obs_p])
    nulls = {"matched_jsd": [], "repro_mass": [], "repro_lift": []}
    nd = [raws_A[f"s{s}_A"][f"{V}__null_disc_joint"] for s in range(3)]
    for pi_ in range(sa.N_PERM):
        pp = [profiles(nd[s][pi_]) for s in range(3)]
        mm = metrics([p[0] for p in pp], [p[1][:, 0] for p in pp])
        for k_ in nulls:
            nulls[k_].append(mm[k_])
    return {"observed": obs, **{f"null_{k_}_mean": float(np.mean(v)) for k_, v in nulls.items()},
            **{f"null_{k_}_ci95": sa.ci95(v) for k_, v in nulls.items()},
            "p_matched_jsd_lower": float((1 + np.sum(np.array(nulls["matched_jsd"]) <= obs["matched_jsd"])) / (1 + sa.N_PERM)),
            "p_repro_mass_higher": float((1 + np.sum(np.array(nulls["repro_mass"]) >= obs["repro_mass"])) / (1 + sa.N_PERM)),
            "p_repro_lift_higher": float((1 + np.sum(np.array(nulls["repro_lift"]) >= obs["repro_lift"])) / (1 + sa.N_PERM))}


def main():
    global m
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract-runs", required=True, help="TSV key<TAB>extract_run_dir<TAB>strategy")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    print("msv1 specialization analysis (CPU)")
    if not guard.require_execution(a.execute, "cpu-model"):
        return
    if (OUT / "spec_analysis.json").exists():
        raise guard.GuardError("spec_analysis.json exists")
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    import multiprocessing as mp
    import numpy as np
    import spec_audit_analyze as mod
    m = mod
    m.setup_globals()  # reads the audit's frozen windows / labels / permutations (read-only)
    guard.register_self("msv1 spec analyze")
    t0 = time.time()
    rows = [ln.split("\t") for ln in Path(a.extract_runs).read_text().splitlines() if ln.strip()]
    jobs = [(k, str(guard.ROOT / rd), strat) for k, rd, strat in rows]
    with mp.get_context("fork").Pool(a.workers) as pool:
        outs = pool.map(analyze_ms, jobs, chunksize=1)
    per = {o["key"]: o for o in outs}
    raws = {k: {kk.replace("__", "__"): v for kk, v in np.load(OUT / "raw" / f"{k.replace('/', '__')}.npz").items()}
            for k in per}
    conds = sorted({k.split("/")[1] for k in per if all(f"seed{s}/{k.split('/')[1]}" in per for s in range(3))})
    xs = {c: _cross_seed_one({f"s{s}_A": raws[f"seed{s}/{c}"] for s in range(3)}) for c in conds}
    res = {"per_checkpoint": per, "cross_seed": xs, "primary_view": m.PRIMARY_VIEW, "robust_view": m.ROBUST_VIEW,
           "functional_layers_blocks": [l + 1 for l in m.PRIMARY_LAYERS], "seconds": time.time() - t0,
           "created": guard.now(), "note": "descriptive secondary analysis; specialization-audit-v1 estimators"}
    guard.atomic_write_json(OUT / "spec_analysis.json", res, mode=0o400)
    guard.ledger("MS_V1_SPEC_ANALYSIS", f"spec/spec_analysis.json n={len(per)} sha256={guard.sha256_file(OUT / 'spec_analysis.json')}")
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
