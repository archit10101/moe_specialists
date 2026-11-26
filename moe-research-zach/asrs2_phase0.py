#!/usr/bin/env python
"""asrs2 Phase 0: independent diagnosis of asrs1 from its recorded evidence (CPU, read-only on asrs1/ccsr1 files).

Writes artifacts/csr-continuous-base-target-free-asrs-v2/phase0_diagnosis.json. Nothing outside the asrs2 artifact
directory is written. Sections:
  H1  base-update starvation: same-row and update-count-matched training-loss comparisons asrs1 base vs the ccsr1 S3
      ordinary continuation (base updated on every step), lr-schedule identity, SEL ALT-base curve.
  H2  target leakage: code facts; cross-fitted predictability of the teacher-SELECTED specialist (teacher caches,
      training rows) and of the per-token ORACLE specialist (SEL eval tables) from target token, input token (causal),
      paper semantic group, region-prior argmax, deployed causal router, majority baseline; utility trajectories.
  H3  alternatives: phase isolation from parameter hashes, router drift at router-phase start, loads/collapse,
      headroom, evaluation identity, resume/legs."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402

V1 = guard.ROOT / "artifacts/csr-alternating-semantic-residual-specialization-v1"
RUN1 = guard.ROOT / "runs/20261122T001339Z_asrs1_S1"
S3 = guard.ROOT / "runs/20261121T194254Z_ccsr1_S3"
OUT = guard.ROOT / "artifacts/csr-continuous-base-target-free-asrs-v2/phase0_diagnosis.json"
TABLES = guard.ROOT / "artifacts/csr-crossfit-semantic-utility-residual-v1/tables"


def jl(p):
    return [json.loads(x) for x in open(p)]


def hsplit(keys, salt=7919):
    """Deterministic cross-fit halves from integer keys (doc / paper ids)."""
    k = (np.asarray(keys, np.int64) * 2654435761 + salt) % 4294967296
    return (k >> 7) % 2 == 0


def majority_acc(feat, lab, fold, K=16, top=2000):
    """Cross-fitted majority-vote accuracy: map feature value -> most common label learned on one fold, scored on the
    other (both directions, token-weighted average). Values outside the `top` most frequent fit values -> 'other'."""
    feat = np.asarray(feat, np.int64)
    lab = np.asarray(lab, np.int64)
    accs, ns = [], []
    for fit in (fold, ~fold):
        ev = ~fit
        f_fit, l_fit = feat[fit], lab[fit]
        vals, cnt = np.unique(f_fit, return_counts=True)
        sk = np.sort(vals[np.argsort(-cnt)[:top]])
        cf = np.where(np.isin(f_fit, sk), np.searchsorted(sk, f_fit), len(sk))
        tab = np.bincount(cf * K + l_fit, minlength=(len(sk) + 1) * K).reshape(len(sk) + 1, K)
        glob = int(np.bincount(l_fit, minlength=K).argmax())
        pred_of = tab.argmax(1)
        pred_of[tab.sum(1) == 0] = glob
        fe = feat[ev]
        ce = np.where(np.isin(fe, sk), np.searchsorted(sk, fe), len(sk))
        accs.append(float((pred_of[ce] == lab[ev]).mean()))
        ns.append(int(ev.sum()))
    return float(np.average(accs, weights=ns))


def h1():
    s3 = {r["step"]: r for r in jl(S3 / "metrics.jsonl") if "lm" in r}
    m1 = [r for r in jl(RUN1 / "metrics.jsonl") if r.get("phase") == "base" and "rec" in r]
    out = {"n_s3_steps": len(s3), "n_asrs1_base_steps": len(m1)}
    same_row, by_cycle = [], {}
    upd_matched = []
    for n, r in enumerate(m1):
        s = r["step"]
        d = r["rec"]["lm"] - s3[s]["lm"]
        same_row.append(d)
        by_cycle.setdefault(r["cycle"], []).append(d)
        upd_matched.append((n, r["rec"]["lm"], s3[318 + n]["lm"], r["lr"], s3[318 + n]["lr"]))
    out["same_row_deficit_by_cycle"] = {int(c): float(np.mean(v)) for c, v in sorted(by_cycle.items())}
    um = np.asarray([(x[1], x[2]) for x in upd_matched])
    win = {}
    for i in range(0, len(um), 32):
        win[f"updates_{i}-{min(i + 32, len(um)) - 1}"] = {"asrs1_lm": float(um[i:i + 32, 0].mean()),
                                                         "s3_lm_same_update_count": float(um[i:i + 32, 1].mean()),
                                                         "diff": float((um[i:i + 32, 0] - um[i:i + 32, 1]).mean())}
    out["update_count_matched"] = win
    out["update_count_matched_mean_diff_after_20"] = float((um[20:, 0] - um[20:, 1]).mean())
    out["lr_identical_at_matched_update_count"] = bool(all(abs(x[3] - x[4]) < 1e-12 for x in upd_matched))
    # S3 training-curve projection of the update deficit (training rows; mean of 32-step windows)
    s3lm = np.asarray([s3[s]["lm"] for s in range(318, 1111)])
    out["s3_lm_window_at_320_updates"] = float(s3lm[304:336].mean())
    out["s3_lm_window_at_793_updates"] = float(s3lm[-32:].mean())
    out["s3_projected_deficit_320_vs_793"] = out["s3_lm_window_at_320_updates"] - out["s3_lm_window_at_793_updates"]
    st1 = json.loads((V1 / "stage1.json").read_text())
    out["sel"] = {"ordinary_base_793_updates": 3.6227712345149667, "alt_base_320_updates": 3.98223,
                  "measured_deficit": 3.98223 - 3.6227712345149667,
                  "alt_base_curve_by_cycle": {}}
    for c in range(1, 11):
        p = V1 / f"analysis/20261122T001339Z_asrs1_S1_c{c:02d}.json"
        if p.exists():
            a = json.loads(p.read_text())
            ce = a.get("ce", {}).get("base") if isinstance(a.get("ce"), dict) else None
            out["sel"]["alt_base_curve_by_cycle"][c] = ce
    out["stage1_keys"] = list(st1)[:20]
    return out


def reconstruct(split, steps, P, Psh, docs_sorted, ptr, sem, shuf):
    """Scored tokens of the given steps in training flattening order: x, y, doc, pos, region-prior argmaxes."""
    X, Y, D, POS = [], [], [], []
    for s in steps:
        for row in split_rows[s]:
            for d in row:
                ids = split.doc(d)
                L = len(ids)
                X.append(ids[:-1])
                Y.append(ids[1:])
                D.append(np.full(L - 1, d, np.int64))
                POS.append(np.arange(L - 1))
    x, y, dd, pos = (np.concatenate(v) for v in (X, Y, D, POS))
    di = np.searchsorted(docs_sorted, dd)
    rid = ptr[di] + pos // 256
    return {"x": x, "y": y, "doc": dd, "pos": pos, "ra_sem": P[rid].argmax(1), "ra_shuf": Psh[rid].argmax(1),
            "g_sem": sem[di], "g_shuf": shuf[di]}


split_rows = None


def h2():
    global split_rows
    from csr.lfr_lib import main_stream
    split, steps, _ = main_stream(245e6)
    split_rows = steps
    z = np.load(TABLES / "psem_main.npz")
    zs = np.load(TABLES / "psem_main_sh.npz")
    P, Psh = z["p"].astype(np.float32), zs["p_sh"].astype(np.float32)
    docs_sorted, ptr = z["docs"], z["ptr"]
    sem, shuf = z["sem"].astype(np.int64), z["shuf"].astype(np.int64)
    cdir = V1 / "teacher_caches/20261122T001339Z_asrs1_S1"
    sel = {}
    traj = {}
    for c in range(1, 11):
        meta = json.loads((cdir / f"cache_c{c:02d}.json").read_text())
        if guard.sha256_file(cdir / f"cache_c{c:02d}.npz") != meta["sha256"]:
            raise guard.GuardError(f"cache c{c} hash mismatch")
        cz = np.load(cdir / f"cache_c{c:02d}.npz")
        st_ = cz["steps"].tolist()
        n_sp = int(cz["n_spec_tokens"])
        r = reconstruct(split, st_, P, Psh, docs_sorted, ptr, sem, shuf)
        if not np.array_equal(r["y"], cz["y"].astype(np.int64)):
            raise guard.GuardError(f"cycle {c}: reconstructed targets differ from the cache")
        fold = hsplit(r["doc"][:n_sp])
        rc = {}
        for b in ("SEM", "SHUF"):
            e = cz[f"e_{b}"][:n_sp].astype(np.int64)
            ra = r["ra_sem" if b == "SEM" else "ra_shuf"][:n_sp]
            g = r["g_sem" if b == "SEM" else "g_shuf"][:n_sp]
            maj = float(np.bincount(e, minlength=16).max() / len(e))
            rc[b] = {"majority_baseline": maj,
                     "target_token": majority_acc(r["y"][:n_sp], e, fold),
                     "input_token_causal": majority_acc(r["x"][:n_sp], e, fold),
                     "region_prior_argmax_direct": float((ra == e).mean()),
                     "region_prior_argmax_mapped": majority_acc(ra, e, fold, top=16),
                     "paper_semantic_group": majority_acc(g, e, fold, top=16),
                     "target_x_region_prior": majority_acc(r["y"][:n_sp] * 16 + ra, e, fold, top=20000),
                     "n_tokens": int(n_sp)}
            traj.setdefault(b, {})[c] = {k: meta["stats"][b][k] for k in
                                         ("s_u", "u_teacher", "u_prior_argmax", "u_mean_specialist", "u_oracle",
                                          "override_frac_spec", "teacher_load_max", "teacher_load_eff")}
        sel[c] = rc
        print(json.dumps({"cycle": c, **{b: {k: round(v, 3) for k, v in rc[b].items()} for b in rc}}), flush=True)
    # causal router agreement with the teacher argmax at the first step of each router phase (out-of-sample)
    m1 = jl(RUN1 / "metrics.jsonl")
    racc = {}
    for rec in m1:
        if rec.get("kind") == "phase_end" and rec.get("phase") == "router":
            for rn in ("SEM_s0", "SEM_s1", "SEM_s2", "SHUF_s0", "SHUF_s1", "SHUF_s2"):
                racc.setdefault(rn, {})[rec["cycle"]] = {"acc_first": rec[rn]["acc_first"], "acc_last": rec[rn]["acc_last"],
                                                         "switch": rec[rn]["switch"]}
    return {"selected_specialist_predictability_training_rows": sel, "utility_trajectory": traj,
            "router_agreement_with_teacher_argmax": racc}


def oracle_sel():
    """Per-token oracle (argmin over the 16-specialist delta table on the SEL swap subset) predictability."""
    out = {}
    for tag, d in (("c00", V1 / "evals/c00"), ("c10", V1 / "evals/20261122T001339Z_asrs1_S1/c10")):
        cm = np.load(d / "common_SEL.npz")
        sub = cm["sub"]
        doc, pos, tgt, paper = cm["doc"], cm["pos"], cm["tgt"], cm["paper"]
        prev = np.concatenate([[-1], tgt[:-1]])
        same = np.concatenate([[False], doc[1:] == doc[:-1]])
        xin = np.where(same & (pos > 0), prev, -1)       # input token = previous target in the document (causal)
        fold = hsplit(paper[sub])
        rr = {}
        for b in ("SEM", "SHUF"):
            bz = np.load(d / f"bank_{b}_SEL.npz")
            ds = bz["d_sub"].astype(np.float32)
            orc = ds.argmin(1)
            dep = np.load(d / f"dep_{b}_s0_SEL.npz")["e"][sub].astype(np.int64)
            lab = cm["lab_sem" if b == "SEM" else "lab_shuf"][sub].astype(np.int64)
            rlab = cm["lab_rsem" if b == "SEM" else "lab_rshuf"][sub].astype(np.int64)
            rr[b] = {"majority_baseline": float(np.bincount(orc, minlength=16).max() / len(orc)),
                     "target_token": majority_acc(tgt[sub], orc, fold),
                     "input_token_causal": majority_acc(xin[sub], orc, fold),
                     "paper_semantic_group": majority_acc(lab, orc, fold, top=16),
                     "region_prior_argmax": majority_acc(rlab, orc, fold, top=16),
                     "deployed_causal_router_s0_direct": float((dep == orc).mean()),
                     "deployed_causal_router_s0_mapped": majority_acc(dep, orc, fold, top=16),
                     "mean_specialist_delta": float(ds.mean()), "oracle_delta": float(ds.min(1).mean()),
                     "n_tokens": int(sub.sum())}
        out[tag] = rr
        print(json.dumps({tag: {b: {k: round(v, 3) for k, v in rr[b].items()} for b in rr}}), flush=True)
    return out


def h3():
    m1 = jl(RUN1 / "metrics.jsonl")
    ph = [r for r in m1 if r.get("kind") == "param_hash"]
    viol = []
    prev = None
    for r in ph:
        if prev is not None:
            ch = {k for k in r["hash"] if r["hash"][k] != prev["hash"][k]}
            allowed = {"base": {"base"}, "refresh": set(), "start": set(),
                       "spec": {"SEM.experts", "SHUF.experts", "ALTG.experts"},
                       "router": {k for k in r["hash"] if k.startswith("router.")} | {"ALTG.router"}}[r["after"]]
            if not ch <= allowed:
                viol.append({"cycle": r["cycle"], "after": r["after"], "unexpected": sorted(ch - allowed)})
        prev = r
    legs = sorted(p.name for p in RUN1.glob("leg_*.json"))
    loads = {}
    for c in range(0, 11):
        p = V1 / ("analysis/c00.json" if c == 0 else f"analysis/20261122T001339Z_asrs1_S1_c{c:02d}.json")
        if p.exists():
            loads[c] = p.name
    st1 = json.loads((V1 / "stage1.json").read_text())
    return {"param_hash_records": len(ph), "phase_isolation_violations": viol, "legs": legs,
            "analysis_files": loads, "stage1_traj_present": "trajectory" in st1}


def main():
    guard.check_root()
    res = {"time": guard.now(), "H1": h1()}
    print(json.dumps(res["H1"], indent=0)[:3000], flush=True)
    res["H2"] = h2()
    res["H2"]["oracle_specialist_predictability_SEL"] = oracle_sel()
    res["H3"] = h3()
    res["code_facts"] = {
        "asrs1_train.py:376": "refresh: u = all_deltas(..., f['h'], f['xn'], Wd, f['y']) on the cycle's specialist+router rows "
                              "(same-token targets y)",
        "asrs1_train.py:393-394": "pi = teacher('R3', u, lp, c, eta, s_u); e = pi.argmax(1)",
        "asrs1_train.py:506-509": "spec_step: e = cache e[lo:hi] for the SAME rows/tokens; correction(xn, e); head_ce_grad(h, a, W, y)",
        "asrs1_train.py:461": "cache_slice asserts the cached targets equal the step targets: same tokens by construction"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    guard.atomic_write_json(OUT, res, overwrite=True)
    print("written", OUT)


if __name__ == "__main__":
    main()
