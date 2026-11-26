#!/usr/bin/env python
"""csr-semantic-edl-long-adaptive-k-v1 Phase 2A groupings (prereg section 6), CPU only, label firewall on (never reads
domain labels; the only inputs are frozen gc features / views / prototype fits).

  slk_cluster.py fit       K = 32 prototype fits, seeds 0/1/2, with gc_lib.fit_prototypes exactly as gc_discover.fit
                           (SEM representation, lambda_coh 0, DISC-FIT papers with >= 2 views)
  slk_cluster.py metrics   label-free DISC-HOLD metrics for K in {4, 8, 16, 32} with the gc_discover.select code
                           (kappa_view, kappa_seed, coherence C, language-distribution LD, masses, eligibility, medoid,
                           composite S) -> latent/k_metrics.json; reproduces the gc values for K = 4/8/16 (check)
  slk_cluster.py tables    per K: medoid paper_q sharpened T = 0.5, matched paper-permutation control (rng 20261117),
                           calib-search / calib-sel tables; K = 16 must equal table_SEM_T05 (check) -> latent/tables.json"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402

KS = (4, 8, 16, 32)
SEEDS = (0, 1, 2)


def lat():
    from csr.slk_paths import ART
    p = ART / "latent"
    p.mkdir(parents=True, exist_ok=True)
    return p


def src_dir(K, s):
    from csr.slk_paths import GC_LAT
    return (GC_LAT if K != 32 else lat()) / f"SEM_K{K}_s{s}"


def fit():
    import numpy as np
    import torch
    from csr.gc_lib import fit_prototypes, view_logits
    from csr.slk_paths import GC_LAT
    torch.set_num_threads(4)
    fz = np.load(GC_LAT / "features.npz")
    zm = np.load(guard.ROOT / "artifacts/gc-edl-v1/discovery/inputs_main.npz")
    nv, part = zm["nv"], fz["part_main"]
    X = fz["S_main"]
    Fh = fz["F_main"] / np.maximum(1e-12, np.linalg.norm(fz["F_main"], axis=-1, keepdims=True))
    tr = np.nonzero((part == 0) & (nv >= 2))[0]
    for s in SEEDS:
        od = src_dir(32, s)
        if (od / "done.json").exists():
            continue
        t0 = time.time()
        net, hist = fit_prototypes(X[tr], nv[tr], Fh[tr].astype(np.float32), 32, 0.0, s)
        od.mkdir(parents=True, exist_ok=True)
        out = {"main": view_logits(net, X)}
        for sp in ("calib_search", "calib_sel"):
            out[sp] = view_logits(net, fz[f"S_{sp}"])
        np.savez(od / "view_logits.npz", **{k: v.astype(np.float32) for k, v in out.items()})
        torch.save({"state": net.state_dict(), "mu": net.mu, "rep": "SEM", "K": 32, "seed": s, "lam_coh": 0.0},
                   od / "net.pt")
        guard.atomic_write_json(od / "done.json", {"hist": hist, "seconds": time.time() - t0, "n_train_papers": len(tr),
                                                   "lam_coh": 0.0, "time": guard.now(),
                                                   "view_logits_sha256": guard.sha256_file(od / "view_logits.npz")})
        print(json.dumps({"K": 32, "seed": s, "s": round(time.time() - t0, 1), "last": hist[-1]}), flush=True)


def softmax(x):
    import numpy as np
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def metrics(nboot=500):
    """Copy of gc_discover.select's computations restricted to the SEM representation, K in KS."""
    import numpy as np
    from csr.gc_lib import matched_kappa, paper_q, ridge_fit, ridge_pred
    from csr.slk_paths import GC_LAT
    out_p = lat() / "k_metrics.json"
    if out_p.exists():
        print("exists")
        return
    fz = np.load(GC_LAT / "features.npz")
    zm = np.load(guard.ROOT / "artifacts/gc-edl-v1/discovery/inputs_main.npz")
    nv, part = zm["nv"], fz["part_main"]
    Fm = fz["F_main"].astype(np.float64)
    Fh = Fm / np.maximum(1e-12, np.linalg.norm(Fm, axis=-1, keepdims=True))
    nuis = fz["nuis_main"]
    logp_g = fz["logp_uni"]
    V = 50304
    hold = np.nonzero((part == 1) & (nv >= 2))[0]
    fitp = np.nonzero((part == 0) & (nv >= 1))[0]
    hold_all = np.nonzero((part == 1) & (nv >= 1))[0]
    H = len(hold)
    rng = np.random.default_rng(20261115)
    shuf_of = hold[rng.permutation(H)]
    same = shuf_of == hold
    shuf_of[same] = hold[(np.nonzero(same)[0] + 1) % H]
    W = rng.multinomial(H, np.ones(H) / H, size=nboot).astype(np.float64)
    views = zm["views"]

    def contributions(vl):
        K = vl.shape[-1]
        p = softmax(vl)
        a = vl.argmax(-1)
        kv = np.zeros((H, K * K))
        sep = np.zeros((H, K * K))
        for h, d in enumerate(hold):
            n = int(nv[d])
            for i in range(n):
                for j in range(i + 1, n):
                    kv[h, a[d, i] * K + a[d, j]] += 0.5
                    kv[h, a[d, j] * K + a[d, i]] += 0.5
            sep[h, a[d, 0] * K + a[d, n - 1]] += 0.5
            sep[h, a[d, n - 1] * K + a[d, 0]] += 0.5
        ii, jj = np.nonzero(np.arange(4)[None, :] < nv[fitp][:, None])
        pf, ff = p[fitp][ii, jj], Fh[fitp][ii, jj]
        mu = pf.T @ ff
        mu /= np.maximum(1e-12, np.linalg.norm(mu, axis=1, keepdims=True))
        cntk = np.zeros((K, V))
        for pi, (d, j) in enumerate(zip(fitp[ii], jj)):
            np.add.at(cntk, (slice(None), views[d, j].astype(np.int64)), pf[pi][:, None])
        Ug = np.exp(logp_g)
        Uk = 0.9 * cntk / np.maximum(1e-12, cntk.sum(1, keepdims=True)) + 0.1 * Ug[None, :]
        lUk = np.log(Uk)
        csum = np.zeros((H, 2))
        lsum = np.zeros((H, 2))
        npairs = np.zeros(H)
        for h, d in enumerate(hold):
            n = int(nv[d])
            ds = shuf_of[h]
            Pv = p[d, :n]
            Ps = np.repeat(p[ds, :1], n, 0)
            G = Pv @ mu
            G /= np.maximum(1e-12, np.linalg.norm(G, axis=1, keepdims=True))
            Gs = Ps @ mu
            Gs /= np.maximum(1e-12, np.linalg.norm(Gs, axis=1, keepdims=True))
            Ft = Fh[d, :n]
            cm = G @ Ft.T
            cs = Gs @ Ft.T
            off = ~np.eye(n, dtype=bool)
            csum[h, 0] = cm[off].sum()
            csum[h, 1] = cs[off].sum()
            tok = views[d, :n].astype(np.int64)
            M = np.exp(lUk[:, tok])
            base = logp_g[tok]
            mix = np.einsum("ik,knt->int", Pv, M)
            mixs = np.einsum("ik,knt->int", Ps, M)
            gain = (np.log(mix) - base[None]).mean(-1)
            gains = (np.log(mixs) - base[None]).mean(-1)
            lsum[h, 0] = gain[off].sum()
            lsum[h, 1] = gains[off].sum()
            npairs[h] = off.sum()
        return {"kv": kv, "sep": sep, "csum": csum, "lsum": lsum, "npairs": npairs, "K": K}

    def kappa_from_counts(c, K):
        c = c.reshape(K, K)
        n = c.sum()
        po = np.trace(c) / n
        pe = (c.sum(0) / n) @ (c.sum(1) / n)
        return (po - pe) / max(1e-12, 1 - pe)

    res = {}
    t0 = time.time()
    for K in KS:
        key = f"SEM_K{K}"
        per_seed, hard_hold, elig_seed = [], [], []
        for s in SEEDS:
            vl = np.load(src_dir(K, s) / "view_logits.npz")["main"].astype(np.float64)
            c = contributions(vl)
            q = paper_q(vl, nv)
            hh = q[hold_all].argmax(1)
            hard_hold.append(hh)
            frac = np.bincount(hh, minlength=K) / len(hh)
            pbar = q[hold_all].mean(0)
            eff = float(np.exp(-(pbar * np.log(np.maximum(1e-12, pbar))).sum()))
            ii, jj = np.nonzero(np.arange(4)[None, :] < nv[hold_all][:, None])
            pv = softmax(vl[hold_all][ii, jj])
            pz = (pv - pv.mean(0)) / (pv.std(0) + 1e-9)
            fo = (np.arange(len(hold_all))[ii] * 7919) % 5
            r2n = {}
            for name, col in (("scout_ce", 0), ("log_grad_norm", 1)):
                y = nuis[hold_all][ii, jj, col][:, None]
                sse = sst = 0.0
                for f in range(5):
                    trm, tem = fo != f, fo == f
                    Wb = ridge_fit(pz[trm], y[trm], 1.0)
                    sse += float(((y[tem] - ridge_pred(Wb, pz[tem])) ** 2).sum())
                    sst += float(((y[tem] - y[trm].mean()) ** 2).sum())
                r2n[name] = 1 - sse / sst
            el = {"frac_min": float(frac.min()), "frac_max": float(frac.max()), "eff_clusters": eff, "nuis_r2": r2n,
                  "mean_norm_entropy_q": float((-(q[hold_all] * np.log(np.maximum(1e-12, q[hold_all]))).sum(1)).mean()
                                               / math.log(K)),
                  "mean_max_q": float(q[hold_all].max(1).mean())}
            el["eligible_mass"] = bool(frac.min() >= 0.2 / K and frac.max() <= 3.0 / K and eff >= 0.75 * K)
            elig_seed.append(el)
            per_seed.append(c)
        pair_counts, pair_k = [], []
        for x in range(3):
            for y in range(x + 1, 3):
                kap, inv = matched_kappa(hard_hold[x], hard_hold[y], K)
                pair_k.append(kap)
                hx, hy = hard_hold[x], inv[hard_hold[y]]
                pos = np.searchsorted(hold_all, hold)
                cc = np.zeros((H, K * K))
                cc[np.arange(H), hx[pos] * K + hy[pos]] = 1.0
                pair_counts.append(cc)
        medoid = int(np.argmax([np.mean([pair_k[i] for i, (x, y) in enumerate([(0, 1), (0, 2), (1, 2)])
                                         if s in (x, y)]) for s in range(3)]))

        def stats(w):
            kv = np.mean([kappa_from_counts(w @ c["kv"], K) for c in per_seed])
            ks = np.mean([kappa_from_counts(w @ cc, K) for cc in pair_counts])
            C = np.mean([(w @ c["csum"][:, 0] - w @ c["csum"][:, 1]) / (w @ c["npairs"]) for c in per_seed])
            LD = np.mean([(w @ c["lsum"][:, 0] - w @ c["lsum"][:, 1]) / (w @ c["npairs"]) for c in per_seed])
            return np.array([kv, ks, C, LD])
        pt = stats(np.ones(H))
        bs = np.stack([stats(W[b]) for b in range(nboot)])
        ci = {m: [float(np.quantile(bs[:, i], 0.025)), float(np.quantile(bs[:, i], 0.975))]
              for i, m in enumerate(("kappa_view", "kappa_seed", "C", "LD"))}
        eligible = (all(e["eligible_mass"] for e in elig_seed) and ci["kappa_view"][0] > 0 and pt[2] > 0 and pt[3] > 0)
        res[key] = {"K": K, "kappa_view": pt[0], "kappa_seed": pt[1], "C": pt[2], "LD": pt[3], "ci95": ci,
                    "kappa_seed_pairs": pair_k, "eligibility_per_seed": elig_seed, "eligible": bool(eligible),
                    "medoid_seed": medoid, "_bs": bs}
        print(key, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[key].items()
                               if k in ("kappa_view", "kappa_seed", "C", "LD", "eligible", "medoid_seed")}),
              f"{time.time() - t0:.0f}s", flush=True)
    names = ("kappa_view", "kappa_seed", "C", "LD")
    star = {m: max(res[k][m] for k in res if res[k]["eligible"]) if any(res[k]["eligible"] for k in res) else 1.0
            for m in names}
    for k in res:
        pt = np.array([res[k][m] for m in names])
        den = np.array([star[m] for m in names])
        res[k]["S"] = float(np.mean(pt / den))
        res[k]["S_se"] = float((res[k]["_bs"] / den).mean(1).std(ddof=1))
    gcd = json.loads((GC_LAT.parent / "discovery_decision.json").read_text())["metrics"]
    check = {f"SEM_K{K}": {m: [res[f"SEM_K{K}"][m], gcd[f"SEM_K{K}"][m]] for m in names + ("medoid_seed",)}
             for K in (4, 8, 16)}
    repro = all(abs(a - b) < 1e-9 for v in check.values() for a, b in v.values())
    clean = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in res.items()}
    guard.atomic_write_json(out_p, {"time": guard.now(), "metrics": clean, "star": star, "nboot": nboot,
                                    "gc_reproduction": check, "gc_reproduced_exactly": bool(repro),
                                    "seconds": time.time() - t0}, mode=0o400)
    print("gc reproduced:", repro)


def tables():
    import numpy as np
    from csr.gc_lib import paper_q
    from csr.slk_lib import sharpen, shuffle_table
    from csr.slk_paths import C6_SHUF_TABLE, C6_SHUF_TABLE_SHA, C6_TABLE, C6_TABLE_SHA, SEM_CS, SEM_SEL
    km = json.loads((lat() / "k_metrics.json").read_text())["metrics"]
    out_p = lat() / "tables.json"
    if out_p.exists():
        print("exists")
        return
    zm = np.load(guard.ROOT / "artifacts/gc-edl-v1/discovery/inputs_main.npz")
    out = {}
    for K in KS:
        med = km[f"SEM_K{K}"]["medoid_seed"]
        vl = np.load(src_dir(K, med) / "view_logits.npz")
        q = paper_q(vl["main"].astype(np.float64), zm["nv"])
        qs = sharpen(q, 0.5)
        sh = shuffle_table(zm["opaque"], qs, 20261117)
        rec = {"medoid_seed": med}
        if K == 16:
            z6 = np.load(guard.ROOT / C6_TABLE)
            assert (z6["opaque"] == zm["opaque"]).all()
            same = bool(np.allclose(z6["q"], qs.astype(np.float32), atol=1e-6))
            z6s = np.load(guard.ROOT / C6_SHUF_TABLE)
            same_s = bool(np.allclose(z6s["q"], sh.astype(np.float32), atol=1e-6))
            if not (same and same_s):
                raise guard.GuardError("K=16 does not reproduce the C6 / SHUF-C6 tables")
            rec.update({"sem": C6_TABLE, "sem_sha256": C6_TABLE_SHA, "shuf": C6_SHUF_TABLE,
                        "shuf_sha256": C6_SHUF_TABLE_SHA, "reproduces_c6": True,
                        "calib_search": str(SEM_CS.relative_to(guard.ROOT)), "calib_search_sha256":
                        guard.sha256_file(SEM_CS), "calib_sel": str(SEM_SEL.relative_to(guard.ROOT)),
                        "calib_sel_sha256": guard.sha256_file(SEM_SEL)})
        else:
            for name, arr in (("sem", qs), ("shuf", sh)):
                p = lat() / f"table_{name.upper()}_K{K}_T05.npz"
                np.savez(p, opaque=zm["opaque"], q=arr.astype(np.float32), K=np.int64(K))
                rec[name], rec[f"{name}_sha256"] = str(p.relative_to(guard.ROOT)), guard.sha256_file(p)
            for sp in ("calib_search", "calib_sel"):
                zc = np.load(guard.ROOT / f"artifacts/gc-edl-v1/discovery/inputs_{sp}.npz")
                p = lat() / f"table_SEM_K{K}_{sp}.npz"
                np.savez(p, opaque=zc["opaque"], q=paper_q(vl[sp].astype(np.float64), zc["nv"]).astype(np.float32),
                         K=np.int64(K))
                rec[sp], rec[f"{sp}_sha256"] = str(p.relative_to(guard.ROOT)), guard.sha256_file(p)
        has = q.sum(1) > 0
        rec["mean_max_q_T05"] = float(qs[has].max(1).mean())
        rec["mass_min_x_uniform"] = float(np.bincount(qs[has].argmax(1), minlength=K).min() / has.sum() * K)
        out[f"K{K}"] = rec
    out["time"] = guard.now()
    guard.atomic_write_json(out_p, out, mode=0o400)
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "metrics", "tables", "all"])
    a = ap.parse_args()
    guard.check_root()
    from csr.slk_paths import check_prereg
    check_prereg()
    if a.cmd in ("fit", "all"):
        fit()
    if a.cmd in ("metrics", "all"):
        metrics()
    if a.cmd in ("tables", "all"):
        tables()


if __name__ == "__main__":
    main()
