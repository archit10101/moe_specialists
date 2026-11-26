#!/usr/bin/env python
"""spb1 Stage 0 (CPU): frozen persistent shuffled control with IDENTICAL cluster sizes (token mass) to the semantic
clusters. No domain label, validation result or expert measurement is read.

Semantic clusters: the frozen orv1 hard labels (argmax of the GC-EDL SEM K16 label-free paper table), reused as-is.
Shuffled clusters: every paper gets ONE persistent random cluster (constant for all of its documents). Papers are
visited in a fixed hash order and each is assigned cluster k with probability proportional to the remaining capacity
min_b(R_b[k] / T_b[k]) over the token-mass targets b (main: the 50M Stage-1 prefix and the full 245M plan; CALIB-SEARCH:
each paper half separately), where T_b[k] = semantic token mass of cluster k and R_b[k] = T_b[k] - mass assigned so far.
This matches the cluster-size distribution to ~1e-3 while being independent of semantics (NMI reported).

Writes TABLES/labels_main.npz (sem, shuf per main doc), labels_cs.npz (sem, shuf per CALIB-SEARCH doc),
tables.json (mass tables, errors, NMI, hashes); all 0400."""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402


def u_hash(tag, s):
    return int.from_bytes(hashlib.sha256(f"{tag}|{s}".encode()).digest()[:8], "little") / 2**64


def nmi(a, b, K):
    import numpy as np
    n = len(a)
    c = np.zeros((K, K))
    np.add.at(c, (a, b), 1)
    pa, pb, pab = c.sum(1) / n, c.sum(0) / n, c / n
    nz = pab > 0
    mi = (pab[nz] * np.log(pab[nz] / np.outer(pa, pb)[nz])).sum()
    ha = -(pa[pa > 0] * np.log(pa[pa > 0])).sum()
    hb = -(pb[pb > 0] * np.log(pb[pb > 0])).sum()
    return float(mi / max(1e-12, np.sqrt(ha * hb)))


def balanced_shuffle(papers, plab, masses, seed, tag, K):
    """papers: list of paper ids; plab: semantic label per paper; masses: list of arrays (one per target), token mass
    of each paper in that target. Returns the shuffled label per paper."""
    import numpy as np
    order = sorted(range(len(papers)), key=lambda i: u_hash(tag, papers[i]))
    T = [np.bincount(plab, weights=m, minlength=K) for m in masses]
    A = [np.zeros(K) for _ in masses]
    rng = np.random.default_rng(seed)
    out = np.full(len(papers), -1, np.int64)
    for i in order:
        score = np.min(np.stack([np.clip((T[b] - A[b]) / np.maximum(T[b], 1e-9), 0, None) for b in range(len(T))]), 0)
        if score.sum() <= 0:
            score = np.min(np.stack([T[b] / T[b].sum() for b in range(len(T))]), 0)
        k = int(rng.choice(K, p=score / score.sum()))
        out[i] = k
        for b in range(len(T)):
            A[b][k] += masses[b][i]
    return refine(out, masses, T, A, rng, K)


def refine(out, masses, T, A, rng, K, n_prop=400_000):
    """Semantics-blind local search on token mass only: propose moving a random paper to a random cluster or swapping
    two papers' clusters; accept iff the summed squared share error over all targets decreases."""
    import numpy as np
    M = np.stack(masses, 1)                       # (P, B)
    Tt = np.stack(T, 1)                           # (K, B)
    tot = Tt.sum(0)
    Aa = np.stack(A, 1)

    def err(Ax):
        return float((((Ax - Tt) / tot) ** 2).sum())

    cur = err(Aa)
    P = len(out)
    for it in range(n_prop):
        i = int(rng.integers(P))
        if it % 2 == 0:
            k = int(rng.integers(K))
            if k == out[i]:
                continue
            new = Aa.copy()
            new[out[i]] -= M[i]
            new[k] += M[i]
            e = err(new)
            if e < cur:
                Aa, cur, out[i] = new, e, k
        else:
            j = int(rng.integers(P))
            if out[i] == out[j]:
                continue
            new = Aa.copy()
            new[out[i]] += M[j] - M[i]
            new[out[j]] += M[i] - M[j]
            e = err(new)
            if e < cur:
                Aa, cur = new, e
                out[i], out[j] = out[j], out[i]
    return out


def shares(lab, w, K):
    import numpy as np
    m = np.bincount(lab, weights=w, minlength=K)
    return m / m.sum()


def main():
    guard.check_root()
    import numpy as np
    from csr.fsp_lib import Split
    from csr.orv1_lib import unflatten_rows
    from csr.spb1_paths import K, ORV1_TABLES, SHUF_SEED_CS, SHUF_SEED_MAIN, TABLES, verify_reused
    from csr import spb1_paths as P
    TABLES.mkdir(parents=True, exist_ok=True)
    if (TABLES / "tables.json").exists():
        print("exists")
        return
    reused = verify_reused()
    out = {"time": guard.now(), "reused_sha256": reused}
    # ---------------- main split
    sp = Split("main")
    lab_m = np.load(ORV1_TABLES / "labels_main.npz")["sem"].astype(np.int64)
    z = np.load(ORV1_TABLES / "plan.npz")
    rows = unflatten_rows(z["docs"], z["row_len"])
    rps = int(z["rows_per_step"])
    steps = [rows[i:i + rps] for i in range(0, len(rows), rps)]
    L = sp.lengths
    cum = np.cumsum([sum(int(L[d]) - 1 for r in st for d in r) for st in steps])
    n50 = int(np.searchsorted(cum, P.BUDGETS["S1"]) + 1)
    docs50 = np.asarray([d for st in steps[:n50] for r in st for d in r])
    docsall = np.asarray([d for st in steps for r in st for d in r])
    papers = sorted({sp.papers[d] for d in range(sp.n_docs) if lab_m[d] >= 0})
    pidx = {p: i for i, p in enumerate(papers)}
    pdoc = np.asarray([pidx.get(p, -1) for p in sp.papers])
    plab = np.full(len(papers), -1, np.int64)
    plab[pdoc[lab_m >= 0]] = lab_m[lab_m >= 0]
    m50 = np.bincount(pdoc[docs50], weights=L[docs50] - 1, minlength=len(papers))
    mall = np.bincount(pdoc[docsall], weights=L[docsall] - 1, minlength=len(papers))
    mtot = np.bincount(pdoc[pdoc >= 0], weights=L[pdoc >= 0] - 1, minlength=len(papers))
    psh = balanced_shuffle(papers, plab, [m50, mall, mtot], SHUF_SEED_MAIN, "spb1-shuf-main", K)
    sh_m = np.where(pdoc >= 0, psh[np.clip(pdoc, 0, None)], -1)
    info = {"n_papers": len(papers), "stage1_steps": n50, "stage1_tokens": int(cum[n50 - 1]),
            "paper_nmi_sem_shuf": nmi(plab, psh, K), "paper_agreement": float((plab == psh).mean())}
    for name, docs in (("S1_50M", docs50), ("S3_245M", docsall)):
        a, b = shares(lab_m[docs], L[docs] - 1, K), shares(sh_m[docs], L[docs] - 1, K)
        info[name] = {"mass_sem": a.round(5).tolist(), "mass_shuf": b.round(5).tolist(),
                      "max_abs_diff": float(np.abs(a - b).max()), "max_rel_diff": float((np.abs(a - b) / a).max()),
                      "min_share": float(a.min())}
    out["main"] = info
    # ---------------- CALIB-SEARCH (each paper half matched separately)
    cs = Split("calib_search")
    lab_c = np.load(ORV1_TABLES / "labels_cs.npz")["sem"].astype(np.int64)
    zp = np.load(ORV1_TABLES / "parts.npz")
    sh_c = np.full(cs.n_docs, -1, np.int64)
    out["cs"] = {}
    for half, name, seed_off in (("SEL", "SEL", 0), ("EVAL", "TRAIN", 1)):
        rows_h = unflatten_rows(zp[f"{half}_docs"], zp[f"{half}_row_len"])
        docs = np.asarray([d for r in rows_h for d in r])
        pp = sorted({cs.papers[d] for d in docs})
        pi = {p: i for i, p in enumerate(pp)}
        pd = np.asarray([pi[cs.papers[d]] for d in docs])
        pl = np.full(len(pp), -1, np.int64)
        pl[pd] = lab_c[docs]
        mass = np.bincount(pd, weights=cs.lengths[docs] - 1, minlength=len(pp))
        ps = balanced_shuffle(pp, pl, [mass], SHUF_SEED_CS + seed_off, f"spb1-shuf-cs-{half}", K)
        sh_c[docs] = ps[pd]
        a, b = shares(lab_c[docs], cs.lengths[docs] - 1, K), shares(sh_c[docs], cs.lengths[docs] - 1, K)
        out["cs"][name] = {"orv1_half": half, "n_papers": len(pp), "n_docs": len(docs),
                           "tokens": int((cs.lengths[docs] - 1).sum()), "mass_sem": a.round(5).tolist(),
                           "mass_shuf": b.round(5).tolist(), "max_abs_diff": float(np.abs(a - b).max()),
                           "paper_nmi_sem_shuf": nmi(pl, ps, K)}
    np.savez(TABLES / "labels_main.npz", sem=lab_m.astype(np.int8), shuf=sh_m.astype(np.int8))
    np.savez(TABLES / "labels_cs.npz", sem=lab_c.astype(np.int8), shuf=sh_c.astype(np.int8))
    out["sha256"] = {}
    for f in sorted(TABLES.glob("*.npz")):
        f.chmod(0o400)
        out["sha256"][f.name] = guard.sha256_file(f)
    guard.atomic_write_json(TABLES / "tables.json", out, mode=0o400)
    print(json.dumps({k: v for k, v in out.items() if k not in ("sha256", "reused_sha256")}, indent=1)[:5000])


if __name__ == "__main__":
    main()
