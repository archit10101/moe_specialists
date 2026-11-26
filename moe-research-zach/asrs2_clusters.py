#!/usr/bin/env python
"""asrs2: frozen label-free K = 5 semantic clusters and region priors (user amendment: exactly K = 5).

Construction (no domain labels, no validation data, no expert-function measurement):
  1. Points = the 16 prototype vectors C_k of the frozen GC-EDL SEM-K16 ProtoNet (artifacts/gc-edl-v1/latent/SEM_K16_s2,
     lambda_coh = 0: pure BGE semantic representation; the pipeline that produced p_sem), unit-normalised (cosine
     space of the ProtoNet scores). Weights m_k = main-stream token mass of regions whose p_sem argmax is k.
  2. Weighted spherical k-means with K = 5: 1000 k-means++ restarts (numpy seeds 0..999), <= 100 Lloyd iterations,
     objective sum_k m_k (1 - cos(C_k, mu_cluster(k))).
  3. Degeneracy floor (prevents a cluster receiving too little data): every cluster's main-stream token share under
     the hard region assignment argmax p5 must be >= 0.05. The selected merge map is the lowest-objective distinct
     partition satisfying the floor (ties: lexicographically smallest canonical map). Sizes are NOT equalised.
  4. p5(c | region) = sum_{k: map[k] = c} p_sem16(k | region); hard assignment = argmax p5. Paper-level group
     sem5 = map[sem16(paper)] (the C6 paper label of the same pipeline).
  5. Persistent shuffled control (same cluster-size distribution, same within-paper structure): per paper a fixed
     permutation sigma_p of the 5 clusters with sigma_p(sem5(p)) = shuf5(p) = map[shuf16(p)] (orv1 length-stratified,
     token-mass-matched paper-label shuffle, so the paper-level label multiset is identical to SEM's) and the other
     four clusters permuted by a per-paper hash; p5_sh[:, sigma_p(c)] = p5[:, c]. [Superseded before any use: see below.] Within-paper switching and entropy
     are identical to SEM by construction. A purely random sigma shrinks large clusters (sorted shares off by ~0.06), so
     and pinning sigma_p(sem5) = shuf5 still leaves 0.017; final rule: papers in sha256 hash order, the whole
     permutation sigma_p (of 120, hash-ordered) keeping the running SHUF region-mass shares closest to the SEM shares.
     Paper-level SHUF group = sigma_p(sem5(p)). Size check: sorted main-stream cluster shares SEM vs SHUF differ by
     <= 0.01; semantic independence check:
     token-weighted NMI(SEM region label, SHUF region label) <= 0.02.

  asrs2_clusters.py build [--splits main,calib_search]     -> ART/tables/k5_merge.json, p5_<split>.npz (0400)
  asrs2_clusters.py cs2                                     -> p5_calib_sel.npz (only after FROZEN_S1.json exists)"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402

K5 = 5
FLOOR = 0.05
SIZE_TOL = 0.01
CSUR1_T = guard.ROOT / "artifacts/csr-crossfit-semantic-utility-residual-v1/tables"
CCSR1_T = guard.ROOT / "artifacts/csr-current-context-specialist-router-v1/tables_cs2"
NET = guard.ROOT / "artifacts/gc-edl-v1/latent/SEM_K16_s2/net.pt"
NET_SHA = "6a5e98673f44c6927ee4f4d52be27dee6684529f2c3425da2e53413f921bdd7a"


def region_tokens(z, lengths_of_docs):
    """Token count of every region (256-token blocks of each document, last one shorter)."""
    ptr = z["ptr"]
    nreg = np.diff(ptr)
    out = np.zeros(ptr[-1], np.float64)
    for i, L in enumerate(lengths_of_docs):
        n = int(nreg[i])
        if n == 0:
            continue
        t = np.clip(L - 256.0 * np.arange(n), 1, 256)
        out[ptr[i]:ptr[i] + n] = t
    return out


def canon(lab):
    """Canonical relabelling (first occurrence order) of a partition of 0..15."""
    m, out = {}, []
    for v in lab:
        m.setdefault(int(v), len(m))
        out.append(m[int(v)])
    return tuple(out)


def kmeans_sph(X, w, k, seed, iters=100):
    rng = np.random.default_rng(seed)
    n = len(X)
    c = [int(rng.choice(n, p=w / w.sum()))]
    for _ in range(1, k):
        d = np.min(1 - X @ X[c].T, 1).clip(min=0) * w
        if d.sum() <= 0:
            return None
        c.append(int(rng.choice(n, p=d / d.sum())))
    mu = X[c].copy()
    lab = None
    for _ in range(iters):
        new = np.argmax(X @ mu.T, 1)
        if lab is not None and np.array_equal(new, lab):
            break
        lab = new
        for j in range(k):
            m = lab == j
            if not m.any():
                return None
            v = (w[m, None] * X[m]).sum(0)
            mu[j] = v / np.linalg.norm(v)
    obj = float((w * (1 - (X * mu[lab]).sum(1))).sum())
    return lab, obj


def nmi(a, b, w):
    J = np.zeros((K5, K5))
    np.add.at(J, (a, b), w)
    J /= J.sum()
    pa, pb = J.sum(1), J.sum(0)
    mi = float((J[J > 0] * np.log(J[J > 0] / np.outer(pa, pb)[J > 0])).sum())
    h = lambda q: float(-(q[q > 0] * np.log(q[q > 0])).sum())
    return mi / max(1e-12, np.sqrt(h(pa) * h(pb)))


def merge_p(p16, mp):
    out = np.zeros((p16.shape[0], K5), np.float32)
    for k in range(16):
        out[:, mp[k]] += p16[:, k].astype(np.float32)
    return out


def sigma_for(paper, s5, t5, salt):
    rest_src = [c for c in range(K5) if c != s5]
    rest_dst = [c for c in range(K5) if c != t5]
    seed = int.from_bytes(hashlib.sha256(f"asrs2-shuf5|{salt}|{paper}".encode()).digest()[:8], "little")
    perm = np.random.default_rng(seed).permutation(len(rest_dst))
    sig = np.zeros(K5, np.int64)
    sig[s5] = t5
    for i, c in enumerate(rest_src):
        sig[c] = rest_dst[perm[i]]
    return sig


def build_split(name, mp, salt=None, log=print):
    """-> arrays dict, stats. name in main | calib_search | calib_sel."""
    from csr.fsp_lib import Split
    if name == "main":
        z = np.load(CSUR1_T / "psem_main.npz")
        sp = Split("main")
        docs = z["docs"]
        sem16, shuf16 = z["sem"].astype(np.int64), z["shuf"].astype(np.int64)
    elif name == "calib_search":
        z = np.load(CSUR1_T / "psem_calib_search.npz")
        sp = Split("calib_search")
        docs = z["docs"]
        sem16, shuf16 = z["sem"].astype(np.int64), z["shuf"].astype(np.int64)
    else:
        z = np.load(CCSR1_T / "psem_calib_sel.npz")
        sp = Split("calib_sel")
        docs = z["docs"]
        sem16, shuf16 = z["sem"].astype(np.int64), z["shuf"].astype(np.int64)
    p5 = merge_p(z["p"], mp)
    sem5, shuf5 = mp[sem16], mp[shuf16]
    papers = [sp.papers[int(d)] for d in docs]
    rt = region_tokens(z, sp.lengths[docs])
    ptr = z["ptr"]
    reg_doc = np.repeat(np.arange(len(docs)), np.diff(ptr))
    a_sem = p5.argmax(1)
    share_sem = np.bincount(a_sem, weights=rt, minlength=K5) / rt.sum()

    def make_sh(salt_):
        """Balanced persistent shuffle: papers in hash order; of the 120 permutations sigma_p of the 5 clusters (hash-ordered)
        choose the one keeping the running SHUF region-mass shares closest to the SEM shares. Uses only the SEM region
        assignment masses (no labels, no function)."""
        import itertools
        reg_mass = np.zeros((len(docs), K5))
        np.add.at(reg_mass, (reg_doc, a_sem), rt)
        by_paper = {}
        for i, pp in enumerate(papers):
            by_paper.setdefault(pp, []).append(i)
        order = sorted(by_paper, key=lambda pp: hashlib.sha256(f"asrs2-shuf5-order|{salt_}|{pp}".encode()).digest())
        target = share_sem
        cum = np.zeros(K5)
        sig = np.zeros((len(docs), K5), np.int64)
        allp = list(itertools.permutations(range(K5)))
        for pp in order:
            idx = by_paper[pp]
            m = reg_mass[idx].sum(0)
            hseed = int.from_bytes(hashlib.sha256(f"asrs2-shuf5|{salt_}|{pp}".encode()).digest()[:8], "little")
            perms = [allp[j] for j in np.random.default_rng(hseed).permutation(len(allp))]
            best, bv = None, None
            tot = cum.sum() + m.sum()
            for pr in perms:
                sg = np.asarray(pr, np.int64)
                new = cum.copy()
                np.add.at(new, sg, m)
                v = float(((new - target * tot) ** 2).sum())
                if bv is None or v < bv - 1e-9:
                    best, bv = sg, v
            np.add.at(cum, best, m)
            sig[idx] = best
        p_sh = np.zeros_like(p5)
        sr = sig[reg_doc]
        for c in range(K5):
            p_sh[np.arange(len(p5)), sr[:, c]] = p5[:, c]
        return sig, p_sh

    if salt is None:
        salt = 0
    sig, p_sh = make_sh(salt)
    dev_ = float(np.abs(np.sort(share_sem) - np.sort(np.bincount(p_sh.argmax(1), weights=rt, minlength=K5) / rt.sum())).max())
    log(f"{name}: balanced shuffle salt {salt}: sorted share deviation {dev_:.4f}")
    shuf5 = sig[np.arange(len(docs)), sem5]          # paper-level SHUF group = sigma_p(sem5(p)) (consistent with p5_sh)
    nm_ = nmi(a_sem, p_sh.argmax(1), rt)
    log(f"{name}: NMI(SEM, SHUF) region labels {nm_:.4f}")
    if dev_ > SIZE_TOL or nm_ > 0.02:
        raise guard.GuardError("balanced shuffle misses the size / independence tolerance")
    share_sh = np.bincount(p_sh.argmax(1), weights=rt, minlength=K5) / rt.sum()
    ent = lambda p: float((-(p * np.log(np.maximum(p, 1e-12))).sum(1) * rt).sum() / rt.sum())
    same = reg_doc[1:] == reg_doc[:-1]
    sw = lambda a: float((a[1:] != a[:-1])[same].mean())
    doc_tok = sp.lengths[docs].astype(np.float64)
    stats = {"split": name, "n_docs": int(len(docs)), "n_regions": int(len(p5)), "salt": int(salt),
             "region_share_sem": share_sem.tolist(), "region_share_shuf": share_sh.tolist(),
             "sorted_share_max_dev": float(np.abs(np.sort(share_sem) - np.sort(share_sh)).max()),
             "paper_label_token_share_sem": (np.bincount(sem5, weights=doc_tok, minlength=K5) / doc_tok.sum()).tolist(),
             "paper_label_token_share_shuf": (np.bincount(shuf5, weights=doc_tok, minlength=K5) / doc_tok.sum()).tolist(),
             "mean_entropy_sem": ent(p5), "mean_entropy_shuf": ent(p_sh), "mean_max_p_sem": float((p5.max(1) * rt).sum() / rt.sum()),
             "within_doc_switch_sem": sw(a_sem), "within_doc_switch_shuf": sw(p_sh.argmax(1)),
             "nmi_sem_vs_shuf_region": nmi(a_sem, p_sh.argmax(1), rt),
             "region_vs_paper_label_agreement_sem": float(((a_sem == sem5[reg_doc]) * rt).sum() / rt.sum())}
    arr = {"docs": docs, "ptr": ptr, "p": p5.astype(np.float16), "p_sh": p_sh.astype(np.float16),
           "sem5": sem5.astype(np.int8), "shuf5": shuf5.astype(np.int8), "sigma": sig.astype(np.int8)}
    return arr, stats


def save_npz(path, arrs):
    tmp = path.with_name(f".{path.stem}.tmp.npz")
    np.savez(tmp, **arrs)
    tmp.rename(path)
    path.chmod(0o400)
    return guard.sha256_file(path)


def build(splits):
    import torch
    from csr.asrs2_paths import TABLES
    guard.check_root()
    TABLES.mkdir(parents=True, exist_ok=True)
    if (TABLES / "k5_merge.json").exists():
        raise guard.GuardError("k5_merge.json exists (frozen); refusing to rebuild")
    if guard.sha256_file(NET) != NET_SHA:
        raise guard.GuardError("ProtoNet hash mismatch")
    ck = torch.load(NET, map_location="cpu", weights_only=False)
    if float(ck["lam_coh"]) != 0.0 or int(ck["K"]) != 16:
        raise guard.GuardError("unexpected ProtoNet (must be SEM K16, lambda_coh 0)")
    C = ck["state"]["C"].numpy().astype(np.float64)
    X = C / np.linalg.norm(C, axis=1, keepdims=True)
    from csr.fsp_lib import Split
    z = np.load(CSUR1_T / "psem_main.npz")
    meta = json.loads((CSUR1_T / "psem_main.json").read_text())
    if guard.sha256_file(CSUR1_T / "psem_main.npz") != meta["sha256"]:
        raise guard.GuardError("psem_main hash mismatch")
    sp = Split("main")
    rt = region_tokens(z, sp.lengths[z["docs"]])
    a16 = z["p"].astype(np.float32).argmax(1)
    w = np.bincount(a16, weights=rt, minlength=16) / rt.sum()
    sols = {}
    for seed in range(1000):
        r = kmeans_sph(X, w, K5, seed)
        if r is None:
            continue
        lab, obj = r
        key = canon(lab)
        if key not in sols or obj < sols[key]["obj"] - 1e-12:
            sols[key] = {"obj": obj, "first_seed": sols.get(key, {}).get("first_seed", seed)}
    p16 = z["p"].astype(np.float32)
    ranked = sorted(sols.items(), key=lambda kv: (round(kv[1]["obj"], 10), kv[0]))
    cand = []
    chosen = None
    for key, v in ranked:
        mp = np.asarray(key, np.int64)
        share = np.bincount(merge_p(p16, mp).argmax(1), weights=rt, minlength=K5) / rt.sum()
        ok = bool(share.min() >= FLOOR)
        cand.append({"map": list(key), "obj": v["obj"], "first_seed": v["first_seed"], "share": share.tolist(), "feasible": ok})
        if ok and chosen is None:
            chosen = mp
        if len(cand) >= 25 and chosen is not None:
            break
    if chosen is None:
        raise guard.GuardError("no feasible K=5 partition")
    out = {"time": guard.now(), "K": K5, "floor": FLOOR, "size_tol": SIZE_TOL, "net": str(NET.relative_to(guard.ROOT)),
           "net_sha256": NET_SHA, "psem_main_sha256": meta["sha256"], "prototype_mass": w.tolist(),
           "n_distinct_partitions": len(sols), "candidates_ranked": cand, "map": chosen.tolist(),
           "procedure": __doc__.split("Construction")[1].split("asrs2_clusters.py build")[0].strip(), "tables": {}}
    for sname in splits:
        arr, stats = build_split(sname, chosen)
        sha = save_npz(TABLES / f"p5_{sname}.npz", arr)
        out["tables"][sname] = {"path": str((TABLES / f"p5_{sname}.npz").relative_to(guard.ROOT)), "sha256": sha, "stats": stats}
        print(json.dumps(stats), flush=True)
    guard.atomic_write_json(TABLES / "k5_merge.json", out, mode=0o400)
    print(json.dumps({"map": out["map"], "n_partitions": len(sols), "top": cand[:5]}, default=str))


def cs2():
    from csr.asrs2_paths import ART, TABLES
    if not (ART / "FROZEN_S1.json").exists():
        raise guard.GuardError("CALIB-SELECTION is locked until FROZEN_S1.json exists")
    m = json.loads((TABLES / "k5_merge.json").read_text())
    if (TABLES / "p5_calib_sel.npz").exists():
        print("exists")
        return
    arr, stats = build_split("calib_sel", np.asarray(m["map"], np.int64))
    sha = save_npz(TABLES / "p5_calib_sel.npz", arr)
    guard.atomic_write_json(TABLES / "p5_calib_sel.json", {"time": guard.now(), "sha256": sha, "stats": stats,
                                                           "merge_map": m["map"]}, mode=0o400)
    print(json.dumps(stats))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        sp_ = sys.argv[sys.argv.index("--splits") + 1].split(",") if "--splits" in sys.argv else ["main", "calib_search"]
        build(sp_)
    elif cmd == "cs2":
        cs2()
