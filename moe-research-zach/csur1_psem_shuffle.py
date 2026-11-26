#!/usr/bin/env python
"""csr-crossfit-semantic-utility-residual-v1: persistent SHUFFLED region prior p_SH by paper-to-paper transplant.

Papers of a split (restricted to the documents in its psem table) are binned into 20 quantile bins of their token
count; within each bin papers are put in a random cyclic order (seeded) and each takes its successor -> source paper
pi(P) != P. Each paper's regions, in
document (segment) order, receive the region priors of pi(P) at the same relative position
(index i -> round(i * (n_src - 1) / max(1, n_dst - 1))). Preserves: the set of prior rows (sharpness/entropy),
within-paper persistence and switching dynamics (of a real paper), paper lengths, number of groups and token mass per
group (length-matched bins); destroys the text-prior link. Label-free.

  csur1_psem_shuffle.py --split main|calib_search|val --execute
Writes ART/tables/psem_<split>_sh.npz (0400): p_sh (R, 16) fp16 aligned with psem_<split>.npz ptr, src_paper per doc;
and psem_<split>_sh.json (stats, sha256)."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402

NBINS = 20
SEED = 20261221


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["main", "calib_search", "val"])
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--tables-dir", help="directory with psem_<split>.npz and for the output (default ART/tables)")
    ap.add_argument("--frozen", help="freeze record required for VAL (default ART/FROZEN_FINAL.json)")
    a = ap.parse_args()
    guard.check_root()
    if not guard.require_execution(a.execute, "cpu-model"):
        return
    import numpy as np
    from csr.csur1_paths import ART
    from csr.fsp_lib import Split
    from orv1_tables import u_hash
    tab = (guard.ROOT / a.tables_dir) if a.tables_dir else ART / "tables"
    fz = (guard.ROOT / a.frozen) if a.frozen else ART / "FROZEN_FINAL.json"
    if a.split == "val" and not fz.exists():
        raise guard.GuardError("VAL shuffled prior requires FROZEN_FINAL.json")
    meta = json.loads((tab / f"psem_{a.split}.json").read_text())
    if guard.sha256_file(tab / f"psem_{a.split}.npz") != meta["sha256"]:
        raise guard.GuardError("psem hash mismatch")
    z = np.load(tab / f"psem_{a.split}.npz")
    docs, ptr, P = z["docs"], z["ptr"], z["p"].astype(np.float32)
    sp = Split(a.split)
    # paper -> ordered list of (doc position in table)
    by_paper = {}
    for i, d in enumerate(docs):
        by_paper.setdefault(sp.papers[d], []).append((int(sp.seg_index[d]), i))
    papers = sorted(by_paper, key=lambda p: u_hash("csur1-transplant", p))
    regs = {p: np.concatenate([np.arange(ptr[i], ptr[i + 1]) for _, i in sorted(by_paper[p])]) for p in papers}
    ntok = np.asarray([sum(int(sp.lengths[docs[i]]) for _, i in by_paper[p]) for p in papers], np.float64)
    edges = np.quantile(ntok, np.linspace(0, 1, NBINS + 1)[1:-1])
    b = np.searchsorted(edges, ntok, side="right")
    rng = np.random.default_rng(SEED)
    src = {}
    for k in range(NBINS):
        idx = np.nonzero(b == k)[0]
        perm = idx[rng.permutation(len(idx))]
        for t in range(len(perm)):  # cyclic derangement: no paper receives its own priors (bins have >= 2 papers)
            src[papers[perm[t]]] = papers[perm[(t + 1) % len(perm)]]
    p_sh = np.zeros_like(P)
    for p in papers:
        dst, s = regs[p], regs[src[p]]
        nd, ns = len(dst), len(s)
        m = np.rint(np.arange(nd) * (ns - 1) / max(1, nd - 1)).astype(np.int64) if nd > 1 else np.asarray([ns // 2])
        p_sh[dst] = P[s[m]]
    out = tab / f"psem_{a.split}_sh.npz"
    if out.exists():
        raise guard.GuardError(f"{out} exists")
    pidx = {p: i for i, p in enumerate(papers)}
    src_of_doc = np.asarray([pidx[src[sp.papers[d]]] for d in docs], np.int64)
    np.savez(out, p_sh=p_sh.astype(np.float16), src_paper_of_doc=src_of_doc, papers=np.asarray(papers))
    out.chmod(0o400)
    w = np.diff(ptr)
    rl = np.zeros(len(P))
    for i, d in enumerate(docs):
        L = int(sp.lengths[d])
        n = ptr[i + 1] - ptr[i]
        rl[ptr[i]:ptr[i + 1]] = np.minimum(256, L - 256 * np.arange(n))
    am, ash = P.argmax(1), p_sh.argmax(1)
    ms = np.bincount(am, weights=rl, minlength=16) / rl.sum()
    mh = np.bincount(ash, weights=rl, minlength=16) / rl.sum()
    soft_s, soft_h = (P * rl[:, None]).sum(0) / rl.sum(), (p_sh * rl[:, None]).sum(0) / rl.sum()
    res = {"split": a.split, "n_papers": len(papers), "n_regions": int(len(P)), "self_mapped_papers": int(sum(src[p] == p for p in papers)),
           "mass_argmax_sem": ms.round(4).tolist(), "mass_argmax_sh": mh.round(4).tolist(),
           "max_abs_mass_diff_argmax": float(np.abs(ms - mh).max()), "max_abs_mass_diff_soft": float(np.abs(soft_s - soft_h).max()),
           "mean_max_p_sem": float((P.max(1) * rl).sum() / rl.sum()), "mean_max_p_sh": float((p_sh.max(1) * rl).sum() / rl.sum()),
           "agreement_argmax_sem_vs_sh": float(((am == ash) * rl).sum() / rl.sum()),
           "nbins": NBINS, "seed": SEED, "source_psem_sha256": meta["sha256"], "sha256": guard.sha256_file(out),
           "time": guard.now()}
    guard.atomic_write_json(tab / f"psem_{a.split}_sh.json", res, mode=0o400)
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, list)}))


if __name__ == "__main__":
    main()
