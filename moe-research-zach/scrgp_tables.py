#!/usr/bin/env python
"""scrgp-v1 Stage 0: frozen derived CALIB-SEARCH assignment tables and label-free row partitions (written once, 0400).
  table_SEM_T05_CS.npz      : GC-EDL SEM K16 CALIB-SEARCH assignment sharpened exactly like Semantic C6 (q^2 renormalised)
  table_SHUF_SEM_T05_CS.npz : the same rows permuted across CALIB-SEARCH papers (fixed seed 20261120; persistent,
                              load-preserving shuffled clusters)
  partitions.json           : A1 / A2 (estimation; CALIB-SEARCH paper-half A split by a paper hash) and B (held-out;
                              paper-half B); documents in a fixed hash order, taken until the token cap; packed rows.
No domain label is read."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402

CAPS = {"A1": 400_000, "A2": 400_000, "B": 500_000}


def main():
    guard.check_root()
    import numpy as np
    from csr.fsp_lib import Split, pack_rows
    from csr.lfr_eval import paper_half
    from csr.scrgp_lib import u_hash
    from csr.scrgp_paths import SEM_CS, SEM_CS_SHA, SHUF_SEED, TABLE_DIR
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    if (TABLE_DIR / "tables.json").exists():
        print("exists")
        return
    if guard.sha256_file(SEM_CS) != SEM_CS_SHA:
        raise guard.GuardError("SEM calib_search table hash mismatch")
    z = np.load(SEM_CS)
    q = z["q"].astype(np.float64)
    has = np.nonzero(q.sum(1) > 0)[0]
    qs = np.zeros_like(q)
    qs[has] = q[has] ** 2
    qs[has] /= qs[has].sum(1, keepdims=True)
    perm = np.random.default_rng(SHUF_SEED).permutation(has)
    sh = qs.copy()
    sh[has] = qs[perm]
    out = {"source": {"path": str(SEM_CS.relative_to(guard.ROOT)), "sha256": SEM_CS_SHA}, "time": guard.now()}
    for name, arr in (("table_SEM_T05_CS", qs), ("table_SHUF_SEM_T05_CS", sh)):
        p = TABLE_DIR / f"{name}.npz"
        np.savez(p, opaque=z["opaque"], q=arr.astype(np.float32), K=np.int64(arr.shape[1]))
        p.chmod(0o400)
        out[name] = {"path": str(p.relative_to(guard.ROOT)), "sha256": guard.sha256_file(p),
                     "mean_max_q": float(arr[has].max(1).mean()), "n_papers": int(len(has))}
    a, b = qs[has].argmax(1), sh[has].argmax(1)
    out["shuffle_check"] = {"hard_agreement": float((a == b).mean()), "same_paper_fraction_kept": float((perm == has).mean())}
    cs = Split("calib_search")
    halfA, halfB = paper_half(cs, 0), paper_half(cs, 1)
    order = lambda docs: sorted(docs, key=lambda i: u_hash("scrgp-order", f"{cs.papers[i]}|{i}"))  # noqa: E731
    sub = lambda i: u_hash("scrgp-sub", cs.papers[i]) < 0.5  # noqa: E731
    pools = {"A1": order([i for i in halfA if sub(i)]), "A2": order([i for i in halfA if not sub(i)]),
             "B": order(halfB)}
    parts = {}
    for name, pool in pools.items():
        docs, used = [], 0
        for i in pool:
            if used >= CAPS[name]:
                break
            docs.append(int(i))
            used += int(cs.lengths[i]) - 1
        rows = pack_rows(cs.lengths, docs)
        parts[name] = {"docs": docs, "rows": rows, "tokens": used, "n_docs": len(docs),
                       "n_papers": len({cs.papers[i] for i in docs}), "n_rows": len(rows)}
    pa = {cs.papers[i] for n in ("A1", "A2") for i in parts[n]["docs"]}
    pb = {cs.papers[i] for i in parts["B"]["docs"]}
    p1 = {cs.papers[i] for i in parts["A1"]["docs"]}
    p2 = {cs.papers[i] for i in parts["A2"]["docs"]}
    if pa & pb or p1 & p2:
        raise guard.GuardError("partition not paper-disjoint")
    pj = TABLE_DIR / "partitions.json"
    guard.atomic_write_json(pj, {"caps": CAPS, "parts": parts, "paper_disjoint": True, "time": guard.now()}, mode=0o400)
    out["partitions"] = {"path": str(pj.relative_to(guard.ROOT)), "sha256": guard.sha256_file(pj),
                         **{n: {k: parts[n][k] for k in ("tokens", "n_docs", "n_papers", "n_rows")} for n in parts}}
    guard.atomic_write_json(TABLE_DIR / "tables.json", out, mode=0o400)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
