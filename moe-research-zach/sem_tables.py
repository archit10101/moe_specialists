#!/usr/bin/env python
"""csr-semantic-edl-v1: frozen derived assignment tables (prereg sections 6-7), built before any model result.
  table_SEM_T05.npz      : q^(1/0.5) renormalised (sharpened, T = 0.5) for papers with an assignment
  table_SHUF_SEM.npz     : q permuted across papers that have an assignment (fixed seed 20261117; load-preserving)
  table_SHUF_SEM_T05.npz : the sharpened table under the same permutation
Writes artifacts/semantic-edl-v1/tables/ + tables.json (hashes, agreement diagnostics vs the source)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402


def main():
    guard.check_root()
    import numpy as np
    from csr.sem_paths import ART, SEM_TABLE, SEM_TABLE_SHA, check_prereg
    check_prereg()
    od = ART / "tables"
    od.mkdir(parents=True, exist_ok=True)
    if (od / "tables.json").exists():
        print("exists")
        return
    if guard.sha256_file(SEM_TABLE) != SEM_TABLE_SHA:
        raise guard.GuardError("SEM table hash mismatch")
    z = np.load(SEM_TABLE)
    q = z["q"].astype(np.float64)
    has = np.nonzero(q.sum(1) > 0)[0]
    qs = np.zeros_like(q)
    qs[has] = q[has] ** 2
    qs[has] /= qs[has].sum(1, keepdims=True)
    perm = np.random.default_rng(20261117).permutation(has)
    out = {}

    def save(name, arr):
        p = od / f"{name}.npz"
        np.savez(p, opaque=z["opaque"], q=arr.astype(np.float32), K=np.int64(arr.shape[1]))
        out[name] = {"path": str(p.relative_to(guard.ROOT)), "sha256": guard.sha256_file(p),
                     "mean_max_q": float(arr[has].max(1).mean()),
                     "mean_norm_entropy": float((-(arr[has] * np.log(np.maximum(1e-12, arr[has]))).sum(1)).mean()
                                                / np.log(arr.shape[1]))}
    save("table_SEM_T05", qs)
    sh = q.copy()
    sh[has] = q[perm]
    save("table_SHUF_SEM", sh)
    shs = qs.copy()
    shs[has] = qs[perm]
    save("table_SHUF_SEM_T05", shs)
    a, b = q[has].argmax(1), sh[has].argmax(1)
    K = q.shape[1]
    C = np.zeros((K, K))
    np.add.at(C, (a, b), 1)
    p = C / C.sum()
    pa, pb = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    mi = float((p[p > 0] * np.log(p[p > 0] / (pa @ pb)[p > 0])).sum())
    out["shuffle_check"] = {"hard_agreement": float((a == b).mean()),
                            "chance_agreement": float((C.sum(1) / C.sum()) @ (C.sum(0) / C.sum())),
                            "mi_nats_sem_vs_shuf": mi, "n_papers": int(len(has)),
                            "same_paper_fraction_kept": float((perm == has).mean())}
    out["source"] = {"path": str(SEM_TABLE.relative_to(guard.ROOT)), "sha256": SEM_TABLE_SHA}
    out["time"] = guard.now()
    guard.atomic_write_json(od / "tables.json", out, mode=0o400)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
