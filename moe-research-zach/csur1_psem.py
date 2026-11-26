#!/usr/bin/env python
"""csr-crossfit-semantic-utility-residual-v1: frozen label-free REGION-level semantic prior p_sem[k | region].

Region j of a document = its tokens [256 j, 256 j + 256) (the last region may be shorter). Each region goes through the
IDENTICAL frozen GC-EDL SEM K16 feature pipeline that produced the C6 / orv1 paper labels (Pythia decode; frozen
bge-small-en-v1.5 CLS embedding; standardisation with the frozen DISC-FIT mean/sd; ProtoNet view logits), and
p_sem = softmax(view logits) (paper_q of a single view). Regions with < 16 tokens inherit the previous region of the
same document (uniform if none). No domain labels are read.

Also writes, per document, the persistent shuffled-prior permutation perm[doc] (16,) used by the SH condition:
p_SH[:, perm[j]] = p_sem[:, j], perm[g_sem(paper)] = g_shuf(paper) (orv1 length-stratified, token-mass-matched paper
shuffle), the other 15 clusters permuted by a per-paper hash. Identical for all documents of a paper.

  csur1_psem.py --split main|calib_search|val  [--device cuda] --execute
Outputs ART/tables/psem_<split>.npz (0400): docs (sorted split doc ids), ptr (n_docs+1), p (n_regions, 16) fp16,
perm (n_docs, 16) int8, sem/shuf (n_docs,) paper-level labels; and psem_<split>.json (stats + sha256)."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402

MIN_TOK = 16
CHUNK = 65536


def split_docs(name, frozen=None):
    import numpy as np
    from csr.csur1_paths import FINAL_TOKENS, ORV1
    if name == "main":
        from csr.lfr_lib import main_stream
        sp, steps, total = main_stream(245_000_000)
        assert total == FINAL_TOKENS, total
        docs = sorted({d for st in steps for r in st for d in r})
        lab = np.load(ORV1 / "tables/labels_main.npz")
        return sp, np.asarray(docs, np.int64), lab["sem"].astype(np.int64), lab["shuf"].astype(np.int64)
    from csr.fsp_lib import Split
    if name == "calib_search":
        sp = Split("calib_search")
        pz = np.load(ORV1 / "tables/parts.npz")
        docs = np.unique(np.concatenate([pz["SEL_docs"], pz["EVAL_docs"]]))
        lab = np.load(ORV1 / "tables/labels_cs.npz")
        return sp, docs.astype(np.int64), lab["sem"].astype(np.int64), lab["shuf"].astype(np.int64)
    if name == "val":
        from csr.csur1_paths import ART
        fz = guard.ROOT / frozen if frozen else ART / "FROZEN_FINAL.json"
        st = json.loads(fz.read_text()) if fz.exists() else None
        if not st:
            raise guard.GuardError("VAL p_sem requires the final configuration to be frozen (FROZEN_FINAL.json)")
        sp = Split("val")
        pz = np.load(ORV1 / "val_assign/parts_val.npz")
        docs = np.unique(pz["VAL_docs"])
        lab = np.load(ORV1 / "val_assign/labels_val.npz")
        return sp, docs.astype(np.int64), lab["sem"].astype(np.int64), lab["shuf"].astype(np.int64)
    raise guard.GuardError(f"unknown split {name}")


def perm_for(paper, gs, gh):
    import numpy as np
    from orv1_tables import u_hash
    rng = np.random.default_rng(int(u_hash("csur1-shufperm", paper) * 2**62))
    perm = np.full(16, -1, np.int64)
    if gs >= 0 and gh >= 0:
        perm[gs] = gh
        src = [j for j in range(16) if j != gs]
        dst = [j for j in range(16) if j != gh]
        perm[src] = rng.permutation(dst)
    else:
        perm[:] = rng.permutation(16)
    return perm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["main", "calib_search", "val"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tables-dir", help="output directory (default ART/tables); used by the cfar1 follow-up")
    ap.add_argument("--frozen", help="freeze record required for VAL (default ART/FROZEN_FINAL.json)")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    if not guard.require_execution(a.execute, "gpu" if a.device == "cuda" else "cpu-model"):
        return
    import numpy as np
    import torch
    from csr.csur1_paths import ART, SEM_NET, SEM_NET_SHA, SEM_RES, SEM_RES_SHA
    from csr.fle_lib import decode_batch
    from csr.gc_lib import BGE, ProtoNet
    from csr.gc_paths import BGE_DIR
    for p, h in ((SEM_NET, SEM_NET_SHA), (SEM_RES, SEM_RES_SHA)):
        if guard.sha256_file(guard.ROOT / p) != h:
            raise guard.GuardError(f"hash mismatch {p}")
    if a.device == "cuda":
        guard.occupancy_gate(f"csur1 psem {a.split}")
        dev = guard.setup_cuda_device()
    else:
        dev = torch.device("cpu")
    guard.register_self(f"csur1 psem {a.split}")
    t_iso = guard.now()
    sp, docs, sem_lab, shuf_lab = split_docs(a.split, a.frozen)
    L = sp.lengths[docs]
    nreg = (L + 255) // 256
    ptr = np.concatenate([[0], np.cumsum(nreg)]).astype(np.int64)
    R = int(ptr[-1])
    # region list: (doc index in docs, start, length)
    di = np.repeat(np.arange(len(docs)), nreg)
    j = np.arange(R) - ptr[di]
    start = j * 256
    rlen = np.minimum(256, L[di] - start)
    ok = rlen >= MIN_TOK
    enc = BGE(dev, BGE_DIR)
    rp = np.load(guard.ROOT / SEM_RES)
    ck = torch.load(guard.ROOT / SEM_NET, map_location="cpu", weights_only=False)
    net = ProtoNet(384, ck["K"], seed=ck["seed"])
    net.load_state_dict(ck["state"])
    net.eval()
    logits = np.zeros((R, 16), np.float32)
    idx_ok = np.nonzero(ok)[0]
    for s in range(0, len(idx_ok), CHUNK):
        ii = idx_ok[s:s + CHUNK]
        seqs = [sp.tokens[sp.offsets[docs[di[r]]] + start[r]: sp.offsets[docs[di[r]]] + start[r] + rlen[r]] for r in ii]
        texts = decode_batch(seqs)
        e = enc.encode(texts)
        S = ((e - rp["sm"]) / rp["ssd"]).astype(np.float32)
        with torch.no_grad():
            logits[ii] = net(torch.as_tensor(S)).numpy()
        print(f"{guard.now()} regions {s + len(ii)}/{len(idx_ok)}", flush=True)
    z = logits - logits.max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    # short regions inherit the previous region of the same document; uniform if the document has none
    for r in np.nonzero(~ok)[0]:
        p[r] = p[r - 1] if (j[r] > 0) else np.full(16, 1.0 / 16)
    perm = np.stack([perm_for(sp.papers[d], int(sem_lab[d]), int(shuf_lab[d])) for d in docs]).astype(np.int8)
    out = (guard.ROOT / a.tables_dir) if a.tables_dir else ART / "tables"
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"psem_{a.split}.npz"
    if f.exists():
        raise guard.GuardError(f"{f} exists")
    np.savez(f, docs=docs, ptr=ptr, p=p.astype(np.float16), perm=perm, sem=sem_lab[docs].astype(np.int8),
             shuf=shuf_lab[docs].astype(np.int8))
    f.chmod(0o400)
    # stats
    w = rlen.astype(np.float64)
    am = p.argmax(1)
    psh_am = perm[di, am].astype(np.int64)
    paper_sem = sem_lab[docs][di]
    has = paper_sem >= 0
    res = {"split": a.split, "n_docs": len(docs), "n_regions": R, "n_short_inherited": int((~ok).sum()),
           "tokens": int(L.sum()), "mean_max_p": float((p.max(1) * w).sum() / w.sum()),
           "mean_entropy_nats": float((-(p * np.log(np.maximum(p, 1e-12))).sum(1) * w).sum() / w.sum()),
           "region_vs_paper_label_agreement": float(((am == paper_sem) * w)[has].sum() / w[has].sum()),
           "mass_region_sem": (np.bincount(am, weights=w, minlength=16) / w.sum()).round(4).tolist(),
           "mass_region_shufprior": (np.bincount(psh_am, weights=w, minlength=16) / w.sum()).round(4).tolist(),
           "within_doc_region_switch_rate": float((am[1:] != am[:-1])[di[1:] == di[:-1]].mean()),
           "encoder_sha256": enc.sha, "net_sha256": SEM_NET_SHA, "sha256": guard.sha256_file(f), "time": guard.now()}
    guard.atomic_write_json(out / f"psem_{a.split}.json", res, mode=0o400)
    if a.device == "cuda":
        guard.gpu_ledger(t_iso, guard.now(), f"csur1_psem_{a.split}", "csur1 psem", "COMPLETE")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
