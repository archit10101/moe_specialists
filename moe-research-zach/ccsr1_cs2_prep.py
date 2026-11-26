#!/usr/bin/env python
"""csr-current-context-specialist-router-v1: CALIB-SELECTION (calib_sel) preparation with the FROZEN pipelines.

The csur1/orv1 scripts are read-only and accept only their original split names, so their logic is re-applied here:
  paper labels   orv1_val_assign.paper_table (frozen GC-EDL SEM K16 ProtoNet, residualizer, bge-small; up to 4 views
                 per paper) -> hard argmax per paper; orv1 length-stratified paper shuffle (20 bins, seed 20261302)
  rows           pack_rows over documents with a label, ordered by u_hash("ccsr1-cs2", paper|doc)
  region prior   csur1_psem region pipeline (256-token regions, >= 16 tokens else inherit previous / uniform)
  shuffled prior csur1_psem_shuffle paper transplant (20 token-count bins, cyclic derangement, seed 20261221)

  ccsr1_cs2_prep.py --mode check   recompute the CALIB-SEARCH region prior + transplant with this code and require
                                   agreement with the frozen csur1 tables (argmax agreement >= 0.999, max |dp| <= 2e-3,
                                   transplant argmax agreement >= 0.999); writes ART/cs2_prep_check.json
  ccsr1_cs2_prep.py --mode cs2     requires FROZEN_S2.json and a passing check; writes ART/cs2_assign/{labels_cs2,
                                   parts_cs2}.npz and ART/tables_cs2/psem_calib_sel{,_sh}.{npz,json} (0400)"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402

MIN_TOK = 16
CHUNK = 65536
NBINS_T = 20
SEED_T = 20261221
SHUF_SEED_CS2 = 20261302


def region_prior(sp, docs, dev):
    import numpy as np
    import torch
    from csr.ccsr1_paths import SEM_NET, SEM_NET_SHA, SEM_RES, SEM_RES_SHA
    from csr.fle_lib import decode_batch
    from csr.gc_lib import BGE, ProtoNet
    from csr.gc_paths import BGE_DIR
    for p, h in ((SEM_NET, SEM_NET_SHA), (SEM_RES, SEM_RES_SHA)):
        if guard.sha256_file(guard.ROOT / p) != h:
            raise guard.GuardError(f"hash mismatch {p}")
    L = sp.lengths[docs]
    nreg = (L + 255) // 256
    ptr = np.concatenate([[0], np.cumsum(nreg)]).astype(np.int64)
    R = int(ptr[-1])
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
        e = enc.encode(decode_batch(seqs))
        S = ((e - rp["sm"]) / rp["ssd"]).astype(np.float32)
        with torch.no_grad():
            logits[ii] = net(torch.as_tensor(S)).numpy()
        print(f"{guard.now()} regions {s + len(ii)}/{len(idx_ok)}", flush=True)
    z = logits - logits.max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    for r in np.nonzero(~ok)[0]:
        p[r] = p[r - 1] if (j[r] > 0) else np.full(16, 1.0 / 16)
    return ptr, p, enc.sha, rlen


def transplant(sp, docs, ptr, P):
    import numpy as np
    from orv1_tables import u_hash
    by_paper = {}
    for i, d in enumerate(docs):
        by_paper.setdefault(sp.papers[d], []).append((int(sp.seg_index[d]), i))
    papers = sorted(by_paper, key=lambda p: u_hash("csur1-transplant", p))
    regs = {p: np.concatenate([np.arange(ptr[i], ptr[i + 1]) for _, i in sorted(by_paper[p])]) for p in papers}
    ntok = np.asarray([sum(int(sp.lengths[docs[i]]) for _, i in by_paper[p]) for p in papers], np.float64)
    edges = np.quantile(ntok, np.linspace(0, 1, NBINS_T + 1)[1:-1])
    b = np.searchsorted(edges, ntok, side="right")
    rng = np.random.default_rng(SEED_T)
    src = {}
    for k in range(NBINS_T):
        idx = np.nonzero(b == k)[0]
        perm = idx[rng.permutation(len(idx))]
        for t in range(len(perm)):
            src[papers[perm[t]]] = papers[perm[(t + 1) % len(perm)]]
    p_sh = np.zeros_like(P)
    for p in papers:
        dst, s = regs[p], regs[src[p]]
        nd, ns = len(dst), len(s)
        m = np.rint(np.arange(nd) * (ns - 1) / max(1, nd - 1)).astype(np.int64) if nd > 1 else np.asarray([ns // 2])
        p_sh[dst] = P[s[m]]
    pidx = {p: i for i, p in enumerate(papers)}
    src_of_doc = np.asarray([pidx[src[sp.papers[d]]] for d in docs], np.int64)
    return p_sh, src_of_doc, papers, sum(src[p] == p for p in papers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["check", "cs2"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    if not guard.require_execution(a.execute, "gpu" if a.device == "cuda" else "cpu-model"):
        return
    import numpy as np
    import torch
    from csr.ccsr1_paths import ART, CSUR1_TABLES, ORV1
    from csr.fsp_lib import Split, pack_rows
    if a.device == "cuda":
        guard.occupancy_gate(f"ccsr1 cs2 prep {a.mode}")
        dev = guard.setup_cuda_device()
    else:
        dev = torch.device("cpu")
    guard.register_self(f"ccsr1 cs2 prep {a.mode}")
    t_iso = guard.now()
    if a.mode == "check":
        sp = Split("calib_search")
        pz = np.load(ORV1 / "tables/parts.npz")
        docs = np.unique(np.concatenate([pz["SEL_docs"], pz["EVAL_docs"]])).astype(np.int64)
        ptr, p, enc_sha, _ = region_prior(sp, docs, dev)
        ref = np.load(CSUR1_TABLES / "psem_calib_search.npz")
        refsh = np.load(CSUR1_TABLES / "psem_calib_search_sh.npz")
        if not (np.array_equal(ref["docs"], docs) and np.array_equal(ref["ptr"], ptr)):
            raise guard.GuardError("region layout differs from the frozen CALIB-SEARCH table")
        pr = ref["p"].astype(np.float32)
        p_sh, _, _, selfm = transplant(sp, docs, ptr, pr)
        res = {"argmax_agreement": float((p.argmax(1) == pr.argmax(1)).mean()), "max_abs_dp": float(np.abs(p - pr).max()),
               "transplant_argmax_agreement": float((p_sh.argmax(1) == refsh["p_sh"].astype(np.float32).argmax(1)).mean()),
               "transplant_max_abs": float(np.abs(p_sh.astype(np.float16).astype(np.float32) - refsh["p_sh"].astype(np.float32)).max()),
               "self_mapped": int(selfm), "encoder_sha256": enc_sha, "time": guard.now()}
        res["ok"] = res["argmax_agreement"] >= 0.999 and res["max_abs_dp"] <= 2e-3 and res["transplant_argmax_agreement"] >= 0.999
        guard.atomic_write_json(ART / "cs2_prep_check.json", res, mode=0o400)
        print(json.dumps(res))
        if a.device == "cuda":
            guard.gpu_ledger(t_iso, guard.now(), "ccsr1_cs2_prep_check", "ccsr1 cs2 prep", "COMPLETE")
        sys.exit(0 if res["ok"] else 1)
    if not (ART / "FROZEN_S2.json").exists():
        raise guard.GuardError("CALIB-SELECTION preparation requires FROZEN_S2.json")
    if not json.loads((ART / "cs2_prep_check.json").read_text())["ok"]:
        raise guard.GuardError("CALIB-SEARCH reproduction check did not pass")
    from orv1_val_assign import paper_table
    from orv1_tables import length_stratified_shuffle, u_hash
    sp = Split("calib_sel")
    opq, papers, q, nv, enc_sha = paper_table(sp, dev)
    lab_p = {p: (int(q[i].argmax()) if nv[i] > 0 else -1) for i, p in enumerate(papers)}
    lab = np.asarray([lab_p[p] for p in sp.papers], np.int64)
    sh, st = length_stratified_shuffle(sp, lab, SHUF_SEED_CS2, 20)
    adir, tdir = ART / "cs2_assign", ART / "tables_cs2"
    adir.mkdir(parents=True, exist_ok=True)
    tdir.mkdir(parents=True, exist_ok=True)
    np.savez(adir / "labels_cs2.npz", sem=lab.astype(np.int8), shuf=sh.astype(np.int8))
    docs_r = sorted([i for i in range(sp.n_docs) if lab[i] >= 0], key=lambda i: u_hash("ccsr1-cs2", f"{sp.papers[i]}|{i}"))
    rows = pack_rows(sp.lengths, docs_r)
    np.savez(adir / "parts_cs2.npz", CS2_docs=np.asarray([d for x in rows for d in x], np.int64),
             CS2_row_len=np.asarray([len(x) for x in rows], np.int64))
    docs = np.unique(np.asarray([d for x in rows for d in x], np.int64))
    ptr, p, enc_sha2, rlen = region_prior(sp, docs, dev)
    np.savez(tdir / "psem_calib_sel.npz", docs=docs, ptr=ptr, p=p.astype(np.float16))
    p16 = p.astype(np.float16).astype(np.float32)
    p_sh, src_of_doc, tpapers, selfm = transplant(sp, docs, ptr, p16)
    np.savez(tdir / "psem_calib_sel_sh.npz", p_sh=p_sh.astype(np.float16), src_paper_of_doc=src_of_doc,
             papers=np.asarray(tpapers))
    for f in (adir / "labels_cs2.npz", adir / "parts_cs2.npz", tdir / "psem_calib_sel.npz", tdir / "psem_calib_sel_sh.npz"):
        f.chmod(0o400)
    di = np.repeat(np.arange(len(docs)), np.diff(ptr))
    w = rlen.astype(np.float64)
    am = p.argmax(1)
    meta = {"split": "calib_sel", "n_docs": int(len(docs)), "n_regions": int(ptr[-1]), "tokens": int(sp.lengths[docs].sum()),
            "mean_max_p": float((p.max(1) * w).sum() / w.sum()),
            "within_doc_region_switch_rate": float((am[1:] != am[:-1])[di[1:] == di[:-1]].mean()),
            "encoder_sha256": enc_sha2, "sha256": guard.sha256_file(tdir / "psem_calib_sel.npz"), "time": guard.now()}
    guard.atomic_write_json(tdir / "psem_calib_sel.json", meta, mode=0o400)
    guard.atomic_write_json(tdir / "psem_calib_sel_sh.json", {"self_mapped_papers": int(selfm), "seed": SEED_T,
                                                              "source_psem_sha256": meta["sha256"],
                                                              "sha256": guard.sha256_file(tdir / "psem_calib_sel_sh.npz"),
                                                              "time": guard.now()}, mode=0o400)
    res = {"n_papers": len(papers), "n_with_views": int((nv > 0).sum()), "n_rows": len(rows), "n_docs": len(docs_r),
           "shuffle": st, "encoder_sha256": enc_sha, "psem": meta,
           "sha256": {f: guard.sha256_file(adir / f) for f in ("labels_cs2.npz", "parts_cs2.npz")}, "time": guard.now()}
    guard.atomic_write_json(adir / "cs2_assign.json", res, mode=0o400)
    if a.device == "cuda":
        guard.gpu_ledger(t_iso, guard.now(), "ccsr1_cs2_prep", "ccsr1 cs2 prep", "COMPLETE")
    print(json.dumps(res, default=str))


if __name__ == "__main__":
    main()
