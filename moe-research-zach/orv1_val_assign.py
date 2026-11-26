#!/usr/bin/env python
"""oracle-residual-v1 Stage 3 (prereg section 10): label-free semantic assignment of VAL papers with the SAME frozen
GC-EDL SEM K16 prototype network (latent/SEM_K16_s2/net.pt, the medoid of the selected config) and the identical feature
pipeline (gc_data view selection: up to 4 non-overlapping 256-token views per paper; Pythia decode; frozen
bge-small-en-v1.5 CLS embedding; standardisation with the frozen DISC-FIT mean/sd in latent/residualizer_params.npz;
view logits; paper q = softmax of the mean view logit; hard argmax). Nothing is written into gc-edl-v1.

  orv1_val_assign.py --split calib_search   validation: must reproduce the frozen CALIB-SEARCH table
                                           (hard-label agreement >= 0.99, fixed before running)
  orv1_val_assign.py --split val            VAL labels + length-stratified shuffle (seed 20261204) + VAL rows
Outputs under artifacts/oracle-residual-v1/val_assign/ (0400)."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402

NET = "artifacts/gc-edl-v1/latent/SEM_K16_s2/net.pt"
NET_SHA = "6a5e98673f44c6927ee4f4d52be27dee6684529f2c3425da2e53413f921bdd7a"
RES = "artifacts/gc-edl-v1/latent/residualizer_params.npz"
RES_SHA = "40c49358f186e0c39742cdd87566821f56622c5892f7ddf96935dcdf83b2cbae"
SHUF_SEED_VAL = 20261204
AGREE_MIN = 0.99


def paper_table(split, device):
    import numpy as np
    import torch
    from csr.fle_lib import decode_batch
    from csr.gc_lib import BGE, ProtoNet, opaque_id, paper_q, view_logits, view_starts
    from csr.gc_paths import BGE_DIR, VIEW
    for p, h in ((NET, NET_SHA), (RES, RES_SHA)):
        if guard.sha256_file(guard.ROOT / p) != h:
            raise guard.GuardError(f"hash mismatch {p}")
    segs = {}
    for di, (p, sid) in enumerate(zip(split.papers, split.ids)):
        segs.setdefault(p, []).append((int(sid.split("#")[1]), di))
    rows = []
    for p, lst in segs.items():
        lst.sort()
        rows.append((opaque_id(p), p, [di for _, di in lst]))
    rows.sort()
    P = len(rows)
    views = np.zeros((P, 4, VIEW), np.uint16)
    nv = np.zeros(P, np.int64)
    for i, (o, p, dis) in enumerate(rows):
        toks = np.concatenate([split.doc(d) for d in dis])
        st = view_starts(len(toks), VIEW)
        nv[i] = len(st)
        for j, s in enumerate(st):
            views[i, j] = toks[s:s + VIEW]
    ii, jj = np.nonzero(np.arange(4)[None, :] < nv[:, None])
    texts = decode_batch(views[ii, jj])
    enc = BGE(device, BGE_DIR)
    sem = np.zeros((P, 4, 384), np.float32)
    sem[ii, jj] = enc.encode(texts)
    rp = np.load(guard.ROOT / RES)
    S = np.zeros((P, 4, 384), np.float32)
    S[ii, jj] = ((sem[ii, jj] - rp["sm"]) / rp["ssd"]).astype(np.float32)
    ck = torch.load(guard.ROOT / NET, map_location="cpu", weights_only=False)
    net = ProtoNet(384, ck["K"], seed=ck["seed"])
    net.load_state_dict(ck["state"])
    net.eval()
    with torch.no_grad():
        vl = view_logits(net, S)
    q = paper_q(vl, nv)
    return [r[0] for r in rows], [r[1] for r in rows], q, nv, enc.sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["calib_search", "val"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    if not guard.require_execution(a.execute, "gpu" if a.device == "cuda" else "cpu-model"):
        return
    import numpy as np
    import torch
    from csr.fsp_lib import Split, pack_rows
    from csr.orv1_paths import ART, check_prereg
    check_prereg()
    out = ART / "val_assign"
    out.mkdir(parents=True, exist_ok=True)
    if a.split == "val":
        chk = json.loads((out / "check_calib_search.json").read_text())
        ex = json.loads((ART / "stage2_extra_rules.json").read_text())
        if not (chk["ok"] and ex["stage2_promote"]):
            raise guard.GuardError("VAL assignment requires the CALIB-SEARCH replication check and Stage-2 promotion")
    if a.device == "cuda":
        guard.occupancy_gate(f"orv1 val_assign {a.split}")
        dev = guard.setup_cuda_device()
    else:
        dev = torch.device("cpu")
    guard.register_self(f"orv1 val_assign {a.split}")
    t_iso = guard.now()
    sp = Split(a.split)
    opq, papers, q, nv, enc_sha = paper_table(sp, dev)
    if a.split == "calib_search":
        z = np.load(guard.ROOT / "artifacts/gc-edl-v1/latent/table_SEM_calib_search.npz")
        idx = {str(o): i for i, o in enumerate(z["opaque"])}
        has = np.asarray([nv[i] > 0 and str(o) in idx for i, o in enumerate(opq)])
        ref = np.stack([z["q"][idx[str(o)]] if str(o) in idx else np.zeros(16) for o in opq])
        agree = float((q[has].argmax(1) == ref[has].argmax(1)).mean())
        res = {"split": a.split, "n_papers": len(opq), "n_with_views": int((nv > 0).sum()), "hard_agreement": agree,
               "max_abs_q_diff": float(np.abs(q[has] - ref[has]).max()), "ok": agree >= AGREE_MIN,
               "agree_min": AGREE_MIN, "encoder_sha256": enc_sha, "time": guard.now()}
        guard.atomic_write_json(out / "check_calib_search.json", res, mode=0o400)
    else:
        from orv1_tables import length_stratified_shuffle, u_hash
        lab_p = {p: (int(q[i].argmax()) if nv[i] > 0 else -1) for i, p in enumerate(papers)}
        lab = np.asarray([lab_p[p] for p in sp.papers], np.int64)
        from csr.orv1_paths import SHUF_LEN_BINS
        sh, st = length_stratified_shuffle(sp, lab, SHUF_SEED_VAL, SHUF_LEN_BINS)
        np.savez(out / "labels_val.npz", sem=lab.astype(np.int8), shuf=sh.astype(np.int8))
        docs = sorted([i for i in range(sp.n_docs) if lab[i] >= 0],
                      key=lambda i: u_hash("orv1-val", f"{sp.papers[i]}|{i}"))
        r = pack_rows(sp.lengths, docs)
        np.savez(out / "parts_val.npz", VAL_docs=np.asarray([d for x in r for d in x], np.int64),
                 VAL_row_len=np.asarray([len(x) for x in r], np.int64))
        w = sp.lengths[docs] - 1
        res = {"split": "val", "n_papers": len(papers), "n_with_views": int((nv > 0).sum()), "n_docs": len(docs),
               "n_rows": len(r), "tokens": int(w.sum()), "unassigned_tokens": int((sp.lengths[lab < 0] - 1).sum()),
               "mass_sem": (np.bincount(lab[docs], weights=w, minlength=16) / w.sum()).round(4).tolist(),
               "mass_shuf": (np.bincount(sh[docs], weights=w, minlength=16) / w.sum()).round(4).tolist(),
               "shuffle": st, "encoder_sha256": enc_sha, "time": guard.now()}
        for f in ("labels_val.npz", "parts_val.npz"):
            (out / f).chmod(0o400)
            res[f"sha256_{f}"] = guard.sha256_file(out / f)
        guard.atomic_write_json(out / "val_assign.json", res, mode=0o400)
    if a.device == "cuda":
        guard.gpu_ledger(t_iso, guard.now(), f"orv1_val_assign_{a.split}", "orv1 val assign", "COMPLETE")
    print(json.dumps(res, indent=1))
    sys.exit(0 if res.get("ok", True) else 1)


if __name__ == "__main__":
    main()
