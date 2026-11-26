#!/usr/bin/env python
"""csr-forecast-latent-edl-v1: e5 evaluation labels for the CALIB-SELECTION audit subset (stage 3 only; requires
STAGE3_OPEN.json). Same windows / encoder / frozen K=16 centroids as the lfr labeler (imported read-only; lfr files are
never written). Output: artifacts/forecast-latent-edl-v1/labels/windows_calib_sel.npz (labels only)."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402

AUDIT_TOK = 200_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    if not guard.require_execution(a.execute, "gpu" if a.device == "cuda" else "cpu-model"):
        return
    import numpy as np
    import torch
    from lfr_labeler import embed, windows
    from csr.artifacts import load_lm_tokenizer
    from csr.fsp_lib import Split
    from csr.fle_paths import ART
    from csr.lfr_eval import eval_rows
    if not (ART / "STAGE3_OPEN.json").exists():
        raise guard.GuardError("CALIB-SELECTION labels only at stage 3")
    device = guard.setup_cuda_device() if a.device == "cuda" else torch.device("cpu")
    t0 = time.time()
    C_p = guard.ROOT / "artifacts/latent-forecast-router-distill-v1/labels/centroids.npy"
    if guard.sha256_file(C_p) != "cddae66fee68891a808002553551501c8857931f1aae96e994047b0e76364d58":
        raise guard.GuardError("lfr centroids hash mismatch")
    C = np.load(C_p)
    sp = Split("calib_sel")
    docs = [d for r in eval_rows(sp, AUDIT_TOK) for d in r]
    win = windows(sp, docs)
    tok = load_lm_tokenizer()
    emb = embed([tok.decode(sp.doc(d)[s:e].tolist()) for d, s, e in win], device)
    d2 = (emb ** 2).sum(1, keepdims=True) - 2 * emb @ C.T + (C ** 2).sum(1)[None]
    cl = d2.argmin(1).astype(np.int16)
    od = ART / "labels"
    od.mkdir(exist_ok=True)
    np.savez(od / "windows_calib_sel.npz", doc=np.asarray([w[0] for w in win]), start=np.asarray([w[1] for w in win]),
             end=np.asarray([w[2] for w in win]), cluster=cl, emb=emb.astype(np.float16))
    out = {"split": "calib_sel", "n_windows": len(win), "sha256": guard.sha256_file(od / "windows_calib_sel.npz"),
           "cluster_sizes": np.bincount(cl, minlength=16).tolist(), "seconds": time.time() - t0, "time": guard.now()}
    guard.atomic_write_json(od / "labels_calib_sel.json", out, mode=0o400)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
