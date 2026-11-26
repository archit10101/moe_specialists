#!/usr/bin/env python
"""csr-gradient-consistent-latent-edl-v1 Stage 1 (prereg section 3): functional-gradient fingerprints, semantic
embeddings and text-derived nuisance features for every view of a discovery export.

  gc_fingerprint.py --split main|calib_search|calib_sel|val   (GPU via gc_driver.py)
Reads ONLY artifacts/gc-edl-v1/discovery/inputs_<split>.npz, the frozen scout (fle G1), the Pythia tokenizer and the
frozen bge-small encoder. A label firewall (audit hook) is installed before any data is read.
Writes artifacts/gc-edl-v1/fingerprints/fp_<split>.npz: g (P,4,128) float64 CountSketch of the per-view gradient of
blocks.9.ffn.w2.weight, gnorm (P,4) exact Frobenius norm, ce (P,4) scout mean CE, sem (P,4,384) bge embedding,
textf (P,4,7) text nuisance features. Padded views are zero.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402


def load_scout(device):
    import torch
    from csr.fsp_lib import build_draft
    from csr.gc_paths import SCOUT_PATH, SCOUT_SHA, THETA_NAME
    p = guard.ROOT / SCOUT_PATH
    if guard.sha256_file(p) != SCOUT_SHA:
        raise guard.GuardError("scout hash mismatch")
    m = build_draft(0)
    sd = torch.load(p, map_location="cpu", weights_only=True)
    m.load_state_dict(sd["model"])
    m = m.double().to(device).eval() if device.type == "cpu" else m.to(device).eval()
    for q in m.parameters():
        q.requires_grad_(False)
    w = dict(m.named_parameters())[THETA_NAME]
    import hashlib
    th = hashlib.sha256(w.detach().float().cpu().numpy().tobytes()).hexdigest()
    return m, {"theta": THETA_NAME, "shape": list(w.shape), "theta_sha256_fp32": th, "scout": SCOUT_PATH,
               "scout_sha256": SCOUT_SHA}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["main", "calib_search", "calib_sel", "val", "val_windows"])
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--m", type=int, default=128, help="projection dim (256 = engineering revision 1)")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    if not guard.require_execution(a.execute, "gpu" if a.device == "cuda" else "cpu-model"):
        return
    import numpy as np
    import torch
    import transformers  # noqa: F401  (import before the firewall)
    from csr.fle_lib import decode_batch, pythia_tok
    from csr.gc_lib import BGE, Sketch, install_label_firewall, scout_fingerprints, text_features
    from csr.gc_paths import ART, BGE_DIR, DISC, FP, M_PROJ, check_prereg
    check_prereg()
    pythia_tok()
    install_label_firewall()
    mon = None
    if a.device == "cuda":
        guard.occupancy_gate("gc fingerprint")
        device = guard.setup_cuda_device()
        mon = guard.OccupancyMonitor(interval=15.0)
        mon.start()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        device = torch.device("cpu")
    guard.register_self(f"gc fingerprint {a.split}")
    t_iso, t0 = guard.now(), time.time()
    suf = "" if a.m == M_PROJ else f"_m{a.m}"
    out = FP / f"fp_{a.split}{suf}.npz"
    if out.exists():
        print(f"{out} exists")
        return
    if a.split == "val" and not (ART / "frozen.json").exists():
        raise guard.GuardError("VAL fingerprints only after the freeze")
    z = np.load(DISC / f"inputs_{a.split}.npz")
    views, nv = z["views"], z["nv"]
    P = len(nv)
    scout, meta = load_scout(device)
    sk = Sketch(d_in=scout.cfg.dense_ffn, d_out=scout.cfg.d_model, m=a.m)
    flat = [(i, j) for i in range(P) for j in range(int(nv[i]))]
    g = np.zeros((P, 4, a.m), np.float64)
    gn = np.zeros((P, 4), np.float64)
    ce = np.zeros((P, 4), np.float64)
    status = "COMPLETE"
    for s in range(0, len(flat), a.batch):
        if mon is not None and mon.contaminated:
            status = "EXTERNALLY_STOPPED"
            break
        if (ART / "STOP").exists():
            status = "EXTERNALLY_STOPPED"
            break
        idx = flat[s:s + a.batch]
        ii = np.asarray([x[0] for x in idx])
        jj = np.asarray([x[1] for x in idx])
        ids = torch.as_tensor(views[ii, jj].astype(np.int64), device=device)
        gg, nn, cc = scout_fingerprints(scout, ids, sk)
        g[ii, jj] = gg.cpu().numpy()
        gn[ii, jj] = nn.cpu().numpy()
        ce[ii, jj] = cc.cpu().numpy()
    sem = np.zeros((P, 4, 384), np.float32)
    textf = np.zeros((P, 4, 7), np.float32)
    enc_sha = None
    if status == "COMPLETE":
        ii = np.asarray([x[0] for x in flat])
        jj = np.asarray([x[1] for x in flat])
        texts = decode_batch(views[ii, jj])
        textf[ii, jj] = np.asarray([text_features(t) for t in texts], np.float32)
        enc = BGE(device, BGE_DIR)
        enc_sha = enc.sha
        sem[ii, jj] = enc.encode(texts)
        del enc
    if mon is not None:
        mon.stop()
        if mon.contaminated:
            status = "EXTERNALLY_STOPPED"
    if status == "COMPLETE":
        FP.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp.npz")
        with open(tmp, "wb") as f:
            np.savez(f, opaque=z["opaque"], nv=nv, g=g, gnorm=gn, ce=ce, sem=sem, textf=textf)
        tmp.replace(out)
        sha = guard.sha256_file(out)
        meta.update({"sketch": sk.spec(), "encoder": "bge-small-en-v1.5", "encoder_sha256": enc_sha,
                     "inputs_sha256": guard.sha256_file(DISC / f"inputs_{a.split}.npz"), "output_sha256": sha,
                     "n_views": len(flat), "device": a.device, "time": guard.now()})
        guard.atomic_write_json(FP / f"fp_{a.split}{suf}.meta.json", meta, overwrite=True)
        with open(ART / "artifact_hashes.tsv", "a") as f:
            f.write(f"{guard.now()}\t{out.relative_to(guard.ROOT)}\t{sha}\t{out.stat().st_size}\n")
    sec = time.time() - t0
    if a.device == "cuda":
        guard.gpu_ledger(t_iso, guard.now(), f"gc_fingerprint_{a.split}{suf}", "gc fingerprints", status)
        with open(ART / "compute_ledger.tsv", "a") as f:
            f.write(f"{t_iso}\t{guard.now()}\t{sec:.1f}\tfingerprint_{a.split}{suf}\tfingerprint\tdiscovery\t{status}\n")
    print(json.dumps({"status": status, "seconds": sec, "views": len(flat)}))
    sys.exit(0 if status == "COMPLETE" else 77)


if __name__ == "__main__":
    main()
