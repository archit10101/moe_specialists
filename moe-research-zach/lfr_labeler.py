#!/usr/bin/env python
"""csr-latent-forecast-router-distill-v1: independent evaluation labels (protocol 12). Frozen encoder intfloat/e5-base-v2
(artifacts/encoder/e5-base-v2; sha256-verified; never used by drafts, aggregator, router or any loss).
Windows = consecutive 256-token pieces of each document (a trailing piece < 32 tokens joins the previous window);
text = Pythia-decoded window, "passage: " prefix, mean pooling, L2 norm, max 512 e5 tokens.

  calib : embed all CALIB-SEARCH windows; k-means K=16 (20 inits, seeds 2026+i) on paper-half-A windows; assign all.
  val   : (stage 4 only, after frozen.json) embed the VAL audit subset windows; assign to the frozen centroids.
Outputs under artifacts/latent-forecast-router-distill-v1/labels/ (labels only; never enter a forward pass)."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402

ENC = "artifacts/encoder/e5-base-v2"
ENC_SHA = "d0d559c47d5f71b1d280b13b62a2657f3e3bc70c0786f9ab91a36545e6a8f693"
W = 256
K = 16
VAL_TOK_PER_DOM = 200_000


def windows(sp, docs):
    """[(doc, start, end)] windows per document."""
    out = []
    for d in docs:
        L = int(sp.lengths[d])
        starts = list(range(0, L, W))
        if len(starts) > 1 and L - starts[-1] < 32:
            starts.pop()
        for i, s in enumerate(starts):
            e = starts[i + 1] if i + 1 < len(starts) else L
            out.append((int(d), s, e))
    return out


def embed(texts, device):
    import numpy as np
    import torch
    from transformers import BertModel, BertTokenizerFast
    d = guard.ROOT / ENC
    if guard.sha256_file(d / "model.safetensors") != ENC_SHA:
        raise guard.GuardError("encoder hash mismatch")
    tk = BertTokenizerFast(tokenizer_file=str(d / "tokenizer.json"), vocab_file=str(d / "vocab.txt"))
    model = BertModel.from_pretrained(str(d), local_files_only=True, add_pooling_layer=False,
                                      use_safetensors=True).to(device).eval()
    texts = ["passage: " + t for t in texts]
    emb = np.zeros((len(texts), model.config.hidden_size), np.float32)
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    with torch.inference_mode():
        for s in range(0, len(order), 256):
            idx = order[s:s + 256]
            enc = tk([texts[i] for i in idx], padding=True, truncation=True, max_length=512,
                     return_tensors="pt").to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                h = model(**enc).last_hidden_state.float()
            m = enc["attention_mask"].unsqueeze(-1).float()
            e = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
            emb[idx] = torch.nn.functional.normalize(e, dim=-1).cpu().numpy()
    return emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["calib", "val"])
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    if not guard.require_execution(a.execute, "gpu" if a.device == "cuda" else "cpu-model"):
        return
    import numpy as np
    import torch
    from csr.artifacts import load_lm_tokenizer
    from csr.fsp_lib import Split, N_DOM
    from csr.lfr_lib import ART
    from csr.lfr_eval import eval_rows, paper_half
    from csr.specaudit import kmeans
    lab = ART / "labels"
    lab.mkdir(parents=True, exist_ok=True)
    if a.mode == "val" and not (ART / "frozen.json").exists():
        raise guard.GuardError("VAL labels only after the freeze (stage 4)")
    if a.device == "cuda":
        guard.occupancy_gate("lfr labeler")
        device = guard.setup_cuda_device()
    else:
        device = torch.device("cpu")
    guard.register_self(f"lfr labeler {a.mode}")
    t_iso, t0 = guard.now(), time.time()
    tok = load_lm_tokenizer()
    if a.mode == "calib":
        sp = Split("calib_search")
        docs = list(range(sp.n_docs))
    else:
        sp = Split("val")
        docs = [d for r in eval_rows(sp, VAL_TOK_PER_DOM) for d in r]
    win = windows(sp, docs)
    texts = [tok.decode(sp.doc(d)[s:e].tolist()) for d, s, e in win]
    emb = embed(texts, device)
    out = {"mode": a.mode, "split": sp.name, "n_windows": len(win), "window": W, "encoder": ENC,
           "encoder_sha256": ENC_SHA}
    if a.mode == "calib":
        halfA = set(paper_half(sp, 0))
        fit = np.asarray([d in halfA for d, _, _ in win])
        C, labs_fit, inertia, best = kmeans(emb[fit], K, n_init=20, seed=2026)
        C = np.asarray(C, np.float32)
        np.save(lab / "centroids.npy", C)
        out.update(K=K, n_fit=int(fit.sum()), inertia=float(inertia), best_init=int(best),
                   centroids_sha256=guard.sha256_file(lab / "centroids.npy"))
    else:
        C = np.load(lab / "centroids.npy")
    d2 = (emb ** 2).sum(1, keepdims=True) - 2 * emb @ C.T + (C ** 2).sum(1)[None]
    cl = d2.argmin(1).astype(np.int16)
    np.savez(lab / f"windows_{a.mode}.npz", doc=np.asarray([w[0] for w in win]), start=np.asarray([w[1] for w in win]),
             end=np.asarray([w[2] for w in win]), cluster=cl, emb=emb.astype(np.float16))
    out["sha256"] = guard.sha256_file(lab / f"windows_{a.mode}.npz")
    out["cluster_sizes"] = np.bincount(cl, minlength=K).tolist()
    dom = np.asarray([sp.domains[w[0]] for w in win])
    out["cluster_by_domain"] = [[int(((cl == k) & (dom == d)).sum()) for d in range(N_DOM)] for k in range(K)]
    out["seconds"] = time.time() - t0
    guard.atomic_write_json(lab / f"labels_{a.mode}.json", out, overwrite=True)
    if a.device == "cuda":
        guard.gpu_ledger(t_iso, guard.now(), f"lfr_labeler_{a.mode}", "lfr labeler", "COMPLETE")
        with open(ART / "compute_ledger.tsv", "a") as f:
            f.write(f"{t_iso}\t{guard.now()}\t{time.time() - t0:.1f}\tlfr_labeler_{a.mode}\tlabeler_{a.mode}\t"
                    f"labeling\tCOMPLETE\n")
    print(json.dumps({k: v for k, v in out.items() if k != "cluster_by_domain"}))


if __name__ == "__main__":
    main()
