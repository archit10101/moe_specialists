#!/usr/bin/env python
"""specialization-audit-v1 §5: independent semantic labels for the fixed analysis windows (labels only; they never
enter any model forward pass).

  fetch   : first suitable encoder from the fixed order (e5-base-v2, bge-base-en-v1.5); exact commit resolved via the
            Hub API; safetensors only; <= 1 GiB; into artifacts/encoder/<name>/.        MOE_RESEARCH_APPROVE=download
  embed   : window texts -> L2-normalized embeddings (GPU via scripts/gpu_guarded.sh).  MOE_RESEARCH_APPROVE=gpu
  cluster : deterministic k-means (K=16/32/64) on discovery windows for the encoder view and the TF-IDF/SVD view;
            effect windows assigned to nearest discovery centroid (CPU).                  MOE_RESEARCH_APPROVE=cpu-model
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402
from csr import specaudit as sa  # noqa: E402

AUD = guard.ROOT / "artifacts" / "specialization-audit-v1"
DATA = AUD / "data"
LAB = AUD / "labels"
TEACHER = "sentence-transformers/all-MiniLM-L6-v2"
CANDIDATES = [  # (repo, local name, prefix, pooling); all-mpnet-base-v2 excluded a priori (same family as teacher)
    ("intfloat/e5-base-v2", "e5-base-v2", "passage: ", "mean"),
    ("BAAI/bge-base-en-v1.5", "bge-base-en-v1.5", "", "cls"),
]
FILES = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.txt",
         "special_tokens_map.json"]
MAX_BYTES = 1 << 30


def window_texts():
    import numpy as np
    from csr.artifacts import load_lm_tokenizer
    from csr.data import TokenSplit
    sp = TokenSplit.load(DATA, "audit")
    W = np.load(DATA / "windows.npz")
    tok = load_lm_tokenizer()
    pieces = [sp.doc(int(d))[int(s):int(e)].tolist() for d, s, e in zip(W["doc"], W["start"], W["end"])]
    return tok.decode_batch(pieces), W


def cmd_fetch(a):
    print("fetch first suitable independent encoder:", [c[0] for c in CANDIDATES])
    if not guard.require_execution(a.execute, "download"):
        return
    guard.require_disk(MAX_BYTES, "specaudit encoder download")
    guard.register_self("spec_audit_labels fetch")
    from huggingface_hub import HfApi
    from csr.artifacts import download_pinned
    api = HfApi()
    tried = []
    for repo, name, prefix, pool in CANDIDATES:
        rec = {"repo": repo}
        try:
            guard.ledger("NETWORK", f"specaudit: Hub API model_info {repo}")
            info = api.model_info(repo, files_metadata=True)
            sizes = {s.rfilename: s.size for s in info.siblings}
            rec.update({"revision": info.sha, "files_present": sorted(sizes)})
            missing = [f for f in FILES if f not in sizes]
            if missing:
                rec["unsuitable"] = f"missing files {missing}"
                tried.append(rec)
                continue
            total = sum(sizes[f] or 0 for f in FILES)
            if total > MAX_BYTES:
                rec["unsuitable"] = f"download {total} bytes > 1 GiB"
                tried.append(rec)
                continue
            out = guard.ROOT / "artifacts" / "encoder" / name
            if out.exists():
                raise guard.GuardError(f"{out} exists")
            guard.ledger("NETWORK", f"specaudit: download {repo}@{info.sha} files={FILES} bytes={total}")
            download_pinned(repo, info.sha, FILES, out)
            cfg = json.loads((out / "config.json").read_text())
            if cfg.get("model_type") not in ("bert",) or cfg.get("auto_map"):
                rec["unsuitable"] = f"architecture {cfg.get('model_type')} / remote code"
                tried.append(rec)
                continue
            files = {f: {"sha256": guard.sha256_file(out / f), "bytes": (out / f).stat().st_size} for f in FILES}
            rec.update({"selected": True, "local_dir": str(out.relative_to(guard.ROOT)), "prefix": prefix,
                        "pooling": pool, "files": files, "bytes": total, "model_type": cfg["model_type"],
                        "teacher": TEACHER, "different_family": True})
            tried.append(rec)
            break
        except guard.GuardError:
            raise
        except Exception as e:  # noqa: BLE001
            rec["unsuitable"] = f"{type(e).__name__}: {e}"
            tried.append(rec)
    sel = [t for t in tried if t.get("selected")]
    LAB.mkdir(parents=True, exist_ok=True)
    guard.atomic_write_json(LAB / "encoder_selection.json", {"candidates_tried": tried, "selected": sel[0] if sel else None,
                                                             "created": guard.now()}, mode=0o400)
    guard.ledger("SPEC_AUDIT_ENCODER", json.dumps({"selected": sel[0]["repo"] if sel else None,
                                                   "revision": sel[0]["revision"] if sel else None}))
    print(json.dumps(tried, indent=1))
    if not sel:
        print("no suitable encoder: TF-IDF/SVD view becomes primary (prereg §5)")


def cmd_embed(a):
    sel = json.loads((LAB / "encoder_selection.json").read_text())["selected"]
    print(f"embed audit windows with {sel and sel['repo']}")
    if not guard.require_execution(a.execute, "gpu"):
        return
    out = LAB / "encoder_embeddings.npy"
    if out.exists():
        raise guard.GuardError(f"{out} exists")
    guard.require_disk(200_000_000, "specaudit embeddings")
    guard.occupancy_gate("spec_audit_labels embed")
    guard.setup_cuda_device()
    mon = guard.OccupancyMonitor(10.0)
    mon.start()
    guard.register_self("spec_audit_labels embed (cuda)")
    t_iso, t0 = guard.now(), time.time()
    status, reason = "INCOMPLETE", ""
    try:
        import numpy as np
        import torch
        from transformers import BertModel, BertTokenizerFast
        d = guard.ROOT / sel["local_dir"]
        tk = BertTokenizerFast(tokenizer_file=str(d / "tokenizer.json"), vocab_file=str(d / "vocab.txt"))
        model = BertModel.from_pretrained(str(d), local_files_only=True, add_pooling_layer=False,
                                          use_safetensors=True).to("cuda:0").eval()
        for p in model.parameters():
            p.requires_grad_(False)
        texts, W = window_texts()
        texts = [sel["prefix"] + t for t in texts]
        emb = np.zeros((len(texts), model.config.hidden_size), np.float32)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        n_trunc = 0
        with torch.inference_mode():
            for s in range(0, len(order), 256):
                idx = order[s:s + 256]
                enc = tk([texts[i] for i in idx], padding=True, truncation=True, max_length=512, return_tensors="pt",
                         return_overflowing_tokens=False).to("cuda:0")
                n_trunc += int((enc["attention_mask"].sum(1) >= 512).sum())
                h = model(**enc).last_hidden_state.float()
                if sel["pooling"] == "mean":
                    m = enc["attention_mask"].unsqueeze(-1).float()
                    e = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
                else:
                    e = h[:, 0]
                emb[idx] = torch.nn.functional.normalize(e, dim=-1).cpu().numpy()
                if mon.contaminated:
                    raise guard.GuardError(f"CONTAMINATED: {mon.detail}")
        assert np.isfinite(emb).all()
        np.save(out, emb)
        out.chmod(0o400)
        guard.atomic_write_json(LAB / "encoder_embeddings.json", {
            "encoder": sel["repo"], "revision": sel["revision"], "prefix": sel["prefix"], "pooling": sel["pooling"],
            "max_length": 512, "n_windows": len(texts), "n_hit_max_length": n_trunc, "dim": int(emb.shape[1]),
            "sha256": guard.sha256_file(out), "seconds": time.time() - t0, "created": guard.now(),
            "role": "evaluation labels only; never enters a model forward pass"}, mode=0o400)
        guard.ledger("SPEC_AUDIT_EMBED", f"{out.relative_to(guard.ROOT)} n={len(texts)} encoder={sel['repo']}@{sel['revision']}")
        status = "COMPLETE"
    except guard.GuardError as e:
        status, reason = ("CONTAMINATED" if "CONTAMINATED" in str(e) else "ABORTED"), str(e)
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        status, reason = "FAILED", f"{type(e).__name__}: {e}"
    finally:
        mon.stop()
        if mon.contaminated and status == "COMPLETE":
            status, reason = "CONTAMINATED", mon.detail
        guard.gpu_ledger(t_iso, guard.now(), "specaudit_v1_embed", "spec_audit_labels embed", status)
        print("STATUS", status, reason)
    sys.exit(0 if status == "COMPLETE" else 1)


def cmd_cluster(a):
    print("cluster: k-means K=16/32/64 on discovery windows, encoder + TF-IDF/SVD views")
    if not guard.require_execution(a.execute, "cpu-model"):
        return
    out = LAB / "labels.npz"
    if out.exists():
        raise guard.GuardError(f"{out} exists")
    guard.register_self("spec_audit_labels cluster")
    import numpy as np
    import torch
    torch.set_num_threads(8)
    t0 = time.time()
    texts, W = window_texts()
    disc = W["half"] == 0
    views, info = {}, {}
    emb_path = LAB / "encoder_embeddings.npy"
    if emb_path.exists():
        views["enc"] = np.load(emb_path)
    vocab, idf = sa.tfidf_fit([t for t, d in zip(texts, disc) if d])
    A = sa.tfidf_transform(texts, vocab, idf)
    ip, ix, dv = A
    rows_d = np.nonzero(disc)[0]
    sub_ptr = np.concatenate([[0], np.cumsum(np.diff(ip)[rows_d])])
    sub_ix = np.concatenate([ix[ip[r]:ip[r + 1]] for r in rows_d])
    sub_dv = np.concatenate([dv[ip[r]:ip[r + 1]] for r in rows_d])
    V = sa.randomized_svd_fit((sub_ptr, sub_ix, sub_dv), len(vocab), k=256)
    views["tfidf"] = sa.svd_project(A, len(vocab), V)
    info["tfidf"] = {"vocab_size": len(vocab), "svd_dim": 256, "n_discovery_fit": int(disc.sum()),
                     "empty_rows": int((np.diff(ip) == 0).sum())}
    res = {}
    for vname, X in views.items():
        for K in (16, 32, 64):
            C, lab_d, inertia, best_i = sa.kmeans(X[disc], K)
            lab = np.empty(len(X), np.int64)
            lab[disc] = lab_d
            lab[~disc] = sa.assign_nearest(X[~disc], C)
            res[f"{vname}_K{K}"] = lab
            res[f"{vname}_K{K}_centroids"] = C
            info[f"{vname}_K{K}"] = {"inertia": inertia, "best_init": best_i,
                                     "sizes_discovery": np.bincount(lab_d, minlength=K).tolist(),
                                     "sizes_effect": np.bincount(lab[~disc], minlength=K).tolist()}
            print(vname, K, inertia, best_i, f"{time.time() - t0:.0f}s", flush=True)
    np.savez(out, **res)
    out.chmod(0o400)
    top_terms = {}
    inv_vocab = {i: t for t, i in vocab.items()}
    for vname in views:
        lab = res[f"{vname}_K32"]
        terms = []
        for k in range(32):
            rows = np.nonzero((lab == k) & disc)[0]
            acc = np.zeros(len(vocab))
            for r in rows:
                np.add.at(acc, ix[ip[r]:ip[r + 1]], dv[ip[r]:ip[r + 1]])
            terms.append([inv_vocab[j] for j in np.argsort(-acc)[:12]])
        top_terms[f"{vname}_K32"] = terms
    guard.atomic_write_json(LAB / "labels.json", {
        "views": list(views), "primary_view": "enc" if "enc" in views else "tfidf", "info": info,
        "top_terms_discovery_K32": top_terms, "sha256": guard.sha256_file(out), "seconds": time.time() - t0,
        "seed": sa.SEED, "n_init": 20, "note": "k-means fit on discovery windows only; effect windows assigned to "
                                               "nearest discovery centroid; names descriptive only", "created": guard.now()},
        mode=0o400)
    guard.ledger("SPEC_AUDIT_LABELS", f"{out.relative_to(guard.ROOT)} views={list(views)} K=16/32/64")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "embed", "cluster"])
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    {"fetch": cmd_fetch, "embed": cmd_embed, "cluster": cmd_cluster}[a.cmd](a)


if __name__ == "__main__":
    main()
