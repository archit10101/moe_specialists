#!/usr/bin/env python
"""Build deterministic boundary caches (fixed / semantic / random) and frozen chunk-prefix embeddings.

tau is calibrated on TRAINING documents only (seeded sample) so that the mean non-final semantic chunk length
is ~128 tokens. The random length distribution is fit on training semantic chunks only. The test split is
never read. CPU only. Requires --execute and MOE_RESEARCH_APPROVE=cpu-model.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402

SPLITS = ("train", "calib", "val")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--data", default="artifacts/data/fwe-v1")
    ap.add_argument("--fixed-size", type=int, default=128)
    ap.add_argument("--min-len", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--target", type=float, default=128.0)
    ap.add_argument("--warmup", type=int, default=16)
    ap.add_argument("--calib-docs", type=int, default=1500)
    ap.add_argument("--calib-seed", type=int, default=11)
    ap.add_argument("--random-seed", type=int, default=7)
    ap.add_argument("--n-bins", type=int, default=4)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    guard.check_root()
    data_dir = guard.inside_root(a.data)
    print(f"segmentation caches for {data_dir} splits={SPLITS} (test never read)")
    if not guard.require_execution(a.execute, "cpu-model"):
        return
    guard.require_disk(400_000_000, "build_segments")
    guard.register_self("build_segments (CPU)")

    import numpy as np
    import torch
    torch.set_num_threads(min(a.threads, 8))
    from csr.artifacts import ENCODER_PREPROC, ENCODER_REPO, ENCODER_REV, FrozenEncoder, load_lm_tokenizer
    from csr.data import TokenSplit, TOKENIZER_REV
    from csr.segcache import (add_prefix_embeddings, build_table, cache_id, calibrate_tau, embed_doc_sentences,
                              make_detector)
    from csr.segment import (SEGMENTER_VERSION, SemanticParams, chunk_lengths, fit_length_distribution,
                             fixed_starts, random_starts, semantic_starts)

    data_man = json.loads((data_dir / "manifest.json").read_text())
    spec = {"data_sha": guard.sha256_json(data_man["splits"]), "tokenizer_rev": TOKENIZER_REV,
            "encoder": f"{ENCODER_REPO}@{ENCODER_REV}", "encoder_preproc": ENCODER_PREPROC, "fixed_size": a.fixed_size,
            "min_len": a.min_len, "max_len": a.max_len, "target": a.target, "warmup": a.warmup,
            "calib_docs": a.calib_docs, "calib_seed": a.calib_seed, "random_seed": a.random_seed, "n_bins": a.n_bins,
            "sentence_rule": "token text ends in .!? (+closing quote/bracket) or contains newline; min 4 tokens"}
    cid = cache_id(spec)
    out = guard.ROOT / "artifacts" / "segments" / cid
    if out.exists():
        print(f"cache exists: {out}")
        return
    tmp = out.with_name("." + out.name + ".tmp")
    tmp.mkdir(parents=True, exist_ok=False)
    timings = {}
    tok = load_lm_tokenizer()
    det = make_detector(tok)
    enc = FrozenEncoder("cpu", batch_size=128)
    enc_ck = enc.param_checksum()
    splits = {s: TokenSplit.load(data_dir, s) for s in SPLITS}

    # --- tau calibration on a seeded sample of training documents
    t0 = time.time()
    tr = splits["train"]
    sample = np.random.default_rng(a.calib_seed).choice(tr.n_docs, size=min(a.calib_docs, tr.n_docs), replace=False)
    sent = embed_doc_sentences(tok, det, enc, [tr.doc(i) for i in sample])
    tau, table = calibrate_tau([tr.lengths()[i] for i in sample], sent, a.target, a.min_len, a.max_len)
    timings["tau_calibration_s"] = time.time() - t0
    print(f"tau={tau}  (calibration {timings['tau_calibration_s']:.0f}s)")
    p = SemanticParams(tau, a.min_len, a.max_len)

    # --- semantic boundaries for every doc of every split
    sem_starts, stats = {}, {}
    for s, sp in splits.items():
        t0 = time.time()
        starts, n_sem, n_forced, trig_len, n_sent = [], 0, 0, [], 0
        keep_emb = s in ("calib", "val")
        ends_all, embs_all = [], []
        for b0 in range(0, sp.n_docs, 400):
            docs = [sp.doc(i) for i in range(b0, min(sp.n_docs, b0 + 400))]
            for ids, (e, x) in zip(docs, embed_doc_sentences(tok, det, enc, docs)):
                st, ev = semantic_starts(len(ids), e, x, p, return_events=True)
                starts.append(st)
                n_sent += len(e)
                for evv in ev:
                    if evv[0] == "semantic":
                        n_sem += 1
                        j = int(np.searchsorted(e, evv[1] - 1))
                        trig_len.append(int(e[j] - (e[j - 1] if j > 0 else -1)))
                    else:
                        n_forced += 1
                if keep_emb:
                    ends_all.append(e)
                    embs_all.append(x.astype(np.float16))
        sem_starts[s] = starts
        timings[f"semantic_{s}_s"] = time.time() - t0
        stats[s] = {"n_sentences": n_sent, "semantic_boundaries": n_sem, "forced_boundaries": n_forced,
                    "detection_delay_tokens_mean": float(np.mean(trig_len)) if trig_len else None,
                    "detection_delay_tokens_p90": float(np.percentile(trig_len, 90)) if trig_len else None}
        if keep_emb:
            off = np.concatenate([[0], np.cumsum([len(e) for e in ends_all])])
            np.savez(tmp / f"{s}_sentences.npz", ends=np.concatenate(ends_all), off=off,
                     embs=np.concatenate(embs_all))
        print(s, stats[s], f"{timings[f'semantic_{s}_s']:.0f}s", flush=True)

    dist = fit_length_distribution(splits["train"].lengths(), sem_starts["train"], a.n_bins, a.min_len, a.max_len)
    (tmp / "random_length_distribution.json").write_text(dist.to_json())
    summary = {}
    for s, sp in splits.items():
        L = sp.lengths()
        tabs = {"fixed": build_table("fixed", sp, [fixed_starts(int(l), a.fixed_size) for l in L]),
                "semantic": build_table("semantic", sp, sem_starts[s]),
                "random": build_table("random", sp, [random_starts(int(l), k, dist, a.random_seed)
                                                     for l, k in zip(L, sp.doc_ids)])}
        for name, t in tabs.items():
            if name in ("fixed", "semantic"):
                t0 = time.time()
                n = add_prefix_embeddings(t, sp, tok, enc, a.warmup)
                timings[f"prefix_{name}_{s}_s"] = time.time() - t0
            t.save(tmp / f"{s}_{name}.npz")
            ls = np.concatenate([chunk_lengths(int(l), t.starts(i)) for i, l in enumerate(L)])
            nonfinal = np.concatenate([chunk_lengths(int(l), t.starts(i))[:-1] for i, l in enumerate(L)])
            elig = sum(max(0, int(x) - a.warmup) for x in ls)
            summary[f"{s}/{name}"] = {"n_chunks": int(len(ls)), "mean_len": float(ls.mean()),
                                      "mean_nonfinal_len": float(nonfinal.mean()) if len(nonfinal) else None,
                                      "p10_p50_p90": np.percentile(ls, [10, 50, 90]).tolist(),
                                      "pool_eligible_token_fraction": elig / float(L.sum())}
        print(s, {k: v for k, v in summary.items() if k.startswith(s)}, flush=True)
    assert enc.param_checksum() == enc_ck, "frozen encoder changed"
    man = {"cache_id": cid, "spec": spec, "segmenter_version": SEGMENTER_VERSION, "tau": tau,
           "tau_calibration_table": table, "boundary_stats": stats, "chunk_summary": summary, "timings_s": timings,
           "cpu_threads": min(a.threads, 8), "device": "cpu", "created": guard.now(),
           "random_distribution_sha256": guard.sha256_json(json.loads(dist.to_json()))}
    (tmp / "manifest.json").write_text(json.dumps(man, indent=2, sort_keys=True))
    for f in tmp.iterdir():
        f.chmod(0o400)
    tmp.rename(out)
    guard.ledger("SEGMENT_BUILD", f"artifacts/segments/{cid} tau={tau} cpu_seconds_total={sum(timings.values()):.0f}")
    print("done", out)


if __name__ == "__main__":
    main()
