#!/usr/bin/env python
"""csr-representative-v1 segmentation caches (docs/REP_V1_PREREG.md §3): semantic (tau calibrated on 3,000 seeded
TRAINING documents), fixed 128, frozen prefix embeddings u for both, and sentence-end tables for F-v3.
Splits train / calib / val only (test never read). The frozen encoder runs on GPU 0 via scripts/gpu_guarded.sh
(--device cuda, MOE_RESEARCH_APPROVE=gpu) or on CPU (MOE_RESEARCH_APPROVE=cpu-model). Dry-run by default."""
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
    ap.add_argument("--data", default="artifacts/data/fwe-rep-v1")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--calib-docs", type=int, default=3000)
    ap.add_argument("--calib-seed", type=int, default=11)
    ap.add_argument("--encoder-batch", type=int, default=512)
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    dd = guard.inside_root(a.data)
    print(f"rep-v1 segmentation for {dd} splits={SPLITS} device={a.device} (test never read)")
    if not guard.require_execution(a.execute, "gpu" if a.device == "cuda" else "cpu-model"):
        return
    guard.require_disk(6_000_000_000, "rep_build_segments")
    mon = None
    if a.device == "cuda":
        guard.occupancy_gate("rep_build_segments")
        guard.setup_cuda_device()
        mon = guard.OccupancyMonitor(15.0)
        mon.start()
    guard.register_self(f"rep_build_segments ({a.device})")
    t_start_iso = guard.now()
    import numpy as np
    import torch
    torch.set_num_threads(8)
    from csr.artifacts import ENCODER_PREPROC, ENCODER_REPO, ENCODER_REV, FrozenEncoder, load_lm_tokenizer
    from csr.data import TOKENIZER_REV, TokenSplit
    from csr.rep import REP_EXPERIMENT, embed_docs_batched
    from csr.segcache import add_prefix_embeddings, build_table, cache_id, calibrate_tau, make_detector
    from csr.segment import SEGMENTER_VERSION, SemanticParams, chunk_lengths, fixed_starts, semantic_starts

    dman = json.loads((dd / "manifest.json").read_text())
    spec = {"experiment": REP_EXPERIMENT, "data_sha": guard.sha256_json(dman["splits"]), "tokenizer_rev": TOKENIZER_REV,
            "encoder": f"{ENCODER_REPO}@{ENCODER_REV}", "encoder_preproc": ENCODER_PREPROC, "encoder_device": a.device,
            "fixed_size": 128, "min_len": 64, "max_len": 256, "target": 128.0, "warmup": 16,
            "calib_docs": a.calib_docs, "calib_seed": a.calib_seed, "tau_grid": "-0.20..0.95 step 0.01",
            "sentence_rule": "token text ends in .!? (+closing quote/bracket) or contains newline; min 4 tokens"}
    cid = cache_id(spec)
    out = guard.ROOT / "artifacts" / "segments" / cid
    if out.exists():
        raise guard.GuardError(f"cache exists: {out}")
    tmp = out.with_name("." + out.name + ".tmp")
    tmp.mkdir(parents=True, exist_ok=False)
    status, reason = "INCOMPLETE", ""
    timings = {}

    def check():
        if mon is not None and mon.contaminated:
            raise guard.GuardError(f"CONTAMINATED: {mon.detail}")

    try:
        tok = load_lm_tokenizer()
        det = make_detector(tok)
        enc = FrozenEncoder("cuda:0" if a.device == "cuda" else "cpu", batch_size=a.encoder_batch)
        enc_ck = enc.param_checksum()
        splits = {s: TokenSplit.load(dd, s) for s in SPLITS}
        t0 = time.time()
        tr = splits["train"]
        sample = np.random.default_rng(a.calib_seed).choice(tr.n_docs, size=min(a.calib_docs, tr.n_docs), replace=False)
        sent = embed_docs_batched(tok, det, enc, [tr.doc(int(i)) for i in sample])
        tau, table = calibrate_tau([int(tr.lengths()[i]) for i in sample], sent, 128.0, 64, 256)
        timings["tau_calibration_s"] = time.time() - t0
        print(f"tau={tau} ({timings['tau_calibration_s']:.0f}s)", flush=True)
        p = SemanticParams(tau, 64, 256)
        sem, stats = {}, {}
        for s, sp in splits.items():
            t0 = time.time()
            starts, ends_all, n_sem, n_forced, n_sent = [], [], 0, 0, 0
            for b0 in range(0, sp.n_docs, 2000):
                docs = [sp.doc(i) for i in range(b0, min(sp.n_docs, b0 + 2000))]
                for ids, (e, x) in zip(docs, embed_docs_batched(tok, det, enc, docs)):
                    st, ev = semantic_starts(len(ids), e, x, p, return_events=True)
                    starts.append(st)
                    ends_all.append(e.astype(np.int32))
                    n_sent += len(e)
                    n_sem += sum(1 for v in ev if v[0] == "semantic")
                    n_forced += sum(1 for v in ev if v[0] == "forced")
                check()
                if b0 % 20000 == 0:
                    print(f"{s} {b0}/{sp.n_docs} {time.time() - t0:.0f}s", flush=True)
            sem[s] = starts
            off = np.concatenate([[0], np.cumsum([len(e) for e in ends_all])]).astype(np.int64)
            np.savez(tmp / f"{s}_ends.npz", ends_flat=np.concatenate(ends_all), ends_off=off)
            timings[f"semantic_{s}_s"] = time.time() - t0
            stats[s] = {"n_sentences": n_sent, "semantic_boundaries": n_sem, "forced_boundaries": n_forced}
            print(s, stats[s], f"{timings[f'semantic_{s}_s']:.0f}s", flush=True)
        summary = {}
        for s, sp in splits.items():
            L = sp.lengths()
            for name, st_list in (("fixed", [fixed_starts(int(l), 128) for l in L]), ("semantic", sem[s])):
                t = build_table(name, sp, st_list)
                t0 = time.time()
                add_prefix_embeddings(t, sp, tok, enc, 16)
                timings[f"prefix_{name}_{s}_s"] = time.time() - t0
                t.save(tmp / f"{s}_{name}.npz")
                ls = np.concatenate([chunk_lengths(int(l), t.starts(i)) for i, l in enumerate(L)])
                nonf = np.concatenate([chunk_lengths(int(l), t.starts(i))[:-1] for i, l in enumerate(L)])
                summary[f"{s}/{name}"] = {"n_chunks": int(len(ls)), "mean_len": float(ls.mean()),
                                          "mean_nonfinal_len": float(nonf.mean()) if len(nonf) else None,
                                          "p10_p50_p90": np.percentile(ls, [10, 50, 90]).tolist(),
                                          "pool_eligible_token_fraction": float(np.maximum(ls - 16, 0).sum() / L.sum())}
                check()
            print(s, {k: v for k, v in summary.items() if k.startswith(s)}, flush=True)
        assert enc.param_checksum() == enc_ck, "frozen encoder changed"
        man = {"cache_id": cid, "spec": spec, "segmenter_version": SEGMENTER_VERSION, "tau": tau,
               "tau_calibration_table": table, "boundary_stats": stats, "chunk_summary": summary, "timings_s": timings,
               "device": a.device, "encoder_param_checksum": enc_ck, "created": guard.now(),
               "splits": list(SPLITS), "test_read": False}
        (tmp / "manifest.json").write_text(json.dumps(man, indent=2, sort_keys=True))
        for f in tmp.iterdir():
            f.chmod(0o400)
        tmp.rename(out)
        guard.ledger("SEGMENT_BUILD", f"artifacts/segments/{cid} rep-v1 tau={tau} device={a.device} "
                                      f"seconds={sum(timings.values()):.0f}")
        status = "COMPLETE"
        print("done", out)
    except guard.GuardError as e:
        status, reason = ("CONTAMINATED" if "CONTAMINATED" in str(e) else "ABORTED"), str(e)
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        status, reason = "FAILED", f"{type(e).__name__}: {e}"
    finally:
        if mon is not None:
            mon.stop()
            if mon.contaminated and status == "COMPLETE":
                status, reason = "CONTAMINATED", mon.detail
            guard.gpu_ledger(t_start_iso, guard.now(), f"rep_segments_{cid}", "rep_build_segments", status)
        print("STATUS", status, reason)
    sys.exit(0 if status == "COMPLETE" else 1)


if __name__ == "__main__":
    main()
