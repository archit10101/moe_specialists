#!/usr/bin/env python
"""csr-domain-forecast-curriculum-v1: cache the frozen draft's causal greedy 4-token forecasts (prereg section 4).

cache/<split>_<source>.npy : uint16 (n_tokens(split), 4); row off[d] + t = 4 greedy tokens generated from the prefix
x_0..x_t of document d with adapter domain[d] (correct) or (domain[d] + 1) % 5 (deranged).
Canonical layout: each document alone in a 1024-long row at position 0; fixed 64-row batches in document-index order
(last batch padded with dummy rows); bf16 autocast; deterministic algorithms. Order: main/correct (timed separately:
it is the SOFT method's cache cost), main/deranged, calib/correct, calib/deranged. Then an invariance re-check on a
sample under a different batch size / order (must be bitwise equal) and calibration forecast accuracy (post hoc vs
the true continuation, which the draft never saw). Dry-run by default.
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402

BATCH = 64
JOBS = [("main", "correct"), ("main", "deranged"), ("calib", "correct"), ("calib", "deranged")]


def canonical_forecasts(draft, split, doms, doc_idx, adapter_of, device, batch=BATCH, T=1024):
    """Yields (doc index, (n, 4) uint16 forecasts) in doc_idx order using the canonical one-doc-per-row layout."""
    import numpy as np
    import torch
    from csr.data import PAD_ID
    from csr.dfc_model import draft_forecasts
    for s in range(0, len(doc_idx), batch):
        chunk = list(doc_idx[s:s + batch])
        ids = np.full((batch, T), PAD_ID, dtype=np.int64)
        did = np.full((batch, T), -1, dtype=np.int64)
        pos = np.zeros((batch, T), dtype=np.int64)
        adom = np.zeros((batch, T), dtype=np.int64)
        lens = []
        for b, di in enumerate(chunk):
            x = split.doc(di)
            n = len(x)
            ids[b, :n] = x
            did[b, :n] = 0
            pos[b, :n] = np.arange(n)
            adom[b, :] = adapter_of(int(doms[di]))
            lens.append(n)
        tb = [torch.as_tensor(v, device=device) for v in (ids, did, pos, adom)]
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = draft_forecasts(draft, *tb)
        out = out.cpu().numpy().astype(np.uint16)
        for b, di in enumerate(chunk):
            yield di, out[b, :lens[b]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--draft-run", required=True)
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    if not guard.require_execution(a.execute, "gpu" if a.device == "cuda" else "cpu-model"):
        return
    import numpy as np
    import torch
    from csr.dfc_lib import CACHE, load_split
    from csr.dfc_model import DERANGE, load_frozen_draft
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    monitor = None
    if a.device == "cuda":
        guard.occupancy_gate("dfc forecast cache")
        device = guard.setup_cuda_device()
        monitor = guard.OccupancyMonitor(interval=15.0)
        monitor.start()
    else:
        device = torch.device("cpu")
    guard.register_self("dfc forecast cache")
    t_iso, t0 = guard.now(), time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    status, reason = "FAILED", ""
    limits = guard.RunLimits(max_seconds=5100, monitor=monitor)
    man = {"created": guard.now(), "draft_run": a.draft_run, "layout": "one doc per 1024 row at position 0; "
           f"{BATCH}-row batches in doc-index order; bf16 autocast; deterministic algorithms; tf32 off",
           "derangement": {str(k): v for k, v in DERANGE.items()}, "jobs": {}}
    try:
        drun = guard.inside_root(a.draft_run)
        dck = drun / "ckpt_final.pt"
        man["draft_ckpt_sha256"] = guard.sha256_file(dck)
        draft = load_frozen_draft(dck, device)
        man["draft_param_trainable"] = sum(p.requires_grad for p in draft.parameters())
        for split_name, source in JOBS:
            final = CACHE / f"{split_name}_{source}.npy"
            if final.exists():
                man["jobs"][f"{split_name}_{source}"] = json.loads((CACHE / f"{split_name}_{source}.json").read_text())
                print("exists", final)
                continue
            sp, doms, _ = load_split(split_name)
            n_tok = int(sp.offsets[-1])
            guard.require_disk(n_tok * 8 + (1 << 28), f"dfc cache {split_name}_{source}")
            adapter_of = (lambda d: d) if source == "correct" else (lambda d: DERANGE[d])
            tmp = CACHE / f".{split_name}_{source}.npy.tmp"
            arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint16, shape=(n_tok, 4))
            if device.type == "cuda":
                torch.cuda.synchronize()
            tj = time.time()
            for di, fc in canonical_forecasts(draft, sp, doms, list(range(sp.n_docs)), adapter_of, device):
                arr[sp.offsets[di]:sp.offsets[di + 1]] = fc
                if di % 20000 == 0:
                    limits.check()
                    print(f"{split_name}/{source} doc {di}/{sp.n_docs} {time.time() - tj:.0f}s", flush=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            secs = time.time() - tj
            arr.flush()
            del arr
            os.replace(tmp, final)
            os.chmod(final, 0o400)
            rec = {"file": str(final.relative_to(guard.ROOT)), "sha256": guard.sha256_file(final), "n_tokens": n_tok,
                   "n_docs": sp.n_docs, "gpu_seconds": secs, "source": source}
            guard.atomic_write_json(CACHE / f"{split_name}_{source}.json", rec, mode=0o400)
            man["jobs"][f"{split_name}_{source}"] = rec
            guard.ledger("DFC_CACHE", f"{final.relative_to(guard.ROOT)} sha={rec['sha256'][:16]} {secs:.0f}s")
        # invariance re-check: sample of docs under a different batch size and reversed order
        inv = {}
        for split_name, source in JOBS:
            sp, doms, _ = load_split(split_name)
            arr = np.load(CACHE / f"{split_name}_{source}.npy", mmap_mode="r")
            rng = np.random.default_rng(7)
            sample = sorted(rng.choice(sp.n_docs, size=min(384, sp.n_docs), replace=False).tolist(), reverse=True)
            adapter_of = (lambda d: d) if source == "correct" else (lambda d: DERANGE[d])
            mism = 0
            for di, fc in canonical_forecasts(draft, sp, doms, sample, adapter_of, device, batch=48):
                mism += int((arr[sp.offsets[di]:sp.offsets[di + 1]] != fc).any())
            inv[f"{split_name}_{source}"] = {"n_docs_checked": len(sample), "docs_mismatched": mism,
                                             "batch_size": 48, "order": "random sample, reversed"}
        man["invariance_check"] = inv
        # calibration forecast accuracy (post hoc; the draft never saw the true continuation)
        sp, doms, _ = load_split("calib")
        acc = {}
        for source in ("correct", "deranged"):
            arr = np.load(CACHE / f"calib_{source}.npy", mmap_mode="r")
            hit = np.zeros((5, 4))
            cnt = np.zeros((5, 4))
            for di in range(sp.n_docs):
                x = sp.doc(di)
                fc = np.asarray(arr[sp.offsets[di]:sp.offsets[di + 1]], dtype=np.int64)
                n = len(x)
                for j in range(4):
                    m = n - 1 - j
                    if m <= 0:
                        continue
                    hit[doms[di], j] += (fc[:m, j] == x[1 + j:1 + j + m]).sum()
                    cnt[doms[di], j] += m
            acc[source] = {"per_domain_per_j": (hit / np.maximum(cnt, 1)).tolist(),
                           "overall_per_j": (hit.sum(0) / np.maximum(cnt.sum(0), 1)).tolist()}
        man["calib_forecast_accuracy"] = acc
        man["seconds_total"] = time.time() - t0
        status = "COMPLETE" if all(v["docs_mismatched"] == 0 for v in inv.values()) else "INVARIANCE_FAIL"
        guard.atomic_write_json(CACHE / f"manifest_{guard.utc_stamp()}.json", man)
        print(json.dumps({k: v for k, v in man.items() if k != "jobs"}, indent=1))
    except guard.GuardError as e:
        status, reason = ("CONTAMINATED" if "CONTAMINATED" in str(e) else "INTERRUPTED"), str(e)
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        status, reason = "FAILED", f"{type(e).__name__}: {e}"
    finally:
        limits.close()
        if monitor is not None:
            monitor.stop()
            if monitor.contaminated and status == "COMPLETE":
                status = "CONTAMINATED"
        if a.device == "cuda":
            guard.gpu_ledger(t_iso, guard.now(), f"dfc_cache_{guard.utc_stamp()}", "dfc forecast cache", status)
        print("STATUS", status, reason)
    sys.exit(0 if status == "COMPLETE" else (75 if status == "INTERRUPTED" else 1))


if __name__ == "__main__":
    main()
