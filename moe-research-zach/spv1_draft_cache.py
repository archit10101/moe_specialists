#!/usr/bin/env python
"""csr-speculative-pool-routing-v1: cache the frozen draft's causal greedy proposals for every position of every
TRAINING document (gamma = 7 tokens; the first B-1 serve block size B). proposals[off[d] + t, j-1] = j-th greedy
draft token generated from the prefix x_0..x_t of training document d (draft_tree_proposals, bf16 autocast).

Written once to artifacts/speculative-pool-routing-v1/draft_cache/ with a manifest (draft checkpoint hash, code
hashes, precision, packing, sha256 of the array) and an agreement check against recomputation under a different
packing. Only the train split is read. Dry-run by default.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--rows-per-batch", type=int, default=32)
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    if not guard.require_execution(a.execute, "gpu" if a.device == "cuda" else "cpu-model"):
        return
    import numpy as np
    import torch
    from csr.data import to_torch, train_order
    from csr.spv1_model import draft_tree_proposals, load_frozen_draft
    from csr.spv1_run import CACHE_DIR, DATA_DIR, GAMMA_MAX, frozen_draft, load_raw_split, require_disk
    from csr.train_lib import BatchSource
    torch.set_num_threads(4)
    monitor = None
    if a.device == "cuda":
        guard.occupancy_gate("spv1 draft cache")
        device = guard.setup_cuda_device()
        monitor = guard.OccupancyMonitor(interval=15.0)
        monitor.start()
    else:
        device = torch.device("cpu")
    guard.register_self("spv1 draft cache")
    t_iso, t0 = guard.now(), time.time()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    final = CACHE_DIR / "train_proposals_g7.npy"
    if final.exists():
        raise guard.GuardError(f"{final} exists; never overwritten")
    status, reason = "FAILED", ""
    limits = guard.RunLimits(max_seconds=5000, monitor=monitor)
    man = {}
    try:
        dpath, drec = frozen_draft()
        draft = load_frozen_draft(dpath, device)
        tr = load_raw_split(guard.ROOT / DATA_DIR, "train")
        require_disk(tr.offsets[-1] * GAMMA_MAX * 2 + (1 << 30), "spv1 proposal cache")
        src = BatchSource(tr, 1024, None, None)  # natural document order
        tmp = CACHE_DIR / f".train_proposals_g7.{guard.utc_stamp()}.npy.tmp"
        arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint16, shape=(int(tr.offsets[-1]), GAMMA_MAX))
        off = tr.offsets
        filled = 0
        R = a.rows_per_batch
        for s in range(0, len(src.rows), R):
            rows = src.rows[s:s + R]
            tb = to_torch(src.batch(rows), device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                P = draft_tree_proposals(draft, tb["input_ids"], tb["doc_ids"], tb["positions"], GAMMA_MAX)
            P = P.cpu().numpy()
            for b, row in enumerate(rows):
                t = 0
                for di in row:
                    L = int(off[di + 1] - off[di])
                    arr[off[di]:off[di + 1]] = P[b, t:t + L]
                    t += L
                    filled += L
            if (s // R) % 200 == 0:
                print(f"{guard.now()} rows {s}/{len(src.rows)} tokens {filled}", flush=True)
                limits.check()
        arr.flush()
        if filled != int(off[-1]):
            raise guard.GuardError(f"filled {filled} != {int(off[-1])}")
        # agreement check under a different packing (shuffled document order, 16 rows per batch)
        alt = BatchSource(tr, 1024, None, train_order(tr.n_docs, 4242))
        agree, tot = 0, 0
        for s in range(0, 256, 16):
            rows = alt.rows[s:s + 16]
            tb = to_torch(alt.batch(rows), device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                P = draft_tree_proposals(draft, tb["input_ids"], tb["doc_ids"], tb["positions"], GAMMA_MAX).cpu().numpy()
            for b, row in enumerate(rows):
                t = 0
                for di in row:
                    L = int(off[di + 1] - off[di])
                    agree += int((arr[off[di]:off[di + 1]] == P[b, t:t + L]).sum())
                    tot += L * GAMMA_MAX
                    t += L
        del arr
        tmp.rename(final)
        sha = guard.sha256_file(final)
        man = {"file": str(final.relative_to(guard.ROOT)), "sha256": sha, "dtype": "uint16",
               "shape": [int(off[-1]), GAMMA_MAX], "index": "train_offsets[doc] + in-document position",
               "split": "train", "draft_checkpoint": drec["checkpoint"],
               "draft_checkpoint_sha256": drec["checkpoint_sha256"], "precision": "bf16 autocast (CUDA)",
               "packing": "natural train document order, 1024-token rows, 32 rows per batch",
               "code_sha256": {p: guard.sha256_file(guard.ROOT / p) for p in
                               ("src/csr/spv1_model.py", "spv1_draft_cache.py", "src/csr/model.py")},
               "agreement_vs_repacked_recompute": {"tokens_compared": tot, "agree": agree,
                                                   "rate": agree / max(tot, 1)},
               "created": guard.now(), "seconds": time.time() - t0}
        guard.atomic_write_json(CACHE_DIR / "train_proposals_manifest.json", man, mode=0o400)
        final.chmod(0o400)
        guard.ledger("SP_V1_PROPOSAL_CACHE", f"{man['file']} sha256={sha} agree={man['agreement_vs_repacked_recompute']}")
        limits.check()
        status = "COMPLETE"
    except guard.GuardError as e:
        status = "CONTAMINATED" if "CONTAMINATED" in str(e) else "ABORTED"
        reason = str(e)
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        status, reason = "FAILED", f"{type(e).__name__}: {e}"
    finally:
        limits.close()
        if monitor is not None:
            monitor.stop()
            if monitor.contaminated and status == "COMPLETE":
                status, reason = "CONTAMINATED", monitor.detail
        if a.device == "cuda":
            guard.gpu_ledger(t_iso, guard.now(), f"spv1_draft_cache_{guard.utc_stamp()}", "spv1 draft cache", status)
        print(f"CACHE_STATUS {status} {reason}")
    sys.exit(0 if status == "COMPLETE" else 1)


if __name__ == "__main__":
    main()
