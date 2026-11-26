#!/usr/bin/env python
"""csr-sequence-routing-search-v1: inference latency benchmark of final checkpoints (prereg §12).

Uses the same causal routing code as training/evaluation (SeqRouteModel.forward for prefill, sr_step for
decoding); boundary-detector time (including the CPU boundary scan in prefill) is included. Data: CALIB-SEARCH
documents only. Prefill: bf16 autocast as in evaluation. Decoding (sr_step): fp32 for every condition, so the router
path matches the fp32 router of training/evaluation (deviation D2). Dry-run by default.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="final run dirs (one per condition)")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--decode-len", type=int, default=256)
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    if not guard.require_execution(a.execute, "gpu" if a.device == "cuda" else "cpu-model"):
        return
    import numpy as np
    import torch
    from csr.data import to_torch
    from csr.seqroute import build_sr_model
    from csr.srs_run import ART, eval_source
    torch.set_num_threads(4)
    monitor = None
    if a.device == "cuda":
        guard.occupancy_gate("srsv1 bench")
        device = guard.setup_cuda_device()
        monitor = guard.OccupancyMonitor(interval=15.0)
        monitor.start()
    else:
        device = torch.device("cpu")
    guard.register_self("srsv1 bench")
    t_iso, t0 = guard.now(), time.time()

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    def timeit(fn, reps, warm=3):
        for _ in range(warm):
            fn()
        sync()
        ts = []
        for _ in range(reps):
            s = time.perf_counter()
            fn()
            sync()
            ts.append(time.perf_counter() - s)
        return statistics.median(ts), ts

    res = {}
    for rdp in a.runs:
        rd = guard.inside_root(rdp)
        cfg = json.loads((rd / "config.json").read_text())
        model = build_sr_model(cfg, device)
        model.load_state_dict(torch.load(rd / "model.pt", map_location="cpu", weights_only=False)["model"])
        model.eval()
        _, src = eval_source(cfg, "calib_search", cfg["data"]["dir"])
        sp = src.split
        docs = [d for r in src.rows for d in r if sp.lengths()[d] >= 1000][:16]
        assert len(docs) == 16
        amp = lambda: torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")  # noqa: E731
        nul = lambda: torch.autocast(device.type, enabled=False)  # noqa: E731  (fp32 decoding)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        r = {"candidate": cfg["candidate"], "family": cfg["family"], "run": str(rd.relative_to(guard.ROOT))}
        # prefill: batch 1 (one 1000-token document) and batch 16 (packed rows)
        b1 = to_torch(src.batch([[docs[0]]]), device)
        rows16 = src.rows[:16]
        b16 = to_torch(src.batch(rows16), device)
        with torch.no_grad():
            def pf1():
                with amp():
                    model(b1)

            def pf16():
                with amp():
                    model(b16)
            m1, _ = timeit(pf1, 20)
            m16, _ = timeit(pf16, 10)
        n16 = int((b16["doc_ids"] >= 0).sum())
        r["prefill_b1_ms"] = 1000 * m1
        r["prefill_b1_tokens"] = int(sp.lengths()[docs[0]])
        r["prefill_b16_ms"] = 1000 * m16
        r["prefill_b16_tokens_per_s"] = n16 / m16
        # decoding: batch 1 (3 docs) and batch 16 lockstep
        L = a.decode_len
        per_tok, calls = [], []
        with torch.no_grad():
            for d in docs[:3]:
                ids = sp.doc(d)[:L]
                st = model.new_sr_state(1)
                wst = model.new_sr_state(1)
                with nul():
                    for p in range(16):  # warm-up on a throwaway state
                        model.sr_step(torch.tensor([int(ids[p])], device=device), p, wst)
                sync()
                s = time.perf_counter()
                with nul():
                    for p in range(L):
                        model.sr_step(torch.tensor([int(ids[p])], device=device), p, st)
                sync()
                per_tok.append((time.perf_counter() - s) / L)
                calls.append(st["router_calls"] / L)
            ids16 = torch.tensor(np.stack([sp.doc(d)[:L] for d in docs]), device=device)
            st = model.new_sr_state(16)
            sync()
            s = time.perf_counter()
            with nul():
                for p in range(L):
                    model.sr_step(ids16[:, p], p, st)
            sync()
            el = time.perf_counter() - s
        r["decode_b1_ms_per_token"] = 1000 * statistics.median(per_tok)
        r["decode_b1_router_calls_per_token"] = statistics.mean(calls)
        r["decode_b16_ms_per_step"] = 1000 * el / L
        r["decode_b16_tokens_per_s"] = 16 * L / el
        r["decode_b16_router_calls_per_step"] = st["router_calls"] / L
        r["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 1024**3 if device.type == "cuda" else 0.0
        res[cfg["candidate"]] = r
        print(json.dumps(r), flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    status = "COMPLETE"
    if monitor is not None:
        monitor.stop()
        if monitor.contaminated:
            status = "CONTAMINATED"
    out = ART / f"latency_{guard.utc_stamp()}.json"
    guard.atomic_write_json(out, {"status": status, "results": res, "device": str(device),
                                  "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
                                  "decode_len": L, "seconds": time.time() - t0})
    if a.device == "cuda":
        guard.gpu_ledger(t_iso, guard.now(), out.stem, "srsv1 bench", status)
    print(f"BENCH_JSON {out.relative_to(guard.ROOT)} {status}")
    sys.exit(0 if status == "COMPLETE" else 1)


if __name__ == "__main__":
    main()
