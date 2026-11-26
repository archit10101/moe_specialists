#!/usr/bin/env python
"""scrgp-v1 Stage-0 engineering timing benchmark (GPU, guarded). Prints only timings and memory (no scientific values:
CE / gradient values are computed but neither printed nor stored). Used to fix the Stage-1 token budget in the prereg.
  scripts/gpu_guarded.sh "scrgp bench" scrgp_bench.py --device cuda --execute"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rows", type=int, default=32)
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    if not guard.require_execution(a.execute, "gpu"):
        return
    import torch
    from fle_evaluate import load_milestone
    from csr.fsp_lib import Split, pack_rows
    from csr.lfr_eval import paper_half
    from csr.scrgp_lib import Probe, make_batch, to_dev, u_hash
    from csr.scrgp_paths import C6_RUNS
    guard.occupancy_gate("scrgp bench")
    dev = guard.setup_cuda_device()
    guard.register_self("scrgp bench")
    t_iso = guard.now()
    m, _ = load_milestone(C6_RUNS[0] + "/ckpt_p100", dev)
    cs = Split("calib_search")
    docs = sorted(paper_half(cs, 0), key=lambda i: u_hash("scrgp-bench", f"{cs.papers[i]}|{i}"))
    rows = pack_rows(cs.lengths, docs)[:a.rows]
    bt = to_dev(make_batch(rows, cs), dev)
    pr = Probe(m, dev)
    out = {}

    def tm(name, fn, n=3):
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n):
            r = fn()
        torch.cuda.synchronize()
        out[name] = (time.time() - t0) / n
        return r

    shape = bt["input_ids"].shape + (m.cfg.d_model,)
    ce, probs, cache, ctx = tm("natural_cache", lambda: pr.natural(bt, keep_cache=True), 2)
    out["mem_after_cache_gib"] = torch.cuda.memory_allocated() / 2**30
    W = torch.randn(4, ce.numel(), device=dev, dtype=torch.float32) * 1e-5
    for li in (0, 5, 10):
        w1, w2 = m.blocks[li + 1].ffn.experts.w1[3], m.blocks[li + 1].ffn.experts.w2[3]

        def fwd():
            with torch.no_grad(), pr.autocast():
                h = pr.forced_hidden(li, 3, cache[li], ctx, shape, w1, w2)
                return pr.ce_of(h, bt["targets"])
        tm(f"forced_fwd_l{li}", fwd)
        torch.cuda.reset_peak_memory_stats()
        tm(f"grads4_l{li}", lambda: pr.expert_grads(li, 3, cache[li], ctx, shape, bt["targets"], W))
        out[f"peak_grads_l{li}_gib"] = torch.cuda.max_memory_allocated() / 2**30
    out["tokens"] = int((bt["targets"] >= 0).sum())
    print(json.dumps(out, indent=1))
    guard.gpu_ledger(t_iso, guard.now(), "scrgp_bench", "scrgp stage-0 timing", "COMPLETE")


if __name__ == "__main__":
    main()
