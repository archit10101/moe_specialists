#!/usr/bin/env python
"""csr-value-routed-compute-v1 POST-HOC timing breakdown (after the diagnostic gate failed; explanatory only, never
changes the gate). Prefill latency at 1K/2K/4K/8K (identical CALIB-SEARCH tokens, 32,768 tokens per batch, bf16,
no grad, 5 warm-up + 20 timed, CUDA-synchronised) for REF and the uniform options; component timings: block-mask
construction (full / local / mixed), expert dispatch (original vs four-slot NULL dispatch at k = 4/2/1).
  scripts/gpu_guarded.sh "vrc posthoc timing" vrc_posthoc_timing.py --run <run dir> --execute"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    if not guard.require_execution(a.execute, "gpu"):
        return
    import torch
    from csr.cla_attn import make_layout, full_or_local_block_mask
    from csr.data import to_torch
    from csr.vm_run import EXP, build_model, check_cfg, eval_source
    from csr.vrc_model import experts_null, mixed_block_mask
    run = guard.inside_root(a.run)
    cfg = json.loads((run / "config.json").read_text())
    key, d = check_cfg(cfg)
    guard.occupancy_gate("vrc posthoc timing")
    dev = guard.setup_cuda_device()
    mon = guard.OccupancyMonitor(interval=15.0)
    mon.start()
    t0iso, t0 = guard.now(), time.time()
    out = {"note": "POST-HOC explanatory timing; the preregistered gate decision (FAIL) is unchanged",
           "run": str(run.relative_to(guard.ROOT))}
    status = "FAILED"
    try:
        model = build_model(cfg, dev, backend="flex")
        model.load_state_dict(torch.load(run / "model.pt", map_location="cpu", weights_only=False)["model"])
        model.eval()
        src = eval_source(cfg, key, d, "calib_search")
        base = to_torch(src.batch(src.rows[:4]), dev)

        def timeit(fn, reps=20, warm=5):
            ts = []
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for i in range(warm + reps):
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    fn()
                    torch.cuda.synchronize()
                    if i >= warm:
                        ts.append(time.perf_counter() - t1)
            return {"p50_ms": 1e3 * float(np.median(ts)), "p95_ms": 1e3 * float(np.quantile(ts, .95))}
        pre = {}
        for C in (1024, 2048, 4096, 8192):
            B = 32768 // C
            tb = {k: v[:, :8192].reshape(4 * (8192 // C), C)[:B].contiguous() for k, v in base.items()}
            # re-index document ids per row after slicing (keep -1 pads); positions keep in-document offsets
            nb = C // 128
            row = {}
            for name, kw in (("REF", {}),
                             ("L1", {"ref": False, "gsel": torch.zeros(B, nb, dtype=torch.bool, device=dev),
                                     "kblk": torch.full((B, nb), 1, device=dev)}),
                             ("L4", {"ref": False, "gsel": torch.zeros(B, nb, dtype=torch.bool, device=dev),
                                     "kblk": torch.full((B, nb), 4, device=dev)}),
                             ("G4", {"ref": False, "gsel": torch.ones(B, nb, dtype=torch.bool, device=dev),
                                     "kblk": torch.full((B, nb), 4, device=dev)})):
                torch.cuda.reset_peak_memory_stats()
                r = timeit(lambda: model(tb, **kw))
                r["tokens_per_s"] = 32768 / (r["p50_ms"] / 1e3)
                r["peak_vram_gib"] = torch.cuda.max_memory_allocated() / 1024**3
                row[name] = r
            for nm in ("L1", "L4", "G4"):
                row[nm]["speedup_vs_REF"] = row["REF"]["p50_ms"] / row[nm]["p50_ms"]
            pre[f"{C}"] = {"batch": B, **row}
        out["prefill"] = pre
        # components at 4 x 8192
        lay = make_layout(base["doc_ids"])
        g = torch.zeros(lay.N // 128, dtype=torch.bool, device=dev)
        comp = {"mask_full_ms": timeit(lambda: full_or_local_block_mask(lay, None))["p50_ms"],
                "mask_local_ms": timeit(lambda: full_or_local_block_mask(lay, 512))["p50_ms"],
                "mask_mixed_ms": timeit(lambda: mixed_block_mask(lay, 512, g))["p50_ms"]}
        moe = model.blocks[1].ffn
        x = torch.randn(32768, model.cfg.d_model, device=dev, dtype=torch.bfloat16)
        with torch.no_grad():
            topi, gates, _ = moe.select(moe.router_logits(x), None)
        comp["experts_original_k4_ms"] = timeit(lambda: moe.experts(x, topi, gates))["p50_ms"]
        for k in (4, 2, 1):
            v = (torch.arange(4, device=dev)[None] < k).expand(32768, 4)
            comp[f"experts_null_k{k}_ms"] = timeit(lambda: experts_null(moe, x, topi, gates, v))["p50_ms"]
        comp["shared_expert_ms"] = timeit(lambda: moe.shared(x))["p50_ms"]
        comp["lm_head_ms"] = timeit(lambda: model.lm_head(model.norm_f(x)))["p50_ms"]
        out["components_4x8192"] = comp
        out["decode"] = "NOT RUN: no pilot (gate failed); decode path with rolling local KV caches not implemented"
        if mon.contaminated:
            raise guard.GuardError(f"CONTAMINATED: {mon.detail}")
        status = "COMPLETE"
    except Exception as e:  # noqa: BLE001
        import traceback
        out["error"] = f"{type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc()
        status = "FAILED"
    finally:
        mon.stop()
        out["status"] = status
        out["seconds"] = time.time() - t0
        st = guard.utc_stamp()
        guard.atomic_write_json(EXP["vrc"]["art"] / f"posthoc_timing_{st}.json", out)
        guard.gpu_ledger(t0iso, guard.now(), f"vrc_posthoc_timing_{st}", "vrc posthoc timing", status)
        print(json.dumps(out, indent=1)[:5000])


if __name__ == "__main__":
    main()
