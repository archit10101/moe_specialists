#!/usr/bin/env python
"""csr-domain-forecast-curriculum-v1: standalone inference throughput of final models (plain CSRModel export, strength 0,
no draft/forecasts/labels). Prefill: 16 x 1024 random-token rows (single document per row), bf16 autocast, 5 warm-up +
20 timed forwards. Decode: batch-1 KV-cache incremental decoding (CSRModel.step, token routing), 128-token prompt then
256 greedy tokens, fp32. Also: forward cost of the curriculum path (SOFT at strength 1 vs 0, same batch) for the
overhead table, and parameter counts. Calibration-free (random tokens). Writes
artifacts/domain-forecast-curriculum-v1/bench_<stamp>.json. Dry-run by default."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="comma list of run dirs (final checkpoints)")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    if not guard.require_execution(a.execute, "gpu"):
        return
    import numpy as np
    import torch
    from csr.dfc_lib import ART, code_hashes
    from csr.dfc_model import build_dfc_model, to_plain_csr
    from csr.model import RoutingConfig, count_parameters
    guard.occupancy_gate("dfc bench")
    device = guard.setup_cuda_device()
    mon = guard.OccupancyMonitor(interval=15.0)
    mon.start()
    guard.register_self("dfc bench")
    t_iso = guard.now()
    out = {"runs": {}, "code_sha256": code_hashes()}
    status = "FAILED"
    try:
        rng = np.random.default_rng(0)
        B, T = 16, 1024
        tb = {"input_ids": torch.tensor(rng.integers(2, 50000, (B, T)), device=device),
              "doc_ids": torch.zeros(B, T, dtype=torch.long, device=device),
              "positions": torch.arange(T, device=device).repeat(B, 1),
              "domain": torch.tensor(rng.integers(0, 5, (B, 1)), device=device).repeat(1, T),
              "fc": torch.tensor(rng.integers(2, 50000, (B, T, 4)), device=device)}
        for rdir in a.runs.split(","):
            rd = guard.inside_root(rdir)
            cfg = json.loads((rd / "config.json").read_text())
            summ = json.loads((rd / "summary.json").read_text())
            ck = torch.load(guard.ROOT / summ["final_checkpoint"], map_location="cpu", weights_only=False)
            dfc = build_dfc_model(cfg["model"], cfg["condition"], cfg["seed"], device)
            dfc.load_state_dict(ck["model"])
            dfc.eval()
            plain = to_plain_csr(dfc).eval()
            rc = RoutingConfig()

            def timeit(fn, warm=5, n=20):
                for _ in range(warm):
                    fn()
                torch.cuda.synchronize()
                t0 = time.time()
                for _ in range(n):
                    fn()
                torch.cuda.synchronize()
                return (time.time() - t0) / n
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                pre = timeit(lambda: plain({k: tb[k] for k in ("input_ids", "doc_ids", "positions")}, rc))
                cur1 = timeit(lambda: dfc(tb, strength=1.0))
                cur0 = timeit(lambda: dfc(tb, strength=0.0))
            prompt = rng.integers(2, 50000, 128).tolist()

            def decode():
                st = plain.new_state()
                tok = None
                for i, x in enumerate(prompt):
                    lg, _ = plain.step(x, i, i, rc, st)
                tok = int(lg.argmax())
                for j in range(256):
                    lg, _ = plain.step(tok, 128 + j, 128 + j, rc, st)
                    tok = int(lg.argmax())
            with torch.no_grad():
                decode()
                torch.cuda.synchronize()
                t0 = time.time()
                decode()
                torch.cuda.synchronize()
                dsec = time.time() - t0
            pc = count_parameters(plain)
            pd = count_parameters(dfc)
            out["runs"][cfg["run_name"]] = {
                "prefill_tokens_per_s": B * T / pre, "prefill_s_per_batch": pre,
                "curriculum_forward_s_strength1": cur1, "curriculum_forward_s_strength0": cur0,
                "curriculum_forward_overhead_frac": cur1 / cur0 - 1,
                "decode_tokens_per_s": (128 + 256) / dsec, "decode_ms_per_token": 1000 * dsec / (128 + 256),
                "params_standalone": pc, "params_with_curriculum_modules": pd["total"],
                "added_params": dfc.added_parameter_count()}
            print(cfg["run_name"], json.dumps(out["runs"][cfg["run_name"]]), flush=True)
        status = "COMPLETE"
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        out["error"] = f"{type(e).__name__}: {e}"
    finally:
        mon.stop()
        if mon.contaminated:
            status = "CONTAMINATED"
        out["status"] = status
        p = ART / f"bench_{guard.utc_stamp()}.json"
        guard.atomic_write_json(p, out)
        guard.gpu_ledger(t_iso, guard.now(), p.stem, "dfc bench", status)
        print("STATUS", status, p)
    sys.exit(0 if status == "COMPLETE" else 1)


if __name__ == "__main__":
    main()
