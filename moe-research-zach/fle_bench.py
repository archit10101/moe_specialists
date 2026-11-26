#!/usr/bin/env python
"""csr-forecast-latent-edl-v1: student-only inference benchmark (prereg 11 cost). Loads p100 milestones of the given
final runs as plain LFRModel (no drafts, encoder, cache, q tables or training-only heads; asserts that no fle forecast
module is imported), compiled bf16 forward on fixed CALIB-SEARCH rows (throughput only), median of 5 timed passes.
  fle_bench.py --runs final_T4_s0 final_FL_s0 ... --device cuda --execute"""
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
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    if not guard.require_execution(a.execute, "gpu" if a.device == "cuda" else "cpu-model"):
        return
    import numpy as np
    import torch
    from csr.fsp_lib import Split, make_batch, to_torch
    from csr.lfr_eval import eval_rows
    from fle_evaluate import load_milestone
    from csr.fle_paths import ART
    device = guard.setup_cuda_device() if a.device == "cuda" else torch.device("cpu")
    st = json.loads((ART / "driver_state.json").read_text())
    sp = Split("calib_search")
    rows = eval_rows(sp, 100_000)[:64]
    bt = to_torch(make_batch(rows, sp), device)
    minp = {"input_ids": bt["input_ids"], "doc_ids": bt["doc_ids"], "positions": bt["positions"]}
    ntok = int((bt["doc_ids"] >= 0).sum())
    out = {"time": guard.now(), "rows": len(rows), "tokens_per_pass": ntok, "runs": {}}
    for rn in a.runs:
        rd = guard.ROOT / st["run_dirs"][rn]
        ck = json.loads((rd / "status.json").read_text())["ckpts"]["p100"]["path"]
        m, _ = load_milestone(ck, device)
        f = torch.compile(m)
        times = []
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            for i in range(8):
                torch.cuda.synchronize()
                t0 = time.time()
                for s in range(0, 64, 16):
                    h = f({k: v[s:s + 16] for k, v in minp.items()})["hidden"]
                    _ = m.lm_logits(h[:, -1])
                torch.cuda.synchronize()
                if i >= 3:
                    times.append(time.time() - t0)
        out["runs"][rn] = {"tokens_per_s_median": ntok / float(np.median(times)), "params": m.param_counts(),
                           "state_dict_keys_training_only": [k for k in m.state_dict() if k.startswith(("fd_", "tc_"))]}
        del m, f
        torch.cuda.empty_cache()
    out["forecast_modules_loaded"] = [k for k in sys.modules if k.startswith("csr.fle_lib")]
    assert not out["forecast_modules_loaded"], "forecast code was loaded at inference"
    guard.atomic_write_json(ART / "inference_bench.json", out, overwrite=True)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
