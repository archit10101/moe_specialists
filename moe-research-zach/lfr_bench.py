#!/usr/bin/env python
"""csr-latent-forecast-router-distill-v1: inference cost (protocol 13 criterion 10). Same harness for every final
condition's 100% seed-0 checkpoint: forward-only, bf16 autocast, torch.compile, CALIB-SEARCH rows (throughput only),
16 rows x 1024 per batch, 3 warm-up + 20 timed batches, median of 5 repetitions. Also inference / active parameter
counts. Writes artifacts/latent-forecast-router-distill-v1/bench.json."""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402

CONDS = ["TOKEN-4", "EDL", "OUTPUT-DISTILL", "CF-FORECAST-STUDENT", "EDL+CF", "FORECAST-SIMILARITY",
         "SHUFFLED-FORECAST", "MEMORY-ONLY"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    if not guard.require_execution(a.execute, "gpu" if a.device == "cuda" else "cpu-model"):
        return
    import numpy as np
    import torch
    from csr.fsp_lib import Split, make_batch, to_torch
    from csr.lfr_lib import ART, RUN_PREFIX
    from csr.lfr_eval import eval_rows
    from lfr_evaluate import load_milestone
    guard.occupancy_gate("lfr bench")
    device = guard.setup_cuda_device()
    guard.register_self("lfr bench")
    torch.backends.cuda.matmul.allow_tf32 = True
    t_iso, t0 = guard.now(), time.time()
    sp = Split("calib_search")
    rows = eval_rows(sp, 200_000)
    batches = [to_torch(make_batch(rows[i:i + 16], sp), device) for i in range(0, 16 * 23, 16)]
    res = {}
    for cond in CONDS:
        c = sorted(glob.glob(str(guard.ROOT / f"runs/*_{RUN_PREFIX}final_{cond}_s0")))
        if not c or not Path(c[-1], "summary.json").exists():
            continue
        summ = json.loads(Path(c[-1], "summary.json").read_text())
        m, _ = load_milestone(summ["ckpts"]["p100"]["path"], device)
        assert not any(n.startswith("tc_") for n, _ in m.named_parameters())
        f = torch.compile(m)
        reps = []
        with torch.no_grad():
            for r in range(5):
                for i in range(3):
                    b = batches[i]
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        f({"input_ids": b["input_ids"], "doc_ids": b["doc_ids"], "positions": b["positions"]}, {})
                torch.cuda.synchronize()
                ts = time.time()
                n = 0
                for b in batches[3:]:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        o = f({"input_ids": b["input_ids"], "doc_ids": b["doc_ids"], "positions": b["positions"]}, {})
                        m.lm_logits(o["hidden"])
                    n += int((b["doc_ids"] >= 0).sum())
                torch.cuda.synchronize()
                reps.append(n / (time.time() - ts))
        res[cond] = {"tok_per_s_median": float(np.median(reps)), "reps": reps, "params": m.param_counts(),
                     "run": Path(c[-1]).name}
        del m, f
        torch.cuda.empty_cache()
    if "TOKEN-4" in res:
        b = res["TOKEN-4"]
        for k, v in res.items():
            v["speed_ratio_vs_token4"] = v["tok_per_s_median"] / b["tok_per_s_median"]
            v["active_ratio_vs_token4"] = v["params"]["active_per_token"] / b["params"]["active_per_token"]
    res["seconds"] = time.time() - t0
    guard.atomic_write_json(ART / "bench.json", res, overwrite=True)
    guard.gpu_ledger(t_iso, guard.now(), "lfr_bench", "lfr bench", "COMPLETE")
    with open(ART / "compute_ledger.tsv", "a") as fh:
        fh.write(f"{t_iso}\t{guard.now()}\t{time.time() - t0:.1f}\tlfr_bench\tbench\tbenchmark\tCOMPLETE\n")
    print(json.dumps({k: (v.get("speed_ratio_vs_token4") if isinstance(v, dict) else v) for k, v in res.items()}))


if __name__ == "__main__":
    main()
