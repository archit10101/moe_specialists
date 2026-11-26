#!/usr/bin/env python
"""csr-context-prior-specialization-v2 inference cost (protocol 10, criterion E). V1's harness: forward-only, bf16
autocast, torch.compile, CALIB-SEARCH rows (throughput only), 16 rows x 1024 per batch, 3 warm-up + 20 timed batches,
median of 5 repetitions, interleaved order repeated twice (A B C C B A) to cancel drift. Models: reused V1 TOKEN-4
final s0, frozen candidate final s0 (no drafts, teacher deleted), M1 final s0. Writes artifacts/.../bench.json."""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402


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
    from csr.cps_lib import ART, RUN_PREFIX, V1_RUN_PREFIX
    from csr.lfr_eval import eval_rows
    from cps_evaluate import load_any
    guard.occupancy_gate("cps bench")
    device = guard.setup_cuda_device()
    guard.register_self("cps bench")
    torch.backends.cuda.matmul.allow_tf32 = True
    t_iso, t0 = guard.now(), time.time()
    sp = Split("calib_search")
    rows = eval_rows(sp, 200_000)
    batches = [to_torch(make_batch(rows[i:i + 16], sp), device) for i in range(0, 16 * 23, 16)]
    runs = {"TOKEN-4": sorted(glob.glob(str(guard.ROOT / f"runs/*_{V1_RUN_PREFIX}final_TOKEN-4_s0"))),
            "CANDIDATE": sorted(glob.glob(str(guard.ROOT / f"runs/*_{RUN_PREFIX}final_CAND_s0"))),
            "MEMORY-ONLY": sorted(glob.glob(str(guard.ROOT / f"runs/*_{RUN_PREFIX}final_MEMORY_s0")))}
    models = {}
    for k, c in runs.items():
        if not c or not Path(c[-1], "summary.json").exists():
            continue
        summ = json.loads(Path(c[-1], "summary.json").read_text())
        m, _ = load_any(summ["ckpts"]["p100"]["path"], device)
        assert not any(n.startswith("tc_") for n, _ in m.named_parameters())
        models[k] = (m, torch.compile(m), Path(c[-1]).name)
    reps = {k: [] for k in models}
    order = list(models) + list(models)[::-1]
    with torch.no_grad():
        for _ in range(2):
            for k in order:
                m, f, _ = models[k]
                for i in range(3):
                    b = batches[i]
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        f({"input_ids": b["input_ids"], "doc_ids": b["doc_ids"], "positions": b["positions"]}, {})
                torch.cuda.synchronize()
                for _r in range(3):
                    ts = time.time()
                    n = 0
                    for b in batches[3:]:
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            o = f({"input_ids": b["input_ids"], "doc_ids": b["doc_ids"], "positions": b["positions"]},
                                  {})
                            m.lm_logits(o["hidden"])
                        n += int((b["doc_ids"] >= 0).sum())
                    torch.cuda.synchronize()
                    reps[k].append(n / (time.time() - ts))
    res = {}
    for k, (m, _, rn) in models.items():
        res[k] = {"tok_per_s_median": float(np.median(reps[k])), "reps": reps[k], "params": m.param_counts(),
                  "run": rn}
    if "TOKEN-4" in res:
        b = res["TOKEN-4"]
        for k, v in res.items():
            v["speed_ratio_vs_token4"] = v["tok_per_s_median"] / b["tok_per_s_median"]
            v["active_ratio_vs_token4"] = v["params"]["active_per_token"] / b["params"]["active_per_token"]
    res["criterion_E"] = {"rule": "speed ratio >= 0.85 and active ratio <= 1.01",
                          "pass": bool("CANDIDATE" in res and res["CANDIDATE"]["speed_ratio_vs_token4"] >= 0.85 and
                                       res["CANDIDATE"]["active_ratio_vs_token4"] <= 1.01)}
    res["seconds"] = time.time() - t0
    guard.atomic_write_json(ART / "bench.json", res, overwrite=True)
    guard.gpu_ledger(t_iso, guard.now(), "cps_bench", "cps bench", "COMPLETE")
    with open(ART / "compute_ledger.tsv", "a") as fh:
        fh.write(f"{t_iso}\t{guard.now()}\t{time.time() - t0:.1f}\tcps_bench\tbench\tbenchmark\tCOMPLETE\n")
    print(json.dumps({k: (v.get("speed_ratio_vs_token4") if isinstance(v, dict) and "speed_ratio_vs_token4" in v
                          else None) for k, v in res.items()}))


if __name__ == "__main__":
    main()
