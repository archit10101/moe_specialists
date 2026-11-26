#!/usr/bin/env python
"""Held-out evaluation of a trained run with its own condition's causal routing and segmentation.

Primary endpoint: token-weighted next-token cross-entropy (nats per valid prediction token). Writes per-document
NLL sums for paired document-level bootstrap. The test split is locked: it needs --unlock-test AND
MOE_RESEARCH_TEST_UNLOCK=1, which are NOT authorized in Stages A/B. Dry-run by default.
"""
import argparse
import json
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402
from train import load_seg  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory containing model.pt")
    ap.add_argument("--split", default="val", choices=["val", "calib", "test"])
    ap.add_argument("--unlock-test", action="store_true")
    ap.add_argument("--max-rows", type=int)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    run = guard.inside_root(a.run)
    if a.split == "test" and not (a.unlock_test and os.environ.get("MOE_RESEARCH_TEST_UNLOCK") == "1"):
        raise guard.GuardError("test split locked (requires separate authorization)")
    if json.loads((run / "status.json").read_text())["status"] != "COMPLETE":
        raise guard.GuardError("run is not COMPLETE; refusing to evaluate an incomplete run")
    print(f"evaluate {run} on {a.split}")
    if not guard.require_execution(a.execute, "gpu" if a.device == "cuda" else "cpu-model"):
        return
    mon = None
    if a.device == "cuda":
        guard.occupancy_gate("evaluate")
        dev = guard.setup_cuda_device()
        mon = guard.OccupancyMonitor(10.0)
        mon.start()
    import torch
    from csr.data import TokenSplit
    from csr.train_lib import BatchSource, evaluate
    from csr.model import CSRModel, ModelConfig
    if a.device == "cpu":
        dev = torch.device("cpu")
    guard.register_self(f"evaluate {run.name} {a.split}")
    t0iso = guard.now()
    lim = guard.RunLimits(guard.MAX_ATOMIC_SECONDS, monitor=mon)
    status, reason, res = "INCOMPLETE", "", {}
    label = "unknown"  # replaced by the checkpoint's resolved run config label once loaded
    out_path = run / f"eval_{a.split}_{guard.utc_stamp()}.json"
    try:
        sd = torch.load(run / "model.pt", map_location="cpu", weights_only=True)
        cfg = sd["config"]
        label = cfg.get("result_label", "experiment")
        model = CSRModel(ModelConfig.from_dict(cfg["model"]))
        model.load_state_dict(sd["model"])
        model.to(dev).eval()
        sp = TokenSplit.load(guard.inside_root(cfg["data"]["dir"]), a.split)
        src = BatchSource(sp, cfg["data"]["seq_len"], load_seg(cfg, a.split))
        res = evaluate(model, src, cfg, dev, cfg["eval"]["rows_per_batch"], a.max_rows, autocast=(a.device == "cuda"))
        lim.check()
        status = "COMPLETE"
    except guard.GuardError as e:
        status, reason = ("CONTAMINATED" if "CONTAMINATED" in str(e) else "ABORTED"), str(e)
    except Exception as e:  # noqa: BLE001
        status, reason = "FAILED", f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        lim.close()
        if mon is not None:
            mon.stop()
        guard.atomic_write_json(out_path, {"status": status, "reason": reason, "split": a.split,
                                             "result_label": label, **res})
        if a.device == "cuda":
            guard.gpu_ledger(t0iso, guard.now(), run.name, f"evaluate:{a.split}", status)
        print(status, reason, {k: v for k, v in res.items() if k not in ("doc_nll", "doc_n")})


if __name__ == "__main__":
    main()
