#!/usr/bin/env python
"""csr-sequence-routing-search-v1: evaluate one checkpoint on CALIB-SELECTION or the validation split.

Access is locked (srs_run.check_eval_access): CALIB-SELECTION only after search_frozen.json lists the run's config;
val only after final_frozen.json lists it. One evaluation per (checkpoint, set). Test split: always refused.
Dry-run by default; GPU only via scripts/gpu_guarded.sh with --execute and MOE_RESEARCH_APPROVE=gpu.
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--set", required=True, choices=["calib_selection", "val", "calib_search"])
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    from csr.srs_run import check_cfg, check_eval_access
    rd = guard.inside_root(a.run)
    cfg = json.loads((rd / "config.json").read_text())
    check_cfg(cfg)
    facts = json.loads((rd / "run_facts.json").read_text())
    check_eval_access(a.set, facts["config_sha256"])
    if json.loads((rd / "status.json").read_text())["status"] != "COMPLETE":
        raise guard.GuardError("run not COMPLETE")
    print(json.dumps({"run": str(rd), "set": a.set, "device": a.device}))
    if not guard.require_execution(a.execute, "gpu" if a.device == "cuda" else "cpu-model"):
        return
    sys.exit(0 if run(cfg, rd, facts, a) == "COMPLETE" else 1)


def run(cfg, rd, facts, a):
    import numpy as np
    import torch
    from csr.seqroute import build_sr_model
    from csr.srs_run import claim_eval, collapse_check, eval_source, sr_evaluate
    torch.set_num_threads(4)
    monitor = None
    if a.device == "cuda":
        guard.occupancy_gate(f"srsv1 eval:{rd.name}:{a.set}")
        device = guard.setup_cuda_device()
        monitor = guard.OccupancyMonitor(interval=15.0)
        monitor.start()
    else:
        device = torch.device("cpu")
    guard.register_self(f"srsv1 eval {rd.name} {a.set}")
    ck = rd / "model.pt"
    ck_sha = guard.sha256_file(ck)
    summ = json.loads((rd / "summary.json").read_text())
    if ck_sha != summ.get("checkpoint_sha256"):
        raise guard.GuardError("checkpoint hash mismatch")
    claim = claim_eval(a.set, ck_sha)
    t_iso, t0 = guard.now(), time.time()
    out_path = rd / f"eval_{a.set}_{guard.utc_stamp()}.json"
    status, reason, res = "FAILED", "", {}
    limits = guard.RunLimits(max_seconds=3600, monitor=monitor)
    try:
        model = build_sr_model(cfg, device)
        sd = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_state_dict(sd["model"], strict=True)
        _, src = eval_source(cfg, a.set, cfg["data"]["dir"])
        use_amp = device.type == "cuda" and cfg["train"]["precision"] == "bf16-autocast"
        res = sr_evaluate(model, src, device, cfg["eval"]["rows_per_batch"], None, autocast=use_amp)
        res["collapse"] = collapse_check(res)
        limits.check()
        status = "COMPLETE"
    except guard.GuardError as e:
        status = "CONTAMINATED" if "CONTAMINATED" in str(e) else "ABORTED"
        reason = str(e)
    except Exception as e:  # noqa: BLE001
        status, reason = "FAILED", f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        limits.close()
        if monitor is not None:
            monitor.stop()
            if monitor.contaminated and status == "COMPLETE":
                status, reason = "CONTAMINATED", monitor.detail
        doc = {"status": status, "reason": reason, "evalset": a.set, "run": str(rd.relative_to(guard.ROOT)),
               "candidate": cfg["candidate"], "family": cfg["family"], "seed": cfg["init"]["seed"],
               "config_sha256": facts["config_sha256"], "checkpoint_sha256": ck_sha,
               "process_seconds": time.time() - t0, **res}
        guard.atomic_write_json(out_path, doc)
        guard.atomic_write_json(claim, {"status": status if status == "COMPLETE" else "FAILED_NO_RESULT",
                                        "time": guard.now(), "result": str(out_path.relative_to(guard.ROOT))},
                                overwrite=True)
        if a.device == "cuda":
            guard.gpu_ledger(t_iso, guard.now(), rd.name, f"srsv1 eval {a.set}", status)
        print(f"EVAL_JSON {out_path.relative_to(guard.ROOT)}")
    return status


if __name__ == "__main__":
    main()
