#!/usr/bin/env python
"""csr-conditional-local-attention-v1: evaluate one frozen checkpoint once on CALIB-SELECTION or the validation split.

Access rules (src/csr/cla_run.check_eval_access): calib_selection only for checkpoints whose config is listed in
search_frozen.json; val only for checkpoints whose config is listed in final_frozen.json and only after every final
run is complete (final_trained.json). One evaluation per (checkpoint, set) via the eval registry. Test is refused.
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
    ap.add_argument("--set", required=True, choices=["calib_selection", "val"])
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    import torch
    from csr.cla_model import build_cla_model
    from csr.cla_run import (ART, check_cfg, check_eval_access, claim_eval, cla_evaluate, collapse_check,
                             eval_source)
    rd = guard.inside_root(a.run)
    cfg = json.loads((rd / "config.json").read_text())
    d = check_cfg(cfg)
    facts = json.loads((rd / "run_facts.json").read_text())
    summ = json.loads((rd / "summary.json").read_text())
    ck = rd / "model.pt"
    if guard.sha256_file(ck) != summ["checkpoint_sha256"]:
        raise guard.GuardError("checkpoint hash mismatch")
    check_eval_access(a.set, facts["config_sha256"])
    print(json.dumps({"run": str(rd), "set": a.set, "candidate": cfg["candidate"]}))
    if not guard.require_execution(a.execute, "gpu" if a.device == "cuda" else "cpu-model"):
        return
    monitor = None
    if a.device == "cuda":
        guard.occupancy_gate(f"clav1 eval {a.set}:{rd.name}")
        device = guard.setup_cuda_device()
        monitor = guard.OccupancyMonitor(interval=15.0)
        monitor.start()
    else:
        device = torch.device("cpu")
    guard.register_self(f"clav1 eval {a.set} {rd.name}")
    reg = claim_eval(a.set, summ["checkpoint_sha256"])
    t_iso = guard.now()
    out = ART / "evals" / f"{a.set}__{rd.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    status, res = "FAILED", {}
    try:
        model = build_cla_model(cfg, device)
        sd = torch.load(ck, map_location=device, weights_only=False)["model"]
        model.load_state_dict(sd)
        _, src = eval_source(cfg, a.set, d)
        t0 = time.time()
        res = cla_evaluate(model, src, device, cfg["eval"]["rows_per_batch"], None,
                           autocast=(device.type == "cuda"), diagnostics=True, token_stats=(a.set == "val"))
        res["collapse"] = collapse_check(res)
        res.update({"evalset": a.set, "run": str(rd.relative_to(guard.ROOT)), "candidate": cfg["candidate"],
                    "seed": cfg["init"]["seed"], "checkpoint_sha256": summ["checkpoint_sha256"],
                    "config_sha256": facts["config_sha256"], "wall_s": time.time() - t0})
        status = "COMPLETE"
        if monitor is not None and monitor.contaminated:
            status = "CONTAMINATED"
    except Exception as e:  # noqa: BLE001
        res = {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()}
    finally:
        if monitor is not None:
            monitor.stop()
        res["status"] = status
        guard.atomic_write_json(out, res, overwrite=True)
        guard.atomic_write_json(reg, {"status": status, "time": guard.now(), "result": str(out.relative_to(guard.ROOT))},
                                overwrite=True)
        if a.device == "cuda":
            guard.gpu_ledger(t_iso, guard.now(), rd.name, f"clav1 eval {a.set}", status)
        print(f"EVAL_JSON {out.relative_to(guard.ROOT)}")
        print(json.dumps({k: res.get(k) for k in ("status", "ce_nats_per_token", "n_valid_tokens")}))
    sys.exit(0 if status == "COMPLETE" else 1)


if __name__ == "__main__":
    main()
