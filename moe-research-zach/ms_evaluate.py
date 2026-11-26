#!/usr/bin/env python
"""csr-multiscale-routing-v1: full-split evaluation of a COMPLETE run with its own routing and boundaries.

Splits: val (primary) or calib. The test split is refused before any file access. Writes
eval_<split>_<stamp>.json (CE, per-document NLL sums/counts, routing usage) and
eval_<split>_<stamp>_tokens.npz (float16 NLL of every valid token in canonical packed order). Dry-run by default.
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
    ap.add_argument("--split", default="val")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--max-rows", type=int)
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    from csr.multiscale import TestSplitLocked
    if a.split not in ("val", "calib"):
        raise TestSplitLocked(f"split {a.split!r} refused")
    run = guard.inside_root(a.run)
    if json.loads((run / "status.json").read_text())["status"] != "COMPLETE":
        raise guard.GuardError("run is not COMPLETE")
    print(f"msv1 evaluate {run} on {a.split}")
    if not guard.require_execution(a.execute, "gpu" if a.device == "cuda" else "cpu-model"):
        return
    mon = None
    if a.device == "cuda":
        guard.occupancy_gate("msv1 evaluate")
        dev = guard.setup_cuda_device()
        mon = guard.OccupancyMonitor(10.0)
        mon.start()
    import numpy as np
    import torch
    from csr.ms_run import build_ms_model, check_cfg, make_sources, ms_evaluate
    from csr.multiscale import MSRouting
    if a.device == "cpu":
        dev = torch.device("cpu")
    guard.register_self(f"msv1 evaluate {run.name} {a.split}")
    t0iso, t0 = guard.now(), time.time()
    lim = guard.RunLimits(guard.MAX_ATOMIC_SECONDS, monitor=mon)
    status, reason, res = "INCOMPLETE", "", {}
    stamp = guard.utc_stamp()
    out_path = run / f"eval_{a.split}_{stamp}.json"
    tok_path = run / f"eval_{a.split}_{stamp}_tokens.npz"
    try:
        sd = torch.load(run / "model.pt", map_location="cpu", weights_only=True)
        cfg = sd["config"]
        if cfg != json.loads((run / "config.json").read_text()):
            raise guard.GuardError("checkpoint config differs from run config")
        d = check_cfg(cfg)
        model = build_ms_model(cfg, dev)
        model.load_state_dict(sd["model"], strict=True)
        _, src = make_sources(cfg, d, a.split)
        res = ms_evaluate(model, src, MSRouting.from_cfg(cfg["routing"]), dev, cfg["eval"]["rows_per_batch"],
                          a.max_rows, autocast=(a.device == "cuda"), per_token=True)
        nll = res.pop("_per_token_nll_f16")
        tmp = run / f".eval_{a.split}_{stamp}_tokens.tmp.npz"
        np.savez(tmp, nll_f16=nll)
        tmp.rename(tok_path)
        res["per_token_file"] = tok_path.name
        res["checkpoint_sha256"] = guard.sha256_file(run / "model.pt")
        res["eval_seconds"] = time.time() - t0
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
            if mon.contaminated and status == "COMPLETE":
                status, reason = "CONTAMINATED", mon.detail
        guard.atomic_write_json(out_path, {"status": status, "reason": reason, "split": a.split, **res})
        if a.device == "cuda":
            guard.gpu_ledger(t0iso, guard.now(), run.name, f"msv1 evaluate:{a.split}", status)
        print(status, reason, {k: v for k, v in res.items() if k not in ("doc_nll", "doc_n")})
        print(f"EVAL_JSON {out_path.relative_to(guard.ROOT)}")
    sys.exit(0 if status == "COMPLETE" else 1)


if __name__ == "__main__":
    main()
