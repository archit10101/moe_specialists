#!/usr/bin/env python
"""Throughput / peak-memory benchmark: N representative optimizer steps of one condition through the exact
training code path (no evaluation, no checkpoint). Dry-run by default; GPU via scripts/gpu_guarded.sh."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402
from train import resolve_config, run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="configs/v2/base.json")
    ap.add_argument("--condition", required=True, choices=list("ABCDEF"))
    ap.add_argument("--set", action="append")
    ap.add_argument("--run-name")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    cfg = resolve_config(a)
    cfg["train"]["save_checkpoint"] = False
    run(cfg, a.device, a.execute, "benchmark", a.steps)


if __name__ == "__main__":
    main()
