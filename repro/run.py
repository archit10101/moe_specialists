"""Command-line entry point. Every stage of the pipeline is one sub-command.

  python -m repro.run list                                   experiments in the registry
  python -m repro.run plan  [--tier core|ablation|all]       the job graph (what submit_all.sh submits)
  python -m repro.run local [--tier ...]                     run the whole graph here, in order
  python -m repro.run prepare [--download]                   data (--download: login-node fetch only)
  python -m repro.run cluster
  python -m repro.run base   --exp base.edl   --seed 0       train + evaluate one base run
  python -m repro.run bank   --exp bank.sem   --seed 0       train + evaluate one residual bank
  python -m repro.run router --exp router.sem --seed 0       train + evaluate one router
  python -m repro.run report
  python -m repro.run bench                                  time each job type here, project GPU hours

Global option: --profile full|smoke (default: $REPRO_PROFILE or full).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from .config import EXPERIMENTS, get_experiment, get_profile, runnable
from .utils import log


def stage_base(p, exp, seed):
    from .eval.base_eval import evaluate_base
    from .train.base import train_base
    train_base(p, exp, seed)
    evaluate_base(p, exp, seed)


def stage_bank(p, exp, seed):
    from .eval.bank_eval import evaluate_bank
    from .train.residual import train_bank
    train_bank(p, exp, seed)
    evaluate_bank(p, exp, seed)


def stage_router(p, exp, seed):
    from .eval.router_eval import evaluate_router
    from .train.router import train_router
    train_router(p, exp, seed)
    evaluate_router(p, exp, seed)


def run_job(p, job) -> None:
    t = time.time()
    if job.stage == "prepare":
        from .data.prepare import build
        build(p)
    elif job.stage == "cluster":
        from .clusters.build import build
        build(p)
    elif job.stage == "report":
        from .report import write_report
        write_report(p)
    else:
        exp = get_experiment(job.exp)
        assert exp.kind == job.stage, f"{job.exp} is a {exp.kind} experiment, not {job.stage}"
        {"base": stage_base, "bank": stage_bank, "router": stage_router}[job.stage](p, exp, job.seed)
    log(f"job {job.key} finished in {time.time() - t:.0f}s")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m repro.run", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["list", "plan", "local", "prepare", "cluster", "base", "bank",
                                        "router", "report", "bench"])
    ap.add_argument("--profile", default=os.environ.get("REPRO_PROFILE", "full"))
    ap.add_argument("--exp", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tier", default="core", choices=["core", "ablation", "all"])
    ap.add_argument("--only", default="", help="comma-separated experiment names (overrides --tier)")
    ap.add_argument("--download", action="store_true", help="prepare: only fetch the raw data (login node)")
    ap.add_argument("--format", default="table", choices=["table", "tsv"])
    a = ap.parse_args(argv)
    p = get_profile(a.profile)
    from .pipeline import Job, build_graph

    if a.command == "list":
        for e in EXPERIMENTS:
            flag = "" if runnable(p, e) else "   (not runnable in this profile)"
            print(f"{e.name:28s} {e.kind:6s} {e.tier:8s} seeds={list(e.seeds)}  {e.doc}{flag}")
        return
    if a.command in ("plan", "local"):
        jobs = build_graph(p, a.tier, [x for x in a.only.split(",") if x] or None)
        if a.command == "plan":
            for j in jobs:
                state = "done" if j.done(p) else "todo"
                if a.format == "tsv":
                    print("\t".join([j.key, j.resource, j.time, state, ",".join(j.deps),
                                     f"{j.priority}:{j.seed_rank}", " ".join(j.args())]))
                else:
                    print(f"{state:4s}  P{j.priority}  {j.key:42s} {j.resource}  deps: {', '.join(j.deps) or '-'}")
            return
        for j in jobs:
            if j.done(p):
                log(f"skip {j.key} (done)")
                continue
            log(f"=== {j.key}")
            run_job(p, j)
        return
    if a.command == "bench":
        from .bench import bench
        bench(p)
        return
    if a.command == "prepare" and a.download:
        from .data.prepare import download
        download(p)
        return
    if a.command in ("base", "bank", "router") and not a.exp:
        sys.exit(f"{a.command} needs --exp")
    run_job(p, Job(f"{a.command}:{a.exp}:s{a.seed}" if a.exp else a.command, a.command, a.exp, a.seed))


if __name__ == "__main__":
    main()
