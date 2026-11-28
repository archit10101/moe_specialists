"""The job graph. Every unit of work is a Job; SLURM (slurm/submit_all.sh) and the local runner
(`python -m repro.run local`) both execute exactly this graph.

  prepare  (cpu)  tokenize + split the corpus
  cluster  (gpu)  embeddings, clusters, every group table
  base     (gpu)  train + evaluate one base run          e.g. base:base.edl:s2
  bank     (gpu)  train + evaluate one residual bank      e.g. bank:bank.sem:s0
  router   (gpu)  train + evaluate one router             e.g. router:router.sem:s1
  report   (cpu)  results/REPORT.md from whatever has finished

A job is skipped when its DONE marker exists, so re-running is always safe (finished work is kept,
unfinished runs resume from their checkpoint).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import EXPERIMENTS, Profile, bank_base_seed, bank_cfg, get_experiment, router_cfg, runnable, seeds_for
from .utils import data_dir, groups_dir, is_done, run_dir

# per-job SLURM time limits (HH:MM:SS); the bench numbers in docs/ORCD.md explain them
# measured on ORCD L40S: a bank ~6 min (+ a few min for the full-val swap matrix), a router ~70 min. Short limits
# let SLURM backfill the jobs into gaps. A job that runs out of time saves a checkpoint; re-running
# submit_all.sh resumes it.
TIME = dict(prepare="01:00:00", cluster="02:00:00", base="06:00:00", bank="00:30:00", router="02:00:00",
            report="00:15:00")
CPU_STAGES = {"prepare", "report"}


@dataclass
class Job:
    key: str
    stage: str
    exp: str = ""
    seed: int = -1
    deps: list = field(default_factory=list)
    priority: int = 0                   # Experiment.priority (0 = first)
    seed_rank: int = 0                  # index of the seed in Experiment.seeds (lower seeds run first)

    @property
    def resource(self) -> str:
        return "cpu" if self.stage in CPU_STAGES else "gpu"

    @property
    def time(self) -> str:
        return TIME[self.stage]

    def args(self) -> list[str]:
        a = [self.stage]
        if self.exp:
            a += ["--exp", self.exp, "--seed", str(self.seed)]
        return a

    def done(self, p: Profile) -> bool:
        if self.stage == "prepare":
            return is_done(data_dir(p), "prepare")
        if self.stage == "cluster":
            return is_done(groups_dir(p), "cluster")
        if self.stage == "report":
            return False
        return is_done(run_dir(p, self.exp, self.seed), "eval")


def key(stage: str, exp: str, seed: int) -> str:
    return f"{stage}:{exp}:s{seed}"


def build_graph(p: Profile, tier: str = "core", only: list[str] | None = None) -> list[Job]:
    """All jobs needed for the selected experiments (dependencies are added even if they belong to
    another tier), in a valid execution order."""
    jobs: dict[str, Job] = {}
    order: list[str] = []

    def add(job: Job):
        if job.key not in jobs:
            jobs[job.key] = job
            order.append(job.key)
        return job.key

    add(Job("prepare", "prepare"))
    add(Job("cluster", "cluster", deps=["prepare"]))

    def need(exp_name: str, seed: int) -> str:
        exp = get_experiment(exp_name)
        if exp.kind == "base":
            deps = ["cluster"]
        elif exp.kind == "bank":
            bc = bank_cfg(p, exp)
            deps = ["cluster", need(bc.base, bank_base_seed(bc, seed))]
        else:
            rc = router_cfg(p, exp)
            deps = [need(rc.bank, rc.bank_seed), need("bank.tok", rc.bank_seed)]
        k = key(exp.kind, exp.name, seed)
        if k not in jobs:
            rank = exp.seeds.index(seed) if seed in exp.seeds else 0
            add(Job(k, exp.kind, exp.name, seed, deps, exp.priority, rank))
        return k

    for exp in EXPERIMENTS:
        if only:
            if exp.name not in only:
                continue
            seeds = exp.seeds
        else:
            seeds = seeds_for(exp, tier)
        if not seeds or not runnable(p, exp):
            continue
        for s in seeds:
            need(exp.name, s)
    add(Job("report", "report", deps=[k for k in order if k not in ("prepare", "cluster")]))
    return [jobs[k] for k in order]
