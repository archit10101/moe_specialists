# Reproducing the successful subject-specialisation MoE experiments

This repository is a reimplementation of the experiments in `docs/source_history.md` (the experiment history
from lambda2). It was first written from that doc alone. Once the original author's code became available
(`moe-research-zach/` on `main`, which has no checkpoints or results), we aligned every setting that could change
a conclusion with it. `docs/RECONSTRUCTION.md` lists what came from where and the differences we kept. The repo
re-tests the three results the history reports as positive, plus ablations.

| # | claim (doc §) | doc result | what is run here |
|---|---|---|---|
| 1 | **Explicit-label EDL beats TOKEN-4** (§3, §18) | val CE 3.0435 vs 3.0533 (−0.0098, 4 seeds, 245M tokens) | `base.token4`, `base.edl`, plus the label-free `base.sem_c6` and its shuffled control `base.shuf_c6` |
| 2 | **Oracle residual specialists are genuinely subject-specific** (§23) | +0.00337 CE over the frozen base; beats shuffled by 0.00240 and the token router by 0.00127; exposure-matched specificity 0.00484 vs ≈0 | `bank.sem`, `bank.shuf`, `bank.tok` on the frozen `base.token4` seed 0 |
| 3 | **A causal router can deploy them** (§27) | beats the generic bank G by 0.0038–0.0041 in 3 router seeds; keeps ~75% of the oracle's semantic-over-shuffled advantage | `router.sem`, `router.shuf` |

Ablations B1–B10 are described in [docs/ABLATIONS.md](docs/ABLATIONS.md). They include B4, a direct test
of the hypothesis that subject experts only pay off once the base is already competent, and B9, which
asks whether the router result replicates across independently trained specialist banks.

## How the code is organised

Everything that defines an experiment is in **one file**, [repro/config.py](repro/config.py): the sizes,
the hyperparameters and the registry of experiments (`EXPERIMENTS`). The rest is plumbing:

```
repro/
  config.py          profiles (full / smoke) + experiment registry       <- start here
  run.py             CLI: one sub-command per pipeline stage
  pipeline.py        job graph (prepare → cluster → base → bank → router → report)
  data/              prepare.py (M2D2 → paper-disjoint token streams), loader.py (batches, group tables)
  clusters/build.py  bge-small multi-view embeddings → balanced k-means centroids → softmax posteriors → semantic / shuffled tables
  model/             moe.py (TOKEN-4 MoE), losses.py (Switch, z, EDL), residual.py (bank, routers)
  train/             base.py, residual.py (banks), router.py
  eval/              base_eval.py (CE, NMI, FSI, interaction, exposure-matched), bank_eval.py (swap matrix),
                     router_eval.py (vs G, positions, retention inputs), metrics.py
  report.py          results/REPORT.md with PASS / FAIL per claim
  bench.py           GPU timing → projected GPU-hours
slurm/               ORCD: setup_login.sh, submit_all.sh, status.sh, job.sbatch, config.sh
tests/               CPU tests (python -m pytest); REPRO_SLOW=1 adds the full smoke pipeline
docs/                source_history.md, RECONSTRUCTION.md (every guess), ABLATIONS.md, ORCD.md, CLOUD.md
```

Each job is `python -m repro.run <stage> [--exp NAME --seed S]`. SLURM, your laptop and a cloud session
all run the same commands.

## Quick start

**Laptop / cloud (CPU, ~2 min):** this runs the whole pipeline on tiny synthetic data.
```bash
pip install torch numpy scipy scikit-learn pytest      # + tiktoken transformers huggingface_hub for real data
python -m pytest                                        # 19 fast tests
python -m repro.run local --profile smoke               # every stage end to end → work/smoke/results/REPORT.md
```

**ORCD (the real run):** see [docs/ORCD.md](docs/ORCD.md).
```bash
./slurm/launch.sh                    # all of the below in one go: setup, smoke test, then the core run
                                     # held behind a smoke "gate" job; detaches, check back tomorrow
./slurm/setup_login.sh               # once, on the login node
./slurm/submit_all.sh --smoke        # whole pipeline on a GPU node, ~15 min
./slurm/submit_all.sh --bench        # measured GPU-hours
./slurm/submit_all.sh                # core reproduction (37 jobs)
./slurm/submit_all.sh --tier ablation
./slurm/status.sh
```

**Claude Code in the cloud:** see [docs/CLOUD.md](docs/CLOUD.md) and [CLOUD_PROMPT.md](CLOUD_PROMPT.md).

Useful commands:
```bash
python -m repro.run list                        # all experiments
python -m repro.run plan --tier all             # the job graph and what is already done
python -m repro.run bank --exp bank.sem --seed 0   # run one job by hand
```

## Reading the report

`work/results/REPORT.md` gives one row per claim, with our paired-over-seeds number, the doc's number,
and **PASS / FAIL / PENDING**. Absolute CEs will not match the doc because the data differ (see
[docs/RECONSTRUCTION.md](docs/RECONSTRUCTION.md)). The reproduction target is the **signs and the
paired contrasts**: EDL − TOKEN-4, semantic − shuffled, semantic bank − generic bank, router − G.

Exposure-matched specialisation, the strict test the doc adopted after §21, is measured three ways:
- per-expert ablations of the base models, restricted to routed tokens and compared within matched bins
  (`matched_domain`);
- the forced-equal-exposure swap matrix of each bank (`swap_own.specificity`);
- the routed-specialist contrast of each router (`matched_interaction`).
# moe_specialists
