# Running on MIT ORCD (Engaging)

Everything runs as SLURM jobs from the login node. The scripts reuse the setup from your `subject-moe`
project: `mit_normal_gpu`, typed H100 requests, conda via `module load miniforge`, and offline compute nodes.

## 0. Get the code onto ORCD

```bash
ssh <kerberos>@orcd-login.mit.edu          # password + Duo
cd ~/orcd/pool                             # or scratch; ~60 GB needed in total (see "Disk")
git clone <your GitHub repo URL> moe-repro && cd moe-repro
# (or from your laptop:  rsync -av --exclude work --exclude .venv ./ <kerberos>@orcd-login.mit.edu:~/orcd/pool/moe-repro/)
```

Optional settings live in `slurm/config.sh` and can be overridden with environment variables:
`GPU_TYPE` (h100/h200), `SLURM_ACCOUNT`, `MAIL_USER`, `REPRO_WORK` (where data, checkpoints and
results go).

## 1. One-time setup (login node, ~30 min)

```bash
./slurm/setup_login.sh
```

This creates the conda env `repro_moe` (torch CUDA build), downloads the five M2D2 arXiv archives
(~11.7 GB), the GPT-2 BPE and the e5 / MiniLM encoders into `./cache`, checks that everything loads
**offline**, and runs the CPU tests.

## 2. Smoke test on a GPU node (~15 min)

```bash
./slurm/submit_all.sh --smoke
./slurm/status.sh --smoke            # until "total: N done, 1 todo" with no failures
cat work/smoke/results/REPORT.md
```

This runs the whole pipeline on tiny synthetic data, using exactly the code paths of the real run. The
smoke numbers mean nothing scientifically; you are only checking that every stage finishes.

## 3. Benchmark (~10 min)

```bash
./slurm/submit_all.sh --bench
cat work/logs/repro_bench_*.out | tail -3
```

This prints tokens/s and the projected hours per job and per tier. Every job must fit the 6 h limit.
If one does not, the job checkpoints at 340 min and exits with code 3; re-running `submit_all.sh`
resumes it.

## 4. The reproduction

```bash
./slurm/submit_all.sh                  # core: 37 jobs (prepare, cluster, 16 base runs, 12 banks, 6 routers, report)
./slurm/status.sh                      # progress, queue, recent failures
```

Order and dependencies are handled for you:

```
prepare (CPU) ─► cluster ─► base ×16 ─► bank ×12 (on token4 seed 0) ─► router ×6 ─► report
```

With 2 GPUs per user at a time, expect about one to two days of wall time for the core tier.

## 5. Ablations

```bash
./slurm/submit_all.sh --tier ablation  # B1–B10, see docs/ABLATIONS.md
```

## 6. Results

`work/results/REPORT.md` contains a PASS/FAIL table for each claim, alongside the doc's numbers. Each
run also writes `work/runs/<experiment>/s<seed>/eval.json` (all metrics) and `train_log.json` (loss
curves). To analyse in a cloud session, copy the small files back into the repo and push them:

```bash
mkdir -p results_orcd && rsync -av --include='*/' --include='*.json' --include='REPORT.md' --exclude='*' work/ results_orcd/
git add results_orcd && git commit -m "ORCD results" && git push
```

## Re-running and failures

* `submit_all.sh` is idempotent. It skips finished jobs (`DONE_*` markers), reuses queued ones, and
  resumes unfinished training from `ckpt_last.pt`.
* A failed job cancels everything that depends on it (`--kill-on-invalid-dep`). Fix the cause, then re-run
  `submit_all.sh`.
* Logs: `work/logs/<job name>_<job id>.out`; every submission is recorded in `work/logs/jobs.tsv`.
* Exit code 3 means the time budget was reached and a checkpoint was saved. Re-run `submit_all.sh`.
  Exit code 143 means preemption or SIGTERM, also with a checkpoint saved; `--requeue` restarts the job.

## Disk

| what | size |
|---|---|
| M2D2 archives (`cache/hf`) | ~11.7 GB |
| tokenized data | ~2 GB |
| base checkpoint (weights only) | 0.42 GB each; token4 also keeps 3 snapshots (B4) |
| training resume state (deleted when a run finishes) | ~1.3 GB per running job |
| banks / routers | small |

The core tier needs about 25 GB of outputs and the ablations add about 10 GB, so plan for 60 GB in total.
