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
`GPU_TYPE`, `SLURM_ACCOUNT`, `MAIL_USER`, `REPRO_WORK` (where data, checkpoints and results go).

**GPU type.** `mit_normal_gpu` has one H100 node (4 GPUs) and eight H200 nodes (8 GPUs each), so the
default is `GPU_TYPE=h200,h100`: take whichever is free first. That becomes
`--gres=gpu:1 --constraint="h200|h100"` if `sinfo` lists those names as node features of the partition.
Otherwise `submit_all.sh` prints a note and requests the first type only (`-G h200:1`). It never sends an
untyped request, which would land on an L40S. Use `GPU_TYPE=h200` to force one type. Both are Hopper GPUs
(same bf16 kernels), so mixing them does not change the experiment.

The smoke test (`--smoke`) only checks that every stage runs on tiny data, so it uses `SMOKE_GPU_TYPE=l40s`
(49 nodes, far less contended) and a `SMOKE_TIME=00:30:00` limit per job. Short limits let SLURM backfill
the jobs into gaps, so the smoke test can start long before an H200 frees up.

## Quick start: one command, check back tomorrow

```bash
MAIL_USER=<you>@mit.edu ./slurm/launch.sh        # or: ./slurm/launch.sh --tier all   (adds the ablations)
```

`launch.sh` detaches itself (log: `$REPRO_WORK/logs/launch_<date>.log`), so you can log out right away.
It then:

1. runs `setup_login.sh` (step 1 below). If setup fails, nothing is submitted.
2. submits the smoke pipeline (step 2 below);
3. submits a small CPU **gate** job after the smoke report. It succeeds only if every smoke job finished;
4. submits the real pipeline with every job except `prepare` held behind the gate (`--after <gate>`).
   If the smoke test passes, the real run starts on its own overnight. If it fails, SLURM cancels the real
   jobs. Only the CPU-only `prepare` job (data tokenisation, needed either way) is not gated.

**Critical path first.** ORCD runs at most 2 GPUs per user at a time, and every bank and router (claims
2-3, ablations B4-B10) needs only the `token4` base models. So base models that no other job depends on
(claim 1's EDL family, B1-B3) are submitted with `--nice=$LEAF_BASE_NICE` (default 1000) and run after the
`token4` base and the cheap bank/router jobs. Set `LEAF_BASE_NICE=0` for equal priority.

**Submit limit.** ORCD caps how many jobs one user can have submitted at once
(`QOSMaxSubmitJobPerUserLimit`). The smoke test plus the real run can exceed that cap, and `--tier all`
certainly does. When `submit_all.sh` hits the cap, it queues what fits, says how many jobs are left and
exits with code 10. `launch.sh` then waits `RETRY_MINUTES` (default 10) and continues as jobs finish. The
launcher therefore keeps running on the login node (mostly asleep) until the last job is queued. If you
use `submit_all.sh` by hand, just re-run it later.

In the morning:

```bash
tail -20 work/logs/launch_*.log     # did setup and submission go through?
./slurm/status.sh --smoke           # smoke result; a failed gate shows up under "recent failures"
./slurm/status.sh                   # the real run
```

If the smoke test failed, fix the cause and run `./slurm/launch.sh` again. Setup is idempotent and finished
jobs are skipped. Delete `work/smoke` first if the fix should redo smoke jobs that had already finished,
and delete `work/data` if the fix touched data preparation. `launch.sh` skips the benchmark; jobs that run
past their time budget checkpoint and exit with code 3, and re-running `submit_all.sh` resumes them.

Steps 1-5 below are the same pipeline run by hand, one stage at a time.

## 1. One-time setup (login node, ~30 min)

```bash
./slurm/setup_login.sh
```

This creates the conda env `repro_moe` (torch CUDA build), downloads the five M2D2 arXiv archives
(~11.7 GB), the GPT-2 BPE and the bge-small / MiniLM encoders into `./cache`, checks that everything loads
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
