# csr-forecast-specialist-pretraining-v2: Operational Amendment 1

**Checkpoint cadence and external-cancellation / GPU-yield policy**

- Recorded: 2026-11-11, before resuming from STOPPED_BY_OPERATOR (after `s2_C11_s101`; search stage 2 in progress).
- Scope: this amendment is operational only. It does not change the scientific protocol
  (docs/FORECAST_SPECIALIST_PRETRAINING_V2_PROTOCOL.md, sha256
  e65065d8b3755c0a2b7e463449244bbcd4f4f69bd4219cd27f46ecc5b876a0d9). That file stays unchanged and its configs keep
  referring to its hash.
- It does not change the architecture, data, splits, seeds, conditions, search candidates, selection rules, budgets
  or fairness constraints.

## A. Checkpoint policy (operator instruction, 2026-11-11)

1. **Cadence.** Every training run writes a fully resumable checkpoint at least every 15 minutes or every 10% of
   its planned optimizer steps, whichever comes first.
   - In practice, every max(1, floor(0.10 × n_steps)) steps, or after 900 s of wall time since the last one.
   - It also writes one at every scientific milestone and on every interruption, at an optimizer-step boundary.
   - The same rule applies to every run of every condition, so the overhead is the same for all of them. Runs
     shorter than 15 minutes use the same rule rather than an exemption, also for uniformity.
2. **Milestones.** The scientific milestone checkpoints are unchanged: 40/50/75/100% for final runs and 100% for
   development runs. They are model-only, as before.
3. **Contents of a resumable checkpoint.**
   - Model weights and AdamW state.
   - Scheduler: the LR is a stateless function `lr_at(step)` of the step index. The checkpoint records the step
     and the next LR.
   - Gradient scaler: none exists. Training uses bf16 autocast, so no GradScaler is used; the field is recorded as
     `null`.
   - RNG states: CPU torch, all CUDA devices, numpy global and python `random`.
     - Teacher-forcing draws already come from a per-step seeded generator.
   - Dataloader cursor: the stream is a deterministic function of the config, so the cursor is the next step index
     plus `n_steps`, `rows_per_step` and the stream name.
   - Step and token counts, plus the accumulators.
   - Run identity and config SHA-256.
   - The driver state: SHA-256 of driver_state.json plus its `active` record (stage and run).
4. **Atomic write.**
   1. Write to a temporary file in the destination directory, then flush and fsync it.
   2. Compute its SHA-256 and verify it by reloading it (memory-mapped) and checking step and token counts.
   3. Rename it to the final path, then fsync the directory.
   4. Only after verification is it added to the run's `resume_index.json`, itself written atomically, and to
      `artifacts/forecast-specialist-pretraining-v2/checkpoint_ledger.tsv` (path, step, tokens, SHA-256, bytes,
      seconds).
   - Milestone checkpoints use the same atomic write and are also recorded in the ledger.
5. **Retention.**
   - Resumable checkpoints have step-numbered names, so none is ever overwritten.
   - The newest verified checkpoint is never deleted.
   - While a run is incomplete, at least the latest two verified resumable checkpoints are kept. Older ones are
     deleted only after a newer one has been verified.
   - Milestone checkpoints are never deleted.
   - After a run is COMPLETE (training, end-of-run evaluation and summary written), its resumable checkpoints are
     no longer needed for resumption. They are deleted, and each deletion is recorded in the ledger.
     - Reason: each one is about 1.25 GB. Keeping two per run for about 36 remaining runs (about 90 GB) would break
       the preregistered 80 GiB free-disk rule, since 142 GiB is free.
6. **Resume.**
   - The newest checkpoint in `resume_index.json` whose SHA-256 still verifies is loaded.
   - If it fails verification, the previous one is used.
   - The legacy single-slot file is used only if no indexed checkpoint exists.
7. **Testing before use.** The cadence must pass a checkpoint/resume equivalence test before the driver relies on
   it. The test (fsp_resume_check2.py) covers:
   - a cooperative SIGINT stop;
   - an abrupt SIGKILL, with fallback to the previous checkpoint after deliberately corrupting the newest one.
   - Each must match an uninterrupted run within 3 × the run-to-run noise (same rule as protocol section 11).
   - All 9 training conditions are smoke-tested again after the code change.

## B. External cancellation / GPU-yield policy (operator instruction, 2026-11-11)

Any cancellation or interruption not started by the experiment driver itself means another person or job needs the
GPU. This includes:
- SIGINT, SIGTERM or SIGHUP sent to the driver or the training process;
- the stop marker `artifacts/forecast-specialist-pretraining-v2/STOP`;
- the training process being killed from outside;
- a foreign process on GPU 0, whether before launch or during a run;
- loss of the GPU or preemption;
- an explicit request to stop.

**Response:**
1. Nothing is restarted automatically, and no foreign process is killed or competed with.
2. A training process that can still run its handler stops at the next optimizer-step boundary, writes a verified
   resumable checkpoint and exits with status EXTERNALLY_STOPPED (exit code 77).
   - If the stop comes during the end-of-run evaluation, the evaluation is abandoned. The final-step checkpoint
     already exists, and the evaluation reruns when the run is resumed.
3. If a process was killed too abruptly to checkpoint, every file is kept. The run is marked EXTERNALLY_STOPPED,
   and the latest previously verified checkpoint is the resume point.
4. A run is never marked COMPLETE unless training finished.
5. The driver then:
   1. records `external_stop` (time, reason, stage, run, run_dir, latest verified checkpoint, exact resume command)
      in driver_state.json and driver_log.txt;
   2. waits for every experiment-owned child process to exit;
   3. confirms that its own processes have released GPU 0;
   4. exits. It does not poll and does not resume because the GPU became free.
6. A foreign process found before a launch blocks the launch. Before this amendment the driver retried every 5
   minutes, up to 40 times; it now yields immediately and records the reason.
7. **Explicit authorization to resume.** The driver refuses to start while the STOP marker exists, or while an
   unresolved `operator_stop` or `external_stop` record exists, unless it is started with `--authorized-resume`.
   That flag is only used on an explicit user instruction. The resolved record is kept in `stop_history`.

**Internal software errors** (a Python exception with a traceback, OOM, or the internal leg time limit) are handled
as before:
- the leg time limit leads to a resumable relaunch;
- OOM halves the micro-batch;
- an exception gets one retry, then is recorded as FAILED.

Failures whose reason shows the GPU itself is unavailable (no CUDA device, device lost, ECC/Xid,
driver-initialization failure) are treated as external stops, not bugs.

## C. Fairness notes

- The checkpoint cadence is the same for all conditions.
- Step timings (`tokens_per_s`) leave out checkpoint time; checkpoint seconds are recorded separately.
- TOKEN-COMPUTE matching (protocol section 7) still uses the compute ledger as written. Both the proposed runs and
  the TOKEN-4 runs used for the tokens-per-second rate include the same per-run checkpoint overhead.
- The search runs completed before this amendment (search stage 1, and 6 of the 8 candidate runs in search stage 2)
  used the old 10-minute single slot. Checkpointing does not change the optimization trajectory, so their results
  are unaffected.
