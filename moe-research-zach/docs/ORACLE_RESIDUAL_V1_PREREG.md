# ORACLE-RESIDUAL-V1 — Label-Free Semantic Residual Capacity Test: preregistration

Frozen 2026-11-20, before any scientific job. Prefix `orv1_`; artifacts `artifacts/oracle-residual-v1/`.

The sha256 of this file is stored in `docs/ORACLE_RESIDUAL_V1_PREREG.md.sha256` and checked by every scientific job. Code is listed in §12. No real domain label is used anywhere: not for training, assignment, selection, promotion or tuning. The test split is never opened or tokenized.

**This is an oracle diagnostic, not a deployable method.** Condition A uses a non-causal, paper-level assignment.

## 1. Question and prior evidence

**Question.** On top of a frozen TOKEN-4 model, can a *subject-specific* residual specialist lower held-out next-token CE more than equally persistent shuffled groups (B) and unrestricted token routing (C)? All three get identical capacity and compute.

**Primary endpoint.** Absolute held-out CE improvement, Delta_CE = CE_frozen_base − CE_candidate.

**Not success criteria.** Routing NMI, exposure, activation separation, ordinary ablation and parameter divergence.

**Prior evidence (reports read).**
- *Semantic EDL / long-K.* Semantic clusters carry subject information, but the gains come mostly from generic paper grouping, with SHUF reaching about 80 % of EDL.
- *SLC.* The Semantic Lift gain is routing exposure.
- *SCRGP.* Semantic-differential gradients are stable but zero-sum. 81 % of the effect sits in the final MoE layer.

The final-MoE-layer insertion point below follows from the SCRGP result.

## 2. Frozen base (reused; read-only; never copied)

TOKEN-4 245M checkpoints `ckpt_p100` (step 3880, 245,032,457 tokens, plan sha `f4bc3c93…`, seeds 0–3):

| seed | run |
|---|---|
| 0 | `runs/20261114T131547Z_forecast_latent_edl_v1_final_T4_s0` |
| 1 | `runs/20261114T145059Z_forecast_latent_edl_v1_final_T4_s1` |
| 2 | `runs/20261114T221957Z_forecast_latent_edl_v1_final_T4_s2` |
| 3 | `runs/20261116T023759Z_semantic_edl_long_k_P1_T4_s3` |

- Every job verifies the checkpoint sha256 against status.json.
- The full in-memory state hash is checked before and after every training and evaluation job.
- No existing parameter is updated: embeddings, attention, routers, routed experts, shared experts, norms and the tied head all have requires_grad = False.

## 3. Assignments (frozen; `orv1_tables.py` → `artifacts/oracle-residual-v1/tables/`, hashes in `tables.json`)

**A, SEMANTIC.**
- The hard argmax of the frozen GC-EDL SEM K = 16 prototype assignment:
  - main split: `artifacts/gc-edl-v1/latent/table_SEM.npz`, sha `308f1e02…`;
  - CALIB-SEARCH: `table_SEM_calib_search.npz`, sha `abdd94c5…`, from the same prototype network.
- The argmax is identical to that of the Semantic-EDL T0.5 table.
- Every segment of a paper gets the paper's label.

**B, SHUFFLED.**
- The hard semantic labels are permuted across papers *within 20 paper-token-mass quantile bins*.
- Seeds: 20261201 (main) and 20261202 (CALIB-SEARCH).
- This keeps papers per cluster exactly (verified) and token mass per cluster approximately. Cluster 14, for example: 15.1 % under A vs 13.0 % under B in the training stream.
- The older SHUF table was *not* used. It kept paper counts but not token mass (15 % vs 5 % for cluster 14), because semantic clusters correlate with paper length.
- Hard agreement with A: 0.074 (main), 0.108 (CALIB-SEARCH).

**Unassigned papers** (no semantic features) are excluded from training and evaluation for every condition:
- 690 unseen main documents;
- 1,118 CALIB-SEARCH tokens.

## 4. Architecture (identical across A, B, C)

**Insertion point.** The residual stream x_t after the final MoE block (block 11), before `norm_f` and the tied LM head:
- h′_t = x_t + a_{e(t)}(n(x_t))
- logits = W_emb · norm_f(h′_t)

**Input normalisation.** n(·) is a parameter-free RMS normalisation (ε 1e-6). It is part of the new bank; no existing parameter is reused for training.

**Bank.** K = 16 specialists, each with the routed-expert structure (W1 384×256, GELU-tanh, W2 256×384, no bias), using `csr.lfr_model.Experts` sorted dispatch.
- Exactly one specialist per token, with gate 1.
- 3,145,728 bank parameters per condition.
- Active compute per token is one specialist for every condition.

**Initialisation.**
- W1 ~ N(0, 0.02²) from the generator seed + 7,000,003, identical for A, B and C within a seed.
- W2 = 0, so every condition starts bit-identical to the frozen base. This is verified on CPU and GPU.

**Routing in each condition.**
- **A:** the token's paper semantic label. Non-causal, paper level: an oracle.
- **B:** the token's paper shuffled label.
- **C, TOKEN-ROUTED:**
  - A linear router n(x_t) → 16 logits, 6,144 parameters, reported separately. A and B get no dummy parameters.
  - Initialised N(0, 0.02²) from the generator seed + 7,000,004.
  - It selects the top-1 argmax, which is causal (it uses only x_t).
  - The output gate p_e / stopgrad(p_e) equals 1 in value and passes the router gradient.
  - Switch load-balance loss α · K · Σ_k f_k P̄_k over valid tokens, with α from §6.3. No semantic information is used.

## 5. Correctness tests (before any scientific job; all must pass)

**CPU** (`orv1_tests.py`, real TOKEN-4 s0 weights):
- T-ZERO: zero-init bit-exactness.
- T-GRAD: gradients only on the bank, plus the router in C; none on the base.
- T-ONE: exactly one active specialist; C gate == 1.
- T-LOAD: A/B paper-load histograms identical up to the permutation.
- T-SAME: identical batches across conditions; same plan across seeds.
- T-RESUME: 4 straight steps == 1+1+1+1 resumed legs, bitwise, covering banks, routers, optimizers, step, tokens and RNG.
- T-CONT: 2 steps → milestone → 4 == 4 straight.
- T-HASH: base hash unchanged.
- T-EVAL: assigned-path CE equals the replacement-matrix entry.

The CPU tests passed on 2026-11-20 (`cpu_tests_*.json`).

**GPU** (`orv1_gpu_tests.py`, G-ZERO, G-HASH, G-BENCH) passed. Benchmark: about 88k tokens/s joint, so 20M ≈ 4 min, 245M ≈ 46 min, and a full EVAL ≈ 4 min.

**G-RESUME (GPU).**
- Runs: two straight 3-step runs and one resumed run.
- Pass condition: either bitwise equality, or a max weight difference ≤ 2× the straight-vs-straight repeat difference + 1e-6.

## 6. Training (identical across A, B, C; one process, shared base forward)

### 6.1 Data

**Residual-training stream.** Main-split documents *not seen by the TOKEN-4 base plan* and with an assignment: 372,638 documents, 367.35M tokens.
- Ordered by sha256("orv1-order|paper|doc").
- Packed into 1024-token rows; 32 rows per step.
- Plan sha `fd1cd554…`, 7,748 steps = 245,032,808 tokens.
- Budgets are prefixes of this plan:

| budget | steps | tokens |
|---|---|---|
| pilot 5M | 159 | |
| 20M | 633 | 20,014,694 |
| 50M | 1,581 | 50,022,549 |
| 245,032,457 | 7,748 | |

- Identical examples, order, target masks, steps and token counts for A, B and C. The plan is the same across seeds; seeds differ in the base checkpoint and the specialist init.

### 6.2 Optimizer

- One separate AdamW per condition: β (0.9, 0.95), weight decay 0.1 on all bank and router weights, grad-clip 1.0 per condition.
- Loss = mean CE over valid tokens of the step (+ α·aux for C).
- Linear warmup over 50 steps, then **constant LR**. A run continued to a larger budget is therefore identical to a run trained straight to it.
- bf16 autocast for matmuls, fp32 master weights, TF32 off.

### 6.3 Pilot (the only hyperparameter selection; condition-agnostic; SEL only)

**LR selection.**
- Candidates LR ∈ {3e-4, 1e-3, 3e-3}; α = 0.01; seed 0; 5M tokens.
- Evaluate on SEL, which is CALIB-SEARCH paper-half A: 6,403,657 tokens, 503 papers.
- LR* = argmin over LR of mean(CE_A, CE_B, CE_C); ties go to the smaller LR.

**α ladder.**
- C collapse at LR* means SEL load max > 0.5 or effective number exp(H(load)) < 4.
- On collapse, rerun the pilot at LR* with α = 0.03, then 0.1. α* is the first value without collapse, or 0.1 if every value collapses.

**What the pilot does not touch.** It never sees EVAL. Nothing else is tuned, and A gets no extra tuning.

### 6.4 Checkpointing

- An atomic rolling slot every 8 min and at each leg end (legs ≤ 80 min).
- Permanent checkpoints at 20M and 50M (and at 245M if reached). Each holds bank and router weights, optimizer, step, tokens, RNG, plan, table and base hashes.
- The base is never copied.

## 7. Evaluation (`orv1_eval.py`)

**Partitions.**
- **EVAL** = CALIB-SEARCH paper-half B: 6,094,287 tokens, 498 papers. Paper-disjoint from SEL and never used for training or selection.
- **Stage 3** uses VAL (§10).

**References.** D = the frozen base on exactly the same tokens. The base forward is shared, so D is exact.

**Per token, stored:**
- CE_D, CE_A, CE_B, CE_C;
- C's routed specialist;
- semantic and shuffled labels, paper, position, document length, target;
- the replacement matrices dA and dB (N×16): CE with specialist k of bank A or B forced for that token, minus CE_D. These are recomputed exactly through norm_f and the head.

**Statistics.**
- Paper bootstrap over EVAL papers: 2,000 replicates, seed 20261203, shared across seeds and conditions.
- Pooled = mean over seeds of the per-seed statistic under the same replicate. 95 % percentile CI.

## 8. Promotion rules

### 8.1 Stage 1: 20M-token screen, seeds 0 and 1 (all of P1–P7 required)

| id | rule |
|---|---|
| P1 | Delta_CE_A > 0 in both seeds |
| P2 | CE_A < CE_B in both seeds |
| P3 | CE_A < CE_C in both seeds, **or** a near-tie (CE_A − CE_C ≤ 0.001 nats in both seeds) together with functional F1 and F2 (§8.4) |
| P4 | pooled paper-bootstrap 95 % CI of (CE_B − CE_A) has lower bound > 0 |
| P5 | ≥ 12 of 16 semantic clusters have pooled per-cluster Delta_CE_A ≥ 0 |
| P6 | not one cluster: contributions c_k = token share × (CE_B − CE_A)_k, pooled; total > 0; max_k c_k ≤ 50 % of Σ positive c_k; every leave-one-cluster-out total > 0 |
| P7 | no pathology, specialist collapse or router collapse in any seed: all CE finite; base hash unchanged; every A/B/C specialist has W2 norm > 0 and finite; C's EVAL load max ≤ 0.5 and effective number ≥ 4 |

**If P2 fails (A does not beat B), stop.** There is no tuning or retry.

### 8.2 Stage 2: 50M confirmation (only if Stage 1 promotes)

- Continue the same runs (constant LR) from 20M to 50M; same seeds 0 and 1; evaluate identically.
- **Promotion requires all of:**
  - P1–P7 hold at 50M;
  - persistence: pooled (CE_B − CE_A) at 50M > 0 and ≥ 0.5 × its 20M value;
  - no global-CE degradation: pooled Delta_CE_A(50M) ≥ pooled Delta_CE_A(20M) − 0.0005.
- **Compute and parameters.** A and B have identical parameters and compute by construction. A has 6,144 fewer parameters than C. So no A advantage can come from extra compute or parameters.

### 8.3 Stage 3: four-seed confirmation (only if Stage 2 promotes)

**Budget rule.** Fixed now and decided on SEL only:
- Compute the mean over A, B, C and seeds 0–1 of [CE_SEL(20M) − CE_SEL(50M)].
- If it is ≥ 0.001 nats, use 245,032,457 tokens. Seeds 0–1 continue from 50M; seeds 2–3 train from scratch.
- Otherwise use 50M. Seeds 2–3 train to 50M; seeds 0–1 reuse the Stage-2 checkpoints.
- The projected GPU time must fit the cap (§11).

**VAL.** Evaluated once, after everything is frozen (§10).

**Final success requires all of:**
1. Pooled VAL Delta_CE_A > 0 with hierarchical CI lower bound > 0.
2. A beats B in ≥ 3 of 4 seeds.
3. A beats C in ≥ 3 of 4 seeds, or a near-tie (≤ 0.001) in ≥ 3 of 4 seeds together with F1 and F2 on VAL.
4. Hierarchical seed × paper CI of (CE_B − CE_A) excludes 0: 10,000 replicates, seeds resampled with replacement plus a shared paper replicate.
5. P5 and P6 on VAL.
6. Base hashes unchanged.
7. Identical residual parameters and active compute: A = B, and C has the 6,144 router parameters reported.

### 8.4 Functional specialisation (all stages; replacement-based, exact downstream CE)

**Definitions.** For bank X ∈ {A, B} with its own labels ℓ_X, let d_X[t, j] = CE(specialist j forced on t) − CE_D(t).

- **Per-token gain of specialist k:** g_k(t) = mean_{j ∉ {k, ℓ_X(t)}} d_X[t, j] − d_X[t, k].
  - This is k against a random *other* specialist. The token's own specialist is excluded from the baseline for other-cluster tokens.
- **U_own(k):** the mean of g_k over own-cluster tokens.
- **U_oth(k):** the mean over other-cluster tokens, re-weighted to the own-cluster distribution over 675 strata:
  - target-frequency tercile (main-split unigram counts);
  - base-CE quintile;
  - position bin (< 128, < 512, ≥ 512);
  - document-length tercile;
  - target class (word start, word continuation, digit, whitespace, other; derived from the tokenizer vocabulary).
  - Only common-support strata are used, and the weights are fixed at full-sample values.
- **Specificity:** S_X(k) = U_own(k) − U_oth(k). S_X is the mean over the 16 clusters.

**Also reported:**
- U_best_alt(k) = (best other specialist's mean CE change on cluster k) − (own specialist's).
- U_shuffled_replacement(k): A's specialist k replaced by B's specialist k on cluster-k tokens.
- Per-specialist training exposure for A and B, as token shares. The shares are matched by construction to within about 2 points.
- The full 16×16 replacement matrix.

**Functional pass requires all of:**

| id | rule |
|---|---|
| F1 | pooled S_A > 0 with CI lower bound > 0 |
| F2 | pooled S_A − S_B > 0 with CI lower bound > 0 |
| F3 | U_own_A(k) > 0 for ≥ 12 of 16 clusters |
| F4 | S_A > 0 and S_A > S_B in every seed |

Routing counts are never used as evidence of function.

## 9. Interpretation (fixed now; all that apply are reported; primary by precedence A > C > E > B > D > MIXED)

| outcome | rule (pooled CIs at the last stage reached) |
|---|---|
| A — semantic residual functions exist | the stage's promotion rules pass **and** functional F1–F4 pass. At Stage 3, the final success rules. |
| C — token-level decomposition is better | CI of (CE_C − CE_A) has upper bound < 0, **and** CI of (CE_B − CE_C) has lower bound > 0 |
| E — semantic specialists help their clusters but hurt global CE | F1 holds **and** the CI of Delta_CE_A has upper bound < 0 |
| B — persistent partitioning helps, semantics do not | CIs of Delta_CE_A and Delta_CE_B both have lower bound > 0, **and** the CI of (CE_B − CE_A) has lower bound ≤ 0 |
| D — no residual-capacity benefit | no CI of Delta_CE_A, Delta_CE_B or Delta_CE_C has lower bound > 0 |
| MIXED (not promoted) | none of the above; reported with the exact failing rules |

No result is reframed after observation.

## 10. VAL (Stage 3 only)

- After Stage 2 promotes and the Stage-3 budget is fixed, compute VAL paper assignments with the *same frozen* GC-EDL SEM prototype network. This uses the identical feature pipeline: `gc_data.py` / `gc_fingerprint.py`, VAL split, label-free.
- Hard argmax; shuffled VAL labels are length-stratified with seed 20261204.
- VAL is evaluated once, with all four seeds, after all decisions are frozen.

## 11. Compute, disk, operations

**GPU caps (GPU-hours).** Total 20; Stage 0 + pilot 1.5; Stage 1 2; Stage 2 3; Stage 3 13. If a cap is exceeded the run stops cleanly with `STOPPED_GPU_CAP`.

**Disk.**
- Keep ≥ 100 GiB free, and before every job require 100 GiB + the job estimate.
- Estimated peak additional use: ≤ 25 GiB. EVAL npz ≈ 0.6 GB per seed per stage, VAL npz ≈ 2.4 GB per seed, checkpoints ≈ 0.12 GB.
- No /dev/shm. No pre-existing file is deleted.
- Protected inventory: 9,941 files, verified at the end.

**Interruption.**
- Treat it as GPU reclamation: stop launchers and watchers and do not restart. Mark `INCOMPLETE_STOPPED_BY_EXTERNAL_OPERATOR`, and record the last checkpoint and the exact resume command in RESUME.md.
- Do not touch a foreign GPU process.
- Work lost on interruption: ≤ 8 min.

**Logs.** decision_log.jsonl records every engineering issue *before* acting (observation, hypothesis, change, expected result, actual result, decision). Also kept: observations.jsonl, deviations.jsonl, run_manifest.tsv, compute_ledger.tsv, disk_ledger.tsv, RESUME.md.

## 12. Allowed implementation changes (engineering only)

**Allowed, logged before acting:**
- correctness bugs;
- memory or micro-batch size (the 32-row step is unchanged);
- kernels and efficiency;
- checkpointing, resume and logging.

**Not allowed:**
- changing assignments, the shuffle, the architecture or insertion point, the optimizer, the LR rule, budgets, seeds, partitions, metrics, thresholds or interpretation rules;
- giving any condition extra tokens, steps, parameters, tuning or seeds;
- dropping clusters or papers;
- using VAL or test for engineering.

**Code:**
- `src/csr/orv1_{paths,lib,log}.py`;
- `orv1_{tables,tests,gpu_tests,train,eval,analyze,run,driver,inventory}.py`.
