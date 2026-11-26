# JOINT_SEMANTIC_RESIDUAL_PRETRAIN_V1 — preregistered protocol

Prefix `jsr_`. Artifacts: `artifacts/joint-semantic-residual-pretrain-v1/` (ART). Temporary state: `.tmp/joint-semantic-residual-pretrain-v1/`. Runs: `runs/<ts>_joint_semantic_residual_pretrain_v1_<name>`.

This file is written and hashed (`.sha256`, mode 0400) **before any candidate result exists**. Every config carries its sha. Changes after freezing are allowed only as logged deviations or as preregistered engineering generations (§12).

## 1. Question, claims, prior evidence

**Question.** Can label-free semantic grouping produce *functionally* subject-specialised residual experts during pretraining, while matching or improving the accuracy of TOKEN-4 and matched residual controls?

**Success requires all three claims:**
- **(C1) accuracy:** comparable or better held-out next-token CE;
- **(C2) routing:** specialist routing that is more subject-correlated than under shuffled grouping;
- **(C3) function:** greater *exposure-matched, per-invocation* usefulness of a specialist on its own semantic group than shuffled-trained or alternative specialists show.

Routing separation alone is never reported as functional specialisation.

**Prior evidence (read before writing this file):**
- ORACLE_RESIDUAL_V1, on a frozen base with a non-causal paper oracle: a semantic residual bank beat shuffled (+0.0024) and token-routed (+0.0013) banks, with S_A ≈ 0.0048 against S_B ≈ 0.
- Semantic EDL C6, Semantic Lift (slc) and SCU v2 changed routing exposure but not exposure-matched per-invocation usefulness. In slc, the conditional interaction E − F was −4.5e-5 (1/4 seeds).
- SCRGP-V1: semantic residual gradients were zero-sum at fixed capacity.

## 2. Reused frozen inputs (read-only; verified by sha256 in `jsr_tables.py` → `ART/tables/tables.json`)

**TOKEN-4 recipe.** The config of `runs/20261114T131547Z_forecast_latent_edl_v1_final_T4_s0`:
- `model` and `train` dicts unchanged; diag_every 100;
- budget 245,000,000, giving 3,880 steps and 245,032,457 valid tokens;
- `lfr_lib.main_stream`, plan sha `f4bc3c93…`;
- WSD schedule: lr 6e-4, warmup_frac 0.01, decay_frac 0.2, min_lr 6e-5;
- 64 rows per step (micro 32 × accumulation 2), AdamW (0.9, 0.95), wd 0.1, clip 1.0, aux 0.01, z 0.001;
- bf16 autocast, torch.compile.

**Reused TOKEN-4 checkpoints (A_reused).**

| seed | run |
|---|---|
| 0 | `runs/20261114T131547Z_…final_T4_s0` |
| 1 | `runs/20261114T145059Z_…final_T4_s1` |
| 2 | `runs/20261114T221957Z_…final_T4_s2` |
| 3 | `runs/20261116T023759Z_semantic_edl_long_k_P1_T4_s3` |

All use ckpt_p100.

**References** (not capacity-matched; evaluated with the same evaluator, base path only):
- explicit EDL and semantic C6, seeds 0–3 at p100;
- paths as in `src/csr/slc_paths.REUSED`.

**Semantic groups.** Hard argmax of the semantic-C6 table `artifacts/semantic-edl-v1/tables/table_SEM_T05.npz` (sha `5d23129b…`).
- It is identical (100.000% agreement) to `artifacts/oracle-residual-v1/tables/labels_main.npz` `sem` on all 621,257 labelled main docs, so the orv1 per-doc label file is used directly.
- Docs without a label (0.03% of base-seen tokens) have group −1 and never enter L_sd.

**Shuffled groups (C).** The persistent length-stratified paper permutation `labels_main.npz` `shuf`:
- seed 20261201, within 20 paper-token-mass quantile bins;
- same number of papers per group, approximately the same token mass.
- *The old C6 SHUF table is not used.* It preserves paper counts but not token mass (group 14 holds 5.0% of base-seen tokens vs 15.2% for semantic). Under the orv1 shuffle it is 12.8%.

**Evaluation partitions (frozen, from orv1):**

| partition | definition | used for |
|---|---|---|
| SEL | CALIB-SEARCH paper half A (`tables/parts.npz`); 6.40M tokens, 503 papers | Stage 1 selection |
| EVAL | CALIB-SEARCH paper half B; 6.09M tokens, 498 papers | Stage 2 |
| VAL | `val_assign/parts_val.npz`; 24,995,467 tokens, 2,015 papers | Stage 3 only, opened once |

- Labels on these partitions: `labels_cs.npz` and `labels_val.npz`, produced by the frozen SEM network.
- `token_feats.npz` supplies frequency terciles and target classes.
- **The test split is never opened.** Real M2D2 domain labels are used only for the descriptive secondary routing audit, never for training, selection or promotion.

## 3. Architecture (identical for every residual condition)

The TOKEN-4 base is unchanged. Let h_t = norm_f(…) be the post-norm final hidden state, and W the tied LM head.

```
q_t = softmax(R · rms0(sg(h_t)) / tau)          R: Linear(384→16, no bias)
e_t = argmax q_t                                 causal: depends only on x_<=t through the base
gate_t = q_t[e_t] / sg(q_t[e_t])                 value 1; router gradient = ∇log q_e (validated orv1 convention)
a_t = Expert_{e_t}(rms0(sg(h_t))) * gate_t       Experts(16, 384, 256), GELU-tanh, the repo's routed-expert module
z_t = sg(W) (sg(h_t) + a_t)  = sg(z_base_t) + sg(W) a_t
```

- rms0 is parameter-free RMS normalisation.
- **Initialisation:** W1 ~ N(0, 0.02) from generator seed + 7,000,003; W2 = 0, so z = z_base at step 0.
- **Router:** N(0, 0.02) from generator seed + 7,000,004.
- Every bank in a process gets the same initial tensors.
- **Parameters:** a routed bank has 3,145,728 expert + 6,144 router = 3,151,872 parameters.
- **Active cost per token:** one expert (2·2·384·256 = 393,216 FLOPs) plus the router (12,288 FLOPs). The LM head is unchanged in cost, because W(h + a) replaces W h.
- The router never receives labels, cluster ids, future tokens, drafts or document-level information.
- **E (oracle):** same bank and loss, but e_t = semantic group of the token's paper and gate 1. No router, so 3,145,728 parameters and one active expert.

## 4. Objective and gradient boundaries

Per optimizer step (64 rows):

1. **Base.** L_base = the TOKEN-4 loss (LM CE + 0.01·Switch + 0.001·z), micro-batched exactly as TOKEN-4. Only base parameters receive it.
2. **Residual phase.** It runs after the base backward and *before* the base optimizer step, on the detached h of all 64 rows of the step (W is the same pre-step W). For each bank X:
   - L_res = mean_t CE(z_t, y_t). It updates the bank's experts, plus the router through the gate.
   - L_bal = K Σ_e f_e · mean_t q_{t,e}, with f_e the top-1 share. Router only; α = 0.01.
   - L_sd (C and D only) = mean over pairs (a<b) of present groups of −log(JSD(R_a, R_b) + 1e-6).
     - R_k is the mean of q_t over the step's valid tokens of group k.
     - A group is present if it has ≥ 256 valid tokens in the step.
     - JSD is in nats. Router only.
   - **Router gradient:** g = ∇L_res + α∇L_bal + β_t∇L_sd.
     - β_t = r · EMA(‖∇_R L_res‖) / EMA(‖∇_R L_sd‖), with EMA ρ = 0.9 over steps, starting from the first step's values.
     - The norms are measured on the step's own gradients.
     - This holds the ratio ‖β∇L_sd‖ / ‖∇L_res‖ ≈ r. The condition is defined by r.
     - The realised ratio is also measured on CALIB-SEARCH SEL rows at each eval checkpoint.
     - B has β = 0 and E has no router.
   - **Optimiser:** AdamW per bank, betas (0.9, 0.95), eps 1e-8, wd 0.1 on all bank matrices, clip 1.0 per bank.
     - LR = λ_r · lr_at(step), the same WSD schedule as the base, with **λ_r = 1.0**.
     - Banks never add base-optimisation steps or tokens.
3. Base optimizer step; then each bank's optimizer step.

All residual conditions of one seed are trained **in one process on one shared base**. The base trajectory receives no gradient from any bank, so the base is a TOKEN-4 replicate (A_internal). Data order, token count, steps, initialisation and base are therefore identical across conditions by construction.

## 5. Conditions

| id | description | trained here |
|---|---|---|
| A | TOKEN-4. A_reused = existing checkpoints; A_internal = z_base of the shared base in each joint run | reused / implicit |
| B | NO-SEMANTIC residual: routed bank, β = 0 | yes |
| C | SHUFFLED-GROUP residual: routed bank, L_sd with shuffled groups, same r, τ, α as D | yes |
| D | SEMANTIC residual (proposed): routed bank, L_sd with semantic groups | yes |
| E | SEMANTIC ORACLE (diagnostic, non-deployable): fixed e = paper's semantic group | yes (no comparable joint oracle exists) |
| ref | explicit EDL, semantic C6 (different objectives and architecture, not matched) | reused |

## 6. Correctness tests (all must pass before any scientific job)

**CPU (`jsr_tests.py`), real LFRModel config:**

| id | test |
|---|---|
| T1 | L_base gives zero (None) gradient to every bank parameter |
| T2 | L_res gives zero gradient to every base parameter, including tok_emb/W and norm_f |
| T3 | L_sd gives non-zero gradient only to the router (experts zero) |
| T4 | L_bal gives non-zero gradient only to the router |
| T5 | 3 training steps with banks vs without: the base state, base optimizer state and losses are bitwise identical (removing the bank reproduces TOKEN-4). The no-bank path is the TOKEN-4 code path |
| T6 | resume: 4 straight steps vs 2 + resume + 2 → bitwise identical base, banks, optimizers and controller state |
| T7 | all banks have identical initial tensors (W1, W2; routers of B/C/D); one process means identical data, steps and tokens |
| T8 | JSD / L_sd: numerical check against a float64 reference; permutation invariance; gradient increases JSD |
| T9 | controller: the realised ‖β g_sd‖ / ‖g_res‖ on a step equals r·(EMA ratio) to 1e-5 |
| T10 | causality: perturbing tokens after position t leaves q_s and e_s unchanged for s ≤ t |
| T11 | the router never sees labels: permuting group labels leaves q unchanged (only L_sd changes) |
| T12 | E routes exactly by the paper label |

**GPU (`jsr_gpu_tests.py`):**
- **G-EQ:** step 0 lm/aux/z of a seed-0 run with banks is bit-identical to the T4 s0 reference metrics.
- **G-ISO:** 60-step runs (logged every 10 steps), banks vs no banks: mean |Δlm| ≤ 1.5 × the mean |Δlm| of two no-bank repeats + 0.002 nats (the repo's established GPU tolerance; kernels are non-deterministic, so bitwise equality is a CPU-only requirement, T5).
- **G-RESUME:** 30 + resume + 30 steps vs 60 straight: for lm and for each bank's L_res, the mean |Δ| over steps 30–59 must be ≤ 1.5 × the same quantity between two independent straight 60-step bank runs + 0.002 (GPU run-to-run reference; bitwise resume is CPU test T6).
- **G-BENCH:** throughput and peak memory.

## 7. Stages

All training runs use the full 245M-token schedule (n_steps 3,880). Shorter stages are exact prefixes, stopped at the step whose cumulative tokens first reach the budget.

### Stage 1 — 20M-token search (seed 100, SEL)

**One process with 8 banks:**
- D(r, τ) for r ∈ {0.005, 0.02, 0.05} × τ ∈ {1.0, 0.5};
- B(τ) for τ ∈ {1.0, 0.5};
- α = 0.01.

Evaluated on SEL (§8; swap matrices on all SEL tokens).

**Rejection** (any one rejects a D setting):
- R1: CE_D > CE_B(τ) + 0.002; or CE_D > CE_A_internal; or non-finite values.
- R2: collapse — max specialist load > 0.25, effective number of experts < 8, or ≥ 2 specialists with load < 1/(10K).
- R3: domination — the median realised training ratio over the last half of steps is > 2r.

**Selection.** Among survivors, maximise I_D^sem − I_B(τ)^sem (§9.1).
- Ties within 1e-4 go to the smaller r, then τ = 1.0.
- If the selected setting has ≥ 2 dead specialists in any bank, one Stage-1b run repeats the selected (r, τ) and B(τ) with α = 0.03, and α = 0.03 is adopted if R2 then passes.

**Stage 1 fails** if no setting survives → engineering generation (§12).

### Stage 2 — 50M confirmation (seeds 0 and 1, EVAL)

One process per seed with banks B, C, D, E at the selected (r, τ, α). The checkpoint is at the step where cumulative tokens first reach 50,000,000.

**Promotion requires all of:**
- P1: pooled CE_D − min(CE_B, CE_C) ≤ 0.005;
- P2: NMI(e; semantic group) of D > that of C, in both seeds;
- P3: I_D^sem − I_C^sem > 0, in both seeds;
- P4: no collapse or unused-specialist pathology in B, C or D (max load ≤ 0.25, effective number ≥ 8, ≤ 1 specialist with load < 1/(10K)), and everything finite.

Failure → engineering generation (§12).

### Stage 3 — 245M final (seeds 0–3, VAL opened once)

- Seeds 0 and 1 continue their Stage-2 processes; seeds 2 and 3 are new, with banks B, C, D, E.
- Milestones: 50M (full), 100M / 150M / 200M (bank-only snapshots), 245M (full).
- VAL is evaluated once, after all configurations are frozen. A_reused, EDL and C6 are evaluated on VAL with the same evaluator.

## 8. Evaluation (`jsr_eval.py`)

**Per token, on all tokens of the partition:**
- CE_base (A_internal);
- CE_X for every bank (routed path);
- top-1 e, top-1 q, top-1/top-2 margin;
- the token's semantic and shuffled labels, paper, position, document length, target id, real domain (descriptive).

**Swap-matrix subset** (all tokens for SEL and EVAL; for VAL, the 50% of papers with u_hash("jsr-val-sub", paper) < 0.5, fixed now):
- d_X[t, j] = CE(sg(h) + a_j(t)) − CE_base, for every specialist j (16, fp16);
- the full q_X[t, :] (fp16), for every bank.

**Also stored:**
- residual-correction norm per token;
- specialist mean-output cosine matrix;
- per-group residual-loss gradient cosine (bank parameters; ≤ 32k tokens per group).

## 9. Metrics

### 9.1 Primary functional metric — exposure-matched own-specialist advantage I_X^G

Here G is a grouping. The semantic grouping is used for every condition (primary); C is additionally scored on its shuffled groups (secondary).

**Usefulness of specialist e on token t, relative to the average alternative:**

g_e(t) = mean_{j≠e} d[t, j] − d[t, e]

which equals gain_own − mean gain_swap.

**Own specialist, cross-fitted.**
- Papers are split by u_hash("jsr-half", paper) into H1/H2.
- On H1, the own specialist of group k is σ(k) = argmax_e P(e_t = e | k) / P(e_t = e), the routing lift.
- σ is then measured on H2, and vice versa; the two directions are averaged.
- For E, σ(k) = k.

**Per group k with e = σ(k):**
- Own set O = H2 tokens of group k with e_t = e (the specialist was actually invoked).
- Other set N = H2 tokens of other groups with e_t = e.
- **Strata:** q_e quintile (per e, among invoked tokens) × top-1/top-2 margin tercile × base-CE quintile × position bin (< 128, < 512, ≥ 512) × doc-length tercile × target-frequency tercile, giving 2,025 strata.
  - For E, the q and margin factors are dropped.
- Common support only. The weights are the own-set stratum distribution.
- I_k = mean_O g_e − Σ_s w_s mean_{N∩s} g_e.
- A group is scored if it has ≥ 2,000 own and ≥ 2,000 matched other tokens in both directions.

**Aggregate.** I_X^G = mean over scored groups of I_k. Also reported:
- the number of groups with I_k > 0;
- split-half reliability (correlation of I_k across the two directions).

**Contrasts:** I_D^sem − I_C^sem and I_D^sem − I_B^sem (primary); I_E^sem (oracle); I_C^shuf (secondary).

### 9.2 Routing (C2)

- NMI(e_t; semantic group), on all partition tokens.
- Secondary routing audit:
  - NMI with real domain;
  - NMI with nuisance labels (target class, frequency tercile, position bin);
  - group-expert top-3 overlap;
  - utilisation and entropy;
  - correction norm and cosine;
  - gradient cosine between groups;
  - CE on the first 128 tokens;
  - the causal-router vs oracle gap CE_D − CE_E;
  - the per-invocation usefulness PIU_X = mean g_{e_t}(t), stratum-matched across conditions on (base-CE quintile × position bin × doc-length tercile × q decile × margin tercile).

### 9.3 Accuracy (C1)

- Per-seed CE and per-semantic-group CE.
- Paired per-paper bootstrap within a seed: 2,000 replicates, seed 20261205.
- Final contrasts use a hierarchical seed × paper bootstrap: 10,000 replicates, seed 20261206; seeds resampled with replacement, plus a shared paper replicate.
- Functional and NMI statistics use the same resampling. I uses the linearised (influence-function) per-paper contributions with fixed full-sample stratum weights; NMI is recomputed from per-paper count tensors.

## 10. Final success criteria (Stage 3, VAL, 4 seeds; all required)

| id | rule |
|---|---|
| S1 (C1) | hierarchical CI upper bounds: CE_D − CE_B < +0.005 and CE_D − CE_C < +0.005 (non-inferiority); CE_D − CE_A_internal < 0 |
| S2 (C2) | NMI_D − NMI_C > 0: CI lower bound > 0 and ≥ 3/4 seeds |
| S3 (C3) | I_D^sem − I_C^sem > 0 and I_D^sem − I_B^sem > 0: each with CI lower bound > 0 and ≥ 3/4 seeds |
| S4 | no pathology (P4 thresholds) in B, C, D; finite; integrity (identical steps, tokens, initialisation; base shared) |

**Accuracy improvement** is a separate claim: CE_D − CE_B and CE_D − CE_C both with CI upper bound < 0.

## 11. Interpretation (all that apply are reported)

Primary by precedence O1 > O6 > O2 > O4 > O3 > O5 > MIXED.

| id | rule | conclusion |
|---|---|---|
| O1 | success (§10) and accuracy improvement | label-free semantic organisation improves functional specialisation (and CE) |
| O1′ | success (§10) without accuracy improvement | functional specialisation with comparable CE |
| O6 | S3 holds and CE_D − min(CE_B, CE_C) has CI lower bound > 0 | real specialisation, but its cost exceeds its predictive benefit |
| O2 | S2 holds, S3 fails | repeat of Semantic Lift: exposure changed, function did not |
| O4 | E beats B (CE CI upper < 0, and I_E^sem − I_B^sem CI lower > 0) and D fails §10 | residual specialisation viable; causal semantic inference is the bottleneck |
| O3 | CE_D − CE_B and CE_C − CE_B CIs upper < 0, CE_D − CE_C CI contains 0, and I_D^sem − I_C^sem fails | learned causal residual routing helps; semantic clusters add nothing |
| O5 | neither D nor E beats B in CE (CI upper < 0 fails for both) | joint training failed to reproduce the oracle mechanism |

The primary outcome is O1/O1′ if §10 passes; otherwise the first applicable in the order above. No result is reframed after observation.

## 12. Bounded engineering (at most 3 generations; only after a Stage-1 or Stage-2 failure)

**Before running**, each generation is logged in DECISION_LOG with:
- the observed failure;
- a falsifiable hypothesis;
- the exact single change;
- the predicted diagnostic signature;
- the allowed interpretations.

**Allowed changes, one per generation:**
1. r / τ / α;
2. router capacity or normalisation (applied identically to B, C, D);
3. residual bottleneck width (identical for all banks, so parameter- and compute-matched).

**Fixed backbone:**
- independently trained TOKEN-4 base;
- detached residual objective;
- causal learned router;
- label-free L_sd;
- matched C and B.

A generation reruns Stage 1 on seed 100 for the changed factor (and Stage 2 if Stage 1 passes), under the same rules. The generations never use EVAL from a failed Stage 2 for selection, and never VAL or test.

## 13. Fairness and anti-cheating

- No extra tokens, steps, parameters or tuning for D. C and B always use the same r, τ, α and code.
- No seeds are dropped and no groups or papers are excluded post hoc.
- Assignments are frozen.
- The selection rules above are the only tuning.
- VAL is opened once; the test split is never opened.

## 14. Compute, disk, operations

**GPU cap: 25 GPU-h total.**

| item | GPU-h |
|---|---|
| tests | 1 |
| Stage 1 | 1.5 |
| Stage 2 | 2 |
| Stage 3 | 14 |
| evals and references | 4 |
| engineering reserve | 2.5 |

**Disk.**
- ≥ 100 GiB must stay free. Before each job, require free ≥ 100 GiB + that job's estimated peak.
- If the reserve would be crossed, stop cleanly and report the exact shortfall.
- Total new disk ≤ 30 GiB. The base is never copied per condition.
- Milestones: 50M and 245M full; 100/150/200M bank-only; one rolling resumable slot.
- Superseded slots of this experiment are deleted only after a newer one verifies. Every deletion is hashed and logged.
- No /dev/shm.

**GPU access.** Only via `scripts/gpu_guarded.sh` through `jsr_driver.py`. Each leg has a 90-minute limit; long runs resume in legs.

**External interruption** (SIGINT/SIGTERM, kill, GPU removed, foreign GPU process):
- checkpoint at the next step boundary;
- write STOPPED_BY_OPERATOR and the exact resume command to RESUME.md;
- stop all launchers and watchers of this experiment and release the GPU;
- no automatic restart, retry or polling.

**Ordinary code failures** are diagnosed and fixed, only if the fix is implementation-only, logged, and resumed from the checkpoint.

## 15. Allowed implementation changes (engineering only, logged)

Allowed:
- bug fixes that do not change the method;
- batching, chunking, precision of *storage*;
- logging;
- resumption;
- evaluation speed.

Anything that changes a metric definition, threshold, condition, or data is a deviation and is never applied to already-observed results.
