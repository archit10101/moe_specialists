# JOINT_SEMANTIC_RESIDUAL_PRETRAIN_V1 — Report

**Terminal state: `TERMINAL_STAGE2_NOT_PROMOTED`.** No engineering generations were used (0 of 3). The work stopped at the 50M-token gate.

**Verdict:** the method improved **neither accuracy nor function** to the preregistered standard.

The Stage-2 gate failed on P2. The shuffled-group control routes by subject just as strongly as the semantic condition, so "more semantic routing" does not depend on the semantic labels.

Two smaller effects survive, but neither is enough for a positive claim:

- **D vs C (seed-consistent):** D is better than C by about 0.0004 nats in CE, and by about +0.0004 in exposure-matched own-specialist usefulness.
- **D vs B:** D is **CE-tied** with the no-semantic control B (seed signs −/+). Its functional advantage over B is mostly shared with the shuffled control C.

All numbers are 50M-token, EVAL-partition (CALIB-SEARCH half B) values. VAL and the locked test split were **never opened**. No 245M result exists.

- Protocol (frozen, sha256 `c4fa3ecc…5d463f`): `docs/JOINT_SEMANTIC_RESIDUAL_PRETRAIN_V1_PROTOCOL.md`
- Artifacts: `artifacts/joint-semantic-residual-pretrain-v1/`

---

## 1. Question

Can label-free semantic grouping produce functionally subject-specialised residual experts during pretraining, while matching TOKEN-4 accuracy? Success requires all three of:

- (i) CE comparable or better;
- (ii) more subject-correlated routing than with shuffled grouping;
- (iii) greater exposure-matched per-invocation usefulness of a specialist on its own group than for shuffled or no-semantic specialists.

## 2. Architecture and gradient boundaries

```
h       = final hidden (post norm_f) of a TOKEN-4 LFR base (d=384, 12 layers, 32 experts top-4; 103,609,728 params)
z_base  = LMHead(h)                                   # base trained by L_base only
q       = softmax(Router(sg(h)) / tau)                 # Linear 384->16, no bias
e       = argmax q                                     # causal, per token, no chunk persistence
a       = Expert[e](RMSNorm(sg(h))) * q_e / sg(q_e)    # 16 experts, 384->256->384 (W2 init 0)
delta_z = sg(W_emb) @ a ;  z = sg(z_base) + delta_z
L = L_base + lambda_r L_res + beta_t L_sd + alpha L_bal      (lambda_r = 1 as bank LR multiplier, alpha = 0.01)
L_sd = mean over present group pairs of -log(JSD(R_a, R_b) + 1e-6), R_g = mean q over group-g tokens in the step
       (groups with >= 256 tokens; router-only gradient; float64)
beta_t = r * EMA||g_res_router|| / EMA||g_sd_router||  (rho 0.9)
```

**Gradient boundaries:**

- The base never receives bank gradients (sg on h, z_base and W). The shared base is therefore an internal TOKEN-4 replicate (A_internal).
- L_sd and L_bal reach only the router.
- All banks of one seed train in one process on the same base and the same 64 rows per step. Each bank has its own optimizer (AdamW, wd 0.1, LR = λ_r · base WSD schedule).

**Gradient tests:**

- CPU tests T1–T12 all pass: isolation, router-only L_sd, permutation invariance, controller, resume, etc. (`cpu_tests_20261121T013540Z.json`).
- GPU checks all pass (`gpu_tests_20261121T014751Z.json`):
  - G-EQ: step-0 bitwise equal to the reused T4 s0.
  - G-ISO: banks do not perturb base training beyond no-bank repeat noise.
  - G-RESUME: resumed vs straight is within the straight-vs-straight run-to-run difference.
  - G-BENCH.

## 3. Conditions, seeds, tokens, parameters, FLOPs

| cond | description | trainable params | active FLOPs/token (bank) |
|---|---|---|---|
| A_internal | shared TOKEN-4 base (identical recipe to the reused TOKEN-4) | 103,609,728 | base |
| B | routed residual bank, β = 0 (no grouping) | 3,151,872 | 393,216 + 12,288 router |
| C | routed bank + L_sd on mass-matched **shuffled** paper groups (orv1 shuffle) | 3,151,872 | same |
| D | routed bank + L_sd on frozen **C6 semantic** groups (proposed) | 3,151,872 | same |
| E | oracle: e = paper's semantic group, gate 1 (non-causal; diagnostic only) | 3,145,728 | 393,216 |

**Fairness:**

- B, C and D share r, τ, α, code, data, rows and steps.
- No condition-specific parameters.
- No seeds were dropped.

| stage | seeds | tokens (steps) | partition | runs |
|---|---|---|---|---|
| Stage 1 search | 100 | 20,057,391 (318) | SEL | 1 process, 8 banks: B(τ = 1, 0.5) + D(r ∈ {0.005, 0.02, 0.05} × τ ∈ {1, 0.5}) |
| Stage 2 confirm | 0, 1 | 50,047,429 (793) | EVAL | 2 processes, banks B, C, D, E at r = 0.05, τ = 0.5, α = 0.01 |
| Stage 3 | — | not run (not promoted) | — | — |

## 4. Stage 1 (seed 100, 20M, SEL) — selection

The internal base CE is 4.67015. Every bank improves on it by about 0.0165–0.0172.

| bank | CE | I_sem | NMI(e;sem) | train ratio |
|---|---|---|---|---|
| B τ1.0 | 4.65289 | +0.00192 | 0.026 | 0 |
| B τ0.5 | 4.65239 | −0.00030 | 0.027 | 0 |
| D r0.005 τ1.0 | 4.65291 | +0.00164 | 0.150 | 0.0051 |
| D r0.02 τ1.0 | 4.65299 | +0.00169 | 0.148 | 0.0221 |
| D r0.05 τ1.0 | 4.65323 | +0.00170 | 0.146 | 0.0562 |
| D r0.005 τ0.5 | 4.65326 | +0.00143 | 0.143 | 0.0051 |
| D r0.02 τ0.5 | 4.65366 | +0.00146 | 0.147 | 0.0215 |
| D r0.05 τ0.5 | 4.65365 | +0.00163 | 0.147 | 0.0532 |

**Selection:** D_r0.05_t0.5, chosen by the maximum of I_D − I_B(τ): +0.0019, CI [+0.0012, +0.0027].

**Caveat logged at the time:**

- The win came from a low B_t0.5.
- At τ = 1, every D was about 0.0002 *below* its B.
- D was slightly worse than B in CE at every setting.

No pathology was seen, so α stayed at 0.01.

Note that NMI_D is flat (0.143–0.150) across r and τ. This flatness matters for the engineering decision (§9).

## 5. Stage 2 (seeds 0, 1; 50M; EVAL) — CE

Absolute CE per seed:

| seed | base (A_int) | B | C | D | E |
|---|---|---|---|---|---|
| 0 | 3.75514 | 3.74500 | 3.74498 | **3.74455** | 3.74485 |
| 1 | 3.76320 | **3.75192** | 3.75289 | 3.75249 | 3.75263 |

Contrasts give the mean, then per-seed values. The CIs come from a paper bootstrap (2,000 replicates; pooled over seeds and per seed):

| contrast | mean | seed 0 | seed 1 | pooled 95% CI |
|---|---|---|---|---|
| D − base | −0.01065 | −0.01059 | −0.01071 | [−0.01122, −0.01013] |
| B − base | −0.01071 | −0.01015 | −0.01128 | [−0.01120, −0.01023] |
| D − B | +0.00006 | **−0.00044** | **+0.00057** | [−0.00013, +0.00026] |
| D − C | −0.00042 | −0.00043 | −0.00040 | [−0.00053, −0.00031] (both seed CIs < 0) |
| C − B | +0.00048 | −0.00002 | +0.00098 | [+0.00030, +0.00065] |
| E − B | +0.00029 | −0.00014 | +0.00071 | [−0.00031, +0.00089] |
| D − E (causal − oracle) | −0.00022 | −0.00030 | −0.00014 | [−0.00077, +0.00034] |

**Reading:**

- All residual banks give about −0.011 nats against the TOKEN-4 replicate. The bank adds about 3% parameters, so this is not an iso-parameter gain.
- D is tied with B, with seed signs disagreeing.
- D beats C consistently but by very little.
- The **oracle E does not beat B** at 50M.
- The causal-vs-oracle CE gap is about 0, within noise.

## 6. Functional results (primary)

**Metric:** exposure-matched own-vs-other usefulness I_X^sem.

- The own specialist σ(k) is fitted on one paper half and measured on the other; the procedure is cross-fitted.
- g = mean CE change of the other specialists minus that of e, with experts forced on the swap subset.
- Own-group tokens are compared with other-group tokens routed to the same specialist.
- Matching is within 2,025 strata: q_e × margin × base CE × position × doclen × frequency.
- CIs come from per-paper influence functions.

| quantity | seed 0 | seed 1 | pooled CI | positive specialists | split-half r |
|---|---|---|---|---|---|
| I_D^sem | +0.00122 | +0.00135 | [+0.00096, +0.00160] | 12/16, 11/16 | 0.51, 0.39 |
| I_C^sem | +0.00079 | +0.00098 | [+0.00052, +0.00125] | 11/16, 11/16 | 0.17, 0.60 |
| I_B^sem | −0.00003 | +0.00069 | [−0.00008, +0.00074] | 9/16, 8/16 | 0.49, 0.35 |
| **I_D − I_C** | +0.00043 | +0.00037 | [+0.00002, +0.00076] (seed CIs include 0) | | |
| **I_D − I_B** | +0.00125 | +0.00066 | [+0.00044, +0.00147] (seed-1 CI includes 0) | | |
| I_C − I_B | +0.00082 | +0.00029 | [+0.00005, +0.00103] | | |
| I_C^shuf (C on its own groups) | +0.00007 | −0.00075 | [−0.00084, +0.00017] | | |
| I_E^sem (oracle, supplementary †) | +0.01211 | +0.01272 | [+0.01179, +0.01306] | 16/16 in swap matrix | |

**† Oracle metric:** the frozen analyser returned NaN for E. Because e equals the label, the routed "other-group, same specialist" set is empty by construction.

- The fixed oracle branch measures E over all swap-subset tokens (135 strata).
- B, C and D values were reproduced bit-identically.
- It is reported separately (`stage2_oracle_supp.json`) and is descriptive, since E uses a different population than the routed banks. See the deviations section.

**Secondary — per-invocation usefulness (PIU):** exposure-matched PIU is *lower* for D than for B (−0.020, −0.023) and than for C (−0.0001, −0.006).

- **Caveat:** D's sharpened router changes q calibration, so cross-bank q-matching is not like-for-like.

**Reading:**

- The grouping pressure itself (C) supplies about 60% of D's functional advantage over B.
- The semantic increment D − C is about +0.0004, with the same sign in both seeds, but neither seed is individually significant.
- The oracle shows that strong subject-specific residual function exists: about 10× D. Yet it gives no CE gain over token-routed B at 50M.

## 7. Routing (secondary evidence; not evidence of function)

| | B | C | D | E |
|---|---|---|---|---|
| NMI(e; semantic group) | 0.024 / 0.028 | 0.216 / 0.224 | 0.223 / 0.220 | 1.0 |
| NMI(e; real domain) | 0.025 / 0.028 | 0.225 / 0.229 | 0.228 / 0.228 | 0.610 |
| NMI(e; shuffled group) | 0.004 / 0.005 | 0.043 / 0.042 | 0.040 / 0.039 | 0.176 |
| NMI(e; target class) | 0.138 / 0.166 | 0.094 / 0.091 | 0.097 / 0.101 | 0.002 |
| utilisation entropy (nats) | 2.76 | 2.73–2.74 | 2.74–2.75 | 2.63 |
| load_max / load_eff / dead | 0.08 / 15.8 / 0 | 0.09–0.10 / 15.3–15.4 / 0 | 0.08–0.09 / 15.5–15.6 / 0 | 0.19 / 13.8 / 0 |
| training per-batch JSD (steps 600–790) | — | 0.212 / 0.220 | 0.301 / 0.310 | — |
| training router entropy | 1.63 / 1.52 | 0.65 / 0.64 | 0.73 / 0.68 | — |

**NMI_D − NMI_C:** +0.0069 / −0.0035, pooled CI [−0.0013, +0.0049]. **P2 fails.**

**Mechanism (the main finding):**

1. L_sd is computed per optimiser step, over groups that contain only a few documents.
2. So it can be satisfied by **document-discriminative routing for any label assignment**. C reaches per-batch JSD 0.21 among *random* groups, yet barely separates those groups at corpus level (NMI 0.042).
3. The dominant between-document axis of h is subject. Any group-separation pressure therefore yields subject-correlated routing: C reaches 0.23 NMI with real domains, the same as D.
4. As a result, routing semanticity in this design does not show that the semantic labels did anything. Both grouped banks move routing away from target-class or frequency features (NMI with target class drops from 0.15 to 0.09–0.10).

**Other secondary diagnostics (seed 0 / 1):**

- First-128-token CE gains are similar across banks (about 0.0061–0.0072 nats vs base). E gains about 0.0137.
- Off-diagonal gradient cosine between specialists: B 0.05, C 0.034, D 0.032.
- Correction cosine: B 0.19–0.22, C and D about 0.18.
- Nuisance probes are near 0 for all banks: NMI with position < 0.001, and NMI with frequency 0.02–0.05.

## 8. Mapping to the interpretation rules

The preregistered final outcomes O1–O5 are defined on Stage 3 (hierarchical CIs, 4 seeds) and **cannot be assigned**. Descriptively, at 50M:

| user rule | holds? |
|---|---|
| D beats C and B in CE and usefulness → evidence | **No.** D beats C in both; it ties B in CE. |
| higher NMI but no usefulness → Semantic Lift repeat | **No.** NMI_D is not higher than NMI_C; usefulness is weakly higher. |
| D and C both beat B, tied → causal routing helps, semantics add nothing | **Partly.** In usefulness, C > B and D > B with a small D − C gap. In CE, neither beats B. |
| E works but D does not → router bottleneck | **Partly.** E has 10× functional specialisation but no CE gain over B at 50M. |
| neither D nor E beats B → joint training failed | **Yes, in CE** at 50M: both CIs include 0. |
| D improves usefulness but worsens CE → cost > benefit | **No.** CE is tied. |

**Summary:** accuracy is not improved (D ≈ B). Function improves weakly over B, but mostly through a label-agnostic grouping effect. The semantic-specific increment is small and individually non-significant. Routing became semantic in the shuffled control too. Per the protocol, **this is not a positive result**.

## 9. Engineering generations

**None run (0 of 3).** The decision is in DECISION_LOG.

**Why none:**

- §12 requires a falsifiable hypothesis that one allowed change fixes the observed failure. The failure is NMI_C ≈ NMI_D.
- r and τ: Stage 1 shows NMI insensitive to them (0.143–0.150 over all six settings), and r → 0 reverts to B.
- Router capacity and normalisation act identically on C and D, because both exploit the same dominant subject axis.
- Bottleneck width does not affect routing.
- The cause sits in the fixed backbone, the per-step label-free L_sd, which §12 forbids changing.
- Running generations anyway would be searching for a positive result.

**What would falsify this decision:** an allowed change giving NMI_D − NMI_C > 0 in both seeds.

## 10. Most defensible next experiment

Make L_sd **unsatisfiable by per-batch document contrast**:

- Compute the separation between each group's current-batch routing and slow cross-step EMA centroids of all groups (or a contrastive form against stored centroids).
- Random groups have identical corpus-level centroids, so C should then revert to B-like routing. Only label-consistent groups can be separated.

**Prediction:** NMI_C falls toward about 0.03, while NMI_D stays well above it. A matched Stage-2 test of I_D − I_C would then isolate the semantic contribution.

Separately, the oracle result (I_E about 0.012, with no CE gain at 50M) suggests checking whether residual specialisation converts to CE only late in training, as in ORV1 at 245M. To do that cheaply, run E vs B at 245M on the existing base without the L_sd machinery.

## 11. Compute, disk

| item | GPU-h |
|---|---|
| tests (GPU checks, benches) | 0.155 |
| Stage 1 (train 0.153 + SEL eval 0.326) | 0.479 |
| Stage 2 (2 × train 0.246 + 2 × EVAL eval 0.161) | 0.813 |
| **total** | **1.447** (cap 25) |

**Throughput:**

- 4-bank step: about 1.03–1.11 s (15 min per 50M); 8-bank step about 1.6 s.
- Peak memory about 17.1 GiB.

**Disk:**

- Free is about 122 GiB (reserve 100 GiB).
- The experiment uses about 12 GB: evals 7.0 GB, resumable slots 4.2 GB, run directories 0.8 GB.
- The test slots and checkpoints of this experiment were deleted, with hashes logged in `deletions.tsv`.
- No previous-experiment artifacts were modified.
- `/dev/shm` was not used.

## 12. Deviations and integrity (`deviations.jsonl`)

**Post-freeze:**

- The oracle-branch fix to `jsr_analyze.functional` for the I_E diagnostic. It cannot affect the conclusion, and B, C and D are reproduced bit-identically.

**Pre-freeze implementation refinements (logged for completeness):**

- float64 L_sd accumulation (T8);
- dense metric logging;
- G-RESUME reference redefinition;
- G-ISO repeat choice.

**Design choice in the protocol:**

- C uses the mass-matched orv1 shuffle, not the C6 SHUF table, which distorts token mass.

**Not run, because Stage 2 failed:**

- Stage 3, VAL, the reference VAL evals and S1–S4.
- So the TOKEN-4 comparison exists only through the internal identical-recipe base on EVAL.

**Integrity:**

- The test split was never opened.
- VAL was never opened.
- No seeds were dropped.
- No retuning was done after Stage 2.
- Runs are not GPU-deterministic (known; the run-to-run difference was measured in G-RESUME).

## 13. Strongest evidence

1. **Routing semanticity is label-agnostic under per-batch L_sd.** The shuffled control reaches the same subject NMI as the semantic one (0.225–0.229 with real domains; both seeds), while failing to separate its own groups at corpus level (0.042).
2. **The semantic increment over shuffled grouping is small but sign-consistent:** CE −0.0004 (both seed CIs < 0) and usefulness +0.0004 (pooled CI just above 0).
3. **Oracle subject specialisation is functionally strong (I_E about 0.012) yet gives no CE gain over token-routed B at 50M.** The bottleneck to accuracy is not only causal inference of the subject.

## 14. Paths

- **Protocol:** `docs/JOINT_SEMANTIC_RESIDUAL_PRETRAIN_V1_PROTOCOL.md` (+ `.sha256`)
- **Artifacts:** `artifacts/joint-semantic-residual-pretrain-v1/`:
  - `stage1.json`, `stage1_decision.json`, `stage2.json`, `stage2_oracle_supp.json`
  - `final_analysis.json`, `DECISION_LOG.jsonl`, `OBSERVATIONS.md`, `deviations.jsonl`
  - `run_manifest.tsv`, `compute_ledger.tsv`, `disk_ledger.tsv`, `deletions.tsv`
  - `RESUME.md`, `STATE.json`, `evals/`
- **Runs:**
  - `runs/20261121T014957Z_…_S1_a0.01_s100`
  - `runs/20261121T022334Z_…_F_r0.05_t0.5_a0.01_s0`
  - `runs/20261121T024755Z_…_F_r0.05_t0.5_a0.01_s1` (ckpt_m50 retained)
- **Code (jsr_ namespace):**
  - `jsr_train.py`, `jsr_eval.py`, `jsr_analyze.py`, `jsr_oracle_supp.py`, `jsr_run.py`, `jsr_driver.py`, `jsr_configs.py`, `jsr_tables.py`, `jsr_tests.py`, `jsr_gpu_check.py`
  - `src/csr/jsr_{paths,bank,ckpt,log}.py`
