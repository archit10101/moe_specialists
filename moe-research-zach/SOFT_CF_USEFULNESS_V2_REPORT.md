# csr-soft-cf-usefulness-v2 — final report

**Status:** TERMINAL (2026-11-17 23:50 EST). The promoted candidate failed the preregistered 50M promotion rule, so
by rule no 245M runs were trained.

**Preregistered conclusion: inconclusive due to estimator noise** (diagnosis A, measurement). Factually, **no
replicated benefit** was observed: semantic and shuffled groupings under the identical soft counterfactual-usefulness
loss did not differ reliably in functional specialization, on dev or on held-out documents.

**Sources.**

- Preregistration: `docs/SOFT_CF_USEFULNESS_V2_PREREG.md` (sha256 bd52f79d…), frozen before any V2 GPU job.
- Decisions: `artifacts/soft-cf-usefulness-v2/{gamma,search_decision,promotion}.json`, all 0400.
- Machine-readable results: `results.json`. Run list: `run_manifest.tsv`.
- Mechanism tables: `tables.md`. Mechanism snapshots: `runs/*soft_cf*/mech/`.
- Descriptive VAL pass: `post_val/`.
- Predecessor: `SEMANTIC_UTILITY_EDL_V1_REPORT.md`.

## 0. Method

- **Backbone and data.** Identical backbone and data to V1 and the reused baselines: the 103.6M LFR model,
  unrestricted dropless per-token top-4 over 32 experts, the unchanged shared expert, and the FINAL plan.
  - No fixed expert is assigned to a group.
  - No inference-time model, label, draft or oracle.
  - Checkpoints hold only the LFR state dict, and inference never opens a table, bank or slot (CPU test tt4).
  - With γ = 0 the trainer is bit-identical to TOKEN-4 (tt1).
- **Groups.** Label-free semantic paper clusters (K16 C6). The control is the persistent shuffled permutation of the
  same table. Real domain labels were used only in the post-terminal descriptive VAL pass.
- **Usefulness measurement.** Exact top-4 replacement on held-out main-split documents, the measurement fold:
  - u = CE_replace − CE_base;
  - remove the selected expert, admit the next-ranked one, recompute downstream;
  - 16 rows × 8 passes per step;
  - ΔU(k,l,e) = own-group minus difficulty-matched other-group usefulness, with per-cell SE.
- **Confidence score.**
  - s = (ΔU − SE)/scale_l if positive, s = (ΔU + SE)/scale_l if negative, else 0.
  - Unsupported cells get s = 0.
- **Soft target.** q(k,l,·) = normalise(R(k,l,·) · e^{s/τ}), where R is the lagged EMA of the group's own routing
  ("tilt"). Alternatives: the literal uniform-base q = softmax(s/τ), and a mean estimator without the confidence
  penalty.
- **Objective.** L = LM + 0.01 Switch + 0.001 z + γ · mean KL(stopgrad q ‖ group-mean router softmax), with no
  Semantic-EDL term.
- **Lagging.** Targets are lagged and detached, refreshed every 10 steps from earlier observations.
- **Folds.** Training docs, measurement docs, dev confirmation (CALIB-SEARCH half B), held-out confirmation
  (CALIB-SELECTION half 1a) and VAL are disjoint.

## 1. Preregistered results (measured facts)

### 1.1 γ anchor

On V1 REF_C6_s100 at 20M, with a 400-step bank: ‖∇L_use‖ = 1.32 and ‖∇L_CF‖ = 0.70. That gives
**γ_mid = 0.006** (raw 0.0057), so the grid is γ ∈ {0.002, 0.006, 0.018}.

The anchor bank shows what the z = 1 confidence bound passes:

| anchor bank | split-half r | Spearman-Brown | positive cells | negative cells | KL(q‖R) |
|---|---|---|---|---|---|
| SEM | 0.12 | 0.21 | 18% | 27% | 0.22 |
| SHUF | −0.02 | — | 17% | 20% | 0.12 |

Under pure noise, about 32% of cells are expected to pass a 1-SE bound. The shuffled bank's 37% matches that, so its
targets are essentially noise. The semantic bank carries additional, partly reliable signal.

### 1.2 Search (20M, seed 100, semantic grouping; dev = CALIB-SEARCH half B; reference TOKEN-4 s100 at 20M)

| cand | design | dev CE | ΔCE vs TOKEN-4 | interaction [95% CI] | Δ vs TOKEN-4 | specialised (frac of 352) | pseudo MI / NMI | e5 NMI | load min/max | entropy | KL(q‖R) at end | bank SB | eligible |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TOKEN-4 | reference | 4.6263 | — | 1.18e-4 [0.45, 1.94] | — | 32 (0.091) | 0.037 / 0.012 | 0.018 | | 0.918 | | | |
| C1 | tilt, lcb, γ 0.006, τ 1 | 4.6260 | −0.0003 | 1.54e-4 [1.05, 3.08] | +3.6e-5 | 34 (0.097) | 0.039 / 0.013 | 0.019 | 0.49/1.77 | 0.919 | 0.10 | −0.14 | yes |
| C2 | γ 0.002 | 4.6252 | −0.0011 | 1.05e-4 [0.47, 3.23] | −1.3e-5 | 30 (0.085) | 0.038 / 0.013 | 0.019 | 0.45/1.65 | 0.917 | 0.10 | 0.08 | no |
| C3 | γ 0.018 | 4.6303 | +0.0040 | 1.32e-4 [0.80, 1.87] | +1.4e-5 | 25 (0.071) | 0.040 / 0.013 | 0.019 | 0.57/1.65 | 0.921 | 0.12 | 0.17 | yes |
| **C4** | τ 0.5 | 4.6234 | −0.0029 | 1.91e-4 [0.13, 2.56] | **+7.2e-5** | 23 (0.065) | 0.041 / 0.014 | 0.020 | 0.55/1.63 | 0.918 | 0.57 | −0.04 | **yes (top)** |
| C5 | mean estimator | 4.6356 | +0.0093 | 1.37e-4 [−1.03, 2.44] | +1.9e-5 | 30 (0.085) | 0.042 / 0.014 | 0.021 | 0.55/1.48 | 0.918 | 0.48 | 0.04 | yes |
| C6 | uniform base (literal) | 4.6291 | +0.0029 | 0.92e-4 [0.31, 1.70] | −2.7e-5 | 24 (0.068) | 0.040 / 0.013 | 0.020 | 0.54/1.63 | 0.919 | 0.10 | 0.01 | no |

- **Activation.** Every candidate's loss was active from step 100 (6.37M tokens), so it influenced about **13.7M of
  20M tokens**. V1 managed about 7.4M.
- **Promoted:** C4.
- **The 20M screen is weak.** All Δs are within the CIs.

### 1.3 Promotion (C4, 50M, seeds 100/101, SEM vs SHUF identical except the table)

| seed | run | dev CE | interaction [95% CI] | spec | pseudo MI / NMI | e5 NMI | entropy | hold50 CE | hold50 interaction [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| 100 | SEM-V2 | 3.7597 | 2.16e-4 [1.18, 3.12] | 30 | 0.040 / 0.013 | 0.020 | 0.809 | 3.6607 | 3.13e-4 [2.02, 3.81] |
| 100 | SHUF-V2 | 3.7669 | 2.57e-4 [1.67, 4.72] | 33 | 0.035 / 0.012 | 0.017 | 0.812 | 3.6680 | 3.29e-4 [2.36, 3.95] |
| 101 | SEM-V2 | 3.7501 | 2.68e-4 [1.86, 3.44] | 37 | 0.042 / 0.014 | 0.021 | 0.794 | 3.6497 | 3.21e-4 [2.32, 4.09] |
| 101 | SHUF-V2 | 3.7461 | 2.21e-4 [1.58, 3.78] | 32 | 0.034 / 0.011 | 0.017 | 0.799 | 3.6477 | 2.79e-4 [2.00, 3.45] |

Reused references, evaluated with the identical evaluator and rows:

| reference (50M) | s100 CE | s100 interaction | s101 CE | s101 interaction |
|---|---|---|---|---|
| TOKEN-4 | 3.7609 | 2.94e-4 | 3.7470 | 1.59e-4 |
| explicit EDL | 3.7504 | 3.55e-4 | 3.7524 | 3.48e-4 |
| Semantic-EDL C6 | 3.7436 | 3.29e-4 | 3.7346 | 3.12e-4 |
| SHUF-C6 | 3.7386 | 3.13e-4 | 3.7360 | 3.86e-4 |

Promotion criteria (all required):

| criterion | value | pass |
|---|---|---|
| 1. CE vs TOKEN-4 (mean ≤ +0.010) | −0.0012 / +0.0032 | ✅ |
| 2. dev SEM > SHUF (mean) | +2e-6 | ✅ (barely) |
| 3. same sign both seeds | **−4.2e-5 / +4.6e-5** | ❌ |
| 4. hold50 pooled paired 95% CI lower > 0 | Δ −1.7e-5 / +4.1e-5; CI **[−5.0e-5, +6.4e-5]** | ❌ |
| 5. no pathology | min load ≥ 0.57×, max ≤ 1.61×, entropy ≥ 0.79 | ✅ |
| 6. bank reliability SB ≥ 0.3 | **0.23 / 0.19** | ❌ |

**Diagnosis** (first matching link): **A, measurement.** SB < 0.3 in both seeds. No 245M runs, so no compute-matched
TOKEN-4 was needed.

### 1.4 Usefulness timing, overhead and resources

- **Activation.** At 50M the loss was active from step 100, influencing **43.7M of 50.05M tokens**.
- **Overhead.** Probe overhead was 0.84–0.87 of base step time. Training throughput was about 65k tok/s, against
  about 123k tok/s without probing.
- **Memory.** Peak allocated VRAM was 17.0 GiB.

## 2. Exploratory / descriptive results (never used for any decision)

**VAL routing pass after all decisions were frozen** (`scu_post_val.py`; 50M checkpoints; semantic-edl-long-k audit
rows, 200k tokens per domain). Real-domain NMI uses the slk definition; the document-label null is about 0.0004–0.0009.

| run | s100 VAL CE | s100 domain NMI | s100 e5 NMI | s101 VAL CE | s101 domain NMI | s101 e5 NMI |
|---|---|---|---|---|---|---|
| TOKEN-4 | 3.7465 | 0.0106 | 0.020 | 3.7327 | 0.0108 | 0.020 |
| explicit EDL | 3.7369 | 0.0499 | 0.074 | 3.7394 | 0.0475 | 0.070 |
| Semantic-EDL C6 (V1 REF) | 3.7319 | 0.0482 | 0.073 | 3.7228 | 0.0449 | 0.068 |
| SHUF-C6 (V1 REF) | 3.7259 | 0.0414 | 0.068 | 3.7228 | 0.0377 | 0.064 |
| V1 SEM-G1a / SHUF-G1a | 3.7205 / 3.7223 | 0.0475 / 0.0394 | 0.071 / 0.066 | 3.7279 / 3.7233 | 0.0471 / 0.0374 | 0.071 / 0.063 |
| **V2 SEM-C4 / SHUF-C4** | 3.7460 / 3.7524 | 0.0120 / 0.0104 | 0.021 / 0.019 | 3.7360 / 3.7322 | 0.0125 / 0.0103 | 0.023 / 0.019 |

**Measured facts.**

- **Routing barely moves.** V2's soft target alone moves routing very little: domain NMI 0.012 against
  TOKEN-4's 0.011, compared with about 0.045–0.050 for EDL and Semantic-EDL.
- **Grouping effect on NMI.** SEM-C4 has slightly higher domain and e5 NMI than SHUF-C4 in both seeds
  (+0.002 domain NMI).
- **CE.** V2's VAL CE is TOKEN-4-like. It lacks the CE gain that every Semantic-EDL-family run (semantic or shuffled)
  shows at 50M.
- **Pseudo-label NMI.** Computed post hoc and exactly from the saved audit profiles. It follows the same ordering:
  V2 about 0.013 (dev) and 0.016–0.019 (hold50), against about 0.05 for the C6 family.

## 3. Post-hoc mechanism diagnostics (descriptive; prereg section 10 chain)

Cells are classified by the sign of their confidence score at step 100 (≈ 500 favoured, ≈ 4,600 neutral, ≈ 500
disfavoured (k,l,e) cells). The table gives means from step 100 to step 700 (6.3M → 44.2M tokens; `tables.md`).

| run | router preference R, favoured (neutral 0.0312) | exposure X, favoured vs neutral | gradient per exposure, favoured / neutral / disfavoured | measured ΔU, favoured (step 100 → 700) | measured ΔU, disfavoured |
|---|---|---|---|---|---|
| SEM s100 | 0.03137 → 0.03177 (+1.3%) | 0.130 vs 0.1246 (+4%) | equal (0.0027 / 0.0027 / 0.0028) | +3.0e-4 → +3.4e-5 | −2.7e-4 → −4.3e-5 |
| SEM s101 | 0.03135 → 0.03159 | 0.128 vs 0.125 (+2%); disfavoured 0.123 | equal | +2.6e-4 → +4.3e-5 | −2.4e-4 → −4.0e-5 |
| SHUF s100 | 0.03132 → 0.03123 (no movement) | 0.125 vs 0.125 | equal | +2.7e-4 → −0.3e-5 | −2.6e-4 → −2.4e-5 |
| SHUF s101 | 0.03134 → 0.03128 | 0.125 vs 0.125 | equal | +2.9e-4 → +2.7e-5 | −2.8e-4 → −3.1e-5 |

Spearman correlations of the chain (cells with evidence; step 100 vs 700):

| link | SEM s100 / s101 | SHUF s100 / s101 |
|---|---|---|
| score → Δlog R | 0.064 / 0.048 | −0.018 / −0.018 |
| Δlog R → ΔX | 0.54 / 0.55 | −0.11 / −0.05 |
| X → G per exposure | 0.10 / 0.13 | — |
| G → ΔΔU | 0.03 / 0.07 | −0.03 / −0.02 |
| ΔX → ΔΔU | **0.16 / 0.18** | **0.00 / 0.00** |
| score → ΔΔU | −0.63 / −0.61 | −0.67 / −0.66 |
| generic positive mass | 0.0 | — |

**Reading the chain link by link (measured):**

- **A, measurement.** Bank reliability is low: Spearman-Brown 0.19–0.23 for SEM and 0.05–0.10 for SHUF, with the sign
  of the split-half r fluctuating. The strong negative score → ΔΔU correlation (−0.6 in both groupings) is
  regression to the mean: most of each cell's initial score was noise.
- **B, target.** Not flat (KL(q‖R) 0.44 for SEM vs 0.27–0.35 for SHUF) and not generic: no positive mass on experts
  favoured by ≥ 8 groups.
- **C, router.** It moves toward the semantic target, but weakly (ρ ≈ 0.05–0.06; R +1–1.3%). It does not move under
  the shuffled target, whose noise-driven tilt does not persist.
- **D, exposure → expert learning.** Exposure follows the router (ρ 0.54) in SEM only. Gradient per unit exposure is
  identical across cell classes, so favoured experts get more gradient only through more exposure. Cells whose
  exposure grew show a small relative usefulness gain in SEM (ρ 0.16–0.18), and none in SHUF (0.00). Favoured
  cells' residual ΔU at 44M tokens is +3.4e-5 / +4.3e-5 (SEM) vs −0.3e-5 / +2.7e-5 (SHUF).
- **E, transfer and F, accuracy.** Not reached as primary links. The held-out difference is null within
  [−5.0e-5, +6.4e-5]. CE is TOKEN-4-like, with no accuracy trade-off.

## 4. Interpretation (not measured facts)

- **Partial chain.** The mechanism chain is present but very weak in the semantic condition, and absent in the
  shuffled one: score → router → exposure → (small) usefulness gain. That is qualitatively what the hypothesis
  predicts, and the shuffled control rules out a pure artefact of the machinery.
- **Why it doesn't show in the outcome.** The effect on routing is about 1% and on held-out functional specificity
  undetectable, because the usefulness estimates are too noisy: in-training reliability about 0.2, and about two thirds
  of the initial score variance regresses away. The binding constraint is measurement (link A), as preregistered.
  Stronger γ (C3) did not help at 20M: CE worsened by 0.004 and the router still barely moved.
- **What the CI rules out and what it doesn't.** It does not exclude a small semantic advantage (< 6.4e-5 seed-mean
  interaction at 50M). It does make a large one unlikely under this method.
- **What V2 lost relative to Semantic-EDL.** Dropping the Semantic-EDL term removed both the routing structure (NMI
  0.05 → 0.012) and the non-semantic CE benefit seen in the EDL family at 50M.

## 5. Remaining hypotheses

1. **Better measurement.** Higher-reliability usefulness estimates could be built from many more probe rows or
   passes. That costs compute: overhead is already 0.85× of step time. Alternatives are amortising the measurement
   across steps with a slower-moving model (EMA weights), or measuring on larger document groups.
2. **Combining terms.** Adding the soft usefulness tilt on top of Semantic-EDL instead of replacing it would keep the
   routing structure and CE benefit, and use usefulness only to choose which experts each group uses. Untested.
3. **Longer training.** At 245M the weak chain might accumulate. This was not run, because the preregistered gate
   was not passed.

## 6. Compute, storage, deviations, integrity

**GPU time: 3.79 GPU-h** (cap 30).

| phase | GPU-h |
|---|---|
| anchor | 0.06 |
| search | 1.35 |
| promotion, including hold50 | 2.35 |
| descriptive VAL pass | 0.05 |

**Wall time:** launch 2026-11-17 20:05 to terminal 23:50, plus the descriptive pass. Together with V1 (7.52 GPU-h)
the session used about 11.3 GPU-h.

**Disk.**

| item | size |
|---|---|
| 9 own run directories | 2.5 GiB |
| retained resumable slots (no-deletion rule) | 11.5 GiB |
| artifacts | 0.01 GiB |
| free at the end | 121.4 GiB (≥ 100 GiB reserve at all times) |

Per-stage storage estimates were logged before each stage.

**Deviation** (`DEVIATIONS.jsonl`, one entry). The artifact directory and test ledger rows were created before the
freeze. The first test run failed for that reason. tt6 was lengthened from 4 to 6 steps, a test-design fix.

**Logging incident** (recorded in DECISION_LOG, not a protocol deviation).

- A "frozen" decision-log entry was appended by a chained script while the freeze had actually refused.
- The refusal happened because V1's RESUME.md had been transiently rewritten to RUNNING by the post-terminal
  evaluation jobs.
- A correction entry was appended, V1 RESUME.md was restored, and the freeze was then performed.

Post-terminal analysis and descriptive scripts (`scu_analyze.py`, `scu_post_val.py`) were written after the freeze.
They are read-only analyses and were never used for a decision.

**Integrity.**

- Prereg hash checked by every script.
- Protected inventory, which now includes all of V1, verified with 0 mismatches over 8,074 files.
- semantic-edl-long-k frozen manifest: 0 mismatches.
- hold50 (CALIB-SELECTION half 1a) was opened once, after the dev stage. Half 1b remains unused.
- The locked test split was never accessed.
- Real labels were used only in the post-terminal descriptive pass.
- No file was deleted, apart from the in-run rotation of superseded resume slots (recorded interpretation).
- No foreign GPU process interrupted any job.
