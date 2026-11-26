# csr-forecast-counterfactual-utility-v1 (prefix `fcu_`) — Preregistration

Written 2026-11-14 ~20:45 EST, before any result of this experiment exists. It is hashed (sha256 in
`FORECAST_COUNTERFACTUAL_UTILITY_V1_PREREG.sha256`) and made read-only. Any later change goes only to
`artifacts/forecast-counterfactual-utility-v1/deviations.jsonl`, with a timestamp and a statement of whether results
had been seen.

## 0. Startup audit

- **Predecessor.** `csr-forecast-latent-edl-v1` (fle) is COMPLETE: report `FORECAST_LATENT_EDL_V1_REPORT.md`, verdict 4,
  driver state 113 COMPLETE / 5 INVALID_BUG, no external stop.
  - GPU 0 (A100 80GB) is idle with no processes; 144.2 GiB free disk.
  - Everything belonging to fle (and to every other prior experiment) is read-only for this experiment.
- **What fle established.**
  - FL-EDL ≈ TOKEN-4 in CE (−0.0003) and specialisation (FSI ratio 0.98 / 0.95).
  - Past-only latent EDL beat FL-EDL in 3/3 seeds.
  - Before training, past-only clusters were more predictive of the upcoming region's tokens than forecast clusters.
  - The selected FL-EDL auxiliary gradient was about 0.015 % of the LM gradient.
  - Explicit-label EDL (β 5e-3, about 2 % gradient) specialised (FSI 2.1×) and improved CE (−0.0105).
- **Reused read-only.**
  - Data, tokenizer, splits (`efcde27e…`), balanced main stream and plan `f4bc3c935a47`.
  - LFRModel backbone, fle trainer and evaluator, fle forecast caches (`cache/calib_search` etc.), fle drafts
    G0–G3, MiniLM, e5 labels (`windows_calib.npz`, `windows_val.npz`).
  - fle final checkpoints and evaluations: T4 s0/s1/s2, PAST s0/s1/s2, FL s0/s1/s2, EDL s0/s1, SHUF, FD.
- **The locked test split is never opened.**

## 1. Question

Do causal draft-generated forecasts carry information about **which experts are genuinely useful** for an upcoming
region beyond what the causal past already gives? Where is the bottleneck: oracle headroom, forecast quality or
aggregation, the training objective, or distillation? Only if the diagnostic justifies it: does a bounded
counterfactual expert-usefulness objective produce functional specialisation at comparable CE, in a deployable model?

**The deployable model** uses:
- unrestricted per-token top-4 routing over all 32 experts;
- the shared expert;
- no domain labels in training or routing;
- no drafts, forecasts, encoders, clusters or auxiliary heads at inference.

## 2. Phase 1 — bottleneck isolation (CALIB-SEARCH and training data only)

### 2.1 Utility model and regions (frozen)

- **Utility model M:** fle `final_T4_s0`, `runs/20261114T131547Z_forecast_latent_edl_v1_final_TOKEN-4…`
  `ckpt_p100` (sha256 `fb113eb6d849abbd…` per fle `checkpoint_ledger.tsv`, verified before use). It is TOKEN-4,
  trained on main only, so CALIB-SEARCH is held out for it.
- **Replication model:** fle `final_T4_s1` p100 (sha `0d46807b…`). It is used for the test half only, to check that the
  Phase-1 decision does not depend on one checkpoint (descriptive).
- **Regions.**
  - Source: the CALIB-SEARCH audit documents of the fle `calib_search` cache (1,984 segments with a valid point).
  - Forecast points s ∈ {256, 512, 768}; region = tokens [s, min(s+256, len)), keeping regions with ≥ 128 tokens.
  - This gives 5,814 regions (domain counts 1162/1172/1163/1152/1165).
  - **Probe-train set = paper-half A (2,908 regions); probe-test set = paper-half B (2,906 regions).** The halves are
    paper-disjoint; the split is fixed by `lfr_eval.paper_half`.
- **Layers and experts.** Λ_U = MoE indices 5–10 (upper half, 6 layers), all 32 experts. That gives 192 (layer,
  expert) cells per region.

### 2.2 Counterfactual utility (frozen definition)

**Normal route.** The unmodified forward of M.

**Equal-compute counterfactual for (region i, layer l, expert e).**
- It applies only at layer l, and only on region tokens t where e ∈ top4_l(t) under the normal route.
- On those tokens, logit_e is set to −∞. Then p' = softmax over the other 31 experts, and top-4 is recomputed:
  - the selected set becomes old top-4 \ {e} ∪ {rank-5 expert};
  - gates = p' of the selected experts, which is the model's own gate rule and equals renormalisation by 1/(1 − p_e).
- Top-4 active compute is unchanged. No expert output is zeroed. Every other token and layer is identical.

**Utility.**
- u_i,l,e = CE_i(counterfactual) − CE_i(normal).
- CE_i is the mean next-token CE over the region's targets: tokens s+1 … end−1, predicted from positions s … end−2.
- Positive u means the expert was useful for the region.
- When e is selected on no region token, u = 0 exactly and the cell is recorded as inactive.

**Recorded per (i, l, e):**
- region utility u;
- direct utility (mean Δ over targets predicted at positions where e was selected);
- n_sel, the number of selected tokens;
- mean p_e on the selected tokens.

Token-level Δ for every region target is stored in fp16.

**Computation.**
- fp32 with TF32 disabled.
- Positions before s are unaffected by a region intervention (causality). Above the intervened block, only region
  positions are recomputed, attending to cached normal-pass prefix keys and values. This is exact.

**Validation (before the dataset is used).**
- (V1) Hand case: a tiny random LFRModel with an explicit per-token loop reference implementation matches the batched
  engine (max |Δu| ≤ 1e-5).
- (V2) Identical compute: exactly 4 experts are selected on every token in the counterfactual, the changed set equals
  {e} → {rank-5}, and the remaining gates equal p/(1 − p_e).
- (V3) Deterministic replay: 256 random cells are recomputed bit-identically.
- (V4) The engine's normal-route CE equals the standard full forward within 1e-4, and the cached-prefix counterfactual
  equals a full-sequence counterfactual within 1e-4.
- (V5) A null intervention (an expert never selected in the region) gives u = 0 exactly.

### 2.3 Information sources (all at forecast point s, from identical frozen regions)

| id | definition | causal? |
|---|---|---|
| H | causal past: [MiniLM embedding of the last 192 prefix tokens (fle cache `emb_past`); M's final normed hidden state at position s−1; mean of M's final hidden states over [s−192, s)] = 1152 dims | yes |
| P2 | past-only capacity control: MiniLM embedding of the last 64 prefix tokens (384 dims) | yes |
| Z-cur-mean | fle 4-draft (G0–G3) forecast: mean of the 4 MiniLM continuation embeddings + 5 standardised set statistics (A1 inputs) = 389 dims | yes (fle cache, prefix-only, test T1 of fle) |
| Z-cur-cat | the 4 per-draft embeddings concatenated in fixed draft order + 3 per-draft stats = 1548 dims (no aggregation) | yes |
| Z-str-mean / Z-str-cat | stronger ensemble (section 2.5); same construction | yes |
| O32 / O128 | **oracle diagnostic only:** MiniLM embedding of the actual tokens [s, s+32) / [s, s+128). Never used in any deployable model, final condition, validation selection or inference. | **no** |
| X-shuf | for every added block X: rows permuted among regions within the same probe split (seed 2026). Identical dimension and marginal. | — |

### 2.4 Probes and metrics (train on half A, evaluate on half B)

**Probes.**
- Features are standardised on half A.
- Ridge regression, closed form. α is chosen per feature set by 5-fold paper-grouped CV on half A over
  {1, 10, 100, 1e3, 1e4, 1e5}.
- Multinomial logistic regression (L2 chosen by the same CV over {1e-4, 1e-3, 1e-2, 1e-1}) for labels.
- Every feature set gets the same probe family and selection procedure. X-shuf controls equalise dimension.

**Targets.**
- Utility: the 192-dimensional u vector, standardised per cell on half A. Inactive cells count as 0 in training.
- Real domain (5 classes).
- e5 K=16 semantic cluster of the 256-token window containing the region (`windows_calib.npz`, frozen e5-base-v2;
  independent of MiniLM, the drafts and M).

**Metrics on half B.** "Active" means n_sel ≥ 8.

| id | metric | how it is computed |
|---|---|---|
| M1 | utility out-of-sample R² | Pooled over active cells, relative to the half-A cell mean: R² = 1 − Σ(u − û)² / Σ(u − ū_cell)² |
| M2 | Pearson correlation | Between û and u after removing half-A cell means (active cells) |
| M3 | AUROC | Predicting u > 0 (active cells) |
| M4 | best-expert accuracy | Per (region, layer): argmax û among active experts equals argmax u |
| M5 | **realised equal-compute routing ΔCE** | Per (region, layer), replace the expert predicted *least* useful (argmin û among active experts) by its rank-5 alternative. Realised ΔCE is the measured u of that cell, an actually-run equal-compute route, averaged over (region, layer). Lower is better. Bounds: oracle min u; random-choice mean. Gap closure G = (R_H − R_X) / (R_H − R_oracle). |
| M6 | domain and e5 prediction | Log loss and accuracy |
| M7 | forecast quality | Teacher-forced CE of the actual next 32 tokens per draft (from the prefix); ensemble-mixture log-likelihood; top-1 calibration (ECE, 10 bins); continuation diversity (mean pairwise 1 − cos, first-token agreement); inter-draft embedding correlation |

**Confidence intervals.** Paired cluster bootstrap over half-B papers, 2,000 resamples, for every difference X − H,
X − X-shuf and X − (H+P2). Results are reported as incremental over H, not as standalone accuracy.

### 2.5 Z-strong (stronger causal ensemble; only if oracle headroom exists, see 2.6)

- **First candidate:** existing, already-trained causal LMs, 4 fle final 103.6M-parameter models that are **not** the
  utility model:
  - T4 s1, T4 s2, EDL s1, PAST s2 (p100);
  - main data only; all strictly causal.
- **Generation.**
  - Each model generates a 32-token continuation from the identical prefix [0, s): T 0.8, top-k 50.
  - Counter-based Gumbel keys use draft indices 10–13 (fle `seed_key`).
  - Generation is a plain causal recompute with no future tokens (test U6).
  - MiniLM embeds the continuations exactly as in fle.
- **"Stronger" must be demonstrated on CALIB-SEARCH (half A):** both mean teacher-forced next-32 CE and
  ensemble-mixture log-likelihood must be better than G0–G3, with paired 95 % CI excluding 0.
- **Fallback.** If this fails, at most two new drafts may be trained (larger dense draft architecture, draft_shared ∪
  draft_cont data, ≤ 1 GPU-h each) and the same test is re-applied.

### 2.6 Mandatory decision tree (frozen thresholds)

**Oracle headroom.** Material if H+O32 or H+O128 satisfies (a) or (b), **and** beats its own shuffled control on that
metric with CI lower bound > 0:
- (a) ΔR²(M1) ≥ 0.02 with 95 % CI lower bound > 0;
- (b) realised-routing improvement R_H − R_HO ≥ 10 % of (R_H − R_oracle), with 95 % CI lower bound > 0.

**Draft information.** For each of the four Z variants (Z-cur-mean, Z-cur-cat, Z-str-mean, Z-str-cat), significant if
**all** of the following hold:
- on M1 or M5, H+Z beats H with **99 %** CI lower bound > 0 (≈ Bonferroni over 4–5 tests);
- the same metric beats H+Z-shuf with 99 % CI lower bound > 0;
- the same metric beats H+P2 with 99 % CI lower bound > 0.

**Branches.**
- **A — no oracle headroom.** Conclude that draft quality is not the principal bottleneck. Train no drafts and run no
  usefulness-loss search. Final state **NO ORACLE HEADROOM**. Z-current is still reported descriptively.
- **B — oracle helps, no Z variant significant.** FORECAST BOTTLENECK. A bounded probe-only search over at most 4
  forecast/aggregation variants, one factor at a time, in this order:
  - B1: first-16-token embeddings (shorter horizon);
  - B2: forecast token-bag features (M's token-embedding mean over continuation tokens, no MiniLM);
  - B3: capacity-matched learned set aggregator (deep-sets φ/ρ, trained on half A to predict utility);
  - B4: 4 samples per draft instead of 1 (Z-strong only).

  The significance rule is the same, with Holm correction over the variants tried. If none is significant, final state
  **FORECAST BOTTLENECK**; no main-model training. If one is significant, proceed to C with it.
- **C — Z significant.** Proceed to Phase 2 (the objective is the candidate bottleneck).
- **D — distillation.** The teacher/student conditions are added only if Phase 1 shows direct forecast-conditioned
  routing value: a significant M5 improvement of H+Z over H, H+Z-shuf and H+P2 at 99 %. D becomes the final state only
  if a teacher improves CE or FSI and the student loses at least half of that improvement. At most 2 schedules.
- **A failed branch is never reinterpreted as success.**

## 3. Phase 2 — bounded counterfactual expert-usefulness training (only after branch C or D)

### 3.1 Objective

**Utility estimates.**
- Soft latent assignment q (K clusters) per 256-token region from the chosen signal: past-only for PAST-U, forecast for
  FORECAST-U.
- Balanced Sinkhorn soft clustering exactly as in fle, frozen before main training, fitted on training regions only.

**Counterfactual sampling during training.**
- Sampled on a fixed, seeded fraction ρ ∈ {10 %, 20 %} of micro-batches.
- On a sampled micro-batch: one layer l ∈ Λ (uniform) and n_e = 4 experts (uniform among those active in the
  micro-batch).
- Each counterfactual uses the same equal-compute replacement as in 2.2, applied to all tokens where e is selected
  in that micro-batch.
- The resulting per-region Δ is a document-wide-intervention utility (cheaper than 2.2; disclosed).
- Region CE is computed with gradient through both the normal and the counterfactual pass.

**EMA estimate.** U_k,l,e = Σ_i q_ik u_i,l,e / Σ_i q_ik, as an EMA (decay 0.99 per update of that cell).

**Assignment.**
- A_k,l,e = stop_gradient(Sinkhorn(U_k,l,e / τ_a)) per layer: K × 32, uniform marginals, 20 iterations.
- τ_a is set so that mean normalised row entropy ≈ 0.5.
- There is no hard ownership, and every expert remains available to every token.

**Difficulty-matched contrast.**
- S_k,l,e = U_k,l,e − Σ_{j≠k} w_k,j U_j,l,e.
- w_k,j is obtained by reweighting the other clusters' regions so that their joint histogram over (baseline-CE decile
  from the frozen TOKEN-4 search reference) × (position bucket s ∈ {256, 512, 768}) × (length bucket) matches cluster
  k's. No domain label is used.
- Value from the EMA; gradient straight-through from the current micro-batch:
  S = sg(S_EMA) + (S_batch − sg(S_batch)).

**Loss.**
- L_use = mean_{k,l,e} A_k,l,e · softplus((m − S_k,l,e) / τ_m), with m = 0.002 nats and τ_m = m / 4.
- The softplus saturates once the margin is reached, so pressure stops increasing.
- L_total = L_LM + 0.01 L_switch + 0.001 L_z + λ_use L_use.
- λ_use is calibrated on training data (5M-token calibration run, seed 100) so that the median backbone-gradient norm
  of λ_use L_use is the target fraction g ∈ {0.5 %, 1 %, 2 %} of the LM gradient norm.

**Layers.** Λ = upper half (MoE 5–10) or upper third (MoE 7–10). Routing is ordinary unrestricted token routing at all
times.

### 3.2 Conditions

| condition | definition |
|---|---|
| TOKEN-4 | Reused: fle T4 s0/s1/s2 checkpoints and evaluations (hash-verified). Valid because fcu_train's TOKEN-4 path must be code-identical to fle_train, which is checked by source diff and step-0 bit identity. |
| PAST-U | L_use with past-only q (fle PAST representation, same K and τ rule) |
| FORECAST-U | L_use with the best justified causal forecast signal from Phase 1 |
| SHUFFLED-U | FORECAST-U with q permuted among the valid regions of each micro-batch |
| ORACLE-U | Diagnostic upper bound only: q from O128 clusters. Search scale (50M, seed 100) only; never final or deployable. |
| COMPUTE-T4 | Only if FORECAST-U's measured training FLOPs, including counterfactual passes, exceed TOKEN-4's by > 5 %: TOKEN-4 with the same total training GPU-seconds, spent on additional steps of the same balanced stream continuing the plan, schedule stretched proportionally. Reported separately. FORECAST-U is never given extra main-model tokens. |
| FL-EDL, explicit EDL, PAST-latent EDL | fle results, reused read-only |
| FORECAST-TEACHER / FORECAST-STUDENT | Only under branch D: the router receives causal forecast features 0–40 %; teacher influence anneals to 0 over 40–60 % while the student router distils the full router distribution; 60–100 % student only with drafts absent. Teacher evaluated at 40 %; student at 100 %. |

### 3.3 Engineering limit and search

- At most **8 main-model candidates** before freezing, seed 100, CALIB-SEARCH prefixes at 5M / 20M / 50M tokens of the
  FINAL plan.
- Permitted dimensions:
  - signal (current vs stronger);
  - aggregator (mean-stat vs capacity-matched learned set aggregator);
  - K ∈ {8, 16, 32};
  - g ∈ {0.5, 1, 2} %;
  - ρ ∈ {10, 20} %;
  - Λ (upper half vs upper third);
  - ≤ 2 teacher/student schedules (branch D only).
- Default incumbent: current signal, mean-stat, K 16, g 1 %, ρ 10 %, upper half.
- Promotion requires paired evidence (per-document paired CI on CALIB-SEARCH) of no CE harm beyond +0.01 at 20M, and an
  S margin above SHUFFLED-U.
- CALIB-SELECTION is opened once to freeze; VAL is opened once after freezing.

**Rejection rules (any one rejects a candidate).**
- An expert below 0.1× uniform load.
- Routing entropy outside T4 ± 0.05.
- CE worse than T4 by > 0.02 at 20M or 50M.
- Mean |U| over all (k, l, e) > 2× the T4 reference (indiscriminate ablation harm).
- Any single (l, e) with U above the T4 reference's 99.9th percentile ×3 (brittle single-expert dependence).
- Replacement-expert CE on non-assigned clusters worse than T4's by > 0.01 (replacements must stay functional).

### 3.4 Finals and success criteria (frozen)

**Finals.**
- FINAL plan, 245,032,457 tokens, seeds 0 and 1.
- Seed 2 is triggered for TOKEN-4 (reused), FORECAST-U and PAST-U if any of:
  - sign disagreement of FORECAST-U − T4;
  - |mean| ≤ 0.01;
  - the specialisation verdict differs between seeds.
- Evaluation: the fle evaluator unchanged (VAL CE, paired per-document bootstrap CI, per-domain CE, the fle audit,
  per-expert interventions, trajectories at 40/60/80/100 %, cross-seed alignment), plus per-cluster equal-compute
  utility matrices U on the VAL audit subset.

**Accuracy.**
- Comparable: mean(X − T4) ≤ +0.01.
- Better: negative in every primary seed and in the mean.

**Forecast usefulness.** The Phase-1 rule (2.6); the forecast gain must exceed the shuffled and past-only controls.

**Functional specialisation.** All of:
- (i) the equal-compute counterfactual effect of the assigned experts on their cluster exceeds the difficulty-matched
  effect on other clusters (S > 0, mean over assigned cells, CI > 0) on VAL, measured on the final model with no
  forecasts or auxiliary signals present;
- (ii) mean FSI ratio vs T4 ≥ 1.15 for both domain and e5 labels;
- (iii) FSI above SHUFFLED-U and FSI − FSI_shuf > 0 in every seed;
- (iv) cross-seed stability and profile alignment reported (descriptive).

**Distillation.** The final student uses no drafts or artifacts. Report teacher CE, student CE and the retained fraction
of the improvement. Teacher-only gains are not deployable success.

**Anti-gaming.** On every condition:
- the rejection rules of 3.3 on VAL;
- no hard masks, domain-to-expert tables, reduced expert availability, extra main-model tokens, VAL tuning or
  inference-time draft cost (verified by parameter and state-dict audit plus a benchmark with forecast artifacts
  absent).

### 3.5 Final states (exactly one)

- **SUPPORTED:** forecasts add information (2.6), FORECAST-U meets every specialisation criterion and beats PAST-U and
  SHUFFLED-U in FSI, the student (if any) retains it, and CE is comparable or better.
- **FORECAST BOTTLENECK:** branch B ended without a significant variant, or Phase 2 shows PAST-U ≥ FORECAST-U while
  forecasts showed information only through a weak or redundant channel.
- **OBJECTIVE BOTTLENECK:** branch C, but no usefulness-loss condition meets the specialisation criteria at comparable
  CE.
- **DISTILLATION BOTTLENECK:** branch D with teacher gain and student loss as defined in 2.6.
- **NO ORACLE HEADROOM:** branch A.
- **INCONCLUSIVE:** budget, interruption or integrity failure prevents a decision, or the criteria are split in a way
  not covered above.

## 4. Safety, interruption, budgets

- **Splits.** CALIB-SEARCH for engineering, CALIB-SELECTION once, VAL once after freezing; test never.
- **Disk.**
  - Every job requires free − planned peak ≥ 100 GiB.
  - Additional persistent footprint ≤ 35 GiB.
  - No /dev/shm; paths only beneath the repository.
  - Only this experiment's disposable temporaries may be deleted, and every deletion is ledgered.
- **Interruption.** If another GPU process appears (OccupancyMonitor) or SIGINT/SIGTERM/SIGHUP or a STOP marker
  arrives, the job:
  1. finishes the current optimizer step (training) or the current chunk (inference);
  2. writes and verifies a resumable checkpoint or chunk;
  3. marks the run EXTERNALLY_STOPPED;
  4. releases the GPU;
  5. stops the driver;
  6. does not relaunch automatically.
- **GPU budget (target).**
  - Phase 1 ≤ 4 GPU-h (+ ≤ 2 GPU-h for the draft-training fallback).
  - Phase 2 ≤ 22 GPU-h: search ≤ 6, finals and evaluations ≤ 16.
  - Total ≤ 28 GPU-h.
  - Wall clock ≤ 40 h.
  - Every draft, forecast generation, counterfactual forward and auxiliary computation is charged to the method in
    `compute_ledger.tsv`.
- **Ledgers.** `run_ledger.tsv`, `compute_ledger.tsv`, `deleted_artifacts.tsv`, `deviations.jsonl`, `driver_state.json`
  (atomic), plus a hash for every output file in `artifact_hashes.tsv`.
