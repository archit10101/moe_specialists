# Forecast Counterfactual Utility v1 — Report

Experiment `csr-forecast-counterfactual-utility-v1` (file prefix `fcu_`), run autonomously on 2026-11-14 from about 20:30
to 22:15 EST on one A100 80GB. Artifacts are in `artifacts/forecast-counterfactual-utility-v1/`.

**Final state: NO ORACLE HEADROOM.**

## 1. Summary

The question was whether forecasts of upcoming text, generated causally by drafts, can tell an MoE model which experts
will be *useful* for a region beyond what the causal past already tells it.

To answer it, I measured exact equal-compute counterfactual utilities. Each removed expert is replaced by its
next-ranked alternative, keeping top-4 compute fixed:
- 5,814 held-out CALIB-SEARCH regions;
- 6 upper MoE layers × 32 experts;
- the frozen TOKEN-4 model.

I then tested how much of that utility different information sources predict, using identical probes on a
paper-disjoint split.

- **Even the actual future text barely helps.** An embedding of the real next 128 tokens (oracle) raises held-out
  utility R² from 0.0295 to 0.0336 (ΔR² +0.0041; preregistered bar 0.02).
- **It does not improve routing.** Choosing which expert to replace closes only 0.4 % of the gap to the best possible
  choice (bar 10 %).
- **The same oracle is very informative about the region's topic:** e5 semantic-cluster log loss improves by +0.23 nats.
  Knowing *what the region is about* does not tell you *which experts will matter in it*.
- **The decision replicates** on a second TOKEN-4 seed (oracle ΔR² +0.0038, gap closure 0.1 %).
- **Region-level utility is mostly a token-level phenomenon.**
  - Split-half reliability is r = 0.09, so full-region reliability is about 0.17.
  - The top 5 % of tokens carry 63 % of the absolute effect.
  - 98 % of a region's utility comes from the tokens where the expert was actually selected.
  - Even an extreme oracle that knows the model's own future routing frequencies for the region reaches R² 0.105 but
    gives no realised routing gain.
- **The current drafts are not the bottleneck.** Their forecasts add a statistically detectable but tiny amount of
  utility information (ΔR² +0.003, beating shuffled and past-only controls at 99 %). That is the same size as the
  oracle's, and it brings no routing benefit.

By the preregistered decision tree (branch A), I did not train stronger drafts, run a usefulness-loss search or train
any main model. No deployable model was produced, and VAL was never opened.

## 2. What was done

### 2.1 Isolation and preregistration

- **Predecessor:** `csr-forecast-latent-edl-v1` (fle).
  - It was verified COMPLETE before starting: report, final analysis, driver state 113 COMPLETE, GPU idle.
  - All of its files were treated as read-only.
  - An integrity check at the end confirms nothing was changed (section 9).
- **Preregistration:** `docs/FORECAST_COUNTERFACTUAL_UTILITY_V1_PREREG.md`
  (sha256 `a9f90644dd2c9dad512a1e3eded69a10073d3da1753e42d413a7e4bd0d94b66d`, see `.sha256`; mode 0400).
  - It was written and hashed before any result existed.
  - It fixes the regions, the utility definition, the information sources, the probes, the decision tree with
    thresholds, the Phase-2 design and limits, the success criteria, the final states and the budgets.

### 2.2 Counterfactual utility (prereg 2.2)

- **Utility model:** fle `final_T4_s0` p100 (sha `fb113eb6…`, verified), fp32 with TF32 off.
- **Replication model:** `final_T4_s1` (sha `0d46807b…`).
- **Regions.**
  - Source: the fle CALIB-SEARCH audit segments.
  - Forecast points s ∈ {256, 512, 768}; region = [s, min(s+256, len)), keeping regions with ≥ 128 tokens.
  - 5,814 regions (domains 1162/1172/1163/1152/1165).
  - Probe-train = paper-half A (2,908 regions, 84 papers); probe-test = paper-half B (2,906 regions, 95 papers).
- **Equal-compute counterfactual for (region, layer l, expert e).**
  - It applies only on region tokens where e is in the normal top-4 at layer l.
  - There, logit_e is set to −∞. The next-ranked expert enters, the gates are the model's own softmax over the
    remaining 31 experts (= p/(1 − p_e)), and exactly 4 experts stay active. No output is zeroed.
  - u = CE(counterfactual) − CE(normal), averaged over the region's next-token targets.
  - Positions before the region are unaffected (causality), so only region positions above the intervened block are
    recomputed, against cached normal-pass prefix keys and values.
  - Region-level and token-level Δ were stored for all 1.12M cells.
- **Validation:** all pass (`tests_cpu_*.json`, `tests_gpu_*.json`).

| test | what it checks | result |
|---|---|---|
| V1 | Batched engine vs the model's own `exclude` forward, on a tiny float64 model | max \|Δu\| 6e-10 over 80 active cells |
| V2 | Replacement rule | Replaced set = old \ {e} ∪ {rank-5}, 4 active experts, gates = p/(1 − p_e) exactly |
| V3 | Replay on the real model | Bit-identical for 192 cells |
| V4 | Engine vs full forward on the real model | 0.0 (normal CE), 2.4e-7 (null), 3.3e-7 (counterfactual); tolerance 1e-4 |
| V5 | Null intervention | Exactly 0.0 |

### 2.3 Information sources (all at the forecast point s)

| id | definition |
|---|---|
| H (causal past) | MiniLM embedding of the last 192 prefix tokens + the utility model's own final hidden state at s−1 + its mean final hidden state over [s−192, s) (1152 dims) |
| P2 | Past-only capacity control: MiniLM of the last 64 prefix tokens |
| Z-current | fle's 4-draft (G0–G3) 32-token sampled forecasts. `mean`: mean embedding + 5 set statistics (A1 inputs). `cat`: all 4 per-draft embeddings + per-draft statistics, with no aggregation. |
| O32 / O128 | **Oracle diagnostic only:** MiniLM embedding of the actual tokens [s, s+32) / [s, s+128) |
| *-shuf | The added block permuted among regions within each split. Same dimension and marginal distribution. |

### 2.4 Probes

- **Identical for every set.** Standardised features; closed-form ridge on the 192 standardised cell utilities, with α
  chosen by 5-fold paper-grouped CV on half A. Multinomial logistic regression for domain and e5 cluster.
- **Metrics on half B (active cells: n_sel ≥ 8):**
  - M1: R² beyond the per-cell mean.
  - M2: Pearson correlation.
  - M3: AUROC for u > 0.
  - M4: accuracy at picking the most useful expert.
  - M5: **realised equal-compute routing ΔCE**. Per (region, layer), the expert predicted least useful is replaced by
    its next-ranked alternative. Its *measured* u is the realised cost, so lower is better.
- **Confidence intervals:** paired cluster bootstrap over test papers (2,000 resamples).

## 3. Results

### 3.1 Utility landscape (T4 s0)

- **How often experts are active:** 94.5 % of (region, layer, expert) cells are active (n_sel ≥ 8); by layer, from
  96 % (MoE 5) to 91 % (MoE 10).
- **Effect of removing an active expert:** mean u = +0.00116 nats/token, mean |u| = 0.00245, and 57.4 % of cells have
  positive utility.
  - Every cell's mean utility is positive (0.00014 to 0.0075): on average every expert is useful where it is selected.
  - Mean utility grows with depth: 0.0006, 0.0006, 0.0009, 0.0012, 0.0017, 0.0020 for MoE 5 to 10.
  - Matrices: `probes/utility_matrix_T4_s0.npz` (mean u, fraction positive, activity per layer × expert).
- **The region-to-region spread of u (0.0036) is the same size as the spread across experts within a region (0.0036).**
- **Token-level structure** (`probes/robust_T4_s0.json`):
  - The top 5 % of a region's tokens carry 63 % of an active cell's absolute Δ.
  - The median share of u coming from the tokens where the expert was selected is 98 %.
  - Split-half reliability (first vs second half of each region, after removing cell means): r = 0.091, so
    Spearman–Brown full-region reliability ≈ 0.17.
  - **No region-level predictor can explain much more than about 17 % of the utility variance beyond cell means.**

### 3.2 Information tests (T4 s0, half B; `probes/phase1_T4_s0.json`)

| feature set | dim | R² (M1) | Pearson (M2) | AUROC u>0 (M3) | best-expert acc (M4) | realised ΔCE ×1e-3 (M5) |
|---|---|---|---|---|---|---|
| H | 1152 | 0.0295 | 0.188 | 0.551 | 0.151 | 0.250 |
| H+P2 | 1536 | 0.0307 | 0.188 | 0.552 | 0.153 | 0.254 |
| H+Z-current mean | 1541 | 0.0323 | 0.192 | 0.552 | 0.155 | 0.252 |
| H+Z-current mean, shuffled | 1541 | 0.0293 | 0.186 | 0.551 | 0.151 | 0.264 |
| H+Z-current cat | 2700 | 0.0332 | 0.190 | 0.552 | 0.155 | 0.257 |
| H+Z-current cat, shuffled | 2700 | 0.0282 | 0.180 | 0.551 | 0.149 | 0.269 |
| H+O32 (oracle) | 1536 | 0.0325 | 0.195 | 0.552 | 0.158 | 0.242 |
| H+O128 (oracle) | 1536 | 0.0336 | 0.199 | 0.553 | 0.158 | 0.228 |
| H+O128, shuffled | 1536 | 0.0292 | 0.186 | 0.551 | 0.149 | 0.268 |
| bounds | | | | | chance 0.033 | best possible −5.33; random +1.17 |

Paired differences (95 % CI over papers):

| comparison | ΔR² | ΔM5 ×1e-3 (lower is better, so + = improvement) | gap closure |
|---|---|---|---|
| H+O128 vs H | **+0.0041** [+0.0037, +0.0046] | +0.022 [+0.008, +0.039] | **0.4 %** |
| H+O128 vs H+O128-shuf | +0.0045 [+0.0040, +0.0050] | +0.041 [+0.025, +0.057] | 0.7 % |
| H+O32 vs H | +0.0030 [+0.0026, +0.0034] | +0.008 [−0.010, +0.027] | 0.1 % |
| H+Z-current mean vs H | +0.0028 [+0.0023, +0.0034] | −0.002 [−0.020, +0.018] | 0.0 % |
| H+Z-current mean vs H+P2 | +0.0016 [+0.0013, +0.0019] | +0.002 [−0.013, +0.019] | 0.0 % |
| H+Z-current mean vs shuffled | +0.0030 [+0.0026, +0.0035] | +0.013 [−0.008, +0.037] | 0.2 % |
| H+Z-current cat vs H | +0.0037 [+0.0029, +0.0046] | −0.007 [−0.030, +0.016] | −0.1 % |
| H+Z-current cat vs H+P2 | +0.0025 [+0.0019, +0.0031] | −0.003 [−0.021, +0.015] | −0.1 % |
| H+P2 vs H | +0.0012 [+0.0009, +0.0015] | −0.004 [−0.019, +0.014] | −0.1 % |

**Label probes.** Δ log loss vs H; positive means the added features help.

| added | domain (5-way) | e5 cluster (16-way) |
|---|---|---|
| O128 | +0.007 [−0.012, +0.025] | **+0.229** [+0.196, +0.267]; accuracy 0.603 → 0.666 |
| O32 | −0.008 [−0.020, +0.002] | +0.100 [+0.085, +0.118] |
| Z-current mean | −0.024 [−0.043, −0.008] | +0.016 [+0.005, +0.028] |
| Z-current cat | −0.049 [−0.080, −0.022] | −0.005 [−0.021, +0.011] |
| P2 | −0.018 [−0.030, −0.006] | +0.005 [−0.003, +0.013] |

H already predicts domain at 72.7 % accuracy and e5 cluster at 60.3 %.

### 3.3 Forecast quality (M7; `features/quality_current.npz`)

- **Teacher-forced CE of the actual next 32 tokens:**
  - per draft: 3.357 (G0), 3.177 (G1), 3.170 (G2), 3.176 (G3); mean 3.220;
  - ensemble mixture: 3.096;
  - top-1 calibration error (ECE) is about 0.01 for each draft, so the drafts are well calibrated.
- **Diversity:**
  - mean pairwise 1 − cos 0.63;
  - first-token agreement 0.39;
  - inter-draft embedding correlation 0.32.
- **Semantic accuracy:**
  - cos(single draft continuation, actual next 32) = 0.33–0.34;
  - cos(draft mean, actual) = 0.465;
  - cos(past-192 embedding, actual) = 0.441.

### 3.4 Preregistered decision (prereg 2.6)

- **Oracle headroom:** neither O32 nor O128 meets (a) ΔR² ≥ 0.02 or (b) gap closure ≥ 10 %, so **no oracle headroom →
  branch A**.
- **Drafts (descriptive under branch A):** both Z-current variants are "significant" on M1. Each beats H, its shuffled
  control and H+P2, all with 99 % CI > 0. Neither is significant on M5, so there is no direct routing value.
- **Branch A actions:**
  - no Z-strong generation (`fcu_zstrong.py` written, not run);
  - no draft training;
  - no Phase-2 usefulness-loss search;
  - no main-model training, no CALIB-SELECTION, no VAL.

### 3.5 Robustness (descriptive, declared before running as deviation 3; cannot change the decision)

**Replication on T4 s1** (`probes/phase1_T4_s1.json`, utilities recomputed on that model): identical decision, branch
A.
- Oracle ΔR²: +0.0038 (O128), +0.0026 (O32).
- Gap closure: 0.1 % / 0.0 %.
- Z-current ΔR²: +0.0024 (mean), +0.0030 (cat), both significant on M1 only.
- Utility landscape almost identical: mean u 0.00117, 57.5 % positive.

**Ridge-penalty grid.** Every preregistered set chose the boundary α = 1e5. Refitting with α up to 1e7 chose
α = 1e5 again and gave the same R² values. The boundary did not limit the conclusion.

**Maximal oracles** built from the utility model's *own* processing of the actual future region
(`features/oracle_model.npz`):

| added to H | ΔR² | gap closure |
|---|---|---|
| region-mean final hidden state | +0.046 [+0.041, +0.053] | −0.8 % [−1.5 %, −0.1 %] |
| region-mean router distribution (knows future selection frequencies) | +0.075 [+0.069, +0.083] | 0.1 % [−0.6 %, +0.9 %] |
| router distribution alone | R² 0.117 | ≈ 2.5 % |

These oracles are not available to any draft-based method. The region-mean router distribution is close to circular,
because utility requires selection. Even so, **no future representation, however privileged, produces a material
realised routing improvement.**

The model-hidden oracle does pass the R² bar on its own. So *some* region-level utility signal exists in future
model-internal states. But it does not translate into better equal-compute routing choices. Its ceiling is bounded by
the about 0.17 reliability, and the preregistered text oracle does not contain it.

## 4. Answers to the eleven questions

**1. Does future forecast information add anything beyond the causal past?**
- Almost nothing.
- Z-current adds ΔR² +0.003 to +0.004 on held-out utility, statistically real (beats shuffled and past-only controls at
  99 %, replicated on T4 s1).
- It gives no improvement in realised equal-compute routing.
- It adds no domain information (overfits) and only +0.016 nats on e5 clusters.

**2. Does oracle future information show genuine headroom?**
- No, for the preregistered oracle. The actual next 32/128 tokens add ΔR² ≤ 0.004 and ≤ 0.4 % routing gap closure, far
  below the frozen bars, in both seeds.
- Privileged model-internal oracles reach ΔR² 0.05–0.08 but still give no realised routing gain.

**3. Are the current drafts weak, too correlated, or merely redundant with H?**
- **Weak, but not the binding constraint.**
  - Next-32 CE is 3.17–3.36, against about 2.97 region CE of the 104M student.
  - Their mean continuation is barely closer to the real future than the past itself (cos 0.465 vs 0.441).
- **Not too correlated:** inter-draft correlation 0.32, first-token agreement 0.39. The drafts are well calibrated
  (ECE ≈ 0.01).
- **Largely redundant with H on semantics:** e5 +0.016 nats vs the oracle's +0.229.
- None of this matters for the experts, because even perfect future text adds only +0.004 R².

**4. Does aggregation destroy useful draft disagreement?**
- Not detectably. The unaggregated per-draft concatenation gives ΔR² +0.0037 vs +0.0028 for the mean (overlapping CIs).
  Neither improves routing.
- The ceiling is set by the oracle, not by aggregation. The branch-B aggregation variants were therefore not run.

**5. Is JSD the bottleneck?**
- Not the primary one; the failing link is upstream.
- In fle, explicit-label EDL with the same JSD form specialised (FSI 2.1× TOKEN-4) and improved CE (−0.0105). So the JSD
  objective can create specialisation when given a strong signal.
- Here the forecast partition carries almost no information about expert utility beyond the past, so no objective
  driven by it has a utility signal to amplify.
- fle's weak auxiliary gradient (about 0.015 % of LM) was a contributing factor there, but it is not what limits this
  premise.

**6. Does the bounded usefulness loss create functional specialisation?** Not tested. The preregistered branch A forbids
the usefulness-loss search when the oracle shows no headroom.

**7. Does specialisation improve, preserve, or hurt CE?** Not tested here (no Phase 2). In fle, label-driven
specialisation helped (EDL −0.0105), while forecast- and past-latent EDL gave none and preserved CE.

**8. Can a draft-free student retain the gain?** Not applicable. There is no teacher gain to distil: Phase 1 shows no
direct routing value (M5 not significant for any forecast), so branch D was never eligible.

**9. Does the method outperform past-only, FL-EDL, explicit-label EDL, and TOKEN-4?**
- No method was trained. The read-only fle results stand: FL-EDL ≈ TOKEN-4, past-only latent EDL slightly better than
  FL-EDL (3/3 seeds), explicit-label EDL best.
- Phase 1 explains why forecast variants cannot beat past-only: forecasts carry no material utility information
  beyond the past.

**10. What is the complete training and inference cost?**
- **Phase-1 GPU: 2,758 s = 0.77 GPU-h:**
  - counterfactual utilities 2 × 1,354 s (T4 s0 with token-level Δ; T4 s1);
  - oracle and past features 5.5 s;
  - draft forecast quality 14.6 s;
  - model-internal oracles 27.7 s;
  - GPU tests 3.6 s.
- **CPU probes:** 1,727 s.
- **Reused (already paid by fle, not recharged):** drafts G1–G3, 0.49 GPU-h; CALIB-SEARCH forecast cache, 70 s.
- **Budget:** 0.77 GPU-h against the 4 GPU-h Phase-1 cap; wall clock about 1.75 h against 40 h.
- **Inference:** no deployable model was produced, so there is no inference cost. Every fle model keeps its unchanged
  inference cost.

**11. If it fails, what is the failed causal link?**

The broken link is **future information → region-level expert utility**. It is the step *before* forecast quality,
aggregation, objective or distillation. Evidence:
1. Actual future text adds ΔR² 0.004 and 0.4 % routing gap closure, in both seeds. Yet it carries a lot of semantic
   content: +0.23 nats on e5 clusters.
2. Region-level utility is only about 17 % reliable. It is dominated by a few tokens (63 % of |Δ| in 5 % of tokens), and
   98 % of it occurs at the tokens where the expert was selected. That is, it is decided token by token by the router,
   which already sees the causal context.
3. Even privileged future model states that predict utility better (ΔR² 0.05–0.08) do not improve realised
   equal-compute routing.
4. The causal-past probe already recovers part of the random → best-possible gap (0.25 vs 1.17 × 1e-3). Every future
   source adds less than 1 % more.

What is **not** evidenced:
- that token-level (rather than region-level) utility is unpredictable;
- that a stronger forecaster could find a different kind of signal. The oracle bounds what text forecasts can carry,
  so this is unlikely, but it is not tested.

**Narrowest justified follow-up (not run).** A token-level utility diagnostic: does any causal token-level feature
predict per-token counterfactual utility better than the router's own logits? If not, the router is already using the
available information, and the region-level specialisation premise should be dropped rather than refined.

## 5. Costs, storage, ledgers

| item | value |
|---|---|
| GPU (ledgered, `compute_ledger.tsv`) | 0.77 GPU-h; no unledgered GPU work |
| Peak VRAM | ≈ 8 GiB (utility job, fp32) |
| New persistent storage | 635 MiB in `artifacts/forecast-counterfactual-utility-v1` (utility 603 MiB including token-level Δ, features 32 MiB); `.tmp` 20 KiB |
| Disk free | 144.24 GiB at start → 143.53 GiB at end (reserve 100 GiB respected; no /dev/shm; nothing outside the repository) |
| Deletions | none (`deleted_artifacts.tsv` empty) |
| Driver | 8 jobs, all COMPLETE; `external_stop` none; no interruption |

## 6. Configurations, hashes, seeds, token counts

- **Prereg:** `a9f90644dd2c9dad…`. **Utility models:** T4 s0 `fb113eb6…`, T4 s1 `0d46807b…` (fle p100, 245,032,457
  training tokens each, seeds 0/1).
- **Forecast drafts (fle):**

| draft | sha | origin |
|---|---|---|
| G0 | `9cafd996…` | fspv2 shared draft |
| G1 | `3544d882…` | seed 11 |
| G2 | `b516c383…` | seed 12 |
| G3 | `9b62722d…` | seed 13 |

- **fle cache:** `cache/calib_search` chunk sha `d28eb0d8…`.
- **Seeds:** probe shuffles 2026; bootstrap 2026.
- **Data:** CALIB-SEARCH only. Regions: 5,814; region tokens per region ≤ 256 (targets ≤ 255).
- **Utility manifests:** T4_s0 `aae5493f…` (24 chunks), T4_s1 `0104bcf0…` (24 chunks). Per-chunk hashes are in
  `artifact_hashes.tsv` (48 chunk entries).
- **Outputs:**

| file | sha (prefix) |
|---|---|
| `features/base.npz` | `c58c48de…` |
| `features/quality_current.npz` | `ee749909…` |
| `features/oracle_model.npz` | `0e1bfb7b…` |
| `probes/phase1_T4_s0.json` | `3099d22b…` |
| `probes/phase1_T4_s1.json` | `1825a4e5…` |
| `probes/robust_T4_s0.json` | `c3cf6060…` |

- **Code:**

| file | sha (prefix) |
|---|---|
| `src/csr/fcu_lib.py` | `74e25741…` |
| `fcu_utility.py` | `57f5b5e4…` |
| `fcu_features.py` | `8c43b814…` |
| `fcu_probe.py` | `ecf5a0c2…` |
| `fcu_robust.py` | `18d1d350…` |
| `fcu_tests.py` | `c850ea00…` |
| `fcu_driver.py` | `e24c34e8…` |
| `fcu_zstrong.py` | `6b1d6f60…` (not run) |
| `fcu_integrity.py` | `93e35221…` |

## 7. Deviations (`deviations.jsonl`)

| # | results seen? | what changed | effect |
|---|---|---|---|
| 1 | no | The T4 s1 replication was computed on both halves, region-level only. The prereg said "test half only", but probe training needs same-model data. | Descriptive only |
| 2 | no | V5 exercised on a 1-token tiny-model region (real regions had every expert active). Inactive cells are u = 0 by definition. V4 tolerances met. | None |
| 3 | yes | Descriptive robustness analyses declared after seeing the decision: split-half reliability, token structure, extended α grid, model-internal oracles, T4 s1 probe. | Cannot change the decision |

## 8. Reproduction

Run from `/home/zapan4/moe-research` after `source env.sh`. GPU jobs go only through `fcu_driver.py`, which calls
`scripts/gpu_guarded.sh`. Every job is idempotent: COMPLETE jobs are skipped and chunked jobs resume from verified
chunks.

```bash
.venv/bin/python fcu_tests.py                                              # CPU V1/V2/V5
MOE_GPU_LAUNCH=1 MOE_RESEARCH_APPROVE=gpu scripts/gpu_guarded.sh "fcu tests gpu" fcu_tests.py --gpu --execute   # V3/V4
.venv/bin/python fcu_driver.py job utility_T4_s0 --gpu --disk-gb 2 -- fcu_utility.py --model T4_s0
.venv/bin/python fcu_driver.py job features_base --gpu -- fcu_features.py --what base
.venv/bin/python fcu_driver.py job quality_current --gpu -- fcu_features.py --what quality --ensemble current
.venv/bin/python fcu_driver.py job probe_phase1_T4_s0 --cpu -- fcu_probe.py --utility T4_s0        # decision
.venv/bin/python fcu_driver.py job utility_T4_s1 --gpu -- fcu_utility.py --model T4_s1 --no-tokens
.venv/bin/python fcu_driver.py job features_oracle_model --gpu -- fcu_features.py --what oracle_model
.venv/bin/python fcu_driver.py job robust_T4_s0 --cpu -- fcu_robust.py --utility T4_s0
.venv/bin/python fcu_driver.py job probe_phase1_T4_s1 --cpu -- fcu_probe.py --utility T4_s1
.venv/bin/python fcu_integrity.py
```

**Resume and interruption.**
- A stop signal, STOP marker or foreign GPU process makes a job finish its current chunk, verify it, exit 77
  (EXTERNALLY_STOPPED) and latch `driver_state.json["external_stop"]`.
- Restart only on explicit instruction: `.venv/bin/python fcu_driver.py authorize-resume`, then rerun the same command.
- The experiment is complete; rerunning any command is a no-op.

## 9. Integrity and final state (2026-11-14 ~22:15 EST)

- **Integrity** (`integrity_20261115T030203Z.json`): all OK.
  - All 5,340 protected pre-fle files match.
  - All 82 fle milestone checkpoints match `checkpoint_ledger.tsv`.
  - All 19 fle cache chunks match their manifests.
  - fle report, final analysis, frozen record and prereg hashes are recorded.
  - The locked test split, CALIB-SELECTION and VAL were never opened.
- **GPU:** A100 0 % utilisation, 0 MiB used, no compute processes.
- **Processes:** no `fcu_*` process running.
- **Disk:** 143.53 GiB free.
- **Storage:** 635 MiB new, all beneath the repository.

## Final state

**NO ORACLE HEADROOM.**
