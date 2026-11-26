# csr-speculative-pool-routing-v1: final report

**Terminal state: `SPECULATIVE POOL ROUTING V1 COMPLETE`** (driver finished 2026-11-06T22:48:21-05:00)

**Result: negative.** Neither finalist reached the preregistered thresholds.
- The causal adaptive-pool finalist (G3-A1) is **+0.0214 nats/token worse** than TOKEN-4 on validation. Per seed: +0.048, +0.004, +0.013. Paired 95% CI [+0.0211, +0.0217].
- The speculative-pool finalist (G3-S1) is **+0.0868 worse**, with CI [+0.0863, +0.0872].
- Neither finalist is accuracy-comparable or substantially better.
- Neither finalist is faster end to end than its matched control.
- Learned (content-conditioned) pools clearly beat load-matched random pools of the same shape. That gain is only relative to random pools: it does not recover what pool restriction loses against unrestricted TOKEN-4.

All numbers below come from files under `artifacts/speculative-pool-routing-v1/`, mainly:
- `final_analysis.json`
- `bench_final_20261107T034750Z.json`
- `*_results.json`

Markdown tables for all generations are also in `.tmp/speculative_pool_v1/report_tables.md`.

---

## 1. Architecture and data provenance

**Foundation:** csr-representative-v1, read from the resolved config `configs/rep-v1/seed0/condition_A.json` (sha256 `14bf92d9…`). No discrepancy with the protocol prose.
- Model shape:
  - d_model 192, 8 causal layers, context 1,024, 8 heads × 24, RoPE.
  - Layer 0 is dense (FFN 768). Layers 1–7 are MoE with 32 routed experts, top-4, expert FFN 128, plus a shared FFN of 256.
  - Tied embeddings, vocab 50,304.
  - 28,873,920 total parameters, 14,423,232 active.
- Training:
  - AdamW, lr 6e-4, betas (0.9, 0.95), weight decay 0.1.
  - WSD schedule: 60 warm-up steps, final 20% decay to 6e-5.
  - Global batch 32 × 1,024 tokens (2 × 16 rows).
  - bf16 with an fp32 router. Aux loss 0.01, z-loss 0.001.

**Data:** `artifacts/data/fwe-rep-v1` (FineWeb-Edu representative).
- train: 126.2M tokens
- calib: 8,179 documents
- val: 16,850 documents
- The locked test split was never loaded. `load_raw_split` refuses it.

**Calibration partition** (`search_partition.json`, frozen before any training): deterministic document hash `sha256('csr-spv1-calib-partition|'+doc_id)`, split 70/30.
- CALIB-SEARCH: search and monitoring.
- CALIB-SELECTION: opened once, after the search freeze.

**Conditions** (code in `src/csr/spv1_model.py`). Every pool condition activates exactly 4 routed experts per token: top-4 of the token's own fp32 router logits **within** the cached pool. The gate is the token's global softmax probability. The shared path is unchanged.
- **T0 TOKEN-4:** unchanged baseline.
- **C1 FIXED-POOL:** the pool is chosen at fixed segment starts (every 4 or 8 tokens) from the segment-start causal hidden state. Pool size 8 or 16.
- **C2 ADAPTIVE-POOL:** a causal detector (miss-mass of the current pool plus a learned linear term) refreshes the pool. Maximum hold 4 or 8. A calibrated threshold controller targets refresh rates of 50, 25 or 12.5–15%.
- **C3 SPECULATIVE-POOL:** a block starts at document start, after a draft mismatch, or after B tokens.
  - The pool is chosen from the block-start hidden state plus features of the frozen draft's deterministic greedy proposals for the current block. The features are α-weighted, RMS-normalised, stop-gradient draft token embeddings.
  - This rule exactly reproduces the block structure of greedy speculative decoding.
- **C4 RANDOM-POOL:** load-preserving deterministic random pools, indexed by (layer, in-document position). They match their finalist's pool size, segmentation and refresh rate (RC-causal, RC-spec).

**Added target parameters**, all below the 0.5% threshold, so no parameter-matched TOKEN-4 control was required:
- C2: 44,359 (0.153%)
- C3: 86,044 (0.297%)
- RC-causal: 1,351
- RC-spec: 0

**Draft model** (`draft_frozen.json`): there was no suitable project-local draft, so one was trained.
- Dense CSRModel: d128, 4 layers, 4 heads, FFN 512, tied embeddings, 7,488,640 parameters.
- Trained on the train split only (seed-0 order, 100M tokens, lr 1e-3).
- Frozen on CALIB-SEARCH (CE 4.548, top-1 0.271) before any target candidate was trained.
- Checkpoint `runs/20261106T192305Z_spv1_train_draft_d128_l4/model.pt`, sha256 `7e8aee08…`.
- Proposal cache: γ = 7 greedy proposals for every train position (`draft_cache/`, 1,452 GPU-s).

## 2. Preregistration and hashes

- `docs/SPECULATIVE_POOL_ROUTING_V1_PREREG.md`, sha256 `9a679555d989971f723cb978650bfc4c2bce78b508556fad5b6ee01a72130434`.
  - Recorded in `ledger/safety_ledger.tsv` (`SP_V1_PREREG`), `manifest.json` and every config (`prereg_sha256`).
  - Re-verified unchanged at the end.
- `manifest.json` records code hashes at preregistration, the base config, the data manifests and the environment.
- `final_frozen.json` records the 15 final config sha256s. `run_inventory.json` hashes every config, checkpoint and evaluation file.

## 3. All candidate configurations and every result (CALIB-SEARCH CE; Δ vs TOKEN-4 trained the same way)

### Generation 1 (seed 98, 5M tokens; T0 = 6.5501)

| candidate | family | config | Δ | refresh | pool recall | top-4 contained |
|---|---|---|---|---|---|---|
| G1-F4M8 | fixed | blocks 4, pool 8 | +0.0143 | 0.250 | 0.448 | 0.172 |
| G1-F4M16 | fixed | blocks 4, pool 16 | +0.0114 | 0.250 | 0.648 | 0.281 |
| G1-F8M8 | fixed | blocks 8, pool 8 | +0.0190 | 0.126 | 0.366 | 0.101 |
| G1-F8M16 | fixed | blocks 8, pool 16 | +0.0127 | 0.126 | 0.593 | 0.196 |
| G1-A4M16R50 | adaptive | max hold 4, pool 16, ~50% | +0.0005 | 0.491 | 0.771 | 0.503 |
| G1-A8M16R25 | adaptive | max hold 8, pool 16, ~25% | +0.0141 | 0.253 | 0.659 | 0.307 |
| G1-A8M8R25 | adaptive | max hold 8, pool 8, ~25% | +0.0132 | 0.249 | 0.417 | 0.117 |
| G1-A8M16R15 | adaptive | max hold 8, pool 16, ~12.5–15% | +0.0126 | 0.140 | 0.603 | 0.211 |
| G1-S4M8 | spec | block 4 (γ 3), pool 8 | **−0.1549** | 0.733 | 0.550 | 0.106 |
| G1-S4M16 | spec | block 4, pool 16 | −0.0806 | 0.733 | 0.735 | 0.332 |
| G1-S8M8 | spec | block 8 (γ 7), pool 8 | −0.0604 | 0.729 | 0.627 | 0.194 |
| G1-S8M16 | spec | block 8, pool 16 | −0.0372 | 0.729 | 0.789 | 0.442 |

### Generation 2 (seeds 98 and 99, 20M tokens; T0 = 5.3828)

Each family's best Gen-1 candidate was promoted unchanged, plus one variant per family (selector lr ×3 and a 10% pool warm-up).

| candidate | parent | Δ (mean) | per seed | status |
|---|---|---|---|---|
| G2-F1 | G1-F8M16 | +0.0174 | +0.0156, +0.0192 | valid |
| G2-F2 | G1-F8M16 variant | +0.0228 | +0.0247, +0.0208 | valid |
| G2-A1 | G1-A4M16R50 | +0.0054 | +0.0034, +0.0074 | valid |
| G2-A2 | G1-A4M16R50 variant | +0.0125 | +0.0127, +0.0124 | valid |
| G2-S1 | G1-S4M8 | −0.0705 | −0.0662, −0.0748 | valid |
| G2-S2 | G1-S4M8 variant | −0.0476 | −0.0431, −0.0520 | **REJECTED**: collapse / unused expert (1 unused, max share 0.064) |

- No candidate was rejected for being worse than T0 by more than 0.05 in both seeds.
- No non-finite runs.

**Note on G1-F8M16 vs G1-F4M16.** G1-F8M16 was promoted from the fixed family even though G1-F4M16 had a slightly lower Gen-1 Δ (+0.0114 vs +0.0127). This follows the preregistered tie-break in `spv1_select.best`:
- The two are 0.0013 nats apart, which is within `TIE_NATS` = 0.002.
- Among tied candidates, the lower refresh rate wins: 0.126 for G1-F8M16 vs 0.250 for G1-F4M16.

### Generation 3 (seeds 97, 98 and 99, 50M tokens; T0 = 4.6662)

| candidate | Δ (mean) | per seed | refresh | recall |
|---|---|---|---|---|
| G3-A1 (from G2-A1) | +0.0218 | +0.0186, +0.0123, +0.0346 | 0.499 | 0.830 |
| G3-S1 (from G2-S1) | +0.0396 | +0.0288, +0.0460, +0.0442 | 0.733 | 0.388 |
| G3-A2 (from G2-A2) | +0.0476 | +0.0275, +0.0515, +0.0637 | 0.502 | 0.682 |

- Search freeze: causal finalist = **G3-A1** (adaptive, max hold 4, pool 16, about 50% refresh). Spec finalist = **G3-S1** (speculative block 4, γ = 3, pool 8). Each was matched to a random control (RC-causal, RC-spec).
- 24 unique-candidate cap respected: 12 + 6 + 3 = 21 slots, of which Gen-3 entries are promotions.

### Blind CALIB-SELECTION (evaluated once)

| finalist | Δ vs TOKEN-4 | status |
|---|---|---|
| causal | +0.0224 | proceeds |
| spec | +0.0404 | proceeds |

- Ranking: TOKEN-4 < causal < spec.

### Budget gate (`budget.json`)

- Selected budget: 100M tokens for all 5 conditions, including the controls.
- Projected total 34,828 s against a 57,600 s limit (80% of 20 h). 12,339 s had been used before the gate.

## 4. Correctness and causality evidence

**CPU tests:** `cpu_tests_20261106T192109Z.json`, 495 passed (91 of them new in `tests/test_speculative_pool_v1.py`). They cover:
- future-token invariance of pools, routes, starts and logits
- fixed-boundary changes only at declared boundaries
- causal adaptive refresh, exactly matching a sequential reference
- spec pools depending only on the prefix and the draft proposals
- rejected suffixes not affecting accepted output (stale-proposal scramble hook)
- exactly 4 distinct experts, all inside the pool, with the shared path unchanged
- independent per-token weights
- TOKEN-4 numerically unchanged
- identical initial backbone within a seed
- full-sequence vs incremental equivalence
- bit-exact resume

**GPU checks:** both PASS.
- `gpu_checks_target_*`: for every Gen-1 candidate, future perturbation gives a max logit difference of 0.0 with identical routes, pools and starts. Incremental vs full gives a max difference of 0.0 and argmax agreement 1.0. Structure: no host sync. One train step is finite.
- `gpu_checks_draft_*`: full vs incremental exact; tree proposals match greedy proposals 420/420; the draft takes 2.77 ms per step vs 9.24 ms for the target.

**Final matching:** `final_analysis.json` asserted, within each seed, identical initial-backbone parameter sha256 across all 5 conditions, identical validation token counts and identical training-token counts.

**Post-hoc diagnostic (NOT preregistered).** Run on G3-S1 seed 97 at 50M tokens, using 4 CALIB-SEARCH rows; output in `posthoc_spec_*.json`.
- Future perturbation: max logit difference 0.0, earlier pools, routes and starts identical. Pass.
- Mechanism:

| draft proposals | CE |
|---|---|
| real | 4.730 |
| from another document | 4.916 |
| random | 5.280 |

## 5. Final validation CE (100M tokens; one evaluation per checkpoint; 16,850 documents)

| condition | seed 0 | seed 1 | seed 2 | mean | Δ vs T0 (per seed) | Δ mean | paired 95% CI | docs improved | accuracy-comparable | substantially better |
|---|---|---|---|---|---|---|---|---|---|---|
| TOKEN-4 | 4.2120 | 4.2328 | 4.2223 | 4.2223 | – | – | – | – | – | – |
| G3-A1 (causal) | 4.2596 | 4.2367 | 4.2348 | 4.2437 | +0.0476, +0.0040, +0.0125 | **+0.0214** | [+0.0211, +0.0217] | 15.9% | no | no |
| G3-S1 (spec) | 4.3060 | 4.3063 | 4.3150 | 4.3091 | +0.0940, +0.0735, +0.0927 | **+0.0868** | [+0.0863, +0.0872] | 0.8% | no | no |
| RC-causal | 4.3268 | 4.3362 | 4.3236 | 4.3289 | +0.1148, +0.1034, +0.1013 | +0.1065 | [+0.1061, +0.1069] | 0.2% | no | no |
| RC-spec | 4.3771 | 4.3716 | 4.3634 | 4.3707 | +0.1651, +0.1388, +0.1411 | +0.1483 | [+0.1478, +0.1489] | 0.06% | no | no |

- No collapse and no unused experts in any final run.
- Training-token and active-expert budgets are identical.
- Note: the document bootstrap is over documents with seed-averaged NLL, so it is far narrower than the between-seed spread. For G3-A1 the per-seed SD of Δ is 0.023.

**Secondary comparisons** (same bootstrap):

| comparison | per-seed Δ | mean | 95% CI |
|---|---|---|---|
| G3-A1 vs RC-causal | −0.067, −0.099, −0.089 | **−0.085** | [−0.0856, −0.0847] |
| G3-S1 vs RC-spec | −0.071, −0.065, −0.048 | **−0.062** | [−0.0621, −0.0610] |
| G3-S1 vs G3-A1 (speculative vs causal) | +0.046, +0.070, +0.080 | **+0.065** | [+0.0649, +0.0659] |

**Learning curves.** Monitor CE minus TOKEN-4 at matched token counts, on the first 128 CALIB-SEARCH rows:

| M tokens | 5 | 10 | 20 | 30 | 50 | 70 | 100 |
|---|---|---|---|---|---|---|---|
| G3-A1 s0/s1/s2 | +.012/+.001/+.001 | +.011/−.007/+.005 | +.009/+.002/+.005 | +.002/−.003/+.008 | +.022/+.002/+.010 | +.042/+.001/+.012 | +.041/+.007/+.008 |
| G3-S1 s0/s1/s2 | −.133/−.149/−.144 | −.109/−.123/−.112 | −.059/−.062/−.056 | −.027/−.035/−.023 | +.020/+.014/+.038 | +.075/+.056/+.088 | +.092/+.077/+.088 |
| RC-causal (mean) | +.015 | +.011 | +.036 | +.043 | +.068 | +.088 | +.107 |
| RC-spec (mean) | +.013 | +.016 | +.048 | +.063 | +.102 | +.128 | +.150 |

- The speculative family starts far ahead, crosses zero between 30M and 50M tokens and keeps losing ground.
- G3-A1 seed 0 diverges from TOKEN-4 after about 50M tokens. The other two seeds stay within +0.01.

## 6. Pool recall, refresh and churn

Validation, seed mean, averaged over the 7 MoE layers:

| condition | refresh rate | mean reuse length | adj. expert Jaccard | distinct experts / 4, 8, 16, 32 tok | pool recall vs unrestricted TOKEN-4 top-4 | TOKEN-4 top-4 fully in pool | router max prob | gate mass on selected 4 | load CV | max expert share |
|---|---|---|---|---|---|---|---|---|---|---|
| TOKEN-4 | – | – | 0.153 | 11.98, 18.58, 25.27, 29.68 | 1 | 1 | 0.227 | 0.480 | 0.053 | 0.037 |
| G3-A1 | 0.497 | 2.01 | 0.214 | 10.95, 16.75, 23.13, 28.09 | 0.811 | 0.523 | 0.229 | 0.425 | 0.065 | 0.038 |
| G3-S1 | 0.733 | 1.36 | 0.240 | 10.84, 16.64, 22.89, 27.64 | 0.475 | 0.0001 | 0.795 | 0.095 | 0.358 | 0.055 |
| RC-causal | 0.498 | 2.01 | 0.177 | 11.77, 18.40, 25.43, 30.10 | 0.510 | 0.056 | 0.179 | 0.307 | 0.035 | 0.034 |
| RC-spec | 0.733 | 1.36 | 0.147 | 13.51, 20.84, 27.68, 31.22 | 0.251 | 0.003 | 0.113 | 0.183 | 0.019 | 0.033 |

**Reuse-length distributions** (fraction of segments of length 1, 2, 3 and 4; seed 0, all layers):

| condition | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| G3-A1 | 0.486 | 0.209 | 0.122 | 0.182 |
| G3-S1 | 0.727 | 0.200 | 0.054 | 0.019 |

- G3-S1 segments are draft-acceptance blocks.
- Pool-selector confidence: the pool takes 0.975–0.997 of the selector's softmax mass in both finalists.
- G3-A1 cuts churn modestly: distinct experts per 8 tokens drop by 9.8% (18.58 → 16.75) and adjacent Jaccard rises from 0.153 to 0.214. The cost is +0.021 nats.

## 7. Draft acceptance

**Teacher-forced, on validation, block 4.**
- Mean accepted draft tokens per block: 0.364. Mean block length: 1.364.
- 97.9% of blocks end at a mismatch.

**Greedy decoding benchmark** (γ = 3, prompt 64 and 128 generated tokens, A100 80GB PCIe, seed-0 checkpoints).
- Generated text is near-repetitive, so acceptance is much higher than teacher-forced.
- "Proposal regenerations" equals the number of blocks: every block regenerates its proposal from the updated prefix, so no pool conditioned on a rejected suffix is reused.

| target | batch | mean accepted / block | accept hist (0, 1, 2, 3) | proposal regenerations | target fwd / emitted token | draft fwd / emitted token |
|---|---|---|---|---|---|---|
| TOKEN-4 + spec | 1 | 1.91 | 7, 11, 6, 21 | 45 | 0.344 | 1.03 |
| TOKEN-4 + spec | 4 | 2.05 | 32, 25, 17, 97 | 171 | 0.328 | 0.98 |
| TOKEN-4 + spec | 16 | 2.06 | 119, 81, 121, 357 | 678 | 0.327 | 0.98 |
| G3-S1 | 1 | 1.98 | 12, 3, 2, 26 | 43 | 0.336 | 1.01 |
| G3-S1 | 4 | 2.22 | 22, 23, 14, 102 | 161 | 0.311 | 0.93 |
| G3-S1 | 16 | 2.00 | 113, 126, 101, 351 | 691 | 0.333 | 1.00 |
| RC-spec | 1 / 4 / 16 | 1.91 / 2.24 / 2.10 | – | 45 / 160 / 670 | 0.344 / 0.308 / 0.323 | 1.03 / 0.93 / 0.97 |

## 8. Exact greedy-equivalence result

| target | fp32, γ 3 | fp64, γ 3 |
|---|---|---|
| TOKEN-4 + spec | **16/16 identical** | **8/8 identical** |
| G3-S1 | **16/16 identical** | **8/8 identical** |
| RC-spec | **16/16 identical** | **8/8 identical** |

- Comparison: batched speculative greedy decoding vs sequential greedy execution of the same speculative-family model.
- No first divergence in any of these.
- Adaptive and random-causal greedy decoding: batched lockstep (batch 16) vs single-sequence = **identical** for both G3-A1 and RC-causal.

## 9. Throughput and VRAM

Same hardware, prompt and generation length. All draft, detector, selection, verification and fallback time is included. Speed gain is relative to the matched control.

| condition | train step tok/s | prefill b1 / b16 tok/s | greedy b1 / b4 / b16 tok/s | spec γ3 b1 / b4 / b16 tok/s |
|---|---|---|---|---|
| TOKEN-4 | 120,171 | 74,383 / 395,616 | 105.4 / 425.6 / 1,659.1 | 145.8 / 582.8 / 2,022.4 (speed control) |
| G3-A1 | 100,439 (−16.4%) | 33,083 / 258,819 (−55.5% / −34.6%) | 80.0 / 318.6 / 1,276.4 (**−24.1% / −25.1% / −23.1%**) | – |
| G3-S1 | 74,073 (−38.4%, draft on the fly) | 33,155 / 137,170 (−55.4% / −65.3%) | – | 130.6 / 509.6 / 1,601.6 (**−10.4% / −12.6% / −20.8%** vs TOKEN-4 + spec) |
| RC-causal | 105,534 | 36,485 / 280,458 | 83.4 / 333.8 / 1,331.9 | – |
| RC-spec | 78,800 | 37,730 / 145,360 | – | 133.6 / 512.4 / 1,807.2 |

- Draft prefill at batch 16: 1.85M tok/s.
- Speculation with the same draft speeds up the plain TOKEN-4 target by +38.3% / +36.9% / +21.9% at batch 1 / 4 / 16. That gain belongs to speculative decoding, not to pool routing.
- No speed claim passes. Every pool condition is slower than its matched control, and none is accuracy-comparable.

**Final training runs:**
- Train tokens/s:

| condition | train tok/s |
|---|---|
| TOKEN-4 | 118k |
| G3-A1 | 99k |
| G3-S1 | 106k |
| RC-causal | 104k |
| RC-spec | 115k |

- G3-S1's training-run throughput excludes draft cost. It reads the γ = 7 proposal cache, whose one-off cost (1,452 GPU-s) is included in the totals below.
- Peak allocated / reserved VRAM was 19.2–20.8 / 21.2–24.9 GiB, well under the 56 GiB limit.
- Decode peak reserved VRAM: TOKEN-4 15.7, G3-A1 15.7, G3-S1 and the controls 18.8 GiB.

## 10. Total GPU time

**28,217 GPU-s = 7.84 GPU-h** of the 20 h cap. This includes everything since the experiment began, including the 36 s interrupted run.

| stage | GPU-s |
|---|---|
| draft training | 379 |
| draft proposal cache | 1,452 |
| GPU checks | 112 |
| Gen 1 | 1,077 |
| Gen 2 (incl. interrupted) | 3,089 |
| Gen 3 | 6,100 |
| CALIB-SELECTION evals | 100 |
| feasibility benchmark | 30 |
| final training (15 runs) | 14,430 |
| validation evals | 1,143 |
| final benchmark | 305 |

- Draft plus target cost for the speculative family: draft training 379 s + cache 1,452 s + per-run training.
- GPU wall span: 14:21 → 22:47 (2026-11-06).

## 11. Deviations and fixes; operational notes

- `deviations.jsonl` is **empty**. No scientific deviation, no rule relaxed, and no candidate, threshold, seed, budget or criterion was changed.
- All correctness fixes were made before preregistered training began (CPU-test stage). There were two:
  - a dtype cast in the spec planner under float64;
  - a test-arithmetic correction.
- **Operational note** (not a scientific amendment; preregistration unchanged): `.tmp/speculative_pool_v1/operational_note_resume_20261106T211819Z.json`.
  - Operator stop at 15:45. Gen-2 run seed99/G2-A1 was interrupted after 36 s with no checkpoint.
  - It is recorded as INCOMPLETE in `failures.tsv` and the GPU ledger, and its directory `runs/20261106T204436Z_spv1_train_gen2_s99_G2-A1` is preserved.
  - The anticipated machine migration did not happen. The run resumed on the same instance at 16:18 with the original driver and was rerun from its original seed-99 initialisation in a new directory.
  - No COMPLETE run was rerun or overwritten.
- **Naming note.** Run directories are named `runs/<ts>_spv1_train_*` (the prefix the guarded launcher uses for this experiment) rather than the `runs/<timestamp>_spec_pool_*` pattern in the protocol prose. This has no scientific effect.

## 12. Protected-artifact audit

- `protected_inventory.verify_20261107T034820Z.json`: **1,813 / 1,813 files match** (0 bad), including every prior experiment's configs, checkpoints, reports, runs, caches and artifacts.
- Final audit (`final_audit_20261107T034821Z.json`): GPU 0 has 0 processes; 289.5 GiB disk free.
- The locked test split was never loaded or hashed.
- Validation was opened only after `final_frozen.json`, once per final checkpoint (15 registered evaluations).
- Previous validation results were not used for any decision.

---

## 13. Answers

**1. Does chunk-level candidate selection retain token-routing accuracy?**
No, by the preregistered definition.
- The best causal pool method (G3-A1) costs +0.0214 nats/token (CI [+0.0211, +0.0217]).
- One seed is +0.048 and only one of three seeds is within +0.01.
- In exchange it reduces churn by about 10% in distinct experts per 8 tokens, and adjacent Jaccard goes from 0.153 to 0.214.
- The restriction is the problem. The pool recovers only 81% of the unrestricted router's top-4, and contains TOKEN-4's full top-4 for only 52% of tokens.

**2. Does adaptive refresh outperform fixed pools?**
Only on search data, and only because it refreshes more often. Fixed pools never reached the final stage, so there is no validation comparison.
- On CALIB-SEARCH the best adaptive candidate beat the best fixed one: Gen-1 +0.0005 vs +0.0114; Gen-2 +0.0054 vs +0.0174 (two seeds each).
- But the winner refreshes at about 50% while fixed blocks refresh at 25% or 12.5%.
- At matched refresh rates, adaptive was no better: about 25%: A8M16R25 +0.0141 vs F4M16 +0.0114. About 12.5–14%: A8M16R15 +0.0126 vs F8M16 +0.0127.
- So there is no evidence that learned refresh timing beats a fixed schedule at the same churn level.

**3. Do speculative current-block proposals outperform causal pool selection?**
No.
- On validation, G3-S1 is +0.065 nats worse than G3-A1 (all three seeds worse, CI [+0.0649, +0.0659]).
- The early speculative advantage (−0.155 at 5M tokens; −0.13 to −0.15 on monitor CE at 5M) is best explained as **knowledge transfer from the stronger draft**, not as better pool lookahead.
  - The draft was trained for 100M tokens before any target existed, so early in target training its proposal embeddings carry information the target does not yet have.
  - The advantage shrinks steadily and crosses zero between 30M and 50M tokens, roughly when the target overtakes the draft (draft CE 4.55; target at 100M 4.22).
  - The not-preregistered mechanism diagnostic agrees: CE 4.73 with real proposals, 4.92 with another document's, 5.28 with random ones.
- The speculative family also yields very short blocks (mean 1.36 tokens, teacher-forced), because acceptance on real text is low. Its pools therefore change on 73% of tokens.

**4. Does either method beat TOKEN-4?**
No. Both are worse in every seed:
- G3-A1: +0.021
- G3-S1: +0.087
- Neither is substantially better or accuracy-comparable.

**5. Does either method provide a measured end-to-end speedup?**
No.
- Adaptive pools are 23–25% slower than TOKEN-4 greedy decoding at batch 1, 4 and 16, and 16% slower in training.
- Speculative pools are 10–21% slower than TOKEN-4 + speculation with the same draft and γ.
- The only real speedup measured (+22–38%) is ordinary speculative decoding on the unrestricted TOKEN-4 target. That is the speed control, and it does not involve pool routing.
- Pool selection saves no compute here: 4 routed experts are still active and experts are tiny. The selector, detector and gather work is pure overhead.

**6. Is any gain content-dependent rather than merely pool restriction?**
Content dependence is real, but there is no gain over TOKEN-4 to attribute to it.
- Learned pools beat load-matched random pools of identical shape by a large margin, in all seeds:
  - causal −0.085 [−0.0856, −0.0847]
  - speculative −0.062 [−0.0621, −0.0610]
- So the selectors do use content.
- Restriction alone costs +0.107 (RC-causal) and +0.148 (RC-spec). Learned selection claws most of that back but not all.

**7. What failed, and why?**

(a) **Restriction loss.**
- Restricting each token's top-4 to a pool of 16 (or 8) chosen ahead of time drops experts the token would have picked.
- At this scale, the churn reduction (about 10%) does not pay for itself, either in accuracy or in measured speed.
- Fewer distinct experts per window could only help with expert offloading or caching, which this model does not have.

(b) **Speculative family: degenerate routing at full budget.**
- At 100M tokens the G3-S1 token router puts most of its global probability (max prob 0.80) on experts the pool excludes. TOKEN-4's top-4 is almost never fully in the pool (0.01%), and only 9.5% of gate mass lands on the 4 selected experts (TOKEN-4: 48%).
- Routed experts therefore contribute weakly, the model leans on the shared path, and expert load becomes uneven (CV 0.36 vs 0.05).
- The pool selector, driven by draft features, and the token router drifted apart, and the gate-equals-global-probability design does not pull them together.
- The cause is the gate design, not the selector failing to learn: the selector still clearly beats random pools.

(c) **Speculative advantage was temporary.** The draft's head start (stronger model early in training) faded and reversed; see Q3.

(d) **Low draft acceptance on real text.** 0.36 accepted tokens per block teacher-forced makes speculative blocks too short to give stable pools. Acceptance in the greedy benchmark (about 2 per block) is higher only because greedy continuations are repetitive.

(e) **One unstable search variant.** G2-S2 (selector lr ×3 + 10% pool warm-up) was rejected for collapse / unused expert.

(f) **Seed sensitivity.** G3-A1 seed 0 drifts away from TOKEN-4 after about 50M tokens (+0.048 at the end), while seeds 1 and 2 stay within +0.013. That suggests training instability in the adaptive controller or selector, which no stage of this experiment diagnosed.

**Scope.** These conclusions hold for a 28.9M-parameter, 32-expert, top-4 model trained on 100M tokens, at batch 1–16 on one A100. They say nothing about pool routing in large MoEs with expert offloading, where expert locality has real I/O value.
