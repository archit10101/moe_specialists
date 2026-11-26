# EXPERIMENT_REPORT.md — Causal Semantic Chunk Routing

Status: **Stage A complete; Stage B results are in §5.** No pilot has been run. Every number in this file
comes from an artifact under `runs/`, `artifacts/` or `ledger/`. Anything not run is marked **NOT RUN**.

## 1. Pre-registration (frozen 2026-10-31, before any training or held-out result)

**Framing.** The model is a FLAME-38M-architecture MoE trained from scratch in this project (MODEL_DECISION.md).
Results are labelled "adaptation of our own small base model" or "small-scale from-scratch training", never
pretraining efficiency or emergent modularity at scale.

**Primary endpoint.** Token-weighted held-out next-token cross-entropy, in nats per valid prediction token.
It is measured on the full validation split (1,165 documents, 753,939 valid prediction tokens) at the end of
training, with each condition using its own causal routing and segmentation. Every condition is scored on the
same documents and the same valid positions; this is asserted at run time.

**Contrasts** (difference = row − column; negative means the first condition is better):

| Role | Contrast | Hypothesis |
|---|---|---|
| Primary H2 | D − F | Semantic vs length-matched random boundaries |
| Primary H3 | E − D | Semantic loss on top of semantic chunk pools |
| Supporting | B − A | Effect of a chunk pool (H1 loss side) |
| Supporting | C − B | Semantic loss with fixed boundaries |
| Supporting | E − C | Boundary effect, both with the semantic loss |
| Secondary | D − B | Semantic vs fixed boundaries (confounded by the length distribution) |

E − A alone is never used as evidence. Total training losses are never compared across conditions with
different auxiliary terms.

**Decision rule.**
- A contrast is "meaningful" only if |Δ| ≥ 0.01 nats/token **and** the 95% document-level paired bootstrap
  CI (2,000 resamples) excludes 0.
- A one-seed pilot is screening evidence only.
- Confirmation needs 3 training seeds (A, D, E, plus C if feasible), with the same sign in every seed.
  Seed variation and document-bootstrap uncertainty are reported separately.

**H1** is supported if the B − A within-chunk consistency (adjacent-token routed-expert Jaccard, averaged over
MoE layers) increases **and** the B − A loss is ≤ +0.02 nats/token. That non-inferiority margin is declared here.

**H3 attribution.** If E − D or C − B is meaningful, attribution to *semantic* affinity also needs a later
shuffled-teacher or matched-strength non-semantic regularizer control. That control is not part of this pilot.

**H4.**
- Target loss = condition A's final validation CE.
- Report the tokens-to-target for each method, linearly interpolated from evaluations every 0.5M tokens.
- Report cumulative end-to-end GPU-hours to target, plus CPU preprocessing (sentence detection, encoder
  embeddings, boundary caches) and evaluation and failed-run overhead.
- Sample efficiency compares equal valid training tokens. Computational efficiency compares cumulative
  GPU-hours plus preprocessing time.

**Secondary endpoints.** Perplexity; loss versus tokens and versus GPU-hours; tokens/s; peak VRAM;
pool-constrained coverage; expert utilization and load imbalance; token and pool entropy; adjacent-token
overlap; pool diversity; semantic-versus-router similarity; single-topic and multi-topic subset loss;
controlled-transition loss by distance from the junction; forced-pool interventions (probe.py).

**Stop rules (all automated).**

| Trigger | Run status |
|---|---|
| Non-finite loss or gradient norm | FAILED |
| Peak VRAM ≥ 95% of 56 GiB | ABORTED |
| Wall clock > 90 min per atomic run | ABORTED |
| Foreign GPU process | CONTAMINATED |
| Valid-token accounting mismatch | ABORTED |

Router collapse is monitored (max expert load and chunk-router entropy). A layer with > 50% of assignments
on one expert is reported as collapse.

**Test-split policy.** The test split (1,201 documents) is never read by training, calibration, the probe or the
evaluation sets. It may be evaluated only once per final checkpoint, after every validation analysis is frozen,
and only with separate user authorization.

## 2. Hyper-parameter freezing

- τ (semantic boundary threshold) is calibrated on training documents only.
- τ_s, τ_r and λ are calibrated by `calibrate.py` on training data at the **pilot initialization checkpoint**,
  using the declared grids. They are then frozen identically for all six conditions.
- m = 12 (= min(E, 2k) with E=64, k=6), w = 16, fixed chunk size 128, min/max chunk length 64/256.

## 2a. Segmentation statistics (training split; read-only diagnostics in `.tmp/diag/`)

**Semantic condition D** (frozen; τ = 0.36, calibrated on training documents only):

| Statistic | Value |
|---|---|
| Chunks | 71,014 |
| Non-final mean (the calibration target) | **128.0** (128.00136) |
| All-chunk mean | **117.4** (117.40799), std 67.15 |
| Final-chunk mean | 69.4 |
| p10 / p25 / median / p75 / p90 | 59 / 72 / 95 / 153 / 250 |
| Exactly 128 tokens | **0.362%** |
| Closed by threshold / forced at 256 / ambiguous at 256 / document end | 72.38% / 9.10% / 0.46% / 18.07% |

The distribution is right-skewed with two peaks: the mode is the 65–80 bin, with a second spike at 241–256.
"128" describes only the non-final mean, not the typical chunk.

**F-v1 (length-sampled random): SUPERSEDED, invalid as a control.**
- It resampled D's non-final lengths, which were already censored by the document end, and then censored them
  at the document end a second time.
- Result: non-final mean 122.0, 73,070 chunks (+2.9%), and fewer 241–256 chunks.
- The tables remain in the immutable semantic cache directory, but `train.py` refuses to load them.

**F-v2 (hazard closer): ACCEPTED WITH A DOCUMENTED GAP (user decision, 2026-10-31).**

| Item | Value |
|---|---|
| Cache | `artifacts/segments_random/8c1102239971ba4e` |
| Policy | `policy.json` (generator `csr-random-hazard-v2`) |
| Seed | 7 |
| Splits generated | train, calib, val (test never read) |
| Configs | `configs/v2/` |

- **Design.** It uses the same eligible completed-sentence boundaries, the "≥ 1 completed sentence in the
  chunk" rule, the 64/256 limits, forced closes and document ends as D.
  - The closure probability depends only on document-length stratum and current chunk length, in 8-token bins.
  - It was fit on training documents by matching D's realized, document-end-censored process.
  - Feeding D's own decisions through this state machine reproduces D exactly on every train, calib and val
    document.
- **Train vs D.**

  | Criterion | Result | Status |
  |---|---|---|
  | KS (all / non-final) | 0.012 / 0.009 | pass |
  | Chunk count | −0.36% | pass |
  | Pool-eligible fraction | +0.02 pp | pass |
  | All-chunk mean | +0.43 | pass |
  | Median / p75 / p90 | within 4 tokens | pass |
  | 241–256 bin | within 0.38 pp | pass |
  | Non-final mean | **−1.09** (126.91 vs 128.00) | **user-exempted** |
  | Final-chunk mean | **+7.5** (76.9 vs 69.4) | **user-exempted** |

- **Cause of the exempted gaps.** D's closure rate rises near the document end (0.33 with ≤ 31 tokens
  remaining vs ≈ 0.25 elsewhere), plausibly because of end-of-document content. A closer limited to chunk
  length and stratum cannot reproduce that dependence.
- **Held-out splits (seed 7).** These are single realizations on small splits; across 8 other seeds the
  non-final gap on val ranges from 0.02 to 2.8 tokens.
  - val: 6,325 vs 6,447 chunks (−1.9%); all-chunk mean +2.3; final mean +12.9; KS 0.019.
  - calib: −2.0% chunks; KS 0.024.
- **Pre-declared caveat for H2 (D − F).** D additionally concentrates closures in the last ~64 tokens of
  documents, so its final chunks are shorter. Any D − F effect is therefore also reported restricted to
  positions more than 256 tokens from the document end. The decision rule still uses the unrestricted
  primary endpoint.

## 2b. Engineering smoke configuration (not evidence)

`configs/smoke-small-v1` is a 15,691,392-parameter model (13,971,072 active; 16 experts, top-2, m = 4). It is
used only for pipeline, memory and throughput checks. Its results are labelled ENGINEERING SMOKE TEST ONLY
and are never reported as evidence for H1–H4. The experiment design remains the frozen full-size `configs/v2`.
See docs/SMOKE_SMALL_V1.md.

## 3–7. Results

See §5 (Stage B). Pilot results: **NOT RUN**.
