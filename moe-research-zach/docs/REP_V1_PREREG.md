# csr-representative-v1 — preregistration

Frozen 2026-11-01, before any new data is downloaded and before any new validation example is read.
The SHA-256 of this file is recorded in `ledger/safety_ledger.tsv` (`REP_V1_PREREG`) and in
`docs/REP_V1_PREREG.sha256`. Any later deviation is reported as a deviation; this file is never edited.

This is a **new prospective follow-up experiment** informed by the smoke-small-v1 pilot. It is not a
confirmation of the original preregistration (EXPERIMENT_REPORT.md §1). It is a three-seed, 100M-token,
~29M-parameter experiment; no general claim follows from it.

## 1. Model (identical across A–F)

| Field | Value |
|---|---|
| Tokenizer / vocab | Pythia (EleutherAI/pythia-12b @ bb1e3e71), vocab 50,304 |
| Embeddings | input embedding and LM head tied (`model.tie_embeddings = true`, one shared Parameter) |
| d_model / context / layers / heads | 192 / 1,024 / 8 / 8 (head dim 24, RoPE base 10,000) |
| Dense / MoE layers | layer 0 dense SwiGLU (FFN 768); layers 1–7 MoE |
| Routed experts / top-k / pool m | 32 / 4 / 8 (= min(E, 2k)); dropless; gate = full-softmax probability ("global") |
| Expert FFN / shared FFN | 128 / 256 (one shared SwiGLU expert per MoE layer) |
| Dropout | 0 |
| Norm / init | RMSNorm eps 1e-6; N(0, 0.02), output projections N(0, 0.02/sqrt(2·8)) |
| Attention | causal, block-diagonal per document (unchanged code path) |
| Router | fp32 router logits; Switch load-balance loss coeff 0.01; router z-loss coeff 0.001 |

Exact parameter counts (analytic, and verified on the instantiated model before this file was frozen):
- total unique parameters **28,873,920**;
- active per token **14,423,232**;
- tied embedding matrix 9,658,368; unique parameters outside it **19,215,552 (66.55%)**.

Requirements: total within 20–30M, ≥ 50% outside the tied embedding, and identical parameter names, shapes
and initial values across A–F within a seed. If any fails, the experiment stops; the architecture is not
changed.

Tied-embedding support is enabled only by the new versioned configuration. Its tests cover: shared storage, no
duplicate optimizer parameters, state-dict save/load, exact counting, identical logits after save/load, causal
and incremental equivalence, and every A–F routing condition.

## 2. Data (new, fresh)

- Source: HuggingFaceFW/fineweb-edu @ 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9, file
  `sample/10BT/000_00000.parquet` (the file already used by this project). Columns id, text, url, token_count,
  score.
- Row groups: every row group of the file except the 16 used by fwe-v1 is eligible. 240 are drawn without
  replacement by `numpy.random.default_rng(20261101).choice(eligible, 240)`. Range reads only.
- Deduplication (fail closed against the old data):
  - The reference set is all 16,000 documents of the fwe-v1 raw file.
  - Exact duplicates: SHA-256 of whitespace-normalized lowercased text.
  - Near duplicates: MinHash with 128 permutations, 32 bands × 4 rows, 5-word shingles, estimated
    Jaccard ≥ 0.8. Parameters are identical to fwe-v1.
  - Reference documents are processed first. A new document is dropped if it duplicates any reference
    document or any earlier new document, in canonical order sorted by id.
- Document format: tokens[:1023] + EOS (id 0); a document never exceeds the context.
- Splits are assigned per document before packing, by the bucket of
  `sha256("csr-rep-v1" + id)` / 2^64:

  | Split | Bucket range | Expected tokens |
  |---|---|---|
  | train | [0, 0.825) | ≈ 129M usable |
  | calib | [0.825, 0.86) | ≈ 5.5M |
  | val | [0.86, 0.93) | ≈ 11M |
  | test (locked) | [0.93, 1.0) | ≈ 11M |

  If train has fewer than 110M usable (valid prediction) tokens, the experiment stops.
- Output: `artifacts/data/fwe-rep-v1/` with a manifest. It records source revision, row groups, document IDs,
  per-split hashes, token counts, dedup statistics and disk usage.
- The test split is written by the deterministic build (deduplication and tokenization must see every
  split). It is never loaded by training, calibration, tuning, segmentation, evaluation or analysis code.
- Packing: greedy sequential into 1,024-token rows with block-diagonal document masks. There are no
  cross-document attention or loss targets. Packed rows are condition-independent; segmentation changes only
  chunk ids and positions.

## 3. Segmentation

**Encoder.** Frozen sentence encoder sentence-transformers/all-MiniLM-L6-v2 @ 1110a243, with the same
preprocessing as before. Embeddings are computed on GPU 0 in fp32 through the guarded launcher. Device is
the only change from the CPU path; the encoder's parameters are checksummed before and after.

**Semantic (D, E).**
- Causal sentence-prefix rule, unchanged (`csr-seg-v1`): sentence end detected from the token itself;
  embedding of that sentence's own tokens; cosine to the running mean of the chunk's completed sentences.
- min/max chunk length 64/256; warmup w = 16.
- τ is calibrated on a seeded sample of 3,000 **training** documents (seed 11). The grid is −0.20…0.95 in
  steps of 0.01. The selection target is a non-final chunk mean closest to 128.

**Fixed (B, C).** Starts every 128 tokens.

**Prefix embeddings.** Frozen u for the first 16 tokens of every chunk of length ≥ 16, for the fixed and
semantic tables.

**Random control F-v3 (`csr-random-hazard-v3`), content-free.**

*Inputs.* The generator's only inputs are document length L, sentence-end positions (from token identity),
the document key and a seed. It uses no text, embeddings, similarities or semantic boundaries.

*Rule.*
- It uses the exact eligibility state machine of D: completed-sentence ends only; ≥ 1 completed sentence in
  the chunk; cur_len ≥ 64; force-close at 256; no boundary after the last token.
- At an eligible boundary it closes with probability h[s][r][b]:
  - s = training document-length quartile;
  - r = tokens remaining after the boundary, binned [0, 32), [32, 64), [64, 128), [128, 256), [256, ∞);
  - b = current chunk length in 8-token bins over [64, 256).
- Knowing r uses document length, as F-v2's strata already did. This is non-causal structural knowledge, not
  content. It is reported as a limitation, and its expected effect favours F, which is conservative for H2.

*Fit (training documents only).*
- h starts as the per-cell closure frequency of D's opportunities, from the exact replay of D through the
  same state machine. A cell with fewer than 50 opportunities falls back to its (s, b) marginal.
- Then up to 10 damped multiplicative corrections are applied, matching per-(s, b) realized life-table
  hazards. Fit seeds are 1001–1004. The best iteration is selected on seeds 2001–2002 by the worst normalized
  acceptance criterion, on train only.

*Acceptance (rerandomization; train and calib only; validation never consulted).*
- Application seeds 0, 1, 2, … , 99 are tried in order. The **first** seed whose realized F passes every
  criterion below on both train and calib is frozen:

  | Criterion | Threshold |
  |---|---|
  | Chunk count difference | < 1% |
  | Non-final mean difference | < 0.5 token |
  | Final-chunk mean difference | < 1 token |
  | Pool-eligible token fraction difference | < 0.5 pp |
  | Non-final chunk-length KS distance | < 0.02 |

- D's replay must also reproduce D exactly on every train and calib document. Document boundaries are
  identical by construction.
- If no seed passes, the experiment stops.
- Val F tables use the frozen seed. Val boundary statistics are reported descriptively only.

Caches go to `artifacts/segments/<id>` (semantic and fixed) and `artifacts/segments_random/<id>`. They cover
train, calib and val only; the test split is never segmented.

## 4. Conditions and contrasts

| Condition | Description |
|---|---|
| A | token routing; no chunks; no semantic loss |
| B | fixed-chunk pools; no semantic loss |
| C | fixed-chunk pools; semantic loss |
| D | semantic-chunk pools; no semantic loss |
| E | semantic-chunk pools; semantic loss |
| F | accepted F-v3 random-chunk pools; no semantic loss |

Only condition, run_name, segmentation.strategy, routing.mode and semantic.enabled may differ. Every other
resolved field matches, including inactive semantic parameters. This is checked fail-closed before every run
and again with runtime facts after training.

Contrasts are row − column, with validation CE in nats per valid prediction token:

| Role | Contrast |
|---|---|
| H2 | D − F |
| H3 | E − D |
| H1 support | B − A |
| Semantic loss with fixed chunks | C − B |
| Semantic vs fixed, both with loss | E − C |
| Secondary semantic vs fixed | D − B |

E − A is reported descriptively only.

## 5. Training (identical across A–F within a seed)

| Field | Value |
|---|---|
| Optimizer | AdamW, β = (0.9, 0.95), eps 1e-8, weight decay 0.1 on ≥ 2-D parameters (incl. tied embedding), 0 on norms |
| LR schedule | peak 6e-4; linear warmup 60 steps; constant; linear decay over the final 20% of steps to 6e-5 (WSD) |
| Batch | 32 packed rows × 1,024 per optimizer step (micro-batch 16 rows, 2 accumulation steps) |
| Precision | bf16 autocast, fp32 router; gradient clip 1.0 |
| Budget | 100,000,000 valid training tokens; the run stops at the first whole optimizer step reaching the budget (overshoot < 1 step, identical across A–F) |
| Seeds | 0, 1, 2. For seed s: init.seed = train.seed = s, train_order_seed = 17 + s |
| Run order | A, B, C, D, E, F per seed, seeds in order 0, 1, 2 |
| Rules | no resumption across conditions; final checkpoint and run facts saved |
| Monitoring | log every 25 steps; router-gradient probe (LM vs weighted semantic) every 25 steps on micro-batch 0 |
| Intermediate evaluation | the frozen monitoring subset (first 128 packed validation rows, natural order) every 5,000,000 tokens and at the end. Monitoring only; it never changes training |
| Limits | 5,400 s wall clock per atomic run; VRAM abort at 95% of 56 GiB |

## 6. Semantic objective calibration (training/calibration data only)

**Fixed fields.** n_max = 256 chunks per micro-batch comparison set; min_chunks = 16.

**τ_s and τ_r.**
- Measured at initialization of the seed-99 E model.
- Uses the first 8 training micro-batches in seed-99 order (train_order_seed 116).
- Grids:
  - τ_s ∈ {0.01, 0.02, 0.05, 0.1, 0.2, 0.5};
  - τ_r ∈ {1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2}.
- Selection:
  - τ_s: median effective semantic-neighbour count exp H(Q) closest to 8;
  - τ_r: median effective router-neighbour count exp H(R), averaged over layers, closest to the selected
    semantic count.
- The two selected counts must differ by ≤ 20% of the semantic count, or the experiment stops.

**λ.**
- Five disposable E tuning runs, λ ∈ {0.0001, 0.0003, 0.001, 0.003, 0.01}.
- Each run: seed 99 (init, train and order seed 116), 5,000,000 training tokens with the same model, data,
  optimizer and schedule definition (WSD over the run's own steps).
- Each run logs and probes the router gradient every 5 steps.
- No validation access. After each run, routing metrics are measured on the first 128 packed **calib** rows.

**λ selection.** Choose the smallest λ satisfying all of:
1. the median weighted-semantic / LM router-gradient ratio over the final 20% of tuning steps is in
   [0.05, 0.20];
2. the mean logged semantic loss over the final 20% of steps is below its mean over the first 20%;
3. no non-finite value (status COMPLETE);
4. no router collapse: no MoE layer with > 50% of calib routing assignments on one expert;
5. the mean training LM CE over the final 20% of steps is no more than 0.02 nats/token above that of the
   next smaller candidate. This is vacuous for the smallest candidate.

If no candidate passes, the experiment stops and reports that the semantic objective could not be made
influential under this grid. τ_s, τ_r and λ are then frozen identically in all A–F configs.

## 7. Feasibility gate (before any tuning or main training)

- The CPU suite passes with CUDA hidden, including the new tests.
- Guarded 20-step GPU benchmarks of A and E (seed 0) both complete with all values finite.
  - E uses the calibrated τ values and λ = 1e-4, the first grid candidate; λ does not affect cost.
- Proceed only if:
  - projected peak reserved VRAM < 56 GiB;
  - projected time per 100M-token run < 90 min;
  - projected disk keeps ≥ 100 GiB free;
  - projected GPU time for the 5 tuning runs plus 18 main runs ≤ 10 h.
- The design is never changed because of throughput.

## 8. Evaluation and analysis

**Primary endpoint.** Token-weighted next-token CE on the complete fresh validation split (no --max-rows),
with condition-specific causal routing and segmentation. Documents, valid positions and per-document counts
must be identical across A–F, which is asserted.

**Secondary endpoints.**
- perplexity;
- adjacent-token expert Jaccard;
- pool-constrained coverage;
- expert-load max/mean and coefficient of variation;
- unused experts;
- router entropy;
- pool entropy and distinct fraction;
- semantic-loss and gradient-ratio trajectories;
- tokens/s, GPU-hours and peak VRAM;
- tokens to A's final monitoring-subset CE, by linear interpolation of the monitoring curve within the same
  seed.

**Statistics.** Per seed and contrast: the paired document bootstrap (2,000 resamples, seed 0) over identical
validation documents.

**Prospective criterion.** A contrast passes only if:
- the sign is the same in all 3 seeds;
- the mean |Δ| across seeds is ≥ 0.01 nats/token;
- the 95% interval excludes 0 in every seed.

**H1.** B − A Jaccard (layer mean) is positive in every seed and B − A CE ≤ +0.02 in every seed.

**H2.** D − F passes the criterion with a negative sign.

**H3.** E − D passes the criterion with a negative sign. Attribution to semantic specificity would still need
a shuffled-teacher or matched-strength non-semantic control, which is not part of this experiment.

**Router collapse.** A layer with > 50% of assignments on one expert is reported; it is not a stop rule.

**Hard stops.**
- foreign GPU process;
- < 100 GiB free;
- non-finite loss or gradients;
- CUDA error or OOM;
- checkpoint corruption;
- matched-field failure;
- valid-token mismatch;
- VRAM at the limit;
- wall-clock limit;
- any FAILED, ABORTED or CONTAMINATED status.

**Test split.** Locked and never read.
