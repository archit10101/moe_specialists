# csr-domain-forecast-curriculum-v1 — preregistration

Written 2026-11-10 (~00:45 EST). This is before any data build, model, training or evaluation for this experiment.
It is a complete experiment, not a pilot: every condition and every seed runs regardless of intermediate results.
After this file is hashed (`docs/DOMAIN_FORECAST_CURRICULUM_V1_PREREG.sha256`), any change is logged in
`artifacts/domain-forecast-curriculum-v1/deviations.jsonl` and is never applied silently. Short name: **dfcv1**.
Code prefix: `dfc_`.

## 1. Question
Can causal 4-token forecasts from a small domain-adapted draft, combined with a soft domain-group router prior, create
useful early expert specialisation in the representative NanoMoE? And does that specialised initialisation give better
final next-token CE after all forecast and domain assistance is removed and training continues as ordinary TOKEN-4?

## 2. Data
- Source: `machelreid/m2d2` at revision `6ae613ba744ee56f645aee9c577d04f7e3d48b30` (the pinned revision of
  m2d2dp-v1), licence CC-BY-NC-4.0. Archives `data/<dom>.tar.gz`.
- Candidate domains in fixed order: cs_l1, math_l1, physics_l1, q-bio_l1, econ_l1 (preferred), then stat_l1, q-fin_l1,
  eess_l1 (fallbacks). All eight archives are inventoried. Each is SHA-256 verified against its LFS hash while
  streaming. Archives over 200 MB are streamed and not stored, because of the disk reserve; smaller ones are stored.
  Up to 6 attempts are made, with exponential backoff (30 s · 2^i, capped at 600 s).
- Members: `train.txt` and `valid.txt` are decoded and pooled. **`test.txt` is hashed as raw bytes and never decoded,
  split or tokenised.**
- Document: one paper. The member text is split on runs of ≥2 newlines, and each non-empty stripped piece is one
  document. Doc hash = sha256(lower-cased, whitespace-collapsed full text).
- Exact dedup: across all eight candidate domains and both members. A hash that occurs more than once anywhere is
  dropped from every place it occurs, because cross-listed papers have an ambiguous domain.
- Candidate pool: per domain, the exact-dedup survivors in increasing `u("dfcv1-cand", hash)` order, at most
  80,000 documents. Each document keeps its first 20,000 characters (only the first 1,023 tokens are used).
- Near dedup on the union of the candidate pools: MinHash over the normalised first 800 words, with 5-word shingles,
  128 permutations, LSH 32 bands × 4 rows, and estimated Jaccard ≥ 0.8. Union-find clusters are then resolved as
  follows:
  - A cluster that spans more than one domain is dropped entirely.
  - Within a single domain, the document with the smallest `u("dfcv1-cand")` is kept.
- Tokenisation: the project Pythia tokenizer, `tokens[:1023] + EOS` (project convention). Documents shorter than 64
  tokens are dropped. Valid (target) tokens = length − 1.
- Required per domain (valid tokens): draft 6,000,000; main 22,500,000; calibration 1,500,000; validation
  1,000,000; 31.0M in total. **Domain selection rule:** the first five candidate domains in the fixed order whose
  post-dedup, post-filter pool reaches 31.0M valid tokens. Substitutions are recorded in `data_manifest.json` and
  `deviations.jsonl` before any training. Model performance is never used.
- Split by document hash: within each selected domain, documents are ordered by `u("dfcv1-split", hash)` and fill
  buckets sequentially in this order: VALIDATION 1.0M → CALIBRATION 1.5M → DRAFT 6.0M → MAIN-P1 9.0M →
  MAIN-P2 2.25M → MAIN-P3 11.25M. The last document of each bucket is truncated to `tokens[:r] + EOS` so the bucket
  holds exactly its quota, and the remainder of that document is discarded. Every document is in at most one bucket.
  Unused documents are never read again.
- Domain ids in the model are 0–4 in the order of selection.
- Packing: each packed 1,024-token row holds documents of one domain and one bucket only. Documents within a
  (domain, bucket) are shuffled with `rng([order_seed, domain, bucket])` and packed greedily. Rows of the five domains
  are interleaved by sorting on fractional position `(i + 0.5) / n_rows(domain)`, with ties broken by domain id.
  - Main training rows: P1 rows, then P2 rows, then P3 rows (no sequential domain training; every phase interleaves
    domains).
  - `order_seed = 1000 + seed`, identical for every condition within a seed.
  - Evaluation rows: per domain, in split-hash order.
- Calibration subsets:
  - CALIB-MON is the first 64 rows per domain, used for learning curves.
  - CALIB-ABL is the first 48 rows per domain, used for ablations.
  - The full calibration split is used for checkpoint audits.
- Manifests (document hashes per bucket, token arrays, offsets) are hashed and locked (mode 0400) before training.
  Validation is readable only after `frozen.json` exists. The test member is never decoded.

## 3. Draft model (one shared draft, seed 0, frozen before main training)
- Dense causal transformer: 4 layers, d 128, 4 heads, SwiGLU 512, RoPE, tied embedding/LM head (shared token embedding,
  backbone and output head). Same tokenizer.
- Adapters: one residual bottleneck adapter per domain in every layer, applied after the block. The adapter is
  `x + U_d · silu(D_d · RMSNorm_d(x))`, with D: 128→32 (init N(0, 0.02)) and U: 32→128 (init 0). Each token uses only
  its own document's domain adapter.
- Training: the 30M-token DRAFT split, one epoch, balanced interleaved rows, causal next-token CE only. Recipe:
  - Batch: 16 rows × accum 2.
  - Optimiser: AdamW (0.9, 0.95), lr 1e-3, WSD (60 warm-up steps, final 20% linear decay to 1e-4), wd 0.1 on
    parameters with dim ≥ 2, clip 1.0.
  - Precision: bf16 autocast.
- The draft never sees main, calibration (except as a held-out diagnostic), validation or test documents. Its
  checkpoint is hashed. All parameters are frozen (requires_grad False) and the main training never loads it: main
  training reads only the cached forecasts.

## 4. Causal forecasts and cache
- Every position t of every main-training document (and every calibration document, for audits) is eligible. The
  frozen draft with the document's domain adapter greedily generates 4 tokens from the prefix x_0..x_t.
  - The method is a parallel tree: one forward pass over the real prefix, then 3 passes of private continuation
    queries that attend to the real prefix ≤ t and to their own earlier proposals.
  - Tokens after t are never visible.
- Canonical layout: each document sits alone in a 1,024-long row at position 0. Batches are fixed at 64 rows in
  document-index order; the last batch is padded with dummy rows. Computation uses bf16 autocast with deterministic
  algorithms. Output: `uint16 [n_tokens(split), 4]`, aligned with the split's token offsets.
- The cache is indexed by (document, position), so it is invariant to training-time packing by construction. Its
  invariance to cache-batch composition is verified on a sample: the same documents are recomputed under a different
  batch order and size, and must be bitwise equal.
- WRONG-FORECAST cache: the same procedure, with each document's adapter chosen by the fixed derangement
  `d → (d + 1) mod 5`. This was fixed now, before any result. No domain maps to itself.
- Caches: MAIN correct, MAIN deranged, CALIB correct, CALIB deranged. No validation cache exists or is needed.

## 5. Main model (identical in every condition)
- The representative NanoMoE: d 192, 8 heads, context 1,024, 8 layers, layer 0 dense (SwiGLU 768), layers 1–7 MoE
  with 32 routed experts (expert FFN 128) and top-4. Shared expert SwiGLU 256. Tied embeddings, RoPE,
  document-causal attention.
- Routing:
  - fp32 router; gates are the full softmax of the final router logits, top-4 by probability, no renormalisation,
    dropless.
  - Losses: Switch load-balance aux 0.01 and z-loss 0.001, both on the final logits.
  - Every token takes exactly 4 distinct routed experts out of all 32, plus the shared path. Nothing is masked or
    forced.
- Forecast/domain modules, instantiated in every condition. They are created after the backbone from a dedicated
  generator, so the initialisation is identical across conditions within a seed:
  - forecast encoder:
    - inputs: `stopgrad(tok_emb(pred_j)) + pos_j` (pos: learned, 4 × 192);
    - one shared 1-layer GRU (192→192);
    - final hidden → Linear 192→192 (the router input dimension) → `F`;
    - `Fn = RMSNorm(F)`.
  - per MoE layer l:
    - forecast residual `W_f[l]: 192→32`, init N(0, 0.02);
    - informative gate `g_l = sigmoid(w_g[l] · [h_l ; Fn] + b_g[l])`, init w N(0, 0.02), b 0. `h_l` is the
      router input (the current causal token state).
  - domain-group prior `P[l, d, :]`: init 2.0 on the 4 provisional experts of domain d (experts 4d..4d+3), 0
    elsewhere. Experts 20–31 are provisional general experts. These are metadata only and nothing is ever restricted.
  - bounded scales `s_f[l], s_p[l] = 2 · sigmoid(ρ)`, with ρ initialised to ln(1/3) (scale 0.5). Each can be learned
    down to ~0.
- Router logits:
  `logits = token_logits + strength · g_l · (s_f[l] · center(W_f[l] Fn) + s_p[l] · center(P[l, domain]))`.
  `center` subtracts the mean over the 32 experts.
  - LABEL: no forecast; Fn := 0 in the gate and the forecast term is absent.
  - FORECAST: the prior term is absent.
  - SOFT and WRONG: both terms.
  - With strength = 0 (or TOKEN conditions), the extra path is not executed, so logits equal token_logits exactly.
  - No contrastive or domain-classification loss: only next-token CE + aux + z.
- Final checkpoints run with strength 0, no draft, no forecasts and no domain labels. The final state dict also loads
  into the plain `CSRModel` (extra parameters dropped) with identical logits.

## 6. Schedule and conditions
- Main training: exactly 112,500,000 valid target tokens (22.5M per domain), one pass over MAIN.
- Phases are by token bucket:
  - P1 (45.0M, 9.0M per domain): strength 1.
  - P2 (11.25M, 2.25M per domain): strength is linear from 1 to 0. Each row's strength is
    `1 − (P2 valid tokens before the row + half the row's tokens) / 11.25M`.
  - P3 (56.25M, 11.25M per domain): strength 0.
  - Strength is per row, so a step that straddles a phase boundary is exact.
- No reset of weights, optimiser or LR schedule.
- Recipe (every condition):
  - Batch: 16 rows × accum 2 (32 rows of 1,024 per step); the last step may have fewer rows.
  - Optimiser: AdamW (0.9, 0.95), lr 6e-4, WSD (60 warm-up steps; the final 20% of steps decay linearly to 6e-5),
    wd 0.1 on parameters with dim ≥ 2, clip 1.0.
  - Precision: bf16 autocast, fp32 router. The schedule spans each run's own step count.
  - Init: global RNG seed = seed for the backbone, then the dedicated generator for the added modules.
- Conditions (seeds 0, 1 and 2 each):
  1. TOKEN-MAIN: strength 0 throughout (the modules exist but are disabled).
  2. LABEL→TOKEN-S40: prior only (true domain), with the gate on the token state.
  3. FORECAST→TOKEN-S40: correct forecasts with the gate; no prior.
  4. SOFT-SPECIALIST→TOKEN-S40: correct forecasts, gate and prior (the proposed method).
  5. WRONG-FORECAST→TOKEN-S40: as SOFT, but the forecasts come from the deranged adapter. The prior uses the true
     domain.
  6. TOKEN-ALL-DATA: ordinary TOKEN-4 from scratch on DRAFT ∪ MAIN, 142.5M tokens (28.5M per domain).
     - Rows: per domain, DRAFT and MAIN documents together, shuffled with `rng([order_seed, domain, 99])`,
       single-domain rows, interleaved as above.
  7. TOKEN-COMPUTE: ordinary TOKEN-4 with a target-token count N matched to
     `C = mean_seeds(SOFT train-step seconds) + draft train-step seconds / 3 + MAIN-correct cache GPU seconds / 3`.
     - `N = C / mean_seeds(TOKEN-MAIN train-step seconds per valid token)`, rounded up to a multiple of 5,000 tokens.
       It is frozen in `compute_match.json` from pre-validation timings, before validation opens.
     - Data: per domain, the first N/5 valid tokens of that seed's ALL-DATA document stream (the last document is
       truncated), cycling from the start only if N/5 > 28.5M. Rows are packed and interleaved as above.
- Fairness: within a seed, every condition has identical initialisation (verified by hash), MAIN rows and order (1–5),
  optimiser, effective batch and context packing.
- OOM rule: halve micro_batch_rows and double accum, keeping the rows per step exact. No other change.

## 7. Checkpoints and audits (calibration only; diagnostic only)
- Checkpoints: model state saved after the first step whose cumulative tokens reach 22.5M (20%), 45.0M (40%),
  56.25M (50%), 84.375M (75%) and 112.5M (100%, final) for conditions 1–5. ALL-DATA and COMPUTE save their finals
  only. Resume state is saved every 10 minutes and at every checkpoint.
- Learning curves: every 5.625M tokens, CE on CALIB-MON, measured both with the run's current strength ("active") and
  with strength 0 ("neutral"). Every run, including ALL-DATA and COMPUTE, follows this schedule.
- Audit of each checkpoint on the full calibration split:
  - Measurements:
    - CE overall and per domain.
    - Per-layer domain–expert counts from top-4 assignments.
    - MI(domain; expert) per layer, in nats, also normalised by H(domain). A load-preserving shuffled-domain null is
      computed for MI.
    - H(domain | expert) per expert, and p(e | d).
    - Load: max share, coefficient of variation, unused experts, and collapse (any unused expert, or a max share
      > 0.5).
    - Provisional-group mass: the share of domain-d assignments landing on experts 4d..4d+3.
  - Functional ablation on CALIB-ABL. The expert's gate is zeroed for all tokens, and the expert stays selectable.
    - Per (layer, domain): ablate that domain's 4 most-enriched experts (p(e|d)/p(e), from this audit's counts).
    - Per domain, all layers: the most-enriched 4 per layer.
    - Per domain, all layers: the provisional group 4d..4d+3.
    - Specificity = ΔCE(own domain) − mean ΔCE(other domains).
- Modes: at 20% and 40%, curriculum checkpoints are audited twice: (A) normal (curriculum active, strength 1, the
  run's own forecast source) and (B) neutralised (strength 0, no retraining). From 50% on, strength is 0, so only one
  mode exists.
- Cross-seed profile stability (analysis):
  - Index-aligned cosine of per-expert domain profiles between seeds. This is meaningful only with provisional
    anchoring.
  - Best-match cosine, which is permutation-insensitive.
- None of these numbers selects anything. The final checkpoint is always the prescribed final step.

## 8. Final evaluation (once per final checkpoint)
- After all 21 runs and every audit of the training-time diagnostics, `frozen.json` is written. It contains the
  config hashes, final checkpoint hashes, compute match, code hashes and this prereg's hash.
- Each final checkpoint is then evaluated exactly once on the complete VALIDATION split (5.0M tokens) at strength 0,
  with no forecasts and no labels. Outputs: per-document NLL, per-domain CE and expert loads.
- Reported:
  - CE per seed; mean ± seed standard deviation.
  - Paired per-document differences with 95% bootstrap intervals (2,000 resamples over documents; token-weighted),
    per seed and pooled (per-document differences averaged over seeds).
  - Per-domain and worst-domain CE.
  - Learning curves.
  - Specialisation trajectories.
  - Spearman correlation between specialisation (MI or specificity at 40% and 100%) and final CE across runs.
  - Throughput and GPU time; draft and cache overhead, both unamortised and amortised over 3 seeds.
  - Prefill (16 × 1,024) and batch-1 KV-cache decoding throughput for standalone final models.
  - Total and active parameters; loads and collapse.
- Ledgers separate dataset build, draft training, cache, main training, audits, evaluation and analysis.

## 9. Primary comparisons and claim rules (fixed)
- Comparisons: SOFT versus TOKEN-MAIN, LABEL, FORECAST, WRONG, ALL-DATA and COMPUTE. Each is a paired per-document
  difference on validation (negative favours SOFT).
- A comparison is "SOFT better" only if the pooled 95% CI excludes 0 below zero **and** SOFT is better in at least
  2 of 3 seeds. Differences under ~0.003 nats are within known run-to-run noise.
- Claim rules:
  - Specialisation helps only if SOFT beats FORECAST.
  - Forecasts help beyond labels only if SOFT beats LABEL.
  - Semantic forecasting matters only if SOFT beats WRONG.
  - Data efficiency only if SOFT beats ALL-DATA; otherwise the trade-off is stated.
  - Compute efficiency only if SOFT beats COMPUTE.
- Functional specialisation exists at 40% if the mean per-(layer, domain) specificity is > 0 with a bootstrap CI
  excluding 0 (over layer × domain × seed cells) in the neutralised mode. It "survives" at 50%, 75% and 100% under
  the same test.
- Negative results are reported as such.

## 10. Execution
- One GPU job at a time via `scripts/gpu_guarded.sh`.
- A persistent tmux driver (`dfc_driver.py`) with a state file:
  - It resumes after interruption.
  - Bounded retries.
  - OOM → microbatch rule.
  - GPU occupied by others → wait.
  - Disk reserve (100 GiB) → wait.
- Bugs are fixed, documented in `deviations.jsonl`, and every symmetrically affected run is rerun. Invalid runs are
  preserved.
- CPU test suite with CUDA hidden; GPU correctness checks; a one-batch smoke test. No accuracy pilot.
- Protected-file inventory before and after the experiment. No deletion or overwrite of anything existing.
