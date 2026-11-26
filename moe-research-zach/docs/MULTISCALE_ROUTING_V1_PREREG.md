# csr-multiscale-routing-v1: preregistration

**Title.** Causal multiscale context-conditioned token routing.

**Status.** Exploratory, prospective follow-up to csr-representative-v1 and specialization-audit-v1. Frozen
before any implementation, calibration, pilot or main-training result exists. Its SHA-256 is recorded in
`docs/MULTISCALE_ROUTING_V1_PREREG.sha256`, in `artifacts/multiscale-routing-v1/manifest.json` and in the ledger
(`MS_V1_PREREG`). This file is never edited after calibration begins. Deviations go to a separate, hashed
deviations log (`artifacts/multiscale-routing-v1/deviations.jsonl`) and the final report. Implementation-only
clarifications made before calibration go to a separately hashed amendment; this file stays unchanged.

**What this does not claim.** The experiment tests whether causal historical context improves token-level expert
selection. It does not:
- implement shared chunk-level expert pools;
- guarantee subject-specialized experts;
- reduce the active expert count;
- accelerate inference;
- claim architectural novelty or generality beyond this setup.

Every token keeps unrestricted global top-4 selection over all 32 routed experts, plus the existing shared expert.
No detached hard candidate pool and no semantic or contrastive auxiliary loss are used.

**Prior results.** All earlier results are unchanged and are not reinterpreted:
- csr-representative-v1:
  - B − A: +0.110 nats;
  - D − F: mixed sign;
  - E − D: +0.026;
  - C − B: +0.022.
- specialization-audit-v1: category 4.

## 1. Questions

1. Does causal broad context improve held-out next-token cross-entropy (CE)?
2. Is broad context more useful in late MoE layers than in early ones?
3. Can local context help independently of broad context?
4. Does combining local and broad context help?
5. Do semantic boundaries outperform fixed and matched-random boundaries?
6. Are apparent gains explained by router-logit scaling?
7. Does functional specialization improve without harming language-model quality?

## 2. Base setup (identical to csr-representative-v1)

**Model**
- d_model 192, 8 layers, 8 heads, RoPE.
- Layer 0 is dense; layers 1–7 are MoE.
- 32 routed experts (FFN 128) with global top-4 and one always-on shared expert (FFN 256).
- Tied embeddings; vocabulary 50,304 (Pythia tokenizer @ bb1e3e71); context length 1024.
- About 29M total / 14.4M active parameters.

**Data**
- `artifacts/data/fwe-rep-v1`, train / calib / val, with splits hash `c28e9845…`.
- Packing is greedy and sequential with block-diagonal document masks. Documents are never split across rows.

**Optimizer**
- AdamW with β = (0.9, 0.95), weight decay 0.1, clip 1.0.
- WSD schedule: lr 6e-4, 60 warm-up steps, linear decay over the final 20% to 6e-5.
- 16 rows × 1024 tokens per micro-batch, accumulation 2.
- bf16 autocast, fp32 router.
- Load-balance coefficient 0.01, z-loss coefficient 0.001.

**Budgets**
- A token budget B means the smallest whole number of optimizer steps whose valid prediction tokens reach B. The
  step plan is condition-independent.
- At 100M this gives 3,861 steps and about 100.007M tokens, as in rep-v1.

**Matching within a seed**
- Identical initial weights (`init.seed = seed`) and train order (`train_order_seed = 17 + seed`).
- Identical optimizer initialization and valid-token budget.
- Segmentation changes only `chunk_ids` and `chunk_pos`.

**Seeds**
- Main: 0, 1, 2.
- Calibration: 98, 99, with train order 115 and 116.

**Locked test split.** It is never loaded, hashed, decoded or evaluated. New loaders refuse `split == "test"`
before any file access.

**Protected artifacts.** These were hashed before implementation
(`artifacts/multiscale-routing-v1/protected_inventory.json`, 317 files, test files excluded by name) and are
re-verified at the end:
- the data;
- the semantic cache `artifacts/segments/8e7edc7e14b3c880`;
- the F-v3 random cache `artifacts/segments_random/0c57f77e51ea3734`;
- the rep-v1 configs, checkpoints and analyses;
- the audit artifacts, docs, reports and all pre-existing source files.

## 3. Router definitions (mode "multiscale")

**Per MoE layer l ∈ {1..7} and token t:**
- h_l,t = RMSNorm_2(x_l,t): the same post-norm hidden state the ordinary router reads.
- z_token = W_l h_l,t.
- The router is `nn.Linear(d, 32, bias=False)`. **It has no bias, and this is recorded.** Context projections
  therefore use W_l only, with no bias term.
- No new parameters and no context-specific routers.

**Local context**
- n_loc(t) = min(chunk_pos(t), 16).
- m_local = mean of h over the first n_loc tokens of t's current chunk (chunk positions 0 … n_loc − 1).
  - This excludes t and all future tokens.
  - After the 16th chunk token it stays at the mean of the first 16 chunk tokens.
- c_local = W_l m_local.
- If n_loc = 0, then c_local = 0.
- A chunk never crosses a document.

**Broad context**
- n_br(t) = min(pos_in_doc(t), 1024).
- m_broad = mean of h over the n_br in-document tokens immediately preceding t. This excludes t and never crosses
  documents.
- c_broad = W_l m_broad, multiplied by the ramp r(t) = min(n_br / 512, 1).
- If n_br = 0, then c_broad = 0.
- Gradients flow through m_broad and m_local. Neither is detached.

**Centering.** c ← c − mean over the 32 experts of c, for each contribution separately.

**Combined logits.** z_raw = z_token + a_l·α·c_local + b_l·β·r·c_broad, where a_l, b_l ∈ {0, 1} come from the
condition's layer schedule.

**Implementation**
- Masked prefix sums along the row in float64 (shifted cumulative sums), cast to fp32.
- The router projection runs in fp32 with autocast off, as for z_token.

**Selection (unchanged FLAME rule)**
- p = softmax(z_route) over all 32 experts; top-4 by p; gate = p of the selected expert (global, no
  renormalization).
- Dropless dispatch; shared expert added.
- The load-balance loss uses p and the selected-expert fractions. The z-loss uses logsumexp(z_route). z_route is
  the logit vector actually used for selection (z_raw, z_scale or z_token).

**Incremental decoding (`step`).** Per layer, a running document sum (and a window of up to 1024 states) and a
running sum of the first ≤ 16 chunk states, reset at chunk_pos = 0. These give exactly the same z_route.

**Characterization.** Pooling is applied after RMSNorm, and W is linear and bias-free. So
W·mean(h) = mean(W·h) in exact arithmetic: **the mechanism is multiscale routing-logit smoothing**, a centered,
ramped average of the layer's own historical router logits. It is not equivalent to normalizing pooled pre-norm
states. The numerical approximation error is measured, not assumed:
- in the tests;
- in the GPU checks;
- on every main checkpoint (max and mean absolute difference between W·mean(h) and mean(W·h) on a fixed calib
  subset).

## 4. Layer schedules and conditions

MoE layers are numbered 1–7.

| Condition | Layers 1–2 | Layer 3 | Layer 4 | Layers 5–7 | Boundaries |
|---|---|---|---|---|---|
| TOKEN | token | token | token | token | none |
| BROAD-EARLY | broad | broad | token | token | none |
| BROAD-LATE | token | token | token | broad | none |
| LOCAL-{FIXED,SEMANTIC,RANDOM} | token | local | local | local | fixed / semantic / random |
| HIER-{FIXED,SEMANTIC,RANDOM} | token | local | local | local + broad | fixed / semantic / random |

- BROAD-EARLY changes layers 1–3 and BROAD-LATE changes layers 5–7: three layers each.

**Boundaries.** All are frozen, and only their starts arrays are read:
- FIXED: `{split}_fixed.npz` of cache 8e7edc7e14b3c880 (128-token chunks restarting at every document). This is
  checked to equal `fixed_starts(L, 128)`.
- SEMANTIC: `{split}_semantic.npz` of the same cache (MiniLM teacher, tau 0.35, min 64, max 256).
- RANDOM: `{split}_random.npz` of the accepted F-v3 cache 0c57f77e51ea3734.
- **Known property, recorded as a limitation, artifact not modified:** the F-v3 closer conditions its content-free
  hazard on document length and remaining distance. Its boundaries therefore carry a small amount of
  document-length information.

**Scaling controls (detached, context-dependent, centered-RMS).** For the reference condition's hypothetical z_raw
(same layers, coefficients, boundaries, examples and computation):
- s = RMS_E(center(z_raw)) / max(RMS_E(center(z_token)), 1e-6), computed per token and detached.
- z_scale = mean_E(z_token) + s · center(z_token).
- No contextual direction is added. s > 0, so expert rankings and top-4 sets equal TOKEN's, except for exact ties.

| Control | Reference | Applied at |
|---|---|---|
| SCALE-ONLY-LOCAL | LOCAL-SEMANTIC | layers 3–7 |
| SCALE-ONLY-BROAD-LATE | BROAD-LATE | layers 5–7 |
| SCALE-ONLY-HIER | HIER-SEMANTIC | layers 3–4 (token + semantic-local RMS) and 5–7 (token + semantic-local + broad RMS) |

These controls match only the centered RMS. They do not match:
- the softmax confidence profile;
- top-k gate margins;
- gate-weight distributions;
- gradient magnitudes or optimization behaviour.

Beating a control shows an advantage beyond that RMS-matched control only. It does not eliminate every
confidence-related explanation.

**Config fields allowed to differ between main conditions:** `condition`, `run_name`, `segmentation.strategy`,
`routing.schedule`, `routing.scale_control`, `routing.reference`. In calibration, `routing.alpha` and
`routing.beta` also differ across candidates. Any other difference fails the matched-config guard.

**In-training monitoring**
- Every 5M tokens, CE on the first 128 packed **calib** rows.
- Main runs only: also the first 128 packed **val** rows, as a validation learning curve. These monitoring values
  are never used for any decision.
- Calibration and pilot runs never read the val split.

## 5. Semantic-boundary causality gate (before any calibration)

Run on GPU 0 with the pinned MiniLM teacher in fp32, as the cache was built.

**G1. Online reproduction**
- A deterministic sample of 200 train, 200 calib and 200 val documents (`default_rng(20261102)`) is re-segmented
  with `OnlineSemanticChunker`. Tokens are pushed one at a time, and each completed sentence is embedded from its
  observed tokens only.
- The starts are compared with the cache.
- Pass: every document matches. A mismatch is tolerated only if the recomputed decision similarity at the differing
  event is within 1e-5 of tau (a numerical tie). Every such case is listed.

**G2. Availability.** For every reproduced boundary b:
- the chunker first reports b while consuming token b − 1, and not earlier;
- b − 1 is a completed sentence end, or b is a forced max-length close.

**G3. Future perturbation**
- For 100 calib documents, cut at a deterministic sentence end c, keep the document length, and replace all tokens
  after c + 1 with tokens from another document.
- Boundaries ≤ c + 1 must be unchanged.

**G4. Model invariance**
- A full-size random-init HIER-SEMANTIC model (α = β = 1, fp32, TF32 off) runs on original and perturbed documents,
  each re-segmented online.
- Logits, z_route and top-4 at positions ≤ c must agree (atol 1e-5; top-4 exact).

**G5. Incremental decoding**
- Tokens are streamed into `step()` while the online chunker supplies chunk_pos from past tokens only.
- Outputs must equal the full-sequence forward on the same document (atol 1e-4 fp32; top-4 exact).

**G6. Future chunk structure**
- Altering only boundaries after position t (endpoints and lengths of later chunks) leaves all outputs ≤ t
  unchanged.

**Failure.** If G1–G6 fail, the experiment stops before training (terminal blocker) and the boundaries are not
redefined.

## 6. Tests (CPU, CUDA hidden) and GPU correctness checks

**New file:** `tests/test_multiscale_v1.py`. It covers at least the items listed in the task:
- α = β = 0 equals TOKEN exactly;
- no bias in context projections;
- measured linear equivalence;
- SCALE-ONLY preserves rankings and top-4, matches centered RMS, and has detached s;
- three-layer early/late schedules;
- local and broad causal exclusion of current and future tokens; document isolation;
- the local ≤ 16 cap and the broad ≤ 1024 cap;
- prefix-reproducible semantic boundaries; invariance to future text;
- incremental equals full;
- exactly 4 distinct routed experts; no candidate pool;
- gradient reach, finiteness and matched configs;
- test-split refusal;
- the calibration aggregation and branch/coefficient selection functions;
- deterministic restart.

**Then:**
- The full CPU suite runs, existing tests unchanged; everything must pass.
- `ms_gpu_checks.py` runs the tiny-model equivalences and the full-size forward on GPU.

**Resume semantics.** An interrupted run is preserved and rerun from step 0 in a new run directory. Determinism of
a restart is tested on CPU.

## 7. Neutral coefficient calibration (training + calib data only)

**Shared design**
- Candidates {0, 0.25, 0.5, 1.0}.
- Seeds 98 and 99 (identical initialization and examples across candidates within a seed).
- 5M valid training tokens (the first steps of the seed's train order).
- Evaluated on the **full calib split**: 8,179 documents, 5,270,143 valid tokens, with each run's own routing and
  boundaries, under bf16 autocast.
- Every cell is run literally, including the zero candidates.

| Calibration | Runs per seed | Total runs | Fixed coefficient |
|---|---|---|---|
| β | {BROAD-EARLY, BROAD-LATE} × 4 candidates | 16 | α = 0 |
| α | {LOCAL-FIXED, LOCAL-SEMANTIC, LOCAL-RANDOM} × 4 candidates | 24 | β = 0 |

**Per candidate.** The aggregate is the mean calib CE over its constituent runs (both seeds and all placements or
boundary types).

**Rejection.** A candidate is rejected for any of the following in any constituent run:
- a non-finite loss or CE, or a non-COMPLETE status;
- collapse: any MoE layer with > 50% of calib top-4 assignments on one expert;
- method-attributable unused experts: an unused expert in a layer where the same seed's same-placement zero
  candidate has none;
- a run more than 0.10 nats worse than the same seed's same-placement zero run.

**Qualification of a nonzero candidate.** All of:
1. not rejected;
2. aggregate ≤ zero aggregate − 0.002;
3. in each seed, the seed-level aggregate difference (candidate − zero, averaged over that seed's runs) is < 0;
4. no placement or boundary type is catastrophically harmed. Catastrophic harm means a two-seed mean difference
   vs zero > +0.05 nats.

**Choice.** The lowest aggregate among qualifying candidates. Ties (|Δ| < 1e-4) go to the smaller coefficient.
With none qualifying, the coefficient is 0.

α and β are calibrated independently. If both are nonzero they are frozen as selected and never jointly retuned.
Every candidate and constituent result is saved to `artifacts/multiscale-routing-v1/calibration.json`.

## 8. Automatic branches

| Branch | Coefficients | Conditions run | Primary scaling comparison |
|---|---|---|---|
| A | α = 0, β = 0 | **Stop** before main training; report the negative calibration (terminal) | — |
| B | α > 0, β = 0 | TOKEN, SCALE-ONLY-LOCAL, LOCAL-FIXED, LOCAL-SEMANTIC, LOCAL-RANDOM | LOCAL-SEMANTIC − SCALE-ONLY-LOCAL |
| C | α = 0, β > 0 | TOKEN, SCALE-ONLY-BROAD-LATE, BROAD-EARLY, BROAD-LATE | BROAD-LATE − SCALE-ONLY-BROAD-LATE |
| D | α > 0, β > 0 | TOKEN, SCALE-ONLY-HIER, BROAD-EARLY, BROAD-LATE, LOCAL-FIXED, LOCAL-SEMANTIC, LOCAL-RANDOM, HIER-FIXED, HIER-SEMANTIC, HIER-RANDOM | HIER-SEMANTIC − SCALE-ONLY-HIER |

All main conditions use the frozen α and β. The schedule decides where each applies.

## 9. Throughput-based budget selection (before the pilot; no main validation result exists)

**Benchmark.** Each active condition runs for 20 guarded steps. Measured:
- tokens/s after warm-up step 10;
- peak allocated and reserved VRAM;
- disk growth.

**Projected total GPU time.**
- Consumed audit-free GPU seconds, from the ledger entries of this experiment.
- Plus the pilot: n_cond × (10M / tps_c + overhead).
- Plus main: 3 × Σ_c (B / tps_c + overhead + val_eval).
- Plus an analysis allowance of **3.0 GPU-h**.
- Multiplied by **1.2**.
- Overhead is the median measured calibration-run (process wall − training loop) time. val_eval is the median
  measured calib-eval time × (10.73M / 5.27M), plus 30 s.

**Budget choice.** The largest B ∈ {100M, 50M, 25M} with:
- projection ≤ 20 GPU-h;
- peak reserved < 56 GiB;
- free disk after projected growth ≥ 80 GiB.

If none fits, there is no replicated main study (terminal).

**Fixed rules**
- The seed count is never reduced.
- If VRAM requires it, only the micro-batch is reduced and accumulation increased (same effective batch, all
  conditions).
- The choice is frozen and ledgered (`MS_V1_BUDGET`) in `feasibility.json` before the pilot.

**Interpretation of the budget**

| Budget | Meaning |
|---|---|
| 100M | intended study |
| 50M | resource-limited replicated study |
| 25M | extended replicated pilot |

## 10. Matched pilot

**Setup.** Seed 0, 10M valid tokens, every active condition, identical order. Full calib-split CE only.

**Futility stop.** Stop only if all three hold:
1. every active *contextual* condition (one that adds context direction; SCALE-ONLY controls are not contextual)
   has pilot calib CE more than 0.05 nats worse than TOKEN;
2. no contextual condition has a more negative final-20% training slope than TOKEN. The slope is the OLS slope of
   logged training LM CE against tokens over steps ≥ 0.8·n_steps;
3. no correctness issue explains the result.

**No winner selection.** No condition is chosen from small pilot rank differences. Output:
`artifacts/multiscale-routing-v1/pilot.json`.

## 11. Main study and primary analysis

**Runs**
- Each active condition × seeds {0, 1, 2} at the frozen budget, in the order of §8 within a seed.
- The final checkpoint and resolved config are saved.
- Full-validation evaluation (16,850 docs, 10,732,175 valid tokens) runs after every run.
- Disk is checked before every condition, and GPU ownership immediately before every GPU command.

**Primary outcome.** Full-validation CE (nats/token), with perplexity reported for readability.

**Contrasts.** Row − column. Only those whose conditions exist in the selected branch are run; negative favours the
row.

| # | Contrast |
|---|---|
| 1 | BROAD-LATE − TOKEN |
| 2 | BROAD-LATE − BROAD-EARLY |
| 3 | BROAD-LATE − SCALE-ONLY-BROAD-LATE |
| 4 | LOCAL-SEMANTIC − TOKEN |
| 5 | LOCAL-SEMANTIC − SCALE-ONLY-LOCAL |
| 6 | LOCAL-SEMANTIC − LOCAL-FIXED |
| 7 | LOCAL-SEMANTIC − LOCAL-RANDOM |
| 8 | HIER-SEMANTIC − SCALE-ONLY-HIER |
| 9 | HIER-SEMANTIC − BROAD-LATE |
| 10 | HIER-SEMANTIC − LOCAL-SEMANTIC |
| 11 | HIER-SEMANTIC − HIER-FIXED |
| 12 | HIER-SEMANTIC − HIER-RANDOM |

**Statistics**
- Per seed, the token-weighted CE difference over identical documents and valid tokens (equal counts are asserted).
- Three-seed mean, SD and range.
- Paired document bootstrap: 2,000 resamples from `default_rng(0)`. Each resample draws one document index vector
  and applies it to all seeds, which gives:
  - the interval of the **three-seed mean** difference;
  - per-seed intervals, reported separately.
- Document-bootstrap intervals are not seed uncertainty.

**Practical support of a contrast.** All of:
- the three-seed mean difference is ≤ −0.01 nats/token;
- the difference is negative in all three seeds;
- the 95% bootstrap interval of the three-seed mean difference excludes 0 (upper bound < 0).

**Interpretation rules (fixed)**

| Result | Reading |
|---|---|
| BROAD-LATE beats TOKEN | broad causal history helps |
| BROAD-LATE beats BROAD-EARLY | placement matters |
| LOCAL-SEMANTIC beats TOKEN | local context helps |
| LOCAL-SEMANTIC beats LOCAL-FIXED and LOCAL-RANDOM | semantic boundaries add value |
| HIER-SEMANTIC beats BROAD-LATE and LOCAL-SEMANTIC | combining timescales helps |
| A context condition beats its SCALE-ONLY control | an advantage beyond that RMS-matched control only |
| A context condition matches its SCALE-ONLY control | consistent with an RMS/confidence explanation |
| Semantic matches fixed and random | no measurable semantic value |
| All contextual conditions lose | contextual routing is harmful here |
| Specialization rises while CE worsens | specialization not useful |

Specialization counts as useful only with improved or non-inferior CE. Non-inferior means a three-seed mean
difference ≤ +0.005 with the upper interval bound < +0.01.

## 12. Secondary analyses (descriptive; never change the primary conclusions)

**Budget and order.** A 3.0 GPU-h cap. If projected over the cap, the deterministic reductions are applied in
the order (d2) window cap, (d2) layers, (c) seeds.

**(a) Router diagnostics** per main checkpoint, on the first 256 packed calib rows. Per layer:
- the mean and centered RMS of z_token, the context contribution and z_route;
- softmax entropy;
- selected-gate margin (4th − 5th probability; top1 − top2);
- the top-4 gate sum;
- the fraction of top-4 sets differing from the same model's z_token top-4;
- expert load, unused experts and the maximum share;
- the linear-equivalence error.

Training logs supply router LM-gradient norms (every 25 steps), training and validation learning curves, and
throughput.

**(b) Validation per-token CE** (saved by the evaluator), broken down by:
- document position;
- available broad-history length (0, 1–15, 16–63, 64–255, 256–511, 512+);
- tokens 1–16 and 17–64 after teacher semantic (non-forced) boundaries vs matched positions after fixed and random
  boundaries.

The last breakdown is descriptive only, because the boundaries are teacher-defined.

**(c) Latency** (seed-0 checkpoint of every condition; 3 warm-up and 10 measured repetitions):
- prefill at batch 1 and batch 8 × 1024;
- autoregressive `step()` decoding of 256 tokens at batch 1.

**(d) Specialization** on the specialization-audit-v1 audit documents, with their frozen e5 K = 32 and TF-IDF K = 32
labels, discovery and effect halves, and native segmentations. Topic–expert information is measured as I(T;E) on
expected gate mass, using the audit's implementation:
- load-preserving permutation null (1,000 permutations; seed 20261101);
- topic probe from routing (macro-F1);
- cross-seed Hungarian alignment;
- functional single-token counterfactual expert ablation at blocks 1, 4, 7 with the audit's stratified ATT
  estimator. Ablations are capped deterministically to fit the analysis allowance.

**Excluded uses.** The val split is used only for primary CE and these descriptive breakdowns. Test is never
used.

## 13. Execution, safety and terminal blockers

**Driver**
- `ms_driver.py` runs in tmux session `msv1`.
- It records atomic per-stage completion (with artifact hashes, verified before skipping), a machine-readable
  status and a heartbeat (`.tmp/multiscale_v1/`).
- It preserves failed and partial runs and never overwrites a run.

**Before every GPU command**
- A GPU 0 foreign-process check. If occupied, wait 5 min and retry for up to 2 h, then stop as a terminal blocker.
- One project GPU process at a time.

**Limits**
- Peak reserved VRAM < 56 GiB (in-process guard).
- ≤ 5,400 s per process.
- Disk ≥ 80 GiB free before every training condition (hard), with a warning below 100 GiB.
- ≤ 20 GPU-h in total; target ≤ 12.

**Ordinary failures** (implementation bug, failed test, recoverable OOM, interruption):
1. preserve;
2. diagnose;
3. make the smallest correction;
4. add a regression test;
5. record it in the deviations log;
6. resume.

**Terminal blockers**
- both coefficients fail calibration (Branch A);
- the pilot futility rule;
- semantic-causality failure;
- a protected-hash mismatch;
- a necessary unauthorized scientific change;
- persistent foreign GPU occupation (> 2 h);
- < 80 GiB disk with no safe continuation;
- inability to stay < 20 GPU-h;
- inability to guarantee test isolation.

## 14. Outputs

**Docs and configs**
- `docs/MULTISCALE_ROUTING_V1_PREREG.md` and its `.sha256`.
- `configs/multiscale-routing-v1/`.

**Artifacts** in `artifacts/multiscale-routing-v1/`:
- `manifest.json`, `protected_inventory.json` and its verification;
- `semantic_gate.json`, `calibration.json`, `feasibility.json`, `pilot.json`, `analysis.json`;
- calibration and main TSV inventories;
- `deviations.jsonl`.

**Ledgers.** GPU time in `ledger/gpu_time_ledger.tsv` (purpose prefix `msv1`) and safety entries prefixed `MS_V1_`.

**Report.** `MULTISCALE_ROUTING_V1_REPORT.md`, answering the 15 required questions and ending in a terminal state.

**Seeds and RNG**
- Model/data seeds as in §2.
- Bootstrap `default_rng(0)`.
- Gate sampling `default_rng(20261102)`.
- Audit permutations `default_rng(20261101)`.
