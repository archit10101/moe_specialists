# csr-sequence-routing-search-v1: preregistration

**Title.** Bounded search for genuinely sequence-routed MoE that beats ordinary token routing.

**Status.** Prospective preregistration, frozen **before any candidate result exists** (no sequence-routing
candidate has been trained or evaluated). Its SHA-256 goes into `docs/SEQUENCE_ROUTING_SEARCH_V1_PREREG.sha256`,
`artifacts/sequence-routing-search-v1/manifest.json`, every config of this experiment (`prereg_sha256`) and the
safety ledger (`SRS_V1_PREREG`). This file is never edited afterwards. Deviations go to
`artifacts/sequence-routing-search-v1/deviations.jsonl` and the report's failures/fixes/deviations section.

The complete user protocol is reproduced verbatim in **Appendix A** and is binding. Sections 1–16 make it concrete;
where they are more specific they implement, and never relax, Appendix A.

## 1. Question

Can sequence routing — one causal route (4 expert ids **and** 4 gate weights) chosen per contiguous segment from
information available before the segment starts, reused for every token of the segment, identically in training and
inference — reach lower held-out next-token cross-entropy than TOKEN-4 under a matched comparison (same backbone,
experts, top-4 + shared expert, tokens, data order, initialization, batch, optimizer, precision, evaluation)?

A null or negative answer is a valid outcome and will be reported as such. The search is bounded (§6–§8) and is not
extended after results are seen.

## 2. Base setup (csr-representative-v1, unchanged)

- Model: d_model 192, 8 layers, 8 heads, RoPE; layer 0 dense (SwiGLU 768); layers 1–7 MoE; 32 routed experts
  (SwiGLU 128) with top-4; one always-on shared expert (SwiGLU 256); tied embeddings; vocabulary 50,304 (Pythia
  tokenizer @ bb1e3e71); context 1024; router `Linear(192, 32, bias=False)` in fp32.
- Data: `artifacts/data/fwe-rep-v1` (FineWeb-Edu; splits hash `c28e9845…`); greedy sequential packing with
  block-diagonal document masks; documents never split across rows; train order `default_rng(17 + seed)`.
- Optimizer: AdamW (0.9, 0.95), weight decay 0.1 on ≥2-D params, clip 1.0; WSD lr 6e-4, 60 warm-up steps, linear
  decay over the final 20% to 6e-5; 16 rows × 1024 per micro-batch, accumulation 2; bf16 autocast, fp32 router;
  load-balance coefficient 0.01, z-loss coefficient 0.001.
- Token budget B: the smallest whole number of optimizer steps whose valid prediction tokens reach B
  (condition-independent step plan; identical rows/order for every condition of a seed).
- Base config: `configs/rep-v1/seed0/condition_A.json`; only `candidate`, `run_name`, `family`, `description` and
  `routing.*` may differ inside a config directory (machine-checked by `srs_make_configs.py`).
- All code is new (`src/csr/seqroute.py`, `srs_run.py`, `srs_select.py`, `srs_*.py`,
  `tests/test_sequence_routing_v1.py`). No existing file is modified. A pre-edit snapshot is at
  `.tmp/srsv1_snapshot_20261106T061354Z/` (sha256 list alongside).

**TOKEN-4** = `SeqRouteModel` in `mode="token"`: per token p = softmax(W_l h_t), top-4 by p, gate = p of the selected
expert (global softmax, no renormalization), dropless, plus the shared expert; Switch load-balance loss and z-loss
over non-pad tokens. It is tested to be bit-identical to the original `CSRModel` token routing. TOKEN-4 is rerun fresh
in every generation and in the final confirmation with this exact code (no earlier checkpoint is reused, because the
software, evaluation sets and monitoring differ from rep-v1).

## 3. Sequence routing (mode "sequence") — frozen implementation (`src/csr/seqroute.py`)

For every MoE layer l (all seven MoE layers are sequence-routed) and every document, the tokens are partitioned into
contiguous segments s_0, s_1, … that never cross a document boundary (packed rows contain whole documents; the first
token of each document always starts a segment).

- **Router input.** h_t = RMSNorm₂(x_t) at layer l, the same tensor the token router reads.
- **First segment (j = 0).** logits = `bos_l` (a learned 32-vector per layer, content-free; init N(0, 0.02²) from a
  dedicated generator). Identical for every document.
- **Later segments (j ≥ 1).** logits = W_l · summary(l, j−1) [+ O_l g_{j−1}], where W_l is the layer's existing
  router matrix applied to a pooled segment summary (never to a single token) and
  - pooling of the **completed** segment j−1: `mean` (mean of h), `last` (h of its final token) or `attn`
    (softmax(q_lᵀh/√d) pooling with a learned query q_l, zero-initialized = mean at init);
  - state across completed segments: `none`; `ema` (e_j = 0.5 e_{j−1} + 0.5 pool_j, e_0 = pool_0, logits = W_l e);
    `gru` (GRUCell(64 → 32) over [W_l pool_j, multi-hot route_j], g_{−1} = 0, logits = W_l pool_j + O_l g_j with
    O_l zero-initialized), reset at every document start.
- **Selection.** p = softmax(logits / T) over 32 experts (T = 1 unless stated); top-4 ids by p; gates = p of the 4
  selected experts (global, no renormalization, same rule as TOKEN-4). The ids and gates are computed once per
  segment and broadcast to every token of the segment; each token executes exactly these 4 routed experts plus the
  shared expert (same dropless dispatch kernel as TOKEN-4).
- **Auxiliary losses.** Switch load-balance loss with f = share of routed slots and P = mean probability, both
  token-weighted over segments (`lb_weighting = token`; `segment` = one weight per segment), and z-loss on the
  segment logits; same coefficients as TOKEN-4.
- **Segmentations** (all causal; decided after observing token t; the new route starts at t+1; never revised):
  - `fixed L` (L ∈ {16, 32, 64, 128, 256}): segment starts at in-document positions 0, L, 2L, …; with opening o:
    0, o, o+L, o+2L, ….
  - `cp` (change point): score_t = 1 − cos(mean src[t−7..t], mean src[t−39..t−8]), windows clipped at the document
    start, valid iff the long window holds ≥ 8 tokens. Boundary after t iff seg_len = 256, or seg_len ≥ 16 and
    score_t > τ. src = h_l (per layer, `boundary_source = layer`) or the residual stream after the dense layer 0
    (`layer0`, one segmentation shared by all MoE layers). Inputs are detached.
  - `lbd` (learned boundary detector): u_t = w_lᵀ sg(h_t) + b_l (193 params per layer); boundary after t iff
    seg_len = 256, or seg_len ≥ 16 and u_t > τ. Training only: u_t regresses the standardized label
    1 − cos(mean h[t+1..t+16], mean h[t−15..t]) (MSE, coefficient 1, gradients reach only w_l, b_l). The label uses
    future tokens **only as a training target**; the boundary decision and route never read it (tested: identical
    routes in train and eval mode, and future-perturbation invariance).
  - opening o = 16: the first segment of every document is forced to end after token 15 (fixed and dynamic).
  - Dynamic limits: minimum 16, maximum 256 (fixed by protocol).
- **Threshold τ** (cp, lbd; one per boundary source). A buffer, part of the checkpoint. Initialized by a no-grad
  pre-pass on the first training micro-batch (layer by layer: the (1 − 1/target) quantile of that source's valid
  scores), then adapted only **between** optimizer steps from the already-completed step's closed segments:
  m = mean length of segments ended by a boundary decision; cp: τ ← τ·exp(0.2·log(target/m)), lbd:
  τ ← τ + 0.2·log(target/m), step clipped to ±0.5. Every forward pass (training or evaluation) uses a fixed τ;
  τ is frozen at evaluation/inference. Reported mean segment lengths are measured, not assumed.
- **Incremental decoding.** `sr_step` processes one token for B documents in lockstep, keeping per-layer running
  pools, the current route, state, window histories and τ; it reproduces the full-sequence routes and logits
  (tested in fp64 on CPU and fp32 on GPU). Training, evaluation, CALIB-SELECTION, validation and latency use the same
  routing code; there is no token-routing fallback anywhere in sequence mode.

**Extra parameters** (full size, measured; base TOKEN-4 = 28,873,920): bos only 224 (0.0008%); + attn query
1,568 total; + lbd detector 1,575 total; + GRU/output 73,472 total (0.254%). The 0.5% limit (≈ 144k) is enforced by
test; a candidate above it would be rejected (no parameter-matched control is planned).

## 4. Evaluation isolation

- **Partition** (`artifacts/sequence-routing-search-v1/search_partition.json`, sha256
  `81454b91c0be330f6ae77f8b03a2fa494d9ba16317ae033aa25bdcd532fb104a`): calib document i is in **CALIB-SEARCH** iff
  int.from_bytes(sha256("csr-srsv1-calib-partition|" + doc_id)[:8], little) / 2⁶⁴ < 0.70, else **CALIB-SELECTION**.
  CALIB-SEARCH: 5,763 documents, 3,738,549 valid tokens; CALIB-SELECTION: 2,416 documents, 1,531,594 valid tokens.
- CALIB-SEARCH: all development, monitoring (first 128 packed rows) and generation ranking.
- CALIB-SELECTION: opened only after candidate generation ended and `search_frozen.json` (hashes of the frozen
  TOKEN-4/best-fixed/best-dynamic Gen-3 configs) exists; one evaluation per checkpoint (registry lock).
- Validation split (existing `val`, 16,850 docs, 10,732,175 tokens): opened only after `final_frozen.json`
  (hashes of all final configs) exists and all final training is complete; one evaluation per checkpoint. No
  validation monitoring during training.
- Test split: never loaded, hashed, decoded or evaluated; loaders refuse `test` before file access.
- No candidate or rule is revised after CALIB-SELECTION or validation results are observed.

## 5. Correctness tests and GPU checks (before any search run)

CPU suite with CUDA hidden (`tests/`, all pre-existing tests plus `tests/test_sequence_routing_v1.py`) must pass and is
recorded in `artifacts/sequence-routing-search-v1/cpu_tests_<stamp>.json`. It covers: identical ids and gates within
segments; a changed token cannot change its segment's already-selected route; future text cannot change earlier
boundaries/routes/outputs; later routes depend on the completed previous segment; document isolation; content-free
first segment; incremental = full (B = 1 and batched); same routing in train and eval mode; exactly 4 distinct routed
experts per token; the router is never applied to token-level tensors; TOKEN mode = original token router; split
locks and selection code isolation; matched backbone initialization across all candidates within a seed; extra
parameters < 0.5%; finite losses and gradients; the dynamic boundary scan equals the token-by-token rule; segment
length limits; τ controller direction and eval-time freezing; resume of an interrupted run reproduces the
uninterrupted run exactly (CPU); selection-rule determinism.

GPU checks (`srs_gpu_checks.py`, full-size model, real data) for TOKEN-4 and all 12 Generation-1 candidates: finite
bf16 loss/gradients with gradients reaching the sequence-router parameters; per-segment identical ids/gates, exactly 4
experts, BOS first route (fp32, TF32 off); future perturbation on real CALIB-SEARCH documents (logits ≤ 1e-4, routes
and boundaries exact); incremental decoding (B = 1 and B = 4) with exact routes and logits within 2e-3. A failure is
a causality/implementation stop: diagnose, fix with a regression test, preserve the failed artifact, rerun.

Minimal implementation fixes are allowed at any time; they are logged as deviations with regression tests. They
never change a frozen rule, criterion, candidate definition or budget.

## 6. Generation 1 (broad screening; fixed list)

Seed 98 (train order 115), 5M tokens, CALIB-SEARCH only, plus TOKEN-4 at the same budget/seed.

| id | family | segmentation | summary / state |
|---|---|---|---|
| G1-F16 | fixed | fixed 16 | mean / none |
| G1-F32 | fixed | fixed 32 | mean / none |
| G1-F64 | fixed | fixed 64 | mean / none |
| G1-F128 | fixed | fixed 128 | mean / none |
| G1-F256 | fixed | fixed 256 | mean / none |
| G1-F32-GRU | fixed | fixed 32 | mean / gru (+ previous route) |
| G1-F32-ATTN | fixed | fixed 32 | attn / none |
| G1-CP32 | dynamic | cp per layer, target 32 | mean / none |
| G1-CP64 | dynamic | cp per layer, target 64 | mean / none |
| G1-CP128 | dynamic | cp per layer, target 128 | mean / none |
| G1-CP64-L0 | dynamic | cp on dense-layer-0 output (shared), target 64 | mean / none |
| G1-LBD64 | dynamic | learned boundary detector, target 64 | mean / none |

Seven fixed-family and five dynamic-family candidates (G1-F32-GRU and G1-F32-ATTN are the summary variants; with
G1-F32 they form the "F32 summary probe"). Not selected, by design: the existing MiniLM semantic boundaries (they need
an external encoder at inference and showed no advantage in csr-multiscale-routing-v1).

## 7. Generation 2 (deterministic rule; `srs_select.gen2_candidates`)

Inputs: Generation-1 CALIB-SEARCH results only. A candidate is **valid** iff its run completed, CE is finite, it has
no collapse (no unused routed expert in any MoE layer and no expert above 50% of routed slots in any layer, on
CALIB-SEARCH) and extra parameters ≤ 0.5%. Within each family, valid candidates are ranked by CALIB-SEARCH CE.
The F32 summary probe orders the summary configurations {mean/none, mean/gru, attn/none} by CE.

Per family, with b1 = best and b2 = second best:
1. b1 unchanged (carry); 2. b2 unchanged (carry);
3. b1 with a 16-token opening segment if applicable (fixed L > 16 or dynamic), otherwise b1 with state = ema;
4. b1 with its summary configuration replaced by the best F32-probe summary configuration different from its own
   (if that duplicates an existing direction, the next one).
A family with fewer valid directions is filled from the other family (its b2..b4 with a 16-token opening, then
carries of b3..b6). At most 8 candidates. Seeds 98 and 99 (train order 115, 116), 10M tokens, CALIB-SEARCH only,
TOKEN-4 per seed.

## 8. Generation 3 (finalists) and search freeze

Within each family, the two valid Generation-2 candidates with the lowest mean CALIB-SEARCH CE over seeds 98 and 99
(both seeds required). Seeds 97, 98, 99; 30M tokens; checkpoints saved; TOKEN-4 per seed. No new candidate is created
in Generation 3. **Candidate modification ends permanently after these runs.**

Freeze: best fixed and best dynamic = lowest 3-seed mean CALIB-SEARCH CE among valid finalists of the family (among
candidates within 0.002 nats of the lowest, lower mean GPU time). `search_frozen.json` records them and the hashes of
the TOKEN-4/best-fixed/best-dynamic Gen-3 configs (all three seeds); `calib_search_results.json` and
`candidates.tsv` are written at this point.

Limits: ≤ 3 generations, ≤ 24 unique sequence-routing candidates (this design creates at most 12 + 8 = 20), never
expanded.

## 9. Blind CALIB-SELECTION and promotion

After the freeze, the frozen Gen-3 checkpoints of TOKEN-4, best fixed and best dynamic (seeds 97–99; 9 checkpoints)
are evaluated once on CALIB-SELECTION. Nothing is modified afterwards. Ranking: 1) CALIB-SELECTION CE (3-seed
mean); 2) absence of collapse/unused experts; 3) lower GPU time only within 0.002 nats. Each family's frozen
candidate is **promoted** to final confirmation unless it is invalid (failed, non-finite, collapsed, >0.5% params) or
catastrophically worse (3-seed mean CE − TOKEN-4 > +0.10 nats). A non-promoted family is preserved and reported, never
replaced. If neither family is promoted, the study stops with search-stage results.

## 10. Throughput and resource gate

Seconds per training token for each promoted condition are measured from the Gen-3 runs (process GPU seconds / valid
training tokens, the maximum over seeds; includes start-up and the CALIB-SEARCH evaluation, so it is conservative).
Overhead: 120 s per validation evaluation + 1,800 s (latency + slack). The final budget is the largest of 100M, 50M,
25M tokens per run such that used GPU seconds + 1.2 × projected seconds ≤ 20 GPU-hours, with three seeds (0, 1, 2)
and the same budget for every condition; if none fits, stop with search-stage results. Replication is never reduced.
Also required: peak reserved VRAM < 56 GiB (Gen-3 peaks), ≥ 80 GiB free disk before every launch, no persistent
foreign GPU process (wait up to 2 h, then stop), no run overwrites.

## 11. Final confirmation, primary outcome and criteria

Train from scratch TOKEN-4, best fixed, best dynamic (promoted ones) with seeds 0, 1, 2 (train order 17, 18, 19),
identical initialization within a seed (backbone parameter hash checked), identical data/order, valid-token budget
and effective batch. After all final training, each checkpoint is evaluated once on the full validation split.

**Primary metric:** validation next-token CE (nats per valid token). For each sequence candidate: Δ = CE_seq − CE_TOKEN
per seed; mean and SD over seeds; paired per-document bootstrap of the seed-averaged per-document NLL difference
(token-weighted ratio, 2,000 resamples, `default_rng(20261106)`), which is not seed uncertainty. Per-seed bootstraps
are reported too.

**"Sequence routing is substantially better"** requires all of: mean Δ ≤ −0.02; Δ < 0 in all three seeds; bootstrap
95% CI entirely below 0; no collapse (validation, any seed); no more active experts (4 + shared, by construction and
test); matched training tokens; ≤ 10% additional analytic training FLOPs vs TOKEN-4; identical routing algorithm in
training and inference.

**"Sequence routing is accuracy-comparable"** requires all of: mean Δ ≤ +0.01; every seed Δ ≤ +0.01; bootstrap upper
bound < +0.01.

These criteria are not weakened after results are seen. CALIB-SEARCH gains are never reported as final evidence.

## 12. Secondary outcomes (reported, not used for decisions)

Perplexity; training curves (train loss, CALIB-SEARCH head CE); validation CE by document (file) and by in-document
position (0–16, 16–64, 64–256, 256–1024) and by in-segment position; expert load (max share, CV, usage entropy), unused
experts; routing entropy; segment counts, mean/closed lengths, length quantiles, boundaries and route changes per 1,000
tokens, consecutive-route Jaccard (route stability), first-segment (BOS) token share; router invocations; dispatch
max/mean load; training tokens/s, validation tokens/s; prefill latency (B = 1 1000-token document; B = 16 packed rows),
batch-1 decoding latency (256 tokens, 3 documents), batched decoding (B = 16 lockstep, 256 tokens); peak VRAM; total
GPU time (ledger, purpose `srsv1 …`); extra router parameters; analytic FLOPs (`srs_run.flops_per_token`); latency
uses the same causal implementation, includes the boundary detector, and uses CALIB-SEARCH documents.

## 13. Compute and parameter limits

20 GPU-hours total for everything tagged `srsv1` (checked before every launch); a candidate using > 10% more training
compute than its matched TOKEN-4 is disclosed and not called compute-matched; extra router parameters must stay
≤ 0.5% of base parameters.

## 14. Prohibited shortcuts (binding; see Appendix A)

No token-dependent expert selection inside a segment; no token/chunk hybrid called sequence routing; no token routing
in training with sequence routing only at inference; no validation/test inspection during iteration; no favorable
seed selection; no discarding of completed runs; no extra training tokens, active experts, backbone or expert size;
no training on validation documents; no tokenizer/data change; no future segment information in routes; no lowering
of criteria; no redefinition of the token baseline; no CALIB-SEARCH gain reported as final evidence; no search beyond
the stated limits.

## 15. Execution and stop conditions

`srs_driver.py` in tmux session `srsv1`: verifies and skips completed stages, preserves failed runs, never overwrites,
atomic status file (`.tmp/sequence_routing_v1/status.json`), GPU-ownership check before every launch, disk check
between runs, terminates on frozen gates. Recoverable failures: a contaminated run (foreign GPU process) is preserved
and rerun; an interrupted/aborted run with saved state is resumed; a candidate that produces non-finite loss twice is
recorded INVALID and reported. Stop only for: causality failure; protected-artifact mismatch; persistent foreign GPU
occupation; < 80 GiB disk; projected/consumed GPU time > 20 h; inability to preserve evaluation isolation; a needed
change outside the search space; or completion of the bounded search with no valid candidates.

## 16. Interpretation and known limitations declared in advance

- Interpretation follows Appendix A ("Interpretation"). No claim that unbounded search would succeed.
- Run-to-run GPU nondeterminism at 5M tokens was measured in csr-multiscale-routing-v1 (within-seed SD ≈ 0.0026
  nats). Generation-1 rankings among close candidates are therefore noisy; this is accepted and disclosed.
- The first segment of every document uses a content-free route by definition; with long segments this route covers
  a large share of tokens (documents average ~640 tokens). This is a property of the tested mechanism, reported via
  the BOS token share and CE-by-position.
- τ initialization uses a no-grad pre-pass on the first training micro-batch (a data-dependent parameter
  initialization, like any learned parameter); every forward uses a fixed τ.
- The LBD label uses future tokens as a training target only (§3).
- Analytic FLOPs treat attention with a mean context of 512 and count the CPU boundary scan as 0 FLOPs; measured GPU
  time is reported alongside.

---

## Appendix A — user protocol (verbatim)

Work autonomously in ~/moe-research and conduct the complete experiment below.

Do not pause for routine confirmations. You are authorized to inspect and modify the project, add tests and configs, train models, perform bounded architecture and training-scheme search, fix ordinary implementation bugs, resume interrupted runs and analyze the results.

Run long work in a named, resumable tmux session. Do not ask for approval between successful stages.

Experiment name:

csr-sequence-routing-search-v1

### Objective

Determine whether genuinely sequence-routed MoE training and inference can achieve lower held-out language-model loss than ordinary token-routed MoE under a fair, matched comparison.

Search and iterate aggressively within the preregistered search space, but do not manufacture a positive result.

If no sequence-routing configuration beats token routing within the frozen search and compute budget, report that honestly.

### Definition of sequence routing

A condition qualifies as sequence routing only if:

1. One expert route is selected for an entire contiguous sequence segment.
2. The selected routed-expert identities are identical for every token in that segment at a given MoE layer.
3. The routing weights are also sequence-level, not independently recalculated from each token.
4. The route is computed only from information available before the routed segment begins.
5. Training and inference use the same causal routing rule.
6. No token-dependent expert-selection fallback is used.
7. No future token, completed current-segment representation, future boundary or future segment length influences the route.
8. All tokens still receive ordinary next-token language-model loss.

The model remains autoregressive and predicts every token. “Sequence routing” means expert selection is reused over a segment; it does not mean predicting an entire segment simultaneously.

For segment j at MoE layer l:

summary(l,j-1) = causal summary of the completed previous segment

route(l,j) = TopK(sequence_router_l(summary(l,j-1)))

The route is reused for every token in segment j.

For the first segment of a document, use a learned or fixed preregistered BOS/default sequence route derived without document content.

### Token-routing baseline

TOKEN-4:

- existing standard token router;
- each token independently selects global top-4 routed experts;
- existing shared expert remains active;
- same csr-representative-v1 backbone and training recipe.

The final comparison must use a freshly verified TOKEN-4 baseline under the same software, data, token budget, seeds, effective batch size and evaluation pipeline as the sequence-routing finalists.

Existing TOKEN checkpoints may be used only if every relevant hash, config, data order, training token count and software behavior matches exactly. Otherwise rerun them.

### Base model

Start from the exact csr-representative-v1 architecture:

- d_model 192;
- eight Transformer layers;
- layer 0 dense;
- layers 1–7 MoE;
- 32 routed experts;
- top-4 routed experts;
- existing shared expert;
- tied embeddings;
- context length 1024;
- same expert dimensions;
- same tokenizer;
- same FineWeb-Edu data;
- same optimizer and learning-rate schedule;
- same precision and ordinary auxiliary losses.

The primary sequence-routing comparison must retain:

- the same experts;
- top-4 active routed experts;
- the same shared expert;
- the same backbone;
- the same training tokens;
- the same context length;
- the same initialization within each seed.

Do not claim an accuracy gain obtained by using more experts, a larger model, more training tokens or additional data.

### Protected artifacts

Do not modify, delete, overwrite or reinterpret previous:

- experiments;
- checkpoints;
- data;
- segmentation caches;
- reports;
- manifests;
- preregistrations.

Snapshot files before editing because the project has no Git history.

Keep the existing test split locked throughout this experiment.

Do not download new datasets, models or encoders.

### Evaluation isolation

Before experimentation, deterministically divide the existing calibration documents into:

- CALIB-SEARCH: 70%;
- CALIB-SELECTION: 30%.

Use a stable document-hash partition and record its hash.

Rules:

- CALIB-SEARCH may be used throughout iterative development.
- CALIB-SELECTION may be used only once, after all candidate generation and modification have ended.
- The existing validation split may be used only after final configurations are frozen and hashed.
- The test split must never be accessed.
- No candidate may be revised after CALIB-SELECTION or validation results are observed.

### Fixed-segment family

Search fixed lengths:

- 16;
- 32;
- 64;
- 128;
- 256 tokens.

Fixed segments must:

- respect document boundaries;
- never combine two documents;
- select the next segment’s route from the previous completed segment;
- use the same segmentation during training and inference.

### Dynamic-segment family

Also search causal dynamic segmentation.

Dynamic segmentation must:

- have minimum segment length 16;
- have maximum segment length 256;
- decide after observing token t whether the current segment ends after t;
- apply the new route beginning with token t+1;
- never revise an earlier boundary;
- never use future text;
- use the same causal algorithm during training and inference;
- include boundary-detector runtime in performance measurements.

Allowed dynamic families include:

1. Existing prefix-causal semantic boundary signals, only if causality is independently verified.
2. A learned causal boundary detector operating on already-observed hidden states.
3. Change-point detection over causal hidden-state summaries.
4. Boundary thresholds targeting mean segment lengths near 32, 64 or 128.

Do not use an offline bidirectional boundary decision during training or inference.

### Allowed sequence summaries

Candidate sequence routers may use:

- mean of the previous completed segment;
- the previous segment’s final hidden state;
- attention pooling with a learned query over the previous completed segment;
- a small GRU updated once per completed segment;
- a causal exponential moving state across completed segments;
- the previous route plus the previous segment summary.

The summary must remain sequence-level and causal.

### Allowed training-scheme changes

Claude may iterate over:

- fixed versus dynamic segment definitions;
- segment length;
- sequence-summary function;
- sequence-router temperature;
- router learning-rate multiplier;
- load-balancing strength;
- router entropy regularization;
- expert dropout;
- route momentum or hysteresis;
- sequence-level router warmup;
- straight-through or differentiable top-k estimators;
- initialization of sequence routers;
- auxiliary losses applied to sequence-level routing;
- per-layer sequence summaries;
- per-layer segment lengths, provided routing remains sequence-level;
- capacity factors and overflow handling;
- gradient clipping and router-specific optimization settings.

Every candidate must still use sequence-level expert selection during both training and inference.

Sequence-level warmup may use soft or stochastic sequence routes, but it may not use token-dependent expert identities. No training phase may secretly switch to token routing.

At most four routed experts may execute for any token. A dense all-expert forward pass is not allowed as the normal training path.

### Prohibited shortcuts and cheating

Do not:

- use token-dependent expert selection inside a sequence;
- call a token/chunk hybrid “sequence routing”;
- use token routing during training and sequence routing only at inference;
- inspect validation or test data while iterating;
- select favorable seeds;
- discard unfavorable completed runs;
- give sequence models more training tokens;
- increase active experts;
- increase backbone or expert size;
- train on validation documents;
- alter tokenizer or data;
- use future segment information;
- lower the success criterion after seeing results;
- redefine the token baseline;
- report CALIB-SEARCH gains as final evidence;
- continue searching indefinitely until a chance winner appears.

If additional router parameters exceed 0.5% of base-model parameters, construct a parameter-matched token-routing control or reject that candidate.

Record total training FLOPs and GPU time. Any candidate using more than 10% additional training compute relative to its matched TOKEN run must be disclosed and cannot be called compute-matched.

### Preregistration

Before examining new candidate results, write and hash:

docs/SEQUENCE_ROUTING_SEARCH_V1_PREREG.md

It must contain:

- this complete protocol;
- data partitions;
- candidate-generation limits;
- promotion rules;
- compute limits;
- success criteria;
- non-inferiority criteria;
- causality requirements;
- final evaluation rules;
- prohibited shortcuts.

Ledger its hash before training.

### Correctness tests

Add tests proving:

- all tokens in a segment receive identical expert identities;
- all tokens in a segment receive identical sequence-routing weights;
- changing a token within a segment cannot change that segment’s already-selected route;
- changing future text cannot change earlier boundaries or routes;
- the next segment may depend on the completed previous segment;
- routes never cross document boundaries;
- first-segment routing uses no document content;
- incremental decoding reproduces full-sequence routing;
- training and inference use the same routing algorithm;
- exactly top-4 routed experts execute per token;
- no token-dependent routing path is active;
- no validation or test data enters candidate selection;
- candidate and baseline initialization match within each seed;
- future tokens never alter earlier outputs;
- gradients and losses are finite;
- interrupted runs resume reproducibly.

Run the full CUDA-hidden CPU suite and focused GPU correctness tests.

Minimal implementation fixes are authorized. Preserve failed artifacts, add regression tests and continue automatically.

### Search budget

Use at most three candidate generations and at most 24 unique sequence-routing candidates in total.

Do not expand this limit after observing results.

#### Generation 1: broad screening

Evaluate up to 12 preregistered candidates:

- at least five fixed-length candidates;
- at least five dynamic candidates;
- one or two additional sequence-summary/training variants.

Use:

- search seed 98;
- 5M training tokens per candidate;
- CALIB-SEARCH only;
- matched TOKEN-4 at the same budget.

Retain complete results for every candidate.

#### Generation 2: informed iteration

Using only Generation-1 CALIB-SEARCH results, generate at most eight candidates within the allowed search space.

Include:

- the four best fixed-family directions;
- the four best dynamic-family directions;

or the nearest available balance if a family has fewer valid candidates.

Use:

- seeds 98 and 99;
- 10M tokens per candidate;
- CALIB-SEARCH only;
- matched TOKEN-4 controls.

Do not inspect CALIB-SELECTION or validation.

#### Generation 3: finalists

Choose at most:

- two fixed-sequence candidates;
- two dynamic-sequence candidates.

Train each for:

- seeds 97, 98 and 99;
- 30M tokens;
- CALIB-SEARCH only.

Candidate modifications end permanently after these runs.

Freeze and hash:

- best fixed candidate;
- best dynamic candidate;
- TOKEN-4 configuration.

### Blind calibration selection

After candidate generation ends:

1. Evaluate frozen TOKEN-4, best fixed and best dynamic checkpoints on CALIB-SELECTION.
2. Do not modify any configuration afterward.
3. Promote the best fixed and best dynamic configurations to final confirmation.
4. If one family is invalid or catastrophically worse, preserve and report it rather than inventing a replacement.

Selection ranking:

1. calibration CE;
2. absence of collapse or unused experts;
3. lower GPU time only as a tie-breaker within 0.002 nats/token.

### Throughput and resource gate

Before final confirmation, benchmark measured throughput and choose the largest uniform final budget that fits under 20 total GPU-hours, including GPU time already consumed and a 20% safety margin:

1. 100M tokens per final run;
2. otherwise 50M;
3. otherwise 25M;
4. otherwise stop with search-stage results.

Keep three final seeds. Do not reduce replication to preserve a larger per-run budget.

Requirements:

- peak reserved VRAM below 56 GiB;
- at least 80 GiB free disk;
- no persistent foreign GPU process;
- no run overwrites.

### Final confirmation

Train from scratch:

1. TOKEN-4
2. best fixed sequence-routing candidate
3. best dynamic sequence-routing candidate

Use:

- main seeds 0, 1 and 2;
- identical initialization within each seed;
- identical training data and order;
- identical valid-token budget;
- identical effective batch size;
- full existing validation split exactly once after training.

Do not modify any candidate after validation is opened.

The test split remains locked.

### Primary outcome

Primary metric:

validation next-token cross-entropy in nats per valid token.

For each sequence candidate calculate:

sequence CE minus TOKEN-4 CE

Negative values favor sequence routing.

Use:

- per-seed differences;
- mean and standard deviation across seeds;
- paired per-document bootstrap;
- 2,000 deterministic bootstrap resamples.

Do not treat the document bootstrap as seed uncertainty.

### Success criteria

“Sequence routing is substantially better” requires:

- mean CE improvement of at least 0.02 nats/token;
- beneficial sign in all three main seeds;
- paired-bootstrap confidence interval entirely below zero;
- no expert collapse;
- no more active experts;
- matched training tokens;
- no more than 10% additional training compute;
- the same routing algorithm during training and inference.

“Sequence routing is accuracy-comparable” requires:

- mean CE degradation no greater than 0.01 nats/token;
- every seed within 0.01;
- paired-bootstrap upper bound below +0.01.

Do not weaken these criteria after observing results.

### Secondary outcomes

Report:

- perplexity;
- training curves;
- validation CE by document;
- expert load and unused experts;
- routing entropy;
- segment lengths and boundary statistics;
- route changes per 1,000 tokens;
- training tokens/second;
- validation tokens/second;
- prefill latency;
- batch-1 decoding latency;
- batched decoding latency;
- router invocations;
- expert dispatch sizes;
- peak VRAM;
- total GPU time;
- additional router parameters;
- training FLOPs;
- route stability;
- specialization diagnostics;
- performance near topic and document transitions.

Use the identical causal sequence-routing implementation in latency measurements.

### Interpretation

Possible conclusions:

- Fixed sequence routing beats TOKEN:
  reusing causal expert routes can improve language modeling.

- Dynamic beats fixed:
  adaptive causal boundaries add useful information.

- Fixed beats dynamic:
  predictable regular segments are sufficient or easier to optimize.

- Sequence is comparable but faster:
  sequence routing is a viable efficiency mechanism.

- Sequence is comparable but not faster:
  router calls were not the bottleneck.

- Sequence is worse:
  tokens require finer expert allocation than the tested sequence routers provide.

- Search gains disappear on CALIB-SELECTION or validation:
  the iterative search overfit CALIB-SEARCH.

- No candidate meets the success criteria:
  report that the bounded search found no evidence that sequence routing beats token routing.

Do not imply that continued unbounded search would necessarily succeed.

### Outputs

Create:

- docs/SEQUENCE_ROUTING_SEARCH_V1_PREREG.md
- configs/sequence-routing-search-v1/
- artifacts/sequence-routing-search-v1/search_partition.json
- artifacts/sequence-routing-search-v1/candidates.tsv
- artifacts/sequence-routing-search-v1/calib_search_results.json
- artifacts/sequence-routing-search-v1/calib_selection_results.json
- artifacts/sequence-routing-search-v1/final_analysis.json
- SEQUENCE_ROUTING_SEARCH_V1_REPORT.md
- checkpoint/config/hash inventory;
- GPU-time ledger;
- candidate genealogy;
- rejected-candidate table;
- failures, fixes and deviations section;
- resumable driver status.

The final report must answer:

1. Did any sequence-routing candidate beat token routing on CALIB-SEARCH?
2. Did the gain survive blind CALIB-SELECTION?
3. Did it survive three-seed validation?
4. Did fixed or dynamic sequence routing perform better?
5. Which segment length and summary worked best?
6. Did sequence routing meet the 0.02-nat substantial-improvement rule?
7. Was it at least non-inferior?
8. Did it reduce or increase measured training and inference time?
9. Did any candidate exploit additional parameters, compute or information?
10. Did expert collapse occur?
11. Were all routes genuinely sequence-level and causal?
12. What was total GPU time, wall time, peak VRAM and remaining disk?
13. What negative results are established?
14. What remains unresolved?

### Autonomous execution

Create a fail-fast, resumable tmux driver before long-running work.

The driver must:

- verify and skip completed stages safely;
- preserve failed runs;
- never overwrite artifacts;
- write an atomic status file;
- check GPU ownership before every launch;
- check disk between runs;
- terminate on frozen scientific gates;
- continue automatically through recoverable failures.

Do not ask for routine approval after preregistration, tests, candidate generations, promotion, blind calibration selection or final confirmation.

Stop only for:

- causality failure;
- protected-artifact mismatch;
- persistent foreign GPU occupation;
- fewer than 80 GiB free disk without safe continuation;
- projected or consumed GPU time above 20 hours;
- inability to preserve evaluation isolation;
- a necessary change outside the allowed search space;
- completion of the bounded search with no valid candidates.

Begin with read-only inspection, hashing, preregistration and correctness tests. Then execute the complete bounded search and final confirmation autonomously.
