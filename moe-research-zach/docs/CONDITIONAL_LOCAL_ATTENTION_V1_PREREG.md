# csr-conditional-local-attention-v1 — preregistration

Written 2026-11-08, before any candidate training or any candidate result. The SHA-256 of this file is recorded in
`ledger/safety_ledger.tsv` (event `CLA_V1_PREREG`) and in `artifacts/conditional-local-attention-v1/manifest.json`.
Every config of this experiment carries it (`prereg_sha256`), and every entry point refuses a config whose hash
differs. The earlier experiments (sequence routing, pool routing) only motivate this one. The primary comparison is
against the unrestricted TOKEN-4 model.

## 1. Question

Can the existing TOKEN-4 MoE keep comparable language-model accuracy when full causal attention is replaced for many
tokens by cheaper local attention? Does that give a real end-to-end training or inference speed-up? Only the attention
computation changes. The MoE mechanism, data, budget and optimisation are unchanged.

## 2. Inherited baseline (resolved from `configs/rep-v1/seed0/condition_A.json`; repository is authoritative)

- **Model.** d_model 192, 8 causal layers, context 1,024, 8 heads × 24, RoPE base 10,000, RMSNorm eps 1e-6.
  Block-diagonal document-causal masks.
  - Layer 0 is a dense SwiGLU FFN of width 768. Layers 1–7 are MoE: 32 routed experts with unrestricted top-4
    (global softmax gates, fp32 router, no renormalisation, dropless), expert FFN 128, and an always-active shared
    SwiGLU of width 256.
  - Tied embeddings, vocabulary 50,304 (Pythia tokenizer).
  - 28,873,920 parameters (14,423,232 active per token).
- **Objective.** Next-token CE, plus 0.01 × Switch load balance (per micro-batch, summed over MoE layers, non-pad
  tokens), plus 0.001 × router z-loss.
- **Optimisation.** AdamW: lr 6e-4, betas (0.9, 0.95), weight decay 0.1 for parameters with dim ≥ 2 (0 otherwise),
  eps 1e-8, clip 1.0.
  - WSD schedule: 60 warm-up steps, constant, then linear decay over the final 20% of steps to 6e-5.
  - bf16 autocast with an fp32 router.
  - 2 micro-batches × 16 packed rows × 1,024 tokens per optimizer step.
- **Data.** `artifacts/data/fwe-rep-v1` (manifest `splits` sha256 c28e9845…).
  - Train: 198,012 documents / 126.2M tokens. Calib: 8,179 documents / 5.28M tokens.
  - Greedy sequential packing in a seeded document order. Packing is condition-independent.
- **Seeds.** Init seed = s. Train-order seed = 17 + s (the convention of every earlier final run).

**The baseline is re-trained.** Existing TOKEN-4 checkpoints are not reused, for two reasons:
- This experiment's FULL-TOKEN-4 uses a different, numerically non-identical attention kernel (section 4).
- It carries the parameter-control gate.

So its code semantics are not bit-identical to the earlier runs. The 100M-token TOKEN-4 checkpoint of
csr-speculative-pool-routing-v1 (seed 0) is used, read-only, only as trained reference weights in the GPU numerical
checks (G1–G4).

## 3. Data isolation

- **CALIB-SEARCH / CALIB-SELECTION.** A document-hash division of the existing calib split:
  `u = int.from_bytes(sha256("csr-clav1-calib-partition|" + doc_id)[:8], "little") / 2**64`; CALIB-SEARCH iff
  u < 0.70.
  - Result: 5,722 search documents (3,681,924 valid tokens) and 2,457 selection documents (1,588,219 valid tokens).
  - Written once, before any training: `search_partition.json`, sha256
    `76c8178c371ffd2f8a995a4735ecf4c4d60161d6a308a0d3c39627a7741cec72`.
- **Search decisions** use CALIB-SEARCH only. Training code refers only to CALIB-SEARCH. This is tested statically,
  and dynamically with an audit hook on file opens.
- **CALIB-SELECTION** opens only after `search_frozen.json` exists: the candidate list, the Gen-2 configs and the
  model/training/evaluation code hashes are frozen. It is evaluated once per frozen checkpoint (eval registry).
- **Validation** opens only after:
  - `final_frozen.json` lists the final config hashes; and
  - `final_trained.json` records all six completed final checkpoints.
  Each checkpoint is evaluated once.
- **The locked test split** is never loaded, decoded, inspected, content-hashed or evaluated. `load_raw_split` refuses
  it before any file access, and inventories exclude it by name.

## 4. Conditions (code: `src/csr/cla_model.py`, `src/csr/cla_attn.py`)

**Attention rule for every condition.** Query q may attend to key k iff both are in the same document and k ≤ q.
- **Local attention** adds `q − k < W`, counted in in-document positions. The window therefore restarts at every
  document start, and a token at in-document position < W sees its whole prefix.
- **Pads** attend only to themselves, exactly as before.
- **Q/K/V and output projections** are shared by all attention paths.

| id | condition | layer modes |
|---|---|---|
| FULL-TOKEN-4 | A | full in all 8 layers (gate present, unused = forced full) |
| LOCAL-W128 / -W256 / -W512 | B / C / D | local W in all layers |
| HYBRID-W256-LATE2 | E | layers 0–5 local W256, layers 6–7 full |
| HYBRID-W256-LATE4 | F | layers 0–3 local W256, layers 4–7 full |
| G128-25, G256-10, G256-25, G256-50, G512-25 | G | conditional (W, target rate) in all layers |

### Conditional global attention

Applied in every layer l, for each token t:

- `local_t` = local attention output (pre-output-projection head outputs, d = 192).
- `z_t = MLP_l(rms(stopgrad(local_t)))`, computed in fp32.
  - MLP: Linear(192 → 32) + SiLU + Linear(32 → 1), with biases.
  - `rms` has no parameters.
  - `local_t` uses only keys ≤ t.
- **Eligibility.** A token is eligible iff it is non-pad and its in-document position is ≥ W. Otherwise its local
  window already covers its full prefix, so full == local exactly.
- **Hard decision.** `sel_t = eligible_t ∧ z_t > λ_l`, where λ_l is a per-layer threshold buffer, frozen in
  evaluation and inference.
- **Output.** `out_t = full_t` if sel_t else `local_t`. A selected token gets full attention *instead of* local
  attention (no double counting).
- **Global-query rate.** Selected tokens / all non-pad tokens. Ineligible tokens are never counted as global.

**Compact execution (ordinary steps).**
- Only the selected queries are gathered into a contiguous prefix of a fixed-size query buffer.
- Query blocks without selected queries get no key blocks, so no work is done for them.
- Full attention is computed only for the selected queries, as block-sparse FlexAttention. Dense full attention is
  never computed and then masked.

**Straight-through estimator (documented).**
- Forward: `out_t += (p_t − sg(p_t)) · (full_t − local_t)`, with `p_t = sigmoid(z_t − λ_l)`. The added term is
  exactly 0, so forward values are the hard decisions.
- Gradient: `∂L/∂z_t = ⟨∂L/∂out_t, full_t − local_t⟩ · σ′(z_t − λ_l)`. This is the first-order estimate of whether
  switching the token to full context lowers the loss.
- Where it applies:
  - ordinary steps: selected tokens only, since full_t exists only for them;
  - counterfactual steps: every eligible token.
- The gate gets no other loss term.
- Because the input is stopped, the gate never sends gradient into the backbone.

**Counterfactual steps.** Exactly round(0.10 × n_steps) optimizer steps per run, chosen as the steps with the
smallest `sha256("clav1-cf|<seed>|<step>")`.
- On these steps, full attention is computed densely (block-sparse document-causal kernel) for all tokens.
- The forward pass keeps the hard decisions.
- No validation or selection data is used.

**Budget controller (Lagrangian dual update on the rate constraint; frozen).**
- λ_l is the multiplier of the per-layer rate constraint. Under Lagrangian relaxation, the optimal hard decision is
  "select iff score > λ_l".
- **Initialisation.** On the first training micro-batch, a no-grad pre-pass sets λ_l to the exact threshold that
  selects round(target × non-pad tokens) eligible tokens, layer by layer.
- **Update after every optimizer step.** `λ_l ← λ_l + 0.1 · (q_l − λ_l)`, where q_l is the threshold that would have
  selected exactly round(target × non-pad tokens of the step) eligible tokens of that step (midpoint between the
  k-th and (k+1)-th largest z).
- There is no objective weight to tune. Within a step, decisions use the previous λ, so no in-batch information
  leaks.
- Targets: G128-25 0.25, G256-10 0.10, G256-25 0.25, G256-50 0.50, G512-25 0.25.

**Parameter control.**
- Every condition, including FULL-TOKEN-4, carries the same gate modules: 8 × 6,209 = 49,672 parameters, 0.172% of
  28,873,920. This is below the 0.3% allowance.
- The gates are created after the backbone and initialised from a dedicated generator
  (seed = first 4 bytes of `sha256("clav1-gate-init|<seed>")`; weights N(0, 0.02²), biases 0).
- So within a seed, every condition has an **identical full initial state** (tested), and the backbone equals the
  inherited TOKEN-4 initialisation bit for bit.
- Conditions that do not use the gate never compute it. Its parameters get no gradient, so AdamW skips them.
- Total parameters: 28,923,592 for every condition. Active per token: 14,423,232 backbone, plus 49,672 gate
  (conditional only).

## 5. Implementation (applies to every condition, including the baseline)

**Attention kernel.** All conditions use torch 2.5.1 FlexAttention (compiled Triton).
- BlockMasks are built directly from vectorised block lists: no dense N×N mask is ever evaluated.
- Head dim 24 is zero-padded to 32, which FlexAttention requires. With the explicit scale 1/√24 this is exact.
- Packed rows are flattened, with per-document ids.
- The document-causal block sparsity speeds up the baseline itself. This general-purpose optimisation is applied
  equally to all conditions.

**Reference implementations.** The original CSRModel attention (SDPA with a dense bool mask) and a dense masked
reference backend are kept for tests and checks. For the speed claim, the baseline is always the FASTEST measured
variant: flex, the original dense-mask SDPA, or (prefill only) the flash `is_causal` kernel.

**Engineering measurements before preregistration.** Random initialisation, real train rows, A100, bf16, matched
batch. Two engineering fixes were applied first: permutation gathers, and direct block-list BlockMask construction.

| implementation | training step time |
|---|---|
| original SDPA baseline | 233–245 ms |
| flex FULL-TOKEN-4 | 213–224 ms |
| local W128 / W256 | ≈ 212 ms |
| hybrid LATE2 | 216 ms |
| conditional G256-25 | 266–277 ms |

These numbers are disclosed so that the Gen-1 throughput rule is not applied blindly. They are not results.

## 6. Correctness gates (before any candidate training)

**CPU suite (CUDA hidden).** All pre-existing tests, plus `tests/test_conditional_local_attention_v1.py`, must pass.
The new tests cover protocol items 1–14:
1. Full baseline bit-identical to the original CSRModel (fp64/fp32, logits, losses, experts).
2. Local attention equal to an independent per-query loop reference, for dense and flex backends. Block lists are
   supersets of allowed pairs, and "full" blocks are exactly full.
3. Future-token perturbations leave earlier logits (≤ 1e-12 fp64) and expert / gate decisions (exact) unchanged.
4. Document and padding isolation; the window restarts at document starts.
5. The compact query path equals the dense reference. Compact equals counterfactual forward. Forced-global equals
   full, and forced-local equals local.
6. Hard-gate layer outputs are exactly local or full; the threshold controller behaves as specified.
7. Gate gradients equal the documented straight-through estimator on ordinary and counterfactual steps. No leakage
   to tokens without a full path.
8. Zero gradients from future tokens.
9. Cached incremental decoding equals full-sequence logits and gate decisions.
10. Interrupted / resumed training equals uninterrupted training bit for bit.
11. Identical expert choices, old vs new baseline, at full size.
12. Identical initial weights across all 11 conditions within a seed, with the backbone equal to the inherited
    TOKEN-4 initialisation.
13. Identical steps and valid tokens across conditions; budgets reached with whole steps.
14. Test refused before file access. Split locks and single-use registry enforced. Static and dynamic (audit hook)
    proof that search runs open no `val_*` / `test_*` file.

The frozen selection rules are also unit-tested.

**GPU checks (`cla_gpu_checks.py --formal`).** Every condition at full size, fp32 and bf16:
- **G1** flex vs dense backend: max |Δlogit| < 2e-3 fp32, expert and gate agreement ≥ 0.995.
- **G2** flex FULL-TOKEN-4 vs the original CSRModel:
  - fp32: max |Δlogit| < 2e-3, |ΔCE| < 1e-4, expert agreement ≥ 0.999;
  - bf16: |ΔCE| < 2e-3, mean |Δlogit| < 0.05, expert agreement ≥ 0.98. The original model's own bf16-vs-fp32
    noise floor is recorded alongside.
- **G3** future perturbation: earlier max |Δlogit| < 1e-4, discrete agreement ≥ 0.999.
- **G4** cached decoding vs full sequence: |Δlogit| < 2e-3, argmax and gate agreement ≥ 0.99.
- **G5** bf16 training steps at the real batch and schedule lr (ordinary and counterfactual): finite, gate gradients
  present only for conditional models, no expert collapse, peak reserved VRAM < 56 GiB.
- **G6** profiling: computed tiles scale with the selection rate; no dense or flash attention kernel or N×N buffer
  on the compact path.

## 7. Bounded search (only the 10 candidates above; no others are ever added)

**"Exactly B effective training tokens"** means the smallest whole number of optimizer steps whose valid prediction
tokens reach B. This is identical for every condition within a seed (tested).

**Generation 1** (seed 98, 5M tokens; FULL-TOKEN-4 plus all 10 candidates; CALIB-SEARCH only). Each run records:
- CE, global-query rate (by layer), training throughput, step GPU time, peak VRAM;
- measured attention work (keys attended per token, i.e. theoretical attention FLOPs).

Throughput is valid tokens per second over optimizer steps after the first 10 steps, with CUDA-synchronised step
timing that excludes monitoring evaluations. All runs use the matched training batch.

A candidate is eliminated if:
- it is invalid: incomplete, non-finite, failed correctness, or expert collapse. The existing criterion for collapse
  is an unused routed expert, or one expert taking more than 50% of routed slots, in any MoE layer;
- CE − baseline CE > +0.05; or
- throughput ≤ 0.90 × baseline AND CE − baseline CE > −0.010. That is, a ≥10% slowdown counts as compensated only by
  a gain of at least 0.010 nats/token.

Survivors are ranked by CE. Ranking is iterative: take the lowest remaining CE; among candidates within 0.002 of it,
take the one with the lowest measured step GPU time. At most 4 advance.

**Generation 2** (seeds 98 and 99, fresh identical initialisation, 20M tokens, checkpoints, CALIB-SEARCH).
- Baseline plus the advanced candidates.
- Valid means every seed is valid.
- Rank by mean paired ΔCE (the mean over seeds of candidate − baseline CE). Within 0.003, prefer the higher mean
  training throughput. At most 2 advance.
- Then `search_frozen.json` freezes the candidate list, the Gen-2 config hashes and the model, training and
  evaluation code hashes. Frozen code is re-verified before every later stage.

## 8. CALIB-SELECTION (once per frozen Gen-2 checkpoint, 2 seeds)

Exactly one final candidate is chosen, by applying these rules in order:
1. Smallest mean paired ΔCE vs FULL-TOKEN-4.
2. Within 0.003, the higher measured end-to-end speed: mean Gen-2 matched-batch training throughput, with relative
   differences < 1% counted as ties.
3. Lower global-query fraction: local 0; hybrid = full layers / 8; conditional = measured CALIB-SELECTION rate;
   differences < 0.01 counted as ties.
4. The simpler design: local < hybrid < conditional.

Nothing is modified afterwards.

## 9. Final training, validation, accuracy criteria

**Final training.** FULL-TOKEN-4 and the winner, for seeds 0, 1 and 2, each from the identical untrained
initialisation:
- exactly 100M effective training tokens;
- identical documents and order, batch, optimizer steps, schedule, checkpoint and evaluation frequency;
- no early stopping, no extra tokens.

Fairness is asserted per seed: init parameter hash, training-row hash, tokens and steps. Interrupted runs resume
from their verified `train_state.pt`.

**Validation.** Once, after all six runs are frozen.

**Accuracy measures.**
- ΔCE_s = CE(candidate, s) − CE(FULL-TOKEN-4, s), in nats per valid token.
- Paired per-document bootstrap: 2,000 resamples, seed 20261108, per seed and seed-averaged. The fraction of
  documents improved is also reported.

**Accuracy verdicts.**
- **Comparable:** mean ΔCE ≤ +0.010 AND every seed ΔCE ≤ +0.020 (bootstrap reported; per-seed signs shown).
- **Better:** mean ΔCE ≤ −0.010 AND every seed ΔCE < 0.

## 10. Speed (measured after accuracy is frozen; seed-0 final checkpoints; `cla_bench.py`)

**Training (matched batch).** 2 × 16 × 1,024 rows, bf16; 100 warm-up + 500 timed optimizer steps.
- Reported: tokens/s, steps/s, peak allocated and reserved VRAM.
- The conditional winner includes its 10% counterfactual steps.
- Two primary conditions are repeated in reverse order as a drift check.

**Maximum safe batch.** Reported separately and never substituted for the matched-batch result.

**Inference.** Batch 1 and 8; prompts of 128, 256, 512 and 1,024 tokens from CALIB-SEARCH documents.
- Prefill: 10 warm-up + 100 timed repetitions; latency and tokens/s.
- Cached greedy decoding of 128 tokens: 3 warm-up + 100 timed generations; tokens/s and per-token latency.
- Memory: KV-cache bytes and peak VRAM.
- Condition order is rotated at every point.
- Decoding from 1,024-token prompts reaches positions up to 1,151, beyond the training context. This is a speed
  measurement only.

**Diagnostic controls.** If the winner is conditional, its gate is also forced always-local and always-global.

**Primary speed measures.** Each is expressed as candidate time / fastest-baseline time, so lower is better:
- (a) matched-batch training throughput;
- (b) end-to-end 100M-token training process time (mean over the 3 seeds);
- (c) prefill latency at batch 8 / prompt 1,024;
- (d) cached decoding throughput at batch 8 / prompt 1,024.

**A qualifying speed-up** needs a ratio ≤ 0.90 on at least one primary measure. The remaining grid points are
secondary. Theoretical FLOPs are reported separately and never counted as speed.

**KV cache.** The cache keeps every position in every condition. This experiment authorises no eviction or
compression, so no KV-memory saving is claimed.

## 11. Verdict (exactly one)

- **SUCCESS:** comparable accuracy and a qualifying speed-up.
- **ACCURACY-ONLY:** comparable accuracy, no qualifying speed-up.
- **SPEED-ONLY:** a qualifying speed-up, but accuracy not comparable.
- **NEGATIVE:** neither.
- **INVALID:** fairness, isolation or correctness could not be established.

If no candidate survives Generation 1 or Generation 2, the result is NEGATIVE, and no new candidates are created.

## 12. Resources, stops, deviations

**Hard limit:** 20 GPU-hours for every ledger row with purpose `clav1…`, including the 142 s of engineering probes
and the pre-preregistration dry runs. The driver checks used + projected time before every launch.

**Stop only for:**
- a required change to a frozen rule;
- a need to access the test split;
- a foreign GPU process that persists for more than 2 h;
- disk below 40 GiB;
- the compute limit;
- a correctness test that cannot pass without changing the question.

**Infrastructure failures.** Contamination, interruption or launcher errors are fixed and the affected stage is
rerun or resumed. Failed artifacts are preserved and logged in the append-only
`artifacts/conditional-local-attention-v1/deviations.jsonl`.

**Non-finite runs.** A non-finite candidate run is recorded INVALID; a non-finite baseline stops the experiment.

**Orchestration.** The driver (`cla_driver.py`, tmux session `clav1`) keeps `.tmp/conditional_local_attention_v1/`
{driver.log, state.json, runs.tsv, evals.tsv, failures.tsv}. GPU time goes to `ledger/gpu_time_ledger.tsv`.

## 13. Known limitations (stated in advance)

**Small attention share.** At d_model 192 and context 1,024, attention score computation is a small fraction of
model FLOPs. Attention is ≈ 4% of training step time with the block-sparse kernel, so the room for any end-to-end
speed-up from local attention is small.

**Documents are short.** The mean is ≈ 640 tokens: 80% / 62% / 33% of tokens lie beyond in-document positions
128 / 256 / 512.

**Scope.** This experiment changes attention only. It does not test semantic or domain expert specialisation.
