# csr-conditional-local-attention-v1 — final report

**Terminal state: driver `CLA V1 COMPLETE` (2026-11-08T05:51:32-05:00). Verdict: NEGATIVE.**

## Summary

**The question.** Can unrestricted TOKEN-4 keep comparable accuracy with cheaper local or conditional attention, and
does that give a real speed-up?

**Accuracy: no.** The preregistered search selected LOCAL-W128: causal 128-token sliding-window attention in every
layer.
- At the full 100M-token budget it is **worse than FULL-TOKEN-4 by +0.0245 nats/token** on validation.
- Per seed: +0.0227, +0.0264, +0.0244, every seed worse.
- Paired per-document bootstrap 95% CI: [+0.0238, +0.0252].
- This fails the comparability criterion (mean ≤ +0.010 and every seed ≤ +0.020).

**Speed: no.**
- None of the four preregistered primary speed measures improves by 10%. Matched-batch training is 0.8% faster,
  end-to-end 100M training 0.4% faster, and batch-8 decoding 0.2% faster.
- Prefill is 14% *slower* than the fastest baseline kernel.
- Local attention removes 71% of the attention-score work in theory. But attention is a small part of this model's
  compute, and the block-sparse kernel already makes full attention cheap.

**The search results were misleading early.**
- Every local and hybrid candidate was *better* than full attention at 5M and 20M tokens (by 0.008–0.028 nats).
- That early advantage reversed between about 45M and 75M tokens.
- The final deficit is concentrated at long in-document positions.

**Learned conditional global attention never paid off.** It was implemented with real compact execution, but it
trained 21–23% slower than the baseline and gave no accuracy advantage over fixed local attention.

All numbers come from `artifacts/conditional-local-attention-v1/`:
- `final_analysis.json` (the machine-readable analysis);
- `gen1_results.json`, `gen2_results.json`, `calib_selection_results.json`;
- `bench_20261108T105054Z.json`;
- `evals/`.

Rendered tables: `.tmp/conditional_local_attention_v1/report_tables.md` (`clav1_report_tables.py`).

---

## 1. Was comparable accuracy achieved?

**No.** On validation (3 seeds, 100M tokens, 16,850 documents, 10,732,175 valid tokens), the mean ΔCE of LOCAL-W128
vs FULL-TOKEN-4 is **+0.0245 nats/token**.
- Every seed exceeds both the +0.010 mean threshold and the +0.020 per-seed threshold.
- The direction is the same in all three seeds.
- The "better" criterion is not met either.

## 2. Validation CE per seed, and mean

| | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| FULL-TOKEN-4 | 4.2178 | 4.2273 | 4.2212 | 4.2221 |
| LOCAL-W128 | 4.2404 | 4.2537 | 4.2456 | 4.2466 |
| ΔCE (LOCAL − FULL) | **+0.0227** | **+0.0264** | **+0.0244** | **+0.0245** |

There was no expert collapse in any final run: no unused expert, and the maximum expert share is far below 50%.

## 3. Paired confidence intervals

The bootstrap resamples documents (2,000 resamples, seed 20261108). It measures token-weighted ΔCE, with the same
documents and token counts in both conditions.

| | seed 0 | seed 1 | seed 2 | seed-averaged |
|---|---|---|---|---|
| 95% CI | [+0.0219, +0.0235] | [+0.0256, +0.0272] | [+0.0236, +0.0252] | [+0.0238, +0.0252] |
| documents improved | 39.3% | 36.1% | 37.6% | 36.6% |

The document bootstrap is narrower than the between-seed spread (SD of ΔCE across seeds ≈ 0.002). Both
uncertainty measures put the deficit at roughly +0.022 to +0.027.

### Where the loss comes from: validation CE by in-document position (mean of 3 seeds)

| position | 0–128 | 128–256 | 256–512 | 512–768 | 768–1024 |
|---|---|---|---|---|---|
| FULL-TOKEN-4 | 4.3169 | 4.1773 | 4.1738 | 4.2185 | 4.2512 |
| LOCAL-W128 | 4.2910 | 4.1820 | 4.2099 | 4.2645 | 4.3230 |
| Δ | **−0.026** | +0.005 | +0.036 | +0.046 | **+0.072** |

- LOCAL-W128 is better on the first 128 tokens of each document, where its window is the whole prefix.
- It is increasingly worse as more context beyond the window becomes available.

### Learning curve (monitor CE on the first 128 CALIB-SEARCH rows; LOCAL − FULL)

| tokens (M) | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|
| seed 0 | −.028 | −.016 | −.004 | −.015 | −.019 | −.022 | −.005 | +.008 | +.020 | +.020 |
| seed 1 | −.022 | −.019 | −.015 | −.010 | −.009 | −.009 | +.004 | +.022 | +.020 | +.024 |
| seed 2 | −.021 | −.005 | −.003 | +.000 | +.009 | +.028 | +.019 | +.020 | +.019 | +.020 |

The 5M- and 20M-token search budgets saw only the early, favourable phase. This is the same short-budget trap seen
in csr-speculative-pool-routing-v1. The search rules were followed as preregistered; they were not designed to catch
a reversal after 20M tokens.

## 4. Global-query rates by layer and token position

**The final winner is a fixed local model.** Its global-query rate is 0 by construction.

**Conditional candidates at Generation 1** (CALIB-SEARCH, 5M tokens):

| candidate | target rate | measured rate |
|---|---|---|
| G128-25 | 0.25 | 0.126 |
| G256-10 | 0.10 | 0.054 |
| G256-25 | 0.25 | 0.124 |
| G256-50 | 0.50 | 0.271 |
| G512-25 | 0.25 | 0.130 |

**Why measured rates fell short.** Rates are fractions of all non-pad tokens; tokens at in-document positions
below W are never eligible.
- The preregistered dual-update controller (λ ← λ + 0.1(q − λ)) lags the gate logits.
- The logits drift fast in short runs. For example, the G256-25 layer-1 threshold moved from −0.02 to −0.73 within
  194 steps.
- So realised rates fell below their targets. At 20M tokens the gap closed: G256-10 measured 0.104 and 0.091 on
  CALIB-SEARCH.
- The controller was frozen before results, so it was not changed. In effect, the conditional candidates were
  tested at lower global rates than their names indicate.

**Post-hoc diagnostic** (not preregistered; informs no decision). This used the Gen-2 G256-10 checkpoints on
CALIB-SEARCH; output in `posthoc_gate_20261108T105306Z.json`.

| | seed 98 | seed 99 |
|---|---|---|
| mean global rate | 0.104 | 0.091 |
| among eligible tokens | ≈ 0.15–0.19 | ≈ 0.12–0.16 |
| per layer (0–7) | 0.092, 0.096, 0.116, 0.103, 0.098, 0.118, 0.089, 0.117 | 0.101, 0.092, 0.090, 0.099, 0.078, 0.077, 0.100, 0.089 |

Fraction of tokens requesting full context, by in-document position (seed 98 / seed 99, averaged over layers):

| position | rate |
|---|---|
| 0–256 (ineligible) | 0 / 0 |
| 256–512 | 0.175 / 0.153 |
| 512–768 | 0.163 / 0.143 |
| 768–1024 | 0.157 / 0.138 |

The gate did not learn a strong layer or position preference. Its rates are close to uniform across layers and
positions.

## 5. Which tokens requested full context

From the same post-hoc diagnostic: eligible occurrences, both seeds, tokens with at least 200 eligible layer-queries.

- **Highest request rates (0.33–0.38).** Discourse and topic-level content words, mostly scientific and expository:
  "earchers", " galaxies", " estimates", " drawbacks", " findings", " legend", " indications", " Scientists",
  " examines", " advances", " discoveries", " astronomy", "Despite", " advantages". These are tokens whose
  continuation depends on what the document is about.
- **By preceding token.** Requests are highest right after "There" (0.51), " examines", " aimed", "Given",
  "Despite", " highlights", "Studies".
- **Lowest (< 0.015).** Mid-word subword pieces and byte fragments: "fn", " spl", " Kur", " Hein", " et",
  " cephal", " hypot", " Guang", and broken UTF-8 bytes. Local context determines these completely.

This is a sensible learned policy, but it brought neither an accuracy advantage over fixed local attention nor
speed.

## 6. Training speed: matched batch and maximum batch

**Matched-batch benchmark.** The training batch is 2 × 16 rows × 1,024 tokens; 100 warm-up + 500 timed optimizer
steps on the seed-0 checkpoints; bf16, A100 80 GB; the order was repeated in reverse as a drift check.

| condition | tokens/s | steps/s | median step ms | peak alloc / reserved GiB |
|---|---|---|---|---|
| FULL-TOKEN-4 (flex, this experiment) | 134,654 (repeat 134,695) | 5.200 | 192.4 | 18.8 / 24.0 |
| FULL-TOKEN-4, original SDPA dense mask | 121,492 | 4.692 | 213.3 | 19.0 / 24.1 |
| LOCAL-W128 | 135,759 (repeat 135,787) | 5.242 | 190.9 | 18.8 / 20.8 |

- Time ratio vs the fastest baseline: **0.992** (0.8% faster).
- End-to-end 100M-token training process time, mean of 3 seeds: 798.1 s FULL vs 794.5 s LOCAL, ratio **0.996**.
- Optimizer-step time only: 767.4 s vs 764.1 s.

**Maximum safe batch** (separate measure; 32 rows per micro-batch, accum 1):

| condition | tokens/s | peak reserved |
|---|---|---|
| FULL-TOKEN-4 | 147,668 | 41.8 GiB |
| LOCAL-W128 | 149,415 | 41.7 GiB |

This is +1.2%. The next size tried, 48 rows, did not fit within the 0.9 × 56 GiB safety ceiling for either condition.

**Conditional candidates during search** (in-run matched-batch throughput, relative to FULL):

| | ratio to FULL |
|---|---|
| Gen-1 conditional candidates | 0.774–0.793 |
| Gen-2 G256-10 | 0.768 |
| Gen-1 local / hybrid candidates | 0.986–1.014 |

## 7. Prefill and decoding speed

**Prefill** (median latency over 100 timed reps; "×" = LOCAL time / fastest baseline variant; all of them use the
same weights):

| batch, prompt | FULL flex | FULL orig. SDPA | FULL flash causal | LOCAL-W128 | × |
|---|---|---|---|---|---|
| 1 × 128 | 17.36 ms | 14.78 | 14.43 | 17.39 | 1.205 |
| 1 × 256 | 17.43 | 14.99 | 14.68 | 17.45 | 1.189 |
| 1 × 512 | 17.39 | 14.99 | 14.69 | 17.41 | 1.185 |
| 1 × 1024 | 17.52 | 15.12 | 14.85 | 17.59 | 1.184 |
| 8 × 128 | 17.52 | 15.12 | 14.79 | 17.59 | 1.189 |
| 8 × 256 | 18.00 | 15.59 | 15.28 | 18.04 | 1.181 |
| 8 × 512 | 18.43 | 15.99 | 15.70 | 18.47 | 1.176 |
| 8 × 1024 (primary) | 19.96 | 19.05 | **17.44** | **19.90** | **1.141** |

- Prefill at these sizes is latency- and launch-bound.
- Local attention needs the flex kernel, which carries per-call block-mask construction overhead.
- The flash `is_causal` baseline is therefore faster at every point. Against the flex baseline, LOCAL-W128 is equal
  (within ±0.4%).

**Cached greedy decoding of 128 tokens** (tokens/s and per-token latency; 100 timed generations):

| batch, prompt | FULL-TOKEN-4 | LOCAL-W128 | ratio |
|---|---|---|---|
| 1 × 128 | 97.5 tok/s / 10.25 ms | 97.8 / 10.23 | 0.997 |
| 1 × 256 | 97.2 / 10.28 | 98.0 / 10.21 | 0.992 |
| 1 × 512 | 97.4 / 10.27 | 97.6 / 10.24 | 0.998 |
| 1 × 1024 | 97.2 / 10.29 | 97.5 / 10.25 | 0.997 |
| 8 × 128 | 766.5 / 10.44 | 767.4 / 10.43 | 0.999 |
| 8 × 256 | 762.8 / 10.49 | 766.3 / 10.44 | 0.995 |
| 8 × 512 | 760.7 / 10.52 | 766.0 / 10.44 | 0.993 |
| 8 × 1024 (primary) | 772.5 / 10.36 | 774.2 / 10.33 | **0.998** |

- Decoding costs about 10 ms per step at every context length. That time goes to eager-mode launches and the MoE
  dispatch, which synchronises with the host in every MoE layer; it does not go to attention reads.
- Reading 128 instead of up to 1,151 cached keys is therefore invisible.
- Decoding from a 1,024-token prompt reaches position 1,151, beyond the training context. This was a speed
  measurement only.

**Primary speed measures** (candidate time / fastest baseline; ≤ 0.90 qualifies):

| measure | ratio | qualifies |
|---|---|---|
| matched-batch training | 0.992 | no |
| end-to-end 100M training | 0.996 | no |
| prefill 8 × 1024 | 1.141 | no |
| decoding 8 × 1024 | 0.998 | no |

## 8. Actual vs theoretical compute savings

Theoretical forward FLOPs per token, using measured validation context lengths: multiply-add = 2; attention
includes the QKV and output projections.

| | FULL-TOKEN-4 | LOCAL-W128 | ratio |
|---|---|---|---|
| keys attended per token per layer | 397.9 | 115.3 | 0.290 |
| attention FLOPs (8 layers) | 4.80 M | 3.07 M | 0.639 |
| total forward FLOPs | 31.28 M | 29.55 M | **0.945** |

- Attention is 15.4% of the baseline's theoretical FLOPs. The tied 50k-vocabulary LM head and the MoE dominate.
- **Theoretical saving 5.5%; measured saving 0.4–0.8% in training, about 0% in decoding, and negative in prefill
  against the flash baseline.**
- Two reasons the measured saving is smaller:
  - the block-sparse document-causal kernel already skips cross-document work for FULL, so attention is only about
    4% of step time;
  - eager decoding is launch-bound.
- Theoretical FLOPs were never counted as speed.

**Profiling of the conditional path** (`gpu_checks_20261108T075120Z.json`, G6).
- Compact global attention computes only the selected queries' tiles: 0 / 178 / 277 / 380 tiles at selection
  rates 0 / 0.10 / 0.25 / 0.5, against 402 for full attention.
- The profile shows no dense, flash or SDPA kernel and no N×N buffer on the path. Its peak memory increment is 60 MB,
  while one head's dense score matrix would be 537 MB.
- Wall time is overhead-bound, however: about 1.3 ms per call regardless of rate (forward only), against 0.33 ms for
  full flex attention. Per-layer block-mask construction, sorting and gathers dominate.
- A counterfactual training step, which computes dense full attention for all tokens, was cheaper than an ordinary
  compact step.

## 9. VRAM and KV cache

**Training peak allocated memory** is identical (18.8 GiB). Peak reserved memory differed in the first pass
(24.0 vs 20.8 GiB), but the drift-check repeat showed 20.9 GiB for both. This is allocator caching, not a model
difference.

**KV cache** is identical by design: every position is kept and nothing is evicted or compressed. The fixed-local
model could in principle use a W-sized ring buffer without changing outputs, but this experiment did not authorise
it, so **no KV-memory saving is claimed**.

| batch, prompt + 128 decoded | KV cache |
|---|---|
| 1 × (128 + 128) | 1.5 MiB |
| 8 × (1,024 + 128) | 54.0 MiB |

Peak decoding memory ≤ 2.22 GiB for both models.

## 10. Parameters

| | parameters |
|---|---|
| every condition, total | 28,923,592 |
| — backbone (inherited TOKEN-4, unchanged) | 28,873,920 |
| — per-layer gate MLPs | 49,672 (0.172% of the backbone; allowance < 0.3%) |
| active per token, FULL-TOKEN-4 and LOCAL-W128 | 14,423,232 |
| active per token, conditional candidates | 14,472,904 (backbone plus gate) |

- The gate is present but never computed in non-conditional conditions, including the baseline. In those runs it
  receives no gradient and is never updated.
- The extra active parameters of conditional candidates are the gate itself.

## 11. All candidates tried, including failures

**Generation 1** (seed 98, 5M tokens, CALIB-SEARCH; FULL = 6.5476):

| candidate | ΔCE | throughput ×FULL | measured global rate | attention work ×FULL | outcome |
|---|---|---|---|---|---|
| LOCAL-W128 | −0.0275 | 1.014 | – | 0.288 | advanced (1) |
| HYBRID-W256-LATE4 | −0.0194 | 1.002 | – | 0.757 | advanced (2) |
| HYBRID-W256-LATE2 | −0.0184 | 0.986 | – | 0.636 | advanced (3) |
| G256-10 | −0.0186 | 0.781 | 0.054 | 0.591 | advanced (4) |
| LOCAL-W256 | −0.0159 | 1.012 | – | 0.514 | rank 5, not advanced |
| G128-25 | −0.0169 | 0.774 | 0.126 | 0.439 | not advanced |
| G256-50 | −0.0115 | 0.793 | 0.271 | 0.899 | not advanced |
| G512-25 | −0.0116 | 0.774 | 0.130 | 1.052 | not advanced |
| LOCAL-W512 | −0.0085 | 0.992 | – | 0.814 | not advanced |
| G256-25 | −0.0011 | 0.775 | 0.124 | 0.690 | **eliminated**: ≥ 10% slower, no ≥ 0.010 compensating gain |

Notes on Generation 1:
- No candidate was non-finite or collapsed, or was more than +0.05 worse.
- Conditional candidates that were ≥ 10% slower survived only through the compensating-benefit clause
  (ΔCE ≤ −0.010).
- The tie rule (within 0.002, prefer lower step GPU time) decided the order among HYBRID-LATE4, HYBRID-LATE2 and
  G256-10.

**Generation 2** (seeds 98 / 99, 20M tokens; FULL = 5.3785 / 5.3779):

| candidate | Δ s98 | Δ s99 | mean | throughput ×FULL | outcome |
|---|---|---|---|---|---|
| LOCAL-W128 | −0.0155 | −0.0150 | −0.0153 | 1.007 | advanced |
| HYBRID-W256-LATE2 | −0.0179 | −0.0101 | −0.0140 | 0.982 | advanced |
| G256-10 | −0.0112 | −0.0093 | −0.0103 | 0.768 | not advanced |
| HYBRID-W256-LATE4 | −0.0123 | −0.0081 | −0.0102 | 0.988 | not advanced |

**CALIB-SELECTION** (evaluated once per frozen Gen-2 checkpoint).

| candidate | mean paired Δ | Gen-2 speed |
|---|---|---|
| LOCAL-W128 | −0.0157 | 130,509 tok/s |
| HYBRID-W256-LATE2 | −0.0143 | 127,314 tok/s |

The two are within 0.003 of each other. LOCAL-W128 was 2.5% faster, outside the 1% speed-tie band, so the rule
selected **LOCAL-W128**.

## 12. Did any benefit come from local attention, hybrid layers or learned gating?

- **Local attention.** It gave a real but transient benefit: a locality inductive bias that speeds early learning
  (−0.015 to −0.028 nats at 5–20M tokens). That benefit reversed by 45–75M tokens and became a +0.0245 deficit at
  100M, concentrated at positions beyond the window. It gave no meaningful speed.
- **Hybrid layers.** The hybrids showed the same early advantage as pure local attention at 5M and 20M tokens
  (−0.008 to −0.019), with no speed gain (×0.98–1.00). Per-position effects were not measured for them. Neither
  hybrid was trained to 100M, because the frozen selection rules picked LOCAL-W128. Whether keeping full attention in
  the final layers would stay comparable at 100M is therefore **untested**.
- **Learned gating.** The gate learned an interpretable token policy (section 5). But it was never better than fixed
  local attention at matched budgets, was 21–23% slower in training, and its realised rates fell below target. No
  benefit.

## 13. Code fixes and deviations

**Engineering before the preregistration** (dry runs; no candidate results):
- head dim 24 zero-padded to 32, because FlexAttention in torch 2.5.1 requires a power of two (exact);
- the compact-path gathers used duplicate-index advanced indexing, whose backward is a serialised accumulate. They
  were replaced with permutation `index_select` gathers in two places, taking the compact path from 7.5 to 1.9 ms
  and the conditional training step from 513 to 270 ms;
- BlockMask construction now builds ordered block lists directly, with equivalence to torch's builder tested;
- prefill right-pads rows to a multiple of 128 for flex;
- GPU-check fixes:
  - the bf16 baseline comparison is bounded by ΔCE, mean |Δlogit| and expert agreement, with the original model's
    own bf16 noise floor recorded alongside;
  - the training-step check uses the real schedule's learning rate;
  - a profiler attribute name was corrected.

  All of these were settled in dry runs before the preregistration was hashed. The dry-run artifacts
  (`gpu_checks_20261108T0658…`, `…0700…`, `…0701…`, `bench_quick_*`) are preserved, and their GPU time is counted.

**After the preregistration, before any candidate training** (`deviations.jsonl`): the first complete CPU suite
failed one resume test at the 1-ulp level.
- Root cause: the dense *reference* compact path gathered keys by duplicate-index advanced indexing. Its CPU
  backward is order-nondeterministic, so even two uninterrupted runs differed.
- Fix: `index_select` gathers (numerically equivalent).
- After the fix, 13 of 13 steps are bit-identical between a full run and an interrupted-and-resumed run. The suite
  was rerun: 604 passed.

**During the experiment:** no run failed, was contaminated, needed a resume, or was rerun. There were no deviations
from the frozen rules.

**Post-hoc (labelled; informs no decision):** the gate diagnostic on CALIB-SEARCH (`clav1_posthoc_gate.py`).

## 14. Fairness confirmations

Within every seed (asserted by the driver and `final_analysis.json` → `fairness`; all true), FULL-TOKEN-4 and
LOCAL-W128 had:
- identical full initial parameter hashes;
- identical training rows in identical order (row-list hash);
- 3,861 optimizer steps and 100,007,082 valid training tokens each;
- the same micro-batch (16 rows) × accumulation (2), learning-rate schedule, precision, checkpoint and evaluation
  frequency;
- no early stopping and no extra tokens.

Further confirmations:
- The same matching was asserted for every search generation.
- The FULL-TOKEN-4 baseline was re-trained from scratch, not reused. Its flex attention matches the original
  CSRModel: fp32 max |Δlogit| 2.7e-5, ΔCE 5e-7, 100% identical expert choices. In bf16 the difference is within the
  original model's own bf16-vs-fp32 noise floor.
- All six final validation evaluations used the same 10,732,175 tokens.
- No candidate was initialised from a trained model. MoE routing (unrestricted top-4), the shared expert and
  capacity were unchanged.

## 15. Locked test split

The test split was never accessed:
- `load_raw_split` refuses `test` before any file access (tested);
- inventories exclude test files by name;
- audited CPU training subprocesses opened only `train_*`, `calib_*` and manifest data files.

**Data isolation for the other splits:**
- CALIB-SELECTION was opened only after `search_frozen.json`.
- Validation was opened only after `final_frozen.json` and `final_trained.json`, and each checkpoint was evaluated
  once (eval registry).
- The protected inventory (2,603 files, 39.7 GB) was verified unchanged after the run: 0 mismatches.
- The preregistration hash (sha256 `fa740ccd…`) was unchanged at the end.

## 16. Limitation

This experiment changes **only the attention computation** of the TOKEN-4 MoE. It does **not** test semantic or
domain expert specialisation, routing or expert-pool structure.

Its conclusions apply to this scale: d_model 192, 8 layers, 1,024-token context, about 640-token documents, an
A100, and eager PyTorch with FlexAttention. Two specific limits:
- At much longer contexts, attention's share of compute and the possible gains from local attention would be
  larger.
- The search budgets (5M / 20M tokens) were too short to reveal the late reversal of local attention's advantage.

## Resources and artifacts

**GPU time.** 3.23 GPU-hours against the 20 h cap (all `clav1` ledger rows, including 142 s of engineering probes
and the dry runs). Disk ≥ 152 GiB free throughout.

**Artifacts.**
- Preregistration: `docs/CONDITIONAL_LOCAL_ATTENTION_V1_PREREG.md` (sha256 `fa740ccd…`, ledger `CLA_V1_PREREG`).
- Manifest: `artifacts/conditional-local-attention-v1/manifest.json`.
- Code: `src/csr/cla_{attn,model,run,select,decode}.py`, `cla_*.py`.
- Tests: `tests/test_conditional_local_attention_v1.py`; CPU suite `cpu_tests_20261108T075031Z.json`, 604 passed.
- Formal GPU checks: `gpu_checks_20261108T075120Z.json` (PASS).
- Driver state: `.tmp/conditional_local_attention_v1/` (driver.log, state.json, runs.tsv, evals.tsv).

**Verdict: NEGATIVE**
