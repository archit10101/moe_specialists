# csr-speculative-pool-routing-v1 — preregistration

Written 2026-11-06, before any draft or target training of this experiment. The SHA-256 of this file is recorded in
`ledger/safety_ledger.tsv` (event `SP_V1_PREREG`) and in `artifacts/speculative-pool-routing-v1/manifest.json`;
every config of this experiment carries it (`prereg_sha256`) and every entry point refuses a config whose hash differs.
The previous csr-sequence-routing-search-v1 validation result (strict one-route-per-segment routing was +0.116 /
+0.144 nats/token worse than TOKEN-4) is motivation only; no configuration below is chosen with any validation CE.

## 1. Questions

* **Primary.** Can chunk-level candidate-expert selection, combined with token-level top-k routing inside the
  candidate pool, retain or improve TOKEN-4 language-model accuracy while reducing expert-routing churn?
* **Secondary.** Can a causally generated speculative draft of the current 4–8-token block select a better pool than a
  pool selected without draft lookahead, while providing useful speculative-decoding acceptance and wall-clock speed?

This is not one-route-per-chunk routing: the chunk mechanism chooses a candidate pool; every token keeps an independent
router decision (its own top-4 inside the pool) and independent mixture weights.

## 2. Protected foundation (verified against the resolved config `configs/rep-v1/seed0/condition_A.json`)

| item | protocol | resolved (repository is authoritative) |
|---|---|---|
| total / active parameters | ≈28.9M / ≈14.4M | 28,873,920 / 14,423,232 (count_parameters) |
| hidden dim, context, layers | 192, 1,024, 8 causal | d_model 192, max_seq_len 1024, n_layer 8 |
| dense / MoE layers | layer 0 dense, 1–7 MoE | dense_ffn 768 (SwiGLU) at layer 0; moe_layers [1..7] |
| experts | 32 routed, top-4 + shared | n_experts 32, top_k 4, expert_ffn 128, shared_ffn 256 |
| embeddings | tied | tie_embeddings true, vocab 50,304 (Pythia tokenizer) |
| attention | existing | 8 heads × 24, RoPE base 10,000, RMSNorm eps 1e-6, block-diagonal causal document masks |
| optimizer / schedule | existing | AdamW lr 6e-4, betas (0.9, 0.95), wd 0.1 (dim ≥ 2), eps 1e-8; WSD: 60 warm-up steps, constant, linear decay over the last 20% to 6e-5; clip 1.0 |
| batch / precision / objective | existing | 2 × 16 packed rows of 1,024 tokens per step; bf16 autocast, fp32 router; next-token CE + 0.01 Switch load balance + 0.001 router z-loss |

No discrepancy was found. Data: `artifacts/data/fwe-rep-v1` (FineWeb-Edu sample/10BT, Pythia tokenizer, frozen
document-hash splits; manifest `splits` sha256 c28e9845…). Train 198,012 documents / 126.2M tokens; calib 8,179 / 5.28M;
val 16,850 / 10.75M. The locked test split is never loaded, decoded, inspected, content-hashed, evaluated or used:
`spv1_run.load_raw_split` refuses it before any file access, inventories exclude it by name.

## 3. Isolation

* CALIB-SEARCH / CALIB-SELECTION: document-hash division of the existing calib split,
  `u = int.from_bytes(sha256("csr-spv1-calib-partition|" + doc_id)[:8], "little") / 2**64`, CALIB-SEARCH iff
  u < 0.70. Written once (`search_partition.json`, sha256 ledgered) before any training.
* All search decisions use CALIB-SEARCH only. CALIB-SELECTION is opened only after `search_frozen.json` exists, once
  per frozen checkpoint (registry `eval_registry/`). Validation is opened only after `final_frozen.json` lists the
  final config hashes and training is complete, once per final checkpoint. No configuration is edited afterwards.
* The draft is trained on the train split only and frozen on CALIB-SEARCH only, before any target candidate is trained.

## 4. Conditions (code: `src/csr/spv1_model.py`)

Every condition: same backbone, same initialisation within a seed (backbone constructed and initialised exactly as
TOKEN-4 under the global seed; extra parameters zero-initialised or drawn from a dedicated generator afterwards),
same training documents/order/tokens, same optimizer and schedule, same batch, exactly 4 routed experts per token +
the unchanged shared expert, same evaluation documents and masks.

Token routing inside a pool: `z_t = W_l h_t` (fp32), `p_t = softmax(z_t / T_r)`; `topi_t` = top-4 of `p_t` restricted
to the cached pool; gates = `p_t[topi_t]` (global probabilities, no renormalisation — TOKEN-4 gate semantics).

* **T0 TOKEN-4** — unrestricted global top-4 (bit-identical to `CSRModel` token mode; test-verified).
* **C1 FIXED-POOL** — blocks start at in-document positions 0, L, 2L… (L ∈ {4, 8}).
* **C2 ADAPTIVE-POOL** — a block starts at token t iff t is a document start, t − s ≥ H (s = current block start,
  max hold H ∈ {4, 8}), or `score_l(t) = (1 − Σ_{e∈pool(s)} p_{t,e}) + w_l·sg(h_t) + b_l > τ_l` (router mass outside the
  cached pool plus a learned linear term; w, b zero-initialised). A refreshed pool applies to the current token.
  τ_l is a buffer: initialised by bisection on the first training micro-batch (no-grad pre-pass, target rate),
  then `τ_l += 0.5·(observed refresh rate − target)` (step clipped to ±0.05) between optimizer steps; frozen in
  evaluation/inference; saved in checkpoints. Target refresh-rate regions 0.50, 0.25 and the "12.5% region" realised
  as target 0.15 with H = 8 (0.125 is the forced-refresh floor of H = 8, i.e. FIXED-8; it cannot be reached adaptively).
* **C3 SPECULATIVE-POOL** — speculative block size B ∈ {4, 8} = target positions per verification block = 1 accepted
  token + γ = B − 1 draft tokens. For every potential block start s the frozen draft proposes `D[s,1..γ]` greedily from
  the prefix `x_0..x_s` only. A block starting at s continues while the real input token `x_{s+j}` equals `D[s,j]`
  and j < B; the first mismatch (or j = B, or a document start) starts a new block. This is exactly the block
  structure of greedy speculative decoding: the target's correction / bonus token starts the next block, whose
  proposal and pool are regenerated from the accepted prefix. Proposals are never replaced by the ground truth.
* **C4 RANDOM-POOL controls** — identical to the frozen contextual finalist (same family, pool size, block rule,
  refresh controller / draft block structure, active experts, load losses) except the pool: a fixed,
  content-independent table indexed by (layer, in-document position of the block start), built from concatenated
  random permutations so that every E/M consecutive table entries partition the 32 experts (load-preserving).
* **Speed control TOKEN-4 + speculation** — TOKEN-4 target decoded speculatively with the same frozen draft and the
  same γ (decoding-speed control only; not a separately trained model).

Pool of a block starting at s (learned assignment), computed only from causal state at s:
`q_s = sg(W_l) h_s + U_l h_s` (+ for C3 `β_l sg(W_l) g_s + V_l g_s`, `g_s = Σ_j a_{l,j} rms(sg(Emb[D[s,j]]))`,
`a_l = softmax(α_l)`), pool = TopM(q_s), M ∈ {8, 16}. h_s is the block-start token's pre-MoE (post-RMSNorm) state —
allowed causal state; `sg` = stop-gradient. U, V zero-initialised, β = 1, α = 0.
Gradient paths (forward values unchanged, so training = inference): selector — gates multiplied by
`exp(log m_s[e] − sg(log m_s[e]))`, `m_s = softmax(q_s / T_p)`; detector — gates multiplied by
`exp(log ρ − sg(log ρ))` at non-forced refreshes and `exp(log(1 − ρ) − sg(·))` at reuses, `ρ = σ((score − τ)/κ)`,
κ = 0.05. Pool-level load balance `E Σ_e f_e P_e` (f = token-weighted pool-membership share / M, P = mean m) with
coefficient 0.01 (searched), optional selector entropy bonus (coefficient 0 unless searched).
Extra parameters: C1 43,008 (0.149%), C2 44,359 (0.153%), C3 ≤ 86,072 (0.297%) — all < 0.5%, so no parameter-matched
TOKEN-4 control is required (any configuration exceeding 0.5% is invalid).

Implementation choices recorded here: the route-state EMA / GRU summary dimension is **not used** (the selector summary
is the block-start state; C3 adds draft-token features) so that every pool rule is exactly reproducible in vectorised
teacher forcing and incremental decoding; frozen-teacher distillation of pool membership is **not used** because
expert indices of independently trained models are not aligned; pool dropout = training-only deterministic masking of
selector candidates (hash of seed/step/micro-batch/layer) before Top-M; the pool-restriction warm-up is a hard switch
(unrestricted for the first fraction of steps). All block structures are computed with GPU-resident pointer doubling
(`orbit_starts`); no Python per-token loop, no CPU pass and no host synchronisation in the structure kernels.

## 5. Draft model

No suitable frozen project-local draft exists (all project checkpoints are MoE target-scale models or ≤5M-token smoke
/ pilot MoE models on an earlier data build). One dense causal draft is trained: `CSRModel` with no MoE layers,
d 128, 4 layers, 4 heads, SwiGLU 512, tied embeddings, same tokenizer (7,488,640 parameters; target 28,873,920),
100M training tokens of the train split (order seed 17), project optimizer/schedule except lr 1e-3 → 1e-4 (smaller
width). It is frozen on CALIB-SEARCH only (finite CE; top-1 accuracy reported), after GPU checks D1–D3 (full ==
incremental logits, tree proposals == repeated greedy generation, smaller and faster than the target). Draft training,
proposal caching and draft inference are counted in all resource and speed totals. The optional full-size TOKEN-4
upper-bound draft diagnostic is not used.

Proposal cache: greedy proposals (γ = 7; the first B − 1 serve block size B) for every position of every training
document, computed once with `draft_tree_proposals` (all prefixes in parallel; bf16 autocast) and stored with the
draft checkpoint hash, code hashes and an agreement check against recomputation under a different packing.
Evaluations (CALIB-SEARCH monitor/final, CALIB-SELECTION, validation) recompute proposals on the fly.

## 6. Decoding semantics

Primary: deterministic greedy. Each block: (1) draft proposes γ tokens from the accepted prefix; (2) the speculative
pool is selected (C3) from the target state at the block start and the proposal; (3) the target evaluates the block
causally; (4) consecutive draft tokens matching the target's greedy choice are accepted; (5) at the first mismatch the
target's choice is emitted, the remaining suffix is discarded and decoding restarts from the updated prefix; (6) the
next proposal and pool are regenerated — nothing computed for a rejected suffix (KV entries, pools) is reused.
The pool of a block is a deterministic function of the accepted prefix at its start (the proposal is generated from
that prefix), exactly as in teacher-forced CE; the accepted block tokens' states therefore are the same in decoding
and teacher forcing. Convention for prompts: the last prompt token starts a block (both in speculative decoding and
in the sequential reference). Batched speculative greedy decoding must reproduce sequential greedy execution of the
same target algorithm exactly (checked in fp64; fp32 reported). Sampling equivalence is not claimed.

## 7. Correctness gates (before training)

CPU suite (CUDA hidden; all pre-existing tests + `tests/test_speculative_pool_v1.py`) must pass, then GPU checks
(`spv1_gpu_checks.py --phase target`, every Gen-1 config + TOKEN-4 at full size): future-token perturbation leaves
logits / pools / starts / expert ids at earlier positions unchanged; teacher forcing == incremental execution;
speculative == sequential greedy; structure kernels run under CUDA sync-debug "error"; one bf16 training step finite
with peak VRAM < 56 GiB. Then `--phase draft` after draft training. Tests cover every item of the protocol list
(causality, fixed boundaries, adaptive decisions from causal states only, speculative pools from prefix + proposal,
rejected-suffix non-reuse, exactly 4 distinct experts in the pool + shared path, independent weights, TOKEN-4
unchanged, identical initialisation, incremental == teacher forcing, resume reproducibility, gradients to selector /
detector / router, finiteness, isolation, test-split refusal, no host loops in structure kernels).

## 8. Bounded search (≤ 24 unique target candidates, excluding T0, random controls, speed-only controls)

Defaults: T_p = 1, selector-lr ×1, T_r = 1, router-lr ×1, pool dropout 0, pool load-balance 0.01, entropy 0, warm-up 0.

* **Generation 1** (seed 98, 5M tokens, CALIB-SEARCH; 12 candidates):
  FIXED L∈{4,8} × M∈{8,16} (G1-F4M8, G1-F4M16, G1-F8M8, G1-F8M16);
  ADAPTIVE (H, M, target) ∈ {(4,16,0.50), (8,16,0.25), (8,8,0.25), (8,16,0.15)};
  SPEC B∈{4,8} × M∈{8,16}. Recorded: CE, pool recall / containment, refresh rate, block lengths, router and selector
  confidence, load, unused experts, distinct experts per window, adjacent Jaccard, throughput, memory.
* **Generation 2** (seeds 98 and 99 from fresh identical initialisation, 20M tokens; ≤ 6): the strongest valid
  Gen-1 candidate of each family promoted unchanged (G2-F1, G2-A1, G2-S1) and one preregistered variant of each
  (G2-F2, G2-A2, G2-S2: selector-lr multiplier 3 and pool-restriction warm-up over the first 10% of steps).
  Rejected: collapse / unused expert, non-finite, causality failure, or worse than T0 by > 0.05 nats/token in both
  seeds.
* **Generation 3** (seeds 97, 98, 99, fresh init, 50M tokens; ≤ 3): best non-rejected causal (C1/C2) Gen-2
  candidate, best non-rejected C3 candidate, and the best remaining non-rejected candidate. Then freeze the best causal
  and the best speculative Gen-3 candidate and define their random-pool controls.
* "Best" = lowest mean CALIB-SEARCH CE; near ties (≤ 0.002 nats) broken by lower refresh rate, fewer distinct experts
  per 8-token window, higher training throughput. Valid = complete in every seed, finite, no collapse (no unused
  routed expert and no expert > 50% of routed slots in any layer), extra parameters ≤ 0.5%.
* A family without a valid candidate is recorded as such; it is never converted into token routing.

## 9. Blind selection

After the search is frozen, T0 and the frozen finalists (Gen-3 checkpoints, 3 seeds) are evaluated once on
CALIB-SELECTION; ranking by CE with the near-tie rule above. A finalist proceeds to final training unless invalid
(failed / collapse). Nothing is edited afterwards; all results are kept.

## 10. Feasibility and compute gate

Benchmarks before final training (Gen-3 seed-97 checkpoints + draft): training step, prefill, greedy and speculative
decoding (measured GPU wall time). Peak VRAM must stay < 56 GiB; disk ≥ 80 GiB. Hard limit: 20 GPU-hours for this
experiment in total (ledger purpose `spv1 …`). Projected total (used + 3 seeds × Σ conditions B × measured seconds per
token from the Gen-3 runs + 15 validation evaluations + benchmarks + slack) must not exceed 16 GPU-hours (20% margin).
Largest equal budget B ∈ {100M, 50M, 25M} for T0 and both finalists; random controls at B if they fit, otherwise at
the selection budget (50M), labelled secondary. Never more tokens for a preferred condition than for T0.

## 11. Final training, validation and benchmarks

Seeds 0, 1, 2 from scratch: T0, best causal finalist, best speculative finalist, their random controls. One
validation evaluation per final checkpoint after training. Final benchmarks (seed-0 checkpoints): training step,
prefill (batch 1 / 16), greedy decoding batch 1 / 4 / 16 (fp32, 64-token prompts from CALIB-SEARCH documents, 128 new
tokens), speculative decoding for TOKEN-4 + speculation and the C3 finalist and control; acceptance length, rejection
positions, regenerations, target / draft forwards per emitted token, VRAM, GPU seconds; exact greedy equivalence.

## 12. Analysis (validation CE, nats/token)

Per candidate vs T0: per-seed CE; Δ_s = CE_cand,s − CE_T0,s; mean and SD of Δ; paired per-document bootstrap
(documents resampled 2,000 times, seed 20261107) of `Σ_d (mean_s NLL_cand,d,s − mean_s NLL_T0,d,s) / Σ_d n_d`;
fraction of documents improved; learning-curve differences at matched token counts (monitor CE on the first 128
CALIB-SEARCH rows every 5M tokens; training CE).
* **Substantially better**: mean Δ ≤ −0.02; Δ_s < 0 for all seeds; bootstrap 95% upper bound < 0; no collapse /
  unused expert; identical training-token and active-expert budgets.
* **Accuracy-comparable**: mean Δ ≤ +0.01; every Δ_s ≤ +0.01; bootstrap upper bound < +0.01.
* Content dependence: finalist vs its random control (same bootstrap). Speculative vs causal: C3 finalist vs C1/C2
  finalist.
* **Speed claim**: ≥ 10% end-to-end wall-clock improvement over the matched control (greedy: T0 at the same batch;
  speculative: TOKEN-4 + speculation with the same draft, γ and batch), all draft / detector / selection /
  verification / fallback time included, same hardware / batch / context / generation length / output mode, and
  accuracy-comparable validation CE. Never from router-call counts or FLOPs.
These criteria are fixed and are not redefined after results are observed.

## 13. Stop conditions and deviations

Stop only for: unresolved causality / future-information violation; validation/test contamination; corrupted or
mismatched protected artifacts; occupied GPU that cannot be resolved (> 2 h); < 80 GiB disk; non-finite training
persisting after one preregistered numerical retry (fresh run, same config; a search candidate is then recorded
INVALID instead of stopping); a required change outside the experiment scope; reaching the hard compute limit.
Interrupted runs resume from their last complete `train_state.pt` (bit-exact continuation). Narrow correctness fixes
are allowed; each is logged in `artifacts/speculative-pool-routing-v1/deviations.jsonl` and never relaxes a rule.
