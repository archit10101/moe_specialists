# csr-value-routed-compute-v1 — protocol (screening pilot, written before any candidate result)

Written 2026-11-09 ~02:55 EST. A screening-quality pilot under a hard 09:00 deadline. It is not a confirmation.
Earlier results (clav1 / spv1 / srsv1) are not reinterpreted. Nothing here changes after results are visible; any
change is logged in `artifacts/value-routed-compute-v1/deviations.jsonl`.

## Question
Can a causal controller, trained on measured counterfactual next-token-loss reductions, match FULL-TOKEN4 accuracy
while reducing measured long-context inference time, by spending long-range attention and extra routed experts only
where useful? This is an inference-efficiency question. No training-speed claim is made.

## Data (existing project data, longer documents)
- The document sets are the existing `artifacts/data/fwe-rep-v1` train and calib documents (same ids, same
  rep-v1 split rule and de-duplication). They are re-tokenised from the existing raw parquet
  (`artifacts/data/raw/fwe_10BT_000_rep_v1_rg240_seed20261101.parquet`, plus the reference raw for ids not found in
  it) with the project Pythia tokenizer and `max_len = C` (tokens[:C-1] + EOS). Output: `artifacts/value-routed-compute-v1/data/`.
- Train: the train documents in `train_order(seed=17)` order, taken until 24M valid tokens. Calib: all calib documents.
  Validation is not used. The locked test split is never loaded (its ids are skipped before any text is read).
- Calib partition: `u = sha256("csr-vrc-v1-calib-partition|" + doc_id)`; CALIB-SEARCH if u < 0.70, else
  CALIB-SELECTION. Index lists and hashes are recorded.

## Context length
C = 8192 if (a) flex block-sparse attention agrees with the dense reference (max abs diff < 2e-2 in bf16 outputs,
< 1e-4 in fp32 on CPU tests), (b) causality / document isolation tests pass, (c) resume-equivalence passes, and (d) a
training step with 2 rows × 8192 per micro-batch (accum 2 → 32,768 tokens per optimizer step, the project's step size)
and a prefill benchmark batch of 4 × 8192 both fit under the project's 56 GiB allocator cap. Otherwise C = 4096
(4 rows per micro-batch, accum 2; benchmark batch 8 × 4096), otherwise 2048. C is fixed before any candidate result.

## Architecture
- FULL-TOKEN4: the existing NanoMoE (d 192, 8 layers, layer 0 dense, layers 1–7 MoE, 32 experts, top-4, shared
  expert, tied embeddings) with document-causal full attention in every layer (FlexAttention, block-sparse).
- VALUE-ROUTED path family (same weights): document-causal local attention with window 512 in every layer; layers 3
  and 7 can additionally run global (full document-causal) attention for a 128-token query block. The shared expert
  always runs. Routed experts k ∈ {1, 2, 4} per 128-token block, via fixed four-slot dispatch: top-4 is always
  computed over all 32 experts; slots ≥ k are NULL (gate 0, never dispatched, no expert FFN executed). Gates are the
  unrenormalised global-softmax probabilities (the TOKEN-4 rule). Full fallback = global in layers 3, 7 + k = 4.
- Block-sparse masks: per query block, the KV-block list is the local list or the full list (exact union semantics),
  so no dense attention is computed for local blocks.

## Mandatory diagnostic gate (A.5)
No compatible long-context checkpoint exists (all existing checkpoints are 1024-context, documents truncated at 1024),
so one FULL-TOKEN4 baseline is trained for 5M valid tokens at context C (seed 0, lr 6e-4, WSD with 60 warm-up steps
and 20% decay to 6e-5, AdamW β (0.9, 0.95), wd 0.1, clip 1.0, aux 0.01, z 0.001, bf16 autocast, fp32 router).

Counterfactual measurement on CALIB-SEARCH with the shared weights, uniform policies (every block the same option):
REF = FULL-TOKEN4 (full attention in all layers, k = 4); and the six candidate options
o = (attn ∈ {L: layers 3, 7 local; G: layers 3, 7 global}) × (k ∈ {1, 2, 4}), all other layers local W512.
Per 128-token query block b (packed-row aligned), NLL sums over valid targets are recorded for each option.
Labels: V_context = CE(L,4) − CE(G,4); V_2 = CE(G,1) − CE(G,2); V_4 = CE(G,2) − CE(G,4);
V_joint = CE(L,1) − CE(G,4).

Costs per block: routed-expert evaluations = k × non-pad tokens × 7 MoE layers; global query blocks = 2 (layers 3,
7) if attn = G and the block has a non-pad token. Time: CUDA-synchronised prefill forward time (bf16, no grad, 5
warm-up + 20 timed repetitions, median) of the whole model, including the LM head, for REF and each uniform option on
identical packed CALIB-SEARCH batches. The projected time of a mixed policy is Σ_o frac_o · T_o (frac_o = fraction of
non-empty blocks choosing o).

Oracle frontier: for λ on a fixed log grid (0 and 1e-6 … 1e2, 60 points), every block chooses
argmin_o ΔNLL_b(o) + λ · ĉ_b(o), where ĉ is the block's share of the measured option time. This is an
additive approximation (blocks interact through attention), so every gate-passing point is re-evaluated exactly with a
mixed-policy forward pass on CALIB-SEARCH.

Gate (all four simultaneously, at one operating point; thresholds fixed now):
1. CE degradation versus REF ≤ +0.010 nats/token (projected additive AND exact re-evaluation);
2. routed-expert evaluations ≤ 75% of top-4;
3. global query blocks in the global-capable layers (3, 7) ≤ 50% of the full fallback's;
4. projected end-to-end prefill time ≤ 90% of REF's measured time.
Primary comparison is against REF (FULL-TOKEN4). The same frontier versus the candidate's own fallback (G,4) is
reported as secondary. If the gate fails, Experiment A stops and the report states the measured reason.

## Pilot (only if the gate passes)
FULL-TOKEN4 and VALUE-ROUTED, one seed, 20M valid tokens each (10M each if time is short), identical documents,
order, initialisation, optimiser and steps. VALUE-ROUTED training: every training block samples its option
uniformly from the six options (deterministic hash of seed, step, block), with a self-distillation term
(KL to the same step's no-grad full-fallback output on 10% of steps; its compute is counted). Controller: a small
MLP over strictly causal features (previous-block mean hidden state after layer 2, previous-block router entropy
and top-k margins in layers 1–2, previous-block local-attention entropy, block start position in its document,
document-boundary flags), trained on CALIB-SEARCH counterfactual labels with an explicit compute penalty
μ = 1e-3 nats per routed-expert evaluation per token (fixed now, never tuned on validation). Thresholds frozen on
CALIB-SEARCH; one evaluation on CALIB-SELECTION; prefill/decode benchmarks at 1K, 2K, 4K, 8K (if C allows) with
p50/p95 latency, tokens/s, peak VRAM, KV bytes, expert evaluations per token, global-query fraction and fallback
rate. Success: ΔCE ≤ +0.010 vs matched FULL-TOKEN4, ≥ 10% measured end-to-end prefill or decode speed-up at C, no
hidden dense attention, no expert collapse.

## Integrity
One GPU job at a time via `scripts/gpu_guarded.sh`; occupancy checked before every GPU command; all GPU time ledgered
(including failures, warm-ups and benchmarks). CPU tests with CUDA hidden. Validation and locked test splits are not
read. Absolute deadlines: no GPU launch after 08:35 EST, all compute stopped by 08:45, reports by 08:55.
