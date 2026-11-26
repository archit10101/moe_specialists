# Pre-GPU audits (task §18)

Date: 2026-10-31. Each audit is backed by code references and the tests named below.

## 1. Mathematical audit

| Item | Implementation | Verified by |
|---|---|---|
| Router | `z = W_r h`, fp32 even under bf16 autocast (`MoE.router_logits` disables autocast). h = router input (post-RMSNorm). | `gpu_checks.py` (tiny reference on GPU) |
| Token routing (A) | softmax over all E=64 → top-k=6 by probability → gate = global probability (no renormalization) → dropless dispatch (padded bmm) + shared SwiGLU(352). No capacity factor, no dropping. | `test_baseline_matches_independent_reference` (naive per-token loop: logits, aux, z, routing equal at 1e-10) |
| Pool | For chunks with a complete w-prefix: `hbar = mean(h[a..a+w-1])`, `P = TopM(W_r hbar)` per layer (M=12). TopM is applied to detached logits (discrete, no gradient). | `test_constrained_experts_inside_pool_and_k_active` |
| Constrained selection | Positions with chunk_pos ≥ w: the original top-k by global probability restricted to P; gates = global probabilities (`gate_norm="global"`). Exactly k experts; m=E reduces exactly to A. | `test_pool_with_m_equal_E_reduces_to_token_routing`, `test_gate_norm_semantics` |
| Normalization order | softmax(all) → restrict → top-k → gate = global prob. EMO-style renormalization would change gate scale (≈0.09 → ≈0.5 near init), so it is not the default. | MODEL_DECISION.md |
| Shapes | logits (N,E); pools (R,E) bool; candidate mask (B·T,E); `chunk_logits` (R,E) per layer, stacked (L,R,E) for the semantic loss | tests |
| Masking | Block-diagonal causal attention over doc_ids; pad rows attend only to themselves (no NaN); targets −100 at each document's last token and at pads | `test_document_isolation_in_packed_rows`, `test_make_batch_targets_never_cross_documents` |
| Causality | A token at chunk_pos ≥ w uses hbar from positions a..a+w−1 < its own position. Boundaries for position t+1 depend on tokens ≤ t. u_i uses only the prefix and is a training target, never a model input. | `test_future_tokens_do_not_change_past` (token and pool), `test_streaming_equals_batch_and_is_prefix_causal`, `test_embed_fn_never_sees_unobserved_tokens` |
| Incremental path | Pool cached only after w prefix tokens are observed (assertion); equals the parallel path | `test_incremental_matches_parallel` |
| Aux loss | Per layer `E·Σ f_i P_i` (Megatron switch form); P from full probabilities, f from actually selected experts; summed over layers; z-loss = mean logsumexp² per layer, summed; non-pad tokens only. The same formula in every condition. | reference test (aux, z equal) |
| Semantic loss | Q_i, R_i over j≠i (diagonal logit −1e9, so exp = 0 exactly); KL summed over j, mean over anchors and layers; fp32; u detached and renormalized; duplicates (prefix hash), non-finite and zero rows removed; skipped if N < min_chunks with a zero-valued graph | `test_semantic_loss.py` (uniform, collapsed, identical u, extreme temperatures, permutation, gradient to every logit, detached targets) |
| Gradient flow | Through `p = softmax(W_r hbar)` to W_r and to hbar, hence to lower layers. The LM gradient reaches the router through gates. TopM itself has no gradient. | `test_gradients_finite_and_frozen_unchanged` |
| Accumulation | LM term = Σ CE_sum / N_valid(step), exactly equal to the full-batch mean. Aux and semantic terms are weighted by each micro-batch's valid-token share. | `test_grad_accumulation_matches_full_batch` (1e-12), `test_aux_terms_weighted_by_token_share` |

**Collapse cases noted.**
- p·p dot products range from 1/E (uniform) to 1 (one-hot). A uniform router gives a uniform R, and the loss
  then pushes the chunk distributions apart. A collapsed router gives a near-uniform R over off-diagonal pairs.
  Both stay finite (tested).
- Popular experts inflate every p_i·p_j, which is why τ_r is calibrated against Q's effective neighbour count.
- Chunk-router entropy, max probability and argmax mode fraction are logged every step (`sem_stats`).

**Known approximations.**
- The sentence rule splits on abbreviations ("b. 1638").
- The semantic boundary is detected with a one-sentence delay by design. The delay is logged.
- Documents longer than 1,023 tokens are truncated.

## 2. Fairness audit

**Matched by construction and checked by `config.check_matched`** (fails closed; only `segmentation.strategy`,
`routing.mode`, `semantic.enabled` and `condition`/`run_name` may differ):
- checkpoint and initialization, tokenizer, model shape, k, m, w, gate norm
- precision, dropout (0), compile (off), activation checkpointing (off)
- data directory and splits hash, packing, train order seed
- optimizer, schedule, clipping, batch rows, token budget, seed
- aux and z coefficients and their scope
- λ, τ_s and τ_r (identical values even where the loss is disabled), evaluation split and rows

Runtime facts (trainable-parameter-name hash, valid training tokens, evaluation tokens) are compared too
(`test_train_all_conditions_tiny_cpu_matched`).

**Identical across conditions** (`test_same_valid_tokens_and_k_across_conditions`): packed rows, targets and
valid-token counts, evaluation documents and positions, and k plus the shared expert for every token.

**Residual confounds** (reported, not hidden):
1. **Coverage.** The fraction of tokens that are pool-constrained depends on the chunk-length distribution.
   It is reported per condition. **Correction (2026-10-31):** F-v1 was *not* length-matched to D: it censored
   the already document-censored distribution a second time, giving a non-final mean of 122.0 and 2.9% more
   chunks. It is superseded, and `train.py` refuses it. F-v2 (hazard closer) is pending acceptance; see
   EXPERIMENT_REPORT.md §2a. B (fixed 128) differs
   from D.
2. **External knowledge.** The frozen MiniLM encoder carries external semantic knowledge, both into the D/E/F
   boundaries (F inherits only the length distribution) and into C/E as a distillation target. "Label-free"
   does not mean "information-free".
3. **Initialization regime.** If every condition adapts a token-routed base model, A is in its home regime
   (MODEL_DECISION.md §5).
4. **Compute.** Active expert FLOPs are identical. B–F add one R×d×E router product per layer, which is
   negligible. C and E add the semantic loss. D, E and F (via fitting) add CPU preprocessing, which is timed
   in the segmentation manifest.
5. **Truncation.** Truncation to 1,024 tokens affects all conditions equally.

## 3. Feasibility audit

| Item | Status |
|---|---|
| Checkpoint | FLAME-MoE-38M-100M is not loadable or validatable within constraints (MODEL_DECISION.md). The from-scratch FLAME-38M-matched model is used: 100.7M total, 37.8M active parameters. |
| Data | 16 FineWeb-Edu row groups (16,000 documents, 0 exact/near duplicates). Train has 8.32M valid tokens, enough for 3M (pilot) and 5M (extension) per condition. Base pretraining would need more data, under separate authorization. |
| Memory | Estimated per micro-batch of 16×1,024: logits about 3.3 GB fp32, activations about 2–4 GB, expert buffers under 1 GB. Expected well under 56 GiB; measured in Stage B. |
| Throughput | NOT MEASURED (Stage B) |
| **Disk (binding)** | Reserve of 100 GiB, against 101.2 GiB free after Stage A. CUDA PyTorch (`torch==2.5.1+cu121` plus nvidia-* wheels, about 4–5 GB) **does not fit** under the conservative reserve. Stage B cannot install a CUDA build without user approval of either a decimal reading of the reserve (100e9 B, about 8.6 GB headroom) or freed space. |

## 4. Safety audit

Covered in SAFETY.md:
- Paths: every artifact is under the root, with no symlinks (asserted after downloads).
- Permissions: 0700 root, 0600 ledgers, 0400 published data.
- Caches: in-root via `env.sh`.
- Network: pinned, metadata-first.
- Process registry: in place.
- Occupancy gate and monitor: in place.
- VRAM, time and thread limits: in place.
- Cleanup: none performed; no destructive commands.

Findings fixed during the audit:
1. The root was group-writable (0775) and is now 0700.
2. The pool gating default was changed to "global" to remove the gate-scale confound.
3. Three short pytest invocations ran without the registry wrapper. They are recorded as NOTE entries in the
   ledger; later runs use the wrapper.
