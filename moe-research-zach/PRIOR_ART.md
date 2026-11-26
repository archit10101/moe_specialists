# PRIOR_ART.md — Causal Semantic Chunk Routing (CSR)

Status: Stage A audit, 2026-10-31. Sources were read from the primary arXiv HTML pages, official GitHub
repositories and Hugging Face cards. Items marked *(summary)* were read only through a model-extracted
summary of a truncated page and must be checked against the PDF before they are quoted in a paper.
No novelty is claimed in this document.

## Sources read

| Work | Primary sources | Revision seen |
|---|---|---|
| FLAME-MoE (arXiv 2505.20225) | arxiv.org/html/2505.20225; github.com/cmu-flame/FLAME-MoE (`configs/model/flame-moe.sh`, `configs/train/flame-moe.sh`, `scripts/release/flame-moe-38m.sh`, training modules); HF `CMU-FLAME/FLAME-MoE-38M-100M` | HF commit `c2b9dcae2f6785463ca7fae813536e12c439b798`; Megatron submodule `yuzc19/Megatron-LM@cbaf684c5d03997e0fdd5347c5e2d371c381a3d8`; GitHub HEAD hash not visible |
| EMO (arXiv 2605.06663 v2) | arxiv.org/abs + html; github.com/allenai/EMO; HF collection allenai/emo | v2 (10 May 2026); repo commit hash not visible |
| Lory (arXiv 2405.03133 v2) | arxiv.org/html/2405.03133 (HTML truncated about halfway; the rest is *(summary)*) | v2 (19 Aug 2024) |
| Empirical MoE design choices (arXiv 2402.13089 v1) | arxiv.org/abs + html v1 | v1 |
| nanoMoE | github.com/wolfecameron/nanoMoE (`model.py`, `config/train_nano_moe.py`, LICENSE) | `661fbcfade3e3081b3ed1150bc07e49082f46b48` (MIT) |

## Facts that constrain this project

- **EMO** (Wang, Bhagia, Min, 2026). The model has 128 experts: 127 routed and 1 shared, with k=8.
  The document pool is `D = TopK((1/T) Σ_t p_t, d)`, where d is sampled uniformly from {k,…,127}. Per-token
  probabilities are renormalized inside the pool, then top-k is applied. No domain labels are used:
  "using document boundaries as the only grouping signal". The pool averages router probabilities over
  **all T tokens of the document**, so as written it is not causal for early tokens. The paper does not
  appear to address this point (inferred from the formula). Load balancing is global, because micro-batch
  load balancing "conflicts with our document-level routing constraint". Checkpoints are 1B-active/14B-total
  and 4B-total, and loading needs `trust_remote_code=True` (custom fork `ryanyxw/transformers`).
- **Lory** (2024). Segments are fixed at T=256 tokens. Segment k is routed by the mean hidden state of segment
  k−1, which is causal, and the routing weights mix expert parameters (soft merging, SMEAR-style), not top-k.
  The first segment routes from itself with a stop-gradient. Similarity-based batching chains documents with
  Contriever embeddings. The paper reports "domain-level specialization without supervision". Prefix-only
  routing (route once from the first segment) is "much worse" than causal segment routing.
- **2402.13089** (2024). On GPT-2-small, N=4, top-2: "Sequence-level routing can result in topic-specific weak
  expert specialization, in contrast to syntax specialization observed with Token-level routing". Token-level
  routing gave lower validation perplexity: 9.907 vs 10.667 for sequence-level layer-wise. A frozen random
  router was comparable to a learned one.
- **FLAME-MoE-38M-100M**. 9 layers, d=256, 16 heads, RoPE, SwiGLU, RMSNorm, untied embeddings, Pythia
  tokenizer. Layer 0 is dense (FFN 1368) and layers 1–8 are MoE: 64 experts (FFN 176), **top-6 routed plus
  2 shared experts** (shared FFN 352), so 8 experts are active. The task text says "top-8, two-shared";
  the official configs show 6 routed plus 2 shared. Routing is a softmax over all 64 experts before top-k,
  with no top-k renormalization (paper Eq. 1). The router runs in fp32 and dispatch is all-to-all.
  No capacity factor is set, so the model is presumably dropless (inferred from Megatron defaults, not verified).
  aux_loss_coeff is 0.01 and z_loss_coeff is 0.001. Training ran on 4.4B DCLM tokens across 32×H100 with EP=8.
  The HF release is a Megatron `torch.distributed.checkpoint` (16 `.distcp` shards plus a pickled
  `common.pt` that references `megatron.*` classes). It has no HF config, tokenizer or modeling code.
- **nanoMoE**. Top-k is taken on logits, with a softmax over the top-k only, so weights are renormalized.
  The capacity factor is 1.25 for training and 2.0 for evaluation, so tokens are dropped. It uses a Switch
  aux loss of 0.01 and a z-loss of 0.001, and every second layer is MoE. Its routing semantics differ from
  FLAME's (renormalization and dropping).

## Claim-by-claim matrix

| # | Component of CSR | Closest prior method | Exact overlap | Difference in CSR | Remaining uncertainty |
|---|---|---|---|---|---|
| 1 | Tokens in a text unit share a candidate expert pool, with top-k inside the pool | **EMO** | Same mechanism: a pool of d ≥ k experts from averaged router output, then a per-token top-k inside the pool with renormalization. No labels. | The unit is a sub-document chunk, not a document. The pool comes from the **causal** mean router input of the first w tokens, not from the full document. m is fixed (2k), not sampled. | Whether EMO's inference path, which this audit did not read, already uses a prefix-based pool. Whether chunk-level pools are in EMO's appendix (only *(summary)* was read). |
| 2 | Causal routing from a prefix or previous segment | **Lory** | Lory routes segment k from segment k−1's mean hidden state, which is causal. | CSR routes the rest of a chunk from the chunk's **own observed prefix** (first w tokens). It keeps sparse top-k, not soft merging, and its chunk boundaries vary rather than being fixed at 256. | Lory's appendices on variable segments were *(summary)* only. Other segment-routing work (e.g., Shen & Henderson 2026 "Temporally extended MoE", arXiv 2604.20156, cited by EMO) was **not read** and may overlap substantially. |
| 3 | Sequence- or segment-level routing produces topic-level specialization | **2402.13089**, Lory, EMO | All three report topic- or domain-like specialization from coarser routing. | No difference in this claim. It is prior work and CSR re-tests it. | None. This is established. |
| 4 | Label-free semantic grouping | EMO (document boundaries), Lory (Contriever similarity batching) | Lory already uses a frozen external encoder for semantic similarity at the **data-batching** level. | CSR uses a frozen encoder for (a) **online causal boundary detection** and (b) a **relational KL target on router-pool similarity**, not for batching. | Topic-segmentation-driven routing and contrastive or relational router regularizers may exist in the literature but were not searched. ModuleFormer (cited by EMO) maximizes token–expert mutual information, which is a related but distinct auxiliary. |
| 5 | Online causal semantic boundary detection used for routing | None found in the sources read | — | Sentence-completion-triggered cosine drop against a running chunk representation, with a 64/256 min/max. | Classical text segmentation (TextTiling and later) is well known. Using it causally to drive MoE pools may already be published. **Novelty is unverified.** |
| 6 | Relational alignment KL(Q‖R) between encoder affinities and router-pool affinities | Knowledge-distillation-style relational losses (not audited here) | The general form (match a teacher similarity distribution) is standard relational KD. | It is applied to per-layer chunk router distributions in an MoE. | **Unverified.** A matched-strength non-semantic regularizer control is required before any attribution to semantics (§9 of the task). |
| 7 | The specific combination 1+2+5+6 under controlled ablations A–F | — | — | — | This is a **novelty hypothesis only**, pending a broader literature review (BlockFFN 2507.08771, Shen & Henderson 2604.20156, ModuleFormer, topic-segmentation literature). |

## What this project must not claim

- Shared pools (EMO), causal segment routing (Lory), and label-free topic specialization (Lory, EMO,
  2402.13089) are each prior art.
- If the chosen model is not pretrained at scale (see MODEL_DECISION.md), results say nothing about
  pretraining efficiency or emergent modularity at scale.
- Routing–semantics correlation is not functional specialization. That requires held-out language-modeling
  evidence or interventions (§15 of the task).
