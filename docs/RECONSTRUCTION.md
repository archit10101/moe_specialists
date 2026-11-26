# Reconstruction choices

This is an independent reimplementation built only from `docs/source_history.md`; we never saw
the original code or data. This page lists every place where the history doc does not pin down a detail
and what we chose instead. If a result fails to reproduce, start here.

Legend: **exact** = taken from the doc · **chosen** = our decision (reason given).

## Data

| item | value | status |
|---|---|---|
| corpus | M2D2 arXiv level-1 archives `cs_l1, math_l1, physics_l1, q-bio_l1, stat_l1` (pinned commit `6ae613b…`) | chosen: §11 names "five balanced M2D2 arXiv domains"; the later 103.6M line only says "papers" and "domain" |
| "paper" | M2D2 stores **one paragraph per line with no paper boundaries**. A paper is a run of consecutive paragraphs of about 2,048 tokens (`DataCfg.doc_tokens`). Consecutive paragraphs come from the same paper in order, so a pseudo-paper is mostly one paper, sometimes the end of one plus the start of the next | chosen (forced by the data). To use a corpus with real paper boundaries, set `source="jsonl"` |
| splits | paper-disjoint contiguous ranges per domain: test 1M, val 1M, calib 1M, bank 10M, base 49.2M tokens | chosen; base ≥ 245,032,457 tokens in total (exact) |
| tokenizer | GPT-2 BPE (tiktoken), EOT after each paper, vocab padded to 50,304 | chosen (same as your earlier projects) |
| batches | non-overlapping 1,024-token windows in a seeded random order. Every condition with the same seed sees identical rows ("paired seeds; identical rows", §2.6) | exact in spirit |

## Base model (§2.2)

| item | value | status |
|---|---|---|
| width / layers / experts / top-k / ctx | 384 / 12 / 32 / 4 / 1,024, shared expert, unrestricted top-4 | exact |
| expert sizes | SwiGLU routed experts with 128 hidden units, shared SwiGLU with 1,536 → **104.8M params** | chosen: the doc says 103,609,728 but not the split; we are 1.2% higher |
| layer 0 | MoE like every other layer | chosen: the doc says "layer 0 dense" only for the 28.9M NanoMoE |
| tokens / steps | 245,032,457 tokens at 65,536 tokens/step = 3,739 steps (doc: 3,880 steps) | tokens exact; batch chosen |
| optimizer | AdamW (0.9, 0.95), wd 0.1, lr 1e-3 cosine to 1e-4, 2% warmup, clip 1.0, bf16 | chosen (your subject-moe defaults) |
| aux losses | `CE + 0.01·Switch + 0.001·z` (layer means) | coefficients exact (§19); averaging over layers chosen |

## EDL (§3, §17–18)

| item | value | status |
|---|---|---|
| loss | `mean over group pairs −log(JSD(R_gi, R_gj) + ε)` on the full router softmax, all MoE layers | exact |
| ε | 1e-6, and JSD clamped at ≥ 0 before the log (float error otherwise gives NaN on near-identical soft groups) | chosen |
| explicit β | 5e-3 | exact |
| groups per step | computed per micro-batch (16 windows ≈ 16k tokens); groups with < 32 tokens are skipped | chosen |
| Semantic C6 | all layers, **sharpened** soft per-paper cluster posteriors, "2% strength" | exact wording. We read "2% EDL-relative strength" as β auto-calibrated at step 0 so that ‖∇EDL‖ = 2% of ‖∇LM‖ (the doc reports explicit EDL at ~5% early / ~2% late). The measured ratio is logged in `train_log.json → grad_ratio` |
| clusters | e5-base-v2 on the first 512 tokens of each paper ("passage: " prefix), spherical k-means with Sinkhorn-balanced assignments, K=16 | e5 + balanced Sinkhorn exact (§14); K=16 chosen (FL-EDL and the oracle bank use 16) |
| soft / sharp | Sinkhorn posteriors at temperature 0.05 (soft) / 0.02 (sharp) on cosine similarity | chosen |
| shuffled | rows of the cluster table permuted across papers: same cluster sizes, random membership | exact ("persistent, mass-matched random paper groupings") |

## Metrics (§2.3–2.4)

| item | value | status |
|---|---|---|
| NMI | `I(group; expert)/sqrt(H·H)` over all top-k slots, averaged over layers | chosen; absolute values may not match the doc's scale |
| FSI | remove each group's 2 most-enriched experts per layer (by routing lift) in all layers at once; FSI = mean over groups of [ΔCE on group − ΔCE on other groups] | chosen (the doc gives only the idea) |
| per-expert interaction | ban (layer, expert); the next-ranked expert is admitted, k stay active, downstream recomputed; own group = highest-lift group | exact procedure; group choice chosen |
| exposure matching | routed tokens only, bins of rank × gate tertile × margin-to-replacement tertile × baseline-CE quartile × position quartile | exact covariates (§2.4); bin counts chosen (ported from Project-davin) |
| probe data | calib split; 24 windows × (largest group count) | chosen |

## Residual bank (§23)

| item | value | status |
|---|---|---|
| placement / form | `h' = h + a_e(RMSNorm(h))` after the final MoE block, before `ln_f` | exact |
| specialist | 384→256→384, GELU, no bias; 16 → 3,145,728 params | exact (the parameter count matches exactly) |
| RMSNorm | no learnable weight (keeps the count exact) | chosen |
| init | output matrices zero, so every bank starts exactly at the base | chosen |
| training | 50M new tokens (bank split), AdamW lr 1e-3 cosine, no weight decay, base frozen | tokens exact; optimizer chosen |
| token router (C / G) | linear on RMSNorm(h) → top-1, straight-through gate scale p/p.detach(), Switch balance 0.01 | chosen |
| specificity | double-centred diagonal of the forced swap matrix (removes cluster difficulty and specialist main effects); hierarchical bootstrap over papers within clusters | chosen; the doc's exact formula is unknown, so compare signs and the semantic − shuffled contrast |

## Router (§27)

| item | value | status |
|---|---|---|
| architecture | `[RMSNorm(h_t), RMSNorm(mean h_{t-15..t})] → Linear(768→64, no bias) → GRU(64) → Linear(64→16)` | exact (75,152 params matches exactly). RMSNorm on the context mean chosen |
| teacher | `q ∝ exp(u/(2·s_u)) · prior^0.5`, with u the centred exact utility (−CE) and s_u its running std | "utility temperature 2·s_u, semantic weight 0.5" exact; form chosen |
| loss | `KL(q‖p) + 0.01·Σ p_e CE_e` | **ambiguous**: the doc says "distillation weight 0.01". We put 0.01 on the expected-CE term and 1 on the teacher KL. B10 `abl.router_ce_only` tests the opposite extreme |
| prior | the paper's soft semantic cluster posterior, used in training only | chosen ("semantic region prior") |
| bank | the §23 semantic bank (whole-paper oracle) | chosen: the doc reuses banks from §25–26 (128-token-view assignments) we did not rebuild |
| G | `bank.tok` with the same base and the same seed as the routed bank | exact in spirit ("matched generic residual bank") |
| not reproduced | the 50M joint-continuation negative result (§27 "continued joint training") | out of scope |
