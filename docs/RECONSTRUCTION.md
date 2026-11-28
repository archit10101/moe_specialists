# Reconstruction choices

This repo started as an independent reimplementation built only from `docs/source_history.md`. Later the
original author's code became available (`moe-research-zach/` on `main`). It has the code, configs and
pre-registrations, but no checkpoints, data or result files. We then aligned every choice that could change a
conclusion with the original code, and we list here the differences we kept. If a result fails to reproduce,
start here.

What this means for the write-up: the experiment code is still ours (nothing was copied), but its settings now
follow the original's. Treat the result as a **reproduction with the original protocol**, not a blind one.

Legend:
- **original**: taken from the original code; the file is named.
- **exact**: taken from the doc.
- **chosen**: our decision, with the reason given.
- **differs**: a known difference from the original that we kept, with the reason.

## Data

| item | value | status |
|---|---|---|
| corpus | M2D2 arXiv level-1 archives `cs_l1, math_l1, physics_l1, q-bio_l1, stat_l1`, pinned commit `6ae613b…`, both `train.txt` and `valid.txt` | original (`dfc_data.py`) |
| "paper" | a block of `train.txt`/`valid.txt` separated by blank lines (runs of ≥ 2 newlines); one paragraph per line inside a paper; about 12k tokens per paper | original (`dfc_data.py _iter_docs`). The first version wrongly assumed M2D2 had no paper boundaries and used 2,048-token pseudo-papers |
| dedup | none | **differs**: the original removed exact and MinHash near-duplicates and kept a paper cross-listed in several domains once, and capped cs/math candidates. We skipped this for lack of time |
| splits | paper-disjoint contiguous ranges per domain: test 1M, val 1M, calib 1M, bank 10M, base 49.2M tokens | **differs**: the original assigned papers by hash, used only papers that were not cross-listed for the eval splits, and had a 25M-token VAL. Our splits are smaller but paper-disjoint |
| tokenizer | GPT-2 BPE (tiktoken), EOT after each paper, vocab padded to 50,304 | **differs**: the original used the Pythia tokenizer (also 50,304). Absolute CE is therefore not comparable with the doc's ≈3.05; the paired contrasts are |
| batches | non-overlapping 1,024-token windows of the concatenated stream in a seeded order; windows can cross paper boundaries | **differs**: the original packed ≤1,024-token segments, restarted attention and positions per segment, and used one fixed order for every condition and seed. With 12k-token papers few windows cross a boundary |

## Base model (§2.2)

| item | value | status |
|---|---|---|
| width / layers / heads / experts / top-k / ctx | 384 / 12 / 6 / 32 / 4 / 1,024 | exact |
| layer 0 | dense tanh-GELU FFN, 1,792 hidden; blocks 1–11 are MoE (11 MoE layers, 352 experts) | original (`lfr_model.py`) |
| experts | 2-matrix tanh-GELU, 256 hidden, no bias; shared expert of the same form, 768 hidden | original. Together: **103,609,728 params, exactly the doc's count** |
| norms / positions | RMSNorm (learnable, eps 1e-6), RoPE (base 1e4), tied embeddings | original |
| router | linear on the block's normalised input, top-4 of the full softmax; **gate = raw softmax p, not renormalised** ("FLAME") | original |
| tokens / steps | 245,032,457 tokens at 65,536 tokens/step = 3,739 steps | tokens exact; the original took 3,880 steps of 64 rows that were not completely full |
| optimizer | AdamW (0.9, 0.95), wd 0.1, **lr 6e-4 WSD**: 1% warmup, constant, linear decay to 6e-5 over the last 20%; clip 1.0, bf16 | original (`fsp_lib.py`, `P1_*.json`) |
| aux losses | `CE + 0.01·Switch + 0.001·z`, Switch and z **summed** over the 11 MoE layers | coefficients exact (§19); summing original |

## EDL (§3, §17–18)

| item | value | status |
|---|---|---|
| loss | mean over group pairs of `−log(JSD(R_gi, R_gj) + 1e-8)` on the full router softmax, JSD clamped ≥ 0; mean over MoE layers | exact + original eps |
| explicit β | 5e-3 | exact |
| groups per step | per micro-batch of 16 windows; groups with < 32 tokens are skipped; R_g is the token-weighted group mean | **differs** (low impact): the original averaged per segment first, then over the segments of a group, with no minimum mass |
| Semantic C6 / shuffled C6 | all layers, sharpened soft labels, **fixed β = 5e-3** | original (`P1_C6`/`P1_SHUF` configs). "2% strength" in the doc only describes the resulting gradient ratio, which is logged in `train_log.json → grad_ratio` |
| clusters | each paper → up to 4 views of 256 tokens spread over the whole paper; bge-small-en-v1.5 CLS embeddings, z-scored on the fit papers; K = 16 centroids; q = softmax(mean over views of cos/0.1) with **no balancing at assignment**; hard = argmax q | original (`gc_lib.py`), except **differs**: the original fitted a SwAV-style prototype network (384→64 projection, batch Sinkhorn targets, 3 seeds, medoid seed kept). We fit spherical k-means with balanced assignments on the same embeddings. Papers shorter than 256 tokens get one view of the whole paper instead of no label |
| sharp (C6) | q² renormalised | original (`sem_tables.py`) |
| shuffled | hard labels and rows permuted across papers **within each split and within 20 paper-length quantile bins**: the same papers per cluster as the semantic table in every split, about the same token mass, random content | original for the bins (`orv1_tables.py`); per split chosen (the original's §18 shuffle covered only the training split) |

## Metrics (§2.3–2.4)

| item | value | status |
|---|---|---|
| NMI | `I(group; expert)/sqrt(H·H)` over all top-k slots, averaged over MoE layers, on val | **differs**: the original used `2I/(H+H)` on 200k VAL tokens per domain, so absolute values are not comparable; rankings are |
| FSI | remove each group's 2 most-enriched experts per layer (by routing lift) in all layers at once; FSI = mean over groups of [ΔCE on group − ΔCE on other groups], on calib | **differs**: the original removed every lift > 1 expert of the group, matched other-group tokens on baseline-CE deciles, normalised by group size, and fitted groups on one calibration half and measured on VAL. Not comparable with the doc's FSI numbers |
| per-expert interaction | ban (layer, expert); the next-ranked expert is admitted and downstream recomputed; own group = highest-lift group; ordinary and exposure-matched (routed tokens, bins of rank × gate × margin × base CE × position) | **differs**: the original §18 F test fitted the own group on one calibration half, measured on VAL over all tokens, matched on base-CE decile × position bin, and averaged over all 352 experts. Treat ours as a sign check |

## Residual bank (§23)

| item | value | status |
|---|---|---|
| frozen base | `base.token4` **seed 0 for every bank seed** (final checkpoint; B4 uses its 10/25/50% checkpoints). B8 puts bank seed 0 on TOKEN-4 seeds 1–3 | **differs**: the original put bank seed s on TOKEN-4 seed s. Changed for the paper tiers, which train one base to save GPU time. A bank seed therefore measures bank-training variance only (init + data order), and base-to-base variance is tested separately by B8. `base_seed=-1` restores the original pairing |
| placement / form | `h' = h + a_e(RMSNorm(h))` after the final block, before the final norm; RMSNorm without weight | original |
| specialist | 16 × (384→256→384), exact GELU, no bias = 3,145,728 params; W1 ~ N(0, 0.02²), W2 = 0 so every bank starts at the base | original (tanh-GELU there, a negligible difference) |
| semantic oracle | the paper's hard `sem` label (argmax q, see Clusters), whole-paper | original |
| shuffled oracle | `sem_shuf` (see shuffled) | original |
| token router (C / G) | linear on RMSNorm(h) → top-1, gate scale p/p.detach(), Switch balance 0.01, no capacity limit | original |
| training | 50M bank-split tokens the base never saw; 32×1,024 tokens per step (~1,526 steps); AdamW (0.9, 0.95), **lr 3e-4, 50-step warmup then constant, wd 0.1** | original (`orv1_paths.py`) |
| data order | identical rows for every condition (sem / shuf / tok / dense) within a seed; a different order per seed (`stream_seed=-1` → 1000 + seed) | **differs**: the original used one order for every seed. With a single base, a fixed order would leave the bank initialisation as the only source of seed variance, which understates the uncertainty. `stream_seed=1000` restores one order |
| specificity | double-centred diagonal of the forced swap matrix (≈0.94× the original's for K = 16), computed on **all val tokens** (`EvalCfg.swap_tokens`, like the original's full-VAL matrices; a 384-window probe covered only ~30 real papers and missed clusters); hierarchical bootstrap over papers within clusters | **differs**: the original's `functional()` compared own-cluster vs other-cluster tokens inside 675 strata. Compare signs and the semantic − shuffled contrast |

## Router (§27)

| item | value | status |
|---|---|---|
| architecture | `[RMSNorm(h_t), mean of RMSNorm(h) over t−15..t] → Linear(768→64, no bias) → GRU(64) → Linear(64→16)`, 75,152 params | original |
| training data | the `calib` split: papers that neither the base, the bank nor val saw | original (paper-disjoint router-train split) |
| teacher | π = softmax(u/(2·s_u) + 0.5·log(prior + 1e-4)); u = utility of each specialist; s_u = median over tokens of the per-token SD of u across specialists, fixed once before training | original (`ccsr1_router.py`, `ccsr1_router_job.py`) |
| loss | Σ_t w_t KL(π_t‖p_t)/Σ_t w_t + **0.01·Σ_k p̄_k log(K p̄_k)** (batch-level balance); w_t = clip(margin/median margin, 0.25, 4) | original. The doc's "distillation weight 0.01" is the balance weight; there is no expected-CE term (B10 `abl.router_ce_only` keeps one as an ablation) |
| optimizer | AdamW lr 3e-3, betas (0.9, 0.99), wd 0.01, 30-step warmup, cosine to 10%, clip 1.0; 2,400 steps × 65,536 tokens | original |
| prior | the paper's soft semantic posterior q (training only) | **differs**: the original used a region-level prior (one per 256-token region), so its teacher had to follow topic switches inside a paper |
| banks routed over | the §23 oracle banks (`bank.sem` / `bank.shuf`) and G = `bank.tok`, same base and seed | **differs**: the original reused §26 banks trained jointly with a base that had seen only 20M tokens, with region-level groups. Our §27 therefore tests "can a causal router deploy the §23 paper-level specialists", which is a different (likely easier) routing task. State this in the write-up |
| metrics | router − G CE, retention (r_shuf − r_sem)/(o_shuf − o_sem) with the paper oracle, exposure-matched interaction, routing NMI, position categories by window position / paper boundary | **differs**: the original's retention denominator is its region-prior teacher; its interaction is the csur1 estimator (real C6 labels for both banks, cross-fitted lift map, 2,025 strata); its positions are document positions and region switches. The router − G sign and the semantic vs shuffled ordering are comparable; the exact numbers are not |
| not reproduced | the 50M joint-continuation negative result (§27 "continued joint training") | out of scope |
