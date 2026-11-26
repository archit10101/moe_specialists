# ORACLE-RESIDUAL-V1 — Label-Free Semantic Residual Capacity Test: Report

**Status:** TERMINAL_S3_PROMOTED. All stages ran (0, 1, 2 and 3). Nothing is running.

**Primary outcome (prereg §9):** A — semantic residual functions exist. All Stage-3 final criteria V1–V8 pass on VAL, which was opened once. The test split was never opened or tokenised.

**What this does and does not show.** This is an *oracle* diagnostic. Condition A's assignment is computed from whole-paper embeddings. It is therefore non-causal (it uses text after the current token) and paper-level. Nothing here supports deployability.

- **Prereg:** `docs/ORACLE_RESIDUAL_V1_PREREG.md`, sha256 `e39cfc1f…dc564`, mode 0400. It was frozen before any scientific job.
- **Artifacts:** `artifacts/oracle-residual-v1/`.
- **Diagnostic notes:** `ORACLE_RESIDUAL_V1_DIAGNOSTIC_NOTES.md`.

## 1. Question

Do label-free semantic paper clusters correspond to different predictive functions, such that giving each cluster separate parameters helps? Each condition adds one K=16 residual-specialist bank to a frozen TOKEN-4 base, after the final MoE layer:

- h' = h + a_e(rms0(h)), with the final projection zero-initialised;
- 3,145,728 parameters per bank;
- exactly one specialist active per token.

The four conditions:

- **A — SEMANTIC-ORACLE.** The hard argmax of the frozen GC-EDL SEM K16 paper assignment. It is label-free; no M2D2 labels were used anywhere.
- **B — SHUFFLED-ORACLE.** A permutation of paper→cluster within 20 token-mass quantile bins. It preserves papers per cluster and approximate token mass.
- **C — TOKEN-ROUTED.** A learned causal top-1 linear router (+6,144 parameters), with a Switch auxiliary loss of weight 0.01.
- **D — the frozen base.**

Δ_CE = CE_D − CE_X (positive means better than the base).

## 2. What was run vs reused; preregistered vs post hoc

**Reused (read-only, never copied):** the TOKEN-4 seed 0–3 checkpoints (ckpt_p100, 245.03M tokens), and the GC-EDL SEM K16 prototype network, residualizer and CALIB-SEARCH table. All were verified by sha256.

**Run in this experiment:**

| stage | what | seeds | budget | eval partition |
|---|---|---|---|---|
| 0 | CPU tests 9/9; GPU tests G-ZERO, G-HASH, bench, bitwise resume; LR pilot {3e-4, 1e-3, 3e-3} | 0 | 5M | SEL (CALIB-SEARCH half A) |
| 1 | joint A/B/C training | 0, 1 | 20,000,000 tokens (633 steps) | EVAL (CALIB-SEARCH half B, 6.09M tokens, 498 papers) |
| 2 | continued from 20M | 0, 1 | 50,000,000 tokens (1,581 steps) | EVAL |
| 3 | seeds 2, 3 trained to 50M; seeds 0, 1 Stage-2 checkpoints reused (budget rule); VAL evaluated once | 0, 1, 2, 3 | 50M (actual 50,022,549 tokens) | VAL (24,995,467 tokens, 2,015 papers) |

**Training details (all conditions identical; one process; shared base forward):**
- LR 3e-4, chosen by the pilot. It is the lowest grid point; the grid was not extended, per the prereg.
- AdamW with betas (0.9, 0.95), weight decay 0.1, clip 1.0.
- 50-step warmup, then constant LR; 32 rows of 2,048 tokens per step.
- Data: 367M base-unseen main-split tokens in hash order.

**Stage-3 budget rule (SEL only):** the mean CE gain from 20M to 50M was 0.00088, below 0.001, so the budget stayed at 50M.

**Preregistered:** all promotion criteria P1–P7, the Stage-2 rules, the final criteria V1–V8, the functional metrics F1–F4 and the interpretation table.

**Post hoc and descriptive only:**
- the gaps by token position, and the mutual information between C's routing and the semantic label (`posthoc_descriptive_S3_VAL.json`, §7);
- no decision used them.

## 3. Results — absolute CE (VAL, Stage 3, 50M, per seed)

| seed | CE_D (base) | CE_A | CE_B | CE_C | Δ_CE_A | Δ_CE_B | Δ_CE_C | CE_B − CE_A | CE_C − CE_A | CE_B − CE_C |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 3.05021 | 3.04668 | 3.04912 | 3.04797 | +0.00353 | +0.00109 | +0.00224 | +0.00244 | +0.00129 | +0.00115 |
| 1 | 3.05756 | 3.05418 | 3.05660 | 3.05547 | +0.00338 | +0.00096 | +0.00209 | +0.00242 | +0.00129 | +0.00113 |
| 2 | 3.05850 | 3.05499 | 3.05753 | 3.05639 | +0.00350 | +0.00096 | +0.00211 | +0.00254 | +0.00140 | +0.00114 |
| 3 | 3.04673 | 3.04366 | 3.04588 | 3.04476 | +0.00307 | +0.00085 | +0.00197 | +0.00222 | +0.00110 | +0.00112 |
| **pooled** | | | | | **+0.00337** | +0.00097 | +0.00210 | **+0.00240** | **+0.00127** | +0.00114 |
| hier. 95% CI (10k, seed × paper) | | | | | [+0.00314, +0.00357] | [+0.00087, +0.00106] | [+0.00199, +0.00221] | [+0.00224, +0.00256] | [+0.00112, +0.00141] | — |

**Per-seed signs:** every difference is positive in all 4 of 4 seeds.

**Size:** Δ_CE_A is about 0.11% of CE, so the effect is small in absolute terms but very consistent. Every condition beats the frozen base.

**Semantic vs shuffled.** A beats B in 4 of 4 seeds by +0.0024 nats; the CI excludes 0. B, which has an identical partition structure and token mass, gets only 29% of A's gain.

**Semantic vs token-routed.** A beats C in 4 of 4 seeds by +0.0013 nats; the CI excludes 0. C beats B by +0.0011.

**Earlier stages (EVAL, seeds 0 and 1, pooled, with 2,000-replicate paper-bootstrap CIs):**

| stage | Δ_CE_A | Δ_CE_B | Δ_CE_C | CE_B − CE_A | CE_C − CE_A |
|---|---|---|---|---|---|
| S1, 20M | +0.00205 [0.00189, 0.00219] (+0.00209 / +0.00200) | +0.00062 | +0.00136 | +0.00143 [0.00127, 0.00157] | +0.00068 [0.00053, 0.00083] |
| S2, 50M | +0.00350 [0.00328, 0.00371] (+0.00356 / +0.00344) | +0.00107 | +0.00216 | +0.00243 [0.00219, 0.00267] | +0.00134 [0.00112, 0.00154] |

**Stage-2 extra rules:**
- persistence: AB at 50M was 0.00243, which is at least 0.5 × the 20M value of 0.00143 → pass;
- no degradation: 0.00350 ≥ 0.00205 − 0.0005 → pass.

## 4. Final criteria (prereg §8.3) — VAL

| criterion | result |
|---|---|
| V1 pooled Δ_CE_A > 0, hierarchical CI lower bound > 0 | pass (lower bound 0.00314) |
| V2 A beats B in ≥ 3 of 4 seeds | pass (4 of 4) |
| V3 A beats C in ≥ 3 of 4 seeds, or near-tie + F1 + F2 | pass (4 of 4 beat it outright) |
| V4 hierarchical CI of CE_B − CE_A excludes 0 | pass [0.00224, 0.00256] |
| V5 P5 (≥ 12 of 16 clusters with Δ_CE_A ≥ 0) and P6 (not one cluster) | pass: 16 of 16 clusters; largest share of the positive A−B contribution is 0.157; leave-one-cluster-out AB ranges 0.00234–0.00247 |
| V6 base hashes unchanged | pass (every eval, every seed) |
| V7 identical budgets and compute; A = B in parameters; C has +6,144 router parameters | pass |
| V8 no pathology | pass (finite; C load max ≤ 0.082, effective number of experts ≥ 15.8; no dead banks) |

**Interpretation table:** only A applies. C, E, B and D do not apply (C − A has a lower bound > 0, so "token-level better" fails).

## 5. Per-cluster results (VAL, seed mean)

The last column of the CE-gap group shows the minimum over the 4 seeds.

| k | VAL tokens | Δ_CE_A | Δ_CE_B | Δ_CE_C | CE_B − CE_A (min seed) | U_own_A | U_oth_A | S_A | U_best_alt_A | S_B |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.93M | +0.0030 | +0.0009 | +0.0020 | +0.0021 (+0.0019) | +0.0046 | +0.0005 | +0.0041 | +0.0020 | +0.00010 |
| 1 | 1.81M | +0.0032 | +0.0009 | +0.0022 | +0.0023 (+0.0020) | +0.0051 | +0.0013 | +0.0038 | +0.0025 | +0.00000 |
| 2 | 1.66M | +0.0036 | +0.0008 | +0.0021 | +0.0028 (+0.0025) | +0.0050 | +0.0007 | +0.0043 | +0.0036 | −0.00007 |
| 3 | 1.06M | +0.0036 | +0.0011 | +0.0023 | +0.0025 (+0.0025) | +0.0052 | −0.0001 | +0.0052 | +0.0020 | −0.00001 |
| 4 | 1.14M | +0.0039 | +0.0008 | +0.0023 | +0.0031 (+0.0026) | +0.0057 | −0.0000 | +0.0057 | +0.0030 | −0.00000 |
| 5 | 1.66M | +0.0027 | +0.0010 | +0.0021 | +0.0017 (+0.0017) | +0.0045 | +0.0008 | +0.0037 | +0.0021 | −0.00004 |
| 6 | 0.78M | +0.0029 | +0.0010 | +0.0027 | +0.0020 (+0.0016) | +0.0045 | +0.0002 | +0.0042 | +0.0015 | +0.00001 |
| 7 | 2.05M | +0.0033 | +0.0011 | +0.0019 | +0.0022 (+0.0020) | +0.0045 | +0.0005 | +0.0040 | +0.0029 | +0.00002 |
| 8 | 1.00M | +0.0045 | +0.0010 | +0.0023 | +0.0035 (+0.0031) | +0.0064 | +0.0005 | +0.0059 | +0.0037 | +0.00004 |
| 9 | 1.36M | +0.0020 | +0.0008 | +0.0020 | +0.0013 (+0.0012) | +0.0032 | +0.0008 | +0.0024 | +0.0016 | −0.00013 |
| 10 | 0.72M | +0.0027 | +0.0010 | +0.0019 | +0.0017 (+0.0015) | +0.0043 | −0.0018 | +0.0061 | +0.0018 | +0.00027 |
| 11 | 2.82M | +0.0036 | +0.0010 | +0.0022 | +0.0026 (+0.0024) | +0.0050 | +0.0006 | +0.0044 | +0.0025 | −0.00004 |
| 12 | 0.82M | +0.0047 | +0.0010 | +0.0023 | +0.0037 (+0.0034) | +0.0062 | +0.0005 | +0.0057 | +0.0033 | +0.00012 |
| 13 | 1.12M | +0.0020 | +0.0010 | +0.0020 | +0.0011 (+0.0009) | +0.0038 | +0.0015 | +0.0024 | +0.0021 | −0.00003 |
| 14 | 3.71M | +0.0036 | +0.0011 | +0.0019 | +0.0025 (+0.0024) | +0.0048 | −0.0002 | +0.0050 | +0.0031 | +0.00016 |
| 15 | 1.36M | +0.0045 | +0.0009 | +0.0022 | +0.0036 (+0.0032) | +0.0062 | −0.0044 | +0.0106 | +0.0038 | −0.00012 |

**Across seeds and clusters:** none of the 64 seed × cluster cells is negative for Δ_CE_A, CE_B − CE_A, Δ_CE_B or Δ_CE_C. The minima are +0.0017, +0.0009, +0.0006 and +0.0017.

**Weakest clusters:** 9 and 13 (A−B about +0.0011 to +0.0013, S_A 0.0024). In these, A ≈ C.

## 6. Functional specialisation (prereg §8.4; replacement matrix with exact downstream CE; VAL)

| quantity | value |
|---|---|
| S_A (pooled) | +0.00484, CI [0.00472, 0.00497]; hierarchical [0.00457, 0.00509] |
| S_B | +0.00002, CI [−0.00001, +0.00005] (shuffled specialists are interchangeable) |
| S_A − S_B | +0.00482, CI [0.00470, 0.00495]; hierarchical [0.00454, 0.00508] |
| per seed S_A | 0.00489 / 0.00486 / 0.00512 / 0.00448 (S_A > S_B in every seed) |
| F1 / F2 / F3 / F4 | pass / pass / pass (U_own_A > 0 in 16 of 16 clusters) / pass |
| U_best_alt_A | > 0 in 16 of 16 clusters (+0.0015 to +0.0038); for B it is ≤ 0 in 15 of 16 |
| U_shuffled_replacement_A | +0.0009 to +0.0041; replacing A's specialist k with B's on cluster-k tokens always hurts |

**Matching:** U_oth is re-weighted over 675 strata to the own-cluster token distribution (frequency × base-CE × position × document length × target class). So S_A is not explained by those token-surface properties.

**Matrix consistency:** the forced-own minus assigned-path CE has mean absolute value about 3e-7 and maximum 0.12 (fp16 storage).

## 7. Post-hoc descriptive analysis (not preregistered; no decision depends on it)

Written to `artifacts/oracle-residual-v1/posthoc_descriptive_S3_VAL.json`.

**A's advantage over C is concentrated at the start of documents.** The pattern holds in all 4 seeds.

| position | share of tokens | CE_C − CE_A | CE_B − CE_A | Δ_CE_A |
|---|---|---|---|---|
| 0–31 | 3.2% | +0.015 to +0.017 | +0.016 to +0.018 | +0.017 to +0.020 |
| 32–127 | 9.7% | +0.0023 to +0.0027 | +0.0034 to +0.0039 | +0.0043 to +0.0048 |
| 128–511 | 38.1% | +0.0006 to +0.0008 | +0.0017 to +0.0020 | +0.0026 to +0.0030 |
| 512–1023 | 48.9% | +0.0003 to +0.0005 | +0.0014 to +0.0016 | +0.0023 to +0.0027 |

- About 60% of the pooled A−C gap comes from positions below 128, which are 13% of tokens. This fits A knowing the paper's subject before the causal context reveals it.
- Late in the document, A−C shrinks to +0.0003 to +0.0005, though it stays positive in every seed. A−B stays at about +0.0015.
- **C did not discover the semantic partition.** The mutual information between idx_C and the semantic label is 0.014–0.021 nats, which is 0.5–0.8% of H(sem) = 2.66 nats. The largest share of any semantic cluster routed to one C specialist is 0.15–0.18.

## 8. Interpretation and caveats

**What the result supports.** The oracle premise passes. Label-free semantic clusters do correspond to separable predictive functions that benefit from separate residual parameters:
- the semantic assignment beats a structure- and mass-matched shuffled assignment in every seed and in all 16 clusters;
- the specialists are functionally specific (S_A ≈ 0.0048 against S_B ≈ 0);
- swapping in another specialist always hurts.

**Caveats.**
1. **Oracle and non-causal.** The assignment uses the whole paper, including text after the current token, and is constant within a paper. A's advantage over C is largest exactly where a causal model cannot yet know the subject (§7). A−B (semantic against matched partition) is the cleaner semantic test; A−C mixes semantics with access to information.
2. **The effects are small:** +0.0034 nats overall; +0.0024 for semantic over shuffled.
3. **Limited setting:** a post-hoc residual added to a frozen base at 50M tokens with a single LR. That LR is the grid boundary, so the absolute values may be underestimates for all conditions alike.
4. **Weak token-routed baseline:** C's router is linear on a single hidden state. A stronger causal router could close part of the A−C gap; this experiment did not test that.
5. **Relation to SCRGP-V1:** there the semantic residual was real but zero-sum *within a fixed-capacity* model. Here the capacity is *added*. The two results are compatible.

**No deployability claim** is made. A deployable version would need causal inference of the cluster from the prefix, and the learned C router did not recover it.

## 9. Compute, memory, disk

**GPU:** 2.33 GPU-h of the 20 cap.

| stage | GPU-h |
|---|---|
| stage 0 | 0.12 |
| stage 1 | 0.31 |
| stage 2 | 0.38 |
| stage 3 | 1.52 |

- **Time per job:** training at 50M took about 590 s per seed; each VAL full eval took about 1,066 s.
- **Memory:** peak CUDA allocation 6.92 GiB.
- **Disk:**
  - free space was about 148 GiB at the start and 134 GiB at the end, and never fell below the 100 GiB reserve;
  - `evals/` holds 13 GB (four VAL npz files with full replacement matrices at 2.4 GB each);
  - `runs/` holds 1.2 GB; the base checkpoints were never copied.
- **Protected inventory:** 9,941 files verified unchanged after the run (all_match).
- **Base unchanged:** the base state hash was unchanged in every job.

## 10. Deviations

All are listed in `deviations.jsonl`. None changed a threshold or a decision.

0. The Stage-2 persistence and no-degradation rules were not coded in the analysis. The gap was logged, then the rules were evaluated once exactly as written (`stage2_extra_rules.json`); both pass by wide margins.
1. The VAL assignment and evaluation path, the hierarchical bootstrap with V1–V8, and the Stage-3/finalize orchestration were written after the prereg froze, but before VAL was opened. They implement §8.3/§10 as written. The VAL assignment was validated by reproducing the frozen CALIB-SEARCH table (agreement 1.0).
2. The pilot LR, 3e-4, sits on the grid boundary. The grid was not extended (prereg).
3. One engineering fix: the RNG state is moved to CPU on resume. It was logged before fixing; the resume test is bitwise identical.
4. The §7 post-hoc analysis is descriptive only.

## 11. Paths

| what | path |
|---|---|
| prereg | `docs/ORACLE_RESIDUAL_V1_PREREG.md` |
| report | `ORACLE_RESIDUAL_V1_REPORT.md` |
| diagnostic notes | `ORACLE_RESIDUAL_V1_DIAGNOSTIC_NOTES.md` |
| final analysis | `artifacts/oracle-residual-v1/final_analysis.json` |
| stage analyses | `artifacts/oracle-residual-v1/stage_S1.json`, `stage_S2.json`, `stage_S3.json`, `stage2_extra_rules.json`, `pilot_selection.json` |
| post hoc | `artifacts/oracle-residual-v1/posthoc_descriptive_S3_VAL.json` |
| logs | `artifacts/oracle-residual-v1/run_manifest.tsv`, `compute_ledger.tsv`, `disk_ledger.tsv`, `decision_log.jsonl`, `observations.jsonl`, `deviations.jsonl`, `RESUME.md`, `STATE.json` |
| checkpoints | `artifacts/oracle-residual-v1/runs/s{0..3}/ckpt_{20000000,50000000}.pt` |
| evals | `artifacts/oracle-residual-v1/evals/` (S1/S2 on EVAL and SEL; S3_s{0..3}_VAL) |
| code | `orv1_*.py`, `src/csr/orv1_{paths,lib,log}.py` |
