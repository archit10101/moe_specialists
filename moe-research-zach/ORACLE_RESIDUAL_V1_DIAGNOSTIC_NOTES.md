# ORACLE-RESIDUAL-V1 — Diagnostic Notes

These are working notes behind `ORACLE_RESIDUAL_V1_REPORT.md`. Anything marked *post hoc* is descriptive and fed no decision.

## Engineering log (each issue was logged before acting)

1. **Inventory write failed** because the artifacts directory did not exist yet. Fixed by rerunning after the tables were created. No effect.
2. **CPU continuation test mismatch.** `test_steps` was included in the cfg dict that resume compares against. It was removed from cfg; the tests then passed 9/9.
3. **GPU resume crash:** `set_rng_state` requires a CPU ByteTensor. Fixed with `.cpu()`; the rerun's two-leg resume was bitwise identical to one uninterrupted run (max diff 0.0).
4. **Stage-2 rules missing.** The prereg §8.2 persistence and no-degradation rules were absent from `orv1_analyze.py`. This was found after S2 analysis and before promotion was acted on. The rules were evaluated as written (`stage2_extra_rules.json`) and recorded as deviation 0.
5. **Shuffle redesign before the prereg froze.** The old GC-EDL SHUF table did not preserve token mass (cluster 14: 15% vs 5%). It was replaced *before* the prereg with a within-quantile-bin permutation (20 paper-token-mass bins). On VAL the token-mass distributions of SEM and SHUF labels match closely (cluster 14 ≈ 0.149 in both).

No external interruptions happened. No foreign GPU processes were seen. GPU occupancy monitors were clean throughout.

## Sanity checks that held

- **Base unchanged:** the base state hash matched in every eval and every seed, and in each `done_*.json`.
- **Zero-initialised W2:** at step 0 every condition reproduces CE_D exactly (G-ZERO).
- **Matrix consistency:** forced-own replacement CE vs assigned-path CE has mean absolute difference about 3e-7 and maximum 0.12. The maximum is a few fp16-storage outliers.
- **C router was never collapsed:** load max 0.078–0.082, effective number of experts 15.8–15.9 in every seed.
- **Bank weight norms:** W2 norms at the end are similar across banks (A about 3.20–3.22, B about 2.89–2.91, C about 3.13–3.19). B learned a comparable amount of weight while gaining far less CE, which fits B learning a generic function replicated 16 times (S_B ≈ 0).
- **The VAL assignment pipeline is faithful.** Rerunning the frozen network on CALIB-SEARCH reproduced the stored table (hard agreement 1.0, max |Δq| 3e-8).

## Trajectory

| quantity | S1 (20M, EVAL) | S2 (50M, EVAL) | S3 (50M, VAL, 4 seeds) |
|---|---|---|---|
| Δ_CE_A | 0.00205 | 0.00350 | 0.00337 |
| CE_B − CE_A | 0.00143 | 0.00243 | 0.00240 |
| CE_C − CE_A | 0.00068 | 0.00134 | 0.00127 |
| S_A | 0.0021 | 0.0048 | 0.0048 |

- Every gap grew from 20M to 50M.
- The SEL learning-curve gain from 20M to 50M was 0.00088, below the 0.001 bar, so the budget rule kept 50M and nothing ran at 245M.
- EVAL (CALIB-SEARCH half B) and VAL agree closely, so there is no sign of over-fitting to the CALIB-SEARCH partitions.

## Pilot

LR {3e-4, 1e-3, 3e-3} on SEL, seed 0, 5M tokens:
- every condition was within 0.001 nats of the base;
- higher LR was worse, and 3e-3 was worse than the base for all conditions;
- 3e-4 won at the boundary of the grid.

Since all conditions share the LR, the comparisons are fair. The absolute gains might be larger at a lower LR; this was not tested, because extending the grid is forbidden by the prereg.

## Post-hoc: where the A−C gap lives

See `artifacts/oracle-residual-v1/posthoc_descriptive_S3_VAL.json`.

**By position:**
- At positions 0–31, C − A is about 0.016 nats.
- At positions 512–1023, it is about 0.0004.
- About 60% of the pooled A−C gap comes from the first 128 positions.

So most of A's advantage over the causal token router is *the oracle knowing the subject before the context reveals it*. Within-paper cluster identity is cheap to know late and costly to infer early.

**What remains late in the document:**
- A−B stays at about +0.0015 late: semantic grouping beats matched shuffled grouping even when much context is available.
- A−C stays positive late (+0.0003 to +0.0005 in every seed). The semantic partition carries something C's linear router does not reproduce from a single hidden state.

**C's routing is close to orthogonal to the semantic clusters:** MI 0.014–0.021 nats, NMI/H_sem < 1%. C finds a different (token-level) decomposition that is worth about 62% of A's total gain at this budget.

## Relation to earlier experiments

| experiment | finding |
|---|---|
| SEMEDL / SUE / SLL / SLC | semantic *routing* inside MoE training mostly gave generic, non-semantic grouping gains; SHUF ≈ 80% of EDL |
| SCRGP-V1 | the semantic residual is real but zero-sum at fixed capacity |
| this experiment | with *added* capacity and an *oracle* paper-level assignment, the semantic partition is specifically useful (B ≈ 29% of A); specialists are functionally specific (S_A ≫ S_B ≈ 0) |

These differ in setting (added residual capacity vs routing within a fixed MoE; oracle vs learned or causal assignment), so they are not in conflict. The open question is whether a *causal* assignment can recover enough of the semantic partition early in a document to capture A−B without the oracle. This experiment does not answer it, and it is not claimed.

## Things worth knowing for any follow-up

- **Disk:** VAL full evals with dA/dB matrices take 2.4 GB per seed. The `--full` flag is needed only for the functional analysis.
- **Speed:** joint training of three banks runs at about 88k tokens/s; one VAL full eval takes about 18 min.
- **Weakest clusters:** 9 and 13 (A−B about 0.0011–0.0013, S_A 0.0024). In these, C ≈ A.
- **Cluster 15** has the largest S_A (0.0106), driven by strongly negative U_oth: its specialist hurts other clusters.
