# semantic-lift-245m-confirm-v1 — final report

**Status:** COMPLETE. All 8 runs ran to exactly 245,032,457 tokens and all 40 frozen evaluations were done. The analysis is pre-registered and was run once. The integrity check passed.

**Pre-registration:** `docs/SEMANTIC_LIFT_245M_CONFIRM_V1_PREREG.md`, sha256 `d949d4f515646daeea84bf9d5818f548c6c177e95ccc354d99d6075fd24294f9`, written and hashed before any new training.

**Artifacts:** `artifacts/semantic-lift-245m-confirm-v1/`
- `results.json` and `tables.md`
- `evals/` (40 × json/npz)
- `train_summaries/`
- ledgers and logs

## 1. Answer

**Primary outcome: C — "apparent functionality mostly from routing exposure".** The trajectory outcome is G, which here applies to the ordinary metric only. The method is **not successful** under the pre-registered rule.

**Routing (what Semantic Lift does).** Semantic Lift (E) reliably makes routing more subject-concentrated than the identical mechanism driven by shuffled groups (F).
- Domain NMI: +0.026, 4/4 seeds.
- Domain top-4 overlap: −0.079, 4/4 seeds.
- Routing-specialised cells: 235 vs 177.

**Ordinary per-expert replacement interaction (also from E).**
- E − F = +1.53e-4, 4/4 seeds, hierarchical 95% CI [+1.17e-4, +1.88e-4]. This is about +25% relative to F's ~6.1e-4.
- The difference-in-differences (E−F) − (C6−SHUF) is also positive: +1.50e-4, 4/4 seeds, CI [+0.87e-4, +2.17e-4].

**Exposure-conditioned interaction (the check that fails).** This compares tokens that actually reached the expert, matched on CE decile, position, router rank, gate decile and margin decile.
- E − F = −4.5e-5, only 1/4 seeds positive, CI [−2.5e-4, +1.75e-4].
- The pre-registered requirement that the conditional analysis point the same way (S4) **fails**.
- So the larger ordinary interaction is what you would expect if E's experts simply *receive more of their subject's tokens, at higher gate values*. It is not evidence that an expert is more useful *per token it actually processes*.

**Accuracy.** Accuracy did not improve.
- E is worse than F by +0.0037 nats VAL CE (3/4 seeds worse; CI [−0.003, +0.011]).
- E is worse than C6 by +0.0034 (3/4; CI [−0.0003, +0.0071]).
- E is worse than EDL by +0.0066 (3/4; CI [+0.0001, +0.0147]).
- The pre-registered flag "CE worsened" (> 0.005 vs C6 or EDL) is **true**, triggered via E − EDL.
- E is better than TOKEN-4 by 0.0032, but this is not reliable: 3/4 seeds, CI [−0.011, +0.007].

**Did experts become genuinely more useful for their subjects?** Not demonstrably.
- They became more *used* by their subjects, and the ordinary interaction measure rises accordingly.
- Once exposure is held fixed, no E-vs-F advantage remains. The point estimate is slightly negative and the CI is wide.

## 2. Design (as executed)

**New conditions** (seeds 0–3, 3,880 steps, 245,032,457 tokens each, paired init/rows/optimizer/software; scheduled alternately as E0 F0 F1 E1 E2 F2 F3 E3):

| condition | description |
|---|---|
| E | frozen C6 semantic groups + slow lift estimator (λ 0.5, multiplier range [0.5, 2.0], 5% collection / 5% ramp / 90% full; parameter-gradient-only reweighting, weighted mean 1) |
| F | identical, with the SHUF C6 group table |

**Reused baselines** (verified, not retrained; `reuse_manifest.json`, all_ok):
- **C6:** slk P1 seeds 0–3.
- **SHUF:** slk P1 seeds 0–3.
- **EDL:** fle final seeds 0–1, plus slk P1_EDL seeds 2–3.
- **T4:** fle final seeds 0–2, plus slk P1_T4_s3.

The checks covered status, tokens, plan, parameters, configs, group tables, code identity and checkpoint SHAs.

**Measurement identity.** C6_s0 full-VAL CE recomputed by the new evaluator was 3.049275172, identical to the stored slk value (|Δ| = 0).

**Prior evidence** (seeds 100/101 at 50M, from sll v1) was not mixed in.

## 3. Primary criteria (prereg §7)

| criterion | value | pass |
|---|---|---|
| S1 D_func = I_ord(E) − I_ord(F) > 0 | +1.53e-4 | ✔ |
| S2 ≥ 3/4 seeds positive | 4/4 (+1.53, +1.59, +1.34, +1.64 e-4) | ✔ |
| S3 hierarchical CI (seeds × documents) excludes 0 | [+1.17e-4, +1.88e-4] | ✔ |
| S4 exposure-conditioned in same direction | −4.5e-5 (−2.34, −0.45, +1.66, −0.69 e-4); CI [−2.48e-4, +1.75e-4] | ✘ |
| S5 no pathology, invariants, finite | no pathology; invariants ✔; finite ✔ | ✔ |

Outcome matching gives **C** (ORD ∧ ¬COND). E and D are excluded by ¬COND. B and A require ¬ORD, so they do not apply.

**Trajectory: G.** ORD holds at 245M, and the ordinary D245 − D50 = +8.2e-5, CI [+4.8e-5, +1.15e-4]. The ordinary E−F gap roughly doubled from 50M (+7.1e-5, 4/4) to 245M.

However, the conditional E−F gap moved the other way, from +1.5e-4 (3/4) at 50M to −0.45e-4 (1/4) at 245M; D245 − D50 (cond) = −1.9e-4, CI [−4.2e-4, +0.4e-5]. The consolidation is in exposure, not in per-token usefulness.

**Outcome F (transient) is not met:** the ordinary gap at 245M did not disappear.

## 4. Functional interaction — all conditions, per seed (245M)

Per-cell interaction is defined as own-subject ΔCE minus other-subject ΔCE when an expert is replaced.
- 352 (layer, expert) cells.
- Ordinary strata: CE decile × position bin.
- Conditional strata: those plus router rank × gate decile × margin decile, restricted to tokens on which the expert was in the original top-4.

| cond | I_ord s0..s3 | I_cond s0..s3 | adequate cond cells | ord-specialised cells |
|---|---|---|---|---|
| E | 7.92, 7.65, 7.43, 7.65 e-4 | −1.14, −1.33, −1.73, −1.25 e-4 | 351, 347, 345, 351 | 163, 169, 162, 154 |
| F | 6.39, 6.06, 6.08, 6.01 e-4 | +1.19, −0.88, −3.39, −0.56 e-4 | 349, 348, 351, 352 | 131, 124, 107, 113 |
| C6 | 6.38, 6.11, 6.44, 6.54 e-4 | −1.61, −2.70, −2.48, −2.29 e-4 | 347, 347, 352, 347 | 140, 128, 128, 141 |
| SHUF | 6.54, 6.78, 6.34, 5.73 e-4 | +1.76, +0.42, −0.79, −1.48 e-4 | 349, 352, 350, 352 | 126, 129, 120, 125 |
| EDL | 6.43, 6.42, 6.91, 6.50 e-4 | −6.81, −3.68, −2.29, −1.46 e-4 | 344, 350, 350, 348 | 141, 154, 149, 151 |
| T4 | 4.03, 3.75, 3.76, 3.57 e-4 | +0.67, −0.39, −0.28, +1.86 e-4 | 348, 347, 350, 344 | 57, 49, 62, 52 |

Conditional support was ample. Median per cell (C6_s0, typical): 18.5k matched own tokens, 98.5% matched, other-side ESS ≈ 30k, 101 own documents. About 345–352 of 352 cells were adequate in every model. The conditional estimate is therefore not starved; it is noisier between seeds.

**Pairwise contrasts at 245M** (mean over seeds; n>0 of 4; hierarchical 95% CI):

| contrast | ordinary | conditional |
|---|---|---|
| **E − F** | **+1.53e-4 (4/4) [+1.17, +1.88]e-4** | **−0.45e-4 (1/4) [−2.48, +1.75]e-4** |
| E − C6 | +1.29e-4 (4/4) [+0.89, +1.70]e-4 | +0.91e-4 (4/4) [−0.54, +2.43]e-4 |
| E − EDL | +1.10e-4 (4/4) [+0.61, +1.56]e-4 | +2.20e-4 (4/4) [−0.31, +5.25]e-4 |
| E − T4 | +3.88e-4 (4/4) [+3.41, +4.35]e-4 | −1.83e-4 (0/4) [−3.65, +0.16]e-4 |
| C6 − SHUF | +0.02e-4 (2/4) [−0.53, +0.59]e-4 | −2.25e-4 (0/4) [−3.92, −0.66]e-4 |
| F − SHUF | −0.21e-4 (1/4) [−0.63, +0.20]e-4 | −0.89e-4 (1/4) [−2.56, +0.83]e-4 |
| F − C6 | −0.23e-4 (1/4) [−0.53, +0.07]e-4 | +1.36e-4 (3/4) [−0.57, +3.20]e-4 |
| DiD (E−F) − (C6−SHUF) | +1.50e-4 (4/4) [+0.87, +2.17]e-4 | +1.79e-4 (4/4) [−0.42, +4.05]e-4 |
| EDL − T4 | +2.79e-4 (4/4) [+2.36, +3.21]e-4 | −4.03e-4 (0/4) [−6.93, −1.73]e-4 |
| C6 − T4 | +2.59e-4 (4/4) [+2.21, +3.00]e-4 | −2.74e-4 (0/4) [−4.36, −1.17]e-4 |

**Secondary comparisons (descriptive, not the primary claim).**
- **E vs C6 and E vs EDL (conditional).** Both have all 4 seeds positive, but their CIs include 0. The positive E−C6 and E−EDL conditional signs are partly shared by F (F − C6 cond +1.36e-4, 3/4). This suggests that much of it comes from the lift *mechanism* (gradient reweighting by group), not from semantic grouping.
- **Ordinary vs conditional disagreement for every grouped method.** EDL − T4 and C6 − T4 are strongly positive on the ordinary metric and significantly *negative* on the conditional one. That is, every grouped/EDL method gains ordinary interaction over TOKEN-4 by concentrating routing, not by per-token usefulness.
- **Implication.** The ordinary interaction metric, which the earlier experiments relied on, is substantially confounded by routing exposure across this whole family of methods. This is the most important by-product of the experiment.

## 5. Accuracy (full VAL, 25,254 rows)

| model | s0 | s1 | s2 | s3 | mean | SD |
|---|---|---|---|---|---|---|
| E | 3.0477 | 3.0486 | 3.0456 | 3.0585 | 3.0501 | 0.0058 |
| F | 3.0446 | 3.0460 | 3.0508 | 3.0440 | 3.0464 | 0.0031 |
| C6 | 3.0493 | 3.0431 | 3.0445 | 3.0499 | 3.0467 | 0.0034 |
| SHUF | 3.0379 | 3.0447 | 3.0540 | 3.0476 | 3.0461 | 0.0067 |
| EDL | 3.0454 | 3.0415 | 3.0474 | 3.0397 | 3.0435 | 0.0035 |
| T4 | 3.0502 | 3.0576 | 3.0585 | 3.0468 | 3.0533 | 0.0057 |

**Paired differences** (per seed; hierarchical CI over seeds × documents):

| contrast | per seed | mean | hier CI | P(rep > 0) |
|---|---|---|---|---|
| E − F | +0.0030, +0.0026, −0.0053, +0.0145 | +0.0037 | [−0.0032, +0.0115] | 0.86 |
| E − C6 | −0.0016, +0.0055, +0.0010, +0.0087 | +0.0034 | [−0.0003, +0.0071] | 0.96 |
| E − EDL | +0.0023, +0.0071, −0.0019, +0.0188 | +0.0066 | [+0.0001, +0.0147] | 0.98 |
| E − T4 | −0.0026, −0.0090, −0.0130, +0.0118 | −0.0032 | [−0.0111, +0.0066] | 0.25 |
| EDL − T4 | −0.0049, −0.0161, −0.0111, −0.0071 | −0.0098 | [−0.0139, −0.0059] | — |

- **By domain.** E − F is positive (worse) in every one of the 5 domains (+0.0030 to +0.0045), and so is E − EDL (+0.0057 to +0.0074). Semantic Lift's stronger routing concentration did not help even the domains it concentrates on.
- **E seed 3.** This is the worst E run (3.0585). It drives much of the spread; without it E − F would be +0.0001 (2/3 positive). This seed was not and may not be excluded.
- **Most accurate model.** Explicit-label EDL remains the most accurate model here, and the only one reliably better than TOKEN-4.

**50M CE** (E/F/C6/SHUF):
- E − F = +0.0065 (2/4), CI [−0.008, +0.021].
- E − C6 = +0.0029 (3/4 positive).

The noisy accuracy disadvantage of E is present at both milestones.

## 6. Routing specialisation (routing metrics only; not functional evidence)

**Mean over seeds** (VAL audit rows, 200k tokens per domain):

| model | domain NMI | domain top-4 overlap | routing-spec cells (lift ≥ 1.5) | e5-cluster NMI | cross-seed profile consistency | full-VAL load min/max × uniform | routing entropy |
|---|---|---|---|---|---|---|---|
| E 245M | 0.0704 | 0.659 | 235 | 0.0957 | 0.910 | 0.84 / 1.20 | 0.631 |
| F 245M | 0.0443 | 0.738 | 177 | 0.0704 | 0.900 | 0.82 / 1.23 | 0.582 |
| C6 245M | 0.0547 | 0.705 | 207 | 0.0774 | 0.909 | 0.85 / 1.26 | 0.599 |
| SHUF 245M | 0.0452 | 0.737 | 177 | 0.0710 | 0.897 | 0.77 / 1.38 | 0.583 |
| EDL 245M | 0.0574 | 0.700 | 210 | 0.0780 | 0.908 | 0.83 / 1.35 | 0.593 |
| T4 245M | 0.0126 | 0.859 | 72 | 0.0219 | 0.882 | 0.83 / 1.30 | 0.657 |
| E 50M | 0.0544 | 0.695 | 230 | 0.0799 | 0.916 | 0.67 / 1.38 | 0.754 |
| F 50M | 0.0393 | 0.745 | 182 | 0.0650 | 0.904 | 0.68 / 1.53 | 0.721 |

**Contrasts.**
- **E − F NMI:** +0.027, +0.026, +0.021, +0.030 (4/4). This exceeds explicit-label EDL.
- **E − F overlap:** −0.081, −0.079, −0.065, −0.091 (4/4).
- **E − C6 NMI:** +0.0157 (4/4).
- **Growth from 50M to 245M:** E − F NMI rises from +0.015 (4/4) to +0.026.

**Cross-seed consistency.** Domain-to-expert lift profiles are highly consistent across seeds after permutation matching, but similarly so for every grouped condition (0.90–0.91), so this does not distinguish E.

**Pathology.** None in any model; load stays within 0.77–1.38 × uniform. No unused or under-used experts.

## 7. Training (new runs)

| run | tok/s | final train lm | mean multiplier deviation (full-strength phase) | clip fraction | lift stability | invariants / finite | train load after ramp (× uniform) |
|---|---|---|---|---|---|---|---|
| E_s0 | 105,097 | 2.9745 | 0.133 | 0.026 | 0.9988 | ✔ / ✔ | 0.27 / 2.01 |
| E_s1 | 106,210 | 2.9768 | 0.136 | 0.027 | 0.9988 | ✔ / ✔ | 0.23 / 1.92 |
| E_s2 | 104,833 | 2.9739 | 0.137 | 0.026 | 0.9987 | ✔ / ✔ | 0.28 / 1.86 |
| E_s3 | 105,494 | 2.9855 | 0.137 | 0.026 | 0.9988 | ✔ / ✔ | 0.25 / 2.00 |
| F_s0 | 105,778 | 2.9718 | 0.140 | 0.018 | 0.563 | ✔ / ✔ | 0.31 / 1.90 |
| F_s1 | 105,270 | 2.9694 | 0.138 | 0.016 | 0.590 | ✔ / ✔ | 0.33 / 1.82 |
| F_s2 | 106,312 | 2.9799 | 0.140 | 0.018 | 0.597 | ✔ / ✔ | 0.28 / 1.91 |
| F_s3 | 105,127 | 2.9676 | 0.140 | 0.019 | 0.583 | ✔ / ✔ | 0.34 / 1.81 |

- **Multipliers.** All multipliers stayed in [0.5, 2.0]; the weighted mean was 1 to within 7e-16; no non-finite losses; no early stop.
- **Train lm.** E's final train lm is higher than F's in 3/4 pairs (mean +0.0055), consistent with its slightly higher VAL CE.
- **Lift stability.** E's lift estimates are stable (r ≈ 0.999). F's are much less stable (≈ 0.58), because shuffled groups carry little consistent signal. The multiplier *magnitude*, however, is matched (|m−1| ≈ 0.137 vs 0.140); see limitations.

## 8. Compute, disk, integrity

**Compute: 19.53 GPU-h** (cap 30).
- Checks: 0.23.
- Training: 5.69 (8 runs × ~43 min).
- Evaluation: 13.60 (40 evaluations × ~20.4 min).

**Throughput:** ~105k tok/s, the same for E and F.

**Disk:** ≥ 150 GiB free throughout (reserve 100). One rolling slot per run, deleted after verification. Permanent s50 and p100 checkpoints are kept. Deletions are in `deleted_artifacts.tsv` and cover only this experiment's scratch. /dev/shm was not used.

**Integrity:**
- **Protected files:** the 8,906-file protected inventory verified unchanged (`protected_inventory_before.verify_20261119T221102Z.json`, all_match).
- **Test split:** the locked test split was never read.
- **Domain labels:** real-domain labels were used only as evaluation annotations.
- **Seeds and settings:** no seeds were added, dropped or rerun, and no tokens were added.
- **Analysis:** the analysis was run once. A rerun only attached the integrity record; every other result field was identical.

**Deviations (`deviations.jsonl`; none can affect the conclusion):**
1. **Evaluator test T-ORD:** bootstrap-mean replicates are compared at 1e-12 relative tolerance instead of bitwise, because of floating summation order. Per-cell values, CIs and means are still compared exactly.
2. **GPU equivalence check (prereg 4.3).**
   - **What failed:** the step-50 tolerance of 0.003 was too strict for cross-session/cross-graph nondeterminism. A plain C6 config today vs the 2026-11-15 C6 run differed by 0.021; eager GPU training is nondeterministic from step 0.
   - **How it was replaced:** a powered eager diagnostic with a rule fixed beforehand. E vs C6 divergence over the collection phase was median 0.0068 / 0.0056, vs C6-C6 noise 0.0050 (threshold 0.018); signed bias +0.0043 (limit 0.01). Verdict: INERT.
   - The model, metrics and analysis were not changed.
3. **Orchestration fault (not a scientific deviation).** The pipeline crashed after E_s0 completed: the `train_summaries/` directory was missing. It was fixed (mkdir; skip a run only if its summary exists), logged and relaunched; completed work was skipped. No training was repeated.

## 9. Limitations and unresolved ambiguity

1. **Power of the conditional test.** The conditional E−F CI (±2.1e-4) is wider than the ordinary effect (1.5e-4), so S4 cannot rule out a real per-token advantage of the same size as the ordinary one.
   - The point estimate is negative, only 1/4 seeds are positive, and the 50M→245M conditional trend is downward. So the honest reading is "no evidence of exposure-independent functional specialisation", not "proof of none".
   - The conditional between-seed spread is large for F and SHUF (sign changes across seeds); for E it is tight (−1.1 to −1.7e-4).
2. **Conditioning may over-control.** Matching on gate and margin deciles conditions on the router's own confidence. If a genuinely better expert attracts higher gates *because* it is better, part of a real effect is absorbed. This was the pre-registered definition and was not altered.
   - It is relevant that the conditional measure is *negative* for C6 and EDL and significantly below TOKEN-4. It may partly capture "concentrated experts are relatively worse on the marginal own-subject token they receive".
3. **F is a magnitude-matched but less stable control.** Shuffled-group lift is noisy (stability ≈ 0.58), so F's multipliers fluctuate rather than settle.
   - E−F therefore mixes "semantic vs non-semantic groups" with "consistent vs fluctuating reweighting".
   - The DiD against C6−SHUF (ordinary +1.50e-4, 4/4) gives the same ordinary answer.
4. **Cross-session GPU nondeterminism.** This affects comparisons with the reused baselines (C6, SHUF, EDL, T4 trained 2026-11-15/earlier), which differ by ~0.02 at step 50 even with identical logic.
   - It is noise, not bias (deviation 2), and the primary E vs F contrast is unaffected: both were trained in the same session window.
   - Secondary E-vs-baseline CE contrasts carry this extra variance.
5. **Accuracy statistics.** CE differences are small (≤ 0.007 nats) relative to the seed SDs (0.003–0.009). Only E − EDL's CI excludes 0, and only barely. Four seeds are enough to detect the routing and ordinary effects, not small CE effects.
6. **Domains.** Domains are the 5 evaluation-annotation domains of this corpus. Semantic groups (C6) are e5-cluster-based, frozen from earlier work.

## 10. Bottom line

Semantic Lift (λ = 0.5, 245M tokens, 4 paired seeds) does what it mechanically targets:
- experts are routed far more subject-specifically than under the shuffled-group control, and this grows with training;
- the existing ordinary replacement-interaction metric rises accordingly (4/4 seeds, CI excluding 0).

But once tokens are matched on routing exposure (rank, gate, margin, CE, position), E's experts are **no more useful per token** for their subjects than F's. VAL CE is **slightly worse** than the shuffled control, C6 and EDL.

The pre-registered classification is **outcome C**: the apparent functional specialisation is mostly routing exposure. **Semantic Lift is not confirmed** as a method for real subject-specific functional specialisation, and it does not improve accuracy.

The broader finding: the ordinary interaction metric used across the slk/sue/scu/sll line rewards routing concentration in general. Future specialisation claims in this codebase should use the exposure-conditioned version.
