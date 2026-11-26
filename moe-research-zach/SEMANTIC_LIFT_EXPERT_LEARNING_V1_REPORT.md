# Semantic Lift-Weighted Expert Learning v1 — Report

Experiment `csr-semantic-lift-expert-learning-v1` (prefix `sll_`), 2026-11-18.
Preregistration: `docs/SEMANTIC_LIFT_EXPERT_LEARNING_V1_PREREG.md`, sha256 `aebf65f42a88c0050bf4cf1cb53c0e89175bd1a7dca4bbf2a06f181876cbe87c`.
It was hashed and frozen before any candidate result existed.
Artifacts are in `artifacts/semantic-lift-expert-learning-v1/`; the derived summary is `results.json`.

## Verdict

**Rejected by the bounded protocol at the 50M gate. Diagnosis C, with B as a secondary match.**

The lift-weighted expert gradient makes experts functionally more useful for their own group, but most of the gain is generic. The same operator, driven by shuffled cluster labels, gives the same increase over its own baseline. On held-out rows, semantic E did not beat matched shuffled F:

| | s100 | s101 | pooled |
|---|---|---|---|
| Δ(E−F) | +7.0e-5 | −1.5e-5 | +2.8e-5, CI [−3.6e-5, +1.0e-4] |
| Δ(C6−SHUF), the backbone's own semantic-vs-shuffled gap | | | +3.8e-5 |

- G-F1 (positive on both seeds) and G-F2 (E−F > C6−SHUF) failed.
- The CE (G-CE) and health (G-H) criteria passed.

No engineering generation was justified, because none of the allowed single-factor changes addresses a generic, non-semantic gain. As a result, the 245M finals were not run. CALIB-SELECTION half 0 (the generation and final partitions) and VAL stayed unopened for every decision.

**The method is not claimed to work.**

A post-terminal, descriptive pass on VAL with real domain labels shows a suggestive real-domain signal (section 6). It is unconfirmatory and cannot reverse the verdict, but it identifies what a fresh, preregistered follow-up would test.

## 1. What was tested

**Backbone.** Semantic C6, frozen from `SEMANTIC_EDL_LONG_K_PHASE2A_REPORT.md`:
- LFR 103.6M model: 11 MoE layers, 32 experts, top-4 dropless, shared expert.
- Paper-latent semantic clusters, table `5d23129b…`, β = 0.005.
- The SHUF control uses the matched shuffled table `d374a4ec…`.
- No explicit domain labels were used in training, selection, clustering or engineering.

**Operator.** The operator is in `src/csr/sll_lift.py`.

- **Lift.** For each MoE layer l, cluster k and expert e:
  - P[k,l,e] = E[q_k·p_e] / E[q_k] and P[l,e] = Σ_k π_k P[k,l,e];
  - s = Σ_k q_k (log P[k,l,e] − log P[l,e]).
- **Estimator.** Segment-level, slow EMA (ρ = 0.98), bias-corrected, shrunk toward the marginal (n0 = 8). It uses previous-step statistics only, is updated after the optimizer step, and everything is stop-gradient.
- **Multiplier.**
  - r = exp(λ·tanh(s / scale_l)), where scale_l = max(0.02, MAD of the selected-token lift in layer l).
  - m is the exact projection of r onto [0.5, 2], with gate-weighted mean exactly 1 (closed-form piecewise-linear root, float64).
- **Where it applies.** A custom autograd function (`ScaledExpertMLP`) scales only the selected expert's parameter gradient (dW) by m. The forward pass, router, shared expert, attention and the gradient into each expert's input are unchanged.
- **Schedule.**
  - 5% collection (m ≡ 1), steps [0, 194);
  - 5% linear ramp, steps [194, 388);
  - 90% full strength from step 388.
- **Constraints.** No extra parameters, tokens, steps or schedules. Checkpoints have exactly the C6 parameter keys.

**Conditions.**

| | Condition | Status |
|---|---|---|
| A | TOKEN-4 | reused |
| B | explicit-label EDL | reused |
| C | Semantic C6 | reused |
| D | SHUF C6 | reused |
| E | C6 + lift | new |
| F | SHUF C6 + identical lift | new; same rows, same operator, shuffled table |

## 2. Correctness

**CPU unit tests: 29/29 pass** (`cpu_tests_20261118T064656Z.json`). They cover:
- projection exactness and bounds, and agreement with a bisection reference;
- custom-backward equivalence to autograd;
- m ≡ 1 being a no-op;
- the forward pass being unchanged;
- only expert dW changing;
- the estimator's bias correction, shrinkage, ESS and stability;
- previous-step-only use of the statistics.

An earlier run (`…064639Z.json`) recorded 2 test-spec failures. They required exact equality m == 1 although the weights were float32-normalized; the fix was a 1e-12 tolerance with float64 weights. The file is retained.

**CPU trainer tests tt1–tt5: all pass.**
- tt1: with λ = 0, the per-step loss and gradient norm are bit-identical to plain C6 for 20 steps.
- tt2: the operator is active after collection, and the invariants hold.
- tt3: resume is exact, including the lift table.
- tt4: checkpoint keys equal C6's, and inference uses the native experts.
- tt5: SEM and SHUF see the same rows.

**GPU checks.**
- **G2, the CUDA operator check: pass.** The forward pass is bitwise identical. Non-expert and expert-input gradients are within native run-to-run noise, and expert dW differs as intended. In a microbenchmark the operator path costs 1.22× the native path.
- **G1 and G3 initially failed against my registered fixed 5e-4 tolerance.** That tolerance was below GPU nondeterminism: two identical runs differ by up to 0.012 nats within 30 steps.
- **Deviation 1:** the tolerance was replaced with the project's established rule, max(0.003, 3 × run-to-run noise). Under it, G1 and G3 pass (`gpu_checks_v2.json`):
  - G1: step 0 is bitwise identical, and the step-50 difference of 0.0067 is below the run-to-run noise of 0.015.
  - G3: resumed-vs-straight differs by 0.041, against straight-vs-straight noise of 0.050.
- **G5 (load over 60 steps) was not judgeable.** Zero-load micro-batches occur equally on the native path early in training, so it was enforced by the registered evaluation-pathology rule instead. No pathology was observed at any milestone.

**Reuse manifest:** 24/24 reused baseline milestones were verified (`reuse_manifest.json`).

## 3. Preregistered search (20M, seed 100, dev = CALIB-SEARCH half B)

| λ | Δ interaction (E−F) | ΔCE vs C6 | Eligible |
|---|---|---|---|
| 0.25 | +5.8e-5 | −0.0012 | yes |
| **0.5** | **+5.1e-5** | **−0.0046** | **yes → promoted** (tie rule: within 5e-5 of the best, take the larger λ) |
| 1.0 | +7.2e-5 | +0.0200 | no (CE cost > 0.010) |

- The reference gap Δ(C6−SHUF) was +1.1e-5.
- All CIs include 0.
- Functionally specialised cells: E 68–72, F 49–53, C6 56, SHUF 55.
- The 20M screen sees only the ramp (λ_eff ≤ 0.64λ).

## 4. Promotion at 50M (λ = 0.5, seeds 100 and 101)

**Dev (CALIB-SEARCH half B)**

| | s100 | s101 |
|---|---|---|
| Δ(E−F) | +1.26e-4 [−4.8e-5, +2.1e-4] | +6.5e-5 [−2.0e-6, +2.4e-4] |
| Δ(C6−SHUF) | +1.6e-5 | −7.4e-5 |
| interaction E / F / C6 / EDL / T4 (e-4) | 4.15 / 2.89 / 3.29 / 3.55 / 2.94 | 4.19 / 3.54 / 3.12 / 3.48 / – |
| CE E / F / C6 / SHUF / EDL / T4 | 3.7374 / 3.7454 / 3.7436 / 3.7386 / 3.7504 / 3.7609 | 3.7516 / 3.7372 / 3.7346 / 3.7360 / 3.7524 / 3.7470 |
| specialised cells E / F / C6 | 79 / 51 / 62 | 81 / 61 / 62 |

**Held-out gate partition (CALIB-SELECTION half 1, sub-half b; opened once)**

| | s100 | s101 |
|---|---|---|
| Δ(E−F) | +7.0e-5 [−4.2e-5, +1.9e-4] | **−1.5e-5** |
| Δ(C6−SHUF) | +9.4e-5 | −1.8e-5 |
| interaction E / F / C6 / SHUF (e-4) | 4.87 / 4.16 / 4.70 / 3.76 | 4.19 / 4.34 / 3.39 / 3.57 |
| specialised cells E / F / C6 / SHUF | 78 / 62 / 58 / 45 | 49 / 49 / 48 / 50 |
| routing MI E / F / C6 / SHUF | 0.177 / 0.133 / 0.164 / 0.140 | – |
| CE E / F / C6 / SHUF | 3.8086 / 3.8176 / 3.8149 / 3.8088 | 3.8238 / 3.8078 / 3.8048 / 3.8073 |

**Pooled:** Δ(E−F) is +2.76e-5 [−3.6e-5, +1.03e-4], against Δ(C6−SHUF) of +3.81e-5.

**Gate criteria**

| Criterion | Result |
|---|---|
| G-F1, both seeds > 0 | **FAIL** |
| G-F2, pooled E−F > pooled C6−SHUF | **FAIL** |
| G-CE, mean dev E−EDL ≤ 0.005 | pass (−0.0069) |
| G-H, health | pass (no pathology; stability 0.999; activity |m−1| = 0.14; clip 1.6%) |

**Multipliers at 50M** (E vs F):

| | E | F |
|---|---|---|
| weighted mean absolute deviation \|m−1\| | 0.142 / 0.146 | 0.124 / 0.121 |
| clip fraction | 1.6% | 0.5% |
| max deviation of the gate-weighted mean from 1 | ≤ 6e-16 | ≤ 6e-16 |
| lift stability (r with the table 5 snapshots earlier) | 0.999 | 0.50 |
| cluster top-4 Jaccard | 0.14–0.16 | 0.74–0.78 |
| minimum ESS (segments) | 435 | 440 |

F's multipliers were therefore about as active as E's, but noise-driven and unstable. That is the intended matched control.

## 5. Diagnosis (preregistered order D, F, C, B, E, A)

**Primary C: the gain is generic, not semantic.**

| Measure (held-out) | Value |
|---|---|
| Functional gain, E vs C6 | +4.8e-5 |
| Functional gain, F vs SHUF | +5.8e-5 |

These are equal. The operator increases own-group usefulness whatever the grouping, including for noise-driven multipliers.

**Also B: routing specialization strengthened without converting into usefulness beyond the control.**
- E raised routing MI over C6 by +0.019.
- E raised specialised cells on dev and on s100 at the gate.
- On held-out rows the E-vs-C6 functional gain (+8.6e-5 on dev) mostly disappeared: +1.6e-5 for s100 at the gate.

**Ruled out**
- D (implementation): all tests pass.
- F (pathology): no collapse.
- E (instability): stability 0.999.
- A (measurement): this was the cause in V2. Here the dev and held-out estimates agree in sign for C6−SHUF and E−F on s100; the failure is a small, inconsistent effect, not an unmeasurable one.

**Why no engineering generation** (preregistration section 12 requires a measured-failure motivation for one allowed factor):

| Allowed factor | Why it was not motivated |
|---|---|
| Lower λ | CE already passed; lowering λ shrinks the E and F effects alike. |
| EMA ρ | The lift is stable. |
| Ramp / warm-up length | No failure points to it. |
| Later-layer-only | No evidence that early layers harm CE. |
| Narrower bounds | Bounds are not limiting (clip 1.6%). |

None addresses diagnosis C. Running a generation anyway would have been selection on noise. The full rationale and rejected alternatives are in `DECISION_LOG.md`.

## 6. Post-terminal descriptive analysis (VAL, real domain labels; NOT preregistered, NOT used for any decision)

**Setup.** After the decision was frozen (STATE `TERMINAL_REJECTED_AT_50M_GATE`), `sll_post_val.py` evaluated all twelve 50M checkpoints on VAL:
- full VAL CE;
- real-domain and e5 routing NMI;
- per-domain top-4 overlap;
- load;
- the per-expert replacement interaction with the 5 real domains (discovery on CALIB-SEARCH half A, measurement 50k tokens per domain on VAL).

Real domain labels were opened here for the first time in this experiment. Paired differences use identical rows and bootstrap samples.

**Per checkpoint**

| ckpt | VAL CE | domain NMI | domain top-4 overlap | domain interaction (e-4) [CI] | domain-spec cells | e5 interaction (e-4) |
|---|---|---|---|---|---|---|
| T4 s100 / s101 | 3.7349 / 3.7223 | 0.011 / 0.011 | 0.868 / 0.866 | 0.6 [−0.1, 1.3] / 2.1 [1.7, 2.5] | 20 / 25 | 3.9 / 4.5 |
| EDL s100 / s101 | 3.7241 / 3.7265 | 0.050 / 0.048 | 0.710 / 0.720 | 2.8 [2.2, 3.2] / 3.1 [2.5, 3.6] | 84 / 82 | 7.1 / 7.1 |
| C6 s100 / s101 | 3.7184 / 3.7089 | 0.048 / 0.045 | 0.718 / 0.728 | 3.1 [2.6, 3.6] / 2.7 [2.2, 3.1] | 96 / 71 | 6.9 / 6.6 |
| SHUF s100 / s101 | 3.7117 / 3.7109 | 0.041 / 0.038 | 0.740 / 0.753 | 3.0 [2.5, 3.5] / 2.8 [2.3, 3.3] | 64 / 74 | 6.9 / 7.0 |
| **E s100 / s101** | 3.7120 / 3.7269 | **0.052 / 0.052** | **0.700 / 0.705** | **3.9 [3.2, 4.4] / 3.6 [3.0, 4.2]** | **104 / 95** | **9.0 / 7.7** |
| F s100 / s101 | 3.7196 / 3.7121 | 0.040 / 0.038 | 0.746 / 0.753 | 2.9 [2.3, 3.4] / 3.4 [2.9, 3.9] | 65 / 83 | 6.4 / 5.8 |

- Load ranged from 0.55× to 1.50× uniform in every condition, with no collapse.
- The NMI null with document-shuffled labels is about 0.001.

**Paired domain-interaction differences** (bootstrap 95% CIs)

| contrast | s100 | s101 |
|---|---|---|
| E − F | **+9.9e-5 [+5.6, +14.3]e-5** | +1.4e-5 [−3.3, +5.8]e-5 |
| C6 − SHUF | +1.0e-5 [−2.6, +4.7]e-5 | −1.9e-5 [−5.1, +1.8]e-5 |
| (E−F) − (C6−SHUF) | +8.9e-5 | +3.2e-5 |
| E − C6 | +7.3e-5 [+3.6, +11.2]e-5 | +9.3e-5 [+4.9, +13.8]e-5 |
| E − EDL | +1.1e-4 [+0.7, +1.5]e-4 | +5.0e-5 [+0.9, +9.6]e-5 |
| F − SHUF | −1.6e-5 [−5.8, +2.4]e-5 | **+6.0e-5 [+3.0, +8.9]e-5** |
| EDL − T4 | +2.1e-4 | +9.7e-5 |

**Reading (descriptive)**
- Measured against real subjects, E has the highest domain routing NMI of every condition, above explicit-label EDL, without ever seeing labels.
- E also has the lowest cross-domain expert overlap and the largest real-domain functional interaction on both seeds. It beats C6 and EDL with CIs excluding 0 on both seeds.
- The semantic-vs-shuffled interaction contrast is positive on both seeds, but only s100's E−F CI excludes 0.
- On s101, the shuffled operator also produces a significant gain over SHUF (F−SHUF +6.0e-5), the same generic component diagnosed at the gate.
- So the descriptive VAL picture is more favourable to E than the preregistered cluster-level gate. It is still two seeds, post hoc, on a different label system (real domains rather than the semantic clusters the method targets), and one seed's E−F CI includes 0.
- It does not meet the protocol's success definition and does not change the verdict.

**VAL CE**

| | s100 | s101 | mean |
|---|---|---|---|
| E − EDL | −0.012 | +0.0004 | −0.006 |
| E − C6 | −0.006 | +0.018 | +0.006 |
| E − F | −0.008 | +0.015 | +0.004 |

E's CE cost on s101 is visible and persists on VAL. C6 and SHUF are the best CE conditions (3.709–3.718). E matches or beats EDL on CE, but does not reliably match C6.

**Other post-hoc diagnostics** (on the gate and dev rows; descriptive; `OBSERVATIONS.md`):
- **Routing-specialised cells** (discovery lift ≥ 1.5): E 286/292, C6 272/278, F 264/267.
- **Conversion rate** (functionally / routing-specialised): E 0.28 vs C6 0.22 on dev; E 0.27/0.17 vs C6 0.21/0.17 at the gate.
- **Routing lift vs functional interaction:** E shows no stronger correlation across cells.
- **Where E's interaction lives:** in cells whose training-lift argmax cluster equals their functional subject (about 49%): 7–8e-4, vs 1–2e-4 elsewhere.
- **Related groups sharing experts:** in E, related semantic clusters share experts — Spearman of cluster co-assignment vs expert-profile correlation is 0.68 / 0.68, vs 0.08–0.10 for F.

## 7. Answers to the required questions

**1. Did routing specialization become expert usefulness?**
Partly, and not specifically.
- E increased per-expert own-minus-other usefulness relative to C6 on dev at 50M (+8.6e-5), on real domains on VAL (+7–9e-5), and on s100 at the gate.
- On held-out semantic-cluster rows, the increment was matched by the shuffled control's increment over SHUF.
- The conversion rate from routing specialization to functional specialization rose modestly (0.28 vs 0.22 on dev), not decisively.

**2. Did semantic beat shuffled?**
Not by the preregistered test.
- Held-out Δ(E−F): +7.0e-5 on s100 and −1.5e-5 on s101. Signs are inconsistent and the pooled CI includes 0.
- The descriptive VAL real-domain contrast is positive on both seeds, but significant only on s100.

**3. Did it match or beat EDL?**
- On CE, yes: dev E−EDL is −0.0069 (G-CE passed) and VAL is −0.006 on average, with one seed −0.012 and the other +0.0004.
- On real-domain functional specialization, E exceeded EDL descriptively on both seeds.
- E did not reliably match Semantic C6's CE: +0.006 on average on VAL, driven by s101.

**4. Was it semantics or generic grouping?**
At the preregistered level, mostly generic (diagnosis C).
- The operator raises own-group usefulness for shuffled groups by about the same amount (gate: F vs SHUF +5.8e-5 vs E vs C6 +4.8e-5).
- Calibration also showed that most semantic routing lift exists even in the SHUF-trained model (0.23–0.27 vs 0.26–0.30): content-driven routing that does not need semantic grouping pressure.

**5. Is it worth scaling?**
Not under this evidence. The bounded protocol rejected it at 50M, and the 245M finals were correctly not run.

The descriptive VAL signal (E > EDL and E > C6 on real-domain usefulness; E−F > C6−SHUF on both seeds) is the only reason to revisit it. Any revisit should be a fresh, preregistered confirmation:
- the λ = 0.5 configuration, unchanged;
- ≥ 4 seeds at 50M;
- a primary endpoint fixed in advance (e.g. the real-domain E−F interaction);
- evaluated on the still-unopened CALIB-SELECTION half-0 partitions.

That costs about 6 GPU-h for E and F, with the baselines reused where seeds exist. The CE cost on some seeds (s101: +0.018 vs C6) must be part of that test.

## 8. Cost, throughput, disk

**GPU time: 6.55 GPU-h in total (cap 40).**

| Phase | GPU-h |
|---|---|
| checks + calibration | 0.08 |
| search | 1.25 |
| promotion (4 × 50M + evaluations) | 2.94 |
| post-terminal descriptive VAL | 2.29 |

**Throughput:** E and F train at about 107k tok/s at 50M, vs about 123k tok/s for C6 — roughly 13% overhead, from the sorted-dispatch custom backward and the statistics. The microbenchmark operator ratio is 1.22×.

**Disk**
- Own run directories total 2.4 GiB; artifacts total 36 MB.
- 13 superseded scratch slots created by this experiment were deleted after hashing and verifying (`deleted_artifacts.tsv`).
- Free disk is 160 GiB; the 100 GiB reserve was respected throughout, and nothing was written to /dev/shm.

**Integrity**
- Protected inventory: 8,386 files, all match (`protected_inventory_before.verify_20261118T153851Z.json`).
- No pre-existing result, report, checkpoint, cluster table or preregistration was modified.
- The locked test split was never read.

## 9. Deviations and unresolved ambiguity

**Deviation 1: GPU-check tolerance** (`deviations.jsonl`, n = 1).
- The registered fixed 5e-4 was below GPU nondeterminism. It was replaced by the project's established noise-referenced rule.
- Exact (bitwise) versions of the λ = 0 and resume tests pass on CPU.
- It cannot affect the conclusion.

**Test-spec fix.** CPU tests T8d and T8e required exact equality; a 1e-12 tolerance is now used. Both files are retained.

**Operational notes**
- My own `timeout` killed the first CPU trainer-test run (≈45 s per step on CPU); the tests were then run in parallel. The slots of the aborted test were deleted and logged.
- The V1/V2 resumable slots had already been removed outside this experiment. Nothing here needed them.

**Unresolved**
- Only 2 seeds at 50M.
- The dev contrast was positive on both seeds, while the held-out contrast was mixed.
- The descriptive VAL real-domain contrast favours E on both seeds, but significantly on only one.
- E's CE varies by seed by ±0.015.
- The effects (1e-5 to 1e-4 nats) are of the same order as the seed-to-seed spread of the baselines' own semantic-vs-shuffled gap (C6−SHUF ranges from −7.4e-5 to +9.4e-5 across seeds and partitions).
- With 2 seeds, the gate could not distinguish a small real semantic effect from noise. The protocol correctly refuses to claim one.
