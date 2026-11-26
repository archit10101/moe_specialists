# csr-continuous-base-target-free-asrs-v2 (asrs2): Phase-0 independent diagnosis of asrs1

Written 2026-11-21, before any asrs2 implementation, GPU job or preregistration. All numbers below were recomputed
from asrs1's recorded files by `asrs2_phase0.py` (CPU only; read-only on asrs1/ccsr1 files). The machine-readable
results are in `artifacts/csr-continuous-base-target-free-asrs-v2/phase0_diagnosis.json` and
`phase0_override_analysis.json`.

The diagnosis tests the hypotheses named in the request, plus alternatives. It does not assume the asrs1 report is
correct, and it corrects that report in one place (H2, section 2).

## 0. What asrs1 did (facts from code and logs)

- **Steps:** 318 to 1110 of the 245M plan, 50,086,846 tokens, the same rows as the ccsr1 S3 ordinary continuation.
- **Cycles:** 10 cycles of BASE 32 → teacher REFRESH → SPEC 32 → ROUTER 15–16 steps. Each token was used once, in
  one phase. The base made 320 optimizer steps, the specialists 320 and the routers 153 (`leg_01.json`).
- **Teacher refresh** (`asrs1_train.py:376–394`):
  - u = `all_deltas(w1, w2, h, xn, W, y)` on the cycle's not-yet-seen specialist and router rows, using those
    tokens' own targets y;
  - π = softmax(u/(2 s_u) + 0.5 log(p_sem + 1e-4)), and e* = argmax π.
- **Specialist step** (`asrs1_train.py:498–509`): each scored token updates specialist e* from the cache. These are
  the same tokens and the same targets; `cache_slice` (line 461) asserts that the cached targets equal the step
  targets.

**Code fact: the true next-token target of a token was used to choose the specialist trained on that token.**

## 1. Ranked causal diagnosis

| rank | cause | binds which failure | kind |
|---|---|---|---|
| 1 | H1: base-update starvation (320 of 793 base steps) | S1-1 (CE vs ordinary G), by +0.36 nats; alone it makes S1-1 unreachable | scientific design failure of the preregistered schedule; not a code bug |
| 2 | H2 (refined): target-dependent selection of the utility-override tokens corrupts the specialists | S1-2 (vs ALT-G, +0.23), S1-5, S1-7; the unstable ΔI | scientific / method failure of the teacher used for specialist training; not a code bug |
| — | H3 alternatives (router drift, router collapse, load collapse, specialist lr, headroom, evaluation fault, phase-isolation or resume bugs) | none | ruled out or not supported (section 3) |

### H1: base-update starvation (confirmed; explains the whole ALT-base deficit)

Evidence for:
- **Same rows and same update count.** asrs1 base training loss vs ccsr1 S3 loss on the identical first 32 rows,
  where both bases had made the same updates: difference **+1.7e-5**. The two training procedures are the same when
  the update counts are equal.
- **Matched update count, different rows** (32-step windows, after the 20-step warm-up): asrs1 base loss minus S3
  loss at the same number of base updates is **−0.015** on average. Per window it ranges from −0.049 to +0.015 (row
  noise), with no positive trend. The learning-rate schedule is identical at matched update counts (checked:
  `lr_identical_at_matched_update_count = true`).
- **Same-row deficit grows with the update gap.** asrs1 base loss minus S3 loss on the same rows, by cycle:
  0.000, 0.116, 0.228, 0.329, 0.374, 0.378, 0.382, 0.374, 0.365, 0.349. It stops growing once the curve flattens.
- **Quantified attribution.** The S3 training curve gives 3.937 at 320 updates and 3.580 at 793, a projected
  **0.357**. On SEL the measured ALT-base deficit is **0.3595** (3.98223 vs 3.62277). Essentially all of the deficit
  comes from the missing updates.
- **Reproduction when the base gets every step.** The ccsr1 S3 run is exactly that procedure (same start, rows, fresh
  moments, re-warm-up), and it reaches 3.62277 on SEL. The asrs1 base followed that trajectory step for step while
  its update count matched.

Evidence against: none found.

Falsifier, tested in asrs2:
- A base that receives every step under the asrs2 code must match the ordinary base: bitwise on CPU, and within
  measured nondeterminism on GPU, step for step.
- At step 1111 it must reproduce the ordinary SEL CE (3.62277) within the cross-process tolerance.
- If it does not, H1 is not the full story and the base path has a bug.

### H2 (refined): target-dependent selection of the override tokens corrupts specialist function

**What the evidence supports.** The teacher assignment was **mostly** the region prior. Agreement of e* with the
region-prior argmax on the specialist rows ran 0.60 (cycle 1) → 0.78 (cycle 10). The utility term overrode the prior
on 40% → 22% of training tokens.

On those override tokens the choice is target-dependent:

| cycle | override share | target → e*, override tokens (majority baseline) | target → e*, prior-consistent tokens (baseline) | KL of each specialist's assigned-target distribution, teacher vs prior-only |
|---|---|---|---|---|
| 1 | 0.40 | 0.341 (0.097) | 0.229 (0.180) | 0.675 vs 0.404 |
| 2 | 0.38 | 0.387 (0.122) | 0.242 (0.199) | 0.680 vs 0.410 |
| 5 | 0.28 | 0.380 (0.148) | 0.218 (0.186) | 0.583 vs 0.406 |
| 10 | 0.22 | 0.393 (0.172) | 0.208 (0.182) | 0.540 vs 0.412 |

Shuffled bank, same pattern, weaker: override tokens 0.23–0.30 vs baseline 0.11–0.15.

So each specialist's training set became skewed towards particular target tokens. The specialists were trained to
predict their selected targets better, and they generalised that boost to tokens with other targets.

**Utility trajectory** (held-out: every refresh measures fresh rows with the current base and current specialists):

| cycle | s_u | teacher-assigned specialist | region-prior specialist (the R1 choice) | average specialist |
|---|---|---|---|---|
| 1 (before any asrs1 specialist update) | 0.12 | +0.165 | **+0.021** | −0.008 |
| 2 (after one 32-step specialist phase) | 0.70 | +0.447 | **−0.217** | −0.310 |
| 5 | 1.48 | +0.333 | −0.394 | −0.830 |
| 10 | 1.65 | +0.259 | −0.299 | −0.856 |

The shuffled bank has the same shape: prior specialist −0.203 at cycle 2 and −0.517 at cycle 10.

Utility is CE_base − CE_specialist, so positive means helpful. Within **one** specialist phase, the prior-chosen
specialist went from helpful to harmful on new tokens.

**Controls that isolate the assignment as the cause.**
- **ALT-G, same process, same schedule.** Experts were updated in the same 32-step specialist phases on the same base,
  but assigned by the bank's own token router, which never sees the target. ALT-G stays better than ALT-base at every
  cycle (−0.035 to −0.041 on SEL). Alternation and phase length alone do not corrupt a bank.
- **ccsr1 S3, the same specialist class.** That run used the same FedBank specialists, the same `head_ce_grad`
  objective, the same AdamW settings and learning rate and fresh moments, on every step of the same rows, with
  target-free assignment by the deployed router.
  - Held-out utility of the chosen specialist, mean over 80-step windows:
    - SEMf: −0.018 → −0.006;
    - SEMj: −0.018 → −0.008;
    - SHUFf: −0.009 → −0.004.
  - Negative means helpful here, because this is CE_specialist − CE_base.
  - These specialists **never** became harmful, and they lost usefulness only gradually, as G's did
    (−0.016 → −0.007).
- What remains different between these controls and asrs1's semantic banks is the target-dependent override
  selection.

**Correction to the asrs1 report.** "Every specialist became a target-token specialist" overstates the evidence:
- The selected assignment is predicted by the target only at 0.21–0.25, against a majority baseline of 0.14–0.16,
  while the region prior predicts it at 0.60–0.78. The skew sits on the override tokens.
- Also, on SEL the per-token oracle specialist was already target-predictable at **0.396** at the frozen stage c00,
  before any asrs1 training. The shuffled bank gives 0.336. That is a generic property of an argmin over specialists
  on a single token. The rise to 0.52 at c10 is modest.
- So "oracle predictable from target" is weak evidence of corruption. The binding evidence is:
  - the override-token target skew;
  - the held-out utility collapse within one phase;
  - the two target-free controls.

**Evidence against or caveats.**
- No asrs1 arm combined FedBank specialists, this phase schedule and target-free prior assignment. The causal claim
  rests on two partial controls (ALT-G: schedule without target dependence; ccsr1 S3: specialist class and optimizer
  without target dependence), not on a single-factor ablation.
- The specialist lr (6e-4) may amplify the effect. It was identical to the stable ccsr1 S3 lr, so it is not
  sufficient on its own.

**Falsifier, tested in asrs2.** With target-free R1 assignment, everything else as close to ccsr1 S3 as possible:
- the held-out utility of the prior-assigned specialist on its own tokens stays negative-CE (helpful), and the mean
  specialist does not diverge;
- otherwise H2 does not explain the corruption.

## 2. Predictability of the specialist identity (cross-fitted by document halves; majority vote per feature value)

**Teacher-selected specialist, training rows** (SEM bank; SHUF in brackets):

| cycle | majority | target token | input token (causal) | paper semantic group | region-prior argmax | target × region prior |
|---|---|---|---|---|---|---|
| 1 | 0.136 (0.110) | 0.248 (0.155) | 0.154 (0.103) | 0.443 (0.103) | 0.599 (0.574) | 0.568 |
| 10 | 0.157 (0.141) | 0.208 (0.156) | 0.174 (0.132) | 0.547 (0.119) | 0.781 (0.718) | 0.670 |

The causal router's out-of-sample agreement with argmax π, at the first step of each router phase, was 0.45 (cycle 1)
to 0.61 (cycle 10) for SEM, and 0.07–0.14 for SHUF.

**Per-token oracle specialist, SEL swap subset** (router s0):

| | majority | target | input (causal) | paper group | region prior | causal router |
|---|---|---|---|---|---|---|
| c00 SEM | 0.099 | 0.396 | 0.151 | 0.098 | 0.102 | 0.107 |
| c10 SEM | 0.133 | 0.523 | 0.171 | 0.137 | 0.139 | 0.140 |
| c00 SHUF | 0.101 | 0.336 | 0.149 | 0.097 | 0.100 | 0.107 |

Mean specialist effect on CE: +0.006 at c00 and +0.878 at c10 (harmful). Oracle effect: −0.259 at c00 and −1.13 at
c10.

**Reading.** The semantic signal predicts the **training assignment**, not the per-token **oracle**. The oracle is
dominated by the target token even for good frozen specialists, so a target-based assignment carries information no
causal router can recover.

## 3. Alternatives (H3)

| alternative | evidence | verdict |
|---|---|---|
| Phase-isolation bug | 41 parameter-hash records at every boundary; **0** unexpected parameter changes. The base changes only in base phases, banks only in specialist phases, routers only in router phases. | ruled out |
| Resume bug | The scientific run completed in a single leg (`leg_01.json`; no resume). Resume was tested bitwise on CPU and on GPU. | ruled out |
| Faulty evaluation | The c00 identity check reproduces ccsr1 stage1.json (CE ≤ 1.2e-7, ΔI ≤ 7e-6, 3 seeds). The ccsr1 m1111 reference re-measured on SEL (SEMj −0.0003 vs G) agrees with ccsr1's CALIB-SELECTION value (−0.0004). | ruled out |
| Router drift caused by base phases | Agreement with argmax π at router-phase start is never systematically below the previous phase end (e.g. SEM_s0 0.45 → 0.61 rising). Region-prior agreement of the deployed router stays at 0.61–0.65. | not supported |
| Router collapse / load collapse | Deployed load max 0.15–0.19, effective number 11.3–12.9, 0 dead at every cycle. Switch rate 0.14 → 0.04, but region agreement rose, so routing became more prior-like, not degenerate. | ruled out |
| Specialist learning rate | Same lr and optimizer as the stable ccsr1 S3 specialists and as ALT-G. Not tested as a single-factor change. | not sufficient; possible amplifier |
| Shrinking residual headroom | ALT-G − ALT-base is −0.035 to −0.041 at every cycle, larger than ordinary G − base (−0.007). The region teacher fell **below the base** (c01 4.841 vs base 4.619), so the bank lost function; headroom did not vanish. | ruled out |
| Teacher noise as such | s_u grew 0.12 → 1.65 (utility spread across specialists rose 14×): a consequence of the corruption, not random noise. | subsumed by H2 |

## 4. Implementation versus scientific failures

- **Implementation:** none found. Isolation, caches, evaluation identity and resume all hold.
- **Scientific / design:**
  1. The schedule withheld 59.6% of the base updates (H1).
  2. The specialist-training assignment used same-token target utility (H2).
  3. The frozen §8.1 rule correctly prevented an engineering generation, because a phase-ratio change could not repair
     (1).

## 5. What asrs1 did and did not rule out

**Ruled out:**
- alternating schedules that withhold base updates, as a way to match ordinary-G CE;
- training specialists on tokens chosen by same-token target utility (hard EM on in-sample utility), even when the
  semantic prior dominates the assignment.

**Not ruled out:**
- target-free specialist assignment (semantic prior only, soft sampling, or utility measured on other tokens);
- a schedule in which the base updates on every batch;
- other numbers of clusters or specialists;
- replication across specialist-training seeds;
- any 245M or VALIDATION behaviour.

## 6. Why the V2 changes address each binding failure

- **H1 → continuous base.** The base takes its ordinary LM update on every main batch, through the unchanged ccsr1 S3
  code path. Residual branches only see detached hidden states and the detached head, so base gradients cannot depend
  on any specialist or router. This is shown bitwise on CPU and within measured nondeterminism on GPU before any
  scientific run.
- **H2 → target-free assignment.**
  - Primary R1: the specialist that trains on a token is a fixed function of the frozen semantic region prior
    (top-1, or a sample from it with an RNG fixed per window). No CE, gradient or target of that token enters the
    choice.
  - The router target is the same prior (KL).
  - Any utility signal (R2) must be measured on other tokens (paper-disjoint), cached and frozen before the window.
- **Alternation is kept only for the auxiliary modules** (32 specialist / 16 router batches). The base updates on all
  48 batches.

## 7. User amendment: K = 5 clusters and specialists (received during Phase 0)

The user then required exactly K = 5 label-free semantic clusters, with five specialists in each bank. The SEM,
shuffled and generic banks must all have identical size, width, active compute and update budget. Consequences:

- **The asrs1/ccsr1 positive frozen bank (K = 16, seed 100) cannot be reused.** asrs2 must build its own frozen stage
  (Stage 0F) for K = 5 before any continuation. That adds a new failure mode: a K = 5 semantic bank may not be
  functionally specialised even when frozen.
- **The K = 5 clusters must be label-free and frozen.** They are built only from the frozen GC-EDL SEM-K16 semantic
  pipeline (BGE embeddings + ProtoNet; λ_coh = 0, so no gradient or functional information; medoid seed chosen
  label-free in GC-EDL). No domain label, validation result or expert-function measurement may enter.
- **K = 5 equals the number of M2D2 domains.** That makes a post-hoc subject audit natural, but the labels stay closed
  until that audit.
- **Thresholds stated for K = 16 must be translated to K = 5** before any data (e.g. "≥ 12/16 positive specialists"
  → "≥ 4/5", the same ≥ 75% fraction).
- **Comparisons with the earlier K = 16 results** (ccsr1 −0.004 vs G) are descriptive only, because K differs.

## 8. Remaining ways V2 could fail

1. **K = 5 frozen stage not functional.** A coarser partition may give generic specialists (semantic ≈ shuffled). Then
   there is nothing to preserve.
2. **Capacity and headroom.** Five 384→256→384 specialists = 983k parameters. ccsr1 S3 showed the generic residual
   gain falling from 0.017 to 0.007 as the base improves, so any semantic advantage may fall below resolution.
3. **The causal router cannot recover the region prior at topic transitions.** ccsr1 lost about 0.005 on
   post-switch tokens.
4. **Semantic ≈ shuffled.** A persistent grouping alone may explain most of the gain (seen in semedl, slc, jsr).
5. **Hard top-1 R1 may over-specialise early tokens of mixed regions; soft sampling may under-specialise.**
6. **Residual drift.** Specialists trained on a moving base may lose own-vs-swap advantage even with target-free
   assignment (ccsr1 SEMj kept only about 30% of ΔI).
7. **Disk.** About 137 GiB free against the 100 GiB reserve. Stage 3 (245M) may not fit.
8. **Specialist-seed variance at K = 5** may exceed the effect.
