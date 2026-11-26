# csr-alternating-semantic-residual-specialization-v1 (asrs1) — final report

**Verdict: TERMINAL_STAGE1_FAIL. Negative.** Alternating optimisation did not preserve the frozen bank's functional
semantic specialisation while keeping CE. It lost badly against both the ordinary generic bank G and the schedule-matched
ALT-G, in all three router seeds. Under the frozen rule (prereg §8.1) no engineering generation was permitted. Stage 2
(CALIB-SELECTION), Stage 3 (245M, VALIDATION) and the draft-forecast ablation were not run. CALIB-SELECTION and
VALIDATION were never opened, and the test split was never loaded.

| item | location |
|---|---|
| Preregistration (frozen before any calibration data) | `docs/CSR_ALTERNATING_SEMANTIC_RESIDUAL_SPECIALIZATION_V1_PREREG.md`, sha256 `898fbfcb…6c6a` |
| Logs | `artifacts/csr-alternating-semantic-residual-specialization-v1/`: OBSERVATIONS.md, DECISION_LOG.jsonl, deviations.jsonl, run_manifest.tsv, compute_ledger.tsv, disk_ledger.tsv, deletions.tsv, checkpoint_ledger.tsv, RESUME.md |
| Results | stage1.json, analysis/*.json (per checkpoint), final_analysis.json, posthoc_oracle_predictability.json |
| Tests and integrity | cpu_tests_20261121T234831Z.json, gpu_check_*.json, reuse_manifest.json, protected_inventory_{before,verify_*}.json |
| Run | `runs/20261122T001339Z_asrs1_S1`: 10 cycle milestones ckpt_c01…c10 and the final resume slot (retained). Teacher caches are in `artifacts/…/teacher_caches/` (0400, hashed). |
| Code | `asrs1_{train,eval,analyze,run,driver,tests,inventory}.py`, `src/csr/asrs1_{paths,cfg,ckpt,log}.py` |

## 1. Method as run

**Starting point.** The hash-verified cfar1/ccsr1 step-318 checkpoint: the TOKEN-4 replicate base at 20.06M tokens, plus
three banks:
- G, the token-routed generic bank;
- L, the semantic bank;
- F2h_T500_w240, the matched persistent shuffled bank.

The ccsr1 R3_c2e0.5 causal routers (seeds 0, 1 and 2) came with it.

**Architecture.** Unchanged from ccsr1. The base is never trained through a residual. Each bank has 16 specialists
(384→256→384), with exactly one active per token. The current-context GRU router (75k parameters) chooses the specialist.

**Schedule.** Steps 318…1110 of the 245M plan, which are exactly the rows of the ccsr1 ordinary continuation (50,086,846
tokens). They were split into 10 complete cycles of BASE 32 steps → frozen TEACHER REFRESH → SPECIALIST 32 → ROUTER
15–16. That gives 20.21M base, 20.20M specialist and 9.67M router tokens; every token was used exactly once, in one phase.

**Teacher.** At each refresh, the frozen snapshot computed exact 16-specialist utilities on the cycle's not-yet-seen
specialist and router rows. The teacher was π = softmax(u/(2·s_u) + 0.5·log(p_region + 1e-4)), the ccsr1 R3_c2e0.5
definition. Specialists trained on argmax π; routers were distilled from π with the validated ccsr1 objective (weighted
KL + 0.01 balance). No routed-LM term had ever been validated, so none was added. Caches were read-only, hashed and
alignment-checked before every use.

**Arms (one process, one base, identical rows):**
- **ALT-G:** G with the same phase/update schedule;
- **ASRS-SEM:** L plus 3 routers;
- **ASRS-SHUF:** the shuffled bank plus 3 routers.

**Reused references (verified, re-measured on SEL, not retrained):**
- **A:** the ordinary base continuation;
- **B:** the ordinary G;
- **C:** the ccsr1 simultaneous SEMj/SEMf, from the same 50M rows.

The frozen stage (c00) is the step-318 checkpoint measured with the same evaluator. It reproduces ccsr1 stage1.json
exactly: CE to 1e-7, ΔI identical.

**Tests before the freeze.** All CPU tests T1–T9 passed:
- phase isolation, verified by parameter hashes at every boundary;
- teacher caches read-only and recomputed bitwise identically;
- identical budgets across arms;
- bitwise resume after a max-steps stop, after SIGINT at a step, and after SIGINT inside a refresh;
- zero residual equal to the base;
- causality of the base and router;
- no held-out access.

The GPU check (isolation, SIGINT/resume) also passed.

## 2. Results (CALIB-SEARCH SEL, 503 papers, 6.41M tokens; cycle 10 = step 1111)

**References (CE, nats):**

| model | CE |
|---|---|
| ordinary base (A) | 3.62277 |
| ordinary G (B) | 3.61554 |
| ccsr1 SEMj s0/s1/s2 (C) | vs ordinary G −0.00029 / −0.00021 / −0.00033 |
| ccsr1 SEMf s0/s1/s2 (C) | vs ordinary G +0.00138 / +0.00141 / +0.00138 |
| **ALT-base** (alternating base, 320 of 793 base steps) | **3.98223** (+0.3595 vs ordinary base; predicted before the data: +0.3, range 0.15–0.45) |
| ALT-G | 3.94685 (ALT-G − ALT-base −0.035) |

**Stage-1 criteria per router seed** (CIs are paper-bootstrap 95%):

| | s0 | s1 | s2 |
|---|---|---|---|
| S1-1 ASRS-SEM − ordinary G ≤ 0.001 | +0.5639 [0.535, 0.593] ✗ | +0.5649 ✗ | +0.5689 ✗ |
| S1-2 ASRS-SEM − ALT-G < 0 | +0.2326 ✗ | +0.2336 ✗ | +0.2376 ✗ |
| S1-3 ΔI > 0, CI lower bound > 0 | +0.0047 [−0.0064, 0.0153] ✗ | +0.0172 [0.0044, 0.0299] ✓ | +0.0129 [0.0022, 0.0249] ✓ |
| S1-4 ΔI ≥ 0.75 × 0.01055 | 0.44× ✗ | 1.63× ✓ | 1.23× ✓ |
| S1-5 ≥ 12/16 positive specialists | 6 ✗ | 10 ✗ | 7 ✗ |
| S1-6 no collapse (0 dead, eff ≥ 8, max ≤ 0.25) | 0 dead, 11.6, 0.176 ✓ | 0, 11.9, 0.166 ✓ | 0, 11.8, 0.172 ✓ |
| S1-7 positions ≥ 128 better than ALT-G; ≥ 12 groups; share ≤ 0.5 | +0.232, 0 of 16 groups ✗ | +0.232, 0 ✗ | +0.234, 0 ✗ |
| S1-8 post-transition / stable vs ALT-G (report only) | +0.309 / +0.196 | +0.309 / +0.198 | +0.311 / +0.200 |
| ASRS-SEM − ccsr1 SEMj (same seed) | +0.564 | +0.565 | +0.569 |

**Per-specialist own-vs-swap utility (cycle 10, 16 specialists).** Three specialists (5, 12, 15) have too few routed
tokens to score in every seed.

| seed | specialists 0–15 |
|---|---|
| s0 | +.090 +.014 −.007 −.002 +.034 n/a +.025 −.056 −.050 −.018 −.019 +.058 n/a +.032 −.015 n/a (6 significantly positive or positive-CI: 0, 4, 11, 13) |
| s1 | +.078 +.027 +.011 +.031 +.061 n/a −.048 +.024 −.122 −.011 +.004 +.051 n/a +.065 +.029 n/a |
| s2 | +.077 −.009 +.009 +.021 +.045 n/a −.033 +.017 −.034 −.012 −.013 +.080 n/a −.009 +.032 n/a |

For comparison, the frozen stage c00 had 13 / 13 / 12 positive specialists (10 / 8 / 9 significant).

**Trajectory (mean over router seeds; full table in stage1.json):**

| cycle | ALT-G − ALT-base | ASRS-SEM − ALT-G | region teacher − ALT-G | ΔI s0/s1/s2 | positive specialists | switch | region agreement |
|---|---|---|---|---|---|---|---|
| 0 (frozen, step 318) | −0.017 | −0.0038 | −0.0078 | .011/.008/.012 | 13/13/12 | 0.14 | 0.63 |
| 1 | −0.039 | +0.296 | +0.261 | .079/.041/.069 | 12/13/12 | 0.13 | 0.62 |
| 2 | −0.041 | +0.463 | +0.444 | .096/.088/.091 | 11/12/11 | 0.11 | 0.62 |
| 4 | −0.041 | +0.347 | +0.399 | .007/.070/.041 | 9/10/9 | 0.06 | 0.63 |
| 6 | −0.041 | +0.274 | +0.352 | .011/.000/.002 | 9/8/8 | 0.05 | 0.63 |
| 8 | −0.039 | +0.250 | +0.342 | .022/.003/−.001 | 8/8/10 | 0.05 | 0.63 |
| 10 | −0.035 | +0.235 | +0.317 | .005/.017/.013 | 6/10/7 | 0.04 | 0.64 |

**Teacher refreshes (training rows, SEM bank):**

| cycle | s_u | utility override share | teacher-assigned specialist | region-prior specialist | mean specialist |
|---|---|---|---|---|---|
| 1 | 0.12 | 0.40 | +0.165 | +0.021 | −0.008 |
| 2 | 0.70 | 0.38 | +0.447 | −0.217 | −0.310 |
| 10 | 1.65 | 0.22 | +0.259 | −0.299 | −0.856 |

The shuffled bank followed the same pattern.

## 3. Answers

**Did alternating preserve functional semantic specialisation with competitive CE?** No.
- The deployed semantic system is 0.23 nats worse than the schedule-matched ALT-G and 0.56 nats worse than the ordinary
  G, in 3 of 3 router seeds.
- Specialisation is not preserved in a functional sense. Only 6–10 of 16 specialists are positive. ΔI is positive in
  2 of 3 seeds, but it swings wildly across cycles: 0.08 at cycle 1, about 0 in cycles 6–9.

**Semantic vs shuffled.** The two banks degraded identically. Their teacher statistics, the collapse of the region-prior
specialist's utility and their CE losses all match. The positive ΔI in s1/s2 is not evidence of semantic function: it
appears when every specialist is harmful off its own tokens, and it is unstable.

**Routing at topic transitions.**
- Post-transition tokens are +0.31 vs ALT-G and stable-region tokens +0.20.
- Positions 0–31 are +0.26.
- The router itself is not the bottleneck. Region-prior agreement stayed at 0.63–0.65 throughout, and the non-causal
  region teacher deploying the same bank was even worse (+0.32 vs ALT-G).

**Retention across continuation.** CE retention: none. The frozen stage was −0.0038 vs G; cycle 10 is +0.23 vs ALT-G.
ΔI retention by seed is 0.44 / 1.63 / 1.23 of the frozen stage, but this metric has lost its meaning here (see the first
answer).

**Which failure was it?** The prompt names five classes; the evidence points to two separate failures.
1. **Insufficient base updates (structural, inherent to the preregistered cycle).**
   - The base gets 320 of 793 optimiser steps, which costs +0.36 nats (predicted before the data).
   - S1-1 is therefore unreachable by about two orders of magnitude, whatever the specialists do.
   - Even at a base share of 0.8, the ordinary training curve predicts a 0.052-nat deficit, against the ≤ 0.003 the
     rule requires.
2. **Specialist corruption caused by the teacher (the dominant failure against ALT-G).** This is closest to "teacher
   noise", but it is systematic rather than noise.
   - The semantic-plus-utility teacher's utility is computed with the target token.
   - Using its argmax to decide which specialist trains on a token amounts to hard EM on target-dependent utility.
   - Within one 2M-token specialist phase, every specialist became a target-token specialist: +0.45 nats on its assigned
     tokens, −0.31 nats for the average specialist on a random token.
   - Post hoc and descriptive only: from the target token, the oracle specialist can be predicted with cross-fitted
     accuracy 0.39 at the frozen stage and 0.52 at cycle 10. From the semantic group or region prior it cannot be
     predicted better than the majority baseline (0.10 → 0.14).
   - Such a bank cannot be deployed without knowing the next token. In ccsr1 the same target was harmless because the
     specialists were frozen and only the router learned from it.

**What it was not:**
- **Base-driven router drift:** there was no agreement drop at the start of router phases, and region agreement stayed
  constant.
- **Router collapse:** loads stayed healthy (max ≤ 0.18, effective number ≥ 11.6, 0 dead). The switch rate fell from
  0.14 to 0.04, but routing stayed aligned with the region prior.
- **Shrinking residual headroom:** ALT-G gains more over its weaker base (−0.035) than the ordinary G does (−0.007).

**Speed, memory, disk, compute.**
- **Training:** 2,556 s for the whole 50M-token alternating run (one leg).
- **Teacher refreshes:** about 210 s each; that is 2 × 16-specialist utilities on 3M tokens per cycle, a training
  overhead that ALT-G does not have.
- **Memory:** peak 17.2 GiB.
- **SEL evaluation:** about 5 min per checkpoint (15.6 min for the 9-bank ccsr1 reference).
- **GPU:** 1.93 GPU-h used against a Stage-1 cap of 12 (GPU check 0.04, Stage 1 1.90).
- **Disk:** free space went from 149 GiB to 137 GiB, never below the 100 GiB reserve.
- **Space used:** 6.3 GB artifacts (evaluations, teacher caches), 2.8 GB run (milestones), 1.4 GB retained final slot.
- **Deletions:** only this experiment's test and GPU-check scratch and rotated resume slots, each logged with sha256 in
  deletions.tsv.
- **Protected inventory:** 613 files, all unchanged.

**Deviations** (deviations.jsonl; none affected a criterion or decision):
- Before the freeze, a specialist-lr multiplier hook (default 1.0) was added while the CPU tests were running. The tests
  that ran on that code matched the earlier run bitwise.
- The oracle-predictability analysis is post hoc and descriptive only.

## 4. Strongest supported conclusion, and what remains unproven

**Supported.** Under the preregistered cycle, alternating optimisation fails for two separate reasons:
- **The schedule** starves the base of updates, so ordinary-G-level CE is impossible.
- **The teacher:** hard assignment of specialist training tokens by a target-dependent semantic-plus-utility teacher
  destroys deployable specialisation. Specialists become target-token experts that no causal router can use, and the
  semantic and shuffled banks degrade identically.

The ccsr1 positive result (causal deployment of *frozen* semantic specialists, −0.004 vs G) is not extended by this
method. Nothing here is a routing-only claim of specialisation. No accuracy or functional criterion passed in more than
2 of 3 seeds, and CE failed everywhere.

**Unproven, not tested here:**
- Whether alternating works with a **target-free** assignment signal for specialist training:
  - the semantic prior alone (R1);
  - cross-fitted or partner-view utility, i.e. utility measured on other tokens, never the token being trained;
  - soft rather than hard assignment.
- Whether alternating works with phases that do not withhold tokens from the base, e.g. specialist and router updates on
  the base-phase tokens. That would need a different, re-preregistered schedule and token accounting.
- Replication across specialist-training seeds. One specialist seed was used; the three router seeds only show router
  variance.
- Any 245M or VALIDATION behaviour.

Any follow-up needs a new preregistration.
