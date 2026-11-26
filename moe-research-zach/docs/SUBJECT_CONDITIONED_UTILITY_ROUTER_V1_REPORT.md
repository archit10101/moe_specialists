# subject-conditioned-utility-router-v1 (scur1) — final report

**Verdict: TERMINAL_STAGEA_FAIL.** No generation's causal router beat the matched generic residual or the
subject-prior-only control.
- **What does work:** a held-out subject-to-expert utility mapping is easy to estimate, reliable, and correct. Every
  semantic subject's own specialist is its best expert.
- **What blocks success:**
  1. **Irreducible causal-information gap.** Most of the semantic teacher's advantage comes from knowing the current
     region's subject at document starts, which no causal router has.
  2. **The teacher itself loses to G on stable text.** On positions ≥ 128, even perfect semantic routing of this bank
     is worse than the generic bank G.
  3. **A subject-anchored correction head fails.** It never learned the token-level utility that a free LM-trained
     router captures.
- **What was not opened:** CALIB-SELECTION, Stage B, VALIDATION and the test split.

| item | location |
|---|---|
| Preregistration (frozen) | `docs/SUBJECT_CONDITIONED_UTILITY_ROUTER_V1_PREREG.md`, sha256 `e33e05f3…9bbe` |
| Audit and logs | `artifacts/subject-conditioned-utility-router-v1/`: OBSERVATIONS.md (audit first), DECISION_LOG.jsonl, deviations.jsonl, RUN_LOG.jsonl, run_manifest.tsv, compute_ledger.tsv, disk_ledger.tsv, deletions.tsv, RESUME.md |
| Results | `stageA_gen{0..4}.json`, `analysis/A_gen{0..4}.json`, `gen{1..4}_change.json`, `final_analysis.json`, `cpu_tests_20261122T071048Z.json` |
| Routers and deployments | `routers/A_gen*/`, `deps/A_gen*/` (0400, sha256 in their json); TRAIN tables `tables/TRAIN_m1111/` |
| Code | `scur1_{extract,router_job,analyze,run,driver,tests,inventory}.py`, `src/csr/scur1_{paths,router,log,ckpt}.py` |

## 0. Audit and starting point

**asrs2 status.** asrs2 had completed as TERMINAL_STAGE1_FAIL, and nothing of it was rerun. Before this experiment
began, two pieces of asrs2 were finished:
- the C1 diagnostic, which found GPU nondeterminism rather than a fault;
- the asrs2 report, `docs/CSR_CONTINUOUS_BASE_TARGET_FREE_ASRS_V2_REPORT.md`.

**Reused, read-only and hash-verified (17 items):**
- asrs2 continuation milestones m510–m1111, which hold the base and the G, ALT-G, SEMh and SHUFh K = 5 banks plus the
  asrs2 R1 routers;
- Stage-0F ckpt_F (4 specialist seeds);
- the K = 5 SEM and matched persistent SHUF tables;
- the asrs2 SEL evaluations (full per-token 5-specialist tables).

**Prompt numbers.** The quoted m894 numbers reproduce exactly.

**Engineering checkpoint.** The pre-declared checkpoint was the final milestone m1111. CE on SEL (503 papers, 6.40M
tokens):

| reference | CE |
|---|---|
| base (TOKEN-4 replicate) | 3.61954 |
| G | 3.61262 |
| ALT-G | 3.61400 |
| existing R1 router | 3.61454 |
| region teacher | 3.61242 (−0.00158 [−0.00185, −0.00133] vs ALT-G; −0.0002 vs G) |
| per-token oracle | 3.51133 |

No artifact was incompatible. Specialist banks were never updated; only routers were trained.

## 1. Method

**Router.** The SCURouter adds 74,762 parameters; the R1 router has 74,437.
- **Backbone:** causal features [rms0(h_t), mean of the previous 16 states] → GRU(64).
- **Subject head:** p_subject.
- **Prior:** p_prior = p_subject·A.
- **Correction:** logits = log(p_prior + 1e-4)/T_prior + ρ·tanh(W_c g).
- **Deployment:** top-1.

**Training data.** Routers train on CALIB-SEARCH TRAIN, which is paper-disjoint from SEL: 498 papers, 6.1M tokens.
Each gets 2,400 steps × 64 documents with an identical seed and document order.

**Utility matrix.**
- U[s, e] = soft-subject-weighted mean of (CE_ALT-G − CE_e), computed on TRAIN only.
- **Cross-fitting:** a token's training targets use U from the other paper fold.
- **Deployed mapping:** A = softmax(U_all/(τ·sd U)).

**Losses:**
- **subject:** KL(q_sem ‖ p_subject);
- **utility:** KL(sg(q_sem·A) ‖ p_route);
- **regret:** subject-level, normalised;
- **routed LM:** through the frozen experts;
- **balance;** EDL is not used.

**Controls**, all at the same budget and seed:
- **SUBJ:** subject-prior-only, no U;
- **UTIL:** utility-only, no subject posterior, routed-LM loss;
- **SHUF twins** of every router (the shuffled-cluster utility router);
- **existing R1 router;**
- **G and ALT-G;**
- **diagnostics:** region teacher, per-token oracle, subject-mapping oracle.

**Tests.** CPU tests 8/8:
- causality;
- bounded correction;
- out-of-fold U;
- router inputs causal-only;
- mapping;
- losses;
- locks;
- smoke.

## 2. Generations (SEL, m1111, router seed 0; all failed)

Each generation's evidence, change, prediction and falsifier were written to DECISION_LOG before it ran.

| gen | change (one mechanism) | candidate − ALT-G [CI] | − G | − SUBJ [CI] | teacher closure (ALT-G) | switch | ΔI [CI], positive specialists | failed |
|---|---|---|---|---|---|---|---|---|
| 0 | τ 1, ρ 1, all losses | +0.00074 [+0.00035, +0.00113] | +0.00213 | +0.00069 | −0.47 | 0.009 | +0.0034 [0.0027, 0.0041], 4/5 | P1, P2, P3, P6, P7 |
| 1 | τ 1 → 0.25 (mapping too flat) | +0.00017 [−0.00015, +0.00050] | +0.00155 | +0.00012 | −0.11 | 0.010 | +0.0043 [0.0037, 0.0050], 5/5 | P1, P2, P3, P7 |
| 2 | drop the utility-KL and regret losses (redundant once A is sharp) | +0.00014 [−0.00017, +0.00045] | +0.00152 | +0.00009 [0.00002, 0.00016] | −0.09 | 0.011 | +0.0044 [0.0038, 0.0050], 5/5 | P1, P2, P3, P7 |
| 3 | ρ 1 → 4 (correction capacity) | +0.00018 [−0.00012, +0.00048] | +0.00156 | +0.00013 [0.00005, 0.00021] | −0.11 | 0.035 | +0.0058 [0.0052, 0.0064], 5/5 | P1, P2, P3, P7 |
| 4 | prior temperature 1 → 4 (softer anchoring) | +0.00011 [−0.00019, +0.00041] | +0.00150 | +0.00007 [−0.00001, +0.00015] | −0.07 | 0.044 | +0.0052 [0.0046, 0.0058], 5/5 | P1, P2, P3, P7 |

**How each generation ended:**
- **Generation 1:** its prediction was confirmed. The A diagonal rose to ≥ 0.88, mapping error fell to −0.00005, and
  P6 passed.
- **Generations 2–4:** each triggered its falsifier. The correction head stayed harmful (+0.00010 to +0.00017) and the
  router rarely switched.

**Controls** (identical in every generation; deterministic):

| router | CE − ALT-G [CI] | − G | − R1 existing | switch | ΔI vs its SHUF twin |
|---|---|---|---|---|---|
| SUBJ (subject-prior only) | +0.00005 [−0.00025, +0.00034] | +0.00143 | −0.00050 | 0.011 | +0.0039 [0.0033, 0.0045], 5/5 |
| UTIL (utility only, free) | **−0.00028 [−0.00053, −0.00001]** | +0.00111 | −0.00082 | 0.340 | **+0.0067 [0.0061, 0.0072]**, 5/5 |
| existing R1 router | +0.00054 | +0.00193 | 0 | 0.013 | +0.0045 [0.0038, 0.0051], 5/5 |
| SHUF twins (SCU / SUBJ / UTIL) | +0.0015 to +0.0019 / +0.0020 / +0.0005 | | | | — |

## 3. Utility matrices (generation 1, TRAIN, nats vs ALT-G; rows = subject, columns = expert)

**U_all.** The own specialist is best in every row: the row maxima lie on the diagonal. Split-half r = 0.975.

| s \ e | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| 0 | **−0.0019** | −0.0112 | −0.0118 | −0.0093 | −0.0132 |
| 1 | −0.0071 | **−0.0044** | −0.0071 | −0.0101 | −0.0100 |
| 2 | −0.0096 | −0.0092 | **−0.0012** | −0.0109 | −0.0120 |
| 3 | −0.0079 | −0.0120 | −0.0118 | **+0.0008** | −0.0151 |
| 4 | −0.0078 | −0.0078 | −0.0062 | −0.0086 | **−0.0031** |

- **Folds.** U_fold0 and U_fold1 match within 0.0011 per entry (full tables in `analysis/A_gen1.json`).
- **Deployed mapping at τ = 0.25:** diagonal 0.999 / 0.876 / 0.999 / 1.000 / 0.942.
- **Shuffled U:** no diagonal structure. Every shuffled subject's best expert is a global one (columns 0 and 2), with
  rows nearly identical (−0.0012 to −0.0040). The shuffled clusters correspond to no residual distinction; the
  semantic ones do.
- **Even own specialists mostly lose to ALT-G.** Only subject 3's own specialist beats ALT-G on average. The semantic
  bank is functionally specific, but its absolute advantage over a token-routed generic bank is small or negative.

## 4. Function, subject inference, transitions, gap decomposition

**Exposure-matched function.**
- Every SEM router gives ΔI > 0 with CI excluding 0 and 5/5 positive specialists (4/5 in generation 0).
- Generation 3's per-expert effects were +0.0040, +0.0047, +0.0027, +0.0116 and +0.0055.
- The ordinary (unmatched) interaction is reported but not used as evidence.

**Subject inference** (candidate, generations 1–4 almost identical):

| | value |
|---|---|
| accuracy vs the region label | 0.77 |
| first 32 positions | 0.51 |
| after a region change | 0.35 |
| stable regions | 0.90 |
| ECE | 0.24 |
| NMI | 0.47 |
| cross-entropy | 1.21 |

The shuffled subjects are causally unpredictable: accuracy 0.21, NMI 0.003.

**CE by position, candidate vs ALT-G** (generation 4):

| first 32 | 32–127 | 128–511 | ≥ 512 | post-transition | stable |
|---|---|---|---|---|---|
| +0.0029 | −0.0007 | −0.0002 | +0.0003 | +0.0026 | −0.0005 |

**Oracle-gap decomposition** (generation 4, CE):

| component | value |
|---|---|
| correction head | +0.0001 (harmful) |
| subject recognition | +0.0017 (first 32: +0.0193; post-transition: +0.0035; stable: +0.0003) |
| subject→expert mapping vs the hindsight-best mapping | −0.00005 (the hindsight mapping is the identity) |
| subject-granularity bank limit (mapping oracle → per-token oracle) | +0.1011 |

**Position decomposition of the teacher's −0.00158 gain over ALT-G:**
- The first 32 positions (3.2% of tokens) contribute −0.0005 (−0.0153 per token).
- Positions 32–127 contribute −0.0004.
- Positions ≥ 128 (87% of tokens) contribute −0.0007 (−0.00079 per token), where **G is −0.00144**.

A router equal to the teacher from position 32 onwards reaches only 0.61 teacher closure.

**Closure** (best SCU, generation 4):

| | value |
|---|---|
| teacher closure (reference ALT-G) | −0.07 |
| per-token-oracle closure | −0.001 |
| UTIL teacher closure | 0.18 |
| UTIL oracle closure | 0.003 |

## 5. Fairness and integrity

- All routers in a comparison share the same TRAIN rows and order, budget, seed, frozen base and banks, evaluation
  tokens and precision, and inference information (causal hidden states only).
- No expert was trained. U and A come only from TRAIN, cross-fitted for training targets.
- No SEL, CS2, VALIDATION or oracle quantity enters U, A or router inputs.
- **Deviation 1:** an optional prior-temperature argument was added before generation 4. Its default reproduces
  earlier generations exactly.
- **Protected inventory:** 1,274 files, all unchanged.
- **Compute:** 0.38 GPU-h (TRAIN extraction 5 min; about 3.5 min per generation for 6 routers; analysis on CPU).
- **Disk:** 129 → 126 GiB free; artifacts 2.8 GB.
- **Deletions:** only CPU-test scratch, logged with sha256.

## 6. What failed, and the most likely causal explanation

| hypothesis | evidence | verdict |
|---|---|---|
| Useful specialists but inadequate subject inference | Specialists are useful (U diagonal; ΔI > 0; 5/5). Causal subject accuracy is 0.51 in the first 32 tokens and 0.35 after transitions. Subject recognition is the largest causal gap (+0.0017), concentrated where the teacher's gain lives. | **supported**, but mostly **irreducible causal uncertainty**: the teacher's advantage there comes from the current region's future text |
| Incorrect subject-to-expert mapping | U is reliable (r 0.975) and its argmax equals the hindsight-best mapping. The only mapping error was a too-flat temperature (fixed in generation 1). | **not supported** |
| Specialist drift during continued training | Not tested here (Stage B not reached). asrs2 showed ΔI shrinking from 0.0167 to 0.0045 over the continuation. | supported by asrs2; untested here |
| Insufficient expert capacity / generic advantage | On 87% of tokens even perfect semantic routing (the teacher) loses to G (−0.00079 vs −0.00144). Own specialists are mostly worse than ALT-G in U. | **supported**: at K = 5 the semantic specialists are specific but not more accurate than generic token-routed experts on stable text |
| Semantic clusters not matching useful residual distinctions | Semantic U is strongly diagonal; shuffled U is flat. | **not supported**: the clusters are real residual distinctions, but small ones |
| Router–expert coupling | A subject-anchored bounded correction cannot learn alternatives: the routed-LM gradient vanishes for experts excluded by a confident prior (switch rate ≤ 0.044 at ρ = 4 and T_prior = 4). A free router (UTIL) does learn token-level utility (−0.00028, highest ΔI). | **supported** as the reason the backbone did not beat SUBJ |

## 7. Recommendation for the next experiment

Do not anchor the routing distribution on the subject prior. Instead, give the semantic subject posterior (and the
utility mapping) to a **free, LM-trained token-level router as input features**. UTIL already beats ALT-G and has the
strongest exposure-matched function; it lacks only subject information, which the subject head supplies cheaply. Test
that on frozen banks with this pipeline.

For the document-start gap, the asrs2 protocol's condition for a draft-forecast follow-up is now met: the teacher beats
the causal router specifically at document starts and transitions. Such a follow-up would use pretrained drafts as a
causal feature. It must show incremental prediction of the teacher's subject beyond h_t before any routing claim, and
it needs its own preregistration.

Either route stays capped by the bank: on stable text the semantic teacher does not beat G. A larger accuracy gain
needs specialists that are more useful than generic experts, for example finer clusters or more capacity per
specialist, not better routing.
