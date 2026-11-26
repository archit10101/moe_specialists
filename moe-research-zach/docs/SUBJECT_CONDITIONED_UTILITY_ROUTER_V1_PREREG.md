# subject-conditioned-utility-router-v1 (scur1) — preregistration

Written 2026-11-22. It was written after the audit (artifacts/subject-conditioned-utility-router-v1/OBSERVATIONS.md)
and the CPU tests (cpu_tests_20261122T071048Z.json, 8/8 pass), and before any scur1 GPU job or result. It is frozen by
the sha256 in the `.sha256` sidecar. Later changes are logged amendments or deviations.

## 0. Question and backbone

**Question.** Can a causal router infer the current semantic subject from past context and select the residual
specialist with demonstrated held-out usefulness for that subject? The aim is to approach the semantic teacher while
preserving exposure-matched subject-specific function.

**Fixed backbone.** Causal subject inference → held-out subject-to-expert utility mapping → bounded token-level
correction → unrestricted top-1 routing over exactly K = 5 frozen label-free semantic specialists. Semantic EDL is not
used (λ_EDL = 0). It may enter only as a tested regulariser in a logged generation.

## 1. Starting point (audited; all numbers recomputed)

- **Base and banks.** The asrs2 seed-0 continuation, final milestone **m1111**: the ordinary TOKEN-4-replicate base at
  50.09M continuation tokens, plus banks G_0, ALTG_0, SEMh_0 and SHUFh_0 (5 × 384→256→384 each). This is the
  pre-declared engineering checkpoint: the endpoint of the completed matched continuation, not a selected one.
  - Milestone sha256 8e19210c…8306.
  - m510, m702 and m894 are used only in Stage B.
- **Clusters.** The K = 5 clusters, SEM prior and matched persistent SHUF prior are the asrs2 tables `k5_merge.json`,
  `p5_main.npz` and `p5_calib_search.npz`, hash-verified.
- **SEL tables.** The asrs2 SEL evaluation of m1111 is reused: the full per-token 5-specialist CE table for every
  token, plus G and ALT-G routing and the asrs2 R1 router deployments. Same rows and evaluator; not recomputed.

**Reference values at m1111** (SEL, 503 papers, 6.40M tokens):

| quantity | value |
|---|---|
| base | 3.61954 |
| G | 3.61262 |
| ALT-G | 3.61400 |
| existing R1 router | 3.61454 |
| region teacher (argmax p5) | 3.61242: −0.00158 [−0.00185, −0.00133] vs ALT-G, −0.0002 vs G |
| per-token oracle | −0.1027 vs ALT-G |

**Headroom caveat.** Beating G requires about 88% of the teacher's gain over ALT-G.

## 2. Data and partitions

| partition | role |
|---|---|
| TRAIN = CALIB-SEARCH half "EVAL" (498 papers, ~6.1M tokens) | router training and U estimation |
| SEL = CALIB-SEARCH half "SEL" | all Stage-A engineering |
| CS2 = CALIB-SELECTION (1,057 papers) | opened once, in Stage B, after FROZEN_A.json |

- TRAIN and SEL are paper-disjoint from each other and from all main-stream training rows.
- VALIDATION is not opened by this protocol. The test split is never loaded.
- No M2D2 domain label is used before the frozen post-hoc audit.

## 3. Method

**Subjects.** s ∈ {0…4} = the frozen K = 5 semantic clusters. Training target:
q_sem(t) = p5(· | the token's current 256-token region), soft. It is non-causal and is a target only, never an input.

**Utility (TRAIN only, detached).**
- Definition: U[s, e] = Σ_t q_sem(t, s)·(CE_ALT-G(t) − CE_e(t)) / Σ_t q_sem(t, s). The matched generic reference is
  ALT-G's routed CE; positive means expert e helps.
- **Cross-fitting:** TRAIN papers are split into two folds by u_hash("scur1-fold", paper). The training targets and the
  training-time mapping of a fold-f token use U estimated on the other fold only.
- **Deployed mapping:** A[s, e] = softmax_e(U_all[s, e] / (τ·sd(U_all))), with U_all from all TRAIN tokens, frozen
  into the router.
- No validation, SEL, CS2 or oracle quantity enters U or A.

**Router (SCURouter).**
- **Backbone:** the ccsr1 causal features [rms0(h_t), causal mean of the preceding 16 states] → Linear → GRU(64).
- **Heads:**
  - subject head p_subject(s | x_≤t);
  - prior p_prior = p_subject·A;
  - correction head δ = tanh(W_c g_t), bounded;
  - logits = log(p_prior + 1e-4) + ρ·δ;
  - deployment = argmax.
- **Parameters:** 74,762. The R1 router has 74,437.
- **Inputs:** only causal hidden states (tested T1, T4).

**Losses** (initial weights: subject 1, utility 1, regret 1, lm 1, balance 0.01; λ_EDL = 0):
- L_subject = KL(q_sem ‖ p_subject).
- L_utility = KL(sg(q_sem·A_fold) ‖ p_route).
- L_regret = Σ_e p_route(e)·sg(max_e′ U_subj(t)[e′] − U_subj(t)[e]) / sd(U), where U_subj(t) = q_sem(t)·U_fold.
- L_LM = Σ_e p_route(e)·(CE_e(t) − CE_ALT-G(t)). This is the routed LM term through **frozen** experts; no expert
  receives any update in scur1.
- L_balance = KL(mean p_route ‖ uniform).

**Training.**
- **Data and steps:** TRAIN only, 2,400 steps × 64 documents (the ccsr1 budget), AdamW lr 3e-3 with 30-step warm-up
  and cosine decay to 10%, weight decay 0.01, clip 1.0.
- **Matching:** every router in a comparison gets an identical budget, documents and order (same seed). No early
  stopping, no evaluation data.

**Initial candidate SCU_g0:** τ = 1, ρ = 1, soft A, base features.

## 4. Controls (same TRAIN budget and seed; deployed on the same SEL tokens)

1. **G and ALT-G:** matched generic residual banks (their own token routers; from the tables).
2. **R1_existing:** the existing asrs2 semantic-prior router deploying SEMh (the "existing EDL / semantic-prior
   router").
3. **SCU_g*_SHUF:** the identical candidate pipeline on the persistent shuffled bank and prior (the shuffled-cluster
   utility router). It is the functional twin.
4. **SUBJ:** the subject-prior-only router (A = I, ρ = 0, subject loss only; R1 retrained on TRAIN), with twin
   SUBJ_SHUF.
5. **UTIL:** the utility-only router without a subject posterior (logits from the correction head, L_LM + balance), with
   twin UTIL_SHUF.
6. **Region teacher T** (argmax p5): diagnostic, non-causal.
7. **Per-token oracle** (target-aware): diagnostic only. Also a subject-level mapping oracle (the best fixed expert per
   subject, chosen on the evaluation data): diagnostic only.
8. **TOKEN-4:** the in-milestone base without residual.
9. **Compute:** all routers have equal training budgets. Inference compute is one GRU-64 router plus one specialist, the
   same as R1. No compute-matched extra baseline is needed (overhead vs R1 < 1%). The router-vs-G cost is reported.

## 5. Measurements (every deployment, every checkpoint)

- **Accuracy:** CE and paired paper-bootstrap 95% CIs (2,000 replicates) vs base, G, ALT-G, R1_existing and SUBJ;
  per-subject CE vs ALT-G.
- **Closure:**
  - closure_teacher = (CE_ref − CE_causal) / (CE_ref − CE_T). The reference is ALT-G (primary, the schedule-matched
    generic); G is also reported.
  - closure_oracle uses the per-token oracle.
- **Subject inference:** cross-entropy vs q_sem, accuracy vs the region label, ECE, NMI; by position (first 32, 32–127,
  128–511, ≥ 512); after region changes (first 64 tokens of a region whose label differs from the previous region);
  stable regions.
- **Routing:** teacher agreement, loads, entropy, switch rate, effective number, margin, confidence.
- **Function:** exposure-matched own-vs-swap interaction I (asrs2_analyze.functional, paper-level K = 5 groups);
  ΔI = I_router − I_SHUF-twin with CI; the 5 per-specialist effects; ordinary (unmatched) I reported but not used as
  evidence.
- **Utility matrices:** U_fold0, U_fold1 and U_all (5 × 5), split-half r, and A.
- **Oracle-gap decomposition** (CE differences):
  - correction head = cand − cand(ρ = 0);
  - subject recognition = cand(ρ = 0) − true-subject utility teacher argmax(q_sem·A), overall and at first-32 /
    post-switch / stable;
  - mapping = utility teacher − subject mapping oracle;
  - subject-granularity bank limit = mapping oracle − per-token oracle.

## 6. Stage A — engineering (SEL, m1111, router seed 0)

**Generation 0:** SCU_g0 + twin + SUBJ, SUBJ_SHUF, UTIL, UTIL_SHUF.

**Promotion criteria for the candidate:**

| criterion | requirement |
|---|---|
| P1 | CE < ALT-G, CI upper < 0 |
| P2 | CE < G (point estimate) |
| P3 | closure_teacher (ALT-G) ≥ 0.75 |
| P4 | ΔI(candidate − SHUF twin) > 0, CI lower > 0 |
| P5 | ≥ 4/5 specialists positive (exposure-matched) |
| P6 | no collapse: 0 dead (load < 0.02), effective number ≥ 3, max load ≤ 0.5 |
| P7 | CE < SUBJ, CI upper < 0 (the U mapping and correction add value over subject-prior-only routing) |

**Engineering loop.** At most **4 generations** after generation 0.
- **Before each generation**, DECISION_LOG records: the observed failure; the proposed causal mechanism; the exact
  change (a JSON override of the candidate, stored as genN_change.json); a quantitative prediction; a falsifier; and
  the config hash.
- One coherent mechanism changes per generation. The full candidate + controls set is rerun at the same seed and budget.
- **Allowed:** τ and normalisation; hard / soft / Sinkhorn A; subject-head context length (causal "ctx" / "chg"
  features); correction capacity / ρ (and schedule); loss weights; confidence-adaptive mixing; transition handling;
  generic fallback; EDL only as a tested low-weight regulariser.
- **Forbidden:** target-aware expert assignment for training (no expert is trained here at all); future tokens as
  inputs; any CS2, VAL or test tuning; hidden generations; extra data, compute or tokens for the candidate only.
- **Stopping rule.** The first generation whose candidate passes P1–P7 is frozen (FROZEN_A.json) and goes to Stage B.
  There is no further search after a pass.
- **Terminal conditions.**
  - If none passes after generation 4, the outcome is TERMINAL_STAGEA_FAIL with a localised failure.
  - It is also terminal earlier if the decomposition shows that the remaining gap is outside the allowed dimensions.
    Example: the true-subject utility teacher itself does not satisfy P2/P3, so no router can.
  - In that case the generation budget is not spent on noise.

## 7. Stage B — held-out confirmation (CS2 opened once; only after FROZEN_A.json)

- **CS2 tables.** The CS2 K = 5 prior is built with the frozen merge map. Extraction tables are built for TRAIN and CS2
  at the milestones needed.
- **B1.** m1111 seed-0 banks: the frozen candidate + all controls with router seeds 0, 1 and 2.
- **B2 (survival across continued training).** m510, m702 and m894 (seed 0): the candidate recipe re-calibrated on
  TRAIN at each milestone, plus SUBJ and R1_existing, router seed 0.
- **B3 (second independent specialist-training seed).** A new continuation of Stage-0F seed 1 with the asrs2 recipe
  (base every step; target-free prior assignment; 32/16 windows; arms G, ALT-G, SEMh, SHUFh; same rows), then the
  frozen candidate + controls with router seeds 0, 1 and 2 at its m1111.
- **Success** requires all of:
  - P1 and P3–P7 hold in ≥ 2/3 router seeds for **both** specialist seeds (B1, B3), and P2 is reported;
  - at every B2 milestone the candidate beats ALT-G and R1_existing (CI upper < 0) and ΔI > 0.
- Stage C (245M, VALIDATION) is not part of this protocol. It would need a new amendment and disk (about 129 GiB free
  now).

## 8. Integrity, interruption, budgets

- **Writes:** only scur1 files and directories. Protected inventory: 1,274 files, verified at the end. No asrs2 file is
  modified.
- **Logs:** OBSERVATIONS.md, DECISION_LOG.jsonl, deviations.jsonl, run_manifest.tsv, compute_ledger.tsv,
  checkpoint_ledger.tsv, deletions.tsv, RESUME.md.
- **Resumability:** every job is resumable at router or job granularity. Router checkpoints and outputs are atomic and
  hashed.
- **External interruption** (signal, STOP marker, foreign GPU process):
  - mark INCOMPLETE_STOPPED_BY_EXTERNAL_USER;
  - stop all scur1 launchers and watchers and release the GPU;
  - write the exact resume command;
  - wait for explicit permission.
- **GPU caps:** Stage A 6 GPU-h, Stage B 10 GPU-h.
- **Disk:** ≥ 100 GiB free at all times; before each job, 100 GiB + its estimate (TRAIN extraction ≈ 0.5 GB per
  milestone; SEL deployment ≈ 0.1 GB per router; the Stage-B continuation ≈ 2.5 GB).

## 9. Pre-data predictions

1. U is reliable (split-half r ≥ 0.9) and close to diagonal (each subject's own specialist is best). The mapping error
   is therefore small, and A ≈ the identity.
2. Subject-recognition error dominates the causal gap and concentrates at positions 0–31 and after region changes.
3. SCU_g0 beats ALT-G modestly: about −0.0005, closure about 0.3. It fails P2/P3. UTIL is worse than SUBJ (per-token
   utility is noisy; ccsr1 R2).
4. The probability that any generation passes P1–P7 is about 15%. Beating G is the binding difficulty, because the
   region teacher only ties G.
