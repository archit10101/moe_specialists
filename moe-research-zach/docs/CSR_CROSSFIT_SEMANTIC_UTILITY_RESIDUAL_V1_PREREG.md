# CSR-CROSSFIT-SEMANTIC-UTILITY-RESIDUAL-V1 (csur1) — Preregistration

Status: frozen when its sha256 is written to `docs/CSR_CROSSFIT_SEMANTIC_UTILITY_RESIDUAL_V1_PREREG.md.sha256`
(mode 0400). No scientific job (Stage 1 or later) runs before the freeze; every config embeds the hash. Tests, the
region-prior tables and the reuse inventory are produced before the freeze and do not depend on any result.

Evidence synthesis: `docs/CSR_CROSSFIT_SEMANTIC_UTILITY_RESIDUAL_V1_EVIDENCE.md`.

## 1. Question and claims

Can label-free semantic information produce genuinely subject-specialised residual experts when assignment is
determined jointly by (i) semantic plausibility and (ii) cross-fitted next-token-loss reduction, and does the
specialisation survive a deployable causal router?

**A positive claim requires all four (§10):**
- exposure-matched semantic functional interaction above shuffled;
- multiple specialists carrying it;
- a causal student CE no worse than the parameter-matched generic residual;
- integrity.

Routing NMI or utilisation alone is never evidence of specialisation.

## 2. Reused inputs (read-only; `artifacts/csr-crossfit-semantic-utility-residual-v1/inventory.json`, all hashes verified)

- **TOKEN-4 recipe**, from the T4 s0 config:
  - LFRModel: d = 384, 12 layers, 32 experts top-4, 103,609,728 params;
  - WSD schedule, lr 6e-4, AdamW (0.9, 0.95), wd 0.1, clip 1.0;
  - 64 rows × 1,024 tokens per step;
  - balanced main stream (plan sha f4bc3c93…);
  - budget 245,032,457 tokens = 3,880 steps.
- **Reference checkpoints:** TOKEN-4 and EDL 245M p100, seeds 0–3. Evaluated on VAL only after the final freeze; never retrained.
- **Semantic pipeline:**
  - GC-EDL SEM K16 ProtoNet (sha 6a5e9867…), residualizer (sha 40c49358…), bge-small-en-v1.5.
  - This is the pipeline that produced the C6/ORV1 paper clusters.
- **ORV1 tables:**
  - paper-level semantic labels and the length-stratified, token-mass-matched paper shuffle (main, CALIB-SEARCH, VAL);
  - partitions SEL = CALIB-SEARCH half A, EVAL = half B, VAL;
  - token features.
- **Budget step indices** (jsr tables.json):
  - S1 318 steps (20,057,391 tokens);
  - S2 793 (50,047,429);
  - M100 1584, M150 2375, M200 3167;
  - final 3880.

**Region prior tables** (`csur1_psem.py`, label-free, written before the freeze):
- Each document is cut into regions of 256 tokens ([256j, 256j+256)).
- Each region goes through the identical semantic pipeline; p_sem = softmax(ProtoNet logits).
- Regions shorter than 16 tokens inherit the previous region.
- CALIB-SEARCH: mean max p 0.445, entropy 1.78 nats, 60% agreement with the paper label, within-document switch rate 0.30.
- VAL priors are computed only after the final freeze.

## 3. Architecture

**Base:** the normal TOKEN-4 MoE. Shared expert path and top-4 token routing are unrestricted and unchanged.

**Residual bank** after the final MoE block and the final norm (the ORV1/JSR residual architecture):
- z = sg(W)(sg(h) + a), with a = Expert_e(rms0(sg(h))).
- K = 16 GELU MLPs 384→256→384: 3,145,728 parameters.
- W1 ~ N(0, 0.02) with seed + 7,000,003; W2 = 0, so the residual output is zero at initialisation.
- Exactly one specialist is active per token.
- No specialist replaces or restricts the top-4 experts.
- The base never receives a gradient from any bank. h and W are detached and the CE gradient is formed explicitly.

**Routing units:**
- view = a 128-token block of a document, positions [128j, 128j+128);
- region = views (2i, 2i+1) = A, B.
- Documents never share views; assignments may change between views.

**Cross-fitted utility (training only, teacher):**
- For every view v and specialist k, using the EMA copy of the specialists:
  u[v,k] = mean over the utility tokens of v of CE(base) − CE(base + specialist_k).
- This is exact: all 16 alternatives, shared base logits, fp32 log-sum-exp.
- Training utility tokens are the scored tokens with (pos + step) mod 4 = 0, a rotating quarter (32 of 128 per full view), kept for compute. The eval teacher uses all tokens.

**Partner:**
- partner(v) = the other view of v's region if that view has ≥ 32 scored tokens;
- otherwise the previous view of the same document if it has;
- otherwise none, in which case the u term is 0.
- The utility of v never weights v's own training (test T3).

**Responsibility (relative utility):**

    uc[v,k] = u[v,k] − ubar_k        (ubar_k = token-weighted mean of u[·,k] over the step's views)
    r[v]    = softmax( uc[partner(v)] / tau_t  +  eta · log(prior[v] + 1e-4)  −  lam · log(K · m) )

- **tau_t** = c · s_t, where s_t = this step's mean within-view across-specialist SD of uc. It is lag-free and c is dimensionless. Eval uses the training EMA (ρ 0.9) of ubar and s_t.
- **prior:**
  - SU, SO: p_sem of v's region;
  - SH: p_SH;
  - UO: none.
- **m:** EMA (ρ 0.99) over steps of the token-weighted mass of r. This is a mild corpus-level usage prior; there is no per-minibatch balance.

**Pre-freeze change (GPU check v1, logged).** Absolute utility with an EMA τ collapsed every utility bank within 20 steps (14/16 dead): it rewarded the generically best specialist. Centring each specialist's utility asks where the specialist is comparatively useful, which is the definition of specialisation. GPU check v2: 0 dead, max teacher load ≤ 0.18 at step 50.

**Specialist training:**
- k_v ~ r[v], one sample per view, from a per-(seed, bank, step) generator; stop-gradient through r.
- The live specialist k_v is trained on all scored tokens of v with CE; LR = lr_at(step).
- This is the unbiased one-active-specialist estimator of Σ_k r[v,k] · L_res_k(v).
- The EMA specialists then update with decay `ema`.

**Causal student router** (params 19,152 ≤ cap 21,873 = 6,144 + 0.5% of the expert params):
- s_j = mean rms0(sg(h)) over view j.
- Linear(384→32) → GRUCell(32,32) → Linear(32→16).
- q[v] uses the GRU state after the completed views 0..v−1 of the same document. View 0 uses the zero state, i.e. a learned prior through the head bias.
- Loss: KL(sg(r[v]) || q[v]) averaged over scored views.
- Router and expert gradients are clipped separately (1.0); one AdamW per bank (wd 0.1).

**Inference modes:**
- **student (deployable, primary):** e(v) = argmax q[v], gate 1, every view including view 0. Uses no semantic embedding, no utility and no future token.
- **teacher (training-only oracle, diagnostic):** e(v) = argmax r[v]. Uses the partner view's utility (possibly future tokens) and the region's prior.
- **student-base0 (secondary):** student with base-only predictions in view 0.

## 4. Conditions (one process per seed: shared base, identical rows and row order, identical expert initialisation)

| id | assignment | params (experts + router) |
|---|---|---|
| A_int | the shared TOKEN-4 base itself (no bank) | 103,609,728 |
| G | generic residual: per-token linear top-1 router with Switch balance 0.01 (ORV1 C / JSR B) | 3,145,728 + 6,144 |
| SU | semantic-utility (proposed): u + η log p_sem + usage prior | 3,145,728 + 19,152 |
| SH | identical method with the persistent shuffled prior p_SH | 3,145,728 + 19,152 |
| SO | semantic-only: η log p_sem + usage prior, no u term | 3,145,728 + 19,152 |
| UO | utility-only: u + usage prior, no prior (secondary) | 3,145,728 + 19,152 |

**Shuffled prior p_SH (persistent paper-to-paper transplant, `csur1_psem_shuffle.py`):**
- Papers are binned into 20 token-count quantile bins. Within each bin they are placed in a seeded random cyclic order, and each paper takes its successor as source π(P) ≠ P.
- P's regions, in document order, receive π(P)'s region priors at the same relative position.

What p_SH preserves:
- the set of prior rows (sharpness, entropy);
- within-paper persistence and switching dynamics (those of a real paper);
- paper lengths and the number of groups;
- token mass per group: on main, region-argmax max |Δmass| = 0.016 and soft 0.011.

What it destroys: the link between text and prior (argmax agreement with p_sem 9.4%, chance level).

A first construction, a per-paper relabelling of clusters, was rejected before the freeze: its region-level token mass deviated by up to 0.054 (T7).

**Fairness:**
- Active inference compute is identical: one 384→256→384 MLP per token in every bank. The router is negligible: a GRU per view vs a linear map per token.
- Router parameters differ by 13,008 (0.41% of a bank).
- Utility measurement costs extra training-time FLOPs in SU, SH and UO, but uses no extra tokens, steps or parameter updates.

**References:**
- reused TOKEN-4 and EDL 245M on VAL (unpaired seeds 0–3);
- ORV1 and JSR results are cited, not re-run.

## 5. Data, partitions, seeds, budgets

- **Training:** the balanced main stream; the same rows for all conditions and seeds.
- **Evaluation:**
  - SEL (Stage 1);
  - EVAL (Stage 2; papers disjoint from SEL);
  - VAL (final, opened once after `FROZEN_FINAL.json`).
- The locked test split is never loaded (guarded).
- **Seeds:**
  - Stage 1: 100;
  - Stage 2: 100 and 101;
  - final: 100, 101, 102, 103. Seeds 100 and 101 continue their Stage-2 runs, which are exact prefixes of the 245M schedule.
- **Swap subset** (forced specialist columns): papers with u_hash("csur1-sub") below a threshold:
  - SEL: 0.25;
  - EVAL: 1.0;
  - VAL: 0.15.

## 6. Tests (all must pass before the freeze)

**CPU tests:**
- T1 layout and partners;
- T2 exact utilities vs float64;
- T3 cross-fit isolation;
- T4 student causality;
- T5 base bitwise identical with and without banks;
- T6 bitwise resume (including EMA specialists, usage prior and scale);
- T7 shuffled-prior properties: rows drawn from p_sem, one source paper per paper, no self-mapping, region-argmax token mass within 0.03;
- T8 equal parameters, initialisation and one-specialist-per-token;
- T9 gradient separation (experts only from residual CE, router only from KL, none to h);
- T10 sampling determinism;
- T11 reliability estimator recovers a planted correlation;
- T12 the test split is locked and VAL is locked before the freeze.

**GPU checks:**
- G-EQ: step-0 base metrics equal to the reused T4 s0.
- G-ISO: banks vs no-bank mean |Δlm| ≤ 1.5 × no-bank repeat noise + 0.002.
- G-RESUME, for lm and every bank's residual loss, both conditions:
  - resumed vs the nearest of three straight runs ≤ 1.5 × the maximum straight-vs-straight distance + 0.002;
  - the first post-resume step within the straight runs' spread + 0.002.
- G-BENCH: tokens/s, step time and peak memory.
- G-EVAL: forced-specialist CE equals the routed CE (max abs < 0.02, fp16 storage) on an eval smoke run.

## 7. Stage 1 — bounded search (seed 100, 20M tokens, SEL)

One process: G plus 7 candidate pairs (SU_c, SH_c) that differ only in the prior.

**Base point:** c = 1.0, η = 0.5, ema = 0.99, λ = 0.3.

**One-factor variants:**
- c ∈ {0.5, 2.0};
- η ∈ {0.25, 1.0};
- ema = 0.9;
- λ = 1.0.

**Gates per candidate:**
- student CE_SU − CE_G ≤ 0.002;
- no pathology for the SU student and teacher loads (max load ≤ 0.25, effective number ≥ 8, ≤ 1 dead specialist with load < 1/160).

**Selection:**
- arg max of the exposure-matched functional interaction I_SU − I_SH among gated candidates;
- ties (< 1e-5) go to the base point, then to grid order.

**Outcomes:**
- Stage 1 fails only if no candidate passes the gates.
- If the best score is ≤ 0, a warning is logged and Stage 2 still runs; it is the test.
- Routing NMI is never a selection criterion.

## 8. Stage 2 — confirmation (seeds 100, 101; 50M; EVAL)

The selected hyperparameters are frozen. Banks: G, SU, SH, SO, UO.

**Promote only if all four hold:**
- **P1:** I_SU − I_SH > 0 in both seeds.
- **P2:** the causal student retains ≥ 40% of the teacher's semantic-over-shuffled CE advantage, pooled over seeds:
  Σ(CE_SH − CE_SU)_student / Σ(CE_SH − CE_SU)_teacher ≥ 0.4. The teacher advantage must be > 0, otherwise P2 fails.
- **P3:** student CE_SU − CE_G ≤ 0.002 in each seed.
- **P4:** no collapse or pathology for SU (student and teacher loads, as in Stage 1).

## 9. Metrics

All CE contrasts are paired over papers (2,000-replicate paper bootstrap; final: 10,000-replicate hierarchical seed × paper bootstrap).

**Functional (primary):** exposure-matched own-specialist usefulness I_X, as in JSR.
- g_e(t) = mean_{j≠e} d[t,j] − d[t,e], with d = CE with live specialist j forced − CE_base on swap-subset tokens.
- The own specialist σ(k) = arg max routing lift of X's deployed (student) assignment for semantic group k, fitted on one paper half and measured on the other (both directions).
- Own set: group-k tokens routed to σ(k). Other set: other-group tokens routed to σ(k).
- Strata (common support, own-stratum weights): q_e quintile × margin tercile × base-CE quintile × position bin × doc-length tercile × target-frequency tercile.
- Groups: paper-level C6 semantic clusters (primary); region-level p_sem arg max and real domains (secondary; real domains post hoc only).

**Per-specialist:**
- I_k per group with per-seed signs and CIs;
- the number of specialists positive and individually significant (CI lower bound > 0);
- the largest share of the positive contribution.

**Swap matrix:**
- the complete 16 × 16 group × specialist matrix;
- diagonal advantage = mean_{j≠σ(k)} M[k,j] − M[k,σ(k)].

**Routing:**
- NMI with semantic group, region prior, shuffled group, and real domain (post hoc);
- load, effective number and entropy;
- stability across disjoint views (P(e_A = e_B)) and across seeds (NMI between seeds' assignments on identical tokens).

**Accuracy:**
- CE by seed and pooled;
- by semantic group and real domain;
- by position (0–127, 128–511, 512–1023);
- vs A_int, G, SH, SO, UO and the references.

**Router retention:**
- semantic-over-shuffled retention;
- teacher gain retention vs base;
- KL, agreement and q entropy by view number;
- teacher-student token agreement.

**Online training diagnostics (every 10 steps):**
- split-half reliability of block utility (double-centred Pearson r between views A and B of a region) and arg-max agreement;
- the utility spread s_t;
- agreement of arg max r with the prior and with the utility term;
- r entropy, usage prior and loads.

## 10. Final (245,032,457 tokens; seeds 100–103; VAL opened once)

**Positive result iff all of:**
- **F1:** I_SU − I_SH hierarchical CI lower bound > 0 and ≥ 3/4 seeds positive.
- **F2:** student CE_SU − CE_G hierarchical CI upper bound < 0.002 (non-inferior to the generic residual). Better than TOKEN-4 or EDL is reported, not required.
- **F3:** ≥ 4 semantic groups whose own specialist has pooled I_k CI lower bound > 0, and the largest positive share ≤ 0.35.
- **F4:** integrity (no pathology for SU; base hashes; tests).

**Outcome labels:**
- POSITIVE (all four);
- SPECIALISED_BUT_COSTLY (F1, F3, not F2);
- ACCURACY_WITHOUT_SEMANTIC_FUNCTION (F2, not F1);
- FUNCTION_CONCENTRATED (F1, not F3);
- NEGATIVE (otherwise).

## 11. Interpretation rules

- Higher NMI without functional gain is a repeat of the Semantic-Lift / JSR pattern, not success.
- Teacher works but student fails: the router is the bottleneck.
- SU ≈ SH in function: semantic information adds nothing beyond utility and the persistence structure.
- SU ≈ SO: utility adds nothing beyond the semantic prior.
- SU ≈ G: neither adds anything beyond generic capacity.

## 12. Engineering generations (at most 3; each tied to a measured failure; logged in DECISION_LOG before running)

A generation applies exactly one preregistered change to the frozen configuration and reruns Stage 2 (seeds 100/101, EVAL). If Stage 1 itself failed (no gated candidate), the change is applied to every candidate and Stage 1 is rerun under the same gates and selection rule. There is no search, so EVAL is re-read at most 3 times (disclosed); VAL is untouched.

**Triggers and changes, applied in this precedence:**
1. **Teacher collapse** (P4 or a Stage-1 pathology gate fails on the teacher r loads): λ × 3 (0.3 → 0.9; then 2.7). If only the student arg-max loads fail while the teacher loads pass, trigger 5 applies instead.
2. **Noisy utility** (P1 fails and the median online split-half reliability of SU over Stage-2 steps 600–793 is < 0.05): aggregate the partner utility over all other views of the same document (leave-own-view-out mean; cross-fit preserved).
   If reliability is ≥ 0.05 but P1 fails with |agree_util − agree_prior| not indicating dominance: c × 2 (softer).
3. **Semantic prior dominates** (P1 fails; SU teacher median agree_prior > 0.9 and agree_util < 0.15): η × 0.5.
4. **Utility dominates and loses subject structure** (P1 fails; SU teacher agree_util > 0.8 and teacher NMI_sem(SU) − NMI_sem(UO) < 0.01): η × 2 (max 2.0).
5. **Teacher works, student fails** (P2 fails with teacher advantage > 0 and P1 passes; or student-only pathology): the router state input becomes [mean, last token] of the view through Linear(768→20) → GRUCell(20,20) → Linear(20→16) (18,216 params ≤ cap), identically for all util banks.

**Stopping:** if no trigger applies, or after 3 generations, Experiment A is declared failed at Stage 2 and Experiment B runs.

**Never allowed:**
- changing data, splits, thresholds or the objective backbone;
- giving a candidate more tokens than its controls;
- using real domain labels;
- continuing to 245M without passing Stage 2.

## 13. Experiment B trigger

If Experiment A fails its Stage-2 promotion (after any generations), Experiment B (cfar1, centered personalised FedAvg residuals) runs under its own preregistration, written and hashed before its first scientific job. It reuses this infrastructure, data, K, the residual budget (±0.5%), the semantic assignments, the causal router, seeds and controls.

## 14. Compute, disk, operations

- **GPU cap (Experiment A):** 32 GPU-h.
- **Measured** (`gpu_tests_20261121T100639Z.json`):
  - 5 banks: 2.70 s/step (0.59 h per 50M run; 2.91 h per 245M run);
  - Stage-1 15 banks: 9.68 s/step (0.85 h);
  - no banks: 0.46 s/step;
  - peak memory 17.3 GiB.
- **Plan:**
  - tests ≈ 0.6 h;
  - Stage 1 ≈ 1.4 h (including the SEL eval);
  - Stage 2 ≈ 1.7 h;
  - final ≈ 14.5 h (4 × 2.9 h of training, plus VAL and reference evals);
  - up to 3 generations × 1.7 h.
- **Disk:** ≥ 100 GiB free before every job, plus that job's peak.
- **Retention:** ≤ 2 verified resumable slots per run. Superseded raw eval arrays of this experiment may be deleted after their analysis JSON exists (sha256 logged in deletions.tsv). Previous experiments are never modified.
- No /dev/shm. Atomic, verified checkpoints at evaluation boundaries and at least every 76 steps or 10 minutes.
- **Operator interruption** (signal, STOP marker, foreign GPU process):
  - checkpoint once;
  - set state STOPPED_BY_OPERATOR, with the resume command in RESUME.md;
  - stop all launchers;
  - never auto-restart.

## 15. Logs

All under `artifacts/csr-crossfit-semantic-utility-residual-v1/`:
- OBSERVATIONS.md, DECISION_LOG.jsonl, RUN_LOG.jsonl, deviations.jsonl;
- run_manifest.tsv, compute_ledger.tsv, disk_ledger.tsv, deletions.tsv, checkpoint_ledger.tsv;
- frozen configs with hashes (configs/csr-crossfit-semantic-utility-residual-v1/);
- stage analyses (stage1.json, stage2.json, final_analysis.json);
- RESUME.md and STATE.json.
