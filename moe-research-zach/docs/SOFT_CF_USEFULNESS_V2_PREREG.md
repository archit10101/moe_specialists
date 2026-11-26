# csr-soft-cf-usefulness-v2 — preregistration

Prefix `scu_`. Artifacts `artifacts/soft-cf-usefulness-v2/`, configs `configs/soft-cf-usefulness-v2/`, temporary
files `.tmp/soft-cf-usefulness-v2/`, runs `runs/<UTC>_soft_cf_usefulness_v2_<name>`. Written before any V2 result.
Its sha256 is stored in `docs/SOFT_CF_USEFULNESS_V2_PREREG.md.sha256`; every V2 script checks it. V1
(csr-semantic-utility-edl-v1) is terminal and is never modified; its artifacts are read-only inputs here.

## 1. Question

Can a **soft counterfactual-usefulness target** produce experts that are functionally more useful for their
label-free semantic paper groups than the identical method trained with shuffled persistent groupings? The target
rewards routing toward experts whose measured removal hurts a group more than difficulty-matched other groups. It
must keep CE comparable to or better than TOKEN-4 and explicit-label EDL.

## 2. Facts from V1 that motivate this design (measured; recorded before any V2 run)

- **V1 search.** Eight hard-margin usefulness candidates at 20M (seed 100): the dev interaction differences
  vs the Semantic-EDL reference were within ±6e-5, with a reference CI width of about 1.2e-4. The usefulness term
  became active only after about 200 steps (12.7M tokens), so at 20M it influenced only about 7.4M tokens.
- **V1 U2 at 50M.** SEM − SHUF interaction −2.6e-5 (s100) and −1.19e-4 (s101). The diagnosis was "noisy":
  split-half reliability of the in-training bank was −0.08 at 50M.
- **V1 generation G1a** (8 probe passes, rho 0.998). Dev 50M SEM − SHUF was +4.6e-5 and +1.0e-4, so it passed and
  was frozen. On held-out CALIB-SELECTION half 0 the sign reversed in both seeds (−3.8e-5, −4.9e-5; pooled
  paired CI [−8.7e-5, +0.9e-5]), and V1 ended not confirmed. With 8 passes, bank coverage reached 1.0, but
  in-training split-half r stayed about 0.05–0.23.
- **Routing moves without function.** Semantic grouping raised pseudo-routing MI over shuffled (≈ +0.02) in every
  V1 50M comparison without a functional advantage.
- **Design consequences.**
  - (a) Use only *confident* evidence: lower/upper confidence bounds, so noisy cells give no pressure.
  - (b) Use a *soft* target over all experts rather than V1's hard group-classification margin.
  - (c) Measure the mechanism chain so that a failure can be localised.

## 3. Fixed grouping (no choice made here)

- **Groups.** The V1 grouping: K = 16 C6 table `artifacts/semantic-edl-v1/tables/table_SEM_T05.npz` (sha
  `5d23129b…89d2`), label-free paper clusters sharpened at T = 0.5.
- **Shuffled control.** The persistent paper permutation `table_SHUF_SEM_T05.npz` (sha `d374a4ec…69fe`). Its group
  masses, assignment entropy and paper persistence are identical by construction.
- **No Semantic-EDL term.** beta = 0; the objective is section 5's. Groups enter only the training-only target and
  loss. No fixed expert is ever assigned to a group.
- **Real domain labels** are never used for training, selection, tuning or stopping. They are opened only for
  descriptive VAL interpretation after every decision of a stage is frozen.

## 4. Model, fairness, folds

- **Model and recipe.** The unchanged 103.6M LFR model: unrestricted dropless per-token top-4 over 32 experts, the
  unchanged shared expert, no pools, slots, masks, hard routing, extra encoders, drafts, oracles or inference-time
  labels. The recipe is identical to the reused TOKEN-4 / EDL / C6 / SHUF runs: init, FINAL plan (3,880 steps,
  245,032,457 tokens, sha f4bc3c93…), optimizer, WSD schedule, bf16, Switch 0.01, z 0.001, tokenizer, context 1024,
  k = 4. Search and promotion runs stop at a milestone of the same plan.
- **Training-only machinery.** The checkpoint contains the LFR state dict only, and inference never opens a table,
  bank or slot (CPU test tt4). With gamma = 0 and beta = 0 the trainer is bit-identical to TOKEN-4 (tt1).
- **Disjoint folds.**

  | fold | documents | use |
  |---|---|---|
  | measurement | main-split documents outside the 245M plan (probe pool, about 367k docs); probe rows of SEM and SHUF runs of a seed identical (tt5) | updates the usefulness bank |
  | training | the FINAL plan documents | trains router / model |
  | dev confirmation | CALIB-SEARCH paper-half B (discovery on half A) | search and promotion screening |
  | held-out confirmation 50M | CALIB-SELECTION paper-half 1, sub-half a | promotion criterion 4 |
  | held-out confirmation 245M | CALIB-SELECTION paper-half 1, sub-half b | final criterion |
  | VAL | — | CE (comparability) and descriptive real-domain interpretation only |

  - Splits main, CALIB-SEARCH, CALIB-SELECTION and VAL share 0 papers.
  - CALIB-SELECTION half 0 was used once by V1 and is never used here.
  - Sub-halves a/b: sha256(`scu-q|<paper>`) < 0.5 → a, else b (paper-disjoint).
- **Lagged, detached targets.** A document contribution is never measured and trained on in the same update.
  Targets are refreshed from observations up to the previous step and are numpy constants (stop-gradient).

## 5. Soft counterfactual-usefulness target and loss (src/csr/scu_cf.py)

- **Usefulness observations.** The V1 exact estimator is reused unchanged (src/csr/sue_cf.py; validated against
  brute-force `exclude` forwards, max |ΔCE| = 0.0 in fp64):
  - u(x,l,e) = CE_replace − CE_base;
  - remove e from the token's top-4, admit the next-ranked expert, renormalise, recompute downstream.
  - Each step: 16 probe rows, 8 replacement passes (layers rotating, least-support selected expert).
  - Bank decay rho 0.998 per step (the V1 G1a support setting, which gave full coverage).
- **Subject-specific usefulness.**
  - ΔU(k,l,e) = q-weighted own-group mean − difficulty-matched (base-CE quartile, own-bin weights) mean over
    the other groups, with a per-cell SE from weighted variances.
  - A cell is valid if own and other q-mass ≥ 2 and ≥ 8 decayed observations of (l,e) exist.
  - Absolute usefulness (own-group mean) and ΔU are both logged in the mechanism snapshots.
- **Confidence score.**
  - Per layer, scale_l = SD of ΔU over valid cells.
  - LCB = (ΔU − z·SE)/scale_l and UCB = (ΔU + z·SE)/scale_l, with z = 1.
  - **estimator "lcb":** s = LCB if LCB > 0, s = UCB if UCB < 0, else 0. Confident positive evidence raises an
    expert, confident negative evidence lowers it, and noise is neutral.
  - **estimator "mean":** s = ΔU/scale_l.
  - Invalid (unsupported) cells: s = 0, so no positive and no negative evidence.
  - |s| ≤ 3.
- **Soft target**, refreshed every 10 steps, lagged:
  - **form "tilt"** (main): q(k,l,·) = normalise(R(k,l,·) · exp(s(k,l,·)/τ)). R is the lagged prior: an EMA
    (0.98 per micro-batch, no grad) of the group-mean router distribution of group k at layer l. Without evidence
    q = R. The target only *tilts* the group's current routing toward confidently useful experts; it does not pull
    groups toward uniform routing.
  - **form "uniform"** (the operator's literal form): q = softmax_e(s/τ).
  - In both forms, (k,l) pairs without any non-zero score contribute no loss term.
- **Loss.**
  - L_CF = mean over (k with q-mass ≥ 0.5 in the micro-batch, l in all 11 MoE layers, evidence) of
    KL(stopgrad q(k,l,·) ‖ P(k,l,·)).
  - P is the q-weighted group mean of the per-sequence mean router softmax (the Semantic-EDL group-mean
    definition).
  - Total: **L = L_LM + 0.01 L_Switch + 0.001 L_z + gamma(t) · L_CF.**
  - gamma(t) = 0 before onset step 100, then a linear ramp over 50 steps.
  - All experts remain eligible, several groups may share an expert, and a group may use several experts.
    Token-level top-4 routing is unrestricted.
  - L_CF updates routing; the routed LM gradients train the chosen experts.
- **Mechanism logging** (diagnosis link D). Every 100 steps a snapshot `run_dir/mech/mech_s<step>.npz` holds:
  - R (router preference);
  - X (exposure: q-weighted fraction of group-k tokens with e in top-4, accumulated over the interval);
  - G (LM-gradient received: q-weighted squared expert-weight gradient norms of 16 single-document LM losses);
  - ΔU, SE, valid, s, q, evidence.
  - Refresh diagnostics are logged in metrics.jsonl: split-half reliability (document halves by hash),
    Spearman-Brown, coverage, evidence fraction, q entropy, KL(q‖R).

## 6. gamma anchor (fixed rule; recorded before the search)

- **Anchor checkpoint.** V1 REF_C6_s100 ckpt_s20 (read-only).
- **Inputs.** A 400-step no-decay probe bank (8 passes × 16 rows, s100 probe schedule). Prior R = mean group-mean
  routing of the anchor over plan steps 268–317.
- **Gradient norms.** On the rows of plan step 318, compute ‖∇ L_use‖ (V1 definition, τ 0.2, V1 targets) and
  ‖∇ L_CF‖ (tilt, lcb, τ 1) on backbone parameters.
- **Rule.** gamma_mid = 0.003 · ‖∇ L_use‖ / ‖∇ L_CF‖, rounded to one significant digit. 0.003 is the gamma of the
  only V1 candidate that passed the 50M rule (G1a), so this matches its gradient strength. If ‖∇ L_CF‖ = 0 (no
  evidence), gamma_mid = 0.01.
- **Recording.** `gamma.json` is written once (0400) before any candidate.

## 7. Compact search (20M, seed 100, SEM grouping, CALIB-SEARCH only)

| cand | form | estimator | gamma | τ | motivation |
|---|---|---|---|---|---|
| C1 | tilt | lcb | gamma_mid | 1.0 | main design |
| C2 | tilt | lcb | gamma_mid/3 | 1.0 | weaker strength (V1: the weaker gamma was the only eligible candidate) |
| C3 | tilt | lcb | 3·gamma_mid | 1.0 | stronger strength (does a larger routing response create function?) |
| C4 | tilt | lcb | gamma_mid | 0.5 | sharper tilt: confident evidence weighted e^(2s) |
| C5 | tilt | mean | gamma_mid | 1.0 | isolates estimator noise: same target without the confidence penalty |
| C6 | uniform | lcb | gamma_mid | 1.0 | the operator's literal target (uniform base measure) |

- **Fixed for all candidates.** onset 100, ramp 50, z 1, prior EMA 0.98, refresh 10, 16 rows × 8 passes,
  rho 0.998.
- **Evaluation.** `dev` at s20: CALIB-SEARCH half B, identical calls to V1's sue_evaluate dev mode.
- **Reused reference.** TOKEN-4 s100 at s20 (fle s_T4_s100 ckpt_s20, evaluated by V1 sue_post.py with the
  identical evaluator).
- **Eligibility** (all required):
  - pseudo interaction > TOKEN-4 s100 s20;
  - CE − TOKEN-4 ≤ +0.010;
  - no pathology: min load ≥ 0.1× uniform, max ≤ 4×, routing entropy ≥ 0.45;
  - L_CF active (> 0 terms) in ≥ 50% of logged steps ≥ onset.
- **Selection.** Promote at most **one**: the eligible candidate with the largest pseudo interaction (ties within
  2e-5 → lower CE). If none is eligible: no promotion; diagnose (section 10) on the search runs and stop.
- **No further V2 candidates.** No engineering generations exist in V2.

## 8. Promotion stage (50M = 793 steps, seeds 100 and 101, SEM vs SHUF)

- **Runs.** The promoted candidate is trained as SEM-V2 and SHUF-V2, identical except the table, on seeds 100 and
  101 to 50M. The seed-100 SEM run continues its 20M search run.
- **Reused (evaluated, not retrained).**
  - TOKEN-4 s100 (fle s_T4_s100) and s101 (semantic-edl-v1 b_T4_s101);
  - EDL s100 / s101 (semantic-edl-v1 b_EDL_s100 / s101);
  - Semantic-EDL REF_C6 and REF_SHUF s100 / s101 (V1).
- **Evaluations.** `dev` at s50 for SEM-V2 and SHUF-V2. `hold50` at s50 (CALIB-SELECTION half 1, sub-half a;
  opened once, after the dev stage of the promotion is complete) for SEM-V2 and SHUF-V2.
- **Promotion to 245M requires all of:**
  1. **CE:** mean over seeds of dev CE(SEM-V2) − CE(TOKEN-4) ≤ +0.010.
  2. **Functional specificity:** dev pseudo interaction SEM-V2 > SHUF-V2 (mean over seeds).
  3. **Sign:** the dev difference (2) is > 0 in both seeds.
  4. **Held-out:** on hold50, the pooled paired document-bootstrap 95% CI of the seed-mean SEM-V2 − SHUF-V2
     interaction difference has lower bound > 0. Both runs use identical documents and bootstrap weights.
  5. **No pathology** in any SEM-V2 or SHUF-V2 run (section-7 bounds, dev and hold50).
  6. **Estimator reliability:** the SEM-V2 bank's Spearman-Brown split-half reliability, averaged over the last
     5 refreshes before step 793, is ≥ 0.3 in both seeds.

## 9. Finals (only if section 8 passes): 245M, seeds 0, 1, 2

- **Seeds.** Three seeds, because TOKEN-4, EDL, C6 and SHUF-C6 are already available symmetrically for seeds 0–2.
- **Runs.** SEM-V2 and SHUF-V2.
- **Compute-matched TOKEN-4** (secondary only). If the probing adds > 5% step time: TOKEN-4 trained on
  245M × (1 + measured overhead) tokens, same recipe, seeds 0–2 within the cap (else seed 0).
- **Evaluation.** `final` (opened after all final training):
  - CALIB-SELECTION half 1 sub-half b pseudo metrics for every condition;
  - VAL CE / routing / real-domain and e5 per-expert interaction / throughput for new runs. Reused runs' VAL numbers
    come from the semantic-edl-long-k evaluations (identical calls).
- **Success** (all required):
  - **F1.** Pseudo interaction SEM-V2 > SHUF-V2 on sub-half b in every seed.
  - **F2.** The pooled paired bootstrap 95% CI of the seed-mean difference excludes 0, and the seed-mean
    difference exceeds the between-seed SD of SHUF-V2's interaction.
  - **F3.** No pathology.
  - **F4.** VAL CE: mean(SEM-V2 − TOKEN-4) ≤ +0.010 and mean(SEM-V2 − EDL) ≤ +0.010.
  - **F5.** The SEM-V2 − SHUF-V2 difference stays > 0 after load-quintile matching (not generic importance).

## 10. Diagnosis (computed for the promoted pair, or for the search runs if nothing is promoted)

The first matching link, each with a numerical rule. Inputs are mechanism snapshots, bank diagnostics and
evaluations.

| link | name | rule |
|---|---|---|
| A | Measurement | Spearman-Brown split-half reliability of the SEM bank, mean of the last 5 refreshes, < 0.3 |
| B | Target | not A, but the target is flat or generic: evidence covers < 10% of (k,l) pairs, or mean KL(q‖R) over evidence cells < 0.005 nats, or > 50% of the positive-score mass sits on experts with positive scores in ≥ 8 of 16 groups |
| C | Router | Spearman correlation across evidence cells between the score s (snapshot at the first post-onset diagnostic) and the change log R(end) − log R(that snapshot) is ≤ 0.05 |
| D | Expert learning | C passed (router moved), but the change in measured ΔU of the tilted cells does not follow; operationalised below |
| E | Transfer | dev SEM-V2 − SHUF-V2 > 0 in both seeds, but the hold50 CI lower bound ≤ 0 |
| F | Accuracy tradeoff | functional criteria 2–4 pass, but criterion 1 fails |

- **D operationalised.** The Spearman correlation between Δ exposure X (end vs first post-onset snapshot) and Δ
  ΔU(k,l,e) over the same interval is ≤ 0.05, or the dev interaction SEM-V2 ≤ REF-C6 at the same seed.
- **D chain reported per seed.**
  - corr(s, Δ log R);
  - corr(Δ log R, Δ X);
  - corr(X, G per unit exposure);
  - corr(G, Δ ΔU);
  - and the time series of these quantities.
- **SEM ≈ SHUF.** If A–F all pass at the measurement / mechanism level but SHUF-V2 shows the same changes, the
  conclusion is "the semantic clusters add no usable functional structure under this objective."

## 11. Budget, integrity, interruption

- **GPU caps.** Total 30 GPU-h: diagnostic/anchor 1, search 3, promotion 5, finals 21. Compute is reported per
  stage, including probe overhead.
- **Execution.**
  - One GPU job at a time via `scripts/gpu_guarded.sh` and `scu_driver.py`.
  - Disk: before every stage, estimate peak storage; stop cleanly if free − estimate < 100 GiB.
  - No deletion, move, compression or overwrite of existing files. Final resumable slots of finished runs are
    retained. Only the in-run rotation of superseded resume slots (one verified newer slot replaces the previous
    one, every ≤ 5M tokens) is used, as in V1.
  - Never /dev/shm; never the locked test split; no validation tuning.
  - A protected-file inventory is taken at the start and verified at the end.
- **External interruption** (signal, foreign GPU process, STOP marker, kill):
  1. checkpoint at the next step boundary;
  2. stop the driver and children;
  3. mark INCOMPLETE_STOPPED_BY_OPERATOR;
  4. write RESUME.md with one exact command;
  5. release the GPU;
  6. never restart automatically.
- **Logs.**
  - Append-only: DECISION_LOG.md, RUN_LOG.jsonl, DEVIATIONS.jsonl, compute_ledger.tsv, RESULTS_LIVE.md.
  - Regenerated: run_manifest.tsv, results JSON.
  - RESUME.md is updated after every completed job.

## 12. Conclusions allowed

1. **Replicated functional specialization with comparable/better CE:** F1–F5.
2. **Specialization improved but accuracy did not:** F1, F2, F5 pass; F4 fails.
3. **Accuracy improved without functional specialization:** CE ≤ TOKEN-4, but F1 or F2 fails.
4. **No replicated benefit.**
5. **Inconclusive due to estimator noise or insufficient power:** diagnosis A, or CIs that cannot exclude effects
   of the size of V1's dev effects.

## 13. Freeze

This document is hashed before any V2 GPU job. The gamma anchor (section 6) is computed by the first GPU job after the
freeze and written once to `artifacts/soft-cf-usefulness-v2/gamma.json` (0400) before any candidate is trained.
