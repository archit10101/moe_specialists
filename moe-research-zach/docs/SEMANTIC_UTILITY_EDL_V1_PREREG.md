# csr-semantic-utility-edl-v1 — preregistration

Prefix `sue_`. Artifacts `artifacts/semantic-utility-edl-v1/`, configs `configs/semantic-utility-edl-v1/`, temporary files
`.tmp/semantic-utility-edl-v1/`, runs `runs/<UTC>_semantic_utility_edl_v1_<name>`. Written before any candidate result.
Its sha256 is stored in `docs/SEMANTIC_UTILITY_EDL_V1_PREREG.md.sha256`, and every script checks it.

## 1. Question

Can label-free semantic paper clusters outperform matched random persistent paper groupings when the auxiliary
objective rewards cluster-specific expert *usefulness*? Usefulness here means counterfactual expert usefulness, not
merely different average routing distributions. Unrestricted token-level top-4 routing and competitive LM CE must
be preserved.

## 2. Motivation (recorded before results)

- Semantic EDL (semantic-edl-v1, semantic-edl-long-k) changed routing and recovered much of explicit EDL's routing
  specialisation.
- A shuffled but paper-consistent grouping recovered roughly 80% of that effect (semantic-edl-v1 post-hoc analysis;
  semantic-edl-long-k Phase 1).
- The Semantic-EDL objective (-log JSD between group-mean router distributions) rewards *different* group-level
  routing distributions, even if the selected experts are not especially useful for their groups.
- The missing link may therefore be functional alignment. An expert should be rewarded only when replacing it hurts
  its assigned semantic group more than matched other groups.

## 3. Fixed semantic grouping (frozen before any usefulness-candidate result)

- **Rule.** If semantic-edl-long-k Phase 2A produced a CALIB-SELECTION-confirmed winner (its
  `phase2/calibsel_2a.json` has `confirmed: true`), use that winner's K, assignment table and beta. Otherwise use the
  preregistered fallback: the K = 16 C6 table (`artifacts/semantic-edl-v1/tables/table_SEM_T05.npz`, sha
  `5d23129b…89d2`, sharpened T = 0.5), all 11 MoE layers, beta = 0.005.
- The shuffled control table is the persistent paper permutation of the same table (rng 20261117). For the fallback
  it is `table_SHUF_SEM_T05.npz` (sha `d374a4ec…69fe`). Its cluster masses, assignment entropy and confidence
  distribution and paper persistence are identical by construction.
- The resolution of this rule is appended in section 13 at freeze time (from Task-1 artifacts only), then hashed.
- Adaptive K is not run. Real five-domain labels are never used to choose K, tune the objective, select candidates
  or stop training. They are opened only for the final descriptive VAL evaluation after the candidate is frozen.
- Evaluation pseudo-labels: CALIB-SEARCH and CALIB-SELECTION documents get the argmax of their paper's q from the
  label-free calib tables of the same K (`artifacts/gc-edl-v1/latent/table_SEM_calib_{search,sel}.npz` for K16).

## 4. Architecture, fairness, training-only machinery

- **Model and training recipe.** The existing 103.6M LFR model is unchanged: unrestricted dropless per-token top-4
  over 32 experts, the unchanged shared expert, no pools, masks, slots, hard routing, extra encoders or
  inference-time labels. The recipe matches the reused TOKEN-4 / EDL / C6 / SHUF runs exactly: initialisation, data
  order (FINAL plan, 3,880 steps, 245,032,457 tokens), optimizer, WSD schedule, bf16, Switch 0.01, z 0.001,
  tokenizer, context 1024 and k = 4.
- **Search runs.** These stop at a milestone of the same plan, with the same schedule and the same data prefix.
- **Objective.** `L = L_LM + 0.01 L_Switch + 0.001 L_z + beta L_SemanticEDL + gamma(t) L_use`.
- **What the usefulness machinery is.** It adds no parameters and is absent at inference. The checkpoint holds the
  LFR state dict only; the target bank lives in resumable slots only.
- **Reuse.** The following are reused, not retrained:
  - TOKEN-4 s0-s2 and explicit-EDL s0-s1 (fle);
  - EDL s2, C6 s0-s2 and SHUF s0-s2 (semantic-edl-long-k Phase 1);
  - the frozen C6 s100 20M checkpoint (semantic-edl-v1 `s_C6_s100`, read-only), as the diagnostic anchor only.
- **Search-scale references (trained once here).** No compatible Semantic-EDL-only runs exist at the search seeds
  beyond 20M, and the protected semantic-edl-v1 run cannot be continued. So two references are trained in this
  namespace: REF-C6 (Semantic-EDL only, the frozen grouping and beta) and REF-SHUF (its shuffled table), seeds 100
  and 101, to 50M (milestones s20, s50). They are references only, never candidates.

## 5. Counterfactual usefulness target (src/csr/sue_cf.py)

- **Probe samples.** Single-document rows from the main-split documents that are *not* in the 245M training plan:
  373k documents, all with q, ≥ 256 tokens.
- **Disjoint samples.** These probe samples, used to estimate the targets, are disjoint from the optimisation
  batches. Held-out evidence uses CALIB-SEARCH, CALIB-SELECTION and VAL, which are paper-disjoint from main and from
  each other (verified: 0 shared papers).
- **Per-step estimation.** Every training step, before the update:
  - one no-grad base pass on `probe_rows = 16` rows records the base per-document CE and the top-4 sets;
  - then `probe_passes = 4` exact replacement passes run. Each uses one MoE layer `l` (rotating over the placement
    layers) and one expert `e_s` per row (the selected expert with the least support, seeded tie-break).
- **Replacement definition.** For the tokens of row `s` whose top-4 contains `e_s`:
  - `e_s`'s logit is set to −inf;
  - the full softmax renormalises over the remaining experts;
  - top-4 admits the next-ranked expert (exactly four active experts);
  - all downstream computation is recomputed, restarting from the cached post-attention residual of layer `l`.

  Then `u = CE_replace − CE_base`, the per-document mean. Validated against the brute-force full `LFRModel.forward`
  with `exclude` + per-token `exclude_mask` (sue_tests t3: max |ΔCE| = 0.0 in fp64).
- **Identical procedure for both tables.** The probe documents, layer schedule, number of extra passes and coverage
  rule are identical for SEM and SHUF runs of the same seed and do not depend on the table.
- **Target bank.** It stores a lagged EMA (decay `rho` per step) of q-weighted sums by group k, layer, expert and
  difficulty bin. The bin is the base-CE quartile within the probe batch.
- **Generic-quality removal.** `ΔU_{k,l,e}` = own-group mean usefulness − the difficulty-matched (same bins, own-bin
  weights) mean usefulness on the other groups.
- **Support rules.**
  - A cell is valid only if both the own and the other q-mass are ≥ 2.0 and ≥ 8 (decayed) raw observations exist
    for (l, e).
  - Invalid cells are masked: excluded from `S` with P renormalised over valid cells, never treated as zero.
  - A (k, l) loss term needs ≥ 2 groups with ≥ 50% valid experts at that layer.
- **Shrinkage and standardisation.** Per-cell standard errors come from weighted variances and effective counts.
  Empirical-Bayes shrinkage per layer:
  - `Z = sqrt(s²) ΔU / (s² + se²)`, with `s² = max(0, var(ΔU) − mean se²)`;
  - `Z = 0` if a layer shows no signal above noise;
  - clipping at ±3.

  Targets are numpy constants (stop-gradient), refreshed every 10 steps from observations up to the previous step
  (lagged).
- **Logging and coverage.** Logged at every refresh: split-half reliability (document halves by hash),
  Spearman-Brown, coverage, signal and noise variance by layer, and the support distribution. Coverage is
  balanced by least-support expert choice and is reported, never imputed.
- **Usefulness loss.** `P_{k,l}` is the Semantic-EDL group-mean router distribution of the training micro-batch
  (gc_edl_loss definition).
  - `S_{k,j,l} = Σ_e P_{k,l,e} Z_{j,l,e}` over valid cells;
  - `L_use = mean_{k,l} −log softmax_j(S_{k,·,l}/τ)[k]`.
  - `gamma(t) = 0` before `onset`, then a linear ramp over `ramp` steps.
  - L_use never optimises its own targets: targets are detached, lagged and computed on other samples.

## 6. Pre-training reliability diagnostic (recorded; never a stopping rule by itself)

- **Checkpoints.** `sue_diag.py` on frozen checkpoints:
  - `anchor` = C6 s100 at 20M;
  - `c6_50m` = P1_C6_s0 at 50M;
  - `shuf_50m` = P1_SHUF_s0 at 50M.
- **Procedure.** 1,000 probe steps, the exact training procedure with no decay, and the same observations
  aggregated with the SEM and SHUF tables.
- **Reported.** Split-half reliability at 80/160/318/640/1000 steps, support, signal vs noise, the SEM-vs-SHUF
  own-group margin with half-to-half spread, and the numerical noise floor of unmodified rows.
- **gamma anchor (fixed rule).** On the anchor checkpoint with its 1,000-step SEM bank and τ = 0.2:
  `gamma_base = 0.5 · ||∇ beta L_SemEDL|| / ||∇ L_use||`, rounded to one significant digit. If `||∇ L_use|| = 0`
  (no signal survives shrinkage), `gamma_base = 0.01`.

- **GPU smoke run (not a candidate).** 30 steps of the real model with the utility machinery (onset 10,
  refresh 10). It checks the compiled-training and eager-probe interplay in bf16, the activation of the usefulness
  term and the per-step overhead. It is recorded under the diagnostic budget; its slot is then dropped.

## 7. Compact search (CALIB-SEARCH only; seed 100; 20M tokens = 318 steps)

- **Base candidate U1.** `gamma_base`, τ 0.2, rho 0.995, onset 100 (ramp 50), all 11 layers.
- **Grid (8 candidates).** Each of U2-U8 changes one factor relative to U1:

  | candidate | change |
  |---|---|
  | U1 | base |
  | U2 | gamma × 0.3 |
  | U3 | gamma × 3 |
  | U4 | τ 0.05 |
  | U5 | τ 1.0 |
  | U6 | rho 0.98 |
  | U7 | onset 30 |
  | U8 | upper placement: MoE layers 5-10 for both estimation and loss |
- **Evaluation.** `dev` at s20 for each candidate and for the reference REF-C6 s100 s20. Expert subjects are discovered
  on CALIB-SEARCH paper-half A and measured on half B (paper-disjoint), with label-free rows.
- **Promotion rule.** Promote ≤ 2 candidates. A candidate is eligible only if all of the following hold:
  - pseudo per-expert interaction > REF-C6 s100 s20;
  - dev CE − reference CE ≤ +0.010;
  - no pathology: load min ≥ 0.1× uniform, max ≤ 4×, normalised routing entropy ≥ 0.45;
  - the L_use term is active (> 0 active terms) in ≥ 50% of post-onset logged steps.

  Rank by interaction gain. Ties within 2e-5 go to lower CE. If none is eligible, apply section 9 with the measured
  failure.

## 8. Promotion to 50M (seeds 100 and 101, matched controls) and freeze

- **Runs.** Each promoted candidate: SEM-UTILITY and SHUF-UTILITY on seeds 100 and 101, to 50M tokens (793 steps).
  The seed-100 SEM run continues its 20M search run. SEM and SHUF are identical except the table.
- **Evaluation.** `dev` at s50.
- **Candidate passes iff** all of:
  - interaction_pseudo(SEM) − interaction_pseudo(SHUF) > 0 in both seeds;
  - mean ΔCE(SEM − SHUF) ≤ +0.005;
  - no pathology in either run.
- **Freeze.** The frozen candidate is the passer with the largest mean interaction difference. It is written to
  `frozen.json` before CALIB-SELECTION is opened. If no candidate passes, apply section 9.

## 9. Engineering generations (≤ 2; each written to docs/SUE_GEN<n>.md and hashed before running)

- **Scope of each generation.** It changes one family, has ≤ 2 candidates, and is run directly at 50M on seeds
  100/101 with SHUF controls. It is judged by the section-8 rule.
- **Diagnosis order.** The measured failure is diagnosed in this order (first matching):
  1. **Collapse or starvation (pathology) → reject.** No generation.
  2. **Noisy utility.** The SEM bank's split-half r < 0.2 at 50M in either seed. Increase support/EMA, not
     strength: probe_passes 8 and rho 0.998.
  3. **Reliable signal but no routing response.** The L_use own-group margin at the end is not above its value at
     onset, or the pseudo routing MI of SEM-UTILITY is ≤ REF-C6 at the same seed and tokens. Adjust strength and
     temperature: (gamma × 3, τ/2) and (gamma × 10, τ).
  4. **Functional gain with CE harm.** The interaction difference is > 0 in both seeds but mean ΔCE > 0.005. Use
     later onset or lower strength: (onset × 2, gamma × 0.5).
  5. **Routing response without functional gain.** The interaction difference is ≤ 0 in some seed while routing MI
     rose. **Reject** the proposed loss; no generation.
  6. **SEM ≈ SHUF with reliable targets and a routing response.** Conclude that the semantic clusters add no
     usable functional structure under this objective. No generation.
- **Termination.** If two generations fail, stop the search. No seed hunting, no validation or test, no real
  domains.

## 10. CALIB-SELECTION confirmation (opened once, for the frozen candidate only)

- **Measurement.** `confsel` at s50 for SEM and SHUF, seeds 100 and 101, on CALIB-SELECTION paper-half 0.
  Discovery is on CALIB-SEARCH half A.
- **Confirmed iff** all of:
  - pseudo interaction SEM > SHUF in both seeds;
  - mean ΔCE ≤ +0.005;
  - the pooled paired document-bootstrap 95% interval of the seed-mean difference has a lower bound > 0. Both runs
    use identical documents and bootstrap weights.

  Finals are trained only if confirmed.

## 11. Finals (only if confirmed): 245M, seeds 0, 1, 2

- **Runs.** SEM-UTILITY and SHUF-UTILITY, seeds 0, 1, 2. Reused: TOKEN-4 s0-2, EDL s0-2, C6 s0-2 and SHUF-C6 s0-2.
- **Compute-matched TOKEN-4 (secondary only).** Run it if the counterfactual estimation adds > 5% step time: TOKEN-4
  on `245M × (1 + overhead)` tokens, same recipe, seeds 0-2 within the budget (else seed 0). It never replaces the
  token-matched primary.
- **Evaluation.** `final` (opened after all final training):
  - CALIB-SELECTION paper-half 1 (never used before) pseudo metrics, for all eight conditions;
  - VAL CE / routing / real-domain and e5 per-expert interaction / throughput for the new runs. The reused runs'
    VAL numbers come from semantic-edl-long-k evals (identical calls).
- **Success (all required):**
  - **S1.** interaction_pseudo(SEM-U) > interaction_pseudo(SHUF-U) on CALIB-SELECTION half 1 in every final seed.
  - **S2.** The pooled paired document-bootstrap 95% interval of the seed-mean difference excludes 0. The seed-mean
    difference also exceeds the reliability noise, defined as the between-seed SD of SHUF-U's interaction.
  - **S3.** No collapse or starvation in any SEM-U seed (section-7 pathology bounds on VAL).
  - **S4.** VAL CE: mean(SEM-U − TOKEN-4) ≤ +0.010 and mean(SEM-U − EDL) ≤ +0.010 (seeds 0-2).
  - **S5.** Not generic importance: the SEM-U − SHUF-U interaction difference stays > 0 (seed mean) after
    load-quintile matching. Correlation of interaction with generic importance is reported.
- **Descriptive only (never used for selection).** Real-domain functional interaction, domain NMI and MI, probes,
  topic entropy, per-domain CE, e5 system, trajectories (20M, 50M, 40/60/80/100%), throughput and latency.

## 12. Budget, integrity, interruption

- **GPU caps.** Total 45 GPU-h: diagnostic ≤ 2, search (sections 7-10) ≤ 14, finals ≤ 29. Search compute and
  final compute are reported separately.
- **Execution and storage.**
  - One GPU job at a time via `scripts/gpu_guarded.sh` and `sue_driver.py`.
  - Guard disk reserve 100 GiB.
  - Never /dev/shm; never the locked test split.
  - A protected-file inventory is taken at the start and verified at the end.
  - Task-1 (semantic-edl-long-k) artifacts are read-only.
- **External interruption.** A foreign GPU process, signal, STOP marker or scheduler means:
  1. stop launching;
  2. checkpoint at the next atomic boundary (model, optimizer, RNG, data cursor, target bank, driver state);
  3. mark INCOMPLETE_STOPPED_BY_OPERATOR;
  4. write RESUME.md with one exact command;
  5. release the GPU;
  6. never restart automatically.
- **Logs.** DECISION_LOG.md, DEVIATIONS.jsonl, RUN_LOG.jsonl, RESULTS_LIVE.md, compute/checkpoint/run ledgers.

## 13. Resolution of the section-3 grouping rule (appended at freeze)

Resolved 2026-11-17T01:57:11-05:00 from read-only Task-1 artifacts (semantic-edl-long-k state STOPPED_BEFORE_PHASE2B_BY_USER), before any Task-2 candidate result:

```
{
 "source": "fallback: no CALIB-SELECTION-confirmed Phase-2A winner (calibsel_2a: winner K8, confirmed False)",
 "K": 16,
 "table": "artifacts/semantic-edl-v1/tables/table_SEM_T05.npz",
 "table_sha256": "5d23129b2cb7aed6a0d2ed768b1a755eac661dae95ddeace89329dc7b56b89d2",
 "shuf_table": "artifacts/semantic-edl-v1/tables/table_SHUF_SEM_T05.npz",
 "shuf_table_sha256": "d374a4ec45db7632ffd865b6101b51cbcaeab35fbb1575b140cb92b5e35c69fe",
 "beta": 0.005,
 "T": 0.5,
 "calib_search": "artifacts/gc-edl-v1/latent/table_SEM_calib_search.npz",
 "calib_search_sha256": "abdd94c5112810df6705aa67a99be915d8223366a57d3b216ae4927978b5e69d",
 "calib_sel": "artifacts/gc-edl-v1/latent/table_SEM_calib_sel.npz",
 "calib_sel_sha256": "2e4c4e747804eae99c606c45612197a16428ddba89bddf84d8f5e9c26c20524a",
 "layers": "all 11 MoE layers",
 "slk_state": "STOPPED_BEFORE_PHASE2B_BY_USER"
}
```
