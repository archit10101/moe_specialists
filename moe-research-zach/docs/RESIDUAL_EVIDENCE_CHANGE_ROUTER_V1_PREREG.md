# residual-evidence-change-router-v1 (recr1) — preregistration

Frozen before any CSEARCH or CSELECT result exists. The sha256 of this file is in `docs/RESIDUAL_EVIDENCE_CHANGE_ROUTER_V1_PREREG.md.sha256`; code hashes at freeze are in `artifacts/residual-evidence-change-router-v1/code_manifest_prereg.json`.

## 1. Question

Can a causal router detect when its active residual specialist stops being useful, search quickly for a better one, and so improve first-32-token and post-topic-change routing? The constraints are: no explicit domain labels, no future-token leakage, and no substantial inference overhead.

## 2. Audit and scope (details in OBSERVATIONS.md, AUDIT 1–4)

**Substrate:**
- the spb1 generation-0 banks: SEM2 (K16 × M2 = 32 semantic residual specialists), SHUF2 and GEN32;
- these sit after the final block of frozen TOKEN-4 245M ckpt_p100, seeds 0/1/2;
- the current unrestricted causal routers are spb1 S2_g1: R_SEM, R_SHUF, R_GEN and R_COARSE16, router seeds 0/1 (bal 0.001);
- the stored semantic oracle is the SEM2 training path (paper cluster plus pair subrouter, `e_tr`).

**Blocker.** The three frozen base checkpoints were deleted in the disk cleanup, and no copy exists. Nothing is substituted. Specifically:
- **No new base passes:** no base hidden states, no new CE tables (CALIB-SELECTION, VALIDATION), no 245M confirmation, and no third router seed.
- **Seed-3 base:** used for latency timing only. It has the same architecture and is never used for any CE or routing result.

**Exactness.** The residual is the last operation before `norm_f` and the LM head of a frozen base. Therefore the CE of any routing policy at token t equals the retained per-token table entry of the specialist it executes. RECR is simulated exactly from the spb1 tables.

**Data.** Routes exist only on spb1 SEL (CALIB-SEARCH half A: 503 papers, 6,403,657 tokens). It is split by paper with `sha256(salt + paper) mod 10` into three partitions (`data_manifest.json`, 0400, sha in `data_manifest.json.sha256`):

| partition | papers | tokens | role |
|---|---|---|---|
| UTRAIN | 187 | 2,684,866 | utility-target and head training, κ calibration (label-free) |
| CSEARCH | 177 | 2,154,321 | engineering |
| CSELECT | 139 | 1,564,470 | opened once for the frozen candidate |

- The orv1 CALIB-SELECTION and VALIDATION partitions are never opened; they cannot be tabulated without the base anyway.
- The test split is never opened.
- Real-domain (M2D2) labels are not used anywhere.
- The region-prior labels (csur1 label-free semantic posterior) and paper semantic labels are used only for evaluation masks and non-deployable diagnostics.

**Prior evidence, verified:**
- spb1 intrinsic SEM−SHUF ΔI is +0.00495.
- Deployed SEM−GEN is −0.00047 overall, −0.00075 stable, +0.00086 in the first 32 tokens, and −0.00002 at transitions.
- The 0.51 / 0.35 / 0.90 subject accuracies and r = 0.975 quoted in the request are scur1 numbers (K=5 bank on another base), not spb1 numbers.

**Pre-freeze looks:**
- a pilot on spb1 TRAIN (half B) tables, never used later. Per-token own-pair advantage is −0.018 (sd 0.106) at positions 0–31 and −0.0026 (sd 0.054) at ≥128, with lag-1 autocorrelation 0.004;
- a code smoke test on 48 UTRAIN papers with tiny heads, whose output is meaningless.

## 3. Method (generation 0)

**Options per token:**
- the 32 bank specialists;
- S, the existing unrestricted token router's choice;
- G, the generic path (GEN32 under R_GEN);
- base (no residual).

**Evidence.** For the executed residual a at token t, the realised evidence is g_t = −d_t,a = log p_a(x_{t+1}) − log p_0(x_{t+1}). It is revealed after the decision at t and can affect routing only at t+1 or later. This is enforced and tested (T1).

**Change statistic (stable mode):**
- C ← max(0, γ·C + τ − g̃), where g̃ is g under evidence normalisation `norm` (gen 0: raw nats).
- An alarm fires when C > κ.
- κ is label-free: it is calibrated per bank seed, router seed and bank on UTRAIN so that the CUSUM on the S path's evidence (reset after each alarm) alarms `kappa_rate` = 4 times per 1,000 tokens. This is done separately for every-token evidence and for strided evidence.

**RECR-E (search):**
- **Document start:** search mode from t = 0.
- **On an alarm at t:** search starts at t+1.
- **During B search tokens:**
  - the prediction uses the fallback path (gen 0: G);
  - two shadow candidates are executed, the head's top two by cumulative predicted advantage over G, among {32 specialists, S}, re-ranked causally at each search token;
  - their exact evidence relative to the executed fallback (δ = d_fallback − d_candidate) is accumulated once each token is revealed.
- **After B tokens:**
  - commit to the candidate with the best mean δ among candidates observed for ≥ B/2 tokens, if that mean exceeds the abstention threshold θ (gen 0: 0);
  - otherwise commit to the fallback.
- **Stable mode:** one residual is executed (the committed option: a specialist, S's token-level choice, or G). The CUSUM runs on its evidence.
- **Fallback = base (engineering option only):** no evidence is available, so a new search is probed after 64 tokens.

**Gen-0 parameters:**

| parameter | value |
|---|---|
| γ | 0.95 |
| τ | 0 |
| kappa_rate | 4 per 1k tokens |
| B | 8 |
| candidates | 2 |
| head | linear |
| norm | raw |
| θ | 0 |
| fallback | generic |
| E_cm stride | 32 |

**RECR-D (filtering, no search).** Same detector. After an alarm or at document start, the fallback is used for B tokens with no shadow execution. Then it commits to the head's best mean predicted advantage over those B tokens (if > θ), otherwise to the fallback.

**RECR-E_cm (compute-matched).** Identical to E, except that stable-mode evidence (one extra base LM head) is computed only on every 32nd position, with its own κ calibrated at the same alarm rate.

**Candidate head (proposal only):**
- **Output:** linear (gen 0) → 34 outputs, the predicted gain vs base of the 32 specialists, S and G.
- **Input (198 causal features from retained causal router outputs at positions ≤ t):**
  - S-router choice one-hot and its causal EMA-16, EMA-128 and document-prefix histograms;
  - the same four for R_COARSE16;
  - S p1 and margin, R_GEN p1, R_COARSE16 p1;
  - log position and a bias.
- **Excluded:** hidden states (unavailable), future tokens, realised next tokens, labels, paper IDs, validation data and oracle routes.
- **Targets:** exact frozen-bank gains −d on UTRAIN.
- **Loss:** Huber (δ 0.25 nats).
- **Training:** AdamW, LR 3e-3 cosine, batch 8,192, **50M token presentations** (≈18.6 epochs of UTRAIN).
  - Seeds: 1000·bank_seed + 10·router_seed.
  - Atomic checkpoints every 2.5M tokens or 10 minutes, each verified on write.
  - Resume is bitwise (T4).
- **Cross-fitting:** two paper-fold heads give out-of-fold diagnostics. UTRAIN papers never appear in CSEARCH or CSELECT, so targets and head evaluation never share papers.

**Seeds.** Three fixed seeds (bank/base seeds 0/1/2), each with spb1 router seeds 0 and 1: 6 combinations per condition, all reported. A third router seed cannot be produced (deviation 0).

## 4. Conditions

All conditions are evaluated on the same partition tokens, with the same base, banks, tables and router outputs.

| id | condition |
|---|---|
| A | G: matched generic residual bank under its unrestricted router R_GEN |
| B | S: the existing unrestricted semantic router R_SEM over SEM2 |
| C | confidence gate: S if R_SEM p1 ≥ θ_C, else G. θ_C ∈ {0.3, 0.5, 0.7, 0.9} is chosen once on CSEARCH (gen 0) by lowest mean CE, then frozen |
| D | RECR-D (residual-evidence filtering, no search) |
| E | RECR-E (top-2 search, B = 8 in gen 0) |
| E_cm | compute-matched RECR-E |
| F | semantic oracle: SEM2 training path (paper cluster + pair subrouter). Upper bound only |
| H | position fallback (extra control): G for positions < 32, then S |
| Ugreedy | head-greedy (extra control): per token, the head's best option if its predicted advantage > θ, else G (no evidence, no search) |
| SHUF twins | E, D, E_cm and S on SHUF2 (R_SHUF features and paths), for exposure-matched function |

**Matching:**
- RECR, C, H and Ugreedy store the SEM32 and GEN32 banks (2 × 6.3M residual parameters).
- A and B store one bank; they are the request's reference controls.
- Active residuals per token are 1, except during search (3).
- An unrestricted 64-way router over both banks would be the ideal parameter-matched control. It cannot be trained without the base hidden states, so it is a limitation.
- Compute is reported as analytic MACs per token: trunk 28.45M, LM head 19.32M, residual 0.197M, router 0.076M. The reference is the S path. RECR also runs R_GEN and R_COARSE16 every token, plus the head.

## 5. Metrics (per bank seed × router seed and pooled; paired per-paper ratio bootstrap, 2,000 replicates)

- **CE vs base by mask:** all; positions 0–7, 0–31, 32–127 and ≥128; transition first 8 and first 32 (region-prior label change at a 256-token region start, `pos%256 < k`); stable context (same label as the previous region).
- **Oracle recovery:** R = (CE_G − CE_X)/(CE_G − CE_F), overall and by mask.
- **Change detection (alarms vs annotations):**
  - semantic changes: region-prior label changes;
  - functional changes: the hindsight best bank specialist of the region differs from the previous region's, defined on bank seed 0's table for pooled masks;
  - recall at 32, 64 and 256 tokens; median delay; precision within 64; false alarms per 1k; chance recall for random alarms at the same rate;
  - recall and CE in semantic-only and functional-only (disagreement) changes.
- **Candidate recall at commits:**
  - the hindsight best of {32, S} over the next 64 tokens in the top 2;
  - the paper's own pair in the top 2.
- **Head calibration:** correlation, slope and a reliability table of predicted vs realised advantage of its top option.
- **Evidence reliability:** correlation and sign agreement between the search-window evidence difference of the two candidates and their realised difference over the next 64 tokens.
- **Functional specialisation:** deployed exposure-matched ΔI = I_SEM(policy) − I_SHUF(twin policy), with the spb1 estimator, strata and paper influence-function CI. Also reported: per-seed values and individually positive/significant specialists. NMI is never used as evidence.
- **Efficiency:**
  - residual experts executed per token; extra LM heads per token; MACs overhead vs S;
  - latency micro-benchmarks on GPU: prefill with and without one extra LM head and extra residuals on the seed-3 architecture, and decode-component timings. LFRModel has no KV cache;
  - simulation wall time.

## 6. Gates (all on CSELECT for the Stage-1 verdict; the same quantities on CSEARCH drive engineering)

For candidate X ∈ {E, E_cm}:

| gate | criterion |
|---|---|
| P1a | X − G < 0 overall with CI upper < 0 |
| P1b | X − S < 0 overall with CI upper < 0 |
| P2 | first 32 positions: X − G ≤ +0.001. Transition first 8: X − S < 0. Transition first 32: X − S < 0 with CI upper < 0 |
| P3 | stable retention (CE_G − CE_X)_stable / (CE_G − CE_S)_stable ≥ 0.75 |
| P4 | X − S < 0 in every bank seed (mean over router seeds) |
| P5 | deployed functional ΔI for E: CI lower > 0 and positive in every bank seed |
| P7 | E_cm analytic overhead ≤ 5% |

- **Stage-1 PASS (accuracy):** E passes P1–P5.
- **Compute-matched PASS:** E_cm passes P1–P5 and P7.
- **Mechanism supported:** PASS, plus E beats D and H with CI upper < 0, both at positions 0–31 and at transition first 32.
- **Oracle recovery:** R ≥ 0.5 is the preregistered ambitious preferred target. It is reported, not gated.

## 7. Engineering loop (CSEARCH only)

- **Limits:** at most 4 generations, one factor each, each motivated by a measured failure.
- **Before each generation:** the diagnosis, evidence, change, prediction and falsifier are written to DECISION_LOG and `gen<g>_change.json`.
- **Permitted factors:**
  - γ / τ / kappa_rate (decay and threshold);
  - B;
  - head capacity (linear / one hidden layer of 64);
  - evidence normalisation (raw / per-option z / clip ±0.2);
  - abstention threshold θ;
  - fallback (generic / base).
- **Forbidden:**
  - changing specialists;
  - future tokens in a current decision;
  - labels, paper IDs or filenames as features;
  - validation or test tuning;
  - adding or dropping seeds;
  - changing gates;
  - changing data;
  - weaker or compute-mismatched baselines.
- **Diagnostics for the failure modes (non-deployable, CSEARCH only):**
  - **O1 evidence oracle:** commit with the realised next-64-token value of the same candidates. Tests whether evidence is too noisy.
  - **O2 candidate oracle:** the candidates are the paper's own pair. Tests recall.
  - **O3 detection oracle:** alarms exactly at annotated changes. Tests slow detection.
  - **O1+2+3 combined.**
  - **O4:** the hindsight best fixed option over each region's first 32 tokens. Tests whether specialists retain value at transitions.
  - Causal unrecoverability is read from O123 vs E vs F. Overhead is read from the compute ledger.
- **Stopping:** stop earlier if a diagnosis shows that no permitted factor addresses the measured cause.
- **Freezing and CSELECT:**
  - after the loop, the candidate is frozen in `FROZEN.json`: the generation passing the most CSEARCH gates, with ties broken by the lowest E − S overall;
  - θ_C is frozen at its gen-0 value;
  - CSELECT is then evaluated exactly once, whatever happened, for an unbiased held-out estimate.

## 8. Stages

- **Stage 0:**
  - CPU tests T1–T8 (`cpu_tests_*.json`, all passed before freeze): policy causality under future perturbation (non-vacuous), feature causality, replay and per-document determinism, bitwise head-training resume, exactness and compute accounting, partition disjointness, κ calibration, label isolation;
  - a GPU latency micro-benchmark after freeze.
- **Stage 1:** heads (50M presentations, 3 bank seeds × 2 router seeds, both banks), the CSEARCH engineering loop, and CSELECT once.
- **Stage 2:** a longer confirmation (245M, VAL) is infeasible without the deleted bases. If Stage 1 passes, the state becomes `STAGE1_PASS_CONFIRMATION_BLOCKED` and a user decision is requested. Nothing is substituted.

## 9. Predictions (before any CSEARCH result)

- **Evidence is nearly uninformative at B = 8:**
  - commit evidence reliability corr < 0.1 and sign agreement < 0.56;
  - the O1 evidence oracle improves E by ≥ 0.0005 overall;
  - failure mode 1 is dominant.
- **Detection is slow:** semantic-change recall at 64 is within 1.5× chance (≈ 0.23 at 4 per 1k); failure mode 2.
- **Candidates:** own-pair-in-top-2 ≈ 0.4–0.6; failure mode 3 is secondary.
- **CE:**
  - E − S ≈ +0.0002 to +0.0006 overall (persistent commits lose token-level utility), so P1b fails;
  - E − G within ±0.0004;
  - first 32: E − G ≤ +0.001 (passes);
  - transition first 32: E − S ≥ 0 (fails);
  - stable retention < 0.75 (fails).
- **Controls:**
  - H beats S at positions 0–31 by ≈ 0.0009, and E does not beat H there;
  - C ≈ S ± 0.0001;
  - Ugreedy is worse than S.
- **Recovery and function:**
  - R(E) < 0.27, which is S's value on spb1 SEL;
  - functional ΔI stays positive (P5 passes).
- **Overhead:** E ≈ +42% MACs; E_cm ≈ 4–5%.
- **Expected verdict:** TERMINAL_STAGE1_FAIL, with failure modes 1 and 2 dominant and 5 (causal unrecoverability at boundaries) as the conclusion.

## 10. Interpretation tree

| observation | conclusion |
|---|---|
| E passes P1–P5 and beats D and H at boundaries | residual-evidence change routing works; confirmation blocked by the missing base |
| E passes, but does not beat H or D | gains come from generic fallback at boundaries, not from evidence or search |
| O1 ≫ E (evidence oracle helps a lot) | evidence too noisy (mode 1) |
| O3 ≫ E | detection too slow (mode 2) |
| O2 ≫ E | candidate recall inadequate (mode 3) |
| O4 small, or specialists ≤ G in transition windows even in hindsight | specialists do not retain value at transitions (mode 4) |
| O123 ≈ F, but causal variants ≪ O123 | oracle advantage is causally unrecoverable with this evidence (mode 5) |
| E_cm fails while E passes | search/evidence overhead too high (mode 6) |
| ΔI ≤ 0 | RECR routing destroys functional specialisation |

## 11. Budget and operations

- **Compute:**
  - CPU simulation and heads: about 1–2 CPU-hours per generation;
  - GPU: latency micro-benchmark only, ≤ 0.2 GPU-h; the cap is 24 GPU-h;
  - every GPU job goes through `scripts/gpu_guarded.sh`.
- **Disk:**
  - heads < 50 MB, analyses < 200 MB, no per-token outputs stored;
  - the 100 GiB reserve is checked before each job;
  - nothing outside `artifacts/residual-evidence-change-router-v1`, `configs/residual-evidence-change-router-v1` and `.tmp/residual-evidence-change-router-v1` is written;
  - no existing file is modified or deleted.
- **Interruption** (SIGINT/TERM/HUP, an external kill, a missing child, or another GPU user):
  - checkpoint if possible;
  - set state STOPPED_EXTERNAL;
  - release the GPU;
  - do not restart or reclaim;
  - write the exact resume command to RESUME.md.
