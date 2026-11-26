# CSR-CURRENT-CONTEXT-SPECIALIST-ROUTER-V1 (ccsr1) — Preregistration

Status: **FROZEN** 2026-11-21, before any SEL or CALIB-SELECTION evaluation. The sha256 is in the `.sha256` file (mode 0400). Evidence summary: `docs/CSR_CURRENT_CONTEXT_SPECIALIST_ROUTER_V1_EVIDENCE.md`.

## 1. Question

Can a strictly causal router that sees the current token's context deploy the existing subject-specialised residual specialists accurately enough to match or beat the parameter- and FLOP-matched generic residual bank G?

This is a routing/deployment experiment. No specialisation loss is introduced, and the specialists are not modified in Stages 1–2.

## 2. Reused artifacts (read-only, sha256-verified on every job)

**Checkpoint:** `runs/20261121T151627Z_cfar1_S1_s100/ckpt_m20` (sha256 80e1325c…9f46).
- cfar1 Stage 1, seed 100, step 318, 20,057,391 tokens.
- Frozen TOKEN-4-replicate base (103,609,728 params, unrestricted top-4 MoE, unchanged) and banks.

**Banks:**
- **SEM = L:** local semantic bank. Hard argmax-p_sem groups; specialist k was trained on group-k views; no federation.
- **SHUF = F2h_T500_w240:** local shuffled bank with hard paper-transplant p_SH groups. It never synchronised (T = 500 > 318 steps). Same process, rows and specialist initialisation as L, except the 16 shared-only hidden units [240, 256) were averaged across specialists at init.
- **G:** the token-routed generic bank (own linear router, 6,144 params) trained on the same rows.

Each bank has 16 specialists 384→256→384 (3,145,728 params), applied as z = W(h + a_k(rms0(h))).

**Tables:**
- CALIB-SEARCH region priors `psem_calib_search` (c33a549d…) and `_sh` (a73337d9…) from csur1;
- orv1 `labels_cs`, `parts` and `token_feats`.

**Existing deployed baselines** (evaluated on identical tokens):
- G with its own router;
- the past-view student of L, i.e. cfar1's meanlast GRU over completed views.

No bank is retrained. The original initialisation and training-token pairing are preserved by construction, since all banks come from one process.

## 3. Data partitions (paper-disjoint; verified by test T9)

| name | source | use |
|---|---|---|
| TRAIN | CALIB-SEARCH half "EVAL" (498 papers, 6.10M tokens; never evaluated by csur1/cfar1) | router training only |
| SEL | CALIB-SEARCH half "SEL" (503 papers, 6.41M tokens) | Stage-1 gates and selection |
| CS2 | CALIB-SELECTION split `calib_sel` (1,057 papers, 12.5M tokens) | Stage-2 confirmation, opened only after `FROZEN_S2.json` |
| VAL | `val` | only if a Stage-3 joint continuation passes; one evaluation |
| test | — | never loaded |

- **CS2 preparation:** paper labels, rows and region priors are built only after the Stage-2 freeze. They use the identical frozen pipelines (orv1 paper assignment; csur1 region prior and transplant shuffle, re-applied by `ccsr1_cs2_prep.py`).
- That script must first reproduce the frozen CALIB-SEARCH tables: argmax agreement ≥ 0.999 and max |Δp| ≤ 2e-3.
- Real M2D2 domain labels are never used for training, tuning, selection or stopping. They appear only post hoc, after decisions.

## 4. Router (deployed) and causality

At token t of a document the router sees x_≤t only, through the frozen base:

- **Input:** feat_t = [rms0(h_t), the causal mean of rms0(h_s) over the last 16 tokens of the document (s ∈ (t−16, t])].
- **Network:** Linear(768→H, no bias) → GRU(H, H), which carries the previous router state → Linear(H→16).
- **Output:** q_t = softmax(logits). The router deploys arg-max q_t at **every token**, and the selected specialist's correction is applied to h_t to predict x_{t+1}.
- **Parameters:** H = 64 gives 75,152 params; H = 32 gives 31,440. Both are below the cap of 0.1% of the base (103,609), and identical across R1/R2/R3 at the same width.
- **Compute:** every condition, G included, runs exactly one specialist per token. The router adds about 150k FLOPs per token at H = 64 (about 0.16% of the model's active FLOPs), vs 12k for G's router.

**Causality requirements (tests T1–T5 and T10, all pass; `cpu_tests_20261121T164738Z.json`, rerun on the final code as `cpu_tests_20261121T171552Z.json`):**
- tokens after t cannot change q_≤t, including later tokens of the same region;
- sequential one-token routing equals batched routing;
- one-token decoding (base re-run on each prefix) chooses the same specialists as the batched path;
- utilities, region priors and targets do not influence the deployed choices;
- the table target at t is x_{t+1}.

## 5. Training targets (training only) and loss

**Utility:** u[t,k] = CE_base(t) − CE(base + residual_k)(t), exact, for all 16 specialists. It comes from the TRAIN table and uses the realised next token for supervision only. p = the bank's own region prior (p_sem for SEM, p_SH for SHUF) for the token's 256-token region.

**Targets:**
- **R1 (semantic teacher):** π = p.
- **R2 (exact utility teacher):** π = softmax(u/τ).
- **R3 (primary):** π = softmax(u/τ + η·log(p + 1e-4)).
- τ = c·s_u, where s_u is the median over TRAIN tokens of the per-token SD of u across specialists (bank-specific).

**Token weights:**
- R2 and R3: w_t = clip(m_t / median(m), 0.25, 4), where m_t = max_k u − mean_k u is the utility margin. No token is discarded.
- R1: w_t = 1.

**Loss:** L = Σ w_t KL(sg(π_t) || q_t) / Σ w_t + λ·Σ_k q̄_k log(K q̄_k). Here q̄ is the mean q over the batch of 64 documents (about 63k tokens from many papers), giving a corpus-level balance term.

**Optimisation:**
- AdamW (lr 3e-3, betas (0.9, 0.99), wd 0.01); 30-step warmup, then cosine decay to 10%; clip 1.0;
- 2,400 steps of 64 TRAIN documents (about 25 epochs);
- fixed; no early stopping; no evaluation data used.

**How the step count was chosen** (before the freeze, TRAIN data only; `gpu_tests_20261121T171447Z.json`): a held-out-paper pilot inside TRAIN. Routers were trained on 80% of TRAIN papers and scored on the other 17.4% of TRAIN tokens (SEM bank, seed 0).
- R3 bp: −0.0197, −0.0206, −0.0208, −0.0207 vs base at 600, 1,200, 2,400, 4,800 steps.
- R1 peaks at 1,200–2,400.
- The value fixes 2,400; this pilot is not used for anything else.
- GPU router training is bitwise reproducible (repeat-run choice agreement 1.0).

**Seeds:** router seeds 0, 1 and 2 set the initialisation and the document order. At a given width the same seed gives the same initialisation and order across configurations and banks.

**Frozen during router training:** the base and the specialists.

## 6. Stage-1 grid (SEL) — 12 R3 configurations plus controls

**Base point:** c = 1, η = 1, H = 64, λ = 0.01.

**R3 configurations** (`ccsr1_configs.py`):
- bp;
- one-factor changes: c ∈ {0.5, 2}, η ∈ {0.5, 2}, H = 32, λ = 0.1;
- fractional combinations: (c 0.5, η 2), (c 2, η 0.5), (c 2, η 2), (c 0.5, η 0.5), (η 2, λ 0.1).

**Controls** (H 64, λ 0.01):
- R1;
- R2 at c ∈ {0.5, 1, 2}.

**Scale:** every configuration is trained for SEM and for SHUF, at 3 seeds each (96 routers). τ: 3 values; η: 3; width: 2; balance: 2.

## 7. Metrics (every configuration and seed; SEL, later CS2)

**CE (exact table lookups):**
- base-only;
- G;
- the past-view student (PV);
- the current-region semantic teacher (argmax region prior);
- the per-token best-specialist oracle (min_k; a loose bound that includes noise selection);
- R1/R2/R3 students;
- regret vs the oracle and vs the teacher;
- teacher advantage retained.

**Routing behaviour:**
- loads (max, effective number, dead);
- route-switch frequency;
- agreement with the region-prior argmax and the utility argmax.

**Breakdowns:**
- post-switch accuracy: tokens in the first 64 positions of a region whose argmax p_sem differs from the previous region;
- position bins 0–31, 32–127, 128–511, 512+;
- per semantic group (paper-level C6 cluster).

**Functional specialisation (deployed choices, the csur1 estimator):**
- exposure-matched own-specialist advantage I (2,025 strata, cross-fitted paper halves, paper bootstrap);
- I_SEM − I_SHUF;
- per-specialist significance;
- swap summary;
- ordinary (unmatched) own-vs-other ablation alongside;
- routing NMI, post hoc only, never used for selection.

## 8. Gates

**G1–G6 apply per router seed** (except G2, which uses the mean). A configuration passes only if all six hold.

- **G1:** CE_student(SEM) − CE_G ≤ 0.002.
- **G2:** mean over seeds of CE_student(SEM) − CE_PV(SEM) ≤ −0.003.
- **G3:** retention = (CE_stu(SHUF) − CE_stu(SEM)) / (CE_T(SHUF) − CE_T(SEM)) ≥ 0.6. The teacher advantage must be > 0.
- **G4:** exposure-matched I_SEM − I_SHUF > 0.
- **G5:** no collapse of the SEM student's arg-max loads: max ≤ 0.25, effective number ≥ 8, ≤ 1 dead (< 1/160).
- **G6:** gains are not confined to the first 128 tokens or to one subject:
  - on positions ≥ 128, CE_stu − CE_G ≤ 0.002 and CE_stu − CE_PV < 0;
  - CE_stu − CE_PV < 0 in ≥ 12 of the 16 semantic groups;
  - no group supplies more than 50% of the total positive improvement over PV.

**Stage 1 (SEL):**
- **Selection:** the R3 configuration with the lowest mean student CE(SEM) among configurations passing G1–G6. Ties within 1e-5 go to bp, then to grid order.
- **Outcome:** Stage 1 passes iff at least one R3 configuration passes.
- **Controls:** R1/R2 are reported but not selectable.

**Stage 2 (CS2):**
- The frozen selected configuration's three Stage-1 routers (and the matched SHUF routers) are evaluated, without retraining.
- G1–G6 are applied on CS2. Stage 2 passes iff all hold.
- R1 and R2 (at the selected c) are also evaluated and reported.
- The **preferred success** additionally has CE_stu < CE_G in every seed. It is reported but is not a gate.

## 9. Engineering generations (at most 2; Stage-1 failures only; logged before running)

**Diagnostics** are computed on the reference configuration (the R3 configuration with the lowest mean student CE) and averaged over seeds. The first trigger that applies, in this precedence order, is used:

1. **E5 load collapse** (G5 fails): λ × 10, corpus-level balance only.
2. **E2 representation insufficient** (G1 fails and student agreement with the region-prior argmax < 0.35): a larger causal context summary. The input becomes [h_t, mean_16, mean_256] (H 64: 99,728 params ≤ cap).
3. **E3 delayed adaptation after switches** (G1 fails and the post-switch gap CE_stu − CE_T exceeds 2× the stable-region gap): a causal change feature. The input becomes [h_t, mean_16, mean_16 − mean over (t−64, t−16]].
4. **E4 harmful low-confidence routing** (≥ 10% of tokens have q_top < 0.25 and their chosen specialists raise CE on average): add a 17th zero-residual (base-only) action. It has u = 0 and prior 1/16 in R3, 0 in R1, and no expert parameters, and it applies to every condition.
5. **E1 semantic prediction adequate but utility regret high** (G1 fails and agreement ≥ 0.35): add an expected-utility term β·(−E_q[u]/s_u), with β = 1, to the R2 and R3 objectives.

**Rules:**
- Changes are cumulative across generations and applied to every configuration. Stage 1 is then rerun on the same partitions, seeds and gates.
- No generation may change the specialists, the semantic clustering, the data, the thresholds or the number of experts or parameters (beyond the zero action), or use domain labels.
- If no trigger applies, or after 2 generations, the experiment ends at Stage 1.
- Stage 2 is never followed by tuning.

## 10. Stage 3 — matched joint continuation (only if Stage 2 passes)

- **Arms:** continue G and the SEM candidate for the same additional 50M tokens (steps 318 → 1,111 of the TOKEN-4 plan) from the reused checkpoint. Rows, order, optimizer budget, base initialisation and active FLOPs are the same for every arm.
  - SEM is routed by the frozen successful router in one arm, and by a router updated by continued R3 distillation in a second arm.
  - SHUF continues with its own frozen router as the functional control.
- **Training:** specialists are updated by the LM loss only, with no auxiliary specialisation objective.
- **Stop rule:** stop if functional specialisation collapses (I_SEM − I_SHUF ≤ 0) or if CE_SEM − CE_G > 0.002 at a 1,000-step checkpoint. Checkpoints are atomic, at every stage boundary and at least every 1,000 steps.
- **Gate to 245M:** the continuation must improve or preserve both CE (vs G) and exposure-matched specialisation for at least two router seeds.
- **Before running:** the Stage-3 implementation is written as a hashed amendment and logged before its first job, before any Stage-3 result. These criteria cannot be changed.

## 11. Efficiency study (only after every-token routing is confirmed in Stage 2)

- Route refresh every 1, 4, 8 and 16 tokens, with the same specialist kept until the next refresh.
- Report CE, route agreement, and measured prefill and decoding speed and kernel behaviour. No speed claim is made from theoretical FLOPs.

## 12. Compute, disk, operations

- **GPU cap:** 16 GPU-h.
- **Measured:** TRAIN extraction 517 s; about 21 s per router at 2,400 steps.
- **Planned:** SEL extraction about 0.15 h; Stage-1 routers about 0.7 h (3 jobs); CS2 preparation and extraction about 0.6 h. Stage 3 is not planned unless reached.
- **Disk:** ≥ 100 GiB free plus each job's peak.
  - Tables: about 0.6 GB per half and 1.2 GB for CS2.
  - Router outputs: about 13 MB each.
  - Hidden states are recomputed in memory, not stored.
- **Rules:** no /dev/shm; only this experiment's scratch is cleaned, with every deletion logged; atomic outputs; the operator-interruption protocol applies (stop, mark STOPPED_BY_OPERATOR, never restart automatically).
- **Logs:** under `artifacts/csr-current-context-specialist-router-v1/`.
