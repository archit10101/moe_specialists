# CSR-CENTERED-FEDAVG-RESIDUAL-V1 (cfar1) — Preregistration (conditional follow-up to csur1)

Runs only if csur1 failed its frozen Stage-2 promotion. The trigger and csur1's outcome are recorded in DECISION_LOG before the freeze.

**Trigger record (written before the freeze, before any cfar1 scientific job).** csur1 ended TERMINAL_STAGE1_FAIL_AFTER_GEN1 (2026-11-21):
- no Stage-1 candidate passed the gates in generation 0 or in generation 1 (prereg §12 trigger 5, the [mean, last] router);
- every failure was the causal student: student CE_SU − CE_G = +0.0046 to +0.0071 (gate 0.002), and student argmax loads were concentrated;
- teacher loads were healthy, and semantic teachers beat G by up to 0.0054;
- the exposure-matched I_SU − I_SH was > 0 in 13 of 14 candidates;
- shuffled teachers were 0.006–0.009 worse than G.

So csur1 never reached Stage 2. Its failure to pass Stage 1 means it cannot be promoted, which is the §13 trigger for this experiment.

Status: **FROZEN** 2026-11-21, before any cfar1 scientific job. The sha256 is in the `.sha256` file (mode 0400), and every config embeds it.

## 1. Question

Can periodically transferring the common component of locally trained semantic-specialist updates into a shared residual improve accuracy while preserving private subject-specific deviations?

The relevant comparison is semantic FedAvg against parameter-matched shuffled, no-transfer and generic residual controls. Beating TOKEN-4 through added parameters does not count.

## 2. Reused from csur1 (unchanged code and tables)

- **Base:** the TOKEN-4 recipe and shared-base design. One process per seed; the base gets no bank gradient; rows, order and expert initialisation are identical across conditions.
- **Residual architecture:** 16 specialists 384→256→384 after the final norm, z = sg(W)(sg(h) + a), one specialist per token.
- **Data, seeds and budgets:**
  - Stage 1: seed 100, 20,057,391 tokens, SEL;
  - Stage 2: seeds 100 and 101, 50,047,429 tokens, EVAL;
  - final: seeds 100–103, 245,032,457 tokens, VAL opened once after `FROZEN_FINAL.json`;
  - the test split is never loaded.
- **Region priors:**
  - semantic p_sem;
  - persistent paper-transplant shuffled p_SH.
  - csur1 tables, hash-locked. VAL priors are built only after the cfar1 freeze, into `artifacts/csr-centered-fedavg-residual-v1/tables_val`.
- **Causal GRU student router:** the csur1 generation-1 'meanlast' router. The input is [mean, last token] of each completed view → Linear(768→20) → GRUCell(20,20) → Linear(20→16), 18,216 params; deployable arg-max; view 0 uses the zero state.
  - This choice was made before any cfar1 result.
  - Reason: in csur1 (within-process, all 7 candidates), the generation-1 router improved student CE relative to G by about 0.001 and reduced the argmax-load concentration.
  - The 19,152-param mean router is not used.
- **Evaluation and analysis estimators:** exposure-matched functional metric, swap matrix, per-specialist significance, routing, retention, position bins, paper and hierarchical bootstraps.

## 3. Federated banks (`src/csr/cfar1_bank.py`)

**Groups:** the group of view v (128 tokens) is the arg max of its region prior (semantic or shuffled). This is the "semantic assignment". Specialist k is the client of group k and trains locally on its views. The student is trained with KL(onehot(group) || q).

**Hidden units:**
- units [0, w_p) are personalised;
- units [w_p, 256) are shared-only.

**Modes:**
- **L (local):** no federation. Independent specialists, hard semantic groups; the no-transfer reference.
- **F1 (standard FedAvg, diagnostic):**
  - all specialists start identical;
  - every T steps the token-mass-weighted mean update since the last synchronisation is applied to the common anchor and broadcast;
  - all specialists are identical right after each synchronisation;
  - specialisation is expected to collapse, and this is measured explicitly.
- **F2 (centered personalised FedAvg):** residual_k = S + P_k in parameter space, with P_k = 0 on shared-only units. Every T steps:
  - D̄ = Σ_k w_k D_k, where w_k is the token mass of group k since the last synchronisation;
  - S ← S + D̄ (transfer of the common component);
  - personalised units: P_k ← P_k + D_k − D̄, so S + P_k is bitwise unchanged (function continuous on the personalised units; test C1);
  - shared-only units: every specialist ← S (FedAvg on the shared units).
- **Optimizer moments:**
  - personalised units keep their AdamW moments;
  - every reset slice gets the token-weighted mean of the specialists' moments.

**Parameter and FLOP accounting.**
- At width 256, a per-token active cost equal to csur1 (exactly one 384→256→384 path), with training parameters within 0.5% of csur1's 3,145,728, is satisfiable only at w_p = 240: S (196,608) + 16 × P_k on 240 units (2,949,120) = 3,145,728 exactly.
- Any larger shared capacity at fixed FLOPs necessarily lowers the parameter count, because sharing saves parameters. Larger shared capacity at fixed parameters raises FLOPs.
- **We keep FLOPs exactly matched to csur1 in every condition** and let the split lower the parameter count:
  - w_p = 240: 3,145,728 training params; deployable folded 2,961,408;
  - w_p = 192: 2,555,904;
  - w_p = 128: 1,769,472.
- Smaller splits therefore have fewer parameters than the controls (conservative). The semantic-vs-shuffled comparison is always parameter- and FLOP-matched.
- The infeasibility of "≤ 0.5% params and matched FLOPs with more sharing" is stated here, before any B result.

## 4. Conditions

- **Stage 1 (one process):**
  - G (generic token-routed bank, csur1/ORV1 C);
  - L;
  - F1s at each T;
  - F2s and F2h (semantic or shuffled groups) for every T ∈ {100, 250, 500} × w_p ∈ {240, 192, 128}.
  - At 20M tokens (318 steps), T = 500 never synchronises. Those arms are local training with the F2 parameterisation; this is disclosed.
- **Stage 2 and final:**
  - G;
  - F2s* and F2h* at the selected (T*, w_p*);
  - F1s at T*;
  - L.

## 5. Stage 1 selection (seed 100, SEL)

**Gates per (T, w_p):**
- student CE_F2s − CE_G ≤ 0.002;
- no pathology of the F2s student and teacher loads (max ≤ 0.25, effective number ≥ 8, ≤ 1 dead).

**Selection:**
- arg max of I_F2s − I_F2h (exposure-matched functional interaction);
- ties (< 1e-5) go to T = 250, w_p = 192.

Stage 1 fails only if nothing passes the gates.

## 6. Stage 2 promotion (seeds 100, 101; EVAL); all required

- **Q1:** I_F2s − I_F2h > 0 in both seeds.
- **Q2:** CE_F2s − CE_L < 0 in both seeds (transfer improves accuracy over no transfer).
- **Q3:** CE_F2s − CE_G ≤ 0.002 in each seed.
- **Q4:** no pathology for F2s (student and teacher loads).

## 7. Final (245M, 4 seeds, VAL); positive iff all

- **BF1:** I_F2s − I_F2h hierarchical CI lower bound > 0 and ≥ 3/4 seeds positive.
- **BF2:** CE_F2s − CE_L hierarchical CI upper bound < 0.
- **BF3:** CE_F2s − CE_G hierarchical CI upper bound < 0.002.
- **BF4:** integrity.

**Outcome labels:**
- POSITIVE;
- ACCURACY_TRANSFER_WITHOUT_SEMANTIC_FUNCTION (BF2 holds, BF1 fails);
- SEMANTIC_FUNCTION_WITHOUT_TRANSFER_GAIN (BF1 holds, BF2 fails);
- NEGATIVE.

**F1 diagnostics (always reported):**
- CE;
- exposure-matched usefulness;
- specialist spread at synchronisation (relative Frobenius spread);
- whether specialisation collapses.

## 8. Engineering (at most 3 generations; each tied to a measured failure; logged before running)

- **Q1 fails and F2s shows specialist spread at synchronisation < 0.05** (the deviations are erased): T × 2, up to 1000.
- **Q2 fails with Q1 passing** (no transfer gain): T × 0.5, down to 50.
- **Q4 fails:** no generation applies. Hard groups cannot collapse, and the router change has already been adopted.

A generation reruns Stage 2 only. EVAL is reused at most 3 times (disclosed); VAL is untouched.

## 9. Tests (all must pass before the freeze)

**CPU:**
- C1 centered sync continuity, transfer, and moment averaging;
- C2 FedAvg broadcast;
- C3 local unchanged, plus parameter accounting;
- C4 hard groups;
- C5 base bitwise identical with and without federated banks;
- C6 bitwise resume, with synchronisations on both sides of the resume point.

Result: ALL PASS (`cpu_tests_20261121T145239Z.json`). The first attempt failed only because the test harness had not created the artifacts directory, so the checkpoint ledger could not be written. That was a non-scientific fix.

**GPU:**
- G-EQ, G-ISO, G-RESUME (3 straight runs + first post-resume step), G-BENCH, G-EVAL (as in csur1).
- **Eval feasibility:** a 128-row SEL eval of the 23-bank bench checkpoint, extrapolated to the full SEL size, must be under 0.8 × 5400 s, the guarded-job cap. If it is not, the Stage-1 SEL eval would be split before the freeze.

**Result:** ALL PASS (`gpu_tests_20261121T151555Z.json`).
- G-EQ: step-0 lm, aux and z are bit-identical to TOKEN-4 s0.
- G-RESUME passes for lm and every bank's L_res (step-30 difference ≤ 0.0002).
- G-EVAL: forced vs routed maximum difference 0.00099.
- Eval feasibility: the full SEL estimate is 2,442 s.
- **G-ISO:** the v1 tolerance, built from one no-bank repeat pair, FAILED (0.0130 > 0.0088; `gpu_tests_20261121T150632Z.json` is preserved).
  - Losses are identical through step 30 and diverge at steps 40–50 through GPU nondeterminism.
  - The banked repeat runs differ from each other by up to 0.0166, more than banked vs no-bank (≤ 0.0145).
  - G-ISO v2 builds its tolerance from all same-type repeat pairs (logged deviation, test design only) and PASSES (0.0145 ≤ 0.0269).
  - Base isolation is also established bitwise on CPU (C5).

## 10. Compute, disk, operations

- **GPU cap:** 18 GPU-h.
- **Measured** (G-BENCH): 23 banks 3.85 s/step; 5 banks 1.19 s/step; no banks 0.46 s/step; peak memory 18.5 GiB.
- **Plan:**
  - Stage 1: train 0.34 h + SEL eval ≈ 0.7 h;
  - Stage 2: 2 × (0.26 h + EVAL eval);
  - final: 4 × 1.28 h + VAL and reference evals;
  - total ≈ 9 GPU-h.
- **Disk:** the shared disk is under external growth (it fell 121 → 100 GiB free over about 5 h). Superseded scratch belonging to this experiment is cleaned only after verification, and every deletion is logged. Jobs are gated on ≥ 100 GiB + job peak.
- **Rules:** identical to csur1 (≥ 100 GiB free + job peak, ≤ 2 slots per run, logged deletions of own scratch only, no /dev/shm, operator-interruption protocol).
- **Logs:** under `artifacts/csr-centered-fedavg-residual-v1/`.
