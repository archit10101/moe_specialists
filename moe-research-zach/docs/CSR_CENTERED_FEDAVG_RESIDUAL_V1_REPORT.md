# CSR-CENTERED-FEDAVG-RESIDUAL-V1 (cfar1) — Final report (Experiment B)

**Outcome: TERMINAL_STAGE1_FAIL. Negative.**

- **Stage 1:** no (T, w_p) configuration passed the preregistered gates. The prereg defines no engineering generation for a Stage-1 failure (§5, §8).
- **Not run:** Stage 2 and the 245M final.
- **Not opened:** VAL. The test split was never loaded.

**Records:**
- Preregistration: `docs/CSR_CENTERED_FEDAVG_RESIDUAL_V1_PREREG.md`, frozen with sha256 d5ce1ddc…7e66 before any scientific job.
- Logs: `artifacts/csr-centered-fedavg-residual-v1/`.
- Analyses: `stage1.json`, `final_analysis.json`.

B was triggered because Experiment A (csur1) failed (`docs/CSR_CROSSFIT_SEMANTIC_UTILITY_RESIDUAL_V1_REPORT.md`).

## 1. What was tested

**Shared with csur1:**
- the same detached TOKEN-4 base and residual architecture (16 specialists 384→256→384, one per 128-token view);
- the same data, label-free semantic region prior and paper-transplant shuffled prior;
- the causal student router, here A's generation-1 version ([mean, last] → GRU(20), 18,216 params).

**Groups.** The group of a view is the arg max of its region prior. Specialist k trains only on group-k views (its "client"). The student is distilled with KL(onehot(group) || q) and deploys arg-max q.

**Federation modes** (personalised units [0, w_p), shared-only units [w_p, 256)):
- **L:** local training, no federation.
- **F1:** standard FedAvg. Every T steps, the token-weighted mean update is broadcast, so all specialists become identical.
- **F2:** centered personalised FedAvg, residual_k = S + P_k. At each sync:
  - D̄ (the token-weighted mean update) is transferred into the shared S;
  - D_k − D̄ stays in the private P_k;
  - the function on personalised units is exactly continuous (measured maximum change 0.0);
  - the shared-only units are averaged;
  - AdamW moments are averaged on every reset slice.

**Matching.** Per-token FLOPs are identical to csur1 in every condition, and w_p = 240 matches csur1's 3,145,728 training parameters exactly.

**Stage 1:**
- seed 100, 20,057,391 tokens (318 steps), SEL;
- 23 banks in one process: G, L, F1s at T ∈ {100, 250, 500}, and F2s/F2h for T ∈ {100, 250, 500} × w_p ∈ {240, 192, 128};
- T = 500 never synchronises within 318 steps (disclosed in the prereg).

**Gates:**
- student CE_F2s − CE_G ≤ 0.002;
- no student or teacher load pathology.

## 2. Results (SEL, seed 100, step 318)

**Gates:**
- Student CE_F2s − CE_G is +0.0120 to +0.0129 in all 9 candidates (gate 0.002).
- Student arg-max loads: max 0.330, effective number 9.66, 1 dead. Every F2s student is identical, because the target and the router initialisation are identical.
- Group loads are fine (max 0.153, effective number 14.4).
- **Result: 0 of 9 gated.**

**Functional interaction.** Exposure-matched I_F2s − I_F2h ranges from +0.0142 to +0.0166, and the paper-bootstrap CI excludes 0 in 9 of 9. For example, T250/w192 gives +0.0155 [0.0127, 0.0181].

**CE change vs base.** Token mean on SEL, in nats; lower is better. The student column is deployed; the teacher column uses the hard group of the current region, which is information available only in training. Ranges are over the T and w_p settings.

| bank | student | teacher | student, pos 0–127 | pos 128–511 | pos 512+ |
|---|---|---|---|---|---|
| G (token-routed) | −0.0168 | — | −0.0173 | −0.0171 | −0.0165 |
| L (local) | −0.0048 | −0.0246 | +0.0107 | −0.0068 | −0.0074 |
| F2s (semantic, centered) | −0.0039 … −0.0048 | −0.0220 … −0.0244 | ≈ +0.010 | ≈ −0.0065 | ≈ −0.0070 |
| F2h (shuffled, centered) | −0.0080 … −0.0089 | −0.0073 … −0.0082 | ≈ −0.0087 | ≈ −0.0086 | ≈ −0.0083 |
| F1s T100 / T250 / T500 | −0.0080 / −0.0068 / −0.0053 | −0.0130 / −0.0194 / −0.0250 | | | |

**Synchronisation diagnostics** (T100, w_p 192; syncs at steps 99, 199 and 299):
- F2s keeps its specialist spread at sync (relative Frobenius 0.93–0.94, partly reflecting initialisation differences).
- The transferred common component is small relative to the local updates: ‖D̄‖ / ‖D‖ = 0.15, 0.09, 0.09.
- F1s spread after each sync is 0 by construction.

## 3. Answers to the preregistered questions

**Was functional specialisation created?** Yes, in the specialists, at Stage-1 level.
- Hard semantic groups produce specialists that are much more useful on their own group's tokens than shuffled-group specialists are on theirs, after exposure matching (CI excludes 0 in 9 of 9).
- With oracle current-region groups, they beat the token-routed generic residual by about 0.007 nats.
- One seed, 20M tokens; not replicated.

**Did it survive the student?** No, not for accuracy.
- The deployed causal student keeps about 20% of the group-routed gain, and its CE is +0.012 worse than G.
- Its first view (zero state, one fixed specialist) actively hurts (+0.011 at positions 0–127).
- At later positions it reaches only about −0.007, against the teacher's −0.023.

**Did accuracy improve?** No. Every federated or local student is far worse than the parameter- and FLOP-matched generic residual.

**Semantic vs shuffled:** semantic wins on exposure-matched function but loses on deployed CE (student −0.0046 vs −0.0085).
- Shuffled groups are persistent per paper, so they are easy to predict from the past.
- But they carry little content, so their specialists are near-generic and teacher and student behave the same.

**Did FedAvg transfer useful knowledge, or merely erase differences?**
- **Centered F2 transfer:** CE-neutral at 20M. F2s T100/T250 and the never-synced T500 arms differ by ≤ 0.001 from each other and from local training L. The common update is only about 9–15% of the local update norm, so there was little shared knowledge to transfer.
- **Standard FedAvg:** erased useful differences. T100 teacher −0.0130 vs local −0.0246, getting worse the more often it syncs.

**Uncertainty and seed consistency:** single seed, single stage. The CE gate failure (+0.012 vs a 0.002 gate) is far beyond any seed noise seen in this programme (cross-process base CE differs by about 0.01, but every gate is within-process).

## 4. Causal explanation

1. Subject-specialised residuals are useful when the current region's subject is known.
2. A causal router sees only completed past views, so it cannot know the current region's subject. The within-document region-switch rate is 0.31, and the first view has no information at all.
3. One specialist per view, wrongly chosen, is worse than per-token generic routing.
4. Federation does not address this bottleneck. It changes how the specialists are trained, not how they are selected at inference.

## 5. What remains unproven

- Whether centered transfer helps at longer horizons (only 1–3 syncs happened at 20M).
- Whether a base-only first view would help. The request allowed it, but neither prereg chose it. It would remove the view-0 penalty (about 0.0014 of the mean) but not the roughly 0.017 gap at later positions.
- Whether a router with access to current-view content (token-level or non-causal) would deploy the semantic specialists. That falls outside both preregistered designs.

## 6. Compute and operations

- **GPU-hours:** 1.19 used of an 18 cap (tests 0.20, Stage 1 0.99).
- **Deviations (logged):**
  - G-ISO v2 tolerance (test design; the v1 failure is preserved);
  - test-harness artifacts-directory creation;
  - the eval-feasibility check.
- **Disk:** superseded scratch belonging to this experiment was cleaned after verification, with every deletion logged with sha256. The shared disk was under external growth throughout.
