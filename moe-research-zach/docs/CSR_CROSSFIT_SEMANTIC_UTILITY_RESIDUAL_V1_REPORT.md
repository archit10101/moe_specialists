# CSR-CROSSFIT-SEMANTIC-UTILITY-RESIDUAL-V1 (csur1) — Final report (Experiment A)

**Outcome: TERMINAL_STAGE1_FAIL_AFTER_GEN1. Negative for the deployable system.**

- **Stage 1:** no configuration passed the preregistered gates, either in generation 0 or after the one applicable engineering generation (generation 1, the router change).
- **Not run:** Stage 2 (50M tokens, 2 seeds) and the 245M final. The frozen rules forbid continuing past a failed gate.
- **Not opened:** VAL. The test split was never loaded.

**Records:**
- Preregistration: `docs/CSR_CROSSFIT_SEMANTIC_UTILITY_RESIDUAL_V1_PREREG.md`, sha256 d90195bb…a4.
- Logs: `artifacts/csr-crossfit-semantic-utility-residual-v1/` (DECISION_LOG.jsonl, OBSERVATIONS.md, deviations.jsonl, ledgers).
- Analyses: `stage1.json`, `stage1_g1.json`, `final_analysis.json`.

## 1. What was tested

**Model.** K=16 residual specialists (384→256→384, 3,145,728 params) sit after the final block of a detached TOKEN-4 base. The base is bitwise isolated from the banks (tested). Every 128-token view is served by exactly one specialist.

**Teacher (training only).** Responsibilities are r ∝ exp(u_partner/τ + η·log p_sem − λ·log(K·m)), where:
- u_partner is the cross-fitted next-token-loss reduction of each EMA specialist on the partner view (B weights A and A weights B), relative to the step mean;
- p_sem is a frozen label-free BGE/ProtoNet region prior;
- the last term is a mild corpus-level balance.

**Student (deployed).** A causal GRU over completed past views is distilled with KL(sg(r) || q) and deploys arg-max q. View 0 uses the zero state.

**Controls:**
- G: parameter-matched token-routed generic residual;
- SH: the identical pipeline with a persistent paper-transplant shuffled prior (token mass preserved to 0.016);
- SO / UO were defined for Stage 2 but were not reached.

**Stage 1:**
- seed 100, 20,057,391 tokens, SEL part of CALIB-SEARCH (503 papers, 6.40M tokens);
- grid: base point (c=1, η=0.5, ema 0.99, λ=0.3) plus 6 one-factor variants, each as an SU/SH pair, all 15 banks in one process.

**Gates per candidate:**
- student CE_SU − CE_G ≤ 0.002;
- no pathology in the student or teacher loads (max ≤ 0.25, effective number ≥ 8, ≤ 1 dead).

## 2. Results

All numbers are SEL, seed 100, step 318 (20.06M tokens). CE is in nats per token, and lower is better.
- **Δ_stu vs G and Δ_tea vs G:** the deployed student's and the training-only teacher's CE minus G's.
- **Retention:** student (CE_SH − CE_SU) / teacher (CE_SH − CE_SU).
- **I_SU − I_SH:** exposure-matched functional interaction, with a paper-bootstrap 95% CI (single seed).
- **Student max load:** the deployed router's largest specialist share.

**Generation 0** (mean-pooled router, 19,152 params): CE_G = 4.6517, CE_base = 4.6686, I_G = 0.0016.

| cand | Δ_stu vs G | Δ_tea vs G | SH teacher vs G | retention | I_SU − I_SH [CI] | student max load | gated |
|---|---|---|---|---|---|---|---|
| bp | +0.0061 | −0.0007 | +0.0082 | 0.41 | +0.0079 [0.0068, 0.0090] | 0.292 | no |
| c0.5 | +0.0071 | +0.0006 | +0.0071 | 0.13 | +0.0059 [0.0041, 0.0075] | 0.332 | no |
| c2 | +0.0065 | +0.0001 | +0.0088 | 0.40 | +0.0057 [0.0049, 0.0066] | 0.278 | no |
| ema0.9 | +0.0059 | −0.0012 | +0.0077 | 0.31 | +0.0087 [0.0077, 0.0097] | 0.267 | no |
| eta0.25 | +0.0070 | +0.0034 | +0.0074 | 0.46 | −0.0026 [−0.0048, −0.0004] | 0.287 | no |
| eta1 | +0.0065 | −0.0045 | +0.0088 | 0.31 | +0.0145 [0.0133, 0.0158] | 0.259 | no |
| lam1 | +0.0060 | −0.0004 | +0.0080 | 0.31 | +0.0073 [0.0064, 0.0082] | 0.268 | no |

**Generation 1** (preregistered §12 trigger 5, the student-only failure; [mean, last token] → Linear(768→20) → GRU(20), 18,216 params): CE_G = 4.6605, CE_base = 4.6785, I_G = 0.0012.

| cand | Δ_stu vs G | Δ_tea vs G | SH teacher vs G | retention | I_SU − I_SH [CI] | student max load | gated |
|---|---|---|---|---|---|---|---|
| bp | +0.0046 | −0.0014 | +0.0071 | 0.46 | +0.0063 [0.0049, 0.0077] | 0.261 | no |
| c0.5 | +0.0051 | −0.0001 | +0.0063 | 0.29 | +0.0043 [0.0027, 0.0057] | 0.340 | no |
| c2 | +0.0046 | −0.0010 | +0.0076 | 0.38 | +0.0043 [0.0029, 0.0058] | 0.248 | no (CE) |
| ema0.9 | +0.0049 | −0.0022 | +0.0068 | 0.29 | +0.0085 [0.0075, 0.0097] | 0.253 | no |
| eta0.25 | +0.0067 | +0.0024 | +0.0066 | 0.69 | +0.0016 [0.0006, 0.0026] | 0.284 | no |
| eta1 | +0.0052 | −0.0054 | +0.0079 | 0.23 | +0.0100 [0.0080, 0.0122] | 0.249 | no |
| lam1 | +0.0049 | −0.0010 | +0.0072 | 0.46 | +0.0037 [0.0021, 0.0053] | 0.241 | no (CE) |

- Teacher loads were healthy in all 14 SU banks: max ≤ 0.154, effective number ≥ 14.5, 0 dead.
- Student routing NMI with the paper-level semantic label: SU 0.20–0.24 vs SH 0.04–0.13. Teacher SU: 0.25–0.42.
- 5 to 11 of the 16 SU specialists are individually useful on their own assigned tokens.
- The base CE differs between the two processes by 0.010. GPU training is not bitwise reproducible across processes, so every gate is a within-process contrast.

## 3. Answers to the preregistered questions

**Was functional specialisation created?** In the training teacher, yes, at Stage-1 level only.
- The exposure-matched own-vs-swap interaction I_SU − I_SH is positive in 13 of 14 candidates; the paper-bootstrap CI excludes 0 in 13 (eta0.25 gen 0 is negative).
- The semantic teacher reaches CE up to 0.0054 below the token-routed generic residual, while the shuffled teacher is always 0.006–0.009 above it.
- This is one seed, at 20M tokens, on SEL. No replicated or held-out claim is made.

**Did it survive the causal student?** Partly in function, not in accuracy.
- The specialists keep a positive exposure-matched I_SU − I_SH under student routing (the I values above use the student's arg-max).
- The student keeps only 13–69% of the teacher's semantic-over-shuffled CE advantage (pooled about 1/3).
- Student CE is worse than G in all 14 candidates, by +0.0046 to +0.0071.

**Did accuracy improve?** No. Against the parameter-matched generic residual, the deployable system lost in every candidate and both generations.

**Semantic vs shuffled:** semantic beats shuffled in both teacher and student CE in all 14 candidates. The student margin is +0.0008 to +0.0042, but that is between two systems that are both worse than G.

**Routing exposure vs function:** the functional metric is exposure-matched (2,025 strata, cross-fitted halves), so the positive interaction is not an exposure artefact. Routing NMI alone was never used for selection.

**Uncertainty and seed consistency:**
- Stage 1 uses one seed by design, and the CIs are within-seed paper bootstraps.
- Seed consistency was never tested because Stage 2 was not reached.
- The CE gate failure is large relative to the gate (+0.0046 vs 0.002) and uniform across 14 candidates and two router architectures.

## 4. Causal explanation (best current account)

1. **View-level hard routing carries a structural cost.** A single specialist for a whole 128-token view is worse than per-token routing (G) unless the assignment carries information. The shuffled teacher has the same utility signal and is 0.006–0.009 worse than G.
2. **The teacher's information is about the current view.** It pays back that cost using the cross-fitted partner-view utility plus the current region's semantic prior.
3. **The student cannot see the current view.** It sees only completed past views, so it recovers about a third of the semantic advantage, which is not enough to beat G.
4. **The router change did not close the gap.** Doubling the router's view of the past (the last token as well as the mean) improved student CE by only about 0.001. This favours explanation (2), information unavailable to a causal router, over router capacity.

## 5. What remains unproven

- Whether the semantic teacher advantage replicates across seeds or at 50M/245M tokens; never tested.
- Whether a student with access to the current view (a non-causal or token-level router) would retain the advantage. That deployment model was outside the preregistered design.
- Whether longer training narrows the student gap. 20M tokens is short, but the gap did not shrink between generations.
- Real-domain (M2D2 label) specialisation; VAL was never opened.

## 6. Compute, deviations, operations

- **GPU-hours:** 4.48 used of a 32 cap (tests 0.36, Stage 1 ×2 4.13).
- **Engineering generations used:** 1 of 3 (trigger 5). No other trigger applied.
- **Deviations (all logged in deviations.jsonl):**
  - the pre-freeze shuffle redesign;
  - post-freeze optional `--tables-dir`/`--frozen` arguments for the B follow-up;
  - the generation-1 implementation, which leaves the generation-0 path bitwise unchanged.
- **Operational:**
  - an obsolete test run directory was deleted while running (logged);
  - the gen-1 SEL eval was refused once by the disk gate (shortfall 2.27 GiB), after which this experiment's superseded scratch was cleaned (logged with sha256) and the pipeline relaunched with no scientific effect.
- **Experiment B (cfar1)** is triggered under §13; see its own report.
