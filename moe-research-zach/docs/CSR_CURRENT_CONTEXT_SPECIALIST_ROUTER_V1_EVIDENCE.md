# CSR-CURRENT-CONTEXT-SPECIALIST-ROUTER-V1 (ccsr1) — Evidence summary (written before the design)

## Sources read

- `docs/CSR_CROSSFIT_SEMANTIC_UTILITY_RESIDUAL_V1_REPORT.md` (csur1, Experiment A)
- `docs/CSR_CENTERED_FEDAVG_RESIDUAL_V1_REPORT.md` (cfar1, Experiment B)
- `docs/CSR_SEMANTIC_RESIDUAL_AB_COMBINED_CONCLUSION.md`
- csur1 and cfar1 `OBSERVATIONS.md` (40 + 23 lines) and `DECISION_LOG.jsonl` (7 + 3 decisions)
- `ORACLE_RESIDUAL_V1_REPORT.md` and `ORACLE_RESIDUAL_V1_DIAGNOSTIC_NOTES.md` (orv1)
- the fcuv1 lesson on per-token utility (memory note on csr-forecast-counterfactual-utility-v1)

## What the earlier experiments established

**1. The specialists are functionally subject-specific.**
- orv1, oracle paper-level assignment, VAL, 4 seeds:
  - the semantic bank beats a mass-matched shuffled bank by +0.0024 nats in 4 of 4 seeds;
  - exposure-matched own-specialist advantage S_A = 0.0048 vs S_B ≈ 0;
  - all 16 clusters are positive.
- csur1, Stage 1: the exposure-matched I_SU − I_SH is > 0 in 13 of 14 candidates.
- cfar1, Stage 1: hard semantic groups give I_F2s − I_F2h = +0.014 to +0.017, with the CI excluding 0 in 9 of 9.

**2. The training-only current-region teacher beats the generic residual G.**

cfar1 Stage 1, SEL, seed 100, CE change vs base (token mean, nats; lower is better):

| bank / router | all | 0–127 | 128–511 | 512+ |
|---|---|---|---|---|
| G (token-routed generic, own linear router) | −0.0168 | −0.0173 | −0.0171 | −0.0165 |
| L (local semantic) — current-region teacher | −0.0246 | −0.0301 | −0.0240 | −0.0236 |
| L — past-view student (deployed) | −0.0048 | +0.0107 | −0.0068 | −0.0074 |
| F2h_T500_w240 (local shuffled) — teacher | −0.0081 | −0.0083 | −0.0082 | −0.0080 |
| F2h_T500_w240 — past-view student | −0.0089 | −0.0091 | −0.0090 | −0.0087 |

- csur1 semantic teachers also beat G, by up to 0.0054.
- **3. The past-view student failed because it committed before seeing the current region.**
  - Its first view, with no information, hurts by +0.011.
  - At later positions it reaches about −0.007, against the teacher's −0.023.
  - A larger past-view router input (csur1 generation 1) changed CE by only about 0.001.
- **4. Subjects switch between adjacent 256-token regions in 31.4% of within-document transitions** (argmax p_sem, CALIB-SEARCH main table).
- **5. Federation:**
  - standard FedAvg erased specialist differences (T100 teacher −0.0130 vs local −0.0246);
  - centered FedAvg was CE-neutral relative to local training (|Δ| ≤ 0.001), with the common update only 9–15% of the local update.

## Further constraints

- **orv1 at a linear token router:** the advantage of the semantic oracle over a linear token router (C) was concentrated in the first 128 positions (about 60% of the gap). C's routing was nearly orthogonal to the semantic clusters (MI 0.014–0.021 nats, under 1% of H(sem)).
  - So a single-hidden-state *linear* router did not discover the subject.
  - Whether a recurrent router over the current context can do so is untested; that is this experiment.
- **fcuv1: per-token utility is largely token-idiosyncratic.**
  - The per-token best-specialist oracle therefore mostly selects noise. A CPU smoke test on 2 rows gave oracle −0.28 vs teacher −0.04.
  - It is a loose upper bound, and regret relative to it will be large for any causal router.
  - The exact-utility teacher R2 may be partly unlearnable.
- **Reuse:** the strongest reusable semantic bank is cfar1's L, with current-region teacher −0.0246 vs base.
  - Its exactly paired shuffled control is F2h_T500_w240: same process, rows and specialist initialisation, except the 16 shared-only hidden units [240, 256) are averaged at init. It never synchronised (T = 500 > 318 steps), so it is local training on persistent shuffled groups.
  - G is in the same checkpoint and saw the same rows.

## Starting hypothesis (H1)

The specialists already encode subject-specific function. What failed was deployment: committing to one specialist per 128-token view from completed past views.

A strictly causal router that updates every token from the current base hidden state h_t, a 16-token causal running mean and its own recurrent state should infer the current subject within a few tokens of a region or document start. It should then:
- recover at least 60% of the current-region teacher's semantic-over-shuffled advantage;
- be at least 0.003 better than the past-view student;
- match or beat G.

**Falsifiers:**
- CE worse than G by more than 0.002 on any router seed;
- retention below 60%;
- loss of the exposure-matched semantic-over-shuffled functional interaction;
- load collapse;
- gains confined to the first 128 tokens or to one subject.

**Expected weak points:**
- document starts, where no current context exists yet;
- the first tokens after a subject switch;
- R2's noisy per-token targets.
