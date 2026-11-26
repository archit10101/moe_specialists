# csr-latent-forecast-router-distill-v1: stopped before the final evaluation

**Terminal status: EXTERNALLY STOPPED.**
- The operator stopped V1 on 2026-11-12 after it froze its candidate and before any final validation evaluation.
- V1 is superseded by csr-context-prior-specialization-v2.

## Why V1 stopped

The user instructed:
- let the active `final_TOKEN-4_s0` baseline finish, as long as it could do so without opening validation;
- then stop before launching any other V1 final condition;
- preserve every V1 artifact.

What actually happened:
- `final_TOKEN-4_s0` finished its training normally at 12:30:02. It produced 40/60/80/100% milestones, with no
  end-of-run evaluation.
- The STOP marker (`artifacts/latent-forecast-router-distill-v1/STOP`) was created as soon as its process exited. The
  driver then recorded the run as complete and exited before launching the final evaluation.
- `driver_state.json` holds both the `external_stop` record and an `operator_stop` record (STOPPED_BY_OPERATOR).
- **Validation was never opened for any V1 run.** The test split was never opened.
- V1 must not be resumed without an explicit user instruction.

## What V1 completed (development only; no confirmatory claim is possible)

- **Protocol:** `docs/LATENT_FORECAST_ROUTER_DISTILL_V1_PROTOCOL.md` (sha256 fe99b29e…, unchanged).
- **Preparation:**
  - forecast cache: dev region, plus the final region covering 60% of FINAL;
  - draft gate passed 5/5 domains;
  - counterfactual first-order estimator gate passed: Spearman 0.81, top-4 overlap 2.88;
  - GPU causality checks passed;
  - resume check passed.
- **Stage 1** (49M tokens, seed 100). The table gives CALIB-SEARCH CE, lower is better; FSI is the domain functional
  specialisation index.

| run | CE | FSI domain |
|---|---|---|
| TOKEN-4 | 3.7896 | 0.00396 |
| EDL β=5e-3 (best EDL) | 3.7715 | 0.00925 |
| OUTPUT-DISTILL α=1.0 | 3.7345 | 0.00377 |
| proposed candidates P01–P12 | 3.860–4.235 (all worse than TOKEN-4) | 0.0005–0.0011 (all below TOKEN-4) |

- **Stage 2** (122.5M tokens, seeds 100/101; CALIB-SEARCH CE):

| run | CE | FSI domain |
|---|---|---|
| TOKEN-4 | 3.302 / 3.294 | 0.0058 / 0.0063 |
| EDL | 3.278 / 3.290 | 0.0139 / 0.0138 |
| OUTPUT-DISTILL | 3.271 / 3.275 | 0.0066 / 0.0069 |
| MEMORY-ONLY | 3.292 / 3.291 | 0.0061 / 0.0061 |
| P07 / P12 / P04 / P09 | 3.366–3.379 | 0.0009–0.0014 |

- **Selection:**
  - No candidate was eligible: none beat TOKEN-4 FSI.
  - The protocol therefore froze the lowest-CALIB-SELECTION-CE candidate, **P12**, as INELIGIBLE. Its selection CE was
    3.335 / 3.340, against TOKEN-4's 3.2685 / 3.2608.
- **Diagnosis (development):**
  - While teacher-routed, the forecast teacher helped language modelling (P04/P07 were ahead of TOKEN-4 at the end of
    P1).
  - After the teacher was stripped, every candidate lost that gain and stayed about 0.07 nats behind TOKEN-4.
  - The teacher's utility head never predicted exact expert utility (Spearman about 0).
  - Every candidate was **less** functionally specialised than TOKEN-4.

## What V1 did not do

V1 ran no final evaluations. It also did not run these final conditions:
- EDL, P12, OUTPUT-DISTILL;
- TOKEN-4 seed 1;
- EDL+CF, FORECAST-SIMILARITY, SHUFFLED-FORECAST;
- TOKEN-COMPUTE, MEMORY-ONLY.

It also did not produce the benchmark or the final analysis. Consequently, `final_analysis.json`,
`specialization_trajectory.json` and `counterfactual_utility_analysis.json` do not exist.

The development record is complete and preserved:
- `run_manifest.tsv`, `candidate_table.tsv`, `compute_ledger.tsv` and `deviations.jsonl`;
- `diagnostics/`, `driver_log.txt` and `driver_state.json`.

The development result is negative for the V1 method: all candidates were worse than TOKEN-4 in both CE and
functional specialisation. It is not a final result.

## Deviations

There were six deviations, all implementation or test fixes with no effect on any metric:
1. Compression sort signature.
2. Weak-draft API fix.
3. Cache 90-minute wall-time resume.
4. GPU causality check run in fp32.
5. Resume-check invariant counted a kept rejected copy.
6. Final-cache prefix-chunk reuse.

The full records are in `artifacts/latent-forecast-router-distill-v1/deviations.jsonl`.

## Compute

- 16.12 GPU-h of the 40 GPU-h budget (V1 `compute_ledger.tsv`).
- The reused forecast cache, the TOKEN-4 runs and the development results are inputs to V2 (hash-verified, read-only).
