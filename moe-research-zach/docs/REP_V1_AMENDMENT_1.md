# csr-representative-v1 — preregistration amendment 1 (v1.1): temperature calibration only

Approved by the user on 2026-11-01. Written and hashed before any code change or rerun. The SHA-256 of this
file is recorded in `ledger/safety_ledger.tsv` (`REP_V1_AMENDMENT_1`) and in `docs/REP_V1_AMENDMENT_1.sha256`.
This file is never edited.

## 1. What is amended

| Item | Value |
|---|---|
| Original preregistration | `docs/REP_V1_PREREG.md`, sha256 `d2ace8020820fa1d40ab5f2c245c9511451d1b75ae048ddd2f1270e05e45528e` (unchanged, preserved) |
| Scope | §6 τ_s / τ_r selection only: the coarse discrete grids are replaced by deterministic log-space bisection |
| Unchanged | everything else in the preregistration, including the 8-neighbour target and both 20% gates |

## 2. Failed calibration (preserved unchanged)

**Run.** `runs/20261101T043501Z_rep_calibrate_tau`, status ABORTED. Reason:
`tau gate failed: relative gap 0.324 > 0.2 (STOP per prereg)`.

**Artifact hashes.**
- `summary.json`: `dc484cbb811d84338bf97b13e6218545cf4e46a6b2b728637d8bb635af3d1090`
- `config.json`: `2355cdfd9b38b0f2047ede362079858f2bdbc458dfd1110f2ae1fe42a3006685`
- `status.json`: `c7800508abfe6fae46f4896552230de6907513f048c9d9825561e5549df954ca`

**Inputs.**
- seed-99 E model at initialization;
- first 8 training micro-batches in seed-99 order (train_order_seed 116);
- comparison set n_max 256, giving 94–109 chunks per batch;
- bf16 autocast forward with fp32 router logits;
- medians of per-chunk exp H. For R, the per-chunk count is first averaged over the 7 MoE layers.

**Observed effective-neighbour counts (all observed values).**

| τ_s | median exp H(Q) |
|---|---|
| 0.01 | 1.0139 |
| 0.02 | 1.3254 |
| 0.05 | 5.3143 |
| 0.1 | 45.3951 |
| 0.2 | 86.0232 |
| 0.5 | 99.4754 |

| τ_r | median exp H(R), layer-averaged |
|---|---|
| 1e-5 | 1.6671 |
| 2e-5 | 3.5917 |
| 5e-5 | 18.6863 |
| 1e-4 | 39.5359 |
| 2e-4 | 64.2603 |
| 5e-4 | 91.3967 |
| 1e-3 | 98.9171 |
| 2e-3 | 101.0398 |
| 5e-3 | 101.8454 |
| 1e-2 | 101.9614 |
| 2e-2 | 101.9903 |

Grid selection was τ_s = 0.05 (5.3143) and τ_r = 2e-5 (3.5917), a relative gap of 0.3241 > 0.20. The
experiment stopped.

## 3. Reason

The preregistered grids were too coarse to satisfy the unchanged 20% gate:
- the semantic count jumps from 5.31 to 45.4 between adjacent τ_s values;
- the router count jumps from 3.59 to 18.7 between adjacent τ_r values.

The purpose of the temperatures is to give the semantic-neighbour and router-neighbour distributions
comparable breadth, i.e. matched sharpness. It is not to optimize validation loss. "Eight effective
neighbours" is a soft entropy-based quantity, not a hard selection of 8 chunks or experts.

The amendment uses only the training-data calibration numbers above. No validation loss, validation example,
test data or experimental outcome was used; none exists.

## 4. Amended procedure

**Frozen inputs.** Exactly the same inputs as §2: calibration examples and their order, architecture,
initialization, semantic embeddings, router initialization, numerical precision, and the effective-neighbour
calculation. The model forward pass is computed once; S and G are frozen, and every temperature is evaluated
on them.

**Search** (exp H increases monotonically with temperature):
- evaluate both bracket endpoints;
- then up to 20 iterations of the log-space midpoint m = sqrt(lo · hi);
- if f(m) < target, set lo = m, otherwise hi = m;
- stop early only on exact equality;
- no random sampling, no replacement of calibration examples, and no bracket expansion;
- every evaluated temperature and count is recorded.

**Selection.** The evaluated temperature whose count is closest to the target. On equal absolute error, the
smaller temperature wins.

**τ_s.**
- Target: 8.
- Initial bracket: [0.05, 0.1].
- Gate: all values finite, and the selected count in [6.4, 9.6].

**τ_r.**
- Target: the count produced by the accepted τ_s.
- Initial bracket: [2e-5, 5e-5].
- Gate: all values finite, and |count_r − count_s| / count_s ≤ 0.20.

**Failure.** If either gate fails, the experiment stops and the full trace is reported. The brackets, target,
gates and all other rules are not changed.

**Provenance check.** The original grid points are re-evaluated on the frozen inputs and compared with the
failed run's recorded values.

## 5. Global freeze and matched comparison

- Exactly one (τ_s, τ_r) pair is frozen globally.
- It is written identically into every A–F resolved configuration for all three seeds and the tuning
  configuration, including conditions with semantic.enabled = false.
- There is no recalibration by condition, seed, checkpoint or training stage. Only C and E activate the
  semantic loss.

Effect on the contrasts:

| Contrast | Effect |
|---|---|
| D − F, B − A | unaffected; no semantic loss in either condition |
| E − D, C − B | differ only by semantic.enabled |
| E − C | identical temperatures; differ only by segmentation |

All architecture, data, initialization, optimizer and training settings remain matched (fail-closed check).

## 6. No other change

The following are unchanged:
- architecture, tied embeddings, data and splits, semantic segmentation, accepted F-v3;
- initialization, objectives and loss formulas, the 8-neighbour target, the 20% thresholds;
- λ grid and selection rule, token budget, seeds, optimizer and schedule;
- validation plan, analysis rules, stop criteria, and the test-split lock.
