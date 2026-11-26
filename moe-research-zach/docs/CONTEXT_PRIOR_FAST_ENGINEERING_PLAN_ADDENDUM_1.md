# Fast-engineering plan, addendum 1 (2026-11-13, after F01's early stop; disclosed as deviation 4)

The section-2 training-CE gate is recalibrated. **Observation:** F01 was stopped at step 544, where its paired
training-loss gap to TOKEN-4 was +0.033 against the +0.02 limit. The same statistic for the 11 V2 S1 runs whose prior
was inert (C01–C09, M1, M5) has these values:

| Window | Median | Range |
|---|---|---|
| [168, 318) | +0.013 | -0.004 to +0.030 |
| [394, 544) | +0.026 | -0.008 to +0.053 |

These runs are TOKEN-4-equivalent (prior-off dCE < 4e-4). The literal rule would have stopped 7 of the 11, including
V2's frozen C07 (+0.030). At this 49M scale, single-batch training losses at three logged steps carry run-to-run
noise of about ±0.02.

**New rule.** Stop if (gap − null median at that gate) > +0.02.

| Plan | Gate step | Null median | Source runs |
|---|---|---|---|
| S1 | 318 | +0.0131 | 11 inert V2 S1 runs |
| S1 | 544 | +0.0255 | 11 inert V2 S1 runs |
| S2 seed 100 | 795 | -0.0044 | V2 S2 C02, C07, M1, M5 |
| S2 seed 100 | 1357 | -0.0003 | V2 S2 C02, C07, M1, M5 |
| S2 seed 101 | 795 | +0.0047 | V2 S2 C02, C07, M1, M5 |
| S2 seed 101 | 1357 | -0.0006 | V2 S2 C02, C07, M1, M5 |

**What is unchanged.** All other gates, the 49M promising criteria, the S2 freeze rule, the budget and the candidate
cap.

**F01.** F01 is re-screened under this rule as F01R: identical knobs and seed, a fresh run. It counts as one
additional screened candidate. F01's early-stopped run remains in the table.
