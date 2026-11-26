# GC-EDL-v1 — Engineering revision 1 (prereg section 9, trigger 1: unstable fingerprints)

Written and hashed 2026-11-15 before any revised fingerprint exists. Base preregistration sha256
75591d88f264155cb4afd73bb35c9e4fc6292d042c2d2b14d98cdb6ab5643e6d.

## Measured failure (discovery_decision.json, DISC-HOLD, 9,546 papers with >= 2 views)
- Selected GRAD candidate GRAD_K4: kappa_view 0.018 (< 0.2 trigger), kappa_seed 0.298, most-separated-pair kappa 0.013,
  mean normalised assignment entropy 0.998 (near-uniform), LD 0.0000 nats/token.
- Functional fingerprint f: same-paper cosine 0.0336 vs different-paper -0.0021 (gap 0.036); semantic gap 0.317.
- Validity (b) failed: C(GRAD_K4) - C(SEM_K16) = -0.0165 [-0.0186, -0.0144]; C(COMBINED_K16) - C(SEM_K16) = +0.00034
  [-0.000007, +0.00067]. Validity (a) passed for both GRAD and COMBINED.
- Nuisance variables explain only 0.9 % of fingerprint-direction variance (held-out), so trigger 2 does not fire.

## Change (only this)
CountSketch projection dimension 128 -> 256 (same seed 20261115, same TensorSketch construction with m = 256). Same
views, scout, theta (blocks.9.ffn.w2.weight), nuisance features and residualisation procedure, standardisation,
representations (COMBINED = [s/sqrt(384), f/sqrt(256)]), prototype model/objective/grid/seeds, selection rule,
eligibility and validity tests. SEM does not depend on the projection: its 9 fits and metrics are reused unchanged.

## Procedure and decision
gc_fingerprint.py --m 256 (main, calib_search, calib_sel) -> gc_discover.py residualize/fit/select with variant r1
(outputs latent/r1/, discovery_decision_r1.json). The r1 decision replaces the original one for screening if valid.
If validity (a) or (b) still fails, no further fingerprint revision exists in section 9 (trigger 1's only remedy is
used) -> GC-EDL-V1 STOPPED — DISCOVERY INVALID, unless a different section-9 trigger (2 or 3) fires on the r1
measurements, in which case that trigger's remedy is revision 2 (the last). Revision counts as 1 of at most 2.
Original 128-d results remain in the ledger.
