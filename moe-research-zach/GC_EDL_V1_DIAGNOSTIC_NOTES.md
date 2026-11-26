# GC-EDL-v1 — Diagnostic notes (addendum of 2026-11-15)

These notes cover logging and post-hoc analysis only. Nothing in them changed a candidate, a selection rule, validation
access or an acceptance criterion.

- **Ledger.** `artifacts/gc-edl-v1/diagnostic_ledger.jsonl` (append-only, ~740 rows). Each row records stage,
  candidate, seed, config hash, data-manifest hash, value with units, n, split, kind and the supporting artifact.
- **Summary.** `artifacts/gc-edl-v1/diagnostic_summary.json`.
- **Supporting files.**
  - `diagnostics/discovery_base.json`, `diagnostics/discovery_r1.json`: descriptive.
  - `diagnostics/posthoc.json`: post-hoc, written after unsealing.
  - `diagnostics/selected_table_*.npz`: compact per-paper tables.
- **Diagnostic storage.** 78 MiB.

**Terminal status: `GC-EDL-V1 STOPPED — DISCOVERY INVALID`.** No main model was trained, so sections 3–6 of the
addendum (EDL optimisation, exposure, functional ablation, trajectory) have no data. They are marked UNRESOLVED (not
reached) rather than inferred.

Unless stated otherwise, all numbers are on DISC-HOLD: the 20 % of training papers held out by hash, 9,546 of which
have ≥ 2 views. "Base" means the 128-d projection; "r1" means engineering revision 1 (256-d).

## 1. Fingerprint diagnostics

**Cosine between separated views of the same paper**, using the most-separated view pair and a matched random-paper
pairing, with 300-resample paper bootstrap CIs:

| representation | same paper | random paper | gap [95 % CI] | gap in random-sd units | FIT gap |
|---|---|---|---|---|---|
| SEM (bge) | 0.2388 | −0.0013 | 0.2401 [0.2357, 0.2446] | 2.19 | 0.2399 |
| COMBINED | 0.1284 | −0.0003 | 0.1287 [0.1257, 0.1314] | 1.80 | 0.1296 |
| GRAD (residual f, 128-d) | 0.0172 | 0.0008 | 0.0164 [0.0138, 0.0189] | 0.18 | 0.0182 |
| raw gradient direction u (128-d) | 0.0181 | 0.0013 | 0.0167 [0.0139, 0.0195] | 0.18 | 0.0190 |
| GRAD (256-d, r1) | 0.0172 | 0.0002 | 0.0170 [0.0147, 0.0190] | 0.25 | 0.0179 |

- **All view pairs.** The mean same-paper cosine of f is 0.0336 against −0.0021 for different papers (gap 0.036). The
  semantic gap is 0.317.
- **Split-half reliability.** Same-paper agreement of 0.034 implies a paper-level share of about 3.4 % of per-view
  fingerprint variance. The 4-view paper mean therefore has an estimated reliability of 4r/(1 + 3r) ≈ 0.13, against
  ≈ 0.65 for the semantic view.
- **Gradient norm** ‖∇W₂ CE‖_F (DISC-HOLD views): q05 0.63, median 0.72, q95 0.91. The norm is narrow and does not
  dominate.
- **Effective dimensionality.** The participation ratio is 115/128 (base) and 208/256 (r1). The first PC explains
  2.2 %, and the first 1/4/8/16/32 PCs explain 0.022/0.069/0.121/0.212/0.365 of the variance. The fingerprints are
  nearly isotropic, with no dominant direction.
- **Projection stability.** 10-NN paper neighbourhoods agree only 0.113 between the 128-d and 256-d sketches
  (chance 0.003). The neighbourhoods are therefore mostly sketch noise.
- **Prototype-seed 10-NN overlap** in q-space: GRAD 0.15–0.32, SEM 0.38–0.56, COMBINED 0.52–0.68.
- **Train vs held-out coherence.** The gaps are equal (GRAD 0.0182 on FIT vs 0.0164 on HOLD), so there is no
  overfitting; the signal is simply absent.

**Nuisance predictability**, held-out R² of each variable predicted from the fingerprint (before = raw u, after =
residual f):

| nuisance | before | after |
|---|---|---|
| frac top-100 tokens | 0.087 | 0.000 |
| mean log unigram prob | 0.086 | −0.000 |
| punctuation rate | 0.058 | −0.000 |
| scout CE | 0.044 | 0.000 |
| log grad norm | 0.040 | −0.000 |
| type/token ratio | 0.037 | −0.000 |
| whitespace | 0.036 | −0.000 |
| mean word length | 0.034 | −0.001 |
| citation density | 0.033 | 0.000 |
| digit rate | 0.026 | −0.000 |
| relative position | 0.024 | −0.000 |
| math/markup | 0.022 | −0.000 |
| uppercase | 0.017 | −0.000 |
| log paper length / log #segments | 0.001 / 0.001 | −0.000 |

- All 31 nuisance variables together explain R² = 0.0093 of fingerprint-direction variance (0.0102 at 256-d).
- The single largest contributor is frac_top100, at R² 0.0017.
- **Token frequency is the nuisance most visible in the raw gradient**, but it explains under 1 % of it.

**Post-hoc, after unsealing** (5-way domain classification, ridge one-vs-rest, trained on DISC-FIT and tested on
DISC-HOLD, majority chance 0.282):

| representation | paper-mean accuracy | single-view accuracy |
|---|---|---|
| SEM | 0.832 | 0.735 |
| COMBINED | 0.833 | — |
| GRAD | 0.325 (r1 0.369) | 0.290 (r1 0.314) |
| raw u | 0.333 (r1 0.374) | 0.292 (r1 0.319) |

- **Domain NMI / AMI of the selected clusterings:**
  - GRAD_K4: 0.007 / 0.007.
  - SEM_K16: 0.437 / 0.419.
  - COMBINED_K16: 0.444 / 0.425 (base); COMBINED_K8 (r1): 0.481 / 0.476.
  - Every GRAD K has NMI ≤ 0.009.
- **Confusion matrices** are in `posthoc.json["base"|"r1"]["confusion"]`, and cluster × domain contingencies in
  `["clusterings"]`.
- **Nearest neighbours**, 3,000 papers (r1):

| neighbour type | same domain (NN vs random) | \|ΔCE\| | formatting distance | token Jaccard |
|---|---|---|---|---|
| gradient NN | 0.287 vs 0.225 | 0.489 vs 0.506 | 2.99 vs 3.07 | 0.124 vs 0.122 |
| semantic NN | 0.784 vs 0.230 | 0.293 vs 0.496 | 2.27 vs 3.07 | 0.155 vs 0.122 |

  Gradient neighbours share subject only slightly more often than random papers, and do not share difficulty, format
  or vocabulary.
- **Representative opaque IDs and view offsets** for every selected cluster, with domain mix and a 48-token snippet,
  are in `posthoc.json[...]["cluster_representatives"]`.

## 2. Cluster-discovery diagnostics (every K and representation; base / r1 in the JSON files)

| config | κ_view | κ_seed | C (held-out coherence) | LD (nats/token) | S ± SE | eligible | confident / diffuse | share min–max | semantic coherence |
|---|---|---|---|---|---|---|---|---|---|
| SEM_K4 | 0.615 | 0.853 | 0.0165 | 0.074 | 0.843 ± 0.011 | yes | 0.80 / 0.05 | 0.175–0.293 | 0.380 |
| SEM_K8 | 0.535 | 0.713 | 0.0192 | 0.096 | 0.848 | yes | 0.49 / 0.05 | 0.081–0.189 | 0.463 |
| SEM_K16 | 0.461 | 0.733 | 0.0215 | 0.113 | 0.887 | yes | 0.26 / 0.05 | 0.036–0.093 | 0.510 |
| GRAD_K4 | 0.018 | 0.298 | 0.0049 | 0.000 | 0.149 | yes | 0.00 / 1.00 | 0.230–0.268 | 0.034 |
| GRAD_K8 | 0.011 | 0.161 | 0.0047 | 0.000 | 0.105 | yes | 0.00 / 1.00 | 0.029–0.295 | 0.033 |
| GRAD_K16 | 0.008 | 0.181 | 0.0056 | 0.000 | 0.119 | no (share) | 0.00 / 1.00 | 0.010–0.172 | 0.055 |
| COMBINED_K4 | 0.612 | 0.873 | 0.0169 | 0.076 | 0.855 | yes | 0.82 / 0.04 | 0.171–0.292 | 0.385 |
| COMBINED_K8 | 0.534 | 0.844 | 0.0195 | 0.098 | 0.895 | yes | 0.50 / 0.05 | 0.084–0.170 | 0.441 |
| COMBINED_K16 | 0.462 | 0.826 | 0.0218 | 0.116 | 0.924 | yes | 0.28 / 0.04 | 0.044–0.086 | 0.503 |

- The r1 numbers are within ±0.02 for SEM/COMBINED and ±0.003 for GRAD (`discovery_r1.json`).
- No near-duplicate prototypes: the maximum prototype cosine is ≤ 0.41.
- **Collapse.** Nothing collapsed. GRAD assignments are instead *uniformly diffuse* (normalised entropy 0.998), which
  means they carry no information.
- **Nuisance eligibility.** R² of scout CE predicted from assignments is 0.21–0.29 for SEM/COMBINED (domain-linked
  difficulty) and 0.00 for GRAD.

## 3–6. EDL optimisation, expert exposure, functional specialisation, trajectory

**UNRESOLVED (not reached).** The preregistered discovery-validity rule stopped the experiment before screening. No
EDL gradient, routing, exposure, ablation or CE measurement exists for GC-EDL. Nothing is inferred for these stages.

## 7. Failure localisation

| point | status | numerical evidence | confidence | cheapest discriminating follow-up | falsified by |
|---|---|---|---|---|---|
| **A. Fingerprint signal** | **Supported** (weak, not absent) | Most-separated-pair gap 0.016 [0.014, 0.019] = 0.18 random-sd. All-pairs gap 0.034 vs 0.317 semantic. Paper-mean reliability ≈ 0.13. Unchanged at 256-d. | High (CIs tight; 2 projections) | On 2k papers, the same-paper gap vs view length (256/1024/full segment), number of views, and θ (last-block all matrices; multi-layer sketch) | A gap ≥ 0.10 and κ_view ≥ 0.3 with a different θ or longer views |
| B. Nuisance domination | **Rejected** | All 31 nuisances explain R² 0.0093 of fingerprint variance; largest single 0.0017; ~0 after residualisation. GRAD cluster identity from mechanics groups: 0.270–0.278 vs chance 0.269. | High | — | Nonlinear nuisance models reaching R² > 0.3 |
| **C. Discovery** | **Supported for GRAD** (as a consequence of A); rejected for SEM/COMBINED | GRAD κ_view 0.008–0.020, κ_seed 0.15–0.31, 0 % confident. SEM/COMBINED κ_view 0.46–0.62, κ_seed 0.71–0.87. Validity (b): C(GRAD) − C(SEM) = −0.0174 [−0.0191, −0.0153]; C(COMBINED_K8) − C(SEM_K16) = −0.0023 [−0.0028, −0.0019]. At equal K16 the difference is +0.0003. | High | Same as A | GRAD κ_view ≥ 0.3 after a fix to A |
| D. Objective | Unresolved (not reached) | — | — | Strength-matched semantic latent EDL at 50M (diag ratios, JSD, top-k overlap) | — |
| E. Exposure | Unresolved | — | — | As D, plus per-expert pseudo-domain exposure entropy | — |
| F. Functional | Unresolved | — | — | 352-expert replacement audit on D's finals | — |
| G. Accuracy trade-off | Unresolved | — | — | — | — |
| H. Retention | Unresolved | — | — | — | — |
| I. Seed instability | Supported for GRAD *clusters* (κ_seed ≤ 0.31); model seeds unresolved | — | High / — | — | — |

## 8. Grammar versus subject (post-hoc, r1 GRAD_K4; base in the JSON)

Which variables predict the GRAD signal (held-out):

| predictor group | GRAD cluster identity accuracy (chance 0.271) | R² of functional fingerprint (paper mean) | R² of raw gradient direction |
|---|---|---|---|
| real domain | **0.300** | **0.0012** | 0.0012 |
| semantic embedding | 0.297 | **0.0043** | 0.0070 |
| equation/code/citation density | 0.275 | −0.0000 | 0.0010 |
| token-frequency statistics | 0.280 | −0.0001 | **0.0091** |
| scout CE | 0.278 | 0.0001 | 0.0010 |
| document position/length | 0.270 | −0.0001 | −0.0000 |
| formatting proxies | 0.278 | −0.0000 | 0.0019 |

- **Interpretation.** The raw gradient is weakly lexical: token frequency is its largest explained component, at
  R² 0.009. The residual functional fingerprint and the GRAD clusters are weakly *subject*-related: domain and the
  semantic embedding are the only predictors above chance. Both effects are tiny.
- **The fingerprints are dominated by neither grammar nor subject.** About 99 % of their variance is per-view
  variability that no measured variable explains and that does not repeat across views of the same paper (A).
- **Recommendation on g_context = g_local − g_full: test only as a diagnostic.**
  - The evidence does not show local-mechanics domination, which is what that residual is designed to remove.
  - It does show low signal-to-noise. Differencing two noisy gradient estimates on identical targets raises variance
    unless the shared component is large.
  - A future preregistered use should be gated on a ≤ 2k-paper diagnostic showing a same-paper cosine gap ≥ 0.10 and
    κ_view ≥ 0.3 for g_context.
  - It should not be used as a primary representation, and as a secondary regulariser only if that gate passes.
  - It is not part of this frozen experiment.

## Final diagnostic summary (causal chain)

| link | status | strongest supporting number |
|---|---|---|
| fingerprints → stable clusters? | **FAIL** | GRAD κ_view 0.020 and κ_seed 0.31 at best. Same-paper gradient cosine gap 0.016 (0.18 sd), against 0.240 (2.19 sd) for semantics. |
| → different routing? | UNRESOLVED (not reached) | — |
| → different expert exposure? | UNRESOLVED | — |
| → functional experts? | UNRESOLVED | — |
| → retained specialisation? | UNRESOLVED | — |
| → CE effect? | UNRESOLVED | — |

1. **What worked.**
   - The streamed gradient sketch is exact against autograd (relative error < 1e-10, float64) and matches across GPU
     and CPU.
   - Label isolation held: firewall, opaque IDs and hash partitions; all 13 CPU tests pass.
   - Nuisance residualisation is complete (residual R² ≤ 0.0001).
   - Semantic and COMBINED pseudo-domains are stable, balanced and domain-aligned (AMI 0.42–0.48) without being
     difficulty bins.
2. **First failed link.** Fingerprints → stable clusters. A single-view final-MLP gradient carries about 3 % of its
   variance as a paper-level signal.
3. **Why later stages could not rescue it.**
   - The protocol only allows EDL training on gradient pseudo-domains that add functional coherence beyond semantics.
   - GRAD clusters are noise, and COMBINED clusters are semantic clusters (C within ±0.003 of SEM).
   - Training on them would test semantic latent EDL, not GC-EDL.
4. **Most informative next experiment.** A preregistered, strength-matched, label-free *semantic* latent EDL, using
   the frozen `table_SEM` / `table_COMBINED` tables, aux/LM ratio 0.5–2 %, and the SHUF and explicit-EDL controls.
   This is the cleanest test of whether any stable label-free partition reproduces explicit EDL's 2.1× FSI at matched
   strength.
5. **What can be reused.**
   - Discovery inputs, fingerprints, residualiser and the frozen tables.
   - `gc_train.py` paper_latent (CPU resume-tested), `gc_screen.py`, `gc_evaluate.py`.
   - The fle TOKEN-4/EDL finals and the seed-100 TOKEN-4 milestones.
6. **What not to try again.**
   - Raw single-view W₂ gradient clustering.
   - Raising the sketch dimension.
   - Clustering by gradients before demonstrating a same-paper fingerprint gap of ≥ 0.1.
