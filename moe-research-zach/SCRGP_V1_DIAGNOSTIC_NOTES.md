# SCRGP-V1 — diagnostic notes

These notes are a companion to `SCRGP_V1_REPORT.md`.
- Everything in §2 and later is **post hoc** and descriptive, taken from `artifacts/scrgp-v1/diagnostics_posthoc.json` (produced by `scrgp_diag.py`, CPU only, read only).
- None of it is a preregistered criterion, and none of it can change the frozen STAGE1_FAIL verdict.

## 1. Was Sim a bug?

Sim_SEM came out at +0.58 to +1.62 nats against a held-out natural CE of about 3.08. That is large enough to suspect the simultaneous-update code. It was checked three ways:

- **Additivity.** Σ over pairs of the per-pair natural all-token CE change (each pair updated alone, using the same code path as N) gives:
  - SEM: 0.373 / 0.460 / 1.029 / 0.508, against Sim 0.576 / 0.746 / 1.620 / 0.730;
  - SHUF: 0.020 / 0.026 / 0.039 / 0.021, against Sim 0.029 / 0.023 / 0.057 / 0.030.

  These are the same order of magnitude, and seed 2 is the largest in both. The mild super-additivity is what one expects when 32 coherent updates in the same layer compound.
- **Hashes.** The state hash was restored after the simultaneous update in every seed.
- **Unit test.** The T-EVAL unit test covers the simultaneous path.

**Conclusion:** Sim is real.

## 2. Per-layer anatomy

All values are means over 4 seeds and 32 experts per layer. Units are nats, and "worse" means CE goes up.

**SEM:**

| layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| F (contrast gain) | .00046 | .00041 | .00044 | .00041 | .00029 | .00037 | .00048 | .00051 | .00052 | .00088 | **.0205** |
| G (global forced, worse) | .00014 | .00010 | .00009 | .00009 | .00005 | .00006 | .00009 | .00007 | .00008 | .00017 | **.0227** |
| assigned-cluster forced Δ | −.00016 | −.00010 | −.00014 | −.00010 | −.00007 | −.00007 | −.00004 | −.00006 | −.00002 | +.00004 | **+.0116** |
| other-cluster forced Δ | +.00016 | +.00011 | +.00011 | +.00011 | +.00006 | +.00007 | +.00010 | +.00008 | +.00009 | +.00018 | **+.0235** |
| fraction of other clusters worse | .74 | .74 | .75 | .74 | .70 | .70 | .72 | .73 | .74 | .83 | **.99** |
| N (exposure-matched) | .00015 | .00010 | .00012 | .00014 | .00011 | .00020 | .00024 | .00024 | .00026 | .00073 | **.0492** |
| natural all-token Δ per pair | .00004 | .00004 | .00003 | .00003 | .00002 | .00002 | .00003 | .00003 | .00004 | .00011 | **.0181** |
| predicted η‖u‖² | .0023 | .0022 | .0021 | .0020 | .0018 | .0020 | .0019 | .0020 | .0020 | .0023 | .0106 |
| realised F / predicted | .20 | .19 | .21 | .21 | .16 | .19 | .25 | .25 | .27 | .39 | 1.93 |
| median relative step | .059 | .057 | .054 | .053 | .050 | .051 | .050 | .051 | .049 | .052 | **.106** |
| split-half cos(u) | .086 | .090 | .092 | .094 | .078 | .096 | .118 | .130 | .142 | .179 | **.413** |

**SHUF:** F is about 0 in every layer (|F| ≤ 4e-5). G is +0.00003 to +0.00007 in layers 0–9 and +0.0021 in layer 10. The assigned cluster and the other clusters worsen alike. Split-half cos(u) is about 0 in layers 0–9 and 0.087 in layer 10.

## 3. Reading

**Layer 10 (block 11, the last MoE layer)** dominates every metric.

- **What the step does.**
  - Its subject residual is the most stable direction (split-half cos 0.41).
  - Under forced insertion, the step makes *every* cluster worse. The assigned cluster worsens half as much as the rest.
  - The positive F and N there are therefore relative effects: "hurts my subject less". They are not "helps my subject".
- **Why the step is so large there.**
  - Layer 10 also got about 2× the relative step of the other layers. This comes from one global η and a larger layer-10 ‖u‖, which is inherent to the preregistered rule.
  - The realised gain was 1.9× the first-order prediction, so the step was in the nonlinear regime for this layer.
- **What the evidence does not settle.**
  - A smaller layer-10 step would reduce both the damage and F.
  - Whether the *sign structure* (assigned worse, other clusters worse still) would flip at a small step cannot be read from these data.
  - At first order the assigned-cluster change is −η d_pos·u and need not be negative. Only the *contrast* d·u = ‖u‖² is guaranteed positive.

**Layers 0–9** show the cleanest picture of what the mechanism does at an in-regime step (relative step ≤ 6 %).

- The assigned cluster gains about 1e-4 per pair, and the other clusters lose about the same.
- G is about 20 % of F, which on its own would satisfy V4(a).
- But the realised gain is only about 20 % of the in-sample prediction. That fits the split-half stability: roughly 10 % of ‖u‖ is reproducible signal and the rest is sample-specific.
- Summed natural damage across the 320 experts in these layers is 0.007–0.019 nats, which is above the +0.001 Sim bound (assuming additivity).
- Effect sizes per invocation are tiny relative to CE (3.08) and are zero-sum.

**Why projection does not help.**
- u is orthogonal to the balanced *global* gradient g. That removes the component that would change the q-weighted average loss to first order.
- It does not remove the component that moves loss *between* subjects. A gain on k paid for by not-k is exactly what d = g_pos − g_ctrl measures.
- The prereg's "do not worsen nonassigned subjects" requirement therefore needs a constraint against the not-k gradients, not just against g. That was not part of SCRGP-V1 as specified.

## 4. Things checked and found not to be the problem

- **Matching coverage.** The matched positive-weight fraction is 1.0 on A1, A2 and B in every seed. V0 failed only on the balance of log doc length (median |SMD| ≈ 0.09) and router rank (≈ 0.07), which put the per-pair maximum just over 0.10 on A1.
- **Semantic content.** The k* lift median is 1.67 vs 1.21 for SHUF. ‖d_SEM‖/‖d_SHUF‖ is 1.19. SEM's split-half stability is about 10× SHUF's in every layer.
- **Numerics.**
  - The probed expert ran in fp64.
  - The GPU gradients matched the CPU fp32 reference (min cos 0.9997).
  - The probe's natural CE was bit-identical to model.forward.
  - u⊥g held to within 2.5e-16.
  - All hash restores were verified.
- **Resume.** No resumes happened in Stage 1 (every job ran in a single leg). The resume path itself was GPU-tested at cos 0.99999999999994.

## 5. What a follow-up would have to address (not authorised; recorded only)

1. **Protect against redistribution between subjects.** Project u against the per-cluster not-k gradients, or require a non-positive first-order change for every non-assigned cluster. Doing this in a new experiment would require a new prereg.
2. **Step rule.** Use a per-layer relative-step rule so that the last layer is not over-stepped.
3. **Matching.** Stratify on document length.
4. **Last layer.** Decide whether last-layer subject residuals (likely vocabulary priors) belong in the "competence" definition at all.

No such follow-up was run.
