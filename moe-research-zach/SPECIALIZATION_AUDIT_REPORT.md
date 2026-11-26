# specialization-audit-v1: report

**What this is.** A prospectively frozen post-training audit, not part of the original preregistration. It was
designed after the csr-representative-v1 validation-CE results were known. It was frozen
(`docs/SPECIALIZATION_AUDIT_V1_PREREG.md`, sha256 `17998b104ac37e25d7d0e75f699daededc44070714d03ae42c515262229c5daa`,
unchanged at the end of the audit) before any audit data, label, routing record or intervention result existed.

**What it did not change.**
- No model was retrained.
- No weight, configuration, segmentation, temperature, λ, routing rule or checkpoint was changed.
- The csr-representative-v1 CE conclusions stand exactly as reported:
  - B − A: +0.110 nats; hard pooling hurt.
  - D − F: mixed sign.
  - E − D: +0.026 nats; semantic loss hurt.
  - C − B: +0.022 nats; semantic loss hurt.
  - H1, H2 and H3 were not supported.

**Frozen decision-rule outcome (§10): category 4, "Functional specialization without clear routing
segregation."**
- In both semantic-alignment comparisons (E − D, C − B), the functional-increase criterion is met.
- The routing-increase criterion is not met. Routing specialization moved significantly in the *opposite*
  direction.
- The prereg's harmful-specialization flag is defined only for categories 1–2, so it is formally not raised (see Q6).
- The prereg calls category 4 "unexpected" and requires a descriptive look at measurement sensitivity, without
  changing the analysis. That look is in §5.

---

## Plain-language answers

**1. Did the semantic loss create stronger routing specialization? No. It created weaker routing specialization.**
- Adding the semantic alignment loss reduced how much routed experts' gate mass tells you about a window's topic:
  - permutation-adjusted MI R fell by about 0.10–0.13 nats (from ≈0.29–0.31 to ≈0.18–0.19);
  - the topic probe's macro-F1 fell by 0.02–0.04.
- This holds in every seed, in both comparisons (E − D, C − B), under both label views (e5 and TF-IDF), and at
  K = 16/32/64.
- Every per-seed bootstrap CI excludes 0, and every observed difference lies below the paired null's 2.5th
  percentile.

**2. Did it create functional specialization? A small, statistically clear increase on the frozen metric.**
- What F measures: removing an expert from a token, then comparing the CE damage on tokens of the expert's
  discovery top topic with matched tokens of other topics. F is the size of that gap.
- With the semantic loss, F rose by +0.0005 to +0.0010 nats in every seed of both comparisons:
  - per-seed bootstrap CIs exclude 0;
  - observed values exceed the 97.5th percentile of the paired null (null 97.5% ≈ +0.00024);
  - the TF-IDF ΔF is positive in every seed.
- So the §10 functional criterion is met for E − D and for C − B.
- The absolute size is small: condition-mean F is 0.0023–0.0026 nats with the loss vs 0.0017 without, and most of
  it comes from block 7.
- This shows that *something* about which experts matter for which topics changed. It does not show a useful
  specialization (see Q6). It does not come with stronger routing segregation (Q1).

**3. Were specialties reproducible across seeds? Only partly, and only on one of the frozen measures.**
- **Lift-based measure (above null):** for a seed-0 expert, check whether its aligned partners in seeds 1 and 2
  share the same argmax-lift topic. This rate is above the load-preserving null in every pooled condition:
  - E: 19.6% vs null 3.7% (p = 0.001);
  - C: 9.4% vs 3.2% (p = 0.001);
  - B, D, F: 11–12% vs ≈7% (p = 0.013–0.030).
  - It is not above null for A (8.9% vs 7.3%, p = 0.24).
- **Mass-based measure (not above null anywhere):** the argmax-mass top topic reproduces in 16–21% of experts,
  which is not above the null in any condition.
- **Matched-JSD measure (cannot be read):** this test cannot detect reproducibility under the frozen null.
  Permuting labels pushes every expert's profile toward the global topic mix, so the null JSD is tiny
  (≈0.0001–0.001) and the observed JSD (0.006–0.036) is always "worse" (p = 1.0). This is a design limitation of
  that null, not evidence against reproducibility.

**4. Did semantic boundaries help beyond matched-random boundaries (D − F)? No.**
- Neither the routing criterion nor the functional criterion is met:
  - ΔR has mixed sign (+0.003, −0.010, −0.006);
  - ΔP is negative in all seeds;
  - ΔF has mixed sign (+0.00026, −0.00011, −0.00017).
- This matches the mixed-sign CE result (+0.008, +0.003, −0.007).

**5. Did hard pooling help beyond token routing (B − A)? It produced more topic-coherent routing and a larger
functional interaction, but not "help".**
- **Routing (criterion not met):** R is far higher with pooling (+0.276 nats; CIs and the null all exceeded;
  TF-IDF agrees). But the probe is *worse* with pooling (ΔP ≈ −0.06 in all seeds), so the frozen routing criterion
  fails on condition (d).
  - Part of the pooled R reflects mechanics: tokens of a chunk share one candidate pool, which inflates agreement
    with window-level labels.
- **Functional (criterion met):** ΔF is +0.0009 to +0.0016 in all seeds; CIs and null exceeded; TF-IDF agrees.
  Token-routed A shows essentially no functional topic interaction: F ≈ 0.0004, below its null 97.5% in 2 of
  3 seeds.
- **CE context:** hard pooling worsened validation CE by +0.110 nats. Nothing here says pooling helped language
  modelling.

**6. Was any specialization useful, neutral or harmful relative to CE? Not useful. Every increase in functional
specialization coincides with worse CE.**
- The conditions with higher F all have worse validation CE than their comparators:
  - E vs D: +0.026 nats;
  - C vs B: +0.022 nats;
  - B vs A: +0.110 nats.
- Per seed, E − D shows the same ordering in ΔF and ΔCE. With three points, this is not a test.
- The audit is observational about the trained models. It cannot show that the specialization *caused* the CE
  loss. The pattern is consistent with the loss buying a small amount of topic-dependent expert function at a cost
  in modelling quality.
- Specialization is therefore **not** described as beneficial. By the prereg's own definition, the formal
  "harmful" flag (categories 1–2 only) is not raised. The plain reading is "associated with harm, not shown to
  cause it".

**7. Is the result robust to the independent labeler and the shuffled null?**
- **Routing decrease:** fully robust. TF-IDF and every e5 K agree in sign in all seeds, and all values lie far
  outside the null.
- **Functional increase, C − B:** robust. e5 K = 16/32/64, TF-IDF and the all-7-layer secondary are positive in all
  seeds; TF-IDF CIs exclude 0 in all seeds.
- **Functional increase, E − D:** less robust.
  - The TF-IDF ΔF is positive in all seeds, which is what §10 requires, but its CI includes 0 in seed 1
    ([−0.00001, +0.00042]).
  - e5 K = 64 is negative in seed 2 (−0.00015). K is a sensitivity analysis, not part of the decision rule.
- **Probe:** above both the majority (F1 ≈ 0.003) and shuffled-label (F1 ≈ 0.03) controls in every checkpoint.
- **Coverage:** no expert was excluded at any primary layer, in any condition or view, so the result is not driven
  by exclusions.

---

## 1. Design recap (as frozen)

**Checkpoints.** 18 csr-representative-v1 checkpoints: seeds 0–2 × conditions A–F.

| Condition | Routing | Segmentation | Semantic loss |
|---|---|---|---|
| A | token | none | off |
| B | pool | fixed 128 | off |
| C | pool | fixed 128 | on |
| D | pool | semantic | off |
| E | pool | semantic | on |
| F | pool | random F-v3 | off |

The semantic loss acts only during training, so C and E route at inference exactly like B and D.

**Checkpoint integrity.** All 18 were verified at Stage 0: sha256, strict load, config identical to the frozen
condition files, and mapping. They were re-hashed after all GPU work and all 18 still match (`SPEC_AUDIT_POST_VERIFY`).

**Audit data.**
- New FineWeb-Edu documents, downloaded only for this audit: 12 row groups of
  `sample/10BT/000_00000.parquet` @ 87f09149, row groups disjoint from all project splits.
- Deduplicated against the model-visible text of non-test splits: 22 exact and 7 near duplicates removed.
- 8,192 documents (5,205,077 tokens), split by document into:
  - discovery half: 4,067 docs, 13,583 windows;
  - effect half: 4,125 docs, 13,760 windows.
- 3,400,236 mask tokens, of which 1,712,538 are effect tokens.

**Labels.**
- Primary view: intfloat/e5-base-v2 @ f52bf8ec window embeddings, deterministic k-means K = 32 (K = 16/64 as
  sensitivity).
- Independent view: TF-IDF/SVD-256 with K = 32 (16/64 as sensitivity).
- Fitted on discovery windows only.

**Routing extraction.** fp32 with TF32 off, on native segmentation. The segmentation provenance check reproduced
the frozen tables with 0 mismatches.

**Metrics.**
- **R** = layer-mean permutation-adjusted MI between topic and gate mass.
- **P** = probe macro-F1, from a window gate profile to topic.
- **F** = mean over blocks 1, 4 and 7 of the equal-expert-weighted ATT interaction. For each expert, the
  interaction compares single-token counterfactual removal on its top-topic tokens with stratified-matched
  other-topic tokens.

**Statistics.**
- Uncertainty: 2,000 paired document-bootstrap resamples.
- Null: 1,000 paired load-preserving label permutations.
- Seed 20261101 throughout.

## 2. Per-condition results (three-seed means; per-seed values in §3)

| Cond | val CE (rep-v1) | R e5 | R TF-IDF | P e5 | F e5 | F TF-IDF | within-expert cos excess | pool MI adj |
|---|---|---|---|---|---|---|---|---|
| A | 4.2198 | 0.0348 | 0.0335 | 0.583 | 0.00039 | −0.00034 | 0.0010 | n/a |
| B | 4.3299 | 0.3112 | 0.1976 | 0.522 | 0.00169 | 0.00086 | 0.0075 | 0.1666 |
| C | 4.3518 | 0.1863 | 0.1040 | 0.486 | 0.00258 | 0.00127 | 0.0041 | 0.1301 |
| D | 4.3276 | 0.2902 | 0.1879 | 0.519 | 0.00165 | 0.00077 | 0.0069 | 0.1501 |
| E | 4.3540 | 0.1836 | 0.1031 | 0.499 | 0.00228 | 0.00117 | 0.0040 | 0.1286 |
| F | 4.3259 | 0.2947 | 0.1884 | 0.528 | 0.00166 | 0.00074 | 0.0069 | 0.1544 |

Notes on the table:
- Probe controls: shuffled-label macro-F1 0.029–0.041 and majority 0.003 in every checkpoint.
- Routing MI exceeds its permutation null at every layer of every checkpoint (layer-mean p = 0.001, the minimum).
  The null MI 97.5% is ≈0.0005 for A and ≈0.003–0.005 for pooled conditions.
- No experts had zero load or low support in any checkpoint.
- The audit's own fp32 CE on audit mask tokens reproduces the validation ordering:
  - A ≈ 4.233–4.242;
  - B, D, F ≈ 4.339–4.355;
  - C, E ≈ 4.354–4.390.

## 3. Per-seed (per-checkpoint) table

These are e5 K = 32 values unless marked. CE values are from rep-v1 validation (bf16 eval).

| key | val CE | R | R TF-IDF | P | P TF-IDF | F | F TF-IDF | F K16 | F K64 | F all-7 (sec.) | null F 97.5% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| s0_A | 4.2163 | 0.0328 | 0.0322 | 0.5852 | 0.4857 | 0.00019 | −0.00022 | −0.00036 | 0.00134 | 0.00025 | 0.00044 |
| s0_B | 4.3364 | 0.3111 | 0.1972 | 0.5281 | 0.3237 | 0.00169 | 0.00084 | 0.00152 | 0.00204 | 0.00142 | 0.00021 |
| s0_C | 4.3640 | 0.1859 | 0.1034 | 0.4862 | 0.2910 | 0.00266 | 0.00124 | 0.00222 | 0.00295 | 0.00216 | 0.00018 |
| s0_D | 4.3354 | 0.2967 | 0.1898 | 0.5212 | 0.3176 | 0.00186 | 0.00082 | 0.00169 | 0.00229 | 0.00153 | 0.00023 |
| s0_E | 4.3730 | 0.1842 | 0.1029 | 0.4998 | 0.2957 | 0.00260 | 0.00150 | 0.00206 | 0.00282 | 0.00203 | 0.00020 |
| s0_F | 4.3272 | 0.2940 | 0.1861 | 0.5304 | 0.3274 | 0.00161 | 0.00071 | 0.00152 | 0.00190 | 0.00142 | 0.00023 |
| s1_A | 4.2261 | 0.0370 | 0.0346 | 0.5885 | 0.4812 | 0.00080 | −0.00018 | 0.00046 | 0.00062 | 0.00040 | 0.00043 |
| s1_B | 4.3266 | 0.3140 | 0.1979 | 0.5220 | 0.3190 | 0.00164 | 0.00085 | 0.00116 | 0.00204 | 0.00143 | 0.00021 |
| s1_C | 4.3450 | 0.1844 | 0.1031 | 0.4840 | 0.2907 | 0.00240 | 0.00114 | 0.00189 | 0.00293 | 0.00202 | 0.00018 |
| s1_D | 4.3254 | 0.2885 | 0.1877 | 0.5199 | 0.3184 | 0.00156 | 0.00068 | 0.00156 | 0.00204 | 0.00144 | 0.00024 |
| s1_E | 4.3538 | 0.1849 | 0.1034 | 0.5001 | 0.3007 | 0.00220 | 0.00089 | 0.00197 | 0.00243 | 0.00209 | 0.00019 |
| s1_F | 4.3220 | 0.2981 | 0.1916 | 0.5299 | 0.3162 | 0.00168 | 0.00083 | 0.00149 | 0.00194 | 0.00146 | 0.00023 |
| s2_A | 4.2172 | 0.0347 | 0.0339 | 0.5746 | 0.4828 | 0.00018 | −0.00063 | −0.00011 | 0.00013 | 0.00007 | 0.00045 |
| s2_B | 4.3268 | 0.3084 | 0.1978 | 0.5164 | 0.3249 | 0.00174 | 0.00088 | 0.00133 | 0.00232 | 0.00145 | 0.00023 |
| s2_C | 4.3462 | 0.1886 | 0.1054 | 0.4863 | 0.2973 | 0.00267 | 0.00141 | 0.00224 | 0.00323 | 0.00208 | 0.00018 |
| s2_D | 4.3221 | 0.2855 | 0.1864 | 0.5167 | 0.3104 | 0.00153 | 0.00082 | 0.00140 | 0.00230 | 0.00126 | 0.00023 |
| s2_E | 4.3352 | 0.1816 | 0.1029 | 0.4963 | 0.2978 | 0.00204 | 0.00111 | 0.00166 | 0.00215 | 0.00205 | 0.00018 |
| s2_F | 4.3286 | 0.2919 | 0.1875 | 0.5240 | 0.3322 | 0.00170 | 0.00069 | 0.00147 | 0.00238 | 0.00149 | 0.00023 |

The full metric set is in `artifacts/specialization-audit-v1/report_tables/per_checkpoint.csv` and
`analysis/routing_metrics.csv`. It includes per-layer MI, NMI, H(T|E), purity, KL, JSD diversity, pool MI and
cosine excess.

## 4. Primary contrasts (paired, per seed)

**Columns.**
- ΔR: the 95% CI is the document bootstrap; "null 97.5%" is the paired permutation null.
- ΔP: probe macro-F1 difference.
- ΔF: e5 K = 32, with its CI and paired null.
- ΔCE: from rep-v1 validation.

### E − D (semantic loss, semantic chunks): routing criterion NOT met; functional criterion MET

**Routing**

| seed | ΔR [CI] | null 97.5% | ΔR TF-IDF | ΔP [CI] |
|---|---|---|---|---|
| 0 | −0.1125 [−0.1161, −0.1097] | −0.0010 | −0.0869 | −0.0214 [−0.0311, −0.0122] |
| 1 | −0.1036 [−0.1073, −0.1009] | −0.0009 | −0.0842 | −0.0198 [−0.0294, −0.0103] |
| 2 | −0.1039 [−0.1077, −0.1010] | −0.0009 | −0.0835 | −0.0204 [−0.0300, −0.0108] |
| mean | −0.1067 | | −0.0849 | −0.0205 |

**Functional and CE**

| seed | ΔF [CI] | null [2.5, 97.5] | ΔF TF-IDF [CI] | ΔF K16 / K64 | ΔCE |
|---|---|---|---|---|---|
| 0 | +0.00073 [+0.00047, +0.00099] | [−0.00028, +0.00022] | +0.00068 [+0.00046, +0.00089] | +0.00036 / +0.00053 | +0.0375 |
| 1 | +0.00063 [+0.00040, +0.00088] | [−0.00028, +0.00024] | +0.00021 [−0.00001, +0.00042] | +0.00042 / +0.00039 | +0.0285 |
| 2 | +0.00051 [+0.00025, +0.00076] | [−0.00027, +0.00024] | +0.00029 [+0.00007, +0.00053] | +0.00026 / −0.00015 | +0.0131 |
| mean | +0.00062 | | +0.00039 | | +0.0264 |

### C − B (semantic loss, fixed chunks): routing criterion NOT met; functional criterion MET

**Routing**

| seed | ΔR [CI] | null 97.5% | ΔR TF-IDF | ΔP [CI] |
|---|---|---|---|---|
| 0 | −0.1252 [−0.1291, −0.1221] | −0.0009 | −0.0938 | −0.0419 [−0.0507, −0.0328] |
| 1 | −0.1296 [−0.1336, −0.1266] | −0.0009 | −0.0948 | −0.0379 [−0.0474, −0.0290] |
| 2 | −0.1198 [−0.1237, −0.1169] | −0.0009 | −0.0923 | −0.0301 [−0.0389, −0.0215] |
| mean | −0.1249 | | −0.0936 | −0.0366 |

**Functional and CE**

| seed | ΔF [CI] | null [2.5, 97.5] | ΔF TF-IDF [CI] | ΔF K16 / K64 | ΔCE |
|---|---|---|---|---|---|
| 0 | +0.00097 [+0.00072, +0.00122] | [−0.00025, +0.00024] | +0.00040 [+0.00019, +0.00061] | +0.00070 / +0.00091 | +0.0276 |
| 1 | +0.00076 [+0.00050, +0.00103] | [−0.00025, +0.00021] | +0.00029 [+0.00008, +0.00050] | +0.00073 / +0.00089 | +0.0185 |
| 2 | +0.00094 [+0.00072, +0.00114] | [−0.00026, +0.00025] | +0.00053 [+0.00033, +0.00073] | +0.00091 / +0.00091 | +0.0195 |
| mean | +0.00089 | | +0.00041 | | +0.0218 |

### D − F (semantic vs matched-random boundaries): neither criterion met

**Routing**

| seed | ΔR [CI] | null 97.5% | ΔR TF-IDF | ΔP [CI] |
|---|---|---|---|---|
| 0 | +0.0026 [+0.0004, +0.0048] | +0.0003 | +0.0037 | −0.0092 [−0.0183, +0.0001] |
| 1 | −0.0096 [−0.0119, −0.0074] | +0.0002 | −0.0039 | −0.0100 [−0.0192, −0.0005] |
| 2 | −0.0064 [−0.0086, −0.0042] | +0.0002 | −0.0011 | −0.0073 [−0.0166, +0.0019] |
| mean | −0.0045 | | −0.0004 | −0.0088 |

**Functional and CE**

| seed | ΔF [CI] | null [2.5, 97.5] | ΔF TF-IDF [CI] | ΔF K16 / K64 | ΔCE |
|---|---|---|---|---|---|
| 0 | +0.00026 [+0.00004, +0.00048] | [−0.00026, +0.00027] | +0.00011 [−0.00009, +0.00031] | +0.00017 / +0.00039 | +0.0083 |
| 1 | −0.00011 [−0.00037, +0.00013] | [−0.00025, +0.00028] | −0.00015 [−0.00037, +0.00005] | +0.00007 / +0.00010 | +0.0034 |
| 2 | −0.00017 [−0.00041, +0.00007] | [−0.00026, +0.00027] | +0.00013 [−0.00009, +0.00036] | −0.00007 / −0.00008 | −0.0065 |
| mean | −0.00001 | | +0.00003 | | +0.0017 |

### B − A (hard pooling vs token routing): routing criterion NOT met (probe condition d fails); functional criterion MET

**Routing**

| seed | ΔR [CI] | null 97.5% | ΔR TF-IDF | ΔP [CI] |
|---|---|---|---|---|
| 0 | +0.2783 [+0.2757, +0.2870] | +0.0043 | +0.1650 | −0.0571 [−0.0662, −0.0475] |
| 1 | +0.2771 [+0.2745, +0.2859] | +0.0043 | +0.1632 | −0.0665 [−0.0763, −0.0564] |
| 2 | +0.2737 [+0.2711, +0.2825] | +0.0043 | +0.1639 | −0.0582 [−0.0684, −0.0483] |
| mean | +0.2764 | | +0.1640 | −0.0606 |

**Functional and CE**

| seed | ΔF [CI] | null [2.5, 97.5] | ΔF TF-IDF [CI] | ΔF K16 / K64 | ΔCE |
|---|---|---|---|---|---|
| 0 | +0.00150 [+0.00101, +0.00196] | [−0.00047, +0.00047] | +0.00105 [+0.00067, +0.00144] | +0.00188 / +0.00070 | +0.1201 |
| 1 | +0.00085 [+0.00033, +0.00135] | [−0.00046, +0.00043] | +0.00104 [+0.00066, +0.00142] | +0.00070 / +0.00142 | +0.1005 |
| 2 | +0.00155 [+0.00111, +0.00202] | [−0.00049, +0.00045] | +0.00151 [+0.00113, +0.00187] | +0.00143 / +0.00219 | +0.1096 |
| mean | +0.00130 | | +0.00120 | | +0.1101 |

### Criteria check (§10)

**Routing increase**, conditions (a)–(e):

| Contrast | (a) ΔR > 0 all seeds | (b) CI excludes 0 | (c) exceeds null | (d) ΔP > 0 | (e) TF-IDF ΔR > 0 | Met |
|---|---|---|---|---|---|---|
| E − D | no | yes | no | no | no | no |
| C − B | no | yes | no | no | no | no |
| D − F | no | yes | no | no | no | no |
| B − A | yes | yes | yes | no | yes | no |

**Functional increase:**

| Contrast | ΔF > 0 all seeds | CI excludes 0 | exceeds null | TF-IDF ΔF > 0 | Met |
|---|---|---|---|---|---|
| E − D | yes | yes | yes | yes | **yes** |
| C − B | yes | yes | yes | yes | **yes** |
| D − F | no | no | no | no | no |
| B − A | yes | yes | yes | yes | **yes** |

**Category.**
- Functional increase without routing increase, in both E − D and C − B → **category 4**.
- Harmful flag: not applicable (categories 1–2 only).
- The routing *decrease* in E − D and C − B is the descriptive mirror of the criteria: all seeds, CIs and null
  2.5% passed in the negative direction.

## 5. Category-4 sensitivity look (descriptive only; the decision is not changed)

These are post-hoc descriptive checks. They are not part of the frozen rule.

**Is the ΔF increase just a larger overall ablation effect?** No.
- The mean global ΔCE from removing a selected expert (blocks 1, 4, 7) is *smaller* with the semantic loss:
  - C ≈ 0.0019 and E ≈ 0.0019 nats;
  - B ≈ 0.0030, D ≈ 0.0029, F ≈ 0.0028;
  - A ≈ 0.012.
- Dividing each layer's interaction by its global ΔCE gives:

| Condition | Scaled F (mean of seeds) |
|---|---|
| C | 1.37 |
| E | 1.26 |
| B | 0.63 |
| D | 0.60 |
| F | 0.63 |
| A | ≈ 0.01 |

- So in C and E, expert removal is cheaper overall but relatively more concentrated on the expert's discovery top
  topic. Table: `report_tables/descriptive_scaled_F.csv`.

**Where does F come from?** Mostly block 7, the last MoE layer: e.g. E 0.0040 and C 0.0049 vs B 0.0035 and D 0.0034.
Blocks 1 and 4 show smaller but same-direction differences (table below).

**How can routing look less topical while function is more topical?**
- The routing metrics measure how gate mass divides over topics across *all* tokens.
- F measures the CE damage on an expert's top-topic tokens relative to matched other tokens, *among tokens that
  selected that expert*.
- The semantic loss appears to have made routing less topic-segregated overall, while the remaining
  topic-dependent experts are slightly more consequential for their topic.
- This is a description, not a mechanism. The audit does not establish one.

## 6. All primary-layer functional interactions

Equal-expert-weighted ATT interaction I in nats, e5 K = 32, per checkpoint × primary block. The
utilization-weighted versions and the TF-IDF view are in `report_tables/primary_layer_functional.csv`. Every
per-expert estimate, with counts, matched fractions and global ΔCE, is in `analysis/functional_experts.csv`
(16,128 rows: 18 checkpoints × 7 layers × 32 experts × 4 views; the primary flag marks blocks 1, 4, 7).

| key | block 1 | block 4 | block 7 |
|---|---|---|---|
| s0_A | −0.00022 | −0.00025 | +0.00104 |
| s0_B | +0.00096 | +0.00076 | +0.00334 |
| s0_C | +0.00135 | +0.00203 | +0.00459 |
| s0_D | +0.00105 | +0.00103 | +0.00352 |
| s0_E | +0.00156 | +0.00179 | +0.00444 |
| s0_F | +0.00064 | +0.00145 | +0.00273 |
| s1_A | +0.00003 | +0.00070 | +0.00166 |
| s1_B | +0.00059 | +0.00068 | +0.00366 |
| s1_C | +0.00120 | +0.00163 | +0.00437 |
| s1_D | +0.00068 | +0.00057 | +0.00344 |
| s1_E | +0.00149 | +0.00126 | +0.00383 |
| s1_F | +0.00063 | +0.00046 | +0.00394 |
| s2_A | −0.00031 | −0.00030 | +0.00115 |
| s2_B | +0.00084 | +0.00100 | +0.00338 |
| s2_C | +0.00122 | +0.00118 | +0.00562 |
| s2_D | +0.00082 | +0.00042 | +0.00335 |
| s2_E | +0.00124 | +0.00110 | +0.00378 |
| s2_F | +0.00076 | +0.00087 | +0.00349 |

Condition means:

| Condition | e5 block 1 | e5 block 4 | e5 block 7 | TF-IDF block 1 | TF-IDF block 4 | TF-IDF block 7 |
|---|---|---|---|---|---|---|
| A | −0.00017 | +0.00005 | +0.00128 | −0.00029 | −0.00001 | −0.00073 |
| B | +0.00079 | +0.00081 | +0.00346 | +0.00026 | +0.00051 | +0.00180 |
| C | +0.00126 | +0.00162 | +0.00486 | +0.00061 | +0.00078 | +0.00240 |
| D | +0.00085 | +0.00067 | +0.00344 | +0.00036 | +0.00041 | +0.00155 |
| E | +0.00143 | +0.00139 | +0.00402 | +0.00070 | +0.00073 | +0.00207 |
| F | +0.00068 | +0.00093 | +0.00339 | +0.00036 | +0.00042 | +0.00145 |

## 7. Cross-seed alignment

Hungarian alignment uses discovery profiles, all 7 layers × 32 experts, and seed pairs (0,1), (0,2), (1,2). The
null is the same 1,000-permutation load-preserving null, with the alignment re-optimized each time.

| Cond | matched JSD (null mean) | top-topic repro, mass (null mean [95%]) | p (higher) | top-topic repro, lift (null mean [95%]) | p (higher) |
|---|---|---|---|---|---|
| A | 0.0056 (0.0001) | 0.210 (0.360 [0.268, 0.478]) | 1.000 | 0.089 (0.073 [0.036, 0.121]) | 0.236 |
| B | 0.0363 (0.0011) | 0.161 (0.222 [0.147, 0.313]) | 0.955 | 0.116 (0.065 [0.031, 0.107]) | 0.013 |
| C | 0.0248 (0.0009) | 0.214 (0.172 [0.116, 0.237]) | 0.101 | 0.094 (0.032 [0.009, 0.058]) | 0.001 |
| D | 0.0332 (0.0009) | 0.192 (0.241 [0.156, 0.330]) | 0.881 | 0.121 (0.072 [0.036, 0.116]) | 0.021 |
| E | 0.0241 (0.0008) | 0.210 (0.188 [0.125, 0.254]) | 0.278 | 0.196 (0.037 [0.013, 0.067]) | 0.001 |
| F | 0.0341 (0.0011) | 0.201 (0.236 [0.156, 0.330]) | 0.798 | 0.112 (0.070 [0.036, 0.112]) | 0.030 |

How to read the three measures:
- **Matched JSD:** p(lower) = 1.000 in all conditions, for the reason given in Q3. The frozen null cannot support
  a reproducibility claim from matched JSD.
- **Mass-based reproduction:** this is dominated by globally frequent topics, which the null also captures. It is
  not above the null anywhere.
- **Lift-based reproduction:** above the null in all pooled conditions, and most clearly in E.

Per-layer values are in `analysis/analysis.json` (`cross_seed`).

## 8. Null distributions and bootstrap CIs

**Null distributions.**
- Every routing MI value has its own null: per checkpoint, layer and view, 1,000 permutations, recorded as null
  mean and 97.5%. Values are in `analysis/routing_metrics.csv`.
- Each F has its null mean and 97.5%, and each contrast has its paired-null 2.5/97.5%. The tables above quote
  them; the full set is in `analysis/contrasts.csv` and `report_tables/contrasts_compact.csv`.
- Null F means are ≈ 0, as expected. Null F 97.5% is 0.00018–0.00024 for pooled conditions and 0.00043–0.00045
  for A.

**Bootstrap CIs.** These are 2,000-resample paired document bootstrap percentile intervals, given for ΔR, ΔP and
ΔF (e5 and TF-IDF) in every contrast and seed.

**Raw per-checkpoint outputs.** `analysis/raw/<key>.npz` holds the bootstrap and null draws.

## 9. Excluded-expert coverage

**Blocks 1, 4, 7, e5 K = 32 and TF-IDF K = 32, all conditions and seeds.**
- Experts: 96 of 96 per condition and layer were eligible (32 × 3 seeds).
- Excluded for zero load: 0. Excluded for insufficient support (< 32 windows in T or O): 0.
- Share of selected (token, expert) events covered by eligible experts: 1.000.

**Routing metrics.** No zero-load or low-support experts in any checkpoint at any of the 7 layers.

Table: `report_tables/coverage_primary.csv`.

## 10. Compute, VRAM, disk, wall time

**GPU time** (`ledger/gpu_time_ledger.tsv`, entries matching `specaudit`/`spec_audit`): **8,428 GPU-s = 2.34
GPU-h** against the 4-GPU-hour ceiling.

| Component | GPU-s |
|---|---|
| Native segmentation | 40 |
| e5 embedding | 54 |
| Benchmark | 32 |
| Aborted s0_A extraction (VRAM guard) | 260 |
| 18 extraction + counterfactual runs (435–452 s each) | 8,042 |

**Capping.** The projection was 2.48 GPU-h for all 7 layers, so no window cap was applied. The all-7-layer
secondary analysis was run.

**VRAM.**
- Completed extraction runs peaked at 10.32 GiB reserved (8.7 GiB allocated) against the 56 GiB ceiling.
- The aborted first s0_A attempt reached 53.48 GiB reserved (17.28 GiB allocated) and was stopped by the in-process
  guard at 95% of the ceiling, before any CUDA OOM.
- Peaks for segmentation and embedding were not recorded separately; the guard's abort threshold did not fire for
  them.

**CPU.**
- Analysis: 2,507 s wall, 8 workers, OMP = 1, CUDA hidden.
- Single-checkpoint smoke: 614 s.
- Tests: 136/136 full suite before GPU work; 26/26 audit file after extraction.

**Wall time.** About 3.8 h, from the first audit action (≈11:27) to the analysis finishing (15:07, 2026-11-01).

**Disk.**

| Item | Size |
|---|---|
| New run directories (22) | 15.40 GB |
| `artifacts/specialization-audit-v1/` | 0.64 GB |
| e5 encoder | 0.44 GB |
| Raw audit parquet | 0.025 GB |

Free space on `/` stayed ≥ 123 GiB, with checks before every GPU run. It was 123 GiB at the end.

**GPU safety.**
- GPU 0 only, with a fresh no-foreign-process check before every GPU command.
- One project GPU process at a time.
- GPU 0 was idle (0 processes) at the end.

**Network.** All network access is ledgered:
- HF range reads of the pinned FineWeb-Edu parquet, 12 row groups;
- Hub `model_info` and the download of e5-base-v2 @ f52bf8ec (438,899,505 bytes).

## 11. Paths and hashes

**Preregistration**

| Path | SHA-256 |
|---|---|
| `docs/SPECIALIZATION_AUDIT_V1_PREREG.md` (frozen) | `17998b104ac37e25d7d0e75f699daededc44070714d03ae42c515262229c5daa` |

**Analysis outputs**

| Path | SHA-256 |
|---|---|
| `artifacts/specialization-audit-v1/analysis/analysis.json` | `149272e88424fb39ac5e7f5b6288cfcb7a8d8c0ec772754043c67d8ff14e849a` |
| `artifacts/specialization-audit-v1/analysis/contrasts.csv` | `5c83b2dbe0865a7b…` |
| `artifacts/specialization-audit-v1/analysis/cross_seed.csv` | `aafc64e38edfbfac…` |
| `artifacts/specialization-audit-v1/analysis/functional_experts.csv` | `15a33ffefb352edd…` |
| `artifacts/specialization-audit-v1/analysis/functional_summary.csv` | `add4c020f094ffa8…` |
| `artifacts/specialization-audit-v1/analysis/probe.csv` | `c7f7547669d644c3…` |
| `artifacts/specialization-audit-v1/analysis/routing_metrics.csv` | `5627026932bb5fd2…` |
| `artifacts/specialization-audit-v1/report_tables/per_checkpoint.csv` | `14eb04ff453ca5abb650f5aed371eaac8c7b36927fd0b8768080b6a2c89deea8` |
| `artifacts/specialization-audit-v1/report_tables/contrasts_compact.csv` | `355d9741f4e00c42d013a0c8b7b5b7b6eb938d4d610a3c44b36197185e235c5a` |
| `artifacts/specialization-audit-v1/report_tables/` (other files) | per_condition `577062ec…`, primary_layer_functional `ee6100ea…`, descriptive_scaled_F `9066ccc1…`, coverage_primary `ea9a902f…`, summary.json `b068fe66…` |

**Inputs, data and labels**

| Path | SHA-256 |
|---|---|
| `artifacts/specialization-audit-v1/inventory.json` | `030337b1d5aaff0a…` |
| `artifacts/specialization-audit-v1/projection.json` | `3e1501e4ff45e6b3…` |
| `artifacts/specialization-audit-v1/compute_runs.csv` | `bf6b5e9fb151caa7…` |
| `artifacts/specialization-audit-v1/data/manifest.json` | `e4d9b8d174a66645…` |
| `artifacts/specialization-audit-v1/labels/labels.npz` | `610037c1d126f557…` |
| `artifacts/specialization-audit-v1/labels/encoder_embeddings.npy` | `b8ec6417086858f4…` |
| `artifacts/data/raw/fwe_10BT_000_specaudit_v1_rg12_seed20261101.parquet` | `85045d9a…` (full hash in the download manifest) |

**Code**

| Path | SHA-256 |
|---|---|
| `spec_audit_inventory.py` | `874558a7…` |
| `spec_audit_data.py` | `b7920327…` |
| `spec_audit_segments.py` | `e6d61c02…` |
| `spec_audit_labels.py` | `3c07d7c3…` |
| `spec_audit_extract.py` | `37182516…` |
| `spec_audit_analyze.py` | `89aba552…` |
| `spec_audit_tables.py` | `d7954b29…` |
| `src/csr/specaudit.py` | `846a3640…` |
| `src/csr/audit.py` | `1dd56c24…` |
| `scripts/spec_audit_main.sh` | `577993c9…` |
| `tests/test_spec_audit_v1.py` | `3c94ae41…` |

**Runs.**
- Extraction runs: `runs/20261101T1*_specaudit_v1_extract_s{0,1,2}_{A..F}`, listed in `.tmp/spec_audit_v1/runs.tsv`.
- Also preserved:
  - two CPU smokes (`…_s0_A_cpusmoke`, `…_s0_E_cpusmoke`);
  - the benchmark (`…_s0_E_bench`);
  - the aborted run `runs/20261101T165452Z_specaudit_v1_extract_s0_A`.
- Per-run seconds, VRAM, bytes and the parameter-fingerprint check are in `compute_runs.csv`. The fingerprints
  before and after are identical in all 18 runs.

**Checkpoints.** All 18 checkpoint sha256 values match `inventory.json` after the audit
(`.tmp/spec_audit_v1/post_ckpt_verify.txt`).

**csr-representative-v1.** `artifacts/rep-v1/analysis.json` (sha256 `b3764237…1d7c35`) and every other
csr-representative-v1 file are unmodified: nothing under `artifacts/rep-v1`, `runs/*_train_rep_v1_*` or
`.tmp/rep_v1_main` is newer than the audit start.

## 12. Limitations, deviations and incidents

**Status and design.**
- **Post-hoc framing.** The audit was designed after the CE results were known. It was frozen before any audit
  result existed, but it is not part of the csr-representative-v1 preregistration and cannot alter it.
- **Three seeds.** The decision rests on sign consistency across 3 seeds plus per-seed CIs and nulls. The document
  bootstrap measures audit-document uncertainty only, **not** training-seed uncertainty.
- **Small absolute functional effects.** F differences are 0.0005–0.0016 nats, concentrated in the last MoE layer.
  They are statistically clear on this audit set, but they are about two orders of magnitude smaller than the CE
  differences between conditions.

**Measurement.**
- **Window-level labels.** Each token inherits its window's topic. Pooled routing shares candidate pools within a
  chunk, which mechanically favours agreement with window labels. This affects how far R can be compared across
  routing types (B − A); it does not affect the within-pooling comparisons (E − D, C − B, D − F).
- **Cross-seed matched-JSD null.** The frozen load-preserving null cannot support a "lower JSD than chance" claim,
  because permuted labels flatten every expert profile. It was reported as frozen and not replaced.
- **Counterfactual scope.** Single-token interventions, with other tokens' keys and values held at their original
  values. This is exact for each token's own prediction, but it does not measure effects on later tokens.
- **Precision.** The audit ran in fp32 with TF32 off, while the rep-v1 evaluation was bf16. Audit fp32 CE
  reproduces the validation ordering, but the values are not identical.

**Data.**
- **Test-split protection.** This relies on source row-group disjointness only. No near-duplicate check was run
  against test content, because the locked test split was never read, hashed, stat'ed or otherwise accessed.
  Deduplication was done against non-test splits only.
- **Single labelers.** The labels come from one neural encoder (e5) and one lexical view (TF-IDF). Neither is the
  training teacher, but "topic" is defined by these clusterings.

**Deviations and incidents.**
- **VRAM abort.** The first s0_A extraction (`runs/20261101T165452Z_specaudit_v1_extract_s0_A`) was stopped by the
  project's VRAM guard at 53.48 GiB reserved. The cause was allocator fragmentation at rows-per-batch 16. There was
  no CUDA error or OOM, so the prereg stop rule for a CUDA error/OOM did not trigger.
  - The run is preserved and ledgered (`SPEC_AUDIT_VRAM_ABORT`), and its 260 GPU-s count toward the budget.
  - It was relaunched with rows-per-batch 8 and `expandable_segments`. This is an execution-batching change; it is
    numerically equivalent and was verified against brute force in the tests.
- **`version.txt` incident.** Earlier in this session, one run of the project venv Python without `env.sh` caused
  a transformers import to create `~/.cache/huggingface/hub/version.txt` (1 byte) outside the project. That single
  file was removed and the event ledgered (`INCIDENT`).
- **Scratchpad smoke.** The single-checkpoint analysis smoke ran in the session scratchpad, outside the project,
  using symlinks to project inputs. It wrote one 29.5 MB npz there. Those scratch files and symlinks were removed
  afterwards (link targets untouched) and the cleanup ledgered (`SPEC_AUDIT_SCRATCH_CLEANUP`).
- **Post-hoc descriptive tables.** `report_tables/`, including the scaled-F check in §5, is derived after the
  decision and is labelled descriptive. No frozen statistic was recomputed or altered.

**Not run.** Nothing in the frozen plan was skipped. The all-7-layer secondary analysis, the K = 16/64 sensitivity
analyses, the pool-level metrics and the probe controls were all run.

SPECIALIZATION AUDIT COMPLETE
