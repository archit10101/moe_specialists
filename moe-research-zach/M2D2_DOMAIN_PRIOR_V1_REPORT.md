# csr-m2d2-domain-prior-v1 — report (screening pilot)

**Verdict.** B2 (stable domain prior plus unrestricted top-4) did **not** reliably beat B1 (TOKEN-4 given the same
domain embedding):
- seed 0: B2−B1 = +0.0001 [−0.0003, +0.0006];
- seed 1: B2−B1 = −0.0033 [−0.0038, −0.0028];
- 10M search seed: −0.0025.

The preregistered rule needs the CI to exclude 0 in both final seeds, so it is not met. The effect is also at or below
the ~0.003-nat noise level, and far smaller than the seed-to-seed spread (0.085 for B2).

Domain information itself helped. The shuffled-label control B3 is worse than B2 in every seed (+0.018, +0.012,
+0.011), so true labels carry roughly 0.01–0.02 nats/token of real value. Most of the large seed-0 gain over B0
(−0.077) is **not** semantic: B3 reproduces 77% of it with meaningless labels. Specialisation rose modestly under the
prior, but the routing decisions the prior changed show no specific accuracy gain.

This is screening-quality evidence from two seeds at 43M tokens per run, not a confirmation.

Protocol: `docs/M2D2_DOMAIN_PRIOR_V1_PROTOCOL.md` (sha256 `3ce1c23d…1acc`, hashed before any data was built).
Machine-readable: `artifacts/m2d2-domain-prior-v1/final_analysis.json`.

## 1. Data (machelreid/m2d2 @ 6ae613ba744ee56f645aee9c577d04f7e3d48b30, CC-BY-NC-4.0)
- **Download.** Only the six L1 archives (2.87 GB), each sha256-verified against its LFS hash
  (`artifacts/m2d2-domain-prior-v1/download_manifest.json`).
- **Documents and dedup.** One document = one Wikipedia article (split on ≥ 3 newlines). Articles under 64 tokens
  were dropped. Exact de-duplication on normalised text across all domains and the train/valid splits dropped
  **every** copy of 364,630 repeated hashes. Many articles belong to several L1 categories; for example, Culture's
  native train went from 395,334 to 109,121 articles.
- **Splits.**
  - VALIDATION = native valid.
  - CALIB-SEARCH = native-train articles with u < 0.05 (document hash).
  - TRAIN = the rest of native train.
  - Documents were taken in hash order; no split shares a hash (tested).
- **Balanced caps.**
  - Mathematics_and_logic could not reach the 10M-token train cap (14,058 usable articles) or the 0.5M
    calib-search cap. Per protocol, every domain was downsampled to the common amount:
    - **train 7,304,098 tokens per domain (43.83M total, 79,707 docs);**
    - **calib-search 389,375 per domain (2.34M, 4,234 docs);**
    - **validation 1,000,000 per domain (6.00M, 13,346 docs).**
  - Tokenisation: Pythia tokenizer, tokens[:1023] + EOS (22–31% of tokenised train articles truncated, by domain).
- **Packing.** Every packed row holds a single domain (tested on the real data).
- **Hashes** (sha256 of the ordered document-hash list):
  - train `70d9f798…`
  - calib-search `9fff11f3…`
  - val `685c8e5f…`
  - splits manifest `af9d90f4d776d057…`

  The full hashes are in `artifacts/m2d2-domain-prior-v1/data/manifest.json`.
- **Locked test.** The six `test.txt` members were sha256-hashed as raw bytes while streaming and never decoded,
  split or tokenised (`manifest.json → locked_test`, `decoded: false`). The loaders refuse `test` before any file
  access. Because test is never decoded, there is no dedup against test (stated in the protocol).

## 2. Conditions and matching
- **Backbone.** The existing 28.9M NanoMoE: d 192, 8 layers, layer 0 dense, layers 1–7 MoE, 32 experts, top-4,
  shared expert, tied embeddings, context 1024. Total parameters: 28,873,920 for B0, 28,875,844 for B2/B3.
- **Added parameters.**
  - B1: a 6 × 192 domain embedding (1,152 parameters).
  - B2/B3: the same embedding, plus a 4 × 6 × 32 router bias initialised at 0, plus a per-layer alpha (init 0.5) in
    layers 4–7. That is 1,924 parameters, 0.0067% of the total, within the 1% overhead limit.
  - Active compute is identical apart from one embedding add and one 32-wide bias add.
- **B3.** Training labels were permuted across training documents (seed 4242, frequencies preserved); evaluation
  used the true labels.
- **Step-0 checks** (`artifacts/m2d2-domain-prior-v1/gpu_checks_*.json`):
  - identical backbone hash across B0–B3 for each seed;
  - B0 equals TOKEN-4 exactly (diff 0.0);
  - B2 equals B1 and B3 equals B2 exactly at step 0.
- **Resume check.** Passed with a caveat: masked-SDPA backward nondeterminism; see `resume_check.json`.
- **Matching across conditions.** Identical documents, order (seed 17), steps, token count, optimiser (AdamW 6e-4,
  WSD, 32,768 tokens per step) and evaluation for every condition. Initialisation is identical within a seed.
- **Budgets.**
  - Search: 10.0M tokens (389 steps), seed 100.
  - Final: **43.0M tokens (1,671 steps, one epoch), seeds 0 and 1**, rather than 50M because only 43.8M balanced
    training tokens exist. This deviation was logged before training (`deviations.jsonl`).
  - All 12 planned runs completed, including the optional 43M B3 control for both seeds.

## 3. Results (validation, nats/token; one evaluation per checkpoint, after `frozen.json`)
**Per run** (overall = token-weighted; macro = mean of the six domain CEs; domains balanced, so they nearly coincide):

| run | overall CE | macro CE | Math | Tech | NatSci | Health | History | Culture | alpha (l4..l7) |
|---|---|---|---|---|---|---|---|---|---|
| final_s0_B0 | 4.5163 | 4.5163 | 4.3804 | 4.6004 | 4.5658 | 4.5340 | 4.4997 | 4.5175 | – |
| final_s0_B1 | 4.4395 | 4.4394 | 4.3045 | 4.5251 | 4.4800 | 4.4611 | 4.4246 | 4.4413 | – |
| final_s0_B2 | 4.4396 | 4.4396 | 4.3037 | 4.5248 | 4.4813 | 4.4615 | 4.4249 | 4.4412 | 1.23, 1.20, 1.17, 1.28 |
| final_s0_B3 | 4.4576 | 4.4576 | 4.3222 | 4.5423 | 4.5020 | 4.4787 | 4.4426 | 4.4579 | 1.30, 1.26, 1.27, 1.32 |
| final_s1_B0 | 4.5361 | 4.5361 | 4.3924 | 4.6187 | 4.5927 | 4.5542 | 4.5237 | 4.5352 | – |
| final_s1_B1 | 4.5284 | 4.5284 | 4.3860 | 4.6102 | 4.5828 | 4.5463 | 4.5169 | 4.5278 | – |
| final_s1_B2 | 4.5250 | 4.5250 | 4.3844 | 4.6067 | 4.5775 | 4.5434 | 4.5130 | 4.5252 | 1.19, 1.24, 1.20, 1.29 |
| final_s1_B3 | 4.5370 | 4.5370 | 4.3959 | 4.6195 | 4.5922 | 4.5543 | 4.5256 | 4.5348 | 1.23, 1.27, 1.29, 1.31 |
| search_s100_B0 | 5.8936 | 5.8936 | 5.7337 | 5.9494 | 6.0075 | 5.9207 | 5.8605 | 5.8899 | – |
| search_s100_B1 | 5.8908 | 5.8908 | 5.7273 | 5.9491 | 6.0025 | 5.9198 | 5.8580 | 5.8881 | – |
| search_s100_B2 | 5.8883 | 5.8883 | 5.7250 | 5.9464 | 6.0003 | 5.9168 | 5.8561 | 5.8853 | 0.68, 0.68, 0.68, 0.68 |
| search_s100_B3 | 5.8995 | 5.8995 | 5.7427 | 5.9548 | 6.0110 | 5.9278 | 5.8648 | 5.8957 | 0.68, 0.68, 0.68, 0.69 |

**Paired per-document bootstrap differences** (b − a; negative = b better; 2,000 resamples overall, 1,000 per domain):

| comparison | Δ overall [95% CI] | Δ macro | Math | Tech | NatSci | Health | History | Culture |
|---|---|---|---|---|---|---|---|---|
| final_s0_B1-B0 | -0.0769 [-0.0780, -0.0757] | -0.0769 | -0.0759 | -0.0753 | -0.0858 | -0.0729 | -0.0751 | -0.0761 |
| final_s0_B2-B0 | -0.0767 [-0.0778, -0.0756] | -0.0767 | -0.0766 | -0.0756 | -0.0845 | -0.0725 | -0.0748 | -0.0762 |
| final_s0_B2-B1 | +0.0001 [-0.0003, +0.0006] | +0.0001 | -0.0007 | -0.0003 | +0.0012 | +0.0004 | +0.0003 | -0.0001 |
| final_s0_B3-B1 | +0.0182 [+0.0176, +0.0188] | +0.0182 | +0.0177 | +0.0172 | +0.0219 | +0.0176 | +0.0180 | +0.0165 |
| final_s0_B3-B2 | +0.0180 [+0.0174, +0.0186] | +0.0180 | +0.0185 | +0.0174 | +0.0207 | +0.0172 | +0.0177 | +0.0166 |
| final_s1_B1-B0 | -0.0078 [-0.0085, -0.0071] | -0.0078 | -0.0064 | -0.0084 | -0.0099 | -0.0079 | -0.0068 | -0.0073 |
| final_s1_B2-B0 | -0.0111 [-0.0118, -0.0104] | -0.0111 | -0.0080 | -0.0119 | -0.0152 | -0.0107 | -0.0107 | -0.0100 |
| final_s1_B2-B1 | -0.0033 [-0.0038, -0.0028] | -0.0033 | -0.0017 | -0.0035 | -0.0053 | -0.0029 | -0.0039 | -0.0026 |
| final_s1_B3-B1 | +0.0087 [+0.0081, +0.0093] | +0.0087 | +0.0098 | +0.0092 | +0.0094 | +0.0080 | +0.0087 | +0.0070 |
| final_s1_B3-B2 | +0.0120 [+0.0114, +0.0126] | +0.0120 | +0.0115 | +0.0127 | +0.0147 | +0.0108 | +0.0126 | +0.0096 |
| search_s100_B1-B0 | -0.0028 [-0.0032, -0.0024] | -0.0028 | -0.0064 | -0.0003 | -0.0050 | -0.0009 | -0.0026 | -0.0019 |
| search_s100_B2-B0 | -0.0053 [-0.0057, -0.0049] | -0.0053 | -0.0087 | -0.0030 | -0.0072 | -0.0038 | -0.0044 | -0.0047 |
| search_s100_B2-B1 | -0.0025 [-0.0026, -0.0023] | -0.0025 | -0.0023 | -0.0027 | -0.0022 | -0.0030 | -0.0018 | -0.0028 |
| search_s100_B3-B1 | +0.0087 [+0.0084, +0.0090] | +0.0087 | +0.0154 | +0.0057 | +0.0085 | +0.0080 | +0.0068 | +0.0076 |
| search_s100_B3-B2 | +0.0112 [+0.0108, +0.0115] | +0.0112 | +0.0177 | +0.0084 | +0.0107 | +0.0110 | +0.0087 | +0.0104 |

**By in-document position** (post-hoc descriptive):

| comparison | pos 0-16 | 16-64 | 64-256 | 256-512 | 512-1024 |
|---|---|---|---|---|---|
| final_s0_B1-B0 | -0.1470 | -0.0569 | -0.0686 | -0.0798 | -0.0823 |
| final_s0_B2-B0 | -0.1428 | -0.0587 | -0.0698 | -0.0783 | -0.0817 |
| final_s0_B2-B1 | +0.0042 | -0.0018 | -0.0012 | +0.0015 | +0.0006 |
| final_s0_B3-B1 | +0.0220 | +0.0185 | +0.0168 | +0.0190 | +0.0183 |
| final_s0_B3-B2 | +0.0178 | +0.0203 | +0.0179 | +0.0175 | +0.0178 |
| final_s1_B1-B0 | -0.0379 | -0.0073 | -0.0058 | -0.0072 | -0.0070 |
| final_s1_B2-B0 | -0.0375 | -0.0061 | -0.0096 | -0.0118 | -0.0107 |
| final_s1_B2-B1 | +0.0004 | +0.0012 | -0.0038 | -0.0046 | -0.0037 |
| final_s1_B3-B1 | +0.0152 | +0.0088 | +0.0071 | +0.0093 | +0.0092 |
| final_s1_B3-B2 | +0.0148 | +0.0076 | +0.0109 | +0.0139 | +0.0129 |
| search_s100_B1-B0 | -0.0238 | -0.0060 | -0.0014 | -0.0018 | -0.0015 |
| search_s100_B2-B0 | -0.0261 | -0.0088 | -0.0039 | -0.0042 | -0.0040 |
| search_s100_B2-B1 | -0.0023 | -0.0028 | -0.0025 | -0.0024 | -0.0024 |
| search_s100_B3-B1 | +0.0092 | +0.0116 | +0.0088 | +0.0082 | +0.0078 |
| search_s100_B3-B2 | +0.0115 | +0.0144 | +0.0112 | +0.0106 | +0.0103 |

**Specialisation diagnostics** (validation; MI in nats, maximum log 6 = 1.79; null = load-preserving domain-label permutation, 20 draws):

| run | MI(domain;expert) l1..l7 | mean MI l4-7 | null max | mean H(domain|expert) | min/max unused | max load share | adj. Jaccard l4-7 | top-4 changed by prior l4-7 | MI w/o prior l4-7 |
|---|---|---|---|---|---|---|---|---|---|
| final_s0_B0 | 0.0140, 0.0118, 0.0108, 0.0166, 0.0301, 0.0121, 0.0193 | 0.0195 | 0.00015 | 1.775 | 0 | 0.041 | 0.202 | – | – |
| final_s0_B1 | 0.0129, 0.0051, 0.0049, 0.0113, 0.0182, 0.0240, 0.0168 | 0.0176 | 0.00020 | 1.779 | 0 | 0.041 | 0.198 | – | – |
| final_s0_B2 | 0.0083, 0.0046, 0.0055, 0.0108, 0.0262, 0.0259, 0.0205 | 0.0209 | 0.00018 | 1.777 | 0 | 0.041 | 0.201 | 0.060, 0.072, 0.060, 0.086 | 0.0094, 0.0221, 0.0227, 0.0177 |
| final_s0_B3 | 0.0120, 0.0052, 0.0082, 0.0115, 0.0270, 0.0183, 0.0170 | 0.0184 | 0.00015 | 1.778 | 0 | 0.043 | 0.197 | 0.039, 0.035, 0.035, 0.050 | 0.0114, 0.0269, 0.0185, 0.0165 |
| final_s1_B0 | 0.0174, 0.0171, 0.0152, 0.0101, 0.0080, 0.0126, 0.0284 | 0.0147 | 0.00014 | 1.777 | 0 | 0.046 | 0.194 | – | – |
| final_s1_B1 | 0.0157, 0.0178, 0.0154, 0.0105, 0.0118, 0.0214, 0.0189 | 0.0157 | 0.00013 | 1.776 | 0 | 0.043 | 0.189 | – | – |
| final_s1_B2 | 0.0138, 0.0210, 0.0213, 0.0173, 0.0082, 0.0285, 0.0172 | 0.0178 | 0.00016 | 1.774 | 0 | 0.042 | 0.190 | 0.082, 0.057, 0.064, 0.077 | 0.0128, 0.0067, 0.0250, 0.0146 |
| final_s1_B3 | 0.0130, 0.0154, 0.0249, 0.0088, 0.0110, 0.0262, 0.0251 | 0.0178 | 0.00018 | 1.774 | 0 | 0.041 | 0.197 | 0.045, 0.036, 0.035, 0.055 | 0.0087, 0.0107, 0.0266, 0.0246 |
| search_s100_B0 | 0.0150, 0.0116, 0.0173, 0.0123, 0.0176, 0.0139, 0.0135 | 0.0143 | 0.00011 | 1.777 | 0 | 0.046 | 0.180 | – | – |
| search_s100_B1 | 0.0116, 0.0163, 0.0209, 0.0165, 0.0161, 0.0229, 0.0193 | 0.0187 | 0.00015 | 1.774 | 0 | 0.042 | 0.176 | – | – |
| search_s100_B2 | 0.0115, 0.0181, 0.0238, 0.0149, 0.0206, 0.0214, 0.0221 | 0.0198 | 0.00014 | 1.773 | 0 | 0.042 | 0.180 | 0.042, 0.046, 0.044, 0.041 | 0.0149, 0.0199, 0.0205, 0.0209 |
| search_s100_B3 | 0.0078, 0.0142, 0.0124, 0.0137, 0.0204, 0.0211, 0.0155 | 0.0177 | 0.00016 | 1.777 | 0 | 0.041 | 0.183 | 0.029, 0.029, 0.030, 0.029 | 0.0131, 0.0204, 0.0209, 0.0153 |

**Changed versus unchanged routing decisions** (tokens whose top-4 the prior changed in any of layers 4–7; Δ = b − a on the same tokens):

| comparison | frac changed | Δ on changed tokens | Δ on unchanged tokens |
|---|---|---|---|
| final_s0_B2-B0 | 0.250 | -0.0799 | -0.0757 |
| final_s0_B2-B1 | 0.250 | +0.0006 | -0.0000 |
| final_s0_B3-B1 | 0.150 | +0.0181 | +0.0182 |
| final_s0_B3-B2 | 0.150 | +0.0181 | +0.0180 |
| final_s1_B2-B0 | 0.251 | -0.0119 | -0.0108 |
| final_s1_B2-B1 | 0.251 | -0.0039 | -0.0031 |
| final_s1_B3-B1 | 0.161 | +0.0085 | +0.0087 |
| final_s1_B3-B2 | 0.161 | +0.0126 | +0.0119 |
| search_s100_B2-B0 | 0.162 | -0.0048 | -0.0054 |
| search_s100_B2-B1 | 0.162 | -0.0025 | -0.0025 |
| search_s100_B3-B1 | 0.112 | +0.0093 | +0.0086 |
| search_s100_B3-B2 | 0.112 | +0.0114 | +0.0111 |

**Functional ablation** (expert with the highest domain share per layer 4–7, gate zeroed for every token; ΔCE on its top domain / on the other domains):

| run | layer: expert (top domain, share) ΔCE top domain / other domains |
|---|---|
| final_s0_B0 | l4: e27 (Math, 0.37) +0.0015 / +0.0003; l5: e14 (Math, 0.52) +0.0078 / +0.0000; l6: e14 (Math, 0.31) +0.0012 / +0.0004; l7: e22 (Math, 0.33) +0.0087 / +0.0018 |
| final_s0_B1 | l4: e3 (Math, 0.25) +0.0011 / +0.0005; l5: e11 (Math, 0.30) +0.0031 / +0.0005; l6: e14 (Math, 0.54) +0.0083 / +0.0001; l7: e7 (Math, 0.27) +0.0039 / +0.0016 |
| final_s0_B2 | l4: e3 (Math, 0.24) +0.0009 / +0.0005; l5: e14 (Math, 0.34) +0.0019 / +0.0003; l6: e14 (Math, 0.55) +0.0072 / +0.0001; l7: e7 (Math, 0.32) +0.0045 / +0.0008 |
| final_s0_B3 | l4: e29 (Math, 0.29) +0.0009 / +0.0004; l5: e14 (Math, 0.33) +0.0021 / +0.0002; l6: e14 (Math, 0.45) +0.0052 / +0.0003; l7: e24 (Math, 0.27) +0.0014 / +0.0005 |
| final_s1_B0 | l4: e18 (Math, 0.26) +0.0019 / +0.0005; l5: e4 (Math, 0.24) +0.0021 / +0.0007; l6: e20 (Math, 0.33) +0.0031 / +0.0008; l7: e30 (NatSci, 0.35) +0.0081 / +0.0026 |
| final_s1_B1 | l4: e11 (Math, 0.27) +0.0033 / +0.0006; l5: e18 (Culture, 0.30) +0.0040 / +0.0010; l6: e19 (Math, 0.47) +0.0061 / +0.0003; l7: e29 (Math, 0.33) +0.0105 / +0.0014 |
| final_s1_B2 | l4: e26 (Culture, 0.38) +0.0032 / +0.0005; l5: e18 (Culture, 0.25) +0.0021 / +0.0008; l6: e19 (Math, 0.52) +0.0106 / +0.0005; l7: e5 (Math, 0.27) +0.0044 / +0.0019 |
| final_s1_B3 | l4: e20 (Math, 0.25) +0.0008 / +0.0003; l5: e4 (Math, 0.36) +0.0035 / +0.0005; l6: e19 (Math, 0.59) +0.0086 / +0.0002; l7: e23 (Culture, 0.33) +0.0048 / +0.0013 |

### Seed spread (final, 43M)
| condition | seed 0 | seed 1 | range |
|---|---|---|---|
| B0 | 4.5163 | 4.5361 | 0.020 |
| B1 | 4.4395 | 4.5284 | 0.089 |
| B2 | 4.4396 | 4.5250 | 0.085 |
| B3 | 4.4576 | 4.5370 | 0.079 |

## 4. Interpretation (preregistered rules)
- **Did B2 beat B1?** No, not by the preregistered rule (CI excluding 0 in both seeds).
  - Seed 0 is null (+0.0001).
  - Seed 1 (−0.0033) and the 10M search seed (−0.0025) are small and significant within a seed.
  - That 0.003 magnitude matches the project's known run-to-run noise and is 25× smaller than the seed spread.
  - At most this is weak, inconsistent evidence for a ≈0.003-nat routing-prior effect. It is not established.
- **Did B2 beat B0?** Yes in every seed (−0.077, −0.011, −0.005). By the fixed rule ("B2 beats B0 but not B1"),
  domain information helped; persistent domain-biased routing did not add reliable value.
- **Semantic versus generic.**
  - B3 (same architecture and init as B2, meaningless training labels) is worse than B2 by +0.018 / +0.012 /
    +0.011 and worse than B1 by +0.018 / +0.009 / +0.009, with CIs excluding 0 in every seed. So meaningful
    labels matter, by ~0.01–0.02 nats.
  - But B3 − B0 = −0.059 in seed 0: most of seed 0's apparent B1/B2 gain over B0 is shared by a model whose labels
    are noise. In seed 1, B3 ≈ B0 (+0.0009).
  - The seed-0 gap opens gradually between ~20M and 35M tokens and is nearly uniform over document positions and
    domains. That fits an initialisation- or trajectory-dependent effect of the extra input pathway at a budget
    where CE is still falling steeply, not domain knowledge.
  - The raw B1−B0 and B2−B0 numbers should therefore not be read as the value of domain information. B3-versus-B1/B2
    is the cleaner estimate.
- **Did specialisation increase?** Modestly.
  - Domain–expert mutual information is small in every model: 0.005–0.03 nats against a maximum of 1.79.
  - It is 50–200× the load-preserving shuffled-domain null (≤ 2e-4), so it is real.
  - B2's mean MI in layers 4–7 exceeds B1's in all three seeds: 0.0209 vs 0.0176; 0.0178 vs 0.0157; 0.0198 vs 0.0187.
  - Measured on the same hidden states with the bias removed, the prior itself adds ≈ 0.001–0.004 nats of MI per
    layer. It changes the top-4 of 6–9% of tokens per layer (25% of tokens in at least one layer) at 43M tokens.
  - Alpha grew from 0.5 to 1.17–1.31, so the prior was used, not suppressed.
  - B0 already shows domain-selective experts: disabling its most Mathematics-heavy expert raises Mathematics CE by
    up to +0.0087 against ≤ +0.0026 on other domains. Under the prior the largest effect was +0.0072 (s0) and +0.0106 (s1), not clearly larger.
- **Did that specialisation improve accuracy?** No evidence. On the tokens whose routing the prior changed, B2−B1 is
  +0.0006 (s0), −0.0039 (s1) and −0.0025 (search). On unchanged tokens it is −0.0000, −0.0031 and −0.0025. The
  changed decisions are no better than the untouched ones, so whatever B2−B1 difference exists is not attributable
  to the re-routed decisions. More specialisation without lower CE is not an accuracy success.
- **Expert health.** No run has an unused expert in any layer; the maximum expert load share is 0.046 (uniform
  would be 0.031). No collapse occurred.

## 5. Limitations
- Two seeds at 43M tokens (one epoch of the balanced subset) on a 28.9M-parameter model. The seed-to-seed spread
  (0.02–0.09) dwarfs the effects of interest, and the B0/B1 trajectories had not converged.
- Strict exact dedup (dropping every copy of multi-category articles) shrinks the data and favours articles with a
  single category. No near-duplicate pass was run.
- The prior acts only through 4 × 6 × 32 biases. Other parameterisations (learned per-chunk priors, predicted
  domains) were not tested. The predicted-domain secondary analysis was not run.
- The functional ablation covers only one expert per layer (the most domain-concentrated one).

## 6. Integrity
- One GPU job at a time via `scripts/gpu_guarded.sh`; occupancy was checked before every launch. GPU time is in
  `ledger/gpu_time_ledger.tsv`; totals are in the final audit JSON.
- Validation was opened only after `frozen.json`, once per checkpoint (`eval_registry/`). The 10M search runs select
  nothing. The locked test was never decoded.
- No existing artifact was modified (protected-inventory verification). Deviations are in
  `artifacts/m2d2-domain-prior-v1/deviations.jsonl`.
- Driver: `vm_driver.py` (state `.tmp/m2d2-domain-prior-v1/driver_state.json`, log `.tmp/m2d2-domain-prior-v1/driver.log`).

### Integrity numbers (final audit `artifacts/m2d2-domain-prior-v1/final_audit_20261109T091820Z.json`)
- **GPU time (this session, all jobs including checks, failures, warm-ups and benchmarks):** 5,088 s = 1.41 GPU-h
  in total: Experiment A 227 s, Experiment B 4,833 s, shared GPU checks 28 s.
- **Wall time:** about 1 h 45 min (02:33 to 04:18 EST); all GPU work ended at 04:15, before every deadline.
- **Peak VRAM** (allocator):
  - A: 20.7 GiB in the 8192 training run, 26.2 GiB in the GPU check step that kept its graph;
  - B: 22.0 GiB;
  - post-hoc prefill: 3.5 GiB.
- **Disk:** 128.8 GiB free at the end.
- **Tests:** CPU tests ran with CUDA hidden. The full suite passed (620 tests); the new `tests/test_vrc_m2d2_v1.py`
  has 16 tests, all passing. GPU checks passed (`gpu_checks_20261109T*.json`).
- **Protected inventory:** 3,004 pre-existing files verified unchanged.
- **Incomplete runs:** none. The two INTERRUPTED run directories are the deliberate `--max-steps 4` halves of the
  resume-equivalence checks.
- **Locked test:** untouched. No test tokens were written; the M2D2 test members were never decoded; the
  fwe-rep-v1 test files were never opened.
