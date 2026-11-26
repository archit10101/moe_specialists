# Forecast-Latent Expert Divergence (FL-EDL) v1 — Report

Experiment `csr-forecast-latent-edl-v1` (file prefix `fle_`), run autonomously on 2026-11-14 from 02:50 to about 20:30
EST on one A100 80GB. Artifacts are in `artifacts/forecast-latent-edl-v1/`. Final run directories are
`runs/*_forecast_latent_edl_v1_final_*`.

## 1. Summary

FL-EDL keeps accuracy: mean FL − TOKEN-4 is −0.0003 nats/token over 3 seeds, inside the frozen +0.01 bar. It does not
make experts more specialised: the FL/TOKEN-4 FSI ratio is 0.976 (domain) and 0.954 (e5 cluster), against a 1.15
threshold.

- **It loses the decisive comparison.** Past-only latent EDL beats FL-EDL in all three seeds (+0.0015, +0.0061,
  +0.0071; mean +0.0049). PAST is also at least as specialised as FL.
- **The forecasts add nothing usable.** The ensemble produces semantically coherent latent domains. But before any
  model was trained, they were already *less* predictive of the upcoming region than clusters built from the causal
  prefix alone.
- **The learning signal was very weak.** In the selected configuration the FL-EDL gradient was about 0.015 % of the LM
  gradient. That is roughly 100× weaker than the explicit-label EDL gradient and about 200× weaker than the
  load-balancing gradient.
- **Only explicit labels worked.** Explicit-label EDL is the only condition that both specialised (FSI 2.1× /
  1.8× TOKEN-4, per-expert interaction 1.7×) and improved CE (−0.0105).

Verdict (section 12): **4. FL-EDL preserves accuracy but does not specialise meaningfully.**

## 2. What was tested

**Question.** Can a causal ensemble of frozen generalist draft LMs discover label-free latent domains that make MoE
experts functionally more specialised than TOKEN-4, at comparable CE?

- The soft assignments are balanced and are used only in a training-time EDL-style divergence on population routing
  distributions.
- The final model keeps unrestricted per-token top-4 routing, the shared expert and all 32 experts for every token. It
  has no drafts, encoder or clusterer at inference.
- The decisive comparison is FL-EDL vs past-only latent EDL.

**Preregistration.** `docs/FORECAST_LATENT_EDL_V1_PREREG.md`
(sha256 `8f5439dd45f3d226b86e20e82d847c3d0601f5a92b88c4cbdc32fbf056264a48`, mode 0400). It was written and hashed at
03:00 EST, before any draft, cache or candidate existed. The hash is also stored in
`preregistration_hashes.json` and in a `.sha256` file. Every later change is listed in `deviations.jsonl` (section 10).

### 2.1 Data, backbone, recipe (unchanged from lfr v1)

**Data.**
- fspv2 five-domain M2D2 arXiv data: cs, math, physics, q-bio, stat (splits_sha256 `efcde27e…`).
- Pythia tokenizer, segments of at most 1024 tokens.
- Paper-disjoint buckets: main, CALIB-SEARCH, CALIB-SELECTION, VAL and a locked test split. **The test split was never
  opened.**

**Training stream.**
- lfr balanced main stream, 64×1024 rows per step.
- FINAL = 245,032,457 tokens = 3880 steps (plan `f4bc3c935a47…`).
- Search milestones 5M / 20M / 50M = steps 80 / 318 / 793 of the same plan.

**Model.** `LFRModel` with the linear router:
- 12 layers, d 384; layer 0 dense.
- Layers 1–11 have 32 routed GELU experts (FFN 256) plus one always-active shared expert (FFN 768).
- Unrestricted top-4 routing with softmax gates, dropless.
- **103,609,728 parameters (43,054,464 active) in every condition**; the FD head is a separate training-only module.

**Optimizer and recipe.**
- AdamW (0.9, 0.95), weight decay 0.1.
- lr 6e-4 WSD schedule: 1 % warm-up, last 20 % decays to 6e-5.
- Gradient clip 1.0, bf16, torch.compile.
- Switch load-balancing loss 0.01, z-loss 0.001.

**Seeds and matching.**
- Search seed 100; final seeds 0 and 1, plus seed 2 by rule.
- Within a seed, every condition has the same initialisation and the same data order (test T11).

**Trainer.** `fle_train.py` is lfr_train's loop with the teacher code removed.
- TOKEN-4 and EDL run the same forward and loss calls as lfr.
- FL-EDL adds `collect=True` (router probabilities returned) and a loss on them.

### 2.2 Causal forecast evidence (training only)

**Drafts: 4 generalists with no domain identity.**
- G0 is the fspv2 shared draft, reused read-only (sha `9cafd996…`).
- G1–G3 are new seeds 11/12/13 of the same 21.4M-parameter dense architecture. Each was trained for one epoch on the
  balanced union draft_shared ∪ draft_cont (250M tokens), which is paper-disjoint from main, calib, val and test.
- Final training LM loss was 3.18 / 3.25 / 3.26 for G1–G3. The fspv2 domain drafts were not used.

**Regions.**
- Region j of a segment covers positions [jL, (j+1)L), with L ∈ {128, 256}.
- The forecast point is s_j = jL, with j ≥ 1 and a minimum region length of 32 tokens.
- The prefix is tokens [0, s_j) of the same segment: exactly the main model's causal context. Region 0 gets no
  assignment.

**Generation.**
- Each draft gets the same prefix and samples 32 tokens: temperature 0.8, top-k 50.
- Noise is counter-based Gumbel, keyed by draft, document and point, so results do not depend on batching.
- Test T1 shows that changing tokens at or after s_j leaves the continuation bit-identical, and that the cached
  generation equals naive incremental generation.

**Cache contents** (immutable, with a manifest per cache: `cache/{calib_search,search_main,final_main,val_audit}/manifest.json`):
- 32 token ids per region and draft;
- per-step token probability, entropy and top-1;
- MiniLM embeddings of the 32- and 16-token continuations;
- the MiniLM embedding of the last 192 prefix tokens, which is the past-only representation.

**Encoder.** all-MiniLM-L6-v2, frozen. It is independent of the drafts, the main model and the e5-base-v2
specialisation evaluator.

### 2.3 Aggregation and balanced latent domains (frozen before any main-model training)

- **A1 meanstat:** normalize([normalize(mean_d e_d), 0.25·standardised(disagreement, mean entropy, mean confidence,
  mean top-1, entropy spread)/√5]).
- **A2 deepsets-cv:** φ/ρ MLPs, mean-pooled over drafts. Trained only on training regions with a forecast-only
  cross-view InfoNCE: two disjoint draft pairs per region.
- **Clustering:** balanced Sinkhorn spherical soft k-means on 100k training regions.
  - The frozen per-region assignment is q = softmax(x·C/τ + b), where b are the Sinkhorn column potentials.
  - τ is chosen on training data so that mean normalised entropy ≈ 0.4.
  - No gradient flows into q, the encoder, the aggregator or the prototypes (test T4).

### 2.4 FL-EDL and the comparators

For each micro-batch and each layer l in Λ:
- r_i = mean router softmax over region i's tokens;
- R_k = Σ_i q_ik r_i / Σ_i q_ik;
- L_FL-EDL = mean over l and pairs k<j of −log(JSD(R_k, R_j) + 1e-8). This is the lfr EDL formula, with JSD clamped
  at 0 (deviation 6).

The total loss is L = L_LM + 0.01 L_Switch + 0.001 L_z + β L_FL-EDL. Routing itself is untouched: no masks, slots,
mappings or chunk routing.

| condition | definition |
|---|---|
| T4 | TOKEN-4, no auxiliary loss |
| EDL | Explicit-label EDL = lfr condition B: genuine domain labels, per-segment means, all 11 layers, β 5e-3 (lfr CALIB-SEARCH tuning). This is the oracle reference. |
| PAST | Frozen FL-EDL config with assignments from the past-only (prefix-embedding) representation; same K, L, τ rule, Λ and β |
| FL | The frozen FL-EDL winner |
| SHUF | FL with q permuted among the valid regions of every micro-batch. Per-batch cluster mass is preserved exactly; region–forecast correspondence is destroyed. |
| FD | FL's q distilled into the final hidden state (training-only linear head on region means, soft cross-entropy), with no divergence. γ = 0.00035 was gradient-matched to FL's β·L_FL-EDL (`finals/fd_gamma.json`). |

### 2.5 Specialisation evaluation (labels and encoders never used by any router or loss)

The audit uses the lfr_audit code unchanged. It covers two independent label systems: genuine domain labels, and K=16
clusters from the frozen e5-base-v2 encoder (`windows_calib.npz` / `windows_val.npz`).

**Sets.**
- Discovery set: CALIB-SEARCH paper-half A.
- Measurement set: the VAL audit subset, 0.2M tokens per domain.

**Metrics measured for every MoE layer and expert:**
- topic distribution and topic entropy;
- topic/expert MI with a load-preserving shuffled-routing null;
- latent-cluster/expert MI;
- within-expert semantic similarity (e5);
- topic prediction from selected experts;
- cross-seed Hungarian-aligned profile correlation with a null;
- full router distributions;
- **FSI**: a group ablation of each topic's lift-assigned experts, difficulty-matched by baseline-CE deciles, with a
  shuffled-label control.

**Per-expert counterfactual interventions.** These ran at 100 % for T4, EDL, PAST and FL (352 interventions per model):
1. For every (layer, expert), find its strongest domain k* by lift on the discovery set.
2. Disable only that expert (logit −inf; top-4 among the rest).
3. Measure ΔCE on held-out k* tokens minus difficulty-matched ΔCE on other-domain tokens. This is the per-expert
   interaction.

**Checkpoints.**
- Search: 5M, 20M and 50M tokens.
- Finals: 40 %, 60 %, 80 % and 100 %. SHUF and FD are audited at 40 % and 100 % only.

**Frozen success rules (prereg §9 and §11, fixed before VAL).**
- "Better" = mean(FL − T4) < 0 and negative in every seed.
- "Comparable" = mean ≤ +0.01.
- Meaningful specialisation requires all four of:
  - (i) FSI(FL) > FSI(T4) in every seed, for both label systems;
  - (ii) mean FSI ratio ≥ 1.15 for both label systems;
  - (iii) FSI − FSI_shuf > 0 in every seed;
  - (iv) per-expert interaction of FL > T4 in every seed.

## 3. Correctness checks before training

- **CPU test suite** (`fle_tests.py`, CUDA hidden): T1–T13 all pass (`cpu_tests_*.json`). The tests cover:
  - causal forecast construction;
  - permutation invariance of aggregation;
  - frozen, detached assignments;
  - balanced clustering;
  - deterministic cache reuse;
  - exact resume;
  - no VAL access during search;
  - student inference without draft artifacts;
  - matched initialisation and data order;
  - unrestricted routing.
- **Existing repository suite:** 639 passed.
- **GPU checks** (`gpu_checks.json`): all pass.
  - All losses are finite.
  - Peak memory is about 17 GiB.
  - Stop-and-resume of a short run matches the uninterrupted run: mean |ΔLM| 0.030 against measured run-to-run noise
    of 0.0425 (deviation 3).
  - Inference succeeds with every forecast artifact unavailable.
- **T-EQ failed** (`equivalence_check.json`). fle_train TOKEN-4 s0 was compared with the lfr final TOKEN-4 s0: step 0
  is bit-identical, but step 50 differs by 0.073, above the 0.01 tolerance (step 100 differs by 0.008).
  - The cause is non-deterministic kernels in a chaotic early phase.
  - Per the prereg, T4 s0 was **retrained** rather than reused. The lfr run is reported only descriptively: its VAL CE
    is 3.0496, against 3.0502 for the retrained T4 s0.

## 4. Training-only search (seed 100; VAL never read)

All candidates ran the FINAL plan (identical LR schedule) and stopped at 5M, 20M and 50M with one rotating resumable
slot. Each comparison is against the TOKEN-4 search reference at the *same step*:
- CE on CALIB-SEARCH paper-half B, 0.4M tokens per domain;
- dev audit with discovery on half A and measurement on half B.

Stage 3 opened CALIB-SELECTION once. Full table: `candidate_table.tsv` and `report_tables.md`; per-milestone diagnostics
are in `diagnostics/`.

### 4.1 Pre-training latent diagnostics (training data only; `latent_calibration_table.md`)

- **Clustering quality.** All configurations reached balanced masses (min/max ≥ 0.93) and the target diffuseness
  (H(q)/log K 0.27–0.47). Stability ranged from 0.59 (K = 8) to 0.79.
- **The forecast clusters are semantically coherent** (`latent/*/cluster_description.json`): optics, inequalities,
  CNNs, fluids, logic, chemistry, category theory, Bayesian statistics, molecular biology, and so on.
- **The key number.** The q-weighted unigram predictiveness of the region's *actual* tokens is **higher for past-only
  clusters than for forecast clusters at every K except a near-tie at K = 8**:

| K | past-only (nats/token) | forecast (nats/token) |
|---|---|---|
| 8 | 0.130 | 0.135 |
| 16 | 0.165 | 0.154 |
| 32 | 0.192 | 0.174 |

- **Past clusters also align more with domains:** NMI 0.29 vs 0.23.
- **FL and PAST clusters are only partly the same:** hard-cluster agreement 0.41.
- **The drafts disagree a lot.** Mean 1 − cos is 0.63 and first-token agreement is 0.60: a diverse but individually
  weak ensemble of 21M-parameter drafts.

### 4.2 Stages

**Stage 1 (5M, registered C01–C10).**
- Every FL-EDL candidate was worse than TOKEN-4 at 5M: +0.012 … +0.059, mean +0.033.
- Two candidates (C03, K=32; C05, β=5e-4) starved one expert below 0.1× uniform load, which makes them invalid under
  the frozen validity rule.
- A replicate pair with identical knobs (C01 before and after the JSD clamp fix) differed by 0.031 nats in dCE and by
  2× in FSI. The preregistered +0.02 gate was therefore below measurement noise; deviation 7 raised the gate to 0.062,
  at 5M only.
- The FSI-rank rule promoted C04 (β 5e-5), C08 (all layers), C02 (K 8) and C07 (A2 deep-sets).
- No diagnosis-driven candidate was triggered, because the incumbent had no flag.

**Stage 2 (20M).**

| candidate | dCE vs T4 | eligible (gate 0.015)? |
|---|---|---|
| C04 | −0.0025 | yes |
| C07 | +0.0101 | yes |
| C02 | +0.0156 | no |
| C08 | +0.0295 | no |

All four were flagged `aux_weak`: their FL-EDL backbone-gradient norm was below 0.1 % of the LM gradient norm.

**Stage 3 (50M).**
- CALIB-SEARCH: C04 −0.0019, C07 +0.0106.
- CALIB-SELECTION: C04 −0.0019 (eligible), C07 +0.0113 (not eligible).

**Frozen winner C04** (`frozen.json`, sha `2c35b1d1…`, mode 0400): A1 meanstat, D = 4, H = 32, L = 256, K = 16,
β = 5e-5, upper half (MoE layers 5–10). On CALIB-SELECTION its domain FSI was 0.00561 vs 0.00515 for TOKEN-4, and its
e5-cluster FSI was 0.00547 vs 0.00596.

### 4.3 Search diagnostics that already pointed at the outcome

- **The winner is the *weakest* auxiliary in the grid.** Its FL-EDL gradient was 0.10 % of the LM gradient at 5M,
  0.05 % at 20M and 0.03 % at 50M. The stronger settings (β 1e-4 / 5e-4, all layers, K 32) either cost CE early (C08
  +0.030 at 20M) or starved experts (C03, C05). The CE gate therefore selected for a near-inert auxiliary.
- **TOKEN-4 develops almost the same alignment with the forecast clusters.** Routing separation between latent clusters
  grows with training (mean JSD between cluster routing means: 0.0005 at 5M → 0.0097 at 50M for C04). But latent MI is
  0.050 for TOKEN-4 vs 0.053 for C04 at 50M, after a large early gap (0.039 vs 0.055 at 5M).

## 5. Final comparison (VAL, 100 % checkpoint, identical recipe within a seed)

All 15 final runs are COMPLETE with identical settings:
- 3880 steps and 245,032,457 tokens each;
- one uninterrupted leg;
- plan `f4bc3c935a47`;
- 103,609,728 parameters;
- unrestricted top-4 routing with the shared path.

Per-run status, hashes and timings are in `run_ledger.tsv`, `checkpoint_ledger.tsv` and each run's `status.json`.

### 5.1 VAL CE (nats/token, full VAL, token-weighted)

| condition | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| T4 (TOKEN-4) | 3.0502 | 3.0576 | 3.0585 | 3.0554 |
| FL (FL-EDL) | 3.0585 | 3.0515 | 3.0556 | 3.0552 |
| PAST | 3.0570 | 3.0453 | 3.0485 | 3.0503 |
| EDL (explicit labels) | 3.0454 | 3.0415 | — | 3.0434 |
| SHUF | 3.0601 | 3.0565 | — | 3.0583 |
| FD | 3.0528 | 3.0590 | — | 3.0559 |

The paired per-document 95 % CI uses 2000 bootstrap resamples over 25,333 documents, with documents averaged over seeds.
**It measures document sampling only, not seed replication.** The seed spread of T4 is 0.0083, an order of magnitude
larger than these CIs.

| comparison | per seed | mean | paired-doc 95 % CI |
|---|---|---|---|
| FL − T4 | +0.0083, −0.0061, −0.0029 | **−0.0003** | [−0.0004, −0.0001] |
| **FL − PAST (decisive)** | **+0.0015, +0.0061, +0.0071** | **+0.0049** | [+0.0047, +0.0051] |
| PAST − T4 | +0.0068, −0.0122, −0.0100 | −0.0052 | [−0.0054, −0.0049] |
| FL − EDL | +0.0131, +0.0100 | +0.0116 | [+0.0113, +0.0118] |
| EDL − T4 | −0.0049, −0.0161 | −0.0105 | [−0.0107, −0.0102] |
| FL − SHUF | −0.0016, −0.0050 | −0.0033 | [−0.0035, −0.0031] |
| SHUF − T4 | +0.0098, −0.0011 | +0.0044 | [+0.0041, +0.0046] |
| FL − FD | +0.0057, −0.0076 | −0.0010 | [−0.0012, −0.0007] |
| FD − T4 | +0.0026, +0.0015 | +0.0020 | [+0.0018, +0.0022] |
| PAST − SHUF | −0.0031, −0.0112 | −0.0071 | [−0.0074, −0.0069] |

Per-domain mean ΔCE vs T4 (cs, math, physics, q-bio, stat):

| comparison | cs | math | physics | q-bio | stat |
|---|---|---|---|---|---|
| FL | +0.0004 | −0.0007 | −0.0006 | −0.0001 | −0.0004 |
| PAST | −0.0044 | −0.0066 | −0.0044 | −0.0045 | −0.0058 |
| EDL | −0.0086 | −0.0118 | −0.0129 | −0.0092 | −0.0100 |
| SHUF | +0.0056 | +0.0036 | +0.0037 | +0.0048 | +0.0041 |
| FD | +0.0031 | +0.0010 | +0.0024 | +0.0023 | +0.0014 |

**Third seed.** The rule triggered on the two primary seeds: the sign of FL − T4 differed and |mean| ≤ 0.01
(`finals/third_seed_decision.json`). Seed 2 was run for T4, FL and PAST. The specialisation conclusions did not
disagree between seeds, so EDL, SHUF and FD needed no third seed.

### 5.2 Functional specialisation at 100 % (VAL audit subset)

| condition | seed | FSI domain | FSI e5 | FSI_shuf domain | MI domain | MI latent | topic-pred domain | within-expert sim | per-expert interaction (frac > 0) |
|---|---|---|---|---|---|---|---|---|---|
| T4 | 0 | 0.00806 | 0.01252 | −0.00012 | 0.0330 | 0.0628 | 0.586 | 0.0012 | 0.00040 (0.733) |
| T4 | 1 | 0.00759 | 0.01173 | −0.00020 | 0.0330 | 0.0629 | 0.571 | 0.0011 | 0.00039 (0.685) |
| T4 | 2 | 0.00782 | 0.01191 | −0.00008 | 0.0331 | 0.0647 | 0.570 | 0.0012 | 0.00042 (0.739) |
| FL | 0 | 0.00775 | 0.01175 | −0.00015 | 0.0357 | 0.0681 | 0.585 | 0.0013 | 0.00042 (0.710) |
| FL | 1 | 0.00776 | 0.01155 | −0.00014 | 0.0324 | 0.0643 | 0.581 | 0.0012 | 0.00040 (0.739) |
| FL | 2 | 0.00739 | 0.01118 | −0.00017 | 0.0323 | 0.0638 | 0.568 | 0.0012 | 0.00039 (0.722) |
| PAST | 0 | 0.00801 | 0.01325 | −0.00017 | 0.0371 | 0.0695 | 0.595 | 0.0013 | 0.00041 (0.719) |
| PAST | 1 | 0.00744 | 0.01234 | −0.00009 | 0.0328 | 0.0663 | 0.556 | 0.0012 | 0.00037 (0.707) |
| PAST | 2 | 0.00816 | 0.01210 | +0.00001 | 0.0319 | 0.0635 | 0.558 | 0.0012 | 0.00040 (0.696) |
| EDL | 0 | 0.01631 | 0.02241 | +0.00015 | 0.1515 | 0.2229 | 0.670 | 0.0034 | 0.00069 (0.818) |
| EDL | 1 | 0.01685 | 0.02217 | +0.00002 | 0.1428 | 0.2159 | 0.661 | 0.0033 | 0.00066 (0.810) |
| SHUF | 0 | 0.00869 | 0.01360 | +0.00004 | 0.0364 | 0.0683 | 0.587 | 0.0013 | — |
| SHUF | 1 | 0.00752 | 0.01090 | −0.00007 | 0.0345 | 0.0662 | 0.583 | 0.0012 | — |
| FD | 0 | 0.00806 | 0.01246 | −0.00005 | 0.0342 | 0.0653 | 0.597 | 0.0012 | — |
| FD | 1 | 0.00762 | 0.01198 | −0.00013 | 0.0325 | 0.0616 | 0.587 | 0.0012 | — |

For all models, the shuffled-routing MI nulls are about 2e-5 for domain labels and about 7e-5 for latent clusters.

Cross-seed Hungarian-aligned domain-profile correlation (null in parentheses):

| T4 | FL | PAST | EDL | SHUF | FD |
|---|---|---|---|---|---|
| 0.880 (0.712) | 0.877 (0.703) | 0.877 (0.699) | **0.912** (0.651) | 0.871 (0.717) | 0.875 (0.719) |

**FSI ratio to T4 (mean over seeds):**

| condition | domain | e5 cluster |
|---|---|---|
| FL | **0.976** | **0.954** |
| PAST | 1.006 | 1.042 |
| SHUF | 1.03 | 1.01 |
| FD | 1.00 | 1.01 |
| EDL | **2.12** | **1.84** |

**Frozen criteria for FL** (`final_analysis.json["criteria"]`):

| criterion | result |
|---|---|
| (i) FSI(FL) > FSI(T4) in every seed, both label systems | **fails** — FL is above T4 only on domain FSI in seed 1 |
| (ii) mean ratio ≥ 1.15 for both | **fails** (0.976 / 0.954) |
| (iii) FSI − FSI_shuf > 0 in every seed | passes |
| (iv) per-expert interaction of FL > T4 in every seed | **fails** — 0.00042 > 0.00040 and 0.00040 > 0.00039, but 0.00039 < 0.00042 in seed 2 |

Meaningful specialisation: **no**. CE: comparable, not better.

### 5.3 Specialisation trajectories (FSI domain / e5; `specialization_trajectory.json`)

| condition | seed | 40 % | 60 % | 80 % | 100 % |
|---|---|---|---|---|---|
| T4 | 0 | 0.00529 / 0.00844 | 0.00638 / 0.01067 | 0.00721 / 0.01109 | 0.00806 / 0.01252 |
| T4 | 1 | 0.00519 / 0.00839 | 0.00736 / 0.01169 | 0.00692 / 0.01107 | 0.00759 / 0.01173 |
| T4 | 2 | 0.00511 / 0.00741 | 0.00629 / 0.00952 | 0.00695 / 0.01151 | 0.00782 / 0.01191 |
| FL | 0 | 0.00544 / 0.00744 | 0.00631 / 0.00919 | 0.00717 / 0.01096 | 0.00775 / 0.01175 |
| FL | 1 | 0.00530 / 0.00782 | 0.00632 / 0.00951 | 0.00657 / 0.01123 | 0.00776 / 0.01155 |
| FL | 2 | 0.00577 / 0.00841 | 0.00655 / 0.00957 | 0.00706 / 0.01079 | 0.00739 / 0.01118 |
| PAST | 0 | 0.00561 / 0.00920 | 0.00685 / 0.01099 | 0.00742 / 0.01196 | 0.00801 / 0.01325 |
| PAST | 1 | 0.00515 / 0.00751 | 0.00620 / 0.00923 | 0.00705 / 0.01019 | 0.00744 / 0.01234 |
| PAST | 2 | 0.00579 / 0.00849 | 0.00675 / 0.01029 | 0.00747 / 0.01080 | 0.00816 / 0.01210 |
| EDL | 0 | 0.01058 / 0.01393 | 0.01287 / 0.01705 | 0.01489 / 0.02048 | 0.01631 / 0.02241 |
| EDL | 1 | 0.01167 / 0.01450 | 0.01371 / 0.01745 | 0.01494 / 0.02050 | 0.01685 / 0.02217 |
| SHUF | 0 | 0.00586 / 0.00957 | — | — | 0.00869 / 0.01360 |
| SHUF | 1 | 0.00504 / 0.00748 | — | — | 0.00752 / 0.01090 |
| FD | 0 | 0.00526 / 0.00781 | — | — | 0.00806 / 0.01246 |
| FD | 1 | 0.00512 / 0.00862 | — | — | 0.00762 / 0.01198 |

- Functional specialisation grows monotonically with training in every condition. For FL, PAST, SHUF and FD it grows at
  the same rate as TOKEN-4.
- FL has a small early edge in domain FSI at 40 % (0.0055 vs 0.0052) and a small early deficit in e5 FSI (0.0079 vs
  0.0081). Neither survives to 100 %.
- Latent-cluster MI shows the same pattern:
  - At 40 % it is 0.0626 for FL vs 0.0591 for TOKEN-4 (mean over seeds).
  - At 100 % it is 0.0654 vs 0.0635.
  - That ~3 % residual is the only routing-level trace FL-EDL leaves, and PAST leaves the same trace (0.0664).
- EDL's specialisation is present at 40 % and keeps growing through 100 %.

## 6. Answers to the primary questions

**1. Does FL-EDL match or beat TOKEN-4 CE?**
- It matches but does not beat: mean −0.0003 nats/token, with seeds +0.0083, −0.0061 and −0.0029.
- The mean is far inside the frozen +0.01 bar, so FL is "accuracy-comparable".
- The seed signs disagree, so it is not "better".

**2. Does FL-EDL create stronger functional specialisation?**
- No. FSI ratio vs TOKEN-4 is 0.976 (domain) and 0.954 (e5).
- Per-expert interaction is 0.00040 vs 0.00040 (mean over seeds). MI, topic prediction, within-expert similarity and
  cross-seed alignment are all within seed noise of TOKEN-4.
- Three of the four frozen specialisation criteria fail.

**3. Does it beat past-only latent EDL?**
- No. PAST has lower CE in all three seeds (FL − PAST = +0.0015, +0.0061, +0.0071; mean +0.0049).
- PAST is also at least as specialised: FSI ratio 1.006 / 1.042, against FL's 0.976 / 0.954.
- By the user's own criterion, the forecasts did not add useful information beyond the main model's causal context.

**4. Does it approach or beat explicit-label EDL without labels?**
- No. FL − EDL = +0.0116 nats/token (both seeds +0.010 or worse).
- EDL more than doubles domain FSI (2.12×) and raises per-expert interaction 1.7× (0.00068 vs 0.00040).
- EDL is the only condition with stronger cross-seed alignment (0.912 vs 0.88).
- FL recovers none of that gap.

**5. Do forecast semantics matter beyond generic regularisation?**
- For specialisation: **no**. SHUF, which keeps exact per-batch cluster mass with the wrong regions, has FSI equal to or
  higher than FL (1.03 / 1.01 × T4). FD's generic hidden-state distillation of the same q also does not specialise
  (1.00 / 1.01 × T4).
- For CE: weakly. FL beats SHUF by 0.0033 (both seeds) and PAST beats SHUF by 0.0071 (both seeds). A divergence with
  *wrong* region assignments costs CE (SHUF − T4 = +0.0044), so correct correspondence avoids a small penalty.
- That benefit is larger for past-derived than for forecast-derived assignments.
- FL − FD is −0.0010 with mixed seed signs: no reliable difference.

**6. Does any specialisation persist to the final checkpoint?**
- FL created no specialisation beyond TOKEN-4 at any checkpoint, so there was nothing to persist.
- Its small 40 % edge in domain FSI and latent MI shrinks by 100 %.
- The specialisation that exists in all models grows steadily to the final checkpoint. Under EDL it persists and widens
  even though EDL's auxiliary-to-LM gradient ratio falls from about 5 % early to about 2 % late.

**7. What is the complete training cost and student-only inference cost?**
- Training and inference costs are given in section 8.
- **Student-only inference is unchanged.** FL s0 runs at 330k tokens/s vs 328k for TOKEN-4. It has the same 103.6M
  parameters, no forecast modules are loaded, and the checkpoint has zero training-only keys.
- **Per final run, FL-EDL training costs +4.7 % wall time** (1940 s vs 1851 s; 130.8k vs 137.0k tokens/s).
- **The one-off forecast pipeline costs about 1.35 GPU-h:**
  - 0.49 h for drafts G1–G3, plus 0.12 h for G0, which was already paid by fspv2;
  - 0.86 h for the final region cache.
- **The FL-specific search costs 2.18 GPU-h**, including the search cache.

## 7. Failure analysis (frozen diagnostics only; no tuning after VAL was opened)

Each hypothesis is marked as evidence or speculation.

| hypothesis | finding | status |
|---|---|---|
| Weak or redundant drafts | Four 21M drafts with LM loss 3.18–3.26. High disagreement (1 − cos 0.63; first-token agreement 0.60), so not redundant. But weak: they are much smaller than the 104M student they are meant to inform. | evidence (weak, not redundant) |
| Inaccurate continuations | Continuation accuracy against the real next tokens was not measured directly. Indirectly, forecast-cluster predictiveness of the *actual* region (0.154 nats/token at K16) is below that of the prefix embedding (0.165). | partial evidence |
| **Forecasts contain no information beyond causal past** | **Primary cause.** Before training, past clusters were more predictive at K 16 and 32 and more domain-aligned (NMI 0.29 vs 0.23). After training, PAST beats FL in CE in 3/3 seeds and matches or exceeds it in FSI. The 32-token sampled continuations are a noisy function of the same prefix the router already sees. | **evidence** |
| Unstable or semantically meaningless clusters | Clusters are semantically coherent (cluster descriptions) and balanced (≥ 0.93), with stability 0.59–0.79. They are not meaningless. | ruled out |
| Overly diffuse soft assignments | H(q)/log K was about 0.4 by design (the target). The SHUF control shows the loss did produce in-batch routing separation under correct q: final training JSD 0.031 for FL vs 0.007 for SHUF. Diffuseness did not prevent the divergence from acting. | ruled out as main cause |
| **Auxiliary gradients too weak** | **Contributing cause.** FL-EDL backbone-gradient norm was 0.01–0.07 % of the LM gradient throughout final training (about 0.015 % late). That is about 100× below EDL (about 2 %) and about 200× below the Switch load-balancing gradient (about 3 %). The CE-gated search *selected* the weakest β (5e-5); stronger settings lost CE early or starved experts. | **evidence** |
| Auxiliary gradients interfering with LM optimisation | At the frozen β, no: FL − T4 ≈ 0. At stronger β in search, yes early: +0.03 at 5M–20M for β 1e-4 all layers; expert starvation at β 5e-4 or K = 32. Whether that early cost would have vanished at full scale (as it did for C04 between 5M and 20M) is untested. | evidence (search scale only) |
| Routing separation without functional expert learning | Partly. FL raised in-batch JSD between latent-cluster routing means, but VAL latent-cluster MI was only about 3 % above TOKEN-4 (PAST matched it), and FSI and per-expert interaction did not move. Even routing separation on held-out data was minimal. | evidence |
| Load-balancing conflict | Not observed. All conditions keep min expert load ≥ 0.70× uniform (FL ≥ 0.79×). FL routing entropy is 0.65–0.67, the same as T4 (0.66–0.67); EDL drops to 0.59. EDL specialises strongly under the same Switch loss, so the balance loss does not prevent specialisation. | ruled out |
| Insufficient training scale | Cannot be excluded for small effects: 245M tokens and seed spread 0.008. But EDL produces a 2× FSI effect at the same scale, so scale is not what blocks a sufficiently informative signal. | speculation |
| Specialisation real but unhelpful for prediction | Not applicable to FL, which produced no specialisation. EDL shows label-driven specialisation *is* helpful here: −0.0105 CE. | ruled out for FL |

**Diagnosis.** The forecast lens adds no information about upcoming text that the 192-token causal prefix does not
already contain. At the only signal strength that survived the CE gate, the divergence was too weak to reshape routing.
Under both conditions FL-EDL converges to TOKEN-4. Past-only latent EDL, with the same weak β, gets a small but
consistent CE gain (−0.0052) without extra specialisation. The gain is plausibly a mild regularisation effect from
correctly grouped regions; that interpretation is speculation.

**Narrowest justified follow-up (not run, because it would be informed by VAL).**
- A preregistered **strength-matched** comparison: FL vs PAST vs SHUF with β set so that the auxiliary/LM gradient ratio
  matches explicit EDL (about 2 %). This tests whether *any* latent partition can drive EDL-like specialisation.
- The selection rule should not gate on 5M CE, which section 4.2 shows is below noise.
- Because PAST already beats FL, a better forecaster is a lower priority than establishing whether label-free
  partitions work at matched strength at all.

## 8. Costs, VRAM, storage, inference

**GPU-hours** (`compute_ledger.tsv`, append-only; `final_analysis.json["gpu_seconds_by_category"]`):

| phase | item | GPU-h |
|---|---|---|
| shared | drafts G1–G3 (G0 reused from fspv2, 0.12 h, not charged) | 0.49 |
| search | search forecast caches (calib_search, search_main) | 0.41 |
| search | search training (T4 reference + C01–C10, including INVALID_BUG reruns) | 0.66 |
| search | search evaluations (dev, CALIB-SEL) and checks | 1.11 |
| final | final forecast caches (final_main, val_audit) | 0.86 |
| final | 15 final training runs | 8.06 |
| final | final VAL evaluations (15 runs + lfr reference CE) + benchmark | 4.85 |
| **total ledgered** | | **16.46** |

- **Unledgered:** two killed partial cache attempts (deviation 2: 30 s; deviation 8: about 11 min) are not in the
  ledger. The true total is about 16.65 GPU-h.
- **Search share:** 2.18 h = 13 % (cap 35 %). **Final share:** 13.79 h = 84 % (floor 65 %).
- **Overrun:** the 16 GPU-h target was exceeded by about 0.5–0.65 h (3–4 %). Evaluations cost 6.0 h against the planned
  2.5 h, because of the 352-intervention per-expert audit, 23 min per primary model. Nothing was dropped (deviation 9).
- **Wall clock:** 02:50 → about 20:30 EST, about 17.7 h of the 18 h budget.

**Per-run training time and throughput (final, 3880 steps):**

| condition | seconds | tokens/s |
|---|---|---|
| T4 | 1846–1856 | ~137.0k |
| FL | 1936–1948 | ~130.6k |
| PAST | 1914–1989 | ~130.0k |
| SHUF | 1948–1949 | ~130.0k |
| FD | 1884–1910 | ~134.0k |
| EDL | 2057–2064 | ~122.8k |

**Peak VRAM.** Training: 17.0 GiB allocated and ≤ 24.5 GiB reserved in every condition (batch 32×1024 micro, accum 2).
Evaluation and benchmark fit well within that.

**Inference** (`inference_bench.json`, 64×~1000-token rows, median):

| model | tokens/s |
|---|---|
| T4 | 327.9k |
| FL | 330.1k |
| PAST | 330.7k |
| EDL | 329.4k |

All have identical parameter counts and no forecast modules or training-only state-dict keys. Inference cost is
unchanged by FL-EDL.

**Storage.** New persistent footprint is about 24 GiB, against the 50 GiB cap:

| path | size |
|---|---|
| `runs/*forecast_latent_edl_v1*` | 17 GiB (finals 12 GiB, drafts 0.25 GiB, search/eq/smoke run dirs about 4.8 GiB) |
| `artifacts/forecast-latent-edl-v1` | 7.0 GiB (forecast caches 6.6 GiB, latent 0.33 GiB, evals 54 MiB) |
| `.tmp/forecast-latent-edl-v1` | 0.4 MiB (search slots already deleted and ledgered in `deleted_artifacts.tsv`) |

- Free disk went from 168.06 GiB at start to **144.26 GiB** at end. The 100 GiB reserve was never crossed: every job
  checked `free − planned peak ≥ 100 GiB`.
- **Protected inventory:** all 5,340 pre-existing files (122 GB) were verified unchanged at the end, with 0 mismatches
  (`protected_inventory_before.verify_20261115T010959Z.json`). No prior experiment was modified, resumed or deleted.

## 9. Deliverables

| item | path (relative to `artifacts/forecast-latent-edl-v1/` unless absolute) |
|---|---|
| hashed preregistration | `docs/FORECAST_LATENT_EDL_V1_PREREG.md`, `.sha256`, `preregistration_hashes.json` |
| data and configuration manifests | `configs/forecast-latent-edl-v1/*.json`; per-run `config.json` and `status.json` (plan and table sha256); `drafts.json` |
| immutable forecast-cache manifests | `cache/{calib_search,search_main,final_main,val_audit}/manifest.json`, `cache_jobs.jsonl` |
| checkpoint and artifact hashes | `checkpoint_ledger.tsv`, `run_ledger.tsv`, `frozen.sha256`, `latent/*/meta.json` |
| append-only compute ledger | `compute_ledger.tsv`; repository-wide GPU ledger `gpu_time_ledger.tsv` |
| candidate-search table | `candidate_table.tsv`, `report_tables.md`, `diagnostics/`, `search/` |
| frozen-selection record | `frozen.json` (0400), `STAGE3_OPEN.json` |
| final condition table | `report_tables.md`, `final_analysis.json["val_ce"]`, `["ce_differences"]` |
| specialisation trajectories | `specialization_trajectory.json`, `evals/final_*/final.json` |
| functional-ablation results | `evals/final_*/final.json` (`audit.p100.per_expert`, FSI groups) |
| machine-readable final analysis | `final_analysis.json` (plus `finals/analysis_after_primary.json`, `finals/third_seed_decision.json`) |
| deviations, invalid runs, deletions | `deviations.jsonl` (1–9), `invalid_runs.tsv`, `deleted_artifacts.tsv` |
| driver state and logs | `driver_state.json` (113 COMPLETE jobs, 5 INVALID_BUG, `external_stop` = none), `driver_log.txt`, `job_logs/` |

## 10. Deviations from the preregistration (`deviations.jsonl`)

| # | type | effect on results |
|---|---|---|
| 1 | Sinkhorn test data changed to overlapping blobs (the original test was unsatisfiable) | none |
| 2 | First search_main cache job killed after 30 s by me, before any output, to apply a bf16 → fp32 KV efficiency fix | none |
| 3 | GPU resume tolerance made concrete: max(0.003, 3 × measured noise), because early training is chaotic | none |
| 4 | scipy missing; replaced with the repository's Hungarian implementation; parallel latent fits relaunched | none |
| 5 | Routing-entropy validity cap set to max(0.98, T4 + 0.01), after seeing only the T4 reference (T4 itself violated 0.98 at 5M) | none on finals |
| 6 | JSD clamp bug (fp32 negative JSD → NaN, C03). Affected search runs marked INVALID_BUG and rerun as v2 | search reruns |
| 7 | Stage-1 CE gate raised from +0.02 to +0.062 after a measured 0.031 replicate noise. The 20M and 50M gates and all final criteria are unchanged | search promotion only |
| 8 | Final cache restarted with batch 512 for efficiency; completed chunks hash-verified; outputs are batch-invariant | none |
| 9 | GPU budget projection: about 16.5 h vs the 16 h target, because evaluation cost more than planned. Nothing dropped | none |

No external interruption occurred. No run was resumed across an interruption. No incomplete run was reused.

## 11. Reproduction and resume commands

Run all commands from `/home/zapan4/moe-research` after `source env.sh`.
- GPU jobs go only through the driver, which calls `scripts/gpu_guarded.sh`.
- Every job is idempotent: a COMPLETE job is skipped, and a resumable trainer continues from its verified slot.

```bash
# tests
.venv/bin/python fle_tests.py                     # CPU suite (CUDA hidden)
.venv/bin/python -m pytest -q tests               # existing suite (639 passed)
# drafts G1-G3 (seeds 11-13), then write artifacts/forecast-latent-edl-v1/drafts.json
MOE_GPU_LAUNCH=1 MOE_RESEARCH_APPROVE=gpu scripts/gpu_guarded.sh "fle draft" fle_draft_train.py --seed 11 --device cuda --execute
# T-EQ and GPU checks
.venv/bin/python fle_driver.py job eq_train --gpu --expect-rc 75 -- fle_train.py --config configs/forecast-latent-edl-v1/eq_T4_s0.json --max-steps 101
.venv/bin/python fle_check_eq.py runs/<eq run>
.venv/bin/python fle_gpu_checks.py
# forecast caches
.venv/bin/python fle_driver.py job cache_calib_search --gpu -- fle_cache.py --name calib_search
.venv/bin/python fle_driver.py job cache_search_main --gpu -- fle_cache.py --name search_main --batch-docs 256
# latent domains (CPU), one per configuration
.venv/bin/python fle_driver.py job latent_fit_A1_K16_L256 --cpu -- fle_latent.py fit --cfg A1_K16_L256
# search
.venv/bin/python fle_search.py t4
.venv/bin/python fle_search.py stage1; .venv/bin/python fle_search.py respond
.venv/bin/python fle_search.py promote --stage 1; .venv/bin/python fle_search.py stage2
.venv/bin/python fle_search.py promote --stage 2; .venv/bin/python fle_search.py stage3   # opens CALIB-SEL, writes frozen.json
# finals
.venv/bin/python fle_finals.py prep       # final-region cache, tables (+merge), VAL-audit latent labels, FD gamma
.venv/bin/python fle_finals.py train --seeds 0 1 --conds FL PAST T4 EDL SHUF FD
.venv/bin/python fle_finals.py eval  --seeds 0 1 --conds FL PAST T4 EDL SHUF FD
.venv/bin/python fle_analyze.py; .venv/bin/python fle_finals.py seed3; .venv/bin/python fle_analyze.py
.venv/bin/python fle_driver.py job bench_inference --gpu -- fle_bench.py --runs final_T4_s0 final_FL_s0 final_EDL_s0 final_PAST_s0
.venv/bin/python fle_report_tables.py
.venv/bin/python fle_inventory.py --verify artifacts/forecast-latent-edl-v1/protected_inventory_before.json
```

**Resume and interruption policy.**
- A job that ends with rc 77/78, or whose launcher is blocked (rc 3), sets `driver_state.json["external_stop"]`. The
  driver then refuses every further job.
- Restart only on explicit instruction: `.venv/bin/python fle_driver.py authorize-resume`, then rerun the same phase
  command.
- `artifacts/forecast-latent-edl-v1/STOP` stops every trainer at its next optimizer step with a verified checkpoint.
- The experiment is complete. Rerunning any phase command is a no-op, because every job is COMPLETE. An evaluation
  refuses to overwrite an existing `final.json`.

## 12. Final state (2026-11-14 ~20:12 EST)

- **GPU:** A100 80GB, 0 % utilisation, 0 MiB used, no compute processes.
- **Processes:** no `fle_*`, chain or guarded-launcher process is running; both chains ended normally ("FINALS DONE",
  "END CHAIN DONE").
- **Driver:** `external_stop` is unset; 113 jobs are COMPLETE and 5 are INVALID_BUG (pre-fix search runs, superseded
  by v2 and ledgered).
- **Disk:** 144.26 GiB free on the repository filesystem (start 168.06 GiB; reserve 100 GiB respected throughout).
- **Storage:** about 24 GiB of new persistent data, all beneath `/home/zapan4/moe-research`. Nothing was written to
  /dev/shm, /data, external symlinks or storage outside the repository.
- **Integrity:** the protected inventory is unchanged (5,340/5,340 files match).

## Verdict

**4. FL-EDL preserves accuracy but does not specialize meaningfully.**
