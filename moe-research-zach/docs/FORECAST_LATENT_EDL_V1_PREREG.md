# csr-forecast-latent-edl-v1 (prefix `fle_`): Forecast-Latent Expert Divergence Learning — preregistration

**Written** 2026-11-14 03:00 EST. No FL-EDL model, cache, draft or candidate result of this experiment exists yet.
The sha256 of this file is stored in `docs/FORECAST_LATENT_EDL_V1_PREREG.sha256` and
`artifacts/forecast-latent-edl-v1/preregistration_hashes.json` before any new result exists. Later changes are logged,
with their reason, in `artifacts/forecast-latent-edl-v1/deviations.jsonl`. No scientific rule is changed after a result
it affects has been seen. Genuine bugs are fixed, get a regression test, are logged, and affected work is rerun.

## 0. Startup audit (2026-11-14 02:50–03:00 EST)

- **GPU.** A100 80GB, 0 MiB used, 0 compute processes. Another user's two CPU-only python processes exist (not on GPU).
- **Disk.** 180,074,127,360 B free (167.71 GiB) at 02:58 (168.06 GiB at 02:50). Reserve 100 GiB (never changed) →
  ~67.7 GiB headroom. This experiment's persistent footprint cap is 50 GiB; plan in section 13 (≤ 40 GiB).
- **Prior experiments** are preserved and read-only. fspv2, lfr v1 and cps v2 are operator-stopped (STOP markers);
  they are never resumed, edited or written to. Protected inventory: `fle_inventory.py` hashes all existing project
  files (code, docs, artifacts, runs) before work and verifies them at the end.
- **Existing forecast caches** (`artifacts/forecast-specialist-pretraining-v2/cache/main`, 19 GiB;
  `.tmp/latent-forecast-router-distill-v1/cache/main`, 6.1 GiB, 147M tokens): **not reusable.** They store per-token
  next-token / horizon ≤ 8 greedy features of six drafts of which five are domain-named drafts (forbidden here: they
  would reintroduce domain labels), contain no 32-token continuations and no region-level summaries. They are not read.
- **TOKEN-4 FINAL s0** (`runs/20261112T165936Z_latent_forecast_router_distill_v1_final_TOKEN-4_s0`; trained by
  `lfr_train.py`; 103,609,728 params; plan f4bc3c93…; milestones 40/60/80/100 % with sha256 in its status.json; never
  evaluated on VAL): **reusable if** the equivalence check (section 10, T-EQ) passes: the `fle_train.py` TOKEN-4 code
  path replays seed 0 with bit-identical step-0 losses and step-50/100 training losses within 0.01 nats of the lfr
  metrics.jsonl. If it fails, TOKEN-4 s0 is retrained with `fle_train.py`.
- **cps v2 TOKEN s1**: **not reused.** It was trained by the V2 trainer with 85,451 extra (dead) context-prior
  parameters (103,695,179 total), so architecture/parameter count are not identical, and the V2 trainer showed a
  −0.001…−0.008 CE offset (CFE report). TOKEN-4 s1 (and s2 if triggered) are trained new.
- **Explicit-label EDL checkpoints**: lfr has EDL at S1/S2 (49M/122.5M plans, seeds 100/101) only — different
  schedules from the plan used here, so not compatible. EDL finals (seeds 0, 1) are trained new. The EDL definition
  and β* = 5e-3 are reused from lfr (tuned there on CALIB-SEARCH at S1; no VAL).
- **TOKEN-4 search milestones** at 5M/20M/50M under the FINAL plan do not exist (lfr S1/S2 runs use other warm-up /
  decay schedules). One search TOKEN-4 run (seed 100, FINAL plan, stopped at 50M) is trained once and reused for every
  search comparison.
- **Encoders available offline**: `artifacts/encoder/all-MiniLM-L6-v2` (forecast/past encoder here) and
  `artifacts/encoder/e5-base-v2` (specialisation evaluator only, as in lfr/cps/cfe). Independent models.
- **Drafts**: only one generalist draft exists (fspv2 shared draft, d256 L10, 125M tokens of `draft_shared`). The five
  fspv2 domain drafts are not used.

## 1. Question

Can a causal ensemble of frozen generalist draft LMs discover useful label-free latent domains whose balanced soft
assignments, used only in a training-time EDL-style divergence on population routing distributions, make MoE experts
functionally more specialised than TOKEN-4 at comparable or better CE — while the final model keeps unrestricted
per-token top-4 routing, the shared expert, all experts for every token, and no drafts / encoder / clusterer at
inference? The decisive comparison is FL-EDL vs past-only latent EDL.

## 2. Data, model, training recipe (all reused unchanged from lfr v1; protocol sha fe99b29e…)

- Data: fspv2 five-domain M2D2 arXiv manifest (splits_sha256 efcde27e…), Pythia tokenizer, ≤1024-token segments,
  paper-disjoint buckets: main (training), calib_search (CALIB-SEARCH), calib_sel (CALIB-SELECTION), val (VALIDATION),
  test (LOCKED, never opened; loaders refuse it).
- Stream: lfr balanced main stream (`lfr_lib.main_stream`), identical for all conditions and seeds; 64×1024 rows/step.
- **FINAL = 245M tokens, 3880 steps (plan sha f4bc3c935a47…)**. Search milestones are prefixes of the same plan:
  5M, 20M, 50M tokens = the first step counts whose cumulative valid tokens reach 5M / 20M / 50M.
- Backbone (`lfr_model.LFRModel`, router `linear`): 12 layers, d 384, 6 heads, layer 0 dense, layers 1–11 with 32
  routed GELU experts (FFN 256) + one always-active shared expert (FFN 768); unrestricted top-4, gates = softmax probs of
  the selected experts; dropless. 103,609,728 parameters, 43,054,464 active. Backbone init from `torch.manual_seed(seed)`
  identical across conditions within a seed.
- Recipe: AdamW (0.9, 0.95), wd 0.1 (none on 1-D), lr 6e-4, WSD (warm-up 1 %, decay over last 20 % to 6e-5), clip 1.0
  on backbone parameters, bf16 autocast, fp32 router math, torch.compile, Switch balance loss 0.01, z-loss 0.001.
- Seeds: search 100; final 0, 1 (+2 by the third-seed rule).

## 3. Causal forecast evidence (training only)

- **Regions.** In each segment (= a document to the model), region j covers positions [jL, min((j+1)L, n)),
  L ∈ {128, 256}. The forecast point of region j is s_j = jL. A region is *valid* iff j ≥ 1 and it has ≥ 32 tokens.
  Region 0 has no causal prefix and gets no assignment (excluded from every latent loss).
- **Causal prefix.** Tokens [0, s_j) of the same segment (exactly the model's own causal context at s_j). No token at
  or after s_j is ever given to a draft or encoder when constructing the forecast signal. Generated tokens are the
  drafts' own samples.
- **Drafts (D = 4 generalist, no domain identity).** G0 = fspv2 shared draft (existing, hash-verified). G1–G3 = new
  draft seeds 11, 12, 13 of the same architecture (`fsp_lib.DRAFT_CFG`, 21.4M params), each trained for one epoch on
  the balanced union `draft_shared ∪ draft_cont` (≈250M tokens; paper-disjoint from main/calib/val/test) in its own
  seeded random order, lr 1e-3 WSD, 64×1024 rows. No domain label is used. Cost is ledgered as forecast cost.
- **Generation.** Each draft, given the identical prefix, generates H = 32 tokens by seeded sampling (temperature
  0.8, top-k 50; counter-based Gumbel noise from (draft, document, region, step), so outputs do not depend on
  batching up to bf16 kernel noise). Implementation: one causal forward over the segment, then 31 decode steps in which
  each forecast point's query attends to real keys < s_j plus its own generated keys (tree attention).
- **Cached per (draft, region):** 32 continuation token ids; per step p(sampled token), full-distribution entropy and
  top-1 probability (uint8-quantised, T = 1 distribution); plus document id, region index, prefix length, and
  draft / tokenizer / data / config hashes in the manifest. The optional second continuation is not generated.
- **Inter-draft statistics per region:** mean pairwise cosine disagreement of continuation embeddings (1 − mean cos),
  mean entropy, mean confidence, entropy spread across drafts, first-token agreement.
- **Encoder (frozen, independent of drafts / main model / evaluator):** all-MiniLM-L6-v2 (mean pooling, L2 norm).
  Each continuation is decoded and embedded (max 64 word pieces). **Past representation** (for the past-only
  comparator): the last 192 prefix tokens decoded and embedded (left-truncated to 256 word pieces).
- **Coverage.** Search region: all documents in the first S50 steps (50M tokens) for L = 128 forecast points (L = 256
  points are the even subset). Final region: built after the freeze for the frozen L only, covering all 3880 FINAL
  steps. Development diagnostics: CALIB-SEARCH. After the freeze, for analysis only: CALIB-SELECTION and the VAL audit
  subset (latent-cluster/expert MI). The cache never covers test.
- Cache files are written once per chunk atomically with sha256; the manifest is immutable (0400) once complete. Every
  candidate reads the same manifest unless forecast construction itself is the registered variable.

## 4. Forecast aggregation (frozen before main training)

Per region, the unordered set {(e_d, stats_d)}_{d=1..D} (e_d = 384-d continuation embedding; stats_d = mean entropy,
mean p(token), mean top-1 prob of draft d) is mapped to one summary vector:

- **A1 `meanstat` (no learning):** u = normalize(mean_d e_d); σ = [disagreement, mean entropy, mean confidence,
  mean top-1, entropy spread] standardised with training-sample mean/std; summary = normalize([u, 0.25·σ/√5]).
- **A2 `deepsets-cv` (learned):** φ: MLP(384+3 → 256 → 256), mean-pool over drafts, ρ: MLP(256+5 → 256 → 128),
  L2-normalised. Trained only on training-partition regions of the search region (up to 200k regions, 3 epochs, AdamW
  1e-3, batch 1024) with a forecast-only cross-view InfoNCE objective: drafts are split into two disjoint random pairs
  per region, ρ(φ(view A)) must identify ρ(φ(view B)) of the same region among the batch (temperature 0.1). No
  validation data, labels or router feedback. Frozen afterwards; applied to the full 4-draft set.
- Set Transformer: not registered (bounded search).
- Past-only: summary = normalize(past embedding) (no forecast); clustered with the identical procedure.

## 5. Label-free balanced latent domains (frozen before main training)

- **Balanced Sinkhorn spherical soft k-means**, K ∈ {8, 16, 32}: fit on 100k valid training regions sampled (seed
  2026) from the search region; k-means++ init (seed 2026); 60 iterations of: S = X Cᵀ; Q = Sinkhorn(exp(S/τ)) with
  row sums 1 and equal column sums (5 iterations, log domain); C_k = normalize(Σ_i Q_ik x_i). The column potentials b
  of the converged Sinkhorn are kept.
- **Frozen per-region assignment:** q_i = softmax_k(x_i·C_k/τ + b_k) — per region, independent of batching.
- **τ (diffuseness) calibration, training-only:** per (representation, K, L), τ ∈ {0.01, 0.02, 0.03, 0.05, 0.1} whose
  mean normalised entropy H(q)/log K on the fit sample is closest to 0.4.
- **Calibration diagnostics (training data only, recorded for every representation, K, L):** cluster sizes,
  assignment entropy, stability (Hungarian-matched argmax agreement between fits on two disjoint 50k samples, scored on
  a third held-out 20k sample), and forecast-cluster predictiveness = held-out gain in nats/token of a per-cluster
  (q-weighted) unigram model of the region's actual tokens over the global unigram (the region's real tokens are used
  only for this diagnostic, never as an input).
- Prototypes, τ, b and assignments are written once (0400, sha256) and are never updated by main-model training; the
  latent losses receive q as a constant (no gradient path into assignments, encoder or aggregator; test T4).

## 6. FL-EDL objective (training only)

For micro-batch m, layer l, valid region i with tokens T_i: r_i^l = mean_{t∈T_i} p_t^l (full router softmax, fp32).
R_k^l = Σ_i q_ik r_i^l / Σ_i q_ik (clusters with Σ_i q_ik < 1e-6 are dropped).
L_FL-EDL = mean_{l∈Λ, k<j} −log(JSD(R_k^l, R_j^l) + 1e-8) (the same JSD/epsilon code as lfr EDL).
Total loss: L_LM + 0.01·L_Switch + 0.001·L_z + β·L_FL-EDL, β ∈ {5e-5, 1e-4, 5e-4}.
Λ = upper half of MoE layers = MoE indices 5–10 (model blocks 6–11; 6 of 11), or all 11 (registered comparison).
Routing is untouched: ordinary top-4 over all 32 experts, shared expert, same Switch/z losses; no masks, slots, mappings
or chunk routing. q never enters a forward pass; inference never sees q.

## 7. Conditions (final; all at FINAL scale, identical recipe within a seed)

| id | condition | definition |
|---|---|---|
| T4 | TOKEN-4 | no auxiliary loss. s0 reused (if T-EQ passes); s1, s2 new |
| EDL | explicit-label EDL (oracle) | lfr condition B exactly: genuine domain labels, per-segment mean router softmax → per-domain mean → −log(JSD+1e-8), all 11 MoE layers, β = 5e-3 |
| PAST | past-only latent EDL | frozen FL-EDL config (K, τ rule, L, Λ, β) but assignments from the past-only representation |
| FL | FL-EDL | frozen winner |
| SHUF | shuffled-cluster FL-EDL | frozen winner; within every micro-batch the q vectors of valid regions are permuted by a seeded permutation (seed = f(seed, step, micro)); preserves per-batch cluster mass and frequency exactly, destroys region–forecast correspondence |
| FD | forecast distillation, no divergence | frozen winner's q; training-only linear head on region means of the final hidden state predicts q (soft cross-entropy), coefficient γ; no divergence loss. γ = gradient-matched: γ = median backbone-gradient norm of β·L_FL-EDL of the frozen winner over its stage-1 diagnostic steps ÷ median backbone-gradient norm of L_FD(γ=1) over a 5M-token calibration run (seed 100, training data); 2 significant digits. The head is deleted at inference |

Two seeds (0, 1) for each newly trained condition. **Third-seed rule:** if the two seeds disagree in the sign of
FL − T4, or |mean(FL − T4)| ≤ 0.01 nats, or the specialisation verdict (section 9) differs between seeds, seed 2 is run
automatically for T4, FL and PAST.

## 8. Training-only search (seed 100; CALIB-SEARCH then CALIB-SELECTION; VAL never read)

**Base configuration B:** A1 meanstat, D = 4, H = 32, L = 256, K = 16, β = 1e-4, Λ = upper half.

**Stage 1 (5M tokens):** registered candidates, each one factor from B (one Stage-1 run each, FINAL plan stopped at
the 5M step with a resumable checkpoint):
C01 B · C02 K=8 · C03 K=32 · C04 β=5e-5 · C05 β=5e-4 · C06 L=128 · C07 A2 deepsets-cv · C08 all layers ·
C09 horizon 16 (first 16 generated tokens) · C10 drafts {G0, G1} only.
Each candidate is diagnosed (section 8.2) before the next is launched. Up to 2 additional diagnosis-driven candidates
may be added in stage 1 using only permitted dimensions, by this table (first matching flag of the incumbent = the
best-ranked eligible candidate):

| flag | response |
|---|---|
| aux gradient weak (median ‖∇β L_FL-EDL‖ / ‖∇L_LM‖ < 1e-3) and no JSD change vs T4 | β ×10 of the incumbent if ≤ 5e-4 exists unrun, else K = 8 |
| aux interferes (ΔCE > +0.02 and aux/LM gradient ratio > 0.1) | β ÷ 2 within the set {5e-5, 1e-4}, else Λ upper half |
| diffuse assignments (H(q)/log K > 0.7) | τ rule target 0.25 instead of 0.4 (optimisation fix) |

**Promotion.** Eligible = valid (finite; every expert load in [0.1/32, 4/32] in every layer on the measurement set;
mean normalised routing entropy in (0.2, 0.98); no data/causality violation) and ΔCE = CE_cand − CE_T4 (same step,
CALIB-SEARCH half B, 0.4M tokens/domain) ≤ +0.02 (stage 1), ≤ +0.015 (stage 2). Rank eligible candidates by the mean of
their ranks in FSI_domain and FSI_cluster (dev audit, section 9); ties → lower ΔCE. Stage 1 promotes ≤ 4, stage 2 ≤ 2.
If none is eligible, the single lowest-ΔCE valid candidate is promoted and flagged INELIGIBLE.

**Stage 2 (20M):** promoted candidates continue from their own 5M resumable state (same config, same cache), compared
with T4 at 20M. **Stage 3 (50M):** ≤ 2 candidates continue to 50M; CALIB-SELECTION is opened once for these and T4
(CE on CALIB-SELECTION, 0.4M tokens/domain; audit with discovery = CALIB-SEARCH half A, measurement = CALIB-SELECTION
0.2M tokens/domain; e5 cluster labels assigned to the frozen lfr centroids). Winner = the candidate with
ΔCE_sel ≤ +0.01 and the best mean FSI rank on CALIB-SELECTION; if none has ΔCE_sel ≤ +0.01, the lowest-ΔCE_sel one
(flagged INELIGIBLE). `frozen.json` (0400, sha256) then fixes every choice. VAL opens only after the final runs.

**Search compute cap:** ≤ 35 % of 16 GPU-h = 5.6 GPU-h (cache/aggregator costs included). Candidates are dropped from
the end of the stage-1 list if the cap would be exceeded; the final comparison is never reduced.

### 8.2 Diagnostics recorded for every search candidate

LM loss / CE; auxiliary loss; LM, Switch+z and FL-EDL gradient norms on backbone parameters (separate autograd.grad on
micro-batch 0 every 25 steps; numerically identical update); router entropy; mean pairwise JSD between latent-cluster
routing means; per-layer expert-marginal JS divergence from T4 at the same milestone (per domain); expert load / min /
max; cluster sizes, assignment entropy, stability, predictiveness; forecast agreement / disagreement; specialisation
by layer (MI per layer); ΔCE trajectory at 5M/20M/50M and the paired training-loss gap vs T4 metrics.

## 9. Specialisation evaluation (labels and encoders never used by any router or loss)

- **Label systems:** (a) genuine domain labels (5); (b) e5-base-v2 K = 16 clusters (lfr labeler centroids, 256-token
  windows); (c) analysis only: the frozen forecast-latent clusters of the winner (latent-cluster/expert MI).
- **Audit (lfr_audit code unchanged):** topic distribution and entropy per expert; topic/expert MI per layer with
  load-preserving shuffled-routing null; topic prediction from selected experts; within-expert semantic similarity;
  cross-seed Hungarian-aligned profile correlation with null; full router distributions; FSI (group ablation of each
  topic's lift-assigned experts, difficulty-matched by baseline-CE deciles) with shuffled-label control.
- **Per-expert counterfactual interventions** (finals at 100 %, conditions T4, EDL, PAST, FL; 0.05M tokens/domain of the
  VAL audit subset): for every (layer, expert), strongest domain k* by lift on discovery; disable only that expert
  (logit −inf, top-4 among the rest); ΔCE on held-out k* tokens minus difficulty-matched ΔCE on other-domain tokens =
  per-expert interaction; reported mean, fraction > 0, topic-by-expert matrix.
- **Sets:** dev = discovery CALIB-SEARCH half A (0.2M/domain), measurement CALIB-SEARCH half B (0.2M/domain); final =
  discovery CALIB-SEARCH half A, measurement VAL audit subset (first 0.2M/domain).
- **Trajectory:** search milestones 5/20/50M; finals 40/60/80/100 % (shuffled-label control at 100 % only; SHUF and FD
  at 40 % and 100 %).
- **Meaningful functional specialisation of FL (frozen):** (i) FSI(FL) > FSI(T4) at 100 % in every primary seed for
  both domain and e5-cluster labels; (ii) mean over seeds of FSI(FL)/FSI(T4) ≥ 1.15 for both label systems;
  (iii) FSI(FL) − FSI_shuf(FL) > 0 in every seed; (iv) mean per-expert interaction (domain) of FL > T4 in every seed.

## 10. Tests (before expensive training)

CPU suite with CUDA hidden (`fle_tests.py` plus the existing `tests/` pytest suite): T1 causal forecast construction
(changing tokens ≥ s_j leaves draft outputs for region j unchanged; generated tokens never read real future tokens);
T2 permutation invariance of A1/A2 over drafts; T3 balanced Sinkhorn (column sums equal within 1 %, rows sum to 1);
T4 frozen/detached assignments (q is a constant; no grad reaches cluster/encoder state; L_FL-EDL gradient only into
router/backbone); T5 FL-EDL correctness on a hand case (equals lfr edl_loss with one-hot q and region = segment);
T6 deterministic cache reuse (re-reading gives identical arrays and sha; rebuilding a chunk is skipped when complete);
T7 exact resume (CPU tiny run: stop/resume equals uninterrupted bitwise); T8 no validation access during search
(loaders refuse val/test; static scan; calib_sel opens only with STAGE3_OPEN); T9 student inference without draft
artifacts (forward accepts no q; works with cache directories hidden); T10 shuffled-cluster permutation preserves the
multiset of q per micro-batch; T11 matched initialisation and data order across conditions; T12 unrestricted routing
(4 distinct experts per token, shared expert always active, no masks).
GPU: T-EQ equivalence check vs lfr TOKEN-4 s0 (section 0); finiteness smoke of every condition (20 compiled steps);
resume check: a 60-step FL-EDL run stopped at step 30 and resumed matches the uninterrupted run within 3× the
registered run-to-run noise (lfr/cfe tolerance: 1e-4 nats per step on the logged loss when deterministic, else
0.003); final inference with the cache directory unavailable.

## 11. Success rules (frozen before VAL)

- CE: full VAL, token-weighted, per seed; paired per-document bootstrap 95 % CI (2000 resamples, documents averaged
  over seeds; never treated as seed replication); per-domain CE; seed spread.
- **Better:** mean(FL − T4) < 0 and FL − T4 < 0 in every primary seed. **Comparable:** mean(FL − T4) ≤ +0.01.
- Verdict (exactly one): (1) better + meaningful specialisation; (2) comparable + meaningful specialisation;
  (3) meaningful specialisation, CE worse than +0.01; (4) comparable/better CE, no meaningful specialisation;
  (5) neither.
- Decisive: FL vs PAST (CE and FSI). Also reported: FL vs EDL, FL vs SHUF (forecast semantics), FL vs FD (generic
  teacher effect), persistence of specialisation 40 → 100 %.
- No tuning after VAL is opened; failure analysis uses frozen diagnostics only.

## 12. Interruption, checkpoints, ledgers

`fle_driver.py`: atomic `driver_state.json`, append-only `driver_log.txt`, `run_ledger.tsv`, `compute_ledger.tsv`,
`checkpoint_ledger.tsv`; STOP marker `artifacts/forecast-latent-edl-v1/STOP`; GPU jobs only through
`scripts/gpu_guarded.sh`; foreign GPU process or signal → child stops at the next optimizer step with a verified
checkpoint, run marked INCOMPLETE/EXTERNALLY_STOPPED, driver exits, no automatic relaunch (restart needs
`--authorized-resume`). Search: one rotating resumable checkpoint per active candidate (`.tmp/forecast-latent-edl-v1/
slots`), deleted when the candidate is eliminated (ledgered); finals: KEEP = 1 rolling slot, milestones never deleted.

## 13. Budgets

- GPU: 16 GPU-h target (≤ 5.6 search). Plan: tests/checks 0.3; drafts 0.75; search cache 0.3; search 2.3; final cache
  0.4; finals 10 runs × 0.55 + 3 × 0.55 (seed 2); evaluations 2.5 → ≈ 13.7.
- Disk (persistent, new): drafts 0.3 GB; caches ≤ 9 GB; search slots ≤ 13 GB transient; milestones ≤ 13 GB; final
  slots ≤ 1.3 GB; logs/evals ≤ 3 GB → peak ≤ 40 GB < 50 GB cap. Before every job: free − planned peak ≥ 100 GiB, else
  checkpoint, stop and report the shortfall.
- Wall clock: 18 h target.
