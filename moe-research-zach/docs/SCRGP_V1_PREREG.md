# SCRGP-V1 — Subject-Conditional Residual Gradient Projection: preregistration

Experiment `scrgp-v1` (prefix `scrgp_`). Written and hashed on 2026-11-19, before any SCRGP gradient, virtual update or held-out quantity was computed.

**Stage-0 engineering already done before this document** (timing and numerics only; no SCRGP quantity inspected):
- `scrgp_bench.py`: timing.
- `scrgp_numerics.py`: bf16-vs-fp32 agreement of a generic mean-CE-gradient step on 64 half-A rows.
- The CPU unit tests listed in §4.

The hash is stored in `docs/SCRGP_V1_PREREG.md.sha256`. Every Stage-1 job checks it.

## 1. Question, prior evidence, status

**Question.** Can label-free semantic clusters produce *genuine subject-specific expert competence*? That means experts more useful *per invocation* for their assigned semantic subject under exposure-matched evaluation, while next-token CE is preserved.

**Prior evidence.**
- **semantic-lift-245m-confirm-v1 (outcome C).** Semantic Lift raised subject-specific routing and the ordinary replacement interaction (4/4 seeds). The exposure-matched interaction showed no advantage over shuffled groups (−4.5e-5, 1/4 seeds), and CE was slightly worse.
- **GC-EDL.** Single-view (256-token) expert-gradient fingerprints are noise (same-paper cosine 0.017). Structure is recoverable only by averaging within semantic groups.
- **slk Phase 1.** Semantic C6 vs shuffled: domain NMI was higher in 4/4 seeds, while the functional interaction showed no difference.

**Consequence.** Routing association, NMI, activation separation and the ordinary replacement interaction are **not** evidence of functional specialisation. Only exposure-matched usefulness counts.

**Staging.**
- **Stage 1** is a mandatory, frozen, held-out virtual-update test on existing checkpoints.
- **Stage 2** is a continuation pilot. It runs only if Stage 1 passes every criterion.
- **Stage 3** is the 245M confirmation. It runs only if Stage 2 passes.

A Stage-1 failure ends the experiment with a negative report. Thresholds are never weakened, noise is never normalised into a signal, and there is no search until something passes.

## 2. What is reused (no retraining)

**Checkpoints.** The frozen Semantic C6 245M checkpoints (slk Phase 1, `ckpt_p100`, step 3880, 245,032,457 tokens) for **every existing seed: 0, 1, 2, 3**.
- C6 = K16 SEM table sharpened (T = 0.5), all 11 MoE layers, β = 0.005, unrestricted top-4 routing and the shared expert.
- Runs are listed in `src/csr/scrgp_paths.py`.
- Each `state.pt` sha256 is verified against its `status.json` before use (`scrgp_reuse.py` → `reuse_manifest.json`).
- These hashes were already verified in `artifacts/semantic-lift-245m-confirm-v1/reuse_manifest.json`.

**Semantic assignment for CALIB-SEARCH.**
- **Source.** `artifacts/gc-edl-v1/latent/table_SEM_calib_search.npz` (sha `abdd94c5…`). These are the frozen, label-free GC-EDL SEM prototypes applied to CALIB-SEARCH papers; the training table does not cover CALIB-SEARCH.
- **Derived tables** (`scrgp_tables.py`, 0400, built before this document; `artifacts/scrgp-v1/tables/tables.json`):
  - `table_SEM_T05_CS.npz`, sha `3a360636…`: the C6 sharpening, q² renormalised.
  - `table_SHUF_SEM_T05_CS.npz`, sha `de6c4a99…`: the same rows permuted across the 1,001 CALIB-SEARCH papers. This is a persistent, load-preserving shuffled grouping with seed 20261120; hard agreement 0.062, chance ≈ 0.066.

**Rows.** `partitions.json`, sha `c9b7cbeb…`, 0400, label-free. CALIB-SEARCH only; VAL and test are never opened in Stage 1. Documents are taken in sha256("scrgp-order|paper|doc") order up to a token cap and packed into rows of 1,024.

| part | source | tokens | docs | papers |
|---|---|---|---|---|
| A1 (estimation) | paper-half A with sha256("scrgp-sub\|paper") < 0.5 | 400,182 | 407 | 151 |
| A2 (estimation) | paper-half A, other papers | 400,107 | 410 | 153 |
| B (held out) | paper-half B (`lfr_eval.paper_half`, salt "lfr-half") | 500,032 | 507 | 240 |

A1, A2 and B are paper-disjoint. No real-domain label is read anywhere in Stage 1: the batch builder `scrgp_lib.make_batch` never constructs domain fields.

## 3. The SCRGP quantities (exact definitions; `src/csr/scrgp_lib.py`)

**Notation.** For MoE layer l (block l + 1), routed expert e (W1 384×256, W2 256×384) and table τ ∈ {SEM, SHUF} with soft per-paper membership q_t(k), K = 16.

**Association.** A_τ(l, k, e) = 1[k = k*_τ(l, e)], where k*_τ(l, e) = argmax_k P(e ∈ top-4 | k) / P(e ∈ top-4).
- Computed from natural routing on all estimation tokens (A1 ∪ A2), with soft q weights.
- It is frozen and stop-gradient by construction.
- Every (l, e) has exactly one associated cluster per table, so both tables have 352 pairs.
- No lift threshold is applied, so the shuffled procedure is not degenerate.

**Forced fourth-slot insertion of e at layer l.**
- At every position where e is *not* among the natural top-3, slot 4 is replaced by e.
- The gate is the original slot-4 gate g4 = p_(4): the raw softmax probability, without renormalisation, as in the model.
- Positions where e is already 4th are unchanged.
- The three unrestricted top experts, the shared expert, attention and every later layer (natural routing) are unchanged.
- Positions where e is in the top-3 keep natural routing, and their use of e carries no gradient.
- The probed expert is evaluated in float64, so small updates are not lost to bf16/TF32 rounding. The rest of the network runs as in evaluation (bf16 autocast).

**Eligible tokens.** Tokens with a valid target where e is not in the top-3.

**Matching** (coarsened exact matching, ATT weights). Strata are:
- CE quintile (natural CE; cuts from A);
- position bin (< 128, 128–511, ≥ 512);
- rank bin of e (= 4, 5–8, ≥ 9);
- g4 tercile (per layer, from A);
- margin tercile (log g4 − log p_e; 0 at rank 4; per layer, from A);
- target-token frequency tercile (token-weighted unigram counts on A).

That gives 1,215 strata.
- **Positive weight:** w⁺ = q_t(k*).
- **Control weight:** w⁻ = (1 − q_t(k*)) × P(s)/C(s), where P and C are the positive and control masses of stratum s in the sample.
- Strata lacking either side are dropped.
- Both sides are normalised to sum to 1, so positive and control have identical stratum distributions, including the gate and margin distributions.
- Document length is reported as a balance diagnostic.

**Gradients.** All exact next-token-CE gradients with respect to (W1[e], W2[e]) under forced insertion. The weights are fixed per sample.
- d_τ(l, e) = ∇ Σ_t (w⁺_t − w⁻_t) CE_t = g_pos − g_control.
- g_glob,τ(l, e) = ∇ Σ_t w^g_t CE_t, with balanced weights w^g_t = Σ_k q_t(k) / (K M_k) and M_k = Σ_eligible q_t(k).
- u_τ(l, e) = d − (d·g_glob / (‖g_glob‖² + 1e-30)) g_glob.
- No normalisation of d or u is ever applied, and u may be (near) zero.

**Estimation.** On each sub-half separately (weights normalised within the sub-half). Then d_A = (d_A1 + d_A2)/2, g_A = (g_A1 + g_A2)/2 and u_A = proj(d_A, g_A).

## 4. Correctness tests (before any Stage-1 job)

**CPU** (`scrgp_tests.py`, tiny float64 models and real CALIB-SEARCH half-A rows):
- **T-NAT:** probe natural forward equals `LFRModel.forward`.
- **T-INS:** forced 4th-slot insertion equals explicit `Experts.forward` with the edited top-4. Rank-4 and e-in-top-3 tokens are unchanged.
- **T-OFF:** with probes disabled, the forward is bit-identical.
- **T-GRAD:** exact gradient equals central finite differences.
  - Last MoE layer: relative error < 1e-5.
  - Earlier layer: ε = 1e-7, relative error < 1e-3, because finite differences can cross downstream discrete top-k routing flips.
- **T-ISO:** gradient isolation. No model parameter requires or receives a gradient, and other experts do not affect the result.
- **T-BATCH:** multi-vector gradients equal separate calls; linearity holds.
- **T-PROJ:** orthogonality, |cos(u, g)| < 1e-12.
- **T-SOFT:** soft cluster weighting.
- **T-GATE:** identical matched stratum (gate) distributions.
- **T-ZERO:** identical positive and control distributions give d = u = 0.
- **T-NORM:** no per-signal norm amplification.
- **T-STRAT:** strata.
- **T-RESUME:** a Stage-1 estimation interrupted after every micro-batch and resumed equals an uninterrupted run, exactly (CPU).
- **T-EVAL:** evaluation end-to-end with exact restore of the model hash.

**GPU** (`scrgp_gpu_tests.py`):
- **G-GRAD:** GPU gradients agree with a CPU fp32 reference on the same rows, cosine ≥ 0.99.
- **G-OFF:** the probe natural CE agrees with `LFRModel.forward` CE (mean |Δ| < 2e-3), and natural_updated(θ' = θ) is bit-identical.
- **G-RESUME:** an interrupted and resumed GPU estimation (test pairs) agrees with an uninterrupted one, cos(u) ≥ 0.999. The GPU is nondeterministic.
- **G-EVAL:** an evaluation smoke run restores the model hash.

Results go to `cpu_tests_*.json` and `gpu_tests_*.json`. **Any failing test blocks Stage 1.**

## 5. Stage 1 — frozen virtual-update test (`scrgp_stage1.py`)

### 5.1 Estimation (per seed, half A)

- **Pass 0** gives the cuts and the association k*.
- **Pass 1** gives the matching totals per sub-half.
- **Pass 2** gives exact gradients for all 352 pairs × 2 tables, the two vectors (d, g) each, per sub-half.

### 5.2 Split-half and other stability measures (reported; criterion V7b uses them)

- **Per pair and table:**
  - cos(d_A1, d_A2), cos(u_A1, u_A2), cos(g_A1, g_A2);
  - ‖d‖/‖g‖ and ‖u‖/‖d‖ without any normalisation;
  - the u-g orthogonality residual.
- **Distributions** by layer, cluster (k*), expert and seed.
- **Concentration:** the share of Σ‖u‖² carried by the top layer, top cluster and top 5 % of pairs.

### 5.3 Virtual update (per seed, half B)

**Step size.** One global step size per checkpoint, the same for both tables:
- η_target = 0.002 / median_{l,e,τ} ‖u_A‖². The median in-sample first-order predicted reduction of (L_pos − L_ctrl) is then 0.002 nats, since d·u = ‖u‖².
- The step is capped: η = η_target × min(1, 0.05 / median_{l,e,τ} (η_target ‖u‖ / ‖θ_e‖)). The median relative step is therefore ≤ 5 %, inside the linear regime measured in Stage-0 numerics.
- No per-pair normalisation.

**Per pair, using temporary tensors (the model is never modified).** For each (l, e) and τ, θ_e′ = θ_e − η u_A,τ(l, e). On B:
- **dF_τ:** per-token change in CE under forced insertion of e (θ_e′ vs θ_e).
- **dN_τ:** per-token change in CE under natural routing (θ_e′ vs θ_e).

**Simultaneous update.** All 352 pairs per table are updated in place (copies kept). Then natural CE on B is computed, the model is restored, and the state hash is verified.

**Per-pair metrics.** Paper-level sums are kept for the bootstrap. Matching weights are held fixed at their full-sample values, with strata ratios computed on B and the A cuts.

| metric | definition |
|---|---|
| **F_τ (primary)** | −[mean_w⁺ dF − mean_w⁻ dF]: assigned-cluster vs matched-control change under forced, equal exposure (positive = the subject residual improved the assigned subject more) |
| **G_τ** | balanced global forced change: mean over clusters k of the q-weighted mean dF over eligible tokens (positive = worse) |
| **N_τ** | exposure-matched natural change in the exact replacement interaction (slc I_cond strata: CE decile × position bin × rank 1–4 × gate decile × margin decile, margin p_e − p_(5), deciles on B; tokens with e in the natural top-4): −[own(k*) mean dN − matched other mean dN]. This equals the change in the exact expert-replacement interaction, because CE with e replaced does not depend on θ_e. |
| **Sim_τ** | natural mean CE change on B from the simultaneous update (positive = worse) |
| predicted | η‖u_A‖² (in-sample first-order); realised / predicted ratio |

**Seed-level statistic.** Mean over the 352 pairs. NaN pairs, where a side is empty, are excluded and counted.

**Statistics.** Hierarchical bootstrap over seeds and papers (crossed design, as in slc).
- 1,000 paper replicates on B (seed 20261121), shared across seeds and pairs.
- 10,000 hierarchical replicates: seeds drawn with replacement, plus one shared paper replicate (`slc_lib.hier_boot`).

### 5.4 Pass criteria (all required; evaluated once by `scrgp_analyze.py`)

| id | criterion |
|---|---|
| V0 matching | on A, in every seed: median over pairs (SEM) of max_vars \|SMD\| ≤ 0.10 for CE, position, rank, log g4, margin, log frequency, log doc length; median matched positive-weight fraction ≥ 0.80 (A1, A2 and B) |
| V1 direction | mean over seeds of F_SEM > 0 |
| V2 interval | hierarchical 95 % CI of F_SEM excludes 0 (lower bound > 0) |
| V3 seeds | F_SEM > 0 in ≥ 3/4 seeds |
| V4 global CE | (a) mean G_SEM ≤ 0.5 × mean F_SEM, and (b) mean Sim_SEM ≤ +0.001 nats |
| V5 exposure | mean N_SEM > 0 and N_SEM > 0 in ≥ 3/4 seeds |
| V6 concentration | using the seed-mean per-pair F_SEM: (a) leave-one-layer-out and leave-one-cluster-out (by k*) means all stay > 0; (b) top 5 % of pairs (18) ≤ 50 % of ΣF, the largest layer ≤ 50 % of ΣF, the largest cluster ≤ 50 % of ΣF |
| V7 semantic > shuffled | (a) Δ = F_SEM − F_SHUF: mean > 0, > 0 in ≥ 3/4 seeds, hierarchical CI excludes 0; (b) mean split-half cos(u_A1, u_A2) is larger for SEM than for SHUF in ≥ 3/4 seeds |

**Stage 1 passes iff V0–V7 all hold.**

### 5.5 Failure-mode classification (reported if Stage 1 fails; all that apply)

| failure mode | rule |
|---|---|
| unstable subject-differential gradients | mean (seeds, pairs) split-half cos(u_A1, u_A2) for SEM < 0.10 |
| no difference from shuffled groups | V7 fails |
| global-CE conflict | V4 fails |
| matching failure | V0 fails |
| insufficient semantic information | median routing lift of k*(SEM) < 1.5, or mean ‖d_SEM‖ ≤ 1.1 × mean ‖d_SHUF‖ |
| layer / expert / cluster concentration | V6 fails |
| first-order prediction fails held-out | realised mean F_SEM < 0.25 × mean predicted (η‖u‖²) while V0 holds |

**If Stage 1 fails:** stop. Write `SCRGP_V1_REPORT.md` and `SCRGP_V1_DIAGNOSTIC_NOTES.md`. No Stage 2, no new training.

## 6. Stage 2 — controlled continuation pilot (only if Stage 1 passes; spec frozen here)

**Start.** Semantic C6 `ckpt_s50` (step 793, 50,047,429 tokens) for screening seeds 0 and 1.
- These checkpoints hold bf16 weights only, with no optimiser state. Each condition therefore starts from the same weights cast to fp32, with fresh AdamW and the original hyperparameters.
- The LR follows the original plan from step 793, with an identical 20-step linear re-warmup for every condition.
- Data: the original plan rows from step 793 onward, identical across conditions, never seen by the s50 checkpoints.

**Conditions** (identical rows, order, steps and tokens):

| condition | description |
|---|---|
| A | Semantic EDL (C6 loss, main-split SEM_T05 table) + SCRGP with the SEM table |
| B | Semantic EDL + SCRGP with the persistent shuffled table (SHUF_SEM_T05; main split) |
| C | Semantic EDL continuation without SCRGP |

Explicit EDL and TOKEN-4 are referenced only descriptively; no compatible continuation exists.

**SCRGP update.** For every routed expert W1/W2 only: g_update(e) = g_LM(e) + λ Σ_k A(l, k, e) u(l, k, e), added to `.grad` before global clipping.
- **Refresh.** u is refreshed every 40 optimiser steps with the exact Stage-1 estimator on 16 rows of the current step's data. It is EMA-accumulated, with ρ = 0.9 applied separately to d and g before projection.
- **Association.** A is refreshed from running routing counts on training batches.
- **λ.** A global scalar fixed once at continuation start from a CALIB-SEARCH half-A probe: λ = r × median‖g_LM(e)‖ / median‖u‖, with r ∈ {0.1, 0.3}. These are the λ candidates.

**Budgets.** A 20M-token screen (318 steps) for both candidates, then a 50M-token confirmation (793 steps) for the selected candidate. No 245M run unless both pass.

**Endpoint.** The exposure-matched functional endpoint S on CALIB-SEARCH half B: slc I_cond definition, clusters = SEM_T05_CS argmax, exact replacement.

**Promotion (each budget).**
- (i) S(A) − S(B) > 0 in both seeds, and the paper-bootstrap CI pooled over seeds excludes 0.
- (ii) S(A) − S(C) > 0 in both seeds.
- (iii) CE noninferiority on CALIB-SEARCH B: mean over seeds of CE(A) − CE(C) ≤ +0.005 nats, and ≤ +0.010 per seed.
- (iv) No pathology: load in [0.1, 4] × uniform; routing entropy ≥ 0.45.
- (v) Not explained by one layer or cluster (as V6 on the endpoint).

**Selection at 20M.** Choose the r with the larger mean S(A) − S(B) among CE-noninferior candidates. If none qualifies, Stage 2 fails.

**Compute.** Auxiliary compute is reported separately. If SCRGP adds > 5 % total compute, efficiency claims require a compute-matched C. GPU cap 12 h.

## 7. Stage 3 — 245M confirmation (only if Stage 2 passes; spec frozen here)

**Runs.** The selected configuration (A and B variants) from scratch, to exactly 245,032,457 tokens (plan sha f4bc3c93…), with four seeds (0–3).

**References.** Reused TOKEN-4, explicit EDL, Semantic C6 and SHUF baselines (slc reuse manifest).

**VAL.** Opened once after freeze.

**Primary endpoint.** Exposure-matched S, Semantic-SCRGP − shuffled-SCRGP. Clusters on VAL come from the frozen GC-EDL SEM prototype pipeline, built label-free before VAL results are opened. Hierarchical CI over seeds × documents; ≥ 3/4 seeds.

**CE noninferiority.** Mean over seeds of (SCRGP − C6) ≤ +0.005 nats.

Real-domain metrics are secondary and post hoc. "Better than EDL" is claimed only if the multi-seed VAL comparison supports it.

## 8. Integrity

Real-domain labels are never used for training, clustering, router inputs, gradients, association, matching, η or λ, thresholds, selection or stopping.

**Prohibited:**
- tuning on VAL or test;
- choosing seeds by outcome;
- discarding seeds;
- rerunning until favourable;
- changing metrics or thresholds after results;
- per-signal normalisation;
- adding tokens or steps to one condition;
- modifying baselines or pre-existing files.

The test split is never read. The protected inventory (`protected_inventory_before.json`, 9,837 files) is verified at the end.

## 9. Operations

**GPU.**
- Through `scrgp_driver.py` → `scripts/gpu_guarded.sh`, one job at a time, 90-min wall limit.
- Stage-1 jobs are legs: `--max-seconds` 4,800, exit rc 75, resumed by the orchestrator.
- Atomic rolling slot every ≤ 9 min, holding accumulators, cursor, cuts, association, matching totals and stage.

**Caps (GPU-h).** Total 30; Stage 0 0.75; Stage 1 9 (estimated 6).

**Disk.**
- Keep ≥ 100 GiB free.
- Pre-job check: 100 GiB + estimated peak. Stage-1 peak ≈ 2.5 GiB per slot plus 0.6 GiB per est file.
- No /dev/shm.
- No deletion of pre-existing files. Only this experiment's superseded slots are deleted, after their outputs are verified.

**External interruption** (signal or kill, foreign GPU process, STOP marker, missing GPU):
- save the slot;
- mark `INCOMPLETE_STOPPED_BY_EXTERNAL_OPERATOR`;
- record the resume command;
- do not relaunch; release the GPU; wait for explicit user permission.

**Logs** (append-only): `decision_log.jsonl`, `observations.jsonl`, `deviations.jsonl`, `run_manifest.tsv`, `compute_ledger.tsv`, `RESUME.md`.

## 10. Permitted engineering changes

Allowed: efficiency, numerical correctness, batching, memory and matching implementation, provided the definitions in §3 and §5 are unchanged.

Each iteration is logged *before* it runs, with:
- the observed failure;
- the causal explanation;
- the exact change;
- the expected falsifiable effect;
- then the result and the decision.

Not allowed:
- switching method;
- adding labels;
- unequal tokens;
- tuning on VAL or test;
- re-inspecting CALIB-SELECTION (never used here);
- redefining the endpoint.
