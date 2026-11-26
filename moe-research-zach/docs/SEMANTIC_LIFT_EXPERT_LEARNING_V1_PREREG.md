# csr-semantic-lift-expert-learning-v1 — preregistration

Prefix `sll_`. Artifacts `artifacts/semantic-lift-expert-learning-v1/`, configs `configs/semantic-lift-expert-learning-v1/`,
temporary state `.tmp/semantic-lift-expert-learning-v1/`, runs `runs/<UTC>_semantic_lift_expert_learning_v1_<name>`.
Written 2026-11-18 before any candidate was trained or evaluated. Its sha256 is stored in
`docs/SEMANTIC_LIFT_EXPERT_LEARNING_V1_PREREG.md.sha256`; every sll script and config checks it. All earlier
experiments (semantic-edl-v1, semantic-edl-long-k, semantic-utility-edl-v1 = V1, soft-cf-usefulness-v2 = V2, fle, …)
are terminal and read-only here (protected inventory `protected_inventory_before.json`, 8,386 files).

## 1. Question and hypothesis

Can **Semantic Lift-Weighted Expert Learning** convert Semantic-EDL's subject-related routing preferences into
genuine subject-specific expert usefulness, while matching or beating explicit-label EDL in CE?

Hypothesis: Semantic EDL creates cluster–expert routing preferences, but experts keep receiving highly mixed learning
signals. Softly amplifying cluster k's ordinary LM gradient to an expert that cluster k disproportionately selects
(without changing the forward pass, top-k routing, gates, router/shared/attention gradients, the gradient into the
expert input, or the total gate-weighted gradient budget) should create the loop
routing preference → more cluster-specific expert updates → greater usefulness for that cluster → functional subject
specialization.

**Primary question.** Not whether E routes differently from C, but whether *semantic* group information makes the
selected experts more useful (own-group minus matched-other-group replacement damage) than *equally persistent
shuffled* group information under the identical mechanism (E vs F).

## 2. Facts that motivate the design (measured before this preregistration)

- semantic-edl-long-k Phase 1 (245M, 4 seeds, VAL): C6 routes more domain-aligned than SHUF (domain NMI 4/4 seeds)
  but per-expert functional interaction C6 − SHUF is +2.3e-6 (2/4 seeds positive) — routing moved, function did not.
- V1 / V2 (counterfactual usefulness): estimator noise; no replicated functional advantage on held-out documents.
- Load balancing is healthy in every C6 / SHUF run (no collapse, no unused experts); low utilisation is not the
  established problem.
- **Pre-registration calibration** (`calibration.json`; reference checkpoints REF_C6_s100 / REF_SHUF_s100 at 20M and
  50M, 60 training-stream steps after each milestone, rho 0.98, n0 8; no candidate involved):
  - SEM table: mean |lift| 0.23–0.30, split-half (even/odd steps) r of the lift table 0.993–0.995, robust scale of the
    token lift over selected pairs 0.19–0.27, Kish ESS ≥ 234 segments per cluster.
  - SHUF table: mean |lift| 0.022–0.027, split-half r 0.24–0.33, robust scale 0.017–0.019, per-cell sampling noise
    SD 0.018–0.027 (the router cannot see random paper groups; their lift is mostly sampling noise).
  - With floor 0.02 the implied gate-weighted mean |m − 1| is matched between SEM and SHUF (λ 0.25: 0.07 / 0.07–0.08;
    λ 0.5: 0.14 / 0.14–0.16; λ 1.0: 0.24 / 0.25–0.27; clipped fraction at λ 1: 12–16 % / 11–18 %).

## 3. Fixed backbone (Semantic C6; nothing chosen here)

- Model / recipe identical to the reused runs: 103.6M LFR, 12 layers (11 MoE), d 384, 32 routed experts + shared
  expert, unrestricted dropless top-4, gates = softmax probabilities of the selected experts; FINAL plan (3,880 steps,
  245,032,457 tokens, plan sha f4bc3c93…), init and token order per seed, AdamW / WSD schedule, bf16, Switch 0.01,
  z 0.001, clip 1.0 (`fle_configs.MODEL/TRAIN`). Search and promotion runs stop at milestones of the same plan.
- Semantic-EDL term unchanged: `paper_latent` β 0.005 over all 11 MoE layers with the frozen K16 C6 table
  `artifacts/semantic-edl-v1/tables/table_SEM_T05.npz` (sha 5d23129b…89d2). Shuffled control: the existing persistent
  paper permutation `table_SHUF_SEM_T05.npz` (sha d374a4ec…69fe) — identical cluster sizes, persistence and q entropy.
- No extra parameters, tokens, steps, schedules, pools, ownership, chunk routing, orthogonality, utility banks,
  disjointness penalties or penalties on other groups. Real domain labels are never used for training, selection,
  clustering or engineering; they are opened only for descriptive VAL analysis after the configurations are frozen.

## 4. Mechanism (conditions E and F; `src/csr/sll_lift.py`)

Statistics (detached; segment = one document chunk of a packed row, the Semantic-EDL representation):
- per optimizer step, for each MoE layer l: per-segment mean router softmax p̄(seg,l,·); accumulate
  N[k,l,e] += Σ_seg q(seg,k) p̄(seg,l,e), D[k] += Σ_seg q(seg,k), D2[k] += Σ_seg q(seg,k)²;
- committed **after** the optimizer step as EMAs with ρ = 0.98 per step (ratio N/D = exponentially weighted mean; no
  zero-init bias); shrinkage P[k,l,e] = (N + n0·P̄[l,e]) / (D + n0) with n0 = 8 segment-weights toward the pooled
  mean P̄ = ΣN/ΣD; marginal P[l,e] = Σ_k π_k P[k,l,e], π_k = D_k/ΣD;
- lift L[l,k,e] = log(P[k,l,e] + 1e-6) − log(P[l,e] + 1e-6). Kish ESS per cluster and table stability (r vs the
  table 50 steps earlier) are logged.

Multipliers (only statistics from previous steps):
- token lift s[i,l,e] = Σ_k q(i,k) L[l,k,e] (q of token i's paper; zero q → s = 0);
- z = s / max(0.02, scale_l), scale_l = bias-corrected EMA (ρ 0.98) of 1.4826·MAD of s over the selected
  (token, expert) pairs of previous steps; floor 0.02 = measured per-cell sampling noise SD of the lift (section 2);
- r = exp(λ_eff · tanh z); m = clip(c·r, 0.5, 2.0) with c > 0 solved exactly (piecewise-linear root) so that
  Σ_{e∈S(i,l)} ĝ[i,l,e] m[i,l,e] = 1, ĝ = gates / Σ gates over the token's selected top-4; pads: m = 1.

Gradient operator: `ScaledExpertMLP` (custom autograd per expert MLP): forward identical; dx ordinary; dW1, dW2 use
the row-scaled upstream gradient (token i's contribution × m[i,l,e]). Router, gate, shared-expert, attention,
embedding, norm and dense-FFN gradients and the gradient into the expert input are unchanged.

Schedule (fraction of the registered 3,880-step plan; c = 194): steps [0, 194) collection, m ≡ 1 (native expert code
path); [194, 388) linear ramp of λ_eff from λ/194 to λ; ≥ 388 full λ. Consequence (registered): a 20M run (318 steps)
sees only the ramp (λ_eff ≤ 0.64 λ); 50M runs see 405 full-strength steps.

Conditions trained here: **E_λ** = Semantic C6 + lift (SEM table for EDL and lift); **F_λ** = SHUF C6 + identical
lift (SHUF table for EDL and lift). Same seed ⇒ same init and token order.

## 5. Correctness tests (before any search run)

CPU (`sll_tests.py`, tiny LFR model, float64 and CPU-bf16; done 2026-11-18, 29/29 pass): 1 λ = 0 bitwise identical to
the native path; m ≡ 1 through the custom operator bitwise identical; 2 forward/logits/top-4/gates identical;
3 router; 4 shared; 5 attention and other non-expert gradients identical; 6 expert-input gradient identical; 7 expert
parameter gradients = explicit per-example reference Σ_i m_i J_iᵀg_i (rel. diff < 1e-10); 8 gate-weighted mean 1
(exact projection = bisection reference); 9 bounds; 11 statistics = direct computation, no future information
(accumulate-without-commit leaves the table unchanged), schedule, static check that the lift module reads no
labels/splits.

GPU (`sll_gpu_checks.py`, through the guarded driver; must all pass before the search): G1 λ = 0 config vs the
reused REF_C6_s100 run: step-0 LM loss identical (|Δ| < 1e-6) and step-50 within the project's established GPU
nondeterminism tolerance (|Δ| ≤ 5e-4; semantic-edl-v1 noise level 4.3e-4); G2 CUDA bf16 m ≡ 1 custom operator vs native:
forward bitwise identical, gradients identical or within 1e-3 relative; G3 resume: a test-only schedule (frac 0.005,
λ 1.0, so the operator is active after step 38) run 0→60 uninterrupted vs 0→30 + resume→60: identical LM loss per
step within 5e-4, identical LiftStats table within 1e-4; G4 multiplier invariants logged in training (weighted mean
max deviation ≤ 1e-6, bounds); G5 test 12: no expert unused (training load min ≥ 0.1× uniform per layer) in the G3
runs; throughput overhead of λ > 0 vs λ = 0 recorded.

## 6. Reused conditions (no retraining)

| cond | 20M / 50M (seeds 100, 101) | 245M (seeds 0–3; VAL p100 full + PE100k evals exist in semantic-edl-long-k) |
|---|---|---|
| A TOKEN-4 | `runs/20261114T090727Z_forecast_latent_edl_v1_s_T4_s100`, `runs/20261115T103553Z_semantic_edl_v1_b_T4_s101`; V1 dev evals POST_T4_s100 (s20, s50), POST_T4_s101 (s50) | fle finals T4_s0–s2, slk P1_T4_s3 |
| B explicit EDL | `runs/20261115T083452Z_semantic_edl_v1_b_EDL_s100`, `…_b_EDL_s101`; V1 dev evals | fle finals EDL_s0–s1, slk P1_EDL_s2–s3 |
| C Semantic C6 | V1 REF_C6_s100 / s101 (ckpt s20, s50); V1 dev evals | slk P1_C6_s0–s3 |
| D SHUF C6 | V1 REF_SHUF_s100 / s101; V1 dev eval s50 (s20 dev eval of REF_SHUF_s100 is run once here, evaluation only) | slk P1_SHUF_s0–s3 |

All share the plan sha, model/train config and table hashes (verified by `sll_run.py reuse-check`, which also
verifies every reused milestone's sha256). The sll evaluator is a path-only copy of the V1/V2 evaluator (identical
measurement calls and label-free row order), so V1 dev evaluations are reused as cached results. Measurement
identity for VAL: before the final comparison, the sll final-mode VAL CE of reused P1_C6_s0 p100 must equal the
stored semantic-edl-long-k value within 1e-4; otherwise every reused 245M baseline is re-evaluated with the sll
evaluator (evaluation only).

## 7. Measurement (the existing exact functional-ablation machinery)

`slk_lib.per_expert_boot` (unchanged): for every (MoE layer, expert) cell, its subject k* = argmax lift of the cell
on discovery tokens (CALIB-SEARCH paper-half A); one exact equal-compute replacement pass per cell (remove the expert,
the next-ranked one is admitted); interaction = own-group ΔCE − difficulty-matched (CE decile × position bin)
other-group ΔCE; the condition-level mean over the 352 cells with a document bootstrap (1,000 resamples, fixed seed;
identical rows and weights for every model ⇒ paired differences). Groups for the primary metric: the frozen
semantic pseudo-clusters (SEM K16 tables of the calibration splits). **Primary interaction** =
mean interaction(E) − mean interaction(F) on the same rows (paired bootstrap of the difference). Also recorded:
own/other ΔCE, number and fraction of specialised cells (lift ≥ 1.5 and per-cell CI > 0), e5 system, pseudo MI/NMI,
entropy, load, generic importance.

Splits (each held-out partition is opened at most once, by guard files):

| use | rows |
|---|---|
| dev (search, selection, diagnosis) | CALIB-SEARCH paper-half B (discovery on half A) |
| 50M gate, generation 0 | CALIB-SELECTION paper-half 1, sub-half b (sha256('scu-q'|paper) ≥ 0.5; never opened) |
| 50M gate, engineering generation 1 / 2 | CALIB-SELECTION paper-half 0, quarter 1 / quarter 2 by sha256('sll-q'|paper) |
| final primary (245M) | CALIB-SELECTION paper-half 0, quarters 3–4 by sha256('sll-q'|paper) |
| final CE and descriptive real-domain analysis | VAL (full; opened only after all final training) |

The locked test split is never read.

## 8. Bounded search (20M, seed 100, dev)

Six runs: E_λ and F_λ for λ ∈ {0.25, 0.5, 1.0}, bounds [0.5, 2.0], to milestone s20 (318 steps). Evaluated at s20
with the dev evaluator. References: REF_C6_s100 / REF_SHUF_s100 / T4 / EDL at s20.

λ is **eligible** iff all hold:
1. CE: dev CE(E_λ) − dev CE(REF_C6_s100) ≤ +0.010;
2. function: Δ_λ = interaction(E_λ) − interaction(F_λ) > −5e-5 (not clearly negative);
3. multiplier activity / stability (E_λ, last logged step): gate-weighted mean |m − 1| ≥ 0.02 averaged over layers;
   lift-table stability r ≥ 0.8; weighted-mean max deviation ≤ 1e-6; bounds respected; no non-finite value;
4. load / collapse (E_λ and F_λ): no dev pathology (load_min ≥ 0.1× uniform, load_max ≤ 4× uniform, routing entropy
   ≥ 0.45 — the V1 rule).

Selection: λ* = argmax Δ_λ over eligible λ; if another eligible λ has Δ within 5e-5 of the best, the largest such λ
is chosen (larger strength = stronger test; differences inside 5e-5 are noise). Routing NMI, overlap or pseudo FSI
alone never select. If no λ is eligible: diagnose (section 11) and consider an engineering generation (section 10).

## 9. Promotion (50M) and the gate to 245M

E_λ* and F_λ* for seeds 100 (continued from the s20 slots) and 101 (from step 0), to s50 (793 steps). Dev evaluation
of all four at s50; then the 50M gate partition is opened once and E, F, REF_C6, REF_SHUF (both seeds) are evaluated
on it.

**Gate (all required):**
- G-F1: held-out Δ(E − F) > 0 in both seeds;
- G-F2: held-out mean Δ(E − F) > mean Δ(C6 − SHUF) over the two seeds (the lift adds a semantic functional
  advantage beyond Semantic EDL's own);
- G-CE: mean over seeds of dev CE(E) − dev CE(EDL) ≤ +0.005 (reused EDL s50 dev evaluations);
- G-H: no pathology for E or F (rule of section 8.4); multipliers active (gate-weighted mean |m − 1| ≥ 0.02) and
  invariants hold in both E runs.
Dev Δ(E − F) at 50M, the pooled paired bootstrap CI on the gate partition and every per-seed sign are reported.

## 10. Final (245M) — only if the gate passes

E_λ* and F_λ* for seeds 0, 1, 2, 3 on the full plan (milestones s20, s50, p040, p060, p080, p100). After all eight
complete, FINAL_OPEN is written; each is evaluated once in final mode (primary partition + VAL block identical to
semantic-edl-long-k `val --full --pe100k`). Reused C6/SHUF p100 checkpoints get the primary-partition measurement
(evaluation only; their VAL results are reused); T4/EDL p100 too (descriptive).

**Success (all required):**
- P1 Δ(E − F) > 0 in 4/4 seeds on the final primary partition;
- P2 paired document-bootstrap 95 % CI of the 4-seed mean Δ(E − F) excludes zero (lower bound > 0);
- P3 mean over seeds of VAL CE(E) − VAL CE(EDL) ≤ +0.005 (comparability margin);
- P4 no expert collapse / load imbalance in any E or F run (section 8.4 rule on the final measurement; training load
  logs) and multiplier invariants held.
Secondary (descriptive): CE vs TOKEN-4, EDL, C6, SHUF (per seed, paired per-document bootstrap on VAL); Δ(E − C6)
on the primary partition; real-domain VAL per-expert interaction (E − F, E − C6, E − EDL); real-domain routing NMI;
top-k overlap between domains; load and entropy; whether related semantic groups share experts (correlation of
cluster-pair lift profiles with cluster-pair q co-assignment); throughput and overhead.
**The method is called working only if P1–P4 all pass.**

## 11. Diagnosis (recorded for every generation that is evaluated)

- A multipliers ≈ 1: gate-weighted mean |m − 1| < 0.02 at full strength, or lift stability r < 0.5.
- B routing ↑ but function ≈ 0: pseudo MI(E) − MI(C6) > 0.005 in the measured seeds while Δ(E − F) ≤ 0 or its
  mean is within ±5e-5.
- C semantic and shuffled improve equally: interaction(E) − interaction(C6) and interaction(F) − interaction(SHUF)
  both > 0 and |Δ(E − F) − Δ(C6 − SHUF)| ≤ 5e-5.
- D function ↑ but CE ↓: Δ(E − F) > 0 (both seeds) but CE(E) − CE(C6) > +0.005 or the CE criterion fails.
- E CE ↑ without semantic function: CE(E) − CE(C6) < −0.005 (mean) without Δ(E − F) > 0 in both seeds.
- F both improve: Δ(E − F) > 0 in all measured seeds and CE criterion passes.
The first matching of D, F, C, B, E, A (in that order) is recorded as the primary diagnosis; all matching ones
are listed.

## 12. Engineering generations (at most two)

Triggered only by a measured failure (search: no eligible λ; gate: failed criterion), diagnosed first. Each
generation: a document `docs/SLL_GEN<n>.md` written and hashed before its results; the same backbone and mechanism
with exactly one changed factor from: lower λ (if CE degrades), EMA timescale ρ (if the lift is demonstrably unstable:
stability r < 0.8), warmup/ramp length, application to later MoE layers only (if earlier-layer gradients are shown to
drive CE degradation), modestly narrower bounds; reruns only the minimal candidate and its matched shuffled control at
the failed stage (search: 20M seed 100; gate: 50M seeds 100/101), evaluated on dev and, for a gate, on its own
held-out partition (section 7). Not allowed: more data/steps, VAL/test-based choices, labels, new groupings, capacity
changes, other objectives, removing the shuffled control, redefining specialisation.

## 13. Operations

- GPU: one job at a time via `scripts/gpu_guarded.sh` through `sll_driver.py`; caps (GPU-h): total 40; checks 1.5;
  search 3; promotion 6; generations 5; final 24.5. A stage that would exceed its cap stops cleanly.
- Disk: ≥ 100 GiB free at all times; peak storage estimated before every stage; no /dev/shm; own resumable slots are
  rotated (KEEP 1) and the final slot of a run is deleted only after the run is COMPLETE (or its stage decision no
  longer needs it) and its milestones verify (ledgered); nothing pre-existing is ever deleted, moved or modified.
- Interruption (signal, kill, foreign GPU process, STOP marker, launcher block): checkpoint at the next step
  boundary, stop the driver and children, release the GPU, mark PAUSED_BY_OPERATOR with the exact resume command;
  never restart automatically.
- Append-only logs: RUN_LOG.md, DECISION_LOG.md, OBSERVATIONS.md, deviations.jsonl, run_manifest.jsonl,
  compute_ledger.tsv, disk_ledger.tsv; RESUME.md after every job.
