# csr-context-prior-specialization-v2: preregistered protocol

- **Frozen:** 2026-11-12, before any V2 GPU work.
- **Changes:** only through `artifacts/context-prior-specialization-v2/deviations.jsonl`.
- **Code:** `cps_*`, `src/csr/cps_*`. Hashes are in `preregistration_hashes.json`.
- **Predecessor:** csr-latent-forecast-router-distill-v1 (V1). On user instruction, V1 was stopped before its final
  evaluation (`LATENT_FORECAST_ROUTER_DISTILL_V1_REPORT.md`). V1 and fspv2 are never modified. A protected inventory
  (`protected_inventory_before.json`) is verified at the end.

## 1. Objective

Keep ordinary unrestricted top-4 token routing, and add a causal, slowly changing chunk/context prior:

`logits(t,e) = W_l h_t[e] + alpha_l(t) * prior_l(c_t)[e]`

The aim is functional expert specialisation that is significantly stronger than TOKEN-4, at accuracy comparable to
TOKEN-4. The primary comparator is **TOKEN-4**. EDL is descriptive only and is not a success requirement.

## 2. Method (src/csr/cps_model.py)

**Backbone.** Identical to TOKEN-4:
- built by `LFRModel.__init__` before any V2 module exists, so parameters are bit-identical within a seed (test T01);
- 12 layers, d384, layer 0 dense, layers 1–11 with 32 GELU experts plus the shared expert;
- unrestricted top-4, with gates equal to the softmax probabilities;
- the shared-expert path is unchanged.

**Context state.**
- `s_t` is a multiscale causal within-document EMA, at half-lives `mem_hls`, of `P RMSNorm(x_t)`. Here `x` is the
  residual stream entering the first injected MoE block.
- Student encoder (inference): `c^S = RMSNorm(W2 GELU(W1 s))`.
- Teacher encoder (training only, prefix `tc_`): `c^T = RMSNorm(W2' GELU(W1' [s, z_t, EMA16(a_t), EMA128(a_t)]))`.
  - `z_t` comes from the V1 deepsets aggregator over the frozen draft ensemble's cached causal forecast features
    (ensemble top-4 next-token prediction and per-draft statistics).
  - `a_t` holds forecast entropy, draft agreement and draft disagreement.
- During teacher phases the executed context is `lam c^T + (1 - lam) c^S`.
- In the student-only phase and at inference the executed context is `c^S`. No drafts, cache or features exist there;
  the teacher is deleted at the start of P3.

**Chunk updates.** The context of token t is the state at the last position before t's chunk (`src = b(t) - 1`). Within
a chunk it is constant. The first chunk of each document gets the null context (prior 0). Two boundary families:
- **fixed:** boundaries at positions 0, L, 2L, … of each document.
- **adaptive:** rising edges of `d_t = 1 - cos(EMA4(R x), EMA64(R x)) > theta_L`, plus forced boundaries every 4L.
  - `R` is a frozen random token embedding that is never trained (seed 20261112).
  - `theta_L` was frozen before any V2 run by `cps_detector_calib.py` on CALIB-SEARCH half A (engineering data),
    giving mean chunk length L: theta = 0.657 / 0.758 / 0.834 / 0.898 for L = 16 / 32 / 64 / 128
    (`detector_calibration.json`).

Real future tokens are never used: tests T03–T05 on CPU and the fp32 GPU checks.

**Prior and gate.**
- `prior_l(c) = U_l c` (32 logits), injected at layers `inject` ∈ {all, middle = MoE 3–7, late = MoE 6–10}.
- With `prior_norm = match`, the prior is centred and rescaled to the RMS of the centred token logits (detached).
  `alpha_l(t)` then sets its weight relative to the token router.
- `alpha_l(t) = alpha_max * sigmoid(b_l + w_l·c_t + v_l·RMSNorm(h_t))`, with `b_l = alpha_init` (−4.6 gives about
  0.01; T06) and `w = v = 0` initially.
- The gate is trained only by the LM objective, plus an optional L1 term `alpha_l1 * mean alpha`. The utility and
  specialisation losses never reach alpha.

**Unrestricted routing.**
- No expert is ever masked, removed or made mandatory, and the token router can always select any expert.
- M6 (`noresid`, development diagnostic only) uses `0.1 W h + prior`.

**Teacher → student.**
- An MSE between `c^S(s.detach())` and `c^T.detach()` is weighted by `c_kl` and by the phase teacher scale (encoder
  weights only).
- Phases: P1 at `lam_max`, P2 with lam annealed linearly to 0, P3 student-only. A schedule is valid only if the drafts
  are used on reused-cache steps, i.e. phases 1–2 end at ≤ 60% of FINAL (`teacher_end_ok`).

**Counterfactual utility supervision (M3).**
- At the fixed preregistered CF sample (V1 `cf_uniform`, `r_cf = 0.075`), first-order counterfactual utilities
  `Delta_e = L(ref) - L(e)` come from the LM gradient tap. These are V1's estimator; V1 validated it against exact
  substitution (Spearman 0.81, top-4 overlap 2.88) and V2 audits it again in-run every 100 steps.
- The prior logits of the executed context, and in teacher phases also of the student context, are trained to rank
  the experts by the standardised utilities, via `util_mode` ∈ {corr: 1 − Pearson, rank: softmax KL}, weighted by
  `c_util`.
- Inputs are stop-gradient: the loss reaches only the context encoders and `U`, never the backbone (T10).
- Applied in all phases (`cf_phases = all`); P3 uses no drafts.

**Utility-conditioned latent consistency (M4; `c_spec`).** On CF rows of different chunks:
- **Pull:** prior distributions are pulled together, by JS divergence, when contexts are similar
  (`cos > 0.5`, soft, T = 0.1) and the utility rankings agree (`corr > 0.2`).
- **Push:** priors are pushed apart, bounded by `JS <= 0.3`, when contexts and rankings both differ
  (`cos < 0.2`, `corr < 0`).
- **Positivity gate:** a pair counts only when, for both rows, the top-4 experts preferred by the prior have positive
  estimated counterfactual utility (sum of first-order Delta > 0).
- Contexts and utilities are detached, and no domain label is used. Load balancing (Switch aux 0.01, z 0.001) is
  unchanged.

**Training.**
- Identical to the V1 TOKEN-4 runs: balanced main stream (prefix plans), 64 rows/step, WSD `lr_at`, AdamW
  (0.9, 0.95, wd 0.1), clip 1.0, bf16.
- Gradient clipping: the backbone's gradient norm is clipped at 1.0 on its own, exactly the TOKEN-4 rule. The V2-only
  parameters (`cp_`, `tc_`) are clipped separately at 1.0, so auxiliary-loss gradients can never rescale backbone
  updates. (In V1 a joint clip let teacher-loss gradients shrink backbone updates.)
- V2 modules use fan-in initialisation (std 1/sqrt(fan_in)), so pre-RMSNorm activations are O(1). The V1
  aggregator keeps V1's initialisation. The backbone draw is unaffected. Both choices were fixed after a CPU dry run
  found std-0.02 encoder initialisation amplifying gradients about 100x, before any V2 result.
- V2 adds no training tokens, post-training or extra steps, and leaves the model size and active expert count
  unchanged.
- `cps_equiv_check.py` must show that the V2 trainer's TOKEN-4 variant reproduces the reused V1 TOKEN-4 run before
  any V2 result.

## 3. Conditions

| id | condition | definition |
|---|---|---|
| M0 | TOKEN-4 | **reused** V1 runs: S1 s100; S2 s100/s101; FINAL s0. FINAL s1 and TOKEN-COMPUTE are new runs of the V2 trainer's `token` variant (shown equivalent) |
| M1 | memory-only | V2 architecture with the student context only: no drafts, no utility, no specialisation loss |
| M2 | context prior | drafts in the teacher context; no utility, no specialisation loss (= C01) |
| M3 | utility context prior | M2 + CF utility supervision (= C02) |
| M4 | specialised context prior | M3 + utility-conditioned latent consistency (= C03, the base) |
| M5 | shuffled context | M4 knobs with contexts rolled across rows of the micro-batch, same column, own padding mask, in training and evaluation; preserves context statistics, destroys context–token correspondence |
| M6 | no token residual | `0.1 W h + prior`; development diagnostic only; never advanced |

Descriptive reused V1 references (not success requirements): OUTPUT-DISTILL, EDL and V1 MEMORY-ONLY development
results.

## 4. Reuse (verified by `cps_verify_reuse.py` → `reuse_manifest.json`; 31 checks, all ok)

| Item | Status |
|---|---|
| fspv2 data splits main, calib_search, calib_sel, val | match the data-manifest hashes |
| tokenizer | files hashed |
| six frozen drafts | match V1 draft provenance |
| V1 forecast cache, main and calib_search | every chunk file matches its COMPLETE sha |
| V1 TOKEN-4 milestones | re-verified; the V1 TOKEN-4 final s0 has never been evaluated on validation |
| V1 descriptive results | hashed |
| V1 labeler windows (calib, val) and centroids | hashed |
| V1 modules imported read-only | hashed; `lfr_audit.py` equals V1's preregistered audit code |
| V1 CF validation | passed |

Nothing that already exists is rerun unless the V2 change makes it incomparable. TOKEN-4 S1/S2 and FINAL s0 are
reused, including their V1 dev audits, which used the identical audit code and rows. FINAL s1 and TOKEN-COMPUTE did not
exist and are new.

**Disclosures.**
1. CALIB-SELECTION was opened once by V1's stage 3, for V1 candidates and references only. No V2 configuration has been
   evaluated on it. V2 opens it once, at its own stage 3, for S2-eligible V2 candidates only. The reused TOKEN-4
   CALIB-SELECTION CEs come from V1.
2. VAL has never been evaluated by V1 or V2. The V1 labeler computed e5 cluster labels for VAL windows; these are labels
   only, with no model evaluation.
3. The test split is locked and never read.

## 5. Splits

- **Engineering:** CALIB-SEARCH only. This covers dev CE, dev audits (discovery = half A, measurement = half B) and all
  search decisions.
- **Freezing:** CALIB-SELECTION, opened once at stage 3.
- **VAL:** opened only after `frozen.json` exists, exactly once per final run. No tuning follows.
- **Guards:** training code opens only `main` and `calib_search` (T13). Evaluation modes are guarded by `STAGE3_OPEN.json`
  and `frozen.json`.

## 6. Scales and seeds

| Scale | Tokens | Steps |
|---|---|---|
| S1 | 49M | 777 |
| S2 | 122.5M | 1939 |
| FINAL | 245M | 3880 |

- **Screen:** S1, seed 100 (one seed).
- **Survivors:** S2, seeds 100 and 101.
- **Final:** seeds 0 and 1 are both primary. There is no seed-2 extension and no seed selection.

## 7. Specialisation measurement

The V1 audit is reused unchanged: `csr.lfr_audit` via `csr.cps_audit.audit_v2`. Labels never enter a model. It covers:
- domain labels and e5-base-v2 K=16 latent clusters;
- topic–expert mutual information per layer, with a load-preserving shuffled-routing null;
- per-expert topic entropy;
- topic prediction from expert-selection histograms;
- within-expert semantic similarity;
- the functional specialisation index **FSI**, computed as follows:
  - disable topic k's inferred specialist group G_k (assigned on discovery data);
  - I_k = the CE increase on topic-k tokens minus the difficulty-matched (decile-reweighted) CE increase on other
    tokens;
  - FSI = mean_k I_k / a_k;
- the shuffled-label control.

V2 adds per-document aggregates of the ablation deltas. These support:
- a **paired document bootstrap** of `FSI(candidate) − FSI(TOKEN-4)` on the identical VAL measurement documents
  (2000 resamples);
- a **per-topic specialist test**: topic k has a functional specialist only if the bootstrap 95% CI of I_k is above 0,
  i.e. disabling it harms its specialty more than matched other contexts.

The document bootstrap is never presented as seed replication.

**Trajectory.** For final runs, milestones at 40/60/80/100% are audited on VAL after the freeze. The 40% milestone falls
in the teacher phase, but the audit still runs drafts-free in inference mode.

Cross-seed stability uses the Hungarian-aligned profile correlation of seeds 0 and 1, plus its null.

## 8. Bounded engineering (src/csr/cps_search.py; at most 12 candidates C01–C12; CALIB-SEARCH only)

**8.1 Space.**

| Knob | Allowed values |
|---|---|
| chunk | {fixed, adaptive} |
| chunk_len | {16, 32, 64, 128} |
| ctx_dim | {32, 64, 128} |
| mem_hls | {(16, 128, 1024), (8, 64, 512), (32, 256, 2048)} |
| inject | {all, middle, late} |
| alpha_init | {−4.6, −2.2} |
| alpha_max | {1, 0.5} |
| alpha_l1 | {0, 1e-3} |
| prior_norm | {match, none} |
| util_mode | {corr, rank} |
| c_util | {0, 1, 3} |
| c_spec | {0, 0.1, 0.3} |
| c_kl | {1, 3} |
| lam_max | {0.9, 0.5, 0} |
| schedule | {40/20/40, 30/30/40} |
| ctx_lr_mult | {1, 3} |
| r_cf | 0.075, fixed |
| cf_phases | all, fixed |

The base is fixed chunks, L 64, ctx 64, mem_hls (16, 128, 1024), inject all, alpha_init −4.6, alpha_max 1, no L1, match
normalisation, corr utility, c_util 1, c_spec 0.1, c_kl 1, lam_max 0.9, 40/20/40.

Stopping gradients from the specialisation loss into token representations is always on. The CF targets exist only
after the LM backward, so the "flow" alternative is not implemented. This is the conservative choice, as in V1.

**8.2 First wave (fixed).**
- Candidates: C01 = M2, C02 = M3, C03 = M4 (the base).
- Controls, which are not candidates and never advance: M1 (base memory-only), M5 (C03 knobs), M6 (C03 knobs).

**8.3 Diagnosis of every candidate** (`diagnose`). The reference is the reused V1 TOKEN-4 S1 s100 run: CE 3.7896, FSI
domain 0.00396, cluster 0.00469.

| Question | What is measured | Flagged when |
|---|---|---|
| Q1 does the prior carry useful preferences? | mean alpha at the end; CALIB-SEARCH CE with the prior disabled minus CE | inert if alpha < 0.03 or the gain < 0.002 |
| Q2 does token routing keep its flexibility? | prior/executed top-4 overlap, alpha | lost if CE gap > +0.01 and (alpha > 0.3 or overlap > 3.0) |
| Q3 degradation during teacher-controlled training | training LM-loss gap to TOKEN-4 at the end of P1, averaged over logged steps within ±4% of the boundary (same data, same steps) | degraded if > +0.02 |
| Q4 degradation during distillation | gap at the end of the run (last 4%) minus gap at the end of P2 | degraded if > +0.02 |
| Q5 does specialisation increase? | dev FSI domain | — |
| Q6 is specialisation functional? | MI | not functional if MI > TOKEN-4 but FSI ≤ TOKEN-4 |
| Q7 forecast-dependent or memory? | candidate vs M1 | — |
| Q8 does shuffled context reproduce it? | candidate vs M5 | — |

**8.4 Progress score.** Lower is better; 0 means both targets are met at this scale:

`score = max(0, CE − CE_T4 − 0.01)/0.01 + max(0, (FSI_T4 − FSI)/FSI_T4)`

The incumbent is the minimum score; ties go to higher FSI, then lower CE, then the earlier id. Class-H runs (non-finite
loss, an expert load < 0.1/32 or > 4/32, routing entropy outside [0.2, 0.98]) are excluded.

**8.5 Failure-response decision tree.**
- Apply the first matching rule, in this order, together with its first untried response.
- Each response is a single knob change to the incumbent.
- If no matched rule has an untried response, apply R7.
- Generation stops at 12 candidates, at the screen GPU cap, or when the tree is exhausted.

| Rule | Trigger | Responses, in order |
|---|---|---|
| R1 | Q3 degraded | lam_max 0.5 → lam_max 0 → alpha_l1 1e-3 |
| R2 | Q4 degraded | c_kl 3 → schedule 30/30/40 → lam_max 0.5 |
| R3 | Q2 lost | alpha_l1 1e-3 → inject late → alpha_max 0.5 |
| R4 | Q1 inert | alpha_init −2.2 → prior_norm none → ctx_lr_mult 3 |
| R6 | Q6 not functional | util_mode rank → c_util 3 |
| R5 | Q5 not increased | c_spec 0.3 → chunk_len 128 → chunk adaptive |
| R7 | all S1 criteria met | chunk_len 32 → mem_hls (8, 64, 512) → ctx_dim 128 |

Every candidate is reported (`candidate_table.tsv`, `diagnostics/*.md`), including failures.

**8.6 Advancement to S2.**
- Candidates advance only when they move toward **both** targets: CE − CE_T4 ≤ +0.02 and FSI domain ≥ FSI_T4 at S1.
  The best three by score advance.
- If none qualifies, the single incumbent advances, marked INELIGIBLE, for confirmation only.
- At S2, M1 and M5 run on the top survivor's knobs, both seeds.

**8.7 S2 eligibility.** The reused TOKEN-4 S2 values are CE 3.3018 / 3.2941 and FSI domain 0.00575 / 0.00626. A
candidate is eligible when:
- the mean CE gap is ≤ +0.01, and every seed is ≤ +0.015;
- FSI domain is above the same-seed TOKEN-4 in **both** seeds;
- there is no class-H failure.

**8.8 Stage 3.**
- If no candidate is S2-eligible, the result is **NEGATIVE at development**: CALIB-SELECTION and VAL stay untouched
  (`NEGATIVE_DEV.json`), no finals run, and V3 receives a new evaluation plan.
- Otherwise, write `STAGE3_OPEN.json` and evaluate CALIB-SELECTION once for each S2-eligible candidate.
- Choose the candidate with the highest mean dev FSI domain among those with selection CE ≤ TOKEN-4 + 0.01 in both
  seeds. The TOKEN-4 selection CEs, reused from V1, are 3.2685 / 3.2608.
- If none passes, freeze the S2-eligible candidate with the smallest selection-CE gap, flagged.
- Write `frozen.json`. Only then may VAL open.

## 9. Final stage (only after the freeze)

- **Runs** at 245M tokens, seeds 0 and 1:
  - the frozen candidate;
  - TOKEN-4 s1 (new);
  - M1 memory-only on the frozen knobs (attribution: forecasts versus memory);
  - TOKEN-COMPUTE:
    - TOKEN-4 trained on more tokens, so that its training GPU time matches the candidate's mean final training GPU
      time plus one seed's share of the reused V1 main forecast-cache GPU time and the draft-training time (877.1 s);
    - capped at 600M tokens;
    - this is the equal-total-compute comparison.
- **Evaluation:** VAL once per run, covering full-VAL CE (overall, per domain, per document), VAL CE with the prior
  disabled, and the milestone audits. The reused TOKEN-4 s0 checkpoint is evaluated by the same V2 job.
- **Inference cost** (`cps_bench.py`): forward-only throughput and active parameters of the frozen candidate (no drafts)
  against TOKEN-4.

## 10. Success criteria and interpretation (all per primary seed s ∈ {0,1}, paired with TOKEN-4 seed s, VAL)

| Criterion | Requirement |
|---|---|
| A (accuracy comparable) | CE_s(cand) − CE_s(T4) ≤ +0.01 in **every** seed |
| A+ (accuracy better) | CE_s(cand) < CE_s(T4) in every seed |
| B (specialisation significantly stronger) | in every seed, FSI domain at 100% exceeds TOKEN-4 with the paired document-bootstrap 95% CI lower bound > 0, **and** FSI cluster is above TOKEN-4 (point) in every seed |
| C (survives draft removal) | B is measured on the 100% student-only checkpoint, which has no drafts or teacher, so B implies C; the prior-off VAL CE is reported |
| D (no drafts, unrestricted top-4 at inference) | by construction (T07, GPU check `stripped`) |
| E (inference cost comparable) | candidate forward throughput ≥ 0.85 × TOKEN-4 and active parameters ≤ TOKEN-4 + 1% |

**Terminal status** (exactly one):

| Status | Condition |
|---|---|
| **FULL SUCCESS** | A+, B, E |
| **ACCURACY-COMPARABLE SPECIALIZATION SUCCESS** | A, B, E (not A+) |
| **ACCURACY IMPROVEMENT, SPECIALIZATION HYPOTHESIS UNSUPPORTED** | A+ and not B |
| **SPECIALIZATION WORKS BUT COSTS TOO MUCH ACCURACY** | B and not A (or B, A and not E: cost), named precisely |
| **NEGATIVE** | neither A+ nor B (including development NEGATIVE without a freeze) |
| **EXTERNALLY STOPPED** | operator or foreign GPU process |
| **FAILED DUE TO IMPLEMENTATION OR RESOURCE ERROR** | — |

Equal-total-compute (TOKEN-COMPUTE), forecast attribution (M1), shuffled control (M5), the cross-seed stability of
specialist profiles and the specialisation trajectory are reported. They are descriptive and do not change the status.

## 11. Forbidden (restated)

- validation or test data for engineering;
- repeated validation evaluation;
- seed selection;
- hiding candidates;
- extra tokens or post-training for V2 only;
- changing the model size or active expert count;
- future tokens or same-position targets as router inputs;
- domain labels in V2 training;
- forced expert assignments or masks;
- changing data or metrics after results;
- single-seed claims;
- calling an unfavourable result a bug;
- tuning after VAL opens;
- touching V1 or fspv2.

A negative final result leads to a read-only diagnosis and a separately preregistered V3 proposal with a new untouched
evaluation plan.

## 12. Tests and checks before results

- **CPU suite** `cps_tests.py` (T01–T15): backbone identity, TOKEN-4 variant equals TOKEN-4, chunk sources, adaptive
  causality, model causality in all modes, alpha initialisation, strip, document-aggregate FSI = V1 FSI, search
  determinism, gradient paths, CF records and shuffle donors, configs and cache coverage, split and label guards, the
  checkpoint module equals V1's, and the evaluation interface.
- **GPU fp32 causality** `cps_gpu_checks.py`.
- **Trainer equivalence** `cps_equiv_check.py`: step-0 LM within 1e-3 and gradient norm within 1%; step-50 LM within 0.03.
- **Smoke runs:** C03, M1, M5, M6, adaptive.
- **Resume equivalence** `cps_resume_check.py`.

Each must pass, or the driver halts. Bugs are fixed with a logged deviation; results are never changed.

## 13. Budget and interruption

- **GPU budget:** 26 GPU-h for V2. Stage caps: checks 1.0, screen 6.0, S2 8.0, finals 11.0.
- **Disk:** guard reserve.
- **Checkpoints:** every 15 minutes or 10% of a run, at phase boundaries and at 40/60/80/100% for finals. Each is
  written atomically through a temporary directory; the latest 2 rolling checkpoints and all milestones are kept.
- **GPU etiquette:**
  - before every GPU job, check GPU 0; if a foreign process is present, launch nothing, record WAITING_FOR_GPU and exit;
  - never kill, pause or modify a foreign process;
  - on SIGINT/SIGTERM/HUP, an operator STOP or a foreign process appearing: checkpoint at the next step boundary, mark
    EXTERNALLY_STOPPED, release the GPU and exit;
  - never auto-restart; resuming needs `--authorized-resume` on explicit user instruction.

## 14. Outputs

**Report:** `CONTEXT_PRIOR_SPECIALIZATION_V2_REPORT.md`.

**Artifacts** in `artifacts/context-prior-specialization-v2/`:
- `final_analysis.json`
- `run_manifest.tsv`
- `compute_ledger.tsv`
- `candidate_table.tsv`
- `specialization_trajectory.json`
- `counterfactual_utility_analysis.json`
- `deviations.jsonl`
- `reuse_manifest.json`
- `diagnostics/`

If the final result is negative, also write `docs/CONTEXT_PRIOR_SPECIALIZATION_V3_PROPOSAL.md`.
