# csr-gradient-consistent-latent-edl-v1 (GC-EDL-v1) — Preregistration

Prefix `gc_`. Artifacts: `artifacts/gc-edl-v1/`. Configs: `configs/gc-edl-v1/`. Temporary files: `.tmp/gc-edl-v1/`.
Runs: `runs/<UTC>_gc_edl_v1_<name>/`. Written 2026-11-15 before any GC-EDL fingerprint, cluster, candidate or result
exists. Every later change goes to `artifacts/gc-edl-v1/deviations.jsonl`, with revisions as separately hashed
addenda (section 9).

## 0. Primary question and terminal statuses

Can stable, label-free pseudo-domains discovered from functional LM gradients produce (1) genuine, reproducible
subject/functional expert specialisation and (2) VAL CE comparable to or better than unrestricted TOKEN-4, without
using explicit domain labels for discovery, training, selection or tuning?

The report ends with exactly one of: `GC-EDL-V1 COMPLETE — POSITIVE`, `GC-EDL-V1 COMPLETE — ACCURACY-COMPARABLE
SPECIALIZATION`, `GC-EDL-V1 COMPLETE — NEGATIVE`, `GC-EDL-V1 STOPPED — DISCOVERY INVALID`, `GC-EDL-V1 STOPPED BY
OPERATOR`, `GC-EDL-V1 STOPPED — INTEGRITY OR RESOURCE FAILURE` (mapping in section 11).

## 1. Inherited setup and reuse decisions (inspected before writing this document)

**Data.** fspv2 deduplicated five-domain M2D2 arXiv corpus (cs, math, physics, q-bio, stat), Pythia tokenizer,
segments ≤ 1024 tokens. Existing paper-disjoint buckets were assigned by paper hash (`u_of("fspv2-split", paper)`):
main (training, 52,592 papers, 613M tokens), calib_search (1,015 papers), calib_sel (1,057), val (2,042), draft_shared,
draft_cont and a locked test split. These buckets are kept: re-partitioning would mix the locked test papers into other
partitions and invalidate every reusable baseline. All new partitions (section 2) are made inside these buckets by
hash. **The test split is never opened or decoded.**

**Backbone and recipe** (identical to `csr-forecast-latent-edl-v1`, "fle"): `LFRModel`, linear router, 12 layers,
d 384, layer 0 dense, layers 1–11 with 32 routed GELU experts (FFN 256) and one always-active shared expert (FFN 768),
unrestricted per-token top-4 softmax routing, dropless, 103,609,728 parameters (43.05M active). AdamW (0.9, 0.95),
wd 0.1, lr 6e-4 WSD (1 % warm-up, last 20 % decay to 6e-5), clip 1.0, bf16, torch.compile, Switch balance 0.01,
z-loss 0.001, 64×1024 rows per step (32 × 2 accumulation), lfr balanced main stream.

**Token budget.** The FINAL plan of this pipeline: 245,032,457 tokens = 3,880 steps (plan sha `f4bc3c935a47…`).
Decision D1: "the full available training-token budget" is interpreted as this established FINAL budget. It is the
budget of every matched TOKEN-4 and explicit-EDL checkpoint the instructions ask to reuse when valid; training at the
613M-token size of the whole main split would make every existing baseline unusable and is not what the comparison
set was built for. Screening uses the 50M-token prefix of the same plan (793 steps, identical LR schedule: warm-up then
constant 6e-4).

**Reuse (all hashes re-verified by `gc_integrity.py` before use; fields compared programmatically).**
- TOKEN-4 finals s0, s1, s2 and explicit-label EDL finals s0, s1 of fle (`runs/*forecast_latent_edl_v1_final_{T4,EDL}_s*`,
  milestones p040/p060/p080/p100, `fle_train.py`). All scientifically relevant fields equal those of the GC runs below:
  data/split, tokenizer, architecture, init (seed), data order (plan), token budget, optimizer/schedule, batch
  construction, context, top-k, shared path, milestone/checkpoint policy, diag cadence (100 steps). Their VAL results
  (`artifacts/forecast-latent-edl-v1/evals/final_{T4,EDL}_s*/final.json`) were produced by `fle_evaluate.py final`,
  whose functions GC evaluation imports unchanged (section 10), so they are reused for CE, per-document CE and the
  domain / e5-cluster audits. New metrics (pseudo-domain systems, multi-label per-expert interventions, gradient
  coherence) are computed once on these checkpoints by a supplementary evaluation after the freeze.
- TOKEN-4 screening reference: fle `s_T4_s100` milestone `s50` (seed 100, step 793, `fle_train.py`).
- `gc_train.py` is `fle_train.py` with only (a) GC paths/prereg check, (b) a slot cadence of ≤ 600 s instead of 900 s
  (no effect on the update), and (c) the new `paper_latent` condition. The TOKEN-4 and EDL code paths are textually
  identical. Equivalence check T-EQ (section 7) must show bit-identical step-0 loss against `s_T4_s100`; if it fails,
  reuse is void and TOKEN-4 (screening and finals) and EDL finals are retrained with `gc_train.py`.
- The balanced main stream orders documents with domain labels (`balanced_order`). This is part of the shared data
  recipe, identical for TOKEN-4, EDL and every GC condition; it is not a discovery or training choice of GC-EDL and no
  label value reaches GC discovery, its loss or its selection.

## 2. Data isolation and partitions

- **Opaque paper ID**: `o = sha256("gcedl-v1-opaque|" + paper_hash)[:20]`.
- **Discovery export** (`gc_data.py export --split {main,calib_search,calib_sel}`; val only after the freeze) writes
  `discovery/inputs_<split>.npz` containing only opaque IDs (sorted by opaque ID, so file order carries no domain
  structure), view token arrays, view offsets, paper token length and segment count. No domain, domain name, path,
  paper hash or document ID is written. The opaque → paper map is written to `sealed/opaque_map_<split>.json`, read
  only by the trainer/evaluator (to map segments to q) and never by discovery.
- **Label firewall**: discovery programs (`gc_fingerprint.py`, `gc_discover.py`) install an audit hook that raises on
  any open of `*_docs.json`, the fspv2 data directory, `labels/`, `windows_*`, `sealed/`, `M2D2` raw data, or any path
  containing `domain`. Test L1 proves the hook blocks these accesses and that the discovery modules import no
  label-bearing module.
- **Discovery partitions of main papers by hash**: DISC-FIT if `u("gcedl-v1-disc", o) < 0.8`, else DISC-HOLD. The
  scout is pre-existing (no fitting). The nuisance model, standardisation and prototypes are fitted on DISC-FIT;
  discovery selection uses DISC-HOLD only. q is then computed for every main paper (FIT and HOLD) with the frozen
  pipeline, and for calib_search / calib_sel papers (label-free dev-audit pseudo-domain metrics).
- **Model selection**: CALIB-SEARCH (engineering, β calibration, revisions), CALIB-SELECTION (opened once, freeze).
  **VAL** is opened once after every configuration and checkpoint is frozen. Each final checkpoint is evaluated once.
- **Labels sealed** until `frozen.json` (the final GC candidate) is written and hashed. `UNSEALED.json` records the
  first access. Before that, no metric uses domain labels: screening and CALIB-SELECTION audits use only e5-cluster
  labels (label-free k-means of a frozen independent encoder, existing fle/lfr label files) and pseudo-domains; per-domain
  CE fields are dropped from screening outputs. The explicit-label EDL screening run is an oracle reference that
  trains on labels by definition; it never enters any selection.

## 3. Stage 1: functional gradient fingerprints

**Views.** Each paper's segments are concatenated in segment order (verified contiguous, no separators). With paper
length N: n_v = min(4, ⌊(N + 256) / 512⌋) (0 if N < 256), so any two views are separated by ≥ 256 tokens (never
adjacent). For n_v ≥ 2, start_i = round(i (N − 256)/(n_v − 1)); for n_v = 1 the view is centred. Views are 256
tokens; offsets are recorded. Papers with no view receive no q (zero weight in the GC loss).

**Scout.** fle generalist draft **G1** (seed 11; 21.4M dense, 10 layers, d 256, SwiGLU 768; trained one epoch on
draft_shared ∪ draft_cont — training-side papers disjoint from main/calib/val/test; no label in its loss; not chosen
with VAL; sha `3544d882…`). Rule: the first generalist trained by the fle code with complete recorded provenance; fixed
before any fingerprint exists. Frozen, fp32, TF32 off.

**θ_s** by the outcome-independent rule: the output projection of the final block's MLP = `blocks.9.ffn.w2.weight`
(256 × 768, no bias). Name and hash recorded.

**Gradient.** raw_g[d,v] = ∇_θs mean_token_CE(view) over its 255 next-token predictions, view fed alone (positions
0–255). Because θ_s = W₂ is linear, raw_g = Σ_t δ_t a_tᵀ / 255 with a_t the W₂ input and δ_t = ∂CE/∂(block-9 output)
(only norm_f and the tied head lie downstream). **Projection**: a seeded CountSketch (seed 20261115) with the
TensorSketch hash h(i,k) = (h₂(i) + h₁(k)) mod 128 and sign s₂(i)s₁(k); g = P vec(raw_g) ∈ R¹²⁸ is computed as
irfft(Σ_t rfft(CS₂(δ_t)) ⊙ rfft(CS₁(a_t))) — the per-example gradient is never materialised. ‖raw_g‖_F is computed
exactly from Gram matrices. Tests G1–G4 compare against explicit autograd gradients and an explicit P matrix.

**Direction vs magnitude.** u = g/‖g‖; ‖raw_g‖ is kept as a diagnostic and nuisance variable only.

**Nuisance residualisation** (DISC-FIT only, frozen). Per-view nuisance vector (31 features): scout CE; log ‖raw_g‖;
log paper length; relative view position; log #segments; mean log unigram probability (DISC-FIT unigram); fraction of
top-100 tokens; type/token ratio; 16-bucket hashed token-ID histogram; punctuation, digit, math/markup
(`\ $ ^ _ { } = < > | ~`), uppercase-of-letters, whitespace/newline rates; citation density
(`[n]`, `et al`, `\cite` per 1000 chars); mean word length. Standardised on DISC-FIT. Ridge from nuisance → each
coordinate of u, α by 5-fold paper-grouped CV on DISC-FIT from {1e-2, 1e-1, 1, 10, 100, 1000}. Residual
r = u − û: cross-fitted (5 paper folds) on DISC-FIT, full fit elsewhere. Functional fingerprint f = r standardised per
coordinate (DISC-FIT), f̂ = f/‖f‖. Reported: R² of nuisance → u; held-out (DISC-HOLD) R² of predicting each nuisance
variable from u and from f (before/after).

**Semantic view.** BAAI/bge-small-en-v1.5 (CLS pooling, L2-normalised, ≤ 512 tokens of the decoded view; copied into
`artifacts/encoder/bge-small-en-v1.5`, sha-verified). Independent of the FL-EDL teacher (MiniLM + drafts), the scout and
the e5-base-v2 evaluator. s standardised per dimension on DISC-FIT.

**Representations** (only these three): SEM = s̃ (384); GRAD = f (128); COMBINED = [s̃/√384, f/√128] (equal block
variance, 512).

## 4. Stage 2: balanced soft prototypes

Model (same for all): encoder z = normalize(W x + b) ∈ R⁶⁴, prototypes C ∈ R^{K×64} (normalised), logits ℓ = z·Cᵀ/τ,
τ = 0.1. Training on DISC-FIT papers with ≥ 2 views; each step samples two distinct views (v₁, v₂) per paper; batch
2048 papers; Adam lr 1e-3; 30 epochs; seeds 0, 1, 2.

Loss = L_cons + λ_coh L_coh + λ_bal L_bal + λ_ent L_ent, λ_bal = 1, λ_ent = 0.05,
λ_coh = 2 for GRAD and COMBINED and **0 for SEM** (SEM is the pure semantic control: no gradient information in its
input or objective; decision D2).
1. **Consistency** (swapped prediction): t(v) = Sinkhorn(ℓ(v), ε 0.05, 3 iterations) balanced over the batch, stopped
   gradient; L_cons = −½ [t(v₂)·log softmax ℓ(v₁) + t(v₁)·log softmax ℓ(v₂)].
2. **Cross-view gradient coherence**: centroids μ_k ∈ R¹²⁸ (stop-gradient, EMA 0.1, updated after the loss from
   t(v)-weighted f̂(v) of the batch, i.e. lagged). Prediction for v₂ from v₁: ĝ = normalize(Σ_k softmax(ℓ(v₁))_k μ_k);
   L_coh = −½ [cos(ĝ(v₁), f̂(v₂)) + cos(ĝ(v₂), f̂(v₁))]. A view is never scored against its own gradient.
3. **Balance**: KL(mean_batch softmax ℓ ‖ uniform).
4. **Confidence**: mean normalised entropy of softmax ℓ (gentle).
Paper assignment: q[d] = softmax(mean_v ℓ(v)), constant for all segments of d; training target only.

**Grid**: {SEM, GRAD, COMBINED} × K ∈ {4, 8, 16} × seeds {0, 1, 2} = 27 fits.

## 5. Discovery selection (label-free; DISC-HOLD only)

Metrics per (rep, K), seed-averaged, papers with ≥ 2 views, all ordered view pairs:
- κ_view: Cohen's κ between argmax assignments of two views of the same paper.
- κ_seed: mean over seed pairs of Cohen's κ after Hungarian matching (paper argmax).
- C (held-out gradient coherence): μ_k from DISC-FIT (p_k-weighted f̂); for held-out pairs
  cos(normalize(Σ_k p_k(v_i) μ_k), f̂(v_j)) minus the same with p from a random other held-out paper (shuffled papers).
- LD (language-distribution predictiveness): per-cluster unigram U_k from DISC-FIT (p-weighted counts, mixed 0.9/0.1
  with the global unigram); nats/token gain of Σ_k p_k(v_i) U_k over the global unigram on the tokens of v_j, minus
  the same with shuffled papers.
- Eligibility: paper-level hard cluster fractions in [0.2/K, 3/K]; exp(H(mean q)) ≥ 0.75 K; held-out ridge R² of
  scout CE and of log gradient norm predicted from p(v) each ≤ 0.5 (not difficulty bins).
- Composite S = mean(κ_view/κ*_view, κ_seed/κ*_seed, C/C*, LD/LD*), * = maximum over the eligible grid. SE by 500
  paper bootstrap resamples (fixed denominators).
- **K per representation**: smallest eligible K with S ≥ S_best(rep) − SE(S_best(rep)). The discovery winner (largest S
  among GRAD/COMBINED selections) supplies the shuffled control's table in screening.
- **Validity** (`discovery_decision.json`): (a) for at least one of GRAD/COMBINED, the selected K has 95 % bootstrap CIs
  of κ_view, C and LD all > 0 (beats shuffled-paper assignments); (b) C(GRAD or COMBINED, selected) − C(SEM, selected)
  has a 95 % CI > 0 for at least one of them. If (a) or (b) fails and no allowed fingerprint revision (section 9) can
  address the measured cause → `GC-EDL-V1 STOPPED — DISCOVERY INVALID`.
Diagnostics (not selection): same-paper vs different-paper cosine of f̂ and of u; most-separated-pair κ.

## 6. Stage 3: GC-EDL loss

Reuse of the explicit EDL implementation (`fsp_model.edl_loss`): per-sequence (segment) mean router softmax m_s, then
per-group means, then mean over group pairs of −log(JSD + 1e-8), averaged over all 11 MoE layers, coefficient β. GC-EDL
replaces the one-hot label by q: P_k = Σ_s q_sk m_s / Σ_s q_sk (groups with Σ_s q_sk < 1e-6 in a micro-batch are
skipped, which reproduces edl_loss exactly for one-hot q; test E1), JSD clamped at ≥ 0 (fle deviation 6). Routing,
top-4, shared expert, LM loss, Switch/z losses, data order, budget, optimizer and schedule are unchanged; q never
enters a forward pass and no expert is masked. Segments of papers without q have zero weight.

**Conditions** (seed 100, 50M screening): T4 (reused), EDL oracle (β 5e-3, labels), SEM, GRAD, COMBINED (each at its
selected K and three β), SHUF = discovery winner's table with q permuted across main papers by a fixed seeded
permutation (paper-level, constant per paper), at that representation's selected β.

**Auxiliary-strength calibration (CALIB-SEARCH / training batches only).** Pilot per representation: 80 steps at
β = 5e-3, gradient diagnostics every 10 steps; ρ = median over steps ≥ 20 of ‖∇ β L_aux‖ / ‖∇ L_LM‖ (backbone + router,
fle `grad_diag`). β_r = 5e-3 · r/ρ for r ∈ {0.5 %, 1 %, 2 %}, two significant digits.

**β choice per representation (CALIB-SEARCH half B, 50M milestone).** Reject: non-finite; any expert with load
< 0.1× uniform in any layer; routing collapse (routing entropy < 0.45 or any expert load > 4× uniform); dCE vs T4 >
+0.020. Among the rest: prefer dCE ≤ +0.010; then highest e5-cluster FSI (dev audit, discovery CS half A, measurement
CS half B); then lowest CE. If none ≤ +0.010, the lowest-dCE accepted setting. If none accepted, the representation has
no eligible candidate.

## 7. Correctness gates before GPU training

CPU suite (`gc_tests.py`, CUDA hidden): G1 streamed sketch = P·vec(explicit autograd gradient) on tiny models
(rel. error < 1e-5, float64); G2 Gram-norm = ‖∇‖_F; G3 fingerprints deterministic (bitwise replay) and
batch-composition tolerant (< 1e-4 relative); G4 tokens outside a view do not change its fingerprint; L1 label firewall
blocks label paths and discovery modules import no label module; L2 discovery export contains only allowed keys and no
paper hash/domain; L3 split isolation (paper-disjointness of all buckets and of DISC-FIT/HOLD); Q1 q constant within
paper and identical for all its segments in a training batch; Q2 GC loss invariant to permuting/zeroing domain labels;
E1 soft loss with one-hot q equals `edl_loss` (≤ 1e-6); E2 GC loss gradients reach only router/backbone, q has no
gradient; I1 inference checkpoint state-dict keys identical to TOKEN-4 and evaluation needs no scout, encoder,
prototypes or q; R1 resume determinism of the trainer on CPU (bitwise). Plus the existing repository test suite.
GPU (`gc_gpu_checks.py`): fingerprint GPU vs CPU agreement and causality on real data; finite GC loss; stop/resume
of a short GC run within max(0.003, 3 × measured noise); T-EQ step-0 bit identity vs `s_T4_s100` (steps 50/100
reported); hashes of all reused checkpoints.

## 8. Freeze (CALIB-SELECTION, opened once)

After screening and any revisions: evaluate once at 50M on CALIB-SELECTION (CE 0.4M tokens/domain; label-free audit:
discovery CS half A, measurement CALIB-SEL, e5 labels `windows_calib_sel.npz`): T4, EDL, the selected
SEM/GRAD/COMBINED candidates, SHUF. **Frozen GC-EDL candidate** among GRAD and COMBINED by priority: (1) finite and
valid; (2) no starvation/collapse; (3) dCE vs T4 ≤ +0.010; (4) highest e5-cluster FSI; (5) lowest CE. If neither meets
(3), the valid one with lower dCE is frozen (the best scientifically valid candidate) and the experiment continues to
its (likely negative) conclusion. SEM's final control uses its CALIB-SEARCH-selected (K, β); the final SHUF uses the
frozen candidate's table, permuted as above, at the frozen β. `frozen.json` (0400) + hash; then labels are unsealed.

## 9. Bounded engineering (at most two revisions; CALIB-SEARCH only; before CALIB-SELECTION)

A revision is allowed only when its measured trigger fires; the first fired trigger in this order is used:
1. **Unstable fingerprints**: discovery validity (a) fails, or κ_view(selected GC) < 0.2 → projection 256 (same views).
2. **Nuisance-dominated**: eligibility fails by the nuisance rule for all GC K → add squared and pairwise products of
   (scout CE, log grad norm, relative position) to the nuisance model.
3. **Assignment collapse/instability**: balance eligibility fails or κ_seed < 0.4 for all GC K → τ 0.1 → 0.2 (or λ_bal
   1 → 3 if collapse persists).
4. **Inert EDL gradient**: median achieved aux/LM ratio over the 25M–50M diagnostics < 0.25 % for every β of a GC
   representation → recalibrate β with ρ measured on those 50M diagnostics (targets unchanged: 0.5/1/2 %).
5. **Excessive auxiliary conflict**: every β of both GC representations is rejected for CE (> +0.020) or none is
   ≤ +0.010, while the EDL oracle is ≤ +0.010 → linear warm-up of β from 0 over the first 10 % of steps (total tokens
   unchanged).
Each revision is written as `docs/GC_EDL_V1_REVISION_<n>.md` and hashed before execution, receives the same 50M
screening, and stays in the ledger. No labels, VAL, larger models, extra tokens, extra parameters, seed picking or hard
masks. If no eligible candidate emerges after two revisions, the best valid candidate is frozen and reported.

## 10. Finals, VAL, specialisation audit

**Finals** (FINAL plan, milestones p040/p060/p080/p100, diag every 100): GC (frozen) s0, s1; SEM s0, s1; SHUF s0, s1.
Reused: T4 s0, s1 (s2), EDL s0, s1. **Third seed**: if sign(GC − T4) differs between s0 and s1, or any
meaningful-specialisation component (below) passes in one seed and fails in the other → GC s2 and SHUF s2 (T4 s2
reused).

**VAL** (once per new final checkpoint; `gc_evaluate.py final`, which imports fle's `evaluate`, `eval_rows`,
`audit_one`-components and per-expert logic unchanged): full-VAL CE (token-weighted) + per-document CE; audits at
p040 and p100 (discovery CS half A, measurement VAL audit subset 0.2M tokens/domain) for label systems domain, e5
cluster, pseudo-GC and pseudo-SEM (paper argmax q): topic/expert MI with load-preserving shuffled-routing null, topic
entropy per expert, topic prediction from selected experts and from gate mass, within-expert e5 similarity and
within-expert gradient-fingerprint coherence, FSI with difficulty matching and shuffled-label control (p100);
per-expert equal-compute counterfactual replacement (disable one expert, top-4 among the rest) at p100 for all 352
experts, with own-domain vs difficulty-matched other-domain interaction for domain, e5 and pseudo labels from the same
intervention passes; domain × expert interaction variance (two-way decomposition) and its relation to generic expert
importance (mean |ΔCE|). Supplementary evaluation of reused T4/EDL checkpoints adds the new metrics once (their domain
interaction is recomputed as a reproducibility check). Cross-seed Hungarian profile stability with null.

**Uncertainty**: per-seed differences are primary; paired per-document bootstrap CIs (2000) describe document sampling
only and are never treated as seed replication.

## 11. Success definitions and terminal mapping

- **Accuracy-comparable**: mean over seeds of paired (GC − T4) VAL CE ≤ +0.005 nats/token and no consistent severe
  domain regression (no domain with ΔCE > +0.015 in every seed).
- **Accuracy improvement**: mean < 0 and every seed's GC − T4 < 0.
- **Meaningful specialisation** (all, in every seed): domain/expert MI(GC) > MI(T4) and > MI(SHUF); per-expert domain
  interaction(GC) > T4 and > SHUF; domain FSI(GC) > T4 and > SHUF.
- **Strong target** (reported): ≥ 75 % of EDL's paired CE improvement over T4 and ≥ 75 % of EDL's gain in per-expert
  domain interaction and in domain FSI.
- Mapping: improvement + meaningful specialisation → POSITIVE; comparable (or improvement) + meaningful specialisation
  without improvement → ACCURACY-COMPARABLE SPECIALIZATION; otherwise NEGATIVE. Cluster/label correlation alone never
  counts as success.

## 12. Failure analysis (measured)

Links: (1) fingerprint stability across views; (2) nuisance share; (3) cluster coherence vs real domains (NMI after
unsealing); (4) EDL routing separation (training JSD, VAL pseudo MI); (5) functional specialisation (interventions);
(6) CE cost; (7) retention in the label-free inference model (training-batch separation vs held-out); (8) seed/run
noise. Reported with aux/LM gradient ratios, assignment stability, routing divergence, counterfactual usefulness and
correlations of each intermediate measure with final CE (across final and screening runs).

## 13. Resources and operator rules

Budgets: ≤ 30 GPU-h, ≤ 48 h wall clock, ≤ 40 GiB new persistent storage, ≥ 100 GiB free at all times (every job checks
free − planned peak ≥ 100 GiB). No `/dev/shm`. GPU only via `scripts/gpu_guarded.sh` through `gc_driver.py`.
Atomic checkpoints and driver state; one rolling resumable slot per active run plus milestones; slots of finished
screening runs are deleted after the milestone verifies (ledgered in `deleted_artifacts.tsv`). Trainers checkpoint at
least every 600 s and every 10 % of steps. Signals (SIGINT/SIGTERM/SIGHUP), the `artifacts/gc-edl-v1/STOP` marker or a
foreign GPU process → checkpoint at the next step, mark `stopped_by_operator`, release the GPU, exit 77; the driver
latches `external_stop` and never relaunches automatically (`gc_driver.py authorize-resume` only on explicit
instruction). Context compaction, tmux detachment or SSH disconnection are not stop signals. Resume verifies hashes and
continues from the exact checkpoint. No prior artifact is modified or deleted; the protected inventory is verified at
the end.
