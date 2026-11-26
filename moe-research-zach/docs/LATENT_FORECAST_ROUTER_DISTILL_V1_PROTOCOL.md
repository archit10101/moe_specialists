# csr-latent-forecast-router-distill-v1 (lfr): protocol

**When this was written.** 2026-11-11, before any data access, cache build, model run or result of this experiment.

**Change control.**
- Hashing comes before any result of this experiment is examined.
- Any later change is logged with its reason in `artifacts/latent-forecast-router-distill-v1/deviations.jsonl`. No
  scientific rule below is changed after a result is seen.
- Genuine implementation bugs are fixed, get a regression test, are logged, and the affected work is rerun.

**Paths.** Everything is new; no existing file is modified.
- Code: `lfr_*.py` and `src/csr/lfr_*.py`.
- Configs: `configs/latent-forecast-router-distill-v1/`.
- Artifacts: `artifacts/latent-forecast-router-distill-v1/`.
- Scratch, caches and rolling checkpoints: `.tmp/latent-forecast-router-distill-v1/`.
- Runs: `runs/<UTC>_latent_forecast_router_distill_v1_<name>`.

**Preserved and used read-only:**
- the stopped experiment `csr-forecast-specialist-pretraining-v2` (fspv2): its data, drafts, code, checkpoints, logs and
  STOP marker;
- every other earlier experiment.

`lfr_inventory.py` hashes all protected files before any work (`protected_inventory_before.json`). They are verified
again at the end.

## 0. Audit at preregistration (2026-11-11 19:26 EST, lambda2)

- **GPU.** GPU 0 (A100 80GB) shows 0 MiB in use and 0 compute processes. Two python processes of another user are
  running, but neither is on the GPU.
- **fspv2.** It is STOPPED_BY_OPERATOR, permanent pause:
  - the STOP marker is present and the `operator_stop` record exists;
  - none of its processes are running;
  - it is not resumed, touched or modified by this experiment.
- **Disk.**
  - 131 GiB is free on `/`. The project guard keeps a 100 GiB reserve, so this experiment can use about 31 GiB.
  - Everything below is sized to fit within 26 GiB (section 15.4).
  - If free disk minus planned writes would fall below the reserve, the driver stops with INSUFFICIENT_RESOURCES.

## 1. Question

**Can an ensemble of frozen causal draft LMs, used only during training, help a teacher router discover useful latent
expert specialisations, and can that teacher be distilled into an ordinary causal student router?**
- The teacher predicts the counterfactual expert utility of each token from its causal forecasts.
- The final model keeps no draft, no forecast cache and no domain label.
- It routes with unrestricted top-4 over all 32 experts plus the shared expert, at about TOKEN-4 inference cost.
- The final model should:
  - be more accurate than a strong EDL baseline;
  - keep measurable functional expert specialisation after all teacher help is removed.

No domain-to-expert mapping, expert mask, specialist slot or domain classifier exists anywhere in the proposed method.

## 2. Data (resolved from the latest specialisation experiment; used read-only)

**Source.** The fspv2 frozen five-domain data manifest:
- `artifacts/forecast-specialist-pretraining-v2/data/manifest.json`, splits_sha256 `efcde27ee890e9831b8bcd7393ddba66b3ed76ad345ed91e58a72fc10185fcbf`;
- built from M2D2 arXiv L1 (`machelreid/m2d2@6ae613ba744ee56f645aee9c577d04f7e3d48b30`);
- Pythia tokenizer (`EleutherAI/pythia-12b@bb1e3e71…`), V = 50304, EOS 0 and PAD 1;
- whole papers cut into ≤1024-token segments, with every split separated by paper hash;
- exact and near dedup (MinHash) across all five domains.

**Domains:**
- cs_l1 (computer science/code);
- math_l1;
- physics_l1;
- q-bio_l1 (biomedical);
- stat_l1.

**Per-domain quotas.** The fallback allocation rule applied, with A = 189,637,367:

| bucket | per domain (valid tokens) | role here |
|---|---|---|
| draft_shared | 25.0M | draft training only (already consumed by the fspv2 drafts) |
| draft_cont | 25.0M | domain-draft adaptation only (already consumed) |
| main | 122.5M | main training (all conditions), TOKEN-COMPUTE extension |
| calib_search | 2.5M | CALIB-SEARCH: stage 1 and 2 engineering, diagnostics, labeler fitting |
| calib_sel | 2.5M | CALIB-SELECTION: opened only at stage 3 |
| val | 5.0M | VALIDATION: opened exactly once per frozen final checkpoint |
| test | 5.0M | LOCKED: never opened (separate 0400 file; loaders refuse the name) |

- Evaluation buckets are filled from exclusive papers only.
- No paper hash occurs in two buckets; the fspv2 overlap audit is re-verified by `lfr_tests.py`.

**Balanced main stream (identical for every condition and seed).**
1. Each domain's main documents are ordered by the fspv2 fixed permutation (`order_seed = 2026`), restricted to that
   domain.
2. The streams are merged by greedy token balancing: the next document comes from the domain with the fewest tokens so
   far, ties broken by domain index.
3. Documents are packed into 1024-token rows (64 rows per optimizer step), in that order.

So every prefix of the stream is balanced across the five domains to within one document. Its hash is
`stream_sha256`, recorded in `artifacts/.../stream_manifest.json` before training. Seeds change only the
initialisation (and the seeded draws named below); the documents and their order never change.

**Budgets** (valid main tokens; steps are the smallest whole number of steps that reach the budget):

| scale | tokens | per domain | use |
|---|---|---|---|
| S1 | 49.0M | 9.8M | stage-1 candidates and references (seed 100) |
| S2 | 122.5M | 24.5M | stage-2 candidates and references (seeds 100 and 101) |
| FINAL | 245.0M | 49.0M | final conditions (seeds 0 and 1, plus 2 when triggered) |

- Every S1 and S2 plan is a prefix of the FINAL plan.
- TOKEN-COMPUTE continues along the same stream, unique tokens only, never more than the 612.5M available.
- The FINAL budget is 40% of the main bucket. That leaves unique tokens for the equal-compute baseline, and the GPU
  (section 15.3) and disk (section 15.4) budgets are planned around it.

## 3. Main-model backbone (identical in every condition; resolved verbatim from fspv2 `FSPConfig`)

| item | value |
|---|---|
| layers / d / heads | 12 / 384 / 6 (head dim 64), RoPE base 1e4, pre-RMSNorm (eps 1e-6), causal and document-masked attention |
| layer 0 | dense GELU FFN 1792 |
| layers 1–11 | 32 routed GELU experts (FFN 256, 2 matrices) + 1 always-active shared GELU expert (FFN 768) |
| routing interface | unrestricted top-4 over all 32 experts; p = softmax(logits), gates = p of the 4 selected (FLAME, no renormalisation); dropless sorted dispatch |
| embeddings | tied input/output, V = 50304, init std 0.02, output projections std 0.02/√24 |
| load balance / z-loss | Switch loss over 32 (coef 0.01) on the executed routing distribution; z-loss 0.001 on the executed logits |
| context | 1024 |

- The fspv2 topic/domain module (`topic_proj`, `dom_*`) belongs to the fspv2 specialist conditions only and is not
  part of this backbone.
- Backbone parameters are initialised first, in a fixed order, from `torch.manual_seed(seed)`. They are identical
  across conditions within a seed. Method-specific modules use separate generators (seeded with
  `seed + 1_000_003·k`) and never change the backbone draw. Verified by test T15.
- TOKEN-4 parameter counts:
  - total = fspv2 total − 295,680 topic-module parameters = 103,609,728;
  - active = 43,054,464 per token.
  - `lfr_tests.py` recomputes these and records the exact numbers.

**Training recipe (every main run):**
- AdamW (0.9, 0.95), weight decay 0.1 (none on 1-D parameters), lr 6e-4, grad-clip 1.0.
- WSD schedule: warmup 1% of steps, constant, then linear decay over the last 20% to 6e-5.
- Batch: 64 × 1024 rows per step (micro-batch 32 × 2 accumulation; halved on OOM with matching accumulation, which is
  mathematically identical).
- Numerics: bf16 autocast, fp32 router math, `torch.compile`.

## 4. Draft ensemble (six frozen causal drafts)

**Reuse decision.** The fspv2 drafts are reused read-only. Their hashes are verified before use, and the provenance
record goes to `draft_provenance.json`.

| draft | path | data |
|---|---|---|
| general (shared) | `runs/20261111T085851Z_fsp_draft_shared/ckpt_final.pt` | one epoch of the balanced `draft_shared` bucket (125M tokens) |
| domain d0–d4 | `runs/20261111T090752Z_fsp_draft_cont_d0`, `…T090924Z…_d1`, `…T091054Z…_d2`, `…T091226Z…_d3`, `…T091359Z…_d4` (`/ckpt_final.pt`; exact list in `artifacts/forecast-specialist-pretraining-v2/drafts.json`) | the shared draft, continued for one epoch on its own domain's `draft_cont` bucket (25M) only |

- **Architecture.** Dense causal LM: d 256, 10 layers, 4 heads, SwiGLU 768, tied V 50304, about 21.4M parameters.
- **Isolation.**
  - Draft training documents (`draft_shared` and `draft_cont`) are paper-disjoint from main, calib_search, calib_sel,
    val and test by manifest construction.
  - No externally pretrained model is used.
- **Cost.** The original training cost (432.4 + 5 × ~89 s = 877.1 GPU-s) is charged to the method in the equal-compute
  view.

**Diagnostics** (`lfr_draft_diag.py`, CALIB-SEARCH subset of 0.4M tokens per domain, before the cache is built):
1. Next-token CE of all 6 drafts on all 5 domains, and the own-domain vs wrong-domain advantage.
2. Top-1 and top-5 forecast accuracy at horizons 1, 4 and 8. Horizon h means the k = h step of the greedy causal
   rollout from prefix ≤ t, compared with x_{t+h}.
3. Calibration: 15-bin ECE of the top-1 probability at h = 1, plus the mean entropy.
4. Pairwise top-1 agreement between drafts at h = 1, 4 and 8.
5. CE and accuracy of the equal-weight ensemble vs the best single draft.
6. Comparison with the previous weak drafts.
   - Those are the dfcv1 drafts (`runs/20261110T074951Z_dfc_draft_d128_l4_adapt`), on the same subset.
   - Caveat: dfcv1 used a different data build, so overlap with these calibration papers cannot be excluded. The
     comparison is descriptive only.
7. The compression error of the cache representation (section 5) against the uncompressed draft outputs on the same
   subset.

**Draft gate (preregistered).** The drafts count as domain-useful only if both hold:
- (a) In ≥ 4 of 5 domains, the own-domain draft's CE is lower than every other domain draft's by ≥ 0.05 nats and
  lower than the shared draft's by ≥ 0.03.
- (b) The mean pairwise h = 1 top-1 disagreement among the six drafts is ≥ 5% of tokens.

If the gate fails, five new domain drafts are trained:
- d 384, 12 layers, initialised from a newly trained shared draft of the same size;
- on the same draft buckets only;
- capped at 2.0 GPU-h;
- the gate is then applied again.

If it fails a second time, the experiment continues, the drafts are recorded as failure class A, and that is reported.

## 5. Causal forecast cache

**Generation.** Each draft is loaded alone, processes a chunk of documents and is unloaded. At no time is more than one
draft resident on the GPU.
- Outputs are computed per document (one document per row from position 0), with the fspv2 causal tree-attention
  rollout (`draft_rollout_fast`).
- Position t sees only the real tokens ≤ t of its own document and its own earlier rollout tokens.
- Outputs per draft and token:
  - the full next-token distribution at h = 1;
  - greedy rollout tokens up to step 8;
  - score_d(t) = log p_d(x_t | x_<t) of the current real token, which is causal because x_t is the input at t.

**Stored representation** (44 bytes per token; one compressed file set per chunk):

| field | bytes |
|---|---|
| ensemble h1 top-4 ids (uint16) and probabilities (uint8, p·255) of the equal-weight mean of the 6 drafts' h1 distributions | 12 |
| per draft (×6): h1 entropy (uint8, H/lnV·255), h1 top-1 probability (uint8), position code (uint8: index of the draft's h1 argmax among the ensemble top-4, or 4 = other; 2-bit indices of its step-4 and step-8 rollout ids among the horizon candidate pairs, or 2 = other), score (uint8, min(−score, 25.5)·10) | 24 |
| horizon candidates: the two most frequent step-4 rollout ids across the 6 drafts and the two most frequent step-8 ids (uint16; ties broken by ensemble and shared-draft order) | 8 |

- The compression error is quantified against the uncompressed outputs on the diagnostic subset: coverage rates of the
  per-draft ids, the ensemble top-4 mass and the quantisation error.
- The ensemble top-4 is computed from each draft's exact top-16 (error bound reported).

**Coverage.**
- *Development region:* every document in the first ⌈0.6·n_steps(S2)⌉ steps of the stream. This covers every S1 and S2
  teacher and anneal phase.
- *Final region:* extended after the freeze to the first ⌈f_end·n_steps(FINAL)⌉ steps, where f_end is the frozen
  schedule's end of the anneal (0.5 or 0.6).
- *Diagnostics:* the CALIB-SEARCH diagnostic subset, 0.1M tokens per domain from half A.

**Constraints.**
- The cache never covers val, calib_sel or test.
- Size limit: 40 GiB as instructed; the planned size is ≤ 7 GiB.
- It is resumable through a chunk cursor, and each chunk is written atomically with a completion marker.
- GPU time and bytes are ledgered as `forecast_cache`.

**Fallback.** If the measured rollout throughput projects more than 4.0 GPU-h for the final region, rollouts are
computed only at causal anchors (every 2nd position).
- At an odd position, the features of the preceding even anchor are used, with horizons shifted by one.
- Anchor error is quantified against the dense features on the diagnostic subset.

**Features computed at load time (causal).** Built from the stored fields at positions ≤ t of the same document:
- **Per draft** (d = 0..5; 11 values):
  - h1 entropy;
  - h1 top-1 probability;
  - one-hots of the h1, h4 and h8 codes (agreement with the ensemble top choice and with the horizon majority);
  - score;
  - causal EMA (half-life 64) of the draft's score minus the mean score of all six;
  - causal EMA of the draft's past h1 hit indicator, `[argmax_d(t−1) = x_t]` from the code at t−1 and the ensemble ids
    at t−1.
- **Aggregate** (8 values):
  - ensemble top-1 probability;
  - mean h1 entropy;
  - agreement at h = 1, 4 and 8: the fraction of drafts that agree with the plurality choice;
  - disagreement: 1 − agreement at h = 1;
  - spread of the score-EMA (max − min);
  - log position.
- **Token embeddings.**
  - E1 = Σ_k p_k·emb(ens_id_k) over the ensemble h1 top-4.
  - E4 and E8: vote-weighted embeddings of the horizon candidate pairs.
  - All three use the main model's tied embedding, detached; the teacher never sends gradient into the embedding.
- Test T1 verifies that no feature at position t reads a real token after t, and that changing future tokens leaves
  features at ≤ t unchanged.

## 6. Proposed method (CF-FORECAST-STUDENT) and its parts

### 6.1 Aggregator

z_t = Agg(per-draft features, aggregate features, E1/E4/E8 restricted to the configured horizons). It has 6 draft-slot
embeddings (draft identity, never a domain label). Three variants:
- `pool`: calibrated opinion pool. Weights w_d = softmax_d(a·scoreEMA_d + b·h1conf_d + c_d) with learned scalars, and
  z = W_o[Σ_d w_d φ(f_d); aggregate; E].
- `deepsets`: z = ρ(Σ_d φ(f_d ⊕ slot_d) ⊕ aggregate ⊕ E), with a 2-layer MLP for φ and ρ.
- `xattn`: one layer of 2-head attention. A learned query (independent of h) attends over the 6 draft tokens
  φ(f_d) + slot_d, followed by ρ.

Width fc_dim ∈ {64, 128}. The aggregator is training-only (prefix `tc_`).

### 6.2 Teacher router (training only, `tc_`)

**Inputs.** For each MoE layer l, the teacher receives [RMSNorm(h_t^l), z_t + layer_emb_l, forecast entropy, agreement,
disagreement].
- A shared trunk (Linear → GELU, width 128) feeds a per-layer head with 64 outputs: predicted utility û(t,·) ∈ ℝ^32
  and log-variance ŝ(t,·) ∈ ℝ^32.

**Teacher logits.** ℓ_T = (û − κ·exp(ŝ/2)) / τ_T, where κ is the uncertainty penalty and τ_T the teacher temperature.
- Executed routing logits: ℓ_route = λ·ℓ_T + (1−λ)·ℓ_S.
- Then ordinary top-4 over all 32 experts, with gates = softmax(ℓ_route) of the selected experts. No expert is masked
  and no expert is forced.

**Injection.** `all` uses the teacher in layers 1–11. `late` uses it only in MoE layers 5–11 (model layers 6–11);
layers 1–4 then always use ℓ_S.

### 6.3 Student router (inference model)

ℓ_S = W_h·RMSNorm(h_t) + W_m·RMSNorm(m_t).
- m_t is a causal within-document memory of the layer's own main-model states: m_t = EMA_{s≤t, same doc}(P·h_s), with
  P: 384→64. The rate is either a fixed half-life (`ema`, 128 by default) or three concatenated half-lives 16/128/1024
  of 64/3 → 3×21 dimensions (`ema3`).
- It is computed as a chunked scan with exact normalisation: O(T·64·64) per layer at training, O(1) per token
  autoregressively.
- Extra inference parameters per MoE layer are 384×64 (P) + 64 (RMSNorm) + 64×32 (W_m, zero-initialised) = 26,688
  beyond TOKEN-4's 384×32 router. That is 293,568 in total (+0.28%) and +0.68% active. The exact numbers are recorded
  by `lfr_tests.py` T15.

### 6.4 Counterfactual expert utility (training targets)

**Sample.** A fixed, preregistered set of training positions: u(main document index, position) < r_cf, where u is
a salted splitmix64 uniform (`lfr_lib.cf_uniform`, salt 20261111) and r_cf ∈ {0.05, 0.075, 0.10}. It is used only in the teacher and anneal phases.

**Reference route.** At (t, l), the reference is the executed route R = (r1..r4) with gates g1..g4 (sorted).
Substituting candidate expert e into slot 4 gives y' = y + g4·(E_e(x) − E_r4(x)). The target is the change in summed
next-token CE of the segment:

    Delta(t,l,e) = L(ref) − L(substituted)

Positive values mean expert e lowers the loss. Delta(t,l,r4) = 0. For e ∈ {r1,r2,r3} it measures doubling that
expert's slot. The loss counts the next-token CE at t and the effect on later tokens of the same document through
attention, which is the quantity LM training optimises.

**Estimator (first order).**
1. The forward pass stores x_t^l at the sampled positions.
2. The main backward pass captures ∂L/∂y_t^l, through a compiler-disabled identity that records the gradient.
3. Under no_grad, all 32 experts are evaluated at the sampled positions.
4. The estimate is Delta^(t,l,e) = −g4·⟨∂L/∂y, E_e(x) − E_r4(x)⟩·n_valid.

**Exact audits.**
- *Pre-use validation* (section 9, stage 1):
  - On the S1 TOKEN-4 reference checkpoint, take 256 CALIB-SEARCH (t,l) pairs with all 32 substitutions each.
  - Each substitution is executed by actually running the perturbed forward, so downstream re-routing is included.
  - The exact forward runs in full fp32 (no autocast, no TF32). A pair counts as route-consistent only if the
    reference copy executes the recorded route at (t, l); the share of consistent pairs is reported.
  - Gate: mean Spearman ρ(first-order, exact) ≥ 0.5 and top-4 overlap ≥ 2.0 of 4. Rank correlation, top-4 overlap
    and calibration error (|slope − 1| and normalised MAE of a linear fit) are reported.
- *During every proposed run:* every 100 steps in the teacher phase, 2 random sampled (t,l) pairs are re-done exactly,
  with all 32 experts. That gives about 80 exact audits per final run, logged.

**If the gate fails.** The first-order estimator is not used.
- Each proposed run instead uses exact substitutions on one (t,l) pair per micro-batch row per step, with all 32
  experts.
- If that projects more than 2× the TOKEN-4 run cost, the sample is reduced until it fits.
- This is recorded as failure class C for the method and as a deviation from the 5–10% sample-rate requirement.

**Targets and losses** (at the sampled (t,l), with stop-gradient on all inputs):
1. Targets are standardised per (t,l) across the 32 experts: ũ = (Delta − mean)/(std + 1e-6).
2. The teacher head is re-applied to the detached stored inputs (h_t^l, cache features).
3. Its loss is added in a second, small backward before the same optimizer step. The route at t was computed before
   the target existed, so the target can never influence the route at t (test T10).
4. Teacher loss, by mode:
   - `rank`: listwise KL(softmax(ũ/0.5) ‖ softmax(û));
   - `reg`: Gaussian NLL ½[(ũ−û)²·e^{−ŝ} + ŝ];
   - `rank` also trains ŝ with the same NLL at weight 0.1, so the uncertainty is always calibrated.
5. Student utility loss: listwise KL(softmax(ũ/0.5) ‖ softmax(ℓ_S)).

### 6.5 Losses and schedule

**Gradient flow of the auxiliary routing losses** (identical in every condition):
- The Switch balance loss, the z-loss and EDL use the executed routing distribution with the standard gradient flow
  (into router weights and hidden states).
- The teacher→student KL and the latent-consistency loss are computed from router logits on stop-gradient hidden
  states, so they train router, student and teacher weights only.
- The captured utility gradient ∂L/∂y therefore contains the LM loss plus the small Switch, z-loss (and, in G, EDL)
  terms. The exact audits measure the LM loss alone and validate the estimator.

**Total loss:**

    LM CE + 0.01·balance + 0.001·z
      + c_util·L_util(teacher) + c_sutil·L_util(student)
      + c_kl·KL(p_T.detach() ‖ p_S) + c_lc·L_lc (optional, section 6.6)

KL is taken over all 32 experts and all tokens, in the teacher and anneal phases.

**Phases** (fractions of planned steps; schedule ∈ {40/20/40 (base), 30/20/50}):

| phase | span (base) | λ (teacher share of executed logits) | teacher losses / KL | drafts / cache |
|---|---|---|---|---|
| P1 teacher | 0–40% | 0.9 | on | read |
| P2 anneal | 40–60% | 0.9 → 0, linear | on, scaled by λ/0.9 | read |
| P3 student | 60–100% | 0 | off | absent from the forward (cache not opened, teacher not executed) |

**Execution in P3.**
- The teacher modules are deleted from the GPU model at the P3 boundary (`strip_teacher`), so P3 executes exactly the
  inference model.
- Optimizer state for teacher parameters is dropped.
- The P3 forward never receives a forecast argument (test T12).

**Forbidden in this method:**
- domain classification losses;
- hard masks;
- specialist slots;
- domain-to-expert assignment.

Domain labels never enter any proposed router input or loss. EDL in condition G uses labels only inside the EDL loss,
which is EDL's definition.

### 6.6 Optional soft latent-consistency loss (one candidate family)

**Pairs.** Within each micro-batch, all pairs (i, j) of CF-sampled positions in the same layer, capped at 4096 random
pairs.

**Similarities.**
- s_z = cos(z_i, z_j), with z detached.
- s_u = cos(ũ_i, ũ_j) of the measured standardised utilities.

**Pull and push.**
- Pull (i, j) with weight w = [s_z > 0.5]·[s_u > 0.3]·s_z·s_u.
- Push with weight w' = [s_z < 0]·[s_u < 0]·|s_z·s_u|.

**Loss.** L_lc = mean(w·JS(p_i, p_j)) − mean(w'·min(JS(p_i, p_j), 0.3)), on the executed routing distributions p, with
c_lc = 0.05.
- It is soft and bounded, and it never removes an expert from any token's eligible set.
- Its ablation is the same configuration without it.

## 7. Conditions

All conditions share:
- the backbone;
- initialisation within a seed;
- the stream, order, steps and tokens;
- the optimizer and schedule;
- top-4 plus shared expert;
- the evaluation code.

Every final evaluation is student-only, without drafts or the cache.

| id | name | router at inference | training-only help |
|---|---|---|---|
| A | TOKEN-4 | linear router over 32 | none |
| B | EDL | linear router | + β·L_EDL (Li et al., ICLR 2026, Eq. 5–8; fspv2 implementation: per-sequence mean router softmax → per-domain mean → mean over present domain pairs of −log(JSD+1e-8), averaged over the 11 layers). β ∈ {5e-4, 5e-3, 2e-2} tuned on CALIB-SEARCH at S1 (seed 100), outside the candidate budget |
| C | FORECAST-SIMILARITY | student | frozen E config, but the teacher is trained only by LM gradients through the gates: no CF targets, κ = 0, c_util = c_sutil = 0, c_lc = 0. KL distillation and schedule as E |
| D | OUTPUT-DISTILL | linear router (TOKEN-4 routing) | the same cache and phases. Loss + α·KD, where KD = −Σ_{k≤4} q_k log p_LM(ens_id_k) (ensemble h1 top-4 of the drafts), α in P1, α·λ/0.9 in P2, 0 in P3. α ∈ {0.3, 1.0} tuned on CALIB-SEARCH at S1, outside the candidate budget |
| E | CF-FORECAST-STUDENT | student | the full method (frozen configuration) |
| F | SHUFFLED-FORECAST | student | E, but every document's cached feature sequence is replaced position-aligned by that of another document of the same optimizer step: documents sorted by length, cyclic shift by one, positions beyond the donor length wrap modulo it. Marginal forecast statistics are preserved; the correspondence to the text is destroyed. CF targets are still measured on the real route |
| G | EDL + CF-FORECAST-STUDENT | student | E + β_EDL·L_EDL on the executed routing distribution (frozen β) |
| M | MEMORY-ONLY (extra control) | student | student architecture, no teacher, no forecasts. Isolates the effect of the router architecture. Knobs: the stage-1 incumbent at S2, the frozen configuration at FINAL |
| TC | TOKEN-COMPUTE | linear router | TOKEN-4 trained on more unique main-stream tokens to equal total compute (section 11) |

## 8. Bounded engineering search (frozen in `search_space.json`, hashed)

**Limit.** At most 12 unique proposed configurations, all on development data only.

**Base configuration P01:**
- horizons {1, 4, 8}, aggregator `deepsets`, fc_dim 64;
- memory `ema` with half-life 128;
- utility `rank`; c_util 1.0, c_sutil 0.3, c_kl 1.0;
- τ_T 1.0, κ 0, router-lr ×1, r_cf 0.075;
- schedule 40/20/40, injection `all`, no latent consistency.

**Initial wave (always run, in order):**

| id | change vs P01 |
|---|---|
| P01 | base |
| P02 | aggregator `pool` |
| P03 | aggregator `xattn` |
| P04 | utility `reg` + κ = 0.5 |
| P05 | + latent consistency (c_lc 0.05); P01 is its ablation |

**Diagnosis-driven candidates (P06 …, up to P12).**
- After every candidate, `lfr_diagnose.py` computes the failure flags (section 10) of every candidate.
- The *incumbent* is the eligible proposed candidate with the lowest CALIB-SEARCH CE so far. If none is eligible, it is
  the lowest-CE candidate.
- The next candidate is the incumbent's configuration plus one preregistered response, chosen as follows:
  1. The first failing class of the incumbent in priority order I, H, B, C, D, E, F, G selects that class's response
     list.
  2. The first response in that list whose resulting configuration has not run yet is used.
  3. If the list is exhausted, the next failing class is used.
  4. If the incumbent has no failing class, or every listed response has run, the refinement list is used.

**Response lists:**

| class | responses (in order; each is one factor applied to the incumbent) |
|---|---|
| I systems (non-bug) | r_cf 0.05 |
| H optimization | router-lr ×0.3; τ_T 2.0 (excessive sharpness) or τ_T 0.5 (excessive entropy), as diagnosed |
| B aggregation | fc_dim 128; horizons {1} |
| C utility target | r_cf 0.10; switch the utility mode (rank ↔ reg) |
| D teacher | τ_T 0.5; κ 0.5; c_util 3.0 |
| E distillation | c_kl 3.0; memory `ema3`; schedule 30/20/50 |
| F specialisation | router-lr ×3; injection `late` |
| G nonfunctional | latent consistency on; injection `late` |
| refinement | schedule 30/20/50; router-lr ×3; r_cf 0.10; memory `ema3`; horizons {1,4} |

Allowed values: horizons ⊂ {1, 4, 8}; aggregator ∈ {pool, deepsets, xattn}; fc_dim ∈ {64, 128}; memory ∈ {ema(32, 128,
512), ema3}; utility ∈ {rank, reg}; c_kl ∈ {1, 3}; c_util ∈ {1, 3}; τ_T ∈ {0.5, 1, 2}; router-lr ∈ {×0.3, ×1, ×3};
r_cf ∈ {0.05, 0.075, 0.10}; schedule ∈ {40/20/40, 30/20/50}; κ ∈ {0, 0.5}; injection ∈ {all, late}; latent
consistency on/off. Schedule 50/20/30 is excluded beforehand because it would need a 17% larger cache under the disk
limit (section 15.4).

**Stopping of stage 1.**
- It stops after 12 proposed candidates, or when its GPU allocation is used (4.5 GPU-h, including references), or when
  the response lists and the refinement list are exhausted.
- A bug fix rerun of the same configuration does not consume a candidate. It is logged, and the buggy run is kept and
  marked INVALID_BUG.

## 9. Engineering stages

**Stage 1: CALIB-SEARCH, S1 (49M), seed 100.**
- *References first:* TOKEN-4, EDL β grid (3 runs), OUTPUT-DISTILL α grid (2 runs).
- *Then the CF estimator validation and gate* (section 6.4).
- *Then the proposed candidates:* P01–P05, then the diagnosis-driven ones.
- *Run outputs:* every run writes in-process diagnostics on CALIB-SEARCH and is followed by a dev audit job (section
  12, development form). Each candidate gets a diagnostic report (`diagnostics/<run>.json` + a markdown summary).
- *Baseline selection:* EDL β* and OUTPUT-DISTILL α* are the lowest CALIB-SEARCH CE in their grids.

**Stage 2: extended CALIB-SEARCH, S2 (122.5M), seeds 100 and 101.**
- *Advancing to stage 2:* at most 4 proposed candidates, under a deterministic rule.
  - Eligible = no class-H failure, finite, **and** dev FSI_domain > the S1 TOKEN-4 reference's.
  - Rank eligible candidates by CALIB-SEARCH CE and take the top 4.
  - If fewer than 4 are eligible, fill with the lowest-CE ineligible ones, marked ineligible.
- *References at S2, both seeds:* TOKEN-4, EDL(β*), OUTPUT-DISTILL(α*) and MEMORY-ONLY (P01 student architecture).
- *Diagnostics and dev audits* as in stage 1.

**Stage 3: CALIB-SELECTION (opened only now).**
- Every stage-2 run is evaluated once on CALIB-SELECTION (student mode).
- Selection rule:
  - choose the S2-eligible candidate with the lowest mean CALIB-SELECTION CE over both seeds;
  - S2-eligible means no class-H failure in either seed and mean dev FSI_domain > the mean of the S2 TOKEN-4
    reference;
  - if none is eligible, choose the lowest-CE candidate and mark it INELIGIBLE (reported).
- The frozen configuration goes to `frozen.json` (hashed, mode 0400) with EDL β* and OUTPUT-DISTILL α*.
- No further method or hyperparameter change follows.

**Stage 4: final.**
- *Final-region cache build.* The final-region cache is extended first.
- *Run order*, at FINAL scale, seeds 0 and 1:
  1. A, B, E, D (both seeds, interleaved by seed);
  2. G, C, F;
  3. seed 2 of A, B, D and E, triggered automatically if E has lower VAL CE than B in both seed 0 and seed 1;
  4. TC (seeds 0 and 1);
  5. M (seeds 0 and 1).
- *Validation evaluation.* Each final checkpoint is evaluated on VAL exactly once, in one evaluation job per run that
  computes:
  - full-VAL CE (overall, per domain and per document);
  - routing statistics;
  - the specialisation audit of the 40/60/80/100% checkpoints.
- *After VAL is opened* nothing is tuned. Read-only diagnosis only.

**VAL and test access.**
- VAL is never read by training code or during stages 1–3. Test is never read.
- The training loader refuses `val`, `calib_sel` and `test`.
- The CALIB-SELECTION loader exists only in the stage-3 job.

## 10. Failure diagnosis (measurements; thresholds frozen)

Computed per development run. "Ref" means the same-stage, same-seed TOKEN-4 and EDL references. CE is measured on the
CALIB-SEARCH diagnostic set.

| class | flagged when |
|---|---|
| A draft | the draft gate (section 4) failed |
| B aggregation | teacher-mode CE with shuffled-document forecasts − teacher-mode CE with real forecasts < 0.002 nats at the end of P1 (in-process), OR the domain-probe accuracy from z_t < 0.8 × the probe accuracy from the raw per-draft features (5-way logistic probe, trained on half and scored on half of the diagnostic set) |
| C utility target | Spearman(û_teacher, Delta_exact) < 0.2 (mean over the exact audits at the end of P1), OR the in-run first-order-vs-exact Spearman < 0.3 |
| D teacher | not C, and teacher-mode CE (λ = 0.9) at the end of P1 ≥ TOKEN-4 CE at the same step − 0.002. The TOKEN-4 reference evaluates the same diagnostic set at the same step in-process (at 30% and 40%) |
| E distillation | not D, and (final student CE ≥ final TOKEN-4 CE, OR top-4 overlap of the pure teacher (λ = 1) and pure student routes at the end of P2 < 2.0 of 4 averaged over layers, OR final student CE − teacher-mode CE at the end of P1 > the TOKEN-4 CE change over the same interval + 0.01) |
| F specialisation | final domain–expert MI (mean over layers, CALIB-SEARCH half B) ≤ TOKEN-4's, OR the expert-profile correlation between the end of P1 and the final checkpoint < 0.5 |
| G nonfunctional | not F, and dev FSI_domain ≤ TOKEN-4's |
| H optimisation | any expert's load < 0.1/32 in any layer, OR max load > 4/32, OR mean normalised routing entropy > 0.98 or < 0.2, OR non-finite loss, OR mean teacher-head gradient norm < 1e-7 over P1 |
| I systems | step time > 2.5× TOKEN-4, OR a cache or feature read error, OR non-deterministic replay beyond the resume tolerance |

- **Success at development level:** CE ≤ EDL ref − 0.005, FSI_domain > TOKEN-4 ref, and no class flagged.
- **Recording.** Each candidate's report records every flag, the evidence values, the chosen response and the reason.

## 11. Comparisons, fairness and compute

**Equal main tokens.** All main conditions use the identical FINAL plan.

**Equal total compute.**

    TC GPU-seconds = mean measured GPU-s of the final E runs
                     + (draft training 877.1 + forecast-cache GPU-s, dev and final regions)
                       / (number of final E seeds)

- Counterfactual-evaluation time is inside the E runs.
- TC tokens = TC GPU-seconds × TOKEN-4's measured training tokens/s (FINAL A runs).
- Its stream is the same main stream, read further, unique tokens only. Its WSD schedule spans its own length.

**Also reported:** unamortised and per-10-trainings amortised views. Engineering costs (stage 1 and 2, audits) are
reported separately and are not charged to TC.

**Parameters.** Inference, active and training-only (`tc_`) parameter counts, plus the training FLOP overheads.

## 12. Specialisation audit (frozen before any specialisation result; `spec_audit_config.json`, hashed)

**Label systems** (never given to any router; evaluation only):
1. **Domain labels:** the five document domains.
2. **Latent clusters.**
   - The frozen encoder is `intfloat/e5-base-v2` (`artifacts/encoder/e5-base-v2`, model.safetensors sha256
     `d0d559c47d5f71b1d280b13b62a2657f3e3bc70c0786f9ab91a36545e6a8f693`). It is not used by the drafts, aggregator,
     router or any loss.
   - Inputs are 256-token windows (decoded text with the prefix "passage: ", mean pooling, L2 norm, max 512 e5
     tokens).
   - Clustering: k-means with K = 16, 20 inits, seeds 2026+i, k-means++, Lloyd (the `specaudit.kmeans`
     implementation).
   - It is fit on all CALIB-SEARCH half-A windows. Other windows are assigned to the nearest centroid. Every token
     takes its window's cluster.

**Sets:**
- *Discovery:* CALIB-SEARCH half A, 0.2M tokens per domain, split by paper hash.
- *Development measurement:* CALIB-SEARCH half B, 0.2M per domain.
- *Final measurement:* the VAL audit subset, the first 0.2M tokens per domain in file order.

**Routing metrics** (per layer and mean; top-4 selections):
- topic–expert mutual information;
- normalised topic entropy per expert;
- topic-prediction accuracy from selected experts (nearest-centroid classifier over 256-window histograms of (layer,
  expert) selections, trained on discovery and scored on measurement);
- within-expert semantic similarity: the selection-weighted mean cosine of window embeddings to the expert centroid,
  minus the same for the global centroid;
- layer profiles;
- expert load (min, max and the count < 0.1/32);
- the token routing entropy;
- cross-seed stability: per-layer Hungarian alignment of expert topic profiles, mean matched correlation;
- the load-preserving shuffled-routing null (expert sets permuted across tokens within a layer) for MI and stability.

**Functional specialisation (FSI).**
1. On discovery, each (layer, expert) gets topic k* = argmax_k lift(e,k), if lift > 1.
2. G_k = all (layer, expert) pairs assigned to topic k.
3. On measurement, for each k, every expert in G_k is disabled in its layer: its logit is set to −inf, top-4 is taken
   among the remaining experts, and gates are the masked softmax.
4. Record ΔCE per token.
5. Difficulty matching: baseline-CE decile strata. X_k reweights other-topic tokens to topic k's decile distribution.
6. Interaction I_k = O_k − X_k, where O_k is the mean ΔCE on topic-k tokens.
7. FSI = (1/K) Σ_k I_k / a_k, with a_k = max(1, |G_k|)/11 (per ablated expert per layer; topics with an empty G_k
   contribute 0).
   - FSI_raw = mean_k I_k is also reported.
8. Shuffled-label control: the same procedure with window labels permuted by a fixed permutation (seed 7), giving
   FSI_shuf.
9. Persistence: FSI at the 40/60/80/100% checkpoints.

**Interpretations (preregistered):**

| CE vs EDL | functional specialisation | interpretation |
|---|---|---|
| lower | yes | useful specialisation |
| lower | no | routing improved, but the specialisation hypothesis is unsupported |
| not lower | yes | specialisation occurred but was not useful for language modelling |
| not lower | no | the mechanism failed |

## 13. Primary success criteria (VAL, final seeds; all required for standalone success)

1. E has lower VAL CE than B (EDL) in every primary seed (0 and 1, plus 2 if run).
2. mean(E − B) ≤ −0.005 nats per token.
3. The paired per-document 95% interval excludes zero.
   - Method: document bootstrap, 2000 resamples, of the per-document CE difference averaged over seeds.
   - It is never treated as seed replication.
4. E beats A (TOKEN-4): lower mean, and lower in every primary seed.
5. E beats D (OUTPUT-DISTILL): lower mean, and lower in every primary seed.
6. FSI(E) > FSI(A) at 100%, in every seed, for both label systems.
7. mean FSI(E) ≥ mean FSI(B) at 100%, for both label systems.
8. FSI(E, 100%) > FSI(A, 100%) and FSI(E, 100%) ≥ 0.5·FSI(E, 40%) in every seed (domain labels). FSI(E) − FSI_shuf(E)
   > 0.
9. No draft, forecast or domain label is used at inference (tests T12 and T13; the export has no `tc_` parameters).
10. Student inference tokens/s ≥ 0.97 × TOKEN-4 and active parameters ≤ 1.03 × TOKEN-4.
    - Same harness, bf16, compiled, CALIB-SEARCH rows (throughput only), median of 5.
11. No collapse: every expert's load ≥ 0.1/32 and ≤ 4/32 in every layer at 100% (VAL), and the Switch loss ≤ 1.5 ×
    TOKEN-4's.

**Terminal status** (exactly one):
- **CONFIRMED STANDALONE IMPROVEMENT OVER EDL:** all of 1–11.
- **HYBRID-ONLY IMPROVEMENT:** E fails 1–3, but G satisfies 1–3 against B.
- **USEFUL SPECIALIZATION WITHOUT ACCURACY IMPROVEMENT:** 6–8 hold for E, but 1–2 do not, and the seeds agree.
- **ACCURACY IMPROVEMENT WITHOUT SPECIALIZATION:** 1–5 hold, but 6–8 do not.
- **UNRESOLVED:** the seed signs of E − B disagree.
- **NEGATIVE:** none of the above.
- **EXTERNALLY STOPPED.**
- **FAILED DUE TO IMPLEMENTATION OR RESOURCE ERROR.**

**Mechanism readings** (reported, not part of the status):

| observation | reading |
|---|---|
| D ≈ E (\|D − E\| < 0.003 mean) | ordinary knowledge distillation |
| F ≈ E | forecast semantics are not responsible |
| C ≈ E | utility supervision adds nothing beyond forecast similarity |
| M ≈ E | the router architecture, not the teacher |
| E specialises but does not improve | specialisation without usefulness |

A mean improvement that depends on one seed is never called confirmed.

## 14. Correctness tests (`lfr_tests.py` with CUDA hidden, plus `lfr_gpu_checks.py`; all must pass before stage 1)

- T1 causal forecast generation: changing tokens after t leaves the draft outputs and cache features at ≤ t unchanged.
- T2 the same for routes and logits of the teacher, student and executed routing at ≤ t.
- T3 document-disjoint splits: no paper hash in two buckets, and draft buckets disjoint from all others.
- T4 no VAL or test access by training code: the loaders refuse; static scan of the training modules.
- T5 unrestricted expert availability: every expert is reachable for every token in every phase, with no mask outside
  the audit ablations.
- T6 correct top-4 (exactly 4 distinct routed experts per token) and the shared expert always active.
- T7 exact counterfactual substitution: the perturbed forward equals manual recomputation, and the slot-4 identity
  gives Delta = 0.
- T8 first-order vs exact agreement on a small CPU model (finite-difference limit).
- T9 no same-position target leakage: changing x_{t+1} leaves the route at t unchanged in the same forward. The utility
  target is applied only after the route has been computed.
- T10 delayed / stop-gradient construction: the utility loss sends no gradient into the backbone states.
- T11 P3 student execution with draft code absent: the module imports no draft or cache code, and runs with the
  `tc_` parameters deleted.
- T12 inference export: no `tc_` parameters, and the forward accepts no forecast argument.
- T13 no domain label in the proposed router inputs (signature inspection plus a label-perturbation invariance test).
- T14 checkpoint/resume equivalence: the GPU resume check (section 15) within 3× the run-to-run noise.
- T15 matched initialisation (identical backbone tensors across conditions within a seed) and identical data order and
  rows.
- T16 matched tokens, steps and active experts across conditions.
- T17 EDL correctness: a hand-computed small case, zero with fewer than 2 domains.
- T18 deterministic candidate selection and diagnosis: the same inputs give the same next candidate; the advancement
  rule and the selection rule are tested on synthetic ledgers.
- T19 specialisation audit correctness:
  - a synthetic router with planted topic experts gives FSI > 0 and correct MI;
  - a random router gives FSI ≈ 0;
  - the shuffled label gives ≈ 0.
- T20 load-preserving shuffled baselines preserve per-expert loads exactly.
- T21 SHUFFLED-FORECAST mapping is a derangement within the step and preserves the multiset of features.
- T22 chunked EMA memory equals the dense causal EMA and resets at document boundaries.
- T23 the CF sample is deterministic with the configured rate, and the phase schedule is exact.
- **Smoke runs.** Before the resume check, every condition (A, B, C, D, E with latent consistency, F, G, M) trains
  40 compiled steps at production shape with in-process diagnostics, through all phases. Engineering only; their
  values are never selection inputs.

The full CPU suite runs again before the final stage (stage 4).

## 15. Checkpointing, external interruption, resources

### 15.1 Checkpoints

**Content.** Each one holds:
- model weights, including the router, teacher and aggregator;
- the AdamW state;
- the scheduler (a stateless WSD of the step: step and next lr recorded);
- the scaler (null: bf16, no GradScaler);
- RNG states (CPU, CUDA, numpy, python);
- the cursor: next step, plan hash and stream hash;
- step and token counts;
- the scientific phase and λ;
- the candidate id and a driver-state snapshot hash;
- the cache manifest hash;
- the config sha256;
- the sha256 of `state.pt`, inside `meta.json`, with a `complete: true` marker.

**Atomic write.**
1. Write a temporary directory `.<name>.tmp/` in the destination.
2. Flush and fsync each file.
3. Reload and verify (step, tokens, run id, number of tensors).
4. Rename atomically, then fsync the parent.
5. Record it in `resume_index.json` (atomic) and in `checkpoint_ledger.tsv`.

**Rolling checkpoints.**
- *When:* every max(1, ⌊0.10·n_steps⌋) steps or 900 s, whichever comes first, plus every phase boundary, every
  scientific milestone and every stop.
- *Retention:* the latest two are kept; older ones are deleted only after a newer one is verified. They are deleted
  after the run is COMPLETE, which is ledgered.

**Milestone checkpoints** are model-only. Matmul weights are stored in bf16; 1-D and router tensors stay fp32. They are
never deleted.
- Final runs: 40/60/80/100%.
- S2 and S1 runs: 100%.

**Other jobs.**
- The cache generator keeps a chunk cursor with atomic chunk files.
- The evaluation and audit jobs are idempotent per checkpoint, with results written atomically.
- Development-run "before long evaluation" coverage: the final-step rolling checkpoint is written before the
  end-of-run evaluation.

### 15.2 External interruption: yield, never fight

**Before each GPU job.** The driver checks the GPU processes (NVML). If any foreign process is present, it launches
nothing, records WAITING_FOR_GPU / EXTERNALLY_STOPPED and exits.

**During a job.** An OccupancyMonitor checks every 15 s.

**Signals.** SIGINT, SIGTERM, SIGHUP, the STOP marker `artifacts/latent-forecast-router-distill-v1/STOP`, or a foreign
GPU process mid-run lead to:
1. the child stops at the next optimizer-step boundary with a verified checkpoint;
2. the child exits with rc 77 and is marked EXTERNALLY_STOPPED / STOPPED_BY_OPERATOR;
3. the driver forwards SIGINT only to its own child, waits for its session to exit, confirms GPU release, records
   the resume command and exits.

**Never:**
- polling;
- automatic restart;
- killing or touching a foreign process.

**Restart.** It requires `--authorized-resume`, which is used only on an explicit user instruction.

**SIGKILL** cannot be handled. The next authorised resume uses the newest verified checkpoint, at most about 15 minutes
or 10% back.

There is no supervisor.

### 15.3 GPU budget (engineering budget)

**Total: 40 GPU-hours**, counted from this protocol's hash. It includes all tests, labeling, diagnostics, cache,
search, finals, audits and benchmarks.

| item | allocation |
|---|---|
| cache | ≤ 4.0 h |
| stage 1 | ≤ 4.5 h |
| stage 2 | ≤ 7.5 h |
| finals and evaluations | the rest |

**Before each launch** the driver checks used + projected ≤ budget. If the budget cannot cover everything, the
priority is:
1. A, B, E, D (seeds 0 and 1);
2. G;
3. seed 2 (if triggered);
4. C, F;
5. TC;
6. M.

Final tokens are never reduced for a single condition.

### 15.4 Disk (planned ≤ 26 GiB new)

| item | size |
|---|---|
| cache | ≤ 6.5 GB (final region at 60%: 147M tokens × 44 B), + 0.1 GB diagnostic subset |
| milestones | about 0.21 GB each. Finals: A, B, E, G at 4 milestones; C, D, F, M at 40/100%; TC at 100%; seed-2 A, B, D, E. S1 and S2: 100% only. ≈ 16 GB worst case |
| rolling checkpoints | ≤ 3 GB at any time |
| logs, evaluation and audit outputs | ≤ 1 GB |

The driver checks free − planned ≥ 100 GiB (guard reserve) before each job.

## 16. Integrity and outputs

**Before training:**
- the protocol hash;
- `search_space.json` and `spec_audit_config.json` hashes;
- `stream_manifest.json`;
- `draft_provenance.json`;
- the protected inventory;
- the CPU suite with CUDA hidden;
- the GPU checks;
- the resume check.

**Ledgers** (all in `artifacts/latent-forecast-router-distill-v1/`):
- `compute_ledger.tsv`, `run_manifest.tsv`, `candidate_table.tsv`, `checkpoint_ledger.tsv`, `deviations.jsonl`,
  `driver_log.txt`;
- plus the shared project ledgers.

**Final outputs:**
- `LATENT_FORECAST_ROUTER_DISTILL_V1_REPORT.md`;
- `artifacts/latent-forecast-router-distill-v1/`:
  - `final_analysis.json`, `run_manifest.tsv`, `compute_ledger.tsv`, `candidate_table.tsv`;
  - `specialization_trajectory.json`, `counterfactual_utility_analysis.json`, `deviations.jsonl`.

The report answers the 15 required questions and lists every candidate, seed, failure and deviation.

**If the result is negative:**
- read-only diagnosis of the saved artifacts;
- a separately preregistered v2 proposal (`docs/LATENT_FORECAST_ROUTER_DISTILL_V2_PROPOSAL.md`);
- v2 is never run against this VAL split without explicit user authorisation and a new untouched evaluation plan.
