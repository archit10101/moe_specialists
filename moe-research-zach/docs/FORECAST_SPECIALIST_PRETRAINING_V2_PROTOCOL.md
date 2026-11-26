# csr-forecast-specialist-pretraining-v2 (fspv2) — protocol

Written 2026-11-11, before any data build, draft training or candidate run of this experiment. The only GPU work
done before hashing was three engineering throughput probes of the backbone on random tokens (`fsp_quickbench.py`,
`fsp_quickprof.py`; no data, no candidate metrics), which motivated `torch.compile` and the split/unbind expert
dispatch. Any change after hashing is logged in `artifacts/forecast-specialist-pretraining-v2/deviations.jsonl` with
the reason; nothing below is changed after any result is observed except through that log.

Code prefix: `fsp_*.py`, `src/csr/fsp_*.py`. Artifacts: `artifacts/forecast-specialist-pretraining-v2/`. Configs:
`configs/forecast-specialist-pretraining-v2/`. Runs: `runs/<stamp>_fsp_<name>`. No existing file is modified.

## 1. Question

Can genuinely domain-specialised 18–25M causal draft LMs, used only as a training-time forecast teacher, teach a
~104M-total / ~43M-active MoE with a mandatory `3 general + 1 specialist` slot to develop persistent, functional
subject specialists, while validation CE is comparable to (≤ +0.005 nats) or better than plain TOKEN-4?

## 2. Main model (all conditions; inference parameter set identical in every condition)

| item | value |
|---|---|
| layers / d / heads | 12 / 384 / 6 (head dim 64), RoPE base 1e4, pre-RMSNorm |
| layer 0 | dense GELU FFN 1792 (= 768 shared + 4×256 routed: same active FFN width as an MoE layer) |
| layers 1–11 | 32 routed GELU experts (FFN 256) + one always-active shared GELU expert (FFN 768) |
| vocab / embeddings | Pythia tokenizer, V = 50304, tied input/output embedding |
| topic/domain module (per MoE layer) | 384→64 projection, fixed-rate causal EMA inside the document, RMSNorm, domain head 5×64 + token head 5×384 |
| total / active | 103,905,408 total; 43,350,144 active per token (24.0M non-embedding active) |

The 2-matrix GELU expert (instead of the project's SwiGLU) is the minimal adjustment that hits the requested
100–115M total / 38–45M active with the literal FFN sizes 256 / 768 at V = 50304. The topic/domain module
(295,680 parameters) exists in every condition. TOKEN-4 and EDL never use it, so it gets no gradient there and
the parameter count stays exactly equal. The forecast teacher (prefix `tc_`) exists only during training and is
deleted for inference. It is reported separately as training-only parameters.

Routing:
- **TOKEN-4** (A, B, C, H at inference): p = softmax(Wh) over 32; top-4 by p; gate = p (FLAME, no renormalisation).
- **SPEC** (D, E, F, G): experts 0–21 are general and 22–31 are specialists; domain d owns {22+2d, 23+2d}. Every
  token in every MoE layer executes the top-3 general experts (by p over the 22), the shared expert, and exactly one
  specialist, chosen by argmax over the 10 specialists of `l'_e + beta·log q(dom(e))`. Here l' are the
  (teacher-modified) router logits and q is the causal domain posterior. Teacher-forced tokens are restricted to
  the label's two specialists. Gates are the joint softmax of l' over all 32 experts, the same gate formula as
  TOKEN-4; only the eligible sets differ. There is no fallback to a 4th general expert in any phase, eval or
  inference.
- Load balance: TOKEN-4 uses the Switch loss over 32. SPEC uses the Switch loss over the 22 general experts plus a
  token-share-weighted within-group two-expert balance loss. Both use coefficient 0.01; z-loss coefficient is 0.001.

## 3. Data (M2D2 arXiv L1, `machelreid/m2d2@6ae613ba744ee56f645aee9c577d04f7e3d48b30`)

The five domains are cs_l1 (computer science/code), math_l1, physics_l1, q-bio_l1 (biomedical) and stat_l1. Papers
are member texts split on `\n{2,}` (train.txt + valid.txt). M2D2's own test.txt members are hashed and never
decoded. The data is built by `fsp_data.py`.

1. **Exact dedup.** Use sha256 of the normalised (lower-cased, whitespace-collapsed) full text across the five
   domains. A paper listed in several domains is kept exactly once and assigned to the listed domain with the
   fewest exclusive papers (scarcity order computed at build time from the data). It is flagged
   `exclusive=false`. Within-domain repeats keep one copy. Rationale: v1 dropped every cross-listed paper, which
   left q-bio with only 119.6M tokens. That is below this protocol's hard minimum of 165M.
2. **Near dedup.** Use MinHash over the first 800 normalised words, with 5-word shingles, 128 permutations, 32×4
   LSH and estimated Jaccard ≥ 0.8. This runs over the union of all five pools (all splits). Each cluster keeps
   one paper: the member in the scarcest domain, ties broken by hash-u. Kept papers from clusters spanning more
   than one domain are flagged non-exclusive.
3. **Segmentation.** The whole paper is tokenised, then EOS is appended. It is cut into consecutive ≤1024-token
   segments, and segments under 64 tokens are dropped. All segments of a paper go to one split. Splits are by
   paper hash.
4. **Buckets per domain**, all valid-token quotas:
   - Evaluation buckets are filled first, from **exclusive papers only**, in salted hash order: `test` (LOCKED),
     `val`, `calib_sel` (CALIB-SELECTION) and `calib_search` (CALIB-SEARCH).
   - Training buckets are then filled from all remaining papers, in hash order: `draft_shared`, `draft_cont` and
     `main`.
   - The last paper of a bucket is cut at the quota.
5. **Allocation rule.** Let A = min over domains of available valid tokens after dedup.
   - Target per domain: shared 50M, cont 50M, main 200M, calib 10M (5 search + 5 selection), val 10M, test 10M
     (330M total).
   - If A ≥ 330M, use the target.
   - Otherwise start from the minimum: 25, 25, 100, 5 (calib: 2.5 + 2.5), 5, 5 (165M). Add the surplus
     A − 165M − 2M (granularity margin) in this order:
     1. main, up to 200M;
     2. calib, val and test equally, up to 10M each;
     3. shared and cont equally, up to 50M each.
   - Every quota is rounded down to 0.5M. The same quotas apply to every domain.
   - If A < 167M the experiment stops (data insufficiency is reported).
   - If the exclusive papers of a domain cannot fill the evaluation buckets, the remainder comes from
     non-exclusive papers. This is logged.
6. **Order and identity.** The main stream is balanced by construction (equal quota per domain). Packing order is
   one fixed permutation (`order_seed = 2026`), used by every condition and seed; seeds change only the
   initialisation and the teacher-forcing draws. Main tokens are a single epoch with no repeats. The test bucket
   is written to a separate 0400 file and is never read by any fspv2 loader. The loader refuses the name `test`.
7. **Manifests.** These record per-split document IDs (paper hash + segment index) and the sha256 of every file.
   An overlap audit asserts that no paper hash occurs in two buckets.

## 4. Drafts (additional resource; cost reported separately)

- **Architecture.** A dense causal LM (project `CSRModel`): d = 256, 10 layers, 4 heads, SwiGLU FFN 768, tied
  V = 50304. That is ≈21.4M parameters, ≈8.5M of them non-embedding.
- **Shared draft.** One epoch on the balanced `draft_shared` corpus. AdamW lr 1e-3, WSD (warmup 1%, decay over the
  last 20% to 0.1×), 64×1024 tokens per step.
- **Domain drafts.** Five complete copies of the shared draft (all weights trainable), each continued for one epoch
  on its own domain's `draft_cont` corpus, at lr 5e-4 with the same schedule shape.
- **Forecasts.** For every causal prefix x≤t, the domain-d draft produces greedy rollouts r_t[1..8]. The draft's
  attention sees only real tokens ≤ t plus its own earlier rollout tokens, using parallel tree attention.
- **Cached per token** for two drafts per document:
  - the **matching** draft (the true domain);
  - the **wrong** draft w(doc), a fixed incorrect domain drawn uniformly by hash from the other four;
  - for each of the two: rollout ids (8), step entropies (8, uint8) and step-1 top-4 probabilities (uint8);
  - for all five drafts: log p_d(x_t | x<t) (fp16) of the actual current token, which is causal.

  The cache covers the documents in the first 50% of the fixed main stream, which is the only part where
  forecasts are used, plus `calib_search` for diagnostics. Its GPU time and disk use are ledgered as
  `forecast_cache`.
- **Router scalars (21).** These are computed causally at load time from the cache for the *assigned* draft:
  - step entropies (8) and top-4 probabilities (4);
  - cross-horizon agreement of r_{t−1}[2..8] with r_t[1..7], and of r_{t−4}[5..8] with r_t[1..4] (2);
  - an EMA (half-life 64) of past hit rates 1[r_{t−h}[h] = x_t] for h = 1, 4, 8 (3);
  - an EMA of the assigned-draft score minus the mean, and minus the max, of the other drafts' scores (2);
  - document-position features (2).

  No feature at position t reads a real token after t.
- **Diagnostics on CALIB-SEARCH** (not stop gates):
  - a 6×5 table of CE and accuracy for every draft (shared + 5) on every domain;
  - rollout accuracy at horizons 1, 4 and 8, where position k of a rollout is correct iff it equals the real
    token;
  - the matching vs non-matching advantage;
  - a 5-way domain classifier from forecast features: a linear probe trained on half of CALIB-SEARCH and scored on
    the other half;
  - the token-agreement similarity between correct-domain and deliberately wrong-domain rollouts.

## 5. Proposed method and controls

- **Forecast code.** c = Enc(mean of detached tied embeddings of r[1..H] + learned horizon embeddings, together
  with the 21 scalars) → 64-dimensional code.
- **Layer residuals.** Each MoE layer l adds a layer-identity vector and has two teacher heads: A_l c (10
  specialist-logit residuals) and G_l c (5 domain-logit residuals).
- **Teacher.**
  - Routing logits: l' = l + σ·A_l c on the specialist logits.
  - Domain posterior: q_t = softmax(student domain logits + σ·G_l c).
  - The specialist gate (joint softmax of l') makes A_l receive LM gradient.
- **Student.** l (router on the main token state) plus q_s = softmax(D_l·norm(EMA_topic(P_l h)) + E_l h). It uses
  only the main model's causal prefix state.
- **Losses**, added to LM CE + 0.01·balance + 0.001·z:
  - dom_coef × [CE(q_s, label) + CE(q_t, label) + KL(q_t ‖ q_s)];
  - distill_coef × within-group KL(teacher ‖ student), with the teacher detached.
- **Curriculum** (by fraction of planned main steps):

  | phase | span | σ | tf | dom_coef |
  |---|---|---|---|---|
  | P1 | 0–40% | 1 | 1 (every token's group is the label's group) | dom12 |
  | P2 | 40–50% | 1 → 0 | 1 → 0 | dom12 |
  | P3 | 50–100% | 0 | 0 | dom3 (small, fixed) |

  - **P2.** Both σ and tf fall linearly, or by a cosine shape for candidate C11. The forcing draw is a per-token
    Bernoulli from a seeded generator. Unforced tokens use the teacher score with σ.
  - **P3.** No forecast and no label is a routing input. Routing is the student only, and the label is used only
    as the domain-CE target. The 3+1 structure is kept.
- **Inference.** The teacher is deleted, drafts are never loaded, and labels are never passed. This is enforced by
  tests and by the inference export.

Conditions (identical init within a seed for shared backbone parameters, data order, optimizer, schedule, batch
construction, main tokens and executed experts per token):

| id | name | routing | domain aux | forecast teacher | notes |
|---|---|---|---|---|---|
| A | TOKEN-4 | token4 | – | – | primary baseline |
| B | DOMAIN-AUX | token4 | student head, dom12→dom3 schedule of the frozen D | – | domain supervision without partition |
| C | EDL | token4 | – | – | + β·L_ED (Li et al., ICLR 2026, Eq. 5–8); see below |
| D | FORECAST-SPECIALIST | spec | yes | matching draft | proposed |
| E | LABEL-SPECIALIST-ORACLE | spec, group = label always (also at eval) | yes | – | oracle, non-deployable |
| F | SHUFFLED | spec | yes, target = s(doc) | draft s(doc) | s(doc) = true w.p. 1/5 else w(doc), so uniform and independent of the true domain; s(doc) also forces groups |
| G | WRONG-DRAFT | spec | yes | draft w(doc) | true labels; wrong forecasts |
| H | FORECAST-NOPART | token4 | yes | 32-logit residual + 32-way KL distill | forecasts without partition |
| TC | TOKEN-COMPUTE | token4 | – | – | equal total compute (section 7) |

**EDL** follows Li et al. (ICLR 2026) Eq. 5–8:
- the full-softmax router distribution, averaged per sequence (packed segment);
- sequence means averaged per domain within the step's micro-batch;
- the mean over present domain pairs of −log(JSD + 1e-8);
- zero if fewer than 2 domains are present.

The paper leaves per-layer aggregation unspecified; here the per-layer losses are averaged over the 11 MoE layers.
The coefficient β ∈ {5e-4 (paper), 5e-3} is chosen by CALIB-SEARCH CE at 50M tokens on development seed 100. This
is EDL's only tuning, and it is outside the 12-candidate budget.

## 6. Training recipe (every main run)

- Optimizer: AdamW (0.9, 0.95), weight decay 0.1 (none on 1-D parameters), lr 6e-4, grad clip 1.0.
- Schedule: WSD with warmup 1% of steps, decaying linearly over the last 20% to 6e-5.
- Batch: 64 rows × 1024 tokens per step (micro-batch 32, 2-step accumulation; halved on OOM with matching
  accumulation).
- Numerics: bf16 autocast, fp32 router/domain math, `torch.compile`.
- Budget: the step count is the smallest number of whole steps whose valid tokens reach the budget.
- Checkpoints: at 40/50/75/100% of steps.
- Chunking: runs are split into guarded ≤85-minute legs with exact resume (model, optimizer, RNG and data cursor).
  Resume equivalence is tested before training.

## 7. Comparisons and compute

1. **Equal main tokens.** Every main condition uses exactly the same main tokens and steps.
2. **Equal total compute.**
   - TOKEN-COMPUTE GPU-seconds target = mean measured GPU-seconds of the final D main runs + (draft training +
     forecast cache GPU-seconds) / (number of final D seeds).
   - Its token budget is that target × TOKEN-4's measured training tok/s.
   - Its stream is the main stream, then the balanced draft corpora (interleaved by domain), then, if still short, a
     second pass over the main stream.
   - Processed and unique tokens are both reported. Its WSD schedule spans its own length.
   - It is trained for seeds 0 and 1 after the D finals.
3. **Draft cost.** Reported unamortized (one main model), amortized over the actual number of final D seeds, and
   projected over 10 downstream trainings.

## 8. Bounded search (D only; ≤ 12 unique configurations; development seeds 100, 101)

Base configuration C01:
- forecast horizon H = 8;
- topic half-life 128 tokens;
- spec_beta 1.0;
- dom12 = 0.1, dom3 = 0.02;
- distill_coef 1.0;
- linear anneal;
- router lr ×1;
- teacher residual init ≈ 0.

The candidates are one-factor variations of C01:

| id | change vs C01 |
|---|---|
| C01 | base |
| C02 | H = 4 |
| C03 | H = 1 |
| C04 | topic half-life 32 |
| C05 | topic half-life 512 |
| C06 | spec_beta 3.0 |
| C07 | spec_beta 0.3 |
| C08 | dom12 0.3, dom3 0.05 |
| C09 | dom12 0.03, dom3 0.005 |
| C10 | distill_coef 0.1 |
| C11 | cosine anneal shape (σ = tf = ½(1+cos(π·u)) over P2) |
| C12 | router, topic and domain-head lr ×3 |

**Stage 1.** All 12 candidates run at 50M main tokens (first 50M of the fixed stream; curriculum 40/10/50 of
50M) on seed 100. Reference runs at the same scale: TOKEN-4 (seed 100), and EDL β grid (seed 100).

Eligibility (specialisation must not be traded for CE). On CALIB-SEARCH in standalone mode (no labels, no
forecasts), the run must have:
- (a) specialist-group selection accuracy vs the domain label ≥ 0.5, averaged over layers;
- (b) every specialist chosen for ≥ 2% of tokens (pooled over layers);
- (c) no NaN or divergence.

Rank by CALIB-SEARCH CE, eligible runs first.

**Stage 2.** The top 4 run at 150M tokens on seeds 100 and 101. TOKEN-4 runs at 150M on seeds 100 and 101 for
reference.

**Freeze.** Choose the eligible candidate with the lowest mean CALIB-SELECTION CE over both seeds. If none is
eligible, choose the lowest-CE one and mark it ineligible. The frozen configuration is written to `frozen.json`
(hashed) before any final run. VAL is never read during search.

**Dev-scale controls.** After the freeze, run E, F, G, H and B with the frozen configuration at 150M on seeds 100
and 101. These are mechanistic evidence and are not selection inputs.

Every attempt, including failures and stopping reasons, is appended to `search_ledger.tsv`. Nothing is deleted.

## 9. Final runs and decision rules

- **Order.** Final main tokens = 5 × main quota (one epoch), seeds 0 and 1, run in this order:
  1. A, C, B, D (both seeds);
  2. TOKEN-COMPUTE (both seeds);
  3. promoted controls in the order E, F, G, H while the budget allows.
- **Primary metric.** VAL CE (nats per valid token), token-weighted over all five domains. Per-domain CE is also
  reported.
- **Provisionally better.** D beats A and C in both seeds. If so, seed 2 of A, C, B and D is trained automatically
  before any positive conclusion.
- **Comparable.** D − A ≤ +0.005 in each seed.
- **Better.** Mean(D − A) ≤ −0.005, with the favourable sign in every completed primary seed.
- **Unresolved.** The seed signs disagree.
- **Bootstrap.** Document-bootstrap intervals are reported but never replace seed variation.
- **Success** requires (i) persistent and functional specialisation (section 10) **and** (ii) comparable-or-better
  CE vs TOKEN-4. Otherwise the result is reported as negative or unresolved, exactly as observed.

## 10. Specialisation (D, plus all spec conditions; TOKEN-4/EDL/DOMAIN-AUX for comparison)

**When and where.** Measured at 40/50/75/100% checkpoints on a VAL subset (≤1M tokens per domain), in standalone
mode.

**Routing metrics:**
- domain–expert mutual information per layer;
- the domain entropy of every specialist;
- routing mass on the intended domain;
- specialist-group accuracy;
- predicted-domain accuracy by document position (buckets 0–63, 64–255, 256–1023 within segment, plus segment
  index within paper);
- specialist load and unused experts;
- cross-seed role stability: the fraction of (layer, specialist) pairs whose argmax domain agrees across seeds, and
  the correlation of domain profiles.

**Functional ablation.** For each domain group g and each eval domain d, on matched token counts:
- exclude g's experts in every MoE layer;
- re-route among the remaining experts (the specialist slot re-chosen among the other 8 specialists, gates from the
  masked-renormalised softmax);
- record ΔCE(d | −g).

Own-domain harm ΔCE(g | −g) is compared with the mean other-domain harm. Matched swaps are also run: force the
group of domain d' on domain-d tokens. TOKEN-4, EDL and DOMAIN-AUX get the same ablation on post-hoc groups: per
layer, the 2 experts with the highest domain lift on CALIB-SEARCH, assigned greedily without overlap.

**Persistent specialisation** (all required):
1. 100% checkpoint specialist accuracy ≥ 0.5 in standalone mode;
2. own-domain ΔCE > other-domain ΔCE for ≥ 4 of 5 groups in each seed;
3. cross-seed role agreement ≥ 0.8 of (layer, specialist) pairs;
4. no specialist below 2% load in any layer.

## 11. Integrity

- **Before training:**
  - protocol hash;
  - frozen data manifests;
  - protected-file inventory (verified at the end);
  - CPU suite with CUDA hidden;
  - GPU correctness and causality checks.

  The tests cover:
  - causal forecast construction (perturbing real tokens after t changes no forecast, feature, route or logit at
    ≤ t);
  - specialist masking;
  - exactly 3 general + 1 specialist per token;
  - annealing schedule values;
  - resume equivalence (tolerance documented);
  - final draft/label removal (the inference export has no `tc_` parameters, and the forward needs no `fc` or
    `domain`).
- **Ledgers:** safety, process and gpu_time ledgers, plus fspv2 `compute_ledger.tsv`, `search_ledger.tsv`,
  `run_manifest.tsv` and `deviations.jsonl`.
- **Budget:** 48 GPU-hours total, including probes. Before each launch the driver checks the cumulative GPU-h plus
  the projected cost, ≥ 80 GiB free disk (and the project guard's 100 GiB reserve), no foreign GPU process, and
  VRAM headroom (allocator cap 56 GiB of 80).
- **If the budget cannot cover everything:**
  - first priority is A, C, B and D for seeds 0 and 1;
  - then TOKEN-COMPUTE;
  - then controls;
  - final tokens are never reduced for a single condition.
