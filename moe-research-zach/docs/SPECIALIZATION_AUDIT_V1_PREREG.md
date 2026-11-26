# specialization-audit-v1: preregistration (prospectively frozen post-training audit)

**Status.** This audit was designed after the csr-representative-v1 main validation-CE results were known. It is
frozen before any audit data, semantic label, routing record or intervention result exists. It is a separate,
post-training analysis. It is **not** part of the csr-representative-v1 preregistration
(`docs/REP_V1_PREREG.md`) or its amendment, and it cannot alter them. The SHA-256 of this file is recorded in
`ledger/safety_ledger.tsv` (`SPEC_AUDIT_V1_PREREG`) and in `docs/SPECIALIZATION_AUDIT_V1_PREREG.sha256`
before any specialization result is inspected. This file is never edited. Any deviation is reported as a
deviation in the report.

**Fixed facts it builds on.** The csr-representative-v1 CE conclusions are final and are not re-litigated:

| Contrast | Validation CE result |
|---|---|
| A (ordinary token routing) | best validation CE |
| B − A | +0.110 nats; hard pooling hurt |
| D − F | mixed sign across seeds; no consistent boundary effect |
| E − D | +0.026 nats; semantic alignment loss hurt |
| C − B | +0.022 nats; semantic alignment loss hurt |

No model is retrained or modified. Weights, configurations, segmentation, temperatures, λ, routing rules and the
18 checkpoints stay unchanged. Checkpoints are opened read-only; their sha256 is verified before and after the audit.

## 1. Questions and comparisons

**Routing specialization.** Are semantic topic categories assigned to different routed experts?

**Functional specialization.** Is a routed expert disproportionately useful (under counterfactual replacement)
for the semantic topic routed to it?

**Primary comparisons.** These are paired, on identical audit documents and tokens, within each training seed:

| Comparison | Question |
|---|---|
| E − D | semantic alignment loss under semantic chunking |
| C − B | semantic alignment loss under fixed chunking |
| D − F | semantic boundaries vs matched-random boundaries |
| B − A | hard chunk pooling vs ordinary token routing |

**Condition mapping.** Stage 0 verified it from the frozen configs and checkpoint-embedded configs
(`artifacts/specialization-audit-v1/inventory.json`):

| Condition | Routing | Segmentation | Semantic loss |
|---|---|---|---|
| A | token | none | off |
| B | pool | fixed 128 | off |
| C | pool | fixed 128 | on |
| D | pool | semantic | off |
| E | pool | semantic | on |
| F | pool | random F-v3 | off |

The semantic loss acts only during training. At inference, C and E route exactly like B and D.

**Model and layers.**
- Model: 8 blocks. Block 0 is dense; blocks 1–7 are MoE with 32 experts, top-4, and one always-on shared expert.
- Pooled conditions use pool size 8, warm-up 16, and gate_norm "global": the gate of a selected expert is its
  full-softmax router probability, with no renormalization.
- Primary intervention layers: MoE blocks **1, 4, 7** (first, middle, last).

## 2. Audit data (new, audit-only)

**Source.**
- HuggingFaceFW/fineweb-edu @ `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, file
  `sample/10BT/000_00000.parquet` (726 row groups). The same pinned revision and file as all earlier data.
- Excluded row groups: the 16 fwe-v1 row groups (smoke, pilot and earlier evalsets) and the 240 fwe-rep-v1 row
  groups (train, calib, val and locked test of csr-representative-v1). That is 256 excluded, leaving 470 eligible.

**Row groups.** `numpy.random.default_rng(20261101).choice(eligible_sorted, 12, replace=False)`, sorted.
Read by range requests, columns id, text, url, token_count and score, into a new raw file
`artifacts/data/raw/fwe_10BT_000_specaudit_v1_rg12_seed20261101.parquet`. The network access is logged.

**Tokenization.** Pythia tokenizer (EleutherAI/pythia-12b @ bb1e3e71), identical to the experiment: tokens[:1023]
+ EOS. Model-visible text = decode(tokens[:1023]).

**De-duplication.** Same method as the main experiment, applied to model-visible text:
- exact: sha256 of normalize_text;
- near: MinHash, 128 permutations (seed 1234), 32 bands, 5-word shingles, estimated Jaccard ≥ 0.8.

**References.** Processed first and never output:
- model-visible text of every document in fwe-rep-v1 train/calib/val and fwe-v1 train/calib/val, decoded from
  their token files;
- then audit candidates in sorted-id order. A candidate is dropped if it duplicates a reference or an earlier
  candidate.

Candidate ids must not appear in any reference split's id list.

**Locked test splits** (fwe-rep-v1 and fwe-v1).
- Their files are never opened, read, hashed or stat'ed.
- Non-overlap with them is guaranteed at the source level: every test document comes from one of the 256 excluded
  row groups, and audit documents come only from the 12 disjoint row groups.
- A content-level near-duplicate check against test documents is **not** performed. It would require reading test
  content. This is a stated limitation.
- Its consequence is bounded: no parameter is trained or tuned on the audit set, and the audit does not evaluate
  the test split.

**Eligibility and selection.**
- Eligible: kept by dedup, and tokenized length L (with EOS) ≥ 65, which guarantees ≥ 1 retained window.
- Selection: eligible ids in sorted order, permuted by `default_rng(20261101).permutation`; take the first
  **8,192**.
- If fewer than 8,192 are eligible: STOP. No top-up without an amendment.
- Stored order: sorted by id.

**Storage.** `artifacts/specialization-audit-v1/data/` holds tokens, offsets, docs.json and manifest.json:
source row groups, revision, file hashes, counts and dedup report. Files are read-only, never overwritten, and
never used for training or tuning.

## 3. Analysis windows and common mask (condition-independent)

- Prediction positions of a document: p = 0 … L−2 (target token p+1).
- Window w covers positions [128w, min(128w+128, L−1)). A final partial window is retained only if it has
  ≥ 64 positions.
- With n_w ≤ 4 candidate windows, all are used. Otherwise use indices floor(x + 0.5) for
  x = linspace(0, n_w − 1, 4), i.e. spread across the document.
- Window text (for labels) = decode(tokens[a : b]) over the window's input positions.
- **Common mask:** exactly the (document, position) pairs inside selected windows. It is identical for all 18
  checkpoints. Windows never change any condition's native segmentation or routing: every model processes the
  whole document.
- **Halves.** u = sha256("specaudit-v1-half" + id)[:8] / 2^64; **discovery** if u < 0.5, else
  **effect-estimation**. Document-level only.

## 4. Native segmentation of audit documents (model inputs, not labels)

Each condition runs with its own native, frozen segmentation rule:

| Condition | Native segmentation |
|---|---|
| A | none |
| B, C | fixed 128 |
| D, E | semantic: frozen teacher all-MiniLM-L6-v2 @ 1110a243, the same sentence detector and `embed_docs_batched` code; τ = 0.35, min 64, max 256 |
| F | F-v3 policy from `artifacts/segments_random/0c57f77e51ea3734/policy.json`; `hazard_starts_v3` with the accepted application seed 1 and doc key = document id |

- Documents are packed greedily in stored order into 1,024-token rows, with block-diagonal attention exactly as in
  evaluation.
- The teacher is used here only because D/E routing requires it. It is never the audit labeler.

## 5. Independent semantic labels

**Primary label view: independent neural encoder.**
- Fixed order: intfloat/e5-base-v2, then BAAI/bge-base-en-v1.5, then sentence-transformers/all-mpnet-base-v2.
- The first suitable candidate is used. Suitable means all of:
  1. not the teacher, and from a meaningfully different model/training family;
  2. a native transformers `BertModel` or `MPNetModel` with safetensors weights and no remote code;
  3. download ≤ 1 GiB.
- all-mpnet-base-v2 belongs to the same sentence-transformers "all-*" 1B-pair training family as the teacher, so
  it is **not** suitable. Only e5 and bge can qualify.
- The exact commit sha is resolved via the Hub API before download and recorded. Files go to
  `artifacts/encoder/<name>/`; the network access is logged.
- Embedding:
  - e5: input "passage: " + window text, mean pooling over the attention mask;
  - bge: no prefix, CLS pooling;
  - both: max length 512, fp32, L2-normalized.
- These embeddings are labels only. They never enter a model forward pass.
- If neither e5 nor bge can be loaded safely, the TF-IDF/SVD view becomes primary and the failure is recorded. No
  unlisted model is substituted.

**Independent robustness view: TF-IDF/SVD.**
- Lower-cased; token pattern `(?u)\b\w\w+\b`; word unigrams + bigrams.
- Vocabulary: the 100,000 most frequent terms in **discovery** windows (ties lexicographic).
- Weights: raw term counts × idf, where idf = ln((1+n)/(1+df)) + 1 on discovery; then row L2.
- Randomized truncated SVD to 256 dimensions fit on discovery windows: oversampling 10, 5 power iterations with QR,
  seed 20261101. Both halves are projected and L2-normalized.

**Clustering.** The same procedure for both views:
- deterministic k-means on discovery windows: k-means++ initialization, n_init = 20, init i seeded with
  20261101 + i;
- Lloyd iterations until assignments are unchanged (maximum 300);
- an empty cluster is re-seeded with the point farthest from its centre;
- the init with the lowest inertia wins (ties to the lowest i);
- effect windows are assigned to the nearest discovery centroid.

K = **32** is primary; K = 16 and 64 are sensitivity analyses. No example is moved by hand. Cluster names (from top
TF-IDF terms and representative texts) are assigned only after the analysis and never enter any calculation.

**Implementation.** scikit-learn and SciPy are not installed and no package installation is authorized.
TF-IDF, randomized SVD, k-means, multinomial logistic regression and the Hungarian assignment are therefore
implemented in-project (numpy/torch), following the definitions above. Unit tests check them on cases with known
answers.

**Tokens and windows.** Each token inherits its window's topic label.

## 6. Routing extraction

- All 18 checkpoints; eval mode; no gradients; **fp32 with TF32 disabled**. The original evaluation used bf16
  autocast with fp32 routers; the audit uses the exact fp32 computation of the same function.
- One forward implementation (`src/csr/audit.py`) produces routing, baseline CE and the counterfactuals. It is
  tested to reproduce `CSRModel.forward` logits and routing.
- Recorded at common-mask positions, per MoE layer:
  - selected experts (top-4) and their gates; normalized gates ĝ = g / Σ_top-4 g;
  - window-mean full pre-mask router probabilities;
  - constrained flag; candidate-pool bitmask for constrained tokens;
  - document, window and native-chunk identifiers; native chunk length; position;
  - baseline per-token CE.
- The shared expert is always active and is excluded from routed-expert statistics.

## 7. Routing-specialization metrics

All metrics are computed per checkpoint and MoE layer on **effect-estimation** windows. Discovery data is used only
for k-means, expert profiles, top topics and probe training.

**Joint distribution.** P(k, e) ∝ Σ over mask tokens with label k of ĝ_{t,e}: expected gate mass, never a sampled
expert.

**Metrics** (all in nats):
- I(T;E) and NMI = I / ((H_T + H_E) / 2);
- **permutation-adjusted MI** I_adj = I − mean over permutations of I_π (1,000 permutations, §9);
- per-expert topic entropy H(T|e); per-expert purity max_k P(k|e) (descriptive only); KL(P(T|e) ‖ P(T));
- between-expert diversity = mean pairwise Jensen–Shannon divergence of P(T|e) over supported experts;
- zero-load experts (no selection on effect windows); low-support experts (selected in < 32 distinct effect
  windows).

**Within-expert semantic similarity.**
- Uses e5 window embeddings x_w and window masses m_{w,e} = Σ over tokens in window w of ĝ.
- Activation-weighted mean pairwise cosine over distinct windows, computed with the exact linear identity
  C_e = (‖Σ_w m_{w,e} x_w‖² − Σ_w m_{w,e}²) / ((Σ_w m_{w,e})² − Σ_w m_{w,e}²).
- Reported as the excess over the same quantity for all routed mass. The identity is exact, so no reservoir
  sampling is needed.

**Pool-level metrics** (pooled conditions): the same MI with mass 1/8 to each pool member of each constrained
token.

**Topic probe.**
- Window features: per-layer mean normalized gate-mass distribution (7 × 32 = 224), standardized with discovery
  mean and standard deviation.
- Multinomial logistic regression with L2 penalty, **C = 1**: objective Σ_i CE_i + ‖W‖² / (2C), intercept
  unpenalized. L-BFGS in float64 from zero to gradient tolerance 1e-6, maximum 500 iterations.
- Trained on discovery windows, evaluated on effect windows.
- Reported: macro-F1, balanced accuracy, accuracy; the majority-class control; a shuffled-label control (trained on
  one stratified label permutation, seed 20261101).

**Primary routing metrics**, used in the decision rule:
1. **R** = mean over the 7 MoE layers of I_adj (e5, K = 32);
2. **P** = probe macro-F1 (e5, K = 32).

All other metrics are secondary or descriptive.

**Cross-seed stability.**
- Per condition and layer: discovery profiles P(T|e) for all 32 experts. An expert without mass gets the global
  P(T).
- Hungarian assignment minimizing JSD for seed pairs (0,1), (0,2), (1,2).
- Reported:
  - mean matched JSD;
  - three-seed top-topic reproduction rate: seed-0 experts whose aligned partners in seeds 1 and 2 share the same
    argmax-mass topic, and separately the same argmax-lift topic;
  - both against the same 1,000-permutation load-preserving null, which includes re-optimizing the alignment.

## 8. Functional specialization (counterfactual expert replacement)

**Scope.** Effect-estimation half only. Primary layers: blocks 1, 4, 7. All 7 layers form a clearly labelled
secondary analysis, run only if it fits the resource ceiling together with the primary analysis.

**Intervention** (per checkpoint, layer L, top-4 rank j ∈ {0, 1, 2, 3}):
- For every mask token t, as its **own single-token counterfactual**: remove e = topi[t, j] from the eligible set
  at layer L for token t only.
- Eligible set: the token's candidate pool if constrained, all 32 experts otherwise.
- Re-select top-4 by full-softmax probability among the remaining eligible experts. The other three are kept, and
  the next-highest eligible expert joins, so exactly 4 routed experts remain. A pooled replacement stays inside
  the frozen pool.
- Gates follow the condition's gate normalization (global probabilities). The shared expert is unchanged.
- Token t's residual stream is recomputed through all later layers. Every other token's keys and values are held at
  their original values. This is exact for t's own prediction: an intervention at t can affect other tokens only
  at positions > t, and t's pools in later layers come from earlier prefix tokens.
- Tokens where the masked expert was not selected are bit-identical to the original.
- ΔCE_t = CE_cf(t) − CE_orig(t), both from the same fp32 code path.

**Top topic.** For each (checkpoint, L, e), the expert's top topic = argmax_k discovery joint mass P(k, e), K = 32
(ties to the lowest k). Experts with zero discovery mass are excluded as zero-load.

**Groups.** Effect tokens where e was selected at L: **T** (window label = top topic) vs **O** (all other topics).

**Stratification cells.** Deciles are computed over T ∪ O of that expert (quantile edges 0.1 … 0.9):
- baseline CE decile;
- selected-gate decile g_{t,e};
- position-in-document decile;
- native chunk-length bin: [1,64), [64,128), [128,192), [192,∞); A has one bin;
- document-length bin: audit-wide quartiles.

**Estimator** (ATT on the top topic): I_e = Σ_c π_c (mean_T,c ΔCE − mean_O,c ΔCE).
- The sum runs over cells containing both groups, with π_c = n_T,c / Σ n_T,c over those cells.
- A positive value means the expert matters more for its apparent specialty.

**Support.**
- At least **32 distinct windows** among matched T tokens and 32 among matched O tokens; otherwise the expert is
  "insufficient support" and excluded from summaries.
- This threshold is never changed. Coverage and exclusions are reported.

**Reported per expert:** I_e, matched fractions, counts, and the global mean ΔCE. The global effect is reported
but is never interpreted as specialization.

**Summaries.**
- Per checkpoint and layer: equal-expert-weighted mean of I_e (primary) and utilization-weighted mean (secondary).
- Primary functional metric **F** = mean over blocks 1, 4, 7 of the equal-expert means (e5, K = 32).
- The same metrics are computed for the TF-IDF view (and K = 16/64 e5 as sensitivity).

**Resource rule.**
- Before the full run, a representative subset is benchmarked and the total projected.
- If the projection exceeds 4 GPU-hours, the number of effect windows is reduced by one deterministic uniform cap:
  windows with sha256("specaudit-v1-cap" + id + window_index) below a fixed fraction, identical for every
  condition, seed, layer and expert. The cap is recorded.
- Conditions, seeds, experts and the three primary layers are never dropped.

## 9. Statistics

**Load-preserving null.**
- Routing outputs are fixed. Window labels are permuted across documents within strata of document-length quartile
  × window type (full 128 vs partial), separately within each half.
- 1,000 permutations from `default_rng(20261101)`. The same permutation indices are used for every checkpoint,
  label view and analysis (routing MI, probe-free metrics, cross-seed alignment, functional top topic and groups).
- Expert loads, computation and global topic frequencies are preserved.
- Reported: observed − null mean, null 2.5/97.5 percentiles, p = (1 + #{null ≥ obs}) / 1001.
- For a contrast X − Y, the paired null uses the same permutation in both conditions: Δ(π) = M_X(π) − M_Y(π).

**Document bootstrap.**
- Effect-estimation documents; 2,000 resamples; seed 20261101; the same resample matrix for every analysis and
  condition (paired). 95% percentile intervals.
- Routing MI: the joint distribution is recomputed from per-document joint masses.
  - The CI for the R contrast is the bootstrap CI of the raw layer-mean MI difference, shifted by the fixed
    difference of null means.
- Probe metrics: per-document predictions are resampled; the probe is not refit.
- Functional interactions: the documents are resampled with the stratification weights (π_c, and the
  O-token importance weights π_c / n_O,c) held fixed at their full-sample values. The T and O parts are
  recomputed as ratio estimators.
- Document-bootstrap intervals measure audit-document uncertainty only, **not** training-seed uncertainty. With
  three seeds, the decision rests on sign consistency, null exceedance and magnitude.

**Reporting.** Each seed separately plus the three-seed mean. No metric or layer is selected by outcome. No
multiplicity correction is applied; the decision rules are conjunctive.

## 10. Decision rules

**Routing increase of X over Y.** All of:
- (a) R_X − R_Y > 0 in all 3 seeds;
- (b) each seed's bootstrap CI excludes 0;
- (c) each seed's observed ΔR exceeds the 97.5th percentile of its paired permutation null;
- (d) P_X − P_Y > 0 in all 3 seeds;
- (e) reproduction: ΔR under TF-IDF K = 32 > 0 in all 3 seeds.

**Functional increase of X over Y.** All of:
- F_X − F_Y > 0 in all 3 seeds;
- each seed's CI excludes 0;
- each seed's ΔF exceeds the 97.5th percentile of its paired null;
- TF-IDF K = 32 ΔF > 0 in all 3 seeds.

**Categories.** These apply to the semantic-alignment question (E − D, C − B):
1. **Strong evidence of semantic specialization:** routing increase AND functional increase in both E − D and
   C − B.
2. **Context-specific specialization:** category 1's conditions hold in exactly one of E − D, C − B.
3. **Routing segregation without functional specialization:** a routing increase in at least one comparison,
   with no functional increase in any comparison that shows the routing increase.
4. **Functional specialization without clear routing segregation:** a functional increase without the routing
   increase. Reported as unexpected; the measurement sensitivity is investigated descriptively without changing
   this analysis.
5. **No evidence of specialization:** neither criterion is met in either comparison.
6. **Harmful specialization (flag):** if category 1 or 2 holds, the already-completed CE results (E − D +0.026,
   C − B +0.022 nats, worse) make it harmful. The mechanism may work while its current form is not useful for
   language modeling.

D − F and B − A are classified with the same routing and functional criteria and reported with their CE context.
Routing consistency, load balance, purity or clean plots alone are never presented as functional specialization.
Specialization is never called beneficial unless both the functional and the CE evidence support it. The
csr-representative-v1 CE conclusion is not weakened or rewritten.

## 11. Seeds

| Use | Seed |
|---|---|
| Row-group selection, document selection, k-means, SVD, permutations, bootstrap, probe shuffle | 20261101 (k-means init i: 20261101 + i) |
| MinHash | 1234 (unchanged) |
| F-v3 application | 1 (unchanged) |

## 12. Resource ceiling and stopping

**Resources.**
- ≤ 4 GPU-hours in total for all audit GPU work (teacher segmentation, label encoder, extraction, interventions).
- GPU 0 only, one project GPU process at a time, a fresh no-foreign-process check before every GPU command.
- 56 GiB VRAM ceiling; ≤ 90 minutes per GPU process; ≥ 100 GiB free on / at all times.
- Downloads: ≤ 12 FineWeb-Edu row groups and one encoder ≤ 1 GiB, inside the project.
- CPU tests never run concurrently with GPU work. The full CPU test suite (CUDA hidden) passes before GPU analysis.

**Stop and report BLOCKED if:**
- a checkpoint is missing, corrupt or ambiguously mapped;
- the audit would require retraining or altering a checkpoint;
- the locked test split would need to be accessed;
- GPU, disk or shared-server safety would be violated;
- a foreign GPU process appears;
- there is a CUDA error or OOM, or non-finite CE;
- fewer than 8,192 eligible audit documents exist;
- an unanticipated methodological decision is required;
- the 4-GPU-hour ceiling cannot be met without changing a primary comparison.
