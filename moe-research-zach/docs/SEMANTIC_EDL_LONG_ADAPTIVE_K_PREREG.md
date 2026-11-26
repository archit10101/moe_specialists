# csr-semantic-edl-long-adaptive-k-v1 — Preregistration

Prefix `slk_`; artifacts `artifacts/semantic-edl-long-k/`; configs `configs/semantic-edl-long-k/`; temporary
`.tmp/semantic-edl-long-k/`; runs `runs/<UTC>_semantic_edl_long_k_<name>/`. Written 2026-11-15, after the audit of
semantic-edl-v1 and before any run of this experiment. Hash in `docs/SEMANTIC_EDL_LONG_ADAPTIVE_K_PREREG.md.sha256`.
Later changes only through `DEVIATIONS.jsonl` (recorded before acting) and hashed generation addenda. semantic-edl-v1 is
not modified.

## 0. Questions

Phase 1: does label-free semantic EDL (recovered C6) develop subject-specific, functionally useful experts and
competitive VAL CE at the full 245M-token budget, beyond the effect of arbitrary persistent paper groups (SHUFFLED-C6)?
Phase 2A: is K = 16 limiting (fixed K ∈ {4, 8, 16, 32})? Phase 2B (gated): does a frozen nested hierarchy with a learned
per-layer granularity mixture beat the best fixed K?

## 1. Audit facts (recovered from saved files; nothing inferred from the instruction text)

- C6 (`configs/semantic-edl-v1/s_C6_s100.json`): `paper_latent` with table
  `artifacts/semantic-edl-v1/tables/table_SEM_T05.npz` (sha256 `5d23129b…89d2`; SEM K = 16 medoid seed 2 table q
  sharpened as q^(1/T), T = 0.5, i.e. q² renormalised), β = 0.005, no `layers` key → all 11 MoE layers, no warm-up.
  C1: unsharpened `table_SEM.npz`, β = 0.0055, all layers. C4: as C1 with layers 5–10. C6 is a hypothesis generated
  post hoc by semantic-edl-v1 (best s20 CE and good post-terminal domain FSI), never confirmed.
- Shuffled C6 table: `table_SHUF_SEM_T05.npz` (sha256 `d374a4ec…69fe`) = the same rows permuted over papers that have
  an assignment (numpy default_rng(20261117)); hard agreement with SEM 0.0674 vs chance 0.0661, MI 0.002 nats. A
  permutation of rows preserves exactly: one persistent assignment per paper, the cluster-mass distribution and the
  confidence (max-q / entropy) distribution; β and layers are identical.
- Paper-level constancy: `gc_lib.PaperQ` maps every document (segment) of a paper to that paper's q.
- Data partitions (fspv2 paper-hash buckets, unchanged): main (training, 613.1M tokens, 52,592 papers), calib_search
  (CALIB-SEARCH, 1,015 papers; half A / half B by lfr paper hash), calib_sel (CALIB-SELECTION, 1,057 papers), val
  (VAL, 2,042 papers, 25.0M tokens), test (locked; `fsp_lib.Split` refuses it). CALIB-SELECTION was opened once in
  semantic-edl-v1 (C4 confirmation); this experiment uses it once more, only for the Phase-2A freeze.
- Reusable baselines (fle, identical recipe, plan sha `f4bc3c93…`, 3,880 steps, 245,032,457 tokens, WSD over 3,880
  steps, milestones p040/p060/p080/p100 = steps 1552/2328/3104/3880 = 98.0M/147.1M/196.1M/245.0M tokens):
  TOKEN-4 s0/s1/s2 and explicit EDL (β 5e-3) s0/s1, all COMPLETE, checkpoint hashes in their status.json. 245M tokens
  is the recovered final budget; it uses < 0.40 epoch of main (no repetition; effective epochs reported).
- Frozen semantic inputs for Phase 2: `artifacts/gc-edl-v1/latent/features.npz` S_* (bge-small-en-v1.5, ≤ 4
  separated 256-token views per paper), gc prototype fits SEM_K{4,8,16}_s{0,1,2} (view logits, nets), gc discovery
  metrics. Disk: guard reserve 100 GiB (stricter than the required 50 GiB) enforced on every job.

## 2. Model and optimizer equivalence (all conditions, within a seed)

LFRModel (12 layers, d 384, 11 MoE layers × 32 GELU experts FFN 256, shared expert FFN 768, dense FFN 1792 in the
non-MoE layer, unrestricted dropless top-4, 103,609,728 parameters), init `LFRModel(cfg, seed)`, Pythia tokenizer,
context 1024, the FINAL plan (same documents, same token order for every condition and seed), 64 × 1024 rows/step,
AdamW (0.9, 0.95, eps 1e-8, wd 0.1), lr 6e-4 WSD (1 % warm-up, last 20 % decay to 6e-5), clip 1.0, Switch 0.01, z
0.001, bf16 autocast + torch.compile. Semantic/shuffled EDL differ from TOKEN-4 only by the training-only auxiliary
loss (no parameters, no forward-pass change, identical inference graph and active FLOPs). New runs use
`slk_train.py`, a textual copy of `sem_train.py` (itself a copy of fle_train with the paper_latent path) with only path
constants changed. Checks before training (section 9).

## 3. Phase 1 — conditions, seeds, budget, checkpoints

| condition | definition | runs |
|---|---|---|
| A TOKEN-4 | cond {} | reuse s0, s1, s2; new s3 |
| B EXPLICIT-EDL | cond {edl_beta 0.005} (exact fle config) | reuse s0, s1; new s2, s3 |
| C SEMANTIC-C6 | paper_latent table_SEM_T05, β 0.005, all layers | new s0–s3 |
| D SHUFFLED-C6 | paper_latent table_SHUF_SEM_T05, β 0.005, all layers | new s0–s3 |

Seeds 0, 1, 2, 3 (four paired seeds; never fewer than three for the final comparison). Budget 3,880 steps / 245,032,457
tokens. Milestones (all new runs): s20 = step 318 (20.1M), s50 = 793 (50.0M), p040 = 1552, p060 = 2328, p080 = 3104,
p100 = 3880 (exactly the reused baselines' milestone steps). Resumable slots every ≤ 76 steps (≤ 5M tokens) and at every
milestone. Reused runs have no s20/s50 checkpoints; their trajectories start at p040 (new s3/s2 baselines have all).
Checkpoints are used for trajectories only; the reported final model is always p100.

## 4. Phase-1 evaluation (VAL opened once, after all Phase-1 training is complete — `PHASE1_VAL_OPEN.json`)

No Phase-1 configuration is selected from data; real domain labels are used only for this evaluation.

Accuracy: full-VAL CE (nats/token), per-domain CE, per-document CE; paired per-document bootstrap (B = 2000) of
condition differences within each seed; mean, seed SD and every per-seed difference; trajectory over milestones.

Routing (VAL audit rows, 200k tokens per domain; expert groups / probe fits from CALIB-SEARCH): per-layer MI and NMI
(2·MI/(H_subject + H_expert)) between real domain and selected experts, with a load-preserving routing-shuffle null and
a document-label-shuffle null; domain probe on complete routing features (per-256-token-window L × E selection
histograms): multinomial logistic regression (L2 1e-3, standardisation from training windows) fitted on CALIB-SEARCH
windows (both halves) and evaluated on VAL windows, plus nearest-centroid accuracy; per-expert subject distribution
p(domain | expert) normalised by p(domain) (load-adjusted) and its entropy; within-expert e5 semantic coherence (lfr
`within_expert_similarity`); cross-seed expert-profile consistency: per layer, Hungarian matching of experts between
two seeds on the correlation of their lift profiles, mean matched correlation over layers and seed pairs; the same
metrics for the label-free e5 cluster system.

Functional (fle replacement protocol, one expert at a time, equal compute: the dropped slot is not refilled): expert
(l, e) gets its apparent subject k* = argmax lift on CALIB-SEARCH half-A tokens only; on VAL tokens (PE rows) the
interaction is mean ΔCE on k*-tokens minus the mean ΔCE of other-subject tokens matched on baseline-CE decile ×
position bin ([0,128), [128,512), [512,1024)) (primary; decile-only also reported). Per-document sufficient statistics
give document-bootstrap CIs (B = 1000) per expert and for the condition mean over all 352 experts. An expert is
*subject-specialised* iff lift(k*) ≥ 1.5 on CALIB-SEARCH and its interaction 95 % CI > 0. Mass control: correlation of
interaction with expert load, and the condition difference within load-quintile bins. PE size: 100k tokens/domain at
p100; 50k tokens/domain at p040 (all conditions) and s50 (new runs). `--full` (p100): e5 FSI and the shuffled-label
FSI control. At every milestone: CE, routing metrics, probes, domain FSI (fle group ablation), coherence.

## 5. Primary Phase-1 decision (SEMANTIC-C6 vs SHUFFLED-C6, VAL, p100)

Let n = number of completed seed pairs (4, or 3 under the budget rule). C6 SUCCEEDS iff all hold:
- R (routing): domain NMI (mean over 11 layers) C6 > SHUF in ≥ n − 1 seeds (n = 4) / all seeds (n = 3) and mean
  difference > 0; and logistic-probe accuracy C6 > SHUF with the same sign rule.
- F (functional): mean per-expert domain interaction (primary matching) C6 > SHUF with the same sign rule and mean
  difference > 0.
- CE: mean over seeds of VAL CE(C6) − CE(TOKEN-4) (paired by seed) ≤ +0.010 nats/token.
- P (no pathology at p100): no expert below 0.1× uniform load in any layer, max load ≤ 4× uniform, normalised routing
  entropy ≥ 0.45; parameter count and active FLOPs equal to TOKEN-4 (verified section 9).
"Semantic information beyond persistent grouping" = R and F pass. Explicit EDL is a reference: report
comparable/beats as in semantic-edl-v1 (CE ≤ +0.005 mean; interaction ≥ 0.8 × EDL; beats only with every seed's sign).

## 6. Phase 2A — fixed-K granularity (runs after Phase-1 training; selection never uses VAL or domain labels)

Groupings: gc `fit_prototypes` (SEM representation, λ_coh 0, 30 epochs, τ 0.1, balanced Sinkhorn targets) on DISC-FIT
papers, prototype seeds 0/1/2, K ∈ {4, 8, 16, 32}; K = 4/8/16 reuse the existing gc fits (hash-recorded), K = 32 is
fitted with identical code. Medoid seed per K by mean pairwise matched κ_seed (gc rule). Table = paper_q of the medoid
(softmax of mean view logits), sharpened T = 0.5 (C6 recipe); K = 16 must reproduce table_SEM_T05 exactly (check).
Matched shuffled control per K: paper permutation over papers with q, rng 20261117 (same procedure as SHUF-C6;
K = 16 reuses table_SHUF_SEM_T05). Calib tables (paper_q on CALIB-SEARCH / CALIB-SELECTION views) for pseudo audits.
Label-free clustering metrics on DISC-HOLD (gc code): κ_view, κ_seed, held-out coherence gain C and language-distribution
predictiveness LD (both vs a shuffled-paper baseline), min/max cluster mass, effective clusters. Clustering-eligible:
frac_min ≥ 0.2/K, frac_max ≤ 3/K, effective clusters ≥ 0.75 K in all 3 seeds, κ_view > 0, C > 0, LD > 0.

Screening (CALIB-SEARCH, label-free): seeds 0 and 1, 50M tokens (793 steps of the FINAL plan; identical schedule),
semantic and shuffled run per K; K = 16 uses the Phase-1 C6/SHUF s0/s1 runs' s50 checkpoints (identical config).
Dev evaluation at s20 and s50: CE on CALIB-SEARCH half B; FSI and routing MI for the e5 system (K-independent) and the
K's own pseudo system (groups from half A, measured on half B). Per K: Δe5 = FSI_e5(sem) − FSI_e5(shuf); Δps =
FSI_pseudo(sem) − FSI_pseudo(shuf); ΔCE = CE(sem) − CE(shuf), at s50.
*Repeatable semantic advantage* for K: clustering-eligible, Δe5 > 0 and Δps > 0 in both seeds, mean ΔCE ≤ +0.005.
Selection: among repeatable K, largest mean Δe5 (ties within 0.0002 → higher gc composite S). Freeze (`frozen_2a.json`),
then CALIB-SELECTION once (`CALIBSEL_OPEN.json`): sem vs shuf of K* at s50, seeds 0/1. Confirmed iff FSI_e5 and
FSI_pseudo sem > shuf in both seeds and mean CE(sem) − CE(shuf) ≤ +0.005. If no K is repeatable or K* is not
confirmed: "no fixed K shows a repeatable semantic advantage; adaptive K is not supported by 2A". Real-domain
alignment (NMI of clusters with domains) is reported afterwards, never used.

## 7. Phase 2B — adaptive granularity (gated)

Gate: run iff Phase-1 R and F both pass, or Phase 2A confirms a K*. Otherwise record "not run (gate closed)".
Hierarchy (frozen before any 2B training): fine level = the K = 32 medoid table (unsharpened q32); its 32 clusters are
merged by Ward linkage on q-weighted centroids of standardised mean-view bge embeddings into 16, 8 and 4 groups (exact
nesting); level tables q^(ℓ) = aggregated q32, each sharpened T = 0.5. A level whose minimum cluster mass < 0.2/K is
dropped. Shuffled hierarchy: the same paper permutation (rng 20261117) applied to q32 before aggregation (nested, masses
preserved).
Controller: per MoE layer l, logits a_l ∈ R^4 (zero init); w_l = 0.05 + 0.80·softmax(a_l) (floor 0.05 per level:
no collapse to one level, no exclusive use of the finest). Aux loss = β · mean_l Σ_ℓ w_lℓ · gc_edl_loss(p_l, q^(ℓ)) +
λ_c · mean_l Σ_ℓ w_lℓ · log2(K_ℓ)/5, β = 0.005, λ_c = 0.05. Controller parameters (44 scalars) are training-only (no
inference parameters or FLOPs), AdamW with the main lr schedule, no weight decay. Real domain labels never enter.
Pilot (CALIB-SEARCH, 50M, seeds 0/1): ADAPT and ADAPT-SHUF, used only for the GEN-C trigger. Finals (245M, seeds 0, 1,
2): ADAPT, ADAPT-SHUF, INERT (controller frozen at uniform: parameter-matched inert control), best fixed K* (if K* ≠
16; otherwise Phase-1 C6), with Phase-1 C6, TOKEN-4, EXPLICIT-EDL reused. VAL evaluation as section 4.
Adaptive improves over the best fixed K iff [(a) VAL CE(ADAPT) − CE(best fixed) < 0 in all 3 seeds, or (b) mean
per-expert domain interaction ADAPT > best fixed in all 3 seeds with mean CE difference ≤ +0.005] AND ADAPT > ADAPT-SHUF
in all 3 seeds on domain NMI and on mean domain interaction. Layer granularity reported from w_l.

## 8. Engineering generations (≤ 3 in total; CALIB-SEARCH only; 2 seeds × 50M; ≤ 4 candidates each; never Phase 1)

Each is written as a hashed `docs/SLK_GEN<n>.md` naming the observed failure before it runs; each candidate is compared
with its unchanged matched control (same K and seeds) and accepted only by the 2A repeatable-advantage rule.
- GEN-A (2A; trigger: no K repeatable): assignment temperature T = 0.25 for the two K with the largest mean Δe5
  (+ their shuffled controls). Hypothesis: diffuse labels limit the semantic signal.
- GEN-B (2A; trigger: some K has Δe5 > 0 in both seeds but fails Δps or CE): strength β ∈ {0.0025, 0.01} (≈ 1 % / 4 %
  EDL-relative; C6 β 0.005 ≈ 2 %) for that K. Hypothesis: strength mis-set.
- GEN-C (2B pilot; trigger: max level weight ≥ 0.8 in ≥ 9 of 11 layers, or the finest level is the argmax in ≥ 9 of 11
  layers, in both seeds at s50): λ_c ∈ {0.025, 0.1}. Hypothesis: complexity penalty mis-scaled. Accept iff the trigger
  clears and FSI_e5(ADAPT) > FSI_e5(ADAPT-SHUF) in both seeds.
A generation's accepted candidate replaces the frozen candidate only through the same freeze / CALIB-SELECTION rule;
CALIB-SELECTION is still opened once.

## 9. Code correctness before expensive training

CPU suite (existing sem tests + new `slk_tests.py`): paper-constant assignment across segments, shuffle matching (exact
multiset of rows, masses, confidence distribution, permutation fixed), sharpening, hierarchy nesting and mass
preservation, loss scaling (gc_edl_loss equals fsp edl_loss for one-hot q), bootstrap statistics equal the direct
per-expert computation, VAL/test access guards. GPU (`slk_gpu_checks.py`): TOKEN-4 path of slk_train bit-identical at
step 0 to the reused fle TOKEN-4 recipe and loss trajectory within noise; β = 0 paper_latent run equals TOKEN-4 at step
0; C6 vs SHUF configs differ only in the table; parameter counts equal; SIGINT → checkpoint → resume within noise. A bug
stops affected runs, keeps the faulty artifacts, is documented, fixed with a regression test, and only invalidated work
is rerun.

## 10. Budget, priority, stopping

GPU cap 60 GPU-h total (Phase 1 ≤ 28, 2A ≤ 12, 2B ≤ 20); one GPU job at a time via `scripts/gpu_guarded.sh` and
`slk_driver.py`; search compute (screening, generations, pilots) reported separately from final training and evaluation.
Priority if the budget binds: Phase-1 C6 vs SHUF (never fewer than 3 seed pairs) > TOKEN-4/EDL comparisons > functional
ablations > 2A > 2B. Phase-1 drop order: s50 per-expert, then p060/p080 audits, then seed 3 (all its conditions).
Storage: guard reserve 100 GiB checked before every job; if it cannot be kept without deleting protected work, checkpoint
and stop. Only disposable resumable slots of COMPLETE runs are deleted (ledgered). External interruption: stop
launching, request a stop at the next atomic checkpoint, save full state, mark INCOMPLETE, write RESUME.md with one
resume command, release the GPU, wait for explicit permission. Never interfere with foreign processes.

## 11. Logging

Append-only from the start: RUN_LOG.jsonl (per command: time, code-tree hash (no git repository: recorded as such),
config hash, data-manifest (plan) hash, seed, tokens, checkpoints, exit status, GPU seconds, peak VRAM, disk, validity),
DECISION_LOG.md, RESULTS_LIVE.md, SPECIALIZATION_TRAJECTORY.json (rewritten from append-only sources), DEVIATIONS.jsonl,
RESUME.md (current state). Final: SEMANTIC_EDL_LONG_ADAPTIVE_K_REPORT.md, final_analysis.json, run_manifest.tsv,
compute_ledger.tsv, configs and hashes, reproduction commands.
