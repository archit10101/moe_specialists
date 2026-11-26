# csr-semantic-edl-v1 (Semantic EDL) — Preregistration

Prefix `sem_`; artifacts `artifacts/semantic-edl-v1/`; configs `configs/semantic-edl-v1/`; temporary `.tmp/semantic-edl-v1/`;
runs `runs/<UTC>_semantic_edl_v1_<name>/`. Written 2026-11-15 before any semantic-EDL model was trained. Later changes
only via `artifacts/semantic-edl-v1/deviations.jsonl` and separately hashed generation addenda.

## 0. Question and outcomes

Can stable semantic pseudo-domains discovered without real domain labels replace explicit-label EDL's labels while
preserving EDL's specialisation and matching or improving its LM accuracy?

Method = frozen paper-level semantic clusters q_d + unrestricted per-token top-4 routing + the existing shared expert +
the explicit-EDL population-divergence objective with the one-hot label replaced by q_d. No hard routing, pools, masks,
inference-time encoders, labels or parameters.

Outcomes reported (section 10): BEATS explicit EDL / COMPARABLE to explicit EDL / FAILS; plus the stricter
"no worse in both CE and functional specialisation"; on failure the best-supported diagnosis A–G.

## 1. Reused artifacts (hash-verified by `sem_integrity.py` before use; recorded in `reuse_manifest.json`)

- Data: fspv2 M2D2 five-domain corpus, existing paper-hash buckets (`u_of("fspv2-split", paper)`): main (training),
  calib_search (CALIB-SEARCH), calib_sel (CALIB-SELECTION), val (VALIDATION), locked test (never loaded). These ARE
  the deterministic document(paper)-hash partitions required; re-partitioning would invalidate every matched baseline
  and risk test exposure. CALIB-SEARCH half A/B (lfr paper-hash halves) = discovery / measurement for search audits.
- Semantic grouping (GC-EDL-v1, label-free, frozen): `artifacts/gc-edl-v1/latent/table_SEM.npz`
  (sha 308f1e02…; SEM_K16, medoid prototype seed 2; bge-small-en-v1.5; ≤ 4 separated 256-token views per paper;
  DISC-HOLD κ_view 0.461, κ_seed 0.733; post-hoc NMI 0.437), calib tables `table_SEM_calib_search.npz` (abdd94c5…),
  `table_SEM_calib_sel.npz` (2e4c4e74…), the frozen net `latent/SEM_K16_s2/net.pt` and standardisation
  (`latent/residualizer_params.npz` sm/ssd) for VAL assignment after the freeze. K = 16 is the only K: no other frozen
  SEM table exists, so K = 8 is not tested.
- Baselines (fle, `fle_train.py`, identical recipe): TOKEN-4 finals s0/s1/s2, explicit EDL finals s0/s1 (245,032,457
  tokens, 3,880 steps, milestones p040/p060/p080/p100, diag every 100) with their VAL evaluations
  (`artifacts/forecast-latent-edl-v1/evals/final_{T4,EDL}_s*/final.json`: CE, per-document CE, domain/e5 audits,
  per-expert domain interactions); TOKEN-4 seed-100 search reference (milestones s5/s20/s50).
- Code reused unchanged by import: `fsp_model.edl_loss` semantics via `gc_lib.gc_edl_loss` (tested equal to edl_loss
  for one-hot q), `gc_lib.PaperQ`, `fle_evaluate` (load_milestone, audit_one, per-expert logic), `lfr_eval`,
  `lfr_audit`, `fle_configs.MODEL/TRAIN`.
- `sem_train.py` = `gc_train.py` (itself `fle_train.py` + the paper_latent condition) with sem paths and one added
  option, `paper_latent.layers` (MoE indices; default all 11). TOKEN-4 and EDL code paths are textually identical.

## 2. Isolation

Real domain labels, VAL and test cannot influence candidate design, selection, clustering, hyperparameters, stopping
or training. Before the final freeze no metric reads domain labels: search/selection audits use only pseudo-domain
labels (SEM calib tables) and e5-cluster labels (label-free k-means, existing lfr/fle files); per-domain CE fields are
dropped. The explicit-EDL baselines train on labels by definition (reference only, never a selection input). Domain
labels are first opened after `frozen.json` and the final-run decisions (UNSEALED.json, timestamped), for descriptive
specialisation analysis only. VAL is opened once, after all final checkpoints exist. Test is never loaded.

## 3. Invariants

Identical across TOKEN-4, explicit EDL and semantic EDL within a seed: LFRModel (12 layers, d 384, 11 MoE layers × 32
GELU experts FFN 256 + shared expert FFN 768, unrestricted dropless top-4, 103,609,728 params), initialisation, Pythia
tokenizer, context 1024, balanced main stream and token order (FINAL plan f4bc3c935a47), 64×1024 rows/step, AdamW
(0.9, 0.95), lr 6e-4 WSD (1 % warm-up, last 20 % decay to 6e-5), wd 0.1, clip 1.0, Switch 0.01, z 0.001, bf16 +
torch.compile, milestone schedule per stage, inference architecture/cost. Semantic EDL differs only in the auxiliary
loss grouping.

## 4. Semantic EDL loss

Per affected MoE layer: per-segment mean router softmax m_s; group means P_k = Σ_s q_sk m_s / Σ_s q_sk (q_s = q_d of the
segment's paper; every segment of a paper identical); mean over group pairs of −log(JSD + 1e-8) (JSD ≥ 0 clamp);
averaged over affected layers; coefficient β (possibly ramped). Equals explicit EDL exactly for one-hot q (test).

**Strength scale.** Ratio ρ = ‖∇ β L_aux‖ / ‖∇ L_LM‖ over all parameters (fle `grad_diag`, 8 rows). Explicit EDL's
ratio is ~2 % for most of training but ~4–5 % at 5–20M tokens (fle final EDL s0: 5.2 % @100, 5.7 % @200, 3.8 % @300,
3.1 % @800, 2.0 % @1500+). Targets are therefore defined on the EDL-relative scale: target "r %" = semantic ratio equal
to (r / 2) × explicit EDL's ratio in the same window (steps 100–160), so "2 %" ≙ EDL-matched, "1 %" half, "0.5 %"
quarter. Calibration: one 160-step pilot per (placement, assignment) variant at β₀ = 5e-3 with diagnostics every 10
steps; ρ_variant = median over steps 100–160; ρ_EDL from the seed-100 EDL baseline run (diagnostics every 10 steps
for steps ≤ 160, then 25) over the same steps; β = β₀ · (r/2) · ρ_EDL / ρ_variant, 2 significant digits. Achieved
ratios are measured in every run (diag every 25 steps search, 100 finals) and reported raw and EDL-relative.

## 5. Stage 0 — correctness (before any candidate)

CPU (CUDA hidden, `sem_tests.py`): q paper-constant and frozen (table hash); soft loss = edl_loss for one-hot q;
semantic-EDL gradients reach router weights (and only the selected layers' routers receive aux gradient under
upper-only placement); TOKEN-4 bit-identical when the term is disabled (β = 0 vs no term, tiny CPU run, every step);
shuffled table destroys the association (paper-level hard-label agreement / MI vs SEM at chance); per-expert
multi-label intervention code reproduces fle `per_expert` exactly for the domain system; exact CPU resume; inference
without semantic artifacts. GPU (`sem_gpu_checks.py`): T-EQ (sem_train TOKEN-4 seed 100 step-0 loss bit-identical to
fle s_T4_s100); explicit-EDL ratio reproduction (seed-100 EDL baseline vs fle EDL finals at steps 100/200/300, within
a factor 1.5); finite losses; stop/resume.

## 6. Stage 1 — search (seed 100, 20M tokens = 318 steps of the FINAL plan, milestones s5/s20)

Eight preregistered candidates (A = assignment soft q or sharpened q^(1/T), T = 0.5, renormalised, frozen table built
before any result; P = placement all 11 MoE layers [exact EDL] or upper MoE 5–10; R = ramp: linear 0 → β over the
first 2 % of FINAL steps = 78 steps ≈ 5M tokens):

| id | P | A | target | ramp |
|---|---|---|---|---|
| C1 | all | soft | 2 % | no |
| C2 | all | soft | 1 % | no |
| C3 | all | soft | 0.5 % | no |
| C4 | upper | soft | 2 % | no |
| C5 | upper | soft | 1 % | no |
| C6 | all | sharp | 2 % | no |
| C7 | all | soft | 2 % | yes |
| C8 | all | sharp | 1 % | no |

Baselines at seed 100: TOKEN-4 (reused, s5/s20/s50), explicit EDL (new, 50M, milestones s5/s20/s50).

Dev evaluation (s5 and s20; CALIB-SEARCH half-B CE, label-free audit discovery half A / measurement half B): systems
pseudo (SEM calib table argmax) and e5; routing MI with load-preserving null, topic entropy, topic prediction, load /
starvation / entropy, FSI (group ablation, difficulty-matched) for both systems.

**Elimination** (any): non-finite; any expert < 0.1× uniform load in any layer or routing entropy < 0.45 or any load
> 4× uniform; CALIB-SEARCH CE > T4 + 0.020 at s20; achieved median ratio (steps 100–318) outside [0.4, 2.5] × intended;
routing without function: pseudo MI > T4's but FSI_pseudo ≤ T4's at s20; instability: FSI_pseudo − T4 > 0 at s5 but
≤ 0 at s20.

**Ranking of survivors**: among those with CE ≤ EDL(s100, s20) + 0.010: highest FSI_pseudo − FSI_pseudo(T4) at s20,
ties by lower CE; if fewer than two, fill with the lowest-CE survivors. The best two advance.

## 7. Stage 2 — confirmation (50M, seeds 100 and 101)

Advancing candidates continue their seed-100 runs from the 20M slot to step 793 (same run; milestone s50) and train
seed 101 fresh (s5/s20/s50). Baselines: TOKEN-4 s100 (reused), EDL s100 (stage-1 run), TOKEN-4 s101 and EDL s101 (new,
50M). Shuffled control SHUF_c for each advancing candidate c: identical config with the paper-permuted table (fixed
seed 20261117 over papers with q; load-preserving), both seeds.

**Advance** (all): (i) mean over seeds of CE(c) − CE(EDL) ≤ +0.005 on CALIB-SEARCH half B at s50; (ii)
FSI_pseudo(c) > FSI_pseudo(T4) and > FSI_pseudo(SHUF_c) in both seeds; (iii) direction: FSI_e5(c) ≥ FSI_e5(T4) in both
seeds; (iv) persistence on disjoint papers: groups are assigned on CALIB-SEARCH half A and measured on half B (built
into (ii)), and the FSI_pseudo advantage over T4 is positive at both s20 and s50 in each seed; (v) no
starvation/collapse (Stage-1 rules) at s50.

## 8. Engineering generations (≤ 2 after Stage 2; ≤ 4 new candidates each)

Allowed factors only: strength target within 0.5–4 % EDL-relative, placement (all / upper / lower MoE 0–4), q
temperature T ∈ {0.25, 0.5, 1}, ramp length ≤ 10 % of steps or a late-decay schedule of β (total tokens fixed).
Each generation is written as `docs/SEMANTIC_EDL_V1_GEN<n>.md` and hashed before running, names its measured failure
mode (section 9 localisation) and receives the Stage-2 protocol directly (50M, seeds 100/101, SHUF control for any
candidate passing (i)). No counterfactual losses, forecasts, pools, encoders, data or architecture changes. Rejected
candidates and completed baselines are never rerun.

## 9. Failure localisation per generation

No routing differentiation (pseudo MI ≤ T4) → inspect sign/gradient path/strength/placement. Differentiation without
function (MI up, FSI not) → routing-only; do not raise β indefinitely. Function with worse CE → ramp / temperature /
upper placement. Early gain disappearing → trajectories. Collapse/starvation → reject.

## 10. Stage 3 — freeze, CALIB-SELECTION, finals, VAL

Freeze one configuration (best advancing candidate by Stage-2 ranking: CE ≤ EDL + 0.005 then highest mean FSI_pseudo
advantage; if none advanced after all generations, the best valid candidate by the same ranking). `frozen.json` (0400).
Open CALIB-SELECTION once: 50M checkpoints of the frozen candidate, its SHUF, T4 and EDL at seeds 100/101 (CE 0.4M
tokens/domain; label-free audit, discovery CS half A, measurement CALIB-SELECTION). **Confirmed** iff mean CE(c) −
CE(EDL) ≤ +0.005, FSI_pseudo(c) > T4 and SHUF in both seeds, and no pathology. Finals only if confirmed; otherwise the
experiment ends with the 50M evidence (no VAL).

**Finals** (FINAL plan, seeds 0/1/2, milestones p040/p060/p080/p100, diag every 100): semantic EDL s0/s1/s2, SHUF
s0/s1/s2 (s2 dropped first if the GPU budget would be exceeded), explicit EDL s2 (the minimum missing matched baseline;
EDL s0/s1 and TOKEN-4 s0/s1/s2 reused).

**VAL** (once per new final checkpoint): full-VAL CE + per-document CE (fle `evaluate`); audits (fle `audit_one`:
domain + e5 systems; discovery CS half A, measurement VAL audit subset) at p040/p060/p080/p100 for semantic EDL and EDL
s2, at p040/p100 for SHUF; pseudo-domain system (VAL q from the frozen SEM net) routing + FSI at the same milestones;
per-expert equal-compute replacement interventions (352 experts, fle protocol, PE 50k tokens/domain) at p100 with
domain, e5 and pseudo interactions from the same passes. Reused TOKEN-4/EDL get the pseudo-domain audit (p040, p100)
once; their domain/e5 results come from fle's final.json (identical code).

**Primary specialisation measure** (same label system for every condition): per-expert real-domain interaction
(fle definition) and domain FSI; pseudo-domain and e5 results secondary.

**Success definitions** (per-seed differences primary; paired per-document bootstrap CIs describe documents only):
- *Comparable to explicit EDL*: mean VAL CE(sem − EDL) ≤ +0.005 with the same direction of (sem − T4) in all final
  seeds [sign of sem − T4 CE negative in all seeds], per-expert domain interaction ≥ 0.8 × EDL's (mean over matched
  seeds) and above TOKEN-4 and SHUF in every seed (also domain FSI above both in every seed).
- *Beats explicit EDL*: (a) mean CE lower than EDL, every seed lower, with interaction ≥ 0.8 × EDL (statistically
  comparable); or (b) interaction above EDL in every seed with mean CE ≤ +0.005.
- *Strict*: CE ≤ EDL (mean) and interaction ≥ EDL (mean).
- Otherwise *fails*; diagnosis A–G from the measurements (stability of q, pseudo-domain/expert usefulness alignment,
  strength vs CE trade-off, routing vs function, interference, label information beyond clusters — measured by the
  domain purity of the SEM clusters vs EDL's one-hot groups, seed variance).

## 11. Budget and operations

≤ 24 GPU-h total (all categories), one GPU job at a time, ≥ 100 GiB free (every job checks free − peak), no /dev/shm,
no deletion of prior artifacts; only resumable slots of finished or rejected search runs are deleted (ledgered).
Trainers checkpoint ≤ 600 s and ≤ 5M tokens (every 79 steps) and at every milestone. GPU only via
`scripts/gpu_guarded.sh` through `sem_driver.py`. External interruption (signal, STOP marker, foreign GPU process,
blocked launch): checkpoint at the next step, mark INCOMPLETE / stopped_by_operator, record step, RNG, optimizer,
checkpoint and pending stage in `RESUME.md`, stop all experiment processes, never relaunch automatically.
Budget rule: if the projected remaining cost would exceed 24 GPU-h, drop in this order: SHUF s2 final, the p060/p080
audits of new runs, the second engineering generation.
