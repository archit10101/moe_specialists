# csr-alternating-semantic-residual-specialization-v1 (asrs1) — preregistration

Written 2026-11-21, before any asrs1 GPU training or any asrs1 evaluation. Frozen by the sha256 in the `.sha256` sidecar
(mode 0400) after the CPU tests and the short GPU correctness check (both use main-stream training rows only; neither
opens any calibration partition). Every later change is an amendment or a logged deviation.

## 0. Question and what counts as success

Can **alternating** optimisation (base phase → frozen teacher refresh → specialist phase → router phase) preserve the
exposure-matched functional semantic specialisation of the frozen cfar1/ccsr1 specialist bank while training continues
for a matched 50M tokens, without losing CE against the generic residual bank G?

Success needs **both** competitive CE **and** exposure-matched functional specialisation (section 7). Routing NMI,
routing concentration and ordinary (unmatched) ablation are never evidence of functional specialisation here.

## 1. Evidence this design rests on (read and verified before writing)

- ORACLE-RESIDUAL-V1 (VAL, 4/4 seeds): semantic oracle residual bank S_A 0.00484 vs shuffled 0.00002; semantic beats
  shuffled by 0.00240 CE; 16/16 clusters positive. Non-causal.
- slk Phase 1/2A, slc: semantic EDL / Semantic Lift change routing, not exposure-matched function; the ordinary
  interaction metric is exposure-confounded.
- csur1 / cfar1: the semantic teacher specialises, but past-view causal students lose to G.
- ccsr1 (frozen specialists, step 318): a causal current-context GRU router deploys the semantic bank −0.0038..−0.0041
  vs G (3/3 router seeds), exposure-matched ΔI ≈ +0.009 (CALIB-SELECTION) / +0.0106 (SEL), 16/16 specialists positive on
  CALIB-SELECTION (12–13/16 on SEL). The matched 50M **simultaneous** continuation failed: frozen routers drifted
  (+0.0013 vs G, switch rate 0.13 → 0.27); jointly distilled routers became near document-constant (−0.0004 vs G,
  ΔI 23–38 % of Stage 2).

## 2. Reused artifacts (read-only, sha256-verified at every use)

| artifact | path | sha256 |
|---|---|---|
| start checkpoint (TOKEN-4 replicate base, banks G, L, F2h_T500_w240; step 318, 20,057,391 tokens) | runs/20261121T151627Z_cfar1_S1_s100/ckpt_m20/state.pt | 80e1325cfa0dd787b9e400d83d7de7e1681c1cb014aa13438a7f587fee9d9f46 |
| cfar1 run config (model/train/budget/seed 100) | runs/20261121T151627Z_cfar1_S1_s100/config.json | df19aee43a4d52a385751a97d30353d53e71f7970a6b67e2b938804e9345b6b0 |
| frozen Stage-1 routers R3_c2e0.5 SEM_s0/s1/s2 | artifacts/csr-current-context-specialist-router-v1/routers/S1/R3_c2e0.5/SEM_s{0,1,2}.pt | 5b36768a…, 4bb3d343…, da4163a5… |
| frozen Stage-1 routers R3_c2e0.5 SHUF_s0/s1/s2 | …/R3_c2e0.5/SHUF_s{0,1,2}.pt | 2292ea19…, c3bd9008…, 787e7b62… |
| main-stream region prior p_sem / persistent shuffled prior p_SH | artifacts/csr-crossfit-semantic-utility-residual-v1/tables/psem_main.npz, psem_main_sh.npz | 32885c9f…, 070cd2b1… |
| CALIB-SEARCH priors | …/tables/psem_calib_search.npz, psem_calib_search_sh.npz | c33a549d…, a73337d9… |
| CALIB-SEARCH partition / paper labels / token features | artifacts/oracle-residual-v1/tables/parts.npz, labels_cs.npz, token_feats.npz | 3d7e2b30…, 2e546d75…, d7650a96… |
| ordinary (simultaneous) continuation milestone, step 1111 (controls A, B, C) | runs/20261121T194254Z_ccsr1_S3/ckpt_m1111/state.pt | 6c0a6adf7a1f42a8fcca728272f430636a3930cf09f082c5e00065be0987629c |
| ccsr1 prereg / Stage-3 amendment / stage1.json / stage2.json / stage3_m1111.json | docs/…, artifacts/csr-current-context-specialist-router-v1/ | 2101fa23…, b220e101…, 0e9b6d35…, 4cdfb98a…, df0c654b… |

Full hashes are in `src/csr/asrs1_paths.py` and the baseline-reuse manifest `artifacts/…/reuse_manifest.json`.

**Data:** main stream `main_stream(245e6)` (3,880 steps, plan sha256 f4bc3c935a4710eb…, 64 rows of 1,024 per step),
steps **318 … 1110** = 50,086,846 scored tokens — exactly the rows, order and token count of the ccsr1 ordinary
continuation. Stage-1 evaluation partition: CALIB-SEARCH half "SEL" (503 papers, 6.41M tokens). Router/teacher
supervision uses main-stream training rows only (paper-disjoint from every calibration partition). CALIB-SELECTION is
opened once, only in Stage 2, after FROZEN_S1.json. VALIDATION only in Stage 3. The test split is never loaded. No M2D2
domain label is used for training, tuning, selection or early stopping (post-hoc interpretation only, after all
decisions).

## 3. Architecture (reused exactly)

Unrestricted TOKEN-4 shared MoE base (103.6M params, 12 layers, 32 experts top-4, unchanged) + one bank of 16 residual
specialists after the final block (384→256→384, h' = h + a_e(rms0(h)), 3,145,728 params) + the ccsr1 causal
current-context router (input [rms0(h_t), causal mean of rms0(h) over the last 16 tokens of the document] → Linear →
GRU(64) → 16 logits; 75,152 params). Exactly one residual specialist per token (argmax q_t); the native MoE experts are
always active and are never made hard subject experts. The base is trained by its own LM loss on its own hidden state
and never receives a gradient from a residual bank (as in cfar1/ccsr1), so all arms in one process share one base
trajectory exactly.

## 4. Alternating continuation (one process, one base, all arms)

**Cycle schedule (10 complete cycles, fixed):** cycle i (0..9) covers steps [318 + ⌊793 i/10⌋, 318 + ⌊793 (i+1)/10⌋)
(79 or 80 steps); inside a cycle of L steps: BASE = 32 steps, SPECIALIST = 32 steps, ROUTER = L − 64 (15 or 16) steps.
Per cycle ≈ 2.02M / 2.02M / 0.94–1.02M tokens; totals 20,209,963 base / 20,202,508 specialist / 9,674,375 router
tokens = 50,086,846. Every token is consumed once, in exactly one phase.

1. **BASE phase:** banks and routers frozen (not run); the base trains exactly as cfar1/ccsr1 (LM + Switch aux 0.01 +
   z 0.001, bf16 autocast, clip 1.0, AdamW betas (0.9, 0.95), wd 0.1, lr = lr_at(step, 3880) (= 6e-4 on these steps)
   × min(1, (base steps done + 1)/20)). Atomic checkpoint at the phase end.
2. **TEACHER REFRESH:** everything frozen. For the cycle's specialist- and router-phase rows (main-stream training rows
   not yet seen by any parameter: temporally cross-fitted), the frozen base computes h; for each semantic/shuffled bank
   the exact utility u[t,k] = CE_base − CE(base + a_k) (all 16 specialists, `all_deltas`), and
   **π_t = softmax(u_t / (c·s_u) + η·log(p_t + 1e-4))** with the ccsr1-selected **c = 2, η = 0.5** (R3_c2e0.5; τ = c·s_u),
   p_t = the bank's own region prior of the token's current 256-token region (p_sem for SEM, p_SH for SHUF);
   s_u = median over the cache tokens of the per-token SD of u over the 16 specialists, med_margin = median of
   (max_k u − mean_k u) — the ccsr1 statistic definitions, recomputed at each refresh on that refresh's tokens.
   Specialist targets: e*_t = argmax π_t. Router targets: π_t and w_t = clip(margin_t / med_margin, 0.25, 4). The cache
   (assignments, π (fp16), w, targets for alignment checks, statistics) is detached, written atomically, made read-only
   and sha256-hashed **before** the phase that uses it; every use re-verifies the hash and the target alignment.
   No gradient ever flows through the estimator.
3. **SPECIALIST phase:** base and routers frozen; only the 16 specialists of each bank train, by the ordinary next-token
   LM loss on sg(h), sg(W) (`head_ce_grad`); every scored token updates only its cached teacher specialist e*_t.
   AdamW betas (0.9, 0.95), wd 0.1, lr = lr_at(step) × min(1, (specialist steps done + 1)/20), clip 1.0 per bank. No
   federation sync, no auxiliary loss, assignments never change inside the phase. Atomic checkpoint at the phase end.
4. **ROUTER phase:** base and specialists frozen; each causal router trains on the cached targets:
   loss = Σ_t w_t KL(π_t ‖ q_t) / Σ_t w_t + 0.01 · Σ_k q̄_k log(K q̄_k) (the validated ccsr1 R3_c2e0.5 objective). The
   prompt's "already validated routed LM term" does not exist: no routed-LM / expected-utility term was ever validated
   (ccsr1 defined one, E1 β_eu, as an untriggered engineering option; the selected config has β_eu = 0), so none is
   added. AdamW lr 3e-4 constant, betas (0.9, 0.99), wd 0.01, clip 1.0 (the ccsr1 continuation router settings), one
   update per step. Atomic checkpoint + cycle milestone at the phase end.
5. **EVALUATE** each cycle milestone on SEL (section 6).

Optimizer states: none exist in the start milestone (the cfar1 resume slot no longer exists), so all AdamW moments are
created fresh at step 318 and then **carried across cycles** (base, each bank, each router: separate optimizers stepped
only in their own phase).

**Arms (one process, identical rows):**

| arm | bank init | specialist assignment in the specialist phase | deployed by |
|---|---|---|---|
| ALT-G | cfar1 G (token-routed generic bank) | its own linear token router (frozen in the phase) | its token router, trained only in router phases (LM-through-gate + Switch balance 0.01, `residual_step`, bank lr schedule with its own 20-step re-warmup) |
| ASRS-SEM | cfar1 L (semantic) | cached semantic-plus-utility teacher (p_sem) | 3 causal routers, init ccsr1 R3_c2e0.5 SEM_s{0,1,2} |
| ASRS-SHUF | cfar1 F2h_T500_w240 (matched persistent shuffled) | identical pipeline with p_SH | 3 causal routers, init ccsr1 R3_c2e0.5 SHUF_s{0,1,2} |

The teacher never depends on a router, so each bank's trajectory is router-independent; the three routers per bank
measure router-seed variance only. ALT-base = the alternating process's base without residual (control A′).

**Reused controls (not rerun; verified):** A = ordinary base continuation and B = ordinary G (ccsr1 S3 milestone m1111,
same 50,086,846 rows, base updated on all of them); C = ccsr1 simultaneous semantic continuation SEMj_r (joint router)
and SEMf_r (frozen router) from the same milestone. F = teacher / oracle diagnostics (evaluation only). These are
evaluated once on SEL with the asrs1 evaluator (a new measurement of an existing checkpoint, not retraining).

**Pre-data prediction (recorded now):** because the alternating base receives only 320 of 793 optimizer steps
(20.2M of 50.1M tokens), ALT-base will be far worse than the ordinary base. ccsr1 S3 training loss was 3.91 at
step ≈ 636 (≈ 318 + 320 base steps) vs 3.58 at steps 1090–1110, so criterion S1-1 / 7.1 (ASRS-SEM ≤ ordinary G + 0.001)
is predicted to **fail by ≈ 0.3 nats (range 0.15–0.45)**, independently of the specialists. The criterion is the
user's and is kept verbatim; it is not reinterpreted. The mechanism question (does alternating preserve exposure-matched
specialisation against the schedule-matched ALT-G and the shuffled control?) is answered by criteria S1-2 … S1-7.

## 5. Checkpoints, interruption, resume

Atomic directory checkpoints (temp dir → fsync → verified reload → meta.json with sha256 → rename). One rolling resume
slot (KEEP = 1) written at every phase end, after every refresh, every ≤ 100 steps or 10 minutes, and on interruption; a
superseded slot is deleted only after the newer one verifies (logged with sha256 in deletions.tsv). Cycle milestones
`ckpt_c01 … ckpt_c10` (base bf16 matmul weights, banks and routers fp32, teacher statistics) are never deleted. The final
resume slot of the run is retained. Teacher caches are retained.

Interruption by anyone else (SIGINT/SIGTERM/SIGHUP, STOP marker `artifacts/…/STOP`, a foreign GPU process, a killed
job): checkpoint once if the handler can do it safely, mark **INCOMPLETE_STOPPED_BY_EXTERNAL_USER**, stop all of this
experiment's launchers/watchers, release the GPU, write the exact resume command to RESUME.md, and wait for explicit user
permission (`asrs1_driver.py authorize-resume`). Never restart automatically, never poll for the GPU.

## 6. Measurements (per cycle milestone and for every reference; SEL, all 503 papers)

The asrs1 evaluator computes, with exact table lookups (CE = CE_base + d[choice]): CE of the base, of G / ALT-G, of
every deployed router-bank pair, of the region teacher (argmax region prior; non-causal), of the stale past-view student
(reference only); the full 16-specialist delta table on the 25 % paper swap subset (u_hash("csur1-sub", paper) < 0.25) →
per-token oracle and the R3 teacher (target-dependent; non-causal) on that subset. The Stage-2 analysis code of ccsr1
(`ccsr1_analyze.evaluate_config`, unchanged, with G := the comparison bank) gives, per router seed:
- **exposure-matched own-vs-swap functional interaction** I (csur1 `functional`: own-specialist advantage on tokens the
  deployed router sent to that specialist, re-weighted over 2,025 strata, cross-fitted halves) for SEM and SHUF;
  **ΔI = I_SEM − I_SHUF** with a 2,000-replicate paper-bootstrap CI;
- per-specialist own-vs-swap utility (16 values, n positive, n significant), swap matrix;
- load, effective number (1/Σp²), dead specialists (load < 1/160), max load; switch rate; region-prior agreement;
- position bins 0–31, 32–127, 128–511, 512+; post-transition tokens (first 64 tokens of a region whose prior label
  differs from the previous region) and stable-region tokens, each vs the comparison bank and vs the teacher;
- per-semantic-group CE vs the comparison bank (16 groups); retention of the teacher's semantic-over-shuffled advantage.

**Frozen-stage reference:** cycle 0 = the start checkpoint (step 318) with the frozen ccsr1 routers, evaluated by the
same evaluator. Measurement identity check: its CE values must match ccsr1 stage1.json (R3_c2e0.5, SEL) within 3e-4 and
its ΔI within 1e-3 per seed; otherwise the evaluator is fixed before any continuation result is examined.

Training-time diagnostics logged per phase: base LM; per bank the assigned-specialist CE, teacher-assignment load /
effective number, share of tokens where argmax π ≠ argmax prior (utility overrides), s_u, med_margin; per router the KL,
agreement with argmax π at the first and last step of the router phase (drift caused by the preceding base and
specialist phases), switch rate, load.

**Failure attribution (per cycle, descriptive):** base drift = router agreement with argmax π at the start of the router
phase vs the end of the previous one, and switch-rate change; router collapse = switch rate < 0.03, region agreement,
load; specialist forgetting = per-specialist own-utility and ΔI under the region teacher across cycles; teacher noise =
utility-override share and teacher-assignment agreement with the region prior; shrinking residual headroom =
CE(G) − CE_base, oracle − base and teacher − base across cycles.

## 7. Criteria

ΔCE values are per router seed r at the final milestone (cycle 10). "Ordinary G" = control B on the same partition.

**Stage 1 (SEL, the existing specialist seed 100; all must hold for every router seed r ∈ {0,1,2}):**
- **S1-1** CE(ASRS-SEM_r) ≤ CE(ordinary G) + 0.001.
- **S1-2** CE(ASRS-SEM_r) < CE(ALT-G).
- **S1-3** ΔI_r > 0 and its paper-bootstrap 95 % CI lower bound > 0.
- **S1-4** ΔI_r ≥ 0.75 × ΔI_frozen, ΔI_frozen = the mean over r of the cycle-0 ΔI_r.
- **S1-5** ≥ 12 of 16 specialists with positive own-vs-swap utility (all 16 reported).
- **S1-6** no collapse of the deployed ASRS-SEM routing: 0 dead specialists, effective number ≥ 8, max load ≤ 0.25.
- **S1-7** not confined to early tokens or one group: positions ≥ 128 CE(ASRS-SEM_r) − CE(ALT-G) < 0; ≥ 12 of 16
  semantic groups better than ALT-G; the largest group's share of the positive improvement ≤ 0.5.
- (S1-8 report only: post-transition and stable-region results, separately.)

Stage 1 passes iff S1-1 … S1-7 hold for every router seed. Comparisons vs ALT-G, A, C and F are all reported.

**Stage 2 (CALIB-SELECTION, opened once after FROZEN_S1.json; 4 specialist-training seeds = seed 100 + 3 new
seeds; required 50M success, the user's list verbatim):** (1) ASRS-SEM CE ≤ ordinary G + 0.001 nats; (2) ASRS-SEM beats
ALT-G in ≥ 3/4 specialist seeds; (3) exposure-matched semantic-minus-shuffled usefulness positive in ≥ 3/4 seeds and its
hierarchical (seed × paper) CI excludes zero; (4) ≥ 75 % retention of each seed's frozen-stage exposure-matched
specialisation; (5) ≥ 12/16 specialists positive own-vs-swap; (6) no dead specialists, effective ≥ 8, max load ≤ 0.25;
(7) gains not confined to the first 128 tokens or one semantic group; (8) post-transition and stable-region reported
separately. Per specialist seed the deployed router is router seed 0 (pre-declared), the other router seeds are reported.

**Stage 3 (245M + VALIDATION):** only if Stage 2 passes, under a hashed amendment frozen before it (candidate and all
analysis code frozen; same alternating mechanism; 4 specialist seeds; VALIDATION opened once; comparisons with TOKEN-4,
explicit EDL, Semantic-EDL C6, shuffled C6, G and ALT-G at matched seeds/budgets; compute-matched TOKEN-4/G controls if
the ASRS overhead exceeds 5 %). A 245M success requires competitive CE and exposure-matched functional specialisation.

## 8. Engineering generations (Stage 1 only; at most 2; CALIB-SEARCH SEL only)

After a Stage-1 failure, a generation is run only if (a) the failed criteria are attributed to one measured failure
class, (b) exactly one allowed factor is changed (cycle length / phase ratio, teacher-refresh frequency, router lr,
specialist lr, assignment temperature c), (c) the evidence and a falsifiable prediction that **all** failed criteria can
pass are logged before it runs, and (d) its analysis code is unchanged. Mapping (precedence top-down):
1. **Base-update deficit (S1-1):** if CE(ASRS-SEM_r) − CE(ordinary G) − 0.001 > 0.004 for any r (more than the largest
   causal deployment advantage over G ever measured, 0.0041), S1-1 is not recoverable by the specialists. A phase-ratio
   generation is permitted only if, with base share ≤ 0.8 (specialist + router phases keep ≥ 20 % of the tokens), the
   ordinary ccsr1 S3 training-loss curve predicts a base deficit ≤ 0.003 at the new base step count; otherwise no
   generation is run and the outcome is **TERMINAL_STAGE1_FAIL** (predicted: base share 0.8 ⇒ 634 of 793 steps ⇒
   ≈ 0.09 nats deficit, so no generation).
2. **Router collapse** (switch rate < 0.03 or S1-6 fails): router lr × 1/3.
3. **Router drift** (router agreement with argmax π at router-phase start < 0.8 × the previous end in ≥ 5 cycles and
   switch rate > 2 × cycle 0): add a second teacher refresh before each router phase (refresh frequency).
4. **Teacher noise** (utility overrides > 50 % of specialist-phase tokens and S1-3/S1-4 fail): c = 4.
5. **Specialist forgetting** (S1-4 fails with the region teacher's ΔI also < 0.75 × its cycle-0 value): specialist lr × 0.5.
No retrospective selection: the generation's result is final for Stage 1 whether better or worse.

## 9. Integrity, fairness, logging

Identical rows/order/budget for paired arms (one process; per-phase row hashes logged); same initialisation and
architecture; parameters: ASRS = G + 69,008 router params (0.07 % of the base); training compute of the teacher refresh
(16-specialist utilities on 29.9M tokens per semantic/shuffled bank) reported as overhead; no extra main-model tokens;
no added/dropped/rerun seeds; every candidate reported. Protected inventory (all reused files and the cfar1/ccsr1/csur1
artifacts and run directories) hashed before the first job and verified at the end. Logs:
artifacts/csr-alternating-semantic-residual-specialization-v1/{OBSERVATIONS.md, DECISION_LOG.jsonl, deviations.jsonl,
run_manifest.tsv, compute_ledger.tsv, disk_ledger.tsv, deletions.tsv, checkpoint_ledger.tsv, final_analysis.json,
RESUME.md}; final report docs/CSR_ALTERNATING_SEMANTIC_RESIDUAL_SPECIALIZATION_V1_REPORT.md. Concise evidence and
decisions only.

## 10. Budgets

GPU caps: Stage 1 including generations 12 GPU-h; Stage 2 30 GPU-h; Stage 3 60 GPU-h (re-checked in its amendment).
Disk: ≥ 100 GiB free at all times; before every job free ≥ 100 GiB + that job's estimated peak. Stage-1 estimated peak
≈ 20 GB (two resume slots of ≈ 2.0 GB during rotation, 11 milestones ≈ 0.26 GB, teacher caches ≈ 1 GB, 12 SEL
evaluations ≈ 0.6 GB each, GPU-check scratch ≈ 4 GB, deleted after verification and logged). Free at writing: 149 GiB.
CPU tests and the GPU check precede the freeze; no scientific data are seen before it.
