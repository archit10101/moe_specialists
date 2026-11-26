# csr-continuous-base-target-free-asrs-v2 (asrs2) — preregistration

Written 2026-11-22. It was written after three things and before any scientific GPU job or any calibration result:
- the Phase-0 diagnosis (`docs/CSR_CONTINUOUS_BASE_TARGET_FREE_ASRS_V2_DIAGNOSIS.md`);
- the user's K = 5 amendment;
- the CPU correctness tests (`cpu_tests_20261122T050113Z.json`, 10/10 pass) and the pre-freeze GPU base-matching check
  (`gpu_check_20261122T051207Z.json`, PASS: base loss mean|all arms − base-only| 1.7e-4 vs the base-only repeat floor 1.4e-4, bias −4.6e-5, parameter distance 0.0060 vs floor 0.0059, SIGINT/resume complete).

All of these use main-stream training rows only. One exception: a 4-row SEL smoke evaluation of a test milestone; its
numbers were discarded and never inspected. The document is frozen by the sha256 in the `.sha256` sidecar (mode
0400). Every later change is a logged amendment or deviation.

## 0. Question and success

**Question.** Can target-free semantic assignments create and preserve exposure-matched functional specialist
capability during continued training, while the base receives its complete ordinary TOKEN-4 training trajectory?

**Success needs both:**
- competitive CE (≤ ordinary generic residual G + 0.001, and better than the schedule-matched ALT-G);
- exposure-matched functional semantic specialisation (section 9).

Routing NMI, routing concentration and ordinary (unmatched) ablation are never evidence of functional specialisation
here; they are reported only.

## 1. Evidence this design rests on (verified; details in the diagnosis)

- **asrs1 failed for two independent reasons.**
  - H1: the base got only 320 of 793 updates. That costs +0.36 nats, fully explained by update count.
  - H2: the specialist-training assignment used same-token target utility. On the 22–40% override tokens it is
    target-skewed. The prior-assigned specialist became harmful on new tokens after one specialist phase.
  - Controls without target dependence did not corrupt: ALT-G (same schedule) and the ccsr1 S3 specialists (same class
    and optimizer).
- **ccsr1 (K = 16, frozen specialists):**
  - a causal current-context GRU router deploys a semantic bank 0.004 better than G;
  - region-prior (R1) distillation gives −0.0025 vs G;
  - in a 50M simultaneous continuation, specialisation shrank to ~30% (joint routers) and CE fell to parity.
- **orv1:** a label-free semantic oracle residual bank beats a shuffled one (VAL 4/4).
- **slc:** ordinary interaction metrics are exposure-confounded.

## 2. User amendment: K = 5 (binding)

- Exactly 5 frozen label-free semantic clusters and 5 residual specialists per bank, one specialist per cluster.
- Shuffled control: 5 persistent random clusters with the identical cluster-size distribution.
- Generic control: 5 specialists with identical architecture and top-1 compute.
- Every condition uses the same rows, order, optimizer steps, active specialist count (1), specialist width (256) and
  parameter budget (983,040 expert parameters per bank).
- Labels (M2D2 domains) are opened only for the frozen post-hoc subject audit.
- **Consequence:** the K = 16 cfar1/ccsr1 bank cannot be reused. asrs2 builds its own K = 5 frozen stage (Stage 0F),
  which replaces "the existing positive frozen specialist seed" of the original request.

## 3. Reused artifacts (read-only, sha256-verified at use)

| artifact | path | sha256 |
|---|---|---|
| TOKEN-4 replicate base at step 318 (its K = 16 banks are not used) | runs/20261121T151627Z_cfar1_S1_s100/ckpt_m20/state.pt | 80e1325cfa0dd787b9e400d83d7de7e1681c1cb014aa13438a7f587fee9d9f46 |
| cfar1 run config (model / train / budget / seed 100) | runs/20261121T151627Z_cfar1_S1_s100/config.json | df19aee43a4d52a385751a97d30353d53e71f7970a6b67e2b938804e9345b6b0 |
| ordinary continuation (cross-process base reference only) | runs/20261121T194254Z_ccsr1_S3/ckpt_m1111/state.pt (+ metrics.jsonl) | 6c0a6adf7a1f42a8fcca728272f430636a3930cf09f082c5e00065be0987629c |
| K = 16 region prior main / CALIB-SEARCH | artifacts/csr-crossfit-semantic-utility-residual-v1/tables/psem_main.npz, psem_calib_search.npz | 32885c9f…9775, c33a549d…abc7 |
| frozen GC-EDL SEM-K16 ProtoNet (λ_coh = 0) | artifacts/gc-edl-v1/latent/SEM_K16_s2/net.pt | 6a5e9867…bdd7a |
| CALIB-SEARCH partition, token features | artifacts/oracle-residual-v1/tables/parts.npz, token_feats.npz | 3d7e2b30…6c4a, d7650a96…79b6 |
| CALIB-SELECTION rows (Stage 2 only) | artifacts/csr-current-context-specialist-router-v1/cs2_assign/parts_cs2.npz | c0342a364f010197b66652bc6a7349d93f936f9c1aa6510eb003a054a0570303 |

The protected inventory (1,085 files: asrs1, ccsr1, cfar1, csur1 artifacts and runs, orv1 tables, the ProtoNet, code
and docs) was hashed before any asrs2 GPU job (`protected_inventory_before.json`) and is verified at the end.

## 4. K = 5 clusters (built before the freeze; label-free)

Built by `asrs2_clusters.py`:
- **Points:** the 16 normalised ProtoNet prototypes, weighted by the main-stream region mass of each prototype.
- **Clustering:** weighted spherical k-means with K = 5, 1,000 restarts, 818 distinct partitions. Selected: the
  lowest objective with every cluster's main-stream share ≥ 0.05.
- **Merge map:** [0, 0, 1, 2, 3, 1, 2, 2, 0, 4, 2, 4, 0, 1, 3, 3].
- **Prior:** p5(c | region) = Σ p_sem16 over the cluster's prototypes. The hard assignment is argmax p5 of the
  token's current 256-token region. Paper group sem5 = map[C6 paper label].

| | cluster 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| main-stream region share SEM | 0.211 | 0.118 | 0.253 | 0.252 | 0.166 |
| main-stream region share SHUF | 0.211 | 0.118 | 0.253 | 0.252 | 0.166 |
| CALIB-SEARCH region share SEM / SHUF | 0.202 / 0.202 | 0.128 / 0.128 | 0.240 / 0.241 | 0.268 / 0.268 | 0.162 / 0.162 |

**Persistent shuffled control (SHUF).**
- Each paper gets one fixed permutation σ_p of the 5 clusters, and p5_sh[:, σ_p(c)] = p5[:, c].
- Within-paper switching (0.233) and mean entropy (1.112 nats) are therefore identical to SEM.
- σ_p is chosen in sha256 order of the papers, as the permutation that keeps the running SHUF shares on the SEM shares.
  A purely random σ shrank the large clusters by about 0.06.
- Measured: sorted shares identical to 1e-5; NMI(SEM, SHUF) region labels = 0.005 main, 0.013 CALIB-SEARCH.
- The paper-level SHUF group is σ_p(sem5).

**Tables (0400):** `tables/k5_merge.json` (sha256 3344999a1b3872b789ae06377c93e3269e8c518bbdba162ff662e966ca681f8b),
`p5_main.npz` (2b6e310213b9b35efe0b6013c06fd0bb28465f46a048020bb3355225d20a1686), `p5_calib_search.npz`
(f1eb8460b4a353518294867a84afebe5626b09cd1ce949d85161a3018ccb47f3). `p5_calib_sel.npz` is built with the same map
only after FROZEN_S1.json exists.

## 5. Architecture

- **Base:** the unrestricted TOKEN-4 shared MoE base (103.6M parameters, 12 layers, 32 experts top-4), unchanged. Its
  native experts are always active.
- **Residual bank:** one bank of K = 5 specialists after the final block, each 384→256→384 (`lfr_model.Experts`),
  h' = h + a_e(rms0(h)). Exactly one specialist per token.
- **Router:** current-context causal GRU, the ccsr1 `CausalRouter`: [rms0(h_t), causal mean of rms0(h) over the
  preceding 16 tokens of the document] → Linear → GRU(64) with recurrent state → 5 logits; 74,437 parameters.
- **Inference:** deployed choice = argmax q.
- **No M2D2 label anywhere** before the post-hoc audit.

## 6. Stage 0F — frozen stage (new; required by the amendment)

**Process.** One process. The base is frozen at the step-318 milestone (forward only, never updated).
- **Rows:** main_stream(245e6) steps 0…317, 20,057,391 tokens. These are the rows the base saw and on which cfar1
  trained its banks.
- **Specialist-training seeds:** s ∈ {0, 1, 2, 3}, independent for both bank and router initialisation.
  G5_s, SEM5_s and SHUF5_s of a seed share the same expert initialisation.

**Per step and per seed:**
- **G5_s (token-routed generic):** experts plus router, by LM-through-gate + 0.01 Switch balance, as in jsr/cfar1.
- **SEM5_s and SHUF5_s:** every token trains specialist argmax p5 (or argmax p5_sh) of its region, by the ordinary
  LM loss on sg(h), sg(W). The choice is target-free.
- **Causal routers R_SEM5_s and R_SHUF5_s:** KL(p5 ‖ q) + 0.01 balance on all tokens.

**Optimizers:**
- banks: AdamW (0.9, 0.95), weight decay 0.1, lr = lr_at(step, 3880) (the base schedule including its warm-up),
  clip 1.0;
- routers: AdamW lr 1e-3 with a 20-update warm-up, betas (0.9, 0.99), weight decay 0.01, clip 1.0.

**Milestone.** `ckpt_F` holds all banks and routers; the base is the reused milestone.

**Stage-0F evaluation.** Seed 0 only on SEL (the other seeds are first evaluated in Stage 2).
- **Evaluator identity check:** the base CE on SEL must equal the ccsr1/asrs1 c00 value 4.671856 within 1e-4.

**Gate (seed 0, SEL):**
- **F0-1:** exposure-matched ΔI = I_SEM5 − I_SHUF5 > 0 with paper-bootstrap 95% CI lower bound > 0.
- **F0-2:** ≥ 4/5 SEM5 specialists with positive own-vs-swap utility.

If the gate fails, the outcome is **TERMINAL_STAGE0F_FAIL**: there is no functional K = 5 bank to preserve. CE versus
G5 is reported but not gated.

## 7. Continuation (Stages 1 and 2) — continuous base, alternating auxiliary modules

**Rows.** Steps 318…1110 of the same plan (plan sha256 f4bc3c93…), 50,086,846 tokens. These are exactly the rows,
order and step count of the ccsr1 S3 ordinary continuation.

**Base.** On **every** step the base receives its ordinary update, exactly the ccsr1 S3 code path:
- LM + 0.01 Switch aux + 0.001 z, bf16 autocast, torch.compile, clip 1.0;
- AdamW (0.9, 0.95), weight decay 0.1;
- lr = lr_at(step) × min(1, (step − 318 + 1)/20);
- fresh moments at step 318.

The residual arms see only the detached final hidden states and the detached pre-step tied head. Base gradients are
therefore independent of every specialist and router. This was shown bitwise on CPU (T1) and within GPU
nondeterminism (gpu_check), and is re-checked by C1.

**Auxiliary windows.** Each window is 48 steps: offsets 0–31 SPEC, 32–47 ROUTER, repeating from step 318. The last
window, steps 1086–1110, is SPEC only. In total: 793 base updates, 537 SPEC steps, 256 ROUTER steps.

**Arms per seed s.** All arms are in one process with identical rows and base, initialised from ckpt_F of seed s.

| arm | SPEC windows | ROUTER windows | deployed by |
|---|---|---|---|
| A base | — | — | — (the in-process base) |
| B G (ordinary generic) | experts + router every step | experts + router every step | its token router |
| C ALT-G (schedule-matched generic) | experts only (router frozen) | router only (experts frozen) | its token router |
| D SEMh (R1 hard) | specialist e = argmax p5 trains on the token | router: KL(p5 ‖ q) + 0.01 balance | causal router |
| E SHUFh | the same with p5_sh | same | causal router |
| D′ SEMs (R1 soft) | e ~ p5: one sample per token from a generator seeded by sha256(seed, step, arm); RNG fixed per step and therefore per window | same | causal router |
| E′ SHUFs | the same with p5_sh | same | causal router |
| G FROZ | bank frozen at ckpt_F (never updated) | router refreshed by R1 on the current base | causal router |
| H diagnostics | region teacher (argmax p5 of the current region; non-causal), per-token oracle (target-dependent; never used for training) | | evaluation only |

**Rules for the assigned arms:**
- Only the assigned specialist updates, by CE(sg(base logits) + sg(W) a_e(sg(xn)), y).
- No token's target, CE or gradient enters any specialist choice.
- No specialist is evaluated on the current target to choose one.

**Continuation optimizers:**
- Experts: AdamW (0.9, 0.95), weight decay 0.1, lr = lr_at(step) × a 20-update re-warm-up counted per arm-module
  (G every step; ALT-G experts and the assigned specialists on SPEC steps; ALT-G router on ROUTER steps), clip 1.0.
- Causal routers: AdamW 3e-4 constant, (0.9, 0.99), weight decay 0.01, clip 1.0 (the ccsr1 continuation setting).
- Moments are fresh at step 318.

**Token accounting.** Logged and reported for every run:
- unique main-model tokens = base-update tokens = 50,086,846;
- specialist-update tokens (537 steps), router-update tokens (256 steps), generic G tokens (793 steps);
- diagnostic compute;
- FLOPs, wall time and aux overhead.

**R2 (partner-view utility).** R2 is a candidate only if the frozen-stage reliability test passes, computed by
`asrs2_analyze.r2_reliability` on the Stage-0F SEL tables:
- U[g, e] = mean CE reduction of specialist e on region-group-g tokens of one paper half;
- **reliable** iff the split-half r of the 5×5 tables is ≥ 0.8 **and** on the other half the region teacher
  argmax(log p5 + U[g]/τ) (τ from {0.5, 1, 2} × sd(U), chosen on the first half) beats argmax p5 by ≥ 0.001 nats.

Otherwise R2 is not run. If it is run, it is an engineering generation (section 11). It needs a hashed amendment
specifying U caches measured on paper-disjoint main-stream folds, frozen before the window, with identical budgets
for SEM and SHUF.

**Training-time logs.**
- **Every step:** base loss, row hash, and per arm the residual CE, gradient norms and router KL.
- **Before the step's update, on the 1-in-4 token subset** ((pos + step) mod 4 = 0): held-out utilities of the
  prior-assigned, router-chosen, average and oracle specialist; router agreement with the prior; switch rate; load.
- **Per-window summaries** go to OBSERVATIONS.
- **Parameter hashes** are logged at every window boundary as isolation evidence.

## 8. Checkpoints, interruption, resume

- **Atomic directory checkpoints:** temp dir → fsync → verified reload → meta with sha256 → rename.
- **Resume slot:** one rolling slot every ≤ 100 steps or 10 minutes and at milestones. A superseded slot is deleted
  only after the newer one verifies, and every deletion is logged with its sha256. The final slot of a complete run is
  retained.
- **Milestones:** continuation m510, m702, m894, m1111; frozen ckpt_F. Milestones are never deleted.
- **External interruption** (signal, STOP marker, foreign GPU process, killed job):
  - checkpoint once if safe;
  - mark **INCOMPLETE_STOPPED_BY_EXTERNAL_USER**;
  - stop this experiment's launchers and watchers and release the GPU;
  - write the exact resume command to RESUME.md;
  - wait for explicit permission (`asrs2_driver.py authorize-resume`).
- Never restart automatically. Never kill another user's process.

## 9. Measurements

All on SEL: CALIB-SEARCH half "SEL", 503 papers, about 6.4M tokens. Stage 2 uses CALIB-SELECTION. Every lookup is
exact, using full per-token 5-specialist tables for every token.

- **CE:** base, G, ALT-G, every causal deployment, region teacher, oracle. Paired paper-bootstrap 95% CIs (2,000
  replicates) for the contrasts against G, ALT-G, base and the matched SHUF.
- **Exposure-matched functional interaction I** (asrs2_analyze docstring; csur1/ccsr1 definition ported to K = 5 and
  full tables):
  - own specialist σ(k) = the routing-lift argmax for paper group k, fitted on one paper half and measured on the other;
  - strata: target-frequency × base CE × position × doc length × q_σ(k) quintile × router-margin tercile;
  - **ΔI = I_SEM − I_SHUF** with a paper-bootstrap CI;
  - **retention** = ΔI(continuation) / ΔI(ckpt_F, same seed).
- **Per specialist:** own-vs-swap I_k (5 values with CI), n positive, n significant, largest positive share. The
  ordinary (unmatched) own advantage is also reported.
- **Routing:** load, effective number, dead (load < 0.02), max load, switch rate, region-prior agreement.
- **Location of gains:** position bins 0–31, 32–127, 128–511, 512+; positions ≥ 128 vs ALT-G; post-transition tokens
  (first 64 of a region whose prior label differs from the previous region) and stable tokens; per paper group (5) vs
  ALT-G with n better and max share.
- **Predictability:** cross-fitted by paper halves, majority vote, for the oracle specialist and the assigned
  specialist, from:
  - target token;
  - input token (causal);
  - paper group;
  - region prior;
  - causal router;
  - majority baseline.

## 10. Criteria

**Stage 1A** (SEL, seed 0, milestone m1111). There are two preregistered candidates in the same run: R1-hard
(primary) and R1-soft. A candidate passes iff all of the following hold.

Sanity:
- **C1 base:** the GPU check passed, and `asrs2_basematch.py` passes for the run:
  - 793 base updates;
  - row hashes equal the plan;
  - |mean per-step loss − ccsr1 S3| ≤ 0.002;
  - |SEL base CE − 3.62277| ≤ 0.003.
- **C2:** mean over specialists and tokens of (CE_k − CE_base) ≤ +0.02 (no catastrophic divergence; asrs1 reached
  +0.88).
- **C3:** target-token predictability of the assigned specialist ≤ the SHUF value + 0.05.
- **C4 = S1-8:** CE(SEM) < CE(matched SHUF) with the CI upper bound < 0. This is the "do not degrade identically"
  check and the semantic-over-shuffled check.
- **C5 = S1-6:** deployed routing has 0 dead specialists, effective number ≥ 3 and max load ≤ 0.5.

Accuracy and function:
- **S1-1:** CE(SEM) − CE(G) ≤ +0.001.
- **S1-2:** CE(SEM) < CE(ALT-G).
- **S1-3:** ΔI > 0 with CI lower bound > 0.
- **S1-4:** ≥ 4/5 positive specialists.
- **S1-5:** ΔI(m1111) ≥ 0.75 × ΔI(ckpt_F).
- **S1-7:** positions ≥ 128 better than ALT-G; ≥ 4/5 paper groups better than ALT-G; largest group share of the
  improvement ≤ 0.5.

**Outcome.**
- Stage 1 passes if at least one candidate passes. If both pass, select the lower CE − G; hard wins ties.
- The FROZ control, the trajectory over milestones, post-transition and stable tokens, and early positions are
  reported for every run.

**Failure localisation** (descriptive, using the request's rules):
- base mismatch → implementation;
- region teacher good but causal student bad → router information or distillation;
- both bad → specialist function or assignment;
- SEM ≈ SHUF → persistent grouping only;
- ordinary rises but exposure-matched does not → routing exposure;
- target predicts the specialist → leakage;
- function without CE → capacity, cost or redundancy.

**Stage 2** (CALIB-SELECTION, opened once after FROZEN_S1.json, which freezes the candidate and code):
- **Run:** one process with seeds 0–3 (four independent specialist-training seeds from Stage 0F), arms G, ALT-G,
  candidate SEM, matched SHUF and FROZ, same rows. Each seed's frozen stage (ckpt_F) and m1111 are evaluated on
  CALIB-SELECTION.
- **Success requires all of:**
  1. base trajectory matches (C1 for the Stage-2 run);
  2. seed-mean CE(SEM) ≤ CE(G) + 0.001;
  3. CE(SEM) < CE(ALT-G) in ≥ 3/4 seeds;
  4. ΔI > 0 in ≥ 3/4 seeds, and the hierarchical (seed × paper) CI of the seed-mean ΔI excludes 0;
  5. ≥ 4/5 specialists positive (pooled over seeds), with per-seed counts reported;
  6. seed-mean ΔI(m1111) ≥ 0.75 × seed-mean ΔI(ckpt_F);
  7. no collapse in any seed (C5);
  8. not confined (S1-7, pooled);
  9. CE(SEM) < CE(SHUF), hierarchical CI excludes 0.

**Stage 3** (245M + VALIDATION, only if Stage 2 passes):
- Needs a hashed amendment frozen before it: candidate, hyperparameters, analysis code and rules frozen; 4 seeds;
  VALIDATION opened once; the test split never loaded.
- Comparisons: TOKEN-4, G, explicit EDL, Semantic-EDL C6, shuffled C6, the oracle residual diagnostic, and the
  current-context frozen-specialist result.
- Compute-matched TOKEN-4/G controls are added if the auxiliary overhead exceeds 5%.
- The VAL K = 5 prior needs the VAL p_sem16 table, built under that amendment.

## 11. Engineering loop (Stage 1B; at most 3 generations; CALIB-SEARCH SEL only)

Each generation must satisfy all of:
- the dominant measured failure is diagnosed from the Stage-1 record;
- evidence, the single change, the prediction and the falsifier are written to DECISION_LOG before it runs;
- exactly one factor changes;
- thresholds, tokens, seeds, capacity and analysis code are unchanged;
- every result is preserved and reported.

A generation reruns the full seed-0 continuation from ckpt_F, with all arms, so the controls stay matched.

**Mapping (precedence top-down):**
1. **C1 fails:** implementation failure. Fix it (logged deviation, not a generation) and rerun before any
   interpretation.
2. **Router information.** Region teacher better than ALT-G (CI) but deployed SEM not:
   - if deployed agreement with the region prior < 0.6, **router lr × 3**;
   - if the switch rate < 0.01, **router lr × 1/3**.
3. **Specialist function.** Region teacher not better than ALT-G and (S1-4 or S1-5 fails, or C2 fails):
   **specialist lr × 0.5**. It applies equally to every SPEC-window expert update: SEM, SHUF, soft and ALT-G experts.
4. **Schedule.** S1-2 passes but S1-1 fails, with G − ALT-G < −0.001 (the SPEC/ROUTER split starves residual updates):
   **window ratio 40 SPEC / 8 ROUTER**.
5. **Semantic adds nothing.** SEM ≈ SHUF (C4 fails) while function or CE otherwise pass: **R2**, only if the
   reliability test passed. Otherwise no generation is possible: TERMINAL.
6. **Function without CE** (S1-3/4/5 pass, S1-1/2 fail, and none of 2–4 applies): capacity / redundancy. No allowed
   change: TERMINAL.

**Not allowed:**
- current-target utility for specialist training;
- domain labels;
- validation or test selection;
- multiple simultaneous changes;
- more tokens, seeds or capacity;
- a search beyond 3 generations.

After a failed generation, the next generation is chosen by the same mapping on the newest evidence. After 3
generations without a pass: TERMINAL_STAGE1_FAIL.

## 12. Fairness and integrity

- Paired arms share one process, one base, identical rows, order and step counts. Row hashes are logged per step.
- Initialisation is identical within a seed (G5 / SEM5 / SHUF5 experts; SEM / SHUF router seeds).
- Parameters are matched: 5 × 196,608 expert parameters per bank. G and ALT-G add 1,920 router parameters; the
  causal-router arms add 74,437 GRU-router parameters each, reported as overhead.
- No extra semantic-only data. No adding or dropping seeds. No validation or test engineering.
- All candidates and failures are reported.

**Logs:** OBSERVATIONS.md, DECISION_LOG.jsonl, deviations.jsonl, RUN_LOG.jsonl, run_manifest.tsv, compute_ledger.tsv,
disk_ledger.tsv, deletions.tsv, checkpoint_ledger.tsv, final_analysis.json, RESUME.md, and the report in docs/.

## 13. Budgets

**GPU caps:**

| stage | cap |
|---|---|
| Stage 0 (GPU check + 0F) | 3 GPU-h |
| Stage 1 including generations | 12 GPU-h |
| Stage 2 | 30 GPU-h |
| Stage 3 | 60 GPU-h, re-checked in its amendment |

**Disk.** Free space must stay ≥ 100 GiB at all times. Before every job: free ≥ 100 GiB + the job's estimated peak.

Free space at writing: 137 GiB. Estimated use:
- Stage 0F: ≈ 1 GB;
- Stage 1: ≈ 7 GB per generation (2 rolling slots of about 1.3 GB, 4 milestones of about 0.25 GB, 4 SEL evaluations of
  about 0.7 GB);
- Stage 2: ≈ 12 GB.

If space becomes binding, the experiment stops cleanly and records the exact shortfall.

## 14. Pre-data predictions (recorded now)

1. **Base:** C1 passes; the in-process base equals the ordinary base, and SEL base CE is ≈ 3.623 ± 0.003.
2. **Stage 0F:** ΔI > 0 is likely, since semantic groups specialise (orv1, cfar1). SEM5 vs G5 CE lands between −0.003
   and +0.002.
3. **Stage 1:** no catastrophic corruption. The held-out prior-specialist utility stays positive (helpful) throughout;
   this is the H2 falsifier.
4. **Specialisation shrinks** during continuation, as in ccsr1 SEMj. The most likely failing criteria are S1-5
   (retention < 0.75) and S1-1 (CE vs G).
5. **Probability that some candidate passes all Stage-1 criteria:** about 20%.
