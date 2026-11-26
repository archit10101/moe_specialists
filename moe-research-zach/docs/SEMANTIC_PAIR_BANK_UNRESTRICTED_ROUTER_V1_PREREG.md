# semantic-pair-bank-unrestricted-router-v1 (spb1) — preregistration

Written 2026-11-22, before any bank training or any spb1 measurement. It is frozen by the sha256 sidecar `docs/SEMANTIC_PAIR_BANK_UNRESTRICTED_ROUTER_V1_PREREG.md.sha256`. Artifacts go to `artifacts/semantic-pair-bank-unrestricted-router-v1/` (ART). The code is `spb1_*.py` and `src/csr/spb1_*.py`.

## 1. Question and hypothesis

Can label-free semantic groupings create functionally specialised residual experts, while a completely unrestricted causal token router keeps the accuracy advantage of ordinary token routing?

**Design.**
- K = 16 frozen label-free semantic clusters, with M = 2 residual specialists per cluster: 32 specialists, exactly one active per token.
- During bank training, semantic grouping decides *which two* specialists a token can train. A causal token-level pair subrouter decides *which of the two*.
- At deployment, a fresh causal router chooses among **all 32** specialists for every token. It has no mask, no semantic loss, no subject-to-expert mapping and no semantic logit bias.

**Evidence this builds on (read before design; summarised in OBSERVATIONS.md):**
- orv1: an oracle K=16 semantic residual bank beats a shuffled bank (+0.0024) and a linear token-routed bank (+0.0013), using paper-level non-causal assignment.
- ccsr1: a GRU current-context router deploys a frozen semantic bank 0.004 better than G on a weak base, but loses this under joint continuation.
- scur1/asrs2: subject-anchored routing loses flexibility. The free LM-trained router was the best causal router.
- Semantic Lift/EDL: shuffled persistent grouping explains most "semantic" effects, so every claim needs SHUF and exposure matching.

## 2. Fixed components (reused, hash-verified, read-only)

| component | source | sha256 |
|---|---|---|
| frozen base: TOKEN-4 245M ckpt_p100 seeds 0/1/2 | orv1 `T4_RUNS` | fb113eb6…, 0d46807b…, 40c61348… (verified at every load) |
| training plan: base-unseen main docs, sha256 order, 32 × 1024-token rows/step | `artifacts/oracle-residual-v1/tables/plan.npz` | 8fb446ef… |
| partitions: SEL = CALIB-SEARCH paper-half A (503 papers, 6.40M tokens); TRAIN = half B (498 papers, 6.09M tokens) | orv1 `parts.npz` | 3d7e2b30… |
| semantic clusters: hard argmax of the frozen GC-EDL SEM K16 label-free paper table | orv1 `labels_main.npz` / `labels_cs.npz` | ab70c965… / 2e546d75… |
| token features (frequency strata) | orv1 `token_feats.npz` | d7650a96… |
| region prior (transition stratifier only; never a router input) | csur1 `psem_calib_search.npz` | c33a549d… |
| COARSE-16: orv1 condition A at 50M (16 specialists, oracle label; same base seed, plan, LR, budget, init scheme) | `artifacts/oracle-residual-v1/runs/s{0,1,2}/ckpt_50000000.pt` | 9cfbdb61…, 62068403…, 8871ea24… |

**Shuffled control (new, Stage 0; `ART/tables`):**
- Built by spb1_tables.py.
- Every paper gets one persistent random cluster.
- Assignment is capacity-proportional, followed by a mass-only swap refinement, so cluster token mass equals the semantic mass:
  - max share error 1.1e-6 on the 50M prefix, 2.9e-7 on 245M, ≤ 3.7e-6 on each CALIB-SEARCH half;
  - paper NMI with semantics 0.0009.
- Smallest semantic cluster share is 0.029 at 50M, so no cluster is degenerate.
- Hashes: labels_main.npz 3358c76d…, labels_cs.npz 3fce7f85…, tables.json 93ba176d….

## 3. Conditions (identical base, data, order, tokens, width, active compute, optimizer, initialisation)

One residual bank after the final MoE layer (orv1 insertion): h' = x + a_e(rms0(x)), with a_e 384→256→384, W2 = 0 at init, and W1 drawn from a generator identical across conditions.

**Parameters:**
- 6,291,456 bank parameters plus a 12,288-parameter training router per condition.
- Active per token: one specialist (196,608 parameters) plus 32 router logits.

**Training-time routing:** a shared linear router 384→32 on rms0(x), with straight-through gate p/sg(p) and Switch auxiliary weight 0.01.

| condition | routing | load balancing |
|---|---|---|
| SEMANTIC-2 (SEM2) | the token's semantic paper cluster c restricts the choice to {2c, 2c+1}; the router picks within the pair | Switch, within each pair |
| SHUFFLED-2 (SHUF2) | identical, with the balanced persistent shuffled cluster | Switch, within each pair |
| GENERIC-32 (GEN32) | unrestricted over all 32 | Switch over 32 |
| COARSE-16 | orv1 A (one specialist per cluster, oracle label); half the bank parameters | reference only, not a matched control |

**Training:**
- AdamW (0.9, 0.95), wd 0.1, clip 1.0, LR 3e-4 (orv1 pilot; not re-tuned), 50-step warmup then constant.
- 50,022,549 tokens (1,581 steps).
- All three banks train in one process on the same batches with a shared frozen base forward.
- Bank seeds 0, 1, 2 pair with TOKEN-4 base seeds 0, 1, 2.
- Rolling resume slot every 8 minutes; bitwise resume verified on CPU and GPU.

**Deployed router (Stage 2), per bank:**
- Discard the training routers.
- Fresh router: input [rms0(x_t), causal mean of rms0(x) over the last 16 document positions] → Linear → GRU(64) → Linear(32). 76,192 parameters, identical for SEM2/SHUF2/GEN32.
- Loss: expected routed CE over the frozen bank's exact per-token CE table, plus 0.01 × KL(mean routing ‖ uniform). This is the only auxiliary term.
- 2,400 steps, 64 documents/step, AdamW LR 3e-3 (warmup 30 + cosine), wd 0.01.
- Trained on TRAIN; deployed by argmax on SEL.
- Router seeds 0 and 1 for every bank seed. COARSE-16 gets the same router over its 16 specialists.

**Secondary ablation (cannot replace the primary):**
- A frozen causal semantic-posterior net (same backbone, trained on TRAIN to predict the paper's semantic cluster).
- Its 16-dim softmax is concatenated to the router input for SEM2/SHUF2/GEN32, router seed 0.
- The router remains unrestricted and may ignore it.

**Oracle diagnostics (non-deployable; never used to train or tune a deployable router):**
- SEM2 under oracle semantic cluster plus its causal pair subrouter (the training path);
- own-cluster pair hindsight best;
- per-token best of 32 for every bank.

## 4. Data access

**Engineering and all Stage-1/2 decisions** use CALIB-SEARCH only:
- routers train on TRAIN;
- all measurements are on SEL.

**VAL (Stage 3 only):**
- opened once, after `ART/FROZEN_FINAL.json`;
- it reuses the frozen orv1 VAL semantic assignment, plus a balanced shuffle built with the same algorithm.

**Never opened:** CALIB-SELECTION is not used. The test split is never opened, and the code refuses it.

**No inputs from labels or targets:** no explicit domain labels anywhere, and no target or future token in any router input.

## 5. Stage 0 verification (done before freezing)

**CPU tests T1–T12** (`ART/cpu_tests_*.json`):
- causality of the deployed router, the posterior net and the base;
- target isolation;
- pair routing confined to the cluster pair, and GEN unrestricted;
- every token can select each of the 32 specialists;
- parameter and active-FLOP matching with identical initialisation;
- G-ZERO;
- bitwise resume;
- table lookup equal to directly routed CE;
- balanced shuffle;
- VAL and test locks;
- functional metric recovering a planted effect and about 0 under the null;
- end-to-end smoke test.

**GPU:**
- G-ZERO bit-exact and G-HASH;
- bitwise GPU resume (`gpu_resume_test.json`);
- benchmark: about 0.17 GPU-h per seed for 50M training (3 banks), about 0.2 GPU-h per extraction partition, about 0.3 min per router.

**Disk:** start at 125 GiB free. Stage 1+2 needs about 12 GiB (extraction tables 3.7 GiB/seed, checkpoints 0.7 GiB). The reserve of 100 GiB plus each job's estimate is checked before every job.

## 6. Metrics (SEL unless stated)

**CE:**
- d_b[t,e] = CE(base + specialist e) − CE_base (exact tables); routed CE = CE_base + d_b[t, e(t)].
- Paired differences use a per-paper ratio bootstrap (2,000 resamples), pooled over seeds and router seeds. SEL papers are common to every seed.

**Functional:**
- Relative usefulness g_e(t) = mean_{j≠e} d[t,j] − d[t,e]. Secondary: base-relative −d[t,e].
- Exposure-matched own-cluster usefulness of specialist e with training cluster c(e) = e // 2:
  - own set: SEL tokens of cluster c(e) (semantic labels for SEM2, shuffled labels for SHUF2) routed to e;
  - other set: tokens of other clusters routed to e by the **same** routing function;
  - matching: common-support strata with own-set weights.
- Stage-1 ("intrinsic") version:
  - routing is the pair subrouter: other-cluster tokens on which e beats its pair partner;
  - strata: target-frequency tercile × base-CE quintile × position bin (0–31, 32–127, 128–511, ≥512) × doc-length tercile × pair-margin tercile.
- Stage-2 ("deployed") version:
  - routing is the deployed unrestricted router;
  - strata: the same 180 base strata × top-1 prob quintile × top-1 margin tercile.
- I_bank = mean over specialists with ≥ 2,000 tokens in each set.
- ΔI = I_SEM2 − I_SHUF2, with per-paper influence-function bootstrap CIs.

**Also reported:**
- per-specialist I with CI, and the number positive and significant;
- per-specialist replacement loss on training-path tokens (remove to base, replace by partner, replace by mean of others);
- loads: max, effective count exp(H) and 1/Σp², dead (< 1/320), number used;
- switch rate; NMI of routes with semantic clusters;
- oracle-to-causal gaps;
- breakdowns for positions 0–31, 32–127, ≥ 128, the first 32 tokens after a detected topic change, and stable context. A detected change means the region-prior argmax of a 256-token region differs from the previous region's; stable means they match.

## 7. Staged protocol and gates

### 7.1 Stage 1 — specialist banks (3 bank seeds, 50M)

**G1a:** intrinsic ΔI CI lower bound > 0, in ≥ 2 of 3 seeds.

**G1b:** mean over seeds of [CE(SEM2 oracle cluster + pair subrouter) − CE(GEN32 own training routing)] ≤ +0.001 nats.

**Advance** if G1a and G1b both pass. All three bank seeds then go to Stage 2 (no seed selection after outcomes).

### 7.2 Stage 2 — unrestricted routers (2 router seeds per bank seed)

| gate | criterion |
|---|---|
| P1 | pooled CE(SEM2+router) − CE(GEN32+router) ≤ −0.001 nats/token, with 95% CI upper bound < 0 |
| P2 | pooled deployed ΔI CI lower bound > 0, and positive in every bank seed |
| P3 | SEM2 − GEN32 < 0 in every bank seed, and every leave-one-seed-out mean ≤ −0.0005 |
| P4 | for every SEM2 router: effective experts exp(H) ≥ 12 and max load ≤ 0.20 |

**Promote** to Stage 3 only if P1 to P4 all pass.

### 7.3 Stage 3 — 245M confirmation (only if promoted)

1. Freeze the configuration: write `FROZEN_FINAL.json`.
2. Continue all three banks of every seed to 245,032,457 tokens. Constant LR makes this identical to training straight to 245M.
3. Re-extract TRAIN and retrain routers (2 seeds).
4. Open VAL once. Evaluation code is written and tested on SEL before VAL opens:
   - routed CE for all tokens via exact on-the-fly routing;
   - full tables on a fixed 25% VAL paper subset for the functional metric, because of disk limits.
5. Report P1–P4 on VAL. COARSE-16 is reported at 50M only.

## 8. Engineering loop (CALIB-SEARCH only)

**Limits:** at most 3 generations, one factor each, only after a written diagnosis of the measured cause. Each generation logs the diagnosis, alternatives, prediction and falsifier in DECISION_LOG and `gen<g>_change.json` before it runs.

**Permitted factors:**
- M = 2 vs 4, with fully matched controls (SHUF4, GEN64);
- deployed-router capacity H;
- router temperature;
- training-subrouter/specialist LR ratio;
- load-balancing strength (training aux or router balance);
- hard vs soft training-time semantic assignment.

**Reruns:**
- Bank-changing generations retrain SEM, SHUF and GEN for all 3 seeds and rerun Stage 1, then Stage 2 if it passes.
- Router-only generations reuse the frozen banks and rerun Stage 2.

**Never changed:** the backbone stays fixed (label-free semantic bank creation, separate residual capacity, unrestricted causal deployment, no labels, no leakage, no deployment restriction). No extra tokens, steps, parameters, seeds or tuning for the semantic condition only.

**Stopping:** stop after the third generation, or earlier if no permitted change addresses the measured cause, and report the negative result.

**Disk policy:** the extraction tables of a completed, analysed, superseded generation are regenerable scratch. They are deterministic from the retained checkpoints. They may be deleted only after logging path, size and sha256 in deletions.tsv. Checkpoints, router outputs and analyses are retained.

## 9. Interpretation tree (reported explicitly)

| observation | conclusion |
|---|---|
| semantic oracle fails (G1b fails or SEM2 oracle not better than SHUF2 oracle) | multiple semantic specialists did not create a useful bank |
| oracle succeeds but unrestricted routing fails (P1) | deployment is the bottleneck |
| succeeds at 50M but not 245M | drift or long-training interference |
| SEM2 ≈ SHUF2 (CE and ΔI) | persistent grouping, not semantics, caused the effect |
| CE improves without exposure-matched ΔI | not evidence of subject-specific competence |
| ΔI without CE improvement | specialisation exists but is not useful enough for its capacity |

## 10. Predictions (before any data)

**Stage 1:**
- SEM2's oracle path beats GEN32's own linear routing by 0.0005–0.0015 (orv1 A − C was 0.0013 at K=16), so G1b passes.
- Intrinsic ΔI > 0 in 3/3 seeds (orv1 S_A ≈ 0.0048 vs S_B ≈ 0), so G1a passes.

**Stage 2:**
- SEM2 + unrestricted router is within ±0.001 of GEN32 + router, with a central estimate of −0.0003. P1 most likely fails, because the asrs2/scur1 strong-base results showed semantic banks do not beat generic capacity once routing is causal.
- Deployed ΔI is positive (CI > 0), P2.
- The router posterior ablation changes CE by less than 0.0005.

**Expected outcome:** the "deployment is the bottleneck" branch, or "specialisation exists but is not useful enough".

## 11. Compute and storage budget

| item | budget |
|---|---|
| GPU cap, total | 24 GPU-h |
| Stage 0 | 1 GPU-h |
| Stage 1 | 6 GPU-h |
| Stage 2 | 3 GPU-h |
| engineering | 8 GPU-h |
| Stage 3 | 10 GPU-h |

Every job runs via scripts/gpu_guarded.sh (90-minute cap per job; training in resumable legs). The disk reserve is 100 GiB plus each job's estimate.

## 12. Operational rules

- No sudo, no system changes, no /dev/shm.
- No existing artifact is modified; inventory of 805 protected files at `ART/protected_inventory_before.json`.
- External interruption:
  1. treat it as GPU preemption;
  2. checkpoint if possible and update RESUME.md;
  3. release the GPU and stop all spb1 launchers and watchers;
  4. never auto-restart, and wait for explicit user permission.
- Logs: RUN_LOG.jsonl, DECISION_LOG.jsonl, OBSERVATIONS.md, deviations.jsonl, run_manifest.tsv, compute_ledger.tsv, disk_ledger.tsv, deletions.tsv, final_analysis.json, RESUME.md, final report.

## 13. Code hashes at freeze

Listed in `ART/code_manifest_prereg.json`, written at freeze. The code tested by the final Stage-0 CPU test run is the frozen code.
