# SCRGP-V1 — Subject-Conditional Residual Gradient Projection: report

**Status: TERMINAL — STAGE1_FAIL.** The mandatory frozen virtual-update test (Stage 1) failed three preregistered criteria:
- V0: matching;
- V4: global CE;
- V6: concentration.

Following prereg §5.5 and the protocol, the experiment stopped. No Stage 2 or Stage 3 was run and no model was trained. Nothing is still running.

**Subject-specific expert competence was not demonstrated.**

Prereg: `docs/SCRGP_V1_PREREG.md` (sha256 `834493c6caf88b981707e6abaeeaf017b36655e37791f8d91c82cf49b5ff09c5`, frozen before any Stage-1 job).

Frozen verdict: `artifacts/scrgp-v1/final_analysis.json` (sha256 `c3db6bbf88ba3696326011df6b130b9a051660dbb1ab32eadd5fb06cb270492a`). Tables are in `artifacts/scrgp-v1/stage1_tables.md`.

Post-hoc diagnostics: `artifacts/scrgp-v1/diagnostics_posthoc.json` (sha256 `8932101a…`). They are descriptive only and are labelled **[post hoc]** wherever used. Mechanistic detail is in `SCRGP_V1_DIAGNOSTIC_NOTES.md`.

## 1. Question and short answer

**Question.** Does the label-free, subject-conditional residual gradient make a routed expert more useful *per invocation* for its assigned semantic subject? The gradient is u(k,e): the positive-minus-matched-control CE gradient, projected orthogonal to the balanced global forced-exposure gradient. The test is under exposure-matched evaluation, without hurting global CE or other subjects.

**Short answer: no.**

- **What the update does.** It is a one-step virtual update θ_e′ = θ_e − η·u(k*, e), applied to frozen Semantic-EDL C6 checkpoints and evaluated on a paper-disjoint held-out half. It moves the assigned-vs-matched-control CE contrast in the predicted direction, for the semantic table and not for the shuffled table, in 4/4 seeds.
- **The contrast is a relative effect.** The update mostly *worsens other subjects* more than it *improves the assigned one*:
  - The balanced global forced CE change G_SEM (+0.00215 nats, worse) is about as large as the contrast gain F_SEM (+0.00229).
  - Applying all 352 per-expert updates together raises held-out natural CE by **+0.58 to +1.62 nats** (threshold +0.001).
  - 81 % of the contrast comes from the last MoE layer. There, the assigned cluster itself gets *worse* under forced insertion, just half as much as the other clusters **[post hoc]**.
- **What would count as competence.** The per-invocation subject benefit would have to be a gain for the subject at no cost to others, which is what the prereg defined as competence. It was not observed.

## 2. What was run and what was reused

| item | run or reused | detail |
|---|---|---|
| Semantic EDL C6 checkpoints, seeds 0–3 (`ckpt_p100`, 245,032,457 tokens, step 3880) | **reused**, frozen | sha256 verified against status.json, C6 table sha and β = 0.005 (`artifacts/scrgp-v1/reuse_manifest.json`, all_ok) |
| GC-EDL CALIB-SEARCH semantic table (K = 16) | **reused** | derived to `tables/table_SEM_T05_CS.npz` (sha `3a360636…`); shuffled control `table_SHUF_SEM_T05_CS.npz` (paper permutation, seed 20261120, sha `de6c4a99…`) |
| CALIB-SEARCH partitions (paper-disjoint) | **run**, label-free hash order | A1 400,182 tokens / 151 papers; A2 400,107 / 153; B 500,032 / 240 (`partitions.json`) |
| Stage 0: inventory, CPU tests (14), GPU tests (G-GRAD, G-OFF, G-RESUME, G-EVAL), bench, numerics | **run** | all pass; 0.057 GPU-h |
| Stage 1: estimation on A1/A2 + virtual update on B, 4 seeds × 2 tables × 352 (layer, expert) pairs | **run** | 8 jobs, 5.376 GPU-h, no interruptions |
| Stage 2 (A/B/C continuation, 20M/50M tokens) | **not run** | gated by Stage 1 |
| Stage 3 (245M, 4 seeds, VAL) | **not run** | gated; VAL never read; test split never read |

The CALIB-SEARCH split was the only data used. No real-domain labels were used for anything. The model, tokenizer, data, context length and checkpoints were all unchanged.

## 3. Stage-1 design (preregistered)

**Association.** For each table τ and each (l, e), k* = argmax lift P(e | k)/P(e) is taken from natural routing on A. This gives one pair per expert: 352 pairs = 11 MoE layers × 32 experts. The median k* lift is SEM 1.67 and SHUF 1.21.

**Gradients.** Exact CE gradients on expert e's W1/W2 under forced insertion of e into the 4th slot. The top-3 experts, the shared expert, attention and the gate distribution are all kept.
- Positive tokens are weighted by the sharpened soft q_k*.
- Controls are matched in 1,215 strata: CE × position × router rank × gate × margin × token frequency.
- d = g_pos − g_ctrl.
- u = d minus its projection on the balanced global forced gradient g.
- No normalisation of d or u is applied anywhere.

**Step.** η_target = 0.002 / median‖u‖² is one global value per checkpoint, the same for both tables. It is capped so the median relative step is ≤ 5 %. The cap was binding in all 4 seeds (η ≈ 332–352).

**Held-out half B.** For each pair the following were computed:
- **F:** forced assigned-vs-matched-control contrast change. This is the primary metric; positive means better for the assigned subject.
- **G:** balanced global forced change; positive means worse.
- **N:** exposure-matched natural change in the exact replacement interaction (slc I_cond strata).
- **Sim:** natural CE with all 352 pairs updated simultaneously; positive means worse.

The parameter hash was restored and verified after every evaluation (`hash_restored: true` in all seeds).

**Statistics.** Hierarchical bootstrap over seeds and papers: 1,000 paper replicates and 10,000 hierarchical replicates.

## 4. Results (preregistered; ordinary vs exposure-matched)

Seed order is s0, s1, s2, s3. Units are nats. The value is the mean over 352 pairs.

| metric | type | per seed | seeds > 0 | mean | hier. 95 % CI |
|---|---|---|---|---|---|
| F_SEM | forced, equal exposure | +0.00157, +0.00156, +0.00422, +0.00183 | 4/4 | **+0.00229** | [+0.00155, +0.00358] |
| F_SHUF | forced, equal exposure | +0.000016, −0.000038, +0.000014, +0.000014 | 3/4 | +0.0000017 | [−0.000034, +0.000037] |
| F_SEM − F_SHUF | | +0.00155, +0.00160, +0.00421, +0.00182 | 4/4 | +0.00229 | [+0.00155, +0.00357] |
| N_SEM | **exposure-matched natural** replacement interaction | +0.00333, +0.00064, +0.01132, +0.00343 | 4/4 | +0.00468 | [+0.00138, +0.00931] |
| N_SHUF | exposure-matched natural | +0.000075, +0.000018, +0.000064, −0.000021 | 3/4 | +0.000034 | [−0.000027, +0.000095] |
| G_SEM (worse if > 0) | balanced global forced | +0.00120, +0.00133, +0.00464, +0.00143 | 4/4 worse | **+0.00215** | [+0.00125, +0.00382] |
| G_SHUF | balanced global forced | +0.00018, +0.00015, +0.00043, +0.00018 | 4/4 worse | +0.00024 | [+0.00016, +0.00037] |
| Sim_SEM (worse if > 0) | ordinary natural, all pairs at once | +0.576, +0.746, +1.620, +0.730 | 4/4 worse | **+0.918** | [+0.620, +1.397] |
| Sim_SHUF | ordinary natural, all pairs at once | +0.029, +0.023, +0.057, +0.030 | 4/4 worse | +0.035 | [+0.025, +0.049] |

- **Baseline.** Held-out B natural CE is 3.072–3.081 nats per seed.
- **Prediction.** The predicted in-sample first-order gain η‖u‖² has a mean of 0.00284 (SEM). The realised F/predicted ratio is 0.60, 0.58, 1.30, 0.66 per seed.
- **Stability (split-half A1 vs A2).** Split-half cosine of u, SEM vs SHUF per seed:

| seed | s0 | s1 | s2 | s3 |
|---|---|---|---|---|
| SEM | 0.128 | 0.137 | 0.141 | 0.146 |
| SHUF | 0.010 | 0.008 | 0.014 | 0.006 |

- **By layer.** The SEM split-half cosine of u rises from 0.08–0.10 in layers 0–5 to **0.41 in layer 10**.
- **Norms.** ‖d‖/‖g‖ ≈ 2.2–2.4 (SEM) vs 2.1 (SHUF). ‖u‖/‖d‖ ≈ 0.98. Mean ‖d_SEM‖/‖d_SHUF‖ = 1.19.

### 4.1 Preregistered criteria (§5.4; evaluated once)

| id | result | evidence |
|---|---|---|
| V0 matching | **FAIL** | Median over pairs of max-variable \|SMD\| is 0.112 / 0.115 / 0.112 / 0.118 on A1 (threshold ≤ 0.10); A2 is 0.098–0.103. Matched positive-weight fraction is 1.0 on A1, A2 and B. The residual imbalance is in log doc length (median \|SMD\| ≈ 0.09) and router rank (≈ 0.07). |
| V1 direction | pass | mean F_SEM = +0.00229 |
| V2 interval | pass | hier. CI [+0.00155, +0.00358] |
| V3 seeds | pass | 4/4 |
| V4 global CE | **FAIL** | (a) G_SEM / F_SEM = 0.94 (needs ≤ 0.5); (b) Sim_SEM = +0.918 nats (needs ≤ +0.001) |
| V5 exposure | pass | N_SEM > 0 in 4/4 seeds, mean +0.00468 |
| V6 concentration | **FAIL** | Layer 10 carries 81 % of ΣF and cluster k* = 14 carries 58 % (each needs ≤ 50 %). The top 5 % of pairs carry 71 % (needs ≤ 50 %). Leave-one-layer-out and leave-one-cluster-out means all stay > 0 (part (a) holds). |
| V7 semantic > shuffled | pass | (a) Δ > 0 in 4/4 seeds, CI excludes 0; (b) split-half cos(u) SEM > SHUF in 4/4 seeds |

**Failure modes (§5.5) that apply:**
- **Global-CE conflict:** V4.
- **Matching failure:** V0.
- **Layer/cluster concentration:** V6.

**Failure modes that do not apply:**
- Unstable subject-differential gradients: mean cos_u12 SEM = 0.138 ≥ 0.10.
- No difference from shuffled: V7 holds.
- Insufficient semantic information: lift 1.67 ≥ 1.5 and ‖d‖ ratio 1.19 > 1.1.
- First-order prediction fails held-out: overall realised/predicted = 0.81 ≥ 0.25.

## 5. Interpretation (evidence vs hypotheses)

### Evidence (from preregistered quantities)

1. **The semantic residual is a real, reproducible signal, and it is not generic.**
   - F_SHUF and N_SHUF are about zero.
   - Split-half stability of u is about 10× higher for SEM than SHUF in every seed.
   - Label-free semantic clusters therefore carry information about *which direction in an expert's weights helps one subject relative to matched controls*. The shuffled grouping carries none.
2. **That direction is not subject-specific competence.**
   - Moving along it costs global CE almost one-for-one with the contrast gain (G ≈ F).
   - Together, the updates are catastrophic for ordinary natural CE (Sim ≈ +0.9 nats).
   - Projecting out the balanced global gradient guarantees only *first-order orthogonality to the global average*. It does not prevent the update from shifting loss from other subjects onto the assigned one.
3. **The signal is concentrated.** The bulk of F, N, G and Sim comes from the last MoE layer (block 11) and one cluster (k* = 14). The cluster holds 5.3 % of the paper mass and is not unusually large.

### Post-hoc descriptive findings (`diagnostics_posthoc.json`; cannot change the verdict)

- **Sim is real, not a bug.** The sum over pairs of the per-pair natural all-token CE change is 0.37 / 0.46 / 1.03 / 0.51 nats against Sim of 0.58 / 0.75 / 1.62 / 0.73. They are the same order and mildly superadditive. Layer 10 carries 97 % of it.
- **Layer 10.**
  - Under forced insertion, the assigned cluster's CE change is **+0.0116 (worse)** and the other clusters' is **+0.0235 (worse)**. Over 99 % of (pair, other-cluster) cells get worse.
  - The large F and N in layer 10 (F/pred ≈ 1.9) are therefore *differential damage*, not benefit.
  - Natural all-token damage is +0.018 nats per updated expert.
  - The median relative step in layer 10 was 10.6 %, twice the global median. This is because η is global and layer-10 ‖u‖ is larger.
- **Layers 0–9.**
  - The assigned cluster improves slightly (forced change −0.00002 to −0.00016) while the other clusters worsen by about the same amount (+0.00006 to +0.00018). 70–83 % of other-cluster cells get worse.
  - F is +0.0003 to +0.0009 per pair and G is about +0.0001.
  - Realised/predicted is 0.16–0.39, so first-order prediction largely fails outside layer 10. This is consistent with only about 10 % of ‖u‖ being reproducible signal (split-half cos about 0.1).
  - The summed natural damage of layers 0–9 alone (0.007–0.019 nats) would still exceed the +0.001 Sim threshold, assuming additivity.
  - So even a post-hoc "drop layer 10" variant shows a small zero-sum reallocation, not competence. *This variant was not preregistered and is not a claim.*

### Hypotheses (not tested)

- **H1.** In the last MoE layer, the subject residual mainly moves expert output along directions that shift the next-token distribution toward the subject's vocabulary. That is a topical prior, which is zero-sum across subjects by construction.
- **H2.** With a smaller per-layer step, or with the LM gradient present (Stage 2 condition C), the damage could be absorbed. Stage 1 cannot test this. The prereg forbids proceeding to test it after a Stage-1 failure.

## 6. Routing vs functional

All Stage-1 quantities are **functional**: CE changes after a parameter change, with routing held fixed (forced) or natural.

**Routing was not trained or changed.** Natural top-4 routing was used only for the k* association and for the N strata. Routing exposure does not explain F (F is under forced, equal exposure) or N (N is exposure-matched). That is the sense in which V5 passed. The failure is functional: the subject gain is paid for by other subjects.

## 7. Limitations

- **One step, frozen, first order.** A single virtual step cannot show what continued training with g_LM + λΣ A·u would do. The protocol made Stage 1 a mandatory gate for exactly that reason.
- **Step size.** η was set by a global median rule and was cap-binding. This made layer-10 steps about 2× larger relative to the weights than elsewhere. A per-layer step rule was not preregistered and was not searched, as the protocol forbids.
- **Matching.** Document length was not a stratification variable, only a balance check. V0 failed narrowly on A1 because of it. V4 and V6 fail independently, so this does not affect the verdict (deviation 1 in `deviations.jsonl`).
- **Data.** CALIB-SEARCH only (1.3M tokens over 544 papers in A1 ∪ A2 ∪ B). Semantic clusters come from an earlier label-free table. Cluster identities were not inspected against real domain labels, by policy.
- **Numerics.** bf16 base model, with the probed expert in fp64. Per-token noise is high in bf16, but the means agreed within 1–6 % in Stage-0 numerics.

## 8. Exact reason for stopping

By prereg §5.4, Stage 1 passes only if V0–V7 all hold. **V0, V4 and V6 failed**, so prereg §5.5 and the protocol stop the experiment. No thresholds were changed, no alternative step sizes or layer subsets were tried, and no training was started.

## 9. Compute, disk, integrity

**GPU.** 5.433 GPU-h in total, against a cap of 30.
- Stage 0: 0.057 GPU-h.
- Stage 1: 5.376 GPU-h, which is 8 jobs at about 0.81 GPU-h per estimation job and 0.54 GPU-h per evaluation job.
- There were no external interruptions and no foreign GPU processes.

**Disk.**
- New data: 2.3 GB in `artifacts/scrgp-v1`. Temporary slots were removed on completion; `.tmp/scrgp-v1` holds 44 KB.
- Free space stayed at ≥ 147.9 GiB throughout (reserve 100 GiB). /dev/shm was not used.
- Nothing was deleted except this experiment's own resume slots.

**Integrity.**
- Protected inventory: 9,837 files, 188 GB, verified unchanged after Stage 1 (`protected_inventory_before.verify_20261120T060247Z.json`).
- All parameter-hash restores were verified.
- The test split was never read, VAL was never read, and real domain labels were never used.

**Deviations** (`artifacts/scrgp-v1/deviations.jsonl`, 2 entries, neither can affect the conclusion):
- (0) Post-hoc diagnostics were added after the verdict.
- (1) Document length was missing from the matching strata.

Engineering iterations before Stage 1 are in `decision_log.jsonl`:
- T-GRAD finite-difference ε;
- the ranks_of argsort fix;
- the bit-exact expert-slot replica fixing G-OFF;
- `.cpu()` on slot tensors;
- the orchestrator glob.

## 10. Paths

- **Report and notes:** `SCRGP_V1_REPORT.md`, `SCRGP_V1_DIAGNOSTIC_NOTES.md`.
- **Prereg:** `docs/SCRGP_V1_PREREG.md`.
- **`artifacts/scrgp-v1/`:**
  - verdict and tables: `final_analysis.json`, `stage1_tables.md`, `diagnostics_posthoc.json`;
  - ledgers and logs: `run_manifest.tsv`, `compute_ledger.tsv`, `disk_ledger.tsv`, `decision_log.jsonl`, `observations.jsonl`, `deviations.jsonl`;
  - state: `RESUME.md`, `STATE.json` (`TERMINAL_STAGE1_FAIL`);
  - per-seed outputs: `stage1/{est,eval}_s{0..3}.*`;
  - tables: `tables/`.
- **Code:**
  - `src/csr/scrgp_{paths,lib,log}.py`;
  - `scrgp_{inventory,tables,reuse,bench,numerics,tests,gpu_tests,stage1,analyze,run,driver,diag}.py`.
