# csr-semantic-utility-edl-v1 — final report

**Status:** TERMINAL (2026-11-17 18:10 EST). The preregistered CALIB-SELECTION confirmation was NOT passed, so by
rule no 245M finals were trained.

**Conclusion: no replicated benefit.** The hard-margin counterfactual-usefulness objective did not make semantic
groups functionally more specialized than shuffled persistent groups on held-out documents.

**Sources.**

- Preregistration: `docs/SEMANTIC_UTILITY_EDL_V1_PREREG.md` (sha256 6b12ebd5…).
- Decisions: `artifacts/semantic-utility-edl-v1/{search_decision,frozen,confsel}.json`, all 0400.
- Generation document: `docs/SUE_GEN1.md` (sha 602700ca…).
- Descriptive tables: `artifacts/semantic-utility-edl-v1/report_tables.{json,md}`.
- Analysis: `final_analysis.json`, `SPECIALIZATION_TRAJECTORY.json`, `run_manifest.tsv`.

The report separates (1) preregistered results, (2) exploratory results, (3) post-hoc diagnostics, (4) measured facts
vs (5) interpretations, and (6) remaining hypotheses.

## 0. Method in one paragraph

- **Model and recipe.** The unchanged 103.6M LFR model with dropless top-4 routing over 32 experts and the shared
  expert. Identical recipe to TOKEN-4 and EDL.
- **Grouping.** Label-free semantic paper groups: K = 16 C6, the preregistered fallback, because semantic-edl-long-k's
  K8 was never confirmed. The shuffled control is the persistent paper permutation of the same table.
- **Objective.** L = LM + 0.01 Switch + 0.001 z + 0.005 Semantic-EDL + γ L_use.
- **Usefulness targets.** Every step, on held-out main-split documents not in the training plan, exact replacement
  passes (remove one selected expert, admit the next-ranked, recompute downstream) estimate
  u = CE_replace − CE_base. These form ΔU, the own-group minus difficulty-matched other-group usefulness. The
  targets are lagged, detached, shrunk and standardised.
- **L_use.** Asks each group's mean routing to score highest against its own ΔU column, a hard group-classification
  margin.
- **Selection measure.** Pseudo-label per-expert replacement interaction on paper-disjoint CALIB-SEARCH half B, with
  discovery on half A.
- **Held-out confirmation.** CALIB-SELECTION half 0, opened once.

## 1. Preregistered results (measured facts)

### 1.1 Reliability diagnostic

Recorded only, never a stopping rule. 1,000 probe steps on frozen checkpoints:

| checkpoint | SEM split-half r | SEM Spearman-Brown | SHUF split-half r |
|---|---|---|---|
| anchor | 0.17 | 0.29 | ≈ 0 |
| C6 50M | 0.20 | 0.34 | ≈ 0 |
| SHUF 50M | 0.26 | 0.41 | ≈ 0 |

- The own-group margin was positive in both document halves for SEM and ≈ 0 for SHUF.
- γ anchor rule: γ_base = 0.01 (raw 0.0137).
- Smoke run: 30 GPU steps without error. Its summary overhead metric was invalid (DEVIATIONS 1 context; decision log).

### 1.2 Search (20M tokens, seed 100; reference REF-C6 = Semantic-EDL only, same seed)

| cand | change | dev CE | ΔCE vs TOKEN-4 | Δ interaction vs REF-C6 | interaction [95% CI] | specialised cells (of 352) | pseudo MI | L_use active | bank SB (last 5) | eligible |
|---|---|---|---|---|---|---|---|---|---|---|
| U1 | base γ 0.01, τ 0.2 | 4.5987 | −0.0275 | −6.3e-5 | 2.13e-4 [1.57, 2.82] | 46 (0.13) | 0.184 | 0.6 | 0.52 | no |
| **U2** | γ × 0.3 | 4.6096 | −0.0166 | **+3.7e-5** | 3.13e-4 [2.31, 3.76] | 54 (0.15) | 0.189 | 0.6 | −0.11 | **yes** |
| U3 | γ × 3 | 4.6117 | −0.0146 | −2.7e-5 | 2.49e-4 [1.27, 3.16] | 58 (0.16) | 0.211 | 0.6 | 0.47 | no |
| U4 | τ 0.05 | 4.6057 | −0.0205 | −3.1e-5 | 2.45e-4 [1.85, 3.45] | 54 (0.15) | 0.190 | 0.6 | 0.11 | no (load 0.01×) |
| U5 | τ 1.0 | 4.6132 | −0.0131 | −4.0e-5 | 2.36e-4 [1.67, 3.29] | 59 (0.17) | 0.188 | 0.6 | 0.32 | no |
| U6 | ρ 0.98 | 4.6079 | −0.0183 | −0.8e-5 | 2.68e-4 [2.04, 3.30] | 62 (0.18) | 0.183 | **0.0** | — | no |
| U7 | onset 30 | 4.6082 | −0.0180 | −1.5e-5 | 2.61e-4 [2.02, 3.88] | 56 (0.16) | 0.179 | 0.5 | 0.24 | no |
| U8 | layers 5–10 | 4.6119 | −0.0143 | −1.4e-5 | 2.62e-4 [1.45, 3.33] | 59 (0.17) | 0.191 | 1.0 | 0.41 | no |

- **References.** REF-C6 s100 at 20M: CE 4.6119, interaction 2.76e-4 [2.22, 3.45]. TOKEN-4 s100 at 20M
  (descriptive, §2): CE 4.6262.
- **Usefulness onset.** The term first became active at about step 200 (12.7M tokens): onset 100 plus the time
  for the bank to reach coverage. It therefore influenced only about **7.4M of the 20M tokens**. U8 (fewer layers)
  activated at step 100. U6's short EMA starved coverage (0.01), so its term never activated.
- **The 20M screen was weak.** Every Δ vs REF-C6 lies within ±6e-5, well inside the per-run CI width (about
  1.2e-4).

### 1.3 Promotion (50M, seeds 100/101, SEM vs SHUF with the identical objective)

| candidate | seed | Δ interaction SEM − SHUF (dev) | ΔCE SEM − SHUF | rule |
|---|---|---|---|---|
| U2 | 100 | −2.6e-5 | −0.0116 | fail |
| U2 | 101 | −1.19e-4 | +0.0040 | fail |
| G1a (GEN1) | 100 | +4.6e-5 | +0.0001 | pass |
| G1a (GEN1) | 101 | +1.00e-4 | +0.0057 | pass (mean ΔCE +0.0029 ≤ 0.005) |

- **GEN1.** The preregistered diagnosis after U2 was **"noisy"**: bank split-half r at 50M was −0.08 for s100.
  This triggered generation 1, a support/EMA change of 8 probe passes and ρ 0.998, with γ 0.003 and τ 0.2
  unchanged. Its document `docs/SUE_GEN1.md` was written and hashed before running.
- **Frozen candidate:** G1a.

### 1.4 CALIB-SELECTION confirmation (paper-disjoint held-out half 0, opened once; G1a at 50M)

| seed | SEM interaction | SHUF interaction | Δ | ΔCE | pseudo MI SEM / SHUF |
|---|---|---|---|---|---|
| 100 | 4.29e-4 | 4.67e-4 | **−3.8e-5** | −0.0020 | 0.181 / 0.158 |
| 101 | 4.38e-4 | 4.87e-4 | **−4.9e-5** | +0.0060 | 0.173 / 0.149 |

- Pooled paired document-bootstrap 95% CI of the seed-mean difference: **[−8.7e-5, +0.9e-5]**.
- **Confirmed: False.** The dev advantage reversed sign in both seeds on held-out documents.
- Finals were not trained, by rule. The compute-matched TOKEN-4 control was therefore never triggered.

## 2. Exploratory / descriptive results (never used for any decision)

Reused TOKEN-4 and explicit-EDL checkpoints were evaluated with the identical dev evaluator after the experiment was
terminal (`sue_post.py`). All runs are on the same plan and the same CALIB-SEARCH half B rows; CE is in nats/token.

| run (50M) | seed | dev CE | ΔCE vs TOKEN-4 | ΔCE vs EDL | ΔCE vs C6 | ΔCE vs SHUF-C6 | interaction [95% CI] | spec | pseudo MI | entropy |
|---|---|---|---|---|---|---|---|---|---|---|
| TOKEN-4 | 100 | 3.7609 | — | | | | 2.94e-4 [2.27, 3.82] | 35 | 0.036 | 0.811 |
| EDL (explicit labels) | 100 | 3.7504 | −0.0105 | — | | | 3.55e-4 [2.49, 4.48] | 69 | 0.171 | 0.738 |
| C6 Semantic-EDL | 100 | 3.7436 | −0.0173 | −0.0068 | — | | 3.29e-4 [2.36, 4.25] | 62 | 0.160 | 0.732 |
| SHUF-C6 | 100 | 3.7386 | −0.0223 | −0.0118 | | — | 3.13e-4 [2.20, 3.98] | 55 | 0.144 | 0.727 |
| SEM-G1a | 100 | 3.7335 | −0.0274 | −0.0168 | −0.0101 | −0.0050 | 3.80e-4 [3.02, 4.76] | 67 | 0.156 | 0.724 |
| SHUF-G1a | 100 | 3.7334 | −0.0275 | −0.0170 | −0.0102 | −0.0052 | 3.34e-4 [2.29, 4.31] | 67 | 0.137 | 0.712 |
| SEM-U2 | 100 | 3.7341 | −0.0268 | −0.0163 | −0.0095 | −0.0045 | 3.23e-4 [2.35, 4.06] | 63 | 0.158 | 0.728 |
| SHUF-U2 | 100 | 3.7457 | −0.0152 | −0.0046 | +0.0021 | +0.0071 | 3.49e-4 [2.66, 4.66] | 57 | 0.135 | 0.722 |
| TOKEN-4 | 101 | 3.7470 | — | | | | 1.59e-4 [0.86, 3.07] | 27 | 0.034 | 0.799 |
| EDL | 101 | 3.7524 | +0.0054 | — | | | 3.48e-4 [1.83, 4.47] | 47 | 0.160 | 0.734 |
| C6 | 101 | 3.7346 | −0.0124 | −0.0178 | — | | 3.12e-4 [2.02, 3.90] | 62 | 0.147 | 0.722 |
| SHUF-C6 | 101 | 3.7360 | −0.0110 | −0.0164 | | — | 3.86e-4 [2.86, 5.28] | 59 | 0.132 | 0.712 |
| SEM-G1a | 101 | 3.7409 | −0.0060 | −0.0115 | +0.0064 | +0.0049 | 3.84e-4 [2.89, 4.70] | 62 | 0.157 | 0.721 |
| SHUF-G1a | 101 | 3.7352 | −0.0117 | −0.0172 | +0.0006 | −0.0008 | 2.83e-4 [0.99, 3.88] | 59 | 0.131 | 0.713 |
| SEM-U2 | 101 | 3.7416 | −0.0053 | −0.0108 | +0.0071 | +0.0056 | 2.68e-4 [0.70, 3.53] | 55 | 0.158 | 0.728 |
| SHUF-U2 | 101 | 3.7376 | −0.0093 | −0.0148 | +0.0031 | +0.0016 | 3.87e-4 [1.76, 4.80] | 65 | 0.131 | 0.710 |

**Measured facts from this table:**

- **CE (a).** Every grouped-routing condition has lower dev CE than TOKEN-4 at 50M in both seeds. That covers C6,
  SHUF-C6, both utility variants and their SHUF twins: −0.005 to −0.028 nats. They are also generally lower than
  explicit EDL.
- **CE (b).** The shuffled grouping gets the same CE benefit, so the CE gain is not semantic.
- **Routing (c).** Semantic grouping raises pseudo-label routing MI over the shuffled twin in every pair
  (+0.019 to +0.027).
- **Function (d).** Semantic grouping does not raise functional interaction consistently. The SEM − SHUF sign
  flips between seeds and conditions, and every per-run CI overlaps its twin.

**Bank reliability.** Spearman-Brown, mean of the last 5 refreshes at 50M:

| run | s100 | s101 |
|---|---|---|
| SEM-G1a | 0.27 | 0.36 |
| SHUF-G1a | 0.01 | 0.03 |
| SEM-U2 | −0.01 | 0.43 |
| SHUF-U2 | 0.14 | 0.06 |

- The usefulness signal of semantic groups is measurable; that of shuffled groups is not.
- The sign of the split-half r fluctuated between refreshes in every run.

**Not available.**

- Real-domain NMI and VAL metrics: finals were not reached, and real labels were never opened in V1.
- Pseudo-label NMI: not computed by the audit (DEVIATIONS 2). MI, topic entropy and topic-prediction accuracy are
  reported instead (`report_tables.md`).

## 3. Post-hoc diagnostics (descriptive)

- **Activation.** At 20M the term was active for about 7.4M tokens (first activation ≈ step 200).
  - At 50M, U2 influenced about 37M tokens and G1a about 44M; G1a activated earlier thanks to 8 passes.
- **Probe overhead.**

  | setting | added step time |
  |---|---|
  | 4 passes | 0.49 |
  | 8 passes | 0.81 |
  | U8 (6 layers) | 0.40 |

- **Throughput** (training tokens/s at 50M):

  | run | tokens/s |
  |---|---|
  | REF-C6 / REF-SHUF (no probing) | about 123k |
  | U2 (4 passes) | about 78k |
  | G1a (8 passes) | about 63k |
- **GEN1's dev gain did not transfer.** It was positive in both dev seeds and negative in both held-out seeds.
  - The held-out CI includes 0 and is centred below it.
  - The dev effect (+7e-5) is of the size of the between-run seed spread of interaction (REF-SHUF s100 vs s101 differ
    by 7.3e-5).
  - This is consistent with selection on noise: one generation was chosen from a failed candidate on the same dev
    split.

## 4. Interpretation (not measured facts)

- **Measurable but noisy.** The counterfactual signal of semantic groups is measurable. Its in-training
  reliability, however, was only about 0.3 with 8 passes and stayed near 0 for many refreshes with 4 passes. The
  hard-margin objective then mostly moved group routing (higher MI) without producing experts that are more
  useful for their own semantic group on held-out documents than shuffled groups.
- **What the CE gain means.** The CE gain over TOKEN-4 and EDL at 50M is real at these seeds. It comes from
  persistent grouping pressure in general, because SHUF matches it, not from semantics or usefulness.
- **Scope.** This does not show that functional specialization is impossible. It shows that this objective, at this
  estimator reliability and 50M tokens, did not produce a semantic functional advantage that survives held-out
  confirmation.

## 5. Remaining hypotheses

- A confidence-aware, soft target may avoid amplifying noise: lower/upper confidence bounds instead of shrunk point
  estimates, and a KL to a soft distribution instead of a hard margin. The operator authorised this as V2:
  `docs/SOFT_CF_USEFULNESS_V2_PREREG.md`.
- Longer training (245M) may let routing preferences turn into expert function. This is untested: finals were not
  reached by rule.
- Higher-reliability measurement (more passes or rows) may help, at a cost: the overhead is already 0.8× at
  8 passes.

## 6. Compute, storage, deviations, integrity

**GPU time: 7.52 GPU-h.**

| phase | GPU-h |
|---|---|
| diagnostic + smoke | 0.22 |
| search + promotion + generation + confirmation | 6.52 |
| post-terminal descriptive evaluations | 0.78 |

The cap was 45 GPU-h.

**Disk.**

- 20 own run directories: 6.0 GiB.
- Retained resumable slots: 23 GB in `.tmp/semantic-utility-edl-v1/slots`. The operator's no-deletion rule of
  2026-11-17 applies.
- Free disk at the end: 135.4 GiB.

**Deviations** (`DEVIATIONS.jsonl`):

1. The artifact directory and test ledger rows were created by CPU tests before the freeze.
2. Orchestration crash on a missing pseudo-NMI key; fixed with `.get`, a reporting-only change.
3. Operator rules: final slots are no longer deleted, RESUME.md is written after every job, and storage is estimated
   per stage. In-run rotation of superseded resume slots is kept (interpretation recorded).
4. Duplicate RESULTS_LIVE rows from relaunches; cosmetic, and fixed.
5. Evaluator crash in CALIB-SELECTION mode before any measurement: the e5 windows cover only 1,010 documents. The
   preregistered pseudo measurement is unchanged; the descriptive e5 measurement now runs on the e5-labelled
   subset.

**Operational stop.** A disk-reserve stop occurred on 2026-11-17 at 03:28. The operator freed disk, and the run
resumed at 12:13 (decision log).

**Integrity.**

- Prereg hash checked by every script.
- Protected inventory: 7,673 files, 0 mismatches (verified 2026-11-17 19:00).
- semantic-edl-long-k frozen manifest: 0 mismatches.
- CALIB-SELECTION half 0 was measured once. The crashed attempt measured nothing.
- The locked test split was never accessed. No real domain labels were used.
- No foreign GPU process interrupted any job.
