# csr-semantic-edl-v1 (Semantic EDL) — Report

**Terminal status: FAILED — NOT CONFIRMED AT CALIB-SELECTION. No finals were trained and VALIDATION was never opened.**

**Outcome against explicit EDL: neither "beat" nor "matched" was established. Under the preregistered protocol
the outcome is FAILED.**

- The frozen candidate (C4) passed Stage 2 on CALIB-SEARCH. It then failed the single preregistered CALIB-SELECTION
  confirmation, because in seed 100 its pseudo-domain FSI fell below its own shuffled-cluster control
  (0.00473 vs 0.00526).
- Post-terminal descriptive analysis on CALIB-SEARCH with real domain labels (deviation 4; no decision used it)
  shows a consistent ordering on every real-domain specialisation measure, in both seeds:
  explicit EDL > semantic EDL > shuffled pseudo-domains > TOKEN-4.
  - Semantic EDL reaches about 89–95 % of explicit EDL's domain FSI.
  - A shuffled, paper-consistent grouping alone reaches about 80 %.
  - Most of what an EDL-type loss does at 50M tokens is therefore generic: it comes from the paper-level grouping
    structure, not from semantic content.
  - The semantic increment is real in direction but small, and two seeds at 50M cannot separate it reliably.

Prereg `docs/SEMANTIC_EDL_V1_PREREG.md` (sha256 `63e03933…0cba`, unchanged; verified at the end). GPU 7.16 of 24 GPU-h.
Protected inventory: 6,498 / 6,498 files unchanged. Test was never loaded. VAL was never loaded. Real domain labels
were read only after the terminal decision.

## 1. What was tested

**Method.** Explicit-EDL population-divergence loss with each document's one-hot domain label replaced by q_d.
- q_d is the frozen, label-free paper-level SEM K = 16 cluster distribution from GC-EDL-v1.
- Everything else is unchanged: unrestricted top-4 routing, the shared expert, Switch balancing and z-loss.
- There are no pools, masks, inference-time encoders, labels or parameters, and no extra data.
- Within each seed, the architecture, initialisation, data order, token budget, optimiser, schedule, precision and
  inference cost are identical to TOKEN-4 and explicit EDL. CPU tests and GPU equivalence checks are in
  `cpu_tests_*.json` and `gpu_checks.json`.
  - TOKEN-4 vs semantic-EDL at β = 0 is bit-identical at step 0.
  - The resume difference is 1.2e-4, against a noise level of 4.3e-4.

**Strength scale.** EDL-relative: "2 %" means the explicit-EDL aux/LM gradient ratio in the same window. This was
calibrated with 160-step pilots (`search/calibration.json`). Every candidate's achieved ratio was 0.99–1.16 × its
target, so selection could not reduce the loss to an inert one.

## 2. Funnel and decisions (all on CALIB-SEARCH; label-free metrics only)

**Stage 1** (20M tokens, seed 100). Baselines at s20: TOKEN-4 CE 4.6469 and FSI_pseudo 0.00139; explicit EDL CE
4.6469 and FSI_pseudo 0.00338.

| id | placement | q | target | ramp | β | CE s20 | ΔCE vs EDL | FSI_pseudo adv. vs T4 | FSI_e5 | stage-1 result |
|---|---|---|---|---|---|---|---|---|---|---|
| C1 | all | soft | 2 % | – | 0.0055 | 4.6296 | −0.0173 | 0.00171 | 0.0072 | survived, not top-2 |
| C2 | all | soft | 1 % | – | 0.0027 | 4.6357 | −0.0112 | 0.00106 | 0.0050 | survived |
| C3 | all | soft | 0.5 % | – | 0.0014 | 4.6491 | +0.0022 | 0.00119 | 0.0055 | survived |
| **C4** | upper (MoE 5–10) | soft | 2 % | – | 0.0055 | 4.6441 | −0.0028 | **0.00222** | 0.0072 | **advanced** |
| C5 | upper | soft | 1 % | – | 0.0027 | 4.6326 | −0.0143 | 0.00114 | 0.0053 | survived |
| C6 | all | sharp (T = 0.5) | 2 % | – | 0.0050 | **4.6264** | −0.0205 | 0.00175 | 0.0064 | survived, not top-2 |
| **C7** | all | soft | 2 % | 2 % | 0.0055 | 4.6490 | +0.0022 | 0.00176 | 0.0068 | **advanced** |
| C8 | all | sharp | 1 % | – | 0.0025 | 4.6388 | −0.0081 | 0.00159 | 0.0054 | survived |

- No candidate was eliminated.
- C7 beat C6 for the second slot by 1e-5 in FSI_pseudo advantage. That is within noise. The ranking was applied as
  preregistered.

**Stage 2** (50M tokens, seeds 100 and 101, with SHUF control). Reference: explicit EDL CE 3.7610 / 3.7628 and
FSI_pseudo 0.00535 / 0.00631.

| run | CE s50 (s100 / s101) | FSI_pseudo s50 | FSI_e5 s50 | verdict |
|---|---|---|---|---|
| TOKEN-4 | 3.7720 / 3.7579 | 0.00263 / 0.00252 | 0.0057 / 0.0054 | – |
| C4 | 3.7625 / 3.7504 (mean Δ vs EDL −0.0055) | 0.00566 / 0.00566 | 0.0107 / 0.0100 | **advanced (i–v all pass)** |
| SHUF_C4 | 3.7597 / 3.7720 | 0.00545 / 0.00512 | 0.0098 / 0.0105 | control |
| C7 | 3.7840 / 3.7559 (mean Δ vs EDL +0.0081) | 0.00633 / 0.00511 | 0.0108 / 0.0098 | failed (i) CE |

C7 also has no SHUF control (criterion (i) failed), so (ii) could not pass either.

**Engineering generations: none were run.** The decision (`search/generation_decision.json`, written before
CALIB-SELECTION) was:
- a candidate had advanced;
- prereg §8–9 ties each generation to a measured failure mode, and none existed;
- further search after a pass would select on noise.

The small FSI margin of C4 over SHUF was recorded as a risk at that point. In hindsight, a generation aimed at that
margin was the one open lever. But the measured margins (§5) are within seed noise, and no allowed factor was
expected to fix that.

**Freeze and CALIB-SELECTION** (opened once; `calibsel_decision.json`):

| s50, CALIB-SELECTION | explicit EDL | C4 | SHUF_C4 | TOKEN-4 |
|---|---|---|---|---|
| CE s100 / s101 | 3.7711 / 3.7730 | 3.7718 / 3.7610 | 3.7691 / 3.7812 | 3.7824 / 3.7684 |
| FSI_pseudo | 0.00500 / 0.00490 | **0.00473** / 0.00502 | **0.00526** / 0.00456 | 0.00269 / 0.00268 |
| FSI_e5 | 0.0100 / 0.0099 | 0.0102 / 0.0101 | 0.0097 / 0.0108 | 0.0060 / 0.0051 |
| MI_pseudo | 0.211 / 0.199 | 0.158 / 0.166 | 0.125 / 0.134 | 0.045 / 0.044 |

Result:
- mean ΔCE vs EDL = −0.0056 (passes);
- no pathology (passes);
- FSI_pseudo > SHUF fails in seed 100.

The outcome is **not confirmed**, and the experiment ends with the 50M evidence (prereg §10).

## 3. CE by seed, with uncertainty

There is no VAL, so the evidence is at 50M tokens on the calibration splits.

| split | sem − EDL (s100 / s101; mean) | sem − T4 | SHUF − EDL |
|---|---|---|---|
| CALIB-SEARCH half B | +0.0015 / −0.0125; −0.0055 | −0.0095 / −0.0076 | −0.0012 / +0.0091 |
| CALIB-SELECTION | +0.0007 / −0.0120; −0.0056 | −0.0106 / −0.0074 | −0.0020 / +0.0082 |

**Uncertainty.**
- Two seeds per arm.
- Seed-to-seed spread for an identical recipe is 0.014 nats (TOKEN-4 s100 vs s101 on CALIB-SEARCH) and 0.002 for
  explicit EDL.
- Paired sem − EDL differences have opposite magnitudes across seeds (≈ 0 and −0.012).
- **The CE result is "no worse than explicit EDL" at 50M, not "better".**
- The seed-101 CE gain is also matched in sign by SHUF's seed-100 gain, so it is not attributable to semantics.

## 4. Specialisation by seed and layer (post-terminal descriptive; CALIB-SEARCH discovery half A / measurement half B; 50M)

| measure (s100 / s101) | explicit EDL | semantic EDL (C4) | SHUF_C4 | TOKEN-4 |
|---|---|---|---|---|
| domain FSI | 0.00909 / 0.00951 | 0.00861 / 0.00844 | 0.00735 / 0.00764 | 0.00405 / 0.00432 |
| domain FSI as fraction of EDL | 1 | 0.95 / 0.89 | 0.81 / 0.80 | 0.45 / 0.45 |
| routing–domain MI (nats) | 0.136 / 0.128 | 0.099 / 0.105 | 0.076 / 0.081 | 0.028 / 0.028 |
| domain topic-prediction accuracy | 0.558 / 0.554 | 0.561 / 0.570 | 0.547 / 0.552 | 0.518 / 0.515 |

**Per-expert functional ablation** (352 equal-compute single-expert replacements, fle protocol; PE 50k tokens per
domain on CALIB-SEARCH half B). Values are mean own-group − difficulty-matched other-group ΔCE, in units of 1e-4 nats
(s100 / s101).

| system | explicit EDL | semantic EDL | SHUF | TOKEN-4 |
|---|---|---|---|---|
| **real domain (primary)** | 2.19 / 1.90 | 1.91 / 2.17 | 1.62 / 2.04 | 1.37 / 1.16 |
| fraction of experts positive (domain) | 0.65 / 0.66 | 0.64 / 0.65 | 0.61 / 0.64 | 0.58 / 0.63 |
| pseudo-domain (SEM) | 4.49 / 4.36 | 4.03 / 4.23 | 4.43 / 4.20 | 2.13 / 2.49 |
| e5 cluster | 7.12 / 6.67 | 6.35 / 7.00 | 6.74 / 7.17 | 5.36 / 5.04 |
| generic importance (mean ΔCE, 1e-4) | 5.3 / 5.3 | 5.3 / 5.7 | 5.2 / 5.8 | 5.5 / 5.4 |

Semantic EDL's domain interaction relative to EDL is 0.87 / 1.14, above TOKEN-4 and SHUF in both seeds. But
SHUF − TOKEN-4 is already +0.25 / +0.88, and the EDL vs SHUF gap is +0.57 / −0.14. So **per-expert interactions at
50M do not separate explicit EDL from a shuffled grouping either**. For scale, the fle VAL finals at 245M tokens had
EDL 6.9 / 6.6 vs TOKEN-4 4.0 / 3.9 / 4.2.

**By layer** (domain interaction, 1e-4; `final_analysis.json` → `per_expert.*.domain.by_layer`).
- Semantic EDL (upper placement) concentrates its interaction in MoE 5–10: 3.1 / 1.7 / 1.2 / 4.8 / 4.7 / 3.6 in s100,
  against 0.2–0.7 in MoE 0–4. Explicit EDL is spread over all layers.
- The upper placement therefore did move function where the loss was applied. Caveat: TOKEN-4 s100 also peaks in
  MoE 8–10 (2.8 / 4.2 / 3.6), so the contrast is mainly the near-zero lower layers.

**Trajectories** (`specialization_trajectories.json`; FSI_pseudo s5 → s20 → s50 for seed 100):

| run | s5 | s20 | s50 |
|---|---|---|---|
| TOKEN-4 | 0.00055 | 0.00139 | 0.00263 |
| EDL | 0.00039 | 0.00338 | 0.00535 |
| C4 | 0.00060 | 0.00361 | 0.00566 |
| C7 | 0.00085 | 0.00315 | 0.00633 |

Domain FSI s20 → s50: EDL 0.0068 → 0.0091; C4 0.0060 → 0.0086. No candidate lost its early advantage.

## 5. Comparisons with TOKEN-4 and shuffled pseudo-domains

**Relative to TOKEN-4.** Semantic EDL is better on every specialisation measure in both seeds, with about 2× domain
FSI and 3.5× routing–domain MI. CE is lower by 0.008–0.011 on both calibration splits.

**Relative to SHUF** (same β, placement and q entropy; papers permuted, so paper consistency is kept and semantics
removed). Semantic EDL is higher on:
- domain FSI (+0.0013 / +0.0008);
- domain MI (+0.023 / +0.024);
- topic prediction (+0.014 / +0.018);
- domain interaction (+0.29 / +0.13 e-4).

The pseudo-FSI comparison is mixed: +0.0002 / +0.0005 on CALIB-SEARCH, and −0.0005 / +0.0005 on CALIB-SELECTION.
Seed-to-seed spread of FSI_pseudo for explicit EDL is 0.0010, larger than every SEM − SHUF pseudo gap. **The direction
favours semantics on the real-domain measures. Magnitudes are at or below the 2-seed noise floor on the preregistered
label-free metric.**

## 6. Which engineering changes helped or hurt (Stage 1, seed 100, s20)

Post-terminal domain FSI at s20: TOKEN-4 0.0028, EDL 0.0068.

| change | evidence |
|---|---|
| **strength ↑ helped** | C3 0.5 % → C2 1 % → C1 2 %: FSI_pseudo adv 0.0012 → 0.0011 → 0.0017; domain FSI 0.0052 → 0.0062 → 0.0075; MI_pseudo 0.105 → 0.146 → 0.204. CE did not get worse (4.649 → 4.636 → 4.630). No strength–CE trade-off up to 2 %; 4 % untested. |
| **upper placement: mixed** | C4 vs C1 had the best label-free FSI_pseudo (0.00361 vs 0.00310), which is why it was selected. But domain FSI was lower (0.0060 vs 0.0075) and CE higher (4.644 vs 4.630). The label-free proxy and real-domain specialisation disagree at s20. C5 vs C2 was neutral. |
| **sharpening (T = 0.5): neutral to slightly positive** | C6 vs C1: CE 4.626 vs 4.630, domain FSI 0.0071 vs 0.0075, FSI_pseudo equal. C8 vs C2: FSI_pseudo higher, domain FSI lower. |
| **ramp (2 %): slightly hurt early CE and specialisation, no long-run gain** | C7 vs C1 at s20: CE +0.019, domain FSI 0.0064 vs 0.0075. At s50, C7 (all-layer) domain FSI 0.0088 / 0.0092 ≈ EDL, but CE was worse in s100 (3.784). |

The ranking metric (FSI_pseudo advantage at s20 on 16 noisy pseudo-groups) selected C4. The all-layer 2 % variants
(C1, C6) had the best real-domain specialisation and the lowest CE at s20. This is descriptive and was not usable
under the protocol. It suggests that label-free selection on FSI_pseudo at 20M tokens is a noisy proxy.

## 7. Evidence-based reasons for the failure (diagnosis A–G, prereg §10)

| | factor | verdict | evidence |
|---|---|---|---|
| A | stability of q | not the cause | q frozen; SEM κ_seed 0.733, κ_view 0.461 (GC-EDL-v1); identical table across seeds. |
| B | pseudo-domain / expert usefulness alignment | **supported (main)** | Per-expert pseudo-domain interaction is the same for SEM- and SHUF-trained models (4.03 / 4.23 vs 4.43 / 4.20 e-4); both are ≈ 2× TOKEN-4. Training on the semantic grouping does not make experts more useful for that grouping than training on a random paper-consistent grouping. |
| C | strength vs CE trade-off | not the cause | Every strength reached its target. Higher strength improved specialisation without CE cost; CE ≤ EDL. |
| D | routing vs function | partly supported | SEM raises routing MI over SHUF (pseudo +0.03; domain +0.023), but FSI_pseudo gains do not follow reliably. Part of the routing differentiation is not functional. |
| E | interference | not supported | No CE harm in any candidate. Generic expert importance is unchanged. |
| F | label information beyond clusters | **supported** | SEM clusters carry 58 % of domain entropy as hard argmax (main-split NMI 0.42, purity 0.78), but **only 25 % as the soft q_d used by the loss** (0.38 of 1.54 nats; mean max q 0.40). Explicit EDL's advantage over SHUF in domain FSI (0.0017 / 0.0019) is about 1.4–2.3× semantic EDL's (0.0013 / 0.0008). |
| G | seed variance | **supported (proximate cause)** | EDL seed-to-seed FSI_pseudo spread is 0.0010. Every SEM − SHUF FSI_pseudo gap is 0.0002–0.0005 with one sign flip. Two seeds at 50M cannot confirm the effect on the preregistered metric. |

**Best-supported explanation.**
- An EDL-type population-divergence loss produces most of its 50M-token specialisation from any consistent
  paper-level grouping (SHUF ≈ 80 % of explicit EDL's domain FSI).
- The semantic content of SEM clusters adds a further, smaller increment (≈ 8–14 % of EDL's domain FSI). It is consistent in
  direction on real-domain measures but below the two-seed noise floor of the label-free confirmation metric (G).
- The increment is limited by how little domain information the soft cluster distribution carries (F), and it does not
  translate into experts that are functionally more aligned with the semantic groups than random-group experts (B).

## 8. What was not tested

- 245M-token finals and any VAL comparison; per-document CE and VAL bootstrap CIs. This follows the protocol, because
  confirmation failed.
- Engineering generations, including stronger targets (4 %), lower placement, T ∈ {0.25, 1}, longer ramps and late
  β decay.
- K ≠ 16 (no other frozen table exists).
- Hard, argmax-only targets (T = 0.5 is the sharpest tested).
- More than two seeds at 50M.
- Per-domain CE.
- Test (locked).

Follow-ups outside this protocol's scope were recorded, not run: counterfactual or utility losses, forecasts, extra
encoders, architecture changes.

## 9. Suggested next steps (each would need a new preregistration)

1. **Power.** Repeat the C4/C1-vs-SHUF comparison with ≥ 4 seeds, or at 245M tokens. Use real-domain-free metrics
   with lower variance, for example pseudo FSI with fixed groups from a held-out discovery set at a larger audit size.
2. **Increase q's information.** Use argmax / low temperature (T = 0.25), which recovers 58 % instead of 25 % of the
   domain entropy. This is cheap: one table, no new encoder.
3. Use all-layer 2–4 % strength, which was the best real-domain specialisation per unit CE at 20M.

## 10. Compute, disk, hygiene

- **GPU.** 7.16 GPU-h of 24 (`compute_ledger.tsv`, `compute_summary.json`): search training 1.56, calibration and
  checks 0.17, evaluation 4.64 (of which post-terminal audits 2.79). One GPU job at a time, all via
  `scripts/gpu_guarded.sh` through `sem_driver.py`. No foreign process was preempted, and there were no external
  interruptions. The two EXTERNALLY_STOPPED rows are the deliberate SIGINT test.
- **Disk.**
  - New runs use 7.54 GiB (`run_manifest.json`); artifacts use 0.05 GiB; 128.5 GiB is free.
  - Only disposable resumable slots of finished search runs were deleted (22 entries, `deleted_artifacts.tsv`).
  - No /dev/shm, no symlinks.
  - Search runs show status INTERRUPTED with reason max-steps: they stopped at their last scheduled milestone by design.
- **Integrity.**
  - `reuse_manifest_*.json` all_ok (23 checkpoints).
  - `protected_inventory_before.verify_20261115T154842Z.json`: 6,498 files, 0 mismatches.
  - Prereg hash unchanged.

**Deviations** (`deviations.jsonl`):
1. Diagnostic cadence of the EDL s100 run.
2. EDL s2 planned as the missing final baseline (never trained, because there were no finals).
3. GPU-check harness fix.
4. Post-terminal descriptive domain-label analysis on CALIB-SEARCH.

## 11. Deliverables

All paths are under `artifacts/semantic-edl-v1/` unless noted.

| item | path |
|---|---|
| final analysis (machine-readable) | `final_analysis.json` |
| candidate manifest | `candidate_manifest.json` |
| run manifest | `run_manifest.json`, `run_dirs.json`, `run_ledger.tsv` |
| compute ledger | `compute_ledger.tsv`, `compute_summary.json` |
| deviations | `deviations.jsonl` |
| protected-file / hash audit | `protected_inventory_before*.json`, `reuse_manifest_*.json`, `preregistration_hashes.json` |
| specialisation trajectories | `specialization_trajectories.json` |
| decisions | `search/{calibration,stage1_decision,stage2_decision,generation_decision}.json`, `frozen.json`, `calibsel_decision.json` |
| raw evaluations | `evals/<run>/{dev_s5,dev_s20,dev_s50,calibsel_s50}.json`, `post/*.json` |
| resume | `RESUME.md` (terminal; nothing to resume) |

## 12. Exact reproduction commands

```
cd /home/zapan4/moe-research && source env.sh
.venv/bin/python sem_tests.py                         # CPU tests
.venv/bin/python sem_integrity.py                     # reuse manifest (hash checks)
.venv/bin/python sem_inventory.py --out artifacts/semantic-edl-v1/protected_inventory_before.json
.venv/bin/python sem_tables.py                        # T=0.5 and shuffled tables
.venv/bin/python sem_gpu_checks.py                    # equivalence / SIGINT / resume checks (GPU, via driver)
.venv/bin/python sem_search.py stage0 && .venv/bin/python sem_search.py stage1 && .venv/bin/python sem_search.py stage2
.venv/bin/python sem_final.py freeze                  # frozen.json + CALIB-SELECTION (once) -> not confirmed
.venv/bin/python sem_post.py purity                   # post-terminal, descriptive
.venv/bin/python sem_post_chain.py                    # post-terminal GPU audits (sem_post.py audit ... via driver)
.venv/bin/python sem_analyze.py
.venv/bin/python sem_inventory.py --verify artifacts/semantic-edl-v1/protected_inventory_before.json
```

Each evaluation writes once, with mode 0400. Re-running a completed step is refused or skipped by the driver
(`driver_state.json`).
