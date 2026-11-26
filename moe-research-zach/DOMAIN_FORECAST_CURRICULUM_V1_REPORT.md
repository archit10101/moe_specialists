# csr-domain-forecast-curriculum-v1: report

**Status: COMPLETE.** All 7 conditions × 3 seeds (21 runs), the shared draft, the forecast caches, every
calibration audit, and 21 single-pass validation evaluations ran on 2026-11-10 (00:18 to 10:30 EST).

**Verdict: NEGATIVE.** The domain-forecast specialisation curriculum does **not** improve final next-token CE.
- SOFT-SPECIALIST→TOKEN-S40 is worse than plain TOKEN-MAIN by +0.013 nats (pooled, CI excludes 0). It is worse in
  all five domains.
- It is also worse than the label-only curriculum (LABEL), worse than spending the same unique data on TOKEN-4
  (ALL-DATA, +0.088), and worse than spending the same compute on TOKEN-4 (COMPUTE, +0.143).
- The curriculum does create real, functional, seed-reproducible expert specialisation. That specialisation survives
  removal of the assistance through 100% of training. It does not buy accuracy.

Preregistration: `docs/DOMAIN_FORECAST_CURRICULUM_V1_PREREG.md` (sha256 `0e10c756…eb51507`, hashed before any data).
Deviations (9): `artifacts/domain-forecast-curriculum-v1/deviations.jsonl`, summarised in §8.

## 1. Plain-language conclusion

- **Curriculum mechanics.** For the first 40% of training, a small domain-adapted draft's 4-token forecasts and a
  learnable domain-group prior pushed the router toward provisional per-domain experts. Over the next 10% this
  influence was annealed to zero. The final 50% was ordinary TOKEN-4.
- **The intended effect happened.** The prescribed expert groups became domain specialists: disabling a domain's
  group hurts that domain much more than the others. They stayed specialists to the end, and the same expert indices
  specialise the same way across seeds.
- **No accuracy benefit.** Every curriculum variant finished slightly *worse* than ordinary TOKEN-4 trained on the
  same 112.5M tokens: label-only +0.007, forecast-only +0.005, both +0.013.
- **Large opportunity cost.** Giving the target model the draft's 30M tokens directly (ALL-DATA) is 0.075 nats
  better than TOKEN-MAIN. Spending the curriculum's compute on more TOKEN-4 training (COMPUTE) is 0.130 better.
- **Specialisation does not predict accuracy.** Across the 15 main-plan runs, more specialisation went with
  equal-or-worse final CE.

## 2. Data (M2D2 @ 6ae613ba, arXiv L1)

- **Selected domains, mechanically:** cs_l1, math_l1, physics_l1, q-bio_l1 and **stat_l1**.
- **Substitution.** stat_l1 replaces econ_l1, whose post-dedup pool holds only 5.06M tokens of the 31.0M required.
- **Inventory.** All 8 candidates were inventoried and SHA-256 verified against their LFS hashes (archives > 200 MB
  streamed, not stored). `test.txt` was never decoded; it was hashed only.
- **Exact quotas per domain:** draft 6.0M, main 22.5M (P1 9.0M / P2 2.25M / P3 11.25M), calibration 1.5M,
  validation 1.0M. Totals: 30M / 112.5M / 7.5M / 5M.
- **Phase balance.** Every training phase is exactly 20% per domain, and domains are interleaved in every 32-row
  step.
- **Dedup.**
  - Exact: sha256 of normalised text. Cross-domain duplicates are dropped everywhere; within-domain repeats keep one
    copy.
  - Near: MinHash with 5-word shingles and Jaccard ≥ 0.8; cross-domain clusters are dropped entirely.
  - Splits are by **paper hash**.
- **Unit of data.** A paper is tokenised whole and cut into ≤ 1,024-token segments, all in the same split.
  - This deviates from the prereg's `tokens[:1023]` rule, and the change is needed to reach the quotas (§8).
  - Validation contains 413 papers (5,073 segments); bootstrap units are papers.
- **Locks.**
  - Manifests are locked (0400) and hashed: `splits_sha256 = 1d7c34b2…`.
  - The data manifest has sha `3bf91907…`. Full detail is in `artifacts/domain-forecast-curriculum-v1/data_manifest.json`.
- **Validation access.** Validation was opened only after `frozen.json` was written (10:13:37). The test split was
  never touched.

## 3. Final validation CE (nats/token; final prescribed checkpoints, strength 0, plain CSRModel export)

| condition | seed 0 | seed 1 | seed 2 | **mean** | sd | worst domain (physics) |
|---|---|---|---|---|---|---|
| TOKEN-MAIN (112.5M) | 3.5730 | 3.6021 | 3.5965 | **3.5905** | 0.0154 | 3.8914 |
| LABEL→TOKEN-S40 | 3.5848 | 3.5985 | 3.6090 | **3.5974** | 0.0122 | 3.8956 |
| FORECAST→TOKEN-S40 | 3.5889 | 3.5993 | 3.5975 | **3.5952** | 0.0055 | 3.8944 |
| **SOFT-SPECIALIST→TOKEN-S40** | 3.5887 | 3.5958 | 3.6271 | **3.6039** | 0.0204 | 3.9022 |
| WRONG-FORECAST→TOKEN-S40 | 3.5911 | 3.5959 | 3.6212 | **3.6027** | 0.0162 | 3.9013 |
| TOKEN-ALL-DATA (142.5M) | 3.5086 | 3.5186 | 3.5199 | **3.5157** | 0.0062 | 3.8098 |
| TOKEN-COMPUTE (169.2M) | 3.4587 | 3.4689 | 3.4551 | **3.4609** | 0.0072 | 3.7538 |

Per-domain means for SOFT: cs 3.8085, math 3.0117, physics 3.9022, q-bio 3.8134, stat 3.4836. The other conditions
are in `final_analysis.json`. Worst-domain CE is physics for every condition.

## 4. Primary comparisons (SOFT − X)

Values are paired, per-paper and token-weighted, with 95% bootstrap intervals (2,000 resamples over 413 papers).
Negative values favour SOFT. The preregistered rule is: "better" requires the pooled CI below 0 **and** ≥ 2 of 3
seeds.

| comparison | seed 0 | seed 1 | seed 2 | pooled [95% CI] | verdict |
|---|---|---|---|---|---|
| SOFT − TOKEN-MAIN | +0.0158 | −0.0063 | +0.0306 | **+0.0134** [+0.0123, +0.0145] | SOFT worse |
| SOFT − LABEL | +0.0039 | −0.0027 | +0.0181 | **+0.0065** [+0.0058, +0.0072] | SOFT worse |
| SOFT − FORECAST | −0.0002 | −0.0035 | +0.0297 | **+0.0087** [+0.0078, +0.0095] | not resolved (CI > 0, but only 1 of 3 seeds worse) |
| SOFT − WRONG-FORECAST | −0.0024 | −0.0001 | +0.0059 | **+0.0012** [+0.0006, +0.0017] | not resolved (within ~0.003 noise) |
| SOFT − TOKEN-ALL-DATA | +0.0802 | +0.0772 | +0.1072 | **+0.0882** [+0.0849, +0.0912] | SOFT worse |
| SOFT − TOKEN-COMPUTE | +0.1301 | +0.1269 | +0.1720 | **+0.1430** [+0.1389, +0.1467] | SOFT worse |

The per-seed CIs (in `final_analysis.json`) are about ±0.0015. The seed-to-seed spread of a difference (±0.015) is
an order of magnitude larger, so the pooled CIs understate between-seed uncertainty. The per-seed sign count is the
more honest guide.

Pooled SOFT − TOKEN-MAIN by domain: cs +0.014, math +0.017, physics +0.011, q-bio +0.012, stat +0.013 (worse in all
five).

**Claim rules (prereg §9), applied:**
- **Does the curriculum improve final accuracy?** No. SOFT is worse than TOKEN-MAIN, and so are LABEL (+0.007),
  FORECAST (+0.005) and WRONG (+0.012).
- **Do forecasts help beyond labels?** No. SOFT is worse than LABEL.
- **Does specialisation help beyond forecast features?** Not supported. SOFT does not beat FORECAST; pooled it is
  worse.
- **Does semantic forecasting matter?** Not supported. SOFT ≈ WRONG (+0.001).
  - This control is weak: the deranged-adapter forecasts are nearly as accurate as the correct ones (next-token
    0.267 vs 0.270), because the draft's adapters barely specialise.
- **Data efficiency?** No. SOFT uses 142.5M unique tokens (30M draft + 112.5M target) and is 0.088 worse than
  TOKEN-4 trained on those same 142.5M tokens.
- **Compute efficiency?** No. Spending the matched compute (169.2M TOKEN-4 tokens) is 0.143 better.

## 5. Specialisation (calibration split only; diagnostic, never used for selection)

All values below are means over 3 seeds. "Specificity" is ΔCE(own domain) − mean ΔCE(other domains) when that
domain's 4 most-enriched experts in one layer are disabled; it is averaged over 7 layers × 5 domains. "Prov. spec."
is the same quantity when the provisional group 4d..4d+3 is disabled in all layers. A = curriculum active;
B = neutralised (strength 0, no retraining).

| condition | ckpt | mode | MI (nats, null ≈ 0.0001) | prov. mass (uniform 0.125) | specificity [CI] | all-layer prov. spec. | cross-seed index-aligned cos | calib CE |
|---|---|---|---|---|---|---|---|---|
| TOKEN-MAIN | 40% | – | 0.026 | 0.125 | +0.0031 [0.0025, 0.0037] | +0.001 | −0.02 | 4.060 |
| TOKEN-MAIN | 100% | – | 0.028 | 0.125 | +0.0049 [0.0040, 0.0058] | +0.001 | −0.02 | 3.556 |
| LABEL | 40% | A | 0.130 | 0.268 | +0.0054 | +0.055 | 0.56 | 4.078 |
| LABEL | 40% | B | 0.035 | 0.162 | +0.0052 [0.0043, 0.0062] | +0.045 | 0.37 | 4.086 |
| LABEL | 100% | – | 0.061 | 0.201 | +0.0078 [0.0065, 0.0092] | +0.071 | 0.44 | 3.564 |
| FORECAST | 40% | A | 0.020 | 0.126 | +0.0022 | +0.002 | 0.01 | 4.139 |
| FORECAST | 40% | B | 0.021 | 0.122 | +0.0032 [0.0026, 0.0039] | +0.002 | 0.00 | 4.275 |
| FORECAST | 100% | – | 0.025 | 0.128 | +0.0044 [0.0037, 0.0052] | +0.003 | 0.00 | 3.561 |
| SOFT | 20% | A | 0.258 | 0.365 | +0.0034 | +0.035 | 0.62 | 4.552 |
| SOFT | 20% | B | 0.013 | 0.027 | +0.0012 [0.0008, 0.0016] | +0.010 | 0.20 | **4.868** (collapse flag: 5 unused experts, max share 0.23) |
| SOFT | 40% | A | 0.146 | 0.292 | +0.0049 | +0.052 | 0.62 | 4.125 |
| SOFT | 40% | B | 0.027 | 0.072 | +0.0040 [0.0031, 0.0050] | +0.033 | 0.40 | 4.278 |
| SOFT | 50% | – | 0.055 | 0.186 | +0.0062 [0.0053, 0.0071] | +0.057 | 0.46 | 4.000 |
| SOFT | 75% | – | 0.062 | 0.207 | +0.0071 [0.0061, 0.0082] | +0.065 | 0.50 | 3.748 |
| SOFT | 100% | – | 0.063 | 0.207 | +0.0079 [0.0068, 0.0092] | +0.075 | 0.50 | 3.570 |
| WRONG | 100% | – | 0.065 | 0.209 | +0.0083 [0.0071, 0.0096] | +0.078 | 0.45 | 3.569 |
| ALL-DATA / COMPUTE | final | – | 0.027 / 0.029 | 0.128 / 0.126 | +0.0053 / +0.0055 | +0.003 / +0.001 | ≈ 0 | 3.480 / 3.425 |

Full trajectory: `specialization_trajectory.json`. The rows not shown above (20% for LABEL and FORECAST, 50% and 75%
for LABEL, FORECAST and WRONG) are there.

**Findings**

- **Functional specialisation exists at 40%** (neutralised mode, CI > 0) in every condition, *including TOKEN-MAIN*.
  TOKEN-4 develops mild domain experts by itself, so the preregistered test is not specific to the curriculum.
- **The curriculum's specialisation is real and durable.** The prior-carrying conditions (LABEL, SOFT, WRONG) differ
  from TOKEN-MAIN in four ways:
  - About 2.3× higher domain–expert MI at 100% (0.06 vs 0.028).
  - 1.6× the per-layer ablation specificity.
  - Provisional groups that became functional domain experts. All-layer group ablation specificity is +0.075 vs
    +0.001 for TOKEN-MAIN, where the same expert indices are a random control.
  - Expert identities reproducible across seeds: index-aligned profile cosine 0.50 vs −0.02. Best-match cosine is
    ≈ 0.95 everywhere.
- **The specialisation survives the anneal.** It *grows* from 40% (neutralised) through 50%, 75% and 100% after all
  assistance is removed.
- **Forecasts alone do not create specialisation.** FORECAST tracks TOKEN-MAIN on every specialisation metric. The
  specialisation comes from the domain-group prior; the forecasts add none.
- **Neutralised routing is badly damaged during the curriculum.** At 20%, SOFT and WRONG routing without assistance
  is degenerate: calibration CE 4.87 vs 4.55 with assistance, and some experts are unused. The model had come to rely
  on the forecast routing, which carries next-token information. The damage is repaired during the anneal and
  ordinary training; no collapse remains at 50% or later, and none appears in training-mode audits.
- **Specialisation vs final CE** (Spearman across 15 runs; negative would mean "more specialisation, better CE"):
  - MI: +0.34 at 40% (neutralised), +0.23 at 100%.
  - Specificity: +0.25 at 40%, −0.09 at 100%.
  - So there is no evidence that specialisation improves CE; if anything the relationship is the reverse.
- **Learned gates and scales.**
  - Informative gate means: 0.79 → 0.63 (SOFT, 20% → 40%), 0.88 → 0.70 (FORECAST), 0.37 → 0.36 (LABEL).
  - Under SOFT/WRONG the top-4 set differs from token-only routing for 95–98% of tokens; under LABEL, 37–50%.
  - Bounded scales stayed near their 0.5 initial value: s_f ≈ 0.57, s_p ≈ 0.40–0.55. The model neither switched the
    assistance off nor amplified it.

**Matched-token learning curves** (CALIB-MON, neutralised CE, mean over seeds):

| tokens | 5.6M | 22.5M | 45M | 56.25M | 84.4M | 112.5M |
|---|---|---|---|---|---|---|
| TOKEN-MAIN | 5.737 | 4.700 | 4.176 | 4.027 | 3.829 | 3.652 |
| LABEL | 5.739 | 4.700 | 4.207 | 4.049 | 3.837 | 3.660 |
| FORECAST | 5.880 | 4.927 | 4.402 | 4.120 | 3.844 | 3.659 |
| SOFT | 5.968 | 4.975 | 4.401 | 4.106 | 3.848 | 3.667 |
| WRONG | 5.922 | 4.957 | 4.419 | 4.126 | 3.851 | 3.665 |

The forecast conditions start far behind on standalone routing and never fully catch up. ALL-DATA and COMPUTE track
TOKEN-MAIN until their own LR decay (see `final_analysis.json`).

## 6. Compute, throughput and inference

| item | GPU seconds |
|---|---|
| Draft training (30M tokens, 947 steps) | 106 step-s (110 ledger); **amortised over 3 seeds: 35** |
| Forecast cache, MAIN correct (the method's cache) | 656; **amortised over 3 seeds: 219** |
| Other cache jobs (MAIN deranged 655, CALIB correct and deranged 44 + 44; controls and audits) | 743 |
| Main training, all 21 runs | 18,412 |
| Calibration audits (105 audit jobs) | 5,763 |
| Final validation evaluations (21) | 292 |
| Benchmarks; checks and tests | 75; 61 |
| **Total GPU (dfc ledger)** | **26,124 s = 7.26 GPU-h** |

- **Data build:** CPU only, 1,981 s for the final attempt. Two earlier failed attempts took about 22 min each.
- **Training throughput (tok/s):**
  - TOKEN-MAIN 156.7k, LABEL 155.2k.
  - FORECAST, SOFT and WRONG about 136.2k, i.e. **+15% train-step time** for the forecast path.
- **Per-run step seconds:** TOKEN-MAIN 719, SOFT 827, ALL-DATA 908, COMPUTE 1,075.
- **SOFT's full cost per seed:**
  - Unamortised: 827 + 106 + 656 = 1,589 s, which is 2.2× TOKEN-MAIN.
  - Amortised over 3 seeds: 1,081 s, which is 1.50× TOKEN-MAIN.
- **Compute match:** frozen before validation opened, giving N = 169,195,000 tokens (1.504 × 112.5M).
  - TOKEN-COMPUTE cycled the 142.5M-token pool partially (2 passes over part of it).
  - Its measured step time of about 1,075 s matched C = 1,081 s.
- **Standalone final models** (seed 0; plain CSRModel export, identical by construction):
  - Prefill: about 390–413k tok/s at 16 × 1,024.
  - Batch-1 KV decoding: 8.5–8.7 ms/token, the same for every condition.
  - The curriculum path, while active, costs +63–70% in forward time on that batch.
- **Parameters:**
  - Standalone: 28,873,920 total, of which 14,423,232 are active per token.
  - With the curriculum modules, instantiated in every condition: 29,180,917 (+306,997, i.e. 1.06%).
  - Every token always used exactly 4 routed experts plus the shared path.
- **Loads:** no collapse at any final checkpoint (validation max expert share ≤ 0.041, 0 unused experts in every run).

## 7. Correctness checks (all passed before main training)

**CPU suite, CUDA hidden** (`cpu_tests_full_20261110T082759Z.json`): 19/19 tests pass.
- Split isolation by paper hash.
- Exact and near-dedup mechanics.
- Exact per-domain, per-phase token counts and domain interleaving.
- Draft adapter selection.
- Forecasts equal explicit autoregressive greedy decoding; future-token perturbation never changes earlier forecasts.
- Cache invariance to batching.
- Identical initialisation across all 7 conditions and equal to CSRModel(seed).
- Strength 0 gives exact TOKEN-4 with no draft and no labels.
- Exactly 4 distinct experts per token, with all 32 eligible.
- Shared path identical across conditions.
- Stop-gradient on the forecast embeddings; gradients flow only when the curriculum is active.
- Future-token perturbation never changes earlier main-model routes or logits.
- Split locks.
- **Bitwise** resume equality using the real training script.

**GPU checks** (`gpu_checks_20261110T081536Z.json`): all pass.
- Initialisation identity at full size.
- Strength 0 bitwise equal to the plain export.
- 4 distinct experts per token, with all 32 used per layer.
- Future perturbation with the real frozen draft: forecasts bitwise unchanged; main logits and routes unchanged.
- Finite one-batch smoke steps for all 7 conditions.

**Other checks**
- **GPU resume check:** 6 steps uninterrupted vs 3 + resume. Data position and init are identical; the maximum
  parameter difference is 1.1e-4, from GPU scatter-add nondeterminism.
- **Cache invariance:** 0 of 384 sampled documents mismatched per cache under a different batch size and order.
- **Frozen draft:** the draft checkpoint was hashed, all its parameters have `requires_grad = False`, and main
  training reads only the cache.
- **Runs:** every run completed in 1 attempt (no OOM, no interruption) and all losses were finite.
- **Final evaluation:** each final checkpoint was evaluated exactly once through the plain-CSRModel export, after
  asserting identical logits to the DFC model.
- **Protected files:** after the experiment, all 3,402 pre-existing files matched their hashes in
  `protected_inventory_before.json` (verify file 2026-11-10T10:20:58, n_bad = 0). The frozen draft checkpoint hash
  (`b42c442d…`) is unchanged.

## 8. Deviations, failures and fixes (`deviations.jsonl`)

1. **Prereg timestamp.** The header says "~00:45"; the file was actually hashed at ~00:28. This is clerical.
2. **Resume slot.** One rotating resume slot instead of a state file per run, to respect the 100 GiB disk reserve.
3. **Domain inventory.** econ_l1 was expected to fail the token requirement; it did, and stat_l1 substituted
   mechanically.
4. **Exact-dedup policy (data build attempt 1 failed).** "Drop every copy" also deleted within-domain repeats
   (q-bio lost 49k of 53k papers). Fixed to keep one copy within a domain; cross-domain duplicates are still dropped.
5. **Document rule (attempt 2 failed).** `tokens[:1023]` per paper made the 5-domain quotas infeasible on every
   candidate list (q-bio 13.8M, stat 18.2M).
   - Fix: whole papers cut into ≤ 1,024-token segments. Split, dedup and buckets remain per paper.
   - Decided on availability only, before any model existed. The failed inventories and logs are preserved.
6. **Bug: GPU-check result writer.** It mixed int and str keys in the result JSON. Fixed and rerun; no training was
   affected.
7. **Bug: prior lookup.** The indexed prior lookup had a nondeterministic backward (found by the CPU bitwise-resume
   test). It was replaced by an equivalent one-hot matmul **before any main run**.
8. **Bug: analysis key collision.** Analysis crashed before writing any file. Fixed and rerun; validation was not
   re-evaluated.
9. **Note: weak WRONG-FORECAST control** (see §4).

No hyperparameter was changed in response to model performance. No run was invalidated, and none was rerun or
selected.

## 9. Files

- **Report:** `DOMAIN_FORECAST_CURRICULUM_V1_REPORT.md`.
- **Experiment files** (`artifacts/domain-forecast-curriculum-v1/`):
  - `final_analysis.json` (sha `47fed394…`), `run_manifest.tsv`, `compute_ledger.tsv`, `data_manifest.json`,
    `specialization_trajectory.json`, `deviations.jsonl`.
  - `frozen.json` (sha `cfa294e1…`), `compute_match.json`.
  - `evals/val_*.json` (21), `cache/` (4 caches + manifests), `bench_*.json`, `gpu_checks_*.json`,
    `resume_check_gpu_*.json`, `cpu_tests_*.json`, `inventory.json` (+ the attempt-1 and attempt-2 FAILED
    inventories), `protected_inventory_before.json`.
- **Runs** (each holds `config.json`, `run_facts_attempt1.json`, `metrics.jsonl`, `log.txt`, `summary.json`,
  `ckpt_*.pt` (0400) and `audit_*.json`):
  - draft: `runs/20261110T074951Z_dfc_draft_d128_l4_adapt`;
  - main runs: `runs/20261110T*_dfc_<COND>_s<seed>`. Exact paths, config hashes, checkpoint paths and hashes are in
    `run_manifest.tsv`.
- **Code:** `dfc_*.py`, `src/csr/dfc_{model,lib,eval}.py`, `tests/test_dfc_v1.py`.
  - Configs: `configs/domain-forecast-curriculum-v1/`.
  - Driver state and logs: `.tmp/domain-forecast-curriculum-v1/`.
