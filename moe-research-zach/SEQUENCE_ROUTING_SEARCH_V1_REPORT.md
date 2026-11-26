# csr-sequence-routing-search-v1: report

**Question.** Can genuinely sequence-routed MoE beat ordinary token-routed MoE (TOKEN-4) on held-out language-model
loss under a matched comparison? Sequence-routed means one causal route (expert ids and gate weights) per segment,
identical in training and inference.

**Terminal state: COMPLETE. The bounded search and the three-seed final confirmation both ran. Result: NEGATIVE.**
- No sequence-routing candidate beat TOKEN-4 at any stage: CALIB-SEARCH (3 generations, 16 unique candidates),
  blind CALIB-SELECTION, or the three-seed validation.
- At 100M tokens on the full validation split:

  | Model | Δ CE vs TOKEN-4 (nats/token) | 95% CI |
  |---|---|---|
  | Best fixed | **+0.116** | [+0.1157, +0.1166] |
  | Best dynamic | **+0.144** | [+0.1435, +0.1445] |

- Both sequence models are worse in every seed and on 99.7% and 99.95% of documents respectively.
- Neither the "substantially better" rule nor the "accuracy-comparable" rule is met.

All stages ran under the frozen preregistration:
- prereg: `docs/SEQUENCE_ROUTING_SEARCH_V1_PREREG.md`, sha256 `7cc213b8…5794d`;
- the val split was opened exactly once per final checkpoint, after the final configurations were frozen and hashed;
- the test split was never accessed;
- protected artifacts were re-verified: 1,101/1,101 unchanged.

## Answers to the required questions

1. **Did any sequence-routing candidate beat token routing on CALIB-SEARCH?** **No.** Every candidate in every
   generation had a higher CALIB-SEARCH CE than the TOKEN-4 run with the same seed and budget.
   - Best valid differences (Δ vs TOKEN-4):

     | Generation | Budget | Best candidate | Δ vs TOKEN-4 |
     |---|---|---|---|
     | 1 | 5M | G1-F16 | +0.0159 |
     | 2 | 10M | G2-F4 (fixed-16 + GRU) | +0.0181 |
     | 3 | 30M | G3-F4 | +0.0603 |

   - The gap grew with training (+0.12 at 100M on CALIB-SEARCH).
   - The lowest Gen-1 CE belonged to G1-F256 (+0.0102), which was invalid because it left 5–7 experts unused per
     layer. Even so, it was worse than TOKEN-4.
2. **Did the gain survive blind CALIB-SELECTION?** There was no gain to survive. On CALIB-SELECTION (3-seed means,
   evaluated once):

   | Model | CE | Δ vs TOKEN-4 |
   |---|---|---|
   | TOKEN-4 | 5.0676 | — |
   | Best fixed (G3-F4) | 5.1279 | +0.0604 |
   | Best dynamic (G3-D4) | 5.1391 | +0.0716 |

   - The shift from CALIB-SEARCH to CALIB-SELECTION was the same for all three: +0.0100 (TOKEN-4), +0.0101 (fixed),
     +0.0106 (dynamic).
   - So there is **no sign that the search overfit CALIB-SEARCH**. The deficit is real and stable.
   - Both families were promoted, as the frozen rule requires: valid, and not worse than +0.10.
3. **Did it survive three-seed validation?** No; the deficit grew. Validation CE (nats/token, mean ± SD over seeds 0,
   1, 2):

   | Model | Val CE | PPL | Δ vs TOKEN-4: per seed | mean ± SD | Paired doc bootstrap 95% CI |
   |---|---|---|---|---|---|
   | TOKEN-4 | 4.2239 ± 0.0034 | 68.30 | — | — | — |
   | G3-F4 (fixed) | 4.3400 ± 0.0084 | 76.71 | +0.1214, +0.1159, +0.1111 | **+0.1161 ± 0.0052** | [+0.1157, +0.1166] |
   | G3-D4 (dynamic) | 4.3679 ± 0.0067 | 78.88 | +0.1483, +0.1362, +0.1475 | **+0.1440 ± 0.0068** | [+0.1435, +0.1445] |

   - The bootstrap resamples 16,850 validation documents, 2,000 times, with `default_rng(20261106)`. It is
     seed-averaged and is not seed uncertainty.
   - Per-seed bootstrap CIs all exclude 0 by more than 0.11.
4. **Did fixed or dynamic sequence routing perform better?** **Fixed**, consistently:
   - **Generation 1:** the best valid fixed candidate (+0.016) beat the best dynamic one (+0.025).
   - **Generation 2:** +0.018 vs +0.023.
   - **Generation 3:** +0.060 vs +0.071.
   - **Validation:** dynamic − fixed = +0.0279 ± 0.0081, positive in all seeds; bootstrap CI [+0.0275, +0.0283].

   Under the prereg interpretation: predictable regular segments were easier to use than adaptive causal boundaries.
   The adaptive boundaries added no useful information.
5. **Which segment length and summary worked best?**
   - **Segment length:** among valid candidates, the **shortest fixed length, 16 tokens**, was best. Lengths 128 and
     256, and change-point target 128, left experts unused (invalid).
   - **Summary:** a **GRU state** updated once per completed segment over [W·mean(previous segment), multi-hot
     previous route]. In the Gen-1 fixed-32 probe (CE, seed 98), GRU was best:

     | Summary | CE |
     |---|---|
     | GRU | 6.5637 |
     | Attention pooling | 6.5658 |
     | Mean | 6.5718 |

     GRU was also best in both families in Gen 2.
   - **Best fixed overall:** G3-F4 = fixed 16 + mean + GRU.
   - **Best dynamic:** G3-D4 = change point on the dense-layer-0 residual stream (target 64) + GRU.
   - Learned boundary detectors and per-layer change points were worst.
6. **Did sequence routing meet the 0.02-nat substantial-improvement rule?** **No.** The mean differences are
   +0.116 and +0.144, against a requirement of ≤ −0.02. The sign, the CI and the mean-size criteria all fail. The
   no-collapse, active-expert, token-matching, compute and same-algorithm criteria are all satisfied.
7. **Was it at least non-inferior?** **No.** The rule requires a mean, every seed, and the bootstrap upper bound all
   ≤ +0.01. Observed:
   - fixed: mean +0.116, worst seed +0.121, upper bound +0.1166;
   - dynamic: mean +0.144, worst seed +0.148, upper bound +0.1445.
8. **Did it reduce or increase measured training and inference time?** It **increased** time, even though it called
   the router far less often.

   | | TOKEN-4 | Fixed G3-F4 | Dynamic G3-D4 |
   |---|---|---|---|
   | Router invocations / 1k tokens / layer | 1000 | 61.5 | 13.9 |
   | Analytic training FLOPs vs TOKEN-4 | — | −0.22% | −0.25% |
   | Training throughput (valid tok/s) | 117.9k | 30.4k | 61.9k |
   | Training GPU time per 100M run | 872 s | 3,353 s (**3.85×**) | 1,652 s (**1.89×**) |
   | Validation throughput (tok/s) | 249k | 74k | 132k |
   | Prefill, B = 1, 1,024-token doc | 13.3 ms | 144.2 ms | 42.0 ms |
   | Prefill, B = 16 packed rows | 30.4 ms | 155.3 ms | 70.1 ms |
   | Decoding, B = 1 (ms/token, 256 tokens) | 8.34 | 8.36 | 8.86 |
   | Decoding, B = 16 lockstep (ms/step) | 8.23 | 8.31 | 9.46 |

   - **Why it is slower:** the cost comes from this implementation's sequential parts: a Python loop over segment
     index for the GRU state, and a CPU causal boundary scan with a GPU→CPU sync for dynamic segmentation.
   - **Not FLOPs:** the FLOP count is slightly *lower*.
   - **Decoding is launch-bound and unchanged.**
   - **Conclusion under the prereg:** router calls were not the bottleneck. A fused implementation could remove most
     of the overhead, but that would not change the accuracy result.
9. **Did any candidate exploit additional parameters, compute or information?**
   - **Parameters:**
     - the finalists add 73,472 parameters (0.254% of 28.87M; limit 0.5%);
     - the bos-only candidates add 224;
     - attention and LBD variants add about 1.6k.
   - **Compute:**
     - analytic FLOPs were −0.2%;
     - measured GPU time was +285% (fixed) and +89% (dynamic). This is disclosed. These models got *more* wall-clock
       for the same tokens and still lost.
   - **Tokens and initialization:** training tokens were identical per seed (same step plan; asserted in the
     analysis), and so was the backbone initialization (hash asserted).
   - **Information:** routes never use the current or future segment. This is verified by tests and GPU checks.
   - **Disclosed edge cases:**
     - The learned-boundary-detector candidates (Gen 1–2 only; never finalists) used a future-window change label
       **as a training target only**.
     - τ is initialized from a no-grad pre-pass on the first training micro-batch.
   - None of these could explain a deficit of this size, and none produced an advantage.
10. **Did expert collapse occur?**
    - **Finalists and final models:** no collapse. Final validation:
      - 0 unused experts in every layer and every seed;
      - max expert share 0.042 (fixed) and 0.052 (dynamic), vs 0.038 for TOKEN-4;
      - load CV 0.12–0.24 vs 0.08.
    - **Generation 1:** partial collapse (unused experts) in three candidates, which were invalid under the frozen
      rule:

      | Candidate | Unused experts per layer |
      |---|---|
      | G1-F128 | 1–3 |
      | G1-F256 | 5–7 |
      | G1-CP128 | 1–2 |

      With 256-token segments, the content-free first-segment (BOS) route covers 38% of tokens. The maximum expert
      share (0.095) equals exactly that share ÷ 4.
11. **Were all routes genuinely sequence-level and causal?** **Yes.** One SeqRouteModel code path serves training,
    evaluation, CALIB-SELECTION, validation and latency. Evidence:
    - **CPU suite:** 165 new tests (404 total, all passing). They cover:
      - identical ids and gates within a segment;
      - an already-selected route unaffected by in-segment changes;
      - future text not changing earlier boundaries, routes or logits;
      - later routes depending on the completed previous segment;
      - document isolation;
      - a content-free BOS route;
      - incremental = full (B = 1 and batched; exact);
      - train-mode = eval-mode routing;
      - exactly 4 distinct experts per token;
      - the router never applied to a token-level tensor;
      - matched initialization;
      - finite gradients;
      - dynamic scan = token-by-token rule;
      - bit-exact resume.
    - **GPU checks** (full size, real documents, all 13 Gen-1 configurations):
      - incremental decoding reproduced the routes with 0 mismatches;
      - future perturbation left earlier routes and boundaries exact.
12. **What was total GPU time, wall time, peak VRAM and remaining disk?**
    - **GPU time:** **31,572 GPU-s = 8.77 GPU-h**, against a 20 h cap; the breakdown is below.
    - **GPU wall span:** 01:40:53 → 10:47:38 on 2026-11-06 (9.1 h).
    - **Whole experiment:** inspection at 01:07 → report at about 11:00 (about 10 h).
    - **Peak reserved VRAM:** 26.24 GiB (final dynamic training, seed 0), against a 56 GiB ceiling.
    - **Disk:** 300 GB free at the end, from 306 GB at the start.
    - **GPU 0:** idle at the end. No foreign GPU process appeared.
13. **What negative results are established?** These hold within this setup (≈ 29M parameters, 32 experts, top-4 +
    shared, FineWeb-Edu, ≤ 100M tokens):
    - **Sequence-level expert selection is clearly and robustly worse than token routing.**
      - At 100M tokens: +0.116 nats with the best fixed router, +0.144 with the best dynamic router; 3/3 seeds; CI
        width about 0.001.
      - The gap widens with training: about +0.016 at 5M → +0.06 at 30M → +0.12 at 100M.
    - **Shorter segments are better.**
      - Long segments (128/256) push a large share of tokens onto the fixed BOS route and leave experts unused.
      - Even 16-token segments lose to per-token routing.
    - **Adaptive causal boundaries do not help.** Change points (per-layer or layer-0) and a learned detector were
      all worse than fixed segmentation of similar mean length.
    - **The deficit is not caused by the content-free first segment.** It is nearly uniform across in-document
      positions (fixed: +0.135 at tokens 0–16, +0.115 at 256–1024).
    - **Sequence routing did not speed anything up** here. Router calls were 16–72× fewer, but they were not the
      bottleneck.
14. **What remains unresolved?** See §9. Briefly:
    - **Under-confident routers.** The routers stayed nearly uniform. Route-logit RMS was about half of token
      routing's, and the 4 selected experts received only 17–26% of the routing mass, vs 34–53% in TOKEN-4 (post-hoc,
      exploratory).
    - **Unexplored knobs.** Temperature, router-LR and initialization-scale settings were allowed by the protocol but
      not reached by the frozen Gen-2 rule. Whether they would close part of the gap is untested.
    - **Out of scope:** behaviour at larger scale and longer training, and a fused fast implementation.
    - **No claim of unbounded success.** Nothing here suggests that continued search would succeed, and no such
      claim is made.

## 1. Implementation (frozen at prereg; unchanged since)

All code is new; no pre-existing file was modified. A pre-edit snapshot is at
`.tmp/srsv1_snapshot_20261106T061354Z/` with its sha256 list alongside.

| File | Role |
|---|---|
| `src/csr/seqroute.py` | `SeqRouteModel` (subclass of `CSRModel`): token mode = TOKEN-4, bit-identical to the original router; sequence mode; batched incremental `sr_step` |
| `src/csr/srs_run.py` | partition, split locks, single-use evaluation registry, evaluation diagnostics, analytic FLOPs |
| `src/csr/srs_select.py` | Gen-1 list and the frozen Gen-2, Gen-3, freeze, promotion, budget and criteria rules (pure functions) |
| `srs_train.py`, `srs_evaluate.py`, `srs_gpu_checks.py`, `srs_bench.py`, `srs_analyze.py` | entry points |
| `srs_make_configs.py`, `srs_partition.py`, `srs_inventory.py` | configs, partition, protected-artifact inventory |
| `srs_driver.py` | resumable driver |
| `srs_report_tables.py`, `srs_posthoc_logit_scale.py` | added after the freeze; read-only reporting |
| `tests/test_sequence_routing_v1.py` | new tests |

**Sequence routing.** For each MoE layer and document, contiguous segments never cross a document.
- **Segment 0:** routed by learned content-free BOS logits.
- **Segment j ≥ 1:** routed by W_l (the layer's own router matrix) applied to a pooled summary of the **completed**
  segment j−1. The pooling is mean, last or attention. The state across segments is none, EMA, or a GRU over
  [W·pool, previous route] with a zero-initialized output layer.
- **Per segment:** softmax/T over 32 experts; the top-4 ids and their global-softmax gates are computed once and
  broadcast to every token. Each token executes exactly those 4 routed experts plus the shared expert, through the
  same dropless dispatch as TOKEN-4.
- **Losses:** load-balance and z-losses use the segment routes, token-weighted, with the same coefficients as
  TOKEN-4.

**Segmentations.**
- **Fixed:** L ∈ {16…256}, with an optional 16-token opening segment.
- **Change point:** cosine change between the mean of the last 8 and the mean of the preceding 32 states, from h_l or
  the layer-0 residual stream. The rule is min 16 / max 256, threshold τ.
- **Learned detector:** a linear probe on detached h_t.
- In every case the decision comes after observing token t, the new route starts at t+1, and boundaries are never
  revised.
- τ is a checkpointed buffer, adapted only between optimizer steps and frozen at evaluation.

## 2. Preregistration and integrity

**Preregistration and partition**

| Item | Value |
|---|---|
| Prereg | `docs/SEQUENCE_ROUTING_SEARCH_V1_PREREG.md`, sha256 `7cc213b88f213d3bdc5d3fbe4c49e1b0e5bcda027c60c57f14efc5cf23e5794d` |
| Prereg ledger | `SRS_V1_PREREG` at 01:33:00, before any candidate run (first GPU run 01:40:53); never edited |
| Prereg content | the complete user protocol (Appendix A) plus concrete rules |
| Partition | `artifacts/sequence-routing-search-v1/search_partition.json`, sha256 `81454b91…b104a` |
| Partition rule | sha256("csr-srsv1-calib-partition\|" + doc_id) < 0.70 |
| CALIB-SEARCH | 5,763 docs / 3,738,549 tokens |
| CALIB-SELECTION | 2,416 docs / 1,531,594 tokens |

**Evaluation isolation**
- CALIB-SELECTION was opened only after `search_frozen.json` was written (sha256 `dc150bc3…`). There were 9
  evaluations, one per checkpoint, enforced by the registry `artifacts/sequence-routing-search-v1/eval_registry/`.
- Validation was opened only after `final_frozen.json` (sha256 `8bd974e0…`) and after all final training. There were
  9 evaluations, one per checkpoint.
- The test split was never touched; loaders refuse it before file access.

**Protected artifacts**
- `protected_inventory.json` covers 1,101 files (23.6 GB), including all previous runs, checkpoints, data (test
  excluded by name), caches, configs, docs, reports and manifests.
- The post-run verification `protected_inventory.verify_20261106T154803Z.json` found **1,101/1,101 matching, 0 bad**.

**Code after the prereg.** Model, training, selection, evaluation and analysis code are byte-identical to the
prereg-time hashes in `manifest.json`. Exactly three files changed (deviations D1–D3, §8).

## 3. Tests and GPU checks

- **CPU suite (CUDA hidden):** **404 passed** = 239 pre-existing (unchanged) + 165 new
  (`cpu_tests_20261106T063811Z.json`). This includes the resume test: an interrupted then resumed run is bit-identical
  to an uninterrupted one on CPU.
- **GPU checks** (`gpu_checks_20261106T064342Z.json`): **PASS** for TOKEN-4 and all 12 Gen-1 candidates:
  - finite bf16 loss and gradients, with gradients reaching the sequence-router parameters;
  - identical ids and gates per segment, exactly 4 experts per token, BOS first route;
  - future perturbation on real CALIB-SEARCH documents: logits ≤ 1e-4, routes and boundaries exact;
  - incremental decoding at B = 1 (512 tokens) and B = 4: 0 route mismatches, logits ≤ 2e-3 (observed about 2e-6);
  - peak 24.9 GiB.

## 4. Search (CALIB-SEARCH only; development evidence, not final evidence)

Seed noise for reference: earlier experiments measured a within-seed run-to-run SD of ≈ 0.0026 at 5M tokens.

### Generation 1 (seed 98, 5M tokens; TOKEN-4 = 6.5465)

| Candidate | Family | Segmentation / summary | Δ vs TOKEN-4 | Valid | Unused experts (max/layer) | k tok/s |
|---|---|---|---|---|---|---|
| G1-F16 | fixed | fixed 16 / mean | +0.0159 | yes | 0 | 101.5 |
| G1-F32 | fixed | fixed 32 / mean | +0.0253 | yes | 0 | 100.6 |
| G1-F64 | fixed | fixed 64 / mean | +0.0251 | yes | 0 | 100.4 |
| G1-F128 | fixed | fixed 128 / mean | +0.0143 | **NO** | 3 | 101.0 |
| G1-F256 | fixed | fixed 256 / mean | +0.0102 | **NO** | 7 | 100.2 |
| G1-F32-GRU | fixed | fixed 32 / mean + GRU | +0.0173 | yes | 0 | 49.2 |
| G1-F32-ATTN | fixed | fixed 32 / attention pooling | +0.0194 | yes | 0 | 98.3 |
| G1-CP32 | dynamic | change point per layer, target 32 | +0.0307 | yes | 0 | 87.8 |
| G1-CP64 | dynamic | change point per layer, target 64 | +0.0300 | yes | 0 | 90.2 |
| G1-CP128 | dynamic | change point per layer, target 128 | +0.0193 | **NO** | 2 | 91.7 |
| G1-CP64-L0 | dynamic | change point on layer-0 output, target 64 | +0.0254 | yes | 0 | 98.5 |
| G1-LBD64 | dynamic | learned boundary detector, target 64 | +0.0290 | yes | 0 | 84.7 |

TOKEN-4 ran at 112.5k tok/s. Measured mean segment lengths matched the targets: change point about 31/64/124, layer-0
62.6, LBD 32–192 depending on the layer.

### Generation 2 (seeds 98 and 99, 10M tokens; TOKEN-4 = 5.9798), produced by the frozen deterministic rule

| Candidate | Parent | Change | Δ vs TOKEN-4 (mean; per seed) | k tok/s |
|---|---|---|---|---|
| G2-F1 | G1-F16 | carry (best fixed) | +0.0284 (+0.0361, +0.0208) | 103.9 |
| G2-F2 | G1-F32-GRU | carry (second) | +0.0274 (+0.0279, +0.0269) | 49.3 |
| G2-F3 | G1-F16 | + EMA state (opening not applicable at L = 16) | +0.0311 (+0.0369, +0.0253) | 50.3 |
| **G2-F4** | G1-F16 | + GRU state (best F32 summary probe) | **+0.0181** (+0.0239, +0.0122) | 29.0 |
| G2-D1 | G1-CP64-L0 | carry (best dynamic) | +0.0324 (+0.0371, +0.0278) | 101.2 |
| G2-D2 | G1-LBD64 | carry (second) | +0.0570 (+0.0639, +0.0500) | 87.0 |
| G2-D3 | G1-CP64-L0 | + 16-token opening segment | +0.0305 (+0.0336, +0.0274) | 99.5 |
| **G2-D4** | G1-CP64-L0 | + GRU state | **+0.0226** (+0.0279, +0.0173) | 58.5 |

All Gen-2 candidates were valid. G2-F1, G2-F2, G2-D1 and G2-D2 are identical configurations to their Gen-1 parents.
Unique sequence candidates in total: **16** (12 + 4 new); the limit is 24.

### Generation 3 (seeds 97, 98, 99, 30M tokens; TOKEN-4 = 5.0576), top 2 per family

| Candidate | = Gen-2 | Gen-1 ancestor | Δ vs TOKEN-4 (mean; per seed) | k tok/s | GPU s/run |
|---|---|---|---|---|---|
| **G3-F4** (frozen best fixed) | G2-F4 | G1-F16 | **+0.0603** (+0.0570, +0.0665, +0.0574) | 29.9 | 1,060 |
| G3-F2 | G2-F2 | G1-F32-GRU | +0.0660 (+0.0606, +0.0657, +0.0718) | 50.5 | 632 |
| **G3-D4** (frozen best dynamic) | G2-D4 | G1-CP64-L0 | **+0.0710** (+0.0681, +0.0762, +0.0687) | 60.2 | 532 |
| G3-D3 | G2-D3 | G1-CP64-L0 | +0.0772 (+0.0718, +0.0819, +0.0779) | 102.5 | 314 |

TOKEN-4 ran at 117.1k tok/s and 276 s per run.

### Candidate genealogy

| Gen-3 | Gen-2 | Gen-1 | Lineage |
|---|---|---|---|
| G3-F4 | G2-F4 | G1-F16 | G1-F16 → +GRU (G2-F4) → G3-F4 → final |
| G3-F2 | G2-F2 | G1-F32-GRU | G1-F32-GRU → carried |
| G3-D4 | G2-D4 | G1-CP64-L0 | G1-CP64-L0 → +GRU (G2-D4) → G3-D4 → final |
| G3-D3 | G2-D3 | G1-CP64-L0 | G1-CP64-L0 → +opening-16 |

### Rejected candidates

| Candidate | Stage | Reason |
|---|---|---|
| G1-F128, G1-F256, G1-CP128 | Gen 1 | **invalid**: unused routed experts on CALIB-SEARCH (collapse rule) |
| G1-F32, G1-F64, G1-F32-ATTN | Gen 1 | not among the top-2 valid fixed candidates; ATTN and mean lost the F32 summary probe to GRU |
| G1-CP32, G1-CP64 | Gen 1 | not among the top-2 valid dynamic candidates |
| G2-F1, G2-F3 | Gen 2 | ranks 3–4 among fixed |
| G2-D1, G2-D2 | Gen 2 | ranks 3–4 among dynamic |
| G3-F2, G3-D3 | Gen 3 | not the lowest 3-seed CE within their family |
| Semantic (MiniLM) boundaries | prereg | not included: they need an external encoder at inference, and csr-multiscale-routing-v1 showed no advantage |

No candidate was rejected for parameters (all ≤ 0.254%) or for non-finite values. No run failed.

## 5. Freeze, blind CALIB-SELECTION and budget gate

**Freeze.** Best fixed = **G3-F4**; best dynamic = **G3-D4**. `search_frozen.json` records the 9 Gen-3 config hashes
for TOKEN-4 and the two finalists. It was written before any CALIB-SELECTION access.

**CALIB-SELECTION.** Per-seed CE (seeds 97 / 98 / 99):

| Model | Seed 97 | Seed 98 | Seed 99 |
|---|---|---|---|
| TOKEN-4 | 5.0786 | 5.0578 | 5.0664 |
| Fixed (G3-F4) | 5.1359 | 5.1250 | 5.1229 |
| Dynamic (G3-D4) | 5.1477 | 5.1347 | 5.1349 |

- There was no collapse.
- Promotion: both families (+0.060 and +0.072 are ≤ +0.10). Ranking: fixed, then dynamic.

**Budget gate** (`budget.json`)
- Measured seconds per token at Gen 3 (including start-up and evaluation): TOKEN-4 9.2e-6, fixed 3.55e-5, dynamic
  1.78e-5.
- For **100M** tokens: used 13,092 s + 1.2 × 21,646 s projected = 39,067 s, which is ≤ 72,000 s. **100M was chosen.**
- Peak Gen-3 VRAM was 25.9 GiB. Disk was ≥ 300 GB free throughout.

## 6. Final confirmation (100M tokens; seeds 0, 1, 2; full validation once per checkpoint)

The primary results are in answer 3. Integrity checks:
- identical valid training tokens per seed (100,007,082 / 100,010,625 / 100,016,977 for seeds 0 / 1 / 2; same step
  plan);
- identical backbone initialization hash within each seed;
- identical validation token count (10,732,175) and document count (16,850) for every checkpoint.

**Validation CE by in-document position** (difference vs TOKEN-4, mean over seeds):

| Model | 0–16 | 16–64 | 64–256 | 256–1024 |
|---|---|---|---|---|
| Fixed G3-F4 | +0.135 | +0.126 | +0.115 | +0.115 |
| Dynamic G3-D4 | +0.133 | +0.159 | +0.148 | +0.141 |

**Per-document seed-averaged CE difference**
- Fixed: median +0.116 (5–95%: +0.059 to +0.174); 0.34% of documents better.
- Dynamic: median +0.144 (5–95%: +0.088 to +0.208); 0.05% of documents better.
- The per-document file is `val_doc_ce.json`.

**Criteria table** (prereg §11)

| Criterion | Fixed G3-F4 | Dynamic G3-D4 |
|---|---|---|
| Mean Δ ≤ −0.02 | ✗ (+0.116) | ✗ (+0.144) |
| All 3 seeds beneficial | ✗ (0/3) | ✗ (0/3) |
| Bootstrap CI < 0 | ✗ | ✗ |
| No collapse | ✓ | ✓ |
| No more active experts (4 + shared) | ✓ | ✓ |
| Matched training tokens | ✓ | ✓ |
| ≤ 10% extra training FLOPs | ✓ (−0.22%) | ✓ (−0.25%) |
| Same algorithm in train and inference | ✓ | ✓ |
| **Substantially better** | **NO** | **NO** |
| Mean Δ ≤ +0.01 / every seed ≤ +0.01 / bootstrap upper < +0.01 | ✗ / ✗ / ✗ | ✗ / ✗ / ✗ |
| **Accuracy-comparable** | **NO** | **NO** |

## 7. Secondary outcomes (seed 0 unless noted; validation)

### Routing and segments

| Metric | TOKEN-4 | Fixed G3-F4 | Dynamic G3-D4 |
|---|---|---|---|
| Routing entropy, layers 1–7 (nats; uniform = 3.47) | 3.20, 2.98, 2.88, 2.75, 2.66, 2.59, 2.80 | 3.37–3.43 (3.26 at layer 7) | 3.33–3.45 |
| Expert usage entropy | 3.462–3.465 | 3.457–3.463 | 3.440–3.461 |
| Max expert share | 0.038 | 0.042 | 0.052 |
| Load CV (max layer) | 0.085 | 0.129 | 0.24 |
| Unused experts | 0 | 0 | 0 |
| Route changes / 1k tokens | ≈ 985–997 | 51–61 | 12.6–13.8 |
| Consecutive-route Jaccard (route stability) | 0.11–0.19 | 0.26–0.55 | 0.21–0.43 |
| Mean segment length | — | 15.9 | 64.7 (closed 64.8) |
| Segment-length quantiles (p10/25/50/75/90) | — | — | 16 / 24 / 44 / 84 / 145 |
| BOS-route token share | — | 3.0% | 5.8% |
| Expert dispatch, max/mean per micro-batch | 2.55 (max 3.64) | 2.60 (max 4.08) | 2.66 (max 3.94) |

The sequence routers settle on near-uniform route probabilities. Token routers become sharper.

**Specialization diagnostic.** Route stability is higher under sequence routing (Jaccard 0.26–0.55 vs 0.11–0.19 for
token routing), so experts are reused across consecutive segments. That is consistent with coarse, topic-level
assignment. However, the routing entropy shows the routers barely discriminate, and expert usage stays nearly
uniform. There is no evidence that the extra reuse buys anything: CE is uniformly worse.

**Near topic transitions** (dynamic, validation CE by in-segment position, layer 1):

| In-segment tokens | 0–4 | 4–16 | 16–64 | 64+ |
|---|---|---|---|---|
| CE | 4.58 | 4.47 | 4.37 | 4.30 |

The tokens right after a detected change point are hard. This is partly selection: boundaries sit where content
shifts.

**Near document transitions.** The deficit at the first 16 tokens of a document (+0.13) is similar to the overall
deficit. So the content-free BOS route is not the main cause.

### Other measurements

- **Training curves** (CALIB-SEARCH head CE at 5/25/45/65/85M):

  | Model | 5M | 25M | 45M | 65M | 85M |
  |---|---|---|---|---|---|
  | TOKEN-4 | 6.467 | 5.217 | 4.828 | 4.591 | 4.404 |
  | Fixed | 6.486 | 5.266 | 4.896 | 4.691 | 4.524 |
  | Dynamic | 6.486 | 5.280 | 4.922 | 4.717 | 4.544 |

  The gap widens monotonically. Full curves are in `final_analysis.json` (`per_run.*.curves`).
- **Throughput and latency:** answer 8.
- **Peak VRAM** (training, reserved):

  | Model | Peak reserved |
  |---|---|
  | TOKEN-4 | 24.7 GiB |
  | Fixed | 25.3 GiB |
  | Dynamic | 26.2 GiB |

  Inference peak was 2.2–2.7 GiB.
- **Extra router parameters:** 0 / 73,472 / 73,472.
- **Training FLOPs** per 100M run: 9.596e15 / 9.575e15 / 9.573e15.

### Post-hoc mechanism diagnostic (exploratory; not preregistered; no decision depends on it)

Source: `posthoc_logit_scale.json`, seed-0 final checkpoints, 4 CALIB-SEARCH rows.

| Measure | TOKEN-4 | Sequence models |
|---|---|---|
| Centered route-logit RMS | 0.67–1.04 | 0.32–0.58 |
| Router-matrix norm | 5.1–6.3 | 3.8–4.6 |
| Gate mass on the 4 selected experts | 0.34–0.53 | 0.17–0.26 |

The sequence routers learn weak, low-confidence routes. The routed experts therefore contribute about half as much
mixture weight, and the shared expert carries more of the load.

## 8. Failures, fixes and deviations

Logged in `artifacts/sequence-routing-search-v1/deviations.jsonl` (sha256 `5d243ee6…`) and the safety ledger.

| Id | What | Effect |
|---|---|---|
| D1 | `srs_gpu_checks.py`: the replacement "future" document in C3 could be shorter than the original (broadcast error) and crashed the first GPU-check attempt after about 13 s. That attempt wrote no ledger entry; its log is preserved. Replacement tokens are now `np.resize`d. | Check script only. The rerun passed. |
| D2 | `srs_bench.py` (before it ever ran): decoding latency in fp32 for every condition, so the GRU/router path matches the fp32 router of training and evaluation. Prefill stays bf16. | Latency script only. |
| D3 | `srs_driver.py`: after all 9 CALIB-SELECTION evaluations finished, `aggregate()` raised `KeyError: 'train_flops'`, and the uncaught exception left `status.json` at "running". Fixed (optional field; exceptions now set `needs_attention`). The driver was rerun and reused the 9 registered evaluations; none was repeated. | Bookkeeping only. No rule, candidate or criterion changed. |

**Other notes**
- **Failed or aborted runs:** none. There were 0 contaminated runs and 0 resumes needed; `failures.tsv` is empty. All
  75 `srsv1` GPU ledger entries are COMPLETE or PASS.
- **Cosmetic:** in `final_analysis.json` the field `extra_compute_le_10pct` is serialized as the string `"True"` (a
  numpy bool). The value is correct.
- **CPU work during GPU jobs:**
  - Only JSON summarization scripts, which read result files; no model computation.
  - The post-hoc diagnostic ran after the driver finished.
- **Overlap with a foreign process:** none observed.

## 9. Conclusion, negative results and open questions

**Conclusion.** The bounded, preregistered search found **no evidence that sequence routing beats token routing**,
and strong evidence that it is worse:
- it ran through 3 generations, 16 unique candidates, fixed and dynamic families, 5 summary/state variants and
  3 seeds per finalist;
- under matched data, tokens, initialization, experts (4 + shared) and FLOPs, the best sequence router is 0.116
  nats/token (PPL 76.7 vs 68.3) worse than TOKEN-4 at 100M tokens;
- the deficit grows with training.

Interpretation, following the prereg list:
- **Sequence is worse:** tokens require finer expert allocation than the tested sequence routers provide.
- **Fixed beats dynamic:** predictable regular segments are easier to optimize, and adaptive causal boundaries added
  nothing useful.
- **No speed benefit:** sequence routing was not faster. Router calls were not the bottleneck, and this
  implementation's sequential components made training and prefill slower.

**Established negative results.** These are listed under answer 13.

**Unresolved**
- **Router confidence.** Whether temperature < 1, a router-LR multiplier, larger router initialization, or
  normalization of the pooled summary would sharpen sequence routes and recover part of the gap. These levers were
  allowed but not reached by the frozen deterministic Gen-2 rule (§7 of the prereg), which explored segmentation and
  summary axes. Given a deficit of +0.116 that grows with training, it is unlikely they would yield the required
  −0.02, but this is untested.
- **Hybrid designs.** Whether finer-grained schemes would help, such as a sequence-level candidate pool with token
  selection inside it. These are not sequence routing under this protocol and are out of scope.
- **Scale and implementation.** Behaviour at larger models, more experts or longer training; and whether a fused
  implementation would make sequence routing faster in practice.
- **Small effects.** Nothing here addresses effects much smaller than the observed gap. They are irrelevant to the
  conclusion, because the gap is about 0.12 nats with a CI width of about 0.001.

## 10. Paths, hashes and inventory

**Artifacts** (`artifacts/sequence-routing-search-v1/`)

| File | sha256 |
|---|---|
| `search_partition.json` | `81454b91c0be330f6ae77f8b03a2fa494d9ba16317ae033aa25bdcd532fb104a` |
| `candidates.tsv` | `ad7cea0704c96bdb3a5026240c0554e50a34615dc8579d3d9bccea310dc821b8` |
| `calib_search_results.json` | `b5670fa533d78bb9b3e894024512b3b5a1b741418bf2e5424d1488c7f0e12ff7` |
| `calib_selection_results.json` | `91b6f8a39ad01eb7a9d5b86dbfb1d9b8651a82d2a5a27f1b6f35c10d67a06001` |
| `final_analysis.json` | `f56244fd694c2f739ac6a4c6b4c4750fa3d8133ddda7c976e7b75aace11d9e8f` |
| `search_frozen.json` | `dc150bc3254e2c6186c8b839fc2b1eeec8e41fb18349022bd3043d410360a354` |
| `final_frozen.json` | `8bd974e02e2358afee1592b8b49c923e15748f1337ad8753a0ee8ab1ebcac4bb` |
| `budget.json` | `8a90fac56192505d2901a9ccedc42937a720b9ac29230aeb6703b32db2195622` |
| `gen{1,2,3}_results.json` | `1b23a09b…`, `c756c292…`, `56c31514…` |
| `gen2_specs.json`, `gen3_specs.json`, `final_specs.json` | `7e51a683…`, `d7fd1298…`, `82f25350…` |
| `val_doc_ce.json` | `e5b3ba94f7c5909b77cbb9884090b054a882578de853d12ef072c38207648c79` |
| `latency_20261106T154738Z.json` | `91cd51489a233ab04d1de09a317d808f8d7d106349d9d25df589cb1016ef199b` |
| `run_inventory.json` (every config, run, checkpoint and evaluation hash) | `003e5707eda3edd5633ce6019909848b002a7f955337f98e33737380e87b7cdb` |
| `protected_inventory.json` / `.verify_20261106T154803Z.json` | `563f1926…` / `e16743ee…` (1,101/1,101 match) |
| `gpu_checks_20261106T064342Z.json`, `cpu_tests_20261106T063811Z.json` | `29039dba…`, `f371de0a…` |
| `manifest.json` | `22269cdc…` (prereg hash, partition hash, inventory hash, code hashes at prereg) |
| `posthoc_logit_scale.json` (exploratory) | `02dda783…` |

**Configs:** `configs/sequence-routing-search-v1/{gen1/seed98, gen2/seed{98,99}, gen3/seed{97,98,99},
final_B100M/seed{0,1,2}}/`, each with an `INDEX.json` (per-file sha256, matched-field check passed).

**Runs:** 55 training runs (13 + 18 + 15 + 9). The registries `.tmp/sequence_routing_v1/{gen1,gen2,gen3,final}_runs.tsv`
are hashed in `run_inventory.json`. The 9 final checkpoints are `runs/20261106T*_srsv1_train_final_B100M_s{0,1,2}_*/model.pt`.

**GPU-time ledger** (`ledger/gpu_time_ledger.tsv`, purpose `srsv1 …`)

| Stage | Entries | GPU-s | GPU-h |
|---|---|---|---|
| GPU checks | 1 | 169 | 0.05 |
| Generation 1 | 13 | 1,003 | 0.28 |
| Generation 2 | 18 | 3,353 | 0.93 |
| Generation 3 | 15 | 8,441 | 2.34 |
| CALIB-SELECTION evaluations | 9 | 126 | 0.04 |
| Final training (100M) | 9 | 17,629 | 4.90 |
| Validation evaluations | 9 | 813 | 0.23 |
| Latency benchmark | 1 | 38 | 0.01 |
| **Total** | **75** | **31,572** | **8.77** |

**Driver:** `srs_driver.py` in tmux session `srsv1`. Final status `.tmp/sequence_routing_v1/status.json`: `stage:
complete, state: done`. Ledger: `SRS_V1_DRIVER_DONE`.

**Code (sha256 prefixes):**

| File | sha256 |
|---|---|
| `seqroute.py` | `2bcd45ed` |
| `srs_run.py` | `36f940db` |
| `srs_select.py` | `1424e17f` |
| `srs_train.py` | `36b95bad` |
| `srs_evaluate.py` | `b0ef56a0` |
| `srs_driver.py` | `170b638e` |
| `srs_gpu_checks.py` | `a8bcdb13` |
| `srs_bench.py` | `9aba2a28` |
| `srs_analyze.py` | `14b45850` |
| `srs_make_configs.py` | `6bf6c0de` |
| `srs_partition.py` | `b18a68e6` |
| `srs_inventory.py` | `8e279de8` |
| `srs_report_tables.py` | `4770120c` |
| `srs_posthoc_logit_scale.py` | `812ae389` |
| `tests/test_sequence_routing_v1.py` | `dc538473` |

**TERMINAL STATE: CSR-SEQUENCE-ROUTING-SEARCH-V1 COMPLETE — NEGATIVE.** No sequence-routing candidate beat TOKEN-4 at
any stage. On three-seed validation at 100M tokens, the best fixed router is +0.116 and the best dynamic router +0.144
nats/token worse. Neither the substantial-improvement rule nor the non-inferiority rule is met.
