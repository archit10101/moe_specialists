# csr-value-routed-compute-v1 — report (screening pilot)

**Verdict: the diagnostic gate FAILED. Experiment A stopped at the gate (protocol A.5), so no pilot, controller or
decode benchmark was run.** This is a screening-quality negative result on one 5M-token checkpoint, not a
definitive one.

Protocol: `docs/VALUE_ROUTED_COMPUTE_V1_PROTOCOL.md` (sha256 `8431e2aa…54df`, written and hashed before any result).
Machine-readable: `artifacts/value-routed-compute-v1/final_analysis.json`, `artifacts/value-routed-compute-v1/diagnostic_20261109T075311Z.json`.

## 1. Setup as run
- **Data.** The existing fwe-rep-v1 train and calib documents (same ids), re-tokenised at `max_len` 8192 from the
  existing raw parquet. The rows were filtered by id inside Arrow, so no validation or test text was materialised.
  Train is the first 24.0M valid tokens in `train_order(17)`; calib is all 8,179 docs (8.10M tokens; 57% of calib
  tokens sit at document position ≥ 512, and 35% at ≥ 1024). CALIB-SEARCH and CALIB-SELECTION use a 70/30
  document-hash partition (salt `csr-vrc-v1-calib-partition`). CALIB-SEARCH has 791 packed 8192-token rows,
  44,361 non-empty 128-token blocks and 5,626,587 valid tokens. CALIB-SELECTION, validation and test were never
  read. Manifest: `artifacts/value-routed-compute-v1/data/fwe-rep-v1-L8192/manifest.json` (splits sha `7969f862…`).
- **Context = 8192.** The longest allowed context passed every check
  (`artifacts/value-routed-compute-v1/gpu_checks_*.json`):
  - flex block-sparse versus dense reference, max abs logit diff: 2.9e-6 (fp32 REF), 3.1e-6 (fp32 mixed policy),
    0.016 (bf16);
  - causality: exactly 0 change before a perturbed token, for both REF and a random mixed policy;
  - matched path: the fallback equals REF where local = global (0.016 in bf16);
  - NULL-slot accounting: exact (k × tokens × 7);
  - memory: 26.2 GiB for a 2×8192 train step, 16.6 GiB for a 4×8192 prefill;
  - resume: bitwise identical (`resume_check.json`).
- **Baseline.** No compatible long-context checkpoint existed (every earlier checkpoint is 1024-context), so one
  FULL-TOKEN4 was trained for 5,029,729 valid tokens at 8192: 175 steps of 4 rows × 8192, with the project recipe.
  Run: `runs/20261109T075009Z_vrc_train_diag_FULL-TOKEN4_L8192_s0` (≈40 s). The model is very undertrained
  (CALIB-SEARCH CE 6.70), and that limits what the diagnostic can show (see §4).
- **Candidate path family** (`src/csr/vrc_model.py`, same weights):
  - document-causal local attention, W = 512, in every layer;
  - layers 3 and 7 may run global attention per 128-token query block (exact block-list union, so no dense
    attention for local blocks);
  - the shared expert always runs;
  - k ∈ {1, 2, 4} routed experts per block, by fixed four-slot dispatch. Top-4 is always computed over all 32
    experts; NULL slots are never dispatched; the expert GEMMs are grouped padded bmm with no Python loops.

## 2. Counterfactual measurements (CALIB-SEARCH, all blocks, shared weights)
CE is in nats/token. Δ is against REF = FULL-TOKEN4 (full attention in all 8 layers, top-4).

| Option | ΔCE vs REF | routed-expert evals vs top-4 | prefill 4×8192 p50 (ms) | time / REF |
|---|---|---|---|---|
| REF (FULL-TOKEN4) | 0 (CE 6.7036) | 1.00 | 43.98 | 1.000 |
| L4 (local everywhere, top-4) | −0.0011 | 1.00 | 46.56 | 1.059 |
| L2 | +0.0107 | 0.50 | 47.67 | 1.084 |
| L1 | +0.0397 | 0.25 | 47.84 | 1.088 |
| G4 (global in 3, 7; = full fallback) | −0.0010 | 1.00 | 47.25 | 1.075 |
| G2 | +0.0109 | 0.50 | 48.26 | 1.097 |
| G1 | +0.0399 | 0.25 | 48.49 | 1.103 |

Times are the mean of two passes of 20 timed repetitions after 5 warm-ups, CUDA-synchronised, bf16, no grad, on
identical rows.

Per-block labels (nats per valid token of the block; token-weighted mean, median, fraction > 0):

| Label | mean | median | > 0 | > 0.01 |
|---|---|---|---|---|
| V_context = CE_local − CE_global | −0.00018 | 0.0000 | 29.6% | 0.4% |
| V_context, blocks starting at doc position ≥ 512 (25,108 blocks) | −0.00032 | – | 46.9% | – |
| V_2 = CE_top1 − CE_top2 | +0.0290 | +0.0292 | 95.7% | 87.7% |
| V_4 = CE_top2 − CE_top4 | +0.0119 | +0.0121 | 78.6% | 55.8% |
| V_joint = CE_cheap(L1) − CE_full(G4) | +0.0407 | +0.0413 | 92.6% | 86.7% |

So long-range attention is worth essentially nothing to this checkpoint: local-only is slightly better than full.
Extra experts help almost everywhere: going from 4 to 2 hurts 79% of blocks, and from 2 to 1 hurts 96%.

## 3. Gate (fixed thresholds) — FAIL
The oracle frontier uses the preregistered λ sweep (cost = measured option time), and the CE criterion is against REF.

| Criterion | Required | Best achievable at ΔCE ≤ +0.010 | Met? |
|---|---|---|---|
| CE degradation | ≤ +0.010 | −0.004 (oracle min-CE point) | yes |
| routed-expert evaluations | ≤ 75% | 88.3% on the preregistered time-cost frontier (see note) | no on the frontier |
| global query blocks (layers 3, 7) | ≤ 50% of fallback | 0% (all-local point, ΔCE −0.0011) | yes |
| projected end-to-end prefill time | ≤ 90% of REF | ≥ 105.9% at **every** point | **no** |

- No point passed the projection, so the exact re-evaluation list is empty.
- **Measured reason.** Every option in the candidate path family is *slower* than FULL-TOKEN4 at 8192, by 5.9% to
  10.3%. So no mixture of options can project below 1.059 × REF, whatever the controller, oracle or cost weighting.
  The time criterion is decisive on its own.
- **Note on the expert criterion.** The preregistered sweep weights *time*, and in this implementation fewer experts
  measured slightly slower (L1 > L4). So that sweep never explores low k, and the 88.3% figure is an artefact of
  the cost choice. The uniform options show the trade-off itself: L2 halves the routed-expert evaluations at
  +0.0107. An expert-count-weighted frontier would probably satisfy criterion 2. This is reported only to place
  the failure correctly; it changes nothing, because criterion 4 cannot be met.
- The secondary frontier against the candidate's own fallback (G4) gives the same picture: best ΔCE −0.003,
  experts 88.3%, time ≥ 1.059.

## 4. Post-hoc timing breakdown (explanatory only; the gate decision is unchanged)
`artifacts/value-routed-compute-v1/posthoc_timing_20261109T091508Z.json`. Prefill of 32,768 identical CALIB-SEARCH tokens per batch (B × C), bf16, no grad, 5 warm-up + 20 timed, CUDA-synchronised. Speed-up is REF p50 / option p50; < 1 means slower than FULL-TOKEN4.

| context (batch) | REF p50 / p95 ms | REF tok/s | L1 p50 ms (speed-up) | L4 p50 ms (speed-up) | G4 p50 ms (speed-up) | peak VRAM (all) |
|---|---|---|---|---|---|---|
| 1024 (32) | 41.77 / 42.07 | 784,405 | 47.52 (0.879×) | 45.84 (0.911×) | 45.83 (0.911×) | 3.50 GiB |
| 2048 (16) | 42.56 / 42.86 | 769,880 | 47.64 (0.893×) | 46.14 (0.922×) | 46.31 (0.919×) | 3.50 GiB |
| 4096 (8) | 43.81 / 44.00 | 747,989 | 47.48 (0.923×) | 46.37 (0.945×) | 47.06 (0.931×) | 3.50 GiB |
| 8192 (4) | 43.98 / 44.17 | 744,982 | 47.67 (0.923×) | 46.31 (0.950×) | 47.17 (0.933×) | 3.50 GiB |

Components at 32,768 tokens (p50 ms):
- block-mask construction: full 0.75, local 0.80, mixed 1.14. The candidate builds two masks (local + mixed) where REF builds one.
- routed-expert dispatch per MoE layer: original top-4 1.59; four-slot NULL dispatch k=4 1.89, k=2 1.87, k=1 1.80.
  Dropping from 4 to 1 expert saves only 5% of dispatch time, because the dispatch is bound by sorting, scatter/gather and launch overhead rather than expert FLOPs. The NULL bookkeeping costs more (+19% vs the original dispatch) than the expert FLOPs it removes.
- shared expert 0.18; tied LM head 5.05 (the single largest dense op).

Conclusion: at d = 192 the forward pass is overhead-bound, so removing attention or expert FLOPs does not reduce latency.
Every candidate path is 5–14% slower than FULL-TOKEN4 (speed-up 0.879–0.950×) at every tested context (1K–8K).
Decoding was not benchmarked: no pilot was run, and the rolling-KV decode path was not implemented.

## 5. Answers to the required questions
- **Diagnostic gate.** FAIL (time criterion impossible; expert criterion not met on the preregistered frontier).
- **Accuracy difference.**
  - Full fallback G4 versus FULL-TOKEN4: −0.0010 nats (local W512 in six layers costs nothing at 5M tokens).
  - Cheapest path L1: +0.040.
  - Oracle min-CE operating point: −0.004 (an oracle that uses targets, so an upper bound).
- **Measured speed difference.** At 8192 (4×8192 prefill), every candidate option is 5.9–10.3% slower than
  FULL-TOKEN4. Post-hoc measurements at 1K / 2K / 4K / 8K prefill (§4): speed-up 0.879–0.923× (L1), 0.911–0.950× (L4), 0.911–0.933× (G4), i.e. 5–14% slower everywhere. Decoding was not benchmarked (no pilot).
- **Computation actually used.** Not applicable, because no controller ran. Options range from 25% to 100% of
  top-4 routed-expert evaluations and from 0% to 100% of global query blocks in layers 3 and 7.
- **Did the controller help?** Not tested; the gate failed first. Since V_context ≈ 0, a context controller would
  have nothing to save on this checkpoint. Since V_2 and V_4 are positive almost everywhere, an expert controller
  would trade accuracy for evaluations that are not faster here.

## 6. Limitations
- The diagnostic checkpoint is a 5M-token model. clav1 showed that the value of long-range attention appears
  only later in training (its local-attention advantage reversed at about 45–75M tokens), so V_context here
  understates the value of long-range attention in a trained model.
- This argument does not rescue the gate, which fails on measured time. The model is small (d = 192), and its
  per-token cost is dominated by the tied 50,304-row LM head, the dense layer and the shared experts. The extra
  mask construction and NULL-slot bookkeeping cost more than the attention and expert FLOPs they remove.
- One seed, one checkpoint, prefill only.
- The rolling local KV cache and the decode path were not implemented, because the protocol stops at the gate.

## 7. Integrity
- One GPU job at a time; occupancy was checked before every launch; GPU time is in `ledger/gpu_time_ledger.tsv`.
- CPU tests ran with CUDA hidden (`tests/test_vrc_m2d2_v1.py` plus the full suite; see final_analysis.json).
- CALIB-SELECTION, validation and locked test were not read.
- No existing file was modified (`protected_inventory.json` verify).
- Deviations for A: none (no `deviations.jsonl` written). The post-hoc timing (§4) is labelled explanatory and does not change the gate.

### Integrity numbers (final audit `artifacts/value-routed-compute-v1/final_audit_20261109T091820Z.json`)
- **GPU time (this session, all jobs including checks, failures, warm-ups and benchmarks):** 5,088 s = 1.41 GPU-h
  in total: Experiment A 227 s, Experiment B 4,833 s, shared GPU checks 28 s.
- **Wall time:** about 1 h 45 min (02:33 to 04:18 EST); all GPU work ended at 04:15, before every deadline.
- **Peak VRAM** (allocator):
  - A: 20.7 GiB in the 8192 training run, 26.2 GiB in the GPU check step that kept its graph;
  - B: 22.0 GiB;
  - post-hoc prefill: 3.5 GiB.
- **Disk:** 128.8 GiB free at the end.
- **Tests:** CPU tests ran with CUDA hidden. The full suite passed (620 tests); the new `tests/test_vrc_m2d2_v1.py`
  has 16 tests, all passing. GPU checks passed (`gpu_checks_20261109T*.json`).
- **Protected inventory:** 3,004 pre-existing files verified unchanged.
- **Incomplete runs:** none. The two INTERRUPTED run directories are the deliberate `--max-steps 4` halves of the
  resume-equivalence checks.
- **Locked test:** untouched. No test tokens were written; the M2D2 test members were never decoded; the
  fwe-rep-v1 test files were never opened.
