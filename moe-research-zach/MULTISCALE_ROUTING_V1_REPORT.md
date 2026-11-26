# csr-multiscale-routing-v1: report

**Causal multiscale context-conditioned token routing.** An exploratory, prospectively preregistered follow-up to
csr-representative-v1 and specialization-audit-v1.

**Terminal state: STOPPED AT PREREGISTERED GATE (Branch A).** Neither the broad-context coefficient β nor the
local-context coefficient α survived neutral calibration. Under the frozen branch rule (prereg §8) the experiment
stops before main training.

**What did not run:**
- benchmark and budget selection;
- the pilot;
- main training and full validation;
- the primary contrasts;
- the secondary analyses (router diagnostics, latency, specialization).

Each is marked **NOT RUN** below. The val split was never used for any model evaluation, and the locked test split
was never accessed.

## Answers to the required questions

1. **Did α or β survive neutral calibration?** No.
   - **β, broad context:**

     | β | Calib CE vs β = 0 |
     |---|---|
     | 0.25 | +0.00024 |
     | 0.5 | +0.00512 |
     | 1.0 | −0.00044 |

     None reached the required ≤ −0.002 nats/token with a beneficial sign in both seeds. β = 1.0 was beneficial in
     both seeds, but only by 0.0004–0.0005, about 0.2 noise standard errors.
   - **α, local context:** every nonzero α was worse than α = 0 in both seeds.

     | α | Calib CE vs α = 0 |
     |---|---|
     | 0.25 | +0.0027 |
     | 0.5 | +0.0027 |
     | 1.0 | +0.0070 |

   No candidate was rejected for collapse, unused experts, non-finite values or a > 0.10 loss. They simply did not
   qualify, so α = β = 0 by rule.
2. **Which branch ran?** **Branch A** (α = 0, β = 0): stop and report the negative calibration.
3. **Which main budget was selected, and why?** None. **NOT RUN.** Budget selection happens after calibration, and
   Branch A ends the study before it.
4. **Did broad late context beat TOKEN?** **NOT RUN** at main scale. In calibration (5M tokens, calib split),
   BROAD-LATE vs the matched β = 0 run was:

   | β | BROAD-LATE vs β = 0 (two-seed mean) |
   |---|---|
   | 0.25 | −0.0015 |
   | 0.5 | +0.0028 |
   | 1.0 | −0.0011 |

   All of these are within the measured noise floor (item 12). No evidence that it helps.
5. **Did late placement beat equal-layer-count early placement?** **NOT RUN** as a contrast. Descriptively, late was
   a little less harmful than early at every β (two-seed means vs zero):

   | β | Early | Late |
   |---|---|---|
   | 0.25 | +0.0020 | −0.0015 |
   | 0.5 | +0.0074 | +0.0028 |
   | 1.0 | +0.0002 | −0.0011 |

   The early − late differences are 0.001–0.005 nats, comparable to run-to-run noise. Not interpretable as evidence.
6. **Did local context help independently?** No. Every α > 0 was worse than α = 0 in both calibration seeds:
   - α = 1.0 was 0.0070 nats worse, about 4.6 noise SEs, with consistent sign across seeds and boundary types.
   - α = 0.25 and 0.5 were each about 0.0027 worse, about 1.8 SE.
7. **Did semantic boundaries beat fixed and random boundaries?** **NOT RUN** as a main contrast. Within calibration
   there is no consistent pattern. LOCAL-SEMANTIC's difference vs zero was:

   | α | LOCAL-SEMANTIC vs zero |
   |---|---|
   | 0.25 | +0.0006 |
   | 0.5 | +0.0055 |
   | 1.0 | +0.0046 |

   It was the least harmed boundary type at α = 0.25 and 1.0, and the most harmed at α = 0.5. All values are within
   noise.
8. **Did the combined hierarchy beat broad-only and local-only?** **NOT RUN.** HIER conditions exist only in Branch D.
9. **Did contextual direction beat its RMS-matched scaling control?** **NOT RUN.** SCALE-ONLY controls exist only in
   Branches B, C and D.
10. **What confidence, margin and gradient differences remained after that control?** **NOT RUN.** One
    calibration-level observation from training logs: at step 0, from identical weights, adding context raised the
    load-balance and z-loss terms monotonically with the coefficient. For seed 98:

    | Setting | Load-balance loss | z-loss |
    |---|---|---|
    | No context | 7.271 | 85.95 |
    | α = 1 | 7.63 | 87.32 |
    | β = 1, late | 7.43 | 86.50 |

    So context made the initial routing less balanced and its logits larger. This is a mechanism check only, not the
    preregistered control comparison.
11. **Did functional specialization improve without worse CE?** **NOT RUN.** The specialization extraction and
    counterfactual pipeline was implemented and tested (brute-force equivalence) but never run on checkpoints.
12. **Were results consistent across seeds?** The decision was consistent: no candidate qualified in either seed.
    The calibration also exposed a material **noise floor**:
    - The five zero-coefficient runs per seed are the identical TOKEN computation. They have the same init hash and
      data, and bit-identical step-0 loss.
    - Yet their final calib CE differs, through nondeterministic GPU backward kernels:
      - range 0.0081 nats (seed 98) and 0.0046 nats (seed 99);
      - pooled within-seed SD 0.0026.
    - The preregistered selection margin (0.002) is smaller than one run-to-run SD. A 4-run or 6-run aggregate has a
      noise SE of about 0.0015–0.0019.
    - **Consequence:** only effects of roughly 0.005 nats or more are resolvable at this calibration scale. Small
      true benefits (< 0.003) could not have been detected; they are not excluded, only undetected.
13. **Compute and resources**

    | Item | Value |
    |---|---|
    | Total GPU time (ledger, `msv1`) | **2,979 GPU-s = 0.83 GPU-h** (cap 20 h, target 12 h) |
    | GPU correctness checks | 22 s |
    | Semantic gate | 138 s |
    | Calibration runs | 40 runs, 2,819 s, about 67–75 s each |
    | GPU wall span | 3,066 s, 03:54–04:45 on 2026-11-02 |
    | Mean training throughput, zero runs | 110.9k valid tokens/s |
    | Mean training throughput, β runs | 105.1k valid tokens/s |
    | Mean training throughput, α runs | 101.7k valid tokens/s |
    | Peak VRAM | 24.9 GiB reserved, 21.0 GiB allocated (ceiling 56) |
    | Free disk on `/` at the end | 127.7 GB (≈ 119 GiB) |
    | GPU 0 at the end | idle (0 processes) |

    The context paths cost about 5–8% in calibration-scale throughput. No foreign GPU process appeared.
14. **Failures, fixes, deviations and resource contingencies.** These are listed in full in §6. There were no failed
    or aborted GPU runs and no resource contingencies.
15. **What conclusion is justified, and what remains unknown?** See §7.

## 1. Implementation (frozen)

**Router facts.** For MoE layer l and token t, the router reads h = RMSNorm₂(x). It is a bias-free
`Linear(192, 32)`; the absent bias is recorded.

**Logits**

```
z_token = W h_t
c_local = center(W · mean(h[chunk_start : chunk_start + min(chunk_pos, 16)]))            (0 at chunk_pos = 0)
c_broad = center(W · mean(h[t − n : t])) · min(n / 512, 1),  n = min(pos_in_doc, 1024)  (0 at n = 0)
z_raw   = z_token + a_l·α·c_local + b_l·β·c_broad
```

**Selection.** Global softmax top-4 with global-probability gates, dropless, plus the shared expert. There are no
candidate pools, no new parameters and no detach on the context path. The load-balance and z-losses use the routing
logits actually used for selection.

**Schedules.**
- BROAD-EARLY: layers 1–3.
- BROAD-LATE: layers 5–7.
- LOCAL: layers 3–7.
- HIER: local at 3–4, local + broad at 5–7.

**Scale controls.** SCALE-ONLY controls are detached and centered-RMS-matched. They were implemented and tested,
but not run.

**Characterization.**
- Pooling happens after RMSNorm and the router is linear and bias-free, so W·mean(h) = mean(W·h). The mechanism is
  multiscale routing-logit smoothing.
- Measured error: ≤ 2.4e-7 (fp32, full-size, every condition) and < 1e-12 (fp64 tests).

**Code (all new files; no existing file was modified):**
- `src/csr/multiscale.py`, `ms_select.py`, `ms_run.py`, `ms_audit.py`;
- `ms_train.py`, `ms_evaluate.py`, `ms_make_configs.py`, `ms_decide.py`, `ms_driver.py`;
- `ms_gpu_checks.py`, `ms_semantic_gate.py`, `ms_inventory.py`;
- `ms_diag.py`, `ms_spec_extract.py`, `ms_spec_analyze.py`, `ms_analyze.py`;
- `tests/test_multiscale_v1.py`.

## 2. Preregistration and integrity

**Preregistration**
- `docs/MULTISCALE_ROUTING_V1_PREREG.md`, sha256 `5ab16f8ed3e17598a0140fad2e0aaa07fde082cffb6ddd0798123d85acdb4a73`.
- Frozen before implementation; ledger entry `MS_V1_PREREG`.
- Unchanged at the end, and never amended.

**Protected inventory**
- `artifacts/multiscale-routing-v1/protected_inventory.json` covers 317 files.
- Its scope:
  - the data, except the test split (excluded by name and never opened);
  - the semantic cache 8e7edc7e14b3c880 and the F-v3 random cache 0c57f77e51ea3734;
  - rep-v1 configs, checkpoints, evals and analyses;
  - audit artifacts, docs, reports and all pre-existing source files.
- Post-run verification `protected_inventory.verify_20261102T094621Z.json`: **317/317 match, 0 bad.**

**Pre-edit snapshot.** `.tmp/multiscale_v1_snapshot_20261102T083911Z/` (no existing files were to be modified, and
none were).

## 3. Gates and tests

**CPU tests (CUDA hidden)**
- The full suite gave **234 passed**: 137 pre-existing, unchanged, plus 97 new.
- 5 counterfactual brute-force tests were added afterwards; they pass.

**GPU correctness checks** (`gpu_checks_20261102T085415Z.json`): **all pass.**
- Tiny model in fp64, for all 12 conditions:
  - zero coefficients equal TOKEN exactly;
  - future invariance holds (difference 0.0);
  - incremental decoding equals the full sequence to within 2.2e-16;
  - scale controls keep TOKEN's top-4 and match the reference centered RMS to within 6e-17.
- Full-size model, every condition: finite fp32 loss and gradients, exactly 4 distinct experts, finite bf16
  forward.

**Semantic-causality gate** (`semantic_gate_20261102T085438Z.json`): **pass.**

| Check | What it tested | Result |
|---|---|---|
| G1 | Online prefix-only re-segmentation of 600 sampled documents (200 each of train, calib and val), streamed **without document-length knowledge** | All 2,678 cached boundaries reproduced; 0 mismatches, 0 numerical ties |
| G2 | Each boundary b becomes available only while consuming token b−1, at a completed sentence end or forced max-length close | 0 violations |
| G3 | Future-text perturbation (100 calib docs) | Earlier boundaries unchanged |
| G4 | Full-size HIER-SEMANTIC model on perturbed docs | Earlier logits, z_route and top-4 bit-identical |
| G5 | Incremental `ms_step` with online streaming boundaries (10 docs, 5,888 tokens) | Max logit difference 2.1e-6; top-4 exact; boundaries exact |
| G6 | Changing only later chunk endpoints | Earlier outputs bit-identical |
| — | Fixed cache vs `fixed_starts(L, 128)` | Equal for every train, calib and val doc |

## 4. Calibration

**Setup**
- 5M valid training tokens per run: 194 steps, 5,008,362 tokens.
- Seeds 98 and 99 (train order 115 and 116), with identical initialization within a seed.
- Evaluated on the full calib split: 8,179 docs, 5,270,143 tokens.
- 40 runs, all COMPLETE.
- Inventory: `artifacts/multiscale-routing-v1/calibration_runs.tsv`. Decision: `calibration.json` (sha256
  `d79d7a73…`).

**β, broad context (aggregate over BROAD-EARLY and BROAD-LATE × 2 seeds)**

| β | Aggregate calib CE | Δ vs 0 | Δ seed 98 | Δ seed 99 | Δ early | Δ late | Qualifies |
|---|---|---|---|---|---|---|---|
| 0 | 6.55338 | — | — | — | — | — | — |
| 0.25 | 6.55362 | +0.00024 | +0.0022 | −0.0017 | +0.0020 | −0.0015 | no (gain, seeds) |
| 0.5 | 6.55850 | +0.00512 | +0.0050 | +0.0052 | +0.0074 | +0.0028 | no |
| 1.0 | 6.55294 | −0.00044 | −0.0005 | −0.0004 | +0.0002 | −0.0011 | no (gain < 0.002) |

**α, local context (aggregate over LOCAL-FIXED, LOCAL-SEMANTIC and LOCAL-RANDOM × 2 seeds)**

| α | Aggregate calib CE | Δ vs 0 | Δ seed 98 | Δ seed 99 | Δ fixed | Δ semantic | Δ random | Qualifies |
|---|---|---|---|---|---|---|---|---|
| 0 | 6.55233 | — | — | — | — | — | — | — |
| 0.25 | 6.55505 | +0.00271 | +0.0037 | +0.0017 | +0.0035 | +0.0006 | +0.0040 | no |
| 0.5 | 6.55500 | +0.00266 | +0.0040 | +0.0013 | +0.0019 | +0.0055 | +0.0006 | no |
| 1.0 | 6.55934 | +0.00701 | +0.0090 | +0.0050 | +0.0101 | +0.0046 | +0.0063 | no |

**Rejection checks**
- No constituent run exceeded +0.10 vs its zero run.
- No collapse: the maximum calib expert share in any layer was 0.061, against a 0.5 threshold.
- No unused experts in any run.
- Training losses for the final 20% of steps move in the same direction as the calib CE differences
  (`analysis.json` → `descriptive_effect_vs_noise`).

**Selection.** α = 0 and β = 0 give Branch A, decided by `ms_decide.py` with the frozen rule. The decision is
deterministic, and the unit tests cover it.

## 5. Not run (Branch A)

- Benchmark and budget selection (`feasibility.json`).
- The 10M-token pilot (`pilot.json`).
- Main training at any budget; full validation.
- The 12 primary contrasts and the paired document bootstrap.
- Router diagnostics and latency (`ms_diag.py`).
- Specialization extraction and analysis (`ms_spec_extract.py`, `ms_spec_analyze.py`).
- The final main analysis (`ms_analyze.py`).

The code for these stages exists. Of these scripts, only `ms_spec_extract.py` was smoke-tested (on CPU); none ran on
real checkpoints.

**Terminal-stage outputs.**
- `artifacts/multiscale-routing-v1/analysis.json` (sha256 `42417e78…`) records the calibration tables, the noise
  floor, the compute and the not-run list.
- Driver status is in `.tmp/multiscale_v1/status.json` (`state: terminal`, `BRANCH_A`). The driver exited with code
  2; the ledger entry is `MS_V1_TERMINAL`.

## 6. Failures, fixes and deviations

Logged in `artifacts/multiscale-routing-v1/deviations.jsonl` and the safety ledger.

1. **Test-only fixes before the passing suite**
   - A positional-argument `argsort` call was rewritten with keywords.
   - The exception class was imported under an alias to avoid pytest collection.
   - No implementation change.
2. **Process deviation: CPU work during a GPU job**
   - What ran: a ~1 s tiny-model pytest and a 24 s CPU smoke of `ms_spec_extract.py`.
   - When: while the semantic-gate GPU job held GPU 0. SAFETY.md's one-job-at-a-time policy forbids this.
   - Effect: none on the gate, which is correctness-only. Logged as `MS_V1_INCIDENT`.
   - Afterwards: no further CPU model work overlapped GPU jobs.
3. **Gate output names were timestamped** before first use, so a preserved failed gate cannot block a rerun. This
   was implementation-only.
4. **Driver analysis stages were added to the driver code** while calibration ran. The running process never used
   them, and they were not reached.
5. **Failed GPU runs:** none. There were no OOMs, contaminations, aborts or failed evaluations.

**Smoke runs** (preserved, outside the experiment's inventories):
- `runs/20261102T084608Z_msv1_train_cpusmoke_HIER-SEMANTIC`;
- `runs/20261102T084617Z_msv1_train_cpusmoke_SCALE-ONLY-HIER`;
- `runs/20261102T085637Z_msv1_specextract_smoke_HIER-SEMANTIC_cpusmoke`.

**Limitations**
- **Noise floor.** It sits above the preregistered 0.002 margin; see item 12.
- **Scale.** Calibration is short (5M tokens, about 4% of a 100M run) and in the early high-loss regime (CE ≈ 6.55).
  An effect that would emerge only later in training cannot be seen here. The prereg deliberately stops instead of
  rescuing the experiment.
- **Random boundaries.** The F-v3 random boundaries condition on document length (a content-free but non-causal
  covariate). The artifact is unchanged; this did not matter, because no random-boundary condition qualified.

## 7. Conclusion and what remains unknown

**Justified conclusion (restrained)**
- In this setup (about 29M parameters, 14.4M active, 32 experts with top-4), adding causal context to token routing
  did not help at the neutral-calibration scale.
- Broad context was neutral within noise; local context was mildly harmful, clearly so at α = 1.
- The calibration rule therefore stopped the study at its preregistered gate. That holds whether the context is a
  ramped average of the full document history (β) or the mean of the current chunk's first ≤ 16 tokens (α), applied
  as centered, bias-free projections through each layer's own router. That mechanism is exactly a smoothing of the
  layer's historical router logits.
- Semantic chunk boundaries showed no advantage over fixed or matched-random boundaries in the local-context
  calibration.

**What remains unknown**
- Whether broad context helps at 100M tokens or later in training. The pilot and main study did not run.
- Whether effects smaller than about 0.005 nats exist. They are below the measured run-to-run noise at this scale.
- Whether any gain would beat the RMS-matched scaling controls, or would change confidence, margins or
  specialization. None of these were run.

**Interpretation limits.** Nothing here concerns FLOPs, latency, novelty or generality beyond this setup. The
earlier csr-representative-v1 and specialization-audit-v1 conclusions are unchanged.

**Cheapest justified next step (not run, not authorized here).** First, quantify and reduce calibration noise:
- either deterministic GPU kernels or replicate zero runs;
- with the selection margin set relative to the measured noise SD.

Then, if desired, a single prospectively preregistered β-only (BROAD-LATE) study at a longer budget, with a new
preregistration.

## 8. Paths and hashes

**Docs and configs**

| Path | SHA-256 |
|---|---|
| `docs/MULTISCALE_ROUTING_V1_PREREG.md` | `5ab16f8e…4a73` |
| `docs/MULTISCALE_ROUTING_V1_PREREG.sha256` | (contains the hash above) |
| `configs/multiscale-routing-v1/calibration/seed{98,99}/` | 40 configs; `INDEX.json` holds per-file sha256 and the matched-field check |

**Artifacts** (in `artifacts/multiscale-routing-v1/`)

| Path | SHA-256 |
|---|---|
| `manifest.json` | `f7cd51d2…` |
| `protected_inventory.json` | `1e9ac307…` |
| `protected_inventory.verify_20261102T094621Z.json` | `be2a3935…` |
| `gpu_checks_20261102T085415Z.json` | `09b7f3c9…` |
| `semantic_gate_20261102T085438Z.json` | `271ce0bc…` |
| `calibration.json` | `d79d7a73…` |
| `calibration_runs.tsv` | `3ee25b3e…` |
| `analysis.json` | `42417e78…` |
| `deviations.jsonl` | (2 records) |

**Runs.** 40 calibration runs: `runs/20261102T0857*…T0944*_msv1_train_cal_s{98,99}_{alpha,beta}*`, all listed in
`calibration_runs.tsv`.

**Ledgers.**
- `ledger/gpu_time_ledger.tsv`: purpose `msv1 …`.
- `ledger/safety_ledger.tsv`, entries `MS_V1_*`:
  - PREREG, INVENTORY, CONFIGS, TESTS, DRIVER_LAUNCH, GPU_CHECKS, SEMANTIC_GATE;
  - CALIBRATION, TERMINAL, VERIFY, ANALYSIS, INCIDENT, REPORT.

**Code (sha256 prefixes)**

| File | SHA-256 |
|---|---|
| `src/csr/multiscale.py` | `5a892d1f…` |
| `src/csr/ms_select.py` | `bc193b9a…` |
| `src/csr/ms_run.py` | `38eeb628…` |
| `src/csr/ms_audit.py` | `1b8723fd…` |
| `ms_train.py` | `37124aad…` |
| `ms_evaluate.py` | `112df5d8…` |
| `ms_driver.py` | `1e74d81f…` |
| `ms_decide.py` | `fd0c95fb…` |
| `ms_make_configs.py` | `cdd1551f…` |
| `ms_semantic_gate.py` | `979bdf59…` |
| `ms_gpu_checks.py` | `8d0b5de0…` |
| `ms_inventory.py` | `dbca8040…` |
| `ms_diag.py` | `bb445f35…` |
| `ms_spec_extract.py` | `a6a2e62d…` |
| `ms_spec_analyze.py` | `f91af7ab…` |
| `ms_analyze.py` | `50b4d4ac…` |
| `tests/test_multiscale_v1.py` | `846e5244…` |

**TERMINAL STATE: CSR-MULTISCALE-ROUTING-V1 STOPPED AT PREREGISTERED GATE — BRANCH A (both α and β failed neutral calibration; no main training).**
