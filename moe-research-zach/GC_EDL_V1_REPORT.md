# Gradient-Consistent Latent EDL (GC-EDL) v1 — Report

Experiment `csr-gradient-consistent-latent-edl-v1` (prefix `gc_`), run autonomously on 2026-11-15 from about 01:05 to
02:25 EST on one A100 80GB. Artifacts are in `artifacts/gc-edl-v1/`. The diagnostic notes are in
`GC_EDL_V1_DIAGNOSTIC_NOTES.md`.

## 1. Summary

GC-EDL stopped at its preregistered discovery-validity gate, before any MoE model was trained.

**The gradient fingerprints barely repeat within a paper.** Each fingerprint is the CountSketch of the gradient of the
scout's final MLP output projection on one 256-token view.
- Two separated views of the same paper have a cosine of 0.017 above random papers. That is 0.18 random-sd units,
  against 0.240 (2.19 sd) for a semantic embedding.
- Assignments learned from these fingerprints alone (GRAD) are near-uniform and unstable: view agreement κ 0.02, seed
  agreement κ ≤ 0.31.
- The COMBINED (semantic + gradient) clusters are stable, but they are effectively semantic clusters. They add no
  held-out gradient coherence beyond semantic-only clusters, and the selected COMBINED candidate is slightly *worse*
  (−0.0023 [−0.0028, −0.0019]).

**One preregistered engineering revision did not change this.** Raising the projection dimension from 128 to 256
left the same-paper gap unchanged.

**The preregistered rule therefore ends the experiment.** Under prereg section 5 and Revision 1, gradient-derived
pseudo-domains must beat shuffled papers *and* add functional coherence beyond semantics before any EDL training.

- **No training happened.** No screening, CALIB-SELECTION, final training or VAL evaluation was run. CALIB-SELECTION
  and VAL were never opened, and the test split was never touched.
- **The specialisation and CE questions remain untested.** The experiment produced no evidence about routing
  specialisation, functional specialisation or CE relative to TOKEN-4.

**Terminal status: `GC-EDL-V1 STOPPED — DISCOVERY INVALID`.**

## 2. What was preregistered and done

- **Prereg** `docs/GC_EDL_V1_PREREG.md`, sha256 `75591d88…43e6d`, mode 0400, hashed at 01:16 before any GC artifact
  existed. **Revision 1** `docs/GC_EDL_V1_REVISION_1.md`, sha `c861a169…a8c3`, hashed before execution. Hashes are
  also in `preregistration_hashes.json`.
- **Reuse audit (done before writing the prereg).**
  - The fle TOKEN-4 finals (s0–s2), the explicit-EDL finals (s0–s1) and the seed-100 TOKEN-4 50M reference were
    produced by the same trainer, recipe, data plan, seeds and checkpoint policy as GC-EDL. They were declared
    reusable after hash and T-EQ checks.
  - Training stopped before these were needed, so neither the T-EQ GPU check nor any reuse was executed.
- **Data isolation.**
  - **Splits.** The existing paper-hash buckets of the fspv2 M2D2 corpus were kept.
  - **Discovery exports.** `discovery/inputs_*.npz` hold only opaque IDs (sha256 of a salted paper hash), views,
    offsets, lengths and segment counts.
  - **Sealed map.** The opaque → paper map sits in `sealed/`.
  - **Firewall.** Discovery programs run under an audit-hook label firewall, and test L1 shows it blocks label paths.
  - **Partitions.** DISC-FIT/HOLD were split 80/20 by opaque-ID hash. All fitting used DISC-FIT and all selection
    used DISC-HOLD.
  - **Labels.** Labels were unsealed only after the terminal decision, at 01:56:48 (`UNSEALED.json`), and used only
    for post-hoc description.
- **Stage 1: fingerprints.**
  - **Views.** 52,592 main papers yield 183,591 views of 256 tokens. Up to four views per paper are separated by
    ≥ 256 tokens, and offsets are recorded.
  - **Scout.** The frozen fle generalist G1 (sha `3544d882…`), with θ = `blocks.9.ffn.w2.weight` (256 × 768) by the
    preregistered rule.
  - **Sketch.** A streamed TensorSketch/CountSketch with m = 128, seed 20261115; no per-example gradient is
    materialised.
  - **Exactness.** The sketch equals P·vec(explicit autograd gradient) with relative error < 1e-10 in float64 (test
    G1), and the Gram-matrix norm is exact (G2).
  - **Nuisance and semantics.** 31 nuisance features with DISC-FIT ridge residualisation (cross-fitted); the semantic
    view is bge-small-en-v1.5, independent of MiniLM, the drafts, the scout and the e5 evaluator.
- **Stage 2: prototypes.**
  - **Model.** Balanced Sinkhorn swapped-prediction prototypes with lagged cross-view gradient coherence (λ_coh 2 for
    GRAD/COMBINED, 0 for the pure-semantic SEM control), balance and confidence terms.
  - **Grid.** {SEM, GRAD, COMBINED} × K {4, 8, 16} × 3 seeds.
  - **Selection.** A label-free composite S of κ_view, κ_seed, held-out gradient coherence C and held-out
    language-distribution predictiveness LD. The smallest K within one SE of the best is chosen per representation.
- **CPU test suite** (`cpu_tests_20261115T071627Z.json`): 13/13 pass. The tests cover:
  - sketch exactness, norm, determinism, causality;
  - the label firewall, a label-free export, split isolation;
  - q constant per paper, loss independent of labels, soft loss equal to explicit EDL with one-hot q (≤ 1e-9);
  - gradient routing, balanced prototypes;
  - exact CPU resume of `gc_train.py`, and inference with every GC artifact blocked.

## 3. Results (DISC-HOLD, 9,546 papers with ≥ 2 views)

### 3.1 Fingerprints

| representation | same-paper cosine | random-paper cosine | gap [95 % CI] | gap (random-sd units) |
|---|---|---|---|---|
| SEM | 0.239 | −0.001 | 0.240 [0.236, 0.245] | 2.19 |
| GRAD, m = 128 | 0.017 | 0.001 | 0.016 [0.014, 0.019] | 0.18 |
| GRAD, m = 256 (revision 1) | 0.017 | 0.000 | 0.017 [0.015, 0.019] | 0.25 |

- **Same-paper signal.** Over all view pairs, the same-paper cosine of the functional fingerprint is 0.034 against
  −0.002 across papers, which implies a paper-mean reliability of about 0.13.
- **Projection stability.** 10-NN neighbourhoods agree only 0.11 between the 128-d and 256-d sketches.
- **Shape.** The participation ratio is 115/128, so the fingerprints are close to isotropic.
- **Nuisance.** All 31 nuisance features explain R² 0.009 of fingerprint direction. After residualisation, every
  nuisance has held-out R² ≤ 0.0001.

### 3.2 Discovery (revision 1; the original 128-d run is in `discovery_decision.json`)

| config | κ_view | κ_seed | C | LD (nats/token) | S ± SE | selected |
|---|---|---|---|---|---|---|
| SEM_K16 | 0.461 | 0.733 | 0.0225 | 0.113 | 0.889 ± 0.010 | ✓ |
| GRAD_K4 | 0.020 | 0.307 | 0.0051 | 0.000 | 0.154 ± 0.008 | ✓ |
| COMBINED_K8 | 0.540 | 0.850 | 0.0202 | 0.099 | 0.900 ± 0.010 | ✓ |
| COMBINED_K16 | 0.460 | 0.751 | 0.0228 | 0.117 | 0.905 ± 0.010 | |

**Validity.**
- **(a) Beats shuffled papers:** passes for GRAD and COMBINED.
- **(b) Gradient coherence beyond SEM:** fails for both.
  - GRAD −0.0174 [−0.0191, −0.0153].
  - COMBINED_K8 −0.0023 [−0.0028, −0.0019].
  - The original 128-d run also failed: GRAD −0.0165; COMBINED_K16 +0.0003 [−0.000007, +0.0007].
- **No further revision applies.**
  - Trigger 1's only remedy (projection 256) has been used.
  - Trigger 2 (nuisance) does not fire: assignment R² is ≤ 0.29 for CE and ≤ 0.07 for gradient norm.
  - Trigger 3 does not fire: it needs κ_seed < 0.4 for all GC K, and COMBINED has 0.75–0.86.

### 3.3 Post-hoc, after unsealing

- **Domain prediction** (5-way, chance 0.28):
  - SEM 0.83 and COMBINED 0.83.
  - GRAD 0.33 (base) and 0.37 (256-d); the raw gradient direction gives 0.33 and 0.37.
- **Cluster–domain agreement.**
  - GRAD clusters: NMI 0.007.
  - SEM_K16 and COMBINED: 0.44–0.48.
- **Nearest neighbours.**
  - Gradient nearest neighbours share a domain 0.26–0.29 of the time against 0.23 for random pairs. They share no
    difficulty, formatting or vocabulary.
  - Semantic nearest neighbours share a domain 0.78 of the time.
- **What the raw gradient encodes.**
  - Its largest explained component is token frequency, at R² 0.009.
  - The residual fingerprint and the GRAD clusters are best predicted by domain and semantics, with GRAD
    cluster-identity accuracy 0.30 against a chance level of 0.27.
  - Both effects are tiny. The fingerprints are dominated by per-view variability, not by grammar or by subject.
- **Diagnostic notes** (section 8) record the recommendation on g_context.

## 4. Answers to the required questions

| question | answer |
|---|---|
| Did GC-EDL beat, match or lose to TOKEN-4? | **Not tested.** No model was trained; the experiment stopped at discovery. |
| Routing specialisation? | Not tested. |
| Functional specialisation? | Not tested. |
| Share of explicit-label EDL's accuracy and specialisation gain recovered? | Not tested. The fle reference values are unchanged: EDL −0.0105 CE, 2.1× FSI. |
| Which latent clusters were discovered (post-hoc)? | **GRAD:** near-uniform, domain-agnostic assignments (NMI ≤ 0.009); four balanced groups of noise. **SEM / COMBINED:** subject clusters (AMI 0.42–0.48) with semantic coherence 0.38–0.52. Representatives (opaque IDs, view offsets, domain mix, snippet) are in `diagnostics/posthoc.json`. |
| Did functional gradients add value beyond semantic embeddings? | **No.** Held-out gradient coherence of gradient-derived clusters is not above that of semantic clusters (section 3.2). Semantic clusters predict held-out gradient directions *better* than gradient clusters do (C 0.0225 vs 0.0051). |
| Compute, disk, deviations, failed candidates, revisions | Section 5. All 54 prototype fits (27 base + 27 r1, where the r1 SEM fits are the reused base fits) are ledgered in `latent/` and `discovery_decision*.json`. Revision 1 is the only engineering revision. |
| Strongest evidence-supported explanation | A single-view gradient of the scout's final MLP projection carries only ~3 % paper-level variance. Its noise is not explained by any nuisance and does not repeat across a paper's views. Any partition learned from it is noise, and any partition that is stable comes from the semantic channel. The useful paper-level structure in gradients is recoverable only by averaging within semantic groups. |

## 5. Costs, storage, deviations, integrity

- **GPU.** 0.29 GPU-h (1,045 s; six fingerprint jobs, all through `gc_driver.py` → `scripts/gpu_guarded.sh`), against a
  30 GPU-h budget. Ledgered in `compute_ledger.tsv` and `ledger/gpu_time_ledger.tsv`.
- **CPU.** About 25 min of discovery, diagnostics and tests.
- **Wall clock.** About 1.4 h.
- **Storage.** About 3.3 GiB new:
  - 1.3 GiB fingerprints, 1.8 GiB latent, 0.1 GiB exports, 78 MiB diagnostics;
  - plus 129 MiB for the bge-small encoder copy.
  - Free disk went from 144 GiB to 140 GiB; the reserve was never approached.
  - No deletions of prior artifacts. `deleted_artifacts.tsv` is empty because no slot was created.
  - Nothing was written to `/dev/shm`.
- **Deviations** (`deviations.jsonl`):
  1. Medoid prototype seed for frozen tables and a fixed SHUF permutation (pre-result clarification).
  2. κ_seed bootstrap with fixed Hungarian matching; nuisance-eligibility ridge details (pre-result).
  3. User diagnostic addendum: logging only, no protocol change.
  4. Engineering revision 1 (projection 256), hashed before execution.
  5. Terminal decision (`terminal_decision.json`, sha `6d787bf3…`).
- **Interruptions.** No external interruption and no GPU contention. `driver_state.json` reads
  `terminal: GC-EDL-V1 STOPPED — DISCOVERY INVALID`.
- **Integrity.** The protected inventory of 6,241 pre-existing files (147 GB, including every fle and fcu file) was
  verified at 02:18:56: **6,241 / 6,241 match, 0 bad** (`protected_inventory_before.verify_20261115T071856Z.json`).

## 6. Deliverables

`artifacts/gc-edl-v1/`:
- `final_analysis.json`, `run_manifest.tsv`, `compute_ledger.tsv`, `deviations.jsonl`;
- `specialization_analysis/` (README only: not reached);
- `driver_state.json`, `driver_log.txt`, `job_logs/`;
- `discovery_decision.json`, `discovery_decision_r1.json`, `terminal_decision.json`, `UNSEALED.json`;
- `latent/` (tables, residualiser, fits), `fingerprints/`, `discovery/`, `sealed/`;
- `diagnostic_ledger.jsonl`, `diagnostic_summary.json`, `diagnostics/`.

Code: `gc_{data,fingerprint,discover,diag_discovery,posthoc,train,configs,screen,evaluate,gpu_checks,tests,driver,
inventory}.py` and `src/csr/gc_{paths,lib,ckpt,diag}.py`.

**Reproduce** (after `source env.sh`):

```bash
.venv/bin/python gc_data.py export --split main --execute     # also calib_search, calib_sel
.venv/bin/python gc_driver.py job fingerprint_main --gpu -- gc_fingerprint.py --split main [--m 256]
.venv/bin/python gc_discover.py residualize|fit|select [--variant r1] --execute
.venv/bin/python gc_diag_discovery.py [--variant r1] --execute; .venv/bin/python gc_posthoc.py
CUDA_VISIBLE_DEVICES= .venv/bin/python gc_tests.py
```

**Not reached:** `gc_screen.py` and `gc_gpu_checks.py` were written and passed review, but never ran.

## 7. Next step

The most informative follow-up is a separately preregistered experiment: strength-matched, label-free **semantic**
latent EDL. It would use the frozen `table_SEM` and `table_COMBINED` tables, aux/LM gradient ratio 0.5–2 %, SHUF and
explicit-EDL controls, and reuse the fle TOKEN-4/EDL finals.

- **It reuses the prepared pipeline.** It uses this experiment's screening, evaluation and trainer code unchanged.
- **It answers the open question.** It tests whether any stable label-free partition can reproduce explicit EDL's
  specialisation. Neither FL-EDL, which was inert at 0.015 % aux ratio, nor GC-EDL, which stopped at discovery, could
  answer that.

**GC-EDL-V1 STOPPED — DISCOVERY INVALID**
