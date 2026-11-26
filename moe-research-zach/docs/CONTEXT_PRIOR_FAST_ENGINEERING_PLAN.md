# csr-context-prior-fast-engineering — plan (written 2026-11-13, before any engineering run)

Started on user instruction (2026-11-13) after the V2 experiment (`csr-context-prior-specialization-v2`, prefix `cps_`)
was paused by operator instruction during its finals (STOP marker + `operator_stop`; resumable; nothing deleted).
Code prefix `cfe_`; artifacts `artifacts/context-prior-fast-engineering/`; runs `runs/*_context_prior_fast_engineering_*`.

## 0. Starting point and diagnosed failure

V2 result so far (development only): in every V2 candidate the context prior was inert. The gate alpha stayed at
its init (~0.01) or shrank (C04: 0.10 -> 0.026), and disabling the prior changed CALIB-SEARCH CE by < 4e-4. V2's frozen
candidate C07 passed V2's screen/S2 rules only within noise; its controls (memory-only, shuffled) were
indistinguishable from it.

Read-only diagnosis (`cfe_diagnose.py`, `diagnosis_inert_*.json`, V2 S2 C07 s100/s101 checkpoints, CALIB-SEARCH
100k tokens/domain + main-stream training batches):

* The LM gradient on the gate bias is ~1e-5 summed over layers. The aux (x0.01) and z (x0.001) terms push the other
  way with comparable magnitude, so alpha has no systematic drive away from its init.
* Forcing alpha with the learned (never co-adapted) prior gives the following CE and routing changes against
  prior-off (s100 / s101):
  * alpha 0.1: +0.0013 / +0.0013 CE, 26% of top-4 slots changed.
  * alpha 0.2: +0.0056 / +0.0060 CE, 46% of slots changed.
  * alpha 0.4: +0.024 / +0.027 CE, 72% of slots changed.
  * Domain-expert MI rises only modestly, from 0.037 to 0.044 at alpha 0.4.

  Routing tolerates a sizeable context prior cheaply. The prior never receives enough weight for it and the experts
  to co-adapt.

## 1. Reuse (no recomputation, no TOKEN-4 retraining)

* Drafts, forecast cache, splits, stream, tokenizer, backbone: as in V2 (hash-verified in V2 `reuse_manifest.json`).
* V2 model, audit and evaluation code: CFEModel subclasses CPSModel. With default knobs it is identical (cfe_tests).
* TOKEN-4 references (read-only):

| Stage | Run | CALIB-SEARCH CE | FSI domain / cluster | Training-loss trajectory |
|---|---|---|---|---|
| S1 (49M tokens, seed 100) | V1 `s1_TOKEN-4_s100` | 3.78961 | 0.003958 / 0.004693 | `metrics.jsonl`, every 50 steps |
| S2 (122.5M tokens, seed 100) | V1 `s2_TOKEN-4_s100` | 3.30178 | 0.005753 / 0.008370 | same |
| S2 (122.5M tokens, seed 101) | V1 `s2_TOKEN-4_s101` | 3.29407 | 0.006264 / 0.008445 | same |

  * CALIB-SELECTION CE at S2 is 3.26850 (s100) and 3.26080 (s101).
  * FINAL: V1 `final_TOKEN-4_s0` and V2 `final_TOKEN_s1`, with VAL evaluations already in the V2 artifact directory.
    These are read only after a freeze here.

  Every candidate uses the identical backbone init, data order, optimizer, LR schedule, steps, tokens and top-4.
  Candidates differ only in the context-prior modules (85k parameters, inference active params +0.2%) and their
  training-only teacher.

## 2. Fast screen (one seed, 100; CALIB-SEARCH only)

Each candidate trains on the S1 plan: 777 steps, 49.0M tokens, the same WSD schedule as TOKEN-4. Registered
early-stop probes run at step 318 (20.1M tokens, end of the teacher phase) and step 544 (70%, student-only, drafts
removed). Each probe uses 20k tokens per domain of CALIB-SEARCH in inference mode (no drafts), with the prior on and
off. The run stops (EARLY_STOPPED, ledgered, reported) if any of the following holds:

1. The mean paired training LM loss minus the TOKEN-4 run's loss over the identical batches is > +0.02 nats.
   This uses the logged steps in [gate - 150, gate).
2. A probe CE is non-finite, or the context mechanism has collapsed: prior-off changes < 2% of top-4 slots AND
   |dCE| < 5e-4.
3. Expert collapse: any layer's expert load share is > 0.10 or < 0.005 (uniform is 0.031).
4. Both specialization (probe domain-expert MI) and context influence (fraction of slots changed by prior-off) are
   <= the incumbent's values at the same step.

Surviving runs complete the 49M plan. This is the promotion to the existing 49M screening budget. Promotion continues
the same run, so no token is spent twice. Each surviving run then gets CALIB-SEARCH CE with the prior on and off,
plus the V2 development audit, which includes FSI domain/cluster, MI and the routing change caused by prior-off.

**Promising at 49M:** a candidate is promising only if all three hold:

* CE <= TOKEN-4 + 0.01.
* FSI domain > TOKEN-4 AND FSI cluster > TOKEN-4.
* Not inert: prior-off changes >= 5% of top-4 slots on the measurement rows OR |dCE_off| >= 0.002.

## 3. Two-seed development (seeds 100, 101; S2 plan: 1939 steps, 122.5M tokens)

The S2 plan has registered early-stop gates at steps 795 and 1357. They use the same rules 1–3, measured against
the matched-seed TOKEN-4 trajectory. Each run gets the V2 development audit.

**Freeze rule:** freeze only if in BOTH seeds:

* CE <= TOKEN-4(seed) + 0.01.
* FSI domain > TOKEN-4(seed) AND FSI cluster > TOKEN-4(seed).
* Not inert, as defined in section 2.

A candidate that fails is not frozen. Engineering continues within the budget.

## 4. After a freeze (each step exactly once)

1. Controls:
   * Context disabled: evaluation-time prior-off on the frozen S2 models (no training).
   * Shuffled context (M5: contexts rolled across rows): S2, seed 100, trained once. This tests whether context
     content matters, rather than just extra routing variation.
2. CALIB-SELECTION CE of the frozen S2 runs, opened only after `frozen.json`. Compared against the recorded TOKEN-4
   S2 CALIB-SELECTION CE.
3. Final validation once: the FINAL plan (245M tokens), seeds 0 and 1. VAL is evaluated once per run, with the
   specialization audit at 40/60/80/100%. The comparison is against the recorded TOKEN-4 FINAL s0 (V1) and s1 (V2).

## 5. Engineering scope, budget, stopping

**Allowed knobs.** One conceptual component per candidate:

* chunk length / causal adaptive boundaries
* context encoder / memory aggregation
* prior layer placement
* per-layer / per-token gates (incl. the gate floor)
* prior normalization / scale
* teacher/student schedule
* forecast-distillation target
* utility / specialization objectives
* LR multipliers for context/router parts

Each new candidate states the observed failure it addresses.

**Budget.** At most 10 screened candidates. Engineering (screen + S2) <= 7 GPU-h. Post-freeze <= 4 GPU-h.

**Stopping.** Stop when one of the following happens:

* A candidate passes section 3.
* The budget is exhausted.
* The same diagnosed failure persists after 3 consecutive modifications targeting it.

**Forbidden.**

* More tokens than the matched TOKEN-4 run.
* Larger model, active experts, parameters or FLOPs.
* Validation or test before a freeze.
* Selecting on validation.
* Future ground-truth tokens at inference.
* Drafts at inference.
* Repeated validation.
* Claiming success from one seed.
* Rerunning TOKEN-4.

**Interruption.** On external SIGINT/SIGTERM, a vanished process, the cfe STOP marker or a foreign GPU process:
checkpoint at the nearest safe boundary, mark the run INTERRUPTED (not FAILED), terminate the experiment's GPU
processes, do not restart, and record an exact resume command.

## 6. Disclosures

* The V2 finals had already evaluated VAL for V2's C07 (s0, s1), TOKEN s1 and MEMORY s0 before the pause. Those VAL
  results are sealed: not read, not analysed, and not used for any engineering decision here.
* CALIB-SELECTION was opened by V1 (its candidates) and by V2 (C07 only). Here it is opened once, after a freeze.
* The engineering candidates start from V2's C07 knobs (the V2 frozen candidate).
