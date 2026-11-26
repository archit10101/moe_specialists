# csr-context-prior-fast-engineering — report (2026-11-13)

**Status: NO FREEZE — engineering stopped by the preregistered rule.** The same diagnosed failure persisted after three
consecutive targeted modifications: functional domain specialization was not above TOKEN-4. No candidate passed the
two-seed development criteria.

* Validation and test were not opened.
* CALIB-SELECTION was not opened in this phase.
* TOKEN-4 was never retrained.
* GPU use: 2.28 h of the 7 h engineering budget, all on the GPU with no foreign processes. No external interruption
  occurred.

Plan: `docs/CONTEXT_PRIOR_FAST_ENGINEERING_PLAN.md` and addendum 1. Artifacts: `artifacts/context-prior-fast-engineering/`.
Key files there:

* `incumbent_table.md` / `.tsv`
* `driver_state.json`
* `deviations.jsonl`
* `diagnosis_inert_*.json`
* `compute_ledger.tsv`
* `checkpoint_ledger.tsv`
* `code_hashes_*.json`
* `ENGINEERING_STOPPED.json`

## 1. Bottom line

1. **The inert-prior failure of V2 is solved.**
   * A gate floor (alpha >= 0.2 or 0.4) makes the context prior genuinely used after the drafts are removed.
   * For F04 at 122.5M tokens, disabling the prior changes 70% of top-4 routing slots and costs +0.020 nats in both
     seeds. In V2, disabling it cost < 0.0004 nats.
   * Accuracy stays comparable to TOKEN-4: F04 S2 CE was -0.0030 (seed 100) and -0.0039 (seed 101).
2. **Context use did not become functional specialization.**
   * Every F04 specialization number lies inside the range of the V2 runs whose prior was inert. Those runs are
     TOKEN-4-equivalent and were trained with the same code at the same scale. This holds for domain FSI, cluster FSI
     and domain-expert MI (table in section 3).
   * F04 S2 seed 100 passed the freeze rule; seed 101 failed it (FSI domain 0.00613 <= 0.00626).
   * The CE advantage over the reused TOKEN-4 runs is shared by the inert V2 runs: -0.001 to -0.008. It is a
     V2-trainer offset, not an effect of the prior.
3. **Interpretation.** The router can be made to depend strongly on causal context with no accuracy loss. The experts
   co-adapt to whatever the prior routes to them, so the model then needs the prior. But in this backbone at this
   scale, a different but equally good assignment of experts carries no more topic-specific function than TOKEN-4's.
   Prior dependence is not specialization.

## 2. What was run

The screen used the S1 plan: 49M tokens, seed 100, the same data, init, optimizer and schedule as the reused V1
`s1_TOKEN-4_s100`. It had registered early-stop gates at steps 318 and 544. S2 used the 122.5M-token plan with seeds
100 and 101 against V1 `s2_TOKEN-4_s100` / `s2_TOKEN-4_s101`.

| ID | Change (one component) | Failure addressed | Outcome |
|---|---|---|---|
| F01 | V2-C07 + gate floor alpha_min 0.2 | prior inert (diagnosis: LM gradient on the gate ~1e-5, cancelled by aux/z) | early-stopped at step 544 by the training-CE gate (+0.033); the gate was miscalibrated (see deviation 4) |
| F01R | F01 re-screened under the null-corrected gate | (same) | complete: CE -0.0016, prior used (49% slots, +0.0046 CE), FSI domain 0.00352 <= T4 0.00396 -> not promising |
| F02 | F01R + memory half-lives 128/512/2048 (drop 16) | prior used but not topical | early-stopped at step 318: probe MI and context influence both below the incumbent |
| F03 | F01R + utility-conditioned cross-chunk consistency (c_spec 0.1) | prior not topical (token-local utility target) | complete: CE +0.0047, FSI domain 0.00363 -> not promising |
| F04 | F03 + gate floor 0.4 | same failure, third and last allowed modification | screen PROMISING (CE +0.0031, FSI 0.00401 / 0.00503 vs 0.00396 / 0.00469; prior-off +0.029 CE, 77% slots); S2 seed 100 pass, seed 101 FAIL |

Six screened candidates out of the cap of 10. One candidate reached S2. No controls, CALIB-SELECTION or VAL were run,
because the plan runs those only after a freeze.

## 3. F04 at 122.5M tokens vs TOKEN-4 and vs the inert-prior null (V2 S2 runs, same code and scale)

| | CE | FSI domain | FSI cluster | MI domain | prior-off dCE | prior-off routing change |
|---|---|---|---|---|---|---|
| TOKEN-4 s100 / s101 (reused V1) | 3.3018 / 3.2941 | 0.00575 / 0.00626 | 0.00837 / 0.00845 | 0.0339 / 0.0314 | n/a | n/a |
| **F04 s100** | 3.2988 | 0.00616 | 0.00858 | 0.0344 | +0.0202 | 70.4% |
| **F04 s101** | 3.2902 | 0.00613 | 0.00906 | 0.0314 | +0.0200 | 70.0% |
| inert V2 S2, s100 (C02, C07, M1, M5) | 3.2977–3.3010 | 0.00581–0.00692 | 0.00744–0.00900 | 0.0313–0.0331 | <= 0.0004 | (not measured) |
| inert V2 S2, s101 (C02, C07, M1, M5) | 3.2862–3.2966 | 0.00570–0.00650 | 0.00812–0.00924 | 0.0301–0.0332 | <= 0.0004 | (not measured) |

**Throughput.** F04 trains at about 102k tok/s against TOKEN-4's 137k. The gap comes mostly from the training-only
teacher (draft features) and the extra per-step loss passes. The inference cost of the stripped model was not
benchmarked, since nothing was frozen. The prior adds 85k parameters, +0.2% of active parameters per token.

**Prior effect on routing.** Probe MI is higher with the prior on than off: 0.055 vs 0.047 at step 1357 (seed 100),
and 0.034 vs 0.029 on the audit rows. So the prior injects some domain information into routing. That information does
not show up as function: the expert-group ablations (FSI) are no more topic-selective than under the null.

## 4. Diagnosis (read-only)

1. **V2 inertness was a gradient-magnitude problem.** The LM gradient on the gate bias was ~1e-5 summed over 11
   layers, opposed by the aux and z terms. Forcing alpha on a trained V2 model changed 26–46% of slots for only
   +0.001 to +0.006 nats. Routing is highly permutation-tolerant, so there is little LM pressure in either direction.
2. **A floored prior co-adapts but does not specialize.** With alpha >= 0.2, the experts adapt to the
   context-conditioned assignment, which is why removing the prior then hurts (+0.005 at a 0.2 floor, +0.02 to +0.03
   at 0.4).
   * Raising the floor from 0.2 to 0.4 raised the prior's share of routing.
   * Domain FSI at 49M went from 0.0035–0.0036 to 0.0040, which is inside the null's 0.0036–0.0044.
   * At 122.5M, F04 sits mid-range of the null.
3. **Objective pressure for topical consistency (c_spec) moved MI slightly but not FSI.** At 49M, MI rose from 0.028 to
   0.031, still inside the null. The utility target is token-local, while the consistency loss acts only on CF rows
   with positive utility. Neither defines an expert's function by topic.
4. **Measurement noise is large at this scale.**
   * Single-seed domain FSI spans 0.0036–0.0044 at 49M and 0.0057–0.0069 at 122.5M across TOKEN-4-equivalent runs.
   * Single-batch training-loss gaps vary by about ±0.02 at 49M.
   * Any real specialization effect below ~15% of FSI is not detectable with two seeds.

## 5. Deviations (all in `deviations.jsonl`)

| # | Kind | What happened |
|---|---|---|
| 1 | Bug fix, pre-result | `CPSModel.strip_teacher` dropped the cfe knob. Fixed with an override and test T2b. |
| 2 | Operational, disk | Rolling resume checkpoints were put on tmpfs via a symlink. The project guard refused the out-of-root path, so F01's first leg aborted at step 77 with no result. Reverted; the guard was not bypassed. |
| 3 | Operational, disk | One rolling slot per run, and AdamW moments stored in bf16 in the slot (0.79 GB instead of 1.24 GB). A smoke test showed resume matches the uninterrupted run within 1e-4 nats per step. Deleting the older redundant rolling copy of V2's paused `final_MEMORY_s1` checkpoint was attempted, denied by the permission system, and not done. V2 slots are untouched. |
| 4 | Rule change after a result (disclosed) | The training-CE gate is null-corrected (addendum 1). The literal +0.02 rule would have stopped 7 of 11 inert V2 runs, including V2's frozen C07. The change was made after F01's stop and favours F01. F01 was re-screened as F01R. |
| 5, 5b | Operational, disk | Finished audits are gzip-compressed losslessly (sha256 kept). Milestones of non-promising or superseded screens are deleted after their audits (sha256 ledgered). The pre-launch disk estimate was set to the measured slot size. |

**Also fixed before any comparison:** an incumbent-selection bug made an early-stopped run the incumbent. It affected
only F01R's comparison against its own identical predecessor, and F01R passed its gates.

## 6. V2 status (unchanged by this phase)

`csr-context-prior-specialization-v2` is paused by operator instruction during its finals. It has a STOP marker and an
`operator_stop` record in `driver_state.json`, and `final_MEMORY_s1` is resumable from its verified step-2265
checkpoint.

The VAL evaluations of V2's finished finals are sealed: C07 s0/s1, TOKEN s1 and MEMORY s0 were not read or analysed.
Resuming V2 requires your explicit instruction. Run `cps_driver.py run --authorized-resume` after removing the STOP
marker.

## 7. What would be needed next (proposal, not run)

* **Make specialization a measured target, not a hoped-for by-product.** At this scale the backbone's experts are
  interchangeable enough that context-driven assignment produces no topic-specific function. Options inside the
  backbone:
  1. An expert-level objective that rewards disabling a context group to hurt its own context more than matched
     others. This is a differentiable FSI-style proxy on CF rows, with a CE guard.
  2. Fewer, larger experts per topic group at the same active parameters, so a group cannot be replaced by its
     neighbours.
  3. Evaluation at a longer budget (FINAL, 245M tokens), where V2/V1 showed FSI rising with scale.
* **Power.** A two-seed test cannot resolve FSI effects below ~15%. A claim of significantly stronger specialization
  needs 4 or more seeds per arm, or a paired, lower-variance FSI estimator (more audit tokens, bootstrap over
  documents).
* **Disk.** The machine has about 1.8 GiB above the 100 GiB guard reserve. A FINAL-scale continuation (four milestones
  per run) needs one of two things: about 2 GB freed (for example the redundant older V2 slot copy), or your OK to
  delete scratch directories.
