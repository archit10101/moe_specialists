# csr-continuous-base-target-free-asrs-v2 (asrs2) — final report

**Verdict: TERMINAL_STAGE1_FAIL.** With the base updated on every batch and target-free semantic assignment, the five
K = 5 semantic specialists stay functional and never corrupt. But the deployed causal router does not reach the
accuracy of the matched generic banks:

| | vs G | vs ALT-G |
|---|---|---|
| R1-hard (primary) | +0.0019 | +0.0005 |
| R1-soft | +0.0016 | +0.0002 |

Specialisation retention is 0.27 (hard) and 0.15 (soft). Several Stage-1 criteria fail in both candidates. Under the
frozen engineering mapping no generation was permitted, so CALIB-SELECTION, Stage 2, Stage 3 and VALIDATION were not
run. The test split was never loaded.

| item | location |
|---|---|
| Phase-0 diagnosis of asrs1 | `docs/CSR_CONTINUOUS_BASE_TARGET_FREE_ASRS_V2_DIAGNOSIS.md` |
| Preregistration (frozen) | `docs/CSR_CONTINUOUS_BASE_TARGET_FREE_ASRS_V2_PREREG.md`, sha256 `afb54058…dfee9` |
| Logs | `artifacts/csr-continuous-base-target-free-asrs-v2/`: OBSERVATIONS.md, DECISION_LOG.jsonl, deviations.jsonl, RUN_LOG.jsonl, ledgers, RESUME.md |
| Results | `phase0_diagnosis.json`, `phase0_override_analysis.json`, `gpu_check_20261122T051207Z.json`, `stage0f.json`, `stage1_gen0.json`, `windows_<run>.json`, `base_match_<run>.json`, `c1_diagnostic.json`, `analysis/*.json`, `final_analysis.json` |
| K = 5 tables | `tables/k5_merge.json`, `p5_main.npz`, `p5_calib_search.npz` (0400, hashed) |
| Runs | `runs/20261122T051254Z_asrs2_S0F` (ckpt_F: 4 seeds); `runs/20261122T053544Z_asrs2_S1` (m510, m702, m894, m1111); C1 diagnostics `…_C1B1`, `…_C1B2` |
| Code | `asrs2_{phase0,clusters,train,eval,analyze,basematch,run,driver,tests,inventory}.py`, `src/csr/asrs2_{paths,lib,cfg,ckpt,log}.py` |

## 1. Phase 0 — independent diagnosis of asrs1 (summary)

- **H1, base starvation: confirmed. It fully explains the +0.36 deficit.**
  - At matched update counts, asrs1's base loss equals the ordinary continuation's: 1.7e-5 on identical rows,
    −0.015 on average otherwise.
  - The S3 curve projects 0.357 for 320 vs 793 updates; SEL measured 0.3595.
- **H2, target leakage: confirmed, but narrower than the asrs1 report claimed.**
  - The teacher's choice agreed with the region prior 60–78% of the time.
  - The target skew sat on the 22–40% utility-override tokens: target → choice 0.34–0.39 vs a majority baseline of
    0.10–0.17.
  - The held-out prior specialist turned harmful after one specialist phase (+0.021 → −0.217).
  - Controls without target dependence did not corrupt: ALT-G (same schedule) and the ccsr1 S3 specialists (same
    class and optimizer).
  - The per-token oracle is already target-predictable (0.40) in good frozen banks, so that alone is weak evidence.
- **H3 alternatives: ruled out.** No phase-isolation, resume or evaluation fault; no router drift or collapse; no
  loss of headroom.

## 2. Method as run

**User amendment.** Exactly K = 5 label-free clusters and 5 specialists. The 16 frozen GC-EDL SEM ProtoNet prototypes
(λ_coh = 0) were merged by weighted spherical k-means (1,000 restarts, 0.05 share floor):

| | value |
|---|---|
| merge map | [0,0,1,2,3,1,2,2,0,4,2,4,0,1,3,3] |
| main-stream region shares | 0.211 / 0.118 / 0.253 / 0.252 / 0.166 |

The persistent shuffled control applies a balanced per-paper permutation. Its sizes are identical to 1e-5, within-paper
switching and entropy are identical, and NMI with SEM is 0.005.

**Stage 0F (frozen stage).** The base was frozen at step 318. On main-stream rows 0…317 (20.06M tokens), four
independent specialist-training seeds each trained G5, SEM5 and SHUF5 banks (384→256→384 × 5) plus R1 causal
routers, with target-free hard prior assignment.

**Stage 1A (continuation).** Steps 318…1110 (50,086,846 tokens), with the base updated on all 793 steps through the
ccsr1 S3 path.
- Auxiliary windows of 32 SPEC / 16 ROUTER steps: 537 specialist-update steps and 256 router-update steps.
- Arms in one process: G, ALT-G, SEMh/SHUFh (hard prior), SEMs/SHUFs (sampled prior), FROZ (frozen bank, refreshed
  router).
- No token's target ever chose a specialist.

**Checks before the freeze.**
- CPU tests 10/10, including bitwise base identity with and without the residual arms and bitwise resume.
- GPU check passed: all arms vs base-only gave a per-step loss difference of 1.7e-4 against a 1.4e-4 nondeterminism
  floor, parameter distance 0.0060 vs 0.0059, and SIGINT/resume completed.

## 3. Results

**Stage 0F** (SEL, seed 0; the base CE reproduces c00 exactly: 4.671856).

- **Function:**
  - ΔI = I_SEM5 − I_SHUF5 = **+0.0167 [0.0146, 0.0188]**.
  - 5/5 specialists positive and significant.
  - Ordinary (unmatched) own advantage: 0.022 vs −0.001.
- **CE:** SEM5 with its causal router is **+0.0106** vs G5, and the region teacher is +0.0073 vs G5 (G5 − base:
  −0.076). Pre-data prediction 2 (−0.003…+0.002) **missed**.
- **R2 partner-view utility:** reliable (split-half r 0.97) but adds nothing beyond R1, so it was not a candidate.

**Stage 1 trajectory** (SEL, seed 0; CE minus ALT-G unless noted):

| milestone | base | G − ALT-G | SEMh | region teacher SEMh | SEMh − SHUFh | SEMs | FROZ | ΔI SEMh (ΔI SEMs) | positive specialists |
|---|---|---|---|---|---|---|---|---|---|
| ckpt_F (318; vs G5) | 4.67186 | — | +0.0106 (vs G5) | +0.0073 (vs G5) | −0.0108 | — | — | 0.0167 | 5 |
| m510 | 4.22652 | −0.0010 | +0.0024 | −0.0010 | −0.0036 | +0.0017 | +0.047 | 0.0082 (0.0066) | 5 |
| m702 | 3.90478 | −0.0006 | +0.0019 | −0.0003 | −0.0025 | +0.0015 | +0.063 | 0.0070 (0.0039) | 5 |
| m894 | 3.73893 | −0.0004 | +0.0009 | −0.0012 | −0.0018 | +0.0006 | +0.054 | 0.0047 (0.0022) | 5 |
| m1111 | 3.61954 | −0.0014 | +0.0005 [0.0003, 0.0008] | −0.0016 [−0.0019, −0.0013] | −0.0011 [−0.0013, −0.0008] | +0.0002 | +0.058 | 0.0045 [0.0038, 0.0051] (0.0025) | 5 |

**Stage-1 criteria at m1111:**

| criterion | R1-hard | R1-soft |
|---|---|---|
| C1 cross-process base match | FAIL as specified (see §4) | FAIL as specified |
| C2 mean specialist delta ≤ +0.02 | +0.003 ✓ | −0.003 ✓ |
| C3 target predictability ≤ SHUF + 0.05 | 0.246 vs 0.283 ✓ | ✓ |
| C4 / S1-8 SEM < SHUF | −0.0011 ✓ | −0.0012 ✓ |
| C5 / S1-6 no collapse | loads 0.25 / 0.12 / 0.23 / 0.22 / 0.18 ✓ | ✓ |
| S1-1 ≤ G + 0.001 | +0.0019 ✗ | +0.0016 ✗ |
| S1-2 < ALT-G | +0.0005 ✗ | +0.0002 ✗ |
| S1-3 ΔI > 0 with CI | +0.0045 [0.0038, 0.0051] ✓ | +0.0025 ✓ |
| S1-4 ≥ 4/5 positive | 5/5 ✓ | 5/5 ✓ |
| S1-5 retention ≥ 0.75 | 0.27 ✗ | 0.15 ✗ |
| S1-7 not confined | positions ≥ 128 +0.0004; 1/5 groups ✗ | ✗ |

**Where the deployed router loses** (hard, m1111):
- positions 0–31: +0.0053 vs ALT-G, where the teacher is −0.0153;
- post-transition tokens (4.5% of tokens): +0.0031 vs ALT-G and +0.0038 vs the teacher;
- stable tokens (42%): −0.0003 vs ALT-G.

Deployed routing is nearly document-constant (switch rate 0.013), with region-prior agreement 0.747.

**Predictability** (cross-fitted):
- The oracle specialist is predicted by the target at 0.39, against 0.23 for everything causal and semantic, the same
  as SHUF.
- The assigned specialist is predicted by the target at 0.25, at the majority baseline. Assignments are target-free.

**H2 falsifier (per-window record).** The held-out utility of the prior-assigned SEM specialist stayed positive in
all 17 windows: +0.016, peaking at +0.021, ending at +0.007. SHUF ended at +0.0035. The FROZ bank fell from −0.002 to
−0.050. Target-free assignment does not corrupt; asrs1's corruption came from target-dependent assignment.

## 4. Deviations

1. **Shuffled-control construction (before any use).** Random per-paper permutations could not meet the size
   requirement (deviation about 0.06), and pinning the dominant cluster still left 0.017. The final rule is a balanced
   hash-ordered permutation. The first table set was deleted (logged) to make the paper-level SHUF group consistent.
2. **C1 failed its preregistered cross-process tolerance** (in-process base −0.0032 SEL and −0.0028 per step vs ccsr1
   S3).
   - The preregistered response was a diagnostic: two base-only reruns differ from each other by 0.0025, and one
     differs from ccsr1 S3 by 0.0035.
   - The tolerance was below 793-step GPU nondeterminism. No implementation fault exists, and no rerun was made.
   - All scientific comparisons are within-process and unaffected.
3. The C1-diagnostic phase was added to the orchestrator after Stage 1, logged in DECISION_LOG before it ran. It is not
   an engineering generation.

## 5. Interpretation and failure localisation

- **Not base drift.** The in-process base is the ordinary base, and the cross-process spread is nondeterminism.
- **Not specialist corruption.** C2 holds, positive utilities persist, and 5/5 specialists are positive.
- **Not semantic ≈ shuffled.** SEM beats SHUF in CE and function at every milestone.
- **Router-information / distillation failure.** The non-causal region teacher deploying the same continued bank beats
  ALT-G (−0.0016) and ties G. The causal R1 router loses 0.0021 to it, mostly at document starts and topic
  transitions. Under the frozen mapping, rule 2's remedies (router lr) did not apply, so the experiment is terminal.
- **Retention.** Exposure-matched specialisation shrinks from 0.0167 to 0.0045 as the base improves. Every residual
  bank's usefulness shrinks: G's gain over base fell to −0.007, and held-out utilities halved. A never-updated frozen
  bank becomes harmful.
- **Frozen-stage accuracy gap.** At K = 5, hard semantic grouping gives functional specialists that are less accurate
  than a token-routed generic bank: +0.007 even for the teacher at step 318. The gap closes during continuation, and
  by m1111 the teacher reaches G.

## 6. Compute, disk

| item | value |
|---|---|
| GPU total | 1.84 h (GPU check 0.12, Stage 0F 0.37, Stage 1 1.08, C1 diagnostic 0.26), within the 3 + 12 h caps |
| Stage-1 training | 2.1 s/step with all 7 arms vs 0.47–0.52 s base-only, peak 17.3 GiB |
| SEL evaluation | about 4 min per milestone |
| Disk | 137 → 129 GiB free, never below the 100 GiB reserve; artifacts 3.1 GB, runs 1.4 GB, retained final slots 3.7 GB |
| Deletions | only this experiment's test and GPU-check scratch and rotated slots, 45 rows in deletions.tsv with sha256 |
| Protected inventory | 1,085 files, all unchanged |

## 7. Strongest conclusion and what remains unproven

**Supported.**
- A continuous base plus target-free semantic assignment fixes both asrs1 failures. The base trajectory is ordinary,
  and the specialists stay useful and semantically specific: SEM > SHUF and 5/5 positive at every milestone.
- What remains is deployment. A causal R1 router recovers the region-prior assignment in stable text but not at
  document starts and topic transitions.
- The semantic advantage over generic residuals is small at K = 5: the region teacher only ties G at m1111.

**Unproven:**
- whether a router trained toward demonstrated subject-conditioned utility (instead of prior distillation) closes the
  0.002 deployment gap;
- multi-seed specialist replication (seeds 1–3 exist at Stage 0F only);
- any CALIB-SELECTION, 245M or VALIDATION result.

The follow-up `subject-conditioned-utility-router-v1` reuses these banks read-only.
