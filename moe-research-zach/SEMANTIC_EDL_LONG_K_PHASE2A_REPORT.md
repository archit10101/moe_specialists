# csr-semantic-edl-long-adaptive-k-v1 — Phase 1 / Phase 2A report

Terminal state: **STOPPED_BEFORE_PHASE2B_BY_USER** (2026-11-17T01:51:19-05:00). This is a user scope reduction, not a scientific outcome. Phase 1 and Phase 2A ran to completion under the preregistration (`docs/SEMANTIC_EDL_LONG_ADAPTIVE_K_PREREG.md`, sha256 f33068886139…). No Phase-2B job was implemented or dispatched. The block marker `PHASE2B_BLOCKED_BY_USER.json` (0400) is intact.

Machine-readable data: `artifacts/semantic-edl-long-k/phase2a_analysis.json` (sha 359d2230b50d…). Generated tables: `artifacts/semantic-edl-long-k/phase2a_tables.md`. Decisions:

- `phase1_decision.json` (3c424e88…)
- `phase2/decision_2a.json` (4921edd0…)
- `phase2/frozen_2a.json` (4140a6f4…)
- `phase2/calibsel_2a.json` (7e458d19…)
- `phase2/gate_2b.json` (8f6f643d…)

## 1. Bottom line

1. **Phase 1 (245M tokens, 4 paired seeds, VAL): SEMANTIC-C6 does NOT succeed.**
   - C6 routes more domain-aligned than its matched shuffled-paper control (SHUF): domain NMI was higher in 4/4 seeds, probe accuracy in 3/4.
   - The functional criterion F fails. Per-expert domain interaction C6 − SHUF was positive in only 2/4 seeds, with mean +2.3e-6 (≈ 0.4 % of the effect size).
   - In other words, the semantic grouping changes *where* tokens go, but not *what the experts are functionally specialized for*, beyond what any persistent random paper grouping produces.
   - Preregistered phrase: "semantic information beyond persistent grouping" = **not shown**.
2. **Phase 2A (fixed K ∈ {4, 8, 16, 32}, 50M, 2 seeds, CALIB-SEARCH): only K = 8 met the repeatable-advantage rule, and it was frozen.**
   - Its CALIB-SELECTION confirmation **failed**: FSI_e5 sem − shuf was −2.3e-5 in seed 0, while pseudo-FSI and CE favored semantic in both seeds.
   - Preregistered conclusion: *no fixed K shows a repeatable semantic advantage; adaptive K is not supported by 2A.*
3. **GEN-A / GEN-B did not trigger**, because their exact conditions were not met (section 5). No engineering generation ran.
4. **The Phase-2B gate would have been CLOSED.** It needs Phase-1 R∧F or a confirmed K*, and neither happened. So even without the user block, the preregistration would not have run Phase 2B.

## 2. Phase 1 — four-seed results (VAL, p100 = 245,032,457 tokens, 3,880 steps)

All 16 runs share one plan (sha f4bc3c93…) and the same token count and step count. C6 and SHUF configs differ only in the latent table path and hash. Parameters are equal.

| cond | CE mean (sd) | domain NMI | probe acc | domain FSI | interaction | spec. experts | e5 NMI |
|---|---|---|---|---|---|---|---|
| TOKEN-4 | 3.0533 (0.0057) | 0.0126 | 0.690 | 0.00762 | 0.000382 | 55.0 | 0.0219 |
| EDL (explicit) | 3.0435 (0.0035) | 0.0574 | 0.752 | 0.01687 | 0.000657 | 148.8 | 0.0780 |
| SEMANTIC-C6 | 3.0467 (0.0034) | 0.0547 | 0.736 | 0.01599 | 0.000637 | 134.3 | 0.0774 |
| SHUFFLED-C6 | 3.0461 (0.0067) | 0.0452 | 0.730 | 0.01490 | 0.000634 | 125.0 | 0.0710 |

Per-seed values for all four conditions are in `phase2a_tables.md` and `phase2a_analysis.json` (`phase1.metrics`).

Mean CE by VAL domain, over domains 0–4 (descriptive):

| cond | d0 | d1 | d2 | d3 | d4 |
|---|---|---|---|---|---|
| T4 | 3.1783 | 2.6326 | 3.2856 | 3.1667 | 3.0031 |
| EDL | 3.1696 | 2.6220 | 3.2738 | 3.1581 | 2.9939 |
| C6 | 3.1719 | 2.6257 | 3.2768 | 3.1614 | 2.9975 |
| SHUF | 3.1717 | 2.6241 | 3.2772 | 3.1609 | 2.9965 |

### Every per-seed C6 − SHUF difference (p100)

| metric | s0 | s1 | s2 | s3 | mean | positive |
|---|---|---|---|---|---|---|
| CE (lower is better) | +0.0113 | −0.0017 | −0.0095 | +0.0022 | +0.0006 | C6 worse in 2/4 |
| domain NMI | +0.0119 | +0.0117 | +0.0055 | +0.0088 | +0.0095 | 4/4 |
| probe accuracy | −0.0068 | +0.0182 | +0.0129 | +0.0005 | +0.0062 | 3/4 |
| domain FSI | +0.00067 | +0.00081 | +0.00030 | +0.00259 | +0.00109 | 4/4 |
| per-expert interaction | −0.000016 | −0.000067 | +0.000010 | +0.000081 | +0.000002 | 2/4 |
| specialized experts | +14 | −1 | +8 | +16 | +9.25 | 3/4 |
| e5 NMI | +0.0105 | +0.0072 | +0.0038 | +0.0044 | +0.0065 | 4/4 |
| e5 coherence | +0.00025 | +0.00018 | +0.00004 | +0.00012 | +0.00015 | 4/4 |

Paired per-document bootstrap of CE, C6 − SHUF (25,333 VAL docs; 95 % CI):

| seed | diff | 95 % CI |
|---|---|---|
| s0 | +0.0113 | [+0.0110, +0.0116] |
| s1 | −0.0017 | [−0.0020, −0.0013] |
| s2 | −0.0095 | [−0.0099, −0.0091] |
| s3 | +0.0022 | [+0.0020, +0.0025] |

Every seed's interval excludes 0, but the sign flips between seeds. The CE difference is dominated by seed-to-seed variation, not by the condition.

### Preregistered Phase-1 criteria (n = 4; sign rule ≥ 3/4 and mean > 0)

| criterion | result | pass |
|---|---|---|
| R: domain NMI C6 > SHUF | 4/4, mean +0.0095 | **pass** |
| R: probe acc C6 > SHUF | 3/4, mean +0.0062 | **pass** |
| F: per-expert domain interaction C6 > SHUF | 2/4, mean +2.3e-6 | **FAIL** |
| CE: mean C6 − T4 ≤ +0.010 | per seed −0.0010, −0.0145, −0.0140, +0.0031; mean −0.0066 | **pass** |
| P: no pathology, params/FLOPs equal | all seeds ok | **pass** |

- **Overall:** C6 does not succeed.
- **Explicit-EDL reference:** C6 − EDL CE per seed was +0.0039, +0.0016, −0.0029, +0.0102 (mean +0.0032). The comparable flag is false: C6 − T4 is positive in seed 3, and C6 interaction ≤ SHUF in seeds 0 and 1. The beats flag is also false.
- **Mass control (descriptive):** C6 − SHUF interaction after load-matching was −8.9e-6, −6.0e-5, +1.5e-5, +8.7e-5. This is also null.
- **Trajectory (C6 − SHUF; 4 seeds each):**
  - Domain NMI is positive 4/4 at every milestone: +0.013 at s20, +0.009 at s50, and +0.009–0.0096 from p040 to p100.
  - Domain FSI is positive 4/4 at every milestone and grows from +0.0004 to +0.0011.
  - Per-expert interaction stays sign-mixed: s50 3/4 positive, p040 1/4, p100 2/4.
  - So the routing-alignment gap appears early and stays constant. The functional gap never emerges.

## 3. Phase 2A — fixed-K screening (CALIB-SEARCH half B, 50M tokens; matched shuffled controls)

Rule: a K is *repeatable* iff it is clustering-eligible, Δe5 > 0 and Δpseudo > 0 in both seeds, and mean ΔCE ≤ +0.005. Every Δ is sem − shuf at s50.

| K | seed | Δ FSI_e5 | Δ FSI_pseudo | ΔCE | pathology |
|---|---|---|---|---|---|
| 4 | 0 | −0.00053 | +0.00062 | −0.0071 | no |
| 4 | 1 | +0.00010 | +0.00162 | +0.0144 | no |
| 8 | 0 | +0.00040 | +0.00084 | −0.0134 | no |
| 8 | 1 | +0.00177 | +0.00125 | −0.0239 | no |
| 16 | 0 | +0.00016 | +0.00081 | −0.0001 | no |
| 16 | 1 | −0.00097 | +0.00007 | −0.0096 | no |
| 32 | 0 | +0.00091 | −0.00004 | +0.0020 | no |
| 32 | 1 | −0.00007 | +0.00058 | −0.0031 | no |

| K | mean Δe5 | mean Δpseudo | mean ΔCE | eligible | repeatable |
|---|---|---|---|---|---|
| 4 | −0.00021 | +0.00112 | +0.0036 | yes | no (Δe5 sign) |
| 8 | +0.00108 | +0.00105 | −0.0187 | yes | **yes** |
| 16 | −0.00040 | +0.00044 | −0.0048 | yes | no (Δe5 sign) |
| 32 | +0.00042 | +0.00027 | −0.0006 | yes | no (Δe5, Δpseudo sign) |

- **Selected (frozen) K\* = 8.** It was the only repeatable K.
- s20 rows are descriptive and are in `phase2a_tables.md`.

**CALIB-SELECTION (opened once; K8 sem vs shuf at s50, seeds 0/1).** Confirmation needs FSI_e5 and FSI_pseudo sem > shuf in both seeds, and mean ΔCE ≤ +0.005.

| seed | Δ FSI_e5 | Δ FSI_pseudo | ΔCE |
|---|---|---|---|
| 0 | **−0.0000226** | +0.00118 | −0.0140 |
| 1 | +0.000130 | +0.00122 | −0.0226 |

- **Result: NOT confirmed.** The e5 criterion fails in seed 0.
- The seed-0 FSI_e5 deficit is small, and the pseudo-FSI and CE differences favor semantic in both seeds. As a descriptive note only, K8 is the most promising K screened. Under the preregistered rule, however, no fixed K shows a repeatable semantic advantage, and adaptive K is not supported by 2A.

## 4. Would the Phase-2B gate have opened?

**No.**

- Gate = (Phase-1 R ∧ F) ∨ (Phase-2A K\* confirmed).
- R passed but F failed, and K8 was not confirmed.
- `gate_2b.json`: gate_open false.

## 5. GEN-A / GEN-B

- **GEN-A** triggers if no K is repeatable. K8 was repeatable, so it was **not triggered**.
- **GEN-B** triggers if some K has Δe5 > 0 in both seeds but fails Δpseudo or CE. The only K with Δe5 > 0 in both seeds is K8, and it passed both. So GEN-B was **not triggered**.
- No `docs/SLK_GEN*.md` exists, and `decision_2a.json` has generations = [].

## 6. What remains unresolved

- **Whether K8's semantic advantage is real or noise.** It is positive in 2/2 screening seeds and in 3 of 4 confirmation metric/seed cells. The one failing cell is tiny, at −2e-5. Two seeds at 50M cannot separate a small real effect from noise.
- **Whether a 245M K8 run would show the functional (F) effect that K16 C6 lacks.** It was not tested, because Phase-1 finals are K16 only.
- **Phase 2B** (adaptive/hierarchical K) was not run. It was gate-closed and user-blocked, and its code is not implemented.
- **The "routing aligns but function doesn't" gap at K16.** Neither phase explains it. Task 2 (csr-semantic-utility-edl-v1) targets this gap directly.

## 7. Compute, disk, deviations, integrity

**GPU time: 24.54 GPU-h total** (compute ledger; cap 60).

| phase | train | eval | total | cap |
|---|---|---|---|---|
| Phase 1 | 6.87 h | 14.11 h | 20.98 h | 28 |
| Phase 2A | 1.56 h | 2.03 h | 3.58 h | 12 |
| checks | — | — | 0.07 h | — |

Wall clock: 2026-11-15 15:55 → 2026-11-17 01:51 EST, including a user pause on 2026-11-16 from 11:27 to 20:45.

**Disk:**

| item | size |
|---|---|
| own run dirs | 23 dirs, 17.4 GiB |
| artifacts | 0.29 GiB |
| .tmp | 0 |
| free | 108.6 GiB (reserve 100 GiB respected) |

**Deviations** (`DEVIATIONS.jsonl`):

1. The CPU suite ran as a path-only port, so that semantic-edl-v1 stays untouched.
2. **User scope reduction** to Phase 1 + 2A; Phase 2B is prohibited.
3. Orchestration-only edits of `slk_phase2.py`: stage limit, block marker, and the GEN budget-projection guard. No thresholds, comparisons, selection rules, splits, metrics or scientific code changed.
4. User-requested pause at the job boundary after `val_P1_EDL_s1_p080`. Resumed 2026-11-16 20:45 via `slk_run.py resume-through-2a`.

Also logged in DECISION_LOG: the integrity scanner's test-split pattern was tightened after a false positive (`latest_verified`). This changed the checker, not the experiment.

**Integrity** (`integrity_20261117T065427Z.json`, all_ok true):

- Prereg hash ok.
- PHASE1_VAL_OPEN, CALIBSEL_OPEN and PHASE2B_BLOCKED are all 0400.
- Protected inventory: 6,498 files, 0 mismatches.
- semantic-edl-v1 and reused runs: 0 files modified after prereg.
- Locked test split: 0 code references and 0 reads after prereg (atime supplementary evidence).
- All Phase-1 runs are COMPLETE with an identical plan and token count.
- CALIB-SELECTION was opened exactly once for the frozen K\*.
- No Phase-2B job was dispatched.

## 8. Resume (not executed)

`artifacts/semantic-edl-long-k/RESUME.md` holds the one command that could later start 2B:

```
cd /home/zapan4/moe-research && source env.sh && .venv/bin/python slk_phase2.py all --allow-phase2b
```

It must not be run without explicit user instruction. Because the gate is closed, under the preregistration it would only re-record the gate. The 2B code is not implemented.
