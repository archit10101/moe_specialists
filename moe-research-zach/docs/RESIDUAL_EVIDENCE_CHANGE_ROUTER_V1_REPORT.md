# residual-evidence-change-router-v1 (recr1) — final report

**State: TERMINAL_STAGE1_FAIL.**
- The engineering loop stopped after generation 0 because no permitted factor addresses the measured cause.
- CSELECT was opened once for the frozen candidate.
- No validation, CALIB-SELECTION or test data was opened, and nothing ran at 245M.

| item | location |
|---|---|
| Preregistration (frozen) | `docs/RESIDUAL_EVIDENCE_CHANGE_ROUTER_V1_PREREG.md`, sha256 `64352342…09e9` |
| Logs | `artifacts/residual-evidence-change-router-v1/`: OBSERVATIONS.md (audit first), DECISION_LOG.jsonl, deviations.jsonl, RUN_LOG.jsonl, run_manifest.tsv, compute_ledger.tsv, disk_ledger.tsv, RESUME.md |
| Manifests | data_manifest.json (+ .sha256), reuse_manifest.json, protected_inventory_before.json / _verify_*.json, code_manifest_prereg.json, FROZEN.json |
| Results | analysis/gen0_CSEARCH.json, analysis/CSELECT.json, final_analysis.json, timing.json, cpu_tests_*.json |
| Code | recr1_{eval,run,tests,manifest,timing}.py, src/csr/recr1_{lib,paths,log}.py |

## Answer

**No.** Residual evidence cannot tell the router when its specialist has stopped being useful, or which candidate is better:
- **Not informative.** The realised per-token gain of a residual over the base is essentially white noise: lag-1 autocorrelation 0.004, sd ≈ 0.07–0.1 nats, against true differences of 0.001–0.01 nats.
- **Alarms at chance.** CUSUM alarms caught semantic changes at chance level: recall within 64 tokens was 0.17, against 0.17 for random alarms at the same rate. Median delay was about 100 tokens.
- **Search ranking at chance.** Search evidence ranked the two candidates correctly 48–50% of the time, a coin flip. Even 256 search tokens would reach only 0.52.

**The persistent hold made things worse.** RECR holds one causally chosen option between alarms, and that loses the token-level utility of the existing unrestricted router in stable context. On held-out CSELECT, RECR-E:
- was worse than the existing semantic router by +0.00046 nats/token [+0.00035, +0.00058];
- tied the generic path (+0.00006 [−0.00002, +0.00014]);
- recovered −4% of the oracle advantage.

**The specialists stay specialised under RECR routing** (exposure-matched ΔI +0.0020, CI > 0). The value they hold at document starts and transitions is real in hindsight, but it is not causally recoverable from residual evidence.

## 1. What ran

- **Stage 0:** audit; manifests; CPU tests T1–T8; a GPU latency micro-benchmark.
  - Tests T1–T8 all pass: policy causality under future-token and future-evidence perturbation (0 violations, non-vacuous), feature causality, replay and per-document determinism, bitwise head-training resume, exactness and compute accounting, partition disjointness, κ calibration, label isolation.
- **Stage 1:**
  - 36 candidate heads: 2 banks × 3 bank seeds × 2 router seeds, plus 2 cross-fit folds each, at 50M token presentations per full head;
  - generation 0 on CSEARCH (all conditions plus oracle diagnostics);
  - a follow-up evidence-reliability diagnostic, then the stop decision;
  - CSELECT once.
- **Stage 2 (245M / VAL):** not reached. It is infeasible anyway (next section).

## 2. Provenance and the scope reduction

**Reused, hash-verified (`reuse_manifest.json`, 100 files):**
- spb1 g0 banks SEM2 / SHUF2 / GEN32, seeds 0–2 (`runs/g0_s*/ckpt_50000000.pt`);
- spb1 S2_g1 routers R_SEM / R_SHUF / R_GEN / R_COARSE16, router seeds 0 and 1 (`.pt` plus SEL routes);
- spb1 SEL per-token CE tables of all 32 specialists of every bank, plus the SEM2 training-path route (the semantic oracle);
- spb1 analyses; orv1 token_feats and parts.

**Missing: the frozen shared base.** TOKEN-4 245M ckpt_p100 seeds 0/1/2 (sha fb113eb6…, 0d46807b…, 40c61348…) were deleted in the disk cleanup. I searched the filesystem and hashed every same-size `state.pt`; no copy exists.
- **Nothing was substituted.** Only the architecture-identical seed-3 checkpoint was used, for latency timing.
- **Why the result is still exact:** the spb1 residual is the last operation before `norm_f` and the LM head. Any policy's CE equals the table entry of the specialist it executes, so RECR was simulated exactly.
- **What the missing base makes impossible:**
  - hidden-state features for the candidate head; it used only the retained causal router outputs instead;
  - new partitions (CALIB-SELECTION, VAL);
  - a 245M confirmation;
  - a third router seed;
  - a 64-way unrestricted router control.

**Data.** spb1 SEL (CALIB-SEARCH half A) was split by paper with a salted hash (`data_manifest.json`, fixed before any result):

| partition | papers | tokens |
|---|---|---|
| UTRAIN | 187 | 2.68M |
| CSEARCH | 177 | 2.15M |
| CSELECT | 139 | 1.56M |

- No real-domain labels were used.
- Region-prior and paper labels served only as evaluation masks and non-deployable diagnostics.

**Prompt numbers, checked against the files:**
- **Reproduced:** intrinsic ΔI +0.00495, SEM−GEN −0.00047, stable −0.00075, first-32 +0.00086, transitions −0.00002.
- **Not spb1 numbers:** the 0.51 / 0.35 / 0.90 subject accuracies and split-half r 0.975 come from scur1 (a K=5 bank on another base). The spb1 causal SubjectNet accuracy is 0.53.

## 3. Method and generations

**RECR-E (preregistered generation 0):**
- **Evidence:** g = log p_e(x_{t+1}) − log p_0(x_{t+1}), applied from t+1 onward.
- **Detector:** C ← max(0, 0.95·C − g). κ is calibrated label-free on UTRAIN to 4 alarms per 1,000 tokens.
- **Document start or alarm:** B = 8 search tokens on the generic path, while the head's top 2 candidates among {32 specialists, S} are shadow-executed. Their exact evidence relative to the generic path is accumulated.
- **Commit:** to the best candidate if its mean evidence is > 0, otherwise to the generic path. The committed option is then held until the next alarm.
- **Candidate head:** linear, 198 causal features (router-choice histories and confidences), 34 outputs; 6,766 parameters and 6,732 MACs per token. Trained by Huber regression on UTRAIN targets.
- **RECR-D:** the same detector, without shadow search.
- **RECR-E_cm:** stable-mode evidence on every 32nd token only.

**Engineering generations: none beyond generation 0** (DECISION_LOG):
- **Observed failure:** coin-flip commit evidence, alarms at chance, and a large loss in stable context from persistent holds.
- **Supporting evidence (CSEARCH s0 r0, 1,500 random windows; sign agreement and correlation between search evidence and the next 64 tokens):**

  | search tokens B | sign agreement (raw) | correlation (raw) | clipped ±0.2 |
  |---|---|---|---|
  | 8 | 0.500 | 0.002 | 0.485 / 0.000 |
  | 32 | 0.508 | 0.033 | — |
  | 128 | 0.512 | 0.034 | 0.513 / 0.103 |
  | 256 | 0.521 | 0.060 | 0.522 / 0.070 |

  Candidate-vs-generic sign agreement stayed ≤ 0.538.
- **Why each permitted factor fails:**
  - B and evidence normalisation cannot create ranking information;
  - κ / γ do not bind: the detection oracle is worse than RECR;
  - head capacity does not decide commits, and RECR-D with head-only commits also loses in stable context;
  - abstention and fallback changes push towards G, which is 0.0007 worse than S in stable context.
- **Decision:** stop rather than tune noise. Generation 0 was frozen and CSELECT opened once.

## 4–6. Results (CE in nats/token relative to the frozen base, mean over 3 bank seeds × 2 router seeds)

**CE vs base by region**

| condition | CSEARCH all | CSELECT all | CSELECT pos 0–31 | CSELECT trans. first 8 | CSELECT trans. first 32 | CSELECT stable |
|---|---|---|---|---|---|---|
| A: G, generic | −0.00123 | −0.00114 | −0.00123 | −0.00141 | −0.00149 | −0.00112 |
| B: S, existing semantic router | −0.00166 | −0.00155 | +0.00007 | −0.00189 | −0.00141 | −0.00180 |
| C: confidence gate (θ 0.7) | −0.00171 | −0.00161 | −0.00042 | −0.00207 | −0.00162 | −0.00179 |
| H: G for the first 32 positions, then S | −0.00169 | −0.00159 | −0.00123 | −0.00189 | −0.00141 | −0.00180 |
| Ugreedy: head per token | −0.00150 | −0.00153 | −0.00079 | −0.00170 | −0.00130 | −0.00175 |
| D: RECR, no search | −0.00132 | −0.00130 | −0.00110 | −0.00145 | −0.00126 | −0.00149 |
| **E: RECR, top-2 search, B = 8** | −0.00114 | **−0.00108** | −0.00079 | −0.00106 | −0.00114 | −0.00124 |
| E_cm: compute-matched | −0.00111 | −0.00112 | −0.00075 | −0.00160 | −0.00120 | −0.00134 |
| F: semantic oracle (non-deployable) | −0.00292 | −0.00268 | −0.01933 | −0.00314 | −0.00181 | −0.00220 |

**CSELECT per bank seed (all tokens)**

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| G | −0.00113 | −0.00123 | −0.00106 |
| S | −0.00157 | −0.00156 | −0.00151 |
| E | −0.00109 | −0.00115 | −0.00100 |
| E_cm | −0.00114 | −0.00115 | −0.00108 |
| D | −0.00131 | −0.00133 | −0.00126 |

E − S is positive in every seed and router seed: +0.00048 / +0.00041 / +0.00051 by bank seed.

**Paired comparisons on CSELECT (95% paper-bootstrap CI)**

| comparison | all | pos 0–31 | trans. first 8 | trans. first 32 | stable |
|---|---|---|---|---|---|
| E − G (P1a) | +0.00006 [−0.00002, +0.00014] | +0.00044 | +0.00035 | +0.00035 | −0.00013 |
| **E − S (P1b)** | **+0.00046 [+0.00035, +0.00058]** | −0.00086 | +0.00083 | +0.00027 [−0.00007, +0.00061] | +0.00055 |
| E_cm − S | +0.00042 [+0.00032, +0.00053] | −0.00082 | +0.00029 | +0.00021 | +0.00045 |
| E − D | +0.00022 [+0.00017, +0.00026] | +0.00031 | +0.00039 | +0.00013 | +0.00024 |
| E − H | +0.00051 [+0.00040, +0.00061] | +0.00044 | +0.00083 | +0.00027 | +0.00055 |
| E − C | +0.00053 | −0.00037 | +0.00101 | +0.00048 | +0.00055 |
| C − S | −0.00007 [−0.00011, −0.00003] | −0.00049 | −0.00017 | −0.00021 [−0.00040, −0.00002] | +0.00000 |
| H − S | −0.00004 [−0.00006, −0.00002] | −0.00130 | 0 | 0 | 0 |
| S − G | −0.00041 | +0.00130 | −0.00049 | +0.00008 | −0.00068 |

**Gates (CSELECT)**

| gate | result |
|---|---|
| P1a | fail |
| P1b | fail |
| P2 first 32 vs G (≤ +0.001) | pass (+0.00044) |
| P2 transition first 8 | fail |
| P2 transition first 32 | fail |
| P3 stable retention | **0.19** (needs 0.75) — fail |
| P4 seeds | fail |
| P5 functional | pass |
| E_cm | the same CE failures; overhead 4.6%, so P7 passes |
| Mechanism (E beats D and H) | not supported: E loses to both |

CSEARCH results match: E − S +0.00052 [+0.00040, +0.00063], retention 0.09.

**Oracle recovery R = (CE_G − CE_X)/(CE_G − CE_oracle), CSELECT**

| condition | R [CI] |
|---|---|
| E | −0.04 [−0.11, +0.01] |
| E_cm | −0.01 |
| D | 0.10 |
| S | 0.26 |
| H | 0.29 |
| C | 0.31 [0.23, 0.39] |

The preferred target of ≥ 0.5 was not met by any causal condition.

## 7. Functional specialisation (CSELECT, exposure-matched, SEM − SHUF under the same policy)

| policy | ΔI [CI] | by seed | specialists (seed 0, router 0) |
|---|---|---|---|
| RECR-E | +0.00197 [0.00164, 0.00230] | +0.00216 / +0.00198 / +0.00177 | 21/25 positive, 12 significant |
| S | +0.00153 [0.00121, 0.00183] | | 27/32 positive, 16 significant |

The bank's specialisation is intact under RECR routing; the shuffled twin's I is ≈ 0. Specialisation is not what failed.

## Mechanism diagnostics

**Change detection (CSELECT, per combination)**
- **Alarms:** 2.8–3.1 per 1,000 tokens; false alarms 2.7–2.9 per 1,000; precision within 64 tokens 0.06.
- **Recall within 64 tokens:**

  | change type | recall@64 |
  |---|---|
  | semantic changes (1,446) | 0.16–0.18 |
  | functional changes (≈4,300; hindsight best specialist changes) | 0.17–0.18 |
  | semantic-only | 0.07–0.18 |
  | functional-only | 0.17–0.19 |
  | random alarms at the same rate (chance) | 0.17 |

- **Delay:** median 91–106 tokens.
- **Reading:** detection is at chance for semantic changes, functional changes, and the cases where they disagree.

**Candidates and evidence (CSELECT)**
- The paper's own pair is in the top 2 at 44–46% of commits.
- The hindsight best of 33 options over the next 64 tokens is in the top 2 at 6–7%.
- Evidence reliability: correlation 0.00–0.045, sign agreement 0.48–0.50.
- Committed options do no better than the fallback over the hold window (−0.00025 to +0.00003).
- 34% of commits fall back to G.
- Execution shares: G 42%, a held specialist 46%, S 11%.

**Head calibration.** Per-token correlation between the predicted and realised advantage of its top option is 0.003–0.006. Decile reliability is monotone but shrunk (slope 0.27–0.47).

**Oracle decomposition (CSEARCH, non-deployable, vs E)**

| diagnostic | Δ vs E |
|---|---|
| O1: commit on realised next-64 value | −0.00137 (upper bound: the same tokens choose and score) |
| O2: paper's own pair as candidates | −0.00075 |
| O3: alarms exactly at semantic changes | **+0.00036 (worse)** |
| O1+O2+O3 | −0.00178 (= oracle F overall; at transitions −0.0048) |
| O4: hindsight best option per region start | −0.004 (selection on noise; not causal value) |

## 8. Compute and overhead

**Analytic MACs per token (CSELECT)**

| condition | overhead vs S path | residuals executed per token | extra LM heads per token | tokens in search mode |
|---|---|---|---|---|
| E | +41.8% | 1.06 | 1.03 | 3.1% |
| E_cm | +4.6% | 1.08 | 0.11 | 3.8% |
| D | +39.3% | — | — | — |

The S-path reference is 48.0M MACs: trunk 28.4M, LM head 19.3M, residual 0.20M, router 0.08M. Router overhead (3 routers + head) is +0.3%.

**Measured on A100** (seed-3 architecture, timing only; prefill of 16 × 1,024 tokens, bf16, eager)

| configuration | prefill time | vs S path | peak memory |
|---|---|---|---|
| base | 60.7 ms | — | |
| S path | 63.5 ms | — | 2.18 GB |
| exact evidence (one extra LM head) | 67.1 ms | **+5.7%** | **3.75 GB (+72%)** |
| search token | 75.4 ms | +18.6% | |

- **Decode components:** LFRModel has no KV cache, so only components were timed. At batch 1: LM head 0.15 ms, residual 0.15 ms, router GRU step 0.36 ms.
- **Spend:** GPU 0.002 GPU-h (timing only; cap 24). CPU about 0.56 h wall: heads 15 min, CSEARCH 15 min, CSELECT 2.5 min.
- **Disk:** 4 MB of artifacts; 201 GiB free at the end. No per-token outputs were stored, and nothing outside this experiment's namespace was written. The protected inventory (870 files) and all 100 reused files re-verified unchanged.

## 9. Why it failed

1. **Residual evidence is too noisy (failure mode 1): the dominant, unfixable cause.**
   - Per-token gains are independent draws with sd 15–50× the differences to be detected.
   - Sign agreement is 0.50 at B = 8 and 0.52 at B = 256, so search commits are random among the candidates and the fallback.
2. **Change detection is slow (mode 2): real but not binding.** Alarms are at chance and arrive about 100 tokens late. Alarms placed exactly at the true semantic changes made RECR *worse*, because more frequent re-search was cutting short bad holds.
3. **Candidate recall (mode 3): secondary.** Own-pair recall is 0.44. Even perfect candidates cannot be exploited with coin-flip evidence.
4. **Specialists do retain value at transitions in hindsight (mode 4 not supported).** The oracle F and O123 show large transition gains, but these depend on the current region's own text.
5. **The oracle advantage is causally unrecoverable with this signal (mode 5): the conclusion.** O123 equals the oracle overall, but every causal variant recovers ≤ 0.31, and RECR recovers less than nothing.
6. **Overhead (mode 6) is not the reason.**
   - E_cm stays within budget (4.6% MACs) and fails identically.
   - Exact evidence is cheap in wall time (+5.7%), though not in memory (+72%).
7. **Backbone cost (not in the request's list).** Holding a single causally chosen option between alarms loses the token-level utility of the unrestricted router in stable context: +0.0005 vs S for E, +0.0003 for D. That accounts for most of RECR's overall deficit.

## 10. What remains causally unrecoverable

**Doc starts:**
- The oracle's document-start advantage (−0.019 nats/token at positions 0–31, against −0.0012 for G) needs knowledge of the paper's or region's subject.
- Neither residual evidence (no information) nor router-output histories supply it: the own-pair recall of 0.44 matches the earlier 0.53 subject-accuracy ceiling.

**Transitions:**
- At transitions the oracle beats G by only 0.0003 on CSELECT, so there is little to recover there at all.
- The best causal transition result is the simple confidence gate C: −0.00021 [−0.00040, −0.00002] vs S at transition first 32.

## 11. Strongest justified next step

- **Drop residual evidence as a routing signal at this scale.** Per-token CE differences between residual specialists are noise-dominated, and that is a property of the bank, not the detector.
- **What does help causally:**
  - a confidence gate C (−0.00007 overall, −0.00049 at positions 0–31, −0.00021 at transition first 32, all CI < 0 vs S);
  - the position fallback H (−0.00004).
- **Proposed experiment:** a preregistered, matched test of a *learned* confidence gate. The router's own uncertainty, plus position, would decide S vs G per token, trained by the routed-LM loss and compared against C, H and a 64-way unrestricted router over both banks.
- **Prerequisite:** restore or re-create the frozen base checkpoints. Any hidden-state router, new partition, VAL, or 245M confirmation of this bank needs the deleted TOKEN-4 seeds 0–2. Re-creating them would mean retraining the banks, which is a user decision and a new experiment.

## Fairness and integrity

- Every condition is evaluated on the same tokens, base tables, banks and router outputs.
- Thresholds were calibrated label-free on UTRAIN.
- The gate threshold θ_C and RECR's configuration were tuned only on CSEARCH, with comparable budgets; RECR had no extra tuning generations.
- All seeds are reported.
- CSELECT was opened once; VAL and test were never opened; no labels entered any feature.
- **Deviations:**
  - 0: two router seeds per bank seed;
  - 1: scope reduction forced by the deleted base;
  - 2: a post-freeze logging-only fix (RUN_LOG key).
- **Pre-freeze looks:** a TRAIN-half pilot and a UTRAIN-only code smoke test; the smoke test set only the overhead stride.
