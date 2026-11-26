# csr-current-context-specialist-router-v1 (ccsr1) — final report

**Verdict.** Stage 1 PASS, Stage 2 PASS, Stage 3a FAIL (terminal); Stage 3b (245M) and VALIDATION were not run.

On frozen specialists, a strictly causal current-token+context GRU router deploys the existing semantic local bank:
- **vs G:** better than the parameter- and FLOP-matched generic bank G, by 0.0038–0.0041 nats in 3 of 3 router seeds on held-out CALIB-SELECTION papers.
- **vs the teacher:** it keeps about 75% of the region teacher's semantic-over-shuffled advantage.
- **Function:** functionally semantic, exposure-matched.

This advantage did **not** survive the preregistered matched 50M joint continuation:
- **Frozen routers:** the semantic arm fell behind G (+0.0013).
- **Jointly-updated routers:** the arm stayed marginally ahead of G (−0.0004), but kept only about 30% of its exposure-matched specialisation.

Under the frozen criteria neither arm passes. All results come from one specialist-training seed (cfar1 seed 100); the three router seeds measure router variance only.

## Documents and artifacts

| item | location |
|---|---|
| Prereg (frozen) | `docs/CSR_CURRENT_CONTEXT_SPECIALIST_ROUTER_V1_PREREG.md` (sha256 2101fa23…) |
| Stage-3 amendment (frozen before any Stage-3 job) | `docs/CSR_CURRENT_CONTEXT_SPECIALIST_ROUTER_V1_STAGE3_AMENDMENT.md` (sha256 b220e101…) |
| Evidence summary | `docs/CSR_CURRENT_CONTEXT_SPECIALIST_ROUTER_V1_EVIDENCE.md` |
| Logs | `artifacts/csr-current-context-specialist-router-v1/`: OBSERVATIONS.md, DECISION_LOG.jsonl, RUN_LOG.jsonl, deviations.jsonl, run_manifest.tsv, compute_ledger.tsv, disk_ledger.tsv, deletions.tsv, checkpoint_ledger.tsv, RESUME.md |
| Machine-readable results | stage1.json, stage2.json, stage3/stage3_m1111.json, efficiency/{refresh_ce,speed}.json, final_analysis.json |
| Code | ccsr1_*.py, src/csr/ccsr1_*.py |
| Stage-3 run | runs/20261121T194254Z_ccsr1_S3 (milestone ckpt_m1111; resume slot retained) |

## Setup

- **Reused artifacts (hash-verified, no retraining):** the cfar1 Stage-1 checkpoint (TOKEN-4 replicate base, 103.6M params, step 318) with three banks:
  - SEM = L, the local semantic bank;
  - SHUF = F2h_T500_w240, the paired never-synchronised shuffled control;
  - G, token-routed.

  Each bank is 16 specialists, 384→256→384.
- **Router:** feature [rms0(h_t), causal mean of the last 16 states] → Linear → GRU(64) → 16 logits; 75,152 params, under the 0.1%-of-base cap. Causal per token, one specialist per token, no fallback to G.
- **Exactness:** the router's CE is an exact table lookup over all 16 specialists' per-token CE.
- **Tests:** causality and equal compute were checked by tests T1–T10:
  - future tokens cannot change q_t;
  - cached and sequential routing match;
  - batched and one-token routing match;
  - no teacher signal reaches inference;
  - h_t is legal;
  - no future-region leak.
- **Partitions (paper-disjoint):**

  | partition | role | papers | tokens |
  |---|---|---|---|
  | TRAIN = CALIB-SEARCH half "EVAL" | router supervision | 498 | 6.1M |
  | SEL = CALIB-SEARCH half "SEL" | Stage 1 | 503 | 6.4M |
  | CS2 = CALIB-SELECTION | Stage 2; opened after FROZEN_S2.json | 1,057 | 12.5M |

  VALIDATION was never opened and the test split was never loaded. No M2D2 domain label was used before all decisions were made.
- **Search:** 12 R3 configurations plus the R1 and R2 controls, 3 router seeds each, and the SEM and SHUF banks: 96 routers. Selection used lowest mean CE among configurations passing G1–G6, never NMI.
- **Engineering generations:** none were needed.

## Stage 1 — CALIB-SEARCH selection half (pass)

- 11 of 12 R3 configurations pass G1–G6 in every seed. The three failures (c0.5, c0.5e0.5, e0.5) fail only G3, retention below 0.6.
- R1 passes; R2 fails at every τ.
- **Selected:** R3_c2e0.5 (τ = 2·s_u, η = 0.5, H = 64, λ = 0.01). Mean CE 4.65124, against 4.65148 for the base point; the 2.4e-4 gap is outside the 1e-5 tie band.

| SEL (CE, nats) | value |
|---|---|
| base | 4.67186 |
| G | 4.65504 |
| past-view student (cfar1) | 4.66703 |
| current-region teacher (SEM) | 4.64728 |
| per-token oracle (SEM) | 4.41530 |
| **R3_c2e0.5 student**, s0 / s1 / s2 | 4.65121 / 4.65128 / 4.65123 (−0.0038 vs G in each seed) |
| R1 student | 4.65256 (−0.0025 vs G) |
| R2_c2 student | 4.66049 (+0.0055 vs G) |

## Stage 2 — CALIB-SELECTION, everything frozen (pass)

**CE (nats).** Per-seed values are s0 / s1 / s2.

| | CE | − G | − past-view | retention |
|---|---|---|---|---|
| base | 4.61228 | +0.01725 | | |
| G | 4.59503 | 0 | | |
| past-view student | 4.60667 | +0.01164 | | |
| current-region teacher SEM / SHUF | 4.58726 / 4.60347 | −0.0078 / +0.0084 | | |
| per-token oracle SEM | 4.35767 | −0.237 | | |
| **R3_c2e0.5** | 4.59094 / 4.59119 / 4.59097 | **−0.00409 / −0.00384 / −0.00406** | −0.0157 / −0.0155 / −0.0157 | 0.744 / 0.746 / 0.748 |
| R1 (region-prior distillation) | 4.5923 / 4.5925 / 4.5922 | −0.0027 / −0.0026 / −0.0028 | −0.0144 | 0.67–0.68 |
| R2_c2 (utility softmax) | 4.6009 / 4.6004 / 4.6002 | +0.0059 / +0.0053 / +0.0052 | −0.006 | about 0 |

**Gates for R3_c2e0.5.** Every gate passes in every seed.

| gate | criterion | result |
|---|---|---|
| G1 | CE ≤ G + 0.002 | pass: better than G in 3 of 3 seeds |
| G2 | mean CE ≥ 0.003 better than the past-view student | pass: −0.0156 |
| G3 | retention ≥ 0.6 | pass: 0.74–0.75 |
| G4 | functional semantic − shuffled > 0 | pass: +0.0088 / +0.0091 / +0.0089 |
| G5 | no collapse | pass: load_max 0.18, effective experts 11.7, 0 dead |
| G6 | gains not only from the first 128 tokens or one subject | pass: positions ≥ 128 are −0.0047 vs G; 16 of 16 groups improve vs past-view, max share 0.14 |

**Deployment detail (R3, seed 0).**
- **Regret:** 0.0037 against the teacher, 0.233 against the per-token oracle.
- **Teacher gain over G retained:** 49–53% of the teacher's gain over G.
- **Utilization:** effective experts 11.7.
- **Switch frequency:** 0.13 per token, against 0.24–0.29 for the SHUF router.
- **Post-switch tokens** (first 64 tokens after a change of region label, 5.6% of tokens): +0.0023 to +0.0029 vs G and +0.0054 to +0.0060 vs the teacher.
- **Stable tokens** (38%): −0.0074 vs G and +0.0013 vs the teacher.

**Position bins** (CE change vs base; G is −0.017 throughout):

| positions | student | teacher | past-view |
|---|---|---|---|
| 0–31 | −0.0090 | −0.039 | +0.011 |
| 32–127 | −0.0203 | −0.028 | +0.009 |
| 128–511 | −0.0215 | −0.024 | −0.0075 |
| 512+ | −0.0222 | −0.024 | −0.0082 |

**Functional specialisation.**
- **Exposure-matched own-specialist advantage:** I_SEM 0.0092–0.0094 vs I_SHUF 0.0001–0.0005.
- **Ordinary (unmatched):** 0.017 vs about 0.
- **Swap matrix:** the own specialist beats the mean of the others by 0.027 on average, and is better for 16 of 16 groups.
- **Individually positive specialists:** 16 of 16 (11–12 significant), largest positive share 0.17.
- **NMI with semantic labels (post hoc):** 0.395 for the SEM router vs 0.048 for the SHUF router.

**Post-hoc M2D2 real-domain breakdown** (after all decisions). Student − G by domain d0–d4: −0.0050, −0.0017, −0.0026, −0.0052, −0.0058 (seed 0; same signs in all 3 seeds). The gain is present in every domain.

## §11 efficiency study (frozen Stage-2 candidate; report only)

**Refresh every r tokens.** The specialist is held between refreshes; the router still runs every token. The table gives the mean over 3 seeds; all per-seed signs are identical.

| r | CE − G | agreement with r = 1 | switch rate |
|---|---|---|---|
| 1 | −0.00400 | 1 | 0.134 |
| 4 | −0.00343 | 0.89 | 0.040 |
| 8 | −0.00312 | 0.86 | 0.021 |
| 16 | −0.00275 | 0.84 | 0.012 |

**Measured speed (A100 80GB PCIe, eager PyTorch, bf16).**
- **Prefill** (64 rows, 63k tokens, including the LM head): base 176.8 ms; +G 179.4 ms (+1.5%); +router+specialist 183.3–183.6 ms (+3.7–3.9%). No difference between refresh rates. The added cost is the router's sequential GRU and feature path (+36 kernel types).
- **Decoding:** LFRModel has no KV-cache path, so only the added components were timed. Batch 1: G 0.67 ms per token; router+specialist 0.87 (r = 1) to 0.81 (r = 16). Batch 64: 1.65 vs 1.83–1.81.
- **Conclusion:** coarser refresh keeps most of the CE gain but buys no measured speed in this implementation.

## Stage 3a — matched 50M joint continuation (fail, terminal)

**Setup (amendment §A).**
- One process, one base continuing the TOKEN-4 245M plan from step 318 to 1111 (50.1M tokens, fresh AdamW moments, 20-step re-warmup).
- Ten banks:
  - G with its own router;
  - SEMf_r: L with the frozen Stage-1 router;
  - SEMj_r: L with its router updated by continued R3 distillation;
  - SHUFf_r: the shuffled control with its frozen router.
- Specialists are trained by the LM loss only: no federation and no auxiliary objective.
- **CPU tests:** bitwise resume, base independent of banks, frozen routers unchanged, SIGINT exits 77 and resumes.

**Results** (CALIB-SELECTION). Continued base CE 3.56419; continued G 3.55696, so G's gain over base shrank from −0.017 to −0.0072.

| arm (s0 / s1 / s2) | CE − G | exposure-matched ΔI (Stage 2) | retention | switch rate |
|---|---|---|---|---|
| SEMf | +0.00124 / +0.00144 / +0.00128 | 0.0044 / 0.0043 / 0.0051 (0.0088 / 0.0091 / 0.0089) | 0.25–0.33 | 0.27 |
| SEMj | −0.00038 / −0.00041 / −0.00041 | 0.0020 / 0.0035 / 0.0024 | 0.43–0.49 | 0.024 |

**Criteria (amendment §C).**
- **Stop rule:** never violated. CE − G ≤ 0.002 throughout, and ΔI > 0 with the CI lower bound > 0.
- **SEMf:** CE is never preserved (it is worse than G). Specialisation is preserved in 2 of 3 seeds; seed 1 is at 0.47× its Stage-2 value.
- **SEMj:** CE is preserved in 3 of 3 seeds, but specialisation is not preserved in any seed (0.23–0.38×).
- **Outcome:** neither arm has ≥ 2 seeds satisfying both, so neither passes. The 245M gate is closed.
- **Disk:** Stage 3b would also have exceeded the 100 GiB free-disk reserve.

**Diagnosis.**
- **Frozen routers:** as the base drifts, the frozen router's input distribution moves. Its switch rate doubles (0.13 → 0.27) and agreement with the region prior falls to about 0.4. The specialists then train on those choices. On the continued SEMf bank, even the region teacher is only level with G.
- **Joint routers:** continued R3 distillation drives the router to near-document-constant routing (switch 0.024, region agreement 0.70, NMI 0.44). This keeps the bank marginally ahead of G, with the teacher at −0.0018 vs G. But the exposure-matched interaction collapses to roughly 30% of its Stage-2 value.

## Answers to the required questions

- **Did the router recover the teacher advantage?** Mostly.
  - It keeps 74–75% of the current-region teacher's semantic-over-shuffled advantage, in 3 of 3 seeds on held-out papers.
  - It keeps about 50% of the teacher's CE gain over G.
  - Its regret against the teacher is 0.0037–0.0039 nats.
- **Did it match or beat G?** It beat G in every seed: on SEL by 0.0038, and on CALIB-SELECTION by 0.0041 / 0.0038 / 0.0041. This held at frozen specialists. After matched continuation it no longer did with frozen routers (+0.0013), and only marginally did with joint routers (−0.0004).
- **Did functionality stay exposure-matched?** Yes at frozen specialists. ΔI was +0.009 with I_SHUF ≈ 0, all 16 specialists were individually positive, and the swap-matrix diagonal advantage was 0.027. After continuation ΔI stays positive (CI > 0) but shrinks: 49–56% of its Stage-2 value with frozen routers, 23–38% with joint routers.
- **Where did routing fail?** Just after region changes: the 5.6% post-switch tokens cost about 0.005 against the teacher, while stable tokens cost only 0.0013. Early positions also lag (positions 0–31: −0.009 vs the teacher's −0.039). Under continued training, frozen routers fail through input drift, and jointly distilled routers collapse to document-level routing.
- **Seed replication:** 3 router seeds, identical gate outcomes and signs, router-seed spread about 0.0003. Specialist-training replication was not tested: one specialist seed throughout.
- **Joint continuation:** run, matched and preregistered, and it failed the frozen §10 criteria in both arms. Stage 3b and VALIDATION were not run.
- **What remains unproven:**
  - whether the deployment advantage survives at the 245M scale or with a router and specialists co-designed for continued training;
  - whether it replicates across specialist-training seeds;
  - whether it survives with optimizer state carried over (fresh moments were forced by the missing cfar1 resume slot);
  - any real decoding-speed benefit, since the base has no KV-cache path and refresh gave no measured prefill gain;
  - VALIDATION performance.
- **The result is mixed and is not reinterpreted.** The deployment question at frozen specialists is answered positively. The continuation gate is a negative result.

## Deviations (all before the relevant result; no gate, metric or data changed)

1. Before freezing, TRAIN_STEPS was set to 2400 from a TRAIN-only held-out-paper pilot.
2. The Stage-1 analysis crashed on an argument-shape bug in the reporting helper; it was fixed before any result existed.
3. The Stage-3 evaluation was refused by the disk guard. Its outputs are now losslessly compressed, and the pre-freeze pilot scratch (0.17 GB) was deleted with each file's sha256 logged.

The analysis job's disk estimate was also lowered from 0.5 to 0.05 GB; it writes only a small JSON.

## Compute and disk

- **GPU-hours:** 2.83 of the 16-hour cap.

  | stage | GPU-h |
  |---|---|
  | pilot | 0.41 |
  | Stage 1 | 0.75 |
  | Stage 2 | 0.37 |
  | Stage-3 tests | 0.01 |
  | efficiency | 0.01 |
  | Stage 3 | 1.28 |

- **Disk:** 100.2 GiB free at the end, above the 100 GiB reserve. Deletions were confined to this experiment's test and pilot scratch, each logged in deletions.tsv with its sha256.
