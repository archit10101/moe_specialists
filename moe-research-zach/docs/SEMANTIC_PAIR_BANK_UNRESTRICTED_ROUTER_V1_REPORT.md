# semantic-pair-bank-unrestricted-router-v1 (spb1) — final report

**State: TERMINAL_STAGE2_FAIL** (after 1 of 3 permitted engineering generations; stopped under prereg §8 because no permitted factor addresses the measured cause).
Prereg: `docs/SEMANTIC_PAIR_BANK_UNRESTRICTED_ROUTER_V1_PREREG.md` (sha256 6d3dc3f8…2546, frozen before any bank training). Artifacts: `artifacts/semantic-pair-bank-unrestricted-router-v1/` (final_analysis.json, analysis/, DECISION_LOG.jsonl, OBSERVATIONS.md, deviations.jsonl, RUN_LOG.jsonl, run_manifest.tsv, compute_ledger.tsv, RESUME.md). VAL was never opened; the test split was never opened; Stage 3 did not run.

## 1. Answer to the question

*Can label-free semantic groupings create functionally specialised residual experts, while a completely unrestricted causal token router preserves the accuracy advantage of ordinary token routing?*

- **Specialisation: yes, robustly.** All 32 SEM2 specialists show significant exposure-matched own-cluster advantage. Intrinsic ΔI (SEM2 − SHUF2) is +0.00495 [0.00473, 0.00517], positive in 3/3 seeds. Under the deployed causal router the specialisation survives: deployed ΔI is +0.0019, CI > 0 in every seed.
- **Accuracy advantage under unrestricted causal routing: small, not enough.** SEM2 + router beats GEN32 + router by −0.00047 nats/token, CI [−0.00058, −0.00035], after generation 1. The preregistered bar was ≤ −0.001. It beats SHUF2 by −0.00076.
- **Where the advantage is lost:** about 55% of tokens cannot be causally assigned to their paper's semantic cluster. A dedicated causal classifier reaches only 53%, and those misassigned tokens account for 85% of the gap between the deployed router and the training path. This is the "oracle succeeds but unrestricted routing fails → deployment is the bottleneck" branch, as predicted.

## 2. Methodology (as preregistered)

- **Base:** frozen TOKEN-4 245M `ckpt_p100`, seeds 0/1/2 (sha verified at every load).
- **Residual bank:**
  - one residual bank after the final MoE block: h' = x + a_e(rms0(x)), with a_e 384→256→384;
  - W2 zero-initialised; W1 from an init generator identical across conditions;
  - exactly one specialist per token, and no always-on expert.
- **Banks, 50M tokens each (1,581 steps × 32 × 1024), trained jointly on identical batches with one shared base forward:**
  - **SEM2:** K=16 frozen label-free GC-EDL semantic clusters (orv1 hard labels) × M=2. A token's paper cluster restricts it to its pair; a causal linear subrouter chooses within the pair, with Switch balance within the pair only.
  - **SHUF2:** the same, with a balanced persistent random paper→cluster map at identical cluster token shares (share error ≤ 3.7e-6; paper NMI to SEM 0.0009).
  - **GEN32:** unrestricted Switch token routing over 32.
  - **COARSE16:** the existing orv1 condition-A checkpoints (one specialist per cluster, same base, plan, budget, LR, init scheme, half the bank parameters).
- **Optimiser:** AdamW 3e-4, wd 0.1, aux 0.01, clip 1.0, 50-step warmup. Bitwise resume verified on CPU and GPU.
- **Deployed router (Stage 2):**
  - training subrouters discarded; banks frozen;
  - fresh router: [rms0 x_t, causal mean over the last 16 positions] → Linear → GRU64 → Linear(E), 76,192 params, identical for every bank;
  - loss: expected routed CE over the exact frozen per-token CE table, plus `bal`·KL(q̄‖U); no semantic loss, mask, bias or mapping;
  - 2,400 steps × 64 docs, LR 3e-3 cosine;
  - trained on CALIB-SEARCH half B (TRAIN) and deployed by argmax on half A (SEL, 6.40M tokens, 503 papers);
  - 2 router seeds per bank seed. Ablation: a causal SubjectNet posterior as an extra input, router seed 0.
- **Statistics:** CE differences use a paired per-paper ratio bootstrap (2,000), pooled over bank × router seeds. Functional ΔI is exposure-matched (same routing function; strata of frequency × base-CE × position × doc length × margin), with an influence-function paper bootstrap.

## 3. Configurations and seeds

| item | values |
|---|---|
| bank seeds | 0, 1, 2 (paired with base seeds 0, 1, 2); all three promoted, no selection |
| router seeds | 0, 1 per bank seed (6 routers per condition); posterior ablation r0 (3 per condition) |
| gen 0 (prereg) | M=2, hard assignment, sub_lr_mult 1, aux 0.01, router H 64, temp 1, router bal 0.01 |
| gen 1 | router bal 0.001 (router-only, applied to every condition incl. COARSE16 and ablations; banks and tables reused) |
| gens 2–3 | not run (§8) |

## 4. Stage 1 — banks (SEL, pooled over 3 seeds; CE in nats/token relative to the frozen base)

| bank | training-path CE vs base | per-token oracle (best of E) vs base | train-route NMI vs SEM |
|---|---|---|---|
| COARSE16 (oracle cluster) | −0.0034 | −0.117 | 1.00 |
| SEM2 (oracle cluster + causal pair subrouter) | −0.0029 | −0.133 | 0.89 |
| GEN32 (own Switch routing) | −0.0017 | −0.098 | 0.008 |
| SHUF2 | −0.0006 | −0.095 | 0.17 |

- **G1a:** intrinsic ΔI +0.00494 / +0.00487 / +0.00505 by seed, pooled +0.00495 [0.00473, 0.00517]. SEM2 has 32/32 significant specialists; SHUF2 I ≈ −0.0001. **Pass.**
- **G1b:** SEM2 oracle path − GEN32 = −0.00129 / −0.00118 / −0.00135 by seed, mean −0.00127 (predicted 0.0005–0.0015 better). **Pass.**
- **Pair behaviour:** pairs are balanced (max member share ≤ 0.61). Replacing a token's chosen specialist by its partner costs +0.0016.
- **Where the oracle gain sits:** the SEM2-over-GEN32 oracle gain is concentrated at positions 0–31 (−0.0175). At ≥128 it is only −0.0005, in stable context −0.0007, and at transitions ≈ 0. This was an early warning.

## 5. Stage 2 — unrestricted deployed routers (SEL)

### 5.1 Deployed CE vs base (mean of 6 routers)

| condition | gen 0 (bal 0.01) | gen 1 (bal 0.001) | training path (Stage 1) |
|---|---|---|---|
| COARSE16 | −0.00195 | **−0.00225** | −0.0034 |
| SEM2 | −0.00138 | −0.00164 | −0.0029 |
| GEN32 | −0.00119 | −0.00117 | −0.0017 |
| SHUF2 | −0.00083 | −0.00087 | −0.0006 |

### 5.2 Paired comparisons (pooled over 3 bank seeds × 2 router seeds, 95% CI)

| comparison | gen 0 | gen 1 |
|---|---|---|
| **SEM2 − GEN32 (P1)** | −0.00019 [−0.00027, −0.00011] | **−0.00047 [−0.00058, −0.00035]** |
| SEM2 − SHUF2 | −0.00055 [−0.00063, −0.00047] | −0.00076 [−0.00088, −0.00064] |
| SEM2 − COARSE16 | +0.00057 [+0.00050, +0.00063] | +0.00061 [+0.00055, +0.00068] |
| GEN32 − COARSE16 | +0.00076 [+0.00065, +0.00086] | +0.00108 [+0.00095, +0.00120] |
| SEM2+post − SEM2 | +0.00000 [−0.00003, +0.00004] | −0.00004 [−0.00008, −0.00000] |
| SHUF2+post − SHUF2 | +0.00000 | −0.00003 |
| GEN32+post − GEN32 | −0.00001 | +0.00000 |
| SEM2+post − GEN32+post | −0.00019 | −0.00050 [−0.00063, −0.00037] |

SEM2 − GEN32 per bank seed: gen 0 −0.00021 / −0.00011 / −0.00025; gen 1 −0.00046 / −0.00040 / −0.00055. Leave-one-seed-out means in gen 1: −0.00047 / −0.00050 / −0.00043. No single-seed dependence; all six router combinations are negative in both generations.

### 5.3 Position, transition and stable-context breakdowns (pooled)

| mask (tokens) | SEM−GEN g0 | SEM−GEN g1 | SEM−SHUF g1 | SEM−COARSE16 g1 | GEN−COARSE16 g1 |
|---|---|---|---|---|---|
| positions 0–31 (207k) | +0.00104 | +0.00086 | +0.00005 | −0.00157 | −0.00242 |
| positions 32–127 (621k) | −0.00010 | −0.00044 | −0.00074 | +0.00066 | +0.00110 |
| positions ≥128 (5.58M) | −0.00024 | −0.00052 | −0.00080 | +0.00069 | +0.00121 |
| first 32 after a topic change (187k) | +0.00005 | −0.00002 | −0.00027 | +0.00045 | +0.00047 |
| stable context (3.27M) | −0.00041 | −0.00075 | −0.00102 | +0.00078 | +0.00152 |

Under causal routing the SEM2 advantage lives in stable mid- and late-document context. It is reversed at document start, where the oracle had placed most of its gain, and absent after topic changes. COARSE16 is worse than base at positions 0–31: +0.0009 (g0) and +0.0007 (g1).

### 5.4 Exposure-matched functional result (deployed routing)

- Deployed ΔI (SEM2 − SHUF2, same deployed routers): g0 +0.00197 [0.00178, 0.00215]; g1 +0.00187 [0.00170, 0.00204].
- By seed (g0): +0.00210 / +0.00183 / +0.00197.
- About 26/32 SEM2 specialists are individually significant under deployed routing.
- **P2 passes.** Specialisation is not an artefact of oracle routing.

### 5.5 Loads and effective expert counts (gen 1; ranges over 6 routers)

| condition | exp(H) effective experts | max load | switch rate | NMI vs SEM labels |
|---|---|---|---|---|
| SEM2 | 29.3–29.7 / 32 | 0.065–0.073 | 0.55–0.58 | 0.28–0.30 |
| SHUF2 | 29.9–30.2 / 32 | 0.065–0.076 | 0.79–0.81 | 0.03–0.05 |
| GEN32 | 31.0–31.6 / 32 | 0.042–0.051 | 0.82–0.84 | 0.03–0.04 |
| COARSE16 | 14.7–15.0 / 16 | 0.12–0.135 | 0.25–0.31 | 0.34–0.36 |

No dead specialists anywhere (32/32 used). **P4 passes** in both generations (SEM2 minimum eff 31.6 at g0 and 29.3 at g1; maximum load ≤ 0.073).

### 5.6 Gates

| gate | gen 0 | gen 1 |
|---|---|---|
| P1: SEM−GEN ≤ −0.001, CI upper < 0 | fail (−0.00019) | **fail (−0.00047)** |
| P2: deployed ΔI CI > 0, every seed positive | pass | pass |
| P3: every seed < 0 and every LOSO ≤ −0.0005 | fail | fail (LOSO −0.00047, −0.00043) |
| P4: eff ≥ 12, max load ≤ 0.20 | pass | pass |
| promote to Stage 3 | no | **no** |

## 6. Oracle-to-causal gap and diagnosis

| SEM2, pooled | CE vs base |
|---|---|
| per-token oracle (best of 32) | −0.133 |
| own-cluster pair hindsight (best member of paper pair) | −0.032 |
| training path (paper cluster + causal pair subrouter) | −0.0029 |
| deployed unrestricted router, gen 1 | −0.0016 |

GEN32 loses only 0.0005 between its training path and deployment, while SEM2 loses 0.0013. The two engineering diagnoses below were measured on SEL with router seed 0.

**Gen 0 diagnosis: the balance term.**
- SEM2's natural (training-path) loads follow the cluster sizes, with KL(q‖U) = 0.155 nats; GEN32's is 0.009.
- With `bal`=0.01, cluster-faithful routing is charged 0.0016, which cancels about half its gain. On the router objective, the training path scored −0.00145 against the deployed router's −0.00136, so there was almost no incentive to route by cluster. Pair agreement was 0.34.
- Generation 1 lowered `bal` to 0.001 for every condition.
- Prediction: SEM2 improves 0.0003–0.0008, GEN32 changes by less than 0.0002, and SEM−GEN lands at about −0.0006 (range −0.0003 to −0.0010) with P1 still narrowly failing.
- Observed: SEM2 improved 0.00026, GEN32 changed by 0.00002, SEM−GEN was −0.00047, and pair agreement rose to 0.45–0.46 (the falsifier was < 0.40). **Confirmed: the balance term was a real handicap, but only a partial one.**

**Gen 1 diagnosis: causal unidentifiability of the paper-level cluster.**
- 54–55% of tokens are routed outside their paper's pair. Those tokens contribute +0.0011 of the +0.0013 remaining gap: −0.0027 on the training path versus −0.0006 deployed. Member choice within the right pair contributes only +0.0002.
- The SubjectNet, trained directly to predict the paper's cluster causally, reaches 0.53 SEL token accuracy. Feeding its posterior to the router changes CE by −0.00004.
- At `bal` 0.001 the training path scores −0.0028 on the router's own objective versus −0.0016 deployed, so the objective no longer prevents the better solution; the router cannot reach it from causal information.
- Remaining permitted factors:
  - router H: the posterior net shows that more cluster-predictive capacity adds nothing;
  - temperature: routers are already sharp, with mean top-1 probability 0.83;
  - router `bal` → 0: it no longer binds;
  - M=4 or the subrouter LR ratio: these act on in-pair choice, which is 0.0002 of the gap;
  - soft training-time assignment: either keeps the non-causal paper-level target (trading specialisation for robustness), or uses a causal posterior, which changes the hypothesis and repeats the csur1, cfar1 and asrs2 designs that lost to generic routing.

None of these raises causal cluster identifiability. P1 needs a further −0.00053, more than the gen-1 fix delivered. **Decision: stop, per §8.**

## 7. Comparison against every control

- **vs GEN32 (generic unrestricted routing; the primary comparison):** SEM2 is better by 0.0005, small, significant and consistent, but below the preregistered 0.001 materiality bar. At document start it is worse by 0.0009.
- **vs SHUF2 (persistent grouping without semantics):** SEM2 is better by 0.00076 in CE, with deployed ΔI of +0.0019. The effect is semantic, not merely persistent grouping, and SHUF2 is worse than GEN32.
- **vs COARSE16 (one specialist per cluster, half the parameters, oracle-trained):** COARSE16 is the best deployed condition. It beats SEM2 by 0.0006 and GEN32 by 0.0011 overall, and by 0.0015 in stable context. It loses at positions 0–31, where it is worse than base.
  - Caveats: COARSE16 is a reused orv1 checkpoint. Each of its specialists saw twice the tokens of a 32-bank specialist, and it has half the parameters, so it is not a matched condition for P1.
  - It is the most interesting by-product. Coarser semantic specialists are easier to identify causally and deploy better than both finer semantic pairs and generic capacity.
- **vs the oracle:** the oracle is a diagnostic only. The deployed SEM2 router recovers about 55% of its own training-path gain and about 1% of the per-token oracle.
- **Posterior ablation:** CE changes by ≤ 0.00004 in every bank, matching the prediction of < 0.0005.

## 8. Prediction accuracy

| prediction (prereg §10, or written before gen 1) | outcome |
|---|---|
| G1b passes; SEM oracle 0.0005–0.0015 better than GEN | correct (0.00127) |
| intrinsic ΔI > 0 in 3/3 seeds | correct |
| Stage-2 SEM−GEN within ±0.001, central −0.0003; P1 likely fails | correct (g0 −0.00019; g1 −0.00047) |
| deployed ΔI CI > 0 (P2) | correct |
| posterior ablation changes CE < 0.0005 | correct (≤ 0.00004) |
| expected branch: deployment is the bottleneck | correct |
| gen 1: SEM −GEN ≈ −0.0006 (−0.0003 to −0.0010), GEN Δ < 0.0002, pair agreement > 0.45, P1 still fails | correct |
| not predicted | COARSE16 beats every 32-specialist condition under causal routing |

## 9. Interpretation tree

```
semantic oracle succeeds? ── yes (G1a, G1b pass; SEM2 ≫ SHUF2)
└─ unrestricted causal routing beats GEN32 by ≥ 0.001? ── no (−0.00047 after gen 1)
   ├─ SEM2 ≈ SHUF2? ── no (SEM2 better by 0.00076; deployed ΔI +0.0019) → semantics, not persistence, drives the effect
   ├─ CE improves without exposure-matched ΔI? ── no (both present)
   ├─ ΔI without CE improvement? ── partly: ΔI is large and CE improves, but not enough for the capacity
   └─ conclusion: DEPLOYMENT IS THE BOTTLENECK
         ├─ fixable part: router balance term penalised cluster-proportional loads (gen 1 recovered 0.0003)
         └─ remaining part: paper-level cluster causally identifiable for only ~45–53% of tokens
              (accounts for 85% of the remaining deployment gap; no permitted factor addresses it)
```

## 10. Compute and storage

- **GPU:** 2.53 GPU-h total, against a 24 h cap:
  - Stage 1 training 0.51 h;
  - Stage 1 SEL extraction 0.83 h;
  - Stage 2 TRAIN extraction 0.67 h;
  - routers 0.52 h (g0 + g1).
  - Plus earlier Stage-0 GPU tests and benchmarks (gpu_tests, gpu_resume_test). Every job ran through `scripts/gpu_guarded.sh`; no external interruptions.
- **Disk:**
  - The artifact directory is about 13 GB: checkpoints for 3 seeds × 3 banks, extraction tables about 11 GB, routers, analyses.
  - Free space at the end was 111.8 GiB, above the 100 GiB reserve.
  - Nothing outside `artifacts/…spb1` and `.tmp/…spb1` was written.
  - Protected inventory: 805 files, re-verified at the end, all match.
- **Deletions:** only this experiment's scratch (CPU/GPU test outputs, the extraction-fix smoke test, and the superseded first shuffle build), each logged in `deletions.tsv`. The first shuffle build is logged without a sha (deviation 0).
- **Deviations (3):**
  - 0: shuffle-table rebuild, first build deleted unhashed; nothing had read it;
  - 1: an orchestration-only `final` phase added after the freeze;
  - 2: an extraction-route tie bug (topk-of-logits vs argmax-of-softmax), fixed before any extraction output was used.
  - None can affect the conclusion.

## 11. Fairness

- All conditions share the base, data, token order, budget, width, active parameters per token, optimiser, init pairing and router recipe.
- The generation-1 change applied to every condition.
- All bank seeds, router seeds and both generations are reported.
- No validation or test labels, future tokens or domain labels entered any router input.
- COARSE16 is the reused, unmatched reference described in §7.

## 12. Limitations

- **Scale:** 50M-token banks on a frozen 245M base, with one residual bank after the last block. The effects are small in absolute terms (≈ 0.001–0.003 nats).
- **Partition:** all decisions use CALIB-SEARCH. Per-paper bootstrap over 503 SEL papers; VAL was not reached.
- **Clusters:** the semantic clusters are paper-level hard labels. The causal identifiability limit is specific to this paper-level grouping and to a GRU64 router over a 16-token window plus the hidden state. A larger-context router was not tested; within permitted factors the posterior ablation stands in for it.
- **COARSE16:** reused from orv1. It differs in tokens per specialist and total parameters.
- **Engineering coverage:** only one engineering generation was run. Soft assignment and M=4 were not tested because the diagnosis did not support them; that is a judgement, documented in DECISION_LOG.

## 13. Next experiment

The measured lever is **causal identifiability of the grouping, not specialisation**. Two directions follow:

1. **Group by what a causal router can see.** Build the K clusters from a causally predictable target, for example the cluster structure of the causal posterior itself, or a coarser K chosen where causal accuracy is high. Then train specialists under it, matched against SHUF and GEN.
   - COARSE16's deployed win (−0.0011 vs GEN32 with half the parameters) suggests coarse, identifiable semantic specialists are the promising regime.
   - Its document-start loss suggests pairing it with a generic fallback specialist for low-confidence early positions. That fallback must be matched in every control.
2. **A matched test of COARSE16 against GEN16/GEN32.** Use identical tokens per specialist and identical total parameters, with the same unrestricted causal router, and a prereg gate on stable-context CE.
