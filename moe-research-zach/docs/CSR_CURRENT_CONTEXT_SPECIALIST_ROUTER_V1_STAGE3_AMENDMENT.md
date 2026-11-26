# ccsr1 — Stage-3 implementation amendment (joint continuation) and §11 efficiency-study implementation

Written after Stage 2 passed (stage2.json, 2026-11-21) and **before any Stage-3 or efficiency job or result**. It
implements prereg §10 and §11 (prereg sha256 2101fa23…); it does not change any §10 criterion, only fixes how each is
measured. Frozen by its sha256 (`.sha256` sidecar), logged in DECISION_LOG before the first Stage-3 job.

## A. Stage 3a — matched 50M joint continuation (steps 318 → 1111 of the 245M TOKEN-4 plan)

**One process, one base, ten residual banks** (the base never receives a gradient from a bank, so every arm sees
exactly the same base trajectory, rows, order and optimizer budget):

| arm | bank init (cfar1 ckpt_m20) | router deploying it |
|---|---|---|
| `G` | G | its own linear token router (trained jointly, Switch balance, exactly as in cfar1) |
| `SEMf_r` (r = 0,1,2) | L | frozen Stage-1 router `R3_c2e0.5/SEM_s{r}` |
| `SEMj_r` | L | `R3_c2e0.5/SEM_s{r}`, updated by continued R3 distillation |
| `SHUFf_r` | F2h_T500_w240 | frozen Stage-1 router `R3_c2e0.5/SHUF_s{r}` (functional control) |

- **Base:** model and train config of the cfar1 run (`config.json` of runs/…_cfar1_S1_s100), stream
  `main_stream(245e6)` (3,880 steps), steps 318…1110 (50.09M scored tokens), lr `lr_at(step, 3880, t)`. The cfar1
  resume slot no longer exists, so the base starts from milestone m20 (bf16 matmul weights → fp32 master weights) with
  **fresh AdamW moments**; base and bank learning rates are multiplied by `min(1, (step − 318 + 1)/20)` (20-step
  re-warmup). Identical for every arm.
- **Router input:** rms0 of the detached final hidden state of the step's scored tokens from the training forward,
  per document (one document per row; positions 0..n−1 verified), exactly the Stage-1 feature path (`build_feats`
  'base').
- **Specialists:** `e_t = argmax q_t` of the arm's router (computed before any update in the step); the chosen
  specialist is trained by the LM loss on sg(h), sg(W) only (`head_ce_grad`); experts only, AdamW with the cfar1 bank
  settings (betas 0.9/0.95, wd 0.1, lr = base lr), clip 1.0 per bank. **No federation sync, no auxiliary
  specialisation loss**; the FedBank past-view student and EMA buffers are not updated (PV is reported as a stale
  reference only). G: `residual_step` as in cfar1 (experts + router).
- **Joint router (SEMj_r):** on scored tokens with `(pos + step) % 4 == 0`: u = CE_base − CE_k from `all_deltas` of the
  arm's live specialists (before the step's update), log(p_sem(current 256-token region) + 1e-4) from cfar1
  `psem_main`; π = R3(c = 2, η = 0.5, s_u from the Stage-1 router's TRAIN stats); w_t = clip(margin / med_margin,
  0.25, 4) (Stage-1 stats); loss = Σw·KL(π‖q)/Σw + 0.01·balance. AdamW lr 3e-4 constant (the Stage-1 final lr),
  betas (0.9, 0.99), wd 0.01, clip 1.0; one update per step after the step's choices are fixed.
- **Checkpoints:** atomic resumable slot every ≤ 100 steps or 10 min (one kept: a superseded slot is deleted, logged
  with sha256, only after the newer one verifies; the latest slot is retained so that Stage 3b can resume);
  stage-boundary milestone at step 1111 (base bf16 + all banks + routers fp32). 793 steps < 1,000, so the §10 stop
  rule is applied at the stage-boundary checkpoint. Stage 3a runs the step-3880 plan with `--max-steps 1111`.

## B. Stage-3 evaluation (CALIB-SELECTION, all 1,057 papers, same rows and order as Stage 2)

Continued base at the milestone; for each router seed r: `SEMf_r`, `SEMj_r` and `SHUFf_r` each deployed by its own
router; continued G by its router; region teacher T and the stale PV student for reference. Per-token chosen-specialist
CE for every token; the full 16-specialist delta table only on the swap-subset tokens (25% of papers; disk), so the
oracle is reported on that subset. Every other metric is computed with the Stage-2 code (`evaluate_config`) on all
tokens.

## C. Stage-3 criteria (operationalising §10; fixed now)

Per router seed r and SEM arm A ∈ {SEMf, SEMj}, at the stage checkpoint, against the same-step continued G and the
matched `SHUFf_r`:
- ΔCE = CE(A_r) − CE(G); ΔI = I_SEM(A_r) − I_SHUF(SHUFf_r) (exposure-matched functional metric, paper-bootstrap CI).
- **Stop-rule violation:** ΔCE > 0.002 or ΔI ≤ 0.
- **CE preserved or improved:** ΔCE ≤ 0 (reported as improved if ΔCE < that seed's Stage-2 value).
- **Specialisation preserved or improved:** ΔI ≥ 0.5 × that seed's Stage-2 ΔI and its 95% CI lower bound > 0.
- **Arm A passes** iff ≥ 2 of 3 seeds satisfy both preserved conditions and no seed violates the stop rule.

## D. Stage 3b — 245M (only if ≥ 1 SEM arm passes 3a)

The same run continues, all ten banks, to the end of the plan (step 3880). Evaluations (section B) at the 1,000-step
checkpoints 2111 and 3111 and at 3880; checkpoints every ≤ 100 steps plus milestones at those steps. An arm violating
the stop rule in any seed at a checkpoint is marked stopped (its later results are reported, never used); the run
stops when no SEM arm is alive. Final gate at 3880 = section C. If an arm passes: FROZEN_FINAL.json, then a single
VALIDATION evaluation (prepared with the frozen CS2 pipelines; report only). The test split is never loaded. If no arm
passes 3a, the continuation is terminal and reported.

## E. §11 efficiency study (frozen Stage-2 candidate, step-318 checkpoint, CALIB-SELECTION)

- Refresh every r ∈ {1, 4, 8, 16} tokens: the router runs on every token (its GRU state is unchanged), the deployed
  specialist is re-chosen only at positions with pos % r == 0 and held in between. CE (exact table lookup), route
  agreement with r = 1, per router seed.
- Measured speed (CUDA events, warm-up, median of repeats; same GPU): prefill of 64 CS2 rows (~1024 tokens each)
  including the LM head for base, base + G, base + router + specialist at each r; top CUDA kernels (profiler).
  The LFRModel base has no KV-cache decoding path, so one-token decoding (batch 1 and 64) is measured for the added
  components only (G router + expert + head vs causal-router step + specialist + head at each r, LM head alone as the
  reference), and a full-recompute base step at context 1024 is reported as an explicit upper bound. Reported as
  measured; no speed claim from FLOPs.
