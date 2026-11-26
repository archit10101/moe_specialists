# csur1 — Evidence synthesis (written before the preregistration)

Sources read: ORACLE_RESIDUAL_V1 report + diagnostic notes, SEMANTIC_LIFT_245M_CONFIRM_V1, SCRGP_V1,
JOINT_SEMANTIC_RESIDUAL_PRETRAIN_V1, FORECAST_COUNTERFACTUAL_UTILITY_V1, SEMANTIC_UTILITY_EDL_V1, SOFT_CF_USEFULNESS_V2
(and their artifacts where needed). All are read-only inputs.

## What is established

1. **Semantic residual functions exist under an oracle (ORV1).** On a frozen 245M TOKEN-4 base at 50M tokens, a K=16
   residual bank with a non-causal paper-level label-free semantic assignment beat a token-mass-matched shuffled
   assignment by +0.0024 nats (VAL, 4/4 seeds, CI [0.00224, 0.00256]) and a learned token router by +0.0013. Its
   specialists are functionally specific (S_A 0.0048 vs S_B 0.00002; swapping always hurts). About 60% of the gap to the
   token router is in positions < 128 (13% of tokens); late in documents semantic − shuffled stays ≈ +0.0015.
2. **Grouping pressure is mostly generic.** In MoE routing experiments the shuffled grouping reaches ~80% of the
   semantic gain (SEM-EDL, SUE). In JSR (joint pretraining, the same residual bank as here) a per-step label-free
   group-separation loss gave the shuffled control the *same* subject-correlated routing as the semantic one
   (NMI 0.22 vs 0.22) because per-batch group contrast is satisfiable by document discrimination; the semantic
   increment was small (CE −0.0004, usefulness +0.0004, not individually significant) and D was CE-tied with no-group B.
3. **Ordinary ablation/interaction is exposure-confounded (SLC).** Only exposure-matched contrasts count.
4. **Semantic last-layer residuals can be zero-sum within fixed capacity (SCRGP)**; added capacity (ORV1) is different.
5. **Utility signals are noisy and token-idiosyncratic (FCU, SUE, SCU).** For MoE experts, the split-half reliability of
   region utility between the two 128-token halves of a 256-token region was r = 0.091; 63% of |Δ| sits in 5% of tokens;
   confidence-bound scores regress to the mean (ρ ≈ −0.63). Residual-specialist utility reliability has not been measured.
6. **No causal router has recovered the semantic partition yet** (ORV1 token router MI < 1% of H(sem)).
7. **Joint-pretraining residual banks give large generic gains at 50M** (JSR: ≈ −0.011 nats vs the shared TOKEN-4
   replicate for every bank, whatever its routing), i.e. most of the bank's value is generic capacity.

## Consequences for this design

- Cross-fitting is mandatory and its reliability must be measured online (split-half r of block utility, logged).
- The shuffled control must be immune to per-batch contrast: here assignment is by responsibilities, not a per-step
  separation loss; the shuffled prior is a persistent, per-paper relabelling of the same region priors (token-mass
  matched through the orv1 shuffle), so it carries identical non-semantic statistics.
- The deployable claim rests only on the causal student; the first 128-token view has no causal context (the student
  uses its zero-state prediction there); results are reported by position.
- Expected effect sizes are small (≤ 0.002 nats semantic − shuffled); promotion uses exposure-matched functional
  contrasts with paired paper-bootstrap CIs and per-seed signs.
- The generic token-routed residual control is strong and must be parameter- and compute-matched.

## Reusable inputs (hash-verified, `artifacts/csr-crossfit-semantic-utility-residual-v1/inventory.json`)

- TOKEN-4 and EDL 245M p100 checkpoints, seeds 0–3 (VAL reference evaluation only, no retraining).
- GC-EDL SEM K16 prototype network + residualizer + bge-small encoder (the C6/ORV1 semantic pipeline) → region priors.
- ORV1 tables: paper-level semantic labels and token-mass-matched shuffle (main, CALIB-SEARCH, VAL), partitions
  SEL/EVAL/VAL, token features.
- Not reusable: no 245M joint generic residual run exists; ORV1's semantic-only bank is paper-level, frozen-base and
  non-causal → generic (G) and semantic-only (SO) banks are trained in-process under identical conditions.
