# Combined conclusion — csur1 (Experiment A) and cfar1 (Experiment B)

**Both experiments are negative. Neither passed its preregistered Stage-1 gates, so neither was run at 50M or 245M tokens. VAL was never opened, and the test split was never read.**

## The question

Can label-free semantic information produce subject-specialised residual experts that improve accuracy once they are deployed with a causal router? The comparison is against a parameter- and FLOP-matched generic token-routed residual (G).

## The answer

1. **Functional specialisation was created.**
   - In A, cross-fitted utility plus a semantic prior gave an exposure-matched own-vs-swap interaction above the shuffled-prior control, positive in 13 of 14 candidates.
   - In B, hard semantic groups gave the same effect (9 of 9).
   - When the current region's subject is known, which is information available only in training, the semantic specialists beat G (A: by up to 0.0054 nats; B: by about 0.007).
   - Shuffled-prior specialists do not beat G.
   - These are single-seed, 20M-token Stage-1 results on SEL.

2. **It did not survive the causal student.**
   - The deployable router keeps about 1/3 (A) or 1/5 (B) of the semantic advantage.
   - Its CE is worse than G in every candidate of both experiments: A by +0.005 to +0.007, B by +0.012.
   - A larger router input (A's generation 1) changed this by only about 0.001.

3. **Accuracy did not improve** relative to the matched generic residual.

4. **Semantic vs shuffled:**
   - Semantic beats shuffled on exposure-matched function in both experiments, and on CE in A.
   - In B, the shuffled student has better CE: persistent paper-level groups are predictable from the past, but their specialists are near-generic.

5. **FedAvg:**
   - Centered personalised FedAvg was CE-neutral relative to local training at 20M. The common component was only 9–15% of the local updates, so there was little to transfer.
   - Standard FedAvg erased useful differences, and erased more the more often it synced.

## Causal explanation

The value of a subject specialist lies in knowing the *current* region's subject. A causal router that sees only past views cannot recover that reliably:
- the within-document subject switch rate is about 0.31;
- the first view carries no information.

One mis-chosen specialist per 128-token view does worse than per-token generic routing. Both experiments failed at this deployment bottleneck, not at specialist training.

## Uncertainty and seed consistency

- Seed consistency was not tested (single-seed Stage 1 by design).
- The deployment gap (+0.005 to +0.012) is 2.5–6 times the 0.002 gate. It is uniform across 23 configurations, 2 router architectures and 2 training schemes.

## What remains unproven

- Whether deploying with a router that sees the current view (token-level or within-view) would let the semantic specialists beat G. This is the natural next question.
- Effects at longer training (50M/245M) and across seeds.
- Real-domain (M2D2) specialisation on VAL.

## Reports

- `docs/CSR_CROSSFIT_SEMANTIC_UTILITY_RESIDUAL_V1_REPORT.md`
- `docs/CSR_CENTERED_FEDAVG_RESIDUAL_V1_REPORT.md`

**Compute:** A 4.48 GPU-h, B 1.19 GPU-h.
