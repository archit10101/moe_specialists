# csr-m2d2-domain-prior-v1 — protocol (screening pilot, written before any result)

Written 2026-11-09 ~02:55 EST, before any tokenisation or training. Screening-quality pilot under a hard
deadline; not a confirmation. Changes after results are visible are logged in
`artifacts/m2d2-domain-prior-v1/deviations.jsonl`, never applied silently.

## Question
On a balanced dataset with stable explicit document domains, does a persistent domain prior on the router improve
unrestricted token routing and next-token CE beyond TOKEN-4 and beyond TOKEN-4 given the same domain information?

## Data
- `machelreid/m2d2` at revision 6ae613ba744ee56f645aee9c577d04f7e3d48b30, licence CC-BY-NC-4.0. Only the six L1
  archives `data/<L1>.tar.gz` are downloaded (sha256 verified against the LFS hashes): Mathematics_and_logic,
  Technology_and_applied_sciences, Natural_and_physical_sciences, Health_and_fitness, History_and_events,
  Culture_and_the_arts. Domain ids 0..5 in that order.
- Documents: each archive holds native `train.txt`, `valid.txt`, `test.txt`. A document is one Wikipedia article:
  the text is split on runs of three or more newlines; each piece (title line, blank line, body) is one document,
  stripped. Documents with fewer than 64 tokens are dropped.
- Locked test: the `test.txt` members are hashed as raw bytes (sha256, byte count) while streaming. They are never
  decoded, split, tokenised or inspected.
- Doc hash = sha256(normalised text) (lower-case, whitespace collapsed). Exact de-duplication across all domains and
  the train and valid splits: any hash occurring more than once anywhere is dropped from every place it occurs.
  (No near-duplicate pass, because of the deadline.)
- Splits: native valid → VALIDATION. Native train → partitioned by `u = sha256("m2d2dp-v1-part|" + hash)`:
  u < 0.05 → CALIB-SEARCH pool, otherwise TRAIN pool. Within each pool, documents are taken in increasing
  `sha256("m2d2dp-v1-order|" + hash)` order until the cap of valid tokens per domain: TRAIN 10M, CALIB-SEARCH 0.5M,
  VALIDATION 1M. If any domain cannot reach a cap, every domain is downsampled to the largest common amount.
- Tokenisation: the project Pythia tokenizer, tokens[:1023] + EOS (project convention; truncation counts are reported).
  Valid tokens = targets = length − 1 per document.
- Packing: each packed 1024-token row holds documents of one domain only (no packed row crosses domains). Train rows:
  documents shuffled per domain (seed 17), packed per domain, then rows interleaved by a seeded permutation (seed 17).
  Evaluation rows: per domain in hash order.

## Model (all conditions)
The existing ~28.9M NanoMoE: d 192, context 1024, 8 layers, layer 0 dense (768), layers 1–7 MoE, 32 routed experts
(ffn 128), top-4, shared expert (ffn 256), tied embeddings, document-causal attention, fp32 router with global softmax
top-4, no renormalisation, dropless, aux 0.01, z 0.001. Backbone initialisation identical within a seed (global RNG);
added parameters come from a dedicated generator.

## Conditions
- B0 TOKEN-4: no domain information.
- B1 DOMAIN-CONDITIONED TOKEN-4: a learned domain embedding E_dom (6 × 192, init N(0, 0.02)) added to the token
  embedding at every position of the document. Ordinary unrestricted top-4 routing. No router bias.
- B2 DOMAIN PRIOR: B1's domain embedding, plus in layers 4–7 router_logits = token_logits + alpha[l] · bias[l, domain]
  (bias 4 × 6 × 32, init 0, so step-0 B2 equals B1 exactly; alpha per layer, init 0.5, learned). Every token scores all
  32 experts and takes its own top-4; no pool, mask or prohibition. Layers 1–3 ordinary. Gates are the softmax of
  the biased logits (the same rule as TOKEN-4).
- B3 SHUFFLED-DOMAIN: the architecture of B2, but training documents' domain labels are permuted across training
  documents (seeded permutation, seed 4242; domain frequencies preserved). Evaluation labels are the true ones.
Overhead: B1 1,152 parameters; B2/B3 1,152 + 768 + 4 = 1,924 (< 0.01% of total). Active compute is identical except
an embedding add and a (32-wide) bias add.

## Run plan (fixed priority)
1. Data integrity checks; matched step-0 checks (identical backbone hashes; B2 at step 0 equals B1; B0 unchanged).
2. One 10M-token search seed (seed 100) of B0, B1, B2, B3, evaluated on CALIB-SEARCH.
3. Freeze. No setting is chosen from the 10M results (there is nothing to select); they are screening evidence only.
4. If time remains: B0, B1, B2 for 50M tokens on seeds 0 and 1 (order: seed 0 B0, B1, B2, then seed 1 B0, B1, B2).
5. B3 at 50M (seed 0, then seed 1) only if it is projected to finish before 08:35.
6. No third seed.
Training settings: 16 rows × accum 2 × 1024 (32,768 tokens per step), lr 6e-4, WSD (60 warm-up, 20% decay to 6e-5),
AdamW (0.9, 0.95), wd 0.1 for parameters with dim ≥ 2, clip 1.0, bf16 autocast. Identical for every condition.

## Evaluation
Validation is read only after the settings are frozen (`frozen.json`), once per checkpoint.
Primary: overall validation CE (token-weighted); macro average of the six per-domain CEs; each domain's CE;
B2−B1, B2−B0, B3−B2 per seed with paired per-document bootstrap 95% intervals (2,000 resamples); seed-to-seed spread.
Specialisation diagnostics (validation): domain–expert mutual information by layer (top-4 assignment counts); domain
entropy per expert; within-document routing stability (Jaccard of consecutive tokens' top-4 sets); expert load and
unused experts; load-preserving shuffled-domain null for MI (domain labels permuted across documents); alpha by layer;
fraction of tokens whose top-4 the prior changes (B2/B3, layers 4–7, computed from the same hidden states with the
bias removed); CE on changed versus unchanged tokens, and B2−B1 per-token NLL on those tokens.
Optional functional ablation if time permits: disable each expert's apparent top domain expert on that domain's
validation documents versus other domains' documents.

## Interpretation (fixed)
B2 < B0 but not < B1: domain information helps, the routing prior does not. B2 < B1 (CI excluding 0 in both seeds):
the prior adds value beyond equal conditioning. B3 ≈ B2: the effect is not semantic. More specialisation without lower
CE is not an accuracy success. Differences below ~0.003 nats are within known run-to-run noise.

## Integrity
One GPU job at a time via `scripts/gpu_guarded.sh`; GPU time ledgered; locked test never decoded. Deadlines:
no GPU launch after 08:35 EST, all compute stopped by 08:45, reports by 08:55.
