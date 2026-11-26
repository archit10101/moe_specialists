# Complete Experimental History: Chunk Routing, Semantic EDL, Forecast Routing, and Functional MoE Specialization

**Prepared:** 2026-11-22  
**Project machine:** `lambda2`, repository `/home/zapan4/moe-research`  
**Purpose:** A self-contained record of the methodologies tested, their experimental setups, results, and what was learned.

> This document reports experiments that were actually run. Ideas that were discussed but never executed are separated in the final section. “Negative” means that the preregistered success condition was not met; it does not necessarily mean that every intermediate observation was uninteresting.

---

## 1. Executive summary

The research began with a broad question: **can tokens be routed in larger semantic chunks, or with domain/context information, so that MoE experts specialize more coherently while preserving or improving language-model accuracy and possibly reducing computation?** It later narrowed to a harder question: **can subject-correlated routing be converted into genuine subject-specific expert competence—meaning an expert is more useful per invocation on its own subject, not merely selected more often for that subject?**

The most important findings are:

1. **Hard chunk routing was consistently worse than unrestricted per-token routing.** Fixed, dynamic, semantic, random, and speculative candidate pools reduced router calls but imposed a large accuracy cost. The strongest representative fixed-chunk result was about **+0.110 nats/token worse** than token routing; a later sequence-routing search was **+0.116 to +0.144 worse** at 100M tokens. Implementations were also slower because routing was not the dominant optimized kernel and the chunk logic introduced overhead.

2. **Explicit-label EDL was the strongest ordinary MoE result.** At 245M tokens, explicit domain-label EDL achieved mean validation CE **3.0435 versus 3.0533 for TOKEN-4**, an improvement of about **0.0098 nats/token**, while substantially increasing domain-aligned routing and the original functional-specialization metrics. However, a later exposure-conditioned audit showed that EDL’s increased ordinary ablation interaction is largely explained by sending more subject tokens to particular experts. It did **not** establish that those experts were more useful per invocation than TOKEN-4 experts after matching routing exposure.

3. **Label-free semantic EDL reproduced most of explicit EDL’s routing behavior and remained accuracy-competitive, but semantics added only a small increment over persistent random paper groupings.** At 245M tokens, Semantic C6 had CE **3.0467**, domain NMI **0.0547**, and domain FSI **0.01599**, compared with explicit EDL’s **3.0435 / 0.0574 / 0.01687**. But the shuffled persistent grouping had CE **3.0461**, NMI **0.0452**, and FSI **0.01490**. The semantic condition reliably aligned routing more closely with subjects, yet the per-expert functional interaction over shuffled grouping was essentially zero.

4. **The project’s definition of “functional specialization” had to be tightened.** Early work used ordinary expert removal/replacement interactions. Semantic Lift later showed that this metric is confounded by routing exposure. Semantic Lift increased ordinary interaction by **+1.53e-4**, but after matching tokens on router rank, gate, margin, CE, and position, the effect became **−4.5e-5** with a confidence interval spanning zero. From that point onward, the strongest evidence requires an **exposure-matched or forced-equal-exposure** comparison.

5. **Forecasts of future tokens did not provide the hoped-for routing signal.** FL-EDL matched TOKEN-4 in CE but did not increase specialization. Past-only latent clusters beat forecast clusters. A direct counterfactual study found that even the embedding of the **actual next 128 tokens** increased held-out expert-utility R² by only **0.0041** and closed only **0.4%** of the routing gap. Draft forecasts added similarly tiny information and no realized routing improvement. This rules against “the drafts were merely too weak” as the main explanation for region-level expert routing.

6. **A genuine positive result emerged after adding a separate residual-specialist bank.** In ORACLE-RESIDUAL-V1, a frozen TOKEN-4 base received one of 16 added residual experts selected by a label-free whole-paper semantic assignment. Across four seeds, the semantic oracle improved CE by **0.00337** over the base, beat a mass-matched shuffled assignment by **0.00240**, beat a learned token router by **0.00127**, and produced strong exposure-matched subject specificity (**0.00484 versus approximately zero for shuffled specialists**). This proved that the semantic clusters correspond to distinct learnable residual functions when separate capacity is added.

7. **A causal router could deploy frozen semantic residual specialists successfully.** A current-token plus previous-16-hidden-state GRU router beat a matched generic residual bank by **0.0038–0.0041 nats** in three router seeds and retained about **75%** of the semantic teacher’s semantic-over-shuffled advantage. This is the clearest deployable positive result. It did not survive continued joint training: frozen routers drifted, while jointly updated routers became almost document-constant and lost most exposure-matched specialization.

8. **The remaining bottleneck is now well localized.** Subject-specific residual functions can be learned. The remaining difficulty is co-training a causal router and specialists so that:
   - the router receives enough current-region information, especially at document starts and topic transitions;
   - the specialists remain useful rather than drifting or becoming target-token specialists;
   - the semantic bank remains at least as accurate as a flexible token-routed generic bank on stable text; and
   - the result survives multiple specialist-training seeds and long training.

---

## 2. Common terminology, models, and evaluation rules

### 2.1 Cross-entropy

**CE** is next-token cross-entropy in **nats per token**. Lower is better. A difference of `−0.010` means the first model assigns better next-token probabilities by 0.010 nats/token. Unless explicitly marked as validation, a number may be from a calibration or development split and should not be treated as a final generalization result.

### 2.2 The principal backbones

Two main model scales were used.

1. **Early NanoMoE / TOKEN-4 backbone**
   - Approximately 28.9M parameters.
   - Width 192, eight layers; layer 0 dense and layers 1–7 MoE.
   - 32 routed experts, unrestricted top-4 routing, plus a shared expert path.
   - Context length usually 1,024.
   - Used for the early chunk-routing, M2D2 prior, forecast curriculum, and local-attention studies.

2. **Later LFR / TOKEN-4 backbone**
   - 103,609,728 parameters.
   - Width 384, 12 layers, 32 experts with top-4 routing and a shared path.
   - Final long runs used 245,032,457 tokens and 3,880 steps.
   - Used for FL-EDL, Semantic EDL, the specialization audits, and the residual-specialist line.

### 2.3 What “routing specialization” means

Routing specialization asks whether subject or semantic-group information predicts the experts selected by the router. Metrics included:

- mutual information or normalized mutual information between group/domain and expert;
- topic prediction from routing profiles;
- top-4 overlap between domains;
- number of routing-specialized `(layer, expert)` cells;
- cross-seed alignment of expert profiles.

These show that routing distributions differ by subject. They do **not** alone prove that an expert learned a subject-specific function.

### 2.4 What “functional specialization” means

Several progressively stronger metrics were used:

1. **Group ablation / FSI:** remove or disable a group’s enriched experts and compare the CE damage on that group against matched other groups.
2. **Per-expert replacement interaction:** remove a selected expert, admit the next-ranked expert, recompute downstream layers, and compare the damage on the expert’s assigned subject versus other subjects.
3. **Exposure-matched interaction:** make the same comparison only among tokens that actually reached the expert and match them on router rank, gate, margin, baseline CE, position, and related covariates.
4. **Forced-equal-exposure swap matrix:** force each expert on the same token population and compare its exact downstream CE with alternative experts.

The later experiments established that ordinary FSI and replacement interaction can be inflated by routing concentration. The most credible evidence of functional specialization therefore comes from methods 3 and 4.

### 2.5 Core controls

Most experiments used some combination of:

- **TOKEN-4:** unrestricted top-4 routing baseline.
- **Explicit EDL:** real domain labels drive routing-distribution divergence during training.
- **Semantic grouping:** label-free paper or region clusters from frozen semantic embeddings/prototypes.
- **SHUF:** persistent, mass-matched random paper groupings. This is crucial because any stable grouping can alter routing.
- **Generic residual G:** same residual capacity, but token-routed without semantic grouping.
- **Oracle:** non-causal assignment using the current region or whole paper; diagnostic, not deployable.
- **Data- and compute-matched TOKEN-4:** controls for extra draft-training data or extra training cost.

### 2.6 Experimental integrity

The mature experiments used paper-disjoint train, calibration, selection, validation, and locked-test partitions; frozen preregistrations; paired seeds; identical rows, initialization, steps, and token counts; shuffled semantic controls; hash-verified reused checkpoints; one GPU job at a time; checkpoint/resume tests; and a 100 GiB disk reserve. Many experiments stopped at a preregistered gate and therefore never opened validation. Those are reported as development evidence, not final results.

---

## 3. Reference method: explicit-label Expert Divergence Learning (EDL)

Explicit EDL is the strongest repeated reference condition. It did not use chunk routing. Routing remained unrestricted top-4 per token.

For each labeled group or domain `g`, the model averaged its router distributions within the group:

\[
R_g = \operatorname{mean}_{t\in g} p(e\mid h_t).
\]

The auxiliary loss encouraged different groups to have different mean routing distributions, using pairwise Jensen–Shannon divergence:

\[
L_{\text{EDL}} = \frac{1}{|P|}\sum_{(g_i,g_j)\in P}
-\log\left(\operatorname{JSD}(R_{g_i},R_{g_j})+\epsilon\right).
\]

The main long-run reference used the loss across all MoE layers with `beta = 5e-3`. The auxiliary gradient was about 5% of the LM gradient early and about 2% late. Experts were not assigned rigidly to domains; the loss separated **group-average routing distributions**, while each token retained unrestricted top-4 routing.

At 245M tokens over four seeds:

| condition | mean VAL CE | domain NMI | domain FSI | original per-expert interaction |
|---|---:|---:|---:|---:|
| TOKEN-4 | 3.0533 | 0.0126 | 0.00762 | about 0.00038 |
| Explicit EDL | **3.0435** | **0.0574** | **0.01687** | about 0.00066 |

EDL was reliably more accurate than TOKEN-4 by about **0.0098 nats/token** and strongly increased subject-correlated routing. However, the Semantic Lift confirmation later found that EDL’s **exposure-conditioned** interaction was lower than TOKEN-4’s. Thus the safe conclusion is:

- EDL improved CE and created stronger, reproducible subject-correlated routing.
- Its experts looked more specialized under ordinary ablation metrics.
- It was **not proven** that an EDL expert was more useful per invocation for its subject after controlling for exposure.

---

# Part I — Chunk routing and conditional-compute experiments

## 4. Representative causal semantic chunk routing (`csr-representative-v1`)

### Method

The first broad test compared six conditions over three seeds:

| ID | routing | segmentation | semantic alignment loss |
|---|---|---|---|
| A | unrestricted token routing | none | off |
| B | one candidate pool per chunk | fixed 128-token chunks | off |
| C | pooled | fixed 128 | on |
| D | pooled | semantic boundaries | off |
| E | pooled | semantic boundaries | on |
| F | pooled | matched-random boundaries | off |

The semantic segmenter used a causal streaming representation and targeted a mean chunk length near 128, with approximately 64–256-token limits. A chunk chose a candidate expert pool; tokens inside the chunk routed within that pool. The semantic loss tried to align chunk representation and expert routing.

### Results

| condition | mean validation CE |
|---|---:|
| A: token routing | **4.2198** |
| B: fixed pool | 4.3299 |
| C: fixed pool + semantic loss | 4.3518 |
| D: semantic chunks | 4.3276 |
| E: semantic chunks + semantic loss | 4.3540 |
| F: random boundaries | 4.3259 |

Primary contrasts:

- Hard pooling, `B − A`: **+0.1101** nats; much worse.
- Semantic versus matched-random boundaries, `D − F`: **+0.0017 mean with mixed seed signs**; no benefit.
- Semantic loss on semantic chunks, `E − D`: **+0.0264**; worse.
- Semantic loss on fixed chunks, `C − B`: **+0.0218**; worse.

The later specialization audit found that pooling increased apparent topic coherence and ordinary functional interaction, but every such increase coincided with worse CE. The semantic loss actually **reduced** routing-topic MI by about 0.10–0.13 nats while increasing the original functional metric by only 0.0005–0.0010.

### Conclusion

The experiment rejected the original hard-chunk hypothesis. Semantic boundaries did not beat random boundaries, and constraining tokens to a shared pool removed useful token-level flexibility.

**Source:** `SPECIALIZATION_AUDIT_REPORT.md`; original protocol in `EXPERIMENT_REPORT.md`.

## 5. Sequence-level routing search

### Method

A 29M-parameter model with 32 experts, top-4 routing, and a shared expert compared:

- TOKEN-4 unrestricted routing;
- fixed-length segment routing;
- dynamic causal boundaries;
- mean, GRU, and attention segment summaries;
- multiple segment lengths and boundary thresholds.

Sixteen unique candidates were evaluated across three engineering generations. The best fixed design used 16-token segments with mean+GRU summaries. The best dynamic design used a layer-0 causal change-point detector targeting length 64 plus a GRU.

### Results at 100M tokens, three seeds

| method | mean CE | delta vs TOKEN-4 |
|---|---:|---:|
| TOKEN-4 | **4.2239 ± 0.0034** | — |
| fixed sequence router | 4.3400 ± 0.0084 | **+0.1161** `[+0.1157,+0.1166]` |
| dynamic router | 4.3679 ± 0.0067 | **+0.1440** `[+0.1435,+0.1445]` |

The gap grew with training: roughly +0.016 at 5M, +0.06 at 30M, and +0.12 at 100M. Shorter segments were consistently better. Long segments underused experts, and selected gate mass was only 17–26%, versus 34–53% for TOKEN-4.

Router calls fell by roughly 16–72×, but measured training was **3.85× slower** for fixed routing and **1.89× slower** for dynamic routing because the sequential Python/CPU logic and small irregular kernels dominated. Decoding was unchanged or slower.

### Conclusion

The result was strongly negative. Lower router-call count did not imply lower wall time, and one routing decision per segment was too restrictive.

**Source:** `SEQUENCE_ROUTING_SEARCH_V1_REPORT.md`.

## 6. Speculative and adaptive candidate-pool routing

### Method

On the 28.9M NanoMoE, this experiment restricted each token’s top-4 selection to a cached candidate pool:

- adaptive causal pools refreshed from the model’s recent routing;
- speculative pools proposed using a 7.49M dense draft model trained for 100M tokens;
- random causal and random speculative controls;
- ordinary TOKEN-4.

The adaptive finalist used a maximum hold of four tokens and pool size 16. The speculative finalist used four-token blocks, three speculative proposals, and pool size 8.

### Results at 100M tokens

| method | mean CE | delta vs TOKEN-4 |
|---|---:|---:|
| TOKEN-4 | **4.2223** | — |
| adaptive pool | 4.2437 | **+0.0214** |
| speculative pool | 4.3091 | **+0.0868** |
| random causal | — | +0.1065 |
| random speculative | — | +0.1483 |

Learned content pools beat random pools, proving that the selectors contained signal, but they could not recover unrestricted routing. Adaptive recall of the baseline top-4 was 0.811; speculative recall was 0.475. The speculative teacher had a large early training advantage because it was pretrained, but that advantage reversed between approximately 30M and 50M tokens.

Plain speculative decoding applied to TOKEN-4 did improve greedy decoding throughput by about **38%, 37%, and 22%** at batch sizes 1, 4, and 16. The pool-routed models themselves were slower than matched controls in training, prefill, and decoding.

### Conclusion

Speculative decoding can accelerate generation, but using draft predictions to restrict the MoE expert pool harmed accuracy and did not accelerate this implementation.

**Source:** `SPECULATIVE_POOL_ROUTING_V1_REPORT.md`.

## 7. Conditional local attention

### Method

The experiment compared full attention with:

- fixed sliding-window local attention;
- hybrid local/full layers;
- a learned gate that selected only some queries for full global attention;
- compact block-sparse execution so unselected global queries were not secretly computed.

The preregistered search selected a **128-token sliding window in every layer** (`LOCAL-W128`) for the final 100M-token comparison.

### Accuracy results, three seeds

| condition | mean validation CE |
|---|---:|
| FULL-TOKEN-4 | **4.2221** |
| LOCAL-W128 | 4.2466 |
| difference | **+0.0245** `[+0.0238,+0.0252]` |

Local attention was better on positions 0–128 by about −0.026 nats, but increasingly worse at longer positions: +0.005 for 128–256, +0.036 for 256–512, +0.046 for 512–768, and +0.072 for 768–1024. Local candidates looked better at 5M–20M tokens, but the ordering reversed around 45M–75M.

### Speed results

- Matched-batch training: candidate/baseline time **0.992**.
- End-to-end 100M training: **0.996**.
- Decoding, batch 8 with a 1,024-token prompt: **0.998**.
- Prefill: **1.141**, or 14.1% slower.

Theoretical forward FLOPs fell by 5.5%, but attention was only about 4% of step time after the block-sparse kernel. Decoding was launch-overhead limited. No KV-cache eviction was implemented, so no memory saving was claimed.

The learned conditional-global version was genuinely sparse and interpretable—it selected topic/discourse words—but trained 21–23% slower and never beat fixed local attention.

### Conclusion

At context 1,024, local attention was neither accurate enough nor meaningfully faster. The result does not rule out gains at much longer contexts with a better kernel and KV eviction.

**Source:** `CONDITIONAL_LOCAL_ATTENTION_V1_REPORT.md`.

## 8. Value-routed compute at context 8,192

### Method

This was a diagnostic on a minimally trained 8,192-context TOKEN-4 checkpoint: 5.03M training tokens, CE about 6.70. Candidate paths combined:

- local attention with a 512-token window;
- optional global attention in selected layers;
- one, two, or four routed experts per 128-token block.

The goal was to determine whether cheaper paths could save at least 10% prefill time without unacceptable next-token loss.

### Results

- Full reference prefill, batch 4 × 8,192: **43.98 ms**.
- Local/top-4: CE −0.0011 but time ratio **1.059**.
- Local/top-2: CE +0.0107, time ratio **1.084**.
- Local/top-1: CE +0.0397, time ratio **1.088**.
- Every path was **5.9–10.3% slower**.

The value of long-range attention was nearly zero at this undertrained checkpoint, but extra experts were valuable almost everywhere: reducing 4→2 experts hurt 79% of cases, and 2→1 hurt 96%. Sort/scatter/mask overhead and the shared output layer dominated.

### Conclusion

The diagnostic gate failed. No controller, pilot, or decoding benchmark was run. This was not a mature long-context result because the model was severely undertrained.

**Source:** `VALUE_ROUTED_COMPUTE_V1_REPORT.md`.

## 9. Multiscale causal routing priors

### Method

Instead of hard chunking, this experiment kept unrestricted token routing and added projected context summaries to router logits:

\[
z_t = W h_t + \alpha W\bar h_{\text{local}} + \beta W\bar h_{\text{broad}}.
\]

The local view covered up to 16 tokens in the current chunk; the broad view summarized causal history up to 1,024 tokens. Variants placed these signals in early, late, or all MoE layers and tried semantic boundaries.

### Results

At the 5M-token calibration scale, beta values 0.25, 0.5, and 1.0 changed CE by **+0.00024, +0.00512, and −0.00044**. Alpha values 0.25, 0.5, and 1.0 changed CE by **+0.00271, +0.00266, and +0.00701**. None met the frozen improvement rule, and semantic boundaries had no consistent benefit. The measured pooled within-seed noise was about 0.0026 nats.

### Conclusion

No candidate advanced to main training, validation, or functional evaluation. Multiscale context did not provide a robust early accuracy signal.

**Source:** `MULTISCALE_ROUTING_V1_REPORT.md`.

---

# Part II — Explicit domains, draft forecasts, and context priors

## 10. M2D2 stable domain prior with unrestricted token routing

### Method

Six balanced M2D2 Wikipedia domains were used: Mathematics and Logic, Technology, Natural Science, Health, History, and Culture. After strict exact deduplication, each domain contributed 7.304M training tokens; total training data was 43.83M tokens.

The 28.9M NanoMoE conditions were:

- **B0:** ordinary TOKEN-4.
- **B1:** TOKEN-4 plus a learned six-way domain embedding added to token representations.
- **B2:** the same embedding plus domain-dependent router biases in layers 4–7, with learned scale alpha; top-4 routing remained unrestricted.
- **B3:** B2 with domain labels permuted across training documents.

B2 added only 1,924 parameters, about 0.0067%.

### Results at 43M tokens, two final seeds

| condition | seed 0 CE | seed 1 CE |
|---|---:|---:|
| B0 TOKEN-4 | 4.5163 | 4.5361 |
| B1 domain embedding | 4.4395 | 4.5284 |
| B2 domain prior | 4.4396 | 4.5250 |
| B3 shuffled labels | 4.4576 | 4.5370 |

The key comparison `B2 − B1` was **+0.0001** `[−0.0003,+0.0006]` in seed 0 and **−0.0033** `[−0.0038,−0.0028]` in seed 1. The preregistered rule required both seeds, so it failed.

Meaningful domain information did matter: B3 was 0.012–0.018 nats worse than B2. But B3 reproduced 77% of the large seed-0 B1/B2 improvement over B0, indicating a substantial extra-input/training-trajectory effect rather than domain knowledge.

B2 modestly increased domain–expert MI and changed the top-4 in 6–9% of tokens per affected layer, but the changed tokens were no better than unchanged tokens. B0 already contained domain-selective experts.

### Conclusion

Providing domain information in the representation helped relative to meaningless labels, but adding a persistent domain routing prior on top of the same domain embedding did not reliably improve CE. Extra routing specialization was not tied to accuracy.

**Source:** `M2D2_DOMAIN_PRIOR_V1_REPORT.md`.

## 11. Domain forecast specialization curriculum

### Method

Five balanced M2D2 arXiv domains were used: computer science, mathematics, physics, quantitative biology, and statistics. Per domain:

- 6M tokens trained a shared draft with domain adapters: 30M draft tokens total;
- 22.5M tokens trained the main model: 112.5M total;
- 1.5M calibration and 1M validation tokens.

During the first **40%** of main training, four-token draft forecasts and/or an explicit domain-group prior influenced routing. The next **10%** annealed the assistance to zero. The final **50%** used ordinary TOKEN-4 routing and loss. Conditions included token-only, label-only, forecast-only, both (`SOFT`), wrong-domain forecasts, a unique-data-matched TOKEN-4, and a compute-matched TOKEN-4.

### Final validation CE, three seeds

| condition | mean CE |
|---|---:|
| TOKEN-MAIN, 112.5M | **3.5905** |
| LABEL→TOKEN | 3.5974 |
| FORECAST→TOKEN | 3.5952 |
| SOFT label+forecast→TOKEN | 3.6039 |
| WRONG-FORECAST→TOKEN | 3.6027 |
| TOKEN-ALL-DATA, 142.5M | **3.5157** |
| TOKEN-COMPUTE, 169.2M | **3.4609** |

SOFT was +0.0134 worse than TOKEN-MAIN, +0.0065 worse than LABEL, +0.0882 worse than the unique-data control, and +0.1430 worse than the compute-matched control.

### Specialization result

The domain prior did create durable specialization. At 100% training, SOFT had domain-expert MI 0.063 versus 0.028 for TOKEN-MAIN; all-layer provisional-group specificity was 0.075 versus 0.001; and expert identities were reproducible across seeds. Specialization continued growing after all assistance had been removed.

Forecasts alone did **not** create specialization. FORECAST tracked TOKEN-MAIN. Correct and wrong domain adapters had almost identical next-token forecast accuracy, 0.270 versus 0.267, so the wrong-forecast control was weak. Greater specialization did not correlate with lower CE.

The forecast path added 15% training step time. Amortized full cost was 1.50× a TOKEN-MAIN run. Final exported models had identical inference speed because drafts and curriculum modules were removed.

### Conclusion

This experiment proved that explicit routing pressure can create durable domain specialists, but those specialists did not improve accuracy. Spending the same data or compute on ordinary token training was much better.

**Source:** `DOMAIN_FORECAST_CURRICULUM_V1_REPORT.md`.

## 12. Latent forecast router distillation V1

### Method

This line used forecast-informed teachers, context memories, counterfactual first-order estimates, and a student router intended to retain training-time forecast guidance after the drafts were removed. It compared TOKEN-4, explicit EDL, output distillation, memory-only routing, and twelve forecast/context candidates `P01–P12`.

### Development results

At 49M tokens, seed 100:

| condition | CE | domain FSI |
|---|---:|---:|
| TOKEN-4 | 3.7896 | 0.00396 |
| explicit EDL | 3.7715 | 0.00925 |
| OUTPUT-DISTILL | **3.7345** | 0.00377 |
| P01–P12 | 3.860–4.235 | 0.0005–0.0011 |

At 122.5M tokens, seeds 100/101:

| condition | CE | domain FSI |
|---|---|---|
| TOKEN-4 | 3.302 / 3.294 | 0.0058 / 0.0063 |
| EDL | 3.278 / 3.290 | 0.0139 / 0.0138 |
| OUTPUT-DISTILL | **3.271 / 3.275** | 0.0066 / 0.0069 |
| MEMORY-ONLY | 3.292 / 3.291 | 0.0061 / 0.0061 |
| P04/P07/P09/P12 | 3.366–3.379 | 0.0009–0.0014 |

While teacher-routed, some candidates temporarily beat TOKEN-4. After the teacher was stripped, they remained about 0.07 nats behind, and the utility head had near-zero Spearman correlation with exact expert utility. All proposed candidates were less functionally specialized than TOKEN-4.

### Status and conclusion

The operator stopped V1 after one baseline final training run and before any final validation evaluation. Validation was never opened. This is a negative development result, not a final comparison. V1 was superseded by the context-prior work.

**Source:** `LATENT_FORECAST_ROUTER_DISTILL_V1_REPORT.md`.

## 13. Context-prior specialization and fast engineering

### Method

The successor retained a causal context prior after removing the draft teacher. The engineering loop tested:

- a minimum prior gate floor of 0.2;
- longer context-memory half-lives;
- a utility-conditioned cross-chunk consistency term;
- a stronger gate floor of 0.4.

The aim was to fix an earlier failure where the learned prior became inert after forecast assistance was stripped.

### Results

The final candidate `F04` solved prior inertness. At 122.5M tokens, turning the prior off changed about 70% of top-4 slots and worsened CE by +0.020 in both seeds. F04 remained accuracy-comparable to TOKEN-4: **−0.0030 and −0.0039 nats** in seeds 100 and 101.

However, functional specialization did not improve. F04’s domain FSI, cluster FSI, and domain MI all lay inside the range of prior-inert controls. Seed 100 passed the freeze rule, but seed 101 failed because domain FSI was 0.00613 versus TOKEN-4’s 0.00626. F04 trained at about 102k tokens/s versus 137k for TOKEN-4.

### Conclusion

The model could be forced to **depend** on context-conditioned routing without losing accuracy. That dependence was not evidence of topical expert function: experts co-adapted to an arbitrary usable assignment. The engineering phase ended without a frozen final candidate.

**Source:** `CONTEXT_PRIOR_FAST_ENGINEERING_REPORT.md`. The predecessor V2 remained paused and was never converted into a final validation claim.

## 14. Forecast-Latent EDL (`FL-EDL`)

### Method

Four pretrained draft models generated 32-token sampled continuations from the causal prefix at 256-token region boundaries. Their continuations were embedded and aggregated with either mean statistics or a DeepSets model. Balanced Sinkhorn spherical clustering produced latent regions, usually `K=16`.

FL-EDL then applied the EDL JSD loss to mean router distributions of the discovered forecast clusters. Comparators were:

- TOKEN-4;
- past-only clusters from prefix embeddings;
- explicit-label EDL;
- shuffled forecast clusters;
- forecast-feature distillation without divergence.

All final models used unrestricted top-4 routing and had 103.6M parameters. Drafts were training-only.

### Final 245M-token results

| condition | mean VAL CE | FSI ratio to TOKEN-4, domain/e5 |
|---|---:|---:|
| TOKEN-4 | 3.0554 | 1.00 / 1.00 |
| FL-EDL | 3.0552 | **0.976 / 0.954** |
| PAST latent EDL | **3.0503** | 1.006 / 1.042 |
| explicit EDL | **3.0434** | **2.12 / 1.84** |
| SHUF | 3.0583 | 1.03 / 1.01 |
| forecast distillation | 3.0559 | 1.00 / 1.01 |

FL-EDL matched TOKEN-4 in CE (`−0.0003` mean) but did not increase specialization. PAST beat FL in all three seeds by a mean of **0.0049 nats**. Explicit EDL beat FL by 0.0116.

Before training, past-only clusters were more predictive of the actual region and more domain-aligned than forecast clusters: domain NMI approximately 0.29 versus 0.23. The CE-gated search selected an extremely weak FL auxiliary: its gradient was only 0.01–0.07% of the LM gradient, roughly 100× below explicit EDL. Stronger settings hurt CE early or starved experts.

### Conclusion

Forecast-derived latent groups preserved accuracy but added no useful information beyond the causal past. Draft-free inference speed was unchanged; final FL training cost about 4.7% extra, plus roughly 1.35 GPU-hours for the one-off forecast pipeline.

**Source:** `FORECAST_LATENT_EDL_V1_REPORT.md`.

## 15. Forecast counterfactual utility diagnostic

### Method

Rather than infer utility indirectly, this experiment measured exact equal-compute expert utility on 5,814 held-out regions. For each selected expert, it removed that expert, admitted the next-ranked expert, kept four active experts, recomputed downstream layers, and measured:

\[
u = CE_{\text{replacement}} - CE_{\text{normal}}.
\]

Probes predicted these utilities from:

- causal past `H`;
- extra past-only capacity;
- aggregated or concatenated draft forecasts;
- embeddings of the **actual next 32 or 128 tokens**, used only as a non-deployable oracle.

### Results

| feature set | held-out utility R² | realized routing cost ×1e-3 |
|---|---:|---:|
| causal past H | 0.0295 | 0.250 |
| H + draft mean | 0.0323 | 0.252 |
| H + all draft embeddings | 0.0332 | 0.257 |
| H + actual next 32 | 0.0325 | 0.242 |
| H + actual next 128 | **0.0336** | **0.228** |

The real next 128 tokens added only **+0.0041 R²** and closed **0.4%** of the gap to the best possible replacement decision; the preregistered bars were +0.02 R² or 10% closure. The result replicated on a second TOKEN-4 seed.

The actual future strongly predicted semantic topic—e5 cluster accuracy rose from 0.603 to 0.666—yet barely predicted which expert would matter. Region utility was only about 0.17 reliable, 63% of absolute effect came from 5% of tokens, and 98% came from tokens on which the expert was selected. This made utility fundamentally token-local rather than region-level.

The draft ensemble was reasonably diverse and calibrated. Its next-32 ensemble CE was 3.096, and its mean continuation embedding had cosine 0.465 with the real future versus 0.441 for the past. Drafts were weaker than the main model, but even the real future failed the utility test.

### Conclusion

The broken causal link was **future semantic information → region-level expert utility**. No main model or usefulness loss was trained because the oracle headroom gate failed. Stronger drafts alone were therefore unlikely to rescue this region-level routing premise.

**Source:** `FORECAST_COUNTERFACTUAL_UTILITY_V1_REPORT.md`.

---

# Part III — Label-free semantic EDL and attempts to induce function

## 16. Gradient-Consistent latent EDL (`GC-EDL`)

### Method

The idea was to discover groups that were functionally coherent, not merely semantically similar. A frozen scout model computed the gradient of its final MLP output projection on up to four separated 256-token views per paper. A CountSketch compressed each per-view gradient to 128 dimensions, later 256 in the one allowed revision. Balanced prototype learning compared:

- semantic embeddings only;
- gradient fingerprints only;
- combined semantic and gradient features;
- `K ∈ {4,8,16}`.

The discovery gate required gradient-derived clusters to beat shuffled papers and add held-out gradient coherence beyond semantic clusters before any MoE training.

### Results

- Same-paper cosine gap: semantic **0.240**; gradient 128-d **0.016**; gradient 256-d **0.017**.
- Gradient cluster view agreement kappa: **0.02**; seed agreement at most 0.31.
- Post-hoc domain NMI: gradient clusters **0.007**; semantic/combined **0.44–0.48**.
- Semantic clusters predicted held-out gradient directions better than gradient clusters.
- Raising projection dimension did not help.

### Conclusion

The experiment stopped at discovery. No MoE was trained, so CE and specialization were not tested. Single-view gradient fingerprints were dominated by view-to-view noise rather than stable paper function.

**Source:** `GC_EDL_V1_REPORT.md`.

## 17. Semantic EDL V1

### Method

Stable label-free semantic paper clusters replaced explicit domain labels in the EDL loss. Soft per-paper cluster distributions were frozen before training. Candidate dimensions included:

- all-layer versus upper-layer placement;
- 0.5–2% EDL-relative auxiliary strength;
- soft versus sharpened cluster labels;
- immediate versus ramped strength.

The persistent shuffled-paper table was the matched control. `C4`—upper layers, soft labels, 2% strength—was selected using pseudo-cluster FSI at 20M. `C6`—all layers, sharpened labels, 2% strength—was later found descriptively stronger on real-domain metrics.

### 50M results

C4 passed its initial development gate but failed the one-time CALIB-SELECTION confirmation because seed 100 pseudo-domain FSI was **0.00473**, below its shuffled control’s **0.00526**. No V1 245M final was run.

Post-terminal real-domain analysis showed:

- explicit EDL domain FSI: 0.00909 / 0.00951;
- Semantic C4: 0.00861 / 0.00844, about 89–95% of explicit EDL;
- shuffled C4: 0.00735 / 0.00764, about 80% of explicit EDL;
- TOKEN-4: 0.00405 / 0.00432.

Semantic grouping reliably raised routing-domain MI over shuffled grouping, but much of EDL’s effect came from having **any persistent paper-level grouping**. The soft labels retained only about 25% of the real-domain information; hard assignments retained about 58%.

### Conclusion

Semantic EDL was accuracy-competitive and clearly more domain-aligned than TOKEN-4. It was not confirmed under its original label-free selection rule, and most of its apparent specialization was shared with the shuffled persistent grouping.

**Source:** `SEMANTIC_EDL_V1_REPORT.md`.

## 18. Long Semantic EDL and fixed cluster-count study

### Method

Phase 1 trained TOKEN-4, explicit EDL, Semantic C6, and shuffled C6 for **245M tokens over four seeds**. Phase 2A tested fixed `K = 4, 8, 16, 32` at 50M tokens over two seeds. Adaptive K was planned for Phase 2B but required a gate that never opened.

### Phase 1 final results

| condition | CE | domain NMI | domain FSI | per-expert interaction |
|---|---:|---:|---:|---:|
| TOKEN-4 | 3.0533 | 0.0126 | 0.00762 | 0.000382 |
| explicit EDL | **3.0435** | **0.0574** | **0.01687** | 0.000657 |
| Semantic C6 | 3.0467 | 0.0547 | 0.01599 | 0.000637 |
| shuffled C6 | 3.0461 | 0.0452 | 0.01490 | 0.000634 |

Semantic C6 beat TOKEN-4 in mean CE by 0.0066 and had higher subject-aligned routing in every seed, but it did **not** beat shuffled C6 on the primary per-expert interaction: mean difference **+2.3e-6**, positive in only 2/4 seeds. Routing aligned; functional differentiation over persistent grouping did not emerge.

### Fixed-K screen

| K | mean semantic−shuffled e5 FSI | pseudo FSI | CE | repeated screen result |
|---:|---:|---:|---:|---|
| 4 | −0.00021 | +0.00112 | +0.0036 | fail |
| 8 | **+0.00108** | +0.00105 | **−0.0187** | passed screen |
| 16 | −0.00040 | +0.00044 | −0.0048 | fail |
| 32 | +0.00042 | +0.00027 | −0.0006 | fail |

K=8 then failed held-out confirmation because seed 0 e5 FSI was −2.26e-5 versus shuffled, though pseudo-FSI and CE favored semantic grouping in both seeds.

### Conclusion

No fixed K showed a confirmed semantic functional advantage, so adaptive K was neither supported nor run. K=8 remained suggestive but unconfirmed.

**Source:** `SEMANTIC_EDL_LONG_K_PHASE2A_REPORT.md`.

## 19. Hard counterfactual-usefulness EDL

### Method

This experiment kept Semantic EDL and added a direct usefulness objective:

\[
L = L_{LM}+0.01L_{Switch}+0.001L_z+0.005L_{Semantic\ EDL}+\gamma L_{use}.
\]

On held-out training-plan documents, it repeatedly removed selected experts, admitted next-ranked experts, and estimated `ΔU`: own-group usefulness minus difficulty-matched other-group usefulness. The estimates were lagged, detached, shrunk, and standardized. A hard-margin loss asked each group’s mean routing distribution to score highest on its own `ΔU` column.

### Results

At 20M, only a weak setting `U2` was eligible; its interaction gain over Semantic C6 was just **+3.7e-5**. At 50M it failed, triggering one allowed engineering generation with more probe passes and slower EMA. That generation showed a small development advantage but reversed on held-out CALIB-SELECTION:

| seed | semantic interaction | shuffled interaction | difference |
|---|---:|---:|---:|
| 100 | 4.29e-4 | 4.67e-4 | **−3.8e-5** |
| 101 | 4.38e-4 | 4.87e-4 | **−4.9e-5** |

The pooled held-out CI was `[−8.7e-5,+0.9e-5]`. No 245M final was run. Eight-pass probing added roughly 81% to step time and reduced throughput from about 123k to 63k tokens/s.

### Conclusion

The semantic usefulness signal was measurable but too noisy. Development selection overfit that noise. Persistent grouping produced CE benefits, but semantic usefulness did not replicate over shuffled grouping.

**Source:** `SEMANTIC_UTILITY_EDL_V1_REPORT.md`.

## 20. Soft confidence-aware counterfactual usefulness

### Method

V2 replaced the hard margin with uncertainty-aware soft targets. For each `(group, layer, expert)` cell, it formed a lower-confidence usefulness score from `ΔU ± SE`, tilted the group’s lagged routing distribution by that score, and trained with:

\[
L = L_{LM}+0.01L_{Switch}+0.001L_z+\gamma KL(\operatorname{stopgrad}q\;||\;R_g).
\]

Semantic EDL was deliberately removed to test whether usefulness alone could create structure.

### Results

At 20M the best candidate `C4` had CE 4.6234 versus TOKEN-4 4.6263 and interaction +7.2e-5 over TOKEN-4. At 50M, the semantic-minus-shuffled interaction had opposite signs in the two seeds (`−4.2e-5` and `+4.6e-5` on development), and the held-out pooled interval was `[−5.0e-5,+6.4e-5]`. Bank reliability was only 0.19–0.23, below the 0.3 criterion.

The mechanism did weakly operate: semantic usefulness scores shifted routing by about 1%, exposure followed routing, and increased exposure weakly correlated with later usefulness. But approximately two-thirds of initial score variation regressed away. Routing domain NMI was only 0.012, near TOKEN-4’s 0.011 and far below Semantic EDL’s 0.045–0.050.

### Conclusion

No replicated functional advantage. Replacing Semantic EDL with noisy usefulness targets lost both routing structure and the EDL-family CE benefit. Probe overhead was 84–87% of base step time.

**Source:** `SOFT_CF_USEFULNESS_V2_REPORT.md`.

## 21. Semantic Lift gradient reweighting

### Method

Semantic Lift asked whether experts receiving more of their subject’s training gradient would become functionally specialized. It used the C6 semantic groups and a slow lift estimator to create clipped gradient multipliers in `[0.5,2.0]`, with gate-weighted mean exactly 1. The schedule was 5% collection, 5% ramp, and 90% full-strength reweighting. The shuffled-group twin used the identical mechanism.

### 50M precursor

At 50M, lambda 0.5 increased ordinary interaction and routing-specialized cells. It also produced strong real-domain routing NMI in post-terminal analysis. However, the semantic-minus-shuffled functional result was inconsistent across seeds, and CE varied substantially. This justified a four-seed 245M confirmation rather than a claim.

### 245M confirmation

Semantic Lift reliably increased routing concentration:

- domain NMI **0.0704** versus 0.0443 for shuffled;
- 235 versus 177 routing-specialized cells;
- ordinary interaction `E − F = +1.53e-4`, positive in 4/4 seeds.

But exposure-matched interaction was **−4.5e-5**, positive in only 1/4 seeds, with CI `[−2.5e-4,+1.75e-4]`. CE was worse than shuffled by +0.0037, worse than C6 by +0.0034, and worse than EDL by +0.0066.

### Conclusion

Semantic Lift made experts **receive** more subject tokens; it did not make them more useful per invocation. This experiment exposed a major confound in the earlier ordinary interaction metric and changed how all prior EDL-family specialization claims should be interpreted.

**Sources:** `SEMANTIC_LIFT_EXPERT_LEARNING_V1_REPORT.md`, `SEMANTIC_LIFT_245M_CONFIRM_V1_REPORT.md`.

## 22. Subject-conditional residual gradient projection (`SCRGP`)

### Method

SCRGP directly modified an expert’s parameters in a direction intended to improve one semantic subject relative to matched controls. For each semantic cluster `k` and expert `e`, it computed:

- positive versus matched-control CE gradients under forced equal exposure;
- their difference `d`;
- `u`, the component of `d` orthogonal to the balanced global forced-exposure gradient.

It then applied a one-step virtual update `theta'_e = theta_e − eta u(k,e)` to frozen Semantic C6 checkpoints and evaluated the result on paper-disjoint held-out data. Routing was held fixed; this was a functional test.

### Results

The assigned-versus-control contrast moved in the predicted direction:

- semantic forced contrast gain `F = +0.00229`, 4/4 seeds;
- shuffled `F ≈ 0`;
- exposure-matched natural interaction `N = +0.00468`, 4/4 seeds.

But the update was zero-sum or destructive:

- balanced global forced CE worsened by **+0.00215**, almost equal to the contrast gain;
- applying all 352 expert updates raised natural CE by **+0.58 to +1.62 nats**;
- 81% of the signal came from the final MoE layer and 58% from one cluster;
- in the dominant final layer, the assigned subject itself worsened, just less than other subjects.

### Conclusion

Semantic differential gradients were real and reproducible, but they shifted damage between subjects rather than creating competence. The experiment stopped before training because matching, global-CE, and concentration criteria failed.

**Source:** `SCRGP_V1_REPORT.md`.

---

# Part IV — Added residual specialists and causal deployment

## 23. Oracle residual specialists — the strongest functional positive

### Method

A frozen 245M-token TOKEN-4 base received a residual bank after its final MoE block:

\[
h' = h + a_e(\operatorname{RMSNorm}(h)).
\]

Each bank contained 16 specialists of shape `384→256→384`, adding 3,145,728 parameters; exactly one was active per token. The base remained frozen. Conditions were:

- **A semantic oracle:** whole-paper label-free semantic cluster chooses the specialist;
- **B shuffled oracle:** mass-matched random paper cluster;
- **C token-routed:** learned causal top-1 linear router;
- **D frozen base:** no residual bank.

Residual banks trained for 50M new tokens over four seeds.

### Validation results

| condition | improvement over frozen base |
|---|---:|
| semantic oracle A | **+0.00337** `[+0.00314,+0.00357]` |
| shuffled oracle B | +0.00097 |
| token-routed C | +0.00210 |

A beat B by **0.00240** and C by **0.00127**, in 4/4 seeds. All 16 clusters improved. Exposure-matched functional specificity was:

- semantic specialists: **0.00484**;
- shuffled specialists: **0.00002**;
- semantic-minus-shuffled: **0.00482**, with a hierarchical CI excluding zero.

Replacing a semantic specialist with another specialist always hurt its own cluster. This was genuine functional specialization, not merely routing exposure.

### Limitation

The assignment used the whole paper and was therefore non-causal. About 60% of A’s advantage over the token router came from positions below 128, only 13% of tokens. The experiment proved capacity and semantic separability, not deployability.

### Conclusion

Label-free semantic clusters correspond to distinct predictive residual functions when separate parameters are added. This is the strongest positive causal premise in the project.

**Source:** `ORACLE_RESIDUAL_V1_REPORT.md`.

## 24. Joint semantic residual pretraining

### Method

This attempted to learn the base, residual bank, and router during pretraining rather than post-training. A 103.6M TOKEN-4 base was isolated from bank gradients. Sixteen residual experts and a causal per-token top-1 router were trained with:

\[
L=L_{base}+L_{res}+\beta_tL_{semantic-divergence}+0.01L_{balance}.
\]

The semantic-divergence term separated group-average router distributions. Conditions were no grouping, shuffled grouping, semantic C6 grouping, and a non-causal semantic oracle.

### Results at 50M, two seeds

Every residual bank improved its internal base by about 0.011 nats. Semantic D was approximately tied with no-group B in CE, while beating shuffled C by about 0.0004. Exposure-matched own-specialist usefulness was also about +0.0004 over C.

However, shuffled C routed by real subject just as strongly as semantic D: real-domain NMI was about **0.225–0.229** for both. The within-step divergence loss could be satisfied by separating arbitrary groups among the current few documents. Because documents differ mainly by subject, even random grouping pressure induced subject-correlated routing at corpus scale.

The oracle had roughly **10×** stronger functional specificity (`~0.012`) but no CE gain over the no-group routed bank at 50M.

### Conclusion

The semantic increment over shuffled grouping was small and sign-consistent but did not meet the gate. Per-step group separation was the wrong objective because it did not enforce persistent corpus-wide semantic identities. No 245M stage was run.

**Source:** `JOINT_SEMANTIC_RESIDUAL_PRETRAIN_V1_REPORT.md`.

## 25. Cross-fitted semantic-utility residual teacher/student

### Method

Sixteen residual specialists sat after a detached TOKEN-4 base, one active for each 128-token view. A training-only teacher combined:

- cross-fitted next-token-loss reduction of EMA specialists on a partner view;
- a frozen label-free semantic region prior;
- a mild global balance term.

A deployed GRU student saw only **completed past views**, was distilled from the teacher, and selected one specialist. The generic token-routed residual G and a shuffled semantic pipeline were controls.

### Results at 20M, one seed

The semantic teacher created real function:

- it beat generic G by as much as **0.0054 nats**;
- shuffled teachers were 0.006–0.009 worse than G;
- exposure-matched semantic-minus-shuffled interaction was positive.

The causal student retained only 13–69% of the teacher’s semantic-over-shuffled advantage, about one-third pooled. Every student candidate was **+0.0046 to +0.0071 worse than G**. Adding the last hidden state to the router improved CE by only about 0.001.

### Conclusion

The training teacher learned useful semantic specialists, but the student could not infer the current 128-token view’s subject from completed past views. The experiment failed at Stage 1; replication and longer training were not run.

**Source:** `docs/CSR_CROSSFIT_SEMANTIC_UTILITY_RESIDUAL_V1_REPORT.md`.

## 26. Centered federated averaging of residual specialists

### Method

The same 16 residual specialists were treated as semantic “clients.” Three modes were tested:

- local specialist training;
- standard FedAvg, which broadcasts the token-weighted mean update;
- centered personalized FedAvg, decomposing each specialist into shared `S` and private `P_k`, transferring the mean update into `S` while preserving centered private updates.

Sync intervals were 100, 250, or 500 steps; private widths were 240, 192, or 128. A causal student routed using past views.

### Results at 20M, one seed

All nine centered-FedAvg candidates failed the deployed CE gate:

- student CE was **+0.0120 to +0.0129 worse than G**;
- the semantic oracle teacher beat G by about **0.007**;
- exposure-matched semantic-minus-shuffled interaction was **+0.0142 to +0.0166** in 9/9 candidates.

The student retained about 20% of the oracle gain. Centered FedAvg was CE-neutral relative to local training because the shared update was only 9–15% of local update norm. Standard FedAvg erased specialist differences; more frequent synchronization was worse.

### Conclusion

Federation did not solve the inference bottleneck. It changed how experts shared knowledge, not whether a causal router knew the current region’s subject.

**Source:** `docs/CSR_CENTERED_FEDAVG_RESIDUAL_V1_REPORT.md`.

## 27. Current-context causal specialist router — deployable positive, continuation negative

### Method

This reused the frozen semantic, shuffled, and generic banks from the previous experiments. Instead of routing once per completed view, it routed **every token** using:

\[
[\operatorname{RMSNorm}(h_t),\;\operatorname{mean}(h_{t-15:t})]
\rightarrow \text{Linear}\rightarrow \text{GRU}(64)\rightarrow 16\text{ logits}.
\]

The router had 75,152 parameters, used only causal information, and selected one residual specialist per token. It was trained against a teacher combining semantic region prior and exact specialist utility. Ninety-six routers were searched; the final `R3_c2e0.5` used utility temperature `2*s_u`, semantic weight 0.5, and distillation weight 0.01.

### Frozen-specialist results

On held-out CALIB-SELECTION papers:

- candidate CE: **4.59094 / 4.59119 / 4.59097**;
- generic G: **4.59503**;
- improvement over G: **−0.00409 / −0.00384 / −0.00406**;
- retained semantic-over-shuffled teacher advantage: **0.744–0.748**;
- exposure-matched semantic interaction: **0.0092–0.0094** versus 0.0001–0.0005 for shuffled;
- all 16 specialists were individually useful;
- semantic routing NMI 0.395 versus 0.048 for shuffled.

The main failures were the first 32 positions and the first 64 tokens after a region change. Stable tokens beat G by about 0.0074. Refreshing the specialist only every 16 tokens retained a −0.00275 gain over G, but did not improve measured prefill speed. Router+specialist prefill cost was 3.7–3.9% above the base.

### Continued joint training

After a matched 50M continuation:

- frozen routers fell behind G by about **+0.0013** as the base representation drifted;
- jointly updated routers remained marginally ahead, about **−0.0004**, but became nearly document-constant and retained only 23–38% of exposure-matched specialization.

### Conclusion

This is a real positive result for deploying **frozen** semantic specialists causally. It did not establish a stable end-to-end pretraining method. The three seeds varied only the router; specialist training still had one seed.

**Source:** `docs/CSR_CURRENT_CONTEXT_SPECIALIST_ROUTER_V1_REPORT.md`.

## 28. Alternating semantic residual specialization

### Method

To preserve function during continuation, training alternated ten cycles of:

1. 32 base-only steps;
2. exact teacher refresh;
3. 32 specialist steps;
4. 15–16 router steps.

The teacher used exact specialist utility plus semantic region prior. The starting point was the successful frozen current-context router/bank.

### Results

The method failed badly:

- CE ended about **+0.56 nats worse than ordinary G** and +0.23 worse than a schedule-matched generic bank;
- only 6–10 of 16 specialists remained positive;
- no semantic group beat the schedule-matched generic bank;
- semantic and shuffled banks degraded identically.

Two causes were identified:

1. **Base starvation:** the base received only 320 of 793 optimizer steps, explaining about +0.36 nats of the deficit.
2. **Target-dependent teacher corruption:** exact utility used the next token being predicted to select which specialist trained on that token. Specialists became target-token experts: excellent on assigned tokens but harmful elsewhere, and no causal router could predict that assignment.

### Conclusion

Alternating optimization was not a remedy. It destroyed both CE and deployable specialization. This motivated continuous base updates and target-free assignments.

**Source:** `docs/CSR_ALTERNATING_SEMANTIC_RESIDUAL_SPECIALIZATION_V1_REPORT.md`.

## 29. Continuous-base, target-free specialization with five clusters

### Method

The next experiment fixed both diagnosed failures:

- the base updated on every batch;
- no target token ever selected a specialist;
- exactly five label-free semantic clusters and five residual specialists;
- hard or sampled semantic prior assignments;
- auxiliary windows trained specialists and routers while the ordinary base trajectory continued.

The five clusters were produced by merging the 16 semantic prototypes with weighted spherical k-means.

### Results

At the frozen 20M stage:

- exposure-matched semantic-minus-shuffled interaction: **+0.0167** `[+0.0146,+0.0188]`;
- 5/5 specialists positive;
- semantic causal router **+0.0106 worse than G5**;
- even the semantic region teacher was +0.0073 worse than G5.

After 50M continuation:

- hard semantic router: +0.0019 versus G and +0.0005 versus ALT-G;
- soft router: +0.0016 versus G and +0.0002 versus ALT-G;
- semantic teacher: −0.0016 versus ALT-G and approximately tied with G;
- hard specialization interaction remained +0.0045 with 5/5 specialists positive;
- semantic beat shuffled at every milestone.

The deployed router was almost document-constant. It lost mainly on positions 0–31 and after topic transitions. Stable tokens were slightly better than ALT-G.

### Conclusion

Continuous base training and target-free assignments successfully prevented corruption. Genuine functional specialists persisted. The remaining failure was deployment and capacity: the causal router lacked current-region information, and a five-specialist semantic bank was not more accurate than a flexible generic residual bank on stable text.

**Source:** `docs/CSR_CONTINUOUS_BASE_TARGET_FREE_ASRS_V2_REPORT.md`.

## 30. Subject-conditioned utility router

### Method

With the five frozen specialists, a causal router was explicitly taught:

- a subject posterior from causal hidden states;
- a cross-fitted subject-to-expert utility matrix;
- a bounded per-token correction around the subject prior;
- routed LM, subject, utility, regret, and balance losses.

The specialist banks were frozen. Four engineering generations sharpened the subject-to-expert mapping, removed redundant losses, increased correction capacity, and softened the prior. Controls included subject-only, free utility-only routing, shuffled twins, the previous router, G, ALT-G, a region teacher, and a per-token oracle.

### Results

The utility mapping was reliable: split-half correlation **0.975**, and every semantic subject’s own specialist was its best expert. The router predicted subject with 0.77 overall accuracy and 0.90 on stable regions, but only 0.51 in the first 32 positions and 0.35 just after a topic transition.

No subject-anchored generation beat ALT-G or G. The best candidate remained roughly +0.0001 versus ALT-G and +0.0015 versus G. The free utility-only router did slightly beat ALT-G by **−0.00028** and had the strongest exposure-matched function, but remained +0.00111 worse than G.

The key decomposition showed:

- subject recognition gap: +0.0017;
- subject-to-expert mapping error: essentially zero;
- subject-granularity bank limit versus per-token oracle: +0.1011;
- on 87% of stable tokens, even perfect semantic routing was less helpful than generic G.

### Conclusion

The mapping was not the problem. Subject uncertainty at starts/transitions mattered, but the larger limit was that one subject specialist per token was too coarse and usually less accurate than a flexible generic token-routed bank. Anchoring the router to a confident subject prior also starved alternatives of learning signal.

**Source:** `docs/SUBJECT_CONDITIONED_UTILITY_ROUTER_V1_REPORT.md`.

---

## 31. Consolidated result matrix

| experiment | central intervention | best supported result | final status |
|---|---|---|---|
| Representative chunk routing | fixed/semantic shared expert pools | pooling +0.110 CE; semantic boundaries no better than random | negative final |
| Sequence routing search | one route per fixed/dynamic segment | +0.116 to +0.144 CE; slower | negative final |
| Speculative pool routing | draft/adaptive candidate pools | learned pools beat random but not TOKEN-4 | negative final |
| Conditional local attention | local W128 / learned global queries | +0.0245 CE; <1% train gain, slower prefill | negative final |
| Value-routed compute | local/global attention and top-k compute at 8K | all cheap paths 6–10% slower | diagnostic fail |
| Multiscale routing | local/broad causal router priors | no robust 5M CE gain | stopped at screen |
| M2D2 domain prior | explicit domain bias + unrestricted top-4 | domain info useful; routing prior not reliably useful | negative pilot |
| Forecast curriculum | 40/10/50 assisted→anneal→token | durable specialists; +0.013 CE worse; compute controls much better | negative final |
| Latent forecast distillation | forecast teacher → student router | teacher gains vanished after stripping | externally stopped; negative development |
| Context-prior engineering | force persistent causal prior use | strong prior dependence, no added function | no freeze |
| FL-EDL | forecast-cluster EDL | CE matched; no specialization; past-only better | negative final |
| Forecast utility diagnostic | exact utility prediction from forecasts/oracle future | actual future closed only 0.4% routing gap | oracle gate fail |
| GC-EDL | gradient fingerprints + semantic prototypes | gradients unstable/noisy | discovery invalid |
| Semantic EDL | semantic clusters in EDL | near EDL routing/CE; small increment over shuffled | V1 unconfirmed |
| Long Semantic EDL | 245M/4 seeds and K study | C6 aligns routing; no function over shuffled; K8 unconfirmed | Phase 1 negative, 2A negative |
| Hard utility EDL | lagged exact usefulness + margin | development effect reversed held-out | negative |
| Soft utility | uncertainty-aware KL target | weak chain, estimator reliability ~0.2 | negative |
| Semantic Lift | semantic gradient reweighting | routing exposure increased; per-invocation function did not | negative final |
| SCRGP | subject gradient orthogonal to global gradient | real differential signal, globally destructive | virtual gate fail |
| Oracle residual | added semantic residual capacity | +0.00337 CE; strong true function | **positive oracle** |
| Joint residual pretraining | residual bank + semantic divergence | shuffled induced same subject routing | negative at 50M |
| Cross-fit utility residual | semantic+utility teacher, past-view student | teacher functional; student worse than G | negative Stage 1 |
| Centered FedAvg | shared/private specialist updates | function preserved; deployment worse; transfer negligible | negative Stage 1 |
| Current-context router | token+previous-16 GRU over frozen bank | −0.004 vs G, 75% teacher retention | **positive frozen**, negative continuation |
| Alternating optimization | base/teacher/specialist/router cycles | base starvation and target leakage | strongly negative |
| Continuous target-free | ordinary base + target-free K5 specialists | 5/5 useful; router still worse than G | negative Stage 1 |
| Subject-conditioned router | explicit causal subject posterior + utility matrix | mapping correct; causal/bank limits remain | negative Stage A |

---

## 32. What the full body of evidence now supports

### 32.1 Strongly supported

1. **Hard chunk-level expert restriction is a poor accuracy trade-off for this model scale and implementation.** It removes token-level flexibility and does not yield wall-time gains.
2. **Stable group pressure readily changes routing distributions.** Explicit domains, semantic clusters, and even persistent shuffled groups can all cause experts to be selected differently across groups.
3. **Routing specialization is not the same as functional specialization.** An expert may look important simply because it receives more high-gate subject tokens.
4. **Explicit EDL improves CE and subject-correlated routing.** It is the strongest ordinary MoE baseline in these runs.
5. **Label-free semantic clusters contain real subject information.** Semantic routing consistently exceeds shuffled routing in domain NMI, and semantic residual utility matrices are diagonal while shuffled ones are flat.
6. **Separate semantic residual capacity can learn genuinely different functions.** ORACLE-RESIDUAL and the residual teacher experiments established this under equal-exposure or exposure-matched tests.
7. **A causal current-context router can deploy frozen semantic specialists better than a generic residual bank.** This worked across three router seeds.
8. **Future-token semantics are not a strong region-level expert-utility signal.** Even actual future text barely predicted which experts mattered.

### 32.2 Supported failure mechanisms

1. **Chunk restriction:** useful token-specific experts lie outside the chunk pool.
2. **Persistent-group confound:** any stable grouping can induce document discrimination and subject-correlated routing because subject is the dominant between-document axis.
3. **Exposure confound:** ordinary ablation rewards experts that simply see more subject tokens at larger gates.
4. **Noisy counterfactual banks:** online expert-utility estimates had reliability around 0.2–0.4 and caused development overfitting.
5. **Target leakage:** assigning specialist training examples using the same target token creates non-causal target-token experts.
6. **Base starvation:** alternating schedules that reduce base optimizer steps produce unavoidable CE deficits.
7. **Representation drift:** a frozen router ceases to match a continuously changing base.
8. **Over-anchoring:** strong semantic priors prevent a router from exploring alternative experts.
9. **Coarse bank granularity:** a single subject expert is often less accurate than flexible token-level generic routing on stable text.
10. **Starts and topic transitions:** causal subject inference is weakest exactly where oracle semantic routing has its largest advantage.

### 32.3 Still unproven

1. Whether the frozen current-context router result replicates across **independent specialist-training seeds**, not just router seeds.
2. Whether a jointly designed base, semantic residual bank, and free token-level router can preserve the frozen-bank gain through 245M tokens.
3. Whether finer region-level semantic clusters or multiple specialists per semantic region can beat generic token routing on stable text without reintroducing hard chunk constraints.
4. Whether a draft model adds useful causal subject information specifically at document starts and topic transitions. The prior forecast-utility study was about region-level expert utility, not this narrower subject-inference target.
5. Whether any method produces real inference-speed gains with an optimized KV-cached implementation. Most existing speed studies were dominated by eager kernels and orchestration overhead.

---

## 33. Ideas discussed but not established as experiments

The following should not be presented as completed findings:

- adaptive/hierarchical K: planned but gate-closed and never implemented;
- a long 245M K=8 semantic EDL confirmation: not run;
- stronger modern pretrained draft ensembles for start/transition subject inference: not run;
- SP-ERC/EMA teacher-student expert-response coupling: discussed, but no corresponding report was found;
- hybrid local attention with full attention only in the last layers at 100M: the final search selected pure local, so this long run was not performed;
- a stable end-to-end version of the successful current-context frozen-specialist router: not yet demonstrated.

---

## 34. Bottom line

The project did produce positive evidence, but not for the original hard chunk-routing thesis.

- **Negative:** restricting MoE routing by chunks, pools, draft forecasts, or fixed subject assignments generally hurt accuracy and did not accelerate the current implementation.
- **Positive:** semantic information is real, explicit EDL improves CE, and label-free semantic clusters can support genuinely different residual functions when separate specialist capacity is trained.
- **Most promising demonstrated mechanism:** train semantic residual specialists with target-free assignments, freeze them, and deploy them using a free causal token-level router that sees current-token and recent-context states.
- **Unsolved problem:** preserve that functional specialization and CE advantage during long joint training without router drift, prior over-anchoring, target leakage, or loss of flexibility.

That is a narrower and better-supported research problem than “make chunk routing beat token routing.”

---

## 35. Primary report index

All paths below are relative to `/home/zapan4/moe-research` on `lambda2`.

- `SPECIALIZATION_AUDIT_REPORT.md`
- `SEQUENCE_ROUTING_SEARCH_V1_REPORT.md`
- `SPECULATIVE_POOL_ROUTING_V1_REPORT.md`
- `CONDITIONAL_LOCAL_ATTENTION_V1_REPORT.md`
- `VALUE_ROUTED_COMPUTE_V1_REPORT.md`
- `MULTISCALE_ROUTING_V1_REPORT.md`
- `M2D2_DOMAIN_PRIOR_V1_REPORT.md`
- `DOMAIN_FORECAST_CURRICULUM_V1_REPORT.md`
- `LATENT_FORECAST_ROUTER_DISTILL_V1_REPORT.md`
- `CONTEXT_PRIOR_FAST_ENGINEERING_REPORT.md`
- `FORECAST_LATENT_EDL_V1_REPORT.md`
- `FORECAST_COUNTERFACTUAL_UTILITY_V1_REPORT.md`
- `GC_EDL_V1_REPORT.md`
- `SEMANTIC_EDL_V1_REPORT.md`
- `SEMANTIC_EDL_LONG_K_PHASE2A_REPORT.md`
- `SEMANTIC_UTILITY_EDL_V1_REPORT.md`
- `SOFT_CF_USEFULNESS_V2_REPORT.md`
- `SEMANTIC_LIFT_EXPERT_LEARNING_V1_REPORT.md`
- `SEMANTIC_LIFT_245M_CONFIRM_V1_REPORT.md`
- `SCRGP_V1_REPORT.md`
- `ORACLE_RESIDUAL_V1_REPORT.md`
- `JOINT_SEMANTIC_RESIDUAL_PRETRAIN_V1_REPORT.md`
- `docs/CSR_CROSSFIT_SEMANTIC_UTILITY_RESIDUAL_V1_REPORT.md`
- `docs/CSR_CENTERED_FEDAVG_RESIDUAL_V1_REPORT.md`
- `docs/CSR_CURRENT_CONTEXT_SPECIALIST_ROUTER_V1_REPORT.md`
- `docs/CSR_ALTERNATING_SEMANTIC_RESIDUAL_SPECIALIZATION_V1_REPORT.md`
- `docs/CSR_CONTINUOUS_BASE_TARGET_FREE_ASRS_V2_REPORT.md`
- `docs/SUBJECT_CONDITIONED_UTILITY_ROUTER_V1_REPORT.md`

