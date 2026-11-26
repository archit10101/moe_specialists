# Ablations

All ablations are entries in `repro/config.py → EXPERIMENTS` with `tier="ablation"`. Run them with
`./slurm/submit_all.sh --tier ablation` once the core results look sane; any core job they need is
pulled in automatically. Results land in the same `REPORT.md`.

| ID | question | experiments | what would count as an answer |
|---|---|---|---|
| **B1** | Is the EDL CE gain sensitive to its strength? | `abl.edl_beta1e-3`, `abl.edl_beta2e-2` (2 seeds) | CE vs TOKEN-4 for β ∈ {1e-3, 5e-3, 2e-2}: a U-shape means the doc's 5e-3 was tuned; a flat curve means the gain is robust |
| **B2** | Does EDL need every layer? | `abl.edl_upper` | upper-half EDL ≈ full EDL → the effect lives in late layers |
| **B3** | Do *true* domain labels matter, or does any persistent grouping work? | `abl.edl_shuflabels` (explicit EDL on paper-permuted domain labels) | if ≈ explicit EDL in CE, the doc's §18 finding (most of the gain comes from grouping, not semantics) also holds for labels |
| **B4** | **Your hypothesis: subject specialisation only pays once the base already "knows English"** | `abl.bank_{sem,shuf,tok}_at{10,25,50}` + the core banks at 100% | report table "B4": if the gap between semantic and generic banks (sem − tok) grows as the base checkpoint improves, the hypothesis is supported; if it is flat or shrinks, it is not |
| **B5** | Granularity: are 16 specialists the sweet spot? (doc §29–30: K=5 lost to G) | `abl.bank_{sem,shuf,tok}_k{4,8,32}`, total width fixed at 4,096 units | sem − tok as a function of K |
| **B6** | Is the gain routing or just extra capacity? | `abl.bank_dense_compute` (one 256-unit MLP, same compute per token), `abl.bank_dense_params` (one 4,096-unit MLP, same parameters) | the semantic bank must beat the compute-matched dense residual; beating the parameter-matched one would be strong evidence |
| **B7** | Does cluster quality matter? | `abl.bank_minilm`, `abl.bank_tfidf`, `abl.bank_domain` (5 true domains) | improvement vs the e5 bank; the domain oracle bounds what coarse labels give |
| **B8** | Does it depend on the particular base? | removed: core bank seed *s* now sits on TOKEN-4 seed *s* (as in the original), so the core banks already span 4 bases | sem > shuf and sem > tok on every base seed |
| **B9** | **The doc's top open question (§32.3 #1):** does the router result replicate across *independently trained specialist banks*? | `abl.router_{sem,shuf}_bank{1,2,3}` | router − G < 0 on every bank seed |
| **B10** | Which router design choices matter? | `abl.router_token_only`, `_ctx_only`, `_mlp`, `_ctx4`, `_ctx64`, `_sem0`, `_sem1`, `_ce_only` | router − G per variant; position breakdown shows whether context helps at starts and transitions |

### Adding your own ablation

Add one `Experiment(...)` line to `EXPERIMENTS` in `repro/config.py`:

```python
Experiment("abl.bank_sem_wide", "bank", (0, 1), "ablation", "wider specialists",
           bank=dict(routing="oracle", groups="sem", spec_hidden=512)),
```

Then run `python -m repro.run plan --only abl.bank_sem_wide` to check the job graph, and
`./slurm/submit_all.sh --only abl.bank_sem_wide` to submit it. Any field of `BankCfg`, `RouterCfg` or
`EDLCfg` can be overridden this way, with no code changes.
