# Prompts for Claude Code cloud sessions

Copy one block into a new cloud session at claude.ai/code with this repo selected. The repo's
`CLAUDE.md` is loaded automatically, so these prompts only state the task.

---

## Prompt 1: verify the code before the ORCD run

```
This repo reproduces three positive results from docs/source_history.md (explicit EDL §18, oracle
residual specialists §23, causal current-context router §27) with ablations B1–B10. It is a
reimplementation from that doc only. Read README.md, docs/RECONSTRUCTION.md and repro/config.py first.

Your job is to verify the code before I spend ~45 GPU-hours on MIT ORCD. This session has no GPU:
use only the smoke profile.

1. Check that torch is importable. If it is not, pip install the CPU build plus requirements.txt.
2. Run `python -m pytest`, then `REPRO_SLOW=1 python -m pytest tests/test_smoke_pipeline.py`,
   then `python -m repro.run local --profile smoke --tier all`. Every job must finish.
3. Review these files against docs/source_history.md, looking for bugs that would silently bias the results:
   repro/model/losses.py (EDL), repro/clusters/build.py (Sinkhorn, shuffled controls),
   repro/train/residual.py and repro/eval/bank_eval.py (oracle bank, swap matrix),
   repro/train/router.py and repro/eval/router_eval.py (causality, teacher, comparison with G),
   repro/eval/base_eval.py (exposure matching).
   Specifically check for: future-token leakage into routers; the target token leaking into any
   deployed (evaluation-time) decision; semantic and shuffled conditions seeing different rows or
   having different capacity; metrics computed on training data.
4. Fix real bugs with minimal changes and add a test for each fix. Record any change to a
   reconstruction choice in docs/RECONSTRUCTION.md. Do not refactor for style.
5. Commit to a branch and summarise: what you ran, what failed, what you fixed (with file:line), and
   anything you are unsure about. Do not open a PR unless I ask.
```

---

## Prompt 2: add an ablation

```
Read README.md and docs/ABLATIONS.md. Add this ablation to repro/config.py → EXPERIMENTS as
tier="ablation": <DESCRIBE IT, e.g. "semantic bank with 512-unit specialists, 2 seeds">.
Use config overrides only. If the code truly cannot express it, make the smallest general change to
the code and add a test. Run `python -m repro.run plan --only <name> --profile full` to show the job
graph, then run the smoke pipeline for it (`--profile smoke --only <name>`) and `python -m pytest`.
Document the ablation in docs/ABLATIONS.md and the report if needed. Commit to a branch.
```

---

## Prompt 3: analyse the ORCD results

```
The real results from MIT ORCD are committed under results_orcd/ (eval.json and train_log.json per run,
plus results/REPORT.md and PAPER.md). Do not modify them. They come from the "paper" tier and, if it
was run, "paper_extra" (docs/ABLATIONS.md). The paper is about post-hoc subject specialists (§23) and
the causal router that deploys them (§27); EDL (§18) was not run and is out of scope. Read README.md,
docs/ABLATIONS.md, docs/RECONSTRUCTION.md and docs/source_history.md §23, §27 and §32, then:

1. Regenerate the report from these results without writing into results_orcd/:
   `REPRO_WORK=results_orcd REPRO_REPORT_DIR=docs/report python -m repro.run report`
   (if it fails, explain why rather than editing the result files). It writes REPORT.md, PAPER.md,
   per_seed.csv and figures/ into docs/report.
2. Main claims (PAPER.md "Main results"): semantic bank vs the frozen base, vs shuffled and vs the
   token-routed bank G; exposure-matched specificity; router − G; the share of the oracle's
   semantic-over-shuffled advantage the router retains. For each: our paired mean, 95% CI, n and sign
   count vs the doc's number, and whether the CI supports the claim. Do not round FAILs up to PASSes.
3. Ablations: B4 (does the semantic-vs-generic and semantic-vs-shuffled gap grow with base competence?
   use the slope and its CI), B6 (routing vs capacity: compute- and parameter-matched dense residuals),
   B9 (does router − G < 0 replicate across bank seeds 0-3?). If the paper_extra runs exist, also B8
   (other base seeds), the explicit 5-domain oracle, and the router-input variants (token only vs
   context only), plus the router-by-position table.
4. Note where a reconstruction choice or a kept difference from the original (docs/RECONSTRUCTION.md)
   could explain a disagreement with the doc, in particular that our §27 routes over the §23
   paper-level banks rather than the original's region-level §26 banks.
5. Write docs/RESULTS.md (a short findings write-up with the tables and the figures from docs/report)
   and commit it to a branch.
```
