# Instructions for Claude sessions in this repo

This repo is an independent reproduction of the positive results in `docs/source_history.md`
(explicit EDL §18, oracle residual specialists §23, causal router §27), plus ablations B1–B10.

- **Single source of truth:** `repro/config.py`. Experiments are entries in `EXPERIMENTS`. Change
  behaviour by adding or overriding config fields, not by special-casing names in the code.
- **Every guess about the original setup** goes in `docs/RECONSTRUCTION.md`. If you change one,
  update that file in the same commit.
- **No GPU in cloud sessions.** Never start the `full` profile here. Use `--profile smoke`, which is
  synthetic data on CPU and takes about 2 minutes. Real training runs on MIT ORCD via `slurm/` (see
  `docs/ORCD.md`).
- **Before you commit:** run `python -m pytest`. If you changed anything under `repro/`, also run
  `REPRO_SLOW=1 python -m pytest tests/test_smoke_pipeline.py`.
- **Do not edit** files under `results_orcd/`. They are real results copied back from ORCD.
- The smoke profile only checks that the code runs. Never present smoke numbers as findings.
