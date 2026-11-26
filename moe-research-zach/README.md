# Causal Semantic Chunk Routing for MoE — controlled single-GPU experiment

Can a causal Transformer use the observed prefix of a text chunk to choose a shared per-layer candidate expert
pool, while keeping token-level top-k selection inside that pool? This repository tests the idea skeptically.
Null results count as results.

| Document | Contents |
|---|---|
| PRIOR_ART.md | Claim-by-claim overlap with EMO, Lory, 2402.13089, FLAME-MoE, nanoMoE |
| MODEL_DECISION.md | Why FLAME-MoE-38M-100M was not used; the from-scratch FLAME-38M-matched model; trainable sets |
| SAFETY.md | Shared-host controls, interlocks, ledgers |
| EXPERIMENT_MANIFEST.json | Machine-checked matched fields for conditions A–F |
| EXPERIMENT_REPORT.md | Pre-registration and results |

## Status labels

| Stage | Status |
|---|---|
| A — audit, implementation, CPU tests, small pinned downloads | **DONE** |
| B — guarded GPU smoke and diagnostics | See EXPERIMENT_REPORT.md §5 |
| Six-condition pilot, seeds, 5M extension, test split, profiling | **NOT AUTHORIZED / NOT RUN** |
| `configs/smoke-small-v1` (16 experts, top-2, m=4): engineering smoke tests only, never evidence | CPU checks DONE; GPU commands NOT RUN (docs/SMOKE_SMALL_V1.md) |
| Full-size design `configs/v2` | FROZEN |

## Layout

```
src/csr/model.py         FLAME-38M-matched MoE; token routing and causal chunk-pool routing (one code path); incremental step
src/csr/segment.py       fixed / online causal semantic / length-matched random boundaries; streaming chunker
src/csr/semantic_loss.py relational KL(Q||R) objective with diagnostics
src/csr/segcache.py      deterministic boundary caches and frozen prefix embeddings
src/csr/data.py          subset download (range reads), dedup, doc-level splits, tokenization, packing
src/csr/train_lib.py     loss accounting, schedule, evaluation, paired bootstrap
src/csr/config.py        conditions A–F, fail-closed manifest checker
src/csr/guard.py         interlocks, GPU occupancy gate/monitor, VRAM/time limits, atomic IO, run dirs
prepare_data.py build_segments.py build_evalsets.py make_configs.py
train.py evaluate.py probe.py benchmark.py calibrate.py gpu_checks.py
scripts/gpu_guarded.sh   the only way to launch a CUDA-capable command
tests/                   CPU tests (CUDA hidden) for correctness gates 1–15
```

## Commands

Every command runs from the project root after `source ./env.sh`. Every executable is a dry-run unless it gets
both `--execute` and `MOE_RESEARCH_APPROVE=<action>`.

```bash
# Stage A (CPU)
MOE_RESEARCH_APPROVE=download  .venv/bin/python prepare_data.py download --execute
MOE_RESEARCH_APPROVE=cpu-model .venv/bin/python prepare_data.py build --execute
MOE_RESEARCH_APPROVE=cpu-model .venv/bin/python build_segments.py --execute
SEG=artifacts/segments/<cache_id>
MOE_RESEARCH_APPROVE=cpu-model .venv/bin/python build_evalsets.py --segments $SEG --execute
.venv/bin/python make_configs.py --segments $SEG --tag v1
CUDA_VISIBLE_DEVICES= .venv/bin/python -m pytest -q tests

# Stage B (GPU, one command at a time; each is re-gated on GPU0 occupancy)
MOE_GPU_LAUNCH=1 MOE_RESEARCH_APPROVE=gpu scripts/gpu_guarded.sh "gpu checks" gpu_checks.py --execute
MOE_GPU_LAUNCH=1 MOE_RESEARCH_APPROVE=gpu scripts/gpu_guarded.sh "benchmark E" benchmark.py --condition E --steps 150 --execute \
    --set semantic.lambda=<smoke> --set semantic.tau_s=<smoke> --set semantic.tau_r=<smoke>
MOE_GPU_LAUNCH=1 MOE_RESEARCH_APPROVE=gpu scripts/gpu_guarded.sh "smoke A" train.py --device cuda --condition A \
    --set train.token_budget=300000 --run-name smoke_A --execute
MOE_GPU_LAUNCH=1 MOE_RESEARCH_APPROVE=gpu scripts/gpu_guarded.sh "probe" probe.py --checkpoint runs/<smoke_A>/model.pt --execute
```

The exact commands that were actually executed, with their run directories, are listed in
EXPERIMENT_REPORT.md §5.
