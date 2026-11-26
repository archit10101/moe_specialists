# SAFETY.md — shared-host controls for this project

Project root: `/home/zapan4/moe-research`. It is owned by UID 1027, is not a symlink, and is mode 0700. Its
original mode was 0775 (group-writable); it was tightened on 2026-10-31 and the change is recorded in the ledger.
Every artifact, cache, environment and log stays under this root. No symlinks are created; this is asserted
after each Hugging Face download.

## Ledgers (append-only, mode 0600, `ledger/`)

| File | Contents |
|---|---|
| `safety_ledger.tsv` | Root/disk/occupancy checks, network actions, data/segment builds, blocked launches, notes |
| `process_registry.tsv` | Every Python/compute process: PID, PPID, UID, start time, executable, purpose, exit status |
| `gpu_time_ledger.tsv` | Every CUDA-capable run: start, end, seconds, run id, purpose, final status |

## Interlocks

- Every executable that can download, load a model, initialize CUDA, train, evaluate or profile
  (`prepare_data.py`, `build_segments.py`, `train.py`, `benchmark.py`, `gpu_checks.py`, `calibrate.py`,
  `probe.py`, `evaluate.py`) runs as a **dry-run by default**. To execute it needs both `--execute` and
  `MOE_RESEARCH_APPROVE=<download|cpu-model|gpu>`, scoped to that single process.
- `env.sh` hides CUDA (`CUDA_VISIBLE_DEVICES=""`) unless `scripts/gpu_guarded.sh` sets it to `0`.
- `scripts/gpu_guarded.sh` does the following:
  1. Runs a fresh read-only NVML check of GPU 0 (`nvidia-smi -q -d PIDS`, covering compute and graphics).
     If any process is present it refuses, records BLOCKED with a timestamp and count, and does not wait.
  2. Is itself a dry-run unless `MOE_GPU_LAUNCH=1`.
  3. Exposes only GPU 0.
  4. Wraps the command in `timeout --signal=INT --kill-after=120 5400`: SIGINT first, then a kill of **only
     this child** after 120 s.
  5. Registers the process.
- Inside Python, `guard.occupancy_gate` repeats the check before CUDA initialization.
  `guard.setup_cuda_device` caps the allocator with `set_per_process_memory_fraction` at 56 GiB.
  `guard.RunLimits` aborts cleanly at 95% of 56 GiB, at the wall-clock limit (with a SIGALRM backstop), or when
  `OccupancyMonitor` (polls every 10–15 s, read-only) detects a foreign GPU PID. Partial artifacts are kept and
  the run is marked `CONTAMINATED` or `ABORTED`.
- Foreign processes are only counted. No command line, owner, environment, file or memory of another process
  is inspected.
- Threads: OMP/MKL/OpenBLAS/numexpr/rayon are capped at 4, and torch intra-op is capped at 4. The CPU encoder
  preprocessing uses 8 threads. There are no data-loader workers, and the tokenizer runs without parallelism.
- One job at a time. CPU preprocessing is never run during a GPU run.

## Disk policy

The reserve is **100 GiB** (the conservative reading of "100 GB"). `guard.require_disk(planned)` fails closed
if `free − planned < 100 GiB`. At the start, 110,074,671,104 bytes were free, so this project has only about
2.5 GiB in total. That is the binding constraint on installing CUDA PyTorch (see the Stage B report).

## Network

Allowed hosts: HF API/Hub for the named, pinned repositories (FLAME metadata only; FineWeb-Edu row groups by
range read; the Pythia `tokenizer.json`; MiniLM), PyPI and `download.pytorch.org/whl/cpu`, and arXiv/GitHub
pages (read for the audit).

Not used: `trust_remote_code`, git clones, hooks, curl-pipe-shell, telemetry (W&B and HF telemetry are
disabled in `env.sh`), uploads, ports or servers.

## Destructive operations

None are used: no `rm -rf`, no `git clean`/`reset`, no overwrites. Results use timestamped run directories and
atomic `tmp`+`rename` writes. `guard.atomic_write_*` refuses to overwrite, and data and segment caches are
published read-only (0400).

## Not authorized and not used

sudo/apt/containers, GPU reset/clock/power/MIG/MPS changes, DCGM, KempnerPulse, Nsight, profilers,
multi-GPU, test-split access (`evaluate.py` needs `--unlock-test` + `MOE_RESEARCH_TEST_UNLOCK=1`).
