#!/usr/bin/env bash
# Guarded launcher for ONE CUDA-capable command. Defaults to dry-run.
#   scripts/gpu_guarded.sh "<purpose>" <python script> [args...]
# Executes only if MOE_GPU_LAUNCH=1 is set in the environment of this invocation; the Python program
# additionally requires --execute and MOE_RESEARCH_APPROVE=gpu. Steps: fresh read-only occupancy check of GPU0
# (compute + graphics PIDs via NVML), refuse if any process is present, expose only GPU 0, enforce a 90-minute
# wall-clock limit with SIGINT first (then SIGTERM after 120 s), and register the process.
set -u
umask 077
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
[ "$ROOT" = "/home/zapan4/moe-research" ] && [ "$(pwd -P)" = "$ROOT" ] || { echo "run from project root" >&2; exit 2; }
purpose="$1"; shift
n=$(nvidia-smi -i 0 -q -d PIDS | grep -c "Process ID")
ts=$(date -Is)
if [ "$n" != "0" ]; then
  printf '%s\tGPU_BLOCKED\t%s: %s process(es) on GPU0; not launched\n' "$ts" "$purpose" "$n" >> "$ROOT/ledger/safety_ledger.tsv"
  echo "BLOCKED $ts: GPU0 has $n process(es)" >&2; exit 3
fi
printf '%s\tGPU_OCCUPANCY\tlauncher pre-check for "%s": 0 processes\n' "$ts" "$purpose" >> "$ROOT/ledger/safety_ledger.tsv"
if [ "${MOE_GPU_LAUNCH:-0}" != "1" ]; then echo "[dry-run] set MOE_GPU_LAUNCH=1 to launch: $*"; exit 0; fi
export MOE_CUDA_VISIBLE_DEVICES=0
source "$ROOT/env.sh"
[ "$CUDA_VISIBLE_DEVICES" = "0" ] || { echo "CUDA_VISIBLE_DEVICES not 0" >&2; exit 2; }
exec "$ROOT/scripts/run_registered.sh" "$purpose" timeout --signal=INT --kill-after=120 5400 "$ROOT/.venv/bin/python" "$@"
