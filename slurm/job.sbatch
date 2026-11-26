#!/usr/bin/env bash
# The ONE job script. Every pipeline job is:   python -m repro.run <args>
# submit_all.sh passes the args (e.g. "base --exp base.edl --seed 2") and the resources.
#
# Exit codes from repro.run:
#   0 done | 3 time budget reached (checkpoint saved; re-run submit_all.sh to resume)
#   143 SIGTERM / preemption (checkpoint saved; --requeue restarts the job and it resumes)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --signal=B:TERM@120
set -euo pipefail
source "$REPRO_ROOT/slurm/env.sh"
repro_activate
cd "$REPRO_ROOT"
echo "=== $(date '+%F %T') job ${SLURM_JOB_ID:-local} ${SLURM_JOB_NAME:-} on $(hostname)"
echo "    python -m repro.run $* --profile $REPRO_PROFILE   (work: $REPRO_WORK)"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
set +e
python -u -m repro.run "$@" --profile "$REPRO_PROFILE" &
pid=$!
trap 'kill -TERM $pid 2>/dev/null' TERM
wait $pid; rc=$?
if [ "$rc" -gt 128 ] && kill -0 $pid 2>/dev/null; then wait $pid; rc=$?; fi
set -e
echo "=== $(date '+%F %T') exit code $rc"
case "$rc" in
  0) ;;
  3) echo "time budget reached; checkpoint saved. Re-run ./slurm/submit_all.sh to continue." ;;
  143) echo "stopped by signal; checkpoint saved (SLURM requeue will resume it)." ;;
  *) echo "FAILED - see the traceback above." ;;
esac
exit "$rc"
