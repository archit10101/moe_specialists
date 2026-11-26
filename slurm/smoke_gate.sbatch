#!/usr/bin/env bash
# Gate between the smoke test and the real run (submitted by launch.sh, after the smoke report job).
# Exits 0 only if every smoke job finished (its DONE marker exists) and the smoke report was written.
# The real run's jobs (all but prepare) depend on this job with afterok, so a failed smoke test cancels them.
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -euo pipefail
source "$REPRO_ROOT/slurm/env.sh"
repro_activate
cd "$REPRO_ROOT"
echo "=== $(date '+%F %T') smoke gate (job ${SLURM_JOB_ID:-local})"
plan="$(python -m repro.run plan --profile smoke --tier "${GATE_TIER:-core}" --format tsv)"
todo="$(echo "$plan" | awk -F'\t' '$4 != "done" && $1 != "report" { print $1 }')"
report="$REPRO_WORK/smoke/results/REPORT.md"
if [ -n "$todo" ] || [ ! -f "$report" ]; then
  echo "FAILED - the smoke test did not finish. Unfinished smoke jobs:"
  echo "${todo:-  (none; but $report is missing)}" | sed 's/^/  /'
  echo "SLURM will cancel the real run's jobs (all but prepare). See ./slurm/status.sh --smoke for the logs."
  exit 1
fi
echo "smoke test OK: every smoke job finished ($report). Releasing the real run."
