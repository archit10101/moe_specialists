#!/usr/bin/env bash
# Build the report from whatever has finished so far (login node, CPU, ~1 min). Unfinished claims show
# PENDING; finished seeds are already paired. Writes $REPRO_WORK/results/REPORT.md and prints the claims.
#   ./slurm/report_now.sh            the real run
#   ./slurm/report_now.sh --smoke    the smoke test
set -euo pipefail
REPRO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPRO_ROOT
[ "${1:-}" = --smoke ] && export REPRO_PROFILE=smoke
source "$REPRO_ROOT/slurm/env.sh"
repro_activate
cd "$REPRO_ROOT"
python -m repro.run report --profile "$REPRO_PROFILE"
echo
echo "full report: $REPRO_WORK$([ "$REPRO_PROFILE" = smoke ] && echo /smoke)/results/REPORT.md"
