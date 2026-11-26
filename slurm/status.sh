#!/usr/bin/env bash
# Progress overview (login node, read-only):  ./slurm/status.sh [--smoke] [--tier all]
set -euo pipefail
REPRO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPRO_ROOT
TIER=all
while [ $# -gt 0 ]; do
  case "$1" in
    --smoke) export REPRO_PROFILE=smoke JOB_PREFIX="${JOB_PREFIX:-repro_}smoke_" ;;
    --tier) shift; TIER="$1" ;;
  esac
  shift
done
source "$REPRO_ROOT/slurm/env.sh"
repro_activate
echo "== queue ($JOB_PREFIX*)"
squeue -u "${USER:-$(id -un)}" -o '%.10i %.45j %.9T %.10M %.12l %R' 2>/dev/null | awk -v p="$JOB_PREFIX" 'NR==1 || index($2, p) == 1' || true
echo
echo "== pipeline (profile $REPRO_PROFILE, tier $TIER)"
python -m repro.run plan --profile "$REPRO_PROFILE" --tier "$TIER" --format tsv \
  | awk -F'\t' '{ n[$4]++; split($1, k, ":"); s[k[1] "\t" $4]++ } END {
      for (x in s) { split(x, y, "\t"); printf "  %-8s %-5s %d\n", y[1], y[2], s[x] }
      printf "  total: %d done, %d todo\n", n["done"], n["todo"] }' | sort
LOGS="$REPRO_WORK/logs"
if [ -d "$LOGS" ]; then
  failed="$(grep -l 'FAILED\|Traceback' "$LOGS"/*.out 2>/dev/null | tail -5 || true)"
  if [ -n "$failed" ]; then echo; echo "== recent failures (last lines)"; for f in $failed; do echo "-- $f"; tail -n 4 "$f"; done; fi
  latest="$(ls -t "$LOGS"/*.out 2>/dev/null | head -1 || true)"
  [ -n "$latest" ] && { echo; echo "== latest log: $latest"; tail -n 3 "$latest"; }
fi
[ -f "$REPRO_WORK/results/REPORT.md" ] && echo && echo "report: $REPRO_WORK/results/REPORT.md"
[ -f "$REPRO_WORK/smoke/results/REPORT.md" ] && echo "smoke report: $REPRO_WORK/smoke/results/REPORT.md"
exit 0
