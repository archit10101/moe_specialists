#!/usr/bin/env bash
# Start everything with one command on the ORCD LOGIN node, then log out and check back later:
#
#   ./slurm/launch.sh                  setup + smoke test + core reproduction
#   ./slurm/launch.sh --tier all       ... + the ablations
#   ./slurm/launch.sh --foreground     same, but stay attached (default: detach and log to a file)
#   ./slurm/launch.sh --dry-run        skip setup, print the sbatch commands, submit nothing
#
# What it does, in order:
#   1. slurm/setup_login.sh (conda env, downloads, offline check, CPU tests). On failure nothing is submitted.
#   2. the smoke pipeline (submit_all.sh --smoke, ~15 min of GPU).
#   3. a small CPU "gate" job after the smoke report: it succeeds only if every smoke job finished.
#   4. the real pipeline (submit_all.sh --after <gate>): every job waits for the gate to SUCCEED.
#      If the smoke test failed, SLURM cancels the real jobs (--kill-on-invalid-dep); only the CPU-only
#      `prepare` job (data tokenisation) starts immediately either way, since the real run needs it.
#
# In the morning:   tail $REPRO_WORK/logs/launch_*.log;  ./slurm/status.sh --smoke;  ./slurm/status.sh
# Re-running after a fix is safe: setup is idempotent and finished jobs are skipped.
set -euo pipefail
REPRO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPRO_ROOT
TIER=core; FG=0; DRY=0; ARGS=( "$@" )
while [ $# -gt 0 ]; do
  case "$1" in
    --tier) shift; TIER="$1" ;;
    --foreground) FG=1 ;;
    --dry-run) DRY=1; FG=1 ;;
    -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done
source "$REPRO_ROOT/slurm/config.sh"
LOGS="$REPRO_WORK/logs"; mkdir -p "$LOGS"

if [ "$FG" = 0 ]; then                      # detach so the run survives logging out
  log="$LOGS/launch_$(date '+%Y%m%d_%H%M%S').log"
  nohup setsid "$0" --foreground "${ARGS[@]}" > "$log" 2>&1 < /dev/null &
  echo "launched in the background (pid $!). You can log out now."
  echo "  progress:  tail -f $log"
  echo "  tomorrow:  ./slurm/status.sh --smoke  and  ./slurm/status.sh"
  exit 0
fi

step() { echo; echo "=== $(date '+%F %T') $*"; }
DRYFLAG=(); [ "$DRY" = 1 ] && DRYFLAG=( --dry-run )

if [ "$DRY" = 1 ]; then step "1/4 setup: skipped (dry run)"
else step "1/4 setup (login node)"; "$REPRO_ROOT/slurm/setup_login.sh"; fi

step "2/4 smoke pipeline"
smoke_out="$("$REPRO_ROOT/slurm/submit_all.sh" --smoke "${DRYFLAG[@]}")"
echo "$smoke_out"
# the smoke report job runs after every other smoke job (afterany), so the gate follows it
report_id="$(echo "$smoke_out" | awk '$1 == "report" && $2 == "submitted" { print $3 } $1 == "report" && $2 == "already" { gsub(/[()]/, "", $4); print $4 }' | head -1)"
[ -n "$report_id" ] || { echo "ERROR: could not find the smoke report job id in the output above" >&2; exit 1; }

step "3/4 smoke gate (after smoke report job $report_id)"
gate_flags=( -J "${JOB_PREFIX}smoke_gate" -t 00:20:00 -p "$CPU_PARTITION" -c 1 --mem=4G
             -o "$LOGS/%x_%j.out" --chdir="$REPRO_ROOT"
             --export="ALL,REPRO_ROOT=$REPRO_ROOT,REPRO_WORK=$REPRO_WORK"
             --dependency="afterany:$report_id" )
[ -n "$SLURM_ACCOUNT" ] && gate_flags+=( -A "$SLURM_ACCOUNT" )
[ -n "$MAIL_USER" ] && gate_flags+=( --mail-user="$MAIL_USER" --mail-type=FAIL )
[ -n "$EXTRA_SBATCH" ] && gate_flags+=( $EXTRA_SBATCH )
if [ "$DRY" = 1 ]; then
  echo "sbatch ${gate_flags[*]} slurm/smoke_gate.sbatch" | tee -a "$LOGS/dry_run.txt"; gate_id="dry-gate"
else
  gate_id="$(sbatch --parsable "${gate_flags[@]}" "$REPRO_ROOT/slurm/smoke_gate.sbatch")"; gate_id="${gate_id%%;*}"
  printf '%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$gate_id" "${JOB_PREFIX}smoke_gate" "afterany:$report_id" >> "$LOGS/jobs.tsv"
fi
echo "gate job: $gate_id"

step "4/4 real pipeline (tier $TIER; every job but prepare waits for gate $gate_id)"
"$REPRO_ROOT/slurm/submit_all.sh" --tier "$TIER" --after "$gate_id" "${DRYFLAG[@]}"

step "all submitted"
echo "If the smoke test fails, the gate fails and SLURM cancels the real jobs (except prepare): fix, then run"
echo "./slurm/launch.sh again (finished jobs are skipped; delete $REPRO_WORK/smoke to redo the smoke test)."
