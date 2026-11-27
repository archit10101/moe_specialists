#!/usr/bin/env bash
# Start everything with one command on the ORCD LOGIN node, then log out and check back later:
#
#   ./slurm/launch.sh                  setup + smoke test + core reproduction
#   ./slurm/launch.sh --tier all       ... + the ablations
#   ./slurm/launch.sh --tier paper     the §23/§27 paper runs only (later: ./slurm/submit_all.sh --tier paper_extra)
#   ./slurm/launch.sh --foreground     same, but stay attached (default: detach and log to a file)
#   ./slurm/launch.sh --dry-run        skip setup, print the sbatch commands, submit nothing
#
# What it does, in order:
#   1. slurm/setup_login.sh (conda env, downloads, offline check, CPU tests), then the data preparation for the
#      smoke and the real run, both on the login node (minutes). On failure nothing is submitted.
#   2. the smoke pipeline (submit_all.sh --smoke, ~15 min of GPU).
#   3. a small CPU "gate" job after the smoke report: it succeeds only if every smoke job finished.
#   4. the real pipeline (submit_all.sh --after <gate>): every job waits for the gate to SUCCEED.
#      If the smoke test failed, SLURM cancels the real jobs (--kill-on-invalid-dep); only the CPU-only
#      `prepare` job (data tokenisation) starts immediately either way, since the real run needs it.
#
# SLURM's per-user cap on submitted jobs is handled: whatever does not fit is submitted later, every
# RETRY_MINUTES (default 10) as jobs finish, so the launcher keeps running until the last job is queued.
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
    -h|--help) sed -n '2,23p' "$0"; exit 0 ;;
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
else
  step "1/4 setup (login node)"; "$REPRO_ROOT/slurm/setup_login.sh"
  # Data preparation is CPU-only and takes minutes, and the raw data is already cached here, so build it on the
  # login node instead of waiting for a CPU node (the job graph then skips `prepare`).
  step "1/4 data preparation (login node)"
  (
    source "$REPRO_ROOT/slurm/env.sh"; repro_activate; cd "$REPRO_ROOT"
    for prof in smoke full; do OMP_NUM_THREADS=4 python -m repro.run prepare --profile "$prof"; done
  )
fi

# SLURM caps how many jobs one user may have submitted (ORCD: QOSMaxSubmitJobPerUserLimit).
# submit_all.sh then queues what fits and exits 10; we wait for jobs to finish and run it again.
RETRY_MINUTES="${RETRY_MINUTES:-10}"
RETRY_SECONDS="${RETRY_SECONDS:-$((RETRY_MINUTES * 60))}"
OUT=""
submit_until_done() {   # args for submit_all.sh; its last output ends up in $OUT
  local rc
  while true; do
    rc=0; OUT="$("$REPRO_ROOT/slurm/submit_all.sh" "$@" "${DRYFLAG[@]}")" || rc=$?
    echo "$OUT"
    [ "$rc" = 10 ] || return "$rc"
    echo "-- $(date '+%F %T') submit limit reached; retrying in $RETRY_MINUTES min"
    sleep "$RETRY_SECONDS"
  done
}
gate_state() {          # PENDING / RUNNING / COMPLETED / FAILED / CANCELLED ... of a job
  local st; st="$(squeue -h -j "$1" -o %T 2>/dev/null || true)"
  [ -n "$st" ] && { echo "$st"; return; }
  sacct -n -X -j "$1" -o State 2>/dev/null | awk 'NF { print $1; exit }'
}

step "2/4 smoke pipeline"
submit_until_done --smoke --tier "$TIER"      # smoke-test the same job graph the real run will use
# the smoke report job runs after every other smoke job (afterany), so the gate follows it
report_id="$(echo "$OUT" | awk '$1 == "report" && $2 == "submitted" { print $3 } $1 == "report" && $2 == "already" { gsub(/[()]/, "", $4); print $4 }' | head -1)"
[ -n "$report_id" ] || { echo "ERROR: could not find the smoke report job id in the output above" >&2; exit 1; }

step "3/4 smoke gate (after smoke report job $report_id)"
gate_id=""
[ "$DRY" = 1 ] || gate_id="$(squeue -h -u "${USER:-$(id -un)}" -n "${JOB_PREFIX}smoke_gate" -o %i 2>/dev/null | head -1 || true)"
if [ -n "$gate_id" ]; then
  echo "reusing the queued gate job $gate_id"
else
  gate_flags=( -J "${JOB_PREFIX}smoke_gate" -t 00:20:00 -p "$CPU_PARTITION" -c 1 --mem=4G
               -o "$LOGS/%x_%j.out" --chdir="$REPRO_ROOT"
               --export="ALL,REPRO_ROOT=$REPRO_ROOT,REPRO_WORK=$REPRO_WORK,GATE_TIER=$TIER"
               --dependency="afterany:$report_id" )
  [ -n "$SLURM_ACCOUNT" ] && gate_flags+=( -A "$SLURM_ACCOUNT" )
  [ -n "$MAIL_USER" ] && gate_flags+=( --mail-user="$MAIL_USER" --mail-type=FAIL )
  [ -n "$EXTRA_SBATCH" ] && gate_flags+=( $EXTRA_SBATCH )
  if [ "$DRY" = 1 ]; then
    echo "sbatch ${gate_flags[*]} slurm/smoke_gate.sbatch" | tee -a "$LOGS/dry_run.txt"; gate_id="dry-gate"
  else
    until gate_id="$(sbatch --parsable "${gate_flags[@]}" "$REPRO_ROOT/slurm/smoke_gate.sbatch")"; do
      echo "-- gate submission failed (submit limit?); retrying in $RETRY_MINUTES min"; sleep "$RETRY_SECONDS"
    done
    gate_id="${gate_id%%;*}"
    printf '%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$gate_id" "${JOB_PREFIX}smoke_gate" "afterany:$report_id" >> "$LOGS/jobs.tsv"
  fi
fi
echo "gate job: $gate_id"

step "4/4 real pipeline (tier $TIER; every job but prepare waits for gate $gate_id)"
while true; do
  after=( --after "$gate_id" )
  if [ "$DRY" != 1 ]; then
    st="$(gate_state "$gate_id")"
    case "$st" in
      COMPLETED) after=() ;;                       # smoke test passed: no need to wait for it any more
      PENDING|RUNNING|CONFIGURING|COMPLETING|"") ;;
      *) echo "smoke gate $gate_id ended as $st: the smoke test FAILED, so nothing more of the real run is"
         echo "submitted, and SLURM cancels the real jobs already queued (except prepare). Fix it and re-run."
         exit 1 ;;
    esac
  fi
  rc=0; "$REPRO_ROOT/slurm/submit_all.sh" --tier "$TIER" "${after[@]}" "${DRYFLAG[@]}" || rc=$?
  [ "$rc" = 0 ] && break
  [ "$rc" = 10 ] || exit "$rc"
  echo "-- $(date '+%F %T') submit limit reached; retrying in $RETRY_MINUTES min (jobs already queued are kept)"
  sleep "$RETRY_SECONDS"
done

step "all submitted"
echo "If the smoke test fails, the gate fails and SLURM cancels the real jobs (except prepare): fix, then run"
echo "./slurm/launch.sh again (finished jobs are skipped; delete $REPRO_WORK/smoke to redo the smoke test)."
