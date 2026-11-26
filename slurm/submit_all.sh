#!/usr/bin/env bash
# Submit every job of the pipeline that is not finished yet, wired with SLURM dependencies.
# Safe to re-run at any time: finished jobs are skipped, queued/running ones are reused, and
# unfinished training resumes from its checkpoint.
#
#   ./slurm/submit_all.sh                     core experiments (the three positive results)
#   ./slurm/submit_all.sh --tier ablation     the ablations B1-B10 (+ any core job they need)
#   ./slurm/submit_all.sh --tier all          everything
#   ./slurm/submit_all.sh --only bank.sem,router.sem
#   ./slurm/submit_all.sh --smoke             whole pipeline on the tiny synthetic profile (~15 min)
#   ./slurm/submit_all.sh --bench             one GPU job timing each job type (do this first)
#   ./slurm/submit_all.sh --dry-run ...       print the sbatch commands, submit nothing
set -euo pipefail
REPRO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPRO_ROOT

TIER=core; ONLY=""; SMOKE=0; BENCH=0; DRY=0; DRY_N=0
while [ $# -gt 0 ]; do
  case "$1" in
    --tier) shift; TIER="$1" ;;
    --only) shift; ONLY="$1" ;;
    --smoke) SMOKE=1 ;;
    --bench) BENCH=1 ;;
    --dry-run) DRY=1 ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done
if [ "$SMOKE" = 1 ]; then export REPRO_PROFILE=smoke JOB_PREFIX="${JOB_PREFIX:-repro_}smoke_"; fi
source "$REPRO_ROOT/slurm/env.sh"
repro_activate
die() { echo "ERROR: $*" >&2; exit 1; }
[ "$DRY" = 1 ] || command -v sbatch >/dev/null 2>&1 || die "sbatch not found - run this on orcd-login.mit.edu"
[ -n "$GPU_TYPE" ] || die "GPU_TYPE is empty in slurm/config.sh (untyped requests get a slow L40S)"
LOGS="$REPRO_WORK/logs"; mkdir -p "$LOGS"

common_flags() {  # name time resource
  local f=( -J "$1" -t "$2" -o "$LOGS/%x_%j.out" --chdir="$REPRO_ROOT"
            --export="ALL,REPRO_ROOT=$REPRO_ROOT,REPRO_WORK=$REPRO_WORK,REPRO_PROFILE=$REPRO_PROFILE" )
  if [ "$3" = gpu ]; then f+=( -p "$GPU_PARTITION" -G "${GPU_TYPE}:1" -c "$GPU_CPUS" --mem="$GPU_MEM" --requeue )
  else f+=( -p "$CPU_PARTITION" -c "$CPU_CPUS" --mem="$CPU_MEM" ); fi
  [ -n "$SLURM_ACCOUNT" ] && f+=( -A "$SLURM_ACCOUNT" )
  [ -n "$MAIL_USER" ] && f+=( --mail-user="$MAIL_USER" --mail-type=FAIL )
  [ -n "$EXTRA_SBATCH" ] && f+=( $EXTRA_SBATCH )
  printf '%s\n' "${f[@]}"
}

submit() {  # name time resource dependency args...
  local name="$1" time="$2" res="$3" dep="$4"; shift 4
  local flags=(); while IFS= read -r l; do flags+=( "$l" ); done < <(common_flags "$name" "$time" "$res")
  [ -n "$dep" ] && flags+=( --dependency="$dep" --kill-on-invalid-dep=yes )
  if [ "$DRY" = 1 ]; then echo "sbatch ${flags[*]} slurm/job.sbatch $*" >> "$LOGS/dry_run.txt"; echo "dry-$name"; return; fi
  local jid; jid="$(sbatch --parsable "${flags[@]}" "$REPRO_ROOT/slurm/job.sbatch" "$@")" || die "sbatch failed for $name"
  jid="${jid%%;*}"
  printf '%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$jid" "$name" "$dep" >> "$LOGS/jobs.tsv"
  echo "$jid"
}

if [ "$BENCH" = 1 ]; then
  jid="$(submit "${JOB_PREFIX}bench" 00:45:00 gpu "" bench)"
  echo "submitted bench job $jid -> $LOGS/${JOB_PREFIX}bench_${jid}.out"; exit 0
fi

PLAN="$(python -m repro.run plan --profile "$REPRO_PROFILE" --tier "$TIER" ${ONLY:+--only "$ONLY"} --format tsv)"
if [ "$DRY" != 1 ] && echo "$PLAN" | grep -q $'^prepare\t.*\ttodo\t'; then
  if [ "$REPRO_PROFILE" = full ] && ! python -m repro.run prepare --download --profile full >/dev/null 2>&1; then
    die "raw data not in the cache (compute nodes are offline): run ./slurm/setup_login.sh first"
  fi
fi
ACTIVE="$( { [ "$DRY" = 1 ] || squeue -h -u "$USER" -o '%j|%i'; } 2>/dev/null | grep "^${JOB_PREFIX}" || true)"
IDS=""                                    # "key id" lines (bash 3.2 has no associative arrays)
get_id() { echo "$IDS" | awk -v k="$1" '$1 == k { print $2; exit }'; }
n_new=0; n_done=0; n_active=0
while IFS=$'\t' read -r key res time state deps args; do
  [ -n "$key" ] || continue
  name="${JOB_PREFIX}$(echo "$key" | tr ':' '_')"
  if [ "$state" = done ]; then n_done=$((n_done + 1)); continue; fi
  a="$(echo "$ACTIVE" | awk -F'|' -v n="$name" '$1 == n { print $2; exit }')"
  if [ -n "$a" ]; then IDS="$IDS
$key $a"; n_active=$((n_active + 1)); printf '  %-44s already queued (%s)\n' "$key" "$a"; continue; fi
  ids=""
  for d in ${deps//,/ }; do i="$(get_id "$d")"; [ -n "$i" ] && ids="$ids:$i"; done
  dep=""
  if [ -n "$ids" ]; then
    if [ "$key" = report ]; then dep="afterany$ids"; else dep="afterok$ids"; fi
  fi
  # shellcheck disable=SC2086
  jid="$(submit "$name" "$time" "$res" "$dep" $args)"
  IDS="$IDS
$key $jid"; n_new=$((n_new + 1))
  printf '  %-44s submitted %s%s\n' "$key" "$jid" "${dep:+ (after ${dep#*:})}"
done <<< "$PLAN"
echo "profile=$REPRO_PROFILE tier=$TIER: $n_new submitted, $n_active already queued, $n_done done."
echo "logs: $LOGS   progress: ./slurm/status.sh${SMOKE:+ }$( [ "$SMOKE" = 1 ] && echo --smoke )"
