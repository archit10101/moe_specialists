#!/usr/bin/env bash
# Run one command as a registered child process: records PID, PPID, UID, start time, executable,
# purpose and exit status in ledger/process_registry.tsv. Usage: run_registered.sh "<purpose>" cmd args...
set -u
umask 077
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
[ "$ROOT" = "/home/zapan4/moe-research" ] || { echo "bad root $ROOT" >&2; exit 2; }
REG="$ROOT/ledger/process_registry.tsv"
purpose="$1"; shift
"$@" &
pid=$!
start=$(ps -o lstart= -p "$pid" 2>/dev/null | tr -s ' ' || echo unknown)
exe=$(readlink "/proc/$pid/exe" 2>/dev/null || echo "$1")
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "$pid" "$$" "$(id -u)" "$start" "$exe" "$purpose" "started" >> "$REG"
trap 'kill -INT "$pid" 2>/dev/null' INT TERM
wait "$pid"; rc=$?
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "$pid" "$$" "$(id -u)" "$start" "$exe" "$purpose" "exit=$rc" >> "$REG"
exit $rc
