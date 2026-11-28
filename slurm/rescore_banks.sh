#!/usr/bin/env bash
# Re-score finished banks whose swap matrix came from the old small val probe (eval.json without
# "swap_tokens"): removes their DONE_eval marker and re-submits the tier. Training is skipped (DONE_train
# stays), so each job only re-runs the evaluation (minutes). Routers and everything else are untouched.
#   ./slurm/rescore_banks.sh            tier paper
#   ./slurm/rescore_banks.sh paper_extra
#   ./slurm/rescore_banks.sh --dry-run  only list what would be re-scored
set -euo pipefail
REPRO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPRO_ROOT
DRY=0; TIER=paper
for a in "$@"; do case "$a" in --dry-run) DRY=1 ;; *) TIER="$a" ;; esac; done
source "$REPRO_ROOT/slurm/config.sh"
n=0
for d in "$REPRO_WORK"/runs/bank.*/s* "$REPRO_WORK"/runs/abl.bank_*/s*; do
  [ -f "$d/DONE_eval" ] && [ -f "$d/eval.json" ] || continue
  grep -q '"swap_tokens"' "$d/eval.json" && continue          # already scored on all of val
  grep -q '"n_spec": 1[,}]' "$d/eval.json" && continue         # dense residuals have no swap matrix
  echo "re-score ${d#$REPRO_WORK/runs/}"
  n=$((n + 1))
  [ "$DRY" = 1 ] || rm -f "$d/DONE_eval"
done
echo "$n bank runs to re-score"
[ "$DRY" = 1 ] || [ "$n" = 0 ] || "$REPRO_ROOT/slurm/submit_all.sh" --tier "$TIER"
