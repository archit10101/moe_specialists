#!/usr/bin/env bash
# specialization-audit-v1: extraction + counterfactuals for the 18 checkpoints, sequentially, one GPU process at a time.
# Per run: fresh GPU0 occupancy check (also inside the launcher), disk reserve check, cumulative audit GPU-time check
# against the 4 GPU-hour ceiling, guarded launch, status gate. Stops on the first failure. Resumable: keys already in
# runs.tsv are skipped.
set -u
umask 077
ROOT=/home/zapan4/moe-research
cd "$ROOT" || exit 2
OUT=.tmp/spec_audit_v1
mkdir -p "$OUT"
TSV=$OUT/runs.tsv; LOG=$OUT/driver.log
touch "$TSV"
log(){ echo "$(date -Is) $*" | tee -a "$LOG"; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CEIL=14400; PER_RUN=600; RESERVE=$((100*1024*1024*1024 + 3*1024*1024*1024))
for S in ${SEEDS:-0 1 2}; do for C in ${CONDS:-A B C D E F}; do
  K="s${S}_${C}"
  if grep -q "^$K	" "$TSV"; then log "skip $K (done)"; continue; fi
  n=$(nvidia-smi -i 0 -q -d PIDS | grep -c "Process ID")
  if [ "$n" != "0" ]; then log "STOP: GPU0 has $n process(es) before $K"; exit 3; fi
  avail=$(df --output=avail -B1 / | tail -1)
  log "disk avail $avail before $K"
  if [ "$avail" -lt "$RESERVE" ]; then log "STOP: disk below reserve before $K"; exit 4; fi
  used=$(awk -F'\t' 'NR>1 && ($4 ~ /specaudit/ || $5 ~ /spec_audit/) {s+=$3} END {print s+0}' ledger/gpu_time_ledger.tsv)
  if [ $((used + PER_RUN)) -gt $CEIL ]; then log "STOP: audit GPU seconds $used + $PER_RUN would exceed $CEIL"; exit 5; fi
  log "launch $K (audit GPU s used so far: $used)"
  MOE_GPU_LAUNCH=1 MOE_RESEARCH_APPROVE=gpu scripts/gpu_guarded.sh "specaudit extract $K" spec_audit_extract.py \
    --key "$K" --rows-per-batch 8 --cf-layers 0,1,2,3,4,5,6 --execute > "$OUT/$K.log" 2>&1
  rc=$?
  rd=$(ls -d runs/*_specaudit_v1_extract_${K} 2>/dev/null | tail -1)
  st=$( [ -n "$rd" ] && .venv/bin/python -c "import json,sys;print(json.load(open('$rd/status.json'))['status'])" 2>/dev/null || echo NONE)
  log "$K rc=$rc run=$rd status=$st $(tail -1 "$OUT/$K.log" | cut -c1-200)"
  if [ "$rc" != "0" ] || [ "$st" != "COMPLETE" ]; then log "STOP: $K not COMPLETE"; exit 6; fi
  printf '%s\t%s\n' "$K" "$rd" >> "$TSV"
done; done
log "ALL DONE"
