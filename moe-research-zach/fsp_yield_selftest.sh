#!/usr/bin/env bash
# fspv2 Operational Amendment 1 (B): driver-level external-cancellation / GPU-yield self-test. Uses a separate driver
# state file (FSP_DRIVER_STATE) and ~150-step FORECAST-SPECIALIST selftest runs; never touches the experiment state.
#  T1 SIGTERM to the driver mid-run -> child stops cooperatively with checkpoint, EXTERNALLY_STOPPED, driver exits
#  T2 restart without --authorized-resume -> refused; with it -> same run dir resumes from its checkpoint -> COMPLETE
#  T3 foreign GPU process appears mid-run -> stop + checkpoint + yield (foreign process untouched)
#  T4 foreign GPU process present before launch -> nothing launched, yield
#  T5 STOP marker mid-run -> stop + checkpoint; driver then refuses to start while the marker exists
# The "foreign" process is a tiny test fixture (64 MiB CUDA tensor) started outside the guard on purpose.
set -u
cd /home/zapan4/moe-research
source env.sh
ART=artifacts/forecast-specialist-pretraining-v2
OUT=.tmp/fspv2/yield_selftest_$1
mkdir -p "$OUT"
PY=.venv/bin/python
run_dir_of() { ls -d runs/*_fsp_selftest$1 2>/dev/null | tail -1; }
wait_step() {  # wait until the run's metrics show step >= $2 (log_every 50)
  for i in $(seq 1 120); do d=$(run_dir_of "$1"); [ -n "$d" ] && grep -q "\"step\": $2," "$d/metrics.jsonl" 2>/dev/null && return 0; sleep 2; done; return 1; }
fixture() { CUDA_VISIBLE_DEVICES=0 $PY -c "import torch,time;x=torch.zeros(16*1024*1024,device='cuda');time.sleep($1)" > /dev/null 2>&1 & echo $!; }
drv() { FSP_DRIVER_STATE=$OUT/state_$1.json FSP_SELFTEST_TAG=$1 $PY fsp_driver.py run --only selftest "${@:2}"; }

T=$1t1
ts() { date +%T; }
echo "== T1 $(ts)"; (drv $T > $OUT/T1.out 2>&1) & D=$!
sleep 1; DP=$(pgrep -f "FSP_SELFTEST_TAG=$T" | head -1)
wait_step $T 50; DPID=$(pgrep -f "fsp_driver.py run --only selftest" | head -1); echo "driver pid $DPID"; kill -TERM "$DPID"
wait $D; echo "T1 driver rc=$?"
echo "== T2a (no authorization)"; drv $T > $OUT/T2a.out 2>&1; echo "T2a rc=$?"
echo "== T2b (authorized)"; drv $T --authorized-resume > $OUT/T2b.out 2>&1; echo "T2b rc=$?"

T=$1t3
echo "== T3"; (drv $T > $OUT/T3.out 2>&1) & D=$!
wait_step $T 50; FX=$(fixture 600); echo "fixture pid $FX $(ts)"
wait $D; echo "T3 driver rc=$?"; kill -0 $FX 2>/dev/null && echo "fixture still alive (untouched)"
nvidia-smi --query-compute-apps=pid --format=csv,noheader; echo "== T4 (fixture still on GPU) $(ts)"; drv $1t4 > $OUT/T4.out 2>&1; echo "T4 rc=$?"
echo "T4 done $(ts)"; kill $FX 2>/dev/null; wait $FX 2>/dev/null; sleep 3

T=$1t5
echo "== T5"; (drv $T > $OUT/T5.out 2>&1) & D=$!
wait_step $T 50; touch $ART/STOP
wait $D; echo "T5 driver rc=$?"
drv $T --authorized-resume > $OUT/T5b.out 2>&1; echo "T5b (marker present, even with flag) rc=$?"
rm -f $ART/STOP
nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l
echo DONE
