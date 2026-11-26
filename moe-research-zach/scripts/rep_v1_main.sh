#!/usr/bin/env bash
# csr-representative-v1 main runs: seeds 0,1,2 x conditions A..F, sequential, one guarded GPU process at a time.
# Per run: matched-field check, disk + GPU0 occupancy checks, train (100M tokens), status/checkpoint/config gate,
# full-validation evaluation, coverage gate. Stops at the first failed gate. Never deletes or overwrites.
set -u
umask 077
cd /home/zapan4/moe-research || exit 2
OUT=.tmp/rep_v1_main
mkdir -p "$OUT"
LOG=$OUT/driver.log
RES=$OUT/runs.tsv
RESERVE=107374182400
NEED=1000000000
say(){ echo "$(date -Is) $*" | tee -a "$LOG"; }
gpu_free(){ n=$(nvidia-smi -i 0 -q -d PIDS | grep -c "Process ID"); say "GPU0 procs before $1: $n"; [ "$n" = 0 ]; }
disk_ok(){ a=$(df -B1 --output=avail / | tail -1); say "disk avail $a"; [ "$a" -ge $((RESERVE + NEED)) ]; }
G="env MOE_GPU_LAUNCH=1 MOE_RESEARCH_APPROVE=gpu scripts/gpu_guarded.sh"
PY=.venv/bin/python
VAL_DOCS=$($PY -I -c "import json;print(json.load(open('artifacts/data/fwe-rep-v1/manifest.json'))['splits']['val']['n_docs'])")
VAL_TOK=$($PY -I -c "import json;print(json.load(open('artifacts/data/fwe-rep-v1/manifest.json'))['splits']['val']['valid_pred_tokens'])")
say "start; val docs=$VAL_DOCS tokens=$VAL_TOK"
for S in ${SEEDS:-0 1 2}; do
  B=configs/rep-v1/seed$S/base.json
  for C in ${CONDS:-A B C D E F}; do
    if grep -q "^$S	$C	" "$RES" 2>/dev/null; then say "skip s$S $C (already complete in $RES)"; continue; fi
    disk_ok || { say "STOP: disk below reserve before s$S $C"; exit 10; }
    (source ./env.sh && $PY -I - "$S" "$C" <<'EOF'
import json, sys
sys.path.insert(0, "src")
from csr.config import CONDITIONS, build_condition_config, check_matched
s, c = sys.argv[1], sys.argv[2]
d = f"configs/rep-v1/seed{s}"
base = json.load(open(f"{d}/base.json"))
cfgs = {k: json.load(open(f"{d}/condition_{k}.json")) for k in CONDITIONS}
check_matched(cfgs)
want = build_condition_config(base, c, run_name=f"rep_v1_s{s}_{c}")
assert want == cfgs[c], "condition file differs from base + condition"
assert base["init"]["seed"] == base["train"]["seed"] == int(s) and base["data"]["train_order_seed"] == 17 + int(s)
assert all(cfgs[k]["semantic"][x] is not None for k in cfgs for x in ("lambda", "tau_s", "tau_r"))
print("matched OK", s, c)
EOF
    ) >> "$LOG" 2>&1 || { say "STOP: matched-field check failed s$S $C"; exit 16; }
    gpu_free "train s$S $C" || { say "STOP: GPU0 occupied"; exit 11; }
    $G "rep-v1 train s$S $C" train.py --base "$B" --condition $C --device cuda --run-name rep_v1_s${S}_$C --execute \
       > "$OUT/train_s${S}_$C.log" 2>&1
    say "train s$S $C launcher exit $?"
    R=$(ls -d runs/*_train_rep_v1_s${S}_$C 2>/dev/null | tail -1)
    [ -n "$R" ] || { say "STOP: no run dir s$S $C"; exit 12; }
    chk=$($PY -I - "$R" "$S" "$C" <<'EOF'
import json, sys, hashlib
r, s, c = sys.argv[1:4]
st = json.load(open(f"{r}/status.json"))["status"]
sm = json.load(open(f"{r}/summary.json"))
cfg = json.load(open(f"{r}/config.json"))
want = json.load(open(f"configs/rep-v1/seed{s}/condition_{c}.json"))
h = hashlib.sha256(open(f"{r}/model.pt", "rb").read()).hexdigest() if st == "COMPLETE" else None
ok = st == "COMPLETE" and h == sm.get("checkpoint_sha256") and cfg == want and sm["valid_train_tokens"] >= 100_000_000
print(("OK" if ok else "BAD"), st, sm.get("valid_train_tokens"), sm.get("peak_reserved_gib"), sm.get("reason", ""))
EOF
)
    say "train s$S $C $R $chk"
    case "$chk" in OK*) ;; *) say "STOP: train gate failed s$S $C"; exit 13;; esac
    gpu_free "eval s$S $C" || { say "STOP: GPU0 occupied"; exit 11; }
    $G "rep-v1 eval s$S $C val" evaluate.py --run "$R" --split val --device cuda --execute > "$OUT/eval_s${S}_$C.log" 2>&1
    say "eval s$S $C launcher exit $?"
    E=$(ls "$R"/eval_val_*.json 2>/dev/null | tail -1)
    [ -n "$E" ] || { say "STOP: no eval json s$S $C"; exit 14; }
    chk=$($PY -I - "$E" "$VAL_DOCS" "$VAL_TOK" <<'EOF'
import json, math, sys
e = json.load(open(sys.argv[1])); nd, nt = int(sys.argv[2]), int(sys.argv[3])
v = list(e.get("doc_n", {}).values())
ok = e["status"] == "COMPLETE" and e["split"] == "val" and e["n_valid_tokens"] == nt and len(v) == nd and sum(v) == nt \
     and math.isfinite(e["ce_nats_per_token"])
print("OK" if ok else "BAD", e["status"], e.get("n_valid_tokens"), len(v), e.get("ce_nats_per_token"))
EOF
)
    say "eval s$S $C $E $chk"
    case "$chk" in OK*) ;; *) say "STOP: eval gate failed s$S $C"; exit 15;; esac
    printf '%s\t%s\t%s\t%s\n' "$S" "$C" "$R" "$E" >> "$RES"
  done
done
disk_ok || say "WARN: disk below reserve+margin after completion"
say "ALL DONE"
