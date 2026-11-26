#!/usr/bin/env python
"""csr-multiscale-routing-v1 fail-fast, resumable driver (run inside tmux session `msv1`).

Stages (each verified before being skipped; outputs are never overwritten; failed/partial runs are preserved):
  gpu_checks -> semantic_gate -> calibration runs (40) -> calibration decision -> branch configs -> benchmark ->
  feasibility/budget (+ main configs) -> pilot runs -> pilot futility -> main runs (+ full validation) -> analysis.
State: .tmp/multiscale_v1/{status.json, heartbeat.json, driver.log, *_runs.tsv, failures.tsv}.
Exit codes: 0 done; 2 terminal blocker (status.json "terminal"); 10 needs attention (implementation/run failure).
"""
import fcntl
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path("/home/zapan4/moe-research")
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)
from csr import guard  # noqa: E402
from csr.multiscale import BRANCH_CONDITIONS  # noqa: E402

ST = ROOT / ".tmp" / "multiscale_v1"
ART = ROOT / "artifacts" / "multiscale-routing-v1"
CFG = ROOT / "configs" / "multiscale-routing-v1"
PY = str(ROOT / ".venv" / "bin" / "python")
CALIB_TOKENS, VAL_TOKENS, VAL_DOCS = 5_270_143, 10_732_175, 16_850
GPU_CAP_S = 20 * 3600
DISK_MIN = 80 * 1024**3
FOREIGN_WAIT_S, FOREIGN_MAX_S = 300, 7200
STATE = {"stage": "start", "state": "running"}


class Terminal(Exception):
    pass


class Attention(Exception):
    pass


def log(msg):
    line = f"{guard.now()} {msg}"
    print(line, flush=True)
    with open(ST / "driver.log", "a") as f:
        f.write(line + "\n")


def write_json(p, obj):
    tmp = p.with_name(f".{p.name}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, p)


def set_status(stage=None, state=None, **kw):
    if stage:
        STATE["stage"] = stage
    if state:
        STATE["state"] = state
    STATE.update(kw)
    STATE["updated"] = guard.now()
    write_json(ST / "status.json", STATE)


def heartbeat():
    while True:
        try:
            write_json(ST / "heartbeat.json", {"time": guard.now(), "pid": os.getpid(), "stage": STATE.get("stage"),
                                               "state": STATE.get("state")})
        except Exception:
            pass
        time.sleep(30)


def gpu_count():
    out = subprocess.run(["nvidia-smi", "-i", "0", "-q", "-d", "PIDS"], capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        return -1
    return out.stdout.count("Process ID")


def msv1_gpu_seconds():
    s = 0
    for ln in (ROOT / "ledger" / "gpu_time_ledger.tsv").read_text().splitlines()[1:]:
        p = ln.split("\t")
        if len(p) >= 5 and (p[4].startswith("msv1") or "msv1" in p[3]):
            s += int(p[2])
    return s


def gpu_ready(tag):
    waited = 0
    while True:
        n = gpu_count()
        if n == 0:
            return
        log(f"GPU0 has {n} process(es) before {tag}; waiting {FOREIGN_WAIT_S}s (waited {waited}s)")
        set_status(state="waiting_foreign_gpu", waited_s=waited)
        if waited >= FOREIGN_MAX_S:
            raise Terminal(f"FOREIGN_GPU: GPU0 occupied for more than {FOREIGN_MAX_S}s before {tag}")
        time.sleep(FOREIGN_WAIT_S)
        waited += FOREIGN_WAIT_S


def disk_ok(tag):
    free = guard.disk_free_bytes()
    log(f"disk free {free / 1024**3:.1f} GiB before {tag}")
    if free < DISK_MIN:
        raise Terminal(f"DISK: {free} bytes free < 80 GiB before {tag}")


def cap_ok(tag, upcoming_s=0):
    used = msv1_gpu_seconds()
    if used + upcoming_s > GPU_CAP_S:
        raise Terminal(f"GPU_CAP: used {used}s + upcoming {upcoming_s}s > {GPU_CAP_S}s before {tag}")
    return used


def launch(purpose, args, logname):
    gpu_ready(purpose)
    cap_ok(purpose)
    env = dict(os.environ, MOE_GPU_LAUNCH="1", MOE_RESEARCH_APPROVE="gpu")
    lp = ST / "logs" / logname
    lp.parent.mkdir(exist_ok=True)
    log(f"launch [{purpose}] {' '.join(args)} -> {lp.relative_to(ROOT)}")
    set_status(state="running", current=purpose)
    with open(lp, "a") as f:
        rc = subprocess.run(["scripts/gpu_guarded.sh", purpose] + args, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
    out = lp.read_text().splitlines()
    run_dir = next((x.split(" ", 1)[1] for x in reversed(out) if x.startswith("RUN_DIR ")), None)
    ev = next((x.split(" ", 1)[1] for x in reversed(out) if x.startswith("EVAL_JSON ")), None)
    log(f"done [{purpose}] rc={rc} run_dir={run_dir} eval={ev}")
    return rc, run_dir, ev


def cpu(args, logname, approve=None):
    lp = ST / "logs" / logname
    lp.parent.mkdir(exist_ok=True)
    log(f"cpu {' '.join(args)}")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1")
    if approve:
        env["MOE_RESEARCH_APPROVE"] = approve
    with open(lp, "a") as f:
        rc = subprocess.run([PY] + args, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
    log(f"cpu rc={rc} ({lp.relative_to(ROOT)})")
    return rc


def tsv(name):
    p = ST / name
    if not p.exists():
        return {}
    return {ln.split("\t")[0]: ln.split("\t")[1:] for ln in p.read_text().splitlines() if ln.strip()}


def tsv_add(name, key, *vals):
    with open(ST / name, "a") as f:
        f.write("\t".join([key, *map(str, vals)]) + "\n")


def fail(kind, key, run_dir, detail):
    tsv_add("failures.tsv", f"{guard.now()}|{kind}|{key}", run_dir or "-", detail)


def status_of(rd):
    try:
        return json.loads((ROOT / rd / "status.json").read_text())["status"]
    except Exception:
        return "MISSING"


def verify_train(rd, cfg_path, need_calib, need_ckpt):
    rd = ROOT / rd
    if status_of(rd) != "COMPLETE":
        return False
    cfg = json.loads(Path(cfg_path).read_text())
    if json.loads((rd / "config.json").read_text()) != cfg:
        return False
    s = json.loads((rd / "summary.json").read_text())
    if s.get("valid_train_tokens", 0) < cfg["train"]["token_budget"]:
        return False
    if need_calib:
        if not (rd / "calib_eval.json").exists():
            return False
        ce = json.loads((rd / "calib_eval.json").read_text())
        if ce["n_valid_tokens"] != CALIB_TOKENS:
            return False
    if need_ckpt and guard.sha256_file(rd / "model.pt") != s.get("checkpoint_sha256"):
        return False
    return True


def verify_eval(ej, rd):
    try:
        e = json.loads((ROOT / ej).read_text())
    except Exception:
        return False
    v = list(e.get("doc_n", {}).values())
    return (e.get("status") == "COMPLETE" and e.get("split") == "val" and e.get("n_valid_tokens") == VAL_TOKENS
            and len(v) == VAL_DOCS and sum(v) == VAL_TOKENS
            and e.get("checkpoint_sha256") == guard.sha256_file(ROOT / rd / "model.pt"))


def train_item(tsvname, key, cfg_path, mode="train", max_steps=None, need_calib=False, need_ckpt=False):
    """Run (or verify and skip) one training config. Returns the run dir."""
    done = tsv(tsvname)
    if key in done:
        rd = done[key][0]
        ok = status_of(rd) == "COMPLETE" if mode == "benchmark" else verify_train(rd, cfg_path, need_calib, need_ckpt)
        if ok:
            return rd
        raise Attention(f"{tsvname}:{key} recorded but fails verification ({rd})")
    for attempt in range(3):
        disk_ok(key)
        args = ["ms_train.py", "--config", str(Path(cfg_path).relative_to(ROOT)), "--device", "cuda", "--mode", mode,
                "--execute"]
        if max_steps:
            args[-1:-1] = ["--max-steps", str(max_steps)]
        rc, rd, _ = launch(f"msv1 {mode} {key}", args, f"{tsvname.split('.')[0]}__{key.replace('/', '_')}.log")
        st = status_of(rd) if rd else "NO_RUN_DIR"
        ok = rd is not None and (st == "COMPLETE" if mode == "benchmark" else verify_train(rd, cfg_path, need_calib, need_ckpt))
        if ok:
            tsv_add(tsvname, key, rd)
            return rd
        fail(mode, key, rd, f"rc={rc} status={st}")
        if st == "CONTAMINATED":
            log(f"{key}: CONTAMINATED by a foreign GPU process; preserved; retrying after the GPU is free")
            continue
        raise Attention(f"{key}: run not COMPLETE/verified (rc={rc} status={st} run={rd})")
    raise Attention(f"{key}: contaminated three times")


def eval_item(key, rd):
    done = tsv("main_evals.tsv")
    if key in done and verify_eval(done[key][0], rd):
        return done[key][0]
    for attempt in range(3):
        rc, _, ej = launch(f"msv1 eval {key}", ["ms_evaluate.py", "--run", rd, "--split", "val", "--device", "cuda",
                                                "--execute"], f"eval__{key.replace('/', '_')}.log")
        if ej and verify_eval(ej, rd):
            tsv_add("main_evals.tsv", key, ej)
            return ej
        st = json.loads((ROOT / ej).read_text()).get("status") if ej else "NO_JSON"
        fail("eval", key, rd, f"rc={rc} status={st} eval={ej}")
        if st == "CONTAMINATED":
            continue
        raise Attention(f"{key}: val evaluation failed (rc={rc} status={st})")
    raise Attention(f"{key}: eval contaminated three times")


def latest_pass(prefix, key):
    for p in sorted(ART.glob(f"{prefix}_*.json"), reverse=True):
        d = json.loads(p.read_text())
        if d.get("status") == "COMPLETE" and d.get(key):
            return p
    return None


def gate(prefix, key, script, args=()):
    p = latest_pass(prefix, key)
    if p:
        log(f"{prefix}: verified pass ({p.name})")
        return p
    rc, _, _ = launch(f"msv1 {prefix}", [script, *args, "--execute"], f"{prefix}.log")
    p = latest_pass(prefix, key)
    if p is None:
        raise Attention(f"{prefix} did not pass (rc={rc}); see newest {prefix}_*.json")
    return p


def main():
    ST.mkdir(parents=True, exist_ok=True)
    lockf = open(ST / "driver.lock", "w")
    try:
        fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("another driver holds the lock", flush=True)
        sys.exit(1)
    threading.Thread(target=heartbeat, daemon=True).start()
    set_status("start", "running", terminal=None, pid=os.getpid())
    log(f"driver start pid={os.getpid()} prereg={guard.sha256_file(ROOT / 'docs/MULTISCALE_ROUTING_V1_PREREG.md')}")
    try:
        run_all()
        set_status("complete", "done", terminal=None)
        log("driver: all stages complete")
        sys.exit(0)
    except Terminal as e:
        set_status(state="terminal", terminal=str(e))
        log(f"TERMINAL: {e}")
        guard.ledger("MS_V1_TERMINAL", str(e))
        sys.exit(2)
    except Attention as e:
        set_status(state="needs_attention", attention=str(e))
        log(f"ATTENTION: {e}")
        guard.ledger("MS_V1_ATTENTION", str(e))
        sys.exit(10)


def run_all():
    pre = json.loads((ART / "manifest.json").read_text())["prereg_sha256"]
    if guard.sha256_file(ROOT / "docs/MULTISCALE_ROUTING_V1_PREREG.md") != pre:
        raise Terminal("PREREG hash changed")
    # 1. GPU correctness checks and the semantic-causality gate
    set_status("gpu_checks")
    gate("gpu_checks", "all_pass", "ms_gpu_checks.py")
    set_status("semantic_gate")
    p = latest_pass("semantic_gate", "gate_pass")
    if p is None:
        rc, _, _ = launch("msv1 semantic_gate", ["ms_semantic_gate.py", "--execute"], "semantic_gate.log")
        p = latest_pass("semantic_gate", "gate_pass")
        if p is None:
            newest = sorted(ART.glob("semantic_gate_*.json"))[-1:]
            d = json.loads(newest[0].read_text()) if newest else {}
            if d.get("status") == "COMPLETE":  # the gate ran and failed: terminal per prereg (after diagnosis)
                raise Attention(f"SEMANTIC GATE FAILED ({newest[0].name}); verify it is not an implementation defect")
            raise Attention(f"semantic gate did not complete (rc={rc})")
    log(f"semantic gate passed: {p.name}")
    # 2. calibration (seed 98 then 99; beta cells then alpha cells)
    set_status("calibration")
    for seed in (98, 99):
        d = CFG / "calibration" / f"seed{seed}"
        names = sorted(n for n in json.loads((d / "INDEX.json").read_text())["configs"])
        names = [n for n in names if n.startswith("beta_")] + [n for n in names if n.startswith("alpha_")]
        for n in names:
            train_item("calibration_runs.tsv", f"seed{seed}/{n}", d / f"{n}.json", need_calib=True)
    set_status("calibration_decision")
    if not (ART / "calibration.json").exists():
        if cpu(["ms_decide.py", "calibration", "--runs", str(ST / "calibration_runs.tsv")], "decide_calibration.log"):
            raise Attention("calibration decision failed")
    cal = json.loads((ART / "calibration.json").read_text())
    alpha, beta, branch = cal["alpha"], cal["beta"], cal["branch"]
    log(f"calibration: alpha={alpha} beta={beta} branch={branch}")
    set_status(alpha=alpha, beta=beta, branch=branch)
    if branch == "A":
        raise Terminal("BRANCH_A: both alpha and beta failed neutral calibration")
    conds = BRANCH_CONDITIONS[branch]
    # 3. benchmark + budget
    if not (CFG / "benchmark").exists():
        if cpu(["ms_make_configs.py", "branch", "--alpha", str(alpha), "--beta", str(beta)], "configs_branch.log"):
            raise Attention("branch config generation failed")
    set_status("benchmark")
    for c in conds:
        train_item("bench_runs.tsv", c, CFG / "benchmark" / f"{c}.json", mode="benchmark", max_steps=20)
    set_status("feasibility")
    if not (ART / "feasibility.json").exists():
        if cpu(["ms_decide.py", "feasibility", "--runs", str(ST / "bench_runs.tsv")], "decide_feasibility.log"):
            raise Attention("feasibility decision failed")
    fea = json.loads((ART / "feasibility.json").read_text())
    budget = fea["budget"]
    set_status(budget=budget)
    if budget is None:
        raise Terminal("NO_BUDGET: no uniform replicated main budget fits under 20 GPU-hours")
    mdir = CFG / f"main_B{budget // 1_000_000}M"
    if not mdir.exists():
        if cpu(["ms_make_configs.py", "main", "--alpha", str(alpha), "--beta", str(beta), "--budget", str(budget)],
               "configs_main.log"):
            raise Attention("main config generation failed")
    # 4. pilot
    set_status("pilot")
    for c in conds:
        train_item("pilot_runs.tsv", c, CFG / "pilot" / f"{c}.json", need_calib=True)
    if not (ART / "pilot.json").exists():
        rc = cpu(["ms_decide.py", "pilot", "--runs", str(ST / "pilot_runs.tsv")], "decide_pilot.log")
        if rc not in (0, 3):
            raise Attention("pilot decision failed")
    pil = json.loads((ART / "pilot.json").read_text())
    if pil["futility_stop"]:
        raise Terminal("FUTILITY: frozen pilot futility rule met")
    # 5. main
    set_status("main")
    per_run = {c: budget / fea["benchmark"][c]["tokens_per_s"] + fea["overhead_s"] + fea["val_eval_s_projected"]
               for c in conds}
    for s in (0, 1, 2):
        for c in conds:
            key = f"seed{s}/{c}"
            cap_ok(key, upcoming_s=int(per_run[c]))
            rd = train_item("main_runs.tsv", key, mdir / f"seed{s}" / f"{c}.json", need_ckpt=True)
            eval_item(key, rd)
            set_status(main_done=len(tsv("main_evals.tsv")), main_total=3 * len(conds),
                       gpu_s_used=msv1_gpu_seconds())
    log("main study complete")
    analysis_stages(conds)


ANALYSIS_ALLOWANCE_S = 3 * 3600


def analysis_seconds():
    return sum(int(ln.split("\t")[2]) for ln in (ROOT / "ledger" / "gpu_time_ledger.tsv").read_text().splitlines()[1:]
               if len(ln.split("\t")) >= 5 and ln.split("\t")[4] in ("msv1 diag", "msv1 spec_extract"))


def analysis_stages(conds):
    """Secondary analyses (prereg section 12): router diagnostics + latency, specialization extraction, CPU analyses."""
    set_status("analysis_diag")
    if latest_pass("diag", "per_checkpoint") is None:
        rc, _, _ = launch("msv1 diag", ["ms_diag.py", "--runs", str(ST / "main_runs.tsv"), "--execute"], "diag.log")
        if latest_pass("diag", "per_checkpoint") is None:
            raise Attention(f"diag failed (rc={rc})")
    set_status("analysis_spec_extract")
    runs = tsv("main_runs.tsv")
    done = tsv("spec_runs.tsv")
    per_run_est = 260
    for s in (0, 1, 2):
        for c in conds:
            key = f"seed{s}/{c}"
            if key in done and status_of(done[key][0]) == "COMPLETE":
                continue
            todo = sum(1 for ss in (0, 1, 2) for cc in conds if f"seed{ss}/{cc}" not in done)
            if analysis_seconds() + per_run_est > ANALYSIS_ALLOWANCE_S:
                raise Attention(f"analysis allowance would be exceeded ({analysis_seconds()}s used, {todo} extractions left)")
            rd = runs[key][0]
            strat = json.loads((ROOT / rd / "config.json").read_text())["segmentation"]["strategy"]
            rc, xd, _ = launch(f"msv1 spec_extract {key}", ["ms_spec_extract.py", "--key", key, "--run", rd,
                                                            "--cf-layers", "0,3,6", "--execute"],
                               f"spec_extract__{key.replace('/', '_')}.log")
            if xd is None or status_of(xd) != "COMPLETE":
                fail("spec_extract", key, xd, f"rc={rc}")
                raise Attention(f"spec extraction failed for {key} (rc={rc}, {xd})")
            tsv_add("spec_runs.tsv", key, xd, strat)
            done = tsv("spec_runs.tsv")
    set_status("analysis_spec_cpu")
    if not (ART / "spec" / "spec_analysis.json").exists():
        if cpu(["ms_spec_analyze.py", "--extract-runs", str(ST / "spec_runs.tsv"), "--workers", "4", "--execute"],
               "spec_analyze.log", approve="cpu-model"):
            raise Attention("spec analysis failed")
    set_status("analysis_final")
    if not (ART / "analysis.json").exists():
        if cpu(["ms_analyze.py", "--execute"], "analyze.log", approve="cpu-model"):
            raise Attention("final analysis failed")
    log("analysis complete")


if __name__ == "__main__":
    main()
