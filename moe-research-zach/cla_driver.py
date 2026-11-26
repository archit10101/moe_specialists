#!/usr/bin/env python
"""csr-conditional-local-attention-v1 fail-fast, resumable driver (run inside tmux session `clav1`).

Stages (each verified before being skipped; outputs written once; failed/partial runs preserved):
  preflight -> GPU checks -> Gen-1 (seed 98, 5M, CALIB-SEARCH) -> Gen-2 (seeds 98/99, 20M) -> search + code freeze ->
  CALIB-SELECTION (once per frozen checkpoint) -> final freeze -> final training (seeds 0/1/2, 100M) -> final_trained
  -> validation (once per final checkpoint) -> speed benchmarks -> analysis -> protected-inventory verification.
State: .tmp/conditional_local_attention_v1/{state.json, driver.log, runs.tsv, evals.tsv, failures.tsv, logs/}.
Prints exactly one terminal line: CLA V1 COMPLETE | CLA V1 STOPPED: <reason>
Exit codes: 0 complete; 2 stopped (frozen rule / resource / integrity); 10 stopped (implementation needs attention).
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
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from csr import guard  # noqa: E402
from csr import cla_select as sel  # noqa: E402
from csr.cla_run import (ART, CFG_ROOT, FINAL_FROZEN, FINAL_TRAINED, FROZEN_CODE, PARTITION, PREREG,  # noqa: E402
                         SEARCH_FROZEN, ST_DIR, code_hashes, deviation)

ST = ST_DIR
PY = str(ROOT / ".venv" / "bin" / "python")
GPU_CAP_S = 20 * 3600
DISK_MIN = 40 * 1024**3
FOREIGN_WAIT_S, FOREIGN_MAX_S = 300, 7200
STATE = {"stage": "start", "state": "running"}
GEN = {1: ("gen1", (98,), 5_000_000), 2: ("gen2", (98, 99), 20_000_000)}
FINAL_SEEDS = (0, 1, 2)
EST_S_PER_TOKEN = {"FULL-TOKEN-4": 9.5e-6, "default": 1.3e-5}


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


def jread(p):
    return json.loads(Path(p).read_text())


def gpu_seconds():
    s = 0
    for ln in (ROOT / "ledger" / "gpu_time_ledger.tsv").read_text().splitlines():
        p = ln.split("\t")
        if len(p) >= 5 and p[4].startswith("clav1"):
            try:
                s += int(p[2])
            except ValueError:
                pass
    return s


def set_status(stage=None, state=None, **kw):
    if stage:
        STATE["stage"] = stage
    if state:
        STATE["state"] = state
    STATE.update(kw)
    STATE["updated"] = guard.now()
    STATE["gpu_s_used"] = gpu_seconds()
    write_json(ST / "state.json", STATE)


def heartbeat():
    while True:
        try:
            write_json(ST / "heartbeat.json", {"time": guard.now(), "pid": os.getpid(), "stage": STATE.get("stage")})
        except Exception:
            pass
        time.sleep(30)


def write_once(p: Path, obj):
    if p.exists():
        return jread(p)
    guard.atomic_write_json(p, obj, mode=0o400)
    guard.ledger("CLA_V1_ARTIFACT", f"{p.relative_to(ROOT)} sha256={guard.sha256_file(p)}")
    return obj


def gpu_count():
    out = subprocess.run(["nvidia-smi", "-i", "0", "-q", "-d", "PIDS"], capture_output=True, text=True, timeout=60)
    return -1 if out.returncode != 0 else out.stdout.count("Process ID")


def gpu_ready(tag):
    waited = 0
    while True:
        n = gpu_count()
        if n == 0:
            return
        log(f"GPU0 has {n} process(es) before {tag}; waiting {FOREIGN_WAIT_S}s (waited {waited}s)")
        set_status(state="waiting_foreign_gpu", waited_s=waited)
        if waited >= FOREIGN_MAX_S:
            raise Terminal(f"occupied GPU: GPU0 held by a foreign process for more than {FOREIGN_MAX_S}s before {tag}")
        time.sleep(FOREIGN_WAIT_S)
        waited += FOREIGN_WAIT_S


def disk_ok(tag):
    free = guard.disk_free_bytes()
    if free < DISK_MIN:
        raise Terminal(f"insufficient disk space: {free / 1024**3:.1f} GiB free < 40 GiB before {tag}")
    return free


def cap_ok(tag, upcoming_s=0):
    used = gpu_seconds()
    if used + upcoming_s > GPU_CAP_S:
        raise Terminal(f"hard compute limit: used {used}s + upcoming {upcoming_s}s > {GPU_CAP_S}s before {tag}")
    return used


def launch(purpose, args, logname, upcoming_s=0):
    gpu_ready(purpose)
    cap_ok(purpose, upcoming_s)
    free = disk_ok(purpose)
    env = dict(os.environ, MOE_GPU_LAUNCH="1", MOE_RESEARCH_APPROVE="gpu")
    lp = ST / "logs" / logname
    lp.parent.mkdir(exist_ok=True)
    log(f"launch [{purpose}] {' '.join(args)} (disk {free / 1024**3:.0f} GiB, GPU-s {gpu_seconds()})")
    set_status(state="running", current=purpose)
    with open(lp, "a") as f:
        rc = subprocess.run(["scripts/gpu_guarded.sh", purpose] + args, stdout=f, stderr=subprocess.STDOUT,
                            env=env).returncode
    out = lp.read_text().splitlines()
    tag = lambda k: next((x.split(" ", 1)[1].split(" ")[0] for x in reversed(out) if x.startswith(k + " ")), None)  # noqa
    log(f"done [{purpose}] rc={rc} run_dir={tag('RUN_DIR')} eval={tag('EVAL_JSON')}")
    return rc, tag("RUN_DIR"), tag("EVAL_JSON"), out


def cpu(args, logname, approve=None):
    lp = ST / "logs" / logname
    lp.parent.mkdir(exist_ok=True)
    log(f"cpu {' '.join(args)}")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4")
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
        return jread(ROOT / rd / "status.json")["status"]
    except Exception:
        return "MISSING"


def reason_of(rd):
    try:
        return jread(ROOT / rd / "summary.json").get("reason", "")
    except Exception:
        return ""


def search_tokens():
    return jread(PARTITION)["search_valid_tokens"]


def verify_train(rd, cfg_path, need_ckpt):
    rd = ROOT / rd
    if status_of(rd) != "COMPLETE":
        return False
    cfg = jread(cfg_path)
    if jread(rd / "config.json") != cfg:
        return False
    s = jread(rd / "summary.json")
    f = jread(rd / "run_facts.json")
    if s.get("valid_train_tokens", 0) < cfg["train"]["token_budget"] or s.get("optimizer_steps") != f["n_steps"]:
        return False
    if f["config_sha256"] != guard.sha256_file(cfg_path):
        return False
    if not (rd / "calib_search_eval.json").exists():
        return False
    if jread(rd / "calib_search_eval.json")["n_valid_tokens"] != search_tokens():
        return False
    if need_ckpt and guard.sha256_file(rd / "model.pt") != s.get("checkpoint_sha256"):
        return False
    return True


def train_item(key, cfg_path, need_ckpt=False, upcoming_s=0, is_baseline=False):
    """Run (or verify and skip) one training config. Returns (run dir, None) or (None, reason) for a candidate whose
    run is invalid on scientific grounds (non-finite); infrastructure failures are retried / resumed."""
    done = tsv("runs.tsv")
    if key in done:
        rd, st = done[key][0], done[key][1]
        if st == "INVALID":
            return None, done[key][2] if len(done[key]) > 2 else "invalid"
        if verify_train(rd, cfg_path, need_ckpt):
            return rd, None
        raise Attention(f"runs.tsv:{key} recorded but fails verification ({rd})")
    resume_from = None
    for attempt in range(5):
        args = ["cla_train.py", "--config", str(Path(cfg_path).relative_to(ROOT)), "--device", "cuda", "--execute"]
        if resume_from:
            args[-1:-1] = ["--resume", resume_from]
        rc, rd, _, _ = launch(f"clav1 train {key}", args, f"train__{key.replace('/', '_')}.log", upcoming_s)
        st = status_of(rd) if rd else "NO_RUN_DIR"
        if rd is not None and verify_train(rd, cfg_path, need_ckpt):
            tsv_add("runs.tsv", key, rd, "COMPLETE")
            return rd, None
        why = reason_of(rd) if rd else ""
        fail("train", key, rd, f"rc={rc} status={st} reason={why[:300]}")
        if rd:
            tsv_add("runs.tsv", f"{key}#attempt{attempt}", rd, "INCOMPLETE", why[:200])
        if st == "CONTAMINATED":
            deviation("contaminated_run", f"{key}: foreign GPU process during run {rd}; preserved; rerun from scratch")
            resume_from = None
            continue
        if st in ("ABORTED", "INTERRUPTED") and rd and (ROOT / rd / "train_state.pt").exists() and "VRAM" not in why:
            deviation("resume", f"{key}: {st} ({why}); resuming from verified state of {rd}")
            resume_from = rd
            continue
        if st == "FAILED" and ("non-finite" in why or "FloatingPoint" in why):
            if is_baseline:
                raise Terminal(f"baseline non-finite: {key} ({rd})")
            tsv_add("runs.tsv", key, rd, "INVALID", f"non-finite: {why[:200]}")
            log(f"{key}: non-finite -> candidate INVALID (preserved)")
            return None, f"non-finite: {why[:200]}"
        raise Attention(f"{key}: run not COMPLETE/verified (rc={rc} status={st} reason={why[:300]} run={rd})")
    raise Attention(f"{key}: failed repeatedly")


def run_result(rd, reason=None):
    if rd is None:
        return {"complete": False, "finite": False, "collapsed": True, "ce": None, "reason": reason}
    s = jread(ROOT / rd / "summary.json")
    ev = jread(ROOT / rd / "calib_search_eval.json")
    f = jread(ROOT / rd / "run_facts.json")
    return {"complete": True, "finite": True, "ce": ev["ce_nats_per_token"], "collapsed": ev["collapse"]["collapsed"],
            "collapse": ev["collapse"], "train_tok_s": s["train_valid_tokens_per_s_after_warmup"],
            "train_steps_per_s": s["train_steps_per_s_after_warmup"], "gpu_s": s["train_step_seconds"],
            "process_s": s["process_seconds"], "cf_step_s": s.get("cf_step_seconds"), "n_cf_steps": s.get("n_cf_steps"),
            "peak_alloc_gib": s["peak_alloc_gib"], "peak_reserved_gib": s["peak_reserved_gib"],
            "valid_train_tokens": s["valid_train_tokens"], "optimizer_steps": s["optimizer_steps"],
            "global_query_rate": ev.get("global_query_rate_mean"),
            "global_query_rate_by_layer": ev.get("global_query_rate_by_layer"),
            "keys_attended_per_token_by_layer": ev.get("keys_attended_per_token_by_layer"),
            "attention_work_fraction_vs_full": ev.get("attention_work_fraction_vs_full"),
            "ce_by_doc_position": ev.get("ce_by_doc_position"), "init_param_sha256": f["init_param_sha256"],
            "train_rows_sha256": f["train_rows_sha256"], "params": f["params"], "run": rd}


def run_generation(gen, ids):
    stage, seeds, budget = GEN[gen]
    set_status(stage)
    per = {i: {} for i in ids}
    for seed in seeds:
        d = CFG_ROOT / stage / f"seed{seed}"
        for i in ids:
            est = int(budget * EST_S_PER_TOKEN.get(i, EST_S_PER_TOKEN["default"])) + 240
            rd, why = train_item(f"{stage}/seed{seed}/{i}", d / f"{i}.json", need_ckpt=(gen == 2), upcoming_s=est,
                                 is_baseline=(i == sel.BASELINE))
            per[i][seed] = run_result(rd, why)
            set_status(done=f"{stage} seed{seed}/{i}")
    for seed in seeds:  # matched initialisation / data / budget within each seed (fail closed)
        b = per[sel.BASELINE][seed]
        for i in ids:
            r = per[i][seed]
            if r["complete"]:
                for k in ("init_param_sha256", "train_rows_sha256", "valid_train_tokens", "optimizer_steps"):
                    if r[k] != b[k]:
                        raise Terminal(f"{stage} seed{seed} {i}: {k} differs from the baseline (fairness)")
    return per


def stopped(reason, code):
    set_status(state="stopped", stopped=reason)
    log(f"STOPPED: {reason}")
    guard.ledger("CLA_V1_STOPPED", reason)
    print(f"CLA V1 STOPPED: {reason}", flush=True)
    sys.exit(code)


def main():
    ST.mkdir(parents=True, exist_ok=True)
    lockf = open(ST / "driver.lock", "w")
    try:
        fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("another driver holds the lock", flush=True)
        sys.exit(1)
    threading.Thread(target=heartbeat, daemon=True).start()
    set_status("start", "running", stopped=None, pid=os.getpid())
    log(f"driver start pid={os.getpid()} prereg={guard.sha256_file(PREREG)}")
    try:
        run_all()
    except Terminal as e:
        stopped(str(e), 2)
    except Attention as e:
        stopped(f"implementation needs attention: {e}", 10)
    except Exception as e:  # noqa: BLE001
        import traceback
        log(traceback.format_exc())
        stopped(f"driver exception {type(e).__name__}: {e}", 10)
    set_status("complete", "done")
    guard.ledger("CLA_V1_DRIVER_DONE", "all stages complete")
    log("driver: all stages complete")
    print("CLA V1 COMPLETE", flush=True)
    sys.exit(0)


def preflight():
    man = jread(ART / "manifest.json")
    if guard.sha256_file(PREREG) != man["prereg_sha256"]:
        raise Terminal("preregistration hash changed")
    if guard.sha256_file(PARTITION) != man["search_partition_sha256"]:
        raise Terminal("calib partition file changed")
    if guard.sha256_file(ART / "protected_inventory.json") != man["protected_inventory_sha256"]:
        raise Terminal("protected inventory changed")
    tests = sorted(ART.glob("cpu_tests_*.json"))
    if not tests or jread(tests[-1]).get("status") != "PASS":
        raise Attention("CPU test suite has not passed")
    if jread(tests[-1]).get("code_sha256") != code_hashes():
        raise Attention("code changed since the passing CPU test suite")


def gpu_check():
    passes = [p for p in sorted(ART.glob("gpu_checks_2*.json")) if jread(p).get("status") == "PASS"
              and jread(p).get("formal")]
    if passes:
        return passes[-1]
    for attempt in range(2):
        rc, _, _, out = launch("clav1 gpu checks", ["cla_gpu_checks.py", "--formal", "--execute"], "gpu_checks.log",
                               upcoming_s=1200)
        files = sorted(ART.glob("gpu_checks_2*.json"))
        last = jread(files[-1]) if files else {}
        if last.get("status") == "PASS" and last.get("formal"):
            return files[-1]
        if last.get("status") == "CONTAMINATED":
            continue
        break
    raise Terminal(f"GPU correctness checks did not pass (rc={rc}); correctness cannot be established")


def frozen_code_check(tag):
    fz = jread(SEARCH_FROZEN)["code_sha256"]
    now = code_hashes()
    bad = [f for f in FROZEN_CODE if fz.get(f) != now.get(f)]
    if bad:
        raise Terminal(f"frozen code changed after the search freeze ({bad}) before {tag}")


def eval_item(evalset, key, rd):
    done = tsv("evals.tsv")
    k = f"{evalset}/{key}"
    if k in done:
        e = jread(ROOT / done[k][0])
        if e.get("status") == "COMPLETE":
            return done[k][0]
    for attempt in range(3):
        rc, _, ej, _ = launch(f"clav1 eval {evalset} {key}", ["cla_evaluate.py", "--run", rd, "--set", evalset,
                                                              "--device", "cuda", "--execute"],
                              f"eval_{evalset}__{key.replace('/', '_')}.log", upcoming_s=400)
        if ej and jread(ROOT / ej).get("status") == "COMPLETE":
            tsv_add("evals.tsv", k, ej)
            return ej
        st = jread(ROOT / ej).get("status") if ej else "NO_JSON"
        fail(f"eval_{evalset}", key, rd, f"rc={rc} status={st} eval={ej}")
        if st == "CONTAMINATED":
            continue
        raise Attention(f"{key}: {evalset} evaluation failed (rc={rc} status={st})")
    raise Attention(f"{key}: {evalset} evaluation contaminated three times")


def run_all():
    preflight()
    gc = gpu_check()
    log(f"GPU checks PASS: {gc.name}")
    # ---------------- Generation 1
    if not (CFG_ROOT / "gen1").exists():
        if cpu(["cla_make_configs.py", "gen1"], "configs_gen1.log"):
            raise Attention("gen1 config generation failed")
    ids1 = [sel.BASELINE] + [c[0] for c in sel.CANDIDATES]
    per1 = run_generation(1, ids1)
    res1 = {i: per1[i][98] for i in ids1}
    adv1, order1, elim1 = sel.gen1_advance(res1)
    g1 = write_once(ART / "gen1_results.json", {
        "generation": 1, "evalset": "calib_search", "seed": 98, "budget": GEN[1][2], "results": res1,
        "elimination": {k: list(v) for k, v in elim1.items()}, "ranking": order1, "advance": adv1,
        "rule": "cla_select.gen1_advance (prereg section 7)"})
    adv1 = g1["advance"]
    log(f"Gen-1 advance: {adv1}")
    if not adv1:
        write_once(SEARCH_FROZEN, {"candidate_generation_ended": guard.now(), "advanced": [], "config_sha256s": [],
                                   "code_sha256": code_hashes(), "note": "no candidate survived Generation 1"})
        set_status("negative_no_candidates")
        return finish_negative("no candidate survived the preregistered Generation-1 rules")
    # ---------------- Generation 2
    if not (CFG_ROOT / "gen2").exists():
        if cpu(["cla_make_configs.py", "gen2", "--ids", ",".join(adv1)], "configs_gen2.log"):
            raise Attention("gen2 config generation failed")
    ids2 = [sel.BASELINE] + adv1
    per2 = run_generation(2, ids2)
    adv2, order2, md2, tput2, inval2 = sel.gen2_advance(per2, adv1)
    g2 = write_once(ART / "gen2_results.json", {
        "generation": 2, "evalset": "calib_search", "seeds": [98, 99], "budget": GEN[2][2],
        "results": {i: {str(s): r for s, r in per2[i].items()} for i in ids2}, "mean_paired_delta": md2,
        "mean_train_tok_s": tput2, "invalid": inval2, "ranking": order2, "advance": adv2,
        "rule": "cla_select.gen2_advance (prereg section 7)"})
    adv2 = g2["advance"]
    log(f"Gen-2 advance: {adv2}")
    # ---------------- search + code freeze
    set_status("search_freeze")
    shas = [guard.sha256_file(CFG_ROOT / "gen2" / f"seed{s}" / f"{i}.json") for s in (98, 99)
            for i in [sel.BASELINE] + adv2]
    write_once(SEARCH_FROZEN, {"candidate_generation_ended": guard.now(), "advanced": adv2, "config_sha256s": shas,
                               "code_sha256": code_hashes(), "gen1_sha256": guard.sha256_file(ART / "gen1_results.json"),
                               "gen2_sha256": guard.sha256_file(ART / "gen2_results.json")})
    if not adv2:
        return finish_negative("no candidate valid after Generation 2")
    frozen_code_check("CALIB-SELECTION")
    # ---------------- CALIB-SELECTION (once per frozen checkpoint)
    set_status("calib_selection")
    runs = tsv("runs.tsv")
    selp = {}
    for i in [sel.BASELINE] + adv2:
        selp[i] = {}
        for s in (98, 99):
            rd = runs[f"gen2/seed{s}/{i}"][0]
            ej = eval_item("calib_selection", f"gen2/seed{s}/{i}", rd)
            e = jread(ROOT / ej)
            selp[i][s] = {"complete": True, "finite": True, "ce": e["ce_nats_per_token"],
                          "collapsed": e["collapse"]["collapsed"], "global_query_rate": e.get("global_query_rate_mean"),
                          "eval": ej}
    speed = {i: g2["mean_train_tok_s"][i] for i in adv2}
    gqf = {i: sel.global_query_fraction(i, None if sel.PLANS[i]["kind"] != "conditional" else
                                        sum(selp[i][s]["global_query_rate"] for s in (98, 99)) / 2) for i in adv2}
    winner, info = sel.blind_select(selp, adv2, speed, gqf)
    write_once(ART / "calib_selection_results.json", {
        "evalset": "calib_selection (30% of calib documents; evaluated once per frozen Gen-2 checkpoint)",
        "per_seed": {i: {str(s): r for s, r in v.items()} for i, v in selp.items()}, "winner": winner,
        "selection_info": info, "rule": "cla_select.blind_select (prereg section 8)"})
    winner = jread(ART / "calib_selection_results.json")["winner"]
    log(f"CALIB-SELECTION winner: {winner}")
    if winner is None:
        return finish_negative("no valid candidate on CALIB-SELECTION")
    # ---------------- final freeze, training (matched initialisation and data within each seed)
    set_status("final_freeze")
    if not (CFG_ROOT / "final").exists():
        if cpu(["cla_make_configs.py", "final", "--winner", winner], "configs_final.log"):
            raise Attention("final config generation failed")
    fids = [sel.BASELINE, winner]
    shas = [guard.sha256_file(CFG_ROOT / "final" / f"seed{s}" / f"{i}.json") for s in FINAL_SEEDS for i in fids]
    write_once(FINAL_FROZEN, {"frozen": guard.now(), "winner": winner, "ids": fids, "config_sha256s": shas,
                              "code_sha256": code_hashes()})
    frozen_code_check("final training")
    set_status("final_train")
    fr = {}
    for s in FINAL_SEEDS:
        for i in fids:
            est = int(100_000_000 * EST_S_PER_TOKEN.get(i, EST_S_PER_TOKEN["default"])) + 600
            rd, why = train_item(f"final/seed{s}/{i}", CFG_ROOT / "final" / f"seed{s}" / f"{i}.json", need_ckpt=True,
                                 upcoming_s=est, is_baseline=(i == sel.BASELINE))
            if rd is None:
                raise Terminal(f"final run invalid ({i} seed {s}: {why}); final comparison cannot be completed")
            fr[f"seed{s}/{i}"] = rd
            set_status(final_done=len(fr))
    per = {i: {s: run_result(fr[f"seed{s}/{i}"]) for s in FINAL_SEEDS} for i in fids}
    for s in FINAL_SEEDS:
        for k in ("init_param_sha256", "train_rows_sha256", "valid_train_tokens", "optimizer_steps"):
            if per[winner][s][k] != per[sel.BASELINE][s][k]:
                raise Terminal(f"final seed {s}: {k} differs between baseline and winner (fairness)")
    write_once(FINAL_TRAINED, {"time": guard.now(), "runs": fr,
                               "checkpoints": {k: jread(ROOT / v / "summary.json")["checkpoint_sha256"]
                                               for k, v in fr.items()}})
    # ---------------- validation, once per final checkpoint
    set_status("final_validation")
    frozen_code_check("validation")
    for k, rd in fr.items():
        eval_item("val", f"final/{k}", rd)
    # ---------------- speed benchmarks (accuracy frozen)
    set_status("bench")
    if not [p for p in ART.glob("bench_2*.json") if jread(p).get("status") == "COMPLETE"]:
        for attempt in range(2):
            rc, _, _, _ = launch("clav1 bench", ["cla_bench.py", "--runs", f"{sel.BASELINE}={fr['seed0/' + sel.BASELINE]}",
                                                 f"{winner}={fr['seed0/' + winner]}", "--execute"], "bench.log",
                                 upcoming_s=5400)
            if [p for p in ART.glob("bench_2*.json") if jread(p).get("status") == "COMPLETE"]:
                break
        else:
            fail("bench", "final", None, f"rc={rc}")
            deviation("bench_failed", "final benchmark failed twice; analysis reports speed as missing")
    analysis_and_verify()


def finish_negative(why):
    log(f"NEGATIVE path: {why}")
    deviation("negative_path", why)
    analysis_and_verify()


def analysis_and_verify():
    set_status("analysis")
    if cpu(["cla_analyze.py", "--execute"], "analyze.log", approve="cpu-model"):
        raise Attention("final analysis failed")
    set_status("verify")
    if cpu(["cla_inventory.py", "--verify", str((ART / "protected_inventory.json").relative_to(ROOT))], "verify.log"):
        raise Terminal("corrupted or mismatched protected artifacts (see verify log)")
    write_json(ART / f"final_audit_{guard.utc_stamp()}.json", {"gpu0_processes": gpu_count(),
                                                               "disk_free_bytes": guard.disk_free_bytes(),
                                                               "gpu_seconds_used": gpu_seconds(), "time": guard.now()})


if __name__ == "__main__":
    main()
