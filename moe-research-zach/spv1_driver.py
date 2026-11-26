#!/usr/bin/env python
"""csr-speculative-pool-routing-v1 fail-fast, resumable driver (run inside tmux session `spv1`).

Stages (each verified before being skipped; outputs written once; failed/partial runs preserved):
  preflight -> GPU checks (target) -> draft config/training -> draft GPU checks + freeze -> proposal cache ->
  gen1 -> gen2 rule/configs/runs -> gen3 rule/configs/runs -> search freeze -> blind CALIB-SELECTION ->
  feasibility benchmark + budget gate -> final freeze/configs -> final training -> validation (once per checkpoint)
  -> final benchmarks -> analysis -> protected-inventory verification -> GPU-idle and disk audit.
State: .tmp/speculative_pool_v1/{status.json, heartbeat.json, driver.log, *_runs.tsv, failures.tsv, logs/}.
Prints exactly one terminal line: SPECULATIVE POOL ROUTING V1 COMPLETE | SPECULATIVE POOL ROUTING V1 STOPPED: <reason>
Exit codes: 0 complete; 2 stopped (frozen gate / resource / integrity); 10 stopped (implementation needs attention).
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
from csr import spv1_select as sel  # noqa: E402
from csr.spv1_run import (ART, CACHE_DIR, CFG_ROOT, DRAFT_FROZEN, FINAL_FROZEN, PARTITION, PREREG,  # noqa: E402
                          SEARCH_FROZEN)

ST = ROOT / ".tmp" / "speculative_pool_v1"
PY = str(ROOT / ".venv" / "bin" / "python")
GPU_CAP_S = 20 * 3600
DISK_MIN = 80 * 1024**3
VRAM_MAX_GIB = 56.0
FOREIGN_WAIT_S, FOREIGN_MAX_S = 300, 7200
STATE = {"stage": "start", "state": "running"}
GEN_SEEDS = {1: (98,), 2: (98, 99), 3: (97, 98, 99)}
GEN_BUDGET = {1: 5_000_000, 2: 20_000_000, 3: 50_000_000}
MAIN_SEEDS = (0, 1, 2)
EST_S_PER_TOKEN = 1.6e-5


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


def gpu_seconds():
    s = 0
    for ln in (ROOT / "ledger" / "gpu_time_ledger.tsv").read_text().splitlines()[1:]:
        p = ln.split("\t")
        if len(p) >= 5 and p[4].startswith("spv1"):
            s += int(p[2])
    return s


def set_status(stage=None, state=None, **kw):
    if stage:
        STATE["stage"] = stage
    if state:
        STATE["state"] = state
    STATE.update(kw)
    STATE["updated"] = guard.now()
    STATE["gpu_s_used"] = gpu_seconds()
    write_json(ST / "status.json", STATE)


def heartbeat():
    while True:
        try:
            write_json(ST / "heartbeat.json", {"time": guard.now(), "pid": os.getpid(), "stage": STATE.get("stage"),
                                               "state": STATE.get("state")})
        except Exception:
            pass
        time.sleep(30)


def write_once(p: Path, obj):
    if p.exists():
        return json.loads(p.read_text())
    guard.atomic_write_json(p, obj, mode=0o400)
    guard.ledger("SP_V1_ARTIFACT", f"{p.relative_to(ROOT)} sha256={guard.sha256_file(p)}")
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
    log(f"disk free {free / 1024**3:.1f} GiB before {tag}")
    if free < DISK_MIN:
        raise Terminal(f"insufficient disk space: {free / 1024**3:.1f} GiB free < 80 GiB before {tag}")


def cap_ok(tag, upcoming_s=0):
    used = gpu_seconds()
    if used + upcoming_s > GPU_CAP_S:
        raise Terminal(f"hard compute limit: used {used}s + upcoming {upcoming_s}s > {GPU_CAP_S}s before {tag}")
    return used


def launch(purpose, args, logname, upcoming_s=0):
    gpu_ready(purpose)
    cap_ok(purpose, upcoming_s)
    disk_ok(purpose)
    env = dict(os.environ, MOE_GPU_LAUNCH="1", MOE_RESEARCH_APPROVE="gpu")
    lp = ST / "logs" / logname
    lp.parent.mkdir(exist_ok=True)
    log(f"launch [{purpose}] {' '.join(args)} -> {lp.relative_to(ROOT)}")
    set_status(state="running", current=purpose)
    with open(lp, "a") as f:
        rc = subprocess.run(["scripts/gpu_guarded.sh", purpose] + args, stdout=f, stderr=subprocess.STDOUT,
                            env=env).returncode
    out = lp.read_text().splitlines()
    tag = lambda k: next((x.split(" ", 1)[1] for x in reversed(out) if x.startswith(k + " ")), None)  # noqa: E731
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


def jread(p):
    return json.loads(Path(p).read_text())


def status_of(rd):
    try:
        return jread(ROOT / rd / "status.json")["status"]
    except Exception:
        return "MISSING"


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
    if s.get("valid_train_tokens", 0) < cfg["train"]["token_budget"]:
        return False
    if not (rd / "calib_search_eval.json").exists():
        return False
    if jread(rd / "calib_search_eval.json")["n_valid_tokens"] != search_tokens():
        return False
    if need_ckpt and guard.sha256_file(rd / "model.pt") != s.get("checkpoint_sha256"):
        return False
    return True


def reason_of(rd):
    try:
        return jread(ROOT / rd / "summary.json").get("reason", "")
    except Exception:
        return ""


def train_item(tsvname, key, cfg_path, need_ckpt=False, upcoming_s=0, allow_invalid=True):
    """Run (or verify and skip) one training config. Returns the run dir, or None for a candidate that stayed
    non-finite after the one preregistered numerical retry (recorded INVALID; preserved)."""
    done = tsv(tsvname)
    if key in done:
        rd, st = done[key][0], done[key][1] if len(done[key]) > 1 else "COMPLETE"
        if st == "INVALID":
            return None
        if verify_train(rd, cfg_path, need_ckpt):
            return rd
        raise Attention(f"{tsvname}:{key} recorded but fails verification ({rd})")
    resume_from, numeric_fail = None, 0
    for attempt in range(5):
        args = ["spv1_train.py", "--config", str(Path(cfg_path).relative_to(ROOT)), "--device", "cuda", "--execute"]
        if resume_from:
            args[-1:-1] = ["--resume", resume_from]
        rc, rd, _, _ = launch(f"spv1 train {key}", args, f"{tsvname.split('.')[0]}__{key.replace('/', '_')}.log",
                              upcoming_s)
        st = status_of(rd) if rd else "NO_RUN_DIR"
        if rd is not None and verify_train(rd, cfg_path, need_ckpt):
            tsv_add(tsvname, key, rd, "COMPLETE")
            return rd
        why = reason_of(rd) if rd else ""
        fail("train", key, rd, f"rc={rc} status={st} reason={why[:300]}")
        if st == "CONTAMINATED":
            log(f"{key}: CONTAMINATED by a foreign GPU process; preserved; retrying from scratch")
            resume_from = None
            continue
        if st in ("ABORTED", "INTERRUPTED") and rd and (ROOT / rd / "train_state.pt").exists() \
                and "VRAM" not in why:
            log(f"{key}: {st} ({why}); resuming from {rd}")
            resume_from = rd
            continue
        if st == "FAILED" and "non-finite" in why:
            numeric_fail += 1
            if numeric_fail >= 2:
                if allow_invalid:
                    tsv_add(tsvname, key, rd, "INVALID")
                    log(f"{key}: non-finite after the preregistered retry -> candidate INVALID (preserved)")
                    return None
                raise Terminal(f"non-finite training persists after one preregistered retry: {key} ({rd})")
            log(f"{key}: non-finite; one preregistered numerical retry (fresh run, identical config)")
            resume_from = None
            continue
        raise Attention(f"{key}: run not COMPLETE/verified (rc={rc} status={st} reason={why[:300]} run={rd})")
    raise Attention(f"{key}: failed repeatedly")


def mean_or_none(x):
    import numpy as np
    return float(np.mean(x)) if x is not None else None


def run_result(rd):
    if rd is None:
        return {"complete": False, "ce": None, "collapsed": True}
    s = jread(ROOT / rd / "summary.json")
    ev = jread(ROOT / rd / "calib_search_eval.json")
    f = jread(ROOT / rd / "run_facts.json")
    d8 = ev.get("distinct_experts_per_window", {}).get("8")
    return {"complete": True, "ce": ev["ce_nats_per_token"], "collapsed": ev["collapse"]["collapsed"],
            "collapse": ev["collapse"], "extra_param_fraction": f["extra_param_fraction"],
            "extra_params": f["extra_params"], "gpu_s": s["process_seconds"],
            "train_tok_s": s["train_valid_tokens_per_s_after_warmup"], "peak_reserved_gib": s["peak_reserved_gib"],
            "train_flops": s.get("train_flops"), "valid_train_tokens": s["valid_train_tokens"],
            "refresh_rate": mean_or_none(ev.get("refresh_rate")), "distinct8": mean_or_none(d8),
            "pool_recall": mean_or_none(ev.get("pool_recall_vs_unrestricted_top4")),
            "contained": mean_or_none(ev.get("unrestricted_top4_contained_frac")),
            "adjacent_jaccard": mean_or_none(ev.get("adjacent_expert_jaccard")),
            "teacher_forced_spec": ev.get("teacher_forced_spec"), "run": rd,
            "init_backbone_param_sha256": f["init_backbone_param_sha256"]}


def aggregate(per_seed: dict):
    import numpy as np
    vals = list(per_seed.values())
    comp = all(v["complete"] for v in vals)
    agg = {"complete": comp, "per_seed": per_seed, "n_seeds": len(vals)}
    if comp:
        mean = lambda k: float(np.mean([v[k] for v in vals])) if all(v.get(k) is not None for v in vals) else None  # noqa
        agg.update({"ce": mean("ce"), "collapsed": any(v["collapsed"] for v in vals),
                    "extra_param_fraction": max(v["extra_param_fraction"] for v in vals),
                    "gpu_s": mean("gpu_s"), "train_tok_s": mean("train_tok_s"), "refresh_rate": mean("refresh_rate"),
                    "distinct8": mean("distinct8"), "pool_recall": mean("pool_recall"),
                    "contained": mean("contained"), "train_flops": mean("train_flops")})
    else:
        agg.update({"ce": None, "collapsed": True})
    return agg


def run_generation(gen, specs):
    stage = {1: "gen1", 2: "gen2", 3: "gen3"}[gen]
    set_status(stage)
    ids = ["TOKEN-4"] + [s["id"] for s in specs]
    per = {i: {} for i in ids}
    need_ckpt = gen == 3
    for seed in GEN_SEEDS[gen]:
        d = CFG_ROOT / stage / f"seed{seed}"
        for i in ids:
            rd = train_item(f"{stage}_runs.tsv", f"seed{seed}/{i}", d / f"{i}.json", need_ckpt=need_ckpt,
                            upcoming_s=int(GEN_BUDGET[gen] * EST_S_PER_TOKEN), allow_invalid=(i != "TOKEN-4"))
            if rd is None and i == "TOKEN-4":
                raise Attention("TOKEN-4 failed")
            per[i][seed] = run_result(rd)
            set_status(done=f"{stage} seed{seed}/{i}")
    for seed in GEN_SEEDS[gen]:
        tok = per["TOKEN-4"][seed]["init_backbone_param_sha256"]
        for i in ids:
            r = per[i][seed]
            if r["complete"] and r["init_backbone_param_sha256"] != tok:
                raise Attention(f"{stage} seed{seed} {i}: backbone initialization differs from TOKEN-4")
    return {i: aggregate(per[i]) for i in ids}


def candidates_tsv_lines(gen, specs, agg, rejected=None):
    lines = []
    tok = agg["TOKEN-4"]["ce"]
    for s in specs:
        a = agg[s["id"]]
        rej = (rejected or {}).get(s["id"], (False, ""))
        valid = sel.is_valid(a) and not rej[0]
        lines.append("\t".join(map(str, [
            s["id"], gen, s["family"], s.get("parent") or "-", s.get("same_as") or "-",
            json.dumps(s["routing"], sort_keys=True), s.get("desc") or s.get("change", ""),
            "valid" if valid else f"REJECTED: {rej[1] or 'invalid (failed / collapse / parameters)'}",
            f"{a['ce']:.5f}" if a["ce"] is not None else "-",
            f"{a['ce'] - tok:+.5f}" if a["ce"] is not None else "-",
            f"{a.get('refresh_rate'):.4f}" if a.get("refresh_rate") is not None else "-",
            f"{a.get('pool_recall'):.4f}" if a.get("pool_recall") is not None else "-"])))
    return lines


def stopped(reason, code):
    set_status(state="stopped", stopped=reason)
    log(f"STOPPED: {reason}")
    guard.ledger("SP_V1_STOPPED", reason)
    print(f"SPECULATIVE POOL ROUTING V1 STOPPED: {reason}", flush=True)
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
    guard.ledger("SP_V1_DRIVER_DONE", "all stages complete")
    log("driver: all stages complete")
    print("SPECULATIVE POOL ROUTING V1 COMPLETE", flush=True)
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


def gpu_check(phase, extra=()):
    pat = f"gpu_checks_{phase}_2*.json"
    passes = [p for p in sorted(ART.glob(pat)) if jread(p).get("status") == "PASS"]
    if passes:
        return passes[-1]
    for attempt in range(2):
        rc, _, _, out = launch(f"spv1 gpu checks {phase}", ["spv1_gpu_checks.py", "--phase", phase, *extra,
                                                            "--device", "cuda", "--execute"], f"gpu_checks_{phase}.log")
        files = sorted(ART.glob(pat))
        if files and jread(files[-1]).get("status") == "PASS":
            return files[-1]
        if files and jread(files[-1]).get("status") == "CONTAMINATED":
            continue
        break
    raise Attention(f"GPU correctness checks ({phase}) did not pass (rc={rc}); causality/implementation must be "
                    f"diagnosed ({files[-1].name if files else 'no result file'})")


def draft_stage():
    set_status("draft")
    if DRAFT_FROZEN.exists():
        rec = jread(DRAFT_FROZEN)
        if guard.sha256_file(ROOT / rec["checkpoint"]) != rec["checkpoint_sha256"]:
            raise Terminal("frozen draft checkpoint changed")
        return rec
    dcfg = CFG_ROOT / "draft" / "draft_d128_l4.json"
    if not dcfg.exists():
        if cpu(["spv1_make_configs.py", "draft"], "configs_draft.log"):
            raise Attention("draft config generation failed")
    rd = train_item("draft_runs.tsv", "draft", dcfg, need_ckpt=True, upcoming_s=1500, allow_invalid=False)
    ev = jread(ROOT / rd / "calib_search_eval.json")
    import math
    if not math.isfinite(ev["ce_nats_per_token"]):
        raise Attention("draft CALIB-SEARCH CE not finite")
    gc = gpu_check("draft", ("--draft-run", rd))
    rec = {"checkpoint": f"{rd}/model.pt", "checkpoint_sha256": guard.sha256_file(ROOT / rd / "model.pt"),
           "run": rd, "config": str(dcfg.relative_to(ROOT)), "config_sha256": guard.sha256_file(dcfg),
           "calib_search_ce": ev["ce_nats_per_token"], "calib_search_top1": ev["top1_accuracy"],
           "gpu_checks": str(gc.relative_to(ROOT)), "frozen": guard.now(),
           "provenance": "trained by spv1_train.py on the existing train split only (seed 0 order, 100M tokens); "
                         "selected/frozen on CALIB-SEARCH only, before any target candidate was trained"}
    return write_once(DRAFT_FROZEN, rec)


def cache_stage():
    set_status("proposal_cache")
    man_p = CACHE_DIR / "train_proposals_manifest.json"
    if not man_p.exists():
        rc, _, _, out = launch("spv1 draft cache", ["spv1_draft_cache.py", "--device", "cuda", "--execute"],
                               "draft_cache.log", upcoming_s=3000)
        if not man_p.exists():
            raise Attention(f"proposal cache generation failed (rc={rc})")
    man = jread(man_p)
    if guard.sha256_file(ROOT / man["file"]) != man["sha256"]:
        raise Terminal("proposal cache hash mismatch")
    if man["draft_checkpoint_sha256"] != jread(DRAFT_FROZEN)["checkpoint_sha256"]:
        raise Terminal("proposal cache not produced by the frozen draft")
    return man


def run_all():
    preflight()
    gpu_check("target")
    draft_stage()
    cache_stage()
    # ---------------- Generation 1
    if not (CFG_ROOT / "gen1").exists():
        if cpu(["spv1_make_configs.py", "gen1"], "configs_gen1.log"):
            raise Attention("gen1 config generation failed")
    g1 = run_generation(1, sel.GEN1)
    g1 = write_once(ART / "gen1_results.json", {"generation": 1, "evalset": "calib_search", "seeds": [98],
                                                "budget": GEN_BUDGET[1], "results": g1})["results"]
    # ---------------- Generation 2
    set_status("gen2_rule")
    g2_specs = write_once(ART / "gen2_specs.json", {"rule": "spv1_select.gen2_candidates (prereg §8)",
                                                    "specs": sel.gen2_candidates(g1)})["specs"]
    if not g2_specs:
        raise Terminal("no valid candidate in any family after Generation 1")
    if not (CFG_ROOT / "gen2").exists():
        if cpu(["spv1_make_configs.py", "gen", "--gen", "2", "--specs",
                str((ART / "gen2_specs.json").relative_to(ROOT))], "configs_gen2.log"):
            raise Attention("gen2 config generation failed")
    g2 = run_generation(2, g2_specs)
    tokp = g2["TOKEN-4"]["per_seed"]
    rej = {s["id"]: sel.gen2_reject(g2[s["id"]]["per_seed"], tokp) for s in g2_specs}
    g2 = write_once(ART / "gen2_results.json", {"generation": 2, "evalset": "calib_search", "seeds": [98, 99],
                                                "budget": GEN_BUDGET[2], "results": g2,
                                                "rejections": {k: list(v) for k, v in rej.items()}})
    rej = {k: tuple(v) for k, v in g2["rejections"].items()}
    g2 = g2["results"]
    # ---------------- Generation 3
    set_status("gen3_rule")
    g3_specs = write_once(ART / "gen3_specs.json", {"rule": "spv1_select.gen3_finalists (prereg §8)",
                                                    "specs": sel.gen3_finalists(g2, g2_specs, rej)})["specs"]
    if not g3_specs:
        raise Terminal("no valid candidate in any family after Generation 2")
    if not (CFG_ROOT / "gen3").exists():
        if cpu(["spv1_make_configs.py", "gen", "--gen", "3", "--specs",
                str((ART / "gen3_specs.json").relative_to(ROOT))], "configs_gen3.log"):
            raise Attention("gen3 config generation failed")
    g3 = run_generation(3, g3_specs)
    g3 = write_once(ART / "gen3_results.json", {"generation": 3, "evalset": "calib_search", "seeds": [97, 98, 99],
                                                "budget": GEN_BUDGET[3], "results": g3})["results"]
    # ---------------- search freeze (candidate generation ends permanently)
    set_status("search_freeze")
    runs3 = tsv("gen3_runs.tsv")
    if not SEARCH_FROZEN.exists():
        fz = sel.freeze_finalists(g3, g3_specs)
        frozen = {"TOKEN-4": {"id": "TOKEN-4", "family": "token", "routing": sel.TOKEN}}
        for role in ("causal", "spec"):
            if fz[role] is not None:
                frozen[role] = fz[role]
        controls = {f"RC-{role}": sel.random_control(frozen[role], role) for role in ("causal", "spec")
                    if role in frozen}
        sha = [guard.sha256_file(CFG_ROOT / "gen3" / f"seed{seed}" / f"{s['id']}.json")
               for s in frozen.values() for seed in GEN_SEEDS[3]]
        lines = ["candidate\tgeneration\tfamily\tparent\tsame_as\trouting\tdescription\tvalidity\tcalib_search_ce"
                 "\tdelta_vs_TOKEN4_same_gen\trefresh_rate\tpool_recall"]
        lines += candidates_tsv_lines(1, sel.GEN1, g1)
        lines += candidates_tsv_lines(2, g2_specs, g2, rej)
        lines += candidates_tsv_lines(3, g3_specs, g3)
        guard.atomic_write_bytes(ART / "candidates.tsv", ("\n".join(lines) + "\n").encode(), mode=0o400)
        uniq = {sel.routing_key(s["routing"]) for s in sel.GEN1 + g2_specs + g3_specs}
        write_once(ART / "calib_search_results.json", {
            "evalset": "calib_search (70% of calib documents; development set; NOT final evidence)",
            "gen1": g1, "gen2": g2, "gen2_rejections": {k: list(v) for k, v in rej.items()}, "gen3": g3,
            "gen2_specs": g2_specs, "gen3_specs": g3_specs, "unique_candidates": len(uniq),
            "max_unique": sel.MAX_UNIQUE, "frozen": frozen, "random_controls": controls})
        if len(uniq) > sel.MAX_UNIQUE:
            raise Terminal("candidate limit exceeded")
        write_once(SEARCH_FROZEN, {"candidate_generation_ended": guard.now(), "frozen": frozen,
                                   "random_controls": controls, "config_sha256s": sha,
                                   "calib_search_results_sha256": guard.sha256_file(ART / "calib_search_results.json")})
    sf = jread(SEARCH_FROZEN)
    frozen, controls = sf["frozen"], sf["random_controls"]
    roles = [r for r in ("causal", "spec") if r in frozen]
    if not roles:
        raise Terminal("no valid causal or speculative finalist after Generation 3")
    # ---------------- blind CALIB-SELECTION (one evaluation per checkpoint)
    set_status("calib_selection")
    sel_per = {}
    for k, s in frozen.items():
        sel_per[k] = {}
        for seed in GEN_SEEDS[3]:
            rd = runs3[f"seed{seed}/{s['id']}"][0]
            ej = eval_item("calib_selection", f"seed{seed}/{s['id']}", rd)
            e = jread(ROOT / ej)
            f = jread(ROOT / rd / "run_facts.json")
            sel_per[k][seed] = {"complete": True, "ce": e["ce_nats_per_token"],
                                "collapsed": e["collapse"]["collapsed"],
                                "extra_param_fraction": f["extra_param_fraction"],
                                "train_tok_s": jread(ROOT / rd / "summary.json")["train_valid_tokens_per_s_after_warmup"],
                                "refresh_rate": mean_or_none(e.get("refresh_rate")),
                                "distinct8": mean_or_none(e.get("distinct_experts_per_window", {}).get("8")),
                                "eval": ej}
    sel_agg = {k: aggregate(v) for k, v in sel_per.items()}
    bsel = sel.blind_selection({"TOKEN-4": sel_agg["TOKEN-4"], "causal": sel_agg.get("causal"),
                                "spec": sel_agg.get("spec")})
    csr = write_once(ART / "calib_selection_results.json", {
        "evalset": "calib_selection (30% of calib documents; evaluated once, after candidate generation ended)",
        "frozen": frozen, "results": sel_agg, "blind_selection": bsel,
        "search_vs_selection": {k: {"calib_search_gen3_ce": g3[frozen[k]["id"]]["ce"],
                                    "calib_selection_ce": sel_agg[k]["ce"]} for k in frozen}})
    roles = [r for r in roles if csr["blind_selection"]["finalists"][r]["proceeds"]]
    log(f"blind selection: {csr['blind_selection']}")
    if not roles:
        raise Terminal("no finalist proceeds after blind CALIB-SELECTION (invalid)")
    # ---------------- feasibility benchmark and compute gate
    set_status("feasibility")
    feas = [p for p in sorted(ART.glob("bench_feasibility_*.json")) if jread(p).get("status") == "COMPLETE"]
    if not feas:
        rl = [f"TOKEN-4={runs3['seed97/TOKEN-4'][0]}"] + [f"{r}={runs3['seed97/' + frozen[r]['id']][0]}" for r in roles]
        rc, _, _, _ = launch("spv1 bench feasibility", ["spv1_bench.py", "--mode", "feasibility", "--runs", *rl,
                                                        "--device", "cuda", "--execute"], "bench_feasibility.log",
                             upcoming_s=1200)
        feas = [p for p in sorted(ART.glob("bench_feasibility_*.json")) if jread(p).get("status") == "COMPLETE"]
        if not feas:
            raise Attention(f"feasibility benchmark failed (rc={rc})")
    fb = jread(feas[-1])
    set_status("budget_gate")
    spt, peaks = {}, []
    for k in ["TOKEN-4"] + roles:
        sid = frozen[k]["id"]
        rs = [run_result(runs3[f"seed{seed}/{sid}"][0]) for seed in GEN_SEEDS[3]]
        spt[k] = max(r["gpu_s"] for r in rs) / min(r["valid_train_tokens"] for r in rs)
        peaks += [r["peak_reserved_gib"] for r in rs]
    for r in roles:
        spt[f"RC-{r}"] = spt[r]
    for lab, c in fb["results"]["conditions"].items():
        peaks.append(c.get("decode_peak_reserved_gib", 0.0))
    overhead = 15 * 120.0 + 3600.0 + 1800.0  # validation evals + final benchmarks + slack
    used = gpu_seconds()
    budget, cbudget, rows = sel.choose_budget(used, spt, ["TOKEN-4"] + roles, [f"RC-{r}" for r in roles],
                                              overhead_s=overhead)
    bud = write_once(ART / "budget.json", {"sec_per_token": spt, "rows": rows, "budget": budget,
                                           "controls_budget": cbudget, "used_gpu_s": used,
                                           "peak_reserved_gib": max(peaks), "roles": roles,
                                           "feasibility_bench": str(feas[-1].relative_to(ROOT)),
                                           "disk_free_bytes": guard.disk_free_bytes(), "overhead_s": overhead})
    budget, cbudget = bud["budget"], bud["controls_budget"]
    if max(peaks) >= VRAM_MAX_GIB:
        raise Terminal(f"peak VRAM {max(peaks):.1f} GiB >= {VRAM_MAX_GIB} GiB")
    if budget is None:
        raise Terminal("hard compute limit: even 25M tokens x 3 seeds does not fit in 20 GPU-hours with a 20% margin")
    log(f"budget {budget} controls {cbudget}: {rows}")
    # ---------------- final freeze and configs (before any final training / validation access)
    set_status("final_freeze")
    fspecs = []
    for r in roles:
        fspecs.append({"id": frozen[r]["id"], "family": frozen[r]["family"], "routing": frozen[r]["routing"],
                       "role": "finalist", "desc": f"frozen best {r} pool candidate"})
    for r in roles:
        c = controls[f"RC-{r}"]
        fspecs.append({**c, "role": "random_control"})
    write_once(ART / "final_specs.json", {"specs": fspecs, "budget": budget, "controls_budget": cbudget})
    fdir = CFG_ROOT / f"final_B{budget // 1_000_000}M"
    if not fdir.exists():
        if cpu(["spv1_make_configs.py", "final", "--specs", str((ART / "final_specs.json").relative_to(ROOT)),
                "--budget", str(budget), "--controls-budget", str(cbudget)], "configs_final.log"):
            raise Attention("final config generation failed")
    fids = ["TOKEN-4"] + [s["id"] for s in fspecs]
    rolemap = {"TOKEN-4": "TOKEN-4", **{r: frozen[r]["id"] for r in roles}, **{f"RC-{r}": f"RC-{r}" for r in roles}}
    if not FINAL_FROZEN.exists():
        shas = [guard.sha256_file(fdir / f"seed{s}" / f"{i}.json") for s in MAIN_SEEDS for i in fids]
        write_once(FINAL_FROZEN, {"frozen": guard.now(), "budget": budget, "controls_budget": cbudget, "ids": fids,
                                  "roles": rolemap, "config_sha256s": shas, "config_dir": str(fdir.relative_to(ROOT))})
    # ---------------- final training, then validation exactly once per checkpoint
    set_status("final_train")
    fruns = {}
    for seed in MAIN_SEEDS:
        for i in fids:
            cfgp = fdir / f"seed{seed}" / f"{i}.json"
            B = jread(cfgp)["train"]["token_budget"]
            sk = next((k for k, v in rolemap.items() if v == i), "TOKEN-4")
            rd = train_item("final_runs.tsv", f"seed{seed}/{i}", cfgp, need_ckpt=True,
                            upcoming_s=int(B * spt.get(sk, EST_S_PER_TOKEN)), allow_invalid=False)
            fruns[f"seed{seed}/{i}"] = rd
            set_status(final_done=len(fruns), final_total=3 * len(fids))
    set_status("final_validation")
    for key, rd in fruns.items():
        eval_item("val", key, rd)
    # ---------------- final benchmarks
    set_status("final_bench")
    if not [p for p in ART.glob("bench_final_*.json") if jread(p).get("status") == "COMPLETE"]:
        rl = [f"{lab}={fruns['seed0/' + i]}" for lab, i in rolemap.items()]
        for attempt in range(2):
            rc, _, _, _ = launch("spv1 bench final", ["spv1_bench.py", "--mode", "final", "--runs", *rl,
                                                      "--device", "cuda", "--execute"], "bench_final.log",
                                 upcoming_s=2400)
            if [p for p in ART.glob("bench_final_*.json") if jread(p).get("status") == "COMPLETE"]:
                break
        else:
            fail("bench", "final", None, f"rc={rc}")
            log("final benchmark failed twice; continuing to analysis (reported as missing)")
    # ---------------- analysis and verification
    set_status("analysis")
    if not (ART / "final_analysis.json").exists():
        if cpu(["spv1_analyze.py", "--execute"], "analyze.log", approve="cpu-model"):
            raise Attention("final analysis failed")
    set_status("verify")
    if cpu(["spv1_inventory.py", "--verify", str((ART / "protected_inventory.json").relative_to(ROOT))], "verify.log"):
        raise Terminal("corrupted or mismatched protected artifacts (see verify log)")
    n = gpu_count()
    free = guard.disk_free_bytes()
    write_once(ART / f"final_audit_{guard.utc_stamp()}.json", {"gpu0_processes": n, "disk_free_bytes": free,
                                                                "gpu_seconds_used": gpu_seconds(),
                                                                "time": guard.now()})
    log(f"final audit: GPU0 processes={n} disk free={free / 1024**3:.1f} GiB GPU-s={gpu_seconds()}")


def eval_item(evalset, key, rd):
    reg = f"{evalset}_evals.tsv"
    done = tsv(reg)
    if key in done:
        e = jread(ROOT / done[key][0])
        if e.get("status") == "COMPLETE":
            return done[key][0]
    for attempt in range(3):
        rc, _, ej, _ = launch(f"spv1 eval {evalset} {key}", ["spv1_evaluate.py", "--run", rd, "--set", evalset,
                                                             "--device", "cuda", "--execute"],
                              f"eval_{evalset}__{key.replace('/', '_')}.log", upcoming_s=300)
        if ej and jread(ROOT / ej).get("status") == "COMPLETE":
            tsv_add(reg, key, ej)
            return ej
        st = jread(ROOT / ej).get("status") if ej else "NO_JSON"
        fail(f"eval_{evalset}", key, rd, f"rc={rc} status={st} eval={ej}")
        if st == "CONTAMINATED":
            continue
        raise Attention(f"{key}: {evalset} evaluation failed (rc={rc} status={st})")
    raise Attention(f"{key}: {evalset} evaluation contaminated three times")


if __name__ == "__main__":
    main()
