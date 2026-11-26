#!/usr/bin/env python
"""csr-sequence-routing-search-v1 fail-fast, resumable driver (run inside tmux session `srsv1`).

Stages (each verified before being skipped; outputs never overwritten; failed/partial runs preserved):
  preflight -> gpu_checks -> gen1 runs -> gen2 rule+configs -> gen2 runs -> gen3 rule+configs -> gen3 runs ->
  search freeze -> CALIB-SELECTION evals + promotion -> throughput/budget gate -> final freeze+configs ->
  final runs -> validation (once per checkpoint) -> latency -> analysis -> protected-inventory verification.
State: .tmp/sequence_routing_v1/{status.json, heartbeat.json, driver.log, *_runs.tsv, failures.tsv}.
Exit codes: 0 done; 2 terminal (frozen scientific/resource gate); 10 needs attention (implementation/run failure).
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
from csr import srs_select as sel  # noqa: E402
from csr.srs_run import ART, CFG_ROOT, FINAL_FROZEN, PREREG, SEARCH_FROZEN  # noqa: E402

ST = ROOT / ".tmp" / "sequence_routing_v1"
PY = str(ROOT / ".venv" / "bin" / "python")
GPU_CAP_S = 20 * 3600
DISK_MIN = 80 * 1024**3
VRAM_MAX_GIB = 56.0
FOREIGN_WAIT_S, FOREIGN_MAX_S = 300, 7200
STATE = {"stage": "start", "state": "running"}
GEN_SEEDS = {1: (98,), 2: (98, 99), 3: (97, 98, 99)}
MAIN_SEEDS = (0, 1, 2)


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
    STATE["gpu_s_used"] = srs_gpu_seconds()
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
    """Artifacts are written once; an existing file is returned unchanged (resume) and never overwritten."""
    if p.exists():
        return json.loads(p.read_text())
    guard.atomic_write_json(p, obj, mode=0o400)
    guard.ledger("SRS_V1_ARTIFACT", f"{p.relative_to(ROOT)} sha256={guard.sha256_file(p)}")
    return obj


def gpu_count():
    out = subprocess.run(["nvidia-smi", "-i", "0", "-q", "-d", "PIDS"], capture_output=True, text=True, timeout=60)
    return -1 if out.returncode != 0 else out.stdout.count("Process ID")


def srs_gpu_seconds():
    s = 0
    for ln in (ROOT / "ledger" / "gpu_time_ledger.tsv").read_text().splitlines()[1:]:
        p = ln.split("\t")
        if len(p) >= 5 and p[4].startswith("srsv1"):
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
    used = srs_gpu_seconds()
    if used + upcoming_s > GPU_CAP_S:
        raise Terminal(f"GPU_CAP: used {used}s + upcoming {upcoming_s}s > {GPU_CAP_S}s before {tag}")
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
    return jread(ART / "search_partition.json")["search_valid_tokens"]


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
    """Run (or verify and skip) one training config. Returns the run dir, or None for a candidate that failed
    numerically twice (recorded as INVALID; preserved)."""
    done = tsv(tsvname)
    if key in done:
        rd, st = done[key][0], done[key][1] if len(done[key]) > 1 else "COMPLETE"
        if st == "INVALID":
            return None
        if verify_train(rd, cfg_path, need_ckpt):
            return rd
        raise Attention(f"{tsvname}:{key} recorded but fails verification ({rd})")
    resume_from, numeric_fail = None, 0
    for attempt in range(4):
        args = ["srs_train.py", "--config", str(Path(cfg_path).relative_to(ROOT)), "--device", "cuda", "--execute"]
        if resume_from:
            args[-1:-1] = ["--resume", resume_from]
        rc, rd, _, _ = launch(f"srsv1 train {key}", args, f"{tsvname.split('.')[0]}__{key.replace('/', '_')}.log",
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
        if st in ("ABORTED", "INTERRUPTED") and rd and (ROOT / rd / "train_state.pt").exists():
            log(f"{key}: {st} ({why}); resuming from {rd}")
            resume_from = rd
            continue
        if st == "FAILED" and ("non-finite" in why) and allow_invalid:
            numeric_fail += 1
            if numeric_fail >= 2:
                tsv_add(tsvname, key, rd, "INVALID")
                log(f"{key}: non-finite twice -> candidate INVALID (preserved, reported)")
                return None
            continue
        raise Attention(f"{key}: run not COMPLETE/verified (rc={rc} status={st} reason={why[:300]} run={rd})")
    raise Attention(f"{key}: failed repeatedly")


def run_result(rd):
    """Per-run CALIB-SEARCH record used by the selection rules."""
    if rd is None:
        return {"complete": False, "ce": None, "collapsed": True}
    s = jread(ROOT / rd / "summary.json")
    ev = jread(ROOT / rd / "calib_search_eval.json")
    f = jread(ROOT / rd / "run_facts.json")
    return {"complete": True, "ce": ev["ce_nats_per_token"], "collapsed": ev["collapse"]["collapsed"],
            "collapse": ev["collapse"], "extra_param_fraction": f["extra_router_param_fraction"],
            "extra_params": f["extra_router_params"], "gpu_s": s["process_seconds"],
            "train_tok_s": s["train_valid_tokens_per_s_after_warmup"], "peak_reserved_gib": s["peak_reserved_gib"],
            "train_flops": s["train_flops"], "valid_train_tokens": s["valid_train_tokens"],
            "mean_segment_length": ev.get("mean_segment_length"), "run": rd,
            "init_backbone_param_sha256": f["init_backbone_param_sha256"]}


def aggregate(per_seed: dict):
    """per_seed: {seed: run_result}. Mean over seeds; complete only if every seed is complete."""
    vals = list(per_seed.values())
    comp = all(v["complete"] for v in vals)
    agg = {"complete": comp, "per_seed": per_seed, "n_seeds": len(vals)}
    if comp:
        import numpy as np
        agg.update({"ce": float(np.mean([v["ce"] for v in vals])), "collapsed": any(v["collapsed"] for v in vals),
                    "extra_param_fraction": max(v["extra_param_fraction"] for v in vals),
                    "gpu_s": float(np.mean([v["gpu_s"] for v in vals])),
                    "train_flops": float(np.mean([v["train_flops"] for v in vals])) if all("train_flops" in v for v in vals)
                    else None})
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
                            allow_invalid=(i != "TOKEN-4"))
            if rd is None and i == "TOKEN-4":
                raise Attention("TOKEN-4 failed")
            per[i][seed] = run_result(rd)
            set_status(done=f"{stage} seed{seed}/{i}")
    # init matching within seed: candidate backbone init == TOKEN-4 backbone init
    for seed in GEN_SEEDS[gen]:
        tok = per["TOKEN-4"][seed]["init_backbone_param_sha256"]
        for i in ids:
            r = per[i][seed]
            if r["complete"] and r["init_backbone_param_sha256"] != tok:
                raise Attention(f"{stage} seed{seed} {i}: backbone initialization differs from TOKEN-4")
    agg = {i: aggregate(per[i]) for i in ids}
    return agg


def candidates_tsv_lines(gen, specs, agg):
    lines = []
    tok = agg["TOKEN-4"]["ce"]
    for s in specs:
        a = agg[s["id"]]
        lines.append("\t".join(map(str, [
            s["id"], gen, s["family"], s.get("parent") or "-", s.get("same_as") or "-",
            sel.routing_key(s["routing"]), json.dumps(s["routing"], sort_keys=True),
            s.get("desc") or s.get("change", ""), "valid" if sel.is_valid(a) else "INVALID",
            f"{a['ce']:.5f}" if a["ce"] is not None else "-",
            f"{a['ce'] - tok:+.5f}" if a["ce"] is not None else "-"])))
    return lines


def main():
    ST.mkdir(parents=True, exist_ok=True)
    lockf = open(ST / "driver.lock", "w")
    try:
        fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("another driver holds the lock", flush=True)
        sys.exit(1)
    threading.Thread(target=heartbeat, daemon=True).start()
    set_status("start", "running", terminal=None, attention=None, pid=os.getpid())
    log(f"driver start pid={os.getpid()} prereg={guard.sha256_file(PREREG)}")
    try:
        run_all()
        set_status("complete", "done")
        log("driver: all stages complete")
        guard.ledger("SRS_V1_DRIVER_DONE", "all stages complete")
        sys.exit(0)
    except Terminal as e:
        set_status(state="terminal", terminal=str(e))
        log(f"TERMINAL: {e}")
        guard.ledger("SRS_V1_TERMINAL", str(e))
        sys.exit(2)
    except Attention as e:
        set_status(state="needs_attention", attention=str(e))
        log(f"ATTENTION: {e}")
        guard.ledger("SRS_V1_ATTENTION", str(e))
        sys.exit(10)
    except Exception as e:  # noqa: BLE001  (driver bug: record it atomically, then fail)
        import traceback
        set_status(state="needs_attention", attention=f"driver exception {type(e).__name__}: {e}")
        log(f"ATTENTION: driver exception {type(e).__name__}: {e}\n{traceback.format_exc()}")
        guard.ledger("SRS_V1_ATTENTION", f"driver exception {type(e).__name__}: {e}")
        sys.exit(10)


def preflight():
    man = jread(ART / "manifest.json")
    if guard.sha256_file(PREREG) != man["prereg_sha256"]:
        raise Terminal("PREREG hash changed")
    if guard.sha256_file(ART / "search_partition.json") != man["search_partition_sha256"]:
        raise Terminal("partition file changed")
    tests = sorted(ART.glob("cpu_tests_*.json"))
    if not tests or jread(tests[-1]).get("status") != "PASS":
        raise Attention("CPU test suite has not passed")


def gpu_checks():
    set_status("gpu_checks")
    passes = [p for p in sorted(ART.glob("gpu_checks_2*.json")) if "_quick" not in p.name
              and jread(p).get("status") == "PASS"]
    if passes:
        log(f"gpu checks: verified pass ({passes[-1].name})")
        return
    rc, _, _, out = launch("srsv1 gpu checks", ["srs_gpu_checks.py", "--device", "cuda", "--execute"], "gpu_checks.log")
    passes = [p for p in sorted(ART.glob("gpu_checks_2*.json")) if "_quick" not in p.name
              and jread(p).get("status") == "PASS"]
    if not passes:
        raise Attention(f"GPU correctness checks did not pass (rc={rc}); causality/implementation must be diagnosed")


def run_all():
    preflight()
    gpu_checks()
    # ---------------- Generation 1
    g1 = run_generation(1, sel.GEN1)
    g1r = write_once(ART / "gen1_results.json", {"generation": 1, "evalset": "calib_search", "seeds": [98],
                                                 "budget": 5_000_000, "results": g1})
    g1 = g1r["results"]
    # ---------------- Generation 2 (deterministic rule on Gen-1 CALIB-SEARCH only)
    set_status("gen2_rule")
    g2_specs = write_once(ART / "gen2_specs.json", {"rule": "srs_select.gen2_candidates (prereg §7)",
                                                    "specs": sel.gen2_candidates(g1)})["specs"]
    if not g2_specs:
        raise Terminal("NO_VALID_CANDIDATES: no valid Generation-1 candidate")
    if not (CFG_ROOT / "gen2").exists():
        if cpu(["srs_make_configs.py", "gen", "--gen", "2", "--specs", str((ART / "gen2_specs.json").relative_to(ROOT))],
               "configs_gen2.log"):
            raise Attention("gen2 config generation failed")
    g2 = run_generation(2, g2_specs)
    g2 = write_once(ART / "gen2_results.json", {"generation": 2, "evalset": "calib_search", "seeds": [98, 99],
                                                "budget": 10_000_000, "results": g2})["results"]
    # ---------------- Generation 3 finalists
    set_status("gen3_rule")
    fin = sel.gen3_finalists(g2, g2_specs)
    g3_specs = write_once(ART / "gen3_specs.json", {"rule": "srs_select.gen3_finalists (prereg §8)",
                                                    "specs": [{**s, "parent": s["id"], "id": s["id"].replace("G2", "G3")}
                                                              for s in fin]})["specs"]
    if not g3_specs:
        raise Terminal("NO_VALID_CANDIDATES: no valid Generation-2 candidate")
    if not (CFG_ROOT / "gen3").exists():
        if cpu(["srs_make_configs.py", "gen", "--gen", "3", "--specs", str((ART / "gen3_specs.json").relative_to(ROOT))],
               "configs_gen3.log"):
            raise Attention("gen3 config generation failed")
    g3 = run_generation(3, g3_specs)
    g3 = write_once(ART / "gen3_results.json", {"generation": 3, "evalset": "calib_search", "seeds": [97, 98, 99],
                                                "budget": 30_000_000, "results": g3})["results"]
    # ---------------- freeze (candidate generation ends permanently here)
    set_status("search_freeze")
    best = sel.freeze_best(g3, g3_specs)
    frozen = {"TOKEN-4": {"id": "TOKEN-4", "family": "token", "routing": sel.TOKEN}}
    for fam in ("fixed", "dynamic"):
        if best[fam] is not None:
            frozen[fam] = best[fam]
    sha = []
    runs = tsv("gen3_runs.tsv")
    for k, s in frozen.items():
        for seed in GEN_SEEDS[3]:
            sha.append(guard.sha256_file(CFG_ROOT / "gen3" / f"seed{seed}" / f"{s['id']}.json"))
    if not SEARCH_FROZEN.exists():
        lines = ["candidate\tgeneration\tfamily\tparent\tsame_as_gen1\trouting_key\trouting\tdescription\tvalidity"
                 "\tcalib_search_ce\tdelta_vs_TOKEN4_same_gen"]
        lines += candidates_tsv_lines(1, sel.GEN1, g1)
        lines += candidates_tsv_lines(2, g2_specs, g2)
        lines += candidates_tsv_lines(3, g3_specs, g3)
        guard.atomic_write_bytes(ART / "candidates.tsv", ("\n".join(lines) + "\n").encode(), mode=0o400)
        uniq = {sel.routing_key(s["routing"]) for s in sel.GEN1 + g2_specs + g3_specs}
        write_once(ART / "calib_search_results.json", {
            "evalset": "calib_search (70% of calib documents; development set; NOT final evidence)",
            "gen1": g1, "gen2": g2, "gen3": g3, "gen2_specs": g2_specs, "gen3_specs": g3_specs,
            "unique_sequence_candidates": len(uniq), "max_unique": sel.MAX_UNIQUE,
            "frozen_best": {k: v for k, v in frozen.items()}})
        if len(uniq) > sel.MAX_UNIQUE:
            raise Terminal("candidate limit exceeded")
        write_once(SEARCH_FROZEN, {"candidate_generation_ended": guard.now(), "frozen": frozen,
                                   "config_sha256s": sha,
                                   "calib_search_results_sha256": guard.sha256_file(ART / "calib_search_results.json")})
    frozen = jread(SEARCH_FROZEN)["frozen"]
    # ---------------- blind CALIB-SELECTION (one evaluation per checkpoint)
    set_status("calib_selection")
    sel_per = {}
    for k, s in frozen.items():
        sel_per[k] = {}
        for seed in GEN_SEEDS[3]:
            rd = runs[f"seed{seed}/{s['id']}"][0]
            ej = eval_item("calib_selection", f"seed{seed}/{s['id']}", rd)
            e = jread(ROOT / ej)
            f = jread(ROOT / rd / "run_facts.json")
            sel_per[k][seed] = {"complete": True, "ce": e["ce_nats_per_token"], "collapsed": e["collapse"]["collapsed"],
                                "extra_param_fraction": f["extra_router_param_fraction"],
                                "gpu_s": jread(ROOT / rd / "summary.json")["process_seconds"], "eval": ej}
    sel_agg = {k: aggregate(v) for k, v in sel_per.items()}
    prom = sel.promote({"TOKEN": sel_agg["TOKEN-4"], "fixed": sel_agg.get("fixed"), "dynamic": sel_agg.get("dynamic")})
    csr = write_once(ART / "calib_selection_results.json", {
        "evalset": "calib_selection (30% of calib documents; evaluated once, after candidate generation ended)",
        "frozen": frozen, "results": sel_agg, "promotion": prom,
        "search_vs_selection": {k: {"calib_search_gen3_ce": g3[frozen[k]["id"]]["ce"] if k != "TOKEN-4" else
                                    g3["TOKEN-4"]["ce"], "calib_selection_ce": sel_agg[k]["ce"]} for k in frozen}})
    prom = csr["promotion"]
    promoted = [f for f in ("fixed", "dynamic") if prom["families"][f]["promoted"]]
    log(f"promotion: {prom}")
    if not promoted:
        raise Terminal("NO_VALID_CANDIDATES: no sequence-routing family promoted after blind CALIB-SELECTION")
    # ---------------- throughput / resource gate and budget
    set_status("budget_gate")
    spt, peaks = {}, []
    for k in ["TOKEN-4"] + promoted:
        sid = frozen[k]["id"]
        rs = [run_result(runs[f"seed{seed}/{sid}"][0]) for seed in GEN_SEEDS[3]]
        spt[k] = max(r["gpu_s"] for r in rs) / min(r["valid_train_tokens"] for r in rs)
        peaks += [r["peak_reserved_gib"] for r in rs]
    val_eval_s = 120.0  # per checkpoint, conservative (10.7M tokens)
    overhead = 3 * (1 + len(promoted)) * val_eval_s + 1800.0  # validation evals + latency + slack
    used = srs_gpu_seconds()
    budget, rows = sel.choose_budget(used, spt, n_seeds=3, overhead_s=overhead)
    bud = write_once(ART / "budget.json", {"sec_per_token": spt, "rows": rows, "budget": budget, "used_gpu_s": used,
                                           "peak_reserved_gib_gen3": max(peaks), "promoted": promoted,
                                           "disk_free_bytes": guard.disk_free_bytes()})
    budget = bud["budget"]
    if max(peaks) >= VRAM_MAX_GIB:
        raise Terminal(f"VRAM: peak reserved {max(peaks):.1f} GiB >= {VRAM_MAX_GIB}")
    if budget is None:
        raise Terminal("NO_BUDGET: even 25M x 3 seeds does not fit under 20 GPU-hours with a 20% margin")
    log(f"budget {budget}: {rows}")
    # ---------------- final freeze and configs (before any final training / validation access)
    set_status("final_freeze")
    fspecs = [{"id": frozen[f]["id"], "family": f, "routing": frozen[f]["routing"],
               "desc": f"best {f} (frozen at search end)"} for f in promoted]
    write_once(ART / "final_specs.json", {"specs": fspecs, "budget": budget})
    fdir = CFG_ROOT / f"final_B{budget // 1_000_000}M"
    if not fdir.exists():
        if cpu(["srs_make_configs.py", "final", "--specs", str((ART / "final_specs.json").relative_to(ROOT)),
                "--budget", str(budget)], "configs_final.log"):
            raise Attention("final config generation failed")
    fids = ["TOKEN-4"] + [s["id"] for s in fspecs]
    if not FINAL_FROZEN.exists():
        shas = [guard.sha256_file(fdir / f"seed{s}" / f"{i}.json") for s in MAIN_SEEDS for i in fids]
        write_once(FINAL_FROZEN, {"frozen": guard.now(), "budget": budget, "ids": fids, "config_sha256s": shas,
                                  "config_dir": str(fdir.relative_to(ROOT))})
    # ---------------- final training (all runs), then validation exactly once per checkpoint
    set_status("final_train")
    per_run_s = {k: budget * v for k, v in spt.items()}
    fruns = {}
    for seed in MAIN_SEEDS:
        for k, i in zip(["TOKEN-4"] + promoted, fids):
            rd = train_item("final_runs.tsv", f"seed{seed}/{i}", fdir / f"seed{seed}" / f"{i}.json", need_ckpt=True,
                            upcoming_s=int(per_run_s[k]), allow_invalid=False)
            fruns[f"seed{seed}/{i}"] = rd
            set_status(final_done=len(fruns), final_total=3 * len(fids))
    set_status("final_validation")
    for key, rd in fruns.items():
        eval_item("val", key, rd)
    # ---------------- latency
    set_status("latency")
    if not [p for p in ART.glob("latency_*.json") if jread(p).get("status") == "COMPLETE"]:
        rc, _, _, _ = launch("srsv1 bench", ["srs_bench.py", "--runs"] + [fruns[f"seed0/{i}"] for i in fids]
                             + ["--device", "cuda", "--execute"], "bench.log")
        if not [p for p in ART.glob("latency_*.json") if jread(p).get("status") == "COMPLETE"]:
            fail("bench", "latency", None, f"rc={rc}")
            log("latency benchmark failed; continuing to analysis (reported as missing)")
    # ---------------- analysis and verification
    set_status("analysis")
    if not (ART / "final_analysis.json").exists():
        if cpu(["srs_analyze.py", "--execute"], "analyze.log", approve="cpu-model"):
            raise Attention("final analysis failed")
    set_status("verify")
    if cpu(["srs_inventory.py", "--verify", str((ART / "protected_inventory.json").relative_to(ROOT))], "verify.log"):
        raise Terminal("PROTECTED_ARTIFACT_MISMATCH: see verify log")


def eval_item(evalset, key, rd):
    reg = f"{evalset}_evals.tsv"
    done = tsv(reg)
    if key in done:
        e = jread(ROOT / done[key][0])
        if e.get("status") == "COMPLETE":
            return done[key][0]
    for attempt in range(3):
        rc, _, ej, _ = launch(f"srsv1 eval {evalset} {key}", ["srs_evaluate.py", "--run", rd, "--set", evalset,
                                                              "--device", "cuda", "--execute"],
                              f"eval_{evalset}__{key.replace('/', '_')}.log")
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
