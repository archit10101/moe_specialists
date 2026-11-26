#!/usr/bin/env python
"""csr-forecast-specialist-pretraining-v2 driver: runs every stage of the protocol sequentially, one guarded GPU job at
a time, idempotently (state in artifacts/forecast-specialist-pretraining-v2/driver_state.json). Re-running resumes.

  .venv/bin/python fsp_driver.py run [--until STAGE] [--authorized-resume]
  (CPU orchestration; each GPU job via scripts/gpu_guarded.sh)

Operational Amendment 1 (docs/FORECAST_SPECIALIST_PRETRAINING_V2_OPERATIONAL_AMENDMENT_1.md): any external
cancellation (signal to the driver or a child, the STOP marker, a child killed from outside, a foreign GPU process
before or during a run, GPU loss) makes the driver yield: the training child stops cooperatively at a step boundary with
a verified checkpoint, the run is marked EXTERNALLY_STOPPED, state + resume command are recorded, and the driver exits
without polling. It refuses to start again while an unresolved stop record or the STOP marker exists unless given
--authorized-resume (explicit user instruction only).
"""
import copy
import glob
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from csr import guard  # noqa: E402
from csr.fsp_lib import ART, CFG_DIR, protocol_sha  # noqa: E402

STATE = Path(os.environ["FSP_DRIVER_STATE"]) if os.environ.get("FSP_DRIVER_STATE") else ART / "driver_state.json"
STOP_MARKER = ART / "STOP"
RESUME_CMD = ("cd /home/zapan4/moe-research && source env.sh && nohup .venv/bin/python fsp_driver.py run "
              "--authorized-resume >> .tmp/fspv2/driver.out 2>&1 &")
GPU_LOSS = re.compile(r"no CUDA GPUs|CUDA-capable device|cudaErrorDevicesUnavailable|cudaErrorNoDevice|GPU is lost|"
                      r"device not ready|uncorrectable ECC|Xid|CUDA driver initialization failed|"
                      r"cudaErrorDevicesUnavailable|NVML_ERROR_GPU_IS_LOST", re.I)
DRIVER = {"stop": None, "child": None, "active": None, "stage": None}


class ExternalStop(Exception):
    """External cancellation / GPU yield (amendment B). Never treated as a software error."""
BUDGET_H = 48.0
PROBE_GPU_S = 420.0  # pre-protocol throughput probes (fsp_quickbench/quickprof), conservatively
DEV_SEEDS = (100, 101)
FINAL_SEEDS = (0, 1)
M = 1_000_000
TRAIN = {"lr": 6e-4, "min_lr": 6e-5, "warmup_frac": 0.01, "decay_frac": 0.2, "rows_per_step": 64,
         "micro_batch_rows": 32, "weight_decay": 0.1, "betas": [0.9, 0.95], "clip": 1.0, "aux_coeff": 0.01,
         "z_coeff": 0.001, "log_every": 50, "slot_every_s": 600}
DRAFT_TRAIN = {**TRAIN, "lr": 1e-3, "min_lr": 1e-4, "micro_batch_rows": 32}
BASE = {"fc_horizon": 8, "topic_halflife": 128.0, "spec_beta": 1.0, "dom12": 0.1, "dom3": 0.02, "distill_coef": 1.0,
        "anneal_shape": "linear", "router_lr_mult": 1.0}
CANDIDATES = {
    "C01": {}, "C02": {"fc_horizon": 4}, "C03": {"fc_horizon": 1}, "C04": {"topic_halflife": 32.0},
    "C05": {"topic_halflife": 512.0}, "C06": {"spec_beta": 3.0}, "C07": {"spec_beta": 0.3},
    "C08": {"dom12": 0.3, "dom3": 0.05}, "C09": {"dom12": 0.03, "dom3": 0.005}, "C10": {"distill_coef": 0.1},
    "C11": {"anneal_shape": "cosine"}, "C12": {"router_lr_mult": 3.0},
}
EDL_BETAS = {"EDL-b5e-4": 5e-4, "EDL-b5e-3": 5e-3}


# ------------------------------------------------------------------ state / ledgers
def load_state():
    return json.loads(STATE.read_text()) if STATE.exists() else {"done": {}, "runs": {}}


def save_state(s):
    s["active"] = DRIVER["active"]
    guard.atomic_write_json(STATE, s, overwrite=True)


def log(msg):
    line = f"{guard.now()} {msg}"
    print(line, flush=True)
    with open(ART / "driver_log.txt", "a") as f:
        f.write(line + "\n")


def deviation(kind, detail):
    with open(ART / "deviations.jsonl", "a") as f:
        f.write(json.dumps({"time": guard.now(), "kind": kind, "detail": detail}) + "\n")
    log(f"DEVIATION {kind}: {detail}")


def gpu_seconds_used():
    p = ART / "compute_ledger.tsv"
    s = PROBE_GPU_S
    if p.exists():
        for line in p.read_text().splitlines():
            f = line.split("\t")
            if len(f) >= 3:
                try:
                    s += float(f[2])
                except ValueError:
                    pass
    return s


def budget_ok(projected_s, reserve_h=0.0, what=""):
    used = gpu_seconds_used()
    ok = used + projected_s <= (BUDGET_H - reserve_h) * 3600
    if not ok:
        log(f"BUDGET: skip {what}: used {used / 3600:.2f} h + projected {projected_s / 3600:.2f} h > "
            f"{BUDGET_H - reserve_h:.1f} h")
    return ok


def disk_ok():
    free = guard.disk_free_bytes()
    if free < 80 * 1024**3 + 5 * 1024**3:
        raise guard.GuardError(f"disk: only {free / 2**30:.1f} GiB free (< 85 GiB working floor)")


# ------------------------------------------------------------------ guarded launch
def _session_pids(sid):
    out = []
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as f:
                fields = f.read().rsplit(")", 1)[1].split()
            if int(fields[3]) == sid:  # fields after the comm: state ppid pgrp session ...
                out.append(int(d))
        except (OSError, IndexError, ValueError):
            pass
    return out


def _cmdline(pid):
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return ""


def _argv(pid):
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return [x.decode(errors="replace") for x in f.read().split(b"\0") if x]
    except OSError:
        return []


def _is_experiment_main(pid, script=None):
    """The experiment's own python process (argv[0] = python, argv[1] = fsp_*.py), not timeout/bash wrappers or
    torch.compile worker subprocesses."""
    a = _argv(pid)
    return len(a) >= 2 and a[0].endswith("/python") and os.path.basename(a[1]).startswith("fsp_") and \
        (script is None or os.path.basename(a[1]) == os.path.basename(script))


def _forward_stop(sig_name):
    """Cooperative stop: SIGINT to the experiment's main python process of the running child session only."""
    ch = DRIVER["child"]
    if ch is None or ch.poll() is not None:
        return
    script = (DRIVER["active"] or {}).get("script")
    for pid in _session_pids(ch.pid):
        if _is_experiment_main(pid, script):
            try:
                os.kill(pid, signal.SIGINT)
                log(f"forwarded cooperative SIGINT to child python pid {pid} ({sig_name})")
            except ProcessLookupError:
                pass


def _driver_signal(signum, frame):
    name = signal.Signals(signum).name
    if DRIVER["stop"] is None:
        DRIVER["stop"] = f"driver received {name}"
        log(f"EXTERNAL STOP: driver received {name}; stopping at the next safe point")
        _forward_stop(name)


def pre_launch_check(purpose):
    if DRIVER["stop"]:
        raise ExternalStop(DRIVER["stop"])
    if STOP_MARKER.exists():
        raise ExternalStop(f"stop marker {STOP_MARKER.relative_to(ROOT)} present before launching {purpose}")


def classify_rc(rc):
    if DRIVER["stop"]:
        return DRIVER["stop"]
    if rc in (3, 78):
        return f"foreign GPU process present before launch (rc={rc}); launch blocked, yielding"
    if rc == 77:
        return "child reported EXTERNALLY_STOPPED"
    if rc < 0 or (rc >= 128 and rc != 124):
        return f"child terminated from outside (rc={rc})"
    if STOP_MARKER.exists():
        return "stop marker present"
    return None


def launch(purpose, script, args, defer_external=False):
    """One guarded launch. Never polls a busy GPU (amendment B6). Returns rc, or (rc, external_reason) when
    defer_external; otherwise raises ExternalStop on any external cancellation."""
    env = dict(os.environ, MOE_GPU_LAUNCH="1", MOE_RESEARCH_APPROVE="gpu")
    pre_launch_check(purpose)
    disk_ok()
    DRIVER["active"] = {"stage": DRIVER["stage"], "purpose": purpose, "script": script, "args": args,
                        "launched": guard.now()}
    ch = subprocess.Popen(["scripts/gpu_guarded.sh", purpose, script, *args], cwd=ROOT, env=env,
                          start_new_session=True)
    DRIVER["child"] = ch
    if DRIVER["stop"]:  # signal arrived between the check and Popen
        _forward_stop("late")
    rc = ch.wait()
    # the wrapper may exit before its python grandchild finished writing: wait for the whole session (<= 20 min)
    t0 = time.time()
    while _session_pids(ch.pid) and time.time() - t0 < 1200:
        time.sleep(2)
    left = _session_pids(ch.pid)
    if left:
        log(f"WARNING: child session {ch.pid} still has processes {left} after 20 min")
    DRIVER["child"] = None
    ext = classify_rc(rc)
    if defer_external:
        return rc, ext
    if ext:
        raise ExternalStop(f"{purpose}: {ext}")
    DRIVER["active"] = None
    return rc


def write_cfg(sub, name, cfg):
    d = CFG_DIR / sub
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.json"
    if p.exists():
        old = json.loads(p.read_text())
        if old != cfg:
            raise guard.GuardError(f"config {p} exists with different content")
        return p
    guard.atomic_write_json(p, cfg, mode=0o400)
    return p


def latest_ckpt(rd):
    """Latest verified resumable checkpoint entry of a run dir (amendment A6), or None."""
    if rd is None:
        return None
    from csr.fsp_ckpt import ResumeStore
    try:
        return ResumeStore(ROOT / ".tmp/fspv2/slots", rd.name, "").latest_verified(log=lambda m: None)
    except Exception as e:
        return {"error": str(e)[:300]}


def mark_external(rec, rd, status, ext):
    ck = latest_ckpt(rd)
    if rd is not None and status in ("RUNNING",) or (rd is not None and status.startswith("rc")):
        # killed too abruptly to write its own status: mark it (never COMPLETE); all files are kept
        old = json.loads((rd / "status.json").read_text()) if (rd / "status.json").exists() else {}
        guard.atomic_write_json(rd / "status.json", {**old, "status": "EXTERNALLY_STOPPED", "time": guard.now(),
                                                     "reason": f"terminated externally without checkpointing: {ext}",
                                                     "resume_from": ck}, overwrite=True)
    rec["status"] = "EXTERNALLY_STOPPED"
    rec.setdefault("external_stops", []).append({"time": guard.now(), "reason": ext, "resume_from": ck})


def find_run_dir(run_name):
    c = sorted(glob.glob(str(ROOT / f"runs/*_fsp_{run_name}")))
    return Path(c[-1]) if c else None


def train_run(state, cfg_path, run_name, purpose):
    """Launch legs until COMPLETE. Returns summary dict (or None if failed twice)."""
    rec = state["runs"].setdefault(run_name, {"attempts": 0, "legs": 0, "cfg": str(cfg_path.relative_to(ROOT))})
    if rec.get("status") == "COMPLETE":
        return json.loads((ROOT / rec["run_dir"] / "summary.json").read_text())
    mbr = rec.get("micro_batch_rows")
    while True:
        rd = find_run_dir(run_name)
        args = ["fsp_train.py", "--config", str(cfg_path.relative_to(ROOT)), "--device", "cuda", "--execute"]
        if rd is not None and rec.get("run_dir") == str(rd.relative_to(ROOT)):
            args += ["--run-dir", str(rd.relative_to(ROOT))]
        if mbr:
            args += ["--micro-batch-rows", str(mbr)]
        rc, ext = launch(purpose, args[0], args[1:], defer_external=True)
        rd = find_run_dir(run_name)
        if rc in (3, 78) and rd is not None and rec.get("run_dir") != str(rd.relative_to(ROOT)):
            rd = None  # blocked before start: no new run dir was created; do not adopt an older one
        rec["run_dir"] = str(rd.relative_to(ROOT)) if rd else rec.get("run_dir")
        if rc not in (3, 78):
            rec["legs"] += 1
        st = json.loads((rd / "status.json").read_text()) if rd and (rd / "status.json").exists() else {}
        status = st.get("status", f"rc{rc}")
        log(f"{run_name}: leg {rec['legs']} rc={rc} status={status}")
        save_state(state)
        if status == "COMPLETE":
            rec["status"] = "COMPLETE"
            DRIVER["active"] = None
            save_state(state)
            if ext:
                raise ExternalStop(f"{run_name}: {ext} (run had completed and was recorded)")
            return json.loads((rd / "summary.json").read_text())
        if ext is None and status == "FAILED" and GPU_LOSS.search(st.get("reason", "")):
            ext = "GPU unavailable/lost during run: " + GPU_LOSS.search(st.get("reason", "")).group(0)
        if ext is None and status == "EXTERNALLY_STOPPED" and rc != 124:
            ext = st.get("reason", "EXTERNALLY_STOPPED")
        if ext:
            mark_external(rec, rd, status, ext)
            save_state(state)
            raise ExternalStop(f"{run_name}: {ext}")
        if rc in (75, 124) or status in ("INTERRUPTED",):  # internal leg limit / guard wall clock: resumable leg
            continue
        if rc == 76 or status == "OOM":
            mbr = max(4, (mbr or 32) // 2)
            rec["micro_batch_rows"] = mbr
            deviation("oom", f"{run_name}: micro-batch -> {mbr}")
            continue
        rec["attempts"] += 1
        rec["failures"] = rec.get("failures", []) + [st.get("reason", "")[-1500:]]
        save_state(state)
        if rec["attempts"] >= 2:
            rec["status"] = "FAILED"
            save_state(state)
            log(f"{run_name}: FAILED twice; recorded")
            return None
        # a fresh start after a failure (the failed dir is kept and recorded)
        rec["failed_dirs"] = rec.get("failed_dirs", []) + [rec.get("run_dir")]
        rec["run_dir"] = None


# ------------------------------------------------------------------ configs
def data_manifest():
    return json.loads((ART / "data/manifest.json").read_text())


def main_quota_total():
    """Final main budget = valid tokens of every WHOLE 64-row step of the single-epoch main stream (the trailing
    partial step, < 64 rows, is dropped identically for every condition)."""
    p = ART / "final_budget.json"
    if p.exists():
        return json.loads(p.read_text())["final_main_tokens"]
    from csr.fsp_lib import Split, main_order, pack_rows
    sp = Split("main")
    rows = pack_rows(sp.lengths, main_order(sp))
    n = len(rows) // TRAIN["rows_per_step"] * TRAIN["rows_per_step"]
    tot = int(sum(sp.lengths[d] - 1 for r in rows[:n] for d in r))
    guard.atomic_write_json(p, {"final_main_tokens": tot, "n_steps": n // TRAIN["rows_per_step"], "n_rows": len(rows),
                                "quota_total": 5 * data_manifest()["quotas_valid_tokens_per_domain"]["main"]})
    return tot


def model_cfg(routing, teacher, knobs):
    from csr.fsp_lib import N_SCALAR
    from csr.fsp_model import FSPConfig
    return FSPConfig(routing=routing, teacher=teacher, fc_horizon=knobs["fc_horizon"],
                     topic_halflife=knobs["topic_halflife"], spec_beta=knobs["spec_beta"],
                     n_scalar=N_SCALAR).to_dict()


def condition_cfg(cond_name, knobs, edl_beta=None):
    """Returns (model_cfg dict, cond dict)."""
    k = knobs
    dom = {"use_dom": True, "dom12": k["dom12"], "dom3": k["dom3"]}
    fc = {"curriculum": True, "distill_coef": k["distill_coef"], "anneal_shape": k["anneal_shape"],
          "router_lr_mult": k["router_lr_mult"]}
    if cond_name in ("TOKEN-4", "TOKEN-COMPUTE"):
        return model_cfg("token4", "none", k), {}
    if cond_name == "DOMAIN-AUX":
        return model_cfg("token4", "none", k), {**dom}
    if cond_name == "EDL":
        return model_cfg("token4", "none", k), {"edl_beta": edl_beta}
    if cond_name == "FORECAST-SPECIALIST":
        return model_cfg("spec", "spec", k), {**dom, **fc, "forecast_source": "match"}
    if cond_name == "LABEL-ORACLE":
        return model_cfg("spec", "none", k), {**dom, "oracle": True, "router_lr_mult": k["router_lr_mult"]}
    if cond_name == "SHUFFLED":
        return model_cfg("spec", "spec", k), {**dom, **fc, "forecast_source": "shuffled", "label_mode": "shuffled"}
    if cond_name == "WRONG-DRAFT":
        return model_cfg("spec", "spec", k), {**dom, **fc, "forecast_source": "wrong"}
    if cond_name == "FORECAST-NOPART":
        return model_cfg("token4", "nopart", k), {**dom, **fc, "forecast_source": "match"}
    raise KeyError(cond_name)


def main_cfg(run_name, cond_name, seed, budget, knobs, eval_splits, ckpt_fracs, edl_beta=None, stream="main",
             category="main_training"):
    mc, cond = condition_cfg(cond_name, knobs, edl_beta)
    return {"experiment": "csr-forecast-specialist-pretraining-v2", "protocol_sha256": protocol_sha(),
            "data_splits_sha256": data_manifest()["splits_sha256"], "run_name": run_name, "kind": "main",
            "condition": cond_name, "seed": seed, "budget_tokens": int(budget), "stream": stream, "model": mc,
            "cond": cond, "knobs": knobs, "train": TRAIN, "eval_splits": eval_splits,
            "checkpoint_fracs": ckpt_fracs, "cache_dir": "artifacts/forecast-specialist-pretraining-v2/cache/main",
            "category": category}


# ------------------------------------------------------------------ stages
def stage_tests(state):
    if state["done"].get("tests"):
        return
    env = dict(os.environ)
    r = subprocess.run([".venv/bin/python", "fsp_tests.py"], cwd=ROOT, env=env)
    if r.returncode != 0:
        raise guard.GuardError("CPU test suite failed")
    rc = launch("fspv2 gpu checks", "fsp_gpu_checks.py", ["--execute"])
    if rc != 0:
        raise guard.GuardError("GPU checks failed")
    state["done"]["tests"] = guard.now()
    save_state(state)


def stage_drafts(state):
    if state["done"].get("drafts"):
        return
    base = {"experiment": "csr-forecast-specialist-pretraining-v2", "protocol_sha256": protocol_sha(),
            "data_splits_sha256": data_manifest()["splits_sha256"], "seed": 0, "train": DRAFT_TRAIN,
            "checkpoint_fracs": [], "category": "draft_training"}
    c = write_cfg("drafts", "draft_shared", {**base, "run_name": "draft_shared", "kind": "draft_shared"})
    s = train_run(state, c, "draft_shared", "fspv2 shared draft")
    if s is None:
        raise guard.GuardError("shared draft failed")
    shared = s["ckpts"]["final"]["path"]
    paths = []
    for d in range(5):
        cfg = {**base, "run_name": f"draft_cont_d{d}", "kind": "draft_cont", "domain": d, "init_from": shared,
               "train": {**DRAFT_TRAIN, "lr": 5e-4, "min_lr": 5e-5}}
        c = write_cfg("drafts", f"draft_cont_d{d}", cfg)
        s = train_run(state, c, f"draft_cont_d{d}", f"fspv2 domain draft {d}")
        if s is None:
            raise guard.GuardError(f"domain draft {d} failed")
        paths.append(s["ckpts"]["final"]["path"])
    guard.atomic_write_json(ART / "drafts.json", {"shared": shared, "domain": paths}, overwrite=True)
    (ART / "draft_paths.json").write_text(json.dumps(paths))
    state["done"]["drafts"] = guard.now()
    save_state(state)


def stage_draft_diag(state):
    if (ART / "draft_diagnostics.json").exists():
        state["done"]["draft_diag"] = state["done"].get("draft_diag") or guard.now()
        save_state(state)
        return
    rc = launch("fspv2 draft diagnostics", "fsp_draft_diag.py", ["--execute"])
    if rc != 0:
        deviation("draft_diag_failed", f"rc={rc}; continuing (diagnostics are not stop gates)")
    state["done"]["draft_diag"] = guard.now()
    save_state(state)


def stage_cache(state):
    if state["done"].get("cache"):
        return
    out = "artifacts/forecast-specialist-pretraining-v2/cache/main"
    for leg in range(12):
        rc = launch("fspv2 forecast cache", "fsp_cache.py", ["--drafts", "artifacts/forecast-specialist-pretraining-v2/"
                    "draft_paths.json", "--split", "main", "--budget-tokens", str(main_quota_total()), "--out", out,
                                                              "--execute"])
        if rc == 0:
            break
        if rc != 75:
            raise guard.GuardError(f"cache failed rc={rc}")
    state["done"]["cache"] = guard.now()
    save_state(state)


def stage_resume(state):
    if state["done"].get("resume_check"):
        return
    r = subprocess.run([".venv/bin/python", "fsp_resume_check.py"], cwd=ROOT)
    if r.returncode != 0:
        raise guard.GuardError("resume-equivalence check failed")
    state.update(load_state()["runs"] and {"runs": load_state()["runs"]})
    state["done"]["resume_check"] = guard.now()
    save_state(state)


def ledger_row(row):
    p = ART / "search_ledger.tsv"
    hdr = ["time", "stage", "id", "condition", "seed", "budget", "run_dir", "status", "ce_calib_search",
           "ce_calib_sel", "spec_group_acc", "min_spec_share", "eligible", "tokens_per_s", "note"]
    new = not p.exists()
    with open(p, "a") as f:
        if new:
            f.write("\t".join(hdr) + "\n")
        f.write("\t".join(str(row.get(h, "")) for h in hdr) + "\n")


def run_eval_fields(summary):
    out = {}
    ev = (summary or {}).get("eval", {})
    for sp in ("calib_search", "calib_sel"):
        if sp in ev:
            out[f"ce_{sp}"] = ev[sp]["ce"]
    r = ev.get("calib_search", {}).get("routing", {})
    if "spec_group_acc_mean" in r:
        out["spec_group_acc"] = r["spec_group_acc_mean"]
        out["min_spec_share"] = r["min_spec_share"]
        out["eligible"] = bool(r["spec_group_acc_mean"] >= 0.5 and r["min_spec_share"] >= 0.02 / 1.0 * 1.0 and
                               math.isfinite(ev["calib_search"]["ce"]))
    return out


def dev_run(state, stage, rid, cond_name, seed, budget, knobs, eval_splits, edl_beta=None, note=""):
    name = f"{stage}_{rid}_s{seed}"
    cfg = main_cfg(name, cond_name, seed, budget, knobs, eval_splits, [1.0], edl_beta, category="search")
    proj = budget / (150_000 if "spec" in cfg["model"]["routing"] else 170_000) + 300
    if not budget_ok(proj, reserve_h=0, what=name):
        ledger_row({"time": guard.now(), "stage": stage, "id": rid, "condition": cond_name, "seed": seed,
                    "budget": budget, "status": "SKIPPED_BUDGET"})
        return None
    c = write_cfg(stage, name, cfg)
    already = state["runs"].get(name, {}).get("status") == "COMPLETE"
    s = train_run(state, c, name, f"fspv2 {stage} {rid} s{seed}")
    if not already:
        ledger_row({"time": guard.now(), "stage": stage, "id": rid, "condition": cond_name, "seed": seed,
                    "budget": budget, "run_dir": state["runs"][name].get("run_dir"),
                    "status": "COMPLETE" if s else "FAILED", "tokens_per_s": (s or {}).get("tokens_per_s"),
                    "note": note, **run_eval_fields(s)})
    return s


def stage_search1(state):
    if state["done"].get("search1"):
        return
    res = {}
    for rid, ch in CANDIDATES.items():
        s = dev_run(state, "s1", rid, "FORECAST-SPECIALIST", DEV_SEEDS[0], 50 * M, {**BASE, **ch}, ["calib_search"])
        res[rid] = run_eval_fields(s) if s else {"failed": True}
    ref = {}
    s = dev_run(state, "s1", "TOKEN-4", "TOKEN-4", DEV_SEEDS[0], 50 * M, BASE, ["calib_search"], note="reference")
    ref["TOKEN-4"] = run_eval_fields(s) if s else {}
    for rid, b in EDL_BETAS.items():
        s = dev_run(state, "s1", rid, "EDL", DEV_SEEDS[0], 50 * M, BASE, ["calib_search"], edl_beta=b,
                    note="EDL beta grid")
        ref[rid] = run_eval_fields(s) if s else {}
    # rank: eligible first, then CALIB-SEARCH CE
    rk = sorted([r for r in res if not res[r].get("failed")],
                key=lambda r: (not res[r].get("eligible", False), res[r].get("ce_calib_search", 1e9)))
    adv = rk[:4]
    edl = min(EDL_BETAS, key=lambda r: ref.get(r, {}).get("ce_calib_search", 1e9))
    out = {"candidates": res, "reference": ref, "ranking": rk, "advanced": adv, "edl_choice": edl,
           "edl_beta": EDL_BETAS[edl], "rule": "eligible first, then CALIB-SEARCH CE (protocol 8)"}
    guard.atomic_write_json(ART / "search_stage1.json", out, overwrite=True)
    log(f"stage1 advanced {adv}; EDL beta {EDL_BETAS[edl]}")
    state["done"]["search1"] = guard.now()
    save_state(state)


def stage_search2(state):
    if state["done"].get("search2"):
        return
    s1 = json.loads((ART / "search_stage1.json").read_text())
    res = {}
    for rid in s1["advanced"]:
        for seed in DEV_SEEDS:
            s = dev_run(state, "s2", rid, "FORECAST-SPECIALIST", seed, 150 * M, {**BASE, **CANDIDATES[rid]},
                        ["calib_search", "calib_sel"])
            res.setdefault(rid, {})[seed] = run_eval_fields(s) if s else {"failed": True}
    ref = {}
    for seed in DEV_SEEDS:
        s = dev_run(state, "s2", "TOKEN-4", "TOKEN-4", seed, 150 * M, BASE, ["calib_search", "calib_sel"],
                    note="reference")
        ref[seed] = run_eval_fields(s) if s else {}

    def key(rid):
        rs = list(res[rid].values())
        ok = all(not r.get("failed") for r in rs)
        elig = ok and all(r.get("eligible", False) for r in rs)
        ce = sum(r.get("ce_calib_sel", 1e9) for r in rs) / len(rs) if ok else 1e9
        return (not elig, ce)
    rk = sorted(res, key=key)
    frozen_id = rk[0]
    frozen = {"id": frozen_id, "knobs": {**BASE, **CANDIDATES[frozen_id]}, "eligible": not key(frozen_id)[0],
              "mean_ce_calib_sel": key(frozen_id)[1], "stage2": res, "reference_token4": ref,
              "edl_beta": s1["edl_beta"], "edl_choice": s1["edl_choice"], "frozen_at": guard.now(),
              "rule": "eligible in both dev seeds first, then mean CALIB-SELECTION CE (protocol 8)"}
    guard.atomic_write_json(ART / "frozen.json", frozen, overwrite=False, mode=0o400)
    log(f"FROZEN {frozen_id} sha={guard.sha256_file(ART / 'frozen.json')}")
    guard.ledger("FSP_FREEZE", f"frozen.json {frozen_id} sha256={guard.sha256_file(ART / 'frozen.json')}")
    state["done"]["search2"] = guard.now()
    save_state(state)


def frozen():
    return json.loads((ART / "frozen.json").read_text())


def stage_dev_controls(state, seeds=DEV_SEEDS):
    if state["done"].get("dev_controls"):
        return
    fz = frozen()
    for cond in ("LABEL-ORACLE", "SHUFFLED", "WRONG-DRAFT", "FORECAST-NOPART", "DOMAIN-AUX"):
        for seed in seeds:
            dev_run(state, "dc", cond, cond, seed, 150 * M, fz["knobs"], ["calib_search", "calib_sel"],
                    note="dev-scale mechanistic control (not a selection input)")
    state["done"]["dev_controls"] = guard.now()
    save_state(state)


FINAL_FRACS = [0.4, 0.5, 0.75, 1.0]


def final_run(state, cond, seed, budget=None, stream="main", reserve_h=0.0):
    fz = frozen()
    budget = budget or main_quota_total()
    name = f"final_{cond}_s{seed}"
    if state["runs"].get(name, {}).get("status") == "COMPLETE":
        return state["runs"][name]
    cfg = main_cfg(name, cond, seed, budget, fz["knobs"], [], FINAL_FRACS,
                   edl_beta=fz["edl_beta"] if cond == "EDL" else None, stream=stream)
    rate = 150_000 if cfg["model"]["routing"] == "spec" or cfg["model"]["teacher"] != "none" else 170_000
    if not budget_ok(budget / rate + 600, reserve_h=reserve_h, what=name):
        state["runs"].setdefault(name, {})["status"] = "SKIPPED_BUDGET"
        save_state(state)
        return None
    c = write_cfg("final", name, cfg)
    s = train_run(state, c, name, f"fspv2 final {cond} s{seed}")
    return s


def eval_job(state, run_name, what):
    key = f"eval:{run_name}:{what}"
    if state["done"].get(key):
        return
    rd = state["runs"].get(run_name, {}).get("run_dir")
    if not rd:
        return
    rc = launch(f"fspv2 eval {run_name} {what}", "fsp_evaluate.py", ["--run-dir", rd, "--what", what, "--execute"])
    if rc != 0:
        deviation("eval_failed", f"{key} rc={rc}")
        return
    state["done"][key] = guard.now()
    save_state(state)


PRIMARY = ("TOKEN-4", "EDL", "DOMAIN-AUX", "FORECAST-SPECIALIST")


def stage_finals(state):
    for seed in FINAL_SEEDS:
        for cond in PRIMARY:
            final_run(state, cond, seed)
            eval_job(state, f"final_{cond}_s{seed}", "full")


def compute_match(state):
    """Protocol 7: TOKEN-COMPUTE token budget from measured GPU seconds."""
    rows = [l.split("\t") for l in (ART / "compute_ledger.tsv").read_text().splitlines()]
    sec = lambda pred: sum(float(r[2]) for r in rows if len(r) >= 7 and pred(r))
    draft_s = sec(lambda r: r[5] == "draft_training")
    cache_s = sec(lambda r: r[5] == "forecast_cache" and "calib" not in r[4])
    d_runs = [f"final_FORECAST-SPECIALIST_s{s}" for s in FINAL_SEEDS]
    d_s = [sec(lambda r, n=n: r[4] == n) for n in d_runs]
    t4 = [json.loads((ROOT / state["runs"][f"final_TOKEN-4_s{s}"]["run_dir"] / "summary.json").read_text())
          for s in FINAL_SEEDS]
    t4_rate = sum(x["seen"] for x in t4) / sum(sec(lambda r, s=s: r[4] == f"final_TOKEN-4_s{s}") for s in FINAL_SEEDS)
    n_seeds = len([x for x in d_s if x > 0])
    target = sum(d_s) / n_seeds + (draft_s + cache_s) / n_seeds
    budget = int(target * t4_rate)
    out = {"draft_training_gpu_s": draft_s, "forecast_cache_gpu_s": cache_s, "proposed_main_gpu_s": d_s,
           "n_final_proposed_seeds": n_seeds, "target_gpu_s_per_model": target,
           "token4_end_to_end_tokens_per_gpu_s": t4_rate, "token_compute_budget_tokens": budget,
           "main_tokens": main_quota_total(), "created": guard.now()}
    guard.atomic_write_json(ART / "compute_match.json", out, overwrite=True)
    return out


def stage_token_compute(state):
    if not state["done"].get("compute_match"):
        cm = compute_match(state)
        state["done"]["compute_match"] = guard.now()
        save_state(state)
    cm = json.loads((ART / "compute_match.json").read_text())
    for seed in FINAL_SEEDS:
        final_run(state, "TOKEN-COMPUTE", seed, budget=cm["token_compute_budget_tokens"], stream="compute")
        eval_job(state, f"final_TOKEN-COMPUTE_s{seed}", "ce")


def primary_val(state, cond, seed):
    p = ART / "evals" / f"final_{cond}_s{seed}_full.json"
    return json.loads(p.read_text())["p100"]["val"]["ce"] if p.exists() else None


def stage_seed2(state):
    d = [primary_val(state, "FORECAST-SPECIALIST", s) for s in FINAL_SEEDS]
    a = [primary_val(state, "TOKEN-4", s) for s in FINAL_SEEDS]
    c = [primary_val(state, "EDL", s) for s in FINAL_SEEDS]
    if None in d + a + c:
        log("seed-2 rule: missing primary results; not triggered")
        return
    trig = all(d[i] < a[i] and d[i] < c[i] for i in range(len(FINAL_SEEDS)))
    state["done"]["seed2_triggered"] = trig
    save_state(state)
    log(f"seed-2 rule triggered={trig}")
    if trig:
        for cond in PRIMARY:
            final_run(state, cond, 2)
            eval_job(state, f"final_{cond}_s2", "full")


def stage_final_controls(state):
    for cond in ("LABEL-ORACLE", "SHUFFLED", "WRONG-DRAFT", "FORECAST-NOPART"):
        for seed in FINAL_SEEDS:
            s = final_run(state, cond, seed, reserve_h=2.5)
            if s is None:
                continue
            eval_job(state, f"final_{cond}_s{seed}", "full")


def stage_bench(state):
    if state["done"].get("bench"):
        return
    rc = launch("fspv2 benchmarks", "fsp_bench.py", ["--execute"])
    if rc != 0:
        deviation("bench_failed", f"rc={rc}")
    state["done"]["bench"] = guard.now()
    save_state(state)


STAGES = [("tests", stage_tests), ("drafts", stage_drafts), ("draft_diag", stage_draft_diag), ("cache", stage_cache),
          ("resume_check", stage_resume),
          ("search1", stage_search1), ("search2", stage_search2), ("dev_controls", stage_dev_controls),
          ("finals", stage_finals), ("token_compute", stage_token_compute), ("seed2", stage_seed2),
          ("final_controls", stage_final_controls), ("bench", stage_bench)]


def record_external_stop(state, reason):
    act = DRIVER["active"] or {}
    rd = None
    m = re.match(r"^(\S+?):", reason)
    if m and m.group(1) in state["runs"]:
        rdp = state["runs"][m.group(1)].get("run_dir")
        rd = ROOT / rdp if rdp else None
    gpu = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True,
                         text=True).stdout.split()
    mine = [p for p in gpu if p.isdigit() and _is_experiment_main(int(p))]
    state["external_stop"] = {"status": "EXTERNALLY_STOPPED", "time": guard.now(), "reason": reason,
                              "stage": DRIVER["stage"], "active": act,
                              "run_dir": str(rd.relative_to(ROOT)) if rd else None,
                              "resume_from_checkpoint": latest_ckpt(rd) if rd else None,
                              "gpu0_pids_after_yield": gpu, "experiment_pids_on_gpu_after_yield": mine,
                              "resume_command": RESUME_CMD,
                              "note": "do not resume without an explicit user instruction (amendment B10)"}
    DRIVER["active"] = None
    save_state(state)
    log(f"EXTERNALLY_STOPPED at stage {DRIVER['stage']}: {reason}; GPU0 pids now {gpu} (experiment-owned: {mine}); "
        f"waiting for explicit user authorization. Resume: {RESUME_CMD}")


def main():
    guard.check_root()
    until = sys.argv[sys.argv.index("--until") + 1] if "--until" in sys.argv else None
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    authorized = "--authorized-resume" in sys.argv
    state = load_state()
    if STOP_MARKER.exists():
        log(f"REFUSING to start: stop marker {STOP_MARKER.relative_to(ROOT)} exists (remove it only on user "
            f"instruction)")
        sys.exit(3)
    pending = [k for k in ("operator_stop", "external_stop") if k in state]
    if pending and not authorized:
        log(f"REFUSING to start: unresolved {pending}; needs explicit user authorization (--authorized-resume)")
        sys.exit(3)
    if pending:
        hist = state.setdefault("stop_history", [])
        for k in pending:
            hist.append({**state.pop(k), "kind": k, "resumed": guard.now(), "resumed_by": "--authorized-resume"})
        log(f"AUTHORIZED RESUME (resolving {pending})")
        save_state(state)
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _driver_signal)
    try:
        if only == "selftest":
            stage_selftest(state)
            log("driver finished requested stages")
            return
        for name, fn in STAGES:
            if only and name != only:
                continue
            DRIVER["stage"] = name
            log(f"== stage {name} (GPU used so far {gpu_seconds_used() / 3600:.2f} h)")
            fn(state)
            if DRIVER["stop"]:
                raise ExternalStop(DRIVER["stop"])
            if until == name:
                break
        log("driver finished requested stages")
    except ExternalStop as e:
        record_external_stop(state, str(e))
        sys.exit(0)


def stage_selftest(state):
    """Amendment test only (separate FSP_DRIVER_STATE): one ~150-step FORECAST-SPECIALIST run through train_run."""
    DRIVER["stage"] = "selftest"
    tag = os.environ.get("FSP_SELFTEST_TAG", "x")
    name = f"selftest{tag}"
    cfg = main_cfg(name, "FORECAST-SPECIALIST", 100, 150 * 64 * 1000, BASE, [], [1.0], category="checks_and_tests")
    c = write_cfg("checks", name, cfg)
    s = train_run(state, c, name, f"fspv2 amendment selftest {tag}")
    log(f"selftest {name}: {'COMPLETE' if s else 'not complete'}")


if __name__ == "__main__":
    main()
