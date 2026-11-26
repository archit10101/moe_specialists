#!/usr/bin/env python
"""csr-context-prior-specialization-v2 driver: every protocol stage sequentially, one guarded GPU job at a time,
idempotently (state: artifacts/context-prior-specialization-v2/driver_state.json). Re-running continues.

  .venv/bin/python cps_driver.py run [--until STAGE] [--only STAGE] [--authorized-resume]

External cancellation (signal to the driver or a child, the V2 STOP marker, a child killed from outside, a foreign GPU
process before or during a job, GPU loss) -> the child stops cooperatively at a step boundary with a verified
checkpoint, state + resume command are recorded, the GPU is released and the driver exits without polling. It refuses to
start while an unresolved stop record or the STOP marker exists unless given --authorized-resume (explicit user
instruction only). No auto-restart, no supervisor. V1 files are only read.
"""
import glob
import json
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
from csr.cps_lib import ART, CFG_DIR, BUDGET, RUN_PREFIX, STOP_MARKER, V1_ART, V1_RUN_PREFIX, protocol_sha  # noqa
from csr import cps_search as S  # noqa: E402
from csr import cps_jsonz as jz  # noqa: E402

STATE = ART / "driver_state.json"
RESUME_CMD = ("cd /home/zapan4/moe-research && source env.sh && nohup .venv/bin/python cps_driver.py run "
              "--authorized-resume >> .tmp/context-prior-specialization-v2/driver.out 2>&1 &")
GPU_LOSS = re.compile(r"no CUDA GPUs|CUDA-capable device|cudaErrorDevicesUnavailable|cudaErrorNoDevice|GPU is lost|"
                      r"device not ready|uncorrectable ECC|Xid|CUDA driver initialization failed|NVML_ERROR_GPU_IS_LOST",
                      re.I)
DRIVER = {"stop": None, "child": None, "active": None, "stage": None}
BUDGET_H = 26.0
STAGE_CAP_H = {"checks_and_tests": 1.0, "screen": 6.0, "s2": 8.0, "finals": 11.0}
SCREEN_SEED = 100
DEV_SEEDS = (100, 101)
FINAL_SEEDS = (0, 1)
TRAIN = {"lr": 6e-4, "min_lr": 6e-5, "warmup_frac": 0.01, "decay_frac": 0.2, "rows_per_step": 64,
         "micro_batch_rows": 32, "weight_decay": 0.1, "betas": [0.9, 0.95], "clip": 1.0, "aux_coeff": 0.01,
         "z_coeff": 0.001, "log_every": 50}  # identical to V1 (TOKEN-4 reference runs)
BACKBONE = {"vocab_size": 50304, "n_layer": 12, "d_model": 384, "n_head": 6, "dense_ffn": 1792, "n_experts": 32,
            "top_k": 4, "expert_ffn": 256, "shared_ffn": 768, "rope_base": 10000.0, "norm_eps": 1e-6, "init_std": 0.02}
# reused V1 TOKEN-4 runs (read-only)
V1_T4 = {"S1": {100: "s1_TOKEN-4_s100"}, "S2": {100: "s2_TOKEN-4_s100", 101: "s2_TOKEN-4_s101"},
         "FINAL": {0: "final_TOKEN-4_s0"}}
V1_DESCRIPTIVE = ["s1_OD-a1.0_s100", "s1_OD-a0.3_s100", "s2_OUTPUT-DISTILL_s100", "s2_OUTPUT-DISTILL_s101",
                  "s2_MEMORY-ONLY_s100", "s2_MEMORY-ONLY_s101", "s1_EDL-b5e-3_s100", "s2_EDL_s100", "s2_EDL_s101"]


class ExternalStop(Exception):
    """External cancellation / GPU yield. Never treated as a software error."""


class Halt(Exception):
    """A preregistered stop condition (insufficient resources, budget, failed gate needing a new implementation)."""


# ------------------------------------------------------------------ state / ledgers
def load_state():
    return json.loads(STATE.read_text()) if STATE.exists() else {"done": {}, "runs": {}, "jobs": {}}


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


def gpu_seconds_used(category=None):
    p = ART / "compute_ledger.tsv"
    s = 0.0
    if p.exists():
        for line in p.read_text().splitlines():
            f = line.split("\t")
            if len(f) >= 7:
                try:
                    if category is None or f[5] == category:
                        s += float(f[2])
                except ValueError:
                    pass
    return s


def budget_ok(projected_s, what="", stage_cat=None):
    used = gpu_seconds_used()
    if used + projected_s > BUDGET_H * 3600:
        log(f"BUDGET: cannot run {what}: used {used / 3600:.2f} h + projected {projected_s / 3600:.2f} h > {BUDGET_H} h")
        return False
    return True


def disk_ok(planned_bytes, what):
    try:
        guard.require_disk(int(planned_bytes), f"cps {what}")
    except guard.GuardError as e:
        raise Halt(f"INSUFFICIENT_RESOURCES (disk): {e}")


# ------------------------------------------------------------------ guarded launch
def _session_pids(sid):
    out = []
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as f:
                fields = f.read().rsplit(")", 1)[1].split()
            if int(fields[3]) == sid:
                out.append(int(d))
        except (OSError, IndexError, ValueError):
            pass
    return out


def _argv(pid):
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return [x.decode(errors="replace") for x in f.read().split(b"\0") if x]
    except OSError:
        return []


def _is_experiment_main(pid):
    a = _argv(pid)
    return len(a) >= 2 and a[0].endswith("/python") and os.path.basename(a[1]).startswith("cps_")


def _forward_stop(sig_name):
    ch = DRIVER["child"]
    if ch is None or ch.poll() is not None:
        return
    for pid in _session_pids(ch.pid):
        if _is_experiment_main(pid):
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
        raise ExternalStop(f"stop marker present before launching {purpose}")
    n = subprocess.run(["nvidia-smi", "-i", "0", "--query-compute-apps=pid", "--format=csv,noheader"],
                       capture_output=True, text=True).stdout.split()
    if n:
        raise ExternalStop(f"WAITING_FOR_GPU: foreign process(es) {n} on GPU 0 before launching {purpose}; "
                           f"nothing launched")


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


def launch(purpose, script, args):
    env = dict(os.environ, MOE_GPU_LAUNCH="1", MOE_RESEARCH_APPROVE="gpu")
    pre_launch_check(purpose)
    DRIVER["active"] = {"stage": DRIVER["stage"], "purpose": purpose, "script": script, "args": args,
                        "launched": guard.now()}
    ch = subprocess.Popen(["scripts/gpu_guarded.sh", purpose, script, *args], cwd=ROOT, env=env,
                          start_new_session=True)
    DRIVER["child"] = ch
    if DRIVER["stop"]:
        _forward_stop("late")
    rc = ch.wait()
    t0 = time.time()
    while _session_pids(ch.pid) and time.time() - t0 < 1200:
        time.sleep(2)
    DRIVER["child"] = None
    return rc, classify_rc(rc)


def job(state, key, script, args, purpose, planned_bytes=50e6, projected_s=600):
    """Idempotent non-training GPU job. Retries a failure once; rc 75/124 = resumable leg."""
    j = state["jobs"].setdefault(key, {"attempts": 0})
    if j.get("status") == "COMPLETE":
        return True
    disk_ok(planned_bytes, key)
    if not budget_ok(projected_s, key):
        raise Halt(f"BUDGET_EXHAUSTED before {key}")
    while True:
        rc, ext = launch(purpose, script, args)
        log(f"job {key}: rc={rc}")
        if ext:
            j["status"] = "EXTERNALLY_STOPPED"
            save_state(state)
            raise ExternalStop(f"{key}: {ext}")
        if rc == 0:
            j["status"] = "COMPLETE"
            DRIVER["active"] = None
            save_state(state)
            return True
        if rc in (75, 124):
            j["legs"] = j.get("legs", 0) + 1
            save_state(state)
            if j["legs"] <= 6:
                continue
        j["attempts"] += 1
        save_state(state)
        if j["attempts"] >= 2:
            j["status"] = "FAILED"
            save_state(state)
            log(f"job {key}: FAILED twice")
            return False


def write_cfg(sub, name, cfg):
    d = CFG_DIR / sub
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.json"
    if p.exists():
        if json.loads(p.read_text()) != cfg:
            raise guard.GuardError(f"config {p} exists with different content")
        return p
    guard.atomic_write_json(p, cfg, mode=0o400)
    return p


def find_run_dir(run_name):
    c = sorted(glob.glob(str(ROOT / f"runs/*_{RUN_PREFIX}{run_name}")))
    return Path(c[-1]) if c else None


def v1_run_dir(run_name):
    c = sorted(glob.glob(str(ROOT / f"runs/*_{V1_RUN_PREFIX}{run_name}")))
    return Path(c[-1]) if c else None


def latest_ckpt(rd):
    if rd is None:
        return None
    from csr.cps_ckpt import ResumeStore
    try:
        return ResumeStore(rd.name, "").latest_verified(log=lambda m: None)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)[:300]}


def mark_external(rec, rd, status, ext):
    ck = latest_ckpt(rd)
    if rd is not None and (status == "RUNNING" or status.startswith("rc")):
        old = json.loads((rd / "status.json").read_text()) if (rd / "status.json").exists() else {}
        guard.atomic_write_json(rd / "status.json", {**old, "status": "EXTERNALLY_STOPPED", "time": guard.now(),
                                                     "reason": f"terminated externally without checkpointing: {ext}",
                                                     "resume_from": ck}, overwrite=True)
    rec["status"] = "EXTERNALLY_STOPPED"
    rec.setdefault("external_stops", []).append({"time": guard.now(), "reason": ext, "resume_from": ck})


def projected_seconds(state, cfg):
    kind = "teacher" if cfg["ccfg"]["teacher"] else ("token" if cfg["ccfg"]["variant"] == "token" else "plain")
    rates = [r["tok_s"] for r in state.get("rates", []) if r["kind"] == kind]
    rate = sorted(rates)[len(rates) // 2] if rates else {"teacher": 75e3, "token": 120e3, "plain": 100e3}[kind]
    return cfg["budget_tokens"] / rate + 300


def train_run(state, cfg_path, run_name, purpose):
    """Launch legs until COMPLETE. Returns summary dict (or None if failed twice)."""
    rec = state["runs"].setdefault(run_name, {"attempts": 0, "legs": 0, "cfg": str(cfg_path.relative_to(ROOT))})
    if rec.get("status") == "COMPLETE":
        return json.loads((ROOT / rec["run_dir"] / "summary.json").read_text())
    cfg = json.loads(cfg_path.read_text())
    if not budget_ok(projected_seconds(state, cfg), run_name):
        raise Halt(f"BUDGET_EXHAUSTED before {run_name}")
    disk_ok(0.25e9 * len(cfg.get("checkpoint_fracs", [1.0])) + 3.0e9, run_name)
    mbr = rec.get("micro_batch_rows")
    while True:
        rd = find_run_dir(run_name)
        args = ["--config", str(cfg_path.relative_to(ROOT)), "--device", "cuda", "--execute"]
        if rd is not None and rec.get("run_dir") == str(rd.relative_to(ROOT)):
            args += ["--run-dir", str(rd.relative_to(ROOT))]
        if mbr:
            args += ["--micro-batch-rows", str(mbr)]
        rc, ext = launch(purpose, "cps_train.py", args)
        rd = find_run_dir(run_name)
        if rc in (3, 78) and rd is not None and rec.get("run_dir") != str(rd.relative_to(ROOT)):
            rd = None
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
            summ = json.loads((rd / "summary.json").read_text())
            kind = "teacher" if cfg["ccfg"]["teacher"] else ("token" if cfg["ccfg"]["variant"] == "token" else "plain")
            state.setdefault("rates", []).append({"kind": kind, "tok_s": summ["seen"] / max(1.0, summ["leg_seconds"]),
                                                  "run": run_name})
            save_state(state)
            if ext:
                raise ExternalStop(f"{run_name}: {ext} (run had completed and was recorded)")
            return summ
        if ext is None and status == "FAILED" and GPU_LOSS.search(st.get("reason", "")):
            ext = "GPU unavailable/lost during run: " + GPU_LOSS.search(st.get("reason", "")).group(0)
        if ext is None and status == "EXTERNALLY_STOPPED" and rc != 124:
            ext = st.get("reason", "EXTERNALLY_STOPPED")
        if ext:
            mark_external(rec, rd, status, ext)
            save_state(state)
            raise ExternalStop(f"{run_name}: {ext}")
        if rc in (75, 124) or status == "INTERRUPTED":
            continue
        if rc == 76 or status == "OOM":
            mbr = max(4, (mbr or 32) // 2)
            rec["micro_batch_rows"] = mbr
            deviation("oom", f"{run_name}: micro-batch -> {mbr} (identical math)")
            continue
        rec["attempts"] += 1
        rec["failures"] = rec.get("failures", []) + [st.get("reason", "")[-1500:]]
        save_state(state)
        if rec["attempts"] >= 2:
            rec["status"] = "FAILED"
            save_state(state)
            log(f"{run_name}: FAILED twice; recorded")
            return None
        rec["failed_dirs"] = rec.get("failed_dirs", []) + [rec.get("run_dir")]
        rec["run_dir"] = None


# ------------------------------------------------------------------ configs
def det_theta(L):
    return json.loads((ART / "detector_calibration.json").read_text())["theta"][str(L)]


def cond_cfg(cond, knobs):
    """cond in TOKEN, MEMORY (M1), CAND (M2/M3/M4 candidates), SHUF (M5), NORESID (M6), TOKEN-COMPUTE."""
    k = knobs or S.BASE
    cc = {"variant": "prior", "teacher": True, "chunk": k["chunk"], "chunk_len": k["chunk_len"],
          "det_theta": det_theta(k["chunk_len"]) if k["chunk"] == "adaptive" else 0.0, "ctx_dim": k["ctx_dim"],
          "mem_hls": k["mem_hls"], "mem_w": 32, "enc_hidden": 128, "inject": k["inject"],
          "alpha_init": k["alpha_init"], "alpha_max": k["alpha_max"], "prior_norm": k["prior_norm"], "fc_dim": 64,
          "horizons": [1], "shuffle_ctx": False}
    c = {"name": cond, "c_util": k["c_util"], "util_mode": k["util_mode"], "c_spec": k["c_spec"], "c_kl": k["c_kl"],
         "lam_max": k["lam_max"], "schedule": k["schedule"], "r_cf": k["r_cf"], "cf_phases": k["cf_phases"],
         "ctx_lr_mult": k["ctx_lr_mult"], "alpha_l1": k["alpha_l1"]}
    if cond in ("TOKEN", "TOKEN-COMPUTE"):
        cc.update(variant="token", teacher=False)
        c = {"name": cond}
    elif cond == "MEMORY":
        cc["teacher"] = False
        c.update(c_util=0.0, c_spec=0.0, c_kl=0.0, lam_max=0.0)
    elif cond == "SHUF":
        cc["shuffle_ctx"] = True
    elif cond == "NORESID":
        cc["variant"] = "noresid"
    return cc, c


def run_cfg(run_name, cond, seed, budget, knobs=None, final=False, candidate=None, category="search"):
    cc, c = cond_cfg(cond, knobs)
    fracs = [1.0]
    if final and cond in ("TOKEN", "CAND"):
        fracs = [0.4, 0.6, 0.8, 1.0]  # MEMORY / TOKEN-COMPUTE finals: p100 only (disk policy, deviation 1)
    return {"experiment": "csr-context-prior-specialization-v2", "run_name": run_name, "kind": "main",
            "condition": cond, "seed": seed, "budget_tokens": int(budget), "backbone": BACKBONE, "ccfg": cc,
            "cond": c, "train": TRAIN, "checkpoint_fracs": fracs, "end_eval": [] if final else ["calib_search"],
            "candidate": candidate, "knobs": knobs, "category": category, "cf_audit_every": 100,
            "protocol_sha256": protocol_sha()}


# ------------------------------------------------------------------ ledgers
def manifest_row(stage, name, cfg, summ, ce=None, note=""):
    p = ART / "run_manifest.tsv"
    hdr = ["time", "stage", "run_name", "condition", "seed", "budget", "candidate", "run_dir", "status", "ce",
           "tokens_per_s", "leg_seconds", "note"]
    new = not p.exists()
    rd = find_run_dir(name)
    with open(p, "a") as f:
        if new:
            f.write("\t".join(hdr) + "\n")
        f.write("\t".join(str(x) for x in [guard.now(), stage, name, cfg["condition"], cfg["seed"],
                                            cfg["budget_tokens"], cfg.get("candidate"),
                                            rd.relative_to(ROOT) if rd else "", summ.get("status") if summ else
                                            "FAILED", ce, summ.get("tokens_per_s") if summ else "",
                                            summ.get("leg_seconds") if summ else "", note]) + "\n")


def candidate_row(rec):
    p = ART / "candidate_table.tsv"
    hdr = ["time", "stage", "id", "seed", "knobs", "reason", "ce_calib_search", "ce_minus_t4", "ce_prior_off",
           "fsi_domain", "fsi_cluster", "mi_domain", "alpha_end", "score", "h_failure", "diagnosis", "run_name"]
    new = not p.exists()
    with open(p, "a") as f:
        if new:
            f.write("\t".join(hdr) + "\n")
        f.write("\t".join(str(x) for x in [guard.now(), rec.get("stage"), rec["id"], rec.get("seed"),
                                            json.dumps(rec["knobs"], sort_keys=True), rec.get("reason"),
                                            rec.get("ce"), rec.get("q", {}).get("ce_minus_t4"),
                                            rec.get("ce_prior_off"), rec.get("fsi_domain"), rec.get("fsi_cluster"),
                                            rec.get("mi_domain"), (rec.get("stats") or {}).get("alpha"),
                                            rec.get("score"), rec.get("h_failure"),
                                            json.dumps(rec.get("q"), default=str), rec.get("run_name")]) + "\n")


# ------------------------------------------------------------------ records
def metrics_lm(rd):
    lm, stats = {}, {}
    p = rd / "metrics.jsonl"
    if p.exists():
        for line in p.read_text().splitlines():
            r = json.loads(line)
            if "lm" in r:
                lm[r["step"]] = r["lm"]
                for k in ("cs_alpha", "cs_prior_top4_overlap", "cs_prior_rel_scale", "cs_prior_entropy"):
                    if r.get(k) is not None:
                        stats.setdefault(k[3:], []).append(r[k])
    end = {k: sum(v[-3:]) / len(v[-3:]) for k, v in stats.items() if v}
    return lm, end


def audit_fields(a):
    return {"fsi_domain": a["domain"]["fsi"], "fsi_cluster": a.get("cluster", {}).get("fsi"),
            "mi_domain": a["domain"]["routing"]["mi_mean"], "mi_cluster": a.get("cluster", {}).get("routing",
                                                                                                 {}).get("mi_mean"),
            "fsi_shuf": a["domain"]["fsi_shuf"], "routing_entropy": a.get("routing_entropy"), "load": a.get("load"),
            "topic_entropy_domain": a["domain"]["routing"]["topic_entropy_mean"]}


def v1_ref(name):
    """Reused V1 run (read-only): CE (CALIB-SEARCH), audit fields, lm curve, CALIB-SELECTION CE if any."""
    rd = v1_run_dir(name)
    s = json.loads((rd / "summary.json").read_text())
    rec = {"run_name": name, "run_dir": str(rd.relative_to(ROOT)), "reused_v1": True,
           "ce": s.get("eval", {}).get("calib_search", {}).get("ce"), "tok_s": s.get("tokens_per_s")}
    if jz.exists(rd / "dev_audit.json"):
        a = jz.load_json(rd / "dev_audit.json")["p100"]
        rec.update(audit_fields(a))
        rec["audit"] = {"load": a["load"], "routing_entropy": a["routing_entropy"]}
    if (rd / "calib_sel_eval.json").exists():
        rec["sel_ce"] = json.loads((rd / "calib_sel_eval.json").read_text())["calib_sel"]["ce"]
    rec["lm"], _ = metrics_lm(rd)
    return rec


def dev_record(name, summ, cfg):
    rd = find_run_dir(name)
    rec = {"run_name": name, "ce": None, "n_steps": None, "teacher": cfg["ccfg"]["teacher"],
           "schedule": cfg["cond"].get("schedule", "40/20/40")}
    if summ is None:
        rec["h_failure"] = True
        return rec
    ev = summ.get("eval", {})
    rec["ce"] = ev.get("calib_search", {}).get("ce")
    rec["ce_prior_off"] = ev.get("calib_search_prior_off", {}).get("ce")
    rec["n_steps"] = summ.get("n_steps")
    rec["tok_s"] = summ.get("tokens_per_s")
    rec["lm"], rec["stats"] = metrics_lm(rd)
    ap = ART / "evals" / rd.name / "dev_audit.json"
    if jz.exists(ap):
        a = jz.load_json(ap)["p100"]
        rec.update(audit_fields(a))
        rec["audit"] = {"load": a["load"], "routing_entropy": a["routing_entropy"]}
        rec["prior_off_meas"] = a.get("prior_off_meas")
    au_ = summ.get("cfaudit") or []
    if au_:
        rec["cfaudit_rho"] = float(sum(x["rho_fo_exact"] for x in au_) / len(au_))
    rec["h_failure"] = bool(S.h_failure(rec))
    return rec


def dev_run(state, stage, name, cond, seed, budget, knobs=None, candidate=None):
    cfg = run_cfg(name, cond, seed, budget, knobs, candidate=candidate, category=stage if stage != "s1" else "screen")
    p = write_cfg(stage, name, cfg)
    summ = train_run(state, p, name, f"cps {stage} {name}")
    rd = find_run_dir(name)
    if summ is not None and rd is not None:
        job(state, f"devaudit_{name}", "cps_evaluate.py", ["dev-audit", "--run-dir", str(rd.relative_to(ROOT)),
                                                           "--device", "cuda", "--execute"],
            f"cps dev audit {name}", projected_s=600)
    if not state["runs"][name].get("manifested"):
        ce = summ.get("eval", {}).get("calib_search", {}).get("ce") if summ else None
        manifest_row(stage, name, cfg, summ or {}, ce)
        state["runs"][name]["manifested"] = True
        save_state(state)
    rec = dev_record(name, summ, cfg)
    rec["category"] = cfg["category"]
    if stage == "s1" and state["jobs"].get(f"devaudit_{name}", {}).get("status") == "COMPLETE":
        drop_milestones(state, name, why="screen run: p100 evaluated (dev audit); disk policy (deviation 1)")
    return rec


def drop_milestones(state, run_name, keep=(), why=""):
    """Disk policy (deviation 1): delete a finished V2 run's milestone directories after their evaluations, keeping
    sha256 / step / path in the checkpoint ledger and the driver state (all metrics and audits are kept)."""
    import shutil
    from csr.cps_ckpt import ledger
    rd = find_run_dir(run_name)
    if rd is None or not (rd / "summary.json").exists():
        return
    summ = json.loads((rd / "summary.json").read_text())
    for tag, e in summ.get("ckpts", {}).items():
        p = ROOT / e["path"]
        if tag in keep or not p.exists():
            continue
        shutil.rmtree(p)
        ledger({"time": guard.now(), "run_id": rd.name, "run_name": run_name, "kind": "milestone_deleted_disk",
                "path": e["path"], "step": e["step"], "seen": e["seen"], "sha256": e["sha256"], "note": why})
        state.setdefault("deleted_milestones", []).append({"run": run_name, "tag": tag, "sha256": e["sha256"],
                                                           "why": why, "time": guard.now()})
    save_state(state)


def compact(rec):
    return {k: v for k, v in rec.items() if k not in ("lm", "audit")}


def write_diag_report(rec, ref):
    d = ART / "diagnostics"
    d.mkdir(exist_ok=True)
    guard.atomic_write_json(d / f"{rec['run_name']}.json", {"record": compact(rec), "t4": compact(ref)},
                            overwrite=True)
    q = rec.get("q") or {}
    lines = [f"# {rec.get('id', rec['run_name'])} ({rec['run_name']})", "",
             f"- knobs: `{json.dumps(rec.get('knobs'), sort_keys=True)}`", f"- reason: {rec.get('reason')}",
             f"- CALIB-SEARCH CE {rec.get('ce')} (TOKEN-4 {ref.get('ce')}); prior disabled {rec.get('ce_prior_off')}",
             f"- FSI domain {rec.get('fsi_domain')} (TOKEN-4 {ref.get('fsi_domain')}), cluster {rec.get('fsi_cluster')}"
             f" (TOKEN-4 {ref.get('fsi_cluster')}), MI domain {rec.get('mi_domain')} (TOKEN-4 {ref.get('mi_domain')})",
             f"- end-of-run routing stats: {rec.get('stats')}", f"- score {rec.get('score')}, H {rec.get('h_failure')}"]
    for k, v in q.items():
        lines.append(f"- {k}: `{json.dumps(v, default=str)}`")
    (d / f"{rec['run_name']}.md").write_text("\n".join(lines) + "\n")


# ------------------------------------------------------------------ stages
def stage_prereg(state):
    if state["done"].get("prereg"):
        return
    for f in ("reuse_manifest.json", "detector_calibration.json"):
        if not (ART / f).exists():
            raise Halt(f"prereg: {f} missing (run cps_verify_reuse.py / cps_detector_calib.py first)")
    rm = json.loads((ART / "reuse_manifest.json").read_text())
    if not rm.get("all_ok"):
        raise Halt("prereg: reuse verification failed (hash mismatch)")
    ss = {"BASE": S.BASE, "SPACE": S.SPACE, "WAVE_A": S.WAVE_A, "RULES": S.RULES, "TH": S.TH,
          "MAX_CANDIDATES": S.MAX_CANDIDATES, "code_sha256": guard.sha256_file(ROOT / "src/csr/cps_search.py")}
    guard.atomic_write_json(ART / "search_space.json", ss, mode=0o400)
    rec = {"protocol_sha256": protocol_sha(), "search_space_sha256": guard.sha256_file(ART / "search_space.json"),
           "detector_calibration_sha256": guard.sha256_file(ART / "detector_calibration.json"),
           "reuse_manifest_sha256": guard.sha256_file(ART / "reuse_manifest.json"),
           "code": {f: guard.sha256_file(ROOT / f) for f in
                    ["src/csr/cps_model.py", "src/csr/cps_search.py", "src/csr/cps_audit.py", "src/csr/cps_lib.py",
                     "src/csr/cps_ckpt.py", "cps_train.py", "cps_evaluate.py", "cps_driver.py"]},
           "time": guard.now()}
    guard.atomic_write_json(ART / "preregistration_hashes.json", rec, mode=0o400)
    log(f"prereg: {json.dumps(rec)[:600]}")
    state["done"]["prereg"] = True
    save_state(state)


def stage_tests(state):
    env = {k: v for k, v in os.environ.items()}
    env["CUDA_VISIBLE_DEVICES"] = ""
    r = subprocess.run([str(ROOT / ".venv/bin/python"), "cps_tests.py"], cwd=ROOT, env=env, capture_output=True,
                       text=True)
    log(f"tests: rc={r.returncode} {r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-500:]}")
    if r.returncode != 0:
        raise Halt("CPU suite failed; fix implementation bugs before continuing")


def stage_checks(state):
    if not job(state, "gpu_checks", "cps_gpu_checks.py", ["--device", "cuda", "--execute"], "cps gpu checks",
               projected_s=900):
        raise Halt("GPU checks failed")
    if not job(state, "equiv_check", "cps_equiv_check.py", ["--execute"], "cps trainer equivalence",
               planned_bytes=4e9, projected_s=900):
        raise Halt("trainer equivalence (V2 token variant vs reused V1 TOKEN-4) failed")
    smoke = [("smoke_C03", "CAND", S.BASE), ("smoke_M1", "MEMORY", S.BASE), ("smoke_M5", "SHUF", S.BASE),
             ("smoke_M6", "NORESID", S.BASE), ("smoke_ADAPT", "CAND", S.apply(S.BASE, {"chunk": "adaptive"}))]
    for name, cond, k in smoke:
        if state["runs"].get(name, {}).get("status") == "COMPLETE":
            continue
        cfg = run_cfg(name, cond, 100, 40 * 63_000, k, category="checks_and_tests")
        cfg["end_eval"] = []
        cfg["cf_audit_every"] = 10
        cfg["checkpoint_fracs"] = []  # disk policy (deviation 1): smoke runs keep no milestone
        p = write_cfg("smoke", name, cfg)
        if train_run(state, p, name, f"cps smoke {name}") is None:
            raise Halt(f"smoke run {name} failed: fix implementation bugs before continuing")
    if not job(state, "resume_check", "cps_resume_check.py", ["--execute"], "cps resume check", planned_bytes=8e9,
               projected_s=1200):
        raise Halt("resume check failed")


def refs(state):
    if "refs" not in state:
        r = {"S1": {s: v1_ref(n) for s, n in V1_T4["S1"].items()}, "S2": {s: v1_ref(n) for s, n in V1_T4["S2"].items()}}
        r["descriptive"] = {n: compact(v1_ref(n)) for n in V1_DESCRIPTIVE}
        state["refs"] = {"S1": {str(k): v for k, v in r["S1"].items()}, "S2": {str(k): v for k, v in r["S2"].items()},
                         "descriptive": r["descriptive"]}
        save_state(state)
    return state["refs"]


def stage_screen(state):
    state.setdefault("dev", {})
    R = refs(state)
    ref = R["S1"][str(SCREEN_SEED)]
    t4_lm = {int(k): v for k, v in ref["lm"].items()}
    cands = state.setdefault("cands", [])

    def run_cand(cid, knobs, why):
        name = f"s1_{cid}_s{SCREEN_SEED}"
        log(f"screen candidate {cid}: {why}: {json.dumps(knobs, sort_keys=True)}")
        rec = dev_run(state, "s1", name, "CAND", SCREEN_SEED, BUDGET["S1"], knobs, candidate=cid)
        rec.update(id=cid, knobs=knobs, reason=why, stage="s1", seed=SCREEN_SEED)
        finish(rec)
        cands.append(compact(rec) | {"lm": rec.get("lm", {})})
        save_state(state)

    def finish(rec):
        if rec.get("ce") is not None and rec.get("fsi_domain") is not None:
            lmr = {int(k): v for k, v in rec["lm"].items()}
            ctrl = state["dev"]
            m1 = compact(ctrl["s1_M1_s100"]) if "s1_M1_s100" in ctrl else None
            m5 = compact(ctrl["s1_M5_s100"]) if "s1_M5_s100" in ctrl else None
            rec["q"] = S.diagnose({**rec, "lm": lmr}, ref, t4_lm, m1, m5)
            rec["score"] = S.score(rec, ref)
        candidate_row(rec)
        write_diag_report(rec, ref)

    have = {c["id"] for c in cands}
    for cid, why, ch in S.WAVE_A:
        if cid not in have:
            run_cand(cid, S.apply(S.BASE, ch), f"wave A: {why}")
    base_c03 = [c for c in cands if c["id"] == "C03"][0]["knobs"]
    for name, cond, k, why in [("s1_M1_s100", "MEMORY", S.BASE, "M1 memory-only control"),
                               ("s1_M5_s100", "SHUF", base_c03, "M5 shuffled-context control (C03 knobs)"),
                               ("s1_M6_s100", "NORESID", base_c03, "M6 no-token-residual diagnostic (C03 knobs)")]:
        if name not in state["dev"]:
            rec = dev_run(state, "s1", name, cond, SCREEN_SEED, BUDGET["S1"], k, candidate=name.split("_")[1])
            rec.update(id=name.split("_")[1], knobs=k, reason=why, stage="s1", seed=SCREEN_SEED)
            if rec.get("ce") is not None and rec.get("fsi_domain") is not None:
                rec["score"] = S.score(rec, ref)
            candidate_row(rec)
            write_diag_report(rec, ref)
            state["dev"][name] = compact(rec)
            save_state(state)
    while len(cands) < S.MAX_CANDIDATES:
        if gpu_seconds_used("screen") > STAGE_CAP_H["screen"] * 3600:
            log("screen GPU allocation used; stopping candidate generation")
            break
        for c in cands:  # refresh diagnoses (controls may have finished after wave A)
            if c.get("ce") is not None and c.get("fsi_domain") is not None:
                lmr = {int(k): v for k, v in c["lm"].items()}
                c["q"] = S.diagnose({**c, "lm": lmr}, ref, t4_lm, compact(state["dev"]["s1_M1_s100"]),
                                    compact(state["dev"]["s1_M5_s100"]))
                c["score"] = S.score(c, ref)
        inc = S.incumbent(cands, ref)
        if inc is None:
            raise Halt("screen: no finished candidate")
        tried = {S.key(c["knobs"]) for c in cands}
        rid, ch = S.next_change(inc["knobs"], inc["q"], tried)
        if rid is None:
            log("decision tree exhausted for the incumbent; stopping candidate generation")
            break
        cid = f"C{len(cands) + 1:02d}"
        run_cand(cid, S.apply(inc["knobs"], ch), f"incumbent {inc['id']} {rid} -> {json.dumps(ch)}")
    state["done"]["screen"] = True
    save_state(state)


def stage_s2(state):
    R = refs(state)
    ref1 = R["S1"][str(SCREEN_SEED)]
    cands = state["cands"]
    if "advanced" not in state:
        state["advanced"] = S.advance(cands, ref1)
        log(f"S2 advancement: {state['advanced']}")
        save_state(state)
    if not state["advanced"]:
        raise Halt("S2: nothing to advance (all screen runs failed)")
    s2 = state.setdefault("s2", {})
    top = state["advanced"][0][0]
    top_k = [c for c in cands if c["id"] == top][0]["knobs"]
    jobs_ = [(cid, "CAND", [c for c in cands if c["id"] == cid][0]["knobs"]) for cid, _ in state["advanced"]]
    jobs_ += [("M1", "MEMORY", top_k), ("M5", "SHUF", top_k)]
    for cid, cond, k in jobs_:
        for seed in DEV_SEEDS:
            name = f"s2_{cid}_s{seed}"
            if name in s2:
                continue
            rec = dev_run(state, "s2", name, cond, seed, BUDGET["S2"], k, candidate=cid)
            rec.update(id=cid, knobs=k, stage="s2", seed=seed, reason="S2 survivor" if cond == "CAND" else
                       f"{cid} control on {top} knobs")
            reft = R["S2"][str(seed)]
            if rec.get("ce") is not None and rec.get("fsi_domain") is not None:
                rec["score"] = S.score(rec, reft)
            candidate_row(rec)
            write_diag_report(rec, reft)
            s2[name] = compact(rec)
            save_state(state)
    for cid, el in state["advanced"]:
        by = {str(s): s2.get(f"s2_{cid}_s{s}") for s in DEV_SEEDS}
        ok = all(v and v.get("ce") is not None and v.get("fsi_domain") is not None for v in by.values())
        state.setdefault("s2_eligible", {})[cid] = bool(ok and S.s2_eligible(by, R["S2"]))
    log(f"S2 eligibility: {state['s2_eligible']}")
    state["done"]["s2"] = True
    save_state(state)


def stage_select(state):
    if (ART / "frozen.json").exists() or (ART / "NEGATIVE_DEV.json").exists():
        return
    R = refs(state)
    el = [cid for cid, ok in state["s2_eligible"].items() if ok]
    if not el:
        out = {"time": guard.now(), "result": "NEGATIVE (development)", "advanced": state["advanced"],
               "s2_eligible": state["s2_eligible"],
               "note": "no candidate met the S2 development criteria; CALIB-SELECTION and VAL stay untouched"}
        guard.atomic_write_json(ART / "NEGATIVE_DEV.json", out, mode=0o400)
        for name in list(state.get("s2", {})):
            drop_milestones(state, name, why="development NEGATIVE; disk policy (deviation 1)")
        log("SELECT: no S2-eligible candidate -> NEGATIVE at development; validation not opened")
        return
    if not (ART / "STAGE3_OPEN.json").exists():
        guard.atomic_write_json(ART / "STAGE3_OPEN.json", {"time": guard.now(), "candidates": el}, mode=0o400)
        log(f"stage 3: CALIB-SELECTION opened for {el}")
    s2 = state["s2"]
    for cid in el:
        for seed in DEV_SEEDS:
            name = f"s2_{cid}_s{seed}"
            rd = find_run_dir(name)
            if not job(state, f"calibsel_{name}", "cps_evaluate.py", ["calib-sel", "--run-dir",
                                                                      str(rd.relative_to(ROOT)), "--device", "cuda",
                                                                      "--execute"], f"cps calib-sel {name}",
                       projected_s=300):
                raise Halt(f"calib-sel evaluation failed for {name}")
            s2[name]["sel_ce"] = json.loads((ART / "evals" / rd.name / "calib_sel_eval.json").read_text()
                                            )["calib_sel"]["ce"]
            save_state(state)
    t4_sel = {str(s): R["S2"][str(s)]["sel_ce"] for s in DEV_SEEDS}
    pool = []
    for cid in el:
        pool.append({"id": cid, "s2_eligible": True,
                     "sel_ce": {str(s): s2[f"s2_{cid}_s{s}"]["sel_ce"] for s in DEV_SEEDS},
                     "fsi_mean": sum(s2[f"s2_{cid}_s{s}"]["fsi_domain"] for s in DEV_SEEDS) / len(DEV_SEEDS),
                     "knobs": s2[f"s2_{cid}_s{DEV_SEEDS[0]}"]["knobs"]})
    best, why = S.select(pool, t4_sel)
    fz = {"time": guard.now(), "selected": best["id"], "knobs": best["knobs"], "rule": why, "pool": pool,
          "t4_sel": t4_sel}
    guard.atomic_write_json(ART / "frozen.json", fz, mode=0o400)
    (ART / "frozen.sha256").write_text(guard.sha256_file(ART / "frozen.json") + "  frozen.json\n")
    log(f"FROZEN: {best['id']} ({why}) knobs={json.dumps(best['knobs'], sort_keys=True)}")
    for name in list(state.get("s2", {})):
        if not name.startswith(f"s2_{best['id']}_"):
            drop_milestones(state, name, why="S2 run not frozen; selection complete; disk policy (deviation 1)")


def final_run(state, cond, seed, budget=None, fracs=None):
    fz = json.loads((ART / "frozen.json").read_text())
    name = f"final_{cond}_s{seed}"
    knobs = fz["knobs"]
    cfg = run_cfg(name, cond, seed, budget or BUDGET["FINAL"], knobs, final=True, candidate=fz["selected"],
                  category="finals")
    if fracs is None and cond in ("MEMORY", "TOKEN-COMPUTE") and not (find_run_dir(name) or None):
        fracs = [1.0]  # deviation 1 as written: MEMORY-ONLY and TOKEN-COMPUTE finals keep p100 only (deviation 6)
    if fracs is not None:
        cfg["checkpoint_fracs"] = fracs
    p = write_cfg("final", name, cfg)
    summ = train_run(state, p, name, f"cps final {name}")
    rd = find_run_dir(name)
    if summ is not None:
        job(state, f"final_eval_{name}", "cps_evaluate.py", ["final", "--run-dir", str(rd.relative_to(ROOT)),
                                                              "--device", "cuda", "--execute"],
            f"cps final eval {name}", projected_s=2400)
        if state["jobs"].get(f"final_eval_{name}", {}).get("status") == "COMPLETE":
            jz.compress(ART / "evals" / rd.name / "final_eval.json", "final eval written; disk policy (deviation 6)")
            if cond in ("MEMORY", "TOKEN-COMPUTE"):
                drop_milestones(state, name, keep=("p100",),
                                why="MEMORY/TOKEN-COMPUTE final: keep p100 only, after final eval (deviation 1/6)")
    if not state["runs"][name].get("manifested"):
        manifest_row("final", name, cfg, summ or {}, None)
        state["runs"][name]["manifested"] = True
        save_state(state)
    return summ


def stage_finals(state):
    if (ART / "NEGATIVE_DEV.json").exists():
        log("finals skipped: development NEGATIVE (no frozen candidate)")
        return
    # reused V1 TOKEN-4 seed 0 (V1 run directory read-only; V2 evaluation output in the V2 artifact directory)
    rd0 = v1_run_dir(V1_T4["FINAL"][0])
    if not job(state, "final_eval_TOKEN-4_s0_v1", "cps_evaluate.py", ["final", "--run-dir", str(rd0.relative_to(ROOT)),
                                                                       "--device", "cuda", "--execute"],
               "cps final eval reused V1 final_TOKEN-4_s0", projected_s=2400):
        raise Halt("final eval of reused TOKEN-4 s0 failed")
    for seed in FINAL_SEEDS:
        final_run(state, "CAND", seed)
    final_run(state, "TOKEN", 1)
    for seed in FINAL_SEEDS:
        final_run(state, "MEMORY", seed)
    if "tc_budget" not in state:
        secs = []
        for s in FINAL_SEEDS:
            rd = find_run_dir(f"final_CAND_s{s}")
            secs.append(sum(json.loads(p.read_text())["leg_seconds"] for p in sorted(rd.glob("leg_*.json"))))
        t4r = [json.loads((v1_run_dir(V1_T4["FINAL"][0]) / "summary.json").read_text())["tokens_per_s"],
               json.loads((find_run_dir("final_TOKEN_s1") / "summary.json").read_text())["tokens_per_s"]]
        cache_s = 0.0
        for line in (V1_ART / "compute_ledger.tsv").read_text().splitlines()[1:]:
            f = line.split("\t")
            if len(f) >= 7 and f[5] == "forecast_cache" and "main" in f[4]:
                cache_s += float(f[2])
        draft_s = 877.1
        target = sum(secs) / len(secs) + (cache_s + draft_s) / len(FINAL_SEEDS)
        tok = min(int(target * sum(t4r) / len(t4r)), 600_000_000)
        state["tc_budget"] = {"target_gpu_s": target, "cand_run_gpu_s": secs, "cache_main_gpu_s": cache_s,
                              "draft_gpu_s": draft_s, "t4_tok_s": t4r, "tokens": tok,
                              "rule": "mean frozen-candidate final training GPU-s + (V1 main forecast cache + draft "
                                      "training GPU-s) / 2 seeds, times the TOKEN-4 training throughput"}
        log(f"TOKEN-COMPUTE budget: {state['tc_budget']}")
        save_state(state)
    for seed in FINAL_SEEDS:
        final_run(state, "TOKEN-COMPUTE", seed, state["tc_budget"]["tokens"])


def stage_sensitivity(state):
    """Deviation 4: V2-code TOKEN-4 seed 0 (p100 only) as a conservative second seed-0 comparator."""
    if (ART / "NEGATIVE_DEV.json").exists():
        return
    final_run(state, "TOKEN", 0, fracs=[1.0])


def stage_bench(state):
    if (ART / "NEGATIVE_DEV.json").exists():
        return
    job(state, "bench", "cps_bench.py", ["--device", "cuda", "--execute"], "cps bench", projected_s=900)


def stage_analyze(state):
    r = subprocess.run([str(ROOT / ".venv/bin/python"), "cps_analyze.py"], cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    log(f"analyze rc={r.returncode} {r.stdout[-800:]} {r.stderr[-800:]}")


STAGES = [("prereg", stage_prereg), ("tests", stage_tests), ("checks", stage_checks), ("screen", stage_screen),
          ("s2", stage_s2), ("select", stage_select), ("finals", stage_finals), ("sensitivity", stage_sensitivity),
          ("bench", stage_bench), ("analyze", stage_analyze)]


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
    kind = "WAITING_FOR_GPU" if "WAITING_FOR_GPU" in reason else "EXTERNALLY_STOPPED"
    state["external_stop"] = {"status": kind, "time": guard.now(), "reason": reason, "stage": DRIVER["stage"],
                              "active": act, "run_dir": str(rd.relative_to(ROOT)) if rd else None,
                              "resume_from_checkpoint": latest_ckpt(rd) if rd else None,
                              "gpu0_pids_after_yield": gpu, "experiment_pids_on_gpu_after_yield": mine,
                              "resume_command": RESUME_CMD,
                              "note": "do not resume without an explicit user instruction"}
    DRIVER["active"] = None
    save_state(state)
    log(f"{kind} at stage {DRIVER['stage']}: {reason}; GPU0 pids now {gpu} (experiment-owned: {mine}); waiting for "
        f"explicit user authorization. Resume: {RESUME_CMD}")


def main():
    guard.check_root()
    until = sys.argv[sys.argv.index("--until") + 1] if "--until" in sys.argv else None
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    authorized = "--authorized-resume" in sys.argv
    ART.mkdir(parents=True, exist_ok=True)
    state = load_state()
    state.setdefault("jobs", {})
    if STOP_MARKER.exists():
        log("REFUSING to start: stop marker exists (remove it only on user instruction)")
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
        for name, fn in STAGES:
            if only and name != only:
                continue
            if state["done"].get(name) and name not in ("tests", "analyze"):
                continue
            DRIVER["stage"] = name
            log(f"== stage {name} (GPU used so far {gpu_seconds_used() / 3600:.2f} h)")
            fn(state)
            if DRIVER["stop"]:
                raise ExternalStop(DRIVER["stop"])
            state["done"][name] = True
            save_state(state)
            if until == name:
                break
        log("driver finished requested stages")
    except ExternalStop as e:
        record_external_stop(state, str(e))
        sys.exit(0)
    except Halt as e:
        state["halt"] = {"time": guard.now(), "stage": DRIVER["stage"], "reason": str(e)}
        save_state(state)
        log(f"HALT at stage {DRIVER['stage']}: {e}")
        sys.exit(4)


if __name__ == "__main__":
    main()
