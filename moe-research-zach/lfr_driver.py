#!/usr/bin/env python
"""csr-latent-forecast-router-distill-v1 driver: every protocol stage sequentially, one guarded GPU job at a time,
idempotently (state: artifacts/latent-forecast-router-distill-v1/driver_state.json). Re-running continues.

  .venv/bin/python lfr_driver.py run [--until STAGE] [--only STAGE] [--authorized-resume]

External cancellation (signal to the driver or a child, the STOP marker, a child killed from outside, a foreign GPU
process before or during a job, GPU loss) -> the child stops cooperatively at a step boundary with a verified
checkpoint, state + resume command are recorded, the driver exits without polling. It refuses to start while an
unresolved stop record or the STOP marker exists unless given --authorized-resume (explicit user instruction only).
No auto-restart, no supervisor.
"""
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
from csr.lfr_lib import ART, CFG_DIR, BUDGET, RUN_PREFIX, STOP_MARKER, protocol_sha  # noqa: E402
from csr import lfr_search as S  # noqa: E402

STATE = Path(os.environ["LFR_DRIVER_STATE"]) if os.environ.get("LFR_DRIVER_STATE") else ART / "driver_state.json"
RESUME_CMD = ("cd /home/zapan4/moe-research && source env.sh && nohup .venv/bin/python lfr_driver.py run "
              "--authorized-resume >> .tmp/latent-forecast-router-distill-v1/driver.out 2>&1 &")
GPU_LOSS = re.compile(r"no CUDA GPUs|CUDA-capable device|cudaErrorDevicesUnavailable|cudaErrorNoDevice|GPU is lost|"
                      r"device not ready|uncorrectable ECC|Xid|CUDA driver initialization failed|NVML_ERROR_GPU_IS_LOST",
                      re.I)
DRIVER = {"stop": None, "child": None, "active": None, "stage": None}
BUDGET_H = 40.0
STAGE_CAP_H = {"cache": 4.0, "search1": 4.5, "search2": 7.5}
DEV_SEEDS = (100, 101)
FINAL_SEEDS = (0, 1)
TRAIN = {"lr": 6e-4, "min_lr": 6e-5, "warmup_frac": 0.01, "decay_frac": 0.2, "rows_per_step": 64,
         "micro_batch_rows": 32, "weight_decay": 0.1, "betas": [0.9, 0.95], "clip": 1.0, "aux_coeff": 0.01,
         "z_coeff": 0.001, "log_every": 50}
EDL_BETAS = {"EDL-b5e-4": 5e-4, "EDL-b5e-3": 5e-3, "EDL-b2e-2": 2e-2}
OD_ALPHAS = {"OD-a0.3": 0.3, "OD-a1.0": 1.0}
BACKBONE = {"vocab_size": 50304, "n_layer": 12, "d_model": 384, "n_head": 6, "dense_ffn": 1792, "n_experts": 32,
            "top_k": 4, "expert_ffn": 256, "shared_ffn": 768, "rope_base": 10000.0, "norm_eps": 1e-6, "init_std": 0.02}


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


def budget_ok(projected_s, what=""):
    used = gpu_seconds_used()
    if used + projected_s > BUDGET_H * 3600:
        log(f"BUDGET: cannot run {what}: used {used / 3600:.2f} h + projected {projected_s / 3600:.2f} h > {BUDGET_H} h")
        return False
    return True


def disk_ok(planned_bytes, what):
    try:
        guard.require_disk(int(planned_bytes), f"lfr {what}")
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


def _is_experiment_main(pid, script=None):
    a = _argv(pid)
    return len(a) >= 2 and a[0].endswith("/python") and os.path.basename(a[1]).startswith("lfr_") and \
        (script is None or os.path.basename(a[1]) == os.path.basename(script))


def _forward_stop(sig_name):
    ch = DRIVER["child"]
    if ch is None or ch.poll() is not None:
        return
    for pid in _session_pids(ch.pid):  # every experiment-owned python main in the child session (incl. nested legs)
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


def launch(purpose, script, args, defer_external=False):
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
    ext = classify_rc(rc)
    if defer_external:
        return rc, ext
    if ext:
        raise ExternalStop(f"{purpose}: {ext}")
    DRIVER["active"] = None
    return rc


def job(state, key, script, args, purpose, planned_bytes=50e6, projected_s=600):
    """Idempotent non-training GPU job (cache, labeler, diagnostics, evaluation). Retries a failure once."""
    j = state["jobs"].setdefault(key, {"attempts": 0})
    if j.get("status") == "COMPLETE":
        return True
    disk_ok(planned_bytes, key)
    if not budget_ok(projected_s, key):
        raise Halt(f"BUDGET_EXHAUSTED before {key}")
    while True:
        rc, ext = launch(purpose, script, args, defer_external=True)
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
        if rc in (75, 124):  # internal leg time limit: resumable continuation (cursor-based jobs), not a failure
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


def latest_ckpt(rd):
    if rd is None:
        return None
    from csr.lfr_ckpt import ResumeStore
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


def projected_seconds(cfg):
    """Projection from measured throughput of completed runs of the same kind (else conservative defaults)."""
    kind = "teacher" if cfg["model"]["teacher"] else ("od" if cfg["cond"].get("od_alpha") else "plain")
    rates = [r["tok_s"] for r in STATE_CACHE.get("rates", []) if r["kind"] == kind]
    rate = sorted(rates)[len(rates) // 2] if rates else {"teacher": 70e3, "od": 120e3, "plain": 140e3}[kind]
    return cfg["budget_tokens"] / rate + 240


STATE_CACHE = {}


def train_run(state, cfg_path, run_name, purpose):
    """Launch legs until COMPLETE. Returns summary dict (or None if failed twice)."""
    rec = state["runs"].setdefault(run_name, {"attempts": 0, "legs": 0, "cfg": str(cfg_path.relative_to(ROOT))})
    if rec.get("status") == "COMPLETE":
        return json.loads((ROOT / rec["run_dir"] / "summary.json").read_text())
    cfg = json.loads(cfg_path.read_text())
    proj = projected_seconds(cfg)
    if not budget_ok(proj, run_name):
        raise Halt(f"BUDGET_EXHAUSTED before {run_name}")
    fr = cfg.get("checkpoint_fracs", [1.0])
    disk_ok(0.25e9 * len(fr) + 3.0e9, run_name)
    mbr = rec.get("micro_batch_rows")
    while True:
        rd = find_run_dir(run_name)
        args = ["--config", str(cfg_path.relative_to(ROOT)), "--device", "cuda", "--execute"]
        if rd is not None and rec.get("run_dir") == str(rd.relative_to(ROOT)):
            args += ["--run-dir", str(rd.relative_to(ROOT))]
        if mbr:
            args += ["--micro-batch-rows", str(mbr)]
        rc, ext = launch(purpose, "lfr_train.py", args, defer_external=True)
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
            kind = "teacher" if cfg["model"]["teacher"] else ("od" if cfg["cond"].get("od_alpha") else "plain")
            state.setdefault("rates", []).append({"kind": kind, "tok_s": summ["seen"] / max(1.0, summ["leg_seconds"]),
                                                  "run": run_name})
            STATE_CACHE["rates"] = state["rates"]
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
def model_cfg(router="linear", teacher=False, knobs=None):
    m = dict(BACKBONE)
    m.update({"router": router, "teacher": teacher})
    if knobs:
        m.update({"mem": knobs["mem"], "mem_halflife": knobs["mem_halflife"], "agg": knobs["agg"],
                  "fc_dim": knobs["fc_dim"], "horizons": knobs["horizons"], "inject": knobs["inject"],
                  "tau_t": knobs["tau_t"], "kappa": knobs["kappa"]})
    return m


def cond_cfg(name, knobs=None, edl_beta=0.0, od_alpha=0.0, schedule="40/20/40"):
    """Condition -> (model cfg, cond dict). name in TOKEN-4, EDL, OUTPUT-DISTILL, CF-FORECAST-STUDENT,
    FORECAST-SIMILARITY, SHUFFLED-FORECAST, EDL+CF, MEMORY-ONLY, TOKEN-COMPUTE."""
    if name in ("TOKEN-4", "TOKEN-COMPUTE"):
        return model_cfg(), {"name": name}
    if name == "EDL":
        return model_cfg(), {"name": name, "edl_beta": edl_beta}
    if name == "OUTPUT-DISTILL":
        return model_cfg(), {"name": name, "od_alpha": od_alpha, "schedule": schedule}
    k = knobs
    if name == "MEMORY-ONLY":
        return model_cfg("student", False, k), {"name": name, "router_lr_mult": k["router_lr_mult"]}
    c = {"name": name, "cf": True, "util_mode": k["util_mode"], "c_util": k["c_util"], "c_sutil": k["c_sutil"],
         "c_kl": k["c_kl"], "c_lc": k["c_lc"], "r_cf": k["r_cf"], "schedule": k["schedule"],
         "router_lr_mult": k["router_lr_mult"]}
    m = model_cfg("student", True, k)
    if name == "FORECAST-SIMILARITY":
        c.update(cf=False, c_util=0.0, c_sutil=0.0, c_lc=0.0)
        m["kappa"] = 0.0
    elif name == "SHUFFLED-FORECAST":
        c["shuffle"] = True
    elif name == "EDL+CF":
        c["edl_beta"] = edl_beta
    return m, c


def diag_cfg(m, c, final=False):
    if m["teacher"]:
        a_, b_, _ = [int(v) for v in c["schedule"].split("/")]
        return {"at": {"p1end": a_ / 100, "p2end": (a_ + b_) / 100}, "at_end": ["end"]}
    return {"at": {"p1end_40": 0.4, "p1end_30": 0.3}, "at_end": ["end"]}


def run_cfg(run_name, cond_name, seed, budget, knobs=None, edl_beta=0.0, od_alpha=0.0, schedule="40/20/40",
            final=False, candidate=None, category="search"):
    m, c = cond_cfg(cond_name, knobs, edl_beta, od_alpha, schedule)
    fracs = [1.0]
    if final:
        fracs = {"TOKEN-4": [0.4, 0.6, 0.8, 1.0], "EDL": [0.4, 0.6, 0.8, 1.0], "CF-FORECAST-STUDENT":
                 [0.4, 0.6, 0.8, 1.0], "EDL+CF": [0.4, 0.6, 0.8, 1.0]}.get(cond_name, [0.4, 1.0])
        if cond_name == "TOKEN-COMPUTE":
            fracs = [1.0]
    return {"experiment": "csr-latent-forecast-router-distill-v1", "run_name": run_name, "kind": "main",
            "condition": cond_name, "seed": seed, "budget_tokens": int(budget), "model": m, "cond": c,
            "train": TRAIN, "checkpoint_fracs": fracs, "diag": diag_cfg(m, c, final),
            "end_eval": [] if final else ["calib_search"], "candidate": candidate, "category": category,
            "cf_audit_every": 100, "protocol_sha256": protocol_sha(),
            "cache_manifest_sha256": cache_sha()}


def cache_sha():
    p = ART / "cache_manifest_main.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text())
    return guard.sha256_json([c["sha256"] for c in m["chunks"]]) if m.get("chunks") else None


def all_condition_configs_for_test():
    k = S.apply(S.BASE, {})
    names = ["TOKEN-4", "EDL", "OUTPUT-DISTILL", "CF-FORECAST-STUDENT", "FORECAST-SIMILARITY", "SHUFFLED-FORECAST",
             "EDL+CF", "MEMORY-ONLY"]
    return [run_cfg(f"t_{n}", n, 0, BUDGET["FINAL"], k, 5e-3, 0.3, final=True) for n in names]


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
    hdr = ["time", "stage", "id", "seed", "knobs", "reason", "ce_calib_search", "fsi_domain", "mi_mean", "flags",
           "eligible", "success", "run_name", "note"]
    new = not p.exists()
    with open(p, "a") as f:
        if new:
            f.write("\t".join(hdr) + "\n")
        f.write("\t".join(str(x) for x in [guard.now(), rec.get("stage"), rec["id"], rec.get("seed"),
                                            json.dumps(rec["knobs"], sort_keys=True), rec.get("reason"),
                                            rec.get("ce"), rec.get("audit", {}).get("fsi_domain"),
                                            rec.get("audit", {}).get("mi_mean"), json.dumps(rec.get("flags")),
                                            rec.get("eligible"), rec.get("success"), rec.get("run_name"),
                                            rec.get("note", "")]) + "\n")


# ------------------------------------------------------------------ dev records
def dev_record(state, run_name, summ, audit_key):
    """Development record for diagnosis from a completed run: CE (CALIB-SEARCH), diag, audit, throughput."""
    rd = find_run_dir(run_name)
    rec = {"run_name": run_name, "ce": None, "finite": summ is not None}
    if summ is None:
        return rec
    rec["ce"] = summ.get("eval", {}).get("calib_search", {}).get("ce")
    rec["d"] = summ.get("diag", {})
    rec["tok_s"] = summ.get("tokens_per_s")
    au = rd / "dev_audit.json"
    if au.exists():
        a = json.loads(au.read_text())["p100"]
        rec["audit"] = {"fsi_domain": a["domain"]["fsi"], "mi_mean": a["domain"]["routing"]["mi_mean"],
                        "fsi_cluster": a.get("cluster", {}).get("fsi"), "fsi_shuf": a["domain"]["fsi_shuf"]}
    if "p1end" in rec["d"]:
        rec["profiles_p1"] = rec["d"]["p1end"].get("profiles")
    if "end" in rec["d"]:
        rec["profiles_end"] = rec["d"]["end"].get("profiles")
    au_ = summ.get("cfaudit") or []
    if au_:
        rec["cfaudit_rho"] = float(sum(a["rho_fo_exact"] for a in au_) / len(au_))
    gn = []
    if (rd / "metrics.jsonl").exists():
        for line in (rd / "metrics.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r.get("phase") == 1 and "tc_gn" in r:
                gn.append(r["tc_gn"])
    rec["tc_gn_p1"] = float(sum(gn) / len(gn)) if gn else None
    return rec


def dev_run(state, stage, name, cond_name, seed, budget, knobs=None, edl_beta=0.0, od_alpha=0.0, candidate=None):
    cfg = run_cfg(name, cond_name, seed, budget, knobs, edl_beta, od_alpha, candidate=candidate, category="search")
    p = write_cfg(stage, name, cfg)
    summ = train_run(state, p, name, f"lfr {stage} {name}")
    rd = find_run_dir(name)
    if summ is not None and rd is not None:
        job(state, f"devaudit_{name}", "lfr_evaluate.py", ["dev-audit", "--run-dir", str(rd.relative_to(ROOT)),
                                                           "--device", "cuda", "--execute"],
            f"lfr dev audit {name}", projected_s=900)
    if not state["runs"][name].get("manifested"):
        ce = summ.get("eval", {}).get("calib_search", {}).get("ce") if summ else None
        manifest_row(stage, name, cfg, summ or {}, ce)
        state["runs"][name]["manifested"] = True
        save_state(state)
    return dev_record(state, name, summ, f"devaudit_{name}")


# ------------------------------------------------------------------ stages
def stage_prereg(state):
    if state["done"].get("prereg"):
        return
    from csr.lfr_lib import stream_manifest
    ss = S.search_space_doc()
    guard.atomic_write_json(ART / "search_space.json", ss, mode=0o400)
    sac = {"label_systems": {"domain": "five document domains", "cluster": {
        "encoder": "intfloat/e5-base-v2@f52bf8ec8c7124536f0efb74aca902b2995e5bcd (artifacts/encoder/e5-base-v2)",
        "model_sha256": "d0d559c47d5f71b1d280b13b62a2657f3e3bc70c0786f9ab91a36545e6a8f693",
        "window_tokens": 256, "prefix": "passage: ", "pooling": "mean", "max_len": 512, "K": 16, "n_init": 20,
        "seeds": "2026+i", "algorithm": "k-means++ / Lloyd (csr.specaudit.kmeans)",
        "fit": "CALIB-SEARCH paper-half-A windows", "assign": "nearest centroid"}},
        "discovery": "CALIB-SEARCH half A, 0.2M tokens/domain", "dev_measurement": "CALIB-SEARCH half B, 0.2M/domain",
        "final_measurement": "VAL first 0.2M tokens/domain (file order)",
        "metrics": ["mi (top-4 pairs) per layer + load-preserving null", "topic entropy per expert",
                    "topic prediction accuracy (nearest-centroid over 256-window histograms)",
                    "within-expert semantic similarity (excess over global)", "layer profiles", "load/entropy",
                    "cross-seed Hungarian-aligned profile correlation + null", "FSI (difficulty-matched, decile)",
                    "FSI_raw", "FSI shuffled-label control (seed 7)"],
        "fsi": "mean_k I_k / a_k, a_k = max(1,|G_k|)/11; I_k = O_k - X_k (decile reweighted)",
        "code": {"src/csr/lfr_audit.py": guard.sha256_file(ROOT / "src/csr/lfr_audit.py")}}
    guard.atomic_write_json(ART / "spec_audit_config.json", sac, mode=0o400)
    sm = stream_manifest()
    guard.atomic_write_json(ART / "stream_manifest.json", sm, mode=0o400)
    rec = {"protocol_sha256": protocol_sha(), "search_space_sha256": guard.sha256_file(ART / "search_space.json"),
           "spec_audit_config_sha256": guard.sha256_file(ART / "spec_audit_config.json"),
           "stream_manifest_sha256": guard.sha256_file(ART / "stream_manifest.json"), "time": guard.now()}
    guard.atomic_write_json(ART / "preregistration_hashes.json", rec, mode=0o400)
    log(f"prereg: {rec}")
    state["done"]["prereg"] = True
    save_state(state)


def stage_tests(state, key="tests"):
    if state["done"].get(key):
        return
    env = {k: v for k, v in os.environ.items()}
    env["CUDA_VISIBLE_DEVICES"] = ""
    r = subprocess.run([str(ROOT / ".venv/bin/python"), "lfr_tests.py"], cwd=ROOT, env=env, capture_output=True,
                       text=True)
    log(f"{key}: rc={r.returncode} {r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-500:]}")
    if r.returncode != 0:
        raise Halt(f"CPU suite failed ({key}); fix implementation bugs before continuing")
    state["done"][key] = True
    save_state(state)


def stage_labeler(state):
    if not job(state, "labeler_calib", "lfr_labeler.py", ["calib", "--device", "cuda", "--execute"],
               "lfr labeler calib", planned_bytes=200e6, projected_s=900):
        raise Halt("labeler failed")


def stage_draft_diag(state):
    if not job(state, "draft_diag", "lfr_draft_diag.py", ["--device", "cuda", "--execute"], "lfr draft diag",
               projected_s=900):
        raise Halt("draft diagnostics failed")
    dd = json.loads((ART / "draft_diagnostics.json").read_text())
    if not state["done"].get("draft_gate_logged"):
        log(f"draft gate: {dd['gate']}")
        state["done"]["draft_gate_logged"] = True
        save_state(state)
    if not dd["gate"]["pass"]:
        raise Halt("DRAFT GATE FAILED: preregistered response = train new drafts (protocol 4); implement before "
                   "continuing")


def dev_cache_steps():
    from csr.lfr_lib import main_stream
    _, s2, _ = main_stream(BUDGET["S2"])
    return math.ceil(0.6 * len(s2))


def stage_cache_dev(state):
    if not job(state, "cache_diag", "lfr_cache.py", ["--region", "calib_search", "--device", "cuda", "--execute"],
               "lfr cache calib_search", planned_bytes=0.1e9, projected_s=600):
        raise Halt("cache (diag region) failed")
    k = dev_cache_steps()
    if not state["done"].get("cache_projection"):
        dd = json.loads((ART / "draft_diagnostics.json").read_text())
        rate = min(dd["cache_path_tok_per_s"])  # tokens/s per draft (cache path)
        from csr.lfr_lib import main_stream
        _, sf, tf = main_stream(BUDGET["FINAL"])
        proj_final_h = (0.6 * tf) * 6 / rate / 3600
        state["cache_projection"] = {"rate_tok_s_per_draft": rate, "final_region_hours": proj_final_h}
        log(f"cache projection: {state['cache_projection']}")
        if proj_final_h > STAGE_CAP_H["cache"]:
            raise Halt(f"cache projection {proj_final_h:.2f} h > {STAGE_CAP_H['cache']} h: preregistered anchor "
                       f"fallback (protocol 5) must be implemented before continuing")
        state["done"]["cache_projection"] = True
        save_state(state)
    if not job(state, "cache_dev", "lfr_cache.py", ["--region", "main", "--steps", str(k), "--device", "cuda",
                                                     "--execute"], "lfr cache main dev", planned_bytes=4e9,
               projected_s=state["cache_projection"]["final_region_hours"] * 3600 / 2):
        raise Halt("cache (dev region) failed")


def stage_gpu_checks(state):
    if not job(state, "gpu_checks", "lfr_gpu_checks.py", ["--device", "cuda", "--execute"], "lfr gpu checks",
               projected_s=1800):
        raise Halt("GPU checks failed")


SMOKE_CONDS = ["TOKEN-4", "EDL", "OUTPUT-DISTILL", "CF-FORECAST-STUDENT", "FORECAST-SIMILARITY", "SHUFFLED-FORECAST",
               "EDL+CF", "MEMORY-ONLY"]


def stage_smoke(state):
    knobs = S.apply(S.BASE, {"c_lc": 0.05})
    for cn in SMOKE_CONDS:
        name = f"smoke_{cn}"
        if state["runs"].get(name, {}).get("status") == "COMPLETE":
            continue
        cfg = run_cfg(name, cn, 100, 40 * 63_000, knobs, 5e-3, 0.3, category="checks_and_tests")
        cfg["end_eval"] = []
        cfg["cf_audit_every"] = 10
        p = write_cfg("smoke", name, cfg)
        if train_run(state, p, name, f"lfr smoke {cn}") is None:
            raise Halt(f"smoke run {cn} failed: fix implementation bugs before continuing")


def stage_resume_check(state):
    if not job(state, "resume_check", "lfr_resume_check.py", ["--execute"], "lfr resume check", planned_bytes=8e9,
               projected_s=1500):
        raise Halt("resume check failed")


def ref_record(state, name):
    return state["dev"].get(name)


def stage_search1(state):
    state.setdefault("dev", {})
    s1 = BUDGET["S1"]
    seed = DEV_SEEDS[0]
    refs = [("s1_TOKEN-4_s100", "TOKEN-4", {})]
    refs += [(f"s1_{k}_s100", "EDL", {"edl_beta": b}) for k, b in EDL_BETAS.items()]
    refs += [(f"s1_{k}_s100", "OUTPUT-DISTILL", {"od_alpha": a}) for k, a in OD_ALPHAS.items()]
    for name, cn, kw in refs:
        if name not in state["dev"]:
            state["dev"][name] = dev_run(state, "s1", name, cn, seed, s1, **kw)
            save_state(state)
    t4 = state["dev"]["s1_TOKEN-4_s100"]
    edl_best = min(EDL_BETAS, key=lambda k: state["dev"][f"s1_{k}_s100"]["ce"] or 1e9)
    od_best = min(OD_ALPHAS, key=lambda k: state["dev"][f"s1_{k}_s100"]["ce"] or 1e9)
    state["baselines"] = {"edl_beta": EDL_BETAS[edl_best], "edl_run": f"s1_{edl_best}_s100",
                          "od_alpha": OD_ALPHAS[od_best], "od_run": f"s1_{od_best}_s100"}
    save_state(state)
    edl = state["dev"][state["baselines"]["edl_run"]]
    # counterfactual estimator validation gate on the S1 TOKEN-4 reference
    if not job(state, "cfval", "lfr_cfval.py", ["--run-dir", str(find_run_dir("s1_TOKEN-4_s100").relative_to(ROOT)),
                                                 "--device", "cuda", "--execute"], "lfr cf validation", projected_s=900):
        raise Halt("CF validation job failed")
    cv = json.loads((ART / "cf_validation.json").read_text())
    if not cv["gate_pass"]:
        raise Halt("CF ESTIMATOR GATE FAILED: preregistered fallback (exact sparse targets, protocol 6.4) must be "
                   "implemented before continuing")
    cands = state.setdefault("cands", [])
    while True:
        if gpu_seconds_used("search") > STAGE_CAP_H["search1"] * 3600:
            log("stage 1 GPU allocation used; stopping candidate generation")
            break
        nxt = S.next_candidate(cands, t4)
        if nxt is None:
            break
        cid, knobs, why = nxt
        name = f"s1_{cid}_s100"
        log(f"stage 1 candidate {cid}: {why}: {json.dumps(knobs, sort_keys=True)}")
        rec = dev_run(state, "s1", name, "CF-FORECAST-STUDENT", seed, s1, knobs, candidate=cid)
        rec.update(id=cid, knobs=knobs, reason=why, stage="s1", seed=seed)
        rec["p1name_ref"] = "p1end_30" if knobs["schedule"].startswith("30") else "p1end_40"
        flags, succ = S.diagnose(rec, t4, edl) if rec.get("ce") is not None else ({"H": {"failed": True}}, False)
        rec["flags"], rec["success"] = flags, succ
        rec["eligible"] = S.eligible(rec, t4)
        cands.append({k: v for k, v in rec.items() if k not in ("profiles_p1", "profiles_end", "d")})
        candidate_row(rec)
        write_diag_report(rec, t4, edl)
        save_state(state)
    state["done"]["search1"] = True
    save_state(state)


def write_diag_report(rec, t4, edl):
    d = ART / "diagnostics"
    d.mkdir(exist_ok=True)
    guard.atomic_write_json(d / f"{rec['run_name']}.json", {"record": {k: v for k, v in rec.items()
                                                                       if k not in ("profiles_p1", "profiles_end")},
                                                            "t4": {k: t4.get(k) for k in ("ce", "audit", "tok_s")},
                                                            "edl": {k: edl.get(k) for k in ("ce", "audit")}},
                            overwrite=True)
    lines = [f"# {rec['id']} ({rec['run_name']})", "", f"- knobs: `{json.dumps(rec['knobs'], sort_keys=True)}`",
             f"- reason: {rec.get('reason')}", f"- CALIB-SEARCH CE {rec.get('ce')} (TOKEN-4 {t4.get('ce')}, EDL "
             f"{edl.get('ce')})", f"- dev audit: {rec.get('audit')}", f"- flags: `{json.dumps(rec.get('flags'))}`",
             f"- success: {rec.get('success')}, eligible: {rec.get('eligible')}"]
    p1 = (rec.get("d") or {}).get("p1end", {})
    if p1:
        lines.append(f"- end of P1: teacher CE {p1.get('ce_teacher')}, shuffled {p1.get('ce_teacher_shuffled')}, "
                     f"student {p1.get('ce_student')}, probe ratio {p1.get('probe_ratio')}, rho(teacher, exact) "
                     f"{p1.get('rho_teacher_exact_mean')}, rho(first-order, exact) {p1.get('rho_fo_exact_mean')}")
    (d / f"{rec['run_name']}.md").write_text("\n".join(lines) + "\n")


def stage_search2(state):
    s2 = BUDGET["S2"]
    t4 = state["dev"]["s1_TOKEN-4_s100"]
    s1c = [c for c in state["cands"] if c.get("stage") == "s1"]
    if "advanced" not in state:
        state["advanced"] = S.advance(s1c, t4)
        log(f"stage 2 advancement: {state['advanced']}")
        save_state(state)
    b = state["baselines"]
    for seed in DEV_SEEDS:
        for name, cn, kw in [(f"s2_TOKEN-4_s{seed}", "TOKEN-4", {}),
                             (f"s2_EDL_s{seed}", "EDL", {"edl_beta": b["edl_beta"]}),
                             (f"s2_OUTPUT-DISTILL_s{seed}", "OUTPUT-DISTILL", {"od_alpha": b["od_alpha"]})]:
            if name not in state["dev"]:
                state["dev"][name] = dev_run(state, "s2", name, cn, seed, s2, **kw)
                save_state(state)
    for cid, el in state["advanced"]:
        knobs = [c for c in s1c if c["id"] == cid][0]["knobs"]
        for seed in DEV_SEEDS:
            name = f"s2_{cid}_s{seed}"
            if name not in state["dev"]:
                rec = dev_run(state, "s2", name, "CF-FORECAST-STUDENT", seed, s2, knobs, candidate=cid)
                rec.update(id=cid, knobs=knobs, stage="s2", seed=seed, reason=f"advanced (eligible={el})")
                rec["p1name_ref"] = "p1end_30" if knobs["schedule"].startswith("30") else "p1end_40"
                t4s, edls = state["dev"][f"s2_TOKEN-4_s{seed}"], state["dev"][f"s2_EDL_s{seed}"]
                flags, succ = S.diagnose(rec, t4s, edls) if rec.get("ce") is not None else ({"H": {}}, False)
                rec["flags"], rec["success"], rec["eligible"] = flags, succ, S.eligible(rec, t4s)
                state["dev"][name] = rec
                candidate_row(rec)
                write_diag_report(rec, t4s, edls)
                save_state(state)
    mk = state["cands"][0]["knobs"] if state["cands"] else S.BASE
    best_s1 = S.incumbent(s1c, t4)
    mk = best_s1["knobs"] if best_s1 else mk
    for seed in DEV_SEEDS:
        name = f"s2_MEMORY-ONLY_s{seed}"
        if name not in state["dev"]:
            state["dev"][name] = dev_run(state, "s2", name, "MEMORY-ONLY", seed, s2, mk)
            save_state(state)
    state["done"]["search2"] = True
    save_state(state)


def stage_select(state):
    if (ART / "frozen.json").exists():
        return
    if not (ART / "STAGE3_OPEN.json").exists():
        guard.atomic_write_json(ART / "STAGE3_OPEN.json", {"time": guard.now(), "note": "CALIB-SELECTION opened"},
                                mode=0o400)
        log("stage 3: CALIB-SELECTION opened")
    for name in sorted(n for n in state["dev"] if n.startswith("s2_")):
        rd = find_run_dir(name)
        if rd is not None and (rd / "summary.json").exists():
            job(state, f"calibsel_{name}", "lfr_evaluate.py", ["calib-sel", "--run-dir", str(rd.relative_to(ROOT)),
                                                               "--device", "cuda", "--execute"],
                f"lfr calib-sel {name}", projected_s=300)
    sel = {}
    for cid, _ in state["advanced"]:
        r = {"ce_sel": [], "fsi": [], "H": []}
        for seed in DEV_SEEDS:
            name = f"s2_{cid}_s{seed}"
            ev = find_run_dir(name) / "calib_sel_eval.json"
            r["ce_sel"].append(json.loads(ev.read_text())["calib_sel"]["ce"] if ev.exists() else 1e9)
            rec = state["dev"][name]
            r["fsi"].append(rec.get("audit", {}).get("fsi_domain", -1e9))
            r["H"].append("H" in rec.get("flags", {}))
        sel[cid] = r
    t4_fsi = sum(state["dev"][f"s2_TOKEN-4_s{s}"].get("audit", {}).get("fsi_domain", 0.0) for s in DEV_SEEDS) / 2
    cid, ok = S.select(sel, t4_fsi)
    knobs = [c for c in state["cands"] if c["id"] == cid][0]["knobs"]
    fz = {"selected": cid, "eligible": ok, "knobs": knobs, "edl_beta": state["baselines"]["edl_beta"],
          "od_alpha": state["baselines"]["od_alpha"], "selection_table": sel, "t4_fsi_mean_s2": t4_fsi,
          "time": guard.now(), "protocol_sha256": protocol_sha()}
    guard.atomic_write_json(ART / "frozen.json", fz, mode=0o400)
    (ART / "frozen.sha256").write_text(guard.sha256_file(ART / "frozen.json") + "  frozen.json\n")
    log(f"FROZEN: {cid} eligible={ok} knobs={json.dumps(knobs, sort_keys=True)} sha256 "
        f"{guard.sha256_file(ART / 'frozen.json')}")


def frozen():
    return json.loads((ART / "frozen.json").read_text())


def stage_final_prep(state):
    fz = frozen()
    a_, b_, _ = [int(v) for v in fz["knobs"]["schedule"].split("/")]
    from csr.lfr_lib import main_stream
    _, sf, _ = main_stream(BUDGET["FINAL"])
    k = math.ceil((a_ + b_) / 100 * len(sf)) + 1
    if not job(state, "cache_final", "lfr_cache.py", ["--region", "main", "--steps", str(k), "--device", "cuda",
                                                       "--execute"], "lfr cache main final", planned_bytes=4e9,
               projected_s=state["cache_projection"]["final_region_hours"] * 3600 / 2):
        raise Halt("final cache failed")
    job(state, "draft_diag_weak", "lfr_draft_diag.py", ["--weak-only", "--device", "cuda", "--execute"],
        "lfr draft diag weak", projected_s=300)
    if not job(state, "labeler_val", "lfr_labeler.py", ["val", "--device", "cuda", "--execute"], "lfr labeler val",
               projected_s=300):
        raise Halt("val labeler failed")
    stage_tests(state, "tests_final")


FINAL_ORDER = [["TOKEN-4", "EDL", "CF-FORECAST-STUDENT", "OUTPUT-DISTILL"], ["EDL+CF"], ["FORECAST-SIMILARITY",
                                                                                         "SHUFFLED-FORECAST"]]


def final_run(state, cond, seed, budget=None):
    fz = frozen()
    name = f"final_{cond}_s{seed}"
    cfg = run_cfg(name, cond, seed, budget or BUDGET["FINAL"], fz["knobs"], fz["edl_beta"], fz["od_alpha"],
                  schedule=fz["knobs"]["schedule"], final=True, candidate=fz["selected"], category="main_training")
    p = write_cfg("final", name, cfg)
    summ = train_run(state, p, name, f"lfr final {name}")
    rd = find_run_dir(name)
    if summ is not None:
        job(state, f"final_eval_{name}", "lfr_evaluate.py", ["final", "--run-dir", str(rd.relative_to(ROOT)),
                                                              "--device", "cuda", "--execute"],
            f"lfr final eval {name}", projected_s=2400)
    if not state["runs"][name].get("manifested"):
        ev = rd / "final_eval.json" if rd else None
        ce = json.loads(ev.read_text())["val"]["ce"] if ev and ev.exists() else None
        manifest_row("final", name, cfg, summ or {}, ce)
        state["runs"][name]["manifested"] = True
        save_state(state)
    return summ


def val_ce(cond, seed):
    rd = find_run_dir(f"final_{cond}_s{seed}")
    p = rd / "final_eval.json" if rd else None
    return json.loads(p.read_text())["val"]["ce"] if p and p.exists() else None


def stage_finals(state):
    for group in FINAL_ORDER:
        for seed in FINAL_SEEDS:
            for cond in group:
                final_run(state, cond, seed)
        if group is FINAL_ORDER[0]:
            stage_seed2(state)
    # equal total compute
    stage_token_compute(state)
    for seed in FINAL_SEEDS:
        final_run(state, "MEMORY-ONLY", seed)
    state["done"]["finals"] = True
    save_state(state)


def stage_seed2(state):
    e = [val_ce("CF-FORECAST-STUDENT", s) for s in FINAL_SEEDS]
    b = [val_ce("EDL", s) for s in FINAL_SEEDS]
    if None in e or None in b:
        return
    trig = all(x < y for x, y in zip(e, b))
    if "seed2_trigger" not in state:
        state["seed2_trigger"] = {"triggered": trig, "E": e, "B": b, "time": guard.now()}
        log(f"seed-2 trigger: {state['seed2_trigger']}")
        save_state(state)
    if trig:
        for cond in ("TOKEN-4", "EDL", "OUTPUT-DISTILL", "CF-FORECAST-STUDENT"):
            final_run(state, cond, 2)


def stage_token_compute(state):
    if "tc_budget" not in state:
        runs = [f"final_CF-FORECAST-STUDENT_s{s}" for s in FINAL_SEEDS]
        secs = []
        for r in runs:
            rd = find_run_dir(r)
            secs.append(sum(json.loads(p.read_text())["leg_seconds"] for p in sorted(rd.glob("leg_*.json"))))
        t4_rates = [json.loads((find_run_dir(f"final_TOKEN-4_s{s}") / "summary.json").read_text())["tokens_per_s"]
                    for s in FINAL_SEEDS]
        cache_s = gpu_seconds_used("forecast_cache")
        target = sum(secs) / len(secs) + (877.1 + cache_s) / len(runs)
        tc_tokens = int(target * sum(t4_rates) / len(t4_rates))
        tc_tokens = min(tc_tokens, 600_000_000)
        state["tc_budget"] = {"target_gpu_s": target, "e_run_gpu_s": secs, "draft_s": 877.1, "cache_s": cache_s,
                              "t4_tok_s": t4_rates, "tokens": tc_tokens}
        log(f"TOKEN-COMPUTE budget: {state['tc_budget']}")
        save_state(state)
    for seed in FINAL_SEEDS:
        final_run(state, "TOKEN-COMPUTE", seed, state["tc_budget"]["tokens"])


def stage_bench(state):
    job(state, "bench", "lfr_bench.py", ["--device", "cuda", "--execute"], "lfr bench", projected_s=900)


def stage_analyze(state):
    r = subprocess.run([str(ROOT / ".venv/bin/python"), "lfr_analyze.py"], cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    log(f"analyze rc={r.returncode} {r.stdout[-800:]} {r.stderr[-800:]}")


STAGES = [("prereg", stage_prereg), ("tests", stage_tests), ("labeler", stage_labeler),
          ("draft_diag", stage_draft_diag), ("cache_dev", stage_cache_dev), ("gpu_checks", stage_gpu_checks),
          ("smoke", stage_smoke), ("resume_check", stage_resume_check), ("search1", stage_search1), ("search2", stage_search2),
          ("select", stage_select), ("final_prep", stage_final_prep), ("finals", stage_finals),
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
    state = load_state()
    state.setdefault("jobs", {})
    STATE_CACHE["rates"] = state.get("rates", [])
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
            if state["done"].get(name) and name not in ("tests",):
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
