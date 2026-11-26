#!/usr/bin/env python
"""csr-context-prior-fast-engineering driver (plan: docs/CONTEXT_PRIOR_FAST_ENGINEERING_PLAN.md).

  cfe_driver.py add ID --stage screen|s2|control|calibsel|final [--knobs JSON] [--parent ID] --reason TEXT
  cfe_driver.py run [--authorized-resume]     process the queue in order; exits when it is empty
  cfe_driver.py table                          rewrite incumbent_table.{tsv,md}

Entries: screen = S1 seed 100 with registered early-stop gates (+ end-of-run CALIB-SEARCH CE on/off + dev audit when
it completes); s2 = seeds 100 and 101 at S2 (+ dev audits + two-seed freeze rule -> frozen.json); control = shuffled
context (M5) at S2 seed 100 for the frozen candidate; calibsel = CALIB-SELECTION CE of the frozen S2 runs; final = FINAL
plan seeds 0 / 1 + one VAL evaluation each. TOKEN-4 is never trained here (recorded references only).
External stop (signal, STOP marker, foreign GPU process, child killed from outside): the active run is checkpointed by
its trainer, marked INTERRUPTED, the driver exits and does not restart (needs --authorized-resume)."""
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
sys.path.insert(0, str(ROOT))
from csr import guard  # noqa: E402
from csr.cfe_lib import ART, BUDGET, CFG_DIR, RUN_PREFIX, STOP_MARKER, V1_RUN_PREFIX, plan_sha  # noqa: E402
from csr.cps_lib import ART as V2_ART, RUN_PREFIX as V2_RUN_PREFIX  # noqa: E402
from csr import cps_jsonz as jz  # noqa: E402

STATE = ART / "driver_state.json"
QUEUE = ART / "queue.json"
RESUME_CMD = ("cd /home/zapan4/moe-research && source env.sh && nohup .venv/bin/python cfe_driver.py run "
              "--authorized-resume >> .tmp/context-prior-fast-engineering/driver.out 2>&1 &")
DRIVER = {"stop": None, "child": None, "active": None}
BUDGET_ENG_H, BUDGET_POST_H, MAX_SCREEN = 7.0, 4.0, 10
TRAIN = {"lr": 6e-4, "min_lr": 6e-5, "warmup_frac": 0.01, "decay_frac": 0.2, "rows_per_step": 64,
         "micro_batch_rows": 32, "weight_decay": 0.1, "betas": [0.9, 0.95], "clip": 1.0, "aux_coeff": 0.01,
         "z_coeff": 0.001, "log_every": 50}  # identical to the TOKEN-4 reference runs
BACKBONE = {"vocab_size": 50304, "n_layer": 12, "d_model": 384, "n_head": 6, "dense_ffn": 1792, "n_experts": 32,
            "top_k": 4, "expert_ffn": 256, "shared_ffn": 768, "rope_base": 10000.0, "norm_eps": 1e-6, "init_std": 0.02}
V2_C07 = json.loads((V2_ART / "frozen.json").read_text())["knobs"]
BASE = {**V2_C07, "alpha_min": 0.0}
T4 = {"S1": {100: {"run": "s1_TOKEN-4_s100", "ce": 3.789612529453125, "fsi_domain": 0.003957914390381391,
                   "fsi_cluster": 0.0046932049716374435, "mi_domain": 0.0265843486601014, "tok_s": 136140.3}},
      "S2": {100: {"run": "s2_TOKEN-4_s100", "ce": 3.301776115546875, "fsi_domain": 0.005752903832356276,
                   "fsi_cluster": 0.00837048958482539, "mi_domain": 0.033852790773076194, "sel_ce": 3.26849727453125,
                   "tok_s": 137319.5},
             101: {"run": "s2_TOKEN-4_s101", "ce": 3.2940681612890623, "fsi_domain": 0.006263995433824226,
                   "fsi_cluster": 0.008445013026350041, "mi_domain": 0.0313892995999768, "sel_ce": 3.26080085640625,
                   "tok_s": 136853.9}}}
GATES = {"S1": [318, 544], "S2": [795, 1357]}
# addendum 1 (deviation 4): median paired training-loss gap to TOKEN-4 of the inert V2 runs at each gate window
NULL_DLM = {("S1", 100): {"318": 0.013105, "544": 0.025541},
            ("S2", 100): {"795": -0.004393, "1357": -0.000331}, ("S2", 101): {"795": 0.004692, "1357": -0.000587}}
INERT = {"changed": 0.05, "dce": 0.002}


class ExternalStop(Exception):
    pass


class Halt(Exception):
    pass


# ------------------------------------------------------------------ state
def load(p, default):
    return json.loads(p.read_text()) if p.exists() else default


def save_state(s):
    s["active"] = DRIVER["active"]
    guard.atomic_write_json(STATE, s, overwrite=True)


def log(msg):
    line = f"{guard.now()} {msg}"
    print(line, flush=True)
    with open(ART / "driver_log.txt", "a") as f:
        f.write(line + "\n")


def gpu_hours(cats=None):
    p = ART / "compute_ledger.tsv"
    s = 0.0
    if p.exists():
        for line in p.read_text().splitlines():
            f = line.split("\t")
            if len(f) >= 7:
                try:
                    if cats is None or f[5] in cats:
                        s += float(f[2])
                except ValueError:
                    pass
    return s / 3600


# ------------------------------------------------------------------ guarded launch (V2 logic)
def _session_pids(sid):
    out = []
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                with open(f"/proc/{d}/stat") as f:
                    if int(f.read().rsplit(")", 1)[1].split()[3]) == sid:
                        out.append(int(d))
            except (OSError, IndexError, ValueError):
                pass
    return out


def _is_experiment_main(pid):
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            a = [x.decode(errors="replace") for x in f.read().split(b"\0") if x]
    except OSError:
        return False
    return len(a) >= 2 and a[0].endswith("/python") and os.path.basename(a[1]).startswith("cfe_")


def _forward_stop(why):
    ch = DRIVER["child"]
    if ch is None or ch.poll() is not None:
        return
    for pid in _session_pids(ch.pid):
        if _is_experiment_main(pid):
            try:
                os.kill(pid, signal.SIGINT)
                log(f"forwarded cooperative SIGINT to child pid {pid} ({why})")
            except ProcessLookupError:
                pass


def _driver_signal(signum, frame):
    name = signal.Signals(signum).name
    if DRIVER["stop"] is None:
        DRIVER["stop"] = f"driver received {name}"
        log(f"EXTERNAL STOP: driver received {name}; stopping at the next safe point")
        _forward_stop(name)


def launch(purpose, script, args):
    if DRIVER["stop"]:
        raise ExternalStop(DRIVER["stop"])
    if STOP_MARKER.exists():
        raise ExternalStop(f"cfe stop marker present before launching {purpose}")
    n = subprocess.run(["nvidia-smi", "-i", "0", "--query-compute-apps=pid", "--format=csv,noheader"],
                       capture_output=True, text=True).stdout.split()
    if n:
        raise ExternalStop(f"WAITING_FOR_GPU: foreign process(es) {n} on GPU 0 before launching {purpose}")
    env = dict(os.environ, MOE_GPU_LAUNCH="1", MOE_RESEARCH_APPROVE="gpu")
    DRIVER["active"] = {"purpose": purpose, "script": script, "args": args, "launched": guard.now()}
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
    ext = None
    if DRIVER["stop"]:
        ext = DRIVER["stop"]
    elif rc in (3, 78):
        ext = f"foreign GPU process present before launch (rc={rc})"
    elif rc == 77:
        ext = "child reported EXTERNALLY_STOPPED (signal / stop marker / foreign GPU process)"
    elif rc < 0 or (rc >= 128 and rc != 124):
        ext = f"child terminated from outside (rc={rc})"
    elif STOP_MARKER.exists():
        ext = "cfe stop marker present"
    return rc, ext


def find_run_dir(name, prefix=RUN_PREFIX):
    c = sorted(glob.glob(str(ROOT / f"runs/*_{prefix}{name}")))
    return Path(c[-1]) if c else None


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


def interrupted(state, key, rd, why):
    from csr.cfe_ckpt import ResumeStore
    ck = None
    if rd is not None:
        try:
            ck = ResumeStore(rd.name, "").latest_verified(log=lambda m: None)
        except Exception as e:  # noqa: BLE001
            ck = {"error": str(e)[:200]}
        old = load(rd / "status.json", {})
        guard.atomic_write_json(rd / "status.json", {**old, "status": "INTERRUPTED", "time": guard.now(),
                                                     "reason": f"external interruption: {why}",
                                                     "resume_from": ck}, overwrite=True)
    state["interrupted"] = {"time": guard.now(), "job": key, "reason": why,
                            "run_dir": str(rd.relative_to(ROOT)) if rd else None, "resume_from_checkpoint": ck,
                            "resume_command": RESUME_CMD, "note": "do not resume without explicit user instruction"}
    save_state(state)
    raise ExternalStop(f"{key}: {why}")


def det_theta(L):
    return json.loads((V2_ART / "detector_calibration.json").read_text())["theta"][str(L)]


def run_cfg(name, k, seed, stage, cond="CAND", gates=True, fracs=(1.0,), end_eval=("calib_search",)):
    cc = {"variant": "prior", "teacher": True, "chunk": k["chunk"], "chunk_len": k["chunk_len"],
          "det_theta": det_theta(k["chunk_len"]) if k["chunk"] == "adaptive" else 0.0, "ctx_dim": k["ctx_dim"],
          "mem_hls": k["mem_hls"], "mem_w": 32, "enc_hidden": 128, "inject": k["inject"],
          "alpha_init": k["alpha_init"], "alpha_max": k["alpha_max"], "prior_norm": k["prior_norm"], "fc_dim": 64,
          "horizons": [1], "shuffle_ctx": cond == "SHUF", "alpha_min": k.get("alpha_min", 0.0)}
    c = {"name": cond, "c_util": k["c_util"], "util_mode": k["util_mode"], "c_spec": k["c_spec"], "c_kl": k["c_kl"],
         "lam_max": k["lam_max"], "schedule": k["schedule"], "r_cf": k["r_cf"], "cf_phases": k["cf_phases"],
         "ctx_lr_mult": k["ctx_lr_mult"], "alpha_l1": k["alpha_l1"]}
    budget = BUDGET[stage]
    g = None
    if gates and stage in GATES:
        g = {"steps": GATES[stage], "tok_per_dom": 20000, "window": 150, "max_dlm": 0.02, "collapse_max": 0.10,
             "collapse_min": 0.005, "inert_changed": 0.02, "inert_dce": 5e-4,
             "t4_metrics": str((find_run_dir(T4[stage][seed]["run"], V1_RUN_PREFIX) / "metrics.jsonl"
                                ).relative_to(ROOT)), "incumbent": None, "null_dlm": NULL_DLM.get((stage, seed))}
    return {"experiment": "csr-context-prior-fast-engineering", "run_name": name, "kind": "main", "condition": cond,
            "seed": seed, "budget_tokens": int(budget), "backbone": BACKBONE, "ccfg": cc, "cond": c, "train": TRAIN,
            "checkpoint_fracs": list(fracs), "end_eval": list(end_eval), "knobs": k, "category": stage,
            "cf_audit_every": 100, "gate": g, "plan_sha256": plan_sha()}


def train(state, key, cfg_sub, cfg, purpose):
    """Launch legs until COMPLETE / EARLY_STOPPED. Returns the summary dict or None (failed twice)."""
    p = write_cfg(cfg_sub, cfg["run_name"], cfg)
    rec = state["runs"].setdefault(cfg["run_name"], {"attempts": 0, "legs": 0})
    if rec.get("status") in ("COMPLETE", "EARLY_STOPPED"):
        return json.loads((ROOT / rec["run_dir"] / "summary.json").read_text())
    try:
        # deviation 3: one rolling slot of ~0.83 GB (peak 2 during rotation) + milestones (0.2 GB each) + evaluations
        guard.require_disk(int(1.6e9 + 0.21e9 * len(cfg.get("checkpoint_fracs") or [1.0]) + 0.03e9),
                           f"cfe {cfg['run_name']}")
    except guard.GuardError as e:
        raise Halt(f"INSUFFICIENT_RESOURCES (disk): {e}")
    while True:
        rd = find_run_dir(cfg["run_name"])
        args = ["--config", str(p.relative_to(ROOT)), "--device", "cuda", "--execute"]
        if rd is not None:
            args += ["--run-dir", str(rd.relative_to(ROOT))]
        rc, ext = launch(f"cfe {key} {cfg['run_name']}", "cfe_train.py", args)
        rd = find_run_dir(cfg["run_name"])
        st = load(rd / "status.json", {}).get("status") if rd else None
        log(f"{cfg['run_name']}: rc={rc} status={st}")
        if ext:
            interrupted(state, key, rd, ext)
        rec["run_dir"] = str(rd.relative_to(ROOT)) if rd else None
        if rc == 0 and st in ("COMPLETE", "EARLY_STOPPED"):
            rec["status"] = st
            save_state(state)
            return json.loads((rd / "summary.json").read_text())
        if rc == 75:
            rec["legs"] += 1
            save_state(state)
            if rec["legs"] <= 6:
                continue
        rec["attempts"] += 1
        save_state(state)
        if rec["attempts"] >= 2:
            rec["status"] = "FAILED"
            save_state(state)
            return None


def eval_job(state, key, mode, rd):
    j = state["jobs"].setdefault(key, {"attempts": 0})
    if j.get("status") == "COMPLETE":
        return True
    while True:
        rc, ext = launch(f"cfe eval {mode} {rd.name}", "cfe_evaluate.py",
                         [mode, "--run-dir", str(rd.relative_to(ROOT)), "--device", "cuda", "--execute"])
        log(f"job {key}: rc={rc}")
        if ext:
            interrupted(state, key, None, ext)
        if rc == 0:
            j["status"] = "COMPLETE"
            save_state(state)
            return True
        j["attempts"] += 1
        save_state(state)
        if j["attempts"] >= 2:
            j["status"] = "FAILED"
            save_state(state)
            return False


# ------------------------------------------------------------------ disk policy (deviation 5)
def compress_eval(path, why):
    """Lossless verified gzip of a finished evaluation JSON (same method as cps_jsonz, cfe artifact directory)."""
    import gzip
    import hashlib
    path = Path(path)
    if not path.exists():
        return
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    gz, tmp = path.with_name(path.name + ".gz"), path.with_name(path.name + ".gz.tmp")
    with open(tmp, "wb") as f:
        with gzip.GzipFile(filename=path.name, mode="wb", compresslevel=6, fileobj=f, mtime=0) as g:
            g.write(raw)
    with gzip.open(tmp, "rb") as g:
        assert hashlib.sha256(g.read()).hexdigest() == sha
    os.replace(tmp, gz)
    os.chmod(gz, 0o400)
    with open(ART / "compressed_files.jsonl", "a") as f:
        f.write(json.dumps({"time": guard.now(), "path": str(path), "orig_sha256": sha, "orig_bytes": len(raw),
                            "gz_bytes": gz.stat().st_size, "why": why}) + "\n")
    os.chmod(path, 0o600)
    path.unlink()


def drop_losing_milestones(name, why):
    import shutil
    from csr.cfe_ckpt import ledger
    rd = find_run_dir(name)
    if rd is None or not (rd / "summary.json").exists():
        return
    for tag, e in json.loads((rd / "summary.json").read_text()).get("ckpts", {}).items():
        if (ROOT / e["path"]).exists():
            shutil.rmtree(ROOT / e["path"])
            ledger({"time": guard.now(), "run_id": rd.name, "run_name": name, "kind": "milestone_deleted_disk",
                    "path": e["path"], "step": e["step"], "seen": e["seen"], "sha256": e["sha256"], "note": why})


# ------------------------------------------------------------------ records
def record(name, summ, stage, seed):
    rd = find_run_dir(name)
    r = {"run_name": name, "stage": stage, "seed": seed, "status": summ.get("status") if summ else "FAILED",
         "tok_s": (summ or {}).get("tokens_per_s"), "params": (summ or {}).get("params"),
         "gates": (summ or {}).get("gates"), "steps_done": (summ or {}).get("steps_done")}
    ev = (summ or {}).get("eval") or {}
    if "calib_search" in ev:
        r["ce"] = ev["calib_search"]["ce"]
        r["ce_prior_off"] = ev["calib_search_prior_off"]["ce"]
    ap = ART / "evals" / rd.name / "dev_audit.json" if rd else None
    if ap is not None and jz.exists(ap):
        a = jz.load_json(ap)["p100"]
        r.update(fsi_domain=a["domain"]["fsi"], fsi_cluster=a.get("cluster", {}).get("fsi"),
                 mi_domain=a["domain"]["routing"]["mi_mean"], mi_cluster=a.get("cluster", {}).get("routing",
                                                                                               {}).get("mi_mean"),
                 ctx=a.get("ctx_influence"), prior_off_meas=(a.get("prior_off_meas") or {}).get("ce"),
                 ce_meas=a.get("ce_meas"))
    lm, stats = {}, {}
    if rd is not None and (rd / "metrics.jsonl").exists():
        for line in (rd / "metrics.jsonl").read_text().splitlines():
            x = json.loads(line)
            if "lm" in x:
                for k in ("cs_alpha", "cs_prior_rel_scale"):
                    if x.get(k) is not None:
                        stats.setdefault(k[3:], []).append(x[k])
    r["alpha_end"] = sum(stats["alpha"][-3:]) / len(stats["alpha"][-3:]) if stats.get("alpha") else None
    r["prior_rel_scale_end"] = (sum(stats["prior_rel_scale"][-3:]) / len(stats["prior_rel_scale"][-3:])
                                if stats.get("prior_rel_scale") else None)
    return r


def verdict(r, ref):
    """Plan section 2/3 rule for one run against its matched TOKEN-4 reference."""
    if r.get("status") != "COMPLETE" or r.get("ce") is None or r.get("fsi_domain") is None:
        return {"promising": False, "why": [f"status {r.get('status')}"]}
    why = []
    dce = r["ce"] - ref["ce"]
    if dce > 0.01:
        why.append(f"CE {dce:+.4f} > +0.01")
    if not r["fsi_domain"] > ref["fsi_domain"]:
        why.append(f"FSI domain {r['fsi_domain']:.5f} <= T4 {ref['fsi_domain']:.5f}")
    if not (r.get("fsi_cluster") or 0) > ref["fsi_cluster"]:
        why.append(f"FSI cluster {r.get('fsi_cluster')} <= T4 {ref['fsi_cluster']:.5f}")
    ch = (r.get("ctx") or {}).get("routing_changed_off", 0.0)
    dce_off = abs((r.get("ce_prior_off") or r["ce"]) - r["ce"])
    if ch < INERT["changed"] and dce_off < INERT["dce"]:
        why.append(f"inert (prior-off changes {ch:.3f} of slots, |dCE| {dce_off:.5f})")
    return {"promising": not why, "why": why, "dce_vs_t4": dce, "routing_changed_off": ch, "dce_off": dce_off}


def write_table(state):
    hdr = ["id", "stage", "seed", "status", "parent", "addresses", "change", "ce", "dce_vs_t4", "fsi_domain",
           "t4_fsi_domain", "fsi_cluster", "t4_fsi_cluster", "mi_domain", "t4_mi_domain", "alpha_end",
           "prior_rel_scale_end", "ce_prior_off", "dce_prior_off", "routing_changed_off", "tok_s", "t4_tok_s",
           "active_params", "gate_dlm", "verdict"]
    rows = []
    for cid, c in state["cands"].items():
        for rr in c.get("runs", []):
            ref = T4[rr["stage"]][rr["seed"]]
            v = rr.get("verdict") or {}
            g = rr.get("gates") or {}
            gl = [f"{s}:{x.get('dlm_vs_t4'):+.4f}" if x.get("dlm_vs_t4") is not None else f"{s}:-"
                  for s, x in sorted(g.items(), key=lambda kv: int(kv[0]))]
            rows.append([cid, rr["stage"], rr["seed"], rr["status"], c.get("parent"), c.get("addresses"),
                         json.dumps(c.get("change")), rr.get("ce"),
                         None if rr.get("ce") is None else rr["ce"] - ref["ce"], rr.get("fsi_domain"),
                         ref["fsi_domain"], rr.get("fsi_cluster"), ref["fsi_cluster"], rr.get("mi_domain"),
                         ref["mi_domain"], rr.get("alpha_end"), rr.get("prior_rel_scale_end"), rr.get("ce_prior_off"),
                         None if rr.get("ce_prior_off") is None else rr["ce_prior_off"] - rr["ce"],
                         (rr.get("ctx") or {}).get("routing_changed_off"), rr.get("tok_s"), ref["tok_s"],
                         (rr.get("params") or {}).get("active_per_token"), " ".join(gl),
                         ("PROMISING" if v.get("promising") else "; ".join(v.get("why", []))) or None])
    with open(ART / "incumbent_table.tsv.tmp", "w") as f:
        f.write("\t".join(hdr) + "\n")
        for r in rows:
            f.write("\t".join("" if x is None else (f"{x:.5f}" if isinstance(x, float) else str(x)) for x in r) + "\n")
    os.replace(ART / "incumbent_table.tsv.tmp", ART / "incumbent_table.tsv")
    fmt = lambda x, n=4: "-" if x is None else (f"{x:+.{n}f}" if isinstance(x, float) else str(x))
    md = ["# Fast-engineering incumbent table", "", f"incumbent: {state.get('incumbent')}  |  GPU-h used: "
          f"{gpu_hours():.2f} (post-freeze training {gpu_hours(('post',)):.2f})", "",
          "| id | stage/seed | status | CE - T4 | FSI dom (T4) | FSI clu (T4) | MI dom (T4) | alpha | prior-off "
          "dCE | prior-off routing chg | tok/s (T4) | verdict |", "|" + "---|" * 12]
    for r in rows:
        md.append(f"| {r[0]} | {r[1]}/{r[2]} | {r[3]} | {fmt(r[8])} | {fmt(r[9], 5)} ({r[10]:.5f}) | "
                  f"{fmt(r[11], 5)} ({r[12]:.5f}) | {fmt(r[13])} ({r[14]:.4f}) | {fmt(r[15], 3)} | {fmt(r[18])} | "
                  f"{fmt(r[19], 3)} | {'-' if r[20] is None else int(r[20])} ({int(r[21])}) | {r[24] or '-'} |")
    (ART / "incumbent_table.md").write_text("\n".join(md) + "\n")


# ------------------------------------------------------------------ entries
def do_screen(state, e):
    cid = e["id"]
    c = state["cands"][cid]
    n_screen = sum(1 for x in state["cands"].values() if x.get("screened"))
    if not c.get("screened") and n_screen >= MAX_SCREEN:
        raise Halt(f"screen cap {MAX_SCREEN} reached")
    if gpu_hours() > BUDGET_ENG_H:
        raise Halt(f"engineering GPU budget {BUDGET_ENG_H} h exhausted")
    c["screened"] = True
    name = f"s1_{cid}_s100"
    cfg = run_cfg(name, c["knobs"], 100, "S1")
    inc = state.get("incumbent")
    if inc and inc != cid and state["cands"].get(inc, {}).get("runs") and \
            state["cands"][inc]["knobs"] != c["knobs"]:
        gi = state["cands"][inc]["runs"][0].get("gates") or {}
        cfg["gate"]["incumbent"] = {s: {"mi_on": x["mi_on"], "routing_changed_off": x["routing_changed_off"]}
                                    for s, x in gi.items()} or None
    summ = train(state, f"screen {cid}", "s1", cfg, f"cfe screen {cid}")
    if summ is not None and summ.get("status") == "COMPLETE":
        eval_job(state, f"devaudit_{name}", "dev-audit", find_run_dir(name))
    r = record(name, summ, "S1", 100)
    r["verdict"] = verdict(r, T4["S1"][100])
    c["runs"] = [r]
    if rd_ := find_run_dir(name):
        compress_eval(ART / "evals" / rd_.name / "dev_audit.json", "dev audit consumed (deviation 5)")
    if not r["verdict"]["promising"]:
        drop_losing_milestones(name, "non-promising screen; dev audit done (deviation 5 disk policy)")
    if summ is not None and summ.get("status") == "EARLY_STOPPED":
        c["early_stopped"] = True
    if r["verdict"]["promising"]:
        c["promising"] = True
    # incumbent: promising first, then smallest max(0, dCE-0.01) + FSI deficit, among completed runs
    def key_(x):
        rr = x[1]["runs"][0]
        v = rr.get("verdict") or {}
        if rr.get("ce") is None:
            return (2, 0)
        return (0 if v.get("promising") else 1, max(0.0, rr["ce"] - T4["S1"][100]["ce"] - 0.01) * 10
                - (rr.get("fsi_domain", 0) - T4["S1"][100]["fsi_domain"]) * 100
                - ((rr.get("ctx") or {}).get("routing_changed_off", 0.0)))
    done = [(k, v) for k, v in state["cands"].items() if v.get("runs") and v["runs"][0].get("stage") == "S1"
            and v["runs"][0].get("ce") is not None]  # only completed screens can be the incumbent
    state["incumbent"] = min(done, key=key_)[0] if done else None
    save_state(state)
    log(f"SCREEN {cid}: {json.dumps({k: r.get(k) for k in ('status', 'ce', 'fsi_domain', 'fsi_cluster', 'mi_domain', 'alpha_end', 'ce_prior_off')}, default=str)} "
        f"verdict={r['verdict']} incumbent={state['incumbent']}")


def do_s2(state, e):
    cid = e["id"]
    c = state["cands"][cid]
    if gpu_hours() > BUDGET_ENG_H:
        raise Halt(f"engineering GPU budget {BUDGET_ENG_H} h exhausted")
    runs = []
    for seed in (100, 101):
        name = f"s2_{cid}_s{seed}"
        cfg = run_cfg(name, c["knobs"], seed, "S2")
        summ = train(state, f"s2 {cid} s{seed}", "s2", cfg, f"cfe s2 {cid}")
        if summ is not None and summ.get("status") == "COMPLETE":
            eval_job(state, f"devaudit_{name}", "dev-audit", find_run_dir(name))
        r = record(name, summ, "S2", seed)
        r["verdict"] = verdict(r, T4["S2"][seed])
        runs.append(r)
        if rd_ := find_run_dir(name):
            compress_eval(ART / "evals" / rd_.name / "dev_audit.json", "dev audit consumed (deviation 5)")
        log(f"S2 {cid} s{seed}: ce={r.get('ce')} fsi={r.get('fsi_domain')}/{r.get('fsi_cluster')} "
            f"verdict={r['verdict']}")
        if not r["verdict"]["promising"]:
            break  # plan: no GPU on a candidate that already failed the two-seed rule in one seed
    c["runs"] = [x for x in c.get("runs", []) if x["stage"] == "S1"] + runs
    ok = len(runs) == 2 and all(r["verdict"]["promising"] for r in runs)
    c["s2_pass"] = ok
    if ok and not (ART / "frozen.json").exists():
        guard.atomic_write_json(ART / "frozen.json", {"time": guard.now(), "selected": cid, "knobs": c["knobs"],
                                                      "s2": runs, "rule": "plan section 3 (both seeds)"}, mode=0o400)
        log(f"FROZEN: {cid}")
    save_state(state)


def do_control(state, e):
    fz = load(ART / "frozen.json", None)
    if fz is None:
        raise Halt("control requested without a freeze")
    if gpu_hours(("post",)) > BUDGET_POST_H:
        raise Halt("post-freeze GPU budget exhausted")
    name = f"s2_{fz['selected']}-SHUF_s100"
    cfg = run_cfg(name, fz["knobs"], 100, "S2", cond="SHUF", gates=False)
    cfg["category"] = "post"
    summ = train(state, "control SHUF", "post", cfg, "cfe control")
    if summ is not None and summ.get("status") == "COMPLETE":
        eval_job(state, f"devaudit_{name}", "dev-audit", find_run_dir(name))
    state["control"] = record(name, summ, "S2", 100)
    save_state(state)
    log(f"CONTROL SHUF: {state['control']}")


def do_calibsel(state, e):
    fz = load(ART / "frozen.json", None)
    if fz is None:
        raise Halt("calibsel requested without a freeze")
    if not (ART / "STAGE3_OPEN.json").exists():
        guard.atomic_write_json(ART / "STAGE3_OPEN.json", {"time": guard.now(), "candidates": [fz["selected"]]})
    for seed in (100, 101):
        name = f"s2_{fz['selected']}_s{seed}"
        eval_job(state, f"calibsel_{name}", "calib-sel", find_run_dir(name))
    save_state(state)


def do_final(state, e):
    fz = load(ART / "frozen.json", None)
    if fz is None:
        raise Halt("final requested without a freeze")
    for seed in (0, 1):
        name = f"final_{fz['selected']}_s{seed}"
        cfg = run_cfg(name, fz["knobs"], seed, "FINAL", gates=False, fracs=(0.4, 0.6, 0.8, 1.0), end_eval=())
        cfg["category"] = "post"
        summ = train(state, f"final s{seed}", "final", cfg, "cfe final")
        if summ is not None and summ.get("status") == "COMPLETE":
            eval_job(state, f"final_eval_{name}", "final", find_run_dir(name))
    save_state(state)


HANDLERS = {"screen": do_screen, "s2": do_s2, "control": do_control, "calibsel": do_calibsel, "final": do_final}


def cmd_add(argv):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("id")
    ap.add_argument("--stage", required=True, choices=list(HANDLERS))
    ap.add_argument("--knobs", default=None, help="JSON overrides of the parent's knobs")
    ap.add_argument("--parent", default=None)
    ap.add_argument("--addresses", default=None, help="observed failure this modification addresses")
    ap.add_argument("--reason", required=True)
    a = ap.parse_args(argv)
    state = load(STATE, {"runs": {}, "jobs": {}, "cands": {}, "incumbent": None})
    if a.stage == "screen":
        if a.id in state["cands"]:
            raise SystemExit(f"{a.id} exists")
        par = state["cands"][a.parent]["knobs"] if a.parent and a.parent in state["cands"] else BASE
        ch = json.loads(a.knobs) if a.knobs else {}
        bad = set(ch) - set(BASE)
        if bad:
            raise SystemExit(f"unknown knobs {bad}")
        state["cands"][a.id] = {"knobs": {**par, **ch}, "change": ch, "parent": a.parent or "V2-C07",
                                "addresses": a.addresses, "reason": a.reason, "added": guard.now()}
    elif a.stage == "s2" and a.id not in state["cands"]:
        raise SystemExit(f"unknown candidate {a.id}")
    q = load(QUEUE, [])
    q.append({"id": a.id, "stage": a.stage, "reason": a.reason, "added": guard.now(), "status": "PENDING"})
    guard.atomic_write_json(QUEUE, q, overwrite=True)
    save_state(state)
    print(f"queued {a.stage} {a.id}")


def cmd_run(argv):
    authorized = "--authorized-resume" in argv
    state = load(STATE, {"runs": {}, "jobs": {}, "cands": {}, "incumbent": None})
    if STOP_MARKER.exists():
        log("REFUSING to start: cfe stop marker exists")
        sys.exit(3)
    if "interrupted" in state and not authorized:
        log("REFUSING to start: unresolved interruption; needs explicit user authorization (--authorized-resume)")
        sys.exit(3)
    if "interrupted" in state:
        state.setdefault("interrupt_history", []).append({**state.pop("interrupted"), "resumed": guard.now()})
        save_state(state)
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _driver_signal)
    try:
        while True:
            q = load(QUEUE, [])
            pend = [x for x in q if x["status"] == "PENDING"]
            if not pend:
                log("queue empty; driver exiting")
                break
            e = pend[0]
            log(f"== {e['stage']} {e['id']} ({e['reason']}) | GPU-h {gpu_hours():.2f}")
            HANDLERS[e["stage"]](state, e)
            if DRIVER["stop"]:
                raise ExternalStop(DRIVER["stop"])
            q = load(QUEUE, [])
            for x in q:
                if x["id"] == e["id"] and x["stage"] == e["stage"] and x["status"] == "PENDING":
                    x["status"] = "DONE"
                    x["done"] = guard.now()
                    break
            guard.atomic_write_json(QUEUE, q, overwrite=True)
            write_table(state)
    except ExternalStop as ex:
        if "interrupted" not in state:
            state["interrupted"] = {"time": guard.now(), "reason": str(ex), "resume_command": RESUME_CMD,
                                    "note": "do not resume without explicit user instruction"}
            save_state(state)
        log(f"INTERRUPTED: {ex}; no automatic restart. Resume: {RESUME_CMD}")
        sys.exit(0)
    except Halt as h:
        state["halt"] = {"time": guard.now(), "reason": str(h)}
        save_state(state)
        log(f"HALT: {h}")
    write_table(state)


def main():
    guard.check_root()
    ART.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) < 2:
        print(__doc__)
        return
    cmd, argv = sys.argv[1], sys.argv[2:]
    if cmd == "add":
        cmd_add(argv)
    elif cmd == "run":
        cmd_run(argv)
    elif cmd == "table":
        write_table(load(STATE, {"cands": {}}))


if __name__ == "__main__":
    main()
