#!/usr/bin/env python
"""csr-domain-forecast-curriculum-v1: persistent, resumable experiment driver (run inside tmux).

  .venv/bin/python dfc_driver.py            # runs/continues the full plan; state in .tmp/domain-forecast-curriculum-v1/

Every GPU job is launched through scripts/gpu_guarded.sh (one at a time). State (completed tasks, run dirs, attempts,
micro-batch per run) is saved atomically after every transition, so the driver can be restarted at any time and
continues where it stopped (training resumes in place from the rotating resume slot). Recovery rules:
  rc 0 complete; rc 75/124/130 interrupted -> resume in place; rc 76 OOM -> halve micro-batch rows (accumulation
  doubles; rows per step unchanged) and resume; GPU busy (launcher exit 3 / BLOCKED) -> wait 120 s, retry forever;
  other failures -> up to 3 attempts (resume if possible), then STOP with state FAILED (needs a fix).
The driver never deletes anything. Validation opens only after FREEZE.
"""
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/zapan4/moe-research")
sys.path.insert(0, str(ROOT / "src"))
from csr import guard  # noqa: E402
from csr.dfc_lib import ART, CACHE, CFG_DIR, FROZEN, MAIN_TOKENS, ST, code_hashes, prereg_sha  # noqa: E402

STATE = ST / "driver_state.json"
LOGS = ST / "logs"
SEEDS = (0, 1, 2)
CURR = ("LABEL", "FORECAST", "SOFT", "WRONG")
MAIN_ORDER = ("TOKEN-MAIN", "LABEL", "FORECAST", "SOFT", "WRONG")
MAX_FAIL = 3


def now():
    return guard.now()


def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"created": now(), "done": {}, "runs": {}, "fail": {}, "mbr": {}, "events": [], "launches": []}


def save_state(s):
    guard.atomic_write_json(STATE, s, overwrite=True)


def event(s, msg, **kw):
    rec = {"time": now(), "msg": msg, **kw}
    s["events"].append(rec)
    save_state(s)
    print(f"[driver {rec['time']}] {msg} {json.dumps(kw) if kw else ''}", flush=True)
    with open(ST / "driver.log", "a") as f:
        f.write(json.dumps(rec) + "\n")


def launch(s, purpose, args, gpu=True, log_name=None):
    """Returns (rc, stdout_tail, log path). gpu: via scripts/gpu_guarded.sh. Waits while GPU busy."""
    LOGS.mkdir(parents=True, exist_ok=True)
    while True:
        logp = LOGS / f"{guard.utc_stamp()}_{log_name or purpose.replace(' ', '_')}.log"
        env = dict(os.environ)
        if gpu:
            env.update(MOE_GPU_LAUNCH="1", MOE_RESEARCH_APPROVE="gpu")
            cmd = ["scripts/gpu_guarded.sh", purpose] + args
        else:
            env.update(CUDA_VISIBLE_DEVICES="", MOE_RESEARCH_APPROVE="cpu-model")
            cmd = [str(ROOT / ".venv/bin/python")] + args
        t0 = time.time()
        with open(logp, "w") as f:
            p = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
        out = logp.read_text()
        s["launches"].append({"time": now(), "purpose": purpose, "args": args, "rc": p.returncode,
                              "seconds": time.time() - t0, "log": str(logp.relative_to(ROOT))})
        save_state(s)
        if gpu and p.returncode == 3 and "BLOCKED" in out:
            event(s, "GPU busy (foreign process); waiting 120 s", purpose=purpose)
            time.sleep(120)
            continue
        return p.returncode, out[-4000:], logp


def run_dir_from(out):
    for ln in out.splitlines()[::-1]:
        if ln.startswith("RUN_DIR "):
            return ln.split(" ", 1)[1].strip()
    return None


def ensure_disk(s, need, what):
    while True:
        free = guard.disk_free_bytes()
        if free - need >= guard.DISK_RESERVE_BYTES:
            return
        event(s, "disk below reserve; waiting 600 s", free=free, need=need, what=what)
        time.sleep(600)


def train(s, key, cfg_path):
    """Train one config to completion (resume in place on interruption/OOM)."""
    if s["done"].get(key):
        return
    cfg = json.loads(Path(cfg_path).read_text())
    base_mbr = cfg["train"]["micro_batch_rows"]
    while True:
        ensure_disk(s, 2 * 1024**3, key)
        rd = s["runs"].get(key)
        mbr = s["mbr"].get(key, base_mbr)
        args = ["dfc_train.py", "--config", str(cfg_path), "--device", "cuda", "--execute", "--micro-batch-rows",
                str(mbr)]
        slot = ST / "resume_slot.pt"
        resumable = False
        if rd:
            try:
                import torch
                ck = torch.load(slot, map_location="cpu", weights_only=False, mmap=True)
                resumable = ck["run_dir"] == str(ROOT / rd)
            except Exception:  # noqa: BLE001
                resumable = False
            if resumable:
                args += ["--run-dir", rd]
            else:
                event(s, "no resume slot for this run; starting a fresh attempt (old dir kept, marked superseded)",
                      key=key, old=rd)
                s.setdefault("superseded", {}).setdefault(key, []).append(rd)
                s["runs"].pop(key)
        rc, out, logp = launch(s, f"dfc train {key}", args, log_name=f"train_{key}")
        rdn = run_dir_from(out)
        if rdn:
            s["runs"][key] = rdn
            save_state(s)
        if rc == 0:
            s["done"][key] = {"time": now(), "run_dir": s["runs"][key]}
            event(s, "train complete", key=key, run_dir=s["runs"][key])
            return
        if rc in (75, 124, 130):
            n_int = s.setdefault("interrupts", {}).get(key, 0) + 1
            s["interrupts"][key] = n_int
            event(s, "train interrupted; resuming", key=key, rc=rc, n=n_int)
            if n_int > 40:
                s["status"] = f"FAILED at {key} (too many interruptions)"
                save_state(s)
                raise SystemExit(f"STOP: {key} interrupted {n_int} times")
            continue
        if rc == 76:
            new = max(1, mbr // 2)
            if new == mbr:
                raise RuntimeError(f"{key}: OOM at micro-batch 1")
            s["mbr"][key] = new
            event(s, "OOM: micro-batch halved (accumulation doubled, rows/step unchanged)", key=key, mbr=new)
            guard.ledger("DFC_OOM", f"{key}: micro_batch_rows {mbr} -> {new}")
            continue
        s["fail"][key] = s["fail"].get(key, 0) + 1
        event(s, "train failed", key=key, rc=rc, n=s["fail"][key], log=str(logp.relative_to(ROOT)), tail=out[-1500:])
        if s["fail"][key] >= MAX_FAIL:
            s["status"] = f"FAILED at {key}"
            save_state(s)
            raise SystemExit(f"STOP: {key} failed {MAX_FAIL} times; see {logp}")
        time.sleep(30)


def simple(s, key, purpose, args, gpu=True, ok_rcs=(0,)):
    if s["done"].get(key):
        return
    while True:
        rc, out, logp = launch(s, purpose, args, gpu=gpu, log_name=key)
        if rc in ok_rcs:
            s["done"][key] = {"time": now(), "log": str(logp.relative_to(ROOT))}
            event(s, "task complete", key=key)
            return
        if rc in (75, 124, 130):
            event(s, "task interrupted; relaunching", key=key, rc=rc)
            continue
        s["fail"][key] = s["fail"].get(key, 0) + 1
        event(s, "task failed", key=key, rc=rc, n=s["fail"][key], log=str(logp.relative_to(ROOT)), tail=out[-1500:])
        if s["fail"][key] >= MAX_FAIL:
            s["status"] = f"FAILED at {key}"
            save_state(s)
            raise SystemExit(f"STOP: {key} failed {MAX_FAIL} times; see {logp}")
        time.sleep(30)


def audit_jobs(cond):
    if cond in CURR:
        return "p20:normal,p20:neutral,p40:normal,p40:neutral,p50:neutral,p75:neutral,p100:neutral"
    if cond == "TOKEN-MAIN":
        return "p20:neutral,p40:neutral,p50:neutral,p75:neutral,p100:neutral"
    return "final:neutral"


def train_and_audit(s, cond, seed):
    key = f"{cond}_s{seed}"
    train(s, key, CFG_DIR / "main" / f"{key}.json")
    simple(s, f"audit_{key}", f"dfc audit {key}", ["dfc_audit.py", "--run-dir", s["runs"][key], "--jobs",
                                                     audit_jobs(cond), "--device", "cuda", "--execute"])


def compute_match(s):
    p = ART / "compute_match.json"
    if p.exists():
        return json.loads(p.read_text())["n_tokens"]

    def summ(key):
        return json.loads((ROOT / s["runs"][key] / "summary.json").read_text())
    soft = [summ(f"SOFT_s{x}")["train_step_seconds"] for x in SEEDS]
    tokm = [summ(f"TOKEN-MAIN_s{x}")["train_step_seconds"] / summ(f"TOKEN-MAIN_s{x}")["valid_train_tokens"]
            for x in SEEDS]
    draft = summ("draft")["train_step_seconds"]
    cache = json.loads((CACHE / "main_correct.json").read_text())["gpu_seconds"]
    C = sum(soft) / 3 + draft / 3 + cache / 3
    rate = sum(tokm) / 3
    n = int(math.ceil(C / rate / 5000) * 5000)
    doc = {"created": now(), "rule": "C = mean_seeds(SOFT train-step s) + draft train-step s / 3 + MAIN-correct cache "
           "GPU s / 3; N = C / mean_seeds(TOKEN-MAIN train-step s per valid token), rounded up to 5,000",
           "soft_train_step_seconds": soft, "token_main_seconds_per_token": tokm, "draft_train_step_seconds": draft,
           "main_correct_cache_gpu_seconds": cache, "C_seconds": C, "n_tokens": n,
           "n_tokens_over_main": n / MAIN_TOKENS, "validation_opened": FROZEN.exists()}
    assert not FROZEN.exists()
    guard.atomic_write_json(p, doc, mode=0o400)
    guard.ledger("DFC_COMPUTE_MATCH", f"N={n} C={C:.0f}s")
    event(s, "compute match frozen", n_tokens=n)
    return n


def freeze(s):
    if FROZEN.exists():
        return
    runs = {}
    for seed in SEEDS:
        for c in MAIN_ORDER + ("TOKEN-ALL-DATA", "TOKEN-COMPUTE"):
            key = f"{c}_s{seed}"
            rd = ROOT / s["runs"][key]
            sm = json.loads((rd / "summary.json").read_text())
            ckp = sm["final_checkpoint"]
            runs[key] = {"run_dir": s["runs"][key], "final_checkpoint": ckp,
                         "final_checkpoint_sha256": guard.sha256_file(ROOT / ckp),
                         "config_sha256": guard.sha256_file(rd / "config.json"),
                         "valid_train_tokens": sm["valid_train_tokens"]}
    doc = {"created": now(), "prereg_sha256": prereg_sha(), "runs": runs, "code_sha256": code_hashes(),
           "compute_match_sha256": guard.sha256_file(ART / "compute_match.json"),
           "draft_run": s["runs"]["draft"], "note": "all training, audits and configurations frozen; validation opens"}
    guard.atomic_write_json(FROZEN, doc, mode=0o400)
    guard.ledger("DFC_FROZEN", str(FROZEN.relative_to(ROOT)))
    event(s, "FROZEN; validation unlocked")


def main():
    guard.check_root()
    ST.mkdir(parents=True, exist_ok=True)
    s = load_state()
    s["status"] = "RUNNING"
    s["pid"] = os.getpid()
    event(s, "driver start", prereg=prereg_sha()[:16])
    if not (ART / "data/manifest.json").exists():
        raise SystemExit("data not built")
    if not (CFG_DIR / "draft.json").exists():
        simple(s, "configs_base", "dfc configs", ["dfc_make_configs.py", "base"], gpu=False)
    simple(s, "cpu_tests_pre", "dfc cpu tests", ["dfc_cpu_tests.py", "pre"], gpu=False)
    simple(s, "gpu_checks_pre", "dfc gpu checks pre", ["dfc_gpu_checks.py", "--execute"])
    train(s, "draft", CFG_DIR / "draft.json")
    simple(s, "cache", "dfc forecast cache", ["dfc_cache.py", "--draft-run", s["runs"]["draft"], "--execute"])
    simple(s, "gpu_checks_post", "dfc gpu checks post", ["dfc_gpu_checks.py", "--draft-run", s["runs"]["draft"],
                                                          "--execute"])
    simple(s, "resume_check_gpu", "dfc resume check", ["dfc_resume_check.py"], gpu=False)
    simple(s, "cpu_tests_full", "dfc cpu tests", ["dfc_cpu_tests.py", "full"], gpu=False)
    for seed in SEEDS:
        for c in MAIN_ORDER:
            train_and_audit(s, c, seed)
    for seed in SEEDS:
        train_and_audit(s, "TOKEN-ALL-DATA", seed)
    n = compute_match(s)
    if not (CFG_DIR / "main/TOKEN-COMPUTE_s0.json").exists():
        simple(s, "configs_compute", "dfc configs compute", ["dfc_make_configs.py", "compute", str(n)], gpu=False)
    for seed in SEEDS:
        train_and_audit(s, "TOKEN-COMPUTE", seed)
    simple(s, "bench", "dfc bench", ["dfc_bench.py", "--runs",
                                     ",".join(s["runs"][f"{c}_s0"] for c in MAIN_ORDER + ("TOKEN-ALL-DATA",
                                                                                           "TOKEN-COMPUTE")),
                                     "--execute"])
    freeze(s)
    for seed in SEEDS:
        for c in MAIN_ORDER + ("TOKEN-ALL-DATA", "TOKEN-COMPUTE"):
            key = f"{c}_s{seed}"
            simple(s, f"val_{key}", f"dfc val eval {key}", ["dfc_evaluate.py", "--run-name", key, "--execute"])
    simple(s, "analysis", "dfc analysis", ["dfc_analyze.py"], gpu=False)
    s["status"] = "DONE"
    event(s, "ALL TASKS DONE")


if __name__ == "__main__":
    main()
