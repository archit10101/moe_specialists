"""ccsr1 Stage 3 (joint continuation) and section-11 efficiency study orchestration (amendment sections A-E).
Called from ccsr1_run.py with its guarded job() wrapper:
  stage3a     config -> train 318 -> 1111 (legs) -> evaluate m1111 on CS2 -> criteria
  stage3b     only if stage3a passed: continue to 3880 with evaluations at m2111, m3111, m3880
  efficiency  refresh CE (CPU) + measured speed (GPU)"""
import json

from csr import guard
from csr import ccsr1_log as LG
from csr.ccsr1_paths import ART, CFG_DIR


def run_dir():
    c = sorted((guard.ROOT / "runs").glob("*_ccsr1_S3"))
    return c[-1] if c else None


def write_config():
    from csr.ccsr1_paths import check_prereg
    from csr import ccsr1_s3_paths as P
    p = CFG_DIR / "S3.json"
    if p.exists():
        return p
    base = json.loads((guard.ROOT / P.CFAR1_RUN / "config.json").read_text())
    cfg = {"experiment": "csr-current-context-specialist-router-v1", "run_name": "S3",
           "prereg_sha256": check_prereg(), "amend_sha256": P.check_amendment(),
           "base_ckpt": P.CKPT, "base_ckpt_sha256": P.CKPT_SHA,
           "base_config": {k: base[k] for k in ("model", "train", "budget_tokens", "seed")},
           "start_step": P.START, "end_step": P.END_PLAN, "milestones": P.MILESTONES, "arms": P.arms(),
           "joint": P.JOINT, "rewarm": P.REWARM, "slot_every_steps": P.SLOT_EVERY_STEPS, "slot_every_s": P.SLOT_EVERY_S}
    guard.atomic_write_json(p, cfg, mode=0o400)
    return p


def train_to(job, max_steps, stage):
    cp = write_config()
    leg = 1
    while True:
        rd = run_dir()
        if rd is not None:
            stj = json.loads((rd / "status.json").read_text())
            if stj["status"] == "COMPLETE":
                return rd
            if stj.get("steps_done", 0) >= max_steps and stj["status"] == "INTERRUPTED":
                return rd
        argv = ["ccsr1_s3_train.py", "--config", str(cp.relative_to(guard.ROOT)), "--max-steps", str(max_steps)]
        if rd is not None:
            argv += ["--run-dir", str(rd.relative_to(guard.ROOT))]
        st = json.loads((ART / "driver_state.json").read_text())
        while st["jobs"].get(f"s3_train_to{max_steps}_leg{leg}", {}).get("status") == "COMPLETE":
            leg += 1
        job(f"s3_train_to{max_steps}_leg{leg}", "stage3", argv, expect="0,75", disk_gb=4.0)
        leg += 1
        if leg > 12:
            raise guard.GuardError("too many training legs")


def evaluate(job, tag):
    from csr.ccsr1_s3_paths import S3, S3_ROUTERS, S3_TABLES
    rd = run_dir()
    out = S3_TABLES / tag
    if not (out / "summary_CS2.json").exists():
        job(f"s3_eval_{tag}", "stage3", ["ccsr1_s3_eval.py", "--ckpt", str((rd / f"ckpt_{tag}").relative_to(guard.ROOT)),
                                         "--out-dir", str(out.relative_to(guard.ROOT)),
                                         "--router-dir", str((S3_ROUTERS / tag).relative_to(guard.ROOT))], disk_gb=1.8)
    if not (S3 / f"stage3_{tag}.json").exists():
        job(f"s3_analyze_{tag}", "stage3", ["ccsr1_s3_analyze.py", "--tag", tag], gpu=False, disk_gb=0.05)
    return json.loads((S3 / f"stage3_{tag}.json").read_text())


def stage3a(job):
    from csr import ccsr1_s3_paths as P
    s2 = json.loads((ART / "stage2.json").read_text())
    if not s2["stage2_pass"]:
        raise guard.GuardError("Stage 3 requires a passing Stage 2")
    P.check_amendment()
    LG.set_state("STAGE3A_RUNNING")
    train_to(job, P.END_3A, "stage3")
    res = evaluate(job, "m1111")
    LG.set_state("STAGE3A_PASS" if res["any_arm_pass"] else "TERMINAL_STAGE3A_FAIL")
    return res


def stage3b(job):
    from csr import ccsr1_s3_paths as P
    from csr.ccsr1_s3_paths import S3
    r3a = json.loads((S3 / "stage3_m1111.json").read_text())
    if not r3a["any_arm_pass"]:
        raise guard.GuardError("Stage 3b requires a passing Stage 3a")
    LG.set_state("STAGE3B_RUNNING")
    out = {}
    for tag in ("m2111", "m3111", "m3880"):
        train_to(job, P.MILESTONES[tag], "stage3")
        out[tag] = evaluate(job, tag)
        alive = [k for k, v in out[tag]["arms"].items() if not v["criteria"]["any_stop_violation"]]
        if not alive:
            LG.set_state("TERMINAL_STAGE3B_STOP", at=tag)
            return out
    LG.set_state("STAGE3B_PASS" if out["m3880"]["any_arm_pass"] else "TERMINAL_STAGE3B_FAIL")
    return out


def efficiency(job):
    from csr.ccsr1_s3_paths import EFF
    if not (EFF / "refresh_ce.json").exists():
        job("eff_ce", "efficiency", ["ccsr1_eff.py", "ce"], gpu=False, disk_gb=0.2)
    if not (EFF / "speed.json").exists():
        job("eff_speed", "efficiency", ["ccsr1_eff.py", "speed"], disk_gb=0.2)
