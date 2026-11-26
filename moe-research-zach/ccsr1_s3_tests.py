#!/usr/bin/env python
"""ccsr1 Stage-3 trainer CPU tests (2 rows per step, steps 318 -> 321):
  S1 resume equivalence: straight 3 steps == (--max-steps 320, exit 75) + resume, bitwise (model, banks, routers, opts)
  S2 bank independence: base after 3 steps with only G == base with all test arms (no bank gradient reaches the base)
  S3 frozen routers and FedBank student/EMA buffers unchanged; joint router and all specialists changed
  S4 external SIGINT at step 319 -> exit 77 with a verified slot; resume completes to 321
  ccsr1_s3_tests.py   (MOE_RESEARCH_APPROVE=cpu-model is set for the child runs)"""
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402

PY = str(guard.ROOT / ".venv/bin/python")
TD = guard.ROOT / ".tmp/csr-current-context-specialist-router-v1/s3tests"


def mkcfg(name, arm_names, end=321):
    from csr import ccsr1_s3_paths as P
    base = json.loads((guard.ROOT / P.CFAR1_RUN / "config.json").read_text())
    A = [a for a in P.arms() if a["name"] in arm_names]
    cfg = {"experiment": "csr-current-context-specialist-router-v1", "run_name": name, "test": True,
           "base_ckpt": P.CKPT, "base_ckpt_sha256": P.CKPT_SHA, "verify_ckpt": False,
           "base_config": {k: base[k] for k in ("model", "train", "budget_tokens", "seed")},
           "start_step": 318, "end_step": end, "milestones": {}, "arms": A, "joint": P.JOINT, "rewarm": 20,
           "slot_every_steps": 1, "slot_every_s": 600}
    TD.mkdir(parents=True, exist_ok=True)
    p = TD / f"{name}.json"
    p.write_text(json.dumps(cfg))
    return p


def train(cfgp, *extra):
    env = {**os.environ, "MOE_RESEARCH_APPROVE": "cpu-model"}
    r = subprocess.run([PY, "ccsr1_s3_train.py", "--config", str(cfgp.relative_to(guard.ROOT)), "--device", "cpu",
                        "--test-rows", "2", "--execute", *extra], cwd=guard.ROOT, env=env, capture_output=True, text=True)
    run_dir = None
    for line in (r.stdout or "").splitlines():
        if "stream:" in line:
            pass
    cands = sorted((guard.ROOT / "runs").glob(f"*_ccsr1_{json.loads(cfgp.read_text())['run_name']}"))
    run_dir = cands[-1] if cands else None
    return r.returncode, run_dir, r.stdout[-2000:] + r.stderr[-2000:]


def slot_state(run_dir, step):
    import torch
    p = guard.ROOT / ".tmp/csr-current-context-specialist-router-v1/slots" / run_dir.name / f"resume_s{step:07d}" / "state.pt"
    return torch.load(p, map_location="cpu", weights_only=False)


def drop_slots(run_dir):
    """Delete this experiment's CPU-test resume slots (test scratch; disk headroom), logged with sha256."""
    import shutil
    from csr import ccsr1_log as LG
    d = guard.ROOT / ".tmp/csr-current-context-specialist-router-v1/slots" / run_dir.name
    if not d.exists():
        return
    idx = json.loads((d / "resume_index.json").read_text()) if (d / "resume_index.json").exists() else {"entries": []}
    for e in idx["entries"]:
        LG.deletion(e["path"], e["sha256"], e["bytes"], "ccsr1 Stage-3 CPU-test resume slot (test scratch)")
    shutil.rmtree(d)


def same(a, b):
    import torch
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if torch.is_tensor(a):
        return a.shape == b.shape and a.dtype == b.dtype and bool(torch.equal(a, b))
    return a == b


def main():
    import torch
    res = {}
    ARMS = ["G", "SEMf_s0", "SEMj_s0", "SHUFf_s0"]
    ca = mkcfg("S3TA", ARMS)
    rc, ra, log = train(ca)
    assert rc == 0, log
    A = slot_state(ra, 321)
    drop_slots(ra)
    cb = mkcfg("S3TB", ARMS)
    rc1, rb, log = train(cb, "--max-steps", "320")
    assert rc1 == 75, log
    rc2, rb2, log = train(cb, "--run-dir", str(rb.relative_to(guard.ROOT)))
    assert rc2 == 0 and rb2 == rb, log
    B = slot_state(rb, 321)
    drop_slots(rb)
    res["S1_resume_equivalence"] = all(same(A[k], B[k]) for k in ("model", "banks", "routers", "opt", "bank_opt", "router_opt"))
    cc = mkcfg("S3TC", ["G"])
    rc, rcd, log = train(cc)
    assert rc == 0, log
    C = slot_state(rcd, 321)
    drop_slots(rcd)
    res["S2_base_independent_of_banks"] = same(A["model"], C["model"]) and same(A["opt"], C["opt"])
    ck0 = torch.load(guard.ROOT / json.loads(ca.read_text())["base_ckpt"] / "state.pt", map_location="cpu", weights_only=False)
    from csr import ccsr1_s3_paths as P
    arms = {x["name"]: x for x in P.arms()}
    ok = True
    for n in ("SEMf_s0", "SHUFf_s0"):
        r0 = torch.load(guard.ROOT / arms[n]["router_ckpt"], map_location="cpu", weights_only=False)["state"]
        ok &= same(r0, A["routers"][n])
    rj0 = torch.load(guard.ROOT / arms["SEMj_s0"]["router_ckpt"], map_location="cpu", weights_only=False)["state"]
    joint_changed = not same(rj0, A["routers"]["SEMj_s0"])
    frozen_student = True
    experts_changed = True
    for n in ("SEMf_s0", "SEMj_s0", "SHUFf_s0"):
        src = arms[n]["init"]
        b0, b1 = ck0["banks"][src], A["banks"][n]
        for k in b0:
            if k.startswith(("inp.", "gru.", "head.", "ema_", "S1", "S2", "A1", "A2", "mass", "n_sync", "m", "scale", "ubar")):
                frozen_student &= bool(torch.equal(b0[k], b1[k]))
        experts_changed &= not torch.equal(b0["experts.w1"], b1["experts.w1"])
    experts_changed &= not torch.equal(ck0["banks"]["G"]["experts.w1"], A["banks"]["G"]["experts.w1"])
    res["S3_frozen_routers_unchanged"] = bool(ok)
    res["S3_joint_router_changed"] = bool(joint_changed)
    res["S3_fedbank_student_and_buffers_unchanged"] = bool(frozen_student)
    res["S3_all_specialists_changed"] = bool(experts_changed)
    cd = mkcfg("S3TD", ARMS)
    rc1, rd_, log = train(cd, "--test-signal-at-step", "319")
    ok77 = rc1 == 77
    rc2, _, log2 = train(cd, "--run-dir", str(rd_.relative_to(guard.ROOT)))
    D = slot_state(rd_, 321)
    drop_slots(rd_)
    res["S4_sigint_exit77_and_resume"] = bool(ok77 and rc2 == 0 and all(same(A[k], D[k]) for k in ("model", "banks", "routers")))
    res["all"] = all(res.values())
    out = guard.ROOT / f"artifacts/csr-current-context-specialist-router-v1/s3_cpu_tests_{guard.utc_stamp()}.json"
    guard.atomic_write_json(out, {"time": guard.now(), "results": res,
                                  "runs": [str(x.relative_to(guard.ROOT)) for x in (ra, rb, rcd, rd_)]})
    print(json.dumps(res, indent=1))
    sys.exit(0 if res["all"] else 1)


if __name__ == "__main__":
    main()
