#!/usr/bin/env python
"""fspv2 Operational Amendment 1 (A7): checkpoint/resume equivalence under the new cadence, before the driver relies
on it. Production-shape FORECAST-SPECIALIST runs of 25 steps (all curriculum phases; cadence = every 2 steps):
  U1, U2  uninterrupted (U2 - U1 = run-to-run noise floor)
  RS      cooperative external stop: SIGINT delivered before step 11 -> EXTERNALLY_STOPPED (rc 77) with a verified
          checkpoint at step 11 -> resumed with --run-dir
  RK      abrupt SIGKILL before step 13 (no checkpoint) -> newest verified checkpoint (step 12) deliberately corrupted ->
          resume must reject it (sha256) and fall back to step 10 -> resumed
Pass: for RS and RK, max|theta - theta_U1| <= 3 x max|theta_U2 - theta_U1| + 1e-6 and per-step LM loss within
3 x the U2-U1 spread + 1e-4; resume points as expected; ledger/index/rotation/prune invariants hold.
Each leg via scripts/gpu_guarded.sh. Writes artifacts/forecast-specialist-pretraining-v2/resume_check2_<stamp>.json."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from csr import guard  # noqa: E402


def leg(purpose, cfg_rel, extra):
    env = dict(os.environ, MOE_GPU_LAUNCH="1", MOE_RESEARCH_APPROVE="gpu")
    r = subprocess.run(["scripts/gpu_guarded.sh", purpose, "fsp_train.py", "--config", cfg_rel, "--device", "cuda",
                        "--execute", *extra], cwd=ROOT, env=env)
    return r.returncode


def main():
    guard.check_root()
    import torch
    import fsp_driver as drv
    from csr.fsp_ckpt import LEDGER, ResumeStore
    from csr.fsp_lib import ART
    budget = 24 * 64 * 1000
    tag = sys.argv[1] if len(sys.argv) > 1 else "a"
    names, cfgs = {}, {}
    for t in ("U1", "U2", "RS", "RK"):
        n = f"resumecheck2{tag}_{t}"
        cfg = drv.main_cfg(n, "FORECAST-SPECIALIST", 100, budget, drv.BASE, [], [1.0], category="checks_and_tests")
        names[t], cfgs[t] = n, str(drv.write_cfg("checks", n, cfg).relative_to(ROOT))
    out = {"time": guard.now(), "legs": {}}
    for t in ("U1", "U2"):
        out["legs"][t] = leg(f"fspv2 resume check2 {t}", cfgs[t], [])
    rc = leg("fspv2 resume check2 RS leg1", cfgs["RS"], ["--test-signal-at-step", "11"])
    rs_dir = drv.find_run_dir(names["RS"])
    rs_st1 = json.loads((rs_dir / "status.json").read_text())
    out["legs"]["RS1"] = {"rc": rc, "status": rs_st1["status"], "reason": rs_st1.get("reason")}
    rs_idx1 = ResumeStore(ROOT / ".tmp/fspv2/slots", rs_dir.name, "").index()
    out["legs"]["RS2"] = leg("fspv2 resume check2 RS leg2", cfgs["RS"], ["--run-dir", str(rs_dir.relative_to(ROOT))])
    rc = leg("fspv2 resume check2 RK leg1", cfgs["RK"], ["--test-kill-at-step", "13"])
    rk_dir = drv.find_run_dir(names["RK"])
    rk_st1 = json.loads((rk_dir / "status.json").read_text())
    store = ResumeStore(ROOT / ".tmp/fspv2/slots", rk_dir.name, "")
    idx = store.index()
    out["legs"]["RK1"] = {"rc": rc, "status_after_kill": rk_st1["status"],
                          "index_steps_after_kill": [e["step"] for e in idx["entries"]]}
    newest = ROOT / idx["entries"][-1]["path"]
    with open(newest, "r+b") as f:  # deliberate corruption of the newest checkpoint (test only)
        f.seek(newest.stat().st_size // 2)
        b = f.read(1)
        f.seek(-1, 1)
        f.write(bytes([b[0] ^ 0xFF]))
    out["legs"]["RK1"]["corrupted"] = idx["entries"][-1]["path"]
    out["legs"]["RK2"] = leg("fspv2 resume check2 RK leg2", cfgs["RK"], ["--run-dir", str(rk_dir.relative_to(ROOT))])

    def load(t):
        rd = drv.find_run_dir(names[t])
        s = json.loads((rd / "summary.json").read_text())
        sd = torch.load(ROOT / s["ckpts"]["p100"]["path"], map_location="cpu", weights_only=True)["model"]
        lm = {}
        for line in (rd / "metrics.jsonl").read_text().splitlines():
            r = json.loads(line)
            lm[r["step"]] = r["lm"]
        logtxt = (rd / "log.txt").read_text()
        return sd, lm, s, logtxt, rd
    U1, U2, RS, RK = load("U1"), load("U2"), load("RS"), load("RK")
    dmax = lambda a, b: max(float((a[k].float() - b[k].float()).abs().max()) for k in a)
    noise = dmax(U2[0], U1[0])
    common = sorted(set(U1[1]) & set(U2[1]))
    lnoise = max(abs(U2[1][s] - U1[1][s]) for s in common)
    res = {"param_noise_u2_u1": noise, "loss_noise_u2_u1": lnoise, "steps_compared": common}
    ok = True
    for t, R in (("RS", RS), ("RK", RK)):
        d = dmax(R[0], U1[0])
        ld = max(abs(R[1][s] - U1[1][s]) for s in common)
        p = d <= 3 * noise + 1e-6 and ld <= 3 * lnoise + 1e-4 and R[2]["legs"] >= 2
        res[t] = {"param_maxdiff_vs_u1": d, "loss_maxdiff_vs_u1": ld, "legs": R[2]["legs"], "pass": p,
                  "resume_log": [ln[26:] for ln in R[3].splitlines() if "resume" in ln]}
        ok &= p
    # resume points
    ok_rs = out["legs"]["RS1"]["rc"] == 77 and out["legs"]["RS1"]["status"] == "EXTERNALLY_STOPPED" and \
        rs_idx1["entries"][-1]["step"] == 11 and any("resumed at step 11" in x for x in res["RS"]["resume_log"])
    ok_rk = out["legs"]["RK1"]["rc"] != 0 and out["legs"]["RK1"]["status_after_kill"] == "RUNNING" and \
        out["legs"]["RK1"]["index_steps_after_kill"] == [10, 12] and \
        any("FAILED sha256 verification" in x for x in res["RK"]["resume_log"]) and \
        any("resumed at step 10" in x for x in res["RK"]["resume_log"])
    # invariants: ledger rows, rotation kept <= 2 during the run, prune after complete, milestones recorded
    led = [ln.split("\t") for ln in LEDGER.read_text().splitlines()[1:]]
    inv = {}
    for t, R in (("U1", U1), ("RS", RS), ("RK", RK)):
        rid = R[4].name
        rows = [r for r in led if r[1] == rid]
        kinds = [r[3] for r in rows]
        left = list((ROOT / ".tmp/fspv2/slots" / rid).glob("resume_s*.pt"))
        inv[t] = {"n_resume": kinds.count("resume"), "n_rotated": kinds.count("rotated_out"),
                  "n_pruned": kinds.count("pruned_after_complete"), "n_milestone": kinds.count("milestone"),
                  "n_verify_failed": kinds.count("verify_failed"), "resume_files_left": len(left),
                  "ckpt_seconds": R[2].get("ckpt_seconds"), "n_resume_ckpts": R[2].get("n_resume_ckpts")}
    ok_inv = all(v["resume_files_left"] == 0 and v["n_milestone"] >= 1 and v["n_pruned"] <= 2 and v["n_resume"] >= 10
                 for v in inv.values()) and inv["RK"]["n_verify_failed"] >= 1
    res.update(resume_points={"RS": ok_rs, "RK": ok_rk}, invariants=inv, invariants_ok=ok_inv)
    out.update(results=res, pass_=bool(ok and ok_rs and ok_rk and ok_inv),
               tolerance="resume diff <= 3x run-to-run noise (+1e-6 params, +1e-4 loss), as protocol 11")
    out["pass"] = out.pop("pass_")
    guard.atomic_write_json(ART / f"resume_check2_{guard.utc_stamp()}.json", out)
    print(json.dumps(out, indent=1, default=str))
    sys.exit(0 if out["pass"] else 1)


if __name__ == "__main__":
    main()
