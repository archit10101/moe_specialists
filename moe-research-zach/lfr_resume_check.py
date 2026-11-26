#!/usr/bin/env python
"""csr-latent-forecast-router-distill-v1: checkpoint/resume equivalence (protocol 14 T14, 15.1). Production-shape
CF-FORECAST-STUDENT runs (base knobs + latent consistency, 30 steps: P1 0-11, P2 12-17, P3 18-29; cadence 3 steps;
phase-boundary checkpoints; teacher strip at P3):
  U1, U2  uninterrupted (U2 - U1 = run-to-run noise floor)
  RS      cooperative SIGINT before step 14 (P2) -> EXTERNALLY_STOPPED (rc 77), verified checkpoint at 14 -> resumed
  RK      SIGKILL before step 22 (P3, after the strip) -> newest checkpoint (21) deliberately corrupted -> resume must
          reject it and fall back to step 18 -> resumed (re-strips the teacher)
Pass: RS and RK final weights within 3x the U2-U1 noise (+1e-6) and per-step LM loss within 3x the spread (+1e-4);
resume points as expected; ledger / index / rotation / prune invariants. Each leg via scripts/gpu_guarded.sh."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from csr import guard  # noqa: E402

N_STEPS_TOKENS = 30 * 63_000


def leg(purpose, cfg_rel, extra):
    env = dict(os.environ, MOE_GPU_LAUNCH="1", MOE_RESEARCH_APPROVE="gpu")
    r = subprocess.run(["scripts/gpu_guarded.sh", purpose, "lfr_train.py", "--config", cfg_rel, "--device", "cuda",
                        "--execute", *extra], cwd=ROOT, env=env)
    return r.returncode


def main():
    guard.check_root()
    if "--execute" not in sys.argv:
        print("dry run")
        return
    import torch
    import lfr_driver as drv
    from csr import lfr_search as S
    from csr.lfr_ckpt import LEDGER, ResumeStore
    from csr.lfr_lib import ART
    tag = os.environ.get("LFR_RESUME_TAG", "b")
    knobs = S.apply(S.BASE, {"c_lc": 0.05})
    names, cfgs = {}, {}
    for t in ("U1", "U2", "RS", "RK"):
        n = f"resumecheck{tag}_{t}"
        cfg = drv.run_cfg(n, "CF-FORECAST-STUDENT", 100, N_STEPS_TOKENS, knobs, category="checks_and_tests")
        cfg["diag"] = None
        cfg["end_eval"] = []
        cfg["cf_audit_every"] = 5
        names[t], cfgs[t] = n, str(drv.write_cfg("checks", n, cfg).relative_to(ROOT))
    out = {"time": guard.now(), "legs": {}}
    for t in ("U1", "U2"):
        out["legs"][t] = leg(f"lfr resume check {t}", cfgs[t], [])
    rc = leg("lfr resume check RS leg1", cfgs["RS"], ["--test-signal-at-step", "14"])
    rs_dir = drv.find_run_dir(names["RS"])
    st1 = json.loads((rs_dir / "status.json").read_text())
    out["legs"]["RS1"] = {"rc": rc, "status": st1["status"], "reason": st1.get("reason")}
    rs_idx1 = ResumeStore(rs_dir.name, "").index()
    out["legs"]["RS2"] = leg("lfr resume check RS leg2", cfgs["RS"], ["--run-dir", str(rs_dir.relative_to(ROOT))])
    rc = leg("lfr resume check RK leg1", cfgs["RK"], ["--test-kill-at-step", "22"])
    rk_dir = drv.find_run_dir(names["RK"])
    stk = json.loads((rk_dir / "status.json").read_text())
    store = ResumeStore(rk_dir.name, "")
    idx = store.index()
    out["legs"]["RK1"] = {"rc": rc, "status_after_kill": stk["status"],
                          "index_steps_after_kill": [e["step"] for e in idx["entries"]]}
    newest = ROOT / idx["entries"][-1]["path"] / "state.pt"
    with open(newest, "r+b") as f:  # deliberate corruption (test only)
        f.seek(newest.stat().st_size // 2)
        b = f.read(1)
        f.seek(-1, 1)
        f.write(bytes([b[0] ^ 0xFF]))
    out["legs"]["RK1"]["corrupted"] = idx["entries"][-1]["path"]
    out["legs"]["RK2"] = leg("lfr resume check RK leg2", cfgs["RK"], ["--run-dir", str(rk_dir.relative_to(ROOT))])

    def load(t):
        rd = drv.find_run_dir(names[t])
        s = json.loads((rd / "summary.json").read_text())
        sd = torch.load(ROOT / s["ckpts"]["p100"]["path"] / "state.pt", map_location="cpu", weights_only=False)["model"]
        lm = {}
        for line in (rd / "metrics.jsonl").read_text().splitlines():
            r = json.loads(line)
            if "lm" in r:
                lm[r["step"]] = r["lm"]
        return sd, lm, s, (rd / "log.txt").read_text(), rd
    U1, U2, RS, RK = load("U1"), load("U2"), load("RS"), load("RK")
    dmax = lambda a, b: max(float((a[k].float() - b[k].float()).abs().max()) for k in a)
    noise = dmax(U2[0], U1[0])
    common = sorted(set(U1[1]) & set(U2[1]))
    lnoise = max(abs(U2[1][s] - U1[1][s]) for s in common)
    res = {"param_noise_u2_u1": noise, "loss_noise_u2_u1": lnoise, "steps_compared": common}
    ok = True
    for t, R in (("RS", RS), ("RK", RK)):
        d = dmax(R[0], U1[0])
        ld = max(abs(R[1][s] - U1[1][s]) for s in common if s in R[1])
        p = d <= 3 * noise + 1e-6 and ld <= 3 * lnoise + 1e-4 and R[2]["legs"] >= 2
        res[t] = {"param_maxdiff_vs_u1": d, "loss_maxdiff_vs_u1": ld, "legs": R[2]["legs"], "pass": p,
                  "resume_log": [ln[26:] for ln in R[3].splitlines() if "resume" in ln or "P3" in ln]}
        ok &= p
    ok_rs = out["legs"]["RS1"]["rc"] == 77 and out["legs"]["RS1"]["status"] == "EXTERNALLY_STOPPED" and \
        rs_idx1["entries"][-1]["step"] == 14 and any("resumed at step 14" in x for x in res["RS"]["resume_log"])
    ok_rk = out["legs"]["RK1"]["rc"] != 0 and out["legs"]["RK1"]["status_after_kill"] == "RUNNING" and \
        out["legs"]["RK1"]["index_steps_after_kill"] == [18, 21] and \
        any("FAILED sha256 verification" in x for x in res["RK"]["resume_log"]) and \
        any("resumed at step 18" in x for x in res["RK"]["resume_log"]) and \
        any("teacher stripped" in x for x in res["RK"]["resume_log"])
    led = [ln.split("\t") for ln in LEDGER.read_text().splitlines()[1:]]
    inv = {}
    for t, R in (("U1", U1), ("RS", RS), ("RK", RK)):
        rid = R[4].name
        kinds = [r[3] for r in led if r[1] == rid]
        allf = list((ROOT / ".tmp/latent-forecast-router-distill-v1/slots" / rid).glob("resume_s*"))
        left = [f for f in allf if ".rejected-" not in f.name]  # rejected (corrupted) checkpoints are kept aside
        inv[t] = {"n_resume": kinds.count("resume"), "n_rotated": kinds.count("rotated_out"),
                  "n_pruned": kinds.count("pruned_after_complete"), "n_milestone": kinds.count("milestone"),
                  "n_verify_failed": kinds.count("verify_failed"), "resume_files_left": len(left),
                  "rejected_kept": len(allf) - len(left)}
    ok_inv = all(v["resume_files_left"] == 0 and v["n_milestone"] >= 1 and v["n_resume"] >= 9 for v in inv.values()) \
        and inv["RK"]["n_verify_failed"] >= 1 and inv["RK"]["rejected_kept"] == 1
    res.update(resume_points={"RS": ok_rs, "RK": ok_rk}, invariants=inv, invariants_ok=ok_inv)
    out["results"] = res
    out["pass"] = bool(ok and ok_rs and ok_rk and ok_inv)
    out["tolerance"] = "resume diff <= 3x run-to-run noise (+1e-6 params, +1e-4 loss)"
    guard.atomic_write_json(ART / f"resume_check_{guard.utc_stamp()}.json", out)
    print(json.dumps(out, indent=1, default=str))
    sys.exit(0 if out["pass"] else 1)


if __name__ == "__main__":
    main()
