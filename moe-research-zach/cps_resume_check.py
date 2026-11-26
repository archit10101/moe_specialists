#!/usr/bin/env python
"""csr-context-prior-specialization-v2 checkpoint/resume equivalence (V1 15.1 procedure on the V2 trainer): base
candidate (C03 knobs), 30 steps (P1 0-11, P2 12-17, P3 18-29), CF audits every 5 steps:
  U1, U2  uninterrupted (U2 - U1 = run-to-run noise floor)
  RS      cooperative SIGINT at step 14 (P2) -> EXTERNALLY_STOPPED (rc 77) with a verified checkpoint at 14 -> resumed
          (must strip the teacher on entering P3 after the resume)
Pass: RS final weights within 3x the U2-U1 noise (+1e-6) and per-step LM loss within 3x the spread (+1e-4); resume
point 14; rotation / prune invariants. (The corrupted-checkpoint fallback path is the unchanged V1 store code, verified
by the V1 resume check.) Each leg via scripts/gpu_guarded.sh."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from csr import guard  # noqa: E402

N_TOK = 30 * 63_000


def leg(purpose, cfg_rel, extra):
    env = dict(os.environ, MOE_GPU_LAUNCH="1", MOE_RESEARCH_APPROVE="gpu")
    return subprocess.run(["scripts/gpu_guarded.sh", purpose, "cps_train.py", "--config", cfg_rel, "--device", "cuda",
                           "--execute", *extra], cwd=ROOT, env=env).returncode


def main():
    guard.check_root()
    if "--execute" not in sys.argv:
        print("dry run")
        return
    import torch
    import cps_driver as drv
    from csr import cps_search as S
    from csr.cps_ckpt import LEDGER, ResumeStore
    from csr.cps_lib import ART
    names, cfgs = {}, {}
    for t in ("U1", "U2", "RS"):
        n = f"resumecheck_{t}"
        cfg = drv.run_cfg(n, "CAND", 100, N_TOK, S.BASE, category="checks_and_tests")
        cfg["end_eval"] = []
        cfg["cf_audit_every"] = 5
        names[t], cfgs[t] = n, str(drv.write_cfg("checks", n, cfg).relative_to(ROOT))
    out = {"time": guard.now(), "legs": {}}
    for t in ("U1", "U2"):
        if drv.find_run_dir(names[t]) is None:
            out["legs"][t] = leg(f"cps resume check {t}", cfgs[t], [])
    rs_dir = drv.find_run_dir(names["RS"])
    if rs_dir is None:
        rc = leg("cps resume check RS leg1", cfgs["RS"], ["--test-signal-at-step", "14"])
        rs_dir = drv.find_run_dir(names["RS"])
        st1 = json.loads((rs_dir / "status.json").read_text())
        out["legs"]["RS1"] = {"rc": rc, "status": st1["status"], "reason": st1.get("reason"),
                              "index_steps": [e["step"] for e in ResumeStore(rs_dir.name, "").index()["entries"]]}
        out["legs"]["RS2"] = leg("cps resume check RS leg2", cfgs["RS"], ["--run-dir", str(rs_dir.relative_to(ROOT))])

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
    U1, U2, RS = load("U1"), load("U2"), load("RS")
    dmax = lambda a, b: max(float((a[k].float() - b[k].float()).abs().max()) for k in a)
    noise = dmax(U2[0], U1[0])
    common = sorted(set(U1[1]) & set(U2[1]))
    lnoise = max(abs(U2[1][s] - U1[1][s]) for s in common)
    d = dmax(RS[0], U1[0])
    ld = max(abs(RS[1][s] - U1[1][s]) for s in common if s in RS[1])
    rlog = [ln[26:] for ln in RS[3].splitlines() if "resume" in ln or "P3" in ln]
    ok_eq = d <= 3 * noise + 1e-6 and ld <= 3 * lnoise + 1e-4 and RS[2]["legs"] >= 2
    ok_pt = any("resumed at step 14" in x for x in rlog) and any("stripped" in x for x in rlog)
    if "RS1" in out["legs"]:
        ok_pt = ok_pt and out["legs"]["RS1"]["rc"] == 77 and out["legs"]["RS1"]["status"] == "EXTERNALLY_STOPPED"
    led = [ln.split("\t") for ln in LEDGER.read_text().splitlines()[1:]]
    inv = {}
    for t, R in (("U1", U1), ("RS", RS)):
        rid = R[4].name
        kinds = [r[3] for r in led if r[1] == rid]
        left = [f for f in (ROOT / ".tmp/context-prior-specialization-v2/slots" / rid).glob("resume_s*")
                if ".rejected-" not in f.name]
        inv[t] = {"n_resume": kinds.count("resume"), "n_milestone": kinds.count("milestone"),
                  "n_pruned": kinds.count("pruned_after_complete"), "resume_files_left": len(left)}
    ok_inv = all(v["resume_files_left"] == 0 and v["n_milestone"] >= 1 and v["n_resume"] >= 3 for v in inv.values())
    out["results"] = {"param_noise_u2_u1": noise, "loss_noise_u2_u1": lnoise, "RS_param_maxdiff": d,
                      "RS_loss_maxdiff": ld, "resume_log": rlog, "invariants": inv, "equivalence_ok": ok_eq,
                      "resume_point_ok": ok_pt, "invariants_ok": ok_inv}
    out["pass"] = bool(ok_eq and ok_pt and ok_inv)
    guard.atomic_write_json(ART / f"resume_check_{guard.utc_stamp()}.json", out)
    if out["pass"]:  # disk policy (deviation 1): check-run milestones are not results; delete after a pass (ledgered)
        import shutil
        from csr.cps_ckpt import ledger
        for t, R in (("U1", U1), ("U2", U2), ("RS", RS)):
            for tag, e in R[2]["ckpts"].items():
                if (ROOT / e["path"]).exists():
                    shutil.rmtree(ROOT / e["path"])
                    ledger({"time": guard.now(), "run_id": R[4].name, "run_name": names[t],
                            "kind": "milestone_deleted_disk", "path": e["path"], "step": e["step"], "seen": e["seen"],
                            "sha256": e["sha256"], "note": "resume-check milestone after PASS (deviation 1)"})
    print(json.dumps(out, indent=1, default=str))
    sys.exit(0 if out["pass"] else 1)


if __name__ == "__main__":
    main()
