#!/usr/bin/env python
"""fspv2 resume-equivalence check (protocol 11). Three guarded legs of a short FORECAST-SPECIALIST run (all three
curriculum phases inside 24 steps, real data + forecast cache, compiled, production batch shape):
  U1 uninterrupted, U2 uninterrupted (run-to-run noise floor), R interrupted at step 11 (mid-anneal) then resumed.
Pass rule (documented tolerance): max|theta_R - theta_U1| <= 3 x max|theta_U2 - theta_U1| + 1e-6, and the per-step LM
loss of R equals U1 within 3 x the U2-U1 loss spread + 1e-4. CPU orchestration; each leg via scripts/gpu_guarded.sh."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from csr import guard  # noqa: E402


def main():
    guard.check_root()
    import torch
    import fsp_driver as drv
    from csr.fsp_lib import ART
    fz_knobs = drv.BASE
    steps = 24
    budget = steps * 64 * 1000  # ~24 steps of 64 rows
    names = {}
    for tag in ("U1", "U2", "R"):
        name = f"resumecheck_{tag}"
        cfg = drv.main_cfg(name, "FORECAST-SPECIALIST", 100, budget, fz_knobs, [], [1.0], category="checks_and_tests")
        names[tag] = (name, drv.write_cfg("checks", name, cfg))
    state = drv.load_state()
    for tag in ("U1", "U2"):
        drv.train_run(state, names[tag][1], names[tag][0], f"fspv2 resume check {tag}")
    name, cp = names["R"]
    rec = state["runs"].setdefault(name, {"attempts": 0, "legs": 0, "cfg": str(cp.relative_to(ROOT))})
    if rec.get("status") != "COMPLETE":
        if not rec.get("run_dir"):
            rc = drv.launch("fspv2 resume check R leg1", "fsp_train.py",
                            ["--config", str(cp.relative_to(ROOT)), "--device", "cuda", "--execute", "--max-steps", "11"])
            rd = drv.find_run_dir(name)
            rec["run_dir"] = str(rd.relative_to(ROOT))
            rec["leg1_rc"] = rc
            drv.save_state(state)
        drv.train_run(state, cp, name, "fspv2 resume check R leg2")
    def load(tag):
        rd = ROOT / state["runs"][names[tag][0]]["run_dir"]
        s = json.loads((rd / "summary.json").read_text())
        sd = torch.load(ROOT / s["ckpts"]["p100"]["path"], map_location="cpu", weights_only=True)["model"]
        lm = {}
        for line in (rd / "metrics.jsonl").read_text().splitlines():
            r = json.loads(line)
            lm[r["step"]] = r["lm"]
        return sd, lm, s
    (u1, l1, s1), (u2, l2, s2), (r_, lr_, sr) = load("U1"), load("U2"), load("R")
    dmax = lambda a, b: max(float((a[k].float() - b[k].float()).abs().max()) for k in a)
    noise = dmax(u2, u1)
    dres = dmax(r_, u1)
    common = sorted(set(l1) & set(l2) & set(lr_))
    lnoise = max(abs(l2[s] - l1[s]) for s in common)
    lres = max(abs(lr_[s] - l1[s]) for s in common)
    ok = dres <= 3 * noise + 1e-6 and lres <= 3 * lnoise + 1e-4 and sr["legs"] >= 2
    out = {"time": guard.now(), "pass": ok, "param_maxdiff_resume_vs_u1": dres, "param_maxdiff_u2_vs_u1": noise,
           "loss_maxdiff_resume_vs_u1": lres, "loss_maxdiff_u2_vs_u1": lnoise, "steps_compared": common,
           "resumed_legs": sr["legs"], "interrupt_step": 11, "n_steps": s1["n_steps"],
           "tolerance": "resume diff <= 3x run-to-run noise (+1e-6 params, +1e-4 loss)"}
    guard.atomic_write_json(ART / f"resume_check_{guard.utc_stamp()}.json", out)
    print(json.dumps(out, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
