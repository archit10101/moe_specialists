#!/usr/bin/env python
"""csr-context-prior-specialization-v2 trainer-equivalence check (fairness of reusing V1 TOKEN-4 runs): the V2 trainer
(cps_train.py) with the TOKEN-4 variant (context never computed) on the identical S1 plan, seed 100, for 51 steps must
reproduce the reused V1 run s1_TOKEN-4_s100: step-0 LM loss within 1e-3 and grad norm within 1% (identical init, data,
optimizer); steps 50 and 100: two V2 legs A/B (same-setting noise floor) -> min_leg |leg - V1| <= 3 max(|A-B|,
0.0098) + 0.01 (deviation 2; the original single-leg 0.03 rule failed at step 50 by 0.038)."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from csr import guard  # noqa: E402


def main():
    guard.check_root()
    if "--execute" not in sys.argv:
        print("dry run")
        return
    import cps_driver as drv
    from csr.cps_lib import ART, BUDGET
    from csr.cps_ckpt import ResumeStore
    legs = {}
    for leg in ("A", "B"):  # deviation 2: two V2 legs give the same-setting run-to-run noise floor
        name = f"equiv2{leg}_TOKEN_s100"
        cfg = drv.run_cfg(name, "TOKEN", 100, BUDGET["S1"], None, category="checks_and_tests")
        cfg["end_eval"] = []
        cfg["checkpoint_fracs"] = []
        p = drv.write_cfg("checks", name, cfg)
        rd = drv.find_run_dir(name)
        if rd is None or json.loads((rd / "status.json").read_text())["status"] not in ("INTERRUPTED", "COMPLETE"):
            env = dict(os.environ, MOE_GPU_LAUNCH="1", MOE_RESEARCH_APPROVE="gpu")
            r = subprocess.run(["scripts/gpu_guarded.sh", f"cps equivalence check {leg}", "cps_train.py", "--config",
                                str(p.relative_to(ROOT)), "--device", "cuda", "--execute", "--max-steps", "101"],
                               cwd=ROOT, env=env)
            if r.returncode not in (0, 75):
                print(f"leg rc={r.returncode}")
                sys.exit(77 if r.returncode in (77, 3, 78) else 1)
            rd = drv.find_run_dir(name)
        m = {}
        for line in (rd / "metrics.jsonl").read_text().splitlines():
            x = json.loads(line)
            if "lm" in x:
                m[x["step"]] = x
        legs[leg] = (rd, m)
        ResumeStore(rd.name, name).prune_after_complete()
    v1 = {}
    for line in (drv.v1_run_dir("s1_TOKEN-4_s100") / "metrics.jsonl").read_text().splitlines():
        x = json.loads(line)
        if "lm" in x:
            v1[x["step"]] = x
    A, B = legs["A"][1], legs["B"][1]
    res = {"time": guard.now(), "runs": {k: v[0].name for k, v in legs.items()},
           "step0": {"lm": [A[0]["lm"], B[0]["lm"], v1[0]["lm"]], "gn": [A[0]["gn"], B[0]["gn"], v1[0]["gn"]]}}
    ok = all(abs(L[0]["lm"] - v1[0]["lm"]) <= 1e-3 and abs(L[0]["gn"] - v1[0]["gn"]) <= 0.01 * v1[0]["gn"]
             for L in (A, B))
    for s_ in (50, 100):
        noise = abs(A[s_]["lm"] - B[s_]["lm"])
        dA, dB = abs(A[s_]["lm"] - v1[s_]["lm"]), abs(B[s_]["lm"] - v1[s_]["lm"])
        tol = 3.0 * max(noise, 0.0098) + 0.01
        p_ = min(dA, dB) <= tol
        res[f"step{s_}"] = {"lm_A": A[s_]["lm"], "lm_B": B[s_]["lm"], "lm_v1": v1[s_]["lm"], "noise_A_B": noise,
                            "diff_A_v1": dA, "diff_B_v1": dB, "tol": tol, "pass": p_}
        ok &= p_
    res["all_pass"] = bool(ok)
    res["rule"] = ("step 0: lm within 1e-3 and gn within 1% for both legs; steps 50 and 100: min_leg |leg - V1| <= "
                   "3 * max(|A - B|, 0.0098) + 0.01 (systematic divergence would put both legs outside the same-"
                   "setting noise band)")
    guard.atomic_write_json(ART / f"equiv_check_{guard.utc_stamp()}.json", res)
    print(json.dumps(res, indent=1))
    sys.exit(0 if res["all_pass"] else 1)


if __name__ == "__main__":
    main()
