#!/usr/bin/env python
"""asrs2 base-trajectory match of a continuation run against the ordinary continuation (prereg C1).

In-process, the asrs2 base IS the ordinary base (identical code path; bitwise on CPU, T1; within GPU nondeterminism,
gpu_check). This job adds the cross-process comparison with the recorded ccsr1 S3 ordinary continuation (same start,
rows, schedule): per-step training loss, base update count, row hashes, and SEL CE at step 1111 against the ccsr1 S3
base re-measured on SEL by asrs1 (3.62277). Pass: 793 base updates; row hashes equal to the plan; the latest GPU check
passed; |mean per-step loss difference| <= 0.002; |SEL CE difference| <= 0.003.
  asrs2_basematch.py --run RUN_ID   -> ART/base_match_<RUN_ID>.json"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402


def main():
    from csr import asrs2_paths as P
    from csr.lfr_lib import main_stream
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    rd = guard.ROOT / "runs" / a.run
    M = [json.loads(x) for x in open(rd / "metrics.jsonl")]
    M = [m for m in M if "rows_sha" in m]
    s3 = {r["step"]: r["lm"] for r in (json.loads(x) for x in open(guard.ROOT / P.CCSR1_S3_RUN / "metrics.jsonl")) if "lm" in r}
    _, steps, _ = main_stream(245e6)
    plan_rows = [hashlib.sha256(json.dumps(steps[s]).encode()).hexdigest()[:16] for s in range(P.START, P.END)]
    d = np.asarray([m["lm"] - s3[m["step"]] for m in M])
    st = json.loads((rd / "status.json").read_text())
    ref_sel = json.loads((guard.ROOT / "artifacts/csr-alternating-semantic-residual-specialization-v1/stage1.json").read_text())
    ref_base = 3.6227712345149667
    for k in ("references",):
        r = ref_sel.get(k, {})
        if isinstance(r, dict) and "ordinary_base" in r:
            ref_base = float(r["ordinary_base"])
    fin = sorted((P.ANALYSIS).glob(f"{a.run}_m1111.json"))
    sel = json.loads(fin[-1].read_text())["ce_base"] if fin else None
    gc = sorted(P.ART.glob("gpu_check_*.json"))
    gpass = bool(gc) and json.loads(gc[-1].read_text())["pass"]
    win = {f"{i}-{i + 99}": float(d[i:i + 100].mean()) for i in range(0, len(d), 100)}
    out = {"time": guard.now(), "run": a.run, "n_steps": len(M), "base_updates": st["n_upd"].get("base"),
           "rows_equal_plan": [m["rows_sha"] for m in M] == plan_rows,
           "mean_dlm_vs_ccsr1_S3": float(d.mean()), "mean_abs_dlm": float(np.abs(d).mean()), "dlm_by_100": win,
           "sel_ce_base_1111": sel, "sel_ce_ccsr1_S3_base": ref_base,
           "sel_diff": None if sel is None else sel - ref_base, "gpu_check_pass": gpass}
    out["pass"] = bool(out["base_updates"] == P.END - P.START and out["rows_equal_plan"] and gpass
                       and abs(out["mean_dlm_vs_ccsr1_S3"]) <= 0.002 and sel is not None and abs(out["sel_diff"]) <= 0.003)
    guard.atomic_write_json(P.ART / f"base_match_{a.run}.json", out, mode=0o400)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
