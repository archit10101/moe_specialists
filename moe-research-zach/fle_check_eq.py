#!/usr/bin/env python
"""T-EQ (prereg 0/10): compare the fle_train TOKEN-4 seed-0 replay with the reused lfr final_TOKEN-4_s0 metrics.
Pass: step-0 lm/aux/z bit-identical (as logged) and |lm| differences at steps 50 and 100 <= 0.01 nats."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402
from csr.fle_paths import ART, T4_S0_RUN  # noqa: E402


def main():
    guard.check_root()
    run = guard.inside_root(sys.argv[1])
    ref = {r["step"]: r for r in map(json.loads, (guard.ROOT / T4_S0_RUN / "metrics.jsonl").read_text().splitlines())
           if "step" in r}
    new = {r["step"]: r for r in map(json.loads, (run / "metrics.jsonl").read_text().splitlines()) if "step" in r}
    rows = {s: {"ref_lm": ref[s]["lm"], "new_lm": new[s]["lm"], "diff": new[s]["lm"] - ref[s]["lm"],
                "ref_aux": ref[s]["aux"], "new_aux": new[s]["aux"], "ref_z": ref[s]["z"], "new_z": new[s]["z"]}
            for s in (0, 50, 100) if s in new}
    ok0 = rows[0]["ref_lm"] == rows[0]["new_lm"] and rows[0]["ref_aux"] == rows[0]["new_aux"] and \
        rows[0]["ref_z"] == rows[0]["new_z"]
    ok = ok0 and all(abs(rows[s]["diff"]) <= 0.01 for s in (50, 100))
    ref_cfg = json.loads((guard.ROOT / T4_S0_RUN / "config.json").read_text())
    new_cfg = json.loads((run / "config.json").read_text())
    same = {k: ref_cfg[k] == new_cfg[k] for k in ("model", "train", "seed", "budget_tokens")}
    ref_st = json.loads((guard.ROOT / T4_S0_RUN / "status.json").read_text())
    new_st = json.loads((run / "status.json").read_text())
    out = {"time": guard.now(), "run": str(run.relative_to(guard.ROOT)), "rows": rows, "step0_identical": ok0,
           "config_fields_identical": same, "plan_sha_identical": ref_st["plan_sha256"] == new_st["plan_sha256"],
           "params_identical": ref_st["params"] == new_st["params"], "pass": bool(ok and all(same.values())
                                                                              and ref_st["plan_sha256"] == new_st["plan_sha256"]
                                                                              and ref_st["params"] == new_st["params"])}
    guard.atomic_write_json(ART / "equivalence_check.json", out, overwrite=False, mode=0o400)
    print(json.dumps(out, indent=1))
    sys.exit(0 if out["pass"] else 1)


if __name__ == "__main__":
    main()
