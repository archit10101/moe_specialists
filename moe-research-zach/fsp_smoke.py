#!/usr/bin/env python
"""fspv2 end-to-end smoke test (engineering; added after the resume check exposed a compiled-training dtype bug).
Runs every training condition for ~5 production-shape steps on the main stream with the real forecast cache, compiled,
then the end-of-run CALIB-SEARCH evaluation, via scripts/gpu_guarded.sh. Results are engineering checks only (no
selection use). Writes artifacts/forecast-specialist-pretraining-v2/smoke_<stamp>.json."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from csr import guard  # noqa: E402

CASES = [("TOKEN-4", {}, None), ("EDL", {}, 5e-4), ("DOMAIN-AUX", {}, None), ("FORECAST-SPECIALIST", {}, None),
         ("FORECAST-SPECIALIST", {"fc_horizon": 1, "anneal_shape": "cosine", "router_lr_mult": 3.0}, None),
         ("LABEL-ORACLE", {}, None), ("SHUFFLED", {}, None), ("WRONG-DRAFT", {}, None), ("FORECAST-NOPART", {}, None)]


def main():
    guard.check_root()
    import fsp_driver as drv
    from csr.fsp_lib import ART
    tag = sys.argv[1] if len(sys.argv) > 1 else "v1"
    out = {}
    for i, (cond, ch, beta) in enumerate(CASES):
        name = f"smoke{tag}_{i}_{cond}"
        cfg = drv.main_cfg(name, cond, 100, 5 * 64 * 1000, {**drv.BASE, **ch}, ["calib_search"], [1.0], beta,
                           category="checks_and_tests")
        p = drv.write_cfg("smoke", name, cfg)
        rc = drv.launch(f"fspv2 smoke {cond}", "fsp_train.py",
                        ["--config", str(p.relative_to(ROOT)), "--device", "cuda", "--execute"])
        rd = drv.find_run_dir(name)
        st = json.loads((rd / "status.json").read_text()) if rd and (rd / "status.json").exists() else {}
        out[name] = {"rc": rc, "status": st.get("status"), "reason": (st.get("reason") or "")[-600:],
                     "run_dir": str(rd.relative_to(ROOT)) if rd else None}
        print(name, rc, st.get("status"), (st.get("reason") or "")[-300:], flush=True)
    ok = all(v["status"] == "COMPLETE" for v in out.values())
    guard.atomic_write_json(ART / f"smoke_{guard.utc_stamp()}.json", {"time": guard.now(), "all_pass": ok,
                                                                      "cases": out})
    print("ALL PASS" if ok else "FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
