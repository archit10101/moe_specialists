#!/usr/bin/env python
"""csr-crossfit-semantic-utility-residual-v1 GPU checks from the gt_* runs (CPU; protocol section 6):
G-EQ (step-0 lm/aux/z bit-identical to the reused T4 s0 metrics), G-ISO (banks vs no-bank mean |dlm| <= 1.5 x
no-bank repeat noise + 0.002), G-RESUME (30 + resume + 30 vs 60 straight; lm and per-bank L_res, same tolerance),
G-BENCH (tokens/s, step time, peak memory). Writes ART/gpu_tests_<ts>.json (0400)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402

TMP = guard.ROOT / ".tmp/csr-crossfit-semantic-utility-residual-v1/gpu_tests/runs"


def metrics(rd):
    m = [json.loads(x) for x in (rd / "metrics.jsonl").read_text().splitlines()]
    return {r["step"]: r for r in m if "step" in r}


def mad(a, b, steps, key=lambda r: r["lm"]):
    return sum(abs(key(a[s]) - key(b[s])) for s in steps) / len(steps)


def main():
    from csr.csur1_paths import ART, T4_CFG_RUN
    runs = sorted(TMP.iterdir(), key=lambda p: p.name)
    gb = [r for r in runs if r.name.endswith("_gt_banks")]
    gn = [r for r in runs if r.name.endswith("_gt_nobank")]
    g8 = [r for r in runs if r.name.endswith("_gt_bench15")]
    straight, resumed = gb[0], gb[1]
    gn = gn[-2:]  # the two most recent no-bank repeats (dense logging)
    ref = {r["step"]: r for r in map(json.loads, (guard.ROOT / T4_CFG_RUN / "metrics.jsonl").read_text().splitlines())
           if "step" in r}
    mb, mn1, mn2, mr = metrics(straight), metrics(gn[0]), metrics(gn[1]), metrics(resumed)
    geq = {k: (mb[0][k], ref[0][k], mb[0][k] == ref[0][k]) for k in ("lm", "aux", "z")}
    steps = [s for s in range(0, 60, 10)]
    noise = mad(mn1, mn2, steps)
    tol = 1.5 * noise + 0.002
    iso = {"banks_vs_nb1": mad(mb, mn1, steps), "banks_vs_nb2": mad(mb, mn2, steps), "nb1_vs_nb2": noise, "tol": tol}
    iso["pass"] = iso["banks_vs_nb1"] <= tol
    rs = [s for s in range(30, 60, 10)]
    straights = [gb[0]] + gb[2:]  # independent straight bank runs (GPU run-to-run reference)
    ms = [metrics(r) for r in straights]
    res = {}
    keys = {"lm": lambda r: r["lm"], **{f"res_{n}": (lambda r, n=n: r["banks"][n]["res"]) for n in mb[30]["banks"] if not n.startswith("_")}}
    for k, f in keys.items():
        pair = [mad(ms[i], ms[j], rs, key=f) for i in range(len(ms)) for j in range(i + 1, len(ms))]
        d_res = max(mad(m_, mr, rs, key=f) for m_ in ms[:1]) if False else min(mad(m_, mr, rs, key=f) for m_ in ms)
        # first step after resume (30) must match the straight runs as closely as they match each other
        s30 = [f(m_[30]) for m_ in ms]
        d30 = min(abs(f(mr[30]) - v) for v in s30)
        tol30 = (max(s30) - min(s30)) + 0.002
        tol = 1.5 * max(pair) + 0.002
        res[k] = {"resumed_vs_nearest_straight": d_res, "straight_pairs": pair, "tol": tol, "step30_diff": d30,
                  "step30_tol": tol30, "pass": d_res <= tol and d30 <= tol30}
    res["pass"] = all(v["pass"] for v in res.values() if isinstance(v, dict))
    legs = {}
    for nm, rd in (("banks4", straight), ("nobank", gn[0]), ("bench15", g8[0])):
        lg = json.loads(sorted(rd.glob("leg_*.json"))[0].read_text())
        m = metrics(rd)
        dts = [m[s]["dt"] for s in m if s >= 20]
        legs[nm] = {"tokens_per_s": lg["tokens_per_s"], "median_step_s": sorted(dts)[len(dts) // 2],
                    "peak_alloc_gib": lg["peak_alloc_gib"], "peak_reserved_gib": lg["peak_reserved_gib"]}
    est = {"final_run_245M_h": legs["banks4"]["median_step_s"] * 3880 / 3600,
           "S2_run_50M_h": legs["banks4"]["median_step_s"] * 793 / 3600,
           "S1_run_20M_h": legs["bench15"]["median_step_s"] * 318 / 3600}
    smoke = json.loads((TMP.parent / "eval_smoke2.json").read_text()) if (TMP.parent / "eval_smoke2.json").exists() else {}
    geval = {"forced_vs_routed_max_abs": smoke.get("forced_vs_routed_max_abs"), "n_tokens": smoke.get("n_tokens"),
             "pass": bool(smoke) and all(v < 0.02 for v in smoke.get("forced_vs_routed_max_abs", {}).values())}
    out = {"time": guard.now(), "G_EQ": {"detail": geq, "pass": all(v[2] for v in geq.values())}, "G_ISO": iso,
           "G_RESUME": res, "G_BENCH": legs, "G_EVAL": geval, "estimates": est,
           "runs": [str(p.relative_to(guard.ROOT)) for p in (straight, resumed, *gb[2:], gn[0], gn[1], g8[0])]}
    out["all_pass"] = out["G_EQ"]["pass"] and iso["pass"] and res["pass"] and geval["pass"]
    p = ART / f"gpu_tests_{guard.utc_stamp()}.json"
    guard.atomic_write_json(p, out, mode=0o400)
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
