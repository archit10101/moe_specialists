#!/usr/bin/env python
"""csr-gradient-consistent-latent-edl-v1 screening (prereg sections 6, 8, 9). Seed 100, FINAL plan, 50M = step 793.

  gc_screen.py pilot   --rep R      : 80-step pilot at beta 5e-3 (diag every 10) -> rho, beta grid (screen/pilot_R.json)
  gc_screen.py grid    --rep R      : 3 screening runs (0.5/1/2 % targets) + dev evals at s50
  gc_screen.py edl                  : explicit-label EDL oracle reference (beta 5e-3) + dev eval
  gc_screen.py t4dev                : dev eval of the reused fle seed-100 TOKEN-4 s50 milestone
  gc_screen.py choose               : beta choice per representation (CALIB-SEARCH) -> screen/beta_choice.json
  gc_screen.py shuf                 : shuffled-paper control at the winner representation's chosen beta + dev eval
  gc_screen.py traj                 : descriptive dev evals at s5/s20 (chosen candidates, T4, EDL, SHUF)
  gc_screen.py freeze               : opens CALIB-SELECTION once, evaluates, writes frozen.json (0400), unseals labels
Decision source: discovery_decision_r1.json if present (engineering revision 1) else discovery_decision.json."""
import argparse
import json
import math
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402
from csr.gc_paths import ART, EDL_BETA, S50_STEP, T4_S100_RUN, TMP  # noqa: E402

SDIR = ART / "screen"
RUNS = ART / "run_dirs.json"
TARGETS = {"r050": 0.005, "r100": 0.01, "r200": 0.02}


def decision():
    p = ART / "discovery_decision_r1.json"
    return json.loads((p if p.exists() else ART / "discovery_decision.json").read_text())


def run_dirs():
    return json.loads(RUNS.read_text()) if RUNS.exists() else {}


def remember(rn):
    from csr.gc_paths import RUN_PREFIX
    cands = sorted((guard.ROOT / "runs").glob(f"*_{RUN_PREFIX}{rn}"))
    d = run_dirs()
    if cands and rn not in d:
        d[rn] = str(cands[-1].relative_to(guard.ROOT))
        guard.atomic_write_json(RUNS, d, overwrite=True)
    return d.get(rn)


def drop_slot(rn, why):
    rd = run_dirs().get(rn)
    if not rd:
        return
    st = json.loads((guard.ROOT / rd / "status.json").read_text())
    from csr.gc_ckpt import verify_dir
    for c in st.get("ckpts", {}).values():
        if not verify_dir(guard.ROOT / c["path"]):
            raise guard.GuardError(f"milestone {c['path']} fails verification; slot kept")
    sl = TMP / "slots" / Path(rd).name
    if sl.exists():
        nb = sum(p.stat().st_size for p in sl.rglob("*") if p.is_file())
        shutil.rmtree(sl)
        with open(ART / "deleted_artifacts.tsv", "a") as f:
            f.write(f"{guard.now()}\t{sl.relative_to(guard.ROOT)}\t{nb}\t{why}\n")


def train(rn, cond, kind, diag_every, max_steps, seed=100):
    from gc_configs import make
    from gc_driver import run_job
    p, _ = make(rn, seed, cond, kind, diag_every=diag_every)
    prev = run_dirs().get(rn)
    args = ["gc_train.py", "--config", str(p), "--max-steps", str(max_steps)] + (["--run-dir", prev] if prev else [])
    remember_before = set((guard.ROOT / "runs").glob("*"))
    run_job(f"train_{rn}", args, gpu=True, expect=(75, 0), disk_gb=4.0)
    remember(rn)
    _ = remember_before
    return run_dirs()[rn]


def dev(rn, tag="s50", run_dir=None):
    from gc_driver import run_job
    rd = run_dir or run_dirs()[rn]
    run_job(f"dev_{rn}_{tag}", ["gc_evaluate.py", "dev", "--run-dir", rd, "--run-name", rn, "--tag", tag], gpu=True,
            disk_gb=0.5)
    return json.loads((ART / "evals" / rn / f"dev_{tag}.json").read_text())


def pilot(rep):
    import numpy as np
    from gc_configs import gc_cond
    out = SDIR / f"pilot_{rep}.json"
    if out.exists():
        return json.loads(out.read_text())
    tab = decision()["tables"][rep]["path"]
    rn = f"p_{rep}_s100"
    rd = train(rn, gc_cond(tab, EDL_BETA), "pilot", 10, 80)
    m = [json.loads(x)["diag"] for x in (guard.ROOT / rd / "metrics.jsonl").read_text().splitlines() if "diag" in x]
    ratios = [d["gn_aux"] / d["gn_lm"] for d in m if d["step"] >= 20]
    rho = float(np.median(ratios))
    betas = {k: float(f"{EDL_BETA * t / rho:.2g}") for k, t in TARGETS.items()}
    rec = {"rep": rep, "table": tab, "pilot_beta": EDL_BETA, "rho": rho, "ratios": ratios, "betas": betas,
           "run_dir": rd, "time": guard.now()}
    guard.atomic_write_json(out, rec, mode=0o400)
    drop_slot(rn, "pilot run resumable slot (pilot complete, metrics kept)")
    return rec


def grid(rep):
    from gc_configs import gc_cond
    pl = pilot(rep)
    for k, b in pl["betas"].items():
        rn = f"s_{rep}_{k}_s100"
        train(rn, gc_cond(pl["table"], b), "search", 25, S50_STEP)
        drop_slot(rn, "screening run resumable slot (50M milestone verified; screening runs never resume)")
        dev(rn)


def ratio_stats(rd):
    import numpy as np
    m = [json.loads(x)["diag"] for x in (guard.ROOT / rd / "metrics.jsonl").read_text().splitlines() if "diag" in x]
    r = [(d["step"], d["gn_aux"] / d["gn_lm"] if d.get("gn_lm") else 0.0) for d in m if "gn_aux" in d]
    late = [v for s, v in r if 397 <= s <= 793]
    return {"median_ratio_25M_50M": float(np.median(late)) if late else None,
            "median_ratio_all": float(np.median([v for _, v in r])) if r else None, "n": len(r)}


def summarize(ev, t4ce):
    a = ev["audit"]
    return {"ce": ev["calib_search_B"]["ce"] if "calib_search_B" in ev else ev["calib_sel"]["ce"],
            "dce": (ev["calib_search_B"]["ce"] if "calib_search_B" in ev else ev["calib_sel"]["ce"]) - t4ce,
            "fsi_e5": a["e5"].get("fsi"), "routing_entropy": a["routing_entropy"],
            "load_min_x_uniform": a["load_min_x_uniform"], "load_max_x_uniform": a["load_max_x_uniform"],
            "n_below_0.1_uniform": a["n_below_0.1_uniform"],
            "mi_e5": a["e5"]["routing"]["mi_mean"],
            **{f"mi_{k}": v["routing"]["mi_mean"] for k, v in a.items() if k.startswith("pseudo_")},
            **{f"fsi_{k}": v.get("fsi") for k, v in a.items() if k.startswith("pseudo_")}}


def valid(s, ce_lim):
    reasons = []
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (s["ce"], s["fsi_e5"] or 0.0)):
        reasons.append("non-finite")
    if s["n_below_0.1_uniform"] > 0:
        reasons.append("starvation")
    if s["routing_entropy"] < 0.45 or s["load_max_x_uniform"] > 4.0:
        reasons.append("collapse")
    if s["dce"] > ce_lim:
        reasons.append("ce")
    return reasons


def choose():
    out = SDIR / "beta_choice.json"
    if out.exists():
        return json.loads(out.read_text())
    t4 = json.loads((ART / "evals" / "s_T4_s100" / "dev_s50.json").read_text())
    t4ce = t4["calib_search_B"]["ce"]
    rec = {"t4_ce": t4ce, "t4": summarize(t4, t4ce), "reps": {}, "time": guard.now()}
    for rep in ("SEM", "GRAD", "COMBINED"):
        if decision()["chosen"].get(rep) is None:
            rec["reps"][rep] = {"chosen": None, "note": "no eligible discovery candidate"}
            continue
        pl = json.loads((SDIR / f"pilot_{rep}.json").read_text())
        rows = {}
        for k, b in pl["betas"].items():
            rn = f"s_{rep}_{k}_s100"
            ev = json.loads((ART / "evals" / rn / "dev_s50.json").read_text())
            s = summarize(ev, t4ce)
            s.update({"beta": b, "target": TARGETS[k], **ratio_stats(run_dirs()[rn])})
            s["reject"] = valid(s, 0.020)
            rows[k] = s
        acc = [k for k in rows if not rows[k]["reject"]]
        comp = [k for k in acc if rows[k]["dce"] <= 0.010]
        if comp:
            ch = max(comp, key=lambda k: (rows[k]["fsi_e5"], -rows[k]["ce"]))
        elif acc:
            ch = min(acc, key=lambda k: rows[k]["dce"])
        else:
            ch = None
        rec["reps"][rep] = {"rows": rows, "accepted": acc, "comparable": comp, "chosen": ch,
                            "beta": rows[ch]["beta"] if ch else None}
    guard.atomic_write_json(out, rec, mode=0o400)
    return rec


def shuf():
    from gc_configs import gc_cond
    dec = decision()
    w = dec["winner_gc"]
    bc = choose()
    b = bc["reps"][w]["beta"]
    if b is None:
        b = json.loads((SDIR / f"pilot_{w}.json").read_text())["betas"]["r100"]
    rn = "s_SHUF_s100"
    train(rn, gc_cond(dec["tables"]["SHUF"]["path"], b), "search", 25, S50_STEP)
    drop_slot(rn, "screening run resumable slot (50M milestone verified)")
    dev(rn)


def edl():
    rn = "s_EDL_s100"
    train(rn, {"edl_beta": EDL_BETA}, "search", 25, S50_STEP)
    drop_slot(rn, "screening run resumable slot (50M milestone verified)")
    dev(rn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pilot", "grid", "edl", "t4dev", "choose", "shuf", "traj", "freeze"])
    ap.add_argument("--rep")
    a = ap.parse_args()
    guard.check_root()
    from csr.gc_paths import check_prereg
    check_prereg()
    SDIR.mkdir(parents=True, exist_ok=True)
    if a.cmd == "pilot":
        print(json.dumps(pilot(a.rep), indent=1))
    elif a.cmd == "grid":
        grid(a.rep)
    elif a.cmd == "edl":
        edl()
    elif a.cmd == "t4dev":
        dev("s_T4_s100", "s50", T4_S100_RUN)
    elif a.cmd == "choose":
        print(json.dumps(choose(), indent=1, default=str))
    elif a.cmd == "shuf":
        shuf()
    elif a.cmd == "traj":
        bc = choose()
        names = [f"s_{r}_{v['chosen']}_s100" for r, v in bc["reps"].items() if v.get("chosen")]
        for rn in names + ["s_EDL_s100", "s_SHUF_s100"]:
            for tag in ("s5", "s20"):
                dev(rn, tag)
        for tag in ("s5", "s20"):
            dev("s_T4_s100", tag, T4_S100_RUN)
    elif a.cmd == "freeze":
        raise SystemExit("freeze is implemented in gc_freeze.py")


if __name__ == "__main__":
    main()
