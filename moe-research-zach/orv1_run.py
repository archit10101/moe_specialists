#!/usr/bin/env python
"""oracle-residual-v1 orchestrator (prereg section 11). Idempotent; skips completed work; never relaunches after an
external stop; enforces the GPU caps and the disk reserve (via the driver).
  orv1_run.py gputests   G-RESUME on GPU (test-mode training jobs: straight x2 vs resumed legs)
  orv1_run.py pilot      LR selection on SEL (seed 0, 5M tokens; condition-agnostic) + C collapse ladder
  orv1_run.py stage1     seeds 0,1: train to 20M (joint A/B/C), evaluate EVAL (full) and SEL; frozen analysis
  orv1_run.py stage2     only if stage 1 promoted: continue to 50M, evaluate, frozen analysis
  orv1_run.py all        gputests -> pilot -> stage1 -> (stage2) -> final_analysis.json"""
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402
from csr import orv1_log as LG  # noqa: E402
from csr.orv1_paths import (ART, AUX_LADDER, BUDGETS, EVALS, GPU_CAP_H, LR_CANDIDATES, PILOT_TOKENS, RUNS,
                            STAGE1_SEEDS, STOP_MARKER, TMP, check_prereg)  # noqa: E402

PY = str(guard.ROOT / ".venv/bin/python")


def stage_of(job_id):
    for s in ("stage1", "stage2", "stage3"):
        if job_id.startswith(s):
            return s
    return "stage0"


def seed_of(job_id):
    for p in job_id.split("_"):
        if p.startswith("s") and p[1:].isdigit():
            return int(p[1:])
    return ""


def driver(job_id, argv, gpu=True, expect="0", disk_gb=2.0):
    cmd = [PY, "orv1_driver.py", "job", job_id, "--gpu" if gpu else "--cpu", "--expect-rc", expect,
           "--disk-gb", str(disk_gb), "--"] + argv
    LG.disk(job_id, disk_gb)
    t_iso, t0 = guard.now(), time.time()
    rc = subprocess.run(cmd, cwd=guard.ROOT).returncode
    dt = time.time() - t0
    st = json.loads((ART / "driver_state.json").read_text())
    j = st["jobs"].get(job_id, {})
    status = j.get("status", "UNKNOWN")
    if gpu and status != "UNKNOWN" and j.get("seconds") is not None and "end" in j and j.get("start", "") >= t_iso[:19]:
        LG.compute(job_id, stage_of(job_id), j["start"], j["end"], float(j["seconds"]), status)
    LG.manifest({"time": guard.now(), "job": job_id, "stage": stage_of(job_id), "seed": seed_of(job_id),
                 "status": status, "rc": j.get("rc", rc), "seconds": round(dt, 1),
                 "gpu_h": round(float(j.get("seconds", 0)) / 3600, 4) if gpu else 0, "cmd": " ".join(argv)})
    if rc == 77 or st.get("external_stop"):
        external_stop(job_id)
    if rc != 0:
        LG.set_state("FAILED", job=job_id, rc=rc)
        LG.write_resume("FAILED", f"{PY} orv1_run.py all", f"job {job_id} failed rc={rc}; see job_logs/{job_id}.log")
        sys.exit(rc)
    return j.get("rc", 0)


def external_stop(job_id):
    state = "INCOMPLETE_STOPPED_BY_EXTERNAL_OPERATOR"
    LG.set_state(state, job=job_id)
    LG.decision(f"job {job_id} stopped externally (signal / foreign GPU process / STOP marker)",
                "intentional GPU reclamation by someone outside this session (policy)", "stop all launchers; no relaunch",
                decision_="wait for explicit user permission")
    LG.write_resume(state, f"{PY} orv1_driver.py authorize-resume && {PY} orv1_run.py all",
                    "The last complete slot / milestone checkpoint is kept under artifacts/oracle-residual-v1/runs/<run>/;"
                    " the pipeline skips completed work and resumes training from slot.pt.")
    print(state, flush=True)
    sys.exit(77)


def gpu_cap(stage, need_h):
    tot, by = LG.gpu_hours()
    if tot + need_h > GPU_CAP_H["total"] or by.get(stage, 0.0) + need_h > GPU_CAP_H.get(stage, 1e9):
        LG.set_state("STOPPED_GPU_CAP", stage=stage, used=by)
        LG.write_resume("STOPPED_GPU_CAP", f"{PY} orv1_run.py all", f"GPU cap reached: {by}")
        sys.exit(3)


def legs(prefix, argv, disk_gb, need_h):
    n = 1
    st = json.loads((ART / "driver_state.json").read_text()) if (ART / "driver_state.json").exists() else {"jobs": {}}
    while f"{prefix}_leg{n}" in st["jobs"]:
        j = st["jobs"][f"{prefix}_leg{n}"]
        if j.get("rc") == 0 and j["status"] == "COMPLETE":
            return
        n += 1
    while True:
        if STOP_MARKER.exists():
            external_stop(prefix)
        gpu_cap(stage_of(prefix), need_h)
        rc = driver(f"{prefix}_leg{n}", argv, expect="0,75", disk_gb=disk_gb)
        if rc == 0:
            return
        n += 1


def train(job, seed, name, lr, aux, target, miles, root=None, extra=()):
    argv = ["orv1_train.py", "--seed", str(seed), "--name", name, "--lr", str(lr), "--aux", str(aux), "--target",
            str(target), "--milestones", ",".join(str(m) for m in miles), *extra]
    if root:
        argv += ["--out-root", str(root)]
    legs(job, argv, 2.0, 0.3)


def evaluate(job, seed, ckpt, part, out, full):
    if Path(out).exists():
        return
    if STOP_MARKER.exists():
        external_stop(job)
    gpu_cap(stage_of(job), 0.15)
    driver(job, ["orv1_eval.py", "--seed", str(seed), "--ckpt", str(ckpt), "--part", part, "--out", str(out)] +
           (["--full"] if full else []), disk_gb=3.0)


# ---------------------------------------------------------------------------------------------------- phases
def gputests():
    import torch
    done = ART / "gpu_tests_resume.json"
    if done.exists() and json.loads(done.read_text())["ok"]:
        return
    root = TMP / "gpu_resume"
    common = dict(seed=0, lr=1e-3, aux=0.01, target=0, miles=[3])
    ex = ("--test-rows", "32", "--test-steps", "3")
    train("gputest_full1", name="full1", root=root, extra=ex, **common)
    train("gputest_full2", name="full2", root=root, extra=ex, **common)
    train("gputest_legs", name="legs", root=root, extra=ex + ("--max-seconds", "0"), **common)
    S = {n: torch.load(root / n / "ckpt_3.pt", map_location="cpu", weights_only=False) for n in ("full1", "full2", "legs")}

    def maxdiff(a, b):
        return max(float((a["conds"][c][k].float() - b["conds"][c][k].float()).abs().max())
                   for c in a["conds"] for k in a["conds"][c])

    def bitwise(a, b):
        return all(torch.equal(a["conds"][c][k], b["conds"][c][k]) for c in a["conds"] for k in a["conds"][c])
    d_legs, d_rep = maxdiff(S["full1"], S["legs"]), maxdiff(S["full1"], S["full2"])
    ok = (bitwise(S["full1"], S["legs"]) or d_legs <= 2 * d_rep + 1e-6) and S["legs"]["step"] == 3 and \
        S["legs"]["tokens"] == S["full1"]["tokens"]
    guard.atomic_write_json(done, {"time": guard.now(), "bitwise_legs": bitwise(S["full1"], S["legs"]),
                                   "bitwise_repeat": bitwise(S["full1"], S["full2"]), "maxdiff_legs": d_legs,
                                   "maxdiff_repeat": d_rep, "ok": ok}, mode=0o400)
    LG.observe("GPU resume test", bitwise_legs=bitwise(S["full1"], S["legs"]), maxdiff_legs=d_legs,
               maxdiff_repeat=d_rep, ok=ok)
    import shutil
    for p in root.rglob("*"):
        if p.is_file():
            p.chmod(0o600)
    shutil.rmtree(root, ignore_errors=True)
    if not ok:
        LG.set_state("FAILED_GPU_TESTS")
        sys.exit(1)


def pilot():
    sel = ART / "pilot_selection.json"
    if sel.exists():
        return json.loads(sel.read_text())
    res = {}
    for lr in LR_CANDIDATES:
        name = f"pilot_lr{lr:g}_aux{AUX_LADDER[0]:g}"
        train(f"pilot_train_lr{lr:g}", 0, name, lr, AUX_LADDER[0], PILOT_TOKENS, [PILOT_TOKENS])
        out = EVALS / f"{name}_SEL.npz"
        evaluate(f"pilot_eval_lr{lr:g}", 0, RUNS / name / f"ckpt_{PILOT_TOKENS}.pt", "SEL", out, False)
        j = json.loads(out.with_suffix(".json").read_text())
        res[lr] = {"ce": {c: j[f"ce_{c}"] for c in "DABC"}, "score": (j["ce_A"] + j["ce_B"] + j["ce_C"]) / 3,
                   "loadC_max": j["loadC_max"], "loadC_eff": j["loadC_eff"]}
    best = min(res, key=lambda lr: (round(res[lr]["score"], 6), lr))
    aux, ladder = AUX_LADDER[0], []
    collapsed = not (res[best]["loadC_max"] <= 0.5 and res[best]["loadC_eff"] >= 4.0)
    ladder.append({"aux": aux, "collapsed": collapsed})
    for aux_ in AUX_LADDER[1:]:
        if not collapsed:
            break
        name = f"pilot_lr{best:g}_aux{aux_:g}"
        train(f"pilot_train_lr{best:g}_aux{aux_:g}", 0, name, best, aux_, PILOT_TOKENS, [PILOT_TOKENS])
        out = EVALS / f"{name}_SEL.npz"
        evaluate(f"pilot_eval_lr{best:g}_aux{aux_:g}", 0, RUNS / name / f"ckpt_{PILOT_TOKENS}.pt", "SEL", out, False)
        j = json.loads(out.with_suffix(".json").read_text())
        aux = aux_
        collapsed = not (j["loadC_max"] <= 0.5 and j["loadC_eff"] >= 4.0)
        ladder.append({"aux": aux_, "collapsed": collapsed, "loadC_max": j["loadC_max"], "loadC_eff": j["loadC_eff"]})
    doc = {"time": guard.now(), "per_lr": {f"{k:g}": v for k, v in res.items()}, "lr": best, "aux": aux,
           "aux_ladder": ladder, "rule": "argmin over LR of mean(CE_A, CE_B, CE_C) on SEL (seed 0, 5M tokens); "
                                         "smallest aux on the ladder without C collapse at that LR"}
    guard.atomic_write_json(sel, doc, mode=0o400)
    LG.decision("pilot complete", "prereg 6.3 condition-agnostic LR selection on SEL", f"lr={best:g}, aux={aux:g}",
                result=json.dumps(doc["per_lr"]), decision_="freeze lr/aux for all stages")
    return doc


def run_stage(stage, target, seeds, pl):
    for s in seeds:
        train(f"{stage}_train_s{s}", s, f"s{s}", pl["lr"], pl["aux"], target,
              [BUDGETS["S1"], BUDGETS["S2"]])
        ck = RUNS / f"s{s}" / f"ckpt_{target}.pt"
        tag = {"stage1": "S1", "stage2": "S2"}[stage]
        evaluate(f"{stage}_evalEVAL_s{s}", s, ck, "EVAL", EVALS / f"{tag}_s{s}_EVAL.npz", True)
        evaluate(f"{stage}_evalSEL_s{s}", s, ck, "SEL", EVALS / f"{tag}_s{s}_SEL.npz", False)
    out = ART / f"stage_{ {'stage1': 'S1', 'stage2': 'S2'}[stage] }.json"
    if not out.exists():
        r = subprocess.run([PY, "orv1_analyze.py", "--stage", {"stage1": "S1", "stage2": "S2"}[stage]], cwd=guard.ROOT)
        if r.returncode:
            LG.set_state("FAILED_ANALYSIS", stage=stage)
            sys.exit(r.returncode)
    R = json.loads(out.read_text())
    LG.decision(f"{stage} analysis complete", "prereg 8 criteria evaluated once",
                f"promote={R['promote']}", result=json.dumps(R["criteria"]),
                decision_=f"outcome {R['outcomes']['primary']}; " + ("continue" if R["promote"] else "STOP"))
    return R


def stage3(pl):
    ex = json.loads((ART / "stage2_extra_rules.json").read_text())
    if not ex["stage2_promote"]:
        raise guard.GuardError("stage 3 requires Stage-2 promotion")
    budget = ex["stage3_budget_rule"]["budget"]
    if not (ART / "val_assign" / "val_assign.json").exists():
        gpu_cap("stage3", 0.1)
        driver("stage3_valassign_val", ["orv1_val_assign.py", "--split", "val"], disk_gb=1.0)
    for s in (2, 3):
        train(f"stage3_train_s{s}", s, f"s{s}", pl["lr"], pl["aux"], budget, [BUDGETS["S1"], BUDGETS["S2"]])
    for s in (0, 1, 2, 3):
        out = EVALS / f"S3_s{s}_VAL.npz"
        if out.exists():
            continue
        if STOP_MARKER.exists():
            external_stop(f"stage3_evalVAL_s{s}")
        gpu_cap("stage3", 0.4)
        driver(f"stage3_evalVAL_s{s}", ["orv1_eval.py", "--seed", str(s), "--ckpt", str(RUNS / f"s{s}" / f"ckpt_{budget}.pt"),
                                        "--part", "VAL", "--out", str(out), "--full"], disk_gb=4.0)
    out = ART / "stage_S3.json"
    if not out.exists():
        r = subprocess.run([PY, "orv1_analyze.py", "--stage", "S3", "--seeds", "0,1,2,3"], cwd=guard.ROOT)
        if r.returncode:
            LG.set_state("FAILED_ANALYSIS", stage="stage3")
            sys.exit(r.returncode)
    R = json.loads(out.read_text())
    LG.decision("stage3 analysis complete (VAL opened once)", "prereg 8.3 final criteria evaluated once",
                f"final success={R['promote']}", result=json.dumps(R["final_criteria"]),
                decision_=f"outcome {R['outcomes']['primary']}; experiment terminal")
    return R


def finalize(last, R, pl):
    tot, by = LG.gpu_hours()
    fa = {"time": guard.now(), "last_stage": last, "promote_last": R["promote"], "primary_outcome":
          R["outcomes"]["primary"], "outcomes": R["outcomes"], "pilot": pl, "gpu_hours": {"total": tot, **by},
          "stages": {}}
    for extra in ("stage2_extra_rules.json", "pilot_selection.json", "gpu_tests_resume.json", "val_assign/val_assign.json",
                  "val_assign/check_calib_search.json"):
        if (ART / extra).exists():
            fa[extra] = json.loads((ART / extra).read_text())
    for tag in ("S1", "S2", "S3"):
        p = ART / f"stage_{tag}.json"
        if p.exists():
            fa["stages"][tag] = json.loads(p.read_text())
    inv = sorted(ART.glob("protected_inventory_before.verify_*.json"))
    if not inv:
        subprocess.run([PY, "orv1_inventory.py", "--verify", str(ART / "protected_inventory_before.json")],
                       cwd=guard.ROOT)
        inv = sorted(ART.glob("protected_inventory_before.verify_*.json"))
    fa["integrity"] = json.loads(inv[-1].read_text()) if inv else None
    if fa["integrity"]:
        fa["integrity"].pop("bad", None) if not fa["integrity"]["bad"] else None
    guard.atomic_write_json(ART / "final_analysis.json", fa, mode=0o400, overwrite=False)
    LG.set_state(f"TERMINAL_{last}_{'PROMOTED' if R['promote'] else 'NOT_PROMOTED'}", outcome=R["outcomes"]["primary"])
    LG.write_resume(f"TERMINAL ({last}; promote={R['promote']}; outcome {R['outcomes']['primary']})",
                    "true  # nothing to resume", "See final_analysis.json.")


def main():
    guard.check_root()
    check_prereg()
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    LG.set_state(f"RUNNING_{what}")
    LG.write_resume(f"RUNNING_{what}", f"{PY} orv1_run.py all", "Completed jobs are skipped automatically.")
    if what in ("gputests", "all"):
        gputests()
    pl = None
    if what in ("pilot", "all"):
        pl = pilot()
    if what in ("stage1", "all"):
        pl = pl or json.loads((ART / "pilot_selection.json").read_text())
        R1 = run_stage("stage1", BUDGETS["S1"], STAGE1_SEEDS, pl)
        if not R1["promote"]:
            finalize("S1", R1, pl)
            return
        R2 = run_stage("stage2", BUDGETS["S2"], STAGE1_SEEDS, pl)
        if not R2["promote"]:
            finalize("S2", R2, pl)
            return
        ex = json.loads((ART / "stage2_extra_rules.json").read_text()) if (ART / "stage2_extra_rules.json").exists() \
            else None
        if ex is None:
            LG.set_state("STAGE2_PROMOTED_STAGE3_PENDING")
            LG.write_resume("STAGE2_PROMOTED_STAGE3_PENDING", f"{PY} orv1_run.py all",
                            "Stage-2 extra rules (prereg 8.2) and the Stage-3 budget rule must be evaluated first.")
            return
        if not ex["stage2_promote"]:
            finalize("S2", R2, pl)
            return
        R3 = stage3(pl)
        finalize("S3", R3, pl)


if __name__ == "__main__":
    main()
