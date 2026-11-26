#!/usr/bin/env python
"""spb1 orchestrator (adapted from scur1_run.py). Every job goes through spb1_driver.py (GPU jobs through
scripts/gpu_guarded.sh, 90-minute cap per job; long training runs in resumable legs).

  spb1_run.py stage1 --gen G     bank training (SEM-M / SHUF-M / GEN-KM jointly, 3 bank seeds, 50M tokens; skipped for
                                 router-only generations), SEL extraction (+ COARSE-16), Stage-1 analysis + gate
  spb1_run.py stage2 --gen G     TRAIN extraction, fresh unrestricted routers (2 router seeds per bank; posterior
                                 ablation router seed 0), Stage-2 analysis + promotion gate
  spb1_run.py change --gen G     record generation G's single-factor change (env SPB1_CHANGE = JSON with keys change,
                                 diagnosis, prediction, falsifier, alternatives) BEFORE it runs
  spb1_run.py final              final_analysis.json + protected inventory verification
External interruption (driver exit 77): INCOMPLETE_STOPPED_BY_EXTERNAL_USER + exact resume command; never restarts."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402
from csr import spb1_log as LG  # noqa: E402
from csr import spb1_paths as P  # noqa: E402

PY = str(guard.ROOT / ".venv/bin/python")
CMD_NOW = ""


def rel(p):
    return str(Path(p).relative_to(guard.ROOT))


def stop_external(job_id):
    LG.set_state("INCOMPLETE_STOPPED_BY_EXTERNAL_USER", job=job_id)
    LG.write_resume("INCOMPLETE_STOPPED_BY_EXTERNAL_USER",
                    f"{PY} spb1_driver.py authorize-resume && nohup {PY} spb1_run.py {CMD_NOW} > "
                    f"{rel(P.ART)}/run_{'_'.join(CMD_NOW.split()).replace('--', '')}.log 2>&1 &",
                    notes=f"Interrupted job: `{job_id}`. Treated as another user needing the GPU (preemption): the job "
                          f"checkpointed its rolling slot if it could, all spb1 launchers/watchers stopped, the GPU is "
                          f"released, nothing restarts automatically. Wait for explicit user permission. Rerunning the "
                          f"phase skips completed jobs and resumes training from the slot.")
    print(f"EXTERNAL STOP at {job_id}; see RESUME.md", flush=True)
    sys.exit(77)


def group_of(stage):
    for g in ("stage0", "stage1", "stage2", "stage3"):
        if stage.startswith(g):
            return g
    return "eng"


def job(job_id, stage, argv, gpu=True, expect="0", disk_gb=1.0, seed=""):
    tot, by = LG.gpu_hours()
    grp = group_of(stage)
    used = sum(v for k, v in by.items() if group_of(k) == grp)
    if gpu and (used >= P.GPU_CAP_H[grp] or tot >= P.GPU_CAP_H["total"]):
        LG.set_state("STOPPED_GPU_CAP", gpu_h=used, group=grp)
        LG.write_resume("STOPPED_GPU_CAP", f"{PY} spb1_run.py {CMD_NOW}", notes=f"{grp} GPU cap reached ({used:.2f} h)")
        raise SystemExit(f"GPU cap reached ({used:.2f} h)")
    free = guard.disk_free_bytes() / 2**30
    if free - disk_gb < P.DISK_RESERVE_GIB:
        LG.set_state("STOPPED_DISK", free_gib=free, need_gib=disk_gb)
        LG.write_resume("STOPPED_DISK", f"{PY} spb1_run.py {CMD_NOW}",
                        notes=f"Disk shortfall: free {free:.2f} GiB, need 100 + {disk_gb:.2f} GiB "
                              f"(shortfall {100 + disk_gb - free:.2f} GiB).")
        raise SystemExit(f"disk shortfall {100 + disk_gb - free:.2f} GiB")
    LG.disk(job_id, disk_gb)
    st = json.loads((P.ART / "driver_state.json").read_text()) if (P.ART / "driver_state.json").exists() else {"jobs": {}}
    if st["jobs"].get(job_id, {}).get("status") == "COMPLETE":
        return 0
    LG.write_resume(f"RUNNING {job_id}", f"{PY} spb1_run.py {CMD_NOW}",
                    notes="Running normally. If this process is gone without a final state, rerun the command above "
                          "(completed jobs are skipped; training resumes from its rolling slot).")
    t0, t_iso = time.time(), guard.now()
    cmd = [PY, "spb1_driver.py", "job", job_id, "--gpu" if gpu else "--cpu", "--expect-rc", expect, "--disk-gb",
           str(disk_gb), "--"] + argv
    rc = subprocess.run(cmd, cwd=guard.ROOT, env=os.environ.copy()).returncode
    dt = time.time() - t0
    st = json.loads((P.ART / "driver_state.json").read_text())
    j = st["jobs"].get(job_id, {})
    if gpu:
        LG.compute(job_id, stage, t_iso, guard.now(), dt, j.get("status", "?"))
    LG.manifest({"time": guard.now(), "job": job_id, "stage": stage, "seed": seed, "status": j.get("status"),
                 "rc": j.get("rc"), "seconds": round(dt, 1), "gpu_h": round(dt / 3600, 4) if gpu else 0,
                 "cmd": " ".join(argv)})
    LG.run_log(job=job_id, stage=stage, seed=seed, cmd=argv, gpu=gpu, start=t_iso, end=guard.now(), rc=rc,
               status=j.get("status"), driver_rc=j.get("rc"))
    if rc == 77:
        stop_external(job_id)
    if rc != 0:
        LG.set_state("FAILED", job=job_id, rc=rc)
        LG.write_resume("FAILED", f"{PY} spb1_run.py {CMD_NOW}", notes=f"Job {job_id} failed (rc {rc}); see job_logs.")
        raise SystemExit(f"job {job_id} failed rc={rc}")
    return j.get("rc")


def train_bank(gen, seed, budget_key):
    bg = P.bank_gen(gen)
    rd = P.RUNS / f"g{bg}_s{seed}"
    tgt = P.BUDGETS[budget_key]
    ck = rd / f"ckpt_{tgt}.pt"
    leg = 0
    while not ck.exists():
        rc = job(f"train_g{bg}_s{seed}_{budget_key}_leg{leg}", f"stage1_train" if budget_key == "S1" else "stage3_train",
                 ["spb1_train.py", "--seed", str(seed), "--gen", str(bg), "--target", str(tgt), "--milestones",
                  ",".join(str(P.BUDGETS[k]) for k in P.BUDGETS if P.BUDGETS[k] <= tgt), "--max-seconds", "4800"],
                 expect="0,75", disk_gb=1.0, seed=seed)
        leg += 1
        if rc == 0 and not ck.exists():
            raise SystemExit(f"training finished without {ck}")
        if leg > 12:
            raise SystemExit("too many training legs")
    return ck


def extract(gen, seed, part, budget_key="S1"):
    bg = P.bank_gen(gen)
    ck = P.RUNS / f"g{bg}_s{seed}" / f"ckpt_{P.BUDGETS[budget_key]}.pt"
    od = P.EXTRACT / f"g{bg}_s{seed}_{budget_key}"
    if (od / f"summary_{part}.json").exists():
        return od
    gb = {"TRAIN": 2.0, "SEL": 2.6, "VAL": 3.0}[part]
    job(f"extract_g{bg}_s{seed}_{budget_key}_{part}", "stage1_extract" if part == "SEL" else "stage2_extract",
        ["spb1_extract.py", "--seed", str(seed), "--ckpt", rel(ck), "--part", part, "--out-dir", rel(od), "--coarse"],
        disk_gb=gb, seed=seed)
    return od


def analyze(cmd, gen, seeds, stage_name=""):
    args = ["spb1_analyze.py", cmd, "--gen", str(gen), "--seeds", ",".join(map(str, seeds))]
    if stage_name:
        args += ["--stage-name", stage_name]
    job(f"analyze_{cmd}_g{gen}", f"{cmd}_analysis", args, gpu=False, disk_gb=0.05)
    return json.loads((P.ANALYSIS / f"{cmd}_g{gen}{('_' + stage_name) if stage_name else ''}.json").read_text())


def router_cfgs(gen):
    G = P.gen_cfg(gen)
    M = int(G["M"])
    s, h, g = P.bank_names(M)
    base = {"H": int(G["router_h"]), "temp": float(G["router_temp"]), "bal": float(G["router_bal"])}
    out = []
    for rs in P.ROUTER_SEEDS:
        for tag, bank in (("R_SEM", s), ("R_SHUF", h), ("R_GEN", g), ("R_COARSE16", "COARSE16")):
            out.append({"tag": tag, "bank": bank, "router_seed": rs, **base})
    for tag, bank in (("R_SEM_POST", s), ("R_SHUF_POST", h), ("R_GEN_POST", g)):
        out.append({"tag": tag, "bank": bank, "router_seed": 0, "post": True, **base})
    return out


def write_cfg(name, obj):
    P.CFG_DIR.mkdir(parents=True, exist_ok=True)
    p = P.CFG_DIR / f"{name}.json"
    if p.exists():
        if json.loads(p.read_text()) != obj:
            raise guard.GuardError(f"config {p} exists with different content")
        return p
    guard.atomic_write_json(p, obj, mode=0o400)
    return p


def stage1(gen):
    P.check_prereg()
    if gen > 0 and not (P.ART / f"gen{gen}_change.json").exists():
        raise guard.GuardError(f"generation {gen} change not recorded")
    LG.set_state(f"RUNNING_STAGE1_G{gen}")
    for s in P.BANK_SEEDS:
        train_bank(gen, s, "S1")
        extract(gen, s, "SEL")
    A = analyze("stage1", gen, P.BANK_SEEDS)
    g = A["gate"]
    LG.decision(evidence={"gate": g, "dI_intrinsic": {s: A["per_seed"][s]["dI_intrinsic"] for s in A["per_seed"]},
                          "sem_oracle_minus_gen": {s: A["per_seed"][s]["sem_oracle_minus_gen"] for s in A["per_seed"]}},
                competing="semantic specialisation vs persistent-grouping (SHUF) vs generic capacity (GEN)",
                decision_="ADVANCE_TO_STAGE2" if g["advance"] else "STAGE1_FAIL",
                rationale="prereg section 7.1 gate (G1a functional >= 2/3 seeds, G1b SEM oracle not > GEN + 0.001)",
                falsifier="n/a (preregistered gate)", gen=gen)
    LG.set_state(f"STAGE1_G{gen}_{'PASS' if g['advance'] else 'FAIL'}", gate=g)
    return A


def stage2(gen):
    P.check_prereg()
    A1 = json.loads((P.ANALYSIS / f"stage1_g{P.bank_gen(gen)}.json").read_text())
    if not A1["gate"]["advance"]:
        raise guard.GuardError("Stage 2 requires a passed Stage-1 gate for this bank generation")
    LG.set_state(f"RUNNING_STAGE2_G{gen}")
    cfgp = write_cfg(f"routers_g{gen}", router_cfgs(gen))
    bg = P.bank_gen(gen)
    for s in P.BANK_SEEDS:
        od = extract(gen, s, "TRAIN")
        job(f"routers_g{gen}_s{s}", "stage2_routers",
            ["spb1_router_job.py", "--seed", str(s), "--tables", rel(od), "--common",
             rel(P.EXTRACT / f"common_s{s}"), "--configs", rel(cfgp), "--stage", f"S2_g{gen}"], disk_gb=0.5, seed=s)
    A = analyze("stage2", gen, P.BANK_SEEDS, stage_name=f"S2_g{gen}")
    g = A["gate"]
    LG.decision(evidence={"gate": g, "SEM_minus_GEN": A["comparisons"].get("SEM_minus_GEN", {}).get("all"),
                          "dI_deployed": A.get("dI_deployed_pooled")},
                competing="unrestricted router exploits semantic bank vs generic capacity equivalence",
                decision_="PROMOTE_TO_STAGE3" if g["promote"] else "STAGE2_FAIL",
                rationale="prereg section 7.2 gate P1-P4", falsifier="n/a (preregistered gate)", gen=gen, bank_gen=bg)
    LG.set_state(f"STAGE2_G{gen}_{'PROMOTE' if g['promote'] else 'FAIL'}", gate=g)
    return A


def change(gen):
    ch = json.loads(os.environ["SPB1_CHANGE"])
    if gen < 1 or gen > P.MAX_GENERATIONS:
        raise guard.GuardError("generation out of range")
    if (P.ART / f"gen{gen}_change.json").exists():
        raise guard.GuardError("generation change already recorded")
    allowed = {"M", "router_h", "router_temp", "sub_lr_mult", "aux", "router_bal", "assign"}
    if len(ch["change"]) != 1 or not set(ch["change"]) <= allowed:
        raise guard.GuardError("exactly one permitted factor per generation")
    for k in ("diagnosis", "prediction", "falsifier", "alternatives"):
        if not ch.get(k):
            raise guard.GuardError(f"missing {k}")
    guard.atomic_write_json(P.ART / f"gen{gen}_change.json", {"time": guard.now(), "gen": gen, **ch}, mode=0o400)
    LG.decision(evidence=ch["diagnosis"], competing=ch["alternatives"], decision_=f"GENERATION_{gen}: {ch['change']}",
                rationale=ch.get("rationale", ""), falsifier=ch["falsifier"], prediction=ch["prediction"], gen=gen)
    print(json.dumps({"recorded": gen, "bank_gen": P.bank_gen(gen)}))


def final(state):
    """final_analysis.json (all stage analyses, gates, generations, compute, disk) + protected inventory verification."""
    tot, by = LG.gpu_hours()
    an = {}
    for f in sorted(P.ANALYSIS.glob("*.json")):
        an[f.stem] = json.loads(f.read_text())
    gens = {}
    for g in range(1, P.MAX_GENERATIONS + 1):
        f = P.ART / f"gen{g}_change.json"
        if f.exists():
            gens[g] = json.loads(f.read_text())
    subprocess.run([PY, "spb1_inventory.py", "verify"], cwd=guard.ROOT, check=True)
    ver = sorted(P.ART.glob("protected_inventory_verify_*.json"))[-1]
    out = {"time": guard.now(), "state": state, "prereg_sha256": P.check_prereg(), "gpu_hours": {"total": tot, "by_stage": by},
           "disk_free_gib_end": guard.disk_free_bytes() / 2**30, "generations": gens,
           "inventory_verify": json.loads(ver.read_text()), "analyses": {k: {"gate": v.get("gate")} for k, v in an.items()},
           "analysis_files": {k: guard.sha256_file(P.ANALYSIS / f"{k}.json") for k in an}}
    guard.atomic_write_json(P.ART / "final_analysis.json", out, overwrite=True)
    LG.set_state(state)
    LG.write_resume(state, "(nothing to resume)", notes="Terminal. See docs/SEMANTIC_PAIR_BANK_UNRESTRICTED_ROUTER_V1_REPORT.md.")
    print(json.dumps({"state": state, "gpu_h": tot, "inventory_all_match": out["inventory_verify"]["all_match"]}))


def main():
    global CMD_NOW
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stage1", "stage2", "change", "final"])
    ap.add_argument("--gen", type=int, default=0)
    ap.add_argument("--state", default="")
    a = ap.parse_args()
    guard.check_root()
    CMD_NOW = " ".join(sys.argv[1:])
    if a.cmd == "stage1":
        stage1(a.gen)
    elif a.cmd == "stage2":
        stage2(a.gen)
    elif a.cmd == "change":
        change(a.gen)
    elif a.cmd == "final":
        final(a.state)
    print(json.dumps({"done": a.cmd, "gen": a.gen, "state": json.loads((P.ART / "STATE.json").read_text())["state"]
                      if (P.ART / "STATE.json").exists() else None}))


if __name__ == "__main__":
    main()
