#!/usr/bin/env python
"""recr1 orchestrator. Every step is logged (RUN_LOG, run_manifest, compute_ledger, DECISION_LOG); an external
interruption (signal, killed child) sets STOPPED_EXTERNAL, writes RESUME.md and never restarts anything.

  recr1_run.py freeze                 hash prereg + code manifest (once)
  recr1_run.py gen --gen G            heads (if needed) + CSEARCH evaluation of generation G
  recr1_run.py change --gen G         record generation G's single-factor change (env RECR1_CHANGE JSON)
  recr1_run.py timing                 GPU latency micro-benchmark (seed-3 architecture; timing only)
  recr1_run.py select                 freeze the candidate (FROZEN.json) and evaluate CSELECT once
  recr1_run.py final --state STATE    final_analysis.json, inventory verification, RESUME.md"""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402
from csr import recr1_log as LG  # noqa: E402
from csr import recr1_paths as P  # noqa: E402

PY = str(guard.ROOT / ".venv/bin/python")
CODE = ["recr1_eval.py", "recr1_run.py", "recr1_tests.py", "recr1_manifest.py", "recr1_timing.py",
        "src/csr/recr1_lib.py", "src/csr/recr1_paths.py", "src/csr/recr1_log.py", "spb1_analyze.py"]


def disk_ok(gb):
    free = guard.disk_free_bytes() / 2**30
    LG.disk(f"pre-job check (estimate {gb} GiB)", gb)
    if free - gb < P.DISK_RESERVE_GIB:
        raise guard.GuardError(f"disk reserve: free {free:.1f} GiB - {gb} < {P.DISK_RESERVE_GIB}")


def stopped(job, rc):
    LG.set_state("STOPPED_EXTERNAL", job=job, rc=rc)
    LG.write_resume("STOPPED_EXTERNAL", f".venv/bin/python recr1_run.py {' '.join(sys.argv[1:])}",
                    notes=f"Job {job} ended with rc={rc} (signal or external kill). Treated as preemption: not "
                          "restarted. Completed heads/analyses are skipped on resume (done markers + sha256).")
    LG.observe(f"STOPPED_EXTERNAL: job {job} rc={rc}; nothing restarted")
    sys.exit(77)


def job(name, stage, argv, gpu=False, disk_gb=0.2):
    if P.STOP_MARKER.exists():
        stopped(name, "STOP marker")
    disk_ok(disk_gb)
    start, t0 = guard.now(), time.time()
    cmd = (["scripts/gpu_guarded.sh", f"recr1 {name}"] + argv + ["--execute"]) if gpu else [PY] + argv
    env = dict(os.environ)
    if gpu:
        env.update({"MOE_GPU_LAUNCH": "1", "MOE_RESEARCH_APPROVE": "gpu"})
    LG.run_log(kind="start", job=name, stage=stage, cmd=cmd)
    logp = P.ART / "job_logs" / f"{name}.log"
    logp.parent.mkdir(parents=True, exist_ok=True)
    with open(logp, "a") as lf:
        pr = subprocess.Popen(cmd, cwd=guard.ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT)
        try:
            rc = pr.wait()
        except KeyboardInterrupt:
            pr.send_signal(signal.SIGINT)
            rc = pr.wait()
    sec = time.time() - t0
    status = "COMPLETE" if rc == 0 else f"FAILED_rc{rc}"
    LG.compute(name, stage, start, guard.now(), sec, status if gpu else status + "_CPU")
    LG.manifest({"job": name, "stage": stage, "start": start, "end": guard.now(), "rc": rc, "gpu": gpu,
                 "seconds": round(sec, 1), "log": str(logp.relative_to(guard.ROOT))})
    LG.run_log(kind="end", job=name, rc=rc, seconds=sec)
    if rc in (-2, -15, -1, -9, 130, 143, 77) or P.STOP_MARKER.exists():
        stopped(name, rc)
    if rc != 0:
        LG.set_state("FAILED", job=name, rc=rc)
        raise SystemExit(f"job {name} failed rc={rc}; see {logp}")
    return rc


def freeze():
    if P.PREREG_SHA_FILE.exists():
        raise guard.GuardError("already frozen")
    h = P.prereg_sha()
    guard.atomic_write_bytes(P.PREREG_SHA_FILE, f"{h}  {P.PREREG.name}\n".encode(), mode=0o444)
    man = {"time": guard.now(), "prereg_sha256": h, "code": {c: guard.sha256_file(guard.ROOT / c) for c in CODE
                                                             if (guard.ROOT / c).exists()},
           "data_manifest_sha256": guard.sha256_file(P.ART / "data_manifest.json")}
    guard.atomic_write_json(P.ART / "code_manifest_prereg.json", man, mode=0o400)
    LG.set_state("PREREG_FROZEN")
    LG.run_log(kind="freeze", prereg_sha256=h)
    print(h)


def gen(g):
    P.check_prereg()
    if g > 0 and not (P.ART / f"gen{g}_change.json").exists():
        raise guard.GuardError("record the change first")
    LG.set_state(f"RUNNING_GEN{g}")
    job(f"heads_g{g}", "stage1_heads", ["recr1_eval.py", "heads", "--gen", str(g)], disk_gb=0.1)
    job(f"eval_g{g}_CSEARCH", "stage1_csearch", ["recr1_eval.py", "eval", "--gen", str(g), "--part", "CSEARCH"])
    res = json.loads((P.ANALYSIS / f"gen{g}_CSEARCH.json").read_text())
    LG.set_state(f"GEN{g}_CSEARCH_DONE", gates={k: res["gates"][k].get("PASS") if isinstance(res["gates"][k], dict)
                                                 else res["gates"][k] for k in ("E", "Ecm")})
    print(json.dumps(res["gates"])[:1500])


def change(g):
    ch = json.loads(os.environ["RECR1_CHANGE"])
    if not 1 <= g <= P.MAX_GENERATIONS or (P.ART / f"gen{g}_change.json").exists():
        raise guard.GuardError("bad or duplicate generation")
    if len(ch["change"]) != 1 or not set(ch["change"]) <= P.PERMITTED:
        raise guard.GuardError("exactly one permitted factor")
    for k in ("failure", "evidence", "prediction", "falsifier", "alternatives"):
        if not ch.get(k):
            raise guard.GuardError(f"missing {k}")
    guard.atomic_write_json(P.ART / f"gen{g}_change.json", {"time": guard.now(), "gen": g, **ch}, mode=0o400)
    LG.decision(evidence=ch["evidence"], competing=ch["alternatives"], decision_=f"GENERATION_{g}: {ch['change']}",
                rationale=ch["failure"], falsifier=ch["falsifier"], prediction=ch["prediction"], gen=g)
    print("recorded", g)


def select():
    P.check_prereg()
    fz = P.ART / "FROZEN.json"
    if not fz.exists():
        best = None
        for g in range(P.MAX_GENERATIONS + 1):
            f = P.ANALYSIS / f"gen{g}_CSEARCH.json"
            if not f.exists():
                continue
            r = json.loads(f.read_text())
            ge = r["gates"]["E"]
            n = sum(bool(ge[k]) for k in ("P1a", "P1b", "P2_first32_vs_G", "P2_trans8_vs_S", "P2_trans32_vs_S",
                                          "P3_stable", "P4_seeds", "P5_functional"))
            key = (-n, r["comparisons"]["E_minus_S"]["all"]["mean"])
            if best is None or key < best[0]:
                best = (key, g, r)
        g0 = json.loads((P.ANALYSIS / "gen0_CSEARCH.json").read_text())
        frozen = {"time": guard.now(), "gen": best[1], "cfg": best[2]["cfg"], "theta_C": g0["theta_C"],
                  "rule": "most CSEARCH gates passed by E, tie -> lowest E-S overall; theta_C frozen at gen 0",
                  "code": {c: guard.sha256_file(guard.ROOT / c) for c in CODE if (guard.ROOT / c).exists()}}
        guard.atomic_write_json(fz, frozen, mode=0o400)
        LG.decision(evidence={"gen": best[1], "gates_passed_csearch": -best[0][0]}, competing="all run generations",
                    decision_=f"FREEZE gen {best[1]} for one-time CSELECT", rationale=frozen["rule"],
                    falsifier="n/a")
    job("eval_CSELECT", "stage1_cselect", ["recr1_eval.py", "eval", "--part", "CSELECT", "--frozen"])
    res = json.loads((P.ANALYSIS / "CSELECT.json").read_text())
    st = ("STAGE1_PASS_CONFIRMATION_BLOCKED" if res["gates"]["E"]["PASS"] or res["gates"]["compute_matched_PASS"]
          else "TERMINAL_STAGE1_FAIL")
    LG.set_state(st, gates=res["gates"]["E"])
    print(st, json.dumps(res["gates"])[:1500])


def timing():
    job("timing", "stage0_timing", ["recr1_timing.py"], gpu=True, disk_gb=0.05)


def final(state):
    tot, by = LG.gpu_hours()
    an = {f.stem: json.loads(f.read_text()) for f in sorted(P.ANALYSIS.glob("*.json"))}
    subprocess.run([PY, "recr1_manifest.py", "verify"], cwd=guard.ROOT, check=True)
    ver = sorted(P.ART.glob("protected_inventory_verify_*.json"))[-1]
    out = {"time": guard.now(), "state": state, "prereg_sha256": P.check_prereg(), "gpu_hours": {"total": tot, "by": by},
           "disk_free_gib_end": guard.disk_free_bytes() / 2**30,
           "generations": {g: json.loads((P.ART / f"gen{g}_change.json").read_text())
                           for g in range(1, P.MAX_GENERATIONS + 1) if (P.ART / f"gen{g}_change.json").exists()},
           "gates": {k: v.get("gates") for k, v in an.items()},
           "analysis_sha256": {k: guard.sha256_file(P.ANALYSIS / f"{k}.json") for k in an},
           "inventory_verify": json.loads(ver.read_text())}
    timing_f = P.ART / "timing.json"
    if timing_f.exists():
        out["timing"] = json.loads(timing_f.read_text())
    guard.atomic_write_json(P.ART / "final_analysis.json", out, overwrite=True)
    LG.set_state(state)
    LG.write_resume(state, "(nothing to resume)", notes=f"Terminal. See {P.REPORT.relative_to(guard.ROOT)}.")
    print(json.dumps({"state": state, "inventory_all_match": out["inventory_verify"]["all_match"]}))


def main():
    guard.check_root()
    cmd = sys.argv[1]
    arg = lambda k: sys.argv[sys.argv.index(k) + 1]
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        if cmd == "freeze":
            freeze()
        elif cmd == "gen":
            gen(int(arg("--gen")))
        elif cmd == "change":
            change(int(arg("--gen")))
        elif cmd == "timing":
            timing()
        elif cmd == "select":
            select()
        elif cmd == "final":
            final(arg("--state"))
    except KeyboardInterrupt:
        stopped(cmd, "signal")


if __name__ == "__main__":
    main()
