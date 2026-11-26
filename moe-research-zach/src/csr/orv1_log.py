"""oracle-residual-v1 append-only logs (copy of scrgp_log.py): decision_log.jsonl, observations.jsonl, deviations.jsonl, run_manifest.tsv,
compute_ledger.tsv, disk_ledger.tsv; STATE.json and RESUME.md (atomic rewrite)."""
import json

from csr import guard
from csr.orv1_paths import ART, GPU_CAP_H


def _append(name, line):
    ART.mkdir(parents=True, exist_ok=True)
    with open(ART / name, "a") as f:
        f.write(line)


def decision(observation, explanation, action, expected="", result="", decision_="", **kw):
    _append("decision_log.jsonl", json.dumps({"time": guard.now(), "observation": observation,
                                              "explanation": explanation, "action": action, "expected": expected,
                                              "result": result, "decision": decision_, **kw}, default=str) + "\n")


def observe(text, **kw):
    _append("observations.jsonl", json.dumps({"time": guard.now(), "text": text, **kw}, default=str) + "\n")


def deviation(text, effect, can_affect_conclusion, **kw):
    n = sum(1 for _ in open(ART / "deviations.jsonl")) if (ART / "deviations.jsonl").exists() else 0
    _append("deviations.jsonl", json.dumps({"n": n, "time": guard.now(), "text": text, "effect": effect,
                                            "can_affect_conclusion": can_affect_conclusion, **kw}, default=str) + "\n")


def manifest(row):
    new = not (ART / "run_manifest.tsv").exists()
    if new:
        _append("run_manifest.tsv", "time\tjob\tstage\tseed\tstatus\trc\tseconds\tgpu_h\toutputs\tsha256\tcmd\n")
    _append("run_manifest.tsv", "\t".join(str(row.get(k, "")) for k in
                                          ("time", "job", "stage", "seed", "status", "rc", "seconds", "gpu_h",
                                           "outputs", "sha256", "cmd")) + "\n")


def compute(job, stage, start, end, seconds, status):
    new = not (ART / "compute_ledger.tsv").exists()
    if new:
        _append("compute_ledger.tsv", "job\tstage\tstart\tend\tseconds\tgpu_h\tstatus\n")
    _append("compute_ledger.tsv", f"{job}\t{stage}\t{start}\t{end}\t{seconds:.1f}\t{seconds / 3600:.4f}\t{status}\n")


def disk(what, planned_gib):
    new = not (ART / "disk_ledger.tsv").exists()
    if new:
        _append("disk_ledger.tsv", "time\twhat\tfree_gib\tplanned_gib\treserve_gib\n")
    _append("disk_ledger.tsv", f"{guard.now()}\t{what}\t{guard.disk_free_bytes() / 2**30:.2f}\t{planned_gib:.2f}\t100\n")


def gpu_hours():
    p = ART / "compute_ledger.tsv"
    if not p.exists():
        return 0.0, {}
    tot, by = 0.0, {}
    for line in p.read_text().splitlines()[1:]:
        f = line.split("\t")
        h = float(f[5])
        tot += h
        by[f[1]] = by.get(f[1], 0.0) + h
    return tot, by


def set_state(state, **extra):
    guard.atomic_write_json(ART / "STATE.json", {"state": state, "time": guard.now(), **extra}, overwrite=True)


def write_resume(state, cmd, notes=""):
    tot, _ = gpu_hours()
    paused = "STOPPED" in state or "PAUSED" in state or "FAILED" in state
    txt = (f"# RESUME — oracle-residual-v1\n\nUpdated {guard.now()}.\n\n**State:** {state}\n\nGPU used: {tot:.2f} GPU-h (cap {GPU_CAP_H['total']:.0f}). "
           f"Disk free: {guard.disk_free_bytes() / 2**30:.1f} GiB (reserve 100 GiB).\n\n**Resume command"
           f"{' (only after explicit user permission; first run orv1_driver.py authorize-resume)' if paused else ''}:**"
           f"\n\n```\ncd {guard.ROOT} && source env.sh && {cmd}\n```\n\n{notes}\n")
    guard.atomic_write_bytes(ART / "RESUME.md", txt.encode(), overwrite=True)
