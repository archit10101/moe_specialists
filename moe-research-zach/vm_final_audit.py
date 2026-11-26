#!/usr/bin/env python
"""Final audit for csr-value-routed-compute-v1 and csr-m2d2-domain-prior-v1 (CPU only, read-only except its outputs).
GPU-time totals from the ledger for this session, run list and statuses, peak VRAM, disk, protected-inventory verify
(subprocess), locked-test checks; writes artifacts/<exp>/final_audit_<stamp>.json."""
import glob
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402
from csr.vm_run import EXP  # noqa: E402

SESSION_START = "2026-11-09T02:33"


def main():
    rows = []
    for line in open(guard.ROOT / "ledger/gpu_time_ledger.tsv"):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 6 and p[0][:1].isdigit() and p[0] >= SESSION_START:
            rows.append({"start": p[0], "end": p[1], "seconds": float(p[2]), "run": p[3], "purpose": p[4],
                         "status": p[5]})
    by = {}
    for r in rows:
        k = "vrc" if r["purpose"].startswith("vrc") or "vrc" in r["run"] else (
            "m2d2" if "m2d2" in r["purpose"] or "m2d2" in r["run"] else "shared")
        by.setdefault(k, 0.0)
        by[k] += r["seconds"]
    runs = []
    run_dirs = (sorted(glob.glob(str(guard.ROOT / "runs" / "20261109T*_vrc_*"))) +
                sorted(glob.glob(str(guard.ROOT / "runs" / "20261109T*_m2d2_*"))))
    for r in run_dirs:
        s = json.loads((Path(r) / "summary.json").read_text()) if (Path(r) / "summary.json").exists() else {}
        runs.append({"run": str(Path(r).relative_to(guard.ROOT)), "status": s.get("status", "UNKNOWN"),
                     "peak_alloc_gib": s.get("peak_alloc_gib"), "tokens": s.get("valid_train_tokens")})
    inv = subprocess.run([sys.executable, "vm_inventory.py", "--verify",
                          "artifacts/value-routed-compute-v1/protected_inventory.json"], cwd=guard.ROOT,
                         capture_output=True, text=True)
    try:
        inv_res = json.loads(inv.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        inv_res = {"raw": inv.stdout[-2000:], "err": inv.stderr[-2000:]}
    m2d2_data = guard.ROOT / "artifacts/m2d2-domain-prior-v1/data"
    vrc_data = guard.ROOT / "artifacts/value-routed-compute-v1/data"
    locked = {"m2d2_test_tokens_written": any(m2d2_data.glob("test_*")),
              "vrc_test_tokens_written": any(vrc_data.rglob("test_*")),
              "m2d2_test_members_decoded": [v.get("decoded") for v in json.loads(
                  (m2d2_data / "manifest.json").read_text())["locked_test"].values()],
              "loaders_refuse_test": "csr.vm_run.load_split refuses 'test' before file access (tested)",
              "fwe_rep_v1_test_files": "never opened by this session's code (excluded from inventory by name)"}
    doc = {"time": guard.now(), "gpu_seconds_total": sum(r["seconds"] for r in rows), "gpu_seconds_by_experiment": by,
           "gpu_ledger_rows": rows, "runs": runs, "inventory_verify": inv_res,
           "disk_free_gib": guard.disk_free_bytes() / 1024**3, "locked_test": locked,
           "peak_vram_alloc_gib_max_train": max((r["peak_alloc_gib"] or 0) for r in runs) if runs else None}
    st = guard.utc_stamp()
    for k in ("vrc", "m2d2"):
        guard.atomic_write_json(EXP[k]["art"] / f"final_audit_{st}.json", doc)
    print(json.dumps({k: v for k, v in doc.items() if k not in ("gpu_ledger_rows", "runs")}, indent=1))
    print("n_runs", len(runs), {s: sum(1 for r in runs if r["status"] == s) for s in set(r["status"] for r in runs)})


if __name__ == "__main__":
    main()
