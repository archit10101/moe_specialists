#!/usr/bin/env python
"""csr-semantic-edl-long-adaptive-k-v1 integrity checks (CPU, read-only on everything outside this experiment).
Writes artifacts/semantic-edl-long-k/integrity_<UTC>.json.
  1. preregistration hash; block marker; VAL / CALIB-SELECTION opened at most once (0400 markers)
  2. protected inventory: re-hash every file of artifacts/semantic-edl-v1/protected_inventory_before.json (all project
     files that existed before semantic-edl-v1; includes fle/fcu/gc and the reused baseline runs) — the verification
     result is written HERE, not into semantic-edl-v1
  3. semantic-edl-v1 own files and reused baseline run dirs: none modified after this experiment's preregistration
  4. locked test split: no slk code references a test-split file; access times of artifacts/data test files
  5. Phase-1 runs: COMPLETE, identical plan hash / tokens / steps; C6 vs SHUF configs differ only in the table; params
     equal (gpu_checks)"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402
from csr.slk_paths import ART, check_prereg  # noqa: E402
import ms_inventory as msi  # noqa: E402

ROOT = guard.ROOT
PREREG = ROOT / "docs/SEMANTIC_EDL_LONG_ADAPTIVE_K_PREREG.md"


def flat(d, p=""):
    o = {}
    for k, v in d.items():
        o.update(flat(v, p + k + ".") if isinstance(v, dict) else {p + k: v})
    return o


def main():
    guard.check_root()
    out = {"time": guard.now()}
    check_prereg()
    out["prereg_sha256"] = guard.sha256_file(PREREG)
    out["prereg_ok"] = True
    t0 = PREREG.stat().st_mtime
    out["experiment_start_reference"] = "mtime of the preregistration file"
    for m in ("PHASE1_VAL_OPEN.json", "CALIBSEL_OPEN.json", "PHASE2B_BLOCKED_BY_USER.json"):
        p = ART / m
        out[m] = {"exists": p.exists(), "mode": oct(p.stat().st_mode & 0o777) if p.exists() else None}
    # 2. protected inventory
    inv = json.loads((ROOT / "artifacts/semantic-edl-v1/protected_inventory_before.json").read_text())["files"]
    bad = []
    for f, r in inv.items():
        p = ROOT / f
        if not p.exists():
            bad.append((f, "MISSING"))
        elif msi.sha256(p) != r["sha256"]:
            bad.append((f, "MISMATCH"))
    out["protected_inventory"] = {"source": "artifacts/semantic-edl-v1/protected_inventory_before.json",
                                  "n_files": len(inv), "n_bad": len(bad), "bad": bad[:200], "all_match": not bad}
    # 3. semantic-edl-v1 own files + reused runs unmodified since the slk preregistration
    from csr.slk_paths import REUSED
    roots = ["artifacts/semantic-edl-v1", "configs/semantic-edl-v1", "docs"] + list(REUSED.values())
    newer = []
    for r in roots:
        for dp, _, fns in os.walk(ROOT / r):
            for fn in fns:
                p = Path(dp) / fn
                rel = str(p.relative_to(ROOT))
                if r == "docs" and not fn.startswith(("SEMANTIC_EDL_V1", "SEM_")):
                    continue
                if p.stat().st_mtime > t0:
                    newer.append(rel)
    for p in list(ROOT.glob("sem_*.py")) + list((ROOT / "src/csr").glob("sem_*.py")) + [ROOT / "SEMANTIC_EDL_V1_REPORT.md"]:
        if p.exists() and p.stat().st_mtime > t0:
            newer.append(str(p.relative_to(ROOT)))
    out["semantic_edl_v1_and_reused_runs_unmodified"] = {"n_modified_after_prereg": len(newer), "files": newer[:200],
                                                        "ok": not newer}
    # 4. test split
    refs = []
    for p in list(ROOT.glob("slk_*.py")) + list((ROOT / "src/csr").glob("slk_*.py")):
        txt = p.read_text()
        for tok in ("test_tokens", "test_docs", "test_offsets", "split=\"test\"", "'test'", "\"test\""):
            if tok in txt and p.name not in ("slk_tests.py", "slk_tests_ported.py", "slk_integrity.py"):
                for ln, line in enumerate(txt.splitlines(), 1):
                    if tok in line and "def test_" not in line:
                        refs.append(f"{p.name}:{ln}: {line.strip()[:120]}")
    data = ROOT / "artifacts/data"
    tfiles = [p for p in data.rglob("*") if p.is_file() and msi.is_test_split_name(str(p.relative_to(ROOT)))] \
        if data.exists() else []
    acc = [{"file": str(p.relative_to(ROOT)), "atime": p.stat().st_atime, "read_after_prereg": p.stat().st_atime > t0}
           for p in tfiles]
    out["test_split"] = {"slk_code_references": refs, "n_test_files": len(tfiles),
                         "n_accessed_after_prereg": sum(a["read_after_prereg"] for a in acc), "files": acc[:50],
                         "note": "atime under relatime is updated on a read when the previous atime is older than "
                                 "mtime or > 24 h; supplementary evidence only. The evaluator modes are val / dev / "
                                 "calibsel; none loads a test split."}
    # 5. Phase-1 runs
    import slk_run
    rd = slk_run.run_dirs()
    p1 = {k: v for k, v in rd.items() if k.startswith("P1_")}
    rows = {}
    for rn, d in sorted(p1.items()):
        st = json.loads((ROOT / d / "status.json").read_text()) if (ROOT / d / "status.json").exists() else {}
        rows[rn] = {"dir": d, "status": st.get("status"), "steps": st.get("steps_done"), "tokens": st.get("seen"),
                    "plan_sha256": st.get("plan_sha256")}
    plans = {r["plan_sha256"] for r in rows.values()}
    toks = {r["tokens"] for r in rows.values()}
    stp = {r["steps"] for r in rows.values()}
    cfgdiff = {}
    for s in (0, 1, 2, 3):
        a = ROOT / "configs/semantic-edl-long-k" / f"P1_C6_s{s}.json"
        b = ROOT / "configs/semantic-edl-long-k" / f"P1_SHUF_s{s}.json"
        if a.exists() and b.exists():
            fa, fb = flat(json.loads(a.read_text())), flat(json.loads(b.read_text()))
            cfgdiff[s] = sorted(k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k))
    gc = json.loads((ART / "gpu_checks.json").read_text())
    out["phase1_runs"] = {"runs": rows, "all_complete": all(r["status"] == "COMPLETE" for r in rows.values()),
                          "plan_sha256_set": sorted(str(x) for x in plans), "tokens_set": sorted(str(x) for x in toks),
                          "steps_set": sorted(str(x) for x in stp),
                          "c6_vs_shuf_config_diff_keys": cfgdiff,
                          "c6_vs_shuf_differ_only_in_table": all(set(v) <= {"run_name", "cond.paper_latent.table",
                                                                            "cond.paper_latent.table_sha256"}
                                                                 for v in cfgdiff.values()),
                          "params_equal": gc.get("params_equal")}
    out["all_ok"] = bool(out["protected_inventory"]["all_match"] and out["semantic_edl_v1_and_reused_runs_unmodified"]["ok"]
                         and not refs and out["phase1_runs"]["all_complete"] and len(plans) == 1 and len(toks) == 1
                         and out["phase1_runs"]["c6_vs_shuf_differ_only_in_table"] and gc.get("params_equal"))
    p = ART / f"integrity_{guard.utc_stamp()}.json"
    guard.atomic_write_json(p, out, mode=0o400)
    print(json.dumps({k: v for k, v in out.items() if k not in ("phase1_runs",)}, default=str)[:4000])
    print(json.dumps({k: v for k, v in out["phase1_runs"].items() if k != "runs"}))


if __name__ == "__main__":
    main()
