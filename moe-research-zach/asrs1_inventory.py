#!/usr/bin/env python
"""asrs1 protected inventory (prereg section 9) and baseline-reuse manifest.

  asrs1_inventory.py before   hash every file of the protected set -> ART/protected_inventory_before.json (0400) and
                              ART/reuse_manifest.json (0400; reused checkpoints, routers, tables, results + evidence)
  asrs1_inventory.py verify   re-hash the protected set -> ART/protected_inventory_verify_<stamp>.json (all_match)
Protected set: the cfar1 / ccsr1 / csur1 artifact directories, the reused cfar1 and ccsr1 run directories, the ccsr1 /
cfar1 / csur1 code and reports, and the reused orv1 tables. Nothing outside ART is written."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402

DIRS = ["artifacts/csr-current-context-specialist-router-v1", "artifacts/csr-centered-fedavg-residual-v1",
        "artifacts/csr-crossfit-semantic-utility-residual-v1", "runs/20261121T151627Z_cfar1_S1_s100",
        "runs/20261121T194254Z_ccsr1_S3", "artifacts/oracle-residual-v1/tables"]
GLOBS = ["ccsr1_*.py", "cfar1_*.py", "csur1_*.py", "src/csr/ccsr1_*.py", "src/csr/cfar1_*.py", "src/csr/csur1_*.py",
         "src/csr/jsr_bank.py", "src/csr/lfr_model.py", "src/csr/lfr_lib.py", "src/csr/fsp_lib.py", "src/csr/guard.py",
         "docs/CSR_CURRENT_CONTEXT_SPECIALIST_ROUTER_V1_*", "docs/CSR_CROSSFIT_SEMANTIC_UTILITY_RESIDUAL_V1_*",
         "docs/CSR_CENTERED_FEDAVG_RESIDUAL_V1_*", "ORACLE_RESIDUAL_V1_REPORT.md", "SEMANTIC_EDL_LONG_K_PHASE2A_REPORT.md",
         "SEMANTIC_LIFT_245M_CONFIRM_V1_REPORT.md"]


def files():
    out = []
    for d in DIRS:
        out += [p for p in sorted((guard.ROOT / d).rglob("*")) if p.is_file()]
    for g in GLOBS:
        out += [p for p in sorted(guard.ROOT.glob(g)) if p.is_file()]
    seen, uniq = set(), []
    for p in out:
        if p not in seen and "/.driver.lock" not in str(p):
            seen.add(p)
            uniq.append(p)
    return uniq


def snapshot():
    return {str(p.relative_to(guard.ROOT)): {"sha256": guard.sha256_file(p), "bytes": p.stat().st_size} for p in files()}


def reuse_manifest():
    from csr import asrs1_paths as P
    man = P.verify_reused()
    s3 = json.loads((guard.ROOT / "runs/20261121T194254Z_ccsr1_S3/status.json").read_text())
    c1 = json.loads((guard.ROOT / P.CFAR1_RUN / "status.json").read_text())
    return {"time": guard.now(), "verified_sha256": man,
            "start_checkpoint": {"path": P.CKPT, "sha256": P.CKPT_SHA, "step": 318, "seen": 20057391,
                                 "cfar1_status_ckpts": c1.get("ckpts", {}).get("m20")},
            "ordinary_continuation": {"path": P.CCSR1_S3, "sha256": P.CCSR1_S3_SHA, "steps": [318, 1111],
                                      "seen": s3["seen"], "plan_sha256": s3["plan_sha256"], "status": s3["status"],
                                      "reason": s3["reason"], "controls": {"A": "ordinary base", "B": "ordinary G",
                                                                           "C": "SEMj_r (joint router), SEMf_r (frozen router)"},
                                      "evaluator": "re-measured on SEL by asrs1_eval.py (--format ccsr1); not retrained",
                                      "matched": "same plan sha, same rows 318..1110, same 50,086,846 tokens, same start "
                                                 "checkpoint and fresh AdamW moments + 20-step re-warmup"},
            "frozen_stage_routers": {k: v for k, v in man.items() if "/routers/" in k},
            "not_reused": {"ALT-G": "no schedule-matched generic control exists; trained in the asrs1 process",
                           "ASRS-SHUF": "trained in the asrs1 process (identical pipeline, p_SH prior)"},
            "plan_sha256_expected": P.PLAN_SHA, "plan_match": s3["plan_sha256"] == P.PLAN_SHA}


def main():
    from csr.asrs1_paths import ART
    guard.check_root()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "before"
    ART.mkdir(parents=True, exist_ok=True)
    if cmd == "before":
        p = ART / "protected_inventory_before.json"
        if p.exists():
            print("exists")
            return
        snap = snapshot()
        guard.atomic_write_json(p, {"time": guard.now(), "n_files": len(snap), "files": snap}, mode=0o400)
        rm = reuse_manifest()
        guard.atomic_write_json(ART / "reuse_manifest.json", rm, mode=0o400)
        print(json.dumps({"n_files": len(snap), "plan_match": rm["plan_match"]}))
    else:
        before = json.loads((ART / "protected_inventory_before.json").read_text())["files"]
        now = snapshot()
        changed = [k for k, v in before.items() if now.get(k, {}).get("sha256") != v["sha256"]]
        missing = [k for k in before if k not in now]
        added = [k for k in now if k not in before]
        res = {"time": guard.now(), "n_before": len(before), "n_now": len(now), "changed": changed, "missing": missing,
               "added_outside_baseline": added, "all_match": not changed and not missing}
        guard.atomic_write_json(ART / f"protected_inventory_verify_{guard.utc_stamp()}.json", res, mode=0o400)
        print(json.dumps({k: (v if not isinstance(v, list) else len(v)) for k, v in res.items()}))


if __name__ == "__main__":
    main()
