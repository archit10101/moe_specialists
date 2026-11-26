#!/usr/bin/env python
"""scur1 protected inventory (copy of asrs2_inventory.py with the scur1 protected set: asrs2 added);
asrs2 protected inventory (copy of asrs1_inventory.py with the asrs2 protected set);
asrs1 protected inventory (prereg section 9) and baseline-reuse manifest.

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

DIRS = ["artifacts/csr-continuous-base-target-free-asrs-v2", "runs/20261122T051254Z_asrs2_S0F",
        "runs/20261122T053544Z_asrs2_S1", "runs/20261122T064506Z_asrs2_C1B1", "runs/20261122T065255Z_asrs2_C1B2",
        "artifacts/csr-alternating-semantic-residual-specialization-v1", "runs/20261122T001339Z_asrs1_S1",
        "artifacts/csr-current-context-specialist-router-v1", "artifacts/csr-centered-fedavg-residual-v1",
        "artifacts/csr-crossfit-semantic-utility-residual-v1", "runs/20261121T151627Z_cfar1_S1_s100",
        "runs/20261121T194254Z_ccsr1_S3", "artifacts/oracle-residual-v1/tables", "artifacts/gc-edl-v1/latent/SEM_K16_s2"]
GLOBS = ["asrs2_*.py", "src/csr/asrs2_*.py", "docs/CSR_CONTINUOUS_BASE_TARGET_FREE_ASRS_V2_*", "asrs1_*.py", "src/csr/asrs1_*.py", "docs/CSR_ALTERNATING_SEMANTIC_RESIDUAL_SPECIALIZATION_V1_*",
         "src/csr/ccsr1_router.py", "src/csr/csur1_bank.py", "src/csr/orv1_lib.py", "orv1_tables.py", "cfar1_train.py",
         "ccsr1_*.py", "cfar1_*.py", "csur1_*.py", "src/csr/ccsr1_*.py", "src/csr/cfar1_*.py", "src/csr/csur1_*.py",
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
    from csr import scur1_paths as P
    man = P.verify_reused()
    return {"time": guard.now(), "verified_sha256": man,
            "start": {"milestone": P.MILESTONES[P.PRIMARY], "note": "asrs2 Stage-1 seed-0 continuation final milestone "
                      "(base + G_0, ALTG_0, SEMh_0, SHUFh_0 banks and asrs2 R1 routers); frozen, never updated by scur1"},
            "sel_tables": "asrs2 SEL evaluation of m1111 (same rows/order/evaluator; reused, not recomputed)"}


def main():
    from csr.scur1_paths import ART
    guard.check_root()
    cmd = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "before"
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
        print(json.dumps({"n_files": len(snap), "reused_verified": len(rm["verified_sha256"])}))
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
