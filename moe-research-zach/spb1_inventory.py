#!/usr/bin/env python
"""spb1 protected inventory (adapted from scur1_inventory.py) and reuse manifest.

  spb1_inventory.py before   hash every file of the protected set -> ART/protected_inventory_before.json (0400) and
                             ART/reuse_manifest.json (0400)
  spb1_inventory.py verify   re-hash -> ART/protected_inventory_verify_<stamp>.json (all_match)
Protected set: reused TOKEN-4 base checkpoints (seeds 0-2: status + ckpt_p100), the orv1 tables / runs / VAL assignment /
results, the csur1 region-prior tables, the GC-EDL SEM K16 net, the asrs2 / scur1 / ccsr1 result directories that were
read, the reused code, and the prior reports read for this experiment. Nothing outside ART/TMP is written."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402

DIRS = ["runs/20261114T131547Z_forecast_latent_edl_v1_final_T4_s0/ckpt_p100",
        "runs/20261114T145059Z_forecast_latent_edl_v1_final_T4_s1/ckpt_p100",
        "runs/20261114T221957Z_forecast_latent_edl_v1_final_T4_s2/ckpt_p100",
        "artifacts/oracle-residual-v1/tables", "artifacts/oracle-residual-v1/runs", "artifacts/oracle-residual-v1/val_assign",
        "artifacts/csr-crossfit-semantic-utility-residual-v1/tables", "artifacts/gc-edl-v1/latent/SEM_K16_s2",
        "artifacts/subject-conditioned-utility-router-v1", "artifacts/csr-continuous-base-target-free-asrs-v2",
        "artifacts/csr-current-context-specialist-router-v1/routers"]
GLOBS = ["runs/20261114T131547Z_forecast_latent_edl_v1_final_T4_s0/status.json",
         "runs/20261114T145059Z_forecast_latent_edl_v1_final_T4_s1/status.json",
         "runs/20261114T221957Z_forecast_latent_edl_v1_final_T4_s2/status.json",
         "artifacts/oracle-residual-v1/*.json", "artifacts/oracle-residual-v1/*.jsonl", "artifacts/oracle-residual-v1/*.md",
         "artifacts/csr-current-context-specialist-router-v1/*.json", "artifacts/csr-current-context-specialist-router-v1/*.md",
         "orv1_*.py", "src/csr/orv1_*.py", "ccsr1_*.py", "src/csr/ccsr1_*.py", "scur1_*.py", "src/csr/scur1_*.py",
         "src/csr/lfr_model.py", "src/csr/lfr_lib.py", "src/csr/lfr_eval.py", "src/csr/fsp_lib.py", "src/csr/guard.py",
         "src/csr/jsr_bank.py", "fle_evaluate.py", "scripts/gpu_guarded.sh", "scripts/run_registered.sh", "env.sh",
         "ORACLE_RESIDUAL_V1_REPORT.md", "ORACLE_RESIDUAL_V1_DIAGNOSTIC_NOTES.md", "docs/ORACLE_RESIDUAL_V1_PREREG.md",
         "docs/CSR_CURRENT_CONTEXT_SPECIALIST_ROUTER_V1_*", "docs/SUBJECT_CONDITIONED_UTILITY_ROUTER_V1_*",
         "docs/CSR_CONTINUOUS_BASE_TARGET_FREE_ASRS_V2_*", "SEMANTIC_LIFT_EXPERT_LEARNING_V1_REPORT.md",
         "SEMANTIC_LIFT_245M_CONFIRM_V1_REPORT.md", "SEMANTIC_EDL_V1_REPORT.md", "SEMANTIC_EDL_LONG_K_PHASE2A_REPORT.md",
         "docs/SEMANTIC_LIFT_*", "docs/SEMANTIC_EDL_*"]


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


def main():
    from csr import spb1_paths as P
    guard.check_root()
    cmd = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "before"
    P.ART.mkdir(parents=True, exist_ok=True)
    if cmd == "before":
        p = P.ART / "protected_inventory_before.json"
        if p.exists():
            print("exists")
            return
        snap = snapshot()
        guard.atomic_write_json(p, {"time": guard.now(), "n_files": len(snap), "files": snap}, mode=0o400)
        rm = {"time": guard.now(), "verified_sha256": P.verify_reused(),
              "base": "TOKEN-4 245M ckpt_p100 seeds 0/1/2 (orv1 T4_RUNS; sha verified by orv1_lib.load_base at load)",
              "plan": "orv1 plan.npz: base-unseen main documents in fixed sha256 order, 32 x 1024-token rows/step",
              "partitions": "orv1 parts.npz: SEL = CALIB-SEARCH paper-half A; EVAL (here TRAIN) = paper-half B",
              "semantic_labels": "orv1 hard argmax of the frozen GC-EDL SEM K16 label-free table (main + CS)",
              "coarse16": "orv1 condition-A 50M checkpoints, seeds 0-2 (same base, plan, LR, budget, init scheme)",
              "region_prior": "csur1 psem_calib_search (evaluation stratifier for transitions only)"}
        guard.atomic_write_json(P.ART / "reuse_manifest.json", rm, mode=0o400)
        print(json.dumps({"n_files": len(snap), "reused_verified": len(rm["verified_sha256"])}))
    else:
        before = json.loads((P.ART / "protected_inventory_before.json").read_text())["files"]
        now = snapshot()
        changed = [k for k, v in before.items() if now.get(k, {}).get("sha256") != v["sha256"]]
        missing = [k for k in before if k not in now]
        res = {"time": guard.now(), "n_before": len(before), "n_now": len(now), "changed": changed, "missing": missing,
               "all_match": not changed and not missing}
        guard.atomic_write_json(P.ART / f"protected_inventory_verify_{guard.utc_stamp()}.json", res, mode=0o400)
        print(json.dumps({k: (v if not isinstance(v, list) else len(v)) for k, v in res.items()}))


if __name__ == "__main__":
    main()
