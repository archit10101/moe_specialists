#!/usr/bin/env python
"""recr1 manifests and protected inventory.

  recr1_manifest.py before   protected_inventory_before.json + reuse_manifest.json + data_manifest.json (all 0400)
  recr1_manifest.py verify   re-hash protected set and reused files -> protected_inventory_verify_<stamp>.json

data_manifest: paper-level split of the spb1 SEL partition (CALIB-SEARCH half A) into UTRAIN / CSEARCH / CSELECT by
sha256(SPLIT_SALT + paper_sha) mod 10 (independent of any result). Nothing outside ART/CFG_DIR/TMP is written."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402
from csr import recr1_paths as P  # noqa: E402

PROTECT_DIRS = ["artifacts/semantic-pair-bank-unrestricted-router-v1", "artifacts/subject-conditioned-utility-router-v1",
                "artifacts/csr-current-context-specialist-router-v1/routers", "artifacts/oracle-residual-v1/tables",
                "runs/20261116T023759Z_semantic_edl_long_k_P1_T4_s3/ckpt_p100"]
PROTECT_GLOBS = ["docs/SEMANTIC_PAIR_BANK_UNRESTRICTED_ROUTER_V1_*", "docs/CSR_CURRENT_CONTEXT_SPECIALIST_ROUTER_V1_*",
                 "docs/SUBJECT_CONDITIONED_UTILITY_ROUTER_V1_*", "artifacts/csr-current-context-specialist-router-v1/*.json",
                 "artifacts/csr-current-context-specialist-router-v1/*.md", "spb1_*.py", "src/csr/spb1_*.py",
                 "src/csr/guard.py", "src/csr/orv1_*.py", "src/csr/lfr_model.py", "scripts/gpu_guarded.sh",
                 "scripts/run_registered.sh", "env.sh"]


def protected_files():
    out = []
    for d in PROTECT_DIRS:
        out += [p for p in sorted((guard.ROOT / d).rglob("*")) if p.is_file()]
    for g in PROTECT_GLOBS:
        out += [p for p in sorted(guard.ROOT.glob(g)) if p.is_file()]
    seen, uniq = set(), []
    for p in out:
        if p not in seen and "/.driver.lock" not in str(p):
            seen.add(p)
            uniq.append(p)
    return uniq


def reused_files():
    fs = []
    for s in P.BANK_SEEDS:
        fs += [P.EXTRACT / f"common_s{s}/common_SEL.npz", P.EXTRACT / f"common_s{s}/common_SEL.json"]
        for b in P.BANKS:
            fs.append(P.EXTRACT / f"g0_s{s}_S1/bank_{b}_SEL.npz")
        fs.append(P.EXTRACT / f"g0_s{s}_S1/summary_SEL.json")
        for r in P.ROUTER_SEEDS:
            for t in ("R_SEM", "R_GEN", "R_SHUF", "R_COARSE16"):
                fs += [P.ROUTES / f"{t}_s{s}_r{r}_SEL.npz", P.ROUTES / f"{t}_s{s}_r{r}_SEL.json", P.ROUTES / f"{t}_s{s}_r{r}.pt"]
        fs.append(P.SPB1 / f"runs/g0_s{s}/ckpt_50000000.pt")
    fs += [P.SPB1 / "analysis/stage1_g0.json", P.SPB1 / "analysis/stage2_g1_S2_g1.json", P.SPB1 / "final_analysis.json",
           P.SPB1 / "tables/labels_main.npz", P.SPB1 / "tables/labels_cs.npz",
           guard.ROOT / "artifacts/oracle-residual-v1/tables/token_feats.npz",
           guard.ROOT / "artifacts/oracle-residual-v1/tables/parts.npz"]
    return fs


def split_of(paper_sha):
    k = int(hashlib.sha256((P.SPLIT_SALT + paper_sha).encode()).hexdigest(), 16) % 10
    for name, rg in P.SPLITS.items():
        if k in rg:
            return name
    raise AssertionError


def data_manifest():
    c = np.load(P.EXTRACT / "common_s0/common_SEL.npz")
    papers = [str(x) for x in c["papers"]]
    paper = c["paper"].astype(np.int64)
    ntok = np.bincount(paper, minlength=len(papers))
    doc = c["doc"]
    ndoc = np.bincount(paper[np.r_[True, doc[1:] != doc[:-1]]], minlength=len(papers))
    for s in P.BANK_SEEDS[1:]:     # identical SEL tokens/papers across seeds (spb1 pooled analyses rely on this)
        cs = np.load(P.EXTRACT / f"common_s{s}/common_SEL.npz")
        assert list(cs["papers"]) == papers and np.array_equal(cs["tgt"], c["tgt"]) and np.array_equal(cs["paper"], c["paper"])
    out = {"time": guard.now(), "source": "spb1 SEL = orv1 parts CALIB-SEARCH half A (paper-disjoint from spb1 TRAIN "
                                          "= half B, from CALIB-SELECTION, VALIDATION and test)",
           "salt": P.SPLIT_SALT, "rule": "sha256(salt + paper_sha256) mod 10: 0-3 UTRAIN, 4-6 CSEARCH, 7-9 CSELECT",
           "partitions": {}}
    for name in P.SPLITS:
        idx = [i for i, p in enumerate(papers) if split_of(p) == name]
        out["partitions"][name] = {"paper_index": idx, "paper_sha256": [papers[i] for i in idx], "n_papers": len(idx),
                                   "n_docs": int(ndoc[idx].sum()), "n_tokens": int(ntok[idx].sum())}
    return out


def main():
    guard.check_root()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "before"
    P.ART.mkdir(parents=True, exist_ok=True)
    if cmd == "before":
        p = P.ART / "protected_inventory_before.json"
        if p.exists():
            print("exists")
            return
        snap = {str(f.relative_to(guard.ROOT)): {"sha256": guard.sha256_file(f), "bytes": f.stat().st_size}
                for f in protected_files()}
        guard.atomic_write_json(p, {"time": guard.now(), "n_files": len(snap), "files": snap}, mode=0o400)
        rm = {"time": guard.now(), "files": {str(f.relative_to(guard.ROOT)): snap.get(str(f.relative_to(guard.ROOT)), {}).get(
            "sha256") or guard.sha256_file(f) for f in reused_files()},
              "missing_required": {
                  "TOKEN-4 245M ckpt_p100 seed 0": "runs/20261114T131547Z_forecast_latent_edl_v1_final_T4_s0 (sha fb113eb6...) DELETED",
                  "TOKEN-4 245M ckpt_p100 seed 1": "runs/20261114T145059Z_forecast_latent_edl_v1_final_T4_s1 (sha 0d46807b...) DELETED",
                  "TOKEN-4 245M ckpt_p100 seed 2": "runs/20261114T221957Z_forecast_latent_edl_v1_final_T4_s2 (sha 40c61348...) DELETED"},
              "timing_only": "runs/20261116T023759Z_semantic_edl_long_k_P1_T4_s3/ckpt_p100 (same architecture; used ONLY for "
                             "latency micro-benchmarks, never for any CE or routing result)"}
        guard.atomic_write_json(P.ART / "reuse_manifest.json", rm, mode=0o400)
        dm = data_manifest()
        guard.atomic_write_json(P.ART / "data_manifest.json", dm, mode=0o400)
        guard.atomic_write_bytes(P.ART / "data_manifest.json.sha256",
                                 (guard.sha256_file(P.ART / "data_manifest.json") + "  data_manifest.json\n").encode())
        print(json.dumps({"n_protected": len(snap), "n_reused": len(rm["files"]),
                          "partitions": {k: {kk: v[kk] for kk in ("n_papers", "n_docs", "n_tokens")}
                                         for k, v in dm["partitions"].items()}}))
    else:
        before = json.loads((P.ART / "protected_inventory_before.json").read_text())["files"]
        now = {str(f.relative_to(guard.ROOT)): guard.sha256_file(f) for f in protected_files()}
        changed = [k for k, v in before.items() if now.get(k) != v["sha256"]]
        missing = [k for k in before if k not in now]
        rm = json.loads((P.ART / "reuse_manifest.json").read_text())["files"]
        reuse_bad = [k for k, h in rm.items() if guard.sha256_file(guard.ROOT / k) != h]
        res = {"time": guard.now(), "n_before": len(before), "n_now": len(now), "changed": changed, "missing": missing,
               "reuse_mismatch": reuse_bad, "all_match": not changed and not missing and not reuse_bad}
        guard.atomic_write_json(P.ART / f"protected_inventory_verify_{guard.utc_stamp()}.json", res, mode=0o400)
        print(json.dumps({k: (v if not isinstance(v, list) else len(v)) for k, v in res.items()}))


if __name__ == "__main__":
    main()
