#!/usr/bin/env python
"""csr-context-prior-specialization-v2: verify every reused artifact against its independent hash record BEFORE any
V2 GPU work (CPU only, read-only). Writes artifacts/context-prior-specialization-v2/reuse_manifest.json (0400).
Never reads the locked test split.
  data      : fspv2 split files (main, calib_search, calib_sel, val) vs data/manifest.json
  tokenizer : sha of every file under artifacts/tokenizer (+ revision string from the data manifest)
  drafts    : six frozen draft checkpoints vs V1 draft_provenance.json
  cache     : every V1 forecast-cache chunk (.npy) vs its COMPLETE.json sha256 (and the V1 cache manifests)
  V1 runs   : TOKEN-4 milestones (S1 s100, S2 s100/s101, FINAL s0) re-verified (meta sha256 vs state.pt) + sha of the
              reused result files; descriptive OUTPUT-DISTILL / MEMORY-ONLY / EDL development results
  labels    : V1 labeler windows / centroids
  code      : V1 modules imported by V2 (sha recorded; lfr_audit compared with V1 spec_audit_config.json)
  cf        : V1 counterfactual-estimator validation (cf_validation.json, gate) and V1 cfaudits of TOKEN-4"""
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402

ROOT = guard.ROOT


def main():
    guard.check_root()
    from csr.cps_lib import ART, V1_ART, CACHE_DIR, V1_RUN_PREFIX
    from csr import cps_ckpt
    out_p = ART / "reuse_manifest.json"
    if out_p.exists():
        print(out_p.read_text()[:2000])
        return
    sha = guard.sha256_file
    res = {"time": guard.now(), "checks": {}, "failures": []}

    def chk(name, ok, **info):
        res["checks"][name] = {"ok": bool(ok), **info}
        if not ok:
            res["failures"].append(name)
    # data
    D = ROOT / "artifacts/forecast-specialist-pretraining-v2/data"
    man = json.loads((D / "manifest.json").read_text())
    for s in ("main", "calib_search", "calib_sel", "val"):
        m = man["splits"][s]
        got = {"tokens": sha(D / f"{s}_tokens.npy"), "offsets": sha(D / f"{s}_offsets.npy"),
               "docs": sha(D / f"{s}_docs.json")}
        chk(f"data:{s}", got["tokens"] == m["sha256_tokens"] and got["offsets"] == m["sha256_offsets"] and
            got["docs"] == m["sha256_docs"], **got)
    chk("data:manifest", True, sha256=sha(D / "manifest.json"), splits_sha256=man["splits_sha256"],
        test_split="locked, not read")
    # tokenizer
    tk = {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / "artifacts/tokenizer").rglob("*")) if p.is_file()}
    chk("tokenizer", len(tk) > 0, files=tk, revision=man["tokenizer"])
    # drafts
    prov = json.loads((V1_ART / "draft_provenance.json").read_text())["drafts"]
    for d in prov:
        chk(f"draft:{d['slot']}", sha(ROOT / d["path"]) == d["sha256"], path=d["path"], sha256=d["sha256"])
    # forecast cache (V1, read-only)
    for region in ("main", "calib_search"):
        mm = json.loads((V1_ART / f"cache_manifest_{region}.json").read_text())
        ok_all, n_tok = True, 0
        for c in mm["chunks"]:
            cd = CACHE_DIR / region / c["chunk"]
            meta = json.loads((cd / "COMPLETE.json").read_text())
            ok = meta["sha256"] == c["sha256"] and all(sha(cd / f"{k}.npy") == v for k, v in meta["sha256"].items())
            ok_all &= ok
            n_tok += meta["n_tokens"]
        chk(f"cache:{region}", ok_all and mm.get("complete", True), n_chunks=len(mm["chunks"]), n_tokens=n_tok,
            manifest_sha256=sha(V1_ART / f"cache_manifest_{region}.json"))
    # V1 runs
    def v1dir(n):
        c = sorted(glob.glob(str(ROOT / f"runs/*_{V1_RUN_PREFIX}{n}")))
        return Path(c[-1]) if c else None
    for n in ["s1_TOKEN-4_s100", "s2_TOKEN-4_s100", "s2_TOKEN-4_s101", "final_TOKEN-4_s0"]:
        rd = v1dir(n)
        s = json.loads((rd / "summary.json").read_text())
        ms = {}
        ok = s["status"] == "COMPLETE"
        for tag, e in s["ckpts"].items():
            p = ROOT / e["path"]
            v = cps_ckpt.verify_dir(p) and json.loads((p / "meta.json").read_text())["sha256"] == e["sha256"]
            ms[tag] = {"sha256": e["sha256"], "ok": bool(v)}
            ok &= bool(v)
        files = {f: sha(rd / f) for f in ("summary.json", "metrics.jsonl", "config.json", "dev_audit.json",
                                          "calib_sel_eval.json") if (rd / f).exists()}
        no_val = not (rd / "final_eval.json").exists()
        chk(f"v1run:{n}", ok and no_val, run_dir=str(rd.relative_to(ROOT)), milestones=ms, files=files,
            validation_never_evaluated=no_val)
    for n in ["s1_OD-a1.0_s100", "s1_OD-a0.3_s100", "s2_OUTPUT-DISTILL_s100", "s2_OUTPUT-DISTILL_s101",
              "s2_MEMORY-ONLY_s100", "s2_MEMORY-ONLY_s101", "s1_EDL-b5e-3_s100", "s2_EDL_s100", "s2_EDL_s101"]:
        rd = v1dir(n)
        files = {f: sha(rd / f) for f in ("summary.json", "metrics.jsonl", "dev_audit.json", "calib_sel_eval.json")
                 if (rd / f).exists()}
        chk(f"v1desc:{n}", json.loads((rd / "summary.json").read_text())["status"] == "COMPLETE",
            run_dir=str(rd.relative_to(ROOT)), files=files)
    # labels
    L = V1_ART / "labels"
    chk("labels", all((L / f).exists() for f in ("windows_calib.npz", "windows_val.npz", "centroids.npy")),
        files={f.name: sha(f) for f in sorted(L.iterdir())})
    # code reused read-only
    code = {f: sha(ROOT / f) for f in ["src/csr/lfr_model.py", "src/csr/lfr_eval.py", "src/csr/lfr_audit.py",
                                       "src/csr/lfr_lib.py", "src/csr/lfr_ckpt.py", "lfr_train.py", "src/csr/fsp_lib.py",
                                       "src/csr/fsp_model.py", "src/csr/model.py", "src/csr/guard.py"]}
    sac = json.loads((V1_ART / "spec_audit_config.json").read_text())
    chk("code:lfr_audit_matches_v1_prereg", code["src/csr/lfr_audit.py"] == sac["code"]["src/csr/lfr_audit.py"],
        files=code)
    # counterfactual-utility artifacts
    cv = json.loads((V1_ART / "cf_validation.json").read_text())
    chk("cf_validation", cv["gate_pass"], sha256=sha(V1_ART / "cf_validation.json"), mean_spearman=cv["mean_spearman"],
        mean_top4_overlap=cv["mean_top4_overlap"])
    chk("v1_protocol", True, sha256=sha(ROOT / "docs/LATENT_FORECAST_ROUTER_DISTILL_V1_PROTOCOL.md"),
        v1_stop=json.loads((V1_ART / "driver_state.json").read_text()).get("operator_stop", {}).get("status"))
    res["all_ok"] = not res["failures"]
    guard.atomic_write_json(out_p, res, mode=0o400)
    print(json.dumps({"all_ok": res["all_ok"], "failures": res["failures"], "n_checks": len(res["checks"])}))
    sys.exit(0 if res["all_ok"] else 1)


if __name__ == "__main__":
    main()
