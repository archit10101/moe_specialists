#!/usr/bin/env python
"""csr-semantic-edl-v1: hash audit of every reused artifact (read-only). Writes artifacts/semantic-edl-v1/
reuse_manifest_<stamp>.json: fle final/milestone checkpoints (vs fle checkpoint_ledger.tsv and status.json), fle VAL
evaluations, semantic tables / net / standardisation (vs gc decision), split manifest, preregistration, configs."""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402


def main():
    guard.check_root()
    from csr.sem_paths import ART, FINAL_RUNS, FLE_ART, GC_LAT, PREREG_SHA, SEM_CS_SHA, SEM_SEL_SHA, SEM_TABLE_SHA, \
        T4_S100_RUN, check_prereg
    check_prereg()
    R = guard.ROOT
    out = {"time": guard.now(), "prereg_sha256": PREREG_SHA, "checkpoints": {}, "bad": []}
    led = {r["path"]: r["sha256"] for r in csv.DictReader(open(FLE_ART / "checkpoint_ledger.tsv"), delimiter="\t")
           if r["kind"] == "milestone"}
    for name, rd in list(FINAL_RUNS.items()) + [("T4_s100", T4_S100_RUN)]:
        st = json.loads((R / rd / "status.json").read_text())
        cfg = json.loads((R / rd / "config.json").read_text())
        for tag, c in st["ckpts"].items():
            h = guard.sha256_file(R / c["path"] / "state.pt")
            ok = h == c["sha256"] and led.get(c["path"]) == h
            out["checkpoints"][f"{name}:{tag}"] = {"path": c["path"], "sha256": h, "ok": ok, "step": c["step"]}
            if not ok:
                out["bad"].append(c["path"])
        out["checkpoints"][f"{name}:config"] = {"seed": cfg["seed"], "cond": cfg["cond"], "status": st["status"],
                                                "plan_sha256": st.get("plan_sha256"),
                                                "config_sha256": guard.sha256_json(cfg)}
    out["fle_val_evals"] = {k: guard.sha256_file(FLE_ART / "evals" / f"final_{k}" / "final.json")
                            for k in FINAL_RUNS}
    tabs = {"table_SEM.npz": SEM_TABLE_SHA, "table_SEM_calib_search.npz": SEM_CS_SHA,
            "table_SEM_calib_sel.npz": SEM_SEL_SHA}
    out["semantic"] = {}
    for f, s in tabs.items():
        h = guard.sha256_file(GC_LAT / f)
        out["semantic"][f] = {"sha256": h, "ok": h == s}
        if h != s:
            out["bad"].append(f)
    for f in ("SEM_K16_s2/net.pt", "SEM_K16_s2/view_logits.npz", "residualizer_params.npz", "features.npz"):
        out["semantic"][f] = {"sha256": guard.sha256_file(GC_LAT / f)}
    dec = json.loads((R / "artifacts/gc-edl-v1/discovery_decision.json").read_text())
    m = dec["metrics"]["SEM_K16"]
    out["semantic"]["discovery_metrics_SEM_K16"] = {k: m[k] for k in ("kappa_view", "kappa_seed", "C", "LD", "medoid_seed")}
    out["semantic"]["decision_table_sha_matches"] = dec["tables"]["SEM"]["sha256"] == SEM_TABLE_SHA
    out["semantic"]["encoder_sha256"] = guard.sha256_file(R / "artifacts/encoder/bge-small-en-v1.5/model.safetensors")
    man = json.loads((R / "artifacts/forecast-specialist-pretraining-v2/data/manifest.json").read_text())
    out["splits_sha256"] = man.get("splits_sha256")
    out["tables_json"] = json.loads((ART / "tables" / "tables.json").read_text())
    out["all_ok"] = not out["bad"]
    p = ART / f"reuse_manifest_{guard.utc_stamp()}.json"
    guard.atomic_write_json(p, out, mode=0o400)
    print(json.dumps({"all_ok": out["all_ok"], "n_ckpts": len([k for k in out["checkpoints"] if not k.endswith(":config")]),
                      "bad": out["bad"], "splits": out["splits_sha256"], "p": str(p)}))
    sys.exit(0 if out["all_ok"] else 1)


if __name__ == "__main__":
    main()
