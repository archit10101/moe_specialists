#!/usr/bin/env python
"""csr-forecast-counterfactual-utility-v1: read-only integrity verification of everything this experiment relied on
or must not have touched. Output only to artifacts/forecast-counterfactual-utility-v1/integrity_<stamp>.json.

  (1) the pre-fle protected inventory (5,340 files of earlier experiments) still matches;
  (2) every fle final/milestone checkpoint listed in fle checkpoint_ledger.tsv (kind=milestone, under runs/) matches;
  (3) fle cache chunks match their manifests; fle report / final_analysis / frozen / prereg hashes recorded.
  .venv/bin/python fcu_integrity.py
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402


def main():
    guard.check_root()
    from csr.fcu_lib import ART, FLE_ART
    R = guard.ROOT
    out = {"time": guard.now()}
    inv = json.loads((FLE_ART / "protected_inventory_before.json").read_text())
    bad = [p for p, v in inv["files"].items() if not (R / p).exists() or guard.sha256_file(R / p) != v["sha256"]]
    out["protected_inventory"] = {"n_files": len(inv["files"]), "n_bad": len(bad), "bad": bad[:20]}
    rows = list(csv.DictReader(open(FLE_ART / "checkpoint_ledger.tsv"), delimiter="\t"))
    ck = [r for r in rows if r["kind"] == "milestone" and r["path"].startswith("runs/")]
    badc = []
    for r in ck:
        p = R / r["path"] / "state.pt"
        if not p.exists() or guard.sha256_file(p) != r["sha256"]:
            badc.append(r["path"])
    out["fle_milestone_checkpoints"] = {"n": len(ck), "n_bad": len(badc), "bad": badc[:20]}
    cb = []
    nchunk = 0
    for man_p in sorted((FLE_ART / "cache").glob("*/manifest.json")):
        man = json.loads(man_p.read_text())
        for c in man["chunks"]:
            nchunk += 1
            if guard.sha256_file(man_p.parent / c["file"]) != c["sha256"]:
                cb.append(str(man_p.parent / c["file"]))
    out["fle_cache_chunks"] = {"n": nchunk, "n_bad": len(cb), "bad": cb}
    out["fle_hashes"] = {p: guard.sha256_file(R / p) for p in
                         ("FORECAST_LATENT_EDL_V1_REPORT.md", "artifacts/forecast-latent-edl-v1/final_analysis.json",
                          "artifacts/forecast-latent-edl-v1/frozen.json", "docs/FORECAST_LATENT_EDL_V1_PREREG.md")}
    out["all_ok"] = not (bad or badc or cb)
    p = ART / f"integrity_{guard.utc_stamp()}.json"
    guard.atomic_write_json(p, out)
    print(json.dumps({k: v for k, v in out.items() if k != "fle_hashes"}, indent=1))


if __name__ == "__main__":
    main()
