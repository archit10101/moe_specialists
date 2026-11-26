#!/usr/bin/env python
"""scrgp-v1: verify the reused frozen Semantic C6 p100 checkpoints (sha256 vs status.json, step/tokens, C6 condition and
table hash) -> artifacts/scrgp-v1/reuse_manifest.json (0400)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402

C6_TABLE_SHA = "5d23129b2cb7aed6a0d2ed768b1a755eac661dae95ddeace89329dc7b56b89d2"


def main():
    guard.check_root()
    from csr.scrgp_paths import ART, C6_RUNS
    out = ART / "reuse_manifest.json"
    if out.exists():
        print("exists")
        return
    rec, ok_all = {}, True
    for s, rd in C6_RUNS.items():
        st = json.loads((guard.ROOT / rd / "status.json").read_text())
        cfg = json.loads((guard.ROOT / rd / "config.json").read_text())
        ck = st["ckpts"]["p100"]
        h = guard.sha256_file(guard.ROOT / ck["path"] / "state.pt")
        pl = cfg["cond"].get("paper_latent", {})
        r = {"run_dir": rd, "status": st["status"], "seen": st["seen"], "step": ck["step"], "ckpt": ck["path"],
             "sha_recorded": ck["sha256"], "sha_recomputed": h, "seed": cfg["seed"], "cond": cfg["cond"],
             "checks": {"sha": h == ck["sha256"], "complete": st["status"] == "COMPLETE",
                        "tokens": st["seen"] == 245032457 and ck["step"] == 3880, "seed": cfg["seed"] == s,
                        "c6": pl.get("table_sha256") == C6_TABLE_SHA and pl.get("beta") == 0.005}}
        r["ok"] = all(r["checks"].values())
        ok_all &= r["ok"]
        rec[f"C6_s{s}"] = r
    guard.atomic_write_json(out, {"time": guard.now(), "all_ok": ok_all, "runs": rec}, mode=0o400)
    print(json.dumps({"all_ok": ok_all, **{k: v["checks"] for k, v in rec.items()}}, indent=1))
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
