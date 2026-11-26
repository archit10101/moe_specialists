#!/usr/bin/env python
"""csr-crossfit-semantic-utility-residual-v1: inventory of reused (read-only) artifacts with sha256 verification.
Writes ART/inventory.json (0400). Nothing outside ART is written."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402


def main():
    from csr.csur1_paths import ART, ORV1, REUSED, SEM_NET, SEM_NET_SHA, SEM_RES, SEM_RES_SHA, T4_CFG_RUN
    out = {"time": guard.now(), "checkpoints": {}, "tables": {}, "encoder": {}, "notes": []}
    for kind, d in REUSED.items():
        for s, rd in d.items():
            st = json.loads((guard.ROOT / rd / "status.json").read_text())
            ck = st["ckpts"]["p100"]
            h = guard.sha256_file(guard.ROOT / ck["path"] / "state.pt")
            out["checkpoints"][f"{kind}_s{s}"] = {"run": rd, "status": st["status"], "seen": st.get("seen"),
                                                  "p100": ck["path"], "sha256_ok": h == ck["sha256"], "sha256": h}
    for f in ["tables/labels_main.npz", "tables/labels_cs.npz", "tables/parts.npz", "tables/token_feats.npz",
              "val_assign/labels_val.npz", "val_assign/parts_val.npz"]:
        out["tables"][f"orv1/{f}"] = guard.sha256_file(ORV1 / f)
    va = json.loads((ORV1 / "val_assign/val_assign.json").read_text())
    out["tables"]["orv1_val_assign_recorded"] = {k: va[k] for k in va if k.startswith("sha256_")}
    out["tables"]["orv1_val_assign_match"] = all(
        guard.sha256_file(ORV1 / "val_assign" / k[len("sha256_"):]) == v for k, v in va.items() if k.startswith("sha256_"))
    out["tables"]["jsr/tables.json"] = guard.sha256_file(guard.ROOT / "artifacts/joint-semantic-residual-pretrain-v1/tables/tables.json")
    out["encoder"] = {"sem_net": SEM_NET, "sem_net_ok": guard.sha256_file(guard.ROOT / SEM_NET) == SEM_NET_SHA,
                      "residualizer": SEM_RES, "residualizer_ok": guard.sha256_file(guard.ROOT / SEM_RES) == SEM_RES_SHA,
                      "bge": "artifacts/encoder/bge-small-en-v1.5"}
    out["t4_config_run"] = T4_CFG_RUN
    out["reused_results"] = {
        "ORACLE_RESIDUAL_V1": "frozen-base paper-level semantic oracle bank (A), shuffled (B), token-routed (C) at 50M; "
                              "reference only (different setting: frozen 245M base, LR 3e-4, 2048-token rows)",
        "JOINT_SEMANTIC_RESIDUAL_PRETRAIN_V1": "joint-pretraining residual banks at 50M on seeds 0/1 (EVAL); reference only",
        "TOKEN4/EDL 245M": "re-evaluated on VAL from the hash-verified p100 checkpoints (no retraining)"}
    out["not_reusable"] = {
        "generic residual at 245M": "no 245M joint generic-residual run exists (jsr stopped at 50M); trained in-process (G)",
        "semantic-only block-routed": "orv1 A is paper-level, frozen-base and non-causal; trained in-process (SO)"}
    out["all_ok"] = all(v["sha256_ok"] for v in out["checkpoints"].values()) and out["encoder"]["sem_net_ok"] and \
        out["encoder"]["residualizer_ok"] and out["tables"]["orv1_val_assign_match"]
    guard.atomic_write_json(ART / "inventory.json", out, mode=0o400)
    print(json.dumps({"all_ok": out["all_ok"], "n_ckpts": len(out["checkpoints"])}))


if __name__ == "__main__":
    main()
