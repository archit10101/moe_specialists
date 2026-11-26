#!/usr/bin/env python
"""joint-semantic-residual-pretrain-v1: frozen data manifest (CPU, read-only on all reused inputs).
Writes ART/tables/tables.json (0400): sha256 of every reused input, plan sha + steps per budget (prefix of the 245M
TOKEN-4 plan), C6-table vs orv1 label agreement, group token masses (semantic / shuffled / old C6 SHUF) on the
base-seen stream, partition sizes."""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402


def main():
    import numpy as np
    from csr.fsp_lib import Split
    from csr.gc_lib import PaperQ
    from csr.jsr_paths import (BUDGETS, C6_TABLE, C6_TABLE_SHA, ORV1, REUSED, T4_CFG_RUN, T4_PLAN_SHA, TABLES)
    from csr.lfr_lib import main_stream
    guard.check_root()
    TABLES.mkdir(parents=True, exist_ok=True)
    out = TABLES / "tables.json"
    if out.exists():
        print("exists")
        return
    files = {"labels_main": ORV1 / "tables/labels_main.npz", "labels_cs": ORV1 / "tables/labels_cs.npz",
             "parts_cs": ORV1 / "tables/parts.npz", "token_feats": ORV1 / "tables/token_feats.npz",
             "labels_val": ORV1 / "val_assign/labels_val.npz", "parts_val": ORV1 / "val_assign/parts_val.npz",
             "c6_table": guard.ROOT / C6_TABLE, "c6_shuf_table_unused": guard.ROOT / "artifacts/semantic-edl-v1/tables/table_SHUF_SEM_T05.npz",
             "t4_config": guard.ROOT / T4_CFG_RUN / "config.json"}
    sha = {k: guard.sha256_file(v) for k, v in files.items()}
    if sha["c6_table"] != C6_TABLE_SHA:
        raise guard.GuardError("C6 table hash mismatch")
    reused = {}
    for kind, d in REUSED.items():
        for s, rp in d.items():
            st = json.loads((guard.ROOT / rp / "status.json").read_text())
            ck = st["ckpts"]["p100"]
            reused[f"{kind}_s{s}"] = {"run": rp, "status": st["status"], "seen": st.get("seen"),
                                      "plan_sha256": st.get("plan_sha256"), "ckpt": ck["path"], "sha256": ck["sha256"]}
    split, steps, planned = main_stream(245_000_000)
    plan_sha = hashlib.sha256(json.dumps(steps).encode()).hexdigest()
    if plan_sha != T4_PLAN_SHA:
        raise guard.GuardError("plan sha mismatch")
    L = split.lengths
    cum = np.cumsum([sum(int(L[d]) - 1 for r in s_ for d in r) for s_ in steps])
    step_of = {k: int(np.searchsorted(cum, b) + 1) for k, b in BUDGETS.items()}
    step_tokens = {k: int(cum[v - 1]) for k, v in step_of.items()}
    z = np.load(files["labels_main"])
    sem, shuf = z["sem"].astype(np.int64), z["shuf"].astype(np.int64)
    pq = PaperQ(files["c6_table"], split, C6_TABLE_SHA)
    has = pq.q_of_doc.sum(1) > 0
    m = has & (sem >= 0)
    agree = float((pq.q_of_doc.argmax(1)[m] == sem[m]).mean())
    if agree != 1.0:
        raise guard.GuardError(f"C6 argmax != orv1 sem ({agree})")
    seen = np.zeros(split.n_docs, bool)
    for s_ in steps:
        for r in s_:
            seen[list(r)] = True
    w = (L - 1).astype(np.float64)
    ps = PaperQ(files["c6_shuf_table_unused"], split)
    c6shuf = np.where(ps.q_of_doc.sum(1) > 0, ps.q_of_doc.argmax(1), -1)

    def mass(lab):
        ok = seen & (lab >= 0)
        return (np.bincount(lab[ok], weights=w[ok], minlength=16) / w[seen].sum()).round(4).tolist()
    res = {"time": guard.now(), "sha256": sha, "reused": reused, "plan_sha256": plan_sha, "n_steps": len(steps),
           "planned_tokens": int(planned), "budget_steps": step_of, "budget_step_tokens": step_tokens,
           "c6_vs_orv1_sem_agreement": agree, "n_labelled_main_docs": int(m.sum()),
           "seen_docs": int(seen.sum()), "seen_unlabelled_token_frac": float(w[seen & (sem < 0)].sum() / w[seen].sum()),
           "mass_sem": mass(sem), "mass_shuf_orv1": mass(shuf), "mass_c6_shuf_unused": mass(c6shuf),
           "papers_per_group_sem": np.bincount(sem[sem >= 0], minlength=16).tolist(),
           "papers_per_group_shuf": np.bincount(shuf[shuf >= 0], minlength=16).tolist()}
    guard.atomic_write_json(out, res, mode=0o400)
    print(json.dumps({k: v for k, v in res.items() if k not in ("reused", "sha256")}, indent=1))


if __name__ == "__main__":
    main()
