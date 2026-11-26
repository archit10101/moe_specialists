#!/usr/bin/env python
"""csr-representative-v1 condition F-v3 (docs/REP_V1_PREREG.md §3): fit the content-free closure hazard on TRAINING
documents, then rerandomize: application seeds 0..99 in order, freeze the FIRST seed whose realized F passes every
acceptance criterion on BOTH train and calib. Validation is never consulted for acceptance (val tables are generated
with the frozen seed; their statistics are descriptive). Test never read. CPU; MOE_RESEARCH_APPROVE=cpu-model."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402

SPLITS = ("train", "calib", "val")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments", required=True)
    ap.add_argument("--data", default="artifacts/data/fwe-rep-v1")
    ap.add_argument("--max-seed", type=int, default=99)
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    sd, dd = guard.inside_root(a.segments), guard.inside_root(a.data)
    print(f"F-v3 fit on train from {sd}; acceptance on train+calib; generate {SPLITS}")
    if not guard.require_execution(a.execute, "cpu-model"):
        return
    guard.register_self("rep_build_random (CPU)")
    import numpy as np
    from csr.data import TokenSplit
    from csr.rep import (ACCEPT_V3, RANDOM_HAZARD_V3, compare_v3, describe_v3, fit_hazard_v3, replay_v3,
                         simulate_v3)
    from csr.segcache import SegTable, build_table, cache_id

    t0 = time.time()
    sman = json.loads((sd / "manifest.json").read_text())
    assert sman["spec"]["experiment"] == "csr-representative-v1"
    sp, docs, dst = {}, {}, {}
    for s in SPLITS:
        sp[s] = TokenSplit.load(dd, s)
        z = np.load(sd / f"{s}_ends.npz")
        ef, eo = z["ends_flat"], z["ends_off"]
        L = sp[s].lengths()
        docs[s] = [(int(L[i]), ef[eo[i]:eo[i + 1]].astype(np.int64), sp[s].doc_ids[i]) for i in range(sp[s].n_docs)]
        t = SegTable.load(sd / f"{s}_semantic.npz", "semantic")
        dst[s] = [t.starts(i) for i in range(sp[s].n_docs)]
    Ltr = np.array([d[0] for d in docs["train"]])
    edges = [0.0] + np.quantile(Ltr, [0.25, 0.5, 0.75]).tolist() + [float(Ltr.max() + 1)]
    pol, hist, mism_fit = fit_hazard_v3(docs["train"], dst["train"], edges, log=lambda r: print(r, flush=True))
    t_fit = time.time() - t0
    replay_mismatch = {s: int(sum(replay_v3(L, e, st, pol)[0] != list(st) for (L, e, _), st in zip(docs[s], dst[s])))
                       for s in ("train", "calib")}
    print("replay mismatch", replay_mismatch, flush=True)
    Ds = {s: describe_v3([d[0] for d in docs[s]], dst[s]) for s in SPLITS}
    tried, chosen, f_starts = [], None, {}
    if all(v == 0 for v in replay_mismatch.values()):
        for seed in range(a.max_seed + 1):
            rec = {"seed": seed}
            ok = True
            cand = {}
            for s in ("train", "calib"):
                cand[s] = simulate_v3(docs[s], pol, seed)
                c = compare_v3(Ds[s], describe_v3([d[0] for d in docs[s]], cand[s]))
                rec[s] = {k: v for k, v in c.items()}
                ok = ok and c["all_pass"]
                if not ok:
                    break
            tried.append(rec)
            print(json.dumps(rec, default=float), flush=True)
            if ok:
                chosen, f_starts = seed, cand
                break
    accepted = chosen is not None
    report = {}
    if accepted:
        f_starts["val"] = simulate_v3(docs["val"], pol, chosen)
        for s in SPLITS:
            F = describe_v3([d[0] for d in docs[s]], f_starts[s])
            report[s] = {"D": {k: v for k, v in Ds[s].items() if not k.startswith("_")},
                         "F": {k: v for k, v in F.items() if not k.startswith("_")},
                         "compare": compare_v3(Ds[s], F),
                         "role": "acceptance" if s != "val" else "descriptive only (not used for acceptance)"}
    spec = {"version": RANDOM_HAZARD_V3, "parent_semantic_cache": sman["cache_id"], "policy": json.loads(pol.to_json()),
            "seed": chosen, "splits": list(SPLITS), "rerandomization": f"seeds 0..{a.max_seed}, first passing train+calib"}
    rid = cache_id(spec)
    man = {"random_cache_id": rid, "spec": spec, "fit_history": hist, "fit_replay_mismatch_train": mism_fit,
           "replay_mismatch_docs": replay_mismatch, "acceptance_thresholds": ACCEPT_V3,
           "acceptance_splits": ["train", "calib"], "seeds_tried": tried, "accepted": accepted, "report": report,
           "inputs_used_by_policy": ["document length (stratum, remaining distance)", "sentence-end positions",
                                     "doc key", "seed", "current chunk length"],
           "timings_s": {"fit": t_fit, "total": time.time() - t0}, "created": guard.now(), "test_read": False}
    if not accepted:
        rej = guard.ROOT / ".tmp" / f"rep_random_v3_REJECTED_{guard.utc_stamp()}.json"
        guard.atomic_write_json(rej, man)
        guard.ledger("RANDOM_V3_REJECTED", str(rej.relative_to(guard.ROOT)))
        print("NOT ACCEPTED; nothing published:", rej)
        sys.exit(1)
    out = guard.ROOT / "artifacts" / "segments_random" / rid
    if out.exists():
        raise guard.GuardError(f"exists {out}")
    tmp = out.with_name("." + out.name + ".tmp")
    tmp.mkdir(parents=True)
    (tmp / "policy.json").write_text(pol.to_json())
    for s in SPLITS:
        build_table("random", sp[s], f_starts[s]).save(tmp / f"{s}_random.npz")
    (tmp / "manifest.json").write_text(json.dumps(man, indent=2, sort_keys=True, default=float))
    for f in tmp.iterdir():
        f.chmod(0o400)
    tmp.rename(out)
    guard.ledger("RANDOM_V3_BUILD", f"artifacts/segments_random/{rid} accepted on train+calib; seed={chosen}")
    print("published", out, "seed", chosen)


if __name__ == "__main__":
    main()
