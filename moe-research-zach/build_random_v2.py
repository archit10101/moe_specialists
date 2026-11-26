#!/usr/bin/env python
"""Condition F v2: fit the random closure hazard on TRAINING documents (matching D's realized, document-end-
censored chunk process), freeze it, and generate random boundary caches for train / calib / val only.

Reads (never writes) the semantic cache. Publishes artifacts/segments_random/<id>/ ONLY if the train-split
acceptance criteria pass (fail closed); calib / val comparisons are reported as held-out checks.
CPU only; requires --execute and MOE_RESEARCH_APPROVE=cpu-model.
"""
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
    ap.add_argument("--segments", default="artifacts/segments/611d129276301aee")
    ap.add_argument("--data", default="artifacts/data/fwe-v1")
    ap.add_argument("--len-bin", type=int, default=8)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7, help="frozen application seed")
    ap.add_argument("--from-rejected", help="publish EXACTLY the policy recorded in this rejected report (no refit)")
    ap.add_argument("--user-exempt", action="append", default=[],
                    help="acceptance criterion exempted by explicit user decision (recorded in the manifest)")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    sd, dd = guard.inside_root(a.segments), guard.inside_root(a.data)
    print(f"fit F-v2 hazard on train from {sd}; generate {SPLITS} (test never read)")
    if not guard.require_execution(a.execute, "cpu-model"):
        return
    guard.register_self("build_random_v2 (CPU)")
    import numpy as np
    from csr.artifacts import load_lm_tokenizer
    from csr.data import TokenSplit
    from csr.random_hazard import ACCEPT, compare, describe, fit_hazard, scripted_replay, simulate
    from csr.segcache import SegTable, build_table, cache_id, make_detector
    from csr.segment import RANDOM_HAZARD_VERSION

    t0 = time.time()
    sman = json.loads((sd / "manifest.json").read_text())
    edges = json.loads((sd / "random_length_distribution.json").read_text())["bin_edges"]  # train-length quartiles
    det = make_detector(load_lm_tokenizer())
    sp, docs, dst = {}, {}, {}
    for s in SPLITS:
        sp[s] = TokenSplit.load(dd, s)
        docs[s] = [(int(len(sp[s].doc(i))), det.sentence_ends(sp[s].doc(i)), sp[s].doc_ids[i]) for i in range(sp[s].n_docs)]
        t = SegTable.load(sd / f"{s}_semantic.npz", "semantic")
        dst[s] = [t.starts(i) for i in range(sp[s].n_docs)]
    t_prep = time.time() - t0
    if a.from_rejected:
        from csr.segment import HazardPolicy
        rej = json.loads(guard.inside_root(a.from_rejected).read_text())
        pd = dict(rej["spec"]["policy"]); pd.pop("version", None)
        pol, hist = HazardPolicy(**pd), rej["fit_history"] + [{"frozen_from": a.from_rejected}]
        assert rej["spec"]["seed"] == a.seed and rej["spec"]["parent_semantic_cache"] == sman["cache_id"]
    else:
        pol, hist = fit_hazard(docs["train"], dst["train"], edges, a.len_bin, sman["spec"]["min_len"],
                               sman["spec"]["max_len"], a.iters, log=lambda r: print(r, flush=True))
    # exact-replay check on every split: D's boundaries are reproducible by the shared eligibility state machine
    replay_mismatch = {s: int(sum(scripted_replay(L, e, st, pol)[0] != list(st) for (L, e, _), st in zip(docs[s], dst[s])))
                       for s in SPLITS}
    report, f_starts = {}, {}
    for s in SPLITS:
        f_starts[s] = simulate(docs[s], pol, a.seed)
        Ls = [d[0] for d in docs[s]]
        D, F = describe(Ls, dst[s], sman["spec"]["warmup"]), describe(Ls, f_starts[s], sman["spec"]["warmup"])
        spread = [compare(D, describe(Ls, simulate(docs[s], pol, sd_), sman["spec"]["warmup"])) for sd_ in range(31, 39)]
        report[s] = {"D": {k: v for k, v in D.items() if not k.startswith("_")},
                     "F": {k: v for k, v in F.items() if not k.startswith("_")}, "compare": compare(D, F),
                     "compare_over_8_other_seeds": {k: {"mean": float(np.mean([c[k] for c in spread])),
                                                        "min": float(np.min([c[k] for c in spread])),
                                                        "max": float(np.max([c[k] for c in spread]))}
                                                    for k in ACCEPT}}
        print(s, json.dumps(report[s]["compare"]), flush=True)
    other_seeds = {sd_: compare(describe([d[0] for d in docs["train"]], dst["train"]),
                                describe([d[0] for d in docs["train"]], simulate(docs["train"], pol, sd_)))["all_pass"]
                   for sd_ in (11, 12, 13)}
    tr_pass = report["train"]["compare"]["pass"]
    exempt = sorted(set(a.user_exempt))
    assert all(k in ACCEPT for k in exempt), exempt
    ok = all(v for k, v in tr_pass.items() if k not in exempt) and all(v == 0 for v in replay_mismatch.values())
    if a.from_rejected:  # the regenerated boundaries must reproduce the recorded candidate exactly
        rec = rej["report"]["train"]["compare"]
        same = all(abs(rec[k] - report["train"]["compare"][k]) < 1e-12 for k in ACCEPT)
        ok = ok and same
        print("reproduces recorded candidate metrics:", same)
    spec = {"version": RANDOM_HAZARD_VERSION, "parent_semantic_cache": sman["cache_id"], "policy": json.loads(pol.to_json()),
            "seed": a.seed, "splits": list(SPLITS)}
    rid = cache_id(spec)
    man = {"random_cache_id": rid, "spec": spec, "fit_history": hist, "replay_mismatch_docs": replay_mismatch,
           "acceptance_thresholds": ACCEPT, "acceptance_split": "train", "report": report,
           "train_all_pass_other_seeds": other_seeds, "accepted": ok,
           "acceptance_mode": ("all criteria" if not exempt else
                               "all criteria except user-exempted: " + ", ".join(exempt)),
           "user_exemptions": {k: {"value_train": report["train"]["compare"][k], "threshold": ACCEPT[k],
                                   "decision": "user: 'Accept with documented gap' (2026-10-31); cause: D's "
                                               "closure rate rises near document end, not reproducible from "
                                               "chunk length + stratum"} for k in exempt},
           "timings_s": {"prep": t_prep, "total": time.time() - t0}, "created": guard.now(),
           "inputs_used_by_policy": ["document length (stratum)", "sentence-end positions", "doc key", "seed",
                                     "current chunk length"]}
    if not ok:
        rej = guard.ROOT / ".tmp" / f"random_v2_REJECTED_{guard.utc_stamp()}.json"
        guard.atomic_write_json(rej, man)
        guard.ledger("RANDOM_V2_REJECTED", str(rej.relative_to(guard.ROOT)))
        print("NOT ACCEPTED; nothing published:", rej)
        return
    out = guard.ROOT / "artifacts" / "segments_random" / rid
    if out.exists():
        print("exists", out)
        return
    tmp = out.with_name("." + out.name + ".tmp")
    tmp.mkdir(parents=True)
    (tmp / "policy.json").write_text(pol.to_json())
    for s in SPLITS:
        build_table("random", sp[s], f_starts[s]).save(tmp / f"{s}_random.npz")
    (tmp / "manifest.json").write_text(json.dumps(man, indent=2, sort_keys=True, default=float))
    for f in tmp.iterdir():
        f.chmod(0o400)
    tmp.rename(out)
    guard.ledger("RANDOM_V2_BUILD", f"artifacts/segments_random/{rid} accepted=train-criteria; seed={a.seed}")
    print("published", out)


if __name__ == "__main__":
    main()
