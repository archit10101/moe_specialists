#!/usr/bin/env python
"""specialization-audit-v1 §4: each condition's native, frozen segmentation applied to the audit documents.

fixed 128 (B, C); semantic (D, E): frozen teacher all-MiniLM-L6-v2 + same sentence detector / embed_docs_batched /
semantic_starts with the cached tau; random (F): frozen F-v3 policy, application seed 1, doc key = document id.
Provenance check first: the same code must reproduce the frozen fwe-rep-v1 val semantic and random tables on the
first 300 val documents (segmentation inputs only; no model, no results). GPU via scripts/gpu_guarded.sh."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402

SEG = guard.ROOT / "artifacts" / "segments" / "8e7edc7e14b3c880"
RND = guard.ROOT / "artifacts" / "segments_random" / "0c57f77e51ea3734"
DATA = guard.ROOT / "artifacts" / "specialization-audit-v1" / "data"
OUT = guard.ROOT / "artifacts" / "specialization-audit-v1" / "segments"


def segment_docs(docs, keys, tok, det, enc, p, pol, seed):
    from csr.rep import embed_docs_batched, hazard_starts_v3
    from csr.segment import fixed_starts, semantic_starts
    sem, rnd, fix, ends_all = [], [], [], []
    for b0 in range(0, len(docs), 1000):
        chunk = docs[b0:b0 + 1000]
        for ids, key, (e, x) in zip(chunk, keys[b0:b0 + 1000], embed_docs_batched(tok, det, enc, chunk)):
            L = len(ids)
            sem.append(semantic_starts(L, e, x, p))
            rnd.append(hazard_starts_v3(L, e, pol, key, seed))
            fix.append(fixed_starts(L, 128))
            ends_all.append(e)
    return fix, sem, rnd, ends_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    print(f"audit native segmentation -> {OUT}")
    if not guard.require_execution(a.execute, "gpu" if a.device == "cuda" else "cpu-model"):
        return
    if OUT.exists():
        raise guard.GuardError(f"{OUT} exists")
    guard.require_disk(200_000_000, "specaudit segments")
    mon = None
    if a.device == "cuda":
        guard.occupancy_gate("spec_audit_segments")
        guard.setup_cuda_device()
        mon = guard.OccupancyMonitor(10.0)
        mon.start()
    guard.register_self(f"spec_audit_segments ({a.device})")
    t_iso, t0 = guard.now(), time.time()
    status, reason = "INCOMPLETE", ""
    try:
        import numpy as np
        import torch
        torch.set_num_threads(8)
        from csr.artifacts import FrozenEncoder, load_lm_tokenizer
        from csr.data import TokenSplit
        from csr.rep import HazardPolicyV3
        from csr.segcache import SegTable, build_table, make_detector
        from csr.segment import SemanticParams, chunk_lengths
        sman = json.loads((SEG / "manifest.json").read_text())
        rman = json.loads((RND / "manifest.json").read_text())
        assert rman["accepted"] and rman["spec"]["parent_semantic_cache"] == sman["cache_id"]
        pol = HazardPolicyV3.from_json((RND / "policy.json").read_text())
        seed = int(rman["spec"]["seed"])
        p = SemanticParams(float(sman["tau"]), sman["spec"]["min_len"], sman["spec"]["max_len"])
        tok = load_lm_tokenizer()
        det = make_detector(tok)
        enc = FrozenEncoder("cuda:0" if a.device == "cuda" else "cpu", batch_size=512)
        ck = enc.param_checksum()
        # provenance: reproduce frozen val tables on the first 300 val documents
        val = TokenSplit.load(guard.ROOT / "artifacts" / "data" / "fwe-rep-v1", "val")
        n = 300
        vdocs = [val.doc(i) for i in range(n)]
        _, vsem, vrnd, _ = segment_docs(vdocs, val.doc_ids[:n], tok, det, enc, p, pol, seed)
        ts, tr = SegTable.load(SEG / "val_semantic.npz", "semantic"), SegTable.load(RND / "val_random.npz", "random")
        mis_s = sum(vsem[i] != ts.starts(i) for i in range(n))
        mis_r = sum(vrnd[i] != tr.starts(i) for i in range(n))
        print(f"provenance mismatches semantic={mis_s} random={mis_r} of {n}", flush=True)
        if mis_s or mis_r:
            raise guard.GuardError(f"native segmentation does not reproduce frozen val tables ({mis_s}, {mis_r})")
        sp = TokenSplit.load(DATA, "audit")
        docs = [sp.doc(i) for i in range(sp.n_docs)]
        fix, sem, rnd, ends = segment_docs(docs, sp.doc_ids, tok, det, enc, p, pol, seed)
        if mon is not None and mon.contaminated:
            raise guard.GuardError(f"CONTAMINATED: {mon.detail}")
        assert enc.param_checksum() == ck
        tmp = OUT.with_name("." + OUT.name + ".tmp")
        tmp.mkdir(parents=True, exist_ok=False)
        summ = {}
        L = sp.lengths()
        for name, st in (("fixed", fix), ("semantic", sem), ("random", rnd)):
            build_table(name, sp, st).save(tmp / f"audit_{name}.npz")
            ls = np.concatenate([chunk_lengths(int(l), s) for l, s in zip(L, st)])
            nonf = np.concatenate([chunk_lengths(int(l), s)[:-1] for l, s in zip(L, st)])
            summ[name] = {"n_chunks": int(len(ls)), "mean_len": float(ls.mean()),
                          "mean_nonfinal_len": float(nonf.mean()) if len(nonf) else None}
        off = np.concatenate([[0], np.cumsum([len(e) for e in ends])]).astype(np.int64)
        np.savez(tmp / "audit_ends.npz", ends_flat=np.concatenate(ends).astype(np.int32), ends_off=off)
        man = {"audit": "specialization-audit-v1", "role": "native model inputs (not labels)",
               "semantic_cache": sman["cache_id"], "tau": sman["tau"], "random_cache": rman["random_cache_id"],
               "random_seed": seed, "teacher": sman["spec"]["encoder"], "encoder_param_checksum": ck,
               "provenance_val_first300": {"semantic_mismatch": mis_s, "random_mismatch": mis_r},
               "chunk_summary": summ, "seconds": time.time() - t0, "created": guard.now(), "device": a.device}
        man["sha256"] = {f.name: guard.sha256_file(f) for f in sorted(tmp.iterdir())}
        (tmp / "manifest.json").write_text(json.dumps(man, indent=2, sort_keys=True))
        for f in tmp.iterdir():
            f.chmod(0o400)
        tmp.rename(OUT)
        guard.ledger("SPEC_AUDIT_SEGMENTS", f"artifacts/specialization-audit-v1/segments {json.dumps(summ)}")
        print(json.dumps(man, indent=1))
        status = "COMPLETE"
    except guard.GuardError as e:
        status, reason = ("CONTAMINATED" if "CONTAMINATED" in str(e) else "ABORTED"), str(e)
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        status, reason = "FAILED", f"{type(e).__name__}: {e}"
    finally:
        if mon is not None:
            mon.stop()
            if mon.contaminated and status == "COMPLETE":
                status, reason = "CONTAMINATED", mon.detail
            guard.gpu_ledger(t_iso, guard.now(), "specaudit_v1_segments", "spec_audit_segments", status)
        print("STATUS", status, reason)
    sys.exit(0 if status == "COMPLETE" else 1)


if __name__ == "__main__":
    main()
