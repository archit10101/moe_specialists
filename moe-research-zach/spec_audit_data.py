#!/usr/bin/env python
"""specialization-audit-v1 Stage 2: new audit-only documents (docs/SPECIALIZATION_AUDIT_V1_PREREG.md §2-3).

  download : 12 unused row groups of the pinned FineWeb-Edu file (range reads).   MOE_RESEARCH_APPROVE=download
  build    : tokenize -> dedup vs model-visible text of all non-test splits (fwe-rep-v1, fwe-v1) -> eligibility ->
             8,192 seeded documents -> windows, halves -> artifacts/specialization-audit-v1/data/.
             MOE_RESEARCH_APPROVE=cpu-model
Locked test splits are never opened: test_* files are not read, hashed or stat'ed; source-level disjointness is
guaranteed by excluding all 256 previously used row groups. Dry-run by default.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402
from csr.data import DATASET_FILE, DATASET_LICENSE, DATASET_REPO, DATASET_REV, TOKENIZER_REPO, TOKENIZER_REV  # noqa: E402
from csr.rep import OLD_ROW_GROUPS  # noqa: E402
from csr import specaudit as sa  # noqa: E402

ART = guard.ROOT / "artifacts"
RAW = ART / "data" / "raw" / f"fwe_10BT_000_specaudit_v1_rg{sa.N_ROW_GROUPS}_seed{sa.SEED}.parquet"
OUT = ART / "specialization-audit-v1" / "data"
REF_SPLITS = [("fwe-rep-v1", s) for s in ("train", "calib", "val")] + [("fwe-v1", s) for s in ("train", "calib", "val")]


def used_row_groups():
    rep = json.loads((ART / "download_manifest_rep_v1_20261101T033146Z.json").read_text())["row_groups"]
    old = json.loads((ART / "download_manifest_20261031T212112Z.json").read_text())["row_groups"]
    assert sorted(old) == sorted(OLD_ROW_GROUPS)
    return sorted(set(rep) | set(old))


def cmd_download(a):
    print(f"download {sa.N_ROW_GROUPS} unused row groups of {DATASET_REPO}@{DATASET_REV}/{DATASET_FILE} -> {RAW}")
    if not guard.require_execution(a.execute, "download"):
        return
    if RAW.exists():
        raise guard.GuardError(f"{RAW} exists; never overwritten")
    guard.require_disk(200_000_000, "specaudit download")
    guard.register_self("spec_audit_data download")
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    used = used_row_groups()
    t0 = time.time()
    p = f"datasets/{DATASET_REPO}@{DATASET_REV}/{DATASET_FILE}"
    guard.ledger("NETWORK", f"specaudit: HF range reads {p} (pinned revision), {sa.N_ROW_GROUPS} row groups")
    tmp = RAW.with_name("." + RAW.name + ".tmp")
    with HfFileSystem().open(p, "rb", block_size=1 << 20) as f:
        pf = pq.ParquetFile(f)
        n_total = pf.metadata.num_row_groups
        rgs, n_elig = sa.choose_audit_row_groups(n_total, used)
        assert not set(rgs) & set(used)
        tbl = pf.read_row_groups(rgs, columns=["id", "text", "url", "token_count", "score"])
        tbl = tbl.append_column("row_group", pa.array(np.repeat(rgs, [pf.metadata.row_group(r).num_rows for r in rgs])))
    pq.write_table(tbl, tmp, compression="zstd")
    os.chmod(tmp, 0o400)
    tmp.replace(RAW)
    rec = {"time": guard.now(), "seconds": time.time() - t0, "file_row_groups_total": n_total,
           "excluded_row_groups": used, "n_eligible_row_groups": n_elig, "row_groups": rgs, "rng_seed": sa.SEED,
           "rows": tbl.num_rows, "files": {str(RAW.relative_to(guard.ROOT)): {"sha256": guard.sha256_file(RAW),
                                                                            "bytes": RAW.stat().st_size}},
           "dataset": {"repo": DATASET_REPO, "revision": DATASET_REV, "file": DATASET_FILE, "license": DATASET_LICENSE}}
    guard.atomic_write_json(ART / f"download_manifest_specaudit_v1_{guard.utc_stamp()}.json", rec)
    guard.ledger("DOWNLOAD", f"specaudit-v1: row groups {rgs}; rows={tbl.num_rows}; bytes={RAW.stat().st_size}")
    print(json.dumps({k: v for k, v in rec.items() if k != "excluded_row_groups"}))


def cmd_build(a):
    print(f"build {RAW} -> {OUT}")
    if not guard.require_execution(a.execute, "cpu-model"):
        return
    if OUT.exists():
        raise guard.GuardError(f"{OUT} exists; never overwritten")
    guard.require_disk(500_000_000, "specaudit build")
    guard.register_self("spec_audit_data build")
    import numpy as np
    import pyarrow.parquet as pq
    from csr.artifacts import load_lm_tokenizer
    from csr.data import TokenSplit, tokenize_docs
    from csr.rep import dedup_against_reference
    t0 = time.time()
    tok = load_lm_tokenizer()
    raw = pq.read_table(RAW, columns=["id", "text", "url", "row_group"])
    ids, texts = raw.column("id").to_pylist(), raw.column("text").to_pylist()
    urls, rgs = raw.column("url").to_pylist(), raw.column("row_group").to_pylist()
    assert len(set(ids)) == len(ids), "duplicate ids inside the audit raw file"
    docs, trunc = tokenize_docs(tok, texts, 1024)
    vis = tok.decode_batch([d[:-1].tolist() for d in docs])
    ref_texts, ref_ids, ref_counts = [], set(), {}
    for ds, sp in REF_SPLITS:
        assert sp != "test"
        s = TokenSplit.load(ART / "data" / ds, sp)
        dec = tok.decode_batch([s.doc(i)[:-1].tolist() for i in range(s.n_docs)])
        ref_texts += dec
        ref_ids.update(s.doc_ids)
        ref_counts[f"{ds}/{sp}"] = s.n_docs
        print(ds, sp, s.n_docs, f"{time.time() - t0:.0f}s", flush=True)
    id_overlap = sorted(set(ids) & ref_ids)
    if id_overlap:
        raise guard.GuardError(f"audit ids present in reference splits: {id_overlap[:5]}")
    keep, rep = dedup_against_reference(ref_texts, ids, vis, processes=a.processes)
    rep["reference_splits"] = ref_counts
    rep["reference_text"] = "model-visible text decode(tokens[:1023]) of every non-test document"
    rep["id_overlap_with_references"] = 0
    rep["test_content_checked"] = False
    del ref_texts
    L = np.array([len(d) for d in docs])
    elig = sorted(ids[i] for i in range(len(ids)) if keep[i] and L[i] >= 65)
    chosen = set(sa.select_documents(elig))
    sel = sorted((i for i in range(len(ids)) if ids[i] in chosen), key=lambda i: ids[i])
    assert len(sel) == sa.N_AUDIT_DOCS
    tmp = OUT.with_name("." + OUT.name + ".tmp")
    tmp.mkdir(parents=True, exist_ok=False)
    lens = L[sel]
    offs = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    toks = np.concatenate([docs[i] for i in sel]).astype(np.uint16)
    np.save(tmp / "audit_tokens.npy", toks)
    np.save(tmp / "audit_offsets.npy", offs)
    sel_ids = [ids[i] for i in sel]
    (tmp / "audit_docs.json").write_text(json.dumps({"ids": sel_ids, "urls": [urls[i] for i in sel],
                                                      "row_groups": [int(rgs[i]) for i in sel],
                                                      "truncated": [bool(trunc[i]) for i in sel]}))
    qedges = sa.doc_length_quartile_edges(lens)
    W = {k: [] for k in ("doc", "win_index", "start", "end", "half", "doc_len", "stratum")}
    for d, (did, Ld) in enumerate(zip(sel_ids, lens)):
        h = sa.half_of(did)
        for wi, s0, e0 in sa.select_windows(int(Ld)):
            W["doc"].append(d)
            W["win_index"].append(wi)
            W["start"].append(s0)
            W["end"].append(e0)
            W["half"].append(h)
            W["doc_len"].append(int(Ld))
    W = {k: np.asarray(v, np.int64) for k, v in W.items() if v}
    W["stratum"] = sa.window_strata(W["doc_len"], W["end"] - W["start"], qedges)
    np.savez(tmp / "windows.npz", **W)
    half_docs = np.array([sa.half_of(i) for i in sel_ids])
    wl = W["end"] - W["start"]
    man = {"audit": "specialization-audit-v1", "prereg_sha256": (guard.ROOT / "docs" / "SPECIALIZATION_AUDIT_V1_PREREG.sha256").read_text().split()[0],
           "raw": str(RAW.relative_to(guard.ROOT)), "raw_sha256": guard.sha256_file(RAW),
           "dataset": {"repo": DATASET_REPO, "revision": DATASET_REV, "file": DATASET_FILE, "license": DATASET_LICENSE},
           "tokenizer": {"repo": TOKENIZER_REPO, "revision": TOKENIZER_REV}, "doc_format": "tokens[:1023] + [EOS=0]",
           "source_row_groups": sorted(set(int(r) for r in rgs)), "n_raw": len(ids), "dedup": rep,
           "n_kept": int(keep.sum()), "n_eligible": len(elig), "eligibility": "kept and L >= 65",
           "selection": f"default_rng({sa.SEED}).permutation over sorted eligible ids; first {sa.N_AUDIT_DOCS}",
           "n_docs": len(sel), "n_tokens": int(offs[-1]), "n_pred_tokens": int(offs[-1] - len(sel)),
           "truncated_docs": int(sum(trunc[i] for i in sel)), "doc_length_quartile_edges": qedges,
           "halves": {"discovery_docs": int((half_docs == 0).sum()), "effect_docs": int((half_docs == 1).sum()),
                      "rule": f"sha256('{sa.HALF_SALT}'+id)[:8]/2^64 < 0.5 -> discovery"},
           "windows": {"n": int(len(W["doc"])), "discovery": int((W["half"] == 0).sum()),
                       "effect": int((W["half"] == 1).sum()), "full_128": int((wl == 128).sum()),
                       "partial": int((wl < 128).sum()), "mask_tokens": int(wl.sum()),
                       "mask_tokens_discovery": int(wl[W["half"] == 0].sum()),
                       "mask_tokens_effect": int(wl[W["half"] == 1].sum()),
                       "per_doc_hist": np.bincount(np.bincount(W["doc"], minlength=len(sel))).tolist()},
           "locked_test_policy": "test_* files of fwe-rep-v1 and fwe-v1 never opened; source rows disjoint by row group",
           "seconds": time.time() - t0, "created": guard.now()}
    for f in ("audit_tokens.npy", "audit_offsets.npy", "audit_docs.json", "windows.npz"):
        man.setdefault("sha256", {})[f] = guard.sha256_file(tmp / f)
    (tmp / "manifest.json").write_text(json.dumps(man, indent=2, sort_keys=True))
    for f in tmp.iterdir():
        f.chmod(0o400)
    tmp.rename(OUT)
    guard.ledger("SPEC_AUDIT_DATA", f"artifacts/specialization-audit-v1/data docs={len(sel)} windows={len(W['doc'])} "
                                    f"mask_tokens={int(wl.sum())} dedup_kept={int(keep.sum())}/{len(ids)}")
    print(json.dumps({k: v for k, v in man.items() if k not in ("dedup",)}, indent=1))
    print(json.dumps(rep))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["download", "build"])
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--processes", type=int, default=6)
    a = ap.parse_args()
    guard.check_root()
    {"download": cmd_download, "build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    main()
