#!/usr/bin/env python
"""csr-representative-v1 fresh data (docs/REP_V1_PREREG.md §2).

  download : 240 pinned FineWeb-Edu row groups (range reads), excluding fwe-v1's 16.   MOE_RESEARCH_APPROVE=download
  build    : dedup (vs fwe-v1 raw + within) -> split -> tokenize -> artifacts/data/fwe-rep-v1/.  MOE_RESEARCH_APPROVE=cpu-model
Dry-run by default. The test split is written by the build but never loaded by any later step.
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
from csr.rep import (MIN_TRAIN_TOKENS, REP_DATA_VERSION, REP_N_ROW_GROUPS, REP_RG_SEED, REP_SPLIT_SALT, REP_SPLITS,  # noqa: E402
                     choose_rep_row_groups)

ART = guard.ROOT / "artifacts"
RAW = ART / "data" / "raw"
RAW_NEW = RAW / f"fwe_10BT_000_rep_v1_rg{REP_N_ROW_GROUPS}_seed{REP_RG_SEED}.parquet"
RAW_OLD = RAW / "fwe_10BT_000_rg16_seed20261031.parquet"


def cmd_download(a):
    print(f"download {REP_N_ROW_GROUPS} row groups of {DATASET_REPO}@{DATASET_REV}/{DATASET_FILE} -> {RAW_NEW}")
    if not guard.require_execution(a.execute, "download"):
        return
    if RAW_NEW.exists():
        raise guard.GuardError(f"{RAW_NEW} exists; never overwritten")
    guard.require_disk(2_000_000_000, "rep download")
    guard.register_self("rep_prepare_data download")
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    t0 = time.time()
    fs = HfFileSystem()
    p = f"datasets/{DATASET_REPO}@{DATASET_REV}/{DATASET_FILE}"
    tmp = RAW_NEW.with_name("." + RAW_NEW.name + ".tmp")
    n_rows = 0
    with fs.open(p, "rb", block_size=1 << 20) as f:
        pf = pq.ParquetFile(f)
        n_total = pf.metadata.num_row_groups
        rgs = choose_rep_row_groups(n_total)
        writer = None
        for i in range(0, len(rgs), 20):
            grp = rgs[i:i + 20]
            tbl = pf.read_row_groups(grp, columns=["id", "text", "url", "token_count", "score"])
            tbl = tbl.append_column("row_group", pa.array(np.repeat(grp, [pf.metadata.row_group(r).num_rows for r in grp])))
            if writer is None:
                writer = pq.ParquetWriter(tmp, tbl.schema, compression="zstd")
            writer.write_table(tbl)
            n_rows += tbl.num_rows
            print(f"row groups {i + len(grp)}/{len(rgs)} rows={n_rows} {time.time() - t0:.0f}s", flush=True)
        writer.close()
    os.chmod(tmp, 0o400)
    tmp.replace(RAW_NEW)
    rec = {"time": guard.now(), "seconds": time.time() - t0, "file_row_groups_total": n_total, "row_groups": rgs,
           "excluded_fwe_v1_row_groups": True, "rng_seed": REP_RG_SEED, "rows": n_rows,
           "files": {str(RAW_NEW.relative_to(guard.ROOT)): {"sha256": guard.sha256_file(RAW_NEW),
                                                           "bytes": RAW_NEW.stat().st_size}},
           "dataset": {"repo": DATASET_REPO, "revision": DATASET_REV, "file": DATASET_FILE, "license": DATASET_LICENSE}}
    guard.atomic_write_json(ART / f"download_manifest_rep_v1_{guard.utc_stamp()}.json", rec)
    guard.ledger("DOWNLOAD", f"rep-v1: {len(rgs)} row groups of {n_total}; rows={n_rows}; bytes={RAW_NEW.stat().st_size}")


def cmd_build(a):
    out_dir = ART / "data" / REP_DATA_VERSION
    print(f"build {RAW_NEW} (+ reference {RAW_OLD}) -> {out_dir}")
    if not guard.require_execution(a.execute, "cpu-model"):
        return
    if out_dir.exists():
        raise guard.GuardError(f"{out_dir} exists; never overwritten")
    guard.require_disk(2_000_000_000, "rep build")
    guard.register_self("rep_prepare_data build")
    import numpy as np
    import pyarrow.parquet as pq
    from csr.artifacts import load_lm_tokenizer
    from csr.data import tokenize_docs
    from csr.rep import dedup_against_reference, rep_split_of
    t0 = time.time()
    old = pq.read_table(RAW_OLD, columns=["id", "text"])
    new = pq.read_table(RAW_NEW, columns=["id", "text", "url", "row_group"])
    ids, texts, urls = new.column("id").to_pylist(), new.column("text").to_pylist(), new.column("url").to_pylist()
    old_ids = set(old.column("id").to_pylist())
    assert not (old_ids & set(ids)), "fwe-v1 document ids present in the new raw file"
    keep, rep = dedup_against_reference(old.column("text").to_pylist(), ids, texts, processes=a.processes)
    rep["id_overlap_with_fwe_v1"] = 0
    t_dedup = time.time() - t0
    print(json.dumps(rep), f"{t_dedup:.0f}s", flush=True)
    tok = load_lm_tokenizer()
    tmp = out_dir.with_name("." + out_dir.name + ".tmp")
    tmp.mkdir(parents=True, exist_ok=False)
    split_info = {}
    order = sorted(range(len(ids)), key=lambda i: ids[i])
    for sp in REP_SPLITS:
        sel = [i for i in order if keep[i] and rep_split_of(ids[i]) == sp]
        docs, trunc = [], []
        for b in range(0, len(sel), 20000):
            d_, t_ = tokenize_docs(tok, [texts[i] for i in sel[b:b + 20000]], a.max_len)
            docs += d_
            trunc += t_
        lens = np.array([len(d) for d in docs], dtype=np.int64)
        offs = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
        toks = np.concatenate(docs).astype(np.uint16) if docs else np.zeros(0, np.uint16)
        assert toks.max() < 65535
        np.save(tmp / f"{sp}_tokens.npy", toks)
        np.save(tmp / f"{sp}_offsets.npy", offs)
        (tmp / f"{sp}_docs.json").write_text(json.dumps({"ids": [ids[i] for i in sel], "urls": [urls[i] for i in sel],
                                                         "truncated": trunc}))
        split_info[sp] = {"n_docs": len(sel), "n_tokens": int(offs[-1]), "valid_pred_tokens": int(offs[-1] - len(sel)),
                          "truncated_docs": int(sum(trunc)),
                          "sha256_tokens": guard.sha256_file(tmp / f"{sp}_tokens.npy"),
                          "sha256_offsets": guard.sha256_file(tmp / f"{sp}_offsets.npy"),
                          "sha256_docs": guard.sha256_file(tmp / f"{sp}_docs.json")}
        print(sp, {k: v for k, v in split_info[sp].items() if not k.startswith("sha")}, flush=True)
    ok = split_info["train"]["valid_pred_tokens"] >= MIN_TRAIN_TOKENS
    du = sum(f.stat().st_size for f in tmp.iterdir())
    man = {"data_version": REP_DATA_VERSION, "experiment": "csr-representative-v1",
           "raw": str(RAW_NEW.relative_to(guard.ROOT)), "raw_sha256": guard.sha256_file(RAW_NEW),
           "reference_raw": str(RAW_OLD.relative_to(guard.ROOT)), "reference_raw_sha256": guard.sha256_file(RAW_OLD),
           "dataset": {"repo": DATASET_REPO, "revision": DATASET_REV, "file": DATASET_FILE, "license": DATASET_LICENSE},
           "tokenizer": {"repo": TOKENIZER_REPO, "revision": TOKENIZER_REV}, "max_len": a.max_len,
           "doc_format": "tokens[:max_len-1] + [EOS=0]", "split_rule": f"sha256('{REP_SPLIT_SALT}'+id) buckets {REP_SPLITS}",
           "dedup": rep, "splits": split_info, "train_min_valid_tokens": MIN_TRAIN_TOKENS, "train_requirement_met": ok,
           "disk_bytes": du, "build_seconds": time.time() - t0, "dedup_seconds": t_dedup, "created": guard.now(),
           "test_access_policy": "test split written by this build only; never loaded by csr-representative-v1"}
    (tmp / "manifest.json").write_text(json.dumps(man, indent=2, sort_keys=True))
    for f in tmp.iterdir():
        f.chmod(0o400)
    tmp.rename(out_dir)
    guard.ledger("DATA_BUILD", f"{out_dir.relative_to(guard.ROOT)} " + json.dumps({k: v['valid_pred_tokens'] for k, v in split_info.items()}))
    if not ok:
        raise guard.GuardError(f"train valid tokens {split_info['train']['valid_pred_tokens']} < {MIN_TRAIN_TOKENS}: STOP")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["download", "build"])
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--processes", type=int, default=6)
    a = ap.parse_args()
    guard.check_root()
    {"download": cmd_download, "build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    main()
