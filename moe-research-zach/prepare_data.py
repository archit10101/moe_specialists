#!/usr/bin/env python
"""Download pinned small artifacts and build the de-duplicated, document-level split dataset.

  download : tokenizer.json (Pythia), frozen MiniLM encoder, N FineWeb-Edu row groups (range reads).
             requires --execute and MOE_RESEARCH_APPROVE=download
  build    : dedup -> split -> tokenize -> artifacts/data/<version>/ + manifest (CPU only).
             requires --execute and MOE_RESEARCH_APPROVE=cpu-model
Defaults to dry-run.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402
from csr.data import (DATASET_FILE, DATASET_LICENSE, DATASET_REPO, DATASET_REV, SPLIT_SALT, SPLITS, TOKENIZER_REPO,
                      TOKENIZER_REV, dedup, download_subset, split_of, tokenize_docs)  # noqa: E402

ART = guard.ROOT / "artifacts"
RAW = ART / "data" / "raw"
DATA_VERSION = "fwe-v1"


def cmd_download(a):
    plan = {"tokenizer": f"{TOKENIZER_REPO}@{TOKENIZER_REV} tokenizer.json (~2.1 MB)",
            "encoder": "sentence-transformers/all-MiniLM-L6-v2@1110a243 (~91.6 MB)",
            "dataset": f"{DATASET_REPO}@{DATASET_REV}/{DATASET_FILE}: {a.row_groups} row groups (~3 MB each)",
            "estimated_bytes": 100_000_000 + a.row_groups * 4_000_000}
    print(json.dumps(plan, indent=2))
    if not guard.require_execution(a.execute, "download"):
        return
    guard.require_disk(plan["estimated_bytes"] * 2, "prepare_data download")
    guard.register_self("prepare_data download")
    from csr.artifacts import (ENCODER_DIR, ENCODER_FILES, ENCODER_REPO, ENCODER_REV, TOKENIZER_DIR,
                               download_pinned)
    t0 = time.time()
    out = {}
    if not (TOKENIZER_DIR / "tokenizer.json").exists():
        download_pinned(TOKENIZER_REPO, TOKENIZER_REV, ["tokenizer.json"], TOKENIZER_DIR)
    if not (ENCODER_DIR / "model.safetensors").exists():
        download_pinned(ENCODER_REPO, ENCODER_REV, ENCODER_FILES, ENCODER_DIR)
    RAW.mkdir(parents=True, exist_ok=True)
    raw_path = RAW / f"fwe_10BT_000_rg{a.row_groups}_seed{a.seed}.parquet"
    if raw_path.exists():
        print(f"exists, not re-downloading: {raw_path}")
        rgs = None
    else:
        rgs, n = download_subset(raw_path, a.row_groups, a.seed)
        print(f"downloaded {n} rows from row groups {rgs}")
    for f in [TOKENIZER_DIR / "tokenizer.json"] + [ENCODER_DIR / x for x in ENCODER_FILES] + [raw_path]:
        out[str(f.relative_to(guard.ROOT))] = {"sha256": guard.sha256_file(f), "bytes": f.stat().st_size}
    rec = {"time": guard.now(), "seconds": time.time() - t0, "row_groups": rgs, "seed": a.seed, "files": out,
           "dataset": {"repo": DATASET_REPO, "revision": DATASET_REV, "file": DATASET_FILE, "license": DATASET_LICENSE}}
    guard.atomic_write_json(ART / f"download_manifest_{guard.utc_stamp()}.json", rec)
    guard.ledger("DOWNLOAD", f"tokenizer+encoder+{a.row_groups} row groups; files={len(out)}; "
                             f"bytes={sum(v['bytes'] for v in out.values())}")


def cmd_build(a):
    raw_path = RAW / f"fwe_10BT_000_rg{a.row_groups}_seed{a.seed}.parquet"
    out_dir = ART / "data" / DATA_VERSION
    print(f"build {raw_path} -> {out_dir} (max_len={a.max_len})")
    if not guard.require_execution(a.execute, "cpu-model"):
        return
    if out_dir.exists():
        raise guard.GuardError(f"{out_dir} exists; never overwritten (bump DATA_VERSION)")
    guard.require_disk(200_000_000, "prepare_data build")
    guard.register_self("prepare_data build")
    import numpy as np
    import pyarrow.parquet as pq
    from csr.artifacts import load_lm_tokenizer
    t0 = time.time()
    tbl = pq.read_table(raw_path)
    ids, texts = tbl.column("id").to_pylist(), tbl.column("text").to_pylist()
    urls = tbl.column("url").to_pylist()
    keep, rep = dedup(ids, texts)
    print(json.dumps(rep))
    tok = load_lm_tokenizer()
    tmp = out_dir.with_name("." + out_dir.name + ".tmp")
    tmp.mkdir(parents=True, exist_ok=False)
    split_info = {}
    order = sorted(range(len(ids)), key=lambda i: ids[i])
    for sp in SPLITS:
        sel = [i for i in order if keep[i] and split_of(ids[i]) == sp]
        docs, trunc = tokenize_docs(tok, [texts[i] for i in sel], a.max_len)
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
    man = {"data_version": DATA_VERSION, "raw": str(raw_path.relative_to(guard.ROOT)),
           "raw_sha256": guard.sha256_file(raw_path), "dataset": {"repo": DATASET_REPO, "revision": DATASET_REV,
                                                                 "file": DATASET_FILE, "license": DATASET_LICENSE},
           "tokenizer": {"repo": TOKENIZER_REPO, "revision": TOKENIZER_REV}, "max_len": a.max_len,
           "doc_format": "tokens[:max_len-1] + [EOS=0]", "split_rule": f"sha256('{SPLIT_SALT}'+id) buckets {SPLITS}",
           "dedup": rep, "splits": split_info, "build_seconds": time.time() - t0, "created": guard.now(),
           "test_access_policy": "test split is never read by train/calibrate/probe; evaluate.py requires --unlock-test "
                                 "plus MOE_RESEARCH_TEST_UNLOCK=1, not authorized in Stages A/B"}
    (tmp / "manifest.json").write_text(json.dumps(man, indent=2, sort_keys=True))
    for f in tmp.iterdir():
        f.chmod(0o400)
    tmp.rename(out_dir)
    guard.ledger("DATA_BUILD", f"{out_dir.relative_to(guard.ROOT)} " + json.dumps({k: v['n_docs'] for k, v in split_info.items()}))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if not kk.startswith('sha')} for k, v in split_info.items()}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["download", "build"])
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--row-groups", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20261031)
    ap.add_argument("--max-len", type=int, default=1024)
    a = ap.parse_args()
    guard.check_root()
    {"download": cmd_download, "build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    main()
