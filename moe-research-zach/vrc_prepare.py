#!/usr/bin/env python
"""csr-value-routed-compute-v1: re-tokenise the existing fwe-rep-v1 train / calib documents at a long max_len
(docs/VALUE_ROUTED_COMPUTE_V1_PROTOCOL.md). CPU only.

The document sets are exactly the ids of artifacts/data/fwe-rep-v1/{train,calib}_docs.json. Rows of the raw parquet
are filtered by id inside Arrow (pyarrow.compute.is_in) before any text is converted to Python, so the text of
validation / locked-test documents is never materialised, decoded into Python, tokenised or inspected.
  vrc_prepare.py --max-len 8192 --execute
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402

SRC = guard.ROOT / "artifacts/data/fwe-rep-v1"
RAWS = [guard.ROOT / "artifacts/data/raw/fwe_10BT_000_rep_v1_rg240_seed20261101.parquet",
        guard.ROOT / "artifacts/data/raw/fwe_10BT_000_rg16_seed20261031.parquet"]
TRAIN_CAP = 24_000_000
ORDER_SEED = 17
PARTITION_SALT = "csr-vrc-v1-calib-partition"


def texts_for(ids):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    want = pa.array(sorted(set(ids)))
    found = {}
    for raw in RAWS:
        t = pq.read_table(raw, columns=["id", "text"])
        t = t.filter(pc.is_in(t["id"], value_set=want))  # filtered in Arrow before Python conversion
        for i, x in zip(t["id"].to_pylist(), t["text"].to_pylist()):
            found.setdefault(i, x)
    missing = [i for i in ids if i not in found]
    if missing:
        raise guard.GuardError(f"{len(missing)} ids not found in raw parquet")
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-len", type=int, required=True)
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    if not guard.require_execution(a.execute, "cpu-model"):
        return
    out = guard.ROOT / f"artifacts/value-routed-compute-v1/data/fwe-rep-v1-L{a.max_len}"
    if out.exists():
        raise guard.GuardError(f"{out} exists; refusing to overwrite")
    guard.require_disk(1024**3, "vrc data build")
    from csr.artifacts import load_lm_tokenizer
    from csr.data import EOS_ID, train_order
    tok = load_lm_tokenizer()
    t0 = time.time()
    tr_ids = json.loads((SRC / "train_docs.json").read_text())["ids"]
    ca_ids = json.loads((SRC / "calib_docs.json").read_text())["ids"]
    order = train_order(len(tr_ids), ORDER_SEED)
    # tokenise train docs in train order until the cap (fetch texts for a generous prefix of the order)
    out.mkdir(parents=True)
    meta = {}
    for split, ids_in in (("train", [tr_ids[i] for i in order]), ("calib", ca_ids)):
        take = ids_in if split == "calib" else ids_in[:60000]
        texts = texts_for(take)
        arrs, ids, n_valid, n_trunc = [], [], 0, 0
        for s in range(0, len(take), 512):
            chunk = take[s:s + 512]
            for i, e in zip(chunk, tok.encode_batch([texts[i] for i in chunk])):
                if split == "train" and n_valid >= TRAIN_CAP:
                    break
                x = e.ids
                n_trunc += len(x) > a.max_len - 1
                arr = np.asarray(x[:a.max_len - 1] + [EOS_ID], dtype=np.uint16)
                arrs.append(arr)
                ids.append(i)
                n_valid += len(arr) - 1
            if split == "train" and n_valid >= TRAIN_CAP:
                break
        if split == "train" and n_valid < TRAIN_CAP:
            raise guard.GuardError("train cap not reached")
        tk = np.concatenate(arrs)
        off = np.concatenate([[0], np.cumsum([len(x) for x in arrs])]).astype(np.int64)
        np.save(out / f"{split}_tokens.npy", tk)
        np.save(out / f"{split}_offsets.npy", off)
        (out / f"{split}_docs.json").write_text(json.dumps({"ids": ids}))
        L = np.diff(off)
        meta[split] = {"n_docs": len(ids), "n_tokens": int(tk.size), "valid_tokens": int(n_valid),
                       "truncated_docs": int(n_trunc), "docs_ge_512": int((L > 512).sum()),
                       "tokens_at_pos_ge_512": int(np.clip(L - 512, 0, None).sum()),
                       "tokens_at_pos_ge_1024": int(np.clip(L - 1024, 0, None).sum()),
                       "max_len_doc": int(L.max()),
                       "sha256_tokens": guard.sha256_file(out / f"{split}_tokens.npy"),
                       "sha256_offsets": guard.sha256_file(out / f"{split}_offsets.npy"),
                       "sha256_docs": guard.sha256_file(out / f"{split}_docs.json")}
    # calib partition (document-hash; every hash recorded via the index lists)
    search, selection = [], []
    for i, d in enumerate(ca_ids):
        u = int.from_bytes(hashlib.sha256(f"{PARTITION_SALT}|{d}".encode()).digest()[:8], "little") / 2**64
        (search if u < 0.70 else selection).append(i)
    part = {"salt": PARTITION_SALT, "search_fraction": 0.70, "search_doc_indices": search,
            "selection_doc_indices": selection,
            "search_indices_sha256": hashlib.sha256(json.dumps(search).encode()).hexdigest(),
            "selection_indices_sha256": hashlib.sha256(json.dumps(selection).encode()).hexdigest(),
            "calib_docs_sha256": hashlib.sha256(json.dumps(ca_ids).encode()).hexdigest()}
    guard.atomic_write_json(out / "calib_partition.json", part)
    man = {"experiment": "csr-value-routed-compute-v1", "created": guard.now(), "source": str(SRC),
           "raw": [str(r) for r in RAWS], "raw_sha256": [guard.sha256_file(r) for r in RAWS], "max_len": a.max_len,
           "doc_format": "tokens[:max_len-1] + [EOS=0]", "train_rule": f"fwe-rep-v1 train ids in train_order(seed="
           f"{ORDER_SEED}) until {TRAIN_CAP} valid tokens (train split stored in that order)",
           "test_policy": "validation / test ids never selected; raw rows filtered in Arrow before Python conversion",
           "splits": meta, "build_seconds": time.time() - t0}
    man["splits_sha256"] = guard.sha256_json({k: {kk: v[kk] for kk in ("sha256_tokens", "sha256_offsets",
                                                                       "sha256_docs")} for k, v in meta.items()})
    guard.atomic_write_json(out / "manifest.json", man)
    for p in out.iterdir():
        p.chmod(0o400)
    guard.ledger("VRC_DATA_BUILD", f"{out}: {json.dumps({k: v['valid_tokens'] for k, v in meta.items()})}")
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
