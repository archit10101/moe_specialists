"""Dataset subset, de-duplication, document-level splits, tokenization, packing and batch assembly.

Packing is condition-independent: identical rows, document order, valid targets and token counts for
every condition. Segmentation affects only chunk_ids / chunk_pos.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATASET_REPO = "HuggingFaceFW/fineweb-edu"
DATASET_REV = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
DATASET_FILE = "sample/10BT/000_00000.parquet"
DATASET_LICENSE = "odc-by"
TOKENIZER_REPO = "EleutherAI/pythia-12b"
TOKENIZER_REV = "bb1e3e710cdf6b524461d543cfb5ba773f0a81b6"
EOS_ID = 0  # <|endoftext|>
PAD_ID = 1  # <|padding|>
SPLIT_SALT = "csr-split-v1"
SPLITS = {"train": (0.0, 0.80), "calib": (0.80, 0.85), "val": (0.85, 0.925), "test": (0.925, 1.0)}


# ------------------------------------------------------------------ download (range reads of row groups)
def choose_row_groups(n_total: int, n: int, seed: int):
    rng = np.random.default_rng(seed)
    return sorted(int(x) for x in rng.choice(n_total, size=n, replace=False))


def download_subset(out_path: Path, n_row_groups: int, seed: int):
    from huggingface_hub import HfFileSystem
    import pyarrow as pa
    import pyarrow.parquet as pq
    fs = HfFileSystem()
    p = f"datasets/{DATASET_REPO}@{DATASET_REV}/{DATASET_FILE}"
    with fs.open(p, "rb", block_size=1 << 20) as f:
        pf = pq.ParquetFile(f)
        rgs = choose_row_groups(pf.metadata.num_row_groups, n_row_groups, seed)
        tbl = pf.read_row_groups(rgs, columns=["id", "text", "url", "token_count", "score"])
    tbl = tbl.append_column("row_group", pa.array(np.repeat(rgs, [pf.metadata.row_group(r).num_rows for r in rgs])))
    tmp = out_path.with_name("." + out_path.name + ".tmp")
    pq.write_table(tbl, tmp, compression="zstd")
    tmp.replace(out_path)
    return rgs, tbl.num_rows


# ------------------------------------------------------------------ de-duplication
def normalize_text(t: str) -> str:
    return re.sub(r"\s+", " ", t.lower()).strip()


_MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def minhash_signature(text: str, perms_a, perms_b, shingle=5):
    words = normalize_text(text).split(" ")
    if len(words) < shingle:
        sh = [" ".join(words)]
    else:
        sh = [" ".join(words[i:i + shingle]) for i in range(len(words) - shingle + 1)]
    hv = np.fromiter((int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little") for s in set(sh)),
                     dtype=np.uint64)
    with np.errstate(over="ignore"):
        x = hv[None, :] * perms_a[:, None] + perms_b[:, None]  # wraps mod 2^64
    return x.min(axis=1)


def dedup(ids, texts, n_perm=128, bands=32, threshold=0.8, seed=1234):
    """Returns (keep_mask, report). Exact duplicates by normalized-text SHA-256; near duplicates by MinHash
    LSH (5-word shingles, 32x4 bands) with estimated Jaccard >= threshold. Canonical order = sorted ids; the
    later document of a duplicate pair is dropped."""
    order = sorted(range(len(ids)), key=lambda i: ids[i])
    keep = np.ones(len(ids), dtype=bool)
    seen = {}
    n_exact = 0
    for i in order:
        h = hashlib.sha256(normalize_text(texts[i]).encode()).hexdigest()
        if h in seen:
            keep[i] = False
            n_exact += 1
        else:
            seen[h] = i
    rng = np.random.default_rng(seed)
    a = rng.integers(1, 2**63, size=n_perm, dtype=np.uint64) | np.uint64(1)
    b = rng.integers(0, 2**63, size=n_perm, dtype=np.uint64)
    sigs = {i: minhash_signature(texts[i], a, b) for i in order if keep[i]}
    r = n_perm // bands
    buckets = {}
    n_near = 0
    for i in order:
        if not keep[i]:
            continue
        s = sigs[i]
        dup_of = None
        keys = [(bi, s[bi * r:(bi + 1) * r].tobytes()) for bi in range(bands)]
        for key in keys:
            for j in buckets.get(key, ()):
                if np.mean(sigs[j] == s) >= threshold:
                    dup_of = j
                    break
            if dup_of is not None:
                break
        if dup_of is not None:
            keep[i] = False
            n_near += 1
            continue
        for key in keys:
            buckets.setdefault(key, []).append(i)
    return keep, {"n_in": len(ids), "exact_dups": n_exact, "near_dups": n_near, "n_kept": int(keep.sum()),
                  "near_dup_method": f"minhash n_perm={n_perm} bands={bands} 5-word shingles jaccard>={threshold}"}


def split_of(doc_id: str) -> str:
    u = int.from_bytes(hashlib.sha256((SPLIT_SALT + doc_id).encode()).digest()[:8], "little") / 2**64
    for name, (lo, hi) in SPLITS.items():
        if lo <= u < hi:
            return name
    return "test"


# ------------------------------------------------------------------ tokenized split storage
@dataclass
class TokenSplit:
    tokens: np.ndarray  # uint16 concatenated
    offsets: np.ndarray  # int64 (n_docs + 1)
    doc_ids: list

    @property
    def n_docs(self):
        return len(self.offsets) - 1

    def doc(self, i):
        return self.tokens[self.offsets[i]:self.offsets[i + 1]].astype(np.int64)

    def lengths(self):
        return np.diff(self.offsets)

    @classmethod
    def load(cls, d: Path, split: str):
        tok = np.load(d / f"{split}_tokens.npy", mmap_mode="r")
        off = np.load(d / f"{split}_offsets.npy")
        ids = json.loads((d / f"{split}_docs.json").read_text())["ids"]
        return cls(tok, off, ids)


def tokenize_docs(tokenizer, texts, max_len):
    """Each document: tokens truncated to max_len-1, then EOS. Returns list of int arrays, truncated flags."""
    out, trunc = [], []
    for enc in tokenizer.encode_batch(texts):
        ids = enc.ids
        trunc.append(len(ids) > max_len - 1)
        out.append(np.asarray(ids[:max_len - 1] + [EOS_ID], dtype=np.int64))
    return out, trunc


# ------------------------------------------------------------------ packing and batches
def pack_rows(lengths, order, T):
    rows, cur, used = [], [], 0
    for i in order:
        L = int(lengths[i])
        assert L <= T
        if used + L > T:
            rows.append(cur)
            cur, used = [], 0
        cur.append(int(i))
        used += L
    if cur:
        rows.append(cur)
    return rows


def train_order(n_docs, seed):
    return np.random.default_rng(seed).permutation(n_docs).tolist()


def make_batch(rows, split: TokenSplit, T, starts_of=None, chunk_key_of=None):
    """rows: list of lists of doc indices. starts_of(doc_idx) -> chunk starts list (None: one chunk per doc).
    chunk_key_of(doc_idx) -> base key such that chunk j of the doc has key base+j (or None)."""
    B = len(rows)
    a = {k: np.full((B, T), fill, dtype=np.int64) for k, fill in
         (("input_ids", PAD_ID), ("doc_ids", -1), ("positions", 0), ("targets", -100), ("chunk_ids", -1),
          ("chunk_pos", 0), ("gdoc", -1), ("chunk_key", -1))}
    from .segment import starts_to_arrays
    for b, row in enumerate(rows):
        t, cbase = 0, 0
        for local, di in enumerate(row):
            ids = split.doc(di)
            L = len(ids)
            sl = slice(t, t + L)
            a["input_ids"][b, sl] = ids
            a["doc_ids"][b, sl] = local
            a["positions"][b, sl] = np.arange(L)
            a["targets"][b, t:t + L - 1] = ids[1:]
            a["gdoc"][b, sl] = di
            st = starts_of(di) if starts_of is not None else [0]
            cidx, cpos = starts_to_arrays(L, st)
            a["chunk_ids"][b, sl] = cbase + cidx
            a["chunk_pos"][b, sl] = cpos
            if chunk_key_of is not None:
                kb = chunk_key_of(di)
                if kb is not None:
                    a["chunk_key"][b, sl] = kb + cidx
            cbase += len(st)
            t += L
    return a


def to_torch(batch, device):
    import torch
    return {k: torch.as_tensor(v, device=device) for k, v in batch.items()}
