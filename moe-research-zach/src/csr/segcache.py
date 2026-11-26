"""Deterministic boundary caches and frozen chunk-prefix embeddings, keyed by a content hash of all inputs."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .segment import (SEGMENTER_VERSION, SentenceDetector, SemanticParams, chunk_lengths, fixed_starts,
                      random_starts, semantic_starts, sentence_spans, LengthDistribution)

STRATEGIES = ("none", "fixed", "semantic", "random")


def make_detector(tok, vocab_size=50304):
    n = tok.get_vocab_size()
    return SentenceDetector(lambda i: tok.decode([i], skip_special_tokens=False) if i < n else "", vocab_size)


def sentence_texts(tok, ids, ends):
    return [tok.decode(ids[s:e + 1].tolist()).strip() for s, e in sentence_spans(ends)]


def embed_doc_sentences(tok, det, enc, docs):
    """Returns per doc (ends, embs). Embeddings computed in one batched pass."""
    all_ends, texts, owners = [], [], []
    for d, ids in enumerate(docs):
        e = det.sentence_ends(ids)
        all_ends.append(e)
        texts.extend(sentence_texts(tok, ids, e))
        owners.extend([d] * len(e))
    embs = enc.embed(texts) if texts else np.zeros((0, enc.dim), np.float32)
    out, k = [], 0
    for e in all_ends:
        out.append((e, embs[k:k + len(e)]))
        k += len(e)
    return out


def calibrate_tau(doc_lengths, sent, target=128.0, min_len=64, max_len=256, grid=None):
    """Training-data-only calibration: tau whose mean NON-FINAL semantic chunk length is closest to target."""
    grid = np.round(np.arange(-0.20, 0.951, 0.01), 2) if grid is None else grid
    table = []
    for tau in grid:
        p = SemanticParams(float(tau), min_len, max_len)
        ls = []
        for L, (e, x) in zip(doc_lengths, sent):
            ls.extend(chunk_lengths(L, semantic_starts(L, e, x, p))[:-1].tolist())
        table.append((float(tau), float(np.mean(ls)) if ls else float("nan"), len(ls)))
    ok = [t for t in table if np.isfinite(t[1])]
    best = min(ok, key=lambda t: abs(t[1] - target))
    return best[0], table


def prefix_hash(ids):
    return int.from_bytes(hashlib.sha256(np.asarray(ids, np.int64).tobytes()).digest()[:8], "little") & ((1 << 63) - 1)


@dataclass
class SegTable:
    strategy: str
    starts_flat: np.ndarray
    starts_off: np.ndarray  # (n_docs+1)
    u: np.ndarray | None = None  # (n_chunks, D) float16 prefix embeddings; zero rows for chunks shorter than w
    phash: np.ndarray | None = None  # (n_chunks,) int64

    def starts(self, di):
        return self.starts_flat[self.starts_off[di]:self.starts_off[di + 1]].tolist()

    def chunk_base(self, di):
        return int(self.starts_off[di])

    def save(self, path: Path):
        tmp = path.with_name("." + path.name + ".tmp.npz")
        kw = dict(starts_flat=self.starts_flat, starts_off=self.starts_off)
        if self.u is not None:
            kw.update(u=self.u, phash=self.phash)
        np.savez(tmp, **kw)
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path, strategy):
        z = np.load(path)
        return cls(strategy, z["starts_flat"], z["starts_off"], z["u"] if "u" in z else None,
                   z["phash"] if "phash" in z else None)


def build_table(strategy, split, starts_list):
    flat = np.concatenate([np.asarray(s, np.int32) for s in starts_list]) if starts_list else np.zeros(0, np.int32)
    off = np.concatenate([[0], np.cumsum([len(s) for s in starts_list])]).astype(np.int64)
    return SegTable(strategy, flat, off)


def add_prefix_embeddings(table: SegTable, split, tok, enc, w):
    texts, rows, hashes = [], [], np.zeros(len(table.starts_flat), np.int64)
    for di in range(split.n_docs):
        ids = split.doc(di)
        st = table.starts(di)
        ls = chunk_lengths(len(ids), st)
        for j, (a, L) in enumerate(zip(st, ls)):
            r = table.chunk_base(di) + j
            if L >= w:
                pre = ids[a:a + w]
                texts.append(tok.decode(pre.tolist()).strip())
                rows.append(r)
                hashes[r] = prefix_hash(pre)
    emb = enc.embed(texts) if texts else np.zeros((0, enc.dim), np.float32)
    u = np.zeros((len(table.starts_flat), enc.dim), np.float16)
    u[rows] = emb.astype(np.float16)
    table.u, table.phash = u, hashes
    return len(texts)


def cache_id(spec: dict):
    return hashlib.sha256(json.dumps({**spec, "segmenter": SEGMENTER_VERSION}, sort_keys=True).encode()).hexdigest()[:16]


def timed(fn, *a, **k):
    t0 = time.time()
    r = fn(*a, **k)
    return r, time.time() - t0
