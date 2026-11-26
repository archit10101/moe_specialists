"""Deterministic, versioned, causal chunk segmentation of tokenized documents.

All strategies return chunk start positions (sorted, starting at 0) for one document of length L.
Causality contract: whether position t+1 starts a new chunk depends only on tokens <= t.

  fixed     : starts every `size` tokens.
  semantic  : a sentence ending at token t (decided from token t's own decoded text) is embedded by a
              frozen encoder using only tokens of that sentence (<= t). If the current chunk length
              (t+1-a) >= min_len and cosine(sentence, running mean of the chunk's completed sentence
              embeddings) < tau, a new chunk starts at t+1. The triggering sentence remains in the old chunk
              (detection delay) and seeds the new chunk's representation. A chunk reaching max_len is
              force-closed. No boundary is placed after the document's last token.
  random    : chunk lengths drawn i.i.d. from the empirical non-final semantic chunk-length distribution
              (fit on training documents only, stratified by document-length bin), seeded per document
              key, never conditioned on the document's text.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict

import numpy as np

SEGMENTER_VERSION = "csr-seg-v1"
SENT_PUNCT = (".", "!", "?")
SENT_TRAIL = "\"')]”’»"


def stable_hash64(s: str) -> int:
    return int.from_bytes(hashlib.sha256(s.encode("utf-8")).digest()[:8], "little")


# ----------------------------- sentence detection -----------------------------
class SentenceDetector:
    """Per-token-id flag: token's decoded text ends a sentence (terminal punctuation, optionally followed by
    closing quotes/brackets, or contains a newline). Depends only on the token itself, hence causal."""

    def __init__(self, decode_one, vocab_size: int, min_sentence_tokens: int = 4):
        flags = np.zeros(vocab_size, dtype=bool)
        for i in range(vocab_size):
            try:
                s = decode_one(i)
            except Exception:
                s = ""
            flags[i] = self.text_ends_sentence(s)
        self.flags = flags
        self.min_sentence_tokens = min_sentence_tokens

    @staticmethod
    def text_ends_sentence(s: str) -> bool:
        if "\n" in s:
            return True
        t = s.rstrip(" \t").rstrip(SENT_TRAIL)
        return len(t) > 0 and t.endswith(SENT_PUNCT)

    def sentence_ends(self, ids: np.ndarray) -> np.ndarray:
        """Positions t ending a sentence; a sentence must have >= min_sentence_tokens tokens."""
        cand = np.nonzero(self.flags[np.asarray(ids, dtype=np.int64)])[0]
        out, prev = [], -1
        for t in cand:
            if t - prev >= self.min_sentence_tokens:
                out.append(int(t))
                prev = t
        return np.asarray(out, dtype=np.int64)


def sentence_spans(ends: np.ndarray):
    """[start, end] inclusive spans: sentence i covers (ends[i-1], ends[i]]."""
    spans, prev = [], -1
    for e in ends:
        spans.append((prev + 1, int(e)))
        prev = int(e)
    return spans


# ----------------------------- strategies -----------------------------
def fixed_starts(L: int, size: int = 128):
    return list(range(0, L, size))


@dataclass
class SemanticParams:
    tau: float
    min_len: int = 64
    max_len: int = 256


def semantic_starts(L: int, ends: np.ndarray, embs: np.ndarray, p: SemanticParams, return_events=False):
    """Batch form of the online decision rule. embs[i] is the L2-normalized embedding of the sentence
    ending at ends[i]. Shares `_semantic_update` with the streaming chunker."""
    st = _SemState()
    starts = [0]
    events = []
    for t_end, e in zip(ends.tolist(), embs):
        _semantic_update(st, starts, int(t_end), e, L, p, events)
    while L - st.a > p.max_len:
        st.a += p.max_len
        starts.append(st.a)
        events.append(("forced", st.a))
    return (starts, events) if return_events else starts


class _SemState:
    __slots__ = ("a", "rep", "n")

    def __init__(self):
        self.a, self.rep, self.n = 0, None, 0


def _semantic_update(st, starts, t_end, e, L, p, events):
    while t_end + 1 - st.a > p.max_len:
        st.a += p.max_len
        starts.append(st.a)
        st.rep, st.n = None, 0
        events.append(("forced", st.a))
    if t_end + 1 >= L:
        return
    cur_len = t_end + 1 - st.a
    if st.n > 0 and cur_len >= p.min_len:
        r = st.rep / max(np.linalg.norm(st.rep), 1e-12)
        sim = float(np.dot(e, r))
        if sim < p.tau:
            st.a = t_end + 1
            starts.append(st.a)
            st.rep, st.n = np.array(e, dtype=np.float64), 1
            events.append(("semantic", st.a, sim))
            return
    st.rep = np.array(e, dtype=np.float64) if st.rep is None else st.rep + e
    st.n += 1


class OnlineSemanticChunker:
    """Streaming form: push tokens one at a time; `push` returns True iff the NEXT token starts a chunk.
    `embed_fn(list_of_token_id_lists) -> (n, D) normalized np array` sees only observed sentence tokens.
    `L` (document length) is needed only to avoid a boundary after the final token; pass None while
    streaming open-ended text (then a boundary after the last pushed token may be reported)."""

    def __init__(self, detector: SentenceDetector, embed_fn, p: SemanticParams, L=None):
        self.det, self.embed_fn, self.p, self.L = detector, embed_fn, p, L
        self.ids, self.st, self.starts, self.events = [], _SemState(), [0], []
        self.prev_end = -1

    def push(self, tok: int) -> bool:
        t = len(self.ids)
        self.ids.append(int(tok))
        n_before = len(self.starts)
        is_end = bool(self.det.flags[tok]) and (t - self.prev_end >= self.det.min_sentence_tokens)
        if is_end:
            sent = self.ids[self.prev_end + 1: t + 1]
            self.prev_end = t
            e = self.embed_fn([sent])[0]
            L = self.L if self.L is not None else 1 << 60
            _semantic_update(self.st, self.starts, t, e, L, self.p, self.events)
        # forced boundary for the next token if the chunk is full
        if t + 1 - self.st.a >= self.p.max_len and (self.L is None or t + 1 < self.L):
            self.st.a = t + 1
            self.starts.append(self.st.a)
            self.st.rep, self.st.n = None, 0
            self.events.append(("forced", self.st.a))
        return len(self.starts) > n_before and self.starts[-1] == t + 1


# ----------------------------- random length-matched -----------------------------
@dataclass
class LengthDistribution:
    bin_edges: list  # document-length strata edges (len(bins)+1)
    lengths: list  # per-stratum sorted list of observed non-final chunk lengths
    min_len: int
    max_len: int

    def stratum(self, L):
        return int(np.clip(np.searchsorted(self.bin_edges, L, side="right") - 1, 0, len(self.lengths) - 1))

    def to_json(self):
        return json.dumps(asdict(self), sort_keys=True)


def fit_length_distribution(doc_lengths, doc_starts, n_bins=4, min_len=64, max_len=256):
    """Fit on TRAINING documents' semantic chunks only. Non-final chunks (the last chunk is truncated by the
    document end, which random sampling reproduces mechanically)."""
    doc_lengths = np.asarray(doc_lengths)
    edges = np.quantile(doc_lengths, np.linspace(0, 1, n_bins + 1)).astype(float).tolist()
    edges[0], edges[-1] = 0.0, float("inf")
    per = [[] for _ in range(n_bins)]
    tmp = LengthDistribution(edges, per, min_len, max_len)
    for L, st in zip(doc_lengths.tolist(), doc_starts):
        ls = np.diff(np.asarray(list(st) + [L]))[:-1]
        per[tmp.stratum(L)].extend(int(x) for x in ls)
    pooled = sorted(x for b in per for x in b)
    lengths = [sorted(b) if len(b) >= 50 else pooled for b in per]  # fall back to pooled for sparse strata
    assert pooled, "no non-final semantic chunks to fit"
    assert min(pooled) >= 1 and max(pooled) <= max_len
    return LengthDistribution(edges, lengths, min_len, max_len)


def random_starts(L: int, doc_key: str, dist: LengthDistribution, seed: int):
    rng = np.random.default_rng([seed, stable_hash64(doc_key) & 0x7FFFFFFF, stable_hash64(doc_key) >> 33])
    pool = dist.lengths[dist.stratum(L)]
    starts, a = [0], 0
    while True:
        a += int(pool[int(rng.integers(len(pool)))])
        if a >= L:
            break
        starts.append(a)
    return starts


# ----------------------------- shared helpers -----------------------------
def starts_to_arrays(L: int, starts):
    """chunk index and position-in-chunk per token."""
    s = np.asarray(starts, dtype=np.int64)
    assert s[0] == 0 and np.all(np.diff(s) > 0) and s[-1] < L, (s[:5], L)
    cidx = np.zeros(L, dtype=np.int64)
    cidx[s[1:]] = 1
    cidx = np.cumsum(cidx)
    cpos = np.arange(L) - s[cidx]
    return cidx, cpos


def chunk_lengths(L, starts):
    return np.diff(np.asarray(list(starts) + [L]))


# ----------------------------- random hazard closer (condition F, v2) -----------------------------
# NOTE: `random_starts` / `LengthDistribution` above are the SUPERSEDED F-v1 generator (it resampled the
# document-censored semantic non-final distribution and censored it again; non-final mean 122). Do not use for F.
RANDOM_HAZARD_VERSION = "csr-random-hazard-v2"


@dataclass
class HazardPolicy:
    """Closure probability at an ELIGIBLE completed-sentence boundary as a function of document-length stratum and
    current chunk length (bins of `len_bin` tokens over [min_len, max_len)). At cur_len == max_len no random
    decision is taken (the forced-close rule applies, exactly as in the semantic chunker)."""
    strata_edges: list
    len_bin: int
    min_len: int
    max_len: int
    hazard: list  # [n_strata][n_len_bins]

    def stratum(self, L):
        return int(np.clip(np.searchsorted(self.strata_edges, L, side="right") - 1, 0, len(self.hazard) - 1))

    def h(self, stratum, cur_len):
        if cur_len >= self.max_len:
            return 0.0
        return float(self.hazard[stratum][(cur_len - self.min_len) // self.len_bin])

    def to_json(self):
        return json.dumps({"version": RANDOM_HAZARD_VERSION, **asdict(self)}, sort_keys=True)

    @classmethod
    def from_json(cls, s):
        d = json.loads(s)
        d.pop("version", None)
        return cls(**d)


def hazard_starts(L: int, ends, policy: HazardPolicy, doc_key: str, seed: int, decide=None, return_events=False):
    """Random closures with the SAME eligibility state machine as `semantic_starts`/`_semantic_update`:
    completed-sentence ends only; no closure after the document's last token; closure eligible only if the chunk
    already holds >= 1 completed sentence (reset to 0 by a forced close, seeded to 1 by a closure) and
    cur_len >= min_len; force-close at max_len. Inputs: document length L (-> stratum), sentence-end positions,
    the frozen policy, the document key and seed. No text, embeddings or similarities are accepted.
    `decide(stratum, cur_len, t_end) -> bool` replaces the random draw (used only for tests and for fitting)."""
    h = stable_hash64(doc_key)
    rng = np.random.default_rng([seed, h & 0x7FFFFFFF, h >> 33, 0x46])
    s = policy.stratum(L)
    a, n = 0, 0
    starts, events = [0], []
    for t_end in np.asarray(ends).tolist():
        while t_end + 1 - a > policy.max_len:
            a += policy.max_len
            starts.append(a)
            n = 0
            events.append(("forced", a))
        if t_end + 1 >= L:
            break
        cur_len = t_end + 1 - a
        if n > 0 and cur_len >= policy.min_len:
            if decide is not None:
                close = bool(decide(s, cur_len, t_end))
            else:
                close = rng.random() < policy.h(s, cur_len)
            if close:
                a = t_end + 1
                starts.append(a)
                n = 1
                events.append(("random", a))
                continue
        n += 1
    while L - a > policy.max_len:
        a += policy.max_len
        starts.append(a)
        events.append(("forced", a))
    return (starts, events) if return_events else starts
