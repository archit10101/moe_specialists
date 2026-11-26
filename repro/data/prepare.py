"""Stage `prepare`: download (login node) / build (CPU job) the paper-disjoint token streams.

Output in ``$WORK/data/``:
  {split}.tokens.bin   uint16 token ids, papers concatenated, each followed by the end-of-text token
  {split}.docs.bin     int32 global paper id of every token
  docs.npz             per paper: domain, split, n_tokens
  meta.json            tokenizer, eot id, domain names, token counts per split

Splits (per domain, contiguous ranges of the source order, filled test -> val -> calib -> bank -> base):
  test  locked final evaluation          val    model selection / reported CE
  calib cluster fitting, probes          bank   the 50M "new tokens" for residual banks and routers
  base  the 245M-token base-model training set

Sources:
  m2d2_arxiv  M2D2 arXiv level-1 archives (train.txt + valid.txt). Papers are separated by blank lines
              (one paragraph per line inside a paper), as in the original pipeline.
  jsonl       any file of {"text": ..., "domain": ...} lines (e.g. a corpus with real paper boundaries)
  synthetic   generated pseudo-text with domains and latent sub-topics (smoke tests; no download)
"""
from __future__ import annotations

import codecs
import json
import re
import tarfile
from pathlib import Path

import numpy as np

from ..config import Profile
from ..utils import data_dir, is_done, log, mark_done, write_json

M2D2_REPO = "machelreid/m2d2"
M2D2_REVISION = "6ae613ba744ee56f645aee9c577d04f7e3d48b30"   # pinned dataset commit
MIN_DOC_TOKENS = 64
MAX_DOC_TOKENS = 50_000
SHUFFLE_SEED = 12345


# ============================================================================== tokenizers

class Tokenizer:
    def __init__(self, kind: str):
        self.kind = kind
        if kind == "gpt2":
            import tiktoken
            self.enc = tiktoken.get_encoding("gpt2")
            self.eot, self.n_vocab = self.enc.eot_token, self.enc.n_vocab
        elif kind == "bytes":
            self.eot, self.n_vocab = 256, 257
        else:
            raise ValueError(f"unknown tokenizer {kind!r}")

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        if self.kind == "gpt2":
            return self.enc.encode_ordinary_batch(texts, num_threads=16)
        return [list(t.encode("utf-8")) for t in texts]

    def decode(self, ids) -> str:
        ids = [int(i) for i in ids if int(i) != self.eot]
        if self.kind == "gpt2":
            return self.enc.decode(ids)
        return bytes(i for i in ids if i < 256).decode("utf-8", errors="ignore")


# ============================================================================== sources

def m2d2_archive(domain: str) -> Path:
    """Local path of a domain archive (downloads it if online and not cached)."""
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(M2D2_REPO, f"data/{domain}_l1.tar.gz", repo_type="dataset",
                                revision=M2D2_REVISION))


def m2d2_docs(domain: str, doc_chars: int):
    """Yield papers. M2D2 separates papers with blank lines (runs of >= 2 newlines) and keeps one
    paragraph per line inside a paper; the original pipeline split on exactly that (dfc_data.py
    `_iter_docs`), over train.txt and then valid.txt. `doc_chars` is unused here (synthetic data only)."""
    path = m2d2_archive(domain)
    found = False
    with tarfile.open(path, "r|gz") as tar:
        for member in tar:
            if not member.name.endswith(("/train.txt", "/valid.txt")):
                continue
            found = True
            yield from split_papers(tar.extractfile(member))
    if not found:
        raise RuntimeError(f"{path} has no train.txt / valid.txt")


def split_papers(fobj, chunk: int = 1 << 24):
    """Stream a text file and yield its blank-line-separated blocks, stripped, non-empty."""
    dec = codecs.getincrementaldecoder("utf-8")(errors="ignore")
    pat = re.compile(r"\n{2,}")
    carry = ""
    for b in iter(lambda: fobj.read(chunk), b""):
        parts = pat.split(carry + dec.decode(b))
        carry = parts.pop()                       # the last block may continue in the next chunk
        for part in parts:
            if part.strip():
                yield part.strip()
    tail = (carry + dec.decode(b"", final=True)).strip()
    for part in pat.split(tail):
        if part.strip():
            yield part.strip()


def jsonl_docs(path: str, domains: tuple):
    by_dom = {d: [] for d in domains}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r["domain"] in by_dom:
                by_dom[r["domain"]].append(r["text"])
    return by_dom


def synthetic_docs(domain_index: int, n_domains: int, doc_chars: int, seed: int = 0):
    """Endless pseudo-text. Each domain has 3 latent sub-topics; words come from a shared vocabulary,
    a domain vocabulary and a sub-topic vocabulary, so both labels and label-free clusters carry signal."""
    rng = np.random.default_rng(seed)
    letters = np.array(list("abcdefghijklmnopqrstuvwxyz"))

    def words(n, lo=2, hi=7):
        return ["".join(rng.choice(letters, rng.integers(lo, hi))) for _ in range(n)]

    common = words(200)
    dom_words = [words(60) for _ in range(n_domains)]
    sub_words = [[words(30) for _ in range(3)] for _ in range(n_domains)]
    zipf = lambda n: (lambda p: p / p.sum())(1.0 / np.arange(1, n + 1))
    pc, pd, ps = zipf(200), zipf(60), zipf(30)
    drng = np.random.default_rng(1000 + domain_index)
    while True:
        sub = drng.integers(3)
        out, size = [], 0
        while size < doc_chars:
            u = drng.random()
            if u < 0.55:
                w = common[drng.choice(200, p=pc)]
            elif u < 0.8:
                w = dom_words[domain_index][drng.choice(60, p=pd)]
            else:
                w = sub_words[domain_index][sub][drng.choice(30, p=ps)]
            out.append(w)
            size += len(w) + 1
            if drng.random() < 0.08:
                out.append(".\n")
        yield " ".join(out)


# ============================================================================== build

def download(p: Profile) -> None:
    """Login node: fetch everything `build` needs into the HF cache."""
    if p.data.source == "m2d2_arxiv":
        for d in p.data.domains:
            log(f"download {d}_l1.tar.gz -> {m2d2_archive(d)}")
    if p.data.tokenizer == "gpt2":
        Tokenizer("gpt2")
        log("tiktoken gpt2 encoding cached")


def build(p: Profile) -> None:
    out = data_dir(p)
    if is_done(out, "prepare"):
        log(f"data already built in {out}")
        return
    out.mkdir(parents=True, exist_ok=True)
    dc = p.data
    tok = Tokenizer(dc.tokenizer)
    assert tok.n_vocab <= p.model.vocab_size, "model vocab smaller than tokenizer vocab"
    assert tok.n_vocab < 2 ** 16, "uint16 token storage"
    quotas = dict(dc.split_tokens)
    order = [s for s, _ in dc.split_tokens]
    need = sum(quotas.values())
    doc_chars = dc.doc_tokens * (4 if dc.tokenizer == "gpt2" else 1)

    jsonl = jsonl_docs(dc.jsonl_path, dc.domains) if dc.source == "jsonl" else None
    split_docs = {s: [] for s in order}          # split -> list of (domain, token array)
    for di, dom in enumerate(dc.domains):
        if dc.source == "m2d2_arxiv":
            it = m2d2_docs(dom, doc_chars)
        elif dc.source == "jsonl":
            it = iter(jsonl[dom])
        elif dc.source == "synthetic":
            it = synthetic_docs(di, len(dc.domains), doc_chars)
        else:
            raise SystemExit(f"unknown data source {dc.source!r}")
        si, filled, total = 0, 0, 0
        while si < len(order):
            batch = [t for _, t in zip(range(256), it)]
            if not batch:
                raise SystemExit(f"domain {dom} ran out of text in split {order[si]} "
                                 f"({total:,} of {need:,} tokens)")
            for ids in tok.encode_batch(batch):
                ids = ids[: MAX_DOC_TOKENS - 1] + [tok.eot]
                if len(ids) < MIN_DOC_TOKENS or si >= len(order):
                    continue
                split_docs[order[si]].append((di, np.asarray(ids, dtype=np.uint16)))
                filled += len(ids)
                total += len(ids)
                if filled >= quotas[order[si]]:
                    si, filled = si + 1, 0
        log(f"{dom}: {total:,} tokens")

    rng = np.random.default_rng(SHUFFLE_SEED)
    doc_domain, doc_split, doc_len = [], [], []
    meta_splits = {}
    for s_idx, s in enumerate(order):
        docs = split_docs[s]
        perm = rng.permutation(len(docs))           # mix domains inside each split stream
        toks = np.concatenate([docs[i][1] for i in perm])
        first_id = len(doc_domain)
        ids = np.concatenate([np.full(len(docs[i][1]), first_id + j, dtype=np.int32)
                              for j, i in enumerate(perm)])
        for i in perm:
            doc_domain.append(docs[i][0]); doc_split.append(s_idx); doc_len.append(len(docs[i][1]))
        toks.tofile(out / f"{s}.tokens.bin")
        ids.tofile(out / f"{s}.docs.bin")
        meta_splits[s] = int(len(toks))
        log(f"split {s}: {len(toks):,} tokens, {len(docs):,} papers")
    np.savez(out / "docs.npz", domain=np.array(doc_domain, np.int16), split=np.array(doc_split, np.int8),
             n_tokens=np.array(doc_len, np.int32))
    write_json(out / "meta.json", dict(source=dc.source, tokenizer=dc.tokenizer, eot=tok.eot,
                                       n_vocab=tok.n_vocab, domains=list(dc.domains), splits=meta_splits,
                                       split_order=order, n_docs=len(doc_domain), doc_tokens=dc.doc_tokens,
                                       m2d2_revision=M2D2_REVISION if dc.source == "m2d2_arxiv" else None))
    mark_done(out, "prepare")
