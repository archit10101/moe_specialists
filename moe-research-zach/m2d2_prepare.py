#!/usr/bin/env python
"""csr-m2d2-domain-prior-v1: deterministic balanced M2D2 subset (docs/M2D2_DOMAIN_PRIOR_V1_PROTOCOL.md).

CPU only. Streams each L1 archive; test.txt members are hashed as raw bytes and never decoded. Writes TokenSplit-format
files (train / calib_search / val) with per-document domain labels to artifacts/m2d2-domain-prior-v1/data/.
"""
import hashlib
import json
import re
import sys
import tarfile
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402

DOMS = ["Mathematics_and_logic", "Technology_and_applied_sciences", "Natural_and_physical_sciences",
        "Health_and_fitness", "History_and_events", "Culture_and_the_arts"]
RAW = guard.ROOT / "artifacts/m2d2-domain-prior-v1/raw/data"
OUT = guard.ROOT / "artifacts/m2d2-domain-prior-v1/data"
CAPS = {"train": 10_000_000, "calib_search": 500_000, "val": 1_000_000}
MAX_LEN = 1024
MIN_TOK = 64
CALIB_FRAC = 0.05


def norm_hash(t):
    return hashlib.sha256(re.sub(r"\s+", " ", t.lower()).strip().encode()).hexdigest()


def u_of(salt, h):
    return int.from_bytes(hashlib.sha256(f"{salt}|{h}".encode()).digest()[:8], "little") / 2**64


def split_docs(text):
    return [d.strip() for d in re.split(r"\n{3,}", text) if d.strip()]


def read_domain(dom):
    """Returns dict: test hash info, and for train/valid lists of (hash, text)."""
    p = RAW / f"{dom}.tar.gz"
    res = {"domain": dom, "archive_sha256": guard.sha256_file(p), "members": {}}
    with tarfile.open(p, "r|gz") as tf:
        for m in tf:
            if not m.isfile():
                continue
            name = m.name.split("/")[-1]
            f = tf.extractfile(m)
            if name == "test.txt":  # LOCKED: raw-byte hash only, never decoded
                h = hashlib.sha256()
                n = 0
                for b in iter(lambda: f.read(1 << 22), b""):
                    h.update(b)
                    n += len(b)
                res["members"]["test"] = {"member": m.name, "bytes": n, "raw_sha256": h.hexdigest(),
                                          "decoded": False}
                continue
            if name not in ("train.txt", "valid.txt"):
                res["members"][name] = {"member": m.name, "bytes": m.size, "ignored": True}
                continue
            raw = f.read()
            sp = "train" if name == "train.txt" else "valid"
            docs = split_docs(raw.decode("utf-8"))
            res["members"][sp] = {"member": m.name, "bytes": len(raw), "raw_sha256": hashlib.sha256(raw).hexdigest(),
                                  "n_docs_raw": len(docs)}
            res[sp] = [(norm_hash(d), d) for d in docs]
    return res


def tokenize_take(args):
    """Take docs in order until the cap of valid tokens. Returns (hashes, token arrays, stats)."""
    dom, sp, items, cap = args
    from csr.artifacts import load_lm_tokenizer
    from csr.data import EOS_ID
    tok = load_lm_tokenizer()
    hs, arrs, n_valid, n_trunc, n_short, i = [], [], 0, 0, 0, 0
    B = 256
    while n_valid < cap and i < len(items):
        chunk = items[i:i + B]
        i += B
        encs = tok.encode_batch([t for _, t in chunk])
        for (h, _), e in zip(chunk, encs):
            if n_valid >= cap:
                break
            ids = e.ids
            if len(ids) + 1 < MIN_TOK:
                n_short += 1
                continue
            if len(ids) > MAX_LEN - 1:
                n_trunc += 1
            a = np.asarray(ids[:MAX_LEN - 1] + [EOS_ID], dtype=np.uint16)
            hs.append(h)
            arrs.append(a)
            n_valid += len(a) - 1
    return dom, sp, hs, arrs, {"valid_tokens": n_valid, "n_docs": len(hs), "truncated": n_trunc,
                               "dropped_short": n_short, "reached_cap": n_valid >= cap,
                               "candidates_available": len(items)}


def main():
    guard.check_root()
    if not guard.require_execution("--execute" in sys.argv, "cpu-model"):
        return
    if OUT.exists():
        raise guard.GuardError(f"{OUT} exists; refusing to overwrite")
    guard.require_disk(3 * 1024**3, "m2d2 data build")
    t0 = time.time()
    with Pool(6) as pool:
        reads = pool.map(read_domain, DOMS)
    print("read", time.time() - t0, flush=True)
    # exact de-duplication across domains and train/valid splits: drop every copy of a repeated hash
    from collections import Counter
    cnt = Counter(h for r in reads for sp in ("train", "valid") for h, _ in r[sp])
    dup = {h for h, c in cnt.items() if c > 1}
    jobs, dedup = [], {}
    for r in reads:
        dom = r["domain"]
        tr = [(h, t) for h, t in r["train"] if h not in dup]
        va = [(h, t) for h, t in r["valid"] if h not in dup]
        dedup[dom] = {"train_in": len(r["train"]), "train_kept": len(tr), "valid_in": len(r["valid"]),
                      "valid_kept": len(va)}
        cal = [x for x in tr if u_of("m2d2dp-v1-part", x[0]) < CALIB_FRAC]
        trn = [x for x in tr if u_of("m2d2dp-v1-part", x[0]) >= CALIB_FRAC]
        for sp, items in (("train", trn), ("calib_search", cal), ("val", va)):
            items = sorted(items, key=lambda x: u_of("m2d2dp-v1-order", x[0]))
            jobs.append((dom, sp, items, CAPS[sp]))
        del r["train"], r["valid"]
    with Pool(6) as pool:
        toks = pool.map(tokenize_take, jobs)
    print("tokenized", time.time() - t0, flush=True)
    got = {(d, s): (hs, arrs, st) for d, s, hs, arrs, st in toks}
    # balance: if any domain misses a cap, downsample every domain to the common minimum
    final_caps = {}
    for sp in CAPS:
        mins = min(got[(d, sp)][2]["valid_tokens"] for d in DOMS)
        final_caps[sp] = min(CAPS[sp], mins)
    OUT.mkdir(parents=True)
    splits_meta = {}
    all_h = {}
    for sp in CAPS:
        tokens, offsets, ids, domains = [], [0], [], []
        per_dom = {}
        for di, d in enumerate(DOMS):
            hs, arrs, st = got[(d, sp)]
            n = 0
            k = 0
            for h, a in zip(hs, arrs):
                if n >= final_caps[sp]:
                    break
                tokens.append(a)
                offsets.append(offsets[-1] + len(a))
                ids.append(h)
                domains.append(di)
                n += len(a) - 1
                k += 1
            per_dom[d] = {**st, "taken_docs": k, "taken_valid_tokens": n}
        for h in ids:
            if h in all_h:
                raise guard.GuardError(f"hash {h} in two splits ({all_h[h]}, {sp})")
            all_h[h] = sp
        tk = np.concatenate(tokens).astype(np.uint16)
        off = np.asarray(offsets, dtype=np.int64)
        np.save(OUT / f"{sp}_tokens.npy", tk)
        np.save(OUT / f"{sp}_offsets.npy", off)
        (OUT / f"{sp}_docs.json").write_text(json.dumps({"ids": ids, "domains": domains}))
        splits_meta[sp] = {"n_docs": len(ids), "n_tokens": int(tk.size), "valid_tokens": int(tk.size - len(ids)),
                           "sha256_tokens": guard.sha256_file(OUT / f"{sp}_tokens.npy"),
                           "sha256_offsets": guard.sha256_file(OUT / f"{sp}_offsets.npy"),
                           "sha256_docs": guard.sha256_file(OUT / f"{sp}_docs.json"),
                           "doc_hashes_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                           "per_domain": per_dom}
    man = {"experiment": "csr-m2d2-domain-prior-v1", "created": guard.now(), "dataset": "machelreid/m2d2",
           "revision": "6ae613ba744ee56f645aee9c577d04f7e3d48b30", "license": "cc-by-nc-4.0", "domains": DOMS,
           "tokenizer": "EleutherAI/pythia-12b@bb1e3e710cdf6b524461d543cfb5ba773f0a81b6 (artifacts/tokenizer)",
           "doc_rule": "split on \\n{3,}; strip; drop < 64 tokens; tokens[:1023]+EOS",
           "dedup": {"method": "exact sha256(normalised text) across all domains and train/valid; every copy of a "
                               "repeated hash dropped", "n_dup_hashes": len(dup), "per_domain": dedup},
           "partition": {"calib_search": "u('m2d2dp-v1-part') < 0.05 of native train", "train": "rest of native "
                         "train", "val": "native valid", "order": "increasing u('m2d2dp-v1-order')"},
           "caps_requested": CAPS, "caps_final": final_caps,
           "archives": {r["domain"]: {"sha256": r["archive_sha256"], "members": r["members"]} for r in reads},
           "locked_test": {r["domain"]: r["members"].get("test") for r in reads},
           "splits": splits_meta, "build_seconds": time.time() - t0}
    man["splits_sha256"] = guard.sha256_json({k: {kk: v[kk] for kk in ("sha256_tokens", "sha256_offsets",
                                                                       "sha256_docs")} for k, v in splits_meta.items()})
    guard.atomic_write_json(OUT / "manifest.json", man)
    for p in OUT.iterdir():
        p.chmod(0o400)
    guard.ledger("M2D2_DATA_BUILD", f"{OUT}: {json.dumps({k: v['valid_tokens'] for k, v in splits_meta.items()})}")
    print(json.dumps({k: {"n_docs": v["n_docs"], "valid_tokens": v["valid_tokens"]} for k, v in splits_meta.items()}))
    print("caps_final", final_caps, "seconds", time.time() - t0)


if __name__ == "__main__":
    main()
