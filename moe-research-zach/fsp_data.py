#!/usr/bin/env python
"""csr-forecast-specialist-pretraining-v2 (fspv2): data build (protocol section 3). CPU only. Dry-run by default.

  MOE_RESEARCH_APPROVE=cpu-model .venv/bin/python fsp_data.py build --execute

Streams the five selected M2D2 arXiv L1 archives (sha256-verified; test.txt members hashed, never decoded), exact
dedup with cross-listed papers assigned once to the scarcest listing domain, MinHash near-dedup over the union,
whole-paper tokenisation into <=1024-token segments, exclusive-first evaluation buckets, the deterministic allocation
rule, and frozen manifests. Reuses (imports, never modifies) the v1 streaming / MinHash functions in dfc_data.py.
Output: artifacts/forecast-specialist-pretraining-v2/data/ (0400) ; locked test bucket in data/locked/ (0400).
"""
import hashlib
import json
import multiprocessing as mp
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from csr import guard  # noqa: E402
import dfc_data as v1  # noqa: E402  (read_domain, minhash, near_dedup, paper_chunks, LFS, REPO, REV)

DOMAINS = ["cs_l1", "math_l1", "physics_l1", "q-bio_l1", "stat_l1"]
ART = guard.ROOT / "artifacts/forecast-specialist-pretraining-v2"
OUT = ART / "data"
M = 1_000_000
TARGET = {"draft_shared": 50 * M, "draft_cont": 50 * M, "main": 200 * M, "calib": 10 * M, "val": 10 * M,
          "test": 10 * M}
MINIMUM = {"draft_shared": 25 * M, "draft_cont": 25 * M, "main": 100 * M, "calib": 5 * M, "val": 5 * M,
           "test": 5 * M}
MARGIN = 2 * M
KEEP_TOP_LARGE = 150_000
N_CAND = {"cs_l1": 60_000, "math_l1": 30_000}  # others: every assigned paper
TOKENIZE_STOP = 345 * M  # lazily tokenise a surplus domain only this far (>= 330M target + margin)
EVAL_ORDER = ["test", "val", "calib_sel", "calib_search"]
TRAIN_ORDER = ["draft_shared", "draft_cont", "main"]


def u_of(salt, h):
    return int.from_bytes(hashlib.sha256(f"{salt}|{h}".encode()).digest()[:8], "little") / 2**64


def allocation(A):
    """Protocol section 3.5. A = min available valid tokens per domain. Returns per-domain quotas (valid tokens)."""
    if A >= sum(TARGET.values()):
        q = dict(TARGET)
        rule = "target"
    else:
        if A < sum(MINIMUM.values()) + MARGIN:
            raise guard.GuardError(f"data insufficient: A={A} < minimum {sum(MINIMUM.values()) + MARGIN}")
        q = dict(MINIMUM)
        s = A - sum(MINIMUM.values()) - MARGIN
        add = min(s, TARGET["main"] - q["main"]); q["main"] += add; s -= add
        add = min(s, 3 * (TARGET["val"] - MINIMUM["val"])); s -= add
        for k in ("calib", "val", "test"):
            q[k] += add / 3
        add = min(s, 2 * (TARGET["draft_shared"] - MINIMUM["draft_shared"])); s -= add
        for k in ("draft_shared", "draft_cont"):
            q[k] += add / 2
        rule = "fallback"
    half = M // 2
    q = {k: int(v // half * half) for k, v in q.items()}
    quotas = {"draft_shared": q["draft_shared"], "draft_cont": q["draft_cont"], "main": q["main"],
              "calib_search": q["calib"] // 2 // half * half, "val": q["val"], "test": q["test"]}
    quotas["calib_sel"] = q["calib"] - quotas["calib_search"]
    return quotas, rule


def _tok_batch(texts):
    from csr.artifacts import load_lm_tokenizer
    tok = load_lm_tokenizer()
    return [v1.paper_chunks(e.ids) for e in tok.encode_batch(texts)]


def _chunks(xs, n):
    return [xs[i:i + n] for i in range(0, len(xs), n)]


def _read(dom):
    return v1.read_domain(dom, keep_top=KEEP_TOP_LARGE if dom in N_CAND else None)


def build():
    if OUT.exists():
        raise guard.GuardError(f"{OUT} exists; refusing to overwrite")
    guard.require_disk(10 * 1024**3, "fspv2 data build")
    from huggingface_hub import HfApi
    t0 = time.time()
    info = HfApi().get_paths_info(v1.REPO, [f"data/{d}.tar.gz" for d in DOMAINS], repo_type="dataset",
                                  revision=v1.REV)
    for it in info:
        d = it.path.split("/")[-1][:-len(".tar.gz")]
        if (it.size, it.lfs.sha256) != v1.LFS[d]:
            raise guard.GuardError(f"LFS metadata changed for {d}")
    guard.ledger("NETWORK", f"fspv2: stream {len(DOMAINS)} archives from {v1.REPO}@{v1.REV[:8]} (not stored; sha256 "
                 f"verified)")
    with mp.get_context("fork").Pool(len(DOMAINS)) as pool:
        reads = pool.map(_read, DOMAINS)
    reads = {r["domain"]: r for r in reads}
    print("read", round(time.time() - t0), {d: len(r["all"]) for d, r in reads.items()}, flush=True)
    # ---------------- exact dedup + cross-list assignment
    hdoms = {}
    for dom in DOMAINS:
        for h, _, _ in reads[dom]["all"]:
            hdoms.setdefault(h, set()).add(dom)
    excl_count = {d: len({h for h, _, _ in reads[d]["all"] if len(hdoms[h]) == 1}) for d in DOMAINS}
    scarcity = sorted(DOMAINS, key=lambda d: (excl_count[d], d))
    rank = {d: i for i, d in enumerate(scarcity)}
    assigned = {h: min(ds, key=lambda d: rank[d]) for h, ds in hdoms.items()}
    inv = {}
    cands = {}
    for dom in DOMAINS:
        r = reads[dom]
        mine = sorted({(v1.u_of("dfcv1-cand", h), h) for h, _, _ in r["all"] if assigned[h] == dom})
        if dom in N_CAND:
            mine = mine[:N_CAND[dom]]
        cand = [h for _, h in mine]
        missing = [h for h in cand if h not in r["texts"]]
        if missing:
            print(f"[{dom}] re-stream (missing {len(missing)})", flush=True)
            r["texts"] = v1.read_domain(dom, keep_top=None)["texts"]
        cands[dom] = cand
        inv[dom] = {"archive_sha256": r["archive_sha256"], "archive_bytes": r["archive_bytes"], "members": r["members"],
                    "read_seconds": r["seconds"], "attempts": r["attempts"], "n_docs_raw": len(r["all"]),
                    "n_unique_hashes": len({h for h, _, _ in r["all"]}),
                    "n_exclusive_hashes": excl_count[dom],
                    "n_crosslisted_assigned_here": sum(1 for h in {h for h, _, _ in r["all"]}
                                                       if assigned[h] == dom and len(hdoms[h]) > 1),
                    "n_crosslisted_assigned_elsewhere": sum(1 for h in {h for h, _, _ in r["all"]}
                                                            if assigned[h] != dom),
                    "n_candidates": len(cand)}
    # ---------------- near dedup over the union of candidates
    flat = [(dom, h) for dom in DOMAINS for h in cands[dom]]
    texts = [reads[dom]["texts"][h][1] for dom, h in flat]
    t1 = time.time()
    with mp.get_context("fork").Pool(24) as pool:
        sigs = np.concatenate(pool.map(v1._mh_batch, _chunks(texts, 500)))
    uf, n_pairs = v1.near_dedup(sigs)
    clusters = {}
    for i in range(len(flat)):
        clusters.setdefault(uf.find(i), []).append(i)
    drop, nonexcl_near = set(), set()
    n_cross, n_within = 0, 0
    for mem in clusters.values():
        if len(mem) < 2:
            continue
        keep = min(mem, key=lambda i: (rank[flat[i][0]], u_of("fspv2-near", flat[i][1])))
        drop.update(i for i in mem if i != keep)
        if len({flat[i][0] for i in mem}) > 1:
            nonexcl_near.add(keep)
            n_cross += len(mem) - 1
        else:
            n_within += len(mem) - 1
    print("near-dedup", round(time.time() - t1), "pairs", n_pairs, "drop", len(drop), flush=True)
    kept = {dom: [] for dom in DOMAINS}
    for i, (dom, h) in enumerate(flat):
        if i in drop:
            continue
        exclusive = len(hdoms[h]) == 1 and i not in nonexcl_near
        kept[dom].append((u_of("fspv2-split", h), h, exclusive, i))
    # ---------------- lazy tokenisation in split order
    per_dom = {}
    for dom in DOMAINS:
        docs = sorted(kept[dom])
        out, tot, n_short = [], 0, 0
        pos = 0
        with mp.get_context("fork").Pool(24) as pool:
            while pos < len(docs) and tot < TOKENIZE_STOP:
                block = docs[pos:pos + 6144]
                pos += len(block)
                toks = pool.map(_tok_batch, _chunks([texts[i] for _, _, _, i in block], 256))
                toks = [x for b in toks for x in b]
                for (u, h, ex, i), chunks in zip(block, toks):
                    if not chunks:
                        n_short += 1
                        continue
                    out.append((h, ex, chunks))
                    tot += int(sum(len(a) - 1 for a in chunks))
        per_dom[dom] = out
        inv[dom].update({"near_dup_dropped": sum(1 for i in drop if flat[i][0] == dom),
                         "n_kept_after_near": len(docs), "n_tokenised_docs": len(out),
                         "tokenised_valid_tokens": tot, "fully_tokenised": pos >= len(docs),
                         "dropped_short": n_short,
                         "exclusive_tokenised_valid_tokens": int(sum(sum(len(a) - 1 for a in c)
                                                                     for _, ex, c in out if ex))})
        print(dom, inv[dom]["tokenised_valid_tokens"], inv[dom]["fully_tokenised"], flush=True)
    A = min(inv[d]["tokenised_valid_tokens"] for d in DOMAINS)
    quotas, rule = allocation(A)
    print("A", A, "rule", rule, quotas, flush=True)
    # ---------------- buckets
    OUT.mkdir(parents=True)
    (OUT / "locked").mkdir()
    buckets = {b: {"tokens": [], "ids": [], "papers": [], "domains": [], "excl": []} for b in EVAL_ORDER + TRAIN_ORDER}
    per_dom_meta, deviations = {}, []
    for di, dom in enumerate(DOMAINS):
        docs = per_dom[dom]
        used = set()
        meta = {}

        def fill(bucket, pool_docs, n0=0):
            q = quotas[bucket]
            n, k, nseg, tr = n0, 0, 0, 0
            for h, ex, chunks in pool_docs:
                if n >= q:
                    break
                if h in used:
                    continue
                used.add(h)
                k += 1
                for ci, a in enumerate(chunks):
                    need = q - n
                    if len(a) - 1 > need:
                        a = a[:need + 1]
                        tr += 1
                    if len(a) < 2:
                        break
                    buckets[bucket]["tokens"].append(a)
                    buckets[bucket]["ids"].append(f"{h}#{ci}")
                    buckets[bucket]["papers"].append(h)
                    buckets[bucket]["domains"].append(di)
                    buckets[bucket]["excl"].append(bool(ex))
                    n += len(a) - 1
                    nseg += 1
                    if n >= q:
                        break
            return {"n_papers": k, "n_segments": nseg, "valid_tokens": n, "truncated_last_papers": tr}
        for b in EVAL_ORDER:
            meta[b] = fill(b, [d for d in docs if d[1]])
            if meta[b]["valid_tokens"] < quotas[b]:
                extra = fill(b, [d for d in docs if not d[1]], n0=meta[b]["valid_tokens"])
                deviations.append({"domain": dom, "bucket": b, "exclusive_tokens": meta[b]["valid_tokens"],
                                   "filled_from_non_exclusive": extra["valid_tokens"] - meta[b]["valid_tokens"]})
                meta[b] = {"exclusive_part": meta[b], "non_exclusive_part": extra, "valid_tokens": extra["valid_tokens"]}
            assert meta[b]["valid_tokens"] == quotas[b], (dom, b, meta[b], quotas[b])
        for b in TRAIN_ORDER:
            meta[b] = fill(b, docs)
            assert meta[b]["valid_tokens"] == quotas[b], (dom, b, meta[b], quotas[b])
        per_dom_meta[dom] = meta
    seen = {}
    splits_meta = {}
    for b, bk in buckets.items():
        for pp in bk["papers"]:
            if seen.get(pp, b) != b:
                raise guard.GuardError(f"paper {pp} in two buckets ({seen[pp]}, {b})")
            seen[pp] = b
        d = OUT / "locked" if b == "test" else OUT
        tk = np.concatenate(bk["tokens"]).astype(np.uint16)
        offsets = np.zeros(len(bk["tokens"]) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([len(a) for a in bk["tokens"]])
        np.save(d / f"{b}_tokens.npy", tk)
        np.save(d / f"{b}_offsets.npy", offsets)
        (d / f"{b}_docs.json").write_text(json.dumps({"ids": bk["ids"], "domains": bk["domains"],
                                                       "papers": bk["papers"], "exclusive": bk["excl"]}))
        splits_meta[b] = {"n_docs": len(bk["ids"]), "n_tokens": int(tk.size), "valid_tokens": int(tk.size - len(bk["ids"])),
                          "sha256_tokens": guard.sha256_file(d / f"{b}_tokens.npy"),
                          "sha256_offsets": guard.sha256_file(d / f"{b}_offsets.npy"),
                          "sha256_docs": guard.sha256_file(d / f"{b}_docs.json"),
                          "doc_ids_sha256": hashlib.sha256(json.dumps(bk["ids"]).encode()).hexdigest(),
                          "n_papers": len(set(bk["papers"])), "frac_exclusive_segments": float(np.mean(bk["excl"])),
                          "locked": b == "test"}
    man = {"experiment": "csr-forecast-specialist-pretraining-v2", "created": guard.now(), "dataset": v1.REPO,
           "revision": v1.REV, "license": "cc-by-nc-4.0", "domains": DOMAINS,
           "domain_ids": {d: i for i, d in enumerate(DOMAINS)},
           "domain_names": {"cs_l1": "computer science/code", "math_l1": "mathematics", "physics_l1": "physics",
                            "q-bio_l1": "biomedical/q-bio", "stat_l1": "statistics"},
           "tokenizer": "EleutherAI/pythia-12b@bb1e3e710cdf6b524461d543cfb5ba773f0a81b6 (artifacts/tokenizer)",
           "scarcity_order": scarcity, "exclusive_counts": excl_count, "A_min_available_valid_tokens": A,
           "allocation_rule": rule, "quotas_valid_tokens_per_domain": quotas, "per_domain": per_dom_meta,
           "splits": splits_meta, "locked_test_member_hashes": {d: inv[d]["members"].get("test") for d in DOMAINS},
           "near_dedup": {"pairs_merged": n_pairs, "cross_domain_dropped": n_cross, "within_domain_dropped": n_within,
                          "method": "MinHash first 800 normalised words, 5-word shingles, 128 perms, 32x4 LSH, est. "
                                    "Jaccard>=0.8 over the union of all five candidate pools; one paper kept per "
                                    "cluster (scarcest domain, then hash order); cross-domain keepers non-exclusive"},
           "exact_dedup": "sha256(normalised full text) across the five domains; cross-listed papers assigned once to "
                          "the scarcest listing domain (non-exclusive); within-domain repeats keep one copy",
           "inventory": inv, "deviations": deviations, "build_seconds": time.time() - t0}
    man["splits_sha256"] = guard.sha256_json({k: {kk: v[kk] for kk in ("sha256_tokens", "sha256_offsets",
                                                                       "sha256_docs")} for k, v in splits_meta.items()})
    guard.atomic_write_json(OUT / "manifest.json", man)
    for p in list(OUT.iterdir()) + list((OUT / "locked").iterdir()):
        if p.is_file():
            p.chmod(0o400)
    guard.ledger("FSP_DATA_BUILD", f"{OUT.relative_to(guard.ROOT)} rule={rule} quotas={quotas} "
                 f"splits_sha256={man['splits_sha256']}")
    print(json.dumps({"A": A, "rule": rule, "quotas": quotas,
                      "splits": {k: v["valid_tokens"] for k, v in splits_meta.items()},
                      "seconds": round(time.time() - t0)}, indent=1))


def main():
    guard.check_root()
    if sys.argv[1] == "build":
        if guard.require_execution("--execute" in sys.argv, "cpu-model"):
            build()
    else:
        raise SystemExit("unknown command")


if __name__ == "__main__":
    main()
