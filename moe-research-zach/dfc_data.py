#!/usr/bin/env python
"""csr-domain-forecast-curriculum-v1 (dfcv1): inventory, de-duplicate, select domains, split and tokenise M2D2 arXiv L1
domains (docs/DOMAIN_FORECAST_CURRICULUM_V1_PREREG.md section 2). CPU only. Dry-run by default.

  MOE_RESEARCH_APPROVE=download  .venv/bin/python dfc_data.py fetch-small --execute   # archives < 200 MB stored
  MOE_RESEARCH_APPROVE=cpu-model .venv/bin/python dfc_data.py build --execute          # streams the large archives

test.txt members are hashed as raw bytes and never decoded. Output: artifacts/domain-forecast-curriculum-v1/data/
(TokenSplit files per split + docs json + manifest.json, all 0400) and inventory.json.
"""
import codecs
import hashlib
import heapq
import json
import multiprocessing as mp
import re
import sys
import tarfile
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402

REPO = "machelreid/m2d2"
REV = "6ae613ba744ee56f645aee9c577d04f7e3d48b30"
CANDIDATES = ["cs_l1", "math_l1", "physics_l1", "q-bio_l1", "econ_l1", "stat_l1", "q-fin_l1", "eess_l1"]
PREFERRED = CANDIDATES[:5]
LFS = {  # from list_repo_tree at REV (2026-11-10); re-verified at build time
    "cs_l1": (3547715436, "e764f640816b7fb511e2cf10d29ea463f41eaefee1ba21089489ed2f45e3dabf"),
    "math_l1": (5361590226, "1a3eafea0216c2c43b0acd24ad032352fac3feb305c1280c9bef188dfe83eaf0"),
    "physics_l1": (1369929637, "9964a24e5250e82bb10a510c1c54258d4b6588dfdfe624fecb45ee1fd96c26bf"),
    "q-bio_l1": (624302371, "d3b315e7e7d43212691e598123fcbff883c1deb50aee019932c91b8070c3ea68"),
    "econ_l1": (20595309, "bad195320000020538b30bf89677adeccfdae2df0ca1e0b943c910755a90364d"),
    "stat_l1": (782271893, "60372f0b62e6f19088f7726280527df137a288fe2ed72af251a785f7c68f0298"),
    "q-fin_l1": (117115057, "88a9b406d0d3f113a04a17af1cf886577f30a5fce176917b46592b8b11f2ffeb"),
    "eess_l1": (148655208, "46433c0081afb991bf78f4c247547878c20313e872f97d4aa2b3453a8e9d5c1d"),
}
STORE_MAX = 200_000_000
ART = guard.ROOT / "artifacts/domain-forecast-curriculum-v1"
RAW = ART / "raw/data"
OUT = ART / "data"
KEEP_TOP = 150_000  # per-domain texts retained (smallest u_cand) before exact-dedup filtering
N_CAND = 80_000
PREFIX_CHARS = 20_000
NEAR_WORDS = 800
MAX_LEN = 1024
MIN_TOK = 64
QUOTAS = [("val", 1_000_000), ("calib", 1_500_000), ("draft", 6_000_000), ("main_p1", 9_000_000),
          ("main_p2", 2_250_000), ("main_p3", 11_250_000)]
REQUIRED = sum(q for _, q in QUOTAS)  # 31,000,000
N_SELECT = 5
ATTEMPTS = 6


def norm(t):
    return re.sub(r"\s+", " ", t.lower()).strip()


def u_of(salt, h):
    return int.from_bytes(hashlib.sha256(f"{salt}|{h}".encode()).digest()[:8], "little") / 2**64


class HashReader:
    def __init__(self, f):
        self.f, self.h, self.n = f, hashlib.sha256(), 0

    def read(self, n=-1):
        b = self.f.read(n)
        self.h.update(b)
        self.n += len(b)
        return b

    def drain(self):
        while self.read(1 << 22):
            pass


def _open_source(dom):
    size, _ = LFS[dom]
    if size < STORE_MAX:
        return open(RAW / f"{dom}.tar.gz", "rb"), "stored"
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    return fs.open(f"datasets/{REPO}@{REV}/data/{dom}.tar.gz", "rb", block_size=8 << 20), "streamed"


def _iter_docs(fobj):
    """Stream-split a text member on runs of >= 2 newlines; yields stripped non-empty documents."""
    dec = codecs.getincrementaldecoder("utf-8")()
    carry = ""
    pat = re.compile(r"\n{2,}")
    for b in iter(lambda: fobj.read(64 << 20), b""):
        s = carry + dec.decode(b)
        parts = pat.split(s)
        carry = parts.pop()  # may be incomplete
        for p in parts:
            p = p.strip()
            if p:
                yield p
    s = (carry + dec.decode(b"", final=True)).strip()
    if s:
        for p in pat.split(s):
            p = p.strip()
            if p:
                yield p


def read_domain(dom, keep_top=KEEP_TOP):
    last = None
    for att in range(ATTEMPTS):
        try:
            return _read_domain_once(dom, keep_top) | {"attempts": att + 1}
        except Exception as e:  # noqa: BLE001  (transient network / corruption: bounded retry)
            last = f"{type(e).__name__}: {e}"
            print(f"[{dom}] attempt {att + 1} failed: {last}", flush=True)
            time.sleep(min(600, 30 * 2 ** att))
    raise RuntimeError(f"{dom}: all {ATTEMPTS} attempts failed: {last}")


def _read_domain_once(dom, keep_top):
    t0 = time.time()
    raw, how = _open_source(dom)
    hr = HashReader(raw)
    allrec = []  # (hash, u_cand, member)
    heap = []  # max-heap on u via negative: keep smallest keep_top u
    members = {}
    with tarfile.open(fileobj=hr, mode="r|gz") as tf:
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
                members["test"] = {"member": m.name, "bytes": n, "raw_sha256": h.hexdigest(), "decoded": False}
                continue
            if name not in ("train.txt", "valid.txt"):
                members[name] = {"member": m.name, "bytes": m.size, "ignored": True}
                continue
            sp = "train" if name == "train.txt" else "valid"
            mh = hashlib.sha256()

            class _R:
                def read(self, n):
                    b = f.read(n)
                    mh.update(b)
                    return b
            nd, nchars = 0, 0
            for d in _iter_docs(_R()):
                h = hashlib.sha256(norm(d).encode()).hexdigest()
                u = u_of("dfcv1-cand", h)
                allrec.append((h, u, sp))
                nd += 1
                nchars += len(d)
                item = (-u, h, sp, d)  # full paper text (deviation: whole papers chunked, see deviations.jsonl)
                if keep_top is None or len(heap) < keep_top:
                    heapq.heappush(heap, item)
                elif -heap[0][0] > u:
                    heapq.heapreplace(heap, item)
            members[sp] = {"member": m.name, "bytes": m.size, "raw_sha256": mh.hexdigest(), "n_docs_raw": nd,
                           "chars": nchars}
    hr.drain()
    sha = hr.h.hexdigest()
    if sha != LFS[dom][1] or hr.n != LFS[dom][0]:
        raise IOError(f"{dom}: archive sha/size mismatch {sha} {hr.n}")
    texts = {h: (sp, t) for _, h, sp, t in heap}
    return {"domain": dom, "source": how, "archive_sha256": sha, "archive_bytes": hr.n, "members": members,
            "all": allrec, "texts": texts, "keep_top": keep_top, "seconds": time.time() - t0}


# ------------------------------------------------------------------ MinHash near-dedup
_P = np.uint64(1099511628211)
_rng = np.random.default_rng(20261110)
PERM_A = _rng.integers(1, 2**63, size=128, dtype=np.uint64) | np.uint64(1)
PERM_B = _rng.integers(0, 2**63, size=128, dtype=np.uint64)
_wcache = {}


def _wh(w):
    v = _wcache.get(w)
    if v is None:
        v = int.from_bytes(hashlib.blake2b(w.encode(), digest_size=8).digest(), "little")
        if len(_wcache) < 2_000_000:
            _wcache[w] = v
    return v


def minhash(text):
    words = norm(text[:PREFIX_CHARS]).split(" ")[:NEAR_WORDS]
    wh = np.fromiter((_wh(w) for w in words), dtype=np.uint64, count=len(words))
    if len(wh) >= 5:
        with np.errstate(over="ignore"):
            sh = wh[:-4].copy()
            for k in range(1, 5):
                sh = sh * _P + wh[k:len(wh) - 4 + k]
    else:
        sh = wh[:1] if len(wh) else np.zeros(1, dtype=np.uint64)
    sh = np.unique(sh)
    with np.errstate(over="ignore"):
        return (sh[None, :] * PERM_A[:, None] + PERM_B[:, None]).min(axis=1)


def _mh_batch(texts):
    return np.stack([minhash(t) for t in texts]) if texts else np.zeros((0, 128), np.uint64)


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[max(a, b)] = min(a, b)


def near_dedup(sigs, bands=32, thr=0.8):
    n, r = len(sigs), 128 // bands
    uf = UF(n)
    n_pairs = 0
    for bi in range(bands):
        buckets = {}
        sub = np.ascontiguousarray(sigs[:, bi * r:(bi + 1) * r])
        for i in range(n):
            buckets.setdefault(sub[i].tobytes(), []).append(i)
        for mem in buckets.values():
            if len(mem) < 2:
                continue
            for a_i in range(len(mem)):
                for b_i in range(a_i + 1, len(mem)):
                    a, b = mem[a_i], mem[b_i]
                    if uf.find(a) == uf.find(b):
                        continue
                    if np.mean(sigs[a] == sigs[b]) >= thr:
                        uf.union(a, b)
                        n_pairs += 1
    return uf, n_pairs


# ------------------------------------------------------------------ tokenisation
def paper_chunks(ids):
    """Whole paper -> consecutive segments of <= 1024 tokens of (ids + [EOS]); segments shorter than MIN_TOK (only
    possibly the last) are dropped. Each segment is an independent model document (positions/attention restart)."""
    from csr.data import EOS_ID
    seq = np.asarray(list(ids) + [EOS_ID], dtype=np.uint16)
    return [seq[i:i + MAX_LEN] for i in range(0, len(seq), MAX_LEN) if len(seq[i:i + MAX_LEN]) >= MIN_TOK]


def _tok_batch(texts):
    from csr.artifacts import load_lm_tokenizer
    tok = load_lm_tokenizer()
    return [(paper_chunks(e.ids), False) for e in tok.encode_batch(texts)]


def _chunks(xs, n):
    return [xs[i:i + n] for i in range(0, len(xs), n)]


# ------------------------------------------------------------------ commands
def fetch_small():
    from huggingface_hub import hf_hub_download
    RAW.mkdir(parents=True, exist_ok=True)
    rec = {}
    for dom in CANDIDATES:
        size, sha = LFS[dom]
        if size >= STORE_MAX:
            continue
        p = RAW / f"{dom}.tar.gz"
        for att in range(ATTEMPTS):
            try:
                if not p.exists():
                    guard.ledger("NETWORK", f"dfcv1: download {dom}.tar.gz ({REPO}@{REV[:8]})")
                    hf_hub_download(REPO, f"data/{dom}.tar.gz", repo_type="dataset", revision=REV,
                                    local_dir=str(ART / "raw"))
                break
            except Exception as e:  # noqa: BLE001
                print(f"{dom} attempt {att + 1}: {e}", flush=True)
                time.sleep(min(600, 30 * 2 ** att))
        got = guard.sha256_file(p)
        if got != sha:
            raise guard.GuardError(f"{dom}: sha mismatch {got}")
        rec[dom] = {"path": str(p.relative_to(guard.ROOT)), "bytes": p.stat().st_size, "sha256": got, "verified": True}
    print(json.dumps(rec, indent=1))
    return rec


def build():
    if OUT.exists():
        raise guard.GuardError(f"{OUT} exists; refusing to overwrite")
    guard.require_disk(2 * 1024**3, "dfcv1 data build")
    from huggingface_hub import HfApi
    t0 = time.time()
    # re-verify LFS metadata at the pinned revision
    info = HfApi().get_paths_info(REPO, [f"data/{d}.tar.gz" for d in CANDIDATES], repo_type="dataset", revision=REV)
    for it in info:
        d = it.path.split("/")[-1][:-len(".tar.gz")]
        if (it.size, it.lfs.sha256) != LFS[d]:
            raise guard.GuardError(f"LFS metadata changed for {d}")
    guard.ledger("NETWORK", f"dfcv1: stream {len([d for d in CANDIDATES if LFS[d][0] >= STORE_MAX])} archives from "
                 f"{REPO}@{REV[:8]} (not stored; sha256 verified)")
    with mp.get_context("fork").Pool(len(CANDIDATES)) as pool:
        reads = pool.map(read_domain, CANDIDATES)
    reads = {r["domain"]: r for r in reads}
    print("read", round(time.time() - t0), {d: len(r["all"]) for d, r in reads.items()}, flush=True)
    # exact dedup (deviation 2026-11-10, see deviations.jsonl): a hash present in MORE THAN ONE domain is dropped from
    # every place it occurs (ambiguous domain); repeated copies WITHIN one domain keep exactly one copy.
    cnt = Counter(h for r in reads.values() for h, _, _ in r["all"])
    hdoms = {}
    for dom in CANDIDATES:
        for h, _, _ in reads[dom]["all"]:
            hdoms.setdefault(h, set()).add(dom)
    dup = {h for h, ds in hdoms.items() if len(ds) > 1}  # cross-domain duplicates
    np.savez_compressed(ART / f"exact_hash_table_{guard.utc_stamp()}.npz",
                        **{dom: np.asarray([h for h, _, _ in reads[dom]["all"]]) for dom in CANDIDATES})
    inv = {}
    pools = {}
    for dom in CANDIDATES:
        r = reads[dom]
        surv = sorted({(u, h) for h, u, _ in r["all"] if h not in dup})
        cand = [h for _, h in surv[:N_CAND]]
        missing = [h for h in cand if h not in r["texts"]]
        if missing:  # retained top-K too small after dup removal: re-stream keeping everything
            print(f"[{dom}] re-stream (missing {len(missing)} candidate texts)", flush=True)
            r2 = read_domain(dom, keep_top=None)
            r["texts"] = r2["texts"]
        pools[dom] = cand
        inv[dom] = {"source": r["source"], "archive_sha256": r["archive_sha256"], "archive_bytes": r["archive_bytes"],
                    "lfs_sha256": LFS[dom][1], "verified": True, "members": r["members"], "attempts": r["attempts"],
                    "read_seconds": r["seconds"], "n_docs_raw": len(r["all"]),
                    "n_exact_dup_dropped": sum(1 for h, _, _ in r["all"] if h in dup),
                    "n_exact_cross_domain_docs_dropped": sum(1 for h, _, _ in r["all"] if h in dup),
                    "n_within_domain_repeat_copies_removed": sum(1 for h, _, _ in r["all"] if h not in dup)
                    - len({h for h, _, _ in r["all"] if h not in dup}),
                    "n_hashes_repeated_within_domain": sum(1 for h, _, _ in r["all"] if h not in dup and cnt[h] > 1),
                    "n_exact_survivors": len(surv), "n_candidates": len(cand)}
    # near dedup on the union of candidate pools
    flat = [(dom, h) for dom in CANDIDATES for h in pools[dom]]
    texts = [reads[dom]["texts"][h][1] for dom, h in flat]
    t1 = time.time()
    with mp.get_context("fork").Pool(24) as pool:
        sigs = np.concatenate(pool.map(_mh_batch, _chunks(texts, 500)))
    uf, n_pairs = near_dedup(sigs)
    clusters = {}
    for i in range(len(flat)):
        clusters.setdefault(uf.find(i), []).append(i)
    drop = set()
    n_cross, n_within = 0, 0
    for mem in clusters.values():
        if len(mem) < 2:
            continue
        doms = {flat[i][0] for i in mem}
        if len(doms) > 1:
            drop.update(mem)
            n_cross += len(mem)
        else:
            keep = min(mem, key=lambda i: u_of("dfcv1-cand", flat[i][1]))
            drop.update(i for i in mem if i != keep)
            n_within += len(mem) - 1
    print("near-dedup", round(time.time() - t1), "pairs", n_pairs, "drop", len(drop), flush=True)
    kept = [i for i in range(len(flat)) if i not in drop]
    with mp.get_context("fork").Pool(24) as pool:
        toks = pool.map(_tok_batch, _chunks([texts[i] for i in kept], 256))
    toks = [x for b in toks for x in b]
    per_dom = {d: [] for d in CANDIDATES}
    n_short, n_cut_short = Counter(), Counter()
    for i, (chunks, _) in zip(kept, toks):
        dom, h = flat[i]
        if not chunks:
            n_short[dom] += 1
            continue
        per_dom[dom].append((h, chunks))
    for dom in CANDIDATES:
        inv[dom].update({"near_dup_dropped": sum(1 for i in drop if flat[i][0] == dom),
                         "dropped_short": n_short[dom], "prefix_cut_short": n_cut_short[dom],
                         "n_docs_final_pool": len(per_dom[dom]),
                         "n_segments_final_pool": int(sum(len(c) for _, c in per_dom[dom])),
                         "valid_tokens_final_pool": int(sum(len(a) - 1 for _, c in per_dom[dom] for a in c))})
        inv[dom]["eligible"] = inv[dom]["valid_tokens_final_pool"] >= REQUIRED
    selected = [d for d in CANDIDATES if inv[d]["eligible"]][:N_SELECT]
    if len(selected) < N_SELECT:
        guard.atomic_write_json(ART / "inventory.json", {"candidates": inv, "selected": selected, "FAILED": True})
        raise guard.GuardError(f"only {len(selected)} eligible domains: {selected}")
    substitutions = [{"preferred": p, "eligible": inv[p]["eligible"],
                      "valid_tokens_final_pool": inv[p]["valid_tokens_final_pool"]} for p in PREFERRED
                     if p not in selected]
    guard.atomic_write_json(ART / "inventory.json", {
        "created": guard.now(), "repo": REPO, "revision": REV, "required_valid_tokens_per_domain": REQUIRED,
        "candidates": inv, "selected": selected, "substitutions": substitutions,
        "near_dedup": {"pairs_merged": n_pairs, "cross_domain_docs_dropped": n_cross, "within_domain_dropped": n_within,
                       "method": f"MinHash first {NEAR_WORDS} normalised words, 5-word shingles, 128 perms, 32x4 LSH, "
                                 f"est. Jaccard>=0.8; cross-domain clusters dropped entirely"},
        "exact_dedup": {"n_cross_domain_dup_hashes": len(dup), "method": "sha256(normalised full text) across all 8 "
                        "candidate domains, train+valid; hashes in >1 domain dropped everywhere; within-domain repeats "
                        "keep one copy (deviation from prereg 'every copy dropped')"}})
    # ---------------- split by document hash with exact quotas
    OUT.mkdir(parents=True)
    from csr.data import EOS_ID
    buckets = {b: {"tokens": [], "ids": [], "papers": [], "domains": [], "trunc": 0} for b, _ in QUOTAS}
    per_dom_meta = {}
    for di, dom in enumerate(selected):
        docs = sorted(per_dom[dom], key=lambda x: u_of("dfcv1-split", x[0]))
        it = iter(docs)
        meta = {}
        for b, q in QUOTAS:
            n, k, nseg, tr = 0, 0, 0, 0
            while n < q:  # whole papers (all their segments) in hash order; the last paper is cut to the quota
                h, chunks = next(it)
                k += 1
                for ci, a in enumerate(chunks):
                    need = q - n
                    if len(a) - 1 > need:
                        a = a[:need + 1]  # exactly `need` targets; rest of this paper discarded
                        tr += 1
                    buckets[b]["tokens"].append(a)
                    buckets[b]["ids"].append(f"{h}#{ci}")
                    buckets[b]["papers"].append(h)
                    buckets[b]["domains"].append(di)
                    n += len(a) - 1
                    nseg += 1
                    if n >= q:
                        break
            assert n == q, (dom, b, n, q)
            buckets[b]["trunc"] += tr
            meta[b] = {"n_papers": k, "n_segments": nseg, "valid_tokens": n, "truncated_last_papers": tr}
        per_dom_meta[dom] = meta
    # main = p1 + p2 + p3 with phase labels
    files = {"val": ["val"], "calib": ["calib"], "draft": ["draft"], "main": ["main_p1", "main_p2", "main_p3"]}
    splits_meta, seen = {}, {}
    for sp, bl in files.items():
        tokens, offsets, ids, doms, phase, papers = [], [0], [], [], [], []
        for pi, b in enumerate(bl):
            for a, h, d, pp in zip(buckets[b]["tokens"], buckets[b]["ids"], buckets[b]["domains"],
                                   buckets[b]["papers"]):
                if seen.get(pp, b) != b:
                    raise guard.GuardError(f"paper {pp} in two buckets ({seen[pp]}, {b})")
                seen[pp] = b
                papers.append(pp)
                tokens.append(a)
                offsets.append(offsets[-1] + len(a))
                ids.append(h)
                doms.append(d)
                phase.append(pi + 1)
        tk = np.concatenate(tokens).astype(np.uint16)
        np.save(OUT / f"{sp}_tokens.npy", tk)
        np.save(OUT / f"{sp}_offsets.npy", np.asarray(offsets, dtype=np.int64))
        dj = {"ids": ids, "domains": doms, "papers": papers}
        if sp == "main":
            dj["phase"] = phase
        (OUT / f"{sp}_docs.json").write_text(json.dumps(dj))
        splits_meta[sp] = {"n_docs": len(ids), "n_tokens": int(tk.size), "valid_tokens": int(tk.size - len(ids)),
                           "sha256_tokens": guard.sha256_file(OUT / f"{sp}_tokens.npy"),
                           "sha256_offsets": guard.sha256_file(OUT / f"{sp}_offsets.npy"),
                           "sha256_docs": guard.sha256_file(OUT / f"{sp}_docs.json"),
                           "doc_hashes_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                           "buckets": bl}
    man = {"experiment": "csr-domain-forecast-curriculum-v1", "created": guard.now(), "dataset": REPO,
           "revision": REV, "license": "cc-by-nc-4.0", "domains": selected,
           "domain_ids": {d: i for i, d in enumerate(selected)},
           "tokenizer": "EleutherAI/pythia-12b@bb1e3e710cdf6b524461d543cfb5ba773f0a81b6 (artifacts/tokenizer)",
           "doc_rule": "paper = member text split on \\n{2,}, stripped; whole paper tokenised, + EOS, cut into consecutive "
                       "<=1024-token segments (segments < 64 tokens dropped); each segment is one model document; "
                       "splits, buckets and dedup are by paper hash (all segments of a paper in one bucket)",
           "quotas_valid_tokens_per_domain": dict(QUOTAS), "per_domain": per_dom_meta, "splits": splits_meta,
           "locked_test": {d: inv[d]["members"].get("test") for d in CANDIDATES},
           "derangement": {"rule": "d -> (d+1) mod 5", "map": {i: (i + 1) % 5 for i in range(5)}},
           "build_seconds": time.time() - t0, "inventory_sha256": guard.sha256_file(ART / "inventory.json")}
    man["splits_sha256"] = guard.sha256_json({k: {kk: v[kk] for kk in ("sha256_tokens", "sha256_offsets",
                                                                       "sha256_docs")} for k, v in splits_meta.items()})
    guard.atomic_write_json(OUT / "manifest.json", man)
    for p in OUT.iterdir():
        p.chmod(0o400)
    guard.ledger("DFC_DATA_BUILD", f"{OUT.relative_to(guard.ROOT)} selected={selected} splits_sha256="
                 f"{man['splits_sha256']}")
    print(json.dumps({"selected": selected, "substitutions": substitutions,
                      "splits": {k: v["valid_tokens"] for k, v in splits_meta.items()},
                      "seconds": round(time.time() - t0)}, indent=1))


def main():
    guard.check_root()
    cmd = sys.argv[1]
    if cmd == "fetch-small":
        if guard.require_execution("--execute" in sys.argv, "download"):
            fetch_small()
    elif cmd == "build":
        if guard.require_execution("--execute" in sys.argv, "cpu-model"):
            build()
    else:
        raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    main()
