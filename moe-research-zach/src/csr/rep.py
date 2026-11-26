"""csr-representative-v1 helpers (new, versioned; nothing here changes the behaviour of earlier experiments).

- fresh-data row-group selection, split rule and deduplication against a reference set (docs/REP_V1_PREREG.md §2)
- batched sentence embedding for the semantic cache (same texts as segcache.embed_doc_sentences)
- F-v3 content-free random closer with a remaining-distance covariate, its fit and the acceptance criteria (§3)
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np

from .segment import chunk_lengths, stable_hash64

REP_EXPERIMENT = "csr-representative-v1"
REP_DATA_VERSION = "fwe-rep-v1"
REP_SPLIT_SALT = "csr-rep-v1"
REP_SPLITS = {"train": (0.0, 0.825), "calib": (0.825, 0.86), "val": (0.86, 0.93), "test": (0.93, 1.0)}
REP_N_ROW_GROUPS = 240
REP_RG_SEED = 20261101
OLD_ROW_GROUPS = [19, 34, 90, 144, 222, 252, 322, 402, 407, 494, 495, 496, 622, 627, 637, 679]  # fwe-v1
MIN_TRAIN_TOKENS = 110_000_000


def choose_rep_row_groups(n_total: int):
    eligible = [i for i in range(n_total) if i not in set(OLD_ROW_GROUPS)]
    rng = np.random.default_rng(REP_RG_SEED)
    return sorted(int(x) for x in rng.choice(eligible, size=REP_N_ROW_GROUPS, replace=False))


def rep_split_of(doc_id: str) -> str:
    u = int.from_bytes(hashlib.sha256((REP_SPLIT_SALT + doc_id).encode()).digest()[:8], "little") / 2**64
    for name, (lo, hi) in REP_SPLITS.items():
        if lo <= u < hi:
            return name
    return "test"


# ------------------------------------------------------------------ dedup against a reference set
_PERM = None


def _perms(n_perm=128, seed=1234):
    rng = np.random.default_rng(seed)  # identical permutations to csr.data.dedup
    a = rng.integers(1, 2**63, size=n_perm, dtype=np.uint64) | np.uint64(1)
    b = rng.integers(0, 2**63, size=n_perm, dtype=np.uint64)
    return a, b


def _sig_worker(text):
    from .data import minhash_signature
    global _PERM
    if _PERM is None:
        _PERM = _perms()
    return minhash_signature(text, *_PERM)


def exact_key(text):
    from .data import normalize_text
    return hashlib.sha256(normalize_text(text).encode()).hexdigest()


def dedup_against_reference(ref_texts, new_ids, new_texts, n_perm=128, bands=32, threshold=0.8, processes=6):
    """Reference documents are processed first and never output; a new document is dropped if it duplicates (exactly,
    or MinHash-Jaccard >= threshold) any reference document or an earlier new document (canonical order = sorted id).
    Same signature, banding and threshold as csr.data.dedup. Returns (keep_mask over new docs, report)."""
    import multiprocessing as mp
    order = sorted(range(len(new_ids)), key=lambda i: new_ids[i])
    all_texts = list(ref_texts) + [new_texts[i] for i in order]
    n_ref = len(ref_texts)
    with mp.get_context("fork").Pool(processes) as pool:
        sigs = pool.map(_sig_worker, all_texts, chunksize=256)
        ekeys = pool.map(exact_key, all_texts, chunksize=512)
    r = n_perm // bands
    seen, buckets = set(), {}
    keep = np.ones(len(new_ids), dtype=bool)
    stats = {"exact_vs_reference": 0, "exact_within_new": 0, "near_vs_reference": 0, "near_within_new": 0}
    ref_exact = set(ekeys[:n_ref])
    for k, (s, ek) in enumerate(zip(sigs, ekeys)):
        is_ref = k < n_ref
        if not is_ref and ek in seen:
            keep[order[k - n_ref]] = False
            stats["exact_vs_reference" if ek in ref_exact else "exact_within_new"] += 1
            continue
        seen.add(ek)
        keys = [(bi, s[bi * r:(bi + 1) * r].tobytes()) for bi in range(bands)]
        dup_of = None
        if not is_ref:
            for key in keys:
                for j in buckets.get(key, ()):
                    if np.mean(sigs[j] == s) >= threshold:
                        dup_of = j
                        break
                if dup_of is not None:
                    break
        if dup_of is not None:
            keep[order[k - n_ref]] = False
            stats["near_vs_reference" if dup_of < n_ref else "near_within_new"] += 1
            continue
        for key in keys:
            buckets.setdefault(key, []).append(k)
    rep = {"n_reference": n_ref, "n_new_in": len(new_ids), "n_new_kept": int(keep.sum()), **stats,
           "method": f"exact sha256(normalized) + minhash n_perm={n_perm} bands={bands} 5-word shingles "
                     f"jaccard>={threshold}; reference first, then new docs sorted by id"}
    return keep, rep


# ------------------------------------------------------------------ batched sentence embedding
def embed_docs_batched(tok, det, enc, docs):
    """Same texts as segcache.embed_doc_sentences (decode(ids[s:e+1]).strip()), decoded in one batch."""
    from .segment import sentence_spans
    all_ends, pieces = [], []
    for ids in docs:
        e = det.sentence_ends(ids)
        all_ends.append(e)
        pieces.extend(ids[s:t + 1].tolist() for s, t in sentence_spans(e))
    texts = [t.strip() for t in tok.decode_batch(pieces)] if pieces else []
    embs = enc.embed(texts) if texts else np.zeros((0, enc.dim), np.float32)
    out, k = [], 0
    for e in all_ends:
        out.append((e, embs[k:k + len(e)]))
        k += len(e)
    return out


# ------------------------------------------------------------------ F-v3 random closer
RANDOM_HAZARD_V3 = "csr-random-hazard-v3"
END_EDGES = [0, 32, 64, 128, 256]  # remaining tokens after the boundary: [0,32),[32,64),[64,128),[128,256),[256,inf)


@dataclass
class HazardPolicyV3:
    strata_edges: list
    end_edges: list
    len_bin: int
    min_len: int
    max_len: int
    hazard: list  # [n_strata][n_end_bins][n_len_bins]

    def stratum(self, L):
        return int(np.clip(np.searchsorted(self.strata_edges, L, side="right") - 1, 0, len(self.hazard) - 1))

    def end_bin(self, remaining):
        return int(np.searchsorted(self.end_edges, remaining, side="right") - 1)

    def len_b(self, cur_len):
        return (cur_len - self.min_len) // self.len_bin

    def h(self, s, remaining, cur_len):
        if cur_len >= self.max_len:
            return 0.0
        return float(self.hazard[s][self.end_bin(remaining)][self.len_b(cur_len)])

    def to_json(self):
        return json.dumps({"version": RANDOM_HAZARD_V3, **asdict(self)}, sort_keys=True)

    @classmethod
    def from_json(cls, s):
        d = json.loads(s)
        d.pop("version", None)
        return cls(**d)


def hazard_starts_v3(L: int, ends, policy: HazardPolicyV3, doc_key: str, seed: int, decide=None):
    """Same eligibility state machine as segment.hazard_starts / _semantic_update (completed-sentence ends only,
    >= 1 completed sentence in the chunk, cur_len >= min_len, force-close at max_len, no boundary after the last
    token). Inputs: L, sentence-end positions, policy, doc key, seed -- no text, embeddings or semantic boundaries.
    decide(s, remaining, cur_len, t_end) -> bool replaces the draw (fitting / tests only)."""
    h = stable_hash64(doc_key)
    rng = np.random.default_rng([seed, h & 0x7FFFFFFF, h >> 33, 0x4633])
    s = policy.stratum(L)
    a, n = 0, 0
    starts = [0]
    for t_end in np.asarray(ends).tolist():
        while t_end + 1 - a > policy.max_len:
            a += policy.max_len
            starts.append(a)
            n = 0
        if t_end + 1 >= L:
            break
        cur_len = t_end + 1 - a
        if n > 0 and cur_len >= policy.min_len:
            rem = L - (t_end + 1)
            close = bool(decide(s, rem, cur_len, t_end)) if decide is not None else \
                rng.random() < policy.h(s, rem, cur_len)
            if close:
                a = t_end + 1
                starts.append(a)
                n = 1
                continue
        n += 1
    while L - a > policy.max_len:
        a += policy.max_len
        starts.append(a)
    return starts


def replay_v3(L, ends, starts, policy):
    sset = set(int(x) for x in starts[1:])
    opp = []

    def decide(s, rem, cur_len, t_end):
        c = (t_end + 1) in sset
        opp.append((s, policy.end_bin(rem), cur_len, c))
        return c
    return hazard_starts_v3(L, ends, policy, "replay", 0, decide=decide), opp


def simulate_v3(docs, policy, seed):
    return [hazard_starts_v3(L, e, policy, key, seed) for (L, e, key) in docs]


ACCEPT_V3 = {"chunk_count_rel": 0.01, "nonfinal_mean_abs": 0.5, "final_mean_abs": 1.0, "eligible_frac_abs_pp": 0.5,
             "ks_nonfinal": 0.02}


def describe_v3(Ls, starts_list, w=16):
    allc, nonf, fin = [], [], []
    for L, st in zip(Ls, starts_list):
        ls = chunk_lengths(int(L), st)
        allc.append(ls)
        nonf.append(ls[:-1])
        fin.append(int(ls[-1]))
    allc, nonf, fin = np.concatenate(allc), np.concatenate(nonf), np.array(fin)
    return {"n_chunks": int(len(allc)), "all_mean": float(allc.mean()), "nonfinal_mean": float(nonf.mean()),
            "final_mean": float(fin.mean()), "pool_eligible_fraction": float(np.maximum(allc - w, 0).sum() / np.sum(Ls)),
            "all_p10_p50_p90": np.percentile(allc, [10, 50, 90]).tolist(),
            "mass_241_256_pct": 100 * float(((allc >= 241) & (allc <= 256)).mean()), "_nonfinal": nonf}


def _ks(a, b):
    grid = np.arange(1, max(a.max(), b.max()) + 1)
    ca = np.searchsorted(np.sort(a), grid, side="right") / len(a)
    cb = np.searchsorted(np.sort(b), grid, side="right") / len(b)
    return float(np.max(np.abs(ca - cb)))


def compare_v3(d, f):
    m = {"chunk_count_rel": abs(d["n_chunks"] - f["n_chunks"]) / d["n_chunks"],
         "nonfinal_mean_abs": abs(d["nonfinal_mean"] - f["nonfinal_mean"]),
         "final_mean_abs": abs(d["final_mean"] - f["final_mean"]),
         "eligible_frac_abs_pp": 100 * abs(d["pool_eligible_fraction"] - f["pool_eligible_fraction"]),
         "ks_nonfinal": _ks(d["_nonfinal"], f["_nonfinal"])}
    m["pass"] = {k: bool(m[k] < ACCEPT_V3[k]) for k in ACCEPT_V3}
    m["all_pass"] = all(m["pass"].values())
    return m


def _life_table(docs, starts_list, pol):
    """Per (stratum, len bin): realized events / chunks at risk (identical rule for D and F)."""
    nb = (pol.max_len - pol.min_len) // pol.len_bin
    ns = len(pol.strata_edges) - 1
    risk, ev = np.zeros((ns, nb)), np.zeros((ns, nb))
    for (L, _, _), st in zip(docs, starts_list):
        s = pol.stratum(L)
        ls = chunk_lengths(int(L), st)
        for j, ln in enumerate(ls):
            event = j < len(ls) - 1 and ln < pol.max_len
            top = min(nb, max(0, (int(ln) - pol.min_len) // pol.len_bin + 1)) if ln >= pol.min_len else 0
            risk[s, :top] += 1
            if event and ln >= pol.min_len:
                ev[s, (int(ln) - pol.min_len) // pol.len_bin] += 1
    return np.where(risk > 0, ev / np.maximum(risk, 1), np.nan)


def fit_hazard_v3(docs, d_starts, strata_edges, len_bin=8, min_len=64, max_len=256, iters=10, min_cell=50,
                  fit_seeds=(1001, 1002, 1003, 1004), select_seeds=(2001, 2002), damping=0.5, log=None):
    """docs: (L, sentence_ends, doc_key) of TRAINING documents; d_starts: D's starts. Training data only."""
    nb = (max_len - min_len) // len_bin
    ns, ne = len(strata_edges) - 1, len(END_EDGES)
    pol = HazardPolicyV3(list(strata_edges), list(END_EDGES), len_bin, min_len, max_len,
                         np.zeros((ns, ne, nb)).tolist())
    cnt, clo = np.zeros((ns, ne, nb)), np.zeros((ns, ne, nb))
    mismatch = 0
    for (L, ends, _), st in zip(docs, d_starts):
        rep, opp = replay_v3(L, ends, st, pol)
        mismatch += rep != list(st)
        for s, eb, cl, c in opp:
            if cl < max_len:
                b = (cl - min_len) // len_bin
                cnt[s, eb, b] += 1
                clo[s, eb, b] += c
    marg = clo.sum(1) / np.maximum(cnt.sum(1), 1)  # (s, b)
    h = np.where(cnt >= min_cell, clo / np.maximum(cnt, 1), marg[:, None, :])
    pol.hazard = h.tolist()
    Ls = [d[0] for d in docs]
    D = describe_v3(Ls, d_starts)
    lam_d = _life_table(docs, d_starts, pol)

    def score(p):
        cm = [compare_v3(D, describe_v3(Ls, simulate_v3(docs, p, sd))) for sd in select_seeds]
        crit = {k: float(np.mean([c[k] for c in cm])) for k in ACCEPT_V3}
        return max(crit[k] / ACCEPT_V3[k] for k in ACCEPT_V3), crit
    sc, crit = score(pol)
    history = [{"iter": 0, "replay_mismatch_docs": int(mismatch), "cells_pooled": int((cnt < min_cell).sum()),
                "select_score": sc, **{f"sel_{k}": v for k, v in crit.items()}}]
    best, best_score = np.array(pol.hazard), sc
    if log:
        log(history[-1])
    for it in range(1, iters + 1):
        lam_f = np.nanmean([_life_table(docs, simulate_v3(docs, pol, sd), pol) for sd in fit_seeds], axis=0)
        ratio = np.where((lam_f > 0) & np.isfinite(lam_d), lam_d / np.where(lam_f > 0, lam_f, 1), 1.0)
        h = np.clip(np.asarray(pol.hazard) * (ratio ** damping)[:, None, :], 0.0, 1.0)
        pol.hazard = h.tolist()
        sc, crit = score(pol)
        history.append({"iter": it, "select_score": sc, **{f"sel_{k}": v for k, v in crit.items()}})
        if log:
            log(history[-1])
        if sc < best_score:
            best, best_score = h.copy(), sc
    pol.hazard = best.tolist()
    history.append({"selected_score": best_score})
    return pol, history, int(mismatch)
