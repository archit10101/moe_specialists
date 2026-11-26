"""csr-domain-forecast-curriculum-v1 shared library: paths, split locks, row/step plans (phase-aware, exact domain
balance), batches with forecasts / domains / per-row strength, evaluation and specialisation metrics.
New module; existing modules are imported, never modified."""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from . import guard
from .data import EOS_ID, PAD_ID, TokenSplit, pack_rows

EXP_ID = "csr-domain-forecast-curriculum-v1"
ART = guard.ROOT / "artifacts/domain-forecast-curriculum-v1"
DATA = ART / "data"
CACHE = ART / "cache"
CFG_DIR = guard.ROOT / "configs/domain-forecast-curriculum-v1"
ST = guard.ROOT / ".tmp/domain-forecast-curriculum-v1"
PREREG = guard.ROOT / "docs/DOMAIN_FORECAST_CURRICULUM_V1_PREREG.md"
FROZEN = ART / "frozen.json"
T_SEQ = 1024
N_DOM = 5
ROWS_PER_STEP = 32
MAIN_TOKENS = 112_500_000
PHASE_TOKENS = {1: 45_000_000, 2: 11_250_000, 3: 56_250_000}
CKPT_FRACS = {"p20": 0.20, "p40": 0.40, "p50": 0.50, "p75": 0.75, "p100": 1.00}
MON_EVERY = 5_625_000
MON_ROWS_PER_DOM = 64
ABL_ROWS_PER_DOM = 48


class SplitLocked(RuntimeError):
    pass


def prereg_sha():
    return guard.sha256_file(PREREG)


def manifest():
    return json.loads((DATA / "manifest.json").read_text())


def deviation(kind, detail, **extra):
    ART.mkdir(parents=True, exist_ok=True)
    with open(ART / "deviations.jsonl", "a") as f:
        f.write(json.dumps({"time": guard.now(), "kind": kind, "detail": detail, **extra}) + "\n")
    guard.ledger("DFC_DEVIATION", f"{kind}: {detail[:300]}")


def code_hashes():
    files = sorted(list(guard.ROOT.glob("dfc_*.py")) + list((guard.ROOT / "src/csr").glob("dfc_*.py")) +
                   [guard.ROOT / f"src/csr/{x}.py" for x in ("model", "data", "train_lib", "guard")])
    return {str(p.relative_to(guard.ROOT)): guard.sha256_file(p) for p in files}


# ------------------------------------------------------------------ split access (validation locked until frozen)
def load_split(split):
    if split not in ("main", "draft", "calib", "val"):
        raise SplitLocked(f"split {split!r} refused (locked / not part of dfcv1)")
    if split == "val" and not FROZEN.exists():
        raise SplitLocked("validation locked until frozen.json exists")
    sp = TokenSplit.load(DATA, split)
    dj = json.loads((DATA / f"{split}_docs.json").read_text())
    doms = np.asarray(dj["domains"], dtype=np.int64)
    phase = np.asarray(dj["phase"], dtype=np.int64) if "phase" in dj else None
    return sp, doms, phase


def load_cache(split, source):
    """source in {'correct', 'deranged'}; uint16 (n_tokens, 4) memmap aligned with the split's token offsets."""
    if split not in ("main", "calib"):
        raise SplitLocked(f"no forecast cache for {split}")
    p = CACHE / f"{split}_{source}.npy"
    return np.load(p, mmap_mode="r")


class MemSplit:
    """In-memory TokenSplit-compatible container (derived splits: ALL-DATA, COMPUTE)."""

    def __init__(self, tokens, offsets, doc_ids):
        self.tokens, self.offsets, self.doc_ids = tokens, offsets, doc_ids

    @property
    def n_docs(self):
        return len(self.offsets) - 1

    def doc(self, i):
        return self.tokens[self.offsets[i]:self.offsets[i + 1]].astype(np.int64)

    def lengths(self):
        return np.diff(self.offsets)


def interleave(per_dom_rows):
    """Interleave domains' row lists by fractional position (i + 0.5) / n_d, ties by domain id."""
    keys = []
    for d, rows in enumerate(per_dom_rows):
        n = len(rows)
        keys += [((i + 0.5) / n, d, i) for i in range(n)]
    keys.sort()
    return [per_dom_rows[d][i] for _, d, i in keys], [d for _, d, _ in keys]


def valid_of(rows, lengths):
    return int(sum(int(lengths[r].sum()) - len(r) for r in rows))


class Plan:
    """A training plan: split (+ per-doc domains, optional cache), rows, per-row strength, steps."""

    def __init__(self, split, doms, rows, row_strength, row_phase, cache=None, name=""):
        self.split, self.doms, self.rows, self.cache, self.name = split, doms, rows, cache, name
        self.row_strength = np.asarray(row_strength, dtype=np.float64)
        self.row_phase = np.asarray(row_phase, dtype=np.int64)
        L = split.lengths()
        self.row_valid = np.asarray([int(L[r].sum()) - len(r) for r in rows], dtype=np.int64)
        self.steps = [list(range(i, min(i + ROWS_PER_STEP, len(rows)))) for i in range(0, len(rows), ROWS_PER_STEP)]
        self.total = int(self.row_valid.sum())

    def step_tokens(self, si):
        return int(self.row_valid[self.steps[si]].sum())

    def fingerprint(self):
        return hashlib.sha256(json.dumps([self.rows, self.row_strength.round(12).tolist()]).encode()).hexdigest()

    def per_domain_phase_tokens(self):
        L = self.split.lengths()
        out = {}
        for r, ph in zip(self.rows, self.row_phase):
            for di in r:
                k = f"p{int(ph)}_d{int(self.doms[di])}"
                out[k] = out.get(k, 0) + int(L[di]) - 1
        return out


def main_plan(seed, forecast_source=None):
    sp, doms, phase = load_split("main")
    L = sp.lengths()
    order_seed = 1000 + seed
    rows, rstr, rph = [], [], []
    for ph in (1, 2, 3):
        per = []
        for d in range(N_DOM):
            idx = np.nonzero((doms == d) & (phase == ph))[0]
            idx = idx[np.random.default_rng([order_seed, d, ph]).permutation(len(idx))]
            per.append(pack_rows(L, idx.tolist(), T_SEQ))
        prow, _ = interleave(per)
        if ph == 1:
            s = [1.0] * len(prow)
        elif ph == 3:
            s = [0.0] * len(prow)
        else:
            tot = valid_of(prow, L)
            assert tot == PHASE_TOKENS[2], tot
            s, cum = [], 0
            for r in prow:
                v = int(L[r].sum()) - len(r)
                s.append(1.0 - (cum + 0.5 * v) / tot)
                cum += v
        rows += prow
        rstr += s
        rph += [ph] * len(prow)
    cache = load_cache("main", forecast_source) if forecast_source else None
    p = Plan(sp, doms, rows, rstr, rph, cache, f"main_s{seed}")
    assert p.total == MAIN_TOKENS, p.total
    return p


def _combined_alldata():
    dsp, ddoms, _ = load_split("draft")
    msp, mdoms, _ = load_split("main")
    tokens = np.concatenate([np.asarray(dsp.tokens), np.asarray(msp.tokens)])
    offsets = np.concatenate([dsp.offsets, msp.offsets[1:] + dsp.offsets[-1]])
    return MemSplit(tokens, offsets, list(dsp.doc_ids) + list(msp.doc_ids)), np.concatenate([ddoms, mdoms])


def alldata_streams(seed, comb, doms):
    order_seed = 1000 + seed
    out = []
    for d in range(N_DOM):
        idx = np.nonzero(doms == d)[0]
        out.append(idx[np.random.default_rng([order_seed, d, 99]).permutation(len(idx))])
    return out


def alldata_plan(seed):
    comb, doms = _combined_alldata()
    L = comb.lengths()
    per = [pack_rows(L, s.tolist(), T_SEQ) for s in alldata_streams(seed, comb, doms)]
    rows, _ = interleave(per)
    p = Plan(comb, doms, rows, [0.0] * len(rows), [0] * len(rows), None, f"alldata_s{seed}")
    assert p.total == MAIN_TOKENS + 30_000_000, p.total
    return p


def compute_plan(seed, n_tokens):
    """Per domain: the first n_tokens/5 valid tokens of that seed's ALL-DATA stream (last doc truncated + EOS),
    cycling from the start only if required."""
    assert n_tokens % (5 * 1000) == 0
    q = n_tokens // N_DOM
    comb, doms = _combined_alldata()
    streams = alldata_streams(seed, comb, doms)
    toks, offs, ids, ndoms = [], [0], [], []
    per_dom_docidx = []
    cycles = 0
    for d in range(N_DOM):
        n, mine, c = 0, [], 0
        while n < q:
            for di in streams[d]:
                a = comb.doc(di).astype(np.uint16)
                need = q - n
                if len(a) - 1 > need:
                    a = np.concatenate([a[:need], np.asarray([EOS_ID], np.uint16)])
                mine.append(len(ids))
                toks.append(a)
                offs.append(offs[-1] + len(a))
                ids.append(comb.doc_ids[di] + (f"#c{c}" if c else ""))
                ndoms.append(d)
                n += len(a) - 1
                if n >= q:
                    break
            c += 1
        cycles = max(cycles, c)
        per_dom_docidx.append(mine)
    sp = MemSplit(np.concatenate(toks), np.asarray(offs, dtype=np.int64), ids)
    L = sp.lengths()
    per = [pack_rows(L, m, T_SEQ) for m in per_dom_docidx]
    rows, _ = interleave(per)
    p = Plan(sp, np.asarray(ndoms), rows, [0.0] * len(rows), [0] * len(rows), None, f"compute_s{seed}")
    assert p.total == n_tokens, (p.total, n_tokens)
    p.cycles = cycles
    return p


def plan_for(cfg):
    c, seed = cfg["condition"], cfg["seed"]
    if c == "TOKEN-ALL-DATA":
        return alldata_plan(seed)
    if c == "TOKEN-COMPUTE":
        return compute_plan(seed, cfg["train"]["token_budget"])
    from .dfc_model import CONDITIONS
    src = CONDITIONS[c][3]
    p = main_plan(seed, src)
    if not CONDITIONS[c][0]:
        p.row_strength[:] = 0.0
    return p


# ------------------------------------------------------------------ batches
def make_batch(rows, split, doms, cache=None, T=T_SEQ):
    B = len(rows)
    a = {k: np.full((B, T), fill, dtype=np.int64) for k, fill in
         (("input_ids", PAD_ID), ("doc_ids", -1), ("positions", 0), ("targets", -100), ("gdoc", -1),
          ("domain", -1))}
    if cache is not None:
        a["fc"] = np.zeros((B, T, 4), dtype=np.int64)
    off = split.offsets
    for b, row in enumerate(rows):
        t = 0
        for local, di in enumerate(row):
            ids = split.doc(di)
            n = len(ids)
            sl = slice(t, t + n)
            a["input_ids"][b, sl] = ids
            a["doc_ids"][b, sl] = local
            a["positions"][b, sl] = np.arange(n)
            a["targets"][b, t:t + n - 1] = ids[1:]
            a["gdoc"][b, sl] = di
            a["domain"][b, sl] = doms[di]
            if cache is not None:
                a["fc"][b, sl] = cache[off[di]:off[di] + n]
            t += n
    return a


def to_torch(batch, device):
    import torch
    return {k: torch.as_tensor(v, device=device) for k, v in batch.items()}


# ------------------------------------------------------------------ evaluation rows
def eval_rows(split, doms, rows_per_dom=None):
    L = split.lengths()
    rows, rdom = [], []
    for d in range(N_DOM):
        r = pack_rows(L, np.nonzero(doms == d)[0].tolist(), T_SEQ)
        if rows_per_dom is not None:
            r = r[:rows_per_dom]
        rows += r
        rdom += [d] * len(r)
    return rows, rdom


# ------------------------------------------------------------------ metrics
def mi_from_counts(C):
    """C (D, E) assignment counts -> MI(domain; expert) in nats, H(domain), H(domain|expert) per expert."""
    C = np.asarray(C, dtype=np.float64)
    P = C / max(C.sum(), 1.0)
    pd, pe = P.sum(1, keepdims=True), P.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        mi = float(np.nansum(np.where(P > 0, P * np.log(P / (pd * pe)), 0.0)))
        hd = float(-np.nansum(np.where(pd > 0, pd * np.log(pd), 0.0)))
        pde = np.where(pe > 0, P / pe, 0.0)
        hde = -np.nansum(np.where(pde > 0, pde * np.log(pde), 0.0), axis=0)
    return mi, hd, hde


def load_stats(counts_e):
    c = np.asarray(counts_e, dtype=np.float64)
    share = c / max(c.sum(), 1.0)
    return {"max_share": float(share.max()), "min_share": float(share.min()), "cv": float(share.std() / share.mean()),
            "unused": int((c == 0).sum()), "lt_0p1pct": int((share < 0.001).sum())}


def bootstrap_diff(dnll_a, dn_a, dnll_b, n_tok, B=2000, seed=0):
    """Token-weighted paired difference (a - b) over documents; returns mean and 95% CI."""
    a, b, n = np.asarray(dnll_a), np.asarray(dnll_b), np.asarray(n_tok, dtype=np.float64)
    diff = (a - b)
    point = float(diff.sum() / n.sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(n), size=(B, len(n)))
    bs = diff[idx].sum(1) / n[idx].sum(1)
    return {"mean": point, "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "n_docs": len(n)}


def finite(x):
    return x is not None and math.isfinite(x)


def now_s():
    return time.time()
