"""csr-latent-forecast-router-distill-v1 (lfr) shared library: paths, balanced main stream, causal draft outputs,
compressed forecast cache (write + gather), causal forecast features, CF position sample, shuffled-forecast donor map,
phase schedule. Protocol: docs/LATENT_FORECAST_ROUTER_DISTILL_V1_PROTOCOL.md. Data are the frozen fspv2 splits (read-only).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import guard
from .fsp_lib import (EOS_ID, PAD_ID, T_SEQ, N_DOM, Split, pack_rows, plan_steps, make_batch, to_torch,  # noqa: F401
                      build_draft, lr_at, _qkv)
from .model import rope_cos_sin

EXP = "csr-latent-forecast-router-distill-v1"
ART = guard.ROOT / "artifacts/latent-forecast-router-distill-v1"
CFG_DIR = guard.ROOT / "configs/latent-forecast-router-distill-v1"
TMP = guard.ROOT / ".tmp/latent-forecast-router-distill-v1"
PROTOCOL = guard.ROOT / "docs/LATENT_FORECAST_ROUTER_DISTILL_V1_PROTOCOL.md"
STOP_MARKER = ART / "STOP"
CACHE_DIR = TMP / "cache"
RUN_PREFIX = "latent_forecast_router_distill_v1_"
FSP_DATA = guard.ROOT / "artifacts/forecast-specialist-pretraining-v2/data"
DRAFTS = json.loads((guard.ROOT / "artifacts/forecast-specialist-pretraining-v2/drafts.json").read_text())
DRAFT_PATHS = [DRAFTS["shared"]] + list(DRAFTS["domain"])  # slot 0 = general, slots 1..5 = domain d0..d4
N_DRAFT = 6
LN_V = math.log(50304)
M = 1_000_000
BUDGET = {"S1": 49 * M, "S2": 122_500_000, "FINAL": 245 * M}
TRAIN_SPLITS = ("main",)  # training code may only open these (+ calib_search for in-process diagnostics)
N_FEAT_DRAFT = 11
N_FEAT_AGG = 8


def protocol_sha():
    return guard.sha256_file(PROTOCOL)


# ------------------------------------------------------------------ balanced stream
def balanced_order(split: Split, seed=2026):
    """fspv2 fixed permutation restricted per domain, merged by greedy token balancing (fewest tokens first, ties by
    domain index). Every prefix is balanced to within one document per domain."""
    perm = np.random.default_rng(seed).permutation(split.n_docs)
    per = [[int(i) for i in perm if split.domains[i] == d] for d in range(N_DOM)]
    ptr = [0] * N_DOM
    tok = np.zeros(N_DOM, dtype=np.int64)
    out = []
    total = split.n_docs
    while len(out) < total:
        best = None
        for d in range(N_DOM):
            if ptr[d] < len(per[d]) and (best is None or tok[d] < tok[best]):
                best = d
        i = per[best][ptr[best]]
        ptr[best] += 1
        tok[best] += int(split.lengths[i]) - 1
        out.append(i)
    return out


_STREAM = {}


def main_stream(budget):
    """Returns (split, steps, planned valid tokens) for the balanced main stream at a token budget (prefix plans)."""
    key = "rows"
    if key not in _STREAM:
        sp = Split("main")
        order = balanced_order(sp)
        _STREAM[key] = (sp, pack_rows(sp.lengths, order), order)
    sp, rows, _ = _STREAM[key]
    steps, total = plan_steps(rows, sp.lengths, 64, budget)
    return sp, steps, total


def stream_manifest():
    sp, rows, order = (main_stream(BUDGET["S1"]) and _STREAM["rows"])
    h = hashlib.sha256(np.asarray(order, dtype=np.int64).tobytes()).hexdigest()
    out = {"order_sha256": h, "n_docs": len(order), "n_rows": len(rows)}
    for k, b in BUDGET.items():
        _, steps, total = main_stream(b)
        dom = np.zeros(N_DOM, dtype=np.int64)
        for r in steps:
            for row in r:
                for d in row:
                    dom[sp.domains[d]] += sp.lengths[d] - 1
        out[k] = {"budget": b, "n_steps": len(steps), "valid_tokens": total, "per_domain": dom.tolist(),
                  "steps_sha256": hashlib.sha256(json.dumps(steps).encode()).hexdigest()}
    return out


# ------------------------------------------------------------------ phases
def phase_at(frac, schedule="40/20/40", lam_max=0.9):
    """Returns (lam, phase, teacher_scale). P1: lam_max; P2: linear lam_max->0; P3: 0."""
    a, b, _ = [int(x) for x in schedule.split("/")]
    p1, p2 = a / 100.0, (a + b) / 100.0
    if frac < p1:
        return lam_max, 1, 1.0
    if frac < p2:
        u = (frac - p1) / (p2 - p1)
        lam = lam_max * (1.0 - u)
        return lam, 2, lam / lam_max
    return 0.0, 3, 0.0


def phase_end_frac(schedule):
    a, b, _ = [int(x) for x in schedule.split("/")]
    return (a + b) / 100.0


# ------------------------------------------------------------------ hashing helpers (CF sample)
def _splitmix(x):
    x = (x + np.uint64(0x9E3779B97F4A7C15)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    z = x
    z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    return z ^ (z >> np.uint64(31))


def cf_uniform(gdoc, positions, salt=20261111):
    """Deterministic uniform in [0,1) per (main document id, position): the fixed preregistered CF sample."""
    with np.errstate(over="ignore"):
        g = np.asarray(gdoc, dtype=np.int64).astype(np.uint64)
        p = np.asarray(positions, dtype=np.int64).astype(np.uint64)
        h = _splitmix(_splitmix(g * np.uint64(1_000_003) + np.uint64(salt)) ^ p)
    return (h >> np.uint64(11)).astype(np.float64) / float(1 << 53)


# ------------------------------------------------------------------ causal draft outputs (cache + diagnostics)
@torch.no_grad()
def draft_outputs(draft, ids, lengths, H=8, topk1=16, diag_topk=0):
    """Rows hold ONE document each from position 0 (right padded). For every prefix x_<=t: the full h=1 distribution
    (top-`topk1` ids/probs, entropy), greedy rollout argmax ids at steps 1..H (step k predicts x_{t+k}), and
    score(t) = log p(x_t | x_<t). Real-prefix attention: causal flash attention over real keys <= t merged by
    log-sum-exp with the per-query rollout branch keys; real tokens after t never enter position t.
    diag_topk > 0 additionally returns top-`diag_topk` ids at every rollout step (diagnostics only)."""
    B, T = ids.shape
    cfg = draft.cfg
    hd = cfg.d_model // cfg.n_head
    dev = ids.device
    dt = torch.bfloat16 if dev.type == "cuda" else draft.tok_emb.weight.dtype
    positions = torch.arange(T, device=dev).expand(B, T)
    W = draft.tok_emb.weight.to(dt)
    x = draft.tok_emb(ids).to(dt)
    cos, sin = rope_cos_sin(positions, hd, cfg.rope_base, dt)
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    Kr, Vr = [], []
    with torch.autocast("cuda" if dev.type == "cuda" else "cpu", dtype=dt, enabled=dev.type == "cuda"):
        for blk in draft.blocks:
            q, k, v = _qkv(blk, x, cos, sin)
            q, k, v = q.to(dt), k.to(dt), v.to(dt)
            Kr.append(k)
            Vr.append(v)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + blk.attn.proj(y.transpose(1, 2).reshape(B, T, -1)).to(dt)
            x = x + blk.ffn(blk.norm2(x)).to(dt)
        hN = draft.norm_f(x).to(dt).reshape(B * T, -1)
    out = {}
    tid, tp, ent, lse_all, sc_next, am = [], [], [], [], [], []
    nxt = torch.cat([ids[:, 1:], ids[:, :1]], 1).reshape(-1)
    for s in range(0, B * T, 8192):
        lg = (hN[s:s + 8192] @ W.t()).float()
        lse = torch.logsumexp(lg, -1)
        p = torch.exp(lg - lse.unsqueeze(-1))
        ent.append(lse - (p * lg).sum(-1))
        tv = torch.topk(lg, topk1, -1)
        tid.append(tv.indices)
        tp.append(torch.exp(tv.values - lse.unsqueeze(-1)))
        sc_next.append(lg.gather(1, nxt[s:s + 8192].view(-1, 1)).squeeze(1) - lse)
        am.append(tv.indices[:, 0])
    out["top_ids"] = torch.cat(tid).view(B, T, topk1)
    out["top_p"] = torch.cat(tp).view(B, T, topk1)
    out["ent1"] = torch.cat(ent).view(B, T)
    valid = torch.arange(T, device=dev).view(1, T) < lengths.view(B, 1)
    scn = torch.cat(sc_next).view(B, T)
    score = torch.zeros(B, T, device=dev)
    score[:, 1:] = scn[:, :-1] * valid[:, 1:]
    out["score"] = score
    cur = torch.cat(am).view(B, T)
    props = [cur]
    diag = [out["top_ids"][..., :diag_topk]] if diag_topk else None
    nh = cfg.n_head
    Kp = [torch.zeros(B, nh, T, H - 1, hd, device=dev, dtype=torch.float32) for _ in draft.blocks]
    Vp = [torch.zeros(B, nh, T, H - 1, hd, device=dev, dtype=torch.float32) for _ in draft.blocks]
    scale = 1.0 / math.sqrt(hd)
    for j in range(2, H + 1):
        x = draft.tok_emb(cur).to(dt)
        c2, s2 = rope_cos_sin(positions + (j - 1), hd, cfg.rope_base, dt)
        c2, s2 = c2.unsqueeze(1), s2.unsqueeze(1)
        n = j - 1
        with torch.autocast("cuda" if dev.type == "cuda" else "cpu", dtype=dt, enabled=dev.type == "cuda"):
            for li, blk in enumerate(draft.blocks):
                q, k, v = _qkv(blk, x, c2, s2)
                q, k, v = q.to(dt), k.to(dt), v.to(dt)
                Kp[li][..., n - 1, :] = k.float()
                Vp[li][..., n - 1, :] = v.float()
                if dev.type == "cuda":
                    o = torch.ops.aten._scaled_dot_product_flash_attention(q, Kr[li], Vr[li], 0.0, True, False,
                                                                         scale=scale)
                    yr, lse_r = o[0].float(), o[1].float()
                else:  # CPU reference path (tests): explicit causal softmax over real keys
                    sr = torch.matmul(q.float(), Kr[li].float().transpose(-1, -2)) * scale
                    cm = torch.ones(T, T, dtype=torch.bool, device=dev).tril()
                    sr = sr.masked_fill(~cm, float("-inf"))
                    lse_r = torch.logsumexp(sr, -1)
                    yr = torch.matmul(torch.softmax(sr, -1), Vr[li].float())
                kp, vp = Kp[li][..., :n, :], Vp[li][..., :n, :]
                sp = torch.einsum("bhtd,bhtnd->bhtn", q.float(), kp) * scale
                mx = torch.maximum(lse_r, sp.max(-1).values)
                wr = torch.exp(lse_r - mx)
                wp = torch.exp(sp - mx.unsqueeze(-1))
                y = (yr * wr.unsqueeze(-1) + torch.einsum("bhtn,bhtnd->bhtd", wp, vp)) / \
                    (wr + wp.sum(-1)).unsqueeze(-1)
                x = x + blk.attn.proj(y.to(dt).transpose(1, 2).reshape(B, T, -1)).to(dt)
                x = x + blk.ffn(blk.norm2(x)).to(dt)
            hN = draft.norm_f(x).to(dt).reshape(B * T, -1)
        nxt_ids, dtop = [], []
        for s in range(0, B * T, 8192):
            lg = hN[s:s + 8192] @ W.t()
            if diag_topk:
                ti = torch.topk(lg.float(), diag_topk, -1).indices
                dtop.append(ti)
                nxt_ids.append(ti[:, 0])
            else:
                nxt_ids.append(lg.argmax(-1))
        cur = torch.cat(nxt_ids).view(B, T)
        props.append(cur)
        if diag_topk:
            diag.append(torch.cat(dtop).view(B, T, diag_topk))
    out["roll"] = torch.stack(props, -1)  # (B,T,H): roll[...,k-1] predicts x_{t+k}
    if diag_topk:
        out["roll_topk"] = torch.stack(diag, -2)  # (B,T,H,k)
    return out


def compress_ensemble(top_ids, top_p, ent1, score, roll4, roll8):
    """Torch (any device). top_ids (N,D,16) long, top_p (N,D,16) float, ent1/score (N,D), roll4/roll8 (N,D) long
    (greedy rollout ids at steps 4 and 8). Returns (dict of compressed numpy arrays (section 5), coverage stats)."""
    N, D, K = top_ids.shape
    flat_ids = top_ids.reshape(N, D * K)
    flat_p = top_p.reshape(N, D * K).float() / D
    sid, perm = torch.sort(flat_ids, dim=1, stable=True)
    sp_ = flat_p.gather(1, perm)
    newgrp = torch.ones_like(sid, dtype=torch.bool)
    newgrp[:, 1:] = sid[:, 1:] != sid[:, :-1]
    grp = torch.cumsum(newgrp.long(), 1) - 1
    mass = torch.zeros(N, D * K, device=top_ids.device).scatter_add_(1, grp, sp_)
    gid = torch.zeros(N, D * K, dtype=torch.long, device=top_ids.device).scatter_(1, grp, sid)
    tv = torch.topk(mass, 4, 1)  # ties -> lower group index = lower token id (deterministic)
    ens_ids = gid.gather(1, tv.indices)
    ens_p = tv.values
    am = top_ids[:, :, 0]
    code1 = torch.full((N, D), 4, dtype=torch.long, device=top_ids.device)
    for k in range(3, -1, -1):
        code1 = torch.where(am == ens_ids[:, k:k + 1], torch.full_like(code1, k), code1)
    cands, codes = {}, {}
    slot = torch.arange(D, device=top_ids.device).view(1, D)
    for h, r in ((4, roll4), (8, roll8)):
        cnt = (r[:, :, None] == r[:, None, :]).sum(-1)  # (N,D) frequency of each slot's id
        first = torch.argmax(cnt * 16 - slot, 1)  # most frequent; earliest slot (shared draft first) on ties
        c1 = r.gather(1, first.view(-1, 1)).squeeze(1)
        cnt2 = torch.where(r == c1[:, None], torch.full_like(cnt, -1), cnt)
        sec = torch.argmax(cnt2 * 16 - slot, 1)
        c2 = torch.where(cnt2.max(1).values >= 0, r.gather(1, sec.view(-1, 1)).squeeze(1), c1)
        cands[h] = torch.stack([c1, c2], 1)
        codes[h] = torch.where(r == c1[:, None], 0, torch.where(r == c2[:, None], 1, 2))
    q8 = lambda x: torch.clamp(torch.round(x), 0, 255).to(torch.uint8).cpu().numpy()
    out = {"ens_ids": ens_ids.to(torch.int32).cpu().numpy().astype(np.uint16), "ens_p": q8(ens_p * 255),
           "cand4": cands[4].to(torch.int32).cpu().numpy().astype(np.uint16),
           "cand8": cands[8].to(torch.int32).cpu().numpy().astype(np.uint16),
           "code": (code1 | (codes[4] << 3) | (codes[8] << 5)).to(torch.uint8).cpu().numpy(),
           "ent1": q8(ent1.float() / LN_V * 255), "p1": q8(top_p[:, :, 0].float() * 255),
           "score": q8(torch.clamp(-score.float(), max=25.5) * 10)}
    stats = {"n": int(N), "h1_argmax_in_ens_top4": float((code1 < 4).float().mean()),
             "h4_in_cands": float((codes[4] < 2).float().mean()), "h8_in_cands": float((codes[8] < 2).float().mean()),
             "ens_top4_mass_mean": float(ens_p.sum(1).mean()),
             "ens_mass_lower_bound_gap_max": float(top_p[:, :, -1].max())}
    return out, stats


FIELDS = {"ens_ids": (4, np.uint16), "ens_p": (4, np.uint8), "cand4": (2, np.uint16), "cand8": (2, np.uint16),
          "code": (6, np.uint8), "ent1": (6, np.uint8), "p1": (6, np.uint8), "score": (6, np.uint8)}


class ForecastCache:
    """Reader of the compressed cache. index.json: {chunk: {"docs": [split doc ids], "offsets": [...]}} + per chunk
    directory <cache>/<split>/chunk_XXXX/<field>.npy (memory-mapped). Only chunks with a COMPLETE marker are used."""

    def __init__(self, root=CACHE_DIR, split="main"):
        self.root = Path(root) / split
        self.loc = {}
        self.arr = {}
        if not self.root.exists():
            return
        for cdir in sorted(self.root.glob("chunk_*")):
            if not (cdir / "COMPLETE.json").exists():
                continue
            meta = json.loads((cdir / "COMPLETE.json").read_text())
            c = cdir.name
            self.arr[c] = {k: np.load(cdir / f"{k}.npy", mmap_mode="r") for k in FIELDS}
            for d, o in zip(meta["docs"], meta["offsets"]):
                self.loc[int(d)] = (c, int(o))

    def has(self, doc):
        return int(doc) in self.loc

    def doc(self, doc, L):
        c, o = self.loc[int(doc)]
        return {k: np.asarray(self.arr[c][k][o:o + L]) for k in FIELDS}


def shuffle_donors(steps_rows, lengths):
    """SHUFFLED-FORECAST donor map for one optimizer step: all documents of the step sorted by (length, doc id),
    cyclic shift by one -> donor. A derangement whenever the step holds >= 2 documents."""
    docs = sorted({int(d) for row in steps_rows for d in row}, key=lambda d: (int(lengths[d]), d))
    if len(docs) < 2:
        return {}
    return {docs[i]: docs[(i + 1) % len(docs)] for i in range(len(docs))}


def gather_raw(rows, split: Split, cache: ForecastCache, donors=None):
    """Compressed cache fields + source tokens laid out like make_batch(rows) (B,T,...). Pads are zero. With donors,
    every document takes the (tiled) cached fields AND tokens of its donor document (position aligned)."""
    B = len(rows)
    out = {k: np.zeros((B, T_SEQ, w), dt) for k, (w, dt) in FIELDS.items()}
    src_tok = np.full((B, T_SEQ), PAD_ID, np.int64)
    for b, row in enumerate(rows):
        t = 0
        for di in row:
            L = int(split.lengths[di])
            s = donors.get(int(di), int(di)) if donors else int(di)
            Ls = int(split.lengths[s])
            if not cache.has(s):
                raise guard.GuardError(f"doc {s} not in forecast cache")
            f = cache.doc(s, Ls)
            tok = split.doc(s)
            idx = np.arange(L) % Ls
            for k in FIELDS:
                v = f[k]
                out[k][b, t:t + L] = (v if v.ndim == 2 else v[:, None])[idx]
            src_tok[b, t:t + L] = tok[idx]
            t += L
    out["src_tok"] = src_tok
    return out


def build_features(raw, doc_ids, positions, halflife=64.0):
    """Causal forecast features from gathered raw fields (torch tensors on device; (B,T,...)).
    Returns dict: dfeat (B,T,6,11), afeat (B,T,8), ens_ids (B,T,4) long, ens_p (B,T,4), cand4/cand8 (B,T,2) long,
    vote4/vote8 (B,T,2). Only fields at positions <= t of the same document enter position t."""
    from .lfr_model import causal_ema
    ens_ids = raw["ens_ids"].long()
    ens_p = raw["ens_p"].float() / 255.0
    code = raw["code"].long()  # (B,T,6)
    c1, c4, c8 = code & 7, (code >> 3) & 3, (code >> 5) & 3
    ent = raw["ent1"].float() / 255.0  # (B,T,6) in units of lnV
    p1 = raw["p1"].float() / 255.0
    score = -(raw["score"].float() / 10.0)  # log p_d(x_t | x_<t), clipped at -25.5
    B, T, D = code.shape
    valid = (doc_ids >= 0)
    same_prev = torch.zeros_like(valid)
    same_prev[:, 1:] = (doc_ids[:, 1:] == doc_ids[:, :-1]) & valid[:, 1:]
    # past h1 hit of draft d: its argmax at t-1 (ensemble id at index code1, if covered) equals the real x_t
    src = raw["src_tok"].long()
    prev_ids = torch.zeros_like(ens_ids)
    prev_ids[:, 1:] = ens_ids[:, :-1]
    prev_c1 = torch.full_like(c1, 4)
    prev_c1[:, 1:] = c1[:, :-1]
    pred_prev = prev_ids.gather(2, prev_c1.clamp(max=3))  # (B,T,6)
    hit = ((pred_prev == src.unsqueeze(-1)) & (prev_c1 < 4) & same_prev.unsqueeze(-1)).float()
    sc_rel = score - score.mean(-1, keepdim=True)
    sc_rel = sc_rel * (positions > 0).unsqueeze(-1).float()
    ema_in = torch.cat([sc_rel, hit], -1)  # (B,T,12)
    ema = causal_ema(ema_in, doc_ids, positions, halflife)
    sc_e, hit_e = ema[..., :D] / 4.0, ema[..., D:]
    oh1 = (c1 == 0).float()  # agrees with ensemble top-1
    oh4 = (c4 == 0).float()  # agrees with horizon-4 plurality
    oh8 = (c8 == 0).float()
    dfeat = torch.stack([ent, p1, oh1, (c1 < 4).float(), oh4, (c4 < 2).float(), oh8, (c8 < 2).float(),
                         (score / 10.0).clamp(min=-2.55), sc_e, hit_e], -1)  # (B,T,6,11)
    agree1, agree4, agree8 = oh1.mean(-1), oh4.mean(-1), oh8.mean(-1)
    spread = sc_e.max(-1).values - sc_e.min(-1).values
    afeat = torch.stack([ens_p[..., 0], ent.mean(-1), agree1, agree4, agree8, 1.0 - agree1, spread,
                         torch.log1p(positions.float()) / 7.0], -1)
    vote = lambda c: torch.stack([(c == 0).float().mean(-1), (c == 1).float().mean(-1)], -1)
    m = valid.unsqueeze(-1).float()
    return {"dfeat": dfeat * m.unsqueeze(-1), "afeat": afeat * m, "ens_ids": ens_ids, "ens_p": ens_p * m,
            "cand4": raw["cand4"].long(), "cand8": raw["cand8"].long(), "vote4": vote(c4) * m, "vote8": vote(c8) * m}


def save_cache_chunk(cdir: Path, arrays: dict, docs, offsets, extra):
    """Atomic chunk write: tmp dir -> files fsynced -> verify -> COMPLETE marker -> rename."""
    cdir = Path(cdir)
    tmp = cdir.parent / f".{cdir.name}.tmp"
    if cdir.exists():
        raise guard.GuardError(f"cache chunk {cdir} exists")
    if tmp.exists():
        for p in tmp.iterdir():
            p.unlink()
    tmp.mkdir(parents=True, exist_ok=True)
    shas = {}
    for k, v in arrays.items():
        p = tmp / f"{k}.npy"
        with open(p, "wb") as f:
            np.save(f, v)
            f.flush()
            os.fsync(f.fileno())
        shas[k] = guard.sha256_file(p)
        ld = np.load(p, mmap_mode="r")
        assert ld.shape == v.shape and ld.dtype == v.dtype
    meta = {"docs": [int(d) for d in docs], "offsets": [int(o) for o in offsets], "sha256": shas,
            "n_tokens": int(next(iter(arrays.values())).shape[0]), "time": guard.now(), **extra}
    with open(tmp / "COMPLETE.json", "w") as f:
        json.dump(meta, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, cdir)
    fd = os.open(str(cdir.parent), os.O_RDONLY)
    os.fsync(fd)
    os.close(fd)
    return meta
