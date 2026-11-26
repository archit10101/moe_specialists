"""csr-forecast-specialist-pretraining-v2 shared library: data, batching, draft model + causal rollouts, cached forecast
features, curriculum schedules. New module (protocol docs/FORECAST_SPECIALIST_PRETRAINING_V2_PROTOCOL.md)."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import guard
from .model import CSRModel, ModelConfig, apply_rope, rope_cos_sin

EXP = "csr-forecast-specialist-pretraining-v2"
ART = guard.ROOT / "artifacts/forecast-specialist-pretraining-v2"
DATA = ART / "data"
CFG_DIR = guard.ROOT / "configs/forecast-specialist-pretraining-v2"
PROTOCOL = guard.ROOT / "docs/FORECAST_SPECIALIST_PRETRAINING_V2_PROTOCOL.md"
EOS_ID, PAD_ID = 0, 1
T_SEQ = 1024
N_DOM = 5
OPEN_SPLITS = ("draft_shared", "draft_cont", "main", "calib_search", "calib_sel", "val")
N_SCALAR = 21
LN_V = math.log(50304)


def protocol_sha():
    return guard.sha256_file(PROTOCOL)


def u_of(salt, h):
    return int.from_bytes(hashlib.sha256(f"{salt}|{h}".encode()).digest()[:8], "little") / 2**64


# ------------------------------------------------------------------ data
class Split:
    def __init__(self, name):
        if name not in OPEN_SPLITS:
            raise guard.GuardError(f"split {name!r} is not loadable (locked test or unknown)")
        self.name = name
        self.tokens = np.load(DATA / f"{name}_tokens.npy", mmap_mode="r")
        self.offsets = np.load(DATA / f"{name}_offsets.npy")
        d = json.loads((DATA / f"{name}_docs.json").read_text())
        self.ids, self.domains, self.papers = d["ids"], np.asarray(d["domains"]), d["papers"]
        self.exclusive = np.asarray(d.get("exclusive", [True] * len(self.ids)))
        self.seg_index = np.asarray([int(i.split("#")[1]) for i in self.ids])
        self.lengths = np.diff(self.offsets)
        self.n_docs = len(self.ids)
        # fixed per-paper control maps (protocol 4/5): wrong draft w(doc), shuffled label s(doc)
        self.wrong = np.asarray([(d_ + 1 + int(u_of("fspv2-wrong", p) * 4)) % N_DOM
                                 for d_, p in zip(self.domains, self.papers)])
        self.shuffled = np.asarray([d_ if u_of("fspv2-shuffle", p) < 0.2 else w
                                    for d_, p, w in zip(self.domains, self.papers, self.wrong)])

    def doc(self, i):
        return np.asarray(self.tokens[self.offsets[i]:self.offsets[i + 1]], dtype=np.int64)

    def valid_tokens(self, idx=None):
        L = self.lengths if idx is None else self.lengths[idx]
        return int((L - 1).sum())


def pack_rows(lengths, order, T=T_SEQ):
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


def main_order(split: Split, seed=2026):
    return np.random.default_rng(seed).permutation(split.n_docs).tolist()


def plan_steps(rows, lengths, rows_per_step, budget):
    """Smallest number of whole steps whose valid tokens reach budget. Returns list of row-lists and token count."""
    steps, total = [], 0
    for s in range(0, len(rows) - rows_per_step + 1, rows_per_step):
        if total >= budget:
            break
        rr = rows[s:s + rows_per_step]
        steps.append(rr)
        total += int(sum(lengths[d] - 1 for r in rr for d in r))
    if total < budget:
        raise ValueError(f"not enough data: {total} < {budget}")
    return steps, total


def make_batch(rows, split: Split, label_mode="true"):
    """rows: list of (split, doc) pairs lists or doc lists for `split`. Returns numpy dict (B, T)."""
    B = len(rows)
    a = {k: np.full((B, T_SEQ), f, dtype=np.int64) for k, f in
         (("input_ids", PAD_ID), ("targets", -100), ("doc_ids", -1), ("positions", 0), ("domain", -1),
          ("true_domain", -1), ("gdoc", -1), ("seq_id", -1), ("seg_index", 0))}
    sid = 0
    for b, row in enumerate(rows):
        t = 0
        for local, di in enumerate(row):
            ids = split.doc(di)
            L = len(ids)
            a["input_ids"][b, t:t + L] = ids
            a["targets"][b, t:t + L - 1] = ids[1:]
            a["doc_ids"][b, t:t + L] = local
            a["positions"][b, t:t + L] = np.arange(L)
            lab = {"true": split.domains, "shuffled": split.shuffled}[label_mode][di]
            a["domain"][b, t:t + L] = lab
            a["true_domain"][b, t:t + L] = split.domains[di]
            a["gdoc"][b, t:t + L] = di
            a["seq_id"][b, t:t + L] = sid
            a["seg_index"][b, t:t + L] = split.seg_index[di]
            sid += 1
            t += L
    return a


def to_torch(a, device):
    return {k: torch.as_tensor(v, device=device) for k, v in a.items()}


# ------------------------------------------------------------------ drafts
DRAFT_CFG = {"vocab_size": 50304, "n_layer": 10, "d_model": 256, "n_head": 4, "dense_ffn": 768, "moe_layers": [],
             "n_experts": 1, "top_k": 1, "expert_ffn": 1, "shared_ffn": 0, "rope_base": 10000.0, "norm_eps": 1e-6,
             "max_seq_len": 1024, "init_std": 0.02, "tie_embeddings": True}


def build_draft(seed=0):
    torch.manual_seed(seed)
    return CSRModel(ModelConfig.from_dict(DRAFT_CFG))


def draft_hidden(draft, ids, doc_ids, positions):
    x = draft.tok_emb(ids)
    cos, sin = rope_cos_sin(positions, draft.cfg.d_model // draft.cfg.n_head, draft.cfg.rope_base, x.dtype)
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    mask = CSRModel.attention_mask(doc_ids, positions)
    for blk in draft.blocks:
        x = x + blk.attn(blk.norm1(x), cos, sin, mask)
        x = x + blk.ffn(blk.norm2(x))
    return draft.norm_f(x)


def _qkv(blk, x, cos, sin):
    a = blk.attn
    B, T, _ = x.shape
    q, k, v = a.qkv(blk.norm1(x)).view(B, T, 3, a.n_head, a.hd).permute(2, 0, 3, 1, 4)
    return apply_rope(q, cos, sin), apply_rope(k, cos, sin), v


def _stats(logits):
    lf = logits.float()
    lp = torch.log_softmax(lf, -1)
    ent = -(lp.exp() * lp).sum(-1)
    top4 = lp.topk(4, -1).values.exp()
    return lp, ent, top4


@torch.no_grad()
def draft_rollout(draft, ids, doc_ids, positions, H=8, need_lp_only=False):
    """Causal greedy rollouts from EVERY prefix in parallel (tree attention).
    roll[b,t,k-1] = k-th greedy token generated from the real prefix x_0..x_t of t's document; proposal k attends to
    the real prefix <= t (document-causal mask) and its own proposals 1..k-1 only. Real tokens after t never enter.
    Returns roll (B,T,H) long, ent (B,T,H) float (entropy of the step-k distribution), top4 (B,T,4) step-1 probs,
    lp_cur (B,T) = log p(x_t | x_<t) of the real current token (0 at document starts)."""
    B, T = ids.shape
    cfg = draft.cfg
    hd = cfg.d_model // cfg.n_head
    x = draft.tok_emb(ids)
    cos, sin = rope_cos_sin(positions, hd, cfg.rope_base, x.dtype)
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    mask = CSRModel.attention_mask(doc_ids, positions)
    Kr, Vr = [], []
    for blk in draft.blocks:
        q, k, v = _qkv(blk, x, cos, sin)
        Kr.append(k)
        Vr.append(v)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + blk.attn.proj(y.transpose(1, 2).reshape(B, T, -1))
        x = x + blk.ffn(blk.norm2(x))
    logits = draft.lm_head(draft.norm_f(x))
    lp, ent1, top4 = _stats(logits)
    prev_lp = lp[:, :-1].gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    same = (doc_ids[:, 1:] == doc_ids[:, :-1]) & (doc_ids[:, 1:] >= 0)
    lp_cur = torch.zeros(B, T, device=ids.device)
    lp_cur[:, 1:] = torch.where(same, prev_lp, torch.zeros_like(prev_lp))
    if need_lp_only:
        return None, None, None, lp_cur
    cur = lp.argmax(-1)
    del lp
    props, ents = [cur], [ent1]
    Kp = [[] for _ in draft.blocks]
    Vp = [[] for _ in draft.blocks]
    neg = torch.finfo(torch.float32).min
    scale = 1.0 / math.sqrt(hd)
    for j in range(2, H + 1):
        x = draft.tok_emb(cur)
        c2, s2 = rope_cos_sin(positions + (j - 1), hd, cfg.rope_base, x.dtype)
        c2, s2 = c2.unsqueeze(1), s2.unsqueeze(1)
        for li, blk in enumerate(draft.blocks):
            q, k, v = _qkv(blk, x, c2, s2)
            Kp[li].append(k)
            Vp[li].append(v)
            sr = torch.matmul(q, Kr[li].transpose(-1, -2)).float() * scale
            sr = sr.masked_fill(~mask, neg)
            kp = torch.stack(Kp[li], -2)
            vp = torch.stack(Vp[li], -2)
            sp = (q.unsqueeze(-2) * kp).sum(-1).float() * scale
            w = torch.softmax(torch.cat([sr, sp], -1), -1)
            wr, wp = w[..., :T], w[..., T:]
            y = torch.matmul(wr.to(Vr[li].dtype), Vr[li]) + (wp.unsqueeze(-1).to(vp.dtype) * vp).sum(-2)
            x = x + blk.attn.proj(y.transpose(1, 2).reshape(B, T, -1))
            x = x + blk.ffn(blk.norm2(x))
        lpj, entj, _ = _stats(draft.lm_head(draft.norm_f(x)))
        cur = lpj.argmax(-1)
        props.append(cur)
        ents.append(entj)
    return torch.stack(props, -1), torch.stack(ents, -1), top4, lp_cur


# ------------------------------------------------------------------ forecast cache access + causal features
class ForecastCache:
    """Per-token cached draft outputs for a split (protocol section 4). index[doc] = start offset in the arrays
    (-1 = not cached). Arrays: roll_m/roll_w (n,8) uint16, ent_m/ent_w (n,8) uint8 (ent/LN_V*255), top_m/top_w (n,4)
    uint8 (p*255), lp (n,5) float16 (log p_d(x_t|x_<t) for drafts d=0..4)."""

    def __init__(self, cdir):
        cdir = Path(cdir)
        self.index = np.load(cdir / "index.npy")
        self.arr = {k: np.load(cdir / f"{k}.npy", mmap_mode="r") for k in
                    ("roll_m", "roll_w", "ent_m", "ent_w", "top_m", "top_w", "lp")}

    def gather(self, rows, split: Split, source):
        """source: 'match' | 'wrong' | 'shuffled' -> per-doc choice of the cached draft. Returns numpy dict
        roll (B,T,8) int64, ent (B,T,8) f32, top (B,T,4) f32, lp_a (B,T) f32 assigned-draft score,
        lp_o (B,T,4) f32 other drafts' scores; zeros on pads."""
        B = len(rows)
        out = {"roll": np.zeros((B, T_SEQ, 8), np.int64), "ent": np.zeros((B, T_SEQ, 8), np.float32),
               "top": np.zeros((B, T_SEQ, 4), np.float32), "lp_a": np.zeros((B, T_SEQ), np.float32),
               "lp_o": np.zeros((B, T_SEQ, 4), np.float32)}
        for b, row in enumerate(rows):
            t = 0
            for di in row:
                L = int(split.lengths[di])
                s = int(self.index[di])
                if s < 0:
                    raise guard.GuardError(f"doc {di} not in forecast cache")
                true = int(split.domains[di])
                if source == "match":
                    use_w, a = False, true
                elif source == "wrong":
                    use_w, a = True, int(split.wrong[di])
                else:
                    a = int(split.shuffled[di])
                    use_w = a != true
                k = "w" if use_w else "m"
                out["roll"][b, t:t + L] = self.arr[f"roll_{k}"][s:s + L]
                out["ent"][b, t:t + L] = self.arr[f"ent_{k}"][s:s + L] * (LN_V / 255.0)
                out["top"][b, t:t + L] = self.arr[f"top_{k}"][s:s + L] / 255.0
                lp = np.asarray(self.arr["lp"][s:s + L], dtype=np.float32)
                out["lp_a"][b, t:t + L] = lp[:, a]
                out["lp_o"][b, t:t + L] = np.delete(lp, a, axis=1)
                t += L
        return out


def causal_scalars(fc, input_ids, doc_ids, positions, seg_index, halflife=64.0):
    """21 causal router scalars from gathered cache tensors (torch, on device). Only quantities at positions <= t of
    the same document enter the features of position t."""
    from .fsp_model import ema_matrix
    roll, ent, top = fc["roll"], fc["ent"], fc["top"]
    B, T, H = roll.shape
    same_prev = lambda k: torch.cat([torch.zeros(B, k, dtype=torch.bool, device=roll.device),
                                     (doc_ids[:, k:] == doc_ids[:, :-k]) & (doc_ids[:, k:] >= 0)], 1)

    def shift(x, k):  # x[t-k] aligned at t
        pad = torch.zeros_like(x[:, :k])
        return torch.cat([pad, x[:, :-k]], 1)
    # cross-horizon agreement
    v1 = same_prev(1)
    a1 = (shift(roll, 1)[..., 1:8] == roll[..., 0:7]).float().mean(-1) * v1
    v4 = same_prev(4)
    a4 = (shift(roll, 4)[..., 4:8] == roll[..., 0:4]).float().mean(-1) * v4
    # past hit indicators: rollout made at t-h, step h, vs real token x_t
    hits = []
    for h_ in (1, 4, 8):
        vh = same_prev(h_)
        hits.append(((shift(roll[..., h_ - 1], h_) == input_ids) & vh).float())
    hits = torch.stack(hits, -1)
    lp_a, lp_o = fc["lp_a"], fc["lp_o"]
    sc = torch.stack([lp_a - lp_o.mean(-1), lp_a - lp_o.max(-1).values], -1)
    W = ema_matrix(doc_ids, positions, halflife, torch.float32)
    # EMA over positions where the quantity is defined (t >= 1 for scores; >= h for hits): simple causal EMA
    hit_e = torch.bmm(W, hits)
    sc_e = torch.bmm(W, sc) / 4.0
    pos = torch.stack([torch.log1p(positions.float()) / 7.0, torch.log1p(seg_index.float()) / 3.0], -1)
    scal = torch.cat([ent / LN_V, top, a1.unsqueeze(-1), a4.unsqueeze(-1), hit_e, sc_e, pos], -1)
    assert scal.shape[-1] == N_SCALAR
    return scal * (doc_ids >= 0).unsqueeze(-1)


# ------------------------------------------------------------------ curriculum
def curriculum(frac, shape="linear"):
    """Returns (sigma, tf, phase). P1 [0,0.4): 1,1 ; P2 [0.4,0.5): anneal 1->0 ; P3 [0.5,1]: 0,0."""
    if frac < 0.4:
        return 1.0, 1.0, 1
    if frac < 0.5:
        u = (frac - 0.4) / 0.1
        v = 1.0 - u if shape == "linear" else 0.5 * (1.0 + math.cos(math.pi * u))
        return v, v, 2
    return 0.0, 0.0, 3


def lr_at(step, n_steps, t):
    w = max(1, int(round(t["warmup_frac"] * n_steps)))
    if step < w:
        return t["lr"] * (step + 1) / w
    d0 = int(n_steps * (1 - t["decay_frac"]))
    if step < d0:
        return t["lr"]
    return t["lr"] + (t["min_lr"] - t["lr"]) * (step - d0) / max(1, n_steps - d0)


class ConcatSplit:
    """Virtual concatenation of open splits (TOKEN-COMPUTE stream). Global doc id = base[part] + local id."""

    def __init__(self, names):
        self.parts = [Split(n) for n in names]
        self.base = np.cumsum([0] + [p.n_docs for p in self.parts])[:-1].tolist()
        cat = lambda attr: np.concatenate([getattr(p, attr) for p in self.parts])
        self.lengths, self.domains, self.seg_index = cat("lengths"), cat("domains"), cat("seg_index")
        self.wrong, self.shuffled = cat("wrong"), cat("shuffled")
        self.n_docs = int(sum(p.n_docs for p in self.parts))
        self.name = "+".join(names)

    def doc(self, i):
        k = int(np.searchsorted(self.base, i, side="right") - 1)
        return self.parts[k].doc(i - self.base[k])


# ------------------------------------------------------------------ fast causal rollouts (cache construction)
def _vocab_stats(h, w):
    """h (N, d) bf16 final-normed draft states; w (V, d). Returns argmax, entropy, top4 probs, lse, logits(bf16)."""
    lg = (h @ w.t()).float()
    lse = torch.logsumexp(lg, -1)
    p = torch.exp(lg - lse.unsqueeze(-1))
    ent = lse - (p * lg).sum(-1)
    top = torch.topk(lg, 4, -1)
    return top.indices[:, 0], ent, torch.exp(top.values - lse.unsqueeze(-1)), lse, lg


def _vocab_stats2(h, w):
    """Steps >= 2: argmax and entropy only (fused reductions over the bf16 logits)."""
    lg = (h @ w.t()).float()
    lse = torch.logsumexp(lg, -1)
    ent = lse - (torch.softmax(lg, -1) * lg).sum(-1)
    return lg.argmax(-1), ent


_vocab_stats_c = None
_vocab_stats2_c = None


@torch.no_grad()
def draft_rollout_fast(draft, ids, lengths, H=8):
    """Same outputs as draft_rollout, for rows that each hold ONE document starting at position 0 (right-padded;
    lengths (B,) long). Real-prefix attention uses causal flash attention (query t sees real keys <= t) merged by
    log-sum-exp with the per-query branch keys of the rollout. Real tokens after t never enter position t."""
    global _vocab_stats_c, _vocab_stats2_c
    if _vocab_stats_c is None:
        _vocab_stats_c = torch.compile(_vocab_stats, dynamic=False)
        _vocab_stats2_c = torch.compile(_vocab_stats2, dynamic=False)
    B, T = ids.shape
    cfg = draft.cfg
    hd = cfg.d_model // cfg.n_head
    dt = torch.bfloat16
    positions = torch.arange(T, device=ids.device).expand(B, T)
    W = draft.tok_emb.weight.to(dt)
    x = draft.tok_emb(ids).to(dt)
    cos, sin = rope_cos_sin(positions, hd, cfg.rope_base, dt)
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    Kr, Vr = [], []
    with torch.autocast("cuda", dtype=dt):
        for blk in draft.blocks:
            q, k, v = _qkv(blk, x, cos, sin)
            q, k, v = q.to(dt), k.to(dt), v.to(dt)
            Kr.append(k)
            Vr.append(v)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + blk.attn.proj(y.transpose(1, 2).reshape(B, T, -1))
            x = x + blk.ffn(blk.norm2(x))
        hN = draft.norm_f(x).to(dt).reshape(B * T, -1)
    am, ent1, top4, lse, lg = _vocab_stats_c(hN, W)
    nxt = torch.cat([ids[:, 1:], ids[:, :1]], 1).reshape(-1)
    lp_next = lg.gather(1, nxt.view(-1, 1)).squeeze(1) - lse  # log p(x_{t+1} | x_<=t)
    del lg
    valid = torch.arange(T, device=ids.device).view(1, T) < lengths.view(B, 1)
    lp_cur = torch.zeros(B, T, device=ids.device)
    lp_cur[:, 1:] = lp_next.view(B, T)[:, :-1] * valid[:, 1:]
    cur = am.view(B, T)
    props, ents = [cur], [ent1.view(B, T)]
    nh = cfg.n_head
    Kp = [torch.zeros(B, nh, T, H - 1, hd, device=ids.device, dtype=torch.float32) for _ in draft.blocks]
    Vp = [torch.zeros(B, nh, T, H - 1, hd, device=ids.device, dtype=torch.float32) for _ in draft.blocks]
    scale = 1.0 / math.sqrt(hd)
    for j in range(2, H + 1):
        x = draft.tok_emb(cur).to(dt)
        c2, s2 = rope_cos_sin(positions + (j - 1), hd, cfg.rope_base, dt)
        c2, s2 = c2.unsqueeze(1), s2.unsqueeze(1)
        n = j - 1
        with torch.autocast("cuda", dtype=dt):
            for li, blk in enumerate(draft.blocks):
                q, k, v = _qkv(blk, x, c2, s2)
                q, k, v = q.to(dt), k.to(dt), v.to(dt)
                Kp[li][..., n - 1, :] = k.float()
                Vp[li][..., n - 1, :] = v.float()
                o = torch.ops.aten._scaled_dot_product_flash_attention(q, Kr[li], Vr[li], 0.0, True, False,
                                                                     scale=scale)
                yr, lse_r = o[0].float(), o[1].float()  # (B,H,T,hd), (B,H,T)
                kp, vp = Kp[li][..., :n, :], Vp[li][..., :n, :]
                sp = torch.einsum("bhtd,bhtnd->bhtn", q.float(), kp) * scale
                mx = torch.maximum(lse_r, sp.max(-1).values)
                wr = torch.exp(lse_r - mx)
                wp = torch.exp(sp - mx.unsqueeze(-1))
                y = (yr * wr.unsqueeze(-1) + torch.einsum("bhtn,bhtnd->bhtd", wp, vp)) / \
                    (wr + wp.sum(-1)).unsqueeze(-1)
                x = x + blk.attn.proj(y.to(dt).transpose(1, 2).reshape(B, T, -1))
                x = x + blk.ffn(blk.norm2(x))
            hN = draft.norm_f(x).to(dt).reshape(B * T, -1)
        am, entj = _vocab_stats2_c(hN, W)
        cur = am.view(B, T)
        props.append(cur)
        ents.append(entj.view(B, T))
    return torch.stack(props, -1), torch.stack(ents, -1), top4.view(B, T, 4), lp_cur


@torch.no_grad()
def draft_score_fast(draft, ids, lengths):
    """Score-only pass (lp_cur) for single-document rows."""
    B, T = ids.shape
    cfg = draft.cfg
    dt = torch.bfloat16
    positions = torch.arange(T, device=ids.device).expand(B, T)
    with torch.autocast("cuda", dtype=dt):
        x = draft.tok_emb(ids)
        cos, sin = rope_cos_sin(positions, cfg.d_model // cfg.n_head, cfg.rope_base, x.dtype)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        for blk in draft.blocks:
            q, k, v = _qkv(blk, x, cos, sin)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + blk.attn.proj(y.transpose(1, 2).reshape(B, T, -1))
            x = x + blk.ffn(blk.norm2(x))
        lg = (draft.norm_f(x) @ draft.tok_emb.weight.to(dt).t()).float()
    lp = torch.log_softmax(lg[:, :-1], -1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    valid = torch.arange(T, device=ids.device).view(1, T) < lengths.view(B, 1)
    out = torch.zeros(B, T, device=ids.device)
    out[:, 1:] = lp * valid[:, 1:]
    return out
