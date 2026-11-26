"""csr-forecast-latent-edl-v1 (fle) library: causal region forecasts from frozen generalist drafts, frozen MiniLM
encoding, forecast aggregation (A1 meanstat / A2 deepsets-cv), balanced Sinkhorn soft clustering, frozen per-region
assignment tables, and the FL-EDL / FD training-only losses. New module; nothing existing is modified.
Prereg: docs/FORECAST_LATENT_EDL_V1_PREREG.md."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import guard
from .fsp_lib import DRAFT_CFG, PAD_ID, T_SEQ, apply_rope, build_draft, rope_cos_sin  # noqa: F401
from .fle_paths import ART, CACHE, DRAFT_G0, DRAFT_G0_SHA, MINILM

H_MAX = 32
L_BASE = 128
N_PTS = 7  # forecast points per segment at multiples of 128: 128..896
MIN_REGION = 32
LN_V = math.log(50304)


# ------------------------------------------------------------------ regions
def region_points(length, L=L_BASE):
    """Forecast points s_j = j*L (j >= 1) of a segment of `length` tokens whose region has >= MIN_REGION tokens."""
    return [s for s in range(L, int(length), L) if int(length) - s >= MIN_REGION]


# ------------------------------------------------------------------ counter-based Gumbel noise
_M64 = (1 << 64) - 1


def _s64(c):
    return c - (1 << 64) if c >= (1 << 63) else c


_C1, _C2, _C3 = _s64(0x9E3779B97F4A7C15), _s64(0xBF58476D1CE4E5B9), _s64(0x94D049BB133111EB)


def _srl(x, n):
    return (x >> n) & ((1 << (64 - n)) - 1)


def splitmix(x):
    """splitmix64 on int64 tensors (wrapping arithmetic); returns int64."""
    z = x + _C1
    z = (z ^ _srl(z, 30)) * _C2
    z = (z ^ _srl(z, 27)) * _C3
    return z ^ _srl(z, 31)


def uniform_from(keys):
    """keys int64 tensor -> uniform (0,1) float64, deterministic per key."""
    z = splitmix(keys)
    return (_srl(z, 11).to(torch.float64) + 0.5) / float(1 << 53)


def seed_key(draft_idx, gdoc, s):
    return (int(draft_idx) * 1_000_003 + int(gdoc)) * 4099 + int(s)


# ------------------------------------------------------------------ drafts
def load_draft(path, device, expect_sha=None):
    p = guard.ROOT / path
    if expect_sha is not None and guard.sha256_file(p) != expect_sha:
        raise guard.GuardError(f"draft hash mismatch {path}")
    m = build_draft(0)
    sd = torch.load(p, map_location="cpu", weights_only=True)
    m.load_state_dict(sd["model"])
    m = m.to(device).eval()
    for q in m.parameters():
        q.requires_grad_(False)
    return m


def draft_list():
    """[(name, path, sha256)] of the approved generalist drafts (G0 + new seeds), from artifacts/.../drafts.json."""
    d = json.loads((ART / "drafts.json").read_text())
    return [(x["name"], x["path"], x["sha256"]) for x in d["drafts"]]


def _qkv(blk, x, cos, sin):
    a = blk.attn
    B, T, _ = x.shape
    q, k, v = a.qkv(blk.norm1(x)).view(B, T, 3, a.n_head, a.hd).permute(2, 0, 3, 1, 4)
    return apply_rope(q, cos, sin), apply_rope(k, cos, sin), v


@torch.no_grad()
def draft_forecast(draft, ids, lengths, points, keys, H=H_MAX, temp=0.8, topk=50):
    """Causal H-token continuations from forecast points (prereg 3).
    ids (B,T) long: one document per row from position 0, right-padded; points (B,P) long forecast points s (the
    prefix is [0, s)); -1 = unused slot; keys (B,P) int64 per-(draft, doc, point) seeds.
    Real keys at positions >= s never enter the computation of point s (mask), so changing tokens >= s leaves the
    continuation from s unchanged (test T1).
    Returns tok (B,P,H) long, ptok, ent, top1 (B,P,H) float32 (T = 1 distribution of the draft)."""
    B, T = ids.shape
    P = points.shape[1]
    cfg = draft.cfg
    nh, hd = cfg.n_head, cfg.d_model // cfg.n_head
    dev = ids.device
    dt = torch.bfloat16 if dev.type == "cuda" else torch.float32
    pos = torch.arange(T, device=dev).expand(B, T)
    W = draft.tok_emb.weight.to(dt)
    x = draft.tok_emb(ids).to(dt)
    cos, sin = rope_cos_sin(pos, hd, cfg.rope_base, dt)
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    Kr, Vr = [], []
    with torch.autocast(device_type=dev.type, dtype=dt, enabled=dev.type == "cuda"):
        for blk in draft.blocks:
            q, k, v = _qkv(blk, x, cos, sin)
            q, k, v = q.to(dt), k.to(dt), v.to(dt)
            Kr.append(k)
            Vr.append(v)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + blk.attn.proj(y.transpose(1, 2).reshape(B, T, -1))
            x = x + blk.ffn(blk.norm2(x))
        hN = draft.norm_f(x).to(dt)
    valid = points >= 1
    sp = points.clamp(min=1)
    # first token: from the hidden state at s-1 (sees real tokens <= s-1 only, causal)
    h0 = hN.gather(1, (sp - 1).unsqueeze(-1).expand(B, P, hN.shape[-1]))  # (B,P,d)
    kmask = torch.arange(T, device=dev).view(1, 1, 1, T) < sp.view(B, 1, P, 1)  # keys < s
    toks, pts, ents, tops = [], [], [], []
    Kr = [k_.float() for k_ in Kr]  # converted once (identical values to per-step conversion)
    Vr = [v_.float() for v_ in Vr]
    Kg = [torch.zeros(B, nh, P, H, hd, device=dev, dtype=torch.float32) for _ in draft.blocks]
    Vg = [torch.zeros(B, nh, P, H, hd, device=dev, dtype=torch.float32) for _ in draft.blocks]
    scale = 1.0 / math.sqrt(hd)
    hcur = h0
    for j in range(1, H + 1):
        lg = (hcur.reshape(B * P, -1) @ W.t()).float()
        lp = torch.log_softmax(lg, -1)
        ent = -(lp.exp() * lp).sum(-1)
        top1 = lp.max(-1).values.exp()
        tv, ti = (lg / temp).topk(topk, -1)
        r = torch.arange(topk, device=dev, dtype=torch.int64).view(1, -1)
        kk = keys.reshape(-1, 1).to(torch.int64) * 131 + j * 1_000_033 + r
        g = -torch.log(-torch.log(uniform_from(kk))).to(torch.float32)
        choice = (tv + g).argmax(-1)
        tok = ti.gather(1, choice.view(-1, 1)).squeeze(1)
        ptok = lp.gather(1, tok.view(-1, 1)).squeeze(1).exp()
        toks.append(tok.view(B, P))
        pts.append(ptok.view(B, P))
        ents.append(ent.view(B, P))
        tops.append(top1.view(B, P))
        if j == H:
            break
        # feed tok at position s + j - 1
        xg = draft.tok_emb(tok.view(B, P)).to(dt)
        pg = sp + (j - 1)
        c2, s2 = rope_cos_sin(pg, hd, cfg.rope_base, dt)
        c2, s2 = c2.unsqueeze(1), s2.unsqueeze(1)  # (B,1,P,hd)
        n = j  # generated keys after this step
        with torch.autocast(device_type=dev.type, dtype=dt, enabled=dev.type == "cuda"):
            for li, blk in enumerate(draft.blocks):
                q, k, v = _qkv(blk, xg, c2, s2)  # (B,nh,P,hd)
                q32 = q.float()
                Kg[li][..., n - 1, :] = k.float()
                Vg[li][..., n - 1, :] = v.float()
                sr = torch.einsum("bhpd,bhtd->bhpt", q32, Kr[li]) * scale
                sr = sr.masked_fill(~kmask, float("-inf"))
                sg = torch.einsum("bhpd,bhpnd->bhpn", q32, Kg[li][..., :n, :]) * scale
                mx = torch.maximum(sr.max(-1).values, sg.max(-1).values)
                wr = torch.exp(sr - mx.unsqueeze(-1))
                wg = torch.exp(sg - mx.unsqueeze(-1))
                y = (torch.einsum("bhpt,bhtd->bhpd", wr, Vr[li]) +
                     torch.einsum("bhpn,bhpnd->bhpd", wg, Vg[li][..., :n, :])) / \
                    (wr.sum(-1) + wg.sum(-1)).unsqueeze(-1)
                xg = xg + blk.attn.proj(y.to(dt).transpose(1, 2).reshape(B, P, -1))
                xg = xg + blk.ffn(blk.norm2(xg))
            hcur = draft.norm_f(xg).to(dt)
    tok = torch.stack(toks, -1)
    tok = torch.where(valid.unsqueeze(-1), tok, torch.full_like(tok, PAD_ID))
    return tok, torch.stack(pts, -1), torch.stack(ents, -1), torch.stack(tops, -1)


# ------------------------------------------------------------------ text + MiniLM encoder
_TOK = None


def pythia_tok():
    global _TOK
    if _TOK is None:
        from tokenizers import Tokenizer
        _TOK = Tokenizer.from_file(str(guard.ROOT / "artifacts/tokenizer/pythia-12b/tokenizer.json"))
    return _TOK


def decode_batch(seqs):
    return pythia_tok().decode_batch([list(map(int, s)) for s in seqs], skip_special_tokens=True)


class MiniLM:
    """Frozen all-MiniLM-L6-v2 (mean pooling + L2 norm). Independent of drafts, main model and the evaluator."""

    def __init__(self, device):
        from transformers import BertModel, BertTokenizerFast
        d = MINILM
        self.sha = guard.sha256_file(d / "model.safetensors")
        self.tk = BertTokenizerFast(tokenizer_file=str(d / "tokenizer.json"), vocab_file=str(d / "vocab.txt"))
        self.model = BertModel.from_pretrained(str(d), local_files_only=True, add_pooling_layer=False,
                                               use_safetensors=True).to(device).eval()
        self.device = device

    @torch.inference_mode()
    def encode(self, texts, max_len, left=False, bs=1024):
        self.tk.truncation_side = "left" if left else "right"
        out = np.zeros((len(texts), 384), np.float16)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            enc = self.tk([texts[i] for i in idx], padding=True, truncation=True, max_length=max_len,
                          return_tensors="pt").to(self.device)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16,
                                enabled=self.device.type == "cuda"):
                h = self.model(**enc).last_hidden_state.float()
            m = enc["attention_mask"].unsqueeze(-1).float()
            e = (h * m).sum(1) / m.sum(1).clamp(min=1)
            out[idx] = F.normalize(e, dim=-1).cpu().numpy().astype(np.float16)
        return out


# ------------------------------------------------------------------ cache access
class RegionCache:
    """Read-only access to completed cache chunks of one region set (manifest-verified)."""

    def __init__(self, name):
        self.dir = CACHE / name
        man = json.loads((self.dir / "manifest.json").read_text())
        if not man.get("complete"):
            raise guard.GuardError(f"cache {name} incomplete")
        self.man = man
        self.chunks = man["chunks"]

    def load(self, fields=None, verify=True):
        """Concatenate all chunks: dict of arrays (regions table + requested fields)."""
        out = {}
        for c in self.chunks:
            p = self.dir / c["file"]
            if verify and guard.sha256_file(p) != c["sha256"]:
                raise guard.GuardError(f"cache chunk hash mismatch {p}")
            z = np.load(p)
            for k in z.files:
                if fields is None or k in fields or k in ("gdoc", "s", "length"):
                    out.setdefault(k, []).append(z[k])
        return {k: np.concatenate(v, 0) for k, v in out.items()}


# ------------------------------------------------------------------ aggregation
def draft_stats(ptok_q, ent_q, top1_q, H):
    """Per-(region, draft) stats from uint8 quantised arrays (R, D, Hmax) over the first H steps -> (R, D, 3)."""
    p = ptok_q[..., :H].astype(np.float32) / 255.0
    e = ent_q[..., :H].astype(np.float32) / 255.0 * LN_V
    t = top1_q[..., :H].astype(np.float32) / 255.0
    return np.stack([e.mean(-1), p.mean(-1), t.mean(-1)], -1)


def set_stats(emb, ds, first_tok):
    """Inter-draft statistics (R, 5): disagreement (1 - mean pairwise cos), mean entropy, mean confidence, mean top-1,
    entropy spread; emb (R, D, E) normalised; ds (R, D, 3); first_tok (R, D)."""
    R, D, _ = emb.shape
    e = emb.astype(np.float32)
    G = np.einsum("rde,rfe->rdf", e, e)
    iu = np.triu_indices(D, 1)
    dis = 1.0 - G[:, iu[0], iu[1]].mean(-1)
    return np.stack([dis, ds[..., 0].mean(-1), ds[..., 1].mean(-1), ds[..., 2].mean(-1), ds[..., 0].std(-1)], -1)


def first_token_agreement(first_tok):
    R, D = first_tok.shape
    agree = np.zeros(R, np.float32)
    for r in range(R):
        _, c = np.unique(first_tok[r], return_counts=True)
        agree[r] = c.max() / D
    return agree


def a1_meanstat(emb, ss, mu, sd):
    """A1: normalize([normalize(mean_d e_d), 0.25 * standardised stats / sqrt(5)]). Permutation invariant."""
    u = emb.astype(np.float32).mean(1)
    u /= np.maximum(1e-9, np.linalg.norm(u, axis=1, keepdims=True))
    s = (ss - mu) / np.maximum(1e-6, sd)
    z = np.concatenate([u, 0.25 * s / math.sqrt(5.0)], 1)
    return z / np.maximum(1e-9, np.linalg.norm(z, axis=1, keepdims=True))


class DeepSetsCV(nn.Module):
    """A2: phi MLP(384+3 -> 256 -> 256), mean pool over drafts, rho MLP(256+5 -> 256 -> 128), L2-normalised."""

    def __init__(self):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(387, 256), nn.GELU(), nn.Linear(256, 256))
        self.rho = nn.Sequential(nn.Linear(261, 256), nn.GELU(), nn.Linear(256, 128))

    def forward(self, emb, ds, ss, mask=None):
        """emb (N,D,384), ds (N,D,3) standardised, ss (N,5) standardised; mask (N,D) bool of drafts in the set."""
        h = self.phi(torch.cat([emb, ds], -1))
        if mask is None:
            pooled = h.mean(1)
        else:
            m = mask.unsqueeze(-1).to(h.dtype)
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return F.normalize(self.rho(torch.cat([pooled, ss], -1)), dim=-1)


def view_stats(emb, ds, mask):
    """Torch version of set_stats restricted to a draft subset mask (N,D) (used for A2 cross-view training)."""
    m = mask.to(emb.dtype)
    G = torch.einsum("nde,nfe->ndf", emb, emb)
    pm = m.unsqueeze(-1) * m.unsqueeze(-2) * (1 - torch.eye(emb.shape[1], device=emb.device)).unsqueeze(0)
    dis = 1.0 - (G * pm).sum((1, 2)) / pm.sum((1, 2)).clamp(min=1)
    cnt = m.sum(1).clamp(min=1)
    mean = lambda x: (x * m).sum(1) / cnt
    e = ds[..., 0]
    spread = torch.sqrt(((e - mean(e).unsqueeze(1)) ** 2 * m).sum(1) / cnt)
    return torch.stack([dis, mean(e), mean(ds[..., 1]), mean(ds[..., 2]), spread], -1)


# ------------------------------------------------------------------ balanced Sinkhorn soft clustering
def sinkhorn_log(S, tau, iters=5):
    """S (N,K) similarities. Returns (Q (N,K) rows sum 1 with equal column masses, b (K,) column log-potential)."""
    N, K = S.shape
    logits = S / tau
    b = torch.zeros(K, dtype=S.dtype, device=S.device)
    for _ in range(iters):
        lq = torch.log_softmax(logits + b, 1)  # rows sum to 1
        col = torch.logsumexp(lq, 0)  # log column mass
        b = b - (col - math.log(N / K))
    Q = torch.softmax(logits + b, 1)
    return Q, b


def fit_balanced(X, K, tau, seed=2026, iters=60, device="cpu"):
    """Spherical balanced soft k-means. X (N,D) L2-normalised float32 numpy. Returns centroids (K,D), b (K,)."""
    g = np.random.default_rng(seed)
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    N = Xt.shape[0]
    # k-means++ init (cosine distance)
    idx = [int(g.integers(N))]
    d2 = (1 - Xt @ Xt[idx[0]]).clamp(min=0)
    for _ in range(1, K):
        p = (d2 / d2.sum()).double().cpu().numpy()
        p = p / p.sum()
        i = int(g.choice(N, p=p))
        idx.append(i)
        d2 = torch.minimum(d2, (1 - Xt @ Xt[i]).clamp(min=0))
    C = Xt[idx].clone()
    b = torch.zeros(K, device=device)
    for _ in range(iters):
        Q, b = sinkhorn_log(Xt @ C.t(), tau)
        C = F.normalize(Q.t() @ Xt, dim=-1)
    Q, b = sinkhorn_log(Xt @ C.t(), tau)
    return C.cpu().numpy(), b.cpu().numpy()


def assign(X, C, b, tau):
    """Frozen per-region assignment q = softmax(x.C/tau + b) (no batch dependence)."""
    lg = (X.astype(np.float32) @ C.T) / tau + b
    lg -= lg.max(1, keepdims=True)
    q = np.exp(lg)
    return q / q.sum(1, keepdims=True)


def norm_entropy(q):
    K = q.shape[1]
    return float((-(q * np.log(np.clip(q, 1e-12, 1))).sum(1) / math.log(K)).mean())


def calibrate_tau(X, K, target=0.4, grid=(0.01, 0.02, 0.03, 0.05, 0.1), seed=2026, device="cpu"):
    best = None
    rows = []
    for tau in grid:
        C, b = fit_balanced(X, K, tau, seed=seed, device=device)
        h = norm_entropy(assign(X, C, b, tau))
        rows.append({"tau": tau, "entropy": h})
        if best is None or abs(h - target) < abs(best[0] - target):
            best = (h, tau, C, b)
    return best[1], best[2], best[3], rows


def hungarian_agreement(a, b, K):
    """Fraction of regions whose argmax clusters agree after the best one-to-one relabelling."""
    from .lfr_eval import hungarian
    M = np.zeros((K, K))
    np.add.at(M, (a, b), 1)
    c = hungarian(-M)
    return float(M[np.arange(K), c].sum() / max(1, len(a)))


# ------------------------------------------------------------------ assignment tables (training-time lookup)
class QTable:
    """Frozen soft assignments for (gdoc, region index j) at region length L. q is returned as a constant tensor."""

    def __init__(self, path):
        z = np.load(path)
        self.L = int(z["L"])
        self.K = int(z["K"])
        self.gdoc = z["gdoc"].astype(np.int64)
        self.j = z["j"].astype(np.int64)
        self.q = z["q"].astype(np.float32)
        self.key = {}
        for i, (g, j) in enumerate(zip(self.gdoc.tolist(), self.j.tolist())):
            self.key[(g, j)] = i
        self.sha = guard.sha256_file(path)

    def batch(self, gdoc_bt, pos_bt, doc_ids_bt, perm_seed=None):
        """For a (B,T) batch: region id per token (-1 = no valid region) and q per region (R,K) float32.
        perm_seed: SHUF control - permute q among the batch's valid regions (multiset preserved).
        Vectorised: a dense (gdoc, j) -> table-row index; regions are numbered in token order of first appearance."""
        if not hasattr(self, "_dense"):
            n = int(self.gdoc.max()) + 1
            J = int(self.j.max()) + 1
            self._dense = np.full((n, J), -1, np.int64)
            self._dense[self.gdoc, self.j] = np.arange(len(self.gdoc))
        g = np.asarray(gdoc_bt, np.int64).reshape(-1)
        jj = np.asarray(pos_bt, np.int64).reshape(-1) // self.L
        ok = (np.asarray(doc_ids_bt).reshape(-1) >= 0) & (g >= 0) & (g < self._dense.shape[0]) & \
            (jj < self._dense.shape[1])
        tix = np.full(g.shape, -1, np.int64)
        tix[ok] = self._dense[g[ok], jj[ok]]
        valid = tix >= 0
        reg = np.full(g.shape, -1, np.int64)
        if valid.any():
            u, first, inv = np.unique(tix[valid], return_index=True, return_inverse=True)
            order = np.argsort(first, kind="stable")  # number regions by first appearance
            rank = np.empty(len(u), np.int64)
            rank[order] = np.arange(len(u))
            reg[valid] = rank[inv]
            q = self.q[u[order]]
        else:
            q = np.zeros((0, self.K), np.float32)
        if perm_seed is not None and len(q) > 1:
            q = q[np.random.default_rng(perm_seed).permutation(len(q))]
        return reg.reshape(np.shape(gdoc_bt)), q

    def batch_reference(self, gdoc_bt, pos_bt, doc_ids_bt, perm_seed=None):
        """Original loop implementation (kept for the equivalence test)."""
        B, T = gdoc_bt.shape
        reg = np.full((B, T), -1, np.int64)
        qs = []
        rid = 0
        for b in range(B):
            dids = doc_ids_bt[b]
            for d in np.unique(dids[dids >= 0]):
                m = dids == d
                g = int(gdoc_bt[b][m][0])
                pos = pos_bt[b][m]
                jj = pos // self.L
                cols = np.flatnonzero(m)
                for j in np.unique(jj):
                    i = self.key.get((g, int(j)))
                    if i is None:
                        continue
                    reg[b, cols[jj == j]] = rid
                    qs.append(self.q[i])
                    rid += 1
        q = np.stack(qs) if qs else np.zeros((0, self.K), np.float32)
        if perm_seed is not None and len(q) > 1:
            q = q[np.random.default_rng(perm_seed).permutation(len(q))]
        return reg, q


def fledl_loss(p_layers, reg, q, eps=1e-8):
    """L_FL-EDL = mean_{l, k<j} -log(JSD(R_k^l, R_j^l) + eps), R_k^l = sum_i q_ik r_i^l / sum_i q_ik,
    r_i^l = mean router softmax over region i's tokens. p_layers: list of (N,E) fp32 router probs (grad);
    reg (N,) long region id per token (-1 none); q (R,K) constant. Returns (loss, mean JSD) tensors."""
    dev = p_layers[0].device
    R = q.shape[0]
    if R < 2:
        z = torch.zeros((), device=dev)
        return z, z
    m = reg >= 0
    ridx = reg[m]
    cnt = torch.zeros(R, device=dev).index_add_(0, ridx, torch.ones_like(ridx, dtype=torch.float32))
    qw = q.detach()
    mass = qw.sum(0)
    keep = mass > 1e-6
    qw, mass = qw[:, keep], mass[keep]
    K = qw.shape[1]
    if K < 2:
        z = torch.zeros((), device=dev)
        return z, z
    a, b = torch.triu_indices(K, K, 1, device=dev)
    tot, jtot = 0.0, 0.0
    for p in p_layers:
        E = p.shape[1]
        r = torch.zeros(R, E, device=dev, dtype=p.dtype).index_add_(0, ridx, p[m]) / cnt.unsqueeze(1)
        Rk = (qw.t() @ r) / mass.unsqueeze(1)  # (K,E)
        pa, pb = Rk[a], Rk[b]
        mm = 0.5 * (pa + pb)
        kl = lambda x, y: (x * (torch.log(x.clamp(min=1e-12)) - torch.log(y.clamp(min=1e-12)))).sum(-1)
        jsd = (0.5 * kl(pa, mm) + 0.5 * kl(pb, mm)).clamp(min=0.0)  # JSD >= 0 (fp rounding can go below 0)
        tot = tot + (-torch.log(jsd + eps)).mean()
        jtot = jtot + jsd.mean().detach()
    return tot / len(p_layers), jtot / len(p_layers)


def fd_loss(hidden, reg, q, head):
    """Forecast distillation without divergence: soft CE between q (constant) and softmax(head(mean hidden of region))."""
    m = reg >= 0
    R = q.shape[0]
    if R == 0:
        return torch.zeros((), device=hidden.device)
    ridx = reg[m]
    hf = hidden.reshape(-1, hidden.shape[-1])[m].float()
    cnt = torch.zeros(R, device=hidden.device).index_add_(0, ridx, torch.ones_like(ridx, dtype=torch.float32))
    hm = torch.zeros(R, hf.shape[1], device=hidden.device).index_add_(0, ridx, hf) / cnt.unsqueeze(1)
    lp = torch.log_softmax(head(hm), -1)
    return -(q.detach() * lp).sum(-1).mean()


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()
