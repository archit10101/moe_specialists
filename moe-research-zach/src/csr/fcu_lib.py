"""csr-forecast-counterfactual-utility-v1 (fcu) shared library. New module; every predecessor file is read-only.

Prereg docs/FORECAST_COUNTERFACTUAL_UTILITY_V1_PREREG.md section 2:
  * regions(): the frozen Phase-1 region table (fle calib_search cache points s in {256,512,768}, >=128 tokens).
  * UtilityEngine: exact equal-compute counterfactual expert utilities u_{i,l,e} (fp32). An intervention at MoE
    layer l touches only region tokens where e is in the normal top-4; there logit_e = -inf, so the next-ranked
    expert replaces e and the gates are the model's own softmax over the remaining 31 experts (= p / (1 - p_e)).
    Positions before the region are unaffected (causal), so the counterfactual recomputes only region positions
    above the intervened block, attending to cached normal-pass prefix keys/values.
  * probes: closed-form ridge, multinomial logistic regression, paper-grouped CV, cluster bootstrap.
"""
from __future__ import annotations

import json
import math

import numpy as np
import torch
import torch.nn.functional as F

from csr import guard
from csr.model import apply_rope, rope_cos_sin

EXP = "csr-forecast-counterfactual-utility-v1"
ART = guard.ROOT / "artifacts/forecast-counterfactual-utility-v1"
TMP = guard.ROOT / ".tmp/forecast-counterfactual-utility-v1"
PREREG = guard.ROOT / "docs/FORECAST_COUNTERFACTUAL_UTILITY_V1_PREREG.md"
STOP_MARKER = ART / "STOP"
FLE_ART = guard.ROOT / "artifacts/forecast-latent-edl-v1"
LAYERS = list(range(5, 11))
R_LEN = 256
MIN_REG = 128
POINTS = (256, 512, 768)
CKPTS = {
    "T4_s0": ("runs/20261114T131547Z_forecast_latent_edl_v1_final_T4_s0/ckpt_p100",
              "fb113eb6d849abbdedc52414d9425995f0a6ecc9c0e2f02a2db1352b496ca694"),
    "T4_s1": ("runs/20261114T145059Z_forecast_latent_edl_v1_final_T4_s1/ckpt_p100",
              "0d46807b5c7ab4cedf549aeb4081381426591f72c12465af6895e3cf45b3d513"),
}


def prereg_sha():
    return guard.sha256_file(PREREG)


def load_model(name, device, dtype=torch.float32):
    from csr.lfr_model import LFRConfig, LFRModel
    path, sha = CKPTS[name]
    p = guard.ROOT / path / "state.pt"
    if guard.sha256_file(p) != sha:
        raise guard.GuardError(f"checkpoint hash mismatch {p}")
    ck = torch.load(p, map_location="cpu", weights_only=False)
    m = LFRModel(LFRConfig.from_dict({**ck["model_cfg"], "teacher": False}), ck["config"]["seed"])
    m.load_state_dict(ck["model"], strict=True)
    m = m.to(device=device, dtype=dtype).eval()
    for q in m.parameters():
        q.requires_grad_(False)
    return m


# ------------------------------------------------------------------ regions
def regions():
    """Frozen Phase-1 region table (prereg 2.1). Returns dict of arrays sorted by (s, gdoc):
    gdoc, s, end, half (0 = probe-train paper-half A, 1 = probe-test B), domain, cidx (row in fle calib_search cache).
    """
    from csr.fsp_lib import Split
    from csr.lfr_eval import eval_rows, paper_half
    c = json.loads((FLE_ART / "cache/calib_search/manifest.json").read_text())
    if not c.get("complete") or len(c["chunks"]) != 1:
        raise guard.GuardError("unexpected fle calib_search cache layout")
    p = FLE_ART / "cache/calib_search" / c["chunks"][0]["file"]
    if guard.sha256_file(p) != c["chunks"][0]["sha256"]:
        raise guard.GuardError("fle calib_search cache hash mismatch")
    z = np.load(p)
    g, s, L = z["gdoc"].astype(np.int64), z["s"].astype(np.int64), z["length"].astype(np.int64)
    sp = Split("calib_search")
    half = {}
    for h in (0, 1):
        for r in eval_rows(sp, 200_000, paper_half(sp, h)):
            for d in r:
                half[int(d)] = h
    end = np.minimum(s + R_LEN, L)
    m = np.isin(s, POINTS) & (end - s >= MIN_REG)
    idx = np.nonzero(m)[0]
    order = np.lexsort((g[idx], s[idx]))
    idx = idx[order]
    out = {"cidx": idx, "gdoc": g[idx], "s": s[idx], "end": end[idx],
           "half": np.asarray([half[int(d)] for d in g[idx]], np.int64),
           "domain": np.asarray(sp.domains)[g[idx]].astype(np.int64)}
    return out, sp, z


# ------------------------------------------------------------------ utility engine
def _qkv(blk, x, cos, sin):
    a = blk.attn
    B, T, _ = x.shape
    q, k, v = a.qkv(blk.norm1(x)).view(B, T, 3, a.n_head, a.hd).permute(2, 0, 3, 1, 4)
    return apply_rope(q, cos, sin), apply_rope(k, cos, sin), v


def moe_ffn(moe, h, k, kill=None):
    """h (N,d). kill (N,E) bool: logits set to -inf before the softmax (equal-compute replacement)."""
    lg = F.linear(h, moe.router.weight)
    if kill is not None:
        lg = lg.masked_fill(kill, float("-inf"))
    p = torch.softmax(lg, -1)
    gates, topi = p.topk(k, -1)
    return moe.experts(h, topi, gates) + moe.shared(h), p, topi


def _ce_rows(model, hN, tgt, chunk=8192):
    """hN (N, T, d), tgt (N, T) long -> CE (N, T) fp32 via chunked logits (tied embedding)."""
    N, T, d = hN.shape
    flat, tf = hN.reshape(-1, d), tgt.reshape(-1)
    odt = torch.float64 if hN.dtype == torch.float64 else torch.float32
    out = torch.empty(flat.shape[0], device=hN.device, dtype=odt)
    W = model.tok_emb.weight
    for a in range(0, flat.shape[0], chunk):
        lg = (flat[a:a + chunk] @ W.t()).to(odt)
        out[a:a + chunk] = torch.logsumexp(lg, -1) - lg.gather(1, tf[a:a + chunk].view(-1, 1)).squeeze(1)
    return out.view(N, T)


class UtilityEngine:
    """Batch of regions sharing the same forecast point s; rows = one segment each, tokens [0, s + R_LEN) (pad 1)."""

    def __init__(self, model, layers=LAYERS):
        self.m = model
        self.layers = list(layers)
        self.b_min = min(self.layers) + 1  # first intervened block

    @torch.no_grad()
    def normal(self, ids, s, ends):
        m = self.m
        cfg = m.cfg
        B, T = ids.shape
        R = T - s
        dev = ids.device
        dt = m.tok_emb.weight.dtype
        pos = torch.arange(T, device=dev).expand(B, T)
        cos, sin = rope_cos_sin(pos, cfg.d_model // cfg.n_head, cfg.rope_base, dt)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        x = m.tok_emb(ids)
        st = {"s": s, "R": R, "B": B, "kv": {}, "xmid": {}, "topi": {}, "p": {}}
        mi = 0
        for bi, blk in enumerate(m.blocks):
            q, k, v = _qkv(blk, x, cos, sin)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + blk.attn.proj(y.transpose(1, 2).reshape(B, T, -1))
            if bi >= self.b_min:
                st["kv"][bi] = (k[:, :, :s].contiguous(), v[:, :, :s].contiguous())
            h = blk.norm2(x)
            if not blk.is_moe:
                x = x + blk.ffn(h)
                continue
            yf, p, topi = moe_ffn(blk.ffn, h.reshape(B * T, -1), cfg.top_k)
            if mi in self.layers:
                st["xmid"][mi] = x[:, s:].clone()
                st["topi"][mi] = topi.view(B, T, -1)[:, s:].clone()
                st["p"][mi] = p.view(B, T, -1)[:, s:].clone()
            x = x + yf.view(B, T, -1)
            mi += 1
        hN = m.norm_f(x)
        ce = _ce_rows(m, hN[:, s:T - 1], ids[:, s + 1:T])  # (B, R-1)
        rel = torch.arange(R - 1, device=dev).view(1, -1)
        st["valid"] = (s + 1 + rel) < ends.view(-1, 1)  # target position < end
        st["rvalid"] = (s + torch.arange(R, device=dev).view(1, -1)) < ends.view(-1, 1)  # region token inside doc
        st["ce_full"] = ce
        st["hlast"] = hN[:, s - 1].float()
        st["hmean"] = hN[:, max(0, s - 192):s].float().mean(1)
        st["ids"] = ids
        return st

    @torch.no_grad()
    def tail(self, st, mi, rows, experts):
        """Counterfactual CE (N, R-1) for pairs (rows[n], experts[n]); experts[n] = -1 means the null route
        (identical computation path, nothing killed)."""
        m = self.m
        cfg = m.cfg
        s, R = st["s"], st["R"]
        dev = st["ids"].device
        dt = m.tok_emb.weight.dtype
        rows_t = torch.as_tensor(rows, device=dev, dtype=torch.long)
        ex_t = torch.as_tensor(experts, device=dev, dtype=torch.long)
        N = rows_t.numel()
        b = mi + 1
        blk = m.blocks[b]
        xm = st["xmid"][mi].index_select(0, rows_t)
        topi = st["topi"][mi].index_select(0, rows_t)  # (N,R,k)
        sel = (topi == ex_t.clamp(min=0).view(N, 1, 1)).any(-1) & (ex_t >= 0).view(N, 1)
        sel = sel & st["rvalid"].index_select(0, rows_t)
        kill = torch.zeros(N, R, cfg.n_experts, dtype=torch.bool, device=dev)
        kill.scatter_(2, ex_t.clamp(min=0).view(N, 1, 1).expand(N, R, 1), sel.unsqueeze(-1))
        h = blk.norm2(xm)
        yf, _, _ = moe_ffn(blk.ffn, h.reshape(N * R, -1), cfg.top_k, kill.reshape(N * R, -1))
        x = xm + yf.view(N, R, -1)
        pos = (s + torch.arange(R, device=dev)).expand(N, R)
        cos, sin = rope_cos_sin(pos, cfg.d_model // cfg.n_head, cfg.rope_base, dt)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        qi = torch.arange(R, device=dev).view(R, 1)
        kj = torch.arange(s + R, device=dev).view(1, s + R)
        amask = (kj < s) | ((kj - s) <= qi)  # (R, s+R): whole prefix + causal within region
        for bi in range(b + 1, len(m.blocks)):
            blk = m.blocks[bi]
            q, k, v = _qkv(blk, x, cos, sin)
            pk, pv = st["kv"][bi]
            K = torch.cat([pk.index_select(0, rows_t), k], 2)
            V = torch.cat([pv.index_select(0, rows_t), v], 2)
            y = F.scaled_dot_product_attention(q, K, V, attn_mask=amask)
            x = x + blk.attn.proj(y.transpose(1, 2).reshape(N, R, -1))
            h = blk.norm2(x)
            if blk.is_moe:
                yf, _, _ = moe_ffn(blk.ffn, h.reshape(N * R, -1), cfg.top_k)
                x = x + yf.view(N, R, -1)
            else:
                x = x + blk.ffn(h)
        hN = m.norm_f(x)
        tgt = st["ids"].index_select(0, rows_t)[:, s + 1:s + R]
        return _ce_rows(m, hN[:, :R - 1], tgt), sel

    @torch.no_grad()
    def utilities(self, st, max_pairs=256, store_tokens=True):
        """All (layer in self.layers, expert) cells for every row. Returns dict of numpy arrays."""
        B, R = st["B"], st["R"]
        E = self.m.cfg.n_experts
        Lr = len(self.layers)
        valid = st["valid"]
        nv = valid.sum(1).clamp(min=1).float()
        u = np.zeros((B, Lr, E), np.float32)
        ud = np.zeros((B, Lr, E), np.float32)
        nsel = np.zeros((B, Lr, E), np.int16)
        pe = np.zeros((B, Lr, E), np.float16)
        tokd = np.zeros((B, Lr, E, R - 1), np.float16) if store_tokens else None
        ce_null = None
        for li, mi in enumerate(self.layers):
            base, _ = self.tail(st, mi, list(range(B)), [-1] * B)
            if ce_null is None:
                ce_null = base
            tp = st["topi"][mi]
            pm = st["p"][mi]
            rv = st["rvalid"]
            act = torch.zeros(B, E, dtype=torch.long, device=tp.device)
            for kk in range(tp.shape[-1]):
                act.scatter_add_(1, tp[..., kk] * 1, rv.long())
            pairs = [(r, e) for r in range(B) for e in range(E) if int(act[r, e]) > 0]
            for a in range(0, len(pairs), max_pairs):
                chunk = pairs[a:a + max_pairs]
                rows = [r for r, _ in chunk]
                exs = [e for _, e in chunk]
                cf, sel = self.tail(st, mi, rows, exs)
                rt = torch.as_tensor(rows, device=cf.device)
                d = (cf - base.index_select(0, rt)) * valid.index_select(0, rt)
                uu = d.sum(1) / nv.index_select(0, rt)
                dsel = sel[:, :R - 1] & valid.index_select(0, rt)
                udd = (d * dsel).sum(1) / dsel.sum(1).clamp(min=1)
                ns = sel.sum(1)
                psel = torch.stack([pm[r, :, e] for r, e in chunk])  # (n, R)
                pmean = (psel * sel).sum(1) / ns.clamp(min=1)
                uu, udd, ns, pmean = (t_.cpu().numpy() for t_ in (uu, udd, ns, pmean))
                dn = d.cpu().numpy().astype(np.float16) if store_tokens else None
                for n, (r, e) in enumerate(chunk):
                    u[r, li, e] = uu[n]
                    ud[r, li, e] = udd[n]
                    nsel[r, li, e] = ns[n]
                    pe[r, li, e] = pmean[n]
                    if store_tokens:
                        tokd[r, li, e] = dn[n]
        out = {"u": u, "ud": ud, "nsel": nsel, "pe": pe, "ce_null": ((ce_null * valid).sum(1) / nv).cpu().numpy(),
               "ce_full": ((st["ce_full"] * valid).sum(1) / nv).cpu().numpy(),
               "hlast": st["hlast"].cpu().numpy(), "hmean": st["hmean"].cpu().numpy(),
               "nvalid": valid.sum(1).cpu().numpy().astype(np.int16)}
        if store_tokens:
            out["tokd"] = tokd
        return out


def build_rows(sp, gdocs, s, R=R_LEN):
    ids = np.full((len(gdocs), s + R), 1, np.int64)
    for b, d in enumerate(gdocs):
        x = sp.doc(int(d))[:s + R]
        ids[b, :len(x)] = x
    return ids


# ------------------------------------------------------------------ probes (numpy / torch, CPU)
def standardize(Xtr, Xte):
    mu = Xtr.mean(0, keepdims=True)
    sd = Xtr.std(0, keepdims=True) + 1e-6
    return (Xtr - mu) / sd, (Xte - mu) / sd


def ridge_fit(X, Y, alpha):
    """X (n,p) standardized, Y (n,q) -> W (p+1, q) with an unpenalized intercept (centered solution)."""
    xm, ym = X.mean(0), Y.mean(0)
    Xc, Yc = X - xm, Y - ym
    n, p = Xc.shape
    if p <= n:
        A = Xc.T @ Xc + alpha * np.eye(p)
        W = np.linalg.solve(A, Xc.T @ Yc)
    else:
        K = Xc @ Xc.T + alpha * np.eye(n)
        W = Xc.T @ np.linalg.solve(K, Yc)
    b = ym - xm @ W
    return np.vstack([W, b[None]])


def ridge_pred(W, X):
    return X @ W[:-1] + W[-1]


def group_folds(groups, k=5, seed=2026):
    ug = np.unique(groups)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(ug)
    fold_of = {g: i % k for i, g in enumerate(perm)}
    return np.asarray([fold_of[g] for g in groups])


def ridge_cv(X, Y, groups, alphas=(1, 10, 100, 1e3, 1e4, 1e5), mask=None, k=5):
    """Choose alpha by grouped k-fold CV on masked squared error (mask (n,q) bool of scored entries)."""
    folds = group_folds(groups, k)
    best = None
    scores = {}
    for a in alphas:
        sse = 0.0
        for f in range(k):
            tr, va = folds != f, folds == f
            W = ridge_fit(X[tr], Y[tr], a)
            P = ridge_pred(W, X[va])
            err = (P - Y[va]) ** 2
            if mask is not None:
                err = err[mask[va]]
            sse += float(err.sum())
        scores[a] = sse
        if best is None or sse < scores[best]:
            best = a
    return best, scores


def logreg_fit(X, y, C, wd, iters=200):
    Xt = torch.as_tensor(X, dtype=torch.float64)
    yt = torch.as_tensor(y, dtype=torch.long)
    W = torch.zeros(X.shape[1], C, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(C, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(Xt @ W + b, yt) + wd * (W ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return W.detach().numpy(), b.detach().numpy()


def logreg_logprobs(Wb, X):
    W, b = Wb
    z = X @ W + b
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def logreg_cv(X, y, C, groups, wds=(1e-4, 1e-3, 1e-2, 1e-1), k=5):
    folds = group_folds(groups, k)
    best, scores = None, {}
    for wd in wds:
        tot = 0.0
        for f in range(k):
            tr, va = folds != f, folds == f
            lp = logreg_logprobs(logreg_fit(X[tr], y[tr], C, wd), X[va])
            tot += float(-lp[np.arange(va.sum()), y[va]].sum())
        scores[wd] = tot
        if best is None or tot < scores[best]:
            best = wd
    return best, scores


def auroc(score, label):
    label = label.astype(bool)
    npos, nneg = label.sum(), (~label).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    r = np.argsort(np.argsort(score, kind="mergesort"), kind="mergesort") + 1.0
    # average ranks for ties
    order = np.argsort(score, kind="mergesort")
    ss = score[order]
    ranks = np.empty(len(score))
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    del r
    return float((ranks[label].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def cluster_bootstrap(stat_fn, groups, n=2000, seed=2026):
    """stat_fn(index_array) -> float or array. Resamples groups (papers) with replacement."""
    ug, inv = np.unique(groups, return_inverse=True)
    members = [np.nonzero(inv == g)[0] for g in range(len(ug))]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        pick = rng.integers(0, len(ug), len(ug))
        idx = np.concatenate([members[g] for g in pick])
        out.append(stat_fn(idx))
    return np.asarray(out)
