"""csr-semantic-utility-edl-v1: exact counterfactual expert-usefulness targets and the usefulness-alignment loss.

  trunk()            lean no-grad forward of an LFRModel (router 'linear', no teacher) that (a) reproduces
                     LFRModel.forward, (b) caches the post-attention residual of every MoE block, and (c) can restart
                     at MoE layer l from that cache with a per-row expert removal: for the tokens of row s whose top-k
                     contains e_s, e_s's logit is set to -inf, the full softmax renormalises over the remaining experts,
                     top-k admits the next-ranked expert (exactly k active), and all downstream computation is redone.
                     Tokens that did not select e_s are untouched. Validated against LFRModel.forward with
                     exclude={l: [e]} and an explicit per-token exclude_mask (sue_tests.py).
  ProbePlan          deterministic probe rows (single-document rows from the main-split documents that are NOT in the
                     245M training plan: disjoint from the optimisation batches; held-out evidence uses other splits),
                     layer rotation and support-balanced expert choice. Independent of the assignment table.
  TargetBank         lagged EMA of q-weighted usefulness sums by (group k, layer, expert, difficulty bin), two
                     document halves for split-half reliability; DeltaU = own-group usefulness minus the
                     difficulty-matched (within-probe-batch base-CE quartile) usefulness on the other groups; support
                     thresholds (missing cells are MASKED, never zero); per-cell standard errors; empirical-Bayes
                     shrinkage, clipping and per-layer standardisation. Targets are numpy constants (stop-gradient).
  use_loss()         L_use = mean_{k,l} -log softmax_j(S_{k,j,l}/tau)[k], S_{k,j,l} = sum_e P_{k,l,e} Z_{j,l,e} over
                     cells valid for j (P renormalised over them); P_{k,l} = the Semantic-EDL group-mean router
                     distribution (gc_edl_loss definition)."""
from __future__ import annotations

import hashlib
import math

import numpy as np
import torch
import torch.nn.functional as F

from .model import rope_cos_sin

N_BIN = 4


# ------------------------------------------------------------------------------------------------ forward
def _route(moe, hf, k, rd, kill_expert_tok=None):
    with torch.autocast(device_type=hf.device.type, enabled=False):
        h32 = hf.to(rd)
        lS = F.linear(h32, moe.router.weight.to(rd))
        p = torch.softmax(lS, -1)
        gates, topi = p.topk(k, -1)
        if kill_expert_tok is not None:
            ke = kill_expert_tok
            sel = (ke >= 0) & (topi == ke.clamp(min=0).unsqueeze(1)).any(-1)
            if bool(sel.any()):
                l2 = lS.clone()
                rows = sel.nonzero().squeeze(1)
                l2[rows, ke[rows]] = float("-inf")
                g2, t2 = torch.softmax(l2, -1).topk(k, -1)
                gates = torch.where(sel.unsqueeze(1), g2, gates)
                topi = torch.where(sel.unsqueeze(1), t2, topi)
    return gates, topi


@torch.no_grad()
def trunk(model, ids, doc_ids, positions, cache=False, start=None, kill=None):
    """ids/doc_ids/positions (B,T). start = (mi0, x_mid) restarts at MoE layer mi0 from the cached post-attention
    residual; kill = (mi, kill_expert (B,) long, -1 = none). Returns dict(hidden, topi [per MoE layer (N,k)],
    mids [per MoE layer (B,T,d)] if cache)."""
    cfg = model.cfg
    assert cfg.router == "linear" and not cfg.teacher
    B, T = ids.shape
    N = B * T
    rd = torch.float64 if model.tok_emb.weight.dtype == torch.float64 else torch.float32
    cos, sin = rope_cos_sin(positions, cfg.d_model // cfg.n_head, cfg.rope_base, model.tok_emb.weight.dtype)
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    mask = model.attention_mask(doc_ids)
    x = model.tok_emb(ids) if start is None else None
    mids, topis = [], []
    mi = 0
    for blk in model.blocks:
        if start is not None and blk.is_moe and mi < start[0]:
            mi += 1
            continue
        if start is not None and not blk.is_moe:
            continue
        if start is not None and mi == start[0]:
            x = start[1]
        else:
            x = x + blk.attn(blk.norm1(x), cos, sin, mask)
        if not blk.is_moe:
            x = x + blk.ffn(blk.norm2(x))
            continue
        if cache:
            mids.append(x)
        moe = blk.ffn
        hf = blk.norm2(x).reshape(N, -1)
        ket = None
        if kill is not None and kill[0] == mi:
            ket = kill[1].repeat_interleave(T)
        gates, topi = _route(moe, hf, cfg.top_k, rd, ket)
        topis.append(topi)
        y = moe.experts(hf, topi, gates) + moe.shared(hf)
        x = x + y.view(B, T, -1)
        mi += 1
    return {"hidden": model.norm_f(x), "topi": topis, "mids": mids}


@torch.no_grad()
def row_ce(model, hidden, targets):
    """Mean CE per row over valid targets (B,) and valid counts (B,)."""
    B, T, d = hidden.shape
    hf, tf = hidden.reshape(-1, d), targets.reshape(-1)
    dt = torch.float64 if hidden.dtype == torch.float64 else torch.float32
    ce = torch.zeros(tf.numel(), device=hidden.device, dtype=dt)
    for s in range(0, tf.numel(), 8192):
        ce[s:s + 8192] = F.cross_entropy(model.lm_logits(hf[s:s + 8192]).to(dt), tf[s:s + 8192], ignore_index=-100,
                                         reduction="none")
    v = (tf >= 0).view(B, T).to(dt)
    ce = ce.view(B, T)
    n = v.sum(1)
    return (ce * v).sum(1) / n.clamp(min=1), n


# ------------------------------------------------------------------------------------------------ probe plan
def probe_pool(split, plan_steps, q_of_doc, min_len=256):
    """Main-split documents not used by the full training plan, with an assignment row and >= min_len tokens."""
    used = np.zeros(split.n_docs, bool)
    for s in plan_steps:
        for r in s:
            used[r] = True
    ok = (~used) & (q_of_doc.sum(1) > 0) & (split.lengths >= min_len)
    return np.nonzero(ok)[0]


class ProbePlan:
    """Stateless schedule: probe rows for step t, layer of pass m, expert choice (support-balanced)."""

    def __init__(self, pool, seed, rows, passes, layers):
        self.perm = np.random.default_rng(1_000_003 * seed + 77).permutation(pool)
        self.seed, self.rows, self.passes, self.layers = seed, rows, passes, list(layers)

    def docs(self, t):
        n = len(self.perm)
        i = (t * self.rows) % n
        idx = np.arange(i, i + self.rows) % n
        return self.perm[idx]

    def layer(self, t, m):
        return self.layers[(t * self.passes + m) % len(self.layers)]

    def experts(self, t, m, topi_rows, valid_rows, support_le):
        """topi_rows (B,T,k) base top-k at the pass layer; support_le (E,) raw observation counts at that layer.
        For each row: the selected expert with the least support (random tie-break, seeded by (seed, t, m))."""
        rng = np.random.default_rng([self.seed, t, m, 9])
        B, _, k = topi_rows.shape
        E = support_le.shape[0]
        out = np.full(B, -1, np.int64)
        for b in range(B):
            sel = np.unique(topi_rows[b][valid_rows[b]].reshape(-1))
            if len(sel) == 0:
                continue
            s = support_le[sel] + rng.random(len(sel)) * 0.5
            out[b] = int(sel[np.argmin(s)])
        assert out.max() < E
        return out


def doc_half(gdoc):
    return int.from_bytes(hashlib.sha256(f"sue-half|{int(gdoc)}".encode()).digest()[:4], "little") & 1


# ------------------------------------------------------------------------------------------------ target bank
class TargetBank:
    FIELDS = ("num", "den", "sq", "den2")

    def __init__(self, K, L, E, rho, m_min=2.0, n_min=8, clip=3.0, nb=N_BIN):
        self.K, self.L, self.E, self.nb = K, L, E, nb
        self.rho, self.m_min, self.n_min, self.clip = rho, m_min, n_min, clip
        sh = (2, K, L, E, nb)
        self.a = {f: np.zeros(sh) for f in self.FIELDS}
        self.cnt = np.zeros((2, L, E))
        self.support = np.zeros((L, E))  # raw, undecayed observation counts (sampling balance + coverage)
        self.n_obs = 0

    def state(self):
        return {"a": {k: v.copy() for k, v in self.a.items()}, "cnt": self.cnt.copy(), "support": self.support.copy(),
                "n_obs": self.n_obs, "rho": self.rho}

    def load(self, st):
        self.a = {k: np.asarray(v).copy() for k, v in st["a"].items()}
        self.cnt, self.support = np.asarray(st["cnt"]).copy(), np.asarray(st["support"]).copy()
        self.n_obs = st["n_obs"]

    def decay(self):
        for v in self.a.values():
            v *= self.rho
        self.cnt *= self.rho

    def add(self, q, layer, expert, u, b, half):
        """q (n,K) doc assignments; layer/expert/b/half (n,) ints; u (n,) usefulness (CE_replace - CE_base)."""
        for i in range(len(u)):
            if expert[i] < 0 or not math.isfinite(u[i]):
                continue
            h, l_, e, bb, qq = int(half[i]), int(layer[i]), int(expert[i]), int(b[i]), q[i]
            self.a["num"][h, :, l_, e, bb] += qq * u[i]
            self.a["den"][h, :, l_, e, bb] += qq
            self.a["sq"][h, :, l_, e, bb] += qq * u[i] * u[i]
            self.a["den2"][h, :, l_, e, bb] += qq * qq
            self.cnt[h, l_, e] += 1
            self.support[l_, e] += 1
            self.n_obs += 1

    def delta(self, halves=(0, 1)):
        """Raw DeltaU (K,L,E), SE (K,L,E), valid mask (K,L,E) from the given halves."""
        A = {f: self.a[f][list(halves)].sum(0) for f in self.FIELDS}
        cnt = self.cnt[list(halves)].sum(0)
        tot = {f: A[f].sum(0, keepdims=True) for f in self.FIELDS}
        own = A
        oth = {f: tot[f] - own[f] for f in self.FIELDS}
        eps = 1e-9
        ok_b = (own["den"] > eps) & (oth["den"] > eps)
        uo = np.where(ok_b, own["num"] / np.maximum(own["den"], eps), 0.0)
        ux = np.where(ok_b, oth["num"] / np.maximum(oth["den"], eps), 0.0)
        w = np.where(ok_b, own["den"], 0.0)
        wsum = w.sum(-1, keepdims=True)
        w = w / np.maximum(wsum, eps)
        d = (w * (uo - ux)).sum(-1)
        vo = np.maximum(0.0, np.where(ok_b, own["sq"] / np.maximum(own["den"], eps) - uo ** 2, 0.0))
        vx = np.maximum(0.0, np.where(ok_b, oth["sq"] / np.maximum(oth["den"], eps) - ux ** 2, 0.0))
        neo = np.where(ok_b, own["den"] ** 2 / np.maximum(own["den2"], eps), 1.0)
        nex = np.where(ok_b, oth["den"] ** 2 / np.maximum(oth["den2"], eps), 1.0)
        se2 = (w ** 2 * (vo / np.maximum(neo, 1.0) + vx / np.maximum(nex, 1.0))).sum(-1)
        own_m = np.where(ok_b, own["den"], 0.0).sum(-1)
        oth_m = np.where(ok_b, oth["den"], 0.0).sum(-1)
        valid = (own_m >= self.m_min) & (oth_m >= self.m_min) & (cnt[None] >= self.n_min) & (wsum[..., 0] > eps)
        return d, np.sqrt(se2), valid

    def targets(self):
        """Standardised shrunk targets Z (K,L,E) (0 where invalid), valid mask, diagnostics."""
        d, se, valid = self.delta()
        Z = np.zeros_like(d)
        lay = []
        for l_ in range(self.L):
            v = valid[:, l_, :]
            if v.sum() < 4:
                lay.append({"layer": l_, "n_valid": int(v.sum()), "signal_var": 0.0, "noise_var": None})
                continue
            dv, sv = d[:, l_, :][v], se[:, l_, :][v]
            s2 = max(0.0, float(dv.var() - (sv ** 2).mean()))
            if s2 > 0:
                z = math.sqrt(s2) * d[:, l_, :] / (s2 + se[:, l_, :] ** 2)
                Z[:, l_, :] = np.where(v, np.clip(z, -self.clip, self.clip), 0.0)
            lay.append({"layer": l_, "n_valid": int(v.sum()), "signal_var": s2, "noise_var": float((sv ** 2).mean()),
                        "raw_var": float(dv.var())})
        # split-half reliability on cells valid in both halves
        dA, _, vA = self.delta((0,))
        dB, _, vB = self.delta((1,))
        both = vA & vB
        r = float(np.corrcoef(dA[both], dB[both])[0, 1]) if both.sum() > 10 else float("nan")
        diag = {"n_obs": self.n_obs, "coverage_valid": float(valid.mean()), "n_valid": int(valid.sum()),
                "split_half_r": r, "split_half_n": int(both.sum()),
                "spearman_brown": (2 * r / (1 + r)) if math.isfinite(r) and r > -1 else float("nan"),
                "layers": lay, "support_min": float(self.support.min()), "support_max": float(self.support.max()),
                "support_cv": float(self.support.std() / max(1e-9, self.support.mean()))}
        return Z, valid, diag


# ------------------------------------------------------------------------------------------------ loss
def group_router_mean(p, qseq, seq_id, nonpad, min_mass=1e-6):
    """gc_edl_loss's P: per-sequence mean router probability -> q-weighted group means. Returns (P (Kk,E), keep idx)."""
    m = nonpad & (seq_id >= 0)
    p, sid = p[m], seq_id[m]
    S = qseq.shape[0]
    E = p.shape[1]
    ssum = p.new_zeros(S, E).index_add_(0, sid, p)
    scnt = p.new_zeros(S).index_add_(0, sid, torch.ones_like(sid, dtype=p.dtype))
    ok = scnt > 0
    smean = ssum[ok] / scnt[ok].unsqueeze(1)
    q = qseq[ok].to(p.dtype)
    mass = q.sum(0)
    keep = torch.nonzero(mass > min_mass).squeeze(1)
    P = (q[:, keep].t() @ smean) / mass[keep].unsqueeze(1)
    return P, keep


def use_loss(p_layers, layers, qseq, seq_id, nonpad, Z, V, tau, min_cov=0.5):
    """p_layers: list of (N,E) router softmax for the MoE layers `layers`; Z/V (K,L,E) torch constants (no grad).
    Returns (loss, n_active_terms, mean own-group margin S_kk - mean_j S_kj)."""
    terms, margins = [], []
    for p, l_ in zip(p_layers, layers):
        P, keep = group_router_mean(p, qseq, seq_id, nonpad)
        Vl, Zl = V[:, l_, :].to(P.dtype), Z[:, l_, :].to(P.dtype)
        J = torch.nonzero(Vl.mean(1) >= min_cov).squeeze(1)
        if J.numel() < 2:
            continue
        kin = torch.isin(keep, J)
        if not bool(kin.any()):
            continue
        rows = keep[kin]
        Pk = P[kin]
        W = Pk.unsqueeze(1) * Vl[J].unsqueeze(0)
        S = (W * Zl[J].unsqueeze(0)).sum(-1) / W.sum(-1).clamp(min=1e-12)
        tgt = torch.searchsorted(J, rows)
        terms.append(F.cross_entropy(S / tau, tgt, reduction="none"))
        with torch.no_grad():
            margins.append((S.gather(1, tgt.view(-1, 1)).squeeze(1) - S.mean(1)))
    if not terms:
        z = p_layers[0].new_zeros(())
        return z, 0, float("nan")
    allt = torch.cat(terms)
    return allt.mean(), int(allt.numel()), float(torch.cat(margins).mean())
