"""csr-soft-cf-usefulness-v2 (scu): soft counterfactual-usefulness target (docs/SOFT_CF_USEFULNESS_V2_PREREG.md).

Reuses, read-only, the exact replacement estimator of csr-semantic-utility-edl-v1 (src/csr/sue_cf.py: trunk, row_ce,
ProbePlan, TargetBank, probe_pool, group_router_mean). New here:

  confidence_scores(d, se, valid, z, mode)  per-layer standardised, uncertainty-aware usefulness score s(k,l,e):
      mode "lcb":  s = LCB if LCB > 0, UCB if UCB < 0, else 0, with LCB/UCB = (DeltaU -/+ z*SE) / scale_l
      mode "mean": s = DeltaU / scale_l   (no uncertainty penalty; isolates estimator noise)
      unsupported (invalid) cells get s = 0 (no evidence, neither positive nor negative); |s| <= clip.
  soft_target(s, R, tau, form)  q(k,l,.) = normalise(R(k,l,.) * exp(s/tau))   (form "tilt": lagged router prior)
                                q(k,l,.) = softmax_e(s/tau)                       (form "uniform": operator's literal form)
      evidence(k,l) = any s(k,l,.) != 0; (k,l) without evidence contribute no loss term.
  cf_loss(p_layers, layers, qseq, seq_id, nonpad, Q, EV)  mean over (k present in the micro-batch, l, evidence) of
      KL(stopgrad q(k,l,.) || P(k,l,.)), P = gc_edl_loss's q-weighted group mean of the router softmax.
  Mechanism accumulators (diagnosis link D): prior R (EMA of P), exposure X (q-weighted top-k selection fractions),
      gradient attribution G (per-document expert-weight gradient norms, q-weighted by group)."""
import math

import numpy as np
import torch

from .sue_cf import group_router_mean  # noqa: F401  (re-exported)


def confidence_scores(d, se, valid, z=1.0, mode="lcb", clip=3.0):
    """d, se, valid: (K, L, E) from TargetBank.delta(). Returns s (K, L, E) and per-layer scale (L,)."""
    K, L, E = d.shape
    s = np.zeros_like(d)
    scale = np.zeros(L)
    for l_ in range(L):
        v = valid[:, l_, :]
        if v.sum() < 4:
            continue
        sc = float(np.std(d[:, l_, :][v]))
        if not (sc > 0 and math.isfinite(sc)):
            continue
        scale[l_] = sc
        dz, sz = d[:, l_, :] / sc, se[:, l_, :] / sc
        if mode == "lcb":
            lo, hi = dz - z * sz, dz + z * sz
            x = np.where(lo > 0, lo, np.where(hi < 0, hi, 0.0))
        elif mode == "mean":
            x = dz
        else:
            raise ValueError(mode)
        s[:, l_, :] = np.where(v, np.clip(x, -clip, clip), 0.0)
    return s, scale


def soft_target(s, R, tau, form="tilt"):
    """s (K, L, E) scores; R (K, L, E) lagged prior (rows sum to 1). Returns Q (K, L, E), EV (K, L) bool."""
    ev = np.abs(s).max(-1) > 0
    a = s / tau
    if form == "tilt":
        lq = np.log(np.maximum(R, 1e-12)) + a
    elif form == "uniform":
        lq = a.copy()
    else:
        raise ValueError(form)
    lq = lq - lq.max(-1, keepdims=True)
    q = np.exp(lq)
    q = q / q.sum(-1, keepdims=True)
    return q, ev


def target_stats(Q, R, EV):
    """Flatness / tilt diagnostics: mean entropy of q (nats) and mean KL(q || R) over evidence cells."""
    if not EV.any():
        return {"n_evidence": 0, "frac_evidence": 0.0, "q_entropy": float("nan"), "kl_q_prior": float("nan"),
                "q_max": float("nan")}
    q, r = Q[EV], np.maximum(R[EV], 1e-12)
    ent = -(q * np.log(np.maximum(q, 1e-12))).sum(-1)
    kl = (q * (np.log(np.maximum(q, 1e-12)) - np.log(r))).sum(-1)
    return {"n_evidence": int(EV.sum()), "frac_evidence": float(EV.mean()), "q_entropy": float(ent.mean()),
            "kl_q_prior": float(kl.mean()), "q_max": float(q.max(-1).mean())}


def cf_loss(p_layers, layers, qseq, seq_id, nonpad, Q, EV, min_mass=0.5):
    """p_layers: list of (N, E) router softmax (fp32) for MoE layers `layers`; Q (K, L, E) / EV (K, L) torch constants.
    Returns (loss, n_terms, mean KL, P list for the prior update)."""
    terms, Ps = [], []
    for p, l_ in zip(p_layers, layers):
        P, keep = group_router_mean(p, qseq, seq_id, nonpad)
        Ps.append((l_, P.detach(), keep))
        mass = qseq.to(P.dtype)
        m = nonpad & (seq_id >= 0)
        sid = seq_id[m]
        S = qseq.shape[0]
        has = torch.zeros(S, dtype=torch.bool, device=p.device)
        has[sid] = True
        gm = mass[has].sum(0)[keep]
        sel = EV[keep, l_] & (gm >= min_mass)
        if not bool(sel.any()):
            continue
        Pk = P[sel].clamp(min=1e-8)
        qk = Q[keep[sel], l_, :].to(Pk.dtype)
        terms.append((qk * (qk.clamp(min=1e-12).log() - Pk.log())).sum(-1))
    if not terms:
        z = p_layers[0].new_zeros(())
        return z, 0, float("nan"), Ps
    allt = torch.cat(terms)
    return allt.mean(), int(allt.numel()), float(allt.detach().mean()), Ps


class Mechanism:
    """Prior R (EMA of group-mean router distributions), exposure X and gradient attribution G (sums between
    diagnostic points). Arrays (K, L, E), numpy float64; part of every resumable checkpoint."""

    def __init__(self, K, L, E, prior_rho):
        self.K, self.L, self.E, self.rho = K, L, E, prior_rho
        self.R = np.full((K, L, E), 1.0 / E)
        self.r_upd = np.zeros((K, L))
        self.Xs, self.Xm = np.zeros((K, L, E)), np.zeros((K, L))
        self.Gs, self.Gm = np.zeros((K, L, E)), np.zeros(K)

    def state(self):
        return {k: getattr(self, k).copy() for k in ("R", "r_upd", "Xs", "Xm", "Gs", "Gm")}

    def load(self, st):
        for k, v in st.items():
            setattr(self, k, np.asarray(v).copy())

    def update_prior(self, Ps, qseq, seq_id, nonpad, min_mass=0.5):
        m = nonpad & (seq_id >= 0)
        sid = seq_id[m]
        has = torch.zeros(qseq.shape[0], dtype=torch.bool, device=qseq.device)
        has[sid] = True
        gm_all = qseq[has].sum(0).double().cpu().numpy()
        for l_, P, keep in Ps:
            kk = keep.cpu().numpy()
            Pn = P.double().cpu().numpy()
            for i, k in enumerate(kk):
                if gm_all[k] >= min_mass:
                    self.R[k, l_] = self.rho * self.R[k, l_] + (1 - self.rho) * Pn[i]
                    self.r_upd[k, l_] += 1

    def add_exposure(self, topi_layers, layers, qseq, seq_id, nonpad, E):
        """topi_layers: list of (N, k) top-k indices. X(k,l,e) = q-weighted fraction of group-k tokens with e in top-k."""
        m = nonpad & (seq_id >= 0)
        sid = seq_id[m]
        S = qseq.shape[0]
        ntok = torch.zeros(S, device=qseq.device, dtype=torch.float64).index_add_(
            0, sid, torch.ones_like(sid, dtype=torch.float64))
        q = qseq.double()
        for ti, l_ in zip(topi_layers, layers):
            cnt = torch.nn.functional.one_hot(ti[m], E).sum(1).double()
            cs = torch.zeros(S, E, device=qseq.device, dtype=torch.float64).index_add_(0, sid, cnt)
            self.Xs[:, l_] += (q.t() @ cs).cpu().numpy()
            self.Xm[:, l_] += (q.t() @ ntok).cpu().numpy()

    def add_grad(self, qdoc, gn2):
        """qdoc (K,) assignment of one document; gn2 (L, E) squared expert-weight gradient norms of its LM loss."""
        self.Gs += qdoc[:, None, None] * gn2[None]
        self.Gm += qdoc

    def snapshot_and_reset(self):
        X = self.Xs / np.maximum(self.Xm[..., None], 1e-12)
        G = self.Gs / np.maximum(self.Gm[:, None, None], 1e-12)
        out = {"R": self.R.copy(), "X": X, "X_mass": self.Xm.copy(), "G": G, "G_mass": self.Gm.copy()}
        self.Xs[:], self.Xm[:], self.Gs[:], self.Gm[:] = 0, 0, 0, 0
        return out


def expert_grad_norms(model, ids, dids, pos, targets, use_amp):
    """One document row: LM loss gradient w.r.t. every MoE layer's stacked expert weights -> (L, E) squared norms."""
    import torch.nn.functional as F
    ws = []
    for blk in model.blocks:  # MoE index order = block order (block 0 is dense)
        if blk.is_moe:
            ws += [blk.ffn.experts.w1, blk.ffn.experts.w2]
    with torch.autocast(device_type=ids.device.type, dtype=torch.bfloat16, enabled=use_amp):
        out = model({"input_ids": ids, "doc_ids": dids, "positions": pos}, {"teach": False, "edl": False, "lc": False,
                                                                           "cf_store": None, "tap": False})
        h = out["hidden"].reshape(-1, out["hidden"].shape[-1])
        y = targets.reshape(-1)
        lg = F.linear(h, model.tok_emb.weight).float()
        v = y >= 0
        loss = F.cross_entropy(lg[v], y[v])
    gs = torch.autograd.grad(loss, ws, allow_unused=True)
    L = len(ws) // 2
    E = ws[0].shape[0]
    out_ = np.zeros((L, E))
    for i in range(L):
        g1, g2 = gs[2 * i], gs[2 * i + 1]
        n = torch.zeros(E, device=ids.device, dtype=torch.float64)
        if g1 is not None:
            n += (g1.double() ** 2).sum((1, 2))
        if g2 is not None:
            n += (g2.double() ** 2).sum((1, 2))
        out_[i] = n.cpu().numpy()
    return out_
