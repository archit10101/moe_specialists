"""csr-semantic-lift-expert-learning-v1 (sll): Semantic Lift-Weighted Expert Learning (training-only gradient operator).

Nothing here changes a forward value, a routing decision, a gate, a router gradient, a shared-expert gradient, an
attention/non-expert gradient or the gradient passed to the expert input. Only the contribution of token i to the
parameter gradient of its selected routed expert e at MoE layer l is multiplied by m[i,l,e] (prereg sections 4-5):

  running stats (segment level, detached, previous optimizer steps only, committed AFTER each optimizer step):
      P[k,l,e] = (N[k,l,e] + n0 * Pbar[l,e]) / (D[k] + n0)      N = EMA_rho sum_seg q(seg,k) * mean_t p(t,l,e)
                                                                  D = EMA_rho sum_seg q(seg,k)
      Pbar[l,e] = sum_k N[k,l,e] / sum_k D[k]  (shrinkage target), P[l,e] = sum_k pi_k P[k,l,e], pi_k = D[k] / sum D
      lift L[l,k,e] = log(P[k,l,e] + eps) - log(P[l,e] + eps)
  token lift      s[i,l,e] = sum_k q(i,k) L[l,k,e]     (q = the frozen per-paper assignment of token i's segment)
  standardised    z = s / max(floor, scale_l)          scale_l = bias-corrected EMA of 1.4826 * MAD of s over the
                                                       selected (token, expert) pairs of previous steps
  raw ratio       r = exp(lambda_eff * tanh(z))
  multiplier      m = clip(c * r, lo, hi) with c > 0 solved EXACTLY (piecewise-linear root) so that
                  sum_{e in S(i,l)} ghat[i,l,e] m[i,l,e] = 1, ghat = gates / sum(gates) (top-4 gates, renormalised)
  gradient        d theta_e <- sum_i m[i,l,e] * (d theta_e)_i   via a custom autograd Function per expert MLP.

Schedule (fraction of the registered 3,880-step plan): steps [0, c) collect statistics with m = 1 (native code path),
steps [c, 2c) ramp lambda_eff linearly 0 -> lambda, steps >= 2c full lambda; c = round(0.05 * n_steps)."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from .lfr_model import Experts

EPS = 1e-6
MAD_C = 1.4826


# ------------------------------------------------------------------------------------------------ projection
def project_weighted_mean_one(r, w, lo=0.5, hi=2.0):
    """Exact bounded projection. r (N,k) > 0, w (N,k) >= 0 with rows summing to 1. Returns m = clip(c r, lo, hi) with
    c > 0 such that sum_j w_j m_j = 1 for every row (requires lo <= 1 <= hi). f(c) = sum_j w_j clip(c r_j, lo, hi) is
    continuous, nondecreasing and linear between the 2k breakpoints {lo/r_j, hi/r_j}; f(0+) = lo, f(inf) = hi, so the
    root lies on one linear segment and is found in closed form. Computed in float64."""
    r64, w64 = r.double(), w.double()
    N, k = r64.shape
    bp = torch.cat([lo / r64, hi / r64], 1).sort(1).values  # (N, 2k)
    f = (w64.unsqueeze(1) * (bp.unsqueeze(2) * r64.unsqueeze(1)).clamp(lo, hi)).sum(2)  # (N, 2k)
    ge = f >= 1.0
    j = torch.where(ge.any(1), ge.float().argmax(1), torch.full((N,), 2 * k - 1, device=r.device))
    j = j.clamp(min=1)
    b1, b0 = bp.gather(1, j.view(-1, 1)).squeeze(1), bp.gather(1, (j - 1).view(-1, 1)).squeeze(1)
    f1, f0 = f.gather(1, j.view(-1, 1)).squeeze(1), f.gather(1, (j - 1).view(-1, 1)).squeeze(1)
    den = f1 - f0
    c = torch.where(den > 0, b0 + (1.0 - f0) * (b1 - b0) / den.clamp(min=1e-300), b1)
    m = (c.unsqueeze(1) * r64).clamp(lo, hi)
    return m


def project_bisect(r, w, lo=0.5, hi=2.0, iters=200):
    """Reference (tests only): bisection on log c."""
    r64, w64 = r.double(), w.double()
    a = torch.full((r.shape[0],), -60.0, dtype=torch.float64, device=r.device)
    b = torch.full_like(a, 60.0)
    for _ in range(iters):
        mid = 0.5 * (a + b)
        f = (w64 * (torch.exp(mid).unsqueeze(1) * r64).clamp(lo, hi)).sum(1)
        a, b = torch.where(f < 1.0, mid, a), torch.where(f < 1.0, b, mid)
    return (torch.exp(0.5 * (a + b)).unsqueeze(1) * r64).clamp(lo, hi)


def multipliers(s_sel, gates, lam, scale, lo=0.5, hi=2.0):
    """s_sel (N,k) token lift of the selected experts, gates (N,k) executed gates, scale (scalar > 0)."""
    z = s_sel.double() / scale
    r = torch.exp(lam * torch.tanh(z))
    g = gates.detach().double()
    w = g / g.sum(1, keepdim=True).clamp(min=1e-300)
    return project_weighted_mean_one(r, w, lo, hi), w


# ------------------------------------------------------------------------------------------------ expert operator
def _autocast_dtype(dev_type):
    try:
        return torch.get_autocast_dtype(dev_type)
    except AttributeError:  # older torch
        return torch.get_autocast_gpu_dtype() if dev_type == "cuda" else torch.get_autocast_cpu_dtype()


class ScaledExpertMLP(torch.autograd.Function):
    """y = gelu(x W1) W2 (tanh GELU). Forward identical to the native expert computation (same casts under autocast,
    same kernels). Backward: dx is the ordinary gradient; dW1, dW2 use the per-row multiplied upstream gradient
    (row i scaled by m_i), i.e. dW = sum_i m_i (dW)_i. m is a detached constant (no gradient)."""

    @staticmethod
    def forward(ctx, x, w1, w2, m):
        dev = x.device.type
        ac = torch.is_autocast_enabled(dev) if dev != "cpu" else torch.is_autocast_cpu_enabled()
        dt = _autocast_dtype(dev) if ac else x.dtype
        with torch.autocast(device_type=dev, enabled=False):
            xb, w1b, w2b = x.to(dt), w1.to(dt), w2.to(dt)
            pre = xb @ w1b
            a = F.gelu(pre, approximate="tanh")
            out = a @ w2b
        ctx.save_for_backward(xb, w1b, w2b, pre, a, m)
        ctx.dts = (x.dtype, w1.dtype, w2.dtype)
        return out

    @staticmethod
    def backward(ctx, g):
        xb, w1b, w2b, pre, a, m = ctx.saved_tensors
        dev = g.device.type
        with torch.autocast(device_type=dev, enabled=False):
            g = g.to(a.dtype)
            mm = m.view(-1, 1)
            acc = torch.promote_types(g.dtype, torch.float32)
            gm = (g.to(acc) * mm.to(acc)).to(g.dtype)
            gw2 = a.t() @ gm
            da = g @ w2b.t()
            dpre = torch.ops.aten.gelu_backward(da, pre, approximate="tanh")
            dx = dpre @ w1b.t()
            dprem = (dpre.to(acc) * mm.to(acc)).to(dpre.dtype)
            gw1 = xb.t() @ dprem
        return dx.to(ctx.dts[0]), gw1.to(ctx.dts[1]), gw2.to(ctx.dts[2]), None


class LiftContext:
    """Per-step / per-micro-batch state shared by the 11 LiftExperts modules (set by the trainer)."""

    def __init__(self, n_moe, K, E, lo=0.5, hi=2.0):
        self.n_moe, self.K, self.E, self.lo, self.hi = n_moe, K, E, lo, hi
        self.lam = 0.0          # lambda_eff of this step
        self.table = None       # (L, K, E) lift table (float32, device) from previous steps
        self.scale = None       # list (L,) of python floats: effective (floored) standardisation scale, previous steps
        self.q_tok = None       # (N, K) per-token assignment of the current micro-batch (zeros for pads / no q)
        self.nonpad = None      # (N,) bool
        self.record = False     # collect diagnostics this micro-batch
        self.diag = {}          # layer -> dict of diagnostics
        self.s_mad = {}         # layer -> list of (median, MAD) of s over selected pairs (for the scale EMA)
        self.m_lists = {}       # layer -> sampled multipliers (diagnostics)
        self.force_m = None     # tests: layer -> explicit (N,k) multipliers

    def reset_step(self):
        self.diag, self.s_mad, self.m_lists = {}, {}, {}


class LiftExperts(Experts):
    """Experts with the same Parameters (w1, w2) and the same sorted dropless dispatch; with lam > 0 the per-expert
    MLP runs through ScaledExpertMLP with the token multipliers. With lam == 0 (collection phase, lambda = 0 control)
    the forward is the native Experts.forward (bit-identical code path)."""

    def __init__(self, base: Experts, layer: int, ctx: LiftContext):
        torch.nn.Module.__init__(self)
        self.E = base.E
        self.w1, self.w2 = base.w1, base.w2  # same Parameter objects (state_dict / optimizer unchanged)
        self.layer = layer
        self.ctx = ctx

    @torch.no_grad()
    def _lift_sel(self, topi):
        c = self.ctx
        qL = c.q_tok.to(torch.float32) @ c.table[self.layer]  # (N, E)
        return qL.gather(1, topi)

    @torch.compiler.disable
    def forward(self, x, topi, gates):
        c = self.ctx
        if c is None or c.table is None or c.q_tok is None:
            return Experts.forward(self, x, topi, gates)
        s_sel = self._lift_sel(topi)
        npm = c.nonpad
        with torch.no_grad():
            sv = s_sel[npm].reshape(-1).float()
            if sv.numel():
                med = sv.median()
                c.s_mad.setdefault(self.layer, []).append((sv - med).abs().median())
        if c.force_m is None and c.lam <= 0.0:
            return Experts.forward(self, x, topi, gates)
        with torch.no_grad():
            if c.force_m is not None:
                m = c.force_m[self.layer].double()
                w = gates.detach().double() / gates.detach().double().sum(1, keepdim=True)
            else:
                m, w = multipliers(s_sel, gates, c.lam, c.scale[self.layer], c.lo, c.hi)
            m = torch.where(npm.view(-1, 1), m, torch.ones_like(m))
            if c.record:
                mv = m[npm]
                wv = w[npm]
                dev1 = ((wv * mv).sum(1) - 1.0).abs().max() if mv.numel() else torch.zeros((), device=m.device)
                mf = mv.reshape(-1).float()
                c.diag[self.layer] = {
                    "m_mean": float(mf.mean()), "m_std": float(mf.std()), "m_wmean_maxdev": float(dev1),
                    "m_min": float(mf.min()), "m_max": float(mf.max()),
                    "clip_lo": float((mf <= c.lo + 1e-9).float().mean()), "clip_hi": float((mf >= c.hi - 1e-9).float().mean()),
                    "m_absdev_mean": float((mf - 1).abs().mean()),
                    "m_absdev_wmean": float((wv * (mv - 1).abs()).sum(1).mean()),
                    "q": [float(x) for x in torch.quantile(mf[:: max(1, mf.numel() // 200000)],
                                                            torch.tensor([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99],
                                                                         device=mf.device)).tolist()],
                    "s_absmean": float(sv.abs().mean()) if sv.numel() else 0.0}
            m32 = m.to(torch.float64 if x.dtype == torch.float64 else torch.float32)
        N, d = x.shape
        k = topi.shape[1]
        flat = topi.reshape(-1)
        order = torch.argsort(flat, stable=True)
        counts = torch.bincount(flat, minlength=self.E).tolist()
        tok = order // k
        xs = x.index_select(0, tok)
        ms = m32.reshape(-1).index_select(0, order)
        w1s, w2s = self.w1.to(x.dtype).unbind(0), self.w2.to(x.dtype).unbind(0)
        parts = torch.split(xs, counts)
        mparts = torch.split(ms, counts)
        outs = [ScaledExpertMLP.apply(parts[e], w1s[e], w2s[e], mparts[e]) for e, cnt in enumerate(counts) if cnt]
        ys = torch.cat(outs, 0) * gates.reshape(-1)[order].to(x.dtype).unsqueeze(-1)
        out = torch.empty_like(ys)
        out[order] = ys
        return out.view(N, k, d).sum(1)


def install(model, ctx):
    """Replace every MoE layer's Experts by LiftExperts sharing its Parameters. Returns the list of modules."""
    mods = []
    mi = 0
    for blk in model.blocks:
        if not blk.is_moe:
            continue
        le = LiftExperts(blk.ffn.experts, mi, ctx)
        blk.ffn.experts = le
        mods.append(le)
        mi += 1
    return mods


# ------------------------------------------------------------------------------------------------ running statistics
class LiftStats:
    """Detached running (cluster, layer, expert) routing statistics at the segment level (the Semantic-EDL
    representation: per-segment mean router softmax p, weighted by the segment's frozen q). EMA with factor rho per
    optimizer step; the ratio N/D is the exponentially weighted mean (no zero-initialisation bias); shrinkage of
    n0 segment-weights toward the pooled mean protects small clusters; the standardisation scale is a bias-corrected
    EMA of the per-step robust scale, floored."""

    def __init__(self, K, L, E, rho, n0, floor, device):
        self.K, self.L, self.E = K, L, E
        self.rho, self.n0, self.floor = rho, n0, floor
        self.dev = device
        z = lambda *s: torch.zeros(*s, dtype=torch.float64, device=device)  # noqa: E731
        self.N, self.D, self.D2 = z(K, L, E), z(K), z(K)
        self.sc_num, self.sc_w = z(L), 0.0
        self.n_commits = 0
        self.snap = []  # [(step, lift table numpy)] every snap_every commits (stability)
        self._reset_acc()

    def _reset_acc(self):
        z = lambda *s: torch.zeros(*s, dtype=torch.float64, device=self.dev)  # noqa: E731
        self.aN, self.aD, self.aD2 = z(self.K, self.L, self.E), z(self.K), z(self.K)
        self.a_sc, self.a_sc_n = z(self.L), torch.zeros(self.L, dtype=torch.float64, device=self.dev)

    @torch.no_grad()
    def accumulate(self, p_layers, qseq, seq_id, nonpad):
        """p_layers: list (len L) of (N, E) router softmax tensors; qseq (S, K) float; seq_id (N,); nonpad (N,)."""
        m = nonpad & (seq_id >= 0)
        sid = seq_id[m]
        if sid.numel() == 0:
            return
        S = qseq.shape[0]
        cnt = torch.zeros(S, dtype=torch.float64, device=self.dev).index_add_(0, sid, torch.ones_like(sid, dtype=torch.float64))
        ok = cnt > 0
        q = qseq.to(self.dev, torch.float64)[ok]
        sm = []
        for p in p_layers:
            ss = torch.zeros(S, self.E, dtype=torch.float64, device=self.dev).index_add_(0, sid, p.detach()[m].double())
            sm.append(ss[ok] / cnt[ok].unsqueeze(1))
        smean = torch.stack(sm, 1)  # (S', L, E)
        self.aN += torch.einsum("sk,sle->kle", q, smean)
        self.aD += q.sum(0)
        self.aD2 += (q * q).sum(0)

    def accumulate_scale(self, s_mad):
        """s_mad: layer -> list of 0-d tensors (MAD of s over the selected pairs of one micro-batch)."""
        for l_, lst in s_mad.items():
            for mad in lst:
                self.a_sc[l_] += MAD_C * mad.double()
                self.a_sc_n[l_] += 1

    @torch.no_grad()
    def commit(self, step=None, snap_every=10, keep=12):
        r = self.rho
        self.N = r * self.N + self.aN
        self.D = r * self.D + self.aD
        self.D2 = (r * r) * self.D2 + self.aD2
        has = self.a_sc_n > 0
        if bool(has.any()):
            cur = torch.where(has, self.a_sc / self.a_sc_n.clamp(min=1), self.scale_raw())
            self.sc_num = r * self.sc_num + (1 - r) * cur
            self.sc_w = r * self.sc_w + (1 - r)
        self.n_commits += 1
        self._reset_acc()
        if step is not None and self.n_commits % snap_every == 0:
            self.snap.append((int(step), self.lift().cpu().numpy().astype(np.float32)))
            self.snap = self.snap[-keep:]

    def scale_raw(self):
        if self.sc_w <= 0:
            return torch.full((self.L,), self.floor, dtype=torch.float64, device=self.dev)
        return self.sc_num / self.sc_w

    def scale(self):
        return self.scale_raw().clamp(min=self.floor)

    def probs(self):
        tot = self.D.sum()
        if float(tot) <= 0:
            return None, None
        pbar = self.N.sum(0) / tot  # (L, E)
        Pk = (self.N + self.n0 * pbar.unsqueeze(0)) / (self.D + self.n0).view(-1, 1, 1)
        pi = self.D / tot
        Pm = (pi.view(-1, 1, 1) * Pk).sum(0)
        return Pk, Pm

    def lift(self):
        """(L, K, E) float64 lift table (zeros before any statistics)."""
        Pk, Pm = self.probs()
        if Pk is None:
            return torch.zeros(self.L, self.K, self.E, dtype=torch.float64, device=self.dev)
        return (torch.log(Pk + EPS) - torch.log(Pm + EPS).unsqueeze(0)).permute(1, 0, 2).contiguous()

    def ess(self):
        """Kish effective number of segments per cluster: (sum w)^2 / sum w^2 with w = rho^age * q."""
        return (self.D ** 2 / self.D2.clamp(min=1e-300)).cpu().numpy()

    def stability(self, lag_snaps=5):
        """Pearson r between the current lift table and the one lag_snaps snapshots earlier (all cells)."""
        if len(self.snap) <= lag_snaps:
            return float("nan"), None
        a, b = self.snap[-1][1].reshape(-1), self.snap[-1 - lag_snaps][1].reshape(-1)
        if a.std() <= 0 or b.std() <= 0:
            return float("nan"), self.snap[-1 - lag_snaps][0]
        return float(np.corrcoef(a, b)[0, 1]), self.snap[-1 - lag_snaps][0]

    def summary(self):
        Lt = self.lift().cpu().numpy()  # (L, K, E)
        Pk, _ = self.probs()
        ess = self.ess()
        out = {"n_commits": self.n_commits, "ess_min": float(ess.min()), "ess_median": float(np.median(ess)),
               "pi": (self.D / self.D.sum().clamp(min=1e-300)).cpu().numpy().round(4).tolist(),
               "lift_absmean": float(np.abs(Lt).mean()), "lift_absmax": float(np.abs(Lt).max()),
               "lift_absmean_by_layer": np.abs(Lt).mean((1, 2)).round(5).tolist(),
               "lift_absmean_by_cluster": np.abs(Lt).mean((0, 2)).round(5).tolist(),
               "lift_max_by_expert_mean_over_layers": Lt.max(1).mean(0).round(5).tolist(),
               "scale_raw": self.scale_raw().cpu().numpy().round(5).tolist()}
        st, lag = self.stability()
        out["stability_r"], out["stability_vs_step"] = st, lag
        if Pk is not None:  # overlap of the clusters' top-4 preferred experts (mean pairwise Jaccard, per layer)
            top = Pk.argsort(-1, descending=True)[..., :4].cpu().numpy()  # (K, L, 4)
            jac = []
            for l_ in range(self.L):
                sets = [set(top[k, l_].tolist()) for k in range(self.K)]
                js = [len(sets[a] & sets[b]) / len(sets[a] | sets[b]) for a in range(self.K) for b in range(a + 1, self.K)]
                jac.append(float(np.mean(js)))
            out["cluster_top4_jaccard_by_layer"] = [round(x, 4) for x in jac]
        return out

    def state(self):
        return {"N": self.N.cpu().numpy(), "D": self.D.cpu().numpy(), "D2": self.D2.cpu().numpy(),
                "sc_num": self.sc_num.cpu().numpy(), "sc_w": self.sc_w, "n_commits": self.n_commits,
                "snap": [(s, a) for s, a in self.snap], "cfg": [self.K, self.L, self.E, self.rho, self.n0, self.floor]}

    def load(self, s):
        if list(s["cfg"]) != [self.K, self.L, self.E, self.rho, self.n0, self.floor]:
            raise ValueError("LiftStats config mismatch on resume")
        t = lambda a: torch.as_tensor(a, dtype=torch.float64, device=self.dev)  # noqa: E731
        self.N, self.D, self.D2, self.sc_num = t(s["N"]), t(s["D"]), t(s["D2"]), t(s["sc_num"])
        self.sc_w, self.n_commits = float(s["sc_w"]), int(s["n_commits"])
        self.snap = [(int(a), np.asarray(b)) for a, b in s["snap"]]
        self._reset_acc()


def lam_at(step, n_steps, lam, frac=0.05):
    """Registered schedule: c = round(frac * n_steps); 0 on [0, c); linear ramp on [c, 2c); lam afterwards."""
    c = int(round(frac * n_steps))
    if step < c:
        return 0.0
    if step < 2 * c:
        return lam * (step - c + 1) / c
    return lam


def q_tokens(qseq, seq_id):
    """(N, K) per-token assignment from per-segment q (zeros where seq_id < 0)."""
    sid = seq_id.reshape(-1)
    q = qseq[sid.clamp(min=0)]
    return q * (sid >= 0).to(q.dtype).unsqueeze(1)
