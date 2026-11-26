"""scrgp-v1: Subject-Conditional Residual Gradient Projection — probe library (prereg docs/SCRGP_V1_PREREG.md).

Nothing here modifies a model. The probe re-implements the LFR forward (linear router, no teacher) so that a single MoE
layer l can be run with expert e forced into the fourth routed slot, and the exact next-token-CE gradient with respect to
that expert's W1/W2 can be taken for several token-weight vectors from one forward:

  natural pass (cached per MoE layer l): x_mid (post-attention residual), hf = norm2(x_mid), router p, top-4 (gates =
      p of the selected experts, no renormalisation, exactly as the model), per-slot expert outputs slots (N, 4, d)
      from an exact replica of Experts.forward, shared output sh. Natural output = slots.sum(1) + sh (the model's ops).
  forced insertion of e at layer l: on every position where e is NOT among the top-3, slot 4 is replaced by e with the
      ORIGINAL slot-4 gate g4 = p_(4) (positions where e is already the 4th expert are unchanged); the three
      unrestricted top experts, the shared expert, attention and every later layer (natural routing) are unchanged.
      Positions where e is in the top-3 keep natural routing and carry no gradient to theta_e (stop-gradient copy).
  gradient: theta_e = (w1[e], w2[e]) as fresh leaf tensors (every model parameter has requires_grad False, so nothing
      else can receive gradient); dCE_t/dhidden_t is computed once per position (the LM head is per position), then
      for weight vector w: grad_theta_e sum_t w_t CE_t = VJP(hidden -> theta_e, w[:, None] * c) (exact, linear in w).

Matching (coarsened exact matching, ATT weights): for MoE layer l, expert e and table tau with associated cluster
k* = k*(tau, l, e), eligible tokens (valid target, e not in top-3) are stratified by
  CE quintile x position bin (<128, 128-511, >=512) x rank bin of e (=4, 5-8, >=9) x gate tercile (g4) x margin
  tercile (log g4 - log p_e; 0 for rank 4) x target-token frequency tercile            (5*3*3*3*3*3 = 1215 strata)
  positive weight w+ = q_t(k*); control weight w- = (1 - q_t(k*)) * P(s)/C(s); strata without both sides dropped;
  both normalised to sum 1 over the sub-sample -> identical stratum distributions.
  balanced global weight wg = sum_k q_t(k) / (K * M_k), M_k = sum over eligible tokens of q_t(k).
  d = grad(w+ - w-) . CE, g = grad(wg . CE), u = d - (d.g / (|g|^2 + eps)) g.   (no normalisation of d or u)
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
import torch
import torch.nn.functional as F

from .fsp_lib import PAD_ID, T_SEQ
from .model import rope_cos_sin

N_CE, N_POS, N_RANK, N_GATE, N_MARG, N_FREQ = 5, 3, 3, 3, 3, 3
N_STRATA = N_CE * N_POS * N_RANK * N_GATE * N_MARG * N_FREQ  # 1215
POS_EDGES = (128, 512)
EPS_PROJ = 1e-30
# exposure-matched natural (replacement-interaction) strata: CE decile x pos bin x rank (1-4) x gate decile x margin
N_DEC = 10
N_COND = N_DEC * N_POS * 4 * N_DEC * N_DEC  # 12000


# ------------------------------------------------------------------------------------------------ data (label-free)
def u_hash(tag, s):
    return int.from_bytes(hashlib.sha256(f"{tag}|{s}".encode()).digest()[:8], "little") / 2**64


def make_batch(rows, split):
    """Label-free batch (no domain fields are ever built): input_ids, targets, doc_ids, positions, gdoc (B, T)."""
    B = len(rows)
    a = {k: np.full((B, T_SEQ), f, dtype=np.int64) for k, f in
         (("input_ids", PAD_ID), ("targets", -100), ("doc_ids", -1), ("positions", 0), ("gdoc", -1))}
    for b, row in enumerate(rows):
        t = 0
        for local, di in enumerate(row):
            ids = split.doc(di)
            L = len(ids)
            a["input_ids"][b, t:t + L] = ids
            a["targets"][b, t:t + L - 1] = ids[1:]
            a["doc_ids"][b, t:t + L] = local
            a["positions"][b, t:t + L] = np.arange(L)
            a["gdoc"][b, t:t + L] = di
            t += L
    return a


def to_dev(a, device):
    return {k: torch.as_tensor(v, device=device) for k, v in a.items()}


# ------------------------------------------------------------------------------------------------ projection
def project(d, g, eps=EPS_PROJ):
    """u = d - (d.g / (|g|^2 + eps)) g along the last dim (float64)."""
    d64, g64 = d.double(), g.double()
    a = (d64 * g64).sum(-1, keepdim=True) / ((g64 * g64).sum(-1, keepdim=True) + eps)
    return d64 - a * g64


def cos(a, b):
    a, b = a.double(), b.double()
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp(min=1e-300)


# ------------------------------------------------------------------------------------------------ strata
class Cuts:
    """Frozen cutpoints (computed on the estimation sample only): ce (N_CE-1,), gate (L, N_GATE-1), marg (L, N_MARG-1),
    freq_class (V,) int64 in 0..2 (target-token frequency tercile, token-weighted)."""

    def __init__(self, ce, gate, marg, freq_class):
        self.ce, self.gate, self.marg, self.freq_class = ce, gate, marg, freq_class

    def state(self):
        return {"ce": self.ce, "gate": self.gate, "marg": self.marg, "freq_class": self.freq_class}

    @classmethod
    def load(cls, s):
        return cls(s["ce"], s["gate"], s["marg"], s["freq_class"])


def pos_bin(pos):
    return (pos >= POS_EDGES[0]).long() + (pos >= POS_EDGES[1]).long()


def ranks_of(p):
    """rank (0-based) of every expert under p (N, E). The first four ranks are taken from p.topk(4) exactly as the model
    routes (identical tie-breaking); ranks >= 4 from a stable descending sort."""
    order = torch.argsort(p, dim=-1, descending=True, stable=True)
    r = torch.empty_like(order)
    r.scatter_(-1, order, torch.arange(p.shape[-1], device=p.device).expand_as(order).contiguous())
    top4 = p.topk(4, -1).indices
    r = torch.where(r < 4, torch.full_like(r, 4), r)  # provisional; overwritten for the true top-4 below
    r.scatter_(-1, top4, torch.arange(4, device=p.device).expand_as(top4).contiguous())
    return r


def strata_all(p, ce, pos, tgt, li, cuts, dev):
    """Forced-insertion strata for every expert at one layer. p (N,E) fp32 natural router probs; ce (N,) natural CE;
    pos, tgt (N,). Returns stratum (N, E) int64 in [0, 1215) or -1 (ineligible: e in top-3 or invalid target)."""
    r = ranks_of(p)
    g4 = p.topk(4, -1).values[:, 3].contiguous()
    cb = torch.bucketize(ce, torch.as_tensor(cuts.ce, device=dev, dtype=ce.dtype))
    pb = pos_bin(pos)
    gb = torch.bucketize(g4, torch.as_tensor(cuts.gate[li], device=dev, dtype=g4.dtype))
    fc = torch.as_tensor(cuts.freq_class, device=dev)[tgt.clamp(min=0)]
    rb = torch.where(r == 3, 0, torch.where(r <= 7, 1, 2))
    lm = torch.log(g4.clamp(min=1e-30)).unsqueeze(1) - torch.log(p.clamp(min=1e-30))
    mb = torch.where(r == 3, 0, torch.bucketize(lm.contiguous(),
                                                  torch.as_tensor(cuts.marg[li], device=dev, dtype=lm.dtype)))
    s = ((((cb.unsqueeze(1) * N_POS + pb.unsqueeze(1)) * N_RANK + rb) * N_GATE + gb.unsqueeze(1)) * N_MARG + mb) \
        * N_FREQ + fc.unsqueeze(1)
    ok = (r >= 3) & (tgt >= 0).unsqueeze(1)
    return torch.where(ok, s, torch.full_like(s, -1))


def cond_strata_all(p, ce, pos, tgt, li, cdec, dev):
    """Exposure-matched natural strata (the slc conditional-interaction definition): tokens with e in the natural top-4;
    CE decile x pos bin x rank (1-4) x gate decile (p_e) x margin decile (p_e - p_(5)). cdec: {"ce": (9,),
    "gate": (L, 9), "marg": (L, 9)}. Returns (N, E) in [0, 12000) or -1."""
    r = ranks_of(p)
    p5 = p.topk(5, -1).values[:, 4]
    cb = torch.bucketize(ce, torch.as_tensor(cdec["ce"], device=dev, dtype=ce.dtype))
    pb = pos_bin(pos)
    gb = torch.bucketize(p.contiguous(), torch.as_tensor(cdec["gate"][li], device=dev, dtype=p.dtype))
    mg = (p - p5.unsqueeze(1)).contiguous()
    mb = torch.bucketize(mg, torch.as_tensor(cdec["marg"][li], device=dev, dtype=p.dtype))
    s = (((cb.unsqueeze(1) * N_POS + pb.unsqueeze(1)) * 4 + r.clamp(max=3)) * N_DEC + gb) * N_DEC + mb
    ok = (r <= 3) & (tgt >= 0).unsqueeze(1)
    return torch.where(ok, s, torch.full_like(s, -1))


def expert_slots(ex, x, topi, gates):
    """Exactly Experts.forward (same dispatch, same per-expert matmuls, same gate multiply) but returning the per-slot
    outputs (N, k, d) before the final sum, so that slots.sum(1) is bit-identical to Experts.forward."""
    N, d = x.shape
    k = topi.shape[1]
    flat = topi.reshape(-1)
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=ex.E).tolist()
    tok = order // k
    xs = x.index_select(0, tok)
    w1s, w2s = ex.w1.to(x.dtype).unbind(0), ex.w2.to(x.dtype).unbind(0)
    parts = torch.split(xs, counts)
    outs = [F.gelu(parts[e] @ w1s[e], approximate="tanh") @ w2s[e] for e, c in enumerate(counts) if c]
    ys = torch.cat(outs, 0) * gates.reshape(-1)[order].to(x.dtype).unsqueeze(-1)
    out = torch.empty_like(ys)
    out[order] = ys
    return out.view(N, k, d)


# ------------------------------------------------------------------------------------------------ probe
class Probe:
    def __init__(self, model, device):
        self.m = model.eval()
        model.requires_grad_(False)
        self.dev = device
        self.cfg = model.cfg
        self.E, self.L = model.cfg.n_experts, model.n_moe
        self.amp = device.type == "cuda"
        self.rd = torch.float64 if model.tok_emb.weight.dtype == torch.float64 else torch.float32

    def autocast(self):
        return torch.autocast(device_type=self.dev.type, dtype=torch.bfloat16, enabled=self.amp)

    def _ctx(self, bt):
        m, cfg = self.m, self.cfg
        x = m.tok_emb(bt["input_ids"])
        cs, sn = rope_cos_sin(bt["positions"], cfg.d_model // cfg.n_head, cfg.rope_base, x.dtype)
        return x, (cs.unsqueeze(1), sn.unsqueeze(1), m.attention_mask(bt["doc_ids"]))

    def _route(self, moe, hf):
        with torch.autocast(device_type=self.dev.type, enabled=False):
            p = torch.softmax(F.linear(hf.to(self.rd), moe.router.weight.to(self.rd)), -1)
            gates, topi = p.topk(self.cfg.top_k, -1)
        return p, gates, topi

    def _run(self, x, ctx, b0, cache=None, probs=None):
        """Blocks b0.. with natural routing. cache: list to fill with per-MoE-layer dicts; probs: list for p."""
        cs, sn, mask = ctx
        B, T, D = x.shape
        N = B * T
        for b in range(b0, len(self.m.blocks)):
            blk = self.m.blocks[b]
            x = x + blk.attn(blk.norm1(x), cs, sn, mask)
            h = blk.norm2(x)
            if not blk.is_moe:
                x = x + blk.ffn(h)
                continue
            moe = blk.ffn
            hf = h.reshape(N, -1)
            p, gates, topi = self._route(moe, hf)
            slots = expert_slots(moe.experts, hf, topi, gates)
            sh = moe.shared(hf)
            if cache is not None:
                cache.append({"x_mid": x, "hf": hf, "gates": gates, "topi": topi, "slots": slots, "sh": sh})
            if probs is not None:
                probs.append(p)
            x = x + (slots.sum(1) + sh).view(B, T, -1)
        return self.m.norm_f(x)

    def ce_of(self, hidden, tgt):
        hf, tf = hidden.reshape(-1, hidden.shape[-1]), tgt.reshape(-1)
        out = torch.zeros(tf.shape, device=hidden.device, dtype=torch.float32 if self.rd == torch.float32 else self.rd)
        for s in range(0, tf.numel(), 8192):
            out[s:s + 8192] = F.cross_entropy(self.m.lm_logits(hf[s:s + 8192]).to(out.dtype), tf[s:s + 8192],
                                              ignore_index=-100, reduction="none")
        return out

    @torch.no_grad()
    def natural(self, bt, keep_cache=False):
        """Returns ce (N,), probs list (L) of (N, E) fp32, cache (list per MoE layer) or None, ctx."""
        with self.autocast():
            x, ctx = self._ctx(bt)
            cache, probs = ([] if keep_cache else None), []
            hidden = self._run(x, ctx, 0, cache, probs)
            ce = self.ce_of(hidden, bt["targets"])
        return ce, probs, cache, ctx

    def _expert_out(self, hf, w1e, w2e):
        """The single probed expert is evaluated in float64 with autocast disabled, so that a small virtual update is
        not lost to bf16 / TF32 rounding inside the expert (the gradient path is identical up to precision)."""
        with torch.autocast(device_type=self.dev.type, enabled=False):
            return F.gelu(hf.double() @ w1e.double(), approximate="tanh") @ w2e.double()

    def _from(self, li, y, c, ctx, shape):
        B, T, D = shape
        x = c["x_mid"] + y.view(B, T, -1)
        return self._run(x, ctx, li + 2)  # MoE layer li is block li + 1; an empty range returns norm_f(x)

    def forced_hidden(self, li, e, c, ctx, shape, w1e, w2e):
        """Hidden states with expert e forced into slot 4 at MoE layer li (block li + 1) using theta_e = (w1e, w2e)."""
        top3 = (c["topi"][:, :3] == e).any(1)
        g4 = c["gates"][:, 3]
        sl = c["slots"]
        ye = self._expert_out(c["hf"], w1e, w2e) * g4.unsqueeze(1).double()
        s4 = torch.where(top3.unsqueeze(1), sl[:, 3], ye.to(sl.dtype))
        y = torch.cat([sl[:, :3], s4.unsqueeze(1)], 1).sum(1) + c["sh"]
        return self._from(li, y, c, ctx, shape)

    def natural_updated_hidden(self, li, e, c, ctx, shape, w1e, w2e, w1n, w2n):
        """Natural routing at layer li with theta_e replaced by (w1n, w2n): y = y_nat + g_e (E'(h) - E(h)) on positions
        where e is in the top-4 (exactly y_natural when (w1n, w2n) == (w1e, w2e))."""
        sel = c["topi"] == e
        ge = (c["gates"] * sel).sum(1)
        on = sel.any(1)
        dlt = (self._expert_out(c["hf"], w1n, w2n) - self._expert_out(c["hf"], w1e, w2e)) * ge.unsqueeze(1).double()
        y = c["slots"].sum(1) + c["sh"]
        y = torch.where(on.unsqueeze(1), y + dlt.to(y.dtype), y)
        return self._from(li, y, c, ctx, shape)

    def dce_dhidden(self, hidden, tgt):
        """c_t = dCE_t / dhidden_t for every position (zero for ignored targets)."""
        hd = hidden.detach().requires_grad_(True)
        hf, tf = hd.reshape(-1, hd.shape[-1]), tgt.reshape(-1)
        with self.autocast():
            for s in range(0, tf.numel(), 8192):
                F.cross_entropy(self.m.lm_logits(hf[s:s + 8192]).float(), tf[s:s + 8192], ignore_index=-100,
                                reduction="sum").backward()
        return hd.grad

    def expert_grads(self, li, e, c, ctx, shape, tgt, W):
        """W (J, N) token-weight vectors. Returns (J, d*f + f*d) exact gradients of sum_t W_j,t CE_t w.r.t. theta_e under
        forced insertion of e at layer li (fp32)."""
        moe = self.m.blocks[li + 1].ffn
        w1e = moe.experts.w1[e].detach().clone().requires_grad_(True)
        w2e = moe.experts.w2[e].detach().clone().requires_grad_(True)
        with torch.enable_grad():
            with self.autocast():
                hidden = self.forced_hidden(li, e, c, ctx, shape, w1e, w2e)
            cvec = self.dce_dhidden(hidden, tgt)
            hflat = hidden.reshape(-1, hidden.shape[-1])
            out = []
            J = W.shape[0]
            for j in range(J):
                go = (W[j].unsqueeze(1) * cvec.reshape(-1, cvec.shape[-1])).to(hflat.dtype)
                g1, g2 = torch.autograd.grad(hflat, (w1e, w2e), grad_outputs=go, retain_graph=j < J - 1,
                                             allow_unused=True)
                g1 = torch.zeros_like(w1e) if g1 is None else g1
                g2 = torch.zeros_like(w2e) if g2 is None else g2
                out.append(torch.cat([g1.reshape(-1), g2.reshape(-1)]).float())
        return torch.stack(out)


def theta_e(model, li, e):
    ex = model.blocks[li + 1].ffn.experts
    return ex.w1[e], ex.w2[e]


def split_theta(vec, d, f):
    return vec[:d * f].view(d, f), vec[d * f:].view(f, d)


def state_hash(model):
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ weights
def token_q(q_of_doc, gdoc):
    """(N, K) per-token soft membership (zeros for padding)."""
    g = gdoc.reshape(-1)
    q = torch.as_tensor(q_of_doc, device=gdoc.device)[g.clamp(min=0)]
    return torch.where((g >= 0).unsqueeze(1), q, torch.zeros_like(q))


def pair_weights(qtok, stratum, kstar, ratio, ptot, mk):
    """For one (l, e) and table: qtok (N, K); stratum (N,) in [0,1215) or -1; kstar int; ratio (1215,) P/C (0 where
    unmatched); ptot = sum of matched positive weight; mk (K,) eligible mass per cluster. Returns w_plus, w_minus, w_glob
    (N,) float64."""
    ok = stratum >= 0
    s = stratum.clamp(min=0)
    qk = qtok[:, kstar].double()
    matched = ok & (ratio[s] > 0)
    wp = torch.where(matched, qk, torch.zeros_like(qk)) / ptot
    wm = torch.where(matched, (1.0 - qk) * ratio[s], torch.zeros_like(qk)) / ptot
    K = qtok.shape[1]
    wg = torch.where(ok, (qtok.double() / (K * mk.clamp(min=1e-300)).unsqueeze(0)).sum(1), torch.zeros_like(qk))
    return wp, wm, wg


# ------------------------------------------------------------------------------------------------ bootstrap
def paper_boot_weights(n_papers, n_boot, seed):
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_papers, np.ones(n_papers) / n_papers, size=n_boot).astype(np.float64)


def contrast_boot(pos_s, pos_w, ctl_s, ctl_w, W):
    """pos_s etc. (P_pairs, n_papers). Returns point (P,) and boot (B, P) of mean_pos - mean_ctl (NaN if a side empty)."""
    def ratio(s, w, Wt):
        num, den = Wt @ s.T, Wt @ w.T
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    one = np.ones((1, pos_s.shape[1]))
    pt = ratio(pos_s, pos_w, one)[0] - ratio(ctl_s, ctl_w, one)[0]
    bt = ratio(pos_s, pos_w, W) - ratio(ctl_s, ctl_w, W)
    return pt, bt
