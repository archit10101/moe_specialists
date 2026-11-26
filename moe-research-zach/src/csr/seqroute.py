"""csr-sequence-routing-search-v1: genuinely sequence-routed MoE (docs/SEQUENCE_ROUTING_SEARCH_V1_PREREG.md).

New module; CSRModel and every earlier routing mode are unchanged. SeqRouteModel subclasses CSRModel, so the backbone
(embedding, attention, dense layer 0, experts, shared experts, router matrices W_l) is constructed and initialized
exactly as in TOKEN-4 for the same seed. Extra sequence-router parameters are created afterwards and initialized from a
dedicated generator, so they never change the backbone initialization.

Routing modes
  token    : TOKEN-4. Per token: p = softmax(W_l h_t), top-4 by p, gate = p (global, no renorm), dropless, + shared.
  sequence : per MoE layer l and document, contiguous segments s_0, s_1, ... (never crossing documents).
             route(l, 0) = TopK(softmax(bos_l / T))                       (learned, content-free)
             route(l, j) = TopK(softmax(W_l summary(l, j-1) [+ O_l g_{j-1}] / T))   for j >= 1
             The 4 expert ids AND the 4 gate values are computed once per segment and broadcast to every token of
             the segment. W_l is the layer's existing router matrix (applied to a pooled summary, never to a token).
Summaries of the completed previous segment j-1 (h = post-RMSNorm router input of layer l):
  mean | last | attn (learned query, softmax pooling inside the segment);
  state: none | ema (e_j = lam e_{j-1} + (1-lam) pool_j, route from e) | gru (GRUCell over [W pool_j, multi-hot
  route_j] -> g_j; logits = W pool_j + O g_j, O zero-initialized).
Segmentations (all causal; decided after observing token t, new route from token t+1; never revised):
  fixed : starts at positions 0, o, o+L, o+2L, ...   (o = opening length, default o = L)
  cp    : change point. score_t = 1 - cos(mean src[t-7..t], mean src[t-39..t-8]) (windows clipped to the document,
          valid iff the long window holds >= 8 tokens). Boundary after t iff seg_len >= 256, or seg_len >= 16 and
          score_t > tau. src = h_l (per layer) or the residual stream after the dense layer 0 (shared by all layers).
  lbd   : learned boundary detector u_t = w_l . sg(h_t) + b_l, boundary iff (seg_len >= 16 and u_t > tau) or
          seg_len >= 256. Trained by regression on a future-window change label (label only; decisions use h_t).
  opening o > 0 forces the first boundary of every document after token o-1 (fixed and dynamic).
tau (cp/lbd) is a buffer: initialized by a no-grad pre-pass on the first training micro-batch and adapted between
optimizer steps (log tau += tau_lr * log(mean closed segment length / target)) from previous steps only; it is
frozen in eval/inference. It is part of the checkpoint.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import CSRModel, ForwardOut, ModelConfig, rope_cos_sin

EXPERIMENT = "csr-sequence-routing-search-v1"
SEGS = ("fixed", "cp", "lbd")
SUMMARIES = ("mean", "last", "attn")
STATES = ("none", "ema", "gru")
EXTRA_SEED_OFFSET = 1_000_003


@dataclass
class SRRouting:
    mode: str = "sequence"  # token | sequence
    seg: str = "fixed"  # fixed | cp | lbd
    seg_len: int = 64  # fixed
    opening: int = 0  # 0: none; else forced first-segment length
    target_len: int = 64  # cp / lbd controller target (mean length of closed segments)
    min_len: int = 16
    max_len: int = 256
    boundary_source: str = "layer"  # layer | layer0 (cp only)
    cp_short: int = 8
    cp_long: int = 32
    cp_min_long: int = 8
    lbd_horizon: int = 16
    summary: str = "mean"
    state: str = "none"
    ema_decay: float = 0.5
    gru_hidden: int = 32
    temperature: float = 1.0
    lb_weighting: str = "token"  # token | segment
    tau_lr: float = 0.2
    bd_coeff: float = 1.0

    def validate(self):
        assert self.mode in ("token", "sequence"), self.mode
        if self.mode == "token":
            return self
        assert self.seg in SEGS and self.summary in SUMMARIES and self.state in STATES
        assert self.boundary_source in ("layer", "layer0")
        assert self.boundary_source == "layer" or self.seg == "cp"
        assert self.min_len == 16 and self.max_len == 256, "dynamic limits are fixed by the protocol"
        if self.seg == "fixed":
            assert self.seg_len in (16, 32, 64, 128, 256), self.seg_len
        assert self.opening == 0 or 16 <= self.opening <= 256
        assert self.temperature > 0 and 0 <= self.ema_decay < 1 and self.lb_weighting in ("token", "segment")
        return self

    @property
    def dynamic(self):
        return self.mode == "sequence" and self.seg in ("cp", "lbd")

    @property
    def family(self):
        return "token" if self.mode == "token" else ("dynamic" if self.dynamic else "fixed")

    @classmethod
    def from_cfg(cls, r: dict):
        keys = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in r.items() if k in keys}).validate()

    def to_dict(self):
        return asdict(self)


# ------------------------------------------------------------------ segment index (parallel forward)
@dataclass
class SegIndex:
    gid: torch.Tensor  # (N,) segment id per flattened token, -1 for pad
    S: int
    first: torch.Tensor  # (S,) flat index of first token
    count: torch.Tensor  # (S,) tokens
    j: torch.Tensor  # (S,) index of the segment inside its document
    doc: torch.Tensor  # (S,) document index (batch-local, row-major order)
    D: int
    doc_first_seg: torch.Tensor  # (D,)
    Jmax: int
    closed: torch.Tensor  # (S,) bool: ended by a boundary decision (not by the end of the document)


def seg_index(starts, positions, nonpad):
    """starts (B, T) bool, True at the first token of every segment (must include every document start)."""
    B, T = starts.shape
    st = (starts & nonpad).reshape(-1)
    npf = nonpad.reshape(-1)
    pos = positions.reshape(-1)
    assert bool((st | ~(npf & (pos == 0))).all()), "every document start must start a segment"
    gid = torch.cumsum(st.long(), 0) - 1
    gid = torch.where(npf, gid, torch.full_like(gid, -1))
    first = st.nonzero().squeeze(1)
    S = int(first.numel())
    count = torch.bincount(gid[npf], minlength=S)
    is_doc_first = pos[first] == 0
    doc = torch.cumsum(is_doc_first.long(), 0) - 1
    doc_first_seg = is_doc_first.nonzero().squeeze(1)
    j = torch.arange(S, device=starts.device) - doc_first_seg[doc]
    nxt_doc_first = torch.ones(S, dtype=torch.bool, device=starts.device)
    if S > 1:
        nxt_doc_first[:-1] = is_doc_first[1:]
    closed = ~nxt_doc_first  # next segment continues the same document
    D = int(doc_first_seg.numel())
    Jmax = int(j.max().item()) + 1 if S else 0
    return SegIndex(gid, S, first, count, j, doc, D, doc_first_seg, Jmax, closed)


def fixed_starts_mask(positions, nonpad, L, opening=0):
    p = positions
    if opening and opening != L:
        s = (p == 0) | ((p >= opening) & ((p - opening) % L == 0))
    else:
        s = (p % L) == 0
    return s & nonpad


def window_change_score(src, positions, nonpad, short, long, min_long):
    """score_t = 1 - cos(mean src[t-short+1 .. t], mean src[t-short-long+1 .. t-short]); windows clipped at the
    document start; valid iff >= min_long tokens in the long window. src (B, T, d) -> (score (B,T) f64, valid)."""
    B, T, d = src.shape
    cs = F.pad(torch.cumsum(src.double(), dim=1), (0, 0, 1, 0))  # cs[:, i] = sum src[:, :i]
    t = torch.arange(T, device=src.device).unsqueeze(0).expand(B, T)
    a = (t - positions).clamp(min=0)  # document start
    lo_s = torch.maximum(t - short + 1, a)
    lo_l = torch.maximum(t - short - long + 1, a)
    hi_l = lo_s

    def g(idx):
        return cs.gather(1, idx.unsqueeze(-1).expand(B, T, d))

    s_sum = g(t + 1) - g(lo_s)
    l_sum = g(hi_l) - g(lo_l)
    n_s = (t + 1 - lo_s).double().unsqueeze(-1)
    n_l = (hi_l - lo_l)
    ms = s_sum / n_s
    ml = l_sum / n_l.clamp(min=1).double().unsqueeze(-1)
    cos = (ms * ml).sum(-1) / (ms.norm(dim=-1) * ml.norm(dim=-1)).clamp(min=1e-12)
    valid = nonpad & (n_l >= min_long)
    return 1.0 - cos, valid


def future_change_target(src, positions, doc_ids, nonpad, H):
    """LBD training label (never used for routing): 1 - cos(mean src[t+1..t+H], mean src[t-H+1..t]); valid iff both
    full windows lie inside the document."""
    B, T, d = src.shape
    cs = F.pad(torch.cumsum(src.double(), dim=1), (0, 0, 1, 0))
    t = torch.arange(T, device=src.device).unsqueeze(0).expand(B, T)
    di = doc_ids.clamp(min=0)
    lens = torch.zeros(B, T, dtype=torch.long, device=src.device).scatter_add_(1, di, nonpad.long())
    rem = lens.gather(1, di) - 1 - positions  # tokens after t inside the document
    valid = nonpad & (positions >= H - 1) & (rem >= H)
    ti = t.clamp(max=T - 1 - H).clamp(min=H - 1)

    def g(idx):
        return cs.gather(1, idx.unsqueeze(-1).expand(B, T, d))

    fut = (g(ti + 1 + H) - g(ti + 1)) / H
    past = (g(ti + 1) - g(ti + 1 - H)) / H
    cos = (fut * past).sum(-1) / (fut.norm(dim=-1) * past.norm(dim=-1)).clamp(min=1e-12)
    return (1.0 - cos), valid


def scan_dynamic_starts(cand, positions, nonpad, min_len, max_len, opening=0):
    """Causal boundary rule, exactly as applied token by token: after token t of a segment that began at s,
    boundary iff (opening phase: j == 0 and t - s + 1 == opening) or (otherwise: t - s + 1 == max_len, or
    t - s + 1 >= min_len and cand[t]). cand/positions/nonpad: numpy (B, T). Returns starts (B, T) bool."""
    B, T = cand.shape
    starts = np.zeros((B, T), dtype=bool)
    p = positions.ravel()
    n = nonpad.ravel()
    flat_c = np.flatnonzero(cand.ravel() & n)
    doc_a = np.flatnonzero((p == 0) & n)
    nxt_ok = np.zeros_like(n)
    nxt_ok[:-1] = n[1:] & (p[1:] == p[:-1] + 1)
    col = np.arange(B * T) % T
    is_last = n & (~nxt_ok | (col == T - 1))
    doc_L = p[np.flatnonzero(is_last)] + 1
    st = starts.ravel()
    for a, L in zip(doc_a.tolist(), doc_L.tolist()):
        end = a + L
        s, j = a, 0
        st[a] = True
        while True:
            if opening and j == 0:
                ns = a + opening
            else:
                k = np.searchsorted(flat_c, s + min_len - 1)
                if k < flat_c.size and flat_c[k] < end and flat_c[k] + 1 - s <= max_len:
                    ns = int(flat_c[k]) + 1
                else:
                    ns = s + max_len
            if ns >= end:
                break
            st[ns] = True
            s, j = ns, j + 1
    return starts


# ------------------------------------------------------------------ model
class SeqRouteModel(CSRModel):
    def __init__(self, cfg: ModelConfig, sr: SRRouting, extra_seed: int = 0):
        super().__init__(cfg)  # backbone: identical construction + init to TOKEN-4 under the same global seed
        self.sr = sr.validate()
        n_moe, E, d = len(cfg.moe_layers), cfg.n_experts, cfg.d_model
        self.n_moe = n_moe
        if sr.mode == "sequence":
            gen = torch.Generator().manual_seed(EXTRA_SEED_OFFSET + int(extra_seed))
            self.bos = nn.Parameter(torch.randn(n_moe, E, generator=gen) * cfg.init_std)
            if sr.summary == "attn":
                self.attn_q = nn.Parameter(torch.zeros(n_moe, d))
            if sr.state == "gru":
                H = sr.gru_hidden
                self.gru = nn.ModuleList([nn.GRUCell(2 * E, H) for _ in range(n_moe)])
                self.gru_out = nn.ModuleList([nn.Linear(H, E) for _ in range(n_moe)])
                with torch.no_grad():
                    for cell in self.gru:
                        for p_ in cell.parameters():
                            p_.copy_((torch.rand(p_.shape, generator=gen) * 2 - 1) / math.sqrt(H))
                    for o in self.gru_out:
                        o.weight.zero_()
                        o.bias.zero_()
            if sr.seg == "lbd":
                self.bd_w = nn.Parameter(torch.randn(n_moe, d, generator=gen) * cfg.init_std)
                self.bd_b = nn.Parameter(torch.zeros(n_moe))
            if sr.dynamic:
                n_src = 1 if sr.boundary_source == "layer0" else n_moe
                self.register_buffer("tau", torch.full((n_src,), float("nan"), dtype=torch.float64))
        self._tau_stats = None
        self._calibrating = False

    # -------------------------------------------------------------- parameter accounting
    def extra_parameter_names(self):
        base = {"bos", "attn_q", "bd_w", "bd_b"}
        return sorted(n for n, _ in self.named_parameters() if n.split(".")[0] in base or n.startswith("gru"))

    def extra_parameter_count(self):
        names = set(self.extra_parameter_names())
        return sum(p.numel() for n, p in self.named_parameters() if n in names)

    def backbone_state(self):
        names = set(self.extra_parameter_names())
        return {k: v for k, v in self.state_dict().items() if k not in names and k != "tau"}

    # -------------------------------------------------------------- tau controller
    def _src_index(self, li):
        return 0 if self.sr.boundary_source == "layer0" else li

    def tau_stats_reset(self):
        n = self.tau.numel() if hasattr(self, "tau") else 0
        self._tau_stats = [[0, 0] for _ in range(n)]  # closed segments, closed tokens

    @torch.no_grad()
    def tau_update(self):
        """Between optimizer steps (training only): uses statistics of already-completed micro-batches."""
        if not self.sr.dynamic or self._tau_stats is None:
            return None
        out = []
        for i, (ns, nt) in enumerate(self._tau_stats):
            if ns > 0:
                m = nt / ns
                step = self.sr.tau_lr * math.log(self.sr.target_len / m)
                step = max(-0.5, min(0.5, step))
                # m > target (segments too long, too few boundaries) -> step < 0 -> lower threshold
                if self.sr.seg == "cp":
                    self.tau[i] = float(self.tau[i]) * math.exp(step)  # cp scores are positive: multiplicative
                else:
                    self.tau[i] = float(self.tau[i]) + step  # lbd logits: additive
                out.append(m)
            else:
                out.append(None)
        self.tau_stats_reset()
        return out

    @torch.no_grad()
    def tau_calibrate(self, batch):
        """No-grad pre-pass on one training micro-batch before step 0: sets every NaN tau (layer by layer) to the
        (1 - 1/target) quantile of that source's valid scores. The training forward always uses a fixed tau."""
        self._calibrating = True
        was = self.training
        self.eval()
        try:
            self.forward(batch)
        finally:
            self._calibrating = False
            self.train(was)
        return self.tau.tolist()

    # -------------------------------------------------------------- boundary decisions
    def _cand(self, li, src, positions, nonpad, src_index):
        sr = self.sr
        if sr.seg == "cp":
            score, valid = window_change_score(src, positions, nonpad, sr.cp_short, sr.cp_long, sr.cp_min_long)
        else:
            score = (src.detach().double() @ self.bd_w[li].double() + self.bd_b[li].double())
            valid = nonpad
        if self._calibrating and torch.isnan(self.tau[src_index]):
            v = score[valid].float()
            q = 1.0 - 1.0 / sr.target_len
            self.tau[src_index] = float(torch.quantile(v, q)) if v.numel() else 0.0
        tau = self.tau[src_index]
        if torch.isnan(tau):
            raise RuntimeError("tau not initialized (run tau_calibrate on a training micro-batch first)")
        return (score > tau) & valid, score

    def _dynamic_starts(self, cand, positions, nonpad):
        sr = self.sr
        st = scan_dynamic_starts(cand.cpu().numpy(), positions.cpu().numpy(), nonpad.cpu().numpy(), sr.min_len,
                                 sr.max_len, sr.opening)
        return torch.as_tensor(st, device=cand.device)

    # -------------------------------------------------------------- summaries and routes
    def _pool(self, li, hf, si: SegIndex, npm):
        sr = self.sr
        S, d = si.S, hf.shape[1]
        g = si.gid[npm]
        hv = hf[npm]
        if sr.summary == "mean":
            s = hf.new_zeros(S, d).index_add_(0, g, hv)
            return s / si.count.clamp(min=1).unsqueeze(1).to(hf.dtype)
        if sr.summary == "last":
            return hf[si.first + si.count - 1]
        sc = (hv @ self.attn_q[li].to(hv.dtype)) / math.sqrt(d)
        mx = torch.full((S,), float("-inf"), dtype=sc.dtype, device=sc.device).scatter_reduce(0, g, sc, "amax")
        e = torch.exp(sc - mx[g])
        den = sc.new_zeros(S).index_add_(0, g, e)
        num = hf.new_zeros(S, d).index_add_(0, g, hv * e.unsqueeze(1))
        return num / den.unsqueeze(1)

    def _route_logits(self, li, moe, pooled, si: SegIndex):
        """(S, E) fp32 routing logits: BOS for j == 0, else from the completed previous segment (and state)."""
        sr = self.sr
        E = moe.E
        bos = self.bos[li].to(torch.float32 if pooled.dtype != torch.float64 else pooled.dtype)
        if sr.state == "none":
            zp = moe.router_logits(pooled)  # W_l applied to segment summaries only
            prev = (torch.arange(si.S, device=pooled.device) - 1).clamp(min=0)
            return torch.where((si.j == 0).unsqueeze(1), bos.unsqueeze(0).expand(si.S, E), zp[prev])
        # recurrent states across completed segments of the same document: loop over the segment index
        zp_all = moe.router_logits(pooled)
        tab = torch.full((si.D, si.Jmax), -1, dtype=torch.long, device=pooled.device)
        tab[si.doc, si.j] = torch.arange(si.S, device=pooled.device)
        z = [None] * si.Jmax
        st = None
        for jj in range(si.Jmax):
            rows = tab[:, jj]
            ok = rows >= 0
            if jj == 0:
                zj = bos.unsqueeze(0).expand(si.D, E)
            else:
                zj = st["z_next"]
            z[jj] = zj
            if jj == si.Jmax - 1:
                break
            r = rows.clamp(min=0)
            if sr.state == "ema":
                pj = pooled[r]
                e = pj if jj == 0 else sr.ema_decay * st["e"] + (1 - sr.ema_decay) * pj
                e = torch.where(ok.unsqueeze(1), e, st["e"] if st is not None else e)
                st = {"e": e, "z_next": moe.router_logits(e)}
            else:  # gru over [W pool_j, multi-hot route_j]
                topi = torch.softmax(zj / sr.temperature, -1).topk(moe.k, dim=-1).indices
                mh = torch.zeros(si.D, E, dtype=zp_all.dtype, device=zp_all.device).scatter_(1, topi, 1.0)
                u = torch.cat([zp_all[r], mh], dim=1)
                h0 = st["g"] if st is not None else u.new_zeros(si.D, sr.gru_hidden)
                gnew = self.gru[li](u, h0)
                gnew = torch.where(ok.unsqueeze(1), gnew, h0)
                st = {"g": gnew, "z_next": zp_all[r] + self.gru_out[li](gnew)}
        zt = torch.stack(z, 1)  # (D, Jmax, E)
        return zt[si.doc, si.j]

    def _seq_moe(self, li, moe, h, positions, nonpad, starts, rec):
        sr = self.sr
        B, T, d = h.shape
        N = B * T
        hf = h.reshape(N, d)
        npm = nonpad.reshape(-1)
        si = seg_index(starts, positions, nonpad)
        rd = torch.float32 if hf.dtype in (torch.float16, torch.bfloat16) else hf.dtype
        with torch.autocast(device_type=h.device.type, enabled=False):
            pooled = self._pool(li, hf.to(rd), si, npm)
            z_seg = self._route_logits(li, moe, pooled, si)  # (S, E)
            z_all = torch.cat([z_seg, self.bos[li].to(z_seg.dtype).unsqueeze(0)], 0)  # row S: pad tokens
            probs = torch.softmax(z_all / sr.temperature, dim=-1)
            gates_s, topi_s = probs.topk(moe.k, dim=-1)
        g_tok = torch.where(si.gid >= 0, si.gid, torch.full_like(si.gid, si.S))
        topi = topi_s[g_tok]
        gates = gates_s[g_tok]
        y = moe.experts(hf, topi, gates)
        if moe.shared is not None:
            y = y + moe.shared(hf)
        # load-balance / z-loss on the routing actually used (token- or segment-weighted)
        w = si.count.to(probs.dtype) if sr.lb_weighting == "token" else torch.ones(si.S, dtype=probs.dtype,
                                                                                   device=probs.device)
        n = w.sum().clamp(min=1)
        P = (probs[:si.S] * w.unsqueeze(1)).sum(0) / n
        f = torch.zeros(moe.E, dtype=probs.dtype, device=probs.device).index_add_(
            0, topi_s[:si.S].reshape(-1), w.repeat_interleave(moe.k)) / (n * moe.k)
        aux = moe.E * (f * P).sum()
        zl = (torch.logsumexp(z_seg / sr.temperature, dim=-1).pow(2) * w).sum() / n
        if rec is not None:
            rec.update({"topi": topi.view(B, T, -1), "gates": gates.view(B, T, -1), "seg_gid": si.gid.view(B, T),
                        "starts": starts, "seg_topi": topi_s[:si.S], "seg_gates": gates_s[:si.S], "seg_j": si.j,
                        "seg_count": si.count, "seg_closed": si.closed, "seg_first": si.first, "S": si.S,
                        "probs_seg": probs[:si.S], "z_seg": z_seg})
        if self.training and self._tau_stats is not None and self.sr.dynamic and self.sr.boundary_source == "layer":
            c = si.closed
            self._tau_stats[li][0] += int(c.sum())
            self._tau_stats[li][1] += int(si.count[c].sum())
        return y.view(B, T, d), aux, zl

    def _token_moe(self, moe, h, nonpad, rec):
        B, T, d = h.shape
        hf = h.reshape(B * T, d)
        z = moe.router_logits(hf)
        probs = torch.softmax(z, dim=-1)
        gates, topi = probs.topk(moe.k, dim=-1)
        y = moe.experts(hf, topi, gates)
        if moe.shared is not None:
            y = y + moe.shared(hf)
        npm = nonpad.reshape(-1)
        n = npm.sum().clamp(min=1)
        pr = probs[npm]
        counts = torch.bincount(topi[npm].reshape(-1), minlength=moe.E).to(pr.dtype)
        aux = moe.E * ((counts / (n * moe.k)) * (pr.sum(0) / n)).sum()
        zl = torch.logsumexp(z[npm], dim=-1).pow(2).sum() / n
        if rec is not None:
            rec.update({"topi": topi.view(B, T, -1), "gates": gates.view(B, T, -1),
                        "probs_tok": probs.view(B, T, -1)})
        return y.view(B, T, d), aux, zl

    def forward(self, batch, collect: bool = False):
        sr = self.sr
        ids, doc_ids, positions = batch["input_ids"], batch["doc_ids"], batch["positions"]
        B, T = ids.shape
        x = self.tok_emb(ids)
        cos, sin = rope_cos_sin(positions, self.cfg.d_model // self.cfg.n_head, self.cfg.rope_base, x.dtype)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        mask = self.attention_mask(doc_ids, positions)
        nonpad = doc_ids >= 0
        acc = torch.float64 if x.dtype == torch.float64 else torch.float32
        aux_total = x.new_zeros((), dtype=acc)
        z_total = x.new_zeros((), dtype=acc)
        bd_total = x.new_zeros((), dtype=acc)
        records = []
        shared_starts = None
        if sr.mode == "sequence" and sr.seg == "fixed":
            shared_starts = fixed_starts_mask(positions, nonpad, sr.seg_len, sr.opening)
        moe_i = 0
        for bi, blk in enumerate(self.blocks):
            x = x + blk.attn(blk.norm1(x), cos, sin, mask)
            h = blk.norm2(x)
            if not blk.is_moe:
                x = x + blk.ffn(h)
                if sr.dynamic and sr.boundary_source == "layer0" and bi == 0:
                    with torch.autocast(device_type=x.device.type, enabled=False):
                        cand, _ = self._cand(None, x.detach().double() if x.dtype == torch.float64 else
                                             x.detach().float(), positions, nonpad, 0)
                    shared_starts = self._dynamic_starts(cand, positions, nonpad)
                    if self.training and self._tau_stats is not None:
                        si0 = seg_index(shared_starts, positions, nonpad)
                        self._tau_stats[0][0] += int(si0.closed.sum())
                        self._tau_stats[0][1] += int(si0.count[si0.closed].sum())
                continue
            moe = blk.ffn
            rec = {} if collect else None
            if sr.mode == "token":
                y, aux, zl = self._token_moe(moe, h, nonpad, rec)
            else:
                if sr.dynamic and sr.boundary_source == "layer":
                    with torch.autocast(device_type=h.device.type, enabled=False):
                        hs = h.detach().float() if h.dtype != torch.float64 else h.detach()
                        cand, score = self._cand(moe_i, hs, positions, nonpad, moe_i)
                        if sr.seg == "lbd" and self.training and not self._calibrating:
                            tgt, tv = future_change_target(hs, positions, doc_ids, nonpad, sr.lbd_horizon)
                            if bool(tv.any()):
                                a = tgt[tv]
                                a = ((a - a.mean()) / a.std().clamp(min=1e-6)).detach()
                                u = (hs.double() @ self.bd_w[moe_i].double() + self.bd_b[moe_i].double())[tv]
                                bd_total = bd_total + ((u - a) ** 2).mean().to(acc)
                    starts = self._dynamic_starts(cand, positions, nonpad)
                else:
                    starts = shared_starts
                y, aux, zl = self._seq_moe(moe_i, moe, h, positions, nonpad, starts, rec)
            x = x + y
            aux_total = aux_total + aux.to(acc)
            z_total = z_total + zl.to(acc)
            if collect:
                rec["h"] = h
                records.append(rec)
            moe_i += 1
        logits = self.lm_head(self.norm_f(x))
        out = ForwardOut(logits, aux_total, z_total, records, None)
        out.bd_loss = bd_total
        return out

    # ------------------------------------------------------------------ incremental decoding (batched lockstep)
    def new_sr_state(self, B):
        return {"kv": [dict() for _ in self.blocks], "moe": [dict() for _ in range(self.n_moe)], "B": B,
                "shared": {}, "router_calls": 0, "boundary_events": 0}

    def _src_window_score(self, hist, B):
        """hist: list of (B, d) source vectors of the current document, oldest first, including the current one."""
        sr = self.sr
        n = len(hist)
        ns = min(sr.cp_short, n)
        nl = min(sr.cp_long, n - ns)
        if nl < sr.cp_min_long:
            return None
        hs = torch.stack(hist[n - ns:], 0).double().sum(0) / ns
        hl = torch.stack(hist[n - ns - nl:n - ns], 0).double().sum(0) / nl
        cos = (hs * hl).sum(-1) / (hs.norm(dim=-1) * hl.norm(dim=-1)).clamp(min=1e-12)
        return 1.0 - cos

    def _boundary_now(self, st, cand):
        """Decision after the current token (st holds seg_len after counting it, and j)."""
        sr = self.sr
        sl, j = st["seg_len"], st["j"]
        if sr.seg == "fixed":
            if sr.opening and sr.opening != sr.seg_len:
                return torch.where(j == 0, sl == sr.opening, sl == sr.seg_len)
            return sl == sr.seg_len
        opening_phase = (j == 0) & (sr.opening > 0)
        dyn = (sl >= sr.max_len) | ((sl >= sr.min_len) & cand)
        return torch.where(opening_phase, sl == sr.opening, dyn)

    def _route_from_logits(self, z, k):
        probs = torch.softmax(z / self.sr.temperature, -1)
        gates, topi = probs.topk(k, dim=-1)
        return topi, gates

    @torch.no_grad()
    def sr_step(self, token_ids, position: int, state):
        """One token for each of B documents in lockstep (all at the same in-document position). Returns logits
        (B, V) and per-MoE-layer topi (B, k). Uses only the current and previous tokens."""
        sr = self.sr
        dev = self.tok_emb.weight.device
        B = token_ids.shape[0]
        x = self.tok_emb(token_ids.view(B, 1).to(dev))
        pos = torch.full((B, 1), position, device=dev)
        cos, sin = rope_cos_sin(pos, self.cfg.d_model // self.cfg.n_head, self.cfg.rope_base, x.dtype)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        out = []
        moe_i = 0
        sh = state["shared"]
        for bi, blk in enumerate(self.blocks):
            x = x + blk.attn(blk.norm1(x), cos, sin, None, kv_cache=state["kv"][bi])
            h = blk.norm2(x)
            if not blk.is_moe:
                x = x + blk.ffn(h)
                if sr.dynamic and sr.boundary_source == "layer0" and bi == 0:
                    if position == 0:
                        sh.clear()
                        sh["hist"] = []
                    sh["hist"].append(x.view(B, -1).double() if x.dtype == torch.float64 else x.view(B, -1).float())
                    if len(sh["hist"]) > sr.cp_short + sr.cp_long:
                        sh["hist"].pop(0)
                    sc = self._src_window_score(sh["hist"], B)
                    sh["cand"] = (sc > self.tau[0]) if sc is not None else torch.zeros(B, dtype=torch.bool, device=dev)
                continue
            moe = blk.ffn
            hv = h.view(B, -1)
            if sr.mode == "token":
                z = moe.router_logits(hv)
                state["router_calls"] += 1
                probs = torch.softmax(z, -1)
                gates, topi = probs.topk(moe.k, dim=-1)
                y = moe.experts(hv, topi, gates)
                if moe.shared is not None:
                    y = y + moe.shared(hv)
                x = x + y.view(B, 1, -1)
                out.append(topi)
                moe_i += 1
                continue
            st = state["moe"][moe_i]
            rd = torch.float64 if hv.dtype == torch.float64 else torch.float32
            E = moe.E
            if position == 0:
                st.clear()
                bos = self.bos[moe_i].to(rd).unsqueeze(0).expand(B, E)
                st["topi"], st["gates"] = self._route_from_logits(bos, moe.k)
                st["seg_len"] = torch.zeros(B, dtype=torch.long, device=dev)
                st["j"] = torch.zeros(B, dtype=torch.long, device=dev)
                st["sum"] = torch.zeros(B, hv.shape[1], dtype=rd, device=dev)
                st["am"] = torch.full((B,), float("-inf"), dtype=rd, device=dev)
                st["aden"] = torch.zeros(B, dtype=rd, device=dev)
                st["anum"] = torch.zeros(B, hv.shape[1], dtype=rd, device=dev)
                st["hist"] = []
                st["e"] = None
                st["g"] = torch.zeros(B, sr.gru_hidden, dtype=rd, device=dev) if sr.state == "gru" else None
            y = moe.experts(hv, st["topi"], st["gates"].to(hv.dtype))
            if moe.shared is not None:
                y = y + moe.shared(hv)
            x = x + y.view(B, 1, -1)
            out.append(st["topi"].clone())
            # ---- accumulate the current (now observed) token into the open segment
            hr = hv.to(rd)
            st["seg_len"] = st["seg_len"] + 1
            st["sum"] = st["sum"] + hr
            st["last"] = hr
            if sr.summary == "attn":
                sc = (hr @ self.attn_q[moe_i].to(rd)) / math.sqrt(hr.shape[1])
                m_new = torch.maximum(st["am"], sc)
                scale = torch.exp(st["am"] - m_new)
                st["aden"] = st["aden"] * scale + torch.exp(sc - m_new)
                st["anum"] = st["anum"] * scale.unsqueeze(1) + hr * torch.exp(sc - m_new).unsqueeze(1)
                st["am"] = m_new
            # ---- causal boundary decision after this token
            if sr.seg == "fixed":
                cand = None
            elif sr.boundary_source == "layer0":
                cand = sh["cand"]
            elif sr.seg == "cp":
                st["hist"].append(hr)
                if len(st["hist"]) > sr.cp_short + sr.cp_long:
                    st["hist"].pop(0)
                sc_ = self._src_window_score(st["hist"], B)
                cand = (sc_ > self.tau[moe_i]) if sc_ is not None else torch.zeros(B, dtype=torch.bool, device=dev)
            else:
                u = hr.double() @ self.bd_w[moe_i].double() + self.bd_b[moe_i].double()
                cand = u > self.tau[moe_i]
            bnd = self._boundary_now(st, cand)
            if bool(bnd.any()):
                state["boundary_events"] += int(bnd.sum())
                if sr.summary == "mean":
                    pooled = st["sum"] / st["seg_len"].clamp(min=1).unsqueeze(1).to(rd)
                elif sr.summary == "last":
                    pooled = st["last"]
                else:
                    pooled = st["anum"] / st["aden"].unsqueeze(1)
                zp = moe.router_logits(pooled)
                state["router_calls"] += 1
                if sr.state == "none":
                    znew = zp
                elif sr.state == "ema":
                    e = pooled if st["e"] is None else torch.where(
                        (st["j"] == 0).unsqueeze(1), pooled, sr.ema_decay * st["e"] + (1 - sr.ema_decay) * pooled)
                    st["e"] = torch.where(bnd.unsqueeze(1), e, st["e"] if st["e"] is not None else e)
                    znew = moe.router_logits(st["e"])
                else:
                    mh = torch.zeros(B, E, dtype=zp.dtype, device=dev).scatter_(1, st["topi"], 1.0)
                    gnew = self.gru[moe_i](torch.cat([zp, mh], 1), st["g"].to(zp.dtype))
                    st["g"] = torch.where(bnd.unsqueeze(1), gnew, st["g"].to(gnew.dtype))
                    znew = zp + self.gru_out[moe_i](gnew)
                ti, ga = self._route_from_logits(znew, moe.k)
                st["topi"] = torch.where(bnd.unsqueeze(1), ti, st["topi"])
                st["gates"] = torch.where(bnd.unsqueeze(1), ga.to(st["gates"].dtype), st["gates"])
                z0 = torch.zeros_like(st["sum"])
                st["sum"] = torch.where(bnd.unsqueeze(1), z0, st["sum"])
                st["anum"] = torch.where(bnd.unsqueeze(1), z0, st["anum"])
                st["aden"] = torch.where(bnd, torch.zeros_like(st["aden"]), st["aden"])
                st["am"] = torch.where(bnd, torch.full_like(st["am"], float("-inf")), st["am"])
                st["j"] = st["j"] + bnd.long()
                st["seg_len"] = torch.where(bnd, torch.zeros_like(st["seg_len"]), st["seg_len"])
            moe_i += 1
        return self.lm_head(self.norm_f(x))[:, 0], out


def build_sr_model(cfg: dict, device, dtype=torch.float32):
    """Global seed -> backbone init (identical to TOKEN-4 for the same seed); extras from a dedicated generator."""
    torch.manual_seed(cfg["init"]["seed"])
    sr = SRRouting.from_cfg(cfg["routing"])
    m = SeqRouteModel(ModelConfig.from_dict(cfg["model"]), sr, extra_seed=cfg["init"]["seed"])
    return m.to(device=device, dtype=dtype)
