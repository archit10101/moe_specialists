"""csr-conditional-local-attention-v1: attention kernels (docs/CONDITIONAL_LOCAL_ATTENTION_V1_PREREG.md).

Attention semantics (identical for every backend; equal to CSRModel.attention_mask for window=None):
  query at flat position q may attend key k iff  gdoc[k] == gdoc[q]  and  k <= q  and (window is None or q - k < W).
  Packed rows are flattened to one sequence of N = B*T positions; every document has a unique global id; every pad
  position has its own unique id (a pad attends only to itself, exactly as the original `pad_self` rule).

Backends
  "flex"  : torch FlexAttention (compiled Triton kernel) with BlockMasks built directly from vectorised block lists
            (no dense N x N mask is ever evaluated or materialised). Head dim 24 is zero-padded to 32 (FlexAttention
            in torch 2.5.1 requires a power of two); with the explicit scale 1/sqrt(24) this is exact: padded q/k
            columns add 0 to every dot product and padded v columns produce outputs that are discarded.
  "dense" : independent reference with explicit boolean masks and F.scaled_dot_product_attention in the original
            (B, H, T, hd) layout. For window=None it is literally the original CSRModel computation.

Compact global attention (conditional candidates): only the selected queries are gathered into a contiguous prefix
of a fixed-size query buffer (P = N slots; unused slots are marked -1) and attend to all keys of their document.
Query blocks without any selected query get zero key blocks in the BlockMask, so the kernel performs no work for
them (forward and backward). The buffer has a fixed shape so the compiled kernel never recompiles, and building it
needs no host synchronisation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

BLOCK = 128
HD_PAD = 32
_FLEX = None


def flex_fn():
    global _FLEX
    if _FLEX is None:
        import torch._dynamo
        from torch.nn.attention.flex_attention import flex_attention
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 256)
        _FLEX = torch.compile(flex_attention, dynamic=False)
    return _FLEX


@dataclass
class Layout:
    B: int
    T: int
    N: int
    gdoc: torch.Tensor  # (N,) int64 global document id; pads: unique negative id
    dstart: torch.Tensor  # (N,) flat index of the first position of the document (pads: own index)
    nonpad: torch.Tensor  # (N,) bool


def make_layout(doc_ids: torch.Tensor) -> Layout:
    B, T = doc_ids.shape
    N = B * T
    dev = doc_ids.device
    idx = torch.arange(N, device=dev)
    d = doc_ids.reshape(-1)
    gdoc = torch.where(d >= 0, (idx // T) * T + d, -(idx + 1))
    new = torch.ones(N, dtype=torch.bool, device=dev)
    new[1:] = gdoc[1:] != gdoc[:-1]
    dstart = torch.cummax(torch.where(new, idx, torch.zeros_like(idx)), 0).values
    return Layout(B, T, N, gdoc, dstart, d >= 0)


# ------------------------------------------------------------------ block lists
def block_lists(qpos: torch.Tensor, lay: Layout, window, n_kv: int):
    """qpos: (P,) query flat positions, ascending, empty slots (-1) only at the end; P % BLOCK == 0.
    Returns (partial, full) int8 (n_qb, n_kb): `full` blocks need no mask (every pair allowed), `partial` blocks are
    the remaining blocks that may contain an allowed pair. Exact superset of the allowed pairs (tested)."""
    P = qpos.numel()
    nqb, nkb = P // BLOCK, (n_kv + BLOCK - 1) // BLOCK
    qb = qpos.view(nqb, BLOCK)
    valid = qb >= 0
    nvalid = valid.sum(1)
    anyv, allv = nvalid > 0, nvalid == BLOCK
    a = qb[:, 0].clamp(min=0)
    b = qb.gather(1, (nvalid - 1).clamp(min=0).unsqueeze(1)).squeeze(1).clamp(min=0)
    ds_a, ds_b = lay.dstart[a], lay.dstart[b]
    lo = ds_a if window is None else torch.maximum(ds_a, a - window + 1)
    ks = torch.arange(nkb, device=qpos.device) * BLOCK
    ke = ks + BLOCK - 1
    cand = anyv[:, None] & (ke[None] >= lo[:, None]) & (ks[None] <= b[:, None])
    full = allv[:, None] & (ds_b[:, None] <= ks[None]) & (ke[None] <= a[:, None])
    if window is not None:
        full = full & (b[:, None] - ks[None] < window)
    full = full & cand
    return (cand & ~full).to(torch.int8), full.to(torch.int8)


def _ordered(dense):
    """Same result as torch's _dense_to_ordered: per row, the number of set blocks and their indices first (stable)."""
    d = dense.to(torch.int32)
    return (d.sum(-1).to(torch.int32)[None, None].contiguous(),
            torch.argsort(d, dim=-1, descending=True, stable=True).to(torch.int32)[None, None].contiguous())


def _block_mask(partial, full, mask_mod):
    """BlockMask from dense (n_qb, n_kb) block matrices. Equivalent to torch's _create_sparse_block_from_block_mask
    (kv lists from the rows, q lists from the transpose) without its vmap round trip (pure launch overhead)."""
    from torch.nn.attention.flex_attention import BlockMask
    kvn, kvi = _ordered(partial)
    fkvn, fkvi = _ordered(full)
    qn, qi = _ordered(partial.transpose(0, 1))
    fqn, fqi = _ordered(full.transpose(0, 1))
    return BlockMask(kvn, kvi, fkvn, fkvi, qn, qi, fqn, fqi, BLOCK_SIZE=(BLOCK, BLOCK), mask_mod=mask_mod)


def full_or_local_block_mask(lay: Layout, window):
    gdoc = lay.gdoc
    if window is None:
        def mm(b, h, q, k):
            return (gdoc[q] == gdoc[k]) & (k <= q)
    else:
        W = int(window)

        def mm(b, h, q, k):
            return (gdoc[q] == gdoc[k]) & (k <= q) & (q - k < W)
    qpos = torch.arange(lay.N, device=gdoc.device)
    p, f = block_lists(qpos, lay, window, lay.N)
    return _block_mask(p, f, mm)


def compact_block_mask(qpos: torch.Tensor, lay: Layout):
    """Full (window=None) document-causal attention for the queries listed in qpos (-1 = empty slot)."""
    gdoc = lay.gdoc
    qdoc = torch.where(qpos >= 0, gdoc[qpos.clamp(min=0)], torch.full_like(qpos, -(lay.N + 7)))

    def mm(b, h, qi, k):
        qp = qpos[qi]
        return (qp >= 0) & (qdoc[qi] == gdoc[k]) & (k <= qp)
    p, f = block_lists(qpos, lay, None, lay.N)
    return _block_mask(p, f, mm)


def compact_order(sel_flat: torch.Tensor):
    """(qpos, order): order = permutation of 0..N-1 with the selected flat positions first (ascending); qpos = order
    with every slot after the selected ones set to -1 (empty). Fixed size N, no host synchronisation."""
    N = sel_flat.numel()
    idx = torch.arange(N, device=sel_flat.device)
    order = torch.argsort(torch.where(sel_flat, idx, idx + N))
    nsel = sel_flat.sum()
    return torch.where(idx < nsel, order, torch.full_like(order, -1)), order


def compact_positions(sel_flat: torch.Tensor):
    return compact_order(sel_flat)[0]


# ------------------------------------------------------------------ kernels
def _flex_eager():
    from torch.nn.attention.flex_attention import flex_attention
    return flex_attention


def _pad(x):
    return F.pad(x, (0, HD_PAD - x.shape[-1])) if x.shape[-1] < HD_PAD else x


def flex_attend(q, k, v, block_mask, hd):
    """q (1, H, P, hd); k, v (1, H, N, hd) -> (1, H, P, hd). Under autocast the inputs are cast to the autocast
    dtype first (exactly what autocast does to scaled_dot_product_attention inputs in the original model)."""
    dt = torch.get_autocast_dtype(q.device.type) if torch.is_autocast_enabled(q.device.type) else q.dtype
    fn = flex_fn() if q.device.type == "cuda" else _flex_eager()  # CPU: eager reference (tests only)
    with torch.autocast(device_type=q.device.type, enabled=False):
        y = fn(_pad(q.to(dt)), _pad(k.to(dt)), _pad(v.to(dt)), block_mask=block_mask,
                      scale=1.0 / math.sqrt(hd))
    return y[..., :hd]


def dense_mask(doc_ids, window):
    """(B, 1, T, T) bool; window=None reproduces CSRModel.attention_mask exactly."""
    B, T = doc_ids.shape
    same = doc_ids.unsqueeze(2) == doc_ids.unsqueeze(1)
    causal = torch.ones(T, T, dtype=torch.bool, device=doc_ids.device).tril()
    m = same & causal & (doc_ids >= 0).unsqueeze(2)
    if window is not None:
        ar = torch.arange(T, device=doc_ids.device)
        m = m & ((ar[:, None] - ar[None, :]) < window)
    pad_self = (doc_ids < 0).unsqueeze(2) & torch.eye(T, dtype=torch.bool, device=doc_ids.device)
    return (m | pad_self).unsqueeze(1)


class AttnContext:
    """Per-forward cache of masks (shared by all layers). q, k, v are (B, H, T, hd)."""

    def __init__(self, doc_ids, backend):
        assert backend in ("flex", "dense", "sdpa_causal"), backend
        self.backend = backend
        self.doc_ids = doc_ids
        self.B, self.T = doc_ids.shape
        self.lay = make_layout(doc_ids)
        if backend == "flex":
            assert self.lay.N % BLOCK == 0, "flex backend needs B*T to be a multiple of 128"
        if backend == "sdpa_causal":  # benchmark-only baseline variant: one document per row, no padding
            assert bool((doc_ids == 0).all()), "sdpa_causal requires one unpadded document per row"
        self._masks = {}

    def _mask(self, window):
        if window not in self._masks:
            self._masks[window] = (full_or_local_block_mask(self.lay, window) if self.backend == "flex"
                                   else dense_mask(self.doc_ids, window))
        return self._masks[window]

    @staticmethod
    def _flat(x):  # (B, H, T, hd) -> (1, H, B*T, hd)
        B, H, T, hd = x.shape
        return x.transpose(0, 1).reshape(1, H, B * T, hd)

    def attend(self, q, k, v, window):
        """Dense-query attention (full: window=None; local: window=W). Returns (B, T, H*hd)."""
        B, H, T, hd = q.shape
        if self.backend == "sdpa_causal":
            assert window is None, "sdpa_causal is a full-attention baseline variant only"
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # flash kernel (benchmarks only)
            return y.transpose(1, 2).reshape(B, T, H * hd)
        if self.backend == "dense":
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=self._mask(window))
            return y.transpose(1, 2).reshape(B, T, H * hd)
        y = flex_attend(self._flat(q), self._flat(k), self._flat(v), self._mask(window), hd)
        return y.view(H, B, T, hd).permute(1, 2, 0, 3).reshape(B, T, H * hd)

    def attend_compact(self, q, k, v, sel):
        """Full document-causal attention computed ONLY for the selected queries. sel (B, T) bool.
        Returns (qpos (P,), order (P,), y (P, H*hd)): qpos = flat positions (-1 = empty slot; its row of y is 0);
        order = a permutation-valued index with qpos[i] == order[i] wherever qpos[i] >= 0 (unique indices, so
        gathers with it have cheap backward passes)."""
        B, H, T, hd = q.shape
        N = B * T
        sel_flat = sel.reshape(-1)
        if self.backend == "dense":
            qpos = torch.nonzero(sel_flat).squeeze(1)  # reference path: host sync allowed
            n = qpos.numel()
            if n == 0:
                return qpos, qpos, q.new_zeros(0, H * hd)
            rows, cols = qpos // T, qpos % T
            qs = q.transpose(1, 2).reshape(B * T, H, hd).index_select(0, qpos).unsqueeze(2)  # (n, H, 1, hd)
            ks, vs = k.index_select(0, rows), v.index_select(0, rows)  # (n, H, T, hd); index_add backward
            d = self.doc_ids.index_select(0, rows)  # (n, T)
            ar = torch.arange(T, device=q.device)
            m = (d == d.gather(1, cols[:, None])) & (ar[None] <= cols[:, None])
            y = F.scaled_dot_product_attention(qs, ks, vs, attn_mask=m[:, None, None, :])
            return qpos, qpos, y.reshape(n, H * hd)
        qpos, order = compact_order(sel_flat)
        bm = compact_block_mask(qpos, self.lay)
        qf = self._flat(q)
        # gather with the full permutation (unique indices -> cheap backward); the unselected queries that fill the
        # empty slots are masked (qpos = -1) and their query blocks have no key blocks, so they are never computed
        qg = qf.index_select(2, order)
        y = flex_attend(qg, self._flat(k), self._flat(v), bm, hd)  # (1, H, P, hd)
        return qpos, order, y[0].transpose(0, 1).reshape(N, H * hd)


def kv_blocks_computed(block_mask):
    """Number of (q-block, kv-block, head-broadcast) tiles the kernel computes (partial + full)."""
    n = int(block_mask.kv_num_blocks.sum())
    if block_mask.full_kv_num_blocks is not None:
        n += int(block_mask.full_kv_num_blocks.sum())
    return n
