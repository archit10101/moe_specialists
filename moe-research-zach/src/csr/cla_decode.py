"""csr-conditional-local-attention-v1: cached prefill + incremental greedy decoding for CLAModel.

Rows are independent single documents of equal length (no padding). The KV cache keeps every position (no eviction or
compression: this experiment does not authorise it), so cache memory is identical for every attention condition.
Per decoding step and layer:
  full : attend to cache[0 .. t]
  local: attend to cache[t-W+1 .. t]  (only the last W keys are read)
  cond : local first; gate z from the local output (frozen threshold); rows with t >= W and z > lambda (or forced)
         additionally attend to cache[0 .. t] and use that output instead (only the selected rows are computed).
The MoE / dense FFN path is the CSRModel token-mode code path.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import rope_cos_sin


class DecodeCache:
    def __init__(self, model, B, t_max, device, dtype):
        H, hd = model.cfg.n_head, model.hd
        self.k = [torch.zeros(B, H, t_max, hd, device=device, dtype=dtype) for _ in model.blocks]
        self.v = [torch.zeros(B, H, t_max, hd, device=device, dtype=dtype) for _ in model.blocks]
        self.t = 0
        self.t_max = t_max

    def nbytes(self):
        return sum(x.numel() * x.element_size() for x in self.k + self.v)


def _single_doc_batch(ids):
    B, L = ids.shape
    return {"input_ids": ids, "doc_ids": torch.zeros_like(ids),
            "positions": torch.arange(L, device=ids.device).unsqueeze(0).expand(B, L).contiguous()}


@torch.no_grad()
def prefill(model, ids, t_max, cache_dtype):
    """ids (B, L). Returns (last-position logits (B, V), cache, forward output)."""
    B, L = ids.shape
    tb = _single_doc_batch(ids)
    Lp = L
    if model.backend == "flex" and (B * L) % 128:
        Lp = (L + 127) // 128 * 128  # right-pad every row with pad positions (own-document pads; no effect on 0..L-1)
        tb = {"input_ids": F.pad(ids, (0, Lp - L), value=1), "doc_ids": F.pad(tb["doc_ids"], (0, Lp - L), value=-1),
              "positions": F.pad(tb["positions"], (0, Lp - L))}
    out = model(tb, return_kv=True)
    cache = DecodeCache(model, B, t_max, ids.device, cache_dtype)
    for li, (k, v) in enumerate(out.kv):
        cache.k[li][:, :, :L] = k[:, :, :L].to(cache_dtype)
        cache.v[li][:, :, :L] = v[:, :, :L].to(cache_dtype)
    cache.t = L
    return out.logits[:, L - 1], cache, out


@torch.no_grad()
def step(model, tok, cache: DecodeCache):
    """tok (B,) token at position cache.t. Returns (logits (B, V), per-conditional-layer selected mask (B,))."""
    B = tok.shape[0]
    t = cache.t
    assert t < cache.t_max
    x = model.tok_emb(tok.view(B, 1))
    pos = torch.full((B, 1), t, device=tok.device, dtype=torch.long)
    cos, sin = rope_cos_sin(pos, model.hd, model.cfg.rope_base, x.dtype)
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    sels = []
    for li, blk in enumerate(model.blocks):
        q, k, v = model._qkv(blk, blk.norm1(x), cos, sin)  # (B, H, 1, hd)
        dt = cache.k[li].dtype
        cache.k[li][:, :, t] = k[:, :, 0].to(dt)
        cache.v[li][:, :, t] = v[:, :, 0].to(dt)
        mode, W = model.modes[li]
        if mode == "full":
            K, V = cache.k[li][:, :, :t + 1], cache.v[li][:, :, :t + 1]
            y = F.scaled_dot_product_attention(q.to(dt), K, V)
        else:
            lo = max(0, t + 1 - W)
            y = F.scaled_dot_product_attention(q.to(dt), cache.k[li][:, :, lo:t + 1], cache.v[li][:, :, lo:t + 1])
            if mode == "cond":
                yl = y.transpose(1, 2).reshape(B, 1, -1)
                z = model.gate_logits(li, yl)[:, 0]
                if model.plan.force == "global":
                    sel = torch.full((B,), t >= W, device=tok.device)
                elif model.plan.force == "local":
                    sel = torch.zeros(B, dtype=torch.bool, device=tok.device)
                else:
                    sel = (z > model.thr[li]) & (t >= W)
                sels.append(sel)
                if t >= W:
                    idx = torch.nonzero(sel).squeeze(1)  # host sync: the branch decides whether to compute
                    if idx.numel():
                        yf = F.scaled_dot_product_attention(q[idx].to(dt), cache.k[li][idx, :, :t + 1],
                                                            cache.v[li][idx, :, :t + 1])
                        y = y.index_copy(0, idx, yf)
        x = x + blk.attn.proj(y.transpose(1, 2).reshape(B, 1, -1))
        h = blk.norm2(x)
        if blk.is_moe:
            y2, _ = blk.ffn(h.reshape(B, -1), None, "global")
            x = x + y2.view(B, 1, -1)
        else:
            x = x + blk.ffn(h)
    cache.t = t + 1
    return model.lm_head(model.norm_f(x))[:, 0], sels


@torch.no_grad()
def greedy(model, prompt, n_new, cache_dtype):
    """Deterministic greedy decoding. Returns (generated (B, n_new), per-step logits list, per-step selections)."""
    t_max = prompt.shape[1] + n_new
    logits, cache, _ = prefill(model, prompt, t_max, cache_dtype)
    toks, all_logits, all_sel = [], [logits], []
    tok = logits.argmax(-1)
    for i in range(n_new):
        toks.append(tok)
        if i == n_new - 1:
            break
        logits, sels = step(model, tok, cache)
        all_logits.append(logits)
        all_sel.append(sels)
        tok = logits.argmax(-1)
    return torch.stack(toks, 1), all_logits, all_sel
