"""csr-multiscale-routing-v1 model-side audit passes (mirror of csr.audit for MultiscaleModel; new file).

ms_original_pass : full forward (manual fp32 attention) returning logits, routing records and per-block caches
                   (keys/values, residual after attention, h, z_route/probs, top-k, and the causal context means used by
                   each contextual layer).
ms_counterfactual: remove one selected expert at MoE layer L for a query token only; re-select the global top-k among
                   the remaining 31 experts by the original routing probabilities; recompute that token's own stream
                   through later blocks with every other token's keys/values held at their original values. A token's
                   context means exclude the token itself and depend only on earlier tokens, which are unchanged, so the
                   cached context of the query token is exact under its own intervention.
Records carry `constrained` (all False) and `pool_tok` (None) so the specialization-audit-v1 extraction format holds.
"""
from __future__ import annotations

import math

import torch

from .audit import _attend_full, _qkv, select_topk
from .model import apply_rope, rope_cos_sin
from .multiscale import MSRouting, MultiscaleModel, context_means


@torch.no_grad()
def ms_original_pass(model: MultiscaleModel, batch, routing: MSRouting, keep_cache=True):
    ids, doc_ids, positions = batch["input_ids"], batch["doc_ids"], batch["positions"]
    B, T = ids.shape
    x = model.tok_emb(ids)
    cos, sin = rope_cos_sin(positions, model.cfg.d_model // model.cfg.n_head, model.cfg.rope_base, x.dtype)
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    mask = model.attention_mask(doc_ids, positions)
    nonpad = doc_ids >= 0
    cache, records = [], []
    moe_i = 0
    for blk in model.blocks:
        q, k, v = _qkv(blk.attn, blk.norm1(x), cos, sin)
        y = _attend_full(q, k, v, mask)
        x = x + blk.attn.proj(y.transpose(1, 2).reshape(B, T, -1))
        c = {"k": k, "v": v} if keep_cache else {}
        h = blk.norm2(x)
        if not blk.is_moe:
            x = x + blk.ffn(h)
            cache.append(c)
            continue
        moe = blk.ffn
        kind = routing.schedule[moe_i]
        nl, nb = routing.uses_local(kind), routing.uses_broad(kind)
        ctx = None
        if nl or nb:
            m_l, n_l, m_b, n_b = context_means(h, positions, batch["chunk_pos"], nonpad, nl, nb, routing.local_max,
                                               routing.broad_window)
            ctx = {}
            if nl:
                ctx["m_local"] = m_l.reshape(B * T, -1)
            if nb:
                ctx["m_broad"] = m_b.reshape(B * T, -1)
                ctx["ramp"] = (n_b.to(h.dtype) / routing.broad_ramp).clamp(max=1.0).reshape(-1)
        hf = h.reshape(B * T, -1)
        z = model.route_logits(moe, hf, kind, routing, ctx)
        probs = torch.softmax(z, dim=-1)
        topi, gates = select_topk(probs, None, moe.k, "global", z)
        yy = moe.experts(hf, topi, gates)
        if moe.shared is not None:
            yy = yy + moe.shared(hf)
        if keep_cache:
            c.update({"x_mid": x, "h": h, "probs": probs.view(B, T, -1), "logits": z.view(B, T, -1),
                      "topi": topi.view(B, T, -1), "kind": kind,
                      "ctx": None if ctx is None else {kk: vv.view(B, T, *vv.shape[1:]) for kk, vv in ctx.items()}})
        x = x + yy.view(B, T, -1)
        records.append({"topi": topi.view(B, T, -1), "gates": gates.view(B, T, -1),
                        "constrained": torch.zeros(B, T, dtype=torch.bool, device=ids.device),
                        "probs_full": probs.view(B, T, -1), "pool_tok": None})
        c["constrained"], c["pool_tok"] = records[-1]["constrained"], None
        cache.append(c)
        moe_i += 1
    logits = model.lm_head(model.norm_f(x))
    return {"logits": logits, "records": records, "cache": cache, "mask": mask, "cos": cos, "sin": sin}


@torch.no_grad()
def ms_counterfactual(model: MultiscaleModel, orig, routing: MSRouting, moe_layer: int, qpos, qvalid, remove_expert,
                      force_recompute=False):
    blocks = list(model.blocks)
    moe_idx = [i for i, b in enumerate(blocks) if b.is_moe]
    bL = moe_idx[moe_layer]
    cL = orig["cache"][bL]
    moe = blocks[bL].ffn
    B, N = qpos.shape
    E, k = moe.E, moe.k
    ar = torch.arange(B, device=qpos.device)[:, None]
    topi0 = cL["topi"][ar, qpos]
    affected = (topi0 == remove_expert[..., None]).any(-1) & qvalid
    orig_logits = orig["logits"][ar, qpos]
    if not force_recompute and not bool(affected.any()):
        return {"logits": orig_logits, "new_topi": topi0, "affected": affected}
    probs = cL["probs"][ar, qpos].reshape(B * N, E)
    rm = torch.where(affected | (force_recompute & qvalid), remove_expert, torch.full_like(remove_expert, -1))
    cand = ~(torch.arange(E, device=qpos.device)[None] == rm.reshape(-1, 1))
    topi, gates = select_topk(probs, cand, k, "global")
    h = cL["h"][ar, qpos].reshape(B * N, -1)
    y = moe.experts(h, topi, gates)
    if moe.shared is not None:
        y = y + moe.shared(h)
    xq = cL["x_mid"][ar, qpos] + y.view(B, N, -1)
    mask, cos, sin = orig["mask"], orig["cos"], orig["sin"]
    cq, sq = cos[ar, 0, qpos].unsqueeze(1), sin[ar, 0, qpos].unsqueeze(1)
    mrow = mask[ar, 0, qpos].unsqueeze(1)
    hidx = qpos[:, None, :, None]
    for bi in range(bL + 1, len(blocks)):
        blk = blocks[bi]
        c = orig["cache"][bi]
        K, V = c["k"], c["v"]
        H, hd = blk.attn.n_head, blk.attn.hd
        qkv = blk.attn.qkv(blk.norm1(xq)).view(B, N, 3, H, hd).permute(2, 0, 3, 1, 4)
        q = apply_rope(qkv[0], cq, sq)
        k_self = apply_rope(qkv[1], cq, sq)
        v_self = qkv[2]
        s = (q @ K.transpose(-1, -2)) / math.sqrt(hd)
        s = s.scatter(3, hidx.expand(-1, H, -1, -1), ((q * k_self).sum(-1, keepdim=True) / math.sqrt(hd)))
        s = s.masked_fill(~mrow, float("-inf"))
        p = torch.softmax(s, dim=-1)
        o = p @ V
        p_self = p.gather(3, hidx.expand(-1, H, -1, -1))
        v_orig_self = V.gather(2, qpos[:, None, :, None].expand(-1, H, -1, hd))
        o = o + p_self * (v_self - v_orig_self)
        xq = xq + blk.attn.proj(o.transpose(1, 2).reshape(B, N, H * hd))
        hq = blk.norm2(xq).reshape(B * N, -1)
        if not blk.is_moe:
            xq = xq + blk.ffn(hq).view(B, N, -1)
            continue
        m2 = blk.ffn
        ctx_q = None
        if c["ctx"] is not None:  # the query token's own causal context (excludes itself; earlier tokens unchanged)
            ctx_q = {kk: vv[ar, qpos].reshape(B * N, *vv.shape[2:]) for kk, vv in c["ctx"].items()}
        z = model.route_logits(m2, hq, c["kind"], routing, ctx_q)
        pr = torch.softmax(z, dim=-1)
        ti, gt = select_topk(pr, None, m2.k, "global")
        y2 = m2.experts(hq, ti, gt)
        if m2.shared is not None:
            y2 = y2 + m2.shared(hq)
        xq = xq + y2.view(B, N, -1)
    cf_logits = model.lm_head(model.norm_f(xq))
    use = affected | (force_recompute & qvalid) if force_recompute else affected
    logits = torch.where(use[..., None], cf_logits, orig_logits)
    new_topi = torch.where(use[..., None], topi.view(B, N, k), topi0)
    return {"logits": logits, "new_topi": new_topi, "affected": affected}
