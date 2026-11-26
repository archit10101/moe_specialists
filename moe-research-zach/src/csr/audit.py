"""specialization-audit-v1 model-side analysis (docs/SPECIALIZATION_AUDIT_V1_PREREG.md §6, §8).

One forward implementation for routing extraction, baseline CE and single-token counterfactual expert replacement.
It reads model parameters only (never writes them) and takes no label input of any kind.

original_pass : full forward mirroring CSRModel.forward (manual fp32 attention), returning logits, routing records
                and per-block caches (residual after attention, keys/values, pools, constrained flags).
counterfactual: for a set of query tokens, remove one selected expert at MoE layer L for that token only, re-select
                top-k among the remaining eligible experts (pool for constrained tokens), keep the condition's gate
                normalization and the shared expert, and recompute that token's own stream through later blocks with
                every other token's keys/values held at their original values.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .model import CSRModel, RoutingConfig, apply_rope, build_chunk_layout, rope_cos_sin


def _qkv(attn, xn, cos, sin):
    B, T, D = xn.shape
    q, k, v = attn.qkv(xn).view(B, T, 3, attn.n_head, attn.hd).permute(2, 0, 3, 1, 4)
    return apply_rope(q, cos, sin), apply_rope(k, cos, sin), v


def _attend_full(q, k, v, mask):
    s = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    s = s.masked_fill(~mask, float("-inf"))
    return torch.softmax(s, dim=-1) @ v


def select_topk(probs_full, cand, k, gate_norm, logits=None):
    """probs_full (N, E); cand (N, E) bool or None. Same rule as MoE.select."""
    if cand is None:
        gates, topi = probs_full.topk(k, dim=-1)
        return topi, gates
    if gate_norm == "global":
        _, topi = probs_full.masked_fill(~cand, -1.0).topk(k, dim=-1)
        return topi, probs_full.gather(1, topi)
    gates, topi = torch.softmax(logits.masked_fill(~cand, float("-inf")), dim=-1).topk(k, dim=-1)
    return topi, gates


@torch.no_grad()
def original_pass(model: CSRModel, batch, routing: RoutingConfig, keep_cache=True):
    ids, doc_ids, positions = batch["input_ids"], batch["doc_ids"], batch["positions"]
    B, T = ids.shape
    x = model.tok_emb(ids)
    cos, sin = rope_cos_sin(positions, model.cfg.d_model // model.cfg.n_head, model.cfg.rope_base, x.dtype)
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    mask = model.attention_mask(doc_ids, positions)
    nonpad = doc_ids >= 0
    layout = build_chunk_layout(batch["chunk_ids"], batch["chunk_pos"], routing.warmup) \
        if routing.mode == "pool" else None
    cache, records = [], []
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
        constrained = torch.zeros(B, T, dtype=torch.bool, device=ids.device)
        cand = None
        pool_tok = None
        if layout is not None:
            hbar = h[layout.rows_b.unsqueeze(1), layout.prefix_idx].mean(dim=1)
            pidx = moe.router_logits(hbar).topk(routing.pool_size, dim=-1).indices
            pool_mask = torch.zeros(hbar.shape[0], moe.E, dtype=torch.bool, device=ids.device).scatter_(1, pidx, True)
            constrained = layout.eligible & nonpad
            pool_tok = torch.zeros(B, T, moe.E, dtype=torch.bool, device=ids.device)
            pool_tok[constrained] = pool_mask[layout.token_row[constrained]]
            cand = torch.ones(B, T, moe.E, dtype=torch.bool, device=ids.device)
            cand[constrained] = pool_tok[constrained]
            cand = cand.view(B * T, moe.E)
        logits = moe.router_logits(h.reshape(B * T, -1))
        probs = torch.softmax(logits, dim=-1)
        topi, gates = select_topk(probs, cand, moe.k, routing.gate_norm, logits)
        y = moe.experts(h.reshape(B * T, -1), topi, gates)
        if moe.shared is not None:
            y = y + moe.shared(h.reshape(B * T, -1))
        if keep_cache:
            c.update({"x_mid": x, "h": h, "probs": probs.view(B, T, -1), "logits": logits.view(B, T, -1),
                      "topi": topi.view(B, T, -1), "y": y.view(B, T, -1)})
        x = x + y.view(B, T, -1)
        records.append({"topi": topi.view(B, T, -1), "gates": gates.view(B, T, -1), "constrained": constrained,
                        "probs_full": probs.view(B, T, -1), "pool_tok": pool_tok})
        c["constrained"], c["pool_tok"] = constrained, pool_tok
        cache.append(c)
    logits = model.lm_head(model.norm_f(x))
    return {"logits": logits, "records": records, "cache": cache, "mask": mask, "cos": cos, "sin": sin}


@torch.no_grad()
def counterfactual(model: CSRModel, orig, routing: RoutingConfig, moe_layer: int, qpos, qvalid, remove_expert,
                   force_recompute=False):
    """Single-token counterfactuals for query tokens arranged per batch row.

    qpos (B, N) long positions, qvalid (B, N) bool, remove_expert (B, N) long: expert removed at MoE layer
    `moe_layer` for that token only. Each query is its own counterfactual: its stream is recomputed through later
    blocks while all other tokens' keys/values stay original. Queries where the removed expert was not selected are
    returned with their original logits (bit-identical) unless force_recompute.
    Returns dict(logits (B, N, V), new_topi (B, N, k), affected (B, N) bool)."""
    blocks = list(model.blocks)
    bL = [i for i, b in enumerate(blocks) if b.is_moe][moe_layer]
    cL = orig["cache"][bL]
    moe = blocks[bL].ffn
    B, N = qpos.shape
    E, k = moe.E, moe.k
    ar = torch.arange(B, device=qpos.device)[:, None]
    topi0 = cL["topi"][ar, qpos]  # (B, N, k)
    affected = (topi0 == remove_expert[..., None]).any(-1) & qvalid
    orig_logits = orig["logits"][ar, qpos]
    if not force_recompute and not bool(affected.any()):
        return {"logits": orig_logits, "new_topi": topi0, "affected": affected}
    probs = cL["probs"][ar, qpos].reshape(B * N, E)
    lg_r = cL["logits"][ar, qpos].reshape(B * N, E)
    if cL["pool_tok"] is not None:
        con = cL["constrained"][ar, qpos]
        pt = cL["pool_tok"][ar, qpos]
        cand = torch.where(con[..., None], pt, torch.ones_like(pt)).reshape(B * N, E)
    else:
        cand = torch.ones(B * N, E, dtype=torch.bool, device=qpos.device)
    rm = torch.where(affected | (force_recompute & qvalid), remove_expert, torch.full_like(remove_expert, -1))
    cand = cand & ~(torch.arange(E, device=qpos.device)[None] == rm.reshape(-1, 1))
    topi, gates = select_topk(probs, cand, k, routing.gate_norm, lg_r)
    h = cL["h"][ar, qpos].reshape(B * N, -1)
    y = moe.experts(h, topi, gates)
    if moe.shared is not None:
        y = y + moe.shared(h)
    xq = cL["x_mid"][ar, qpos] + y.view(B, N, -1)  # (B, N, d)
    mask, cos, sin = orig["mask"], orig["cos"], orig["sin"]
    cq, sq = cos[ar, 0, qpos].unsqueeze(1), sin[ar, 0, qpos].unsqueeze(1)  # (B, 1, N, hd)
    mrow = mask[ar, 0, qpos].unsqueeze(1)  # (B, 1, N, T)
    hidx = qpos[:, None, :, None]  # (B, 1, N, 1)
    for bi in range(bL + 1, len(blocks)):
        blk = blocks[bi]
        c = orig["cache"][bi]
        K, V = c["k"], c["v"]  # (B, H, T, hd)
        H, hd = blk.attn.n_head, blk.attn.hd
        qkv = blk.attn.qkv(blk.norm1(xq)).view(B, N, 3, H, hd).permute(2, 0, 3, 1, 4)  # (3, B, H, N, hd)
        q = apply_rope(qkv[0], cq, sq)
        k_self = apply_rope(qkv[1], cq, sq)
        v_self = qkv[2]
        s = (q @ K.transpose(-1, -2)) / math.sqrt(hd)  # (B, H, N, T)
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
        lg = m2.router_logits(hq)
        pr = torch.softmax(lg, dim=-1)
        cand2 = None
        if c["pool_tok"] is not None:
            con2 = c["constrained"][ar, qpos]
            pt2 = c["pool_tok"][ar, qpos]
            cand2 = torch.where(con2[..., None], pt2, torch.ones_like(pt2)).reshape(B * N, E)
        ti, gt = select_topk(pr, cand2, m2.k, routing.gate_norm, lg)
        y2 = m2.experts(hq, ti, gt)
        if m2.shared is not None:
            y2 = y2 + m2.shared(hq)
        xq = xq + y2.view(B, N, -1)
    cf_logits = model.lm_head(model.norm_f(xq))
    use = affected | (force_recompute & qvalid) if force_recompute else affected
    logits = torch.where(use[..., None], cf_logits, orig_logits)
    new_topi = torch.where(use[..., None], topi.view(B, N, k), topi0)
    return {"logits": logits, "new_topi": new_topi, "affected": affected}


def token_ce(logits, targets):
    return F.cross_entropy(logits.float(), targets, reduction="none")


def param_fingerprint(model):
    """Exact fingerprint of all parameter tensors (sha256 over bytes) for no-mutation checks."""
    import hashlib
    h = hashlib.sha256()
    for n, p in sorted(model.state_dict().items()):
        h.update(n.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
