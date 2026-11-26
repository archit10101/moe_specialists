"""csr-speculative-pool-routing-v1: incremental (KV-cached) execution, greedy decoding and greedy speculative
decoding for the target families and the frozen draft (docs/SPECULATIVE_POOL_ROUTING_V1_PREREG.md §6).

Ragged batched KV caches: slot = in-document position; each sequence has its own accepted length; stale slots of
rejected tokens are always overwritten before they can be attended to.

Greedy speculative decoding (every block):
  1. the draft proposes D_1..D_gamma (gamma = B - 1) greedily from the accepted prefix x_0..x_s;
  2. the target processes [x_s, D_1..D_gamma] at positions s..s+gamma as ONE pool block whose pool is selected from
     the target's causal state at s and the proposal (spec family) or from nothing (TOKEN-4);
  3. accept the longest prefix of D matching the target's greedy choices; emit the target's choice at the first
     mismatch (or the bonus token) -> it starts the next block, whose proposal and pool are regenerated.
This is exactly the teacher-forced spec-family block rule, so the speculative output must equal sequential greedy
execution of the same target algorithm (verified by spv1_gpu_checks / spv1_bench).
Decoding convention (both for speculation and for the sequential reference): the last prompt token starts a block.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import CSRModel, RoutingConfig, rope_cos_sin
from .spv1_model import (RAND_TABLE_LEN, PoolRouteModel, _attn_qkv, block_start_index)


# ------------------------------------------------------------------ caches
def new_cache(model: CSRModel, Bt: int, Lmax: int, device, dtype):
    H = model.cfg.n_head
    hd = model.cfg.d_model // H
    return [{"k": torch.zeros(Bt, H, Lmax, hd, device=device, dtype=dtype),
             "v": torch.zeros(Bt, H, Lmax, hd, device=device, dtype=dtype)} for _ in model.blocks]


def _inc_attn(blk, x, pos, cos, sin, c):
    Bt, n, D = x.shape
    q, k, v = _attn_qkv(blk, x, cos, sin)
    bidx = torch.arange(Bt, device=x.device).unsqueeze(1).expand(Bt, n)
    c["k"][bidx, :, pos] = k.permute(0, 2, 1, 3).to(c["k"].dtype)
    c["v"][bidx, :, pos] = v.permute(0, 2, 1, 3).to(c["v"].dtype)
    L = c["k"].shape[2]
    mask = (torch.arange(L, device=x.device).view(1, 1, L) <= pos.unsqueeze(-1)).unsqueeze(1)  # (Bt,1,n,L)
    y = F.scaled_dot_product_attention(q, c["k"].to(q.dtype), c["v"].to(q.dtype), attn_mask=mask)
    return blk.attn.proj(y.transpose(1, 2).reshape(Bt, n, D))


def _rope(model, pos, dtype):
    cos, sin = rope_cos_sin(pos, model.cfg.d_model // model.cfg.n_head, model.cfg.rope_base, dtype)
    return cos.unsqueeze(1), sin.unsqueeze(1)


@torch.no_grad()
def draft_inc(draft: CSRModel, tokens, pos, cache):
    """tokens/pos (Bt, n) -> logits (Bt, n, V). Dense draft, one document per sequence."""
    x = draft.tok_emb(tokens)
    cos, sin = _rope(draft, pos, x.dtype)
    for bi, blk in enumerate(draft.blocks):
        x = x + _inc_attn(blk, x, pos, cos, sin, cache[bi])
        x = x + blk.ffn(blk.norm2(x))
    return draft.lm_head(draft.norm_f(x))


def target_prefill(model: PoolRouteModel, tokens, cache, forced_last=False):
    """Full teacher-forced forward over equal-length prompts (Bt, P) at positions 0..P-1; fills the KV cache and
    returns (logits, routing state at the last prompt position)."""
    Bt, P = tokens.shape
    dev = tokens.device
    batch = {"input_ids": tokens, "doc_ids": torch.zeros_like(tokens),
             "positions": torch.arange(P, device=dev).unsqueeze(0).expand(Bt, P).contiguous()}
    with torch.no_grad():
        out = model(batch, collect=True, return_kv=True)
    for bi, (k, v) in enumerate(out.kv):
        cache[bi]["k"][:, :, :P] = k.to(cache[bi]["k"].dtype)
        cache[bi]["v"][:, :, :P] = v.to(cache[bi]["v"].dtype)
    rstate = []
    if model.pr.family != "token":
        for r in out.routing:
            rstate.append({"pool": r["cand"][:, -1].clone(), "hold": (P - 1 - r["bs"][:, -1] + 1).clone()})
    return out.logits, rstate


@torch.no_grad()
def target_inc(model: PoolRouteModel, tokens, pos, cache, rstate, blk_start=None, props=None):
    """Process n new tokens per sequence (Bt, n) at positions pos (Bt, n) with the KV cache and the routing carry
    state. blk_start (Bt, n) bool and props (Bt, n, gamma) are required for the spec family (caller-determined block
    starts and the proposal of each block start). Adaptive family: n must be 1. Returns logits (Bt, n, V) and per-MoE
    layer routing info."""
    pr = model.pr
    Bt, n = tokens.shape
    x = model.tok_emb(tokens)
    cos, sin = _rope(model, pos, x.dtype)
    ehat = None
    if pr.family == "spec" and pr.pool_assign == "learned":
        ehat = model._draft_features(props)  # (Bt, n, gamma, d)
    info = []
    moe_i = 0
    for bi, blk in enumerate(model.blocks):
        x = x + _inc_attn(blk, x, pos, cos, sin, cache[bi])
        h = blk.norm2(x)
        if not blk.is_moe:
            x = x + blk.ffn(h)
            continue
        moe = blk.ffn
        hf = h.reshape(Bt * n, -1)
        z = moe.router_logits(hf)
        if pr.family == "token":
            probs = torch.softmax(z, dim=-1)
            gates, topi = probs.topk(moe.k, dim=-1)
            info.append({"topi": topi.view(Bt, n, -1)})
        else:
            E = moe.E
            probs = torch.softmax(z / pr.router_temperature, -1).view(Bt, n, E)
            hs = h.float() if h.dtype != torch.float64 else h
            ghat_l = model._ghat(moe_i, ehat) if ehat is not None else None
            P_all, _ = model._pools(moe_i, moe, hs, pos, ghat_l)
            st = rstate[moe_i]
            if pr.family == "adaptive":
                assert n == 1, "adaptive incremental execution is one token per call"
                pool, hold = st["pool"], st["hold"]
                learned = (hs[:, 0] @ model.det_w[moe_i].to(hs.dtype) + model.det_b[moe_i].to(hs.dtype))
                miss = 1.0 - (probs[:, 0] * pool.to(probs.dtype)).sum(-1)
                force = (pos[:, 0] == 0) | (hold >= pr.seg_len)
                refresh = force | ((miss + learned) > model.tau[moe_i].to(miss.dtype))
                newpool = torch.where(refresh.unsqueeze(-1), P_all[:, 0], pool)
                st["pool"], st["hold"] = newpool, torch.where(refresh, torch.ones_like(hold), hold + 1)
                cand = newpool.unsqueeze(1)
                starts = refresh.unsqueeze(1)
            else:
                starts = (pos % pr.seg_len == 0) if pr.family == "fixed" else blk_start
                carry = st.get("pool")
                if carry is None:
                    carry = torch.zeros(Bt, E, dtype=torch.bool, device=x.device)
                P_ext = torch.cat([carry.unsqueeze(1), P_all], 1)
                idx = torch.cummax(torch.where(starts, torch.arange(1, n + 1, device=x.device).unsqueeze(0),
                                               torch.zeros_like(pos)), dim=1).values
                cand = P_ext.gather(1, idx.unsqueeze(-1).expand(Bt, n, E))
                st["pool"] = cand[:, -1]
            _, topi = probs.masked_fill(~cand, -1.0).topk(moe.k, dim=-1)
            gates = probs.gather(-1, topi)
            topi, gates = topi.reshape(Bt * n, -1), gates.reshape(Bt * n, -1)
            info.append({"topi": topi.view(Bt, n, -1), "cand": cand, "starts": starts})
        y = moe.experts(hf, topi, gates)
        if moe.shared is not None:
            y = y + moe.shared(hf)
        x = x + y.view(Bt, n, -1)
        moe_i += 1
    return model.lm_head(model.norm_f(x)), info


# ------------------------------------------------------------------ greedy decoding
def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def greedy_generate(model: PoolRouteModel, prompts, G):
    """Lockstep greedy decoding (token / fixed / adaptive families and random controls of these): prefill the
    prompt, then one token per step. Returns generated tokens (Bt, G) and stats."""
    assert model.pr.family in ("token", "fixed", "adaptive")
    Bt, P = prompts.shape
    dev = prompts.device
    dtype = next(model.parameters()).dtype
    cache = new_cache(model, Bt, P + G + 2, dev, dtype)
    logits, rstate = target_prefill(model, prompts, cache)
    cur = logits[:, -1].argmax(-1)
    out = [cur]
    starts = []
    for i in range(G - 1):
        pos = torch.full((Bt, 1), P + i, device=dev, dtype=torch.long)
        lg, info = target_inc(model, cur.unsqueeze(1), pos, cache, rstate)
        cur = lg[:, -1].argmax(-1)
        out.append(cur)
        if info and "starts" in info[0]:
            starts.append(torch.stack([r["starts"][:, 0] for r in info], 0))
    st = {"target_forwards": G, "draft_forwards": 0}
    if starts:
        st["refresh_rate_decode"] = float(torch.stack(starts, 0).float().mean())
    return torch.stack(out, 1), st


@torch.no_grad()
def draft_greedy_full(draft, prefix, gamma):
    """Reference draft proposal: gamma greedy tokens by repeated full forwards over the prefix (1, t)."""
    seq = prefix
    out = []
    for _ in range(gamma):
        b = {"input_ids": seq, "doc_ids": torch.zeros_like(seq),
             "positions": torch.arange(seq.shape[1], device=seq.device).unsqueeze(0)}
        nt = draft(b, RoutingConfig()).logits[:, -1].argmax(-1)
        out.append(nt)
        seq = torch.cat([seq, nt.unsqueeze(1)], 1)
    return torch.stack(out, 1)


@torch.no_grad()
def sequential_spec_family(model: PoolRouteModel, draft, prompt, G):
    """Reference: sequential greedy execution of the spec-family target algorithm for ONE sequence (1, P), one target
    token per step, block starts by the teacher-forced rule (last prompt token starts a block), draft proposals by
    independent full recomputation."""
    pr = model.pr
    assert pr.family == "spec" and prompt.shape[0] == 1
    P = prompt.shape[1]
    dev = prompt.device
    dtype = next(model.parameters()).dtype
    cache = new_cache(model, 1, P + G + 2, dev, dtype)
    target_prefill(model, prompt[:, :P - 1], cache)
    rstate = [{} for _ in range(model.n_moe)]
    seq = prompt
    tok = prompt[:, P - 1]
    D, j = None, None
    gen = []
    n_blocks = 0
    for i in range(G):
        p = P - 1 + i
        new_block = D is None or j >= pr.seg_len or bool(tok[0] != D[0, j - 1])
        if new_block:
            D = draft_greedy_full(draft, seq, pr.gamma)
            j = 0
            n_blocks += 1
        lg, _ = target_inc(model, tok.view(1, 1), torch.tensor([[p]], device=dev), cache, rstate,
                           blk_start=torch.tensor([[new_block]], device=dev), props=D.view(1, 1, -1))
        j += 1
        tok = lg[:, -1].argmax(-1)
        gen.append(tok)
        seq = torch.cat([seq, tok.view(1, 1)], 1)
    return torch.stack(gen, 1), {"blocks": n_blocks}


@torch.no_grad()
def spec_generate(model: PoolRouteModel, draft, prompts, G, gamma, scramble_stale=False):
    """Batched lockstep greedy speculative decoding (target family token or spec; gamma draft tokens per block)."""
    assert model.pr.family in ("token", "spec")
    if model.pr.family == "spec":
        assert gamma == model.pr.gamma
    Bt, P = prompts.shape
    dev = prompts.device
    dtype = next(model.parameters()).dtype
    B = gamma + 1
    Lmax = P + G + 2 * B + 2
    tcache = new_cache(model, Bt, Lmax, dev, dtype)
    dcache = new_cache(draft, Bt, Lmax, dev, dtype)
    target_prefill(model, prompts[:, :P - 1], tcache)
    pre = prompts[:, :P - 1]
    draft_inc(draft, pre, torch.arange(P - 1, device=dev).unsqueeze(0).expand(Bt, P - 1).contiguous(), dcache)
    toks = torch.zeros(Bt, Lmax, dtype=torch.long, device=dev)
    toks[:, :P] = prompts
    n_acc = torch.full((Bt,), P, dtype=torch.long, device=dev)  # accepted length; slot n_acc-1 not yet in target
    bidx = torch.arange(Bt, device=dev)
    rstate = [{} for _ in range(getattr(model, "n_moe", 0))]
    ar = torch.arange(B, device=dev).unsqueeze(0)
    acc_hist = torch.zeros(B, dtype=torch.long, device=dev)  # histogram of accepted draft tokens per block
    rounds = 0
    first = torch.zeros(Bt, B, dtype=torch.bool, device=dev)
    first[:, 0] = True
    done_at = torch.full((Bt,), -1, dtype=torch.long, device=dev)
    blocks_done = torch.zeros(Bt, dtype=torch.long, device=dev)
    while True:
        active = n_acc < P + G
        if not bool(active.any()):
            break
        # ---- draft: re-feed the last two accepted tokens, then gamma - 1 single steps
        p2 = torch.stack([n_acc - 2, n_acc - 1], 1)
        lg = draft_inc(draft, toks.gather(1, p2), p2, dcache)
        props = [lg[:, -1].argmax(-1)]
        for j in range(1, gamma):
            pj = (n_acc - 1 + j).unsqueeze(1)
            lg = draft_inc(draft, props[-1].unsqueeze(1), pj, dcache)
            props.append(lg[:, -1].argmax(-1))
        D = torch.stack(props, 1)  # (Bt, gamma)
        # ---- target verification of one block
        vt = torch.cat([toks.gather(1, (n_acc - 1).unsqueeze(1)), D], 1)
        vpos = (n_acc - 1).unsqueeze(1) + ar
        tl, _ = target_inc(model, vt, vpos, tcache, rstate, blk_start=first,
                           props=D.unsqueeze(1).expand(Bt, B, gamma))
        g = tl.argmax(-1)  # (Bt, B): g[:, i] is the target's greedy token for position s + i + 1
        k = torch.cumprod((D == g[:, :gamma]).long(), 1).sum(1)  # accepted draft tokens
        c = g.gather(1, k.unsqueeze(1)).squeeze(1)
        emit = torch.where(ar[:, :gamma + 1] < k.unsqueeze(1), torch.cat([D, D[:, :1]], 1), c.unsqueeze(1))
        wmask = (ar <= k.unsqueeze(1)) & active.unsqueeze(1)
        slots = (n_acc.unsqueeze(1) + ar).clamp(max=Lmax - 1)
        cur = toks.gather(1, slots)
        toks.scatter_(1, slots, torch.where(wmask, emit, cur))
        acc_hist += torch.bincount(k[active], minlength=B)[:B]
        blocks_done += active.long()
        n_acc = torch.where(active, n_acc + k + 1, n_acc)
        if scramble_stale:  # test hook: overwrite every KV slot at or beyond each accepted length with noise
            for c in tcache + dcache:
                stale = (torch.arange(Lmax, device=dev).view(1, 1, Lmax, 1) >= (n_acc - 1).view(Bt, 1, 1, 1))
                for key in ("k", "v"):
                    c[key].copy_(torch.where(stale, torch.randn_like(c[key]) * 7.0, c[key]))
        done_at = torch.where((done_at < 0) & (n_acc >= P + G), torch.full_like(done_at, rounds + 1), done_at)
        rounds += 1
    blocks = int(acc_hist.sum())
    accepted = int((acc_hist * torch.arange(B, device=dev)).sum())
    emitted = accepted + blocks
    st = {"rounds": rounds, "blocks": blocks, "accepted_draft_tokens": accepted,
          "mean_accepted_per_block": accepted / max(blocks, 1), "mean_emitted_per_block": emitted / max(blocks, 1),
          "accept_hist": acc_hist.tolist(),
          "rejection_position_hist": acc_hist[:gamma].tolist(),  # first mismatch at draft index k (0-based)
          "proposal_regenerations": blocks,
          "target_forwards_per_emitted_token": blocks / max(emitted, 1),
          "draft_forwards_per_emitted_token": blocks * gamma / max(emitted, 1),
          "per_seq_blocks": blocks_done.tolist()}
    return toks[:, P:P + G], st


@torch.no_grad()
def greedy_reference_token(model: PoolRouteModel, prompts, G):
    """Plain greedy TOKEN-4 (or fixed/adaptive) reference = greedy_generate."""
    return greedy_generate(model, prompts, G)[0]
