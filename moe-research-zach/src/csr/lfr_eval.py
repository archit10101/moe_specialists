"""lfr evaluation primitives: token-weighted CE (overall / per domain / per document), routing statistics, optional
teacher-mode evaluation with cached forecasts (development splits only), exact counterfactual expert substitution,
rank statistics. Student / inference mode = no forecasts and no labels reach the model."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from .fsp_lib import N_DOM, Split, make_batch, pack_rows, to_torch
from .lfr_lib import build_features, gather_raw, shuffle_donors


def eval_rows(split: Split, max_tokens_per_domain=None, docs=None):
    """Deterministic rows: documents in file order (optionally a given doc subset), capped per domain."""
    cand = range(split.n_docs) if docs is None else docs
    if max_tokens_per_domain is None:
        order = list(cand)
    else:
        used = np.zeros(N_DOM, dtype=np.int64)
        order = []
        for i in cand:
            d = split.domains[i]
            if used[d] < max_tokens_per_domain:
                order.append(int(i))
                used[d] += split.lengths[i] - 1
    return pack_rows(split.lengths, order)


def paper_half(split: Split, half):
    """Paper-hash half (0 = A, 1 = B) of a split: documents whose paper hash maps to the half (fixed salt)."""
    import hashlib
    out = []
    for i, p in enumerate(split.papers):
        u = int.from_bytes(hashlib.sha256(f"lfr-half|{p}".encode()).digest()[:8], "little") / 2**64
        if (u >= 0.5) == bool(half):
            out.append(i)
    return out


def inputs(rows, split, device, cache=None, lam=None, donors=None, need_lab=False):
    a = make_batch(rows, split)
    bt = to_torch(a, device)
    mi = {"input_ids": bt["input_ids"], "doc_ids": bt["doc_ids"], "positions": bt["positions"]}
    if need_lab:
        mi["domain"], mi["seq_id"] = bt["domain"], bt["seq_id"]
    if cache is not None and lam is not None:
        raw = gather_raw(rows, split, cache, donors)
        rt = {k: torch.as_tensor(v.astype(np.int64) if v.dtype == np.uint16 else v, device=device)
              for k, v in raw.items()}
        mi["fc"] = build_features(rt, bt["doc_ids"], bt["positions"])
        mi["lam"] = torch.tensor(float(lam), device=device)
    return a, bt, mi


@torch.no_grad()
def evaluate(model, split: Split, rows, device, cache=None, lam=None, shuffled=False, exclude=None, batch_rows=32,
             per_token=False, routing=True):
    """Returns dict: ce, n, ce_dom, n_dom, doc_ce {gdoc: [sum, n]}, load (L,E), ent (normalized mean routing entropy),
    and with per_token: arrays ce, dom, gdoc, pos, topi (N, L, 4) uint8, tok (N,) for valid target positions.
    lam != None with a cache = teacher mode (development splits only)."""
    model.eval()
    L, E = model.n_moe, model.cfg.n_experts
    ce_dom, n_dom = np.zeros(N_DOM), np.zeros(N_DOM)
    doc_ce = {}
    load = np.zeros((L, E))
    ent_sum, ent_n = 0.0, 0
    pt = {k: [] for k in ("ce", "dom", "gdoc", "pos", "topi")}
    teach = cache is not None and lam is not None and model.cfg.teacher
    for i in range(0, len(rows), batch_rows):
        rr = rows[i:i + batch_rows]
        donors = None
        if shuffled:
            donors = shuffle_donors(rr, split.lengths)
        a, bt, mi = inputs(rr, split, device, cache if teach else None, lam if teach else None, donors)
        if exclude:
            mi["exclude_mask"] = bt["doc_ids"] >= 0
        ctl = {"teach": teach, "collect": routing or per_token, "exclude": exclude}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(mi, ctl)
            h = out["hidden"]
        tgt = bt["targets"]
        ce = torch.zeros(tgt.shape, device=device)
        hf = h.reshape(-1, h.shape[-1])
        tf = tgt.reshape(-1)
        cef = ce.view(-1)
        for s in range(0, tf.numel(), 8192):
            cef[s:s + 8192] = F.cross_entropy(model.lm_logits(hf[s:s + 8192]).float(), tf[s:s + 8192],
                                              ignore_index=-100, reduction="none")
        valid = tgt >= 0
        dom = bt["true_domain"]
        for d in range(N_DOM):
            m = valid & (dom == d)
            ce_dom[d] += float(ce[m].sum())
            n_dom[d] += int(m.sum())
        gd = bt["gdoc"][valid].cpu().numpy()
        cv = ce[valid].double().cpu().numpy()
        uq, inv = np.unique(gd, return_inverse=True)
        for g_, s_, c_ in zip(uq, np.bincount(inv, weights=cv), np.bincount(inv)):
            r = doc_ce.setdefault(int(g_), [0.0, 0])
            r[0] += float(s_)
            r[1] += int(c_)
        if routing or per_token:
            vm = (bt["doc_ids"] >= 0)
            for li, rec in enumerate(out["records"]):
                ti = rec["topi"][vm]
                load[li] += np.bincount(ti.reshape(-1).cpu().numpy(), minlength=E)
                p = rec["p"][vm].float().clamp(min=1e-12)
                ent_sum += float((-(p * p.log()).sum(-1) / math.log(E)).sum())
                ent_n += int(p.shape[0])
        if per_token:
            pt["ce"].append(ce[valid].float().cpu().numpy())
            pt["dom"].append(dom[valid].cpu().numpy().astype(np.int8))
            pt["gdoc"].append(bt["gdoc"][valid].cpu().numpy())
            pt["pos"].append(bt["positions"][valid].cpu().numpy().astype(np.int16))
            pt["topi"].append(torch.stack([r["topi"][valid] for r in out["records"]], 1).to(torch.uint8).cpu().numpy())
    n = n_dom.sum()
    res = {"ce": float(ce_dom.sum() / max(1, n)), "n": int(n),
           "ce_dom": (ce_dom / np.maximum(1, n_dom)).tolist(), "n_dom": n_dom.astype(int).tolist(),
           "doc_ce": doc_ce}
    if routing or per_token:
        res["load"] = (load / np.maximum(1, load.sum(1, keepdims=True))).tolist()
        res["ent"] = ent_sum / max(1, ent_n)
    if per_token:
        res["per_token"] = {k: np.concatenate(v) for k, v in pt.items()}
    return res


# ------------------------------------------------------------------ counterfactual substitution
def first_order_delta(experts, x, grad, r4, g4, n_scale):
    """Delta^(e) = -g4 <dL/dy, E_e(x) - E_r4(x)> * n_scale for every expert (rows: sampled positions)."""
    dots = experts.all_dot(x, grad)  # (n, E)
    ref = dots.gather(1, r4.view(-1, 1))
    return -(g4.view(-1, 1).float()) * (dots - ref) * n_scale


@torch.no_grad()
def exact_delta(model, mi_row, layer, t, x, r4, g4, device, expect_topi=None):
    """Exact substitution at (position t, MoE layer `layer`) of a single row: 33 copies (reference + every expert e in
    slot 4 with the same gate), full re-forward including downstream re-routing. Returns Delta (E,) = L(ref) - L(e)
    with L = summed next-token CE of the row. mi_row: model inputs of ONE row (B=1) incl. targets under '_targets'."""
    E = model.cfg.n_experts
    B = E + 1
    rep = {}
    for k, v in mi_row.items():
        if k in ("_targets",):
            continue
        if k == "fc":
            rep[k] = {kk: vv.expand(B, *vv.shape[1:]).contiguous() for kk, vv in v.items()}
        elif k == "lam":
            rep[k] = v
        else:
            rep[k] = v.expand(B, *v.shape[1:]).contiguous()
    T = mi_row["input_ids"].shape[1]
    ex = model.blocks[layer + 1].ffn.experts
    xr = x.view(1, -1)
    base = ex.one(xr, int(r4))
    delta = torch.zeros(B, xr.shape[-1], device=device, dtype=torch.float64 if xr.dtype == torch.float64 else
                        torch.float32)
    for e in range(E):
        delta[e + 1] = float(g4) * (ex.one(xr, e) - base)[0]
    idx = torch.arange(B, device=device) * T + int(t)
    ctl = {"teach": "fc" in rep, "perturb": (layer, idx, delta), "collect": expect_topi is not None}
    model.eval()
    # full precision (no autocast, no TF32) so that small substitution effects are not lost in rounding noise
    tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        out = model(rep, ctl)
        h = out["hidden"]
        dt = torch.float64 if h.dtype == torch.float64 else torch.float32
        tgt = mi_row["_targets"].expand(B, T)
        ce = F.cross_entropy(model.lm_logits(h).to(dt).reshape(-1, model.cfg.vocab_size), tgt.reshape(-1),
                             ignore_index=-100, reduction="none").view(B, T).double().sum(1)
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32
    if expect_topi is not None:  # reference copy must execute the recorded route at (t, layer)
        got = out["records"][layer]["topi"][0, int(t)].tolist()
        exact_delta.route_ok = sorted(got) == sorted(list(expect_topi))
    return (ce[0] - ce[1:]).double()


def ranks(x):
    x = np.asarray(x, dtype=np.float64)
    o = np.argsort(x, kind="stable")
    r = np.empty_like(o, dtype=np.float64)
    r[o] = np.arange(len(x))
    # average ties
    vals, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    if (cnt > 1).any():
        s = np.bincount(inv, weights=r)
        r = (s / cnt)[inv]
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    d = math.sqrt(float((ra * ra).sum() * (rb * rb).sum()))
    return float((ra * rb).sum() / d) if d > 0 else 0.0


def topk_overlap(a, b, k=4):
    return len(set(np.argsort(-np.asarray(a))[:k].tolist()) & set(np.argsort(-np.asarray(b))[:k].tolist()))


def hungarian(cost):
    """Minimum-cost perfect assignment for a square matrix (Kuhn-Munkres with potentials). Returns col for each row."""
    C = np.asarray(cost, dtype=np.float64)
    n = C.shape[0]
    INF = float("inf")
    u, v = np.zeros(n + 1), np.zeros(n + 1)
    p, way = np.zeros(n + 1, dtype=int), np.zeros(n + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, INF)
        used = np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], INF, 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = C[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    col = np.zeros(n, dtype=int)
    for j in range(1, n + 1):
        col[p[j] - 1] = j - 1
    return col
