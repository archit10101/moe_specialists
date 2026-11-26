"""fspv2 evaluation: token-weighted CE (overall / per domain / per paper), routing specialisation statistics and
functional re-routing ablations. Standalone mode = no labels and no forecasts reach the model (only the
LABEL-SPECIALIST-ORACLE receives labels, by definition)."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from .fsp_lib import N_DOM, Split, make_batch, pack_rows, to_torch
from .fsp_model import N_GENERAL, N_SPEC

POS_BUCKETS = [(0, 64), (64, 256), (256, 1024)]


def eval_rows(split: Split, max_tokens_per_domain=None):
    """Deterministic eval rows: documents in file order, optionally capped per domain (whole documents)."""
    if max_tokens_per_domain is None:
        order = list(range(split.n_docs))
    else:
        used = np.zeros(N_DOM, dtype=np.int64)
        order = []
        for i in range(split.n_docs):
            d = split.domains[i]
            if used[d] < max_tokens_per_domain:
                order.append(i)
                used[d] += split.lengths[i] - 1
    return pack_rows(split.lengths, order)


def _batch(rows, split, device, oracle, force_dom=None):
    a = make_batch(rows, split)
    bt = to_torch(a, device)
    model_in = {"input_ids": bt["input_ids"], "doc_ids": bt["doc_ids"], "positions": bt["positions"]}
    if oracle:
        lab = bt["true_domain"] if force_dom is None else force_dom(bt["true_domain"])
        model_in["domain"] = lab
        model_in["forced"] = lab >= 0
    elif force_dom is not None:
        lab = force_dom(bt["true_domain"])
        model_in["domain"] = lab
        model_in["forced"] = lab >= 0
    return a, bt, model_in


@torch.no_grad()
def evaluate(model, split: Split, rows, device, oracle=False, routing_stats=True, batch_rows=32, exclude=None,
             force_dom=None):
    """Returns dict with ce, per-domain ce, per-doc sums, routing stats. exclude: {moe_idx: experts} applied to every
    token (functional ablation with re-routing). force_dom: fn(true_domain)->domain used to force the specialist
    group (swap analysis)."""
    model.eval()
    cfg = model.cfg
    spec = cfg.routing == "spec"
    L = model.n_moe
    E = cfg.n_experts
    ce_dom = np.zeros(N_DOM)
    n_dom = np.zeros(N_DOM)
    doc_sum = {}
    cnt = np.zeros((L, N_DOM, E))
    spec_hit = np.zeros((L, N_DOM))
    qacc = np.zeros((L, len(POS_BUCKETS), 2))
    qacc_seg = np.zeros((L, 3, 2))
    use_dom = spec or cfg.teacher != "none" or getattr(model, "_eval_use_dom", False)
    for i in range(0, len(rows), batch_rows):
        rr = rows[i:i + batch_rows]
        a, bt, mi = _batch(rr, model_split_guard(split), device, oracle, force_dom)
        if exclude:
            mi["exclude_mask"] = bt["doc_ids"] >= 0
        ctl = {"use_dom": use_dom, "collect": routing_stats, "exclude": exclude}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(mi, ctl)
            logits = model.lm_logits(out["hidden"])
        tgt = bt["targets"]
        ce = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=-100,
                             reduction="none").view(tgt.shape)
        valid = tgt >= 0
        dom = bt["true_domain"]
        cev = (ce * valid).double()
        for d in range(N_DOM):
            m = (dom == d) & valid
            ce_dom[d] += float(cev[m].sum())
            n_dom[d] += int(m.sum())
        gd = bt["gdoc"][valid].cpu().numpy()
        cs = cev[valid].cpu().numpy()
        s = np.bincount(gd, weights=cs, minlength=split.n_docs)
        c = np.bincount(gd, minlength=split.n_docs)
        for k in np.nonzero(c)[0]:
            ps, pc = doc_sum.get(int(k), (0.0, 0))
            doc_sum[int(k)] = (ps + float(s[k]), pc + int(c[k]))
        if routing_stats:
            npm = bt["doc_ids"] >= 0
            domv = dom[npm]
            pos = bt["positions"][npm]
            seg = bt["seg_index"][npm]
            for li, rec in enumerate(out["records"]):
                topi = rec["topi"][npm]  # (n, k)
                oh = torch.zeros(topi.shape[0], E, device=device).scatter_(1, topi, 1.0)
                cnt[li] += torch.zeros(N_DOM, E, device=device).index_add_(0, domv, oh).cpu().numpy()
                if spec:
                    g = (topi[:, 3] - N_GENERAL) // 2
                    for d in range(N_DOM):
                        spec_hit[li, d] += float(((g == d) & (domv == d)).sum())
                if "q_s" in rec:
                    pred = rec["q_s"][npm].argmax(-1)
                    ok = (pred == domv).float()
                    for bi, (lo, hi) in enumerate(POS_BUCKETS):
                        m = (pos >= lo) & (pos < hi)
                        qacc[li, bi] += [float(ok[m].sum()), float(m.sum())]
                    for bi, (lo, hi) in enumerate([(0, 1), (1, 4), (4, 10**6)]):
                        m = (seg >= lo) & (seg < hi)
                        qacc_seg[li, bi] += [float(ok[m].sum()), float(m.sum())]
    res = {"ce": float(ce_dom.sum() / max(1, n_dom.sum())), "n_tokens": int(n_dom.sum()),
           "ce_domain": (ce_dom / np.maximum(1, n_dom)).tolist(), "n_domain": n_dom.tolist()}
    res["doc_ce"] = {int(k): v for k, v in doc_sum.items()}
    if routing_stats:
        res["routing"] = routing_summary(cnt, spec_hit, qacc, qacc_seg, spec, cfg.top_k)
        res["counts"] = cnt.tolist()
    return res


def model_split_guard(split):
    return split


def mutual_info(c):
    """c (D, E) joint counts -> I(domain; expert) in bits."""
    p = c / max(c.sum(), 1)
    pd = p.sum(1, keepdims=True)
    pe = p.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = p * np.log2(p / (pd * pe))
    return float(np.nansum(t))


def routing_summary(cnt, spec_hit, qacc, qacc_seg, spec, k):
    L, D, E = cnt.shape
    tok_dom = cnt.sum(2) / k  # tokens per (layer, domain)
    out = {"mi_bits": [mutual_info(cnt[l]) for l in range(L)]}
    load = cnt.sum(1) / np.maximum(1, cnt.sum((1, 2)))[:, None]  # (L, E) share of slots
    out["load_share"] = load.tolist()
    out["unused_experts_per_layer"] = [int((load[l] < 1e-4).sum()) for l in range(L)]
    # domain profile of each expert: P(domain | expert), normalised by domain size
    prof = cnt / np.maximum(1, tok_dom)[:, :, None]  # routing rate of domain d to expert e
    prof = prof / np.maximum(1e-12, prof.sum(1, keepdims=True))
    ent = -(prof * np.log2(np.maximum(prof, 1e-12))).sum(1)  # (L, E) domain entropy (bits), balanced-domain
    out["expert_domain_entropy_bits"] = ent.tolist()
    out["expert_argmax_domain"] = prof.argmax(1).tolist()
    out["expert_domain_profile"] = prof.tolist()
    if spec:
        sp = slice(N_GENERAL, N_GENERAL + N_SPEC)
        intended = np.repeat(np.arange(N_DOM), 2)
        out["spec_intended_mass"] = [float(np.mean([prof[l, intended[j], N_GENERAL + j] for j in range(N_SPEC)]))
                                     for l in range(L)]
        out["spec_group_acc"] = (spec_hit.sum(1) / np.maximum(1, tok_dom.sum(1))).tolist()
        out["spec_group_acc_by_domain"] = (spec_hit / np.maximum(1, tok_dom)).tolist()
        spec_share = cnt[:, :, sp].sum(1) / np.maximum(1, cnt[:, :, sp].sum((1, 2)))[:, None]
        out["spec_share"] = spec_share.tolist()
        out["min_spec_share"] = float(spec_share.min())
        out["spec_domain_entropy_bits"] = ent[:, sp].tolist()
        out["spec_group_acc_mean"] = float(np.mean(out["spec_group_acc"]))
    if qacc[..., 1].sum() > 0:
        out["domain_pred_acc_by_pos"] = (qacc[..., 0] / np.maximum(1, qacc[..., 1])).tolist()
        out["domain_pred_acc_by_segidx"] = (qacc_seg[..., 0] / np.maximum(1, qacc_seg[..., 1])).tolist()
        out["domain_pred_acc"] = (qacc[..., 0].sum(1) / np.maximum(1, qacc[..., 1].sum(1))).tolist()
    return out


def posthoc_groups(counts, k=2):
    """TOKEN-4-style models: per layer, greedily assign each domain the k experts with the highest domain lift
    (routing rate of domain d to e divided by the mean rate), without overlap. counts (L, D, E)."""
    counts = np.asarray(counts)
    L, D, E = counts.shape
    groups = {}
    for l in range(L):
        rate = counts[l] / np.maximum(1, counts[l].sum(1, keepdims=True))
        lift = rate / np.maximum(1e-12, rate.mean(0, keepdims=True))
        taken = set()
        g = {d: [] for d in range(D)}
        pairs = sorted(((lift[d, e], d, e) for d in range(D) for e in range(E)), reverse=True)
        for v, d, e in pairs:
            if len(g[d]) < k and e not in taken:
                g[d].append(int(e))
                taken.add(e)
        groups[l] = g
    return groups


def spec_groups(n_moe):
    return {l: {d: [N_GENERAL + 2 * d, N_GENERAL + 2 * d + 1] for d in range(N_DOM)} for l in range(n_moe)}


@torch.no_grad()
def functional_ablation(model, split, rows, device, groups, oracle=False, base=None):
    """For each domain group g: exclude g's experts in every MoE layer (re-route, renormalise), per-domain dCE."""
    base = base or evaluate(model, split, rows, device, oracle=oracle, routing_stats=False)
    out = {"base_ce_domain": base["ce_domain"], "delta": []}
    for g in range(N_DOM):
        excl = {l: groups[l][g] for l in groups}
        r = evaluate(model, split, rows, device, oracle=oracle, routing_stats=False, exclude=excl)
        out["delta"].append([r["ce_domain"][d] - base["ce_domain"][d] for d in range(N_DOM)])
    dl = np.asarray(out["delta"])
    own = np.diag(dl)
    other = np.array([np.delete(dl[g], g).mean() for g in range(N_DOM)])
    out["own"] = own.tolist()
    out["other_mean"] = other.tolist()
    out["own_minus_other"] = (own - other).tolist()
    out["n_groups_own_gt_other"] = int((own > other).sum())
    return out


@torch.no_grad()
def swap_analysis(model, split, rows, device):
    """Spec models: force own group vs force each other group (cyclic shifts) on all tokens; per-domain CE."""
    res = {}
    for sh in range(N_DOM):
        r = evaluate(model, split, rows, device, routing_stats=False,
                     force_dom=lambda t, sh=sh: torch.where(t >= 0, (t + sh) % N_DOM, t))
        res[sh] = r["ce_domain"]
    own = np.asarray(res[0])
    others = np.mean([res[s] for s in range(1, N_DOM)], 0)
    return {"ce_by_shift": res, "forced_own": own.tolist(), "forced_other_mean": others.tolist(),
            "swap_harm": (others - own).tolist()}
