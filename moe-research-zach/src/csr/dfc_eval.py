"""csr-domain-forecast-curriculum-v1 evaluation passes and specialisation audit metrics (calibration only for audits;
validation only via dfc_evaluate.py after frozen.json)."""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from .dfc_lib import N_DOM, load_stats, make_batch, mi_from_counts, to_torch
from .dfc_model import provisional_group


@torch.no_grad()
def eval_pass(model, rows, split, doms, device, cache=None, strength=0.0, disable_fn=None, counts=False,
              per_doc=False, rows_per_batch=32, gate_stats=False):
    """Token-weighted CE overall / per domain; optional per-(layer, domain, expert) top-4 counts, per-document NLL and
    per-document (layer, expert) counts. disable_fn(tb) -> list of (layer_id, experts, token mask) for ablation."""
    was = model.training
    model.eval()
    E, Lm = model.cfg.n_experts, len(model.cfg.moe_layers)
    dom_nll = np.zeros(N_DOM)
    dom_n = np.zeros(N_DOM)
    C = np.zeros((Lm, N_DOM, E)) if counts else None
    doc_nll, doc_n, doc_counts = {}, {}, {}
    G = np.zeros((Lm, N_DOM, 2)) if gate_stats else None  # gate sum, n
    RR = np.zeros((Lm, N_DOM, 2)) if gate_stats else None  # rerouted (top-4 set differs from token-only), n
    t0 = time.time()
    use_amp = device.type == "cuda"
    for s in range(0, len(rows), rows_per_batch):
        rr = rows[s:s + rows_per_batch]
        tb = to_torch(make_batch(rr, split, doms, cache if strength and cache is not None else None), device)
        dis = disable_fn(tb) if disable_fn is not None else None
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
            out = model(tb, strength=strength, collect=gate_stats, disable=dis)
        tgt = tb["targets"]
        valid = tgt != -100
        nll = F.cross_entropy(out.logits.float().view(-1, out.logits.shape[-1]), tgt.view(-1), ignore_index=-100,
                              reduction="none").view(tgt.shape)
        if not torch.isfinite(nll[valid]).all():
            raise FloatingPointError("non-finite evaluation NLL")
        dm = tb["domain"]
        for d in range(N_DOM):
            m = valid & (dm == d)
            dom_nll[d] += float(nll[m].double().sum())
            dom_n[d] += int(m.sum())
        if per_doc:
            g = tb["gdoc"][valid].cpu().numpy()
            v = nll[valid].double().cpu().numpy()
            ug, inv = np.unique(g, return_inverse=True)
            for dd, sv, cv in zip(ug, np.bincount(inv, weights=v), np.bincount(inv)):
                doc_nll[int(dd)] = doc_nll.get(int(dd), 0.0) + float(sv)
                doc_n[int(dd)] = doc_n.get(int(dd), 0) + int(cv)
        if counts:
            nonpad = tb["doc_ids"] >= 0
            dflat = dm[nonpad]
            gflat = tb["gdoc"][nonpad]
            for li, r in enumerate(out.routing):
                ti = r["topi"][nonpad]  # (N, 4)
                key = dflat.unsqueeze(1) * E + ti
                C[li] += torch.bincount(key.reshape(-1), minlength=N_DOM * E).view(N_DOM, E).double().cpu().numpy()
                if per_doc:
                    ug, inv = torch.unique(gflat, return_inverse=True)
                    k2 = inv.unsqueeze(1) * E + ti
                    cc = torch.bincount(k2.reshape(-1), minlength=len(ug) * E).view(len(ug), E).cpu().numpy()
                    for j, dd in enumerate(ug.cpu().numpy().tolist()):
                        if dd not in doc_counts:
                            doc_counts[dd] = np.zeros((Lm, E), dtype=np.int64)
                        doc_counts[dd][li] += cc[j]
                if gate_stats and "gate" in r:
                    gv = r["gate"][nonpad].float()
                    rer = (r["topi"][nonpad].sort(-1).values != r["topi_token"][nonpad].sort(-1).values).any(-1)
                    for d in range(N_DOM):
                        m = dflat == d
                        G[li, d] += [float(gv[m].sum()), int(m.sum())]
                        RR[li, d] += [float(rer[m].float().sum()), int(m.sum())]
    if was:
        model.train()
    res = {"ce": float(dom_nll.sum() / dom_n.sum()), "ce_domain": (dom_nll / np.maximum(dom_n, 1)).tolist(),
           "n_tokens": int(dom_n.sum()), "n_tokens_domain": dom_n.tolist(), "seconds": time.time() - t0}
    if counts:
        res["counts"] = C
    if per_doc:
        res["doc_nll"], res["doc_n"] = doc_nll, doc_n
        if counts:
            res["doc_counts"] = doc_counts
    if gate_stats:
        res["gate_mean"] = (G[..., 0] / np.maximum(G[..., 1], 1)).tolist()
        res["rerouted_frac"] = (RR[..., 0] / np.maximum(RR[..., 1], 1)).tolist()
    return res


def routing_metrics(C, doc_counts=None, doc_dom=None, n_null=20, seed=0):
    """C (L, D, E). Returns per-layer MI, normalised MI, H(d|e), p(e|d), loads, provisional-group mass, and a
    load-preserving shuffled-domain null (domain labels permuted across documents)."""
    L, D, E = C.shape
    out = {"mi": [], "mi_norm": [], "h_dom_given_expert": [], "p_e_given_d": [], "load": [], "prov_mass": [],
           "prov_mass_uniform": 4 / E}
    for li in range(L):
        mi, hd, hde = mi_from_counts(C[li])
        out["mi"].append(mi)
        out["mi_norm"].append(mi / hd if hd > 0 else 0.0)
        out["h_dom_given_expert"].append(hde.tolist())
        ped = C[li] / np.maximum(C[li].sum(1, keepdims=True), 1)
        out["p_e_given_d"].append(ped.tolist())
        out["load"].append(load_stats(C[li].sum(0)))
        out["prov_mass"].append([float(ped[d, provisional_group(d)].sum()) for d in range(D)])
    if doc_counts is not None:
        keys = sorted(doc_counts)
        X = np.stack([doc_counts[k] for k in keys])  # (n_docs, L, E)
        dd = np.asarray([doc_dom[k] for k in keys])
        rng = np.random.default_rng(seed)
        nulls = []
        for _ in range(n_null):
            pdm = rng.permutation(dd)
            Cn = np.stack([np.stack([X[pdm == d, li].sum(0) for d in range(D)]) for li in range(L)])
            nulls.append([mi_from_counts(Cn[li])[0] for li in range(L)])
        nulls = np.asarray(nulls)
        out["mi_null_mean"] = nulls.mean(0).tolist()
        out["mi_null_p95"] = np.percentile(nulls, 95, axis=0).tolist()
    return out


def enriched_experts(C_layer, d, k=4):
    """Top-k experts by enrichment p(e|d) / p(e) (ties by index)."""
    ped = C_layer[d] / max(C_layer[d].sum(), 1)
    pe = C_layer.sum(0) / max(C_layer.sum(), 1)
    enr = np.where(pe > 0, ped / np.maximum(pe, 1e-12), 0.0)
    return [int(e) for e in np.argsort(-enr, kind="stable")[:k]]
