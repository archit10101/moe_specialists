"""lfr in-process development diagnostics (protocol 10) on the CALIB-SEARCH diagnostic set (paper-hash half A, first
0.1M tokens per domain, cached forecasts). Never touches calib_sel, val or test."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .fsp_lib import Split
from .lfr_lib import CACHE_DIR, ForecastCache
from .lfr_eval import eval_rows, evaluate, exact_delta, first_order_delta, inputs, paper_half, spearman, topk_overlap

DIAG_TOK_PER_DOM = 100_000
_SETS = {}


def diag_set():
    if "d" not in _SETS:
        sp = Split("calib_search")
        rows = eval_rows(sp, DIAG_TOK_PER_DOM, paper_half(sp, 0))
        _SETS["d"] = (sp, rows)
    return _SETS["d"]


def diag_docs():
    sp, rows = diag_set()
    return sorted({d for r in rows for d in r})


@torch.no_grad()
def _topi(model, rows, sp, device, cache, lam):
    outs = []
    for i in range(0, len(rows), 16):
        rr = rows[i:i + 16]
        _, bt, mi = inputs(rr, sp, device, cache, lam)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(mi, {"teach": lam is not None, "collect": True})
        vm = bt["doc_ids"] >= 0
        outs.append(torch.stack([r["topi"][vm] for r in out["records"]], 1).cpu())
    return torch.cat(outs)


def _probe(Xtr, ytr, Xte, yte, device, steps=300):
    Xtr, Xte = torch.as_tensor(Xtr, device=device).float(), torch.as_tensor(Xte, device=device).float()
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    ytr, yte = torch.as_tensor(ytr, device=device).long(), torch.as_tensor(yte, device=device).long()
    W = torch.zeros(Xtr.shape[1], 5, device=device, requires_grad=True)
    b = torch.zeros(5, device=device, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=0.05)
    with torch.enable_grad():
        for _ in range(steps):
            opt.zero_grad()
            F.cross_entropy(Xtr @ W + b, ytr).backward()
            opt.step()
    return float(((Xte @ W + b).argmax(-1) == yte).float().mean())


@torch.no_grad()
def _probe_data(model, rows, sp, device, cache, max_tok=60_000):
    Z, R, Y, G = [], [], [], []
    for i in range(0, len(rows), 16):
        rr = rows[i:i + 16]
        _, bt, mi = inputs(rr, sp, device, cache, 0.9)
        vm = (bt["doc_ids"] >= 0).reshape(-1)
        z, _ = model.aggregate(mi["fc"])
        fc = mi["fc"]
        raw = torch.cat([fc["dfeat"].reshape(*fc["dfeat"].shape[:2], -1), fc["afeat"]], -1).reshape(-1, 74)
        Z.append(z[vm].cpu())
        R.append(raw[vm].cpu())
        Y.append(bt["true_domain"].reshape(-1)[vm].cpu())
        G.append(bt["gdoc"].reshape(-1)[vm].cpu())
    Z, R, Y, G = torch.cat(Z), torch.cat(R), torch.cat(Y), torch.cat(G)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(Y))[:max_tok]
    Z, R, Y, G = Z[idx], R[idx], Y[idx], G[idx]
    tr = (G % 2 == 0)
    return Z, R, Y, tr


def _cf_quality(model, rows, sp, device, cache, n_pairs=16, seed=0):
    """Exact vs first-order vs teacher-predicted utility on n_pairs (position, layer) pairs of the diag set."""
    rng = np.random.default_rng(seed)
    res = []
    model.eval()
    for k in range(n_pairs):
        rr = [rows[int(rng.integers(len(rows)))]]
        a, bt, mi = inputs(rr, sp, device, cache if model.cfg.teacher else None, 0.9 if model.cfg.teacher else None)
        T = bt["input_ids"].shape[1]
        L = int((bt["doc_ids"][0] >= 0).sum())
        tpos = int(rng.integers(1, max(2, L - 1)))
        li = int(rng.integers(model.n_moe))
        store = {}
        mi["cf_idx"] = torch.tensor([tpos], device=device)
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = model(mi, {"teach": model.cfg.teacher, "cf_store": store, "tap": True})
                lg = model.lm_logits(out["hidden"]).float()
            loss = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), bt["targets"].reshape(-1), ignore_index=-100,
                                   reduction="sum")
            loss.backward()
        model.zero_grad(set_to_none=True)
        rec, gr = store[f"rec{li}"], store[f"grad{li}"]
        fo = first_order_delta(model.blocks[li + 1].ffn.experts, rec["x"], gr, rec["r4"], rec["g4"], 1.0)[0]
        row = {k_: v for k_, v in mi.items() if k_ != "cf_idx"}
        row["_targets"] = bt["targets"]
        ex = exact_delta(model, row, li, tpos, rec["x"][0], int(rec["r4"][0]), float(rec["g4"][0]), device,
                         expect_topi=rec["topi"][0].tolist())
        ex, fo = ex.cpu().numpy(), fo.cpu().numpy()
        r = {"layer": li, "rho_fo_exact": spearman(fo, ex), "top4_fo_exact": topk_overlap(fo, ex),
             "route_ok": bool(getattr(exact_delta, "route_ok", True))}
        if "u" in rec:
            r["rho_teacher_exact"] = spearman(rec["u"][0].cpu().numpy(), ex)
            r["top4_teacher_exact"] = topk_overlap(rec["u"][0].cpu().numpy(), ex)
        res.append(r)
    return res


def inprocess_diag(model, cfg, device, name, schedule):
    sp, rows = diag_set()
    teach = model.cfg.teacher and hasattr(model, "tc_trunk")
    cache = ForecastCache(CACHE_DIR, "calib_search") if teach else None
    out = {"name": name}
    st = evaluate(model, sp, rows, device, routing=True, per_token=True)
    from .lfr_audit import joint_counts
    pt = st.pop("per_token")
    C = joint_counts(pt["topi"], pt["dom"].astype(np.int64), 5, model.cfg.n_experts)
    out["profiles"] = (C / np.maximum(1, C.sum(1, keepdims=True))).round(5).tolist()
    out["ce_student"] = st["ce"]
    out["ce_dom_student"] = st["ce_dom"]
    out["load_min"] = float(np.min(st["load"]))
    out["load_max"] = float(np.max(st["load"]))
    out["ent"] = st["ent"]
    if teach:
        out["ce_teacher"] = evaluate(model, sp, rows, device, cache=cache, lam=0.9, routing=False)["ce"]
        out["ce_teacher_shuffled"] = evaluate(model, sp, rows, device, cache=cache, lam=0.9, shuffled=True,
                                              routing=False)["ce"]
        tt = _topi(model, rows[:64], sp, device, cache, 1.0)
        ts = _topi(model, rows[:64], sp, device, None, None)
        ov = []
        for li in range(model.n_moe):
            a_, b_ = tt[:, li], ts[:, li]
            ov.append(float((a_.unsqueeze(-1) == b_.unsqueeze(-2)).any(-1).float().sum(-1).mean()))
        out["top4_overlap_teacher_student"] = ov
        out["top4_overlap_mean"] = float(np.mean(ov))
        Z, R, Y, tr = _probe_data(model, rows, sp, device, cache)
        out["probe_z"] = _probe(Z[tr], Y[tr], Z[~tr], Y[~tr], device)
        out["probe_raw"] = _probe(R[tr], Y[tr], R[~tr], Y[~tr], device)
        out["probe_ratio"] = out["probe_z"] / max(1e-9, out["probe_raw"])
    if cfg.get("cond", {}).get("cf") or name.startswith("cfq") or teach:
        q = _cf_quality(model, rows, sp, device, cache if teach else None)
        out["cf_pairs"] = q
        out["rho_fo_exact_mean"] = float(np.mean([r["rho_fo_exact"] for r in q]))
        if teach:
            out["rho_teacher_exact_mean"] = float(np.mean([r["rho_teacher_exact"] for r in q]))
    model.train()
    return out
