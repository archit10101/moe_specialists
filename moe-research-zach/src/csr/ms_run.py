"""csr-multiscale-routing-v1 training/evaluation helpers (new; reuses train_lib batching, optimizer and schedule)."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import guard
from .data import to_torch, train_order
from .multiscale import EXPERIMENT, MSRouting, MultiscaleModel, load_split, load_starts
from .model import ModelConfig
from .train_lib import BatchSource, plan_steps, valid_tokens

DISK_HARD_MIN = 80 * 1024**3
DISK_WARN = 100 * 1024**3
PREREG = guard.ROOT / "docs" / "MULTISCALE_ROUTING_V1_PREREG.md"


def prereg_sha():
    return guard.sha256_file(PREREG)


def require_disk_ms(extra_bytes: int, what: str):
    free = guard.disk_free_bytes()
    if free - extra_bytes < DISK_HARD_MIN:
        raise guard.GuardError(f"disk: free={free} need={extra_bytes} for {what}; hard minimum {DISK_HARD_MIN}")
    guard.ledger("MS_V1_DISK_CHECK", f"{what}: free={free} planned={extra_bytes} min={DISK_HARD_MIN}"
                                     f"{' WARN<100GiB' if free - extra_bytes < DISK_WARN else ''} ok")


def check_cfg(cfg):
    if cfg.get("experiment") != EXPERIMENT:
        raise guard.GuardError("not a multiscale-routing-v1 config")
    if cfg.get("prereg_sha256") != prereg_sha():
        raise guard.GuardError("config prereg hash does not match the frozen preregistration")
    d = guard.inside_root(cfg["data"]["dir"])
    man = json.loads((d / "manifest.json").read_text())
    if guard.sha256_json(man["splits"]) != cfg["data"]["splits_sha256"]:
        raise guard.GuardError("data manifest hash mismatch")
    return d


def build_ms_model(cfg, device):
    torch.manual_seed(cfg["init"]["seed"])
    m = MultiscaleModel(ModelConfig.from_dict(cfg["model"]))
    return m.to(device=device, dtype=torch.float32)


def make_sources(cfg, d, split, order=None):
    sp = load_split(d, split)
    seg = load_starts(cfg, split, guard.ROOT)
    return sp, BatchSource(sp, cfg["data"]["seq_len"], seg, order)


def train_plan(cfg, d):
    tr = load_split(d, "train")
    plain = BatchSource(tr, cfg["data"]["seq_len"], None, train_order(tr.n_docs, cfg["data"]["train_order_seed"]))
    t = cfg["train"]
    steps, total = plan_steps(plain, t["micro_batch_rows"] * t["accum"], t["token_budget"])
    return tr, steps, total


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


@torch.no_grad()
def ms_evaluate(model, source, routing: MSRouting, device, rows_per_batch=16, max_rows=None, autocast=True,
                per_token=False):
    """Token-weighted CE (nats per valid prediction token), per-document NLL sums, routing usage per layer.
    per_token: also return float16 NLL of every valid token in canonical (row-major packed) order."""
    model.eval()
    rows_all = source.rows if max_rows is None else source.rows[:max_rows]
    E = model.cfg.n_experts
    n_moe = len(model.cfg.moe_layers)
    usage = np.zeros((n_moe, E), np.int64)
    ent = np.zeros(n_moe)
    ent_n = 0
    doc_nll, doc_n = {}, {}
    tot_nll, tot_n = 0.0, 0
    toks = []
    for s in range(0, len(rows_all), rows_per_batch):
        tb = to_torch(source.batch(rows_all[s:s + rows_per_batch]), device)
        ctx = torch.autocast(device.type, dtype=torch.bfloat16) if autocast else _null()
        with ctx:
            out = model(tb, routing, collect=False)
        logits = out.logits.float()
        tgt = tb["targets"]
        valid = tgt != -100
        nll = F.cross_entropy(logits.view(-1, logits.shape[-1]), tgt.view(-1), ignore_index=-100,
                              reduction="none").view(tgt.shape)
        if not torch.isfinite(nll[valid]).all():
            raise FloatingPointError("non-finite evaluation NLL")
        v = nll[valid].double()
        tot_nll += float(v.sum())
        tot_n += int(valid.sum())
        if per_token:
            toks.append(nll[valid].to(torch.float16).cpu().numpy())
        g = tb["gdoc"][valid].cpu().numpy()
        ug, inv = np.unique(g, return_inverse=True)
        sums = np.bincount(inv, weights=v.cpu().numpy())
        cnts = np.bincount(inv)
        for dd, sv, cv in zip(ug, sums, cnts):
            doc_nll[int(dd)] = doc_nll.get(int(dd), 0.0) + float(sv)
            doc_n[int(dd)] = doc_n.get(int(dd), 0) + int(cv)
        nonpad = tb["doc_ids"] >= 0
        for li, r in enumerate(out.routing):
            usage[li] += np.bincount(r["topi"][nonpad].reshape(-1).cpu().numpy(), minlength=E)
        ent_n += int(nonpad.sum())
    model.train()
    frac = usage / usage.sum(1, keepdims=True).clip(min=1)
    res = {"ce_nats_per_token": tot_nll / max(tot_n, 1), "n_valid_tokens": tot_n,
           "ppl": math.exp(tot_nll / max(tot_n, 1)), "doc_nll": doc_nll, "doc_n": doc_n,
           "expert_max_share": frac.max(1).tolist(), "experts_unused": (usage == 0).sum(1).tolist(),
           "expert_load_cv": (frac.std(1) / frac.mean(1).clip(min=1e-12)).tolist(),
           "usage_entropy": (-(frac * np.log(frac.clip(min=1e-30))).sum(1)).tolist(), "n_rows": len(rows_all)}
    if per_token:
        res["_per_token_nll_f16"] = np.concatenate(toks) if toks else np.zeros(0, np.float16)
    return res


def router_lm_gradnorm(model, lm_mean):
    rp = [b.ffn.router.weight for b in model.moe_blocks()]
    g = torch.autograd.grad(lm_mean, rp, retain_graph=True, allow_unused=True)
    return [float(x.float().norm()) if x is not None else 0.0 for x in g]
