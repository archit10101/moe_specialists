"""Shared run library for csr-value-routed-compute-v1 (vrc) and csr-m2d2-domain-prior-v1 (m2d2): paths, deadlines,
split locks, batch sources, model building, evaluation. New module; existing modules are imported, never modified."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
import torch.nn.functional as F

from . import guard
from .data import TokenSplit, make_batch, pack_rows, to_torch
from .train_lib import valid_tokens

NY = ZoneInfo("America/New_York")
DAY = _dt.date(2026, 11, 9)
LAUNCH_DEADLINE = _dt.datetime(2026, 11, 9, 8, 35, tzinfo=NY).timestamp()
STOP_DEADLINE = _dt.datetime(2026, 11, 9, 8, 45, tzinfo=NY).timestamp()
REPORT_DEADLINE = _dt.datetime(2026, 11, 9, 8, 55, tzinfo=NY).timestamp()

EXP = {
    "vrc": {"id": "csr-value-routed-compute-v1", "art": guard.ROOT / "artifacts/value-routed-compute-v1",
            "cfg": guard.ROOT / "configs/value-routed-compute-v1", "st": guard.ROOT / ".tmp/value-routed-compute-v1",
            "protocol": guard.ROOT / "docs/VALUE_ROUTED_COMPUTE_V1_PROTOCOL.md"},
    "m2d2": {"id": "csr-m2d2-domain-prior-v1", "art": guard.ROOT / "artifacts/m2d2-domain-prior-v1",
             "cfg": guard.ROOT / "configs/m2d2-domain-prior-v1", "st": guard.ROOT / ".tmp/m2d2-domain-prior-v1",
             "protocol": guard.ROOT / "docs/M2D2_DOMAIN_PRIOR_V1_PROTOCOL.md"},
}
DISK_HARD_MIN = 40 * 1024**3


class SplitLocked(RuntimeError):
    pass


def exp_key(cfg):
    for k, v in EXP.items():
        if cfg.get("experiment") == v["id"]:
            return k
    raise guard.GuardError(f"unknown experiment {cfg.get('experiment')}")


def deviation(key, kind, detail, **extra):
    a = EXP[key]["art"]
    a.mkdir(parents=True, exist_ok=True)
    with open(a / "deviations.jsonl", "a") as f:
        f.write(json.dumps({"time": guard.now(), "kind": kind, "detail": detail, **extra}) + "\n")
    guard.ledger(f"{key.upper()}_DEVIATION", f"{kind}: {detail[:300]}")


def require_disk(extra, what):
    free = guard.disk_free_bytes()
    if free - extra < DISK_HARD_MIN:
        raise guard.GuardError(f"disk: free={free} need={extra} for {what}; hard minimum {DISK_HARD_MIN}")


def check_cfg(cfg):
    key = exp_key(cfg)
    if cfg.get("protocol_sha256") != guard.sha256_file(EXP[key]["protocol"]):
        raise guard.GuardError("config protocol hash does not match the frozen protocol")
    d = guard.inside_root(cfg["data"]["dir"])
    man = json.loads((d / "manifest.json").read_text())
    if man["splits_sha256"] != cfg["data"]["splits_sha256"]:
        raise guard.GuardError("data manifest hash mismatch")
    return key, d


# ------------------------------------------------------------------ split access
ALLOWED = {"vrc": ("train", "calib"), "m2d2": ("train", "calib_search", "val")}


def load_split(key, d, split):
    if split not in ALLOWED[key]:
        raise SplitLocked(f"{key}: split {split!r} refused (locked / not used)")
    if key == "m2d2" and split == "val":
        fz = EXP["m2d2"]["art"] / "frozen.json"
        if not fz.exists():
            raise SplitLocked("m2d2 validation locked until frozen.json exists")
    sp = TokenSplit.load(Path(d), split)
    doms = None
    if key == "m2d2":
        doms = np.asarray(json.loads((Path(d) / f"{split}_docs.json").read_text())["domains"], dtype=np.int64)
    return sp, doms


class Source:
    """Packed rows over a TokenSplit; optional per-document domain labels (m2d2)."""

    def __init__(self, split, T, rows, domains=None):
        self.split, self.T, self.rows, self.domains = split, T, rows, domains

    def batch(self, rows):
        b = make_batch(rows, self.split, self.T)
        if self.domains is not None:
            g = b["gdoc"]
            b["domain"] = np.where(g >= 0, self.domains[np.clip(g, 0, None)], -1)
        return b


def m2d2_train_rows(sp, doms, T, seed):
    rows = []
    L = sp.lengths()
    for d in range(int(doms.max()) + 1):
        idx = np.nonzero(doms == d)[0]
        order = idx[np.random.default_rng([seed, d]).permutation(len(idx))].tolist()
        rows += pack_rows(L, order, T)
    perm = np.random.default_rng(seed).permutation(len(rows))
    return [rows[i] for i in perm]


def m2d2_eval_rows(sp, doms, T):
    rows = []
    for d in range(int(doms.max()) + 1):
        rows += pack_rows(sp.lengths(), np.nonzero(doms == d)[0].tolist(), T)
    return rows


def shuffled_labels(doms, seed):
    return doms[np.random.default_rng(seed).permutation(len(doms))]


def train_source(cfg, key, d):
    T = cfg["data"]["seq_len"]
    sp, doms = load_split(key, d, "train")
    if key == "vrc":
        rows = pack_rows(sp.lengths(), list(range(sp.n_docs)), T)  # split stored in train order
        src = Source(sp, T, rows)
    else:
        rows = m2d2_train_rows(sp, doms, T, cfg["data"]["train_order_seed"])
        lab = doms
        if cfg["domain"].get("shuffle_train_labels"):
            lab = shuffled_labels(doms, cfg["domain"]["shuffle_seed"])
        src = Source(sp, T, rows, lab)
    t = cfg["train"]
    rps = t["micro_batch_rows"] * t["accum"]
    steps, total = [], 0
    for s in range(0, len(src.rows) - rps + 1, rps):
        if total >= t["token_budget"]:
            break
        r = src.rows[s:s + rps]
        steps.append(r)
        total += valid_tokens(r, sp)
    if total < t["token_budget"]:
        raise ValueError(f"not enough training data: {total} < {t['token_budget']}")
    return sp, src, steps, total


def vrc_partition(d):
    p = json.loads((Path(d) / "calib_partition.json").read_text())
    for k in ("search", "selection"):
        if hashlib.sha256(json.dumps(p[f"{k}_doc_indices"]).encode()).hexdigest() != p[f"{k}_indices_sha256"]:
            raise guard.GuardError("partition hash mismatch")
    return p


def eval_source(cfg, key, d, evalset):
    T = cfg["data"]["seq_len"]
    if key == "vrc":
        if evalset not in ("calib_search", "calib_selection"):
            raise SplitLocked(evalset)
        if evalset == "calib_selection" and not (EXP["vrc"]["art"] / "thresholds_frozen.json").exists():
            raise SplitLocked("vrc CALIB-SELECTION locked until thresholds are frozen")
        sp, _ = load_split(key, d, "calib")
        p = vrc_partition(d)
        order = p["search_doc_indices" if evalset == "calib_search" else "selection_doc_indices"]
        return Source(sp, T, pack_rows(sp.lengths(), order, T))
    split = {"calib_search": "calib_search", "val": "val"}[evalset]
    sp, doms = load_split(key, d, split)
    return Source(sp, T, m2d2_eval_rows(sp, doms, T), doms)


# ------------------------------------------------------------------ models
def build_model(cfg, device, backend=None):
    key = exp_key(cfg)
    if key == "vrc":
        from .vrc_model import build_vrc_model
        return build_vrc_model(cfg, device, backend)
    from .m2d2_model import build_domain_model
    return build_domain_model(cfg, device)


def param_sha(named):
    h = hashlib.sha256()
    for n, p in sorted(named):
        h.update(n.encode())
        h.update(p.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def code_hashes():
    files = sorted(list(guard.ROOT.glob("vrc_*.py")) + list(guard.ROOT.glob("m2d2_*.py")) +
                   list(guard.ROOT.glob("vm_*.py")) + list((guard.ROOT / "src/csr").glob("vrc_*.py")) +
                   list((guard.ROOT / "src/csr").glob("m2d2_*.py")) + list((guard.ROOT / "src/csr").glob("vm_*.py")) +
                   [guard.ROOT / f"src/csr/{x}.py" for x in ("model", "data", "train_lib", "guard", "cla_attn")])
    return {str(p.relative_to(guard.ROOT)): guard.sha256_file(p) for p in files}


# ------------------------------------------------------------------ evaluation
def _amp(device, on):
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=bool(on and device.type == "cuda"))


@torch.no_grad()
def basic_eval(model, source, device, rows_per_batch=16, max_rows=None, autocast=True, fwd_kwargs=None):
    """Token-weighted CE + per-document NLL sums + expert usage (for collapse checks)."""
    was = model.training
    model.eval()
    rows_all = source.rows if max_rows is None else source.rows[:max_rows]
    E, n_moe = model.cfg.n_experts, len(model.cfg.moe_layers)
    usage = np.zeros((n_moe, E))
    doc_nll, doc_n = {}, {}
    tot, n = 0.0, 0
    t0 = time.time()
    for s in range(0, len(rows_all), rows_per_batch):
        tb = to_torch(source.batch(rows_all[s:s + rows_per_batch]), device)
        with _amp(device, autocast):
            out = model(tb, **(fwd_kwargs or {}))
        tgt = tb["targets"]
        valid = tgt != -100
        nll = F.cross_entropy(out.logits.float().view(-1, out.logits.shape[-1]), tgt.view(-1), ignore_index=-100,
                              reduction="none").view(tgt.shape)
        if not torch.isfinite(nll[valid]).all():
            raise FloatingPointError("non-finite evaluation NLL")
        v = nll[valid].double()
        tot += float(v.sum())
        n += int(valid.sum())
        g = tb["gdoc"][valid].cpu().numpy()
        ug, inv = np.unique(g, return_inverse=True)
        for dd, sv, cv in zip(ug, np.bincount(inv, weights=v.cpu().numpy()), np.bincount(inv)):
            doc_nll[int(dd)] = doc_nll.get(int(dd), 0.0) + float(sv)
            doc_n[int(dd)] = doc_n.get(int(dd), 0) + int(cv)
        nonpad = tb["doc_ids"] >= 0
        for li, r in enumerate(out.routing):
            usage[li] += np.bincount(r["topi"][nonpad].reshape(-1).cpu().numpy(), minlength=E)
    if was:
        model.train()
    frac = usage / usage.sum(1, keepdims=True).clip(min=1)
    return {"ce_nats_per_token": tot / max(n, 1), "n_valid_tokens": n, "doc_nll": doc_nll, "doc_n": doc_n,
            "experts_unused": (usage == 0).sum(1).tolist(), "expert_max_share": frac.max(1).tolist(),
            "eval_seconds": time.time() - t0, "n_rows": len(rows_all)}


def collapse_check(ev, max_share=0.5):
    unused = max(ev["experts_unused"])
    ms = max(ev["expert_max_share"])
    return {"collapsed": bool(unused > 0 or ms > max_share), "max_unused": int(unused), "max_share": float(ms)}


def ce_math(x):
    return float(x) if x is not None and math.isfinite(x) else None
