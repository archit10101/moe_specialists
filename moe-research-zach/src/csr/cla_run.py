"""csr-conditional-local-attention-v1: paths, calib partition, split-access locks, config checks, batch sources and
evaluation with attention / gate diagnostics.

Evaluation sets
  calib_search    : 70% of the existing calib documents (salted document-hash partition) - development only.
  calib_selection : the other 30% - only after the search froze (search_frozen.json lists the checkpoint's config
                    hash); one evaluation per checkpoint.
  val             : existing validation split - only after the final runs are complete and frozen (final_frozen.json
                    lists the config hash and final_trained.json exists); one evaluation per checkpoint.
  test            : always refused before any file access.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import guard
from .cla_model import EXPERIMENT, AttnPlan
from .data import TokenSplit, to_torch, train_order
from .train_lib import BatchSource, plan_steps

ART = guard.ROOT / "artifacts" / "conditional-local-attention-v1"
CFG_ROOT = guard.ROOT / "configs" / "conditional-local-attention-v1"
ST_DIR = guard.ROOT / ".tmp" / "conditional_local_attention_v1"
PREREG = guard.ROOT / "docs" / "CONDITIONAL_LOCAL_ATTENTION_V1_PREREG.md"
PARTITION = ART / "search_partition.json"
SEARCH_FROZEN = ART / "search_frozen.json"
FINAL_FROZEN = ART / "final_frozen.json"
FINAL_TRAINED = ART / "final_trained.json"
EVAL_REGISTRY = ART / "eval_registry"
DEVIATIONS = ART / "deviations.jsonl"
PARTITION_SALT = "csr-clav1-calib-partition"
SEARCH_FRACTION = 0.70
DISK_HARD_MIN = 40 * 1024**3
DATA_DIR = "artifacts/data/fwe-rep-v1"
POS_BUCKETS = [(0, 128), (128, 256), (256, 512), (512, 768), (768, 1024)]


class SplitLocked(RuntimeError):
    pass


def prereg_sha():
    return guard.sha256_file(PREREG)


def deviation(kind: str, detail: str, **extra):
    """Append-only deviation ledger."""
    ART.mkdir(parents=True, exist_ok=True)
    with open(DEVIATIONS, "a") as f:
        f.write(json.dumps({"time": guard.now(), "kind": kind, "detail": detail, **extra}) + "\n")
    guard.ledger("CLA_V1_DEVIATION", f"{kind}: {detail[:300]}")


# ------------------------------------------------------------------ partition
def partition_u(doc_id: str) -> float:
    return int.from_bytes(hashlib.sha256(f"{PARTITION_SALT}|{doc_id}".encode()).digest()[:8], "little") / 2**64


def calib_partition(doc_ids):
    search, selection = [], []
    for i, d in enumerate(doc_ids):
        (search if partition_u(d) < SEARCH_FRACTION else selection).append(i)
    return search, selection


def load_raw_split(data_dir, split):
    """The only data entry point of this experiment. The locked test split is refused before any file access."""
    if str(split) not in ("train", "calib", "val"):
        raise SplitLocked(f"split {split!r} refused (locked test split / unknown split)")
    return TokenSplit.load(Path(data_dir), split)


def load_partition():
    p = json.loads(PARTITION.read_text())
    for k in ("search", "selection"):
        if hashlib.sha256(json.dumps(p[f"{k}_doc_indices"]).encode()).hexdigest() != p[f"{k}_indices_sha256"]:
            raise guard.GuardError("partition index hash mismatch")
    return p


# ------------------------------------------------------------------ access locks
def _frozen(path):
    return json.loads(path.read_text()) if path.exists() else None


def check_eval_access(evalset, cfg_sha256=None):
    if evalset == "calib_search":
        return True
    if evalset == "calib_selection":
        fz = _frozen(SEARCH_FROZEN)
        if fz is None:
            raise SplitLocked("calib_selection locked: search not frozen (no search_frozen.json)")
        if cfg_sha256 not in set(fz.get("config_sha256s", [])):
            raise SplitLocked("calib_selection: checkpoint config is not in the frozen search set")
        return True
    if evalset == "val":
        fz = _frozen(FINAL_FROZEN)
        if fz is None:
            raise SplitLocked("val locked: final configurations are not frozen (no final_frozen.json)")
        if cfg_sha256 not in set(fz.get("config_sha256s", [])):
            raise SplitLocked("val: checkpoint config is not in the frozen final set")
        tr = _frozen(FINAL_TRAINED)
        if tr is None or len(tr.get("checkpoints", {})) != len(fz.get("config_sha256s", [])):
            raise SplitLocked("val locked: not all final runs are complete and frozen (final_trained.json)")
        return True
    raise SplitLocked(f"evaluation set {evalset!r} refused")


def claim_eval(evalset, ckpt_sha):
    """One evaluation per (checkpoint, evaluation set). A claim whose evaluation produced no result can be retried."""
    EVAL_REGISTRY.mkdir(parents=True, exist_ok=True)
    p = EVAL_REGISTRY / f"{evalset}__{ckpt_sha}.json"
    if p.exists():
        prev = json.loads(p.read_text())
        if prev.get("status") == "COMPLETE":
            raise SplitLocked(f"{evalset} already evaluated for checkpoint {ckpt_sha[:12]} ({prev.get('result')})")
    guard.atomic_write_json(p, {"status": "CLAIMED", "time": guard.now(), "pid": os.getpid()}, overwrite=True)
    return p


def eval_source(cfg, evalset, data_dir):
    if evalset in ("calib_search", "calib_selection"):
        sp = load_raw_split(data_dir, "calib")
        part = load_partition()
        if part["calib_docs_sha256"] != hashlib.sha256(json.dumps(sp.doc_ids).encode()).hexdigest():
            raise guard.GuardError("calib documents changed since partition")
        order = part["search_doc_indices" if evalset == "calib_search" else "selection_doc_indices"]
    elif evalset == "val":
        sp = load_raw_split(data_dir, "val")
        order = None
    else:
        raise SplitLocked(evalset)
    return sp, BatchSource(sp, cfg["data"]["seq_len"], None, order)


# ------------------------------------------------------------------ configs and plans
def check_cfg(cfg):
    if cfg.get("experiment") != EXPERIMENT:
        raise guard.GuardError("not a conditional-local-attention-v1 config")
    if cfg.get("prereg_sha256") != prereg_sha():
        raise guard.GuardError("config prereg hash does not match the frozen preregistration")
    d = guard.inside_root(cfg["data"]["dir"])
    man = json.loads((d / "manifest.json").read_text())
    if guard.sha256_json(man["splits"]) != cfg["data"]["splits_sha256"]:
        raise guard.GuardError("data manifest hash mismatch")
    AttnPlan.from_cfg(cfg["attention"])
    return d


def train_source(cfg, d):
    tr = load_raw_split(d, "train")
    order = train_order(tr.n_docs, cfg["data"]["train_order_seed"])
    src = BatchSource(tr, cfg["data"]["seq_len"], None, order)
    t = cfg["train"]
    steps, total = plan_steps(src, t["micro_batch_rows"] * t["accum"], t["token_budget"])
    return tr, src, steps, total


def require_disk(extra_bytes: int, what: str):
    free = guard.disk_free_bytes()
    if free - extra_bytes < DISK_HARD_MIN:
        raise guard.GuardError(f"disk: free={free} need={extra_bytes} for {what}; hard minimum {DISK_HARD_MIN}")
    guard.ledger("CLA_V1_DISK_CHECK", f"{what}: free={free} planned={extra_bytes} min={DISK_HARD_MIN} ok")


def param_sha(tensors: dict):
    h = hashlib.sha256()
    for n, p in sorted(tensors.items()):
        h.update(n.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def model_param_sha(model):
    return param_sha(dict(model.named_parameters()))


def code_hashes():
    files = sorted([p for p in guard.ROOT.glob("cla_*.py")] + [p for p in (guard.ROOT / "src" / "csr").glob("cla_*.py")]
                   + [guard.ROOT / "src/csr/model.py", guard.ROOT / "src/csr/train_lib.py",
                      guard.ROOT / "src/csr/data.py", guard.ROOT / "src/csr/guard.py"])
    return {str(p.relative_to(guard.ROOT)): guard.sha256_file(p) for p in files}


FROZEN_CODE = ("src/csr/cla_model.py", "src/csr/cla_attn.py", "src/csr/cla_run.py", "cla_train.py",
               "cla_evaluate.py", "src/csr/model.py", "src/csr/train_lib.py", "src/csr/data.py")


# ------------------------------------------------------------------ evaluation with diagnostics
class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


@torch.no_grad()
def cla_evaluate(model, source: BatchSource, device, rows_per_batch=16, max_rows=None, autocast=True,
                 diagnostics=True, token_stats=False):
    """Token-weighted CE (nats per valid prediction token), per-document NLL sums, routing / attention / gate
    diagnostics. Gate decisions use the frozen thresholds (model.eval())."""
    was_training = model.training
    model.eval()
    rows_all = source.rows if max_rows is None else source.rows[:max_rows]
    cfg = model.cfg
    E, n_moe, L = cfg.n_experts, len(cfg.moe_layers), cfg.n_layer
    usage = np.zeros((n_moe, E))
    nb = len(POS_BUCKETS)
    sel_pos = np.zeros((L, nb))
    elig_pos = np.zeros((L, nb))
    tok_pos = np.zeros(nb)
    keys = np.zeros(L)  # keys attended (theoretical attention work) summed over non-pad tokens
    keys_full_ref = 0.0
    n_tok = 0
    doc_nll, doc_n = {}, {}
    tot_nll, tot_n = 0.0, 0
    pos_nll = np.zeros(nb)
    pos_n = np.zeros(nb)
    tokid_sel = {}  # token id -> [n occurrences as eligible query (summed over layers), n selected]
    prev_sel = {}
    zstats = [[] for _ in range(L)]
    modes = model.modes
    t0 = time.time()
    if device.type == "cuda":
        torch.cuda.synchronize()
    for s in range(0, len(rows_all), rows_per_batch):
        tb = to_torch(source.batch(rows_all[s:s + rows_per_batch]), device)
        ctx = torch.autocast(device.type, dtype=torch.bfloat16) if autocast else _null()
        with ctx:
            out = model(tb)
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
        g = tb["gdoc"][valid].cpu().numpy()
        ug, inv = np.unique(g, return_inverse=True)
        sums = np.bincount(inv, weights=v.cpu().numpy())
        cnts = np.bincount(inv)
        for dd, sv, cv in zip(ug, sums, cnts):
            doc_nll[int(dd)] = doc_nll.get(int(dd), 0.0) + float(sv)
            doc_n[int(dd)] = doc_n.get(int(dd), 0) + int(cv)
        if not diagnostics:
            continue
        pos = tb["positions"]
        nonpad = tb["doc_ids"] >= 0
        n_tok += int(nonpad.sum())
        for bi, (a, b) in enumerate(POS_BUCKETS):
            m = valid & (pos >= a) & (pos < b)
            pos_nll[bi] += float(nll[m].double().sum())
            pos_n[bi] += int(m.sum())
            tok_pos[bi] += int((nonpad & (pos >= a) & (pos < b)).sum())
        ctx_full = (pos + 1).double()[nonpad]
        keys_full_ref += float(ctx_full.sum())
        gi = iter(out.gate)
        for li, (mode, W) in enumerate(modes):
            if mode == "full":
                keys[li] += float(ctx_full.sum())
            elif mode == "local":
                keys[li] += float(ctx_full.clamp(max=W).sum())
            else:
                gd = next(gi)
                sel, elig = gd["sel"], gd["elig"]
                keys[li] += float(ctx_full.clamp(max=W).sum()) + float((pos + 1).double()[sel].sum())
                for bi, (a, b) in enumerate(POS_BUCKETS):
                    m = (pos >= a) & (pos < b)
                    sel_pos[li, bi] += int((sel & m).sum())
                    elig_pos[li, bi] += int((elig & m).sum())
                ze = gd["z"][elig]
                if ze.numel():
                    zstats[li].append(torch.stack([ze.mean(), ze.std() if ze.numel() > 1 else ze.mean() * 0,
                                                   ze.min(), ze.max()]).cpu())
                if token_stats:
                    idsn = tb["input_ids"][elig].cpu().numpy()
                    sn = sel[elig].cpu().numpy()
                    prev = torch.zeros_like(tb["input_ids"])
                    prev[:, 1:] = tb["input_ids"][:, :-1]
                    pn = prev[elig].cpu().numpy()
                    for arr, dct in ((idsn, tokid_sel), (pn, prev_sel)):
                        u, iv = np.unique(arr, return_inverse=True)
                        ce = np.bincount(iv)
                        cs = np.bincount(iv, weights=sn.astype(np.float64))
                        for tid, a_, b_ in zip(u, ce, cs):
                            r = dct.setdefault(int(tid), [0, 0])
                            r[0] += int(a_)
                            r[1] += int(b_)
        for li, r in enumerate(out.routing):
            usage[li] += np.bincount(r["topi"][nonpad].reshape(-1).cpu().numpy(), minlength=E)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    if was_training:
        model.train()
    res = {"ce_nats_per_token": tot_nll / max(tot_n, 1), "n_valid_tokens": tot_n,
           "ppl": math.exp(tot_nll / max(tot_n, 1)), "doc_nll": doc_nll, "doc_n": doc_n, "n_rows": len(rows_all),
           "eval_seconds": elapsed, "eval_valid_tokens_per_s": tot_n / max(elapsed, 1e-9),
           "thresholds": model.thr.tolist(), "plan": model.plan.to_dict()}
    if diagnostics:
        tok = max(n_tok, 1)
        frac = usage / usage.sum(1, keepdims=True).clip(min=1)
        cond = [li for li, (m, _) in enumerate(modes) if m == "cond"]
        res.update({
            "expert_max_share": frac.max(1).tolist(), "experts_unused": (usage == 0).sum(1).tolist(),
            "expert_load_cv": (frac.std(1) / frac.mean(1).clip(min=1e-12)).tolist(),
            "ce_by_doc_position": {f"{a}-{b}": (pos_nll[i] / pos_n[i] if pos_n[i] else None)
                                   for i, (a, b) in enumerate(POS_BUCKETS)},
            "tokens": n_tok, "tokens_by_position": tok_pos.tolist(),
            "keys_attended_per_token_by_layer": (keys / tok).tolist(),
            "keys_attended_per_token_full_reference": keys_full_ref / tok,
            "attention_work_fraction_vs_full": float(keys.sum() / max(keys_full_ref * L, 1)),
            "layer_modes": [list(m) for m in modes],
        })
        if cond:
            res.update({
                "global_query_rate_by_layer": [float(sel_pos[li].sum() / tok) if li in cond else None
                                               for li in range(L)],
                "global_query_rate_mean": float(np.mean([sel_pos[li].sum() / tok for li in cond])),
                "eligible_fraction": float(np.mean([elig_pos[li].sum() / tok for li in cond])),
                "global_rate_among_eligible_by_layer": [float(sel_pos[li].sum() / max(elig_pos[li].sum(), 1))
                                                        if li in cond else None for li in range(L)],
                "global_query_rate_by_layer_and_position": {
                    f"{a}-{b}": [float(sel_pos[li, i] / max(tok_pos[i], 1)) if li in cond else None
                                 for li in range(L)] for i, (a, b) in enumerate(POS_BUCKETS)},
                "gate_logit_stats_by_layer": [torch.stack(zs).mean(0).tolist() if zs else None for zs in zstats],
            })
            if token_stats:
                res["token_request_counts"] = {str(k): v for k, v in tokid_sel.items()}
                res["prev_token_request_counts"] = {str(k): v for k, v in prev_sel.items()}
    return res


def collapse_check(ev, max_share=0.5):
    """Existing project criterion: any MoE layer with an unused routed expert or one expert taking > 50% of slots."""
    unused = max(ev["experts_unused"])
    ms = max(ev["expert_max_share"])
    return {"collapsed": bool(unused > 0 or ms > max_share), "max_unused": int(unused), "max_share": float(ms)}
