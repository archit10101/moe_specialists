"""csr-sequence-routing-search-v1: data partition, split-access locks, config checks and evaluation.

Evaluation sets
  calib_search    : 70% of the existing calib documents (stable document-hash partition) - iterative development.
  calib_selection : the other 30% - only after candidate generation ended (search_frozen.json exists and lists the
                    checkpoint's config hash); one evaluation per checkpoint.
  val             : existing validation split - only after final configurations were frozen (final_frozen.json lists
                    the config hash) and training completed; one evaluation per checkpoint.
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
from .data import TokenSplit, to_torch, train_order
from .seqroute import EXPERIMENT, SRRouting, build_sr_model
from .train_lib import BatchSource, plan_steps, valid_tokens

ART = guard.ROOT / "artifacts" / "sequence-routing-search-v1"
CFG_ROOT = guard.ROOT / "configs" / "sequence-routing-search-v1"
PREREG = guard.ROOT / "docs" / "SEQUENCE_ROUTING_SEARCH_V1_PREREG.md"
PARTITION = ART / "search_partition.json"
SEARCH_FROZEN = ART / "search_frozen.json"
FINAL_FROZEN = ART / "final_frozen.json"
EVAL_REGISTRY = ART / "eval_registry"
PARTITION_SALT = "csr-srsv1-calib-partition"
SEARCH_FRACTION = 0.70
DISK_HARD_MIN = 80 * 1024**3
EVALSETS = ("calib_search", "calib_selection", "val")


class SplitLocked(RuntimeError):
    pass


def prereg_sha():
    return guard.sha256_file(PREREG)


# ------------------------------------------------------------------ partition
def partition_u(doc_id: str) -> float:
    return int.from_bytes(hashlib.sha256(f"{PARTITION_SALT}|{doc_id}".encode()).digest()[:8], "little") / 2**64


def calib_partition(doc_ids):
    """Deterministic document-hash partition of the calib documents (depends on document ids only)."""
    search, selection = [], []
    for i, d in enumerate(doc_ids):
        (search if partition_u(d) < SEARCH_FRACTION else selection).append(i)
    return search, selection


def load_raw_split(data_dir, split):
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
def _frozen_hashes(path):
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    return set(d.get("config_sha256s", []))


def check_eval_access(evalset, cfg_sha256=None):
    if evalset == "calib_search":
        return True
    if evalset == "calib_selection":
        fz = _frozen_hashes(SEARCH_FROZEN)
        if fz is None:
            raise SplitLocked("calib_selection locked: candidate generation has not ended (no search_frozen.json)")
        if cfg_sha256 not in fz:
            raise SplitLocked("calib_selection: checkpoint config is not in the frozen search set")
        return True
    if evalset == "val":
        fz = _frozen_hashes(FINAL_FROZEN)
        if fz is None:
            raise SplitLocked("val locked: final configurations are not frozen (no final_frozen.json)")
        if cfg_sha256 not in fz:
            raise SplitLocked("val: checkpoint config is not in the frozen final set")
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
    """BatchSource over the documents of an evaluation set (natural document order within the set)."""
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
        raise guard.GuardError("not a sequence-routing-search-v1 config")
    if cfg.get("prereg_sha256") != prereg_sha():
        raise guard.GuardError("config prereg hash does not match the frozen preregistration")
    d = guard.inside_root(cfg["data"]["dir"])
    man = json.loads((d / "manifest.json").read_text())
    if guard.sha256_json(man["splits"]) != cfg["data"]["splits_sha256"]:
        raise guard.GuardError("data manifest hash mismatch")
    SRRouting.from_cfg(cfg["routing"])
    return d


def train_source(cfg, d):
    tr = load_raw_split(d, "train")
    src = BatchSource(tr, cfg["data"]["seq_len"], None, train_order(tr.n_docs, cfg["data"]["train_order_seed"]))
    t = cfg["train"]
    steps, total = plan_steps(src, t["micro_batch_rows"] * t["accum"], t["token_budget"])
    return tr, src, steps, total


def require_disk_srs(extra_bytes: int, what: str):
    free = guard.disk_free_bytes()
    if free - extra_bytes < DISK_HARD_MIN:
        raise guard.GuardError(f"disk: free={free} need={extra_bytes} for {what}; hard minimum {DISK_HARD_MIN}")
    guard.ledger("SRS_V1_DISK_CHECK", f"{what}: free={free} planned={extra_bytes} min={DISK_HARD_MIN} ok")


def param_sha(state_dict):
    h = hashlib.sha256()
    for n, p in sorted(state_dict.items()):
        h.update(n.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def flops_per_token(mcfg, sr: SRRouting, seq_len=1024, mean_seg_len=None):
    """Analytic forward FLOPs per token (multiply-add = 2). Training = 3x forward. Attention scores use the mean
    causal context (seq_len/2, an upper bound with document masks). Router: per token (token) or per segment."""
    d, E, k, f, fs = mcfg.d_model, mcfg.n_experts, mcfg.top_k, mcfg.expert_ffn, mcfg.shared_ffn
    n_moe = len(mcfg.moe_layers)
    n_dense = mcfg.n_layer - n_moe
    attn = mcfg.n_layer * (2 * 4 * d * d + 2 * 2 * d * (seq_len / 2))
    dense = n_dense * 2 * 3 * d * mcfg.dense_ffn
    experts = n_moe * (k * 2 * 3 * d * f + 2 * 3 * d * fs)
    lm = 2 * d * mcfg.vocab_size
    if sr.mode == "token":
        router = n_moe * 2 * d * E
        extra = 0.0
    else:
        L = mean_seg_len or 64
        router = n_moe * 2 * d * E / L
        extra = n_moe * 2 * d  # pooling
        if sr.summary == "attn":
            extra += n_moe * 4 * d
        if sr.seg == "cp":
            extra += (1 if sr.boundary_source == "layer0" else n_moe) * 8 * d
        if sr.seg == "lbd":
            extra += n_moe * (2 * d + 8 * d)  # detector + training label
        if sr.state == "gru":
            extra += n_moe * (2 * 3 * (2 * E + sr.gru_hidden) * sr.gru_hidden) / L
    fwd = attn + dense + experts + lm + router + extra
    return {"forward": fwd, "train": 3 * fwd, "router_forward": router}


# ------------------------------------------------------------------ evaluation
class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


POS_BUCKETS = [(0, 16), (16, 64), (64, 256), (256, 1024)]
SEGPOS_BUCKETS = [(0, 4), (4, 16), (16, 64), (64, 1024)]


@torch.no_grad()
def sr_evaluate(model, source: BatchSource, device, rows_per_batch=16, max_rows=None, autocast=True, diagnostics=True):
    """Token-weighted CE (nats per valid prediction token), per-document NLL sums and routing diagnostics."""
    model.eval()
    sr = model.sr
    rows_all = source.rows if max_rows is None else source.rows[:max_rows]
    E = model.cfg.n_experts
    K = model.cfg.top_k
    n_moe = len(model.cfg.moe_layers)
    usage = np.zeros((n_moe, E), np.float64)
    ent = np.zeros(n_moe)
    n_tok_routed = 0
    doc_nll, doc_n = {}, {}
    tot_nll, tot_n = 0.0, 0
    pos_nll = np.zeros(len(POS_BUCKETS))
    pos_n = np.zeros(len(POS_BUCKETS))
    segpos_nll = np.zeros(len(SEGPOS_BUCKETS))
    segpos_n = np.zeros(len(SEGPOS_BUCKETS))
    seg_n = np.zeros(n_moe)
    seg_closed_n = np.zeros(n_moe)
    seg_closed_tok = np.zeros(n_moe)
    seg_lens = [[] for _ in range(n_moe)]
    route_changes = np.zeros(n_moe)
    jac_sum = np.zeros(n_moe)
    jac_n = np.zeros(n_moe)
    router_calls = np.zeros(n_moe)
    bos_tokens = np.zeros(n_moe)
    dispatch_max_over_mean = []
    t0 = time.time()
    if device.type == "cuda":
        torch.cuda.synchronize()
    for s in range(0, len(rows_all), rows_per_batch):
        tb = to_torch(source.batch(rows_all[s:s + rows_per_batch]), device)
        ctx = torch.autocast(device.type, dtype=torch.bfloat16) if autocast else _null()
        with ctx:
            out = model(tb, collect=diagnostics)
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
        for bi, (a, b) in enumerate(POS_BUCKETS):
            m = valid & (pos >= a) & (pos < b)
            pos_nll[bi] += float(nll[m].double().sum())
            pos_n[bi] += int(m.sum())
        nonpad = tb["doc_ids"] >= 0
        n_np = int(nonpad.sum())
        n_tok_routed += n_np
        for li, r in enumerate(out.routing):
            topi = r["topi"][nonpad]
            usage[li] += np.bincount(topi.reshape(-1).cpu().numpy(), minlength=E)
            cnt = torch.bincount(r["topi"].reshape(-1), minlength=E).float()
            dispatch_max_over_mean.append(float(cnt.max() / cnt.mean().clamp(min=1e-9)))
            if sr.mode == "token":
                p = r["probs_tok"][nonpad].float()
                ent[li] += float((-(p * p.clamp_min(1e-30).log()).sum(-1)).sum())
                same_next = nonpad[:, 1:] & (tb["doc_ids"][:, 1:] == tb["doc_ids"][:, :-1])
                ti = r["topi"]
                oh = torch.zeros(*ti.shape[:2], E, device=device).scatter_(2, ti, 1.0)
                inter = (oh[:, 1:] * oh[:, :-1]).sum(-1)
                jac = inter / (2 * K - inter)
                route_changes[li] += float(((inter < K) & same_next).sum())
                jac_sum[li] += float(jac[same_next].sum())
                jac_n[li] += int(same_next.sum())
                router_calls[li] += n_np
                continue
            S = r["S"]
            cntS = r["seg_count"].double()
            p = r["probs_seg"].double()
            ent[li] += float(((-(p * p.clamp_min(1e-300).log()).sum(-1)) * cntS).sum())
            j = r["seg_j"]
            seg_n[li] += S
            cl = r["seg_closed"]
            seg_closed_n[li] += int(cl.sum())
            seg_closed_tok[li] += float(cntS[cl].sum())
            seg_lens[li].append(r["seg_count"].cpu().numpy())
            bos_tokens[li] += float(cntS[j == 0].sum())
            later = (j > 0).nonzero().squeeze(1)
            router_calls[li] += int(later.numel())
            if later.numel():
                ts = r["seg_topi"]
                oh = torch.zeros(S, E, device=device).scatter_(1, ts, 1.0)
                inter = (oh[later] * oh[later - 1]).sum(-1)
                route_changes[li] += float((inter < K).sum())
                jac_sum[li] += float((inter / (2 * K - inter)).sum())
                jac_n[li] += int(later.numel())
            if li == 0 or sr.seg in ("cp", "lbd"):
                # CE by position inside the layer-1 segment (first MoE layer; for fixed all layers coincide)
                if li == 0:
                    gid = r["seg_gid"]
                    first = r["seg_first"]
                    flat_t = torch.arange(gid.numel(), device=device).view_as(gid)
                    segpos = flat_t - first[gid.clamp(min=0)]
                    for bi, (a, b) in enumerate(SEGPOS_BUCKETS):
                        m = valid & (segpos >= a) & (segpos < b) & (gid >= 0)
                        segpos_nll[bi] += float(nll[m].double().sum())
                        segpos_n[bi] += int(m.sum())
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    model.train()
    res = {"ce_nats_per_token": tot_nll / max(tot_n, 1), "n_valid_tokens": tot_n,
           "ppl": math.exp(tot_nll / max(tot_n, 1)), "doc_nll": doc_nll, "doc_n": doc_n, "n_rows": len(rows_all),
           "eval_seconds": elapsed, "eval_valid_tokens_per_s": tot_n / max(elapsed, 1e-9)}
    if diagnostics:
        frac = usage / usage.sum(1, keepdims=True).clip(min=1)
        tok = max(n_tok_routed, 1)
        res.update({
            "expert_max_share": frac.max(1).tolist(), "experts_unused": (usage == 0).sum(1).tolist(),
            "expert_load_cv": (frac.std(1) / frac.mean(1).clip(min=1e-12)).tolist(),
            "usage_entropy": (-(frac * np.log(frac.clip(min=1e-30))).sum(1)).tolist(),
            "routing_entropy_mean": (ent / tok).tolist(),
            "route_changes_per_1k_tokens": (1000 * route_changes / tok).tolist(),
            "consecutive_route_jaccard": (jac_sum / np.maximum(jac_n, 1)).tolist(),
            "router_invocations_per_1k_tokens": (1000 * router_calls / tok).tolist(),
            "ce_by_doc_position": {f"{a}-{b}": (pos_nll[i] / pos_n[i] if pos_n[i] else None)
                                   for i, (a, b) in enumerate(POS_BUCKETS)},
            "tokens_by_doc_position": {f"{a}-{b}": int(pos_n[i]) for i, (a, b) in enumerate(POS_BUCKETS)},
            "dispatch_max_over_mean": {"mean": float(np.mean(dispatch_max_over_mean)),
                                       "max": float(np.max(dispatch_max_over_mean))},
            "routed_tokens": n_tok_routed,
        })
        if sr.mode == "sequence":
            allL = [np.concatenate(x) if x else np.zeros(0) for x in seg_lens]
            res.update({
                "segments": seg_n.tolist(),
                "mean_segment_length": (n_tok_routed / np.maximum(seg_n, 1)).tolist(),
                "mean_closed_segment_length": (seg_closed_tok / np.maximum(seg_closed_n, 1)).tolist(),
                "segment_length_quantiles": [np.quantile(a, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist() if a.size else []
                                             for a in allL],
                "boundaries_per_1k_tokens": (1000 * seg_closed_n / tok).tolist(),
                "bos_route_token_fraction": (bos_tokens / tok).tolist(),
                "ce_by_segment_position_layer1": {f"{a}-{b}": (segpos_nll[i] / segpos_n[i] if segpos_n[i] else None)
                                                  for i, (a, b) in enumerate(SEGPOS_BUCKETS)},
            })
    return res


def collapse_check(ev, max_share=0.5):
    """Collapse: any MoE layer with an unused routed expert or one expert taking > max_share of routed slots."""
    unused = max(ev["experts_unused"])
    ms = max(ev["expert_max_share"])
    return {"collapsed": bool(unused > 0 or ms > max_share), "max_unused": int(unused), "max_share": float(ms)}
