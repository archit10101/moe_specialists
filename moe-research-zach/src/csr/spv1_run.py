"""csr-speculative-pool-routing-v1: paths, calib partition, split-access locks, config checks, draft-proposal
cache, batch sources and evaluation diagnostics.

Evaluation sets
  calib_search    : 70% of the existing calib documents (salted document-hash partition) - development only.
  calib_selection : the other 30% - only after candidate generation ended (search_frozen.json lists the checkpoint's
                    config hash); one evaluation per checkpoint.
  val             : existing validation split - only after the final configurations were frozen (final_frozen.json
                    lists the config hash); one evaluation per checkpoint.
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
from .data import TokenSplit, make_batch, to_torch, train_order
from .spv1_model import EXPERIMENT, PoolRouting
from .train_lib import BatchSource, plan_steps

ART = guard.ROOT / "artifacts" / "speculative-pool-routing-v1"
CFG_ROOT = guard.ROOT / "configs" / "speculative-pool-routing-v1"
ST_DIR = guard.ROOT / ".tmp" / "speculative_pool_v1"
PREREG = guard.ROOT / "docs" / "SPECULATIVE_POOL_ROUTING_V1_PREREG.md"
PARTITION = ART / "search_partition.json"
SEARCH_FROZEN = ART / "search_frozen.json"
FINAL_FROZEN = ART / "final_frozen.json"
DRAFT_FROZEN = ART / "draft_frozen.json"
EVAL_REGISTRY = ART / "eval_registry"
CACHE_DIR = ART / "draft_cache"
PARTITION_SALT = "csr-spv1-calib-partition"
SEARCH_FRACTION = 0.70
DISK_HARD_MIN = 80 * 1024**3
GAMMA_MAX = 7
DATA_DIR = "artifacts/data/fwe-rep-v1"


class SplitLocked(RuntimeError):
    pass


def prereg_sha():
    return guard.sha256_file(PREREG)


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
def _frozen_hashes(path):
    if not path.exists():
        return None
    return set(json.loads(path.read_text()).get("config_sha256s", []))


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


# ------------------------------------------------------------------ draft provenance and proposal cache
def frozen_draft():
    """(checkpoint path, record) of the frozen draft; raises if not frozen or hash differs."""
    if not DRAFT_FROZEN.exists():
        raise guard.GuardError("draft not frozen (no draft_frozen.json)")
    rec = json.loads(DRAFT_FROZEN.read_text())
    p = guard.ROOT / rec["checkpoint"]
    if guard.sha256_file(p) != rec["checkpoint_sha256"]:
        raise guard.GuardError("frozen draft checkpoint hash mismatch")
    return p, rec


def cache_manifest():
    p = CACHE_DIR / "train_proposals_manifest.json"
    if not p.exists():
        raise guard.GuardError("train proposal cache missing")
    return json.loads(p.read_text())


def open_proposal_cache(verify=False):
    man = cache_manifest()
    path = guard.ROOT / man["file"]
    if verify and guard.sha256_file(path) != man["sha256"]:
        raise guard.GuardError("proposal cache hash mismatch")
    _, rec = frozen_draft()
    if man["draft_checkpoint_sha256"] != rec["checkpoint_sha256"]:
        raise guard.GuardError("proposal cache was not produced by the frozen draft")
    return np.load(path, mmap_mode="r"), man


class PropBatchSource(BatchSource):
    """Training batches with cached causal draft proposals: proposals[b, t, j-1] = j-th greedy draft token generated
    from the prefix ending at t of the document at t (computed once by spv1_draft_cache.py from the frozen draft)."""

    def __init__(self, split, seq_len, order, cache, gamma_cache=GAMMA_MAX):
        super().__init__(split, seq_len, None, order)
        self.cache = cache
        self.g = gamma_cache

    def batch(self, rows):
        a = make_batch(rows, self.split, self.T)
        P = np.zeros((len(rows), self.T, self.g), dtype=np.int64)
        off = self.split.offsets
        for b, row in enumerate(rows):
            t = 0
            for di in row:
                L = int(off[di + 1] - off[di])
                P[b, t:t + L] = self.cache[off[di]:off[di + 1]]
                t += L
        a["proposals"] = P
        return a


# ------------------------------------------------------------------ configs and plans
def check_cfg(cfg):
    if cfg.get("experiment") != EXPERIMENT:
        raise guard.GuardError("not a speculative-pool-routing-v1 config")
    if cfg.get("prereg_sha256") != prereg_sha():
        raise guard.GuardError("config prereg hash does not match the frozen preregistration")
    d = guard.inside_root(cfg["data"]["dir"])
    man = json.loads((d / "manifest.json").read_text())
    if guard.sha256_json(man["splits"]) != cfg["data"]["splits_sha256"]:
        raise guard.GuardError("data manifest hash mismatch")
    if cfg.get("kind", "target") == "target":
        PoolRouting.from_cfg(cfg["routing"])
    return d


def train_source(cfg, d, with_proposals):
    tr = load_raw_split(d, "train")
    order = train_order(tr.n_docs, cfg["data"]["train_order_seed"])
    if with_proposals:
        cache, _ = open_proposal_cache()
        src = PropBatchSource(tr, cfg["data"]["seq_len"], order, cache)
    else:
        src = BatchSource(tr, cfg["data"]["seq_len"], None, order)
    t = cfg["train"]
    steps, total = plan_steps(src, t["micro_batch_rows"] * t["accum"], t["token_budget"])
    return tr, src, steps, total


def require_disk(extra_bytes: int, what: str):
    free = guard.disk_free_bytes()
    if free - extra_bytes < DISK_HARD_MIN:
        raise guard.GuardError(f"disk: free={free} need={extra_bytes} for {what}; hard minimum {DISK_HARD_MIN}")
    guard.ledger("SP_V1_DISK_CHECK", f"{what}: free={free} planned={extra_bytes} min={DISK_HARD_MIN} ok")


def param_sha(state_dict):
    h = hashlib.sha256()
    for n, p in sorted(state_dict.items()):
        h.update(n.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def flops_per_token(mcfg, pr: PoolRouting, seq_len=1024, refresh_rate=None, draft_cfg=None):
    """Analytic forward FLOPs per target token (multiply-add = 2; training = 3x forward). Attention uses the mean
    causal context seq_len/2. Pool families: token router (unchanged) + selector logits at every position (all
    positions are potential block starts in the parallel implementation) + pool gather/top-M. Spec: the draft
    proposal cost (gamma draft forwards per token) is reported separately (draft_forward)."""
    d, E, k, f, fs = mcfg.d_model, mcfg.n_experts, mcfg.top_k, mcfg.expert_ffn, mcfg.shared_ffn
    n_moe = len(mcfg.moe_layers)
    n_dense = mcfg.n_layer - n_moe
    attn = mcfg.n_layer * (2 * 4 * d * d + 2 * 2 * d * (seq_len / 2))
    dense = n_dense * 2 * 3 * d * mcfg.dense_ffn
    experts = n_moe * (k * 2 * 3 * d * f + 2 * 3 * d * fs)
    lm = 2 * d * mcfg.vocab_size
    router = n_moe * 2 * d * E
    extra = 0.0
    if pr.family != "token":
        if pr.pool_assign == "learned":
            extra += n_moe * 2 * d * E * (2 if pr.family == "spec" else 1)
        if pr.family == "adaptive":
            extra += n_moe * (2 * d + 2 * E * pr.seg_len)
    out = {"forward": attn + dense + experts + lm + router + extra, "router_forward": router, "pool_extra": extra}
    out["train"] = 3 * out["forward"]
    if pr.family == "spec" and draft_cfg is not None:
        dd = draft_cfg.d_model
        dfw = draft_cfg.n_layer * (2 * 4 * dd * dd + 2 * 2 * dd * (seq_len / 2) + 2 * 3 * dd * draft_cfg.dense_ffn) \
            + 2 * dd * draft_cfg.vocab_size
        out["draft_forward_per_proposal_token"] = dfw
        out["draft_tree_forward_per_target_token"] = pr.gamma * dfw
    return out


# ------------------------------------------------------------------ evaluation with diagnostics
class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


POS_BUCKETS = [(0, 16), (16, 64), (64, 256), (256, 1024)]
WINDOWS = (4, 8, 16, 32)
MAXLEN_HIST = 9


@torch.no_grad()
def pool_evaluate(model, source: BatchSource, device, rows_per_batch=16, max_rows=None, autocast=True,
                  diagnostics=True):
    """Token-weighted CE (nats per valid prediction token), per-document NLL sums, routing / pool diagnostics."""
    model.eval()
    pr = model.pr
    rows_all = source.rows if max_rows is None else source.rows[:max_rows]
    E, K = model.cfg.n_experts, model.cfg.top_k
    n_moe = len(model.cfg.moe_layers)
    z = lambda: np.zeros(n_moe)  # noqa: E731
    usage = np.zeros((n_moe, E))
    acc = {k: z() for k in ("ent", "maxp", "gate_mass", "jac", "starts", "recall", "contained", "sel_mass",
                            "sel_ent", "sel_n")}
    win = {w: [z(), z()] for w in WINDOWS}
    blen_hist = np.zeros((n_moe, MAXLEN_HIST + 1))
    n_tok = 0
    n_pairs = 0
    doc_nll, doc_n = {}, {}
    tot_nll, tot_n = 0.0, 0
    pos_nll = np.zeros(len(POS_BUCKETS))
    pos_n = np.zeros(len(POS_BUCKETS))
    spec = {"blocks": 0, "mismatch_ends": 0, "tokens": 0, "accepted": 0}
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
        doc_ids = tb["doc_ids"]
        nonpad = doc_ids >= 0
        n_np = int(nonpad.sum())
        n_tok += n_np
        same_next = nonpad[:, 1:] & (doc_ids[:, 1:] == doc_ids[:, :-1])
        n_pairs += int(same_next.sum())
        Bb, T = doc_ids.shape
        for li, r in enumerate(out.routing):
            topi = r["topi"]
            usage[li] += np.bincount(topi[nonpad].reshape(-1).cpu().numpy(), minlength=E)
            p = r["probs"].float()
            pn = p[nonpad]
            acc["ent"][li] += float((-(pn * pn.clamp_min(1e-30).log()).sum(-1)).sum())
            acc["maxp"][li] += float(pn.max(-1).values.sum())
            acc["gate_mass"][li] += float(p.gather(-1, topi)[nonpad].sum())
            oh = torch.zeros(Bb, T, E, device=device).scatter_(2, topi, 1.0)
            inter = (oh[:, 1:] * oh[:, :-1]).sum(-1)
            acc["jac"][li] += float((inter / (2 * K - inter))[same_next].sum())
            for w in WINDOWS:
                nw = T // w
                ohw = oh[:, :nw * w].view(Bb, nw, w, E)
                dw = doc_ids[:, :nw * w].view(Bb, nw, w)
                okw = (dw >= 0).all(-1) & (dw == dw[:, :, :1]).all(-1)
                distinct = (ohw.amax(2) > 0).sum(-1).float()
                win[w][0][li] += float(distinct[okw].sum())
                win[w][1][li] += int(okw.sum())
            if pr.family == "token":
                acc["recall"][li] += n_np
                acc["contained"][li] += n_np
                continue
            cand = r["cand"]
            st = r["starts"] & nonpad
            acc["starts"][li] += int(st.sum())
            unr = p.topk(K, dim=-1).indices
            inp = cand.gather(-1, unr)[nonpad].float()
            acc["recall"][li] += float(inp.mean(-1).sum())
            acc["contained"][li] += float(inp.all(-1).float().sum())
            if "logm_tok" in r:
                lm = r["logm_tok"].float()[st]
                mm = lm.exp()
                acc["sel_mass"][li] += float((mm * cand[st].float()).sum())
                acc["sel_ent"][li] += float((-(mm * lm).sum(-1)).sum())
                acc["sel_n"][li] += int(st.sum())
            # block lengths (diagnostic, host-side)
            stn = r["starts"].cpu().numpy()
            npn = nonpad.cpu().numpy()
            for b in range(Bb):
                idx = np.flatnonzero(stn[b] & npn[b])
                endv = np.append(idx[1:], npn[b].sum())
                lens = endv - idx
                blen_hist[li] += np.bincount(np.minimum(lens, MAXLEN_HIST + 1) - 1, minlength=MAXLEN_HIST + 1)[
                    :MAXLEN_HIST + 1]
        if pr.family == "spec" and out.spec_starts is not None:
            sst = (out.spec_starts & nonpad)
            props = out.proposals
            spec["blocks"] += int(sst.sum())
            spec["tokens"] += n_np
            # a block ends early iff the next block starts because of a proposal mismatch (not max length / doc end)
            bsi = torch.cummax(torch.where(sst, torch.arange(T, device=device).unsqueeze(0),
                                           torch.full_like(doc_ids, -1)), 1).values
            off = torch.arange(T, device=device).unsqueeze(0) - torch.cat([bsi[:, :1] * 0, bsi[:, :-1]], 1)
            prev_same = torch.cat([torch.zeros_like(nonpad[:, :1]), same_next], 1)
            mism = sst & prev_same & (off < pr.seg_len)
            spec["mismatch_ends"] += int(mism.sum())
            spec["accepted"] += n_np - int(sst.sum())
            del props
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    model.train()
    res = {"ce_nats_per_token": tot_nll / max(tot_n, 1), "n_valid_tokens": tot_n,
           "ppl": math.exp(tot_nll / max(tot_n, 1)), "doc_nll": doc_nll, "doc_n": doc_n, "n_rows": len(rows_all),
           "eval_seconds": elapsed, "eval_valid_tokens_per_s": tot_n / max(elapsed, 1e-9)}
    if diagnostics:
        tok = max(n_tok, 1)
        frac = usage / usage.sum(1, keepdims=True).clip(min=1)
        res.update({
            "expert_max_share": frac.max(1).tolist(), "experts_unused": (usage == 0).sum(1).tolist(),
            "expert_load_cv": (frac.std(1) / frac.mean(1).clip(min=1e-12)).tolist(),
            "expert_load_frac": frac.tolist(),
            "router_entropy": (acc["ent"] / tok).tolist(), "router_max_prob": (acc["maxp"] / tok).tolist(),
            "selected_gate_mass": (acc["gate_mass"] / tok).tolist(),
            "adjacent_expert_jaccard": (acc["jac"] / max(n_pairs, 1)).tolist(),
            "distinct_experts_per_window": {str(w): (win[w][0] / np.maximum(win[w][1], 1)).tolist() for w in WINDOWS},
            "pool_recall_vs_unrestricted_top4": (acc["recall"] / tok).tolist(),
            "unrestricted_top4_contained_frac": (acc["contained"] / tok).tolist(),
            "ce_by_doc_position": {f"{a}-{b}": (pos_nll[i] / pos_n[i] if pos_n[i] else None)
                                   for i, (a, b) in enumerate(POS_BUCKETS)},
            "routed_tokens": n_tok,
        })
        if pr.family != "token":
            res.update({
                "refresh_rate": (acc["starts"] / tok).tolist(),
                "mean_block_length": (tok / np.maximum(acc["starts"], 1)).tolist(),
                "block_length_hist": blen_hist.tolist(),
                "selector_pool_mass": (acc["sel_mass"] / np.maximum(acc["sel_n"], 1)).tolist()
                if pr.pool_assign == "learned" else None,
                "selector_entropy": (acc["sel_ent"] / np.maximum(acc["sel_n"], 1)).tolist()
                if pr.pool_assign == "learned" else None,
            })
        if pr.family == "spec":
            res["teacher_forced_spec"] = {
                "blocks": spec["blocks"], "mean_block_length": spec["tokens"] / max(spec["blocks"], 1),
                "mean_accepted_draft_tokens_per_block": spec["accepted"] / max(spec["blocks"], 1),
                "blocks_started_by_mismatch_frac": spec["mismatch_ends"] / max(spec["blocks"], 1)}
    return res


def collapse_check(ev, max_share=0.5):
    """Collapse: any MoE layer with an unused routed expert or one expert taking > max_share of routed slots."""
    unused = max(ev["experts_unused"])
    ms = max(ev["expert_max_share"])
    return {"collapsed": bool(unused > 0 or ms > max_share), "max_unused": int(unused), "max_share": float(ms)}
