"""Loss accounting, optimizer/schedule, step planning and held-out evaluation shared by all entry points."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .data import make_batch, pack_rows, to_torch, train_order
from .model import CSRModel, ModelConfig, RoutingConfig
from .semantic_loss import relational_kl, select_valid_chunks


def routing_from_cfg(cfg) -> RoutingConfig:
    r = cfg["routing"]
    rc = RoutingConfig(mode=r["mode"], pool_size=r["pool_size"], warmup=r["warmup"],
                       gate_norm=r.get("gate_norm", "global"))
    mc = ModelConfig.from_dict(cfg["model"])
    rc.validate(mc)
    if rc.pool_size != min(mc.n_experts, 2 * mc.top_k) and not r.get("pool_size_override_reason"):
        raise ValueError(f"pool_size {rc.pool_size} != min(E, 2k) = {min(mc.n_experts, 2 * mc.top_k)}")
    return rc


def build_model(cfg, device, dtype=torch.float32):
    torch.manual_seed(cfg["init"]["seed"])
    model = CSRModel(ModelConfig.from_dict(cfg["model"]))
    if cfg["init"]["type"] == "checkpoint":
        sd = torch.load(cfg["init"]["path"], map_location="cpu", weights_only=True)
        model.load_state_dict(sd["model"] if "model" in sd else sd, strict=True)
    return model.to(device=device, dtype=dtype)


# ------------------------------------------------------------------ batches
class BatchSource:
    """Condition-independent packing; segmentation only supplies chunk starts / keys."""

    def __init__(self, split, seq_len, seg_table=None, order=None):
        self.split, self.T, self.seg = split, seq_len, seg_table
        order = list(range(split.n_docs)) if order is None else order
        self.rows = pack_rows(split.lengths(), order, seq_len)

    def batch(self, rows):
        if self.seg is None:
            return make_batch(rows, self.split, self.T)
        return make_batch(rows, self.split, self.T, starts_of=self.seg.starts, chunk_key_of=self.seg.chunk_base)


def valid_tokens(rows, split):
    return int(sum(split.lengths()[r].sum() - len(r) for r in rows))


def plan_steps(source: BatchSource, rows_per_step: int, token_budget: int):
    """Smallest number of whole steps whose valid prediction tokens reach token_budget. Condition-independent."""
    steps, total = [], 0
    for s in range(0, len(source.rows) - rows_per_step + 1, rows_per_step):
        if total >= token_budget:
            break
        rows = source.rows[s:s + rows_per_step]
        steps.append(rows)
        total += valid_tokens(rows, source.split)
    if total < token_budget:
        raise ValueError(f"not enough training data: {total} < {token_budget}")
    return steps, total


# ------------------------------------------------------------------ losses
@dataclass
class LossParts:
    lm_sum: torch.Tensor
    n_valid: int
    aux: torch.Tensor
    z: torch.Tensor
    sem: torch.Tensor
    sem_stats: dict
    coverage_constrained: int


def forward_losses(model, tb, cfg, seg_table=None, need_chunk_logits=None):
    routing = routing_from_cfg(cfg)
    sem_on = cfg["semantic"]["enabled"]
    out = model(tb, routing, need_chunk_logits=sem_on if need_chunk_logits is None else need_chunk_logits)
    tgt = tb["targets"]
    valid = tgt != -100
    logits = out.logits.float()
    lm_sum = F.cross_entropy(logits.view(-1, logits.shape[-1]), tgt.view(-1), ignore_index=-100, reduction="sum")
    n_valid = int(valid.sum().item())
    cov = int((out.routing[0]["constrained"] & valid).sum().item()) if out.routing else 0
    sem = lm_sum.new_zeros(())
    stats = {}
    if sem_on:
        lay = out.layout
        keys = tb["chunk_key"][lay.rows_b, lay.rows_t0]
        assert (keys >= 0).all(), "chunk keys missing for semantic loss"
        kc = keys.cpu().numpy()
        u = torch.as_tensor(seg_table.u[kc].astype(np.float32), device=lm_sum.device)
        ph = seg_table.phash[kc]
        idx = select_valid_chunks(u, ph, cfg["semantic"]["n_max"])
        cl = torch.stack([r["chunk_logits"] for r in out.routing], 0)[:, idx]
        doc_of = tb["gdoc"][lay.rows_b, lay.rows_t0][idx]
        sem, stats = relational_kl(u[idx], cl, cfg["semantic"]["tau_s"], cfg["semantic"]["tau_r"],
                                   cfg["semantic"]["min_chunks"], doc_of=doc_of)
    return LossParts(lm_sum, n_valid, out.aux_loss, out.z_loss, sem, stats, cov), out


def step_objective(parts: LossParts, n_valid_step: int, cfg):
    """Contribution of one micro-batch to the optimizer-step objective. The LM term is token-weighted across
    micro-batches (exactly equal to the full-batch mean CE); per-micro-batch auxiliary terms are weighted by
    the micro-batch's share of valid tokens."""
    t = cfg["train"]
    w = parts.n_valid / n_valid_step
    lam = cfg["semantic"]["lambda"] if cfg["semantic"]["enabled"] else 0.0
    aux_terms = t["aux_coeff"] * parts.aux + t["z_coeff"] * parts.z + lam * parts.sem
    return parts.lm_sum / n_valid_step + w * aux_terms


def router_params(model):
    return [b.ffn.router.weight for b in model.moe_blocks()]


def router_grad_norms(model, parts: LossParts, cfg):
    """Separate router-gradient norms of the LM term and the weighted semantic term (monitoring/calibration)."""
    rp = router_params(model)
    g_lm = torch.autograd.grad(parts.lm_sum / max(parts.n_valid, 1), rp, retain_graph=True, allow_unused=True)
    n_lm = math.sqrt(sum(float(g.float().pow(2).sum()) for g in g_lm if g is not None))
    n_sem = 0.0
    if cfg["semantic"]["enabled"] and parts.sem.requires_grad:
        g_s = torch.autograd.grad(parts.sem, rp, retain_graph=True, allow_unused=True)
        n_sem = math.sqrt(sum(float(g.float().pow(2).sum()) for g in g_s if g is not None))
    return n_lm, n_sem  # n_sem is UNweighted; weighted = lambda * n_sem


# ------------------------------------------------------------------ optimizer / schedule
def make_optimizer(model, cfg, device):
    t = cfg["train"]
    decay = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    groups = [{"params": decay, "weight_decay": t["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=t["lr"], betas=tuple(t["betas"]), eps=1e-8,
                             fused=(device.type == "cuda"))


def lr_at(step, n_steps, t):
    """WSD: linear warmup, constant, linear decay over the final decay_frac of steps to min_lr."""
    if step < t["warmup_steps"]:
        return t["lr"] * (step + 1) / t["warmup_steps"]
    d0 = int(n_steps * (1 - t["decay_frac"]))
    if step < d0:
        return t["lr"]
    frac = (step - d0) / max(1, n_steps - d0)
    return t["lr"] + (t["min_lr"] - t["lr"]) * frac


# ------------------------------------------------------------------ evaluation
@torch.no_grad()
def evaluate(model, source: BatchSource, cfg, device, rows_per_batch=16, max_rows=None, autocast=None,
             seg_table=None, collect_routing=True):
    """Token-weighted held-out CE (nats / valid prediction token), per-document sums, routing metrics."""
    model.eval()
    routing = routing_from_cfg(cfg)
    rows_all = source.rows if max_rows is None else source.rows[:max_rows]
    E = model.cfg.n_experts
    n_moe = len(model.cfg.moe_layers)
    doc_nll, doc_n = {}, {}
    tot_nll, tot_n, tot_cov = 0.0, 0, 0
    usage = np.zeros((n_moe, E), np.int64)
    pool_usage = np.zeros((n_moe, E), np.int64)
    adj_j, adj_n = np.zeros(n_moe), 0
    tok_ent, ent_n = np.zeros(n_moe), 0
    pools_seen = [set() for _ in range(n_moe)]
    n_chunks = 0
    for s in range(0, len(rows_all), rows_per_batch):
        tb = to_torch(source.batch(rows_all[s:s + rows_per_batch]), device)
        ctx = torch.autocast(device.type, dtype=torch.bfloat16) if autocast else _null()
        with ctx:
            out = model(tb, routing, need_chunk_logits=collect_routing)
        logits = out.logits.float()
        tgt = tb["targets"]
        valid = tgt != -100
        nll = F.cross_entropy(logits.view(-1, logits.shape[-1]), tgt.view(-1), ignore_index=-100,
                              reduction="none").view(tgt.shape)
        tot_nll += float(nll[valid].double().sum())
        tot_n += int(valid.sum())
        tot_cov += int((out.routing[0]["constrained"] & valid).sum())
        g = tb["gdoc"][valid].cpu().numpy()
        v = nll[valid].double().cpu().numpy()
        ug, inv = np.unique(g, return_inverse=True)
        sums = np.bincount(inv, weights=v)
        cnts = np.bincount(inv)
        for d, sv, cv in zip(ug, sums, cnts):
            doc_nll[int(d)] = doc_nll.get(int(d), 0.0) + float(sv)
            doc_n[int(d)] = doc_n.get(int(d), 0) + int(cv)
        if not collect_routing:
            continue
        nonpad = tb["doc_ids"] >= 0
        same_next = nonpad[:, 1:] & (tb["doc_ids"][:, 1:] == tb["doc_ids"][:, :-1])
        for li, r in enumerate(out.routing):
            topi = r["topi"]
            usage[li] += np.bincount(topi[nonpad].reshape(-1).cpu().numpy(), minlength=E)
            p = r["probs_full"][nonpad].float()
            tok_ent[li] += float((-(p * p.clamp_min(1e-30).log()).sum(-1)).sum())
            oh = torch.zeros(*topi.shape[:2], E, device=device).scatter_(2, topi, 1.0)
            inter = (oh[:, 1:] * oh[:, :-1]).sum(-1)
            jac = inter / (2 * topi.shape[-1] - inter)
            adj_j[li] += float(jac[same_next].sum())
            if r["chunk_logits"] is not None and r["chunk_logits"].numel():
                pidx = r["chunk_logits"].topk(routing.pool_size, dim=-1).indices
                pool_usage[li] += np.bincount(pidx.reshape(-1).cpu().numpy(), minlength=E)
                for row in torch.sort(pidx, dim=-1).values.cpu().numpy():
                    pools_seen[li].add(row.tobytes())
                if li == 0:
                    n_chunks += pidx.shape[0]
        adj_n += int(same_next.sum())
        ent_n += int(nonpad.sum())
    model.train()
    res = {"ce_nats_per_token": tot_nll / max(tot_n, 1), "n_valid_tokens": tot_n,
           "ppl": math.exp(tot_nll / max(tot_n, 1)), "pool_constrained_coverage": tot_cov / max(tot_n, 1),
           "doc_nll": doc_nll, "doc_n": doc_n}
    if collect_routing:
        frac = usage / usage.sum(1, keepdims=True).clip(min=1)
        res.update({
            "expert_load_max_over_mean": (frac.max(1) * E).tolist(),
            "expert_load_cv": (frac.std(1) / frac.mean(1).clip(min=1e-12)).tolist(),
            "experts_unused": (usage == 0).sum(1).tolist(),
            "usage_entropy": (-(frac * np.log(frac.clip(min=1e-30))).sum(1)).tolist(),
            "token_router_entropy": (tok_ent / max(ent_n, 1)).tolist(),
            "adjacent_token_expert_jaccard": (adj_j / max(adj_n, 1)).tolist(),
            "pool_usage_entropy": [float(-(q * np.log(q.clip(min=1e-30))).sum()) for q in
                                   (pool_usage / pool_usage.sum(1, keepdims=True).clip(min=1))],
            "pool_distinct_fraction": [len(s) / max(n_chunks, 1) for s in pools_seen],
            "n_chunks_with_prefix": n_chunks,
        })
    return res


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def paired_bootstrap(doc_a: dict, doc_b: dict, n_a: dict, n_b: dict, n_boot=2000, seed=0):
    """Document-level paired bootstrap of the token-weighted CE difference (b - a)."""
    docs = sorted(set(doc_a) & set(doc_b))
    assert all(n_a[d] == n_b[d] for d in docs), "valid-token counts differ between methods"
    A = np.array([doc_a[d] for d in docs])
    Bv = np.array([doc_b[d] for d in docs])
    N = np.array([n_a[d] for d in docs], dtype=float)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(docs), len(docs))
        diffs.append((Bv[i].sum() - A[i].sum()) / N[i].sum())
    point = (Bv.sum() - A.sum()) / N.sum()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": float(point), "ci95": [float(lo), float(hi)], "n_docs": len(docs)}
