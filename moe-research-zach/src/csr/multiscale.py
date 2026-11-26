"""csr-multiscale-routing-v1: causal multiscale context-conditioned token routing (docs/MULTISCALE_ROUTING_V1_PREREG.md).

New module; it does not change CSRModel or any earlier routing mode. MultiscaleModel subclasses CSRModel with the
same parameters (no new parameters) and adds a "multiscale" forward and incremental step.

Per MoE layer l and token t (h = post-RMSNorm hidden state read by the ordinary router; W = bias-free router):
  z_token = W h_t
  c_local = center(W mean(h[chunk_start : chunk_start + min(chunk_pos, 16)]))    (0 if chunk_pos == 0)
  c_broad = center(W mean(h[t - n : t])) * min(n / 512, 1),  n = min(pos_in_doc, 1024)  (0 if n == 0)
  z_raw   = z_token + a_l * alpha * c_local + b_l * beta * c_broad
  scale control: z_route = mean_E(z_token) + sg(s) * center(z_token),  s = RMS(center(z_raw)) / max(RMS(center(z_token)), eps)
Selection: softmax(z_route) over all experts, top-k by probability, gate = full-softmax probability, dropless,
plus the shared expert (identical to the FLAME token rule).
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .model import CSRModel, ForwardOut, rope_cos_sin

EXPERIMENT = "csr-multiscale-routing-v1"
KINDS = ("token", "local", "broad", "hier")
N_MOE = 7

_TOK7 = ("token",) * 7
SCHEDULES = {
    "TOKEN": _TOK7,
    "BROAD-EARLY": ("broad",) * 3 + ("token",) * 4,
    "BROAD-LATE": ("token",) * 4 + ("broad",) * 3,
    "LOCAL": ("token",) * 2 + ("local",) * 5,
    "HIER": ("token",) * 2 + ("local",) * 2 + ("hier",) * 3,
}
# condition -> (schedule key, segmentation strategy, scale_control, reference condition)
CONDITIONS = {
    "TOKEN": ("TOKEN", "none", False, None),
    "BROAD-EARLY": ("BROAD-EARLY", "none", False, None),
    "BROAD-LATE": ("BROAD-LATE", "none", False, None),
    "LOCAL-FIXED": ("LOCAL", "fixed", False, None),
    "LOCAL-SEMANTIC": ("LOCAL", "semantic", False, None),
    "LOCAL-RANDOM": ("LOCAL", "random", False, None),
    "HIER-FIXED": ("HIER", "fixed", False, None),
    "HIER-SEMANTIC": ("HIER", "semantic", False, None),
    "HIER-RANDOM": ("HIER", "random", False, None),
    "SCALE-ONLY-LOCAL": ("LOCAL", "semantic", True, "LOCAL-SEMANTIC"),
    "SCALE-ONLY-BROAD-LATE": ("BROAD-LATE", "none", True, "BROAD-LATE"),
    "SCALE-ONLY-HIER": ("HIER", "semantic", True, "HIER-SEMANTIC"),
}
CONTEXTUAL = {c for c, v in CONDITIONS.items() if c != "TOKEN" and not v[2]}
BRANCH_CONDITIONS = {
    "A": [],
    "B": ["TOKEN", "SCALE-ONLY-LOCAL", "LOCAL-FIXED", "LOCAL-SEMANTIC", "LOCAL-RANDOM"],
    "C": ["TOKEN", "SCALE-ONLY-BROAD-LATE", "BROAD-EARLY", "BROAD-LATE"],
    "D": ["TOKEN", "SCALE-ONLY-HIER", "BROAD-EARLY", "BROAD-LATE", "LOCAL-FIXED", "LOCAL-SEMANTIC", "LOCAL-RANDOM",
          "HIER-FIXED", "HIER-SEMANTIC", "HIER-RANDOM"],
}
CONTRASTS = [  # (id, row, column)
    (1, "BROAD-LATE", "TOKEN"), (2, "BROAD-LATE", "BROAD-EARLY"), (3, "BROAD-LATE", "SCALE-ONLY-BROAD-LATE"),
    (4, "LOCAL-SEMANTIC", "TOKEN"), (5, "LOCAL-SEMANTIC", "SCALE-ONLY-LOCAL"), (6, "LOCAL-SEMANTIC", "LOCAL-FIXED"),
    (7, "LOCAL-SEMANTIC", "LOCAL-RANDOM"), (8, "HIER-SEMANTIC", "SCALE-ONLY-HIER"), (9, "HIER-SEMANTIC", "BROAD-LATE"),
    (10, "HIER-SEMANTIC", "LOCAL-SEMANTIC"), (11, "HIER-SEMANTIC", "HIER-FIXED"), (12, "HIER-SEMANTIC", "HIER-RANDOM"),
]
APPROVED_DIFF_MAIN = {"condition", "run_name", "segmentation.strategy", "routing.schedule", "routing.scale_control",
                      "routing.reference"}
APPROVED_DIFF_CALIB = APPROVED_DIFF_MAIN | {"routing.alpha", "routing.beta"}


def branch_of(alpha: float, beta: float) -> str:
    return {(False, False): "A", (True, False): "B", (False, True): "C", (True, True): "D"}[(alpha > 0, beta > 0)]


@dataclass
class MSRouting:
    alpha: float = 0.0
    beta: float = 0.0
    schedule: tuple = _TOK7
    scale_control: bool = False
    local_max: int = 16
    broad_window: int = 1024
    broad_ramp: int = 512
    eps: float = 1e-6
    mode: str = "multiscale"

    def validate(self, n_moe=N_MOE):
        assert self.mode == "multiscale"
        assert len(self.schedule) == n_moe and all(k in KINDS for k in self.schedule), self.schedule
        assert self.alpha >= 0 and self.beta >= 0
        assert self.local_max >= 1 and self.broad_window >= 1 and self.broad_ramp >= 1
        if self.scale_control:
            assert any(k != "token" for k in self.schedule)
        return self

    def uses_local(self, kind):
        return kind in ("local", "hier") and self.alpha != 0.0

    def uses_broad(self, kind):
        return kind in ("broad", "hier") and self.beta != 0.0

    @classmethod
    def from_cfg(cls, r: dict):
        return cls(alpha=float(r["alpha"]), beta=float(r["beta"]), schedule=tuple(r["schedule"]),
                   scale_control=bool(r["scale_control"]), local_max=int(r.get("local_max", 16)),
                   broad_window=int(r.get("broad_window", 1024)), broad_ramp=int(r.get("broad_ramp", 512)),
                   eps=float(r.get("eps", 1e-6))).validate()

    def to_dict(self):
        d = asdict(self)
        d["schedule"] = list(self.schedule)
        return d


def center(z):
    return z - z.mean(dim=-1, keepdim=True)


def rms(z):
    return z.pow(2).mean(dim=-1).sqrt()


def context_means(h, positions, chunk_pos, nonpad, need_local, need_broad, local_max=16, broad_window=1024):
    """Causal means of h (B, T, d) excluding the current token, via shifted float64 prefix sums along the packed row.
    Rows hold whole documents, so pos_in_doc preceding tokens are exactly the in-document history.
    Returns (m_local, n_local, m_broad, n_broad); means have h.dtype and are zero where the count is zero."""
    B, T, d = h.shape
    cs = F.pad(torch.cumsum(h.double(), dim=1), (0, 0, 1, 0))  # cs[:, i] = sum h[:, :i]
    t = torch.arange(T, device=h.device).unsqueeze(0).expand(B, T)
    zero = torch.zeros_like(positions)
    out = [None, None, None, None]

    def gather(idx):
        return cs.gather(1, idx.unsqueeze(-1).expand(B, T, d))

    if need_local:
        n = torch.where(nonpad, chunk_pos.clamp(max=local_max), zero)
        t0 = (t - torch.where(nonpad, chunk_pos, zero)).clamp(min=0)
        s = gather(t0 + n) - gather(t0)
        out[0] = (s / n.clamp(min=1).unsqueeze(-1).double()).to(h.dtype) * (n > 0).unsqueeze(-1)
        out[1] = n
    if need_broad:
        n = torch.where(nonpad, positions.clamp(max=broad_window), zero)
        s = gather(t) - gather(t - n)
        out[2] = (s / n.clamp(min=1).unsqueeze(-1).double()).to(h.dtype) * (n > 0).unsqueeze(-1)
        out[3] = n
    return tuple(out)


class MultiscaleModel(CSRModel):
    """Same parameters as CSRModel; forward(batch, MSRouting)."""

    def route_logits(self, moe, h_flat, kind, routing: MSRouting, ctx, collect=None):
        """h_flat (N, d). ctx: dict with m_local, m_broad, ramp (flattened) or None. Returns z_route (N, E) fp32."""
        z_tok = moe.router_logits(h_flat)
        if kind == "token" or (not routing.uses_local(kind) and not routing.uses_broad(kind)):
            if collect is not None:
                collect.update(z_token=z_tok, ctx=None)
            return z_tok
        z_raw = z_tok
        ctx_sum = torch.zeros_like(z_tok)
        if routing.uses_local(kind):
            c = center(moe.router_logits(ctx["m_local"]))
            ctx_sum = ctx_sum + routing.alpha * c
        if routing.uses_broad(kind):
            c = center(moe.router_logits(ctx["m_broad"])) * ctx["ramp"].unsqueeze(-1)
            ctx_sum = ctx_sum + routing.beta * c
        z_raw = z_tok + ctx_sum
        if routing.scale_control:
            ct = center(z_tok)
            s = (rms(center(z_raw)) / rms(ct).clamp(min=routing.eps)).detach()
            z_route = z_tok.mean(dim=-1, keepdim=True) + s.unsqueeze(-1) * ct
        else:
            z_route = z_raw
        if collect is not None:
            collect.update(z_token=z_tok, ctx=ctx_sum, z_raw=z_raw)
        return z_route

    @staticmethod
    def select_apply(moe, h_flat, z_route):
        probs = torch.softmax(z_route, dim=-1)
        gates, topi = probs.topk(moe.k, dim=-1)
        y = moe.experts(h_flat, topi, gates)
        if moe.shared is not None:
            y = y + moe.shared(h_flat)
        return y, topi, gates, probs

    def forward(self, batch, routing: MSRouting, collect: bool = False):
        ids, doc_ids, positions = batch["input_ids"], batch["doc_ids"], batch["positions"]
        B, T = ids.shape
        x = self.tok_emb(ids)
        cos, sin = rope_cos_sin(positions, self.cfg.d_model // self.cfg.n_head, self.cfg.rope_base, x.dtype)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        mask = self.attention_mask(doc_ids, positions)
        nonpad = doc_ids >= 0
        chunk_pos = batch.get("chunk_pos")
        acc = torch.float64 if x.dtype == torch.float64 else torch.float32
        aux_total = x.new_zeros((), dtype=acc)
        z_total = x.new_zeros((), dtype=acc)
        records = []
        moe_i = 0
        for blk in self.blocks:
            x = x + blk.attn(blk.norm1(x), cos, sin, mask)
            h = blk.norm2(x)
            if not blk.is_moe:
                x = x + blk.ffn(h)
                continue
            moe = blk.ffn
            kind = routing.schedule[moe_i]
            nl, nb = routing.uses_local(kind), routing.uses_broad(kind)
            ctx = None
            hf = h.reshape(B * T, -1)
            if nl or nb:
                rd = torch.float32 if h.dtype in (torch.float16, torch.bfloat16) else h.dtype
                with torch.autocast(device_type=h.device.type, enabled=False):
                    m_l, n_l, m_b, n_b = context_means(h.to(rd), positions, chunk_pos, nonpad, nl, nb,
                                                       routing.local_max, routing.broad_window)
                ctx = {}
                if nl:
                    ctx["m_local"] = m_l.reshape(B * T, -1)
                if nb:
                    ctx["m_broad"] = m_b.reshape(B * T, -1)
                    ctx["ramp"] = (n_b.to(torch.float32) / routing.broad_ramp).clamp(max=1.0).reshape(-1)
            rec = {} if collect else None
            z_route = self.route_logits(moe, hf, kind, routing, ctx, rec)
            y, topi, gates, probs = self.select_apply(moe, hf, z_route)
            x = x + y.view(B, T, -1)
            npm = nonpad.reshape(-1)
            n = npm.sum().clamp(min=1)
            pr = probs[npm]
            counts = torch.bincount(topi[npm].reshape(-1), minlength=moe.E).to(pr.dtype)
            P = pr.sum(0) / n
            f = counts / (n * moe.k)
            aux_total = aux_total + (moe.E * (f * P).sum()).to(acc)
            z_total = z_total + (torch.logsumexp(z_route[npm], dim=-1).pow(2).sum() / n).to(acc)
            r = {"topi": topi.view(B, T, -1), "gates": gates.view(B, T, -1), "kind": kind}
            if collect:
                r.update({"probs_full": probs.view(B, T, -1), "z_route": z_route.view(B, T, -1),
                          "z_token": rec["z_token"].view(B, T, -1), "h": h,
                          "ctx": None if rec["ctx"] is None else rec["ctx"].view(B, T, -1)})
                if ctx is not None:
                    r["m_local"] = ctx.get("m_local")
                    r["m_broad"] = ctx.get("m_broad")
            records.append(r)
            moe_i += 1
        logits = self.lm_head(self.norm_f(x))
        return ForwardOut(logits, aux_total, z_total, records, None)

    # ---------------- incremental (single document, batch 1) ----------------
    def new_ms_state(self):
        return {"kv": [dict() for _ in self.blocks], "moe": [dict() for _ in self.moe_blocks()]}

    @torch.no_grad()
    def ms_step(self, token_id: int, position: int, chunk_pos: int, routing: MSRouting, state):
        """One token of one document. chunk_pos must be derived from past tokens only (causal boundaries).
        Returns logits (V,) and per-layer (topi, z_route)."""
        dev = self.tok_emb.weight.device
        x = self.tok_emb(torch.tensor([[token_id]], device=dev))
        cos, sin = rope_cos_sin(torch.tensor([[position]], device=dev), self.cfg.d_model // self.cfg.n_head,
                                self.cfg.rope_base, x.dtype)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        out = []
        moe_i = 0
        for bi, blk in enumerate(self.blocks):
            x = x + blk.attn(blk.norm1(x), cos, sin, None, kv_cache=state["kv"][bi])
            h = blk.norm2(x)
            if not blk.is_moe:
                x = x + blk.ffn(h)
                continue
            moe, st = blk.ffn, state["moe"][moe_i]
            kind = routing.schedule[moe_i]
            hv = h.view(1, -1)
            hd = hv[0].double()
            if position == 0:
                st.clear()
            if chunk_pos == 0:
                st["c_sum"], st["c_n"] = torch.zeros_like(hd), 0
            st.setdefault("d_hist", deque())
            st.setdefault("d_sum", torch.zeros_like(hd))
            ctx = None
            nl, nb = routing.uses_local(kind), routing.uses_broad(kind)
            if nl or nb:
                ctx = {}
                if nl:
                    n = st["c_n"]
                    ctx["m_local"] = ((st["c_sum"] / max(n, 1)).to(hv.dtype) * (n > 0)).view(1, -1)
                if nb:
                    n = len(st["d_hist"])
                    ctx["m_broad"] = ((st["d_sum"] / max(n, 1)).to(hv.dtype) * (n > 0)).view(1, -1)
                    ctx["ramp"] = torch.tensor([min(n / routing.broad_ramp, 1.0)], device=dev, dtype=torch.float32)
            z_route = self.route_logits(moe, hv, kind, routing, ctx)
            y, topi, _, _ = self.select_apply(moe, hv, z_route)
            x = x + y.view(1, 1, -1)
            out.append((topi[0], z_route[0]))
            # update histories AFTER routing (the current token never sees itself)
            if st["c_n"] < routing.local_max:
                st["c_sum"] = st["c_sum"] + hd
                st["c_n"] += 1
            st["d_hist"].append(hd)
            st["d_sum"] = st["d_sum"] + hd
            if len(st["d_hist"]) > routing.broad_window:
                st["d_sum"] = st["d_sum"] - st["d_hist"].popleft()
            moe_i += 1
        return self.lm_head(self.norm_f(x))[0, 0], out


# ------------------------------------------------------------------ configs, data access, guards
def config_for(base: dict, condition: str, alpha: float, beta: float, run_name: str) -> dict:
    import copy
    sched, seg, scale, ref = CONDITIONS[condition]
    cfg = copy.deepcopy(base)
    cfg["condition"] = condition
    cfg["run_name"] = run_name
    cfg["routing"] = {"mode": "multiscale", "gate_norm": "global", "alpha": float(alpha), "beta": float(beta),
                      "schedule": list(SCHEDULES[sched]), "scale_control": bool(scale), "reference": ref,
                      "local_max": 16, "broad_window": 1024, "broad_ramp": 512, "eps": 1e-6, "center": True,
                      "router_bias": False}
    cfg["segmentation"]["strategy"] = seg
    return cfg


def check_matched(configs: dict, approved=APPROVED_DIFF_MAIN):
    from .config import flatten
    flats = {k: flatten(c) for k, c in configs.items()}
    keys = set().union(*[f.keys() for f in flats.values()])
    bad = []
    for k in sorted(keys):
        if k in approved:
            continue
        vals = {c: json.dumps(f.get(k, "<MISSING>"), sort_keys=True) for c, f in flats.items()}
        if len(set(vals.values())) > 1:
            bad.append((k, vals))
    if bad:
        raise ValueError("unmatched config fields: " + "; ".join(f"{k}: {v}" for k, v in bad))
    return True


class TestSplitLocked(RuntimeError):
    pass


def load_split(data_dir, split):
    """The only split loader used by this experiment. Refuses the locked test split before any file access."""
    if str(split) not in ("train", "calib", "val"):
        raise TestSplitLocked(f"split {split!r} refused (locked test split / unknown split)")
    from .data import TokenSplit
    return TokenSplit.load(Path(data_dir), split)


class StartsTable:
    """Chunk starts only (no prefix embeddings are loaded)."""

    def __init__(self, flat, off):
        self.flat, self.off = flat, off

    def starts(self, di):
        return self.flat[self.off[di]:self.off[di + 1]].tolist()

    def chunk_base(self, di):
        return int(self.off[di])


def load_starts(cfg, split, root):
    if str(split) not in ("train", "calib", "val"):
        raise TestSplitLocked(f"split {split!r} refused")
    seg = cfg["segmentation"]
    strat = seg["strategy"]
    if strat == "none":
        return None
    if strat == "random":
        d = Path(root) / seg["random_cache"]
        man = json.loads((d / "manifest.json").read_text())
        if not man.get("accepted") or man["random_cache_id"] != seg["random_cache_id"] or \
                man["spec"]["parent_semantic_cache"] != seg["cache_id"]:
            raise RuntimeError("random cache not accepted or mismatched")
        z = np.load(d / f"{split}_random.npz")
    elif strat in ("fixed", "semantic"):
        d = Path(root) / seg["cache"]
        man = json.loads((d / "manifest.json").read_text())
        if man["cache_id"] != seg["cache_id"]:
            raise RuntimeError("stale/mismatched segmentation cache")
        z = np.load(d / f"{split}_{strat}.npz")
    else:
        raise ValueError(strat)
    return StartsTable(np.asarray(z["starts_flat"]), np.asarray(z["starts_off"]))
