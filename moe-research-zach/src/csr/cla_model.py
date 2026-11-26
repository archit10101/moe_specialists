"""csr-conditional-local-attention-v1: TOKEN-4 MoE with full / local / hybrid / conditional-global attention
(docs/CONDITIONAL_LOCAL_ATTENTION_V1_PREREG.md).

New module; CSRModel is unchanged. CLAModel subclasses CSRModel, so the backbone (embedding, Q/K/V and output
projections, RMSNorms, dense layer 0, routers, experts, shared experts, LM head) is constructed and initialised exactly
as TOKEN-4 for the same seed. The MoE computation, routing (unrestricted fp32 top-4, global-softmax gates, dropless),
load-balance and z losses are the CSRModel token-mode code path, unchanged. Only the attention computation differs.

Every condition (including the FULL-TOKEN-4 baseline) carries the same per-layer gate MLP (parameter control); it is
created after the backbone and initialised from a dedicated generator, so it never changes the backbone
initialisation. Conditions that do not use the gate never compute it (no gradient; AdamW skips it).

Per-layer attention modes: "full" (document-causal), "local" (document-causal, window W), "cond" (conditional):
  local_t = LocalAttention_W(q_t)                                  (shared Q/K/V/output projections)
  z_t     = MLP_l(rms(stopgrad(local_t)))                          (fp32; local_t = pre-projection head outputs)
  elig_t  = non-pad and in-document position >= W                  (otherwise full == local exactly)
  sel_t   = elig_t and z_t > lambda_l                               (hard gate; lambda_l = frozen-at-eval threshold)
  out_t   = full_t if sel_t else local_t;   full_t computed only for selected queries (compact kernel)
  Straight-through gradient: out_t += (p_t - sg(p_t)) * (full_t - local_t), p_t = sigmoid(z_t - lambda_l), applied
  wherever full_t is available: for selected tokens on ordinary steps and for every eligible token on counterfactual
  steps (where full_t is computed densely for all tokens). Forward values are exactly the hard decisions.
  Budget controller (deterministic, frozen before any result): lambda_l is the Lagrange multiplier / threshold of the
  rate constraint. Initialised on the first training micro-batch (no-grad pre-pass) to the exact quantile giving the
  target rate; after every optimizer step lambda_l <- lambda_l + 0.1 * (q_l - lambda_l), q_l = the threshold that
  would have selected exactly round(target * non-pad tokens of the step) eligible tokens of that step (dual update).
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cla_attn import AttnContext
from .model import CSRModel, ForwardOut, ModelConfig, apply_rope, rope_cos_sin

EXPERIMENT = "csr-conditional-local-attention-v1"
KINDS = ("full", "local", "hybrid", "conditional")


@dataclass(frozen=True)
class AttnPlan:
    kind: str = "full"
    window: int = 0  # local window (local / hybrid / conditional)
    n_full_late: int = 0  # hybrid: number of final layers with full attention
    target_rate: float = 0.0  # conditional: global-query rate target (fraction of non-pad tokens)
    cf_frac: float = 0.10  # conditional: fraction of training steps with the dense counterfactual
    gate_hidden: int = 32
    thr_ema: float = 0.10
    force: str = ""  # "" | "local" | "global": diagnostic forcing of the gate (benchmarks only)

    def __post_init__(self):
        assert self.kind in KINDS, self.kind
        assert self.force in ("", "local", "global"), self.force
        if self.kind != "full":
            assert self.window > 0
        if self.kind == "hybrid":
            assert self.n_full_late > 0
        if self.kind == "conditional":
            assert 0.0 < self.target_rate < 1.0

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_cfg(cls, d):
        return cls(**d)

    def layer_modes(self, n_layer):
        if self.kind == "full":
            return [("full", None)] * n_layer
        if self.kind == "local":
            return [("local", self.window)] * n_layer
        if self.kind == "hybrid":
            return [("full", None) if i >= n_layer - self.n_full_late else ("local", self.window)
                    for i in range(n_layer)]
        return [("cond", self.window)] * n_layer


class GateMLP(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.fc1 = nn.Linear(d, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, 1, bias=True)

    def forward(self, u):  # u (..., d) fp32
        return self.fc2(F.silu(self.fc1(u))).squeeze(-1)


def gate_seed(init_seed: int) -> int:
    return int.from_bytes(hashlib.sha256(f"clav1-gate-init|{init_seed}".encode()).digest()[:4], "little")


def cf_steps(seed: int, n_steps: int, frac: float):
    """Exactly round(frac * n_steps) counterfactual steps, chosen by a deterministic hash of (seed, step)."""
    n_cf = int(round(frac * n_steps))
    keyed = sorted(range(n_steps), key=lambda s: hashlib.sha256(f"clav1-cf|{seed}|{s}".encode()).digest())
    return frozenset(keyed[:n_cf])


class CLAModel(CSRModel):
    def __init__(self, cfg: ModelConfig, plan: AttnPlan, init_seed: int, backend: str = "dense"):
        super().__init__(cfg)  # backbone: identical construction / initialisation to TOKEN-4 (global RNG)
        self.plan = plan
        self.backend = backend
        self.modes = plan.layer_modes(cfg.n_layer)
        self.gates = nn.ModuleList([GateMLP(cfg.d_model, plan.gate_hidden) for _ in range(cfg.n_layer)])
        g = torch.Generator().manual_seed(gate_seed(init_seed))
        with torch.no_grad():
            for gm in self.gates:
                for lin in (gm.fc1, gm.fc2):
                    lin.weight.copy_(torch.randn(lin.weight.shape, generator=g) * cfg.init_std)
                    lin.bias.zero_()
        self.register_buffer("thr", torch.zeros(cfg.n_layer, dtype=torch.float32))
        self.register_buffer("thr_ready", torch.zeros((), dtype=torch.bool))
        self.hd = cfg.d_model // cfg.n_head

    # ------------------------------------------------------------ bookkeeping
    def gate_parameter_names(self):
        return sorted(n for n, _ in self.named_parameters() if n.startswith("gates."))

    def gate_parameter_count(self):
        return sum(p.numel() for n, p in self.named_parameters() if n.startswith("gates."))

    def backbone_state(self):
        return {k: v for k, v in self.state_dict().items() if not k.startswith(("gates.", "thr"))}

    def uses_gate(self):
        return self.plan.kind == "conditional"

    # ------------------------------------------------------------ forward
    def _qkv(self, blk, h, cos, sin):
        B, T, D = h.shape
        q, k, v = blk.attn.qkv(h).view(B, T, 3, self.cfg.n_head, self.hd).permute(2, 0, 3, 1, 4)
        return apply_rope(q, cos, sin), apply_rope(k, cos, sin), v

    def gate_logits(self, li, y_loc):
        with torch.autocast(device_type=y_loc.device.type, enabled=False):
            u = y_loc.detach().to(self.gates[li].fc1.weight.dtype)
            u = u * torch.rsqrt(u.pow(2).mean(-1, keepdim=True) + 1e-6)
            return self.gates[li](u)

    def forward(self, batch, cf: bool = False, calibrate: bool = False, collect: bool = False, return_kv=False):
        """batch: input_ids, doc_ids, positions (B, T). cf: counterfactual training step (dense full attention for all
        tokens, forward = hard decisions). calibrate: set lambda_l to the exact per-micro-batch quantile (first-step
        pre-pass only). Returns ForwardOut with .gate (per layer dicts) and optionally .kv."""
        ids, doc_ids, positions = batch["input_ids"], batch["doc_ids"], batch["positions"]
        B, T = ids.shape
        x = self.tok_emb(ids)
        cos, sin = rope_cos_sin(positions, self.hd, self.cfg.rope_base, x.dtype)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        ctx = AttnContext(doc_ids, self.backend)
        nonpad = doc_ids >= 0
        acc = torch.float64 if x.dtype == torch.float64 else torch.float32
        aux_total = x.new_zeros((), dtype=acc)
        z_total = x.new_zeros((), dtype=acc)
        records, gate_info, kvs = [], [], []
        train_st = torch.is_grad_enabled() and self.training
        for li, blk in enumerate(self.blocks):
            q, k, v = self._qkv(blk, blk.norm1(x), cos, sin)
            if return_kv:
                kvs.append((k, v))
            mode, W = self.modes[li]
            if mode == "full":
                y = ctx.attend(q, k, v, None)
            elif mode == "local":
                y = ctx.attend(q, k, v, W)
            else:
                y, gi = self._conditional(li, ctx, q, k, v, W, positions, nonpad, cf, calibrate, train_st, collect)
                gate_info.append(gi)
            x = x + blk.attn.proj(y)
            h = blk.norm2(x)
            if not blk.is_moe:
                x = x + blk.ffn(h)
                continue
            # ---- MoE: CSRModel token-mode code path, unchanged
            moe = blk.ffn
            y2, info = moe(h.reshape(B * T, -1), None, "global")
            x = x + y2.view(B, T, -1)
            npm = nonpad.reshape(-1)
            n = npm.sum().clamp(min=1)
            probs = info["probs_full"][npm]
            counts = torch.bincount(info["topi"][npm].reshape(-1), minlength=moe.E).to(probs.dtype)
            P = probs.sum(0) / n
            f = counts / (n * moe.k)
            aux_total = aux_total + (moe.E * (f * P).sum()).to(acc)
            z_total = z_total + (torch.logsumexp(info["logits"][npm], dim=-1).pow(2).sum() / n).to(acc)
            records.append({"topi": info["topi"].view(B, T, -1), "gates": info["gates"].view(B, T, -1),
                            "probs_full": info["probs_full"].view(B, T, -1) if collect else None})
        logits = self.lm_head(self.norm_f(x))
        out = ForwardOut(logits, aux_total, z_total, records, None)
        out.gate = gate_info
        out.kv = kvs if return_kv else None
        return out

    def _conditional(self, li, ctx, q, k, v, W, positions, nonpad, cf, calibrate, train_st, collect):
        B, H, T, hd = q.shape
        y_loc = ctx.attend(q, k, v, W)  # (B, T, D)
        z = self.gate_logits(li, y_loc)  # (B, T) fp32
        elig = nonpad & (positions >= W)
        if calibrate:
            with torch.no_grad():
                self.thr[li] = quantile_threshold(z[elig], int(round(self.plan.target_rate * int(nonpad.sum()))))
        thr = self.thr[li]
        if self.plan.force == "global":
            sel = elig
        elif self.plan.force == "local":
            sel = torch.zeros_like(elig)
        else:
            sel = elig & (z > thr)
        p = torch.sigmoid(z - thr) if train_st else None
        if cf:
            y_full = ctx.attend(q, k, v, None)
            y = torch.where(sel.unsqueeze(-1), y_full, y_loc)
            if train_st:
                st = ((p - p.detach()) * elig.to(p.dtype)).unsqueeze(-1).to(y.dtype)
                y = y + st * (y_full - y_loc)
        else:
            qpos, order, y_c = ctx.attend_compact(q, k, v, sel)  # y_c (P, D) rows for qpos (-1 = empty slot)
            D = y_loc.shape[-1]
            yl = y_loc.reshape(B * T, D)
            if train_st:  # gather with the unique-index permutation; empty-slot rows are discarded below
                pg = p.reshape(-1).index_select(0, order)
                st = (pg - pg.detach()).unsqueeze(-1).to(y_c.dtype)
                y_c = y_c + st * (y_c - yl.index_select(0, order))
            dst = torch.where(qpos >= 0, qpos, torch.full_like(qpos, B * T))
            ext = torch.cat([yl, yl.new_zeros(1, D)], 0).index_copy(0, dst, y_c.to(yl.dtype))
            y = ext[:B * T].view(B, T, D)
        gi = {"z": z.detach(), "elig": elig, "sel": sel}
        return y, gi

    # ------------------------------------------------------------ budget controller
    @torch.no_grad()
    def thr_update(self, z_lists, n_nonpad_step):
        """z_lists[layer] = list of eligible-token gate logits of every micro-batch of the step. Returns step rates."""
        rates = []
        k = int(round(self.plan.target_rate * n_nonpad_step))
        for li, zs in enumerate(z_lists):
            zc = torch.cat(zs) if zs else self.thr.new_zeros(0)
            q = quantile_threshold(zc, k)
            self.thr[li] += self.plan.thr_ema * (q - self.thr[li])
            rates.append(float((zc > self.thr[li]).sum()) / max(n_nonpad_step, 1))
        return rates


def quantile_threshold(z_elig: torch.Tensor, k: int):
    """Threshold t such that exactly k of z_elig satisfy z > t (ties aside): midpoint between the k-th and (k+1)-th
    largest values; k <= 0 -> above the max; k >= n -> below the min."""
    n = z_elig.numel()
    if n == 0:
        return torch.zeros((), device=z_elig.device)
    zs = torch.sort(z_elig.float(), descending=True).values
    if k <= 0:
        return zs[0] + 1e-3
    if k >= n:
        return zs[-1] - 1e-3
    return 0.5 * (zs[k - 1] + zs[k])


def build_cla_model(cfg: dict, device, backend=None, dtype=torch.float32, plan_override=None):
    torch.manual_seed(cfg["init"]["seed"])
    plan = plan_override or AttnPlan.from_cfg(cfg["attention"])
    be = backend or ("flex" if torch.device(device).type == "cuda" else "dense")
    model = CLAModel(ModelConfig.from_dict(cfg["model"]), plan, cfg["init"]["seed"], be)
    return model.to(device=device, dtype=dtype)


def count_cla_parameters(model: CLAModel):
    from .model import count_parameters
    c = count_parameters(model)
    g = model.gate_parameter_count()
    c.update({"gate": g, "gate_fraction": g / c["total"], "backbone_total": c["total"] - g,
              "active_per_token_excluding_gate": c["active_per_token"] - g,
              "gate_used": model.uses_gate()})
    return c


def attention_flops_per_token(cfg: ModelConfig, plan: AttnPlan, keys_per_token_by_layer):
    """Analytic forward attention FLOPs per token per layer (multiply-add = 2) from MEASURED mean keys attended per
    token (local keys for every token + full keys for selected tokens): QKV/out projections + QK^T + AV + gate MLP."""
    d = cfg.d_model
    proj = 2 * 4 * d * d
    gate = 2 * (d * plan.gate_hidden + plan.gate_hidden + plan.gate_hidden)
    return [proj + 4 * d * kpt + (gate if mode == "cond" else 0)
            for (mode, _), kpt in zip(plan.layer_modes(cfg.n_layer), keys_per_token_by_layer)]


def fmt_rate(x):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"
