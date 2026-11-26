"""csr-m2d2-domain-prior-v1: TOKEN-4 with explicit document-domain information (docs/M2D2_DOMAIN_PRIOR_V1_PROTOCOL.md).

New module; CSRModel is imported, never modified. DomainModel subclasses CSRModel, so the backbone is constructed and
initialised exactly as TOKEN-4 for the same seed (global RNG); added parameters come from a dedicated generator and
are created afterwards.

  B0 "token"      : no domain parameters are used (none are created).
  B1 "domain_emb" : x0 = tok_emb(ids) + dom_emb[domain]; ordinary unrestricted top-4 routing.
  B2 "prior"      : B1 + in PRIOR_LAYERS: router_logits = token_logits + alpha[l] * bias[l, domain] (bias init 0,
                    alpha init 0.5). Every token scores all 32 experts and takes its own top-4 (no pool or mask).
  B3 "prior" with shuffled training labels (data-side; same module as B2).
Attention: document-causal, the original dense mask (CSRModel.attention_mask) via the original Attention modules.
"""
from __future__ import annotations

import hashlib

import torch
import torch.nn as nn

from .model import CSRModel, ForwardOut, ModelConfig, rope_cos_sin

EXPERIMENT = "csr-m2d2-domain-prior-v1"
N_DOMAINS = 6
PRIOR_LAYERS = (4, 5, 6, 7)
ALPHA_INIT = 0.5
KINDS = ("token", "domain_emb", "prior")


def added_seed(init_seed: int) -> int:
    return int.from_bytes(hashlib.sha256(f"m2d2dp-v1-added|{init_seed}".encode()).digest()[:4], "little")


class DomainModel(CSRModel):
    def __init__(self, cfg: ModelConfig, kind: str, init_seed: int):
        super().__init__(cfg)  # backbone first: identical to TOKEN-4
        assert kind in KINDS, kind
        self.kind = kind
        self.moe_layer_ids = [i for i, b in enumerate(self.blocks) if b.is_moe]
        if kind in ("domain_emb", "prior"):
            self.dom_emb = nn.Parameter(torch.empty(N_DOMAINS, cfg.d_model))
            g = torch.Generator().manual_seed(added_seed(init_seed))
            with torch.no_grad():
                self.dom_emb.copy_(torch.randn(self.dom_emb.shape, generator=g) * cfg.init_std)
        if kind == "prior":
            self.dom_bias = nn.Parameter(torch.zeros(len(PRIOR_LAYERS), N_DOMAINS, cfg.n_experts))
            self.alpha = nn.Parameter(torch.full((len(PRIOR_LAYERS),), ALPHA_INIT))

    def added_parameter_count(self):
        return sum(p.numel() for n, p in self.named_parameters() if n in ("dom_emb", "dom_bias", "alpha"))

    def backbone_state(self):
        return {k: v for k, v in self.named_parameters() if k not in ("dom_emb", "dom_bias", "alpha")}

    def forward(self, batch, collect=False, prior_off=False, disable=None):
        """batch: input_ids, doc_ids, positions, domain (B, T) (domain -1 on pads). prior_off: drop the router bias
        (diagnostic). disable: optional (layer_id, expert, (B, T) bool token mask) - zero that expert's gate for the
        masked tokens (functional ablation; the expert stays eligible for top-4 selection)."""
        ids, doc_ids, positions = batch["input_ids"], batch["doc_ids"], batch["positions"]
        B, T = ids.shape
        x = self.tok_emb(ids)
        dom = batch["domain"].clamp(min=0) if self.kind != "token" else None
        if self.kind != "token":
            x = x + self.dom_emb[dom].to(x.dtype)
        hd = self.cfg.d_model // self.cfg.n_head
        cos, sin = rope_cos_sin(positions, hd, self.cfg.rope_base, x.dtype)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        mask = self.attention_mask(doc_ids, positions)
        nonpad = doc_ids >= 0
        acc = torch.float64 if x.dtype == torch.float64 else torch.float32
        aux_total = x.new_zeros((), dtype=acc)
        z_total = x.new_zeros((), dtype=acc)
        records = []
        for li, blk in enumerate(self.blocks):
            x = x + blk.attn(blk.norm1(x), cos, sin, mask)
            h = blk.norm2(x)
            if not blk.is_moe:
                x = x + blk.ffn(h)
                continue
            moe = blk.ffn
            hf = h.reshape(B * T, -1)
            tok_logits = moe.router_logits(hf)
            logits = tok_logits
            if self.kind == "prior" and li in PRIOR_LAYERS and not prior_off:
                pi = PRIOR_LAYERS.index(li)
                logits = tok_logits + (self.alpha[pi] * self.dom_bias[pi][dom.reshape(-1)]).to(tok_logits.dtype)
            topi, gates, probs_full = moe.select(logits, None)
            if disable is not None and disable[0] == li:
                _, e_dis, m_dis = disable
                kill = (topi == e_dis) & m_dis.reshape(-1, 1)
                gates = gates.masked_fill(kill, 0.0)
            y = moe.experts(hf, topi, gates) + moe.shared(hf)
            x = x + y.view(B, T, -1)
            npm = nonpad.reshape(-1)
            n = npm.sum().clamp(min=1)
            pr = probs_full[npm]
            counts = torch.bincount(topi[npm].reshape(-1), minlength=moe.E).to(pr.dtype)
            aux_total = aux_total + (moe.E * ((counts / (n * moe.k)) * (pr.sum(0) / n)).sum()).to(acc)
            z_total = z_total + (torch.logsumexp(logits[npm], dim=-1).pow(2).sum() / n).to(acc)
            rec = {"topi": topi.view(B, T, -1), "gates": gates.view(B, T, -1)}
            if collect:
                rec["probs_full"] = probs_full.view(B, T, -1)
                if self.kind == "prior" and li in PRIOR_LAYERS:
                    rec["topi_noprior"] = tok_logits.topk(moe.k, dim=-1).indices.view(B, T, -1)
            records.append(rec)
        return ForwardOut(self.lm_head(self.norm_f(x)), aux_total, z_total, records, None)


def build_domain_model(cfg, device, dtype=torch.float32):
    torch.manual_seed(cfg["init"]["seed"])
    m = DomainModel(ModelConfig.from_dict(cfg["model"]), cfg["domain"]["kind"], cfg["init"]["seed"])
    return m.to(device=device, dtype=dtype)
