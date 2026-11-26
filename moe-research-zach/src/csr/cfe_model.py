"""csr-context-prior-fast-engineering (cfe) model: CPSModel (V2) plus engineering knobs. With every new knob at its
default the computation is identical to CPSModel (tested in cfe_tests.py).

Knobs (one conceptual component each):
  alpha_min  gate floor: alpha_l(t) = alpha_min + (alpha_max - alpha_min) * sigmoid(b_l + w_l.c_t + v_l.RMSNorm(h_t)).
             Diagnosis 2026-11-13 (diagnosis_inert_*.json): the LM gradient on the V2 gate bias is ~1e-5 and is
             partly cancelled by the aux / z terms, so alpha never leaves its init (~0.01) and the prior never
             co-adapts; forcing alpha up on a trained V2 model changes 26% / 46% of top-4 routing slots at
             alpha 0.1 / 0.2 for only +0.001 / +0.006 nats. The floor keeps the prior in the routing from step 0 so
             that the prior and the experts can co-adapt; the LM objective can still raise alpha above the floor.
The token router is never removed or masked; no expert is ever masked or mandated; drafts only in teacher phases.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import torch

from .cps_model import CPSConfig, CPSModel


@dataclass
class CFEConfig(CPSConfig):
    alpha_min: float = 0.0

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


class CFEModel(CPSModel):
    def __init__(self, ccfg: CFEConfig, seed: int, backbone: dict | None = None):
        super().__init__(ccfg, seed, backbone)

    def alpha(self, j, c, h32):
        cc = self.ccfg
        hn = h32 * torch.rsqrt(h32.pow(2).mean(-1, keepdim=True) + 1e-6)
        a = self.cp_ab[j] + c @ self.cp_aw[j].to(c.dtype) + hn @ self.cp_av[j].to(hn.dtype)
        if cc.alpha_min > 0:
            return cc.alpha_min + (cc.alpha_max - cc.alpha_min) * torch.sigmoid(a)
        return cc.alpha_max * torch.sigmoid(a)

    def strip_teacher(self):
        for n in ("tc_agg", "tc_enc"):
            if hasattr(self, n):
                delattr(self, n)
        self.ccfg = CFEConfig.from_dict({**self.ccfg.to_dict(), "teacher": False})
        return self
