"""Optimizer, LR schedule, checkpoint/resume and the time budget shared by every trainer.

Exit codes used by every training stage (the SLURM wrapper relies on them):
  0  finished          3  time budget reached, checkpoint saved (re-run submit_all.sh to resume)
  143 stopped by SIGTERM / preemption, checkpoint saved (SLURM --requeue resumes it)
"""
from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import torch

from ..utils import StopFlag, load_torch, log, save_torch

EXIT_TIME, EXIT_SIGNAL = 3, 143


def make_optimizer(params, lr: float, weight_decay: float, dev, betas=(0.9, 0.95)) -> torch.optim.AdamW:
    params = [p for p in params if p.requires_grad]
    decay = [p for p in params if p.ndim >= 2]
    other = [p for p in params if p.ndim < 2]
    groups = [dict(params=decay, weight_decay=weight_decay), dict(params=other, weight_decay=0.0)]
    return torch.optim.AdamW(groups, lr=lr, betas=tuple(betas), fused=dev.type == "cuda")


def lr_at(step: int, total: int, lr: float, min_lr: float, warmup_frac: float, schedule: str = "cosine",
          decay_frac: float = 0.2, warmup_steps: int | None = None) -> float:
    """Linear warmup, then cosine to min_lr | wsd (constant, linear decay to min_lr over the last
    decay_frac of steps; the original base schedule) | constant (the original bank schedule)."""
    warm = max(1, warmup_steps if warmup_steps is not None else int(warmup_frac * total))
    if step < warm:
        return lr * (step + 1) / warm
    if schedule == "constant":
        return lr
    if schedule == "wsd":
        start = int(total * (1 - decay_frac))
        if step < start:
            return lr
        return lr + (min_lr - lr) * min(1.0, (step - start + 1) / max(1, total - start))
    t = (step - warm) / max(1, total - warm)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * min(1.0, t)))


def set_lr(opt, lr: float) -> None:
    for g in opt.param_groups:
        g["lr"] = lr


class Budget:
    """Wall-clock budget + periodic checkpoints + preemption signal."""

    def __init__(self):
        self.t0 = time.time()
        self.max_s = float(os.environ.get("REPRO_MAX_MINUTES", "340")) * 60
        self.every_s = float(os.environ.get("REPRO_CKPT_MINUTES", "15")) * 60
        self.last = self.t0
        self.flag = StopFlag()

    def should_checkpoint(self) -> bool:
        if time.time() - self.last >= self.every_s:
            self.last = time.time()
            return True
        return False

    def exit_code(self) -> int | None:
        if self.flag.stop:
            return EXIT_SIGNAL
        if time.time() - self.t0 >= self.max_s:
            return EXIT_TIME
        return None


def save_state(path: Path, step: int, modules: dict, opt, extra: dict) -> None:
    save_torch(path, dict(step=step, modules={k: m.state_dict() for k, m in modules.items()},
                          opt=opt.state_dict(), extra=extra,
                          rng=dict(torch=torch.get_rng_state(),
                                   cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)))


def load_state(path: Path, modules: dict, opt) -> tuple[int, dict]:
    st = load_torch(path)
    for k, m in modules.items():
        m.load_state_dict(st["modules"][k])
    opt.load_state_dict(st["opt"])
    torch.set_rng_state(st["rng"]["torch"])
    if st["rng"]["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["rng"]["cuda"])
    log(f"resumed from {path} at step {st['step']}")
    return st["step"], st["extra"]


def stop(code: int, path: Path, step: int, modules: dict, opt, extra: dict) -> None:
    save_state(path, step, modules, opt, extra)
    log(f"checkpoint saved at step {step}; exiting with code {code}")
    sys.exit(code)
