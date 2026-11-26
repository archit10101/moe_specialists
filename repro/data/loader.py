"""Token streams, deterministic training batches, evaluation windows and group tables.

A batch is a dict of LongTensors on the target device:
  x    [B, T] input tokens        y    [B, T] next tokens
  doc  [B, T] global paper id of each input token (used to look up domain / cluster groups)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..config import Profile
from ..utils import data_dir, groups_dir, read_json


class Split:
    def __init__(self, p: Profile, name: str):
        d = data_dir(p)
        self.name = name
        self.tokens = np.memmap(d / f"{name}.tokens.bin", dtype=np.uint16, mode="r")
        self.docs = np.memmap(d / f"{name}.docs.bin", dtype=np.int32, mode="r")
        self.ctx = p.model.ctx
        self.n_windows = (len(self.tokens) - 1) // self.ctx

    def windows(self, idx) -> dict:
        idx = np.asarray(idx, dtype=np.int64)
        starts = idx[:, None] * self.ctx + np.arange(self.ctx + 1)[None]
        t = torch.from_numpy(self.tokens[starts].astype(np.int64))
        doc = torch.from_numpy(self.docs[starts[:, :-1]].astype(np.int64))
        return dict(x=t[:, :-1], y=t[:, 1:], doc=doc)


class TrainStream:
    """Non-overlapping windows in a seeded random order; epoch e uses permutation seed+e.
    Batch contents depend only on (seed, step), so every condition sharing a seed sees identical rows
    and resuming from a checkpoint needs nothing but the step number."""

    def __init__(self, split: Split, seed: int, micro_bs: int, accum: int):
        self.split, self.seed, self.micro_bs, self.accum = split, seed, micro_bs, accum
        self._perm_epoch, self._perm = None, None

    def _perm_for(self, epoch: int):
        if self._perm_epoch != epoch:
            self._perm = np.random.default_rng([self.seed, epoch, 7]).permutation(self.split.n_windows)
            self._perm_epoch = epoch
        return self._perm

    def micro_batch(self, step: int, micro: int) -> dict:
        n = self.split.n_windows
        first = (step * self.accum + micro) * self.micro_bs
        idx = []
        for j in range(first, first + self.micro_bs):
            idx.append(self._perm_for(j // n)[j % n])
        return self.split.windows(idx)


def eval_windows(split: Split, max_tokens: int) -> np.ndarray:
    n = min(split.n_windows, max(1, max_tokens // split.ctx))
    return np.arange(n)


def to_device(batch: dict, dev) -> dict:
    return {k: v.to(dev, non_blocking=True) for k, v in batch.items()}


# ============================================================================== groups

class GroupTable:
    """Per-paper group assignment: hard [n_docs] ints and optional soft [n_docs, K] posteriors."""

    def __init__(self, name: str, hard: np.ndarray, soft: np.ndarray | None, k: int):
        self.name, self.k = name, int(k)
        self.hard = torch.from_numpy(hard.astype(np.int64))
        self.soft = None if soft is None else torch.from_numpy(soft.astype(np.float32))
        self._dev = {}

    @classmethod
    def load(cls, p: Profile, name: str) -> "GroupTable":
        path = groups_dir(p) / f"{name}.npz"
        if not path.exists():
            raise SystemExit(f"group table {path} missing: run the `cluster` stage first")
        z = np.load(path)
        return cls(name, z["hard"], z["soft"] if "soft" in z.files else None, int(z["k"]))

    def _on(self, dev):
        key = str(dev)
        if key not in self._dev:
            self._dev[key] = (self.hard.to(dev), None if self.soft is None else self.soft.to(dev))
        return self._dev[key]

    def hard_of(self, doc: torch.Tensor) -> torch.Tensor:
        return self._on(doc.device)[0][doc]

    def weights_of(self, doc: torch.Tensor) -> torch.Tensor:
        """[..., K] group weights: soft posterior if present, else one-hot."""
        hard, soft = self._on(doc.device)
        if soft is not None:
            return soft[doc]
        return torch.nn.functional.one_hot(hard[doc], self.k).float()


def load_meta(p: Profile) -> dict:
    return read_json(data_dir(p) / "meta.json")


def load_docs(p: Profile) -> dict:
    z = np.load(data_dir(p) / "docs.npz")
    return {k: z[k] for k in z.files}
