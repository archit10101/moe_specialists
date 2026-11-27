"""Small shared helpers: logging, devices, seeding, atomic json / checkpoint io, run directories."""
from __future__ import annotations

import json
import os
import random
import signal
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from .config import Profile, work_dir


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def device() -> torch.device:
    if os.environ.get("REPRO_DEVICE"):
        return torch.device(os.environ["REPRO_DEVICE"])
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")   # MPS is skipped on purpose: CPU is deterministic and fast enough for smoke


def autocast(dev: torch.device):
    """bf16 autocast on CUDA, plain fp32 elsewhere. REPRO_CPU_AUTOCAST=1 turns on bf16 autocast on CPU too, so
    CPU tests exercise the mixed-precision code paths a GPU run takes (dtype mismatches are GPU-only bugs)."""
    if dev.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if os.environ.get("REPRO_CPU_AUTOCAST") == "1":
        return torch.autocast("cpu", dtype=torch.bfloat16)
    return nullcontext()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=_jsonable))
    os.replace(tmp, path)


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def _jsonable(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, torch.Tensor):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, tuple):
        return list(x)
    raise TypeError(f"not JSON serializable: {type(x)}")


def save_torch(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_torch(path: Path, map_location="cpu"):
    return torch.load(path, map_location=map_location, weights_only=False)


# ---------------------------------------------------------------------------------- directories

def data_dir(p: Profile) -> Path:
    return work_dir(p.name) / "data"


def groups_dir(p: Profile) -> Path:
    return work_dir(p.name) / "groups"


def run_dir(p: Profile, exp_name: str, seed: int) -> Path:
    return work_dir(p.name) / "runs" / exp_name / f"s{seed}"


def results_dir(p: Profile) -> Path:
    return work_dir(p.name) / "results"


def done_marker(d: Path, what: str) -> Path:
    return Path(d) / f"DONE_{what}"


def mark_done(d: Path, what: str) -> None:
    Path(d).mkdir(parents=True, exist_ok=True)
    done_marker(d, what).write_text(time.strftime("%F %T") + "\n")


def is_done(d: Path, what: str) -> bool:
    return done_marker(d, what).exists()


# ---------------------------------------------------------------------------------- preemption

class StopFlag:
    """Set on SIGTERM/SIGUSR1 (SLURM preemption / time limit) so loops can checkpoint and exit."""

    def __init__(self):
        self.stop = False
        for s in (signal.SIGTERM, signal.SIGUSR1):
            try:
                signal.signal(s, self._handler)
            except (ValueError, OSError):  # not in the main thread
                pass

    def _handler(self, signum, frame):
        log(f"received signal {signum}: will checkpoint and exit")
        self.stop = True
