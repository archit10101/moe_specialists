"""Safety interlocks, GPU occupancy guard, resource limits, atomic IO and run directories.

Every executable that can download, load a model, initialize CUDA, train, evaluate or profile calls
`require_execution(args, action)`: without BOTH `--execute` and MOE_RESEARCH_APPROVE=<action> it stays
in dry-run. The variable is an interlock, not permission to exceed the authorized stage.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path("/home/zapan4/moe-research")
LEDGER = ROOT / "ledger"
VRAM_CEILING_GIB = 56.0
VRAM_ABORT_FRACTION = 0.95
DISK_RESERVE_BYTES = 100 * 1024**3  # conservative: 100 GiB
MAX_ATOMIC_SECONDS = 90 * 60

ACTIONS = {"download", "gpu", "cpu-model"}


class GuardError(RuntimeError):
    pass


def now():
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def utc_stamp():
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ------------------------------------------------------------------ root / paths
def check_root():
    cwd = Path.cwd().resolve()
    if cwd != ROOT:
        raise GuardError(f"must run from {ROOT}, cwd={cwd}")
    st = ROOT.stat()
    if ROOT.is_symlink() or st.st_uid != os.getuid() or (st.st_mode & 0o022):
        raise GuardError("project root ownership/permission check failed")
    for var in ("TMPDIR", "HF_HOME", "XDG_CACHE_HOME", "TORCH_HOME", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"):
        v = os.environ.get(var)
        if not v or not Path(v).resolve().is_relative_to(ROOT):
            raise GuardError(f"{var} not inside project root; source ./env.sh")
    os.umask(0o077)


def inside_root(p) -> Path:
    rp = Path(p).resolve()
    if not rp.is_relative_to(ROOT):
        raise GuardError(f"path escapes project root: {p}")
    return rp


def require_execution(execute_flag: bool, action: str) -> bool:
    """True => execute; False => dry run. Raises if --execute is given without the matching interlock."""
    assert action in ACTIONS, action
    if not execute_flag:
        print(f"[dry-run] pass --execute and set MOE_RESEARCH_APPROVE={action} to run this '{action}' action.")
        return False
    if os.environ.get("MOE_RESEARCH_APPROVE") != action:
        raise GuardError(f"--execute given but MOE_RESEARCH_APPROVE != {action!r}; refusing")
    return True


def ledger(event: str, detail: str):
    LEDGER.mkdir(exist_ok=True)
    with open(LEDGER / "safety_ledger.tsv", "a") as f:
        f.write(f"{now()}\t{event}\t{detail}\n")


def gpu_ledger(start, end, run_id, purpose, status):
    with open(LEDGER / "gpu_time_ledger.tsv", "a") as f:
        f.write(f"{start}\t{end}\t{(_dt.datetime.fromisoformat(end) - _dt.datetime.fromisoformat(start)).total_seconds():.0f}"
                f"\t{run_id}\t{purpose}\t{status}\n")


def register_self(purpose: str, status: str = "running"):
    pid = os.getpid()
    try:
        start = subprocess.run(["ps", "-o", "lstart=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    except Exception:
        start = "unknown"
    with open(LEDGER / "process_registry.tsv", "a") as f:
        f.write(f"{now()}\t{pid}\t{os.getppid()}\t{os.getuid()}\t{start}\t{sys.executable}\t{purpose}\t{status}\n")


# ------------------------------------------------------------------ disk
def disk_free_bytes(path=ROOT):
    return shutil.disk_usage(path).free


def require_disk(extra_bytes: int, what: str):
    free = disk_free_bytes()
    if free - extra_bytes < DISK_RESERVE_BYTES:
        raise GuardError(f"disk reserve: free={free} need={extra_bytes} for {what}; reserve={DISK_RESERVE_BYTES}")
    ledger("DISK_CHECK", f"{what}: free={free} planned={extra_bytes} reserve={DISK_RESERVE_BYTES} ok")


# ------------------------------------------------------------------ GPU occupancy (read-only, NVML via nvidia-smi)
def gpu_processes(gpu_index: int = 0):
    """PIDs of compute and graphics processes on the GPU, from `nvidia-smi -q -d PIDS`. Read-only."""
    out = subprocess.run(["nvidia-smi", "-i", str(gpu_index), "-q", "-d", "PIDS"], capture_output=True,
                         text=True, timeout=30)
    if out.returncode != 0:
        raise GuardError(f"nvidia-smi failed rc={out.returncode}")
    pids = []
    for line in out.stdout.splitlines():
        s = line.strip()
        if s.startswith("Process ID"):
            pids.append(int(s.split(":")[1]))
    return pids


def own_pids():
    me = os.getpid()
    pids = {me}
    try:
        kids = subprocess.run(["ps", "-o", "pid=", "--ppid", str(me)], capture_output=True, text=True).stdout.split()
        pids.update(int(k) for k in kids)
    except Exception:
        pass
    return pids


def occupancy_gate(tag: str):
    """Fresh check immediately before CUDA init. Proceed only if GPU 0 has no process."""
    pids = gpu_processes(0)
    if pids:
        ledger("GPU_BLOCKED", f"{tag}: {len(pids)} process(es) present on GPU0; not launching")
        raise GuardError(f"BLOCKED {now()}: GPU0 has {len(pids)} running process(es)")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise GuardError("CUDA_VISIBLE_DEVICES must be exactly '0' for a guarded GPU run")
    ledger("GPU_OCCUPANCY", f"{tag}: gpu0 compute+graphics procs=0 -> proceed")


class OccupancyMonitor(threading.Thread):
    """Polls GPU0 read-only; sets `.contaminated` if a process other than this task's appears."""

    def __init__(self, interval=15.0):
        super().__init__(daemon=True)
        self.interval, self.contaminated, self.detail = interval, False, ""
        self._stop_evt = threading.Event()

    def run(self):
        while not self._stop_evt.wait(self.interval):
            try:
                foreign = [p for p in gpu_processes(0) if p not in own_pids()]
            except Exception as e:  # treat monitoring failure as contamination (fail closed)
                foreign, self.detail = [-1], f"monitor error: {type(e).__name__}"
            if foreign:
                self.contaminated = True
                self.detail = self.detail or f"{len(foreign)} foreign GPU process(es) at {now()}"
                return

    def stop(self):
        self._stop_evt.set()


class RunLimits:
    """Wall-clock and VRAM limits; `check()` raises GuardError (caller saves partial artifacts)."""

    def __init__(self, max_seconds=MAX_ATOMIC_SECONDS, vram_gib=VRAM_CEILING_GIB, monitor=None):
        self.t0, self.max_seconds, self.vram_gib, self.monitor = time.time(), max_seconds, vram_gib, monitor
        self.alarm = False

        def _alarm(signum, frame):
            self.alarm = True
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(int(max_seconds) + 30)  # backstop

    def peak_gib(self):
        import torch
        if not torch.cuda.is_available():
            return 0.0, 0.0
        return torch.cuda.max_memory_allocated() / 1024**3, torch.cuda.max_memory_reserved() / 1024**3

    def check(self):
        if self.alarm or time.time() - self.t0 > self.max_seconds:
            raise GuardError("wall-clock limit reached")
        a, r = self.peak_gib()
        if max(a, r) > VRAM_ABORT_FRACTION * self.vram_gib:
            raise GuardError(f"VRAM near ceiling: alloc={a:.2f} reserved={r:.2f} GiB")
        if self.monitor is not None and self.monitor.contaminated:
            raise GuardError(f"CONTAMINATED: {self.monitor.detail}")

    def close(self):
        signal.alarm(0)


def setup_cuda_device():
    """Call only after occupancy_gate. Caps this process's allocator at the project ceiling."""
    import torch
    assert torch.cuda.is_available(), "CUDA not available"
    assert torch.cuda.device_count() == 1, "exactly one visible GPU expected"
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, VRAM_CEILING_GIB * 1024**3 / total), 0)
    torch.cuda.reset_peak_memory_stats()
    return torch.device("cuda:0")


# ------------------------------------------------------------------ atomic IO / runs
def atomic_write_bytes(path, data: bytes, mode=0o600, overwrite=False):
    path = inside_root(path)
    if path.exists() and not overwrite:
        raise GuardError(f"refusing to overwrite {path}")
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.chmod(tmp, mode)
    os.replace(tmp, path)
    return path


def atomic_write_json(path, obj, overwrite=False, mode=0o600):
    return atomic_write_bytes(path, (json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n").encode(),
                              mode=mode, overwrite=overwrite)


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def sha256_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


class RunDir:
    """runs/<UTC>_<name>/ with immutable effective config, append-only log, metrics, status."""

    def __init__(self, name: str, config: dict, base=None):
        base = Path(os.environ.get("MOE_RUNS_DIR", ROOT / "runs")) if base is None else base
        self.path = inside_root(Path(base) / f"{utc_stamp()}_{name}")
        self.path.mkdir(parents=True, exist_ok=False)
        atomic_write_json(self.path / "config.json", config, mode=0o400)
        self.run_id = self.path.name
        self.set_status("RUNNING")

    def set_status(self, status, **extra):
        atomic_write_json(self.path / "status.json", {"status": status, "time": now(), **extra}, overwrite=True)

    def log(self, msg):
        line = f"{now()} {msg}"
        print(line, flush=True)
        with open(self.path / "log.txt", "a") as f:
            f.write(line + "\n")

    def metric(self, rec: dict):
        with open(self.path / "metrics.jsonl", "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
