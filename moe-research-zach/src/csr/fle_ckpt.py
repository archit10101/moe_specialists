"""fle checkpoints (copy of lfr_ckpt with fle paths; prereg section 12). Directory checkpoints written atomically: temporary directory in the destination
(.<name>.tmp/) -> state.pt flushed + fsynced -> verified reload (step, tokens, run id, tensor count) -> meta.json with
sha256 + complete marker (fsynced) -> os.replace(dir) -> fsync(parent) -> resume_index.json (atomic) -> ledger.
Rolling checkpoints keep >= KEEP verified; older ones are deleted only after a newer one verifies; all rolling
checkpoints are deleted (ledgered) once the run is COMPLETE. Milestones are never deleted."""
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

from csr import guard
from csr.fle_paths import ART, TMP, STOP_MARKER  # noqa: F401

LEDGER = ART / "checkpoint_ledger.tsv"
DRIVER_STATE = ART / "driver_state.json"
SLOTS = TMP / "slots"
KEEP = 1
HDR = ["time", "run_id", "run_name", "kind", "path", "step", "seen", "sha256", "bytes", "save_seconds", "note"]


def sha256_file(p, bufsize=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def ledger(row):
    new = not LEDGER.exists()
    with open(LEDGER, "a") as f:
        if new:
            f.write("\t".join(HDR) + "\n")
        f.write("\t".join(str(row.get(h, "")) for h in HDR) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _fsync_dir(d):
    fd = os.open(str(d), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_ckpt_dir(obj, final: Path, check, meta_extra=None):
    """Write obj into directory `final` atomically. check(loaded) must be True. Returns (sha256, bytes, seconds)."""
    import torch
    t0 = time.time()
    final = Path(final)
    if final.exists():
        raise guard.GuardError(f"refusing to overwrite checkpoint {final}")
    tmp = final.parent / f".{final.name}.tmp"
    if tmp.exists():  # debris of an interrupted earlier attempt at this step: kept aside, never reused
        os.replace(tmp, final.parent / f".{final.name}.debris-{guard.utc_stamp()}")
    tmp.mkdir(parents=True)
    sp = tmp / "state.pt"
    with open(sp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    sha = sha256_file(sp)
    ld = torch.load(sp, map_location="cpu", weights_only=False, mmap=True)
    if not check(ld):
        shutil.rmtree(tmp)
        raise guard.GuardError(f"checkpoint verification failed for {final}")
    del ld
    meta = {"sha256": sha, "bytes": sp.stat().st_size, "complete": True, "time": guard.now(), **(meta_extra or {})}
    with open(tmp / "meta.json", "w") as f:
        json.dump(meta, f)
        f.flush()
        os.fsync(f.fileno())
    _fsync_dir(tmp)
    os.replace(tmp, final)
    _fsync_dir(final.parent)
    return sha, meta["bytes"], time.time() - t0


def verify_dir(d):
    d = Path(d)
    try:
        meta = json.loads((d / "meta.json").read_text())
        return bool(meta.get("complete")) and sha256_file(d / "state.pt") == meta["sha256"]
    except (OSError, ValueError, KeyError):
        return False


def driver_snapshot():
    try:
        raw = DRIVER_STATE.read_bytes()
        st = json.loads(raw)
        return {"state_sha256": hashlib.sha256(raw).hexdigest(), "active": st.get("active")}
    except Exception as e:  # informational
        return {"state_sha256": None, "error": str(e)[:200]}


class ResumeStore:
    """<SLOTS>/<run_id>/resume_s<step>/ (state.pt + meta.json) + resume_index.json."""

    def __init__(self, run_id: str, run_name: str, root=SLOTS):
        self.dir = Path(root) / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.run_id, self.run_name = run_id, run_name
        self.index_p = self.dir / "resume_index.json"

    def index(self):
        return json.loads(self.index_p.read_text()) if self.index_p.exists() else {"entries": []}

    def save(self, payload, step, seen, note=""):
        final = self.dir / f"resume_s{step:07d}"
        rel = str(final.relative_to(guard.ROOT))
        if final.exists():
            ent = [e for e in self.index()["entries"] if e["path"] == rel]
            if ent and verify_dir(final):
                return ent[-1]
            q = final.parent / f"{final.name}.rejected-{guard.utc_stamp()}"
            os.replace(final, q)
            _fsync_dir(final.parent)
            idx = self.index()
            idx["entries"] = [e for e in idx["entries"] if e["path"] != rel]
            idx.setdefault("rejected", []).append({"path": str(q.relative_to(guard.ROOT)), "step": step,
                                                   "time": guard.now()})
            guard.atomic_write_json(self.index_p, idx, overwrite=True)
            ledger({"time": guard.now(), "run_id": self.run_id, "run_name": self.run_name, "kind": "rejected_moved",
                    "path": str(q.relative_to(guard.ROOT)), "step": step, "seen": seen,
                    "note": "failed verification; moved aside (kept)"})
        sha, nbytes, sec = atomic_ckpt_dir(
            payload, final, lambda ld: ld["st"]["next_step"] == step and ld["st"]["seen"] == seen
            and ld["run_id"] == self.run_id and len(ld["model"]) == len(payload["model"]),
            {"run_id": self.run_id, "step": step, "seen": seen, "note": note})
        ent = {"path": rel, "step": step, "seen": seen, "sha256": sha, "bytes": nbytes, "time": guard.now(),
               "save_seconds": round(sec, 2), "note": note}
        idx = self.index()
        idx["entries"].append(ent)
        guard.atomic_write_json(self.index_p, idx, overwrite=True)
        ledger({"time": ent["time"], "run_id": self.run_id, "run_name": self.run_name, "kind": "resume",
                "path": rel, "step": step, "seen": seen, "sha256": sha, "bytes": nbytes,
                "save_seconds": ent["save_seconds"], "note": note})
        self._rotate()
        return ent

    def _rotate(self):
        idx = self.index()
        ents = idx["entries"]
        if len(ents) <= KEEP:
            return
        keep, drop = ents[-KEEP:], ents[:-KEEP]
        if not verify_dir(guard.ROOT / keep[-1]["path"]):
            return
        idx["entries"] = keep
        guard.atomic_write_json(self.index_p, idx, overwrite=True)
        for e in drop:
            p = guard.ROOT / e["path"]
            if p.exists():
                shutil.rmtree(p)
            ledger({"time": guard.now(), "run_id": self.run_id, "run_name": self.run_name, "kind": "rotated_out",
                    "path": e["path"], "step": e["step"], "seen": e["seen"], "sha256": e["sha256"],
                    "note": f"deleted after newer verified checkpoint step {keep[-1]['step']}"})

    def latest_verified(self, log=print):
        for e in reversed(self.index()["entries"]):
            p = guard.ROOT / e["path"]
            if not p.exists():
                log(f"resume: {e['path']} missing; trying previous")
                continue
            if not verify_dir(p) or json.loads((p / "meta.json").read_text())["sha256"] != e["sha256"]:
                log(f"resume: {e['path']} FAILED sha256 verification; trying previous")
                ledger({"time": guard.now(), "run_id": self.run_id, "run_name": self.run_name,
                        "kind": "verify_failed", "path": e["path"], "step": e["step"], "seen": e["seen"],
                        "sha256": e["sha256"], "note": "skipped at resume"})
                continue
            return e
        return None

    def prune_after_complete(self):
        idx = self.index()
        for e in idx["entries"]:
            p = guard.ROOT / e["path"]
            if p.exists():
                shutil.rmtree(p)
            ledger({"time": guard.now(), "run_id": self.run_id, "run_name": self.run_name,
                    "kind": "pruned_after_complete", "path": e["path"], "step": e["step"], "seen": e["seen"],
                    "sha256": e["sha256"], "note": "run COMPLETE (training + end-of-run eval + summary)"})
        idx["entries"] = []
        idx["pruned_after_complete"] = guard.now()
        guard.atomic_write_json(self.index_p, idx, overwrite=True)


def rng_state():
    import random

    import numpy as np
    import torch
    return {"cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(), "python": random.getstate()}


def set_rng_state(r):
    import random

    import numpy as np
    import torch
    torch.set_rng_state(r["cpu"])
    if r.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(r["cuda"])
    np.random.set_state(r["numpy"])
    random.setstate(r["python"])
