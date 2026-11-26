#!/usr/bin/env python
"""csr-forecast-counterfactual-utility-v1: equal-compute counterfactual expert-utility dataset (prereg 2.1-2.2).

For every frozen Phase-1 region (CALIB-SEARCH, s in {256,512,768}) and every (MoE layer 5..10, expert) cell, the
exact counterfactual utility u = CE(replace e by the next-ranked expert on region tokens where e is selected) -
CE(normal route), region-level and token-level, plus the causal-past model features H_model (final hidden at s-1 and
mean over [s-192, s)). fp32, TF32 off. Chunked (atomic, sha256), resumable, external-stop aware.

  fcu_driver.py job utility_T4_s0 --gpu -- fcu_utility.py --model T4_s0
  fcu_driver.py job utility_T4_s1 --gpu -- fcu_utility.py --model T4_s1 --half 1 --no-tokens   (replication)
"""
import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402

CHUNK = 256
BATCH = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="T4_s0")
    ap.add_argument("--half", type=int, default=-1)
    ap.add_argument("--no-tokens", action="store_true")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    if not guard.require_execution(a.execute, "gpu" if a.device == "cuda" else "cpu-model"):
        return
    import numpy as np
    import torch
    from csr.fcu_lib import ART, LAYERS, STOP_MARKER, UtilityEngine, build_rows, load_model, prereg_sha, regions
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    mon = None
    if a.device == "cuda":
        guard.occupancy_gate(f"fcu utility {a.model}")
        device = guard.setup_cuda_device()
        mon = guard.OccupancyMonitor(interval=15.0)
        mon.start()
    else:
        device = torch.device("cpu")
    guard.register_self(f"fcu utility {a.model}")
    reason = None

    def on_sig(signum, frame):
        nonlocal reason
        reason = "signal " + signal.Signals(signum).name
    for s_ in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(s_, on_sig)
    reg, sp, _ = regions()
    idx_all = np.arange(len(reg["s"]))
    if a.half >= 0:
        idx_all = idx_all[reg["half"] == a.half]
    # chunks: consecutive regions of the same s
    chunks = []
    for s in sorted(set(reg["s"][idx_all].tolist())):
        ii = idx_all[reg["s"][idx_all] == s]
        for c in range(0, len(ii), CHUNK):
            chunks.append((int(s), ii[c:c + CHUNK]))
    name = a.model + ("" if a.half < 0 else f"_half{a.half}")
    od = ART / "utility" / name
    od.mkdir(parents=True, exist_ok=True)
    cfg = {"model": a.model, "half": a.half, "layers": LAYERS, "chunk": CHUNK, "batch": BATCH,
           "tokens": not a.no_tokens, "n_regions": int(len(idx_all)), "prereg_sha256": prereg_sha(),
           "dtype": "fp32, tf32 off"}
    man_p = od / "manifest.json"
    man = json.loads(man_p.read_text()) if man_p.exists() else {"config": cfg, "chunks": [], "complete": False,
                                                                 "created": guard.now()}
    if man["config"] != json.loads(json.dumps(cfg)):
        raise guard.GuardError("utility config differs from existing manifest")
    if man.get("complete"):
        print("already complete")
        return
    per_region = (len(LAYERS) * 32 * 255 * 2 if not a.no_tokens else 0) + 8000
    guard.require_disk(int(len(idx_all) * per_region * 1.2), f"fcu utility {name}")
    model = load_model(a.model, device)
    eng = UtilityEngine(model)
    done = {c["index"] for c in man["chunks"]}
    t_iso, t0 = guard.now(), time.time()
    status = "COMPLETE"
    for ci, (s, ii) in enumerate(chunks):
        if ci in done:
            c = [c for c in man["chunks"] if c["index"] == ci][0]
            if guard.sha256_file(od / c["file"]) != c["sha256"]:
                raise guard.GuardError(f"chunk {ci} fails verification")
            continue
        if reason is None and STOP_MARKER.exists():
            reason = "stop marker"
        if reason is None and mon is not None and mon.contaminated:
            reason = f"foreign GPU process {mon.detail}"
        if reason is not None:
            status = "EXTERNALLY_STOPPED"
            break
        parts = []
        for b in range(0, len(ii), BATCH):
            bi = ii[b:b + BATCH]
            ids = torch.as_tensor(build_rows(sp, reg["gdoc"][bi], s), device=device)
            ends = torch.as_tensor(reg["end"][bi], device=device)
            st = eng.normal(ids, s, ends)
            parts.append(eng.utilities(st, max_pairs=BATCH * 32, store_tokens=not a.no_tokens))
            del st
        arr = {k: np.concatenate([p[k] for p in parts], 0) for k in parts[0]}
        arr["ridx"] = ii.astype(np.int64)
        fn = f"chunk_{ci:04d}.npz"
        tmp = od / f".{fn}.tmp.npz"
        with open(tmp, "wb") as f:
            np.savez(f, **arr)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, od / fn)
        sha = guard.sha256_file(od / fn)
        man["chunks"].append({"index": ci, "file": fn, "sha256": sha, "s": s, "n": int(len(ii)),
                              "bytes": (od / fn).stat().st_size, "time": guard.now()})
        man["chunks"].sort(key=lambda c: c["index"])
        guard.atomic_write_json(man_p, man, overwrite=True)
        with open(ART / "artifact_hashes.tsv", "a") as f:
            f.write(f"{guard.now()}\t{(od / fn).relative_to(guard.ROOT)}\t{sha}\t{(od / fn).stat().st_size}\n")
        print(f"{guard.now()} chunk {ci + 1}/{len(chunks)} s={s} n={len(ii)} {time.time() - t0:.0f}s", flush=True)
    if mon is not None:
        mon.stop()
        if mon.contaminated and status == "COMPLETE":
            status = "EXTERNALLY_STOPPED"
    if status == "COMPLETE":
        man["complete"] = True
        man["completed"] = guard.now()
        man["seconds"] = time.time() - t0
        guard.atomic_write_json(man_p, man, overwrite=True)
    sec = time.time() - t0
    if a.device == "cuda":
        guard.gpu_ledger(t_iso, guard.now(), f"fcu_utility_{name}", "fcu utility", status)
        with open(ART / "compute_ledger.tsv", "a") as f:
            f.write(f"{guard.now()}\tutility_{name}\tcounterfactual_utility\t{sec:.1f}\t{status}\t{reason or ''}\n")
    print(json.dumps({"status": status, "seconds": sec, "reason": reason}))
    sys.exit(0 if status == "COMPLETE" else 77)


if __name__ == "__main__":
    main()
