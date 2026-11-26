#!/usr/bin/env python
"""csr-forecast-latent-edl-v1: train ONE generalist draft (prereg section 3): fsp DRAFT_CFG architecture, one epoch
on the balanced union draft_shared + draft_cont (paper-disjoint from main/calib/val/test) in a seed-specific random
order, lr 1e-3 WSD (warm-up 1 %, decay last 20 % to 1e-4), 64x1024 rows, AdamW (0.9, 0.95) wd 0.1, clip 1.0, bf16.
No domain label is read. Resumable: verified rolling slot every 300 s (KEEP=1). Exit 0 COMPLETE, 77 external stop.
GPU only via scripts/gpu_guarded.sh ... --device cuda --execute."""
import argparse
import hashlib
import json
import math
import os
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402


def build_stream(seed, rps=64):
    import numpy as np
    from csr.fsp_lib import ConcatSplit, pack_rows
    cs = ConcatSplit(["draft_shared", "draft_cont"])
    order = np.random.default_rng(1000 + seed).permutation(cs.n_docs).tolist()
    rows = pack_rows(cs.lengths, order)
    steps = [rows[s:s + rps] for s in range(0, len(rows) - rps + 1, rps)]
    return cs, steps, int(sum(cs.lengths[d] - 1 for r in steps for row in [r] for rr in row for d in rr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--run-dir")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    if not guard.require_execution(a.execute, "gpu" if a.device == "cuda" else "cpu-model"):
        return
    import numpy as np
    import torch
    import torch.nn.functional as F
    from csr import fle_ckpt
    from csr.fle_paths import RUN_PREFIX, STOP_MARKER, ART, prereg_sha
    from csr.fsp_lib import DRAFT_CFG, build_draft, draft_hidden, lr_at, make_batch, to_torch
    torch.set_num_threads(4)
    mon = None
    if a.device == "cuda":
        guard.occupancy_gate(f"fle draft seed {a.seed}")
        device = guard.setup_cuda_device()
        mon = guard.OccupancyMonitor(interval=15.0)
        mon.start()
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        device = torch.device("cpu")
    cfg = {"experiment": "csr-forecast-latent-edl-v1", "kind": "draft", "run_name": f"draft_G_s{a.seed}",
           "seed": a.seed, "prereg_sha256": prereg_sha(), "arch": DRAFT_CFG,
           "data": "draft_shared+draft_cont union, one epoch, order rng 1000+seed",
           "train": {"lr": 1e-3, "min_lr": 1e-4, "warmup_frac": 0.01, "decay_frac": 0.2, "betas": [0.9, 0.95],
                     "weight_decay": 0.1, "clip": 1.0, "rows_per_step": 64, "micro_batch_rows": 32}}
    if a.run_dir:
        rdp = guard.inside_root(a.run_dir)
        if json.loads((rdp / "config.json").read_text()) != cfg:
            raise guard.GuardError("resume: config differs")
    else:
        rdp = guard.RunDir(RUN_PREFIX + cfg["run_name"], cfg).path
    guard.register_self(f"fle draft {cfg['run_name']}")
    log = lambda m: (print(f"{guard.now()} {m}", flush=True), open(rdp / "log.txt", "a").write(f"{guard.now()} {m}\n"))
    t = cfg["train"]
    sp, steps, planned = build_stream(a.seed)
    n_steps = len(steps)
    log(f"draft stream: {n_steps} steps, {planned} valid tokens")
    model = build_draft(a.seed).to(device)
    dec, nodec = [], []
    for n, p in model.named_parameters():
        (dec if p.dim() >= 2 else nodec).append(p)
    opt = torch.optim.AdamW([{"params": dec, "weight_decay": t["weight_decay"]}, {"params": nodec, "weight_decay": 0.0}],
                            lr=t["lr"], betas=tuple(t["betas"]), eps=1e-8, fused=device.type == "cuda")
    store = fle_ckpt.ResumeStore(rdp.name, cfg["run_name"])
    st = {"next_step": 0, "seen": 0, "step_s": 0.0}
    ent = store.latest_verified(log=log) if a.run_dir else None
    if ent is not None:
        ck = torch.load(guard.ROOT / ent["path"] / "state.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        st = ck["st"]
        fle_ckpt.set_rng_state(ck["rng"])
        log(f"resumed at step {st['next_step']}")
    fwd = torch.compile(lambda ids, d, p: draft_hidden(model, ids, d, p)) if device.type == "cuda" else \
        (lambda ids, d, p: draft_hidden(model, ids, d, p))
    ce_fn = lambda h, w, y: F.cross_entropy(F.linear(h, w).float(), y, ignore_index=-100, reduction="sum")
    if device.type == "cuda":
        ce_fn = torch.compile(ce_fn)
    reason = None

    def on_sig(signum, frame):
        nonlocal reason
        reason = "signal " + signal.Signals(signum).name
    for s_ in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(s_, on_sig)

    def save(note):
        payload = {"run_id": rdp.name, "model": model.state_dict(), "opt": opt.state_dict(), "st": st,
                   "rng": fle_ckpt.rng_state()}
        store.save(payload, st["next_step"], st["seen"], note=note)
    t_iso, t0p = guard.now(), time.time()
    last = time.time()
    status, rc = "INCOMPLETE", 1
    model.train()
    try:
        while st["next_step"] < n_steps:
            step = st["next_step"]
            if reason is None and STOP_MARKER.exists():
                reason = "stop marker"
            if reason is None and mon is not None and mon.contaminated:
                reason = f"foreign GPU process {mon.detail}"
            if reason is not None:
                status, rc = "EXTERNALLY_STOPPED", 77
                break
            if a.max_steps is not None and step >= a.max_steps:
                status, rc = "INTERRUPTED", 75
                break
            lr = lr_at(step, n_steps, t)
            for g in opt.param_groups:
                g["lr"] = lr
            t0 = time.time()
            rows = steps[step]
            n_valid = int(sum(sp.lengths[d] - 1 for r in rows for d in r))
            tot = 0.0
            for mi in range(2):
                bt = to_torch(make_batch(rows[mi * 32:(mi + 1) * 32], sp), device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    h = fwd(bt["input_ids"], bt["doc_ids"], bt["positions"])
                    hf, y = h.reshape(-1, h.shape[-1]), bt["targets"].reshape(-1)
                    lm = sum(ce_fn(hf[i:i + 8192], model.tok_emb.weight, y[i:i + 8192]) for i in range(0, y.numel(), 8192))
                (lm / n_valid).backward()
                tot += float(lm.detach())
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), t["clip"])
            opt.step()
            opt.zero_grad(set_to_none=True)
            if not math.isfinite(tot):
                raise FloatingPointError("non-finite draft loss")
            st["next_step"] = step + 1
            st["seen"] += n_valid
            st["step_s"] += time.time() - t0
            if step % 100 == 0 or step == n_steps - 1:
                rec = {"step": step, "lr": lr, "lm": tot / n_valid, "gn": float(gn), "seen": st["seen"]}
                open(rdp / "metrics.jsonl", "a").write(json.dumps(rec) + "\n")
                if step % 500 == 0:
                    log(json.dumps(rec))
            if time.time() - last > 300 and st["next_step"] < n_steps:
                save("cadence")
                last = time.time()
        else:
            status, rc = "COMPLETE", 0
        if status != "COMPLETE":
            save(f"{status} {reason}")
    finally:
        if mon is not None:
            mon.stop()
    if status == "COMPLETE":
        p = rdp / "ckpt_final.pt"
        tmp = rdp / ".ckpt_final.pt.tmp"
        with open(tmp, "wb") as f:
            torch.save({"model": model.state_dict(), "cfg": DRAFT_CFG, "seed": a.seed, "steps": n_steps,
                        "seen": st["seen"]}, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
        sha = guard.sha256_file(p)
        guard.atomic_write_json(rdp / "summary.json", {"status": status, "ckpt": str(p.relative_to(guard.ROOT)),
                                                       "sha256": sha, "n_steps": n_steps, "seen": st["seen"],
                                                       "train_step_seconds": st["step_s"],
                                                       "tokens_per_s": st["seen"] / max(1e-9, st["step_s"])},
                                overwrite=True)
        store.prune_after_complete()
    guard.atomic_write_json(rdp / "status.json", {"status": status, "reason": reason, "time": guard.now(),
                                                  "next_step": st["next_step"]}, overwrite=True)
    if a.device == "cuda":
        guard.gpu_ledger(t_iso, guard.now(), rdp.name, f"fle draft {cfg['run_name']}", status)
        with open(ART / "compute_ledger.tsv", "a") as f:
            f.write(f"{t_iso}\t{guard.now()}\t{time.time() - t0p:.1f}\t{rdp.name}\t{cfg['run_name']}\tforecast_draft\t{status}\n")
    log(f"{status} {reason or ''}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
