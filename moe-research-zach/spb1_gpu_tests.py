#!/usr/bin/env python
"""spb1 GPU tests + benchmark (guarded GPU job; TOKEN-4 s0; plan rows / TRAIN rows only; no result is used).
  G-ZERO   zero-initialised SEM/SHUF/GEN banks reproduce the frozen TOKEN-4 per-token CE bit-exactly (bf16 autocast)
  G-HASH   base state hash unchanged after the passes below
  G-BENCH  joint 3-bank training step (32 rows), extraction micro-batch (16 rows, 3 x 32 + 16 forced passes), deployed
           router training step (64 docs) -> projected GPU-hours per job type; peak memory
Writes ART/gpu_tests_<UTC>.json (0400). (G-RESUME runs as separate guarded spb1_train.py test jobs.)"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    if not guard.require_execution(a.execute, "gpu"):
        return
    import numpy as np
    import torch
    from csr.fsp_lib import Split
    from csr.orv1_lib import base_forward, head_ce, load_base, load_plan, make_batch, state_hash, to_dev
    from csr.spb1_lib import FreeRouter, make_banks, rms0, router_loss, token_groups
    from csr import spb1_paths as P
    guard.occupancy_gate("spb1 gpu tests")
    dev = guard.setup_cuda_device()
    guard.register_self("spb1 gpu tests")
    torch.backends.cuda.matmul.allow_tf32 = False
    t_iso = guard.now()
    res = {}
    model, _ = load_base(0, dev)
    h0 = state_hash(model)
    sp = Split("main")
    zl = np.load(P.TABLES / "labels_main.npz")
    ls = torch.as_tensor(zl["sem"].astype(np.int64), device=dev)
    lh = torch.as_tensor(zl["shuf"].astype(np.int64), device=dev)
    steps, _ = load_plan()
    banks = make_banks(0, P.K, 2, dev)
    names = list(banks)
    bt = to_dev(make_batch(steps[0][:16], sp), dev)
    x, hid = base_forward(model, bt)
    tgt = bt["targets"].reshape(-1)
    v = tgt >= 0
    ce_d = head_ce(model, hid, tgt, from_hidden=True)
    xn = rms0(x)
    gs, gh = token_groups(bt, ls, lh)
    grp = {names[0]: gs, names[1]: gh, names[2]: gs}
    z = {}
    with torch.no_grad():
        for c in names:
            delta, _, _ = banks[c](xn, grp[c], v)
            z[c] = bool(torch.equal(head_ce(model, x + delta, tgt)[v], ce_d[v]))
    res["G_ZERO"] = {"pass": all(z.values()), **z}
    # bench training steps
    opts = {c: torch.optim.AdamW(banks[c].parameters(), lr=3e-4, fused=True) for c in names}
    torch.cuda.reset_peak_memory_stats()
    ts = []
    for it in range(1, 7):
        torch.cuda.synchronize()
        t1 = time.time()
        bts = [to_dev(make_batch(steps[it][i:i + 16], sp), dev) for i in (0, 16)]
        nv = sum(int((b["targets"] >= 0).sum()) for b in bts)
        for c in names:
            opts[c].zero_grad(set_to_none=True)
        for b in bts:
            x, hid = base_forward(model, b)
            t_ = b["targets"].reshape(-1)
            vv = t_ >= 0
            with torch.no_grad():
                head_ce(model, hid, t_, from_hidden=True)
            xn = rms0(x)
            g1, g2 = token_groups(b, ls, lh)
            gg = {names[0]: g1, names[1]: g2, names[2]: g1}
            for c in names:
                delta, _, aux = banks[c](xn, gg[c], vv)
                (head_ce(model, x + delta, t_)[vv].sum() / nv + 0.01 * aux).backward()
        for c in names:
            opts[c].step()
        torch.cuda.synchronize()
        ts.append(time.time() - t1)
    st = float(np.median(ts[1:]))
    res["bench_train"] = {"s_per_step": st, "proj_50M_h": st * 1581 / 3600, "proj_245M_h": st * 7748 / 3600,
                          "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30}
    # bench extraction micro-batch (3 x 32 + 16 forced passes)
    torch.cuda.synchronize()
    t1 = time.time()
    with torch.no_grad():
        x, hid = base_forward(model, bt)
        xn = rms0(x)
        for c in names:
            for e in range(32):
                head_ce(model, x + banks[c].one(xn, e), tgt)
        for e in range(16):
            head_ce(model, x + banks[names[0]].one(xn, e), tgt)
    torch.cuda.synchronize()
    te = time.time() - t1
    ntok = int(v.sum())
    res["bench_extract"] = {"s_per_microbatch": te, "tokens": ntok,
                            "proj_partition_h": te * 6.25e6 / ntok / 3600}
    # bench deployed router steps
    r = FreeRouter(64, 0, 32).to(dev)
    opt = torch.optim.AdamW(r.parameters(), lr=3e-3)
    X = torch.randn(64, 1024, 384, device=dev).half()
    dd = torch.randn(64 * 1024, 32, device=dev) * 0.05
    tr = []
    for it in range(12):
        torch.cuda.synchronize()
        t1 = time.time()
        loss, _, _ = router_loss(r(X).reshape(-1, 32), dd, 0.01)
        opt.zero_grad()
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        tr.append(time.time() - t1)
    srt = float(np.median(tr[2:]))
    res["bench_router"] = {"s_per_step_upper": srt, "proj_router_min": srt * P.ROUTER_STEPS / 60}
    res["G_HASH"] = {"pass": state_hash(model) == h0}
    out = {"time": guard.now(), "start": t_iso, "all_pass": res["G_ZERO"]["pass"] and res["G_HASH"]["pass"], "tests": res}
    p = P.ART / f"gpu_tests_{guard.utc_stamp()}.json"
    guard.atomic_write_json(p, out, mode=0o400)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
