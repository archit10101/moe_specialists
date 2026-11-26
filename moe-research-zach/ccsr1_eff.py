#!/usr/bin/env python
"""ccsr1 prereg section 11 efficiency study (amendment section E) for the frozen Stage-2 candidate (step-318 ckpt).

  ccsr1_eff.py ce        (CPU) refresh every r in {1,4,8,16}: CE, route agreement with r = 1, stu - G per router seed
                         (exact CS2 table lookup; the specialist is re-chosen only where pos % r == 0, held between)
  ccsr1_eff.py speed     (guarded GPU) measured prefill (64 x 1024-token CS2 rows) for base / base + G / base + causal
                         router + specialist (each r), one-token decode steps of the added components (batch 1 and 64),
                         and top CUDA kernels. The LFRModel base has no KV-cache decoding path, so the base decode
                         reference is a full-recompute step (upper bound), reported as such."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402


def held(e, pos, r):
    import numpy as np
    t = np.arange(len(e))
    return e[t - (pos.astype(np.int64) % r)]


def ce_study():
    import numpy as np
    from csr.ccsr1_paths import ART, ROUTER_SEEDS, ROUTERS
    from csr.ccsr1_s3_paths import EFF, REFRESH, SELECTED
    from ccsr1_analyze import ce_of, load_table
    z, _, meta = load_table("CS2")
    cb = z["ce_base"].astype(np.float64)
    ceG = (cb + z["dce_G"].astype(np.float64)).mean()
    pos = z["pos"]
    doc = z["doc"]
    same = doc[1:] == doc[:-1]
    out = {"time": guard.now(), "table_sha256": meta["sha256"], "ce_G": ceG, "ce_base": cb.mean(), "seeds": {}}
    for s in ROUTER_SEEDS:
        rec = {}
        for b in ("SEM", "SHUF"):
            e = np.load(ROUTERS / "S2" / SELECTED / f"{b}_s{s}_CS2.npz")["e"].astype(np.int64)
            for r in REFRESH:
                eh = held(e, pos, r)
                ce = ce_of(z, b, eh).mean()
                rec[f"{b}_r{r}"] = {"ce": ce, "minus_G": ce - ceG, "agree_r1": float((eh == e).mean()),
                                    "switch_rate": float((eh[1:] != eh[:-1])[same].mean())}
        out["seeds"][s] = rec
        print(s, json.dumps({k: round(v["minus_G"], 5) for k, v in rec.items() if k.startswith("SEM")}), flush=True)
    out["mean"] = {k: {m: float(np.mean([out["seeds"][s][k][m] for s in ROUTER_SEEDS])) for m in ("ce", "minus_G", "agree_r1", "switch_rate")}
                   for k in out["seeds"][ROUTER_SEEDS[0]]}
    EFF.mkdir(parents=True, exist_ok=True)
    guard.atomic_write_json(EFF / "refresh_ce.json", out, overwrite=True)
    print(json.dumps(out["mean"], indent=0))


def speed_study(a):
    import numpy as np
    import torch
    from csr.ccsr1_paths import CKPT
    from csr.ccsr1_router import CausalRouter
    from csr.ccsr1_s3_paths import EFF, REFRESH, SELECTED
    from csr.ccsr1_paths import ROUTERS, WIN
    from csr.jsr_bank import rms0
    from csr.orv1_lib import make_batch, to_dev
    from fle_evaluate import load_milestone
    from ccsr1_extract import load_banks, partition, verify_reused
    from ccsr1_s3_train import doc_batch
    verify_reused()
    guard.occupancy_gate("ccsr1 eff speed")
    mon = guard.OccupancyMonitor(interval=15.0)
    mon.start()
    guard.register_self("ccsr1 eff speed")
    dev = guard.setup_cuda_device()
    torch.backends.cuda.matmul.allow_tf32 = True
    model, ck = load_milestone(CKPT, dev)
    for p in model.parameters():
        p.requires_grad_(False)
    W = model.tok_emb.weight.detach()
    banks = load_banks(ck, dev)
    G, L = banks["G"], banks["SEM"]
    st = torch.load(ROUTERS / "S1" / SELECTED / "SEM_s0.pt", map_location="cpu", weights_only=False)
    R = CausalRouter(int(st["cfg"]["H"]), 0, "base", 16).to(dev).eval()
    R.load_state_dict(st["state"])
    sp, rows, _, _, _ = partition("CS2")
    bt = to_dev(make_batch(rows[:64], sp), dev)
    inp = {"input_ids": bt["input_ids"], "doc_ids": bt["doc_ids"], "positions": bt["positions"]}
    dm = bt["doc_ids"].reshape(-1) >= 0
    gd = bt["gdoc"].reshape(-1)[dm].long()
    pos = bt["positions"].reshape(-1)[dm].long()
    ntok = int(dm.sum())
    ac = lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    def timeit(fn, reps=a.reps, warm=3):
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        return float(np.median(ts)), float(np.percentile(ts, 10)), float(np.percentile(ts, 90))

    def hidden():
        with ac():
            o = model(inp)
        return o["hidden"].reshape(-1, o["hidden"].shape[-1])[dm]

    def v_base():
        with torch.no_grad():
            h = hidden()
            with ac():
                return torch.nn.functional.linear(h, W)

    def v_G():
        with torch.no_grad():
            h = hidden()
            xn = rms0(h)
            e = G.probs(xn).argmax(1)
            a_ = G.correction(xn, e, torch.ones_like(e, dtype=torch.float32))
            with ac():
                return torch.nn.functional.linear(h + a_, W)

    def v_R(r):
        def f():
            with torch.no_grad():
                h = hidden()
                xn = rms0(h)
                idx, mask, lens = doc_batch(gd, pos)
                lg = R(xn.half()[idx], lens)[mask]
                e = torch.empty(ntok, dtype=torch.long, device=dev)
                e[idx[mask]] = lg.argmax(1)
                if r > 1:
                    e = e[torch.arange(ntok, device=dev) - pos % r]
                a_ = L.correction(xn, e)
                with ac():
                    return torch.nn.functional.linear(h + a_, W)
        return f

    res = {"time": guard.now(), "gpu": torch.cuda.get_device_name(), "prefill_tokens": ntok, "prefill_ms": {}, "decode_ms": {}}
    res["prefill_ms"]["base"] = timeit(v_base)
    res["prefill_ms"]["base+G"] = timeit(v_G)
    for r in REFRESH:
        res["prefill_ms"][f"base+router+spec_r{r}"] = timeit(v_R(r))
    b0 = res["prefill_ms"]["base"][0]
    res["prefill_rel_to_base"] = {k: v[0] / b0 for k, v in res["prefill_ms"].items()}
    res["prefill_tok_per_s"] = {k: ntok / (v[0] / 1000) for k, v in res["prefill_ms"].items()}
    # ---- one-token decode steps of the added components (batch B), averaged over 16 consecutive steps
    Hd = R.H
    for B in (1, 64):
        x = torch.randn(B, 16, 384, device=dev)
        h = torch.randn(B, 16, 384, device=dev)

        def d_G():
            with torch.no_grad():
                for t in range(16):
                    xn = x[:, t]
                    e = G.probs(xn).argmax(1)
                    a_ = G.correction(xn, e, torch.ones_like(e, dtype=torch.float32))
                    with ac():
                        torch.nn.functional.linear(h[:, t] + a_, W)

        def d_R(r):
            def f():
                with torch.no_grad():
                    ring = torch.zeros(B, WIN, 384, device=dev)
                    state = torch.zeros(1, B, Hd, device=dev)
                    e = torch.zeros(B, dtype=torch.long, device=dev)
                    for t in range(16):
                        xt = x[:, t]
                        ring[:, t % WIN] = xt
                        m = ring[:, :min(t + 1, WIN)].mean(1)
                        z_ = R.inp(torch.cat([xt, m], -1)).unsqueeze(1)
                        s_, state = R.gru(z_, state)
                        if t % r == 0:
                            e = R.head(s_[:, 0]).argmax(1)
                        a_ = L.correction(xt, e)
                        with ac():
                            torch.nn.functional.linear(h[:, t] + a_, W)
            return f

        def d_head():
            with torch.no_grad():
                for t in range(16):
                    with ac():
                        torch.nn.functional.linear(h[:, t], W)
        res["decode_ms"][f"B{B}"] = {"lm_head_only_per_tok": timeit(d_head)[0] / 16, "G_per_tok": timeit(d_G)[0] / 16,
                                     **{f"router+spec_r{r}_per_tok": timeit(d_R(r))[0] / 16 for r in REFRESH}}
        bb = {k: v[:B] for k, v in inp.items()}

        def full_recompute():
            with torch.no_grad(), ac():
                model(bb)
        res["decode_ms"][f"B{B}"]["base_full_recompute_ctx1024_upper_bound"] = timeit(full_recompute)[0]
    # ---- kernel behaviour (top CUDA kernels by time, prefill)
    from torch.profiler import ProfilerActivity, profile
    res["kernels"] = {}
    for name, fn in (("base", v_base), ("base+G", v_G), ("base+router+spec_r1", v_R(1))):
        fn()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn()
            torch.cuda.synchronize()
        ka = sorted(prof.key_averages(), key=lambda k: -k.device_time_total if hasattr(k, "device_time_total") else -k.cuda_time_total)
        tot = sum((k.device_time_total if hasattr(k, "device_time_total") else k.cuda_time_total) for k in ka)
        res["kernels"][name] = {"n_kernel_types": len(ka), "n_calls": int(sum(k.count for k in ka)), "cuda_us_total": tot,
                                "top": [{"name": k.key[:80], "us": (k.device_time_total if hasattr(k, "device_time_total") else k.cuda_time_total),
                                         "calls": k.count} for k in ka[:8]]}
    mon.stop()
    if mon.contaminated:
        sys.exit(77)
    EFF.mkdir(parents=True, exist_ok=True)
    guard.atomic_write_json(EFF / "speed.json", res, overwrite=True)
    print(json.dumps({k: res[k] for k in ("prefill_rel_to_base", "decode_ms")}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["ce", "speed"])
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    if a.cmd == "ce":
        ce_study()
    else:
        if not guard.require_execution(a.execute, "gpu"):
            return
        speed_study(a)


if __name__ == "__main__":
    main()
