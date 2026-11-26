#!/usr/bin/env python
"""oracle-residual-v1 GPU tests + benchmark (guarded GPU job; TOKEN-4 s0; SEL rows only).
  G-ZERO   zero-initialised A, B, C reproduce frozen TOKEN-4 per-token CE bit-exactly on GPU (bf16 autocast), 16 rows
  G-HASH   base state hash unchanged after the forward/backward passes below
  G-BENCH  joint training step throughput (32 rows; base forward shared by A, B, C), base-forward-only time, full
           evaluation micro-batch time (assigned path + 2 x 16 replacement passes), peak memory
Writes artifacts/oracle-residual-v1/gpu_tests_<UTC>.json (0400). (G-RESUME runs as separate guarded test jobs.)"""
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
    import torch
    from csr.fsp_lib import Split
    from csr.orv1_lib import (CONDS, base_forward, head_ce, load_base, load_labels, load_part, make_batch, make_conds,
                              rms0, state_hash, to_dev, token_assign)
    from csr.orv1_paths import ART, K
    guard.occupancy_gate("orv1 gpu tests")
    dev = guard.setup_cuda_device()
    guard.register_self("orv1 gpu tests")
    torch.backends.cuda.matmul.allow_tf32 = False
    t_iso, t0 = guard.now(), time.time()
    res = []
    model, _ = load_base(0, dev)
    h0 = state_hash(model)
    cs = Split("calib_search")
    sem, shuf = load_labels("cs")
    ls, lh = torch.as_tensor(sem, device=dev), torch.as_tensor(shuf, device=dev)
    rows = load_part("SEL")
    bt = to_dev(make_batch(rows[:16], cs), dev)
    x, hid = base_forward(model, bt)
    tgt = bt["targets"].reshape(-1)
    v = tgt >= 0
    ce_d = head_ce(model, hid, tgt, from_hidden=True)
    conds = make_conds(0, dev)
    xn = rms0(x)
    aa, ab = token_assign(bt, ls, lh)
    z = {}
    with torch.no_grad():
        for c in CONDS:
            delta, idx, _ = conds[c](xn, {"A": aa, "B": ab, "C": None}[c], v)
            z[c] = bool(torch.equal(head_ce(model, x + delta, tgt)[v], ce_d[v]))
    res.append({"test": "G-ZERO zero-init bit-exact on GPU", "ok": all(z.values()), **z})
    # bench: joint training steps on 32 SEL rows (not used for anything else)
    opts = {c: torch.optim.AdamW(conds[c].parameters(), lr=1e-3, fused=True) for c in CONDS}
    torch.cuda.reset_peak_memory_stats()
    times = []
    for it in range(6):
        torch.cuda.synchronize()
        t1 = time.time()
        bts = [to_dev(make_batch(rows[32 * it + i:32 * it + i + 16], cs), dev) for i in (0, 16)]
        nv = sum(int((b["targets"] >= 0).sum()) for b in bts)
        for c in CONDS:
            opts[c].zero_grad(set_to_none=True)
        for b in bts:
            x, hid = base_forward(model, b)
            t_ = b["targets"].reshape(-1)
            vv = t_ >= 0
            with torch.no_grad():
                head_ce(model, hid, t_, from_hidden=True)
            xn = rms0(x)
            a1, a2 = token_assign(b, ls, lh)
            for c in CONDS:
                delta, idx, aux = conds[c](xn, {"A": a1, "B": a2, "C": None}[c], vv)
                loss = head_ce(model, x + delta, t_)[vv].sum() / nv + (0.01 * aux if c == "C" else 0)
                loss.backward()
        for c in CONDS:
            opts[c].step()
        torch.cuda.synchronize()
        times.append((time.time() - t1, nv))
    st = sum(t for t, _ in times[1:]) / 5
    tok = sum(n for _, n in times[1:]) / 5
    torch.cuda.synchronize()
    t1 = time.time()
    for it in range(4):
        b = to_dev(make_batch(rows[16 * it:16 * it + 16], cs), dev)
        base_forward(model, b)
    torch.cuda.synchronize()
    tb = (time.time() - t1) / 4
    t1 = time.time()
    with torch.no_grad():
        for it in range(2):
            b = to_dev(make_batch(rows[16 * it:16 * it + 16], cs), dev)
            x, hid = base_forward(model, b)
            t_ = b["targets"].reshape(-1)
            cd = head_ce(model, hid, t_, from_hidden=True)
            xn = rms0(x)
            for c in ("A", "B"):
                outs = conds[c].all_outputs(xn)
                for k in range(K):
                    head_ce(model, x + outs[:, k], t_)
    torch.cuda.synchronize()
    te = (time.time() - t1) / 2
    res.append({"test": "G-BENCH", "ok": True, "train_step_s": st, "train_tokens_per_step": tok,
                "train_tokens_per_s": tok / st, "base_forward_16rows_s": tb, "eval_full_microbatch16_s": te,
                "peak_alloc_gib": torch.cuda.max_memory_allocated() / 2**30,
                "est_hours": {"20M_joint": 20e6 / (tok / st) / 3600, "50M_joint": 50e6 / (tok / st) / 3600,
                              "245M_joint": 245e6 / (tok / st) / 3600,
                              "eval_full_EVAL_6.1M": 6164 / 16 * te / 3600}})
    res.append({"test": "G-HASH base unchanged", "ok": state_hash(model) == h0})
    doc = {"time": guard.now(), "seconds": time.time() - t0, "n": len(res), "n_fail": sum(not r["ok"] for r in res),
           "results": res}
    p = ART / f"gpu_tests_{guard.utc_stamp()}.json"
    guard.atomic_write_json(p, doc, mode=0o400)
    print(json.dumps(doc, indent=1))
    guard.gpu_ledger(t_iso, guard.now(), "orv1_gpu_tests", "orv1 gpu tests", "COMPLETE")
    sys.exit(0 if doc["n_fail"] == 0 else 1)


if __name__ == "__main__":
    main()
