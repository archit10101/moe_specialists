#!/usr/bin/env python
"""scrgp-v1 GPU correctness tests (prereg section 4; guarded GPU job). Real C6_s0 p100 weights, CALIB-SEARCH half-A rows.
  G-GRAD  GPU (bf16 autocast, fp64 probed expert) exact forced-insertion gradients vs a CPU fp32 reference (autocast off)
          on the same rows: cosine >= 0.99 for 3 weight vectors x 3 (layer, expert) pairs
  G-OFF   probe natural CE vs LFRModel.forward CE on GPU (mean |delta| < 2e-3); natural_updated(theta' = theta) and
          forced(theta' = theta) bit-identical to their references
Writes artifacts/scrgp-v1/gpu_tests_<UTC>.json (0400). (G-RESUME / G-EVAL run as separate guarded stage-1 test jobs.)"""
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
    from fle_evaluate import load_milestone
    from csr.fsp_lib import Split, pack_rows
    from csr.lfr_eval import paper_half
    from csr.scrgp_lib import Probe, make_batch, to_dev, u_hash
    from csr.scrgp_paths import ART, C6_RUNS, check_prereg
    check_prereg()
    guard.occupancy_gate("scrgp gpu tests")
    dev = guard.setup_cuda_device()
    guard.register_self("scrgp gpu tests")
    t_iso, t0 = guard.now(), time.time()
    torch.set_num_threads(16)
    res = []
    cs = Split("calib_search")
    docs = sorted(paper_half(cs, 0), key=lambda i: u_hash("scrgp-gputest", f"{cs.papers[i]}|{i}"))
    rows = pack_rows(cs.lengths, docs)[:2]
    mg, _ = load_milestone(C6_RUNS[0] + "/ckpt_p100", dev)
    mc, _ = load_milestone(C6_RUNS[0] + "/ckpt_p100", torch.device("cpu"))
    btg, btc = to_dev(make_batch(rows, cs), dev), to_dev(make_batch(rows, cs), torch.device("cpu"))
    pg, pc = Probe(mg, dev), Probe(mc, torch.device("cpu"))
    ceg, _, cg, xg = pg.natural(btg, keep_cache=True)
    cec, _, cc, xc = pc.natural(btc, keep_cache=True)
    shape = tuple(btg["input_ids"].shape) + (mg.cfg.d_model,)
    v = btc["targets"].reshape(-1) >= 0
    g = torch.Generator().manual_seed(0)
    W = torch.rand(3, v.numel(), generator=g) * v
    W[1] -= W[1][v].mean() * v
    W = W / W.abs().sum(1, keepdim=True)
    coss = []
    for li, e in ((0, 4), (5, 3), (10, 20)):
        Gg = pg.expert_grads(li, e, cg[li], xg, shape, btg["targets"], W.to(dev)).double().cpu()
        Gc = pc.expert_grads(li, e, cc[li], xc, shape, btc["targets"], W).double()
        coss += [float((Gg[j] @ Gc[j]) / (Gg[j].norm() * Gc[j].norm())) for j in range(3)]
    res.append({"test": "G-GRAD GPU vs CPU fp32 gradients", "ok": min(coss) >= 0.99, "cos": coss})
    with torch.no_grad(), pg.autocast():
        o = mg({"input_ids": btg["input_ids"], "doc_ids": btg["doc_ids"], "positions": btg["positions"]})
        ref = pg.ce_of(o["hidden"], btg["targets"])
    vg = btg["targets"].reshape(-1) >= 0
    dmean = float((ceg - ref)[vg].abs().mean())
    li, e = 4, 7
    c = cg[li]
    w1, w2 = mg.blocks[li + 1].ffn.experts.w1[e], mg.blocks[li + 1].ffn.experts.w2[e]
    B, T, D = shape
    with torch.no_grad(), pg.autocast():
        hn = pg._run(c["x_mid"] + (c["slots"].sum(1) + c["sh"]).view(B, T, -1), xg, li + 2)
        hu = pg.natural_updated_hidden(li, e, c, xg, shape, w1, w2, w1.clone(), w2.clone())
        f0 = pg.forced_hidden(li, e, c, xg, shape, w1, w2)
        f1 = pg.forced_hidden(li, e, c, xg, shape, w1.clone(), w2.clone())
    res.append({"test": "G-OFF probe natural vs model.forward; disabled probes bit-identical",
                "ok": dmean < 2e-3 and bool(torch.equal(hn, hu)) and bool(torch.equal(f0, f1)),
                "mean_abs_ce_delta": dmean, "ce_bitwise_equal": bool(torch.equal(ceg[vg], ref[vg])), "upd_equal": bool(torch.equal(hn, hu)), "forced_equal": bool(torch.equal(f0, f1))})
    doc = {"time": guard.now(), "seconds": time.time() - t0, "n": len(res), "n_fail": sum(not r["ok"] for r in res),
           "results": res}
    p = ART / f"gpu_tests_{guard.utc_stamp()}.json"
    guard.atomic_write_json(p, doc, mode=0o400)
    print(json.dumps(doc, indent=1))
    guard.gpu_ledger(t_iso, guard.now(), "scrgp_gpu_tests", "scrgp gpu tests", "COMPLETE")
    sys.exit(0 if doc["n_fail"] == 0 else 1)


if __name__ == "__main__":
    main()
