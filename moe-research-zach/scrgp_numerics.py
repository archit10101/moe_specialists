#!/usr/bin/env python
"""scrgp-v1 Stage-0 numerics check (GPU, guarded; engineering only). Is a small virtual expert update measurable under the
probe's bf16-autocast downstream computation? On 64 CALIB-SEARCH half-A rows (estimation half; B is never touched):
direction = exact forced-insertion gradient of the MEAN CE over all eligible tokens (a generic direction, not an SCRGP
quantity) for a few (layer, expert) pairs; step sizes eta such that the first-order predicted mean change is
-{0.0005, 0.002, 0.008} nats. Compares per-token dCE from the bf16 probe path with a full-fp32 reference (autocast off,
TF32 off). Prints agreement metrics only.
  scripts/gpu_guarded.sh "scrgp numerics" scrgp_numerics.py --device cuda --execute"""
import argparse
import json
import sys
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
    from fle_evaluate import load_milestone
    from csr.fsp_lib import Split, pack_rows
    from csr.lfr_eval import paper_half
    from csr.scrgp_lib import Probe, make_batch, split_theta, to_dev, u_hash
    from csr.scrgp_paths import C6_RUNS
    guard.occupancy_gate("scrgp numerics")
    dev = guard.setup_cuda_device()
    guard.register_self("scrgp numerics")
    t_iso = guard.now()
    m, _ = load_milestone(C6_RUNS[0] + "/ckpt_p100", dev)
    cs = Split("calib_search")
    docs = sorted(paper_half(cs, 0), key=lambda i: u_hash("scrgp-numerics", f"{cs.papers[i]}|{i}"))
    bt = to_dev(make_batch(pack_rows(cs.lengths, docs)[:64], cs), dev)
    shape = tuple(bt["input_ids"].shape) + (m.cfg.d_model,)
    d, f = m.cfg.d_model, m.cfg.expert_ffn
    res = []
    for amp_ref in (True, False):
        pass
    pr = Probe(m, dev)
    ce0, probs, cache, ctx = pr.natural(bt, keep_cache=True)
    tgt = bt["targets"].reshape(-1)
    v = tgt >= 0
    for li, e in ((0, 5), (5, 3), (9, 17)):
        top3 = (cache[li]["topi"][:, :3] == e).any(1)
        w = (v & ~top3).float()
        w = w / w.sum()
        g = pr.expert_grads(li, e, cache[li], ctx, shape, bt["targets"], w.unsqueeze(0))[0].double()
        ex = m.blocks[li + 1].ffn.experts
        w1e, w2e = ex.w1[e], ex.w2[e]
        for target in (0.0005, 0.002, 0.008):
            eta = target / float((g * g).sum())
            d1, d2 = split_theta(g.float(), d, f)
            w1n, w2n = w1e - eta * d1, w2e - eta * d2
            out = {"layer": li, "expert": e, "target": target,
                   "rel_step": float(eta * g.norm() / torch.cat([w1e.reshape(-1), w2e.reshape(-1)]).double().norm())}
            for mode in ("bf16_probe", "fp32_ref"):
                if mode == "fp32_ref":
                    pr.amp = False
                    torch.backends.cuda.matmul.allow_tf32 = False
                    torch.backends.cudnn.allow_tf32 = False
                    _, _, cache_r, ctx_r = pr.natural(bt, keep_cache=True)
                else:
                    pr.amp = True
                    cache_r, ctx_r = cache, ctx
                with torch.no_grad(), pr.autocast():
                    c0 = pr.ce_of(pr.forced_hidden(li, e, cache_r[li], ctx_r, shape, w1e, w2e), bt["targets"])
                    c1 = pr.ce_of(pr.forced_hidden(li, e, cache_r[li], ctx_r, shape, w1n, w2n), bt["targets"])
                dl = ((c1 - c0).double() * w.double())
                out[mode + "_dce"] = (c1 - c0).double()
                out[mode + "_mean"] = float(dl.sum())
                del cache_r
            x, y = out.pop("bf16_probe_dce")[v], out.pop("fp32_ref_dce")[v]
            out["per_token_corr"] = float(torch.corrcoef(torch.stack([x, y]))[0, 1])
            out["rel_err_mean"] = abs(out["bf16_probe_mean"] - out["fp32_ref_mean"]) / max(1e-30, abs(out["fp32_ref_mean"]))
            # error of means over random token subsets of 2,000 (the size of a small matched group)
            rng = np.random.default_rng(0)
            idx = torch.nonzero(v & ~top3).flatten().cpu().numpy()
            errs, refs = [], []
            xs, ys = (out_x := (x.cpu().numpy())), y.cpu().numpy()
            full_x = torch.zeros_like(tgt, dtype=torch.float64)
            for _ in range(200):
                s = rng.choice(len(idx), 2000, replace=False)
                vv = torch.nonzero(v).flatten().cpu().numpy()
                pos = np.searchsorted(vv, idx[s])
                errs.append(xs[pos].mean() - ys[pos].mean())
                refs.append(ys[pos].mean())
            out["subset2000_err_sd"] = float(np.std(errs))
            out["subset2000_ref_mean_abs"] = float(np.mean(np.abs(refs)))
            res.append(out)
            print(json.dumps(out), flush=True)
    guard.gpu_ledger(t_iso, guard.now(), "scrgp_numerics", "scrgp stage-0 numerics", "COMPLETE")


if __name__ == "__main__":
    main()
