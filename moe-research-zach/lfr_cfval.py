#!/usr/bin/env python
"""csr-latent-forecast-router-distill-v1: counterfactual-utility estimator validation (protocol 6.4) on the S1 TOKEN-4
reference checkpoint. 256 (position, layer) pairs from CALIB-SEARCH paper half B: first-order estimate (bf16 autocast
forward/backward, as in training) vs exact slot-4 substitution of all 32 experts (full re-forward in fp32).
Gate: mean Spearman >= 0.5 and mean top-4 overlap >= 2.0. Writes artifacts/.../cf_validation.json."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402

N_PAIRS = 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    if not guard.require_execution(a.execute, "gpu" if a.device == "cuda" else "cpu-model"):
        return
    import numpy as np
    import torch
    import torch.nn.functional as F
    from csr.fsp_lib import Split
    from csr.lfr_lib import ART
    from csr.lfr_eval import eval_rows, paper_half, inputs, first_order_delta, exact_delta, spearman, topk_overlap
    from lfr_evaluate import load_milestone
    if a.device == "cuda":
        guard.occupancy_gate("lfr cfval")
        device = guard.setup_cuda_device()
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        device = torch.device("cpu")
    guard.register_self("lfr cfval")
    t_iso, t0 = guard.now(), time.time()
    rd = guard.inside_root(a.run_dir)
    summ = json.loads((rd / "summary.json").read_text())
    model, _ = load_milestone(summ["ckpts"]["p100"]["path"], device)
    sp = Split("calib_search")
    rows = eval_rows(sp, 200_000, paper_half(sp, 1))
    rng = np.random.default_rng(20261111)
    pairs = []
    fo_all, ex_all = [], []
    while len(pairs) < N_PAIRS:
        r = [rows[int(rng.integers(len(rows)))]]
        _, bt, mi = inputs(r, sp, device)
        L = int((bt["doc_ids"][0] >= 0).sum())
        ts = sorted(set(int(x) for x in rng.integers(1, max(2, L - 1), 8)))
        mi["cf_idx"] = torch.as_tensor(ts, device=device)
        store = {}
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(mi, {"cf_store": store, "tap": True})
            lg = model.lm_logits(out["hidden"]).float()
        F.cross_entropy(lg.reshape(-1, lg.shape[-1]), bt["targets"].reshape(-1), ignore_index=-100,
                        reduction="sum").backward()
        model.zero_grad(set_to_none=True)
        row = {k: v for k, v in mi.items() if k != "cf_idx"}
        row["_targets"] = bt["targets"]
        for j, tpos in enumerate(ts):
            li = int(rng.integers(model.n_moe))
            rec, gr = store[f"rec{li}"], store[f"grad{li}"]
            fo = first_order_delta(model.blocks[li + 1].ffn.experts, rec["x"][j:j + 1], gr[j:j + 1],
                                   rec["r4"][j:j + 1], rec["g4"][j:j + 1], 1.0)[0].cpu().numpy()
            ex = exact_delta(model, row, li, tpos, rec["x"][j], int(rec["r4"][j]), float(rec["g4"][j]), device,
                             expect_topi=rec["topi"][j].tolist()).cpu().numpy()
            ok = bool(getattr(exact_delta, "route_ok", True))
            pairs.append({"layer": li, "t": tpos, "rho": spearman(fo, ex), "top4": topk_overlap(fo, ex),
                          "route_ok": ok, "ex_std": float(ex.std()), "fo_std": float(fo.std())})
            fo_all.append(fo)
            ex_all.append(ex)
            if len(pairs) >= N_PAIRS:
                break
    fo_all, ex_all = np.concatenate(fo_all), np.concatenate(ex_all)
    slope = float((fo_all * ex_all).sum() / max(1e-30, (fo_all * fo_all).sum()))
    nmae = float(np.abs(ex_all - fo_all).mean() / max(1e-30, np.abs(ex_all).mean()))
    ok_pairs = [p for p in pairs if p["route_ok"]]
    res = {"n_pairs": len(pairs), "n_route_ok": len(ok_pairs),
           "mean_spearman": float(np.mean([p["rho"] for p in pairs])),
           "mean_top4_overlap": float(np.mean([p["top4"] for p in pairs])),
           "mean_spearman_route_ok": float(np.mean([p["rho"] for p in ok_pairs])) if ok_pairs else None,
           "calibration_slope_exact_on_fo": slope, "abs_slope_minus_1": abs(slope - 1.0), "nmae": nmae,
           "per_layer_spearman": {str(li): float(np.mean([p["rho"] for p in pairs if p["layer"] == li]))
                                  for li in sorted({p["layer"] for p in pairs})},
           "pairs": pairs, "checkpoint": summ["ckpts"]["p100"], "seconds": time.time() - t0}
    res["gate_pass"] = bool(res["mean_spearman"] >= 0.5 and res["mean_top4_overlap"] >= 2.0)
    guard.atomic_write_json(ART / "cf_validation.json", res, overwrite=True)
    if a.device == "cuda":
        guard.gpu_ledger(t_iso, guard.now(), "lfr_cfval", "lfr cf validation", "COMPLETE")
        with open(ART / "compute_ledger.tsv", "a") as f:
            f.write(f"{t_iso}\t{guard.now()}\t{time.time() - t0:.1f}\tlfr_cfval\tcfval\tdiagnostics\tCOMPLETE\n")
    print(json.dumps({k: v for k, v in res.items() if k != "pairs"}, indent=1))


if __name__ == "__main__":
    main()
