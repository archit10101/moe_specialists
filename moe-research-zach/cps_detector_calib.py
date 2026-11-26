#!/usr/bin/env python
"""csr-context-prior-specialization-v2: freeze the adaptive chunk-boundary thresholds before any V2 run (CPU only).
For each chunk length L in {16, 32, 64, 128}: theta_L such that the mean chunk length of the causal detector
(rising edges of d_t = 1 - cos(EMA_4(R x), EMA_64(R x)) above theta, plus forced boundaries every 4L) on CALIB-SEARCH
paper-half-A documents (engineering split) equals L (bisection). R is the frozen random token embedding of
cps_model (seed DET_SEED). Writes artifacts/context-prior-specialization-v2/detector_calibration.json (0400)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402


def main():
    guard.check_root()
    import numpy as np
    import torch
    from csr.fsp_lib import Split, make_batch, to_torch
    from csr.lfr_eval import eval_rows, paper_half
    from csr.cps_lib import ART
    from csr.cps_model import DET_DIM, DET_SEED, detector_stat
    torch.set_num_threads(8)
    out_p = ART / "detector_calibration.json"
    if out_p.exists():
        print(out_p.read_text())
        return
    sp = Split("calib_search")
    rows = eval_rows(sp, 100_000, paper_half(sp, 0))
    R = torch.randn(50304, DET_DIM, generator=torch.Generator().manual_seed(DET_SEED))
    batches = [to_torch(make_batch(rows[i:i + 16], sp), torch.device("cpu")) for i in range(0, len(rows), 16)]

    pre = []
    with torch.no_grad():
        for b in batches:
            d = detector_stat(b["input_ids"], b["doc_ids"], b["positions"], R)
            pre.append((d, b["positions"], b["doc_ids"] >= 0))

    def mean_len(theta, L):
        nb, nt = 0, 0
        for d, pos, v in pre:  # identical rule to cps_model.adaptive_boundaries
            above = d > theta
            prev = torch.zeros_like(above)
            prev[:, 1:] = above[:, :-1]
            prev = prev & (pos > 0)
            bd = (above & ~prev) | ((pos % (4 * L)) == 0)
            nb += int((bd & v).sum())
            nt += int(v.sum())
        return nt / max(1, nb)
    res = {"detector": "rising edge of 1 - cos(EMA_4(R x), EMA_64(R x)) > theta, plus forced every 4L",
           "R": f"randn(50304, {DET_DIM}) seed {DET_SEED} (frozen buffer, never trained)",
           "data": "CALIB-SEARCH paper half A, 0.1M tokens/domain (engineering split)", "theta": {}, "mean_len": {}}
    for L in (16, 32, 64, 128):
        lo, hi = 0.0, 1.5
        for _ in range(22):
            mid = 0.5 * (lo + hi)
            if mean_len(mid, L) < L:  # too many boundaries -> raise theta
                lo = mid
            else:
                hi = mid
        th = 0.5 * (lo + hi)
        res["theta"][str(L)] = th
        res["mean_len"][str(L)] = mean_len(th, L)
        print(L, th, res["mean_len"][str(L)], flush=True)
    res["time"] = guard.now()
    guard.atomic_write_json(out_p, res, mode=0o400)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
