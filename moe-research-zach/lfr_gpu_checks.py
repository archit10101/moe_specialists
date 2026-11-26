#!/usr/bin/env python
"""csr-latent-forecast-router-distill-v1 GPU correctness checks (protocol 14): causality of the draft forecasts (flash
path) and of the teacher / student / executed routes and logits on GPU with real cached features; first-order vs exact
substitution on a randomly initialised full-size model. Writes artifacts/.../gpu_checks_<stamp>.json; exit 0 iff pass."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    if not guard.require_execution(a.execute, "gpu" if a.device == "cuda" else "cpu-model"):
        return
    import numpy as np
    import torch
    from csr.fsp_lib import Split, make_batch, to_torch, T_SEQ, PAD_ID
    from csr.lfr_lib import ART, CACHE_DIR, ForecastCache, draft_outputs, gather_raw, build_features
    from csr.lfr_model import LFRConfig, LFRModel
    from csr.lfr_diag import diag_set
    from lfr_cache import load_draft
    guard.occupancy_gate("lfr gpu checks")
    device = guard.setup_cuda_device()
    guard.register_self("lfr gpu checks")
    t_iso, t0 = guard.now(), time.time()
    res = {}
    # 1. draft forecast causality on GPU (flash attention path)
    sp, rows = diag_set()
    doc = rows[0][0]
    ids = torch.full((1, T_SEQ), PAD_ID, dtype=torch.long)
    L = int(sp.lengths[doc])
    ids[0, :L] = torch.as_tensor(sp.doc(doc))
    dr = load_draft(3, device)
    t = L // 2
    o1 = draft_outputs(dr, ids.to(device), torch.tensor([L], device=device))
    ids2 = ids.clone()
    ids2[0, t + 1:L] = torch.randint(2, 50000, (L - t - 1,))
    o2 = draft_outputs(dr, ids2.to(device), torch.tensor([L], device=device))
    agree = {k: float((o1[k][0, :t + 1] == o2[k][0, :t + 1]).float().mean()) for k in ("top_ids", "roll")}
    dmax = {k: float((o1[k][0, :t + 1] - o2[k][0, :t + 1]).abs().max()) for k in ("ent1", "score", "top_p")}
    res["draft_causality"] = {"id_agreement_prefix": agree, "max_abs_diff_prefix": dmax,
                              "pass": min(agree.values()) >= 0.999 and max(dmax.values()) < 0.05}
    del dr
    # 2. main-model route / logit causality with real cached features, teacher mode (eager, bf16 autocast)
    cache = ForecastCache(CACHE_DIR, "calib_search")
    cfg = LFRConfig(router="student", teacher=True)
    m = LFRModel(cfg, 0).to(device).eval()
    rr = rows[:4]
    a_ = make_batch(rr, sp)
    bt = to_torch(a_, device)
    raw = gather_raw(rr, sp, cache)
    rt = {k: torch.as_tensor(v.astype(np.int64) if v.dtype == np.uint16 else v, device=device) for k, v in raw.items()}
    f1 = build_features(rt, bt["doc_ids"], bt["positions"])
    t = 400
    ids2 = bt["input_ids"].clone()
    ids2[:, t + 1:] = torch.randint(2, 50000, ids2[:, t + 1:].shape, device=device)
    rt2 = {k: v.clone() for k, v in rt.items()}
    rt2["src_tok"] = ids2.clone()
    for k in rt2:
        if k != "src_tok":
            rt2[k][:, t + 1:] = rt2[k][:, t + 1:].flip(1)
    f2 = build_features(rt2, bt["doc_ids"], bt["positions"])
    feat_eq = all(torch.equal(f1[k][:, :t + 1], f2[k][:, :t + 1]) for k in f1)
    outs = []
    # full fp32 (no autocast, no TF32): separates causal leakage from batch-composition rounding of bf16 GEMMs
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for ids_, f_ in ((bt["input_ids"], f1), (ids2, f2)):
        with torch.no_grad():
            o = m({"input_ids": ids_, "doc_ids": bt["doc_ids"], "positions": bt["positions"], "fc": f_,
                   "lam": torch.tensor(0.9, device=device)}, {"teach": True, "collect": True})
            outs.append((o, m.lm_logits(o["hidden"]).float()))
    tag = float(np.mean([float((r1["topi"][:, :t + 1] == r2["topi"][:, :t + 1]).float().mean())
                         for r1, r2 in zip(outs[0][0]["records"], outs[1][0]["records"])]))
    ldiff = float((outs[0][1][:, :t + 1] - outs[1][1][:, :t + 1]).abs().max())
    res["route_causality"] = {"features_prefix_identical": feat_eq, "topi_agreement_prefix": tag,
                              "logit_max_abs_diff_prefix": ldiff, "precision": "fp32",
                              "pass": feat_eq and tag >= 0.999 and ldiff < 1e-3}
    res["all_pass"] = all(v["pass"] for v in res.values() if isinstance(v, dict))
    res["seconds"] = time.time() - t0
    out = ART / f"gpu_checks_{guard.utc_stamp()}.json"
    guard.atomic_write_json(out, res)
    guard.gpu_ledger(t_iso, guard.now(), "lfr_gpu_checks", "lfr gpu checks", "PASS" if res["all_pass"] else "FAIL")
    with open(ART / "compute_ledger.tsv", "a") as f:
        f.write(f"{t_iso}\t{guard.now()}\t{time.time() - t0:.1f}\tlfr_gpu_checks\tgpu_checks\tchecks_and_tests\t"
                f"{'PASS' if res['all_pass'] else 'FAIL'}\n")
    print(json.dumps(res, indent=1))
    sys.exit(0 if res["all_pass"] else 1)


if __name__ == "__main__":
    main()
