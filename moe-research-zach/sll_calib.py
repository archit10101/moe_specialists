#!/usr/bin/env python
"""csr-semantic-lift-expert-learning-v1 PRE-REGISTRATION calibration (reference checkpoints only; no candidate is
trained or evaluated). For existing Semantic-C6 / SHUF-C6 milestones (V1 REF runs, seed 100), run no-grad forwards on
the next training-stream steps of the registered plan, accumulate LiftStats (rho as registered candidate) for the SEM
and the SHUF table, and report: lift magnitudes, the robust scale of the token lift s over selected pairs, Kish ESS,
split-half (even / odd steps) reliability of the lift table and the implied multiplier activity for each lambda and
candidate floor. Writes artifacts/semantic-lift-expert-learning-v1/calibration.json (0400)."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402

REFS = {"REF_C6_s100": "runs/20261117T073831Z_semantic_utility_edl_v1_REF_C6_s100",
        "REF_SHUF_s100": "runs/20261117T182907Z_semantic_utility_edl_v1_REF_SHUF_s100"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    if not guard.require_execution(a.execute, "gpu" if a.device == "cuda" else "cpu-model"):
        return
    import numpy as np
    import torch
    from fle_evaluate import load_milestone
    from csr.fsp_lib import make_batch, to_torch
    from csr.gc_lib import PaperQ
    from csr.lfr_lib import main_stream
    from csr.sll_lift import MAD_C, LiftStats, multipliers, q_tokens
    from csr.sll_paths import ART, SHUF_TABLE, SHUF_TABLE_SHA, TABLE, TABLE_SHA
    mon = None
    if a.device == "cuda":
        guard.occupancy_gate("sll calib")
        device = guard.setup_cuda_device()
        mon = guard.OccupancyMonitor(interval=15.0)
        mon.start()
    else:
        device = torch.device("cpu")
    guard.register_self("sll calibration (reference checkpoints)")
    t_iso, t0 = guard.now(), time.time()
    split, steps, _ = main_stream(245_000_000)
    tabs = {"SEM": PaperQ(guard.ROOT / TABLE, split, TABLE_SHA), "SHUF": PaperQ(guard.ROOT / SHUF_TABLE, split, SHUF_TABLE_SHA)}
    out = {"time": guard.now(), "note": "pre-registration calibration on existing reference checkpoints (training stream)",
           "rho": 0.98, "n0": 8.0, "steps": a.steps, "results": {}}
    status = "COMPLETE"
    try:
        for rn, rd in REFS.items():
            ck = json.loads((guard.ROOT / rd / "status.json").read_text())["ckpts"]
            for tag in ("s20", "s50"):
                m, _ = load_milestone(ck[tag]["path"], device)
                m.eval()
                E, Lm = m.cfg.n_experts, m.n_moe
                st0 = ck[tag]["step"]
                S = {}
                for tn, pq in tabs.items():
                    S[tn] = {"all": LiftStats(pq.K, Lm, E, 0.98, 8.0, 0.0, device),
                             "even": LiftStats(pq.K, Lm, E, 0.98, 8.0, 0.0, device),
                             "odd": LiftStats(pq.K, Lm, E, 0.98, 8.0, 0.0, device)}
                keep = []  # (qseq per table, seq_id, nonpad, topi list, gates list) of the last 8 steps
                for j in range(a.steps):
                    rows = steps[st0 + j]
                    for mi_ in range(2):
                        npb = make_batch(rows[mi_ * 32:(mi_ + 1) * 32], split)
                        bt = to_torch(npb, device)
                        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                                             enabled=device.type == "cuda"):
                            o = m({"input_ids": bt["input_ids"], "doc_ids": bt["doc_ids"], "positions": bt["positions"]},
                                  {"collect": True})
                        sid = bt["seq_id"].reshape(-1)
                        nonpad = (bt["doc_ids"] >= 0).reshape(-1)
                        ps = [r_["p"].reshape(-1, E).float() for r_ in o["records"]]
                        qs = {tn: torch.as_tensor(pq.qseq(npb), device=device) for tn, pq in tabs.items()}
                        for tn in tabs:
                            S[tn]["all"].accumulate(ps, qs[tn], sid, nonpad)
                            S[tn]["even" if j % 2 == 0 else "odd"].accumulate(ps, qs[tn], sid, nonpad)
                        if j >= a.steps - 8:
                            keep.append((qs, sid, nonpad, [r_["topi"].reshape(-1, 4) for r_ in o["records"]],
                                         [r_["p"].reshape(-1, E).float().gather(1, r_["topi"].reshape(-1, 4))
                                          for r_ in o["records"]]))
                    for tn in tabs:
                        S[tn]["all"].commit()
                        S[tn]["even" if j % 2 == 0 else "odd"].commit()
                res = {"step0": st0}
                for tn in tabs:
                    L = S[tn]["all"].lift()  # (Lm, K, E)
                    La, Lb = S[tn]["even"].lift(), S[tn]["odd"].lift()
                    a_, b_ = La.reshape(-1).cpu().numpy(), Lb.reshape(-1).cpu().numpy()
                    r_sh = float(np.corrcoef(a_, b_)[0, 1])
                    noise_sd = float(np.std(a_ - b_) / 2.0)  # SE of the full-data lift per cell (approx.)
                    svals, mads, mstats = [], [], {}
                    for (qs, sid, nonpad, topis, gs) in keep:
                        qt = q_tokens(qs[tn], sid)
                        for l_ in range(Lm):
                            s_sel = (qt.float() @ L[l_].float()).gather(1, topis[l_])[nonpad]
                            svals.append(s_sel.reshape(-1).cpu().numpy())
                            sv = s_sel.reshape(-1)
                            mads.append(float(MAD_C * (sv - sv.median()).abs().median()))
                    sall = np.concatenate(svals)
                    scale = float(np.median(mads))
                    for fl in (0.0, 0.02, 0.05, 0.1):
                        for lam in (0.25, 0.5, 1.0):
                            ad, cl = [], []
                            for (qs, sid, nonpad, topis, gs) in keep[-2:]:
                                qt = q_tokens(qs[tn], sid)
                                for l_ in range(Lm):
                                    s_sel = (qt.float() @ L[l_].float()).gather(1, topis[l_])[nonpad]
                                    mm, w = multipliers(s_sel, gs[l_][nonpad], lam, max(fl, scale))
                                    ad.append(float((w * (mm - 1).abs()).sum(1).mean()))
                                    cl.append(float(((mm <= 0.5 + 1e-9) | (mm >= 2 - 1e-9)).double().mean()))
                            mstats[f"floor{fl}_lam{lam}"] = {"m_absdev_wmean": float(np.mean(ad)), "clip_frac": float(np.mean(cl))}
                    ess = S[tn]["all"].ess()
                    res[tn] = {"lift_absmean": float(L.abs().mean()), "lift_absmax": float(L.abs().max()),
                               "lift_q": np.quantile(L.reshape(-1).cpu().numpy(), [0.01, 0.1, 0.5, 0.9, 0.99]).tolist(),
                               "s_sel_q": np.quantile(sall, [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]).tolist(),
                               "s_sel_absmean": float(np.abs(sall).mean()), "robust_scale_median": scale,
                               "robust_scale_by_layer_first8": mads[:Lm],
                               "split_half_r_lift": r_sh, "noise_sd_per_cell": noise_sd,
                               "ess_min": float(ess.min()), "ess_median": float(np.median(ess)),
                               "multiplier_activity": mstats}
                    print(json.dumps({"ref": rn, "tag": tag, "table": tn, "scale": scale, "r_sh": r_sh,
                                      "noise_sd": noise_sd, "ess_min": float(ess.min()),
                                      "lift_absmean": res[tn]["lift_absmean"]}), flush=True)
                out["results"][f"{rn}@{tag}"] = res
                del m
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    except KeyboardInterrupt:
        status = "EXTERNALLY_STOPPED"
    finally:
        if mon is not None:
            mon.stop()
            if mon.contaminated and status == "COMPLETE":
                status = "EXTERNALLY_STOPPED"
    out["seconds"] = time.time() - t0
    if status == "COMPLETE":
        guard.atomic_write_json(ART / "calibration.json", out, mode=0o400)
    if a.device == "cuda":
        guard.gpu_ledger(t_iso, guard.now(), "sll_calib", "sll calibration", status)
        with open(ART / "compute_ledger.tsv", "a") as f:
            f.write(f"{t_iso}\t{guard.now()}\t{time.time() - t0:.1f}\tcalibration\tcalibration\tchecks\t{status}\n")
    print(json.dumps({"status": status, "seconds": out["seconds"]}))
    sys.exit(0 if status == "COMPLETE" else 77)


if __name__ == "__main__":
    main()
