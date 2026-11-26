#!/usr/bin/env python
"""csr-soft-cf-usefulness-v2 gamma anchor + pre-search measurement diagnostic (prereg section 6). Frozen V1 checkpoint
only (REF_C6_s100 ckpt_s20, read-only); no training, no held-out split, no real labels.
  1. 400 probe steps of the exact training-time procedure (ProbePlan seed 100, 16 rows x 8 passes, all 11 MoE layers,
     no decay), aggregated with the SEM table and its persistent permutation (SHUF) — sue_diag.run_one, unchanged:
     split-half reliability, coverage, own-group margin.
  2. prior R = mean q-weighted group-mean routing of the anchor over plan steps 268-317 (forward only).
  3. on the rows of plan step 318: ||grad L_use|| (V1 definition, tau 0.2, V1 targets Z) and ||grad L_CF|| (tilt, lcb,
     z 1, tau 1) on backbone parameters; target statistics (evidence fraction, KL(q||R), q entropy) for SEM and SHUF.
Writes artifacts/soft-cf-usefulness-v2/anchor.json (once)."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402

ANCHOR = "runs/20261117T073831Z_semantic_utility_edl_v1_REF_C6_s100/ckpt_s20"


def prior_from(m, split, steps, pq, device, idx):
    import numpy as np
    import torch
    from csr.fsp_lib import make_batch, to_torch
    from csr.scu_cf import group_router_mean
    K, L, E = pq.K, m.n_moe, m.cfg.n_experts
    S, W = np.zeros((K, L, E)), np.zeros((K, L))
    for i in idx:
        npb = make_batch(steps[i], split)
        bt = to_torch(npb, device)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = m({"input_ids": bt["input_ids"], "doc_ids": bt["doc_ids"], "positions": bt["positions"]},
                    {"collect": True})
        qs = torch.as_tensor(pq.qseq(npb), device=device)
        sid, nonpad = bt["seq_id"].reshape(-1), (bt["doc_ids"] >= 0).reshape(-1)
        mass = qs.double().sum(0).cpu().numpy()
        for li in range(L):
            P, keep = group_router_mean(out["records"][li]["p"].reshape(-1, E).float(), qs, sid, nonpad)
            kk = keep.cpu().numpy()
            S[kk, li] += P.double().cpu().numpy() * mass[kk, None]
            W[kk, li] += mass[kk]
    R = S / np.maximum(W[..., None], 1e-12)
    R[W == 0] = 1.0 / E
    return R


def grads(m, bank, R, split, steps, pq, device):
    import numpy as np
    import torch
    from csr.fsp_lib import make_batch, to_torch
    from csr.scu_cf import cf_loss, confidence_scores, soft_target, target_stats
    from csr.sue_cf import use_loss
    m.train()
    Z, V, _ = bank.targets()
    d, se, v = bank.delta()
    s, scale = confidence_scores(d, se, v, 1.0, "lcb")
    Q, EV = soft_target(s, R, 1.0, "tilt")
    params = [p for p in m.parameters() if p.requires_grad]
    npb = make_batch(steps[318][:16], split)
    bt = to_torch(npb, device)
    E = m.cfg.n_experts
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        out = m({"input_ids": bt["input_ids"], "doc_ids": bt["doc_ids"], "positions": bt["positions"]}, {"collect": True})
    qs = torch.as_tensor(pq.qseq(npb), device=device)
    sid, nonpad = bt["seq_id"].reshape(-1), (bt["doc_ids"] >= 0).reshape(-1)
    pl = [out["records"][li]["p"].reshape(-1, E).float() for li in range(m.n_moe)]
    with torch.autocast(device_type=device.type, enabled=False):
        lu, nact, marg = use_loss(pl, list(range(m.n_moe)), qs, sid, nonpad,
                                  torch.as_tensor(Z, device=device, dtype=torch.float32), torch.as_tensor(V, device=device),
                                  0.2)
        lc, ncf, kl, _ = cf_loss(pl, list(range(m.n_moe)), qs, sid, nonpad,
                                 torch.as_tensor(Q, device=device, dtype=torch.float32), torch.as_tensor(EV, device=device))
    res = {"use_value": float(lu), "use_terms": nact, "cf_value": float(lc), "cf_terms": ncf,
           "target": target_stats(Q, R, EV), "score_pos_frac": float((s > 0).mean()), "score_neg_frac": float((s < 0).mean()),
           "scale_by_layer": scale.tolist()}
    for name, val in (("use_unit", lu), ("cf_unit", lc)):
        if not val.requires_grad:
            res[f"gn_{name}"] = 0.0
            continue
        gs = torch.autograd.grad(val, params, retain_graph=True, allow_unused=True)
        res[f"gn_{name}"] = float(torch.sqrt(sum((g.float() ** 2).sum() for g in gs if g is not None)))
    m.zero_grad(set_to_none=True)
    return res, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    if not guard.require_execution(a.execute, "gpu" if a.device == "cuda" else "cpu-model"):
        return
    import numpy as np
    import torch
    from fle_evaluate import load_milestone
    from sue_diag import run_one
    from csr.gc_lib import PaperQ
    from csr.lfr_lib import main_stream
    from csr.scu_cf import confidence_scores, soft_target, target_stats
    from csr.scu_paths import ART, FINAL_BUDGET, SHUF_TABLE, SHUF_TABLE_SHA, TABLE, TABLE_SHA, check_prereg
    from csr.sue_cf import ProbePlan, probe_pool
    check_prereg()
    outp = ART / "anchor.json"
    if outp.exists():
        raise guard.GuardError("anchor already written")
    mon = None
    if a.device == "cuda":
        guard.occupancy_gate("scu anchor")
        device = guard.setup_cuda_device()
        mon = guard.OccupancyMonitor(interval=15.0)
        mon.start()
    else:
        device = torch.device("cpu")
    guard.register_self("scu anchor")
    t_iso, t0 = guard.now(), time.time()
    split, steps, _ = main_stream(FINAL_BUDGET)
    pq = PaperQ(guard.ROOT / TABLE, split, TABLE_SHA)
    pqs = PaperQ(guard.ROOT / SHUF_TABLE, split, SHUF_TABLE_SHA)
    pool = probe_pool(split, steps, pq.q_of_doc)
    res = {"time": guard.now(), "anchor": ANCHOR, "pool_docs": int(len(pool)), "rows": 16, "passes": 8,
           "steps": a.steps}
    status = "COMPLETE"
    try:
        m, _ = load_milestone(ANCHOR, device)
        plan = ProbePlan(pool, 100, 16, 8, range(m.n_moe))
        import sue_diag
        sue_diag.SIZES = (100, 200, a.steps)
        out, banks, noise = run_one(m, split, plan, {"SEM": pq.q_of_doc, "SHUF": pqs.q_of_doc}, device, a.steps, 16, 8)
        res["by_size"], res["noise_floor"] = out, noise
        R = prior_from(m, split, steps, pq, device, range(268, 318))
        Rs = prior_from(m, split, steps, pqs, device, range(268, 318))
        g, s = grads(m, banks["SEM"], R, split, steps, pq, device)
        res["grad_anchor"] = g
        d, se, v = banks["SHUF"].delta()
        ss, _ = confidence_scores(d, se, v, 1.0, "lcb")
        Qs, EVs = soft_target(ss, Rs, 1.0, "tilt")
        res["shuf_target"] = {**target_stats(Qs, Rs, EVs), "score_pos_frac": float((ss > 0).mean()),
                              "score_neg_frac": float((ss < 0).mean())}
    except KeyboardInterrupt:
        status = "EXTERNALLY_STOPPED"
    finally:
        if mon is not None:
            mon.stop()
            if mon.contaminated and status == "COMPLETE":
                status = "EXTERNALLY_STOPPED"
    res["status"], res["seconds"] = status, time.time() - t0
    if status == "COMPLETE":
        guard.atomic_write_json(outp, res, mode=0o400)
    if a.device == "cuda":
        guard.gpu_ledger(t_iso, guard.now(), "scu_anchor", "scu anchor", status)
        with open(ART / "compute_ledger.tsv", "a") as f:
            f.write(f"{t_iso}\t{guard.now()}\t{time.time() - t0:.1f}\tanchor\tanchor\tdiagnostic\t{status}\n")
    print(json.dumps({"status": status, "seconds": res["seconds"]}))
    sys.exit(0 if status == "COMPLETE" else 77)


if __name__ == "__main__":
    main()
