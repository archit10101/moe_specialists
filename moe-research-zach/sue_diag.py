#!/usr/bin/env python
"""csr-semantic-utility-edl-v1 pre-training reliability diagnostic (prereg section 5). Frozen checkpoints only; no
training, no held-out split, no real domain labels. For each checkpoint: run the exact training-time probe procedure
(ProbePlan seed 100, probe_rows x probe_passes, all 11 MoE layers) for N probe steps without decay; aggregate the SAME
counterfactual observations with the frozen semantic table (SEM) and its persistent paper permutation (SHUF):
  - split-half reliability of DeltaU (document halves), Spearman-Brown, at several sample sizes
  - support / coverage by layer and expert; signal vs noise variance per layer (empirical Bayes)
  - own-group utility margin: mean_{k,l} sum_e F_{k,l,e} DeltaU_{k,l,e} (F = q-weighted routing frequency of group k),
    per document half (interval = half-to-half spread), SEM vs SHUF
  - numerical noise floor: CE change of rows that are NOT modified in a pass where other rows are
  - gradient-norm anchor for gamma: ||grad L_use|| (unit gamma, tau 0.2, SEM targets) vs ||grad beta L_SemEDL|| and
    ||grad L_LM|| on one training batch (plan step 318)
Writes artifacts/semantic-utility-edl-v1/diagnostic.json (once)."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402

SIZES = (80, 160, 318, 640, 1000)


def run_one(m, split, plan, qs, device, n_steps, rows, passes):
    import numpy as np
    import torch
    from csr.fsp_lib import make_batch, to_torch
    from csr.sue_cf import N_BIN, TargetBank, doc_half, row_ce, trunk
    K = qs["SEM"].shape[1]
    L, E = m.n_moe, m.cfg.n_experts
    banks = {n: TargetBank(K, L, E, rho=1.0) for n in qs}
    F = {n: np.zeros((2, K, L, E)) for n in qs}  # q-weighted routing counts per half
    out = {n: {} for n in qs}
    noise = []
    t0 = time.time()
    for t in range(n_steps):
        docs = plan.docs(t)
        bt = to_torch(make_batch([[int(d)] for d in docs], split), device)
        ids, dids, pos = bt["input_ids"], bt["doc_ids"], bt["positions"]
        B, T = ids.shape
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                             enabled=device.type == "cuda"):
            base = trunk(m, ids, dids, pos, cache=True)
            bce, _ = row_ce(m, base["hidden"], bt["targets"])
            bce = bce.double().cpu().numpy()
            bins = (np.argsort(np.argsort(bce)) * N_BIN) // B
            halves = np.asarray([doc_half(d) for d in docs])
            valid = (dids >= 0).cpu().numpy()
            for li in range(L):
                tl = base["topi"][li].view(B, T, -1).cpu().numpy()
                cnt = np.zeros((B, E))
                for b in range(B):
                    cnt[b] = np.bincount(tl[b][valid[b]].reshape(-1), minlength=E) / max(1, valid[b].sum())
                for n, q in qs.items():
                    qd = q[docs]
                    for h in (0, 1):
                        sel = halves == h
                        F[n][h, :, li, :] += qd[sel].T @ cnt[sel]
            for mm in range(passes):
                l_ = plan.layer(t, mm)
                tl = base["topi"][l_].view(B, T, -1).cpu().numpy()
                ke = plan.experts(t, mm, tl, valid, banks["SEM"].support[l_])
                if t % 20 == 0 and mm == 0:  # noise floor: modify the first half of the rows only
                    ken = ke.copy()
                    ken[B // 2:] = -1
                    cfn = trunk(m, ids, dids, pos, start=(l_, base["mids"][l_]),
                                kill=(l_, torch.as_tensor(ken, device=device)))
                    cn, _ = row_ce(m, cfn["hidden"], bt["targets"])
                    noise.extend(np.abs(cn.double().cpu().numpy()[B // 2:] - bce[B // 2:]).tolist())
                cf = trunk(m, ids, dids, pos, start=(l_, base["mids"][l_]), kill=(l_, torch.as_tensor(ke, device=device)))
                cce, _ = row_ce(m, cf["hidden"], bt["targets"])
                u = cce.double().cpu().numpy() - bce
                for n, q in qs.items():
                    banks[n].add(q[docs].astype(np.float64), np.full(B, l_), ke, u, bins, halves)
        if (t + 1) in SIZES:
            for n in qs:
                Z, V, dg = banks[n].targets()
                dA, _, vA = banks[n].delta((0,))
                dB, _, vB = banks[n].delta((1,))
                marg = []
                for h, (dh, vh) in enumerate(((dA, vA), (dB, vB))):
                    Fh = F[n][h] / np.maximum(1e-9, F[n][h].sum(-1, keepdims=True))
                    w = Fh * vh
                    marg.append(float((np.where(vh, dh, 0.0) * w).sum() / max(1e-9, w.sum())))
                out[n][str(t + 1)] = {"n_obs": dg["n_obs"], "coverage_valid": dg["coverage_valid"],
                                      "split_half_r": dg["split_half_r"], "split_half_n": dg["split_half_n"],
                                      "spearman_brown": dg["spearman_brown"],
                                      "signal_var_by_layer": [x.get("signal_var") for x in dg["layers"]],
                                      "noise_var_by_layer": [x.get("noise_var") for x in dg["layers"]],
                                      "frac_layers_with_signal": float(np.mean([(x.get("signal_var") or 0) > 0
                                                                                for x in dg["layers"]])),
                                      "own_margin_halves": marg, "own_margin_mean": float(np.mean(marg)),
                                      "support_min": dg["support_min"], "support_cv": dg["support_cv"],
                                      "seconds": time.time() - t0}
            print(json.dumps({"t": t + 1, **{n: {k: out[n][str(t + 1)][k] for k in ("split_half_r", "own_margin_mean",
                                                                                   "coverage_valid")} for n in qs}}),
                  flush=True)
    return out, banks, {"abs_ce_change_unmodified_rows": {"mean": float(np.mean(noise)), "p99": float(np.quantile(noise, 0.99)),
                                                         "max": float(np.max(noise)), "n": len(noise)}}


def grad_anchor(m, bank, split, steps, pq, beta, device):
    import numpy as np
    import torch
    from csr.fsp_lib import make_batch, to_torch
    from csr.gc_lib import gc_edl_loss
    from csr.sue_cf import use_loss
    m.train()
    Z, V, _ = bank.targets()
    Zt, Vt = torch.as_tensor(Z, device=device, dtype=torch.float32), torch.as_tensor(V, device=device)
    params = [p for p in m.parameters() if p.requires_grad]
    rr = steps[318][:16]
    npb = make_batch(rr, split)
    bt = to_torch(npb, device)
    E = m.cfg.n_experts
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        out = m({"input_ids": bt["input_ids"], "doc_ids": bt["doc_ids"], "positions": bt["positions"]},
                {"collect": True})
        h = out["hidden"].reshape(-1, out["hidden"].shape[-1])
        y = bt["targets"].reshape(-1)
        lm = torch.nn.functional.cross_entropy(m.lm_logits(h).float(), y, ignore_index=-100)
    qs = torch.as_tensor(pq.qseq(npb), device=device)
    sid, nonpad = bt["seq_id"].reshape(-1), (bt["doc_ids"] >= 0).reshape(-1)
    pl = [out["records"][li]["p"].reshape(-1, E).float() for li in range(m.n_moe)]
    with torch.autocast(device_type=device.type, enabled=False):
        sem = torch.stack([gc_edl_loss(p, qs, sid, nonpad) for p in pl]).mean()
        lu, nact, marg = use_loss(pl, list(range(m.n_moe)), qs, sid, nonpad, Zt, Vt, 0.2)
    res = {"n_active_terms": nact, "use_value": float(lu), "use_margin": marg}
    for name, v in (("lm", lm), ("sem_edl_beta", beta * sem), ("use_unit", lu)):
        if not v.requires_grad:
            res[f"gn_{name}"] = 0.0
            continue
        gs = torch.autograd.grad(v, params, retain_graph=True, allow_unused=True)
        res[f"gn_{name}"] = float(torch.sqrt(sum((g.float() ** 2).sum() for g in gs if g is not None)))
    m.zero_grad(set_to_none=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--table-sha", required=True)
    ap.add_argument("--shuf-table", required=True)
    ap.add_argument("--shuf-table-sha", required=True)
    ap.add_argument("--beta", type=float, required=True)
    ap.add_argument("--ckpts", required=True, help="name=path/to/ckpt_dir,...")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--rows", type=int, default=16)
    ap.add_argument("--passes", type=int, default=4)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    if not guard.require_execution(a.execute, "gpu" if a.device == "cuda" else "cpu-model"):
        return
    import numpy as np
    import torch
    from fle_evaluate import load_milestone
    from csr.gc_lib import PaperQ
    from csr.lfr_lib import main_stream
    from csr.sue_cf import ProbePlan, probe_pool
    from csr.sue_paths import ART, FINAL_BUDGET, check_prereg
    check_prereg()
    outp = ART / "diagnostic.json"
    if outp.exists():
        raise guard.GuardError("diagnostic already written")
    mon = None
    if a.device == "cuda":
        guard.occupancy_gate("sue diagnostic")
        device = guard.setup_cuda_device()
        mon = guard.OccupancyMonitor(interval=15.0)
        mon.start()
    else:
        device = torch.device("cpu")
    guard.register_self("sue diagnostic")
    t_iso, t0 = guard.now(), time.time()
    split, steps, _ = main_stream(FINAL_BUDGET)
    pq = PaperQ(guard.ROOT / a.table, split, a.table_sha)
    pqs = PaperQ(guard.ROOT / a.shuf_table, split, a.shuf_table_sha)
    pool = probe_pool(split, steps, pq.q_of_doc)
    res = {"time": guard.now(), "table": a.table, "shuf_table": a.shuf_table, "pool_docs": int(len(pool)),
           "rows": a.rows, "passes": a.passes, "steps": a.steps, "models": {}}
    status = "COMPLETE"
    try:
        for spec in a.ckpts.split(","):
            name, path = spec.split("=")
            m, _ = load_milestone(path, device)
            plan = ProbePlan(pool, 100, a.rows, a.passes, range(m.n_moe))
            out, banks, noise = run_one(m, split, plan, {"SEM": pq.q_of_doc, "SHUF": pqs.q_of_doc}, device, a.steps,
                                        a.rows, a.passes)
            rec = {"ckpt": path, "by_size": out, "noise_floor": noise}
            if name.startswith("anchor"):
                rec["grad_anchor"] = grad_anchor(m, banks["SEM"], split, steps, pq, a.beta, device)
            res["models"][name] = rec
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
    res["status"], res["seconds"] = status, time.time() - t0
    if status == "COMPLETE":
        guard.atomic_write_json(outp, res, mode=0o400)
    if a.device == "cuda":
        guard.gpu_ledger(t_iso, guard.now(), "sue_diagnostic", "sue diagnostic", status)
        with open(ART / "compute_ledger.tsv", "a") as f:
            f.write(f"{t_iso}\t{guard.now()}\t{time.time() - t0:.1f}\tdiagnostic\tdiagnostic\tdiagnostic\t{status}\n")
    print(json.dumps({"status": status, "seconds": res["seconds"]}))
    sys.exit(0 if status == "COMPLETE" else 77)


if __name__ == "__main__":
    main()
