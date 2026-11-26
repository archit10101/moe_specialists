#!/usr/bin/env python
"""csr-representative-v1 tau_s / tau_r selection (docs/REP_V1_PREREG.md §6), training data only.

At initialization of the seed-99 E model, on the first 8 training micro-batches in seed-99 order (train_order_seed
116), with the training comparison set (n_max, dedup): per-chunk effective neighbour counts exp H(Q_i) (semantic)
and exp H(R_i^l) averaged over MoE layers (router). tau_s = grid value whose median exp H(Q) is closest to 8;
tau_r = grid value whose median exp H(R) is closest to the selected semantic median. Gate: the two selected medians
differ by <= 20% of the semantic median, else STOP. Dry-run by default; GPU via scripts/gpu_guarded.sh.

AMENDMENT 1 (docs/REP_V1_AMENDMENT_1.md): the discrete-grid selection is replaced by deterministic log-space bisection
on the SAME frozen inputs (S and G computed once): tau_s in [0.05, 0.1] toward 8 (gate: count in [6.4, 9.6]); tau_r in
[2e-5, 5e-5] toward the accepted tau_s count (gate: relative gap <= 20%). <= 20 midpoint iterations each, closest
evaluated temperature wins, ties -> smaller temperature. The original grid points are re-evaluated for provenance."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402
from train import load_seg, plan, resolve_config  # noqa: E402

TAU_S_GRID = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
TAU_R_GRID = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2]
TARGET = 8.0
MAX_REL_GAP = 0.20
N_BATCHES = 8
# amendment 1
BRACKET_S = (0.05, 0.1)
BRACKET_R = (2e-5, 5e-5)
MAX_ITER = 20
S_GATE = (6.4, 9.6)  # within 20% of 8
FAILED_RUN = "runs/20261101T043501Z_rep_calibrate_tau"
AMENDMENT = "docs/REP_V1_AMENDMENT_1.md"


def eff_counts(logits_or_S, tau):
    """Per-row exp(entropy) of the off-diagonal softmax of (scores / tau). scores (..., N, N)."""
    import torch
    from csr.semantic_loss import _offdiag_log_softmax
    logp, eye = _offdiag_log_softmax(logits_or_S / tau)
    H = -(logp.exp() * logp).masked_fill(eye, 0.0).sum(-1)
    return torch.exp(H)


def select(q_med, r_med):
    tau_s = min(TAU_S_GRID, key=lambda t: abs(q_med[t] - TARGET))
    tau_r = min(TAU_R_GRID, key=lambda t: abs(r_med[tau_s][t] - q_med[tau_s]))
    gap = abs(r_med[tau_s][tau_r] - q_med[tau_s]) / q_med[tau_s]
    return tau_s, tau_r, gap, gap <= MAX_REL_GAP


def log_midpoint(lo, hi):
    import math
    return math.exp(0.5 * (math.log(lo) + math.log(hi)))


def bisect_log(f, lo, hi, target, max_iter=MAX_ITER):
    """Deterministic log-space bisection for an increasing f. Evaluates both endpoints, then <= max_iter midpoints;
    f(m) < target -> lo = m else hi = m; stops early only on exact equality. Never expands the bracket.
    Returns the trace [{"tau", "value", "role", "lo", "hi"}] in evaluation order."""
    trace = [{"tau": lo, "value": float(f(lo)), "role": "bracket_lo", "lo": lo, "hi": hi},
             {"tau": hi, "value": float(f(hi)), "role": "bracket_hi", "lo": lo, "hi": hi}]
    for it in range(max_iter):
        m = log_midpoint(lo, hi)
        v = float(f(m))
        trace.append({"tau": m, "value": v, "role": f"iter{it + 1}", "lo": lo, "hi": hi})
        if v == target:
            break
        if v < target:
            lo = m
        else:
            hi = m
    return trace


def select_closest(trace, target):
    """Evaluated temperature with the smallest |value - target|; ties -> smaller temperature."""
    return min(trace, key=lambda r: (abs(r["value"] - target), r["tau"]))


def calibrate_from_frozen(S_list, G_list, max_iter=MAX_ITER):
    """Amendment-1 search on frozen semantic similarity matrices S_b (N_b, N_b) and router overlap matrices
    G_b (L, N_b, N_b). Same effective-neighbour calculation as the original grid rule."""
    import math
    import numpy as np

    def f_s(t):
        return float(np.median(np.concatenate([eff_counts(S, t).cpu().numpy() for S in S_list])))

    def f_r(t):
        return float(np.median(np.concatenate([eff_counts(G, t).mean(0).cpu().numpy() for G in G_list])))
    tr_s = bisect_log(f_s, *BRACKET_S, TARGET, max_iter)
    sel_s = select_closest(tr_s, TARGET)
    fin_s = all(math.isfinite(r["value"]) for r in tr_s)
    gate_s = fin_s and S_GATE[0] <= sel_s["value"] <= S_GATE[1]
    out = {"tau_s_trace": tr_s, "tau_s": sel_s["tau"], "selected_q_eff": sel_s["value"], "tau_s_all_finite": fin_s,
           "tau_s_gate": [S_GATE[0], S_GATE[1]], "tau_s_gate_pass": gate_s}
    if not gate_s:
        out.update({"gate_pass": False, "failed_criterion": f"tau_s count {sel_s['value']:.4f} outside {S_GATE} "
                                                              f"(or non-finite={not fin_s})"})
        return out
    tr_r = bisect_log(f_r, *BRACKET_R, sel_s["value"], max_iter)
    sel_r = select_closest(tr_r, sel_s["value"])
    fin_r = all(math.isfinite(r["value"]) for r in tr_r)
    gap = abs(sel_r["value"] - sel_s["value"]) / sel_s["value"]
    gate_r = fin_r and gap <= MAX_REL_GAP
    out.update({"tau_r_trace": tr_r, "tau_r": sel_r["tau"], "selected_r_eff": sel_r["value"], "relative_gap": gap,
                "relative_gap_pct": 100 * gap, "tau_r_all_finite": fin_r, "gate_max_relative_gap": MAX_REL_GAP,
                "tau_r_gate_pass": gate_r, "gate_pass": gate_r})
    if not gate_r:
        out["failed_criterion"] = f"tau_r relative gap {gap:.4f} > {MAX_REL_GAP} (or non-finite={not fin_r})"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="configs/rep-v1/precal/base.json")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    sets = ["init.seed=99", "train.seed=99", "data.train_order_seed=116"]
    cfg = resolve_config(argparse.Namespace(base=a.base, condition="E", set=sets, run_name="rep_calibrate_tau"))
    print(f"rep tau calibration: grids tau_s={TAU_S_GRID} tau_r={TAU_R_GRID}; init seed 99; {N_BATCHES} train batches")
    if not guard.require_execution(a.execute, "gpu" if a.device == "cuda" else "cpu-model"):
        return
    mon = None
    if a.device == "cuda":
        guard.occupancy_gate("rep_calibrate_tau")
        dev = guard.setup_cuda_device()
        mon = guard.OccupancyMonitor(10.0)
        mon.start()
    import numpy as np
    import torch
    from csr.data import to_torch, train_order
    from csr.semantic_loss import select_valid_chunks
    from csr.train_lib import BatchSource, build_model, routing_from_cfg
    if a.device == "cpu":
        dev = torch.device("cpu")
    guard.register_self("rep_calibrate_tau")
    rd = guard.RunDir("rep_calibrate_tau_amend1", cfg)
    t0iso = guard.now()
    lim = guard.RunLimits(30 * 60, monitor=mon)
    status, reason, res = "INCOMPLETE", "", {}
    try:
        d, tr, _, _ = plan(cfg, require_semantic_calibration=False)  # this run produces tau_s / tau_r
        seg = load_seg(cfg, "train")
        src = BatchSource(tr, cfg["data"]["seq_len"], seg, train_order(tr.n_docs, cfg["data"]["train_order_seed"]))
        model = build_model(cfg, dev).eval()
        mb = cfg["train"]["micro_batch_rows"]
        S_list, G_list, n_chunks = [], [], []
        for bi in range(N_BATCHES):
            tb = to_torch(src.batch(src.rows[bi * mb:(bi + 1) * mb]), dev)
            with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                out = model(tb, routing_from_cfg(cfg), need_chunk_logits=True)
            lay = out.layout
            keys = tb["chunk_key"][lay.rows_b, lay.rows_t0].cpu().numpy()
            u = torch.as_tensor(seg.u[keys].astype(np.float32), device=dev)
            idx = select_valid_chunks(u, seg.phash[keys], cfg["semantic"]["n_max"])
            assert idx.numel() >= cfg["semantic"]["min_chunks"]
            n_chunks.append(int(idx.numel()))
            uu = torch.nn.functional.normalize(u[idx].float(), dim=-1)
            S = uu @ uu.T
            p = torch.softmax(torch.stack([r["chunk_logits"] for r in out.routing])[:, idx].float(), dim=-1)
            G = p @ p.transpose(1, 2)
            S_list.append(S)  # frozen: every temperature below is evaluated on these exact matrices
            G_list.append(G)
            lim.check()
        # provenance: the original grid points on the frozen inputs vs the failed run's recorded values
        q_med = {t: float(np.median(np.concatenate([eff_counts(S, t).cpu().numpy() for S in S_list]))) for t in TAU_S_GRID}
        r_med = {t: float(np.median(np.concatenate([eff_counts(G, t).mean(0).cpu().numpy() for G in G_list])))
                 for t in TAU_R_GRID}
        old = json.loads((guard.ROOT / FAILED_RUN / "summary.json").read_text())
        repro = {"failed_run": FAILED_RUN, "n_chunks_identical": old["n_chunks_per_batch"] == n_chunks,
                 "max_abs_diff_q": max(abs(q_med[t] - old["q_eff_median"][str(t)]) for t in TAU_S_GRID),
                 "max_abs_diff_r": max(abs(r_med[t] - old["r_eff_median"][str(t)]) for t in TAU_R_GRID)}
        cal = calibrate_from_frozen(S_list, G_list)
        res = {"method": "amendment-1 deterministic log-space bisection on frozen inputs",
               "amendment_sha256": guard.sha256_file(guard.ROOT / AMENDMENT), **cal,
               "bracket_s": list(BRACKET_S), "bracket_r": list(BRACKET_R), "max_iter": MAX_ITER,
               "target_neighbours": TARGET, "grid_reevaluation": {"q_eff_median": q_med, "r_eff_median": r_med, **repro},
               "n_chunks_per_batch": n_chunks, "init_seed": 99, "train_order_seed": 116, "batches": N_BATCHES}
        ok = cal["gate_pass"]
        status = "COMPLETE" if ok else "ABORTED"
        reason = "" if ok else f"amended tau gate failed: {cal['failed_criterion']} (STOP per amendment 1)"
    except guard.GuardError as e:
        status, reason = ("CONTAMINATED" if "CONTAMINATED" in str(e) else "ABORTED"), str(e)
    except Exception as e:  # noqa: BLE001
        import traceback
        status, reason = "FAILED", f"{type(e).__name__}: {e}"
        rd.log(traceback.format_exc())
    finally:
        lim.close()
        if mon is not None:
            mon.stop()
            if mon.contaminated and status == "COMPLETE":
                status, reason = "CONTAMINATED", mon.detail
        guard.atomic_write_json(rd.path / "summary.json", {"status": status, "reason": reason,
                                                           "result_label": cfg.get("result_label"), **res})
        rd.set_status(status, reason=reason)
        if a.device == "cuda":
            guard.gpu_ledger(t0iso, guard.now(), rd.run_id, "rep_calibrate_tau", status)
        print(json.dumps(res, indent=1), "STATUS", status, reason)
    sys.exit(0 if status == "COMPLETE" else 1)


if __name__ == "__main__":
    main()
