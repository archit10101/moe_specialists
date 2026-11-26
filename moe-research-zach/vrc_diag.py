#!/usr/bin/env python
"""csr-value-routed-compute-v1 mandatory counterfactual diagnostic gate (protocol A.5) on CALIB-SEARCH.
  scripts/gpu_guarded.sh "vrc diag" vrc_diag.py --run <FULL-TOKEN4 run dir> --execute
Uniform-policy NLL per 128-token block for REF and the six (attn x k) options, measured prefill times, oracle
frontier (lambda sweep), exact re-evaluation of gate-passing points, labels V_context / V_2 / V_4 / V_joint."""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402

OPTS = [(a, k) for a in ("L", "G") for k in (1, 2, 4)]
GATE = {"dce": 0.010, "expert_ratio": 0.75, "global_ratio": 0.50, "time_ratio": 0.90}
LAMBDAS = [0.0] + list(np.logspace(-6, 2, 59))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    if not guard.require_execution(a.execute, "gpu"):
        return
    import torch
    import torch.nn.functional as F
    from csr.data import to_torch
    from csr.vm_run import EXP, build_model, check_cfg, eval_source
    run = guard.inside_root(a.run)
    cfg = json.loads((run / "config.json").read_text())
    key, d = check_cfg(cfg)
    guard.occupancy_gate("vrc diag")
    dev = guard.setup_cuda_device()
    mon = guard.OccupancyMonitor(interval=15.0)
    mon.start()
    guard.register_self("vrc diag")
    t0iso, t0 = guard.now(), time.time()
    out = {"run": str(run.relative_to(guard.ROOT)), "checkpoint_sha256": guard.sha256_file(run / "model.pt"),
           "gate_thresholds": GATE, "options": [f"{x}{k}" for x, k in OPTS]}
    status = "FAILED"
    try:
        model = build_model(cfg, dev, backend="flex")
        sd = torch.load(run / "model.pt", map_location="cpu", weights_only=False)["model"]
        model.load_state_dict(sd)
        model.eval()
        src = eval_source(cfg, key, d, "calib_search")
        T = cfg["data"]["seq_len"]
        nb = T // 128
        rpb = 4
        rows = src.rows

        def kw(o, B, gsel=None, kblk=None):
            if o == "REF":
                return {}
            att, k = o
            if gsel is None:
                gsel = torch.full((B, nb), att == "G", dtype=torch.bool, device=dev)
                kblk = torch.full((B, nb), k, dtype=torch.long, device=dev)
            return {"ref": False, "gsel": gsel, "kblk": kblk}

        policies = ["REF"] + OPTS
        blk_nll = {str(p): [] for p in policies}
        blk_n, blk_tok, blk_pos = [], [], []
        evals = {str(p): 0 for p in policies}
        with torch.no_grad():
            for s in range(0, len(rows), rpb):
                tb = to_torch(src.batch(rows[s:s + rpb]), dev)
                B = tb["input_ids"].shape[0]
                valid = tb["targets"] != -100
                nonpad = tb["doc_ids"] >= 0
                blk_n.append(valid.view(B, nb, 128).sum(-1).reshape(-1).cpu().numpy())
                blk_tok.append(nonpad.view(B, nb, 128).sum(-1).reshape(-1).cpu().numpy())
                blk_pos.append(tb["positions"].view(B, nb, 128)[:, :, 0].reshape(-1).cpu().numpy())
                for p in policies:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        o = model(tb, **kw(p, B))
                    nll = F.cross_entropy(o.logits.float().view(-1, o.logits.shape[-1]), tb["targets"].view(-1),
                                          ignore_index=-100, reduction="none").view(B, T)
                    if not torch.isfinite(nll[valid]).all():
                        raise FloatingPointError(f"non-finite NLL for policy {p}")
                    blk_nll[str(p)].append((nll * valid).view(B, nb, 128).sum(-1).double().reshape(-1).cpu().numpy())
                    evals[str(p)] += o.n_expert_evals
                mon_check(mon)
        bn = {p: np.concatenate(v) for p, v in blk_nll.items()}
        n = np.concatenate(blk_n).astype(np.float64)
        tok = np.concatenate(blk_tok).astype(np.float64)
        pos0 = np.concatenate(blk_pos)
        live = tok > 0
        N = n.sum()
        ce = {p: float(bn[p].sum() / N) for p in bn}
        out["n_rows"] = len(rows)
        out["n_blocks_nonempty"] = int(live.sum())
        out["n_valid_tokens"] = int(N)
        out["uniform_ce"] = ce
        out["uniform_dce_vs_ref"] = {p: ce[p] - ce["REF"] for p in ce}
        out["uniform_expert_evals"] = evals
        # labels (per block, nats per valid token of the block)
        lab = {}
        w = np.where(n > 0, n, 1)
        V = {"V_context": (bn[str(("L", 4))] - bn[str(("G", 4))]) / w,
             "V_2": (bn[str(("G", 1))] - bn[str(("G", 2))]) / w,
             "V_4": (bn[str(("G", 2))] - bn[str(("G", 4))]) / w,
             "V_joint": (bn[str(("L", 1))] - bn[str(("G", 4))]) / w}
        m = n > 0
        for k_, v in V.items():
            vv = v[m]
            lab[k_] = {"token_weighted_mean": float((v * n)[m].sum() / n[m].sum()), "q05": float(np.quantile(vv, .05)),
                       "q50": float(np.median(vv)), "q95": float(np.quantile(vv, .95)),
                       "frac_positive": float((vv > 0).mean()), "frac_gt_0.01": float((vv > 0.01).mean())}
        far = m & (pos0 >= 512)
        lab["V_context_blocks_at_doc_pos_ge_512"] = {
            "n_blocks": int(far.sum()), "token_weighted_mean": float((V["V_context"] * n)[far].sum() / max(n[far].sum(), 1)),
            "frac_positive": float((V["V_context"][far] > 0).mean()) if far.any() else None}
        out["labels"] = lab
        out["label_sample"] = "all CALIB-SEARCH 128-token blocks (deterministic: complete partition)"
        # ---- timing (identical prompts: the first 4 CALIB-SEARCH rows; bf16; no grad; 5 warm-up + 20 timed)
        tbb = to_torch(src.batch(rows[:4]), dev)
        Bb = 4

        def timeit(kwargs, reps=20, warm=5):
            ts = []
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for i in range(warm + reps):
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    model(tbb, **kwargs)
                    torch.cuda.synchronize()
                    if i >= warm:
                        ts.append(time.perf_counter() - t1)
            return {"median_s": float(np.median(ts)), "p95_s": float(np.quantile(ts, .95)), "min_s": float(min(ts))}
        times = {}
        for p in policies:
            times[str(p)] = timeit(kw(p, Bb))
        for p in policies:  # second pass, report both; use the per-policy minimum of the two medians? no: mean of medians
            t2 = timeit(kw(p, Bb))
            times[str(p)]["median_s_pass2"] = t2["median_s"]
            times[str(p)]["median_s_used"] = 0.5 * (times[str(p)]["median_s"] + t2["median_s"])
        out["prefill_times_4x8192"] = times
        Tref = times["REF"]["median_s_used"]
        rel = {str(o): times[str(o)]["median_s_used"] / Tref for o in OPTS}
        out["uniform_time_ratio_vs_ref"] = rel
        # ---- oracle frontier
        dn = np.stack([bn[str(o)] - bn["REF"] for o in OPTS], 1)  # (nblk, 6)
        dn_fb = np.stack([bn[str(o)] - bn[str(("G", 4))] for o in OPTS], 1)
        cost = np.array([rel[str(o)] for o in OPTS])
        kk = np.array([k for _, k in OPTS], dtype=np.float64)
        gg = np.array([1.0 if x == "G" else 0.0 for x, _ in OPTS])

        def frontier(delta):
            pts = []
            for lam in LAMBDAS:
                ch = np.argmin(delta + lam * tok[:, None] * cost[None, :], axis=1)
                ch = np.where(live, ch, OPTS.index(("L", 1)))
                fr = np.bincount(ch[live], minlength=6) / live.sum()
                pts.append({"lambda": float(lam), "dce": float(delta[np.arange(len(ch)), ch].sum() / N),
                            "expert_ratio": float((kk[ch] * tok).sum() / (4 * tok.sum())),
                            "global_ratio": float(gg[ch][live].mean()),
                            "time_ratio_projected": float((fr * cost).sum()),
                            "option_fractions": {f"{x}{k}": float(f) for (x, k), f in zip(OPTS, fr)},
                            "_choice": ch})
            return pts
        pts = frontier(dn)
        for p in pts:
            p["passes_projection"] = bool(p["dce"] <= GATE["dce"] and p["expert_ratio"] <= GATE["expert_ratio"] and
                                          p["global_ratio"] <= GATE["global_ratio"] and
                                          p["time_ratio_projected"] <= GATE["time_ratio"])
        cand = sorted([p for p in pts if p["passes_projection"]], key=lambda p: p["time_ratio_projected"])
        exact = []
        for p in cand[:10]:
            ch = p["_choice"]
            gsel_all = torch.as_tensor(gg[ch] > 0)
            k_all = torch.as_tensor(kk[ch].astype(np.int64))
            tot, idx = 0.0, 0
            with torch.no_grad():
                for s in range(0, len(rows), rpb):
                    tb = to_torch(src.batch(rows[s:s + rpb]), dev)
                    B = tb["input_ids"].shape[0]
                    gs = gsel_all[idx:idx + B * nb].view(B, nb).to(dev)
                    kb = k_all[idx:idx + B * nb].view(B, nb).to(dev)
                    idx += B * nb
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        o = model(tb, ref=False, gsel=gs, kblk=kb)
                    tot += float(F.cross_entropy(o.logits.float().view(-1, o.logits.shape[-1]),
                                                 tb["targets"].view(-1), ignore_index=-100, reduction="sum"))
            dce_exact = tot / N - ce["REF"]
            gs_b = gsel_all[:Bb * nb].view(Bb, nb).to(dev)
            kb_b = k_all[:Bb * nb].view(Bb, nb).to(dev)
            tm = timeit({"ref": False, "gsel": gs_b, "kblk": kb_b})
            exact.append({"lambda": p["lambda"], "dce_exact": dce_exact, "measured_mixed_time_ratio_bench_rows":
                          tm["median_s"] / Tref, "passes_exact": bool(dce_exact <= GATE["dce"])})
            mon_check(mon)
        out["exact_reevaluations"] = exact
        out["frontier_vs_ref"] = [{k: v for k, v in p.items() if k != "_choice"} for p in pts]
        fb = frontier(dn_fb)
        out["frontier_vs_fallback_G4_secondary"] = [{k: v for k, v in p.items() if k != "_choice"} for p in fb]
        # best achievable on each gate dimension subject to the CE constraint (diagnostic of the failure reason)
        ok_ce = [p for p in pts if p["dce"] <= GATE["dce"]]
        out["best_under_ce_constraint"] = {
            "min_expert_ratio": min((p["expert_ratio"] for p in ok_ce), default=None),
            "min_global_ratio": min((p["global_ratio"] for p in ok_ce), default=None),
            "min_time_ratio_projected": min((p["time_ratio_projected"] for p in ok_ce), default=None),
            "n_points_ce_ok": len(ok_ce)}
        out["min_time_ratio_any_point"] = min(p["time_ratio_projected"] for p in pts)
        passed = any(e["passes_exact"] for e in exact)
        out["gate_pass"] = bool(passed)
        out["gate_note"] = ("PASS: at least one oracle operating point meets all four criteria (exact CE re-check)"
                            if passed else "FAIL: no oracle operating point meets all four criteria simultaneously")
        if mon.contaminated:
            raise guard.GuardError(f"CONTAMINATED: {mon.detail}")
        status = "COMPLETE"
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc()
        status = "CONTAMINATED" if "CONTAMINATED" in str(e) else "FAILED"
    finally:
        mon.stop()
        out["status"] = status
        out["seconds"] = time.time() - t0
        st = guard.utc_stamp()
        p = EXP["vrc"]["art"] / f"diagnostic_{st}.json"
        guard.atomic_write_json(p, out)
        guard.gpu_ledger(t0iso, guard.now(), f"vrc_diag_{st}", "vrc diag", status)
        print(json.dumps({k: v for k, v in out.items() if not k.startswith("frontier")}, indent=1, default=str)[:6000])
        print("DIAG_JSON", p)


def mon_check(mon):
    if mon.contaminated:
        raise guard.GuardError(f"CONTAMINATED: {mon.detail}")


if __name__ == "__main__":
    main()
