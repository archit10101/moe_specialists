#!/usr/bin/env python
"""Stage B items 1-2: minimal load + forward check and GPU baseline-equivalence / causal-leakage checks on tiny
synthetic and tiny fixed real examples. No training. Dry-run by default; launch via scripts/gpu_guarded.sh."""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE / "tests"))
from csr import guard  # noqa: E402
from train import resolve_config, load_seg, check_data  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="configs/v2/base.json")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    print(f"config {a.base}")
    print("GPU checks: tiny-model reference equivalence, causal invariance, incremental==parallel (fp32);"
          " full-size forward fp32/bf16 on 4 real rows; full-size causal check with semantic boundaries")
    if not guard.require_execution(a.execute, "gpu"):
        return
    guard.occupancy_gate("gpu_checks")
    dev = guard.setup_cuda_device()
    mon = guard.OccupancyMonitor(10.0)
    mon.start()
    import numpy as np
    import torch
    from conftest import single_doc_batch, tiny_cfg
    from test_model import reference_forward, rand_ids
    from csr.data import TokenSplit, to_torch
    from csr.model import CSRModel, RoutingConfig
    from csr.segment import fixed_starts, starts_to_arrays
    from csr.train_lib import BatchSource, build_model, forward_losses
    guard.register_self("gpu_checks")
    cfg = resolve_config(argparse.Namespace(base=a.base, condition="E", set=None, run_name="gpu_checks"))
    placeholders = {}
    for k, v in (("tau_s", 0.1), ("tau_r", 0.01), ("lambda", 0.0)):  # finiteness check only; never used for training
        if cfg["semantic"][k] is None:
            cfg["semantic"][k] = placeholders[k] = v
    rd = guard.RunDir("gpu_checks", cfg)
    t0iso = guard.now()
    lim = guard.RunLimits(30 * 60, monitor=mon)
    res, status, reason = {}, "INCOMPLETE", ""
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        TOK, POOL = RoutingConfig("token", 4, 4), RoutingConfig("pool", 4, 4)
        # ---- tiny model, fp32 on GPU
        torch.manual_seed(0)
        tm = CSRModel(tiny_cfg()).to(dev).eval()
        ids = rand_ids(40)
        b = {k: v.to(dev) for k, v in single_doc_batch(ids, [0]).items()}
        with torch.no_grad():
            out = tm(b, TOK)
            ref, aux, z, topis = reference_forward(tm.cpu().double(), ids)
        tm = tm.float().to(dev)
        res["tiny_ref_max_abs_logit_diff"] = float((out.logits[0].double().cpu() - ref).abs().max())
        res["tiny_ref_aux_diff"] = abs(float(out.aux_loss) - aux)
        res["tiny_ref_routing_equal"] = all([sorted(x) for x in r["topi"][0].tolist()] == tl
                                            for r, tl in zip(out.routing, topis))
        # causal invariance + incremental on tiny
        ids = rand_ids(90, 3)
        st = fixed_starts(90, 20)
        ids2 = ids.copy()
        cut = 47
        ids2[cut + 1:] = rand_ids(90 - cut - 1, 99)
        with torch.no_grad():
            o1 = tm({k: v.to(dev) for k, v in single_doc_batch(ids, st).items()}, POOL)
            o2 = tm({k: v.to(dev) for k, v in single_doc_batch(ids2, st).items()}, POOL)
        res["tiny_causal_max_prefix_diff"] = float((o1.logits[0, :cut + 1] - o2.logits[0, :cut + 1]).abs().max())
        res["tiny_causal_routing_equal"] = all(torch.equal(r1["topi"][0, :cut + 1], r2["topi"][0, :cut + 1])
                                               for r1, r2 in zip(o1.routing, o2.routing))
        _, cpos = starts_to_arrays(90, st)
        with torch.no_grad():
            state = tm.new_state()
            mx, req = 0.0, True
            for t in range(90):
                lg, rt = tm.step(int(ids[t]), t, int(cpos[t]), POOL, state)
                mx = max(mx, float((lg - o1.logits[0, t]).abs().max()))
                req &= all(torch.equal(tp, o1.routing[li]["topi"][0, t]) for li, (tp, _) in enumerate(rt))
        res["tiny_incremental_max_diff"], res["tiny_incremental_routing_equal"] = mx, req
        lim.check()
        # ---- full-size model on 4 real training rows
        d = check_data(cfg)
        tr = TokenSplit.load(d, "train")
        seg = load_seg(cfg, "train")
        src = BatchSource(tr, cfg["data"]["seq_len"], seg)
        tb = to_torch(src.batch(src.rows[:4]), dev)
        model = build_model(cfg, dev)
        torch.cuda.reset_peak_memory_stats()
        for name, amp in (("fp32", False), ("bf16", True)):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                t0 = time.time()
                parts, out = forward_losses(model, tb, cfg, seg)
                torch.cuda.synchronize()
            res[f"full_{name}_ce_init"] = float(parts.lm_sum) / parts.n_valid
            res[f"full_{name}_forward_s"] = time.time() - t0
            res[f"full_{name}_coverage"] = parts.coverage_constrained / parts.n_valid
            res[f"full_{name}_sem"] = float(parts.sem)
        res["ln_vocab"] = float(np.log(cfg["model"]["vocab_size"]))
        res["semantic_placeholders_used"] = placeholders
        res["result_label"] = cfg.get("result_label", "experiment")
        # full-size causal check: alter the second half of row 0's first document, keep boundaries <= cut
        from csr.train_lib import routing_from_cfg
        FULL_POOL = routing_from_cfg({**cfg, "routing": {**cfg["routing"], "mode": "pool"}})
        di = src.rows[0][0]
        doc = tr.doc(di)
        cut = len(doc) // 2
        rows_ids = tb["input_ids"].clone()
        rows_ids[0, cut + 1:len(doc)] = torch.randint(2, 50000, (len(doc) - cut - 1,), device=dev)
        tb2 = dict(tb)
        tb2["input_ids"] = rows_ids
        with torch.no_grad():
            o1 = model(tb, FULL_POOL)
            o2 = model(tb2, FULL_POOL)
        res["full_causal_max_prefix_logit_diff"] = float((o1.logits[0, :cut + 1] - o2.logits[0, :cut + 1]).abs().max())
        res["full_causal_routing_equal"] = all(torch.equal(r1["topi"][0, :cut + 1], r2["topi"][0, :cut + 1])
                                               for r1, r2 in zip(o1.routing, o2.routing))
        res["full_other_rows_unchanged"] = float((o1.logits[1:] - o2.logits[1:]).abs().max())
        with torch.no_grad():
            o3 = model(tb, FULL_POOL)
        res["full_repeat_forward_max_diff"] = float((o1.logits - o3.logits).abs().max())
        res["peak_alloc_gib"], res["peak_reserved_gib"] = lim.peak_gib()
        ok = (res["tiny_ref_max_abs_logit_diff"] < 1e-4 and res["tiny_ref_routing_equal"]
              and res["tiny_causal_max_prefix_diff"] < 1e-5 and res["tiny_causal_routing_equal"]
              and res["tiny_incremental_max_diff"] < 1e-4 and res["tiny_incremental_routing_equal"]
              and res["full_causal_max_prefix_logit_diff"] < 1e-4 and res["full_causal_routing_equal"]
              and res["full_other_rows_unchanged"] < 1e-4 and abs(res["full_fp32_ce_init"] - res["ln_vocab"]) < 0.5)
        res["all_pass"] = bool(ok)
        status = "COMPLETE" if ok else "FAILED"
        reason = "" if ok else "a GPU correctness check exceeded tolerance"
    except guard.GuardError as e:
        status, reason = ("CONTAMINATED" if "CONTAMINATED" in str(e) else "ABORTED"), str(e)
    except Exception as e:  # noqa: BLE001
        status, reason = "FAILED", f"{type(e).__name__}: {e}"
        rd.log(traceback.format_exc())
    finally:
        lim.close()
        mon.stop()
        guard.atomic_write_json(rd.path / "summary.json", {"status": status, "reason": reason, **res})
        rd.set_status(status, reason=reason)
        guard.gpu_ledger(t0iso, guard.now(), rd.run_id, "gpu_checks", status)
        print(json.dumps(res, indent=1))
        print("STATUS", status, reason)


if __name__ == "__main__":
    main()
