#!/usr/bin/env python
"""csr-sequence-routing-search-v1: focused GPU correctness checks on the full-size model (prereg §5).

For TOKEN-4 and every Generation-1 candidate (random init, real data):
  C1 training-mode bf16 forward/backward: finite loss and gradients; gradients reach the sequence-router params
  C2 fp32 (TF32 off) eval forward: every token of a segment has identical expert ids and gate values; exactly 4
     distinct routed experts per token; one route per segment; first segment uses the BOS route only
  C3 future perturbation (real CALIB-SEARCH documents): tokens after position c replaced -> logits, routes and
     boundaries at positions <= c unchanged
  C4 incremental decoding (sr_step, batch 1 and batch 4 lockstep) reproduces the full-sequence routes exactly and
     logits within 2e-3 (fp32)
  C5 peak VRAM of one training micro-batch (16 x 1024)
Dry-run by default.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--quick", action="store_true", help="CPU smoke: fewer rows/positions")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    if not guard.require_execution(a.execute, "gpu" if a.device == "cuda" else "cpu-model"):
        return
    import numpy as np
    import torch
    from csr.data import to_torch
    from csr.seqroute import build_sr_model
    from csr.srs_run import ART, eval_source, train_source
    from csr.srs_select import GEN1, TOKEN
    torch.set_num_threads(4)
    monitor = None
    if a.device == "cuda":
        guard.occupancy_gate("srsv1 gpu checks")
        device = guard.setup_cuda_device()
        monitor = guard.OccupancyMonitor(interval=15.0)
        monitor.start()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        device = torch.device("cpu")
    guard.register_self("srsv1 gpu checks")
    t_iso, t0 = guard.now(), time.time()
    base = json.loads(sorted((guard.ROOT / "configs/sequence-routing-search-v1/gen1/seed98").glob("TOKEN-4.json"))[0]
                      .read_text())
    tr, src, steps, _ = train_source(base, base["data"]["dir"])
    _, cs = eval_source(base, "calib_search", base["data"]["dir"])
    rows_train = steps[0][:16 if not a.quick else 2]
    tb_train = to_torch(src.batch(rows_train), device)
    # documents for C3/C4: first 4 CALIB-SEARCH docs with >= 300 tokens
    sp = cs.split
    order = [d for r in cs.rows for d in r]
    long_docs = [d for d in order if sp.lengths()[d] >= 300][:4]
    results, ok_all = {}, True
    specs = [{"id": "TOKEN-4", "family": "token", "routing": TOKEN}] + GEN1
    for spec in specs:
        cfg = json.loads(json.dumps(base))
        cfg["routing"] = spec["routing"]
        r = {}
        torch.manual_seed(0)
        model = build_sr_model(cfg, device)
        sr = model.sr
        if sr.dynamic:
            model.train()
            model.tau_calibrate(tb_train)
        # C1
        model.train()
        torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(tb_train)
        lg = out.logits.float()
        lm = torch.nn.functional.cross_entropy(lg.view(-1, lg.shape[-1]), tb_train["targets"].view(-1),
                                               ignore_index=-100)
        loss = lm + 0.01 * out.aux_loss + 0.001 * out.z_loss + sr.bd_coeff * out.bd_loss
        loss.backward()
        grads_finite = all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)
        extra = model.extra_parameter_names()
        # GRU cell weights sit behind the zero-initialized output layer gru_out, so their gradient is structurally
        # zero at initialization (it becomes nonzero after the first update of gru_out); they are excluded here.
        extra_grad = {n: (p.grad is not None and float(p.grad.abs().sum()) > 0) for n, p in model.named_parameters()
                      if n in extra and not n.startswith("gru.")}
        r["C1"] = {"loss": float(loss), "finite_loss": bool(torch.isfinite(loss)), "finite_grads": grads_finite,
                   "extra_params_receive_grad": extra_grad,
                   "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3 if device.type == "cuda" else 0}
        model.zero_grad(set_to_none=True)
        # C2 (fp32)
        model.eval()
        n4 = True
        seg_ok = True
        bos_ok = True
        with torch.no_grad():
            o = model(tb_train, collect=True)
        nonpad = tb_train["doc_ids"] >= 0
        for li, rec in enumerate(o.routing):
            ti = rec["topi"][nonpad]
            n4 &= bool((torch.sort(ti, -1).values.diff(dim=-1) > 0).all()) and ti.shape[-1] == 4
            if sr.mode == "sequence":
                gid = rec["seg_gid"][nonpad]
                gt = rec["gates"][nonpad]
                seg_ok &= bool(torch.equal(ti, rec["seg_topi"][gid])) and bool(torch.equal(gt, rec["seg_gates"][gid]))
                bos = torch.softmax(model.bos[li].float() / sr.temperature, -1).topk(4).indices
                j0 = rec["seg_j"] == 0
                bos_ok &= bool((rec["seg_topi"][j0] == bos.unsqueeze(0)).all())
        r["C2"] = {"exactly_4_distinct": n4, "segment_identical_ids_and_gates": seg_ok, "first_segment_bos": bos_ok}
        # C3 future perturbation and C4 incremental, on real documents
        c3, c4 = [], []
        n_inc = 2 if not a.quick else 1
        for k, di in enumerate(long_docs[:n_inc + (0 if a.quick else 2)]):
            ids = sp.doc(di)[:512 if not a.quick else 300]
            L = len(ids)
            cut = 180 + 17 * k
            other = np.resize(sp.doc(long_docs[(k + 1) % len(long_docs)]), L)  # repeat if shorter (fix D1)
            pert = ids.copy()
            pert[cut + 1:] = other[cut + 1:L]
            b1, b2 = _one(ids, device), _one(pert, device)
            with torch.no_grad():
                o1, o2 = model(b1, collect=True), model(b2, collect=True)
            dl = float((o1.logits[0, :cut + 1] - o2.logits[0, :cut + 1]).abs().max())
            rt = all(torch.equal(x["topi"][0, :cut + 1], y["topi"][0, :cut + 1]) for x, y in zip(o1.routing, o2.routing))
            bd = True
            if sr.mode == "sequence":
                bd = all(torch.equal(x["starts"][0, :cut + 2], y["starts"][0, :cut + 2])
                         for x, y in zip(o1.routing, o2.routing))
            c3.append({"doc": int(di), "cut": cut, "max_logit_diff": dl, "routes_equal": rt, "boundaries_equal": bd})
            if k < n_inc:
                st = model.new_sr_state(1)
                mx, mism = 0.0, 0
                with torch.no_grad():
                    for p, tok in enumerate(ids.tolist()):
                        lgi, rts = model.sr_step(torch.tensor([tok], device=device), p, st)
                        mx = max(mx, float((lgi[0] - o1.logits[0, p]).abs().max()))
                        mism += sum(int(not torch.equal(rr[0], o1.routing[li]["topi"][0, p])) for li, rr in enumerate(rts))
                c4.append({"doc": int(di), "len": L, "max_logit_diff": mx, "route_mismatches": mism,
                           "router_calls": st["router_calls"]})
        # batched lockstep decoding (4 docs, common length)
        if not a.quick:
            Lb = 256
            docsb = [sp.doc(d)[:Lb] for d in long_docs[:4]]
            st = model.new_sr_state(4)
            mism, mx = 0, 0.0
            fulls = []
            with torch.no_grad():
                for dd in docsb:
                    fulls.append(model(_one(dd, device), collect=True))
                for p in range(Lb):
                    lgi, rts = model.sr_step(torch.tensor([int(dd[p]) for dd in docsb], device=device), p, st)
                    for bi in range(4):
                        mx = max(mx, float((lgi[bi] - fulls[bi].logits[0, p]).abs().max()))
                        mism += sum(int(not torch.equal(rr[bi], fulls[bi].routing[li]["topi"][0, p]))
                                    for li, rr in enumerate(rts))
            c4.append({"batched_lockstep_4": True, "len": Lb, "max_logit_diff": mx, "route_mismatches": mism})
        r["C3"] = c3
        r["C4"] = c4
        ok = (r["C1"]["finite_loss"] and r["C1"]["finite_grads"] and all(r["C1"]["extra_params_receive_grad"].values())
              and n4 and seg_ok and bos_ok and all(x["routes_equal"] and x["boundaries_equal"] and x["max_logit_diff"] < 1e-4
                                                   for x in c3)
              and all(x["route_mismatches"] == 0 and x["max_logit_diff"] < 2e-3 for x in c4))
        r["pass"] = bool(ok)
        ok_all &= ok
        results[spec["id"]] = r
        print(spec["id"], "PASS" if ok else "FAIL", json.dumps({k: v for k, v in r.items() if k != "C3"})[:600], flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if monitor is not None:
        monitor.stop()
    status = "CONTAMINATED" if monitor is not None and monitor.contaminated else ("PASS" if ok_all else "FAIL")
    out = ART / f"gpu_checks_{guard.utc_stamp()}{'_quick' if a.quick else ''}.json"
    guard.atomic_write_json(out, {"status": status, "device": str(device), "results": results,
                                  "seconds": time.time() - t0})
    if a.device == "cuda":
        guard.gpu_ledger(t_iso, guard.now(), out.stem, "srsv1 gpu checks", status)
    guard.ledger("SRS_V1_GPU_CHECKS", f"{out.relative_to(guard.ROOT)} status={status}")
    print(f"CHECKS_JSON {out.relative_to(guard.ROOT)} {status}")
    sys.exit(0 if status == "PASS" else 1)


def _one(ids, device):
    import numpy as np
    import torch
    L = len(ids)
    tgt = np.full(L, -100)
    tgt[:-1] = ids[1:]
    t = lambda x: torch.as_tensor(np.asarray(x, dtype=np.int64), device=device).unsqueeze(0)  # noqa: E731
    return {"input_ids": t(ids), "doc_ids": t(np.zeros(L)), "positions": t(np.arange(L)), "targets": t(tgt),
            "gdoc": t(np.zeros(L))}


if __name__ == "__main__":
    main()
