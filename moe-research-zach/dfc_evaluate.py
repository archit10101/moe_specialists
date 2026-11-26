#!/usr/bin/env python
"""csr-domain-forecast-curriculum-v1: the single final VALIDATION evaluation of one final checkpoint (prereg section 8).
Requires frozen.json (validation is locked before it) and that the checkpoint hash equals the frozen hash. Strength 0,
no forecasts, no domain labels in the forward pass (labels only group the reported CE). The evaluation runs through
the plain CSRModel export (to_plain_csr), after asserting identical logits to the DFC model at strength 0 on one
batch. Writes artifacts/domain-forecast-curriculum-v1/evals/val_<run_name>.json exactly once (refuses to overwrite).
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
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    if not guard.require_execution(a.execute, "gpu" if a.device == "cuda" else "cpu-model"):
        return
    import numpy as np
    import torch
    import torch.nn.functional as F
    from csr.dfc_eval import routing_metrics
    from csr.dfc_lib import ART, FROZEN, N_DOM, eval_rows, load_split, load_stats, make_batch, to_torch
    from csr.dfc_model import build_dfc_model, to_plain_csr
    from csr.model import RoutingConfig
    torch.set_num_threads(4)
    frozen = json.loads(FROZEN.read_text())
    fr = frozen["runs"][a.run_name]
    out_dir = ART / "evals"
    out_dir.mkdir(exist_ok=True)
    outp = out_dir / f"val_{a.run_name}.json"
    if outp.exists():
        raise guard.GuardError(f"{outp} exists: each final checkpoint is evaluated exactly once")
    monitor = None
    if a.device == "cuda":
        guard.occupancy_gate("dfc val eval")
        device = guard.setup_cuda_device()
        monitor = guard.OccupancyMonitor(interval=15.0)
        monitor.start()
    else:
        device = torch.device("cpu")
    guard.register_self(f"dfc val eval {a.run_name}")
    t_iso, t0 = guard.now(), time.time()
    status, reason = "FAILED", ""
    limits = guard.RunLimits(max_seconds=5100, monitor=monitor)
    try:
        ckp = guard.ROOT / fr["final_checkpoint"]
        if guard.sha256_file(ckp) != fr["final_checkpoint_sha256"]:
            raise guard.GuardError("final checkpoint hash differs from frozen.json")
        ck = torch.load(ckp, map_location="cpu", weights_only=False)
        cfg = ck["config"]
        dfc = build_dfc_model(cfg["model"], cfg["condition"], cfg["seed"], device)
        dfc.load_state_dict(ck["model"])
        dfc.eval()
        plain = to_plain_csr(dfc).eval()
        vsp, vdoms, _ = load_split("val")
        from csr.dfc_lib import DATA
        vpapers = json.loads((DATA / "val_docs.json").read_text())["papers"]
        rows, _ = eval_rows(vsp, vdoms)
        E, Lm = plain.cfg.n_experts, len(plain.cfg.moe_layers)
        C = np.zeros((Lm, N_DOM, E))
        doc_nll, doc_n = {}, {}
        dom_nll, dom_n = np.zeros(N_DOM), np.zeros(N_DOM)
        checked = False
        with torch.no_grad():
            for s in range(0, len(rows), 32):
                b = make_batch(rows[s:s + 32], vsp, vdoms, None)
                dm = torch.as_tensor(b.pop("domain"), device=device)  # grouping only; never given to the model
                tb = to_torch(b, device)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    out = plain(tb, RoutingConfig())
                    if not checked:
                        ref = dfc(tb, strength=0.0)
                        same = bool(torch.equal(ref.logits, out.logits))
                        checked = True
                        if not same:
                            raise guard.GuardError("plain CSRModel export differs from DFC model at strength 0")
                tgt = tb["targets"]
                valid = tgt != -100
                nll = F.cross_entropy(out.logits.float().view(-1, out.logits.shape[-1]), tgt.view(-1),
                                      ignore_index=-100, reduction="none").view(tgt.shape)
                if not torch.isfinite(nll[valid]).all():
                    raise FloatingPointError("non-finite validation NLL")
                for d in range(N_DOM):
                    m = valid & (dm == d)
                    dom_nll[d] += float(nll[m].double().sum())
                    dom_n[d] += int(m.sum())
                g = tb["gdoc"][valid].cpu().numpy()
                v = nll[valid].double().cpu().numpy()
                ug, inv = np.unique(g, return_inverse=True)
                for dd, sv, cv in zip(ug, np.bincount(inv, weights=v), np.bincount(inv)):
                    doc_nll[int(dd)] = doc_nll.get(int(dd), 0.0) + float(sv)
                    doc_n[int(dd)] = doc_n.get(int(dd), 0) + int(cv)
                nonpad = tb["doc_ids"] >= 0
                for li, r in enumerate(out.routing):
                    key = dm[nonpad].unsqueeze(1) * E + r["topi"][nonpad]
                    C[li] += torch.bincount(key.reshape(-1), minlength=N_DOM * E).view(N_DOM, E).double().cpu().numpy()
                limits.check()
        met = routing_metrics(C)
        keys = sorted(doc_nll)
        doc = {"run_name": a.run_name, "condition": cfg["condition"], "seed": cfg["seed"],
               "final_checkpoint": fr["final_checkpoint"], "final_checkpoint_sha256": fr["final_checkpoint_sha256"],
               "val_ce": float(dom_nll.sum() / dom_n.sum()), "val_ce_domain": (dom_nll / dom_n).tolist(),
               "val_tokens": int(dom_n.sum()), "val_tokens_domain": dom_n.tolist(), "n_docs": len(keys),
               "doc_index": keys, "doc_nll": [doc_nll[k] for k in keys], "doc_n": [doc_n[k] for k in keys],
               "doc_domain": [int(vdoms[k]) for k in keys], "doc_paper": [vpapers[k] for k in keys],
               "counts": C.tolist(),
               "mi": met["mi"], "mi_norm": met["mi_norm"], "load": met["load"],
               "load_overall": [load_stats(C[li].sum(0)) for li in range(Lm)],
               "plain_export_logits_equal": True, "used_forecasts": False, "used_domain_labels_in_model": False,
               "frozen_sha256": guard.sha256_file(FROZEN), "seconds": time.time() - t0, "created": guard.now()}
        guard.atomic_write_json(outp, doc, mode=0o400)
        guard.ledger("DFC_VAL_EVAL", f"{outp.relative_to(guard.ROOT)} ce={doc['val_ce']:.5f}")
        print(f"VAL {a.run_name} ce {doc['val_ce']:.5f} domains {np.round(doc['val_ce_domain'], 4).tolist()}")
        status = "COMPLETE"
    except guard.GuardError as e:
        status, reason = ("CONTAMINATED" if "CONTAMINATED" in str(e) else "ABORTED"), str(e)
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        status, reason = "FAILED", f"{type(e).__name__}: {e}"
    finally:
        limits.close()
        if monitor is not None:
            monitor.stop()
            if monitor.contaminated and status == "COMPLETE":
                status = "CONTAMINATED"
        if a.device == "cuda":
            guard.gpu_ledger(t_iso, guard.now(), f"dfc_val_{a.run_name}", "dfc val eval", status)
        print("STATUS", status, reason)
    sys.exit(0 if status == "COMPLETE" else 1)


if __name__ == "__main__":
    main()
