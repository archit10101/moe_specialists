#!/usr/bin/env python
"""csr-multiscale-routing-v1 secondary GPU analysis (prereg §12a, §12c), one guarded process over all main checkpoints.

(a) Router diagnostics per checkpoint on the first 256 packed calib rows (own boundaries), fp32, TF32 off, per MoE
    layer: mean and centered RMS of z_token / context contribution / z_route, softmax entropy, selected-gate margins
    (p4 - p5, p1 - p2), top-4 gate sum, fraction of top-4 sets differing from the same model's z_token top-4, expert
    load (max share, unused), and the linear-equivalence error max|W mean(h) - mean(W h)|.
(c) Latency for seed-0 checkpoints: prefill batch 1 and batch 8 x 1024 (bf16 autocast) and autoregressive ms_step
    decoding of 256 tokens at batch 1 (fp32); 3 warm-up + 10 measured repetitions each, identical inputs.
Writes artifacts/multiscale-routing-v1/diag_<UTC>.json. Read-only on checkpoints (sha256 verified before/after).
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402

ART = guard.ROOT / "artifacts" / "multiscale-routing-v1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="main_runs.tsv (key<TAB>run_dir)")
    ap.add_argument("--rows", type=int, default=256)
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    out_p = ART / f"diag_{guard.utc_stamp()}.json"
    print("msv1 router diagnostics + latency")
    if not guard.require_execution(a.execute, "gpu"):
        return
    guard.occupancy_gate("msv1 diag")
    dev = guard.setup_cuda_device()
    mon = guard.OccupancyMonitor(10.0)
    mon.start()
    guard.register_self("msv1 diag")
    t0iso, t0 = guard.now(), time.time()
    lim = guard.RunLimits(guard.MAX_ATOMIC_SECONDS, monitor=mon)
    status, reason, res = "INCOMPLETE", "", {}
    try:
        res = run(a, dev, lim)
        status = "COMPLETE"
    except guard.GuardError as e:
        status, reason = ("CONTAMINATED" if "CONTAMINATED" in str(e) else "ABORTED"), str(e)
    except Exception as e:  # noqa: BLE001
        status, reason = "FAILED", f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        lim.close()
        mon.stop()
        if mon.contaminated and status == "COMPLETE":
            status, reason = "CONTAMINATED", mon.detail
        res.update(status=status, reason=reason, seconds=time.time() - t0, created=guard.now())
        guard.atomic_write_json(out_p, res, mode=0o400)
        guard.gpu_ledger(t0iso, guard.now(), "msv1_diag", "msv1 diag", status)
        guard.ledger("MS_V1_DIAG", f"{out_p.name} status={status} {reason}")
        print(status, reason, f"RESULT {out_p.relative_to(guard.ROOT)}")
    sys.exit(0 if status == "COMPLETE" else 1)


def run(a, dev, lim):
    import numpy as np
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from csr.data import to_torch
    from csr.ms_run import build_ms_model, check_cfg, make_sources
    from csr.multiscale import MSRouting, center, context_means, rms
    runs = [ln.split("\t")[:2] for ln in Path(a.runs).read_text().splitlines() if ln.strip()]
    out = {"per_checkpoint": {}, "latency": {}, "rows": a.rows}
    for key, rd in runs:
        rd = guard.ROOT / rd
        sha = json.loads((rd / "summary.json").read_text())["checkpoint_sha256"]
        if guard.sha256_file(rd / "model.pt") != sha:
            raise guard.GuardError(f"checkpoint sha mismatch {rd}")
        sd = torch.load(rd / "model.pt", map_location="cpu", weights_only=True)
        cfg = sd["config"]
        d = check_cfg(cfg)
        m = build_ms_model(cfg, dev)
        m.load_state_dict(sd["model"], strict=True)
        m.eval()
        r = MSRouting.from_cfg(cfg["routing"])
        _, src = make_sources(cfg, d, "calib")
        n_moe, E = 7, 32
        acc = {k: np.zeros(n_moe) for k in ("zt_mean", "zt_crms", "ctx_crms", "zr_mean", "zr_crms", "entropy",
                                             "margin45", "margin12", "top4sum", "differs", "scale_s")}
        load = np.zeros((n_moe, E), np.int64)
        lin = np.zeros(n_moe)
        ntok = 0
        with torch.no_grad():
            for s in range(0, a.rows, 16):
                tb = to_torch(src.batch(src.rows[s:s + 16]), dev)
                o = m(tb, r, collect=True)
                nonpad = tb["doc_ids"] >= 0
                nt = int(nonpad.sum())
                ntok += nt
                for li, q in enumerate(o.routing):
                    zt, zr = q["z_token"][nonpad], q["z_route"][nonpad]
                    p = q["probs_full"][nonpad]
                    ps = p.sort(-1, descending=True).values
                    acc["zt_mean"][li] += float(zt.mean(-1).sum())
                    acc["zt_crms"][li] += float(rms(center(zt)).sum())
                    acc["zr_mean"][li] += float(zr.mean(-1).sum())
                    acc["zr_crms"][li] += float(rms(center(zr)).sum())
                    acc["scale_s"][li] += float((rms(center(zr)) / rms(center(zt)).clamp(min=1e-6)).sum())
                    if q["ctx"] is not None:
                        acc["ctx_crms"][li] += float(rms(center(q["ctx"][nonpad])).sum())
                    acc["entropy"][li] += float((-(p * p.clamp_min(1e-30).log()).sum(-1)).sum())
                    acc["margin45"][li] += float((ps[:, 3] - ps[:, 4]).sum())
                    acc["margin12"][li] += float((ps[:, 0] - ps[:, 1]).sum())
                    acc["top4sum"][li] += float(ps[:, :4].sum())
                    tk = zt.topk(4, -1).indices.sort(-1).values
                    acc["differs"][li] += float((q["topi"][nonpad].sort(-1).values != tk).any(-1).sum())
                    load[li] += np.bincount(q["topi"][nonpad].reshape(-1).cpu().numpy(), minlength=E)
                    if q["ctx"] is not None:
                        W = m.moe_blocks()[li].ffn.router.weight
                        nl, nb = r.uses_local(q["kind"]), r.uses_broad(q["kind"])
                        ml, _, mb, _ = context_means(q["h"], tb["positions"], tb["chunk_pos"], nonpad, nl, nb)
                        zl, _, zb, _ = context_means(q["z_token"], tb["positions"], tb["chunk_pos"], nonpad, nl, nb)
                        for mm, zz in ((ml, zl), (mb, zb)):
                            if mm is not None:
                                lin[li] = max(lin[li], float((mm @ W.T - zz).abs().max()))
                del o
                lim.check()
        frac = load / load.sum(1, keepdims=True)
        rec = {k: (v / ntok).tolist() for k, v in acc.items()}
        rec.update({"n_tokens": ntok, "expert_max_share": frac.max(1).tolist(), "experts_unused": (load == 0).sum(1).tolist(),
                    "linear_equivalence_max_abs_err": lin.tolist(), "schedule": list(r.schedule),
                    "scale_control": r.scale_control, "alpha": r.alpha, "beta": r.beta})
        out["per_checkpoint"][key] = rec
        if key.startswith("seed0/"):
            out["latency"][key] = latency(m, r, src, dev, lim)
        if guard.sha256_file(rd / "model.pt") != sha:
            raise guard.GuardError("checkpoint changed")
        del m
        torch.cuda.empty_cache()
        print(key, "ok", flush=True)
    return out


def latency(m, r, src, dev, lim, warm=3, reps=10):
    import numpy as np
    import torch
    from csr.data import to_torch
    res = {}
    with torch.no_grad():
        for bsz in (1, 8):
            tb = to_torch(src.batch(src.rows[:bsz]), dev)
            ts = []
            for i in range(warm + reps):
                torch.cuda.synchronize()
                t = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    m(tb, r)
                torch.cuda.synchronize()
                if i >= warm:
                    ts.append(time.perf_counter() - t)
            ntok = int((tb["doc_ids"] >= 0).sum())
            res[f"prefill_b{bsz}"] = {"median_s": float(np.median(ts)), "min_s": float(np.min(ts)), "tokens": ntok,
                                      "tokens_per_s": ntok / float(np.median(ts))}
        doc = next(i for i in range(src.split.n_docs) if len(src.split.doc(i)) >= 256)
        ids = src.split.doc(doc)[:256]
        starts = src.seg.starts(doc) if src.seg is not None else [0]
        from csr.segment import starts_to_arrays
        _, cpos = starts_to_arrays(len(src.split.doc(doc)), starts)
        ts = []
        for i in range(warm + reps):
            st = m.new_ms_state()
            torch.cuda.synchronize()
            t = time.perf_counter()
            for p in range(256):
                m.ms_step(int(ids[p]), p, int(cpos[p]), r, st)
            torch.cuda.synchronize()
            if i >= warm:
                ts.append(time.perf_counter() - t)
            lim.check()
        res["decode_b1_256"] = {"median_s": float(np.median(ts)), "per_token_ms": 1000 * float(np.median(ts)) / 256}
    return res


if __name__ == "__main__":
    main()
