#!/usr/bin/env python
"""csr-speculative-pool-routing-v1: measured wall-clock benchmarks (prereg §11). Dry-run by default.

--mode feasibility : before final training, on Generation-3 checkpoints (seed 97) and the frozen draft: training
                     forward+backward step, prefill, greedy and speculative decoding (batch 1) -> budget gate.
--mode final       : final seed-0 checkpoints: training step, prefill (batch 1 / 16, bf16 autocast, including on-the-fly
                     draft proposals for the spec family), greedy decoding at batch 1 / 4 / 16 (fp32) for
                     TOKEN-4, the causal finalist and its random control; speculative greedy decoding for TOKEN-4 +
                     speculation (same draft, same gamma) and the spec finalist and its random control; acceptance
                     statistics; exact greedy-equivalence checks (fp32 and fp64).
All timings: CUDA-synchronised wall clock, median of repeats, prompts of 64 tokens from CALIB-SEARCH documents,
128 generated tokens; draft, detector, selection, verification, fallback and proposal regeneration included.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402

P_LEN, G_LEN = 64, 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["feasibility", "final"], required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="label=run_dir")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    if not guard.require_execution(a.execute, "gpu" if a.device == "cuda" else "cpu-model"):
        return
    import torch
    torch.set_num_threads(4)
    monitor = None
    if a.device == "cuda":
        guard.occupancy_gate(f"spv1 bench {a.mode}")
        device = guard.setup_cuda_device()
        monitor = guard.OccupancyMonitor(interval=15.0)
        monitor.start()
    else:
        device = torch.device("cpu")
    guard.register_self(f"spv1 bench {a.mode}")
    t_iso, t0 = guard.now(), time.time()
    status, reason, res = "FAILED", "", {}
    try:
        res = run(a, device)
        status = "COMPLETE"
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        reason = f"{type(e).__name__}: {e}"
    if monitor is not None:
        monitor.stop()
        if monitor.contaminated:
            status, reason = "CONTAMINATED", monitor.detail
    from csr.spv1_run import ART
    out = ART / f"bench_{a.mode}_{guard.utc_stamp()}.json"
    guard.atomic_write_json(out, {"status": status, "reason": reason, "mode": a.mode, "results": res,
                                  "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
                                  "prompt_len": P_LEN, "gen_len": G_LEN, "seconds": time.time() - t0})
    if a.device == "cuda":
        guard.gpu_ledger(t_iso, guard.now(), out.stem, f"spv1 bench {a.mode}", status)
    print(f"BENCH_JSON {out.relative_to(guard.ROOT)} {status} {reason}")
    sys.exit(0 if status == "COMPLETE" else 1)


def run(a, device):
    import numpy as np
    import torch
    from csr.data import to_torch
    from csr.spv1_decode import greedy_generate, sequential_spec_family, spec_generate
    from csr.spv1_model import build_pool_model, load_frozen_draft
    from csr.spv1_run import DATA_DIR, eval_source, frozen_draft

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    def timeit(fn, reps, warm=2):
        for _ in range(warm):
            fn()
        sync()
        ts = []
        for _ in range(reps):
            s = time.perf_counter()
            fn()
            sync()
            ts.append(time.perf_counter() - s)
        return statistics.median(ts)

    dpath, drec = frozen_draft()
    draft32 = load_frozen_draft(dpath, device)
    sp, src = eval_source({"data": {"seq_len": 1024}}, "calib_search", guard.ROOT / DATA_DIR)
    docs = [d for r in src.rows for d in r if sp.lengths()[d] >= P_LEN + G_LEN][:16]
    assert len(docs) == 16
    prompts16 = torch.stack([torch.as_tensor(sp.doc(d)[:P_LEN]) for d in docs]).to(device)
    labels = dict(x.split("=", 1) for x in a.runs)
    out = {"draft": drec, "conditions": {}}
    models = {}
    for lab, rdp in labels.items():
        rd = guard.inside_root(rdp)
        cfg = json.loads((rd / "config.json").read_text())
        m = build_pool_model(cfg, device)
        m.load_state_dict(torch.load(rd / "model.pt", map_location="cpu", weights_only=False)["model"])
        m.eval()
        if m.pr.family == "spec":
            m.attach_draft(draft32)
        models[lab] = (m, cfg, rd)
    batches = (1,) if a.mode == "feasibility" else (1, 4, 16)
    gamma_spec = None
    for lab, (m, cfg, rd) in models.items():
        if m.pr.family == "spec":
            gamma_spec = m.pr.gamma
    gammas = sorted({gamma_spec or 3, 3 if a.mode == "final" else gamma_spec or 3})
    for lab, (m, cfg, rd) in models.items():
        r = {"run": str(rd.relative_to(guard.ROOT)), "candidate": cfg["candidate"], "family": cfg["family"],
             "role": cfg.get("role")}
        fam = m.pr.family
        # ---- training step (forward + backward, bf16 autocast, micro-batch 16 x 1024)
        tb = to_torch(src.batch(src.rows[:16]), device)
        ntok = int((tb["targets"] != -100).sum())
        m.train()
        if fam == "spec":
            m.draft_autocast = True

        def step():
            with torch.autocast(device.type, dtype=torch.bfloat16):
                o = m(tb)
            lm = torch.nn.functional.cross_entropy(o.logits.float().view(-1, o.logits.shape[-1]),
                                                   tb["targets"].view(-1), ignore_index=-100)
            (lm + 0.01 * o.aux_loss + m.objective_extras(o)).backward()
            m.zero_grad(set_to_none=True)
        if fam == "adaptive":
            pass  # tau is loaded from the checkpoint; training-mode statistics are not used here
        m.rate_stats_reset()
        st = timeit(step, 5)
        r["train_step_s"] = st
        r["train_step_tokens_per_s"] = ntok / st
        m.eval()
        # ---- prefill (teacher-forced forward), bf16 autocast, incl. on-the-fly proposals
        with torch.no_grad():
            def pf16():
                with torch.autocast(device.type, dtype=torch.bfloat16):
                    m(tb)
            tb1 = to_torch(src.batch([src.rows[0]]), device)

            def pf1():
                with torch.autocast(device.type, dtype=torch.bfloat16):
                    m(tb1)
            r["prefill_b16_tokens_per_s"] = int((tb["doc_ids"] >= 0).sum()) / timeit(pf16, 5)
            r["prefill_b1_tokens_per_s"] = int((tb1["doc_ids"] >= 0).sum()) / timeit(pf1, 5)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        # ---- decoding (fp32)
        m.draft_autocast = False
        dec = {}
        for Bt in batches:
            pr_ = prompts16[:Bt]
            if fam in ("token", "fixed", "adaptive"):
                tg = timeit(lambda: greedy_generate(m, pr_, G_LEN), 3 if Bt > 1 else 3, warm=1)
                _, gst = greedy_generate(m, pr_, G_LEN)
                dec[f"greedy_b{Bt}"] = {"seconds": tg, "tokens_per_s": Bt * G_LEN / tg, **gst}
            if fam in ("token", "spec"):
                for gm in ([m.pr.gamma] if fam == "spec" else gammas):
                    ts_ = timeit(lambda: spec_generate(m, draft32, pr_, G_LEN, gm), 3, warm=1)
                    _, sst = spec_generate(m, draft32, pr_, G_LEN, gm)
                    dec[f"spec_g{gm}_b{Bt}"] = {"seconds": ts_, "tokens_per_s": Bt * G_LEN / ts_,
                                                **{k: v for k, v in sst.items() if k != "per_seq_blocks"}}
        r["decode"] = dec
        r["decode_peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 1024**3 if device.type == "cuda" else 0
        # ---- exact greedy equivalence (final mode)
        if a.mode == "final" and fam in ("token", "spec"):
            eq = {}
            for dt in ("fp32", "fp64"):
                dtype = torch.float32 if dt == "fp32" else torch.float64
                mm = m.to(dtype)
                dd = draft32.to(dtype)
                if fam == "spec":
                    mm.attach_draft(dd)
                n = 16 if dt == "fp32" else 8
                pr_ = prompts16[:n]
                for gm in ([m.pr.gamma] if fam == "spec" else gammas):
                    with torch.no_grad():
                        spo, _ = spec_generate(mm, dd, pr_, G_LEN, gm)
                        if fam == "spec":
                            ref = torch.cat([sequential_spec_family(mm, dd, pr_[i:i + 1], G_LEN)[0]
                                             for i in range(n)], 0)
                        else:
                            ref = torch.cat([greedy_generate(mm, pr_[i:i + 1], G_LEN)[0] for i in range(n)], 0)
                    same = (spo == ref).all(1)
                    firstdiff = [int((spo[i] != ref[i]).nonzero()[0]) if not bool(same[i]) else None
                                 for i in range(n)]
                    eq[f"{dt}_g{gm}"] = {"sequences": n, "identical_sequences": int(same.sum()),
                                         "all_identical": bool(same.all()), "first_divergence": firstdiff}
            m.to(torch.float32)
            draft32.to(torch.float32)
            if fam == "spec":
                m.attach_draft(draft32)
            r["greedy_equivalence"] = eq
        if a.mode == "final" and fam in ("fixed", "adaptive"):
            with torch.no_grad():
                a_ = greedy_generate(m, prompts16, G_LEN)[0]
                b_ = torch.cat([greedy_generate(m, prompts16[i:i + 1], G_LEN)[0] for i in range(16)], 0)
            r["greedy_equivalence"] = {"lockstep_b16_vs_single": bool(torch.equal(a_, b_))}
        out["conditions"][lab] = r
        print(lab, json.dumps({k: v for k, v in r.items() if k != "decode"}), flush=True)
    # ---- draft alone
    tb = to_torch(src.batch(src.rows[:16]), device)
    from csr.model import RoutingConfig
    with torch.no_grad():
        def dpf():
            with torch.autocast(device.type, dtype=torch.bfloat16):
                draft32(tb, RoutingConfig())
        out["draft_prefill_b16_tokens_per_s"] = int((tb["doc_ids"] >= 0).sum()) / timeit(dpf, 5)
    return out


if __name__ == "__main__":
    main()
