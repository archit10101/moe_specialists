#!/usr/bin/env python
"""csr-speculative-pool-routing-v1: GPU correctness checks (prereg §7). Dry-run by default.

--phase target (before any training): every Generation-1 configuration and TOKEN-4 at initialisation (seed 98), full
  size, on real CALIB-SEARCH documents, fp32 (TF32 off) unless stated; spec family with a random-init draft.
  C1 future-token perturbation: logits, pools, block starts, expert ids and gates at positions <= cut unchanged.
  C2 teacher-forced full sequence == incremental (KV-cached) execution.
  C3 batched lockstep speculative greedy decoding == sequential greedy execution (spec family and TOKEN-4).
  C4 structure kernels (block starts, pointer doubling, pools, in-pool top-k) run with CUDA sync debug = error.
  C5 one bf16 training step per config (micro-batch 16 x 1024): finite loss / gradients, peak VRAM < 56 GiB.
--phase draft (after draft training, before freezing): the trained draft.
  D1 full-sequence logits == incremental logits; finite.  D2 tree proposals == repeated greedy generation.
  D3 draft is smaller and faster than the target (parameters; measured batch-1 decode step latency).
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
    ap.add_argument("--phase", choices=["target", "draft"], required=True)
    ap.add_argument("--draft-run", help="draft run dir (phase draft)")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    if not guard.require_execution(a.execute, "gpu" if a.device == "cuda" else "cpu-model"):
        return
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(4)
    monitor = None
    if a.device == "cuda":
        guard.occupancy_gate(f"spv1 gpu checks {a.phase}")
        device = guard.setup_cuda_device()
        monitor = guard.OccupancyMonitor(interval=15.0)
        monitor.start()
    else:
        device = torch.device("cpu")
    guard.register_self(f"spv1 gpu checks {a.phase}")
    t_iso, t0 = guard.now(), time.time()
    res = {}
    status, reason = "FAIL", ""
    try:
        res = phase_target(device) if a.phase == "target" else phase_draft(device, a.draft_run)
        ok = all(v.get("pass") for v in res.values() if isinstance(v, dict) and "pass" in v)
        status = "PASS" if ok else "FAIL"
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        status, reason = "FAIL", f"{type(e).__name__}: {e}"
    if monitor is not None:
        monitor.stop()
        if monitor.contaminated:
            status, reason = "CONTAMINATED", monitor.detail
    from csr.spv1_run import ART
    out = ART / f"gpu_checks_{a.phase}_{guard.utc_stamp()}.json"
    guard.atomic_write_json(out, {"status": status, "reason": reason, "phase": a.phase, "results": res,
                                  "device": str(device), "seconds": time.time() - t0, "time": guard.now(),
                                  "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"})
    if a.device == "cuda":
        guard.gpu_ledger(t_iso, guard.now(), out.stem, f"spv1 gpu checks {a.phase}", status)
    print(f"GPU_CHECKS_JSON {out.relative_to(guard.ROOT)} {status} {reason}")
    sys.exit(0 if status == "PASS" else 1)


def _docs(n, min_len, evalset="calib_search"):
    from csr.spv1_run import DATA_DIR, eval_source
    sp, src = eval_source({"data": {"seq_len": 1024}}, evalset, guard.ROOT / DATA_DIR)
    out = []
    for r in src.rows:
        for d in r:
            if sp.lengths()[d] >= min_len:
                out.append(d)
            if len(out) >= n:
                return sp, src, out
    return sp, src, out


def _cfg(spec, seed=98):
    import spv1_make_configs as mk
    return mk.make(seed, 5_000_000, "gen1", False, spec)


def phase_target(device):
    import numpy as np
    import torch
    import spv1_make_configs as mk
    from csr.data import to_torch
    from csr.model import CSRModel
    from csr.spv1_decode import (draft_greedy_full, greedy_generate, new_cache, sequential_spec_family,
                                 spec_generate, target_inc)
    from csr.spv1_model import (block_start_index, build_pool_model, draft_model_config, spec_starts)
    from csr.spv1_select import GEN1
    res = {}
    sp, src, docs = _docs(4, 400)
    rows = [src.rows[0], src.rows[1]]
    tb = to_torch(src.batch(rows), device)
    torch.manual_seed(0)
    draft = CSRModel(draft_model_config()).to(device).eval()
    specs = [mk.TOKEN_SPEC] + GEN1
    for spec in specs:
        cid = spec["id"]
        cfg = _cfg(spec)
        m = build_pool_model(cfg, device)
        g = torch.Generator().manual_seed(5)
        with torch.no_grad():  # non-trivial selector / detector (zero at init) so the checks are not vacuous
            for n, p in m.named_parameters():
                if n in m.EXTRA_NAMES and n != "spec_beta":
                    p.copy_((torch.randn(p.shape, generator=g) * 0.05).to(device))
        if m.pr.family == "spec":
            m.attach_draft(draft)
            m.draft_autocast = False
        if m.pr.family == "adaptive":
            m.train()
            m.tau_calibrate(tb)
        m.eval()
        r = {}
        # C1 future perturbation
        with torch.no_grad():
            base = m(tb, collect=True)
            worst, route_ok = 0.0, True
            rng = np.random.default_rng(1)
            for cut in (0, 5, 63, 200, 511, 700, 1000):
                b2 = {k: v.clone() for k, v in tb.items()}
                npd = b2["doc_ids"] >= 0
                fut = torch.zeros_like(npd)
                fut[:, cut + 1:] = True
                noise = torch.as_tensor(rng.integers(2, 50000, size=tuple(npd.shape)), device=device)
                b2["input_ids"] = torch.where(fut & npd, noise, b2["input_ids"])
                o2 = m(b2, collect=True)
                worst = max(worst, float((base.logits[:, :cut + 1] - o2.logits[:, :cut + 1]).abs().max()))
                for r1, r2 in zip(base.routing, o2.routing):
                    route_ok &= bool(torch.equal(r1["topi"][:, :cut + 1], r2["topi"][:, :cut + 1]))
                    if "cand" in r1:
                        route_ok &= bool(torch.equal(r1["cand"][:, :cut + 1], r2["cand"][:, :cut + 1]))
                        route_ok &= bool(torch.equal(r1["starts"][:, :cut + 1], r2["starts"][:, :cut + 1]))
        r["C1_future_perturbation"] = {"max_abs_logit_diff": worst, "routes_pools_starts_identical": route_ok,
                                       "pass": route_ok and worst < 1e-3}
        # C2 incremental == teacher forcing (two single-document sequences of 300 tokens)
        T = 300
        ids = torch.stack([torch.as_tensor(sp.doc(d)[:T]) for d in docs[:2]]).to(device)
        Bt = 2
        b = {"input_ids": ids, "doc_ids": torch.zeros_like(ids),
             "positions": torch.arange(T, device=device).expand(Bt, T).contiguous()}
        with torch.no_grad():
            full = m(b).logits
            cache = new_cache(m, Bt, T + 2, device, torch.float32)
            if m.pr.family == "adaptive":
                rstate = [{"pool": torch.zeros(Bt, 32, dtype=torch.bool, device=device),
                           "hold": torch.zeros(Bt, dtype=torch.long, device=device)} for _ in range(m.n_moe)]
            else:
                rstate = [{} for _ in range(getattr(m, "n_moe", 7))]
            D, j, outs = [None] * Bt, [0] * Bt, []
            for t in range(T):
                bs = props = None
                if m.pr.family == "spec":
                    bs = torch.zeros(Bt, 1, dtype=torch.bool, device=device)
                    props = torch.zeros(Bt, 1, m.pr.gamma, dtype=torch.long, device=device)
                    for bb in range(Bt):
                        nb = D[bb] is None or j[bb] >= m.pr.seg_len or int(ids[bb, t]) != int(D[bb][0, j[bb] - 1])
                        if nb:
                            D[bb] = draft_greedy_full(draft, ids[bb:bb + 1, :t + 1], m.pr.gamma)
                            j[bb] = 0
                        bs[bb, 0] = nb
                        props[bb, 0] = D[bb][0]
                        j[bb] += 1
                lg, _ = target_inc(m, ids[:, t:t + 1], torch.full((Bt, 1), t, device=device), cache, rstate,
                                   blk_start=bs, props=props)
                outs.append(lg)
            inc = torch.cat(outs, 1)
            d_inc = float((inc - full).abs().max())
            agree = float((inc.argmax(-1) == full.argmax(-1)).float().mean())
        r["C2_incremental_vs_full"] = {"max_abs_logit_diff": d_inc, "argmax_agreement": agree,
                                       "pass": d_inc < 2e-3 and agree > 0.999}
        # C3 speculative == sequential greedy (spec family and TOKEN-4 with gamma 3)
        if m.pr.family in ("spec", "token"):
            gamma = m.pr.gamma if m.pr.family == "spec" else 3
            prompts = torch.stack([torch.as_tensor(sp.doc(d)[:48]) for d in docs[:4]]).to(device)
            with torch.no_grad():
                a_, st = spec_generate(m, draft, prompts, 48, gamma)
                if m.pr.family == "spec":
                    ref = torch.cat([sequential_spec_family(m, draft, prompts[i:i + 1], 48)[0] for i in range(4)], 0)
                else:
                    ref = greedy_generate(m, prompts, 48)[0]
            r["C3_speculative_vs_sequential"] = {"identical": bool(torch.equal(a_, ref)),
                                                 "accept_hist": st["accept_hist"], "pass": bool(torch.equal(a_, ref))}
        # C4 sync-free structure kernels
        if m.pr.family != "token":
            with torch.no_grad():
                rr = base.routing[0]
                props = m.prepare_proposals(tb) if m.pr.family == "spec" else None
                hs = torch.randn(*tb["input_ids"].shape, 192, device=device)
                torch.cuda.synchronize()
                torch.cuda.set_sync_debug_mode("error")
                try:
                    if m.pr.family == "spec":
                        st_ = spec_starts(tb["input_ids"], tb["positions"], props, m.pr.seg_len)
                    elif m.pr.family == "adaptive":
                        st_ = m._adaptive_starts_given_tau(rr["probs"], rr["cand"].float(), torch.zeros_like(
                            rr["probs"][..., 0]), tb["positions"] == 0, m.pr.seg_len, m.tau[0].float())
                    else:
                        st_ = (tb["positions"] % m.pr.seg_len) == 0
                    bsi = block_start_index(st_)
                    ghat = m._ghat(0, m._draft_features(props)) if (m.pr.family == "spec" and
                                                                    m.pr.pool_assign == "learned") else None
                    P_all, _ = m._pools(0, m.blocks[1].ffn, hs, tb["positions"], ghat)
                    cand = P_all.gather(1, bsi.unsqueeze(-1).expand(*bsi.shape, 32))
                    rr["probs"].masked_fill(~cand, -1.0).topk(4, -1)
                    sync_ok = True
                except RuntimeError as e:
                    sync_ok = False
                    r["C4_error"] = str(e)[:300]
                finally:
                    torch.cuda.set_sync_debug_mode("default")
            r["C4_no_host_sync_structure"] = {"pass": sync_ok}
        # C5 one bf16 training step at full micro-batch size
        m.train()
        if m.pr.family == "spec":
            m.draft_autocast = True
        tbf = to_torch(src.batch(src.rows[:16]), device)
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = m(tbf)
        tgt = tbf["targets"]
        lm = torch.nn.functional.cross_entropy(out.logits.float().view(-1, out.logits.shape[-1]), tgt.view(-1),
                                               ignore_index=-100)
        obj = lm + 0.01 * out.aux_loss + 0.001 * out.z_loss + m.objective_extras(out)
        obj.backward()
        fin = bool(torch.isfinite(obj)) and all(p.grad is None or bool(torch.isfinite(p.grad).all())
                                                for p in m.parameters())
        peak = torch.cuda.max_memory_reserved() / 1024**3
        r["C5_train_step"] = {"loss": float(lm), "finite": fin, "peak_reserved_gib": peak,
                              "pass": fin and peak < 56.0}
        res[cid] = r
        print(cid, json.dumps({k: v.get("pass") for k, v in r.items() if isinstance(v, dict)}), flush=True)
        del m
        torch.cuda.empty_cache()
    return res


def phase_draft(device, draft_run):
    import numpy as np
    import torch
    from csr.data import to_torch
    from csr.model import RoutingConfig, count_parameters
    from csr.spv1_decode import draft_greedy_full, draft_inc, new_cache
    from csr.spv1_model import draft_tree_proposals, load_frozen_draft
    rd = guard.inside_root(draft_run)
    m = load_frozen_draft(rd / "model.pt", device)
    sp, src, docs = _docs(4, 300)
    res = {}
    T = 300
    ids = torch.stack([torch.as_tensor(sp.doc(d)[:T]) for d in docs[:3]]).to(device)
    b = {"input_ids": ids, "doc_ids": torch.zeros_like(ids),
         "positions": torch.arange(T, device=device).expand(3, T).contiguous()}
    with torch.no_grad():
        full = m(b, RoutingConfig()).logits
        cache = new_cache(m, 3, T + 2, device, torch.float32)
        inc = torch.cat([draft_inc(m, ids[:, t:t + 1], torch.full((3, 1), t, device=device), cache)
                         for t in range(T)], 1)
    d = float((full - inc).abs().max())
    res["D1_full_vs_incremental"] = {"max_abs_logit_diff": d, "finite": bool(torch.isfinite(full).all()),
                                     "argmax_agreement": float((full.argmax(-1) == inc.argmax(-1)).float().mean()),
                                     "pass": d < 2e-3 and bool(torch.isfinite(full).all())}
    tb = to_torch(src.batch(src.rows[:2]), device)
    with torch.no_grad():
        P = draft_tree_proposals(m, tb["input_ids"], tb["doc_ids"], tb["positions"], 7)
        rng = np.random.default_rng(0)
        agree, n = 0, 0
        npd = (tb["doc_ids"] >= 0).cpu().numpy()
        cand = [(b_, t) for b_ in range(2) for t in range(1024) if npd[b_, t]]
        for i in rng.choice(len(cand), 60, replace=False):
            b_, t = cand[i]
            st = t - int(tb["positions"][b_, t])
            ref = draft_greedy_full(m, tb["input_ids"][b_:b_ + 1, st:t + 1], 7)
            agree += int((ref[0] == P[b_, t]).sum())
            n += 7
    res["D2_tree_vs_greedy"] = {"agree": agree, "n": n, "rate": agree / n, "pass": agree / n > 0.995}
    # D3 smaller and faster than the target
    import spv1_make_configs as mk
    from csr.spv1_model import build_pool_model
    tgt = build_pool_model(_cfg(mk.TOKEN_SPEC), device).eval()
    pd, pt = count_parameters(m)["total"], count_parameters(tgt)["total"]

    def lat(model, step):
        c = new_cache(model, 1, 300, device, torch.float32)
        x = torch.tensor([[5]], device=device)
        with torch.no_grad():
            for t in range(20):
                step(model, x, torch.tensor([[t]], device=device), c)
            torch.cuda.synchronize()
            s = time.perf_counter()
            for t in range(20, 220):
                step(model, x, torch.tensor([[t]], device=device), c)
            torch.cuda.synchronize()
        return (time.perf_counter() - s) / 200

    from csr.spv1_decode import target_inc
    ld = lat(m, draft_inc)
    lt = lat(tgt, lambda mm, x, p, c: target_inc(mm, x, p, c, []))
    res["D3_smaller_faster"] = {"draft_params": pd, "target_params": pt, "draft_step_ms": 1000 * ld,
                                "target_step_ms": 1000 * lt, "pass": pd < pt and ld < lt}
    print(json.dumps(res), flush=True)
    return res


if __name__ == "__main__":
    main()
