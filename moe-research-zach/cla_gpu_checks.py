#!/usr/bin/env python
"""csr-conditional-local-attention-v1 GPU correctness checks (full-size models, CALIB-SEARCH + train rows only).

For the baseline and every candidate (seed-98 initialisation; conditional thresholds calibrated on a training
micro-batch exactly as in training):
  G1 fp32: flex (compiled Triton) backend vs dense reference backend: logits / loss / aux diffs, expert-choice and
     gate-decision agreement (incl. compact-query outputs vs dense compact reference).
  G2 FULL-TOKEN-4 vs the ORIGINAL CSRModel (SDPA, dense mask) in fp32 and bf16: logits, losses, expert selections.
  G3 future-token perturbation (flex, fp32): earlier logits / experts / gate decisions.
  G4 incremental cached decoding vs full-sequence forward (fp32).
  G5 one bf16 training step at the real batch (2 x 16 rows), ordinary and counterfactual: finite loss / grads, gate
     gradients, no expert collapse, peak VRAM < 56 GiB.
  G6 profiling: compact global attention computes only the selected queries' tiles (tile counts vs full; kernel time
     vs forced-local / forced-global; peak-memory increment far below an N x N score matrix).
Writes artifacts/conditional-local-attention-v1/gpu_checks_<stamp>.json with status PASS / FAIL.
"""
import argparse
import copy
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402

REF_CKPT = "runs/20261106T232209Z_spv1_train_final_B100M_s0_TOKEN-4/model.pt"
TOL = {"fp32_logit": 2e-3, "agree": 0.995, "decode_logit": 2e-3, "bf16_ce": 2e-3, "bf16_mean_logit": 0.05,
       "bf16_agree": 0.98}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cuda"], default="cuda")
    ap.add_argument("--formal", action="store_true", help="post-preregistration check counted by the driver")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    if not guard.require_execution(a.execute, "gpu"):
        return
    import numpy as np
    import torch
    import torch.nn.functional as F
    from csr.cla_attn import AttnContext, compact_block_mask, compact_positions, kv_blocks_computed
    from csr.cla_decode import greedy
    from csr.cla_model import AttnPlan, build_cla_model
    from csr.cla_run import ART, DATA_DIR, eval_source, load_raw_split, collapse_check
    from csr.cla_select import BASELINE, PLANS
    from csr.data import to_torch, train_order
    from csr.model import RoutingConfig
    from csr.train_lib import BatchSource, build_model
    guard.occupancy_gate("clav1 gpu checks")
    dev = guard.setup_cuda_device()
    mon = guard.OccupancyMonitor(15.0)
    mon.start()
    guard.register_self("clav1 gpu checks")
    t_iso = guard.now()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    base = json.loads((guard.ROOT / "configs/rep-v1/seed0/condition_A.json").read_text())
    res = {"tolerances": TOL, "conditions": {}}
    fails = []

    def cfg_for(cid, seed=98):
        c = copy.deepcopy(base)
        c["init"] = {"type": "random", "seed": seed}
        c["attention"] = AttnPlan.from_cfg(PLANS[cid]).to_dict()
        return c

    def check(cond, name, ok, detail):
        if not ok:
            fails.append(f"{cond}:{name}: {detail}")
    status = "FAIL"
    try:
        d = guard.ROOT / DATA_DIR
        sp, cs = eval_source({"data": {"seq_len": 1024}}, "calib_search", d)
        tr = load_raw_split(d, "train")
        trs = BatchSource(tr, 1024, None, train_order(tr.n_docs, 17 + 98))
        evb = to_torch(cs.batch(cs.rows[:4]), dev)
        mb = [to_torch(trs.batch(trs.rows[i * 16:(i + 1) * 16]), dev) for i in range(2)]

        def ce(out, tb):
            lg = out.logits.float()
            return float(F.cross_entropy(lg.view(-1, lg.shape[-1]), tb["targets"].view(-1), ignore_index=-100))

        def agree(ra, rb, mask):
            fr = [float((x["topi"].sort(-1).values == y["topi"].sort(-1).values).all(-1)[mask].float().mean())
                  for x, y in zip(ra, rb)]
            return min(fr)

        nonpad = evb["doc_ids"] >= 0
        # trained reference weights for the numerical comparisons G1-G4 (read-only; routing at random init is
        # near-tied, so discrete agreement would only measure tie noise): csr-speculative-pool-routing-v1 final
        # TOKEN-4 seed 0 (same architecture; CSRModel state dict)
        ref_p = guard.ROOT / REF_CKPT
        ref_sd = torch.load(ref_p, map_location="cpu", weights_only=False)["model"]
        res["reference_checkpoint"] = {"path": REF_CKPT, "sha256": guard.sha256_file(ref_p)}

        def with_ref(m):
            m.load_state_dict({**m.state_dict(), **{k: v.to(dev) for k, v in ref_sd.items()}})
            return m
        # ---- G2 baseline vs original CSRModel
        c = cfg_for(BASELINE)
        orig = build_model(c, dev)
        orig.load_state_dict({k: v.to(dev) for k, v in ref_sd.items()})
        orig.eval()
        new = with_ref(build_cla_model(c, dev)).eval()
        g2 = {}
        with torch.no_grad():
            o32 = orig(evb, RoutingConfig())
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o16 = orig(evb, RoutingConfig())
        g2["original_bf16_vs_fp32_noise_floor"] = {
            "max_logit_diff": float((o32.logits.float() - o16.logits.float()).abs().max()),
            "mean_logit_diff": float((o32.logits.float() - o16.logits.float()).abs().mean()),
            "ce_diff": abs(ce(o32, evb) - ce(o16, evb)),
            "expert_agreement_min_layer": agree(o32.routing, o16.routing, nonpad)}
        del o32, o16
        for prec in ("fp32", "bf16"):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=(prec == "bf16")):
                o, n = orig(evb, RoutingConfig()), new(evb)
            g2[prec] = {"max_logit_diff": float((o.logits.float() - n.logits.float()).abs().max()),
                        "mean_logit_diff": float((o.logits.float() - n.logits.float()).abs().mean()),
                        "ce_orig": ce(o, evb), "ce_new": ce(n, evb), "aux_diff": float((o.aux_loss - n.aux_loss).abs()),
                        "z_diff": float((o.z_loss - n.z_loss).abs()),
                        "expert_agreement_min_layer": agree(o.routing, n.routing, nonpad)}
            dce = abs(g2[prec]["ce_orig"] - g2[prec]["ce_new"])
            if prec == "fp32":
                ok = g2[prec]["max_logit_diff"] < TOL["fp32_logit"] and dce < 1e-4 \
                    and g2[prec]["expert_agreement_min_layer"] >= 0.999
            else:  # bf16: bounded by loss, mean logit difference and expert agreement (max |dlogit| recorded)
                ok = dce < TOL["bf16_ce"] and g2[prec]["mean_logit_diff"] < TOL["bf16_mean_logit"] \
                    and g2[prec]["expert_agreement_min_layer"] >= TOL["bf16_agree"]
            check(BASELINE, f"G2_{prec}", ok, g2[prec])
        res["G2_baseline_vs_original"] = g2
        del orig, new
        for cid in PLANS:
            r = {}
            c = cfg_for(cid)
            mf = with_ref(build_cla_model(c, dev, backend="flex"))
            if mf.plan.kind == "conditional":
                mf.train()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    mf(mb[0], calibrate=True)
            mf.eval()
            md = build_cla_model(c, dev, backend="dense")
            md.load_state_dict(mf.state_dict())
            md.eval()
            # ---- G1 flex vs dense (fp32)
            with torch.no_grad():
                of, od = mf(evb), md(evb)
            r["G1"] = {"max_logit_diff": float((of.logits - od.logits).abs().max()), "ce_flex": ce(of, evb),
                       "ce_dense": ce(od, evb), "expert_agreement_min_layer": agree(of.routing, od.routing, nonpad)}
            if of.gate:
                r["G1"]["gate_agreement_min_layer"] = min(float((x["sel"] == y["sel"])[nonpad].float().mean())
                                                          for x, y in zip(of.gate, od.gate))
                r["G1"]["global_rate_by_layer"] = [float(x["sel"].sum() / nonpad.sum()) for x in of.gate]
            check(cid, "G1", r["G1"]["max_logit_diff"] < TOL["fp32_logit"] and r["G1"]["expert_agreement_min_layer"]
                  >= TOL["agree"] and r["G1"].get("gate_agreement_min_layer", 1.0) >= TOL["agree"], r["G1"])
            # ---- G3 future perturbation (flex)
            t = 600
            tb2 = {k: v.clone() for k, v in evb.items()}
            gen = torch.Generator(device="cpu").manual_seed(0)
            tb2["input_ids"][:, t + 1:] = torch.randint(2, 50000, tb2["input_ids"][:, t + 1:].shape, generator=gen).to(dev)
            with torch.no_grad():
                op = mf(tb2)
            r["G3"] = {"max_earlier_logit_diff": float((op.logits[:, :t + 1] - of.logits[:, :t + 1]).abs().max()),
                       "earlier_expert_agreement": agree([{"topi": x["topi"][:, :t + 1]} for x in op.routing],
                                                         [{"topi": x["topi"][:, :t + 1]} for x in of.routing],
                                                         nonpad[:, :t + 1])}
            if of.gate:
                r["G3"]["earlier_gate_agreement"] = min(float((x["sel"][:, :t + 1] == y["sel"][:, :t + 1]).float().mean())
                                                        for x, y in zip(op.gate, of.gate))
            check(cid, "G3", r["G3"]["max_earlier_logit_diff"] < 1e-4 and r["G3"]["earlier_expert_agreement"] >= 0.999
                  and r["G3"].get("earlier_gate_agreement", 1.0) >= 0.999, r["G3"])
            # ---- G4 incremental decoding vs full sequence (fp32)
            P0 = 480  # crosses every window (128 / 256 / 512) during the 64 decoded tokens
            doc0 = next(sp.doc(i) for row in cs.rows for i in row if len(sp.doc(i)) >= P0)
            prompt = torch.as_tensor(doc0[:P0], device=dev).view(1, -1).repeat(2, 1)
            gen_t, logits, sels = greedy(mf, prompt, 64, torch.float32)
            seq = torch.cat([prompt, gen_t[:, :-1]], 1)
            L = seq.shape[1]
            pad_to = (L + 127) // 128 * 128  # flex needs a multiple of 128: right-pad with a separate pad region
            tb = {"input_ids": torch.nn.functional.pad(seq, (0, pad_to - L), value=1),
                  "doc_ids": torch.nn.functional.pad(torch.zeros_like(seq), (0, pad_to - L), value=-1),
                  "positions": torch.nn.functional.pad(torch.arange(L, device=dev).unsqueeze(0).expand(2, L), (0, pad_to - L))}
            with torch.no_grad():
                ofs = mf(tb)
            full = ofs.logits[:, P0 - 1:L]
            inc = torch.stack(logits, 1)
            r["G4"] = {"max_logit_diff": float((full - inc).abs().max()),
                       "argmax_agreement": float((full.argmax(-1) == inc.argmax(-1)).float().mean())}
            if sels:
                agr = [float((s == ofs.gate[ci]["sel"][:, P0 + i]).float().mean()) for i, st in enumerate(sels)
                       for ci, s in enumerate(st)]
                r["G4"]["gate_agreement"] = float(np.mean(agr)) if agr else 1.0
            check(cid, "G4", r["G4"]["max_logit_diff"] < TOL["decode_logit"] and r["G4"]["argmax_agreement"] >= 0.99
                  and r["G4"].get("gate_agreement", 1.0) >= 0.99, r["G4"])
            del md
            # ---- G5 bf16 training steps (ordinary + counterfactual) from the real seed-98 initialisation
            mf = build_cla_model(c, dev, backend="flex")
            mf.train()
            if mf.plan.kind == "conditional":
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    mf(mb[0], calibrate=True)
            from csr.train_lib import lr_at
            opt = torch.optim.AdamW(mf.parameters(), lr=lr_at(0, 3052, base["train"]), betas=(0.9, 0.95),
                                    weight_decay=0.0, fused=True)  # real WSD schedule lr of steps 0 / 1
            torch.cuda.reset_peak_memory_stats()
            g5 = {}
            for kind in (["ordinary", "cf"] if mf.plan.kind == "conditional" else ["ordinary"]):
                tot = 0.0
                for x in mb:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        out = mf(x, cf=(kind == "cf"))
                    lg = out.logits.float()
                    loss = F.cross_entropy(lg.view(-1, lg.shape[-1]), x["targets"].view(-1), ignore_index=-100) \
                        + 0.01 * out.aux_loss + 0.001 * out.z_loss
                    (loss / 2).backward()
                    tot += float(loss)
                grads_ok = all(torch.isfinite(p.grad).all() for p in mf.parameters() if p.grad is not None)
                gg = [p.grad for n, p in mf.named_parameters() if n.startswith("gates.") and p.grad is not None]
                gn = float(torch.sqrt(sum((x.float() ** 2).sum() for x in gg))) if gg else 0.0
                gnorm = float(torch.nn.utils.clip_grad_norm_(mf.parameters(), 1.0))
                for gr in opt.param_groups:
                    gr["lr"] = lr_at(len(g5), 3052, base["train"])
                opt.step()
                opt.zero_grad(set_to_none=True)
                g5[kind] = {"loss": tot / 2, "finite": bool(np.isfinite(tot) and grads_ok and np.isfinite(gnorm)),
                            "grad_norm": gnorm, "gate_grad_norm": gn}
                check(cid, f"G5_{kind}", g5[kind]["finite"] and (gn > 0 if mf.plan.kind == "conditional" else gn == 0),
                      g5[kind])
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                oo = mf(mb[0])
            usage = [np.bincount(x["topi"][mb[0]["doc_ids"] >= 0].reshape(-1).cpu().numpy(), minlength=32)
                     for x in oo.routing]
            ev = {"experts_unused": [int((u == 0).sum()) for u in usage],
                  "expert_max_share": [float(u.max() / u.sum()) for u in usage]}
            g5["collapse"] = collapse_check(ev)
            g5["peak_alloc_gib"] = torch.cuda.max_memory_allocated() / 2**30
            g5["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
            check(cid, "G5_collapse_vram", not g5["collapse"]["collapsed"] and g5["peak_reserved_gib"] < 56, g5)
            r["G5"] = g5
            res["conditions"][cid] = r
            del mf, opt
            torch.cuda.empty_cache()
        # ---- G6 profiling of the compact path (G256-25 init; real CALIB-SEARCH layout, 16 rows)
        tb16 = to_torch(cs.batch(cs.rows[:16]), dev)
        ctx = AttnContext(tb16["doc_ids"], "flex")
        H, hd = 8, 24
        g = torch.Generator(device=dev).manual_seed(0)
        q, k, v = (torch.randn(16, H, 1024, hd, device=dev, dtype=torch.bfloat16, generator=g) for _ in range(3))
        elig = (tb16["doc_ids"] >= 0) & (tb16["positions"] >= 256)

        def tm(fn, n=30):
            for _ in range(5):
                fn()
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(n):
                fn()
            torch.cuda.synchronize()
            return (time.time() - t0) / n * 1e3
        g6 = {"full_attention_ms": tm(lambda: ctx.attend(q, k, v, None)),
              "local256_attention_ms": tm(lambda: ctx.attend(q, k, v, 256)),
              "full_tiles": kv_blocks_computed(ctx._mask(None)), "local256_tiles": kv_blocks_computed(ctx._mask(256)),
              "N": ctx.lay.N}
        for rate in (0.0, 0.10, 0.25, 0.50, 1.0):
            sel = elig & (torch.rand(16, 1024, device=dev, generator=g) < rate / max(float(elig.float().mean()), 1e-9))
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base_mem = torch.cuda.memory_allocated()
            ctx.attend_compact(q, k, v, sel)
            torch.cuda.synchronize()
            inc = torch.cuda.max_memory_allocated() - base_mem
            qpos = compact_positions(sel.reshape(-1))
            g6[f"rate{rate}"] = {"selected": int(sel.sum()), "compact_ms": tm(lambda: ctx.attend_compact(q, k, v, sel)),
                                 "tiles": kv_blocks_computed(compact_block_mask(qpos, ctx.lay)),
                                 "peak_mem_increment_bytes": int(inc)}
        nn_bytes = ctx.lay.N * ctx.lay.N * 2
        g6["dense_scores_bytes_bf16_one_head"] = nn_bytes
        g6["materialisation_check"] = all(g6[f"rate{r_}"]["peak_mem_increment_bytes"] < nn_bytes / 4
                                          for r_ in (0.0, 0.10, 0.25, 0.50, 1.0))
        g6["tiles_scale_with_rate"] = g6["rate0.0"]["tiles"] == 0 and g6["rate0.1"]["tiles"] < g6["rate0.25"]["tiles"] \
            < g6["rate0.5"]["tiles"] <= g6["rate1.0"]["tiles"]
        from torch.profiler import ProfilerActivity, profile
        sel = elig & (torch.rand(16, 1024, device=dev, generator=g) < 0.4)
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(3):
                ctx.attend_compact(q, k, v, sel)
            torch.cuda.synchronize()
        dtt = lambda e: getattr(e, "device_time_total", None) or getattr(e, "cuda_time_total", 0.0)  # noqa: E731
        g6["profile_top_kernels"] = [{"name": e.key[:80], "cuda_us": dtt(e) / 3, "calls": e.count / 3}
                                     for e in sorted(prof.key_averages(), key=lambda e: -dtt(e))[:10]]
        g6["no_dense_attention_kernel"] = not any(("bmm" in e.key or "softmax" in e.key.lower() or "sdpa" in e.key.lower()
                                                   or "fmha" in e.key.lower() or "flash" in e.key.lower())
                                                  for e in prof.key_averages())
        check("G6", "compact_profile", g6["materialisation_check"] and g6["tiles_scale_with_rate"]
              and g6["no_dense_attention_kernel"], {k: v for k, v in g6.items() if k != "profile_top_kernels"})
        res["G6_profile"] = g6
        status = "PASS" if not fails else "FAIL"
        if mon.contaminated:
            status = "CONTAMINATED"
    except Exception as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {e}"
        res["trace"] = traceback.format_exc()
        print(res["trace"])
    finally:
        mon.stop()
        from csr.cla_run import code_hashes, prereg_sha
        res.update({"status": status, "failures": fails, "time": guard.now(), "device": torch.cuda.get_device_name(0),
                    "formal": bool(a.formal), "code_sha256": code_hashes(),
                    "prereg_sha256": prereg_sha() if a.formal else None,
                    "torch": torch.__version__})
        out = ART / f"gpu_checks_{guard.utc_stamp()}.json"
        guard.atomic_write_json(out, res)
        guard.gpu_ledger(t_iso, guard.now(), out.stem, "clav1 gpu checks", status)
        guard.ledger("CLA_V1_GPU_CHECKS", f"{out.relative_to(guard.ROOT)} status={status} fails={len(fails)}")
        print(f"GPU_CHECKS {out.relative_to(guard.ROOT)} {status}")
        for f_ in fails:
            print("FAIL", f_[:400])
    sys.exit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()
