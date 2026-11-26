#!/usr/bin/env python
"""GPU correctness / memory / step-0 checks for csr-value-routed-compute-v1 and csr-m2d2-domain-prior-v1.
Dry-run by default; GPU only via scripts/gpu_guarded.sh ... vm_gpu_checks.py --execute. Writes one JSON per experiment
to artifacts/<exp>/gpu_checks_<stamp>.json. Uses only training / CALIB-SEARCH rows."""
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402


def main():
    if not guard.require_execution("--execute" in sys.argv, "gpu"):
        return
    import torch
    from csr.data import to_torch
    from csr.vm_run import EXP, build_model, check_cfg, eval_source, param_sha, train_source
    guard.occupancy_gate("vm gpu checks")
    dev = guard.setup_cuda_device()
    mon = guard.OccupancyMonitor(interval=15.0)
    mon.start()
    guard.register_self("vm gpu checks")
    t0iso, t0 = guard.now(), time.time()
    res = {"vrc": {}, "m2d2": {}}
    status = "FAILED"
    try:
        # ---------------------------------------------------------------- VRC
        cfg = json.loads((EXP["vrc"]["cfg"] / "diag/FULL-TOKEN4_L8192_s0.json").read_text())
        key, d = check_cfg(cfg)
        src = eval_source(cfg, key, d, "calib_search")
        from csr.vrc_model import VRCModel
        torch.manual_seed(0)
        m_flex = build_model(cfg, dev, backend="flex")
        m_dense = build_model(cfg, dev, backend="dense")
        m_dense.load_state_dict(m_flex.state_dict())
        m_flex.eval(), m_dense.eval()
        # agreement at 2048 (dense reference memory) using the first 2048 positions of real rows
        tb = to_torch(src.batch(src.rows[:2]), dev)
        tb = {k: v[:, :2048].contiguous() for k, v in tb.items()}
        nb = 2048 // 128
        g = torch.Generator().manual_seed(7)
        gsel = (torch.rand(2, nb, generator=g) > 0.5).to(dev)
        kblk = torch.tensor([1, 2, 4], device=dev)[torch.randint(0, 3, (2, nb), generator=g).to(dev)]
        valid = tb["targets"] != -100
        agree = {}
        with torch.no_grad():
            for name, kw in (("ref", {}), ("mixed", {"ref": False, "gsel": gsel, "kblk": kblk})):
                for prec in ("fp32", "bf16"):
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=prec == "bf16"):
                        a = m_flex(tb, **kw).logits.float()
                        b = m_dense(tb, **kw).logits.float()
                    agree[f"{name}_{prec}_max_abs"] = float((a - b)[valid].abs().max())
                    agree[f"{name}_{prec}_mean_abs"] = float((a - b)[valid].abs().mean())
        res["vrc"]["flex_vs_dense"] = agree
        res["vrc"]["flex_vs_dense_pass"] = bool(agree["ref_fp32_max_abs"] < 1e-3 and agree["mixed_fp32_max_abs"] < 1e-3
                                                and agree["ref_bf16_max_abs"] < 0.25 and
                                                agree["mixed_bf16_max_abs"] < 0.25)
        # causality (flex, mixed policy, fp32): perturb one token; earlier logits unchanged
        tb1 = to_torch(src.batch(src.rows[:1]), dev)
        T = tb1["input_ids"].shape[1]
        nb8 = T // 128
        gs1 = (torch.rand(1, nb8, generator=g) > 0.5).to(dev)
        kb1 = torch.tensor([1, 2, 4], device=dev)[torch.randint(0, 3, (1, nb8), generator=g).to(dev)]
        p = 5000
        with torch.no_grad():
            base = m_flex(tb1, ref=False, gsel=gs1, kblk=kb1).logits.float()
            tb2 = {k: v.clone() for k, v in tb1.items()}
            tb2["input_ids"][0, p] = (tb2["input_ids"][0, p] + 11) % 50000 + 2
            pert = m_flex(tb2, ref=False, gsel=gs1, kblk=kb1).logits.float()
            baser = m_flex(tb1).logits.float()
            pertr = m_flex(tb2).logits.float()
        res["vrc"]["causality_max_abs_before_p"] = float((base[0, :p] - pert[0, :p]).abs().max())
        res["vrc"]["causality_ref_max_abs_before_p"] = float((baser[0, :p] - pertr[0, :p]).abs().max())
        res["vrc"]["causality_changed_after_p"] = float((base[0, p:] - pert[0, p:]).abs().max())
        res["vrc"]["causality_pass"] = bool(res["vrc"]["causality_max_abs_before_p"] < 1e-3 and
                                            res["vrc"]["causality_ref_max_abs_before_p"] < 1e-3)
        # matched path: fallback (G, 4) equals REF exactly where local == full (doc position < 512)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            ones = torch.ones(1, nb8, dtype=torch.bool, device=dev)
            fb = m_flex(tb1, ref=False, gsel=ones, kblk=torch.full((1, nb8), 4, device=dev)).logits.float()
            rf = m_flex(tb1).logits.float()
        short = (tb1["positions"][0] < 512) & (tb1["doc_ids"][0] >= 0)
        res["vrc"]["matched_path_short_positions_max_abs"] = float((fb[0, short] - rf[0, short]).abs().max())
        res["vrc"]["matched_path_pass"] = res["vrc"]["matched_path_short_positions_max_abs"] < 0.05
        # NULL slot accounting
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            ev = {}
            for k in (1, 2, 4):
                o = m_flex(tb1, ref=False, gsel=ones, kblk=torch.full((1, nb8), k, device=dev))
                ev[k] = o.n_expert_evals
        npad = int((tb1["doc_ids"] >= 0).sum())
        res["vrc"]["expert_evals"] = {str(k): v for k, v in ev.items()}
        res["vrc"]["null_slots_pass"] = all(ev[k] == k * npad * 7 for k in ev)
        del m_dense
        torch.cuda.empty_cache()
        # memory: train step 2 x 8192 (fwd+bwd) and prefill 4 x 8192
        mem = {}
        m_flex.train()
        _, tsrc, steps, _ = train_source(cfg, key, d)
        torch.cuda.reset_peak_memory_stats()
        tbt = to_torch(tsrc.batch(steps[0][:2]), dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = m_flex(tbt)
        loss = torch.nn.functional.cross_entropy(out.logits.float().view(-1, out.logits.shape[-1]),
                                                 tbt["targets"].view(-1), ignore_index=-100)
        loss.backward()
        torch.cuda.synchronize()
        mem["train_2x8192_peak_gib"] = torch.cuda.max_memory_allocated() / 1024**3
        mem["train_loss_finite"] = bool(torch.isfinite(loss))
        gfin = all(torch.isfinite(p.grad).all() for p in m_flex.parameters() if p.grad is not None)
        mem["train_grads_finite"] = bool(gfin)
        m_flex.zero_grad(set_to_none=True)
        m_flex.eval()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            tbp = to_torch(src.batch(src.rows[:4]), dev)
            m_flex(tbp)
        torch.cuda.synchronize()
        mem["prefill_4x8192_peak_gib"] = torch.cuda.max_memory_allocated() / 1024**3
        res["vrc"]["memory"] = mem
        res["vrc"]["memory_pass"] = bool(mem["train_2x8192_peak_gib"] < 50 and mem["prefill_4x8192_peak_gib"] < 50
                                         and mem["train_loss_finite"] and gfin)
        res["vrc"]["ctx8192_pass"] = all(res["vrc"][k] for k in ("flex_vs_dense_pass", "causality_pass",
                                                                  "matched_path_pass", "null_slots_pass",
                                                                  "memory_pass"))
        del m_flex
        torch.cuda.empty_cache()
        # ---------------------------------------------------------------- M2D2 step-0 checks
        from csr.model import CSRModel, ModelConfig, RoutingConfig
        step0 = {}
        for s in (100, 0, 1):
            shas, models = {}, {}
            stage = "search" if s == 100 else "final"
            for c in ("B0", "B1", "B2", "B3"):
                mc = json.loads((EXP["m2d2"]["cfg"] / stage / f"s{s}_{c}.json").read_text())
                k2, d2 = check_cfg(mc)
                mm = build_model(mc, dev)
                shas[c] = param_sha(mm.backbone_state().items())
                models[c] = (mm, mc)
            msrc = eval_source(models["B0"][1], "m2d2", d2, "calib_search")
            tbm = to_torch(msrc.batch(msrc.rows[:8]), dev)
            with torch.no_grad():
                l = {c: models[c][0](tbm).logits.float() for c in models}
                torch.manual_seed(s)
                ref = CSRModel(ModelConfig.from_dict(models["B0"][1]["model"])).to(dev)
                lr_ = ref(tbm, RoutingConfig(mode="token", pool_size=8, warmup=16)).logits.float()
            step0[f"seed{s}"] = {
                "backbone_sha_equal": len(set(shas.values())) == 1, "backbone_sha": shas["B0"],
                "B0_vs_TOKEN4_max_abs": float((l["B0"] - lr_).abs().max()),
                "B2_vs_B1_max_abs": float((l["B2"] - l["B1"]).abs().max()),
                "B3_vs_B2_max_abs": float((l["B3"] - l["B2"]).abs().max()),
                "B1_vs_B0_max_abs": float((l["B1"] - l["B0"]).abs().max()),
                "dom_emb_equal_B1_B2": bool(torch.equal(models["B1"][0].dom_emb, models["B2"][0].dom_emb)),
            }
            # fwd/bwd finite for B2
            mm = models["B2"][0]
            mm.train()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = mm(tbm)
            ls = torch.nn.functional.cross_entropy(o.logits.float().view(-1, o.logits.shape[-1]),
                                                   tbm["targets"].view(-1), ignore_index=-100)
            ls.backward()
            step0[f"seed{s}"]["B2_fwd_bwd_finite"] = bool(torch.isfinite(ls)) and all(
                torch.isfinite(p.grad).all() for p in mm.parameters() if p.grad is not None)
            step0[f"seed{s}"]["B2_bias_grad_nonzero"] = bool(mm.dom_bias.grad.abs().sum() > 0)
            del models, ref
            torch.cuda.empty_cache()
        res["m2d2"]["step0"] = step0
        res["m2d2"]["step0_pass"] = all(v["backbone_sha_equal"] and v["B0_vs_TOKEN4_max_abs"] == 0.0 and
                                        v["B2_vs_B1_max_abs"] == 0.0 and v["B3_vs_B2_max_abs"] == 0.0 and
                                        v["dom_emb_equal_B1_B2"] and v["B2_fwd_bwd_finite"] and
                                        v["B2_bias_grad_nonzero"] for v in step0.values())
        if mon.contaminated:
            raise guard.GuardError(f"CONTAMINATED: {mon.detail}")
        status = "COMPLETE"
    except Exception as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {e}"
        res["traceback"] = traceback.format_exc()
        status = "CONTAMINATED" if "CONTAMINATED" in str(e) else "FAILED"
    finally:
        mon.stop()
        res["status"] = status
        res["seconds"] = time.time() - t0
        st = guard.utc_stamp()
        for k in ("vrc", "m2d2"):
            guard.atomic_write_json(EXP[k]["art"] / f"gpu_checks_{st}.json", {**res, "experiment": EXP[k]["id"]})
        guard.gpu_ledger(t0iso, guard.now(), f"gpu_checks_{st}", "vm gpu checks", status)
        print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
