#!/usr/bin/env python
"""csr-domain-forecast-curriculum-v1: targeted GPU correctness checks (implementation checks, not a scientific gate).
Full-size models, real MAIN rows (+ real forecasts when the cache exists, else random ids):
  1 identical initialisation across all 7 conditions per seed (full and backbone hashes), backbone == CSRModel(seed)
  2 strength 0: logits/routes bitwise equal to the plain CSRModel export, with no forecast/domain inputs
  3 exactly 4 distinct routed experts per token at strength 1; no expert ever masked (all 32 used somewhere)
  4 future-token perturbation: draft forecasts at <= t bitwise unchanged; main-model routes/logits at <= t unchanged
    (bf16 GPU: tolerance, because the dropless dispatch buffer shape depends on later tokens' routes)
  5 one-batch smoke train step per condition (forward + backward + AdamW step) at micro-batch 16: finite, peak memory
  6 shared-expert outputs identical across conditions
Writes artifacts/domain-forecast-curriculum-v1/gpu_checks_<stamp>.json. Dry-run by default."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft-run")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    if not guard.require_execution(a.execute, "gpu"):
        return
    import numpy as np
    import torch
    import torch.nn.functional as F
    from csr.dfc_lib import ART, CACHE, code_hashes, main_plan, make_batch, to_torch
    from csr.dfc_model import (CONDITIONS, MAIN_MODEL_CFG, build_dfc_model, build_draft, draft_forecasts,
                               load_frozen_draft, to_plain_csr)
    from csr.model import CSRModel, ModelConfig, RoutingConfig
    from dfc_train import make_optimizer, param_sha
    guard.occupancy_gate("dfc gpu checks")
    device = guard.setup_cuda_device()
    mon = guard.OccupancyMonitor(interval=15.0)
    mon.start()
    guard.register_self("dfc gpu checks")
    t_iso, t0 = guard.now(), time.time()
    res, ok = {}, True
    status = "FAILED"
    try:
        have_cache = (CACHE / "main_correct.npy").exists()
        plan = main_plan(0, "correct" if have_cache else None)
        rows = [plan.rows[i] for i in range(16)]
        b = make_batch(rows, plan.split, plan.doms, plan.cache)
        if not have_cache:
            b["fc"] = np.random.default_rng(0).integers(2, 50000, (16, 1024, 4))
        tb = to_torch(b, device)
        # 1 init identity
        r1 = {}
        for seed in (0, 1, 2):
            shas = {}
            for c in CONDITIONS:
                m = build_dfc_model(MAIN_MODEL_CFG, c, seed, device)
                shas[c] = (param_sha(m.named_parameters()), param_sha(m.backbone_state().items()))
            torch.manual_seed(seed)
            plain = CSRModel(ModelConfig.from_dict(MAIN_MODEL_CFG)).to(device)
            bb_ok = shas["SOFT"][1] == param_sha(plain.named_parameters())
            r1[str(seed)] = {"all_conditions_identical": len(set(shas.values())) == 1, "backbone_equals_csrmodel": bb_ok,
                        "sha": shas["SOFT"][0]}
            ok &= r1[str(seed)]["all_conditions_identical"] and bb_ok
        r1["seeds_differ"] = len({r1[str(s)]["sha"] for s in (0, 1, 2)}) == 3
        res["init_identity"] = r1
        # 2, 3, 6
        r2, r3 = {}, {}
        shared_ref = None
        xs = torch.randn(64, 192, device=device)
        nolab = {k: tb[k] for k in ("input_ids", "doc_ids", "positions")}
        for c in CONDITIONS:
            m = build_dfc_model(MAIN_MODEL_CFG, c, 0, device).eval()
            plain = to_plain_csr(m).eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                o0 = m(nolab, strength=0.0)
                op = plain(nolab, RoutingConfig())
                o1 = m(tb, strength=1.0)
            eq = torch.equal(o0.logits, op.logits) and all(torch.equal(x["topi"], y["topi"])
                                                           for x, y in zip(o0.routing, op.routing))
            r2[c] = eq
            ok &= eq
            nonpad = tb["doc_ids"] >= 0
            distinct = all(bool((r["topi"][nonpad].sort(-1).values.diff(dim=-1) > 0).all()) for r in o1.routing)
            used = [int(torch.unique(r["topi"][nonpad]).numel()) for r in o1.routing]
            changed = float(np.mean([float((x["topi"] != y["topi"]).any(-1)[nonpad].float().mean())
                                     for x, y in zip(o1.routing, o0.routing)]))
            r3[c] = {"four_distinct": distinct, "experts_used_per_layer": used, "frac_tokens_rerouted": changed,
                     "finite": bool(torch.isfinite(o1.logits).all())}
            ok &= distinct and r3[c]["finite"]
            with torch.no_grad():
                sh = m.blocks[1].ffn.shared(xs)
            if shared_ref is None:
                shared_ref = sh
            r3[c]["shared_equal"] = bool(torch.equal(sh, shared_ref))
            ok &= r3[c]["shared_equal"]
        res["strength0_equals_plain"] = r2
        res["routing"] = r3
        # 4 future perturbation (draft + main)
        if a.draft_run:
            draft = load_frozen_draft(guard.inside_root(a.draft_run) / "ckpt_final.pt", device)
        else:
            draft = build_draft(0, device).eval()
            with torch.no_grad():
                for la in draft.adapters:
                    for ad in la:
                        ad.up.weight.normal_(0, 0.02)
        di = rows[0][0]
        x = plan.split.doc(di)
        n = len(x)
        dom = int(plan.doms[di])
        r4 = []
        m = build_dfc_model(MAIN_MODEL_CFG, "SOFT", 0, device).eval()
        for t in (10, n // 2, n - 20):
            outs, fcs = [], []
            for v in range(2):
                xx = x.copy()
                if v:
                    xx[t + 1:] = np.random.default_rng(t).integers(2, 50000, n - t - 1)
                bb = {"input_ids": torch.tensor(xx[None], device=device),
                      "doc_ids": torch.zeros(1, n, dtype=torch.long, device=device),
                      "positions": torch.arange(n, device=device)[None],
                      "domain": torch.full((1, n), dom, device=device)}
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    bb["fc"] = draft_forecasts(draft, bb["input_ids"], bb["doc_ids"], bb["positions"], bb["domain"])
                    outs.append(m(bb, strength=1.0))
                fcs.append(bb["fc"])
            fc_eq = bool(torch.equal(fcs[0][0, :t + 1], fcs[1][0, :t + 1]))
            fc_later_changed = bool((fcs[0][0, t + 1:] != fcs[1][0, t + 1:]).any())
            dl = float((outs[0].logits[0, :t + 1].float() - outs[1].logits[0, :t + 1].float()).abs().max())
            ragree = float(np.mean([float((p["topi"][0, :t + 1].sort(-1).values ==
                                           q["topi"][0, :t + 1].sort(-1).values).all(-1).float().mean())
                                    for p, q in zip(outs[0].routing, outs[1].routing)]))
            r4.append({"t": int(t), "forecast_prefix_bitwise_equal": fc_eq, "later_forecasts_changed": fc_later_changed,
                       "main_logit_maxabs_diff_prefix": dl, "main_route_agreement_prefix": ragree})
            ok &= fc_eq and dl < 0.05 and ragree > 0.999
        res["future_perturbation"] = r4
        # 5 smoke train step per condition
        r5 = {}
        for c in CONDITIONS:
            torch.cuda.reset_peak_memory_stats()
            m = build_dfc_model(MAIN_MODEL_CFG, c, 0, device).train()
            t_ = json.loads((guard.ROOT / "configs/domain-forecast-curriculum-v1/main/SOFT_s0.json").read_text())["train"]
            opt = make_optimizer(m, t_, device)
            ts = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = m(tb, strength=torch.ones(16, device=device))
            loss = F.cross_entropy(out.logits.float().view(-1, out.logits.shape[-1]), tb["targets"].view(-1),
                                   ignore_index=-100) + 0.01 * out.aux_loss + 0.001 * out.z_loss
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            torch.cuda.synchronize()
            r5[c] = {"loss": float(loss), "grad_norm": float(gn), "finite": bool(torch.isfinite(loss)) and
                     bool(torch.isfinite(gn)), "peak_alloc_gib": torch.cuda.max_memory_allocated() / 1024**3,
                     "seconds": time.time() - ts}
            ok &= r5[c]["finite"]
            del m, opt, out, loss
        res["smoke_step"] = r5
        status = "PASS" if ok else "FAIL"
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        res["error"] = f"{type(e).__name__}: {e}"
        status = "FAIL"
    finally:
        mon.stop()
        if mon.contaminated:
            status = "CONTAMINATED"
        doc = {"status": status, "results": res, "time": guard.now(), "seconds": time.time() - t0,
               "draft_run": a.draft_run, "code_sha256": code_hashes()}
        out = ART / f"gpu_checks_{guard.utc_stamp()}.json"
        guard.atomic_write_json(out, doc)
        guard.gpu_ledger(t_iso, guard.now(), out.stem, "dfc gpu checks", status)
        guard.ledger("DFC_GPU_CHECKS", f"{out.relative_to(guard.ROOT)} {status}")
        print(json.dumps(doc, indent=1, default=str)[:6000])
        print("STATUS", status)
    sys.exit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()
