#!/usr/bin/env python
"""fspv2 GPU correctness + causality checks at full size (random weights, random tokens; no data results).
1. Draft rollout causality on GPU (full draft size, bf16): perturbing real tokens after t leaves rollouts/scores <= t
   unchanged (exact for ids; scores within 1e-3).
2. Main model (spec + forecast teacher, bf16 autocast, eager and compiled): perturbing real tokens AND cached forecast
   features after t leaves hidden states / routes / domain posteriors at <= t unchanged within tolerance. Exact
   bit-equality is not expected: the dropless dispatch groups tokens by expert, so later tokens change matmul shapes
   (kernel choice) for earlier tokens. Tolerance: max |dh| <= 0.05 and route agreement >= 0.999.
3. Mandatory 3 general + 1 specialist on GPU (forced, annealed and standalone).
4. Compiled vs eager loss agreement (relative 2e-2).
Writes artifacts/forecast-specialist-pretraining-v2/gpu_checks_<stamp>.json. Dry-run by default."""
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402


def main():
    guard.check_root()
    if not guard.require_execution("--execute" in sys.argv, "gpu"):
        return
    import torch
    from csr.fsp_lib import N_SCALAR, build_draft, causal_scalars, draft_rollout
    from csr.fsp_model import FSPConfig, FSPModel, N_GENERAL
    guard.occupancy_gate("fsp gpu checks")
    dev = guard.setup_cuda_device()
    guard.register_self("fsp gpu checks")
    t_iso, t0 = guard.now(), time.time()
    res = {}
    B, T, t = 4, 1024, 600
    g = torch.Generator(device="cpu").manual_seed(0)
    ids = torch.randint(2, 50000, (B, T), generator=g).to(dev)
    doc = torch.zeros(B, T, dtype=torch.long, device=dev)
    doc[:, 800:] = 1
    pos = torch.arange(T, device=dev).expand(B, T).clone()
    pos[:, 800:] -= 800
    lab = torch.randint(0, 5, (B, 1), generator=g).to(dev).expand(B, T).clone()
    ids2 = ids.clone()
    ids2[:, t + 1:] = torch.randint(2, 50000, (B, T - t - 1), generator=g).to(dev)

    def check(name, fn):
        try:
            res[name] = {"pass": True, **(fn() or {})}
        except Exception:
            res[name] = {"pass": False, "error": traceback.format_exc()[-2000:]}
        print(name, json.dumps(res[name])[:400], flush=True)

    draft = build_draft(3).to(dev).eval()

    def fc_of(x):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            roll, ent, top, lp = draft_rollout(draft, x, doc, pos, H=8)
        lp5 = torch.stack([lp + 0.1 * k for k in range(5)], -1).float()
        return {"roll": roll, "ent": ent.float(), "top": top.float(), "lp_a": lp5[..., 0], "lp_o": lp5[..., 1:]}

    f1, f2 = fc_of(ids), fc_of(ids2)

    def c1():
        assert torch.equal(f1["roll"][:, :t + 1], f2["roll"][:, :t + 1])
        d = (f1["lp_a"][:, :t + 1] - f2["lp_a"][:, :t + 1]).abs().max().item()
        assert d <= 1e-3, d
        assert not torch.equal(f1["roll"][:, t + 1:], f2["roll"][:, t + 1:])
        return {"max_lp_diff_prefix": d}
    check("draft_rollout_causal_gpu", c1)
    f2["roll"][:, t + 1:] = torch.randint(2, 50000, f2["roll"][:, t + 1:].shape, device=dev)
    cfg = FSPConfig(routing="spec", teacher="spec", n_scalar=N_SCALAR, teacher_init=1.0)
    m = FSPModel(cfg, 0).to(dev)

    def mi_of(x, f, sigma, forced):
        sc = causal_scalars(f, x, doc, pos, torch.zeros_like(pos))
        return {"input_ids": x, "doc_ids": doc, "positions": pos, "domain": lab, "forced": forced,
                "fc": {"roll": f["roll"], "scal": sc}, "sigma": torch.tensor(sigma, device=dev)}

    def run(model_fn, x, f, sigma=1.0, forced=None, amp=True):
        forced = forced if forced is not None else torch.zeros_like(lab, dtype=torch.bool)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            return model_fn(mi_of(x, f, sigma, forced), {"teach": sigma > 0, "use_dom": True, "collect": True})

    def diffs(o1, o2, rows, upto):
        dh = (o1["hidden"][rows, :upto].float() - o2["hidden"][rows, :upto].float()).abs().max().item()
        agree = min(float((r1["topi"][rows, :upto] == r2["topi"][rows, :upto]).float().mean())
                    for r1, r2 in zip(o1["records"], o2["records"]))
        dq = max((r1["q_s"][rows, :upto] - r2["q_s"][rows, :upto]).abs().max().item()
                 for r1, r2 in zip(o1["records"], o2["records"]))
        return dh, agree, dq

    def c2_strict():
        # fp32 (TF32 off): perturb real tokens AND forecasts after t in every row; prefix must match to fp32 noise
        torch.backends.cuda.matmul.allow_tf32 = False
        m32 = m.float()
        o1, o2 = run(m32, ids, f1, amp=False), run(m32, ids2, f2, amp=False)
        dh, agree, dq = diffs(o1, o2, slice(None), t + 1)
        assert dh <= 1e-3 and agree >= 0.9999 and dq <= 1e-3, (dh, agree, dq)
        return {"fp32_max_hidden_diff_prefix": dh, "fp32_min_route_agreement_prefix": agree, "fp32_max_q_diff": dq}

    def c2(fn_name, fn):
        def inner():
            # bf16: prefix perturbation effect must not exceed the noise floor produced by perturbing ONLY OTHER rows
            # (row 0 untouched in the control: any change there is pure batch-composition numerics, not leakage)
            o1, o2 = run(fn, ids, f1), run(fn, ids2, f2)
            dh, agree, dq = diffs(o1, o2, slice(None), t + 1)
            idc = ids.clone()
            idc[1:, :] = ids2[1:, :]
            fc_ = {k: v.clone() for k, v in f1.items()}
            for k in fc_:
                fc_[k][1:] = f2[k][1:]
            oc = run(fn, idc, fc_)
            nh, nagree, nq = diffs(o1, oc, slice(0, 1), T)
            assert agree >= min(nagree, 0.999) - 0.005 and dh <= max(2 * nh, 0.02), (dh, agree, nh, nagree)
            return {"bf16_max_hidden_diff_prefix": dh, "bf16_min_route_agreement_prefix": agree,
                    "bf16_noise_floor_hidden_diff_untouched_row": nh, "bf16_noise_floor_route_agreement": nagree}
        return inner
    check("main_causality_fp32_strict", c2_strict)
    m.to(dev)
    torch.backends.cuda.matmul.allow_tf32 = True
    check("main_causality_eager_bf16", c2("eager", m))
    mc = torch.compile(m)
    check("main_causality_compiled_bf16", c2("compiled", mc))

    def c3():
        out = {}
        for nm, sig, forced in (("forced", 1.0, lab >= 0), ("anneal", 0.5, torch.rand(B, T, device=dev) < 0.5),
                                ("standalone", 0.0, None)):
            o = run(m, ids, f1, sig, forced)
            for r in o["records"]:
                ti = r["topi"].reshape(-1, 4)
                assert (ti[:, :3] < N_GENERAL).all() and (ti[:, 3] >= N_GENERAL).all()
                assert (ti.sort(-1).values.diff(dim=-1) > 0).all()
                if forced is not None:
                    f = forced.reshape(-1)
                    assert ((ti[:, 3] - N_GENERAL) // 2)[f].eq(lab.reshape(-1)[f]).all()
            out[nm] = "ok"
        return out
    check("mandatory_3plus1_gpu", c3)

    def c4():
        import torch.nn.functional as F
        x = mi_of(ids, f1, 1.0, lab >= 0)
        ctl = {"teach": True, "use_dom": True}
        vals = []
        for fn in (m, mc):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                o = fn(x, ctl)
                lg = m.lm_logits(o["hidden"]).float()
                ce = F.cross_entropy(lg.view(-1, lg.shape[-1]), ids.view(-1))
            vals.append([float(ce), float(o["aux"]), float(o["dom_ce"]), float(o["distill"])])
        rel = max(abs(a - b) / max(1e-6, abs(a)) for a, b in zip(*vals))
        assert rel <= 2e-2, vals
        return {"eager": vals[0], "compiled": vals[1], "max_rel_diff": rel}
    check("compiled_vs_eager", c4)
    ok = all(v["pass"] for v in res.values())
    p = guard.ROOT / f"artifacts/forecast-specialist-pretraining-v2/gpu_checks_{guard.utc_stamp()}.json"
    guard.atomic_write_json(p, {"time": guard.now(), "all_pass": ok, "results": res})
    t_end = guard.now()
    guard.gpu_ledger(t_iso, t_end, "fsp_gpu_checks", "fsp gpu checks", "PASS" if ok else "FAIL")
    with open(guard.ROOT / "artifacts/forecast-specialist-pretraining-v2/compute_ledger.tsv", "a") as f:
        f.write(f"{t_iso}\t{t_end}\t{time.time() - t0:.1f}\tfsp_gpu_checks\tgpu_checks\tchecks_and_tests\t"
                f"{'PASS' if ok else 'FAIL'}\n")
    print("ALL PASS" if ok else "FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
