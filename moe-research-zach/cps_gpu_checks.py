#!/usr/bin/env python
"""csr-context-prior-specialization-v2 GPU correctness checks: causality of routes and logits of the V2 model on GPU in
full fp32 (no autocast, no TF32) with real cached draft features (V1 calib_search diagnostic cache): teacher mode, student
mode, adaptive boundaries and the shuffled-context control; the future (tokens AND cached forecast features after t) is
replaced and the prefix routes / logits must not change. Also: the stripped model has no teacher parameters and runs
without drafts. Writes artifacts/.../gpu_checks_<stamp>.json; exit 0 iff all pass."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    if not guard.require_execution(a.execute, "gpu" if a.device == "cuda" else "cpu-model"):
        return
    import numpy as np
    import torch
    from csr.fsp_lib import make_batch, to_torch
    from csr.cps_lib import ART, CACHE_DIR, ForecastCache, gather_raw, build_features
    from csr.cps_model import CPSConfig, CPSModel
    from csr.lfr_diag import diag_set
    if a.device == "cuda":
        guard.occupancy_gate("cps gpu checks")
        device = guard.setup_cuda_device()
    else:
        device = torch.device("cpu")
    guard.register_self("cps gpu checks")
    t_iso, t0 = guard.now(), time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    th = json.loads((ART / "detector_calibration.json").read_text())["theta"]
    sp, rows = diag_set()
    cache = ForecastCache(CACHE_DIR, "calib_search")
    rr = rows[:4]
    bt = to_torch(make_batch(rr, sp), device)
    raw = gather_raw(rr, sp, cache)
    rt = {k: torch.as_tensor(v.astype(np.int64) if v.dtype == np.uint16 else v, device=device) for k, v in raw.items()}
    f1 = build_features(rt, bt["doc_ids"], bt["positions"])
    t = 400
    ids2 = bt["input_ids"].clone()
    ids2[:, t + 1:] = torch.randint(2, 50000, ids2[:, t + 1:].shape, device=device)
    rt2 = {k: v.clone() for k, v in rt.items()}
    rt2["src_tok"] = ids2.clone()
    for k in rt2:
        if k != "src_tok":
            rt2[k][:, t + 1:] = rt2[k][:, t + 1:].flip(1)
    f2 = build_features(rt2, bt["doc_ids"], bt["positions"])
    res = {}
    cases = {"teacher_fixed64": (CPSConfig(teacher=True), True),
             "student_fixed64": (CPSConfig(teacher=True), False),
             "teacher_adaptive64": (CPSConfig(teacher=True, chunk="adaptive", det_theta=th["64"]), True),
             "teacher_shuffled": (CPSConfig(teacher=True, shuffle_ctx=True), True),
             "teacher_middle_fixed16": (CPSConfig(teacher=True, inject="middle", chunk_len=16), True),
             "noresid": (CPSConfig(teacher=True, variant="noresid"), True)}
    for name, (cc, teach) in cases.items():
        m = CPSModel(cc, 0).to(device).eval()
        with torch.no_grad():  # make the prior non-trivial so that leakage would show
            m.cp_ab.fill_(0.0)
            m.cp_U.mul_(50.0)
        outs = []
        for ids_, f_ in ((bt["input_ids"], f1), (ids2, f2)):
            b = {"input_ids": ids_, "doc_ids": bt["doc_ids"], "positions": bt["positions"]}
            if teach:
                b.update(fc=f_, lam=torch.tensor(0.9, device=device))
            with torch.no_grad():
                o = m(b, {"teach": teach, "collect": True, "stats": True})
            outs.append((o, m.lm_logits(o["hidden"]).float()))
        agree = float(np.mean([float((r1["topi"][:, :t + 1] == r2["topi"][:, :t + 1]).float().mean())
                               for r1, r2 in zip(outs[0][0]["records"], outs[1][0]["records"])]))
        ld = float((outs[0][1][:, :t + 1] - outs[1][1][:, :t + 1]).abs().max())
        res[name] = {"topi_agreement_prefix": agree, "logit_max_abs_diff_prefix": ld,
                     "alpha_mean": outs[0][0]["cstats"]["alpha"][0] if outs[0][0]["cstats"]["alpha"] else None,
                     "pass": agree >= 0.999 and ld < 1e-3}
        del m
    # bf16 autocast training-mode forward + backward must be finite in every mode (deviation 5)
    torch.backends.cuda.matmul.allow_tf32 = True
    for name, (cc, teach) in cases.items():
        m = CPSModel(cc, 0).to(device).train()
        b = {"input_ids": bt["input_ids"], "doc_ids": bt["doc_ids"], "positions": bt["positions"]}
        if teach:
            b.update(fc=f1, lam=torch.tensor(0.9, device=device))
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            o = m(b, {"teach": teach})
            ce = torch.nn.functional.cross_entropy(m.lm_logits(o["hidden"]).float().reshape(-1, 50304),
                                                   bt["targets"].reshape(-1), ignore_index=-100)
            loss = ce + 0.01 * o["aux"] + 0.001 * o["z"] + o["kl"]
        loss.backward()
        gfin = all(bool(torch.isfinite(p_.grad).all()) for p_ in m.parameters() if p_.grad is not None)
        res[f"bf16_{name}"] = {"loss": float(loss), "finite_loss": bool(torch.isfinite(loss)), "finite_grads": gfin,
                               "pass": bool(torch.isfinite(loss)) and gfin}
        del m
    torch.backends.cuda.matmul.allow_tf32 = False
    # stripped model: no teacher parameters, inference without drafts
    m = CPSModel(CPSConfig(teacher=True), 0).to(device)
    m.strip_teacher()
    names = [n for n, _ in m.named_parameters()]
    with torch.no_grad():
        o = m({"input_ids": bt["input_ids"], "doc_ids": bt["doc_ids"], "positions": bt["positions"]}, {})
    res["stripped"] = {"n_tc_params": sum(n.startswith("tc_") for n in names),
                       "finite": bool(torch.isfinite(o["hidden"]).all()),
                       "pass": not any(n.startswith("tc_") for n in names) and bool(torch.isfinite(o["hidden"]).all())}
    res["all_pass"] = all(v["pass"] for v in res.values() if isinstance(v, dict))
    res["seconds"] = time.time() - t0
    out = ART / f"gpu_checks_{guard.utc_stamp()}.json"
    guard.atomic_write_json(out, res)
    if a.device == "cuda":
        guard.gpu_ledger(t_iso, guard.now(), "cps_gpu_checks", "cps gpu checks", "PASS" if res["all_pass"] else "FAIL")
        with open(ART / "compute_ledger.tsv", "a") as f:
            f.write(f"{t_iso}\t{guard.now()}\t{time.time() - t0:.1f}\tcps_gpu_checks\tgpu_checks\tchecks_and_tests\t"
                    f"{'PASS' if res['all_pass'] else 'FAIL'}\n")
    print(json.dumps(res, indent=1))
    sys.exit(0 if res["all_pass"] else 1)


if __name__ == "__main__":
    main()
