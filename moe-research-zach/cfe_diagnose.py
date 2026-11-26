#!/usr/bin/env python
"""csr-context-prior-fast-engineering read-only diagnosis of the inert-prior failure (CALIB-SEARCH + main training
batches only; no validation / test). On existing V2 development checkpoints (S2 C07 s100/s101 p100, student-only):
  (1) alpha sweep: CE, routing change vs prior-off, topic(domain)-expert MI with the gate forced to constants
      (prior shape as learned) -> is there any alpha at which the learned prior helps / how much does it cost?
  (2) gradient decomposition on main-stream batches of the run's own plan (last steps): d/d(alpha bias) and
      d/d(prior weights) of the LM term, the Switch aux term (x0.01) and the router z term (x0.001), at the
      learned alpha and at forced alpha = 0.3 -> which loss pushes alpha down?
Writes artifacts/context-prior-fast-engineering/diagnosis_inert_<stamp>.json."""
import argparse
import json
import math
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402

RUNS = ["s2_C07_s100", "s2_C07_s101"]
ALPHAS = [None, 0.0, 0.03, 0.1, 0.2, 0.4, 0.7, 1.0]  # None = as learned


def mi_domain(topi, dom, E=32, n_dom=5):
    import numpy as np
    out = []
    for l in range(topi.shape[1]):
        c = np.zeros((n_dom, E))
        for k in range(topi.shape[2]):
            np.add.at(c, (dom, topi[:, l, k].astype(np.int64)), 1.0)
        p = c / c.sum()
        pd, pe = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
        nz = p > 0
        out.append(float((p[nz] * np.log(p[nz] / (pd @ pe)[nz])).sum()))
    return float(np.mean(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--tok-per-dom", type=int, default=100_000)
    a = ap.parse_args()
    guard.check_root()
    if not guard.require_execution(a.execute, "gpu" if a.device == "cuda" else "cpu-model"):
        return
    import numpy as np
    import torch
    from csr.fsp_lib import Split, make_batch, to_torch
    from csr.lfr_eval import eval_rows, evaluate
    from csr.cfe_lib import ART, BUDGET, main_stream
    from cps_evaluate import load_any
    import cps_driver as drv
    if a.device == "cuda":
        guard.occupancy_gate("cfe diagnose")
        device = guard.setup_cuda_device()
    else:
        device = torch.device("cpu")
    guard.register_self("cfe diagnose inert prior")
    t_iso, t0 = guard.now(), time.time()
    cs = Split("calib_search")
    rows = eval_rows(cs, a.tok_per_dom)
    split, steps, _ = main_stream(BUDGET["S2"])
    res = {"time": guard.now(), "tok_per_dom": a.tok_per_dom, "runs": {}}
    for rn in RUNS:
        rd = drv.find_run_dir(rn)
        summ = json.loads((rd / "summary.json").read_text())
        m, ck = load_any(summ["ckpts"]["p100"]["path"], device)
        orig_alpha = m.alpha
        R = {"ckpt": summ["ckpts"]["p100"]["path"], "sweep": {}}

        def set_alpha(v):
            if v is None:
                m.alpha = orig_alpha
            else:
                m.alpha = types.MethodType(lambda self, j, c, h32: torch.full((c.shape[0],), float(v), device=c.device,
                                                                              dtype=torch.float32), m)
        m.force_ctx_off = True
        base = evaluate(m, cs, rows, device, per_token=True, routing=True)
        pt0 = base.pop("per_token")
        base["topi"], base["dom"] = pt0["topi"], pt0["dom"]
        m.force_ctx_off = False
        R["sweep"]["off"] = {"ce": base["ce"], "ce_dom": base["ce_dom"],
                             "mi_domain": mi_domain(base["topi"], base["dom"])}
        for v in ALPHAS:
            set_alpha(v)
            r = evaluate(m, cs, rows, device, per_token=True, routing=True)
            pt_ = r.pop("per_token")
            r["topi"], r["dom"] = pt_["topi"], pt_["dom"]
            same = (np.sort(r["topi"], -1) == np.sort(base["topi"], -1)).all(-1)  # (N, L) identical top-4 set
            ov = np.mean([np.mean([len(set(x) & set(y)) for x, y in zip(r["topi"][::50, l], base["topi"][::50, l])])
                          for l in range(r["topi"].shape[1])])
            R["sweep"]["learned" if v is None else f"{v:.2f}"] = {
                "ce": r["ce"], "dce_vs_off": r["ce"] - base["ce"], "ce_dom": r["ce_dom"],
                "frac_slots_changed_vs_off": float(1.0 - same.mean()), "mean_top4_overlap_vs_off": float(ov),
                "mi_domain": mi_domain(r["topi"], r["dom"]), "ent": r["ent"]}
            e_ = R["sweep"]["learned" if v is None else f"{v:.2f}"]
            print(rn, v, f"ce={e_['ce']:.5f} dce={e_['dce_vs_off']:+.5f} changed={e_['frac_slots_changed_vs_off']:.3f} "
                         f"mi={e_['mi_domain']:.5f}", flush=True)
        set_alpha(None)
        # gradient decomposition on the run's own last training batches (student-only phase)
        m.train()
        grads = {}
        for tag, v in (("learned", None), ("forced_0.30", 0.3)):
            set_alpha(v)
            acc = {}
            for sidx in (len(steps) - 3, len(steps) - 2):
                rr = steps[sidx][:8]
                bt = to_torch(make_batch(rr, split), device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    out = m({"input_ids": bt["input_ids"], "doc_ids": bt["doc_ids"], "positions": bt["positions"]},
                            {"teach": False})
                    h = out["hidden"].reshape(-1, out["hidden"].shape[-1])
                    y = bt["targets"].reshape(-1)
                    lg = torch.nn.functional.linear(h, m.tok_emb.weight).float()
                    valid = y >= 0
                    lm = ((torch.logsumexp(lg, -1) - lg.gather(1, y.clamp(min=0).view(-1, 1)).squeeze(1)) * valid
                          ).sum() / valid.sum()
                terms = {"lm": lm, "aux": 0.01 * out["aux"], "z": 0.001 * out["z"]}
                params = {"cp_ab": m.cp_ab, "cp_U": m.cp_U, "cp_aw": m.cp_aw, "cp_av": m.cp_av}
                for tn, tv in terms.items():
                    gs = torch.autograd.grad(tv, list(params.values()), retain_graph=True, allow_unused=True)
                    for (pn, _), g in zip(params.items(), gs):
                        if g is None:
                            continue
                        e = acc.setdefault(f"{tn}:{pn}", {"norm": 0.0, "sum": 0.0, "per_layer": None})
                        e["norm"] += float(g.float().norm()) / 2
                        e["sum"] += float(g.float().sum()) / 2
                        if pn == "cp_ab":
                            pl = g.float().cpu().numpy() / 2
                            e["per_layer"] = (pl if e["per_layer"] is None else e["per_layer"] + pl)
                del out, h, lg, terms, lm
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            for k_, e in acc.items():
                if e["per_layer"] is not None:
                    e["per_layer"] = [float(x) for x in e["per_layer"]]
            grads[tag] = acc
            print(rn, tag, {k_: round(e["sum"], 6) for k_, e in acc.items() if k_.endswith("cp_ab")}, flush=True)
        set_alpha(None)
        R["grads"] = grads
        R["note"] = ("gradient sign: a positive d(term)/d(cp_ab) means gradient descent on that term DEcreases the "
                     "alpha bias (shrinks alpha). alpha forced via a constant gate; 'learned' uses the trained gate.")
        res["runs"][rn] = R
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()
    res["seconds"] = time.time() - t0
    out = ART / f"diagnosis_inert_{guard.utc_stamp()}.json"
    guard.atomic_write_json(out, res)
    if a.device == "cuda":
        guard.gpu_ledger(t_iso, guard.now(), "cfe_diagnose", "cfe diagnose inert prior", "COMPLETE")
        with open(ART / "compute_ledger.tsv", "a") as f:
            f.write(f"{t_iso}\t{guard.now()}\t{time.time() - t0:.1f}\tcfe_diagnose\tdiagnose_inert\tdiagnosis\t"
                    f"COMPLETE\n")
    print(out)


if __name__ == "__main__":
    main()
