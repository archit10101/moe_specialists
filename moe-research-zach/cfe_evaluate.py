#!/usr/bin/env python
"""csr-context-prior-fast-engineering evaluation jobs (copy of cps_evaluate.py bound to the cfe paths; dev-audit adds
the routing change caused by disabling the context prior). Original V2 description: (inference mode: no drafts, no cache, no labels in the model).

  dev-audit --run-dir R : development audit of the 100% checkpoint on CALIB-SEARCH (discovery = paper half A,
                          measurement = paper half B, 0.2M tokens / domain each; identical to V1) + document aggregates
                          + CE with the context prior disabled (Q1).
  calib-sel --run-dir R : CALIB-SELECTION CE (stage 3 only: requires artifacts/.../STAGE3_OPEN.json of V2).
  final     --run-dir R : VAL, exactly once per final run, only after the V2 freeze (frozen.json): full-VAL CE (overall,
                          per domain, per document), VAL CE with the prior disabled, specialisation audit of every
                          milestone (40/60/80/100%) with document aggregates (discovery = CALIB-SEARCH half A,
                          measurement = VAL audit subset 0.2M tokens / domain; identical rows for every condition).
Works on V2 runs and on reused V1 TOKEN-4 runs (V1 files are only read; outputs go to the V2 artifact directory).
GPU only via scripts/gpu_guarded.sh (--device cuda --execute)."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402

AUDIT_TOK = 200_000


def load_any(path, device):
    """V2 milestone (ccfg) or V1 LFR milestone (model_cfg, linear router = TOKEN-4)."""
    import torch
    ck = torch.load(guard.ROOT / path / "state.pt", map_location="cpu", weights_only=False)
    if "ccfg" in ck:
        from csr.cfe_model import CFEConfig, CFEModel
        m = CFEModel(CFEConfig.from_dict({**ck["ccfg"], "teacher": False}), ck["config"]["seed"], ck["backbone"])
    else:
        from csr.lfr_model import LFRConfig, LFRModel
        mc = LFRConfig.from_dict({**ck["model_cfg"], "teacher": False})
        if mc.router != "linear":
            raise guard.GuardError("only V1 TOKEN-4 (linear router) checkpoints are reused by V2")
        m = LFRModel(mc, ck["config"]["seed"])
    m.load_state_dict(ck["model"])
    return m.to(device).eval(), ck


def out_path(rd, mode):
    from csr.cfe_lib import ART
    name = rd.name
    d = ART / "evals" / name
    d.mkdir(parents=True, exist_ok=True)
    return d / {"dev-audit": "dev_audit.json", "calib-sel": "calib_sel_eval.json", "final": "final_eval.json"}[mode]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["dev-audit", "calib-sel", "final"])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    if not guard.require_execution(a.execute, "gpu" if a.device == "cuda" else "cpu-model"):
        return
    import numpy as np
    import torch
    from csr.fsp_lib import Split
    from csr.cfe_lib import ART, V1_ART
    from csr.lfr_eval import eval_rows, evaluate, paper_half
    from csr.cps_audit import audit_v2
    rd = guard.inside_root(a.run_dir)
    summ = json.loads((rd / "summary.json").read_text())
    out_p = out_path(rd, a.mode)
    if out_p.exists():
        raise guard.GuardError(f"{out_p} exists (each evaluation runs exactly once)")
    if a.mode == "calib-sel" and not (ART / "STAGE3_OPEN.json").exists():
        raise guard.GuardError("CALIB-SELECTION is opened only after the fast-engineering freeze")
    if a.mode == "final" and not (ART / "frozen.json").exists():
        raise guard.GuardError("VAL is opened only after the fast-engineering freeze")
    mon = None
    if a.device == "cuda":
        guard.occupancy_gate(f"cfe evaluate {a.mode}")
        device = guard.setup_cuda_device()
        mon = guard.OccupancyMonitor(interval=15.0)
        mon.start()
    else:
        device = torch.device("cpu")
    guard.register_self(f"cfe evaluate {a.mode} {rd.name}")
    t_iso, t0 = guard.now(), time.time()
    lab = V1_ART / "labels"  # reused V1 labeler outputs (hash-verified in reuse_manifest.json)
    wc = dict(np.load(lab / "windows_calib.npz"))
    cs = Split("calib_search")
    disc_rows = eval_rows(cs, AUDIT_TOK, paper_half(cs, 0))
    res = {"mode": a.mode, "run": rd.name, "time": guard.now()}
    status = "COMPLETE"

    def prior_off(m, sp, rows):
        if not hasattr(m, "ccfg") or m.ccfg.variant == "token":
            return None
        m.force_ctx_off = True
        try:
            r0 = evaluate(m, sp, rows, device, routing=True)
        finally:
            m.force_ctx_off = False
        return {"ce": r0["ce"], "ce_dom": r0["ce_dom"], "ent": r0["ent"]}
    try:
        if a.mode == "dev-audit":
            m, _ = load_any(summ["ckpts"]["p100"]["path"], device)
            meas_rows = eval_rows(cs, AUDIT_TOK, paper_half(cs, 1))
            res["p100"] = audit_v2(m, cs, disc_rows, cs, meas_rows, device, wc, wc)
            res["p100"]["prior_off_meas"] = prior_off(m, cs, meas_rows)
            if hasattr(m, "ccfg") and m.ccfg.variant != "token":  # context influence on the measurement rows
                from csr.cfe_metrics import mi_domain, routing_change
                r1 = evaluate(m, cs, meas_rows, device, per_token=True, routing=True)["per_token"]
                m.force_ctx_off = True
                try:
                    r0 = evaluate(m, cs, meas_rows, device, per_token=True, routing=True)["per_token"]
                finally:
                    m.force_ctx_off = False
                ch, ov = routing_change(r1["topi"], r0["topi"])
                res["p100"]["ctx_influence"] = {"routing_changed_off": ch, "top4_overlap_off": ov,
                                                "mi_on": mi_domain(r1["topi"], r1["dom"]),
                                                "mi_off": mi_domain(r0["topi"], r0["dom"])}
        elif a.mode == "calib-sel":
            m, _ = load_any(summ["ckpts"]["p100"]["path"], device)
            sp = Split("calib_sel")
            r = evaluate(m, sp, eval_rows(sp), device, routing=True)
            res["calib_sel"] = {k: v for k, v in r.items() if k != "doc_ce"}
        else:
            wv = dict(np.load(lab / "windows_val.npz"))
            sp = Split("val")
            m, _ = load_any(summ["ckpts"]["p100"]["path"], device)
            r = evaluate(m, sp, eval_rows(sp), device, routing=True)
            res["val"] = {k: v for k, v in r.items() if k != "doc_ce"}
            res["val_doc_ce"] = {str(k): v for k, v in r["doc_ce"].items()}
            res["val_prior_off"] = prior_off(m, sp, eval_rows(sp))
            del m
            meas_rows = eval_rows(sp, AUDIT_TOK)
            res["audit"] = {}
            for tag in sorted(summ["ckpts"]):
                mt, _ = load_any(summ["ckpts"][tag]["path"], device)
                res["audit"][tag] = audit_v2(mt, cs, disc_rows, sp, meas_rows, device, wc, wv)
                del mt
    except KeyboardInterrupt:
        status = "EXTERNALLY_STOPPED"
    finally:
        if mon is not None:
            mon.stop()
            if mon.contaminated and status == "COMPLETE":
                status = "EXTERNALLY_STOPPED"
    res["seconds"] = time.time() - t0
    if status == "COMPLETE":
        guard.atomic_write_json(out_p, res, mode=0o400)
    if a.device == "cuda":
        guard.gpu_ledger(t_iso, guard.now(), f"cfe_eval_{a.mode}_{rd.name}", f"cfe evaluate {a.mode}", status)
        with open(ART / "compute_ledger.tsv", "a") as f:
            f.write(f"{t_iso}\t{guard.now()}\t{time.time() - t0:.1f}\t{rd.name}\teval_{a.mode}\tevaluation\t{status}\n")
    print(json.dumps({"status": status, "seconds": res["seconds"]}))
    sys.exit(0 if status == "COMPLETE" else 77)


if __name__ == "__main__":
    main()
