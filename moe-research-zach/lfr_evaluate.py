#!/usr/bin/env python
"""csr-latent-forecast-router-distill-v1 evaluation jobs (student / inference mode: no drafts, cache or labels).

  dev-audit --run-dir R : development specialisation audit of the 100% checkpoint on CALIB-SEARCH
                          (discovery = paper half A, measurement = paper half B; 0.2M tokens / domain each)
  calib-sel --run-dir R : CALIB-SELECTION CE (stage 3 only: requires artifacts/.../STAGE3_OPEN.json)
  final     --run-dir R : VAL, exactly once per frozen final run (requires frozen.json): full-VAL CE (overall, per
                          domain, per document) of the 100% checkpoint + specialisation audit of every milestone
                          (discovery = CALIB-SEARCH half A, measurement = VAL audit subset 0.2M tokens / domain)
GPU only via scripts/gpu_guarded.sh (--device cuda --execute). Results written atomically next to the run."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402

AUDIT_TOK = 200_000


def load_milestone(path, device):
    import torch
    from csr.lfr_model import LFRConfig, LFRModel
    ck = torch.load(guard.ROOT / path / "state.pt", map_location="cpu", weights_only=False)
    mc = LFRConfig.from_dict({**ck["model_cfg"], "teacher": False})
    m = LFRModel(mc, ck["config"]["seed"])
    m.load_state_dict(ck["model"])
    return m.to(device).eval(), ck


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
    from csr.lfr_lib import ART
    from csr.lfr_eval import eval_rows, evaluate, paper_half
    from csr.lfr_audit import audit
    rd = guard.inside_root(a.run_dir)
    summ = json.loads((rd / "summary.json").read_text())
    out_p = rd / {"dev-audit": "dev_audit.json", "calib-sel": "calib_sel_eval.json", "final": "final_eval.json"}[a.mode]
    if out_p.exists():
        raise guard.GuardError(f"{out_p} exists (each evaluation runs exactly once)")
    if a.mode == "calib-sel" and not (ART / "STAGE3_OPEN.json").exists():
        raise guard.GuardError("CALIB-SELECTION is opened only at stage 3")
    if a.mode == "final" and not (ART / "frozen.json").exists():
        raise guard.GuardError("VAL is opened only after the freeze")
    mon = None
    if a.device == "cuda":
        guard.occupancy_gate(f"lfr evaluate {a.mode}")
        device = guard.setup_cuda_device()
        mon = guard.OccupancyMonitor(interval=15.0)
        mon.start()
    else:
        device = torch.device("cpu")
    guard.register_self(f"lfr evaluate {a.mode} {rd.name}")
    t_iso, t0 = guard.now(), time.time()
    lab = ART / "labels"
    wc = dict(np.load(lab / "windows_calib.npz")) if (lab / "windows_calib.npz").exists() else None
    cs = Split("calib_search")
    disc_rows = eval_rows(cs, AUDIT_TOK, paper_half(cs, 0))
    res = {"mode": a.mode, "run": rd.name, "time": guard.now()}
    status = "COMPLETE"
    try:
        if a.mode == "dev-audit":
            m, _ = load_milestone(summ["ckpts"]["p100"]["path"], device)
            meas_rows = eval_rows(cs, AUDIT_TOK, paper_half(cs, 1))
            res["p100"] = audit(m, cs, disc_rows, cs, meas_rows, device, wc, wc)
        elif a.mode == "calib-sel":
            m, _ = load_milestone(summ["ckpts"]["p100"]["path"], device)
            sp = Split("calib_sel")
            r = evaluate(m, sp, eval_rows(sp), device, routing=True)
            res["calib_sel"] = {k: v for k, v in r.items() if k != "doc_ce"}
        else:
            wv = dict(np.load(lab / "windows_val.npz"))
            sp = Split("val")
            m, _ = load_milestone(summ["ckpts"]["p100"]["path"], device)
            r = evaluate(m, sp, eval_rows(sp), device, routing=True)
            res["val"] = {k: v for k, v in r.items() if k != "doc_ce"}
            res["val_doc_ce"] = {str(k): v for k, v in r["doc_ce"].items()}
            meas_rows = eval_rows(sp, AUDIT_TOK)
            res["audit"] = {}
            for tag in sorted(summ["ckpts"]):
                mt, _ = load_milestone(summ["ckpts"][tag]["path"], device)
                res["audit"][tag] = audit(mt, cs, disc_rows, sp, meas_rows, device, wc, wv)
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
        guard.gpu_ledger(t_iso, guard.now(), f"lfr_eval_{a.mode}_{rd.name}", f"lfr evaluate {a.mode}", status)
        with open(ART / "compute_ledger.tsv", "a") as f:
            f.write(f"{t_iso}\t{guard.now()}\t{time.time() - t0:.1f}\t{rd.name}\teval_{a.mode}\tevaluation\t{status}\n")
    print(json.dumps({"status": status, "seconds": res["seconds"]}))
    sys.exit(0 if status == "COMPLETE" else 77)


if __name__ == "__main__":
    main()
