#!/usr/bin/env python
"""csr-soft-cf-usefulness-v2 post-terminal DESCRIPTIVE VAL routing pass (opened only after every V1 and V2 decision was
frozen; never used for any decision). For each 50M checkpoint: VAL CE on the semantic-edl-long-k audit rows
(eval_rows(val, 200k tokens per domain)), real-domain and e5-cluster routing NMI / MI per layer (slk_lib.routing_subject,
identical definition to semantic-edl-long-k: NMI = 2 MI / (H_topic + H_expert)), load-preserving and document-label
null NMIs, and per-domain CE. Writes artifacts/soft-cf-usefulness-v2/post_val/<key>.json (once each, 0400)."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402

AUDIT_TOK = 200_000


def targets():
    import scu_run as R
    from csr.scu_paths import SUE_ART
    sue = json.loads((SUE_ART / "run_dirs.json").read_text())
    scu = R.run_dirs()
    t = {f"T4_s{s}": R.REUSED50[f"T4_s{s}"] for s in (100, 101)}
    t.update({f"EDL_s{s}": R.REUSED50[f"EDL_s{s}"] for s in (100, 101)})
    for s in (100, 101):
        for n in ("REF_C6", "REF_SHUF", "S_U2", "H_U2", "S_G1a", "H_G1a"):
            t[f"V1_{n}_s{s}"] = sue[f"{n}_s{s}"]
        for n in ("S_C4", "H_C4"):
            t[f"V2_{n}_s{s}"] = scu[f"{n}_s{s}"]
    return t


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
    from fle_evaluate import load_milestone
    from csr.fsp_lib import Split
    from csr.lfr_audit import token_labels
    from csr.lfr_eval import eval_rows, evaluate
    from csr.scu_paths import ART, LFR_LABELS, check_prereg
    from csr.slk_lib import routing_subject
    check_prereg()
    if not (ART / "promotion.json").exists():
        raise guard.GuardError("descriptive VAL pass only after the V2 promotion decision is frozen")
    od = ART / "post_val"
    od.mkdir(exist_ok=True)
    mon = None
    if a.device == "cuda":
        guard.occupancy_gate("scu post val")
        device = guard.setup_cuda_device()
        mon = guard.OccupancyMonitor(interval=15.0)
        mon.start()
    else:
        device = torch.device("cpu")
    guard.register_self("scu post val")
    t_iso, t0 = guard.now(), time.time()
    sp = Split("val")
    wv = dict(np.load(LFR_LABELS / "windows_val.npz"))
    rows = eval_rows(sp, AUDIT_TOK)
    status = "COMPLETE"
    done = []
    try:
        for key, rd in targets().items():
            outp = od / f"{key}.json"
            if outp.exists():
                continue
            ck = json.loads((guard.ROOT / rd / "status.json").read_text())["ckpts"]["s50"]
            m, _ = load_milestone(ck["path"], device)
            E = m.cfg.n_experts
            r = evaluate(m, sp, rows, device, per_token=True, routing=True)
            meas = r["per_token"]
            res = {"key": key, "run_dir": rd, "ckpt": ck, "time": guard.now(), "val_audit_rows": "eval_rows(val, 200k/domain)",
                   "ce": r["ce"], "ce_dom": r["ce_dom"], "n": r["n"], "routing_entropy": r["ent"], "load": r["load"]}
            for sysn in ("domain", "cluster"):
                lm, K = token_labels(meas, sp, sysn, wv)
                rs = routing_subject(meas["topi"], lm, meas["gdoc"], max(K, int(lm.max()) + 1), E)
                res["routing_" + ("domain" if sysn == "domain" else "e5")] = {
                    k: rs[k] for k in ("mi_mean", "nmi_mean", "nmi_null_route_mean", "nmi_null_doclabel_mean",
                                       "subject_entropy_load_adj_mean", "nmi")}
            guard.atomic_write_json(outp, res, mode=0o400)
            done.append(key)
            print(json.dumps({"key": key, "ce": res["ce"], "nmi_domain": res["routing_domain"]["nmi_mean"],
                              "nmi_e5": res["routing_e5"]["nmi_mean"]}), flush=True)
            del m
            if device.type == "cuda":
                torch.cuda.empty_cache()
    except KeyboardInterrupt:
        status = "EXTERNALLY_STOPPED"
    finally:
        if mon is not None:
            mon.stop()
            if mon.contaminated and status == "COMPLETE":
                status = "EXTERNALLY_STOPPED"
    if a.device == "cuda":
        guard.gpu_ledger(t_iso, guard.now(), "scu_post_val", "scu post val", status)
        with open(ART / "compute_ledger.tsv", "a") as f:
            f.write(f"{t_iso}\t{guard.now()}\t{time.time() - t0:.1f}\tpost_val\tpost_val\tevaluation\t{status}\n")
    print(json.dumps({"status": status, "done": done, "seconds": time.time() - t0}))
    sys.exit(0 if status == "COMPLETE" else 77)


if __name__ == "__main__":
    main()
