#!/usr/bin/env python
"""csr-semantic-edl-v1 evaluation (inference mode: the model receives no encoder, table or label).
Outputs artifacts/semantic-edl-v1/evals/<run_name>/<mode>_<tag>.json (written once, 0400).

  dev      --tag T : CALIB-SEARCH half-B CE (0.4M tok/domain rows; overall only) + LABEL-FREE audit (discovery CS half A,
                     measurement CS half B): systems pseudo (frozen SEM calib table) and e5; FSI for both
  calibsel --tag s50 : only after CALIBSEL_OPEN.json; CALIB-SELECTION CE + label-free audit (measurement CALIB-SEL)
  final   --tags ... [--per-expert] : only after UNSEALED.json; full-VAL CE + per-document CE (fle evaluate), fle audit_one
                     (domain + e5; shuffled-label control at p100) at each tag, pseudo-domain audit at each tag, and
                     per-expert interventions at p100 with domain / e5 / pseudo interactions from the same passes
  supp    --tags ... : reused fle checkpoints: pseudo-domain audit only (domain/e5 results come from fle final.json)"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402

AUDIT_TOK = 200_000
CE_TOK = 400_000
PE_TOK = 50_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["dev", "calibsel", "final", "supp"])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--tags", default="p040,p060,p080,p100")
    ap.add_argument("--per-expert", action="store_true")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    if not guard.require_execution(a.execute, "gpu" if a.device == "cuda" else "cpu-model"):
        return
    import numpy as np
    import torch
    from fle_evaluate import audit_one, load_milestone
    from csr.fsp_lib import Split
    from csr.lfr_audit import token_labels
    from csr.lfr_eval import eval_rows, evaluate, paper_half
    from csr.sem_lib import label_free_audit, per_expert_multi, pseudo_windows
    from csr.sem_paths import (ART, FLE_ART, LFR_LABELS, SEM_CS, SEM_CS_SHA, SEM_SEL, SEM_SEL_SHA, check_prereg)
    check_prereg()
    rd = guard.inside_root(a.run_dir)
    ckpts = json.loads((rd / "status.json").read_text())["ckpts"]
    od = ART / "evals" / a.run_name
    od.mkdir(parents=True, exist_ok=True)
    out_p = od / (f"{a.mode}_{a.tag}.json" if a.mode in ("dev", "calibsel") else f"{a.mode}.json")
    if out_p.exists():
        raise guard.GuardError(f"{out_p} exists (each evaluation runs exactly once)")
    if a.mode == "calibsel" and not (ART / "CALIBSEL_OPEN.json").exists():
        raise guard.GuardError("CALIB-SELECTION is opened only at the freeze step")
    if a.mode in ("final", "supp") and not (ART / "UNSEALED.json").exists():
        raise guard.GuardError("VAL / domain labels only after the freeze (UNSEALED.json)")
    for p, s in ((SEM_CS, SEM_CS_SHA), (SEM_SEL, SEM_SEL_SHA)):
        if guard.sha256_file(p) != s:
            raise guard.GuardError(f"table hash mismatch {p}")
    mon = None
    if a.device == "cuda":
        guard.occupancy_gate(f"sem evaluate {a.mode}")
        device = guard.setup_cuda_device()
        mon = guard.OccupancyMonitor(interval=15.0)
        mon.start()
    else:
        device = torch.device("cpu")
    guard.register_self(f"sem evaluate {a.mode} {a.run_name} {a.tag}")
    t_iso, t0 = guard.now(), time.time()
    cs = Split("calib_search")
    wc = dict(np.load(LFR_LABELS / "windows_calib.npz"))
    disc_rows = eval_rows(cs, AUDIT_TOK, paper_half(cs, 0))
    pw_cs = pseudo_windows(cs, SEM_CS)
    res = {"mode": a.mode, "run": rd.name, "run_name": a.run_name, "time": guard.now(),
           "labels_sha256": {"windows_calib.npz": guard.sha256_file(LFR_LABELS / "windows_calib.npz"),
                             "table_SEM_calib_search": SEM_CS_SHA}}
    status = "COMPLETE"
    try:
        if a.mode in ("dev", "calibsel"):
            res["tag"], res["ckpt"] = a.tag, ckpts[a.tag]
            m, _ = load_milestone(ckpts[a.tag]["path"], device)
            if a.mode == "dev":
                r = evaluate(m, cs, eval_rows(cs, CE_TOK, paper_half(cs, 1)), device, routing=True)
                res["calib_search_B"] = {k: r[k] for k in ("ce", "n", "load", "ent")}
                res["audit"] = label_free_audit(m, cs, disc_rows, cs, eval_rows(cs, AUDIT_TOK, paper_half(cs, 1)),
                                                device, {"pseudo": (pw_cs, pw_cs), "e5": (wc, wc)},
                                                ("pseudo", "e5"))
            else:
                sp = Split("calib_sel")
                ws = dict(np.load(FLE_ART / "labels" / "windows_calib_sel.npz"))
                r = evaluate(m, sp, eval_rows(sp, CE_TOK), device, routing=True)
                res["calib_sel"] = {k: r[k] for k in ("ce", "n", "load", "ent")}
                res["audit"] = label_free_audit(m, cs, disc_rows, sp, eval_rows(sp, AUDIT_TOK), device,
                                                {"pseudo": (pw_cs, pseudo_windows(sp, SEM_SEL)), "e5": (wc, ws)},
                                                ("pseudo", "e5"))
        else:
            sp = Split("val")
            wv = dict(np.load(LFR_LABELS / "windows_val.npz"))
            vt = ART / "tables" / "table_SEM_val.npz"
            res["labels_sha256"].update({"windows_val.npz": guard.sha256_file(LFR_LABELS / "windows_val.npz"),
                                         "table_SEM_val": guard.sha256_file(vt)})
            pw_v = pseudo_windows(sp, vt)
            meas_rows = eval_rows(sp, AUDIT_TOK)
            res["ckpts"] = ckpts
            if a.mode == "final":
                m, _ = load_milestone(ckpts["p100"]["path"], device)
                r = evaluate(m, sp, eval_rows(sp), device, routing=True)
                res["val"] = {k: v for k, v in r.items() if k != "doc_ce"}
                res["val_doc_ce"] = {str(k): v for k, v in r["doc_ce"].items()}
                del m
            res["audit"], res["pseudo_audit"] = {}, {}
            for tag in a.tags.split(","):
                if tag not in ckpts:
                    continue
                mt, _ = load_milestone(ckpts[tag]["path"], device)
                disc = evaluate(mt, cs, disc_rows, device, per_token=True)["per_token"]
                mf = evaluate(mt, sp, meas_rows, device, per_token=True)
                res["pseudo_audit"][tag] = label_free_audit(mt, cs, disc_rows, sp, meas_rows, device,
                                                            {"pseudo": (pw_cs, pw_v)}, ("pseudo",), disc=disc,
                                                            meas_full=mf)
                if a.mode == "final":
                    res["audit"][tag] = audit_one(mt, cs, disc_rows, sp, meas_rows, device, wc, wv, tag == "p100")
                    if a.per_expert and tag == "p100":
                        lab_cs_p = pw_cs["cluster"]
                        lab_v_p = pw_v["cluster"]
                        e5d, Ke = token_labels(disc, cs, "cluster", wc)
                        systems = {"domain": (disc["dom"].astype(np.int64), lambda b: b["dom"], 5),
                                   "pseudo": (lab_cs_p[disc["gdoc"].astype(np.int64)],
                                              lambda b: lab_v_p[b["gdoc"].astype(np.int64)], 17),
                                   "e5": (e5d, lambda b: token_labels(b, sp, "cluster", wv)[0], Ke)}
                        res["per_expert"] = per_expert_multi(mt, cs, disc_rows, sp, eval_rows(sp, PE_TOK), device,
                                                             disc, systems)
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
        guard.gpu_ledger(t_iso, guard.now(), f"sem_eval_{a.mode}_{a.run_name}_{a.tag}", f"sem evaluate {a.mode}", status)
        with open(ART / "compute_ledger.tsv", "a") as f:
            f.write(f"{t_iso}\t{guard.now()}\t{time.time() - t0:.1f}\t{rd.name}\teval_{a.mode}_{a.tag}_{a.run_name}\t"
                    f"evaluation\t{status}\n")
    print(json.dumps({"status": status, "seconds": res["seconds"]}))
    sys.exit(0 if status == "COMPLETE" else 77)


if __name__ == "__main__":
    main()
