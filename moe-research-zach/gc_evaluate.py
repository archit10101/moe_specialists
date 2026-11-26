#!/usr/bin/env python
"""csr-gradient-consistent-latent-edl-v1 evaluation (inference mode: the model receives no scout, encoder, prototypes
or q). Outputs artifacts/gc-edl-v1/evals/<run_name>/<mode>_<tag>.json (written once, 0400).

  dev      --run-dir R --run-name N --tag T : CALIB-SEARCH half-B CE (0.4M tok/domain rows, overall only) + LABEL-FREE
                                              audit (discovery CS half A, measurement CS half B): e5-cluster system and
                                              pseudo-domain systems (SEM/GRAD/COMBINED tables); routing metrics, FSI,
                                              topic prediction, load/starvation/entropy. No domain metric.
  calibsel --run-dir R --run-name N --tag s50: only after CALIBSEL_OPEN.json; CALIB-SELECTION CE + label-free audit
                                              (measurement = CALIB-SELECTION).
Reuses fle/lfr functions unchanged (lfr_eval.evaluate, eval_rows, paper_half; lfr_audit metrics)."""
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


def pseudo_windows(split, table_npz):
    """windows-format labels (one window per document at start 0) = argmax of the paper's q; K = unassigned."""
    import numpy as np
    from csr.gc_lib import opaque_id
    z = np.load(table_npz)
    idx = {str(o): i for i, o in enumerate(z["opaque"])}
    q = z["q"]
    K = q.shape[1]
    lab = np.full(split.n_docs, K, np.int64)
    cache = {}
    for d, p in enumerate(split.papers):
        o = cache.setdefault(p, opaque_id(p))
        i = idx.get(o)
        if i is not None and q[i].sum() > 0:
            lab[d] = int(q[i].argmax())
    return {"doc": np.arange(split.n_docs), "start": np.zeros(split.n_docs, np.int64), "cluster": lab,
            "K_assigned": K}


def label_free_audit(model, sd, rows_d, sm, rows_m, device, systems, fsi_systems):
    """systems: {name: (windows_disc, windows_meas)}. Label-free (no 'dom' field is read)."""
    import numpy as np
    from csr.lfr_audit import (assign_topics, functional, routing_metrics, token_labels, topic_prediction_acc,
                               window_hist)
    from csr.lfr_eval import evaluate
    E = model.cfg.n_experts
    disc = evaluate(model, sd, rows_d, device, per_token=True)["per_token"]
    mf = evaluate(model, sm, rows_m, device, per_token=True)
    meas = mf["per_token"]
    out = {"ce_meas": mf["ce"], "load": mf["load"], "routing_entropy": mf["ent"]}
    ld_ = np.asarray(mf["load"])
    out["load_min_x_uniform"] = float(ld_.min() * E)
    out["load_max_x_uniform"] = float(ld_.max() * E)
    out["n_below_0.1_uniform"] = int((ld_ < 0.1 / E).sum())
    for name, (wd, wm) in systems.items():
        ld, K = token_labels(disc, sd, "cluster", wd)
        lm, K2 = token_labels(meas, sm, "cluster", wm)
        K = max(K, K2)
        rm = routing_metrics(meas["topi"], lm, K, E)
        res = {"K": K, "routing": {k: v for k, v in rm.items() if k not in ("topic_entropy_per_expert", "profiles")},
               "profiles": rm["profiles"]}
        wdk = disc["gdoc"].astype(np.int64) * 10000 + disc["pos"].astype(np.int64) // 256
        wmk = meas["gdoc"].astype(np.int64) * 10000 + meas["pos"].astype(np.int64) // 256
        Hd, yd = window_hist(disc, disc["topi"], E, ld, wdk, K)
        Hm, ym = window_hist(meas, meas["topi"], E, lm, wmk, K)
        res["topic_pred_acc"] = topic_prediction_acc(Hd, yd, Hm, ym, K)
        res["topic_pred_chance"] = float(np.bincount(ym, minlength=K).max() / len(ym))
        if name in fsi_systems:
            groups = assign_topics(disc["topi"], ld, K, E)
            fs = functional(model, sm, rows_m, device, meas, lm, K, groups)
            res.update({"fsi": fs["fsi"], "fsi_raw": fs["fsi_raw"], "functional": fs})
        out[name] = res
    return out


def systems_for(split_m, split_name_m, cs, wc, wm_e5):
    from csr.gc_paths import LAT
    sysd = {"e5": (wc, wm_e5)}
    for rep in ("SEM", "GRAD", "COMBINED"):
        tc = LAT / f"table_{rep}_calib_search.npz"
        tm = LAT / f"table_{rep}_{split_name_m}.npz"
        if tc.exists() and tm.exists():
            sysd[f"pseudo_{rep}"] = (pseudo_windows(cs, tc), pseudo_windows(split_m, tm))
    return sysd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["dev", "calibsel"])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--tag", required=True)
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
    from csr.gc_paths import ART, FLE_ART, LFR_LABELS, check_prereg
    from csr.lfr_eval import eval_rows, evaluate, paper_half
    check_prereg()
    rd = guard.inside_root(a.run_dir)
    st = json.loads((rd / "status.json").read_text())
    ckpts = st["ckpts"]
    od = ART / "evals" / a.run_name
    od.mkdir(parents=True, exist_ok=True)
    out_p = od / f"{a.mode}_{a.tag}.json"
    if out_p.exists():
        raise guard.GuardError(f"{out_p} exists (each evaluation runs exactly once)")
    if a.mode == "calibsel" and not (ART / "CALIBSEL_OPEN.json").exists():
        raise guard.GuardError("CALIB-SELECTION is opened only at the freeze step")
    mon = None
    if a.device == "cuda":
        guard.occupancy_gate(f"gc evaluate {a.mode}")
        device = guard.setup_cuda_device()
        mon = guard.OccupancyMonitor(interval=15.0)
        mon.start()
    else:
        device = torch.device("cpu")
    guard.register_self(f"gc evaluate {a.mode} {a.run_name} {a.tag}")
    t_iso, t0 = guard.now(), time.time()
    wc = dict(np.load(LFR_LABELS / "windows_calib.npz"))
    cs = Split("calib_search")
    disc_rows = eval_rows(cs, AUDIT_TOK, paper_half(cs, 0))
    res = {"mode": a.mode, "run": rd.name, "run_name": a.run_name, "tag": a.tag, "time": guard.now(),
           "ckpt": ckpts[a.tag], "labels_sha256": {"windows_calib.npz": guard.sha256_file(LFR_LABELS /
                                                                                          "windows_calib.npz")}}
    status = "COMPLETE"
    try:
        m, _ = load_milestone(ckpts[a.tag]["path"], device)
        if a.mode == "dev":
            r = evaluate(m, cs, eval_rows(cs, CE_TOK, paper_half(cs, 1)), device, routing=True)
            res["calib_search_B"] = {k: v for k, v in r.items() if k in ("ce", "n", "load", "ent")}
            res["audit"] = label_free_audit(m, cs, disc_rows, cs, eval_rows(cs, AUDIT_TOK, paper_half(cs, 1)), device,
                                            systems_for(cs, "calib_search", cs, wc, wc),
                                            ("e5", "pseudo_SEM", "pseudo_GRAD", "pseudo_COMBINED"))
        else:
            sp = Split("calib_sel")
            ws = dict(np.load(FLE_ART / "labels" / "windows_calib_sel.npz"))
            res["labels_sha256"]["windows_calib_sel.npz"] = guard.sha256_file(FLE_ART / "labels" /
                                                                              "windows_calib_sel.npz")
            r = evaluate(m, sp, eval_rows(sp, CE_TOK), device, routing=True)
            res["calib_sel"] = {k: v for k, v in r.items() if k in ("ce", "n", "load", "ent")}
            res["audit"] = label_free_audit(m, cs, disc_rows, sp, eval_rows(sp, AUDIT_TOK), device,
                                            systems_for(sp, "calib_sel", cs, wc, ws),
                                            ("e5", "pseudo_SEM", "pseudo_GRAD", "pseudo_COMBINED"))
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
        guard.gpu_ledger(t_iso, guard.now(), f"gc_eval_{a.mode}_{a.run_name}_{a.tag}", f"gc evaluate {a.mode}",
                         status)
        with open(ART / "compute_ledger.tsv", "a") as f:
            f.write(f"{t_iso}\t{guard.now()}\t{time.time() - t0:.1f}\t{rd.name}\teval_{a.mode}_{a.tag}_{a.run_name}\t"
                    f"evaluation\t{status}\n")
    print(json.dumps({"status": status, "seconds": res["seconds"]}))
    sys.exit(0 if status == "COMPLETE" else 77)


if __name__ == "__main__":
    main()
