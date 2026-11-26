#!/usr/bin/env python
"""csr-semantic-edl-long-adaptive-k-v1 evaluation (inference mode: no table, label or encoder enters the model).
Outputs artifacts/semantic-edl-long-k/evals/<run_name>/<mode>_<tag>.json (written once, 0400).

  val  --tag T [--full] [--pe TOK]   PHASE-1 ONLY, after PHASE1_VAL_OPEN.json (all Phase-1 configs fixed by the prereg,
        all Phase-1 training complete). Full-VAL CE (overall, per domain, per document); routing subject metrics on the
        VAL audit rows (domain and label-free e5 systems; MI/NMI with load-preserving and doc-label nulls; load-
        adjusted per-expert subject entropy; profiles); routing probes fitted on CALIB-SEARCH windows and evaluated on
        VAL windows (logistic regression + nearest centroid); domain FSI (expert groups from CALIB-SEARCH half A);
        within-expert e5 semantic coherence. --full adds e5 FSI and the shuffled-domain-label FSI control. --pe TOK runs
        per-expert replacement interventions (expert subject from CALIB-SEARCH half A; measured on VAL, TOK tokens per
        domain) with document-bootstrap CIs.
  dev  --tag T --pseudo-table P      label-free CALIB-SEARCH (Phase 2): CE on half B; pseudo (P = calib-search table of
        that K) and e5 systems: routing MI, FSI (groups from half A, measured on half B). Never reads domain labels.
  calibsel --tag T --pseudo-table-cs P --pseudo-table-sel P2   after CALIBSEL_OPEN.json (Phase 2 frozen selection)."""
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


def windows_feats(per_token, E, lab, K):
    import numpy as np
    from csr.lfr_audit import window_hist
    wk = per_token["gdoc"].astype(np.int64) * 10000 + per_token["pos"].astype(np.int64) // 256
    return window_hist(per_token, per_token["topi"], E, lab, wk, K), wk


def coherence(meas, wm, E):
    import numpy as np
    from csr.lfr_audit import within_expert_similarity
    emb = wm["emb"].astype(np.float32)
    emb /= np.maximum(1e-9, np.linalg.norm(emb, axis=1, keepdims=True))
    key = wm["doc"].astype(np.int64) * 10000 + wm["start"].astype(np.int64) // 256
    pos_of = {int(k_): i for i, k_ in enumerate(key)}
    wmk = meas["gdoc"].astype(np.int64) * 10000 + meas["pos"].astype(np.int64) // 256
    sel = np.asarray([pos_of.get(int(k_), -1) for k_ in wmk])
    ok = sel >= 0
    Lr = meas["topi"].shape[1]
    cnt = np.zeros((len(emb), Lr * E), np.float32)
    for li in range(Lr):
        for kk in range(meas["topi"].shape[2]):
            np.add.at(cnt, (sel[ok], li * E + meas["topi"][ok, li, kk].astype(np.int64)), 1.0)
    return within_expert_similarity(cnt, emb), float(ok.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["val", "dev", "calibsel"])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--pe", type=int, default=0)
    ap.add_argument("--pseudo-table", default="")
    ap.add_argument("--pseudo-table-sel", default="")
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
    from csr.lfr_audit import assign_topics, functional, shuffled_labels, token_labels, topic_prediction_acc
    from csr.lfr_eval import eval_rows, evaluate, paper_half
    from csr.sem_lib import label_free_audit, pseudo_windows
    from csr.slk_lib import per_expert_boot, probe_logreg, routing_subject
    from csr.slk_paths import ART, LFR_LABELS, check_prereg
    check_prereg()
    rd = guard.inside_root(a.run_dir)
    ckpts = json.loads((rd / "status.json").read_text())["ckpts"]
    if a.tag not in ckpts:
        raise guard.GuardError(f"{a.run_name}: no checkpoint {a.tag}")
    od = ART / "evals" / a.run_name
    od.mkdir(parents=True, exist_ok=True)
    suffix = ("_full" if a.full else "") + (f"_pe{a.pe // 1000}k" if a.pe else "")
    out_p = od / f"{a.mode}_{a.tag}{suffix}.json"
    if out_p.exists():
        raise guard.GuardError(f"{out_p} exists (each evaluation runs exactly once)")
    if a.mode == "val" and not (ART / "PHASE1_VAL_OPEN.json").exists():
        raise guard.GuardError("VAL is opened only after all Phase-1 training is complete (PHASE1_VAL_OPEN.json)")
    if a.mode == "calibsel" and not (ART / "CALIBSEL_OPEN.json").exists():
        raise guard.GuardError("CALIB-SELECTION is opened only at the Phase-2 freeze (CALIBSEL_OPEN.json)")
    mon = None
    if a.device == "cuda":
        guard.occupancy_gate(f"slk evaluate {a.mode}")
        device = guard.setup_cuda_device()
        mon = guard.OccupancyMonitor(interval=15.0)
        mon.start()
    else:
        device = torch.device("cpu")
    guard.register_self(f"slk evaluate {a.mode} {a.run_name} {a.tag}")
    t_iso, t0 = guard.now(), time.time()
    cs = Split("calib_search")
    wc = dict(np.load(LFR_LABELS / "windows_calib.npz"))
    disc_rows = eval_rows(cs, AUDIT_TOK, paper_half(cs, 0))
    res = {"mode": a.mode, "run": rd.name, "run_name": a.run_name, "tag": a.tag, "ckpt": ckpts[a.tag],
           "time": guard.now(), "full": a.full, "pe_tokens_per_domain": a.pe}
    status = "COMPLETE"
    try:
        m, _ = load_milestone(ckpts[a.tag]["path"], device)
        E = m.cfg.n_experts
        if a.mode == "val":
            sp = Split("val")
            wv = dict(np.load(LFR_LABELS / "windows_val.npz"))
            r = evaluate(m, sp, eval_rows(sp), device, routing=True)
            res["val"] = {k: v for k, v in r.items() if k != "doc_ce"}
            res["val_doc_ce"] = {str(k): v for k, v in r["doc_ce"].items()}
            disc = evaluate(m, cs, disc_rows, device, per_token=True)["per_token"]
            cs_b = evaluate(m, cs, eval_rows(cs, AUDIT_TOK, paper_half(cs, 1)), device, per_token=True)["per_token"]
            meas_rows = eval_rows(sp, AUDIT_TOK)
            mf = evaluate(m, sp, meas_rows, device, per_token=True)
            meas = mf["per_token"]
            res["meas"] = {"ce": mf["ce"], "ce_dom": mf["ce_dom"], "load": mf["load"], "routing_entropy": mf["ent"],
                           "n": mf["n"]}
            sysd = {}
            for sysn in ("domain", "cluster"):
                ld, K = token_labels(disc, cs, sysn, wc)
                lb, _ = token_labels(cs_b, cs, sysn, wc)
                lm, _ = token_labels(meas, sp, sysn, wv)
                K = max(K, int(lm.max()) + 1, int(ld.max()) + 1)
                rs = routing_subject(meas["topi"], lm, meas["gdoc"], K, E)
                (Htr1, ytr1), _ = windows_feats(disc, E, ld, K)
                (Htr2, ytr2), _ = windows_feats(cs_b, E, lb, K)
                (Hte, yte), _ = windows_feats(meas, E, lm, K)
                Htr, ytr = np.concatenate([Htr1, Htr2]), np.concatenate([ytr1, ytr2])
                rs["probe_logreg"] = probe_logreg(Htr, ytr, Hte, yte, K)
                rs["probe_centroid_acc"] = topic_prediction_acc(Htr, ytr, Hte, yte, K)
                if sysn == "domain" or a.full:
                    groups = assign_topics(disc["topi"], ld, K, E)
                    fs = functional(m, sp, meas_rows, device, meas, lm, K, groups)
                    rs["fsi"], rs["fsi_raw"], rs["fsi_detail"] = fs["fsi"], fs["fsi_raw"], fs
                    if a.full:
                        lds = shuffled_labels(disc, ld, sysn, wc)
                        lms = shuffled_labels(meas, lm, sysn, wv)
                        fss = functional(m, sp, meas_rows, device, meas, lms, K, assign_topics(disc["topi"], lds, K, E))
                        rs["fsi_shuffled_labels"] = fss["fsi"]
                rs["K"] = K
                sysd[sysn] = rs
            res["routing"] = sysd
            res["coherence_e5"], res["coherence_coverage"] = coherence(meas, wv, E)
            if a.pe:
                e5d, Ke = token_labels(disc, cs, "cluster", wc)
                systems = {"domain": (disc["dom"].astype(np.int64), lambda b: b["dom"], 5),
                           "e5": (e5d, lambda b: token_labels(b, sp, "cluster", wv)[0], Ke)}
                res["per_expert"] = per_expert_boot(m, cs, disc_rows, sp, eval_rows(sp, a.pe), device, disc, systems)
        elif a.mode == "dev":
            pt_cs = guard.inside_root(a.pseudo_table)
            res["pseudo_table"] = {"path": str(pt_cs.relative_to(guard.ROOT)), "sha256": guard.sha256_file(pt_cs)}
            r = evaluate(m, cs, eval_rows(cs, CE_TOK, paper_half(cs, 1)), device, routing=True)
            res["calib_search_B"] = {k: r[k] for k in ("ce", "n", "load", "ent")}
            pw = pseudo_windows(cs, pt_cs)
            res["audit"] = label_free_audit(m, cs, disc_rows, cs, eval_rows(cs, AUDIT_TOK, paper_half(cs, 1)), device,
                                            {"pseudo": (pw, pw), "e5": (wc, wc)}, ("pseudo", "e5"))
        else:
            from csr.slk_paths import FLE_ART
            sp = Split("calib_sel")
            ws = dict(np.load(FLE_ART / "labels" / "windows_calib_sel.npz"))
            pt_cs, pt_sel = guard.inside_root(a.pseudo_table), guard.inside_root(a.pseudo_table_sel)
            res["pseudo_table"] = {"cs": guard.sha256_file(pt_cs), "sel": guard.sha256_file(pt_sel)}
            r = evaluate(m, sp, eval_rows(sp, CE_TOK), device, routing=True)
            res["calib_sel"] = {k: r[k] for k in ("ce", "n", "load", "ent")}
            res["audit"] = label_free_audit(m, cs, disc_rows, sp, eval_rows(sp, AUDIT_TOK), device,
                                            {"pseudo": (pseudo_windows(cs, pt_cs), pseudo_windows(sp, pt_sel)),
                                             "e5": (wc, ws)}, ("pseudo", "e5"))
    except KeyboardInterrupt:
        status = "EXTERNALLY_STOPPED"
    finally:
        if mon is not None:
            mon.stop()
            if mon.contaminated and status == "COMPLETE":
                status = "EXTERNALLY_STOPPED"
    res["seconds"] = time.time() - t0
    res["peak_vram_gib"] = (torch.cuda.max_memory_allocated() / 2**30) if a.device == "cuda" else 0.0
    if status == "COMPLETE":
        guard.atomic_write_json(out_p, res, mode=0o400)
    if a.device == "cuda":
        guard.gpu_ledger(t_iso, guard.now(), f"slk_eval_{a.mode}_{a.run_name}_{a.tag}", f"slk evaluate {a.mode}", status)
        with open(ART / "compute_ledger.tsv", "a") as f:
            f.write(f"{t_iso}\t{guard.now()}\t{time.time() - t0:.1f}\t{rd.name}\teval_{a.mode}_{a.tag}{suffix}_"
                    f"{a.run_name}\tevaluation\t{status}\n")
    print(json.dumps({"status": status, "seconds": res["seconds"], "out": str(out_p)}))
    sys.exit(0 if status == "COMPLETE" else 77)


if __name__ == "__main__":
    main()
