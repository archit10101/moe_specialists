#!/usr/bin/env python
"""csr-semantic-utility-edl-v1 evaluation (inference mode: no table, bank, label or encoder enters the model).
Outputs artifacts/semantic-utility-edl-v1/evals/<run_name>/<mode>_<tag>[_<sys>].json (written once, 0400).
Row selection is label-free (documents ordered by sha256('sue-order|'+doc id), capped by tokens).

  dev     CALIB-SEARCH (engineering split): expert subjects discovered on paper-half A, measured on paper-half B
          (paper-disjoint): CE (B, 400k tok), pseudo-domain and e5 routing metrics (MI/NMI, topic prediction, FSI),
          per-expert equal-compute replacement interaction (own pseudo-group vs difficulty-matched other groups,
          document bootstrap) for the pseudo and e5 systems (200k tok), generic importance, load, entropy.
          --pseudo-table P = the calib_search table of the frozen grouping.
  confsel CALIB-SELECTION paper-half 0 (opened once at the freeze; guard CONFSEL_OPEN.json): same metrics, discovery on
          CALIB-SEARCH half A. --pseudo-table (calib_search) --pseudo-table-sel (calib_sel).
  final   (guard FINAL_OPEN.json) CALIB-SELECTION paper-half 1 (never used before): the preregistered final pseudo test;
          plus VAL (descriptive / CE): full-VAL CE overall / per domain / per document, real-domain and e5 routing
          metrics and probes, domain FSI, per-expert replacement interaction for the domain and e5 systems
          (the semantic-edl-long-k 'val --full --pe' measurement, identical library calls) and inference throughput."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402

AUDIT_TOK = 200_000
CE_TOK = 400_000
PE_TOK = 200_000


def lf_rows(split, docs, n_tokens):
    from csr.fsp_lib import pack_rows
    order = sorted(docs, key=lambda i: hashlib.sha256(f"sue-order|{split.ids[i]}".encode()).hexdigest())
    out, used = [], 0
    for i in order:
        if used >= n_tokens:
            break
        out.append(int(i))
        used += int(split.lengths[i]) - 1
    return pack_rows(split.lengths, out)


def pseudo_block(m, cs, disc_rows, sm, ce_rows, au_rows, pe_rows, device, wd_p, wm_p, wd_e, wm_e, disc, e5_rows=None):
    """e5_rows=None (dev, CALIB-SEARCH: every document has e5 windows): both systems on the same rows. e5_rows =
    (au_rows_e5, pe_rows_e5) (CALIB-SELECTION: e5 windows exist only for a 1,010-document subset): the preregistered
    pseudo measurement stays on the label-free rows; the descriptive e5 measurement runs on the e5-labelled documents
    of the same paper-half (fix of 2026-11-17, DEVIATIONS)."""
    import numpy as np
    from csr.lfr_audit import token_labels
    from csr.lfr_eval import evaluate
    from csr.sem_lib import label_free_audit
    from csr.slk_lib import per_expert_boot
    res = {}
    r = evaluate(m, sm, ce_rows, device, routing=True)
    res["ce_block"] = {k: r[k] for k in ("ce", "n", "load", "ent")}
    lp, Kp = token_labels(disc, cs, "cluster", wd_p)
    le, Ke = token_labels(disc, cs, "cluster", wd_e)
    sys_p = {"pseudo": (lp, lambda b: token_labels(b, sm, "cluster", wm_p)[0], Kp)}
    sys_e = {"e5": (le, lambda b: token_labels(b, sm, "cluster", wm_e)[0], Ke)}
    if e5_rows is None:
        res["audit"] = label_free_audit(m, cs, disc_rows, sm, au_rows, device,
                                        {"pseudo": (wd_p, wm_p), "e5": (wd_e, wm_e)}, ("pseudo",), disc=disc)
        pe = per_expert_boot(m, cs, disc_rows, sm, pe_rows, device, disc, {**sys_p, **sys_e})
    else:
        res["audit"] = label_free_audit(m, cs, disc_rows, sm, au_rows, device, {"pseudo": (wd_p, wm_p)}, ("pseudo",),
                                        disc=disc)
        res["audit"]["e5"] = label_free_audit(m, cs, disc_rows, sm, e5_rows[0], device, {"e5": (wd_e, wm_e)}, (),
                                              disc=disc)["e5"]
        pe = per_expert_boot(m, cs, disc_rows, sm, pe_rows, device, disc, sys_p)
        pe_e = per_expert_boot(m, cs, disc_rows, sm, e5_rows[1], device, disc, sys_e)
        pe["e5"] = pe_e["e5"]
        pe["e5_rows"] = {"note": "e5-labelled documents of the same paper-half", "n_docs": pe_e.get("n_docs"),
                         "n_tokens": pe_e.get("n_tokens")}
    for k_ in ("pseudo", "e5"):
        pe[k_].pop("boot_mean_samples_full", None)
    res["per_expert"] = pe
    ld = np.asarray(r["load"])
    E = ld.shape[1]
    res["pathology"] = {"load_min_x_uniform": float(ld.min() * E), "load_max_x_uniform": float(ld.max() * E),
                        "n_below_0.1_uniform": int((ld < 0.1 / E).sum()), "routing_entropy": r["ent"]}
    return res


def throughput(m, device, n_rows=64, reps=5):
    import torch
    ids = torch.randint(2, 50000, (n_rows, 1024), device=device, generator=torch.Generator(device=device).manual_seed(0))
    mi = {"input_ids": ids, "doc_ids": torch.zeros_like(ids), "positions": torch.arange(1024, device=device).repeat(n_rows, 1)}
    ts = []
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for i in range(reps + 1):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            m(mi, {})
            if device.type == "cuda":
                torch.cuda.synchronize()
            if i:
                ts.append(time.time() - t0)
    return {"rows": n_rows, "tokens_per_s_median": float(n_rows * 1024 / sorted(ts)[len(ts) // 2]),
            "latency_s_median": float(sorted(ts)[len(ts) // 2])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["dev", "confsel", "final"])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--pseudo-table", required=True)
    ap.add_argument("--pseudo-table-sel", default="")
    ap.add_argument("--val-pe", type=int, default=100_000)
    ap.add_argument("--skip-val", action="store_true", help="reused runs whose VAL metrics already exist")
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
    from csr.sem_lib import pseudo_windows
    from csr.slk_lib import per_expert_boot, probe_logreg, routing_subject
    from csr.sue_paths import ART, FLE_ART, LFR_LABELS, check_prereg
    check_prereg()
    rd = guard.inside_root(a.run_dir)
    ckpts = json.loads((rd / "status.json").read_text())["ckpts"]
    if a.tag not in ckpts:
        raise guard.GuardError(f"{a.run_name}: no checkpoint {a.tag}")
    od = ART / "evals" / a.run_name
    od.mkdir(parents=True, exist_ok=True)
    out_p = od / f"{a.mode}_{a.tag}.json"
    if out_p.exists():
        raise guard.GuardError(f"{out_p} exists (each evaluation runs exactly once)")
    if a.mode == "confsel" and not (ART / "CONFSEL_OPEN.json").exists():
        raise guard.GuardError("CALIB-SELECTION half 0 opens only at the freeze (CONFSEL_OPEN.json)")
    if a.mode == "final" and not (ART / "FINAL_OPEN.json").exists():
        raise guard.GuardError("final evaluation opens only after all final training (FINAL_OPEN.json)")
    mon = None
    if a.device == "cuda":
        guard.occupancy_gate(f"sue evaluate {a.mode}")
        device = guard.setup_cuda_device()
        mon = guard.OccupancyMonitor(interval=15.0)
        mon.start()
    else:
        device = torch.device("cpu")
    guard.register_self(f"sue evaluate {a.mode} {a.run_name} {a.tag}")
    t_iso, t0 = guard.now(), time.time()
    cs = Split("calib_search")
    wc = dict(np.load(LFR_LABELS / "windows_calib.npz"))
    pt_cs = guard.inside_root(a.pseudo_table)
    wp_cs = pseudo_windows(cs, pt_cs)
    hA, hB = paper_half(cs, 0), paper_half(cs, 1)
    disc_rows = lf_rows(cs, hA, AUDIT_TOK)
    res = {"mode": a.mode, "run": rd.name, "run_name": a.run_name, "tag": a.tag, "ckpt": ckpts[a.tag],
           "time": guard.now(), "pseudo_table": {"path": str(pt_cs.relative_to(guard.ROOT)),
                                                 "sha256": guard.sha256_file(pt_cs)},
           "rows": "label-free sha256 order; discovery CALIB-SEARCH half A"}
    status = "COMPLETE"
    try:
        m, _ = load_milestone(ckpts[a.tag]["path"], device)
        disc = evaluate(m, cs, disc_rows, device, per_token=True)["per_token"]
        if a.mode == "dev":
            res["block"] = "calib_search_B"
            res.update(pseudo_block(m, cs, disc_rows, cs, lf_rows(cs, hB, CE_TOK), lf_rows(cs, hB, AUDIT_TOK),
                                    lf_rows(cs, hB, PE_TOK), device, wp_cs, wp_cs, wc, wc, disc))
        else:
            sp = Split("calib_sel")
            ws = dict(np.load(FLE_ART / "labels" / "windows_calib_sel.npz"))
            pt_sel = guard.inside_root(a.pseudo_table_sel)
            res["pseudo_table_sel"] = {"path": str(pt_sel.relative_to(guard.ROOT)), "sha256": guard.sha256_file(pt_sel)}
            wp_sel = pseudo_windows(sp, pt_sel)
            half = paper_half(sp, 0 if a.mode == "confsel" else 1)
            res["block"] = f"calib_sel_half{0 if a.mode == 'confsel' else 1}"
            wdocs = set(np.unique(np.asarray(ws["doc"])).tolist())
            half_e5 = [i for i in half if int(i) in wdocs]
            res["e5_block_docs"] = len(half_e5)
            res.update(pseudo_block(m, cs, disc_rows, sp, lf_rows(sp, half, CE_TOK), lf_rows(sp, half, AUDIT_TOK),
                                    lf_rows(sp, half, PE_TOK), device, wp_cs, wp_sel, wc, ws, disc,
                                    e5_rows=(lf_rows(sp, half_e5, AUDIT_TOK), lf_rows(sp, half_e5, PE_TOK))))
        if a.mode == "final" and not a.skip_val:
            # VAL: identical measurement calls to semantic-edl-long-k 'val --full --pe' (descriptive + CE)
            E = m.cfg.n_experts
            sp = Split("val")
            wv = dict(np.load(LFR_LABELS / "windows_val.npz"))
            drows = eval_rows(cs, AUDIT_TOK, paper_half(cs, 0))
            dsc = evaluate(m, cs, drows, device, per_token=True)["per_token"]
            r = evaluate(m, sp, eval_rows(sp), device, routing=True)
            res["val"] = {k: v for k, v in r.items() if k != "doc_ce"}
            res["val_doc_ce"] = {str(k): v for k, v in r["doc_ce"].items()}
            cs_b = evaluate(m, cs, eval_rows(cs, AUDIT_TOK, paper_half(cs, 1)), device, per_token=True)["per_token"]
            meas_rows = eval_rows(sp, AUDIT_TOK)
            mf = evaluate(m, sp, meas_rows, device, per_token=True)
            meas = mf["per_token"]
            res["val_meas"] = {"ce": mf["ce"], "ce_dom": mf["ce_dom"], "load": mf["load"], "routing_entropy": mf["ent"],
                               "n": mf["n"]}
            sysd = {}
            for sysn in ("domain", "cluster"):
                ld, K = token_labels(dsc, cs, sysn, wc)
                lb, _ = token_labels(cs_b, cs, sysn, wc)
                lm, _ = token_labels(meas, sp, sysn, wv)
                K = max(K, int(lm.max()) + 1, int(ld.max()) + 1)
                rs = routing_subject(meas["topi"], lm, meas["gdoc"], K, E)
                from slk_evaluate import windows_feats
                (Htr1, ytr1), _ = windows_feats(dsc, E, ld, K)
                (Htr2, ytr2), _ = windows_feats(cs_b, E, lb, K)
                (Hte, yte), _ = windows_feats(meas, E, lm, K)
                Htr, ytr = np.concatenate([Htr1, Htr2]), np.concatenate([ytr1, ytr2])
                rs["probe_logreg"] = probe_logreg(Htr, ytr, Hte, yte, K)
                rs["probe_centroid_acc"] = topic_prediction_acc(Htr, ytr, Hte, yte, K)
                groups = assign_topics(dsc["topi"], ld, K, E)
                fs = functional(m, sp, meas_rows, device, meas, lm, K, groups)
                rs["fsi"], rs["fsi_raw"] = fs["fsi"], fs["fsi_raw"]
                lds = shuffled_labels(dsc, ld, sysn, wc)
                lms = shuffled_labels(meas, lm, sysn, wv)
                rs["fsi_shuffled_labels"] = functional(m, sp, meas_rows, device, meas, lms, K,
                                                       assign_topics(dsc["topi"], lds, K, E))["fsi"]
                rs["K"] = K
                sysd[sysn] = rs
            res["val_routing"] = sysd
            from slk_evaluate import coherence
            res["val_coherence_e5"], res["val_coherence_coverage"] = coherence(meas, wv, E)
            e5d, Ke = token_labels(dsc, cs, "cluster", wc)
            systems = {"domain": (dsc["dom"].astype(np.int64), lambda b: b["dom"], 5),
                       "e5": (e5d, lambda b: token_labels(b, sp, "cluster", wv)[0], Ke)}
            res["val_per_expert"] = per_expert_boot(m, cs, drows, sp, eval_rows(sp, a.val_pe), device, dsc, systems)
            res["throughput"] = throughput(m, device)
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
        guard.gpu_ledger(t_iso, guard.now(), f"sue_eval_{a.mode}_{a.run_name}_{a.tag}", f"sue evaluate {a.mode}", status)
        with open(ART / "compute_ledger.tsv", "a") as f:
            f.write(f"{t_iso}\t{guard.now()}\t{time.time() - t0:.1f}\t{rd.name}\teval_{a.mode}_{a.tag}_"
                    f"{a.run_name}\tevaluation\t{status}\n")
    print(json.dumps({"status": status, "seconds": res["seconds"], "out": str(out_p)}))
    sys.exit(0 if status == "COMPLETE" else 77)


if __name__ == "__main__":
    main()
