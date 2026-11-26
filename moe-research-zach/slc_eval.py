#!/usr/bin/env python
"""semantic-lift-245m-confirm-v1 frozen evaluation of ONE checkpoint (prereg sections 5-8). Runs only after
FINAL_OPEN.json exists (all eight E/F runs complete and frozen). Real domain labels are evaluation annotations only.

  1. accuracy: full-VAL CE (overall, per domain, per-document sums/counts for the hierarchical bootstrap)
  2. routing (VAL audit rows, 200k tokens/domain): real-domain routing MI / NMI (+ nulls), e5-cluster NMI and
     cluster-to-expert lift, domain-conditioned expert-selection distributions, mean pairwise top-4 distribution
     overlap (overall and by layer), domain-to-expert lift profiles (L, E, 5), routing-specialised cells
     (VAL routing lift >= 1.5), load / normalised routing entropy / under-used experts
  3. function (slc_lib.per_expert_slc): discovery = CALIB-SEARCH paper-half A (200k tokens/domain), measurement =
     VAL 100k tokens/domain; ordinary (existing) interaction + exposure-conditioned interaction, document bootstrap
     (1,000 replicates, fixed seed, identical rows => replicate b is the same document resample for every model)
Writes artifacts/semantic-lift-245m-confirm-v1/evals/<key>_<tag>.json and .npz (0400, once).
  slc_eval.py --key E_s0 --tag p100 --device cuda --execute"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402

AUDIT_TOK = 200_000
PE_TOK = 100_000
UNDER = 0.1  # "under-used": VAL load < 0.1 x uniform (the registered pathology threshold)


def ckpt_of(key, tag):
    from csr.slc_paths import ART, REUSED
    if key in REUSED:
        rd = REUSED[key]
    else:
        rd = json.loads((ART / "run_dirs.json").read_text())[key]
    st = json.loads((guard.ROOT / rd / "status.json").read_text())
    return rd, st["ckpts"][tag]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--tag", required=True, choices=["s50", "p100"])
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--rows-limit", type=int, default=0, help="test only (CPU): cap rows of every row set")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    if not guard.require_execution(a.execute, "gpu" if a.device == "cuda" else "cpu-model"):
        return
    import numpy as np
    import torch
    from fle_evaluate import load_milestone, route_dists
    from csr.fsp_lib import Split
    from csr.lfr_audit import token_labels
    from csr.lfr_eval import eval_rows, evaluate, paper_half
    from csr.slc_lib import per_expert_slc
    from csr.slc_paths import ART, LFR_LABELS, check_prereg
    from csr.slk_lib import joint, routing_subject
    check_prereg()
    test_mode = a.rows_limit > 0
    if not test_mode and not (ART / "FINAL_OPEN.json").exists():
        raise guard.GuardError("evaluation only after FINAL_OPEN.json (all eight runs complete and frozen)")
    od = Path(a.out_dir) if a.out_dir else ART / "evals"
    od.mkdir(parents=True, exist_ok=True)
    outj, outn = od / f"{a.key}_{a.tag}.json", od / f"{a.key}_{a.tag}.npz"
    if outj.exists():
        print("exists", outj)
        return
    mon = None
    if a.device == "cuda":
        guard.occupancy_gate("slc eval")
        device = guard.setup_cuda_device()
        mon = guard.OccupancyMonitor(interval=15.0)
        mon.start()
    else:
        device = torch.device("cpu")
    guard.register_self(f"slc eval {a.key} {a.tag}")
    t_iso, t0 = guard.now(), time.time()
    status = "COMPLETE"
    lim = ((lambda r: [r[i] for i in sorted(set(np.linspace(0, len(r) - 1, a.rows_limit).astype(int).tolist()))])
           if test_mode else (lambda r: r))
    try:
        rd, ck = ckpt_of(a.key, a.tag)
        if not test_mode:
            h = guard.sha256_file(guard.ROOT / ck["path"] / "state.pt")
            if h != ck["sha256"]:
                raise guard.GuardError(f"checkpoint hash mismatch {ck['path']}")
        m, _ = load_milestone(ck["path"], device)
        E, Lm = m.cfg.n_experts, m.n_moe
        sp, cs = Split("val"), Split("calib_search")
        wv = dict(np.load(LFR_LABELS / "windows_val.npz"))
        res = {"key": a.key, "tag": a.tag, "run_dir": rd, "ckpt": ck, "time": guard.now(), "test_mode": test_mode,
               "pe_tokens_per_domain": PE_TOK, "audit_tokens_per_domain": AUDIT_TOK}
        times = {}
        # 1. accuracy
        t1 = time.time()
        r = evaluate(m, sp, lim(eval_rows(sp)), device, routing=True)
        gd = np.asarray(sorted(r["doc_ce"]), np.int64)
        dsum = np.asarray([r["doc_ce"][g][0] for g in gd])
        dn = np.asarray([r["doc_ce"][g][1] for g in gd], np.int64)
        res["val"] = {"ce": r["ce"], "n": r["n"], "ce_dom": r["ce_dom"], "n_dom": r["n_dom"], "ent": r["ent"]}
        load_full = np.asarray(r["load"])
        times["val_ce"] = time.time() - t1
        # 2. routing
        t1 = time.time()
        meas = evaluate(m, sp, lim(eval_rows(sp, AUDIT_TOK)), device, per_token=True, routing=True)
        mt = meas["per_token"]
        rout = {}
        for sysn, name in (("domain", "domain"), ("cluster", "e5")):
            lab, K = token_labels(mt, sp, sysn, wv)
            K = max(K, int(lab.max()) + 1)
            rs = routing_subject(mt["topi"], lab, mt["gdoc"], K, E)
            C = joint(mt["topi"], np.asarray(lab, np.int64), K, E)  # (L, K, E)
            pk = C.sum((0, 2)) / C.sum()
            lift = (C / np.maximum(1.0, C.sum(1, keepdims=True))) / np.maximum(1e-12, pk)[None, :, None]
            lmax = lift.max(1)  # (L, E)
            rout[name] = {k: rs[k] for k in ("mi_mean", "nmi_mean", "nmi_null_route_mean", "nmi_null_doclabel_mean",
                                             "subject_entropy_load_adj_mean", "nmi")}
            rout[name].update({"n_routing_specialized_cells": int((lmax >= 1.5).sum()),
                               "mean_max_lift": float(lmax.mean()), "K": int(K)})
            if name == "domain":
                prof = np.transpose(lift, (0, 2, 1))  # (L, E, K)
        res["routing_domain"], res["routing_e5"] = rout["domain"], rout["e5"]
        rdist = route_dists(mt, E)  # (L, 5, E)
        ov = np.asarray([[[float(np.minimum(rdist[li][i], rdist[li][j]).sum()) for j in range(5)] for i in range(5)]
                         for li in range(rdist.shape[0])])
        iu = np.triu_indices(5, 1)
        res["domain_topk_overlap"] = {"mean_pairwise_overlap": float(ov[:, iu[0], iu[1]].mean()),
                                      "by_layer": ov[:, iu[0], iu[1]].mean(1).tolist(),
                                      "pair_matrix_layer_mean": ov.mean(0).tolist()}
        la = np.asarray(meas["load"])
        res["load"] = {"audit_min_x_uniform": float(la.min() * E), "audit_max_x_uniform": float(la.max() * E),
                       "full_val_min_x_uniform": float(load_full.min() * E),
                       "full_val_max_x_uniform": float(load_full.max() * E),
                       "n_underused_full_val": int((load_full * E < UNDER).sum()),
                       "n_unused_full_val": int((load_full == 0).sum()),
                       "min_by_layer_x_uniform": (load_full.min(1) * E).tolist(),
                       "max_by_layer_x_uniform": (load_full.max(1) * E).tolist(),
                       "routing_entropy_full_val": r["ent"], "routing_entropy_audit": meas["ent"]}
        res["pathology"] = bool(load_full.min() * E < 0.1 or load_full.max() * E > 4.0 or r["ent"] < 0.45)
        times["routing"] = time.time() - t1
        # 3. function
        t1 = time.time()
        drows = lim(eval_rows(cs, AUDIT_TOK, paper_half(cs, 0)))
        dsc = evaluate(m, cs, drows, device, per_token=True)["per_token"]
        pe_rows = lim(eval_rows(sp, PE_TOK))
        summ, arr = per_expert_slc(m, cs, drows, sp, pe_rows, device, dsc["topi"], dsc["dom"], 5)
        res["function"] = summ
        times["function"] = time.time() - t1
        res["seconds"] = times
        np.savez_compressed(outn, doc_gdoc=gd, doc_sum=dsum, doc_n=dn, load_full=load_full, route_dists=rdist,
                            domain_lift_profiles=prof, overlap=ov, **{f"f_{k}": v for k, v in arr.items()})
        res["npz_sha256"] = guard.sha256_file(outn)
        guard.atomic_write_json(outj, res, mode=0o400)
        import os
        os.chmod(outn, 0o400)
        print(json.dumps({"key": a.key, "tag": a.tag, "ce": r["ce"], "nmi": rout["domain"]["nmi_mean"],
                          "inter": summ["mean_interaction"], "cinter": summ["cond_mean_interaction"],
                          "cond_n_adequate": summ["cond_n_adequate"], "seconds": times}), flush=True)
    except KeyboardInterrupt:
        status = "EXTERNALLY_STOPPED"
    finally:
        if mon is not None:
            mon.stop()
            if mon.contaminated and status == "COMPLETE":
                status = "EXTERNALLY_STOPPED"
    if a.device == "cuda":
        guard.gpu_ledger(t_iso, guard.now(), f"slc_eval_{a.key}_{a.tag}", "slc eval", status)
    print(json.dumps({"status": status, "seconds": time.time() - t0}))
    sys.exit(0 if status == "COMPLETE" else 77)


if __name__ == "__main__":
    main()
