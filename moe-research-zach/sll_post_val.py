#!/usr/bin/env python
"""csr-semantic-lift-expert-learning-v1 POST-TERMINAL DESCRIPTIVE VAL pass (opened only after the 50M gate decision
was frozen and the protocol became terminal; never used for any decision; real domain labels used here only).
For each 50M checkpoint (E/F lambda 0.5 and the reused C6 / SHUF / T4 / EDL, seeds 100/101): full-VAL CE (overall,
per domain, per document), real-domain and e5 routing NMI/MI (slk_lib.routing_subject on the semantic-edl-long-k audit
rows), per-domain top-4 expert-selection distributions and their pairwise overlap, load/entropy, and the per-expert
replacement interaction for the real-domain and e5 systems (slk_lib.per_expert_boot; discovery CALIB-SEARCH half A,
measurement eval_rows(VAL, 50k/domain) = the semantic-edl-long-k 'val_s50_pe50k' measurement).
Writes artifacts/semantic-lift-expert-learning-v1/post_val/<key>.json (once each, 0400)."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402

AUDIT_TOK = 200_000
PE_TOK = 50_000


def targets():
    import sll_run as R
    t = {}
    for s in (100, 101):
        t[f"E_s{s}"] = R.run_dirs()[f"S_E_L05_s{s}"]
        t[f"F_s{s}"] = R.run_dirs()[f"S_F_L05_s{s}"]
        for c in ("C6", "SHUF", "EDL", "T4"):
            t[f"{c}_s{s}"] = R.REUSED50[f"{c}_s{s}"]
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--only", default="")
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
    from csr.slk_lib import per_expert_boot, routing_subject
    from csr.sll_paths import ART, LFR_LABELS, check_prereg
    check_prereg()
    st = json.loads((ART / "STATE.json").read_text())["state"]
    if not st.startswith("TERMINAL") or not (ART / "promotion.json").exists():
        raise guard.GuardError("descriptive VAL pass only after the protocol is terminal")
    od = ART / "post_val"
    od.mkdir(exist_ok=True)
    mon = None
    if a.device == "cuda":
        guard.occupancy_gate("sll post val")
        device = guard.setup_cuda_device()
        mon = guard.OccupancyMonitor(interval=15.0)
        mon.start()
    else:
        device = torch.device("cpu")
    guard.register_self("sll post val (descriptive)")
    t_iso, t0 = guard.now(), time.time()
    sp, cs = Split("val"), Split("calib_search")
    wv = dict(np.load(LFR_LABELS / "windows_val.npz"))
    wc = dict(np.load(LFR_LABELS / "windows_calib.npz"))
    drows = eval_rows(cs, AUDIT_TOK, paper_half(cs, 0))
    meas_rows = eval_rows(sp, AUDIT_TOK)
    pe_rows = eval_rows(sp, PE_TOK)
    status, done = "COMPLETE", []
    try:
        for key, rd in targets().items():
            if a.only and key not in a.only.split(","):
                continue
            outp = od / f"{key}.json"
            if outp.exists():
                continue
            ck = json.loads((guard.ROOT / rd / "status.json").read_text())["ckpts"]["s50"]
            m, _ = load_milestone(ck["path"], device)
            E = m.cfg.n_experts
            r = evaluate(m, sp, eval_rows(sp), device, routing=True)
            res = {"key": key, "run_dir": rd, "ckpt": ck, "time": guard.now(), "descriptive": True,
                   "val": {k: v for k, v in r.items() if k != "doc_ce"}, "val_doc_ce": {str(k): v for k, v in r["doc_ce"].items()}}
            mf = evaluate(m, sp, meas_rows, device, per_token=True, routing=True)
            meas = mf["per_token"]
            for sysn in ("domain", "cluster"):
                lm, K = token_labels(meas, sp, sysn, wv)
                rs = routing_subject(meas["topi"], lm, meas["gdoc"], max(K, int(lm.max()) + 1), E)
                res["routing_" + ("domain" if sysn == "domain" else "e5")] = {
                    k: rs[k] for k in ("mi_mean", "nmi_mean", "nmi_null_route_mean", "nmi_null_doclabel_mean",
                                       "subject_entropy_load_adj_mean", "nmi")}
            rdist = route_dists(meas, E)  # (L, 5, E)
            ov = []
            for li in range(rdist.shape[0]):
                P = rdist[li]
                ov.append([[float(np.minimum(P[i], P[j]).sum()) for j in range(5)] for i in range(5)])
            ov = np.asarray(ov)
            iu = np.triu_indices(5, 1)
            res["domain_topk_overlap"] = {"mean_pairwise_overlap": float(ov[:, iu[0], iu[1]].mean()),
                                          "by_layer": ov[:, iu[0], iu[1]].mean(1).round(4).tolist(),
                                          "pair_matrix_layer_mean": ov.mean(0).round(4).tolist()}
            res["load"], res["routing_entropy"] = mf["load"], mf["ent"]
            dsc = evaluate(m, cs, drows, device, per_token=True)["per_token"]
            e5d, Ke = token_labels(dsc, cs, "cluster", wc)
            systems = {"domain": (dsc["dom"].astype(np.int64), lambda b: b["dom"], 5),
                       "e5": (e5d, lambda b: token_labels(b, sp, "cluster", wv)[0], Ke)}
            pe = per_expert_boot(m, cs, drows, sp, pe_rows, device, dsc, systems)
            for k_ in ("domain", "e5"):
                pe[k_].pop("boot_mean_samples_full", None)
            res["per_expert"] = pe
            guard.atomic_write_json(outp, res, mode=0o400)
            done.append(key)
            print(json.dumps({"key": key, "ce": r["ce"], "nmi_domain": res["routing_domain"]["nmi_mean"],
                              "inter_domain": pe["domain"]["mean_interaction"]}), flush=True)
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
        guard.gpu_ledger(t_iso, guard.now(), "sll_post_val", "sll post val", status)
        with open(ART / "compute_ledger.tsv", "a") as f:
            f.write(f"{t_iso}\t{guard.now()}\t{time.time() - t0:.1f}\tpost_val\tpost_val\tpost_terminal_descriptive\t{status}\n")
    print(json.dumps({"status": status, "done": done, "seconds": time.time() - t0}))
    sys.exit(0 if status == "COMPLETE" else 77)


if __name__ == "__main__":
    main()
