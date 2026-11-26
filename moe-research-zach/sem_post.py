#!/usr/bin/env python
"""csr-semantic-edl-v1 POST-TERMINAL descriptive analysis (deviation 4; no decision depends on it).

Run only after calibsel_decision.json exists with confirmed=False (the experiment's terminal decision: no finals, VAL
never opened). Reads real domain labels of CALIB-SEARCH / main only, for diagnosis A-G. Never loads VAL or test.

  sem_post.py purity                      CPU: domain purity / information of the SEM clusters (and the shuffled table)
  sem_post.py audit --run-name R --tag T [--per-expert]
                                          GPU: fle audit_one (domain + e5; discovery CS half A, measurement CS half B)
                                          and, with --per-expert, per-expert interventions (domain / pseudo / e5
                                          interactions from the same passes) on CS half B.
Outputs artifacts/semantic-edl-v1/post/ (0400, written once)."""
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


def _terminal(ART):
    p = ART / "calibsel_decision.json"
    if not p.exists() or json.loads(p.read_text())["criteria"]["confirmed"]:
        raise guard.GuardError("post-terminal analysis only after an unconfirmed CALIB-SELECTION decision")


def _info(lab, dom, K, D=5):
    import numpy as np
    C = np.zeros((K, D))
    np.add.at(C, (lab, dom), 1)
    P = C / C.sum()
    pk, pd = P.sum(1), P.sum(0)
    nz = P > 0
    mi = float((P[nz] * np.log(P[nz] / (pk[:, None] * pd[None, :])[nz])).sum())
    hd = float(-(pd[pd > 0] * np.log(pd[pd > 0])).sum())
    hk = float(-(pk[pk > 0] * np.log(pk[pk > 0])).sum())
    purity = float(C.max(1).sum() / C.sum())
    return {"n": int(C.sum()), "mi_nats": mi, "H_domain": hd, "H_cluster": hk, "mi_over_H_domain": mi / hd,
            "nmi_arith": 2 * mi / (hd + hk), "purity": purity, "majority_domain_baseline": float(pd.max()),
            "counts": C.astype(int).tolist(),
            "cluster_purity": (C.max(1) / np.maximum(1, C.sum(1))).tolist(),
            "cluster_mass": pk.tolist()}


def purity(ART):
    import numpy as np
    from csr.fsp_lib import Split
    from csr.gc_lib import opaque_id
    from csr.sem_paths import SEM_CS, SEM_TABLE
    res = {"time": guard.now(), "note": "descriptive, post-terminal; real labels of main and CALIB-SEARCH papers"}
    tabs = {"SEM": SEM_TABLE, "SHUF_SEM": ART / "tables" / "table_SHUF_SEM.npz"}
    for split_name in ("main", "calib_search"):
        sp = Split(split_name)
        pap_dom = {}
        for p, d in zip(sp.papers, sp.domains):
            pap_dom.setdefault(p, []).append(int(d))
        for tn, tp in list(tabs.items()) + ([("SEM_calib", SEM_CS)] if split_name == "calib_search" else []):
            if split_name == "calib_search" and tn != "SEM_calib":
                continue
            z = np.load(tp)
            idx = {str(o): i for i, o in enumerate(z["opaque"])}
            q = z["q"]
            labs, doms, soft = [], [], []
            for p, ds in pap_dom.items():
                i = idx.get(opaque_id(p))
                if i is None or q[i].sum() <= 0:
                    continue
                labs.append(int(q[i].argmax()))
                doms.append(int(np.bincount(ds).argmax()))
                soft.append(q[i] / q[i].sum())
            labs, doms, soft = np.asarray(labs), np.asarray(doms), np.asarray(soft)
            r = _info(labs, doms, q.shape[1])
            # soft-q information: E_paper sum_k q_k log (p(k|d)/p(k))
            pkd = np.stack([soft[doms == d].mean(0) for d in range(5)])
            pk = soft.mean(0)
            pdm = np.bincount(doms, minlength=5) / len(doms)
            r["soft_mi_nats"] = float(sum(pdm[d] * (pkd[d] * np.log(np.maximum(pkd[d], 1e-12) / np.maximum(pk, 1e-12)))
                                          .sum() for d in range(5)))
            r["mean_max_q"] = float(soft.max(1).mean())
            r["n_papers_total"] = len(pap_dom)
            res[f"{split_name}/{tn}"] = r
    return res


def audit(a, ART):
    import numpy as np
    from fle_evaluate import audit_one, load_milestone
    from csr.fsp_lib import Split
    from csr.lfr_audit import token_labels
    from csr.lfr_eval import eval_rows, evaluate, paper_half
    from csr.sem_lib import per_expert_multi, pseudo_windows
    from csr.sem_paths import FINAL_RUNS, LFR_LABELS, SEM_CS, T4_S100_RUN
    rd_map = json.loads((ART / "run_dirs.json").read_text())
    rd_map.setdefault("b_T4_s100", T4_S100_RUN)
    rd = guard.inside_root(rd_map[a.run_name])
    ckpts = json.loads((rd / "status.json").read_text())["ckpts"]
    od = ART / "post"
    od.mkdir(exist_ok=True)
    out_p = od / f"audit_{a.run_name}_{a.tag}{'_pe' if a.per_expert else ''}.json"
    if out_p.exists():
        raise guard.GuardError(f"{out_p} exists")
    mon = None
    if a.device == "cuda":
        guard.occupancy_gate("sem post audit")
        device = guard.setup_cuda_device()
        mon = guard.OccupancyMonitor(interval=15.0)
        mon.start()
    else:
        import torch
        device = torch.device("cpu")
    guard.register_self(f"sem post audit {a.run_name} {a.tag}")
    t_iso, t0 = guard.now(), time.time()
    cs = Split("calib_search")
    wc = dict(np.load(LFR_LABELS / "windows_calib.npz"))
    pw = pseudo_windows(cs, SEM_CS)
    disc_rows = eval_rows(cs, AUDIT_TOK, paper_half(cs, 0))
    meas_rows = eval_rows(cs, AUDIT_TOK, paper_half(cs, 1))
    res = {"run": rd.name, "run_name": a.run_name, "tag": a.tag, "ckpt": ckpts[a.tag], "time": guard.now(),
           "protocol": "discovery CALIB-SEARCH half A, measurement CALIB-SEARCH half B; descriptive post-terminal"}
    status = "COMPLETE"
    try:
        m, _ = load_milestone(ckpts[a.tag]["path"], device)
        res["audit"] = audit_one(m, cs, disc_rows, cs, meas_rows, device, wc, wc, True)
        if a.per_expert:
            disc = evaluate(m, cs, disc_rows, device, per_token=True)["per_token"]
            lab = pw["cluster"]
            e5d, Ke = token_labels(disc, cs, "cluster", wc)
            systems = {"domain": (disc["dom"].astype(np.int64), lambda b: b["dom"], 5),
                       "pseudo": (lab[disc["gdoc"].astype(np.int64)], lambda b: lab[b["gdoc"].astype(np.int64)], 17),
                       "e5": (e5d, lambda b: token_labels(b, cs, "cluster", wc)[0], Ke)}
            res["per_expert"] = per_expert_multi(m, cs, disc_rows, cs, eval_rows(cs, PE_TOK, paper_half(cs, 1)),
                                                 device, disc, systems)
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
        guard.gpu_ledger(t_iso, guard.now(), f"sem_post_{a.run_name}_{a.tag}", "sem post audit", status)
        with open(ART / "compute_ledger.tsv", "a") as f:
            f.write(f"{t_iso}\t{guard.now()}\t{time.time() - t0:.1f}\t{rd.name}\tpost_audit_{a.tag}_{a.run_name}"
                    f"{'_pe' if a.per_expert else ''}\tevaluation\t{status}\n")
    print(json.dumps({"status": status, "seconds": res["seconds"]}))
    sys.exit(0 if status == "COMPLETE" else 77)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["purity", "audit"])
    ap.add_argument("--run-name", default="")
    ap.add_argument("--tag", default="s50")
    ap.add_argument("--per-expert", action="store_true")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    from csr.sem_paths import ART, check_prereg
    check_prereg()
    _terminal(ART)
    if a.mode == "purity":
        p = ART / "post" / "cluster_domain_purity.json"
        p.parent.mkdir(exist_ok=True)
        if not p.exists():
            guard.atomic_write_json(p, purity(ART), mode=0o400)
        print(p)
        return
    if not guard.require_execution(a.execute, "gpu" if a.device == "cuda" else "cpu-model"):
        return
    audit(a, ART)


if __name__ == "__main__":
    main()
