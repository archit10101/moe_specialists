#!/usr/bin/env python
"""csr-semantic-edl-v1 Stage 3 (prereg section 10).

  sem_final.py freeze      choose + write frozen.json (0400); open CALIB-SELECTION once; confirmation decision
  sem_final.py train       (confirmed only) final runs: SEM s0/s1/s2, SHUF s0/s1/s2, explicit EDL s2
  sem_final.py valq        (after training) UNSEALED.json; VAL views -> frozen bge -> frozen SEM net -> table_SEM_val.npz
  sem_final.py eval        VAL evaluations (new runs: final; reused fle T4/EDL: supp)
  sem_final.py --child-valq  (GPU child for valq)"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402
from csr.sem_paths import ART, EDL_BETA, FINAL_RUNS, S50, T4_S100_RUN  # noqa: E402

SD = ART / "search"


def candidates_pool():
    import sem_search as S
    pool = {}
    for name in ["stage2"] + [f"gen{n}" for n in (1, 2)]:
        p = SD / (f"{name}_decision.json")
        if p.exists():
            for cid, r in json.loads(p.read_text())["candidates"].items():
                pool[cid] = r
    return pool, S


def freeze():
    from sem_driver import run_job
    fz_p = ART / "frozen.json"
    pool, S = candidates_pool()
    if not fz_p.exists():
        adv = [k for k, r in pool.items() if r["advance"]]
        valid = [k for k, r in pool.items() if r["criteria"]["v_no_pathology"]]
        cands = adv or valid
        if not cands:
            raise SystemExit("no valid candidate to freeze")

        def key(k):
            r = pool[k]
            fa = sum(r["summ"][f"{s}_s50"]["fsi_pseudo"] - r["t4"][f"{s}_s50"]["fsi_pseudo"] for s in (100, 101)) / 2
            return (0 if r["mean_dce_edl_s50"] <= 0.005 else 1, -fa, r["mean_dce_edl_s50"])
        win = sorted(cands, key=key)[0]
        rec = {"time": guard.now(), "frozen": win, "from_advanced": bool(adv), "cand": pool[win]["cand"],
               "beta": pool[win]["beta"], "pool": {k: {"advance": r["advance"], "mean_dce_edl_s50":
                                                       r["mean_dce_edl_s50"], "criteria": r["criteria"]}
                                                   for k, r in pool.items()}}
        guard.atomic_write_json(fz_p, rec, mode=0o400)
        S.log_event(f"FROZEN {win} (advanced={bool(adv)})")
    fz = json.loads(fz_p.read_text())
    win = fz["frozen"]
    if not (ART / "CALIBSEL_OPEN.json").exists():
        guard.atomic_write_json(ART / "CALIBSEL_OPEN.json", {"time": guard.now(), "frozen": win}, mode=0o400)
    c, beta = fz["cand"], fz["beta"]
    # SHUF control at both seeds must exist (it does when criterion (i) passed); otherwise train it now
    for s in (100, 101):
        rn = f"shuf_{win}_s{s}"
        if rn not in S.run_dirs():
            S.train(rn, S.cand_cond(c, beta, shuf=True), "search", 25, S50, s)
            S.drop_slot(rn, "50M milestone verified")
    runs = {f"s_{win}_s100": None, f"s_{win}_s101": None, f"shuf_{win}_s100": None, f"shuf_{win}_s101": None,
            "b_T4_s100": T4_S100_RUN, "b_T4_s101": None, "b_EDL_s100": None, "b_EDL_s101": None}
    for rn, rd in runs.items():
        run_job(f"calibsel_{rn}", ["sem_evaluate.py", "calibsel", "--run-dir", rd or S.run_dirs()[rn],
                                   "--run-name", rn, "--tag", "s50"], gpu=True, disk_gb=0.5)
    ev = {rn: json.loads((ART / "evals" / rn / "calibsel_s50.json").read_text()) for rn in runs}
    sm = {rn: {"ce": e["calib_sel"]["ce"], "fsi_pseudo": e["audit"]["pseudo"]["fsi"], "fsi_e5": e["audit"]["e5"]["fsi"],
               "mi_pseudo": e["audit"]["pseudo"]["routing"]["mi_mean"], "routing_entropy": e["audit"]["routing_entropy"],
               "n_starved": e["audit"]["n_below_0.1_uniform"], "load_max_x": e["audit"]["load_max_x_uniform"]}
          for rn, e in ev.items()}
    d = [(sm[f"s_{win}_s{s}"]["ce"] - sm[f"b_EDL_s{s}"]["ce"]) for s in (100, 101)]
    crit = {"mean_dce_edl": sum(d) / 2, "ce_ok": sum(d) / 2 <= 0.005,
            "fsi_ok": all(sm[f"s_{win}_s{s}"]["fsi_pseudo"] > sm[f"b_T4_s{s}"]["fsi_pseudo"] and
                          sm[f"s_{win}_s{s}"]["fsi_pseudo"] > sm[f"shuf_{win}_s{s}"]["fsi_pseudo"] for s in (100, 101)),
            "no_pathology": all(sm[f"s_{win}_s{s}"]["n_starved"] == 0 and sm[f"s_{win}_s{s}"]["routing_entropy"] >= 0.45
                                and sm[f"s_{win}_s{s}"]["load_max_x"] <= 4.0 for s in (100, 101))}
    crit["confirmed"] = bool(crit["ce_ok"] and crit["fsi_ok"] and crit["no_pathology"])
    out = {"time": guard.now(), "frozen": win, "summary": sm, "criteria": crit}
    p = ART / "calibsel_decision.json"
    if not p.exists():
        guard.atomic_write_json(p, out, mode=0o400)
    S.log_event(f"CALIB-SELECTION: confirmed={crit['confirmed']} {json.dumps(crit)}")
    return json.loads(p.read_text())


def train_finals():
    import sem_search as S
    dec = json.loads((ART / "calibsel_decision.json").read_text())
    if not dec["criteria"]["confirmed"]:
        raise SystemExit("not confirmed: no finals under this protocol")
    fz = json.loads((ART / "frozen.json").read_text())
    c, beta = fz["cand"], fz["beta"]
    plan = [("final_SEM_s0", S.cand_cond(c, beta), 0), ("final_SEM_s1", S.cand_cond(c, beta), 1),
            ("final_EDL_s2", {"edl_beta": EDL_BETA}, 2), ("final_SEM_s2", S.cand_cond(c, beta), 2),
            ("final_SHUF_s0", S.cand_cond(c, beta, True), 0), ("final_SHUF_s1", S.cand_cond(c, beta, True), 1),
            ("final_SHUF_s2", S.cand_cond(c, beta, True), 2)]
    for rn, cond, seed in plan:
        st = S.run_dirs().get(rn)
        if st and json.loads((guard.ROOT / st / "status.json").read_text())["status"] == "COMPLETE":
            continue
        if rn == "final_SHUF_s2" and not S.budget_ok(0.6 + 0.9 + remaining_eval_hours(), rn):
            S.log_event("budget rule: SHUF s2 final dropped")
            continue
        S.train(rn, cond, "final", 100, 3880, seed)


def remaining_eval_hours():
    done = {p.parent.name for p in (ART / "evals").glob("*/final.json")} | \
           {p.parent.name for p in (ART / "evals").glob("*/supp.json")}
    need = {"final_SEM_s0": 1.0, "final_SEM_s1": 1.0, "final_SEM_s2": 1.0, "final_EDL_s2": 1.0, "final_SHUF_s0": 0.75,
            "final_SHUF_s1": 0.75, "T4_s0": 0.15, "T4_s1": 0.15, "T4_s2": 0.15, "EDL_s0": 0.15, "EDL_s1": 0.15}
    return sum(v for k, v in need.items() if k not in done)


def child_valq():
    import numpy as np
    import torch
    from csr.fle_lib import decode_batch
    from csr.fsp_lib import Split
    from csr.gc_lib import BGE, ProtoNet, opaque_id, paper_q, view_starts
    from csr.gc_paths import BGE_DIR
    from csr.sem_paths import GC_LAT
    guard.occupancy_gate("sem valq")
    device = guard.setup_cuda_device()
    def build(split_name):
        return _views_q(split_name, device)
    # reproduction check: the same code on CALIB-SEARCH must reproduce the frozen calib table
    oc, qc, _ = build("calib_search")
    zc = np.load(GC_LAT / "table_SEM_calib_search.npz")
    assert (zc["opaque"] == oc).all()
    repro = float(np.abs(zc["q"] - qc).max())
    agree = float((zc["q"].argmax(1) == qc.argmax(1))[zc["q"].sum(1) > 0].mean())
    if agree < 0.99:
        raise guard.GuardError(f"VAL assignment pipeline does not reproduce the frozen calib table (agree {agree})")
    ov, q, nv = build("val")
    out = ART / "tables" / "table_SEM_val.npz"
    np.savez(out, opaque=ov, q=q.astype(np.float32), K=np.int64(16), nv=nv)
    print(json.dumps({"papers": len(ov), "views": int(nv.sum()), "sha256": guard.sha256_file(out),
                      "calib_repro_max_abs_q_diff": repro, "calib_repro_argmax_agreement": agree}))


def _views_q(split_name, device):
    import numpy as np
    import torch
    from csr.fle_lib import decode_batch
    from csr.fsp_lib import Split
    from csr.gc_lib import BGE, ProtoNet, opaque_id, paper_q, view_starts
    from csr.gc_paths import BGE_DIR
    from csr.sem_paths import GC_LAT
    sp = Split(split_name)
    segs = {}
    for di, (p, sid) in enumerate(zip(sp.papers, sp.ids)):
        segs.setdefault(p, []).append((int(sid.split("#")[1]), di))
    rows = sorted((opaque_id(p), [d for _, d in sorted(v)]) for p, v in segs.items())
    P = len(rows)
    views = np.zeros((P, 4, 256), np.int64)
    nv = np.zeros(P, np.int64)
    for i, (o, dis) in enumerate(rows):
        toks = np.concatenate([sp.doc(d) for d in dis])
        st = view_starts(len(toks), 256)
        nv[i] = len(st)
        for j, s in enumerate(st):
            views[i, j] = toks[s:s + 256]
    ii, jj = np.nonzero(np.arange(4)[None, :] < nv[:, None])
    enc = BGE(device, BGE_DIR)
    sem = np.zeros((P, 4, 384), np.float32)
    sem[ii, jj] = enc.encode(decode_batch(views[ii, jj]))
    rp = np.load(GC_LAT / "residualizer_params.npz")
    S_ = np.zeros_like(sem)
    S_[ii, jj] = ((sem[ii, jj] - rp["sm"]) / rp["ssd"]).astype(np.float32)
    ck = torch.load(GC_LAT / "SEM_K16_s2" / "net.pt", weights_only=False)
    net = ProtoNet(384, 16, seed=2)
    net.load_state_dict(ck["state"])
    with torch.no_grad():
        vl = net(torch.as_tensor(S_.reshape(-1, 384))).reshape(P, 4, 16).numpy().astype(np.float64)
    return np.asarray([r[0] for r in rows]), paper_q(vl, nv), nv


def valq():
    from sem_driver import run_job
    if not (ART / "UNSEALED.json").exists():
        guard.atomic_write_json(ART / "UNSEALED.json", {"first_label_and_val_access": guard.now(),
                                                         "frozen_sha256": guard.sha256_file(ART / "frozen.json"),
                                                         "note": "all candidate, selection and final-training "
                                                                 "decisions are complete"}, mode=0o400)
    if not (ART / "tables" / "table_SEM_val.npz").exists():
        run_job("valq", ["sem_final.py", "--child-valq"], gpu=True, disk_gb=0.5)


def evals():
    import sem_search as S
    from sem_driver import run_job
    for rn in ("final_SEM_s0", "final_SEM_s1", "final_SEM_s2", "final_EDL_s2", "final_SHUF_s0", "final_SHUF_s1",
               "final_SHUF_s2"):
        rd = S.run_dirs().get(rn)
        if not rd:
            continue
        tags = "p040,p100" if "SHUF" in rn else "p040,p060,p080,p100"
        run_job(f"val_{rn}", ["sem_evaluate.py", "final", "--run-dir", rd, "--run-name", rn, "--tags", tags,
                              "--per-expert"], gpu=True, disk_gb=0.5)
    for k, rd in FINAL_RUNS.items():
        run_job(f"supp_{k}", ["sem_evaluate.py", "supp", "--run-dir", rd, "--run-name", f"fle_final_{k}",
                              "--tags", "p040,p100"], gpu=True, disk_gb=0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=["freeze", "train", "valq", "eval"])
    ap.add_argument("--child-valq", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    from csr.sem_paths import check_prereg
    check_prereg()
    if a.child_valq:
        child_valq()
    elif a.cmd == "freeze":
        print(json.dumps(freeze()["criteria"]))
    elif a.cmd == "train":
        train_finals()
    elif a.cmd == "valq":
        valq()
    elif a.cmd == "eval":
        evals()


if __name__ == "__main__":
    main()
