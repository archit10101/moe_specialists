#!/usr/bin/env python
"""oracle-residual-v1 exact evaluation (guarded GPU job; CPU for tests). One frozen TOKEN-4 seed + one joint checkpoint
(A, B, C banks) on a CALIB-SEARCH partition (SEL = paper-half A, EVAL = paper-half B). The frozen base forward is shared;
D (frozen TOKEN-4) is the zero-residual reference on exactly the same tokens.

Per valid token it stores: ce_D, ce_A, ce_B, ce_C (fp32; assigned-path CE), idx_C (C's routed specialist), lab_sem /
lab_shuf (frozen CALIB-SEARCH semantic / length-stratified shuffled label of the token's paper), paper index, position,
document length, target id; with --full also dA, dB (N, 16) fp16 = CE with specialist k of bank A / B forced for every
token minus ce_D (the replacement matrix for the functional analysis; recomputed exactly through norm_f and the head).
The base state hash is verified against the checkpoint before and after.

  orv1_eval.py --seed S --ckpt PATH --part SEL|EVAL --out NPZ [--full] [--rows-limit N]"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--part", choices=["SEL", "EVAL", "VAL"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--rows-limit", type=int, default=0)
    ap.add_argument("--micro-rows", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    if not guard.require_execution(a.execute, "gpu" if a.device == "cuda" else "cpu-model"):
        return
    from csr.orv1_paths import check_prereg
    if not a.rows_limit:
        check_prereg()
    mon = None
    if a.device == "cuda":
        guard.occupancy_gate(f"orv1 eval {Path(a.out).name}")
        mon = guard.OccupancyMonitor(interval=15.0)
        mon.start()
    guard.register_self(f"orv1 eval {Path(a.out).name}")
    t_iso, t0 = guard.now(), time.time()
    status, rc = "COMPLETE", 0
    try:
        run(a, mon)
    except SystemExit as ex:
        rc = int(ex.code or 0)
        status = {77: "EXTERNALLY_STOPPED", 0: "COMPLETE"}.get(rc, "FAILED")
    except KeyboardInterrupt:
        status, rc = "EXTERNALLY_STOPPED", 77
    finally:
        if mon is not None:
            mon.stop()
            if mon.contaminated and status == "COMPLETE":
                status, rc = "EXTERNALLY_STOPPED", 77
    if a.device == "cuda":
        guard.gpu_ledger(t_iso, guard.now(), f"orv1_eval_{Path(a.out).stem}", "orv1 eval", status)
    print(json.dumps({"status": status, "seconds": round(time.time() - t0, 1)}), flush=True)
    sys.exit(rc)


def run(a, mon):
    import numpy as np
    import torch
    from csr.fsp_lib import Split
    from csr.orv1_lib import (CONDS, base_forward, head_ce, load_base, load_labels, load_part, make_batch, make_conds,
                              rms0, state_hash, to_dev, token_assign)
    from csr.orv1_paths import K
    out = Path(a.out)
    if out.exists():
        print("exists")
        return
    dev = guard.setup_cuda_device() if a.device == "cuda" else torch.device("cpu")
    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    model, ck_sha = load_base(a.seed, dev)
    h0 = state_hash(model)
    S = torch.load(a.ckpt, map_location=dev, weights_only=False)
    if S["base_state_hash"] != h0 or S["cfg"]["seed"] != a.seed:
        raise guard.GuardError("checkpoint does not belong to this frozen base")
    conds = make_conds(a.seed, dev)
    for c in CONDS:
        conds[c].load_state_dict(S["conds"][c])
        conds[c].eval()
    if a.part == "VAL":  # Stage 3 only (prereg 10): frozen VAL assignment from orv1_val_assign.py
        from csr.orv1_lib import unflatten_rows
        from csr.orv1_paths import ART
        cs = Split("val")
        zl = np.load(ART / "val_assign" / "labels_val.npz")
        sem, shuf = zl["sem"].astype(np.int64), zl["shuf"].astype(np.int64)
        zp = np.load(ART / "val_assign" / "parts_val.npz")
        rows = unflatten_rows(zp["VAL_docs"], zp["VAL_row_len"])
    else:
        cs = Split("calib_search")
        sem, shuf = load_labels("cs")
        rows = load_part(a.part)
    lab_sem, lab_shuf = torch.as_tensor(sem, device=dev), torch.as_tensor(shuf, device=dev)
    if a.rows_limit:
        rows = rows[:a.rows_limit]
    papers = sorted({cs.papers[d] for r in rows for d in r})
    pidx = {p: i for i, p in enumerate(papers)}
    paper_of_doc = np.full(cs.n_docs, -1, np.int64)
    for r in rows:
        for d in r:
            paper_of_doc[d] = pidx[cs.papers[d]]
    keys = ["ce_D", "ce_A", "ce_B", "ce_C", "idx_C", "lab_sem", "lab_shuf", "paper", "pos", "doclen", "tgt"]
    buf = {k: [] for k in keys}
    if a.full:
        buf["dA"], buf["dB"] = [], []
    cons = {"A": [0.0, 0.0], "B": [0.0, 0.0]}
    with torch.no_grad():
        for i in range(0, len(rows), a.micro_rows):
            if mon is not None and mon.contaminated:
                sys.exit(77)
            bt = to_dev(make_batch(rows[i:i + a.micro_rows], cs), dev)
            x, hid = base_forward(model, bt)
            tgt = bt["targets"].reshape(-1)
            v = tgt >= 0
            ce_d = head_ce(model, hid, tgt, from_hidden=True)
            xn = rms0(x)
            asg_a, asg_b = token_assign(bt, lab_sem, lab_shuf)
            ces, idxc = {}, None
            for c in CONDS:
                delta, idx, _ = conds[c](xn, {"A": asg_a, "B": asg_b, "C": None}[c], v)
                ces[c] = head_ce(model, x + delta, tgt)
                if c == "C":
                    idxc = idx
            gd = bt["gdoc"].reshape(-1)[v]
            buf["ce_D"].append(ce_d[v].cpu().numpy())
            for c in CONDS:
                buf[f"ce_{c}"].append(ces[c][v].cpu().numpy())
            buf["idx_C"].append(idxc[v].cpu().numpy().astype(np.int8))
            buf["lab_sem"].append(asg_a[v].cpu().numpy().astype(np.int8))
            buf["lab_shuf"].append(asg_b[v].cpu().numpy().astype(np.int8))
            gdn = gd.cpu().numpy()
            buf["paper"].append(paper_of_doc[gdn].astype(np.int32))
            buf["pos"].append(bt["positions"].reshape(-1)[v].cpu().numpy().astype(np.int16))
            buf["doclen"].append(cs.lengths[gdn].astype(np.int16))
            buf["tgt"].append(tgt[v].cpu().numpy().astype(np.int32))
            if a.full:
                for c, lab in (("A", asg_a), ("B", asg_b)):
                    outs = conds[c].all_outputs(xn)
                    d = torch.empty(int(v.sum()), K, device=dev)
                    for k in range(K):
                        d[:, k] = head_ce(model, x + outs[:, k], tgt)[v] - ce_d[v]
                    own = d.gather(1, lab[v].unsqueeze(1)).squeeze(1)
                    cons[c][0] += float((own - (ces[c][v] - ce_d[v])).abs().sum())
                    cons[c][1] = max(cons[c][1], float((own - (ces[c][v] - ce_d[v])).abs().max()))
                    buf[f"d{c}"].append(d.cpu().numpy().astype(np.float16))
    h1 = state_hash(model)
    arr = {k: np.concatenate(v_) for k, v_ in buf.items()}
    n = len(arr["ce_D"])
    loadC = np.bincount(arr["idx_C"].astype(np.int64), minlength=K) / n
    summ = {"seed": a.seed, "part": a.part, "ckpt": str(a.ckpt), "ckpt_step": S["step"], "ckpt_tokens": S["tokens"],
            "n_tokens": int(n), "n_papers": len(papers), "base_state_hash": h0, "base_unchanged": h1 == h0,
            **{f"ce_{c}": float(arr[f"ce_{c}"].astype(np.float64).mean()) for c in ("D", "A", "B", "C")},
            "loadC": loadC.round(5).tolist(), "loadC_max": float(loadC.max()),
            "loadC_eff": float(np.exp(-(loadC[loadC > 0] * np.log(loadC[loadC > 0])).sum())),
            "matrix_consistency_mean_abs": {c: cons[c][0] / max(1, n) for c in cons} if a.full else None,
            "matrix_consistency_max_abs": {c: cons[c][1] for c in cons} if a.full else None, "time": guard.now()}
    if h1 != h0:
        raise guard.GuardError("frozen base state changed during evaluation")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.stem}.tmp.npz")
    np.savez(tmp, papers=np.asarray(papers), **arr)
    tmp.rename(out)
    out.chmod(0o400)
    summ["sha256"] = guard.sha256_file(out)
    guard.atomic_write_json(out.with_suffix(".json"), summ, mode=0o400)
    print(json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
