#!/usr/bin/env python
"""spb1 exact extraction (guarded GPU job; CPU for tests). One frozen TOKEN-4 seed + one spb1 bank checkpoint (three
banks) [+ the COARSE-16 orv1 bank of the same seed] on a CALIB-SEARCH partition:
  TRAIN = CALIB-SEARCH paper-half "EVAL" (deployed-router training only), SEL = paper-half "SEL" (all Stage-1/2
  measurements). VAL only after ART/FROZEN_FINAL.json (Stage 3). Same rows, order and micro-batching for every bank.

Writes into --out-dir (all 0400; sha256 in summary):
  common_<P>.npz      (written once per base seed into EXTRACT/common_s<seed>; reused) ce_base fp32, tgt, pos, doc,
                      paper (+ papers), doclen, lab_sem / lab_shuf (paper-level semantic / balanced-shuffled cluster),
                      lab_rsem / lab_rsem_prev (region-prior argmax of the current / previous 256-token region of the
                      document; CALIB-SEARCH only; evaluation stratifier, never a router input)
  bank_<B>_<P>.npz    d (N, E) fp16 = CE(base + specialist e) - CE_base for EVERY token and specialist; e_tr int8 = the
                      training-time route (pair mode: own cluster + pair subrouter; free mode: own router); marg_tr fp16
                      (top-1 minus top-2 allowed logit); SEL pair banks also pdiff (N, K) fp16 = logit difference of
                      the two members of every cluster pair (M = 2) or lg (N, E) fp16 (M > 2)
  coarse_<P>.npz      (EXTRACT/coarse_s<seed>) COARSE-16 (orv1 A) d (N, 16) fp16, e_tr = oracle semantic label
  summary_<P>.json    mean CEs, consistency of table lookups vs the directly routed CE, sha256

  spb1_extract.py --seed S --ckpt RUN/ckpt_T.pt --part TRAIN|SEL --out-dir DIR [--coarse] [--rows-limit N]"""
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
    ap.add_argument("--part", required=True, choices=["TRAIN", "SEL", "VAL"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--common-dir", default="")
    ap.add_argument("--coarse", action="store_true")
    ap.add_argument("--rows-limit", type=int, default=0)
    ap.add_argument("--micro-rows", type=int, default=16)
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    if not guard.require_execution(a.execute, "gpu" if a.device == "cuda" else "cpu-model"):
        return
    from csr.spb1_paths import check_prereg
    if not a.test:
        check_prereg()
    mon = None
    if a.device == "cuda":
        guard.occupancy_gate("spb1 extract")
        mon = guard.OccupancyMonitor(interval=15.0)
        mon.start()
    guard.register_self("spb1 extract")
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
        guard.gpu_ledger(t_iso, guard.now(), f"spb1_extract_{a.part}", "spb1 extract", status)
    print(json.dumps({"status": status, "seconds": round(time.time() - t0, 1)}), flush=True)
    sys.exit(rc)


def partition(part):
    """(split, rows, sem labels, shuf labels, region prior (off, P) or None)."""
    import numpy as np
    from csr.fsp_lib import Split
    from csr.orv1_lib import unflatten_rows
    from csr import spb1_paths as P
    if part == "VAL":
        if not (P.ART / "FROZEN_FINAL.json").exists():
            raise guard.GuardError("VAL is locked until FROZEN_FINAL.json exists")
        sp = Split("val")
        zl = np.load(P.TABLES / "labels_val.npz")
        zp = np.load(P.ORV1_VAL / "parts_val.npz")
        rows = unflatten_rows(zp["VAL_docs"], zp["VAL_row_len"])
        return sp, rows, zl["sem"].astype(np.int64), zl["shuf"].astype(np.int64), None
    sp = Split("calib_search")
    zl = np.load(P.TABLES / "labels_cs.npz")
    zp = np.load(P.ORV1_TABLES / "parts.npz")
    half = {"TRAIN": "EVAL", "SEL": "SEL"}[part]
    rows = unflatten_rows(zp[f"{half}_docs"], zp[f"{half}_row_len"])
    z = np.load(guard.ROOT / "artifacts/csr-crossfit-semantic-utility-residual-v1/tables/psem_calib_search.npz")
    off = np.full(sp.n_docs, -1, np.int64)
    off[z["docs"]] = z["ptr"][:-1]
    return sp, rows, zl["sem"].astype(np.int64), zl["shuf"].astype(np.int64), (off, z["p"].astype(np.float32))


def save(path, arrs):
    import numpy as np
    tmp = path.with_name(f".{path.stem}.tmp.npz")
    np.savez(tmp, **arrs)
    tmp.rename(path)
    path.chmod(0o400)
    return guard.sha256_file(path)


def run(a, mon):
    import numpy as np
    import torch
    from csr.orv1_lib import Cond, base_forward, head_ce, load_base, make_batch, state_hash, to_dev
    from csr.spb1_lib import make_banks, rms0, token_groups
    from csr import spb1_paths as P
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cdir = Path(a.common_dir) if a.common_dir else P.EXTRACT / f"common_s{a.seed}"
    cdir.mkdir(parents=True, exist_ok=True)
    if (out / f"summary_{a.part}.json").exists():
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
    G = S["cfg"]["gen_cfg"]
    M, K = int(G["M"]), P.K
    E = K * M
    banks = make_banks(a.seed, K, M, dev)
    names = list(banks)
    for c in names:
        banks[c].load_state_dict(S["banks"][c])
        banks[c].eval()
    coarse = None
    want_coarse = a.coarse and not (cdir / f"coarse_summary_{a.part}.json").exists()
    if want_coarse:
        cp, csha = P.COARSE_CKPT[a.seed]
        if not a.test and guard.sha256_file(guard.ROOT / cp) != csha:
            raise guard.GuardError("COARSE-16 checkpoint hash mismatch")
        SC = torch.load(guard.ROOT / cp, map_location=dev, weights_only=False)
        if SC["base_state_hash"] != h0:
            raise guard.GuardError("COARSE-16 checkpoint does not belong to this base")
        coarse = Cond("A", a.seed).to(dev)
        coarse.load_state_dict(SC["conds"]["A"])
        coarse.eval()
    want_common = not (cdir / f"common_{a.part}.json").exists()
    sp, rows, sem, shuf, reg = partition(a.part)
    if a.rows_limit:
        rows = rows[:a.rows_limit]
    papers = sorted({sp.papers[d] for r in rows for d in r})
    pidx = {p: i for i, p in enumerate(papers)}
    paper_of_doc = np.full(sp.n_docs, -1, np.int64)
    for r in rows:
        for d in r:
            paper_of_doc[d] = pidx[sp.papers[d]]
    lab_sem, lab_shuf = torch.as_tensor(sem, device=dev), torch.as_tensor(shuf, device=dev)
    if reg is not None:
        off_t = torch.as_tensor(reg[0], device=dev)
        lab_reg = torch.as_tensor(reg[1].argmax(1).astype(np.int64), device=dev)
    sel_extra = a.part != "TRAIN"
    cbuf = {k: [] for k in ("ce_base", "tgt", "pos", "doc", "paper", "doclen", "lab_sem", "lab_shuf", "lab_rsem",
                            "lab_rsem_prev")}
    bbuf = {c: {"d": [], "e_tr": [], "marg_tr": [], "pdiff": [], "lg": []} for c in names}
    kbuf = {"d": [], "e_tr": []}
    cons = {c: 0.0 for c in names + ["COARSE"]}
    tie_mis = {c: 0 for c in names}
    ce_tr = {c: 0.0 for c in names + ["COARSE"]}
    n_tot = 0
    with torch.no_grad():
        for i in range(0, len(rows), a.micro_rows):
            if mon is not None and mon.contaminated:
                sys.exit(77)
            bt = to_dev(make_batch(rows[i:i + a.micro_rows], sp), dev)
            x, hid = base_forward(model, bt)
            tgt = bt["targets"].reshape(-1)
            v = tgt >= 0
            nv = int(v.sum())
            n_tot += nv
            ce_b = head_ce(model, hid, tgt, from_hidden=True)
            xn = rms0(x)
            g_sem, g_shuf = token_groups(bt, lab_sem, lab_shuf)
            grp = {names[0]: g_sem, names[1]: g_shuf, names[2]: g_sem}
            gd = bt["gdoc"].reshape(-1)[v]
            pos = bt["positions"].reshape(-1)[v]
            if want_common:
                cbuf["ce_base"].append(ce_b[v].cpu().numpy())
                cbuf["tgt"].append(tgt[v].cpu().numpy().astype(np.int32))
                cbuf["pos"].append(pos.cpu().numpy().astype(np.int16))
                cbuf["doc"].append(gd.cpu().numpy().astype(np.int32))
                gdn = gd.cpu().numpy()
                cbuf["paper"].append(paper_of_doc[gdn].astype(np.int32))
                cbuf["doclen"].append(sp.lengths[gdn].astype(np.int16))
                cbuf["lab_sem"].append(g_sem[v].cpu().numpy().astype(np.int8))
                cbuf["lab_shuf"].append(g_shuf[v].cpu().numpy().astype(np.int8))
                if reg is not None:
                    r_cur = off_t[gd] + torch.div(pos, 256, rounding_mode="floor")
                    has_prev = pos >= 256
                    r_prev = torch.where(has_prev, r_cur - 1, r_cur)
                    cbuf["lab_rsem"].append(lab_reg[r_cur].cpu().numpy().astype(np.int8))
                    cbuf["lab_rsem_prev"].append(torch.where(has_prev, lab_reg[r_prev], torch.full_like(r_prev, -1))
                                                 .cpu().numpy().astype(np.int8))
            for c in names:
                b = banks[c]
                lg = b.logits(xn)
                al = b.allowed(grp[c])
                lgm = lg.masked_fill(~al, float("-inf"))
                top2 = lgm.topk(2, -1)
                delta, idx, _ = b(xn, grp[c], v)
                tie_mis[c] += int((idx[v] != top2.indices[:, 0][v]).sum())
                e_tr = idx
                d = torch.empty(nv, E, device=dev)
                for e in range(E):
                    d[:, e] = head_ce(model, x + b.one(xn, e), tgt)[v] - ce_b[v]
                ce_direct = head_ce(model, x + delta, tgt)[v]
                look = ce_b[v] + d.gather(1, e_tr[v].view(-1, 1)).squeeze(1)
                cons[c] = max(cons[c], float((ce_direct - look).abs().max()))
                if not bool((e_tr[v] // M == grp[c][v]).all()) and b.mode == "pair":
                    raise guard.GuardError(f"pair route left its cluster in bank {c}")
                ce_tr[c] += float(ce_direct.double().sum())
                bbuf[c]["d"].append(d.cpu().numpy().astype(np.float16))
                bbuf[c]["e_tr"].append(e_tr[v].cpu().numpy().astype(np.int8))
                bbuf[c]["marg_tr"].append((top2.values[:, 0] - top2.values[:, 1])[v].cpu().numpy().astype(np.float16))
                if sel_extra and b.mode == "pair":
                    if M == 2:
                        bbuf[c]["pdiff"].append((lg[:, 0::2] - lg[:, 1::2])[v].cpu().numpy().astype(np.float16))
                    else:
                        bbuf[c]["lg"].append(lg[v].cpu().numpy().astype(np.float16))
            if coarse is not None:
                d = torch.empty(nv, 16, device=dev)
                for e in range(16):
                    idx = torch.full((xn.shape[0],), e, dtype=torch.long, device=dev)
                    delta, _, _ = coarse(xn, idx, v)
                    d[:, e] = head_ce(model, x + delta, tgt)[v] - ce_b[v]
                delta, _, _ = coarse(xn, g_sem, v)
                ce_direct = head_ce(model, x + delta, tgt)[v]
                look = ce_b[v] + d.gather(1, g_sem[v].view(-1, 1)).squeeze(1)
                cons["COARSE"] = max(cons["COARSE"], float((ce_direct - look).abs().max()))
                ce_tr["COARSE"] += float(ce_direct.double().sum())
                kbuf["d"].append(d.cpu().numpy().astype(np.float16))
                kbuf["e_tr"].append(g_sem[v].cpu().numpy().astype(np.int8))
    h1 = state_hash(model)
    if h1 != h0:
        raise guard.GuardError("frozen base state changed during extraction")
    summ = {"seed": a.seed, "part": a.part, "ckpt": str(a.ckpt), "ckpt_step": S["step"], "ckpt_tokens": S["tokens"],
            "gen_cfg": G, "n_tokens": n_tot, "n_papers": len(papers), "base_state_hash": h0, "base_unchanged": True,
            "ce_train_path": {c: ce_tr[c] / n_tot for c in names}, "lookup_vs_direct_max_abs": cons, "route_topk_vs_argmax_disagreements": tie_mis,
            "rows_limit": a.rows_limit, "time": guard.now(), "sha256": {}}
    if want_common:
        arr = {k: np.concatenate(v_) for k, v_ in cbuf.items() if v_}
        arr["papers"] = np.asarray(papers)
        sha = save(cdir / f"common_{a.part}.npz", arr)
        guard.atomic_write_json(cdir / f"common_{a.part}.json", {"sha256": sha, "n_tokens": n_tot, "seed": a.seed,
                                                                 "ce_base": float(arr["ce_base"].astype(np.float64).mean()),
                                                                 "time": guard.now()}, mode=0o400)
    summ["common_sha256"] = json.loads((cdir / f"common_{a.part}.json").read_text())["sha256"]
    for c in names:
        arr = {k: np.concatenate(v_) for k, v_ in bbuf[c].items() if v_}
        summ["sha256"][f"bank_{c}_{a.part}.npz"] = save(out / f"bank_{c}_{a.part}.npz", arr)
    if coarse is not None:
        arr = {k: np.concatenate(v_) for k, v_ in kbuf.items()}
        sha = save(cdir / f"coarse_{a.part}.npz", arr)
        guard.atomic_write_json(cdir / f"coarse_summary_{a.part}.json",
                                {"sha256": sha, "ckpt": P.COARSE_CKPT[a.seed][0], "ce_oracle": ce_tr["COARSE"] / n_tot,
                                 "lookup_vs_direct_max_abs": cons["COARSE"], "time": guard.now()}, mode=0o400)
    guard.atomic_write_json(out / f"summary_{a.part}.json", summ, mode=0o400)
    print(json.dumps({k: v for k, v in summ.items() if k != "sha256"}), flush=True)


if __name__ == "__main__":
    main()
