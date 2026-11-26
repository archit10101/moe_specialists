#!/usr/bin/env python
"""scur1 extraction (copy of asrs2_eval.py with TRAIN / SEL / CS2 partitions, scur1 locks and a bank filter).
asrs2 evaluation of one milestone (Stage-0F frozen milestone ckpt_F or a continuation milestone) on a calibration
partition (guarded GPU job). SEL = CALIB-SEARCH half "SEL" (Stage 0F/1); CS2 = CALIB-SELECTION (only after
FROZEN_S1.json). Same rows, order and micro-batching for every milestone, so per-token CE is paired across milestones.

Writes into --out-dir (all 0400, sha256 in summary):
  common_<P>.npz   ce_base, tgt, pos, doc, paper, doclen, papers, lab_sem/lab_shuf (paper-level K=5 groups),
                   lab_rsem/lab_rshuf (region-prior argmax), rid
  bank_<A>_<P>.npz full K-specialist table d (N, 5) = CE_k - CE_base (fp16) for EVERY token; token-routed banks also
                   e (their own router's choice)
  dep_<A>_<P>.npz  causal-router deployment of arm A: e, q (N, 5) fp16
  summary_<P>.json mean CEs (base, every deployment, region teachers, oracles), switch rates, sha256
  asrs2_eval.py --ckpt RUN/ckpt_TAG --part SEL|CS2 --out-dir DIR [--rows-limit N]"""
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
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--part", default="SEL", choices=["TRAIN", "SEL", "CS2"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--rows-limit", type=int, default=0)
    ap.add_argument("--micro-rows", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--test", action="store_true", help="CPU tests: skip hash checks of unfrozen test milestones")
    ap.add_argument("--seeds", default="", help="evaluate only the arms of these specialist seeds (comma list)")
    ap.add_argument("--banks", default="", help="evaluate only these bank names (comma list)")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    if not guard.require_execution(a.execute, "gpu" if a.device == "cuda" else "cpu-model"):
        return
    mon = None
    if a.device == "cuda":
        guard.occupancy_gate("scur1 extract")
        mon = guard.OccupancyMonitor(interval=15.0)
        mon.start()
    guard.register_self("scur1 extract")
    t0 = time.time()
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
    print(json.dumps({"status": status, "seconds": round(time.time() - t0, 1)}), flush=True)
    sys.exit(rc)


def partition(part):
    """(split, rows, K=5 prior table dict, sha) for TRAIN (CALIB-SEARCH half EVAL), SEL (half SEL) or CS2
    (CALIB-SELECTION, only after scur1 FROZEN_A.json)."""
    import numpy as np
    from csr import asrs2_paths as P2
    from csr import scur1_paths as P
    from csr.fsp_lib import Split
    from csr.orv1_lib import unflatten_rows
    if part == "CS2":
        if not (P.ART / "FROZEN_A.json").exists():
            raise guard.GuardError("CALIB-SELECTION is locked until FROZEN_A.json exists")
        sp = Split("calib_sel")
        zp = np.load(guard.ROOT / "artifacts/csr-current-context-specialist-router-v1/cs2_assign/parts_cs2.npz")
        rows = unflatten_rows(zp["CS2_docs"], zp["CS2_row_len"])
        meta = json.loads((P.TABLES / "p5_calib_sel.json").read_text())
        p = P.TABLES / "p5_calib_sel.npz"
        if guard.sha256_file(p) != meta["sha256"]:
            raise guard.GuardError("p5_calib_sel hash mismatch")
        return sp, rows, dict(np.load(p)), meta["sha256"]
    sp = Split("calib_search")
    pz = guard.ROOT / "artifacts/oracle-residual-v1/tables/parts.npz"
    if guard.sha256_file(pz) != P.REUSED_FILES["artifacts/oracle-residual-v1/tables/parts.npz"]:
        raise guard.GuardError("parts.npz hash mismatch")
    zp = np.load(pz)
    half = {"TRAIN": "EVAL", "SEL": "SEL"}[part]
    rows = unflatten_rows(zp[f"{half}_docs"], zp[f"{half}_row_len"])
    tab, sha = P2.tables("calib_search")
    return sp, rows, tab, sha


def load(ckpt, dev, test=False):
    import torch
    from csr import asrs2_ckpt as CK  # verify_dir only (read-only)
    from csr import asrs2_paths as P
    from csr.asrs2_lib import Bank5
    from csr.ccsr1_router import CausalRouter
    from csr.lfr_model import LFRConfig, LFRModel
    p = guard.ROOT / ckpt
    if not test and not CK.verify_dir(p):
        raise guard.GuardError(f"milestone {p} fails verification")
    ck = torch.load(p / "state.pt", map_location="cpu", weights_only=False)
    K = int(ck["K"])
    if ck["mode"] == "frozen":
        bp = guard.ROOT / ck["base_from"]["path"] / "state.pt"
        if not test and guard.sha256_file(bp) != P.CKPT_SHA:
            raise guard.GuardError("frozen-stage base hash mismatch")
        b0 = torch.load(bp, map_location="cpu", weights_only=False)
        msd, mcfg = b0["model"], b0["model_cfg"]
    else:
        msd, mcfg = ck["model"], ck["model_cfg"]
    m = LFRModel(LFRConfig.from_dict({**mcfg, "teacher": False}), ck.get("base_seed", 100))
    m.load_state_dict(msd, strict=True)
    banks, routers = {}, {}
    for name, kind, s in ck["arms"]:
        routed = kind in ("G5", "G", "ALTG")
        b = Bank5(K, P.D_MODEL, P.FFN, seed=int(s), routed=routed, alpha=P.ALPHA_G)
        b.load_state_dict(ck["banks"][name])
        banks[name] = (b, kind, int(s))
        if name in ck["routers"]:
            r = CausalRouter(P.ROUTER_H, int(s), "base", K)
            r.load_state_dict(ck["routers"][name])
            routers[name] = r
    for mod in [m] + [b for b, _, _ in banks.values()] + list(routers.values()):
        mod.to(dev).eval()
        for q in mod.parameters():
            q.requires_grad_(False)
    meta = {"ckpt": ckpt, "mode": ck["mode"], "step": int(ck["step"]), "seen": int(ck.get("seen", 0)), "K": K,
            "arms": ck["arms"], "router_meta": ck["router_meta"], "n_upd": ck.get("n_upd")}
    return m, banks, routers, meta


def run(a, mon):
    import numpy as np
    import torch
    from csr.asrs2_lib import deltas, rms0
    from csr.orv1_lib import make_batch, to_dev
    from ccsr1_s3_train import doc_batch
    out = guard.inside_root(a.out_dir)
    part = a.part
    if (out / f"summary_{part}.json").exists():
        print("exists")
        return
    dev = guard.setup_cuda_device() if a.device == "cuda" else torch.device("cpu")
    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    model, banks, routers, meta = load(a.ckpt, dev, a.test)
    if a.banks:
        keepb = set(a.banks.split(","))
        banks = {n: v for n, v in banks.items() if n in keepb}
        routers = {n: r for n, r in routers.items() if n in banks}
        meta["arms"] = [x for x in meta["arms"] if x[0] in keepb]
    if a.seeds:
        keep = {int(x) for x in a.seeds.split(",")}
        banks = {n: v for n, v in banks.items() if v[2] in keep}
        routers = {n: r for n, r in routers.items() if n in banks}
        meta["arms"] = [x for x in meta["arms"] if int(x[2]) in keep]
        meta["seeds_evaluated"] = sorted(keep)
    K = meta["K"]
    W = model.tok_emb.weight.detach()
    sp, rows, tab, tab_sha = partition(part)
    if a.rows_limit:
        rows = rows[:a.rows_limit]
    papers = sorted({sp.papers[d] for r in rows for d in r})
    pidx = {p_: i for i, p_ in enumerate(papers)}
    paper_of_doc = np.full(sp.n_docs, -1, np.int64)
    for r in rows:
        for d in r:
            paper_of_doc[d] = pidx[sp.papers[d]]
    off = np.full(sp.n_docs, -1, np.int64)
    off[tab["docs"]] = tab["ptr"][:-1]
    g_sem = np.full(sp.n_docs, -1, np.int64)
    g_shuf = np.full(sp.n_docs, -1, np.int64)
    g_sem[tab["docs"]] = tab["sem5"]
    g_shuf[tab["docs"]] = tab["shuf5"]
    T = {k: torch.as_tensor(v, device=dev) for k, v in (("off", off), ("gsem", g_sem), ("gshuf", g_shuf),
                                                         ("pdoc", paper_of_doc), ("len", sp.lengths))}
    Psem = torch.as_tensor(tab["p"].astype(np.float32), device=dev)
    Pshuf = torch.as_tensor(tab["p_sh"].astype(np.float32), device=dev)
    ckeys = ["ce_base", "tgt", "pos", "doc", "paper", "doclen", "lab_sem", "lab_shuf", "lab_rsem", "lab_rshuf", "rid"]
    cbuf = {k: [] for k in ckeys}
    bbuf = {n: {"d": [], "e": []} for n in banks}
    dbuf = {n: {"e": [], "q": []} for n in routers}
    with torch.no_grad():
        for i in range(0, len(rows), a.micro_rows):
            if mon is not None and mon.contaminated:
                sys.exit(77)
            bt = to_dev(make_batch(rows[i:i + a.micro_rows], sp), dev)
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                o = model({"input_ids": bt["input_ids"], "doc_ids": bt["doc_ids"], "positions": bt["positions"]})
            dm = bt["doc_ids"].reshape(-1) >= 0
            H = o["hidden"].reshape(-1, o["hidden"].shape[-1])[dm]
            Y = bt["targets"].reshape(-1)[dm]
            gd = bt["gdoc"].reshape(-1)[dm]
            pos = bt["positions"].reshape(-1)[dm]
            sc = Y >= 0
            hs, ys, gds, pos_s = H[sc], Y[sc], gd[sc], pos[sc]
            xs = rms0(hs)
            if bool((T["off"][gds] < 0).any()):
                raise guard.GuardError("doc without p5 regions")
            rid = T["off"][gds] + torch.div(pos_s, 256, rounding_mode="floor")
            for k, v, dt in (("tgt", ys, np.int32), ("pos", pos_s, np.int16), ("doc", gds, np.int32),
                             ("paper", T["pdoc"][gds], np.int32), ("doclen", T["len"][gds], np.int16),
                             ("lab_sem", T["gsem"][gds], np.int8), ("lab_shuf", T["gshuf"][gds], np.int8),
                             ("lab_rsem", Psem[rid].argmax(1), np.int8), ("lab_rshuf", Pshuf[rid].argmax(1), np.int8),
                             ("rid", rid, np.int32)):
                cbuf[k].append(v.cpu().numpy().astype(dt))
            ceb = None
            for n, (b, kind, s) in banks.items():
                u, ceb = deltas(b.experts.w1, b.experts.w2, hs, xs, W, ys)
                bbuf[n]["d"].append((-u).half().cpu().numpy())
                if b.routed:
                    bbuf[n]["e"].append(b.probs(xs).argmax(1).cpu().numpy().astype(np.int8))
            if ceb is None:
                from csr.jsr_bank import head_ce_grad
                ceb, _ = head_ce_grad(hs, torch.zeros_like(hs, dtype=torch.float32), W, ys, want_grad=False)
            cbuf["ce_base"].append(ceb.cpu().numpy())
            idx, mask, lens = doc_batch(gds.long(), pos_s.long())
            X = xs.to(torch.float16)
            for n, r in routers.items():
                lg = r(X[idx], lens)[mask]
                ti = idx[mask]
                q = torch.empty_like(lg).index_copy_(0, ti, lg).float().softmax(-1)
                dbuf[n]["e"].append(q.argmax(1).cpu().numpy().astype(np.int8))
                dbuf[n]["q"].append(q.half().cpu().numpy())
    out.mkdir(parents=True, exist_ok=True)
    shas = {}

    def save(path, arrs):
        tmp = path.with_name(f".{path.stem}.tmp.npz")
        np.savez_compressed(tmp, **arrs)
        tmp.rename(path)
        path.chmod(0o400)
        shas[path.name] = guard.sha256_file(path)
    common = {"papers": np.asarray(papers), **{k: np.concatenate(v) for k, v in cbuf.items()}}
    save(out / f"common_{part}.npz", common)
    cb = common["ce_base"].astype(np.float64)
    same_doc = common["doc"][1:] == common["doc"][:-1]
    summ = {**meta, "part": part, "p5_sha256": tab_sha, "n_tokens": int(len(cb)), "n_papers": len(papers),
            "ce_base": float(cb.mean()), "banks": {}, "deps": {}, "rows_limit": a.rows_limit}
    ar = np.arange(len(cb))
    for n, (b, kind, s) in banks.items():
        A = {"d": np.concatenate(bbuf[n]["d"])}
        if bbuf[n]["e"]:
            A["e"] = np.concatenate(bbuf[n]["e"])
        save(out / f"bank_{n}_{part}.npz", A)
        d = A["d"].astype(np.float64)
        rec = {"kind": kind, "seed": s, "ce_oracle": float(cb.mean() + d.min(1).mean()),
               "mean_specialist_delta": float(d.mean()),
               "ce_T_sem": float(cb.mean() + d[ar, common["lab_rsem"]].mean()),
               "ce_T_shuf": float(cb.mean() + d[ar, common["lab_rshuf"]].mean())}
        if "e" in A:
            rec["ce_routed"] = float(cb.mean() + d[ar, A["e"].astype(np.int64)].mean())
        summ["banks"][n] = rec
    for n in routers:
        A = {k: np.concatenate(v) for k, v in dbuf[n].items()}
        save(out / f"dep_{n}_{part}.npz", A)
        d = np.load(out / f"bank_{n}_{part}.npz")["d"].astype(np.float64)
        e = A["e"].astype(np.int64)
        summ["deps"][n] = {"ce_R": float(cb.mean() + d[ar, e].mean()),
                           "switch_rate": float((e[1:] != e[:-1])[same_doc].mean())}
    summ["sha256"] = shas
    summ["time"] = guard.now()
    guard.atomic_write_json(out / f"summary_{part}.json", summ, mode=0o400)
    print(json.dumps({k: v for k, v in summ.items() if k not in ("sha256", "arms", "router_meta")}, default=str)[:3000], flush=True)


if __name__ == "__main__":
    main()
