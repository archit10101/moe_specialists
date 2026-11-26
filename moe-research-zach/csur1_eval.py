#!/usr/bin/env python
"""csr-crossfit-semantic-utility-residual-v1 evaluation (guarded GPU job; CPU for tests).

  csur1_eval.py --ckpt RUN/ckpt_TAG --part SEL|EVAL|VAL --out NPZ [--rows-limit N]
      joint checkpoint (shared base + banks). Per scored token: ce_base (fp32; internal TOKEN-4 replicate),
      lab_sem / lab_shuf (paper-level C6 / orv1 shuffle), lab_rsem (argmax region p_sem), paper, pos, doclen, tgt,
      domain (post hoc only), blk (view number in the document), sub.
      Per bank X:  dce_X (fp16) = CE(student / deployable path) - ce_base, e_X (int8) student specialist,
      qtop_X, margin_X (fp16) student router probability of e and top1-top2 margin;
      util banks also dceT_X, eT_X: training-only TEACHER path (argmax of the cross-fitted responsibility r: partner
      view utility with the EMA specialists + region prior + frozen usage prior).
      On the swap subset (SEL/EVAL: papers with u_hash < SUB_FRAC[part]; VAL likewise): d_X (N_sub, 16) fp16 = CE with
      LIVE specialist j forced - ce_base, q_X (N_sub, 16) fp16 student probabilities.
      blkstats_X (8, 4): per view number [n views, sum KL(r || q), sum 1[argmax q == argmax r], sum entropy(q)].
  csur1_eval.py --ref RUN --part VAL --out NPZ
      reused base-only model (TOKEN-4 / EDL p100): per-paper CE sums (all, first 128, 128-511, 512+)."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402

SUB_FRAC = {"SEL": 0.25, "EVAL": 1.0, "VAL": 0.15}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--ref")
    ap.add_argument("--part", choices=["SEL", "EVAL", "VAL"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows-limit", type=int, default=0)
    ap.add_argument("--micro-rows", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    if not guard.require_execution(a.execute, "gpu" if a.device == "cuda" else "cpu-model"):
        return
    from csr.csur1_paths import ART
    if not a.rows_limit:
        from csr.csur1_paths import check_prereg
        check_prereg()
    if a.part == "VAL" and not (ART / "FROZEN_FINAL.json").exists():
        raise guard.GuardError("VAL is locked until FROZEN_FINAL.json exists")
    mon = None
    if a.device == "cuda":
        guard.occupancy_gate(f"csur1 eval {Path(a.out).name}")
        mon = guard.OccupancyMonitor(interval=15.0)
        mon.start()
    guard.register_self(f"csur1 eval {Path(a.out).name}")
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
    import numpy as np
    from csr.csur1_paths import ORV1, TABLES
    from csr.fsp_lib import Split
    from csr.orv1_lib import unflatten_rows
    if part == "VAL":
        sp = Split("val")
        zl = np.load(ORV1 / "val_assign/labels_val.npz")
        zp = np.load(ORV1 / "val_assign/parts_val.npz")
        rows = unflatten_rows(zp["VAL_docs"], zp["VAL_row_len"])
        tab = "psem_val"
    else:
        sp = Split("calib_search")
        zl = np.load(ORV1 / "tables/labels_cs.npz")
        zp = np.load(ORV1 / "tables/parts.npz")
        rows = unflatten_rows(zp[f"{part}_docs"], zp[f"{part}_row_len"])
        tab = "psem_calib_search"
    z = np.load(TABLES / f"{tab}.npz")
    meta = json.loads((TABLES / f"{tab}.json").read_text())
    if guard.sha256_file(TABLES / f"{tab}.npz") != meta["sha256"]:
        raise guard.GuardError(f"{tab} hash mismatch")
    msh = json.loads((TABLES / f"{tab}_sh.json").read_text())
    if guard.sha256_file(TABLES / f"{tab}_sh.npz") != msh["sha256"] or msh["source_psem_sha256"] != meta["sha256"]:
        raise guard.GuardError(f"{tab}_sh hash mismatch")
    zs = np.load(TABLES / f"{tab}_sh.npz")
    off = np.full(sp.n_docs, -1, np.int64)
    off[z["docs"]] = z["ptr"][:-1]
    return sp, rows, zl["sem"].astype(np.int64), zl["shuf"].astype(np.int64), (off, z["p"].astype(np.float32),
                                                                                zs["p_sh"].astype(np.float32))


def run(a, mon):
    import numpy as np
    import torch
    from csr.csur1_bank import UtilBank, all_deltas, block_feats, block_mean, ce_pair, layout
    from csr.csur1_paths import K
    from csr.jsr_bank import Bank, head_ce_grad, rms0
    from csr.orv1_lib import make_batch, to_dev
    from orv1_tables import u_hash
    from fle_evaluate import load_milestone
    from csur1_train import block_priors
    out = Path(a.out)
    if out.exists():
        print("exists")
        return
    dev = guard.setup_cuda_device() if a.device == "cuda" else torch.device("cpu")
    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    path = a.ckpt
    if a.ref:
        st = json.loads((guard.ROOT / a.ref / "status.json").read_text())
        ckinfo = st["ckpts"]["p100"]
        if guard.sha256_file(guard.ROOT / ckinfo["path"] / "state.pt") != ckinfo["sha256"]:
            raise guard.GuardError("reference checkpoint hash mismatch")
        path = ckinfo["path"]
    model, ck = load_milestone(path, dev)
    for p in model.parameters():
        p.requires_grad_(False)
    W = model.tok_emb.weight.detach()
    banks = {}
    if not a.ref:
        from csr.csur1_paths import ALPHA_G
        for n, spec in ck["bank_specs"].items():
            if spec.get("kind") == "util":
                b = UtilBank(spec, ck["config"]["seed"])
            else:
                b = Bank({"name": n, "kind": "routed", "tau": 1.0, "r": 0.0, "alpha": ALPHA_G, "groups": None},
                         ck["config"]["seed"])
            b = b.to(dev)
            b.load_state_dict(ck["banks"][n])
            b.eval()
            for p in b.parameters():
                p.requires_grad_(False)
            banks[n] = b
    sp, rows, sem, shuf, (off, Pt, Psh) = partition(a.part)
    if a.rows_limit:
        rows = rows[:a.rows_limit]
    papers = sorted({sp.papers[d] for r in rows for d in r})
    pidx = {p: i for i, p in enumerate(papers)}
    paper_of_doc = np.full(sp.n_docs, -1, np.int64)
    for r in rows:
        for d in r:
            paper_of_doc[d] = pidx[sp.papers[d]]
    sub_paper = np.asarray([u_hash("csur1-sub", p) < SUB_FRAC[a.part] for p in papers])
    sem_t, shuf_t = torch.as_tensor(sem, device=dev), torch.as_tensor(shuf, device=dev)
    psub_t = torch.as_tensor(sub_paper, device=dev)
    pdoc_t = torch.as_tensor(paper_of_doc, device=dev)
    off_t, P_t, Psh_t = (torch.as_tensor(x, device=dev) for x in (off, Pt, Psh))
    len_t = torch.as_tensor(sp.lengths, device=dev)
    dom_t = torch.as_tensor(np.asarray(sp.domains), device=dev)
    NP = len(papers)
    if a.ref:
        acc = np.zeros((NP, 8))
    buf = {k: [] for k in ("ce_base", "lab_sem", "lab_shuf", "lab_rsem", "paper", "doc", "pos", "doclen", "tgt",
                           "domain", "blk", "sub")}
    blkstats = {}
    for n, b in banks.items():
        for k in ("dce", "e", "qtop", "margin", "d", "q"):
            buf[f"{k}_{n}"] = []
        if isinstance(b, UtilBank):
            buf[f"dceT_{n}"], buf[f"eT_{n}"] = [], []
            blkstats[n] = np.zeros((8, 4))
    anorm = {n: [0.0, 0] for n in banks}
    consist = {n: 0.0 for n in banks}
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
            rowt = torch.arange(bt["doc_ids"].shape[0], device=dev).view(-1, 1).expand_as(bt["doc_ids"]).reshape(-1)[dm]
            sc = Y >= 0
            if a.ref:
                hs, ys = H[sc], Y[sc]
                ce_b, _ = head_ce_grad(hs, torch.zeros_like(hs, dtype=torch.float32), W, ys, want_grad=False)
                pp = pdoc_t[gd[sc]].cpu().numpy()
                c = ce_b.double().cpu().numpy()
                ps = pos[sc].cpu().numpy()
                for j_, m_ in enumerate((np.ones_like(ps, bool), ps < 128, (ps >= 128) & (ps < 512), ps >= 512)):
                    np.add.at(acc[:, 2 * j_], pp[m_], c[m_])
                    np.add.at(acc[:, 2 * j_ + 1], pp[m_], 1)
                continue
            lay = layout(rowt, bt["doc_ids"].reshape(-1)[dm], pos, sc)
            xn_all = rms0(H)
            s_blk = block_feats(xn_all, lay["blk"], lay["nb"])
            lp_sem, lp_shuf = block_priors(lay, gd, off_t, P_t, Psh_t)
            hs, ys, xs, blk_s = H[sc], Y[sc], xn_all[sc], lay["blk"][sc]
            gds = gd[sc]
            pp = pdoc_t[gds]
            sub = psub_t[pp]
            ce_b, _ = ce_pair(hs, None, W, ys)
            buf["ce_base"].append(ce_b.cpu().numpy())
            buf["lab_sem"].append(sem_t[gds].cpu().numpy().astype(np.int8))
            buf["lab_shuf"].append(shuf_t[gds].cpu().numpy().astype(np.int8))
            buf["lab_rsem"].append(lp_sem.argmax(1)[blk_s].cpu().numpy().astype(np.int8))
            buf["paper"].append(pp.cpu().numpy().astype(np.int32))
            buf["doc"].append(gds.cpu().numpy().astype(np.int32))
            buf["pos"].append(pos[sc].cpu().numpy().astype(np.int16))
            buf["doclen"].append(len_t[gds].cpu().numpy().astype(np.int16))
            buf["tgt"].append(ys.cpu().numpy().astype(np.int32))
            buf["domain"].append(dom_t[gds].cpu().numpy().astype(np.int8))
            buf["blk"].append(lay["blk_no"][blk_s].cpu().numpy().astype(np.int8))
            buf["sub"].append(sub.cpu().numpy())
            for n, b in banks.items():
                if isinstance(b, UtilBank):
                    lg = b.student_logits(s_blk, lay).float()
                    qb = torch.softmax(lg, -1)
                    q = qb[blk_s]
                    e = q.argmax(1)
                    u = None
                    if b.util:
                        dd, _ = b.token_deltas(hs, xs, W, ys, use_ema=True)
                        u = b.block_utility(dd, blk_s, lay["nb"])
                        uw = torch.zeros(lay["nb"], device=dev).index_add_(0, blk_s, torch.ones_like(ys, dtype=torch.float32))
                        del dd
                    lp = {"sem": lp_sem, "shuf": lp_shuf}.get(b.prior)
                    r, _ = b.responsibilities(u, lay["partner"], lp, update=False)
                    eT = r.argmax(1)[blk_s]
                    aT = b.correction(xs, eT)
                    _, ceT = ce_pair(hs, aT, W, ys)
                    buf[f"dceT_{n}"].append((ceT - ce_b).cpu().numpy().astype(np.float16))
                    buf[f"eT_{n}"].append(eT.cpu().numpy().astype(np.int8))
                    kl = (r * (torch.log(r.clamp(min=1e-12)) - torch.log(qb.clamp(min=1e-12)))).sum(1)
                    ag = (r.argmax(1) == qb.argmax(1)).float()
                    ent = -(qb * torch.log(qb.clamp(min=1e-12))).sum(1)
                    has = lay["n_scored"] > 0
                    bn = lay["blk_no"].clamp(max=7)
                    for j_, v_ in enumerate((torch.ones_like(kl), kl, ag, ent)):
                        blkstats[n][:, j_] += torch.zeros(8, device=dev).index_add_(0, bn[has], v_[has]).cpu().numpy()
                else:
                    q = b.probs(xs)
                    e = q.argmax(1)
                acor = b.correction(xs, e) if isinstance(b, UtilBank) else b.correction(xs, e, torch.ones_like(e, dtype=torch.float32))
                _, ce_x = ce_pair(hs, acor, W, ys)
                anorm[n][0] += float(acor.norm(dim=1).sum())
                anorm[n][1] += int(acor.shape[0])
                buf[f"dce_{n}"].append((ce_x - ce_b).cpu().numpy().astype(np.float16))
                buf[f"e_{n}"].append(e.cpu().numpy().astype(np.int8))
                t2 = q.topk(2, -1).values
                buf[f"qtop_{n}"].append(t2[:, 0].cpu().numpy().astype(np.float16))
                buf[f"margin_{n}"].append((t2[:, 0] - t2[:, 1]).cpu().numpy().astype(np.float16))
                if bool(sub.any()):
                    dl, _ = all_deltas(b.experts.w1, b.experts.w2, hs[sub], xs[sub], W, ys[sub])
                    d = -dl  # CE with specialist j forced - CE_base
                    consist[n] = max(consist[n], float((d.gather(1, e[sub].view(-1, 1)).squeeze(1) - (ce_x - ce_b)[sub]).abs().max()))
                    buf[f"d_{n}"].append(d.cpu().numpy().astype(np.float16))
                    buf[f"q_{n}"].append(q[sub].cpu().numpy().astype(np.float16))
    if a.ref:
        res = {"papers": np.asarray(papers), "acc": acc}
        summ = {"ref": a.ref, "ckpt": path, "part": a.part, "ce": float(acc[:, 0].sum() / acc[:, 1].sum()),
                "ce_0_127": float(acc[:, 2].sum() / acc[:, 3].sum()), "ce_128_511": float(acc[:, 4].sum() / acc[:, 5].sum()),
                "ce_512p": float(acc[:, 6].sum() / acc[:, 7].sum()), "n_tokens": int(acc[:, 1].sum()), "n_papers": NP}
    else:
        res = {"papers": np.asarray(papers)}
        for k, v_ in buf.items():
            if v_:
                res[k] = np.concatenate(v_)
        for n in blkstats:
            res[f"blkstats_{n}"] = blkstats[n]
        res["bank_names"] = np.asarray(list(banks))
        res["corrcos"] = np.stack([corrcos(b, res, dev) for b in banks.values()]) if banks else np.zeros(0)
        cb = res["ce_base"].astype(np.float64)
        n_tok = len(cb)
        summ = {"ckpt": str(a.ckpt), "step": ck["step"], "seen": ck["seen"], "part": a.part, "n_tokens": int(n_tok),
                "n_papers": NP, "n_sub_tokens": int(res["sub"].sum()), "ce_base": float(cb.mean()),
                **{f"ce_{n}": float(cb.mean() + res[f"dce_{n}"].astype(np.float64).mean()) for n in banks},
                **{f"ceT_{n}": float(cb.mean() + res[f"dceT_{n}"].astype(np.float64).mean()) for n in blkstats},
                "load": {n: (np.bincount(res[f"e_{n}"].astype(np.int64), minlength=K) / n_tok).round(5).tolist() for n in banks},
                "anorm": {n: anorm[n][0] / max(1, anorm[n][1]) for n in banks},
                "forced_vs_routed_max_abs": consist,
                "blkstats": {n: blkstats[n].tolist() for n in blkstats}}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.stem}.tmp.npz")
    np.savez(tmp, **res)
    tmp.rename(out)
    out.chmod(0o400)
    summ["sha256"] = guard.sha256_file(out)
    summ["bytes"] = out.stat().st_size
    summ["time"] = guard.now()
    guard.atomic_write_json(out.with_suffix(".json"), summ, mode=0o400)
    print(json.dumps({k: v for k, v in summ.items() if k not in ("blkstats",)}), flush=True)


def corrcos(b, res, dev):
    """Mean pairwise cosine (K x K) between specialists' forced corrections on a fixed sample of 2048 base states is not
    available post hoc; use the forced CE deltas instead: correlation across subset tokens of d[:, j] columns."""
    import numpy as np
    n = b.name if hasattr(b, "name") else None
    d = res.get(f"d_{n}")
    if d is None or len(d) < 10:
        return np.full((16, 16), np.nan)
    x = d[:200000].astype(np.float64)
    return np.corrcoef(x.T)


if __name__ == "__main__":
    main()
