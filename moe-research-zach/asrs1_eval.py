#!/usr/bin/env python
"""asrs1 evaluation of one checkpoint on CALIB-SEARCH SEL (prereg section 6; guarded GPU job).

Checkpoint formats (--format):
  cfar1   the step-318 start checkpoint (cycle 0 / frozen stage): banks G, L (SEM), F2h_T500_w240 (SHUF) deployed by the
          frozen ccsr1 Stage-1 routers R3_c2e0.5 SEM_s{r} / SHUF_s{r}
  ccsr1   the ordinary (simultaneous) continuation milestone m1111: G, SEMf_r, SEMj_r, SHUFf_r (each its own bank)
  asrs1   an alternating cycle milestone: ALTG, SEM + routers SEM_s{r}, SHUF + routers SHUF_s{r}
Same SEL rows, order, micro-batching (16 rows) and swap subset (u_hash("csur1-sub", paper) < 0.25) for every checkpoint,
so per-token CE is paired across checkpoints. Writes into --out-dir:
  common_SEL.npz        ce_base, labels (paper C6 sem/shuf, region-prior argmax sem/shuf), rid, paper, doc, pos, doclen,
                        tgt, blk, sub, dce_G / e_G (the checkpoint's token-routed generic bank)
  bank_<B>_SEL.npz      per teacher-assigned bank: dT/eT (region teacher), dPV/ePV (stale past-view student),
                        d_sub (swap subset x 16: CE_k - CE_base)
  dep_<D>_SEL.npz       per causal-router deployment: e (choice), qtop / margin (swap subset), dR (CE_choice - CE_base)
  summary_SEL.json      metadata, mean CEs, file sha256
  asrs1_eval.py --format F --ckpt PATH --out-dir DIR [--rows-limit N]"""
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
    ap.add_argument("--format", required=True, choices=["cfar1", "ccsr1", "asrs1"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--rows-limit", type=int, default=0)
    ap.add_argument("--micro-rows", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    if not guard.require_execution(a.execute, "gpu" if a.device == "cuda" else "cpu-model"):
        return
    mon = None
    if a.device == "cuda":
        guard.occupancy_gate("asrs1 eval")
        mon = guard.OccupancyMonitor(interval=15.0)
        mon.start()
    guard.register_self("asrs1 eval")
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


def load(fmt, ckpt, dev):
    """-> model, G bank, {bank name: FedBank}, [deployment dicts], meta."""
    import torch
    from csr import asrs1_ckpt as CK
    from csr import asrs1_paths as P
    from csr.cfar1_bank import FedBank
    from csr.cfar1_paths import ALPHA_G
    from csr.ccsr1_router import CausalRouter
    from csr.jsr_bank import Bank
    from csr.lfr_model import LFRConfig, LFRModel
    p = guard.ROOT / ckpt
    if fmt == "cfar1":
        if guard.sha256_file(p / "state.pt") != P.CKPT_SHA:
            raise guard.GuardError("start checkpoint hash mismatch")
    elif fmt == "ccsr1":
        if guard.sha256_file(p / "state.pt") != P.CCSR1_S3_SHA:
            raise guard.GuardError("ccsr1 m1111 hash mismatch")
    elif not CK.verify_dir(p):
        raise guard.GuardError(f"milestone {p} fails verification")
    ck = torch.load(p / "state.pt", map_location="cpu", weights_only=False)
    seed = ck.get("base_seed", 100)
    m = LFRModel(LFRConfig.from_dict({**ck["model_cfg"], "teacher": False}), seed)
    m.load_state_dict(ck["model"], strict=True)
    mkG = lambda: Bank({"name": "G", "kind": "routed", "tau": 1.0, "r": 0.0, "alpha": ALPHA_G, "groups": None}, seed)
    banks, deps, routers = {}, [], {}
    mkR = lambda cfg_, sd, s: (lambda r: (r.load_state_dict(sd), r)[1])(
        CausalRouter(int(cfg_["H"]), int(s), cfg_.get("feats", "base"), int(cfg_.get("n_out", 16))))
    if fmt == "cfar1":
        G = mkG()
        G.load_state_dict(ck["banks"][P.BANK_G])
        for B, src in (("SEM", P.BANK_SEM), ("SHUF", P.BANK_SHUF)):
            b = FedBank(ck["bank_specs"][src], seed)
            b.load_state_dict(ck["banks"][src])
            banks[B] = b
            for r in P.ROUTER_SEEDS:
                rp = guard.ROOT / P.ROUTER_DIR / f"{B}_s{r}.pt"
                if guard.sha256_file(rp) != P.ROUTER_SHA[f"{B}_s{r}"]:
                    raise guard.GuardError(f"router hash mismatch {rp}")
                rs = torch.load(rp, map_location="cpu", weights_only=False)
                routers[f"{B}_s{r}"] = mkR(rs["cfg"], rs["state"], rs["seed"])
                deps.append({"name": f"{B}_s{r}", "bank": B, "seed": r})
    elif fmt == "ccsr1":
        G = mkG()
        G.load_state_dict(ck["banks"]["G"])
        for arm in ck["arms"]:
            n = arm["name"]
            if arm["router"] == "token":
                continue
            b = FedBank(ck["bank_specs"][n], seed)
            b.load_state_dict(ck["banks"][n])
            banks[n] = b
            routers[n] = mkR(ck["router_meta"][n]["cfg"], ck["routers"][n], arm["seed"])
            deps.append({"name": n, "bank": n, "seed": arm["seed"]})
    else:
        G = mkG()
        G.load_state_dict(ck["banks"]["ALTG"])
        for B in ("SEM", "SHUF"):
            b = FedBank(ck["bank_specs"][B], seed)
            b.load_state_dict(ck["banks"][B])
            banks[B] = b
        for d in ck["deploy"]:
            if d["router"] == "token":
                continue
            rm = ck["router_meta"][d["router"]]
            routers[d["name"]] = mkR(rm["cfg"], ck["routers"][d["router"]], rm["seed"])
            deps.append({"name": d["name"], "bank": d["bank"], "seed": d["seed"]})
    mods = [m, G, *banks.values(), *routers.values()]
    for mod in mods:
        mod.to(dev).eval()
        for q in mod.parameters():
            q.requires_grad_(False)
    meta = {"format": fmt, "ckpt": ckpt, "step": int(ck["step"]), "seen": int(ck.get("seen", 0)),
            "cycle": ck.get("cycle"), "teacher_stats": ck.get("teacher_stats"), "n_done": ck.get("n_done"),
            "banks": {n: {"prior": b.prior, "spec": b.spec} for n, b in banks.items()}, "deployments": deps}
    return m, G, banks, routers, deps, meta


def run(a, mon):
    import numpy as np
    import torch
    from csr.csur1_bank import all_deltas, block_feats, layout
    from csr.jsr_bank import rms0
    from csr.orv1_lib import make_batch, to_dev
    from csr import asrs1_paths as P
    from orv1_tables import u_hash
    from cfar1_train import block_priors
    from ccsr1_extract import partition
    from ccsr1_s3_eval import ce_multi
    from ccsr1_s3_train import doc_batch
    out = guard.inside_root(a.out_dir)
    if (out / "summary_SEL.json").exists():
        print("exists")
        return
    P.verify_reused()
    dev = guard.setup_cuda_device() if a.device == "cuda" else torch.device("cpu")
    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    model, G, banks, routers, deps, meta = load(a.format, a.ckpt, dev)
    W = model.tok_emb.weight.detach()
    sp, rows, sem, shuf, (off, Pt, Psh) = partition("SEL")
    if a.rows_limit:
        rows = rows[:a.rows_limit]
    papers = sorted({sp.papers[d] for r in rows for d in r})
    pidx = {p_: i for i, p_ in enumerate(papers)}
    paper_of_doc = np.full(sp.n_docs, -1, np.int64)
    for r in rows:
        for d in r:
            paper_of_doc[d] = pidx[sp.papers[d]]
    sub_paper = np.asarray([u_hash("csur1-sub", p_) < P.SUB_FRAC for p_ in papers])
    sem_t, shuf_t = torch.as_tensor(sem, device=dev), torch.as_tensor(shuf, device=dev)
    psub_t, pdoc_t = torch.as_tensor(sub_paper, device=dev), torch.as_tensor(paper_of_doc, device=dev)
    off_t, P_t, Psh_t = (torch.as_tensor(x, device=dev) for x in (off, Pt, Psh))
    len_t = torch.as_tensor(sp.lengths, device=dev)
    ckeys = ["ce_base", "lab_sem", "lab_shuf", "lab_rsem", "lab_rshuf", "rid", "paper", "doc", "pos", "doclen", "tgt",
             "blk", "sub", "dce_G", "e_G"]
    cbuf = {k: [] for k in ckeys}
    bbuf = {n: {k: [] for k in ("dT", "eT", "dPV", "ePV", "d_sub")} for n in banks}
    dbuf = {d["name"]: {k: [] for k in ("e", "qtop", "margin", "dR")} for d in deps}
    consist = 0.0
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
            lay = layout(rowt, bt["doc_ids"].reshape(-1)[dm], pos, sc)
            xn_all = rms0(H)
            s_blk = block_feats(xn_all, lay["blk"], lay["nb"])
            lp_sem, lp_shuf = block_priors(lay, gd, off_t, P_t, Psh_t)
            hs, ys, xs, blk_s = H[sc], Y[sc], xn_all[sc], lay["blk"][sc]
            gds, pos_s = gd[sc], pos[sc]
            rid = off_t[gds] + torch.div(pos_s, 256, rounding_mode="floor")
            pp = pdoc_t[gds]
            sub = psub_t[pp]
            for k, v in (("lab_sem", sem_t[gds]), ("lab_shuf", shuf_t[gds]), ("lab_rsem", lp_sem.argmax(1)[blk_s]),
                         ("lab_rshuf", lp_shuf.argmax(1)[blk_s]), ("blk", lay["blk_no"][blk_s])):
                cbuf[k].append(v.cpu().numpy().astype(np.int8))
            cbuf["rid"].append(rid.cpu().numpy().astype(np.int32))
            cbuf["paper"].append(pp.cpu().numpy().astype(np.int32))
            cbuf["doc"].append(gds.cpu().numpy().astype(np.int32))
            cbuf["pos"].append(pos_s.cpu().numpy().astype(np.int16))
            cbuf["doclen"].append(len_t[gds].cpu().numpy().astype(np.int16))
            cbuf["tgt"].append(ys.cpu().numpy().astype(np.int32))
            cbuf["sub"].append(sub.cpu().numpy())
            idx, mask, lens = doc_batch(gds.long(), pos_s.long())
            X = xs.to(torch.float16)
            eg = G.probs(xs).argmax(1)
            corrs, tags = [G.correction(xs, eg, torch.ones_like(eg, dtype=torch.float32))], [("G", "G")]
            for n, b in banks.items():
                epv = b.student_logits(s_blk, lay).float().argmax(1)[blk_s]
                eT = {"sem": lp_sem, "shuf": lp_shuf}[b.prior].argmax(1)[blk_s]
                bbuf[n]["ePV"].append(epv.cpu().numpy().astype(np.int8))
                bbuf[n]["eT"].append(eT.cpu().numpy().astype(np.int8))
                corrs += [b.correction(xs, epv), b.correction(xs, eT)]
                tags += [("bank", n, "dPV"), ("bank", n, "dT")]
            ch = {}
            for d in deps:
                n = d["name"]
                lg = routers[n](X[idx], lens)[mask]
                ti = idx[mask]
                q = torch.empty_like(lg).index_copy_(0, ti, lg).float().softmax(-1)
                t2 = q.topk(2, -1)
                e = t2.indices[:, 0]
                ch[n] = e
                db = dbuf[n]
                db["e"].append(e.cpu().numpy().astype(np.int8))
                db["qtop"].append(t2.values[:, 0][sub].half().cpu().numpy())
                db["margin"].append((t2.values[:, 0] - t2.values[:, 1])[sub].half().cpu().numpy())
                corrs.append(banks[d["bank"]].correction(xs, e))
                tags.append(("dep", n, "dR"))
            ce_b, cex = ce_multi(hs, W, ys, corrs)
            del corrs
            cbuf["ce_base"].append(ce_b.cpu().numpy())
            for tg, c_ in zip(tags, cex):
                v = (c_ - ce_b).cpu().numpy().astype(np.float16)
                if tg[0] == "G":
                    cbuf["dce_G"].append(v)
                    cbuf["e_G"].append(eg.cpu().numpy().astype(np.int8))
                elif tg[0] == "bank":
                    bbuf[tg[1]][tg[2]].append(v)
                else:
                    dbuf[tg[1]]["dR"].append(v)
            for n, b in banks.items():
                if bool(sub.any()):
                    u, _ = all_deltas(b.experts.w1, b.experts.w2, hs[sub], xs[sub], W, ys[sub])
                    bbuf[n]["d_sub"].append((-u).cpu().numpy().astype(np.float16))
                else:
                    bbuf[n]["d_sub"].append(np.zeros((0, 16), np.float16))
            if bool(sub.any()):
                for d in deps:
                    dsub = torch.as_tensor(bbuf[d["bank"]]["d_sub"][-1], device=dev).float()
                    dR_sub = torch.as_tensor(dbuf[d["name"]]["dR"][-1], device=dev)[sub].float()
                    consist = max(consist, float((dsub.gather(1, ch[d["name"]][sub].view(-1, 1)).squeeze(1) - dR_sub).abs().max()))
    out.mkdir(parents=True, exist_ok=True)
    shas = {}

    def save(path, arrs):
        tmp = path.with_name(f".{path.stem}.tmp.npz")
        np.savez_compressed(tmp, **arrs)
        tmp.rename(path)
        path.chmod(0o400)
        shas[path.name] = guard.sha256_file(path)
    common = {"papers": np.asarray(papers), **{k: np.concatenate(v) for k, v in cbuf.items()}}
    save(out / "common_SEL.npz", common)
    cb = common["ce_base"].astype(np.float64)
    same_doc = common["doc"][1:] == common["doc"][:-1]
    summ = {**meta, "n_tokens": int(len(cb)), "n_papers": len(papers), "n_sub_tokens": int(common["sub"].sum()),
            "ce_base": float(cb.mean()), "ce_G": float(cb.mean() + common["dce_G"].astype(np.float64).mean()),
            "banks_ce": {}, "deps_ce": {}, "dsub_vs_routed_max_abs": consist, "rows_limit": a.rows_limit}
    for n in banks:
        A = {k: np.concatenate(v) for k, v in bbuf[n].items()}
        save(out / f"bank_{n}_SEL.npz", A)
        summ["banks_ce"][n] = {"ce_T": float(cb.mean() + A["dT"].astype(np.float64).mean()),
                               "ce_PV": float(cb.mean() + A["dPV"].astype(np.float64).mean())}
    for d in deps:
        A = {k: np.concatenate(v) for k, v in dbuf[d["name"]].items()}
        save(out / f"dep_{d['name']}_SEL.npz", A)
        summ["deps_ce"][d["name"]] = {"ce_R": float(cb.mean() + A["dR"].astype(np.float64).mean()),
                                      "switch_rate": float((A["e"][1:] != A["e"][:-1])[same_doc].mean())}
    summ["sha256"] = shas
    summ["time"] = guard.now()
    guard.atomic_write_json(out / "summary_SEL.json", summ, mode=0o400)
    print(json.dumps({k: v for k, v in summ.items() if k not in ("sha256", "banks")}, default=str)[:3000], flush=True)


if __name__ == "__main__":
    main()
