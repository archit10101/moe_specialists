#!/usr/bin/env python
"""ccsr1 Stage-3 evaluation of a continuation milestone on CALIB-SELECTION (amendment section B; guarded GPU job).

Continued base + the milestone's banks and routers; same CS2 rows, order, micro-batching and swap subset as the Stage-2
tables. Writes into --out-dir:
  common_CS2.npz   ce_base, labels, rid, paper, doc, pos, doclen, tgt, domain, blk, sub, dce_G, e_G (continued G)
  arm_<name>_CS2.npz (every SEM / SHUF arm): dR (deployed-router CE - CE_base, every token), dPV/ePV (stale past-view
                   student), dT/eT (region teacher), d_sub (swap-subset tokens x 16: CE_k - CE_base)
and router-format choice files for the Stage-2 analysis code under --router-dir/<SEMf|SEMj>/{SEM,SHUF}_s<r>_CS2.npz
(+ json; SHUF = the matched SHUFf_r arm).
  ccsr1_s3_eval.py --ckpt RUN/ckpt_m1111 --out-dir DIR --router-dir DIR [--rows-limit N]"""
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
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--router-dir", required=True)
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
        guard.occupancy_gate("ccsr1 s3 eval")
        mon = guard.OccupancyMonitor(interval=15.0)
        mon.start()
    guard.register_self("ccsr1 s3 eval")
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


def ce_multi(h, W, y, corrs, chunk=2048):
    """CE_base and CE(base + a) for every correction in corrs (list of (N, d) fp32), sharing the base logits; same
    numerical path as csur1_bank.ce_pair (bf16 matmuls under autocast, fp32 log-sum-exp)."""
    import torch
    import torch.nn.functional as F
    from csr.csur1_bank import _ac
    N = h.shape[0]
    ceb = torch.empty(N, device=h.device)
    cex = [torch.empty(N, device=h.device) for _ in corrs]
    for s in range(0, N, chunk):
        hh, yy = h[s:s + chunk], y[s:s + chunk]
        with _ac(h.device):
            zb = F.linear(hh, W).float()
        lse_b = torch.logsumexp(zb, -1)
        zy = zb.gather(1, yy.view(-1, 1)).squeeze(1)
        ceb[s:s + chunk] = lse_b - zy
        for j, a in enumerate(corrs):
            with _ac(h.device):
                dl = F.linear(a[s:s + chunk], W).float()
            cex[j][s:s + chunk] = torch.logsumexp(zb + dl, -1) - zy - dl.gather(1, yy.view(-1, 1)).squeeze(1)
            del dl
        del zb
    return ceb, cex


def load_s3(ckdir, dev):
    import torch
    from csr.cfar1_bank import FedBank
    from csr.cfar1_paths import ALPHA_G
    from csr.ccsr1_router import CausalRouter
    from csr.jsr_bank import Bank
    from csr.lfr_model import LFRConfig, LFRModel
    from csr import ccsr1_ckpt as CK
    p = guard.ROOT / ckdir
    if not CK.verify_dir(p):
        raise guard.GuardError(f"milestone {p} fails verification")
    ck = torch.load(p / "state.pt", map_location="cpu", weights_only=False)
    m = LFRModel(LFRConfig.from_dict({**ck["model_cfg"], "teacher": False}), ck["base_seed"])
    m.load_state_dict(ck["model"], strict=True)
    m = m.to(dev).eval()
    banks, routers = {}, {}
    for arm in ck["arms"]:
        n = arm["name"]
        if arm["router"] == "token":
            b = Bank({"name": arm["init"], "kind": "routed", "tau": 1.0, "r": 0.0, "alpha": ALPHA_G, "groups": None},
                     ck["base_seed"])
        else:
            b = FedBank(ck["bank_specs"][n], ck["base_seed"])
            rm = ck["router_meta"][n]["cfg"]
            r = CausalRouter(int(rm["H"]), int(arm["seed"]), rm.get("feats", "base"), int(rm.get("n_out", 16)))
            r.load_state_dict(ck["routers"][n])
            routers[n] = r.to(dev).eval()
        b.load_state_dict(ck["banks"][n])
        banks[n] = b.to(dev).eval()
    for mod in [m, *banks.values(), *routers.values()]:
        for q in mod.parameters():
            q.requires_grad_(False)
    return m, banks, routers, ck


def run(a, mon):
    import numpy as np
    import torch
    from csr.csur1_bank import all_deltas, block_feats, layout
    from csr.ccsr1_paths import SUB_FRAC
    from csr.jsr_bank import rms0
    from csr.orv1_lib import make_batch, to_dev
    from orv1_tables import u_hash
    from cfar1_train import block_priors
    from ccsr1_extract import partition, verify_reused
    from ccsr1_s3_train import doc_batch
    out = Path(a.out_dir)
    rdir = Path(a.router_dir)
    if (out / "summary_CS2.json").exists():
        print("exists")
        return
    verify_reused()
    dev = guard.setup_cuda_device() if a.device == "cuda" else torch.device("cpu")
    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    model, banks, routers, ck = load_s3(a.ckpt, dev)
    W = model.tok_emb.weight.detach()
    arms = [x for x in ck["arms"] if x["router"] != "token"]
    sp, rows, sem, shuf, (off, Pt, Psh) = partition("CS2")
    if a.rows_limit:
        rows = rows[:a.rows_limit]
    papers = sorted({sp.papers[d] for r in rows for d in r})
    pidx = {p: i for i, p in enumerate(papers)}
    paper_of_doc = np.full(sp.n_docs, -1, np.int64)
    for r in rows:
        for d in r:
            paper_of_doc[d] = pidx[sp.papers[d]]
    sub_paper = np.asarray([u_hash("csur1-sub", p) < SUB_FRAC["CS2"] for p in papers])
    sem_t, shuf_t = torch.as_tensor(sem, device=dev), torch.as_tensor(shuf, device=dev)
    psub_t, pdoc_t = torch.as_tensor(sub_paper, device=dev), torch.as_tensor(paper_of_doc, device=dev)
    off_t, P_t, Psh_t = (torch.as_tensor(x, device=dev) for x in (off, Pt, Psh))
    len_t = torch.as_tensor(sp.lengths, device=dev)
    dom_t = torch.as_tensor(np.asarray(sp.domains), device=dev)
    ckeys = ["ce_base", "lab_sem", "lab_shuf", "lab_rsem", "lab_rshuf", "rid", "paper", "doc", "pos", "doclen", "tgt",
             "domain", "blk", "sub", "dce_G", "e_G"]
    cbuf = {k: [] for k in ckeys}
    abuf = {x["name"]: {k: [] for k in ("e", "qtop", "margin", "dR", "dPV", "ePV", "dT", "eT", "d_sub")} for x in arms}
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
                         ("lab_rshuf", lp_shuf.argmax(1)[blk_s]), ("blk", lay["blk_no"][blk_s]), ("domain", dom_t[gds])):
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
            g = banks["G"]
            eg = g.probs(xs).argmax(1)
            corrs, tags = [g.correction(xs, eg, torch.ones_like(eg, dtype=torch.float32))], [("G", None)]
            ch = {}
            for x in arms:
                n = x["name"]
                lg = routers[n](X[idx], lens)[mask]
                ti = idx[mask]
                q = torch.empty_like(lg).index_copy_(0, ti, lg).float().softmax(-1)
                t2 = q.topk(2, -1)
                e = t2.indices[:, 0]
                ab = abuf[n]
                ab["e"].append(e.cpu().numpy().astype(np.int8))
                ab["qtop"].append(t2.values[:, 0][sub].half().cpu().numpy())
                ab["margin"].append((t2.values[:, 0] - t2.values[:, 1])[sub].half().cpu().numpy())
                b = banks[n]
                epv = b.student_logits(s_blk, lay).float().argmax(1)[blk_s]
                eT = {"sem": lp_sem, "shuf": lp_shuf}[b.prior].argmax(1)[blk_s]
                ab["ePV"].append(epv.cpu().numpy().astype(np.int8))
                ab["eT"].append(eT.cpu().numpy().astype(np.int8))
                ch[n] = e
                for nm, e_ in (("dR", e), ("dPV", epv), ("dT", eT)):
                    corrs.append(b.correction(xs, e_))
                    tags.append((n, nm))
            ce_b, cex = ce_multi(hs, W, ys, corrs)
            del corrs
            cbuf["ce_base"].append(ce_b.cpu().numpy())
            for (n, nm), c_ in zip(tags, cex):
                if n == "G":
                    cbuf["dce_G"].append((c_ - ce_b).cpu().numpy().astype(np.float16))
                    cbuf["e_G"].append(eg.cpu().numpy().astype(np.int8))
                else:
                    abuf[n][nm].append((c_ - ce_b).cpu().numpy().astype(np.float16))
            if bool(sub.any()):
                for x in arms:
                    n = x["name"]
                    b = banks[n]
                    dl, _ = all_deltas(b.experts.w1, b.experts.w2, hs[sub], xs[sub], W, ys[sub])
                    dsub = -dl
                    abuf[n]["d_sub"].append(dsub.cpu().numpy().astype(np.float16))
                    dR_sub = torch.as_tensor(abuf[n]["dR"][-1], device=dev)[sub].float()
                    consist = max(consist, float((dsub.gather(1, ch[n][sub].view(-1, 1)).squeeze(1) - dR_sub).abs().max()))
            else:
                for x in arms:
                    abuf[x["name"]]["d_sub"].append(np.zeros((0, 16), np.float16))
    out.mkdir(parents=True, exist_ok=True)
    common = {"papers": np.asarray(papers), **{k: np.concatenate(v) for k, v in cbuf.items()}}
    shas = {}

    def save(path, arrs):
        tmp = path.with_name(f".{path.stem}.tmp.npz")
        np.savez_compressed(tmp, **arrs)  # lossless; disk headroom (deviation 3)
        tmp.rename(path)
        path.chmod(0o400)
        shas[path.name] = guard.sha256_file(path)
    save(out / "common_CS2.npz", common)
    cb = common["ce_base"].astype(np.float64)
    doc = common["doc"]
    same_doc = doc[1:] == doc[:-1]
    summ = {"ckpt": a.ckpt, "step": ck["step"], "seen": ck["seen"], "n_tokens": int(len(cb)), "n_papers": len(papers),
            "n_sub_tokens": int(common["sub"].sum()), "ce_base": float(cb.mean()),
            "ce_G": float(cb.mean() + common["dce_G"].astype(np.float64).mean()), "arms": {},
            "dsub_vs_routed_max_abs": consist}
    arr = {}
    for x in arms:
        n = x["name"]
        A = {k: np.concatenate(v) for k, v in abuf[n].items()}
        arr[n] = A
        save(out / f"arm_{n}_CS2.npz", {k: A[k] for k in ("dR", "dPV", "ePV", "dT", "eT", "d_sub")})
        summ["arms"][n] = {**{f"ce_{k}": float(cb.mean() + A[f"d{k}"].astype(np.float64).mean()) for k in ("R", "PV", "T")},
                           "switch_rate": float((A["e"][1:] != A["e"][:-1])[same_doc].mean())}
    lab_r = {"SEM": common["lab_rsem"].astype(np.int64), "SHUF": common["lab_rshuf"].astype(np.int64)}
    for x in arms:
        if x["bank"] != "SEM":
            continue
        r = x["seed"]
        kind = "SEMf" if x["router"] == "frozen" else "SEMj"
        dd = rdir / kind
        dd.mkdir(parents=True, exist_ok=True)
        for b, n in (("SEM", x["name"]), ("SHUF", f"SHUFf_s{r}")):
            A = arr[n]
            p = dd / f"{b}_s{r}_CS2.npz"
            if not p.exists():
                tmp = p.with_name(f".{p.stem}.tmp.npz")
                np.savez_compressed(tmp, e=A["e"], qtop=A["qtop"], margin=A["margin"])
                tmp.rename(p)
            e = A["e"].astype(np.int64)
            load = np.bincount(e, minlength=16)[:16] / len(e)
            js = {"arm": n, "bank": b, "seed": r, "part": "CS2", "ckpt": a.ckpt, "step": ck["step"],
                  "ce": float((cb + A["dR"].astype(np.float64)).mean()), "load": load.round(5).tolist(),
                  "load_max": float(load.max()), "agree_region_prior": float((e == lab_r[b]).mean()),
                  "switch_rate": float((A["e"][1:] != A["e"][:-1])[same_doc].mean()), "zero_action_frac": 0.0,
                  "npz_sha256": guard.sha256_file(p), "time": guard.now()}
            guard.atomic_write_json(p.with_suffix(".json"), js, overwrite=True)
    summ["sha256"] = shas
    summ["time"] = guard.now()
    guard.atomic_write_json(out / "summary_CS2.json", summ, mode=0o400)
    print(json.dumps({k: v for k, v in summ.items() if k != "sha256"}), flush=True)


if __name__ == "__main__":
    main()
