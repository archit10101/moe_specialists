#!/usr/bin/env python
"""csr-current-context-specialist-router-v1: per-token specialist tables for one partition (guarded GPU job).

Loads the reused, hash-verified cfar1 Stage-1 joint checkpoint (frozen TOKEN-4 replicate base + banks), keeps banks
G (token-routed generic), SEM = L (local semantic) and SHUF = F2h_T500_w240 (local shuffled), and writes for every
scored token of the partition (same row packing and order as csur1/cfar1 evaluation):

  ce_base (fp32), lab_sem / lab_shuf (paper-level C6 cluster / orv1 shuffle), lab_rsem / lab_rshuf (argmax region
  prior p_sem / p_SH), rid (row of the token's 256-token region in the region-prior table), paper, doc, pos, doclen,
  tgt, domain (post hoc only), blk (128-token view number), sub (swap subset)
  dce_G, e_G                  G with its own router (deployable)
  dPV_X, ePV_X (X = SEM|SHUF) existing PAST-VIEW student (cfar1 meanlast GRU over completed views; deployable)
  dT_X, eT_X                  CURRENT-REGION TEACHER: argmax of the region prior (training-only information)
  d_X (N, 16) fp16            CE with specialist k forced - ce_base, for EVERY scored token (all 16 specialists)

A router's CE on this partition is exactly ce_base + d_X[t, choice_t] (one specialist per token, frozen specialists).
  ccsr1_extract.py --part TRAIN|SEL|CS2 --out NPZ [--rows-limit N]"""
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
    ap.add_argument("--part", choices=["TRAIN", "SEL", "CS2"], required=True)
    ap.add_argument("--out", required=True)
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
        guard.occupancy_gate(f"ccsr1 extract {a.part}")
        mon = guard.OccupancyMonitor(interval=15.0)
        mon.start()
    guard.register_self(f"ccsr1 extract {a.part}")
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


def verify_reused():
    from csr.ccsr1_paths import (CKPT, CKPT_SHA, CSUR1_TABLES, FEATS_SHA, LABELS_CS_SHA, ORV1, PARTS_SHA, PSEM_CS_SHA,
                                 PSEM_CS_SH_SHA)
    for p, h in ((guard.ROOT / CKPT / "state.pt", CKPT_SHA), (CSUR1_TABLES / "psem_calib_search.npz", PSEM_CS_SHA),
                 (CSUR1_TABLES / "psem_calib_search_sh.npz", PSEM_CS_SH_SHA), (ORV1 / "tables/labels_cs.npz", LABELS_CS_SHA),
                 (ORV1 / "tables/parts.npz", PARTS_SHA), (ORV1 / "tables/token_feats.npz", FEATS_SHA)):
        if guard.sha256_file(p) != h:
            raise guard.GuardError(f"hash mismatch {p}")


def partition(part):
    """(split, rows, paper-level sem labels, shuf labels, (off, P_sem, P_sh)) for TRAIN / SEL / CS2."""
    import numpy as np
    from csr.ccsr1_paths import ART, CSUR1_TABLES, ORV1, PARTS
    from csr.fsp_lib import Split
    from csr.orv1_lib import unflatten_rows
    split_name, half = PARTS[part]
    sp = Split(split_name)
    if part == "CS2":
        if not (ART / "FROZEN_S2.json").exists():
            raise guard.GuardError("CALIB-SELECTION is locked until FROZEN_S2.json exists")
        adir, tdir, tab = ART / "cs2_assign", ART / "tables_cs2", "psem_calib_sel"
        zl = np.load(adir / "labels_cs2.npz")
        zp = np.load(adir / "parts_cs2.npz")
        rows = unflatten_rows(zp["CS2_docs"], zp["CS2_row_len"])
    else:
        tdir, tab = CSUR1_TABLES, "psem_calib_search"
        zl = np.load(ORV1 / "tables/labels_cs.npz")
        zp = np.load(ORV1 / "tables/parts.npz")
        rows = unflatten_rows(zp[f"{half}_docs"], zp[f"{half}_row_len"])
    z = np.load(tdir / f"{tab}.npz")
    meta = json.loads((tdir / f"{tab}.json").read_text())
    if guard.sha256_file(tdir / f"{tab}.npz") != meta["sha256"]:
        raise guard.GuardError(f"{tab} hash mismatch")
    msh = json.loads((tdir / f"{tab}_sh.json").read_text())
    if guard.sha256_file(tdir / f"{tab}_sh.npz") != msh["sha256"] or msh["source_psem_sha256"] != meta["sha256"]:
        raise guard.GuardError(f"{tab}_sh hash mismatch")
    zs = np.load(tdir / f"{tab}_sh.npz")
    off = np.full(sp.n_docs, -1, np.int64)
    off[z["docs"]] = z["ptr"][:-1]
    return sp, rows, zl["sem"].astype(np.int64), zl["shuf"].astype(np.int64), (off, z["p"].astype(np.float32),
                                                                                zs["p_sh"].astype(np.float32))


def load_banks(ck, dev):
    import torch
    from csr.cfar1_bank import FedBank
    from csr.cfar1_paths import ALPHA_G
    from csr.ccsr1_paths import BANK_G, BANK_SEM, BANK_SHUF
    from csr.jsr_bank import Bank
    out = {}
    for alias, n in (("G", BANK_G), ("SEM", BANK_SEM), ("SHUF", BANK_SHUF)):
        spec = ck["bank_specs"][n]
        if alias == "G":
            b = Bank({"name": n, "kind": "routed", "tau": 1.0, "r": 0.0, "alpha": ALPHA_G, "groups": None},
                     ck["config"]["seed"])
        else:
            b = FedBank(spec, ck["config"]["seed"])
        b = b.to(dev)
        b.load_state_dict(ck["banks"][n])
        b.eval()
        for p in b.parameters():
            p.requires_grad_(False)
        out[alias] = b
    return out


def run(a, mon):
    import numpy as np
    import torch
    from csr.csur1_bank import all_deltas, block_feats, ce_pair, layout
    from csr.ccsr1_paths import CKPT, SUB_FRAC
    from csr.jsr_bank import rms0
    from csr.orv1_lib import make_batch, to_dev
    from orv1_tables import u_hash
    from fle_evaluate import load_milestone
    from cfar1_train import block_priors
    out = Path(a.out)
    if out.exists():
        print("exists")
        return
    verify_reused()
    dev = guard.setup_cuda_device() if a.device == "cuda" else torch.device("cpu")
    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    model, ck = load_milestone(CKPT, dev)
    for p in model.parameters():
        p.requires_grad_(False)
    W = model.tok_emb.weight.detach()
    banks = load_banks(ck, dev)
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
    keys = ["ce_base", "lab_sem", "lab_shuf", "lab_rsem", "lab_rshuf", "rid", "paper", "doc", "pos", "doclen", "tgt",
            "domain", "blk", "sub", "dce_G", "e_G"]
    for X in ("SEM", "SHUF"):
        keys += [f"dPV_{X}", f"ePV_{X}", f"dT_{X}", f"eT_{X}", f"d_{X}"]
    buf = {k: [] for k in keys}
    consist = {"G": 0.0, "SEM": 0.0, "SHUF": 0.0}
    rid_check = 0
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
            gds = gd[sc]
            pos_s = pos[sc]
            rid = off_t[gds] + torch.div(pos_s, 256, rounding_mode="floor")
            # region row from the token position must equal the row used for the token's block prior
            rid_check = max(rid_check, int((torch.log(P_t[rid] + 1e-4) - lp_sem[blk_s]).abs().max()))
            pp = pdoc_t[gds]
            sub = psub_t[pp]
            ce_b, _ = ce_pair(hs, None, W, ys)
            buf["ce_base"].append(ce_b.cpu().numpy())
            buf["lab_sem"].append(sem_t[gds].cpu().numpy().astype(np.int8))
            buf["lab_shuf"].append(shuf_t[gds].cpu().numpy().astype(np.int8))
            buf["lab_rsem"].append(lp_sem.argmax(1)[blk_s].cpu().numpy().astype(np.int8))
            buf["lab_rshuf"].append(lp_shuf.argmax(1)[blk_s].cpu().numpy().astype(np.int8))
            buf["rid"].append(rid.cpu().numpy().astype(np.int32))
            buf["paper"].append(pp.cpu().numpy().astype(np.int32))
            buf["doc"].append(gds.cpu().numpy().astype(np.int32))
            buf["pos"].append(pos_s.cpu().numpy().astype(np.int16))
            buf["doclen"].append(len_t[gds].cpu().numpy().astype(np.int16))
            buf["tgt"].append(ys.cpu().numpy().astype(np.int32))
            buf["domain"].append(dom_t[gds].cpu().numpy().astype(np.int8))
            buf["blk"].append(lay["blk_no"][blk_s].cpu().numpy().astype(np.int8))
            buf["sub"].append(sub.cpu().numpy())
            g = banks["G"]
            eg = g.probs(xs).argmax(1)
            _, ce_g = ce_pair(hs, g.correction(xs, eg, torch.ones_like(eg, dtype=torch.float32)), W, ys)
            buf["dce_G"].append((ce_g - ce_b).cpu().numpy().astype(np.float16))
            buf["e_G"].append(eg.cpu().numpy().astype(np.int8))
            for X in ("SEM", "SHUF"):
                b = banks[X]
                dl, _ = all_deltas(b.experts.w1, b.experts.w2, hs, xs, W, ys)
                d = -dl  # CE with specialist k forced - CE_base
                buf[f"d_{X}"].append(d.cpu().numpy().astype(np.float16))
                lg = b.student_logits(s_blk, lay).float()
                epv = lg.argmax(1)[blk_s]
                _, ce_pv = ce_pair(hs, b.correction(xs, epv), W, ys)
                lp = {"sem": lp_sem, "shuf": lp_shuf}[b.prior]
                eT = lp.argmax(1)[blk_s]
                _, ce_t = ce_pair(hs, b.correction(xs, eT), W, ys)
                for nm, e_, ce_ in (("PV", epv, ce_pv), ("T", eT, ce_t)):
                    buf[f"d{nm}_{X}"].append((ce_ - ce_b).cpu().numpy().astype(np.float16))
                    buf[f"e{nm}_{X}"].append(e_.cpu().numpy().astype(np.int8))
                    consist[X] = max(consist[X], float((d.gather(1, e_.view(-1, 1)).squeeze(1) - (ce_ - ce_b)).abs().max()))
                del dl, d
    res = {"papers": np.asarray(papers)}
    for k, v_ in buf.items():
        res[k] = np.concatenate(v_)
    cb = res["ce_base"].astype(np.float64)
    n_tok = len(cb)
    summ = {"part": a.part, "ckpt": CKPT, "step": ck["step"], "seen": ck["seen"], "n_tokens": int(n_tok),
            "n_papers": len(papers), "n_rows": len(rows), "n_sub_tokens": int(res["sub"].sum()), "ce_base": float(cb.mean()),
            "ce_G": float(cb.mean() + res["dce_G"].astype(np.float64).mean()),
            **{f"ce_{nm}_{X}": float(cb.mean() + res[f"d{nm}_{X}"].astype(np.float64).mean())
               for X in ("SEM", "SHUF") for nm in ("PV", "T")},
            **{f"ce_oracle_{X}": float(cb.mean() + res[f"d_{X}"].astype(np.float64).min(1).mean()) for X in ("SEM", "SHUF")},
            "forced_vs_routed_max_abs": consist, "rid_check_max_abs": rid_check}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.stem}.tmp.npz")
    np.savez(tmp, **res)
    tmp.rename(out)
    out.chmod(0o400)
    summ["sha256"] = guard.sha256_file(out)
    summ["bytes"] = out.stat().st_size
    summ["time"] = guard.now()
    guard.atomic_write_json(out.with_suffix(".json"), summ, mode=0o400)
    print(json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
