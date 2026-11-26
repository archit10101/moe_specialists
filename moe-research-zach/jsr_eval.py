#!/usr/bin/env python
"""joint-semantic-residual-pretrain-v1 evaluation (guarded GPU job; CPU for tests).

  jsr_eval.py --ckpt RUN/ckpt_TAG --part SEL|EVAL|VAL --out NPZ [--rows-limit N]
      joint checkpoint (shared base + banks): per valid token ce_base (A_internal), ce_<bank> (routed path), e_<bank>,
      qtop_<bank>, margin_<bank>, anorm_<bank>, labels (sem, shuf), paper, pos, doclen, tgt, domain; on the swap-matrix
      subset (SEL/EVAL: all tokens; VAL: papers with u_hash("jsr-val-sub", paper) < 0.5) also d_<bank> (N_sub, 16) fp16
      = CE with specialist j forced - ce_base and q_<bank> (N_sub, 16) fp16. Plus per-bank secondary diagnostics:
      per-group residual-loss gradient cosine (bank parameters; <= 8192 subset tokens per semantic group) and mean
      pairwise correction cosine between specialists.
  jsr_eval.py --ref RUN --part VAL --out NPZ
      reused base-only model (TOKEN-4 / EDL / C6 p100): per-paper CE sums (all tokens and first 128 positions)."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402

SUB_FRAC = 0.5
GC_TOK = 8192


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
    if not a.rows_limit:
        from csr.jsr_paths import check_prereg
        check_prereg()
    mon = None
    if a.device == "cuda":
        guard.occupancy_gate(f"jsr eval {Path(a.out).name}")
        mon = guard.OccupancyMonitor(interval=15.0)
        mon.start()
    guard.register_self(f"jsr eval {Path(a.out).name}")
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
    from csr.fsp_lib import Split
    from csr.jsr_paths import ORV1
    from csr.orv1_lib import unflatten_rows
    if part == "VAL":
        sp = Split("val")
        zl = np.load(ORV1 / "val_assign/labels_val.npz")
        zp = np.load(ORV1 / "val_assign/parts_val.npz")
        rows = unflatten_rows(zp["VAL_docs"], zp["VAL_row_len"])
    else:
        sp = Split("calib_search")
        zl = np.load(ORV1 / "tables/labels_cs.npz")
        zp = np.load(ORV1 / "tables/parts.npz")
        rows = unflatten_rows(zp[f"{part}_docs"], zp[f"{part}_row_len"])
    return sp, rows, zl["sem"].astype(np.int64), zl["shuf"].astype(np.int64)


def run(a, mon):
    import numpy as np
    import torch
    from csr.jsr_bank import Bank, head_ce_grad, rms0
    from csr.jsr_paths import K
    from csr.orv1_lib import make_batch, to_dev
    from orv1_tables import u_hash
    from fle_evaluate import load_milestone
    out = Path(a.out)
    if out.exists():
        print("exists")
        return
    dev = guard.setup_cuda_device() if a.device == "cuda" else torch.device("cpu")
    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    path = a.ref if a.ref else a.ckpt
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
        for n, spec in ck["bank_specs"].items():
            b = Bank(spec, ck["config"]["seed"]).to(dev)
            b.load_state_dict(ck["banks"][n])
            b.eval()
            for p in b.parameters():
                p.requires_grad_(False)
            banks[n] = b
    sp, rows, sem, shuf = partition(a.part)
    if a.rows_limit:
        rows = rows[:a.rows_limit]
    papers = sorted({sp.papers[d] for r in rows for d in r})
    pidx = {p: i for i, p in enumerate(papers)}
    paper_of_doc = np.full(sp.n_docs, -1, np.int64)
    for r in rows:
        for d in r:
            paper_of_doc[d] = pidx[sp.papers[d]]
    sub_paper = np.asarray([(u_hash("jsr-val-sub", p) < SUB_FRAC) if a.part == "VAL" else True for p in papers])
    sem_t, shuf_t = torch.as_tensor(sem, device=dev), torch.as_tensor(shuf, device=dev)
    psub_t = torch.as_tensor(sub_paper, device=dev)
    pdoc_t = torch.as_tensor(paper_of_doc, device=dev)
    P = len(papers)
    if a.ref:
        acc = np.zeros((P, 4))
    buf = {k: [] for k in ("ce_base", "lab_sem", "lab_shuf", "paper", "pos", "doclen", "tgt", "domain", "sub")}
    for n in banks:
        for k in ("ce", "e", "qtop", "margin", "anorm", "d", "q"):
            buf[f"{k}_{n}"] = []
    gc_h = {k: [] for k in range(K)}
    gc_n = np.zeros(K, np.int64)
    with torch.no_grad():
        for i in range(0, len(rows), a.micro_rows):
            if mon is not None and mon.contaminated:
                sys.exit(77)
            bt = to_dev(make_batch(rows[i:i + a.micro_rows], sp), dev)
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                o = model({"input_ids": bt["input_ids"], "doc_ids": bt["doc_ids"], "positions": bt["positions"]})
            h = o["hidden"].reshape(-1, o["hidden"].shape[-1])
            y = bt["targets"].reshape(-1)
            v = y >= 0
            h, y = h[v], y[v]
            gd = bt["gdoc"].reshape(-1)[v]
            pos = bt["positions"].reshape(-1)[v]
            ce_b, _ = head_ce_grad(h, torch.zeros_like(h, dtype=torch.float32), W, y, want_grad=False)
            if a.ref:
                pp = pdoc_t[gd].cpu().numpy()
                c = ce_b.double().cpu().numpy()
                f128 = (pos < 128).cpu().numpy()
                np.add.at(acc[:, 0], pp, c)
                np.add.at(acc[:, 1], pp, 1)
                np.add.at(acc[:, 2], pp[f128], c[f128])
                np.add.at(acc[:, 3], pp[f128], 1)
                continue
            ls, lh = sem_t[gd], shuf_t[gd]
            pp = pdoc_t[gd]
            sub = psub_t[pp]
            xn = rms0(h)
            buf["ce_base"].append(ce_b.cpu().numpy())
            buf["lab_sem"].append(ls.cpu().numpy().astype(np.int8))
            buf["lab_shuf"].append(lh.cpu().numpy().astype(np.int8))
            buf["paper"].append(pp.cpu().numpy().astype(np.int32))
            buf["pos"].append(pos.cpu().numpy().astype(np.int16))
            buf["doclen"].append(torch.as_tensor(sp.lengths, device=dev)[gd].cpu().numpy().astype(np.int16))
            buf["tgt"].append(y.cpu().numpy().astype(np.int32))
            buf["domain"].append(np.asarray(sp.domains)[gd.cpu().numpy()].astype(np.int8))
            buf["sub"].append(sub.cpu().numpy())
            for n, b in banks.items():
                e, gate, q, act = b.route(xn, ls if not b.routed else None)
                acor = b.correction(xn, e, gate)
                ce_x, _ = head_ce_grad(h, acor, W, y, want_grad=False)
                buf[f"ce_{n}"].append(ce_x.cpu().numpy())
                buf[f"e_{n}"].append(torch.where(act, e, torch.full_like(e, -1)).cpu().numpy().astype(np.int8))
                buf[f"anorm_{n}"].append(acor.norm(dim=1).cpu().numpy().astype(np.float16))
                if q is not None:
                    t2 = q.topk(2, -1).values
                    buf[f"qtop_{n}"].append(t2[:, 0].cpu().numpy().astype(np.float16))
                    buf[f"margin_{n}"].append((t2[:, 0] - t2[:, 1]).cpu().numpy().astype(np.float16))
                if bool(sub.any()):
                    hs, ys, xs = h[sub], y[sub], xn[sub]
                    d = torch.empty(hs.shape[0], K, device=dev)
                    cb = ce_b[sub]
                    for k in range(K):
                        d[:, k] = head_ce_grad(hs, b.forced(xs, k), W, ys, want_grad=False)[0] - cb
                    buf[f"d_{n}"].append(d.cpu().numpy().astype(np.float16))
                    if q is not None:
                        buf[f"q_{n}"].append(q[sub].cpu().numpy().astype(np.float16))
            # gradient-cosine sample: first GC_TOK subset tokens of each semantic group
            if banks:
                for k in range(K):
                    need = GC_TOK - gc_n[k]
                    if need <= 0:
                        continue
                    m = (ls == k) & sub
                    idx = torch.nonzero(m).squeeze(1)[:need]
                    if idx.numel():
                        gc_h[k].append((h[idx].float().cpu(), y[idx].cpu()))
                        gc_n[k] += idx.numel()
    if a.ref:
        res = {"papers": np.asarray(papers), "sum": acc[:, 0], "n": acc[:, 1], "sum128": acc[:, 2], "n128": acc[:, 3]}
        summ = {"ref": a.ref, "ckpt": path, "part": a.part, "ce": float(acc[:, 0].sum() / acc[:, 1].sum()),
                "ce128": float(acc[:, 2].sum() / acc[:, 3].sum()), "n_tokens": int(acc[:, 1].sum()), "n_papers": P}
    else:
        res = {"papers": np.asarray(papers)}
        for k, v_ in buf.items():
            if v_:
                res[k] = np.concatenate(v_)
        diag = secondary(banks, gc_h, W, dev) if banks else {}
        n_tok = len(res["ce_base"])
        summ = {"ckpt": str(a.ckpt), "step": ck["step"], "seen": ck["seen"], "part": a.part, "n_tokens": int(n_tok),
                "n_papers": P, "n_sub_tokens": int(res["sub"].sum()), "ce_base": float(res["ce_base"].astype(np.float64).mean()),
                **{f"ce_{n}": float(res[f"ce_{n}"].astype(np.float64).mean()) for n in banks},
                "load": {n: (np.bincount(res[f"e_{n}"][res[f"e_{n}"] >= 0].astype(np.int64), minlength=K) / n_tok).round(5).tolist()
                         for n in banks}, "secondary": diag}
        res["gradcos"] = np.asarray([diag[n]["gradcos"] for n in banks]) if diag else np.zeros(0)
        res["corrcos"] = np.asarray([diag[n]["corrcos"] for n in banks]) if diag else np.zeros(0)
        res["bank_names"] = np.asarray(list(banks))
        for n in banks:
            diag[n].pop("gradcos")
            diag[n].pop("corrcos")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.stem}.tmp.npz")
    np.savez(tmp, **res)
    tmp.rename(out)
    out.chmod(0o400)
    summ["sha256"] = guard.sha256_file(out)
    summ["time"] = guard.now()
    guard.atomic_write_json(out.with_suffix(".json"), summ, mode=0o400)
    print(json.dumps({k: v for k, v in summ.items() if k != "secondary"}), flush=True)


def secondary(banks, gc_h, W, dev):
    """Per bank: 16x16 cosine of per-semantic-group residual-loss gradients w.r.t. the bank's expert parameters
    (routed path; router excluded), and the mean pairwise cosine between specialists' forced corrections."""
    import numpy as np
    import torch
    from csr.jsr_bank import head_ce_grad, rms0
    from csr.jsr_paths import K
    out = {}
    hs = {k: (torch.cat([x for x, _ in gc_h[k]]).to(dev), torch.cat([y for _, y in gc_h[k]]).to(dev))
          for k in range(K) if gc_h[k]}
    for n, b in banks.items():
        G = []
        ks = sorted(hs)
        for k in ks:
            h, y = hs[k]
            xn = rms0(h)
            params = [b.experts.w1, b.experts.w2]
            for p in params:
                p.requires_grad_(True)
            with torch.enable_grad():
                lab = torch.full((h.shape[0],), k, device=dev, dtype=torch.long)
                e, gate, _, _ = b.route(xn, lab if not b.routed else None)
                acor = b.correction(xn, e, gate.detach())
                _, ga = head_ce_grad(h, acor.detach(), W, y)
                gs = torch.autograd.grad(acor, params, grad_outputs=ga / h.shape[0])
            for p in params:
                p.requires_grad_(False)
            G.append(torch.cat([g.reshape(-1).float() for g in gs]))
        G = torch.stack(G)
        Gn = G / G.norm(dim=1, keepdim=True).clamp(min=1e-30)
        gc = np.full((K, K), np.nan)
        cm = (Gn @ Gn.t()).cpu().numpy()
        for i, ki in enumerate(ks):
            for j, kj in enumerate(ks):
                gc[ki, kj] = cm[i, j]
        xs = rms0(torch.cat([hs[k][0][:1024] for k in ks]))
        A = torch.stack([b.forced(xs, k) for k in range(K)], 1)  # (N, K, d)
        An = A / A.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        cc = torch.einsum("nkd,njd->kj", An, An).cpu().numpy() / A.shape[0]
        off = cc[~np.eye(K, dtype=bool)]
        out[n] = {"gradcos": gc, "corrcos": cc, "gradcos_offdiag_mean": float(np.nanmean(gc[~np.eye(K, dtype=bool)])),
                  "corrcos_offdiag_mean": float(off.mean())}
    return out


if __name__ == "__main__":
    main()
