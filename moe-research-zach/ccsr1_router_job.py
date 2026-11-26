#!/usr/bin/env python
"""csr-current-context-specialist-router-v1: train and evaluate causal per-token routers (guarded GPU job).

Base and specialists are frozen. Router inputs are recomputed from the frozen base (rms0 of the final hidden state of
every scored token, aligned with the extraction tables and verified token by token); training targets come from the
TRAIN extraction table (utilities) and the frozen CALIB-SEARCH region priors.

  ccsr1_router_job.py --stage S1 --configs CFG.json --seed S --train TRAIN.npz --eval-part SEL --eval EVAL.npz
                      [--banks SEM,SHUF] [--max-routers N]
For each config x bank: train (fixed TRAIN_STEPS, no early stopping, no evaluation data), then evaluate on the eval
partition and write ROUTERS/<stage>/<tag>/<bank>_s<seed>.pt (+ meta) and <bank>_s<seed>_<part>.npz:
  e (N,) int8 deployed argmax choice (16 = zero action if present), qtop / margin (N_sub,) fp16 on the swap subset,
  and a json summary (CE, loads, agreement). Existing outputs are skipped (resumable at router granularity).
  ccsr1_router_job.py --eval-only --stage S2 ... evaluates existing routers on another partition."""
import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--configs", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--eval-part", required=True, choices=["TRAIN", "SEL", "CS2"])
    ap.add_argument("--eval", required=True)
    ap.add_argument("--banks", default="SEM,SHUF")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--router-stage", help="stage directory holding the routers for --eval-only (default --stage)")
    ap.add_argument("--max-routers", type=int, default=0)
    ap.add_argument("--rows-limit", type=int, default=0)
    ap.add_argument("--steps", type=int, default=0, help="tests / pre-freeze pilot only")
    ap.add_argument("--train-papers-frac", type=float, default=1.0,
                    help="pre-freeze pilot only: train on TRAIN papers with u_hash('ccsr1-pilot', paper) < frac")
    ap.add_argument("--out-root", help="tests only")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    if not guard.require_execution(a.execute, "gpu" if a.device == "cuda" else "cpu-model"):
        return
    if not a.steps:
        from csr.ccsr1_paths import check_prereg
        check_prereg()
    mon = None
    if a.device == "cuda":
        guard.occupancy_gate(f"ccsr1 routers {a.stage} s{a.seed}")
        mon = guard.OccupancyMonitor(interval=15.0)
        mon.start()
    guard.register_self(f"ccsr1 routers {a.stage} s{a.seed}")
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


def load_part(part, table_path, model, dev, rows_limit=0):
    import numpy as np
    import torch
    from ccsr1_extract import partition
    from csr.ccsr1_lib import doc_index, hidden_table
    meta = json.loads(Path(table_path).with_suffix(".json").read_text())
    if guard.sha256_file(table_path) != meta["sha256"]:
        raise guard.GuardError(f"table hash mismatch {table_path}")
    z = np.load(table_path)
    sp, rows, _, _, (off, Pt, Psh) = partition(part)
    if rows_limit:
        rows = rows[:rows_limit]
    X, tgt = hidden_table(model, sp, rows, dev)
    if X.shape[0] != len(z["tgt"]) or not bool((tgt.cpu() == torch.as_tensor(z["tgt"].astype(np.int64))).all()):
        raise guard.GuardError(f"hidden table not aligned with {table_path}")
    starts, lens = doc_index(z["doc"], z["pos"])
    T = {"X": X, "starts": starts, "lens": lens, "z": z, "meta": meta,
         "P": {"sem": torch.as_tensor(Pt, device=dev), "shuf": torch.as_tensor(Psh, device=dev)},
         "rid": torch.as_tensor(z["rid"].astype(np.int64), device=dev),
         "ce_base": z["ce_base"].astype(np.float64)}
    for X_ in ("SEM", "SHUF"):
        T[f"d_{X_}"] = torch.as_tensor(z[f"d_{X_}"], device=dev)   # fp16 (N,16): CE_k - CE_base
    return T


def bank_prior(bank):
    return {"SEM": "sem", "SHUF": "shuf"}[bank]


def train_router(cfg, bank, seed, T, dev, steps):
    import numpy as np
    import torch
    from csr.ccsr1_lib import batch_index
    from csr.ccsr1_paths import CLIP, DOCS_PER_STEP, LR, WD
    from csr.ccsr1_router import CausalRouter, log_prior, router_loss, teacher, token_weight
    n_out = int(cfg.get("n_out", 16))
    r = CausalRouter(int(cfg["H"]), seed, cfg.get("feats", "base"), n_out).to(dev)
    opt = torch.optim.AdamW(r.parameters(), lr=LR, weight_decay=WD, betas=(0.9, 0.99))
    stats = T[f"stats_{bank}"]
    rng = np.random.default_rng(1_000_003 * seed + 11)
    allowed = T.get("train_docs")
    nd = len(T["starts"]) if allowed is None else len(allowed)
    perm, pi_ = rng.permutation(nd), 0
    curve = []
    for step in range(steps):
        lr = LR * min(1.0, (step + 1) / 30) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps)))
        for gp in opt.param_groups:
            gp["lr"] = lr
        if pi_ + DOCS_PER_STEP > nd:
            perm, pi_ = rng.permutation(nd), 0
        docs = perm[pi_:pi_ + DOCS_PER_STEP]
        if allowed is not None:
            docs = allowed[docs]
        pi_ += DOCS_PER_STEP
        idx, mask, ln = batch_index(T["starts"], T["lens"], docs, dev)
        x = T["X"][idx]
        logits = r(x, ln)[mask]
        ti = idx[mask]
        u = -T[f"d_{bank}"][ti].float()
        lp = log_prior(T["P"][bank_prior(bank)], T["rid"][ti])
        with torch.no_grad():
            pi = teacher(cfg["target"], u, lp, float(cfg.get("c", 1.0)), float(cfg.get("eta", 0.0)), stats["s_u"], n_out)
            w = token_weight(cfg["target"], u, stats["med_margin"])
        loss, rec = router_loss(logits, pi, w, float(cfg["lam"]), u=u, beta_eu=float(cfg.get("beta_eu", 0.0)),
                                s_u=stats["s_u"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(r.parameters(), CLIP))
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            curve.append({"step": step, "loss": float(loss.detach()), "gn": gn, **rec})
    return r, curve


def evaluate_router(r, bank, T, dev, part):
    import numpy as np
    import torch
    from csr.ccsr1_lib import batch_index
    z = T["z"]
    N = len(z["tgt"])
    e = torch.empty(N, dtype=torch.int8, device=dev)
    qtop = torch.empty(N, dtype=torch.float16, device=dev)
    marg = torch.empty(N, dtype=torch.float16, device=dev)
    order = np.argsort(T["lens"])
    with torch.no_grad():
        for i in range(0, len(order), 256):
            docs = order[i:i + 256]
            idx, mask, ln = batch_index(T["starts"], T["lens"], docs, dev)
            q = torch.softmax(r(T["X"][idx], ln).float(), -1)[mask]
            ti = idx[mask]
            t2 = q.topk(2, -1)
            e[ti] = t2.indices[:, 0].to(torch.int8)
            qtop[ti] = t2.values[:, 0].half()
            marg[ti] = (t2.values[:, 0] - t2.values[:, 1]).half()
    e64 = e.long()
    d = T[f"d_{bank}"]
    dch = torch.where(e64 < 16, d.gather(1, e64.clamp(max=15).view(-1, 1)).squeeze(1).float(), torch.zeros(N, device=dev))
    e_np = e.cpu().numpy()
    sub = z["sub"].astype(bool)
    lab_r = z["lab_rsem" if bank == "SEM" else "lab_rshuf"].astype(np.int64)
    amin = d.float().argmin(1).cpu().numpy()
    load = np.bincount(e_np.astype(np.int64), minlength=17)[:17] / N
    summ = {"ce": float((T["ce_base"] + dch.double().cpu().numpy()).mean()),
            "load": load.round(5).tolist(), "load_max": float(load[:16].max()),
            "load_eff": float(1.0 / max(1e-12, float(((load[:16] / max(1e-12, load[:16].sum())) ** 2).sum()))),
            "n_dead": int((load[:16] < 1 / 160).sum()), "zero_action_frac": float(load[16]) if len(load) > 16 else 0.0,
            "agree_region_prior": float((e_np == lab_r).mean()), "agree_util_argmax": float((e_np == amin).mean()),
            "switch_rate": float((e_np[1:] != e_np[:-1])[np.asarray(z["doc"][1:]) == np.asarray(z["doc"][:-1])].mean())}
    arrays = {"e": e_np, "qtop": qtop.cpu().numpy()[sub], "margin": marg.cpu().numpy()[sub]}
    return summ, arrays


def run(a, mon):
    import numpy as np
    import torch
    from csr.ccsr1_paths import CKPT, ROUTERS, TRAIN_STEPS
    from fle_evaluate import load_milestone
    from ccsr1_extract import verify_reused
    verify_reused()
    dev = guard.setup_cuda_device() if a.device == "cuda" else torch.device("cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    model, ck = load_milestone(CKPT, dev)
    for p in model.parameters():
        p.requires_grad_(False)
    cfgs = json.loads(Path(a.configs).read_text())
    banks = a.banks.split(",")
    root = Path(a.out_root) if a.out_root else ROUTERS
    steps = a.steps or TRAIN_STEPS
    Ttr = None
    if not a.eval_only:
        Ttr = load_part("TRAIN", a.train, model, dev, a.rows_limit)
        if a.train_papers_frac < 1.0:
            from orv1_tables import u_hash
            keep = np.asarray([u_hash("ccsr1-pilot", str(p)) < a.train_papers_frac for p in Ttr["z"]["papers"]])
            dp = Ttr["z"]["paper"][Ttr["starts"]].astype(np.int64)
            Ttr["train_docs"] = np.nonzero(keep[dp])[0]
        for b in banks:
            u = -Ttr[f"d_{b}"].float()
            sd = u.std(1)
            m = u.max(1).values - u.mean(1)
            Ttr[f"stats_{b}"] = {"s_u": float(sd.median()), "med_margin": float(m.median())}
    Tev = load_part(a.eval_part, a.eval, model, dev, a.rows_limit)
    n_done = 0
    for cfg in cfgs:
        for b in banks:
            if mon is not None and mon.contaminated:
                sys.exit(77)
            rdir = root / (a.router_stage or a.stage) / cfg["tag"]
            odir = root / a.stage / cfg["tag"]
            odir.mkdir(parents=True, exist_ok=True)
            ckp = rdir / f"{b}_s{a.seed}.pt"
            outp = odir / f"{b}_s{a.seed}_{a.eval_part}.npz"
            if outp.exists():
                continue
            if a.max_routers and n_done >= a.max_routers:
                return
            t1 = time.time()
            if a.eval_only or ckp.exists():
                st = torch.load(ckp, map_location="cpu", weights_only=False)
                from csr.ccsr1_router import CausalRouter
                r = CausalRouter(int(st["cfg"]["H"]), a.seed, st["cfg"].get("feats", "base"), int(st["cfg"].get("n_out", 16))).to(dev)
                r.load_state_dict(st["state"])
                curve = st["curve"]
                if st["cfg"] != cfg:
                    raise guard.GuardError(f"router {ckp} trained with a different config")
            else:
                r, curve = train_router(cfg, b, a.seed, Ttr, dev, steps)
                tmp = ckp.with_name(f".{ckp.stem}.tmp.pt")
                torch.save({"cfg": cfg, "bank": b, "seed": a.seed, "steps": steps, "state": r.state_dict(),
                            "curve": curve, "stats": Ttr[f"stats_{b}"], "n_params": r.n_params(),
                            "train_table_sha256": Ttr["meta"]["sha256"]}, tmp)
                tmp.rename(ckp)
            r.eval()
            summ, arrays = evaluate_router(r, b, Tev, dev, a.eval_part)
            tmp = outp.with_name(f".{outp.stem}.tmp.npz")
            np.savez(tmp, **arrays)
            tmp.rename(outp)
            summ.update({"cfg": cfg, "bank": b, "seed": a.seed, "part": a.eval_part, "n_params": r.n_params(),
                         "train_curve_last": curve[-1], "eval_table_sha256": Tev["meta"]["sha256"],
                         "router_sha256": guard.sha256_file(ckp), "npz_sha256": guard.sha256_file(outp),
                         "seconds": round(time.time() - t1, 1), "time": guard.now()})
            guard.atomic_write_json(outp.with_suffix(".json"), summ, mode=0o400)
            print(json.dumps({k: summ[k] for k in ("ce", "load_max", "agree_region_prior", "seconds")}), cfg["tag"], b,
                  flush=True)
            n_done += 1


if __name__ == "__main__":
    main()
