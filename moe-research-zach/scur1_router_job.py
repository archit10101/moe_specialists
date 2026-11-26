#!/usr/bin/env python
"""scur1: train causal subject-conditioned utility routers (and controls) on CALIB-SEARCH TRAIN (half "EVAL") for one
frozen milestone (base + specialist banks never updated here) and deploy them on an evaluation partition (guarded GPU
job).

Inputs: the milestone (hash-verified), the TRAIN extraction table (scur1_extract.py) and the evaluation table (asrs2 SEL
evaluation of the same milestone, or a scur1 CS2 extraction). Router inputs are recomputed from the frozen base (rms0 of
the final hidden state of every scored token, aligned token by token with the tables). U is computed from TRAIN only:
two paper folds (u_hash('scur1-fold', paper) < 0.5); a TRAIN token's training targets use the U of the OTHER fold; the
deployed A uses all TRAIN tokens. Reference loss for U = the matched generic ALT-G routed CE.

  scur1_router_job.py --milestone TAG|PATH --ckpt-sha SHA --train TRAIN_DIR --eval EVAL_DIR --eval-part SEL|CS2
                      --configs CFG.json --seed S --stage NAME [--steps N] [--rows-limit N]
Writes ROUTERS/<stage>/<tag>_s<seed>.pt and DEPS/<stage>/<tag>_s<seed>_<part>.npz (e int8, q fp16 (N,K), ps fp16
(N,K) subject posterior) + .json summary. Existing outputs are skipped (resumable per router)."""
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
    ap.add_argument("--milestone", required=True)
    ap.add_argument("--ckpt-sha", default="")
    ap.add_argument("--train", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--eval-part", required=True, choices=["SEL", "CS2", "TRAIN"])
    ap.add_argument("--configs", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--rows-limit", type=int, default=0)
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    if not guard.require_execution(a.execute, "gpu" if a.device == "cuda" else "cpu-model"):
        return
    if not a.test:
        from csr.scur1_paths import check_prereg
        check_prereg()
    mon = None
    if a.device == "cuda":
        guard.occupancy_gate(f"scur1 routers {a.stage} s{a.seed}")
        mon = guard.OccupancyMonitor(interval=15.0)
        mon.start()
    guard.register_self(f"scur1 routers {a.stage} s{a.seed}")
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


def load_table(d, part, banks, dev, model, rows_limit=0):
    """Extraction-table tensors for one partition + base features recomputed and aligned."""
    import numpy as np
    import torch
    from csr.ccsr1_lib import doc_index, hidden_table
    from scur1_extract import partition
    d = Path(d)
    summ = json.loads((d / f"summary_{part}.json").read_text())
    for fn, sha in summ["sha256"].items():
        if fn.startswith(("common", *(f"bank_{b}" for b in banks))) and guard.sha256_file(d / fn) != sha:
            raise guard.GuardError(f"table hash mismatch {d / fn}")
    z = dict(np.load(d / f"common_{part}.npz"))
    sp, rows, tab, tab_sha = partition(part)
    if rows_limit:
        rows = rows[:rows_limit]
    X, tgt = hidden_table(model, sp, rows, dev)
    if X.shape[0] != len(z["tgt"]) or not bool((tgt.cpu() == torch.as_tensor(z["tgt"].astype(np.int64))).all()):
        raise guard.GuardError(f"hidden table not aligned with {d}")
    starts, lens = doc_index(z["doc"], z["pos"])
    T = {"X": X, "starts": starts, "lens": lens, "z": z, "N": len(z["tgt"]), "dir": str(d), "summ": summ,
         "rid": torch.as_tensor(z["rid"].astype(np.int64), device=dev)}
    T["q"] = {"sem": torch.as_tensor(tab["p"].astype(np.float32), device=dev)[T["rid"]],
              "shuf": torch.as_tensor(tab["p_sh"].astype(np.float32), device=dev)[T["rid"]]}
    for k in T["q"]:
        T["q"][k] = (T["q"][k] + 1e-6) / (T["q"][k] + 1e-6).sum(1, keepdim=True)
    for b in banks:
        bz = np.load(d / f"bank_{b}_{part}.npz")
        T[f"d_{b}"] = torch.as_tensor(bz["d"].astype(np.float32), device=dev)
        if "e" in bz.files:
            e = torch.as_tensor(bz["e"].astype(np.int64), device=dev)
            T[f"dr_{b}"] = T[f"d_{b}"].gather(1, e.view(-1, 1)).squeeze(1)
    return T


def train_router(cfg, seed, T, dev, steps, bank, ref, P):
    import numpy as np
    import torch
    from csr.ccsr1_lib import batch_index
    from csr.scur1_router import SCURouter, losses, mapping, utility_matrix
    from orv1_tables import u_hash
    prior = "sem" if cfg["bank"] == "SEM" else "shuf"
    q = T["q"][prior]
    d = T[f"d_{bank}"]                                    # CE_e - CE_base
    gain = (T[f"dr_{ref}"].unsqueeze(1) - d)             # CE_ref - CE_e
    dlm = d - T[f"dr_{ref}"].unsqueeze(1)                 # CE_e - CE_ref (router LM term)
    papers = T["z"]["papers"]
    fold_p = np.asarray([u_hash("scur1-fold", str(p)) < 0.5 for p in papers])
    fold = torch.as_tensor(fold_p[T["z"]["paper"].astype(np.int64)].astype(np.int64), device=dev)
    U = [utility_matrix(q[fold == f], gain[fold == f]) for f in (0, 1)]
    Uall = utility_matrix(q, gain)
    kind = cfg.get("A", "soft")
    tf = float(cfg.get("tau", 1.0))
    Af = torch.stack([mapping(U[1], tf, kind), mapping(U[0], tf, kind)])   # index by token fold -> other fold's U
    Uf = torch.stack([U[1], U[0]])
    Adep = mapping(Uall, tf, kind)
    torch.manual_seed(seed)
    r = SCURouter(P.ROUTER_H, seed, P.K, cfg.get("feats", "base"), cfg["mode"], float(cfg.get("rho", 1.0)), P.PSEM_EPS,
                  float(cfg.get("prior_temp", 1.0))).to(dev)
    r.set_A(Adep)
    opt = torch.optim.AdamW(r.parameters(), lr=P.LR, weight_decay=P.WD, betas=(0.9, 0.99))
    rng = np.random.default_rng(1_000_003 * seed + 11)
    nd = len(T["starts"])
    perm, pi_ = rng.permutation(nd), 0
    w = dict(cfg["w"])
    curve = []
    for step in range(steps):
        lr = P.LR * min(1.0, (step + 1) / 30) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps)))
        for gp in opt.param_groups:
            gp["lr"] = lr
        if pi_ + P.DOCS_PER_STEP > nd:
            perm, pi_ = rng.permutation(nd), 0
        docs = perm[pi_:pi_ + P.DOCS_PER_STEP]
        pi_ += P.DOCS_PER_STEP
        idx, mask, ln = batch_index(T["starts"], T["lens"], docs, dev)
        g = r.backbone(T["X"][idx], ln)[mask]
        ti = idx[mask]
        ft = fold[ti]
        sl, rl = r.heads(g, Af[ft])
        loss, parts = losses(sl, rl, q[ti], Af[ft], Uf[ft], dlm[ti], w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(r.parameters(), P.CLIP))
        opt.step()
        if step % 100 == 0 or step == steps - 1:
            curve.append({"step": step, "loss": float(loss.detach()), "gn": gn, **parts})
    info = {"U_fold0": U[0].tolist(), "U_fold1": U[1].tolist(), "U_all": Uall.tolist(), "A_deploy": Adep.tolist(),
            "U_split_half_r": float(np.corrcoef(U[0].cpu().numpy().ravel(), U[1].cpu().numpy().ravel())[0, 1]),
            "ref": ref, "bank": bank, "prior": prior}
    return r, curve, info


def evaluate(r, T, dev, bank, prior):
    import numpy as np
    import torch
    from csr.ccsr1_lib import batch_index
    N = T["N"]
    K = r.K
    e = torch.empty(N, dtype=torch.int8, device=dev)
    q = torch.empty(N, K, dtype=torch.float16, device=dev)
    ps = torch.empty(N, K, dtype=torch.float16, device=dev)
    order = np.argsort(T["lens"])
    with torch.no_grad():
        for i in range(0, len(order), 256):
            docs = order[i:i + 256]
            idx, mask, ln = batch_index(T["starts"], T["lens"], docs, dev)
            sl, rl = r.heads(r.backbone(T["X"][idx], ln)[mask])
            ti = idx[mask]
            qq = torch.softmax(rl, -1)
            e[ti] = qq.argmax(1).to(torch.int8)
            q[ti] = qq.half()
            ps[ti] = torch.softmax(sl, -1).half()
    d = T[f"d_{bank}"]
    e64 = e.long()
    ce = float((torch.as_tensor(T["z"]["ce_base"], device=dev).double() + d.gather(1, e64.view(-1, 1)).squeeze(1).double()).mean())
    lab = torch.as_tensor(T["z"]["lab_rsem" if prior == "sem" else "lab_rshuf"].astype(np.int64), device=dev)
    ld = torch.bincount(e64, minlength=K).float() / N
    same = np.asarray(T["z"]["doc"][1:]) == np.asarray(T["z"]["doc"][:-1])
    en = e.cpu().numpy()
    summ = {"ce": ce, "load": ld.tolist(), "load_max": float(ld.max()), "load_eff": float(1 / (ld ** 2).sum()),
            "n_dead": int((ld < 1 / (10 * K)).sum()), "agree_region_prior": float((e64 == lab).float().mean()),
            "subject_acc": float((ps.float().argmax(1) == lab).float().mean()),
            "switch_rate": float((en[1:] != en[:-1])[same].mean())}
    return summ, {"e": en, "q": q.cpu().numpy(), "ps": ps.cpu().numpy()}


def run(a, mon):
    import numpy as np
    import torch
    from csr import scur1_paths as P
    from scur1_extract import load as load_ckpt
    dev = guard.setup_cuda_device() if a.device == "cuda" else torch.device("cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    mpath, msha = (P.MILESTONES[a.milestone] if a.milestone in P.MILESTONES else (a.milestone, a.ckpt_sha))
    if not a.test:
        meta = json.loads((guard.ROOT / mpath / "meta.json").read_text())
        if meta["sha256"] != msha or guard.sha256_file(guard.ROOT / mpath / "state.pt") != msha:
            raise guard.GuardError("milestone hash mismatch")
    model, banks_all, _, mmeta = load_ckpt(mpath, dev, a.test)
    for p in model.parameters():
        p.requires_grad_(False)
    seeds_in = sorted({int(x[2]) for x in mmeta["arms"]})
    s_b = seeds_in[0]
    names = {"SEM": f"SEMh_{s_b}", "SHUF": f"SHUFh_{s_b}", "REF": f"ALTG_{s_b}", "G": f"G_{s_b}"}
    del banks_all
    cfgs = json.loads(Path(a.configs).read_text())
    steps = a.steps or P.TRAIN_STEPS
    need = [names["SEM"], names["SHUF"], names["REF"]]
    Ttr = load_table(a.train, "TRAIN", need, dev, model, a.rows_limit)
    Tev = load_table(a.eval, a.eval_part, need, dev, model, a.rows_limit)
    rdir, ddir = P.ROUTERS / a.stage, P.DEPS / a.stage
    rdir.mkdir(parents=True, exist_ok=True)
    ddir.mkdir(parents=True, exist_ok=True)
    for cfg in cfgs:
        if mon is not None and mon.contaminated:
            sys.exit(77)
        bank = names[cfg["bank"]]
        prior = "sem" if cfg["bank"] == "SEM" else "shuf"
        tag = f"{cfg['tag']}_s{a.seed}"
        ckp, outp = rdir / f"{tag}.pt", ddir / f"{tag}_{a.eval_part}.npz"
        if outp.exists():
            continue
        t1 = time.time()
        if ckp.exists():
            from csr.scur1_router import SCURouter
            st = torch.load(ckp, map_location="cpu", weights_only=False)
            if st["cfg"] != cfg:
                raise guard.GuardError(f"router {ckp} trained with a different config")
            r = SCURouter(P.ROUTER_H, a.seed, P.K, cfg.get("feats", "base"), cfg["mode"], float(cfg.get("rho", 1.0)),
                          P.PSEM_EPS, float(cfg.get("prior_temp", 1.0))).to(dev)
            r.load_state_dict(st["state"])
            curve, info = st["curve"], st["info"]
        else:
            r, curve, info = train_router(cfg, a.seed, Ttr, dev, steps, bank, names["REF"], P)
            tmp = ckp.with_name(f".{ckp.stem}.tmp.pt")
            torch.save({"cfg": cfg, "seed": a.seed, "steps": steps, "state": r.state_dict(), "curve": curve, "info": info,
                        "n_params": r.n_params(), "milestone": mpath, "milestone_sha256": msha,
                        "train_table": a.train}, tmp)
            tmp.rename(ckp)
        r.eval()
        summ, arrays = evaluate(r, Tev, dev, bank, prior)
        tmp = outp.with_name(f".{outp.stem}.tmp.npz")
        np.savez_compressed(tmp, **arrays)
        tmp.rename(outp)
        outp.chmod(0o400)
        summ.update({"cfg": cfg, "seed": a.seed, "bank": bank, "part": a.eval_part, "milestone": mpath,
                     "n_params": r.n_params(), "train_curve_last": curve[-1], "info": info,
                     "router_sha256": guard.sha256_file(ckp), "npz_sha256": guard.sha256_file(outp),
                     "eval_dir": a.eval, "seconds": round(time.time() - t1, 1), "time": guard.now()})
        guard.atomic_write_json(outp.with_suffix(".json"), summ, mode=0o400)
        print(json.dumps({k: summ[k] for k in ("ce", "load_max", "agree_region_prior", "subject_acc", "switch_rate", "seconds")}),
              tag, flush=True)


if __name__ == "__main__":
    main()
