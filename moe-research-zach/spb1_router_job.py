#!/usr/bin/env python
"""spb1: train fresh unrestricted causal routers over frozen banks on CALIB-SEARCH TRAIN (paper-half "EVAL") and deploy
them on SEL (guarded GPU job). Base and every specialist are frozen; the training-time routers are not used.

Router input = [rms0(x_t), causal mean of rms0(x) over the last 16 document positions] (+ optional frozen causal
semantic posterior, secondary ablation) recomputed from the frozen base for every scored token and aligned token by token
with the extraction tables. Loss = expected routed CE over the bank's exact per-token CE table + bal * KL(qbar||U).
No mask, no semantic loss, no subject->expert mapping, no logit bias.

  spb1_router_job.py --seed S --tables DIR --common DIR --configs CFG.json --stage NAME [--steps N] [--rows-limit N]
CFG.json: list of {"tag", "bank" (bank name or "COARSE16"), "router_seed", "post" (bool), "H", "temp", "bal"}.
Writes ROUTERS/<stage>/<tag>_s<seed>_r<rseed>.pt and _SEL.npz (e int8, p1 fp16 top-1 prob, marg fp16 top-1 minus
top-2 logit) + .json summary. Existing outputs are skipped (resumable per router)."""
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
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--common", required=True)
    ap.add_argument("--configs", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--rows-limit", type=int, default=0)
    ap.add_argument("--out-root", default="")
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
        guard.occupancy_gate(f"spb1 routers {a.stage}")
        mon = guard.OccupancyMonitor(interval=15.0)
        mon.start()
    guard.register_self(f"spb1 routers {a.stage}")
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
        guard.gpu_ledger(t_iso, guard.now(), f"spb1_routers_{a.stage}_s{a.seed}", "spb1 routers", status)
    print(json.dumps({"status": status, "seconds": round(time.time() - t0, 1)}), flush=True)
    sys.exit(rc)


def hidden_table(model, sp, rows, dev, micro_rows=16):
    """rms0(x) (fp16) for every scored token in exactly the extraction order (same rows / micro-batching / flattening)."""
    import torch
    from csr.orv1_lib import base_forward, make_batch, to_dev
    from csr.spb1_lib import rms0
    xs, ts = [], []
    with torch.no_grad():
        for i in range(0, len(rows), micro_rows):
            bt = to_dev(make_batch(rows[i:i + micro_rows], sp), dev)
            x, _ = base_forward(model, bt)
            tgt = bt["targets"].reshape(-1)
            v = tgt >= 0
            xs.append(rms0(x[v]).to(torch.float16))
            ts.append(tgt[v])
    return torch.cat(xs), torch.cat(ts)


def load_part(a, part, banks, model, dev):
    import numpy as np
    import torch
    from csr.spb1_lib import doc_index
    from spb1_extract import partition
    cdir, tdir = Path(a.common), Path(a.tables)
    cm = json.loads((cdir / f"common_{part}.json").read_text())
    if guard.sha256_file(cdir / f"common_{part}.npz") != cm["sha256"]:
        raise guard.GuardError("common table hash mismatch")
    z = dict(np.load(cdir / f"common_{part}.npz"))
    sp, rows, _, _, _ = partition(part)
    if a.rows_limit:
        rows = rows[:a.rows_limit]
    X, tgt = hidden_table(model, sp, rows, dev)
    if X.shape[0] != len(z["tgt"]) or not bool((tgt.cpu() == torch.as_tensor(z["tgt"].astype(np.int64))).all()):
        raise guard.GuardError(f"hidden table not aligned with {cdir}")
    starts, lens = doc_index(z["doc"], z["pos"])
    T = {"X": X, "starts": starts, "lens": lens, "z": z, "N": len(z["tgt"])}
    summ = json.loads((tdir / f"summary_{part}.json").read_text())
    for b in banks:
        if b == "COARSE16":
            ks = json.loads((cdir / f"coarse_summary_{part}.json").read_text())
            f = cdir / f"coarse_{part}.npz"
            if guard.sha256_file(f) != ks["sha256"]:
                raise guard.GuardError("coarse table hash mismatch")
        else:
            f = tdir / f"bank_{b}_{part}.npz"
            if guard.sha256_file(f) != summ["sha256"][f.name]:
                raise guard.GuardError(f"table hash mismatch {f}")
        T[f"d_{b}"] = torch.as_tensor(np.load(f)["d"].astype(np.float32), device=dev)
    T["lab_sem"] = torch.as_tensor(z["lab_sem"].astype(np.int64), device=dev)
    return T


def sched(step, steps, lr0):
    return lr0 * min(1.0, (step + 1) / 30) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps)))


def train_subject(T, dev, steps, P, seed=0):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from csr.spb1_lib import SubjectNet, batch_index
    torch.manual_seed(seed)
    net = SubjectNet(64, seed, P.K).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=P.ROUTER_LR, weight_decay=P.ROUTER_WD, betas=(0.9, 0.99))
    rng = np.random.default_rng(2_000_003 * seed + 13)
    nd = len(T["starts"])
    perm, pi_ = rng.permutation(nd), 0
    for step in range(steps):
        for gp in opt.param_groups:
            gp["lr"] = sched(step, steps, P.ROUTER_LR)
        if pi_ + P.DOCS_PER_STEP > nd:
            perm, pi_ = rng.permutation(nd), 0
        docs = perm[pi_:pi_ + P.DOCS_PER_STEP]
        pi_ += P.DOCS_PER_STEP
        idx, mask, _ = batch_index(T["starts"], T["lens"], docs, dev)
        lg = net(T["X"][idx])[mask]
        loss = F.cross_entropy(lg, T["lab_sem"][idx[mask]])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), P.ROUTER_CLIP)
        opt.step()
    return net.eval()


def posterior(net, T, dev):
    import numpy as np
    import torch
    from csr.spb1_lib import batch_index
    out = torch.empty(T["N"], net.out.out_features, dtype=torch.float16, device=dev)
    order = np.argsort(T["lens"])
    with torch.no_grad():
        for i in range(0, len(order), 256):
            idx, mask, _ = batch_index(T["starts"], T["lens"], order[i:i + 256], dev)
            out[idx[mask]] = torch.softmax(net(T["X"][idx])[mask], -1).half()
    return out


def train_router(cfg, T, dev, steps, P):
    import numpy as np
    import torch
    from csr.spb1_lib import FreeRouter, batch_index, router_loss
    rs = int(cfg["router_seed"])
    d = T[f"d_{cfg['bank']}"]
    E = d.shape[1]
    post = T.get("post") if cfg.get("post") else None
    torch.manual_seed(rs)
    r = FreeRouter(int(cfg.get("H", 64)), rs, E, n_extra=0 if post is None else post.shape[1],
                   temp=float(cfg.get("temp", 1.0))).to(dev)
    opt = torch.optim.AdamW(r.parameters(), lr=P.ROUTER_LR, weight_decay=P.ROUTER_WD, betas=(0.9, 0.99))
    rng = np.random.default_rng(1_000_003 * rs + 11)
    nd = len(T["starts"])
    perm, pi_ = rng.permutation(nd), 0
    bal = float(cfg.get("bal", 0.01))
    curve = []
    for step in range(steps):
        for gp in opt.param_groups:
            gp["lr"] = sched(step, steps, P.ROUTER_LR)
        if pi_ + P.DOCS_PER_STEP > nd:
            perm, pi_ = rng.permutation(nd), 0
        docs = perm[pi_:pi_ + P.DOCS_PER_STEP]
        pi_ += P.DOCS_PER_STEP
        idx, mask, _ = batch_index(T["starts"], T["lens"], docs, dev)
        lg = r(T["X"][idx], None if post is None else post[idx])[mask]
        loss, lm, b = router_loss(lg, d[idx[mask]], bal)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(r.parameters(), P.ROUTER_CLIP))
        opt.step()
        if step % 100 == 0 or step == steps - 1:
            curve.append({"step": step, "loss": float(loss.detach()), "lm": lm, "bal": b, "gn": gn})
    return r.eval(), curve


def deploy(r, T, dev, bank, post=None):
    import numpy as np
    import torch
    from csr.spb1_lib import batch_index
    N, E = T["N"], r.E
    e = torch.empty(N, dtype=torch.int8 if E <= 127 else torch.int16, device=dev)
    p1 = torch.empty(N, dtype=torch.float16, device=dev)
    mg = torch.empty(N, dtype=torch.float16, device=dev)
    order = np.argsort(T["lens"])
    with torch.no_grad():
        for i in range(0, len(order), 256):
            idx, mask, _ = batch_index(T["starts"], T["lens"], order[i:i + 256], dev)
            lg = r(T["X"][idx], None if post is None else post[idx])[mask]
            ti = idx[mask]
            t2 = lg.topk(2, -1)
            e[ti] = t2.indices[:, 0].to(e.dtype)
            p1[ti] = torch.softmax(lg, -1).max(-1).values.half()
            mg[ti] = (t2.values[:, 0] - t2.values[:, 1]).half()
    d = T[f"d_{bank}"]
    e64 = e.long()
    ce_base = torch.as_tensor(T["z"]["ce_base"], device=dev).double()
    ce = float((ce_base + d.gather(1, e64.view(-1, 1)).squeeze(1).double()).mean())
    ld = (torch.bincount(e64, minlength=E).float() / N).cpu().numpy()
    same = np.asarray(T["z"]["doc"][1:]) == np.asarray(T["z"]["doc"][:-1])
    en = e.cpu().numpy()
    nz = ld[ld > 0]
    summ = {"ce": ce, "ce_base": float(ce_base.mean()), "load": ld.round(5).tolist(), "load_max": float(ld.max()),
            "eff_exp_entropy": float(np.exp(-(nz * np.log(nz)).sum())), "eff_inv_simpson": float(1 / (ld ** 2).sum()),
            "n_dead": int((ld < 1 / (10 * E)).sum()), "n_used": int((ld > 0).sum()),
            "switch_rate": float((en[1:] != en[:-1])[same].mean())}
    return summ, {"e": en, "p1": p1.cpu().numpy(), "marg": mg.cpu().numpy()}


def run(a, mon):
    import numpy as np
    import torch
    from csr.orv1_lib import load_base, state_hash
    from csr import spb1_paths as P
    dev = guard.setup_cuda_device() if a.device == "cuda" else torch.device("cpu")
    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    cfgs = json.loads(Path(a.configs).read_text())
    od = (Path(a.out_root) if a.out_root else P.ROUTERS) / a.stage
    od.mkdir(parents=True, exist_ok=True)
    todo = [c for c in cfgs if not (od / f"{c['tag']}_s{a.seed}_r{c['router_seed']}_SEL.json").exists()]
    if not todo:
        print("all routers exist")
        return
    model, _ = load_base(a.seed, dev)
    h0 = state_hash(model)
    banks = sorted({c["bank"] for c in todo})
    TR = load_part(a, "TRAIN", banks, model, dev)
    SE = load_part(a, "SEL", banks, model, dev)
    steps = a.steps or P.ROUTER_STEPS
    if any(c.get("post") for c in todo):
        net = train_subject(TR, dev, steps, P)
        TR["post"], SE["post"] = posterior(net, TR, dev), posterior(net, SE, dev)
        acc = float((SE["post"].float().argmax(1) == SE["lab_sem"]).float().mean())
        torch.save({"state": net.state_dict(), "sel_acc": acc}, od / f"subjectnet_s{a.seed}.pt")
        print(json.dumps({"subjectnet_sel_acc": acc}), flush=True)
    for c in todo:
        if mon is not None and mon.contaminated:
            sys.exit(77)
        t0 = time.time()
        r, curve = train_router(c, TR, dev, steps, P)
        summ, arr = deploy(r, SE, dev, c["bank"], SE.get("post") if c.get("post") else None)
        stem = f"{c['tag']}_s{a.seed}_r{c['router_seed']}"
        torch.save({"state": r.state_dict(), "cfg": c, "seed": a.seed, "n_params": r.n_params(), "curve": curve},
                   od / f"{stem}.pt")
        p = od / f"{stem}_SEL.npz"
        tmp = p.with_name(f".{p.stem}.tmp.npz")
        np.savez(tmp, **arr)
        tmp.rename(p)
        p.chmod(0o400)
        summ.update({"cfg": c, "seed": a.seed, "n_params": r.n_params(), "steps": steps, "seconds": round(time.time() - t0, 1),
                     "npz_sha256": guard.sha256_file(p), "curve_last": curve[-1], "tables": a.tables})
        guard.atomic_write_json(od / f"{stem}_SEL.json", summ, mode=0o400)
        print(json.dumps({"router": stem, "ce": summ["ce"], "eff": summ["eff_exp_entropy"],
                          "load_max": summ["load_max"], "switch": summ["switch_rate"]}), flush=True)
    if state_hash(model) != h0:
        raise guard.GuardError("frozen base changed")


if __name__ == "__main__":
    main()
