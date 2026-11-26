"""Stage `train_router`: causal current-context router over a frozen bank (doc §27, ablations B9-B10).

Base and bank are frozen. For each training token we compute the exact CE under every specialist
(U[t, e]; cheap - only the residual and the LM head are re-run). Following the original (ccsr1_router.py):

    u_e   = -(U_e - mean_e U)                         utility of specialist e (centring is softmax-invariant)
    s_u   = median over tokens of std_e(u)            fixed once, before training, on `scale_batches`
    pi    = softmax(u / (util_temp * s_u) + sem_weight * log(prior + 1e-4))
    w_t   = clip(margin_t / median margin, 0.25, 4),  margin = max_e u - mean_e u
    loss  = w_teacher * sum_t w_t KL(pi_t || p_t) / sum_t w_t + w_bal * sum_e pbar_e log(K pbar_e)
            [+ w_ce * sum_e p_e U_e, not in the original; B10]

p is the router's softmax, pbar its batch mean. The prior uses the token's paper cluster (non-causal), so it
exists only in training; at evaluation the router alone picks argmax p, causally. The router trains on the
`split` papers (calib), which neither the base, the bank nor the evaluation (val) ever saw. The target token
is used only to score specialists for the router - specialists never train here (cf. doc §28).
"""
from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F

from ..config import Experiment, Profile, router_cfg, to_dict
from ..data.loader import GroupTable, Split, TrainStream, to_device
from ..model.moe import token_ce
from ..model.residual import CtxRouter
from ..utils import autocast, device, is_done, log, mark_done, run_dir, save_torch, seed_all, write_json
from .common import Budget, load_state, lr_at, make_optimizer, save_state, set_lr, stop
from .residual import load_bank


@torch.no_grad()
def all_specialist_ce(base, bank, h, y, chunk: int = 4) -> torch.Tensor:
    """U [B, T, n]: next-token CE of every token under every specialist."""
    dev = h.device
    out = torch.empty(*y.shape, bank.n, device=dev)
    for i in range(0, h.shape[0], chunk):
        hb = h[i:i + chunk].float()
        for e in range(bank.n):
            with autocast(dev):
                out[i:i + chunk, :, e] = token_ce(base.head(bank.forward_fixed(hb, e)), y[i:i + chunk])
    return out


def make_router(p: Profile, rc, n: int) -> CtxRouter:
    return CtxRouter(p.model.d, n, rc.hidden, rc.inputs, rc.ctx_window, rc.arch)


def train_router(p: Profile, exp: Experiment, seed: int) -> None:
    rd = run_dir(p, exp.name, seed)
    if is_done(rd, "train"):
        log(f"{exp.name}/s{seed}: already trained")
        return
    rd.mkdir(parents=True, exist_ok=True)
    rc = router_cfg(p, exp)
    dev, tc, mc = device(), p.train, p.model
    base, bm, _ = load_bank(p, rc.bank, rc.bank_seed)
    prior = GroupTable.load(p, rc.prior_groups)
    n = bm.bank.n
    assert prior.k == n, f"prior table {rc.prior_groups} has {prior.k} groups but the bank has {n} specialists"
    seed_all(2000 + seed)
    router = make_router(p, rc, n).to(dev)
    log(f"{exp.name}/s{seed}: router {sum(q.numel() for q in router.parameters()):,} params over "
        f"{rc.bank}/s{rc.bank_seed} ({n} specialists)")

    accum = max(1, tc.batch_tokens // (tc.micro_bs * mc.ctx))
    total = math.ceil(rc.tokens / (accum * tc.micro_bs * mc.ctx))
    split = Split(p, rc.split)
    stream = TrainStream(split, 2000 + seed, tc.micro_bs, accum)
    opt = make_optimizer(router.parameters(), rc.lr, rc.weight_decay, dev, rc.betas)
    ckpt = rd / "ckpt_last.pt"
    step, extra = 0, dict(log=[])
    if ckpt.exists():
        step, extra = load_state(ckpt, {"router": router}, opt)

    def utilities(b):
        with torch.no_grad(), autocast(dev):
            h = base.trunk(b["x"]).float()
        U = all_specialist_ce(base, bm.bank, h, b["y"])
        return h, U, -(U - U.mean(-1, keepdim=True))

    if "s_u" not in extra:          # fixed scales, from a fixed set of batches of a separate order
        scale_stream = TrainStream(split, 777, tc.micro_bs, 1)
        sds, margins = [], []
        for i in range(rc.scale_batches):
            _, _, u = utilities(to_device(scale_stream.micro_batch(i, 0), dev))
            sds.append(u.std(-1).flatten().cpu())
            margins.append((u.max(-1).values - u.mean(-1)).flatten().cpu())
        extra["s_u"] = float(torch.cat(sds).median())
        extra["margin_median"] = float(torch.cat(margins).median())
        log(f"{exp.name}/s{seed}: s_u = {extra['s_u']:.5f}, median margin = {extra['margin_median']:.5f}")
    tau = rc.util_temp * extra["s_u"] + 1e-12
    budget, t0 = Budget(), time.time()
    while step < total:
        set_lr(opt, lr_at(step, total, rc.lr, rc.lr * 0.1, 0.0, "cosine", warmup_steps=rc.warmup_steps))
        sums = dict(loss=0.0, kl=0.0, bal=0.0, exp_ce=0.0, routed_ce=0.0, teacher_ce=0.0)
        for micro in range(accum):
            b = to_device(stream.micro_batch(step, micro), dev)
            h, U, u = utilities(b)
            logq = F.log_softmax(u / tau + rc.sem_weight * torch.log(prior.weights_of(b["doc"]) + 1e-4), -1)
            logp = F.log_softmax(router(h).float(), -1)
            if rc.token_weights:
                w = ((u.max(-1).values - u.mean(-1)) / (extra["margin_median"] + 1e-12)).clamp(0.25, 4.0)
            else:
                w = torch.ones_like(u[..., 0])
            kl = ((logq.exp() * (logq - logp)).sum(-1) * w).sum() / w.sum()
            pbar = logp.exp().reshape(-1, n).mean(0)
            bal = (pbar * torch.log(n * pbar + 1e-12)).sum()
            exp_ce = (logp.exp() * U).sum(-1).mean()
            loss = rc.w_teacher * kl + rc.w_bal * bal + rc.w_ce * exp_ce
            (loss / accum).backward()
            with torch.no_grad():
                sums["loss"] += float(loss) / accum
                sums["kl"] += float(kl) / accum
                sums["bal"] += float(bal) / accum
                sums["exp_ce"] += float(exp_ce) / accum
                sums["routed_ce"] += float(U.gather(-1, logp.argmax(-1, keepdim=True)).mean()) / accum
                sums["teacher_ce"] += float(U.gather(-1, logq.argmax(-1, keepdim=True)).mean()) / accum
        torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % tc.log_every == 0:
            extra["log"].append(dict(step=step, **{k: round(v, 5) for k, v in sums.items()}))
            log(f"{exp.name}/s{seed} step {step}/{total} kl {sums['kl']:.4f} routed_ce {sums['routed_ce']:.4f} "
                f"teacher_ce {sums['teacher_ce']:.4f} {time.time() - t0:.0f}s")
        step += 1
        code = budget.exit_code()
        if code is not None and step < total:
            stop(code, ckpt, step, {"router": router}, opt, extra)
        if budget.should_checkpoint() and step < total:
            save_state(ckpt, step, {"router": router}, opt, extra)
    save_torch(rd / "router.pt", dict(state=router.state_dict(), cfg=to_dict(rc)))
    write_json(rd / "train_log.json", dict(experiment=exp.name, seed=seed, steps=total, router=to_dict(rc),
                                           params=sum(q.numel() for q in router.parameters()),
                                           s_u=extra["s_u"], margin_median=extra["margin_median"], log=extra["log"]))
    ckpt.unlink(missing_ok=True)
    mark_done(rd, "train")
