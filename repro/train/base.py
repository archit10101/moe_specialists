"""Stage `train_base`: TOKEN-4 and the EDL family (doc §3, §18).

    loss = CE + 0.01 * Switch + 0.001 * z  [+ beta * L_EDL(group weights of the paper of each token)]

``beta`` is either fixed (explicit EDL, 5e-3) or auto-calibrated once at step 0 so that the EDL gradient
norm is ``target_ratio`` times the LM gradient norm (Semantic / shuffled C6: 2%). The measured
gradient ratio is logged periodically for every EDL run (doc: ~5% early, ~2% late for explicit EDL).

Outputs in the run dir: final.pt (weights), frac{f}.pt snapshots, train_log.json, DONE_train.
"""
from __future__ import annotations

import math
import statistics
import time

import torch

from ..config import Experiment, Profile, edl_cfg, to_dict
from ..data.loader import GroupTable, Split, TrainStream, to_device
from ..model.losses import edl_loss, switch_loss, z_loss
from ..model.moe import MoEGPT, token_ce
from ..utils import autocast, device, is_done, log, mark_done, run_dir, save_torch, seed_all, write_json
from .common import Budget, load_state, lr_at, make_optimizer, set_lr, stop


def grad_norm(model) -> float:
    return math.sqrt(sum(float(p.grad.float().pow(2).sum()) for p in model.parameters() if p.grad is not None))


def lm_and_edl_norms(model, batch, table, ecfg, dev) -> tuple[float, float]:
    """|grad CE| and |grad L_EDL| on one micro-batch (two separate backward passes)."""
    model.zero_grad(set_to_none=True)
    with autocast(dev):
        logits = model(batch["x"])
    token_ce(logits, batch["y"]).mean().backward()
    g_lm = grad_norm(model)
    model.zero_grad(set_to_none=True)
    with autocast(dev):
        model(batch["x"])
    l = edl_loss(model, table.weights_of(batch["doc"]), ecfg.layers, ecfg.min_mass)
    g_edl = 0.0
    if l is not None:
        l.backward()
        g_edl = grad_norm(model)
    model.zero_grad(set_to_none=True)
    return g_lm, g_edl


def train_base(p: Profile, exp: Experiment, seed: int) -> None:
    rd = run_dir(p, exp.name, seed)
    if is_done(rd, "train"):
        log(f"{exp.name}/s{seed}: already trained")
        return
    rd.mkdir(parents=True, exist_ok=True)
    dev, tc, mc = device(), p.train, p.model
    seed_all(seed)
    model = MoEGPT(mc).to(dev)
    log(f"{exp.name}/s{seed}: {model.param_count():,} parameters on {dev}")
    ecfg = edl_cfg(p, exp)
    table = GroupTable.load(p, ecfg.groups) if ecfg.groups else None
    model.set_keep_probs(table is not None)

    accum = max(1, tc.batch_tokens // (tc.micro_bs * mc.ctx))
    total = math.ceil(tc.tokens / (accum * tc.micro_bs * mc.ctx))
    stream = TrainStream(Split(p, "base"), seed, tc.micro_bs, accum)
    opt = make_optimizer(model.parameters(), tc.lr, tc.weight_decay, dev)
    ckpt = rd / "ckpt_last.pt"
    step, extra = 0, dict(beta=ecfg.beta if table is not None else 0.0, log=[], ratio_log=[])
    if ckpt.exists():
        step, extra = load_state(ckpt, {"model": model}, opt)

    if table is not None and ecfg.target_ratio and "calibrated" not in extra:
        ratios = []
        for i in range(8):
            g_lm, g_edl = lm_and_edl_norms(model, to_device(stream.micro_batch(0, i % accum), dev), table, ecfg, dev)
            if g_edl > 0:
                ratios.append(g_lm / g_edl)
        extra["beta"] = ecfg.target_ratio * statistics.median(ratios)
        extra["calibrated"] = True
        log(f"calibrated beta = {extra['beta']:.3e} (target |grad EDL| = {ecfg.target_ratio:.0%} |grad LM|)")
    beta = extra["beta"]

    save_fracs = sorted(set(exp.save_fracs) | set(tc.save_fracs))
    frac_steps = {max(1, round(f * total)): f for f in save_fracs}
    probe_every = max(1, total // 8)
    budget, t_log, tok_log = Budget(), time.time(), 0
    model.train()
    while step < total:
        if table is not None and step % probe_every == 0:
            g_lm, g_edl = lm_and_edl_norms(model, to_device(stream.micro_batch(step, 0), dev), table, ecfg, dev)
            extra["ratio_log"].append(dict(step=step, ratio=beta * g_edl / max(g_lm, 1e-12)))
        set_lr(opt, lr_at(step, total, tc.lr, tc.min_lr, tc.warmup_frac, tc.schedule, tc.decay_frac))
        sums = dict(ce=0.0, edl=0.0, switch=0.0, z=0.0)
        for micro in range(accum):
            b = to_device(stream.micro_batch(step, micro), dev)
            with autocast(dev):
                logits = model(b["x"])
            ce = token_ce(logits, b["y"]).mean()
            sw, zl = switch_loss(model, tc.aux_reduce), z_loss(model, tc.aux_reduce)
            loss = ce + tc.switch_coef * sw + tc.z_coef * zl
            if table is not None:
                le = edl_loss(model, table.weights_of(b["doc"]), ecfg.layers, ecfg.min_mass)
                if le is not None:
                    loss = loss + beta * le
                    sums["edl"] += float(le.detach()) / accum
            (loss / accum).backward()
            sums["ce"] += float(ce) / accum
            sums["switch"] += float(sw) / accum
            sums["z"] += float(zl) / accum
            tok_log += b["x"].numel()
        if not all(map(math.isfinite, sums.values())):
            raise SystemExit(f"non-finite loss at step {step}: {sums}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc.clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % tc.log_every == 0 or step == total:
            dt = time.time() - t_log
            extra["log"].append(dict(step=step, **{k: round(v, 5) for k, v in sums.items()},
                                     tok_per_s=round(tok_log / max(dt, 1e-9))))
            log(f"{exp.name}/s{seed} step {step}/{total} ce {sums['ce']:.4f} edl {sums['edl']:.3f} "
                f"switch {sums['switch']:.3f} {tok_log / max(dt, 1e-9):,.0f} tok/s")
            t_log, tok_log = time.time(), 0
        if step in frac_steps:
            save_torch(rd / f"frac{frac_steps[step]}.pt", model.state_dict())
        code = budget.exit_code()
        if code is not None and step < total:
            stop(code, ckpt, step, {"model": model}, opt, extra)
        if budget.should_checkpoint() and step < total:
            from .common import save_state
            save_state(ckpt, step, {"model": model}, opt, extra)

    save_torch(rd / "final.pt", model.state_dict())
    write_json(rd / "train_log.json", dict(experiment=exp.name, seed=seed, steps=total, beta=beta,
                                           tokens=total * accum * tc.micro_bs * mc.ctx,
                                           params=model.param_count(), edl=to_dict(ecfg),
                                           log=extra["log"], grad_ratio=extra["ratio_log"]))
    ckpt.unlink(missing_ok=True)
    mark_done(rd, "train")
    log(f"{exp.name}/s{seed}: done")


def load_base(p: Profile, exp_name: str, seed: int, ckpt: str = "final") -> MoEGPT:
    """Frozen base model in eval mode."""
    rd = run_dir(p, exp_name, seed)
    path = rd / ("final.pt" if ckpt == "final" else f"{ckpt}.pt")
    if not path.exists():
        raise SystemExit(f"base checkpoint {path} missing: train {exp_name}/s{seed} first")
    model = MoEGPT(p.model)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.to(device()).eval()
    for q in model.parameters():
        q.requires_grad_(False)
    return model
