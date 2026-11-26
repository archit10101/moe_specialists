"""Stage `train_bank`: residual specialist banks on a frozen base (doc §23, ablations B4-B8).

Routing modes
  oracle  specialist = hard cluster of the token's paper (whole-paper assignment, non-causal)
  token   causal linear top-1 router on the current hidden state, + Switch balance (= generic bank G)
  dense   a single always-on residual MLP (capacity controls, B6)

The base is frozen and run without gradients; only the bank (and token router) train, on the "bank"
split (new tokens the base never saw). Outputs: bank.pt, train_log.json, DONE_train.
"""
from __future__ import annotations

import math
import time

import torch

from ..config import BankCfg, Experiment, Profile, bank_cfg, get_experiment, to_dict
from ..data.loader import GroupTable, Split, TrainStream, to_device
from ..model.moe import token_ce
from ..model.residual import Bank, TokenRouter
from ..utils import autocast, device, is_done, load_torch, log, mark_done, run_dir, save_torch, seed_all, write_json
from .base import load_base
from .common import Budget, load_state, lr_at, make_optimizer, save_state, set_lr, stop


def bank_shape(p: Profile, bc: BankCfg) -> tuple[int, int]:
    """(number of specialists, hidden units per specialist)."""
    if bc.routing == "dense":
        return 1, (bc.total_hidden if bc.width == "total" else bc.spec_hidden)
    k = GroupTable.load(p, bc.groups).k
    hidden = {"spec": bc.spec_hidden, "split": bc.total_hidden // k, "total": bc.total_hidden}[bc.width]
    return k, hidden


class BankModel(torch.nn.Module):
    """Bank + its routing rule. ``assign(h, doc)`` -> (specialist ids, grad scale, balance loss)."""

    def __init__(self, p: Profile, bc: BankCfg):
        super().__init__()
        self.bc = bc
        n, hidden = bank_shape(p, bc)
        self.bank = Bank(p.model.d, n, hidden)
        self.router = TokenRouter(p.model.d, n) if bc.routing == "token" else None
        self.table = GroupTable.load(p, bc.groups) if bc.routing == "oracle" else None

    def assign(self, h, doc):
        if self.bc.routing == "token":
            return self.router(h)
        if self.bc.routing == "oracle":
            return self.table.hard_of(doc), None, None
        return torch.zeros_like(doc), None, None

    def forward(self, h, doc):
        a, scale, bal = self.assign(h, doc)
        return self.bank(h, a, scale), a, bal


def train_bank(p: Profile, exp: Experiment, seed: int) -> None:
    rd = run_dir(p, exp.name, seed)
    if is_done(rd, "train"):
        log(f"{exp.name}/s{seed}: already trained")
        return
    rd.mkdir(parents=True, exist_ok=True)
    bc = bank_cfg(p, exp)
    dev, tc, mc = device(), p.train, p.model
    base = load_base(p, bc.base, bc.base_seed, bc.base_ckpt)
    seed_all(1000 + seed)
    bm = BankModel(p, bc).to(dev)
    n_params = sum(q.numel() for q in bm.bank.parameters())
    log(f"{exp.name}/s{seed}: {bm.bank.n} specialists x {bm.bank.hidden} hidden = {n_params:,} params "
        f"({bc.routing} routing, groups {bc.groups}, base {bc.base}/s{bc.base_seed}:{bc.base_ckpt})")

    accum = max(1, tc.batch_tokens // (tc.micro_bs * mc.ctx))
    total = math.ceil(bc.tokens / (accum * tc.micro_bs * mc.ctx))
    stream = TrainStream(Split(p, "bank"), 1000 + seed, tc.micro_bs, accum)
    opt = make_optimizer(bm.parameters(), bc.lr, 0.0, dev)
    ckpt = rd / "ckpt_last.pt"
    step, extra = 0, dict(log=[])
    if ckpt.exists():
        step, extra = load_state(ckpt, {"bank": bm}, opt)
    budget, t0 = Budget(), time.time()
    while step < total:
        set_lr(opt, lr_at(step, total, bc.lr, bc.lr * 0.1, 0.02))
        sums = dict(ce=0.0, base_ce=0.0, balance=0.0)
        for micro in range(accum):
            b = to_device(stream.micro_batch(step, micro), dev)
            with torch.no_grad(), autocast(dev):
                h = base.trunk(b["x"])
            with autocast(dev):
                h2, _, bal = bm(h.float(), b["doc"])
                ce = token_ce(base.head(h2), b["y"]).mean()
            loss = ce + (bc.balance_coef * bal if bal is not None else 0.0)
            (loss / accum).backward()
            sums["ce"] += float(ce) / accum
            sums["balance"] += float(bal) / accum if bal is not None else 0.0
            if step % tc.log_every == 0:
                with torch.no_grad(), autocast(dev):
                    sums["base_ce"] += float(token_ce(base.head(h), b["y"]).mean()) / accum
        torch.nn.utils.clip_grad_norm_(bm.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % tc.log_every == 0:
            extra["log"].append(dict(step=step, **{k: round(v, 5) for k, v in sums.items()}))
            log(f"{exp.name}/s{seed} step {step}/{total} ce {sums['ce']:.4f} "
                f"(base {sums['base_ce']:.4f}) {time.time() - t0:.0f}s")
        step += 1
        code = budget.exit_code()
        if code is not None and step < total:
            stop(code, ckpt, step, {"bank": bm}, opt, extra)
        if budget.should_checkpoint() and step < total:
            save_state(ckpt, step, {"bank": bm}, opt, extra)
    save_torch(rd / "bank.pt", dict(state=bm.state_dict(), cfg=to_dict(bc)))
    write_json(rd / "train_log.json", dict(experiment=exp.name, seed=seed, steps=total, bank=to_dict(bc),
                                           n_spec=bm.bank.n, hidden=bm.bank.hidden, params=n_params,
                                           log=extra["log"]))
    ckpt.unlink(missing_ok=True)
    mark_done(rd, "train")


def load_bank(p: Profile, exp_name: str, seed: int):
    """(frozen base, frozen BankModel, BankCfg) of a trained bank run."""
    exp = get_experiment(exp_name)
    bc = bank_cfg(p, exp)
    path = run_dir(p, exp_name, seed) / "bank.pt"
    if not path.exists():
        raise SystemExit(f"{path} missing: train {exp_name}/s{seed} first")
    base = load_base(p, bc.base, bc.base_seed, bc.base_ckpt)
    bm = BankModel(p, bc)
    bm.load_state_dict(load_torch(path)["state"])
    bm.to(device()).eval()
    for q in bm.parameters():
        q.requires_grad_(False)
    return base, bm, bc
