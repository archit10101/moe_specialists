"""`python -m repro.run bench`: time one optimizer step of each job type on this machine with random
tokens (no data needed) and project wall time per job and for each tier. Run it once on an ORCD GPU
(slurm/submit_all.sh --bench) before the full submission to confirm the 6 h job limits are safe."""
from __future__ import annotations

import math
import time

import torch

from .config import EXPERIMENTS, Profile, runnable
from .model.losses import edl_loss, switch_loss, z_loss
from .model.moe import MoEGPT, token_ce
from .model.residual import Bank, CtxRouter
from .pipeline import build_graph
from .train.common import make_optimizer
from .utils import autocast, device, log, results_dir, write_json


def _time(fn, reps: int, dev) -> float:
    fn()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t = time.time()
    for _ in range(reps):
        fn()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t) / reps


def bench(p: Profile, reps: int = 3) -> dict:
    dev, tc, mc = device(), p.train, p.model
    model = MoEGPT(mc).to(dev)
    opt = make_optimizer(model.parameters(), 1e-4, 0.1, dev)
    B, T = tc.micro_bs, mc.ctx
    accum = max(1, tc.batch_tokens // (B * T))
    x = torch.randint(0, 50000 if mc.vocab_size > 50000 else mc.vocab_size - 1, (B, T + 1), device=dev)
    w = torch.nn.functional.one_hot(torch.randint(0, 5, (B, T), device=dev), 5).float()

    def base_step(edl: bool):
        model.set_keep_probs(edl)
        def f():
            for _ in range(accum):
                with autocast(dev):
                    logits = model(x[:, :-1])
                loss = token_ce(logits, x[:, 1:]).mean() + 0.01 * switch_loss(model) + 0.001 * z_loss(model)
                if edl:
                    loss = loss + 5e-3 * edl_loss(model, w, "all", 32.0)
                loss.backward()
            opt.step(); opt.zero_grad(set_to_none=True)
        return f

    res = dict(device=str(dev), gpu=torch.cuda.get_device_name() if dev.type == "cuda" else "cpu",
               params=model.param_count())
    res["base_step_s"] = _time(base_step(False), reps, dev)
    res["edl_step_s"] = _time(base_step(True), reps, dev)
    for q in model.parameters():
        q.requires_grad_(False)
    bank = Bank(mc.d, 16, p.bank.spec_hidden).to(dev)
    bopt = make_optimizer(bank.parameters(), 1e-3, 0.0, dev)
    assign = torch.randint(0, 16, (B, T), device=dev)

    def bank_step():
        for _ in range(accum):
            with torch.no_grad(), autocast(dev):
                h = model.trunk(x[:, :-1])
            with autocast(dev):
                token_ce(model.head(bank(h.float(), assign)), x[:, 1:]).mean().backward()
        bopt.step(); bopt.zero_grad(set_to_none=True)

    res["bank_step_s"] = _time(bank_step, reps, dev)
    router = CtxRouter(mc.d, 16).to(dev)
    ropt = make_optimizer(router.parameters(), 1e-3, 0.0, dev)
    from .train.router import all_specialist_ce

    def router_step():
        for _ in range(accum):
            with torch.no_grad(), autocast(dev):
                h = model.trunk(x[:, :-1]).float()
            U = all_specialist_ce(model, bank, h, x[:, 1:])
            (torch.log_softmax(router(h), -1).exp() * U).sum(-1).mean().backward()
        ropt.step(); ropt.zero_grad(set_to_none=True)

    res["router_step_s"] = _time(router_step, reps, dev)
    tok_step = accum * B * T
    res["tokens_per_s_base"] = tok_step / res["base_step_s"]
    hours = dict(base=math.ceil(tc.tokens / tok_step) * res["edl_step_s"] / 3600 * 1.15,   # +15% eval
                 bank=math.ceil(p.bank.tokens / tok_step) * res["bank_step_s"] / 3600 * 1.3,
                 router=math.ceil(p.router.tokens / tok_step) * res["router_step_s"] / 3600 * 1.3)
    res["projected_hours_per_job"] = hours
    for tier in ("core", "all"):
        jobs = build_graph(p, tier)
        res[f"projected_gpu_hours_{tier}"] = sum(hours.get(j.stage, 0.3 if j.stage == "cluster" else 0) for j in jobs)
    log(f"bench on {res['gpu']}: {res['tokens_per_s_base']:,.0f} tok/s base training; projected hours per job "
        + ", ".join(f"{k} {v:.2f}" for k, v in hours.items())
        + f"; core {res['projected_gpu_hours_core']:.1f} GPU-h, all {res['projected_gpu_hours_all']:.1f} GPU-h")
    for k, v in hours.items():
        if v > 5.5:
            log(f"WARNING: a {k} job is projected at {v:.1f} h, above the 6 h limit; it will stop at the time budget "
                f"and resume when you re-run submit_all.sh")
    write_json(results_dir(p) / "bench.json", res)
    return res
