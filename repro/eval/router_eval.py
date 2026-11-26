"""Stage `eval_router`: the deployed causal router vs the generic bank G and the oracle (doc §27).

On the same val windows it computes, per token:
  router_ce     CE with the router's argmax specialist (causal, deployable)
  refresh16_ce  same, but the decision is only refreshed every 16 tokens
  oracle_ce     CE with the paper's cluster specialist (non-causal teacher assignment)
  g_ce          CE of the generic token-routed bank G (bank.tok, same base, seed = bank seed)
  base_ce       frozen base
and reports means, router - G by position category (first 32 positions / <=64 tokens after a paper
boundary / stable), routing NMI with semantic clusters, the exposure-matched interaction of the routed
specialists, and the prefill time overhead of router + bank.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from ..config import Experiment, Profile, router_cfg
from ..data.loader import GroupTable, Split, eval_windows, to_device
from ..model.moe import token_ce
from ..train.residual import load_bank
from ..train.router import all_specialist_ce, make_router
from ..utils import autocast, is_done, load_torch, log, mark_done, run_dir, write_json
from .metrics import matched_difference, nmi, quantile_bins, summarize


def load_router(p: Profile, exp: Experiment, seed: int, n: int):
    rc = router_cfg(p, exp)
    r = make_router(p, rc, n)
    r.load_state_dict(load_torch(run_dir(p, exp.name, seed) / "router.pt")["state"])
    dev = next(iter(r.parameters())).device
    return r.eval(), rc


def position_category(doc: torch.Tensor) -> torch.Tensor:
    """0 = first 32 positions, 1 = within 64 tokens after a paper boundary, 2 = stable."""
    B, T = doc.shape
    pos = torch.arange(T, device=doc.device).expand(B, T)
    change = torch.zeros_like(doc, dtype=torch.bool)
    change[:, 1:] = doc[:, 1:] != doc[:, :-1]
    last = torch.where(change, pos, torch.zeros_like(pos)).cummax(1).values
    cat = torch.full_like(doc, 2)
    cat[(pos - last < 64) & (last > 0)] = 1
    cat[pos < 32] = 0
    return cat


@torch.no_grad()
def evaluate_router(p: Profile, exp: Experiment, seed: int) -> dict:
    rd = run_dir(p, exp.name, seed)
    if is_done(rd, "eval"):
        log(f"{exp.name}/s{seed}: eval already done")
        return {}
    rc = router_cfg(p, exp)
    base, bm, bc = load_bank(p, rc.bank, rc.bank_seed)
    dev = next(base.parameters()).device
    n = bm.bank.n
    router, _ = load_router(p, exp, seed, n)
    router.to(dev)
    _, g_bank, _ = load_bank(p, "bank.tok", rc.bank_seed)
    prior, sem = GroupTable.load(p, rc.prior_groups), GroupTable.load(p, "sem")
    val = Split(p, "val")
    idx = eval_windows(val, p.eval.val_tokens)
    cols = {k: [] for k in ("router_ce", "refresh16_ce", "oracle_ce", "g_ce", "base_ce", "choice", "sem",
                            "own", "delta", "margin", "pos", "cat")}
    for i in range(0, len(idx), p.eval.batch):
        b = to_device(val.windows(idx[i:i + p.eval.batch]), dev)
        with autocast(dev):
            h = base.trunk(b["x"]).float()
            base_ce = token_ce(base.head(h), b["y"])
            hg, _, _ = g_bank(h, b["doc"])
            g_ce = token_ce(base.head(hg), b["y"])
        U = all_specialist_ce(base, bm.bank, h, b["y"])
        pr = F.softmax(router(h).float(), -1)
        top2 = pr.topk(2, -1)
        choice, alt = top2.indices[..., 0], top2.indices[..., 1]
        T = b["x"].shape[1]
        refresh_idx = (torch.arange(T, device=dev) // 16) * 16
        choice16 = choice[:, refresh_idx]
        own_cluster = prior.hard_of(b["doc"])
        g = lambda ix: U.gather(-1, ix.unsqueeze(-1)).squeeze(-1)
        cols["router_ce"].append(g(choice)); cols["refresh16_ce"].append(g(choice16))
        cols["oracle_ce"].append(g(own_cluster)); cols["g_ce"].append(g_ce); cols["base_ce"].append(base_ce)
        cols["choice"].append(choice); cols["sem"].append(sem.hard_of(b["doc"]))
        cols["own"].append(own_cluster == choice); cols["delta"].append(g(alt) - g(choice))
        cols["margin"].append(top2.values[..., 0] - top2.values[..., 1])
        cols["pos"].append(torch.arange(T, device=dev).expand_as(choice)); cols["cat"].append(position_category(b["doc"]))
    a = {k: torch.cat([x.flatten().float().cpu() for x in v]).numpy() for k, v in cols.items()}
    res = dict(experiment=exp.name, seed=seed, bank=rc.bank, bank_seed=rc.bank_seed, n_spec=n,
               router_params=sum(q.numel() for q in router.parameters()))
    for k in ("router_ce", "refresh16_ce", "oracle_ce", "g_ce", "base_ce"):
        res[k] = float(a[k].mean())
    res["router_minus_g"] = res["router_ce"] - res["g_ce"]
    res["refresh16_minus_g"] = res["refresh16_ce"] - res["g_ce"]
    res["by_position_router_minus_g"] = {
        name: float((a["router_ce"] - a["g_ce"])[a["cat"] == c].mean()) if (a["cat"] == c).any() else None
        for c, name in enumerate(("first32", "after_boundary", "stable"))}
    res["position_token_share"] = {name: float((a["cat"] == c).mean())
                                   for c, name in enumerate(("first32", "after_boundary", "stable"))}
    res["routing_nmi_sem"] = nmi(a["choice"].astype(np.int64), a["sem"].astype(np.int64), n, sem.k)
    res["usage"] = np.bincount(a["choice"].astype(np.int64), minlength=n).tolist()

    # exposure-matched interaction of the routed specialists: on tokens routed to e, how much worse is the
    # router's runner-up for e's own-cluster tokens than for other tokens, inside covariate bins
    bins_all = (quantile_bins(a["margin"], 3) * 4 + quantile_bins(a["base_ce"], 4)) * 4 + (a["pos"] * 4 // p.model.ctx)
    per = []
    for e in range(n):
        m = a["choice"] == e
        if m.sum() < p.eval.min_routed:
            continue
        own = a["own"][m].astype(bool)
        if own.sum() < 20 or (~own).sum() < 20:
            continue
        per.append(matched_difference(a["delta"][m], own, bins_all[m])[0])
    res["matched_interaction"] = summarize(per)
    res["specialists_positive"] = int(sum(x > 0 for x in per if np.isfinite(x)))

    # prefill overhead
    xb = to_device(val.windows(idx[: min(len(idx), p.eval.batch)]), dev)
    def timeit(fn, reps=5):
        fn()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t = time.time()
        for _ in range(reps):
            fn()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        return (time.time() - t) / reps
    def with_router():
        with autocast(dev):
            h = base.trunk(xb["x"])
            ch = router(h.float()).argmax(-1)
            return base.head(bm.bank(h.float(), ch))
    def plain():
        with autocast(dev):
            return base(xb["x"])
    res["prefill_time_ratio"] = timeit(with_router) / timeit(plain)
    log(f"{exp.name}/s{seed}: router {res['router_ce']:.4f} G {res['g_ce']:.4f} oracle {res['oracle_ce']:.4f} "
        f"router-G {res['router_minus_g']:+.5f} matched {res['matched_interaction']['mean']:.5f}")
    write_json(rd / "eval.json", res)
    mark_done(rd, "eval")
    return res
