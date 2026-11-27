"""Stage `eval_bank`: residual-bank evaluation (doc §23).

  base_ce / bank_ce / improvement    val CE of the frozen base, of base + bank, and base - bank
  per_cluster_gain                    improvement inside each semantic cluster (and each domain)
  swap                                forced-equal-exposure swap matrix on a val probe: every token is pushed
                                      through every specialist; M[g, j] = CE of group-g tokens under j.
                                      specificity = mean diagonal of the double-centred utility, with a
                                      paper-level hierarchical bootstrap CI; own_best_frac = share of groups
                                      whose own specialist is their best one
  routing_nmi                         for learned routers: NMI(chosen specialist; semantic cluster)
"""
from __future__ import annotations

import numpy as np
import torch

from ..config import Experiment, Profile, bank_base_seed
from ..data.loader import GroupTable, Split, eval_windows, to_device
from ..model.moe import token_ce
from ..train.residual import load_bank
from ..utils import autocast, is_done, log, mark_done, run_dir, write_json
from .metrics import nmi, swap_specificity


@torch.no_grad()
def bank_pass(base, bm, split: Split, idx, batch: int, all_specs: bool = False):
    dev = next(base.parameters()).device
    out = {k: [] for k in ("base_ce", "bank_ce", "assign", "doc", "U")}
    for i in range(0, len(idx), batch):
        b = to_device(split.windows(idx[i:i + batch]), dev)
        with autocast(dev):
            h = base.trunk(b["x"]).float()
            out["base_ce"].append(token_ce(base.head(h), b["y"]).flatten().cpu())
            h2, a, _ = bm(h, b["doc"])
            out["bank_ce"].append(token_ce(base.head(h2), b["y"]).flatten().cpu())
            if all_specs:
                out["U"].append(torch.stack([token_ce(base.head(bm.bank.forward_fixed(h, e)), b["y"]).flatten()
                                             for e in range(bm.bank.n)], -1).cpu())
        out["assign"].append(a.flatten().cpu())
        out["doc"].append(b["doc"].flatten().cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items() if v}


def evaluate_bank(p: Profile, exp: Experiment, seed: int) -> dict:
    rd = run_dir(p, exp.name, seed)
    if is_done(rd, "eval"):
        log(f"{exp.name}/s{seed}: eval already done")
        return {}
    base, bm, bc = load_bank(p, exp.name, seed)
    sem, dom = GroupTable.load(p, "sem"), GroupTable.load(p, "domain")
    val = Split(p, "val")
    v = bank_pass(base, bm, val, eval_windows(val, p.eval.val_tokens), p.eval.batch)
    gain = v["base_ce"] - v["bank_ce"]
    res = dict(experiment=exp.name, seed=seed, routing=bc.routing, groups=bc.groups, n_spec=bm.bank.n,
               base=bc.base, base_seed=bank_base_seed(bc, seed), base_ckpt=bc.base_ckpt,
               hidden=bm.bank.hidden, params=sum(q.numel() for q in bm.bank.parameters()),
               base_ce=float(v["base_ce"].mean()),   # val CE of the frozen base checkpoint (B4: base competence)
               bank_ce=float(v["bank_ce"].mean()),
               improvement=float(gain.mean()))
    for name, tab in (("sem", sem), ("domain", dom)):
        g = tab.hard[torch.from_numpy(v["doc"])].numpy()
        res[f"per_{name}_gain"] = [float(gain[g == k].mean()) if (g == k).any() else None for k in range(tab.k)]
    res["clusters_improved_frac"] = float(np.mean([x > 0 for x in res["per_sem_gain"] if x is not None]))
    if bc.routing == "token":
        g = sem.hard[torch.from_numpy(v["doc"])].numpy()
        res["routing_nmi_sem"] = nmi(v["assign"], g, bm.bank.n, sem.k)
    log(f"{exp.name}/s{seed}: base {res['base_ce']:.4f} bank {res['bank_ce']:.4f} "
        f"improvement {res['improvement']:+.5f}")

    if bm.bank.n > 1:   # forced-equal-exposure swap matrix on a probe subset of val
        n_win = p.eval.probe_windows_per_group * bm.bank.n
        s = bank_pass(base, bm, val, eval_windows(val, n_win * p.model.ctx), p.eval.batch, all_specs=True)
        own_table = bm.table if bm.table is not None else (sem if sem.k == bm.bank.n else None)
        if own_table is not None:
            g = own_table.hard[torch.from_numpy(s["doc"])].numpy()
            res["swap_own"] = swap_specificity(s["U"], g, s["doc"], bm.bank.n)
            res["swap_own"]["table"] = own_table.name
            log(f"{exp.name}/s{seed}: specificity ({own_table.name}) {res['swap_own']['specificity']:.5f} "
                f"CI {res['swap_own']['ci']}")
        if sem.k == bm.bank.n and (own_table is None or own_table.name != "sem"):
            g = sem.hard[torch.from_numpy(s["doc"])].numpy()
            res["swap_sem"] = swap_specificity(s["U"], g, s["doc"], bm.bank.n)
    write_json(rd / "eval.json", res)
    mark_done(rd, "eval")
    return res
