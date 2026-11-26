"""Stage `eval_base`: CE, routing specialisation and functional specialisation of a base model.

Metrics (doc §2.3-2.4), each computed for two group tables: ``domain`` (true labels) and ``sem``
(label-free clusters):
  val_ce         mean next-token CE on the val split (nats/token)
  nmi            NMI(group; selected expert) per layer over all top-k slots, averaged over layers
  fsi            group ablation: remove each group's most enriched experts (fsi_experts_per_layer per
                 layer, all layers at once); FSI = mean_g [dCE on g - dCE on other groups]
  interaction    per-expert replacement: ban expert e in layer l (next-ranked expert admitted, k stay
                 active, downstream recomputed); own group = the group with highest routing lift;
                 ordinary = dCE(own tokens) - dCE(other tokens), all tokens
  matched        exposure-matched version: only tokens routed to e, compared inside bins of
                 router rank x gate x margin-to-replacement x baseline CE x position (Project-davin port)

Probe metrics use the calib split (never trained on, not used for model selection).
"""
from __future__ import annotations

import numpy as np
import torch

from ..config import Experiment, Profile, to_dict
from ..data.loader import GroupTable, Split, eval_windows, to_device
from ..model.moe import token_ce
from ..train.base import load_base
from ..utils import autocast, device, is_done, log, mark_done, run_dir, write_json
from .metrics import matched_difference, nmi_from_counts, quantile_bins, summarize

TABLES = ("domain", "sem")


@torch.no_grad()
def run_windows(model, split: Split, idx: np.ndarray, batch: int, records: bool = False,
                count_tables: dict | None = None):
    """CE per token, optionally full routing records (probe sets) and/or streamed (group, expert) counts
    for the given group tables (val set; avoids holding every routing decision in memory).
    Returns dict of numpy arrays (tokens flattened in window order) + "counts": {table: [L, G, E]}."""
    dev = next(model.parameters()).device
    out = {k: [] for k in ("ce", "doc", "pos", "top_i", "top_p", "top_raw", "next_p")}
    counts = {}
    for i in range(0, len(idx), batch):
        b = to_device(split.windows(idx[i:i + batch]), dev)
        with autocast(dev):
            logits = model(b["x"])
        out["ce"].append(token_ce(logits, b["y"]).flatten().cpu())
        out["doc"].append(b["doc"].flatten().cpu())
        out["pos"].append(torch.arange(b["x"].shape[1]).repeat(b["x"].shape[0]))
        ms = model.moe_layers()
        if records:
            out["top_i"].append(torch.stack([m.top_i for m in ms]).to(torch.int16).cpu())
            out["top_p"].append(torch.stack([m.top_p for m in ms]).half().cpu())
            out["top_raw"].append(torch.stack([m.top_raw for m in ms]).half().cpu())
            out["next_p"].append(torch.stack([m.next_p for m in ms]).half().cpu())
        for name, tab in (count_tables or {}).items():
            g = tab.hard_of(b["doc"]).flatten().repeat_interleave(ms[0].top_k)
            E = ms[0].n_exp
            c = torch.stack([torch.bincount(g * E + m.top_i.flatten(), minlength=tab.k * E) for m in ms])
            counts[name] = counts.get(name, 0) + c.view(len(ms), tab.k, E).cpu().double()
    res = {k: torch.cat(v, dim=1 if k in ("top_i", "top_p", "top_raw", "next_p") else 0).numpy()
           for k, v in out.items() if v}
    res["counts"] = {k: v.numpy() for k, v in counts.items()}
    return res


def routing_counts(top_i: np.ndarray, grp: np.ndarray, n_groups: int, n_exp: int) -> np.ndarray:
    """[L, G, E] counts of (group, expert) over all top-k slots."""
    L, N, k = top_i.shape
    g = np.repeat(grp.astype(np.int64), k)
    return np.stack([np.bincount(g * n_exp + top_i[l].reshape(-1).astype(np.int64),
                                 minlength=n_groups * n_exp).reshape(n_groups, n_exp) for l in range(L)])


def lift(counts: np.ndarray, grp: np.ndarray, n_groups: int) -> np.ndarray:
    """[L, G, E] share of expert e's slots from group g, divided by g's token share."""
    share_tok = np.bincount(grp, minlength=n_groups) / len(grp)
    share = counts / np.maximum(counts.sum(1, keepdims=True), 1)
    return share / np.maximum(share_tok[None, :, None], 1e-12)


def evaluate_base(p: Profile, exp: Experiment, seed: int, ckpt: str = "final") -> dict:
    rd = run_dir(p, exp.name, seed)
    tag = "eval" if ckpt == "final" else f"eval_{ckpt}"
    if is_done(rd, tag):
        log(f"{exp.name}/s{seed}: {tag} already done")
        return {}
    ec, mc = p.eval, p.model
    model = load_base(p, exp.name, seed, ckpt)
    n_moe = len(model.moe_layers())          # layer indices below are MoE-layer indices (block 0 is dense)
    tables = {t: GroupTable.load(p, t) for t in TABLES}
    res = dict(experiment=exp.name, seed=seed, ckpt=ckpt)

    # ---- val CE + routing NMI
    val = Split(p, "val")
    v = run_windows(model, val, eval_windows(val, ec.val_tokens), ec.batch, count_tables=tables)
    res["val_ce"] = float(v["ce"].mean())
    res["val_tokens"] = int(len(v["ce"]))
    for t, tab in tables.items():
        c = v["counts"][t]
        res[f"nmi_{t}"] = float(np.mean([nmi_from_counts(c[l]) for l in range(n_moe)]))
        res[f"nmi_{t}_per_layer"] = [nmi_from_counts(c[l]) for l in range(n_moe)]
    log(f"{exp.name}/s{seed}: val CE {res['val_ce']:.4f}  NMI domain {res['nmi_domain']:.4f} sem {res['nmi_sem']:.4f}")

    # ---- probe set (calib split)
    calib = Split(p, "calib")
    n_probe = ec.probe_windows_per_group * max(t.k for t in tables.values())
    pidx = eval_windows(calib, n_probe * mc.ctx)
    base = run_windows(model, calib, pidx, ec.batch, records=True)
    groups = {t: tab.hard[torch.from_numpy(base["doc"])].numpy() for t, tab in tables.items()}
    lifts = {t: lift(routing_counts(base["top_i"], g, tables[t].k, mc.n_exp), g, tables[t].k)
             for t, g in groups.items()}

    def banned_ce(bans):
        model.set_ban(bans)
        try:
            return run_windows(model, calib, pidx, ec.batch)["ce"]
        finally:
            model.set_ban(None)

    # ---- group-ablation FSI
    for t, g in groups.items():
        vals = []
        for gi in range(tables[t].k):
            if (g == gi).sum() == 0:
                continue
            bans = {l: np.argsort(-lifts[t][l, gi])[: ec.fsi_experts_per_layer].tolist() for l in range(n_moe)}
            d = banned_ce(bans) - base["ce"]
            vals.append(d[g == gi].mean() - d[g != gi].mean())
        res[f"fsi_{t}"] = summarize(vals)
    log(f"{exp.name}/s{seed}: FSI domain {res['fsi_domain']['mean']:.5f} sem {res['fsi_sem']['mean']:.5f}")

    # ---- per-expert replacement interaction (ordinary + exposure-matched)
    layers = list(ec.interaction_layers) or list(range(n_moe))
    cells = {t: [] for t in groups}
    pos_bin = base["pos"] * 4 // mc.ctx
    ce_bin = quantile_bins(base["ce"], 4)
    for l in layers:
        ti, tp, tr, nx = base["top_i"][l], base["top_p"][l], base["top_raw"][l], base["next_p"][l]
        for e in range(mc.n_exp):
            hit = ti == e
            routed = hit.any(1)
            if routed.sum() < ec.min_routed:
                continue
            d = banned_ce({l: [e]}) - base["ce"]
            rank = hit[routed].argmax(1)
            gate = tp[routed][np.arange(routed.sum()), rank].astype(np.float32)
            margin = tr[routed][np.arange(routed.sum()), rank].astype(np.float32) - nx[routed].astype(np.float32)
            bins = (((rank * 3 + quantile_bins(gate, 3)) * 3 + quantile_bins(margin, 3)) * 4
                    + ce_bin[routed]) * 4 + pos_bin[routed]
            for t, g in groups.items():
                s = int(np.argmax(lifts[t][l, :, e]))
                own = g == s
                if own.sum() == 0 or (~own).sum() == 0:
                    continue
                ordinary = float(d[own].mean() - d[~own].mean())
                matched, cov = matched_difference(d[routed], own[routed], bins)
                cells[t].append(dict(layer=l, expert=e, group=s, n_routed=int(routed.sum()),
                                     own_share=float(own[routed].mean()), ordinary=ordinary,
                                     matched=matched, coverage=cov))
    for t, cs in cells.items():
        res[f"interaction_{t}"] = summarize([c["ordinary"] for c in cs])
        res[f"matched_{t}"] = summarize([c["matched"] for c in cs])
        res[f"cells_{t}"] = cs
    log(f"{exp.name}/s{seed}: interaction domain {res['interaction_domain']['mean']:.2e} "
        f"matched {res['matched_domain']['mean']:.2e}")
    write_json(rd / f"{tag}.json", res)
    mark_done(rd, tag)
    return res
