#!/usr/bin/env python
"""scrgp-v1 POST-HOC diagnostics (CPU, read-only; not preregistered; cannot change the Stage-1 verdict).
Reads artifacts/scrgp-v1/stage1/{est,eval}_s*.{pt,npz,json} and the frozen C6 p100 weights; writes
artifacts/scrgp-v1/diagnostics_posthoc.json (0400). Per layer and seed: F, N, G, Nall (natural all-token dCE per pair),
predicted eta|u|^2, realised/predicted, median relative step; per-cluster forced change for the assigned k* vs the
other clusters (does the residual worsen nonassigned subjects?); the k* = 14 cluster profile; and a post-hoc
'layers 0-9 only' summary."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402
from csr.scrgp_lib import contrast_boot  # noqa: E402
from csr.scrgp_paths import ART, C6_RUNS, SEEDS  # noqa: E402


def main():
    guard.check_root()
    out_p = ART / "diagnostics_posthoc.json"
    if out_p.exists():
        print("exists")
        return
    sd_ = ART / "stage1"
    W1 = np.ones((1, 240))
    R = {"note": "POST HOC; descriptive only; not a preregistered criterion", "per_seed": {}}
    acc = {}
    for s in SEEDS:
        est = torch.load(sd_ / f"est_s{s}.pt", map_location="cpu", weights_only=False)
        e = dict(np.load(sd_ / f"eval_s{s}.npz"))
        eta = json.loads((sd_ / f"eval_s{s}.json").read_text())["eta"]
        st = torch.load(guard.ROOT / C6_RUNS[s] / "ckpt_p100/state.pt", map_location="cpu", weights_only=False)
        st = st.get("model", st)
        th = torch.stack([torch.sqrt((st[f"blocks.{li + 1}.ffn.experts.w1"].float() ** 2).sum((1, 2)) +
                                     (st[f"blocks.{li + 1}.ffn.experts.w2"].float() ** 2).sum((1, 2)))
                          for li in range(11)]).double()
        L, E, NP = e["fps_SEM"].shape
        W1 = np.ones((1, NP))
        kstar = est["stats"]["kstar"]
        for tau in ("SEM", "SHUF"):
            r2 = lambda k: e[f"{k}_{tau}"].reshape(L * E, NP)  # noqa: E731
            F = -contrast_boot(r2("fps"), r2("fpw"), r2("fcs"), r2("fcw"), W1)[0]
            N = -contrast_boot(r2("nos"), r2("now"), r2("nxs"), r2("nxw"), W1)[0]
            gs, gw = e[f"gs_{tau}"].reshape(L * E, NP, -1), e[f"gw_{tau}"].reshape(L * E, NP, -1)
            with np.errstate(invalid="ignore", divide="ignore"):
                gk = gs.sum(1) / gw.sum(1)  # (pairs, K) forced dCE per cluster, positive = worse
            G = np.nanmean(gk, 1)
            ks = np.asarray(kstar[tau]).reshape(-1)
            g_own = gk[np.arange(L * E), ks]
            m = np.ones_like(gk, dtype=bool)
            m[np.arange(L * E), ks] = False
            g_oth = np.nanmean(np.where(m, gk, np.nan), 1)
            frac_oth_worse = np.nanmean(np.where(m, gk > 0, np.nan), 1)
            nall = e[f"nall_{tau}"].reshape(L * E, NP).sum(1) / e["ntok"].sum()
            un = est["u"][tau].double().norm(dim=-1)
            pred = (eta * un ** 2).numpy().reshape(-1)
            rel = (eta * un / th).numpy().reshape(-1)
            d = {"F": F, "N": N, "G": G, "g_own": g_own, "g_other": g_oth, "frac_other_clusters_worse": frac_oth_worse,
                 "Nall": nall, "pred": pred, "rel": rel, "kstar": ks.astype(float)}
            for k, v in d.items():
                acc.setdefault(tau, {}).setdefault(k, []).append(v.reshape(L, E))
            sim = float((e[f"sim_{tau}"] - e["base_sum"]).sum() / e["base_n"].sum())
            R["per_seed"].setdefault(tau, {})[s] = {
                "Sim": sim, "sum_Nall": float(nall.sum()),
                "sum_Nall_layers0_9": float(nall.reshape(L, E)[:10].sum()),
                "sum_Nall_layer10": float(nall.reshape(L, E)[10].sum()),
                "F_layers0_9": float(np.nanmean(F.reshape(L, E)[:10])), "G_layers0_9": float(np.nanmean(G.reshape(L, E)[:10])),
                "N_layers0_9": float(np.nanmean(N.reshape(L, E)[:10])),
                "pred_layers0_9": float(pred.reshape(L, E)[:10].mean()),
                "F_layer10": float(np.nanmean(F.reshape(L, E)[10])), "G_layer10": float(np.nanmean(G.reshape(L, E)[10])),
                "N_layer10": float(np.nanmean(N.reshape(L, E)[10])), "pred_layer10": float(pred.reshape(L, E)[10].mean())}
    by_layer = {}
    for tau, dd in acc.items():
        by_layer[tau] = {}
        for k, v in dd.items():
            a = np.stack(v)  # (S, L, E)
            if k == "kstar":
                continue
            by_layer[tau][k] = {"seed_mean": np.nanmean(a, (0, 2)).tolist(),
                                "per_seed": np.nanmean(a, 2).tolist()}
        a = np.stack(dd["F"])
        p = np.stack(dd["pred"])
        by_layer[tau]["realised_over_pred"] = (np.nanmean(a, (0, 2)) / p.mean((0, 2))).tolist()
    R["by_layer"] = by_layer
    # cluster 14 profile (SEM): pairs with k* = 14 per layer, and their share of F
    ks = np.stack(acc["SEM"]["kstar"])
    F = np.stack(acc["SEM"]["F"])
    R["kstar_counts_by_layer_SEM_seed_mean"] = {int(k): (ks == k).sum((0, 2)).astype(float).__truediv__(len(SEEDS)).tolist()
                                                for k in np.unique(ks)}
    R["F_share_layer10_cluster14_SEM"] = float(np.nansum(np.where((ks == 14), F, 0)[:, 10]) / np.nansum(F))
    R["time"] = guard.now()
    guard.atomic_write_json(out_p, R, mode=0o400)
    t = R["per_seed"]
    for tau in ("SEM", "SHUF"):
        print(tau)
        for s in SEEDS:
            print(" ", s, {k: round(v, 5) for k, v in t[tau][s].items()})
        for k in ("F", "G", "g_own", "g_other", "frac_other_clusters_worse", "N", "Nall", "pred", "rel"):
            print(f"  {k:26s}", " ".join(f"{x:+.5f}" for x in by_layer[tau][k]["seed_mean"]))
        print("  realised/pred             ", " ".join(f"{x:+.3f}" for x in by_layer[tau]["realised_over_pred"]))
    print("F share (layer 10, k*=14):", R["F_share_layer10_cluster14_SEM"])
    print("k*=14 pairs per layer:", R["kstar_counts_by_layer_SEM_seed_mean"].get(14))


if __name__ == "__main__":
    main()
