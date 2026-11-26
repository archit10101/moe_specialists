#!/usr/bin/env python
"""csr-forecast-latent-edl-v1: frozen forecast aggregation + balanced latent domains (prereg sections 4-5). CPU.

  fit   --cfg ID          : representation (A1 meanstat | A2 deepsets-cv | PAST), tau calibration (target entropy 0.4),
                            balanced Sinkhorn soft k-means on 100k training regions of cache search_main, calibration
                            diagnostics (sizes, entropy, stability, predictiveness), assignment table for search_main.
  apply --cfg ID --cache C: assignment table of another cache with the frozen state (no refitting).
Config id: <REP>_K<k>_L<l>[_H16][_D2][_T25], e.g. A1_K16_L256. Outputs: artifacts/forecast-latent-edl-v1/latent/<ID>/
(0400). Real region tokens are read only for the predictiveness diagnostic, never as an input."""
import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402

N_FIT = 100_000
N_STAB = 50_000
N_HELD = 20_000


def parse(cid):
    parts = cid.split("_")
    c = {"rep": parts[0], "K": int(parts[1][1:]), "L": int(parts[2][1:]), "H": 32, "D": 4, "target": 0.4}
    for p in parts[3:]:
        if p == "H16":
            c["H"] = 16
        elif p == "D2":
            c["D"] = 2
        elif p == "T25":
            c["target"] = 0.25
        else:
            raise ValueError(cid)
    return c


def load_regions(cache, c):
    import numpy as np
    from csr.fle_lib import RegionCache
    rc = RegionCache(cache)
    fields = {"emb_past"} if c["rep"] == "PAST" else {"emb16" if c["H"] == 16 else "emb", "ptok", "ent", "top1", "tok"}
    z = rc.load(fields)
    sel = (z["s"].astype(np.int64) % c["L"]) == 0
    z = {k: v[sel] for k, v in z.items()}
    if c["rep"] != "PAST":
        di = list(range(c["D"]))
        z["E"] = z["emb16" if c["H"] == 16 else "emb"][:, di].astype(np.float32)
        for k in ("ptok", "ent", "top1", "tok"):
            z[k] = z[k][:, di]
    return z, rc.man


def features(z, c, state):
    """Representation X (R, dim) with frozen state (mu/sd for A1, weights for A2)."""
    import numpy as np
    import torch
    from csr.fle_lib import DeepSetsCV, a1_meanstat, draft_stats, set_stats
    if c["rep"] == "PAST":
        X = z["emb_past"].astype(np.float32)
        return X / np.maximum(1e-9, np.linalg.norm(X, axis=1, keepdims=True))
    ds = draft_stats(z["ptok"], z["ent"], z["top1"], c["H"])
    ss = set_stats(z["E"], ds, z["tok"][..., 0])
    if c["rep"] == "A1":
        return a1_meanstat(z["E"], ss, np.asarray(state["mu"]), np.asarray(state["sd"]))
    net = DeepSetsCV()
    net.load_state_dict({k: torch.as_tensor(np.asarray(v, np.float32)) for k, v in state["a2"].items()})
    net.eval()
    dsn = (ds - np.asarray(state["ds_mu"])) / np.asarray(state["ds_sd"])
    ssn = (ss - np.asarray(state["mu"])) / np.asarray(state["sd"])
    out = []
    with torch.no_grad():
        for i in range(0, len(ds), 8192):
            out.append(net(torch.as_tensor(z["E"][i:i + 8192]), torch.as_tensor(dsn[i:i + 8192], dtype=torch.float32),
                           torch.as_tensor(ssn[i:i + 8192], dtype=torch.float32)).numpy())
    return np.concatenate(out)


def train_a2(z, idx, ds_mu, ds_sd, seed=2026):
    """Forecast-only cross-view InfoNCE on training regions (prereg 4). Returns state dict (lists)."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from csr.fle_lib import DeepSetsCV, draft_stats, view_stats
    torch.manual_seed(seed)
    g = np.random.default_rng(seed)
    E = torch.as_tensor(z["E"][idx])
    ds = (draft_stats(z["ptok"][idx], z["ent"][idx], z["top1"][idx], 32) - ds_mu) / ds_sd
    ds = torch.as_tensor(ds, dtype=torch.float32)
    dsraw = torch.as_tensor(draft_stats(z["ptok"][idx], z["ent"][idx], z["top1"][idx], 32), dtype=torch.float32)
    net = DeepSetsCV()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.01)
    N, D = E.shape[0], E.shape[1]
    hist = []
    for ep in range(3):
        perm = torch.as_tensor(g.permutation(N))
        for s in range(0, N - 1023, 1024):
            b = perm[s:s + 1024]
            pr = torch.as_tensor(np.stack([g.permutation(D) for _ in range(len(b))]))
            mA = torch.zeros(len(b), D, dtype=torch.bool).scatter_(1, pr[:, :D // 2], True)
            mB = ~mA
            sA = view_stats(E[b], dsraw[b], mA)
            sB = view_stats(E[b], dsraw[b], mB)
            zA = net(E[b], ds[b], sA, mA)
            zB = net(E[b], ds[b], sB, mB)
            lg = zA @ zB.t() / 0.1
            y = torch.arange(len(b))
            loss = 0.5 * (F.cross_entropy(lg, y) + F.cross_entropy(lg.t(), y))
            opt.zero_grad()
            loss.backward()
            opt.step()
            hist.append(float(loss))
    return {k: v.detach().numpy().tolist() for k, v in net.state_dict().items()}, hist


def unigram_gain(split, gdoc, s, L, q, fit_idx, held_idx, alpha=1000.0):
    """Held-out gain (nats/token) of a q-weighted per-cluster unigram of the region's actual tokens over the global
    unigram (diagnostic only)."""
    import numpy as np
    V = 50304
    K = q.shape[1]
    Ck = np.zeros((K, V))
    cg = np.zeros(V)
    for i in fit_idx:
        x = split.doc(int(gdoc[i]))[int(s[i]):int(s[i]) + L]
        c = np.bincount(x, minlength=V)
        cg += c
        Ck += np.outer(q[i], c)
    pg = (cg + 0.5) / (cg.sum() + 0.5 * V)
    pk = (Ck + alpha * pg) / (Ck.sum(1, keepdims=True) + alpha)
    gain, n = 0.0, 0
    for i in held_idx:
        x = split.doc(int(gdoc[i]))[int(s[i]):int(s[i]) + L]
        pm = q[i] @ pk[:, x]
        gain += float(np.log(pm / pg[x]).sum())
        n += len(x)
    return gain / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["fit", "apply", "merge"])
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--cache", default="search_main")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    if not guard.require_execution(a.execute, "cpu-model"):
        return
    import numpy as np
    import torch
    from csr.fle_lib import assign, calibrate_tau, fit_balanced, hungarian_agreement, norm_entropy, first_token_agreement
    from csr.fle_lib import draft_stats, set_stats
    from csr.fle_paths import LATENT, prereg_sha
    from csr.fsp_lib import Split
    torch.set_num_threads(8)
    c = parse(a.cfg)
    od = LATENT / a.cfg
    t0 = time.time()
    if a.mode == "fit":
        if (od / "meta.json").exists():
            print("already fitted")
            return
        od.mkdir(parents=True, exist_ok=True)
        z, man = load_regions("search_main", c)
        R = len(z["gdoc"])
        g = np.random.default_rng(2026)
        perm = g.permutation(R)
        fit_idx = perm[:N_FIT]
        held = perm[N_FIT:N_FIT + N_HELD]
        state = {}
        if c["rep"] != "PAST":
            ds = draft_stats(z["ptok"], z["ent"], z["top1"], c["H"])
            ss = set_stats(z["E"], ds, z["tok"][..., 0])
            state["mu"], state["sd"] = ss[fit_idx].mean(0).tolist(), ss[fit_idx].std(0).tolist()
            if c["rep"] == "A2":
                state["ds_mu"] = ds[fit_idx].reshape(-1, 3).mean(0).tolist()
                state["ds_sd"] = ds[fit_idx].reshape(-1, 3).std(0).tolist()
                a2_idx = perm[:min(200_000, R)]
                state["a2"], hist = train_a2(z, a2_idx, np.asarray(state["ds_mu"]), np.asarray(state["ds_sd"]))
                state["a2_loss_first_last"] = [hist[0], float(np.mean(hist[-20:]))]
        X = features(z, c, state)
        tau, C, b, tau_rows = calibrate_tau(X[fit_idx], c["K"], target=c["target"])
        q = assign(X, C, b, tau).astype(np.float32)
        # stability: fits on two disjoint 50k samples (same tau), scored on the held-out regions
        rest = perm[N_FIT + N_HELD:]
        sA = rest[:N_STAB] if len(rest) >= 2 * N_STAB else perm[:N_STAB]
        sB = rest[N_STAB:2 * N_STAB] if len(rest) >= 2 * N_STAB else perm[N_STAB:2 * N_STAB]
        CA, bA = fit_balanced(X[sA], c["K"], tau, seed=11)
        CB, bB = fit_balanced(X[sB], c["K"], tau, seed=12)
        stab = hungarian_agreement(assign(X[held], CA, bA, tau).argmax(1), assign(X[held], CB, bB, tau).argmax(1),
                                   c["K"])
        sp = Split("main")
        gain = unigram_gain(sp, z["gdoc"], z["s"], c["L"], q, fit_idx[:30000], held, alpha=1000.0)
        hard = q.argmax(1)
        dom = sp.domains[z["gdoc"].astype(np.int64)]
        # diagnostic only (never a selection input): cluster x domain table and NMI
        cd = np.zeros((c["K"], 5), np.int64)
        np.add.at(cd, (hard, dom), 1)
        pj = cd / cd.sum()
        pk_, pd_ = pj.sum(1, keepdims=True), pj.sum(0, keepdims=True)
        nz = pj > 0
        mi = float((pj[nz] * np.log(pj[nz] / (pk_ @ pd_)[nz])).sum())
        hk = float(-(pk_[pk_ > 0] * np.log(pk_[pk_ > 0])).sum())
        hd = float(-(pd_[pd_ > 0] * np.log(pd_[pd_ > 0])).sum())
        diag = {"n_regions": int(R), "tau": tau, "tau_grid": tau_rows, "assign_entropy": norm_entropy(q),
                "cluster_mass": q.sum(0).tolist(), "hard_sizes": np.bincount(hard, minlength=c["K"]).tolist(),
                "max_prob_mean": float(q.max(1).mean()), "stability_hungarian_agreement": stab,
                "predictive_gain_nats_per_token": gain, "domain_nmi_diagnostic": mi / max(1e-9, math.sqrt(hk * hd)),
                "cluster_by_domain_diagnostic": cd.tolist()}
        if c["rep"] != "PAST":
            diag["forecast"] = {"disagreement_mean": float(ss[:, 0].mean()), "entropy_mean": float(ss[:, 1].mean()),
                                "confidence_mean": float(ss[:, 2].mean()), "top1_mean": float(ss[:, 3].mean()),
                                "first_token_agreement": float(first_token_agreement(z["tok"][held, :, 0]).mean())}
        np.save(od / "centroids.npy", C)
        meta = {"cfg_id": a.cfg, "cfg": c, "state": state, "b": b.tolist(), "tau": tau, "diag": diag,
                "source_cache_manifest_sha256": guard.sha256_file(guard.ROOT / "artifacts/forecast-latent-edl-v1/cache/"
                                                                  "search_main/manifest.json"),
                "centroids_sha256": guard.sha256_file(od / "centroids.npy"), "prereg_sha256": prereg_sha(),
                "time": guard.now(), "seconds": time.time() - t0}
        write_table(od / "q_search_main.npz", z, q, c)
        meta["tables"] = {"search_main": guard.sha256_file(od / "q_search_main.npz")}
        guard.atomic_write_json(od / "meta.json", meta, mode=0o400)
        for f in ("centroids.npy", "q_search_main.npz"):
            (od / f).chmod(0o400)
        print(json.dumps({"cfg": a.cfg, **{k: v for k, v in diag.items() if not isinstance(v, (list, dict))}}))
    elif a.mode == "merge":
        out = od / "q_full.npz"
        if out.exists():
            print("merged table exists")
            return
        za, zb = np.load(od / "q_search_main.npz"), np.load(od / "q_final_main.npz")
        assert int(za["L"]) == int(zb["L"]) and int(za["K"]) == int(zb["K"])
        ka = set(zip(za["gdoc"].tolist(), za["j"].tolist()))
        assert not ka & set(zip(zb["gdoc"].tolist(), zb["j"].tolist())), "overlapping regions"
        tmp = out.with_name(".q_full.tmp.npz")
        np.savez(tmp, gdoc=np.concatenate([za["gdoc"], zb["gdoc"]]), j=np.concatenate([za["j"], zb["j"]]),
                 q=np.concatenate([za["q"], zb["q"]]), K=za["K"], L=za["L"])
        import os
        os.replace(tmp, out)
        out.chmod(0o400)
        rec = {"cfg": a.cfg, "merged": ["q_search_main.npz", "q_final_main.npz"], "sha256": guard.sha256_file(out),
               "n": int(len(za["q"]) + len(zb["q"])), "time": guard.now()}
        with open(od / "applied.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec))
    else:
        meta = json.loads((od / "meta.json").read_text())
        out = od / f"q_{a.cache}.npz"
        if out.exists():
            print("table exists")
            return
        C = np.load(od / "centroids.npy")
        z, man = load_regions(a.cache, c)
        X = features(z, c, meta["state"])
        q = assign(X, C, np.asarray(meta["b"]), meta["tau"]).astype(np.float32)
        write_table(out, z, q, c)
        out.chmod(0o400)
        rec = {"cfg": a.cfg, "cache": a.cache, "sha256": guard.sha256_file(out), "n": int(len(q)),
               "assign_entropy": norm_entropy(q), "time": guard.now()}
        with open(od / "applied.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec))


def write_table(path, z, q, c):
    import numpy as np
    import os
    tmp = path.with_name(f".{path.name}.tmp.npz")
    with open(tmp, "wb") as f:
        np.savez(f, gdoc=z["gdoc"].astype(np.int32), j=(z["s"].astype(np.int64) // c["L"]).astype(np.int16), q=q,
                 K=np.int64(c["K"]), L=np.int64(c["L"]))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
