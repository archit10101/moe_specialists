#!/usr/bin/env python
"""scrgp-v1 Stage 1: frozen virtual-update test on ONE frozen Semantic C6 checkpoint (prereg section 5).

  --phase est  (CALIB-SEARCH half A = A1 + A2; paper-disjoint sub-halves)
      pass 0  natural forward over A: CE quintile cuts, gate / margin tercile cuts per layer, target-token frequency
              terciles, cluster-expert routing counts -> association k*(tau, l, e) = argmax_k P(e|k)/P(e) (both tables)
      pass 1  per sub-half: matching stratum totals P(s), C(s) and eligible cluster masses M_k for every (tau, l, e)
      pass 2  per sub-half: exact forced-insertion gradients for every (l, e): d_tau = grad (w+ - w-) . CE,
              g_tau = grad wg . CE (tau in {SEM, SHUF}); matching balance moments
      output  est_s<seed>.pt: u_A = proj((d_A1 + d_A2)/2, (g_A1 + g_A2)/2) (fp32), split-half statistics, cuts, k*
  --phase eval (CALIB-SEARCH half B, held out)
      eta = TARGET / median_{l,e,tau} |u_A|^2 (one global scalar per checkpoint; no per-signal normalisation)
      per (l, e) and tau: theta_e' = theta_e - eta u_A (temporary tensors; the model is never modified here):
         forced insertion of e with theta_e and theta_e' -> per-token dF; natural routing with theta_e' -> dN
      per-paper sums: matched positive / control (forced strata, B-sample ratios), balanced global per cluster,
         exposure-matched own / other (conditional strata = slc I_cond definition) and all-token natural change
      simultaneous update: all (l, e) updated in place (copies kept), natural CE on B, restored, state hash verified
      output  eval_s<seed>.npz + eval_s<seed>.json
Resumable: a rolling slot (.tmp/scrgp-v1/slots) is written atomically at most every 9 min and at leg end; a leg exits
rc 75 after --max-seconds; the job skips completed passes. No domain label is read."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402

TARGET = 0.002          # median in-sample predicted subject-contrast change (nats) defining eta (prereg 5.4)
CAP_REL = 0.05          # cap: median relative step eta |u| / |theta_e| <= 5 %
MON = None
MB_ROWS = 64
SLOT_EVERY_S = 540.0
SUBSAMPLE = 7           # every 7th valid token for gate / margin quantiles
NUIS = ("ce", "pos", "rank", "log_g4", "margin", "log_freq", "log_doclen")


class Ctx:
    pass


def setup(a):
    import numpy as np
    import torch
    from fle_evaluate import load_milestone
    from csr.fsp_lib import Split
    from csr.gc_lib import PaperQ
    from csr.scrgp_lib import Probe
    from csr.scrgp_paths import C6_RUNS, TABLE_DIR, TMP
    c = Ctx()
    c.np, c.torch = np, torch
    c.split = Split("calib_search")
    tj = json.loads((TABLE_DIR / "tables.json").read_text())
    c.q = {}
    for tau, name in (("SEM", "table_SEM_T05_CS"), ("SHUF", "table_SHUF_SEM_T05_CS")):
        pq = PaperQ(guard.ROOT / tj[name]["path"], c.split, tj[name]["sha256"])
        if pq.coverage < 0.999:
            raise guard.GuardError(f"{name} coverage {pq.coverage}")
        c.q[tau] = pq.q_of_doc
    c.parts = json.loads((TABLE_DIR / "partitions.json").read_text())["parts"]
    if guard.sha256_file(TABLE_DIR / "partitions.json") != tj["partitions"]["sha256"]:
        raise guard.GuardError("partitions hash mismatch")
    c.dev = torch.device("cpu") if a.device == "cpu" else guard.setup_cuda_device()
    rd = C6_RUNS[a.seed]
    st = json.loads((guard.ROOT / rd / "status.json").read_text())
    ck = st["ckpts"]["p100"]
    if a.rows_limit == 0 and guard.sha256_file(guard.ROOT / ck["path"] / "state.pt") != ck["sha256"]:
        raise guard.GuardError("checkpoint hash mismatch")
    c.ckpt = ck
    c.model, _ = load_milestone(ck["path"], c.dev)
    if a.device == "cpu" and a.float64:
        c.model = c.model.double()
    c.pr = Probe(c.model, c.dev)
    c.E, c.L, c.K = c.model.cfg.n_experts, c.model.n_moe, 16
    c.d, c.f = c.model.cfg.d_model, c.model.cfg.expert_ffn
    c.P = c.d * c.f * 2
    c.slotdir = (Path(a.out_dir) / "slots") if a.out_dir else (TMP / "slots")
    c.slotdir.mkdir(parents=True, exist_ok=True)
    c.outdir = Path(a.out_dir) if a.out_dir else (guard.ROOT / "artifacts/scrgp-v1/stage1")
    c.outdir.mkdir(parents=True, exist_ok=True)
    pap = sorted(set(c.split.papers))
    c.paper_idx = {p: i for i, p in enumerate(pap)}
    c.doc_paper = np.asarray([c.paper_idx[p] for p in c.split.papers], np.int64)
    c.mb = a.mb_rows
    return c


def layers_of(a, c):
    return (0, c.L - 1) if a.test_pairs else tuple(range(c.L))


def experts_of(a, c):
    return (0, 1, 2) if a.test_pairs else tuple(range(c.E))


def pair_mask(a, c):
    m = c.torch.zeros(c.L, c.E, dtype=c.torch.bool, device=c.dev)
    for li in layers_of(a, c):
        for e in experts_of(a, c):
            m[li, e] = True
    return m


def rows_of(c, name, a):
    rows = c.parts[name]["rows"]
    if a.rows_limit:
        rows = rows[:a.rows_limit]
    return [rows[i:i + c.mb] for i in range(0, len(rows), c.mb)]


def batch(c, rows):
    from csr.scrgp_lib import make_batch, to_dev
    return to_dev(make_batch(rows, c.split), c.dev)


def save_slot(c, path, obj):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        c.torch.save(obj, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def load_slot(c, path):
    return c.torch.load(path, map_location=c.dev, weights_only=False) if path.exists() else None


def flat_tokens(c, bt):
    t = c.torch
    tgt = bt["targets"].reshape(-1)
    return tgt, bt["positions"].reshape(-1), bt["gdoc"].reshape(-1)


# --------------------------------------------------------------------------------------------- estimation
def est(a, c):
    np, torch = c.np, c.torch
    from csr.scrgp_lib import (Cuts, cos, pair_weights, project, strata_all, token_q, N_STRATA, N_CE, N_GATE, N_MARG)
    slot = c.slotdir / f"est_s{a.seed}.pt"
    S = load_slot(c, slot) or {"stage": "pass0", "cursor": 0}
    t_leg = time.time()
    last = [time.time()]
    E, L, K = c.E, c.L, c.K
    dev = c.dev

    def maybe_save(force=False):
        if force or time.time() - last[0] > SLOT_EVERY_S:
            save_slot(c, slot, S)
            last[0] = time.time()
        if MON is not None and MON.contaminated:
            save_slot(c, slot, S)
            print(json.dumps({"external": MON.detail, "stage": S["stage"], "cursor": S["cursor"]}), flush=True)
            sys.exit(77)
        if time.time() - t_leg > a.max_seconds:
            save_slot(c, slot, S)
            print(json.dumps({"leg_end": S["stage"], "cursor": S["cursor"]}), flush=True)
            sys.exit(75)

    mbA = rows_of(c, "A1", a) + rows_of(c, "A2", a)
    # ---------------- pass 0
    if S["stage"] == "pass0":
        if "acc0" not in S:
            S["acc0"] = {"ce": [], "g4": [[] for _ in range(L)], "mg": [[] for _ in range(L)],
                         "cnt": torch.zeros(c.model.cfg.vocab_size, dtype=torch.float64, device=dev),
                         "C": {tau: torch.zeros(L, K, E, dtype=torch.float64, device=dev) for tau in c.q}}
        A0 = S["acc0"]
        for i in range(S["cursor"], len(mbA)):
            bt = batch(c, mbA[i])
            ce, probs, _, _ = c.pr.natural(bt)
            tgt, pos, gd = flat_tokens(c, bt)
            v = tgt >= 0
            A0["ce"].append(ce[v].float().cpu())
            A0["cnt"].index_add_(0, tgt[v], torch.ones(int(v.sum()), dtype=torch.float64, device=dev))
            qt = {tau: token_q(c.q[tau], gd).to(dev) for tau in c.q}
            for li, p in enumerate(probs):
                pv = p[v].float()
                vals, idx = pv.topk(4, -1)
                sub = slice(None, None, SUBSAMPLE)
                A0["g4"][li].append(vals[sub, 3].cpu())
                r = vals[sub, 3:4]
                elig = pv[sub] < r  # rank >= 5 (strictly below the 4th gate)
                A0["mg"][li].append((torch.log(r.clamp(min=1e-30)) - torch.log(pv[sub].clamp(min=1e-30)))[elig].cpu())
                oh = torch.zeros_like(pv, dtype=torch.float64).scatter_(1, idx, 1.0)
                for tau in c.q:
                    A0["C"][tau][li] += qt[tau][v].double().T @ oh
            S["cursor"] = i + 1
            maybe_save()
        ceall = torch.cat([x.cpu() for x in A0["ce"]]).double().numpy()
        cnt = A0["cnt"].cpu().numpy()
        order = np.argsort(-cnt, kind="stable")
        cum = np.cumsum(cnt[order]) / max(1.0, cnt.sum())
        fcls = np.full(cnt.shape[0], 2, np.int64)
        fcls[order[cum <= 1 / 3]] = 0
        fcls[order[(cum > 1 / 3) & (cum <= 2 / 3)]] = 1
        fcls[order[0]] = 0
        qs = lambda x, n: np.quantile(np.asarray(x, np.float64), np.arange(1, n) / n)  # noqa: E731
        cuts = Cuts(qs(ceall, N_CE), np.stack([qs(torch.cat([x.cpu() for x in A0["g4"][li]]).numpy(), N_GATE) for li in range(L)]),
                    np.stack([qs(torch.cat([x.cpu() for x in A0["mg"][li]]).numpy(), N_MARG) for li in range(L)]), fcls)
        kstar, lift = {}, {}
        for tau in c.q:
            C = A0["C"][tau].cpu().numpy()  # (L, K, E)
            pe_k = C / np.maximum(1e-12, C.sum(2, keepdims=True))
            pe = C.sum(1) / np.maximum(1e-12, C.sum((1, 2))[:, None])  # (L, E)
            lf = pe_k / np.maximum(1e-12, pe[:, None, :])
            kstar[tau] = lf.argmax(1)  # (L, E)
            lift[tau] = lf
        S.update({"stage": "pass1", "cursor": 0, "cuts": cuts.state(), "kstar": kstar, "lift": lift,
                  "routing_counts": {tau: A0["C"][tau].cpu().numpy() for tau in c.q}, "freq_counts": cnt,
                  "n_tokens_A": int(len(ceall))})
        del S["acc0"]
        maybe_save(force=True)
    cuts = Cuts.load(S["cuts"])
    ks = {tau: torch.as_tensor(S["kstar"][tau], device=dev) for tau in c.q}
    freq_counts = torch.as_tensor(S["freq_counts"], device=dev)
    halves = ("A1", "A2")
    mbs = {h: rows_of(c, h, a) for h in halves}

    def layer_feats(p, ce, pos, tgt, li):
        st_ = strata_all(p.float(), ce.float(), pos, tgt, li, cuts, dev)  # (N, E)
        return st_

    # ---------------- pass 1
    if S["stage"] == "pass1":
        if "acc1" not in S:
            S["acc1"] = {h: {tau: {"P": torch.zeros(L, E, N_STRATA, dtype=torch.float64, device=dev),
                                   "C": torch.zeros(L, E, N_STRATA, dtype=torch.float64, device=dev),
                                   "M": torch.zeros(L, E, K, dtype=torch.float64, device=dev)} for tau in c.q}
                         for h in halves}
            S["h"] = 0
        A1 = S["acc1"]
        while S["h"] < 2:
            h = halves[S["h"]]
            for i in range(S["cursor"], len(mbs[h])):
                bt = batch(c, mbs[h][i])
                ce, probs, _, _ = c.pr.natural(bt)
                tgt, pos, gd = flat_tokens(c, bt)
                qt = {tau: token_q(c.q[tau], gd).to(dev).double() for tau in c.q}
                for li, p in enumerate(probs):
                    st_ = layer_feats(p, ce, pos, tgt, li)
                    ok = st_ >= 0
                    flat = (torch.arange(E, device=dev) * N_STRATA).unsqueeze(0) + st_.clamp(min=0)
                    for tau in c.q:
                        qk = qt[tau][:, ks[tau][li]]  # (N, E)
                        A1[h][tau]["P"][li].view(-1).index_add_(0, flat[ok], qk[ok])
                        A1[h][tau]["C"][li].view(-1).index_add_(0, flat[ok], (1 - qk)[ok])
                        A1[h][tau]["M"][li] += ok.double().T @ qt[tau]
                S["cursor"] = i + 1
                maybe_save()
            S["h"] += 1
            S["cursor"] = 0
        S["match"] = {}
        for h in halves:
            S["match"][h] = {}
            for tau in c.q:
                Pp, Cc = A1[h][tau]["P"], A1[h][tau]["C"]
                both = (Pp > 0) & (Cc > 0)
                ratio = torch.where(both, Pp / Cc.clamp(min=1e-300), torch.zeros_like(Pp))
                S["match"][h][tau] = {"ratio": ratio, "ptot": (Pp * both).sum(-1), "pall": Pp.sum(-1),
                                      "M": A1[h][tau]["M"]}
        S.update({"stage": "pass2", "cursor": 0, "h": 0})
        del S["acc1"]
        maybe_save(force=True)
    # ---------------- pass 2
    if S["stage"] == "pass2":
        if "acc2" not in S:
            S["acc2"] = {h: torch.zeros(4, L, E, c.P, dtype=torch.float32, device=dev) for h in halves}
            S["bal"] = {h: {tau: torch.zeros(L, E, len(NUIS), 4, dtype=torch.float64, device=dev) for tau in c.q}
                        for h in halves}
        while S["h"] < 2:
            h = halves[S["h"]]
            acc, bal = S["acc2"][h], S["bal"][h]
            for i in range(S["cursor"], len(mbs[h])):
                bt = batch(c, mbs[h][i])
                ce, probs, cache, ctx = c.pr.natural(bt, keep_cache=True)
                tgt, pos, gd = flat_tokens(c, bt)
                shape = tuple(bt["input_ids"].shape) + (c.d,)
                qt = {tau: token_q(c.q[tau], gd).to(dev) for tau in c.q}
                doclen = torch.as_tensor(c.split.lengths, device=dev)[gd.clamp(min=0)].double()
                for li in layers_of(a, c):
                    p = probs[li]
                    st_ = layer_feats(p, ce, pos, tgt, li)
                    from csr.scrgp_lib import ranks_of
                    r = ranks_of(p).double()
                    g4 = p.topk(4, -1).values[:, 3].double()
                    nu = [ce.double(), pos.double(), None, torch.log(g4.clamp(min=1e-30)), None,
                          torch.log(freq_counts[tgt.clamp(min=0)] + 1.0), torch.log(doclen)]
                    for e in experts_of(a, c):
                        W, ws = [], {}
                        for tau in ("SEM", "SHUF"):
                            m_ = S["match"][h][tau]
                            wp, wm, wg = pair_weights(qt[tau], st_[:, e], int(ks[tau][li, e]), m_["ratio"][li, e],
                                                      m_["ptot"][li, e].clamp(min=1e-300), m_["M"][li, e])
                            W += [wp - wm, wg]
                            ws[tau] = (wp, wm)
                        G = c.pr.expert_grads(li, e, cache[li], ctx, shape, tgt, torch.stack(W).float())
                        acc[:, li, e] += G
                        nu[2] = r[:, e]
                        nu[4] = torch.where(r[:, e] == 3, torch.zeros_like(g4),
                                            torch.log(g4.clamp(min=1e-30)) - torch.log(p[:, e].double().clamp(min=1e-30)))
                        for tau in ("SEM", "SHUF"):
                            wp, wm = ws[tau]
                            for vi, x in enumerate(nu):
                                bal[tau][li, e, vi, 0] += (wp * x).sum()
                                bal[tau][li, e, vi, 1] += (wp * x * x).sum()
                                bal[tau][li, e, vi, 2] += (wm * x).sum()
                                bal[tau][li, e, vi, 3] += (wm * x * x).sum()
                del cache
                S["cursor"] = i + 1
                maybe_save()
            S["h"] += 1
            S["cursor"] = 0
        S["stage"] = "final"
        maybe_save(force=True)
    # ---------------- finalize
    acc = S["acc2"]
    res = {"seed": a.seed, "ckpt": c.ckpt, "n_tokens_A": S["n_tokens_A"], "target_eta_def": TARGET, "pairs": {}}
    U = {}
    for ti, tau in enumerate(("SEM", "SHUF")):
        d1, g1 = acc["A1"][2 * ti].double(), acc["A1"][2 * ti + 1].double()
        d2, g2 = acc["A2"][2 * ti].double(), acc["A2"][2 * ti + 1].double()
        u1, u2 = project(d1, g1), project(d2, g2)
        dA, gA = 0.5 * (d1 + d2), 0.5 * (g1 + g2)
        uA = project(dA, gA)
        U[tau] = uA.float()
        nrm = lambda x: x.norm(dim=-1)  # noqa: E731
        st = {"cos_d12": cos(d1, d2), "cos_u12": cos(u1, u2), "cos_g12": cos(g1, g2), "norm_d": nrm(dA),
              "norm_g": nrm(gA), "norm_u": nrm(uA), "ratio_d_g": nrm(dA) / nrm(gA).clamp(min=1e-300),
              "ratio_u_d": nrm(uA) / nrm(dA).clamp(min=1e-300), "cos_dg": cos(dA, gA),
              "orth_ug": (uA * gA).sum(-1) / (nrm(uA) * nrm(gA)).clamp(min=1e-300),
              "norm_d1": nrm(d1), "norm_d2": nrm(d2), "dot_d12": (d1 * d2).sum(-1),
              "norm_u_w1": uA[..., :c.d * c.f].norm(dim=-1), "norm_u_w2": uA[..., c.d * c.f:].norm(dim=-1),
              "theta_norm": torch.stack([torch.stack([torch.cat([c.model.blocks[li + 1].ffn.experts.w1[e].reshape(-1),
                                                                 c.model.blocks[li + 1].ffn.experts.w2[e].reshape(-1)]).double().norm()
                                                      for e in range(E)]) for li in range(L)])}
        bal = {}
        for h in halves:
            b = S["bal"][h][tau]
            mp, vp = b[..., 0], b[..., 1] - b[..., 0] ** 2
            mc, vc = b[..., 2], b[..., 3] - b[..., 2] ** 2
            # weights sum to the matched fraction (positive) / matched fraction (control) of the normalisation
            m_ = S["match"][h][tau]
            mf = (m_["ptot"] / m_["pall"].clamp(min=1e-300))
            bal[h] = {"smd": ((mp - mc) / torch.sqrt(0.5 * (vp.clamp(min=0) + vc.clamp(min=0))).clamp(min=1e-12)),
                      "matched_frac": mf}
        st["smd_A1"] = bal["A1"]["smd"]
        st["smd_A2"] = bal["A2"]["smd"]
        st["matched_frac_A1"] = bal["A1"]["matched_frac"]
        st["matched_frac_A2"] = bal["A2"]["matched_frac"]
        res["pairs"][tau] = {k: v.cpu().numpy() for k, v in st.items()}
    res["kstar"] = {tau: np.asarray(S["kstar"][tau]) for tau in c.q}
    res["lift"] = {tau: np.asarray(S["lift"][tau]) for tau in c.q}
    res["routing_counts"] = S["routing_counts"]
    res["cuts"] = S["cuts"]
    out = c.outdir / f"est_s{a.seed}.pt"
    torch.save({"u": {k: v.cpu() for k, v in U.items()}, "stats": res}, out)
    os.chmod(out, 0o400)
    summ = {"seed": a.seed, "sha256": guard.sha256_file(out), "time": guard.now(),
            "max_abs_orth_ug": {tau: float(np.abs(res["pairs"][tau]["orth_ug"]).max()) for tau in c.q}}
    guard.atomic_write_json(c.outdir / f"est_s{a.seed}.json", summ, mode=0o400)
    print(json.dumps(summ), flush=True)
    slot.unlink(missing_ok=True)


# --------------------------------------------------------------------------------------------- evaluation (B)
def evaluate_b(a, c):
    np, torch = c.np, c.torch
    from csr.scrgp_lib import (Cuts, cond_strata_all, pair_weights, split_theta, state_hash, strata_all, token_q,
                               N_COND, N_DEC, N_STRATA)
    estp = c.outdir / f"est_s{a.seed}.pt"
    E_ = torch.load(estp, map_location="cpu", weights_only=False)
    U = {k: v.to(c.dev) for k, v in E_["u"].items()}
    stats = E_["stats"]
    cuts = Cuts.load(stats["cuts"])
    ks = {tau: torch.as_tensor(stats["kstar"][tau], device=c.dev) for tau in c.q}
    E, L, K, dev = c.E, c.L, c.K, c.dev
    un2 = torch.cat([(U[t].double() ** 2).sum(-1)[pair_mask(a, c)].reshape(-1) for t in U])
    eta_t = float(TARGET / un2.median())
    th = torch.as_tensor(stats["pairs"]["SEM"]["theta_norm"], dtype=torch.float64)[pair_mask(a, c).cpu()]
    rel = torch.cat([eta_t * (U[t].double() ** 2).sum(-1).sqrt()[pair_mask(a, c)].cpu() / th for t in U])
    eta = eta_t * min(1.0, CAP_REL / float(rel.median()))
    eta_info = {"eta_target": eta_t, "median_rel_step_at_target": float(rel.median()), "eta": eta,
                "cap_binding": bool(float(rel.median()) > CAP_REL)}
    slot = c.slotdir / f"eval_s{a.seed}.pt"
    S = load_slot(c, slot) or {"stage": "b0", "cursor": 0}
    t_leg = time.time()
    last = [time.time()]

    def maybe_save(force=False):
        if force or time.time() - last[0] > SLOT_EVERY_S:
            save_slot(c, slot, S)
            last[0] = time.time()
        if MON is not None and MON.contaminated:
            save_slot(c, slot, S)
            print(json.dumps({"external": MON.detail, "stage": S["stage"], "cursor": S["cursor"]}), flush=True)
            sys.exit(77)
        if time.time() - t_leg > a.max_seconds:
            save_slot(c, slot, S)
            print(json.dumps({"leg_end": S["stage"], "cursor": S["cursor"]}), flush=True)
            sys.exit(75)

    if "hash0" not in S:
        S["hash0"] = state_hash(c.model)
    mbB = rows_of(c, "B", a)
    docsB = sorted({int(d) for r in c.parts["B"]["rows"][:(a.rows_limit or None)] for d in r})
    papB = sorted({int(c.doc_paper[d]) for d in docsB})
    pmap = torch.full((int(c.doc_paper.max()) + 1,), -1, dtype=torch.long, device=dev)
    pmap[torch.as_tensor(papB, device=dev)] = torch.arange(len(papB), device=dev)
    doc_paper = torch.as_tensor(c.doc_paper, device=dev)
    NP = len(papB)

    def pid_of(gd):
        return pmap[doc_paper[gd.clamp(min=0)]]

    # ---------------- b0: deciles for the exposure-matched natural strata + base per-paper CE
    if S["stage"] == "b0":
        if "acc" not in S:
            S["acc"] = {"ce": [], "g": [[] for _ in range(L)], "m": [[] for _ in range(L)],
                        "base_sum": torch.zeros(NP, dtype=torch.float64, device=dev),
                        "base_n": torch.zeros(NP, dtype=torch.float64, device=dev)}
        A0 = S["acc"]
        for i in range(S["cursor"], len(mbB)):
            bt = batch(c, mbB[i])
            ce, probs, _, _ = c.pr.natural(bt)
            tgt, pos, gd = flat_tokens(c, bt)
            v = tgt >= 0
            A0["ce"].append(ce[v].float().cpu())
            pid = pid_of(gd)
            A0["base_sum"].index_add_(0, pid[v], ce[v].double())
            A0["base_n"].index_add_(0, pid[v], torch.ones(int(v.sum()), dtype=torch.float64, device=dev))
            for li, p in enumerate(probs):
                pv = p[v].float()[::SUBSAMPLE]
                vals, idx = pv.topk(5, -1)
                A0["g"][li].append(vals[:, :4].reshape(-1).cpu())
                A0["m"][li].append((vals[:, :4] - vals[:, 4:5]).reshape(-1).cpu())
            S["cursor"] = i + 1
            maybe_save()
        qd = lambda x: np.quantile(torch.cat([y.cpu() for y in x]).double().numpy() if isinstance(x, list) else x,  # noqa: E731
                                   np.arange(1, N_DEC) / N_DEC)
        S["cdec"] = {"ce": qd(A0["ce"]), "gate": np.stack([qd(A0["g"][li]) for li in range(L)]),
                     "marg": np.stack([qd(A0["m"][li]) for li in range(L)])}
        S["base"] = {"sum": A0["base_sum"].cpu().numpy(), "n": A0["base_n"].cpu().numpy()}
        del S["acc"]
        S.update({"stage": "b1", "cursor": 0})
        maybe_save(force=True)
    cdec = S["cdec"]
    # ---------------- b1: B-sample matching totals (forced strata with A cuts; conditional strata with B deciles)
    if S["stage"] == "b1":
        if "acc" not in S:
            z = lambda n: {tau: torch.zeros(L, E, n, dtype=torch.float64, device=dev) for tau in c.q}  # noqa: E731
            S["acc"] = {"P": z(N_STRATA), "C": z(N_STRATA), "O": z(N_COND), "X": z(N_COND), "M": z(K)}
        A1 = S["acc"]
        for i in range(S["cursor"], len(mbB)):
            bt = batch(c, mbB[i])
            ce, probs, _, _ = c.pr.natural(bt)
            tgt, pos, gd = flat_tokens(c, bt)
            qt = {tau: token_q(c.q[tau], gd).to(dev).double() for tau in c.q}
            for li, p in enumerate(probs):
                sf = strata_all(p.float(), ce.float(), pos, tgt, li, cuts, dev)
                sc = cond_strata_all(p.float(), ce.float(), pos, tgt, li, cdec, dev)
                for sarr, n, kp, kc in ((sf, N_STRATA, "P", "C"), (sc, N_COND, "O", "X")):
                    ok = sarr >= 0
                    flat = (torch.arange(E, device=dev) * n).unsqueeze(0) + sarr.clamp(min=0)
                    for tau in c.q:
                        qk = qt[tau][:, ks[tau][li]]
                        A1[kp][tau][li].view(-1).index_add_(0, flat[ok], qk[ok])
                        A1[kc][tau][li].view(-1).index_add_(0, flat[ok], (1 - qk)[ok])
                okf = sf >= 0
                for tau in c.q:
                    A1["M"][tau][li] += okf.double().T @ qt[tau]
            S["cursor"] = i + 1
            maybe_save()
        S["mt"] = {}
        for tau in c.q:
            r = {}
            for kp, kc, nm in (("P", "C", "f"), ("O", "X", "c")):
                Pp, Cc = A1[kp][tau], A1[kc][tau]
                both = (Pp > 0) & (Cc > 0)
                r[nm] = torch.where(both, Pp / Cc.clamp(min=1e-300), torch.zeros_like(Pp))
                r[nm + "_frac"] = ((Pp * both).sum(-1) / Pp.sum(-1).clamp(min=1e-300)).cpu().numpy()
            r["M"] = A1["M"][tau]
            S["mt"][tau] = r
        del S["acc"]
        S.update({"stage": "b2", "cursor": 0})
        maybe_save(force=True)
    # ---------------- b2: per-pair virtual updates
    if S["stage"] == "b2":
        if "acc" not in S:
            z = lambda *s: {tau: torch.zeros(*s, dtype=torch.float64, device=dev) for tau in c.q}  # noqa: E731
            S["acc"] = {k: z(L, E, NP) for k in ("fps", "fpw", "fcs", "fcw", "nos", "now", "nxs", "nxw", "nall",
                                                 "fall")}
            S["acc"].update({"gs": z(L, E, NP, K), "gw": z(L, E, NP, K)})
            S["acc"]["ntok"] = torch.zeros(NP, dtype=torch.float64, device=dev)
        A2 = S["acc"]
        for i in range(S["cursor"], len(mbB)):
            bt = batch(c, mbB[i])
            ce0, probs, cache, ctx = c.pr.natural(bt, keep_cache=True)
            tgt, pos, gd = flat_tokens(c, bt)
            v = tgt >= 0
            vd = v.double()
            pid = pid_of(gd).clamp(min=0)
            A2["ntok"].index_add_(0, pid, vd)
            shape = tuple(bt["input_ids"].shape) + (c.d,)
            qt = {tau: token_q(c.q[tau], gd).to(dev).double() for tau in c.q}
            for li in layers_of(a, c):
                p = probs[li].float()
                sf = strata_all(p, ce0.float(), pos, tgt, li, cuts, dev)
                sc = cond_strata_all(p, ce0.float(), pos, tgt, li, cdec, dev)
                ex = c.model.blocks[li + 1].ffn.experts
                for e in experts_of(a, c):
                    w1e, w2e = ex.w1[e], ex.w2[e]
                    with torch.no_grad(), c.pr.autocast():
                        ceF0 = c.pr.ce_of(c.pr.forced_hidden(li, e, cache[li], ctx, shape, w1e, w2e), bt["targets"])
                    for tau in ("SEM", "SHUF"):
                        du1, du2 = split_theta(U[tau][li, e], c.d, c.f)
                        w1n, w2n = (w1e.float() - eta * du1).to(w1e.dtype), (w2e.float() - eta * du2).to(w2e.dtype)
                        with torch.no_grad(), c.pr.autocast():
                            ceF1 = c.pr.ce_of(c.pr.forced_hidden(li, e, cache[li], ctx, shape, w1n, w2n),
                                              bt["targets"])
                            ceN1 = c.pr.ce_of(c.pr.natural_updated_hidden(li, e, cache[li], ctx, shape, w1e, w2e,
                                                                          w1n, w2n), bt["targets"])
                        dF = (ceF1 - ceF0).double() * vd
                        dN = (ceN1 - ce0).double() * vd
                        mt = S["mt"][tau]
                        k_ = int(ks[tau][li, e])
                        s_ = sf[:, e]
                        ok = s_ >= 0
                        rat = mt["f"][li, e][s_.clamp(min=0)]
                        qk = qt[tau][:, k_]
                        wp = torch.where(ok & (rat > 0), qk, torch.zeros_like(qk))
                        wm = torch.where(ok & (rat > 0), (1 - qk) * rat, torch.zeros_like(qk))
                        A2["fps"][tau][li, e].index_add_(0, pid, wp * dF)
                        A2["fpw"][tau][li, e].index_add_(0, pid, wp)
                        A2["fcs"][tau][li, e].index_add_(0, pid, wm * dF)
                        A2["fcw"][tau][li, e].index_add_(0, pid, wm)
                        okd = ok.double()
                        flatk = pid.unsqueeze(1) * K + torch.arange(K, device=dev).unsqueeze(0)
                        A2["gs"][tau][li, e].view(-1).index_add_(0, flatk.reshape(-1),
                                                                 (qt[tau] * (okd * dF).unsqueeze(1)).reshape(-1))
                        A2["gw"][tau][li, e].view(-1).index_add_(0, flatk.reshape(-1),
                                                                 (qt[tau] * okd.unsqueeze(1)).reshape(-1))
                        s2 = sc[:, e]
                        ok2 = s2 >= 0
                        rat2 = mt["c"][li, e][s2.clamp(min=0)]
                        wo = torch.where(ok2 & (rat2 > 0), qk, torch.zeros_like(qk))
                        wx = torch.where(ok2 & (rat2 > 0), (1 - qk) * rat2, torch.zeros_like(qk))
                        A2["nos"][tau][li, e].index_add_(0, pid, wo * dN)
                        A2["now"][tau][li, e].index_add_(0, pid, wo)
                        A2["nxs"][tau][li, e].index_add_(0, pid, wx * dN)
                        A2["nxw"][tau][li, e].index_add_(0, pid, wx)
                        A2["nall"][tau][li, e].index_add_(0, pid, dN)
                        A2["fall"][tau][li, e].index_add_(0, pid, dF)
            del cache
            S["cursor"] = i + 1
            maybe_save()
        S.update({"stage": "b3", "cursor": 0})
        maybe_save(force=True)
    # ---------------- b3: simultaneous update (all pairs) per table, in place with exact restore
    if S["stage"] == "b3":
        if "sim" not in S:
            S["sim"] = {}
        for tau in ("SEM", "SHUF"):
            if tau in S["sim"]:
                continue
            orig = [(blk.ffn.experts.w1.detach().clone(), blk.ffn.experts.w2.detach().clone())
                    for blk in c.model.blocks[1:]]
            with torch.no_grad():
                for li, blk in enumerate(c.model.blocks[1:]):
                    du = U[tau][li]  # (E, P)
                    d1 = du[:, :c.d * c.f].view(E, c.d, c.f)
                    d2 = du[:, c.d * c.f:].view(E, c.f, c.d)
                    blk.ffn.experts.w1.sub_((eta * d1).to(blk.ffn.experts.w1.dtype))
                    blk.ffn.experts.w2.sub_((eta * d2).to(blk.ffn.experts.w2.dtype))
            ssum = torch.zeros(NP, dtype=torch.float64, device=dev)
            for i in range(len(mbB)):
                bt = batch(c, mbB[i])
                ce, _, _, _ = c.pr.natural(bt)
                tgt, pos, gd = flat_tokens(c, bt)
                v = tgt >= 0
                ssum.index_add_(0, pid_of(gd)[v], ce[v].double())
            with torch.no_grad():
                for (w1o, w2o), blk in zip(orig, c.model.blocks[1:]):
                    blk.ffn.experts.w1.copy_(w1o)
                    blk.ffn.experts.w2.copy_(w2o)
            if state_hash(c.model) != S["hash0"]:
                raise guard.GuardError("model state not restored exactly after the simultaneous update")
            S["sim"][tau] = ssum.cpu().numpy()
            maybe_save(force=True)
        S["stage"] = "final"
        maybe_save(force=True)
    if state_hash(c.model) != S["hash0"]:
        raise guard.GuardError("model state hash changed during evaluation")
    A2 = S["acc"]
    arrs = {"papers": np.asarray(papB), "base_sum": S["base"]["sum"], "base_n": S["base"]["n"],
            "ntok": A2["ntok"].cpu().numpy(), "eta": np.float64(eta)}
    for k, d_ in A2.items():
        if k == "ntok":
            continue
        for tau in c.q:
            arrs[f"{k}_{tau}"] = d_[tau].cpu().numpy()
    for tau in c.q:
        arrs[f"sim_{tau}"] = S["sim"][tau]
        arrs[f"ffrac_{tau}"] = S["mt"][tau]["f_frac"]
        arrs[f"cfrac_{tau}"] = S["mt"][tau]["c_frac"]
    out = c.outdir / f"eval_s{a.seed}.npz"
    np.savez_compressed(out, **arrs)
    os.chmod(out, 0o400)
    summ = {"seed": a.seed, "eta": eta, **eta_info, "sha256": guard.sha256_file(out), "hash_restored": True,
            "n_papers_B": NP, "time": guard.now(), "est_sha256": guard.sha256_file(estp)}
    guard.atomic_write_json(c.outdir / f"eval_s{a.seed}.json", summ, mode=0o400)
    print(json.dumps(summ), flush=True)
    slot.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["est", "eval"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--rows-limit", type=int, default=0, help="tests only")
    ap.add_argument("--mb-rows", type=int, default=MB_ROWS)
    ap.add_argument("--float64", action="store_true", help="CPU tests only")
    ap.add_argument("--test-pairs", action="store_true", help="tests only: layers (0, L-1) x experts (0, 1, 2)")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--max-seconds", type=float, default=4800.0)
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    if not guard.require_execution(a.execute, "gpu" if a.device == "cuda" else "cpu-model"):
        return
    from csr.scrgp_paths import check_prereg
    test_mode = a.rows_limit > 0 or a.test_pairs
    if test_mode and not a.out_dir:
        raise guard.GuardError("test mode requires --out-dir")
    if not test_mode:
        check_prereg()
    global MON
    mon = None
    if a.device == "cuda":
        guard.occupancy_gate(f"scrgp stage1 {a.phase} s{a.seed}")
        mon = MON = guard.OccupancyMonitor(interval=15.0)
        mon.start()
    guard.register_self(f"scrgp stage1 {a.phase} s{a.seed}")
    t_iso, t0 = guard.now(), time.time()
    status, rc = "COMPLETE", 0
    try:
        c = setup(a)
        (est if a.phase == "est" else evaluate_b)(a, c)
    except SystemExit as ex:
        rc = int(ex.code or 0)
        status = {75: "LEG_END", 77: "EXTERNALLY_STOPPED", 0: "COMPLETE"}.get(rc, "FAILED")
    except KeyboardInterrupt:
        status, rc = "EXTERNALLY_STOPPED", 77
    finally:
        if mon is not None:
            mon.stop()
            if mon.contaminated and status in ("COMPLETE", "LEG_END"):
                status, rc = "EXTERNALLY_STOPPED", 77
    if a.device == "cuda":
        guard.gpu_ledger(t_iso, guard.now(), f"scrgp_stage1_{a.phase}_s{a.seed}", "scrgp stage1", status)
    print(json.dumps({"status": status, "seconds": time.time() - t0}), flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
