"""semantic-lift-245m-confirm-v1 (slc) functional-specialization measurement (prereg sections 7-9).

  base_pass        one forward over measurement rows in exactly the token order of lfr_eval.evaluate(per_token=True):
                   per-token CE, domain, document, position, top-4 expert ids and the top-5 router probabilities.
  per_expert_slc   the existing equal-compute per-expert replacement procedure (slk_lib.per_expert_boot, unchanged
                   definitions: k* = argmax discovery lift on calibration tokens; one exact replacement pass per
                   (layer, expert) - the excluded expert's logit is masked and the next-ranked expert is admitted;
                   interaction = own-domain dCE - CE-decile x position-bin matched other-domain dCE; document
                   bootstrap with the same fixed-seed weights) PLUS the preregistered exposure-conditioned interaction
                   on the same replacement forwards: tokens where the expert was in the original top-4; strata =
                   CE decile x position bin x router rank x gate-probability decile x router-margin decile (the
                   last two computed within the cell's exposed tokens; margin = p_e - p_(5th)); exact stratum
                   matching I_cond = sum_j w_own(j) (mean own dCE_j - mean other dCE_j) over strata with both sides;
                   support counts and effective sample sizes; adequacy rule.
  hier_boot        two-stage hierarchical bootstrap: seeds resampled with replacement, documents resampled through
                   the shared fixed-seed document-weight replicate (crossed design: same documents for every seed)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .slk_lib import N_DEC, POS_EDGES, interaction_from_stats, joint, strata

N_RANK = 4
N_GDEC = 10
N_MDEC = 10
ADEQ = {"n_own_matched_min": 100, "frac_own_matched_min": 0.5, "ess_other_min": 50.0, "n_own_docs_min": 10}


def _dec_within(x, n=10):
    r = np.argsort(np.argsort(x, kind="stable"), kind="stable")
    return np.minimum((r * n) // max(1, len(x)), n - 1)


@torch.no_grad()
def base_pass(model, split, rows, device, batch_rows=32):
    """Per-token baseline in the token order of lfr_eval.evaluate(per_token=True)."""
    from .lfr_eval import inputs
    model.eval()
    pt = {k: [] for k in ("ce", "dom", "gdoc", "pos", "topi", "top5p")}
    for i in range(0, len(rows), batch_rows):
        rr = rows[i:i + batch_rows]
        a, bt, mi = inputs(rr, split, device, None, None, None)
        ctl = {"teach": False, "collect": True, "exclude": None}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(mi, ctl)
            h = out["hidden"]
        tgt = bt["targets"]
        ce = torch.zeros(tgt.shape, device=device)
        hf = h.reshape(-1, h.shape[-1])
        tf = tgt.reshape(-1)
        cef = ce.view(-1)
        for s in range(0, tf.numel(), 8192):
            cef[s:s + 8192] = F.cross_entropy(model.lm_logits(hf[s:s + 8192]).float(), tf[s:s + 8192],
                                              ignore_index=-100, reduction="none")
        valid = tgt >= 0
        pt["ce"].append(ce[valid].float().cpu().numpy())
        pt["dom"].append(bt["true_domain"][valid].cpu().numpy().astype(np.int8))
        pt["gdoc"].append(bt["gdoc"][valid].cpu().numpy())
        pt["pos"].append(bt["positions"][valid].cpu().numpy().astype(np.int16))
        pt["topi"].append(torch.stack([r["topi"][valid] for r in out["records"]], 1).to(torch.uint8).cpu().numpy())
        pt["top5p"].append(torch.stack([r["p"][valid].float().topk(5, -1).values for r in out["records"]], 1)
                           .cpu().numpy().astype(np.float32))
    return {k: np.concatenate(v) for k, v in pt.items()}


def _cond_stats(d, own, oth, dinv, D, strat, dev):
    """Exposure-conditioned sufficient statistics restricted to strata with at least one own token (compressed)."""
    us, sinv = np.unique(strat[own], return_inverse=True)
    nU = len(us)
    pos_o = np.searchsorted(us, strat[oth])
    pos_o = np.minimum(pos_o, nU - 1)
    inU = us[pos_o] == strat[oth]
    oi = np.nonzero(oth)[0][inU]
    so = pos_o[inU]
    own_i = np.nonzero(own)[0]

    def dense(idx, col, w):
        m = np.bincount(dinv[idx] * nU + col, weights=w, minlength=D * nU).reshape(D, nU)
        return torch.as_tensor(m, device=dev, dtype=torch.float64)
    return (dense(own_i, sinv, d[own_i]), dense(own_i, sinv, None), dense(oi, so, d[oi]), dense(oi, so, None))


def _cond_from(os_, oc, xs, xc):
    """os_, oc, xs, xc: (B, nU). Exact stratum matching; returns I (B,), matched own count (B,)."""
    ok = (oc > 0) & (xc > 0)
    ocm = torch.where(ok, oc, torch.zeros_like(oc))
    num = torch.where(ok, os_ - oc * xs / xc.clamp(min=1e-12), torch.zeros_like(oc)).sum(1)
    den = ocm.sum(1)
    return num / den.clamp(min=1e-12), den


def per_expert_slc(model, sd, rows_d, sm, rows_pe, device, disc_topi, disc_lab, K, n_boot=1000, seed=20261118,
                   lift_min=1.5, progress=None):
    """Real-domain system only (discovery labels disc_lab on calibration tokens; measurement labels = true domain).
    Returns (summary dict, arrays dict)."""
    from .lfr_eval import evaluate
    E, Lm = model.cfg.n_experts, model.n_moe
    bpt = base_pass(model, sm, rows_pe, device)
    bce = bpt["ce"].astype(np.float64)
    lab = bpt["dom"].astype(np.int64)
    S_all, dec = strata(bce, bpt["pos"])
    nS = N_DEC * (len(POS_EDGES) + 1)
    pb = np.digitize(np.asarray(bpt["pos"], np.int64), POS_EDGES)
    docs, dinv = np.unique(bpt["gdoc"], return_inverse=True)
    D = len(docs)
    rng = np.random.default_rng(seed)
    W = rng.multinomial(D, np.ones(D) / D, size=n_boot).astype(np.float64)
    Wt = torch.as_tensor(np.vstack([np.ones((1, D)), W]), device=device, dtype=torch.float64)
    C = joint(disc_topi, np.asarray(disc_lab, np.int64), K, E)
    pk = C.sum((0, 2)) / C.sum()
    n_cell = Lm * E
    rec = {k: np.full(n_cell, np.nan) for k in ("inter", "inter_dec", "own", "oth", "lift", "ci_lo", "ci_hi",
                                                "cinter", "cci_lo", "cci_hi", "c_n_own_exp", "c_n_own_matched",
                                                "c_frac_matched", "c_ess_other", "c_n_own_docs", "c_n_oth_exp",
                                                "c_own_dce", "gen")}
    rec["kstar"] = np.full(n_cell, -1, np.int64)
    rec["n_own"] = np.zeros(n_cell, np.int64)
    boot = np.full((n_boot, n_cell), np.nan)
    cboot = np.full((n_boot, n_cell), np.nan)
    j = 0
    for li in range(Lm):
        ti = bpt["topi"][:, li, :].astype(np.int64)
        for e in range(E):
            r = evaluate(model, sm, rows_pe, device, exclude={li: [e]}, per_token=True, routing=False)
            rp = r["per_token"]
            if not (np.array_equal(rp["gdoc"], bpt["gdoc"]) and np.array_equal(rp["pos"], bpt["pos"])):
                raise RuntimeError("token order of the replacement pass differs from the base pass")
            d = rp["ce"].astype(np.float64) - bce
            rec["gen"][j] = float(d.mean())
            pe = C[li, :, e].sum()
            lift = (C[li, :, e] / max(1.0, pe)) / np.maximum(1e-12, pk)
            k = int(lift.argmax())
            rec["kstar"][j], rec["lift"][j] = k, float(lift[k])
            own = lab == k
            if own.any():
                oth = ~own
                # ---- ordinary (existing) interaction: identical arithmetic to slk_lib.per_expert_boot
                own_s = np.bincount(dinv[own], weights=d[own], minlength=D)
                own_c = np.bincount(dinv[own] * nS + S_all[own], minlength=D * nS).reshape(D, nS).astype(float)
                oth_s = np.bincount(dinv[oth] * nS + S_all[oth], weights=d[oth], minlength=D * nS).reshape(D, nS)
                oth_c = np.bincount(dinv[oth] * nS + S_all[oth], minlength=D * nS).reshape(D, nS).astype(float)
                pt_ = float(interaction_from_stats(own_s, own_c, oth_s, oth_c)[0])
                bs = interaction_from_stats(own_s, own_c, oth_s, oth_c, W)
                wk = np.bincount(dec[own], minlength=N_DEC) / own.sum()
                xm = np.asarray([d[oth & (dec == q)].mean() if (oth & (dec == q)).any() else 0.0
                                 for q in range(N_DEC)])
                o = float(d[own].mean())
                rec["inter"][j], rec["inter_dec"][j] = pt_, o - float((wk * xm).sum())
                rec["own"][j], rec["oth"][j] = o, o - pt_
                rec["n_own"][j] = int(own.sum())
                rec["ci_lo"][j], rec["ci_hi"][j] = float(np.nanquantile(bs, 0.025)), float(np.nanquantile(bs, 0.975))
                boot[:, j] = bs
                # ---- exposure-conditioned interaction
                hit = ti == e
                exp_ = hit.any(1)
                rank = hit.argmax(1)
                ex = np.nonzero(exp_)[0]
                gate = np.zeros(len(bce), np.float64)
                marg = np.zeros(len(bce), np.float64)
                gate[ex] = bpt["top5p"][ex, li, rank[ex]]
                marg[ex] = gate[ex] - bpt["top5p"][ex, li, 4]
                gd = np.zeros(len(bce), np.int64)
                md = np.zeros(len(bce), np.int64)
                gd[ex] = _dec_within(gate[ex], N_GDEC)
                md[ex] = _dec_within(marg[ex], N_MDEC)
                strat = (((dec * (len(POS_EDGES) + 1) + pb) * N_RANK + rank) * N_GDEC + gd) * N_MDEC + md
                own_e, oth_e = own & exp_, oth & exp_
                rec["c_n_own_exp"][j], rec["c_n_oth_exp"][j] = int(own_e.sum()), int(oth_e.sum())
                if own_e.any() and oth_e.any():
                    os_, oc, xs, xc = _cond_stats(d, own_e, oth_e, dinv, D, strat, device)
                    Ib, nm = _cond_from(Wt @ os_, Wt @ oc, Wt @ xs, Wt @ xc)
                    Ib = Ib.cpu().numpy()
                    oc0, xc0 = oc.sum(0).cpu().numpy(), xc.sum(0).cpu().numpy()
                    okj = (oc0 > 0) & (xc0 > 0)
                    n_m = float(oc0[okj].sum())
                    wj = np.where(okj, oc0 / max(1.0, n_m), 0.0)
                    # Kish ESS of the other-side weights: each other token in stratum j carries w_j / n_oth(j)
                    w2 = float(np.sum(np.where(okj, wj ** 2 / np.maximum(1.0, xc0), 0.0)))
                    ess = 1.0 / w2 if w2 > 0 else 0.0
                    own_docs = int(((oc[:, torch.as_tensor(okj, device=device)].sum(1)) > 0).sum().item())
                    rec["c_n_own_matched"][j] = n_m
                    rec["c_frac_matched"][j] = n_m / max(1, own_e.sum())
                    rec["c_ess_other"][j] = ess
                    rec["c_n_own_docs"][j] = own_docs
                    rec["c_own_dce"][j] = float(d[own_e].mean())
                    adequate = (n_m >= ADEQ["n_own_matched_min"] and n_m / own_e.sum() >= ADEQ["frac_own_matched_min"]
                                and ess >= ADEQ["ess_other_min"] and own_docs >= ADEQ["n_own_docs_min"])
                    if adequate:
                        rec["cinter"][j] = float(Ib[0])
                        cb = Ib[1:]
                        cboot[:, j] = cb
                        rec["cci_lo"][j], rec["cci_hi"][j] = float(np.nanquantile(cb, 0.025)), float(np.nanquantile(cb, 0.975))
            j += 1
            if progress:
                progress(li, e)
    inter = rec["inter"]
    ok = np.isfinite(inter)
    cin = rec["cinter"]
    cok = np.isfinite(cin)
    spec = ok & (rec["lift"] >= lift_min) & (rec["ci_lo"] > 0)
    cspec = cok & (rec["lift"] >= lift_min) & (rec["cci_lo"] > 0)
    bmean = np.nanmean(boot[:, ok], 1)
    cbmean = np.nanmean(cboot[:, cok], 1) if cok.any() else np.full(n_boot, np.nan)
    bl = np.stack([np.nanmean(np.where(ok, boot, np.nan).reshape(n_boot, Lm, E), 2)], 0)[0]
    cbl = np.full((n_boot, Lm), np.nan)
    for li in range(Lm):
        m = cok.reshape(Lm, E)[li]
        if m.any():
            cbl[:, li] = np.nanmean(cboot.reshape(n_boot, Lm, E)[:, li, m], 1)
    summ = {"n_tokens": int(len(bce)), "n_docs": int(D), "n_boot": n_boot, "boot_seed": seed,
            "matching_ordinary": "baseline-CE decile x position bin " + str(POS_EDGES) + " (slk_lib.per_expert_boot)",
            "matching_conditional": "exposed tokens; CE decile x position bin x router rank x gate-prob decile x "
                                    "router-margin decile (deciles within the cell's exposed tokens)",
            "adequacy": ADEQ,
            "mean_interaction": float(inter[ok].mean()), "n_defined": int(ok.sum()),
            "mean_interaction_ci95_doc": [float(np.quantile(bmean, 0.025)), float(np.quantile(bmean, 0.975))],
            "mean_interaction_decile_only": float(np.nanmean(rec["inter_dec"])),
            "frac_positive": float((inter[ok] > 0).mean()),
            "n_specialized": int(spec.sum()), "by_layer": np.nanmean(inter.reshape(Lm, E), 1).tolist(),
            "mean_own_dce": float(np.nanmean(rec["own"])), "mean_generic_dce": float(np.nanmean(rec["gen"])),
            "cond_mean_interaction": float(cin[cok].mean()) if cok.any() else None, "cond_n_adequate": int(cok.sum()),
            "cond_mean_interaction_ci95_doc": [float(np.nanquantile(cbmean, 0.025)), float(np.nanquantile(cbmean, 0.975))]
            if cok.any() else None,
            "cond_frac_positive": float((cin[cok] > 0).mean()) if cok.any() else None,
            "cond_n_specialized": int(cspec.sum()),
            "cond_by_layer": [float(np.nanmean(x)) if np.isfinite(x).any() else None for x in cin.reshape(Lm, E)],
            "cond_n_adequate_by_layer": cok.reshape(Lm, E).sum(1).tolist(),
            "cond_support_median": {k: float(np.nanmedian(rec[k])) for k in ("c_n_own_exp", "c_n_own_matched",
                                                                             "c_frac_matched", "c_ess_other",
                                                                             "c_n_own_docs")}}
    arrays = {**{k: v for k, v in rec.items()}, "boot_mean": bmean, "cboot_mean": cbmean, "boot_by_layer": bl,
              "cboot_by_layer": cbl, "load_meas": None}
    arrays.pop("load_meas")
    return summ, arrays


def hier_boot(per_seed_boot, n_rep=10000, seed=7, point=None):
    """per_seed_boot: (S, B) document-bootstrap replicates (shared document weights across seeds, column b = the same
    document resample for every seed). Each hierarchical replicate draws S seeds with replacement and one document
    replicate b; statistic = mean over drawn seeds of column b. Returns point (mean over seeds of `point` or of the
    replicate means), CI95, replicates."""
    X = np.asarray(per_seed_boot, float)
    S, B = X.shape
    rng = np.random.default_rng(seed)
    si = rng.integers(0, S, size=(n_rep, S))
    bi = rng.integers(0, B, size=n_rep)
    reps = X[si, bi[:, None]].mean(1)
    pt = float(np.mean(point)) if point is not None else float(X.mean())
    return {"point": pt, "ci95": [float(np.quantile(reps, 0.025)), float(np.quantile(reps, 0.975))],
            "frac_rep_gt0": float((reps > 0).mean())}, reps
