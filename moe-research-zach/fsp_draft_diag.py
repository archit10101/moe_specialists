#!/usr/bin/env python
"""fspv2 draft diagnostics on held-out CALIB-SEARCH documents (protocol section 4; diagnostics, not stop gates).
For the shared draft and the five domain drafts: next-token CE / accuracy per domain, rollout accuracy at horizons
1/4/8, matching-vs-non-matching advantage, correct-vs-wrong-domain rollout similarity, and domain classification
from forecast features (score argmax; label-free linear probes trained on half of the documents, scored on the other
half). Writes artifacts/forecast-specialist-pretraining-v2/draft_diagnostics.json. Dry-run by default."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402

TOK_PER_DOM = 1_000_000


def main():
    guard.check_root()
    if not guard.require_execution("--execute" in sys.argv, "gpu"):
        return
    import numpy as np
    import torch
    import torch.nn.functional as F
    from csr.fsp_eval import eval_rows
    from csr.fsp_lib import ART, N_DOM, Split, build_draft, draft_rollout_fast, make_batch, to_torch
    from csr.fsp_model import ema_matrix
    guard.occupancy_gate("fsp draft diag")
    device = guard.setup_cuda_device()
    guard.register_self("fsp draft diag")
    t_iso, t0 = guard.now(), time.time()
    dr = json.loads((ART / "drafts.json").read_text())
    paths = [dr["shared"]] + dr["domain"]
    names = ["shared"] + [f"draft_d{d}" for d in range(N_DOM)]
    sp = Split("calib_search")
    rows = [[d] for r in eval_rows(sp, TOK_PER_DOM) for d in r]  # single-document rows (fast causal rollouts)
    per = {n: {"roll": [], "lp": [], "ent": [], "emb": []} for n in names}
    meta = {"x": [], "dom": [], "gdoc": [], "pos": [], "nxt": []}
    for di, (n, p) in enumerate(zip(names, paths)):
        m = build_draft(0)
        m.load_state_dict(torch.load(guard.ROOT / p, map_location="cpu", weights_only=True)["model"])
        m = m.to(device).eval()
        for i in range(0, len(rows), 32):
            bt = to_torch(make_batch(rows[i:i + 32], sp), device)
            Ls = torch.as_tensor([int(sp.lengths[r[0]]) for r in rows[i:i + 32]], device=device)
            roll, ent, top4, lp = draft_rollout_fast(m, bt["input_ids"], Ls, H=8)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                W = ema_matrix(bt["doc_ids"], bt["positions"], 64.0, torch.float32)
                emb = torch.bmm(W, m.tok_emb(roll).float().mean(-2)) if n == "shared" else None
            npm = bt["doc_ids"] >= 0
            per[n]["roll"].append(roll[npm].cpu().numpy().astype(np.int32))
            per[n]["lp"].append(lp[npm].float().cpu().numpy())
            per[n]["ent"].append(ent[npm].float().cpu().numpy())
            if emb is not None:
                per[n]["emb"].append(emb[npm].float().cpu().numpy())
            if di == 0:
                ids = bt["input_ids"]
                B, T = ids.shape
                # x_{t+k} for k=1..8 within the same document (-1 if beyond the document)
                nx = torch.full((B, T, 8), -1, dtype=torch.long, device=device)
                for k in range(1, 9):
                    same = (bt["doc_ids"][:, k:] == bt["doc_ids"][:, :-k]) & (bt["doc_ids"][:, k:] >= 0)
                    nx[:, :-k, k - 1] = torch.where(same, ids[:, k:], torch.full_like(ids[:, k:], -1))
                meta["x"].append(ids[npm].cpu().numpy())
                meta["dom"].append(bt["true_domain"][npm].cpu().numpy())
                meta["gdoc"].append(bt["gdoc"][npm].cpu().numpy())
                meta["pos"].append(bt["positions"][npm].cpu().numpy())
                meta["nxt"].append(nx[npm].cpu().numpy())
        del m
        torch.cuda.empty_cache()
    cat = lambda xs: np.concatenate(xs, 0)
    meta = {k: cat(v) for k, v in meta.items()}
    for n in names:
        per[n] = {k: cat(v) for k, v in per[n].items() if v}
    dom, pos, nxt = meta["dom"], meta["pos"], meta["nxt"]
    has_prev = pos >= 1
    res = {"n_tokens": int(len(dom)), "tokens_per_domain": TOK_PER_DOM, "split": "calib_search", "drafts": paths}
    ce = np.zeros((len(names), N_DOM))
    acc = np.zeros((len(names), N_DOM))
    racc = np.zeros((len(names), N_DOM, 8))
    for i, n in enumerate(names):
        lp, roll = per[n]["lp"], per[n]["roll"]
        prev_hit = np.zeros(len(dom), bool)
        prev_hit[1:] = roll[:-1, 0] == meta["x"][1:]
        for d in range(N_DOM):
            m = (dom == d) & has_prev
            ce[i, d] = float(-lp[m].mean())
            acc[i, d] = float(prev_hit[m].mean())
            for k in range(8):
                mk = (dom == d) & (nxt[:, k] >= 0)
                racc[i, d, k] = float((roll[mk, k] == nxt[mk, k]).mean())
    res["ce"] = {n: ce[i].tolist() for i, n in enumerate(names)}
    res["next_token_acc"] = {n: acc[i].tolist() for i, n in enumerate(names)}
    res["rollout_acc_h1_h4_h8"] = {n: racc[i][:, [0, 3, 7]].tolist() for i, n in enumerate(names)}
    dd = np.arange(N_DOM)
    match_ce = ce[1 + dd, dd]
    other_ce = np.array([np.mean([ce[1 + e, d] for e in range(N_DOM) if e != d]) for d in range(N_DOM)])
    res["matching_vs_nonmatching"] = {
        "ce_matching": match_ce.tolist(), "ce_nonmatching_mean": other_ce.tolist(),
        "ce_advantage": (other_ce - match_ce).tolist(), "ce_shared": ce[0].tolist(),
        "rollout_acc_matching": [racc[1 + d, d][[0, 3, 7]].tolist() for d in range(N_DOM)],
        "rollout_acc_nonmatching_mean": [np.mean([racc[1 + e, d][[0, 3, 7]] for e in range(N_DOM) if e != d], 0)
                                         .tolist() for d in range(N_DOM)],
        "matching_is_best_draft_per_domain": [bool(np.argmin(ce[1:, d]) == d) for d in range(N_DOM)]}
    # correct vs wrong rollout similarity (token agreement per horizon)
    wrong = sp.wrong[meta["gdoc"]]
    R = np.stack([per[f"draft_d{d}"]["roll"] for d in range(N_DOM)], 0)  # (5, n, 8)
    idx = np.arange(len(dom))
    rm, rw = R[dom, idx], R[wrong, idx]
    res["correct_vs_wrong_rollout_agreement_by_horizon"] = (rm == rw).mean(0).tolist()
    res["correct_vs_shared_rollout_agreement_by_horizon"] = (rm == per["shared"]["roll"]).mean(0).tolist()
    pair = np.zeros((N_DOM, N_DOM))
    for a in range(N_DOM):
        for b in range(N_DOM):
            pair[a, b] = float((R[a, :, :4] == R[b, :, :4]).mean())
    res["pairwise_draft_rollout_agreement_h1to4"] = pair.tolist()
    # domain classification from forecast features
    Lp = torch.as_tensor(np.stack([per[f"draft_d{d}"]["lp"] for d in range(N_DOM)], -1), device=device)
    gd = torch.as_tensor(meta["gdoc"], device=device)
    # causal EMA of scores inside each document (sequential order of positions is preserved in the flat arrays)
    sc_ema = torch.zeros_like(Lp)
    r = 0.5 ** (1 / 64.0)
    run = torch.zeros(N_DOM, device=device)
    wsum = 0.0
    lp_np = Lp.cpu().numpy()
    out_np = np.zeros_like(lp_np)
    prev = -1
    rs = np.zeros(N_DOM)
    ws = 0.0
    for t_ in range(len(dom)):
        if meta["gdoc"][t_] != prev or pos[t_] == 0:
            rs[:] = 0
            ws = 0.0
            prev = meta["gdoc"][t_]
        rs = r * rs + lp_np[t_]
        ws = r * ws + 1.0
        out_np[t_] = rs / ws
    pred = out_np.argmax(-1)
    buckets = [(0, 64), (64, 256), (256, 1024)]
    res["domain_cls_score_argmax"] = {"overall": float((pred == dom).mean()),
                                      "by_pos": [float((pred == dom)[(pos >= lo) & (pos < hi)].mean())
                                                 for lo, hi in buckets]}
    ents = np.stack([per[f"draft_d{d}"]["ent"].mean(-1) for d in range(N_DOM)], -1)
    test_doc = np.asarray([((g * 2654435761) % 2**32) / 2**32 < 0.5 for g in meta["gdoc"]])

    def probe(X, tag):
        Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
        mu, sd = Xt[~torch.as_tensor(test_doc)].mean(0), Xt[~torch.as_tensor(test_doc)].std(0) + 1e-6
        Xt = (Xt - mu) / sd
        y = torch.as_tensor(dom, device=device)
        tr = ~torch.as_tensor(test_doc, device=device)
        lin = torch.nn.Linear(Xt.shape[1], N_DOM).to(device)
        o = torch.optim.Adam(lin.parameters(), lr=0.05)
        for _ in range(300):
            o.zero_grad()
            F.cross_entropy(lin(Xt[tr]), y[tr]).backward()
            o.step()
        with torch.no_grad():
            pr = lin(Xt[~tr]).argmax(-1).cpu().numpy()
        yt, pt = dom[test_doc], pos[test_doc]
        return {"overall": float((pr == yt).mean()), "by_pos": [float((pr == yt)[(pt >= lo) & (pt < hi)].mean())
                                                                for lo, hi in buckets], "features": tag}
    res["domain_cls_probe_scores_entropies"] = probe(np.concatenate([out_np, ents], -1),
                                                     "EMA(64) log-prob scores of the 5 domain drafts + their mean "
                                                     "rollout entropies (label-free)")
    res["domain_cls_probe_shared_rollout_content"] = probe(per["shared"]["emb"],
                                                           "EMA(64) of mean shared-draft embeddings of its 8 rollout "
                                                           "tokens (content only)")
    res["seconds"] = time.time() - t0
    guard.atomic_write_json(ART / "draft_diagnostics.json", res, overwrite=True)
    t_end = guard.now()
    guard.gpu_ledger(t_iso, t_end, "fsp_draft_diag", "fsp draft diagnostics", "COMPLETE")
    with open(ART / "compute_ledger.tsv", "a") as f:
        f.write(f"{t_iso}\t{t_end}\t{time.time() - t0:.1f}\tfsp_draft_diag\tdraft_diag\tdiagnostics\tCOMPLETE\n")
    print(json.dumps({k: v for k, v in res.items() if k in ("ce", "matching_vs_nonmatching",
                                                              "domain_cls_score_argmax")}, indent=1))


if __name__ == "__main__":
    main()
