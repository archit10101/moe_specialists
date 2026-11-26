#!/usr/bin/env python
"""csr-latent-forecast-router-distill-v1: draft provenance + held-out draft diagnostics + draft gate (protocol 4) +
cache compression error + rollout throughput projection. CALIB-SEARCH paper half A, first 0.4M tokens per domain.
Writes artifacts/latent-forecast-router-distill-v1/draft_provenance.json and draft_diagnostics.json.
GPU only via scripts/gpu_guarded.sh (--device cuda --execute)."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402

TOK_PER_DOM = 400_000
WEAK = "runs/20261110T074951Z_dfc_draft_d128_l4_adapt/ckpt_final.pt"


def provenance():
    from csr.lfr_lib import DRAFT_PATHS
    import json as _j
    out = {"drafts": []}
    names = ["general"] + [f"domain_d{i}" for i in range(5)]
    for nm, p in zip(names, DRAFT_PATHS):
        rd = guard.ROOT / Path(p).parent
        cfg = _j.loads((rd / "config.json").read_text())
        summ = _j.loads((rd / "summary.json").read_text()) if (rd / "summary.json").exists() else {}
        out["drafts"].append({"slot": nm, "path": p, "sha256": guard.sha256_file(guard.ROOT / p),
                              "kind": cfg.get("kind"), "domain": cfg.get("domain"), "init_from": cfg.get("init_from"),
                              "status": _j.loads((rd / "status.json").read_text()).get("status"),
                              "seen_tokens": summ.get("seen"), "data_split": cfg.get("kind")})
    out["isolation"] = ("draft_shared / draft_cont buckets of the fspv2 manifest (paper-disjoint from main, calib_search,"
                        " calib_sel, val, test by construction; overlap audited in lfr_tests T3)")
    out["original_training_gpu_seconds"] = 432.4 + 88.7 + 87.3 + 89.0 + 89.9 + 89.4
    return out


def weak_compare(res, wd_path, rows, sp, device, N_DOM):
    """Previous weak drafts (dfcv1 d128 l4 + per-domain adapters, correct-domain adapter); descriptive only."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from csr.fsp_lib import make_batch, to_torch
    try:
        from csr.dfc_model import load_frozen_draft
        wd = load_frozen_draft(wd_path, device)
        wce, wn, wacc = np.zeros(N_DOM), np.zeros(N_DOM), np.zeros(N_DOM)
        with torch.no_grad():
            for i in range(0, len(rows), 16):
                bt = to_torch(make_batch(rows[i:i + 16], sp), device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    lg = wd({"input_ids": bt["input_ids"], "doc_ids": bt["doc_ids"], "positions": bt["positions"],
                             "domain": bt["true_domain"]}).logits.float()
                c = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), bt["targets"].reshape(-1), ignore_index=-100,
                                    reduction="none").view(bt["targets"].shape)
                am = (lg.argmax(-1) == bt["targets"])
                for d in range(N_DOM):
                    m = (bt["targets"] >= 0) & (bt["true_domain"] == d)
                    wce[d] += float(c[m].sum())
                    wacc[d] += float(am[m].sum())
                    wn[d] += int(m.sum())
        res["weak_dfcv1_draft"] = {"ce": (wce / wn).tolist(), "acc_h1": (wacc / wn).tolist(),
                                   "caveat": "different data build; overlap with these papers not excluded"}
    except Exception as e:  # noqa: BLE001
        res["weak_dfcv1_draft"] = {"error": str(e)[:300]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--weak-only", action="store_true", help="only the previous-weak-draft comparison")
    a = ap.parse_args()
    guard.check_root()
    if not guard.require_execution(a.execute, "gpu" if a.device == "cuda" else "cpu-model"):
        return
    import numpy as np
    import torch
    import torch.nn.functional as F
    from csr.fsp_lib import Split, make_batch, to_torch, pack_rows
    from csr.lfr_lib import ART, N_DOM
    from csr.lfr_eval import eval_rows, paper_half
    from lfr_cache import load_draft, run_draft, compress_all
    if a.device == "cuda":
        guard.occupancy_gate("lfr draft diag")
        device = guard.setup_cuda_device()
        mon = guard.OccupancyMonitor(interval=15.0)
        mon.start()
    else:
        device, mon = torch.device("cpu"), None
    guard.register_self("lfr draft diag")
    t_iso, t0 = guard.now(), time.time()
    prov = provenance()
    guard.atomic_write_json(ART / "draft_provenance.json", prov, overwrite=True)
    sp = Split("calib_search")
    rows = eval_rows(sp, TOK_PER_DOM, paper_half(sp, 0))
    docs = [d for r in rows for d in r]
    L = np.asarray([int(sp.lengths[d]) for d in docs])
    dom_tok = np.concatenate([np.full(l_, sp.domains[d]) for d, l_ in zip(docs, L)])
    tok = np.concatenate([sp.doc(d) for d in docs])
    pos = np.concatenate([np.arange(l_) for l_ in L])
    docpos_end = np.concatenate([np.full(l_, l_) for l_ in L])
    if a.weak_only:
        res = {"n_tokens": int(L.sum())}
        weak_compare(res, wd_path=guard.ROOT / WEAK, rows=rows, sp=sp, device=device, N_DOM=N_DOM)
        guard.atomic_write_json(ART / "draft_diagnostics_weak.json", res, overwrite=True)
        if mon is not None:
            mon.stop()
        guard.gpu_ledger(t_iso, guard.now(), "lfr_draft_diag_weak", "lfr draft diag weak", "COMPLETE")
        with open(ART / "compute_ledger.tsv", "a") as f:
            f.write(f"{t_iso}\t{guard.now()}\t{time.time() - t0:.1f}\tlfr_draft_diag_weak\tdraft_diag_weak\t"
                    f"diagnostics\tCOMPLETE\n")
        print(json.dumps(res))
        return
    per, tps = [], []
    res = {"n_tokens": int(L.sum()), "tokens_per_domain": np.bincount(dom_tok, minlength=N_DOM).tolist(),
           "drafts": [d["slot"] for d in prov["drafts"]]}
    for slot in range(6):
        dr = load_draft(slot, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        ts = time.time()
        o = run_draft(dr, sp, docs, device, diag_topk=5)
        if device.type == "cuda":
            torch.cuda.synchronize()
        tps.append(int(L.sum()) / (time.time() - ts))
        # throughput of the cache path (no diagnostic top-k) on the first 256 documents
        ts = time.time()
        run_draft(dr, sp, docs[:256], device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        res.setdefault("cache_path_tok_per_s", []).append(int(L[:256].sum()) / (time.time() - ts))
        per.append(o)
        del dr
    # ---------------- metrics
    D = 6
    # CE of predicting x_t from x_<t at positions >= 1 (score = log p(x_t | x_<t))
    valid1 = pos >= 1
    ce = np.zeros((D, N_DOM))
    acc1 = np.zeros((D, N_DOM))
    for s in range(D):
        sc = per[s]["score"].numpy()
        for d in range(N_DOM):
            m = valid1 & (dom_tok == d)
            ce[s, d] = float(-sc[m].mean())
    # rollout accuracy at horizons 1, 4, 8 (top-1 and top-5): roll_topk[t, h-1] vs x_{t+h} inside the document
    hacc = {}
    for h in (1, 4, 8):
        ok = (pos + h) < docpos_end
        idx = np.flatnonzero(ok)
        tgt = tok[idx + h]
        for s in range(D):
            tk = per[s]["roll_topk"][:, h - 1, :].numpy()[idx]
            t1 = (tk[:, 0] == tgt)
            t5 = (tk == tgt[:, None]).any(1)
            for d in range(N_DOM):
                m = dom_tok[idx] == d
                hacc[f"h{h}_top1_s{s}_d{d}"] = float(t1[m].mean())
                hacc[f"h{h}_top5_s{s}_d{d}"] = float(t5[m].mean())
    res["ce"] = ce.tolist()
    res["horizon_acc"] = hacc
    # calibration (h=1): ECE of top-1 prob vs top-1 correctness for x_{t+1}; mean entropy
    ok1 = np.flatnonzero((pos + 1) < docpos_end)
    ece, ent = [], []
    for s in range(D):
        p1 = per[s]["top_p"][:, 0].float().numpy()[ok1]
        corr = (per[s]["top_ids"][:, 0].numpy()[ok1] == tok[ok1 + 1]).astype(np.float64)
        bins = np.minimum((p1 * 15).astype(int), 14)
        e = 0.0
        for b in range(15):
            m = bins == b
            if m.any():
                e += m.mean() * abs(p1[m].mean() - corr[m].mean())
        ece.append(float(e))
        ent.append(float(per[s]["ent1"].numpy().mean()))
    res["ece_h1"], res["entropy_h1_mean"] = ece, ent
    # pairwise top-1 agreement at h = 1, 4, 8
    agree = {}
    for h in (1, 4, 8):
        R = np.stack([per[s]["roll"][:, h - 1].numpy() for s in range(D)])
        agree[f"h{h}"] = [[float((R[i] == R[j]).mean()) for j in range(D)] for i in range(D)]
    res["pairwise_top1_agreement"] = agree
    off = [agree["h1"][i][j] for i in range(D) for j in range(D) if i < j]
    res["mean_pairwise_disagreement_h1"] = float(1.0 - np.mean(off))
    # ensemble (equal-weight mixture): CE from exp(score) mean; best single draft per domain
    P = np.stack([np.exp(per[s]["score"].numpy()) for s in range(D)])
    ens_ce = [float(-np.log(P[:, valid1 & (dom_tok == d)].mean(0)).mean()) for d in range(N_DOM)]
    res["ensemble_ce"] = ens_ce
    res["best_single_ce"] = ce.min(0).tolist()
    # own-domain vs wrong-domain advantage and gate (protocol 4)
    adv_other, adv_shared, gate_dom = [], [], []
    for d in range(N_DOM):
        own = ce[1 + d, d]
        others = [ce[1 + k, d] for k in range(N_DOM) if k != d]
        adv_other.append(float(min(others) - own))
        adv_shared.append(float(ce[0, d] - own))
        gate_dom.append(bool(min(others) - own >= 0.05 and ce[0, d] - own >= 0.03))
    res["own_vs_best_other_domain_draft"] = adv_other
    res["own_vs_general"] = adv_shared
    res["gate"] = {"domains_passing": int(sum(gate_dom)), "per_domain": gate_dom,
                   "disagreement_ok": res["mean_pairwise_disagreement_h1"] >= 0.05,
                   "pass": bool(sum(gate_dom) >= 4 and res["mean_pairwise_disagreement_h1"] >= 0.05)}
    weak_compare(res, wd_path=guard.ROOT / WEAK, rows=rows, sp=sp, device=device, N_DOM=N_DOM)
    # compression error of the cache representation vs uncompressed outputs
    arrays, stats = compress_all([{k: p[k] for k in ("top_ids", "top_p", "ent1", "score", "roll4", "roll8")}
                                  for p in per], device)
    q_ent = np.abs(arrays["ent1"].astype(np.float64) / 255 * np.log(50304) -
                   np.stack([p["ent1"].numpy() for p in per], 1)).mean()
    q_p1 = np.abs(arrays["p1"].astype(np.float64) / 255 - np.stack([p["top_p"][:, 0].float().numpy() for p in per],
                                                                   1)).mean()
    q_sc = np.abs(-(arrays["score"].astype(np.float64) / 10) -
                  np.maximum(np.stack([p["score"].numpy() for p in per], 1), -25.5)).mean()
    res["compression"] = {**stats, "mae_entropy_nats": float(q_ent), "mae_top1_prob": float(q_p1),
                          "mae_score_nats": float(q_sc), "bytes_per_token": 44}
    res["diag_tok_per_s"] = tps
    res["seconds"] = time.time() - t0
    guard.atomic_write_json(ART / "draft_diagnostics.json", res, overwrite=True)
    if mon is not None:
        mon.stop()
    if a.device == "cuda":
        guard.gpu_ledger(t_iso, guard.now(), "lfr_draft_diag", "lfr draft diag", "COMPLETE")
        with open(ART / "compute_ledger.tsv", "a") as f:
            f.write(f"{t_iso}\t{guard.now()}\t{time.time() - t0:.1f}\tlfr_draft_diag\tdraft_diag\tdiagnostics\t"
                    f"COMPLETE\n")
    print(json.dumps({k: res[k] for k in ("ce", "gate", "compression", "cache_path_tok_per_s")}, indent=1))


if __name__ == "__main__":
    main()
