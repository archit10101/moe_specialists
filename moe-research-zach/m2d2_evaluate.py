#!/usr/bin/env python
"""csr-m2d2-domain-prior-v1: evaluate one checkpoint once on an evaluation set (val requires frozen.json; the locked
test split is refused) with CE, per-document / per-domain CE, routing specialisation diagnostics and optional
functional expert ablation.
  scripts/gpu_guarded.sh "m2d2 eval" m2d2_evaluate.py --run <run dir> --evalset val [--ablation] --execute
Outputs artifacts/m2d2-domain-prior-v1/evals/<evalset>__<run id>/ (summary.json, token_nll.npy, changed.npy,
doc_expert_counts.npy)."""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402

N_DOM = 6


def mi_from_counts(c):
    """c (D, E) joint counts -> mutual information (nats)."""
    p = c / max(c.sum(), 1)
    pd, pe = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log(p[nz] / (pd @ pe)[nz])).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--evalset", choices=["calib_search", "val"], required=True)
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    if not guard.require_execution(a.execute, "gpu"):
        return
    import torch
    import torch.nn.functional as F
    from csr.data import to_torch
    from csr.vm_run import EXP, build_model, check_cfg, eval_source
    run = guard.inside_root(a.run)
    cfg = json.loads((run / "config.json").read_text())
    key, d = check_cfg(cfg)
    if json.loads((run / "status.json").read_text())["status"] != "COMPLETE":
        raise guard.GuardError("run not COMPLETE")
    if a.evalset == "val":
        fz = json.loads((EXP["m2d2"]["art"] / "frozen.json").read_text())
        if guard.sha256_file(run / "config.json") not in fz["config_sha256s"]:
            raise guard.GuardError("val: run config not in the frozen set")
    ck_sha = guard.sha256_file(run / "model.pt")
    reg = EXP["m2d2"]["art"] / "eval_registry"
    reg.mkdir(parents=True, exist_ok=True)
    claim = reg / f"{a.evalset}__{ck_sha}.json"
    if claim.exists() and json.loads(claim.read_text()).get("status") == "COMPLETE":
        raise guard.GuardError(f"{a.evalset} already evaluated for {ck_sha[:12]}")
    guard.atomic_write_json(claim, {"status": "CLAIMED", "time": guard.now(), "pid": os.getpid()}, overwrite=True)
    outdir = EXP["m2d2"]["art"] / "evals" / f"{a.evalset}__{run.name}"
    outdir.mkdir(parents=True, exist_ok=False)
    guard.occupancy_gate("m2d2 eval")
    dev = guard.setup_cuda_device()
    mon = guard.OccupancyMonitor(interval=15.0)
    mon.start()
    guard.register_self(f"m2d2 eval {run.name} {a.evalset}")
    t0iso, t0 = guard.now(), time.time()
    res = {"run": str(run.relative_to(guard.ROOT)), "evalset": a.evalset, "checkpoint_sha256": ck_sha,
           "condition": cfg["condition"], "seed": cfg["init"]["seed"], "stage": cfg["stage"]}
    status = "FAILED"
    try:
        model = build_model(cfg, dev)
        model.load_state_dict(torch.load(run / "model.pt", map_location="cpu", weights_only=False)["model"])
        model.eval()
        src = eval_source(cfg, key, d, a.evalset)  # true domain labels (never shuffled at evaluation)
        n_docs = src.split.n_docs
        moe_ids = [i for i, b in enumerate(model.blocks) if b.is_moe]
        L, E = len(moe_ids), model.cfg.n_experts
        dec = np.zeros((n_docs, L, E), np.int32)  # doc x layer x expert top-4 assignment counts
        dec_np = np.zeros((n_docs, L, E), np.int32) if model.kind == "prior" else None
        toks, chg_all, doc_of_tok = [], [], []
        adj_j, adj_n = np.zeros(L), 0
        chg_layer = np.zeros(L)
        n_tok = 0
        with torch.no_grad():
            for s in range(0, len(src.rows), 16):
                tb = to_torch(src.batch(src.rows[s:s + 16]), dev)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(tb, collect=True)
                tgt = tb["targets"]
                valid = tgt != -100
                nll = F.cross_entropy(out.logits.float().view(-1, out.logits.shape[-1]), tgt.view(-1),
                                      ignore_index=-100, reduction="none").view(tgt.shape)
                if not torch.isfinite(nll[valid]).all():
                    raise FloatingPointError("non-finite NLL")
                toks.append(nll[valid].cpu().numpy().astype(np.float32))
                doc_of_tok.append(tb["gdoc"][valid].cpu().numpy().astype(np.int32))
                nonpad = tb["doc_ids"] >= 0
                n_tok += int(nonpad.sum())
                g = tb["gdoc"]
                same_next = nonpad[:, 1:] & (tb["doc_ids"][:, 1:] == tb["doc_ids"][:, :-1])
                adj_n += int(same_next.sum())
                changed_any = torch.zeros_like(valid)
                for li, r in enumerate(out.routing):
                    topi = r["topi"]
                    gd = g[nonpad].repeat_interleave(4)
                    ex = topi[nonpad].reshape(-1)
                    np.add.at(dec[:, li, :], (gd.cpu().numpy(), ex.cpu().numpy()), 1)
                    oh = torch.zeros(*topi.shape[:2], E, device=dev).scatter_(2, topi, 1.0)
                    inter = (oh[:, 1:] * oh[:, :-1]).sum(-1)
                    adj_j[li] += float((inter / (8 - inter))[same_next].sum())
                    if "topi_noprior" in r:
                        tn = r["topi_noprior"]
                        np.add.at(dec_np[:, li, :], (gd.cpu().numpy(), tn[nonpad].reshape(-1).cpu().numpy()), 1)
                        ch = (tn.sort(-1).values != topi.sort(-1).values).any(-1)
                        chg_layer[li] += float((ch & nonpad).sum())
                        changed_any |= ch
                chg_all.append(changed_any[valid].cpu().numpy())
                if mon.contaminated:
                    raise guard.GuardError(f"CONTAMINATED: {mon.detail}")
        nll_t = np.concatenate(toks)
        dot = np.concatenate(doc_of_tok)
        chg = np.concatenate(chg_all)
        doms = src.domains
        np.save(outdir / "token_nll.npy", nll_t)
        np.save(outdir / "token_doc.npy", dot)
        np.save(outdir / "changed.npy", chg)
        np.save(outdir / "doc_expert_counts.npy", dec)
        doc_nll = np.bincount(dot, weights=nll_t.astype(np.float64), minlength=n_docs)
        doc_n = np.bincount(dot, minlength=n_docs)
        dom_ce = {}
        for dd in range(N_DOM):
            m = doms == dd
            dom_ce[dd] = float(doc_nll[m].sum() / max(doc_n[m].sum(), 1))
        res.update({"ce_nats_per_token": float(nll_t.astype(np.float64).mean()), "n_valid_tokens": int(len(nll_t)),
                    "per_domain_ce": dom_ce, "macro_domain_ce": float(np.mean(list(dom_ce.values()))),
                    "doc_nll": doc_nll.tolist(), "doc_n": doc_n.tolist(), "doc_domain": doms.tolist()})
        # specialisation
        rng = np.random.default_rng(12345)
        spec = []
        for li in range(L):
            joint = np.stack([dec[doms == dd, li, :].sum(0) for dd in range(N_DOM)]).astype(np.float64)
            mi = mi_from_counts(joint)
            null = []
            for _ in range(20):  # load-preserving null: routing fixed, document labels permuted
                pd_ = doms[rng.permutation(len(doms))]
                null.append(mi_from_counts(np.stack([dec[pd_ == dd, li, :].sum(0) for dd in range(N_DOM)])))
            pe = joint / joint.sum(0, keepdims=True).clip(min=1)
            ent = -(pe * np.log(pe.clip(min=1e-30))).sum(0)
            load = joint.sum(0) / joint.sum()
            row = {"layer": moe_ids[li], "mi_domain_expert": mi, "mi_null_mean": float(np.mean(null)),
                   "mi_null_max": float(np.max(null)), "domain_entropy_per_expert": ent.tolist(),
                   "mean_domain_entropy_per_expert": float(ent.mean()), "max_entropy": float(np.log(N_DOM)),
                   "expert_load": load.tolist(), "experts_unused": int((joint.sum(0) == 0).sum()),
                   "expert_max_share": float(load.max()), "adjacent_token_jaccard": float(adj_j[li] / max(adj_n, 1)),
                   "top_domain_per_expert": joint.argmax(0).tolist()}
            if dec_np is not None:
                jn = np.stack([dec_np[doms == dd, li, :].sum(0) for dd in range(N_DOM)]).astype(np.float64)
                row["mi_without_prior_same_states"] = mi_from_counts(jn)
                row["frac_tokens_top4_changed_by_prior"] = float(chg_layer[li] / max(n_tok, 1))
            spec.append(row)
        res["specialisation"] = spec
        if hasattr(model, "alpha"):
            res["alpha"] = model.alpha.detach().float().tolist()
            res["prior_bias_abs_mean"] = float((model.alpha[:, None, None] * model.dom_bias).abs().mean())
        if model.kind == "prior":
            res["frac_tokens_any_change"] = float(chg.mean())
            res["ce_changed_tokens"] = float(nll_t[chg].astype(np.float64).mean()) if chg.any() else None
            res["ce_unchanged_tokens"] = float(nll_t[~chg].astype(np.float64).mean()) if (~chg).any() else None
        # functional ablation (optional): per MoE layer 4..7, the expert with the highest domain share; disable it
        # (gate 0, still eligible) on all tokens; CE increase on its top domain's docs vs other domains' docs
        if a.ablation:
            abl = []
            base_doc = doc_nll
            for li in range(L):
                layer = moe_ids[li]
                if layer < 4:
                    continue
                joint = np.stack([dec[doms == dd, li, :].sum(0) for dd in range(N_DOM)]).astype(np.float64)
                share = joint / joint.sum(0, keepdims=True).clip(min=1)
                e = int(share.max(0).argmax())
                td = int(share[:, e].argmax())
                tot = np.zeros(n_docs)
                with torch.no_grad():
                    for s in range(0, len(src.rows), 16):
                        tb = to_torch(src.batch(src.rows[s:s + 16]), dev)
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            o = model(tb, disable=(layer, e, tb["doc_ids"] >= 0))
                        tgt = tb["targets"]
                        valid = tgt != -100
                        nl = F.cross_entropy(o.logits.float().view(-1, o.logits.shape[-1]), tgt.view(-1),
                                             ignore_index=-100, reduction="none").view(tgt.shape)
                        np.add.at(tot, tb["gdoc"][valid].cpu().numpy(), nl[valid].double().cpu().numpy())
                m = doms == td
                abl.append({"layer": layer, "expert": e, "top_domain": td, "top_domain_share": float(share[td, e]),
                            "dce_top_domain": float((tot[m].sum() - base_doc[m].sum()) / doc_n[m].sum()),
                            "dce_other_domains": float((tot[~m].sum() - base_doc[~m].sum()) / doc_n[~m].sum())})
            res["functional_ablation"] = abl
        if mon.contaminated:
            raise guard.GuardError(f"CONTAMINATED: {mon.detail}")
        status = "COMPLETE"
    except Exception as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {e}"
        res["traceback"] = traceback.format_exc()
        status = "CONTAMINATED" if "CONTAMINATED" in str(e) else "FAILED"
    finally:
        mon.stop()
        res["status"] = status
        res["seconds"] = time.time() - t0
        guard.atomic_write_json(outdir / "summary.json", res)
        guard.atomic_write_json(claim, {"status": status, "time": guard.now(), "out": str(outdir.relative_to(guard.ROOT)),
                                        "result": res.get("ce_nats_per_token")}, overwrite=True)
        guard.gpu_ledger(t0iso, guard.now(), f"eval_{a.evalset}_{run.name}", "m2d2 eval", status)
        print(json.dumps({k: v for k, v in res.items() if k not in ("doc_nll", "doc_n", "doc_domain", "traceback",
                                                                    "specialisation")}, default=str)[:3000])
        print("EVAL_DIR", outdir)
        sys.exit(0 if status == "COMPLETE" else 1)


if __name__ == "__main__":
    main()
