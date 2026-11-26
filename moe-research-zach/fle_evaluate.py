#!/usr/bin/env python
"""csr-forecast-latent-edl-v1 evaluation jobs (student / inference mode: no drafts, encoder, cache or labels reach the
model). Outputs: artifacts/forecast-latent-edl-v1/evals/<run_name>/<mode>_<tag>.json (written once, 0400).

  dev      --run-dir R --tag T : CALIB-SEARCH half-B CE (0.4M tok/domain) + dev audit (discovery half A, measurement
                                 half B, 0.2M tok/domain; domain + e5 labels; FSI without the shuffled-label control)
                                 + latent-cluster/expert MI (if --latent ID given) + per-domain routing distributions
  calibsel --run-dir R --tag T : stage 3 only (STAGE3_OPEN.json): CALIB-SELECTION CE + audit (meas = CALIB-SELECTION)
  final    --run-dir R         : only after frozen.json: full-VAL CE of p100 (overall / domain / document) + audits of
                                 every milestone on the VAL audit subset (shuffled-label control at p100) + per-expert
                                 counterfactual interventions at p100 (--per-expert)
GPU only via scripts/gpu_guarded.sh (--device cuda --execute)."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402

AUDIT_TOK = 200_000
CE_TOK = 400_000
PE_TOK = 50_000


def load_milestone(path, device):
    import torch
    from csr.lfr_model import LFRConfig, LFRModel
    ck = torch.load(guard.ROOT / path / "state.pt", map_location="cpu", weights_only=False)
    mc = LFRConfig.from_dict({**ck["model_cfg"], "teacher": False})
    m = LFRModel(mc, ck["config"]["seed"])
    missing = m.load_state_dict(ck["model"], strict=True)
    return m.to(device).eval(), ck


def route_dists(meas, E):
    """Per-domain expert selection distributions (L, 5, E) from per-token top-4."""
    import numpy as np
    topi, dom = meas["topi"], meas["dom"].astype(np.int64)
    L = topi.shape[1]
    out = np.zeros((L, 5, E))
    for li in range(L):
        for d in range(5):
            m = dom == d
            out[li, d] = np.bincount(topi[m, li, :].reshape(-1).astype(np.int64), minlength=E)
    return out / np.maximum(1, out.sum(-1, keepdims=True))


def latent_token_labels(meas, table):
    """Hard forecast-latent cluster per evaluated token (region argmax of the frozen q; -1 for no region)."""
    import numpy as np
    z = np.load(table)
    L = int(z["L"])
    key = {(int(g), int(j)): int(np.argmax(q)) for g, j, q in zip(z["gdoc"], z["j"], z["q"])}
    lab = np.asarray([key.get((int(g), int(p) // L), -1) for g, p in zip(meas["gdoc"], meas["pos"])], np.int64)
    return lab, int(z["K"])


def audit_one(model, sd, rows_d, sm, rows_m, device, wd, wm, shuffled_ctrl, latent_table=None, per_expert_rows=None):
    import numpy as np
    from csr.lfr_audit import (assign_topics, functional, routing_metrics, shuffled_labels, token_labels,
                               topic_prediction_acc, window_hist, within_expert_similarity)
    from csr.lfr_eval import evaluate
    E = model.cfg.n_experts
    disc = evaluate(model, sd, rows_d, device, per_token=True)["per_token"]
    mf = evaluate(model, sm, rows_m, device, per_token=True)
    meas = mf["per_token"]
    out = {"ce_meas": mf["ce"], "ce_dom_meas": mf["ce_dom"], "load": mf["load"], "routing_entropy": mf["ent"],
           "route_dist_by_domain": route_dists(meas, E).tolist()}
    for sysn in ("domain", "cluster"):
        ld, K = token_labels(disc, sd, sysn, wd)
        lm, _ = token_labels(meas, sm, sysn, wm)
        rm = routing_metrics(meas["topi"], lm, K, E)
        groups = assign_topics(disc["topi"], ld, K, E)
        fs = functional(model, sm, rows_m, device, meas, lm, K, groups)
        res = {"routing": {k: v for k, v in rm.items() if k != "topic_entropy_per_expert"}, "functional": fs,
               "fsi": fs["fsi"], "fsi_raw": fs["fsi_raw"],
               "groups": {str(k): {str(l_): v for l_, v in g.items()} for k, g in groups.items()}}
        if shuffled_ctrl:
            lds = shuffled_labels(disc, ld, sysn, wd)
            lms = shuffled_labels(meas, lm, sysn, wm)
            fss = functional(model, sm, rows_m, device, meas, lms, K, assign_topics(disc["topi"], lds, K, E))
            res["fsi_shuf"] = fss["fsi"]
        wdk = disc["gdoc"].astype(np.int64) * 10000 + disc["pos"].astype(np.int64) // 256
        wmk = meas["gdoc"].astype(np.int64) * 10000 + meas["pos"].astype(np.int64) // 256
        Hd, yd = window_hist(disc, disc["topi"], E, ld, wdk, K)
        Hm, ym = window_hist(meas, meas["topi"], E, lm, wmk, K)
        res["topic_pred_acc"] = topic_prediction_acc(Hd, yd, Hm, ym, K)
        res["topic_pred_chance"] = float(np.bincount(ym, minlength=K).max() / len(ym))
        if sysn == "cluster" and "emb" in wm:
            emb = wm["emb"].astype(np.float32)
            key = wm["doc"].astype(np.int64) * 10000 + wm["start"].astype(np.int64) // 256
            pos_of = {int(k_): i for i, k_ in enumerate(key)}
            sel = np.asarray([pos_of.get(int(k_), -1) for k_ in wmk])
            ok = sel >= 0
            Lr = meas["topi"].shape[1]
            cnt = np.zeros((len(emb), Lr * E), np.float32)
            for li in range(Lr):
                for kk in range(meas["topi"].shape[2]):
                    np.add.at(cnt, (sel[ok], li * E + meas["topi"][ok, li, kk].astype(np.int64)), 1.0)
            res["within_expert_similarity"] = within_expert_similarity(cnt, emb)
        out[sysn] = res
    if latent_table is not None:
        lab, K = latent_token_labels(meas, latent_table)
        ok = lab >= 0
        rm = routing_metrics(meas["topi"][ok], lab[ok], K, E)
        out["latent"] = {"mi": rm["mi"], "mi_mean": rm["mi_mean"], "mi_null_mean": rm["mi_null_mean"],
                         "coverage": float(ok.mean()), "K": K}
    if per_expert_rows is not None:
        out["per_expert"] = per_expert(model, sd, rows_d, sm, per_expert_rows, device, disc)
    return out


def per_expert(model, sd, rows_d, sm, rows_pe, device, disc):
    """Per-expert counterfactual interventions (prereg 9): strongest domain k*(l,e) by lift on discovery; disable only
    that expert; interaction = mean dCE on own-domain tokens - difficulty-matched (baseline-CE decile) mean dCE on
    other-domain tokens."""
    import numpy as np
    from csr.lfr_audit import N_DEC, joint_counts
    from csr.lfr_eval import evaluate
    E, Lm = model.cfg.n_experts, model.n_moe
    C = joint_counts(disc["topi"], disc["dom"].astype(np.int64), 5, E)
    pk = C.sum((0, 2)) / C.sum()
    base = evaluate(model, sm, rows_pe, device, per_token=True, routing=False)["per_token"]
    bce, dom = base["ce"], base["dom"].astype(np.int64)
    dec = np.minimum((np.argsort(np.argsort(bce)) * N_DEC) // len(bce), N_DEC - 1)
    inter, ostar, xstar, kstar, lift_star = [], [], [], [], []
    mat = np.zeros((Lm, E, 5))
    for li in range(Lm):
        for e in range(E):
            pe = C[li, :, e].sum()
            lift = (C[li, :, e] / max(1.0, pe)) / np.maximum(1e-12, pk)
            k = int(lift.argmax())
            r = evaluate(model, sm, rows_pe, device, exclude={li: [e]}, per_token=True, routing=False)
            d = r["per_token"]["ce"] - bce
            for dd in range(5):
                mat[li, e, dd] = float(d[dom == dd].mean())
            own = dom == k
            o = float(d[own].mean())
            wk = np.bincount(dec[own], minlength=N_DEC) / max(1, own.sum())
            oth = ~own
            xm = np.asarray([d[oth & (dec == j)].mean() if (oth & (dec == j)).any() else 0.0 for j in range(N_DEC)])
            x = float((wk * xm).sum())
            inter.append(o - x)
            ostar.append(o)
            xstar.append(x)
            kstar.append(k)
            lift_star.append(float(lift[k]))
    inter = np.asarray(inter)
    return {"interaction": inter.tolist(), "mean_interaction": float(inter.mean()),
            "frac_positive": float((inter > 0).mean()), "own": ostar, "other_matched": xstar, "kstar": kstar,
            "lift_kstar": lift_star, "dce_by_domain": mat.tolist(), "n_tokens": int(len(bce))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["dev", "calibsel", "final"])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--tag")
    ap.add_argument("--latent")
    ap.add_argument("--per-expert", action="store_true")
    ap.add_argument("--tags", default="p040,p060,p080,p100")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    if not guard.require_execution(a.execute, "gpu" if a.device == "cuda" else "cpu-model"):
        return
    import numpy as np
    import torch
    from csr.fsp_lib import Split
    from csr.fle_paths import ART, LATENT
    from csr.lfr_eval import eval_rows, evaluate, paper_half
    rd = guard.inside_root(a.run_dir)
    st = json.loads((rd / "status.json").read_text())
    ckpts = st["ckpts"]
    od = ART / "evals" / a.run_name
    od.mkdir(parents=True, exist_ok=True)
    out_p = od / (f"{a.mode}_{a.tag}.json" if a.mode != "final" else "final.json")
    if out_p.exists():
        raise guard.GuardError(f"{out_p} exists (each evaluation runs exactly once)")
    if a.mode == "calibsel" and not (ART / "STAGE3_OPEN.json").exists():
        raise guard.GuardError("CALIB-SELECTION is opened only at stage 3")
    if a.mode == "final" and not (ART / "frozen.json").exists():
        raise guard.GuardError("VAL is opened only after the freeze")
    mon = None
    if a.device == "cuda":
        guard.occupancy_gate(f"fle evaluate {a.mode}")
        device = guard.setup_cuda_device()
        mon = guard.OccupancyMonitor(interval=15.0)
        mon.start()
    else:
        device = torch.device("cpu")
    guard.register_self(f"fle evaluate {a.mode} {a.run_name}")
    t_iso, t0 = guard.now(), time.time()
    lab = guard.ROOT / "artifacts/latent-forecast-router-distill-v1/labels"
    exp = {"windows_calib.npz": None, "windows_val.npz": None}
    wc = dict(np.load(lab / "windows_calib.npz"))
    cs = Split("calib_search")
    disc_rows = eval_rows(cs, AUDIT_TOK, paper_half(cs, 0))
    res = {"mode": a.mode, "run": rd.name, "run_name": a.run_name, "time": guard.now(), "ckpts": ckpts,
           "labels_sha256": {"windows_calib.npz": guard.sha256_file(lab / "windows_calib.npz")}}
    status = "COMPLETE"
    lt = None
    try:
        if a.mode == "dev":
            m, _ = load_milestone(ckpts[a.tag]["path"], device)
            r = evaluate(m, cs, eval_rows(cs, CE_TOK, paper_half(cs, 1)), device, routing=True)
            res["calib_search_B"] = {k: v for k, v in r.items() if k not in ("doc_ce",)}
            if a.latent:
                lt = LATENT / a.latent / "q_calib_search.npz"
            res["audit"] = audit_one(m, cs, disc_rows, cs, eval_rows(cs, AUDIT_TOK, paper_half(cs, 1)), device, wc, wc,
                                     False, lt)
        elif a.mode == "calibsel":
            m, _ = load_milestone(ckpts[a.tag]["path"], device)
            sp = Split("calib_sel")
            r = evaluate(m, sp, eval_rows(sp, CE_TOK), device, routing=True)
            res["calib_sel"] = {k: v for k, v in r.items() if k not in ("doc_ce",)}
            ws = dict(np.load(ART / "labels" / "windows_calib_sel.npz"))
            res["labels_sha256"]["windows_calib_sel.npz"] = guard.sha256_file(ART / "labels" / "windows_calib_sel.npz")
            res["audit"] = audit_one(m, cs, disc_rows, sp, eval_rows(sp, AUDIT_TOK), device, wc, ws, False)
        else:
            wv = dict(np.load(lab / "windows_val.npz"))
            res["labels_sha256"]["windows_val.npz"] = guard.sha256_file(lab / "windows_val.npz")
            sp = Split("val")
            m, _ = load_milestone(ckpts["p100"]["path"], device)
            r = evaluate(m, sp, eval_rows(sp), device, routing=True)
            res["val"] = {k: v for k, v in r.items() if k != "doc_ce"}
            res["val_doc_ce"] = {str(k): v for k, v in r["doc_ce"].items()}
            del m
            meas_rows = eval_rows(sp, AUDIT_TOK)
            pe_rows = eval_rows(sp, PE_TOK)
            if a.latent and (LATENT / a.latent / "q_val_audit.npz").exists():
                lt = LATENT / a.latent / "q_val_audit.npz"
            res["audit"] = {}
            for tag in a.tags.split(","):
                if tag not in ckpts:
                    continue
                mt, _ = load_milestone(ckpts[tag]["path"], device)
                res["audit"][tag] = audit_one(mt, cs, disc_rows, sp, meas_rows, device, wc, wv, tag == "p100", lt,
                                              pe_rows if (a.per_expert and tag == "p100") else None)
                del mt
    except KeyboardInterrupt:
        status = "EXTERNALLY_STOPPED"
    finally:
        if mon is not None:
            mon.stop()
            if mon.contaminated and status == "COMPLETE":
                status = "EXTERNALLY_STOPPED"
    res["seconds"] = time.time() - t0
    if status == "COMPLETE":
        guard.atomic_write_json(out_p, res, mode=0o400)
    if a.device == "cuda":
        guard.gpu_ledger(t_iso, guard.now(), f"fle_eval_{a.mode}_{a.run_name}", f"fle evaluate {a.mode}", status)
        with open(ART / "compute_ledger.tsv", "a") as f:
            f.write(f"{t_iso}\t{guard.now()}\t{time.time() - t0:.1f}\t{rd.name}\teval_{a.mode}_{a.tag or ''}_"
                    f"{a.run_name}\tevaluation\t{status}\n")
    print(json.dumps({"status": status, "seconds": res["seconds"]}))
    sys.exit(0 if status == "COMPLETE" else 77)


if __name__ == "__main__":
    main()
