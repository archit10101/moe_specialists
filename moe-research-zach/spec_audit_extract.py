#!/usr/bin/env python
"""specialization-audit-v1 §6/§8: routing extraction + single-token counterfactual expert replacement for ONE
checkpoint, fp32 (TF32 off), eval mode, no gradients. No label is read here: outputs are label-free.

Writes runs/<UTC>_specaudit_v1_extract_s{S}_{C}/: routing.npz (all mask tokens), cf.npz (effect-half mask tokens,
selected layers x 4 ranks), summary.json, status.json. Verifies checkpoint sha256 and a parameter fingerprint before
and after. --max-rows limits rows per half (benchmark). GPU via scripts/gpu_guarded.sh."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402

AUD = guard.ROOT / "artifacts" / "specialization-audit-v1"
SEGFILE = {"none": None, "fixed": "audit_fixed.npz", "semantic": "audit_semantic.npz", "random": "audit_random.npz"}


def window_index(W, offsets):
    """Global mask-token index: windows in file order, positions ascending."""
    import numpy as np
    wl = W["end"] - W["start"]
    woff = np.concatenate([[0], np.cumsum(wl)]).astype(np.int64)
    return wl, woff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True, help="e.g. s0_E")
    ap.add_argument("--cf-layers", default="0,3,6", help="MoE layer indices (0-based) for counterfactuals")
    ap.add_argument("--rows-per-batch", type=int, default=8)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--cap-fraction", type=float, default=1.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    inv = json.loads((AUD / "inventory.json").read_text())["checkpoints"][a.key]
    print(f"extract {a.key} {inv['run']} cf_layers={a.cf_layers} max_rows={a.max_rows}")
    cuda = a.device == "cuda"
    if not guard.require_execution(a.execute, "gpu" if cuda else "cpu-model"):
        return
    if not cuda and not a.max_rows:
        raise guard.GuardError("CPU mode is a smoke test only (--max-rows required)")
    guard.require_disk(1_500_000_000, f"specaudit extract {a.key}")
    mon = None
    if cuda:
        guard.occupancy_gate(f"spec_audit_extract {a.key}")
        dev = guard.setup_cuda_device()
        mon = guard.OccupancyMonitor(10.0)
        mon.start()
    import numpy as np
    import torch
    if not cuda:
        dev = torch.device("cpu")
        torch.set_num_threads(8)

    def sync():
        if cuda:
            torch.cuda.synchronize()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from csr.audit import counterfactual, original_pass, param_fingerprint, token_ce
    from csr.data import TokenSplit, make_batch, pack_rows, to_torch
    from csr.model import CSRModel, ModelConfig
    from csr.segcache import SegTable
    from csr.train_lib import routing_from_cfg
    from csr import specaudit as sa
    name = f"specaudit_v1_extract_{a.key}{a.tag}"
    rd = guard.RunDir(name, {"audit": "specialization-audit-v1", "key": a.key, "checkpoint": inv["checkpoint"],
                             "checkpoint_sha256": inv["checkpoint_sha256"], "cf_layers": a.cf_layers,
                             "rows_per_batch": a.rows_per_batch, "max_rows": a.max_rows, "cap_fraction": a.cap_fraction,
                             "precision": "fp32, TF32 disabled", "prereg_sha256": (guard.ROOT / "docs" / "SPECIALIZATION_AUDIT_V1_PREREG.sha256").read_text().split()[0]})
    guard.register_self(f"spec_audit_extract {a.key}")
    t_iso, t0 = guard.now(), time.time()
    lim = guard.RunLimits(guard.MAX_ATOMIC_SECONDS, monitor=mon)
    status, reason, summ = "INCOMPLETE", "", {}
    try:
        ck = guard.ROOT / inv["checkpoint"]
        if guard.sha256_file(ck) != inv["checkpoint_sha256"]:
            raise guard.GuardError("checkpoint sha256 mismatch")
        sd = torch.load(ck, map_location="cpu", weights_only=True)
        cfg = sd["config"]
        model = CSRModel(ModelConfig.from_dict(cfg["model"]))
        model.load_state_dict(sd["model"], strict=True)
        model.to(dev).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        fp0 = param_fingerprint(model)
        routing = routing_from_cfg(cfg)
        strat = cfg["segmentation"]["strategy"]
        sp = TokenSplit.load(AUD / "data", "audit")
        seg = SegTable.load(AUD / "segments" / SEGFILE[strat], strat) if SEGFILE[strat] else None
        W = np.load(AUD / "data" / "windows.npz")
        wl, woff = window_index(W, sp.offsets)
        NTOK = int(woff[-1])
        n_moe, E, k = len(cfg["model"]["moe_layers"]), cfg["model"]["n_experts"], cfg["model"]["top_k"]
        cf_layers = [int(x) for x in a.cf_layers.split(",")] if a.cf_layers else []
        doc_wins = {}
        for w in range(len(W["doc"])):
            doc_wins.setdefault(int(W["doc"][w]), []).append(w)
        eff_tok = np.zeros(NTOK, bool)
        for w in range(len(W["doc"])):
            if W["half"][w] == 1:
                if a.cap_fraction >= 1.0 or sa.unit_hash(sa.CAP_SALT, f"{sp.doc_ids[int(W['doc'][w])]}{int(W['win_index'][w])}") < a.cap_fraction:
                    eff_tok[woff[w]:woff[w + 1]] = True
        eff_index = np.full(NTOK, -1, np.int64)
        eff_index[eff_tok] = np.arange(eff_tok.sum())
        NEFF = int(eff_tok.sum())
        R = {"ce0": np.full(NTOK, np.nan, np.float32), "topi": np.zeros((n_moe, NTOK, k), np.uint8),
             "gates": np.zeros((n_moe, NTOK, k), np.float16), "constrained": np.zeros((n_moe, NTOK), bool),
             "pool": np.zeros((n_moe, NTOK), np.uint32), "chunk_len": np.zeros(NTOK, np.int16),
             "chunk_idx": np.zeros(NTOK, np.int16), "position": np.zeros(NTOK, np.int16),
             "win_probs": np.zeros((len(W["doc"]), n_moe, E), np.float32)}
        CF = {"dce": np.full((len(cf_layers), k, NEFF), np.nan, np.float32),
              "repl": np.zeros((len(cf_layers), k, NEFF), np.uint8)}
        seen = np.zeros(NTOK, bool)
        WP = torch.zeros(len(W["doc"]), n_moe, E, dtype=torch.float64, device=dev)
        wl_t = torch.as_tensor(wl, dtype=torch.float64, device=dev)
        bits = (2 ** np.arange(E, dtype=np.uint64)).astype(np.uint64)
        timing = {"orig_s": 0.0, "cf_s": {str(L): 0.0 for L in cf_layers}, "rows": {}}
        L_all = sp.lengths()
        for half in (0, 1):
            docs_h = sorted(d for d in doc_wins if W["half"][doc_wins[d][0]] == half)
            rows = pack_rows(L_all, docs_h, 1024)
            if a.max_rows:
                rows = rows[:a.max_rows]
            timing["rows"][str(half)] = len(rows)
            for s0 in range(0, len(rows), a.rows_per_batch):
                rb = rows[s0:s0 + a.rows_per_batch]
                nb = make_batch(rb, sp, 1024, starts_of=seg.starts if seg else None)
                tb = to_torch(nb, dev)
                # query positions per row (mask tokens of this row's documents)
                qp, qg = [], []
                for b, row in enumerate(rb):
                    t = 0
                    pb, gb = [], []
                    for di in row:
                        for w in doc_wins[di]:
                            ps = np.arange(W["start"][w], W["end"][w])
                            pb.append(t + ps)
                            gb.append(woff[w] + ps - W["start"][w])
                        t += int(L_all[di])
                    qp.append(np.concatenate(pb))
                    qg.append(np.concatenate(gb))
                Nq = max(len(x) for x in qp)
                qpos = np.zeros((len(rb), Nq), np.int64)
                qvalid = np.zeros((len(rb), Nq), bool)
                for b in range(len(rb)):
                    qpos[b, :len(qp[b])] = qp[b]
                    qvalid[b, :len(qp[b])] = True
                g = np.concatenate(qg)
                qpos_t = torch.as_tensor(qpos, device=dev)
                qv_t = torch.as_tensor(qvalid, device=dev)
                ar = torch.arange(len(rb), device=dev)[:, None]
                sync()
                t1 = time.time()
                o = original_pass(model, tb, routing, keep_cache=(half == 1 and bool(cf_layers)))
                tgt = tb["targets"][ar, qpos_t]
                ce0 = token_ce(o["logits"][ar, qpos_t].reshape(-1, o["logits"].shape[-1]), tgt.reshape(-1)).view(qpos.shape)
                if not torch.isfinite(ce0[qv_t]).all():
                    raise guard.GuardError("non-finite baseline CE")
                vflat = qv_t.cpu().numpy()
                wid_t = torch.as_tensor(np.searchsorted(woff, g, side="right") - 1, device=dev)
                R["ce0"][g] = ce0.cpu().numpy()[vflat]
                for li, rec in enumerate(o["records"]):
                    R["topi"][li, g] = rec["topi"][ar, qpos_t].cpu().numpy()[vflat].astype(np.uint8)
                    R["gates"][li, g] = rec["gates"][ar, qpos_t].float().cpu().numpy()[vflat].astype(np.float16)
                    R["constrained"][li, g] = rec["constrained"][ar, qpos_t].cpu().numpy()[vflat]
                    if rec["pool_tok"] is not None:
                        pt = rec["pool_tok"][ar, qpos_t].cpu().numpy()[vflat]
                        R["pool"][li, g] = (pt.astype(np.uint64) * bits).sum(1).astype(np.uint32)
                    pr = rec["probs_full"][ar, qpos_t][qv_t]
                    WP[:, li].index_add_(0, wid_t, pr / wl_t[wid_t][:, None])
                ci = nb["chunk_ids"][np.arange(len(rb))[:, None], qpos][vflat]
                cp = nb["chunk_pos"][np.arange(len(rb))[:, None], qpos][vflat]
                bi = np.repeat(np.arange(len(rb)), qvalid.sum(1))
                for b in range(len(rb)):
                    cnt = np.bincount(nb["chunk_ids"][b][nb["chunk_ids"][b] >= 0])
                    sel = bi == b
                    R["chunk_len"][g[sel]] = cnt[ci[sel]]
                rr = np.arange(len(rb))[:, None]
                dstart = qpos - nb["positions"][rr, qpos]
                R["chunk_idx"][g] = (nb["chunk_ids"][rr, qpos] - nb["chunk_ids"][rr, dstart])[vflat]
                R["position"][g] = nb["positions"][np.arange(len(rb))[:, None], qpos][vflat]
                seen[g] = True
                sync()
                timing["orig_s"] += time.time() - t1
                if half == 1 and cf_layers:
                    ge = eff_index[g]
                    keep = ge >= 0
                    qv_cf = qv_t.clone()
                    qv_cf[qv_t] = torch.as_tensor(keep, device=dev)  # row-major order == order of g
                    for ci_, L in enumerate(cf_layers):
                        t2 = time.time()
                        top_L = o["records"][L]["topi"][ar, qpos_t]
                        for j in range(k):
                            cf = counterfactual(model, o, routing, L, qpos_t, qv_cf, top_L[..., j])
                            ce1 = token_ce(cf["logits"].reshape(-1, cf["logits"].shape[-1]), tgt.reshape(-1)).view(qpos.shape)
                            if not torch.isfinite(ce1[qv_cf]).all():
                                raise guard.GuardError("non-finite counterfactual CE")
                            if not bool(cf["affected"][qv_cf].all()):
                                raise guard.GuardError("counterfactual not applied to every query")
                            newt = cf["new_topi"]
                            repl = (newt.unsqueeze(-1) != top_L.unsqueeze(-2)).all(-1)  # new expert not in old
                            assert bool((repl.sum(-1)[qv_cf] == 1).all()), "exactly one replacement expert"
                            rid = (newt * repl).sum(-1)
                            m = qv_cf.cpu().numpy()
                            dce = (ce1 - ce0).cpu().numpy()[m]
                            CF["dce"][ci_, j, ge[keep]] = dce
                            CF["repl"][ci_, j, ge[keep]] = rid.cpu().numpy()[m].astype(np.uint8)
                        sync()
                        timing["cf_s"][str(L)] += time.time() - t2
                    del o
                lim.check()
                if mon is not None and mon.contaminated:
                    raise guard.GuardError(f"CONTAMINATED: {mon.detail}")
        if not a.max_rows:
            if not seen.all():
                raise guard.GuardError(f"incomplete coverage: {int((~seen).sum())} mask tokens unseen")
            if cf_layers and np.isnan(CF["dce"]).any():
                raise guard.GuardError("incomplete counterfactual coverage")
        fp1 = param_fingerprint(model)
        if fp1 != fp0:
            raise guard.GuardError("model parameters changed during audit")
        R["win_probs"] = WP.cpu().numpy().astype(np.float32)
        np.savez(rd.path / "routing.npz", seen=seen, **R)
        if cf_layers:
            np.savez(rd.path / "cf.npz", eff_index=eff_index, layers=np.array(cf_layers), **CF)
        if guard.sha256_file(ck) != inv["checkpoint_sha256"]:
            raise guard.GuardError("checkpoint file changed")
        summ = {"key": a.key, "n_mask_tokens": NTOK, "n_seen": int(seen.sum()), "n_effect_cf_tokens": NEFF,
                "mean_ce0_seen": float(np.nanmean(R["ce0"][seen])), "param_fingerprint": fp0,
                "param_fingerprint_after": fp1, "timing": timing, "seconds": time.time() - t0,
                "peak_alloc_gib": torch.cuda.max_memory_allocated() / 2**30 if cuda else None,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30 if cuda else None, "device": a.device,
                "mean_dce_by_layer_rank": {str(L): [float(np.nanmean(CF["dce"][i, j])) for j in range(k)]
                                           for i, L in enumerate(cf_layers)} if cf_layers else {}}
        guard.atomic_write_json(rd.path / "summary.json", summ)
        status = "COMPLETE"
    except guard.GuardError as e:
        status, reason = ("CONTAMINATED" if "CONTAMINATED" in str(e) else "ABORTED"), str(e)
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        status, reason = "FAILED", f"{type(e).__name__}: {e}"
    finally:
        lim.close()
        if mon is not None:
            mon.stop()
            if mon.contaminated and status == "COMPLETE":
                status, reason = "CONTAMINATED", mon.detail
        rd.set_status(status, reason=reason)
        if cuda:
            guard.gpu_ledger(t_iso, guard.now(), rd.run_id, "spec_audit_extract", status)
        print("STATUS", status, reason, json.dumps({k: v for k, v in summ.items() if k in ("seconds", "timing", "peak_reserved_gib")}))
    sys.exit(0 if status == "COMPLETE" else 1)


if __name__ == "__main__":
    main()
