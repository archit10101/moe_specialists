#!/usr/bin/env python
"""No-training forced-routing diagnostic (task §13).

For fixed held-out receiver chunks (validation split only), force ONLY the receiver chunk's post-warmup positions
into a donor pool at every MoE layer and compare next-token NLL on exactly those positions against: the original
token routing, the receiver's own causal pool, and close / distant / random / popularity-matched-random donors
from other documents. Donors are chosen with frozen prefix-only embeddings u (first w tokens of each chunk).
Donor pools use the same causal per-layer rule (TopM of the router applied to the mean router input of the
donor's first w tokens, computed in the donor's own document context). Pool size m, k, evaluated positions and
model computation are identical across arms. Document-level paired bootstrap (one receiver per document).
Dry-run by default; GPU via scripts/gpu_guarded.sh.
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402

ARMS = ["original", "self", "close", "distant", "random", "popmatched"]


def doc_rows_batch(docs_starts, device):
    """docs_starts: list of (ids array, starts). One document per row, right-padded."""
    import numpy as np
    import torch
    from csr.data import PAD_ID
    from csr.segment import starts_to_arrays
    T = max(len(i) for i, _ in docs_starts)
    B = len(docs_starts)
    a = {k: np.full((B, T), f, np.int64) for k, f in (("input_ids", PAD_ID), ("doc_ids", -1), ("positions", 0),
                                                       ("targets", -100), ("chunk_ids", -1), ("chunk_pos", 0))}
    for b, (ids, st) in enumerate(docs_starts):
        L = len(ids)
        c, p = starts_to_arrays(L, st)
        a["input_ids"][b, :L], a["doc_ids"][b, :L], a["positions"][b, :L] = ids, 0, np.arange(L)
        a["targets"][b, :L - 1] = ids[1:]
        a["chunk_ids"][b, :L], a["chunk_pos"][b, :L] = c, p
    return {k: torch.as_tensor(v, device=device) for k, v in a.items()}


def receiver_window(doc_len, s, ln, w):
    """Input slice, chunk starts and evaluated positions for receiver chunk [s, e) of a document of length doc_len.

    Never synthesizes a token or a chunk boundary beyond the document:
      input = doc[:min(e + 1, doc_len)]   (one following token only if the document has one)
      starts = [0] (+ [s] if s > 0) (+ [e] if e < doc_len)
      evaluated positions = [s + w, min(e, doc_len - 1)), so a document-final chunk drops only its last token,
      which has no next-token target. Every evaluated position's target lies inside the same document.
    Returns (end, starts, lo, hi)."""
    s, ln, doc_len, w = int(s), int(ln), int(doc_len), int(w)
    e = s + ln
    if not (0 <= s < e <= doc_len and ln >= w):
        raise ValueError(f"bad receiver chunk s={s} ln={ln} doc_len={doc_len} w={w}")
    end = min(e + 1, doc_len)
    starts = [0] + ([s] if s > 0 else []) + ([e] if e < doc_len else [])
    lo, hi = s + w, min(e, doc_len - 1)
    assert starts[-1] < end and hi <= end - 1
    return end, starts, lo, hi


def eval_targets(tb, lo, hi):
    """Evaluated positions [lo, hi) and their targets; identical across all rows (probe arms), all valid, and no
    target crosses a document boundary (position t and its target token t + 1 belong to the same document)."""
    import torch
    pos = torch.arange(lo, hi, device=tb["targets"].device)
    tg = tb["targets"][:, pos]
    assert len(pos) > 0 and bool((tg != -100).all()), "evaluated position without a next-token target"
    assert bool((tg == tg[:1]).all()), "evaluated targets differ across probe arms"
    assert hi < tb["input_ids"].shape[1] and bool((tb["doc_ids"][:, pos] == tb["doc_ids"][:, pos + 1]).all()) \
        and bool((tb["doc_ids"][:, pos] >= 0).all()), "target crosses a document boundary"
    assert bool((tg == tb["input_ids"][:, pos + 1]).all())
    return pos, tg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--segmentation", default="semantic", choices=["semantic", "fixed"])
    ap.add_argument("--n-receivers", type=int, default=200)
    ap.add_argument("--min-eval-positions", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    ck = guard.inside_root(a.checkpoint)
    print(f"probe on {ck} receivers={a.n_receivers} seg={a.segmentation} split=val arms={ARMS}")
    if not guard.require_execution(a.execute, "gpu" if a.device == "cuda" else "cpu-model"):
        return
    mon = None
    if a.device == "cuda":
        guard.occupancy_gate("probe")
        dev = guard.setup_cuda_device()
        mon = guard.OccupancyMonitor(10.0)
        mon.start()
    import numpy as np
    import torch
    import torch.nn.functional as F
    from csr.data import TokenSplit
    from csr.model import CSRModel, ModelConfig, RoutingConfig
    from csr.segcache import SegTable
    from csr.segment import chunk_lengths
    if a.device == "cpu":
        dev = torch.device("cpu")
    guard.register_self("probe")
    sd = torch.load(ck, map_location="cpu", weights_only=True)
    cfg = sd["config"]
    label = cfg.get("result_label", "experiment")
    rd = guard.RunDir("probe", {"result_label": label, "args": vars(a), "checkpoint_sha256": guard.sha256_file(ck),
                                "model_cfg": cfg})
    t0iso = guard.now()
    lim = guard.RunLimits(60 * 60, monitor=mon)
    status, reason, out = "INCOMPLETE", "", {}
    try:
        model = CSRModel(ModelConfig.from_dict(cfg["model"]))
        model.load_state_dict(sd["model"])
        model.to(dev).eval()
        w, m = cfg["routing"]["warmup"], cfg["routing"]["pool_size"]
        TOK = RoutingConfig("token", m, w)
        va = TokenSplit.load(guard.inside_root(cfg["data"]["dir"]), "val")
        seg = SegTable.load(guard.inside_root(cfg["segmentation"]["cache"]) / f"val_{a.segmentation}.npz", a.segmentation)
        # ---- chunk catalogue
        cat = []  # (doc, start, length, row in table)
        for di in range(va.n_docs):
            L = len(va.doc(di))
            st = seg.starts(di)
            for j, (s, ln) in enumerate(zip(st, chunk_lengths(L, st))):
                if ln >= w:
                    cat.append((di, s, int(ln), seg.chunk_base(di) + j))
        cat = np.array(cat)
        u = seg.u[cat[:, 3]].astype(np.float32)
        ok = np.linalg.norm(u, axis=1) > 0.5
        cat, u = cat[ok], u[ok]
        # ---- receivers: first chunk per doc with >= min-eval post-warmup positions (index >= 1 preferred)
        recv = []
        seen_docs = set()
        for i, (di, s, ln, _) in enumerate(cat):
            if di in seen_docs or ln < w + a.min_eval_positions + 1:
                continue
            _, _, lo_, hi_ = receiver_window(len(va.doc(di)), s, ln, w)
            if hi_ - lo_ < a.min_eval_positions:
                continue
            recv.append(i)
            seen_docs.add(di)
            if len(recv) >= a.n_receivers:
                break
        rng = np.random.default_rng(a.seed)
        # ---- donor pools for all candidate chunks that may be used (computed lazily)
        pool_cache = {}

        def donor_pools(idx_list):
            need = [i for i in idx_list if i not in pool_cache]
            for k0 in range(0, len(need), 16):
                grp = need[k0:k0 + 16]
                items = []
                for i in grp:
                    di, s, ln, _ = cat[i]
                    ids = va.doc(di)[:s + w]
                    items.append((ids, [0] + ([int(s)] if s > 0 else [])))
                tb = doc_rows_batch(items, dev)
                with torch.no_grad():
                    o = model(tb, TOK, need_chunk_logits=True)
                lay = o.layout
                for b, i in enumerate(grp):
                    s = int(cat[i][1])
                    r = int(((lay.rows_b == b) & (lay.rows_t0 == s)).nonzero()[0])
                    pool_cache[i] = torch.stack([rr["chunk_logits"][r].topk(m).indices for rr in o.routing])  # (L, m)
            return [pool_cache[i] for i in idx_list]

        # ---- choose donors
        plan = []
        for ri in recv:
            di = cat[ri][0]
            cand = np.nonzero(cat[:, 0] != di)[0]
            sims = u[cand] @ u[ri]
            plan.append({"recv": int(ri), "close": int(cand[np.argmax(sims)]), "distant": int(cand[np.argmin(sims)]),
                         "random": int(rng.choice(cand)), "sim_close": float(sims.max()), "sim_distant": float(sims.min()),
                         "cand": cand})
        # popularity from original routing usage over receivers' documents (first pass)
        E = model.cfg.n_experts
        usage = np.zeros((len(model.cfg.moe_layers), E))
        for p in plan:
            di, s, ln, _ = cat[p["recv"]]
            end, st, _, _ = receiver_window(len(va.doc(di)), s, ln, w)
            tb = doc_rows_batch([(va.doc(di)[:end], st)], dev)
            with torch.no_grad():
                o = model(tb, TOK)
            for li, rr in enumerate(o.routing):
                usage[li] += np.bincount(rr["topi"][0].reshape(-1).cpu().numpy(), minlength=E)
        usage /= usage.sum(1, keepdims=True)

        def popularity(pools):  # mean over layers of summed usage share of pool experts
            return float(np.mean([usage[li, pools[li].cpu().numpy()].sum() for li in range(len(pools))]))

        # popularity-matched random donor: same popularity quintile as the close donor, from a random candidate set
        sample_c = rng.choice(len(cat), size=min(len(cat), 600), replace=False)
        sample_pop = np.array([popularity(pp) for pp in donor_pools(list(map(int, sample_c)))])
        qs = np.quantile(sample_pop, [0.2, 0.4, 0.6, 0.8])
        sample_bin = np.searchsorted(qs, sample_pop)
        res_rows = []
        for p in plan:
            lim.check()
            ri = p["recv"]
            di, s, ln, _ = cat[ri]
            close_pop = popularity(donor_pools([p["close"]])[0])
            qbin = int(np.searchsorted(qs, close_pop))
            same_bin = [int(c) for c, cb in zip(sample_c, sample_bin) if cb == qbin and cat[c][0] != di]
            p["popmatched"] = int(rng.choice(same_bin)) if same_bin else p["random"]
            doc_len = len(va.doc(di))
            end, st, lo, hi = receiver_window(doc_len, s, ln, w)
            assert hi - lo >= a.min_eval_positions
            tb = doc_rows_batch([(va.doc(di)[:end], st)] * len(ARMS), dev)
            with torch.no_grad():
                base = model(tb, TOK, need_chunk_logits=True)
            lay = base.layout
            R = lay.rows_b.numel()
            n_layers = len(base.routing)
            rec_rows = [int(((lay.rows_b == b) & (lay.rows_t0 == s)).nonzero()[0]) for b in range(len(ARMS))]
            own = [base.routing[l]["chunk_logits"][rec_rows[0]].topk(m).indices for l in range(n_layers)]
            dpools = dict(zip(["close", "distant", "random", "popmatched"],
                              donor_pools([p["close"], p["distant"], p["random"], p["popmatched"]])))
            override = [torch.ones(R, E, dtype=torch.bool, device=dev) for _ in range(n_layers)]
            constrain = torch.zeros(R, dtype=torch.bool, device=dev)
            for b, arm in enumerate(ARMS):
                if arm == "original":
                    continue
                r = rec_rows[b]
                constrain[r] = True
                for l in range(n_layers):
                    pool = own[l] if arm == "self" else dpools[arm][l]
                    override[l][r] = False
                    override[l][r, pool] = True
            with torch.no_grad():
                o = model(tb, TOK, pool_override=override, constrain_rows=constrain)
            pos, tg = eval_targets(tb, lo, hi)
            nll = F.cross_entropy(o.logits[:, pos].float().reshape(-1, o.logits.shape[-1]), tg.reshape(-1),
                                  reduction="none").view(len(ARMS), -1)
            # forced arms constrain exactly the receiver chunk's post-warmup tokens [s+w, e) (the same set in every
            # arm); the evaluated positions are those with a target, [s+w, min(e, doc_len-1)), a subset of them
            con = o.routing[0]["constrained"]
            n_con = int(s + ln) - int(s + w)
            assert int(con[0].sum()) == 0 and all(int(con[b].sum()) == n_con for b in range(1, len(ARMS)))
            assert all(bool(con[b, pos].all()) for b in range(1, len(ARMS)))
            assert len(pos) == n_con - (1 if s + ln == doc_len else 0)
            for b in range(1, len(ARMS)):
                for l in range(n_layers):
                    pool = own[l] if ARMS[b] == "self" else dpools[ARMS[b]][l]
                    sel = o.routing[l]["topi"][b, pos]
                    assert torch.isin(sel, pool).all(), "selected expert outside forced pool"
            row = {"doc": int(di), "start": int(s), "len": int(ln), "n_pos": int(len(pos)), "doc_len": doc_len,
                   "doc_final": bool(s + ln == doc_len), "eval_lo": int(lo), "eval_hi": int(hi),
                   "sim_close": p["sim_close"], "sim_distant": p["sim_distant"], "pop_close": close_pop}
            for b, arm in enumerate(ARMS):
                row[f"nll_{arm}"] = float(nll[b].double().sum())
            res_rows.append(row)
        # ---- statistics
        n = np.array([r["n_pos"] for r in res_rows], float)
        rng2 = np.random.default_rng(1)
        boots = rng2.integers(0, len(res_rows), size=(2000, len(res_rows)))

        def contrast(x, y):
            dx = np.array([r[f"nll_{x}"] - r[f"nll_{y}"] for r in res_rows])
            pt = dx.sum() / n.sum()
            bs = dx[boots].sum(1) / n[boots].sum(1)
            per = dx / n
            return {"mean_delta_nats_per_token": float(pt), "ci95": np.percentile(bs, [2.5, 97.5]).tolist(),
                    "per_receiver_quantiles_10_50_90": np.percentile(per, [10, 50, 90]).tolist()}
        out = {"n_receivers": len(res_rows), "n_positions": int(n.sum()), "warmup": w, "pool_size": m,
               "orig_nll_per_token": float(sum(r["nll_original"] for r in res_rows) / n.sum()),
               "delta_vs_original": {arm: contrast(arm, "original") for arm in ARMS[1:]},
               "close_minus_random": contrast("close", "random"), "close_minus_distant": contrast("close", "distant"),
               "close_minus_popmatched": contrast("close", "popmatched"), "self_minus_close": contrast("self", "close"),
               "mean_sim_close": float(np.mean([r["sim_close"] for r in res_rows])),
               "mean_sim_distant": float(np.mean([r["sim_distant"] for r in res_rows]))}
        guard.atomic_write_json(rd.path / "probe_rows.json", res_rows)
        status = "COMPLETE"
    except guard.GuardError as e:
        status, reason = ("CONTAMINATED" if "CONTAMINATED" in str(e) else "ABORTED"), str(e)
    except Exception as e:  # noqa: BLE001
        status, reason = "FAILED", f"{type(e).__name__}: {e}"
        rd.log(traceback.format_exc())
    finally:
        lim.close()
        if mon is not None:
            mon.stop()
        pa, pr = lim.peak_gib()
        guard.atomic_write_json(rd.path / "summary.json", {"status": status, "reason": reason, "result_label": label, "peak_alloc_gib": pa,
                                                           "peak_reserved_gib": pr, **out})
        rd.set_status(status, reason=reason)
        if a.device == "cuda":
            guard.gpu_ledger(t0iso, guard.now(), rd.run_id, "probe", status)
        print(json.dumps(out, indent=1))
        print("STATUS", status, reason)
    return status


def exit_code(status):
    """0 for a dry-run (None) or COMPLETE; nonzero for FAILED / ABORTED / CONTAMINATED / anything else."""
    return 0 if status in (None, "COMPLETE") else 1


if __name__ == "__main__":
    sys.exit(exit_code(main()))
