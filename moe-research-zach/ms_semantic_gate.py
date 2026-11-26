#!/usr/bin/env python
"""csr-multiscale-routing-v1 semantic-boundary causality gate (docs/MULTISCALE_ROUTING_V1_PREREG.md §5, G1-G6).

GPU 0 (guarded launcher): pinned MiniLM teacher (fp32, as the cache was built) and a full-size random-init
multiscale model (fp32, TF32 off). Reads train/calib/val documents and the frozen semantic/fixed caches only.
Writes artifacts/multiscale-routing-v1/semantic_gate.json (never overwritten). Exit 0 iff the gate passes.
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402

OUT = guard.ROOT / "artifacts" / "multiscale-routing-v1" / f"semantic_gate_{guard.utc_stamp()}.json"
CACHE = guard.ROOT / "artifacts" / "segments" / "8e7edc7e14b3c880"
DATA = guard.ROOT / "artifacts" / "data" / "fwe-rep-v1"
TIE = 1e-5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-split", type=int, default=200)
    ap.add_argument("--n-perturb", type=int, default=100)
    ap.add_argument("--n-model", type=int, default=20)
    ap.add_argument("--n-incremental", type=int, default=10)
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    if OUT.exists():
        raise guard.GuardError(f"{OUT} exists")
    print("msv1 semantic-causality gate G1-G6 (GPU)")
    if not guard.require_execution(a.execute, "gpu"):
        return
    guard.occupancy_gate("msv1 semantic gate")
    dev = guard.setup_cuda_device()
    mon = guard.OccupancyMonitor(10.0)
    mon.start()
    guard.register_self("msv1 semantic gate")
    t0iso, t0 = guard.now(), time.time()
    lim = guard.RunLimits(45 * 60, monitor=mon)
    status, reason, res = "INCOMPLETE", "", {}
    try:
        res = run(a, dev, lim)
        status = "COMPLETE"
    except guard.GuardError as e:
        status, reason = ("CONTAMINATED" if "CONTAMINATED" in str(e) else "ABORTED"), str(e)
    except Exception as e:  # noqa: BLE001
        status, reason = "FAILED", f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        lim.close()
        mon.stop()
        if mon.contaminated and status == "COMPLETE":
            status, reason = "CONTAMINATED", mon.detail
        res.update(status=status, reason=reason, seconds=time.time() - t0, created=guard.now())
        guard.atomic_write_json(OUT, res, mode=0o400)
        guard.gpu_ledger(t0iso, guard.now(), "msv1_semantic_gate", "msv1 semantic gate", status)
        guard.ledger("MS_V1_SEMANTIC_GATE", f"status={status} pass={res.get('gate_pass')} {reason}")
        print(json.dumps({k: v for k, v in res.items() if k not in ("details",)}, indent=1, default=str)[:4000])
    sys.exit(0 if status == "COMPLETE" and res.get("gate_pass") else 1)


def replay_margins(ends, embs, L, p):
    """Every eligible semantic decision (n > 0, cur_len >= min_len) and its |sim - tau|; same state machine as
    segment._semantic_update (used only to classify mismatches as numerical ties)."""
    import numpy as np
    a, rep, n = 0, None, 0
    out = []
    for t_end, e in zip(np.asarray(ends).tolist(), embs):
        while t_end + 1 - a > p.max_len:
            a += p.max_len
            rep, n = None, 0
        if t_end + 1 >= L:
            break
        cur_len = t_end + 1 - a
        if n > 0 and cur_len >= p.min_len:
            r = rep / max(np.linalg.norm(rep), 1e-12)
            sim = float(np.dot(e, r))
            out.append((t_end, abs(sim - p.tau)))
            if sim < p.tau:
                a, rep, n = t_end + 1, np.array(e, dtype=np.float64), 1
                continue
        rep = np.array(e, dtype=np.float64) if rep is None else rep + e
        n += 1
    return out


def run(a, dev, lim):
    import numpy as np
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from csr.artifacts import FrozenEncoder, load_lm_tokenizer
    from csr.multiscale import CONDITIONS, SCHEDULES, MSRouting, MultiscaleModel, StartsTable, load_split
    from csr.model import ModelConfig
    from csr.segcache import make_detector
    from csr.segment import OnlineSemanticChunker, SemanticParams, fixed_starts, sentence_spans, starts_to_arrays

    man = json.loads((CACHE / "manifest.json").read_text())
    p = SemanticParams(float(man["tau"]), int(man["spec"]["min_len"]), int(man["spec"]["max_len"]))
    tok = load_lm_tokenizer()
    det = make_detector(tok)
    enc = FrozenEncoder("cuda:0", batch_size=512)
    ck0 = enc.param_checksum()
    n_embed_calls = [0]

    def embed_fn(list_of_ids):
        n_embed_calls[0] += 1
        return enc.embed([tok.decode(list(map(int, x))).strip() for x in list_of_ids])

    def online(ids, L=None):
        ch = OnlineSemanticChunker(det, embed_fn, p, L=L)
        announced = {}
        for t, x in enumerate(ids.tolist()):
            if ch.push(int(x)):
                announced[t + 1] = t
        starts = [s for s in ch.starts if s < len(ids)]
        return starts, announced, ch

    rng = np.random.default_rng(20261102)
    res = {"tau": p.tau, "min_len": p.min_len, "max_len": p.max_len, "details": {}}
    # ---- fixed cache == fixed_starts(L,128) for every document of train/calib/val (CPU)
    fixed_ok = {}
    for split in ("train", "calib", "val"):
        sp = load_split(DATA, split)
        z = np.load(CACHE / f"{split}_fixed.npz")
        tab = StartsTable(np.asarray(z["starts_flat"]), np.asarray(z["starts_off"]))
        L = sp.lengths()
        fixed_ok[split] = all(tab.starts(i) == fixed_starts(int(L[i]), 128) for i in range(sp.n_docs))
    res["fixed_cache_equals_fixed_starts"] = fixed_ok

    # ---- G1 online reproduction (streamed with L=None: no knowledge of document length) + G2 availability
    g1, g2 = {}, {}
    samples = {}
    for split in ("train", "calib", "val"):
        sp = load_split(DATA, split)
        z = np.load(CACHE / f"{split}_semantic.npz")
        tab = StartsTable(np.asarray(z["starts_flat"]), np.asarray(z["starts_off"]))
        idx = sorted(rng.choice(sp.n_docs, size=min(a.n_per_split, sp.n_docs), replace=False).tolist())
        samples[split] = idx
        mism, ties, n_b, avail_bad = [], [], 0, []
        for di in idx:
            ids = sp.doc(di)
            L = len(ids)
            starts, ann, ch = online(ids, None)
            cached = tab.starts(di)
            n_b += len(cached) - 1
            if starts != cached:
                ends = det.sentence_ends(ids)
                embs = embed_fn([ids[s:e + 1] for s, e in sentence_spans(ends)])
                first = next(i for i in range(min(len(starts), len(cached)) + 1)
                             if i >= len(starts) or i >= len(cached) or starts[i] != cached[i])
                lim_pos = min(starts[first] if first < len(starts) else L, cached[first] if first < len(cached) else L)
                margins = [m for t_end, m in replay_margins(ends, embs, L, p) if t_end + 1 <= lim_pos]
                mn = min(margins) if margins else None
                rec = {"doc": int(di), "first_diff_index": first, "online": starts[:first + 2],
                       "cached": cached[:first + 2], "min_margin_upto_diff": mn}
                (ties if (mn is not None and mn < TIE) else mism).append(rec)
            ends = set(det.sentence_ends(ids).tolist())
            for b in starts[1:]:
                is_forced = (b - max(s for s in starts if s < b)) == p.max_len
                if ann.get(b) != b - 1 or not ((b - 1) in ends or is_forced):
                    avail_bad.append({"doc": int(di), "boundary": int(b), "announced_at": ann.get(b)})
            lim.check()
        g1[split] = {"n_docs": len(idx), "n_cached_boundaries": n_b, "mismatched_docs": len(mism),
                     "numerical_tie_docs": len(ties), "pass": len(mism) == 0}
        res["details"][f"G1_{split}"] = {"mismatches": mism, "ties": ties}
        g2[split] = {"violations": len(avail_bad), "pass": not avail_bad}
        res["details"][f"G2_{split}"] = avail_bad[:50]
        print(split, g1[split], g2[split], flush=True)
    res["G1"], res["G2"] = g1, g2

    # ---- G3 future perturbation of text (calib)
    sp = load_split(DATA, "calib")
    cand = [di for di in samples["calib"] if len(det.sentence_ends(sp.doc(di))) >= 4]
    pert = []
    g3_bad = []
    for j, di in enumerate(cand[:a.n_perturb]):
        ids = sp.doc(di)
        L = len(ids)
        ends = det.sentence_ends(ids)
        c = int(ends[len(ends) // 2])
        if c + 2 >= L:
            continue
        other = sp.doc(cand[(j + 1) % len(cand)])
        rep = np.resize(other, L - c - 1)
        ids2 = np.concatenate([ids[:c + 1], rep]).astype(np.int64)
        s1, _, _ = online(ids, None)
        s2, _, _ = online(ids2, None)
        keep1 = [s for s in s1 if s <= c + 1]
        keep2 = [s for s in s2 if s <= c + 1]
        if keep1 != keep2:
            g3_bad.append({"doc": int(di), "cut": c, "orig": keep1, "perturbed": keep2})
        pert.append((ids, ids2, c, s1, s2))
        lim.check()
    res["G3"] = {"n_docs": len(pert), "violations": len(g3_bad), "pass": not g3_bad and len(pert) >= 50}
    res["details"]["G3"] = g3_bad
    assert enc.param_checksum() == ck0, "teacher encoder changed"
    print("G3", res["G3"], flush=True)

    # ---- G4 model invariance, G6 future chunk structure, G5 incremental decoding (full-size, fp32)
    base = json.loads((guard.ROOT / "configs/rep-v1/seed0/condition_A.json").read_text())
    torch.manual_seed(0)
    model = MultiscaleModel(ModelConfig.from_dict(base["model"])).to(dev).float().eval()
    r = MSRouting(alpha=1.0, beta=1.0, schedule=SCHEDULES["HIER"], scale_control=False).validate()

    def batch(ids, starts):
        L = len(ids)
        cid, cpos = starts_to_arrays(L, starts)
        tt = lambda x: torch.as_tensor(np.asarray(x, dtype=np.int64), device=dev).unsqueeze(0)
        return {"input_ids": tt(ids), "doc_ids": tt(np.zeros(L)), "positions": tt(np.arange(L)),
                "chunk_ids": tt(cid), "chunk_pos": tt(cpos)}

    g4 = {"n": 0, "max_abs_logit_diff": 0.0, "max_abs_zroute_diff": 0.0, "topk_mismatch": 0}
    g6 = {"n": 0, "max_abs_logit_diff": 0.0, "topk_mismatch": 0}
    with torch.no_grad():
        for ids, ids2, c, s1, s2 in pert[:a.n_model]:
            o1 = model(batch(ids, s1), r, collect=True)
            o2 = model(batch(ids2, s2), r, collect=True)
            g4["n"] += 1
            g4["max_abs_logit_diff"] = max(g4["max_abs_logit_diff"],
                                           float((o1.logits[0, :c + 1] - o2.logits[0, :c + 1]).abs().max()))
            for q1, q2 in zip(o1.routing, o2.routing):
                g4["max_abs_zroute_diff"] = max(g4["max_abs_zroute_diff"],
                                                float((q1["z_route"][0, :c + 1] - q2["z_route"][0, :c + 1]).abs().max()))
                g4["topk_mismatch"] += int((q1["topi"][0, :c + 1] != q2["topi"][0, :c + 1]).any(-1).sum())
            # G6: identical text, later chunk structure altered (boundaries after c+1 shifted/removed)
            later = [s for s in s1 if s > c + 1]
            alt = [s for s in s1 if s <= c + 1] + sorted({min(s + 7, len(ids) - 1) for s in later[::2]} - {0})
            alt = sorted(set(alt))
            o3 = model(batch(ids, alt), r)
            g6["n"] += 1
            g6["max_abs_logit_diff"] = max(g6["max_abs_logit_diff"],
                                           float((o1.logits[0, :c + 2] - o3.logits[0, :c + 2]).abs().max()))
            lim.check()
    g4["pass"] = g4["max_abs_logit_diff"] <= 1e-5 and g4["max_abs_zroute_diff"] <= 1e-5 and g4["topk_mismatch"] == 0
    g6["pass"] = g6["max_abs_logit_diff"] <= 1e-5
    res["G4"], res["G6"] = g4, g6
    print("G4", g4, "G6", g6, flush=True)

    g5 = {"n_docs": 0, "n_tokens": 0, "max_abs_logit_diff": 0.0, "max_abs_zroute_diff": 0.0, "topk_mismatch": 0,
          "boundary_mismatch_docs": 0}
    with torch.no_grad():
        for ids, _, _, s1, _ in pert[:a.n_incremental]:
            L = len(ids)
            full = model(batch(ids, s1), r, collect=True)
            ch = OnlineSemanticChunker(det, embed_fn, p, L=None)
            st = model.new_ms_state()
            cur = 0
            for t in range(L):
                cpos = t - cur  # chunk start known from tokens < t only
                lg, rt = model.ms_step(int(ids[t]), t, cpos, r, st)
                g5["max_abs_logit_diff"] = max(g5["max_abs_logit_diff"], float((lg - full.logits[0, t]).abs().max()))
                for li, (topi, zr) in enumerate(rt):
                    g5["topk_mismatch"] += int(not torch.equal(topi, full.routing[li]["topi"][0, t]))
                    g5["max_abs_zroute_diff"] = max(g5["max_abs_zroute_diff"],
                                                    float((zr - full.routing[li]["z_route"][0, t]).abs().max()))
                if ch.push(int(ids[t])):
                    cur = t + 1
            if [s for s in ch.starts if s < L] != s1:
                g5["boundary_mismatch_docs"] += 1
            g5["n_docs"] += 1
            g5["n_tokens"] += L
            lim.check()
    g5["pass"] = g5["max_abs_logit_diff"] <= 1e-4 and g5["topk_mismatch"] == 0 and g5["boundary_mismatch_docs"] == 0
    res["G5"] = g5
    print("G5", g5, flush=True)
    res["embed_calls"] = n_embed_calls[0]
    res["gate_pass"] = bool(all(v["pass"] for v in g1.values()) and all(v["pass"] for v in g2.values())
                            and res["G3"]["pass"] and g4["pass"] and g5["pass"] and g6["pass"]
                            and all(fixed_ok.values()))
    res["note"] = ("Online chunker streamed with L=None (no document-length knowledge); boundaries at L dropped. "
                   "Numerical ties (|sim - tau| < 1e-5 before the first differing boundary) are listed, not counted "
                   "as failures.")
    return res


if __name__ == "__main__":
    main()
