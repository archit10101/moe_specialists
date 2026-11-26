#!/usr/bin/env python
"""csr-latent-forecast-router-distill-v1: build the compressed causal forecast cache (protocol section 5).

  lfr_cache.py --region main --steps K   (documents of the first K optimizer steps of the balanced FINAL plan)
  lfr_cache.py --region calib_search     (CALIB-SEARCH diagnostic set: paper half A, 0.1M tokens / domain)

Chunks of documents (~16M tokens); per chunk the six frozen drafts run ONE AT A TIME (load -> all chunk documents ->
unload); per-draft outputs stay in host memory; then the ensemble compression is written atomically with a COMPLETE
marker. Completed chunks are skipped on restart (chunk cursor). An external stop discards only the chunk in progress.
GPU only via scripts/gpu_guarded.sh (--device cuda --execute). Exit 0 COMPLETE, 77 EXTERNALLY_STOPPED, 78 BLOCKED.
"""
import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402

CHUNK_TOKENS = 16_000_000
STOP = {"reason": None}


def _sig(signum, frame):
    if STOP["reason"] is None:
        STOP["reason"] = "signal " + signal.Signals(signum).name


def load_draft(slot, device):
    import torch
    from csr.lfr_lib import DRAFT_PATHS, build_draft
    m = build_draft(0)
    sd = torch.load(guard.ROOT / DRAFT_PATHS[slot], map_location="cpu", weights_only=True)
    m.load_state_dict(sd["model"])
    m = m.to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def run_draft(draft, sp, docs, device, batch_rows=32, diag_topk=0, stop_check=None):
    """Per-token outputs (all positions of each document, concatenated in `docs` order) as host tensors."""
    import torch
    from csr.lfr_lib import draft_outputs, T_SEQ, PAD_ID
    keys = ["top_ids", "top_p", "ent1", "score", "roll4", "roll8"] + (["roll", "roll_topk"] if diag_topk else [])
    acc = {k: [] for k in keys}
    for i in range(0, len(docs), batch_rows):
        if stop_check is not None and stop_check():
            return None
        dd = docs[i:i + batch_rows]
        L = [int(sp.lengths[d]) for d in dd]
        ids = torch.full((len(dd), T_SEQ), PAD_ID, dtype=torch.long)
        for b, d in enumerate(dd):
            ids[b, :L[b]] = torch.as_tensor(sp.doc(d))
        ids = ids.to(device)
        o = draft_outputs(draft, ids, torch.as_tensor(L, device=device), diag_topk=diag_topk)
        oc = {"top_ids": o["top_ids"].to(torch.int32).cpu(), "top_p": o["top_p"].to(torch.float16).cpu(),
              "ent1": o["ent1"].float().cpu(), "score": o["score"].float().cpu(),
              "roll4": o["roll"][..., 3].to(torch.int32).cpu(), "roll8": o["roll"][..., 7].to(torch.int32).cpu()}
        if diag_topk:
            oc["roll"] = o["roll"].to(torch.int32).cpu()
            oc["roll_topk"] = o["roll_topk"].to(torch.int32).cpu()
        for b in range(len(dd)):
            for k in keys:
                acc[k].append(oc[k][b, :L[b]])
    return {k: torch.cat(v) for k, v in acc.items()}


def compress_all(per, device, step=262_144):
    """per: list of 6 draft output dicts (host). Returns compressed numpy arrays and coverage stats."""
    import numpy as np
    import torch
    from csr.lfr_lib import compress_ensemble
    N = per[0]["ent1"].shape[0]
    outs, stats = [], []
    for s in range(0, N, step):
        g = lambda k, dt=None: torch.stack([p[k][s:s + step] for p in per], 1).to(device)
        o, st = compress_ensemble(g("top_ids").long(), g("top_p").float(), g("ent1"), g("score"),
                                  g("roll4").long(), g("roll8").long())
        outs.append(o)
        stats.append(st)
    arrays = {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}
    w = np.asarray([s["n"] for s in stats], dtype=np.float64)
    agg = {k: float(np.average([s[k] for s in stats], weights=w)) for k in stats[0] if k not in ("n",)}
    agg["ens_mass_lower_bound_gap_max"] = float(max(s["ens_mass_lower_bound_gap_max"] for s in stats))
    agg["n"] = int(N)
    return arrays, agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", choices=["main", "calib_search"], required=True)
    ap.add_argument("--steps", type=int, help="main region: documents of the first STEPS steps of the FINAL plan")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    print(json.dumps(vars(a)))
    if not guard.require_execution(a.execute, "gpu" if a.device == "cuda" else "cpu-model"):
        return
    import numpy as np
    import torch
    from csr.lfr_lib import ART, BUDGET, CACHE_DIR, main_stream, save_cache_chunk, protocol_sha
    from csr.fsp_lib import Split
    for s_ in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(s_, _sig)
    mon = None
    if a.device == "cuda":
        guard.occupancy_gate("lfr cache")
        device = guard.setup_cuda_device()
        mon = guard.OccupancyMonitor(interval=15.0)
        mon.start()
    else:
        device = torch.device("cpu")
    guard.register_self(f"lfr cache {a.region} on {a.device}")
    from csr.lfr_lib import STOP_MARKER

    def stop_check():
        if STOP["reason"] is None and STOP_MARKER.exists():
            STOP["reason"] = "stop marker STOP"
        if STOP["reason"] is None and mon is not None and mon.contaminated:
            STOP["reason"] = f"foreign GPU process: {mon.detail}"
        return STOP["reason"] is not None
    t_iso, t0 = guard.now(), time.time()
    if a.region == "main":
        sp, steps, _ = main_stream(BUDGET["FINAL"])
        docs = [d for r in steps[:a.steps] for row in r for d in row]
    else:
        from csr.lfr_diag import diag_docs
        sp = Split("calib_search")
        docs = diag_docs()
    # existing COMPLETE chunks (an earlier, shorter prefix plan) are kept unchanged when their documents form a
    # prefix of this plan's document list; the remaining documents are chunked from the next index (deviation 6:
    # the reader indexes documents through each chunk's own metadata, so the layout does not affect any value)
    root0 = CACHE_DIR / a.region
    pos, ci0, kept = 0, 0, []
    while (root0 / f"chunk_{ci0:04d}" / "COMPLETE.json").exists():
        md = json.loads((root0 / f"chunk_{ci0:04d}" / "COMPLETE.json").read_text())["docs"]
        if [int(d) for d in docs[pos:pos + len(md)]] != md:
            raise guard.GuardError(f"chunk_{ci0:04d}: existing chunk is not a prefix of this plan's documents")
        kept.append([int(d) for d in md])
        pos += len(md)
        ci0 += 1
    docs = docs[pos:]
    # deterministic chunking by cumulative tokens
    chunks, cur, ntok = list(kept), [], 0
    for d in docs:
        cur.append(d)
        ntok += int(sp.lengths[d])
        if ntok >= CHUNK_TOKENS:
            chunks.append(cur)
            cur, ntok = [], 0
    if cur:
        chunks.append(cur)
    root = CACHE_DIR / a.region
    root.mkdir(parents=True, exist_ok=True)
    status, rc, done_tok, draft_s = "COMPLETE", 0, 0, 0.0
    try:
        for ci, cd in enumerate(chunks):
            cdir = root / f"chunk_{ci:04d}"
            if (cdir / "COMPLETE.json").exists():
                meta = json.loads((cdir / "COMPLETE.json").read_text())
                if meta["docs"] != [int(d) for d in cd]:
                    raise guard.GuardError(f"{cdir}: existing chunk holds different documents")
                continue
            ntok = int(sum(int(sp.lengths[d]) for d in cd))
            guard.require_disk(ntok * 48, f"lfr cache chunk {ci}")
            per = []
            ts = time.time()
            for slot in range(6):  # ONE draft resident at a time
                dr = load_draft(slot, device)
                o = run_draft(dr, sp, cd, device, stop_check=stop_check)
                del dr
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                if o is None:
                    raise KeyboardInterrupt(STOP["reason"])
                per.append(o)
            draft_s += time.time() - ts
            arrays, stats = compress_all(per, device)
            del per
            offs = np.concatenate([[0], np.cumsum([int(sp.lengths[d]) for d in cd])[:-1]])
            meta = save_cache_chunk(cdir, arrays, cd, offs, {"region": a.region, "chunk": ci, "stats": stats,
                                                             "draft_seconds": time.time() - ts,
                                                             "protocol_sha256": protocol_sha()})
            done_tok += ntok
            print(json.dumps({"chunk": ci, "of": len(chunks), "tokens": ntok, "seconds": time.time() - ts,
                              "tok_per_s_6drafts": ntok / (time.time() - ts), "stats": stats}), flush=True)
            if stop_check():
                raise KeyboardInterrupt(STOP["reason"])
    except KeyboardInterrupt as e:
        status, rc = "EXTERNALLY_STOPPED", 77
        print(f"EXTERNALLY_STOPPED: {e}", flush=True)
    finally:
        if mon is not None:
            mon.stop()
    # manifest of complete chunks
    man = {"region": a.region, "chunks": [], "time": guard.now()}
    for cdir in sorted(root.glob("chunk_*")):
        if (cdir / "COMPLETE.json").exists():
            m_ = json.loads((cdir / "COMPLETE.json").read_text())
            man["chunks"].append({"chunk": cdir.name, "n_docs": len(m_["docs"]), "n_tokens": m_["n_tokens"],
                                  "sha256": m_["sha256"], "stats": m_["stats"]})
    man["n_tokens"] = int(sum(c["n_tokens"] for c in man["chunks"]))
    man["bytes"] = int(sum(f.stat().st_size for f in root.rglob("*.npy")))
    man["complete"] = status == "COMPLETE" and len(man["chunks"]) == len(chunks)
    guard.atomic_write_json(ART / f"cache_manifest_{a.region}.json", man, overwrite=True)
    if a.device == "cuda":
        guard.gpu_ledger(t_iso, guard.now(), f"lfr_cache_{a.region}", "lfr cache", status)
        with open(ART / "compute_ledger.tsv", "a") as f:
            f.write(f"{t_iso}\t{guard.now()}\t{time.time() - t0:.1f}\tlfr_cache_{a.region}\tcache_{a.region}\t"
                    f"forecast_cache\t{status}\n")
    print(json.dumps({"status": status, "tokens_done_this_leg": done_tok, "seconds": time.time() - t0}))
    sys.exit(rc)


if __name__ == "__main__":
    main()
