#!/usr/bin/env python
"""csr-forecast-latent-edl-v1: causal forecast cache (prereg section 3). For every valid region forecast point of the
selected documents, each approved generalist draft generates a 32-token continuation from the identical causal prefix;
the continuations (full and first-16-token) and the last 192 prefix tokens (past-only representation) are embedded by
frozen MiniLM. Chunked, atomic per chunk (sha256), resumable (completed chunks are skipped after verification), and the
manifest becomes immutable (0400) when complete.

  --name search_main : documents of the first S50 steps of the FINAL plan (points every 128)
  --name final_main  : documents of the FINAL plan not in search_main (points every --L), built after the freeze
  --name calib_search: CALIB-SEARCH audit halves A and B (0.2M tokens / domain each)
  --name calib_sel   : CALIB-SELECTION audit subset (requires STAGE3_OPEN.json; analysis only)
  --name val_audit   : VAL audit subset (requires frozen.json; analysis only, never a training input)
GPU only via scripts/gpu_guarded.sh ... --device cuda --execute."""
import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402

CHUNK_DOCS = 16384
AUDIT_TOK = 200_000


def doc_set(name):
    import numpy as np
    from csr.fsp_lib import Split
    from csr.fle_paths import ART, FINAL_BUDGET, STAGE_TOKENS
    from csr.lfr_lib import main_stream
    from csr.lfr_eval import eval_rows, paper_half
    if name in ("search_main", "final_main"):
        sp, steps, _ = main_stream(FINAL_BUDGET)
        s50 = stage_steps(sp, steps)["s50"]
        first = [d for st in steps[:s50] for r in st for d in r]
        if name == "search_main":
            return sp, first
        seen = set(first)
        return sp, [d for st in steps for r in st for d in r if d not in seen]
    if name == "calib_search":
        sp = Split("calib_search")
        docs = []
        for h in (0, 1):
            docs += [d for r in eval_rows(sp, AUDIT_TOK, paper_half(sp, h)) for d in r]
        return sp, docs
    if name == "calib_sel":
        if not (ART / "STAGE3_OPEN.json").exists():
            raise guard.GuardError("CALIB-SELECTION opens only at stage 3")
        sp = Split("calib_sel")
        return sp, [d for r in eval_rows(sp, AUDIT_TOK) for d in r]
    if name == "val_audit":
        if not (ART / "frozen.json").exists():
            raise guard.GuardError("VAL opens only after the freeze")
        sp = Split("val")
        return sp, [d for r in eval_rows(sp, AUDIT_TOK) for d in r]
    raise ValueError(name)


def stage_steps(sp, steps):
    from csr.fle_paths import STAGE_TOKENS
    out, tot, k = {}, 0, 0
    targets = sorted(STAGE_TOKENS.items(), key=lambda kv: kv[1])
    for i, st in enumerate(steps):
        tot += int(sum(sp.lengths[d] - 1 for r in st for d in r))
        while k < len(targets) and tot >= targets[k][1]:
            out[targets[k][0]] = i + 1
            k += 1
        if k == len(targets):
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--L", type=int, default=128)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--batch-docs", type=int, default=128)
    ap.add_argument("--max-docs", type=int)
    ap.add_argument("--out-name")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    if not guard.require_execution(a.execute, "gpu" if a.device == "cuda" else "cpu-model"):
        return
    import hashlib
    import numpy as np
    import torch
    from csr.fle_lib import (H_MAX, MiniLM, decode_batch, draft_forecast, draft_list, load_draft, region_points,
                             seed_key)
    from csr.fle_paths import ART, CACHE, STOP_MARKER, prereg_sha
    torch.set_num_threads(4)
    mon = None
    if a.device == "cuda":
        guard.occupancy_gate(f"fle cache {a.name}")
        device = guard.setup_cuda_device()
        mon = guard.OccupancyMonitor(interval=15.0)
        mon.start()
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        device = torch.device("cpu")
    guard.register_self(f"fle cache {a.name}")
    sp, docs = doc_set(a.name)
    if a.max_docs:
        docs = docs[:a.max_docs]
    out_name = a.out_name or a.name
    cdir = CACHE / out_name
    cdir.mkdir(parents=True, exist_ok=True)
    drafts = draft_list()
    cfg = {"name": a.name, "split": sp.name, "L_points": a.L, "H": H_MAX, "temperature": 0.8, "topk": 50,
           "min_region": 32, "past_tokens": 192, "past_max_wordpieces": 256, "cont_max_wordpieces": 64,
           "drafts": drafts, "encoder": "all-MiniLM-L6-v2", "prereg_sha256": prereg_sha(),
           "docs_sha256": hashlib.sha256(np.asarray(docs, np.int64).tobytes()).hexdigest(), "n_docs": len(docs),
           "tokenizer_sha256": guard.sha256_file(guard.ROOT / "artifacts/tokenizer/pythia-12b/tokenizer.json"),
           "data_splits_sha256": "efcde27ee890e9831b8bcd7393ddba66b3ed76ad345ed91e58a72fc10185fcbf"}
    man_p = cdir / "manifest.json"
    if man_p.exists():
        man = json.loads(man_p.read_text())
        if man["config"] != json.loads(json.dumps(cfg)):
            raise guard.GuardError("cache config differs from the existing manifest")
        if man.get("complete"):
            print("cache already complete")
            return
    else:
        man = {"config": cfg, "chunks": [], "complete": False, "created": guard.now()}
    reason = None

    def on_sig(signum, frame):
        nonlocal reason
        reason = "signal " + signal.Signals(signum).name
    for s_ in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(s_, on_sig)
    enc = MiniLM(device)
    cfg_enc_sha = enc.sha
    man["encoder_sha256"] = cfg_enc_sha
    t_iso, t0 = guard.now(), time.time()
    done = {c["index"] for c in man["chunks"]}
    status = "COMPLETE"
    n_chunks = (len(docs) + CHUNK_DOCS - 1) // CHUNK_DOCS
    est = len(docs) * 4 * 7 * 2 * 700  # conservative bytes estimate (worst case)
    guard.require_disk(est, f"fle cache {out_name}")
    gen_s = enc_s = 0.0
    for ci in range(n_chunks):
        if ci in done:
            c = [c for c in man["chunks"] if c["index"] == ci][0]
            if guard.sha256_file(cdir / c["file"]) != c["sha256"]:
                raise guard.GuardError(f"completed chunk {ci} fails verification")
            continue
        if reason is None and STOP_MARKER.exists():
            reason = "stop marker"
        if reason is None and mon is not None and mon.contaminated:
            reason = f"foreign GPU process {mon.detail}"
        if reason is not None:
            status = "EXTERNALLY_STOPPED"
            break
        cd = docs[ci * CHUNK_DOCS:(ci + 1) * CHUNK_DOCS]
        pts = [(d, s) for d in cd for s in region_points(sp.lengths[d], a.L)]
        R = len(pts)
        D = len(drafts)
        tok = np.zeros((R, D, H_MAX), np.uint16)
        ptq = np.zeros((R, D, H_MAX), np.uint8)
        enq = np.zeros((R, D, H_MAX), np.uint8)
        t1q = np.zeros((R, D, H_MAX), np.uint8)
        # rows: documents with >= 1 point, sorted by length for padding efficiency
        pd_ = {}
        for i, (d, s) in enumerate(pts):
            pd_.setdefault(d, []).append((i, s))
        rdocs = sorted(pd_, key=lambda d: int(sp.lengths[d]))
        tg = time.time()
        for di, (dname, dpath, dsha) in enumerate(drafts):
            dm = load_draft(dpath, device, dsha)
            for bs in range(0, len(rdocs), a.batch_docs):
                bd = rdocs[bs:bs + a.batch_docs]
                T = int(max(sp.lengths[d] for d in bd))
                ids = np.full((len(bd), T), 1, np.int64)
                P = max(len(pd_[d]) for d in bd)
                points = np.full((len(bd), P), -1, np.int64)
                keys = np.zeros((len(bd), P), np.int64)
                where = np.full((len(bd), P), -1, np.int64)
                for b, d in enumerate(bd):
                    x = sp.doc(d)
                    ids[b, :len(x)] = x
                    for k, (i, s) in enumerate(pd_[d]):
                        points[b, k] = s
                        keys[b, k] = seed_key(di, d, s)
                        where[b, k] = i
                tt = lambda v: torch.as_tensor(v, device=device)
                o_tok, o_p, o_e, o_t = draft_forecast(dm, tt(ids), None, tt(points), tt(keys))
                m = where >= 0
                w = where[m]
                tok[w, di] = o_tok.cpu().numpy()[m].astype(np.uint16)
                ptq[w, di] = np.clip(np.round(o_p.cpu().numpy()[m] * 255), 0, 255).astype(np.uint8)
                enq[w, di] = np.clip(np.round(o_e.cpu().numpy()[m] / np.log(50304) * 255), 0, 255).astype(np.uint8)
                t1q[w, di] = np.clip(np.round(o_t.cpu().numpy()[m] * 255), 0, 255).astype(np.uint8)
            del dm
            if device.type == "cuda":
                torch.cuda.empty_cache()
        gen_s += time.time() - tg
        te = time.time()
        flat = tok.reshape(R * D, H_MAX)
        emb = enc.encode(decode_batch(flat), 64).reshape(R, D, 384)
        emb16 = enc.encode(decode_batch(flat[:, :16]), 64).reshape(R, D, 384)
        past = [sp.doc(d)[max(0, s - 192):s] for d, s in pts]
        embp = enc.encode(decode_batch(past), 256, left=True)
        enc_s += time.time() - te
        arr = {"gdoc": np.asarray([d for d, _ in pts], np.int32), "s": np.asarray([s for _, s in pts], np.int16),
               "length": np.asarray([sp.lengths[d] for d, _ in pts], np.int16), "tok": tok, "ptok": ptq, "ent": enq,
               "top1": t1q, "emb": emb, "emb16": emb16, "emb_past": embp}
        fn = f"chunk_{ci:04d}.npz"
        tmp = cdir / f".{fn}.tmp.npz"
        with open(tmp, "wb") as f:
            np.savez(f, **arr)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, cdir / fn)
        sha = guard.sha256_file(cdir / fn)
        man["chunks"].append({"index": ci, "file": fn, "sha256": sha, "n_regions": R, "n_docs": len(cd),
                              "bytes": (cdir / fn).stat().st_size, "time": guard.now()})
        man["chunks"].sort(key=lambda c: c["index"])
        guard.atomic_write_json(man_p, man, overwrite=True)
        print(f"{guard.now()} chunk {ci + 1}/{n_chunks}: {R} regions gen {gen_s:.0f}s enc {enc_s:.0f}s", flush=True)
    if mon is not None:
        mon.stop()
    if status == "COMPLETE":
        man["complete"] = True
        man["n_regions"] = int(sum(c["n_regions"] for c in man["chunks"]))
        man["bytes"] = int(sum(c["bytes"] for c in man["chunks"]))
        man["completed"] = guard.now()
        guard.atomic_write_json(man_p, man, overwrite=True)
        os.chmod(man_p, 0o400)
    rec = {"name": out_name, "status": status, "reason": reason, "gen_seconds": gen_s, "enc_seconds": enc_s,
           "seconds": time.time() - t0}
    with open(ART / "cache_jobs.jsonl", "a") as f:
        f.write(json.dumps({**rec, "time": guard.now()}) + "\n")
    if a.device == "cuda":
        guard.gpu_ledger(t_iso, guard.now(), f"fle_cache_{out_name}", f"fle cache {out_name}", status)
        with open(ART / "compute_ledger.tsv", "a") as f:
            f.write(f"{t_iso}\t{guard.now()}\t{time.time() - t0:.1f}\tfle_cache_{out_name}\tcache_{out_name}\t"
                    f"forecast_cache\t{status}\n")
    print(json.dumps(rec))
    sys.exit(0 if status == "COMPLETE" else 77)


if __name__ == "__main__":
    main()
