#!/usr/bin/env python
"""fspv2 forecast cache (protocol section 4). For every document in the first 50% (+1 step) of the final main stream
(or every calib_search document with --split calib_search), stores causal draft outputs:
  roll_m/roll_w (n,8) uint16 greedy rollouts of the matching / fixed-wrong draft; ent_m/ent_w (n,8) uint8 step
  entropies; top_m/top_w (n,4) uint8 step-1 top-4 probs; lp (n,5) fp16 log p_d(x_t|x_<t) for all five drafts.
Each draft only processes the documents it is responsible for (rollouts where it is the matching or wrong draft,
score-only passes elsewhere). Dry-run by default; GPU via scripts/gpu_guarded.sh.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafts", required=True, help="json list of 5 domain-draft checkpoint paths")
    ap.add_argument("--split", default="main")
    ap.add_argument("--budget-tokens", type=int, help="final main budget (to locate the first 50%)")
    ap.add_argument("--rows-per-step", type=int, default=64)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-rows", type=int, default=48)
    ap.add_argument("--max-seconds", type=float, default=4800)
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    if not guard.require_execution(a.execute, "gpu"):
        return
    sys.exit(run(a))


def run(a):
    import numpy as np
    import torch
    from numpy.lib.format import open_memmap
    from csr.fsp_lib import LN_V, Split, build_draft, draft_rollout_fast, draft_score_fast, main_order, pack_rows, \
        plan_steps, to_torch, make_batch
    guard.occupancy_gate("fsp cache")
    device = guard.setup_cuda_device()
    mon = guard.OccupancyMonitor(15.0)
    mon.start()
    guard.register_self(f"fsp cache {a.split}")
    t_iso, t0 = guard.now(), time.time()
    out = guard.inside_root(a.out)
    out.mkdir(parents=True, exist_ok=True)
    state_p = out / "progress.json"
    sp = Split(a.split)
    if a.split == "main":
        rows = pack_rows(sp.lengths, main_order(sp))
        steps, _ = plan_steps(rows, sp.lengths, a.rows_per_step, a.budget_tokens)
        n_half = int(math.ceil(0.5 * len(steps))) + 1
        docs = sorted({d for st in steps[:n_half] for r in st for d in r})
    else:
        docs = list(range(sp.n_docs))
    if not (out / "index.npy").exists():
        index = np.full(sp.n_docs, -1, dtype=np.int64)
        off = 0
        for d in docs:
            index[d] = off
            off += int(sp.lengths[d])
        n = off
        guard.require_disk(int(n * 66 * 1.05), "fsp forecast cache")
        free = guard.disk_free_bytes()
        if free - n * 66 < 80 * 1024**3:
            raise guard.GuardError("80 GiB free-disk rule would be violated")
        np.save(out / "index.npy", index)
        for k, shape, dt in (("roll_m", (n, 8), np.uint16), ("roll_w", (n, 8), np.uint16), ("ent_m", (n, 8), np.uint8),
                             ("ent_w", (n, 8), np.uint8), ("top_m", (n, 4), np.uint8), ("top_w", (n, 4), np.uint8),
                             ("lp", (n, 5), np.float16)):
            open_memmap(out / f"{k}.npy", mode="w+", dtype=dt, shape=shape).flush()
        guard.atomic_write_json(state_p, {"done": [], "n_tokens": n, "n_docs": len(docs)})
    index = np.load(out / "index.npy")
    arr = {k: np.load(out / f"{k}.npy", mmap_mode="r+") for k in
           ("roll_m", "roll_w", "ent_m", "ent_w", "top_m", "top_w", "lp")}
    prog = json.loads(state_p.read_text())
    paths = json.loads(Path(a.drafts).read_text()) if a.drafts.endswith(".json") else json.loads(a.drafts)
    status = "COMPLETE"
    try:
        for d in range(5):
            for job in ("roll", "lp"):
                key = f"{d}:{job}"
                if key in prog["done"]:
                    continue
                draft = build_draft(0)
                draft.load_state_dict(torch.load(guard.ROOT / paths[d], map_location="cpu", weights_only=True)["model"])
                draft = draft.to(device).eval()
                resp = [i for i in docs if sp.domains[i] == d or sp.wrong[i] == d]
                mine = resp if job == "roll" else [i for i in docs if not (sp.domains[i] == d or sp.wrong[i] == d)]
                rows = [[i] for i in mine]  # single-document rows (flash causal attention in the fast path)
                start = prog.get(f"{key}:row", 0)
                tj = time.time()
                for ri in range(start, len(rows), a.batch_rows):
                    if time.time() - t0 > a.max_seconds or mon.contaminated:
                        prog[f"{key}:row"] = ri
                        raise TimeoutError("leg limit")
                    rr = rows[ri:ri + a.batch_rows]
                    bt = to_torch(make_batch(rr, sp), device)
                    Ls = torch.as_tensor([int(sp.lengths[r[0]]) for r in rr], device=device)
                    if job == "roll":
                        roll, ent, top4, lp_cur = draft_rollout_fast(draft, bt["input_ids"], Ls, H=8)
                    else:
                        lp_cur = draft_score_fast(draft, bt["input_ids"], Ls)
                    lp_cur = lp_cur.cpu().numpy()
                    if job == "roll":
                        roll = roll.cpu().numpy().astype(np.uint16)
                        entq = np.clip(np.rint(ent.float().cpu().numpy() / LN_V * 255), 0, 255).astype(np.uint8)
                        topq = np.clip(np.rint(top4.float().cpu().numpy() * 255), 0, 255).astype(np.uint8)
                    for b, row in enumerate(rr):
                        t = 0
                        for di in row:
                            L = int(sp.lengths[di])
                            s = int(index[di])
                            arr["lp"][s:s + L, d] = lp_cur[b, t:t + L].astype(np.float16)
                            if job == "roll":
                                k = "m" if sp.domains[di] == d else "w"
                                arr[f"roll_{k}"][s:s + L] = roll[b, t:t + L]
                                arr[f"ent_{k}"][s:s + L] = entq[b, t:t + L]
                                arr[f"top_{k}"][s:s + L] = topq[b, t:t + L]
                            t += L
                    if (ri // a.batch_rows) % 200 == 0:
                        el = time.time() - tj
                        print(f"{key} rows {ri}/{len(rows)} {el:.0f}s", flush=True)
                for v in arr.values():
                    v.flush()
                prog["done"].append(key)
                prog[f"{key}:seconds"] = time.time() - tj
                guard.atomic_write_json(state_p, prog, overwrite=True)
                del draft
                torch.cuda.empty_cache()
    except TimeoutError:
        status = "INTERRUPTED"
    finally:
        for v in arr.values():
            v.flush()
        guard.atomic_write_json(state_p, prog, overwrite=True)
        mon.stop()
    t_end = guard.now()
    sec = time.time() - t0
    guard.gpu_ledger(t_iso, t_end, f"fsp_cache_{a.split}", f"fsp forecast cache {a.split}", status)
    with open(guard.ROOT / "artifacts/forecast-specialist-pretraining-v2/compute_ledger.tsv", "a") as f:
        f.write(f"{t_iso}\t{t_end}\t{sec:.1f}\tfsp_cache_{a.split}\tcache_{a.split}\tforecast_cache\t{status}\n")
    if status == "COMPLETE":
        man = {"split": a.split, "n_docs": prog["n_docs"], "n_tokens": prog["n_tokens"],
               "bytes": sum((out / f"{k}.npy").stat().st_size for k in arr) + (out / "index.npy").stat().st_size,
               "drafts": paths, "files": {k: guard.sha256_file(out / f"{k}.npy") for k in list(arr) + ["index"]},
               "job_seconds": {k: v for k, v in prog.items() if k.endswith(":seconds")}, "created": guard.now()}
        guard.atomic_write_json(out / "manifest.json", man, overwrite=True)
    print(status, round(sec), flush=True)
    return 0 if status == "COMPLETE" else 75


if __name__ == "__main__":
    main()
