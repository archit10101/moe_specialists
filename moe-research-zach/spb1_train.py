#!/usr/bin/env python
"""spb1 joint bank trainer (adapted from orv1_train.py; guarded GPU job; CPU for tests). Trains SEMANTIC-M (pair mode,
semantic paper clusters), SHUFFLED-M (pair mode, size-matched persistent shuffled clusters) and GENERIC-KM (free mode)
residual banks on ONE frozen TOKEN-4 base seed, in ONE process on the SAME batches: the frozen base forward is computed
once per micro-batch and shared, so examples, order, target masks, steps and token counts are identical by
construction. Separate AdamW per condition; identical hyperparameters, initialisation and parameter counts; constant LR
after a linear warmup (a run continued to a larger budget is identical to one trained straight to it).

State: <run>/slot.pt (rolling resumable slot every SLOT_EVERY_S and at leg end) and permanent <run>/ckpt_<tokens>.pt at
milestone budgets. Exit codes: 0 target reached, 75 leg end (--max-seconds; resumable), 77 external stop.

  spb1_train.py --seed S --gen G --target TOKENS [--milestones T1,T2] [--max-seconds N]
               [--out-root DIR --test-rows R --test-steps N]   (test mode: tiny plan, CPU allowed)"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402
from orv1_train import atomic_torch_save, rng_state, set_rng_state  # noqa: E402

MON = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--gen", type=int, default=0)
    ap.add_argument("--target", type=int, required=True)
    ap.add_argument("--milestones", default="")
    ap.add_argument("--max-seconds", type=float, default=4800)
    ap.add_argument("--out-root", default="")
    ap.add_argument("--test-rows", type=int, default=0)
    ap.add_argument("--test-steps", type=int, default=0)
    ap.add_argument("--slot-every", type=float, default=-1)
    ap.add_argument("--cfg-json", default="", help="test only: explicit generation config")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    if not guard.require_execution(a.execute, "gpu" if a.device == "cuda" else "cpu-model"):
        return
    from csr.spb1_paths import check_prereg
    test = a.test_rows > 0
    if test and not a.out_root:
        raise guard.GuardError("test mode requires --out-root")
    if not test:
        check_prereg()
    global MON
    mon = None
    if a.device == "cuda":
        guard.occupancy_gate(f"spb1 train g{a.gen} s{a.seed}")
        mon = MON = guard.OccupancyMonitor(interval=15.0)
        mon.start()
    guard.register_self(f"spb1 train g{a.gen} s{a.seed}")
    t_iso, t0 = guard.now(), time.time()
    status, rc = "COMPLETE", 0
    try:
        train(a)
    except SystemExit as ex:
        rc = int(ex.code or 0)
        status = {75: "LEG_END", 77: "EXTERNALLY_STOPPED", 0: "COMPLETE"}.get(rc, "FAILED")
    except KeyboardInterrupt:
        status, rc = "EXTERNALLY_STOPPED", 77
    finally:
        if mon is not None:
            mon.stop()
            if mon.contaminated and status in ("COMPLETE", "LEG_END"):
                status, rc = "EXTERNALLY_STOPPED", 77
    if a.device == "cuda":
        guard.gpu_ledger(t_iso, guard.now(), f"spb1_train_g{a.gen}_s{a.seed}", "spb1 train", status)
    print(json.dumps({"status": status, "seconds": round(time.time() - t0, 1)}), flush=True)
    sys.exit(rc)


def train(a):
    import numpy as np
    import torch
    from csr.fsp_lib import Split
    from csr.orv1_lib import base_forward, head_ce, load_base, load_plan, make_batch, state_hash, steps_for, to_dev
    from csr.spb1_lib import make_banks, rms0, token_groups
    from csr import spb1_paths as P
    dev = guard.setup_cuda_device() if a.device == "cuda" else torch.device("cpu")
    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    slot_every = P.SLOT_EVERY_S if a.slot_every < 0 else a.slot_every
    G = json.loads(a.cfg_json) if a.cfg_json else P.gen_cfg(a.gen)
    if G.get("assign", "hard") != "hard":
        raise NotImplementedError("soft assignment is an engineering option implemented only if a generation selects it")
    M, K = int(G["M"]), P.K
    root = Path(a.out_root) if a.out_root else P.RUNS
    rd = root / f"g{a.gen}_s{a.seed}"
    rd.mkdir(parents=True, exist_ok=True)
    steps, plan_sha = load_plan()
    if a.test_rows:
        flat = [r for st in steps[:4] for r in st]
        steps = [flat[i * a.test_rows:(i + 1) * a.test_rows] for i in range(a.test_steps)]
        n_target = a.test_steps
        miles = {int(x): int(x) for x in a.milestones.split(",") if x}
    else:
        n_target = steps_for(a.target)
        miles = {steps_for(int(x)): int(x) for x in a.milestones.split(",") if x}
    tab = json.loads((P.TABLES / "tables.json").read_text())["sha256"]
    model, ck_sha = load_base(a.seed, dev)
    h0 = state_hash(model)
    cfg = {"seed": a.seed, "gen": a.gen, "gen_cfg": G, "lr": P.LR, "plan_sha256": plan_sha, "tables_sha256": tab,
           "base_ckpt_sha256": ck_sha, "base_state_hash": h0, "test_rows": a.test_rows}
    if (rd / "config.json").exists():
        old = json.loads((rd / "config.json").read_text())
        if old != cfg:
            raise guard.GuardError(f"resume: config differs {old} vs {cfg}")
    else:
        guard.atomic_write_json(rd / "config.json", cfg)
    split = Split("main")
    zl = np.load(P.TABLES / "labels_main.npz")
    lab_sem = torch.as_tensor(zl["sem"].astype(np.int64), device=dev)
    lab_shuf = torch.as_tensor(zl["shuf"].astype(np.int64), device=dev)
    banks = make_banks(a.seed, K, M, dev)
    names = list(banks)
    sem_n, shuf_n, gen_n = names
    opts = {}
    for c in names:
        b = banks[c]
        opts[c] = torch.optim.AdamW([{"params": list(b.bank.parameters()), "lr_mult": 1.0},
                                     {"params": list(b.router.parameters()), "lr_mult": float(G["sub_lr_mult"])}],
                                    lr=P.LR, betas=P.BETAS, weight_decay=P.WEIGHT_DECAY, fused=dev.type == "cuda")
    step, tokens = 0, 0
    src = None
    if (rd / "slot.pt").exists():
        src = rd / "slot.pt"
    else:
        done = sorted(s for s in miles if s <= n_target and (rd / f"ckpt_{miles[s]}.pt").exists())
        if done:
            src = rd / f"ckpt_{miles[done[-1]]}.pt"
    if src is not None:
        S = torch.load(src, map_location=dev, weights_only=False)
        if S["plan_sha256"] != plan_sha or S["base_state_hash"] != h0 or S["cfg"]["gen_cfg"] != G:
            raise guard.GuardError("resume: plan, base or generation config differs")
        for c in names:
            banks[c].load_state_dict(S["banks"][c])
            opts[c].load_state_dict(S["opts"][c])
        step, tokens = S["step"], S["tokens"]
        set_rng_state(S["rng"])
        print(json.dumps({"resumed_from": str(src.name), "step": step, "tokens": tokens}), flush=True)
    if step > n_target:
        raise guard.GuardError(f"resume point {step} beyond target {n_target}")

    def snapshot():
        return {"step": step, "tokens": tokens, "banks": {c: banks[c].state_dict() for c in names},
                "opts": {c: opts[c].state_dict() for c in names}, "rng": rng_state(), "plan_sha256": plan_sha,
                "base_state_hash": h0, "tables_sha256": tab, "cfg": cfg, "time": guard.now()}

    t_leg, t_slot = time.time(), time.time()
    met = open(rd / "metrics.jsonl", "a")
    E = K * M
    while step < n_target:
        if MON is not None and MON.contaminated:
            atomic_torch_save(snapshot(), rd / "slot.pt")
            print(json.dumps({"external": MON.detail, "step": step}), flush=True)
            sys.exit(77)
        if P.STOP_MARKER.exists() and not a.test_rows:
            atomic_torch_save(snapshot(), rd / "slot.pt")
            print(json.dumps({"stop_marker": True, "step": step}), flush=True)
            sys.exit(77)
        rows = steps[step]
        mbs = [rows[i:i + P.MICRO_ROWS] for i in range(0, len(rows), P.MICRO_ROWS)]
        bts = [to_dev(make_batch(mb, split), dev) for mb in mbs]
        n_valid = int(sum(int((bt["targets"] >= 0).sum()) for bt in bts))
        lr = P.LR * min(1.0, (step + 1) / P.WARMUP_STEPS)
        for c in names:
            for g in opts[c].param_groups:
                g["lr"] = lr * g["lr_mult"]
            opts[c].zero_grad(set_to_none=True)
        acc = {k: 0.0 for k in ["D"] + names + [f"aux_{c}" for c in names]}
        load = {c: torch.zeros(E, device=dev) for c in names}
        for bt in bts:
            x, hid = base_forward(model, bt)
            tgt = bt["targets"].reshape(-1)
            valid = tgt >= 0
            with torch.no_grad():
                acc["D"] += float(head_ce(model, hid, tgt, from_hidden=True)[valid].sum())
            xn = rms0(x)
            g_sem, g_shuf = token_groups(bt, lab_sem, lab_shuf)
            grp = {sem_n: g_sem, shuf_n: g_shuf, gen_n: g_sem}   # GEN ignores groups (free mode)
            frac = float(valid.sum()) / n_valid
            for c in names:
                delta, idx, aux = banks[c](xn, grp[c], valid)
                ce = head_ce(model, x + delta, tgt)
                loss = ce[valid].sum() / n_valid + float(G["aux"]) * aux * frac
                loss.backward()
                acc[c] += float(ce[valid].detach().sum())
                acc[f"aux_{c}"] += float(aux) * frac
                load[c] += torch.bincount(idx[valid], minlength=E).float()
        gn = {}
        for c in names:
            gn[c] = float(torch.nn.utils.clip_grad_norm_(list(banks[c].parameters()), P.CLIP))
            opts[c].step()
        bad = [c for c in names if not (np.isfinite(acc[c]) and np.isfinite(gn[c]))]
        if bad:
            print(json.dumps({"pathology": "non-finite", "conds": bad, "step": step}), flush=True)
            sys.exit(1)
        step += 1
        tokens += n_valid
        rec = {"step": step, "tokens": tokens, "lr": lr, **{f"ce_{k}": acc[k] / n_valid for k in ["D"] + names},
               **{k: v for k, v in acc.items() if k.startswith("aux_")}, "gn": gn, "t": round(time.time() - t_leg, 1)}
        for c in names:
            lc = (load[c] / load[c].sum()).cpu().numpy()
            rec[f"load_max_{c}"] = float(lc.max())
            rec[f"load_eff_{c}"] = float(np.exp(-(lc[lc > 0] * np.log(lc[lc > 0])).sum()))
            if step % 50 == 0 or step == n_target or step <= 3:
                rec[f"load_{c}"] = lc.round(4).tolist()
        met.write(json.dumps(rec) + "\n")
        met.flush()
        if step in miles:
            atomic_torch_save(snapshot(), rd / f"ckpt_{miles[step]}.pt")
            print(json.dumps({"milestone": miles[step], "step": step, "tokens": tokens}), flush=True)
        if step < n_target and time.time() - t_slot > slot_every:
            atomic_torch_save(snapshot(), rd / "slot.pt")
            t_slot = time.time()
        if step < n_target and time.time() - t_leg > a.max_seconds:
            atomic_torch_save(snapshot(), rd / "slot.pt")
            print(json.dumps({"leg_end": step}), flush=True)
            sys.exit(75)
    met.close()
    h1 = state_hash(model)
    if h1 != h0:
        raise guard.GuardError("frozen base state changed during training")
    if (rd / "slot.pt").exists():
        (rd / "slot.pt").unlink()   # superseded rolling slot of this run (the milestone ckpt holds the same state)
    if n_target not in miles:
        atomic_torch_save(snapshot(), rd / f"ckpt_final_step{step}.pt")
    guard.atomic_write_json(rd / f"done_{n_target}.json", {"step": step, "tokens": tokens, "base_state_hash_after": h1,
                                                          "base_unchanged": True, "time": guard.now()}, overwrite=True)
    print(json.dumps({"done": True, "step": step, "tokens": tokens}), flush=True)


if __name__ == "__main__":
    main()
