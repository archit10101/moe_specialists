#!/usr/bin/env python
"""oracle-residual-v1 joint trainer (guarded GPU job; CPU for tests). Trains A (SEMANTIC-ORACLE), B (SHUFFLED-ORACLE)
and C (TOKEN-ROUTED-RESIDUAL) residual banks on top of one frozen TOKEN-4 seed, in ONE process on the SAME batches:
the frozen base forward is computed once per micro-batch and shared, so examples, order, target masks, steps and token
counts are identical by construction. Separate AdamW per condition; identical hyperparameters; constant LR after a
linear warmup (so a run continued to a larger budget is identical to a run trained straight to it).

State: <run>/slot.pt (rolling resumable slot, atomically replaced every SLOT_EVERY_S and at leg end) and permanent
<run>/ckpt_<tokens>.pt at each milestone budget. Each holds bank / router weights, optimizer states, step, RNG states,
plan / table / base hashes. Exit codes: 0 target reached, 75 leg end (--max-seconds; resumable), 77 external stop.

  orv1_train.py --seed S --name NAME --lr LR --aux AUX --target TOKENS [--milestones T1,T2] [--max-seconds N]
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

MON = None


def atomic_torch_save(obj, path):
    import torch
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(obj, tmp)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def rng_state():
    import random
    import numpy as np
    import torch
    st = {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st):
    import random
    import numpy as np
    import torch
    torch.set_rng_state(st["torch"].cpu())
    np.random.set_state(st["numpy"])
    random.setstate(st["python"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in st["cuda"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--aux", type=float, required=True)
    ap.add_argument("--target", type=int, required=True, help="token budget to reach (a BUDGETS value or pilot)")
    ap.add_argument("--milestones", default="", help="comma-separated budgets at which to save permanent ckpts")
    ap.add_argument("--max-seconds", type=float, default=4800)
    ap.add_argument("--out-root", default="")
    ap.add_argument("--test-rows", type=int, default=0)
    ap.add_argument("--test-steps", type=int, default=0)
    ap.add_argument("--slot-every", type=float, default=-1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    if not guard.require_execution(a.execute, "gpu" if a.device == "cuda" else "cpu-model"):
        return
    from csr.orv1_paths import check_prereg
    test = a.test_rows > 0
    if test and not a.out_root:
        raise guard.GuardError("test mode requires --out-root")
    if not test:
        check_prereg()
    global MON
    mon = None
    if a.device == "cuda":
        guard.occupancy_gate(f"orv1 train {a.name}")
        mon = MON = guard.OccupancyMonitor(interval=15.0)
        mon.start()
    guard.register_self(f"orv1 train {a.name}")
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
        guard.gpu_ledger(t_iso, guard.now(), f"orv1_train_{a.name}", "orv1 train", status)
    print(json.dumps({"status": status, "seconds": round(time.time() - t0, 1)}), flush=True)
    sys.exit(rc)


def train(a):
    import numpy as np
    import torch
    from csr.fsp_lib import Split
    from csr.orv1_lib import (CONDS, base_forward, head_ce, load_base, load_labels, load_plan, make_batch, make_conds,
                              rms0, state_hash, steps_for, to_dev, token_assign)
    from csr.orv1_paths import (BETAS, CLIP, K, MICRO_ROWS, RUNS, SLOT_EVERY_S, TABLES, WARMUP_STEPS, WEIGHT_DECAY)
    dev = guard.setup_cuda_device() if a.device == "cuda" else torch.device("cpu")
    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    slot_every = SLOT_EVERY_S if a.slot_every < 0 else a.slot_every
    root = Path(a.out_root) if a.out_root else RUNS
    rd = root / a.name
    rd.mkdir(parents=True, exist_ok=True)
    steps, plan_sha = load_plan()
    if a.test_rows:
        flat = [r for st in steps[:4] for r in st]
        steps = [flat[i * a.test_rows:(i + 1) * a.test_rows] for i in range(a.test_steps)]
        n_target = a.test_steps
        miles = {int(x): int(x) for x in a.milestones.split(",") if x}  # test: milestones given in steps
    else:
        n_target = steps_for(a.target)
        miles = {steps_for(int(x)): int(x) for x in a.milestones.split(",") if x}
    tab = json.loads((TABLES / "tables.json").read_text())["sha256"]
    model, ck_sha = load_base(a.seed, dev)
    h0 = state_hash(model)
    cfg = {"seed": a.seed, "lr": a.lr, "aux": a.aux, "plan_sha256": plan_sha, "tables_sha256": tab,
           "base_ckpt_sha256": ck_sha, "base_state_hash": h0, "test_rows": a.test_rows}
    if (rd / "config.json").exists():
        old = json.loads((rd / "config.json").read_text())
        if old != cfg:
            raise guard.GuardError(f"resume: config differs {old} vs {cfg}")
    else:
        guard.atomic_write_json(rd / "config.json", cfg)
    split = Split("main")
    sem, shuf = load_labels("main")
    lab_sem, lab_shuf = torch.as_tensor(sem, device=dev), torch.as_tensor(shuf, device=dev)
    conds = make_conds(a.seed, dev)
    opts = {c: torch.optim.AdamW(conds[c].parameters(), lr=a.lr, betas=BETAS, weight_decay=WEIGHT_DECAY,
                                 fused=dev.type == "cuda") for c in CONDS}
    step, tokens = 0, 0
    src = None
    if (rd / "slot.pt").exists():
        src = rd / "slot.pt"
    else:
        done = sorted((s for s in miles if s <= n_target and (rd / f"ckpt_{miles[s]}.pt").exists()))
        if done:
            src = rd / f"ckpt_{miles[done[-1]]}.pt"
    if src is not None:
        S = torch.load(src, map_location=dev, weights_only=False)
        if S["plan_sha256"] != plan_sha or S["base_state_hash"] != h0:
            raise guard.GuardError("resume: plan or base hash differs")
        for c in CONDS:
            conds[c].load_state_dict(S["conds"][c])
            opts[c].load_state_dict(S["opts"][c])
        step, tokens = S["step"], S["tokens"]
        set_rng_state(S["rng"])
        print(json.dumps({"resumed_from": str(src.name), "step": step, "tokens": tokens}), flush=True)
    if step > n_target:
        raise guard.GuardError(f"resume point {step} beyond target {n_target}")

    def snapshot():
        return {"step": step, "tokens": tokens, "conds": {c: conds[c].state_dict() for c in CONDS},
                "opts": {c: opts[c].state_dict() for c in CONDS}, "rng": rng_state(), "plan_sha256": plan_sha,
                "base_state_hash": h0, "tables_sha256": tab, "cfg": cfg, "time": guard.now()}

    t_leg, t_slot = time.time(), time.time()
    met = open(rd / "metrics.jsonl", "a")
    while step < n_target:
        if MON is not None and MON.contaminated:
            atomic_torch_save(snapshot(), rd / "slot.pt")
            print(json.dumps({"external": MON.detail, "step": step}), flush=True)
            sys.exit(77)
        rows = steps[step]
        mbs = [rows[i:i + MICRO_ROWS] for i in range(0, len(rows), MICRO_ROWS)]
        bts = [to_dev(make_batch(mb, split), dev) for mb in mbs]
        n_valid = int(sum(int((bt["targets"] >= 0).sum()) for bt in bts))
        lr = a.lr * min(1.0, (step + 1) / WARMUP_STEPS)
        for c in CONDS:
            for g in opts[c].param_groups:
                g["lr"] = lr
            opts[c].zero_grad(set_to_none=True)
        acc = {k: 0.0 for k in ("D", "A", "B", "C", "aux")}
        loadC = torch.zeros(K, device=dev)
        loadA = torch.zeros(K, device=dev)
        for bt in bts:
            x, hid = base_forward(model, bt)
            tgt = bt["targets"].reshape(-1)
            valid = tgt >= 0
            with torch.no_grad():
                acc["D"] += float(head_ce(model, hid, tgt, from_hidden=True)[valid].sum())
            xn = rms0(x)
            asg_a, asg_b = token_assign(bt, lab_sem, lab_shuf)
            for c in CONDS:
                delta, idx, aux = conds[c](xn, {"A": asg_a, "B": asg_b, "C": None}[c], valid)
                ce = head_ce(model, x + delta, tgt)
                loss = ce[valid].sum() / n_valid
                frac = float(valid.sum()) / n_valid
                if c == "C":
                    loss = loss + a.aux * aux * frac
                    loadC += torch.bincount(idx[valid], minlength=K).float()
                    acc["aux"] += float(aux) * frac
                if c == "A":
                    loadA += torch.bincount(idx[valid], minlength=K).float()
                loss.backward()
                acc[c] += float(ce[valid].detach().sum())
        gn = {}
        for c in CONDS:
            ps = [p for p in conds[c].parameters()]
            gn[c] = float(torch.nn.utils.clip_grad_norm_(ps, CLIP))
            opts[c].step()
        bad = [k for k in ("A", "B", "C") if not np.isfinite(acc[k])] + [c for c in CONDS if not np.isfinite(gn[c])]
        if bad:
            print(json.dumps({"pathology": "non-finite", "conds": bad, "step": step}), flush=True)
            sys.exit(1)
        step += 1
        tokens += n_valid
        lc = (loadC / loadC.sum()).cpu().numpy()
        rec = {"step": step, "tokens": tokens, "lr": lr, **{f"ce_{k}": acc[k] / n_valid for k in ("D", "A", "B", "C")},
               "aux": acc["aux"], "gn": gn, "loadC_max": float(lc.max()),
               "loadC_eff": float(np.exp(-(lc[lc > 0] * np.log(lc[lc > 0])).sum())), "t": round(time.time() - t_leg, 1)}
        if step % 10 == 0 or step == n_target or step <= 3:
            rec["loadC"] = lc.round(4).tolist()
            rec["loadA"] = (loadA / loadA.sum()).cpu().numpy().round(4).tolist()
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
        (rd / "slot.pt").unlink()  # superseded rolling slot of this run (the milestone ckpt holds the same state)
    if n_target not in miles:
        atomic_torch_save(snapshot(), rd / f"ckpt_final_step{step}.pt")
    guard.atomic_write_json(rd / f"done_{n_target}.json", {"step": step, "tokens": tokens, "base_state_hash_after": h1,
                                                          "base_unchanged": True, "time": guard.now()})
    print(json.dumps({"done": True, "step": step, "tokens": tokens}), flush=True)


if __name__ == "__main__":
    main()
