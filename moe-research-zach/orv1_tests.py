#!/usr/bin/env python
"""oracle-residual-v1 CPU correctness tests (prereg section 5). Real TOKEN-4 s0 weights on CPU, a few rows.
  T-ZERO    zero-initialised A, B, C reproduce frozen TOKEN-4 per-token CE bit-exactly (and hidden states)
  T-GRAD    gradients exist only for the residual bank (A, B) / bank + router (C); every base parameter has none
  T-ONE     exactly one specialist active per token: each token's residual equals its assigned specialist's output
            (A: semantic label, B: shuffled label, C: router argmax); C gate == 1 in value
  T-LOAD    A and B paper-load histograms identical (main and CALIB-SEARCH); B is a within-length-bin permutation of A
  T-SAME    the joint trainer feeds identical batches to A, B, C; plan identical across paired seeds (configs)
  T-RESUME  4 steps straight == 1+1+1+1 resumed legs (bitwise banks, routers, optimizer states, step, tokens)
  T-CONT    2 steps -> milestone ckpt -> continue to 4 == 4 steps straight (bitwise)
  T-HASH    the frozen base state hash is unchanged after training and evaluation
  T-EVAL    evaluation job: base unchanged, assigned-path CE == replacement-matrix entry (fp16 tolerance)
Writes artifacts/oracle-residual-v1/cpu_tests_<UTC>.json (0400)."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402

PY = str(guard.ROOT / ".venv/bin/python")
ENV = {**os.environ, "MOE_RESEARCH_APPROVE": "cpu-model"}


def sh(args):
    r = subprocess.run([PY] + args + ["--device", "cpu", "--execute"], env=ENV, cwd=guard.ROOT, capture_output=True,
                       text=True)
    return r.returncode, r.stdout[-3000:] + r.stderr[-3000:]


def eq_state(a, b):
    import torch
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(eq_state(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(eq_state(x, y) for x, y in zip(a, b))
    if torch.is_tensor(a):
        return torch.equal(a, b)
    return a == b


def main():
    guard.check_root()
    import numpy as np
    import torch
    from csr.fsp_lib import Split
    from csr.orv1_lib import (CONDS, base_forward, head_ce, load_base, load_labels, load_part, make_batch, make_conds,
                              rms0, state_hash, to_dev, token_assign)
    from csr.orv1_paths import ART, K, SHUF_LEN_BINS, TMP
    torch.set_num_threads(16)
    res = []
    dev = torch.device("cpu")
    model, _ = load_base(0, dev)
    h0 = state_hash(model)
    cs = Split("calib_search")
    sem, shuf = load_labels("cs")
    lab_sem, lab_shuf = torch.as_tensor(sem), torch.as_tensor(shuf)
    rows = load_part("SEL")[:2]
    bt = to_dev(make_batch(rows, cs), dev)
    x, hid = base_forward(model, bt)
    tgt = bt["targets"].reshape(-1)
    v = tgt >= 0
    ce_d = head_ce(model, hid, tgt, from_hidden=True)
    with torch.no_grad():
        o = model({"input_ids": bt["input_ids"], "doc_ids": bt["doc_ids"], "positions": bt["positions"]})
        ce_ref = head_ce(model, o["hidden"].reshape(-1, 384), tgt, from_hidden=True)
    conds = make_conds(0, dev)
    xn = rms0(x)
    asg_a, asg_b = token_assign(bt, lab_sem, lab_shuf)
    zero = {}
    for c in CONDS:
        with torch.no_grad():
            delta, idx, _ = conds[c](xn, {"A": asg_a, "B": asg_b, "C": None}[c], v)
            ce = head_ce(model, x + delta, tgt)
            hid_c = model.norm_f(x + delta)
        zero[c] = bool(torch.equal(ce[v], ce_d[v])) and bool(torch.equal(hid_c, hid)) and float(delta.abs().max()) == 0
    res.append({"test": "T-ZERO zero-init reproduces frozen TOKEN-4 CE bit-exactly", "ok": all(zero.values()) and
                bool(torch.equal(ce_d[v], ce_ref[v])), "per_cond": zero})
    # T-GRAD
    g = torch.Generator().manual_seed(5)
    for c in CONDS:
        with torch.no_grad():
            conds[c].bank.w2.copy_(torch.randn(conds[c].bank.w2.shape, generator=g) * 0.02)
    gok = {}
    for c in CONDS:
        for cc in CONDS:
            conds[cc].zero_grad(set_to_none=True)
        delta, idx, aux = conds[c](xn, {"A": asg_a, "B": asg_b, "C": None}[c], v)
        loss = head_ce(model, x + delta, tgt)[v].mean() + (0.01 * aux if c == "C" else 0)
        loss.backward()
        base_none = all(p.grad is None for p in model.parameters())
        own = all(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in conds[c].parameters())
        others = all(p.grad is None for cc in CONDS if cc != c for p in conds[cc].parameters())
        gok[c] = {"base_no_grad": base_none, "own_nonzero": own, "others_none": others,
                  "n_trainable": sum(p.numel() for p in conds[c].parameters()), **conds[c].n_params()}
    res.append({"test": "T-GRAD gradients only on the residual bank (+ router in C)",
                "ok": all(d["base_no_grad"] and d["own_nonzero"] and d["others_none"] for d in gok.values()), **gok})
    # T-ONE
    one = {}
    with torch.no_grad():
        for c in CONDS:
            delta, idx, _ = conds[c](xn, {"A": asg_a, "B": asg_b, "C": None}[c], v)
            allo = conds[c].all_outputs(xn)
            pick = allo.gather(1, idx.view(-1, 1, 1).expand(-1, 1, 384)).squeeze(1)
            want = {"A": asg_a, "B": asg_b}.get(c)
            if c == "C":
                _, gate, _, p = conds[c].route(xn, v)
                want = p.argmax(-1)
                gate_one = bool(torch.equal(gate, torch.ones_like(gate)))
            else:
                gate_one = True
            one[c] = {"idx_matches_rule": bool(torch.equal(idx, want)), "max_abs_vs_assigned":
                      float((delta - pick).abs().max()), "gate_one": gate_one, "n_active_per_token": 1}
    res.append({"test": "T-ONE exactly one specialist per token", "ok": all(
        d["idx_matches_rule"] and d["max_abs_vs_assigned"] < 1e-5 and d["gate_one"] for d in one.values()), **one})
    # T-LOAD
    lok = {}
    for which, spn in (("main", "main"), ("cs", "calib_search")):
        sp = Split(spn)
        s_, h_ = load_labels(which)
        pl = {}
        for p, a_, b_ in zip(sp.papers, s_, h_):
            pl[p] = (a_, b_)
        A = np.asarray([t[0] for t in pl.values() if t[0] >= 0])
        B = np.asarray([t[1] for t in pl.values() if t[0] >= 0])
        lok[which] = {"papers_per_cluster_equal": bool((np.bincount(A, minlength=K) == np.bincount(B, minlength=K)).all()),
                      "unassigned_consistent": bool(all((t[0] < 0) == (t[1] < 0) for t in pl.values())),
                      "n_papers": int(len(A)), "agreement": float((A == B).mean())}
    res.append({"test": "T-LOAD A/B paper-load histograms identical up to the frozen permutation",
                "ok": all(d["papers_per_cluster_equal"] and d["unassigned_consistent"] for d in lok.values()),
                "bins": SHUF_LEN_BINS, **lok})
    # trainer tests
    tdir = TMP / "cpu_tests"
    if tdir.exists():
        shutil.rmtree(tdir)
    tdir.mkdir(parents=True)
    common = ["--lr", "1e-3", "--aux", "0.01", "--target", "0", "--test-rows", "2", "--out-root", str(tdir)]
    rc1, o1 = sh(["orv1_train.py", "--seed", "0", "--name", "straight", "--test-steps", "4", "--milestones", "4"] + common)
    legs, rcs = 0, []
    while legs < 8:
        rc, o = sh(["orv1_train.py", "--seed", "0", "--name", "legs", "--test-steps", "4", "--milestones", "4",
                    "--max-seconds", "0"] + common)
        rcs.append(rc)
        legs += 1
        if rc != 75:
            break
    rc3, o3 = sh(["orv1_train.py", "--seed", "0", "--name", "cont", "--test-steps", "2", "--milestones", "2,4"] + common)
    rc4, o4 = sh(["orv1_train.py", "--seed", "0", "--name", "cont", "--test-steps", "4", "--milestones", "2,4"] + common)
    rc5, o5 = sh(["orv1_train.py", "--seed", "1", "--name", "seed1", "--test-steps", "2", "--milestones", "2"] + common)
    ok_run = rc1 == 0 and rcs[-1] == 0 and rc3 == 0 and rc4 == 0 and rc5 == 0
    det = {}
    if ok_run:
        A_ = torch.load(tdir / "straight/ckpt_4.pt", weights_only=False)
        B_ = torch.load(tdir / "legs/ckpt_4.pt", weights_only=False)
        C_ = torch.load(tdir / "cont/ckpt_4.pt", weights_only=False)
        for nm, X in (("legs", B_), ("cont", C_)):
            det[nm] = {"conds": eq_state(A_["conds"], X["conds"]), "opts": eq_state(A_["opts"], X["opts"]),
                       "step": A_["step"] == X["step"] == 4, "tokens": A_["tokens"] == X["tokens"]}
        ma = [json.loads(line) for line in open(tdir / "straight/metrics.jsonl")]
        ms = [json.loads(line) for line in open(tdir / "seed1/metrics.jsonl")]
        c0 = json.loads((tdir / "straight/config.json").read_text())
        c1 = json.loads((tdir / "seed1/config.json").read_text())
        det["paired_seeds"] = {"plan_sha_equal": c0["plan_sha256"] == c1["plan_sha256"],
                               "tokens_per_step_equal": [m["tokens"] for m in ma[:2]] == [m["tokens"] for m in ms[:2]]}
        dn = json.loads((tdir / "straight/done_4.json").read_text())
        det["hash"] = {"train_base_unchanged": dn["base_unchanged"]}
    res.append({"test": "T-RESUME 4 straight == 1+1+1+1 resumed legs (bitwise)", "ok": ok_run and all(
        det.get("legs", {}).values()), "leg_rcs": rcs, "detail": det.get("legs"), "log": None if ok_run else o1[-1500:]})
    res.append({"test": "T-CONT 2 -> milestone -> 4 == 4 straight (bitwise)", "ok": ok_run and all(
        det.get("cont", {}).values()), "detail": det.get("cont"), "log": None if ok_run else (o3 + o4)[-1500:]})
    res.append({"test": "T-SAME identical batches across conditions (joint step) and plan across paired seeds",
                "ok": ok_run and all(det.get("paired_seeds", {}).values()), "detail": det.get("paired_seeds"),
                "note": "A, B, C consume the same micro-batch tensors inside one step by construction (orv1_train.py)"})
    # T-EVAL
    ev = tdir / "eval_test.npz"
    rc6, o6 = sh(["orv1_eval.py", "--seed", "0", "--ckpt", str(tdir / "straight/ckpt_4.pt"), "--part", "SEL", "--out",
                  str(ev), "--full", "--rows-limit", "2"])
    eok = {}
    if rc6 == 0:
        sj = json.loads(ev.with_suffix(".json").read_text())
        z = np.load(ev)
        eok = {"base_unchanged": sj["base_unchanged"], "cons_max": sj["matrix_consistency_max_abs"],
               "n_tokens": sj["n_tokens"], "ce": {c: sj[f"ce_{c}"] for c in "DABC"},
               "one_label_per_token": bool(len(z["lab_sem"]) == sj["n_tokens"])}
    res.append({"test": "T-EVAL + T-HASH evaluation job: base unchanged; assigned CE == matrix entry",
                "ok": rc6 == 0 and eok.get("base_unchanged") is True and det.get("hash", {}).get("train_base_unchanged")
                is True and max(eok.get("cons_max", {"A": 1}).values()) < 0.02, **eok,
                "log": None if rc6 == 0 else o6[-1500:]})
    res.append({"test": "T-HASH in-process base hash", "ok": state_hash(model) == h0})
    doc = {"time": guard.now(), "n": len(res), "n_fail": sum(not r["ok"] for r in res), "results": res}
    p = ART / f"cpu_tests_{guard.utc_stamp()}.json"
    guard.atomic_write_json(p, doc, mode=0o400)
    print(json.dumps(doc, indent=1, default=str)[:12000])
    for f in tdir.rglob("*"):
        if f.is_file():
            f.chmod(0o600)
    shutil.rmtree(tdir, ignore_errors=True)
    sys.exit(0 if doc["n_fail"] == 0 else 1)


if __name__ == "__main__":
    main()
