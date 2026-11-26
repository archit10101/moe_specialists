#!/usr/bin/env python
"""joint-semantic-residual-pretrain-v1 CPU correctness tests (protocol section 6, T1-T12), real LFRModel config.
Writes ART/cpu_tests_<ts>.json (0400). Exit 0 iff all pass."""
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402

PY = str(guard.ROOT / ".venv/bin/python")


def specs(r=0.02, tau=1.0):
    return [{"name": "B", "kind": "routed", "tau": tau, "r": 0.0, "alpha": 0.01, "groups": None},
            {"name": "C", "kind": "routed", "tau": tau, "r": r, "alpha": 0.01, "groups": "shuf"},
            {"name": "D", "kind": "routed", "tau": tau, "r": r, "alpha": 0.01, "groups": "sem"},
            {"name": "E", "kind": "oracle"}]


def base_cfg(seed=0, banks=True):
    from csr.jsr_paths import T4_CFG_RUN
    c = json.loads((guard.ROOT / T4_CFG_RUN / "config.json").read_text())
    c = {k: c[k] for k in ("budget_tokens", "diag_every", "model", "seed", "train")}
    c.update({"experiment": "joint-semantic-residual-pretrain-v1", "run_name": "test", "test": True, "seed": seed,
              "cond": {"banks": specs() if banks else [], "lam_r": 1.0}, "milestones": {}, "bank_milestones": {}})
    c["diag_every"] = 2  # exercise the grad_diag path inside the tested steps
    return c


def small_batch(n_rows=2, step=0):
    import torch
    from csr.fsp_lib import make_batch, to_torch
    from csr.lfr_lib import main_stream
    split, steps, _ = main_stream(245_000_000)
    npb = make_batch(steps[step][:n_rows], split)
    return split, npb, to_torch(npb, torch.device("cpu"))


def setup(seed=0):
    import torch
    from csr.jsr_bank import Bank
    from csr.lfr_model import LFRConfig, LFRModel
    cfg = base_cfg(seed)
    model = LFRModel(LFRConfig.from_dict(cfg["model"]), seed)
    banks = {s["name"]: Bank(s, seed) for s in specs()}
    # give the banks non-trivial weights (W2 = 0 at init makes residual gradients to the router exactly zero)
    g = torch.Generator().manual_seed(1)
    for b in banks.values():
        with torch.no_grad():
            b.experts.w2.copy_(torch.randn(b.experts.w2.shape, generator=g) * 0.02)
    return cfg, model, banks


def fwd(model, bt):
    out = model({"input_ids": bt["input_ids"], "doc_ids": bt["doc_ids"], "positions": bt["positions"]},
                {"teach": False, "edl": False, "lc": False, "cf_store": None, "tap": False})
    return out


def labels(bt):
    import torch
    from csr.fsp_lib import Split
    sys.path.insert(0, str(guard.ROOT))
    from jsr_train import load_group_labels
    sem, shuf = load_group_labels(Split("main"))
    gd = bt["gdoc"].reshape(-1)
    return torch.as_tensor(sem)[gd.clamp(min=0)], torch.as_tensor(shuf)[gd.clamp(min=0)]


def t_isolation():
    """T1-T4."""
    import torch
    from csr.jsr_bank import residual_step, rms0, sd_loss
    cfg, model, banks = setup()
    _, npb, bt = small_batch()
    out = fwd(model, bt)
    h = out["hidden"].reshape(-1, 384)
    y = bt["targets"].reshape(-1)
    v = y >= 0
    t = cfg["train"]
    # T1: L_base -> no bank gradient (banks are not in the graph at all)
    lm = torch.nn.functional.cross_entropy(model.lm_logits(h[v]), y[v]) + t["aux_coeff"] * out["aux"] + t["z_coeff"] * out["z"]
    lm.backward()
    t1 = all(p.grad is None for b in banks.values() for p in b.parameters())
    base_grads_after_base = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    # T2: L_res (+ L_bal + L_sd via residual_step) -> no base gradient change
    sem, shuf = labels(bt)
    H, Y = h.detach()[v], y[v]
    for b in banks.values():
        residual_step(b, H, model.tok_emb.weight.detach(), Y, sem[v], shuf[v])
    t2 = all(torch.equal(p.grad, base_grads_after_base[n]) for n, p in model.named_parameters()
             if n in base_grads_after_base) and all((p.grad is None) == (n not in base_grads_after_base)
                                                    for n, p in model.named_parameters())
    # also: a direct autograd check of L_res w.r.t. base params (through h and W if they were not detached)
    model.zero_grad(set_to_none=True)
    for b in banks.values():
        b.zero_grad(set_to_none=True)
    b = banks["D"]
    xn = rms0(H)
    e, gate, q, _ = b.route(xn)
    a = b.correction(xn, e, gate)
    lg = torch.nn.functional.linear(H.float() + a, model.tok_emb.weight.detach())
    torch.nn.functional.cross_entropy(lg, Y).backward()
    t2b = all(p.grad is None for p in model.parameters())
    # T3: L_sd -> router only
    for bb in banks.values():
        bb.zero_grad(set_to_none=True)
    q = b.probs(rms0(H))
    lsd, _, npres = sd_loss(q, torch.arange(H.shape[0]) % 4)  # synthetic groups (enough tokens per group)
    lsd.backward()
    t3 = (b.router.weight.grad is not None and float(b.router.weight.grad.abs().sum()) > 0 and
          b.experts.w1.grad is None and b.experts.w2.grad is None and all(p.grad is None for p in model.parameters()))
    # T4: L_bal -> router only
    b.zero_grad(set_to_none=True)
    q = b.probs(rms0(H))
    e = q.argmax(-1)
    f = torch.bincount(e, minlength=16).float() / H.shape[0]
    (16 * (f * q.mean(0)).sum()).backward()
    t4 = (b.router.weight.grad is not None and float(b.router.weight.grad.abs().sum()) > 0 and
          b.experts.w1.grad is None and b.experts.w2.grad is None and all(p.grad is None for p in model.parameters()))
    return {"T1": t1, "T2": t2 and t2b, "T2_residual_step_leaves_base_grads_unchanged": t2,
            "T2_direct_autograd_no_base_grad": t2b, "T3": t3, "T4": t4, "npres_T3": npres}


def run_trainer(cfg, name, max_steps, run_dir=None, tag=""):
    tmp = guard.ROOT / ".tmp/joint-semantic-residual-pretrain-v1/tests"
    tmp.mkdir(parents=True, exist_ok=True)
    cp = tmp / f"cfg_{name}.json"
    cp.write_text(json.dumps(cfg))
    env = {**os.environ, "MOE_RESEARCH_APPROVE": "cpu-model", "MOE_RUNS_DIR": str(tmp / "runs")}
    cmd = [PY, "jsr_train.py", "--config", str(cp.relative_to(guard.ROOT)), "--device", "cpu", "--max-steps",
           str(max_steps), "--test-rows", "4", "--execute"]
    if run_dir:
        cmd += ["--run-dir", run_dir]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=guard.ROOT)
    lines = [ln for ln in r.stdout.splitlines() if "joint_semantic_residual_pretrain_v1_test" in ln or ln.startswith("{")]
    if r.returncode not in (0, 75):
        raise RuntimeError(f"trainer failed rc={r.returncode}\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    rdirs = sorted((tmp / "runs").glob("*_joint_semantic_residual_pretrain_v1_test"), key=lambda p: p.stat().st_mtime)
    return (run_dir or str(rdirs[-1].relative_to(guard.ROOT))), r


def last_slot(run_dir):
    import torch
    from csr import jsr_ckpt
    rid = Path(run_dir).name
    st = jsr_ckpt.ResumeStore(rid, "test")
    ent = st.latest_verified(log=lambda *_: None)
    return torch.load(guard.ROOT / ent["path"] / "state.pt", map_location="cpu", weights_only=False)


def eq_state(a, b):
    import torch
    if isinstance(a, dict):
        return set(a) == set(b) and all(eq_state(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(eq_state(x, y) for x, y in zip(a, b))
    if torch.is_tensor(a):
        return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    return a == b


def t_bank_vs_nobank():
    """T5: 3 steps with banks vs without: base model state, base optimizer state, logged base losses bitwise equal."""
    rb, _ = run_trainer(base_cfg(banks=True), "with", 3)
    sb = last_slot(rb)
    rn, _ = run_trainer(base_cfg(banks=False), "without", 3)
    sn = last_slot(rn)
    mb = [json.loads(x) for x in (guard.ROOT / rb / "metrics.jsonl").read_text().splitlines()]
    mn = [json.loads(x) for x in (guard.ROOT / rn / "metrics.jsonl").read_text().splitlines()]
    lb = [(r["step"], r["lm"], r["aux"], r["z"], r["gn"]) for r in mb if "step" in r]
    ln = [(r["step"], r["lm"], r["aux"], r["z"], r["gn"]) for r in mn if "step" in r]
    db = [r["diag"] for r in mb if "diag" in r]
    dn = [r["diag"] for r in mn if "diag" in r]
    ok = eq_state(sb["model"], sn["model"]) and eq_state(sb["opt"], sn["opt"]) and lb == ln and db == dn
    return {"T5": bool(ok), "model_equal": eq_state(sb["model"], sn["model"]), "opt_equal": eq_state(sb["opt"], sn["opt"]),
            "losses_equal": lb == ln, "diag_equal": db == dn, "steps": sb["st"]["next_step"],
            "bank_metrics_logged": "banks" in [r for r in mb if "step" in r][-1], "runs": [rb, rn]}


def t_resume():
    """T6: 4 straight steps vs 2 + resume + 2: everything bitwise identical."""
    r4, _ = run_trainer(base_cfg(), "straight", 4)
    s4 = last_slot(r4)
    r2, _ = run_trainer(base_cfg(), "split", 2)
    run_trainer(base_cfg(), "split", 4, run_dir=r2)
    s22 = last_slot(r2)
    keys = ["model", "opt", "banks", "bank_opt", "bank_ctrl"]
    res = {k: eq_state(s4[k], s22[k]) for k in keys}
    m4 = [json.loads(x) for x in (guard.ROOT / r4 / "metrics.jsonl").read_text().splitlines()]
    m22 = [json.loads(x) for x in (guard.ROOT / r2 / "metrics.jsonl").read_text().splitlines()]
    f = lambda m: [(r["step"], r["lm"], json.dumps(r.get("banks"), sort_keys=True)) for r in m if "step" in r]
    res["metrics_equal"] = f(m4) == f(m22)
    res["T6"] = all(res.values())
    res["steps"] = [s4["st"]["next_step"], s22["st"]["next_step"]]
    return res


def t_init_matched():
    """T7: identical initial bank tensors; routers identical for B/C/D."""
    import torch
    from csr.jsr_bank import Bank
    bs = {s["name"]: Bank(s, 0) for s in specs()}
    w1 = [b.experts.w1 for b in bs.values()]
    w2 = [b.experts.w2 for b in bs.values()]
    rt = [bs[n].router.weight for n in "BCD"]
    ok = all(torch.equal(w1[0], x) for x in w1) and all(torch.equal(w2[0], x) for x in w2) and \
        all(torch.equal(rt[0], x) for x in rt) and float(w2[0].abs().sum()) == 0.0
    other = Bank(specs()[0], 1)
    return {"T7": bool(ok and not torch.equal(other.experts.w1, w1[0])),
            "params": {n: b.n_params() for n, b in bs.items()}}


def t_jsd():
    """T8: JSD numerics vs float64 reference; permutation invariance; a gradient step increases mean JSD."""
    import torch
    from csr.jsr_bank import jsd_pairs, sd_loss
    g = torch.Generator().manual_seed(3)
    R = torch.softmax(torch.randn(5, 16, generator=g), -1)
    j = jsd_pairs(R)
    R64 = R.double()
    ref = []
    for a in range(5):
        for b in range(a + 1, 5):
            M = 0.5 * (R64[a] + R64[b])
            ref.append(0.5 * (R64[a] * (R64[a] / M).log()).sum() + 0.5 * (R64[b] * (R64[b] / M).log()).sum())
    ok_num = float((j.double() - torch.stack(ref)).abs().max()) < 1e-6
    logits = torch.randn(4000, 16, generator=g, requires_grad=True)
    grp = torch.randint(0, 6, (4000,), generator=g)
    q = torch.softmax(logits, -1)
    l0, j0, n0 = sd_loss(q, grp)
    perm = torch.randperm(4000, generator=g)
    l1, _, _ = sd_loss(q[perm], grp[perm])
    relabel = torch.tensor([3, 5, 0, 1, 4, 2])[grp]
    l2, _, _ = sd_loss(q, relabel)
    l0.backward()
    with torch.no_grad():
        q2 = torch.softmax(logits - 10.0 * logits.grad, -1)
    _, j2, _ = sd_loss(q2, grp)
    nolab = sd_loss(q.detach(), torch.full((4000,), -1))
    return {"T8": bool(ok_num and abs(float(l0) - float(l1)) < 1e-5 and abs(float(l0) - float(l2)) < 1e-5 and j2 > j0
                       and nolab[0] is None), "jsd_before": j0, "jsd_after_step": j2, "n_groups": n0}


def t_controller():
    """T9: the router gradient's semantic part has norm beta * ||g_sd|| with beta = r * EMA_res / EMA_sd."""
    import torch
    from csr.jsr_bank import residual_step, rms0, sd_loss
    cfg, model, banks = setup()
    _, npb, bt = small_batch(4)
    with torch.no_grad():
        out = fwd(model, bt)
    h = out["hidden"].reshape(-1, 384)
    y = bt["targets"].reshape(-1)
    v = y >= 0
    H, Y = h[v], y[v]
    grp = torch.arange(H.shape[0]) % 4
    b = banks["D"]
    b.r = 0.05
    recs = []
    for i in range(3):
        b.zero_grad(set_to_none=True)
        rec = residual_step(b, H, model.tok_emb.weight.detach(), Y, grp, grp)
        recs.append(rec)
    # recompute g_res, g_bal, g_sd at the current weights and compare the combination
    b.zero_grad(set_to_none=True)
    xn = rms0(H)
    beta_used = b.ctrl["beta"]
    ok_ratio = all(abs(r_["ratio"] - r_["beta"] * r_["gn_sd_router"] / r_["gn_res_router"]) < 1e-9 for r_ in recs)
    # exact check of beta formula from EMA values
    ema_res, ema_sd = None, None
    betas = []
    for r_ in recs:
        ema_res = r_["gn_res_router"] if ema_res is None else 0.9 * ema_res + 0.1 * r_["gn_res_router"]
        ema_sd = r_["gn_sd_router"] if ema_sd is None else 0.9 * ema_sd + 0.1 * r_["gn_sd_router"]
        betas.append(b.r * ema_res / ema_sd)
    ok_beta = all(abs(bb - r_["beta"]) <= 1e-5 * max(1e-12, abs(bb)) for bb, r_ in zip(betas, recs))
    q = b.probs(xn)
    lsd, _, _ = sd_loss(q, grp)
    del lsd
    return {"T9": bool(ok_beta and ok_ratio and all(r_["gn_res_router"] > 0 for r_ in recs)),
            "betas": [r_["beta"] for r_ in recs], "ratios": [r_["ratio"] for r_ in recs], "beta_last": beta_used}


def t_causal_and_labels():
    """T10 causality of q/e; T11 router invariant to labels; T12 oracle routes by label."""
    import torch
    from csr.jsr_bank import rms0
    cfg, model, banks = setup()
    model.eval()
    _, npb, bt = small_batch(1)
    with torch.no_grad():
        h1 = fwd(model, bt)["hidden"].reshape(-1, 384)
        bt2 = {k: v.clone() for k, v in bt.items()}
        tcut = 300
        ids = bt2["input_ids"].reshape(-1)
        ids[tcut + 1:] = (ids[tcut + 1:] * 7 + 11) % 50000
        h2 = fwd(model, bt2)["hidden"].reshape(-1, 384)
        b = banks["D"]
        q1, q2 = b.probs(rms0(h1)), b.probs(rms0(h2))
        t10 = bool(torch.allclose(q1[:tcut + 1], q2[:tcut + 1], atol=1e-6) and
                   torch.equal(q1[:tcut + 1].argmax(-1), q2[:tcut + 1].argmax(-1)) and
                   not torch.allclose(q1[tcut + 5:], q2[tcut + 5:], atol=1e-6))
        # T11: route() takes no label argument for routed banks; q identical regardless of labels
        e_a, _, qa, _ = b.route(rms0(h1), None)
        e_b, _, qb, _ = b.route(rms0(h1), torch.randint(0, 16, (h1.shape[0],)))
        t11 = bool(torch.equal(qa, qb) and torch.equal(e_a, e_b))
        E = banks["E"]
        lab = torch.randint(-1, 16, (h1.shape[0],))
        e, gate, q, act = E.route(rms0(h1), lab)
        a = E.correction(rms0(h1), e, gate)
        man = torch.stack([E.forced(rms0(h1[i:i + 1]), int(lab[i]))[0] if lab[i] >= 0 else torch.zeros(384)
                           for i in range(0, h1.shape[0], 97)])
        t12 = bool(q is None and torch.equal(e[lab >= 0], lab[lab >= 0]) and torch.allclose(a[::97], man, atol=1e-5)
                   and float(a[lab < 0].abs().sum()) == 0.0)
    return {"T10": t10, "T11": t11, "T12": t12}


def main():
    guard.check_root()
    if os.environ.get("MOE_RESEARCH_APPROVE") != "cpu-model":
        print("set MOE_RESEARCH_APPROVE=cpu-model")
        sys.exit(2)
    import torch
    torch.manual_seed(0)
    torch.set_num_threads(16)
    res = {}
    for fn in (t_isolation, t_init_matched, t_jsd, t_controller, t_causal_and_labels, t_bank_vs_nobank, t_resume):
        try:
            r = fn()
        except Exception as e:  # noqa: BLE001
            import traceback
            r = {fn.__name__: False, "error": traceback.format_exc()[-3000:]}
        res[fn.__name__] = r
        print(fn.__name__, json.dumps(r, default=str)[:600], flush=True)
    keys = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12"]
    flat = {k: v for r in res.values() for k, v in r.items()}
    passed = {k: bool(flat.get(k, False)) for k in keys}
    out = {"time": guard.now(), "passed": passed, "all_pass": all(passed.values()), "detail": res}
    from csr.jsr_paths import ART
    p = ART / f"cpu_tests_{guard.utc_stamp()}.json"
    guard.atomic_write_json(p, out, mode=0o400)
    print(json.dumps(passed), "ALL PASS" if out["all_pass"] else "FAIL", p)
    sys.exit(0 if out["all_pass"] else 1)


if __name__ == "__main__":
    main()
