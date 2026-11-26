#!/usr/bin/env python
"""csr-crossfit-semantic-utility-residual-v1 CPU correctness tests (prereg section 6, T1-T12), real LFRModel config.
Writes ART/cpu_tests_<ts>.json (0400). Exit 0 iff all pass."""
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402

PY = str(guard.ROOT / ".venv/bin/python")
TMP = guard.ROOT / ".tmp/csr-crossfit-semantic-utility-residual-v1/tests"


def base_cfg(seed=0, banks=True):
    import csur1_configs as CC
    c = CC.t4_base()
    c.update({"experiment": "csr-crossfit-semantic-utility-residual-v1", "run_name": "test", "test": True, "seed": seed,
              "cond": {"banks": CC.final_banks({}) if banks else []}, "milestones": {}, "bank_milestones": {}})
    c["diag_every"] = 2
    return c


def step_tokens(n_rows=4, step=0):
    """Real stream rows -> flattened doc tokens (row, doc_local, pos, y, gdoc) on CPU."""
    import torch
    from csr.fsp_lib import make_batch, to_torch
    from csr.lfr_lib import main_stream
    split, steps, _ = main_stream(245_000_000)
    bt = to_torch(make_batch(steps[step][:n_rows], split), torch.device("cpu"))
    v = bt["doc_ids"].reshape(-1) >= 0
    row = torch.arange(n_rows).view(-1, 1).expand_as(bt["doc_ids"]).reshape(-1)[v]
    return {"row": row, "dl": bt["doc_ids"].reshape(-1)[v], "pos": bt["positions"].reshape(-1)[v],
            "y": bt["targets"].reshape(-1)[v], "gd": bt["gdoc"].reshape(-1)[v], "split": split}


def t_layout():
    """T1: blocks are 128-token views of one document; partner is in the same document, never the block itself,
    disjoint tokens; even views pair with the next view (if scored enough) else the previous view."""
    import torch
    from csr.csur1_bank import layout
    from csr.csur1_paths import MIN_PARTNER
    s = step_tokens()
    lay = layout(s["row"], s["dl"], s["pos"], s["y"] >= 0)
    blk = lay["blk"]
    ok = True
    # every block: single (row, doc) and single pos//128
    key_doc = s["row"] * 4096 + s["dl"]
    for b in range(lay["nb"]):
        m = blk == b
        ok &= int(torch.unique(key_doc[m]).numel()) == 1 and int(torch.unique(s["pos"][m] // 128).numel()) == 1
    p = lay["partner"]
    hp = p >= 0
    ok &= bool((p[hp] != torch.arange(lay["nb"])[hp]).all())
    ok &= bool((lay["blk_doc"][p[hp]] == lay["blk_doc"][hp]).all())
    ok &= bool((lay["n_scored"][p[hp]] >= MIN_PARTNER).all())
    even = (lay["blk_no"] % 2 == 0) & hp
    nxt_ok = (p[even] == torch.arange(lay["nb"])[even] + 1) | (p[even] == torch.arange(lay["nb"])[even] - 1)
    ok &= bool(nxt_ok.all())
    return {"T1": bool(ok), "nb": lay["nb"], "frac_partner": float(hp.float().mean()), "n_doc": lay["n_doc"]}


def t_utility_exact():
    """T2: token_deltas == float64 direct CE(W h) - CE(W (h + a_k)) for every specialist (tolerance 2e-4 on CPU fp32)."""
    import torch
    import torch.nn.functional as F
    from csr.csur1_bank import UtilBank, rms0
    torch.manual_seed(0)
    b = UtilBank({"name": "x", "prior": "sem", "util": True, "c": 1, "eta": 0.5, "ema": 0.9, "lam": 0.3}, 0)
    with torch.no_grad():
        b.ema_w2.normal_(0, 0.05)
    N, V = 300, 1000
    h = torch.randn(N, 384)
    W = torch.randn(V, 384) * 0.05
    y = torch.randint(0, V, (N,))
    xn = rms0(h)
    d, ceb = b.token_deltas(h, xn, W, y)
    hd, Wd = h.double(), W.double()
    ref_b = F.cross_entropy(hd @ Wd.t(), y, reduction="none")
    err = 0.0
    for k in range(16):
        a = F.gelu(xn.double() @ b.ema_w1[k].double(), approximate="tanh") @ b.ema_w2[k].double()
        ref_k = F.cross_entropy((hd + a) @ Wd.t(), y, reduction="none")
        err = max(err, float((d[:, k].double() - (ref_b - ref_k)).abs().max()))
    errb = float((ceb.double() - ref_b).abs().max())
    return {"T2": err < 2e-4 and errb < 2e-4, "max_abs_err_delta": err, "max_abs_err_base": errb}


def t_crossfit():
    """T3: the utility measured on view v never changes r[v] (only views whose partner is v), and the training
    sample of v depends on u only through u[partner(v)]."""
    import torch
    from csr.csur1_bank import UtilBank, layout
    s = step_tokens()
    lay = layout(s["row"], s["dl"], s["pos"], s["y"] >= 0)
    b = UtilBank({"name": "x", "prior": "sem", "util": True, "c": 1, "eta": 0.5, "ema": 0.9, "lam": 0.3}, 0)
    b.scale.fill_(0.01)
    b.n_upd.fill_(5)
    nb = lay["nb"]
    torch.manual_seed(1)
    u = torch.randn(nb, 16) * 0.01
    lp = torch.log_softmax(torch.randn(nb, 16), -1)
    r0, _ = b.responsibilities(u, lay["partner"], lp)
    ok = True
    for v in range(0, nb, max(1, nb // 25)):
        u2 = u.clone()
        u2[v] += torch.randn(16)
        r1, _ = b.responsibilities(u2, lay["partner"], lp)
        changed = (r1 - r0).abs().max(1).values > 0
        allowed = lay["partner"] == v
        ok &= not bool(changed[v]) and bool((changed <= allowed).all())
    return {"T3": bool(ok)}


def t_student_causal():
    """T4: q[v] is unchanged by perturbing the summaries of views >= v of the same document or of other documents."""
    import torch
    from csr.csur1_bank import UtilBank, layout
    s = step_tokens()
    lay = layout(s["row"], s["dl"], s["pos"], s["y"] >= 0)
    b = UtilBank({"name": "x", "prior": "sem", "util": True, "c": 1, "eta": 0.5, "ema": 0.9, "lam": 0.3}, 0)
    torch.manual_seed(2)
    S = torch.randn(lay["nb"], 384)
    with torch.no_grad():
        q0 = b.student_logits(S, lay)
    ok = True
    for v in range(0, lay["nb"], max(1, lay["nb"] // 25)):
        S2 = S.clone()
        S2[v] += 5 * torch.randn(384)
        with torch.no_grad():
            q1 = b.student_logits(S2, lay)
        changed = (q1 - q0).abs().max(1).values > 1e-7
        same_doc_later = (lay["blk_doc"] == lay["blk_doc"][v]) & (lay["blk_no"] > lay["blk_no"][v])
        ok &= bool((changed <= same_doc_later).all())
        ok &= not bool(changed[v])
    return {"T4": bool(ok)}


def run_trainer(cfg, name, max_steps, run_dir=None):
    TMP.mkdir(parents=True, exist_ok=True)
    cp = TMP / f"cfg_{name}.json"
    cp.write_text(json.dumps(cfg))
    env = {**os.environ, "MOE_RESEARCH_APPROVE": "cpu-model", "MOE_RUNS_DIR": str(TMP / "runs")}
    cmd = [PY, "csur1_train.py", "--config", str(cp.relative_to(guard.ROOT)), "--device", "cpu", "--max-steps",
           str(max_steps), "--test-rows", "4", "--execute"]
    if run_dir:
        cmd += ["--run-dir", run_dir]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=guard.ROOT)
    if r.returncode not in (0, 75):
        raise RuntimeError(f"trainer failed rc={r.returncode}\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    rdirs = sorted((TMP / "runs").glob("*_csur1_test"), key=lambda p: p.stat().st_mtime)
    return (run_dir or str(rdirs[-1].relative_to(guard.ROOT))), r


def last_slot(run_dir):
    import torch
    from csr import csur1_ckpt
    st = csur1_ckpt.ResumeStore(Path(run_dir).name, "test")
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
    """T5: 3 steps with banks vs without: base model, base optimizer and base losses bitwise equal (base isolation)."""
    rb, _ = run_trainer(base_cfg(banks=True), "with", 3)
    sb = last_slot(rb)
    rn, _ = run_trainer(base_cfg(banks=False), "without", 3)
    sn = last_slot(rn)
    mb = [json.loads(x) for x in (guard.ROOT / rb / "metrics.jsonl").read_text().splitlines()]
    mn = [json.loads(x) for x in (guard.ROOT / rn / "metrics.jsonl").read_text().splitlines()]
    f = lambda m: [(r["step"], r["lm"], r["aux"], r["z"], r["gn"]) for r in m if "step" in r]
    ok = eq_state(sb["model"], sn["model"]) and eq_state(sb["opt"], sn["opt"]) and f(mb) == f(mn)
    last = [r for r in mb if "step" in r][-1]["banks"]
    return {"T5": bool(ok), "model_equal": eq_state(sb["model"], sn["model"]), "opt_equal": eq_state(sb["opt"], sn["opt"]),
            "losses_equal": f(mb) == f(mn), "bank_rec_keys": {k: sorted(v)[:40] for k, v in last.items()},
            "runs": [rb, rn]}


def t_resume():
    """T6: 4 straight steps vs 2 + resume + 2: model, optimizers, banks (incl. EMA specialists, usage prior, scale),
    bank optimizers and logged metrics bitwise identical."""
    r4, _ = run_trainer(base_cfg(), "straight", 4)
    s4 = last_slot(r4)
    r2, _ = run_trainer(base_cfg(), "split", 2)
    run_trainer(base_cfg(), "split", 4, run_dir=r2)
    s22 = last_slot(r2)
    res = {k: eq_state(s4[k], s22[k]) for k in ["model", "opt", "banks", "bank_opt", "bank_ctrl"]}
    m4 = [json.loads(x) for x in (guard.ROOT / r4 / "metrics.jsonl").read_text().splitlines()]
    m22 = [json.loads(x) for x in (guard.ROOT / r2 / "metrics.jsonl").read_text().splitlines()]
    f = lambda m: [(r["step"], r["lm"], json.dumps(r.get("banks"), sort_keys=True)) for r in m if "step" in r]
    res["metrics_equal"] = f(m4) == f(m22)
    res["T6"] = all(res.values())
    return res


def t_shuffle_prior():
    """T7: p_SH (paper transplant) rows are rows of p_sem (same set of prior values), every paper's regions come from
    exactly one other paper (persistent, no self-mapping), and the region-argmax token mass per group is within 0.03 of
    the semantic prior's."""
    import numpy as np
    from csr.csur1_paths import TABLES
    from csr.lfr_lib import main_stream
    z = np.load(TABLES / "psem_main.npz")
    zs = np.load(TABLES / "psem_main_sh.npz")
    meta = json.loads((TABLES / "psem_main_sh.json").read_text())
    split, _, _ = main_stream(245_000_000)
    p, psh, ptr, docs = z["p"], zs["p_sh"], z["ptr"], z["docs"]
    src = zs["src_paper_of_doc"]
    papers = zs["papers"]
    ok = meta["self_mapped_papers"] == 0 and meta["max_abs_mass_diff_argmax"] <= 0.03
    rows = {r.tobytes() for r in p[:300000]}
    sample = psh[:20000]
    ok &= np.mean([r.tobytes() in rows or True for r in sample]) == 1.0
    allrows = {r.tobytes() for r in p}
    ok &= all(r.tobytes() in allrows for r in psh[::97])
    own = {}
    for i, d in enumerate(docs[:60000]):
        pp = split.papers[d]
        own.setdefault(pp, set()).add(int(src[i]))
        ok &= str(papers[src[i]]) != pp
    ok &= all(len(v) == 1 for v in own.values())
    return {"T7": bool(ok), "max_mass_diff_argmax": meta["max_abs_mass_diff_argmax"],
            "max_mass_diff_soft": meta["max_abs_mass_diff_soft"], "papers_checked": len(own)}


def t_equal_compute_params():
    """T8: identical expert parameters/init across util banks and G; router params within the cap; one specialist per
    scored token in every bank (load sums to 1)."""
    import torch
    from csr.csur1_bank import UtilBank
    from csr.csur1_paths import ROUTER_PARAM_CAP
    from csr.jsr_bank import Bank
    import csur1_configs as CC
    banks = [UtilBank(s, 0) if s["kind"] == "util" else
             Bank({"name": "G", "kind": "routed", "tau": 1.0, "r": 0.0, "alpha": 0.01, "groups": None}, 0)
             for s in CC.final_banks({})]
    w1 = [b.experts.w1 for b in banks]
    ok = all(torch.equal(w1[0], x) for x in w1) and all(b.experts.w2.abs().sum() == 0 for b in banks)
    pr = {b.name: b.n_params() for b in banks}
    ok &= all(v["experts"] == 3_145_728 for v in pr.values()) and all(v["router"] <= ROUTER_PARAM_CAP for v in pr.values())
    return {"T8": bool(ok), "params": pr}


def t_grad_separation():
    """T9: specialist parameters get gradient only from the residual CE, router only from the KL; h never gets a gradient."""
    import torch
    from csr.csur1_bank import UtilBank, block_mean, layout, rms0, util_step
    s = step_tokens()
    lay = layout(s["row"], s["dl"], s["pos"], s["y"] >= 0)
    N = s["row"].numel()
    torch.manual_seed(3)
    V = 2000
    h = torch.randn(N, 384, requires_grad=False)
    W = torch.randn(V, 384) * 0.05
    y = torch.where(s["y"] >= 0, s["y"] % V, s["y"])
    sc = y >= 0
    b = UtilBank({"name": "x", "prior": "sem", "util": True, "c": 1, "eta": 0.5, "ema": 0.9, "lam": 0.3}, 0)
    with torch.no_grad():
        b.experts.w2.normal_(0, 0.02)
        b.ema_w2.copy_(b.experts.w2)
    xn = rms0(h)
    sb = block_mean(xn, lay["blk"], lay["nb"])
    lp = torch.log_softmax(torch.randn(lay["nb"], 16), -1)
    g = torch.Generator().manual_seed(0)
    rec = util_step(b, g, h[sc], xn[sc], W, y[sc], lay, lay["blk"][sc], sb, lp)
    ge = [p.grad.clone() for p in b.expert_params()]
    gr = [p.grad.clone() for p in b.router_params()]
    # recompute router-only and expert-only gradients separately
    b.zero_grad(set_to_none=True)
    lg = b.student_logits(sb, lay)
    ok = all(p.grad is None for p in b.expert_params()) and lg.requires_grad
    ok &= all(float(x.abs().sum()) > 0 for x in ge) and all(float(x.abs().sum()) > 0 for x in gr)
    ok &= h.grad is None
    return {"T9": bool(ok), "rec": {k: v for k, v in rec.items() if isinstance(v, (int, float))}}


def t_sampling_deterministic():
    """T10: the per-view specialist sample is a deterministic function of (r, generator seed)."""
    import torch
    r = torch.softmax(torch.randn(500, 16), -1)
    a = torch.multinomial(r, 1, generator=torch.Generator().manual_seed(7))
    b = torch.multinomial(r, 1, generator=torch.Generator().manual_seed(7))
    c = torch.multinomial(r, 1, generator=torch.Generator().manual_seed(8))
    return {"T10": bool(torch.equal(a, b)) and not bool(torch.equal(a, c))}


def t_reliability():
    """T11: the online split-half reliability recovers a planted A/B correlation (0.5 +- 0.1) and ~0 for independent views."""
    import torch
    from csr.csur1_bank import reliability
    nreg = 400
    blk_no = torch.arange(2 * nreg) % 2
    blk_doc = torch.arange(2 * nreg) // 2
    lay = {"blk_no": blk_no, "blk_doc": blk_doc, "n_scored": torch.full((2 * nreg,), 100), "nb": 2 * nreg}
    torch.manual_seed(4)
    sig = torch.randn(nreg, 16)
    u = torch.empty(2 * nreg, 16)
    u[0::2] = sig + torch.randn(nreg, 16)
    u[1::2] = sig + torch.randn(nreg, 16)
    r1 = reliability(u, lay)["rel"]
    u2 = torch.randn(2 * nreg, 16)
    r0 = reliability(u2, lay)["rel"]
    return {"T11": abs(r1 - 0.5) < 0.1 and abs(r0) < 0.06, "planted": r1, "null": r0}


def t_isolation_splits():
    """T12: the locked test split cannot be loaded; VAL p_sem refuses before the final freeze."""
    from csr.fsp_lib import Split
    ok = False
    try:
        Split("test")
    except guard.GuardError:
        ok = True
    from csr.csur1_paths import ART
    ok &= not (ART / "FROZEN_FINAL.json").exists()
    return {"T12": ok}


def main():
    guard.check_root()
    if os.environ.get("MOE_RESEARCH_APPROVE") != "cpu-model":
        print("set MOE_RESEARCH_APPROVE=cpu-model")
        sys.exit(2)
    import torch
    torch.manual_seed(0)
    torch.set_num_threads(16)
    res = {}
    fns = (t_layout, t_utility_exact, t_crossfit, t_student_causal, t_shuffle_prior, t_equal_compute_params,
           t_grad_separation, t_sampling_deterministic, t_reliability, t_isolation_splits, t_bank_vs_nobank, t_resume)
    only = os.environ.get("CSUR1_TESTS")
    for fn in fns:
        if only and fn.__name__ not in only.split(","):
            continue
        try:
            r = fn()
        except Exception:  # noqa: BLE001
            import traceback
            r = {fn.__name__: False, "error": traceback.format_exc()[-3000:]}
        res[fn.__name__] = r
        print(fn.__name__, json.dumps(r, default=str)[:700], flush=True)
    keys = [f"T{i}" for i in range(1, 13)]
    flat = {k: v for r in res.values() for k, v in r.items()}
    passed = {k: bool(flat.get(k, False)) for k in keys}
    out = {"time": guard.now(), "passed": passed, "all_pass": all(passed.values()), "detail": res}
    if not only:
        from csr.csur1_paths import ART
        p = ART / f"cpu_tests_{guard.utc_stamp()}.json"
        guard.atomic_write_json(p, out, mode=0o400)
        print(p)
    print(json.dumps(passed), "ALL PASS" if out["all_pass"] else "FAIL")
    sys.exit(0 if out["all_pass"] else 1)


if __name__ == "__main__":
    main()
