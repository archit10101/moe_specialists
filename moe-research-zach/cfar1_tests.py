#!/usr/bin/env python
"""csr-centered-fedavg-residual-v1 CPU tests (federation invariants + trainer isolation/resume). Writes ART/cpu_tests_<ts>.json."""
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402

PY = str(guard.ROOT / ".venv/bin/python")
TMP = guard.ROOT / ".tmp/csr-centered-fedavg-residual-v1/tests"


def mk(mode, wp=192, T=2):
    from csr.cfar1_bank import FedBank
    return FedBank({"name": "x", "kind": "fed", "prior": "sem", "mode": mode, "T": T, "w_p": wp}, 0)


def perturb_and_opt(b):
    import torch
    opt = torch.optim.AdamW(b.parameters(), lr=1e-3)
    torch.manual_seed(5)
    for p in (b.experts.w1, b.experts.w2):
        p.grad = torch.randn_like(p)
    opt.step()
    b.mass.copy_(torch.rand(16) * 100 + 1)
    return opt


def t_centered():
    """C1: centered sync keeps personalised units bitwise (function continuity on them), S <- S + weighted mean update,
    shared-only units of every specialist = S afterwards; moments of reset slices = weighted mean, personalised kept."""
    import torch
    b = mk("centered")
    opt = perturb_and_opt(b)
    W1b, W2b = b.experts.w1.detach().clone(), b.experts.w2.detach().clone()
    A1, S1 = b.A1.clone(), b.S1.clone()
    w = b.mass / b.mass.sum()
    m_before = opt.state[b.experts.w1]["exp_avg"].clone()
    rec = b.sync(opt)
    P = slice(0, b.w_p)
    sh = slice(b.w_p, 256)
    ok = torch.equal(b.experts.w1[:, :, P], W1b[:, :, P]) and torch.equal(b.experts.w2[:, P, :], W2b[:, P, :])
    dbar = torch.einsum("k,kdf->df", w, W1b - A1)
    ok &= torch.allclose(b.S1, S1 + dbar, atol=1e-6)
    ok &= bool((b.experts.w1[:, :, sh] == b.S1[:, sh].unsqueeze(0)).all())
    m = opt.state[b.experts.w1]["exp_avg"]
    ok &= torch.equal(m[:, :, P], m_before[:, :, P])
    ok &= torch.allclose(m[:, :, sh], torch.einsum("k,kdf->df", w, m_before[:, :, sh]).unsqueeze(0).expand_as(m[:, :, sh]), atol=1e-7)
    ok &= rec["cont_max_abs_personalised"] == 0.0 and float(b.mass.sum()) == 0.0
    return {"C1": bool(ok), "rec": rec}


def t_fedavg():
    """C2: FedAvg starts with identical specialists; after sync every specialist = anchor + token-weighted mean update."""
    import torch
    b = mk("fedavg")
    ok = bool((b.experts.w1 == b.experts.w1[0:1]).all())
    opt = perturb_and_opt(b)
    W1b, A1 = b.experts.w1.detach().clone(), b.A1.clone()
    w = b.mass / b.mass.sum()
    b.sync(opt)
    tgt = A1[0] + torch.einsum("k,kdf->df", w, W1b - A1)
    ok &= bool(torch.allclose(b.experts.w1, tgt.unsqueeze(0).expand_as(b.experts.w1), atol=1e-6))
    ok &= bool((b.experts.w1 == b.experts.w1[0:1]).all())
    return {"C2": ok}


def t_local_and_params():
    """C3: local mode never changes on sync; parameter accounting (training-unique and deployable) for every split;
    w_p = 240 is exactly the csur1 expert budget (3,145,728) with identical per-token FLOPs (width 256)."""
    import torch
    b = mk("local")
    opt = perturb_and_opt(b)
    W = b.experts.w1.detach().clone()
    ok = b.sync(opt) is None and torch.equal(W, b.experts.w1)
    pr = {wp: mk("centered", wp).n_params() for wp in (240, 192, 128)}
    ok &= pr[240]["experts_training_unique"] == 3_145_728
    return {"C3": bool(ok), "params": pr}


def t_hard_groups():
    """C4: responsibilities are one-hot on argmax of the region prior; no utility is computed."""
    import torch
    b = mk("centered")
    lp = torch.log_softmax(torch.randn(50, 16), -1)
    r, _ = b.responsibilities(None, torch.full((50,), -1), lp, update=False)
    return {"C4": bool(torch.equal(r.argmax(1), lp.argmax(1))) and bool((r.sum(1) == 1).all()) and not b.util}


def base_cfg(banks=True):
    import cfar1_configs as CC
    c = CC.t4_base()
    fed = CC.fed
    bk = [{"name": "G", "kind": "token"}, fed("F2s", "sem", "centered", 2, 192), fed("F2h", "shuf", "centered", 2, 192),
          fed("F1s", "sem", "fedavg", 2), fed("L", "sem", "local")]
    c.update({"experiment": "csr-centered-fedavg-residual-v1", "run_name": "test", "test": True, "seed": 0,
              "cond": {"banks": bk if banks else []}, "milestones": {}, "bank_milestones": {}})
    c["diag_every"] = 2
    return c


def run_trainer(cfg, name, max_steps, run_dir=None):
    TMP.mkdir(parents=True, exist_ok=True)
    cp = TMP / f"cfg_{name}.json"
    cp.write_text(json.dumps(cfg))
    env = {**os.environ, "MOE_RESEARCH_APPROVE": "cpu-model", "MOE_RUNS_DIR": str(TMP / "runs")}
    cmd = [PY, "cfar1_train.py", "--config", str(cp.relative_to(guard.ROOT)), "--device", "cpu", "--max-steps",
           str(max_steps), "--test-rows", "4", "--execute"]
    if run_dir:
        cmd += ["--run-dir", run_dir]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=guard.ROOT)
    if r.returncode not in (0, 75):
        raise RuntimeError(f"trainer failed rc={r.returncode}\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    rdirs = sorted((TMP / "runs").glob("*_cfar1_test"), key=lambda p: p.stat().st_mtime)
    return run_dir or str(rdirs[-1].relative_to(guard.ROOT))


def last_slot(run_dir):
    import torch
    from csr import cfar1_ckpt
    ent = cfar1_ckpt.ResumeStore(Path(run_dir).name, "test").latest_verified(log=lambda *_: None)
    return torch.load(guard.ROOT / ent["path"] / "state.pt", map_location="cpu", weights_only=False)


def t_trainer():
    """C5: base bitwise identical with/without fed banks (3 steps); C6: 4 straight steps vs 2 + resume + 2 bitwise
    (syncs at steps 1 and 3, i.e. on both sides of the resume point), including S/anchor/mass buffers and moments."""
    from csur1_tests import eq_state
    rb = run_trainer(base_cfg(True), "with", 3)
    rn = run_trainer(base_cfg(False), "without", 3)
    sb, sn = last_slot(rb), last_slot(rn)
    c5 = eq_state(sb["model"], sn["model"]) and eq_state(sb["opt"], sn["opt"])
    r4 = run_trainer(base_cfg(True), "straight", 4)
    r2 = run_trainer(base_cfg(True), "split", 2)
    run_trainer(base_cfg(True), "split", 4, run_dir=r2)
    s4, s22 = last_slot(r4), last_slot(r2)
    c6 = all(eq_state(s4[k], s22[k]) for k in ("model", "opt", "banks", "bank_opt"))
    m = [json.loads(x) for x in (guard.ROOT / r4 / "metrics.jsonl").read_text().splitlines()]
    syncs = [x for x in m if "sync" in x]
    return {"C5": bool(c5), "C6": bool(c6) and len(syncs) >= 6, "n_sync_records": len(syncs),
            "sync_example": syncs[0] if syncs else None}


def main():
    guard.check_root()
    if os.environ.get("MOE_RESEARCH_APPROVE") != "cpu-model":
        print("set MOE_RESEARCH_APPROVE=cpu-model")
        sys.exit(2)
    import torch
    from csr.cfar1_paths import ART as _ART
    _ART.mkdir(parents=True, exist_ok=True)   # checkpoint ledger lives in ART
    torch.set_num_threads(16)
    res = {}
    only = os.environ.get("CFAR1_TESTS")
    for fn in (t_centered, t_fedavg, t_local_and_params, t_hard_groups, t_trainer):
        if only and fn.__name__ not in only.split(","):
            continue
        try:
            r = fn()
        except Exception:  # noqa: BLE001
            import traceback
            r = {fn.__name__: False, "error": traceback.format_exc()[-3000:]}
        res[fn.__name__] = r
        print(fn.__name__, json.dumps(r, default=str)[:700], flush=True)
    keys = [f"C{i}" for i in range(1, 7)]
    flat = {k: v for r in res.values() for k, v in r.items()}
    passed = {k: bool(flat.get(k, False)) for k in keys}
    out = {"time": guard.now(), "passed": passed, "all_pass": all(passed.values()), "detail": res}
    if not only:
        from csr.cfar1_paths import ART
        ART.mkdir(parents=True, exist_ok=True)
        p = ART / f"cpu_tests_{guard.utc_stamp()}.json"
        guard.atomic_write_json(p, out, mode=0o400)
        print(p)
    print(json.dumps(passed), "ALL PASS" if out["all_pass"] else "FAIL")
    sys.exit(0 if out["all_pass"] else 1)


if __name__ == "__main__":
    main()
