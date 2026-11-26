#!/usr/bin/env python
"""spb1 CPU correctness tests (Stage 0). No calibration result is inspected: the smoke runs use 2 TRAIN / 2 SEL rows and
4 plan steps on CPU; all smoke outputs live in TMP/tests and are deleted (logged) when every test passes.

T1  causality: deployed router and causal semantic-posterior net outputs at t unchanged when inputs after t change;
    frozen-base residual x_t unchanged when tokens after t (same document) change
T2  target isolation: router / posterior inputs are built from base_forward (input_ids, doc_ids, positions only);
    deploy() and posterior() never read targets or labels
T3  pair routing: SEM/SHUF training routes always stay inside the token's cluster M-tuple; GEN is unrestricted
T4  unrestricted deployment: for every specialist j there is a router output making EVERY token select j; no mask /
    bias / mapping code in the deployed router path
T5  parameter and active-FLOP matching: identical bank + router parameter counts for SEM/SHUF/GEN; one specialist per
    token (dispatch topi width 1); identical deployed-router parameters across banks
T6  G-ZERO: at initialisation (W2 = 0) every bank reproduces the base CE (step-1 metrics)
T7  resume determinism: 4 steps straight == 1 step + leg end + resume to 4 (bitwise bank and optimizer state)
T8  extraction consistency: table lookup at the training route == directly routed CE; common/bank shapes aligned
T9  balanced persistent shuffle: share error <= 1e-4 (50M, 245M, CS halves), paper NMI < 0.01, constant per paper
T10 locks: VAL refused before FROZEN_FINAL.json; no test-split / CALIB-SELECTION reference in spb1 code
T11 functional metric: recovers a planted own-cluster effect and returns ~0 without one (synthetic)
T12 smoke: router job (all bank types, posterior ablation) + stage-1 / stage-2 analysis end to end"""
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402

PY = str(guard.ROOT / ".venv/bin/python")


def sh(args, ok=(0,), env_extra=None):
    env = dict(os.environ)
    env["MOE_RESEARCH_APPROVE"] = "cpu-model"
    if env_extra:
        env.update(env_extra)
    r = subprocess.run([PY] + args, cwd=guard.ROOT, env=env, capture_output=True, text=True)
    if r.returncode not in ok:
        raise RuntimeError(f"{args} rc={r.returncode}\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    return r


def main():
    import numpy as np
    import torch
    from csr import spb1_log as LG
    from csr import spb1_paths as P
    from csr.spb1_lib import FreeRouter, PairBank, SubjectNet, make_banks
    guard.check_root()
    torch.manual_seed(0)
    res = {}
    K, M = P.K, 2
    E = K * M
    T = P.TMP / "tests"
    if T.exists():
        shutil.rmtree(T)
    T.mkdir(parents=True)
    rel = lambda p: str(Path(p).relative_to(guard.ROOT))
    # ---------------- T1 causality
    x = torch.randn(3, 40, 384)
    ok1 = True
    for net in (FreeRouter(64, 0, E), FreeRouter(64, 1, E, n_extra=16), SubjectNet(64, 0, K)):
        ex = torch.softmax(torch.randn(3, 40, 16), -1)
        o1 = net(x, ex) if isinstance(net, FreeRouter) else net(x)
        x2, ex2 = x.clone(), ex.clone()
        x2[:, 25:] += torch.randn_like(x2[:, 25:]) * 3
        ex2[:, 25:] = torch.softmax(torch.randn(3, 15, 16), -1)
        o2 = net(x2, ex2) if isinstance(net, FreeRouter) else net(x2)
        ok1 &= bool(torch.allclose(o1[:, :25], o2[:, :25], atol=1e-5)) and not bool(torch.allclose(o1[:, 25:], o2[:, 25:]))
    from csr.fsp_lib import Split
    from csr.orv1_lib import base_forward, load_base, make_batch, to_dev, unflatten_rows
    model, _ = load_base(0, torch.device("cpu"))
    cs = Split("calib_search")
    zp = np.load(P.ORV1_TABLES / "parts.npz")
    rows = unflatten_rows(zp["EVAL_docs"], zp["EVAL_row_len"])[:1]
    bt = to_dev(make_batch(rows, cs), torch.device("cpu"))
    xa, _ = base_forward(model, bt)
    did = bt["doc_ids"][0]
    L0 = int((did == 0).sum())
    cut = L0 // 2
    bt2 = {k: v.clone() for k, v in bt.items()}
    bt2["input_ids"][0, cut + 1:L0] = (bt2["input_ids"][0, cut + 1:L0] + 17) % 50000
    xb, _ = base_forward(model, bt2)
    ok1b = bool(torch.allclose(xa[:cut + 1], xb[:cut + 1], atol=1e-5)) and not bool(torch.allclose(xa[cut + 1:L0], xb[cut + 1:L0]))
    res["T1_causality"] = {"pass": bool(ok1 and ok1b), "router_posterior": ok1, "base": ok1b, "doc_len": L0}
    # ---------------- T2 target isolation
    import csr.orv1_lib as OL
    import spb1_router_job as RJ
    bf = inspect.getsource(OL.base_forward)
    dep, post, hid = inspect.getsource(RJ.deploy), inspect.getsource(RJ.posterior), inspect.getsource(RJ.hidden_table)
    ok2 = ('model({"input_ids": bt["input_ids"], "doc_ids": bt["doc_ids"], "positions": bt["positions"]})' in bf
           and "target" not in dep.split("d = T[")[0] and "lab" not in post and "base_forward(model, bt)" in hid
           and list(inspect.signature(FreeRouter.forward).parameters) == ["self", "x", "extra"])
    res["T2_target_isolation"] = {"pass": bool(ok2)}
    # ---------------- T3 pair routing
    xn = torch.randn(5000, 384)
    grp = torch.randint(0, K, (5000,))
    v = torch.ones(5000, dtype=torch.bool)
    b_pair, b_free = PairBank("SEM2", 0, K, M, "pair"), PairBank("GEN32", 0, K, M, "free")
    with torch.no_grad():
        b_pair.router.weight.normal_()
        b_free.router.weight.normal_()
    ip, _, auxp, _ = b_pair.route(xn, grp, v)
    iff, _, auxf, _ = b_free.route(xn, grp, v)
    ok3 = bool((ip // M == grp).all()) and bool((iff // M != grp).float().mean() > 0.5) and len(torch.unique(iff)) == E
    res["T3_pair_routing"] = {"pass": ok3, "aux_pair": float(auxp), "aux_free": float(auxf),
                              "free_out_of_cluster_frac": float((iff // M != grp).float().mean())}
    # ---------------- T4 unrestricted deployment
    r = FreeRouter(64, 0, E)
    xr = torch.randn(4, 50, 384)
    ok4 = True
    for j in range(E):
        with torch.no_grad():
            r.out.bias.zero_()
            r.out.bias[j] = 1e4
        ok4 &= bool((r(xr).argmax(-1) == j).all())
    src_r = inspect.getsource(FreeRouter) + inspect.getsource(RJ.train_router) + dep
    ok4 &= not any(w in src_r for w in ("masked_fill", "allowed(", "lab_sem", "-inf"))
    res["T4_unrestricted_deployment"] = {"pass": bool(ok4), "n_specialists": E}
    # ---------------- T5 parameter / active-FLOP matching
    banks = make_banks(0, K, M, torch.device("cpu"))
    npar = {c: b.n_params() for c, b in banks.items()}
    widths = []
    import csr.lfr_model as LM
    orig = LM.Experts.forward

    def spy(self, xx, topi, gates):
        widths.append(int(topi.shape[1]))
        return orig(self, xx, topi, gates)
    LM.Experts.forward = spy
    try:
        for c, b in banks.items():
            b(xn[:256], grp[:256], v[:256])
    finally:
        LM.Experts.forward = orig
    w1 = {c: [p.clone() for p in b.bank.parameters()] for c, b in banks.items()}
    same_init = all(torch.equal(w1["SEM2"][i], w1[c][i]) for c in w1 for i in range(2))
    same_rinit = all(torch.equal(banks["SEM2"].router.weight, b.router.weight) for b in banks.values())
    rp = {c: FreeRouter(64, 0, E).n_params() for c in banks}
    ok5 = (len({json.dumps(v_) for v_ in npar.values()}) == 1 and set(widths) == {1} and same_init and same_rinit
           and len(set(rp.values())) == 1)
    active = 2 * 384 * 256 * 2 + 2 * 384 * E
    res["T5_param_flop_matching"] = {"pass": bool(ok5), "params": npar, "dispatch_topi_width": sorted(set(widths)),
                                     "identical_init": same_init and same_rinit, "deployed_router_params": rp["SEM2"],
                                     "active_flops_per_token_residual": active,
                                     "coarse16_bank_params": 16 * 2 * 384 * 256}
    # ---------------- T6 / T7 training: G-ZERO and bitwise resume
    gcfg = json.dumps(P.DEFAULT_GEN)
    A_, B_ = T / "trainA", T / "trainB"
    base_args = ["spb1_train.py", "--seed", "0", "--gen", "0", "--target", "0", "--test-rows", "2", "--test-steps", "4",
                 "--milestones", "4", "--device", "cpu", "--cfg-json", gcfg, "--execute"]
    sh(base_args + ["--out-root", rel(A_)])
    sh(base_args + ["--out-root", rel(B_), "--max-seconds", "0"], ok=(75,))
    sh(base_args + ["--out-root", rel(B_)])
    SA = torch.load(A_ / "g0_s0" / "ckpt_4.pt", weights_only=False)
    SB = torch.load(B_ / "g0_s0" / "ckpt_4.pt", weights_only=False)

    def eq(a, b):
        if isinstance(a, dict):
            return all(eq(a[k], b[k]) for k in a)
        if isinstance(a, (list, tuple)):
            return all(eq(x_, y_) for x_, y_ in zip(a, b))
        if torch.is_tensor(a):
            return torch.equal(a, b)
        return a == b
    ok7 = eq(SA["banks"], SB["banks"]) and eq(SA["opts"], SB["opts"]) and SA["step"] == SB["step"] == 4
    m1 = json.loads((A_ / "g0_s0" / "metrics.jsonl").read_text().splitlines()[0])
    zero = {c: abs(m1[f"ce_{c}"] - m1["ce_D"]) for c in ("SEM2", "SHUF2", "GEN32")}
    res["T6_g_zero"] = {"pass": bool(max(zero.values()) < 1e-5), "abs_diff": zero}
    res["T7_resume_bitwise"] = {"pass": bool(ok7)}
    # ---------------- T8 extraction consistency (2 rows TRAIN, 2 rows SEL, with COARSE-16)
    ck = A_ / "g0_s0" / "ckpt_4.pt"
    for part in ("TRAIN", "SEL"):
        sh(["spb1_extract.py", "--seed", "0", "--ckpt", rel(ck), "--part", part, "--out-dir", rel(T / "g0_s0"),
            "--common-dir", rel(T / "common_s0"), "--coarse", "--rows-limit", "2", "--test", "--device", "cpu", "--execute"])
    s8 = json.loads((T / "g0_s0" / "summary_SEL.json").read_text())
    zc = np.load(T / "common_s0" / "common_SEL.npz")
    zb = np.load(T / "g0_s0" / "bank_SEM2_SEL.npz")
    zk = np.load(T / "common_s0" / "coarse_SEL.npz")
    ok8 = (max(s8["lookup_vs_direct_max_abs"].values()) < 2e-3 and zb["d"].shape == (len(zc["tgt"]), E)
           and zb["pdiff"].shape == (len(zc["tgt"]), K) and zk["d"].shape == (len(zc["tgt"]), 16)
           and bool(((zb["e_tr"].astype(np.int64) // M) == zc["lab_sem"].astype(np.int64)).all()))
    res["T8_extraction_consistency"] = {"pass": bool(ok8), "lookup_vs_direct_max_abs": s8["lookup_vs_direct_max_abs"],
                                        "n_tokens": int(len(zc["tgt"]))}
    # ---------------- T9 balanced shuffle
    tj = json.loads((P.TABLES / "tables.json").read_text())
    errs = [tj["main"]["S1_50M"]["max_abs_diff"], tj["main"]["S3_245M"]["max_abs_diff"]] + \
        [v_["max_abs_diff"] for v_ in tj["cs"].values()]
    zl = np.load(P.TABLES / "labels_main.npz")
    sp = Split("main")
    pap = np.asarray(sp.papers)
    shf = zl["shuf"].astype(np.int64)
    o = np.argsort(pap, kind="stable")
    pp, ss = pap[o], shf[o]
    brk = np.r_[True, pp[1:] != pp[:-1]]
    first = np.maximum.accumulate(np.where(brk, np.arange(len(pp)), 0))
    persistent = bool((ss == ss[first]).all())
    ok9 = max(errs) <= 1e-4 and tj["main"]["paper_nmi_sem_shuf"] < 0.01 and persistent
    res["T9_balanced_shuffle"] = {"pass": bool(ok9), "max_share_error": float(max(errs)),
                                  "paper_nmi": tj["main"]["paper_nmi_sem_shuf"], "persistent_per_paper": persistent}
    # ---------------- T10 locks
    r10 = sh(["-c", "import sys; sys.path.insert(0,'src'); sys.path.insert(0,'.'); import spb1_extract as E; "
                    "E.partition('VAL')"], ok=(0, 1))
    bad = []
    for f in sorted(guard.ROOT.glob("spb1_*.py")) + sorted((guard.ROOT / "src/csr").glob("spb1_*.py")):
        txt = f.read_text()
        for pat in (r"Split\(\"test\"\)", r"calib_sel", r"test_tokens"):
            if re.search(pat, txt) and f.name != "spb1_tests.py":
                bad.append((f.name, pat))
    ok10 = (not (P.ART / "FROZEN_FINAL.json").exists()) and r10.returncode == 1 and "locked" in r10.stderr and not bad
    res["T10_locks"] = {"pass": bool(ok10), "bad_refs": bad}
    # ---------------- T11 functional metric (synthetic)
    import spb1_analyze as AN
    rng = np.random.default_rng(0)
    N, P_ = 300_000, 200

    class FakeC:
        pass
    C = FakeC()
    C.N, C.P = N, P_
    C.paper = rng.integers(0, P_, N)
    plab = rng.integers(0, K, P_)
    lab = plab[C.paper]
    C.base_st = rng.integers(0, 180, N)
    pdiff = rng.normal(size=(N, K)).astype(np.float16)
    e_tr = (lab * 2 + (pdiff[np.arange(N), lab] < 0)).astype(np.int64)
    d0 = rng.normal(0, 0.05, size=(N, E)).astype(np.float32)
    f0 = AN.bank_functional(C, d0, 2, lab, {"e_tr": e_tr, "pdiff": pdiff}, "intrinsic")
    d1 = d0.copy()
    own_cols = lab * 2 + np.arange(2).reshape(2, 1)
    for j in range(2):
        d1[np.arange(N), own_cols[j]] -= 0.02   # planted: own-cluster specialists 0.02 nats better on own tokens
    f1 = AN.bank_functional(C, d1, 2, lab, {"e_tr": e_tr, "pdiff": pdiff}, "intrinsic")
    ok11 = abs(f0["I"]) < 0.003 and abs(f1["I"] - 0.02 * (1 + 1 / 31)) < 0.004 and f1["n_scored"] == E
    res["T11_functional_metric"] = {"pass": bool(ok11), "I_null": f0["I"], "I_planted": f1["I"],
                                    "expected_planted": 0.02 * (1 + 1 / 31)}
    # ---------------- T12 smoke: routers + analysis
    guard.atomic_write_json(T / "gen_cfg.json", P.DEFAULT_GEN)
    cfgs = [{"tag": "R_SEM", "bank": "SEM2", "router_seed": 0}, {"tag": "R_SHUF", "bank": "SHUF2", "router_seed": 0},
            {"tag": "R_GEN", "bank": "GEN32", "router_seed": 0}, {"tag": "R_COARSE16", "bank": "COARSE16", "router_seed": 0},
            {"tag": "R_SEM_POST", "bank": "SEM2", "router_seed": 0, "post": True}]
    guard.atomic_write_json(T / "cfg.json", cfgs)
    sh(["spb1_router_job.py", "--seed", "0", "--tables", rel(T / "g0_s0"), "--common", rel(T / "common_s0"),
        "--configs", rel(T / "cfg.json"), "--stage", "S2_g0", "--steps", "5", "--rows-limit", "2", "--out-root",
        rel(T / "routers"), "--test", "--device", "cpu", "--execute"])
    nr = len(list((T / "routers" / "S2_g0").glob("*_SEL.json")))
    a1 = sh(["spb1_analyze.py", "stage1", "--gen", "0", "--seeds", "0", "--root", rel(T), "--out", rel(T / "a1.json")])
    a2 = sh(["spb1_analyze.py", "stage2", "--gen", "0", "--seeds", "0", "--root", rel(T), "--stage-name", "S2_g0",
             "--out", rel(T / "a2.json")])
    j2 = json.loads((T / "a2.json").read_text())
    ok12 = nr == 5 and (T / "a1.json").exists() and "SEM_minus_GEN" in j2["comparisons"] and \
        "SEMpost_minus_SEM" in j2["comparisons"]
    res["T12_smoke"] = {"pass": bool(ok12), "n_routers": nr, "comparisons": sorted(j2["comparisons"])}
    allp = all(v_["pass"] for v_ in res.values())
    out = {"time": guard.now(), "all_pass": allp, "tests": res}
    p = P.ART / f"cpu_tests_{guard.utc_stamp()}.json"
    guard.atomic_write_json(p, out, mode=0o400)
    print(json.dumps({k: v_["pass"] for k, v_ in res.items()}, indent=1), "\nALL PASS" if allp else "\nFAILURES", p.name)
    if allp:
        for f in sorted(T.rglob("*")):
            if f.is_file():
                LG.deletion(rel(f), guard.sha256_file(f), f.stat().st_size, "spb1 CPU-test scratch")
                f.chmod(0o600)
        shutil.rmtree(T)


if __name__ == "__main__":
    main()
