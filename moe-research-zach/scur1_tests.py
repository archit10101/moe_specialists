#!/usr/bin/env python
"""scur1 CPU correctness tests (no calibration result is inspected; the smoke run uses 4 TRAIN / 4 SEL rows of the
frozen m1111 milestone and its outputs are deleted, logged).

T1 causality: route and subject outputs at position t are unchanged when features after t change (all modes)
T2 bounded correction: |route logit - log(p_prior + eps)| <= rho (scu mode); subject mode = log subject posterior
T3 out-of-fold utility: the mapping applied to fold-f tokens is built only from the other fold's tokens (perturbing
   fold-0 gains changes U_fold0 but not the fold-0 tokens' mapping); utility_matrix matches a direct computation
T4 router inputs: forward/heads take only causal features (no target, label or oracle argument); evaluation path
   calls heads(backbone(X)) without targets
T5 mapping kinds: soft rows sum to 1, hard is one-hot argmax, sinkhorn columns balanced
T6 losses: finite gradients; utility KL = 0 when p_route = q_utility; regret = 0 when routing to the subject argmax
T7 locks: CS2 refused before FROZEN_A.json; no validation/test references in scur1 code
T8 smoke: TRAIN + SEL extraction (4 rows each), router job (5 steps, all modes) and stage analysis run end to end"""
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


def sh(args, ok=(0,)):
    r = subprocess.run([PY] + args, cwd=guard.ROOT, env=dict(os.environ), capture_output=True, text=True)
    if r.returncode not in ok:
        raise RuntimeError(f"{args} rc={r.returncode}\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    return r


def main():
    import torch
    from csr import scur1_log as LG
    from csr import scur1_paths as P
    from csr.scur1_router import SCURouter, losses, mapping, utility_matrix
    guard.check_root()
    res = {}
    torch.manual_seed(0)
    K = 5
    # T1 / T2
    x = torch.randn(3, 40, 384)
    ln = torch.tensor([40, 33, 20])
    ok1, ok2 = True, True
    for mode in ("scu", "subject", "utility"):
        r = SCURouter(64, 0, K, "base", mode, rho=0.7)
        r.set_A(torch.softmax(torch.randn(K, K), 1))
        sl, rl = r(x, ln)
        x2 = x.clone()
        x2[:, 25:] += torch.randn_like(x2[:, 25:]) * 3
        sl2, rl2 = r(x2, ln)
        ok1 &= bool(torch.allclose(rl[:, :25], rl2[:, :25], atol=1e-5) and torch.allclose(sl[:, :25], sl2[:, :25], atol=1e-5))
        if mode == "scu":
            pp = torch.softmax(sl, -1) @ r.A
            ok2 &= bool(((rl - torch.log(pp + r.eps)).abs() <= 0.7 + 1e-5).all())
        if mode == "subject":
            ok2 &= bool(torch.allclose(rl, torch.log(torch.softmax(sl, -1) + r.eps), atol=1e-6))
    res["T1_causality"] = {"pass": ok1}
    res["T2_bounded_correction"] = {"pass": ok2}
    # T3
    N = 4000
    q = torch.softmax(torch.randn(N, K) * 2, 1)
    gain = torch.randn(N, K) * 0.1
    fold = torch.arange(N) % 2
    U0, U1 = utility_matrix(q[fold == 0], gain[fold == 0]), utility_matrix(q[fold == 1], gain[fold == 1])
    direct = (q[fold == 0].double().t() @ gain[fold == 0].double()) / q[fold == 0].double().sum(0).unsqueeze(1)
    Af = torch.stack([mapping(U1, 1.0), mapping(U0, 1.0)])
    gain2 = gain.clone()
    gain2[fold == 0] += torch.randn(int((fold == 0).sum()), K)
    U0b = utility_matrix(q[fold == 0], gain2[fold == 0])
    Af2 = torch.stack([mapping(U1, 1.0), mapping(U0b, 1.0)])
    ok3 = bool(torch.allclose(U0.double(), direct, atol=1e-5) and torch.equal(Af[0], Af2[0]) and not torch.allclose(U0, U0b))
    src = (guard.ROOT / "scur1_router_job.py").read_text()
    ok3 &= "Af = torch.stack([mapping(U[1], tf, kind), mapping(U[0], tf, kind)])" in src and "Af[ft]" in src
    res["T3_out_of_fold_U"] = {"pass": ok3}
    # T4
    sig_f = list(inspect.signature(SCURouter.forward).parameters)
    sig_h = list(inspect.signature(SCURouter.heads).parameters)
    ev = src.split("def evaluate(")[1].split("def run(")[0]
    ok4 = sig_f == ["self", "x", "lengths"] and sig_h == ["self", "g", "A"] and "r.heads(r.backbone(T[\"X\"][idx], ln)[mask])" in ev \
        and "tgt" not in ev.split("r.heads")[0]
    res["T4_router_inputs"] = {"pass": bool(ok4), "forward": sig_f, "heads": sig_h}
    # T5
    U = torch.randn(K, K)
    As, Ah, Ak = mapping(U, 1.0), mapping(U, 1.0, "hard"), mapping(U, 1.0, "sinkhorn")
    ok5 = bool(torch.allclose(As.sum(1), torch.ones(K), atol=1e-5) and torch.equal(Ah.argmax(1), U.argmax(1))
               and torch.allclose(Ak.sum(0), torch.ones(K), atol=1e-3) and torch.allclose(Ak.sum(1), torch.ones(K), atol=1e-3))
    res["T5_mapping"] = {"pass": ok5}
    # T6
    n = 64
    qs = torch.softmax(torch.randn(n, K), 1)
    A_tok = As.unsqueeze(0).expand(n, K, K)
    U_tok = U.unsqueeze(0).expand(n, K, K)
    qu = torch.einsum("ns,nse->ne", qs, A_tok)
    sl = torch.randn(n, K, requires_grad=True)
    tot, parts = losses(sl, torch.log(qu), qs, A_tok, U_tok, torch.randn(n, K), {"utility": 1.0})
    us = torch.einsum("ns,nse->ne", qs, U_tok)
    onehot = torch.log(torch.nn.functional.one_hot(us.argmax(1), K).float() + 1e-12)
    _, parts2 = losses(sl, onehot, qs, A_tok, U_tok, torch.randn(n, K), {"regret": 1.0})
    rl = torch.randn(n, K, requires_grad=True)
    tot3, _ = losses(sl, rl, qs, A_tok, U_tok, torch.randn(n, K), {"subject": 1, "utility": 1, "regret": 1, "lm": 1, "balance": 0.01})
    tot3.backward()
    ok6 = abs(parts["utility"]) < 1e-5 and abs(parts2["regret"]) < 1e-5 and bool(torch.isfinite(rl.grad).all())
    res["T6_losses"] = {"pass": bool(ok6), "utility_at_target": parts["utility"], "regret_at_argmax": parts2["regret"]}
    # T7
    r = sh(["-c", "import sys; sys.path.insert(0,'src'); sys.path.insert(0,'.'); import scur1_extract as E; E.partition('CS2')"],
           ok=(0, 1))
    bad = []
    for f in ["scur1_router_job.py", "scur1_extract.py", "scur1_analyze.py", "scur1_run.py", "src/csr/scur1_router.py"]:
        txt = (guard.ROOT / f).read_text()
        for pat in (r"Split\(\"val\"\)", r"Split\(\"test\"\)", r"val_assign", r"\"test_tokens"):
            if re.search(pat, txt):
                bad.append((f, pat))
    res["T7_locks"] = {"pass": bool(not (P.ART / "FROZEN_A.json").exists() and r.returncode == 1 and "locked" in r.stderr
                                    and not bad), "bad_refs": bad}
    # T8 smoke
    T = P.TMP / "tests"
    if T.exists():
        shutil.rmtree(T)
    T.mkdir(parents=True)
    mp, _ = P.MILESTONES[P.PRIMARY]
    banks = f"{P.SEM_BANK},{P.SHUF_BANK},{P.G_BANK},{P.ALTG_BANK}"
    for part in ("TRAIN", "SEL"):
        sh(["scur1_extract.py", "--ckpt", mp, "--part", part, "--out-dir", str((T / part).relative_to(guard.ROOT)), "--banks",
            banks, "--rows-limit", "4", "--device", "cpu", "--execute"])
    cfgs = [{"tag": "TSCU", "mode": "scu", "bank": "SEM", "w": {"subject": 1, "utility": 1, "regret": 1, "lm": 1, "balance": 0.01},
             "tau": 1.0, "rho": 1.0, "feats": "base", "A": "soft"},
            {"tag": "TSCU_SHUF", "mode": "scu", "bank": "SHUF", "w": {"subject": 1, "utility": 1, "regret": 1, "lm": 1},
             "tau": 1.0, "rho": 1.0, "feats": "base", "A": "soft", "twin_of": "TSCU"},
            {"tag": "SUBJ", "mode": "subject", "bank": "SEM", "w": {"subject": 1}},
            {"tag": "TUTIL", "mode": "utility", "bank": "SEM", "w": {"lm": 1, "balance": 0.01}}]
    (T / "cfg.json").write_text(json.dumps(cfgs))
    stg = "TEST_smoke"
    sh(["scur1_router_job.py", "--milestone", P.PRIMARY, "--train", str((T / "TRAIN").relative_to(guard.ROOT)), "--eval",
        str((T / "SEL").relative_to(guard.ROOT)), "--eval-part", "SEL", "--configs", str((T / "cfg.json").relative_to(guard.ROOT)),
        "--seed", "0", "--stage", stg, "--steps", "5", "--rows-limit", "4", "--test", "--device", "cpu", "--execute"])
    r = sh(["scur1_analyze.py", "stage", "--stage", stg, "--eval-dir", str((T / "SEL").relative_to(guard.ROOT)), "--part", "SEL"],
           ok=(0, 1))
    an = P.ANALYSIS / f"{stg}.json"
    smoke = {"analysis_rc": r.returncode, "analysis_written": an.exists(), "stderr_tail": r.stderr[-400:]}
    if an.exists():
        A = json.loads(an.read_text())
        smoke["routers"] = sorted(A["routers"])
        smoke["has_gap_decomposition"] = "gap_decomposition" in A["routers"].get("TSCU_s0", {})
    res["T8_smoke"] = {"pass": bool(r.returncode == 0 and an.exists() and smoke.get("has_gap_decomposition")), **smoke}
    allp = all(v["pass"] for v in res.values())
    out = {"time": guard.now(), "all_pass": allp, "tests": res}
    p = P.ART / f"cpu_tests_{guard.utc_stamp()}.json"
    guard.atomic_write_json(p, out, mode=0o400)
    print(json.dumps({k: v["pass"] for k, v in res.items()}, indent=1), "\nALL PASS" if allp else "\nFAILURES", p.name)
    if allp:
        for d in (T, P.ROUTERS / stg, P.DEPS / stg):
            if d.exists():
                for f in sorted(d.rglob("*")):
                    if f.is_file():
                        LG.deletion(str(f.relative_to(guard.ROOT)), guard.sha256_file(f), f.stat().st_size, "scur1 CPU-test scratch")
                        f.chmod(0o600)
                shutil.rmtree(d)
        if an.exists():
            LG.deletion(str(an.relative_to(guard.ROOT)), guard.sha256_file(an), an.stat().st_size, "scur1 CPU-test scratch")
            an.unlink()


if __name__ == "__main__":
    main()
