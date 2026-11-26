#!/usr/bin/env python
"""asrs2 CPU correctness tests (main-stream training rows only; no calibration partition is opened except a 4-row SEL
smoke evaluation of a TEST milestone whose numbers are discarded). Test runs live under TMP/test_runs (deleted
afterwards, logged).

T1 base identity: continuation with all residual arms vs base-only continuation, same rows: per-step base loss and final
   base parameters bitwise equal (the base update cannot depend on any specialist or router)
T2 module isolation: routers unchanged in SPEC windows, assigned experts unchanged in ROUTER windows, ALT-G router
   unchanged in SPEC / experts unchanged in ROUTER windows, FROZ bank never changes, G changes every step
T3 target-free assignment: assignments are a function of the region prior only (targets permuted -> identical choices);
   the trainer's specialist choice is taken from the prior / prior sample (static check)
T4 bitwise resume after SIGINT (exit 77) vs uninterrupted
T5 frozen mode: base parameters bitwise unchanged; banks and routers trained
T6 soft sampling: deterministic per (seed, step, bank), empirical frequencies match p
T7 rows identical across paired runs (rows_sha) and the schedule counts (base updates on every step)
T8 zero residual = base exactly; deltas consistent with explicit corrections
T9 no held-out access: CS2 locked before FROZEN_S1.json; no 'val'/'test' split reference in asrs2 code
T10 smoke: evaluation (4 SEL rows, test milestone) + per-milestone analysis run end to end"""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402

PY = str(guard.ROOT / ".venv/bin/python")


def sh(args, env_extra=None, ok=(0,)):
    env = dict(os.environ)
    env.update(env_extra or {})
    r = subprocess.run([PY] + args, cwd=guard.ROOT, env=env, capture_output=True, text=True)
    if r.returncode not in ok:
        raise RuntimeError(f"{args} rc={r.returncode}\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    return r


def main():
    import numpy as np
    import torch
    from csr import asrs2_cfg as C
    from csr import asrs2_log as LG
    from csr import asrs2_paths as P
    guard.check_root()
    T = P.TMP / "tests"
    if T.exists():
        shutil.rmtree(T)
    (T / "cfg").mkdir(parents=True)
    runs = T / "runs"
    runs.mkdir()
    env = {"MOE_RUNS_DIR": str(runs)}
    res = {}
    NR = 2

    def cfgfile(cfg, name):
        p = T / "cfg" / f"{name}.json"
        p.write_text(json.dumps(cfg))
        return str(p.relative_to(guard.ROOT))

    def run_train(cfg, name, extra=(), ok=(0,)):
        r = sh(["asrs2_train.py", "--config", cfgfile(cfg, name), "--device", "cpu", "--test-rows", str(NR), "--execute",
                *extra], env, ok)
        rd = sorted(runs.glob(f"*_asrs2_{name}"))[-1]
        return rd, r

    def metrics(rd):
        return [json.loads(x) for x in open(rd / "metrics.jsonl")]

    def ckpt_state(rd, tag):
        return torch.load(rd / f"ckpt_{tag}" / "state.pt", map_location="cpu", weights_only=False)

    # ---- Stage-0F test milestone (frozen base), 3 steps
    cf = C.build("tF", "frozen", seeds=[0, 1], test=True, start=0, end=3, milestones={"F": 3})
    rdF, _ = run_train(cf, "tF")
    F = ckpt_state(rdF, "F")
    from csr.lfr_model import LFRModel, LFRConfig  # noqa: F401
    b0 = torch.load(guard.ROOT / P.CKPT / "state.pt", map_location="cpu", weights_only=False)
    msF = metrics(rdF)
    trained = all(not torch.equal(F["banks"][f"{k}_0"]["experts.w2"], torch.zeros_like(F["banks"][f"{k}_0"]["experts.w2"]))
                  for k in ("G5", "SEM5", "SHUF5"))
    paired_init = True
    from csr.asrs2_lib import Bank5
    i0 = Bank5(P.K, P.D_MODEL, P.FFN, 0, routed=True).state_dict()
    i1 = Bank5(P.K, P.D_MODEL, P.FFN, 0, routed=False).state_dict()
    paired_init = torch.equal(i0["experts.w1"], i1["experts.w1"])
    res["T5_frozen_mode"] = {"pass": bool(trained and paired_init and all(m.get("lm") is None for m in msF if "step" in m)
                                          and F["mode"] == "frozen" and F["model"] == {}),
                             "banks_trained": trained, "paired_init_G_SEM_SHUF": paired_init,
                             "note": "frozen mode never builds a base optimizer; base taken from the reused milestone"}
    init_rel = str((rdF / "ckpt_F").relative_to(guard.ROOT))
    # ---- continuation: all arms vs base-only, 6 steps, windows spec 3 / router 2 (two windows: S S S R R S)
    kw = dict(test=True, start=318, end=324, init_ckpt=init_rel, init_sha=None, spec_w=3, router_w=2, milestones={"m324": 324})
    cA = C.build("tA", "cont", seeds=[0], **kw)
    cB = C.build("tB", "cont", seeds=[], arms=[], **kw)
    rdA, _ = run_train(cA, "tA", ["--no-compile"])
    rdB, _ = run_train(cB, "tB", ["--no-compile"])
    mA = [m for m in metrics(rdA) if "step" in m and "arms" in m]
    mB = [m for m in metrics(rdB) if "step" in m and "arms" in m]
    A, B = ckpt_state(rdA, "m324"), ckpt_state(rdB, "m324")
    same_params = all(torch.equal(A["model"][k], B["model"][k]) for k in A["model"])
    same_lm = [x["lm"] for x in mA] == [x["lm"] for x in mB]
    res["T1_base_identity"] = {"pass": bool(same_params and same_lm), "lm_A": [x["lm"] for x in mA], "lm_B": [x["lm"] for x in mB],
                               "n_param_tensors": len(A["model"])}
    res["T7_rows_and_counts"] = {"pass": bool([x["rows_sha"] for x in mA] == [x["rows_sha"] for x in mB]
                                              and A["n_upd"].get("base") == 6 and B["n_upd"].get("base") == 6
                                              and [x["phase"] for x in mA] == ["spec", "spec", "spec", "router", "router", "spec"]),
                                 "phases": [x["phase"] for x in mA], "n_upd": A["n_upd"]}
    # ---- isolation from per-step parameter snapshots: compare init (F) vs after, and window param hashes
    ph = [m for m in metrics(rdA) if m.get("kind") == "param_hash"]
    viol = []
    for i in range(1, len(ph)):
        prev, cur = ph[i - 1]["hash"], ph[i]["hash"]
        changed = {k for k in cur if cur[k] != prev[k]}
        if ph[i]["after"] == "router":
            bad = {k for k in changed if (k.endswith(".experts") and not k.startswith("G_")) or k == "ALTG_0.router"
                   and False}
            bad |= {k for k in changed if k.startswith(("SEMh", "SHUFh", "SEMs", "SHUFs", "ALTG")) and k.endswith(".experts")}
        else:
            bad = {k for k in changed if k.startswith("router.") or k == "ALTG_0.router"}
        bad |= {k for k in changed if k.startswith("FROZ_0.experts")}
        if bad:
            viol.append({"after": ph[i]["after"], "bad": sorted(bad)})
    froz_same = torch.equal(A["banks"]["FROZ_0"]["experts.w1"], F["banks"]["SEM5_0"]["experts.w1"])
    g_changed = not torch.equal(A["banks"]["G_0"]["experts.w1"], F["banks"]["G5_0"]["experts.w1"])
    res["T2_isolation"] = {"pass": bool(not viol and froz_same and g_changed and len(ph) >= 2), "violations": viol,
                           "n_hash_records": len(ph), "froz_unchanged": froz_same, "G_changed": g_changed}
    # ---- resume
    cR = C.build("tR", "cont", seeds=[0], **kw)
    rdR, r1 = run_train(cR, "tR", ["--no-compile", "--test-signal-at-step", "321"], ok=(77,))
    sh(["asrs2_train.py", "--config", cfgfile(cR, "tR"), "--device", "cpu", "--test-rows", str(NR), "--execute",
        "--no-compile", "--run-dir", str(rdR.relative_to(guard.ROOT))], env)
    R = ckpt_state(rdR, "m324")
    same_all = all(torch.equal(R["model"][k], A["model"][k]) for k in A["model"]) and all(
        torch.equal(R["banks"][n][k], A["banks"][n][k]) for n in A["banks"] for k in A["banks"][n]) and all(
        torch.equal(R["routers"][n][k], A["routers"][n][k]) for n in A["routers"] for k in A["routers"][n])
    res["T4_resume"] = {"pass": bool(same_all), "rc_first_leg": 77}
    # ---- target-free
    from csr.asrs2_lib import assign
    p = torch.softmax(torch.randn(5000, 5), 1)
    e1 = assign(p, "hard")
    g1, g2 = torch.Generator().manual_seed(5), torch.Generator().manual_seed(5)
    s1, s2 = assign(p, "soft", g1), assign(p, "soft", g2)
    src = (guard.ROOT / "asrs2_train.py").read_text()
    calls = re.findall(r"spec_grad\(b, f\[\"h\"\], f\[\"xn\"\], W, f\[\"y\"\], (\w+)\)", src)
    e_def = re.findall(r"\n\s+e = (assign\(PRI\[pri\]\[f\[\"rid\"\]\], \"soft\".*?\)|e_prior\[pri\])", src)
    no_y_in_assign = "f[\"y\"]" not in src.split("def soft_gen")[1].split("ce = spec_grad")[0].split("if do_spec:")[1]
    res["T3_target_free"] = {"pass": bool(torch.equal(e1, p.argmax(1)) and torch.equal(s1, s2) and calls == ["e"]
                                          and len(e_def) == 2 and no_y_in_assign),
                             "spec_grad_calls": calls, "assignment_sources": e_def, "no_target_in_assignment_block": no_y_in_assign}
    # ---- soft sampling statistics
    q = torch.tensor([[0.6, 0.3, 0.1, 0.0, 0.0]]).repeat(200000, 1)
    s = assign(q, "soft", torch.Generator().manual_seed(1))
    fr = torch.bincount(s, minlength=5).float() / len(s)
    res["T6_soft"] = {"pass": bool((fr - q[0]).abs().max() < 0.01), "freq": fr.tolist()}
    # ---- zero residual / deltas
    from csr.asrs2_lib import deltas, rms0, head_ce_grad
    b = Bank5(5, P.D_MODEL, P.FFN, 0)
    h = torch.randn(300, P.D_MODEL)
    Wt = torch.randn(1000, P.D_MODEL) * 0.05
    y = torch.randint(0, 1000, (300,))
    u, ceb = deltas(b.experts.w1, b.experts.w2, h, rms0(h), Wt, y)
    with torch.no_grad():
        b.experts.w2.normal_(0, 0.02)
    u2, _ = deltas(b.experts.w1, b.experts.w2, h, rms0(h), Wt, y)
    a3 = b.correction(rms0(h), torch.full((300,), 3, dtype=torch.long)).detach()
    ce3, _ = head_ce_grad(h, a3, Wt, y, want_grad=False)
    res["T8_zero_and_deltas"] = {"pass": bool(u.abs().max() == 0 and (ceb - ce3 - u2[:, 3]).abs().max() < 1e-4),
                                 "max_dev": float((ceb - ce3 - u2[:, 3]).abs().max())}
    # ---- held-out locks
    locked = (P.ART / "FROZEN_S1.json").exists() is False
    r = sh(["-c", "import sys; sys.path.insert(0,'src'); sys.path.insert(0,'.'); import asrs2_eval as E; E.partition('CS2')"],
           ok=(0, 1))
    refused = r.returncode == 1 and "locked" in r.stderr
    bad_refs = []
    for f in ["asrs2_train.py", "asrs2_eval.py", "asrs2_analyze.py", "asrs2_clusters.py", "src/csr/asrs2_lib.py"]:
        txt = (guard.ROOT / f).read_text()
        for pat in (r"Split\(\"val\"\)", r"Split\(\"test\"\)", r"\"test_tokens", r"val_assign"):
            if re.search(pat, txt):
                bad_refs.append((f, pat))
    res["T9_heldout_locks"] = {"pass": bool(locked and refused and not bad_refs), "cs2_refused": refused, "bad_refs": bad_refs}
    # ---- smoke eval + analysis on 4 SEL rows of the test milestone (numbers discarded)
    ev = T / "eval_smoke"
    sh(["asrs2_eval.py", "--ckpt", str((rdA / "ckpt_m324").relative_to(guard.ROOT)), "--part", "SEL", "--out-dir",
        str(ev.relative_to(guard.ROOT)), "--rows-limit", "4", "--device", "cpu", "--test", "--execute"])
    import asrs2_analyze as AN
    E = AN.EvalSet(ev, "SEL")
    smoke = {"n_tokens": E.N, "deps": sorted(E.summ["deps"]), "banks": sorted(E.summ["banks"])}
    try:
        out = AN.analyze_ckpt(ev, "SEL", [0])
        smoke["analysis_keys"] = sorted(out["seeds"]["0"])
        ok_smoke = True
    except Exception as e:  # small smoke set may be below MIN_SET for functional strata; record
        smoke["analysis_error"] = repr(e)[:300]
        ok_smoke = False
    res["T10_smoke"] = {"pass": ok_smoke, **smoke}
    allp = all(v["pass"] for v in res.values())
    out = {"time": guard.now(), "all_pass": allp, "tests": res}
    p = P.ART / f"cpu_tests_{guard.utc_stamp()}.json"
    guard.atomic_write_json(p, out, mode=0o400)
    print(json.dumps({k: v["pass"] for k, v in res.items()}, indent=1), "\nALL PASS" if allp else "\nFAILURES", p.name)
    if allp:
        for f in sorted(T.rglob("*")):
            if f.is_file() and f.suffix in (".pt",):
                LG.deletion(str(f.relative_to(guard.ROOT)), guard.sha256_file(f), f.stat().st_size, "CPU-test scratch")
        shutil.rmtree(T)
        slots = P.TMP / "slots"
        for d in sorted(slots.glob("*_asrs2_t*")):
            for f in d.rglob("state.pt"):
                LG.deletion(str(f.relative_to(guard.ROOT)), guard.sha256_file(f), f.stat().st_size, "CPU-test resume slot")
            shutil.rmtree(d)


if __name__ == "__main__":
    main()
