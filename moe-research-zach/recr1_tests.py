#!/usr/bin/env python
"""recr1 Stage-0 CPU tests (synthetic data unless noted). Writes ART/cpu_tests_<stamp>.json.

T1 policy causality: perturbing realised deltas at positions >= t0 and head outputs at > t0 never changes any executed
   option at positions <= t0 (modes E and D, generic and base fallback, strided evidence).
T2 feature causality: perturbing router outputs at positions > t0 never changes features at <= t0.
T3 route replay determinism + per-document independence (batched == single-document simulation).
T4 head training resume is bitwise identical to an uninterrupted run (checkpoint written + re-read).
T5 exactness / accounting: simulated per-token delta == table entry of the executed option; S-only policy has 0 overhead;
   evidence on every stable token costs exactly one LM head.
T6 partitions: UTRAIN/CSEARCH/CSELECT paper-disjoint, union = all 503 SEL papers, hash rule reproduces the manifest.
T7 kappa calibration reaches the target alarm rate (+-15%).
T8 label isolation: features are identical when every label/target/paper field of the partition is scrambled."""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402
from csr import recr1_lib as L  # noqa: E402
from csr import recr1_paths as P  # noqa: E402


class FakePart(L.Part):
    def __init__(self, lens, seed=0):
        rng = np.random.default_rng(seed)
        self.lens = np.array(lens)
        self.n_docs, self.N, self.Tm = len(lens), int(sum(lens)), int(max(lens))
        self.starts = np.r_[0, np.cumsum(self.lens)[:-1]]
        self.pos = np.concatenate([np.arange(n) for n in lens])
        self.doc = np.repeat(np.arange(self.n_docs), self.lens)
        self.paper = self.doc // 2
        self.P = int(self.paper.max()) + 1
        self.lab_sem = rng.integers(0, 16, self.N)
        self.r = rng.integers(0, 16, self.N)
        self.rp = rng.integers(-1, 16, self.N)
        self.tgt = rng.integers(0, 50000, self.N)
        self.routes = {}
        for tag in ("R_SEM", "R_SHUF", "R_GEN", "R_COARSE16"):
            n = 16 if tag == "R_COARSE16" else 32
            for r in (0, 1):
                self.routes[(tag, r)] = {"e": rng.integers(0, n, self.N).astype(np.int8),
                                         "p1": rng.random(self.N).astype(np.float16),
                                         "marg": (rng.random(self.N) * 5).astype(np.float16)}

    def sub(self, docs):
        f = FakePart([int(self.lens[i]) for i in docs])
        m = np.isin(self.doc, docs)
        for k, v in self.routes.items():
            f.routes[k] = {kk: vv[m] for kk, vv in v.items()}
        return f, m


def synth(part, rng):
    dB = (rng.normal(0, 0.07, (part.N, 32)) - 0.002).astype(np.float32)
    dS = dB[np.arange(part.N), rng.integers(0, 32, part.N)]
    dG = (rng.normal(0, 0.07, part.N) - 0.001).astype(np.float32)
    U = rng.normal(0, 0.01, (part.N, L.N_OPT)).astype(np.float32)
    return dB, dS, dG, U


CFGS = [dict(P.GEN0), {**P.GEN0, "fallback": "base", "B": 4, "probe": 16}, {**P.GEN0, "gamma": 0.8}]


def t1():
    rng = np.random.default_rng(1)
    part = FakePart([300, 157, 300, 41, 220])
    dB, dS, dG, U = synth(part, rng)
    bad, changed = 0, 0
    for cfg in CFGS + [{**P.GEN0, "norm": "z"}, {**P.GEN0, "norm": "clip"}]:
        for mode in ("E", "D"):
            for stride in (1, 16):
                ns = np.full(35, 0.07)
                base = L.simulate(part, dB, dS, dG, U, cfg, kappa=0.15, mode=mode, stride=stride, norm_sd=ns)["opt"]
                for t0 in (0, 3, 7, 8, 9, 40, 150):
                    dB2, dS2, dG2, U2 = dB.copy(), dS.copy(), dG.copy(), U.copy()
                    fut_d = part.pos >= t0
                    fut_u = part.pos > t0
                    dB2[fut_d] = rng.normal(0, 0.5, dB2[fut_d].shape)
                    dS2[fut_d] = rng.normal(0, 0.5, dS2[fut_d].shape)
                    dG2[fut_d] = rng.normal(0, 0.5, dG2[fut_d].shape)
                    U2[fut_u] = rng.normal(0, 0.5, U2[fut_u].shape)
                    o2 = L.simulate(part, dB2, dS2, dG2, U2, cfg, kappa=0.15, mode=mode, stride=stride, norm_sd=ns)["opt"]
                    bad += int((o2[part.pos <= t0] != base[part.pos <= t0]).sum())
                    changed += int((o2[part.pos > t0] != base[part.pos > t0]).sum())
    return {"pass": bad == 0 and changed > 0, "violations": bad, "later_changes_nonvacuous": changed}


def t2():
    part = FakePart([200, 90, 333])
    f0 = L.features(part, "SEM2", 0)
    bad = 0
    rng = np.random.default_rng(2)
    for t0 in (0, 5, 64, 150):
        p2 = FakePart([200, 90, 333])
        p2.routes = {k: {kk: vv.copy() for kk, vv in v.items()} for k, v in part.routes.items()}
        fut = part.pos > t0
        for k, v in p2.routes.items():
            n = 16 if k[0] == "R_COARSE16" else 32
            v["e"][fut] = rng.integers(0, n, fut.sum())
            v["p1"][fut] = rng.random(fut.sum())
            v["marg"][fut] = rng.random(fut.sum())
        f2 = L.features(p2, "SEM2", 0)
        bad += int((np.abs(f2[~fut] - f0[~fut]) > 0).sum())
    return {"pass": bad == 0, "violations": bad, "n_features": int(f0.shape[1])}


def t3():
    rng = np.random.default_rng(3)
    part = FakePart([300, 120, 280, 77])
    dB, dS, dG, U = synth(part, rng)
    a = L.simulate(part, dB, dS, dG, U, P.GEN0, kappa=0.2)
    b = L.simulate(part, dB, dS, dG, U, P.GEN0, kappa=0.2)
    same = all(np.array_equal(a[k], b[k]) for k in ("opt", "d", "shadows", "ev", "alarm"))
    s, m = part.sub([2])
    c = L.simulate(s, dB[m], dS[m], dG[m], U[m], P.GEN0, kappa=0.2)
    indep = np.array_equal(c["opt"], a["opt"][m]) and np.array_equal(c["d"], a["d"][m])
    return {"pass": bool(same and indep), "replay_identical": bool(same), "single_doc_identical": bool(indep)}


def t4():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(20000, 12)).astype(np.float32)
    X[:, -1] = 1
    Y = (X[:, :1] * 0.01 + rng.normal(0, 0.05, (20000, L.N_OPT))).astype(np.float32)
    old = (P.HEAD_BATCH, P.HEAD_CKPT_TOKENS)
    P.HEAD_BATCH, P.HEAD_CKPT_TOKENS = 256, 256 * 10
    try:
        with tempfile.TemporaryDirectory(dir=P.TMP) as td:
            ref = L.train_head(X, Y, "mlp", 7, 256 * 50, Path(td) / "a.pt", log=lambda *_: None)
            calls = {"n": 0}

            def stop():
                calls["n"] += 1
                return calls["n"] == 23
            try:
                L.train_head(X, Y, "mlp", 7, 256 * 50, Path(td) / "b.pt", log=lambda *_: None, stop_check=stop)
            except KeyboardInterrupt:
                pass
            res = L.train_head(X, Y, "mlp", 7, 256 * 50, Path(td) / "b.pt", log=lambda *_: None)
            same = all(torch.equal(p, q) for p, q in zip(ref.state_dict().values(), res.state_dict().values()))
    finally:
        P.HEAD_BATCH, P.HEAD_CKPT_TOKENS = old
    return {"pass": bool(same)}


def t5():
    rng = np.random.default_rng(5)
    part = FakePart([300, 260])
    dB, dS, dG, U = synth(part, rng)
    sim = L.simulate(part, dB, dS, dG, U, P.GEN0, kappa=0.2)
    Dall = np.concatenate([dB, dS[:, None], dG[:, None]], 1)
    o = sim["opt"].astype(np.int64)
    exact = np.allclose(sim["d"], np.where(o >= 0, Dall[np.arange(part.N), np.maximum(o, 0)], 0.0))
    S_only = {"opt": np.full(part.N, 32), "shadows": np.zeros(part.N), "ev": np.zeros(part.N, bool),
              "search": np.zeros(part.N, bool)}
    m = L.macs()
    ov0 = L.policy_macs(S_only, 0, 1)["overhead"]
    ev_all = dict(S_only, ev=np.ones(part.N, bool))
    ov1 = L.policy_macs(ev_all, 0, 1)["overhead"]
    want = m["lm"] / (m["trunk"] + m["lm"] + m["res"] + m["router"])
    return {"pass": bool(exact and abs(ov0) < 1e-12 and abs(ov1 - want) < 1e-9), "exact": bool(exact),
            "overhead_S": ov0, "overhead_full_evidence": ov1, "lm_fraction": want, "macs": m}


def t6():
    dm = json.loads((P.ART / "data_manifest.json").read_text())
    sets = {k: set(v["paper_sha256"]) for k, v in dm["partitions"].items()}
    disj = all(not (sets[a] & sets[b]) for a in sets for b in sets if a < b)
    c = np.load(P.EXTRACT / "common_s0/common_SEL.npz")
    allp = set(str(x) for x in c["papers"])
    union = set().union(*sets.values()) == allp
    import recr1_manifest as M
    rule = all(M.split_of(p) == k for k, v in sets.items() for p in v)
    return {"pass": bool(disj and union and rule), "disjoint": disj, "union_all": union, "rule": rule,
            "sizes": {k: len(v) for k, v in sets.items()}}


def t7():
    rng = np.random.default_rng(7)
    part = FakePart([1000] * 12)
    dS = rng.normal(-0.001, 0.07, part.N).astype(np.float32)
    k, rate = L.calibrate_kappa(part, dS, 0.95, 0.0, 4.0, 1)
    return {"pass": abs(rate - 4.0) / 4.0 < 0.15, "kappa": k, "rate": rate}


def t8():
    part = FakePart([150, 99])
    f0 = L.features(part, "SEM2", 0)
    rng = np.random.default_rng(8)
    part.lab_sem = rng.integers(0, 16, part.N)
    part.r, part.rp, part.tgt = rng.permutation(part.r), rng.permutation(part.rp), rng.permutation(part.tgt)
    part.paper = rng.permutation(part.paper)
    f1 = L.features(part, "SEM2", 0)
    return {"pass": bool(np.array_equal(f0, f1))}


def main():
    guard.check_root()
    P.TMP.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, fn in [("T1", t1), ("T2", t2), ("T3", t3), ("T4", t4), ("T5", t5), ("T6", t6), ("T7", t7), ("T8", t8)]:
        out[name] = fn()
        print(name, json.dumps(out[name], default=str)[:300], flush=True)
    out["all_pass"] = all(v["pass"] for v in out.values() if isinstance(v, dict))
    guard.atomic_write_json(P.ART / f"cpu_tests_{guard.utc_stamp()}.json", {"time": guard.now(), **out})
    print("ALL_PASS", out["all_pass"])


if __name__ == "__main__":
    main()
