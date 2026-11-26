#!/usr/bin/env python
"""semantic-lift-245m-confirm-v1 evaluator tests (CPU; prereg section 4.2).
  T-ORD   slc_lib.per_expert_slc ordinary interaction == slk_lib.per_expert_boot exactly (tiny random LFR model,
          CALIB-SEARCH rows for discovery (half A) and measurement (half B); no VAL data)
  T-COND  synthetic: planted own effect recovered; pure exposure confound -> ~0 (naive difference large)
  T-HB    hierarchical bootstrap sanity
  T-SMOKE slc_eval.py end-to-end on CPU, test mode (rows-limit 3), reused C6_s0 p100, output in .tmp (deleted;
          values not inspected)
Writes artifacts/semantic-lift-245m-confirm-v1/eval_tests_<UTC>.json (0400)."""
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402

RES = []


def rec(name, ok, **kw):
    RES.append({"test": name, "ok": bool(ok), **kw})
    print(name, ok, json.dumps(kw, default=str)[:400], flush=True)


def t_ord():
    import numpy as np
    import torch
    from csr.fsp_lib import Split
    from csr.lfr_eval import eval_rows, evaluate, paper_half
    from csr.lfr_model import LFRConfig, LFRModel
    from csr.slc_lib import per_expert_slc
    from csr.slk_lib import per_expert_boot
    torch.manual_seed(0)
    tiny = {"n_layer": 3, "d_model": 64, "n_head": 2, "dense_ffn": 128, "n_experts": 8, "expert_ffn": 32,
            "shared_ffn": 64, "vocab_size": 50304, "top_k": 4, "router": "linear", "teacher": False}
    m = LFRModel(LFRConfig.from_dict({**LFRConfig().to_dict(), **tiny}), 3).float()
    dev = torch.device("cpu")
    cs = Split("calib_search")
    drows = eval_rows(cs, 3000, paper_half(cs, 0))
    prow = eval_rows(cs, 3000, paper_half(cs, 1))
    disc = evaluate(m, cs, drows, dev, per_token=True)["per_token"]
    ref = per_expert_boot(m, cs, drows, cs, prow, dev, disc, {"domain": (disc["dom"], lambda b: b["dom"], 5)},
                          n_boot=200)["domain"]
    summ, arr = per_expert_slc(m, cs, drows, cs, prow, dev, disc["topi"], disc["dom"], 5, n_boot=200)
    a, b = np.asarray(ref["interaction"], float), arr["inter"]
    same_nan = np.array_equal(np.isnan(a), np.isnan(b))
    ok_i = same_nan and np.array_equal(a[np.isfinite(a)], b[np.isfinite(b)])
    ci = np.asarray(ref["ci95"], float)
    ok_ci = np.allclose(ci[:, 0], arr["ci_lo"], equal_nan=True, rtol=0, atol=0) and \
        np.allclose(ci[:, 1], arr["ci_hi"], equal_nan=True, rtol=0, atol=0)
    # bootstrap means: same per-cell replicates; the reference averages with NaN columns included (np.nanmean over all
    # cells), slc over defined cells only -> identical up to floating summation order (tolerance 1e-12 relative)
    rb = np.asarray(ref["boot_mean_samples"])
    boot_maxrel = float(np.max(np.abs(rb - arr["boot_mean"]) / np.maximum(1e-30, np.abs(rb))))
    ok_b = boot_maxrel <= 1e-12
    ok_m = ref["mean_interaction"] == summ["mean_interaction"] and ref["n_subject_specialized"] == summ["n_specialized"]
    ok_k = list(ref["kstar"]) == arr["kstar"].tolist()
    rec("T-ORD ordinary interaction identical to slk_lib.per_expert_boot", ok_i and ok_ci and ok_b and ok_m and ok_k,
        per_cell=ok_i, ci=ok_ci, boot=ok_b, boot_maxrel=boot_maxrel, means=ok_m, kstar=ok_k, n_defined=summ["n_defined"],
        mean=summ["mean_interaction"], cond_n_adequate=summ["cond_n_adequate"],
        cond_support=summ["cond_support_median"])


def t_cond():
    import numpy as np
    import torch
    from csr.slc_lib import _cond_from, _cond_stats
    rng = np.random.default_rng(1)
    N, D = 200_000, 400
    dinv = rng.integers(0, D, N)
    strat = rng.integers(0, 50, N)
    eff = 0.02 * strat  # dCE depends on the stratum (exposure: gate / margin / rank)
    own = rng.random(N) < 0.2
    d = eff + 0.5 * own + rng.normal(0, 0.1, N)
    dev = torch.device("cpu")
    W = torch.ones(1, D, dtype=torch.float64)

    def est(d_, own_, strat_):
        os_, oc, xs, xc = _cond_stats(d_, own_, ~own_, dinv, D, strat_, dev)
        I, _ = _cond_from(W @ os_, W @ oc, W @ xs, W @ xc)
        return float(I[0])
    i1 = est(d, own, strat)
    # pure exposure confound: own tokens concentrated in high strata, no own effect
    own2 = rng.random(N) < (0.05 + 0.5 * (strat >= 40))
    d2 = eff + rng.normal(0, 0.1, N)
    i2 = est(d2, own2, strat)
    naive = float(d2[own2].mean() - d2[~own2].mean())
    # strata present only on the own side are ignored (exact matching)
    strat3 = strat.copy()
    strat3[own & (strat == 0)] = 999
    i3 = est(d, own, strat3)
    rec("T-COND conditional estimator", abs(i1 - 0.5) < 0.005 and abs(i2) < 0.005 and naive > 0.2 and abs(i3 - 0.5) < 0.005,
        planted=i1, confound=i2, naive_confound=naive, own_only_stratum=i3)


def t_hb():
    import numpy as np
    from csr.slc_lib import hier_boot
    X = np.full((4, 500), 0.3)
    r1, _ = hier_boot(X, 2000, point=[0.3] * 4)
    rng = np.random.default_rng(0)
    Y = np.asarray([0.1, 0.2, 0.3, 0.4])[:, None] + rng.normal(0, 0.01, (4, 500))
    r2, reps = hier_boot(Y, 4000, point=[0.1, 0.2, 0.3, 0.4])
    Z = np.asarray([1.0, -1.0, 1.0, -1.0])[:, None] + rng.normal(0, 0.01, (4, 500))
    r3, _ = hier_boot(Z, 4000, point=[1, -1, 1, -1])
    ok = (r1["ci95"] == [0.3, 0.3] and abs(r2["point"] - 0.25) < 1e-12 and r2["ci95"][0] < 0.2 < 0.3 < r2["ci95"][1]
          and r3["ci95"][0] < 0 < r3["ci95"][1])
    rec("T-HB hierarchical bootstrap", ok, constant=r1, spread=r2, sign_mixed=r3)


def t_smoke():
    from csr.slc_paths import TMP
    out = TMP / "smoke_eval"
    shutil.rmtree(out, ignore_errors=True)
    env = {**os.environ, "MOE_RESEARCH_APPROVE": "cpu-model", "CUDA_VISIBLE_DEVICES": ""}
    t0 = time.time()
    r = subprocess.run([str(guard.ROOT / ".venv/bin/python"), "slc_eval.py", "--key", "C6_s0", "--tag", "p100",
                        "--device", "cpu", "--rows-limit", "3", "--out-dir", str(out), "--execute"],
                       env=env, capture_output=True, text=True, cwd=guard.ROOT)
    files = sorted(p.name for p in out.glob("*")) if out.exists() else []
    keys = []
    if (out / "C6_s0_p100.json").exists():
        d = json.loads((out / "C6_s0_p100.json").read_text())
        keys = sorted(d.keys())
        fkeys = sorted(d["function"].keys())
    ok = r.returncode == 0 and "C6_s0_p100.json" in files and "C6_s0_p100.npz" in files and "function" in keys
    rec("T-SMOKE slc_eval end-to-end (CPU test mode; values not inspected)", ok, rc=r.returncode, files=files,
        keys=keys, function_keys=fkeys if keys else None, seconds=time.time() - t0,
        stderr_tail=r.stderr[-1500:] if r.returncode else "")
    if out.exists():
        for p in out.glob("*"):
            os.chmod(p, 0o600)
        shutil.rmtree(out)


def main():
    guard.check_root()
    import torch
    torch.set_num_threads(16)
    from csr.slc_paths import ART, check_prereg
    check_prereg()
    names = [a for a in sys.argv[1:] if a.startswith("t_")] or ["t_cond", "t_hb", "t_ord", "t_smoke"]
    t0 = time.time()
    for n in names:
        try:
            globals()[n]()
        except Exception:
            rec(n + " crashed", False, error=traceback.format_exc()[-2500:])
    doc = {"time": guard.now(), "seconds": time.time() - t0, "n": len(RES), "n_fail": sum(not r["ok"] for r in RES),
           "results": RES}
    out = ART / f"eval_tests_{'_'.join(names)}_{guard.utc_stamp()}.json"
    guard.atomic_write_json(out, doc, mode=0o400)
    print(json.dumps({"n": doc["n"], "n_fail": doc["n_fail"], "out": str(out)}))
    sys.exit(0 if doc["n_fail"] == 0 else 1)


if __name__ == "__main__":
    main()
