#!/usr/bin/env python
"""csur1 engineering generation 1 (prereg section 12 trigger 5) tests, run before the gen-1 Stage-1 job:
  G1a router 'meanlast' param count 18,216 <= cap; gen-0 'mean' router unchanged (19,152) and block_feats[:, :384] == block_mean
  G1b causality on real tokens: q[v] unchanged by perturbing any token of views >= v of the same document (or other documents),
      using the full block_feats path (mean and last token)
  G1c T5 (base bitwise identical with vs without gen-1 banks) and T6 (bitwise resume) with the gen-1 router.
Writes ART/cpu_tests_gen1_<ts>.json (0400)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402
import csur1_configs as CC  # noqa: E402
import csur1_tests as CT  # noqa: E402


def g1a():
    import torch
    from csr.csur1_bank import UtilBank, block_feats, block_mean
    from csr.csur1_paths import ROUTER_PARAM_CAP
    b0 = UtilBank({"name": "x", "prior": "sem"}, 0)
    b1 = UtilBank({"name": "x", "prior": "sem", "router": "meanlast"}, 0)
    blk = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2])
    x = torch.randn(9, 384)
    f = block_feats(x, blk, 3)
    ok = (b0.n_params()["router"] == 19152 and b1.n_params()["router"] == 18216 <= ROUTER_PARAM_CAP
          and torch.equal(f[:, :384], block_mean(x, blk, 3)) and torch.equal(f[:, 384:], x[[2, 4, 8]]))
    return {"G1a": bool(ok), "router_params": [b0.n_params()["router"], b1.n_params()["router"]]}


def g1b():
    import torch
    from csr.csur1_bank import UtilBank, block_feats, layout
    s = CT.step_tokens()
    lay = layout(s["row"], s["dl"], s["pos"], s["y"] >= 0)
    b = UtilBank({"name": "x", "prior": "sem", "util": True, "c": 1, "eta": 0.5, "ema": 0.9, "lam": 0.3,
                  "router": "meanlast"}, 0)
    torch.manual_seed(3)
    X = torch.randn(lay["blk"].shape[0], 384)
    with torch.no_grad():
        q0 = b.student_logits(block_feats(X, lay["blk"], lay["nb"]), lay)
    ok = True
    n_checked = 0
    for v in range(0, lay["nb"], max(1, lay["nb"] // 25)):
        toks = torch.nonzero(lay["blk"] == v).squeeze(1)
        for t in (toks[0], toks[-1]):
            X2 = X.clone()
            X2[t] += 5 * torch.randn(384)
            with torch.no_grad():
                q1 = b.student_logits(block_feats(X2, lay["blk"], lay["nb"]), lay)
            changed = (q1 - q0).abs().max(1).values > 1e-7
            later = (lay["blk_doc"] == lay["blk_doc"][v]) & (lay["blk_no"] > lay["blk_no"][v])
            ok &= bool((changed <= later).all()) and not bool(changed[v])
            n_checked += 1
    return {"G1b": bool(ok), "n_perturbations": n_checked}


def g1c():
    CC.GEN_ROUTER["router"] = "meanlast"
    r5 = CT.t_bank_vs_nobank()
    r6 = CT.t_resume()
    keys = r5.get("bank_rec_keys", {})
    return {"G1c": bool(r5["T5"] and r6["T6"]), "T5": r5["T5"], "T6": r6["T6"], "T6_detail": r6, "bank_keys": keys}


def main():
    guard.check_root()
    res = {}
    for fn in (g1a, g1b, g1c):
        try:
            r = fn()
        except Exception:  # noqa: BLE001
            import traceback
            r = {fn.__name__.upper(): False, "error": traceback.format_exc()[-3000:]}
        res[fn.__name__] = r
        print(fn.__name__, json.dumps(r, default=str)[:600], flush=True)
    passed = {k: bool(res[k.lower()].get(k, False)) for k in ("G1a", "G1b", "G1c")}
    from csr.csur1_paths import ART
    out = {"time": guard.now(), "generation": 1, "passed": passed, "all_pass": all(passed.values()), "detail": res}
    p = ART / f"cpu_tests_gen1_{guard.utc_stamp()}.json"
    guard.atomic_write_json(p, out, mode=0o400)
    print(p, json.dumps(passed), "ALL PASS" if out["all_pass"] else "FAIL")
    sys.exit(0 if out["all_pass"] else 1)


if __name__ == "__main__":
    main()
