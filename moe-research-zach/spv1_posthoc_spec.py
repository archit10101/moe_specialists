#!/usr/bin/env python
"""csr-speculative-pool-routing-v1 post-hoc diagnostic (not preregistered; CPU; CALIB-SEARCH rows only; read-only on
checkpoints). For a TRAINED spec-family checkpoint:
  L1 future-token perturbation: logits / pools / block starts / routes at positions <= cut are unchanged when every
     later real token is replaced (proposals recomputed from the perturbed text by the frozen draft).
  M1 mechanism: CE on the same rows with (a) the real causal draft proposals, (b) proposals replaced by random
     tokens, (c) proposals shifted to a different document (block structure and pool features no longer informative).
Writes artifacts/speculative-pool-routing-v1/posthoc_spec_<run>.json once."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    if not guard.require_execution(a.execute, "cpu-model"):
        return
    import numpy as np
    import torch
    import torch.nn.functional as F
    from csr.data import to_torch
    from csr.spv1_model import build_pool_model, load_frozen_draft
    from csr.spv1_run import ART, DATA_DIR, eval_source, frozen_draft
    torch.set_num_threads(8)
    rd = guard.inside_root(a.run)
    cfg = json.loads((rd / "config.json").read_text())
    m = build_pool_model(cfg, "cpu")
    m.load_state_dict(torch.load(rd / "model.pt", map_location="cpu", weights_only=False)["model"])
    m.eval()
    dp, drec = frozen_draft()
    dr = load_frozen_draft(dp, "cpu")
    m.attach_draft(dr)
    _, src = eval_source(cfg, "calib_search", guard.ROOT / DATA_DIR)
    tb = to_torch(src.batch(src.rows[:a.rows]), "cpu")
    res = {"run": str(rd.relative_to(guard.ROOT)), "candidate": cfg["candidate"], "rows": a.rows}
    with torch.no_grad():
        base = m(tb, collect=True)
        worst, same = 0.0, True
        rng = np.random.default_rng(0)
        for cut in (0, 3, 50, 255, 511, 800):
            b2 = {k: v.clone() for k, v in tb.items()}
            npd = b2["doc_ids"] >= 0
            fut = torch.zeros_like(npd)
            fut[:, cut + 1:] = True
            b2["input_ids"] = torch.where(fut & npd, torch.as_tensor(rng.integers(2, 50000, size=tuple(npd.shape))),
                                          b2["input_ids"])
            o2 = m(b2, collect=True)
            worst = max(worst, float((base.logits[:, :cut + 1] - o2.logits[:, :cut + 1]).abs().max()))
            for r1, r2 in zip(base.routing, o2.routing):
                for k in ("topi", "cand", "starts"):
                    same &= bool(torch.equal(r1[k][:, :cut + 1], r2[k][:, :cut + 1]))
            same &= bool(torch.equal(base.proposals[:, :cut + 1], o2.proposals[:, :cut + 1]))
        res["L1_future_perturbation"] = {"max_abs_logit_diff": worst, "earlier_pools_routes_starts_identical": same,
                                         "pass": same and worst < 1e-4}

        def ce(out):
            lg = out.logits.float()
            return float(F.cross_entropy(lg.view(-1, lg.shape[-1]), tb["targets"].view(-1), ignore_index=-100))
        res["M1_ce_real_proposals"] = ce(base)
        rp = torch.as_tensor(rng.integers(2, 50000, size=tuple(base.proposals.shape)))
        res["M1_ce_random_proposals"] = ce(m(tb, proposals=rp))
        res["M1_ce_rolled_proposals"] = ce(m(tb, proposals=torch.roll(base.proposals, 1, dims=0)))
        res["teacher_forced_block_rate"] = float(base.spec_starts[tb["doc_ids"] >= 0].float().mean())
    out = ART / f"posthoc_spec_{rd.name}.json"
    guard.atomic_write_json(out, res, mode=0o400)
    guard.ledger("SP_V1_POSTHOC", f"{out.relative_to(guard.ROOT)} {json.dumps(res)[:300]}")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
