#!/usr/bin/env python
"""fspv2: equivalence + causality + throughput of draft_rollout_fast (flash/LSE-merge, single-document rows) against
the reference draft_rollout (masked fp32 tree attention) on CALIB-SEARCH documents with a trained domain draft.
Writes artifacts/forecast-specialist-pretraining-v2/rollout_fast_check_<stamp>.json. Dry-run by default."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa

def main():
    guard.check_root()
    if not guard.require_execution("--execute" in sys.argv, "gpu"):
        return
    import torch
    from csr.fsp_lib import ART, Split, build_draft, draft_rollout, draft_rollout_fast, draft_score_fast, make_batch, to_torch
    guard.occupancy_gate("fsp rollout check")
    dev = guard.setup_cuda_device()
    guard.register_self("fsp rollout check")
    t_iso, t0 = guard.now(), time.time()
    paths = json.loads((ART / "draft_paths.json").read_text())
    d = build_draft(0)
    d.load_state_dict(torch.load(guard.ROOT / paths[0], map_location="cpu", weights_only=True)["model"])
    d = d.to(dev).eval()
    sp = Split("calib_search")
    docs = list(range(0, sp.n_docs, max(1, sp.n_docs // 16)))[:16]
    res = {}
    rows = [[i] for i in docs]
    bt = to_torch(make_batch(rows, sp), dev)
    L = torch.as_tensor(sp.lengths[docs], device=dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        r0, e0, t0_, l0 = draft_rollout(d, bt["input_ids"], bt["doc_ids"], bt["positions"], H=8)
    r1, e1, t1, l1 = draft_rollout_fast(d, bt["input_ids"], L, H=8)
    ls = draft_score_fast(d, bt["input_ids"], L)
    v = bt["doc_ids"] >= 0
    res["roll_agreement_by_step"] = [float((r0[..., k] == r1[..., k])[v].float().mean()) for k in range(8)]
    res["lp_maxdiff"] = float((l0 - l1)[v].abs().max())
    res["lp_meandiff"] = float((l0 - l1)[v].abs().mean())
    res["lp_score_vs_fast_maxdiff"] = float((ls - l1)[v].abs().max())
    agree_prefix = torch.ones_like(r0[..., 0], dtype=torch.bool)
    cond = []
    for k in range(8):
        m_ = agree_prefix & v
        cond.append(float((r0[..., k] == r1[..., k])[m_].float().mean()))
        agree_prefix = agree_prefix & (r0[..., k] == r1[..., k])
    res["roll_conditional_agreement_by_step"] = cond
    res["ent_step1_meandiff"] = float((e0[..., 0].float() - e1[..., 0].float())[v].abs().mean())
    res["ent_meandiff"] = float((e0.float() - e1.float())[v].abs().mean())
    res["top4_meandiff"] = float((t0_.float() - t1.float())[v].abs().mean())
    # causality of the fast path
    t = 500
    ids2 = bt["input_ids"].clone()
    ids2[:, t + 1:] = torch.randint(2, 50000, ids2[:, t + 1:].shape, device=dev)
    r2, e2, _, l2 = draft_rollout_fast(d, ids2, L, H=8)
    res["fast_causal_roll_prefix_equal"] = bool(torch.equal(r1[:, :t + 1], r2[:, :t + 1]))
    res["fast_causal_lp_prefix_maxdiff"] = float((l1[:, :t + 1] - l2[:, :t + 1]).abs().max())
    rows3 = [[i] for i in list(range(0, sp.n_docs, max(1, sp.n_docs // 48)))[:48]]
    bt3 = to_torch(make_batch(rows3, sp), dev)["input_ids"]
    L3 = torch.as_tensor([int(sp.lengths[r[0]]) for r in rows3], device=dev)
    for name, fn in (("ref", lambda: draft_rollout(d, bt["input_ids"], bt["doc_ids"], bt["positions"], H=8)),
                     ("fast", lambda: draft_rollout_fast(d, bt["input_ids"], L, H=8)),
                     ("fast_b48", lambda: draft_rollout_fast(d, bt3, L3, H=8)),
                     ("score_fast", lambda: draft_score_fast(d, bt["input_ids"], L))):
        for _ in range(2):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                fn()
        torch.cuda.synchronize(); a = time.time()
        for _ in range(3):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                fn()
        torch.cuda.synchronize()
        res[f"tok_per_s_{name}"] = 3 * int((L3 if name == "fast_b48" else L).sum()) / (time.time() - a)
    res["pass"] = (res["fast_causal_roll_prefix_equal"] and res["fast_causal_lp_prefix_maxdiff"] < 1e-3
                   and min(res["roll_conditional_agreement_by_step"]) > 0.97 and res["lp_meandiff"] < 0.02
                   and res["ent_step1_meandiff"] < 0.02)
    guard.atomic_write_json(ART / f"rollout_fast_check_{guard.utc_stamp()}.json", res)
    t_end = guard.now()
    guard.gpu_ledger(t_iso, t_end, "fsp_rollout_check", "fsp rollout check", "PASS" if res["pass"] else "FAIL")
    with open(ART / "compute_ledger.tsv", "a") as f:
        f.write(f"{t_iso}\t{t_end}\t{time.time() - t0:.1f}\tfsp_rollout_check\trollout_check\tchecks_and_tests\t{res['pass']}\n")
    print(json.dumps(res, indent=1))
    sys.exit(0 if res["pass"] else 1)

main()
