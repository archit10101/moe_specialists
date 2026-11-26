#!/usr/bin/env python
"""csr-conditional-local-attention-v1 POST-HOC diagnostic (not preregistered; informs no decision): learned-gate
behaviour of the Gen-2 G256-10 checkpoints (seeds 98, 99) on CALIB-SEARCH only (development data, read after the search
froze; validation is not reused, test never). Writes artifacts/conditional-local-attention-v1/posthoc_gate_<stamp>.json."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402


def main():
    guard.check_root()
    if not guard.require_execution("--execute" in sys.argv, "gpu"):
        return
    import torch
    from cla_analyze import token_analysis
    from csr.cla_model import build_cla_model
    from csr.cla_run import ART, ST_DIR, check_cfg, cla_evaluate, eval_source
    guard.occupancy_gate("clav1 posthoc gate")
    dev = guard.setup_cuda_device()
    guard.register_self("clav1 posthoc gate")
    t_iso = guard.now()
    runs = {ln.split("\t")[0]: ln.split("\t")[1] for ln in (ST_DIR / "runs.tsv").read_text().splitlines()}
    out = {"note": "POST-HOC, not preregistered; CALIB-SEARCH only; no decision depends on it"}
    evs = []
    for s in (98, 99):
        rd = guard.ROOT / runs[f"gen2/seed{s}/G256-10"]
        cfg = json.loads((rd / "config.json").read_text())
        d = check_cfg(cfg)
        m = build_cla_model(cfg, dev)
        m.load_state_dict(torch.load(rd / "model.pt", map_location=dev, weights_only=False)["model"])
        _, src = eval_source(cfg, "calib_search", d)
        e = cla_evaluate(m, src, dev, 16, None, autocast=True, diagnostics=True, token_stats=True)
        evs.append(e)
        out[f"seed{s}"] = {k: e[k] for k in ("ce_nats_per_token", "global_query_rate_by_layer", "global_query_rate_mean",
                                             "eligible_fraction", "global_rate_among_eligible_by_layer",
                                             "global_query_rate_by_layer_and_position", "thresholds",
                                             "gate_logit_stats_by_layer", "keys_attended_per_token_by_layer")}
        out[f"seed{s}"]["run"] = str(rd.relative_to(guard.ROOT))
    out["token_analysis"] = token_analysis(evs, min_count=200)
    p = ART / f"posthoc_gate_{guard.utc_stamp()}.json"
    guard.atomic_write_json(p, out)
    guard.gpu_ledger(t_iso, guard.now(), p.stem, "clav1 posthoc gate", "COMPLETE")
    print("POSTHOC", p)


if __name__ == "__main__":
    main()
