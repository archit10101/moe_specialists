#!/usr/bin/env python
"""fspv2 checkpoint evaluation on VAL (never test). Loads each checkpoint into the INFERENCE EXPORT (teacher deleted,
no draft loaded, no labels except for the LABEL-SPECIALIST-ORACLE) and writes
artifacts/forecast-specialist-pretraining-v2/evals/<run_name>_<what>.json (+ per-doc CE npz for bootstrap).
  --what full : p100 full-VAL CE + routing; functional ablations (all ckpts for spec models, p100 otherwise);
                routing trajectory at p040/p050/p075; swaps for spec models.
  --what ce   : p100 full-VAL CE only.
Dry-run by default; GPU via scripts/gpu_guarded.sh."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402

SUB_TOK = 1_000_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--what", choices=["full", "ce"], default="full")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    if not guard.require_execution(a.execute, "gpu"):
        return
    import numpy as np
    import torch
    from csr.fsp_eval import (eval_rows, evaluate, functional_ablation, posthoc_groups, spec_groups,
                              swap_analysis)
    from csr.fsp_lib import ART, Split
    from csr.fsp_model import FSPConfig, FSPModel, TEACHER_PREFIX
    guard.occupancy_gate("fsp evaluate")
    device = guard.setup_cuda_device()
    mon = guard.OccupancyMonitor(15.0)
    mon.start()
    guard.register_self(f"fsp evaluate {a.run_dir}")
    t_iso, t0 = guard.now(), time.time()
    rd = guard.inside_root(a.run_dir)
    cfg = json.loads((rd / "config.json").read_text())
    summ = json.loads((rd / "summary.json").read_text())
    name = cfg["run_name"]
    oracle = bool(cfg["cond"].get("oracle"))
    (ART / "evals").mkdir(exist_ok=True)
    val = Split("val")
    full_rows = eval_rows(val)
    sub_rows = eval_rows(val, SUB_TOK)
    out = {"run_name": name, "condition": cfg["condition"], "seed": cfg["seed"], "run_dir": a.run_dir,
           "oracle": oracle}

    def load(tag):
        ck = torch.load(guard.ROOT / summ["ckpts"][tag]["path"], map_location="cpu", weights_only=True)
        m = FSPModel(FSPConfig.from_dict(cfg["model"]), cfg["seed"])
        m.load_state_dict({k: v.float() for k, v in ck["model"].items()})
        m.strip_teacher()
        assert not any(n.startswith(TEACHER_PREFIX) for n, _ in m.named_parameters())
        m._eval_use_dom = bool(cfg["cond"].get("use_dom"))
        return m.to(device).eval()

    tags = ["p100"] if a.what == "ce" else [t for t in ("p040", "p050", "p075", "p100") if t in summ["ckpts"]]
    spec = cfg["model"]["routing"] == "spec"
    for tag in tags:
        m = load(tag)
        r = {}
        if tag == "p100":
            v = evaluate(m, val, full_rows, device, oracle=oracle, routing_stats=(a.what == "full"))
            docs = sorted(v.pop("doc_ce").items())
            np.savez_compressed(ART / "evals" / f"{name}_{tag}_val_doc_ce.npz",
                                doc=np.asarray([k for k, _ in docs]), ce_sum=np.asarray([s for _, (s, c) in docs]),
                                n=np.asarray([c for _, (s, c) in docs]))
            r["val"] = v
            r["param_counts_inference"] = m.param_counts()
        if a.what == "full":
            sub = evaluate(m, val, sub_rows, device, oracle=oracle, routing_stats=True)
            sub.pop("doc_ce", None)
            r["val_sub"] = sub
            if spec or tag == "p100":
                if spec:
                    groups = spec_groups(m.n_moe)
                else:
                    cs = Split("calib_search")
                    cr = evaluate(m, cs, eval_rows(cs, SUB_TOK), device, routing_stats=True)
                    groups = posthoc_groups(cr["counts"])
                    r["posthoc_groups"] = {str(k): v for k, v in groups.items()}
                base = {"ce_domain": sub["ce_domain"]}
                r["ablation"] = functional_ablation(m, val, sub_rows, device, groups, oracle=oracle, base=base)
                if spec and tag == "p100":
                    r["swap"] = swap_analysis(m, val, sub_rows, device)
        out[tag] = r
        del m
        torch.cuda.empty_cache()
        print(tag, json.dumps({k: (v["ce"] if isinstance(v, dict) and "ce" in v else None) for k, v in r.items()}),
              flush=True)
    mon.stop()
    if mon.contaminated:
        raise guard.GuardError("CONTAMINATED during eval")
    (ART / "evals").mkdir(exist_ok=True)
    p = ART / "evals" / f"{name}_{a.what}.json"
    guard.atomic_write_json(p, out, overwrite=True)
    t_end = guard.now()
    guard.gpu_ledger(t_iso, t_end, f"fsp_eval_{name}_{a.what}", "fsp evaluate", "COMPLETE")
    with open(ART / "compute_ledger.tsv", "a") as f:
        f.write(f"{t_iso}\t{t_end}\t{time.time() - t0:.1f}\tfsp_eval_{name}\teval_{name}\tevaluation\tCOMPLETE\n")


if __name__ == "__main__":
    main()
