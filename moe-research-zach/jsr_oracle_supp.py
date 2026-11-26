#!/usr/bin/env python
"""joint-semantic-residual-pretrain-v1: supplementary oracle functional metric (diagnostic only) after the
oracle-branch fix in jsr_analyze.functional. Writes ART/<out>_oracle_supp.json (0400); never overwrites stage files.
  jsr_oracle_supp.py --evals a.npz,b.npz --out stage2"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402
import jsr_analyze as JA  # noqa: E402


def main():
    import numpy as np
    from csr.jsr_paths import ART, BOOT_SEED, N_BOOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    seeds = JA.load_seeds(a.evals.split(","))
    W = JA.boot_weights(seeds[0].P, N_BOOT, BOOT_SEED)
    F = {b: [JA.functional(S, b, oracle=(b == "E")) for S in seeds] for b in ("B", "C", "D", "E")}
    out = {"time": guard.now(), "evals": a.evals, "note": "oracle I over all swap-subset tokens (135 strata); routed "
           "banks over routed population (2025 strata): E contrasts are descriptive, not like-for-like"}
    for b in F:
        v, r = JA.func_single(F[b], W)
        out[f"I_{b}_sem"] = {**JA.summarize(v, r), "n_scored": [f["n_scored"] for f in F[b]],
                             "I_k": [f.get("I_k") for f in F[b]]}
    for x, y in (("E", "B"), ("E", "D"), ("E", "C")):
        v, r = JA.func_contrast(F[x], F[y], W)
        out[f"I_{x}-I_{y}"] = JA.summarize(v, r)
    p = ART / f"{a.out}_oracle_supp.json"
    guard.atomic_write_json(p, out, mode=0o400)
    print({k: (v["per_seed"], v["pooled_ci"]) for k, v in out.items() if isinstance(v, dict)})


if __name__ == "__main__":
    main()
