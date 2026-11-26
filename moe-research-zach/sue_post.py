#!/usr/bin/env python
"""csr-semantic-utility-edl-v1 post-terminal DESCRIPTIVE evaluations (not part of any preregistered decision):
dev-mode (CALIB-SEARCH half B) evaluation of reused 50M / 20M TOKEN-4 and explicit-EDL checkpoints at the search seeds
(same evaluator, same rows) so that every candidate can be compared with TOKEN-4 / EDL. Checkpoints are read-only."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402

REUSE = {"POST_T4_s100": "runs/20261114T090727Z_forecast_latent_edl_v1_s_T4_s100",
         "POST_T4_s101": "runs/20261115T103553Z_semantic_edl_v1_b_T4_s101",
         "POST_EDL_s100": "runs/20261115T083452Z_semantic_edl_v1_b_EDL_s100",
         "POST_EDL_s101": "runs/20261115T104319Z_semantic_edl_v1_b_EDL_s101"}


def main():
    guard.check_root()
    from csr.sue_paths import check_prereg
    check_prereg()
    import sue_run as R
    for tag in ("s50", "s20"):
        for key, rd in REUSE.items():
            if tag == "s20" and not key.endswith("s100"):
                continue
            R.evaluate(rd, "dev", tag, "post", key=key)


if __name__ == "__main__":
    main()
