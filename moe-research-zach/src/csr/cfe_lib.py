"""csr-context-prior-fast-engineering (cfe): fast candidate-engineering phase for the context-informed chunk prior,
started 2026-11-13 on user instruction after the V2 (cps) finals were paused (operator stop, resumable).

Reuses WITHOUT recomputation: V1/V2 drafts + forecast cache (via cps_lib / lfr_lib), splits, stream, tokenizer,
backbone, V2 model code (cps_model, subclassed), V2 audit code, and the recorded TOKEN-4 trajectories / checkpoints /
evaluations (V1 s1_TOKEN-4_s100, s2_TOKEN-4_s100/s101, final_TOKEN-4_s0; V2 final_TOKEN_s1). TOKEN-4 is never
retrained here. V1 / V2 / fspv2 files are never written. Engineering uses CALIB-SEARCH only.
"""
from __future__ import annotations

from . import guard
from .cps_lib import (BUDGET, CACHE_DIR, ForecastCache, build_features, cf_uniform, gather_raw, lr_at,  # noqa: F401
                      main_stream, phase_at, teacher_end_ok, V1_ART, V1_RUN_PREFIX)
from .cps_lib import ART as V2_ART, RUN_PREFIX as V2_RUN_PREFIX  # noqa: F401

EXP = "csr-context-prior-fast-engineering"
ART = guard.ROOT / "artifacts/context-prior-fast-engineering"
CFG_DIR = guard.ROOT / "configs/context-prior-fast-engineering"
TMP = guard.ROOT / ".tmp/context-prior-fast-engineering"
STOP_MARKER = ART / "STOP"
RUN_PREFIX = "context_prior_fast_engineering_"
PLAN = guard.ROOT / "docs/CONTEXT_PRIOR_FAST_ENGINEERING_PLAN.md"


def plan_sha():
    return guard.sha256_file(PLAN)
