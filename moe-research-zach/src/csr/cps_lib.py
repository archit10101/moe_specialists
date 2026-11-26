"""csr-context-prior-specialization-v2 (cps) shared library: V2 paths + read-only reuse of V1 (lfr) pure functions.

Reused WITHOUT recomputation (hash-verified by cps_verify_reuse.py before any V2 GPU work): the six frozen fspv2 drafts,
the V1 compressed forecast cache (main region = documents of the first 2329 FINAL steps = 60% of FINAL + 1, i.e. every
step of every teacher / anneal phase that ends at <= 60%; calib_search diagnostic region), the fspv2 data splits, the
balanced main stream, the tokenizer, the backbone architecture, the V1 audit / evaluation code and the V1 TOKEN-4,
OUTPUT-DISTILL and MEMORY-ONLY development results. V1 files are never written.
Protocol: docs/CONTEXT_PRIOR_SPECIALIZATION_V2_PROTOCOL.md.
"""
from __future__ import annotations

from . import guard
from .lfr_lib import (BUDGET, CACHE_DIR, DRAFT_PATHS, N_DRAFT, ForecastCache, build_features, cf_uniform,  # noqa: F401
                      gather_raw, lr_at, main_stream, phase_at, phase_end_frac, shuffle_donors)
from .lfr_lib import ART as V1_ART, RUN_PREFIX as V1_RUN_PREFIX  # noqa: F401

EXP = "csr-context-prior-specialization-v2"
ART = guard.ROOT / "artifacts/context-prior-specialization-v2"
CFG_DIR = guard.ROOT / "configs/context-prior-specialization-v2"
TMP = guard.ROOT / ".tmp/context-prior-specialization-v2"
PROTOCOL = guard.ROOT / "docs/CONTEXT_PRIOR_SPECIALIZATION_V2_PROTOCOL.md"
STOP_MARKER = ART / "STOP"
RUN_PREFIX = "context_prior_specialization_v2_"
V1_CACHE_STEPS_FINAL = 2329  # main cache covers the documents of the first 2329 FINAL-plan steps (prefix plans)
TRAIN_SPLITS = ("main",)


def protocol_sha():
    return guard.sha256_file(PROTOCOL)


def teacher_end_ok(schedule, n_steps_final=3880):
    """A schedule may use drafts only on steps covered by the reused V1 cache (prefix plans: S1/S2 are prefixes)."""
    return int(round(phase_end_frac(schedule) * n_steps_final)) <= V1_CACHE_STEPS_FINAL - 1
