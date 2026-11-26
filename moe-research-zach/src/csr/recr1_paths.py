"""residual-evidence-change-router-v1 (recr1) paths and frozen constants
(docs/RESIDUAL_EVIDENCE_CHANGE_ROUTER_V1_PREREG.md).

RECR is evaluated by EXACT offline simulation over the retained spb1 per-token CE tables: the spb1 residual bank sits
after the final transformer block of a frozen base, so the CE of any routing policy at token t is exactly the table
entry of the specialist it executes at t (no cross-position effect). The frozen TOKEN-4 base checkpoints (seeds 0-2)
that produced those tables were deleted in the disk cleanup (audit in OBSERVATIONS.md); nothing that needs the base
(hidden states, new partitions, VAL, 245M) can be run without user action."""
from csr import guard

EXP = "residual-evidence-change-router-v1"
ART = guard.ROOT / "artifacts/residual-evidence-change-router-v1"
CFG_DIR = guard.ROOT / "configs/residual-evidence-change-router-v1"
TMP = guard.ROOT / ".tmp/residual-evidence-change-router-v1"
HEADS = ART / "heads"
SIMS = ART / "sims"
ANALYSIS = ART / "analysis"
PREREG = guard.ROOT / "docs/RESIDUAL_EVIDENCE_CHANGE_ROUTER_V1_PREREG.md"
PREREG_SHA_FILE = guard.ROOT / "docs/RESIDUAL_EVIDENCE_CHANGE_ROUTER_V1_PREREG.md.sha256"
REPORT = guard.ROOT / "docs/RESIDUAL_EVIDENCE_CHANGE_ROUTER_V1_REPORT.md"
STOP_MARKER = ART / "STOP"

# ---- reused (read-only, hash-verified in reuse_manifest.json)
SPB1 = guard.ROOT / "artifacts/semantic-pair-bank-unrestricted-router-v1"
EXTRACT = SPB1 / "extract"
ROUTES = SPB1 / "routers/S2_g1"     # spb1 generation-1 deployed routers (bal 0.001): the current unrestricted router
BANK_SEEDS = (0, 1, 2)
ROUTER_SEEDS = (0, 1)              # spb1 trained router seeds 0 and 1 only; seed 2 cannot be trained (no base)
BANKS = ("SEM2", "SHUF2", "GEN32")
E = 32

# ---- partitions: paper-level split of spb1 SEL (= CALIB-SEARCH half A, 503 papers, 6.40M tokens)
SPLIT_SALT = "recr1-split-20261123"
SPLITS = {"UTRAIN": range(0, 4), "CSEARCH": range(4, 7), "CSELECT": range(7, 10)}   # sha256(salt+paper) mod 10

# ---- architecture (TOKEN-4 245M: d 384, 12 layers, 32 experts top-4 + shared 768, dense FFN 1792 in layer 0,
# vocab 50304, T 1024) for analytic FLOPs; residual 384->256->384; spb1 router 76,192 params
ARCH = {"d": 384, "L": 12, "dense_ffn": 1792, "shared": 768, "expert_ffn": 256, "topk": 4, "n_exp": 32,
        "vocab": 50304, "T": 1024, "res_ffn": 256, "router_params": 76192}

# ---- head (candidate proposal) training
HEAD_TOKENS = 50_000_000
HEAD_BATCH = 8192
HEAD_LR = 3e-3
HEAD_WD = 1e-4
HEAD_CKPT_TOKENS = 2_500_000
HEAD_CKPT_SECONDS = 600
HUBER = 0.25

# ---- RECR generation-0 configuration (prereg section 4)
GEN0 = {"gamma": 0.95, "tau": 0.0, "kappa_rate": 4.0, "B": 8, "n_cand": 2, "head": "linear", "norm": "raw",
        "abstain": 0.0, "fallback": "generic", "stride_cm": 32}
PERMITTED = {"gamma", "tau", "kappa_rate", "B", "head", "norm", "abstain", "fallback"}
MAX_GENERATIONS = 4
GATE_FIRST32_TOL = 0.001
GATE_STABLE_RET = 0.75
OVERHEAD_TARGET = 0.05
BOOT_SEED, N_BOOT = 20261501, 2000
GPU_CAP_H = 24.0
DISK_RESERVE_GIB = 100.0


import os as _os
SMOKE = _os.environ.get("RECR1_SMOKE") == "1"      # code smoke test: tiny UTRAIN subset, outputs under TMP only
if SMOKE:
    HEADS = TMP / "smoke/heads"
    ANALYSIS = TMP / "smoke/analysis"
    HEAD_TOKENS = 400_000
    SMOKE_PAPERS = 24


def prereg_sha():
    return guard.sha256_file(PREREG)


def check_prereg():
    if SMOKE:
        return "smoke"
    want = PREREG_SHA_FILE.read_text().split()[0]
    h = prereg_sha()
    if h != want:
        raise guard.GuardError(f"prereg hash mismatch: {h} != {want}")
    return h


def gen_cfg(gen):
    import json
    c = dict(GEN0)
    for g in range(1, gen + 1):
        c.update(json.loads((ART / f"gen{g}_change.json").read_text())["change"])
    return c
