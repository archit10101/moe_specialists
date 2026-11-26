"""csr-forecast-latent-edl-v1 (fle) paths and constants. New module."""
import json

from csr import guard

EXP = "csr-forecast-latent-edl-v1"
ART = guard.ROOT / "artifacts/forecast-latent-edl-v1"
TMP = guard.ROOT / ".tmp/forecast-latent-edl-v1"
CFG_DIR = guard.ROOT / "configs/forecast-latent-edl-v1"
CACHE = ART / "cache"
LATENT = ART / "latent"
PREREG = guard.ROOT / "docs/FORECAST_LATENT_EDL_V1_PREREG.md"
STOP_MARKER = ART / "STOP"
RUN_PREFIX = "forecast_latent_edl_v1_"
DRAFT_G0 = "runs/20261111T085851Z_fsp_draft_shared/ckpt_final.pt"
DRAFT_G0_SHA = "9cafd9968f8d82606eba4c819fef6ddb3d8e7563a2abf9b21f1a91c93dc4a094"
MINILM = guard.ROOT / "artifacts/encoder/all-MiniLM-L6-v2"
T4_S0_RUN = "runs/20261112T165936Z_latent_forecast_router_distill_v1_final_TOKEN-4_s0"
M = 1_000_000
FINAL_BUDGET = 245 * M
STAGE_TOKENS = {"s5": 5 * M, "s20": 20 * M, "s50": 50 * M}
UPPER = list(range(5, 11))  # MoE indices 5..10 (model blocks 6..11)
ALL = list(range(11))


def prereg_sha():
    return guard.sha256_file(PREREG)


def drafts():
    p = ART / "drafts.json"
    return json.loads(p.read_text()) if p.exists() else None
