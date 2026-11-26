"""csr-gradient-consistent-latent-edl-v1 (gc) paths and constants. New module (docs/GC_EDL_V1_PREREG.md)."""
from csr import guard

EXP = "csr-gradient-consistent-latent-edl-v1"
ART = guard.ROOT / "artifacts/gc-edl-v1"
TMP = guard.ROOT / ".tmp/gc-edl-v1"
CFG_DIR = guard.ROOT / "configs/gc-edl-v1"
PREREG = guard.ROOT / "docs/GC_EDL_V1_PREREG.md"
PREREG_SHA = "75591d88f264155cb4afd73bb35c9e4fc6292d042c2d2b14d98cdb6ab5643e6d"
STOP_MARKER = ART / "STOP"
RUN_PREFIX = "gc_edl_v1_"
DISC = ART / "discovery"
SEALED = ART / "sealed"
FP = ART / "fingerprints"
LAT = ART / "latent"
FLE_ART = guard.ROOT / "artifacts/forecast-latent-edl-v1"
FLE_CFG = guard.ROOT / "configs/forecast-latent-edl-v1"
LFR_LABELS = guard.ROOT / "artifacts/latent-forecast-router-distill-v1/labels"
SCOUT_PATH = "runs/20261114T080309Z_forecast_latent_edl_v1_draft_G_s11/ckpt_final.pt"
SCOUT_SHA = "3544d8824e6b681f997e3fa14becc61e61541541077710ee8924d1829483ac17"
THETA_NAME = "blocks.9.ffn.w2.weight"
BGE_DIR = guard.ROOT / "artifacts/encoder/bge-small-en-v1.5"
M_PROJ = 128
PROJ_SEED = 20261115
VIEW = 256
M = 1_000_000
FINAL_BUDGET = 245 * M
S50_STEP = 793
ALL_LAYERS = list(range(11))
EDL_BETA = 5e-3
T4_S100_RUN = "runs/20261114T090727Z_forecast_latent_edl_v1_s_T4_s100"
REPS = ("SEM", "GRAD", "COMBINED")
KS = (4, 8, 16)
PROTO_SEEDS = (0, 1, 2)


def prereg_sha():
    return guard.sha256_file(PREREG)


def check_prereg():
    h = prereg_sha()
    if h != PREREG_SHA:
        raise guard.GuardError(f"preregistration hash mismatch: {h}")
    return h
