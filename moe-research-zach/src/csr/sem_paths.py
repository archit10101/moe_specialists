"""csr-semantic-edl-v1 (sem) paths and constants (docs/SEMANTIC_EDL_V1_PREREG.md)."""
from csr import guard

EXP = "csr-semantic-edl-v1"
ART = guard.ROOT / "artifacts/semantic-edl-v1"
TMP = guard.ROOT / ".tmp/semantic-edl-v1"
CFG_DIR = guard.ROOT / "configs/semantic-edl-v1"
PREREG = guard.ROOT / "docs/SEMANTIC_EDL_V1_PREREG.md"
PREREG_SHA = "63e039330ad37b6937b777c4a0a331b5e587739605b93c797373898265b30cba"
STOP_MARKER = ART / "STOP"
RUN_PREFIX = "semantic_edl_v1_"
GC_LAT = guard.ROOT / "artifacts/gc-edl-v1/latent"
SEM_TABLE = GC_LAT / "table_SEM.npz"
SEM_TABLE_SHA = "308f1e02f17e5e759fde45f1fc28e650acac41389974f1fcf14cb6bad7439da4"
SEM_CS = GC_LAT / "table_SEM_calib_search.npz"
SEM_CS_SHA = "abdd94c5112810df6705aa67a99be915d8223366a57d3b216ae4927978b5e69d"
SEM_SEL = GC_LAT / "table_SEM_calib_sel.npz"
SEM_SEL_SHA = "2e4c4e747804eae99c606c45612197a16428ddba89bddf84d8f5e9c26c20524a"
FLE_ART = guard.ROOT / "artifacts/forecast-latent-edl-v1"
LFR_LABELS = guard.ROOT / "artifacts/latent-forecast-router-distill-v1/labels"
T4_S100_RUN = "runs/20261114T090727Z_forecast_latent_edl_v1_s_T4_s100"
FINAL_RUNS = {"T4_s0": "runs/20261114T131547Z_forecast_latent_edl_v1_final_T4_s0",
              "T4_s1": "runs/20261114T145059Z_forecast_latent_edl_v1_final_T4_s1",
              "T4_s2": "runs/20261114T221957Z_forecast_latent_edl_v1_final_T4_s2",
              "EDL_s0": "runs/20261114T173920Z_forecast_latent_edl_v1_final_EDL_s0",
              "EDL_s1": "runs/20261114T191747Z_forecast_latent_edl_v1_final_EDL_s1"}
EDL_BETA = 5e-3
BETA0 = 5e-3
UPPER = list(range(5, 11))
ALL = list(range(11))
S5, S20, S50 = 80, 318, 793
M = 1_000_000
FINAL_BUDGET = 245 * M
GPU_BUDGET_H = 24.0


def prereg_sha():
    return guard.sha256_file(PREREG)


def check_prereg():
    h = prereg_sha()
    if h != PREREG_SHA:
        raise guard.GuardError(f"preregistration hash mismatch: {h}")
    return h
