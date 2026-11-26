"""csr-semantic-edl-long-adaptive-k-v1 (slk) paths and constants (docs/SEMANTIC_EDL_LONG_ADAPTIVE_K_PREREG.md)."""
from csr import guard

EXP = "csr-semantic-edl-long-adaptive-k-v1"
ART = guard.ROOT / "artifacts/semantic-edl-long-k"
TMP = guard.ROOT / ".tmp/semantic-edl-long-k"
CFG_DIR = guard.ROOT / "configs/semantic-edl-long-k"
PREREG = guard.ROOT / "docs/SEMANTIC_EDL_LONG_ADAPTIVE_K_PREREG.md"
PREREG_SHA_FILE = guard.ROOT / "docs/SEMANTIC_EDL_LONG_ADAPTIVE_K_PREREG.md.sha256"
STOP_MARKER = ART / "STOP"
RUN_PREFIX = "semantic_edl_long_k_"
GC_LAT = guard.ROOT / "artifacts/gc-edl-v1/latent"
SEMV1 = guard.ROOT / "artifacts/semantic-edl-v1"
# Phase-1 assignment tables (recovered C6 recipe; hashes verified at use)
C6_TABLE = "artifacts/semantic-edl-v1/tables/table_SEM_T05.npz"
C6_TABLE_SHA = "5d23129b2cb7aed6a0d2ed768b1a755eac661dae95ddeace89329dc7b56b89d2"
C6_SHUF_TABLE = "artifacts/semantic-edl-v1/tables/table_SHUF_SEM_T05.npz"
C6_SHUF_TABLE_SHA = "d374a4ec45db7632ffd865b6101b51cbcaeab35fbb1575b140cb92b5e35c69fe"
C6_BETA = 0.005
SEM_CS = GC_LAT / "table_SEM_calib_search.npz"
SEM_CS_SHA = "abdd94c5112810df6705aa67a99be915d8223366a57d3b216ae4927978b5e69d"
SEM_SEL = GC_LAT / "table_SEM_calib_sel.npz"
SEM_SEL_SHA = "2e4c4e747804eae99c606c45612197a16428ddba89bddf84d8f5e9c26c20524a"
SEM_TABLE = GC_LAT / "table_SEM.npz"
SEM_TABLE_SHA = "308f1e02f17e5e759fde45f1fc28e650acac41389974f1fcf14cb6bad7439da4"
FLE_ART = guard.ROOT / "artifacts/forecast-latent-edl-v1"
LFR_LABELS = guard.ROOT / "artifacts/latent-forecast-router-distill-v1/labels"
REUSED = {"T4_s0": "runs/20261114T131547Z_forecast_latent_edl_v1_final_T4_s0",
          "T4_s1": "runs/20261114T145059Z_forecast_latent_edl_v1_final_T4_s1",
          "T4_s2": "runs/20261114T221957Z_forecast_latent_edl_v1_final_T4_s2",
          "EDL_s0": "runs/20261114T173920Z_forecast_latent_edl_v1_final_EDL_s0",
          "EDL_s1": "runs/20261114T191747Z_forecast_latent_edl_v1_final_EDL_s1"}
EDL_BETA = 5e-3
N_STEPS = 3880
M = 1_000_000
FINAL_BUDGET = 245 * M
# milestones: ~20M, ~50M, then exactly the reused baselines' p040/p060/p080/p100 steps (98.0M/147.1M/196.1M/245.0M)
MILESTONES = {"s20": 318, "s50": 793, "p040": 1552, "p060": 2328, "p080": 3104, "p100": 3880}
SEEDS_P1 = (0, 1, 2, 3)
GPU_CAP_H = {"total": 60.0, "phase1": 28.0, "phase2a": 12.0, "phase2b": 20.0}
DISK_RESERVE_GIB = 100.0  # guard reserve (stricter than the user's 50 GiB minimum)


def prereg_sha():
    return guard.sha256_file(PREREG)


def check_prereg():
    want = PREREG_SHA_FILE.read_text().split()[0]
    h = prereg_sha()
    if h != want:
        raise guard.GuardError(f"preregistration hash mismatch: {h} != {want}")
    return h
