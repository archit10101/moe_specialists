"""csr-semantic-utility-edl-v1 (sue) paths and constants (docs/SEMANTIC_UTILITY_EDL_V1_PREREG.md)."""
from csr import guard

EXP = "csr-semantic-utility-edl-v1"
ART = guard.ROOT / "artifacts/semantic-utility-edl-v1"
TMP = guard.ROOT / ".tmp/semantic-utility-edl-v1"
CFG_DIR = guard.ROOT / "configs/semantic-utility-edl-v1"
PREREG = guard.ROOT / "docs/SEMANTIC_UTILITY_EDL_V1_PREREG.md"
PREREG_SHA_FILE = guard.ROOT / "docs/SEMANTIC_UTILITY_EDL_V1_PREREG.md.sha256"
STOP_MARKER = ART / "STOP"
RUN_PREFIX = "semantic_utility_edl_v1_"
SLK_ART = guard.ROOT / "artifacts/semantic-edl-long-k"  # Task-1 (frozen; read-only here)
GC_LAT = guard.ROOT / "artifacts/gc-edl-v1/latent"
FLE_ART = guard.ROOT / "artifacts/forecast-latent-edl-v1"
LFR_LABELS = guard.ROOT / "artifacts/latent-forecast-router-distill-v1/labels"
# fallback grouping (prereg section 3): K16 C6 table, T = 0.5, all MoE layers, beta 0.005
C6_TABLE = "artifacts/semantic-edl-v1/tables/table_SEM_T05.npz"
C6_TABLE_SHA = "5d23129b2cb7aed6a0d2ed768b1a755eac661dae95ddeace89329dc7b56b89d2"
C6_SHUF_TABLE = "artifacts/semantic-edl-v1/tables/table_SHUF_SEM_T05.npz"
C6_SHUF_TABLE_SHA = "d374a4ec45db7632ffd865b6101b51cbcaeab35fbb1575b140cb92b5e35c69fe"
C6_BETA = 0.005
EDL_BETA = 5e-3
M = 1_000_000
FINAL_BUDGET = 245 * M
N_STEPS = 3880
MILESTONES = {"s20": 318, "s50": 793, "p040": 1552, "p060": 2328, "p080": 3104, "p100": 3880}
SEARCH_SEEDS = (100, 101)
FINAL_SEEDS = (0, 1, 2)
GPU_CAP_H = {"total": 45.0, "diagnostic": 2.0, "search": 14.0, "final": 29.0}
DISK_RESERVE_GIB = 100.0


def prereg_sha():
    return guard.sha256_file(PREREG)


def check_prereg():
    want = PREREG_SHA_FILE.read_text().split()[0]
    h = prereg_sha()
    if h != want:
        raise guard.GuardError(f"preregistration hash mismatch: {h} != {want}")
    return h
