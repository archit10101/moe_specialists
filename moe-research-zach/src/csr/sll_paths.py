"""csr-semantic-lift-expert-learning-v1 (sll) paths and constants (docs/SEMANTIC_LIFT_EXPERT_LEARNING_V1_PREREG.md)."""
from csr import guard

EXP = "csr-semantic-lift-expert-learning-v1"
ART = guard.ROOT / "artifacts/semantic-lift-expert-learning-v1"
TMP = guard.ROOT / ".tmp/semantic-lift-expert-learning-v1"
CFG_DIR = guard.ROOT / "configs/semantic-lift-expert-learning-v1"
PREREG = guard.ROOT / "docs/SEMANTIC_LIFT_EXPERT_LEARNING_V1_PREREG.md"
PREREG_SHA_FILE = guard.ROOT / "docs/SEMANTIC_LIFT_EXPERT_LEARNING_V1_PREREG.md.sha256"
STOP_MARKER = ART / "STOP"
RUN_PREFIX = "semantic_lift_expert_learning_v1_"
SUE_ART = guard.ROOT / "artifacts/semantic-utility-edl-v1"  # V1 (terminal; read-only here)
SCU_ART = guard.ROOT / "artifacts/soft-cf-usefulness-v2"  # V2 (terminal; read-only here)
SLK_ART = guard.ROOT / "artifacts/semantic-edl-long-k"  # long Semantic-EDL study (frozen; read-only here)
GC_LAT = guard.ROOT / "artifacts/gc-edl-v1/latent"
FLE_ART = guard.ROOT / "artifacts/forecast-latent-edl-v1"
LFR_LABELS = guard.ROOT / "artifacts/latent-forecast-router-distill-v1/labels"
# frozen grouping = Semantic C6 (K16 SEM, T 0.5, all MoE layers, beta 0.005) and its persistent paper permutation
TABLE = "artifacts/semantic-edl-v1/tables/table_SEM_T05.npz"
TABLE_SHA = "5d23129b2cb7aed6a0d2ed768b1a755eac661dae95ddeace89329dc7b56b89d2"
SHUF_TABLE = "artifacts/semantic-edl-v1/tables/table_SHUF_SEM_T05.npz"
SHUF_TABLE_SHA = "d374a4ec45db7632ffd865b6101b51cbcaeab35fbb1575b140cb92b5e35c69fe"
C6_BETA = 0.005
CALIB_SEARCH_TABLE = "artifacts/gc-edl-v1/latent/table_SEM_calib_search.npz"
CALIB_SEL_TABLE = "artifacts/gc-edl-v1/latent/table_SEM_calib_sel.npz"
M = 1_000_000
FINAL_BUDGET = 245 * M
N_STEPS = 3880
MILESTONES = {"s20": 318, "s50": 793, "p040": 1552, "p060": 2328, "p080": 3104, "p100": 3880}
SEARCH_SEED = 100
PROMO_SEEDS = (100, 101)
FINAL_SEEDS = (0, 1, 2, 3)
LAMBDAS = (0.25, 0.5, 1.0)
GPU_CAP_H = {"total": 40.0, "checks": 1.5, "search": 3.0, "promotion": 6.0, "generations": 5.0, "final": 24.5}
DISK_RESERVE_GIB = 100.0


def prereg_sha():
    return guard.sha256_file(PREREG)


def check_prereg():
    want = PREREG_SHA_FILE.read_text().split()[0]
    h = prereg_sha()
    if h != want:
        raise guard.GuardError(f"preregistration hash mismatch: {h} != {want}")
    return h
