"""semantic-lift-245m-confirm-v1 (slc) paths and constants (docs/SEMANTIC_LIFT_245M_CONFIRM_V1_PREREG.md).
Confirmatory 245M test of the frozen Semantic Lift-Weighted Expert Learning configuration (sll, lambda 0.5)."""
from csr import guard

EXP = "semantic-lift-245m-confirm-v1"
ART = guard.ROOT / "artifacts/semantic-lift-245m-confirm-v1"
TMP = guard.ROOT / ".tmp/semantic-lift-245m-confirm-v1"
CFG_DIR = guard.ROOT / "configs/semantic-lift-245m-confirm-v1"
PREREG = guard.ROOT / "docs/SEMANTIC_LIFT_245M_CONFIRM_V1_PREREG.md"
PREREG_SHA_FILE = guard.ROOT / "docs/SEMANTIC_LIFT_245M_CONFIRM_V1_PREREG.md.sha256"
STOP_MARKER = ART / "STOP"
RUN_PREFIX = "semantic_lift_245m_confirm_v1_"
SLL_ART = guard.ROOT / "artifacts/semantic-lift-expert-learning-v1"  # sll v1 (terminal; read-only here)
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
M = 1_000_000
FINAL_BUDGET = 245 * M
N_STEPS = 3880
PLAN_SHA = "f4bc3c935a4710eb8ebdbc7d78df8270fc97b385f05839a96b06fd0f3458dc5f"
N_PARAMS = 103_609_728
# permanent checkpoints only at 50M and 245M (one rolling resumable slot in between, KEEP = 1)
MILESTONES = {"s50": 793, "p100": 3880}
SEEDS = (0, 1, 2, 3)
# frozen method (sll v1 lambda* = 0.5; nothing here may change)
LIFT = {"lam": 0.5, "lo": 0.5, "hi": 2.0, "rho": 0.98, "n0": 8.0, "floor": 0.02, "frac": 0.05}
# counterbalanced order within seeds (prereg section 4)
ORDER = (("E", 0), ("F", 0), ("F", 1), ("E", 1), ("E", 2), ("F", 2), ("F", 3), ("E", 3))
# reused 245M baselines (prereg section 3; verified by slc_reuse.py)
REUSED = {"T4_s0": "runs/20261114T131547Z_forecast_latent_edl_v1_final_T4_s0",
          "T4_s1": "runs/20261114T145059Z_forecast_latent_edl_v1_final_T4_s1",
          "T4_s2": "runs/20261114T221957Z_forecast_latent_edl_v1_final_T4_s2",
          "T4_s3": "runs/20261116T023759Z_semantic_edl_long_k_P1_T4_s3",
          "EDL_s0": "runs/20261114T173920Z_forecast_latent_edl_v1_final_EDL_s0",
          "EDL_s1": "runs/20261114T191747Z_forecast_latent_edl_v1_final_EDL_s1",
          "EDL_s2": "runs/20261116T004338Z_semantic_edl_long_k_P1_EDL_s2",
          "EDL_s3": "runs/20261116T031230Z_semantic_edl_long_k_P1_EDL_s3",
          "C6_s0": "runs/20261115T205836Z_semantic_edl_long_k_P1_C6_s0",
          "C6_s1": "runs/20261115T221300Z_semantic_edl_long_k_P1_C6_s1",
          "C6_s2": "runs/20261115T232721Z_semantic_edl_long_k_P1_C6_s2",
          "C6_s3": "runs/20261116T012134Z_semantic_edl_long_k_P1_C6_s3",
          "SHUF_s0": "runs/20261115T213547Z_semantic_edl_long_k_P1_SHUF_s0",
          "SHUF_s1": "runs/20261115T225011Z_semantic_edl_long_k_P1_SHUF_s1",
          "SHUF_s2": "runs/20261116T000530Z_semantic_edl_long_k_P1_SHUF_s2",
          "SHUF_s3": "runs/20261116T015952Z_semantic_edl_long_k_P1_SHUF_s3"}
GPU_CAP_H = {"total": 30.0, "checks": 0.5, "train": 8.0, "eval": 21.5}
DISK_RESERVE_GIB = 100.0


def prereg_sha():
    return guard.sha256_file(PREREG)


def check_prereg():
    want = PREREG_SHA_FILE.read_text().split()[0]
    h = prereg_sha()
    if h != want:
        raise guard.GuardError(f"preregistration hash mismatch: {h} != {want}")
    return h
