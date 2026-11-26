"""joint-semantic-residual-pretrain-v1 (jsr) paths and frozen constants (docs/JOINT_SEMANTIC_RESIDUAL_PRETRAIN_V1_PROTOCOL.md).
TOKEN-4 base trained normally + K=16 residual-specialist banks (B no-semantic, C shuffled-group, D semantic, E oracle)
trained on the detached base prediction in the same process."""
from csr import guard

EXP = "joint-semantic-residual-pretrain-v1"
ART = guard.ROOT / "artifacts/joint-semantic-residual-pretrain-v1"
TMP = guard.ROOT / ".tmp/joint-semantic-residual-pretrain-v1"
CFG_DIR = guard.ROOT / "configs/joint-semantic-residual-pretrain-v1"
EVALS = ART / "evals"
TABLES = ART / "tables"
PREREG = guard.ROOT / "docs/JOINT_SEMANTIC_RESIDUAL_PRETRAIN_V1_PROTOCOL.md"
PREREG_SHA_FILE = guard.ROOT / "docs/JOINT_SEMANTIC_RESIDUAL_PRETRAIN_V1_PROTOCOL.md.sha256"
STOP_MARKER = ART / "STOP"
RUN_PREFIX = "joint_semantic_residual_pretrain_v1_"
ORV1 = guard.ROOT / "artifacts/oracle-residual-v1"  # frozen tables reused read-only

T4_CFG_RUN = "runs/20261114T131547Z_forecast_latent_edl_v1_final_T4_s0"
T4_PLAN_SHA = "f4bc3c935a4710eb8ebdbc7d78df8270fc97b385f05839a96b06fd0f3458dc5f"
N_STEPS = 3880
N_PARAMS_BASE = 103_609_728
C6_TABLE = "artifacts/semantic-edl-v1/tables/table_SEM_T05.npz"
C6_TABLE_SHA = "5d23129b2cb7aed6a0d2ed768b1a755eac661dae95ddeace89329dc7b56b89d2"
REUSED = {"T4": {0: "runs/20261114T131547Z_forecast_latent_edl_v1_final_T4_s0",
                 1: "runs/20261114T145059Z_forecast_latent_edl_v1_final_T4_s1",
                 2: "runs/20261114T221957Z_forecast_latent_edl_v1_final_T4_s2",
                 3: "runs/20261116T023759Z_semantic_edl_long_k_P1_T4_s3"},
          "EDL": {0: "runs/20261114T173920Z_forecast_latent_edl_v1_final_EDL_s0",
                  1: "runs/20261114T191747Z_forecast_latent_edl_v1_final_EDL_s1",
                  2: "runs/20261116T004338Z_semantic_edl_long_k_P1_EDL_s2",
                  3: "runs/20261116T031230Z_semantic_edl_long_k_P1_EDL_s3"},
          "C6": {0: "runs/20261115T205836Z_semantic_edl_long_k_P1_C6_s0",
                 1: "runs/20261115T221300Z_semantic_edl_long_k_P1_C6_s1",
                 2: "runs/20261115T232721Z_semantic_edl_long_k_P1_C6_s2",
                 3: "runs/20261116T012134Z_semantic_edl_long_k_P1_C6_s3"}}

K = 16
D_MODEL, EXPERT_FFN = 384, 256
INIT_STD = 0.02
INIT_SEED_OFFSET = 7_000_003  # experts: seed + offset; router: seed + offset + 1
ALPHA = 0.01
LAM_R = 1.0
EMA_RHO = 0.9
SD_MIN_TOKENS = 256
SD_EPS = 1e-6
R_GRID = (0.005, 0.02, 0.05)
TAU_GRID = (1.0, 0.5)
ALPHA_FALLBACK = 0.03
SEARCH_SEED = 100
STAGE2_SEEDS = (0, 1)
STAGE3_SEEDS = (0, 1, 2, 3)
BUDGETS = {"S1": 20_000_000, "S2": 50_000_000, "M100": 100_000_000, "M150": 150_000_000, "M200": 200_000_000}
BOOT_SEED, N_BOOT = 20261205, 2000
HIER_SEED, N_HIER = 20261206, 10000
GPU_CAP_H = {"total": 25.0, "tests": 1.0, "stage1": 1.5, "stage2": 2.0, "stage3": 14.0, "evals": 4.0, "eng": 2.5}
DISK_RESERVE_GIB = 100.0
DISK_BUDGET_GIB = 30.0


def prereg_sha():
    return guard.sha256_file(PREREG)


def check_prereg():
    want = PREREG_SHA_FILE.read_text().split()[0]
    h = prereg_sha()
    if h != want:
        raise guard.GuardError(f"protocol hash mismatch: {h} != {want}")
    return h
