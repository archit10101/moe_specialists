"""oracle-residual-v1 (orv1) paths and frozen constants (docs/ORACLE_RESIDUAL_V1_PREREG.md).
Label-free semantic residual capacity test: frozen TOKEN-4 + one K=16 residual-specialist bank after the final MoE layer."""
from csr import guard

EXP = "oracle-residual-v1"
ART = guard.ROOT / "artifacts/oracle-residual-v1"
TMP = guard.ROOT / ".tmp/oracle-residual-v1"
RUNS = ART / "runs"
EVALS = ART / "evals"
TABLES = ART / "tables"
PREREG = guard.ROOT / "docs/ORACLE_RESIDUAL_V1_PREREG.md"
PREREG_SHA_FILE = guard.ROOT / "docs/ORACLE_RESIDUAL_V1_PREREG.md.sha256"
STOP_MARKER = ART / "STOP"
TOKENIZER = guard.ROOT / "artifacts/tokenizer/pythia-12b/tokenizer.json"

# frozen TOKEN-4 245M checkpoints (reused; read-only; never copied)
T4_RUNS = {0: "runs/20261114T131547Z_forecast_latent_edl_v1_final_T4_s0",
           1: "runs/20261114T145059Z_forecast_latent_edl_v1_final_T4_s1",
           2: "runs/20261114T221957Z_forecast_latent_edl_v1_final_T4_s2",
           3: "runs/20261116T023759Z_semantic_edl_long_k_P1_T4_s3"}
T4_PLAN_SHA = "f4bc3c935a4710eb8ebdbc7d78df8270fc97b385f05839a96b06fd0f3458dc5f"
T4_BUDGET = 245_000_000  # main_stream budget of the base plan (245,032,457 planned tokens)

# frozen label-free semantic assignment (GC-EDL SEM K16 prototypes; hard argmax)
SEM_MAIN = guard.ROOT / "artifacts/gc-edl-v1/latent/table_SEM.npz"
SEM_MAIN_SHA = "308f1e02f17e5e759fde45f1fc28e650acac41389974f1fcf14cb6bad7439da4"
SEM_CS = guard.ROOT / "artifacts/gc-edl-v1/latent/table_SEM_calib_search.npz"
SEM_CS_SHA = "abdd94c5112810df6705aa67a99be915d8223366a57d3b216ae4927978b5e69d"
K = 16
# shuffled oracle: permutation of hard labels across papers within paper-length bins (preserves papers per cluster
# exactly and token mass per cluster approximately)
SHUF_SEED_MAIN = 20261201
SHUF_SEED_CS = 20261202
SHUF_LEN_BINS = 20

# residual bank / training (identical for A, B, C)
D_MODEL, EXPERT_FFN = 384, 256
INIT_STD = 0.02
ROWS_PER_STEP = 32
MICRO_ROWS = 16
WARMUP_STEPS = 50
BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.1
CLIP = 1.0
LR_CANDIDATES = (3e-4, 1e-3, 3e-3)
AUX_LADDER = (0.01, 0.03, 0.1)
PILOT_TOKENS = 5_000_000
BUDGETS = {"S1": 20_000_000, "S2": 50_000_000, "S3": 245_032_457}
SLOT_EVERY_S = 480
INIT_SEED_OFFSET = 7_000_003  # specialist / router init generator seed = seed + offset (+1 for the router)
BOOT_SEED = 20261203
N_BOOT = 2000
STAGE1_SEEDS = (0, 1)
STAGE3_SEEDS = (0, 1, 2, 3)
NEAR_TIE = 0.001  # nats, A vs C
GPU_CAP_H = {"total": 20.0, "stage0": 1.5, "stage1": 2.0, "stage2": 3.0, "stage3": 13.0}
DISK_RESERVE_GIB = 100.0


def prereg_sha():
    return guard.sha256_file(PREREG)


def check_prereg():
    want = PREREG_SHA_FILE.read_text().split()[0]
    h = prereg_sha()
    if h != want:
        raise guard.GuardError(f"preregistration hash mismatch: {h} != {want}")
    return h
