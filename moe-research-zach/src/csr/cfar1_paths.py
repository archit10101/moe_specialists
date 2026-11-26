"""csr-centered-fedavg-residual-v1 (cfar1) paths and constants (follow-up to csur1; reuses csur1 region-prior tables read-only).
Based on csur1_paths:
(docs/CSR_CROSSFIT_SEMANTIC_UTILITY_RESIDUAL_V1_PREREG.md). TOKEN-4 base trained normally + K=16 residual-specialist
banks trained on the detached base prediction in the same process; block-level (128-token) cross-fitted utility
responsibilities with a frozen label-free region-level semantic prior; causal GRU block router (student)."""
from csr import guard

EXP = "csr-centered-fedavg-residual-v1"
ART = guard.ROOT / "artifacts/csr-centered-fedavg-residual-v1"
CSUR1_ART = guard.ROOT / "artifacts/csr-crossfit-semantic-utility-residual-v1"
TMP = guard.ROOT / ".tmp/csr-centered-fedavg-residual-v1"
CFG_DIR = guard.ROOT / "configs/csr-centered-fedavg-residual-v1"
EVALS = ART / "evals"
TABLES = CSUR1_ART / "tables"  # csur1 region priors, read-only (VAL priors: ART/tables_val)
VAL_TABLES = ART / "tables_val"
PREREG = guard.ROOT / "docs/CSR_CENTERED_FEDAVG_RESIDUAL_V1_PREREG.md"
PREREG_SHA_FILE = guard.ROOT / "docs/CSR_CENTERED_FEDAVG_RESIDUAL_V1_PREREG.md.sha256"
STOP_MARKER = ART / "STOP"
RUN_PREFIX = "cfar1_"
ORV1 = guard.ROOT / "artifacts/oracle-residual-v1"  # frozen tables reused read-only
JSR = guard.ROOT / "artifacts/joint-semantic-residual-pretrain-v1"  # reused read-only (tables.json budgets)

T4_CFG_RUN = "runs/20261114T131547Z_forecast_latent_edl_v1_final_T4_s0"
T4_PLAN_SHA = "f4bc3c935a4710eb8ebdbc7d78df8270fc97b385f05839a96b06fd0f3458dc5f"
N_STEPS = 3880
N_PARAMS_BASE = 103_609_728
FINAL_TOKENS = 245_032_457
SEM_NET = "artifacts/gc-edl-v1/latent/SEM_K16_s2/net.pt"
SEM_NET_SHA = "6a5e98673f44c6927ee4f4d52be27dee6684529f2c3425da2e53413f921bdd7a"
SEM_RES = "artifacts/gc-edl-v1/latent/residualizer_params.npz"
SEM_RES_SHA = "40c49358f186e0c39742cdd87566821f56622c5892f7ddf96935dcdf83b2cbae"
REUSED = {"T4": {0: "runs/20261114T131547Z_forecast_latent_edl_v1_final_T4_s0",
                 1: "runs/20261114T145059Z_forecast_latent_edl_v1_final_T4_s1",
                 2: "runs/20261114T221957Z_forecast_latent_edl_v1_final_T4_s2",
                 3: "runs/20261116T023759Z_semantic_edl_long_k_P1_T4_s3"},
          "EDL": {0: "runs/20261114T173920Z_forecast_latent_edl_v1_final_EDL_s0",
                  1: "runs/20261114T191747Z_forecast_latent_edl_v1_final_EDL_s1",
                  2: "runs/20261116T004338Z_semantic_edl_long_k_P1_EDL_s2",
                  3: "runs/20261116T031230Z_semantic_edl_long_k_P1_EDL_s3"}}

K = 16
D_MODEL, EXPERT_FFN = 384, 256
INIT_STD = 0.02
INIT_SEED_OFFSET = 7_000_003  # experts: seed + offset; router: seed + offset + 1 (same as jsr)
BLOCK = 128                   # routing block (view) length
REGION = 256                  # region = two disjoint views A (even block) and B (odd block)
MIN_PARTNER = 32              # a partner view needs >= this many scored tokens
GRU_DIN, GRU_H = 32, 32       # student router: Linear(384->32) -> GRUCell(32,32) -> Linear(32->16)
ROUTER_PARAM_CAP = 21_873     # 6,144 (token router of G) + 0.5% of 3,145,728 expert params
ALPHA_G = 0.01                # Switch balance of the generic token-routed bank G (orv1 C / jsr B)
PSEM_EPS = 1e-4
SCALE_RHO = 0.9               # EMA (diagnostic + eval) of the per-step utility spread and specialist means
USTRIDE = 4                   # utility measured on scored tokens with (pos + step) % USTRIDE == 0 (rotating subset)
USE_RHO = 0.99                # EMA of corpus-level specialist assignment mass m_k
SEARCH_SEED = 100
STAGE2_SEEDS = (100, 101)
FINAL_SEEDS = (100, 101, 102, 103)
BOOT_SEED, N_BOOT = 20261231, 2000
HIER_SEED, N_HIER = 20261232, 10000
GPU_CAP_H = {"total": 18.0}
T_GRID = (100, 250, 500)
WP_GRID = (240, 192, 128)
DISK_RESERVE_GIB = 100.0


def prereg_sha():
    return guard.sha256_file(PREREG)


def check_prereg():
    want = PREREG_SHA_FILE.read_text().split()[0]
    h = prereg_sha()
    if h != want:
        raise guard.GuardError(f"prereg hash mismatch: {h} != {want}")
    return h
