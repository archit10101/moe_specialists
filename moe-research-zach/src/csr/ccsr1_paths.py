"""csr-current-context-specialist-router-v1 (ccsr1) paths and frozen constants
(docs/CSR_CURRENT_CONTEXT_SPECIALIST_ROUTER_V1_PREREG.md). Router-only deployment experiment: frozen base + frozen
residual-specialist banks reused from cfar1 Stage 1 (seed 100, step 318); a strictly causal per-token GRU router
selects one specialist for every token from the current base hidden state and its causal context."""
from csr import guard

EXP = "csr-current-context-specialist-router-v1"
ART = guard.ROOT / "artifacts/csr-current-context-specialist-router-v1"
TMP = guard.ROOT / ".tmp/csr-current-context-specialist-router-v1"
CFG_DIR = guard.ROOT / "configs/csr-current-context-specialist-router-v1"
TABLES = ART / "tables"          # per-token specialist CE tables (extraction)
ROUTERS = ART / "routers"        # router checkpoints + per-token choices
PREREG = guard.ROOT / "docs/CSR_CURRENT_CONTEXT_SPECIALIST_ROUTER_V1_PREREG.md"
PREREG_SHA_FILE = guard.ROOT / "docs/CSR_CURRENT_CONTEXT_SPECIALIST_ROUTER_V1_PREREG.md.sha256"
STOP_MARKER = ART / "STOP"
RUN_PREFIX = "ccsr1_"

# ---- reused, read-only, hash-verified
CKPT = "runs/20261121T151627Z_cfar1_S1_s100/ckpt_m20"            # cfar1 Stage 1 joint checkpoint (base + 23 banks)
CKPT_SHA = "80e1325cfa0dd787b9e400d83d7de7e1681c1cb014aa13438a7f587fee9d9f46"
CFAR1_ART = guard.ROOT / "artifacts/csr-centered-fedavg-residual-v1"
CSUR1_TABLES = guard.ROOT / "artifacts/csr-crossfit-semantic-utility-residual-v1/tables"
PSEM_CS_SHA = "c33a549d91bc891e00b3f0f7900f686a64587a84a89b0c785aa708489c28abc7"
PSEM_CS_SH_SHA = "a73337d9a6f17436ef54971bf9b069510b4421921f7adb215af87532e37a154c"
ORV1 = guard.ROOT / "artifacts/oracle-residual-v1"
LABELS_CS_SHA = "2e546d756142b5b9d4a6d891abeb65df8bdaeff5571c20c8475686fa827992a9"
PARTS_SHA = "3d7e2b305d06ea9a8527a61171c227ef13dddf5aca891908b79b4681856e6c4a"
FEATS_SHA = "d7650a9640f0d626bc94823fae9ad61dd0017fb23702ec6a48e9412ff8f979b6"
SEM_NET = "artifacts/gc-edl-v1/latent/SEM_K16_s2/net.pt"
SEM_NET_SHA = "6a5e98673f44c6927ee4f4d52be27dee6684529f2c3425da2e53413f921bdd7a"
SEM_RES = "artifacts/gc-edl-v1/latent/residualizer_params.npz"
SEM_RES_SHA = "40c49358f186e0c39742cdd87566821f56622c5892f7ddf96935dcdf83b2cbae"

# banks inside CKPT. SEM = local semantic bank L (hard p_sem groups, no federation). SHUF = F2h_T500_w240: hard p_SH
# (paper-transplant) groups, never synchronised within 318 steps (T = 500), i.e. local training; identical rows and init
# to L except the 16 shared-only hidden units [240, 256) averaged across specialists at init. G = token-routed generic.
BANK_SEM, BANK_SHUF, BANK_G = "L", "F2h_T500_w240", "G"

# partitions (paper-disjoint). TRAIN = CALIB-SEARCH half "EVAL" (never evaluated in csur1/cfar1);
# SEL = CALIB-SEARCH half "SEL" (the existing selection half); CS2 = CALIB-SELECTION split (calib_sel), opened only
# after FROZEN_S2.json; VAL only after FROZEN_FINAL.json (only if a joint continuation passes); test never loaded.
PARTS = {"TRAIN": ("calib_search", "EVAL"), "SEL": ("calib_search", "SEL"), "CS2": ("calib_sel", None)}
SUB_FRAC = {"TRAIN": 0.0, "SEL": 0.25, "CS2": 0.25}   # swap subset for the functional metric (u_hash("csur1-sub", paper))

K = 16
D_MODEL = 384
WIN = 16                     # causal running-mean window (tokens, within document)
PSEM_EPS = 1e-4
N_PARAMS_BASE = 103_609_728
ROUTER_PARAM_CAP = 103_609   # < 0.1% of the base
ROUTER_SEEDS = (0, 1, 2)
TRAIN_STEPS = 2400           # optimizer steps per router (fixed; chosen by the TRAIN-only held-out-paper pilot)
DOCS_PER_STEP = 64
LR = 3e-3
WD = 0.01
CLIP = 1.0
W_CLIP = (0.25, 4.0)         # utility-margin token weight clip (R2/R3)
BOOT_SEED, N_BOOT = 20261301, 2000
GPU_CAP_H = {"total": 16.0}
DISK_RESERVE_GIB = 100.0


def prereg_sha():
    return guard.sha256_file(PREREG)


def check_prereg():
    want = PREREG_SHA_FILE.read_text().split()[0]
    h = prereg_sha()
    if h != want:
        raise guard.GuardError(f"prereg hash mismatch: {h} != {want}")
    return h
