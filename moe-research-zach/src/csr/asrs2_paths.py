"""csr-continuous-base-target-free-asrs-v2 (asrs2) paths and frozen constants
(docs/CSR_CONTINUOUS_BASE_TARGET_FREE_ASRS_V2_PREREG.md). Continuous ordinary base training + K = 5 additive residual
semantic specialists with target-free assignments and alternating specialist / router windows."""
from csr import guard

EXP = "csr-continuous-base-target-free-asrs-v2"
ART = guard.ROOT / "artifacts/csr-continuous-base-target-free-asrs-v2"
TMP = guard.ROOT / ".tmp/csr-continuous-base-target-free-asrs-v2"
CFG_DIR = guard.ROOT / "configs/csr-continuous-base-target-free-asrs-v2"
TABLES = ART / "tables"
EVALS = ART / "evals"
ANALYSIS = ART / "analysis"
PREREG = guard.ROOT / "docs/CSR_CONTINUOUS_BASE_TARGET_FREE_ASRS_V2_PREREG.md"
PREREG_SHA_FILE = guard.ROOT / "docs/CSR_CONTINUOUS_BASE_TARGET_FREE_ASRS_V2_PREREG.md.sha256"
STOP_MARKER = ART / "STOP"
RUN_PREFIX = "asrs2_"

# ---- reused, read-only, sha256-verified
CFAR1_RUN = "runs/20261121T151627Z_cfar1_S1_s100"
CKPT = CFAR1_RUN + "/ckpt_m20"            # TOKEN-4 replicate base at step 318 (its K=16 banks are NOT used)
CKPT_SHA = "80e1325cfa0dd787b9e400d83d7de7e1681c1cb014aa13438a7f587fee9d9f46"
CFAR1_CFG_SHA = "df19aee43a4d52a385751a97d30353d53e71f7970a6b67e2b938804e9345b6b0"
CCSR1_S3_RUN = "runs/20261121T194254Z_ccsr1_S3"
CCSR1_S3 = CCSR1_S3_RUN + "/ckpt_m1111"   # ordinary base continuation (reference A, cross-process)
CCSR1_S3_SHA = "6c0a6adf7a1f42a8fcca728272f430636a3930cf09f082c5e00065be0987629c"
REUSED_FILES = {
    "artifacts/csr-crossfit-semantic-utility-residual-v1/tables/psem_main.npz":
        "32885c9fe52a02dec4ce577f22dbf26cd7e70774da7e998821178633f7129775",
    "artifacts/csr-crossfit-semantic-utility-residual-v1/tables/psem_calib_search.npz":
        "c33a549d91bc891e00b3f0f7900f686a64587a84a89b0c785aa708489c28abc7",
    "artifacts/gc-edl-v1/latent/SEM_K16_s2/net.pt": "6a5e98673f44c6927ee4f4d52be27dee6684529f2c3425da2e53413f921bdd7a",
    "artifacts/oracle-residual-v1/tables/parts.npz": "3d7e2b305d06ea9a8527a61171c227ef13dddf5aca891908b79b4681856e6c4a",
    "artifacts/oracle-residual-v1/tables/token_feats.npz": "d7650a9640f0d626bc94823fae9ad61dd0017fb23702ec6a48e9412ff8f979b6",
    CFAR1_RUN + "/config.json": CFAR1_CFG_SHA,
}
PLAN_SHA = "f4bc3c935a4710eb8ebdbc7d78df8270fc97b385f05839a96b06fd0f3458dc5f"

# ---- architecture (user amendment: exactly K = 5)
K = 5
D_MODEL, FFN = 384, 256
ROUTER_H = 64
ALPHA_G = 0.01                  # Switch balance of the token-routed generic bank (cfar1 / ccsr1 value)

# ---- frozen stage (Stage 0F): base frozen at step 318, banks + routers trained on the rows the base saw (steps 0..317)
F_START, F_END = 0, 318
# ---- continuation: steps 318 .. 1110 of main_stream(245e6) (50,086,846 tokens), base updated on every step
START, END = 318, 1111
SPEC_W, ROUTER_W = 32, 16        # auxiliary windows; the base updates on all 48 steps of every window
REWARM = 20
MILESTONES = (510, 702, 894, 1111)
SPEC_SEEDS = (0, 1, 2, 3)        # independent specialist-training seeds (fresh bank + router init in Stage 0F)
ROUTER_OPT = {"lr": 3e-4, "wd": 0.01, "betas": (0.9, 0.99), "clip": 1.0, "lam": 0.01}
FROZEN_ROUTER_OPT = {"lr": 1e-3, "wd": 0.01, "betas": (0.9, 0.99), "clip": 1.0, "lam": 0.01}
PSEM_EPS = 1e-4
SLOT_EVERY_STEPS, SLOT_EVERY_S = 100, 600
SUB_FRAC = 0.25
GPU_CAP_H = {"stage0": 3.0, "stage1": 12.0, "stage2": 30.0, "stage3": 60.0, "total": 105.0}
DISK_RESERVE_GIB = 100.0
BOOT_SEED, N_BOOT = 20261401, 2000


def window_phase(step, start=START, spec_w=SPEC_W, router_w=ROUTER_W):
    """Auxiliary phase of a continuation step: ('spec' | 'router', window index, offset). Base updates regardless."""
    o = step - start
    w, r = divmod(o, spec_w + router_w)
    return ("spec" if r < spec_w else "router"), w, r


def prereg_sha():
    return guard.sha256_file(PREREG)


def check_prereg():
    want = PREREG_SHA_FILE.read_text().split()[0]
    h = prereg_sha()
    if h != want:
        raise guard.GuardError(f"prereg hash mismatch: {h} != {want}")
    return h


def verify_reused(log=None):
    man = {}
    for rel, want in {**REUSED_FILES, CKPT + "/state.pt": CKPT_SHA, CCSR1_S3 + "/state.pt": CCSR1_S3_SHA}.items():
        h = guard.sha256_file(guard.ROOT / rel)
        if h != want:
            raise guard.GuardError(f"reused artifact hash mismatch {rel}: {h} != {want}")
        man[rel] = h
    if log:
        log(f"verified {len(man)} reused artifacts")
    return man


def tables(split):
    """Frozen K=5 region-prior table of a split (hash-checked against k5_merge.json / p5_calib_sel.json)."""
    import json
    import numpy as np
    m = json.loads((TABLES / "k5_merge.json").read_text())
    if split in m["tables"]:
        rec = m["tables"][split]
    else:
        rec = json.loads((TABLES / f"p5_{split}.json").read_text())
        rec["path"] = str((TABLES / f"p5_{split}.npz").relative_to(guard.ROOT))
    p = guard.ROOT / rec["path"]
    if guard.sha256_file(p) != rec["sha256"]:
        raise guard.GuardError(f"p5 table hash mismatch {p}")
    return dict(np.load(p)), rec["sha256"]
