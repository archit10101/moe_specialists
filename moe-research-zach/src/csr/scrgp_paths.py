"""scrgp-v1 (Subject-Conditional Residual Gradient Projection) paths and constants (docs/SCRGP_V1_PREREG.md)."""
from csr import guard

EXP = "scrgp-v1"
ART = guard.ROOT / "artifacts/scrgp-v1"
TMP = guard.ROOT / ".tmp/scrgp-v1"
PREREG = guard.ROOT / "docs/SCRGP_V1_PREREG.md"
PREREG_SHA_FILE = guard.ROOT / "docs/SCRGP_V1_PREREG.md.sha256"
STOP_MARKER = ART / "STOP"
GC_LAT = guard.ROOT / "artifacts/gc-edl-v1/latent"
# frozen label-free CALIB-SEARCH semantic assignment (GC-EDL SEM K16 prototypes applied to CALIB-SEARCH papers)
SEM_CS = GC_LAT / "table_SEM_calib_search.npz"
SEM_CS_SHA = "abdd94c5112810df6705aa67a99be915d8223366a57d3b216ae4927978b5e69d"
TABLE_DIR = ART / "tables"
SHUF_SEED = 20261120
# frozen Semantic C6 (K16 SEM, T 0.5, all MoE layers, beta 0.005) 245M checkpoints (slk Phase 1), every existing seed
C6_RUNS = {0: "runs/20261115T205836Z_semantic_edl_long_k_P1_C6_s0",
           1: "runs/20261115T221300Z_semantic_edl_long_k_P1_C6_s1",
           2: "runs/20261115T232721Z_semantic_edl_long_k_P1_C6_s2",
           3: "runs/20261116T012134Z_semantic_edl_long_k_P1_C6_s3"}
C6_P100_SHA = {0: "32a77736fc46ab53642f390afe4e9271698114eea34cfda6500816d818a57ff4"}  # others verified in reuse
SEEDS = (0, 1, 2, 3)
GPU_CAP_H = {"total": 30.0, "stage0": 0.75, "stage1": 9.0}
DISK_RESERVE_GIB = 100.0


def prereg_sha():
    return guard.sha256_file(PREREG)


def check_prereg():
    want = PREREG_SHA_FILE.read_text().split()[0]
    h = prereg_sha()
    if h != want:
        raise guard.GuardError(f"preregistration hash mismatch: {h} != {want}")
    return h
