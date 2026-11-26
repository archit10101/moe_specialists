"""subject-conditioned-utility-router-v1 (scur1) paths and frozen constants
(docs/SUBJECT_CONDITIONED_UTILITY_ROUTER_V1_PREREG.md). Causal subject inference -> held-out subject-to-expert utility
mapping -> bounded token-level correction -> top-1 routing over the asrs2 K = 5 semantic residual specialists."""
from csr import guard

EXP = "subject-conditioned-utility-router-v1"
ART = guard.ROOT / "artifacts/subject-conditioned-utility-router-v1"
TMP = guard.ROOT / ".tmp/subject-conditioned-utility-router-v1"
CFG_DIR = guard.ROOT / "configs/subject-conditioned-utility-router-v1"
TABLES = ART / "tables"           # extraction tables (TRAIN / CS2) and the CS2 K=5 prior
ROUTERS = ART / "routers"
DEPS = ART / "deps"               # router deployments on evaluation partitions
ANALYSIS = ART / "analysis"
PREREG = guard.ROOT / "docs/SUBJECT_CONDITIONED_UTILITY_ROUTER_V1_PREREG.md"
PREREG_SHA_FILE = guard.ROOT / "docs/SUBJECT_CONDITIONED_UTILITY_ROUTER_V1_PREREG.md.sha256"
STOP_MARKER = ART / "STOP"
RUN_PREFIX = "scur1_"

A2 = "artifacts/csr-continuous-base-target-free-asrs-v2"
A2_RUN = "runs/20261122T053544Z_asrs2_S1"
MILESTONES = {   # asrs2 Stage-1 continuation milestones (seed 0): base + G, ALTG, SEMh, SHUFh, SEMs, SHUFs, FROZ banks
    "m510": ("runs/20261122T053544Z_asrs2_S1/ckpt_m510", "52c72015f5c0de320967502fdbfa4836a997d528fab7c360c42d88ad35f647d8"),
    "m702": ("runs/20261122T053544Z_asrs2_S1/ckpt_m702", "323098579bbd07636677ac7f577197ba15e1f11ad9bb345b3cb2708988ae7f43"),
    "m894": ("runs/20261122T053544Z_asrs2_S1/ckpt_m894", "2a4da943ffa3bdca3bdc8fca7ad64f731b384ddc67fca6bbcd7766f3cbc1f180"),
    "m1111": ("runs/20261122T053544Z_asrs2_S1/ckpt_m1111", "8e19210cc4a918e5248b93935584a232a0cb8416594f114e5ee6215dfd0c8306"),
}
CKPT_F = ("runs/20261122T051254Z_asrs2_S0F/ckpt_F", "02ba82ffb3122b9e591f63ae9b0205d184616461018340a81f21e89132341df6")
PRIMARY = "m1111"                # pre-declared engineering checkpoint: final milestone of the completed continuation
REUSED_FILES = {
    A2 + "/tables/k5_merge.json": "3344999a1b3872b789ae06377c93e3269e8c518bbdba162ff662e966ca681f8b",
    A2 + "/tables/p5_main.npz": "2b6e310213b9b35efe0b6013c06fd0bb28465f46a048020bb3355225d20a1686",
    A2 + "/tables/p5_calib_search.npz": "f1eb8460b4a353518294867a84afebe5626b09cd1ce949d85161a3018ccb47f3",
    A2 + "/evals/20261122T053544Z_asrs2_S1/m1111/summary_SEL.json": "50d90a21cc98e32a9772e6292a0b9cadb880342296a26673148509289c7cb999",
    A2 + "/evals/20261122T053544Z_asrs2_S1/m894/summary_SEL.json": "77f245e6fc909d81afa3a9e404b3993f8e63d09a8c66a95750df86fbdbbb5bd3",
    A2 + "/stage1_gen0.json": "d1b840fb850bfd7adf93e652f595f1c6c2637b22e09456f49362c06e66117283",
    A2 + "/stage0f.json": "ccd3f2fcc549c15f35857c8502bf0fd14727a8af831271daf6247465f6f7fa0a",
    "docs/CSR_CONTINUOUS_BASE_TARGET_FREE_ASRS_V2_PREREG.md": "afb54058a63ea865ec6ced368dfc765699b4a7fcc9379673e00de2219d4dfee9",
    "artifacts/oracle-residual-v1/tables/parts.npz": "3d7e2b305d06ea9a8527a61171c227ef13dddf5aca891908b79b4681856e6c4a",
    "artifacts/oracle-residual-v1/tables/token_feats.npz": "d7650a9640f0d626bc94823fae9ad61dd0017fb23702ec6a48e9412ff8f979b6",
    "artifacts/csr-current-context-specialist-router-v1/cs2_assign/parts_cs2.npz":
        "c0342a364f010197b66652bc6a7349d93f936f9c1aa6510eb003a054a0570303",
    "artifacts/csr-current-context-specialist-router-v1/tables_cs2/psem_calib_sel.npz": None,   # hash checked via its json
}
K = 5
SEM_BANK, SHUF_BANK, G_BANK, ALTG_BANK = "SEMh_0", "SHUFh_0", "G_0", "ALTG_0"
ROUTER_H = 64
TRAIN_STEPS = 2400               # ccsr1 router budget (identical for every router of a comparison)
DOCS_PER_STEP = 64
LR, WD, CLIP = 3e-3, 0.01, 1.0
PSEM_EPS = 1e-4
N_BOOT, BOOT_SEED = 2000, 20261501
ROUTER_SEEDS = (0, 1, 2)
GPU_CAP_H = {"stageA": 6.0, "stageB": 10.0, "total": 16.0}
DISK_RESERVE_GIB = 100.0
MAX_GENERATIONS = 4


def check_prereg():
    want = PREREG_SHA_FILE.read_text().split()[0]
    h = guard.sha256_file(PREREG)
    if h != want:
        raise guard.GuardError(f"prereg hash mismatch: {h} != {want}")
    return h


def verify_reused(log=None):
    import json
    man = {}
    for rel, want in REUSED_FILES.items():
        p = guard.ROOT / rel
        h = guard.sha256_file(p)
        if want is None:
            want = json.loads(p.with_suffix(".json").read_text())["sha256"]
        if h != want:
            raise guard.GuardError(f"reused artifact hash mismatch {rel}")
        man[rel] = h
    for tag, (path, sha) in {**MILESTONES, "F": CKPT_F}.items():
        meta = json.loads((guard.ROOT / path / "meta.json").read_text())
        if meta["sha256"] != sha or guard.sha256_file(guard.ROOT / path / "state.pt") != sha:
            raise guard.GuardError(f"milestone hash mismatch {path}")
        man[path] = sha
    if log:
        log(f"verified {len(man)} reused artifacts")
    return man
