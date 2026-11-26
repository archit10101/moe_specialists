"""csr-alternating-semantic-residual-specialization-v1 (asrs1) paths and frozen constants
(docs/CSR_ALTERNATING_SEMANTIC_RESIDUAL_SPECIALIZATION_V1_PREREG.md). Alternating continuation of the cfar1/ccsr1
semantic residual-specialist bank: base phase -> frozen teacher refresh -> specialist phase -> router phase."""
from csr import guard

EXP = "csr-alternating-semantic-residual-specialization-v1"
ART = guard.ROOT / "artifacts/csr-alternating-semantic-residual-specialization-v1"
TMP = guard.ROOT / ".tmp/csr-alternating-semantic-residual-specialization-v1"
CFG_DIR = guard.ROOT / "configs/csr-alternating-semantic-residual-specialization-v1"
CACHES = ART / "teacher_caches"
EVALS = ART / "evals"
PREREG = guard.ROOT / "docs/CSR_ALTERNATING_SEMANTIC_RESIDUAL_SPECIALIZATION_V1_PREREG.md"
PREREG_SHA_FILE = guard.ROOT / "docs/CSR_ALTERNATING_SEMANTIC_RESIDUAL_SPECIALIZATION_V1_PREREG.md.sha256"
STOP_MARKER = ART / "STOP"
RUN_PREFIX = "asrs1_"

# ---- reused, read-only, sha256-verified
CFAR1_RUN = "runs/20261121T151627Z_cfar1_S1_s100"
CKPT = CFAR1_RUN + "/ckpt_m20"
CKPT_SHA = "80e1325cfa0dd787b9e400d83d7de7e1681c1cb014aa13438a7f587fee9d9f46"
CFAR1_CFG_SHA = "df19aee43a4d52a385751a97d30353d53e71f7970a6b67e2b938804e9345b6b0"
CCSR1_ART = guard.ROOT / "artifacts/csr-current-context-specialist-router-v1"
ROUTER_DIR = "artifacts/csr-current-context-specialist-router-v1/routers/S1/R3_c2e0.5"
ROUTER_SHA = {
    "SEM_s0": "5b36768a866ce4e8a3e895e0afc60d00d5acd1929a841c70a324b4297fcaf98c",
    "SEM_s1": "4bb3d343b90c45551175909e4c54ed5e6858e9165ac7b9556ca59f864c7b574e",
    "SEM_s2": "da4163a550953c345fc1e0dc3f84a76fd1f93f47a8405a5a57cb27c4b7409cc8",
    "SHUF_s0": "2292ea190a036f7aebf239ed6a0de04edfba9fa047a69081043f022288d08e7a",
    "SHUF_s1": "c3bd90082159f9eff00b37c7edeeb1b11938d115b2469c32ee9c156540221d90",
    "SHUF_s2": "787e7b625d1be227f6738b94918bf7d18ca6d04705f7000b38d3e34cefb4c2ca",
}
CCSR1_S3 = "runs/20261121T194254Z_ccsr1_S3/ckpt_m1111"
CCSR1_S3_SHA = "6c0a6adf7a1f42a8fcca728272f430636a3930cf09f082c5e00065be0987629c"
REUSED_FILES = {
    "artifacts/csr-crossfit-semantic-utility-residual-v1/tables/psem_main.npz":
        "32885c9fe52a02dec4ce577f22dbf26cd7e70774da7e998821178633f7129775",
    "artifacts/csr-crossfit-semantic-utility-residual-v1/tables/psem_main_sh.npz":
        "070cd2b19c914591f100c8430117530ee7108b56a3edd59461cb4ab759c901b7",
    "artifacts/csr-crossfit-semantic-utility-residual-v1/tables/psem_calib_search.npz":
        "c33a549d91bc891e00b3f0f7900f686a64587a84a89b0c785aa708489c28abc7",
    "artifacts/csr-crossfit-semantic-utility-residual-v1/tables/psem_calib_search_sh.npz":
        "a73337d9a6f17436ef54971bf9b069510b4421921f7adb215af87532e37a154c",
    "artifacts/oracle-residual-v1/tables/parts.npz": "3d7e2b305d06ea9a8527a61171c227ef13dddf5aca891908b79b4681856e6c4a",
    "artifacts/oracle-residual-v1/tables/labels_cs.npz": "2e546d756142b5b9d4a6d891abeb65df8bdaeff5571c20c8475686fa827992a9",
    "artifacts/oracle-residual-v1/tables/token_feats.npz": "d7650a9640f0d626bc94823fae9ad61dd0017fb23702ec6a48e9412ff8f979b6",
    "docs/CSR_CURRENT_CONTEXT_SPECIALIST_ROUTER_V1_PREREG.md": "2101fa23d393973ec68f9b71f98d9d8580c48012387f7609863fa21b30b9f2b4",
    "docs/CSR_CURRENT_CONTEXT_SPECIALIST_ROUTER_V1_STAGE3_AMENDMENT.md":
        "b220e10157717613a489816b1071cfcc133f57eca7d34726b08e3c1e20d2f194",
    "artifacts/csr-current-context-specialist-router-v1/stage1.json":
        "0e9b6d3529cd5b7274fe25206e4d05be18d3b58b35613be655f7591889570c5e",
    "artifacts/csr-current-context-specialist-router-v1/stage2.json":
        "4cdfb98a5cdeab173a1b3601515f43f94571357feb918ee58d01585d7edc5a3d",
    "artifacts/csr-current-context-specialist-router-v1/stage3/stage3_m1111.json":
        "df0c654b179b5f696b9f21bfa1e75008bf67fc6d5bc93df94452a0d9b6c95e5b",
    CFAR1_RUN + "/config.json": CFAR1_CFG_SHA,
}
PLAN_SHA = "f4bc3c935a4710eb8ebdbc7d78df8270fc97b385f05839a96b06fd0f3458dc5f"
BANK_SEM, BANK_SHUF, BANK_G = "L", "F2h_T500_w240", "G"

# ---- schedule (prereg section 4)
START, END = 318, 1111          # steps 318 .. 1110 of main_stream(245e6): 50,086,846 scored tokens
N_CYCLES = 10
BASE_STEPS, SPEC_STEPS = 32, 32
REWARM = 20
ROUTER_SEEDS = (0, 1, 2)
TEACHER = {"c": 2.0, "eta": 0.5, "psem_eps": 1e-4, "w_clip": (0.25, 4.0)}
ROUTER_OPT = {"lr": 3e-4, "wd": 0.01, "betas": (0.9, 0.99), "clip": 1.0, "lam": 0.01}
SLOT_EVERY_STEPS, SLOT_EVERY_S = 100, 600
SUB_FRAC = 0.25
K = 16
GPU_CAP_H = {"stage1": 12.0, "stage2": 30.0, "stage3": 60.0, "total": 102.0}
DISK_RESERVE_GIB = 100.0
BOOT_SEED, N_BOOT = 20261301, 2000


def schedule(start=START, end=END, n_cycles=N_CYCLES, base=BASE_STEPS, spec=SPEC_STEPS):
    """[(cycle 1..n, [(phase, s0, s1), ...])] covering [start, end) exactly once."""
    T = end - start
    out = []
    for i in range(n_cycles):
        a = start + (T * i) // n_cycles
        b = start + (T * (i + 1)) // n_cycles
        L = b - a
        nb, ns = min(base, L), min(spec, max(0, L - base))
        nr = L - nb - ns
        if nr <= 0:
            raise ValueError("cycle too short for the phase lengths")
        out.append((i + 1, [("base", a, a + nb), ("spec", a + nb, a + nb + ns), ("router", a + nb + ns, b)]))
    return out


def phase_of(step, sched):
    for c, ph in sched:
        for name, s0, s1 in ph:
            if s0 <= step < s1:
                return c, name, s0, s1
    raise KeyError(step)


def prereg_sha():
    return guard.sha256_file(PREREG)


def check_prereg():
    want = PREREG_SHA_FILE.read_text().split()[0]
    h = prereg_sha()
    if h != want:
        raise guard.GuardError(f"prereg hash mismatch: {h} != {want}")
    return h


def verify_reused(log=None):
    """sha256 of every reused artifact; raises on any mismatch. Returns the manifest dict."""
    man = {}
    for rel, want in {**REUSED_FILES, CKPT + "/state.pt": CKPT_SHA, CCSR1_S3 + "/state.pt": CCSR1_S3_SHA,
                      **{f"{ROUTER_DIR}/{k}.pt": v for k, v in ROUTER_SHA.items()}}.items():
        h = guard.sha256_file(guard.ROOT / rel)
        if h != want:
            raise guard.GuardError(f"reused artifact hash mismatch {rel}: {h} != {want}")
        man[rel] = h
    if log:
        log(f"verified {len(man)} reused artifacts")
    return man
