"""semantic-pair-bank-unrestricted-router-v1 (spb1) paths and frozen constants
(docs/SEMANTIC_PAIR_BANK_UNRESTRICTED_ROUTER_V1_PREREG.md).

Question: can label-free semantic groupings create functionally specialised residual experts (K = 16 frozen semantic
clusters x M = 2 specialists, chosen within the cluster by a causal pair subrouter during bank training) while a fresh,
completely unrestricted causal token router over all K*M specialists keeps the accuracy advantage of token routing?
Frozen TOKEN-4 base (orv1 bases), one residual bank after the final MoE layer, exactly one specialist per token."""
from csr import guard

EXP = "semantic-pair-bank-unrestricted-router-v1"
ART = guard.ROOT / "artifacts/semantic-pair-bank-unrestricted-router-v1"
TMP = guard.ROOT / ".tmp/semantic-pair-bank-unrestricted-router-v1"
RUNS = ART / "runs"            # bank-training runs (one per generation x bank seed x budget)
TABLES = ART / "tables"        # frozen labels (balanced persistent shuffle) + stats
EXTRACT = ART / "extract"      # per-token full specialist CE tables (TRAIN / SEL)
ROUTERS = ART / "routers"      # deployed-router checkpoints + SEL deployments
ANALYSIS = ART / "analysis"
CFG_DIR = ART / "configs"
PREREG = guard.ROOT / "docs/SEMANTIC_PAIR_BANK_UNRESTRICTED_ROUTER_V1_PREREG.md"
PREREG_SHA_FILE = guard.ROOT / "docs/SEMANTIC_PAIR_BANK_UNRESTRICTED_ROUTER_V1_PREREG.md.sha256"
REPORT = guard.ROOT / "docs/SEMANTIC_PAIR_BANK_UNRESTRICTED_ROUTER_V1_REPORT.md"
STOP_MARKER = ART / "STOP"

# ---- reused, read-only, hash-verified (never copied, never written)
ORV1 = guard.ROOT / "artifacts/oracle-residual-v1"
ORV1_TABLES = ORV1 / "tables"
REUSED_SHA = {
    "artifacts/oracle-residual-v1/tables/plan.npz": "8fb446ef677d9423b6dc70457008d244e7d1c3d0538bcec5c178661e2b2076a7",
    "artifacts/oracle-residual-v1/tables/parts.npz": "3d7e2b305d06ea9a8527a61171c227ef13dddf5aca891908b79b4681856e6c4a",
    "artifacts/oracle-residual-v1/tables/labels_main.npz": "ab70c965ebae66cbf0b89c41b779034296497027b565ef716e85382171f75760",
    "artifacts/oracle-residual-v1/tables/labels_cs.npz": "2e546d756142b5b9d4a6d891abeb65df8bdaeff5571c20c8475686fa827992a9",
    "artifacts/oracle-residual-v1/tables/token_feats.npz": "d7650a9640f0d626bc94823fae9ad61dd0017fb23702ec6a48e9412ff8f979b6",
    "artifacts/csr-crossfit-semantic-utility-residual-v1/tables/psem_calib_search.npz":
        "c33a549d91bc891e00b3f0f7900f686a64587a84a89b0c785aa708489c28abc7",
}
# COARSE-16: orv1 condition A (one specialist per semantic cluster, oracle paper label) at 50M on the SAME base seed,
# plan, budget, LR, init scheme and residual width (half the total bank parameters of the 32-specialist banks).
COARSE_CKPT = {0: ("artifacts/oracle-residual-v1/runs/s0/ckpt_50000000.pt",
                   "9cfbdb6190b429e8c29ecd3e76e2b7959130f5e4e1c08ef12b199d94f4947f0e"),
               1: ("artifacts/oracle-residual-v1/runs/s1/ckpt_50000000.pt",
                   "6206840356036de559308def63f58bd29311cfc57b63cc9827db2241278f5842"),
               2: ("artifacts/oracle-residual-v1/runs/s2/ckpt_50000000.pt",
                   "8871ea242cab67536025f978eb846cbda0ff920c88615d535dcb7088a28986ea")}
ORV1_VAL = ORV1 / "val_assign"   # frozen VAL semantic labels/rows (orv1); used only after FROZEN_FINAL.json

# ---- semantic clusters / shuffled control
K = 16
SHUF_SEED_MAIN = 20261401
SHUF_SEED_CS = 20261402
SHUF_SEED_VAL = 20261403

# ---- residual banks / bank training (identical for every condition; orv1 hyperparameters)
D_MODEL, EXPERT_FFN = 384, 256
INIT_STD = 0.02
ROWS_PER_STEP = 32
MICRO_ROWS = 16
WARMUP_STEPS = 50
BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.1
CLIP = 1.0
LR = 3e-4                   # orv1 pilot LR (reused; not re-tuned)
AUX = 0.01                  # orv1 Switch aux weight (reused)
INIT_SEED_OFFSET = 7_100_003
BUDGETS = {"S1": 50_000_000, "S3": 245_032_457}
SLOT_EVERY_S = 480          # rolling resume slot (<= 8 min of work lost)
BANK_SEEDS = (0, 1, 2)      # bank seed s uses TOKEN-4 base seed s (as orv1 / COARSE-16)
DEFAULT_GEN = {"M": 2, "assign": "hard", "sub_lr_mult": 1.0, "aux": AUX, "router_h": 64, "router_temp": 1.0,
               "router_bal": 0.01}

# ---- deployed router (ccsr1 current-context input; trained by the routed LM loss only)
WIN = 16
ROUTER_SEEDS = (0, 1)
ROUTER_STEPS = 2400
DOCS_PER_STEP = 64
ROUTER_LR = 3e-3
ROUTER_WD = 0.01
ROUTER_CLIP = 1.0

# ---- analysis / gates
BOOT_SEED, N_BOOT = 20261404, 2000
MIN_SET = 2000
GATE_ORACLE_MARGIN = 0.001      # Stage 1: SEM oracle not more than this worse than GEN own routing
GATE_CE = -0.001                # Stage 2: SEM - GEN <= -0.001 with CI upper < 0
GATE_LOSO = -0.0005             # leave-one-seed-out mean must stay <= this
GATE_EFF_MIN = 12.0             # effective experts (exp entropy) of the deployed SEM router
GATE_LOAD_MAX = 0.20
GPU_CAP_H = {"total": 24.0, "stage0": 1.0, "stage1": 6.0, "stage2": 3.0, "eng": 8.0, "stage3": 10.0}
DISK_RESERVE_GIB = 100.0
MAX_GENERATIONS = 3


def prereg_sha():
    return guard.sha256_file(PREREG)


def check_prereg():
    want = PREREG_SHA_FILE.read_text().split()[0]
    h = prereg_sha()
    if h != want:
        raise guard.GuardError(f"prereg hash mismatch: {h} != {want}")
    return h


def verify_reused():
    out = {}
    for p, h in REUSED_SHA.items():
        got = guard.sha256_file(guard.ROOT / p)
        if got != h:
            raise guard.GuardError(f"reused artifact hash mismatch {p}")
        out[p] = got
    for s, (p, h) in COARSE_CKPT.items():
        got = guard.sha256_file(guard.ROOT / p)
        if got != h:
            raise guard.GuardError(f"COARSE-16 checkpoint hash mismatch {p}")
        out[p] = got
    return out


def gen_cfg(gen):
    """Bank/router configuration of engineering generation `gen` (0 = preregistered). Generation changes are recorded
    in ART/gen<g>_change.json (one factor each) before the generation runs."""
    import json
    c = dict(DEFAULT_GEN)
    for g in range(1, gen + 1):
        ch = json.loads((ART / f"gen{g}_change.json").read_text())["change"]
        c.update(ch)
    return c


def bank_names(M):
    return (f"SEM{M}", f"SHUF{M}", f"GEN{K * M}")


BANK_KEYS = ("M", "assign", "sub_lr_mult", "aux")


def bank_gen(gen):
    """Earliest generation whose bank-training configuration equals generation `gen`'s (router-only generations reuse
    the frozen banks, checkpoints and extraction tables of that generation)."""
    c = gen_cfg(gen)
    for g in range(gen + 1):
        cg = gen_cfg(g)
        if all(cg[k] == c[k] for k in BANK_KEYS):
            return g
    return gen
