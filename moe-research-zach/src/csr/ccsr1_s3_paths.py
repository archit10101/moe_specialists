"""ccsr1 Stage 3 (joint continuation) and section-11 efficiency study: paths and frozen constants
(docs/CSR_CURRENT_CONTEXT_SPECIALIST_ROUTER_V1_STAGE3_AMENDMENT.md)."""
from csr import guard
from csr.ccsr1_paths import ART, CKPT, CKPT_SHA, ROUTERS, ROUTER_SEEDS

AMEND = guard.ROOT / "docs/CSR_CURRENT_CONTEXT_SPECIALIST_ROUTER_V1_STAGE3_AMENDMENT.md"
AMEND_SHA_FILE = guard.ROOT / "docs/CSR_CURRENT_CONTEXT_SPECIALIST_ROUTER_V1_STAGE3_AMENDMENT.md.sha256"
S3 = ART / "stage3"
S3_TABLES = S3 / "tables"
S3_ROUTERS = ROUTERS / "S3"
EFF = ART / "efficiency"
CFAR1_RUN = "runs/20261121T151627Z_cfar1_S1_s100"
SELECTED = "R3_c2e0.5"
START, END_3A, END_PLAN = 318, 1111, 3880
MILESTONES = {"m1111": 1111, "m2111": 2111, "m3111": 3111, "m3880": 3880}
JOINT = {"ustride": 4, "lr": 3e-4, "wd": 0.01, "betas": [0.9, 0.99], "clip": 1.0, "c": 2.0, "eta": 0.5, "lam": 0.01}
REWARM = 20
SLOT_EVERY_STEPS, SLOT_EVERY_S = 100, 600
REFRESH = (1, 4, 8, 16)


def arms():
    out = [{"name": "G", "init": "G", "router": "token"}]
    for r in ROUTER_SEEDS:
        for name, init, bank, kind in ((f"SEMf_s{r}", "L", "SEM", "frozen"), (f"SEMj_s{r}", "L", "SEM", "joint"),
                                       (f"SHUFf_s{r}", "F2h_T500_w240", "SHUF", "frozen")):
            p = ROUTERS / "S1" / SELECTED / f"{bank}_s{r}.pt"
            out.append({"name": name, "init": init, "router": kind, "seed": r, "bank": bank,
                        "router_ckpt": str(p.relative_to(guard.ROOT)), "router_sha256": guard.sha256_file(p)})
    return out


def check_amendment():
    want = AMEND_SHA_FILE.read_text().split()[0]
    h = guard.sha256_file(AMEND)
    if h != want:
        raise guard.GuardError(f"Stage-3 amendment hash mismatch: {h} != {want}")
    return h


__all__ = ["AMEND", "S3", "S3_TABLES", "S3_ROUTERS", "EFF", "CKPT", "CKPT_SHA", "arms", "check_amendment"]
