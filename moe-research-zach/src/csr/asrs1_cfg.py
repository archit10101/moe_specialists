"""asrs1 run configurations (prereg section 4): the Stage-1 alternating run and small test variants."""
import json

from csr import guard
from csr import asrs1_paths as P


def arms():
    rs = lambda bank: [{"name": f"{bank}_s{r}", "ckpt": f"{P.ROUTER_DIR}/{bank}_s{r}.pt",
                        "sha256": P.ROUTER_SHA[f"{bank}_s{r}"]} for r in P.ROUTER_SEEDS]
    return [{"name": "ALTG", "init": P.BANK_G, "kind": "token"},
            {"name": "SEM", "init": P.BANK_SEM, "kind": "teacher", "prior": "sem", "routers": rs("SEM")},
            {"name": "SHUF", "init": P.BANK_SHUF, "kind": "teacher", "prior": "shuf", "routers": rs("SHUF")}]


def build(run_name, test=False, schedule=None, teacher=None, router_opt=None, slot_every_steps=None, gen=0, change=None):
    base = json.loads((guard.ROOT / P.CFAR1_RUN / "config.json").read_text())
    if guard.sha256_file(guard.ROOT / P.CFAR1_RUN / "config.json") != P.CFAR1_CFG_SHA:
        raise guard.GuardError("cfar1 config hash mismatch")
    sch = schedule or {"start": P.START, "end": P.END, "n_cycles": P.N_CYCLES, "base": P.BASE_STEPS, "spec": P.SPEC_STEPS}
    cfg = {"experiment": P.EXP, "run_name": run_name, "test": bool(test), "generation": gen, "gen_change": change,
           "base_ckpt": P.CKPT, "base_ckpt_sha256": P.CKPT_SHA,
           "base_config": {k: base[k] for k in ("model", "train", "budget_tokens", "seed")},
           "schedule": sch, "arms": arms(), "teacher": dict(teacher or P.TEACHER),
           "router_opt": dict(router_opt or P.ROUTER_OPT), "rewarm": P.REWARM,
           "slot_every_steps": slot_every_steps or P.SLOT_EVERY_STEPS, "slot_every_s": P.SLOT_EVERY_S}
    cfg["teacher"]["w_clip"] = list(cfg["teacher"]["w_clip"])
    cfg["router_opt"]["betas"] = list(cfg["router_opt"]["betas"])
    if not test:
        cfg["prereg_sha256"] = P.check_prereg()
    return cfg
