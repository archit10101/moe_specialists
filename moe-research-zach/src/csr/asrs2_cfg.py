"""asrs2 run configurations: Stage-0F frozen-stage run, continuation runs, base-only controls and test variants."""
import json

from csr import guard
from csr import asrs2_paths as P

CONT_ARMS = ["G", "ALTG", "SEMh", "SHUFh", "SEMs", "SHUFs", "FROZ"]
FROZEN_ARMS = ["G5", "SEM5", "SHUF5"]


def _base():
    base = json.loads((guard.ROOT / P.CFAR1_RUN / "config.json").read_text())
    if guard.sha256_file(guard.ROOT / P.CFAR1_RUN / "config.json") != P.CFAR1_CFG_SHA:
        raise guard.GuardError("cfar1 config hash mismatch")
    return {k: base[k] for k in ("model", "train", "budget_tokens", "seed")}


def build(run_name, mode, seeds, arms=None, test=False, start=None, end=None, init_ckpt=None, init_sha=None,
          router_opt=None, milestones=None, spec_w=None, router_w=None, gen=0, change=None, slot_every_steps=None,
          spec_lr_mult=1.0):
    if mode == "frozen":
        arms = FROZEN_ARMS if arms is None else arms
        start, end = (P.F_START if start is None else start), (P.F_END if end is None else end)
        ro = dict(router_opt or P.FROZEN_ROUTER_OPT)
        ms = {"F": end} if milestones is None else milestones
    else:
        arms = CONT_ARMS if arms is None else arms
        start, end = (P.START if start is None else start), (P.END if end is None else end)
        ro = dict(router_opt or P.ROUTER_OPT)
        ms = {f"m{s}": s for s in P.MILESTONES if start < s <= end} if milestones is None else milestones
    ro["betas"] = list(ro["betas"])
    cfg = {"experiment": P.EXP, "run_name": run_name, "mode": mode, "test": bool(test), "generation": gen,
           "gen_change": change, "K": P.K, "seeds": list(seeds), "arms": list(arms),
           "base_ckpt": P.CKPT, "base_ckpt_sha256": P.CKPT_SHA, "base_config": _base(),
           "start_step": start, "end_step": end, "spec_w": spec_w or P.SPEC_W, "router_w": router_w or P.ROUTER_W,
           "router_opt": ro, "milestones": ms, "init_ckpt": init_ckpt, "init_ckpt_sha256": init_sha,
           "spec_lr_mult": spec_lr_mult, "rewarm": P.REWARM,
           "slot_every_steps": slot_every_steps or P.SLOT_EVERY_STEPS, "slot_every_s": P.SLOT_EVERY_S}
    if not test:
        cfg["prereg_sha256"] = P.check_prereg()
    return cfg


def write(cfg, name):
    P.CFG_DIR.mkdir(parents=True, exist_ok=True)
    p = P.CFG_DIR / f"{name}.json"
    if p.exists():
        if json.loads(p.read_text()) != cfg:
            raise guard.GuardError(f"config {p} exists with different content")
        return p
    guard.atomic_write_json(p, cfg, mode=0o400)
    return p
