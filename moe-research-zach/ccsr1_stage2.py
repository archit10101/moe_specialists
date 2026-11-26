"""ccsr1 Stage 2 (prereg section 8): freeze the Stage-1 selection, prepare CALIB-SELECTION with the frozen pipelines,
extract its tables, evaluate the FROZEN Stage-1 routers (selected R3 configuration + R1 and R2 controls; 3 router seeds;
no retraining, no tuning) and apply the Stage-2 gates."""
import json
from pathlib import Path

from csr import guard
from csr import ccsr1_log as LG
from csr.ccsr1_paths import ART, CFG_DIR, ROUTER_SEEDS, ROUTERS, TABLES


def run(job, gen, sfx):
    s1 = json.loads((ART / f"stage1{sfx}.json").read_text())
    if not s1["stage1_pass"]:
        raise guard.GuardError("Stage 2 requires a passing Stage 1")
    if (ART / f"stage2{sfx}.json").exists():
        return json.loads((ART / f"stage2{sfx}.json").read_text())
    sel = s1["selected"]
    c = s1["selected_cfg"].get("c", 1.0)
    r2 = {0.5: "R2_c0.5", 1.0: "R2_c1", 2.0: "R2_c2"}[float(c)]
    evaluate = [sel, "R1", r2]
    fz_p = ART / "FROZEN_S2.json"
    if not fz_p.exists():
        fz = {"time": guard.now(), "generation": gen, "selected": sel, "selected_cfg": s1["selected_cfg"],
              "evaluate": evaluate, "router_seeds": list(ROUTER_SEEDS), "routers_from": f"S1{sfx}",
              "note": "all architecture, hyperparameters and routers frozen before CALIB-SELECTION is prepared or read"}
        guard.atomic_write_json(fz_p, fz, mode=0o400)
        LG.decision(evidence=json.dumps({"selected": sel, "mean_ce_stu_SEM": s1["configs"][sel]["mean_ce_stu_SEM"]}),
                    competing="none (freeze)", decision_=f"FREEZE Stage-2 candidate {sel} (+ controls R1, {r2}); routers from S1{sfx}",
                    rationale="prereg section 8: Stage 2 evaluates the frozen routers on held-out CALIB-SELECTION",
                    falsifier="Stage-2 gates G1-G6 on CALIB-SELECTION", stage="stage2")
    if not (ART / "cs2_prep_check.json").exists():
        job("cs2_prep_check", "stage2", ["ccsr1_cs2_prep.py", "--mode", "check"], disk_gb=1.0)
    if not (ART / "tables_cs2" / "psem_calib_sel_sh.json").exists():
        job("cs2_prep", "stage2", ["ccsr1_cs2_prep.py", "--mode", "cs2"], disk_gb=1.0)
    cs2 = TABLES / "CS2.npz"
    if not cs2.exists():
        job("extract_CS2", "stage2", ["ccsr1_extract.py", "--part", "CS2", "--out", str(cs2.relative_to(guard.ROOT))],
            disk_gb=4.0)
    s1cfg = {c_["tag"]: c_ for c_ in json.loads((CFG_DIR / f"S1{sfx}.json").read_text())}
    cp = CFG_DIR / f"S2{sfx}.json"
    if not cp.exists():
        guard.atomic_write_json(cp, [s1cfg[t] for t in evaluate], mode=0o400)
    tr = TABLES / "TRAIN.npz"
    for s in ROUTER_SEEDS:
        job(f"routers_S2{sfx}_s{s}", "stage2", ["ccsr1_router_job.py", "--eval-only", "--stage", f"S2{sfx}",
                                                "--router-stage", f"S1{sfx}", "--configs", str(cp.relative_to(guard.ROOT)),
                                                "--seed", str(s), "--train", str(tr.relative_to(guard.ROOT)),
                                                "--eval-part", "CS2", "--eval", str(cs2.relative_to(guard.ROOT))],
            disk_gb=1.0, seed=s)
    job(f"analyze_stage2{sfx}", "stage2", ["ccsr1_analyze.py", "stage2", "--gen", str(gen)], gpu=False, disk_gb=1.0)
    s2 = json.loads((ART / f"stage2{sfx}.json").read_text())
    LG.decision(evidence=json.dumps(s2["configs"][sel]["gates"]), competing="see OBSERVATIONS",
                decision_="Stage 2 PASS -> joint continuation requires the hashed Stage-3 implementation amendment" if s2["stage2_pass"]
                else "Stage 2 FAIL -> terminal (no tuning on CALIB-SELECTION; prereg section 8)",
                rationale="frozen Stage-2 gates", falsifier="n/a", stage="stage2")
    LG.set_state("STAGE2_PASS" if s2["stage2_pass"] else "TERMINAL_STAGE2_FAIL")
    return s2
