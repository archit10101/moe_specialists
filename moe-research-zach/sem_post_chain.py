import sys
sys.path.insert(0, "src"); sys.path.insert(0, ".")
from sem_driver import run_job
import sem_search as S
pe = [f"{r}_s{s}" for s in (100, 101) for r in ("s_C4", "shuf_C4", "b_EDL", "b_T4")]
jobs = [(rn, "s50", True) for rn in pe]
jobs += [(f"s_C7_s{s}", "s50", False) for s in (100, 101)]
jobs += [(f"s_C{i}_s100", "s20", False) for i in range(1, 9)] + [("b_T4_s100", "s20", False), ("b_EDL_s100", "s20", False)]
for rn, tag, pex in jobs:
    if not S.budget_ok(0.4 if pex else 0.1, f"post {rn}"):
        S.log_event("post: budget stop"); break
    run_job(f"post_{rn}_{tag}{'_pe' if pex else ''}", ["sem_post.py", "audit", "--run-name", rn, "--tag", tag]
            + (["--per-expert"] if pex else []), gpu=True, disk_gb=0.5)
print("POST_CHAIN_DONE")
