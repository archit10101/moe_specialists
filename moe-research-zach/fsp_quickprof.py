#!/usr/bin/env python
"""fspv2 engineering profiler of one training step (random tokens). Dry-run by default."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa

def main():
    guard.check_root()
    if not guard.require_execution("--execute" in sys.argv, "gpu"):
        return
    import torch
    guard.occupancy_gate("fsp quickprof")
    dev = guard.setup_cuda_device()
    guard.register_self("fsp quickprof")
    from csr.fsp_model import FSPModel, FSPConfig
    routing = sys.argv[1]
    m = FSPModel(FSPConfig(routing=routing), 0).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, fused=True)
    B = 32
    ids = torch.randint(2, 50000, (B, 1024), device=dev)
    doc = torch.zeros(B, 1024, dtype=torch.long, device=dev)
    pos = torch.arange(1024, device=dev).expand(B, -1).clone()
    lab = torch.randint(0, 5, (B, 1024), device=dev)
    batch = {"input_ids": ids, "doc_ids": doc, "positions": pos, "single_doc": True, "domain": lab}
    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = m(batch, {"use_dom": routing == "spec"})
            loss = torch.nn.functional.cross_entropy(o.logits.float().view(-1, o.logits.shape[-1]), ids.view(-1)) + 0.01 * o.aux + o.dom_ce
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    for _ in range(3): step()
    torch.cuda.synchronize()
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(2): step()
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=35))
    t0=time.time()
    for _ in range(4): step()
    torch.cuda.synchronize(); print("step s", (time.time()-t0)/4)

main()
