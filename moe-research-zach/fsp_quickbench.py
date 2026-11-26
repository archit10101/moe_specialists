#!/usr/bin/env python
"""fspv2 engineering throughput probe of the main backbone (random tokens; no data, no results). Dry-run by default."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa

def main():
    guard.check_root()
    if not guard.require_execution("--execute" in sys.argv, "gpu"):
        return
    import torch
    import torch.nn.functional as F
    guard.occupancy_gate("fsp quickbench")
    dev = guard.setup_cuda_device()
    guard.register_self("fsp quickbench")
    from csr.fsp_model import FSPModel, FSPConfig
    res = {}
    def ce_sum(hidden, w, tgt):
        return F.cross_entropy(F.linear(hidden, w).float(), tgt, ignore_index=-100, reduction="sum")
    ce_c = torch.compile(ce_sum)
    for routing in sys.argv[1].split(","):
        for comp in (False, True):
            cfg = FSPConfig(routing=routing)
            m = FSPModel(cfg, 0).to(dev)
            fwd = torch.compile(m) if comp else m
            ce = ce_c if comp else ce_sum
            opt = torch.optim.AdamW(m.parameters(), lr=1e-4, fused=True)
            B = 32
            torch.cuda.reset_peak_memory_stats()
            ids = torch.randint(2, 50000, (B, 1024), device=dev)
            doc = torch.zeros(B, 1024, dtype=torch.long, device=dev)
            pos = torch.arange(1024, device=dev).expand(B, -1).clone()
            lab = torch.randint(0, 5, (B, 1024), device=dev)
            batch = {"input_ids": ids, "doc_ids": doc, "positions": pos, "domain": lab}
            ctl = {"use_dom": routing == "spec", "single_doc": True}
            ts = []
            for it in range(12):
                torch.cuda.synchronize(); t0 = time.time()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    o = fwd(batch, ctl)
                    hid = o["hidden"].reshape(-1, 384)
                    loss = 0
                    for i in range(0, hid.shape[0], 8192):
                        loss = loss + ce(hid[i:i+8192], m.tok_emb.weight, ids.view(-1)[i:i+8192])
                    loss = loss / hid.shape[0] + 0.01 * o["aux"] + o["dom_ce"]
                loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize(); ts.append(time.time() - t0)
            dt = sum(ts[6:]) / len(ts[6:])
            k = f"{routing}_compile{comp}"
            res[k] = {"tok_s": B * 1024 / dt, "peak_alloc_gib": torch.cuda.max_memory_allocated() / 2**30, "first_step_s": ts[0]}
            print(k, res[k], flush=True)
            del m, opt, fwd
    print(json.dumps(res))

main()
