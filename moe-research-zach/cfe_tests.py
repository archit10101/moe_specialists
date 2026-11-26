#!/usr/bin/env python
"""cfe CPU tests: (1) CFEModel with default knobs == CPSModel bit-for-bit (forward hidden, aux, z, routes) in student
and teacher-free mode; (2) alpha floor: alpha in [alpha_min, alpha_max] and = alpha_min + (1-alpha_min)*sigmoid(b);
(3) configs for every stage build and carry the plan hash; gate references exist; (4) metrics helpers."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402


def main():
    guard.check_root()
    if not guard.require_execution("--execute" in sys.argv, "cpu-model"):
        return
    import numpy as np
    import torch
    from csr.cps_model import CPSConfig, CPSModel
    from csr.cfe_model import CFEConfig, CFEModel
    from csr.cfe_metrics import mi_domain, routing_change, paired_dlm
    import cfe_driver as drv
    torch.manual_seed(0)
    res = {}
    small = {"vocab_size": 512, "n_layer": 4, "d_model": 64, "n_head": 2, "dense_ffn": 128, "n_experts": 32,
             "top_k": 4, "expert_ffn": 32, "shared_ffn": 64, "rope_base": 10000.0, "norm_eps": 1e-6, "init_std": 0.02}
    B, T = 2, 96
    ids = torch.randint(2, 512, (B, T))
    doc = torch.zeros(B, T, dtype=torch.long)
    doc[:, 50:] = 1
    pos = torch.cat([torch.arange(50), torch.arange(46)]).unsqueeze(0).expand(B, -1).contiguous()
    batch = {"input_ids": ids, "doc_ids": doc, "positions": pos}
    kw = dict(teacher=False, chunk_len=32)
    a = CPSModel(CPSConfig(**kw), 3, small).eval()
    b = CFEModel(CFEConfig(**kw), 3, small).eval()
    b.load_state_dict(a.state_dict())
    with torch.no_grad():
        a.cp_ab.fill_(-1.0)
        b.cp_ab.fill_(-1.0)
        oa = a(batch, {"collect": True})
        ob = b(batch, {"collect": True})
    same_init = all(torch.equal(x, y) for x, y in zip(CPSModel(CPSConfig(**kw), 3, small).state_dict().values(),
                                                      CFEModel(CFEConfig(**kw), 3, small).state_dict().values()))
    res["T1_default_identical"] = bool(torch.equal(oa["hidden"], ob["hidden"]) and torch.equal(oa["aux"], ob["aux"])
                                       and torch.equal(oa["z"], ob["z"]) and same_init and
                                       all(torch.equal(r1["topi"], r2["topi"]) for r1, r2 in
                                           zip(oa["records"], ob["records"])))
    c = CFEModel(CFEConfig(alpha_min=0.2, **kw), 3, small)
    with torch.no_grad():
        cc = torch.randn(10, c.ccfg.ctx_dim)
        h = torch.randn(10, 64)
        al = c.alpha(0, cc, h)
        exp = 0.2 + 0.8 * torch.sigmoid(torch.tensor(-4.6))
    c.strip_teacher()
    res["T2b_strip_keeps_knobs"] = bool(c.ccfg.alpha_min == 0.2 and not c.has_teacher())
    res["T2_alpha_floor"] = bool(torch.allclose(al, exp.expand_as(al), atol=1e-6) and float(al.min()) >= 0.2)
    ok = True
    for stage in ("S1", "S2"):
        for seed in ((100,) if stage == "S1" else (100, 101)):
            cfg = drv.run_cfg(f"t_{stage}_{seed}", {**drv.BASE, "alpha_min": 0.2}, seed, stage)
            ok &= cfg["plan_sha256"] is not None and (guard.ROOT / cfg["gate"]["t4_metrics"]).exists()
            ok &= cfg["ccfg"]["alpha_min"] == 0.2 and cfg["train"] == drv.TRAIN
    cfgF = drv.run_cfg("t_F", drv.BASE, 0, "FINAL", gates=False, fracs=(0.4, 0.6, 0.8, 1.0), end_eval=())
    ok &= cfgF["gate"] is None and cfgF["end_eval"] == []
    res["T3_configs"] = bool(ok)
    topi = np.random.randint(0, 32, (1000, 3, 4)).astype(np.uint8)
    dom = np.random.randint(0, 5, 1000)
    ch, ov = routing_change(topi, topi)
    m_ind = mi_domain(topi, dom)
    topi2 = topi.copy()
    topi2[:, :, 0] = (dom * 6)[:, None]
    res["T4_metrics"] = bool(ch == 0.0 and ov == 4.0 and m_ind < 0.05 and mi_domain(topi2, dom) > m_ind and
                             paired_dlm({0: 2.0, 50: 3.0}, {0: 1.0, 50: 2.5}, 0, 100)[0] == 0.75)
    res["all_pass"] = all(res.values())
    guard.atomic_write_json(drv.ART / f"cpu_tests_{guard.utc_stamp()}.json", res)
    print(json.dumps(res, indent=1))
    sys.exit(0 if res["all_pass"] else 1)


if __name__ == "__main__":
    main()
