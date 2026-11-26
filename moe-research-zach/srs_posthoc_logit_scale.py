# POST-HOC, EXPLORATORY (not preregistered; no decision depends on it): router-logit scale on CALIB-SEARCH rows.
import json, sys, torch
sys.path.insert(0, "src")
from csr.seqroute import build_sr_model
from csr.srs_run import eval_source, sr_evaluate
from csr.data import to_torch
torch.set_num_threads(4)
runs = [l.split("\t") for l in open(".tmp/sequence_routing_v1/final_runs.tsv").read().splitlines() if l.startswith("seed0/")]
out = {}
for key, rd, _ in runs:
    cfg = json.load(open(f"{rd}/config.json"))
    m = build_sr_model(cfg, torch.device("cpu")); m.load_state_dict(torch.load(f"{rd}/model.pt", map_location="cpu", weights_only=False)["model"]); m.eval()
    _, src = eval_source(cfg, "calib_search", cfg["data"]["dir"])
    tb = to_torch(src.batch(src.rows[:4]), torch.device("cpu"))
    with torch.no_grad():
        o = m(tb, collect=True)
    npm = tb["doc_ids"] >= 0
    res = []
    for li, r in enumerate(o.routing):
        h = r["h"][npm]
        W = m.moe_blocks()[li].ffn.router.weight
        zt = h @ W.T; zt = zt - zt.mean(-1, keepdim=True)
        row = {"token_logit_rms": float(zt.pow(2).mean().sqrt()), "h_norm": float(h.norm(dim=-1).mean()), "W_fro": float(W.norm())}
        if m.sr.mode == "sequence":
            zs = r["z_seg"] - r["z_seg"].mean(-1, keepdim=True)
            row["seq_logit_rms"] = float(zs.pow(2).mean().sqrt())
            row["top4_gate_sum_mean"] = float(r["seg_gates"].sum(-1).mean())
        else:
            row["top4_gate_sum_mean"] = float(r["gates"][npm].sum(-1).mean())
        res.append({k: round(v, 3) for k, v in row.items()})
    out[key] = res
    print(key, res[0], res[3], res[6])
json.dump(out, open(".tmp/sequence_routing_v1/posthoc_logit_scale.json", "w"), indent=1)
