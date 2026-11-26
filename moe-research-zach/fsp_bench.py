#!/usr/bin/env python
"""fspv2 inference benchmarks of the final INFERENCE EXPORTS (teacher deleted, no draft, no labels): prefill latency
and peak memory at 1K/2K/4K/8K context (batch 1; RoPE positions beyond the 1,024 training length are used only for
timing), and autoregressive decode latency with a KV cache + incremental causal topic state (correctness-checked
against the full forward). Compares FORECAST-SPECIALIST (s0) with TOKEN-4 (s0) [and EDL / DOMAIN-AUX if present].
Writes artifacts/forecast-specialist-pretraining-v2/bench_<stamp>.json. Dry-run by default."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402


def main():
    guard.check_root()
    if not guard.require_execution("--execute" in sys.argv, "gpu"):
        return
    import torch
    import torch.nn.functional as F
    from csr.fsp_lib import ART
    from csr.fsp_model import FSPConfig, FSPModel, N_GENERAL, TEACHER_PREFIX
    from csr.model import rope_cos_sin
    guard.occupancy_gate("fsp bench")
    dev = guard.setup_cuda_device()
    guard.register_self("fsp bench")
    t_iso, t0 = guard.now(), time.time()
    st = json.loads((ART / "driver_state.json").read_text())
    res = {"note": "batch 1, bf16 autocast, eager, A100 80GB PCIe; drafts never loaded"}

    def load(run):
        rd = guard.ROOT / st["runs"][run]["run_dir"]
        cfg = json.loads((rd / "config.json").read_text())
        summ = json.loads((rd / "summary.json").read_text())
        m = FSPModel(FSPConfig.from_dict(cfg["model"]), cfg["seed"])
        m.load_state_dict(torch.load(guard.ROOT / summ["ckpts"]["p100"]["path"], map_location="cpu",
                                     weights_only=True)["model"])
        m.strip_teacher()
        assert not any(n.startswith(TEACHER_PREFIX) for n, _ in m.named_parameters())
        return m.to(dev).eval(), summ

    @torch.no_grad()
    def decode_step(m, tok, posn, state):
        cfg = m.cfg
        x = m.tok_emb(torch.tensor([[tok]], device=dev))
        cos, sin = rope_cos_sin(torch.tensor([[posn]], device=dev), cfg.d_model // cfg.n_head, cfg.rope_base, x.dtype)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        r = 0.5 ** (1.0 / cfg.topic_halflife)
        mi = 0
        for bi, blk in enumerate(m.blocks):
            x = x + blk.attn(blk.norm1(x), cos, sin, None, kv_cache=state["kv"][bi])
            h = blk.norm2(x)
            if not blk.is_moe:
                x = x + blk.ffn(h)
                continue
            moe = blk.ffn
            hf = h.reshape(1, -1).float()
            lg = F.linear(hf, moe.router.weight.float())
            p = torch.softmax(lg, -1)
            if cfg.routing == "spec":
                ts = state["topic"][mi]
                tp = F.linear(hf, moe.topic_proj.weight.float())
                ts["num"] = r * ts["num"] + tp
                ts["den"] = r * ts["den"] + 1.0
                u = ts["num"] / ts["den"]
                q = torch.log_softmax(F.linear(moe.topic_norm(u), moe.dom_topic.weight.float()) +
                                      F.linear(hf, moe.dom_tok.weight.float()), -1)
                g3, i3 = p[:, :N_GENERAL].topk(3, -1)
                si = (lg[:, N_GENERAL:] + cfg.spec_beta * q.repeat_interleave(2, 1)).argmax(-1, keepdim=True)
                topi = torch.cat([i3, si + N_GENERAL], -1)
                gates = torch.cat([g3, p[:, N_GENERAL:].gather(1, si)], -1)
            else:
                gates, topi = p.topk(cfg.top_k, -1)
            y = moe.shared(h.reshape(1, -1))
            for j in range(topi.shape[1]):
                e = int(topi[0, j])
                y = y + gates[0, j].to(y.dtype) * (F.gelu(h.reshape(1, -1) @ moe.experts.w1[e].to(h.dtype),
                                                          approximate="tanh") @ moe.experts.w2[e].to(h.dtype))
            x = x + y.view(1, 1, -1)
            mi += 1
        return m.lm_logits(m.norm_f(x))[0, 0]

    def new_state(m):
        return {"kv": [dict() for _ in m.blocks],
                "topic": [{"num": torch.zeros(1, m.cfg.topic_dim, device=dev), "den": 0.0} for _ in range(m.n_moe)]}

    runs = [r for r in ("final_FORECAST-SPECIALIST_s0", "final_TOKEN-4_s0", "final_EDL_s0", "final_DOMAIN-AUX_s0")
            if st["runs"].get(r, {}).get("status") == "COMPLETE"]
    for run in runs:
        m, summ = load(run)
        rr = {"train_tokens_per_s": summ["tokens_per_s"], "train_peak_alloc_gib": summ["peak_alloc_gib"],
              "train_peak_reserved_gib": summ["peak_reserved_gib"], "params": m.param_counts()}
        for T in (1024, 2048, 4096, 8192):
            ids = torch.randint(2, 50000, (1, T), device=dev)
            b = {"input_ids": ids, "doc_ids": torch.zeros_like(ids), "positions": torch.arange(T, device=dev)[None]}
            torch.cuda.reset_peak_memory_stats()
            ts = []
            for i in range(6):
                torch.cuda.synchronize()
                a = time.time()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    o = m(b, {"use_dom": m.cfg.routing == "spec"})
                    m.lm_logits(o["hidden"][:, -1:])
                torch.cuda.synchronize()
                ts.append(time.time() - a)
            rr[f"prefill_{T}"] = {"ms": 1000 * sorted(ts[1:])[2], "tok_per_s": T / sorted(ts[1:])[2],
                                  "peak_alloc_gib": torch.cuda.max_memory_allocated() / 2**30,
                                  "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
        # decode correctness + latency
        P = 64
        ids = torch.randint(2, 50000, (1, P), device=dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            full = m.lm_logits(m({"input_ids": ids, "doc_ids": torch.zeros_like(ids),
                                  "positions": torch.arange(P, device=dev)[None]},
                                 {"use_dom": m.cfg.routing == "spec"})["hidden"])[0].float()
            s = new_state(m)
            inc = torch.stack([decode_step(m, int(ids[0, i]), i, s) for i in range(P)]).float()
        rr["decode_vs_full_max_abs_logit_diff"] = float((full - inc).abs().max())
        rr["decode_vs_full_argmax_agreement"] = float((full.argmax(-1) == inc.argmax(-1)).float().mean())
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            s = new_state(m)
            tok = 2
            for i in range(512):
                decode_step(m, int(torch.randint(2, 50000, (1,))), i, s)
            torch.cuda.synchronize()
            a = time.time()
            for i in range(512, 640):
                lg = decode_step(m, tok, i, s)
                tok = int(lg.argmax())
            torch.cuda.synchronize()
        rr["decode_ms_per_token_ctx512to640"] = 1000 * (time.time() - a) / 128
        res[run] = rr
        print(run, json.dumps({k: v for k, v in rr.items() if k != "params"})[:800], flush=True)
        del m
        torch.cuda.empty_cache()
    p = ART / f"bench_{guard.utc_stamp()}.json"
    guard.atomic_write_json(p, res)
    t_end = guard.now()
    guard.gpu_ledger(t_iso, t_end, "fsp_bench", "fsp benchmarks", "COMPLETE")
    with open(ART / "compute_ledger.tsv", "a") as f:
        f.write(f"{t_iso}\t{t_end}\t{time.time() - t0:.1f}\tfsp_bench\tbench\tbenchmarks\tCOMPLETE\n")


if __name__ == "__main__":
    main()
