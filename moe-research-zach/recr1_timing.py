#!/usr/bin/env python
"""recr1 GPU latency micro-benchmark (Stage 0; TIMING ONLY). Uses the architecture-identical TOKEN-4 seed-3 ckpt_p100
(sha verified against its meta.json) because the spb1 bases are deleted; random residual weights of the spb1 shape.
No CE or routing result is computed. Writes ART/timing.json.

Prefill (16 rows x 1024 tokens, bf16 autocast): (a) base trunk + LM head; (b) S path = trunk + 1 residual + router +
LM head; (c) b + one extra LM head on the base hidden (exact evidence); (d) c + 2 shadow residuals + 2 LM heads (search
tokens). Decode components (LFRModel has no KV cache): per-token LM head, residual, router-GRU step at batch 1 and 64."""
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csr import guard  # noqa: E402
from csr import recr1_paths as P  # noqa: E402

CK = guard.ROOT / "runs/20261116T023759Z_semantic_edl_long_k_P1_T4_s3/ckpt_p100"


def tm(fn, reps=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2] * 1000


def main():
    if "--execute" not in sys.argv or os.environ.get("MOE_RESEARCH_APPROVE") != "gpu":
        raise SystemExit("dry run: requires --execute and MOE_RESEARCH_APPROVE=gpu")
    guard.check_root()
    from fle_evaluate import load_milestone
    meta = json.loads((CK / "meta.json").read_text())
    if guard.sha256_file(CK / "state.pt") != meta["sha256"]:
        raise guard.GuardError("seed-3 checkpoint hash mismatch")
    dev = torch.device("cuda")
    model, _ = load_milestone(str(CK.relative_to(guard.ROOT)), dev)
    model.eval()
    g = torch.Generator(device="cpu").manual_seed(0)
    B, T, d = 16, 1024, 384
    bt = {"input_ids": torch.randint(2, 50000, (B, T), generator=g).to(dev),
          "doc_ids": torch.zeros(B, T, dtype=torch.long, device=dev),
          "positions": torch.arange(T, device=dev).repeat(B, 1)}
    w1 = [torch.randn(d, 256, device=dev) * 0.02 for _ in range(3)]
    w2 = [torch.randn(256, d, device=dev) * 0.02 for _ in range(3)]
    gru = torch.nn.GRU(2 * d, 64, batch_first=True).to(dev)
    lin = torch.nn.Linear(64, 32).to(dev)
    cap = {}
    hk = model.norm_f.register_forward_pre_hook(lambda mod, inp: cap.__setitem__("x", inp[0]))
    rms = lambda x: x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(x.dtype)
    ac = lambda: torch.autocast("cuda", dtype=torch.bfloat16)

    def trunk():
        model({"input_ids": bt["input_ids"], "doc_ids": bt["doc_ids"], "positions": bt["positions"]})
        return cap["x"]

    @torch.no_grad()
    def run(kind):
        with ac():
            x = trunk()
            if kind == "a":
                return model.lm_logits(model.norm_f(x))
            feat = torch.cat([rms(x), rms(x)], -1)
            lg = lin(gru(feat)[0])
            h = x + torch.relu(rms(x) @ w1[0]) @ w2[0]
            out = model.lm_logits(model.norm_f(h))
            if kind in ("c", "d"):
                model.lm_logits(model.norm_f(x))
            if kind == "d":
                for j in (1, 2):
                    model.lm_logits(model.norm_f(x + torch.relu(rms(x) @ w1[j]) @ w2[j]))
            return out, lg

    res = {"time": guard.now(), "device": torch.cuda.get_device_name(0), "rows": B, "T": T,
           "note": "seed-3 architecture-identical weights; timing only", "prefill_ms": {}, "peak_mem_mb": {}}
    for k in "abcd":
        torch.cuda.reset_peak_memory_stats()
        res["prefill_ms"][k] = tm(lambda: run(k))
        res["peak_mem_mb"][k] = torch.cuda.max_memory_allocated() / 2**20
    hk.remove()
    a = res["prefill_ms"]
    res["prefill_overhead_vs_S"] = {"exact_evidence(c)": a["c"] / a["b"] - 1, "search_token(d)": a["d"] / a["b"] - 1,
                                    "S_vs_base(b)": a["b"] / a["a"] - 1}
    dec = {}
    W = model.tok_emb.weight if hasattr(model, "tok_emb") else None
    for bs in (1, 64):
        h = torch.randn(bs, d, device=dev, dtype=torch.bfloat16)
        hx = torch.zeros(1, bs, 64, device=dev)
        with torch.no_grad(), ac():
            dec[f"lm_head_b{bs}_ms"] = tm(lambda: model.lm_logits(model.norm_f(h)), 50)
            dec[f"residual_b{bs}_ms"] = tm(lambda: h + torch.relu(rms(h) @ w1[0]) @ w2[0], 50)
            dec[f"router_step_b{bs}_ms"] = tm(lambda: lin(gru(torch.cat([h, h], -1)[:, None].float(), hx)[0]), 50)
    res["decode_components"] = dec
    guard.atomic_write_json(P.ART / "timing.json", res, overwrite=True)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
