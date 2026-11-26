#!/usr/bin/env python
"""csr-conditional-local-attention-v1 speed benchmarks (run after accuracy is frozen; seed-0 final checkpoints).

  cla_bench.py --runs FULL-TOKEN-4=<run dir> WINNER=<run dir> --device cuda --execute

Conditions: FULL-TOKEN-4 (flex doc-causal kernel, the trained-experiment implementation), FULL-TOKEN-4 original
(CSRModel, SDPA with a dense document mask; reference), FULL-TOKEN-4 sdpa_causal (flash causal kernel; prefill only,
single-document rows), the winner, and - if the winner is conditional - its gate forced always-local / always-global.
The baseline reference for every speed claim is the FASTEST baseline variant measured at that benchmark point.
Same GPU, precision (bf16 autocast, fp32 router), batch, sequence length, data, and the KV cache keeps every position.

Training (matched batch: 2 x 16 rows x 1024, the training batch): 100 warm-up + 500 timed optimizer steps on train-split
rows (seed-0 order), AdamW at the minimum lr, conditional winner with its deterministic 10% counterfactual steps;
tokens/s, steps/s, peak allocated / reserved VRAM. Maximum safe batch (separate): largest micro-batch in
{16,24,32,48,64,96,128} rows (accum 1) completing with peak reserved <= 0.9 x 56 GiB; 20 warm-up + 60 timed steps.
Inference (batch 1 and 8; prompts 128/256/512/1024 tokens = first tokens of CALIB-SEARCH documents): prefill latency
and tokens/s (10 warm-up + 100 timed); cached greedy decoding of 128 new tokens (3 warm-up + 100 timed generations;
deterministic argmax; no early stop): tokens/s and per-token latency; KV-cache bytes and peak VRAM.
Condition order is rotated at every benchmark point. Writes artifacts/conditional-local-attention-v1/bench_<stamp>.json.
"""
import argparse
import dataclasses
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402

BATCHES = (1, 8)
PROMPTS = (128, 256, 512, 1024)
N_NEW = 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--device", choices=["cuda"], default="cuda")
    ap.add_argument("--train-warm", type=int, default=100)
    ap.add_argument("--train-timed", type=int, default=500)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--quick", action="store_true", help="engineering smoke test (few reps)")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    runs = dict(x.split("=", 1) for x in a.runs)
    print(json.dumps(runs))
    if not guard.require_execution(a.execute, "gpu"):
        return
    if a.quick:
        a.train_warm, a.train_timed, a.reps = 5, 10, 3
    import numpy as np
    import torch
    import torch.nn.functional as F
    from csr.cla_decode import DecodeCache, prefill, step
    from csr.cla_model import CLAModel, cf_steps
    from csr.cla_run import ART, DATA_DIR, eval_source, load_raw_split
    from csr.data import to_torch, train_order
    from csr.model import CSRModel, ModelConfig, RoutingConfig
    from csr.train_lib import BatchSource
    guard.occupancy_gate("clav1 bench")
    dev = guard.setup_cuda_device()
    mon = guard.OccupancyMonitor(15.0)
    mon.start()
    guard.register_self("clav1 bench")
    t_iso = guard.now()
    res = {"settings": vars(a), "runs": runs}
    status = "FAILED"

    def load(label):
        rd = guard.inside_root(runs[label])
        cfg = json.loads((rd / "config.json").read_text())
        ck = torch.load(rd / "model.pt", map_location="cpu", weights_only=False)
        s = json.loads((rd / "summary.json").read_text())
        assert guard.sha256_file(rd / "model.pt") == s["checkpoint_sha256"]
        return cfg, ck["model"]

    def cla(cfg, sd, force="", backend="flex"):
        from csr.cla_model import AttnPlan
        plan = dataclasses.replace(AttnPlan.from_cfg(cfg["attention"]), force=force)
        m = CLAModel(ModelConfig.from_dict(cfg["model"]), plan, cfg["init"]["seed"], backend)
        m.load_state_dict(sd)
        return m.to(dev)

    try:
        bcfg, bsd = load("FULL-TOKEN-4")
        wl = [k for k in runs if k != "FULL-TOKEN-4"][0]
        wcfg, wsd = load(wl)
        models = {"FULL-TOKEN-4": lambda: cla(bcfg, bsd)}

        def orig():
            m = CSRModel(ModelConfig.from_dict(bcfg["model"]))
            m.load_state_dict({k: v for k, v in bsd.items() if not k.startswith(("gates.", "thr"))})
            return m.to(dev)
        models["FULL-TOKEN-4/original-sdpa-mask"] = orig
        models[wl] = lambda: cla(wcfg, wsd)
        cond = wcfg["attention"]["kind"] == "conditional"
        if cond:
            models[wl + "/forced-local"] = lambda: cla(wcfg, wsd, "local")
            models[wl + "/forced-global"] = lambda: cla(wcfg, wsd, "global")
        res["winner"] = wl
        res["winner_attention"] = wcfg["attention"]
        d = guard.ROOT / DATA_DIR
        tr = load_raw_split(d, "train")
        trs = BatchSource(tr, 1024, None, train_order(tr.n_docs, 17))
        sync = torch.cuda.synchronize

        # ---------------- training (matched batch)
        def train_bench(label, mk, mb_rows=16, accum=2, warm=a.train_warm, timed=a.train_timed):
            m = mk()
            m.train()
            is_orig = isinstance(m, CSRModel) and not isinstance(m, CLAModel)
            is_cond = (not is_orig) and m.plan.kind == "conditional" and not m.plan.force
            cfs = cf_steps(0, warm + timed, 0.10) if is_cond else frozenset()
            opt = torch.optim.AdamW(m.parameters(), lr=6e-5, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
            n_rows = len(trs.rows)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            ts, toks = [], []
            for it in range(warm + timed):
                sync()
                t0 = time.time()
                ntok = 0
                for j in range(accum):
                    r0 = ((it * accum + j) * mb_rows) % (n_rows - mb_rows)
                    tb = to_torch(trs.batch(trs.rows[r0:r0 + mb_rows]), dev)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        out = m(tb, RoutingConfig()) if is_orig else m(tb, cf=(it in cfs))
                    lg = out.logits.float()
                    nv = int((tb["targets"] != -100).sum())
                    loss = F.cross_entropy(lg.view(-1, lg.shape[-1]), tb["targets"].view(-1), ignore_index=-100) \
                        + 0.01 * out.aux_loss + 0.001 * out.z_loss
                    (loss / accum).backward()
                    ntok += nv
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                sync()
                if it >= warm:
                    ts.append(time.time() - t0)
                    toks.append(ntok)
            r = {"tokens_per_s": sum(toks) / sum(ts), "steps_per_s": len(ts) / sum(ts),
                 "median_step_ms": float(np.median(ts) * 1e3), "timed_steps": len(ts), "warmup_steps": warm,
                 "rows_per_step": mb_rows * accum, "peak_alloc_gib": torch.cuda.max_memory_allocated() / 2**30,
                 "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                 "cf_steps_in_timed": sum(1 for i in range(warm, warm + timed) if i in cfs)}
            del m, opt
            torch.cuda.empty_cache()
            return r
        order = list(models)
        res["train_matched"] = {}
        for lab in order:
            res["train_matched"][lab] = train_bench(lab, models[lab])
            print("train", lab, json.dumps(res["train_matched"][lab]), flush=True)
        # repeat the two primary conditions in reverse order (drift check)
        res["train_matched_repeat"] = {lab: train_bench(lab, models[lab]) for lab in [wl, "FULL-TOKEN-4"]}
        # ---------------- maximum safe batch
        res["train_max_batch"] = {}
        for lab in ["FULL-TOKEN-4", wl]:
            best = None
            for rows in (16, 24, 32, 48, 64, 96, 128):
                try:
                    r = train_bench(lab, models[lab], mb_rows=rows, accum=1, warm=2, timed=2)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    break
                if r["peak_reserved_gib"] > 0.9 * guard.VRAM_CEILING_GIB:
                    break
                best = rows
            r = train_bench(lab, models[lab], mb_rows=best, accum=1, warm=20 if not a.quick else 2,
                            timed=60 if not a.quick else 3)
            r["max_safe_micro_batch_rows"] = best
            res["train_max_batch"][lab] = r
            print("maxbatch", lab, json.dumps(r), flush=True)
        # ---------------- inference
        sp, cs = eval_source({"data": {"seq_len": 1024}}, "calib_search", d)
        long_docs = [i for row in cs.rows for i in row if len(sp.doc(i)) >= 1024]
        res["prompt_docs"] = {"source": "CALIB-SEARCH documents with >= 1024 tokens (first 8)",
                              "n_available": len(long_docs)}
        inf_models = dict(models)
        inf_models["FULL-TOKEN-4/sdpa-causal-flash"] = lambda: cla(bcfg, bsd, backend="sdpa_causal")
        labels = list(inf_models)
        built = {}

        def get(lab):
            if lab not in built:
                built[lab] = inf_models[lab]().eval()
            return built[lab]
        res["prefill"], res["decode"] = {}, {}
        pi = 0
        for B in BATCHES:
            for L in PROMPTS:
                prompt = torch.as_tensor(np.stack([sp.doc(i)[:L] for i in long_docs[:B]]), device=dev)
                key = f"B{B}_L{L}"
                res["prefill"][key], res["decode"][key] = {}, {}
                rot = labels[pi % len(labels):] + labels[:pi % len(labels)]
                pi += 1
                for lab in rot:
                    m = get(lab)
                    is_orig = isinstance(m, CSRModel) and not isinstance(m, CLAModel)
                    tb = {"input_ids": prompt, "doc_ids": torch.zeros_like(prompt),
                          "positions": torch.arange(L, device=dev).unsqueeze(0).expand(B, L).contiguous()}

                    def fwd():
                        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                            return m(tb, RoutingConfig()) if is_orig else m(tb)
                    for _ in range(10 if not a.quick else 2):
                        fwd()
                    sync()
                    torch.cuda.reset_peak_memory_stats()
                    lat = []
                    for _ in range(a.reps):
                        sync()
                        t0 = time.time()
                        o = fwd()
                        sync()
                        lat.append(time.time() - t0)
                    pr = {"median_latency_ms": float(np.median(lat) * 1e3), "mean_latency_ms": float(np.mean(lat) * 1e3),
                          "tokens_per_s": B * L / float(np.median(lat)), "reps": len(lat),
                          "peak_alloc_gib": torch.cuda.max_memory_allocated() / 2**30}
                    if not is_orig and o.gate:
                        pr["global_query_rate_by_layer"] = [float(g_["sel"].float().mean()) for g_ in o.gate]
                    res["prefill"][key][lab] = pr
                    if is_orig or lab.endswith("sdpa-causal-flash"):
                        continue  # decoding code path is identical for every baseline variant
                    # ---- cached greedy decoding of N_NEW tokens
                    def gen():
                        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                            logits, cache, _ = prefill(m, prompt, L + N_NEW, torch.bfloat16)
                            tok = logits.argmax(-1)
                            sync()
                            t0 = time.time()
                            nsel, ncnt = 0.0, 0
                            for _ in range(N_NEW):
                                logits, sels = step(m, tok, cache)
                                tok = logits.argmax(-1)
                                for s_ in sels:
                                    nsel += float(s_.float().sum())
                                    ncnt += s_.numel()
                            sync()
                            return time.time() - t0, cache.nbytes(), (nsel / ncnt if ncnt else None)
                    for _ in range(3 if not a.quick else 1):
                        gen()
                    torch.cuda.reset_peak_memory_stats()
                    times, kvb, rates = [], 0, []
                    for _ in range(a.reps):
                        t_, kvb, rr = gen()
                        times.append(t_)
                        rates.append(rr)
                    dt = float(np.median(times))
                    res["decode"][key][lab] = {
                        "tokens_per_s": B * N_NEW / dt, "per_token_latency_ms": dt / N_NEW * 1e3,
                        "median_generation_s": dt, "reps": len(times), "kv_cache_bytes": kvb,
                        "peak_alloc_gib": torch.cuda.max_memory_allocated() / 2**30,
                        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                        "decode_global_query_rate": (float(np.mean(rates)) if rates[0] is not None else None),
                        "max_position": L + N_NEW - 1}
                print("inference", key, json.dumps({k: (v.get("median_latency_ms"), res["decode"][key].get(k, {}).get(
                    "tokens_per_s")) for k, v in res["prefill"][key].items()}), flush=True)
        status = "COMPLETE" if not mon.contaminated else "CONTAMINATED"
    except Exception as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {e}"
        res["trace"] = traceback.format_exc()
        print(res["trace"])
    finally:
        mon.stop()
        res.update({"status": status, "time": guard.now(), "device": torch.cuda.get_device_name(0),
                    "torch": torch.__version__})
        out = ART / f"bench_{'quick_' if a.quick else ''}{guard.utc_stamp()}.json"
        guard.atomic_write_json(out, res)
        guard.gpu_ledger(t_iso, guard.now(), out.stem, "clav1 bench", status)
        print(f"BENCH_JSON {out.relative_to(guard.ROOT)} {status}")
    sys.exit(0 if status == "COMPLETE" else 1)


if __name__ == "__main__":
    main()
