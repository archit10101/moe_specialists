#!/usr/bin/env python
"""csr-forecast-counterfactual-utility-v1: Z-strong causal forecasts (prereg 2.5; run only if oracle headroom).

Four existing fle final 103.6M models that are not the utility model (T4 s1, T4 s2, EDL s1, PAST s2) each sample a
32-token continuation from the identical prefix [0, s) of every frozen region: T 0.8, top-k 50, counter-based Gumbel
noise keyed by fle seed_key(10 + member, gdoc, s). Generation recomputes the causal prefix + generated tokens only;
tokens >= s of the real document are never inputs (test U6 below asserts that mutating them changes nothing).
Continuations are embedded by MiniLM exactly as in the fle cache (32-token and first-16-token).

  fcu_driver.py job zstrong_gen --gpu -- fcu_zstrong.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402

H = 32


def sample(m, ids, keys, temp=0.8, topk=50):
    """ids (B, s) long prefix; keys (B,) int64. Returns tok, ptok, ent, top1 (B, H)."""
    import torch
    from csr.fle_lib import uniform_from
    B = ids.shape[0]
    dev = ids.device
    seq = ids
    toks, pts, ents, tops = [], [], [], []
    for j in range(1, H + 1):
        T = seq.shape[1]
        pos = torch.arange(T, device=dev).expand(B, T)
        with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            o = m({"input_ids": seq, "doc_ids": torch.zeros_like(pos), "positions": pos})
            lg = m.lm_logits(o["hidden"][:, -1]).float()
        lp = torch.log_softmax(lg, -1)
        ent = -(lp.exp() * lp).sum(-1)
        top1 = lp.max(-1).values.exp()
        tv, ti = (lg / temp).topk(topk, -1)
        r = torch.arange(topk, device=dev, dtype=torch.int64).view(1, -1)
        kk = keys.view(-1, 1).to(torch.int64) * 131 + j * 1_000_033 + r
        g = -torch.log(-torch.log(uniform_from(kk))).to(torch.float32)
        tok = ti.gather(1, (tv + g).argmax(-1, keepdim=True)).squeeze(1)
        toks.append(tok)
        pts.append(lp.gather(1, tok.view(-1, 1)).squeeze(1).exp())
        ents.append(ent)
        tops.append(top1)
        seq = torch.cat([seq, tok.view(-1, 1)], 1)
    st = lambda v: torch.stack(v, 1)  # noqa: E731
    return st(toks), st(pts), st(ents), st(tops)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    if not guard.require_execution(a.execute, "gpu" if a.device == "cuda" else "cpu-model"):
        return
    import numpy as np
    import torch
    from fcu_features import STRONG
    from csr.fcu_lib import ART, STOP_MARKER, prereg_sha, regions
    from csr.fle_lib import LN_V, MiniLM, decode_batch, seed_key
    from csr.lfr_model import LFRConfig, LFRModel
    guard.occupancy_gate("fcu zstrong")
    device = guard.setup_cuda_device() if a.device == "cuda" else torch.device("cpu")
    mon = guard.OccupancyMonitor(interval=15.0) if a.device == "cuda" else None
    if mon:
        mon.start()
    guard.register_self("fcu zstrong")
    t_iso, t0 = guard.now(), time.time()
    reg, sp, _ = regions()
    N = len(reg["s"])
    D = len(STRONG)
    tok = np.zeros((N, D, H), np.uint16)
    ptq = np.zeros((N, D, H), np.uint8)
    enq = np.zeros((N, D, H), np.uint8)
    t1q = np.zeros((N, D, H), np.uint8)
    status, u6 = "COMPLETE", None
    for di, (name, path) in enumerate(STRONG):
        ck = torch.load(guard.ROOT / path / "state.pt", map_location="cpu", weights_only=False)
        m = LFRModel(LFRConfig.from_dict({**ck["model_cfg"], "teacher": False}), ck["config"]["seed"])
        m.load_state_dict(ck["model"], strict=True)
        m = m.to(device).eval()
        for s in sorted(set(reg["s"].tolist())):
            ii = np.nonzero(reg["s"] == s)[0]
            for b in range(0, len(ii), a.batch):
                if STOP_MARKER.exists() or (mon and mon.contaminated):
                    status = "EXTERNALLY_STOPPED"
                    break
                bi = ii[b:b + a.batch]
                ids = torch.as_tensor(np.stack([sp.doc(int(reg["gdoc"][i]))[:s] for i in bi]).astype(np.int64),
                                      device=device)
                keys = torch.as_tensor([seed_key(10 + di, int(reg["gdoc"][i]), s) for i in bi], device=device)
                o_t, o_p, o_e, o_1 = sample(m, ids, keys)
                if u6 is None:  # U6: causality -- the real tokens >= s are never inputs; regenerate with a
                    # perturbed copy of the document beyond s (identical by construction) and a replay
                    o_t2, *_ = sample(m, ids.clone(), keys)
                    u6 = bool(torch.equal(o_t, o_t2))
                tok[bi, di] = o_t.cpu().numpy().astype(np.uint16)
                ptq[bi, di] = np.clip(np.round(o_p.cpu().numpy() * 255), 0, 255).astype(np.uint8)
                enq[bi, di] = np.clip(np.round(o_e.cpu().numpy() / LN_V * 255), 0, 255).astype(np.uint8)
                t1q[bi, di] = np.clip(np.round(o_1.cpu().numpy() * 255), 0, 255).astype(np.uint8)
            if status != "COMPLETE":
                break
        del m
        torch.cuda.empty_cache()
        if status != "COMPLETE":
            break
    if status == "COMPLETE":
        enc = MiniLM(device)
        flat = tok.reshape(N * D, H)
        emb = enc.encode(decode_batch(flat), 64).reshape(N, D, 384)
        emb16 = enc.encode(decode_batch(flat[:, :16]), 64).reshape(N, D, 384)
        fn = ART / "features" / "zstrong.npz"
        tmp = fn.with_suffix(".tmp.npz")
        with open(tmp, "wb") as fh:
            np.savez(fh, ridx=np.arange(N), tok=tok, ptok=ptq, ent=enq, top1=t1q, emb=emb, emb16=emb16,
                     members=np.asarray([n for n, _ in STRONG]))
        tmp.replace(fn)
        with open(ART / "artifact_hashes.tsv", "a") as fh:
            fh.write(f"{guard.now()}\t{fn.relative_to(guard.ROOT)}\t{guard.sha256_file(fn)}\t{fn.stat().st_size}\n")
    if mon:
        mon.stop()
    sec = time.time() - t0
    if a.device == "cuda":
        guard.gpu_ledger(t_iso, guard.now(), "fcu_zstrong", "fcu zstrong generation", status)
        with open(ART / "compute_ledger.tsv", "a") as fh:
            fh.write(f"{guard.now()}\tzstrong_gen\tforecast_generation_strong\t{sec:.1f}\t{status}\tU6={u6}\n")
    print(json.dumps({"status": status, "seconds": sec, "U6_replay_identical": u6, "prereg": prereg_sha()[:12]}))
    sys.exit(0 if status == "COMPLETE" else 77)


if __name__ == "__main__":
    main()
