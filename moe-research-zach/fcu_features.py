#!/usr/bin/env python
"""csr-forecast-counterfactual-utility-v1: Phase-1 feature job (prereg 2.3, M7).

Per frozen region (order of fcu_lib.regions()):
  * O32 / O128 (ORACLE DIAGNOSTIC ONLY): MiniLM embedding of the actual tokens [s, s+32) / [s, s+128)
  * P2 (past-only control): MiniLM embedding of the last 64 prefix tokens [s-64, s)
  * forecast quality of a causal ensemble (--ensemble current = fle G0-G3, strong = 4 existing fle final models):
    teacher-forced CE of the actual next 32 tokens from the prefix (per member), ensemble-mixture log-likelihood,
    top-1 probability and correctness per token (calibration / ECE).

  fcu_driver.py job features_base --gpu -- fcu_features.py --what base
  fcu_driver.py job quality_strong --gpu -- fcu_features.py --what quality --ensemble strong
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from csr import guard  # noqa: E402

H = 32
STRONG = [("T4_s1", "runs/20261114T145059Z_forecast_latent_edl_v1_final_T4_s1/ckpt_p100"),
          ("T4_s2", "runs/20261114T221957Z_forecast_latent_edl_v1_final_T4_s2/ckpt_p100"),
          ("EDL_s1", "runs/20261114T191747Z_forecast_latent_edl_v1_final_EDL_s1/ckpt_p100"),
          ("PAST_s2", "runs/20261114T232324Z_forecast_latent_edl_v1_final_PAST_s2/ckpt_p100")]


def member_models(ensemble, device):
    """Yield (name, logits_fn) where logits_fn(ids (B,T) long) -> logits (B,T,V) for a causal LM."""
    import torch
    from csr.fle_lib import draft_list, load_draft
    from csr.fsp_lib import draft_hidden
    if ensemble == "current":
        for name, path, sha in draft_list():
            dm = load_draft(path, device, sha)

            def fn(ids, dm=dm):
                B, T = ids.shape
                pos = torch.arange(T, device=ids.device).expand(B, T)
                h = draft_hidden(dm, ids, torch.zeros_like(pos), pos)
                return h @ dm.tok_emb.weight.t()
            yield name, fn
            del dm
    else:
        from csr.lfr_model import LFRConfig, LFRModel
        for name, path in STRONG:
            ck = torch.load(guard.ROOT / path / "state.pt", map_location="cpu", weights_only=False)
            m = LFRModel(LFRConfig.from_dict({**ck["model_cfg"], "teacher": False}), ck["config"]["seed"])
            m.load_state_dict(ck["model"], strict=True)
            m = m.to(device).eval()

            def fn(ids, m=m):
                B, T = ids.shape
                pos = torch.arange(T, device=ids.device).expand(B, T)
                o = m({"input_ids": ids, "doc_ids": torch.zeros_like(pos), "positions": pos})
                return m.lm_logits(o["hidden"])
            yield name, fn
            del m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["base", "quality", "oracle_model"], default="base")
    ap.add_argument("--ensemble", choices=["current", "strong"], default="current")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    guard.check_root()
    if not guard.require_execution(a.execute, "gpu" if a.device == "cuda" else "cpu-model"):
        return
    import numpy as np
    import torch
    from csr.fcu_lib import ART, prereg_sha, regions
    from csr.fle_lib import MiniLM, decode_batch
    mon = None
    if a.device == "cuda":
        guard.occupancy_gate("fcu features")
        device = guard.setup_cuda_device()
        mon = guard.OccupancyMonitor(interval=15.0)
        mon.start()
    else:
        device = torch.device("cpu")
    guard.register_self(f"fcu features {a.what} {a.ensemble}")
    t_iso, t0 = guard.now(), time.time()
    reg, sp, _ = regions()
    N = len(reg["s"])
    docs = [sp.doc(int(d)) for d in reg["gdoc"]]
    out = {"ridx": np.arange(N)}
    guard.require_disk(200 * 2 ** 20, "fcu features")
    if a.what == "base":
        enc = MiniLM(device)
        s_ = reg["s"]
        out["o32"] = enc.encode(decode_batch([docs[i][s_[i]:s_[i] + 32] for i in range(N)]), 64)
        out["o128"] = enc.encode(decode_batch([docs[i][s_[i]:s_[i] + 128] for i in range(N)]), 256)
        out["p2"] = enc.encode(decode_batch([docs[i][s_[i] - 64:s_[i]] for i in range(N)]), 128, left=True)
        out["encoder_sha256"] = np.asarray(enc.sha)
        fn = ART / "features" / "base.npz"
    elif a.what == "oracle_model":
        # DESCRIPTIVE ROBUSTNESS ORACLES (never deployable): the utility model's own processing of the actual
        # future region -- mean final hidden state over [s, end) and region-mean router distribution at MoE 5..10.
        from csr.fcu_lib import LAYERS, build_rows, load_model
        torch.backends.cuda.matmul.allow_tf32 = False
        m = load_model("T4_s0", device)
        om = np.zeros((N, 384), np.float32)
        orr = np.zeros((N, len(LAYERS) * 32), np.float32)
        for s in sorted(set(reg["s"].tolist())):
            ii = np.nonzero(reg["s"] == s)[0]
            for b in range(0, len(ii), 32):
                bi = ii[b:b + 32]
                ids = torch.as_tensor(build_rows(sp, reg["gdoc"][bi], s), device=device)
                B, T = ids.shape
                pos = torch.arange(T, device=device).expand(B, T)
                with torch.no_grad():
                    o = m({"input_ids": ids, "doc_ids": torch.zeros_like(pos), "positions": pos}, {"collect": True})
                rm = ((torch.arange(T, device=device).view(1, -1) >= s) &
                      (torch.arange(T, device=device).view(1, -1) < torch.as_tensor(reg["end"][bi], device=device)
                       .view(-1, 1))).float()
                den = rm.sum(1, keepdim=True)
                om[bi] = ((o["hidden"].float() * rm.unsqueeze(-1)).sum(1) / den).cpu().numpy()
                orr[bi] = torch.cat([(o["records"][mi]["p"].float() * rm.unsqueeze(-1)).sum(1) / den
                                     for mi in LAYERS], 1).cpu().numpy()
        out.update({"o_model_hidden": om, "o_model_route": orr})
        fn = ART / "features" / "oracle_model.npz"
    else:
        names, ce, top1p, top1c, lp_all = [], [], [], [], []
        for name, f in member_models(a.ensemble, device):
            names.append(name)
            c_m = np.zeros((N, H), np.float32)
            p_m = np.zeros((N, H), np.float16)
            k_m = np.zeros((N, H), bool)
            for s in sorted(set(reg["s"].tolist())):
                ii = np.nonzero(reg["s"] == s)[0]
                for b in range(0, len(ii), 64):
                    bi = ii[b:b + 64]
                    ids = torch.as_tensor(np.stack([docs[i][:s + H] for i in bi]).astype(np.int64), device=device)
                    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                                         enabled=device.type == "cuda"):
                        lg = f(ids)
                    lg = lg[:, s - 1:s + H - 1].float()
                    lp = torch.log_softmax(lg, -1)
                    tgt = ids[:, s:s + H]
                    c_m[bi] = (-lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)).cpu().numpy()
                    mx = lp.max(-1)
                    p_m[bi] = mx.values.exp().cpu().numpy().astype(np.float16)
                    k_m[bi] = (mx.indices == tgt).cpu().numpy()
            ce.append(c_m)
            top1p.append(p_m)
            top1c.append(k_m)
            lp_all.append(-c_m)
        lpa = np.stack(lp_all, 1)  # (N, D, H) log p of the actual token per member
        mix = np.log(np.exp(lpa - lpa.max(1, keepdims=True)).mean(1)) + lpa.max(1)
        out.update({"members": np.asarray(names), "ce": np.stack(ce, 1), "top1p": np.stack(top1p, 1),
                    "top1c": np.stack(top1c, 1), "mix_ll": mix.astype(np.float32)})
        fn = ART / "features" / f"quality_{a.ensemble}.npz"
    status = "COMPLETE"
    if mon is not None:
        mon.stop()
        if mon.contaminated:
            status = "EXTERNALLY_STOPPED"
    if status == "COMPLETE":
        tmp = fn.with_suffix(".tmp.npz")
        with open(tmp, "wb") as fh:
            np.savez(fh, **out)
        tmp.replace(fn)
        sha = guard.sha256_file(fn)
        with open(ART / "artifact_hashes.tsv", "a") as fh:
            fh.write(f"{guard.now()}\t{fn.relative_to(guard.ROOT)}\t{sha}\t{fn.stat().st_size}\n")
    sec = time.time() - t0
    if a.device == "cuda":
        guard.gpu_ledger(t_iso, guard.now(), f"fcu_features_{a.what}_{a.ensemble}", "fcu features", status)
        kind = {"base": "oracle_past_features", "oracle_model": "robustness_oracle_features"}.get(
            a.what, f"forecast_quality_{a.ensemble}")
        with open(ART / "compute_ledger.tsv", "a") as fh:
            fh.write(f"{guard.now()}\tfeatures_{a.what}_{a.ensemble}\t{kind}\t{sec:.1f}\t{status}\t\n")
    print(json.dumps({"status": status, "seconds": sec, "prereg": prereg_sha()[:12]}))
    sys.exit(0 if status == "COMPLETE" else 77)


if __name__ == "__main__":
    main()
