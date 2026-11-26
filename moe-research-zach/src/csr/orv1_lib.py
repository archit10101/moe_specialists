"""oracle-residual-v1 core: frozen TOKEN-4 base, residual-specialist banks (A SEMANTIC-ORACLE, B SHUFFLED-ORACLE,
C TOKEN-ROUTED-RESIDUAL), joint training step, exact evaluation.

Insertion point: the residual stream x_t after the final MoE block (block 11) and before the final RMSNorm + tied LM head:
    h'_t = x_t + a_{e(t)}(n(x_t)),   logits_t = W_emb norm_f(h'_t)
n(.) is a parameter-free RMS normalisation (the bank's own input norm; no existing parameter is reused or trained). Each
specialist has the routed-expert structure (W1 384x256, GELU-tanh, W2 256x384; csr.lfr_model.Experts dispatch), W1 drawn
from a seed-specific generator identical across conditions, W2 = 0 (so every condition starts exactly at the base).
Exactly one specialist per token: A = hard semantic label of the token's paper, B = length-stratified shuffled label,
C = argmax of a learned causal linear router on n(x_t) (straight-through gate p/sg(p) == 1 in value; Switch load-balance
aux loss). The frozen base forward is computed once per micro-batch and shared by all three conditions."""
import hashlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from csr import guard
from csr.lfr_model import Experts
from csr.orv1_paths import (D_MODEL, EXPERT_FFN, INIT_SEED_OFFSET, INIT_STD, K, ROWS_PER_STEP, T4_RUNS, TABLES)

T_SEQ = 1024
PAD_ID = 1
CONDS = ("A", "B", "C")
EPS = 1e-6


# ------------------------------------------------------------------------------------------------------ data
def make_batch(rows, split):
    """Label-free batch: input_ids, targets, doc_ids, positions, gdoc (B, T)."""
    B = len(rows)
    a = {k: np.full((B, T_SEQ), f, dtype=np.int64) for k, f in
         (("input_ids", PAD_ID), ("targets", -100), ("doc_ids", -1), ("positions", 0), ("gdoc", -1))}
    for b, row in enumerate(rows):
        t = 0
        for local, di in enumerate(row):
            ids = split.doc(di)
            L = len(ids)
            a["input_ids"][b, t:t + L] = ids
            a["targets"][b, t:t + L - 1] = ids[1:]
            a["doc_ids"][b, t:t + L] = local
            a["positions"][b, t:t + L] = np.arange(L)
            a["gdoc"][b, t:t + L] = di
            t += L
    return a


def to_dev(a, device):
    return {k: torch.as_tensor(v, device=device) for k, v in a.items()}


def unflatten_rows(docs, row_len):
    out, i = [], 0
    for n in row_len:
        out.append([int(d) for d in docs[i:i + n]])
        i += n
    return out


def load_plan():
    """Residual-training plan: list of steps (each ROWS_PER_STEP rows of main doc ids), plan sha."""
    z = np.load(TABLES / "plan.npz")
    rows = unflatten_rows(z["docs"], z["row_len"])
    rps = int(z["rows_per_step"])
    assert rps == ROWS_PER_STEP and len(rows) % rps == 0
    return [rows[i:i + rps] for i in range(0, len(rows), rps)], str(z["plan_sha"])


def load_part(name):
    z = np.load(TABLES / "parts.npz")
    return unflatten_rows(z[f"{name}_docs"], z[f"{name}_row_len"])


def load_labels(which):
    z = np.load(TABLES / f"labels_{which}.npz")
    return z["sem"].astype(np.int64), z["shuf"].astype(np.int64)


def steps_for(budget):
    """Number of plan steps for a token budget (smallest prefix whose valid tokens reach it)."""
    from csr.fsp_lib import Split
    steps, _ = load_plan()
    L = Split("main").lengths
    cum = np.cumsum([sum(int(L[d]) - 1 for r in st for d in r) for st in steps])
    if budget > cum[-1]:
        raise ValueError(f"budget {budget} exceeds plan {cum[-1]}")
    return int(np.searchsorted(cum, budget) + 1)


# ------------------------------------------------------------------------------------------------------ base
def load_base(seed, device):
    """Frozen TOKEN-4 p100 (sha verified against status.json); every parameter requires_grad=False."""
    import json
    from fle_evaluate import load_milestone
    rd = guard.ROOT / T4_RUNS[seed]
    ck = json.loads((rd / "status.json").read_text())["ckpts"]["p100"]
    h = guard.sha256_file(guard.ROOT / ck["path"] / "state.pt")
    if h != ck["sha256"]:
        raise guard.GuardError(f"T4 s{seed} checkpoint hash mismatch")
    m, _ = load_milestone(ck["path"], device)
    for p in m.parameters():
        p.requires_grad_(False)
    return m.eval(), h


def state_hash(model):
    h = hashlib.sha256()
    for n, p in sorted(model.state_dict().items()):
        h.update(n.encode())
        h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def autocast(dev):
    return torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")


@torch.no_grad()
def base_forward(model, bt):
    """Frozen base: returns x (N, d) fp32 residual stream entering norm_f, and the base hidden norm_f(x) (N, d)."""
    cap = {}
    hk = model.norm_f.register_forward_pre_hook(lambda mod, inp: cap.__setitem__("x", inp[0]))
    try:
        with autocast(bt["input_ids"].device):
            out = model({"input_ids": bt["input_ids"], "doc_ids": bt["doc_ids"], "positions": bt["positions"]})
    finally:
        hk.remove()
    x = cap["x"]
    return x.reshape(-1, x.shape[-1]), out["hidden"].reshape(-1, x.shape[-1])


def rms0(x):
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)


def head_ce(model, h, tgt, chunk=8192, from_hidden=False):
    """Per-token CE (fp32) of norm_f(h) (or of an already normalised hidden if from_hidden) under bf16 autocast."""
    dev = h.device
    out = []
    for s in range(0, h.shape[0], chunk):
        with autocast(dev):
            hid = h[s:s + chunk] if from_hidden else model.norm_f(h[s:s + chunk])
            lg = model.lm_logits(hid)
        out.append(F.cross_entropy(lg.float(), tgt[s:s + chunk], ignore_index=-100, reduction="none"))
    return torch.cat(out)


# ------------------------------------------------------------------------------------------------------ bank
class Cond(nn.Module):
    """One residual-specialist bank (K specialists, routed-expert structure) + optional token router (condition C)."""

    def __init__(self, name, seed, router=False):
        super().__init__()
        self.name = name
        self.bank = Experts(K, D_MODEL, EXPERT_FFN)
        g = torch.Generator().manual_seed(seed + INIT_SEED_OFFSET)
        with torch.no_grad():
            self.bank.w1.copy_(torch.randn(self.bank.w1.shape, generator=g) * INIT_STD)
            self.bank.w2.zero_()
        self.router = None
        if router:
            self.router = nn.Linear(D_MODEL, K, bias=False)
            gr = torch.Generator().manual_seed(seed + INIT_SEED_OFFSET + 1)
            with torch.no_grad():
                self.router.weight.copy_(torch.randn(self.router.weight.shape, generator=gr) * INIT_STD)

    def n_params(self):
        return {"bank": self.bank.w1.numel() + self.bank.w2.numel(),
                "router": 0 if self.router is None else self.router.weight.numel()}

    def route(self, xn, valid):
        """C: top-1 index, straight-through gate (== 1 in value), Switch aux loss over valid tokens, probs."""
        with torch.autocast(device_type=xn.device.type, enabled=False):
            p = torch.softmax(F.linear(xn.float(), self.router.weight.float()), -1)
        idx = p.argmax(-1)
        pe = p.gather(1, idx.unsqueeze(1)).squeeze(1)
        gate = pe / pe.detach()
        vf = valid.float()
        nv = vf.sum().clamp(min=1)
        f = (F.one_hot(idx, K).float() * vf.unsqueeze(1)).sum(0) / nv
        P = (p * vf.unsqueeze(1)).sum(0) / nv
        aux = K * (f * P).sum()
        return idx, gate, aux, p

    def forward(self, xn, assign, valid):
        """xn (N, d) normalised base residual; assign (N,) label for A/B (ignored for C). Returns delta (N, d) fp32,
        idx (N,), aux (0-d)."""
        aux = torch.zeros((), device=xn.device)
        if self.router is not None:
            idx, gate, aux, _ = self.route(xn, valid)
        else:
            idx = assign
            gate = torch.ones(xn.shape[0], device=xn.device)
        with autocast(xn.device):
            delta = self.bank(xn, idx.unsqueeze(1), gate.unsqueeze(1))
        return delta.float(), idx, aux

    @torch.no_grad()
    def all_outputs(self, xn):
        """Every specialist's output for every token (N, K, d) fp32 via the same dispatch kernel (one call per k)."""
        outs = []
        for k in range(K):
            idx = torch.full((xn.shape[0],), k, dtype=torch.long, device=xn.device)
            with autocast(xn.device):
                outs.append(self.bank(xn, idx.unsqueeze(1), torch.ones(xn.shape[0], 1, device=xn.device)).float())
        return torch.stack(outs, 1)


def make_conds(seed, device):
    return {"A": Cond("A", seed).to(device), "B": Cond("B", seed).to(device), "C": Cond("C", seed, router=True).to(device)}


def token_assign(bt, lab_sem, lab_shuf):
    """Per-token labels from the token's document (pad -> 0, masked by valid)."""
    gd = bt["gdoc"].reshape(-1)
    ok = gd >= 0
    g = gd.clamp(min=0)
    a = torch.where(ok, lab_sem[g], torch.zeros_like(g))
    b = torch.where(ok, lab_shuf[g], torch.zeros_like(g))
    return a, b
