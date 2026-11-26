"""Gates 4, 12, 13: same valid tokens and active experts across conditions; gradient-accumulation accounting."""
import numpy as np
import torch

from conftest import tiny_cfg
from csr.config import CONDITIONS, build_condition_config
from csr.data import TokenSplit
from csr.model import CSRModel
from csr.segcache import build_table
from csr.segment import fixed_starts, random_starts, fit_length_distribution
from csr.train_lib import BatchSource, forward_losses, plan_steps, step_objective, valid_tokens


def make_split(n=20, seed=0):
    rng = np.random.default_rng(seed)
    docs = [np.concatenate([rng.integers(2, 97, size=int(rng.integers(20, 90))), [0]]) for _ in range(n)]
    off = np.concatenate([[0], np.cumsum([len(d) for d in docs])])
    return TokenSplit(np.concatenate(docs).astype(np.uint16), off, [f"d{i}" for i in range(n)])


def cfg_for(cond, lam=0.5):
    base = {"model": tiny_cfg().to_dict(), "routing": {"mode": "token", "pool_size": 4, "warmup": 4},
            "segmentation": {"strategy": "none"},
            "semantic": {"enabled": False, "lambda": lam, "tau_s": 0.1, "tau_r": 0.05, "n_max": 64, "min_chunks": 2},
            "train": {"aux_coeff": 0.01, "z_coeff": 0.001}, "init": {"type": "random", "seed": 0}}
    return build_condition_config(base, cond)


def tables(split):
    L = split.lengths()
    fixed = build_table("fixed", split, [fixed_starts(int(l), 16) for l in L])
    sem = build_table("semantic", split, [sorted(set([0] + list(range(0, int(l), 13))[1:])) for l in L])
    dist = fit_length_distribution(L, [sem.starts(i) for i in range(split.n_docs)], n_bins=1, min_len=1, max_len=256)
    rnd = build_table("random", split, [random_starts(int(l), k, dist, 3) for l, k in zip(L, split.doc_ids)])
    for t in (fixed, sem):
        n = len(t.starts_flat)
        t.u = np.random.default_rng(1).standard_normal((n, 8)).astype(np.float16)
        t.phash = np.arange(n)
    return {"none": None, "fixed": fixed, "semantic": sem, "random": rnd}


def test_same_valid_tokens_and_k_across_conditions():
    split = make_split()
    tabs = tables(split)
    torch.manual_seed(0)
    model = CSRModel(tiny_cfg()).double()
    targets, counts = {}, {}
    for c in CONDITIONS:
        cfg = cfg_for(c)
        src = BatchSource(split, 128, tabs[cfg["segmentation"]["strategy"]])
        tb = {k: torch.as_tensor(v) for k, v in src.batch(src.rows[:4]).items()}
        parts, out = forward_losses(model, tb, cfg, tabs[cfg["segmentation"]["strategy"]])
        targets[c] = tb["targets"]
        counts[c] = parts.n_valid
        nonpad = tb["doc_ids"] >= 0
        for r in out.routing:
            sel = r["topi"][nonpad]
            assert sel.shape[-1] == model.cfg.top_k
            assert all(len(set(x)) == model.cfg.top_k for x in sel.tolist())
        assert torch.isfinite(parts.sem)
        if cfg["routing"]["mode"] == "pool":
            assert parts.coverage_constrained > 0
    assert len(set(counts.values())) == 1
    assert all(torch.equal(targets["A"], t) for t in targets.values())
    steps, tot = plan_steps(BatchSource(split, 128), 2, 300)
    assert tot >= 300 and tot == sum(valid_tokens(s, split) for s in steps)


def test_grad_accumulation_matches_full_batch():
    split = make_split(seed=1)
    src = BatchSource(split, 128)
    rows = src.rows[:4]
    cfg = cfg_for("A")
    cfg["train"]["aux_coeff"] = 0.0
    cfg["train"]["z_coeff"] = 0.0

    def grads(groups):
        torch.manual_seed(0)
        m = CSRModel(tiny_cfg()).double()
        n_step = valid_tokens(rows, split)
        for g in groups:
            tb = {k: torch.as_tensor(v) for k, v in src.batch(g).items()}
            parts, _ = forward_losses(m, tb, cfg)
            step_objective(parts, n_step, cfg).backward()
        return [p.grad.clone() for p in m.parameters()]

    full = grads([rows])
    acc = grads([rows[:1], rows[1:3], rows[3:]])
    for a, b in zip(full, acc):
        assert torch.allclose(a, b, atol=1e-12)


def test_aux_terms_weighted_by_token_share():
    split = make_split(seed=2)
    src = BatchSource(split, 128)
    cfg = cfg_for("C")
    tabs = tables(split)
    src = BatchSource(split, 128, tabs["fixed"])
    torch.manual_seed(0)
    m = CSRModel(tiny_cfg()).double()
    tb = {k: torch.as_tensor(v) for k, v in src.batch(src.rows[:2]).items()}
    parts, _ = forward_losses(m, tb, cfg, tabs["fixed"])
    n_step = 2 * parts.n_valid
    obj = step_objective(parts, n_step, cfg)
    expect = parts.lm_sum / n_step + 0.5 * (0.01 * parts.aux + 0.001 * parts.z + 0.5 * parts.sem)
    assert torch.allclose(obj, expect)
