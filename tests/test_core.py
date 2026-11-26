"""Fast CPU tests of the pieces every result depends on. Run: python -m pytest -q"""
import dataclasses

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from repro.clusters.build import fit_centroids, sinkhorn, within_split_permutation
from repro.config import EXPERIMENTS, FULL, SMOKE, ModelCfg, runnable
from repro.eval.metrics import matched_difference, quantile_bins, swap_specificity
from repro.model.losses import edl_layer_loss
from repro.model.moe import MoEGPT, token_ce
from repro.model.residual import Bank, CtxRouter, causal_window_mean
from repro.pipeline import build_graph

TINY = ModelCfg(vocab_size=64, ctx=32, n_layer=2, n_head=2, d=32, n_exp=8, top_k=2, expert_hidden=16,
                shared_hidden=32)


# ------------------------------------------------------------------ sizes match the doc
def test_doc_parameter_counts():
    assert sum(p.numel() for p in Bank(384, 16, 256).parameters()) == 3_145_728          # doc §23
    assert sum(p.numel() for p in CtxRouter(384, 16, 64).parameters()) == 75_152        # doc §27
    n = MoEGPT(FULL.model).param_count()
    assert abs(n - 103_609_728) / 103_609_728 < 0.02                                    # doc §2.2 (±2%)


# ------------------------------------------------------------------ causality
def test_moe_model_is_causal():
    torch.manual_seed(0)
    m = MoEGPT(TINY).eval()
    x = torch.randint(0, 64, (2, 32))
    x2 = x.clone()
    x2[:, 20:] = torch.randint(0, 64, (2, 12))
    with torch.no_grad():
        a, b = m(x), m(x2)
    assert torch.allclose(a[:, :20], b[:, :20], atol=1e-5)


@pytest.mark.parametrize("arch", ["gru", "mlp"])
@pytest.mark.parametrize("inputs", ["both", "token", "context"])
def test_router_is_causal(arch, inputs):
    torch.manual_seed(0)
    r = CtxRouter(16, 4, 8, inputs, 4, arch).eval()
    h = torch.randn(2, 30, 16)
    h2 = h.clone()
    h2[:, 17:] += torch.randn(2, 13, 16)
    with torch.no_grad():
        assert torch.allclose(r(h)[:, :17], r(h2)[:, :17], atol=1e-6)


def test_window_mean():
    h = torch.arange(10.0).view(1, 10, 1)
    m = causal_window_mean(h, 3).flatten()
    assert torch.allclose(m, torch.tensor([0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8.0]))


# ------------------------------------------------------------------ MoE mechanics
def test_ban_admits_next_ranked_expert():
    torch.manual_seed(0)
    m = MoEGPT(TINY).eval()
    x = torch.randint(0, 64, (2, 32))
    with torch.no_grad():
        m(x)
        before = m.moe_layers()[1].top_i.clone()
        m.set_ban({1: [int(before[0, 0])]})
        m(x)
        after = m.moe_layers()[1].top_i
        m.set_ban(None)
    assert not (after == before[0, 0]).any()
    assert (after.sort(1).values[:, 0] != after.sort(1).values[:, 1]).all()   # still k distinct experts


def test_bank_starts_at_base():
    b = Bank(16, 4, 8)
    h = torch.randn(2, 5, 16)
    assert torch.equal(b(h, torch.randint(0, 4, (2, 5))), h)


# ------------------------------------------------------------------ EDL
def test_edl_prefers_separated_groups():
    w = F.one_hot(torch.tensor([0] * 50 + [1] * 50), 2).float()
    same = torch.full((100, 4), 0.25)
    apart = torch.cat([torch.tensor([[0.7, 0.1, 0.1, 0.1]]).expand(50, 4),
                       torch.tensor([[0.1, 0.1, 0.1, 0.7]]).expand(50, 4)])
    assert edl_layer_loss(apart, w, 10) < edl_layer_loss(same, w, 10)
    p = same.clone().requires_grad_(True)
    edl_layer_loss(p, w, 10).backward()
    assert torch.isfinite(p.grad).all()                 # identical groups must not give NaN
    assert edl_layer_loss(same, w, 60) is None          # too little mass -> no loss


# ------------------------------------------------------------------ clustering
def test_sinkhorn_balances_skewed_scores():
    torch.manual_seed(0)
    s = torch.randn(4000, 8)
    s[:, 0] += 2.0                                       # cluster 0 would take almost everything
    q = sinkhorn(s, 0.1, 100)
    sizes = torch.bincount(q.argmax(1), minlength=8).float()
    assert (sizes / sizes.mean()).min() > 0.85 and (sizes / sizes.mean()).max() < 1.15


def test_kmeans_recovers_planted_clusters():
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(4, 32))
    X = np.concatenate([c + 0.3 * rng.normal(size=(300, 32)) for c in centers])
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    C = fit_centroids(X.astype(np.float32), 4, 20, 50)
    lab = (torch.from_numpy(X.astype(np.float32)) @ C.T).argmax(1).numpy()
    from repro.eval.metrics import nmi
    assert nmi(lab, np.repeat(np.arange(4), 300), 4, 4) > 0.95


def test_shuffled_tables_are_mass_matched_inside_every_split():
    """A shuffled control must have its semantic twin's cluster sizes in each split (bank: training,
    val: scoring), not only summed over all splits."""
    rng = np.random.default_rng(1)
    split = np.repeat(np.arange(5), [50, 50, 50, 400, 1500])
    hard = np.where(split == 3, rng.integers(0, 2, len(split)), rng.integers(0, 16, len(split)))  # skewed per split
    perm = within_split_permutation(split)
    assert sorted(perm) == list(range(len(split)))                    # a permutation
    assert (split[perm] == split).all()                                # never crosses splits
    for s in range(5):
        m = split == s
        assert (np.bincount(hard[perm][m], minlength=16) == np.bincount(hard[m], minlength=16)).all()
    assert (hard[perm] != hard).mean() > 0.4                           # and actually shuffles


# ------------------------------------------------------------------ specialisation metrics
def test_swap_specificity_detects_diagonal_only():
    rng = np.random.default_rng(0)
    N, k = 4000, 4
    grp = rng.integers(0, k, N)
    paper = grp * 1000 + rng.integers(0, 50, N)
    base = 3.0 + 0.1 * rng.normal(size=(N, 1)) + np.array([0.0, 0.05, 0.1, 0.02])   # specialist main effects
    flat = swap_specificity(base.repeat(1, 1) + np.zeros((N, k)), grp, paper, k, n_boot=50)
    spec = base + np.zeros((N, k))
    spec[np.arange(N), grp] -= 0.01                                            # own specialist helps
    diag = swap_specificity(spec, grp, paper, k, n_boot=50)
    assert abs(flat["specificity"]) < 1e-3 and diag["specificity"] > 0.005 and diag["ci"][0] > 0


def test_exposure_matching_removes_the_exposure_confound():
    """The question from the conversation: an expert used on 15/15 physics tokens and 2/15 robotics tokens,
    equally useful per use. The ordinary contrast says 'physics expert'; the exposure-matched one says 0."""
    rng = np.random.default_rng(0)
    N = 60000
    physics = rng.random(N) < 0.5
    routed = np.where(physics, rng.random(N) < 15 / 15, rng.random(N) < 2 / 15)
    gate = rng.random(N)
    delta = np.where(routed, 0.01 * gate + 0.001 * rng.normal(size=N), 0.0)   # same usefulness per use
    ordinary = delta[physics].mean() - delta[~physics].mean()
    matched, _ = matched_difference(delta[routed], physics[routed], quantile_bins(gate[routed], 5))
    assert ordinary > 0.004 and abs(matched) < 5e-4


# ------------------------------------------------------------------ router can learn its teacher
def test_router_fits_a_teacher():
    torch.manual_seed(0)
    h = torch.randn(4, 64, 16)
    target = (h[..., 0] > 0).long() + 2 * (h[..., 1] > 0).long()               # depends on the current token
    r = CtxRouter(16, 4, 16, "both", 4)
    opt = torch.optim.Adam(r.parameters(), 1e-2)
    for _ in range(300):
        loss = F.cross_entropy(r(h).reshape(-1, 4), target.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    assert (r(h).argmax(-1) == target).float().mean() > 0.9


# ------------------------------------------------------------------ registry / graph
def test_registry_graph_is_consistent():
    for prof in (FULL, SMOKE):
        jobs = build_graph(prof, "all")
        keys = {j.key for j in jobs}
        seen = set()
        for j in jobs:
            assert set(j.deps) <= seen, f"{j.key} depends on a later job"
            seen.add(j.key)
        assert keys == seen
    assert all(runnable(FULL, e) for e in EXPERIMENTS)


def test_token_ce_shape():
    logits = torch.randn(2, 3, 10)
    y = torch.randint(0, 10, (2, 3))
    assert token_ce(logits, y).shape == (2, 3)
