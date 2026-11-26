"""Gate 8: Q and R normalized and finite in ordinary and adversarial cases; gradient reaches all router logits."""
import pytest
import torch

from csr.semantic_loss import _offdiag_log_softmax, relational_kl, select_valid_chunks


def _u(n, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.nn.functional.normalize(torch.randn(n, d, generator=g), dim=-1)


def _check(loss, stats):
    assert torch.isfinite(loss)
    assert loss >= -1e-6


@pytest.mark.parametrize("case", ["ordinary", "uniform", "collapsed", "identical_u", "extreme_temp"])
def test_finite_and_normalized(case):
    N, E, L = 12, 8, 3
    u = _u(N)
    logits = torch.randn(L, N, E, requires_grad=True)
    tau_s, tau_r = 0.1, 0.05
    if case == "uniform":
        logits = torch.zeros(L, N, E, requires_grad=True)
    if case == "collapsed":
        logits = torch.full((L, N, E), -50.0)
        logits[..., 0] = 50.0
        logits.requires_grad_(True)
    if case == "identical_u":
        u = u[:1].repeat(N, 1)
    if case == "extreme_temp":
        tau_s, tau_r = 1e-4, 1e-5
    loss, st = relational_kl(u, logits, tau_s, tau_r, min_chunks=4)
    _check(loss, st)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    # Q and R rows are normalized distributions over j != i
    S = u @ u.T
    logQ, eye = _offdiag_log_softmax(S / tau_s)
    Q = logQ.exp()
    assert torch.allclose(Q.sum(-1), torch.ones(N), atol=1e-5) and torch.all(Q[eye] == 0)
    if case == "identical_u":
        assert torch.allclose(Q[~eye], torch.full_like(Q[~eye], 1 / (N - 1)), atol=1e-6)
    if case == "uniform":
        assert st["r_entropy_eff"] == pytest.approx(N - 1, rel=1e-4)


def test_permutation_invariance():
    N, E, L = 10, 8, 2
    u, logits = _u(N, seed=1), torch.randn(L, N, E)
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(3))
    a, _ = relational_kl(u, logits, 0.1, 0.05, 4)
    b, _ = relational_kl(u[perm], logits[:, perm], 0.1, 0.05, 4)
    assert torch.allclose(a, b, atol=1e-6)


def test_skip_when_too_few_and_graph_valid():
    logits = torch.randn(2, 3, 8, requires_grad=True)
    loss, st = relational_kl(_u(3), logits, 0.1, 0.05, min_chunks=8)
    assert st["skipped"] and float(loss) == 0.0
    loss.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_gradient_reaches_every_router_logit():
    N, E, L = 16, 8, 1
    logits = torch.randn(L, N, E, requires_grad=True)
    loss, _ = relational_kl(_u(N, seed=5), logits, 0.1, 0.02, 4)
    loss.backward()
    assert (logits.grad.abs() > 0).all()  # full (pre-mask) probabilities -> every logit gets gradient


def test_targets_are_detached():
    u = _u(10).requires_grad_(True)
    loss, _ = relational_kl(u, torch.randn(1, 10, 8, requires_grad=True), 0.1, 0.05, 4)
    loss.backward()
    assert u.grad is None


def test_select_valid_chunks_dedup_invalid_and_cap():
    u = _u(10)
    u[3] = float("nan")
    u[4] = 0.0
    ph = torch.arange(10)
    ph[7] = ph[2]  # duplicate prefix
    idx = select_valid_chunks(u, ph.numpy(), n_max=100)
    assert idx.tolist() == [0, 1, 2, 5, 6, 8, 9]
    idx2 = select_valid_chunks(u, ph.numpy(), n_max=3)
    assert len(idx2) == 3 and idx2.tolist() == sorted(idx2.tolist())


def test_same_and_cross_document_stats():
    N = 12
    doc = torch.tensor([0] * 4 + [1] * 4 + [2] * 4)
    _, st = relational_kl(_u(N), torch.randn(2, N, 8), 0.1, 0.05, 4, doc_of=doc)
    assert st["same_doc_pairs"] == 3 * 4 * 3 and st["cross_doc_pairs"] == N * (N - 1) - 36
